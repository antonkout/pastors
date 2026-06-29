"""
Fuse SPOT 6 (1.5 m, R/G/B/NIR) with Sentinel-2 (10 bands) per region.

SPOT 6 is the high-resolution reference grid. All 10 Sentinel-2 bands are
reprojected/resampled onto the SPOT grid. The bands that are radiometrically
correlated with SPOT NIR (B2-B8A) are pan-sharpened using SPOT NIR as the
high-resolution detail source (ratio / Brovey-type). The SWIR bands (B11, B12)
are kept spectrally clean (bilinear resample only), because SPOT carries no
SWIR information and injecting NIR texture into SWIR would fabricate detail.

No band rescaling and no nodata masking are applied: SPOT bands keep their raw
DN values, Sentinel-2 bands keep their original reflectance values.

Regions are matched by name so each SPOT image fuses with its own S2 image
(oman SPOT <-> oman S2, kenya SPOT <-> kenya S2).

Note: for Oman, SPOT is in EPSG:32640 and S2 in EPSG:32639. reproject() below
handles the CRS change automatically by warping S2 onto the SPOT CRS/transform.
"""

import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
from scipy.ndimage import uniform_filter


# SPOT 6 band order in the file: Red(B2), Green(B1), Blue(B0), NIR(B3)
SPOT_BANDS = ["SPOT_red", "SPOT_green", "SPOT_blue", "SPOT_nir"]
SPOT_NIR_INDEX = 3  # 0-based array index of the SPOT NIR band (used as sharpen ref)

# Sentinel-2 band order in the file (60 m bands already removed).
# SWIR (B11/B12) pan-sharpening is controlled by the `pansharpen_swir` flag of
# fuse_spot_sentinel2 (the entries below are the non-SWIR defaults).
S2_BANDS = [
    {"name": "S2_B2_blue",     "pansharpen": True},
    {"name": "S2_B3_green",    "pansharpen": True},
    {"name": "S2_B4_red",      "pansharpen": True},
    {"name": "S2_B5_rededge1", "pansharpen": True},
    {"name": "S2_B6_rededge2", "pansharpen": True},
    {"name": "S2_B7_rededge3", "pansharpen": True},
    {"name": "S2_B8_nir",      "pansharpen": True},
    {"name": "S2_B8A_nir08",   "pansharpen": True},
    {"name": "S2_B11_swir16",  "pansharpen": False},  # overridden by pansharpen_swir
    {"name": "S2_B12_swir22",  "pansharpen": False},  # overridden by pansharpen_swir
]
SWIR_NAMES = {"S2_B11_swir16", "S2_B12_swir22"}


def pansharpen_band(lowres_band, highres_reference, filter_size):
    """
    Ratio (Brovey-type) pan-sharpening: F = L * (P / smooth(P)).

    Preserves the low-res spectral mean while injecting the high-frequency
    spatial detail of the reference band.
    """
    highres = highres_reference.astype(np.float32)
    highres_smoothed = uniform_filter(highres, size=filter_size)
    highres_smoothed = np.where(highres_smoothed == 0, 1e-8, highres_smoothed)

    detail_ratio = highres / highres_smoothed
    sharpened = lowres_band * detail_ratio

    # Preserve the original mean to avoid a brightness shift
    sharpened = sharpened * (np.nanmean(lowres_band) / (np.nanmean(sharpened) + 1e-8))
    return sharpened.astype(np.float32)


def fuse_spot_sentinel2(spot_path, sentinel2_path, output_path,
                        pansharpen_swir=False):
    """
    Fuse one SPOT 6 image with one Sentinel-2 image.

    Output: 4 SPOT bands (raw DN) + 10 resampled Sentinel-2 bands (reflectance)
    = 14 bands, float32. No rescaling, no nodata masking.

    pansharpen_swir : bool
        False (default) -> SWIR B11/B12 kept clean (bilinear 20 m, physically
        correct). True -> SWIR pan-sharpened with SPOT NIR to 1.5 m detail
        (sharper but the detail is NIR-derived / fabricated).
    """
    # Per-band config for this call: SWIR follows the flag.
    band_cfg = [
        {**b, "pansharpen": (pansharpen_swir if b["name"] in SWIR_NAMES
                             else b["pansharpen"])}
        for b in S2_BANDS
    ]
    print(f"Reading SPOT reference: {spot_path}")
    with rasterio.open(spot_path) as spot:
        spot_data = spot.read().astype(np.float32)
        target_crs = spot.crs
        target_transform = spot.transform
        target_height, target_width = spot.height, spot.width
        target_res = abs(target_transform[0])
        print(f"  SPOT: {spot_data.shape}, {target_res:.2f} m, {target_crs}")

    spot_nir = spot_data[SPOT_NIR_INDEX]

    print(f"Reading Sentinel-2: {sentinel2_path}")
    s2_processed = []
    with rasterio.open(sentinel2_path) as s2:
        s2_res = abs(s2.transform[0])
        filter_size = max(int(np.ceil(s2_res / target_res)), 2)
        print(f"  S2: {s2.count} bands, {s2_res:.1f} m, {s2.crs} "
              f"(pan-sharpen filter_size={filter_size})")

        for band_idx, cfg in enumerate(band_cfg, start=1):
            resampled = np.empty((target_height, target_width), dtype=np.float32)
            reproject(
                source=rasterio.band(s2, band_idx),
                destination=resampled,
                src_transform=s2.transform,
                src_crs=s2.crs,
                dst_transform=target_transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear,
            )
            if cfg["pansharpen"]:
                resampled = pansharpen_band(resampled, spot_nir, filter_size)
            mode = "pansharpen" if cfg["pansharpen"] else "bilinear "
            s2_processed.append(resampled)
            print(f"    {cfg['name']:16s} [{mode}]")

    fused = np.vstack([spot_data, np.stack(s2_processed, axis=0)])
    descriptions = SPOT_BANDS + [b["name"] for b in band_cfg]
    print(f"  Fused shape: {fused.shape} ({len(descriptions)} bands)")

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": target_width,
        "height": target_height,
        "count": fused.shape[0],
        "crs": target_crs,
        "transform": target_transform,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    print(f"  Writing {output_path}")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(fused)
        for i, desc in enumerate(descriptions, start=1):
            dst.set_band_description(i, desc)
    print("  Done.\n")
    return output_path


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":

    # ── flip this to compare clean (20 m) vs pan-sharpened (1.5 m) SWIR ──
    PANSHARPEN_SWIR = False   # True -> SWIR sharpened to 1.5 m
    SUFFIX = "_swirsharp" if PANSHARPEN_SWIR else ""

    base = "/Users/antonkout/Documents/Doctorado/code/pastors/data/satellite/transfer"

    regions = {
        "oman": (f"{base}/spot6_oman.TIF", f"{base}/sentinel2_oman.tif"),
        "kenya": (f"{base}/spot6_kenya.TIF", f"{base}/sentinel2_kenya.tif"),
    }

    for region, (spot_path, s2_path) in regions.items():
        print(f"===== {region.upper()}  (pansharpen_swir={PANSHARPEN_SWIR}) =====")
        fuse_spot_sentinel2(
            spot_path=spot_path,
            sentinel2_path=s2_path,
            output_path=f"{base}/fused_spot_s2_{region}{SUFFIX}.tif",
            pansharpen_swir=PANSHARPEN_SWIR,
        )
