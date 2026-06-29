"""
On-the-fly feature operations for the PASTORS pipeline, toggled at runtime
(no regeneration of the fused GeoTIFF):

  - sharpen_band         : ratio (Brovey) pan-sharpen one band with a high-res
                           reference (SPOT NIR). This is what actually brings
                           SWIR to 1.5 m detail.
  - sharpen_swir_bands   : pan-sharpen the SWIR bands of a (bands,H,W) stack
                           using the SPOT_nir band already present in it.
  - embeddings_on_grid   : reproject the native 10 m embeddings onto a target
                           grid; interpolate=True -> bilinear (smooth 1.5 m),
                           False -> nearest (hard 10 m). NOT pan-sharpened.

Both load_satellite_data (training) and predict_raster (inference) call these
with the SAME flags so the feature vectors match.
"""

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import uniform_filter

SWIR_BANDS = ('S2_B11_swir16', 'S2_B12_swir22')
SPOT_NIR = 'SPOT_nir'
# fused S2 was delivered at 16 m -> 16/1.5 ~= 11 (matches the original fusion)
SWIR_FILTER_SIZE = 11


def sharpen_band(lowres_band, highres_reference, filter_size=SWIR_FILTER_SIZE):
    """Ratio (Brovey) pan-sharpen: F = L * (P / smooth(P)), mean preserved."""
    highres = highres_reference.astype(np.float32)
    smoothed = uniform_filter(highres, size=filter_size)
    smoothed = np.where(smoothed == 0, 1e-8, smoothed)
    sharp = lowres_band * (highres / smoothed)
    sharp = sharp * (np.nanmean(lowres_band) / (np.nanmean(sharp) + 1e-8))
    return sharp.astype(np.float32)


def sharpen_swir_bands(data, band_names, filter_size=SWIR_FILTER_SIZE):
    """
    Return a copy of `data` (bands,H,W) with the SWIR bands pan-sharpened using
    the SPOT_nir band found in `band_names`. No-op if SPOT_nir is missing.
    """
    names = list(band_names)
    if SPOT_NIR not in names:
        print("  [sharpen_swir] SPOT_nir band not found; skipped")
        return data
    nir = data[names.index(SPOT_NIR)]
    out = data.copy()
    done = []
    for sb in SWIR_BANDS:
        if sb in names:
            i = names.index(sb)
            out[i] = sharpen_band(data[i], nir, filter_size)
            done.append(sb)
    print(f"  SWIR pan-sharpened to 1.5 m: {done}")
    return out


def embeddings_on_grid(embeddings_path, transform, crs, H, W, interpolate=True):
    """
    Reproject the native 10 m embeddings onto a target grid.
    interpolate=True -> bilinear (smooth), False -> nearest (hard 10 m).
    Returns (data (n,H,W), names list prefixed 'Emb_').
    """
    rs = Resampling.bilinear if interpolate else Resampling.nearest
    with rasterio.open(embeddings_path) as emb:
        src = emb.read()
        nod = emb.nodata
        names = [emb.descriptions[i] or f'A{i:02d}' for i in range(emb.count)]
        dst = np.empty((emb.count, H, W), dtype=np.float32)
        reproject(
            source=src, destination=dst,
            src_transform=emb.transform, src_crs=emb.crs,
            dst_transform=transform, dst_crs=crs,
            src_nodata=nod, dst_nodata=nod, resampling=rs,
        )
    names = [n if n.startswith('Emb_') else f'Emb_{n}' for n in names]
    mode = 'bilinear 1.5 m' if interpolate else 'nearest 10 m'
    print(f"  embeddings reprojected onto grid ({mode}): {dst.shape[0]} bands")
    return dst, names
