"""
Export spectral indices from a satellite raster as single-band GeoTIFFs.

Indices come from the same `_add_spectral_indices` used at training and
inference, so an exported map is exactly the feature the model saw.
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling

from helpers.load_satellite_data import _add_spectral_indices, _read_band_names


# Short name used in the filename -> column produced by _add_spectral_indices.
INDEX_ALIASES = {
    'NDVI':    'NDVI',
    'BSI':     'BSI',
    'NDWI':    'NDWI',
    'Clay':    'Clay_Index',
    'Calcite': 'Calcite_Index',
}


def export_indices(
    satellite,
    dataset,
    indices=('NDVI', 'BSI', 'NDWI', 'Clay', 'Calcite'),
    out_dir=None,
    base_path='./data/satellite',
    image_path=None,
    mask_path=None,
    mask_keep_value=1,
    sharpen_swir=False,
    window=1,
):
    """
    Write one GeoTIFF per index as {out_dir}/{dataset}_{index}.tif.

    `sharpen_swir` and `window` mirror the training toggles: pass the same
    values used in the notebook so the exported maps match the model features.
    If `mask_path` is given, pixels not equal to `mask_keep_value` are NaN.

    Returns
    -------
    dict[str, str]
        Short index name -> written path.
    """
    if image_path is None:
        image_path = os.path.join(base_path, f"{satellite}.tif")
    if out_dir is None:
        out_dir = f"./output/{dataset}"
    os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(image_path) as src:
        profile = src.profile.copy()
        img = src.read().astype(np.float32)
        band_names = _read_band_names(satellite, base_path, src.count)
        out_transform, out_crs = src.transform, src.crs

    B, H, W = img.shape
    print(f"Raster: {B} bands, {H}x{W}")

    if sharpen_swir:
        from helpers.feature_ops import sharpen_swir_bands
        img = sharpen_swir_bands(img, band_names)

    if window and window > 1:
        from scipy.ndimage import median_filter
        img = median_filter(img, size=(1, window, window), mode='nearest')
        print(f"  window: {window}x{window} median filter on bands")

    df_px = pd.DataFrame(img.reshape(B, -1).T, columns=band_names)
    df_px = _add_spectral_indices(df_px, satellite)

    keep = None
    if mask_path is not None:
        with rasterio.open(mask_path) as msk:
            same_grid = (msk.crs == out_crs
                         and msk.transform == out_transform
                         and msk.shape == (H, W))
            if same_grid:
                mask_arr = msk.read(1)
            else:
                mask_arr = np.zeros((H, W), dtype=msk.dtypes[0])
                reproject(
                    source=rasterio.band(msk, 1),
                    destination=mask_arr,
                    src_transform=msk.transform, src_crs=msk.crs,
                    dst_transform=out_transform, dst_crs=out_crs,
                    resampling=Resampling.nearest,
                )
        keep = (mask_arr == mask_keep_value)
        print(f"  mask: keeping {keep.sum():,} / {keep.size:,} "
              f"({100 * keep.mean():.1f}%)")

    profile.update(count=1, dtype="float32", compress="lzw", nodata=np.nan)

    written = {}
    for short in indices:
        col = INDEX_ALIASES.get(short, short)
        if col not in df_px.columns:
            print(f"  [WARN] {short} skipped; {col} not computed for {satellite}")
            continue
        arr = df_px[col].values.astype(np.float32).reshape(H, W)
        if keep is not None:
            arr = np.where(keep, arr, np.nan)
        path = os.path.join(out_dir, f"{dataset}_{short}.tif")
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(arr, 1)
            dst.set_band_description(1, col)
        finite = np.isfinite(arr)
        rng = (f"{np.nanmin(arr):.3f} .. {np.nanmax(arr):.3f}"
               if finite.any() else "all-NaN")
        print(f"  {short:8s} -> {path}  [{rng}]")
        written[short] = path

    return written
