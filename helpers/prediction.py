"""
Inference: apply a trained model to a satellite raster
and write a single-band prediction GeoTIFF.
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling

from helpers.load_satellite_data import _add_spectral_indices, _read_band_names


def predict_raster(
    model,
    satellite,
    feature_cols,
    prediction_image_path=None,
    output_path=None,
    base_path='./data/satellite',
    mask_path=None,
    mask_keep_value=1,
    sharpen_swir=False,
    use_embeddings=False,
    embeddings_path=None,
    emb_interpolate=True,
):
    """
    Run pixel-wise inference on a satellite raster.

    Predicts on every pixel with finite features. If `mask_path` is
    given, predictions outside `mask_keep_value` are set to NaN after
    prediction.

    Parameters
    ----------
    model : fitted regressor
        Must expose .predict and .n_features_in_.
    satellite : str
        'fused_satellite' or 'maxar'.
    feature_cols : list[str]
        Exact training feature columns, in training order.
    prediction_image_path : str, optional
        Defaults to {base_path}/{satellite}.tif.
    output_path : str, optional
        Defaults to ./output/{satellite}_balance_prediction.tif.
    base_path : str
        Directory containing the raster + sidecar JSON.
    mask_path : str, optional
        Path to a binary mask raster. Pixels not equal to
        `mask_keep_value` are set to NaN in the output.
    mask_keep_value : int
        Mask value identifying valid pixels.

    Returns
    -------
    output_path : str
        Path of the written prediction raster.
    """
    if prediction_image_path is None:
        prediction_image_path = os.path.join(base_path, f"{satellite}.tif")
    if output_path is None:
        os.makedirs('./output', exist_ok=True)
        output_path = f"./output/{satellite}_balance_prediction.tif"

    print(f"Model expects {model.n_features_in_} features")

    with rasterio.open(prediction_image_path) as src:
        profile = src.profile.copy()
        img = src.read().astype(np.float32)
        band_names = _read_band_names(satellite, base_path, src.count)
        out_transform = src.transform
        out_crs = src.crs

    B, H, W = img.shape
    print(f"Raster: {B} bands, {H}x{W}")

    # Same on-the-fly ops as training (must match toggles used there).
    if sharpen_swir:
        from helpers.feature_ops import sharpen_swir_bands
        img = sharpen_swir_bands(img, band_names)

    df_px = pd.DataFrame(img.reshape(B, -1).T, columns=band_names)
    df_px = _add_spectral_indices(df_px, satellite)

    if use_embeddings:
        if not embeddings_path or not os.path.exists(embeddings_path):
            raise FileNotFoundError(
                f"use_embeddings=True but embeddings_path missing: {embeddings_path}")
        from helpers.feature_ops import embeddings_on_grid
        emb_grid, emb_names = embeddings_on_grid(
            embeddings_path, out_transform, out_crs, H, W,
            interpolate=emb_interpolate)
        emb_df = pd.DataFrame(emb_grid.reshape(emb_grid.shape[0], -1).T,
                              columns=emb_names, index=df_px.index)
        df_px = pd.concat([df_px, emb_df], axis=1)

    X = df_px[feature_cols].values
    print(f"Feature matrix: {X.shape}, expected {model.n_features_in_}")

    valid = np.isfinite(X).all(axis=1)
    pred = np.full(X.shape[0], np.nan, dtype=np.float32)
    pred[valid] = model.predict(X[valid])
    print(f"  predicted: {valid.sum():,} pixels "
          f"({100 * valid.mean():.1f}%)")

    pred_2d = pred.reshape(H, W)

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
        pred_2d[~keep] = np.nan
        print(f"  mask applied: kept {keep.sum():,} / {keep.size:,} "
              f"({100 * keep.mean():.1f}%)")

    profile.update(count=1, dtype="float32", compress="lzw", nodata=np.nan)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(pred_2d, 1)

    print(f"Saved → {output_path}")
    return output_path