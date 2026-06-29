"""Apply a binary mask to a multi-band raster.

Pixels where mask != keep_value are set to nodata. Mask is resampled
to the target raster's grid (CRS, transform, shape) if needed.
"""
import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling


def mask_raster(raster_path, mask_path, output_path,
                          keep_value=1, nodata=0):
    """
    Parameters
    ----------
    raster_path : str
        Input multi-band raster.
    mask_path : str
        Binary mask raster. Pixels equal to `keep_value` are retained.
    output_path : str
        Where to write the masked raster.
    keep_value : int
        Mask value identifying valid pixels.
    nodata : int or float
        Value written for masked-out pixels. Set as raster nodata in the
        output profile so downstream tools (e.g. predict_raster) skip them.

    Returns
    -------
    str
        output_path
    """
    with rasterio.open(raster_path) as src, rasterio.open(mask_path) as msk:
        data = src.read()
        profile = src.profile.copy()

        same_grid = (msk.crs == src.crs
                     and msk.transform == src.transform
                     and msk.shape == src.shape)
        if same_grid:
            mask_arr = msk.read(1)
        else:
            mask_arr = np.zeros(src.shape, dtype=msk.dtypes[0])
            reproject(
                source=rasterio.band(msk, 1),
                destination=mask_arr,
                src_transform=msk.transform, src_crs=msk.crs,
                dst_transform=src.transform, dst_crs=src.crs,
                resampling=Resampling.nearest,
            )

        keep = (mask_arr == keep_value)
        print(f"  kept pixels: {keep.sum():,} / {keep.size:,} "
              f"({100 * keep.mean():.1f}%)")

        data[:, ~keep] = nodata
        profile.update(nodata=nodata)

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data)

    print(f"✅ Masked raster saved: {output_path}")
    return output_path