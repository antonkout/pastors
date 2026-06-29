"""
Post-process a classified mask with rasterio.features.sieve: remove connected
blobs smaller than `min_size` pixels, reassigning them to the largest
neighbouring class. Cleans salt-and-pepper speckle while preserving boundaries.

Sieving is restricted to valid (non-nodata) pixels so the nodata background is
never merged into the classes and class blobs never leak into the background.
"""

import rasterio
from rasterio.features import sieve as rio_sieve


def sieve_mask(mask_path, output_path=None, min_size=50, connectivity=4):
    """
    Parameters
    ----------
    mask_path : str
        Input single-band integer mask (e.g. 1=bare, 2=non-bare, 0=nodata).
    output_path : str or None
        Destination. If None, overwrites mask_path in place.
    min_size : int
        Minimum blob size in pixels; smaller blobs are removed.
    connectivity : int
        4 or 8 (pixel adjacency for connected components).

    Returns
    -------
    output_path : str
    """
    with rasterio.open(mask_path) as src:
        arr = src.read(1)
        profile = src.profile.copy()
        nodata = src.nodata

    valid = arr != (0 if nodata is None else nodata)
    cleaned = rio_sieve(arr, size=min_size, mask=valid, connectivity=connectivity)
    cleaned[~valid] = arr[~valid]   # keep background exactly as-is

    changed = int((cleaned != arr).sum())
    out = output_path or mask_path
    with rasterio.open(out, 'w', **profile) as dst:
        dst.write(cleaned, 1)

    print(f"  sieve(min_size={min_size}, conn={connectivity}): "
          f"reassigned {changed:,} pixels -> {out}")
    return out
