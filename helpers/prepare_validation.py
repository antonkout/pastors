"""
Align a validation GeoDataFrame with the training set:
restrict to the subcomposition of shared elements, reclose to sum=1,
attach training-set background samples to the validation frame,
and prepare it for background_removal.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import composition_stats as coda


# Non-element numeric columns to exclude when detecting elements.
_META_NUM = {'Longitude', 'Latitude', 'Easting', 'Northing',
             'is_imputed', 'is_outlier', 'Res'}


def prepare_validation(gdf, gdf_val,
                       background_labels=('BACKGROUND', 'Background', 'B'),
                       restrict_training=True):
    """
    Build a comparable subcomposition between training and validation frames.

    Logic
    -----
    1. Element columns = numeric columns common to both frames, minus metadata
       (coordinates, flags) and 'Res' (Res is derived from a sum that no longer
        holds once elements are dropped, so we discard it).
    2. Drop all numeric columns that are not in the shared element set.
    3. Reclose the subcomposition of each frame so rows sum to 1.
       Subcompositional invariance: ratios between shared elements are
       preserved by closure, so the two frames now live on the same scale.
    4. Make sure gdf_val has Serial + Easting + Northing columns.
    5. Concat gdf_val with the training-set BACKGROUND rows (validation has
       none of its own) so background_removal() can correct it.

    Parameters
    ----------
    gdf : GeoDataFrame
        Training data including BACKGROUND rows.
    gdf_val : GeoDataFrame
        Validation points.
    background_labels : iterable[str]
        Values of the 'Type' column flagging background rows.

    Returns
    -------
    gdf : GeoDataFrame
        Training frame, restricted + reclosed on shared elements.
    gdf_val_bg : GeoDataFrame
        Validation + training-background rows, restricted + reclosed,
        ready for background_removal().
    shared_elems : list[str]
        The element columns kept.
    """
    gdf, gdf_val = gdf.copy(), gdf_val.copy()

    # 1. Shared element columns
    num_g = set(gdf.select_dtypes(np.number).columns)
    num_v = set(gdf_val.select_dtypes(np.number).columns)
    shared_elems = sorted((num_g & num_v) - _META_NUM)
    if not shared_elems:
        raise ValueError("No shared element columns between gdf and gdf_val.")

    # 2. Drop non-shared numeric columns from validation (always).
    #    Training is reduced only if restrict_training=True; otherwise it keeps
    #    all its elements (model trains on the full composition) and only the
    #    validation side is restricted to the shared subcomposition.
    drop_v = (num_v - set(shared_elems)) - (_META_NUM - {'Res'})
    gdf_val = gdf_val.drop(columns=list(drop_v), errors='ignore')

    # 3. Reclose subcomposition: each row of shared_elems sums to 1
    def _reclose(frame):
        vals = frame[shared_elems].values.astype(float)
        if (vals <= 0).any():
            vals = coda.multiplicative_replacement(np.where(vals <= 0, 0, vals))
        frame[shared_elems] = coda.closure(vals)
        return frame

    gdf_val = _reclose(gdf_val)
    if restrict_training:
        drop_g = (num_g - set(shared_elems)) - (_META_NUM - {'Res'})
        gdf = gdf.drop(columns=list(drop_g), errors='ignore')
        gdf = _reclose(gdf)

    # 4. Serial + coordinates for gdf_val
    if 'Serial' not in gdf_val.columns:
        gdf_val = gdf_val.reset_index().rename(columns={'index': 'Serial'})
    gdf_val['Easting']  = gdf_val.geometry.x
    gdf_val['Northing'] = gdf_val.geometry.y

    # 5. Concat val + training background (background rows recomputed on the
    #    shared subcomposition so they match the validation scale).
    bg = _reclose(gdf[gdf['Type'].isin(list(background_labels))].copy())
    keep = (['Serial', 'Site', 'Type', 'Easting', 'Northing', 'geometry']
            + shared_elems)
    gdf_val_bg = gpd.GeoDataFrame(
        pd.concat([gdf_val[keep], bg[keep]], ignore_index=True),
        geometry='geometry', crs=gdf_val.crs,
    )

    return gdf, gdf_val_bg, shared_elems