import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import composition_stats as coda


def aggregate_validation_triples(gdf_val):
    """
    Collapse 1.1/1.2/1.3 triples into one row per parent point.

    - Element columns: compositional geometric mean (coda.center).
    - Geometry: centroid of the three points.
    - Coords (Longitude, Latitude, Easting, Northing): mean.
    - Serial: parent integer (1.1 -> 1, 2.3 -> 2).
    - Site, Type kept (assumed identical within group).
    - Boolean flags (is_imputed, is_outlier): True if any sub-sample is True.

    Parameters
    ----------
    gdf_val : GeoDataFrame
        Validation samples with decimal Serial (e.g. '1.1').

    Returns
    -------
    GeoDataFrame, indexed by parent integer Serial.
    """
    df = gdf_val.copy()
    if 'Serial' not in df.columns:
        df = df.reset_index()

    # Parent integer from Serial
    df['_parent'] = df['Serial'].astype(str).str.split('.').str[0].astype(int)

    # Element columns = numeric, exclude meta
    meta = {'Serial', '_parent', 'Site', 'Type',
            'Longitude', 'Latitude', 'Easting', 'Northing',
            'Elevation', 'is_imputed', 'is_outlier', 'geometry'}
    elem_cols = [c for c in df.columns
                 if c not in meta and pd.api.types.is_numeric_dtype(df[c])]

    rows = []
    for parent, grp in df.groupby('_parent', sort=True):
        X = grp[elem_cols].astype(float).to_numpy()
        chem = coda.center(X) if len(grp) > 1 else X[0]

        rec = {'Serial': parent}
        if 'Site' in grp.columns:
            rec['Site'] = grp['Site'].iloc[0]
        if 'Type' in grp.columns:
            rec['Type'] = grp['Type'].iloc[0]
        for col, val in zip(elem_cols, chem):
            rec[col] = val
        rec['geometry'] = Point(grp.geometry.x.mean(), grp.geometry.y.mean())
        rows.append(rec)

    out = gpd.GeoDataFrame(rows, geometry='geometry', crs=gdf_val.crs)
    out = out.set_index('Serial').sort_index()
    return out