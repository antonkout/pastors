"""
Sample a Google Satellite Embedding GeoTIFF at sample-point locations and
append the 64 embedding bands as columns to an existing feature DataFrame.

Used after load_satellite_data() + spectral indices, to enlarge the feature
space with the embedding bands (prefixed 'Emb_').
"""

import numpy as np
import pandas as pd
import rasterio


def add_embeddings(df, points_gdf, embeddings_path, prefix='Emb_'):
    """
    Parameters
    ----------
    df : pd.DataFrame
        Feature frame indexed by Serial (output of load_satellite_data).
    points_gdf : geopandas.GeoDataFrame
        Sample points with geometry, indexed by (or containing) the same Serials
        as df. Must have a .crs.
    embeddings_path : str
        64-band embedding GeoTIFF (e.g. data/satellite/embeddings_oman.tif).
    prefix : str
        Column-name prefix; bands already named 'Emb_Axx' are kept as-is.

    Returns
    -------
    pd.DataFrame
        df with 64 embedding columns joined on the index.
    """
    g = points_gdf.loc[df.index]  # align to df's rows/order

    with rasterio.open(embeddings_path) as src:
        gg = g.to_crs(src.crs) if g.crs != src.crs else g
        coords = [(geom.x, geom.y) for geom in gg.geometry]
        vals = np.array(list(src.sample(coords)), dtype=np.float32)  # (n, 64)
        names = [src.descriptions[i] or f'A{i:02d}' for i in range(src.count)]

    names = [n if n.startswith(prefix) else prefix + n for n in names]
    emb = pd.DataFrame(vals, columns=names, index=df.index)

    # Flag points that fell on the raster nodata / outside coverage
    nodata = src.nodata if src.nodata is not None else None
    if nodata is not None:
        n_bad = int((vals == nodata).all(axis=1).sum())
        if n_bad:
            print(f"  [WARN] {n_bad} points outside embedding coverage (all-nodata)")

    print(f"  added {emb.shape[1]} embedding columns ({names[0]}..{names[-1]})")
    return pd.concat([df, emb], axis=1)
