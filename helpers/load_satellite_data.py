import os
import json
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from pyproj import CRS
import composition_stats as coda


# ── Band mappings for spectral indices ────────────────────────────────────
SPECTRAL_BANDS = {
    'fused_satellite': {
    'red':        'Drag_660nm',
    'nir':        'S2_B8_842nm',
    'nir_narrow': 'S2_B8A_865nm',
    'swir1':      'S2_B11_1614nm',
    'swir2':      'S2_B12_2202nm',
    'green':      'Drag_549nm',
    'blue':       'Drag_503nm',
    },
    'maxar': {
        # Maxar WorldView-3 Multi-8 in standard band order:
        # C, B, G, Y, R, RE, N1, N2.  Names match maxar_metadata.json.
        'coastal':    'Maxar_Coastal_425nm',
        'blue':       'Maxar_Blue_480nm',
        'green':      'Maxar_Green_545nm',
        'yellow':     'Maxar_Yellow_605nm',
        'red':        'Maxar_Red_660nm',
        'rededge':    'Maxar_RedEdge_725nm',
        'nir':        'Maxar_NIR1_832nm',
        'nir_narrow': 'Maxar_NIR2_950nm',
    },
    # Matched by prefix, so 'fused_spot_s2_oman', 'fused_spot_s2_kenya', etc.
    # all resolve here. Indices use Sentinel-2 (reflectance) bands ONLY, never
    # the SPOT bands (raw DN), to avoid mixing scales inside an index.
    'fused_spot_s2': {
        'red':        'S2_B4_red',
        'nir':        'S2_B8_nir',
        'nir_narrow': 'S2_B8A_nir08',
        'swir1':      'S2_B11_swir16',
        'swir2':      'S2_B12_swir22',
        'green':      'S2_B3_green',
        'blue':       'S2_B2_blue',
    },
}


def _extract_window(cube, rows, cols, window):
    """
    Sample a (bands, H, W) cube at integer (rows, cols).

    window <= 1 -> the single pixel (original behaviour).
    window > 1  -> per-band median over a window x window neighbourhood,
                   clipped at the raster edges. The median (not mean) resists
                   a stray shadow / rock / vegetation pixel in the patch.

    Only the satellite FEATURE side is aggregated here; the ground-sample
    chemistry (a composition) is never touched by the window.
    """
    if window is None or window <= 1:
        return cube[:, rows, cols]
    B, H, W = cube.shape
    half = window // 2
    out = np.empty((B, len(rows)), dtype=np.float32)
    for k, (r, c) in enumerate(zip(rows, cols)):
        r0, r1 = max(0, r - half), min(H, r + half + 1)
        c0, c1 = max(0, c - half), min(W, c + half + 1)
        patch = cube[:, r0:r1, c0:c1].reshape(B, -1)
        out[:, k] = np.nanmedian(patch, axis=1)
    return out


def load_satellite_data(satellite_type, df_closed,
                        sharpen_swir=False, use_embeddings=False,
                        embeddings_path=None, emb_interpolate=True,
                        window=1):
    """
    Read raster, extract per-sample pixel values, aggregate samples sharing
    a pixel via compositional geometric mean, and append spectral indices.

    `window` controls the satellite-feature footprint: 1 = the single pixel
    containing each sample; 3 = median over a 3x3 neighbourhood (absorbs
    GPS / co-registration error at 1.5 m). Indices are computed AFTER the
    window median, so they inherit the smoothed bands (no ratio bias).

    Coordinates are taken from df_closed['geometry'] (.x and .y).
    The raster CRS must match the df_closed/geometry CRS - mismatch raises
    ValueError.

    Parameters
    ----------
    satellite_type : str
        'fused_satellite' or 'maxar'.
    df_closed : pd.DataFrame or gpd.GeoDataFrame
        Closed compositional data, indexed by Serial. Must have a 'geometry'
        column (typically a GeoDataFrame). For GeoDataFrames, .crs must be
        set; for plain DataFrames with shapely points, an EPSG matching the
        raster is assumed.

    Returns
    -------
    df_out : pd.DataFrame
        One row per pixel (Serial-indexed). Columns: element compositions +
        satellite bands + spectral indices.
    """
    # Known aliases; any other name falls back to "<satellite_type>.tif".
    file_mapping = {
        'fused_satellite': 'fused_satellite.tif',
        'maxar':           'maxar.tif',
    }
    base_path  = './data/satellite'
    image_path = os.path.join(
        base_path, file_mapping.get(satellite_type, f"{satellite_type}.tif"))
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Could not find image at: {image_path}")
    print(f"Loading raster: {image_path}")

    # Coordinates from geometry only
    if 'geometry' not in df_closed.columns:
        raise ValueError("df_closed must contain a 'geometry' column.")
    xs = df_closed.geometry.x.values
    ys = df_closed.geometry.y.values

    # Element columns: numeric, exclude coords/metadata
    exclude = {'Easting', 'Northing', 'Longitude', 'Latitude',
               'Elevation', 'is_imputed', 'is_outlier', 'geometry',
               'Site', 'Type', 'Serial'}
    element_cols = [c for c in df_closed.columns
                    if c not in exclude
                    and pd.api.types.is_numeric_dtype(df_closed[c])]

    with rasterio.open(image_path) as src:
        # CRS hard check against df_closed.crs if available
        gdf_crs = getattr(df_closed, 'crs', None)
        if gdf_crs is None:
            raise ValueError(
                "df_closed has no .crs attribute. Pass a GeoDataFrame with "
                "a defined CRS so it can be checked against the raster.")
        if CRS.from_user_input(src.crs) != CRS.from_user_input(gdf_crs):
            raise ValueError(
                f"CRS mismatch: raster is {src.crs}, df_closed is {gdf_crs}. "
                "Reproject one of them and retry.")

        rows, cols = rowcol(src.transform, xs, ys)
        rows = np.asarray(rows); cols = np.asarray(cols)
        h, w = src.shape
        in_bounds = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
        n_drop = int((~in_bounds).sum())
        if n_drop:
            print(f"  dropped {n_drop} samples outside raster bounds")

        loc = pd.DataFrame({
            'Serial': df_closed.index.values[in_bounds],
            'row':    rows[in_bounds],
            'col':    cols[in_bounds],
        })
        if loc.empty:
            raise ValueError("No samples remain after bounds check.")

        data      = src.read()                       # (bands, H, W)
        n_bands   = data.shape[0]
        band_names = _read_band_names(satellite_type, base_path, n_bands)
        src_transform, src_crs = src.transform, src.crs
        src_h, src_w = src.height, src.width

    # Optional: pan-sharpen SWIR to 1.5 m on the fly (uses SPOT_nir in the stack)
    if sharpen_swir:
        from helpers.feature_ops import sharpen_swir_bands
        data = sharpen_swir_bands(data, band_names)

    # Optional: reproject native 10 m embeddings onto the raster grid
    emb_grid = emb_names = None
    if use_embeddings:
        if not embeddings_path or not os.path.exists(embeddings_path):
            raise FileNotFoundError(
                f"use_embeddings=True but embeddings_path missing: {embeddings_path}")
        from helpers.feature_ops import embeddings_on_grid
        emb_grid, emb_names = embeddings_on_grid(
            embeddings_path, src_transform, src_crs, src_h, src_w,
            interpolate=emb_interpolate)

    # Aggregate samples sharing a pixel via compositional center
    agg_index = []
    agg_chem  = []
    agg_rows  = []
    agg_cols  = []
    multi_pix = 0
    for (r, c), grp in loc.groupby(['row', 'col'], sort=False):
        serials = grp['Serial'].tolist()
        block   = df_closed.loc[serials, element_cols]
        if len(serials) > 1:
            comp = coda.center(block.values)
            multi_pix += 1
        else:
            comp = block.values[0]
        agg_index.append(serials[0])
        agg_chem.append(comp)
        agg_rows.append(r); agg_cols.append(c)
    if multi_pix:
        print(f"  aggregated {multi_pix} multi-sample pixels")

    df_chem = pd.DataFrame(agg_chem, columns=element_cols, index=agg_index)
    df_chem.index.name = df_closed.index.name or 'Serial'

    ar, ac = np.array(agg_rows), np.array(agg_cols)
    if window and window > 1:
        print(f"  window: {window}x{window} median per sample")
    pixel_values = _extract_window(data, ar, ac, window).T
    df_sat = pd.DataFrame(
        pixel_values,
        columns=band_names[:pixel_values.shape[1]],
        index=agg_index,
    )
    df_sat.index.name = df_chem.index.name

    # Combine chem + bands, then compute indices on the merged frame
    df_out = pd.concat([df_chem, df_sat], axis=1)
    df_out = _add_spectral_indices(df_out, satellite_type)

    # Append embedding columns, sampled at the same aggregated pixels
    if emb_grid is not None:
        emb_vals = _extract_window(emb_grid, ar, ac, window).T
        df_emb = pd.DataFrame(emb_vals, columns=emb_names, index=agg_index)
        df_out = pd.concat([df_out, df_emb], axis=1)
        print(f"  added {emb_vals.shape[1]} embedding columns")

    n_idx = df_out.shape[1] - len(element_cols) - pixel_values.shape[1]
    print(f"  output: {len(df_out)} rows x {df_out.shape[1]} cols "
          f"({len(element_cols)} elements + {pixel_values.shape[1]} bands "
          f"+ {n_idx} indices)")
    
    print(f"✅ Satellite data integration complete")
    return df_out


# ── Helpers ────────────────────────────────────────────────────────────────

def _read_band_names(satellite_type, base_path, n_bands):
    """Pull band names from sidecar JSON if present, else generic."""
    metadata_files = {
        'fused_satellite': 'fused_satellite.json',
        'maxar':           'maxar_metadata.json',
    }
    fname = metadata_files.get(satellite_type, f"{satellite_type}.json")
    path = os.path.join(base_path, fname)
    if not os.path.exists(path):
        return [f"Band_{i+1}" for i in range(n_bands)]
    try:
        with open(path) as f:
            md = json.load(f)
        # Search any asset for eo:bands; tolerant of asset key naming
        bands = None
        for asset in md.get('assets', {}).values():
            if isinstance(asset, dict) and 'eo:bands' in asset:
                bands = asset['eo:bands']
                break
        if bands is None and 'eo:bands' in md:
            bands = md['eo:bands']
        if bands:
            names = [b.get('name') for b in bands if b.get('name')]
            if names:
                return names
        print(f"  no eo:bands in {fname}; using generic names")
    except Exception as e:
        print(f"  metadata parse failed ({e}); using generic names")
    return [f"Band_{i+1}" for i in range(n_bands)]


def _add_spectral_indices(df, satellite_type):
    """Append NDVI/SAVI and (when applicable) BSI/NDMI/Clay/NDWI/Calcite."""
    # Exact match first, then prefix match (e.g. 'fused_spot_s2_oman'
    # resolves to the 'fused_spot_s2' entry).
    b = SPECTRAL_BANDS.get(satellite_type)
    if b is None:
        b = next((v for k, v in SPECTRAL_BANDS.items()
                  if satellite_type.startswith(k)), None)
    if b is None:
        return df
    eps = 1e-8

    missing = [v for v in b.values() if v not in df.columns]
    if missing:
        print(f"  [WARN] indices skipped; missing bands: {missing}")
        return df

    red = df[b['red']]; nir = df[b['nir']]
    df['NDVI'] = (nir - red) / (nir + red + eps)
    df['SAVI'] = ((nir - red) / (nir + red + 0.5)) * 1.5
    print("  added NDVI, SAVI")

    if {'swir1', 'swir2', 'blue'}.issubset(b):
        blue  = df[b['blue']]
        swir1 = df[b['swir1']]; swir2 = df[b['swir2']]
        df['BSI']        = ((swir1 + red) - (nir + blue)) / \
                           ((swir1 + red) + (nir + blue) + eps)
        df['NDMI']       = (nir - swir1) / (nir + swir1 + eps)
        df['Clay_Index'] = swir2 / (swir1 + eps)
        print("  added BSI, NDMI, Clay_Index")

        if 'green' in b:
            green = df[b['green']]
            df['NDWI']          = (green - nir) / (green + nir + eps)
            df['Calcite_Index'] = (swir1 - swir2) / (swir1 + swir2 + eps)
            print("  added NDWI, Calcite_Index")

    return df