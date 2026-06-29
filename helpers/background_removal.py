import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial.distance import cdist
import composition_stats as coda


def background_removal(df, site_col='Site', type_col='Type',
                       easting_col='Easting', northing_col='Northing',
                       serial_col='Serial',
                       background_labels=['BACKGROUND', 'Background', 'B'],
                       buffer_distance=500,
                       use_global_background=False):
    """
    Remove background signal from site measurements using compositional
    perturbation on the simplex.

    For each site:
      1. Find background samples within `buffer_distance` of the site centroid.
      2. Compute the compositional center of those background samples
         (coda.center: closes rows, then geometric mean of closed parts,
         then re-closes). This is the correct "average composition" - NOT
         a per-element geometric mean of raw concentrations.
      3. Perturb each site sample by the inverse of this center:
              x_corrected = closure(x_closed / bg_center)
         Both operands are closed compositions, so the operation is
         subcompositionally coherent.

    Supports multiple background labels (e.g. 'Background' for Kenya, 'B'
    for Oman).
    """

    # Avoid modifying the caller's frame
    df = df.copy()
    if df.index.name == serial_col:
        df = df.reset_index()
    else:
        df = df.reset_index(drop=True)

    # Identify element columns
    exclude_cols = [site_col, type_col, easting_col, northing_col, serial_col,
                    'Longitude', 'Latitude', 'index', 'geometry', 'field_1']
    element_cols = [c for c in df.columns
                    if c not in exclude_cols and df[c].dtype in ['float64', 'int64']]

    # Separate background points
    bg_mask = df[type_col].isin(background_labels)
    df_bg = df[bg_mask].copy()

    if len(df_bg) == 0:
        print(f"Error: No samples found matching labels {background_labels} "
              f"in column '{type_col}'")
        return df

    bg_coords = df_bg[[easting_col, northing_col]].values

    # Non-background sites
    sites = df[~bg_mask][site_col].unique()

    result_frames = []

    for site in sites:
        site_mask = df[site_col] == site
        df_site = df[site_mask].copy()

        # Site centroid
        centroid = df_site[[easting_col, northing_col]].mean().values.reshape(1, -1)

        if use_global_background:
            # Use ALL background samples regardless of distance
            bg_elements = df_bg[element_cols].values
        else:
            # Background points within buffer
            distances = cdist(centroid, bg_coords, metric='euclidean').flatten()
            within_buffer = distances <= buffer_distance

            if within_buffer.sum() == 0:
                print(f"Warning: No background points within {buffer_distance}m "
                      f"of site {site}. Skipping correction.")
                keep = [serial_col] + element_cols + (
                    ['geometry'] if 'geometry' in df_site.columns else [])
                df_result = df_site[keep].set_index(serial_col)
                result_frames.append(df_result)
                continue

            bg_elements = df_bg.iloc[within_buffer.nonzero()[0]][element_cols].values
        site_elements = df_site[element_cols].values

        # Strictly-positive requirement of log-ratio ops.
        # Multiplicative replacement preserves ratios among the other parts.
        if (bg_elements <= 0).any():
            bg_elements = coda.multiplicative_replacement(
                np.where(bg_elements <= 0, 0, bg_elements))
        if (site_elements <= 0).any():
            site_elements = coda.multiplicative_replacement(
                np.where(site_elements <= 0, 0, site_elements))

        # Compositional center of the local background subset.
        # coda.center closes rows first, then takes the geometric mean
        # of the closed parts, then re-closes. This is the correct
        # "average composition" on the simplex.
        bg_center = coda.center(bg_elements)            # shape (D,), sums to 1

        # Close site rows so both operands live on the simplex.
        # atleast_2d guards against single-row sites collapsing to 1D.
        site_closed = np.atleast_2d(coda.closure(site_elements))

        # Perturbation:  x ⊖ g  ==  closure(x / g)
        corrected = coda.closure(site_closed / bg_center)
        # Single-row sites collapse to 1D; force back to 2D (1, D).
        corrected = np.atleast_2d(corrected)
        # --------------------------------------------------------------------

        # Result dataframe with Serial as index
        df_result = pd.DataFrame(corrected, columns=element_cols,
                                 index=df_site[serial_col].values)
        df_result.index.name = serial_col
        df_result['geometry'] = df_site['geometry'].values

        result_frames.append(df_result)

    # Combine all
    df_corrected = pd.concat(result_frames).sort_index()
    df_corrected = gpd.GeoDataFrame(df_corrected, geometry='geometry',
                                    crs=getattr(df, 'crs', None))

    print(f"✅ Background removal complete for {len(sites)} sites.")
    return df_corrected