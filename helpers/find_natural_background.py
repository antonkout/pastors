import numpy as np
import pandas as pd


def find_natural_background(gdf,
                              type_col='Type',
                              background_labels=('BACKGROUND', 'Background', 'B'),
                              threshold=0.5,
                              exclude_elements=('P', 'K', 'Ca', 'Sr'),
                              return_pairs=False):
    """
    Identify lithogenic (natural-background) elements via compositionally-
    correct Spearman correlation on background samples.
    Returns elements appearing in pairs with |rho| > threshold,
    ranked by frequency in those pairs.
    Marker elements never returned as background (e.g. P, K, Ca, Sr).
    """
    bg_mask = gdf[type_col].isin(list(background_labels))
    df_bg = gdf.loc[bg_mask].copy()
    if len(df_bg) < 5:
        raise ValueError(f"Only {len(df_bg)} background samples found.")

    exclude = {'Serial', 'Site', 'Type',
               'Longitude', 'Latitude', 'Easting', 'Northing',
               'Elevation', 'is_imputed', 'is_outlier', 'geometry', 'Res'}
    elem_cols = [c for c in df_bg.columns
                 if c not in exclude
                 and c not in exclude_elements                # NEW
                 and pd.api.types.is_numeric_dtype(df_bg[c])]
    
    X = df_bg[elem_cols].astype(float).to_numpy()
    X = np.where(X > 0, X, np.nan)
    col_min = np.nanmin(X, axis=0)
    fill = np.where(np.isfinite(col_min), col_min * 1e-3, 1e-9)
    X = np.where(np.isnan(X), fill, X)

    log_X = np.log(X)
    clr = log_X - log_X.mean(axis=1, keepdims=True)

    rho = pd.DataFrame(clr, columns=elem_cols).corr(method='spearman')

    iu, ju = np.triu_indices(len(elem_cols), k=1)
    pairs = pd.DataFrame({
        'element_a': np.array(elem_cols)[iu],
        'element_b': np.array(elem_cols)[ju],
        'rho':       rho.values[iu, ju],
    })
    sig = pairs.loc[pairs['rho'].abs() > threshold].copy()
    sig = sig.sort_values('rho', key=lambda s: s.abs(), ascending=False)

    counts = pd.concat([sig['element_a'], sig['element_b']]).value_counts()
    elements = counts.index.tolist()

    print(f"  background samples: {len(df_bg)}")
    print(f"  pairs |rho| > {threshold}: {len(sig)}")
    print(f"  Natural background elements: {elements}")

    if return_pairs:
        return elements, sig.reset_index(drop=True)
    return elements