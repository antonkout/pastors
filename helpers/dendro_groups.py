"""
Compositional dendrogram + automatic marker / natural-background split.

Clusters the parts (elements) by Aitchison variation (log-ratio variance), cuts
the tree, and labels the tightest-covarying cluster as the natural background;
everything else (the high-variability, activity-driven parts) becomes markers.

Replaces a Pearson-correlation selection, which is unsafe on closed data
(spurious correlation from the constant-sum constraint). Log-ratio variation is
the correct compositional association measure.
"""

import os
import numpy as np
import geopandas as gpd
import composition_stats as coda
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, to_tree
from scipy.spatial.distance import squareform


def _sbp_from_tree(Z, D):
    """Sequential binary partition (D-1 x D sign matrix) from a linkage tree."""
    def leaves(n):
        return [n.id] if n.is_leaf() else leaves(n.left) + leaves(n.right)
    rows = []

    def visit(n):
        if n.is_leaf():
            return
        L, R = leaves(n.left), leaves(n.right)
        row = np.zeros(D, int); row[L] = 1; row[R] = -1
        rows.append(row); visit(n.left); visit(n.right)
    visit(to_tree(Z))
    return np.array(rows)


def _principal_balances(SBP, logC, names):
    """Balances [Eq. 3] for the SBP, ranked by variance (principal balances)."""
    bal = []
    for row in SBP:
        nu, de = row == 1, row == -1
        r, s = nu.sum(), de.sum()
        bal.append(np.sqrt(r * s / (r + s)) *
                   (logC[:, nu].mean(1) - logC[:, de].mean(1)))
    B = np.array(bal).T
    varb = B.var(0, ddof=1)
    total = float(varb.sum())
    order = np.argsort(varb)[::-1]
    out = []
    for rank, k in enumerate(order, 1):
        out.append({
            'rank': rank,
            'plus':  [names[i] for i in np.where(SBP[k] == 1)[0]],
            'minus': [names[i] for i in np.where(SBP[k] == -1)[0]],
            'var':   float(varb[k]),
            'pct':   float(100 * varb[k] / total),
        })
    top_balance = B[:, order[0]]          # rank-1 (highest-variance) balance per sample
    return out, total, top_balance

# Non-compositional columns shared by every *_processed_closed.geojson.
# Everything else is treated as a composition part (element / residual).
NON_PART_COLUMNS = {
    'Serial', 'Site', 'Type', 'Longitude', 'Latitude', 'Easting', 'Northing',
    'is_imputed', 'is_outlier', 'geometry',
}


def detect_parts(g):
    """Auto-detect composition parts from a (geo)dataframe.

    Returns the numeric columns that are not metadata/geometry, preserving the
    column order in the file. This adapts to per-dataset element sets (e.g. Kenya
    has Y/Ba, Oman has Ni/Cu, validation has no Zr) instead of failing on a fixed
    list.
    """
    import pandas as pd
    parts = []
    for c in g.columns:
        if c in NON_PART_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(g[c]):
            parts.append(c)
    return parts


# Kept for reference / explicit override; analysis now auto-detects by default.
DEFAULT_PARTS = ['Mg', 'Al', 'Si', 'P', 'K', 'Ca', 'Ti', 'Mn',
                 'Fe',  'Zn', 'Rb', 'Sr', 'Zr', 'Res'] #'Ni', 'Cu',


def _filter(g, include_sites, exclude_sites, type_prefix,
            include_types, exclude_types, exclude_outliers):
    df = g.copy()
    s, t = df['Site'].astype(str), df['Type'].astype(str)
    if include_sites:    df = df[df['Site'].astype(str).isin(include_sites)]
    if exclude_sites:    df = df[~df['Site'].astype(str).isin(exclude_sites)]
    if type_prefix:      df = df[df['Type'].astype(str).str.startswith(type_prefix)]
    if include_types:    df = df[df['Type'].astype(str).isin(include_types)]
    if exclude_types:    df = df[~df['Type'].astype(str).isin(exclude_types)]
    if exclude_outliers and 'is_outlier' in df.columns:
        df = df[~df['is_outlier'].astype(bool)]
    return df


def _variation_matrix(P):
    L = np.log(P)
    D = L.shape[1]
    T = np.zeros((D, D))
    for i in range(D):
        for j in range(i + 1, D):
            T[i, j] = T[j, i] = np.var(L[:, i] - L[:, j], ddof=1)
    return T


def compositional_groups(
    geojson_path=None,
    parts=None,
    exclude_parts=('Res',),
    include_sites=None, exclude_sites=None, type_prefix=None,
    include_types=None, exclude_types=None, exclude_outliers=False,
    linkage_method='ward', color_frac=0.6,
    background_by='variation',          # 'variation' (tightest) or 'size' (largest)
    plot=False, save_path=None, title=None,
    gdf=None,                           # preloaded GeoDataFrame (skips file read)
):
    """
    Returns dict with: markers, natural_bgk, parts, clusters, n_samples, Z, T.

    Pass either `geojson_path` (read from disk) or a preloaded `gdf` (faster when
    calling repeatedly, e.g. from search_site_combinations).
    """
    g = gdf if gdf is not None else gpd.read_file(geojson_path)
    if parts is None:
        parts = detect_parts(g)
    df = _filter(g, include_sites, exclude_sites, type_prefix,
                 include_types, exclude_types, exclude_outliers)
    if len(df) < 3:
        raise ValueError(f"Only {len(df)} rows after filtering.")

    # Only keep requested parts that actually exist in this dataset.
    missing = [p for p in parts if p not in df.columns]
    use = [p for p in parts
           if p in df.columns and p not in set(exclude_parts or ())]
    if not use:
        raise ValueError(
            f"No usable composition parts found. parts={parts}, missing={missing}.")
    P = df[use].astype(float).values
    if (P <= 0).any():
        raise ValueError("Composition has non-positive values; impute zeros first.")
    P = coda.closure(P)

    T = _variation_matrix(P)
    Z = linkage(squareform(np.sqrt(T), checks=False), method=linkage_method)

    # Principal balances (PBA cluster method, Martin-Fernandez 2018 Sect 4.3)
    SBP = _sbp_from_tree(Z, len(use))
    pbs, total_var, top_balance = _principal_balances(SBP, np.log(P), use)

    thr = color_frac * Z[:, 2].max()
    lab = fcluster(Z, t=thr, criterion='distance')

    # cluster stats
    groups = {}
    for ci in np.unique(lab):
        idx = np.where(lab == ci)[0]
        if len(idx) >= 2:
            sub = T[np.ix_(idx, idx)]
            mean_var = sub[np.triu_indices(len(idx), 1)].mean()
        else:
            mean_var = np.inf
        groups[ci] = {'idx': idx, 'size': len(idx), 'mean_var': mean_var,
                      'parts': [use[k] for k in idx]}

    cand = {c: v for c, v in groups.items() if v['size'] >= 2}
    if background_by == 'size':
        bg = max(cand, key=lambda c: (cand[c]['size'], -cand[c]['mean_var']))
    else:
        bg = min(cand, key=lambda c: (cand[c]['mean_var'], -cand[c]['size']))

    natural_bgk = groups[bg]['parts']
    markers = [p for p in use if p not in natural_bgk]

    if plot or save_path:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 6))
        dendrogram(Z, labels=use, leaf_font_size=12,
                   color_threshold=thr, ax=ax)
        ax.set_title((title or f"Compositional dendrogram (n={len(df)}, {len(use)} parts)")
                     + f"\nbackground = {sorted(natural_bgk)}", fontsize=10)
        ax.set_ylabel("√ variation distance = √Var(ln $x_i/x_j$)")
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=140)
        if plot:
            plt.show()
        plt.close(fig)

    return {
        'markers': markers,
        'natural_bgk': natural_bgk,
        'parts': use,
        'n_samples': len(df),
        'clusters': {int(c): v['parts'] for c, v in groups.items()},
        'principal_balances': pbs,
        'total_variance': total_var,
        'top_balance': top_balance,
        'Z': Z, 'T': T,
    }


def _score_sites(gdf, sites, objective, min_samples, comp_kwargs):
    """Run the PBA on one site subset; return its score row or None if invalid."""
    try:
        r = compositional_groups(gdf=gdf, include_sites=list(sites), **comp_kwargs)
    except (ValueError, KeyError):
        return None
    if r['n_samples'] < min_samples:
        return None
    pb1 = r['principal_balances'][0]          # rank-1 (highest-variance) balance
    score = pb1['pct'] if objective == 'pct' else pb1['var']
    return {
        'sites': tuple(sites),
        'n_sites': len(sites),
        'n_samples': r['n_samples'],
        'pb1_var': pb1['var'],
        'pb1_pct': pb1['pct'],
        'total_variance': r['total_variance'],
        'pb1_plus': pb1['plus'],
        'pb1_minus': pb1['minus'],
        'score': score,
    }


def search_site_combinations(
    geojson_path,
    candidate_sites=None,
    objective='var',                  # 'var' = rank-1 PB variance, 'pct' = its %var
    method='exhaustive',              # 'exhaustive' or 'greedy'
    min_sites=1, max_sites=None,
    min_samples=10,
    max_combos=40000,
    top_n=20,
    return_dataframe=True,
    # ---- everything below is forwarded to compositional_groups ----
    parts=None, exclude_parts=('Res',),
    type_prefix=None, include_types=None, exclude_types=None,
    exclude_outliers=False, linkage_method='ward',
):
    """Search Site subsets that maximize the rank-1 principal balance.

    Reproduces the manual INCLUDE_SITES tuning in pba_dendro: instead of guessing
    which sites to add/remove, it scores every (or greedily grows the) combination
    and ranks them by the top principal balance — the "0.248 / 35.2%" figure.

    objective : 'var' ranks by absolute rank-1 variance; 'pct' by its share of
                the total log-ratio variance.
    method    : 'exhaustive' tries all subsets (guarded by max_combos);
                'greedy' does forward selection (good when there are many sites).

    Returns a pandas DataFrame (or list of dicts) sorted best-first. The top row's
    'sites' is what to drop into INCLUDE_SITES.
    """
    import itertools
    import pandas as pd

    g = gpd.read_file(geojson_path)

    # Apply the type/outlier filters ONCE so candidate sites and every scored
    # subset are drawn from the same restricted frame.
    base = _filter(g, None, None, type_prefix,
                   include_types, exclude_types, exclude_outliers)
    if candidate_sites is None:
        candidate_sites = sorted(base['Site'].astype(str).unique())
    else:
        candidate_sites = [str(s) for s in candidate_sites]

    comp_kwargs = dict(
        parts=parts, exclude_parts=exclude_parts, type_prefix=type_prefix,
        include_types=include_types, exclude_types=exclude_types,
        exclude_outliers=exclude_outliers, linkage_method=linkage_method,
    )

    n = len(candidate_sites)
    max_sites = max_sites or n
    rows = []

    if method == 'greedy':
        chosen, results = [], []
        remaining = list(candidate_sites)
        best_overall = None
        while remaining and len(chosen) < max_sites:
            step_best = None
            for s in remaining:
                trial = chosen + [s]
                if len(trial) < min_sites:
                    # still record but can't be selected as final yet; score anyway
                    pass
                row = _score_sites(g, trial, objective, min_samples, comp_kwargs)
                if row and (step_best is None or row['score'] > step_best['score']):
                    step_best = row; step_best_site = s
            if step_best is None:
                break
            chosen.append(step_best_site)
            remaining.remove(step_best_site)
            results.append(step_best)
            if (best_overall is None or step_best['score'] > best_overall['score']):
                best_overall = step_best
        rows = results
    else:
        # count combinations first to fail fast on explosion
        total = sum(_n_choose_k(n, k) for k in range(max(min_sites, 1), max_sites + 1))
        if total > max_combos:
            raise ValueError(
                f"{total} combinations exceed max_combos={max_combos}. "
                f"Use method='greedy', or restrict candidate_sites / max_sites.")
        for k in range(max(min_sites, 1), max_sites + 1):
            for combo in itertools.combinations(candidate_sites, k):
                row = _score_sites(g, combo, objective, min_samples, comp_kwargs)
                if row:
                    rows.append(row)

    rows.sort(key=lambda r: r['score'], reverse=True)
    rows = rows[:top_n] if top_n else rows
    if return_dataframe:
        return pd.DataFrame(rows)
    return rows


def _n_choose_k(n, k):
    from math import comb
    return comb(n, k) if 0 <= k <= n else 0
