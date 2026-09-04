"""
Spatial cross-validation for the PASTORS pipeline.

Random KFold over pixels leaks: pixels from the same site are spatially
autocorrelated and routinely land in both train and test, inflating the
score relative to what the model achieves on an unseen site. This module
holds *whole* groups out at once — either one group per site
(leave-site-out) or KMeans spatial blocks on the sample coordinates — and
reports out-of-fold metrics, including Spearman alongside Pearson.

Public API
----------
    regression_metrics(y, yhat)                 -> dict (incl. Spearman)
    make_spatial_groups(...)                    -> (groups, labels_map)
    spatial_cv_predict(X, y, groups, ...)       -> (oof_pred, fold_r2, name)
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    balanced_accuracy_score, roc_auc_score, matthews_corrcoef,
    cohen_kappa_score, f1_score, confusion_matrix,
)
from xgboost import XGBRegressor


def regression_metrics(y, yhat):
    """R2, RMSE, MAE, Pearson and Spearman on paired arrays."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    return dict(
        R2=float(r2_score(y, yhat)),
        RMSE=float(np.sqrt(mean_squared_error(y, yhat))),
        MAE=float(mean_absolute_error(y, yhat)),
        Pearson=float(np.corrcoef(y, yhat)[0, 1]),
        Spearman=float(spearmanr(y, yhat).correlation),
    )


def resolve_threshold(spec, y_ref):
    """
    Turn a threshold spec into a number on the balance's own scale.

    spec : float          -> used as-is (0.0 = neutral point after background
                             removal: >0 enriched, <0 depleted).
           'antimode'     -> GMM 2-component antimode learned from `y_ref`
                             (the training balance distribution); falls back
                             to 0.0 if that distribution is unimodal.
    """
    if isinstance(spec, (int, float)):
        return float(spec)
    if spec == 'antimode':
        from helpers.pb_populations import population_split
        r = population_split(np.asarray(y_ref, dtype=float), save_path=None)
        return float(r['threshold']) if r.get('bimodal') else 0.0
    raise ValueError(f"threshold spec must be a number or 'antimode', got {spec!r}")


def agreement_metrics(y_true, y_pred, threshold=0.0):
    """
    Direction-not-magnitude agreement: does the prediction put each sample on
    the correct side (elevated vs depleted) of `threshold`?

    Returns balanced accuracy, AUC (threshold-free ranking of the continuous
    prediction against the true class), MCC, Cohen's kappa, F1, plain
    accuracy, class counts and the confusion cells. Class-dependent metrics
    are NaN if the true labels are single-class at this threshold.
    """
    yt_c = np.asarray(y_true, dtype=float)
    yp_c = np.asarray(y_pred, dtype=float)
    yt = (yt_c > threshold).astype(int)
    yp = (yp_c > threshold).astype(int)

    out = dict(threshold=float(threshold),
               n_elevated=int(yt.sum()), n_depleted=int((yt == 0).sum()),
               accuracy=float((yt == yp).mean()))

    if yt.min() != yt.max():
        out['balanced_acc'] = float(balanced_accuracy_score(yt, yp))
        out['MCC'] = float(matthews_corrcoef(yt, yp))
        out['kappa'] = float(cohen_kappa_score(yt, yp))
        out['F1'] = float(f1_score(yt, yp, zero_division=0))
        try:                       # AUC ranks the continuous prediction
            out['AUC'] = float(roc_auc_score(yt, yp_c))
        except ValueError:
            out['AUC'] = float('nan')
    else:
        for k in ('balanced_acc', 'MCC', 'kappa', 'F1', 'AUC'):
            out[k] = float('nan')

    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    out.update(TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))
    return out


def make_spatial_groups(sites=None, coords=None, strategy='site',
                        n_blocks=5, seed=42):
    """
    Build integer group labels for group-aware cross-validation.

    Parameters
    ----------
    sites : array-like, optional
        Per-sample site label. Required for strategy='site'.
    coords : array-like (n, 2), optional
        Per-sample (x, y) coordinates. Required for strategy='block'.
    strategy : {'site', 'block'}
        'site'  -> one group per unique site label (leave-site-out).
        'block' -> KMeans on coordinates into `n_blocks` spatial blocks
                   (use when sites are too few, or one site is huge).
    n_blocks : int
        Number of spatial blocks for strategy='block'.
    seed : int
        KMeans seed.

    Returns
    -------
    groups : np.ndarray[int]
        Integer group id per sample.
    labels_map : dict[int, str]
        group id -> human-readable label (site name or 'block{i}').
    """
    if strategy == 'site':
        if sites is None:
            raise ValueError("strategy='site' requires `sites`.")
        sites = np.asarray(sites)
        order = sorted(set(sites.tolist()))
        uniq = {s: i for i, s in enumerate(order)}
        groups = np.array([uniq[s] for s in sites])
        return groups, {i: s for s, i in uniq.items()}

    if strategy == 'block':
        if coords is None:
            raise ValueError("strategy='block' requires `coords` (n, 2).")
        coords = np.asarray(coords, dtype=float)
        k = min(n_blocks, len(np.unique(coords, axis=0)))
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(coords)
        return km.labels_, {i: f'block{i}' for i in range(k)}

    raise ValueError(f"unknown strategy: {strategy!r} (use 'site' or 'block').")


def _splitter(groups, max_folds=None):
    """LeaveOneGroupOut, or GroupKFold when there are more groups than folds."""
    n_groups = len(np.unique(groups))
    if max_folds is None or max_folds >= n_groups:
        return LeaveOneGroupOut(), n_groups
    return GroupKFold(n_splits=max_folds), max_folds


def _clean_params(params):
    """Drop training-control keys that fit_final/CV set themselves."""
    return {k: v for k, v in params.items()
            if k not in ('n_estimators', 'early_stopping_rounds')}


def spatial_cv_predict(X, y, groups, params=None, n_estimators=None,
                       seed=42, max_folds=None,
                       objective='reg:squarederror', model_fn=None):
    """
    Group-aware out-of-fold predictions: each fold trains on all groups but
    one (or a GroupKFold partition) and predicts the held-out group.

    Model-agnostic: pass `model_fn` to compare any regressor (e.g. TabPFN)
    under the IDENTICAL splits. Without it, the default XGBoost path is used
    so existing callers keep working.

    Parameters
    ----------
    X, y : arrays
        Feature matrix and target.
    groups : array-like[int]
        Group id per sample (from `make_spatial_groups`).
    params : dict, optional
        XGBoost params (as returned by tune_xgb). Used only on the default
        XGBoost path; ignored when `model_fn` is given.
    n_estimators : int, optional
        Boosting rounds per fold (default XGBoost path only).
    max_folds : int, optional
        Cap the number of folds (GroupKFold). None -> leave-one-group-out.
    objective : str
        XGBoost training loss (default path only).
    model_fn : callable, optional
        Zero-arg factory returning a FRESH unfitted estimator exposing
        .fit(X, y) and .predict(X). When supplied, each fold trains a new
        estimator from it and `params`/`n_estimators`/`objective` are unused.
        Example: `model_fn=lambda: TabPFNRegressor(n_estimators=8)`.

    Returns
    -------
    oof_pred : np.ndarray
        Out-of-fold prediction for every sample.
    fold_r2 : list[float]
        Per-fold R2 (folds with a constant held-out target are skipped).
    name : str
        Splitter description, e.g. 'LeaveOneGroupOut' or 'GroupKFold(5)'.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)

    if model_fn is None:
        fit_params = _clean_params(params or {})

        def model_fn():
            return XGBRegressor(**fit_params, n_estimators=n_estimators,
                                objective=objective, random_state=seed,
                                n_jobs=-1)

    splitter, n = _splitter(groups, max_folds)
    oof = np.full(len(y), np.nan, dtype=float)
    fold_r2 = []

    for tr, te in splitter.split(X, y, groups):
        m = model_fn()
        m.fit(X[tr], y[tr])
        p = np.asarray(m.predict(X[te]), dtype=float)
        oof[te] = p
        if len(np.unique(y[te])) > 1:
            fold_r2.append(float(r2_score(y[te], p)))

    name = ('LeaveOneGroupOut'
            if isinstance(splitter, LeaveOneGroupOut)
            else f'GroupKFold({n})')
    return oof, fold_r2, name
