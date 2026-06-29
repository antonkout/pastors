"""
Permutation-importance feature selection.

Public function:
    select_features(model, X, y, feature_names, keep_frac=0.5)
        -> (kept_names, importances_series)
"""

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def select_features(model, X, y, feature_names,
                    keep_frac=0.5, n_repeats=30, seed=42, verbose=True):
    """
    Run permutation importance on a fitted model and return the top features.

    Parameters
    ----------
    model : fitted estimator with .predict
    X, y  : evaluation data (typically the held-out test set)
    feature_names : list[str], same order as columns of X
    keep_frac : float in (0, 1], fraction of *positive-importance* features to keep
    n_repeats : permutation repeats per feature

    Returns
    -------
    kept_names : list[str]
    importances : pd.Series, sorted descending
    """
    perm = permutation_importance(
        model, X, y,
        n_repeats=n_repeats, random_state=seed,
        n_jobs=-1, scoring='r2',
    )
    importances = pd.Series(
        perm.importances_mean, index=feature_names,
    ).sort_values(ascending=False)

    positive = importances[importances > 0]
    keep_n = max(5, int(np.ceil(len(positive) * keep_frac)))
    kept = positive.head(keep_n).index.tolist()

    if verbose:
        print(f"Permutation importance: {len(positive)} / {len(feature_names)} "
              f"features have positive contribution")
        print(f"Keeping top {len(kept)} (keep_frac={keep_frac})")
        print(importances.head(15).round(4))

    return kept, importances