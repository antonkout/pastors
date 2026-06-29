"""
Per-class outlier removal for training pixels using IsolationForest.

Fits an independent IsolationForest on each class's feature rows and drops the
`contamination` fraction flagged as outliers. Operates on the full feature
vector (bands + indices + embeddings), so it removes mixed/edge/mislabeled
pixels that sit far from their class's bulk in feature space.
"""

import numpy as np
from sklearn.ensemble import IsolationForest


def remove_outliers_per_class(X, y, contamination=0.05, random_state=42):
    """
    Parameters
    ----------
    X : np.ndarray (n_samples, n_features)
        Finite feature matrix (NaNs must already be dropped).
    y : np.ndarray (n_samples,)
        Integer class labels.
    contamination : float
        Expected outlier fraction per class (0-0.5).
    random_state : int

    Returns
    -------
    X_clean, y_clean : np.ndarray
        Filtered arrays.
    keep : np.ndarray (bool)
        Mask into the original rows (True = kept).
    """
    X = np.asarray(X)
    y = np.asarray(y)
    keep = np.ones(len(y), dtype=bool)

    print(f"Outlier removal (IsolationForest, contamination={contamination}):")
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if len(idx) < 10:
            print(f"  class {c}: {len(idx)} px (too few, kept all)")
            continue
        iso = IsolationForest(contamination=contamination,
                              random_state=random_state, n_jobs=-1)
        flags = iso.fit_predict(X[idx])      # 1 = inlier, -1 = outlier
        dropped = idx[flags == -1]
        keep[dropped] = False
        print(f"  class {c}: {len(idx)} px -> dropped {len(dropped)} "
              f"({100 * len(dropped) / len(idx):.1f}%)")

    print(f"  total kept: {keep.sum()} / {len(y)}")
    return X[keep], y[keep], keep
