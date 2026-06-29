"""
Principal-balance population detection.

Given the values of a (principal) balance across samples, decide whether the
distribution is one population or two (background vs enriched), via a 1- vs
2-component Gaussian mixture compared by BIC. If bimodal, the antimode (the
0.5-posterior crossover between the two component means) is a data-driven
enrichment threshold.

Idea from Meloni et al. (2026) / Scheffer attraction-basin reading of PB
density distributions: bimodal balance = mixed populations; the valley splits
background from anomalous/activity-enriched samples.
"""

import numpy as np
from sklearn.mixture import GaussianMixture


def population_split(values, seed=42, plot=False, save_path=None, label='PB1'):
    """
    Parameters
    ----------
    values : 1-D array of balance values (one per sample).

    Returns
    -------
    dict with: bimodal, bic1, bic2, and (if bimodal) means, weights,
    threshold (antimode), n_low, n_high.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    X = v.reshape(-1, 1)

    g1 = GaussianMixture(1, random_state=seed).fit(X)
    g2 = GaussianMixture(2, random_state=seed).fit(X)
    bic1, bic2 = g1.bic(X), g2.bic(X)
    bimodal = bic2 < bic1
    res = {'bimodal': bool(bimodal), 'bic1': float(bic1), 'bic2': float(bic2)}

    thr = None
    if bimodal:
        means = g2.means_.ravel()
        order = np.argsort(means)
        lo, hi = means[order]
        grid = np.linspace(v.min(), v.max(), 2000)
        prob_low = g2.predict_proba(grid.reshape(-1, 1))[:, order[0]]
        band = (grid >= lo) & (grid <= hi)
        thr = float(grid[band][np.argmin(np.abs(prob_low[band] - 0.5))])
        res.update(means=means[order].tolist(),
                   weights=g2.weights_[order].tolist(),
                   threshold=thr,
                   n_low=int((v < thr).sum()),
                   n_high=int((v >= thr).sum()))

    if plot or save_path:
        import os
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(v, bins=30, density=True, alpha=0.5, color='steelblue')
        xs = np.linspace(v.min(), v.max(), 500).reshape(-1, 1)
        ax.plot(xs, np.exp(g2.score_samples(xs)), 'k-', lw=2,
                label='2-comp GMM' if bimodal else '2-comp (rejected)')
        if thr is not None:
            ax.axvline(thr, color='red', ls='--',
                       label=f'antimode = {thr:.2f}')
        ax.set_title(f"{label} population split  "
                     f"({'BIMODAL' if bimodal else 'unimodal'}; "
                     f"BIC 1={bic1:.0f} / 2={bic2:.0f})")
        ax.set_xlabel('balance value'); ax.set_ylabel('density'); ax.legend()
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=140)
        if plot:
            plt.show()
        plt.close(fig)

    return res
