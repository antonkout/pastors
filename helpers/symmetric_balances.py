"""
Symmetric Balances for Compositional Data Analysis

This module implements symmetric balances as described in Kynčlová et al. (2017)
for analyzing pairwise relationships in compositional data while respecting
the geometry of the simplex.

Uses composition_stats library for core compositional operations.

References:
    Kynčlová, P., Hron, K., & Filzmoser, P. (2017). Correlation between 
    compositional parts based on symmetric balances. Mathematical Geosciences, 
    49(6), 777-796.
"""

import numpy as np
from typing import Tuple, Optional
import pandas as pd
import composition_stats as coda

closure = coda.closure
multiplicative_replacement = coda.multiplicative_replacement
clr = coda.clr
clr_inv = coda.clr_inv
alr = coda.alr
alr_inv = coda.alr_inv
ilr = coda.ilr
ilr_inv = coda.ilr_inv
center = coda.center
centralize = coda.centralize

def symmetric_balance_coefficients(D: int) -> Tuple[float, float]:
    """
    Calculate the α and β coefficients for symmetric balances.
    """

    if D < 3:
        raise ValueError("D must be at least 3 for symmetric balances")
    
    sqrt_D = np.sqrt(D)
    sqrt_D_minus_2 = np.sqrt(D - 2)
    denominator = D - 1 + sqrt_D * sqrt_D_minus_2
    
    alpha = 1.0 / denominator
    beta = (D - 2 + sqrt_D) / (sqrt_D_minus_2 * denominator)
    
    return alpha, beta


def symmetric_balance_pair(composition: np.ndarray, i: int, j: int, alpha: Optional[float] = None, beta: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute symmetric balances for parts i and j.
    """

    if composition.ndim != 2:
        raise ValueError("Composition must be a 2D array")
    
    n_samples, D = composition.shape
    
    if i == j:
        raise ValueError("Indices i and j must be different")
    if i < 0 or i >= D or j < 0 or j >= D:
        raise ValueError(f"Indices must be between 0 and {D-1}")
    
    if np.any(composition <= 0):
        raise ValueError("All compositional values must be positive. "
                        "Consider using multiplicative replacement for zeros.")
    
    if alpha is None or beta is None:
        alpha, beta = symmetric_balance_coefficients(D)
    
    x_i, x_j = composition[:, i], composition[:, j]
    
    # Calculate product of other parts and create mask for parts that are not i or j
    mask = np.ones(D, dtype=bool)
    mask[i], mask[j] = False, False
    other_parts = composition[:, mask]
    
    # Product of other parts: ∏_{k≠i,j} x_k using log-sum-exp trick for numerical stability
    log_prod_others = np.sum(np.log(other_parts), axis=1)
    
    # Calculate the first term
    sqrt_term = np.sqrt((D - 1 + np.sqrt(D * (D - 2))) / (2 * D))
    log_ratio_ij = np.log(x_i / x_j)
    first_term = sqrt_term * log_ratio_ij
    
    # Calculate the second term: ln[(x_i^α * x_j^α) / (∏_{k≠i,j} x_k)^β] = α*ln(x_i) + α*ln(x_j) - β*ln(∏_{k≠i,j} x_k)
    second_term = alpha * (np.log(x_i) + np.log(x_j)) - beta * log_prod_others
    
    z_i_s = first_term + second_term
    z_j_s = -first_term + second_term
    
    return z_i_s, z_j_s


def group_balance(df, markers, background):
    """
    Compute one ILR balance comparing marker elements against background.

    Sign convention:
        +  markers enriched (e.g. pastoral signal)
        0  equal log-proportions
        -  background dominates

    Parameters
    ----------
    df : pd.DataFrame
        Closed compositional data; rows = samples. Must contain all
        element columns referenced in `markers` and `background`. Any
        additional columns (other elements, satellite bands, indices)
        are ignored.
    markers : list[str]
        Element names assigned +1 in the SBP (numerator).
    background : list[str]
        Element names assigned -1 in the SBP (denominator).

    Returns
    -------
    pd.Series
        Balance value per sample, indexed like df.
    """
    overlap = set(markers) & set(background)
    if overlap:
        raise ValueError(
            f"markers and background overlap: {sorted(overlap)}. "
            "An element cannot be both numerator and denominator.")

    # Keep only parts present in df (drop missing elements, e.g. not measured
    # in the validation set) and compute the balance on what is available.
    dropped = [p for p in list(markers) + list(background) if p not in df.columns]
    if dropped:
        print(f"  [group_balance] dropping missing columns: {dropped}")
    markers    = [m for m in markers    if m in df.columns]
    background = [b for b in background if b in df.columns]
    if not markers or not background:
        raise ValueError(
            "After dropping missing columns, markers or background is empty "
            f"(markers={markers}, background={background}).")
    parts = list(markers) + list(background)

    comp = df[parts].astype(float).to_numpy()
    if (comp <= 0).any():
        comp = coda.multiplicative_replacement(
            np.where(comp <= 0, 0, comp))

    # Re-close to unit sum (operates on this subset only)
    comp = comp / comp.sum(axis=1, keepdims=True)

    # SBP: +1 markers, -1 background
    n_parts = comp.shape[1]
    sbp = np.zeros((1, n_parts), dtype=int)
    sbp[0, :len(markers)] = 1
    sbp[0, len(markers):] = -1

    basis = coda.sbp_basis(sbp)
    if basis.ndim == 1:
        basis = basis.reshape(1, -1)

    balance = coda.ilr(comp, basis=basis).flatten()
    print("✅ Completed balance comparing between marker-elements and natural-background.")
    return pd.Series(balance, index=df.index, name="group_balance")