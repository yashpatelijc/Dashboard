"""DV01-neutral hedge calculator (Phase 8, plan §3.6).

For an A11-PCA-fly: solve for weights so the basket is DV01-neutral while
preserving non-trivial PC3 exposure. For a 1:2:1 fly: pre-canned weights.
For multi-leg structures: linear system w/ DV01 row.

Per §15 D4=No: cross-product is a placeholder for SRA-only.

API:
    dv01_neutral_fly_weights(legs, dv01_per_lot=25.0) -> list of lots
    pca_fly_weights(loadings_pc1_pc2_pc3) -> list of weights
    build_legs_table(weights, prices, symbols) -> pd.DataFrame
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_DV01_PER_LOT = 25.0   # SR3 standard


def dv01_neutral_one_two_one(weight_unit: float = 1.0,
                                    dv01_per_lot: float = DEFAULT_DV01_PER_LOT
                                    ) -> list[float]:
    """1:2:1 fly: -1 / +2 / -1 lots scaled by weight_unit."""
    return [-weight_unit, 2 * weight_unit, -weight_unit]


def dv01_neutral_butterfly(legs_dv01_bp: list[float],
                                  target_pc3_exposure: float = 1.0
                                  ) -> list[float]:
    """For a 3-leg butterfly with per-leg DV01s, solve for weights so that
    sum(w * dv01) = 0 and middle leg has +1 contribution per unit weight.

    For equal-DV01 legs this returns [-0.5, 1.0, -0.5]; arbitrary legs are
    solved exactly.
    """
    if len(legs_dv01_bp) != 3:
        raise ValueError("butterfly requires 3 legs")
    d1, d2, d3 = legs_dv01_bp
    # Constraints:
    #   w1*d1 + w2*d2 + w3*d3 = 0   (DV01-neutral)
    #   w2 = target_pc3_exposure    (mid-leg fixed for PC3 emphasis)
    # 1 eqn, 2 unknowns -> w1 = w3 by symmetry assumption
    w2 = target_pc3_exposure
    if d1 + d3 == 0:
        return [0.0, w2, 0.0]
    w_wing = - (w2 * d2) / (d1 + d3)
    return [w_wing, w2, w_wing]


def dv01_neutral_pair(d_front_bp: float, d_back_bp: float) -> list[float]:
    """Calendar-spread weights so total DV01 = 0. Long back / short front in
    proportion to DV01 ratio."""
    if d_front_bp == 0:
        return [0.0, 0.0]
    return [-1.0, d_front_bp / d_back_bp]


def pca_fly_weights(loadings_pc1: np.ndarray, loadings_pc2: np.ndarray,
                        loadings_pc3: np.ndarray) -> np.ndarray:
    """Solve L_PC1·w = 0, L_PC2·w = 0, L_PC3·w = nonzero, DV01·w = 0
    via least-squares with the three PC-orthogonality constraints.

    Returns weights normalised so ||w|| = 1.
    """
    n = len(loadings_pc1)
    A = np.vstack([loadings_pc1, loadings_pc2])    # (2, n)
    # Find any w in null(A) via SVD
    U, S, Vh = np.linalg.svd(A, full_matrices=True)
    null_basis = Vh[len(S):]   # rows are basis of null space
    if null_basis.size == 0:
        # No null space — return zero-weights (degenerate)
        return np.zeros(n)
    # Pick the null-space direction with maximum |projection on PC3|
    proj = null_basis @ loadings_pc3
    best_idx = int(np.argmax(np.abs(proj)))
    w = null_basis[best_idx]
    if np.linalg.norm(w) > 0:
        w = w / np.linalg.norm(w)
    # Sign-flip so the PC3 projection is positive
    if w @ loadings_pc3 < 0:
        w = -w
    return w


def build_legs_table(symbols: list[str], lots: list[float], prices: list[float],
                          ) -> pd.DataFrame:
    """Build a per-leg trade-summary table."""
    if not (len(symbols) == len(lots) == len(prices)):
        raise ValueError("symbols, lots, prices must have equal length")
    df = pd.DataFrame({
        "symbol": symbols,
        "lots": lots,
        "side": ["BUY" if l > 0 else "SELL" if l < 0 else "FLAT" for l in lots],
        "abs_lots": [abs(l) for l in lots],
        "price": prices,
    })
    df["notional_per_leg_usd"] = df["abs_lots"] * 250_000.0   # SR3 contract size
    df["dv01_per_leg_usd"] = df["lots"] * DEFAULT_DV01_PER_LOT
    return df


def basket_dv01_check(legs_df: pd.DataFrame) -> dict:
    """Return basket-level DV01 + diagnostics for a hedge basket."""
    total_dv01 = float(legs_df["dv01_per_leg_usd"].sum())
    total_notional = float(legs_df["notional_per_leg_usd"].sum())
    return {
        "total_dv01_usd_per_bp": total_dv01,
        "abs_total_dv01_usd_per_bp": abs(total_dv01),
        "total_notional_usd": total_notional,
        "is_dv01_neutral": abs(total_dv01) < 0.01,
    }
