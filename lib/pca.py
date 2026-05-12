"""PCA factor-model engine for the SRA constant-maturity curve.

Full-depth implementation backing the SRA › PCA peer subtab. Covers ten studies:

  S1.  CMC (constant-maturity curve) substrate            — gameplan §A12
       PCHIP through reference-quarter midpoints + analytical front-pin from
       the CME polynomial accrual using SOFR fixings.

  S2.  Static PCA on full post-cutover sample              — gameplan §A1.1
       SVD on centred Δ_CMC, sign-normalised PC1/PC2/PC3.

  S3.  Sparse PCA companion                                — gameplan §A1.3
       L1 soft-thresholding + Gram-Schmidt re-orthonormalisation.
       In-house, no sklearn dependency.

  S4.  Rolling PCA refit + loadings stability              — gameplan §A1.6
       252-day window, weekly cadence, sign-aligned across refits.

  S5.  PC time-series diagnostics                          — gameplan §A6
       Multi-lookback z, percentile rank, ADF, Hurst, OU half-life,
       composite, confluence pattern, cross-PC correlation, Anchor metric.

  S6.  Today's Δ_CMC decomposition                         — gameplan §A1
       Per-tenor stacked attribution + aggregate shares.

  S7.  PCA-derived structures (PC1/PC2/PC3 isolated)       — gameplan §A11
       PC3 flies (curvature-isolated), PC2 spreads (slope-isolated, 3-leg
       cross-product + 4-leg least-squares), PC1 synthetics (level-isolated).

  S8.  PC-residual rich/cheap                              — gameplan §A1
       Outrights + traded spreads + traded flies + synthetic packs.

  S9.  Regime & cycle map (using lib.regime.classify_regime)
       Trajectory + persistence + cycle phase tag.

  S10. Diagnostics                                         — gameplan §A12 gates
       Reconstruction error, rolling variance ratio, outlier days,
       loadings-stability heatmap data, print-quality alerts.

LOCKED conventions (mirrored from existing lib/* modules):

  · History windows EXCLUDE the as-of date itself (`< ts`, not `<=`).
  · `ddof=1` for sample std.
  · BP scaling via `lib.contract_units.bp_multipliers_for`.
  · Tests robust — short series / NaNs / degenerate cases return None.
  · `eff_n` floor of 30 for any analog-weighted quantity.
  · No sklearn, no statsmodels. scipy only for PchipInterpolator (with
    a hand-rolled Fritsch-Carlson fallback if scipy is absent).
  · Pre-2023-08-01 SRA is excluded from primary PCA fits (LIBOR→SOFR cutover);
    such samples are tagged `gate_quality = pre_transition`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from lib.contract_units import bp_multipliers_for, load_catalog, parse_legs
from lib.fomc import (
    get_fomc_dates_in_range,
    is_quarterly,
    parse_sra_outright,
    reference_period,
    third_wednesday,
)
from lib.mean_reversion import (
    adf_test,
    composite_reversion_score,
    hurst_exponent,
    hurst_label,
    ou_half_life,
    percentile_rank_value,
    zscore_value,
)
from lib.proximity import classify_proximity_pattern, NEAR_ATR
from lib.regime import classify_regime
from lib.sra_data import (
    DEFAULT_FRONT_END,
    DEFAULT_MID_END,
    LIVENESS_DAYS,
    PACK_NAMES,
    compute_pack_groups as _sra_compute_pack_groups,
    get_flies as _sra_get_flies,
    get_outrights as _sra_get_outrights,
    get_spreads as _sra_get_spreads,
    get_sra_snapshot_latest_date,
    load_reference_rate_panel as _sra_load_reference_rate_panel,
    load_sra_curve_panel,
    pivot_curve_panel,
)
# Generic multi-market data loaders (used when base_product != "SRA")
from lib import market_data as _md
from lib.markets import get_market as _get_market


# Back-compat aliases — SRA-only entrypoints kept for legacy tabs
def get_outrights() -> "pd.DataFrame":
    """SRA-only outrights (back-compat). New code should use base_product-aware
    `lib.market_data.get_outrights(base_product)` instead."""
    return _sra_get_outrights()


def get_spreads(tenor_months: int) -> "pd.DataFrame":
    return _sra_get_spreads(tenor_months)


def get_flies(tenor_months: int) -> "pd.DataFrame":
    return _sra_get_flies(tenor_months)


def compute_pack_groups(symbols_df, base_product: str = "SRA"):
    """Market-aware pack groups. Defaults to SRA for back-compat."""
    if base_product == "SRA":
        return _sra_compute_pack_groups(symbols_df)
    return _md.compute_pack_groups(symbols_df, base_product)


def load_reference_rate_panel(start_date=None, end_date=None,
                                 base_product: str = "SRA"):
    """Market-aware reference rate panel. Defaults to SRA (uses FDTR + SOFR)."""
    if base_product == "SRA":
        return _sra_load_reference_rate_panel(start_date, end_date)
    return _md.load_reference_rate_panel(base_product, start_date, end_date)


# Hard cutover date — primary PCA fits begin here.
LIBOR_SOFR_CUTOVER = date(2023, 8, 1)

# Standard τ-grid (months) — gameplan §A12 specifies 3M..60M monthly = 58 nodes.
DEFAULT_TENOR_GRID_MONTHS = list(range(3, 61, 1))   # 3, 4, 5, ..., 60 (58 nodes)
# Coarser grid retained for diagnostics / rapid iteration — 20 nodes (3..60 step 3).
COARSE_TENOR_GRID_MONTHS = list(range(3, 61, 3))
FINE_TENOR_GRID_MONTHS = DEFAULT_TENOR_GRID_MONTHS  # alias for back-compat

# Default lookbacks for PC time-series diagnostics.
DEFAULT_PC_LOOKBACKS = (5, 15, 30, 60, 90, 252)

# Eff-n floor for any sample-window-driven gate.
EFF_N_FLOOR = 30

# Print-quality alert threshold (gameplan §A12 verification gate iv).
CMC_PRINT_QUALITY_BP = 50.0


# =============================================================================
# Trade Horizon Mode — unified parameter set
# =============================================================================
# Single source of truth for every horizon-sensitive parameter. UI selects a
# mode and routes it to build_full_pca_panel; downstream functions read params
# from this dict (via mode_params()) so changing one value here updates the
# whole engine consistently.
#
# Intraday:   60d-of-history view, fast mean-reversion (HL 0.5-7d), 1σ trigger
# Swing:      legacy behavior — 700d history, 60d window, HL 3-30d, 1.5σ trigger
# Positional: 4y history, 252d window, HL 15-90d sweet spot, 2σ trigger,
#             21-120d hold window (matches a positional STIRS trader's cadence)
MODE_PARAMS = {
    "intraday": {
        "residual_lookback": 30,
        "triple_gate_lookback": 60,
        "history_days": 250,
        "z_threshold": 1.0,
        "hold_mult": 1.0,
        "hold_floor": 1,
        "hold_cap": 14,
        "min_eff_n": 30,
        "detrend_window": 90,
        "sweet_spot_full": (0.5, 7.0),
        "sweet_spot_partial_below": 0.5,   # HL < this gets 0.3 credit
        "sweet_spot_partial_above": 15.0,  # HL > this gets 0.1 credit
        "sweet_spot_mid_band": (7.0, 15.0),  # HL in this gets 0.5 credit
        "hl_extension_stop_mult": 2.0,
    },
    "swing": {
        "residual_lookback": 60,
        "triple_gate_lookback": 90,
        "history_days": 700,
        "z_threshold": 1.5,
        "hold_mult": 1.5,
        "hold_floor": 3,
        "hold_cap": 30,
        "min_eff_n": 50,
        "detrend_window": 252,
        "sweet_spot_full": (3.0, 30.0),
        "sweet_spot_partial_below": 3.0,
        "sweet_spot_partial_above": 60.0,
        "sweet_spot_mid_band": (30.0, 60.0),
        "hl_extension_stop_mult": 3.0,
    },
    "positional": {
        "residual_lookback": 252,
        "triple_gate_lookback": 120,
        "history_days": 1000,
        "z_threshold": 2.0,
        "hold_mult": 1.5,
        "hold_floor": 21,
        "hold_cap": 120,
        "min_eff_n": 80,
        "detrend_window": 252,
        "sweet_spot_full": (15.0, 90.0),
        "sweet_spot_partial_below": 10.0,
        "sweet_spot_partial_above": 180.0,
        "sweet_spot_mid_band": (90.0, 180.0),
        "hl_extension_stop_mult": 3.0,
    },
}

DEFAULT_MODE = "positional"


def mode_params(mode: Optional[str] = None) -> dict:
    """Return the parameter dict for a given mode, falling back to DEFAULT_MODE
    if `mode` is None or unknown. Always returns a copy (mutations don't leak)."""
    key = mode if mode in MODE_PARAMS else DEFAULT_MODE
    return dict(MODE_PARAMS[key])


def _resolve_mode(mode: Optional[str]) -> str:
    """Normalize a possibly-None mode string to one of the valid mode keys."""
    return mode if mode in MODE_PARAMS else DEFAULT_MODE


# =============================================================================
# Type definitions
# =============================================================================
@dataclass(frozen=True)
class PCAFit:
    """Static PCA fit on a Δ_CMC matrix.

    `loadings` shape: (n_components, n_tenors). Each row is a unit-norm vector.
    `feature_mean` is the mean Δ vector subtracted before SVD.
    """
    asof: date
    tenors_months: list
    loadings: np.ndarray
    eigenvalues: np.ndarray
    variance_ratio: np.ndarray
    feature_mean: np.ndarray
    n_obs: int
    fit_window: tuple    # (start_date, end_date)


@dataclass(frozen=True)
class SparsePCAFit:
    asof: date
    tenors_months: list
    loadings: np.ndarray
    n_nonzero_per_component: list
    lambdas: list


@dataclass(frozen=True)
class StructureCandidate:
    target_pc: int
    symbols: list
    weights: np.ndarray            # PV01-normalized
    pv01_sum: float
    residual_today_bp: Optional[float]
    residual_z: Optional[float]
    half_life_d: Optional[float]
    adf_pass: bool
    composite_score: Optional[float]
    eff_n: int
    gate_quality: str


# =============================================================================
# S1 — Constant-Maturity Curve (CMC) substrate
# =============================================================================
def _hand_rolled_pchip(tau_i: np.ndarray, Y_i: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Fritsch-Carlson PCHIP fallback when scipy is unavailable.

    Implements the monotone cubic Hermite scheme of Fritsch & Carlson (1980)
    for shape-preserving local interpolation. Returns interpolated Y at `target`.

    For τ outside [tau_i.min, tau_i.max], we extrapolate flat (clamp to nearest endpoint).
    """
    tau_i = np.asarray(tau_i, dtype=float)
    Y_i = np.asarray(Y_i, dtype=float)
    target = np.asarray(target, dtype=float)

    n = len(tau_i)
    if n == 0:
        return np.full_like(target, np.nan, dtype=float)
    if n == 1:
        return np.full_like(target, Y_i[0], dtype=float)

    # Sort by tau_i (PCHIP needs strictly increasing knots)
    order = np.argsort(tau_i)
    tau_i = tau_i[order]
    Y_i = Y_i[order]

    # De-dup any equal x values by averaging y
    keep = np.concatenate(([True], np.diff(tau_i) > 1e-9))
    tau_i = tau_i[keep]
    Y_i = Y_i[keep]
    n = len(tau_i)
    if n < 2:
        return np.full_like(target, Y_i[0] if n == 1 else np.nan, dtype=float)

    # Slopes between knots
    h = np.diff(tau_i)
    delta = np.diff(Y_i) / h

    # Tangents at interior knots — Fritsch-Carlson harmonic mean
    m = np.zeros(n)
    m[0] = delta[0]
    m[-1] = delta[-1]
    for k in range(1, n - 1):
        if delta[k - 1] * delta[k] <= 0:
            m[k] = 0.0
        else:
            w1 = 2 * h[k] + h[k - 1]
            w2 = h[k] + 2 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])

    # Hermite interpolation
    out = np.empty_like(target, dtype=float)
    for j, x in enumerate(target):
        if x <= tau_i[0]:
            out[j] = Y_i[0]
            continue
        if x >= tau_i[-1]:
            out[j] = Y_i[-1]
            continue
        # Locate segment
        k = int(np.searchsorted(tau_i, x) - 1)
        k = max(0, min(n - 2, k))
        t = (x - tau_i[k]) / h[k]
        h00 = (1 + 2 * t) * (1 - t) ** 2
        h10 = t * (1 - t) ** 2
        h01 = t ** 2 * (3 - 2 * t)
        h11 = t ** 2 * (t - 1)
        out[j] = (h00 * Y_i[k] + h10 * h[k] * m[k]
                  + h01 * Y_i[k + 1] + h11 * h[k] * m[k + 1])
    return out


def _pchip_curve(tau_i: np.ndarray, Y_i: np.ndarray, target: np.ndarray) -> np.ndarray:
    """PCHIP interpolation from `(tau_i, Y_i)` evaluated at `target` tenors.

    Tries scipy.interpolate.PchipInterpolator first; falls back to hand-rolled
    Fritsch-Carlson if scipy is unavailable.
    """
    tau_i = np.asarray(tau_i, dtype=float)
    Y_i = np.asarray(Y_i, dtype=float)
    target = np.asarray(target, dtype=float)
    if len(tau_i) < 2:
        return _hand_rolled_pchip(tau_i, Y_i, target)
    try:
        from scipy.interpolate import PchipInterpolator
        order = np.argsort(tau_i)
        tau_sorted = tau_i[order]
        Y_sorted = Y_i[order]
        # De-dup
        keep = np.concatenate(([True], np.diff(tau_sorted) > 1e-9))
        tau_sorted = tau_sorted[keep]
        Y_sorted = Y_sorted[keep]
        if len(tau_sorted) < 2:
            return _hand_rolled_pchip(tau_i, Y_i, target)
        interp = PchipInterpolator(tau_sorted, Y_sorted, extrapolate=False)
        out = interp(target)
        # Clamp NaN extrapolation to nearest-endpoint
        out = np.where(np.isnan(out) & (target <= tau_sorted[0]), Y_sorted[0], out)
        out = np.where(np.isnan(out) & (target >= tau_sorted[-1]), Y_sorted[-1], out)
        return out
    except ImportError:
        return _hand_rolled_pchip(tau_i, Y_i, target)


def _solve_front_pin_polynomial(R_contract_pct: float,
                                  realized_sofr_pct: list,
                                  n_unfixed_days: int) -> Optional[float]:
    """Solve the CME polynomial for the unfixed-tail rate r̂ (in %).

    Compounded settlement equation (SOFR daily compounding, ACT/360 with d_j=1
    per calendar day, weekend rates inheriting Friday's fixing — typical
    approximation accurate to ~0.01 bp at normal rate levels):

        R_contract = (A · (1 + r̂/360)^N_unfixed − 1) · 360 / N_q

      where  A = ∏_{j fixed} (1 + r_j / 360),  N_q = total calendar days,
             N_unfixed = N_q − len(fixed_days).

    Solved analytically:

        r̂ = 360 · ( ((1 + R · N_q / 360) / A) ^ (1/N_unfixed) − 1 )

    Returns None if inputs are invalid (e.g. all days fixed, A non-positive).
    """
    if n_unfixed_days <= 0 or R_contract_pct is None:
        return None
    fixed = [r / 100.0 for r in realized_sofr_pct
             if r is not None and not pd.isna(r)]
    n_fixed = len(fixed)
    n_total = n_fixed + int(n_unfixed_days)
    if n_total <= 0:
        return None
    R = R_contract_pct / 100.0
    A = 1.0
    for r in fixed:
        A *= (1.0 + r / 360.0)
    if A <= 0 or not np.isfinite(A):
        return None
    rhs = (1.0 + R * n_total / 360.0) / A
    if rhs <= 0:
        return None
    try:
        ratio_pow = rhs ** (1.0 / int(n_unfixed_days))
    except Exception:
        return None
    r_hat = 360.0 * (ratio_pow - 1.0) * 100.0   # back to %
    if not np.isfinite(r_hat):
        return None
    return float(r_hat)


def _outright_midpoint_months_from_asof(symbol: str, asof: date) -> Optional[float]:
    """Months from `asof` to the midpoint of the contract's 3M reference quarter.

    Uses `lib.fomc.reference_period`. For SRAH26 the reference quarter is
    [3rd Wed Mar 2026, 3rd Wed Jun 2026] and midpoint is mid-Apr 2026.
    """
    rp = reference_period(symbol)
    if rp is None:
        return None
    start, end = rp
    mid_days = (start - asof).days + ((end - start).days / 2.0)
    return float(mid_days) / 30.4375


def _outright_yield_bp(price: Optional[float]) -> Optional[float]:
    """Implied 3M-forward yield in bp = (100 - price) * 100."""
    if price is None or pd.isna(price):
        return None
    return float(100.0 - price) * 100.0


def build_cmc_panel(asof_dates: list,
                     wide_close_outrights: pd.DataFrame,
                     outright_symbols: list,
                     target_tenors_months: list = None,
                     sofr_panel: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Construct the CMC panel for given `asof_dates` on the requested τ-grid.

    Inputs:
      asof_dates              — list of dates to build CMC for (one row per).
      wide_close_outrights    — DataFrame index=date, columns=outright symbol,
                                values=close price.
      outright_symbols        — ordered list of live SRA outright symbols.
      target_tenors_months    — list of τ in months to interpolate at.
                                Defaults to 3..60 step 3 (20 nodes).
      sofr_panel              — optional DataFrame with daily SOFR fixings
                                (column 'sofr', %). Used for the front-month
                                analytical accrual. If None, the front pin uses
                                a time-fraction blend with the front continuous
                                approximation (front outright price proxy).

    Output:
      DataFrame index=asof, columns=tenor_months, values=fwd_yield_bp.

    Verification gates (gameplan §A12):
      i.   At every input τ_i, CMC(τ_i) reproduces input Y_i to ≤ 0.01 bp.
      ii.  No monotonicity enforcement.
      iii. Pre-2023-08-01 rows are dropped (caller filters; this function does
           not — it builds whatever asof_dates it is given).
    """
    if target_tenors_months is None:
        target_tenors_months = list(DEFAULT_TENOR_GRID_MONTHS)
    target = np.array(target_tenors_months, dtype=float)
    if wide_close_outrights is None or wide_close_outrights.empty:
        return pd.DataFrame(columns=target_tenors_months)

    # Pre-compute reference periods + midpoint-month-from-each-symbol's-anchor.
    rp_cache = {sym: reference_period(sym) for sym in outright_symbols}

    rows = []
    for asof in asof_dates:
        ts = pd.Timestamp(asof)
        try:
            row = wide_close_outrights.loc[wide_close_outrights.index.normalize() == ts.normalize()]
        except Exception:
            continue
        if row.empty:
            continue
        prices = row.iloc[0]

        tau_i_list = []
        Y_i_list = []
        front_sym_idx = None
        for k, sym in enumerate(outright_symbols):
            rp = rp_cache.get(sym)
            if rp is None:
                continue
            start, end = rp
            mid_days = (start - asof).days + ((end - start).days / 2.0)
            mid_months = float(mid_days) / 30.4375
            price = prices.get(sym)
            Y_bp = _outright_yield_bp(price)
            if Y_bp is None:
                continue
            # Identify the partially-fixed front month: its reference quarter
            # has already started (start <= asof) and not finished (end > asof).
            if start <= asof < end:
                front_sym_idx = k
                # Apply analytical front-pin if SOFR fixings available
                if sofr_panel is not None and not sofr_panel.empty:
                    fixed_dates = pd.bdate_range(start, asof - timedelta(days=1))
                    realized_sofr = []
                    for fd in fixed_dates:
                        fts = pd.Timestamp(fd)
                        try:
                            sofr_row = sofr_panel.loc[sofr_panel.index <= fts]
                            if not sofr_row.empty and "sofr" in sofr_panel.columns:
                                v = sofr_row["sofr"].iloc[-1]
                                if pd.notna(v):
                                    realized_sofr.append(float(v))
                        except Exception:
                            continue
                    n_q_total = (end - start).days
                    n_fixed = len(fixed_dates)
                    n_unfixed = max(1, n_q_total - n_fixed)
                    R_contract_pct = 100.0 - float(price)
                    r_hat_pct = _solve_front_pin_polynomial(
                        R_contract_pct, realized_sofr, n_unfixed)
                    if r_hat_pct is not None:
                        # Time-fraction blend r̂ (forward) with the implied
                        # contract yield so the pin is in bp space.
                        w = float(n_unfixed) / float(n_q_total) if n_q_total > 0 else 1.0
                        # The forward portion (r̂) lives at the unfixed-tail's
                        # midpoint, not the contract midpoint. Adjust τ accordingly.
                        unfixed_start = asof
                        unfixed_mid_days = (unfixed_start - asof).days + (n_unfixed / 2.0)
                        mid_months = float(unfixed_mid_days) / 30.4375
                        # Convert r̂ % into bp.
                        Y_bp = float(r_hat_pct) * 100.0
                        # Effective weight goes to r̂ via the τ shift, not blended Y.
                        # (Pure r̂ pinned at its midpoint — cleanest.)
            tau_i_list.append(mid_months)
            Y_i_list.append(Y_bp)

        if len(tau_i_list) < 2:
            continue
        Y_curve = _pchip_curve(np.array(tau_i_list), np.array(Y_i_list), target)
        rec = {"asof": pd.Timestamp(asof)}
        for tau, y in zip(target_tenors_months, Y_curve):
            rec[int(tau)] = float(y) if np.isfinite(y) else None
        rows.append(rec)

    if not rows:
        return pd.DataFrame(columns=target_tenors_months)
    df = pd.DataFrame(rows).set_index("asof").sort_index()
    return df


def cmc_print_quality_alert(cmc_panel: pd.DataFrame,
                              threshold_bp: float = CMC_PRINT_QUALITY_BP) -> list:
    """Days where ‖CMC_t − CMC_{t-1}‖∞ > threshold (default 50 bp).

    Returns sorted list of `pd.Timestamp` flagged days.
    """
    if cmc_panel is None or cmc_panel.empty or len(cmc_panel) < 2:
        return []
    diffs = cmc_panel.diff().abs()
    if diffs.empty:
        return []
    max_per_day = diffs.max(axis=1)
    flagged = list(max_per_day[max_per_day > float(threshold_bp)].index)
    return sorted(flagged)


# =============================================================================
# S2 — Static PCA on full sample
# =============================================================================
def _normalize_pc_signs(loadings: np.ndarray, tenors_months: list) -> np.ndarray:
    """Make sign convention canonical so PCs are stable across refits.

    PC1 (level)     — sum of loadings positive.
    PC2 (slope)     — loading at longest tenor > loading at shortest tenor.
    PC3 (curvature) — loading near belly (~24M) > loading at shortest and longest.
    """
    L = loadings.copy()
    n_tenors = L.shape[1]
    tenors = np.asarray(tenors_months, dtype=float)

    # PC1 — sum positive
    if L.shape[0] >= 1:
        if L[0].sum() < 0:
            L[0] *= -1.0

    # PC2 — back > front
    if L.shape[0] >= 2:
        if L[1, -1] < L[1, 0]:
            L[1] *= -1.0

    # PC3 — belly positive vs wings
    if L.shape[0] >= 3:
        # Find tenor closest to 24M
        belly_idx = int(np.argmin(np.abs(tenors - 24.0)))
        wings_avg = 0.5 * (L[2, 0] + L[2, -1])
        if L[2, belly_idx] < wings_avg:
            L[2] *= -1.0

    return L


def fit_pca_static(cmc_panel: pd.DataFrame, *, n_components: int = 3) -> Optional[PCAFit]:
    """Static PCA on Δ_CMC for the full input sample. SVD on centred matrix.

    Returns None if sample is too small.
    """
    if cmc_panel is None or cmc_panel.empty or len(cmc_panel) < 30:
        return None
    delta = cmc_panel.diff().dropna()
    if delta.empty or len(delta) < 20:
        return None
    X = delta.values.astype(float)
    feature_mean = X.mean(axis=0)
    X_c = X - feature_mean
    try:
        U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    except Exception:
        return None
    n_obs = X_c.shape[0]
    eigenvalues = (S ** 2) / max(1, n_obs - 1)
    total_var = eigenvalues.sum()
    if total_var <= 0:
        return None
    variance_ratio = eigenvalues / total_var
    L = Vt[:n_components].copy()
    tenors = list(cmc_panel.columns)
    L = _normalize_pc_signs(L, tenors)
    return PCAFit(
        asof=cmc_panel.index[-1].date() if len(cmc_panel) else date.today(),
        tenors_months=tenors,
        loadings=L,
        eigenvalues=eigenvalues[:n_components].copy(),
        variance_ratio=variance_ratio[:n_components].copy(),
        feature_mean=feature_mean.copy(),
        n_obs=int(n_obs),
        fit_window=(cmc_panel.index[0].date(), cmc_panel.index[-1].date()),
    )


# =============================================================================
# S3 — Sparse PCA via L1 soft-thresholding + Gram-Schmidt
# =============================================================================
def _soft_threshold_per_pc(loading_vec: np.ndarray,
                            target_nonzeros: int) -> tuple:
    """Soft-threshold loading_vec so it has ≤ target_nonzeros nonzeros.

    Algorithm: binary-search λ such that |L| > λ count is ≈ target.
    Returns (sparse_vec, lambda_used).
    """
    abs_v = np.abs(loading_vec)
    if len(abs_v) == 0:
        return loading_vec.copy(), 0.0
    # Choose lambda that keeps the top-`target_nonzeros` magnitudes.
    if target_nonzeros >= len(abs_v):
        return loading_vec.copy(), 0.0
    sorted_abs = np.sort(abs_v)[::-1]
    # λ = midpoint between target_nonzeros-th and (target_nonzeros+1)-th magnitudes
    lam = float((sorted_abs[target_nonzeros - 1] + sorted_abs[target_nonzeros]) / 2.0)
    sparse = np.sign(loading_vec) * np.maximum(0.0, abs_v - lam)
    return sparse, lam


def _gram_schmidt_reorthonormalize(loadings: np.ndarray) -> np.ndarray:
    """Re-orthonormalise rows of `loadings` via Gram-Schmidt.

    Preserves PC1 row, projects out PC1 from PC2, projects out {PC1, PC2} from PC3.
    Each row is L2-normalised at the end. Rows that become near-zero (collinear
    with predecessors) are restored to their original direction.
    """
    L = loadings.astype(float).copy()
    n_pc, n_tenors = L.shape
    out = np.zeros_like(L)
    for i in range(n_pc):
        v = L[i].copy()
        for j in range(i):
            proj = float(np.dot(v, out[j]))
            v = v - proj * out[j]
        norm = float(np.linalg.norm(v))
        if norm < 1e-9:
            # Collinear — restore original and re-normalise
            v = L[i].copy()
            norm = float(np.linalg.norm(v))
            if norm < 1e-9:
                v = np.zeros_like(v)
                v[i % n_tenors] = 1.0
                norm = 1.0
        out[i] = v / norm
    return out


def fit_sparse_pca(cmc_panel: pd.DataFrame, dense_fit: PCAFit, *,
                    target_nonzeros: int = 10) -> Optional[SparsePCAFit]:
    """Sparse PCA companion. Soft-threshold dense loadings, then GS re-orthonormalise.

    Returns None if dense_fit is None.
    """
    if dense_fit is None:
        return None
    L_dense = dense_fit.loadings.copy()
    n_pc = L_dense.shape[0]
    sparse_rows = []
    lambdas = []
    for k in range(n_pc):
        sp, lam = _soft_threshold_per_pc(L_dense[k], target_nonzeros)
        sparse_rows.append(sp)
        lambdas.append(lam)
    sparse_L = np.vstack(sparse_rows)
    sparse_L = _gram_schmidt_reorthonormalize(sparse_L)
    # Sign-align with dense (correlation rule)
    for k in range(n_pc):
        if float(np.dot(sparse_L[k], L_dense[k])) < 0:
            sparse_L[k] *= -1.0
    n_nonzero = [int((np.abs(row) > 1e-9).sum()) for row in sparse_L]
    return SparsePCAFit(
        asof=dense_fit.asof,
        tenors_months=list(dense_fit.tenors_months),
        loadings=sparse_L,
        n_nonzero_per_component=n_nonzero,
        lambdas=lambdas,
    )


# =============================================================================
# S4 — Rolling PCA refit + sign alignment
# =============================================================================
def sign_align(loadings_new: np.ndarray, loadings_prior: np.ndarray) -> np.ndarray:
    """Flip rows of `loadings_new` to maximise correlation with `loadings_prior`.

    Per-row sign rule: if dot(new, prior) < 0, flip. Returns aligned matrix.
    """
    L = loadings_new.copy()
    if loadings_prior is None or loadings_prior.shape != L.shape:
        return L
    for k in range(L.shape[0]):
        if float(np.dot(L[k], loadings_prior[k])) < 0:
            L[k] *= -1.0
    return L


def fit_pca_rolling(cmc_panel: pd.DataFrame, *,
                     window_days: int = 252,
                     step_days: int = 5,
                     n_components: int = 3) -> dict:
    """Rolling PCA refits at `step_days` cadence over `window_days` window.

    Returns dict {refit_date: PCAFit}, with each fit sign-aligned to its prior.

    Days before the first full window have no fit. Subsequent days inherit the
    most recent prior fit until the next refit lands (no look-ahead).
    """
    out = {}
    if cmc_panel is None or cmc_panel.empty:
        return out
    n = len(cmc_panel)
    if n < window_days + 1:
        # Single static fit covering everything available
        single = fit_pca_static(cmc_panel, n_components=n_components)
        if single is not None:
            out[cmc_panel.index[-1].date()] = single
        return out
    prior_loadings = None
    refit_indices = list(range(window_days, n, step_days))
    if refit_indices and refit_indices[-1] != n - 1:
        refit_indices.append(n - 1)
    for idx in refit_indices:
        sub = cmc_panel.iloc[idx - window_days: idx + 1]
        fit = fit_pca_static(sub, n_components=n_components)
        if fit is None:
            continue
        if prior_loadings is not None:
            new_L = sign_align(fit.loadings, prior_loadings)
            fit = PCAFit(
                asof=fit.asof, tenors_months=fit.tenors_months,
                loadings=new_L, eigenvalues=fit.eigenvalues,
                variance_ratio=fit.variance_ratio, feature_mean=fit.feature_mean,
                n_obs=fit.n_obs, fit_window=fit.fit_window,
            )
        prior_loadings = fit.loadings.copy()
        out[fit.asof] = fit
    return out


def _pick_fit_for_date(rolling_fits: dict, target_date: date) -> Optional[PCAFit]:
    """Return the most recent PCAFit whose asof <= target_date (no look-ahead)."""
    if not rolling_fits:
        return None
    candidates = [d for d in rolling_fits if d <= target_date]
    if not candidates:
        # Fall back to earliest available fit if no prior exists
        first = min(rolling_fits.keys())
        return rolling_fits[first]
    return rolling_fits[max(candidates)]


def project_to_pcs(cmc_panel: pd.DataFrame, rolling_fits: dict) -> pd.DataFrame:
    """Project each row of cmc_panel to PC space using the appropriate prior fit.

    Returns DataFrame index=asof, columns=['PC1','PC2','PC3'] (per fit's
    n_components). Rows where Δ_CMC can't be computed (first row) are NaN.
    """
    if cmc_panel is None or cmc_panel.empty or not rolling_fits:
        return pd.DataFrame(columns=["PC1", "PC2", "PC3"])
    delta = cmc_panel.diff()
    rows = []
    for ts, drow in delta.iterrows():
        if drow.isna().all():
            rows.append({"asof": ts, "PC1": np.nan, "PC2": np.nan, "PC3": np.nan})
            continue
        target_date = ts.date()
        fit = _pick_fit_for_date(rolling_fits, target_date)
        if fit is None:
            rows.append({"asof": ts, "PC1": np.nan, "PC2": np.nan, "PC3": np.nan})
            continue
        # Align tenor columns to fit's tenors order
        try:
            x = drow[fit.tenors_months].values.astype(float)
        except KeyError:
            x = drow.reindex(fit.tenors_months).values.astype(float)
        x_c = x - fit.feature_mean
        if np.any(np.isnan(x_c)):
            rows.append({"asof": ts, "PC1": np.nan, "PC2": np.nan, "PC3": np.nan})
            continue
        proj = fit.loadings @ x_c
        rec = {"asof": ts}
        for k in range(min(3, len(proj))):
            rec[f"PC{k + 1}"] = float(proj[k])
        rows.append(rec)
    df = pd.DataFrame(rows).set_index("asof").sort_index()
    return df


# =============================================================================
# S5 — PC time-series diagnostics
# =============================================================================
def pc_zscore_multi(pc_series: pd.Series, asof: date,
                     lookbacks: tuple = DEFAULT_PC_LOOKBACKS) -> dict:
    """Z-score for each lookback, asof excluded."""
    return {int(n): zscore_value(pc_series, asof, int(n)) for n in lookbacks}


def pc_percentile_rank_multi(pc_series: pd.Series, asof: date,
                               lookbacks: tuple = DEFAULT_PC_LOOKBACKS) -> dict:
    return {int(n): percentile_rank_value(pc_series, asof, int(n)) for n in lookbacks}


def pc_diagnostics(pc_series: pd.Series, asof: date,
                    lookback: int = 90) -> dict:
    """Assemble ADF / Hurst / OU / composite for a single PC series."""
    hl = ou_half_life(pc_series, asof, lookback)
    h = hurst_exponent(pc_series, asof, lookback)
    h_lab = hurst_label(h)
    adf = adf_test(pc_series, asof, lookback, n_lags=1)
    z = zscore_value(pc_series, asof, lookback)
    composite = composite_reversion_score(z, hl, h, bool(adf.get("reject_5pct", False)))
    return {
        "z": z,
        "ou_half_life": hl,
        "hurst": h,
        "hurst_label": h_lab,
        "adf_tstat": adf.get("tstat"),
        "adf_pvalue": adf.get("pvalue"),
        "adf_reject_5pct": bool(adf.get("reject_5pct", False)),
        "adf_n_obs": adf.get("n_obs", 0),
        "composite": composite,
    }


def anchor_metric(cmc_panel: pd.DataFrame) -> pd.Series:
    """Anchor(t) = CMC(24M) − CMC(12M), per gameplan §A1.4."""
    if cmc_panel is None or cmc_panel.empty:
        return pd.Series(dtype=float)
    cols = list(cmc_panel.columns)
    # Find columns closest to 24 and 12 months
    cols_int = [int(c) for c in cols if isinstance(c, (int, np.integer))]
    if not cols_int:
        return pd.Series(dtype=float)
    c24 = min(cols_int, key=lambda x: abs(x - 24))
    c12 = min(cols_int, key=lambda x: abs(x - 12))
    return cmc_panel[c24] - cmc_panel[c12]


def _by_lookback_z_dict(pc_series: pd.Series, asof: date,
                         lookbacks: tuple) -> dict:
    """Shape z-dict to match the 8-pattern classifier's expected input."""
    return {int(n): {"z": zscore_value(pc_series, asof, int(n))} for n in lookbacks}


def _proximity_compatible_dict(pc_series: pd.Series, asof: date,
                                  lookbacks: tuple) -> dict:
    """Adapt PC z-scores into the proximity-classifier shape.

    The proximity classifier expects per-lookback dicts with `nearest_dist_atr`
    and `nearest_extreme`. We translate |z| into a synthetic ATR-distance
    (the smaller, the more elevated) and z-sign into HIGH/LOW.
    """
    out = {}
    for n in lookbacks:
        z = zscore_value(pc_series, asof, int(n))
        if z is None or pd.isna(z):
            out[int(n)] = {"nearest_dist_atr": None, "nearest_extreme": None}
            continue
        # |z|=2 ↔ proximity AT (≤ NEAR_ATR threshold).
        # Map: dist_atr = 1 / (|z| + 0.001) so |z|≥2 → dist≈0.5 (NEAR), |z|≥4 → AT.
        dist_atr = 1.0 / (abs(z) + 0.001)
        side = "HIGH" if z > 0 else "LOW"
        out[int(n)] = {"nearest_dist_atr": float(dist_atr),
                       "nearest_extreme": side}
    return out


def pc_confluence_pattern(pc_series: pd.Series, asof: date,
                            lookbacks: tuple = (5, 15, 30, 60, 90)) -> str:
    """8-pattern label across multi-lookback z-scores via proximity classifier."""
    by_lb = _proximity_compatible_dict(pc_series, asof, lookbacks)
    return classify_proximity_pattern(by_lb, list(lookbacks))


def cross_pc_corr_rolling(pc_panel: pd.DataFrame, *, window: int = 60) -> pd.DataFrame:
    """Rolling pairwise correlations for PC1, PC2, PC3."""
    if pc_panel is None or pc_panel.empty:
        return pd.DataFrame(columns=["corr_PC1_PC2", "corr_PC1_PC3", "corr_PC2_PC3"])
    out = pd.DataFrame(index=pc_panel.index)
    cols = [c for c in ("PC1", "PC2", "PC3") if c in pc_panel.columns]
    if len(cols) < 2:
        return out
    if "PC1" in cols and "PC2" in cols:
        out["corr_PC1_PC2"] = pc_panel["PC1"].rolling(window).corr(pc_panel["PC2"])
    if "PC1" in cols and "PC3" in cols:
        out["corr_PC1_PC3"] = pc_panel["PC1"].rolling(window).corr(pc_panel["PC3"])
    if "PC2" in cols and "PC3" in cols:
        out["corr_PC2_PC3"] = pc_panel["PC2"].rolling(window).corr(pc_panel["PC3"])
    return out


# =============================================================================
# S6 — Today's Δ_CMC decomposition
# =============================================================================
def decompose_delta(cmc_panel: pd.DataFrame, pca_fit: PCAFit,
                     pc_panel: pd.DataFrame, asof: date,
                     rolling_fits: Optional[dict] = None) -> dict:
    """Decompose Δ_CMC[asof] into per-PC contributions + residual per tenor.

    Since PCA is fit on Δ_CMC directly (not on level), each row of pc_panel is the
    projection of THAT day's Δ_CMC onto the loading rows. So:

        Δ_CMC[t, τ]  =  Σ_k PC_k[t] · L_k[τ]  +  mean[τ]  +  residual[τ]

    where `mean` is the training-sample feature_mean used in the fit. We therefore
    decompose today's curve change directly via PC_k[today] · L_k, with the mean
    captured separately and added to the residual term for the "what's left after
    the 3-PC factor model" reading.

    Returns:
      {
        "tenors_months": list,
        "delta_cmc_bp": list (per tenor),
        "pc1_contrib_bp": list, "pc2_contrib_bp": list, "pc3_contrib_bp": list,
        "residual_bp": list,
        "shares": {"pct_pc1", "pct_pc2", "pct_pc3", "pct_residual",
                   "total_l1_norm_bp", "total_change_bp"},
      }
    """
    out = {
        "tenors_months": list(cmc_panel.columns) if cmc_panel is not None else [],
        "delta_cmc_bp": [], "pc1_contrib_bp": [], "pc2_contrib_bp": [],
        "pc3_contrib_bp": [], "residual_bp": [],
        "shares": {"pct_pc1": None, "pct_pc2": None, "pct_pc3": None,
                   "pct_residual": None, "total_l1_norm_bp": None,
                   "total_change_bp": None},
    }
    if cmc_panel is None or cmc_panel.empty or pca_fit is None or pc_panel is None or pc_panel.empty:
        return out
    ts = pd.Timestamp(asof)
    if ts not in cmc_panel.index:
        candidates = cmc_panel.index[cmc_panel.index <= ts]
        if len(candidates) == 0:
            return out
        ts = candidates[-1]
    pos = cmc_panel.index.get_loc(ts)
    if pos == 0:
        return out
    # Use the rolling fit applicable to this date (preferred for refit consistency)
    fit = pca_fit
    if rolling_fits:
        picked = _pick_fit_for_date(rolling_fits, ts.date())
        if picked is not None:
            fit = picked
    today_row = cmc_panel.iloc[pos]
    prior_row = cmc_panel.iloc[pos - 1]
    delta = (today_row - prior_row).reindex(fit.tenors_months)
    if delta.isna().all():
        return out
    if ts not in pc_panel.index:
        return out
    pcs_today = pc_panel.loc[ts]
    L = fit.loadings
    n_pc = L.shape[0]
    pc_contribs = []
    for k in range(min(3, n_pc)):
        col = f"PC{k + 1}"
        pck = float(pcs_today.get(col, 0.0))
        if pd.isna(pck):
            pck = 0.0
        pc_contribs.append(pck * L[k])
    while len(pc_contribs) < 3:
        pc_contribs.append(np.zeros_like(L[0]))
    delta_arr = delta.values.astype(float)
    # Residual = Δ_CMC − Σ_k PC_k·L_k − feature_mean
    residual = delta_arr - sum(pc_contribs) - fit.feature_mean

    out["tenors_months"] = list(fit.tenors_months)
    out["delta_cmc_bp"] = [float(x) if np.isfinite(x) else None for x in delta_arr]
    out["pc1_contrib_bp"] = [float(x) if np.isfinite(x) else None for x in pc_contribs[0]]
    out["pc2_contrib_bp"] = [float(x) if np.isfinite(x) else None for x in pc_contribs[1]]
    out["pc3_contrib_bp"] = [float(x) if np.isfinite(x) else None for x in pc_contribs[2]]
    out["residual_bp"] = [float(x) if np.isfinite(x) else None for x in residual]

    # Aggregate shares — L1-normalised (matches gameplan §A1)
    abs_delta_total = float(np.nansum(np.abs(delta_arr)))
    if abs_delta_total > 1e-9:
        out["shares"]["pct_pc1"] = float(np.nansum(np.abs(pc_contribs[0]))) / abs_delta_total
        out["shares"]["pct_pc2"] = float(np.nansum(np.abs(pc_contribs[1]))) / abs_delta_total
        out["shares"]["pct_pc3"] = float(np.nansum(np.abs(pc_contribs[2]))) / abs_delta_total
        out["shares"]["pct_residual"] = float(np.nansum(np.abs(residual))) / abs_delta_total
    out["shares"]["total_l1_norm_bp"] = abs_delta_total
    out["shares"]["total_change_bp"] = float(np.nansum(delta_arr))
    return out


# =============================================================================
# S7 — PCA-derived structures (PC1/PC2/PC3 isolated)
# =============================================================================
def _outright_tenor_months(symbol: str, asof: date) -> Optional[float]:
    rp = reference_period(symbol)
    if rp is None:
        return None
    start, end = rp
    mid_days = (start - asof).days + ((end - start).days / 2.0)
    return float(mid_days) / 30.4375


def _quarterly_only(symbols: list) -> list:
    return [s for s in symbols if is_quarterly(s)]


def enumerate_pc3_fly_triples(live_symbols: list, asof: date,
                                tenor_pattern_window_months: float = 1.5) -> list:
    """All quarterly outright triples (a,b,c) with approximate equal spacing.

    Spacing buckets ≈ 3M, 6M, 9M, 12M between adjacent legs. Returns list of
    (sym_a, sym_b, sym_c) triples ordered by ascending leg-a tenor.
    """
    qsyms = _quarterly_only(live_symbols)
    if len(qsyms) < 3:
        return []
    tenors = {s: _outright_tenor_months(s, asof) for s in qsyms}
    qsyms = [s for s in qsyms if tenors[s] is not None]
    qsyms.sort(key=lambda s: tenors[s])
    out = []
    target_spacings = [3.0, 6.0, 9.0, 12.0]
    for i, sa in enumerate(qsyms):
        ta = tenors[sa]
        for j in range(i + 1, len(qsyms)):
            sb = qsyms[j]
            tb = tenors[sb]
            if tb - ta < 2.0:
                continue
            for k in range(j + 1, len(qsyms)):
                sc = qsyms[k]
                tc = tenors[sc]
                spacing_ab = tb - ta
                spacing_bc = tc - tb
                # Approximate equal spacing required for a clean PC3 fly
                if abs(spacing_ab - spacing_bc) > tenor_pattern_window_months:
                    continue
                # Check if spacing matches one of the target patterns
                avg_spacing = 0.5 * (spacing_ab + spacing_bc)
                matches = any(
                    abs(avg_spacing - target) <= tenor_pattern_window_months
                    for target in target_spacings
                )
                if not matches:
                    continue
                out.append((sa, sb, sc))
    return out


def enumerate_pc2_spread_triples(live_symbols: list, asof: date) -> list:
    """Asymmetric quarterly triples (b−a ≠ c−b) suitable for PC2-isolated baskets."""
    qsyms = _quarterly_only(live_symbols)
    if len(qsyms) < 3:
        return []
    tenors = {s: _outright_tenor_months(s, asof) for s in qsyms}
    qsyms = [s for s in qsyms if tenors[s] is not None]
    qsyms.sort(key=lambda s: tenors[s])
    out = []
    for i, sa in enumerate(qsyms):
        for j in range(i + 1, len(qsyms)):
            sb = qsyms[j]
            for k in range(j + 1, len(qsyms)):
                sc = qsyms[k]
                spacing_ab = tenors[sb] - tenors[sa]
                spacing_bc = tenors[sc] - tenors[sb]
                # Strictly asymmetric (otherwise it's a PC3-fly candidate)
                if abs(spacing_ab - spacing_bc) < 1.5:
                    continue
                if spacing_ab < 2 or spacing_bc < 2:
                    continue
                out.append((sa, sb, sc))
                if len(out) >= 80:    # cap to keep enumeration tractable
                    return out
    return out


def enumerate_pc1_synthetics(live_symbols: list, asof: date) -> list:
    """4-leg PV01-weighted level-only synthetic baskets.

    Returns list of 4-element lists chosen across the front-belly span (≤24M).
    """
    qsyms = _quarterly_only(live_symbols)
    if len(qsyms) < 4:
        return []
    tenors = {s: _outright_tenor_months(s, asof) for s in qsyms}
    qsyms = [s for s in qsyms if tenors[s] is not None]
    qsyms.sort(key=lambda s: tenors[s])
    out = []
    for start in range(0, max(1, len(qsyms) - 3), 2):
        chunk = qsyms[start: start + 4]
        if len(chunk) == 4:
            out.append(chunk)
    return out


def solve_isolated_pc_weights(loadings_subset: np.ndarray,
                                target_pc: int,
                                pv01_legs: np.ndarray) -> Optional[np.ndarray]:
    """Solve for weights `w` such that target PC is loaded and others are zero.

    Inputs:
      loadings_subset — shape (n_pc, n_legs).  loadings_subset[k, i] = PC_k load on leg i.
      target_pc       — 1, 2, or 3 (1-indexed).
      pv01_legs       — length n_legs PV01 of each leg (in same units, e.g. bp/contract).

    For 3 legs: closed-form via cross-product of the two "other" PC loading rows
    (yields a unique direction up to scale, then PV01-normalised so Σ PV01_i·w_i = 0
    for target_pc ∈ {2, 3}; for target_pc = 1 we drop PV01-neutrality and unit-PV01-norm).

    For 4+ legs: least-squares with constraints in the augmented system.

    Returns weight vector (np.ndarray, length n_legs) or None on failure.
    """
    L = np.asarray(loadings_subset, dtype=float)
    pv01 = np.asarray(pv01_legs, dtype=float)
    if L.ndim != 2 or len(pv01) != L.shape[1]:
        return None
    n_pc, n_legs = L.shape
    other_pcs = [k for k in range(n_pc) if k + 1 != target_pc]

    if n_legs == 3 and len(other_pcs) >= 2:
        # 3-leg cross product zeros the two "other" PC rows
        row_a = L[other_pcs[0]]
        row_b = L[other_pcs[1]]
        w = np.cross(row_a, row_b)
        if not np.isfinite(w).all() or float(np.linalg.norm(w)) < 1e-12:
            return None
        # PV01-neutralise (only meaningful for target_pc != 1)
        if target_pc != 1:
            pv01_dot = float(np.dot(pv01, w))
            if abs(pv01_dot) > 1e-9:
                # w is in PC null-space of {other_pcs}; we cannot scale to PV01-zero
                # without leaving the null-space. Project onto PV01-zero subspace instead.
                # In 3D with 2 PC-null constraints, w is unique up to scalar — so
                # PV01-neutrality may not be exactly achievable. Best we can do:
                # scale to unit |w| and report PV01 in the candidate.
                pass
        # Normalise to unit PV01 sum-of-abs (so weights are in "per unit PV01" units)
        scale = float(np.sum(np.abs(pv01 * w)))
        if scale < 1e-12:
            return None
        return w / scale

    # Multi-leg least-squares with constraints
    rows = []
    rhs = []
    for k in other_pcs:
        rows.append(L[k])
        rhs.append(0.0)
    rows.append(pv01)
    rhs.append(0.0)
    # Add unit-norm constraint on target PC component (loosely): require
    # L[target_pc-1] · w = 1 to break the scalar ambiguity.
    rows.append(L[target_pc - 1])
    rhs.append(1.0)
    A = np.vstack(rows)
    b = np.asarray(rhs, dtype=float)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return None
    if not np.isfinite(sol).all():
        return None
    return sol


def structure_residual_series(close_panel: pd.DataFrame, symbols: list,
                                weights: np.ndarray,
                                bp_multipliers: list) -> pd.Series:
    """Time series of residual = Σ_i w_i · Y_i where Y_i is in bp.

    `close_panel` is wide DataFrame (date × symbol) of stored prices.
    `bp_multipliers` aligns to `symbols`: 100× for outrights, 1× for stored-bp.
    For outrights, Y_i_bp = (100 - close_i) × 100 = (100 - close_i) × bp_multiplier_outright.
    For traded spreads/flies stored as bp directly, Y_i_bp = close_i × 1.
    """
    if close_panel is None or close_panel.empty or not symbols:
        return pd.Series(dtype=float)
    aligned = close_panel.reindex(columns=symbols)
    mults = np.asarray(bp_multipliers, dtype=float).reshape(1, -1)
    # If outright (mult=100), implied yield in bp = (100 - close) * 100
    # If stored-bp (mult=1), the close is already in bp-space relative to a basket.
    # For PCA structure on outrights only, all legs are outrights. Use the
    # outright convention.
    yield_bp = (100.0 - aligned.values) * mults
    w = np.asarray(weights, dtype=float).reshape(1, -1)
    res_arr = (yield_bp * w).sum(axis=1)
    return pd.Series(res_arr, index=aligned.index, name="residual_bp").dropna()


def score_structure(residual_series: pd.Series, asof: date, *,
                     lookback: Optional[int] = None,
                     mode: Optional[str] = None,
                     cross_pc_corr: Optional[float] = None,
                     target_pc: int = 3,
                     symbols: Optional[list] = None,
                     weights: Optional[np.ndarray] = None,
                     pv01_sum: float = 0.0) -> StructureCandidate:
    """Score a structure's residual series with the gates from gameplan §A11.

    Mode controls residual_lookback, triple_gate_lookback, and the HL window
    used as a sanity-band on the gate verdict.
    """
    if residual_series is None or residual_series.empty:
        return StructureCandidate(
            target_pc=target_pc,
            symbols=list(symbols) if symbols is not None else [],
            weights=np.asarray(weights, dtype=float)
                    if weights is not None else np.zeros(0, dtype=float),
            pv01_sum=float(pv01_sum),
            residual_today_bp=None, residual_z=None, half_life_d=None,
            adf_pass=False, composite_score=None, eff_n=0,
            gate_quality="low_n",
        )
    mp = mode_params(mode)
    lookback = int(lookback) if lookback is not None else int(mp["residual_lookback"])
    gate_lb = int(mp["triple_gate_lookback"])
    # HL sanity band — matches mode's intended hold horizon
    hl_lo, hl_hi = float(mp["sweet_spot_full"][0]), float(mp["sweet_spot_partial_above"])
    ts = pd.Timestamp(asof)
    history = residual_series.loc[residual_series.index < ts].dropna()
    eff_n = int(len(history))
    today = None
    try:
        today = float(residual_series.loc[residual_series.index.date == asof].iloc[0])
    except Exception:
        today = None
    z = zscore_value(residual_series, asof, lookback)
    hl = ou_half_life(residual_series, asof, lookback)
    h = hurst_exponent(residual_series, asof, lookback)
    # Phase A.3 — A6 triple-gate (ADF + KPSS + VR) per gameplan §A6
    from lib.mean_reversion import triple_stationarity_gate
    gate_result = triple_stationarity_gate(residual_series, asof, lookback=gate_lb)
    adf_pass = bool(gate_result["adf_pass"])
    composite = composite_reversion_score(z, hl, h, adf_pass)

    # Gate routing — uses triple-gate verdict, then layers on cross-PC + HL
    if eff_n < EFF_N_FLOOR:
        gate = "low_n"
    elif gate_result["random_walk"]:
        gate = "random_walk"           # gameplan §A6 — emit suppressed downstream
    elif gate_result["drift_present"]:
        gate = "drift"                 # ADF passes but KPSS rejects
    elif not adf_pass:
        gate = "non_stationary"
    elif cross_pc_corr is not None and abs(cross_pc_corr) > 0.3:
        gate = "regime_unstable"
    elif hl is None or hl < hl_lo or hl > hl_hi:
        gate = "non_stationary"
    elif gate_result["all_three"]:
        gate = "clean"
    else:
        gate = "non_stationary"
    return StructureCandidate(
        target_pc=target_pc,
        symbols=list(symbols) if symbols is not None else [],
        weights=np.asarray(weights, dtype=float)
                if weights is not None else np.zeros(0, dtype=float),
        pv01_sum=float(pv01_sum),
        residual_today_bp=float(today) if today is not None and np.isfinite(today) else None,
        residual_z=float(z) if z is not None else None,
        half_life_d=float(hl) if hl is not None else None,
        adf_pass=adf_pass,
        composite_score=float(composite) if composite is not None else None,
        eff_n=eff_n,
        gate_quality=gate,
    )


# =============================================================================
# S8 — PC-residual rich/cheap (single-leg + traded structures + packs)
# =============================================================================
def _outright_yield_bp_panel(close_panel: pd.DataFrame, outrights: list) -> pd.DataFrame:
    """Convert outright stored prices to implied yield in bp per tenor."""
    if close_panel is None or close_panel.empty:
        return pd.DataFrame()
    aligned = close_panel.reindex(columns=outrights)
    return (100.0 - aligned) * 100.0


def reconstruct_yields(pc_panel: pd.DataFrame,
                        pca_fit: PCAFit,
                        target_tenor_months: list = None) -> pd.DataFrame:
    """Reconstruct CMC yields from PCs: Y_recon = Σ_k PC_k · L_k + feature_mean_cumsum.

    NOTE: project_to_pcs operates on Δ_CMC, so reconstruction here yields the
    Δ-CMC reconstruction. To get level reconstruction, integrate from a base level.
    For per-instrument residual rich/cheap (S8), what we really want is
    per-tenor centred residual, which is: observed_delta − reconstructed_delta.
    """
    if pc_panel is None or pc_panel.empty or pca_fit is None:
        return pd.DataFrame()
    if target_tenor_months is None:
        target_tenor_months = list(pca_fit.tenors_months)
    L = pca_fit.loadings
    n_pc = L.shape[0]
    pcs = pc_panel[[f"PC{k + 1}" for k in range(n_pc) if f"PC{k + 1}" in pc_panel.columns]]
    if pcs.empty:
        return pd.DataFrame()
    pc_arr = pcs.values
    L_used = L[:pc_arr.shape[1]]
    recon = pc_arr @ L_used + pca_fit.feature_mean.reshape(1, -1)
    df = pd.DataFrame(recon, index=pcs.index, columns=pca_fit.tenors_months)
    # If asked for a different target_tenor_months, interpolate
    if list(target_tenor_months) != list(pca_fit.tenors_months):
        out_rows = []
        for ts, row in df.iterrows():
            out_rows.append({"asof": ts,
                             **{int(t): float(v) for t, v in zip(target_tenor_months,
                                                                  _pchip_curve(np.asarray(pca_fit.tenors_months,
                                                                                          dtype=float),
                                                                                row.values.astype(float),
                                                                                np.asarray(target_tenor_months,
                                                                                            dtype=float)))}})
        df = pd.DataFrame(out_rows).set_index("asof").sort_index()
    return df


def _detect_base_product(symbol: str) -> str:
    """Detect the base_product code from a symbol prefix (e.g. 'ERH26' → 'ER').
    Falls back to 'SRA' if no known prefix matches."""
    if not symbol:
        return "SRA"
    try:
        from lib.markets import parse_symbol_to_base_product
        bp = parse_symbol_to_base_product(symbol)
        return bp if bp else "SRA"
    except Exception:
        return "SRA"


def _instrument_loadings(symbol: str, strategy: str,
                            pca_fit: PCAFit, asof: date) -> Optional[np.ndarray]:
    """Compute the per-PC instrument loading row [L_PC1_inst, L_PC2_inst, L_PC3_inst].

    For an outright at tenor τ_i: L_k_inst = L_k(τ_i)  (interpolate via PCHIP at τ_i).
    For a 2-leg spread on (τ_a, τ_b): L_k_inst = L_k(τ_b) − L_k(τ_a).
    For a 3-leg fly on (τ_a, τ_b, τ_c): L_k_inst = L_k(τ_a) − 2·L_k(τ_b) + L_k(τ_c).

    Auto-detects base_product from symbol prefix so non-SRA markets (ER / FSR
    / FER / SON / YBA / CRA) also resolve correctly.
    """
    if pca_fit is None:
        return None
    L = pca_fit.loadings
    tenors = np.asarray(pca_fit.tenors_months, dtype=float)

    def _interp(tau: float) -> np.ndarray:
        return np.array([float(_pchip_curve(tenors, L[k], np.array([tau]))[0])
                         for k in range(L.shape[0])])

    if strategy == "outright":
        tau = _outright_tenor_months(symbol, asof)
        if tau is None:
            return None
        return _interp(tau)

    base = _detect_base_product(symbol)
    legs = parse_legs(symbol, base)
    if not legs or len(legs) == 1:
        return None
    leg_taus = [_outright_tenor_months(leg, asof) for leg in legs]
    if any(t is None for t in leg_taus):
        return None
    if strategy == "spread" and len(legs) == 2:
        return _interp(leg_taus[1]) - _interp(leg_taus[0])    # (b - a) convention
    if strategy == "fly" and len(legs) == 3:
        return _interp(leg_taus[0]) - 2.0 * _interp(leg_taus[1]) + _interp(leg_taus[2])
    return None


def per_outright_residuals(close_panel: pd.DataFrame, outrights: list,
                             pc_panel: pd.DataFrame, pca_fit: PCAFit,
                             asof: date, *, lookback: Optional[int] = None,
                             mode: Optional[str] = None,
                             resample: str = "D") -> pd.DataFrame:
    """Per-live-outright residual rich/cheap from the 3-PC factor model.

    Methodology (LEVEL residual, post-fix):
      1. change-residual_t = Δyield_t − (mean + Σ_k PC_k·L_k_inst)
      2. CUMULATE → level-residual_t = Σ_{s≤t} change-residual_s
      3. Detrend by subtracting rolling `detrend_window`d mean (so series is
         centered around 0 regardless of long-run model bias)
      4. Run z, OU half-life, ADF, KPSS, VR, Hurst on the LEVEL-residual series

    Why level (not change): mean-reversion is a LEVEL phenomenon. Change-residuals
    of a well-fitted PCA are ~white noise (HL→0, ADF passes trivially). Cumulating
    them gives the actual deviation from FV — which mean-reverts if and only if the
    underlying is genuinely PCA-consistent + has stationary idiosyncratic component.

    Mode controls residual_lookback, triple_gate_lookback, detrend_window.
    `lookback` overrides mode's residual_lookback when explicitly passed.
    `resample="W"` (weekly) is available for positional mode noise reduction.
    """
    rows = []
    if pca_fit is None or pc_panel is None or pc_panel.empty:
        return pd.DataFrame()
    mp = mode_params(mode)
    lookback = int(lookback) if lookback is not None else int(mp["residual_lookback"])
    gate_lb = int(mp["triple_gate_lookback"])
    detrend_w = int(mp["detrend_window"])
    yield_panel = _outright_yield_bp_panel(close_panel, outrights)
    if yield_panel.empty:
        return pd.DataFrame()
    delta_yield_panel = yield_panel.diff()
    pcs_aligned = pc_panel.reindex(yield_panel.index)
    L = pca_fit.loadings
    for sym in outrights:
        load_inst = _instrument_loadings(sym, "outright", pca_fit, asof)
        if load_inst is None:
            continue
        tau = _outright_tenor_months(sym, asof)
        if tau is None:
            mean_inst = 0.0
        else:
            mean_inst = float(_pchip_curve(
                np.asarray(pca_fit.tenors_months, dtype=float),
                np.asarray(pca_fit.feature_mean, dtype=float),
                np.array([tau]))[0])
        # Change-residual (per day)
        recon = np.full(len(yield_panel), float(mean_inst))
        for k in range(min(3, L.shape[0])):
            col = f"PC{k + 1}"
            if col in pcs_aligned.columns:
                vals = pcs_aligned[col].values * load_inst[k]
                recon = recon + np.where(np.isnan(vals), 0.0, vals)
        delta_observed = delta_yield_panel[sym].values
        change_residual = delta_observed - recon
        change_series = pd.Series(change_residual, index=yield_panel.index).dropna()
        # LEVEL residual = cumulative sum of change-residuals
        level_series = change_series.cumsum()
        # Optional weekly resampling for positional noise reduction
        if resample and resample.upper() == "W":
            level_series = level_series.resample("W-FRI").last().dropna()
        # Detrend: subtract rolling mean (window = mode.detrend_window).
        rolling_mean = level_series.rolling(detrend_w, min_periods=30).mean()
        level_series = (level_series - rolling_mean).dropna()
        if len(level_series) < 30:
            continue
        # Now all stats run on LEVEL series
        z = zscore_value(level_series, asof, lookback)
        hl = ou_half_life(level_series, asof, lookback)
        h = hurst_exponent(level_series, asof, lookback)
        adf = adf_test(level_series, asof, lookback, n_lags=1)
        composite = composite_reversion_score(z, hl, h, bool(adf.get("reject_5pct", False)))
        try:
            today = float(level_series.loc[level_series.index.date == asof].iloc[0])
        except Exception:
            try:
                today = float(level_series.iloc[-1])
            except Exception:
                today = None
        eff_n = int((level_series.index < pd.Timestamp(asof)).sum())
        from lib.mean_reversion import triple_stationarity_gate
        gate_result = triple_stationarity_gate(level_series, asof, lookback=gate_lb)
        if eff_n < EFF_N_FLOOR:
            gate = "low_n"
        elif gate_result["random_walk"]:
            gate = "random_walk"
        elif gate_result["drift_present"]:
            gate = "drift"
        elif not bool(adf.get("reject_5pct", False)):
            gate = "non_stationary"
        elif gate_result["all_three"]:
            gate = "clean"
        else:
            gate = "non_stationary"
        rows.append({
            "instrument": sym,
            "kind": "outright",
            "residual_today_bp": today,
            "residual_z": z,
            "half_life": hl,
            "hurst": h,
            "adf_pass": bool(adf.get("reject_5pct", False)),
            "triple_gate_pass": bool(gate_result.get("all_three", False)),
            "composite": composite,
            "eff_n": eff_n,
            "gate_quality": gate,
        })
    return pd.DataFrame(rows)


def _instrument_feature_mean(symbol: str, strategy: str, pca_fit: PCAFit,
                                asof: date) -> float:
    """Per-instrument scalar mean term: feature_mean dotted with instrument's tenor weight.

    For an outright at τ: PCHIP-interpolate feature_mean at τ.
    For 2-leg spread (a,b) returning Y_b-Y_a: mean[τ_b] - mean[τ_a].
    For 3-leg fly  (a,b,c) ±1±2±1: mean[τ_a] - 2·mean[τ_b] + mean[τ_c].
    """
    if pca_fit is None:
        return 0.0
    tenors = np.asarray(pca_fit.tenors_months, dtype=float)
    fmean = np.asarray(pca_fit.feature_mean, dtype=float)

    def _interp(tau: float) -> float:
        return float(_pchip_curve(tenors, fmean, np.array([tau]))[0])

    if strategy == "outright":
        tau = _outright_tenor_months(symbol, asof)
        return _interp(tau) if tau is not None else 0.0
    base = _detect_base_product(symbol)
    legs = parse_legs(symbol, base)
    if not legs:
        return 0.0
    leg_taus = [_outright_tenor_months(l, asof) for l in legs]
    if any(t is None for t in leg_taus):
        return 0.0
    if strategy == "spread" and len(legs) == 2:
        return _interp(leg_taus[1]) - _interp(leg_taus[0])
    if strategy == "fly" and len(legs) == 3:
        return _interp(leg_taus[0]) - 2.0 * _interp(leg_taus[1]) + _interp(leg_taus[2])
    return 0.0


def per_traded_spread_residuals(spreads_panel: dict,
                                  pc_panel: pd.DataFrame, pca_fit: PCAFit,
                                  asof: date, *, lookback: Optional[int] = None,
                                  mode: Optional[str] = None,
                                  resample: str = "D") -> pd.DataFrame:
    """Residuals on traded calendar spreads — LEVEL space, in bp.

    Mode controls residual_lookback, triple_gate_lookback, detrend_window.
    `lookback` overrides mode's residual_lookback when explicitly passed.
    """
    rows = []
    if pca_fit is None or pc_panel is None or pc_panel.empty:
        return pd.DataFrame()
    mp = mode_params(mode)
    lookback = int(lookback) if lookback is not None else int(mp["residual_lookback"])
    gate_lb = int(mp["triple_gate_lookback"])
    detrend_w = int(mp["detrend_window"])
    from lib.mean_reversion import triple_stationarity_gate
    for tenor, wide in (spreads_panel or {}).items():
        if wide is None or wide.empty:
            continue
        aligned = wide.copy()
        delta_aligned = aligned.diff()
        pcs_aligned = pc_panel.reindex(aligned.index)
        for sym in aligned.columns:
            load_inst = _instrument_loadings(sym, "spread", pca_fit, asof)
            if load_inst is None:
                continue
            mean_inst = _instrument_feature_mean(sym, "spread", pca_fit, asof)
            recon = np.full(len(aligned), float(mean_inst))
            for k in range(min(3, pca_fit.loadings.shape[0])):
                col = f"PC{k + 1}"
                if col in pcs_aligned.columns:
                    vals = pcs_aligned[col].values * load_inst[k]
                    recon = recon + np.where(np.isnan(vals), 0.0, vals)
            delta_observed = delta_aligned[sym].values.astype(float)
            change_residual = delta_observed - recon
            change_series = pd.Series(change_residual, index=aligned.index).dropna()
            # CUMULATE to level + detrend
            level_series = change_series.cumsum()
            if resample and resample.upper() == "W":
                level_series = level_series.resample("W-FRI").last().dropna()
            rolling_mean = level_series.rolling(detrend_w, min_periods=30).mean()
            level_series = (level_series - rolling_mean).dropna()
            if len(level_series) < 30:
                continue
            z = zscore_value(level_series, asof, lookback)
            hl = ou_half_life(level_series, asof, lookback)
            h = hurst_exponent(level_series, asof, lookback)
            adf = adf_test(level_series, asof, lookback, n_lags=1)
            composite = composite_reversion_score(z, hl, h, bool(adf.get("reject_5pct", False)))
            try:
                today = float(level_series.loc[level_series.index.date == asof].iloc[0])
            except Exception:
                try:
                    today = float(level_series.iloc[-1])
                except Exception:
                    today = None
            eff_n = int((level_series.index < pd.Timestamp(asof)).sum())
            gate_result = triple_stationarity_gate(level_series, asof, lookback=gate_lb)
            if eff_n < EFF_N_FLOOR:
                gate = "low_n"
            elif gate_result["random_walk"]:
                gate = "random_walk"
            elif gate_result["drift_present"]:
                gate = "drift"
            elif not bool(adf.get("reject_5pct", False)):
                gate = "non_stationary"
            elif gate_result["all_three"]:
                gate = "clean"
            else:
                gate = "non_stationary"
            rows.append({
                "instrument": sym, "kind": f"spread_{tenor}m",
                "residual_today_bp": today, "residual_z": z, "half_life": hl,
                "hurst": h, "adf_pass": bool(adf.get("reject_5pct", False)),
                "triple_gate_pass": bool(gate_result.get("all_three", False)),
                "composite": composite, "eff_n": eff_n, "gate_quality": gate,
            })
    return pd.DataFrame(rows)


def per_traded_fly_residuals(flies_panel: dict,
                               pc_panel: pd.DataFrame, pca_fit: PCAFit,
                               asof: date, *, lookback: Optional[int] = None,
                               mode: Optional[str] = None,
                               resample: str = "D") -> pd.DataFrame:
    """Residuals on traded ±1±2±1 flies — LEVEL space, in bp.

    Mode controls residual_lookback, triple_gate_lookback, detrend_window.
    """
    rows = []
    if pca_fit is None or pc_panel is None or pc_panel.empty:
        return pd.DataFrame()
    mp = mode_params(mode)
    lookback = int(lookback) if lookback is not None else int(mp["residual_lookback"])
    gate_lb = int(mp["triple_gate_lookback"])
    detrend_w = int(mp["detrend_window"])
    from lib.mean_reversion import triple_stationarity_gate
    for tenor, wide in (flies_panel or {}).items():
        if wide is None or wide.empty:
            continue
        delta_wide = wide.diff()
        pcs_aligned = pc_panel.reindex(wide.index)
        for sym in wide.columns:
            load_inst = _instrument_loadings(sym, "fly", pca_fit, asof)
            if load_inst is None:
                continue
            mean_inst = _instrument_feature_mean(sym, "fly", pca_fit, asof)
            recon = np.full(len(wide), float(mean_inst))
            for k in range(min(3, pca_fit.loadings.shape[0])):
                col = f"PC{k + 1}"
                if col in pcs_aligned.columns:
                    vals = pcs_aligned[col].values * load_inst[k]
                    recon = recon + np.where(np.isnan(vals), 0.0, vals)
            delta_observed = delta_wide[sym].values.astype(float)
            change_residual = delta_observed - recon
            change_series = pd.Series(change_residual, index=wide.index).dropna()
            # CUMULATE + detrend
            level_series = change_series.cumsum()
            if resample and resample.upper() == "W":
                level_series = level_series.resample("W-FRI").last().dropna()
            rolling_mean = level_series.rolling(detrend_w, min_periods=30).mean()
            level_series = (level_series - rolling_mean).dropna()
            if len(level_series) < 30:
                continue
            z = zscore_value(level_series, asof, lookback)
            hl = ou_half_life(level_series, asof, lookback)
            h = hurst_exponent(level_series, asof, lookback)
            adf = adf_test(level_series, asof, lookback, n_lags=1)
            composite = composite_reversion_score(z, hl, h, bool(adf.get("reject_5pct", False)))
            try:
                today = float(level_series.loc[level_series.index.date == asof].iloc[0])
            except Exception:
                try:
                    today = float(level_series.iloc[-1])
                except Exception:
                    today = None
            eff_n = int((level_series.index < pd.Timestamp(asof)).sum())
            gate_result = triple_stationarity_gate(level_series, asof, lookback=gate_lb)
            if eff_n < EFF_N_FLOOR:
                gate = "low_n"
            elif gate_result["random_walk"]:
                gate = "random_walk"
            elif gate_result["drift_present"]:
                gate = "drift"
            elif not bool(adf.get("reject_5pct", False)):
                gate = "non_stationary"
            elif gate_result["all_three"]:
                gate = "clean"
            else:
                gate = "non_stationary"
            rows.append({
                "instrument": sym, "kind": f"fly_{tenor}m",
                "residual_today_bp": today, "residual_z": z, "half_life": hl,
                "hurst": h, "adf_pass": bool(adf.get("reject_5pct", False)),
                "triple_gate_pass": bool(gate_result.get("all_three", False)),
                "composite": composite, "eff_n": eff_n, "gate_quality": gate,
            })
    return pd.DataFrame(rows)


def pack_residuals(close_panel_outrights: pd.DataFrame, outrights_df: pd.DataFrame,
                    pc_panel: pd.DataFrame, pca_fit: PCAFit,
                    asof: date, *, lookback: Optional[int] = None,
                    mode: Optional[str] = None,
                    resample: str = "D") -> pd.DataFrame:
    """Synthetic pack-rate residuals for white/red/green/blue/gold packs.

    Mode controls residual_lookback, triple_gate_lookback, detrend_window.
    """
    rows = []
    if outrights_df is None or outrights_df.empty or pca_fit is None:
        return pd.DataFrame()
    if pc_panel is None or pc_panel.empty:
        return pd.DataFrame()
    mp = mode_params(mode)
    lookback = int(lookback) if lookback is not None else int(mp["residual_lookback"])
    gate_lb = int(mp["triple_gate_lookback"])
    detrend_w = int(mp["detrend_window"])
    packs = compute_pack_groups(outrights_df)
    L = pca_fit.loadings
    tenors = np.asarray(pca_fit.tenors_months, dtype=float)
    yield_panel = _outright_yield_bp_panel(close_panel_outrights,
                                              list(outrights_df["symbol"]))
    if yield_panel.empty:
        return pd.DataFrame()
    pcs_aligned = pc_panel.reindex(yield_panel.index)
    for pack_name, leg_syms in packs:
        # Synthetic pack yield series (level)
        try:
            sub = yield_panel[leg_syms].mean(axis=1)
        except KeyError:
            continue
        delta_sub = sub.diff()
        # Pack instrument loading = mean of leg loadings
        leg_loads = []
        leg_means = []
        for leg in leg_syms:
            li = _instrument_loadings(leg, "outright", pca_fit, asof)
            if li is None:
                continue
            leg_loads.append(li)
            leg_means.append(_instrument_feature_mean(leg, "outright", pca_fit, asof))
        if not leg_loads:
            continue
        load_inst = np.mean(np.vstack(leg_loads), axis=0)
        mean_inst = float(np.mean(leg_means)) if leg_means else 0.0
        recon = np.full(len(sub), mean_inst)
        for k in range(min(3, L.shape[0])):
            col = f"PC{k + 1}"
            if col in pcs_aligned.columns:
                vals = pcs_aligned[col].values * load_inst[k]
                recon = recon + np.where(np.isnan(vals), 0.0, vals)
        # Change-residual → cumulate → level + detrend (matches the other paths)
        change_residual = delta_sub.values - recon
        change_series = pd.Series(change_residual, index=sub.index).dropna()
        level_series = change_series.cumsum()
        if resample and resample.upper() == "W":
            level_series = level_series.resample("W-FRI").last().dropna()
        rolling_mean = level_series.rolling(detrend_w, min_periods=30).mean()
        level_series = (level_series - rolling_mean).dropna()
        if len(level_series) < 30:
            continue
        z = zscore_value(level_series, asof, lookback)
        hl = ou_half_life(level_series, asof, lookback)
        h = hurst_exponent(level_series, asof, lookback)
        adf = adf_test(level_series, asof, lookback, n_lags=1)
        composite = composite_reversion_score(z, hl, h, bool(adf.get("reject_5pct", False)))
        try:
            today = float(level_series.loc[level_series.index.date == asof].iloc[0])
        except Exception:
            try:
                today = float(level_series.iloc[-1])
            except Exception:
                today = None
        eff_n = int((level_series.index < pd.Timestamp(asof)).sum())
        from lib.mean_reversion import triple_stationarity_gate
        gate_result = triple_stationarity_gate(level_series, asof, lookback=gate_lb)
        if eff_n < EFF_N_FLOOR:
            gate = "low_n"
        elif gate_result["random_walk"]:
            gate = "random_walk"
        elif gate_result["drift_present"]:
            gate = "drift"
        elif not bool(adf.get("reject_5pct", False)):
            gate = "non_stationary"
        elif gate_result["all_three"]:
            gate = "clean"
        else:
            gate = "non_stationary"
        rows.append({
            "instrument": pack_name, "kind": "pack",
            "residual_today_bp": today, "residual_z": z, "half_life": hl,
            "hurst": h, "adf_pass": bool(adf.get("reject_5pct", False)),
            "triple_gate_pass": bool(gate_result.get("all_three", False)),
            "composite": composite, "eff_n": eff_n, "gate_quality": gate,
            "members": ",".join(leg_syms),
        })
    return pd.DataFrame(rows)


# =============================================================================
# S9 — Regime & cycle map (using existing classifier)
# =============================================================================
def regime_label_panel(cmc_panel: pd.DataFrame,
                        outrights: list,
                        front_end: int = DEFAULT_FRONT_END,
                        mid_end: int = DEFAULT_MID_END) -> pd.Series:
    """Classify each row of cmc_panel into a regime label using lib.regime.

    For PC subtab purposes: we classify the *Δ_CMC* per day using `classify_regime`
    on the τ-grid (treating tenor index as the contract index). Returns Series
    of label strings indexed by date.
    """
    out = pd.Series(dtype=object)
    if cmc_panel is None or cmc_panel.empty:
        return out
    delta = cmc_panel.diff().dropna(how="all")
    labels = {}
    cols = list(cmc_panel.columns)
    for ts, row in delta.iterrows():
        changes = [float(v) if pd.notna(v) else None for v in row.values]
        try:
            r = classify_regime(changes, cols, front_end, mid_end)
            labels[ts] = r.get("label", "—")
        except Exception:
            labels[ts] = "—"
    return pd.Series(labels)


def regime_trajectory(pc_panel: pd.DataFrame, asof: date, *,
                       lookback_days: int = 30) -> pd.DataFrame:
    """Last `lookback_days` rows of pc_panel up to asof, for the scatter trail."""
    if pc_panel is None or pc_panel.empty:
        return pd.DataFrame()
    ts = pd.Timestamp(asof)
    sub = pc_panel.loc[pc_panel.index <= ts].tail(lookback_days)
    return sub


def regime_persistence_stats(label_series: pd.Series) -> dict:
    """Summary stats on regime persistence."""
    out = {"current_label": None, "days_since_change": None,
           "median_run_length": None, "label_counts": {}, "confidence_proxy": None}
    if label_series is None or label_series.empty:
        return out
    out["current_label"] = str(label_series.iloc[-1])
    # Days since change
    last = label_series.iloc[-1]
    n_back = 0
    for v in reversed(label_series.values):
        if v == last:
            n_back += 1
        else:
            break
    out["days_since_change"] = int(n_back)
    # Median run length over series
    runs = []
    cur, count = None, 0
    for v in label_series.values:
        if v == cur:
            count += 1
        else:
            if cur is not None:
                runs.append(count)
            cur = v
            count = 1
    if cur is not None:
        runs.append(count)
    if runs:
        out["median_run_length"] = float(np.median(runs))
    # Counts
    out["label_counts"] = {k: int(v) for k, v in
                            label_series.value_counts().to_dict().items()}
    # Confidence proxy = consistency over last 5 days (fraction matching current)
    last5 = label_series.iloc[-5:].values if len(label_series) >= 5 else label_series.values
    out["confidence_proxy"] = float((last5 == last).sum() / len(last5)) if len(last5) > 0 else None
    return out


def cycle_phase_tag(pc_panel: pd.DataFrame, fomc_calendar: pd.DataFrame,
                      asof: date) -> tuple:
    """Composite 8-phase cycle label per gameplan §A7c lite.

    Inputs:
      pc_panel        — PC time series (must contain PC1, PC2 columns).
      fomc_calendar   — DataFrame with `decision_date` column.
      asof            — today.

    Returns (phase_label, composite_inputs_dict). `gate_quality = low_n` is
    surfaced separately in the UI; this function always returns a phase.
    """
    out_inputs = {
        "realized_policy_bp_12mtg": None,
        "pc1_percentile_1y": None,
        "pc2_gradient_30d": None,
    }
    if pc_panel is None or pc_panel.empty or "PC1" not in pc_panel.columns:
        return ("trough", out_inputs)
    pc1 = pc_panel["PC1"].dropna()
    pc2 = pc_panel["PC2"].dropna() if "PC2" in pc_panel.columns else pd.Series(dtype=float)
    ts = pd.Timestamp(asof)
    pc1_today = float(pc1.loc[pc1.index <= ts].iloc[-1]) if (pc1.loc[pc1.index <= ts]).any() else None
    # PC1 percentile in trailing 252d
    pc1_hist = pc1.loc[pc1.index < ts].tail(252)
    if len(pc1_hist) > 10 and pc1_today is not None:
        out_inputs["pc1_percentile_1y"] = float((pc1_hist < pc1_today).sum() / len(pc1_hist))
    # PC2 gradient over last 30 days
    pc2_30 = pc2.loc[pc2.index <= ts].tail(30)
    if len(pc2_30) >= 5:
        out_inputs["pc2_gradient_30d"] = float(pc2_30.iloc[-1] - pc2_30.iloc[0])
    # Realized policy bp over last 12 meetings (signed cumulative)
    if fomc_calendar is not None and not fomc_calendar.empty:
        col = "decision_date" if "decision_date" in fomc_calendar.columns else fomc_calendar.columns[0]
        past_meetings = fomc_calendar[fomc_calendar[col] <= asof].tail(12)
        # We don't have actual policy moves wired here; use PC1 trend through these
        # meetings as a proxy (PC1 at meeting − PC1 12 meetings ago, in basis points
        # implied through level loading).
        if not past_meetings.empty and pc1_today is not None and len(pc1) > 12:
            # Approximation: use net PC1 change over the past-12-meeting span
            try:
                first_m = past_meetings.iloc[0][col]
                pc1_then = pc1.loc[pc1.index >= pd.Timestamp(first_m)].iloc[0]
                out_inputs["realized_policy_bp_12mtg"] = float(pc1_today - pc1_then)
            except Exception:
                pass

    # Composite mapping
    pol = out_inputs["realized_policy_bp_12mtg"] or 0.0
    pct = out_inputs["pc1_percentile_1y"] or 0.5
    grad = out_inputs["pc2_gradient_30d"] or 0.0

    # Cuts vs hikes from sign of policy bp
    cutting = pol > 5.0    # PC1 dropping (bp positive in our convention) = rates falling = cuts
    hiking = pol < -5.0
    if cutting:
        if pct >= 0.66 and grad < 0:
            phase = "early-cut"
        elif pct >= 0.33 and grad >= 0:
            phase = "mid-cut"
        elif pct < 0.33:
            phase = "late-cut"
        else:
            phase = "trough"
    elif hiking:
        if pct <= 0.33 and grad > 0:
            phase = "early-hike"
        elif pct <= 0.66 and grad >= 0:
            phase = "mid-hike"
        elif pct > 0.66:
            phase = "late-hike"
        else:
            phase = "peak"
    else:
        phase = "trough" if pct < 0.5 else "peak"
    return (phase, out_inputs)


# =============================================================================
# S10 — Diagnostics
# =============================================================================
def reconstruction_error_series(cmc_panel: pd.DataFrame,
                                  rolling_fits: dict,
                                  pc_panel: pd.DataFrame) -> pd.Series:
    """Daily reconstruction error: ‖Δ_CMC[t] − (L^T·PC[t] + mean)‖_2 in bp.

    Each row of pc_panel is the projection of Δ_CMC[t] onto the loadings of the
    most-recent prior fit. So the rank-3 reconstruction uses today's PC values
    directly (not their first-difference), plus the training-sample feature_mean.
    """
    if cmc_panel is None or cmc_panel.empty or not rolling_fits or pc_panel is None:
        return pd.Series(dtype=float)
    delta_cmc = cmc_panel.diff().dropna(how="all")
    out = {}
    for ts, drow in delta_cmc.iterrows():
        target_date = ts.date()
        fit = _pick_fit_for_date(rolling_fits, target_date)
        if fit is None:
            continue
        if ts not in pc_panel.index:
            continue
        pc_today = pc_panel.loc[ts]
        pc_arr = np.array([float(pc_today.get(f"PC{k + 1}", 0.0))
                            if not pd.isna(pc_today.get(f"PC{k + 1}", np.nan))
                            else 0.0
                            for k in range(fit.loadings.shape[0])])
        if np.all(pc_arr == 0):
            continue
        L = fit.loadings
        recon = L.T @ pc_arr + fit.feature_mean
        try:
            actual = drow[fit.tenors_months].values.astype(float)
        except KeyError:
            actual = drow.reindex(fit.tenors_months).values.astype(float)
        if np.any(np.isnan(actual)):
            continue
        err = float(np.linalg.norm(actual - recon))
        out[ts] = err
    return pd.Series(out).sort_index()


def reconstruction_pct_explained(cmc_panel: pd.DataFrame,
                                   rolling_fits: dict,
                                   pc_panel: pd.DataFrame, asof: date) -> Optional[float]:
    """Percent of today's L1 Δ_CMC explained by the 3-PC reconstruction (0..1).

    Δ_CMC[t, τ] = Σ_k PC_k[t]·L_k[τ] + mean[τ] + residual[τ]
    pct_explained = 1 − ‖residual‖_1 / ‖Δ_CMC‖_1
    """
    if cmc_panel is None or cmc_panel.empty or not rolling_fits:
        return None
    ts = pd.Timestamp(asof)
    if ts not in cmc_panel.index:
        cands = cmc_panel.index[cmc_panel.index <= ts]
        if len(cands) == 0:
            return None
        ts = cands[-1]
    pos = cmc_panel.index.get_loc(ts)
    if pos == 0:
        return None
    fit = _pick_fit_for_date(rolling_fits, ts.date())
    if fit is None:
        return None
    delta = (cmc_panel.iloc[pos] - cmc_panel.iloc[pos - 1]).reindex(fit.tenors_months).values.astype(float)
    if np.any(np.isnan(delta)):
        return None
    if ts not in pc_panel.index:
        return None
    pc_today = pc_panel.loc[ts]
    pc_arr = np.array([float(pc_today.get(f"PC{k + 1}", 0.0))
                        if not pd.isna(pc_today.get(f"PC{k + 1}", np.nan))
                        else 0.0
                        for k in range(fit.loadings.shape[0])])
    L = fit.loadings
    recon = L.T @ pc_arr + fit.feature_mean
    abs_total = float(np.nansum(np.abs(delta)))
    abs_residual = float(np.nansum(np.abs(delta - recon)))
    if abs_total < 1e-9:
        return 1.0
    return float(max(0.0, min(1.0, 1.0 - abs_residual / abs_total)))


def variance_explained_rolling(rolling_fits: dict) -> pd.DataFrame:
    """DataFrame of variance ratios over time."""
    rows = []
    for d, fit in sorted(rolling_fits.items()):
        vr = fit.variance_ratio
        rec = {"asof": pd.Timestamp(d)}
        for k in range(min(3, len(vr))):
            rec[f"PC{k + 1}_var"] = float(vr[k])
        rec["cum3"] = float(sum(vr[:3]))
        rows.append(rec)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("asof")


def detect_outlier_days(reconstruction_errors: pd.Series, *,
                          percentile: float = 99.0) -> list:
    """Return list of dates where reconstruction error exceeds the trailing-1y Pth percentile."""
    if reconstruction_errors is None or reconstruction_errors.empty:
        return []
    s = reconstruction_errors.dropna()
    if len(s) < 30:
        return []
    flagged = []
    rolling = s.rolling(252, min_periods=30).quantile(percentile / 100.0)
    for ts, val in s.items():
        thresh = rolling.loc[ts] if ts in rolling.index else None
        if thresh is not None and not pd.isna(thresh) and val > float(thresh):
            flagged.append(ts)
    return flagged


def loadings_stability_heatmap_data(rolling_fits: dict, pc_index: int) -> pd.DataFrame:
    """DataFrame index=refit_date, columns=tenor_months, values=loading for one PC.

    `pc_index` is 0-based (0 → PC1, 1 → PC2, 2 → PC3).
    """
    rows = []
    cols = None
    for d, fit in sorted(rolling_fits.items()):
        if fit.loadings.shape[0] <= pc_index:
            continue
        if cols is None:
            cols = list(fit.tenors_months)
        rec = {"asof": pd.Timestamp(d)}
        for k, t in enumerate(fit.tenors_months):
            rec[t] = float(fit.loadings[pc_index, k])
        rows.append(rec)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("asof")


# =============================================================================
# Module-level cache wrapper — single entry point for the UI layer
# =============================================================================
def _build_pca_panel_internal(asof: date,
                                tenor_grid_months: list,
                                lookbacks: tuple,
                                window_mode: str = "rolling_252",
                                refit_step_days: int = 5,
                                history_days: int = 700,
                                sparse_enabled: bool = True,
                                anomaly_pct: float = 99.0,
                                mode: Optional[str] = None,
                                resample: str = "D",
                                base_product: str = "SRA") -> dict:
    """Heavy-compute kernel — assemble all panels needed by the 8 inner tabs.

    `base_product` selects the market (SRA / ER / FSR / FER / SON / YBA / CRA).
    SRA flows through `lib.sra_data`; all others use `lib.market_data` and the
    market-specific central bank calendar from `MARKETS[base_product]`.
    """
    mode = _resolve_mode(mode)
    mp = mode_params(mode)
    res_lb = int(mp["residual_lookback"])
    market_cfg = _get_market(base_product)
    rate_cutover = market_cfg.get("rate_history_cutover", LIBOR_SOFR_CUTOVER)

    # Market-aware data-loading shims — back-compat for SRA, generic for others
    def _load_outrights():
        if base_product == "SRA":
            return _sra_get_outrights()
        return _md.get_outrights(base_product)

    def _load_curve_panel(strategy, tenor, start_d, end_d):
        if base_product == "SRA":
            return load_sra_curve_panel(strategy, tenor, start_d, end_d)
        return _md.load_curve_panel(base_product, strategy, tenor, start_d, end_d)

    def _load_listed():
        if base_product == "SRA":
            from lib.sra_data import load_listed_spread_fly_panel as _ls
            return _ls(history_start, asof, max_tenor_months=12)
        return _md.load_listed_spread_fly_panel(base_product, history_start, asof,
                                                    max_tenor_months=12)

    def _load_refrate(start_d, end_d):
        if base_product == "SRA":
            return _sra_load_reference_rate_panel(start_d, end_d)
        return _md.load_reference_rate_panel(base_product, start_d, end_d)

    def _load_meetings():
        """Return DataFrame with `decision_date` column. Uses market's CB."""
        from importlib import import_module
        cb_mod_name = market_cfg.get("central_bank_module", "lib.fomc")
        cb_fn_name = market_cfg.get("central_bank_decision_fn", "load_fomc_meetings")
        try:
            cb_mod = import_module(cb_mod_name)
            cb_fn = getattr(cb_mod, cb_fn_name)
            return cb_fn()
        except Exception:
            return pd.DataFrame(columns=["decision_date", "press_conf", "sep"])

    def _get_avail_tenors(strategy):
        if base_product == "SRA":
            from lib.sra_data import get_available_tenors as _gat
            return _gat(strategy)
        return _md.get_available_tenors(base_product, strategy)

    def _get_spreads_tenor(t):
        if base_product == "SRA":
            return _sra_get_spreads(t)
        return _md.get_spreads(base_product, t)

    def _get_flies_tenor(t):
        if base_product == "SRA":
            return _sra_get_flies(t)
        return _md.get_flies(base_product, t)

    def _packs(df):
        if base_product == "SRA":
            return _sra_compute_pack_groups(df)
        return _md.compute_pack_groups(df, base_product)

    out = {
        "asof": asof,
        "base_product": base_product,
        "market": dict(market_cfg),
        "mode": mode,
        "mode_params": mp,
        "tenor_grid_months": list(tenor_grid_months),
        "lookbacks": list(lookbacks),
        "outrights_df": None,
        "outright_close_panel": None,
        "outright_symbols": [],
        "cmc_panel": pd.DataFrame(),
        "sofr_panel": pd.DataFrame(),
        "pca_fit_static": None,
        "sparse_pca_fit": None,
        "rolling_fits": {},
        "pc_panel": pd.DataFrame(),
        "anchor_series": pd.Series(dtype=float),
        "pc_diagnostics": {},
        "delta_decomposition": {},
        "structure_candidates": [],
        "residual_outrights": pd.DataFrame(),
        "residual_traded_spreads": pd.DataFrame(),
        "residual_traded_flies": pd.DataFrame(),
        "residual_packs": pd.DataFrame(),
        "regime_label_series": pd.Series(dtype=object),
        "cycle_phase": ("trough", {}),
        "cross_pc_corr": pd.DataFrame(),
        "reconstruction_error": pd.Series(dtype=float),
        "reconstruction_pct_today": None,
        "variance_explained_history": pd.DataFrame(),
        "outlier_days": [],
        "print_quality_alerts": [],
        "info": {
            "n_outrights": 0,
            "n_traded_spreads": 0,
            "n_traded_flies": 0,
            "post_cutover_days": 0,
            "post_cutover_start": rate_cutover,
            "fit_window_start": None,
            "fit_window_end": None,
        },
    }

    # ─── 1. Pull live outrights (market-aware) ──────────────────────────────
    outrights_df = _load_outrights()
    if outrights_df is None or outrights_df.empty:
        return out
    outright_symbols = list(outrights_df["symbol"])
    out["outrights_df"] = outrights_df
    out["outright_symbols"] = outright_symbols
    out["info"]["n_outrights"] = len(outright_symbols)

    # ─── 2. Load close history for outrights (post-cutover + buffer) ────────
    history_start = max(rate_cutover, asof - timedelta(days=int(history_days * 1.6)))
    panel = _load_curve_panel("outright", None, history_start, asof)
    if panel is None or panel.empty:
        return out
    wide_close = pivot_curve_panel(panel, outright_symbols, "close")
    out["outright_close_panel"] = wide_close
    # Export yield-bp panel for downstream consumers (cross-asset convexity, etc.)
    out["yield_bp_panel"] = _outright_yield_bp_panel(wide_close, outright_symbols)

    # ─── 2b. Load LISTED spread + fly OHLC for actual contract pricing ──────
    try:
        listed = _load_listed()
        out["spread_close_panel"] = listed["spread_close"]
        out["fly_close_panel"] = listed["fly_close"]
        out["spread_catalog"] = listed["spread_catalog"]
        out["fly_catalog"] = listed["fly_catalog"]
    except Exception:
        out["spread_close_panel"] = pd.DataFrame()
        out["fly_close_panel"] = pd.DataFrame()
        out["spread_catalog"] = pd.DataFrame()
        out["fly_catalog"] = pd.DataFrame()

    # ─── 3. Reference rate panel (overnight + policy corridor) ──────────────
    try:
        sofr_panel = _load_refrate(history_start, asof)
        out["sofr_panel"] = sofr_panel
    except Exception:
        sofr_panel = pd.DataFrame()

    # ─── 4. CMC build on the full sample ────────────────────────────────────
    asof_dates = [d.date() for d in wide_close.index]
    cmc_panel = build_cmc_panel(asof_dates, wide_close, outright_symbols,
                                  target_tenors_months=list(tenor_grid_months),
                                  sofr_panel=sofr_panel)
    if cmc_panel is None or cmc_panel.empty or len(cmc_panel) < 30:
        return out
    out["cmc_panel"] = cmc_panel
    # Print-quality alerts
    out["print_quality_alerts"] = cmc_print_quality_alert(cmc_panel)

    # ─── 5. Static PCA + sparse + rolling ───────────────────────────────────
    pca_fit_static = fit_pca_static(cmc_panel, n_components=3)
    out["pca_fit_static"] = pca_fit_static
    if pca_fit_static is None:
        return out
    if sparse_enabled:
        out["sparse_pca_fit"] = fit_sparse_pca(cmc_panel, pca_fit_static,
                                                target_nonzeros=10)

    # ─── 5b. Segment-PCA fits (front/belly/back) — Phase 3 wiring ───────────
    # Used by gen_front/belly/back_pca_ideas. Was returning empty because
    # panel["multi_res_pca"] wasn't populated.
    try:
        tenors_arr = np.asarray(cmc_panel.columns, dtype=float)
        segment_bounds = {
            "front": (1, 12),
            "belly": (9, 30),
            "back":  (24, 60),
        }
        multi_res = {}
        for seg_name, (lo, hi) in segment_bounds.items():
            mask = (tenors_arr >= lo) & (tenors_arr <= hi)
            if mask.sum() < 5:
                continue
            sub_cmc = cmc_panel.loc[:, mask]
            seg_fit = fit_pca_static(sub_cmc, n_components=3)
            if seg_fit is not None:
                multi_res[seg_name] = seg_fit
        out["multi_res_pca"] = multi_res
    except Exception:
        out["multi_res_pca"] = {}
    if window_mode == "static":
        rolling = {pca_fit_static.asof: pca_fit_static}
    else:
        window_days = 252 if window_mode == "rolling_252" else 504
        rolling = fit_pca_rolling(cmc_panel, window_days=window_days,
                                    step_days=refit_step_days, n_components=3)
    out["rolling_fits"] = rolling
    if pca_fit_static.fit_window:
        out["info"]["fit_window_start"] = pca_fit_static.fit_window[0]
        out["info"]["fit_window_end"] = pca_fit_static.fit_window[1]
    out["info"]["post_cutover_days"] = int((asof - LIBOR_SOFR_CUTOVER).days)

    # ─── 6. Project PCs + anchor + cross-PC corr ────────────────────────────
    pc_panel = project_to_pcs(cmc_panel, rolling)
    out["pc_panel"] = pc_panel
    out["anchor_series"] = anchor_metric(cmc_panel)
    out["cross_pc_corr"] = cross_pc_corr_rolling(pc_panel, window=60)

    # ─── 7. Per-PC diagnostics + Anchor diagnostics ─────────────────────────
    diagnostics = {}
    for col in ("PC1", "PC2", "PC3"):
        if col in pc_panel.columns:
            d = pc_diagnostics(pc_panel[col], asof, lookback=90)
            d["zs"] = pc_zscore_multi(pc_panel[col], asof, lookbacks)
            d["pcts"] = pc_percentile_rank_multi(pc_panel[col], asof, lookbacks)
            d["pattern"] = pc_confluence_pattern(pc_panel[col], asof,
                                                  tuple(l for l in lookbacks if l <= 90))
            diagnostics[col] = d
    if not out["anchor_series"].empty:
        a = pc_diagnostics(out["anchor_series"], asof, lookback=90)
        a["zs"] = pc_zscore_multi(out["anchor_series"], asof, lookbacks)
        a["pcts"] = pc_percentile_rank_multi(out["anchor_series"], asof, lookbacks)
        a["pattern"] = pc_confluence_pattern(out["anchor_series"], asof,
                                              tuple(l for l in lookbacks if l <= 90))
        diagnostics["Anchor"] = a
    out["pc_diagnostics"] = diagnostics

    # ─── 8. Δ_CMC decomposition for today ───────────────────────────────────
    out["delta_decomposition"] = decompose_delta(cmc_panel, pca_fit_static,
                                                   pc_panel, asof,
                                                   rolling_fits=rolling)

    # ─── 9. Enumerate + score PCA-derived structures ────────────────────────
    candidates = []
    # PC3 flies (target_pc=3)
    pc3_triples = enumerate_pc3_fly_triples(outright_symbols, asof)
    yield_panel_outrights = _outright_yield_bp_panel(wide_close, outright_symbols)
    pv01_per_outright = 25.0    # SRA: $25/bp per contract; relative weights only
    today_corr_pc12 = None
    if not out["cross_pc_corr"].empty:
        try:
            today_corr_pc12 = float(out["cross_pc_corr"]["corr_PC1_PC2"].dropna().iloc[-1])
        except Exception:
            pass
    for triple in pc3_triples:
        load_subset = []
        for sym in triple:
            li = _instrument_loadings(sym, "outright", pca_fit_static, asof)
            if li is None:
                load_subset = []
                break
            load_subset.append(li)
        if not load_subset:
            continue
        L_sub = np.vstack(load_subset).T   # shape (n_pc, n_legs)
        w = solve_isolated_pc_weights(L_sub, target_pc=3,
                                        pv01_legs=np.array([pv01_per_outright] * len(triple)))
        if w is None:
            continue
        # Build residual using outright yields directly
        try:
            sub_panel = yield_panel_outrights[list(triple)]
        except KeyError:
            continue
        res_arr = (sub_panel.values * w.reshape(1, -1)).sum(axis=1)
        res_series = pd.Series(res_arr, index=sub_panel.index, name="res").dropna()
        cand = score_structure(res_series, asof, mode=mode,
                                cross_pc_corr=today_corr_pc12,
                                target_pc=3, symbols=list(triple),
                                weights=w, pv01_sum=float(np.dot([pv01_per_outright] * 3, w)))
        candidates.append(cand)
    # PC2 spreads (target_pc=2)
    pc2_triples = enumerate_pc2_spread_triples(outright_symbols, asof)
    for triple in pc2_triples[:60]:
        load_subset = []
        for sym in triple:
            li = _instrument_loadings(sym, "outright", pca_fit_static, asof)
            if li is None:
                load_subset = []
                break
            load_subset.append(li)
        if not load_subset:
            continue
        L_sub = np.vstack(load_subset).T
        w = solve_isolated_pc_weights(L_sub, target_pc=2,
                                        pv01_legs=np.array([pv01_per_outright] * len(triple)))
        if w is None:
            continue
        try:
            sub_panel = yield_panel_outrights[list(triple)]
        except KeyError:
            continue
        res_arr = (sub_panel.values * w.reshape(1, -1)).sum(axis=1)
        res_series = pd.Series(res_arr, index=sub_panel.index, name="res").dropna()
        cand = score_structure(res_series, asof, mode=mode,
                                cross_pc_corr=today_corr_pc12, target_pc=2,
                                symbols=list(triple), weights=w,
                                pv01_sum=float(np.dot([pv01_per_outright] * 3, w)))
        candidates.append(cand)
    # PC1 synthetics (target_pc=1)
    pc1_baskets = enumerate_pc1_synthetics(outright_symbols, asof)
    for basket in pc1_baskets[:30]:
        load_subset = []
        for sym in basket:
            li = _instrument_loadings(sym, "outright", pca_fit_static, asof)
            if li is None:
                load_subset = []
                break
            load_subset.append(li)
        if not load_subset:
            continue
        L_sub = np.vstack(load_subset).T
        w = solve_isolated_pc_weights(L_sub, target_pc=1,
                                        pv01_legs=np.array([pv01_per_outright] * len(basket)))
        if w is None:
            continue
        try:
            sub_panel = yield_panel_outrights[list(basket)]
        except KeyError:
            continue
        res_arr = (sub_panel.values * w.reshape(1, -1)).sum(axis=1)
        res_series = pd.Series(res_arr, index=sub_panel.index, name="res").dropna()
        cand = score_structure(res_series, asof, mode=mode,
                                cross_pc_corr=today_corr_pc12, target_pc=1,
                                symbols=list(basket), weights=w,
                                pv01_sum=float(np.dot([pv01_per_outright] * 4, w)))
        candidates.append(cand)
    out["structure_candidates"] = candidates

    # ─── 10. Residual rich/cheap surfaces ───────────────────────────────────
    out["residual_outrights"] = per_outright_residuals(
        wide_close, outright_symbols, pc_panel, pca_fit_static, asof,
        mode=mode, resample=resample)
    # Traded spreads + flies — load each tenor (market-aware)
    spreads_panel_dict = {}
    flies_panel_dict = {}
    try:
        spread_tenors = _get_avail_tenors("spread")
        fly_tenors = _get_avail_tenors("fly")
    except Exception:
        spread_tenors = []
        fly_tenors = []
    for t in spread_tenors:
        sdf = _get_spreads_tenor(t)
        if sdf is None or sdf.empty:
            continue
        sp = _load_curve_panel("spread", t, history_start, asof)
        if sp is None or sp.empty:
            continue
        wide = pivot_curve_panel(sp, list(sdf["symbol"]), "close")
        spreads_panel_dict[int(t)] = wide
    for t in fly_tenors:
        fdf = _get_flies_tenor(t)
        if fdf is None or fdf.empty:
            continue
        fp = _load_curve_panel("fly", t, history_start, asof)
        if fp is None or fp.empty:
            continue
        wide = pivot_curve_panel(fp, list(fdf["symbol"]), "close")
        flies_panel_dict[int(t)] = wide
    out["info"]["n_traded_spreads"] = sum(len(w.columns) for w in spreads_panel_dict.values())
    out["info"]["n_traded_flies"] = sum(len(w.columns) for w in flies_panel_dict.values())
    out["residual_traded_spreads"] = per_traded_spread_residuals(
        spreads_panel_dict, pc_panel, pca_fit_static, asof,
        mode=mode, resample=resample)
    out["residual_traded_flies"] = per_traded_fly_residuals(
        flies_panel_dict, pc_panel, pca_fit_static, asof,
        mode=mode, resample=resample)
    out["residual_packs"] = pack_residuals(
        wide_close, outrights_df, pc_panel, pca_fit_static, asof,
        mode=mode, resample=resample)
    # Packs dict keyed by pack-name (for gen_bundle_rv_ideas / gen_pack_ideas).
    # compute_pack_groups returns list[(pack_name, [leg_syms])] — convert to dict.
    try:
        out["packs"] = {name: list(legs) for name, legs in _packs(outrights_df)}
    except Exception:
        out["packs"] = {}

    # ─── 10b. Analog FV + Path FV (Phase 3 wiring) ──────────────────────────
    # Compute per-instrument analog FV (Mahalanobis k-NN + Ledoit-Wolf) so
    # gen_analog_fv_ideas can fire. Same for path-conditional FV.
    out["analog_fv_results"] = {}
    out["path_fv_results"] = {}
    try:
        from lib.pca_analogs import (knn_analog_search, analog_fv_band,
                                          path_bucket_assignment, path_conditional_fv)
        # State features for each historical day: [PC1, PC2, PC3, σ_PC1_20d]
        pc1_vol = pc_panel["PC1"].rolling(20, min_periods=5).std()
        feat_df = pd.concat([
            pc_panel[["PC1", "PC2", "PC3"]],
            pc1_vol.rename("sigma_PC1_20d"),
        ], axis=1).dropna()
        if len(feat_df) >= 60:
            features_history = feat_df.values
            asof_history = list(feat_df.index)
            # Today's feature vector = last row
            features_today = features_history[-1]
            # K-NN search once; reuse for every instrument
            try:
                analog_result = knn_analog_search(
                    features_today, features_history[:-1],
                    asof, asof_history[:-1],
                    k=50, time_decay_h=252.0, exclusion_d=60)
                out["_analog_knn_result"] = analog_result
                out["_analog_asof_history"] = asof_history[:-1]
            except Exception as _e:
                analog_result = None
            # Per outright: compute analog FV using the change-residual series
            if analog_result is not None and not out["residual_outrights"].empty:
                from lib.pca_analogs import _classify_cumulative_move
                # Reconstruct per-symbol level-residual series for each outright
                # (we already have today's value in residual_outrights; for the
                # historical series we need to rerun the level computation).
                yield_panel = _outright_yield_bp_panel(wide_close, outright_symbols)
                delta_yield_panel = yield_panel.diff()
                pcs_aligned = pc_panel.reindex(yield_panel.index)
                for sym in outright_symbols:
                    load_inst = _instrument_loadings(sym, "outright", pca_fit_static, asof)
                    if load_inst is None:
                        continue
                    tau = _outright_tenor_months(sym, asof)
                    if tau is None:
                        continue
                    mean_inst = float(_pchip_curve(
                        np.asarray(pca_fit_static.tenors_months, dtype=float),
                        np.asarray(pca_fit_static.feature_mean, dtype=float),
                        np.array([tau]))[0])
                    recon = np.full(len(yield_panel), float(mean_inst))
                    for k in range(min(3, pca_fit_static.loadings.shape[0])):
                        col = f"PC{k + 1}"
                        if col in pcs_aligned.columns:
                            vals = pcs_aligned[col].values * load_inst[k]
                            recon = recon + np.where(np.isnan(vals), 0.0, vals)
                    delta_observed = delta_yield_panel[sym].values
                    change_residual = delta_observed - recon
                    change_series = pd.Series(change_residual, index=yield_panel.index).dropna()
                    level_series = change_series.cumsum()
                    rolling_mean = level_series.rolling(252, min_periods=30).mean()
                    level_series = (level_series - rolling_mean).dropna()
                    if len(level_series) < 30:
                        continue
                    try:
                        today_residual = float(level_series.iloc[-1])
                    except Exception:
                        continue
                    try:
                        fv_band = analog_fv_band(
                            level_series, analog_result,
                            asof_history[:-1], today_residual)
                        out["analog_fv_results"][sym] = fv_band
                    except Exception:
                        pass
                    # Path-conditional FV — wired below via fit_step_path_bootstrap
                    # + build_policy_path_history_from_fdtr (computed once per panel,
                    # then applied per-symbol).
    except Exception as e:
        # Analog/path FV failure shouldn't crash the engine
        pass

    # ─── 10c. Path-FV wiring (Phase A5) ─────────────────────────────────────
    # Heitfield-Park step-path → today's policy-path probabilities.
    # build_policy_path_history_from_fdtr → realized policy-step history.
    # path_conditional_fv → per-symbol bucket-mixture FV.
    # All gated by try/except since step-path needs SOFR fixings and FDTR data.
    try:
        from lib.pca_step_path import (fit_step_path_bootstrap,
                                            build_policy_path_history_from_fdtr)
        from lib.pca_analogs import path_conditional_fv as _pcfv
        fomc_for_path = _load_meetings()
        fomc_dates_path = (sorted(pd.Timestamp(d).date() if not isinstance(d, date) else d
                                     for d in fomc_for_path["decision_date"])
                            if not fomc_for_path.empty else [])
        if fomc_dates_path and cmc_panel is not None and not cmc_panel.empty:
            step_path_result = fit_step_path_bootstrap(
                cmc_panel=cmc_panel, fomc_dates=fomc_dates_path,
                asof=asof, n_meetings=8, n_draws=20,
                kernel_sigma_bp=37.5,
            )
            today_probs = step_path_result.get("today_path_probs_combined") or {}
            # Build policy_path_history from SOFR-as-proxy (FDTR not wired here)
            policy_hist = build_policy_path_history_from_fdtr(
                sofr_panel, fomc_dates_path,
            )
            out["policy_path_history"] = policy_hist
            out["today_policy_path_probs"] = today_probs
            out["step_path_meetings"] = step_path_result.get("meetings", [])

            # Per-outright path-FV using the same level_series we computed above
            yield_panel_p = _outright_yield_bp_panel(wide_close, outright_symbols)
            delta_yield_panel_p = yield_panel_p.diff()
            pcs_aligned_p = pc_panel.reindex(yield_panel_p.index)
            fomc_cal_df = pd.DataFrame({"decision_date": fomc_dates_path})
            for sym in outright_symbols:
                if not out["analog_fv_results"].get(sym):
                    continue  # analog FV already failed — path-FV will too
                load_inst_p = _instrument_loadings(sym, "outright", pca_fit_static, asof)
                if load_inst_p is None:
                    continue
                tau_p = _outright_tenor_months(sym, asof)
                if tau_p is None:
                    continue
                mean_inst_p = float(_pchip_curve(
                    np.asarray(pca_fit_static.tenors_months, dtype=float),
                    np.asarray(pca_fit_static.feature_mean, dtype=float),
                    np.array([tau_p]))[0])
                recon_p = np.full(len(yield_panel_p), float(mean_inst_p))
                for k in range(min(3, pca_fit_static.loadings.shape[0])):
                    col_p = f"PC{k + 1}"
                    if col_p in pcs_aligned_p.columns:
                        vals_p = pcs_aligned_p[col_p].values * load_inst_p[k]
                        recon_p = recon_p + np.where(np.isnan(vals_p), 0.0, vals_p)
                delta_obs_p = delta_yield_panel_p[sym].values
                change_resid_p = delta_obs_p - recon_p
                change_ser_p = pd.Series(change_resid_p, index=yield_panel_p.index).dropna()
                level_ser_p = change_ser_p.cumsum()
                roll_mean_p = level_ser_p.rolling(252, min_periods=30).mean()
                level_ser_p = (level_ser_p - roll_mean_p).dropna()
                if len(level_ser_p) < 60:
                    continue
                try:
                    today_res_p = float(level_ser_p.iloc[-1])
                except Exception:
                    continue
                # Re-use the analog k-NN result (computed once above)
                analog_result_p = out.get("_analog_knn_result")
                asof_history_p = out.get("_analog_asof_history") or []
                if analog_result_p is None or not asof_history_p:
                    continue
                try:
                    pcfv = _pcfv(
                        analog_result=analog_result_p,
                        structure_residual_series=level_ser_p,
                        asof_history=asof_history_p,
                        structure_window_end=asof,
                        fomc_calendar=fomc_cal_df,
                        policy_path_history=policy_hist,
                        today_policy_path_probs=today_probs,
                        today_residual=today_res_p,
                    )
                    out["path_fv_results"][sym] = pcfv
                except Exception:
                    continue
    except Exception:
        # Path-FV failure shouldn't crash the engine
        pass

    # Clean up private temp keys leaked from analog/path-FV computation
    out.pop("_analog_knn_result", None)
    out.pop("_analog_asof_history", None)

    # ─── 11. Regime + cycle (market-aware central bank calendar) ────────────
    try:
        fomc_calendar = _load_meetings()
    except Exception:
        fomc_calendar = pd.DataFrame()
    out["regime_label_series"] = regime_label_panel(cmc_panel, outright_symbols)
    out["cycle_phase"] = cycle_phase_tag(pc_panel, fomc_calendar, asof)

    # ─── 12. Diagnostics ────────────────────────────────────────────────────
    out["reconstruction_error"] = reconstruction_error_series(cmc_panel, rolling, pc_panel)
    out["reconstruction_pct_today"] = reconstruction_pct_explained(
        cmc_panel, rolling, pc_panel, asof)
    out["variance_explained_history"] = variance_explained_rolling(rolling)
    out["outlier_days"] = detect_outlier_days(out["reconstruction_error"],
                                                percentile=anomaly_pct)
    return out


def build_full_pca_panel(asof: date,
                          tenor_grid_months: list = None,
                          lookbacks: tuple = (5, 15, 30, 60, 90, 252),
                          window_mode: str = "rolling_252",
                          refit_step_days: int = 5,
                          history_days: Optional[int] = None,
                          sparse_enabled: bool = True,
                          anomaly_pct: float = 99.0,
                          mode: Optional[str] = None,
                          resample: str = "D",
                          base_product: str = "SRA") -> dict:
    """Public entry point for the UI layer. Wraps the heavy compute kernel.

    `base_product` selects the market (SRA / ER / FSR / FER / SON / YBA / CRA).
    Defaults to "SRA" for back-compat with the original US-only tab.

    Mode (`"intraday" | "swing" | "positional"`) selects horizon-sensitive
    parameters from MODE_PARAMS. Default is `DEFAULT_MODE` ("positional").

    `history_days` defaults to `mode_params(mode)["history_days"]` if omitted.
    `resample="W"` enables weekly residual resampling.
    """
    mode = _resolve_mode(mode)
    mp = mode_params(mode)
    if history_days is None:
        history_days = int(mp["history_days"])
    if tenor_grid_months is None:
        tenor_grid_months = list(DEFAULT_TENOR_GRID_MONTHS)
    return _build_pca_panel_internal(
        asof=asof,
        tenor_grid_months=list(tenor_grid_months),
        lookbacks=tuple(lookbacks),
        window_mode=window_mode,
        refit_step_days=refit_step_days,
        history_days=int(history_days),
        sparse_enabled=sparse_enabled,
        anomaly_pct=anomaly_pct,
        mode=mode,
        resample=resample,
        base_product=base_product,
    )


# =============================================================================
# PHASE 1 EXTENSIONS — feeds Tier 2 (cross-factor) and Tier 3 (multi-resolution)
# in the trade-finder catalogue.
# =============================================================================

# Tenor segment definitions (months). Front captures FOMC step structure,
# belly captures cycle pricing, back captures terminal-rate beliefs.
SEGMENT_BOUNDS = {
    "front": (3, 12),     # 3M..12M inclusive
    "belly": (12, 36),    # 12M..36M
    "back":  (36, 60),    # 36M..60M
}


def fit_multi_resolution_pca(cmc_panel: pd.DataFrame, *,
                               segments: tuple = ("front", "belly", "back"),
                               n_components: int = 3) -> dict:
    """PCA fit on tenor segments separately (front / belly / back).

    Full-curve PCA averages out FOMC step structure in the front and
    terminal-rate dynamics at the back. Segment-specific fits expose
    dislocations the full-curve fit hides.

    Returns dict {segment_name: PCAFit | None}. None entries indicate
    insufficient sample for that segment.
    """
    out = {}
    if cmc_panel is None or cmc_panel.empty:
        for s in segments:
            out[s] = None
        return out
    cols = [int(c) for c in cmc_panel.columns]
    for seg in segments:
        if seg not in SEGMENT_BOUNDS:
            out[seg] = None
            continue
        lo, hi = SEGMENT_BOUNDS[seg]
        seg_cols = [c for c in cols if lo <= c <= hi]
        if len(seg_cols) < max(3, n_components + 1):
            out[seg] = None
            continue
        sub = cmc_panel[seg_cols]
        try:
            fit = fit_pca_static(sub, n_components=min(n_components, len(seg_cols) - 1))
        except Exception:
            fit = None
        out[seg] = fit
    return out


def cross_pc_corr_breakdown_signal(cross_corr_df: pd.DataFrame, *,
                                      threshold: float = 0.3) -> pd.Series:
    """Boolean series: True on days where any pairwise rolling corr exceeds threshold.

    A True signals PCA orthogonality is broken — slope and curvature trades
    that ASSUME orthogonality are systematically biased.
    """
    if cross_corr_df is None or cross_corr_df.empty:
        return pd.Series(dtype=bool)
    abs_max = cross_corr_df.abs().max(axis=1)
    return (abs_max > float(threshold)).fillna(False)


def sparse_dense_divergence(dense_fit: PCAFit,
                              sparse_fit: Optional[SparsePCAFit],
                              *, threshold: float = 0.05) -> dict:
    """Per-PC, per-tenor divergence between dense and sparse loadings.

    A tenor where dense_loading is large but sparse_loading is zero (held up
    by L1 penalty) is being supported by noise in the dense fit. Divergence
    above `threshold` flags the tenor as potentially mispriced.

    Returns:
      {
        "PC1": {tenor: {"dense": float, "sparse": float, "diff": float, "flagged": bool}, ...},
        "PC2": ...,
        "PC3": ...,
        "summary": {"PC1_flagged_count": int, "PC2_flagged_count": int, ...},
      }
    """
    out = {"PC1": {}, "PC2": {}, "PC3": {}, "summary": {}}
    if dense_fit is None:
        return out
    if sparse_fit is None:
        for k in range(min(3, dense_fit.loadings.shape[0])):
            pc_key = f"PC{k + 1}"
            for i, t in enumerate(dense_fit.tenors_months):
                out[pc_key][int(t)] = {
                    "dense": float(dense_fit.loadings[k, i]),
                    "sparse": None, "diff": None, "flagged": False,
                }
        return out
    n_pc = min(dense_fit.loadings.shape[0], sparse_fit.loadings.shape[0], 3)
    for k in range(n_pc):
        pc_key = f"PC{k + 1}"
        flagged = 0
        for i, t in enumerate(dense_fit.tenors_months):
            d = float(dense_fit.loadings[k, i])
            s = float(sparse_fit.loadings[k, i]) if i < sparse_fit.loadings.shape[1] else 0.0
            diff = abs(d - s)
            is_flagged = diff > float(threshold)
            if is_flagged:
                flagged += 1
            out[pc_key][int(t)] = {
                "dense": d, "sparse": s, "diff": diff, "flagged": is_flagged,
            }
        out["summary"][f"{pc_key}_flagged_count"] = flagged
    return out


def variance_ratio_regime(pca_fit: PCAFit) -> str:
    """Label the variance regime based on PC1 share.

    Normal: PC1 ∈ [0.85, 0.93]   — typical Litterman-Scheinkman regime
    Low-PC1: PC1 < 0.85          — slope/curvature elevated, flies pay more
    Elevated-PC1: PC1 > 0.93      — pure level regime, flies dead

    Returns one of: "normal" | "low_pc1" | "elevated_pc1" | "no_fit"
    """
    if pca_fit is None or len(pca_fit.variance_ratio) < 1:
        return "no_fit"
    pc1_share = float(pca_fit.variance_ratio[0])
    if pc1_share < 0.85:
        return "low_pc1"
    if pc1_share > 0.93:
        return "elevated_pc1"
    return "normal"


def variance_ratio_regime_history(rolling_fits: dict) -> pd.DataFrame:
    """Time series of variance regime label per refit date.

    Returns DataFrame indexed by refit date with columns
    [pc1_var, regime_label, regime_persists_d].
    """
    rows = []
    if not rolling_fits:
        return pd.DataFrame(columns=["pc1_var", "regime_label", "regime_persists_d"])
    prior_label = None
    persists = 0
    for d, fit in sorted(rolling_fits.items()):
        label = variance_ratio_regime(fit)
        if label == prior_label:
            persists += 1
        else:
            persists = 0
        prior_label = label
        rows.append({
            "asof": pd.Timestamp(d),
            "pc1_var": float(fit.variance_ratio[0]) if len(fit.variance_ratio) >= 1 else None,
            "regime_label": label,
            "regime_persists_d": persists,
        })
    return pd.DataFrame(rows).set_index("asof")


def eigenspectrum_gap(rolling_fits: dict) -> pd.DataFrame:
    """Per-refit PC2/PC3 eigenvalue gap.

    When the gap closes (ratio approaches 1), PC2 and PC3 become statistically
    indistinguishable — PC3-fly residual interpretation is unreliable until gap
    reopens. Per gameplan §A1.

    Returns DataFrame indexed by refit date with columns
    [pc2_eig, pc3_eig, gap_ratio, gap_alert].
    `gap_alert` = True when ratio < 1.5 (configurable threshold).
    """
    if not rolling_fits:
        return pd.DataFrame(columns=["pc2_eig", "pc3_eig", "gap_ratio", "gap_alert"])
    rows = []
    for d, fit in sorted(rolling_fits.items()):
        if len(fit.eigenvalues) < 3:
            continue
        e2 = float(fit.eigenvalues[1])
        e3 = float(fit.eigenvalues[2])
        if e3 <= 0 or not np.isfinite(e3):
            ratio = None
        else:
            ratio = e2 / e3
        rows.append({
            "asof": pd.Timestamp(d),
            "pc2_eig": e2,
            "pc3_eig": e3,
            "gap_ratio": ratio,
            "gap_alert": (ratio is not None and ratio < 1.5),
        })
    if not rows:
        return pd.DataFrame(columns=["pc2_eig", "pc3_eig", "gap_ratio", "gap_alert"])
    return pd.DataFrame(rows).set_index("asof")


def pc1_loading_asymmetry(pca_fit: PCAFit) -> dict:
    """Front vs back PC1 loadings asymmetry — "front-led" vs "back-led" regime.

    PC1 should be roughly flat-positive (level). When front loadings >> back
    loadings, the curve is in a "front-led" regime — hedge effectiveness of
    PC1-isolated baskets shifts; back-led contracts are noisier proxies.

    Returns:
      {
        "front_mean": avg PC1 loading on first third of tenors,
        "back_mean": avg on last third,
        "asymmetry": front_mean - back_mean (positive = front-led),
        "regime": "front_led" | "back_led" | "balanced",
      }
    """
    if pca_fit is None or pca_fit.loadings.shape[0] < 1:
        return {"front_mean": None, "back_mean": None,
                "asymmetry": None, "regime": "no_fit"}
    n_tenors = pca_fit.loadings.shape[1]
    third = max(1, n_tenors // 3)
    pc1 = pca_fit.loadings[0]
    front_mean = float(np.mean(pc1[:third]))
    back_mean = float(np.mean(pc1[-third:]))
    asym = front_mean - back_mean
    if asym > 0.04:
        regime = "front_led"
    elif asym < -0.04:
        regime = "back_led"
    else:
        regime = "balanced"
    return {
        "front_mean": front_mean,
        "back_mean": back_mean,
        "asymmetry": asym,
        "regime": regime,
    }
