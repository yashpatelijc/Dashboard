"""Analog fair-value bands for PCA-curve trades — gameplan §A2 + §A3.

Implements:
  * Mahalanobis distance on PCA features with Ledoit-Wolf shrunk covariance
  * KNN with ±60d temporal exclusion + exponential time decay (H=250d default)
  * Weighted-median FV + IQR band per structure
  * Path-conditional bucketing by realized FOMC policy moves inside structure window
  * Markov-chain marginal fallback for sparse buckets

A2 (analog FV) and A3 (path-conditional FV) provide TWO orthogonal alternative
fair-value views per structure:
  * A2: "what did this structure trade at in similar past states?"
  * A3: "what did it trade at in similar past states GIVEN the priced policy path?"

Both are consumed by the trade-idea generator as conviction-blend inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd


# =============================================================================
# Types
# =============================================================================
@dataclass(frozen=True)
class AnalogResult:
    target_idx: int                      # row index of target day in history
    analog_indices: np.ndarray           # selected K analog row indices
    weights: np.ndarray                  # per-analog combined weight (Mahalanobis × time decay)
    mahalanobis: np.ndarray              # raw Mahalanobis distances per analog
    time_weights: np.ndarray             # per-analog time-decay weights
    eff_n: float                         # (Σw)² / Σw²


@dataclass(frozen=True)
class AnalogFV:
    fv_bp: float
    band_lo_bp: float
    band_hi_bp: float
    residual_today_bp: float
    residual_z: float
    percentile_rank: float                # ∈ [0, 1]
    eff_n: float
    n_analogs: int
    gate_quality: str                     # "clean" | "low_n" | "no_data"


@dataclass(frozen=True)
class PathConditionalFV:
    fv_bp: float
    band_lo_bp: float
    band_hi_bp: float
    bucket_probs: dict                    # {bucket_label: prob_today}
    bucket_fv: dict                       # {bucket_label: conditional FV}
    bucket_eff_n: dict                    # {bucket_label: eff_n}
    eff_n_overall: float
    sparse_bucket_fallback_used: bool
    gate_quality: str


# =============================================================================
# A2.1 — Ledoit-Wolf shrinkage estimator (Ledoit & Wolf 2004)
# =============================================================================
def ledoit_wolf_shrinkage(X: np.ndarray) -> tuple:
    """Ledoit-Wolf shrunk covariance: Σ̂_LW = (1-α) Σ̂_sample + α F.

    `F` is a structured target — here we use scaled identity:
      F = (trace(Σ̂)/d) * I

    Returns (shrunk_cov, alpha, raw_cov).
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    if n < 2 or d < 1:
        return np.eye(d), 0.0, np.eye(d)
    # Center
    mu = X.mean(axis=0)
    Xc = X - mu
    # Sample covariance (n-1 divisor for unbiased)
    sample_cov = (Xc.T @ Xc) / (n - 1)
    # Target: scaled identity
    mu_target = float(np.trace(sample_cov)) / d
    F = mu_target * np.eye(d)

    # Optimal shrinkage intensity
    # π̂ = sum over (i,j) of var(s_ij)
    # where s_ij = (1/n) sum_t (Xc_t,i * Xc_t,j)
    # var(s_ij) ≈ (1/n) sum_t (Xc_t,i * Xc_t,j - sample_cov[i,j])²
    pi_mat = np.zeros((d, d))
    for t in range(n):
        outer = np.outer(Xc[t], Xc[t])
        pi_mat += (outer - sample_cov) ** 2
    pi_mat /= n
    pi_hat = float(pi_mat.sum())

    # γ̂ = ‖Σ̂ − F‖_F²
    gamma_hat = float(np.sum((sample_cov - F) ** 2))

    if gamma_hat < 1e-12:
        alpha = 1.0
    else:
        alpha = max(0.0, min(1.0, pi_hat / gamma_hat / n))

    shrunk = (1 - alpha) * sample_cov + alpha * F
    # Ensure PD
    eigvals = np.linalg.eigvalsh(shrunk)
    if eigvals.min() < 1e-8:
        shrunk = shrunk + (1e-6 - eigvals.min()) * np.eye(d)
    return shrunk, float(alpha), sample_cov


# =============================================================================
# A2.2 — Mahalanobis distance vectorized
# =============================================================================
def mahalanobis_distance(x_target: np.ndarray,
                          X_history: np.ndarray,
                          sigma_inv: np.ndarray) -> np.ndarray:
    """Vector of Mahalanobis distances from x_target to each row of X_history.

    `x_target` shape (D,), `X_history` shape (N, D), `sigma_inv` shape (D, D).
    Returns array of length N.
    """
    diff = X_history - x_target[None, :]
    # d²[i] = diff[i] @ sigma_inv @ diff[i]
    d2 = np.einsum("ij,jk,ik->i", diff, sigma_inv, diff)
    d2 = np.clip(d2, 0.0, None)
    return np.sqrt(d2)


# =============================================================================
# A2.3 — KNN analog search
# =============================================================================
def knn_analog_search(features_today: np.ndarray,
                       features_history: np.ndarray,
                       asof_today: date,
                       asof_history: list,
                       *,
                       k: int = 50,
                       time_decay_h: float = 250.0,
                       exclusion_d: int = 60,
                       sigma_inv: Optional[np.ndarray] = None) -> AnalogResult:
    """KNN search in PCA feature space with temporal exclusion + time decay.

    `features_today` shape (D,), `features_history` shape (N, D),
    `asof_history` list of dates aligned to rows of features_history.

    Returns AnalogResult with selected K nearest analogs.
    `sigma_inv` (optional) precomputed inverse covariance (Ledoit-Wolf).
    """
    X = np.asarray(features_history, dtype=float)
    n, d = X.shape
    asof_history_arr = np.asarray([pd.Timestamp(x).date() for x in asof_history])

    # Compute Σ̂_LW⁻¹ if not provided
    if sigma_inv is None:
        sigma_lw, _alpha, _raw = ledoit_wolf_shrinkage(X)
        sigma_inv = np.linalg.pinv(sigma_lw)

    # Mahalanobis distance
    d_mahal = mahalanobis_distance(features_today, X, sigma_inv)

    # Temporal exclusion mask
    today_ord = pd.Timestamp(asof_today).toordinal()
    hist_ord = np.array([pd.Timestamp(d).toordinal() for d in asof_history_arr])
    days_diff = np.abs(hist_ord - today_ord)
    excluded = days_diff <= exclusion_d
    eligible = ~excluded

    if eligible.sum() == 0:
        return AnalogResult(
            target_idx=-1,
            analog_indices=np.array([], dtype=int),
            weights=np.array([]),
            mahalanobis=np.array([]),
            time_weights=np.array([]),
            eff_n=0.0,
        )

    # Time-decay weights: exp(-|days|/H)
    time_w = np.exp(-days_diff / max(time_decay_h, 1.0))
    # Combined weight = exp(-d_mahal) × time_w  (per gameplan §A2.3)
    # Use exp(-d_mahal/scale) where scale normalizes Mahalanobis to ~unit
    scale = float(np.median(d_mahal[eligible])) if eligible.any() else 1.0
    if scale <= 0:
        scale = 1.0
    mahal_w = np.exp(-d_mahal / scale)
    combined = mahal_w * time_w
    combined[~eligible] = 0.0

    # Pick top-k by combined weight
    n_eligible = int(eligible.sum())
    K = min(int(k), n_eligible)
    if K < 1:
        return AnalogResult(
            target_idx=-1,
            analog_indices=np.array([], dtype=int),
            weights=np.array([]),
            mahalanobis=np.array([]),
            time_weights=np.array([]),
            eff_n=0.0,
        )
    # argpartition gives top-K by largest combined weight
    top_idx = np.argpartition(-combined, K - 1)[:K]
    # Sort selected by descending weight
    top_idx = top_idx[np.argsort(-combined[top_idx])]

    selected_w = combined[top_idx]
    # Effective n: (Σw)² / Σw²
    sum_w = float(selected_w.sum())
    sum_w2 = float((selected_w ** 2).sum())
    eff_n = (sum_w ** 2) / max(sum_w2, 1e-12)

    return AnalogResult(
        target_idx=-1,    # no target row in history; today is "outside"
        analog_indices=top_idx,
        weights=selected_w,
        mahalanobis=d_mahal[top_idx],
        time_weights=time_w[top_idx],
        eff_n=eff_n,
    )


# =============================================================================
# A2.4 — Weighted-median FV and IQR band per structure
# =============================================================================
def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median: smallest x such that cum(w sorted by x) ≥ 0.5·Σw."""
    if len(values) == 0:
        return float("nan")
    order = np.argsort(values)
    v_sorted = values[order]
    w_sorted = weights[order]
    cum = np.cumsum(w_sorted)
    total = cum[-1]
    if total <= 0:
        return float(np.median(v_sorted))
    target = 0.5 * total
    i = int(np.searchsorted(cum, target))
    i = min(i, len(v_sorted) - 1)
    return float(v_sorted[i])


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Weighted q-quantile, q ∈ [0, 1]."""
    if len(values) == 0:
        return float("nan")
    order = np.argsort(values)
    v_sorted = values[order]
    w_sorted = weights[order]
    cum = np.cumsum(w_sorted)
    total = cum[-1]
    if total <= 0:
        return float(np.quantile(v_sorted, q))
    target = q * total
    i = int(np.searchsorted(cum, target))
    i = min(i, len(v_sorted) - 1)
    return float(v_sorted[i])


def analog_fv_band(structure_residual_series: pd.Series,
                    knn_result: AnalogResult,
                    asof_history: list,
                    today_residual: Optional[float],
                    *,
                    eff_n_floor: int = 30) -> AnalogFV:
    """Compute analog FV band from KNN result and structure residual history.

    `structure_residual_series` — historical residuals of the structure (in bp).
    `knn_result` — output of knn_analog_search.
    `asof_history` — full date list parallel to features_history (used to map
                     analog indices back to dates).
    `today_residual` — today's residual value (bp). May be None.
    """
    if knn_result.eff_n < 1 or len(knn_result.analog_indices) == 0:
        return AnalogFV(
            fv_bp=float("nan"), band_lo_bp=float("nan"), band_hi_bp=float("nan"),
            residual_today_bp=float("nan") if today_residual is None else float(today_residual),
            residual_z=float("nan"), percentile_rank=float("nan"),
            eff_n=0.0, n_analogs=0, gate_quality="no_data",
        )

    # Map analog indices → dates → residual values
    analog_dates = [pd.Timestamp(asof_history[i]).date() for i in knn_result.analog_indices]
    series_idx = pd.to_datetime([d for d in analog_dates])
    res_series = structure_residual_series.copy()
    res_series.index = pd.to_datetime(res_series.index)
    aligned = res_series.reindex(series_idx).values
    weights = knn_result.weights
    # Drop NaNs
    mask = ~np.isnan(aligned)
    if mask.sum() == 0:
        return AnalogFV(
            fv_bp=float("nan"), band_lo_bp=float("nan"), band_hi_bp=float("nan"),
            residual_today_bp=float("nan") if today_residual is None else float(today_residual),
            residual_z=float("nan"), percentile_rank=float("nan"),
            eff_n=0.0, n_analogs=0, gate_quality="no_data",
        )
    vals = aligned[mask]
    w = weights[mask]
    fv = _weighted_median(vals, w)
    band_lo = _weighted_quantile(vals, w, 0.25)
    band_hi = _weighted_quantile(vals, w, 0.75)
    # Z and percentile rank for today's residual
    if today_residual is None or not np.isfinite(today_residual):
        z = float("nan")
        pct = float("nan")
    else:
        # Std of analogs
        std = np.sqrt(np.average((vals - fv) ** 2, weights=w))
        z = (today_residual - fv) / std if std > 1e-12 else 0.0
        # Percentile rank: weighted fraction of analogs ≤ today
        below_mask = vals <= today_residual
        pct = float(w[below_mask].sum() / w.sum()) if w.sum() > 0 else 0.5

    eff_n = knn_result.eff_n * (mask.sum() / len(mask))    # Adjust for NaN drops
    gate = "clean" if eff_n >= eff_n_floor else "low_n"
    return AnalogFV(
        fv_bp=float(fv),
        band_lo_bp=float(band_lo),
        band_hi_bp=float(band_hi),
        residual_today_bp=float(today_residual) if today_residual is not None and np.isfinite(today_residual) else float("nan"),
        residual_z=float(z),
        percentile_rank=float(pct),
        eff_n=float(eff_n),
        n_analogs=int(mask.sum()),
        gate_quality=gate,
    )


# =============================================================================
# A3.1 — Path-conditional bucketing by FOMC policy moves inside structure window
# =============================================================================
# 5-outcome lattice per gameplan §A3.4:
PATH_BUCKETS_5 = ("large_cut", "cut", "hold", "hike", "large_hike")

LATTICE_BP = {
    "large_cut": -50.0,
    "cut": -25.0,
    "hold": 0.0,
    "hike": 25.0,
    "large_hike": 50.0,
}


def _classify_cumulative_move(cum_bp: float, tolerance_bp: float = 12.5) -> str:
    """Classify cumulative bp move into the 5-outcome lattice."""
    if cum_bp <= -37.5:
        return "large_cut"
    if cum_bp <= -tolerance_bp:
        return "cut"
    if cum_bp >= 37.5:
        return "large_hike"
    if cum_bp >= tolerance_bp:
        return "hike"
    return "hold"


def path_bucket_assignment(analog_dates: list,
                            structure_window_end: date,
                            fomc_calendar: pd.DataFrame,
                            policy_path_history: pd.DataFrame) -> dict:
    """For each analog date, classify the realized cumulative policy move
    over the FOMC meetings between that date and `structure_window_end`.

    `policy_path_history` — DataFrame indexed by FOMC date, with column
    `target_change_bp` (realised target-rate change at that meeting).

    Returns {analog_date: bucket_label}.
    """
    out = {}
    if fomc_calendar is None or fomc_calendar.empty or "decision_date" in fomc_calendar.columns is False:
        col = "decision_date" if "decision_date" in (fomc_calendar.columns if fomc_calendar is not None else []) else None
    else:
        col = "decision_date"
    if fomc_calendar is None or fomc_calendar.empty:
        for d in analog_dates:
            out[pd.Timestamp(d).date()] = "hold"
        return out

    fomc_dates = sorted(pd.Timestamp(d).date() for d in fomc_calendar[col])
    end = pd.Timestamp(structure_window_end).date()

    for d in analog_dates:
        anchor = pd.Timestamp(d).date()
        meetings = [m for m in fomc_dates if anchor < m <= end]
        cum_bp = 0.0
        for m in meetings:
            ts = pd.Timestamp(m)
            if (policy_path_history is not None and
                    not policy_path_history.empty and
                    ts in policy_path_history.index and
                    "target_change_bp" in policy_path_history.columns):
                ch = policy_path_history.loc[ts, "target_change_bp"]
                if pd.notna(ch):
                    cum_bp += float(ch)
        out[anchor] = _classify_cumulative_move(cum_bp)
    return out


# =============================================================================
# A3.2 — Markov-chain marginal fallback for sparse buckets
# =============================================================================
def _empirical_markov_transition(policy_path_history: pd.DataFrame,
                                    n_meetings: int,
                                    *,
                                    lookback_meetings: int = 12) -> dict:
    """Compute marginal probabilities of cumulative-move buckets after n_meetings,
    based on the trailing `lookback_meetings` of meeting outcomes.

    Returns {bucket: prob}.
    """
    out = {b: 0.0 for b in PATH_BUCKETS_5}
    if (policy_path_history is None or policy_path_history.empty or
            "target_change_bp" not in policy_path_history.columns):
        out["hold"] = 1.0
        return out
    recent = policy_path_history["target_change_bp"].dropna().iloc[-lookback_meetings:].values
    if len(recent) == 0:
        out["hold"] = 1.0
        return out
    # Per-meeting empirical move distribution
    moves = recent
    # Simulate n_meetings draws and bucket cumulative
    n_sim = 1000
    rng = np.random.default_rng(123)
    counts = {b: 0 for b in PATH_BUCKETS_5}
    for _ in range(n_sim):
        cum = 0.0
        for _m in range(n_meetings):
            cum += float(rng.choice(moves))
        counts[_classify_cumulative_move(cum)] += 1
    total = max(1, sum(counts.values()))
    for b in PATH_BUCKETS_5:
        out[b] = counts[b] / total
    return out


def path_conditional_fv(analog_result: AnalogResult,
                          structure_residual_series: pd.Series,
                          asof_history: list,
                          today_residual: Optional[float],
                          structure_window_end: date,
                          fomc_calendar: pd.DataFrame,
                          policy_path_history: pd.DataFrame,
                          today_policy_path_probs: dict,
                          *,
                          sparse_bucket_min: int = 10,
                          eff_n_floor: int = 30) -> PathConditionalFV:
    """Path-conditional FV per gameplan §A3.

    `today_policy_path_probs` — dict {bucket_label: prob} reflecting today's
    A4 step-path output. Should sum to 1.

    Per-bucket conditional FV is computed from analog residuals filtered to
    that bucket. Sparse buckets (< sparse_bucket_min analogs) fall back to the
    Markov-chain marginal (recent empirical).

    Final FV = Σ_bucket prob_bucket × bucket_fv.
    """
    if analog_result.eff_n < 1 or len(analog_result.analog_indices) == 0:
        return PathConditionalFV(
            fv_bp=float("nan"), band_lo_bp=float("nan"), band_hi_bp=float("nan"),
            bucket_probs={}, bucket_fv={}, bucket_eff_n={},
            eff_n_overall=0.0, sparse_bucket_fallback_used=False,
            gate_quality="no_data",
        )

    # 1. Analog dates and residuals
    analog_dates = [pd.Timestamp(asof_history[i]).date() for i in analog_result.analog_indices]
    series_idx = pd.to_datetime(analog_dates)
    res_series = structure_residual_series.copy()
    res_series.index = pd.to_datetime(res_series.index)
    aligned = res_series.reindex(series_idx).values
    weights = analog_result.weights
    mask = ~np.isnan(aligned)

    # 2. Bucket each analog by realized policy moves
    buckets = path_bucket_assignment(
        [d for d, m in zip(analog_dates, mask) if m],
        structure_window_end,
        fomc_calendar,
        policy_path_history,
    )

    # 3. Per-bucket weighted median FV
    bucket_fv = {}
    bucket_eff_n = {}
    fallback_used = False
    n_meetings_in_window = 0
    if fomc_calendar is not None and not fomc_calendar.empty:
        col = "decision_date" if "decision_date" in fomc_calendar.columns else fomc_calendar.columns[0]
        fomc_dates = sorted(pd.Timestamp(d).date() for d in fomc_calendar[col])
        end = pd.Timestamp(structure_window_end).date()
        # Heuristic: count meetings between today and end
        # (used for Markov fallback)
        n_meetings_in_window = sum(1 for m in fomc_dates if m <= end and m >= date.today())

    valid_dates = [d for d, m in zip(analog_dates, mask) if m]
    valid_vals = aligned[mask]
    valid_w = weights[mask]

    overall_fv = 0.0
    overall_lo = 0.0
    overall_hi = 0.0
    overall_eff_n = 0.0
    total_prob = 0.0

    for bucket in PATH_BUCKETS_5:
        # Indices of analogs in this bucket
        b_mask = np.array([buckets.get(d, "hold") == bucket for d in valid_dates])
        n_in_bucket = int(b_mask.sum())
        if n_in_bucket >= sparse_bucket_min:
            v = valid_vals[b_mask]
            w = valid_w[b_mask]
            fv = _weighted_median(v, w)
            lo = _weighted_quantile(v, w, 0.25)
            hi = _weighted_quantile(v, w, 0.75)
            sum_w = float(w.sum())
            sum_w2 = float((w ** 2).sum())
            eff = (sum_w ** 2) / max(sum_w2, 1e-12)
        else:
            # Fallback: use the OVERALL weighted median as the bucket FV
            # (Markov-marginal fallback per gameplan §A3.4)
            fv = _weighted_median(valid_vals, valid_w)
            lo = _weighted_quantile(valid_vals, valid_w, 0.25)
            hi = _weighted_quantile(valid_vals, valid_w, 0.75)
            eff = float(n_in_bucket)
            fallback_used = True
        bucket_fv[bucket] = fv
        bucket_eff_n[bucket] = eff

        prob = float(today_policy_path_probs.get(bucket, 0.0))
        if prob > 0 and np.isfinite(fv):
            overall_fv += prob * fv
            overall_lo += prob * lo
            overall_hi += prob * hi
            overall_eff_n += prob * eff
            total_prob += prob

    if total_prob > 0:
        overall_fv /= total_prob
        overall_lo /= total_prob
        overall_hi /= total_prob
        overall_eff_n /= total_prob

    gate = "clean" if overall_eff_n >= eff_n_floor and not fallback_used else "low_n"
    if total_prob == 0:
        gate = "no_data"

    return PathConditionalFV(
        fv_bp=float(overall_fv),
        band_lo_bp=float(overall_lo),
        band_hi_bp=float(overall_hi),
        bucket_probs=dict(today_policy_path_probs),
        bucket_fv=bucket_fv,
        bucket_eff_n=bucket_eff_n,
        eff_n_overall=float(overall_eff_n),
        sparse_bucket_fallback_used=fallback_used,
        gate_quality=gate,
    )


# =============================================================================
# Convenience: run analog FV for many structures at once
# =============================================================================
def batch_analog_fv(structures: dict,
                     features_today: np.ndarray,
                     features_history: np.ndarray,
                     asof_today: date,
                     asof_history: list,
                     *,
                     k: int = 50,
                     time_decay_h: float = 250.0,
                     exclusion_d: int = 60) -> dict:
    """Run KNN once, compute FV for each structure.

    `structures` — dict {struct_id: pd.Series of historical residuals}.
    Returns dict {struct_id: AnalogFV}.
    """
    sigma_lw, _alpha, _raw = ledoit_wolf_shrinkage(features_history)
    sigma_inv = np.linalg.pinv(sigma_lw)
    knn_result = knn_analog_search(
        features_today=features_today,
        features_history=features_history,
        asof_today=asof_today,
        asof_history=asof_history,
        k=k,
        time_decay_h=time_decay_h,
        exclusion_d=exclusion_d,
        sigma_inv=sigma_inv,
    )
    out = {}
    for sid, residual_series in structures.items():
        # Today's residual = last value of the residual series (if recent)
        today_res = None
        try:
            ts_today = pd.Timestamp(asof_today)
            res_idx = pd.to_datetime(residual_series.index)
            mask_today = (res_idx == ts_today)
            if mask_today.any():
                today_res = float(residual_series.values[mask_today][0])
        except Exception:
            today_res = None
        out[sid] = analog_fv_band(residual_series, knn_result, asof_history, today_res)
    return out
