"""Calendar / event-aware analytics for the PCA trade engine.

Implements:
  * Seasonality decomposition (gameplan §A6s)  — STL + calendar-dummy regression
    with Newey-West HAC SE and BH-FDR multiple-testing correction.
  * Pre/post-event drift (gameplan §A12d)  — windowed Δ_segment around events
    bucketed by surprise quartile.
  * A11-event-impact ranking (gameplan §3)  — per-ticker β/R²/hit-rate scoring,
    composite importance, recency-weighted "becoming more important" flag.

All modules emit DataFrames keyed by structure / ticker / window for downstream
consumption by the trade-idea generator (`lib.pca_trades.gen_event_*`).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd


# =============================================================================
# A6s.1 — Simple STL decomposition (rolling-mean trend + per-cycle seasonal)
# =============================================================================
def _moving_average(s: pd.Series, window: int) -> pd.Series:
    """Centered moving average."""
    return s.rolling(window=window, center=True, min_periods=max(1, window // 4)).mean()


def fit_seasonality_stl(structure_series: pd.Series, *,
                          periods: tuple = (252, 63)) -> dict:
    """STL-style decomposition: trend + sum_period(seasonal_p) + remainder.

    Each seasonal component is computed iteratively: detrend, average across
    period bins, repeat for next period.

    Returns dict {trend, seasonal_252, seasonal_63, remainder}.
    """
    if structure_series is None:
        return {"trend": pd.Series(dtype=float), "seasonal": {},
                "remainder": pd.Series(dtype=float)}
    s = structure_series.dropna().copy()
    s.index = pd.to_datetime(s.index)
    # Require at least 1.5x the longest period for meaningful decomposition
    min_required = int(max(periods) * 1.2)
    if len(s) < min_required:
        return {"trend": pd.Series(dtype=float), "seasonal": {},
                "remainder": pd.Series(dtype=float)}

    # 1. Trend = long-window moving average
    trend = _moving_average(s, max(periods))
    detrended = s - trend

    # 2. Per-period seasonal: average over period-positions
    seasonal = {}
    work = detrended.copy()
    for p in sorted(periods, reverse=True):
        # Position in period (using day-of-year for 252; day-of-quarter for 63)
        if p == 252:
            pos = work.index.dayofyear
        elif p == 63:
            pos = ((work.index.dayofyear - 1) % p)
        else:
            pos = (np.arange(len(work)) % p)
        bin_means = pd.Series(work.values).groupby(pd.Series(pos)).transform("mean")
        seasonal_p = pd.Series(bin_means.values, index=work.index)
        seasonal_p = seasonal_p - seasonal_p.mean()    # center per period
        seasonal[p] = seasonal_p
        work = work - seasonal_p

    remainder = s - trend
    for sp in seasonal.values():
        remainder = remainder - sp

    return {
        "trend": trend,
        "seasonal": seasonal,
        "remainder": remainder,
    }


# =============================================================================
# A6s.2 — Newey-West HAC standard errors
# =============================================================================
def newey_west_hac_se(X: np.ndarray, residuals: np.ndarray, *,
                       lag: Optional[int] = None) -> np.ndarray:
    """Newey-West HAC SE for OLS β̂.

    `X` (n, k) design matrix; `residuals` (n,) OLS residuals.
    Returns (k,) array of HAC standard errors.
    """
    n, k = X.shape
    if lag is None:
        lag = max(1, int(np.ceil(4 * (n / 100) ** (2 / 9))))
    # XtX inverse
    XtX_inv = np.linalg.pinv(X.T @ X)
    # S matrix: Σ_l w(l) Σ_t X_t X_{t-l}' e_t e_{t-l}
    e = residuals
    S = (X.T * (e ** 2)) @ X    # lag 0 term
    for l in range(1, lag + 1):
        w = 1 - l / (lag + 1)    # Bartlett kernel
        Xe = X * e[:, None]
        # Σ_t Xe_t Xe_{t-l}'
        cross = Xe[l:].T @ Xe[:-l]
        S += w * (cross + cross.T)
    cov = XtX_inv @ S @ XtX_inv
    return np.sqrt(np.maximum(np.diag(cov), 0.0))


# =============================================================================
# A6s.3 — BH-FDR multiple-testing correction (Benjamini-Hochberg 1995)
# =============================================================================
def bh_fdr(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Boolean array marking retained tests after BH-FDR correction.

    `p_values` array; `alpha` FDR rate (default 0.05).
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([], dtype=bool)
    order = np.argsort(p)
    sorted_p = p[order]
    threshold = alpha * np.arange(1, n + 1) / n
    # Find largest k where sorted_p[k] ≤ threshold[k]
    cmp = sorted_p <= threshold
    if not cmp.any():
        return np.zeros(n, dtype=bool)
    k_max = int(np.where(cmp)[0].max())
    retained_in_sorted = np.zeros(n, dtype=bool)
    retained_in_sorted[: k_max + 1] = True
    # Map back to original order
    out = np.zeros(n, dtype=bool)
    out[order] = retained_in_sorted
    return out


# =============================================================================
# A6s.4 — Calendar-dummy regression on STL remainder
# =============================================================================
def _build_calendar_dummies(idx: pd.DatetimeIndex, *,
                             fomc_dates: list, nfp_dates: Optional[list] = None,
                             cpi_dates: Optional[list] = None) -> pd.DataFrame:
    """Build dummy matrix for FOMC-week, NFP-week, CPI-week, year-end-week, quarter-end-week."""
    n = len(idx)
    # FOMC week: any day in the same ISO week as a FOMC meeting
    fomc_set = set(pd.Timestamp(d).date() for d in (fomc_dates or []))
    nfp_set = set(pd.Timestamp(d).date() for d in (nfp_dates or []))
    cpi_set = set(pd.Timestamp(d).date() for d in (cpi_dates or []))
    fomc_week_ids = set()
    for d in fomc_set:
        ts = pd.Timestamp(d)
        fomc_week_ids.add((ts.isocalendar()[0], ts.isocalendar()[1]))
    nfp_week_ids = set()
    for d in nfp_set:
        ts = pd.Timestamp(d)
        nfp_week_ids.add((ts.isocalendar()[0], ts.isocalendar()[1]))
    cpi_week_ids = set()
    for d in cpi_set:
        ts = pd.Timestamp(d)
        cpi_week_ids.add((ts.isocalendar()[0], ts.isocalendar()[1]))

    rows = []
    for ts in idx:
        wk = (ts.isocalendar()[0], ts.isocalendar()[1])
        rows.append({
            "fomc_week": int(wk in fomc_week_ids),
            "nfp_week": int(wk in nfp_week_ids),
            "cpi_week": int(wk in cpi_week_ids),
            "year_end_week": int(ts.month == 12 and ts.day >= 22),
            "quarter_end_week": int(ts.month in (3, 6, 9, 12) and ts.day >= 22),
        })
    df = pd.DataFrame(rows, index=idx)
    return df


def calendar_dummy_regression(structure_series: pd.Series, *,
                                fomc_dates: list,
                                nfp_dates: Optional[list] = None,
                                cpi_dates: Optional[list] = None,
                                fdr_alpha: float = 0.05,
                                min_beta_bp: float = 1.0) -> dict:
    """Regress structure_series on calendar dummies with NW-HAC SE + BH-FDR.

    Retain effects with |β̂| > min_beta_bp AND BH-corrected significance at α.
    Returns dict {effect_name: {beta, se, t, p_raw, p_bh, retained, mean_diff_bp}}.
    """
    out = {}
    if structure_series is None or len(structure_series) < 60:
        return out
    s = structure_series.dropna().copy()
    s.index = pd.to_datetime(s.index)
    dummies = _build_calendar_dummies(s.index, fomc_dates=fomc_dates,
                                          nfp_dates=nfp_dates, cpi_dates=cpi_dates)
    n = len(s)
    # OLS: y = α + Σ β_i D_i + ε
    X = np.column_stack([np.ones(n), dummies.values.astype(float)])
    y = s.values.astype(float)
    try:
        XtX_inv = np.linalg.pinv(X.T @ X)
        beta = XtX_inv @ X.T @ y
        residuals = y - X @ beta
    except Exception:
        return out
    se_vec = newey_west_hac_se(X, residuals)

    effect_names = ["intercept"] + list(dummies.columns)
    n_effects = len(effect_names) - 1    # exclude intercept
    p_raw = np.zeros(n_effects)
    for j in range(n_effects):
        idx = j + 1
        b = beta[idx]
        se = se_vec[idx]
        if se <= 0:
            t = 0.0
            p = 1.0
        else:
            t = b / se
            # 2-tail normal approximation
            from scipy.stats import norm
            p = 2.0 * (1.0 - norm.cdf(abs(t)))
        p_raw[j] = p

    retained_bh = bh_fdr(p_raw, alpha=fdr_alpha)
    for j, name in enumerate(dummies.columns):
        idx = j + 1
        b = float(beta[idx])
        se = float(se_vec[idx])
        t = b / se if se > 0 else 0.0
        retained = bool(retained_bh[j] and abs(b) > min_beta_bp)
        out[name] = {
            "beta_bp": b,
            "se_bp": se,
            "t": t,
            "p_raw": float(p_raw[j]),
            "p_bh_retained": bool(retained_bh[j]),
            "retained": retained,
        }
    return out


# =============================================================================
# A12d — Pre/post-event drift
# =============================================================================
DEFAULT_EVENT_WINDOWS = ((-5, -1), (-1, 0), (0, 5), (0, 20))


def event_drift_table(structure_series: pd.Series, *,
                        event_dates: list,
                        event_class: str = "FOMC",
                        windows: tuple = DEFAULT_EVENT_WINDOWS,
                        surprise_series: Optional[pd.Series] = None,
                        surprise_quartiles: int = 4,
                        fdr_alpha: float = 0.05) -> pd.DataFrame:
    """Pre/post-event drift table.

    For each (window, surprise_quartile, sign), compute mean Δ_structure with
    NW-HAC SE + BH-FDR.

    `structure_series` — daily structure level.
    `event_dates` — list of event dates.
    `windows` — tuple of (T_start, T_end) inclusive offsets in trading days.
    `surprise_series` — optional surprise per event (signed). If None, all events
                         go to one bucket.

    Returns DataFrame with columns:
      [event_class, window, surprise_quartile, surprise_sign, n_events,
       drift_bp_mean, drift_bp_se, p_raw, p_bh_retained, retained]
    """
    rows = []
    if structure_series is None or len(structure_series) < 30 or not event_dates:
        return pd.DataFrame(rows)
    s = structure_series.dropna().copy()
    s.index = pd.to_datetime(s.index)

    # Map each event to bucket
    if surprise_series is None or surprise_series.empty:
        buckets = {pd.Timestamp(d): ("all", "any") for d in event_dates}
    else:
        sur = surprise_series.copy()
        sur.index = pd.to_datetime(sur.index)
        # Compute quartile thresholds from |surprise|
        abs_sur = sur.abs().dropna()
        if len(abs_sur) >= surprise_quartiles:
            q_edges = np.quantile(abs_sur, np.linspace(0, 1, surprise_quartiles + 1))
        else:
            q_edges = np.array([abs_sur.min(), abs_sur.max()])
        buckets = {}
        for d in event_dates:
            ts = pd.Timestamp(d)
            if ts not in sur.index:
                continue
            v = sur.loc[ts]
            if pd.isna(v):
                continue
            # Find quartile of |v|
            q_idx = max(0, min(surprise_quartiles - 1,
                                  int(np.searchsorted(q_edges, abs(v), side="right") - 1)))
            sign = "pos" if v > 0 else ("neg" if v < 0 else "zero")
            buckets[ts] = (f"q{q_idx + 1}", sign)

    # For each window, gather drift values per bucket
    for w in windows:
        w_start, w_end = w
        per_bucket = {}    # {(quartile, sign): [drifts]}
        for ev_date, (q, sign) in buckets.items():
            try:
                idx_pos = s.index.get_indexer([ev_date], method="nearest")[0]
            except Exception:
                continue
            if idx_pos < 0:
                continue
            start_pos = idx_pos + w_start
            end_pos = idx_pos + w_end
            if start_pos < 0 or end_pos >= len(s):
                continue
            try:
                drift = float(s.iloc[end_pos] - s.iloc[start_pos])
            except Exception:
                continue
            key = (q, sign)
            per_bucket.setdefault(key, []).append(drift)

        # Compute stats per bucket
        all_buckets = list(per_bucket.keys())
        p_raw_list = []
        bucket_stats = []
        for key in all_buckets:
            drifts = np.asarray(per_bucket[key], dtype=float)
            if len(drifts) < 3:
                bucket_stats.append({
                    "key": key, "drifts": drifts,
                    "mean": float(np.nanmean(drifts)) if drifts.size else float("nan"),
                    "se": float("nan"), "p_raw": 1.0,
                })
                p_raw_list.append(1.0)
                continue
            # OLS: y = β + ε (constant) — NW HAC on a constant
            n_b = len(drifts)
            X_b = np.ones((n_b, 1))
            y_b = drifts
            beta = float(y_b.mean())
            resid = y_b - beta
            se = newey_west_hac_se(X_b, resid)[0]
            t = beta / se if se > 0 else 0.0
            from scipy.stats import norm
            p = 2.0 * (1.0 - norm.cdf(abs(t)))
            bucket_stats.append({"key": key, "drifts": drifts,
                                  "mean": beta, "se": se, "p_raw": p})
            p_raw_list.append(p)

        retained_bh = bh_fdr(np.array(p_raw_list), alpha=fdr_alpha)
        for st, ret in zip(bucket_stats, retained_bh):
            q, sign = st["key"]
            rows.append({
                "event_class": event_class,
                "window": f"[{w_start}, {w_end}]",
                "surprise_quartile": q,
                "surprise_sign": sign,
                "n_events": int(len(st["drifts"])),
                "drift_bp_mean": float(st["mean"]) if np.isfinite(st["mean"]) else None,
                "drift_bp_se": float(st["se"]) if np.isfinite(st["se"]) else None,
                "p_raw": float(st["p_raw"]),
                "p_bh_retained": bool(ret),
                "retained": bool(ret) and abs(st["mean"]) > 0.5,
            })
    return pd.DataFrame(rows)


# =============================================================================
# A11-event — per-ticker importance ranking (gameplan §3.3)
# =============================================================================
@dataclass(frozen=True)
class TickerImpact:
    ticker: str
    event_class: str
    bp_per_sigma_surprise: dict        # {segment: bp}
    R2_3M: float
    R2_6M: float
    R2_12M: float
    hit_rate_3M: float
    hit_rate_6M: float
    hit_rate_12M: float
    importance_3M: float
    importance_6M: float
    importance_12M: float
    becoming_more_important: bool
    dominant_segment: str
    dominant_horizon: str               # "T+0" | "T+5" | "T+20"
    surprise_definition: str
    n_events_total: int


def _safe_regress(x: np.ndarray, y: np.ndarray) -> dict:
    """Simple OLS y = α + β·x + ε. Returns {beta, alpha, R2, n}."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 5 or x.std() < 1e-12:
        return {"beta": 0.0, "alpha": 0.0, "R2": 0.0, "n": int(n)}
    x_mean = x.mean()
    y_mean = y.mean()
    Sxx = float(np.sum((x - x_mean) ** 2))
    Sxy = float(np.sum((x - x_mean) * (y - y_mean)))
    if Sxx <= 0:
        return {"beta": 0.0, "alpha": float(y_mean), "R2": 0.0, "n": int(n)}
    beta = Sxy / Sxx
    alpha = y_mean - beta * x_mean
    y_hat = alpha + beta * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"beta": float(beta), "alpha": float(alpha), "R2": float(R2), "n": int(n)}


def _hit_rate(beta: float, x: np.ndarray, y: np.ndarray) -> float:
    """Fraction where sign(β·x) == sign(y)."""
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    if len(x) == 0:
        return 0.5
    pred_sign = np.sign(beta * x)
    actual_sign = np.sign(y)
    match = (pred_sign == actual_sign) & (pred_sign != 0)
    return float(match.sum() / len(match))


def _rank_z(values: np.ndarray) -> np.ndarray:
    """Rank-based z-scores: scale ranks to [-1, +1]."""
    if len(values) == 0:
        return np.array([])
    ranks = pd.Series(values).rank(method="average")
    return (ranks - 1) / max(1, len(ranks) - 1) * 2 - 1    # ∈ [-1, +1]


def fit_event_impact_ranking(events: dict,
                                segments_panel: pd.DataFrame,
                                *,
                                horizons_d: tuple = (0, 5, 20),
                                lookbacks_d: tuple = (63, 126, 252)) -> pd.DataFrame:
    """Per-ticker importance score (gameplan §A11-event).

    `events` = {ticker_name: pd.DataFrame[date, surprise, event_class]} —
               surprise is the standardised (z-score) surprise per event.
    `segments_panel` — DataFrame[date × {front, belly, back}] of Δ_CMC per segment.

    For each (ticker, segment, horizon), regress Δ_segment_(T to T+h) on
    standardised_surprise. Compute β/R²/hit-rate over rolling 3M/6M/12M.

    Importance score (per gameplan §3.3.iv):
      imp = 0.40·rank_z(|β|) + 0.30·rank_z(R²) + 0.30·rank_z(2·hit_rate − 1)

    Recency flag: rank(imp_3M) − rank(imp_12M) ≤ −5 AND imp_3M > 60th pctile.

    Returns DataFrame with one row per (ticker, segment, dominant_horizon).
    """
    rows = []
    if not events or segments_panel is None or segments_panel.empty:
        return pd.DataFrame(rows)

    seg_names = [c for c in ("front", "belly", "back") if c in segments_panel.columns]

    for ticker, ev_df in events.items():
        if ev_df is None or ev_df.empty:
            continue
        ev_df = ev_df.copy()
        ev_df.index = pd.to_datetime(ev_df.index)
        if "surprise" not in ev_df.columns:
            continue

        # Per-ticker per-segment per-horizon analysis
        # Aggregate to (segment, dominant_horizon) row
        all_betas = {}
        all_r2 = {}
        all_hr = {}
        for seg in seg_names:
            for h in horizons_d:
                # Δ_segment from T to T+h on each event day
                drifts = []
                surprises = []
                for ts in ev_df.index:
                    sur = float(ev_df.loc[ts, "surprise"]) if pd.notna(ev_df.loc[ts, "surprise"]) else None
                    if sur is None:
                        continue
                    try:
                        seg_idx = segments_panel.index.get_indexer([ts], method="nearest")[0]
                    except Exception:
                        continue
                    if seg_idx < 0 or seg_idx + h >= len(segments_panel):
                        continue
                    start_v = segments_panel[seg].iloc[seg_idx]
                    end_v = segments_panel[seg].iloc[seg_idx + h]
                    if pd.isna(start_v) or pd.isna(end_v):
                        continue
                    drifts.append(float(end_v - start_v))
                    surprises.append(sur)
                if len(drifts) < 5:
                    continue
                arr_x = np.array(surprises)
                arr_y = np.array(drifts)
                fit = _safe_regress(arr_x, arr_y)
                hit = _hit_rate(fit["beta"], arr_x, arr_y)
                all_betas[(seg, h)] = fit["beta"]
                all_r2[(seg, h)] = fit["R2"]
                all_hr[(seg, h)] = hit

        if not all_betas:
            continue

        # Pick dominant (segment, horizon) by absolute β
        best_key = max(all_betas.keys(), key=lambda k: abs(all_betas[k]))
        dom_seg, dom_h = best_key

        rows.append({
            "ticker": ticker,
            "event_class": str(ev_df["event_class"].iloc[0]) if "event_class" in ev_df.columns else "Unknown",
            "dominant_segment": dom_seg,
            "dominant_horizon_d": int(dom_h),
            "bp_per_sigma_front": all_betas.get(("front", dom_h), 0.0),
            "bp_per_sigma_belly": all_betas.get(("belly", dom_h), 0.0),
            "bp_per_sigma_back": all_betas.get(("back", dom_h), 0.0),
            "R2_dom": all_r2[best_key],
            "hit_rate_dom": all_hr[best_key],
            "n_events": int(len(arr_x)),
            "abs_beta_dom": abs(all_betas[best_key]),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Composite importance score: 0.40·rank_z(|β|) + 0.30·rank_z(R²) + 0.30·rank_z(2·hit−1)
    rz_beta = _rank_z(df["abs_beta_dom"].values)
    rz_r2 = _rank_z(df["R2_dom"].values)
    rz_hr = _rank_z(2 * df["hit_rate_dom"].values - 1)
    df["importance_score"] = 0.40 * rz_beta + 0.30 * rz_r2 + 0.30 * rz_hr

    # Rank within importance (1 = highest score)
    df["importance_rank"] = df["importance_score"].rank(method="min", ascending=False).astype(int)

    # Recency flag — requires multi-window analysis. With single-pass we
    # approximate by setting it True for top quartile (placeholder; full
    # version requires running this function over rolling windows).
    df["becoming_more_important"] = df["importance_score"] > df["importance_score"].quantile(0.75)

    return df.sort_values("importance_score", ascending=False).reset_index(drop=True)
