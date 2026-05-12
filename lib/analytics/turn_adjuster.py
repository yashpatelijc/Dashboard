"""A10 turn / QE / YE adjuster (Phase 3, plan §12).

Per-CMC-node regression of daily CMC close change on calendar dummies
(QE, YE, FOMC, NFP, holiday weeks) with Newey-West HAC standard errors.
Removes the calendar baseline so downstream analytics — most importantly
Phase 4's A1 regime classifier — operate on signal-only variance.

Spec (plan §12 Phase 3):
- per-k regression of CMC change on QE/YE/FOMC-week/NFP-week + holiday-week dummies
- HAC standard errors (Newey-West Bartlett kernel, lag = 5 by default)
- output: ``residual_change`` series alongside the original ``cmc_close`` change
- verification: regress residual_change on the same dummies → expect
  zero coefficients with p > 0.5

Inputs
------
- CMC parquets at ``.cmc_cache/sra_{outright,spread,fly}_<asof>.parquet``
- FOMC schedule from ``lib.cb_meetings.fomc_meetings_in_range``
- LIBOR cutover gate (2023-08-01) from ``lib.libor_cutover``

Outputs
-------
- ``.cmc_cache/turn_residuals_<asof>.parquet`` (long: scope × node × bar)
- ``.cmc_cache/turn_diagnostics_<asof>.parquet`` (per-node regression stats)
- ``.cmc_cache/turn_residuals_manifest_<asof>.json``

Units: close changes are in PRICE-bp (close.diff() * 100). For SR3 this is
the negative of rate-bp; sign convention is preserved through the regression.

References
----------
- Newey-Whitney (1987) "A Simple, Positive Semi-Definite, Heteroskedasticity
  and Autocorrelation Consistent Covariance Matrix" — Econometrica 55(3).
- Plan file ``c-users-yash-patel-downloads-tmia-v13-s-magical-mitten.md`` §12.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


# =============================================================================
# Configuration
# =============================================================================

# Cache directory at project root: ../../.cmc_cache (we are at lib/analytics/)
_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".cmc_cache"
_CACHE_DIR.mkdir(exist_ok=True)

BUILDER_VERSION = "1.0.0"

DEFAULT_HAC_LAG = 5            # one trading week of dependence
MIN_EFF_N_PER_DUMMY = 30       # below this, dummy is flagged low_sample
MIN_OBS_PER_NODE = 30          # below this, node is reported in missing_nodes

DUMMY_NAMES: tuple[str, ...] = ("QE", "YE", "FOMC", "NFP", "HOL")

DUMMY_DEFINITIONS: dict[str, str] = {
    "QE":   "last 5 business days of Mar/Jun/Sep (December subsumed by YE)",
    "YE":   "last 10 BD of December + first 5 BD of January",
    "FOMC": "FOMC blackout window OR same ISO week as a Fed meeting",
    "NFP":  "same ISO week as the first Friday of any month",
    "HOL":  "same ISO week as a US federal holiday "
              "(USFederalHolidayCalendar)",
}


# =============================================================================
# Calendar dummy builders
# =============================================================================

def _to_date_index(idx) -> pd.DatetimeIndex:
    """Coerce an index of dates / Timestamps / strings to DatetimeIndex."""
    return pd.DatetimeIndex(pd.to_datetime(pd.Index(idx)))


def _last_n_business_days_of_month(year: int, month: int, n: int) -> set:
    """Return a set of date objects: last ``n`` business days of (year, month)."""
    if month == 12:
        first_next = date(year + 1, 1, 1)
    else:
        first_next = date(year, month + 1, 1)
    last_day = first_next - timedelta(days=1)
    bdays = pd.bdate_range(end=pd.Timestamp(last_day), periods=n)
    return {d.date() for d in bdays}


def _first_n_business_days_of_month(year: int, month: int, n: int) -> set:
    """Return a set of date objects: first ``n`` business days of (year, month)."""
    first_day = date(year, month, 1)
    bdays = pd.bdate_range(start=pd.Timestamp(first_day), periods=n)
    return {d.date() for d in bdays}


def _qe_dummy(idx: pd.DatetimeIndex) -> pd.Series:
    """Last 5 business days of Mar/Jun/Sep (December excluded — subsumed by YE).

    A binary Series aligned to ``idx`` — 1 inside the QE window, else 0.
    """
    if len(idx) == 0:
        return pd.Series([], index=idx, dtype="int8")
    years = sorted({d.year for d in idx})
    qe_dates: set = set()
    for y in years:
        for m in (3, 6, 9):
            qe_dates |= _last_n_business_days_of_month(y, m, 5)
    fires = pd.Series([d.date() in qe_dates for d in idx], index=idx)
    return fires.astype("int8").rename("QE")


def _ye_dummy(idx: pd.DatetimeIndex) -> pd.Series:
    """Last 10 BD of December + first 5 BD of January."""
    if len(idx) == 0:
        return pd.Series([], index=idx, dtype="int8")
    years = sorted({d.year for d in idx})
    ye_dates: set = set()
    for y in years:
        ye_dates |= _last_n_business_days_of_month(y, 12, 10)
        ye_dates |= _first_n_business_days_of_month(y, 1, 5)
    fires = pd.Series([d.date() in ye_dates for d in idx], index=idx)
    return fires.astype("int8").rename("YE")


def _iso_weeks_for_dates(dates: list) -> set:
    """Return the set of (iso_year, iso_week) for a list of date-like values."""
    weeks: set = set()
    for d in dates:
        if hasattr(d, "isocalendar"):
            iso = d.isocalendar()
            # Pre-3.9 returns tuple; 3.9+ returns IsoCalendarDate. Both
            # support [0]/[1] indexing.
            try:
                weeks.add((iso[0], iso[1]))
            except (TypeError, KeyError):
                weeks.add((iso.year, iso.week))
    return weeks


def _fomc_dummy(idx: pd.DatetimeIndex,
                 fomc_meetings: Optional[list] = None) -> pd.Series:
    """FOMC-week dummy: bar_date is in the same ISO week as a FOMC meeting
    OR inside the meeting's blackout window."""
    if len(idx) == 0:
        return pd.Series([], index=idx, dtype="int8")
    if fomc_meetings is None:
        try:
            from lib.cb_meetings import fomc_meetings_in_range
            start = idx.min().date() - timedelta(days=30)
            end = idx.max().date() + timedelta(days=30)
            fomc_meetings = fomc_meetings_in_range(start, end)
        except Exception:
            fomc_meetings = []
    if not fomc_meetings:
        return pd.Series(0, index=idx, dtype="int8").rename("FOMC")

    # Same-ISO-week match
    meeting_dates = [m.date for m in fomc_meetings]
    fomc_weeks = _iso_weeks_for_dates(meeting_dates)

    # Plus: blackout-window match (12 days before, 1 day after — config-driven)
    blackout_dates: set = set()
    try:
        from lib.cb_meetings import fomc_blackout_window
        for m in fomc_meetings:
            start, end = fomc_blackout_window(m.date)
            cur = start
            while cur <= end:
                blackout_dates.add(cur)
                cur += timedelta(days=1)
    except Exception:
        pass

    fires = []
    for ts in idx:
        d = ts.date()
        iso = ts.isocalendar()
        try:
            iso_pair = (iso[0], iso[1])
        except (TypeError, KeyError):
            iso_pair = (iso.year, iso.week)
        fires.append(int(iso_pair in fomc_weeks or d in blackout_dates))
    return pd.Series(fires, index=idx, dtype="int8").rename("FOMC")


def _nfp_dummy(idx: pd.DatetimeIndex) -> pd.Series:
    """Same ISO week as the first Friday of any month (NFP release week)."""
    if len(idx) == 0:
        return pd.Series([], index=idx, dtype="int8")
    years = sorted({d.year for d in idx})
    nfp_dates: list = []
    for y in years:
        for m in range(1, 13):
            first = date(y, m, 1)
            offset = (4 - first.weekday()) % 7   # 4 = Friday
            nfp_dates.append(first + timedelta(days=offset))
    nfp_weeks = _iso_weeks_for_dates(nfp_dates)
    fires = []
    for ts in idx:
        iso = ts.isocalendar()
        try:
            iso_pair = (iso[0], iso[1])
        except (TypeError, KeyError):
            iso_pair = (iso.year, iso.week)
        fires.append(int(iso_pair in nfp_weeks))
    return pd.Series(fires, index=idx, dtype="int8").rename("NFP")


def _hol_dummy(idx: pd.DatetimeIndex) -> pd.Series:
    """Same ISO week as a US federal holiday."""
    if len(idx) == 0:
        return pd.Series([], index=idx, dtype="int8")
    cal = USFederalHolidayCalendar()
    start = pd.Timestamp(idx.min().date() - timedelta(days=14))
    end = pd.Timestamp(idx.max().date() + timedelta(days=14))
    hol_ts = cal.holidays(start=start, end=end)
    hol_weeks = _iso_weeks_for_dates([d.date() for d in hol_ts])
    fires = []
    for ts in idx:
        iso = ts.isocalendar()
        try:
            iso_pair = (iso[0], iso[1])
        except (TypeError, KeyError):
            iso_pair = (iso.year, iso.week)
        fires.append(int(iso_pair in hol_weeks))
    return pd.Series(fires, index=idx, dtype="int8").rename("HOL")


def build_calendar_dummies(bar_dates,
                              fomc_meetings: Optional[list] = None
                              ) -> pd.DataFrame:
    """Construct the wide dummy matrix indexed by ``bar_dates``.

    Columns (in order): ['QE', 'YE', 'FOMC', 'NFP', 'HOL'].
    Each column is int8 ∈ {0, 1}; multiple dummies may fire on the same date.
    """
    idx = _to_date_index(bar_dates)
    return pd.concat(
        [
            _qe_dummy(idx),
            _ye_dummy(idx),
            _fomc_dummy(idx, fomc_meetings=fomc_meetings),
            _nfp_dummy(idx),
            _hol_dummy(idx),
        ],
        axis=1,
    )


# =============================================================================
# Newey-West HAC standard errors (numpy implementation)
# =============================================================================

def newey_west_hac(X: np.ndarray,
                       residuals: np.ndarray,
                       lag: int = DEFAULT_HAC_LAG
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Bartlett-kernel Newey-West HAC covariance for OLS coefficients.

    Parameters
    ----------
    X : (n, k) regressor matrix (must include intercept column if intended)
    residuals : (n,) OLS residuals
    lag : non-negative int, kernel truncation (Bartlett); default 5

    Returns
    -------
    (se, cov_hac) : (k,) standard errors and (k, k) covariance matrix.
    """
    X = np.asarray(X, dtype=float)
    e = np.asarray(residuals, dtype=float).reshape(-1)
    n, k = X.shape
    if n != e.shape[0]:
        raise ValueError(f"shape mismatch: X={X.shape}, e={e.shape}")
    if n <= k:
        return (np.full(k, np.nan), np.full((k, k), np.nan))

    # OLS bread: (X'X)^-1
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.pinv(XtX)
    except np.linalg.LinAlgError:
        return (np.full(k, np.nan), np.full((k, k), np.nan))

    # Build Omega = sum_h w(h) * sum_t e_t e_{t-h} x_t x_{t-h}'
    # Bartlett kernel: w(h) = 1 - |h| / (lag + 1)
    Omega = np.zeros((k, k))

    # h = 0
    Xe = X * e[:, None]
    Omega += Xe.T @ Xe

    # h = 1..lag (symmetric, so we add T+T')
    for h in range(1, lag + 1):
        w = 1.0 - h / (lag + 1.0)
        # sum_{t=h..n-1} e_t e_{t-h} x_t x_{t-h}'
        Xe_t = X[h:] * e[h:, None]
        Xe_th = X[:n - h] * e[:n - h, None]
        Gamma_h = Xe_t.T @ Xe_th
        Omega += w * (Gamma_h + Gamma_h.T)

    cov_hac = XtX_inv @ Omega @ XtX_inv
    diag = np.diag(cov_hac)
    se = np.sqrt(np.where(diag > 0, diag, np.nan))
    return (se, cov_hac)


# =============================================================================
# Per-node fit
# =============================================================================

@dataclass
class NodeFitResult:
    cmc_node: str
    n_obs: int
    dof: int
    r_squared: float
    raw_var: float
    residual_var: float
    raw_change: pd.Series       # indexed by bar_date
    residual_change: pd.Series  # raw_change minus dummy contribution (excludes intercept)
    fitted_change: pd.Series    # fitted Δ from ALL regressors including intercept
    betas: dict                 # regressor -> beta (incl. intercept)
    se: dict                    # regressor -> Newey-West HAC std err
    p_values: dict              # regressor -> two-sided p-value (normal approx)
    eff_n: dict                 # dummy_name -> sum of fires in sample
    low_sample_dummies: list    # dummies with eff_n < MIN_EFF_N_PER_DUMMY


def _two_sided_p(t_stat: float) -> float:
    """Two-sided p-value under standard normal (HAC asymptotic)."""
    from math import erfc, sqrt
    if not np.isfinite(t_stat):
        return float("nan")
    # P(|Z| > |t|) = erfc(|t| / sqrt(2))
    return float(erfc(abs(t_stat) / sqrt(2.0)))


def fit_turn_adjustment(change_series: pd.Series,
                            dummies: pd.DataFrame,
                            hac_lag: int = DEFAULT_HAC_LAG,
                            cmc_node: str = "?",
                            ) -> NodeFitResult:
    """Per-node OLS of ``change_series`` on intercept + dummies, with
    Newey-West HAC standard errors. Returns a ``NodeFitResult``.

    The residual_change is defined as ``raw - sum_d beta_d * (D_d - mean(D_d))``
    — i.e. the *demeaned* dummy effect is removed. This preserves BOTH:

    1. ``mean(residual_change) == mean(raw_change)`` exactly (modulo float),
       so unconditional drift is unchanged; and
    2. ``cov(residual_change, D_d) == 0`` for every dummy, so the residuals
       are orthogonal to every regressor (the verification gate from plan §12).

    Subtracting the un-demeaned ``beta_d * D_d`` (without re-centering) would
    shift the mean by ``Σ beta_d * mean(D_d)``, which can be material when
    dummies have non-trivial sample frequency.
    """
    # Align + drop NaN
    df = pd.concat([change_series.rename("y"), dummies], axis=1).dropna()
    n = len(df)
    if n < MIN_OBS_PER_NODE:
        empty_idx = change_series.index
        return NodeFitResult(
            cmc_node=cmc_node, n_obs=n, dof=0,
            r_squared=float("nan"),
            raw_var=float("nan"), residual_var=float("nan"),
            raw_change=change_series,
            residual_change=pd.Series([np.nan] * len(empty_idx), index=empty_idx),
            fitted_change=pd.Series([np.nan] * len(empty_idx), index=empty_idx),
            betas={}, se={}, p_values={},
            eff_n={d: int(dummies[d].sum()) for d in dummies.columns},
            low_sample_dummies=list(dummies.columns),
        )

    y = df["y"].to_numpy(dtype=float)
    D_cols = [c for c in dummies.columns if c in df.columns]
    D = df[D_cols].to_numpy(dtype=float)

    # Drop dummies that don't fire at all in the sample (would be perfectly
    # collinear with the constant) — record them but do not regress.
    fire_counts = D.sum(axis=0)
    keep = fire_counts > 0
    dropped = [c for c, k in zip(D_cols, keep) if not k]
    D_kept = D[:, keep]
    kept_names = [c for c, k in zip(D_cols, keep) if k]

    # Design matrix: intercept + kept dummies
    X = np.hstack([np.ones((n, 1)), D_kept])
    reg_names = ["intercept"] + kept_names
    k = X.shape[1]

    # OLS via lstsq (handles near-singular X gracefully)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    resid = y - fitted
    dof = max(n - k, 1)

    # R^2
    y_var = float(np.var(y, ddof=0))
    r_squared = 1.0 - (float(np.var(resid, ddof=0)) / y_var) if y_var > 0 else 0.0

    # Newey-West HAC
    se_vec, _cov = newey_west_hac(X, resid, lag=hac_lag)

    # residual_change = raw - sum_d beta_d * (D_d - mean(D_d))
    # Demean the dummies before subtracting so both mean(residual)==mean(raw)
    # and cov(residual, D_d)==0 hold simultaneously.
    if D_kept.shape[1] > 0:
        D_centered = D_kept - D_kept.mean(axis=0, keepdims=True)
        dummy_calendar_effect = D_centered @ beta[1:]
    else:
        dummy_calendar_effect = np.zeros(n)
    resid_change_kept = y - dummy_calendar_effect

    # Map back to the full bar_date index of the original series
    full_idx = change_series.index
    raw_full = change_series.copy()
    fitted_full = pd.Series([np.nan] * len(full_idx), index=full_idx)
    resid_full = pd.Series([np.nan] * len(full_idx), index=full_idx)
    fitted_full.loc[df.index] = fitted
    resid_full.loc[df.index] = resid_change_kept

    # Build betas / se / p_values dicts (include all 5 dummies; dropped ones get NaN)
    betas = {"intercept": float(beta[0])}
    se_d = {"intercept": float(se_vec[0]) if len(se_vec) > 0 else float("nan")}
    p_d = {"intercept": _two_sided_p(beta[0] / se_d["intercept"])
                  if se_d["intercept"] and np.isfinite(se_d["intercept"]) and se_d["intercept"] > 0
                  else float("nan")}
    # Active dummies
    for i, name in enumerate(kept_names, start=1):
        b = float(beta[i])
        s = float(se_vec[i]) if i < len(se_vec) else float("nan")
        betas[name] = b
        se_d[name] = s
        p_d[name] = _two_sided_p(b / s) if s and np.isfinite(s) and s > 0 else float("nan")
    # Dropped dummies — no fires
    for name in dropped:
        betas[name] = float("nan")
        se_d[name] = float("nan")
        p_d[name] = float("nan")

    eff_n = {c: int(dummies[c].sum()) for c in dummies.columns}
    low_sample = [c for c, n_fires in eff_n.items()
                     if n_fires < MIN_EFF_N_PER_DUMMY]

    return NodeFitResult(
        cmc_node=cmc_node, n_obs=n, dof=dof,
        r_squared=float(r_squared),
        raw_var=float(np.var(y, ddof=0)),
        residual_var=float(np.var(resid_change_kept, ddof=0)),
        raw_change=raw_full,
        residual_change=resid_full,
        fitted_change=fitted_full,
        betas=betas, se=se_d, p_values=p_d,
        eff_n=eff_n, low_sample_dummies=low_sample,
    )


# =============================================================================
# Per-scope orchestration
# =============================================================================

def _load_cmc_panel_for_scope(scope: str,
                                  cmc_asof: date,
                                  product_code: str = "SRA",
                                  ) -> pd.DataFrame:
    """Load CMC panel filtered to post-LIBOR-cutover dates."""
    from lib.cmc import load_cmc_panel
    from lib.libor_cutover import filter_post_cutover
    panel = load_cmc_panel(scope, cmc_asof)
    if panel is None or panel.empty:
        return pd.DataFrame()
    # Ensure bar_date is datetime for cutover filter
    panel = panel.copy()
    panel["bar_date"] = pd.to_datetime(panel["bar_date"])
    panel = filter_post_cutover(panel, date_col="bar_date", product_code=product_code)
    return panel


def adjust_cmc_panel(scope: str,
                         cmc_asof: date,
                         hac_lag: int = DEFAULT_HAC_LAG,
                         product_code: str = "SRA",
                         ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Fit the calendar-dummy regression for every node in ``scope``.

    Returns
    -------
    (residuals_long, diagnostics, run_meta)

    ``residuals_long`` columns:
        scope, cmc_node, bar_date, raw_change, residual_change, fitted_change,
        has_data
    ``diagnostics`` columns (one row per node):
        scope, cmc_node, n_obs, dof, r_squared, raw_var, residual_var,
        var_reduction_pct, beta_*, se_*, p_*, eff_n_*, low_sample_dummies
    """
    panel = _load_cmc_panel_for_scope(scope, cmc_asof, product_code=product_code)
    if panel.empty:
        empty_long = pd.DataFrame(
            columns=["scope", "cmc_node", "bar_date",
                       "raw_change", "residual_change", "fitted_change", "has_data"])
        return (empty_long, pd.DataFrame(), {"n_nodes_fitted": 0,
                                                "n_nodes_skipped": 0,
                                                "skipped_nodes": []})

    # Pre-build the dummy matrix on the union of dates (saves per-node FOMC lookups)
    all_dates = pd.DatetimeIndex(sorted(pd.to_datetime(panel["bar_date"]).unique()))
    try:
        from lib.cb_meetings import fomc_meetings_in_range
        meetings = fomc_meetings_in_range(
            (all_dates.min().date() - timedelta(days=30)),
            (all_dates.max().date() + timedelta(days=30)),
        )
    except Exception:
        meetings = []
    full_dummies = build_calendar_dummies(all_dates, fomc_meetings=meetings)

    rows_long: list[pd.DataFrame] = []
    rows_diag: list[dict] = []
    skipped: list[str] = []

    for node, sub in panel.groupby("cmc_node", sort=True):
        sub = sub.sort_values("bar_date").drop_duplicates("bar_date")
        node_dates = pd.to_datetime(sub["bar_date"])
        # change in PRICE-bp (close.diff() * 100)
        close = pd.Series(sub["close"].values, index=node_dates).astype(float)
        raw_change_bp = (close.diff() * 100.0).rename("raw_change")

        if raw_change_bp.dropna().shape[0] < MIN_OBS_PER_NODE:
            skipped.append(node)
            continue

        node_dummies = full_dummies.reindex(node_dates).fillna(0).astype("int8")
        fit = fit_turn_adjustment(raw_change_bp, node_dummies,
                                       hac_lag=hac_lag, cmc_node=node)

        if fit.n_obs < MIN_OBS_PER_NODE:
            skipped.append(node)
            continue

        # Long output
        long = pd.DataFrame({
            "scope": scope,
            "cmc_node": node,
            "bar_date": node_dates.dt.date.values,
            "raw_change": fit.raw_change.values,
            "residual_change": fit.residual_change.values,
            "fitted_change": fit.fitted_change.values,
            "has_data": (~fit.raw_change.isna()).values & sub["has_data"].values,
        })
        rows_long.append(long)

        # Diagnostics row
        var_red_pct = (
            (fit.raw_var - fit.residual_var) / fit.raw_var * 100.0
            if fit.raw_var > 0 else float("nan")
        )
        diag_row = {
            "scope": scope,
            "cmc_node": node,
            "n_obs": fit.n_obs,
            "dof": fit.dof,
            "r_squared": fit.r_squared,
            "raw_var": fit.raw_var,
            "residual_var": fit.residual_var,
            "var_reduction_pct": var_red_pct,
            "low_sample_dummies": ",".join(fit.low_sample_dummies),
        }
        # Per-regressor: intercept + 5 dummies
        for reg in ["intercept", *DUMMY_NAMES]:
            diag_row[f"beta_{reg}"] = fit.betas.get(reg, float("nan"))
            diag_row[f"se_{reg}"] = fit.se.get(reg, float("nan"))
            diag_row[f"p_{reg}"] = fit.p_values.get(reg, float("nan"))
        for d in DUMMY_NAMES:
            diag_row[f"eff_n_{d}"] = fit.eff_n.get(d, 0)
        rows_diag.append(diag_row)

    residuals_long = (pd.concat(rows_long, ignore_index=True)
                          if rows_long else pd.DataFrame())
    diagnostics = pd.DataFrame(rows_diag) if rows_diag else pd.DataFrame()
    run_meta = {
        "n_nodes_fitted": len(rows_diag),
        "n_nodes_skipped": len(skipped),
        "skipped_nodes": skipped,
    }
    return (residuals_long, diagnostics, run_meta)


# =============================================================================
# Persistence
# =============================================================================

def _residuals_paths(asof_date: date) -> dict:
    """Canonical paths for the turn-adjuster cache artifacts."""
    stamp = asof_date.isoformat()
    return {
        "residuals":   _CACHE_DIR / f"turn_residuals_{stamp}.parquet",
        "diagnostics": _CACHE_DIR / f"turn_diagnostics_{stamp}.parquet",
        "manifest":    _CACHE_DIR / f"turn_residuals_manifest_{stamp}.json",
    }


def write_turn_residuals_cache(asof_date: date,
                                    residuals: pd.DataFrame,
                                    diagnostics: pd.DataFrame,
                                    cmc_asof_date: date,
                                    hac_lag: int = DEFAULT_HAC_LAG,
                                    history_start: Optional[date] = None,
                                    skipped_nodes: Optional[list] = None,
                                    product_code: str = "SRA",
                                    ) -> dict:
    """Atomically write residuals + diagnostics + manifest.

    Atomicity pattern: write to ``.tmp`` paths first, then ``os.replace`` to
    final names. Mirrors ``lib/cmc.py`` builder.
    """
    paths = _residuals_paths(asof_date)
    tmp_residuals = paths["residuals"].with_suffix(".parquet.tmp")
    tmp_diag = paths["diagnostics"].with_suffix(".parquet.tmp")
    tmp_manifest = paths["manifest"].with_suffix(".json.tmp")

    if residuals is None or residuals.empty:
        raise ValueError("residuals DataFrame is empty — nothing to persist")

    residuals.to_parquet(tmp_residuals, index=False)
    diagnostics.to_parquet(tmp_diag, index=False)

    # Build manifest
    by_scope = (residuals.groupby("scope")["cmc_node"].nunique().to_dict()
                  if not residuals.empty else {})

    # Per-node summary for the manifest (kept compact)
    node_coverage: dict[str, dict] = {}
    if not diagnostics.empty:
        for _, row in diagnostics.iterrows():
            node_coverage[str(row["cmc_node"])] = {
                "scope": str(row["scope"]),
                "n_obs": int(row["n_obs"]),
                "r_squared": float(row["r_squared"]),
                "var_reduction_pct": float(row["var_reduction_pct"]),
                "low_sample_dummies": (str(row.get("low_sample_dummies", "") or "")
                                                .split(",") if row.get("low_sample_dummies") else []),
            }

    dummy_fire_counts: dict[str, int] = {}
    if not diagnostics.empty:
        for d in DUMMY_NAMES:
            col = f"eff_n_{d}"
            if col in diagnostics.columns:
                # Take max over nodes (same dummies fire on the same dates regardless of node)
                dummy_fire_counts[d] = int(diagnostics[col].max())

    history_start_str = (history_start.isoformat()
                              if history_start else
                              (residuals["bar_date"].min().isoformat()
                                 if not residuals.empty else None))

    manifest = {
        "builder_version": BUILDER_VERSION,
        "asof_date": asof_date.isoformat(),
        "cmc_asof_date": cmc_asof_date.isoformat(),
        "history_start": history_start_str,
        "post_cutover_only": True,
        "product_code": product_code,
        "hac_lag": int(hac_lag),
        "n_nodes_total": sum(by_scope.values()),
        "n_nodes_outright": int(by_scope.get("outright", 0)),
        "n_nodes_spread": int(by_scope.get("spread", 0)),
        "n_nodes_fly": int(by_scope.get("fly", 0)),
        "dummy_definitions": DUMMY_DEFINITIONS,
        "dummy_fire_counts": dummy_fire_counts,
        "node_coverage": node_coverage,
        "missing_nodes": list(skipped_nodes or []),
    }

    tmp_manifest.write_text(json.dumps(manifest, indent=2, default=str))

    # Atomic replace
    os.replace(tmp_residuals, paths["residuals"])
    os.replace(tmp_diag, paths["diagnostics"])
    os.replace(tmp_manifest, paths["manifest"])

    return manifest


# =============================================================================
# Top-level driver
# =============================================================================

def build_turn_residuals(cmc_asof: date,
                              hac_lag: int = DEFAULT_HAC_LAG,
                              product_code: str = "SRA",
                              ) -> dict:
    """Build residuals for all three scopes (outright + spread + fly) and
    persist. Returns the manifest dict."""
    all_long_parts: list[pd.DataFrame] = []
    all_diag_parts: list[pd.DataFrame] = []
    all_skipped: list[str] = []
    history_starts: list[date] = []

    for scope in ("outright", "spread", "fly"):
        long, diag, meta = adjust_cmc_panel(
            scope, cmc_asof, hac_lag=hac_lag, product_code=product_code,
        )
        if not long.empty:
            all_long_parts.append(long)
            history_starts.append(pd.to_datetime(long["bar_date"]).min().date())
        if not diag.empty:
            all_diag_parts.append(diag)
        all_skipped.extend([f"{scope}/{n}" for n in meta.get("skipped_nodes", [])])

    if not all_long_parts:
        raise RuntimeError("turn_adjuster produced no rows — check CMC cache + cutover")

    residuals = pd.concat(all_long_parts, ignore_index=True)
    diagnostics = pd.concat(all_diag_parts, ignore_index=True)
    history_start = min(history_starts) if history_starts else None

    return write_turn_residuals_cache(
        asof_date=cmc_asof,
        residuals=residuals,
        diagnostics=diagnostics,
        cmc_asof_date=cmc_asof,
        hac_lag=hac_lag,
        history_start=history_start,
        skipped_nodes=all_skipped,
        product_code=product_code,
    )


# =============================================================================
# CLI
# =============================================================================

def _resolve_latest_cmc_asof() -> Optional[date]:
    """Find the newest manifest_*.json in .cmc_cache and parse the date."""
    cands = sorted(_CACHE_DIR.glob("manifest_*.json"),
                       key=os.path.getmtime, reverse=True)
    if not cands:
        return None
    stem = cands[0].stem  # manifest_YYYY-MM-DD
    try:
        return date.fromisoformat(stem.replace("manifest_", ""))
    except ValueError:
        return None


def main(argv=None):
    import sys
    args = list(sys.argv[1:] if argv is None else argv)
    if args:
        try:
            asof = date.fromisoformat(args[0])
        except ValueError:
            print(f"usage: python -m lib.analytics.turn_adjuster [YYYY-MM-DD]")
            return 2
    else:
        asof = _resolve_latest_cmc_asof()
        if asof is None:
            print("[turn_adjuster] no CMC manifest found in .cmc_cache/")
            return 2
    print(f"[turn_adjuster] building turn residuals for asof={asof}")
    manifest = build_turn_residuals(asof)
    print(f"[turn_adjuster] wrote {manifest['n_nodes_total']} nodes "
              f"(outright={manifest['n_nodes_outright']}, "
              f"spread={manifest['n_nodes_spread']}, "
              f"fly={manifest['n_nodes_fly']}) "
              f"history_start={manifest['history_start']} "
              f"skipped={len(manifest['missing_nodes'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
