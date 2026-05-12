"""Combinatorial Purged Cross-Validation (CPCV) — Phase 1.

Per López de Prado 2018 ("Advances in Financial Machine Learning",
chapter 12), CPCV is the only defensible cross-validation framework for
autocorrelated, regime-clustered, label-overlapping financial data.

This module provides:

1. ``cpcv_paths(N, k, embargo, label_horizon)`` — generate the (N, k)
   combinatorial split paths with purge + embargo.
2. ``deflated_sharpe(sr, n_trials, autocorr, n_obs)`` — Bailey-López de
   Prado 2014 deflated Sharpe ratio.
3. ``probability_of_backtest_overfitting(metric_paths)`` — PBO from the
   relative ranking of in-sample vs out-of-sample performance across paths.
4. ``run_cpcv_evaluation(trades, asof_date, label_horizon_bars, ...)`` —
   end-to-end harness: takes a trade DataFrame from ``simulate_trades``,
   produces (N, k) splits, computes path-wise out-of-sample Sharpe, and
   returns deflated Sharpe + PBO.

Default parameters per plan §4 / §15:
  N = 6, k = 2 → C(6,2) = 15 paths
  embargo = 5 business days
  purge by label horizon (60 bars default)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# Config (locked per plan §4 / §15)
# =============================================================================

DEFAULT_N_GROUPS = 6
DEFAULT_K_TEST_GROUPS = 2
DEFAULT_EMBARGO_BARS = 5
DEFAULT_LABEL_HORIZON = 60     # max trade holding bars (= time stop)


# =============================================================================
# Path generation
# =============================================================================

@dataclass
class CPCVPath:
    path_id: int
    train_groups: tuple
    test_groups: tuple
    train_indices: list      # row indices into the trade frame
    test_indices: list


def cpcv_paths(trades: pd.DataFrame,
                  n_groups: int = DEFAULT_N_GROUPS,
                  k_test: int = DEFAULT_K_TEST_GROUPS,
                  embargo_bars: int = DEFAULT_EMBARGO_BARS,
                  label_horizon: int = DEFAULT_LABEL_HORIZON,
                  date_col: str = "entry_dt") -> list[CPCVPath]:
    """Generate the C(n_groups, k_test) combinatorial paths with purge + embargo.

    Steps:
      1. Sort trades by entry_dt and assign each to one of ``n_groups``
         contiguous time-groups.
      2. For each combination of ``k_test`` test-groups, generate a path:
         - test_indices = rows in the test-groups
         - train_indices = rows in the rest, MINUS rows whose entry_dt
           falls within ``embargo_bars`` business days OR
           ``label_horizon`` business days of any test-group boundary.

    Returns a list of CPCVPath objects.
    """
    if trades is None or trades.empty:
        return []
    df = trades.sort_values(date_col).reset_index(drop=True)
    df[date_col] = pd.to_datetime(df[date_col])

    # Assign group IDs by chronological-quantile
    n = len(df)
    group_ids = np.array_split(np.arange(n), n_groups)
    df["_grp"] = -1
    for gi, idx_arr in enumerate(group_ids):
        df.loc[idx_arr, "_grp"] = gi

    # For each group, compute its date range
    group_ranges = {
        gi: (df.loc[df["_grp"] == gi, date_col].min(),
             df.loc[df["_grp"] == gi, date_col].max())
        for gi in range(n_groups) if (df["_grp"] == gi).any()
    }

    paths = []
    for path_id, test_combo in enumerate(combinations(range(n_groups), k_test)):
        test_combo = tuple(test_combo)
        test_mask = df["_grp"].isin(test_combo)
        test_idx = df.loc[test_mask].index.tolist()

        # Train mask = NOT in test groups
        train_mask = ~test_mask
        # Purge: drop train rows whose entry_dt is within label_horizon
        # business days BEFORE any test group's earliest entry_dt
        # OR within embargo_bars AFTER any test group's latest entry_dt
        for gi in test_combo:
            t_start, t_end = group_ranges.get(gi, (None, None))
            if t_start is None:
                continue
            purge_start = t_start - pd.Timedelta(days=int(label_horizon * 7 / 5))
            embargo_end = t_end + pd.Timedelta(days=int(embargo_bars * 7 / 5))
            within_purge = ((df[date_col] >= purge_start) & (df[date_col] < t_start))
            within_embargo = ((df[date_col] > t_end) & (df[date_col] <= embargo_end))
            train_mask = train_mask & ~within_purge & ~within_embargo

        train_idx = df.loc[train_mask].index.tolist()

        paths.append(CPCVPath(
            path_id=path_id,
            train_groups=tuple(g for g in range(n_groups) if g not in test_combo),
            test_groups=test_combo,
            train_indices=train_idx,
            test_indices=test_idx,
        ))

    return paths


# =============================================================================
# Deflated Sharpe Ratio (Bailey-López de Prado 2014)
# =============================================================================

def deflated_sharpe(sr_observed: float,
                       n_trials: int,
                       skewness: float = 0.0,
                       kurtosis: float = 3.0,
                       n_obs: int = 252) -> float:
    """Compute the deflated Sharpe ratio from Bailey & López de Prado 2014.

    Adjusts an observed Sharpe ratio for selection bias when ``n_trials``
    strategies were tested. Returns the probability that the observed SR
    exceeds the expected maximum SR from null backtests.

    Args:
        sr_observed: observed Sharpe ratio of the chosen strategy.
        n_trials:    number of independent strategies tried.
        skewness:    skewness of the strategy's return distribution.
        kurtosis:    kurtosis of the same.
        n_obs:       number of observations used to compute SR.

    Returns:
        DSR ∈ [0, 1] — probability the observed SR is real.
    """
    if n_trials <= 1 or n_obs <= 1:
        return float("nan")
    # Expected maximum SR from N independent null draws:
    # E[max] ≈ (1 - γ) Φ⁻¹(1 - 1/N) + γ Φ⁻¹(1 - 1/(Ne))
    # γ ≈ Euler-Mascheroni constant 0.5772
    gamma = 0.5772156649
    e = np.e
    em = ((1 - gamma) * stats.norm.ppf(1 - 1.0 / n_trials)
            + gamma * stats.norm.ppf(1 - 1.0 / (n_trials * e)))
    # Variance of SR estimator (with finite n + skew + kurt corrections)
    var_sr = ((1 - skewness * sr_observed
                  + (kurtosis - 1) / 4.0 * sr_observed ** 2)
                / (n_obs - 1))
    if var_sr <= 0:
        return float("nan")
    z = (sr_observed - em) / np.sqrt(var_sr)
    return float(stats.norm.cdf(z))


# =============================================================================
# Probability of Backtest Overfitting (PBO)
# =============================================================================

def probability_of_backtest_overfitting(metric_in_sample: list[float],
                                              metric_out_of_sample: list[float]) -> float:
    """PBO per Bailey, Borwein, López de Prado, Zhu 2017.

    For each path, rank the in-sample metric vs all other paths' in-sample;
    the corresponding out-of-sample metric is then ranked. PBO = P(median
    rank of best-in-sample is below median in OOS).

    Returns PBO ∈ [0, 1]; 0 = no overfit, 1 = fully overfit.
    """
    n = len(metric_in_sample)
    if n != len(metric_out_of_sample) or n < 2:
        return float("nan")
    is_arr = np.asarray(metric_in_sample)
    oos_arr = np.asarray(metric_out_of_sample)
    is_ranks = pd.Series(is_arr).rank(pct=True)
    oos_ranks = pd.Series(oos_arr).rank(pct=True)

    # PBO = fraction of paths where best-in-sample ranks below median OOS
    median = 0.5
    n_below = 0
    n_total = 0
    for i in range(n):
        if is_ranks.iloc[i] >= 0.5:   # above-median in-sample
            n_total += 1
            if oos_ranks.iloc[i] < median:
                n_below += 1
    if n_total == 0:
        return float("nan")
    return float(n_below / n_total)


# =============================================================================
# End-to-end harness
# =============================================================================

@dataclass
class CPCVResult:
    n_paths: int
    n_trades_total: int
    in_sample_metrics: list[float] = field(default_factory=list)
    out_of_sample_metrics: list[float] = field(default_factory=list)
    deflated_sharpe: Optional[float] = None
    pbo: Optional[float] = None
    metric_name: str = "sharpe"
    notes: str = ""


def _path_sharpe(returns: np.ndarray) -> float:
    """Annualised Sharpe from per-trade R values."""
    if len(returns) < 2:
        return float("nan")
    sd = np.std(returns)
    if sd <= 0:
        return float("nan")
    return float(np.mean(returns) / sd * np.sqrt(252))


def run_cpcv_evaluation(trades: pd.DataFrame,
                            n_groups: int = DEFAULT_N_GROUPS,
                            k_test: int = DEFAULT_K_TEST_GROUPS,
                            embargo_bars: int = DEFAULT_EMBARGO_BARS,
                            label_horizon: int = DEFAULT_LABEL_HORIZON,
                            metric_col: str = "realized_R",
                            n_trials: int = 1) -> CPCVResult:
    """End-to-end CPCV evaluation of a trade DataFrame.

    For each of C(n_groups, k_test) paths:
      - In-sample Sharpe = annualized Sharpe of trades in train_indices
      - Out-of-sample Sharpe = annualized Sharpe of trades in test_indices

    Then computes deflated Sharpe (n_trials = number of strategies you
    were comparing) + PBO across paths.

    Returns a CPCVResult.
    """
    if trades is None or trades.empty:
        return CPCVResult(n_paths=0, n_trades_total=0,
                            notes="empty trades frame")
    if metric_col not in trades.columns:
        return CPCVResult(n_paths=0, n_trades_total=len(trades),
                            notes=f"metric column {metric_col!r} missing")

    paths = cpcv_paths(trades, n_groups=n_groups, k_test=k_test,
                          embargo_bars=embargo_bars,
                          label_horizon=label_horizon)
    if not paths:
        return CPCVResult(n_paths=0, n_trades_total=len(trades),
                            notes="no paths generated")

    metric_array = trades[metric_col].astype(float).values

    is_metrics = []
    oos_metrics = []
    for p in paths:
        is_returns = metric_array[p.train_indices] if p.train_indices else np.array([])
        oos_returns = metric_array[p.test_indices] if p.test_indices else np.array([])
        is_metrics.append(_path_sharpe(is_returns))
        oos_metrics.append(_path_sharpe(oos_returns))

    # Deflated Sharpe on the OOS-mean Sharpe
    valid_oos = [s for s in oos_metrics if np.isfinite(s)]
    if valid_oos:
        sr = float(np.mean(valid_oos))
        # Estimate skew + kurt from the trade distribution itself
        trade_returns = metric_array[np.isfinite(metric_array)]
        if len(trade_returns) > 30:
            skw = float(stats.skew(trade_returns))
            kur = float(stats.kurtosis(trade_returns)) + 3.0   # to non-excess
        else:
            skw, kur = 0.0, 3.0
        dsr = deflated_sharpe(sr, n_trials=max(1, n_trials),
                                  skewness=skw, kurtosis=kur,
                                  n_obs=len(trade_returns))
    else:
        dsr = None

    pbo = probability_of_backtest_overfitting(is_metrics, oos_metrics)

    return CPCVResult(
        n_paths=len(paths),
        n_trades_total=len(trades),
        in_sample_metrics=is_metrics,
        out_of_sample_metrics=oos_metrics,
        deflated_sharpe=dsr,
        pbo=pbo,
        metric_name=metric_col,
        notes=f"CPCV: N={n_groups}, k={k_test}, embargo={embargo_bars}, horizon={label_horizon}",
    )
