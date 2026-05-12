"""Bootstrap-quantile calibration for NEAR / APPROACHING thresholds.

Used for setup categories where literature defines no canonical threshold
(specifically (d) multi-condition n-of-m and (f) slope/trend gates per the
Phase B research). The calibration is anchored to a target NEAR→FIRED
conversion rate of 25-35% within 5 bars (centre of the 20-40% Macrosynergy
hit-rate band).

This module is invoked from the Phase E recompute cycle. It runs after the
CMC layer is built and produces a parquet that
:func:`lib.setups.near_thresholds.load_calibrated_thresholds` reads.

Method (per Aronson 2007 + Macrosynergy):

  1. For each setup × CMC node × direction, compute per-bar proximity
     ``d(t) ∈ [0, 1]`` (1 = fired).
  2. NEAR threshold = quantile of ``d(t)`` such that NEAR-states convert
     to FIRED within 5 bars at 25-35% rate.
  3. APPROACHING threshold = wider band targeting 10-20% conversion in 10
     bars.
  4. Bootstrap: 1000 resamples with replacement → take median quantile.
  5. Winsorise the proximity distribution at ±2σ before computing
     quantiles (Macrosynergy convention).

Note: the actual per-detector proximity computation is detector-specific
and would normally be wired through ``near_state(...)`` itself. For Phase
B's initial implementation this module exposes the SHELL of the
calibrator; Phase E's recompute cycle is responsible for populating the
parquet with real values once the backtest engine (Phase C) can replay
trades to compute conversion rates.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


CALIBRATED_PARQUET_PATH = (Path(__file__).resolve().parent.parent.parent
                            / "data" / "near_thresholds_calibrated.parquet")


def bootstrap_quantile(values: np.ndarray,
                          target_quantile: float = 0.30,
                          n_bootstrap: int = 1000,
                          winsor_sigma: float = 2.0,
                          rng_seed: int = 42) -> float:
    """Bootstrap the ``target_quantile`` of ``values`` after winsorisation.

    Steps:
      1. Winsorise values at ``±winsor_sigma`` standard deviations of the
         mean (Macrosynergy convention).
      2. Resample with replacement ``n_bootstrap`` times.
      3. Compute the target_quantile of each resample.
      4. Return the MEDIAN of those quantile values.

    Robust to a single-sample fluke; converges to the population quantile
    as n_bootstrap → ∞.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    lo = mean - winsor_sigma * std
    hi = mean + winsor_sigma * std
    arr = np.clip(arr, lo, hi)

    rng = np.random.default_rng(rng_seed)
    n = len(arr)
    quantiles = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        quantiles[i] = np.quantile(sample, target_quantile)
    return float(np.median(quantiles))


def conversion_rate(proximity: pd.Series,
                       fired_flag: pd.Series,
                       lookahead_bars: int = 5) -> dict:
    """For each NEAR threshold candidate, compute the rate at which NEAR
    states convert to FIRED within ``lookahead_bars``.

    Useful diagnostic for verifying bootstrap-derived thresholds hit the
    target conversion rate (20-40%).

    Parameters
    ----------
    proximity : Series indexed by bar
        d(t) ∈ [0, 1] proximity to firing.
    fired_flag : Series indexed by the same bars
        Boolean — True if FIRED at that bar.
    lookahead_bars : int
        Window over which to count conversions.

    Returns
    -------
    dict of {threshold_quantile → conversion_rate}
        e.g. {0.10: 0.42, 0.20: 0.31, 0.30: 0.22, ...}
    """
    if len(proximity) != len(fired_flag):
        raise ValueError("proximity and fired_flag must have same length")
    out = {}
    for q in (0.10, 0.20, 0.30, 0.40, 0.50):
        threshold = float(np.quantile(proximity.dropna(), q))
        near_idx = proximity[(proximity >= threshold) & (~fired_flag)].index
        if len(near_idx) == 0:
            out[q] = None
            continue
        n_converted = 0
        for ix in near_idx:
            pos = proximity.index.get_indexer([ix])[0]
            window = fired_flag.iloc[pos + 1: pos + 1 + lookahead_bars]
            if window.any():
                n_converted += 1
        out[q] = n_converted / len(near_idx)
    return out


def calibrate_setup_thresholds(setup_id: str,
                                  proximity_history: pd.DataFrame,
                                  target_near_conversion: float = 0.30,
                                  target_approach_conversion: float = 0.15,
                                  lookahead_near: int = 5,
                                  lookahead_approach: int = 10,
                                  n_bootstrap: int = 1000) -> dict:
    """Calibrate NEAR / APPROACHING thresholds for a single setup using
    bootstrap-quantile.

    ``proximity_history`` must have columns ``proximity`` (continuous
    [0, 1] distance-to-fire score) and ``fired`` (boolean).
    """
    prox = proximity_history["proximity"].dropna()
    fired = proximity_history["fired"]

    # NEAR quantile that achieves target_near_conversion
    candidates = np.linspace(0.05, 0.50, 19)
    best_near_q = candidates[0]
    best_diff = float("inf")
    for q in candidates:
        threshold = float(np.quantile(prox, q))
        near_states = (prox >= threshold) & (~fired)
        if not near_states.any():
            continue
        n_converted = 0
        for ix in near_states[near_states].index:
            pos = prox.index.get_indexer([ix])[0]
            if pos < 0:
                continue
            window = fired.iloc[pos + 1: pos + 1 + lookahead_near]
            if window.any():
                n_converted += 1
        rate = n_converted / max(near_states.sum(), 1)
        diff = abs(rate - target_near_conversion)
        if diff < best_diff:
            best_diff = diff
            best_near_q = q

    # APPROACHING uses a wider band
    best_approach_q = max(0.05, best_near_q - 0.15)

    # Bootstrap the median of those quantiles for stability
    near_threshold = bootstrap_quantile(prox.values, best_near_q,
                                          n_bootstrap=n_bootstrap)
    approach_threshold = bootstrap_quantile(prox.values, best_approach_q,
                                              n_bootstrap=n_bootstrap)

    return {
        "setup_id": setup_id,
        "near_quantile": best_near_q,
        "approach_quantile": best_approach_q,
        "near_threshold": near_threshold,
        "approach_threshold": approach_threshold,
        "n_bars": int(len(prox)),
    }


def write_calibrated_thresholds(rows: list[dict],
                                   path: Optional[Path] = None) -> Path:
    """Write the calibrated thresholds to parquet for
    :func:`lib.setups.near_thresholds.load_calibrated_thresholds`."""
    target = path or CALIBRATED_PARQUET_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(target, index=False)
    return target
