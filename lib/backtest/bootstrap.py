"""Bootstrap confidence intervals (Phase C)."""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def bootstrap_ci(values: np.ndarray,
                    statistic: Callable[[np.ndarray], float],
                    n_bootstrap: int = 1000,
                    alpha: float = 0.05,
                    rng_seed: int = 42) -> tuple:
    """Compute (lo, hi) percentile-based bootstrap CI for ``statistic(x)``.

    Returns ``(lo, hi)``. Returns ``(None, None)`` if values is too small or
    contains no finite samples.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 30:
        return (None, None)
    rng = np.random.default_rng(rng_seed)
    n = len(arr)
    samples = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        samples[i] = statistic(sample)
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return (lo, hi)


def bootstrap_sharpe_ci(realized_R: np.ndarray, n_bootstrap: int = 1000,
                            alpha: float = 0.05) -> tuple:
    """Bootstrap CI for the per-trade Sharpe ratio (mean / std)."""
    def _sharpe(x):
        s = np.std(x)
        return np.mean(x) / s if s > 0 else 0.0
    return bootstrap_ci(realized_R, _sharpe, n_bootstrap, alpha)


def bootstrap_winrate_ci(realized_R: np.ndarray, n_bootstrap: int = 1000,
                            alpha: float = 0.05) -> tuple:
    """Bootstrap CI for the per-trade win rate."""
    return bootstrap_ci((realized_R > 0).astype(float),
                          lambda x: float(np.mean(x)), n_bootstrap, alpha)


def bootstrap_expectancy_ci(realized_R: np.ndarray, n_bootstrap: int = 1000,
                                alpha: float = 0.05) -> tuple:
    """Bootstrap CI for the per-trade expectancy in R units (= mean R)."""
    return bootstrap_ci(realized_R, lambda x: float(np.mean(x)), n_bootstrap, alpha)
