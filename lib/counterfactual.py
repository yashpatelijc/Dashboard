"""Counterfactual / analog-fires histogram (Phase 8, plan §3.4).

Given a setup_id, returns the historical distribution of all prior fires'
outcomes (R-multiples, win/loss, holding-period). Honest sample-size
gating per spec: if n < 30, returns "insufficient sample" sentinel.

Reads from `.backtest_cache/tmia.duckdb` (Phase E backtest output).

API:
    analog_outcomes(setup_id, n_min=30, asof=None) -> dict
    distribution_quantiles(outcomes_arr) -> dict
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

INSUFFICIENT_SAMPLE_THRESHOLD = 30


def _tmia_path() -> Path:
    return Path(__file__).resolve().parent.parent / ".backtest_cache" / "tmia.duckdb"


def analog_outcomes(setup_id: str, n_min: int = INSUFFICIENT_SAMPLE_THRESHOLD,
                       asof: Optional[date] = None) -> dict:
    """Return historical R-multiple distribution for prior fires of `setup_id`.

    Returns dict with:
        n_fires, sufficient_sample (bool), R_realized array,
        win_rate, avg_win_R, avg_loss_R, expectancy_R,
        quantiles (10/25/50/75/90), reason (if insufficient).
    """
    p = _tmia_path()
    if not p.exists():
        return {"n_fires": 0, "sufficient_sample": False,
                  "reason": "tmia.duckdb not yet built (Phase E backtest)"}
    try:
        import duckdb
        con = duckdb.connect(str(p), read_only=True)
        # Try the common tmia trade-outcome schema
        try:
            df = con.execute(
                "SELECT setup_id, R_realized, fire_date, exit_date "
                "FROM tmia_trades "
                "WHERE setup_id = ? "
                + ("AND fire_date <= ? " if asof is not None else "")
                + "AND R_realized IS NOT NULL",
                ([setup_id, asof.isoformat()] if asof is not None
                 else [setup_id]),
            ).fetchdf()
        except Exception:
            df = pd.DataFrame()
        con.close()
    except Exception:
        return {"n_fires": 0, "sufficient_sample": False,
                  "reason": "tmia query failed"}

    n = int(len(df))
    if n == 0:
        return {"n_fires": 0, "sufficient_sample": False,
                  "reason": f"no historical fires for setup_id={setup_id}"}

    R = df["R_realized"].astype(float).to_numpy()
    if n < n_min:
        return {
            "n_fires": n, "sufficient_sample": False,
            "reason": f"only {n} prior fires (< {n_min} required)",
            "R_realized": R.tolist(),
        }
    return {
        "n_fires": n,
        "sufficient_sample": True,
        "R_realized": R.tolist(),
        "win_rate": float((R > 0).mean()),
        "avg_win_R": float(R[R > 0].mean()) if (R > 0).any() else 0.0,
        "avg_loss_R": float(-R[R < 0].mean()) if (R < 0).any() else 0.0,
        "expectancy_R": float(R.mean()),
        "quantiles": distribution_quantiles(R),
    }


def distribution_quantiles(R: np.ndarray) -> dict:
    if len(R) == 0:
        return {"q10": 0.0, "q25": 0.0, "q50": 0.0, "q75": 0.0, "q90": 0.0}
    return {
        "q10": float(np.quantile(R, 0.10)),
        "q25": float(np.quantile(R, 0.25)),
        "q50": float(np.quantile(R, 0.50)),
        "q75": float(np.quantile(R, 0.75)),
        "q90": float(np.quantile(R, 0.90)),
    }


def histogram_data(R: np.ndarray, n_bins: int = 20) -> dict:
    if len(R) == 0:
        return {"bin_edges": [], "counts": []}
    counts, edges = np.histogram(R, bins=n_bins)
    return {"bin_edges": edges.tolist(), "counts": counts.tolist()}
