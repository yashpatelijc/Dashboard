"""Backtest aggregator (Phase D) — builds the L1–L18 × 5 regimes × N windows
metrics grid from a flat trade DataFrame.

Cell levels (gameplan §7.2):

  L1:  Setup × CMC-node × Direction        (per-instrument, per-direction)
  L2:  Setup × CMC-node                    (BOTH directions combined)
  L3:  Setup × Asset-class                 (SR3 only for now → collapses with L4)
  L4:  Setup overall                       (across all CMC nodes)
  L5:  TREND_COMPOSITE × node × dir × thresh
  L6:  TREND_COMPOSITE × node × thresh
  L7:  TREND_COMPOSITE overall × thresh
  L8:  MR_COMPOSITE × node × dir × thresh
  L9:  MR_COMPOSITE × node × thresh
  L10: MR_COMPOSITE overall × thresh
  L11: FINAL_COMPOSITE × node × dir × thresh
  L12: FINAL_COMPOSITE × node × thresh
  L13: FINAL_COMPOSITE overall × thresh
  L14: Per-CMC-node all-setups blended
  L15: STIR setup × market × instrument (currently-tradable only)
  L16: STIR setup × market × instrument (historical only)
  L17: STIR setup × market × continuous (covered in L1-L4 here)
  L18: STIR setup × market × structure_kind (rolled up by scope)

Windows (5):
  ALL / LAST_2Y / LAST_1Y / LAST_6M / LAST_30D

Regimes (5, gameplan §7.5):
  ALL / TRENDING (ADX≥25) / RANGING (ADX<20) /
  HIGH_VOL (ATR pctile > 0.75) / LOW_VOL (ATR pctile < 0.25)

For each cell × window × regime we compute the metrics pack from
``lib.backtest.metrics.compute_metrics``.

Cell-trend interpretation: for each cell, we compare the metric (default
``avg_R_realized``) across the 5 windows and tag the cell as
**Improving / Declining / Stable / Mixed** using a Spearman-style
recency-rank correlation.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from lib.backtest.metrics import compute_metrics, compute_calendar_slices


# =============================================================================
# Window definitions (cumulative trailing windows)
# =============================================================================

WINDOWS: tuple = ("ALL", "LAST_2Y", "LAST_1Y", "LAST_6M", "LAST_30D")
WINDOW_DAYS: dict = {
    "ALL":      None,
    "LAST_2Y":  730,
    "LAST_1Y":  365,
    "LAST_6M":  180,
    "LAST_30D": 30,
}


REGIMES: tuple = ("ALL", "TRENDING", "RANGING", "HIGH_VOL", "LOW_VOL")


# =============================================================================
# Trend interpretation
# =============================================================================

def interpret_window_trend(metric_by_window: dict[str, Optional[float]],
                              metric_name: str = "metric") -> str:
    """Compare a metric across windows and label the cell as Improving /
    Declining / Stable / Mixed.

    Convention: window list is ordered from longest (oldest) to shortest
    (most recent): ALL → LAST_2Y → LAST_1Y → LAST_6M → LAST_30D. If the
    metric monotonically improves toward the right (recent), tag
    "Improving". If monotonically degrades, "Declining". If essentially
    flat, "Stable". Otherwise "Mixed".

    Returns a single-line interpretation string suitable for tooltip.
    """
    ordered = [(w, metric_by_window.get(w)) for w in WINDOWS]
    valid = [(w, v) for w, v in ordered if v is not None and np.isfinite(v)]
    if len(valid) < 3:
        return f"insufficient data ({len(valid)} windows)"

    values = [v for _, v in valid]
    recency_rank = list(range(len(valid)))    # 0 = oldest, n-1 = most recent
    # Spearman correlation = correlation of ranks
    if len(set(values)) == 1:
        return f"Stable — {metric_name} ~{values[0]:.2f} across all windows"
    val_ranks = pd.Series(values).rank().values
    rec_ranks = np.array(recency_rank, dtype=float)
    if np.std(val_ranks) == 0 or np.std(rec_ranks) == 0:
        rho = 0.0
    else:
        rho = float(np.corrcoef(val_ranks, rec_ranks)[0, 1])
    parts = " > ".join(f"{v:+.2f} ({w})" for w, v in valid)
    if rho >= 0.7:
        return f"Improving — {metric_name} {parts}"
    if rho <= -0.7:
        return f"Declining — {metric_name} {parts}"
    if abs(rho) < 0.3:
        spread = max(values) - min(values)
        return f"Stable — {metric_name} ~{np.mean(values):+.2f} ± {spread/2:.2f}"
    return f"Mixed — {metric_name} {parts}"


# =============================================================================
# Window / regime filters
# =============================================================================

def _filter_window(trades: pd.DataFrame, window: str,
                     asof_date: date) -> pd.DataFrame:
    """Return the subset of trades whose ``entry_dt`` falls within
    ``window`` (counted from ``asof_date``)."""
    if window == "ALL" or trades is None or trades.empty:
        return trades
    days = WINDOW_DAYS[window]
    cutoff = pd.Timestamp(asof_date) - pd.Timedelta(days=days)
    entry = pd.to_datetime(trades["entry_dt"])
    return trades[entry >= cutoff].copy()


def _filter_regime(trades: pd.DataFrame, regime: str) -> pd.DataFrame:
    if regime == "ALL" or trades is None or trades.empty:
        return trades
    if "regime_at_entry" not in trades.columns:
        return trades.iloc[0:0]
    if regime == "TRENDING":
        return trades[trades["regime_at_entry"].str.startswith("TRENDING")]
    if regime == "RANGING":
        return trades[trades["regime_at_entry"].str.startswith("RANGING")]
    if regime == "HIGH_VOL":
        return trades[trades["regime_at_entry"].str.endswith("HIGH_VOL")]
    if regime == "LOW_VOL":
        return trades[trades["regime_at_entry"].str.endswith("LOW_VOL")]
    return trades.iloc[0:0]


# =============================================================================
# Cell-key generator
# =============================================================================

def _cell_keys_for_levels(trades: pd.DataFrame) -> list[dict]:
    """Generate (level, cell_keys_dict) tuples for every L1-L4 cell that
    has at least one trade. L5-L18 are placeholders covered by future
    composite-trade emissions.
    """
    out = []
    if trades is None or trades.empty:
        return out

    setups = trades["setup_id"].dropna().unique()
    nodes = trades["cmc_node"].dropna().unique() if "cmc_node" in trades.columns else []
    directions = trades["direction"].dropna().unique() if "direction" in trades.columns else []

    # L1: Setup × CMC-node × Direction
    for setup in setups:
        for node in nodes:
            for d in directions:
                sub = trades[(trades["setup_id"] == setup)
                             & (trades["cmc_node"] == node)
                             & (trades["direction"] == d)]
                if not sub.empty:
                    out.append({
                        "level": "L1",
                        "setup_id": setup, "cmc_node": node, "direction": d,
                        "_trades": sub,
                    })

    # L2: Setup × CMC-node (both directions)
    for setup in setups:
        for node in nodes:
            sub = trades[(trades["setup_id"] == setup)
                         & (trades["cmc_node"] == node)]
            if not sub.empty:
                out.append({
                    "level": "L2",
                    "setup_id": setup, "cmc_node": node, "direction": "BOTH",
                    "_trades": sub,
                })

    # L4: Setup overall (across all nodes)
    for setup in setups:
        sub = trades[trades["setup_id"] == setup]
        if not sub.empty:
            out.append({
                "level": "L4",
                "setup_id": setup, "cmc_node": "ALL", "direction": "BOTH",
                "_trades": sub,
            })

    # L14: Per-CMC-node all-setups blended
    for node in nodes:
        sub = trades[trades["cmc_node"] == node]
        if not sub.empty:
            out.append({
                "level": "L14",
                "setup_id": "ALL", "cmc_node": node, "direction": "BOTH",
                "_trades": sub,
            })

    return out


# =============================================================================
# Public aggregator
# =============================================================================

def build_metrics_grid(trades: pd.DataFrame,
                          asof_date: date,
                          n_bootstrap: int = 200,
                          interpret_metric: str = "avg_R_realized") -> tuple:
    """Build the full metrics grid: one row per (cell × window × regime).

    Returns ``(metrics_df, calendar_df, trend_df)``:
        metrics_df: rows = cell × window × regime, columns = all metrics
        calendar_df: by-month / by-DoW / by-year aggregates per cell × window
        trend_df: one row per cell with the cross-window trend
                   interpretation (Improving / Declining / Stable / Mixed).
    """
    cell_keys = _cell_keys_for_levels(trades)
    metrics_rows = []
    calendar_rows = []
    trend_rows = []

    for cell in cell_keys:
        cell_trades = cell.pop("_trades")
        cell_metric_per_window = {}
        for window in WINDOWS:
            window_trades = _filter_window(cell_trades, window, asof_date)
            for regime in REGIMES:
                regime_trades = _filter_regime(window_trades, regime)
                m = compute_metrics(regime_trades, n_bootstrap=n_bootstrap)
                row = {
                    "level": cell["level"],
                    "setup_id": cell["setup_id"],
                    "cmc_node": cell["cmc_node"],
                    "direction": cell["direction"],
                    "window": window, "regime": regime,
                    **m,
                }
                metrics_rows.append(row)
                if regime == "ALL":
                    cell_metric_per_window[window] = m.get(interpret_metric)
            # Calendar slices once per (cell × window) at regime=ALL
            if window == "ALL":
                cal = compute_calendar_slices(window_trades)
                for slice_name, sl in cal.items():
                    if sl is None or sl.empty:
                        continue
                    sl = sl.copy()
                    sl["slice_name"] = slice_name
                    sl["level"] = cell["level"]
                    sl["setup_id"] = cell["setup_id"]
                    sl["cmc_node"] = cell["cmc_node"]
                    sl["direction"] = cell["direction"]
                    calendar_rows.append(sl)
        # Trend interpretation for the cell
        trend_text = interpret_window_trend(cell_metric_per_window, interpret_metric)
        trend_rows.append({
            "level": cell["level"],
            "setup_id": cell["setup_id"],
            "cmc_node": cell["cmc_node"],
            "direction": cell["direction"],
            "trend_metric": interpret_metric,
            "trend_text": trend_text,
            **{f"{w}_value": cell_metric_per_window.get(w) for w in WINDOWS},
        })

    metrics_df = pd.DataFrame(metrics_rows)
    calendar_df = pd.concat(calendar_rows, ignore_index=True) if calendar_rows else pd.DataFrame()
    trend_df = pd.DataFrame(trend_rows)
    return (metrics_df, calendar_df, trend_df)
