"""Lightweight 60-day setup track record.

For each setup, we replay the detector across the prior 60 trading days
(per-contract panel as-of each historical date) and compute:

  · fires_60d           total LONG+SHORT fires over the window
  · win_rate_5bar       % of fires where the trade direction was profitable at +5 bars
  · mean_5bar_return_bp average +5-bar return (in bp, sign-adjusted by direction)
  · sample_quality      "STRONG" if fires≥10, "OK" if 3-9, "WEAK" if 1-2, "NO_TRACK_RECORD" if 0

This is NOT a full backtest engine — there's no cooldown, no overlap handling,
no commissions, no slippage. It's a quick "does this setup currently work
on this universe?" filter to gate which fires the user should act on.

Performance: scans 60d × N contracts × ~22 detectors. With caching we keep
the cost bounded — runs only when scan_universe() runs.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from lib.contract_units import bp_multiplier, load_catalog
from lib.setups.base import SetupResult
from lib.setups.mean_reversion import (
    detect_b1, detect_b3, detect_b5, detect_b6, detect_b10, detect_b11, detect_b13,
)
from lib.setups.stir import detect_c4, detect_c5
from lib.setups.trend import (
    detect_a1, detect_a2, detect_a3, detect_a4, detect_a5, detect_a6,
    detect_a8, detect_a10, detect_a11, detect_a12a, detect_a12b, detect_a15,
)


# Setups eligible for per-contract historical replay (excluding C3/C8/C9 which
# need universe-level data — too expensive to recompute 60× over 29 outrights)
PER_CONTRACT_DETECTORS = {
    "A1": detect_a1, "A2": detect_a2, "A3": detect_a3, "A4": detect_a4,
    "A5": detect_a5, "A6": detect_a6, "A8": detect_a8, "A10": detect_a10,
    "A11": detect_a11, "A12a": detect_a12a, "A12b": detect_a12b, "A15": detect_a15,
    "B1": detect_b1, "B3": detect_b3, "B5": detect_b5, "B6": detect_b6,
    "B10": detect_b10, "B11": detect_b11, "B13": detect_b13,
}


def _scope_for(strategy: str) -> str:
    return strategy   # already in canonical form


def _eval_panel_at(panel: pd.DataFrame, sym: str, eval_date: date,
                    bp_mult: float, scope: str,
                    setup_ids: list) -> dict:
    """Run the supplied detectors on the panel as-of ``eval_date``. Return
    {setup_id: SetupResult}."""
    out = {}
    for sid in setup_ids:
        det = PER_CONTRACT_DETECTORS.get(sid)
        if det is None:
            continue
        # Truncate panel to eval_date inclusive (so detectors see only that snapshot)
        sub = panel.loc[panel.index <= pd.Timestamp(eval_date)]
        if sub.empty:
            continue
        try:
            r = det(sub, eval_date, bp_mult, scope=scope)
            out[sid] = r
        except Exception:
            out[sid] = None
    # STIR per-contract MR (C4 for fly, C5 for spread)
    if scope == "fly":
        try:
            r = detect_c4(panel.loc[panel.index <= pd.Timestamp(eval_date)],
                           eval_date, bp_mult, scope="fly")
            out["C4"] = r
        except Exception:
            out["C4"] = None
    elif scope == "spread":
        try:
            r = detect_c5(panel.loc[panel.index <= pd.Timestamp(eval_date)],
                           eval_date, bp_mult, scope="spread")
            out["C5"] = r
        except Exception:
            out["C5"] = None
    return out


def compute_track_record(panels: dict, asof_date: date, strategy: str,
                           window_days: int = 60,
                           forward_bars: int = 5) -> dict:
    """For each applicable per-contract setup, walk back ``window_days`` and
    count fires + +``forward_bars``-bar outcome. Returns:

      {setup_id: {fires_60d, wins, losses, win_rate_5bar,
                  mean_5bar_return_bp, sample_quality}}
    """
    cat = load_catalog()
    scope = _scope_for(strategy)

    # Pick setup ids based on scope
    if scope == "outright":
        setup_ids = ["A1", "A2", "A3", "A4", "A5", "A6", "A8", "A10", "A11",
                      "A12a", "A12b", "A15", "B1", "B3", "B5", "B6", "B10", "B11", "B13"]
    elif scope == "spread":
        setup_ids = ["A1", "A2", "A3", "A4", "A5", "A8", "A10", "A11", "A12a", "A12b",
                      "A15", "B1", "B3", "B5", "B6", "B10", "B11", "B13", "C5"]
    elif scope == "fly":
        setup_ids = ["A1", "A2", "A3", "A4", "A5", "A8", "A10", "A11", "A12a", "A12b",
                      "A15", "B1", "B3", "B5", "B6", "B10", "B11", "B13", "C4"]
    else:
        setup_ids = []

    # Initialise accumulators
    accum = {sid: {"fires": [], "outcomes": []} for sid in setup_ids}

    # Build the eval-date list — last `window_days` PRIOR business days
    eval_dates = []
    cursor = pd.Timestamp(asof_date) - pd.Timedelta(days=1)
    while len(eval_dates) < window_days and cursor > pd.Timestamp(asof_date) - pd.Timedelta(days=int(window_days * 1.6) + 7):
        eval_dates.append(cursor.date())
        cursor -= pd.Timedelta(days=1)
    eval_dates = list(reversed(eval_dates))

    # Per-contract historical replay (base_product auto-detected per symbol)
    try:
        from lib.markets import parse_symbol_to_base_product as _psbp
    except Exception:
        _psbp = lambda s: "SRA"
    for sym, panel in panels.items():
        if panel is None or panel.empty:
            continue
        bp_mult = bp_multiplier(sym, _psbp(sym) or "SRA", cat)
        for ed in eval_dates:
            ts = pd.Timestamp(ed)
            # Need bar at eval date AND bar at eval date + forward_bars
            available_dates = panel.index[panel.index <= ts + pd.Timedelta(days=forward_bars * 2 + 1)]
            if len(available_dates) < 2:
                continue
            if (panel.index.normalize() == ts.normalize()).sum() == 0:
                continue
            results = _eval_panel_at(panel, sym, ed, bp_mult, scope, setup_ids)
            # +N-bar outcome — find bar `forward_bars` bars after eval date
            entry_idx = panel.index.searchsorted(ts, side="right") - 1
            if entry_idx < 0 or entry_idx + forward_bars >= len(panel):
                continue
            entry_close = float(panel["close"].iloc[entry_idx])
            forward_close = float(panel["close"].iloc[entry_idx + forward_bars])
            for sid, r in results.items():
                if r is None or r.state != "FIRED":
                    continue
                accum[sid]["fires"].append((sym, ed))
                # +N-bar return in bp (signed by direction)
                if r.direction == "LONG":
                    ret_bp = (forward_close - entry_close) * bp_mult
                else:
                    ret_bp = (entry_close - forward_close) * bp_mult
                accum[sid]["outcomes"].append(ret_bp)

    # Aggregate
    out = {}
    for sid in setup_ids:
        outs = accum[sid]["outcomes"]
        n = len(outs)
        if n == 0:
            out[sid] = {"fires_60d": 0, "wins": 0, "losses": 0,
                          "win_rate_5bar": None,
                          "mean_5bar_return_bp": None,
                          "sample_quality": "NO_TRACK_RECORD"}
            continue
        wins = int(sum(1 for x in outs if x > 0))
        losses = int(sum(1 for x in outs if x < 0))
        out[sid] = {
            "fires_60d": n,
            "wins": wins,
            "losses": losses,
            "win_rate_5bar": (wins / n * 100.0) if n else None,
            "mean_5bar_return_bp": float(np.mean(outs)),
            "sample_quality": "STRONG" if n >= 10 else "OK" if n >= 3 else "WEAK",
        }
    return out
