"""A10 — Turn / QE / YE adjuster.

Quarter-end and year-end "turns" inflate SOFR and SR3 implied rates as
balance-sheet pressure pushes funding rates higher in the last few days of
the period.  An SR3 contract whose 3-month reference period straddles such a
turn carries a transient yield premium that PCA factor models tend to attribute
incorrectly to PC1 (level) when it is actually a calendar-effect.

This module:

  1. Identifies which contracts' reference quarters span a turn:
       · year-end turn          — 31 Dec
       · quarter-end turn       — 31 Mar / 30 Jun / 30 Sep
       · QT/QE-induced turns    — based on Fed balance-sheet calendar (best-effort
                                   from announced QT/QE schedule; otherwise None)
  2. Estimates the historical turn premium (bp) per contract reference period:
       compares post-turn vs pre-turn anchor metric over the trailing 5 turns
       of the same type (e.g. last 5 year-ends).
  3. Returns a per-contract turn-adjusted "fair fwd yield" = raw fwd yield −
     historical turn premium (when applicable).

The adjustment is conservative: for contracts not spanning a turn, the
adjustment is 0.  For contracts spanning a turn, we subtract the median
historical turn premium of the same type.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from lib.fomc import parse_sra_outright, reference_period


# Turn types
TURN_TYPES = ("YEAR_END", "QUARTER_END")


def _turn_dates_in_window(start: date, end: date) -> list:
    """Return list of (turn_date, turn_type) for turns inside [start, end]."""
    out = []
    for y in range(start.year, end.year + 2):
        # Year end
        ye = date(y, 12, 31)
        if start <= ye <= end:
            out.append((ye, "YEAR_END"))
        # Quarter ends
        for m in (3, 6, 9):
            d = date(y, m, 1)
            # last day of (m)
            if m == 3:
                qe = date(y, 3, 31)
            elif m == 6:
                qe = date(y, 6, 30)
            elif m == 9:
                qe = date(y, 9, 30)
            if start <= qe <= end:
                out.append((qe, "QUARTER_END"))
    return sorted(out, key=lambda x: x[0])


def contract_turn_exposure(symbol: str) -> dict:
    """Return turn-spanning info for an SRA contract.

    Output:
      {
        "spans_year_end":   bool,
        "spans_quarter_end": bool,
        "turns":            list of (date, type) inside the ref-quarter,
        "ref_period":       (start, end) tuple or None,
      }
    """
    out = {"spans_year_end": False, "spans_quarter_end": False,
           "turns": [], "ref_period": None}
    rp = reference_period(symbol)
    if rp is None:
        return out
    start, end = rp
    out["ref_period"] = rp
    turns = _turn_dates_in_window(start, end)
    out["turns"] = turns
    out["spans_year_end"] = any(t == "YEAR_END" for _, t in turns)
    out["spans_quarter_end"] = any(t == "QUARTER_END" for _, t in turns)
    return out


def historical_turn_premium_bp(sofr_panel: pd.DataFrame,
                                 turn_type: str = "YEAR_END",
                                 *, lookback_n_turns: int = 5,
                                 window_d_pre: int = 7,
                                 window_d_post: int = 7) -> dict:
    """Estimate the historical turn premium in SOFR rate (bp).

    For each of the last `lookback_n_turns` turns of `turn_type`, compare the
    SOFR rate AT the turn (last day of the period) to the trailing `window_d_pre`-d
    average and the next `window_d_post`-d average.

    Returns:
      {
        "median_premium_bp": float,
        "mean_premium_bp":   float,
        "p25_premium_bp":    float,
        "p75_premium_bp":    float,
        "n_turns":           int,
        "premiums":          list of (turn_date, premium_bp),
      }
    """
    out = {"median_premium_bp": float("nan"), "mean_premium_bp": float("nan"),
           "p25_premium_bp": float("nan"), "p75_premium_bp": float("nan"),
           "n_turns": 0, "premiums": []}
    if sofr_panel is None or sofr_panel.empty:
        return out
    if not isinstance(sofr_panel.index, pd.DatetimeIndex):
        return out
    # Pick SOFR-like series in priority order. SOFR (overnight rate) DOES spike
    # at turns; FDTR (target rate) does NOT. So prefer 'sofr' first.
    series = None
    for col in ("sofr", "sofr_rate", "SOFRRATE_Index", "PX_LAST"):
        if col in sofr_panel.columns:
            series = sofr_panel[col].dropna()
            break
    if series is None or series.empty:
        for col in sofr_panel.columns:
            if pd.api.types.is_numeric_dtype(sofr_panel[col]):
                series = sofr_panel[col].dropna()
                break
    if series is None or series.empty:
        return out

    # All turns inside the SOFR sample
    s_start = series.index.min().date()
    s_end = series.index.max().date()
    turns = [t for t, ttype in _turn_dates_in_window(s_start, s_end)
              if ttype == turn_type]
    if not turns:
        return out
    turns = turns[-lookback_n_turns:]

    premiums = []
    for tdate in turns:
        ts = pd.Timestamp(tdate)
        # Window of 1 trading day on either side of the turn
        try:
            pre_window = series.loc[ts - timedelta(days=window_d_pre):
                                       ts - timedelta(days=1)]
            post_window = series.loc[ts + timedelta(days=1):
                                        ts + timedelta(days=window_d_post)]
            # Closest trading day TO the turn
            on_turn = series.loc[ts - timedelta(days=3): ts + timedelta(days=3)]
            on_turn = on_turn.loc[on_turn.index <= ts]
            if on_turn.empty or pre_window.empty or post_window.empty:
                continue
            pre_avg = float(pre_window.mean())
            post_avg = float(post_window.mean())
            on_value = float(on_turn.iloc[-1])
            # Premium = peak rate at turn − average of pre/post windows
            baseline = (pre_avg + post_avg) / 2.0
            prem_bp = (on_value - baseline) * 100.0
            premiums.append((tdate, prem_bp))
        except Exception:
            continue

    if not premiums:
        return out
    bp = np.asarray([p[1] for p in premiums], dtype=float)
    out.update({
        "median_premium_bp": float(np.median(bp)),
        "mean_premium_bp": float(np.mean(bp)),
        "p25_premium_bp": float(np.quantile(bp, 0.25)),
        "p75_premium_bp": float(np.quantile(bp, 0.75)),
        "n_turns": int(len(bp)),
        "premiums": [(d, float(b)) for d, b in premiums],
    })
    return out


def turn_adjustment_for_contract(symbol: str,
                                    sofr_panel: pd.DataFrame,
                                    *, lookback_n_turns: int = 5) -> dict:
    """Per-contract turn adjustment.

    Returns:
      {
        "applies":            bool,        — does this contract span a turn?
        "turn_types":         list,
        "turn_dates":         list of dates,
        "premium_bp":         estimated turn premium (bp), 0.0 if not applicable,
        "fair_yield_adjustment_bp": signed adjustment to subtract from raw forward yield,
        "n_historical_turns": int,
      }
    """
    out = {"applies": False, "turn_types": [], "turn_dates": [],
           "premium_bp": 0.0, "fair_yield_adjustment_bp": 0.0,
           "n_historical_turns": 0}
    expo = contract_turn_exposure(symbol)
    if not expo["turns"]:
        return out
    out["applies"] = True
    out["turn_types"] = list(set(t for _, t in expo["turns"]))
    out["turn_dates"] = [d for d, _ in expo["turns"]]

    total_premium = 0.0
    total_n = 0
    for ttype in out["turn_types"]:
        h = historical_turn_premium_bp(sofr_panel, turn_type=ttype,
                                          lookback_n_turns=lookback_n_turns)
        prem = h.get("median_premium_bp")
        if prem is not None and np.isfinite(prem):
            # Spread the daily premium across the 90d ref-period:
            # the contract sees ~1 day of turn premium, scaled to its quarter rate
            # Premium contribution to the 3M rate ≈ 1/90 of the 1-day spike
            scaled = prem * (1.0 / 90.0)
            total_premium += scaled
            total_n += int(h.get("n_turns", 0))
    out["premium_bp"] = total_premium
    out["fair_yield_adjustment_bp"] = total_premium    # subtract from raw fwd yield to "de-turn"
    out["n_historical_turns"] = total_n
    return out
