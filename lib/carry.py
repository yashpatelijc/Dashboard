"""Carry / roll-down calculations for SRA contracts.

Three notions of carry (Ilmanen 2011 chapter 14):

A. Single-contract roll-down — for two adjacent quarterly contracts i and i+1:
        carry_per_day_bps_i = (rate_{i+1} - rate_i) / days_between_expiries × 100
   Interpretation: holding contract i, you earn the ride down the curve toward
   contract i+1's level. Positive in upward-sloping curves.

B. Cumulative carry to expiry — sum of single-contract carry days × bps from
   today through contract's expiry. Useful for "how many bp of carry am I locking in"

C. White-pack carry — average carry/day across the front 4 quarterly contracts.
   The standard cross-market ranking driver in cross-product carry trades.

All carry numbers below are computed from the close-price implied rate
(100 - close), so positive carry means the curve is upward-sloping at that point.
"""
from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from lib.fomc import is_quarterly, parse_sra_outright, reference_period


def implied_rate(close: float) -> float:
    """100 - close (IMM convention)."""
    if close is None or pd.isna(close):
        return None
    return 100.0 - close


def days_between(a: date, b: date) -> int:
    return max(1, (b - a).days)


def compute_per_contract_carry(close_by_symbol: dict, contracts: list) -> dict:
    """Compute single-contract roll-down carry per contract (bp / day).

    For contract i (where contract i+1 exists), carry_i = (rate_{i+1} − rate_i) /
    days_between(expiry_i, expiry_{i+1}) × 100.

    The LAST contract has no successor → carry = None.
    """
    out = {}
    n = len(contracts)
    for i, sym in enumerate(contracts):
        if i == n - 1:
            out[sym] = None
            continue
        rp_i = reference_period(sym)
        rp_n = reference_period(contracts[i + 1])
        c_i = close_by_symbol.get(sym)
        c_n = close_by_symbol.get(contracts[i + 1])
        if (rp_i is None or rp_n is None or c_i is None or c_n is None
                or pd.isna(c_i) or pd.isna(c_n)):
            out[sym] = None
            continue
        rate_i = 100 - c_i
        rate_n = 100 - c_n
        # IMM dates of consecutive contracts — use start of reference period
        days = days_between(rp_i[0], rp_n[0])
        out[sym] = (rate_n - rate_i) / days * 100.0
    return out


def compute_carry_to_expiry(close_by_symbol: dict, contracts: list,
                            asof_date: date) -> dict:
    """Cumulative bps of carry from asof_date forward to each contract's reference start."""
    per_day = compute_per_contract_carry(close_by_symbol, contracts)
    out = {}
    for sym in contracts:
        rp = reference_period(sym)
        if rp is None:
            out[sym] = None
            continue
        days = days_between(asof_date, rp[0])
        cd = per_day.get(sym)
        out[sym] = cd * days if cd is not None else None
    return out


def compute_white_pack_carry(close_by_symbol: dict, contracts: list) -> Optional[float]:
    """Average single-contract carry (bp/day) across the front-4 QUARTERLY contracts."""
    quarterlies = [s for s in contracts if is_quarterly(s)]
    if len(quarterlies) < 4:
        return None
    front4 = quarterlies[:4]
    pc = compute_per_contract_carry(close_by_symbol, contracts)
    vals = [pc.get(s) for s in front4 if pc.get(s) is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def compute_carry_table(wide_close: pd.DataFrame, contracts: list, asof_date: date,
                        lookback_periods: list = (1, 5, 30)) -> pd.DataFrame:
    """Compute carry table with multiple horizons.

    Returns DataFrame indexed by symbol with columns:
        rate · expiry_days · carry_bp_per_day · carry_5d_bp · carry_30d_bp ·
        cum_to_expiry_bp · cum_carry_5d_avg · cum_carry_30d_avg
    """
    try:
        today_row = wide_close.loc[wide_close.index.date == asof_date].iloc[0]
    except Exception:
        return pd.DataFrame()
    close_today = {c: today_row.get(c) for c in contracts}
    pc_today = compute_per_contract_carry(close_today, contracts)
    cte_today = compute_carry_to_expiry(close_today, contracts, asof_date)

    rows = []
    for sym in contracts:
        rp = reference_period(sym)
        rate = implied_rate(close_today.get(sym))
        days_exp = days_between(asof_date, rp[0]) if rp else None
        rows.append({
            "symbol": sym,
            "rate_pct": rate,
            "expiry_days": days_exp,
            "carry_bp_per_day": pc_today.get(sym),
            "cum_to_expiry_bp": cte_today.get(sym),
        })
    df = pd.DataFrame(rows).set_index("symbol")

    # Multiple-period rolling carry — average per-day carry over last N days
    ts = pd.Timestamp(asof_date)
    for n in lookback_periods:
        if n == 1:
            continue
        history = wide_close.loc[wide_close.index <= ts].tail(n)
        per_day_history = []
        for d_idx, d_row in history.iterrows():
            close_d = {c: d_row.get(c) for c in contracts}
            per_day_history.append(compute_per_contract_carry(close_d, contracts))
        # Average carry/day per contract over the window
        avg_carry = {}
        for sym in contracts:
            vals = [d.get(sym) for d in per_day_history
                    if d.get(sym) is not None and not pd.isna(d.get(sym))]
            avg_carry[sym] = sum(vals) / len(vals) if vals else None
        col = f"carry_avg_{n}d_bp_per_day"
        df[col] = pd.Series(avg_carry)

    return df
