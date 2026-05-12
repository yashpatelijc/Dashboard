"""Constant Maturity Curve (CMC) construction for STIR futures.

Backend for the Technicals subtab's setup detection on stable, curve-position-
locked time series. Methodology per industry standard:

- **Roll rule**: 5 business days before observed last-trade-date, with volume
  sanity check (skip the roll if next-contract volume hasn't exceeded the
  front for 3 consecutive sessions).
- **Adjustment**: Backward Panama (additive). At each roll boundary,
  ``gap = new_close - old_close`` on the same date; add ``gap`` to ALL prior
  bars of the rolling continuous series. Mandatory for STIR because (a) ratio
  adjustment breaks across zero / negative rates, and (b) prices live near 100
  so additive preserves rate-space differences exactly.
- **Constant-maturity sampling**: linear interpolation between bracketing
  listed contracts on days-to-expiry. ``CMC(t, T) = (1-w)*adj_close_c1 +
  w*adj_close_c2`` where ``w = (T - dte_c1) / (dte_c2 - dte_c1)``.
- **Spread/fly construction**: derived from CMC outright nodes.
  ``M3M6_spread = M3_outright - M6_outright``;
  ``M3M6M9_fly = M3 - 2*M6 + M9``.
- **Carver correction**: for any %-based indicator (RSI, BB-width, Z-score
  of returns), the numerator uses back-adjusted price *differences* but the
  denominator uses the *raw* listed contract's close at that bar.

References live in HANDOFF.md (§6, once Phase A.8 lands) and the plan file at
``C:\\Users\\yash.patel\\.claude\\plans\\c-users-yash-patel-downloads-tmia-v13-s-magical-mitten.md``.

Public API:
    list_cmc_nodes(scope) -> list[str]
    build_cmc_nodes(scope, asof_date, history_years=5) -> Path  # writes parquet
    load_cmc_panel(scope, asof_date) -> pd.DataFrame
    get_cmc_roll_log(scope, asof_date) -> pd.DataFrame  # forensics
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from lib.connections import get_ohlc_connection


# =============================================================================
# Configuration
# =============================================================================

# Cache directory at project root. Relative to this file: ../.cmc_cache
_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cmc_cache"
_CACHE_DIR.mkdir(exist_ok=True)

# Builder version — bump whenever the algorithm or output schema changes so
# stale caches are detected and rebuilt.
BUILDER_VERSION = "1.0.0"

# Default tenor nodes (in months) for the CMC outright curve.
# Phase 2 (plan §12 Phase 2): expanded from 10 nodes to a full monthly grid
# M0..M60 = 61 nodes. The framework specifies "58-node monthly grid (3M..60M)";
# we add M0/M1/M2 for front-month proximity (used by A12d B0/B1 axes).
# The original 10-node subset (0,3,6,9,12,18,24,36,48,60) is retained as
# OUTRIGHT_NODES_MONTHS_LEGACY for callers that want the smaller grid.
OUTRIGHT_NODES_MONTHS: tuple[int, ...] = tuple(range(0, 61))   # 61 nodes M0..M60
OUTRIGHT_NODES_MONTHS_LEGACY: tuple[int, ...] = (0, 3, 6, 9, 12, 18, 24, 36, 48, 60)

# Spread nodes — paired (front_M, back_M) months. Standard 3M / 6M / 12M /
# 24M tenor pairs across the curve.
SPREAD_NODES_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 3), (3, 6), (6, 9), (9, 12),
    (0, 6), (0, 12),
    (6, 12), (12, 24),
)

# Fly nodes — 1:2:1 triplets pulled from outright CMC nodes.
FLY_NODES_TRIPLES: tuple[tuple[int, int, int], ...] = (
    (0, 3, 6),
    (3, 6, 9),
    (6, 9, 12),
    (0, 6, 12),
)

# Roll-rule parameters (industry standard per Norgate / Pinnacle / Carver).
ROLL_OFFSET_BD = 5                    # business days before LTD
VOL_GUARD_SESSIONS = 3                # require N consecutive sessions of
                                       # next-contract volume > front before
                                       # honouring the calendar roll

# Quarterly month codes (H=Mar, M=Jun, U=Sep, Z=Dec) — standard SR3 quarterlies.
_QUARTERLY_MONTHS = {3, 6, 9, 12}

# Days/year approximation for tenor-to-business-days conversion.
# 252 trading days / 12 months ≈ 21 BD per month.
_BD_PER_MONTH = 21


# =============================================================================
# Public node-list helpers
# =============================================================================

def _outright_node_id(months: int) -> str:
    """Canonical id: M0 / M3 / M6 / ..."""
    return f"M{int(months)}"


def _spread_node_id(front_m: int, back_m: int) -> str:
    """Canonical id: M0_M3, M3_M6, M6_M12, ..."""
    return f"M{int(front_m)}_M{int(back_m)}"


def _fly_node_id(left_m: int, mid_m: int, right_m: int) -> str:
    """Canonical id: M0_M3_M6, M3_M6_M9, ..."""
    return f"M{int(left_m)}_M{int(mid_m)}_M{int(right_m)}"


def list_cmc_nodes(scope: str) -> list[str]:
    """Return the canonical node ids defined for ``scope``.

    Parameters
    ----------
    scope : str
        ``'outright'`` | ``'spread'`` | ``'fly'``.
    """
    if scope == "outright":
        return [_outright_node_id(m) for m in OUTRIGHT_NODES_MONTHS]
    if scope == "spread":
        return [_spread_node_id(f, b) for f, b in SPREAD_NODES_PAIRS]
    if scope == "fly":
        return [_fly_node_id(l, m, r) for l, m, r in FLY_NODES_TRIPLES]
    raise ValueError(f"unknown scope: {scope!r}")


# =============================================================================
# Phase A.1 — contract chain
# =============================================================================

@dataclass(frozen=True)
class ContractRow:
    """One quarterly outright contract with everything we need for chaining.

    ``ltd_canonical`` is computed from the catalog's ``expiry_year`` /
    ``expiry_month`` per the SR3 rule (day before 3rd Wednesday of
    ``expiry_month + 3``). This is authoritative for live contracts whose
    ``ltd_observed`` (max bar in the timeseries) trails the true LTD.

    ``ltd_observed`` is kept for forensics / cross-validation against
    retired contracts whose data naturally ends at LTD.
    """
    symbol: str
    expiry_year: int
    expiry_month: int
    first_bar: date
    ltd_observed: date
    ltd_canonical: date


def _third_wednesday(year: int, month: int) -> date:
    """The 3rd Wednesday of (year, month). Monday = 0 in Python's weekday()."""
    first = date(year, month, 1)
    # Days from 1st of month to the first Wednesday: (2 - weekday) mod 7
    days_to_first_wed = (2 - first.weekday()) % 7
    first_wed = first + timedelta(days=days_to_first_wed)
    return first_wed + timedelta(days=14)


def _ltd_canonical_sr3(expiry_year: int, expiry_month: int) -> date:
    """SR3 quarterly LTD rule: day before the 3rd Wednesday of the month
    that is 3 months after the contract's named month.

    SRAM26 (named Jun 2026) → LTD = day before 3rd Wed of Sep 2026 = Sep 16, 2026.
    """
    ltd_month = expiry_month + 3
    ltd_year = expiry_year
    if ltd_month > 12:
        ltd_month -= 12
        ltd_year += 1
    return _third_wednesday(ltd_year, ltd_month) - timedelta(days=1)


def _load_quarterly_chain(base_product: str = "SRA") -> pd.DataFrame:
    """Return one row per quarterly outright contract for ``base_product``.

    Columns: ``symbol, expiry_year, expiry_month, first_bar, ltd_observed,
    ltd_canonical, n_bars``. Filtered to quarterly month codes
    (H/M/U/Z = 3/6/9/12) only — the universe SR3 CMC outright nodes are
    interpolated against.

    Sorted by ``ltd_canonical`` ascending. Built per call (no in-process cache
    yet — CMC builder caches the parquet output downstream).
    """
    con = get_ohlc_connection()
    if con is None:
        raise RuntimeError("OHLC database unavailable")
    quarterly_set = ",".join(str(m) for m in sorted(_QUARTERLY_MONTHS))
    sql = f"""
        SELECT
            cc.symbol,
            cc.expiry_year,
            cc.expiry_month,
            MIN(DATE(to_timestamp(t.time/1000.0))) AS first_bar,
            MAX(DATE(to_timestamp(t.time/1000.0))) AS ltd_observed,
            COUNT(*) AS n_bars
        FROM mde2_timeseries t
        JOIN mde2_contracts_catalog cc ON cc.symbol = t.symbol
        WHERE cc.base_product = '{base_product}'
          AND cc.strategy = 'outright'
          AND cc.is_continuous = FALSE
          AND cc.expiry_month IN ({quarterly_set})
          AND t.interval = '1D'
          AND t.calc_method = 'api'
        GROUP BY cc.symbol, cc.expiry_year, cc.expiry_month
    """
    df = con.cursor().execute(sql).fetchdf()
    df["first_bar"] = pd.to_datetime(df["first_bar"]).dt.date
    df["ltd_observed"] = pd.to_datetime(df["ltd_observed"]).dt.date
    df["ltd_canonical"] = [
        _ltd_canonical_sr3(int(y), int(m))
        for y, m in zip(df["expiry_year"], df["expiry_month"])
    ]
    df = df.sort_values("ltd_canonical").reset_index(drop=True)
    return df


def _business_days_between(d_from: date, d_to: date) -> int:
    """Inclusive-of-end business-day count. Negative if d_to < d_from.

    Uses np.busday_count which excludes the end date by default; we add 1
    when d_to >= d_from for an inclusive count, and -1 mirror for the other
    direction. This matters because DTE on the LTD itself is 0 (last day of
    trading), not -1.
    """
    if d_to == d_from:
        return 0
    if d_to > d_from:
        return int(np.busday_count(d_from, d_to))
    return -int(np.busday_count(d_to, d_from))


def chain_at_date(chain_df: pd.DataFrame, t: date) -> list[tuple[str, int]]:
    """Return the list of ``(symbol, dte_business_days)`` for every contract
    that's "live" on date ``t`` — first_bar ≤ t ≤ ltd_canonical — sorted by
    DTE ascending (front first).

    DTE is computed against ``ltd_canonical`` (the true LTD, not the data-
    truncated ``ltd_observed``). A DTE of 0 means today is the last trading
    day for that contract.
    """
    if chain_df.empty:
        return []
    mask = (chain_df["first_bar"] <= t) & (chain_df["ltd_canonical"] >= t)
    live = chain_df.loc[mask, ["symbol", "ltd_canonical"]]
    rows = [
        (sym, _business_days_between(t, ltd))
        for sym, ltd in zip(live["symbol"], live["ltd_canonical"])
    ]
    rows.sort(key=lambda x: x[1])
    return rows


# =============================================================================
# Phase A.2 — per-contract panel loader + roll calendar + back-adjustment
# =============================================================================

def _load_per_contract_panels(symbols: list[str],
                                base_product: str = "SRA") -> dict:
    """Bulk-load OHLCV daily bars for a set of contract symbols.

    Returns ``{symbol: DataFrame}`` where each DataFrame is indexed by
    ``bar_date`` (a python ``date``) and has columns ``open, high, low,
    close, volume``. Empty DataFrame if the symbol has no bars.

    One round-trip to DuckDB regardless of the number of symbols — keeps
    the CMC builder fast even with 50+ contracts.
    """
    if not symbols:
        return {}
    con = get_ohlc_connection()
    if con is None:
        raise RuntimeError("OHLC database unavailable")
    syms_csv = ",".join(f"'{s}'" for s in symbols)
    sql = f"""
        SELECT
            symbol,
            DATE(to_timestamp(time/1000.0)) AS bar_date,
            open, high, low, close, volume
        FROM mde2_timeseries
        WHERE symbol IN ({syms_csv})
          AND interval = '1D'
          AND calc_method = 'api'
        ORDER BY symbol, bar_date
    """
    raw = con.cursor().execute(sql).fetchdf()
    if raw.empty:
        return {sym: pd.DataFrame() for sym in symbols}
    raw["bar_date"] = pd.to_datetime(raw["bar_date"]).dt.date
    panels = {}
    for sym in symbols:
        sub = raw.loc[raw["symbol"] == sym, ["bar_date", "open", "high",
                                                "low", "close", "volume"]]
        if sub.empty:
            panels[sym] = pd.DataFrame()
        else:
            panels[sym] = sub.set_index("bar_date").sort_index()
    return panels


def _bd_offset(d: date, days: int) -> date:
    """Subtract ``days`` business days from ``d`` (or add if days<0).

    Uses numpy's busday_offset which respects weekends. Holidays are NOT
    modelled — the surrounding sessions absorb at most ±1 BD slip in roll
    timing, which is well inside the volume-guard's 3-session check.
    """
    if days == 0:
        return d
    forward = days < 0
    offset = -days if forward else days
    if forward:
        return np.busday_offset(d, offset, roll="forward").astype("M8[D]").astype(date)
    return np.busday_offset(d, -offset, roll="backward").astype("M8[D]").astype(date)


def build_roll_calendar(chain_df: pd.DataFrame,
                          panels: Optional[dict] = None,
                          asof_date: Optional[date] = None) -> pd.DataFrame:
    """Compute the roll boundaries between consecutive front-month contracts.

    Default rule: roll out of contract C on ``ltd_canonical(C) - ROLL_OFFSET_BD``
    business days (5 BD per industry standard).

    **Only HISTORICAL rolls are returned.** A roll is included only if its
    target calendar date is ≤ ``asof_date``. Future rolls (between two
    currently-trading contracts whose LTDs are still ahead) produce no entry
    — those rolls haven't happened, so back-adjustment must not be applied.

    Volume guard: if ``next_contract.volume > old_contract.volume`` has not
    held for ``VOL_GUARD_SESSIONS`` consecutive sessions ending on the
    calendar roll date, defer the roll one session at a time until the guard
    is satisfied (cap at ``ltd_canonical(C)`` itself — never past LTD).

    The gap is computed as ``new_close − old_close`` on the actual roll
    date. Both contracts must have a bar on that date; if not, fall back to
    the nearest preceding common bar.

    Parameters
    ----------
    chain_df : DataFrame
        Output of :func:`_load_quarterly_chain`.
    panels : dict, optional
        Pre-loaded per-contract panels keyed by symbol. If None, loads them.
    asof_date : date, optional
        Cutoff for historical rolls. Defaults to the max bar date seen across
        any loaded panel (≈ OHLC snapshot date).

    Returns
    -------
    DataFrame with columns
    ``roll_date, old_sym, new_sym, gap, gap_method, ltd_canonical_old``
    — sorted by ``roll_date`` ascending. Future rolls are omitted entirely.
    """
    chain = chain_df.sort_values("ltd_canonical").reset_index(drop=True)
    if panels is None:
        panels = _load_per_contract_panels(chain["symbol"].tolist())

    if asof_date is None:
        all_max = [p.index.max() for p in panels.values() if not p.empty]
        if not all_max:
            return pd.DataFrame(columns=[
                "roll_date", "old_sym", "new_sym", "gap",
                "gap_method", "ltd_canonical_old",
            ])
        asof_date = max(all_max)

    rows = []
    for i in range(len(chain) - 1):
        old_sym = str(chain.iloc[i]["symbol"])
        new_sym = str(chain.iloc[i + 1]["symbol"])
        ltd_old = chain.iloc[i]["ltd_canonical"]
        target_roll = _bd_offset(ltd_old, ROLL_OFFSET_BD)

        # **Historical-only filter**: a future roll (target > asof_date) means
        # neither contract has rolled out yet — skip silently. Note we don't
        # break: rolls beyond this in the chain are also skipped because LTD
        # is monotonic, so the loop tail is guaranteed-future.
        if target_roll > asof_date:
            break

        old_panel = panels.get(old_sym, pd.DataFrame())
        new_panel = panels.get(new_sym, pd.DataFrame())
        if old_panel.empty or new_panel.empty:
            rows.append({
                "roll_date": None, "old_sym": old_sym, "new_sym": new_sym,
                "gap": None, "gap_method": "skipped: missing panel",
                "ltd_canonical_old": ltd_old,
            })
            continue

        # Bound the roll window
        earliest = max(old_panel.index.min(), new_panel.index.min())
        latest = min(ltd_old, old_panel.index.max(), new_panel.index.max(),
                     asof_date)
        if target_roll < earliest:
            target_roll = earliest
        if target_roll > latest:
            target_roll = latest

        # Volume guard — slide forward (toward LTD) until volume condition
        # holds, capped at the latest bound.
        roll_date = target_roll
        method = "calendar"
        for _ in range(ROLL_OFFSET_BD + 5):
            if (roll_date not in old_panel.index) or (roll_date not in new_panel.index):
                common = old_panel.index.intersection(new_panel.index)
                common = common[common <= roll_date]
                if len(common) == 0:
                    method = "skipped: no common bar"
                    roll_date = None
                    break
                roll_date = common[-1]
            old_recent = old_panel.loc[:roll_date, "volume"].tail(VOL_GUARD_SESSIONS)
            new_recent = new_panel.loc[:roll_date, "volume"].tail(VOL_GUARD_SESSIONS)
            if (len(old_recent) >= VOL_GUARD_SESSIONS
                    and len(new_recent) >= VOL_GUARD_SESSIONS
                    and (new_recent.values > old_recent.values).all()):
                break
            next_roll = _bd_offset(roll_date, -1)  # +1 BD forward
            if next_roll > latest:
                method = "calendar (volume guard not satisfied; capped at LTD)"
                break
            roll_date = next_roll
            method = "volume_deferred"

        if roll_date is None:
            rows.append({
                "roll_date": None, "old_sym": old_sym, "new_sym": new_sym,
                "gap": None, "gap_method": method,
                "ltd_canonical_old": ltd_old,
            })
            continue

        # Compute gap on the chosen roll date
        if roll_date not in old_panel.index or roll_date not in new_panel.index:
            common = old_panel.index.intersection(new_panel.index)
            common = common[common <= roll_date]
            if len(common) == 0:
                rows.append({
                    "roll_date": None, "old_sym": old_sym, "new_sym": new_sym,
                    "gap": None, "gap_method": method + " + no common bar",
                    "ltd_canonical_old": ltd_old,
                })
                continue
            roll_date = common[-1]
            method += " (fallback to nearest common bar)"
        gap = float(new_panel.loc[roll_date, "close"] - old_panel.loc[roll_date, "close"])

        rows.append({
            "roll_date": roll_date, "old_sym": old_sym, "new_sym": new_sym,
            "gap": gap, "gap_method": method,
            "ltd_canonical_old": ltd_old,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("roll_date", na_position="last").reset_index(drop=True)
    return out


def back_adjust_panels(chain_df: pd.DataFrame,
                          panels: dict,
                          roll_calendar: pd.DataFrame) -> dict:
    """Apply Backward Panama (additive) adjustment to each per-contract panel.

    Convention: the **anchor** is the most-recent contract that hasn't yet
    rolled out — i.e. the one with the smallest LTD ≥ all roll-dates in the
    calendar. Its adj_close == raw_close.

    Each older contract's offset = sum of gaps of (a) its own roll-out plus
    (b) every subsequent roll. So at any roll boundary
    ``adj_close[old_sym, roll_date] == adj_close[new_sym, roll_date]``.

    Contracts NEWER than the anchor (deferred quarterlies that haven't been
    front yet) get offset 0 — they have no rolls yet and no Panama
    adjustment applies.

    Returns ``{symbol: DataFrame}`` where each DataFrame has the original
    columns PLUS ``adj_open, adj_high, adj_low, adj_close``. Volume is not
    adjusted (Panama doesn't apply to volume).
    """
    chain = chain_df.sort_values("ltd_canonical").reset_index(drop=True)
    syms_in_order = chain["symbol"].tolist()

    valid = roll_calendar.dropna(subset=["roll_date", "gap"])
    gap_by_old_sym = dict(zip(valid["old_sym"], valid["gap"]))

    # Walk newest→oldest. For each contract:
    #   1. Add this contract's own roll-out gap to cumulative_offset (if it
    #      ever rolled out — i.e. is in gap_by_old_sym).
    #   2. Apply cumulative_offset to this contract's bars.
    # The anchor (current front + everything newer) has no roll-out gap
    # entry, so cumulative_offset stays 0 and adj == raw.
    cumulative_offset = 0.0
    out: dict = {}
    for sym in reversed(syms_in_order):
        gap = gap_by_old_sym.get(sym)
        if gap is not None:
            cumulative_offset += float(gap)
        panel = panels.get(sym)
        if panel is None or panel.empty:
            out[sym] = panel.copy() if panel is not None else pd.DataFrame()
            continue
        adjusted = panel.copy()
        for col in ("open", "high", "low", "close"):
            if col in adjusted.columns:
                adjusted[f"adj_{col}"] = adjusted[col] + cumulative_offset
        out[sym] = adjusted

    # Return in chain order (oldest first → newest last) for caller convenience
    return {sym: out[sym] for sym in syms_in_order if sym in out}


def _build_roll_calendar(chain_df: pd.DataFrame) -> pd.DataFrame:
    """Backwards-compat wrapper for the public name (used by Phase A.5
    orchestrator). Loads panels itself."""
    return build_roll_calendar(chain_df)


# =============================================================================
# Phase A.3 — linear-interpolation constant-maturity sampler
# =============================================================================

def _target_dte_business_days(months: int) -> int:
    """Convert a tenor in months to a business-day target.

    Uses 21 BD/month (252 BD/year ÷ 12). Returns 0 for M0 (front).
    """
    return int(months) * _BD_PER_MONTH


def _active_chain_at_date(chain_df: pd.DataFrame,
                            roll_calendar: pd.DataFrame,
                            t: date) -> list[tuple[str, int]]:
    """Like ``chain_at_date`` but additionally **drops contracts that have
    already rolled out** per the roll calendar.

    A contract C has rolled out at date ``t`` if there's a row in the roll
    calendar with ``old_sym = C`` and ``roll_date < t`` (note: strict <;
    the contract is still "the front" on its own roll date itself).

    Returns ``[(symbol, dte_business_days), ...]`` sorted by DTE ascending,
    excluding rolled-out contracts.
    """
    base = chain_at_date(chain_df, t)
    if not base:
        return []
    if roll_calendar is None or roll_calendar.empty:
        return base
    rolled_out = set(
        roll_calendar.loc[
            (roll_calendar["roll_date"].notna())
            & (roll_calendar["roll_date"] < t),
            "old_sym"
        ].astype(str).tolist()
    )
    return [(sym, dte) for sym, dte in base if sym not in rolled_out]


def _bracketing_contracts(chain_df: pd.DataFrame,
                            asof_t: date,
                            target_dte_bd: int,
                            roll_calendar: Optional[pd.DataFrame] = None) -> Optional[tuple]:
    """For a given calendar date and target DTE in business days, find the
    two contracts whose DTE bracket the target (c1 just ≤, c2 just >).

    Returns ``(sym_c1, dte_c1, sym_c2, dte_c2)`` or None if no bracketing
    pair exists (e.g. chain too short for very deep nodes early in
    history).

    If ``roll_calendar`` is supplied, contracts that have already rolled
    out are dropped — the "front" tracks the active-front per the
    chronological roll sequence, NOT the raw LTD. This prevents the old
    contract from briefly remaining the M0 anchor in the 5 BD between its
    roll-out and its actual LTD.
    """
    if roll_calendar is not None:
        live = _active_chain_at_date(chain_df, roll_calendar, asof_t)
    else:
        live = chain_at_date(chain_df, asof_t)
        live = [(sym, dte) for sym, dte in live if dte >= 0]
    if not live:
        return None

    # Target = 0 (M0 front) — return (front, second). If front has been
    # rolled out per the calendar, _active_chain_at_date already pruned it.
    if target_dte_bd <= live[0][1]:
        if len(live) < 2:
            return None
        c1_sym, c1_dte = live[0]
        c2_sym, c2_dte = live[1]
        return (c1_sym, c1_dte, c2_sym, c2_dte)

    # Find the largest c1 with dte ≤ target, and the smallest c2 with dte > target
    c1 = None
    for sym, dte in live:
        if dte <= target_dte_bd:
            c1 = (sym, dte)
        else:
            return (c1[0], c1[1], sym, dte) if c1 is not None else None
    return None


def _interpolate_cmc_outright_node(panels_adj: dict,
                                     chain_df: pd.DataFrame,
                                     target_months: int,
                                     date_index: pd.DatetimeIndex,
                                     roll_calendar: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Build the CMC outright series for a single target tenor (in months)
    over the supplied ``date_index``.

    For each bar date `t` in the index:
      - Find bracketing live contracts (c1, c2) whose DTE brackets target T.
      - Compute weight ``w = (T - dte_c1) / (dte_c2 - dte_c1)`` (0..1).
      - CMC OHLC at t = ``(1-w) * adj_OHLC[c1, t] + w * adj_OHLC[c2, t]``.
      - Volume = sum of bracketing legs (combined liquidity).

    Per Carver's denominator correction: the **raw_close anchor** is the
    raw close of c1 (the active listed contract whose DTE is closest to T).
    Used by downstream %-return / RSI / BB-width computations as the
    denominator anchor.

    Returns a DataFrame indexed by date with columns:
        c1_sym, c2_sym, weight, dte_c1, dte_c2,
        open, high, low, close, volume_combined,
        raw_close_anchor, has_data
    """
    target_dte = _target_dte_business_days(target_months)
    out_rows = []
    for t in date_index:
        t_date = t.date() if hasattr(t, "date") else t
        bracket = _bracketing_contracts(chain_df, t_date, target_dte,
                                          roll_calendar=roll_calendar)
        if bracket is None:
            out_rows.append({
                "bar_date": t_date,
                "c1_sym": None, "c2_sym": None, "weight": None,
                "dte_c1": None, "dte_c2": None,
                "open": np.nan, "high": np.nan, "low": np.nan, "close": np.nan,
                "volume_combined": np.nan,
                "raw_close_anchor": np.nan,
                "has_data": False,
            })
            continue
        c1_sym, dte_c1, c2_sym, dte_c2 = bracket
        c1_panel = panels_adj.get(c1_sym, pd.DataFrame())
        c2_panel = panels_adj.get(c2_sym, pd.DataFrame())

        # Both contracts must have a bar on `t`
        if t_date not in c1_panel.index or t_date not in c2_panel.index:
            out_rows.append({
                "bar_date": t_date,
                "c1_sym": c1_sym, "c2_sym": c2_sym, "weight": None,
                "dte_c1": dte_c1, "dte_c2": dte_c2,
                "open": np.nan, "high": np.nan, "low": np.nan, "close": np.nan,
                "volume_combined": np.nan,
                "raw_close_anchor": np.nan,
                "has_data": False,
            })
            continue

        denom = dte_c2 - dte_c1
        if denom <= 0:
            # Degenerate (target equals or undershoots c1's DTE) — pure c1
            w = 0.0
        else:
            w = max(0.0, min(1.0, (target_dte - dte_c1) / denom))

        c1_row = c1_panel.loc[t_date]
        c2_row = c2_panel.loc[t_date]
        cmc_open  = (1 - w) * c1_row["adj_open"]  + w * c2_row["adj_open"]
        cmc_high  = (1 - w) * c1_row["adj_high"]  + w * c2_row["adj_high"]
        cmc_low   = (1 - w) * c1_row["adj_low"]   + w * c2_row["adj_low"]
        cmc_close = (1 - w) * c1_row["adj_close"] + w * c2_row["adj_close"]
        # Volume: linearly weight (combined) — additive doesn't make sense
        # because the two legs can be highly correlated. Use weighted avg.
        v1 = c1_row.get("volume", 0.0); v2 = c2_row.get("volume", 0.0)
        cmc_volume = (1 - w) * float(v1 or 0) + w * float(v2 or 0)

        out_rows.append({
            "bar_date": t_date,
            "c1_sym": c1_sym, "c2_sym": c2_sym, "weight": float(w),
            "dte_c1": int(dte_c1), "dte_c2": int(dte_c2),
            "open": float(cmc_open), "high": float(cmc_high),
            "low": float(cmc_low), "close": float(cmc_close),
            "volume_combined": float(cmc_volume),
            # Carver's denominator: the closer-to-target contract's RAW
            # close (not adjusted). Used by %-return indicators.
            "raw_close_anchor": float(c1_row["close"]),
            "has_data": True,
        })

    df = pd.DataFrame(out_rows)
    df["bar_date"] = pd.to_datetime(df["bar_date"]).dt.date
    return df


# =============================================================================
# Phase A.4 — spread / fly CMC derived from outright CMC nodes
# =============================================================================

def _build_outright_cmc_table(panels_adj: dict,
                                 chain_df: pd.DataFrame,
                                 roll_calendar: pd.DataFrame,
                                 date_index: pd.DatetimeIndex,
                                 method: str = "linear") -> dict:
    """Build the full set of outright CMC nodes (one DataFrame per node) over
    ``date_index``. Used both as the public outright CMC artifact AND as the
    input to spread/fly derivation.

    Parameters
    ----------
    method : str, default ``"linear"``
        Interpolation method. ``"linear"`` (legacy, per-node bracketing-pair
        sample) or ``"pchip"`` (Phase 2: PCHIP through the full chain at each
        bar — local, C¹, monotone-preserving, shape-preserving).

    Returns ``{node_id: DataFrame}`` keyed by canonical id (e.g. ``"M3"``).
    """
    if method == "pchip":
        return _build_outright_cmc_table_pchip(
            panels_adj, chain_df, roll_calendar, date_index)
    out = {}
    for months in OUTRIGHT_NODES_MONTHS:
        node_id = _outright_node_id(months)
        out[node_id] = _interpolate_cmc_outright_node(
            panels_adj, chain_df, months, date_index, roll_calendar=roll_calendar)
    return out


def _build_outright_cmc_table_pchip(panels_adj: dict,
                                          chain_df: pd.DataFrame,
                                          roll_calendar: pd.DataFrame,
                                          date_index: pd.DatetimeIndex) -> dict:
    """PCHIP-based CMC outright builder (Phase 2, plan §2.1 A12).

    Per Holton + S&P VIX Futures methodology, but using PCHIP
    (Piecewise Cubic Hermite Interpolating Polynomial) through ALL active
    contracts on the chain at each bar, instead of pairwise linear between
    bracketing contracts. Benefits:
      * **C¹ continuity** across the full curve (linear has corners at each
        listed contract)
      * **Local** — a single bad print contaminates only the immediate
        vicinity, not distant tenors
      * **Shape-preserving** — won't introduce spurious oscillations in
        humped or kinked curves (which polynomial fit would)
      * **Monotonicity preserving** — won't invert the curve where it's
        monotone

    Implementation: for each bar `t`, fit one PCHIP `f_t(dte)` through all
    `(dte_c, adj_close_c)` for live contracts c. Sample at every target
    node's `dte = node_months × 21 BD`. Per Carver, the raw_close_anchor
    is the closest-DTE listed contract's RAW close.
    """
    try:
        from scipy.interpolate import PchipInterpolator
    except ImportError:
        # Fallback to linear if scipy.interpolate unavailable
        return _build_outright_cmc_table(panels_adj, chain_df,
                                              roll_calendar, date_index,
                                              method="linear")

    # Pre-build an index of contract → adjusted-panel for fast lookup
    sym_to_panel = {sym: p for sym, p in panels_adj.items() if not p.empty}

    # Pre-compute per-(date, sym) DTE so we can build the per-bar curve fast
    target_months = list(OUTRIGHT_NODES_MONTHS)
    target_dtes = [_target_dte_business_days(m) for m in target_months]
    node_ids = [_outright_node_id(m) for m in target_months]

    # Allocate output frames — one per node
    out_rows: dict = {nid: [] for nid in node_ids}

    for t in date_index:
        t_date = t.date() if hasattr(t, "date") else t
        # Active chain at this date (respects roll calendar)
        live = _active_chain_at_date(chain_df, roll_calendar, t_date)
        if not live or len(live) < 2:
            for nid in node_ids:
                out_rows[nid].append({
                    "bar_date": t_date, "c1_sym": None, "c2_sym": None,
                    "weight": None, "dte_c1": None, "dte_c2": None,
                    "open": np.nan, "high": np.nan, "low": np.nan, "close": np.nan,
                    "volume_combined": np.nan, "raw_close_anchor": np.nan,
                    "has_data": False,
                })
            continue

        # Build the (dte, OHLCV, raw_close) grid for the chain at t
        dtes = []; opens = []; highs = []; lows = []; closes = []; vols = []; raws = []
        active_syms = []
        for sym, dte in live:
            panel = sym_to_panel.get(sym)
            if panel is None or panel.empty or t_date not in panel.index:
                continue
            row = panel.loc[t_date]
            dtes.append(int(dte))
            opens.append(float(row.get("adj_open", row["open"])))
            highs.append(float(row.get("adj_high", row["high"])))
            lows.append(float(row.get("adj_low", row["low"])))
            closes.append(float(row.get("adj_close", row["close"])))
            vols.append(float(row.get("volume", 0)))
            raws.append(float(row["close"]))
            active_syms.append(sym)
        if len(dtes) < 2:
            for nid in node_ids:
                out_rows[nid].append({
                    "bar_date": t_date, "c1_sym": None, "c2_sym": None,
                    "weight": None, "dte_c1": None, "dte_c2": None,
                    "open": np.nan, "high": np.nan, "low": np.nan, "close": np.nan,
                    "volume_combined": np.nan, "raw_close_anchor": np.nan,
                    "has_data": False,
                })
            continue

        # Sort by DTE for PchipInterpolator
        order = np.argsort(dtes)
        dtes_arr = np.asarray(dtes)[order]
        # PCHIP needs strictly increasing x; coalesce duplicates by averaging
        unique_dtes, inv = np.unique(dtes_arr, return_inverse=True)
        # If duplicates exist, average the y values
        def _coalesce(y_arr):
            y_arr = np.asarray(y_arr)[order]
            if len(unique_dtes) == len(dtes_arr):
                return y_arr
            collapsed = np.zeros(len(unique_dtes))
            counts = np.zeros(len(unique_dtes))
            for i, idx in enumerate(inv):
                collapsed[idx] += y_arr[i]
                counts[idx] += 1
            return collapsed / counts

        try:
            f_open = PchipInterpolator(unique_dtes, _coalesce(opens), extrapolate=False)
            f_high = PchipInterpolator(unique_dtes, _coalesce(highs), extrapolate=False)
            f_low = PchipInterpolator(unique_dtes, _coalesce(lows), extrapolate=False)
            f_close = PchipInterpolator(unique_dtes, _coalesce(closes), extrapolate=False)
            f_vol = PchipInterpolator(unique_dtes, _coalesce(vols), extrapolate=False)
        except (ValueError, IndexError):
            # If PCHIP fit fails (singular x), fall back to linear bracket
            for m, nid in zip(target_months, node_ids):
                target = _target_dte_business_days(m)
                bracket = _bracketing_contracts(chain_df, t_date, target,
                                                  roll_calendar=roll_calendar)
                if bracket is None:
                    out_rows[nid].append({
                        "bar_date": t_date, "c1_sym": None, "c2_sym": None,
                        "weight": None, "dte_c1": None, "dte_c2": None,
                        "open": np.nan, "high": np.nan, "low": np.nan, "close": np.nan,
                        "volume_combined": np.nan, "raw_close_anchor": np.nan,
                        "has_data": False,
                    })
                else:
                    # Use linear interp as fallback for this node
                    res = _interpolate_cmc_outright_node(panels_adj, chain_df,
                                                              m, pd.DatetimeIndex([t]),
                                                              roll_calendar=roll_calendar)
                    if not res.empty:
                        out_rows[nid].append(res.iloc[0].to_dict())
                    else:
                        out_rows[nid].append({
                            "bar_date": t_date, "c1_sym": None, "c2_sym": None,
                            "weight": None, "dte_c1": None, "dte_c2": None,
                            "open": np.nan, "high": np.nan, "low": np.nan, "close": np.nan,
                            "volume_combined": np.nan, "raw_close_anchor": np.nan,
                            "has_data": False,
                        })
            continue

        # Sample PCHIP at each target node's DTE
        for m, target_dte, nid in zip(target_months, target_dtes, node_ids):
            # Out-of-range targets (chain too short) → NaN
            if target_dte < unique_dtes[0] or target_dte > unique_dtes[-1]:
                out_rows[nid].append({
                    "bar_date": t_date, "c1_sym": None, "c2_sym": None,
                    "weight": None, "dte_c1": None, "dte_c2": None,
                    "open": np.nan, "high": np.nan, "low": np.nan, "close": np.nan,
                    "volume_combined": np.nan, "raw_close_anchor": np.nan,
                    "has_data": False,
                })
                continue
            cmc_open = float(f_open(target_dte))
            cmc_high = float(f_high(target_dte))
            cmc_low = float(f_low(target_dte))
            cmc_close = float(f_close(target_dte))
            cmc_vol = float(f_vol(target_dte))
            # Raw_close_anchor: closest contract's raw close (Carver)
            closest_idx = int(np.argmin(np.abs(unique_dtes - target_dte)))
            # Map closest unique_dte back to a sym (use first match in active_syms)
            closest_sym = None
            for sym, d in live:
                if int(d) == int(unique_dtes[closest_idx]):
                    closest_sym = sym; break
            raw_close = raws[order[closest_idx]] if closest_idx < len(raws) else np.nan
            out_rows[nid].append({
                "bar_date": t_date,
                "c1_sym": closest_sym,
                "c2_sym": None,    # PCHIP doesn't have a discrete c2
                "weight": None,
                "dte_c1": int(unique_dtes[closest_idx]),
                "dte_c2": None,
                "open": cmc_open, "high": cmc_high,
                "low": cmc_low, "close": cmc_close,
                "volume_combined": cmc_vol,
                "raw_close_anchor": float(raw_close),
                "has_data": True,
            })

    # Convert lists to DataFrames
    out = {}
    for nid in node_ids:
        df = pd.DataFrame(out_rows[nid])
        if not df.empty:
            df["bar_date"] = pd.to_datetime(df["bar_date"]).dt.date
        out[nid] = df
    return out


def _build_spread_cmc_table(outright_cmc: dict) -> dict:
    """Derive each canonical CMC spread node from already-built CMC outright
    nodes.

    For pair ``(front_m, back_m)`` (e.g. M3, M6):
        spread_OHLC = M{front}_OHLC − M{back}_OHLC
        volume = avg of the two legs (inheriting the linear-blend volume)

    Returns ``{node_id: DataFrame}`` keyed by canonical id (e.g. ``"M3_M6"``).
    Each DataFrame has columns mirroring the outright tables:
        bar_date, c1_left, c1_right, weight_left, weight_right,
        open, high, low, close, volume_combined,
        raw_close_anchor (= front leg's raw_close_anchor),
        has_data
    """
    out = {}
    for front_m, back_m in SPREAD_NODES_PAIRS:
        node_id = _spread_node_id(front_m, back_m)
        left = outright_cmc.get(_outright_node_id(front_m))
        right = outright_cmc.get(_outright_node_id(back_m))
        if left is None or right is None:
            out[node_id] = pd.DataFrame()
            continue
        # Align on bar_date
        merged = left.merge(right, on="bar_date", suffixes=("_L", "_R"), how="inner")
        # Both legs need data
        valid = merged["has_data_L"] & merged["has_data_R"]
        rows = pd.DataFrame({
            "bar_date":          merged["bar_date"],
            "c1_left":           merged["c1_sym_L"],
            "c2_left":           merged["c2_sym_L"],
            "weight_left":       merged["weight_L"],
            "c1_right":          merged["c1_sym_R"],
            "c2_right":          merged["c2_sym_R"],
            "weight_right":      merged["weight_R"],
            "open":              (merged["open_L"]   - merged["open_R"]).where(valid, np.nan),
            "high":              (merged["high_L"]   - merged["low_R"]).where(valid, np.nan),
            "low":               (merged["low_L"]    - merged["high_R"]).where(valid, np.nan),
            "close":             (merged["close_L"]  - merged["close_R"]).where(valid, np.nan),
            "volume_combined":   ((merged["volume_combined_L"] +
                                     merged["volume_combined_R"]) / 2.0).where(valid, np.nan),
            "raw_close_anchor":  merged["raw_close_anchor_L"],
            "has_data":          valid,
        })
        out[node_id] = rows
    return out


def _build_fly_cmc_table(outright_cmc: dict) -> dict:
    """Derive each canonical CMC fly node from CMC outright nodes.

    For triple ``(left_m, mid_m, right_m)`` (e.g. M3, M6, M9):
        fly_close = M{left}_close − 2·M{mid}_close + M{right}_close
        fly_open  = same combo on opens
        fly_high  = M{left}_high − 2·M{mid}_low + M{right}_high   (over-estimate)
        fly_low   = M{left}_low  − 2·M{mid}_high + M{right}_low
        volume = avg of three legs

    The high/low formulas use the worst-case bound (when fly is most positive
    / most negative within the bar). True intra-bar high/low needs tick data
    we don't have; this is the conservative bound used by practitioner CMC
    butterfly references.

    Returns ``{node_id: DataFrame}``.
    """
    out = {}
    for left_m, mid_m, right_m in FLY_NODES_TRIPLES:
        node_id = _fly_node_id(left_m, mid_m, right_m)
        L = outright_cmc.get(_outright_node_id(left_m))
        M = outright_cmc.get(_outright_node_id(mid_m))
        R = outright_cmc.get(_outright_node_id(right_m))
        if L is None or M is None or R is None:
            out[node_id] = pd.DataFrame()
            continue
        merged = (L.merge(M, on="bar_date", suffixes=("_L", "_M"))
                    .merge(R, on="bar_date").rename(
                        columns={c: c + "_R" for c in R.columns if c != "bar_date"}))
        valid = (merged["has_data_L"] & merged["has_data_M"] & merged["has_data_R"])
        rows = pd.DataFrame({
            "bar_date":          merged["bar_date"],
            "c1_left":           merged["c1_sym_L"],
            "c1_mid":            merged["c1_sym_M"],
            "c1_right":          merged["c1_sym_R"],
            "weight_left":       merged["weight_L"],
            "weight_mid":        merged["weight_M"],
            "weight_right":      merged["weight_R"],
            "open":  (merged["open_L"]  - 2*merged["open_M"]  + merged["open_R"]).where(valid, np.nan),
            "high":  (merged["high_L"]  - 2*merged["low_M"]   + merged["high_R"]).where(valid, np.nan),
            "low":   (merged["low_L"]   - 2*merged["high_M"]  + merged["low_R"]).where(valid, np.nan),
            "close": (merged["close_L"] - 2*merged["close_M"] + merged["close_R"]).where(valid, np.nan),
            "volume_combined": ((merged["volume_combined_L"]
                                  + merged["volume_combined_M"]
                                  + merged["volume_combined_R"]) / 3.0).where(valid, np.nan),
            "raw_close_anchor": merged["raw_close_anchor_M"],   # body leg as anchor
            "has_data": valid,
        })
        out[node_id] = rows
    return out


# =============================================================================
# Public orchestration (placeholder; built in Phase A.5 once 2/3/4 land)
# =============================================================================

# =============================================================================
# Phase A.5 — orchestrator + parquet cache writer + manifest
# =============================================================================

def _cache_paths(asof_date: date) -> dict:
    """Return the canonical paths for cached CMC artifacts on a given asof.

    Files:
        sra_outright_<YYYY-MM-DD>.parquet
        sra_spread_<YYYY-MM-DD>.parquet
        sra_fly_<YYYY-MM-DD>.parquet
        manifest_<YYYY-MM-DD>.json   (roll calendar, build version, gap stats)
    """
    stamp = asof_date.isoformat()
    return {
        "outright": _CACHE_DIR / f"sra_outright_{stamp}.parquet",
        "spread":   _CACHE_DIR / f"sra_spread_{stamp}.parquet",
        "fly":      _CACHE_DIR / f"sra_fly_{stamp}.parquet",
        "manifest": _CACHE_DIR / f"manifest_{stamp}.json",
    }


def _stack_outright_nodes(node_dict: dict) -> pd.DataFrame:
    """Stack ``{node_id: DataFrame}`` (10 outright nodes) into a single long-
    format DataFrame indexed by ``(cmc_node, bar_date)`` for parquet storage.
    """
    parts = []
    for nid, df in node_dict.items():
        if df is None or df.empty:
            continue
        out = df.copy()
        out.insert(0, "cmc_node", nid)
        parts.append(out)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def build_cmc_nodes(scope: str, asof_date: date,
                    history_years: int = 5,
                    force_rebuild: bool = False,
                    interpolation: Optional[str] = None) -> Path:
    """Build (or load from cache) the CMC parquet for ``scope``.

    On first call for a given ``asof_date``, runs the full pipeline once
    and persists ALL THREE scopes' parquets plus a manifest — even if only
    one scope was requested. Subsequent calls for any scope at the same
    asof skip the rebuild.

    Args:
        scope: ``'outright'`` | ``'spread'`` | ``'fly'``.
        asof_date: latest bar date to include (typically OHLC snapshot date).
        history_years: history depth (default 5).
        force_rebuild: if True, ignore any existing cache and rebuild.

    Returns the parquet path for ``scope``.
    """
    if scope not in ("outright", "spread", "fly"):
        raise ValueError(f"unknown scope: {scope!r}")
    paths = _cache_paths(asof_date)

    target = paths[scope]
    manifest = paths["manifest"]
    if (not force_rebuild) and target.exists() and manifest.exists():
        return target

    # Run the full pipeline
    chain = _load_quarterly_chain("SRA")
    panels = _load_per_contract_panels(chain["symbol"].tolist())
    rolls = build_roll_calendar(chain, panels=panels, asof_date=asof_date)
    adjusted = back_adjust_panels(chain, panels, rolls)

    start = asof_date - timedelta(days=int(history_years * 365.25))
    date_idx = pd.bdate_range(start, asof_date)

    # Phase 2: read interpolation method from product_spec (default linear for back-compat)
    if interpolation is None:
        try:
            from lib.product_spec import get_product
            interpolation = get_product("SRA").get("interpolation", "linear")
        except Exception:
            interpolation = "linear"
    outright_cmc = _build_outright_cmc_table(adjusted, chain, rolls, date_idx,
                                                  method=interpolation)
    spread_cmc = _build_spread_cmc_table(outright_cmc)
    fly_cmc = _build_fly_cmc_table(outright_cmc)

    _stack_outright_nodes(outright_cmc).to_parquet(paths["outright"], index=False)
    _stack_outright_nodes(spread_cmc).to_parquet(paths["spread"], index=False)
    _stack_outright_nodes(fly_cmc).to_parquet(paths["fly"], index=False)

    # Manifest: roll calendar, gap stats, build metadata
    valid_rolls = rolls.dropna(subset=["roll_date", "gap"])
    manifest_data = {
        "builder_version": BUILDER_VERSION,
        "asof_date": asof_date.isoformat(),
        "history_start": start.isoformat(),
        "history_years": history_years,
        "n_contracts_in_chain": int(len(chain)),
        "n_rolls_historical": int(len(valid_rolls)),
        "outright_node_ids": list_cmc_nodes("outright"),
        "spread_node_ids": list_cmc_nodes("spread"),
        "fly_node_ids": list_cmc_nodes("fly"),
        "outright_node_coverage": {
            nid: {
                "rows": int(len(df)),
                "with_data": int(df["has_data"].sum()) if "has_data" in df.columns else 0,
            }
            for nid, df in outright_cmc.items()
        },
        "spread_node_coverage": {
            nid: {
                "rows": int(len(df)),
                "with_data": int(df["has_data"].sum()) if "has_data" in df.columns else 0,
            }
            for nid, df in spread_cmc.items()
        },
        "fly_node_coverage": {
            nid: {
                "rows": int(len(df)),
                "with_data": int(df["has_data"].sum()) if "has_data" in df.columns else 0,
            }
            for nid, df in fly_cmc.items()
        },
        "rolls": [
            {
                "roll_date": str(row["roll_date"]),
                "old_sym": row["old_sym"],
                "new_sym": row["new_sym"],
                "gap": float(row["gap"]),
                "gap_method": row["gap_method"],
                "ltd_canonical_old": str(row["ltd_canonical_old"]),
            }
            for _, row in valid_rolls.iterrows()
        ],
        "gap_stats_bp": {
            "n": int(len(valid_rolls)),
            "mean_bp":   float(valid_rolls["gap"].mean()  * 100) if len(valid_rolls) else None,
            "std_bp":    float(valid_rolls["gap"].std()   * 100) if len(valid_rolls) else None,
            "min_bp":    float(valid_rolls["gap"].min()   * 100) if len(valid_rolls) else None,
            "max_bp":    float(valid_rolls["gap"].max()   * 100) if len(valid_rolls) else None,
            "abs_median_bp": float(valid_rolls["gap"].abs().median() * 100) if len(valid_rolls) else None,
        },
        "missing_contracts_in_chain": [],   # populated below
    }
    # Forensic flag: known SR3 quarterlies that DON'T appear in our chain
    # (SRAH26 is a known data gap as of the 2026-04-27 snapshot)
    expected_quarterlies = set()
    for yr in range(2018, asof_date.year + 5):
        for mo in (3, 6, 9, 12):
            expected_quarterlies.add(f"SRA{['F','G','H','J','K','M','N','Q','U','V','X','Z'][mo-1]}{str(yr)[-2:]}")
    actual = set(chain["symbol"].astype(str).tolist())
    # Only flag missing contracts whose canonical LTD is in the past or near-future
    for sym in expected_quarterlies - actual:
        # Parse year from symbol (last 2 digits)
        try:
            yr2 = int(sym[-2:])
            yr = 2000 + yr2 if yr2 < 50 else 1900 + yr2
        except ValueError:
            continue
        if yr <= asof_date.year + 1:
            manifest_data["missing_contracts_in_chain"].append(sym)

    paths["manifest"].write_text(json.dumps(manifest_data, indent=2, default=str))
    return target


def load_cmc_panel(scope: str, asof_date: date) -> pd.DataFrame:
    """Load the cached CMC parquet for ``(scope, asof_date)``.

    Returns the long-format DataFrame: one row per ``(cmc_node, bar_date)``.
    Triggers ``build_cmc_nodes`` automatically if the cache is missing.
    """
    paths = _cache_paths(asof_date)
    if not paths[scope].exists():
        build_cmc_nodes(scope, asof_date)
    return pd.read_parquet(paths[scope])


def get_cmc_roll_log(asof_date: date) -> pd.DataFrame:
    """Return the historical roll calendar from the manifest as a DataFrame.

    Convenience accessor for the verification harness and the Settings tab.
    """
    paths = _cache_paths(asof_date)
    if not paths["manifest"].exists():
        build_cmc_nodes("outright", asof_date)   # triggers manifest write
    manifest = json.loads(paths["manifest"].read_text())
    return pd.DataFrame(manifest.get("rolls", []))
