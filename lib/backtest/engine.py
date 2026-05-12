"""Bar-level backtest engine (Phase C).

Executes a fire list against a daily OHLCV panel using the gameplan §6.1
universal rules:

  - Signal at close of bar T → fill at OPEN of bar T+1.
  - Stop / T1 / T2 evaluated intraday using the bar's high/low on T+1
    onward.
  - **Same-bar conflict resolution: stop wins** (Yash 2026-05-07; matches
    the gameplan's conservative-fill convention).
  - Cooldown: ``COOLDOWN_BARS`` (default 3) bars after exit before a new
    entry on the same (setup × node × direction) can be taken.
  - Position concurrency: 1 per (setup × node × direction). A second fire
    while a position is open is ignored (no pyramiding).
  - Slippage = 0, commission = 0 (locked per gameplan).
  - Time stop: ``TIME_STOP_BARS`` (default 60) bars.
  - Reversal exit (optional): a fire in the OPPOSITE direction closes the
    position at the next open.

Trade record per fire (gameplan §7.3):
    entry_dt, exit_dt, direction, cmc_node, setup_id,
    entry_price, stop_price, t1_price, t2_price,
    exit_price, exit_reason ('stop' | 't1' | 't2' | 'time' | 'reversal'),
    holding_bars,
    mae_R, mfe_R, realized_R,
    realized_pnl_usd, lots,
    regime_at_entry, adx_at_entry, atr_at_entry, hurst_at_entry
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd


# =============================================================================
# Defaults — locked per gameplan §6.1 + Yash 2026-05-07
# =============================================================================

DEFAULT_TIME_STOP_BARS = 60
DEFAULT_COOLDOWN_BARS = 3
DEFAULT_RISK_DOLLARS = 10_000.0
SR3_DOLLARS_PER_BP_PER_LOT = 25.0   # CME SR3 DV01


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class ExitPolicy:
    """Exit policy for a single fire. All prices in instrument units (price,
    not bp), even for spread/fly nodes (the panel's close convention).

    direction in {'LONG', 'SHORT'}.

    Either supply absolute prices (entry / stop / t1 / t2), or set
    ``r_multiple`` non-None and the engine will derive T1/T2 from
    ``entry`` and ``stop`` (T1 = +1R, T2 = +2R).
    """
    entry_price: float
    stop_price: float
    t1_price: Optional[float] = None
    t2_price: Optional[float] = None
    direction: str = "LONG"
    time_stop_bars: int = DEFAULT_TIME_STOP_BARS
    r_multiple: Optional[tuple] = None  # (t1_R, t2_R)
    allow_reversal_exit: bool = True


@dataclass
class TradeRecord:
    """Single executed trade — one row per fire that opened a position."""
    setup_id: str
    cmc_node: str
    direction: str
    entry_dt: date
    exit_dt: Optional[date]
    entry_price: float
    stop_price: float
    t1_price: Optional[float]
    t2_price: Optional[float]
    exit_price: Optional[float]
    exit_reason: str            # 'stop' | 't1' | 't2' | 'time' | 'reversal' | 'eod'
    holding_bars: int
    mae_R: float                # max adverse excursion in R units (negative if drawn against)
    mfe_R: float                # max favorable excursion in R units
    realized_R: float
    realized_pnl_usd: float
    lots: int
    bp_per_unit: float          # 100 for outright, 1 for spread/fly
    regime_at_entry: str = ""
    adx_at_entry: Optional[float] = None
    atr_at_entry: Optional[float] = None
    hurst_at_entry: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# Helpers
# =============================================================================

def _pos(idx: pd.Index, t: date) -> int:
    """Return the integer position of ``t`` in ``idx`` (a DatetimeIndex
    convention). -1 if not found."""
    try:
        ts = pd.Timestamp(t)
        if ts not in idx:
            # Find the next-or-equal bar
            after = idx[idx >= ts]
            if len(after) == 0:
                return -1
            ts = after[0]
        return int(idx.get_indexer([ts])[0])
    except Exception:
        return -1


def _lots_at_risk(stop_distance_units: float, bp_per_unit: float,
                    risk_dollars: float = DEFAULT_RISK_DOLLARS,
                    dollars_per_bp_per_lot: float = SR3_DOLLARS_PER_BP_PER_LOT,
                    max_lots: int = 1000) -> int:
    """Integer lot count to risk ≈ ``risk_dollars`` per trade.

    ``stop_distance_units`` is in price units (e.g. 0.05 for 5 bp on an
    outright with bp_per_unit=100). Multiply by bp_per_unit to get bps.
    """
    sd_bp = abs(stop_distance_units) * bp_per_unit
    if sd_bp <= 0:
        return 1
    lots = risk_dollars / (sd_bp * dollars_per_bp_per_lot)
    if not np.isfinite(lots):
        return 1
    return int(max(1, min(max_lots, lots)))


def _classify_regime_at_entry(adx: Optional[float],
                                  atr_pctile: Optional[float]) -> str:
    """Per gameplan §7.5:
        adx_state = TRENDING (≥25) / RANGING (<20) / NEUTRAL
        vol_state = HIGH_VOL (>0.75 pctile) / LOW_VOL (<0.25) / NEUTRAL
    """
    if adx is None or not np.isfinite(adx):
        adx_state = "NEUTRAL"
    elif adx >= 25:
        adx_state = "TRENDING"
    elif adx < 20:
        adx_state = "RANGING"
    else:
        adx_state = "NEUTRAL"
    if atr_pctile is None or not np.isfinite(atr_pctile):
        vol_state = "NEUTRAL"
    elif atr_pctile > 0.75:
        vol_state = "HIGH_VOL"
    elif atr_pctile < 0.25:
        vol_state = "LOW_VOL"
    else:
        vol_state = "NEUTRAL"
    if adx_state == "NEUTRAL" or vol_state == "NEUTRAL":
        return "NEUTRAL"
    return f"{adx_state}_{vol_state}"


# =============================================================================
# Core simulator
# =============================================================================

def simulate_one_trade(panel: pd.DataFrame,
                          policy: ExitPolicy,
                          entry_idx: int,
                          setup_id: str,
                          cmc_node: str,
                          bp_per_unit: float,
                          risk_dollars: float = DEFAULT_RISK_DOLLARS,
                          dollars_per_bp_per_lot: float = SR3_DOLLARS_PER_BP_PER_LOT,
                          opposing_fires: Optional[set] = None) -> Optional[TradeRecord]:
    """Simulate a single trade from a fire at bar position ``entry_idx``.

    The fire is OBSERVED at close of T = ``entry_idx``. Entry fills at
    OPEN of T+1 = ``entry_idx + 1``. Returns None if T+1 doesn't exist
    (fire was on the last bar).
    """
    if entry_idx + 1 >= len(panel):
        return None

    direction = policy.direction
    entry_pos = entry_idx + 1
    entry_dt = panel.index[entry_pos].date()
    entry_price = float(panel.iloc[entry_pos]["open"])

    # Resolve T1/T2 from R-multiples if not absolute
    t1 = policy.t1_price
    t2 = policy.t2_price
    if (t1 is None or t2 is None) and policy.r_multiple is not None:
        r1, r2 = policy.r_multiple
        risk = (entry_price - policy.stop_price) if direction == "LONG" \
               else (policy.stop_price - entry_price)
        if risk > 0:
            if t1 is None:
                t1 = entry_price + r1 * risk * (1 if direction == "LONG" else -1)
            if t2 is None:
                t2 = entry_price + r2 * risk * (1 if direction == "LONG" else -1)

    stop = policy.stop_price
    risk_pp = abs(entry_price - stop)
    if risk_pp <= 0:
        return None

    lots = _lots_at_risk(risk_pp, bp_per_unit, risk_dollars, dollars_per_bp_per_lot)
    bp_per_lot = SR3_DOLLARS_PER_BP_PER_LOT  # convention for SR3
    risk_dollars_per_unit = bp_per_unit * bp_per_lot * lots

    # Walk forward bar-by-bar from entry_pos to entry_pos + time_stop
    max_bars = min(policy.time_stop_bars, len(panel) - entry_pos - 1)
    mae = 0.0  # in price units, signed against direction (negative)
    mfe = 0.0  # in price units, signed in favor (positive)
    exit_pos = None
    exit_price = None
    exit_reason = "time"

    for k in range(1, max_bars + 1):
        pos = entry_pos + k
        bar_high = float(panel.iloc[pos]["high"])
        bar_low = float(panel.iloc[pos]["low"])
        bar_close = float(panel.iloc[pos]["close"])
        bar_dt = panel.index[pos].date()

        if direction == "LONG":
            adverse = bar_low - entry_price       # negative on drawdown
            favor = bar_high - entry_price        # positive on profit
            stop_hit = bar_low <= stop
            t1_hit = (t1 is not None) and (bar_high >= t1)
            t2_hit = (t2 is not None) and (bar_high >= t2)
        else:
            adverse = entry_price - bar_high      # negative on drawdown
            favor = entry_price - bar_low         # positive on profit
            stop_hit = bar_high >= stop
            t1_hit = (t1 is not None) and (bar_low <= t1)
            t2_hit = (t2 is not None) and (bar_low <= t2)

        mae = min(mae, adverse)
        mfe = max(mfe, favor)

        # Same-bar conflict resolution: STOP WINS if both hit
        if stop_hit:
            exit_pos = pos; exit_price = stop; exit_reason = "stop"
            break
        if t2_hit:
            exit_pos = pos; exit_price = t2; exit_reason = "t2"
            break
        if t1_hit:
            exit_pos = pos; exit_price = t1; exit_reason = "t1"
            break

        # Reversal exit: any opposing fire on this bar closes the trade at next open
        if policy.allow_reversal_exit and opposing_fires is not None:
            if bar_dt in opposing_fires and pos + 1 < len(panel):
                exit_pos = pos + 1
                exit_price = float(panel.iloc[pos + 1]["open"])
                exit_reason = "reversal"
                break

    if exit_pos is None:
        # Time stop
        exit_pos = entry_pos + max_bars
        exit_pos = min(exit_pos, len(panel) - 1)
        exit_price = float(panel.iloc[exit_pos]["close"])
        exit_reason = "time"

    exit_dt = panel.index[exit_pos].date()
    holding_bars = exit_pos - entry_pos

    # Compute realized R
    if direction == "LONG":
        pnl_pp = exit_price - entry_price
    else:
        pnl_pp = entry_price - exit_price
    realized_R = pnl_pp / risk_pp
    mae_R = mae / risk_pp
    mfe_R = mfe / risk_pp
    realized_pnl_usd = pnl_pp * bp_per_unit * bp_per_lot * lots

    # Regime tagging
    adx = panel.iloc[entry_pos].get("ADX_14") if "ADX_14" in panel.columns else None
    atr = panel.iloc[entry_pos].get("ATR14") if "ATR14" in panel.columns else None
    hurst = panel.iloc[entry_pos].get("Hurst_60") if "Hurst_60" in panel.columns else None
    atr_pctile = None
    if atr is not None and "ATR14" in panel.columns:
        atr_window = panel["ATR14"].iloc[max(0, entry_pos - 252):entry_pos].dropna()
        if len(atr_window) > 30:
            atr_pctile = (atr_window < atr).sum() / len(atr_window)
    regime = _classify_regime_at_entry(
        float(adx) if adx is not None and pd.notna(adx) else None,
        float(atr_pctile) if atr_pctile is not None else None,
    )

    return TradeRecord(
        setup_id=setup_id, cmc_node=cmc_node, direction=direction,
        entry_dt=entry_dt, exit_dt=exit_dt,
        entry_price=entry_price, stop_price=stop,
        t1_price=t1, t2_price=t2,
        exit_price=exit_price, exit_reason=exit_reason,
        holding_bars=holding_bars,
        mae_R=float(mae_R), mfe_R=float(mfe_R),
        realized_R=float(realized_R),
        realized_pnl_usd=float(realized_pnl_usd),
        lots=lots, bp_per_unit=bp_per_unit,
        regime_at_entry=regime,
        adx_at_entry=float(adx) if adx is not None and pd.notna(adx) else None,
        atr_at_entry=float(atr) if atr is not None and pd.notna(atr) else None,
        hurst_at_entry=float(hurst) if hurst is not None and pd.notna(hurst) else None,
    )


def simulate_trades(panel: pd.DataFrame,
                       fires: pd.DataFrame,
                       setup_id: str,
                       cmc_node: str,
                       bp_per_unit: float = 100.0,
                       cooldown_bars: int = DEFAULT_COOLDOWN_BARS,
                       risk_dollars: float = DEFAULT_RISK_DOLLARS) -> pd.DataFrame:
    """Run the full backtest for one (setup × node) combo.

    Parameters
    ----------
    panel : pd.DataFrame
        Daily OHLCV panel indexed by datetime. Must have ``open``, ``high``,
        ``low``, ``close`` columns. Optional indicator columns
        (``ADX_14``, ``ATR14``, ``Hurst_60``) are used for regime tagging.
    fires : pd.DataFrame
        One row per fire signal. Required columns:
            entry_dt (date), direction ('LONG'/'SHORT'), stop_price,
            t1_price (optional), t2_price (optional)
        Sort by entry_dt ascending.
    setup_id, cmc_node : str
        Tagged onto each trade record.
    bp_per_unit : float
        100 for outright (price), 1 for spread/fly (bp).
    cooldown_bars : int
        Bars of mandatory pause after each exit before a new entry on the
        same (setup × node × direction) is allowed. 0 to disable.
    risk_dollars : float
        Per-trade $ risk (default $10,000).

    Returns
    -------
    pd.DataFrame of TradeRecord rows.
    """
    if fires is None or fires.empty:
        return pd.DataFrame()
    panel = panel.sort_index()
    fires = fires.sort_values("entry_dt").reset_index(drop=True)

    # Pre-build the per-direction fire-date set for reversal-exit lookup
    long_fire_dates = set(fires.loc[fires["direction"] == "LONG", "entry_dt"])
    short_fire_dates = set(fires.loc[fires["direction"] == "SHORT", "entry_dt"])

    # Track last-exit position per direction (for cooldown enforcement)
    last_exit_pos: dict[str, int] = {"LONG": -1_000, "SHORT": -1_000}
    # Track open trade per direction
    open_until_pos: dict[str, int] = {"LONG": -1, "SHORT": -1}

    trades = []
    for _, fire in fires.iterrows():
        ed = fire["entry_dt"]
        d = fire["direction"]
        idx_pos = _pos(panel.index, ed)
        if idx_pos < 0 or idx_pos + 1 >= len(panel):
            continue
        # Concurrency / cooldown gate
        if open_until_pos[d] >= idx_pos:
            continue   # already in a trade in same direction
        if idx_pos - last_exit_pos[d] <= cooldown_bars:
            continue   # cooldown active
        entry_price = float(panel.iloc[idx_pos + 1]["open"])
        # Re-derive stop if missing (use entry +/- 2*ATR fallback)
        stop = fire.get("stop_price")
        if stop is None or pd.isna(stop):
            atr = panel.iloc[idx_pos].get("ATR14") if "ATR14" in panel.columns else None
            if atr is None or pd.isna(atr):
                continue
            stop = entry_price - 2 * atr if d == "LONG" else entry_price + 2 * atr
        policy = ExitPolicy(
            entry_price=entry_price,
            stop_price=float(stop),
            t1_price=float(fire["t1_price"]) if "t1_price" in fire and pd.notna(fire["t1_price"]) else None,
            t2_price=float(fire["t2_price"]) if "t2_price" in fire and pd.notna(fire["t2_price"]) else None,
            direction=d,
            r_multiple=(1.0, 2.0) if (("t1_price" not in fire) or pd.isna(fire["t1_price"])) else None,
        )
        opposing = short_fire_dates if d == "LONG" else long_fire_dates
        rec = simulate_one_trade(
            panel, policy, idx_pos, setup_id, cmc_node, bp_per_unit,
            risk_dollars=risk_dollars,
            opposing_fires=opposing,
        )
        if rec is None:
            continue
        trades.append(rec.to_dict())
        # Update gates
        exit_pos = _pos(panel.index, rec.exit_dt)
        last_exit_pos[d] = exit_pos
        open_until_pos[d] = -1   # closed

    return pd.DataFrame(trades)
