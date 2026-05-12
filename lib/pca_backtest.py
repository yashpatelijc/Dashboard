"""Backtest framework for the PCA trade-idea engine.

Replays `generate_all_trade_ideas` through history, simulating entry with
slippage and dynamic rule-based target/stop exits (`lib.trade_exits`). For
each emitted idea above the conviction threshold, simulates the trade
day-by-day and records P&L, MFE, MAE, holding period, exit tier.

Aggregates results per source, per regime, per conviction bucket, per mode.
Provides per-source empirical hit rates that wire back into `score_conviction`
via `panel["empirical_hit_rates"]` (Phase 5 wiring).

Two execution modes:
  - "fast": single full-history PCA fit, evaluate triggers each day.
            Cheap but technically forward-looking on the PCA fit.
  - "walk_forward": re-fit PCA at each decision date (or every ~30d).
            Slow but unbiased — the only correct way to compute true Sharpe.

Slippage model: 0.25 bp per leg per side for listed contracts (CME spec),
$1.50 commission per contract per side, 1.5×-tick spread proxy for flies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from lib.pca import build_full_pca_panel, mode_params, DEFAULT_MODE
from lib.pca_trades import generate_all_trade_ideas, SRA_TICK_VALUE_USD
from lib.trade_exits import (EntryState, evaluate_dynamic_exit,
                                  entry_state_from_idea)


LISTED_BP_TICK = 0.25
USD_PER_LISTED_BP = 25.0
COMMISSION_USD_PER_LEG_SIDE = 1.50


@dataclass(frozen=True)
class BacktestTrade:
    """One simulated trade — from entry to exit."""
    entry_date: date
    exit_date: date
    days_held: int
    source: str
    primary_source_id: int
    structure_type: str
    direction: str
    leg_fingerprint: str
    n_legs: int

    entry_z: float
    entry_residual_bp: float
    entry_hl_d: float
    entry_conviction: float

    exit_residual_bp: float
    exit_tier: str                    # "T1" | "T2" | ... | "S6" | "OPEN" | "HORIZON_END"
    exit_reason: str

    pnl_bp_gross: float
    pnl_bp_net: float                 # after slippage + commissions
    pnl_dollar: float

    mfe_bp: float                     # max favourable excursion
    mae_bp: float                     # max adverse excursion

    regime_at_entry: Optional[str]
    mode: str


@dataclass
class BacktestResult:
    """Aggregated backtest output."""
    trades: list = field(default_factory=list)
    trades_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    by_source: pd.DataFrame = field(default_factory=pd.DataFrame)
    by_conviction_bucket: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)


def _per_trade_costs_bp(structure_type: str, n_legs: int,
                          local_per_bp: float = USD_PER_LISTED_BP) -> float:
    """Round-trip transaction cost in bp per unit (entry + exit).

    Listed contracts: 0.25 bp tick × 1.5 (slippage proxy) × 2 sides × n_legs
    Plus commissions translated to bp per contract using the market's local
    currency-per-bp (defaults to USD25 for SRA).
    """
    slippage_bp = LISTED_BP_TICK * 1.5 * 2 * max(1, n_legs)
    commissions_local = COMMISSION_USD_PER_LEG_SIDE * 2 * max(1, n_legs)
    commissions_bp = commissions_local / max(local_per_bp, 1.0)
    return float(slippage_bp + commissions_bp)


def _favourable_pnl_bp(entry_residual: float, current_residual: float,
                         direction: str) -> float:
    """How much P&L (in bp) we'd realize if we closed now.

    direction "long" means we bought because residual was negative (cheap);
    closing PnL is (current - entry). If residual climbs (less negative),
    we make money.
    "short" inverts the sign.
    """
    diff = float(current_residual) - float(entry_residual)
    sign = +1.0 if direction == "long" else (-1.0 if direction == "short" else 0.0)
    return sign * diff


def _instrument_history_series(panel: dict, instrument_sym: str) -> Optional[pd.Series]:
    """Return the historical level-residual series for an instrument.

    Looks in residual_outrights / spreads / flies / packs and reconstructs
    a series from the panel's level-residual computation if available.
    Falls back to None if the series isn't easily reconstructible (in
    which case the caller uses a simpler price-only check).
    """
    # For backtest we approximate using current residual_*_bp + a synthetic walk.
    # In production walk-forward, this should use the per-symbol cumsum-detrend
    # series computed in pca.per_outright_residuals.
    # For "fast" mode we use the OHLC close panel directly.
    for kind, panel_key in (("outright", "outright_close_panel"),
                              ("spread", "spread_close_panel"),
                              ("fly", "fly_close_panel")):
        df = panel.get(panel_key)
        if df is None or df.empty:
            continue
        if instrument_sym in df.columns:
            return df[instrument_sym].dropna()
    return None


def _simulate_trade(idea, entry_date: date, panel_today: dict,
                      mode: str, max_days: int) -> Optional[BacktestTrade]:
    """Simulate a single trade from entry through dynamic exit.

    Pulls the instrument's history series from panel and walks day-by-day
    until an exit tier fires or max_days reached.
    """
    legs = getattr(idea, "legs", ()) or ()
    if not legs:
        return None
    primary_sym = legs[0].symbol

    series = _instrument_history_series(panel_today, primary_sym)
    if series is None or series.empty:
        return None

    # Find the future trading days post-entry
    try:
        post = series.loc[series.index.date > entry_date]
    except Exception:
        return None
    if post.empty:
        return None

    entry_state = entry_state_from_idea(idea, entry_date)
    entry_price = float(series.loc[series.index.date <= entry_date].iloc[-1])
    entry_residual = entry_state.entry_residual_bp
    direction = entry_state.direction

    mfe_bp = 0.0
    mae_bp = 0.0
    exit_date = entry_date
    exit_residual = entry_residual
    exit_tier_label = "HORIZON_END"
    exit_reason = "Max-horizon reached without exit trigger."
    days_held = 0

    # Walk forward day-by-day
    for i, (ts, current_price) in enumerate(post.items(), start=1):
        days_held = i
        # Approximate "current residual" as the price-change scaled to bp.
        # Real walk-forward should rebuild the level-residual via build_full_pca_panel
        # at each step; the fast path uses price change as a proxy.
        current_residual = entry_residual + (float(current_price) - entry_price) * 100.0
        pnl_now = _favourable_pnl_bp(entry_residual, current_residual, direction)
        if pnl_now > mfe_bp:
            mfe_bp = pnl_now
        if pnl_now < mae_bp:
            mae_bp = pnl_now

        # Synthetic current_panel = panel_today (fast-mode shortcut)
        # Real walk-forward would re-run build_full_pca_panel(ts.date()).
        synthetic_panel = dict(panel_today)
        synthetic_panel["mode"] = mode

        # Override z, hl in residual tables to reflect current state (proxy)
        # For fast mode we don't recompute these; evaluate_dynamic_exit will
        # use the stale values, which means S2/S3 may not trigger correctly.
        # This is a documented limitation of fast mode.
        es = evaluate_dynamic_exit(idea, entry_state, synthetic_panel,
                                       days_held=days_held, mode=mode,
                                       current_pnl_bp=pnl_now)
        if es.exit_now:
            exit_date = ts.date() if hasattr(ts, "date") else ts
            exit_residual = current_residual
            if es.target_tier is not None:
                exit_tier_label = f"T{es.target_tier}"
            else:
                exit_tier_label = f"S{es.stop_tier}"
            exit_reason = es.reason
            break

        if days_held >= max_days:
            exit_date = ts.date() if hasattr(ts, "date") else ts
            exit_residual = current_residual
            break

    # Compute final P&L using market-aware local currency per bp
    _mkt_cfg = (panel_today or {}).get("market") or {}
    local_per_bp = float(_mkt_cfg.get("dollar_per_bp_local", USD_PER_LISTED_BP))
    pnl_gross = _favourable_pnl_bp(entry_residual, exit_residual, direction)
    costs = _per_trade_costs_bp(getattr(idea, "structure_type", "outright"),
                                   len(legs), local_per_bp=local_per_bp)
    pnl_net = pnl_gross - costs
    pnl_dollar = pnl_net * local_per_bp * max(1, len(legs))

    return BacktestTrade(
        entry_date=entry_date,
        exit_date=exit_date,
        days_held=days_held,
        source=str(getattr(idea, "primary_source", "?")),
        primary_source_id=int(getattr(idea, "source_id", 0)),
        structure_type=str(getattr(idea, "structure_type", "?")),
        direction=direction,
        leg_fingerprint=str(getattr(idea, "leg_fingerprint", "")),
        n_legs=len(legs),
        entry_z=float(entry_state.entry_z),
        entry_residual_bp=float(entry_residual),
        entry_hl_d=float(entry_state.entry_hl_d),
        entry_conviction=float(getattr(idea, "conviction", 0.0) or 0.0),
        exit_residual_bp=float(exit_residual),
        exit_tier=exit_tier_label,
        exit_reason=exit_reason,
        pnl_bp_gross=float(pnl_gross),
        pnl_bp_net=float(pnl_net),
        pnl_dollar=float(pnl_dollar),
        mfe_bp=float(mfe_bp),
        mae_bp=float(mae_bp),
        regime_at_entry=getattr(idea, "regime_label", None),
        mode=mode,
    )


def _aggregate_by_source(trades: list) -> pd.DataFrame:
    """Group trades by source; compute hit rate, avg PnL, Sharpe, etc."""
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame([{
        "source": t.source,
        "pnl_bp": t.pnl_bp_net,
        "pnl_dollar": t.pnl_dollar,
        "days_held": t.days_held,
        "is_winner": t.pnl_bp_net > 0,
        "conviction": t.entry_conviction,
    } for t in trades])
    agg = df.groupby("source").agg(
        n_trades=("pnl_bp", "count"),
        hit_rate=("is_winner", "mean"),
        avg_pnl_bp=("pnl_bp", "mean"),
        median_pnl_bp=("pnl_bp", "median"),
        total_pnl_dollar=("pnl_dollar", "sum"),
        avg_days_held=("days_held", "mean"),
        std_pnl_bp=("pnl_bp", "std"),
    )
    agg["sharpe"] = agg["avg_pnl_bp"] / agg["std_pnl_bp"].replace(0, np.nan)
    return agg.reset_index().sort_values("sharpe", ascending=False)


def _aggregate_by_conviction(trades: list) -> pd.DataFrame:
    """Bucket trades by entry conviction [0.5,0.6) etc, show return per bucket."""
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame([{
        "conviction": t.entry_conviction,
        "pnl_bp": t.pnl_bp_net,
        "is_winner": t.pnl_bp_net > 0,
    } for t in trades])
    buckets = pd.cut(df["conviction"],
                       bins=[0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01],
                       labels=["<0.40", "[0.40,0.50)", "[0.50,0.60)",
                                "[0.60,0.70)", "[0.70,0.80)", "≥0.80"])
    df["bucket"] = buckets
    agg = df.groupby("bucket", observed=True).agg(
        n_trades=("pnl_bp", "count"),
        hit_rate=("is_winner", "mean"),
        avg_pnl_bp=("pnl_bp", "mean"),
    )
    return agg.reset_index()


def _build_equity_curve(trades: list) -> pd.Series:
    """Build cumulative-PnL series indexed by exit_date."""
    if not trades:
        return pd.Series(dtype=float)
    sorted_t = sorted(trades, key=lambda t: t.exit_date)
    df = pd.DataFrame([{"exit_date": t.exit_date, "pnl_dollar": t.pnl_dollar}
                         for t in sorted_t])
    df["cum_pnl"] = df["pnl_dollar"].cumsum()
    return pd.Series(df["cum_pnl"].values, index=pd.to_datetime(df["exit_date"]))


def _max_drawdown(eq: pd.Series) -> float:
    if eq.empty:
        return 0.0
    rolling_max = eq.cummax()
    dd = eq - rolling_max
    return float(dd.min())


def run_engine_backtest(
    start_date: date,
    end_date: date,
    *,
    mode: str = DEFAULT_MODE,
    min_conviction: float = 0.50,
    walk_forward: bool = False,
    walk_step_days: int = 5,
    max_trades_per_day: int = 5,
    sources: Optional[list] = None,
    progress_callback=None,
) -> BacktestResult:
    """Run a full backtest over [start_date, end_date].

    For each trading day in the range:
      1. Build PCA panel at that date (or reuse if walk_step_days > 0).
      2. Generate trade ideas, filter by conviction and source list.
      3. For each candidate idea, simulate the trade through dynamic exits.
      4. Record results.

    Returns BacktestResult with per-source aggregates and an equity curve.
    """
    result = BacktestResult()
    result.config = {
        "start_date": str(start_date),
        "end_date": str(end_date),
        "mode": mode,
        "min_conviction": min_conviction,
        "walk_forward": walk_forward,
        "walk_step_days": walk_step_days,
        "max_trades_per_day": max_trades_per_day,
        "sources": sources or "ALL",
    }
    mp = mode_params(mode)
    max_hold = int(mp.get("hold_cap", 60))

    # Generate the sequence of decision dates
    all_days = pd.bdate_range(start_date, end_date)
    if walk_forward:
        decision_dates = list(all_days[::max(1, walk_step_days)])
    else:
        # Fast mode — single panel build at end_date, walk back over trade dates
        decision_dates = list(all_days[::max(1, walk_step_days)])

    # Cache the panel between walk-forward steps for performance
    last_panel = None
    last_panel_date = None
    trades: list = []

    total = len(decision_dates)
    for idx, day in enumerate(decision_dates):
        if progress_callback is not None:
            try:
                progress_callback(idx, total, day.date() if hasattr(day, 'date') else day)
            except Exception:
                pass

        day_d = day.date() if hasattr(day, "date") else day

        # Build or reuse panel
        if walk_forward or last_panel is None:
            try:
                panel = build_full_pca_panel(day_d, mode=mode)
            except Exception:
                continue
            last_panel = panel
            last_panel_date = day_d
        else:
            panel = last_panel

        if panel is None or panel.get("pca_fit_static") is None:
            continue

        # Generate ideas
        try:
            ideas = generate_all_trade_ideas(panel, day_d)
        except Exception:
            continue
        if not ideas:
            continue

        # Filter by conviction + sources
        cands = [i for i in ideas
                  if getattr(i, "conviction", 0.0) >= min_conviction]
        if sources:
            cands = [i for i in cands if i.primary_source in sources]
        if not cands:
            continue

        # Cap trades per day to avoid over-trading
        cands = cands[:max_trades_per_day]

        for idea in cands:
            try:
                t = _simulate_trade(idea, day_d, panel, mode, max_hold)
            except Exception:
                continue
            if t is not None:
                trades.append(t)

    # Resolve currency from the last panel's market config (built per-market
    # inside the backtest loop). Falls back to USD for SRA / unknown markets.
    _currency = "USD"
    try:
        if last_panel is not None:
            _currency = ((last_panel.get("market") or {})
                           .get("currency", "USD"))
    except Exception:
        pass

    # Build aggregates
    result.trades = trades
    if trades:
        result.trades_df = pd.DataFrame([t.__dict__ for t in trades])
        result.equity_curve = _build_equity_curve(trades)
        result.by_source = _aggregate_by_source(trades)
        result.by_conviction_bucket = _aggregate_by_conviction(trades)
        result.summary = {
            "n_trades": len(trades),
            "currency": _currency,
            "hit_rate": float(np.mean([1.0 if t.pnl_bp_net > 0 else 0.0 for t in trades])),
            "total_pnl_bp": float(sum(t.pnl_bp_net for t in trades)),
            "total_pnl_dollar": float(sum(t.pnl_dollar for t in trades)),
            "avg_days_held": float(np.mean([t.days_held for t in trades])),
            "max_drawdown_dollar": _max_drawdown(result.equity_curve),
            "sharpe_per_trade": (float(np.mean([t.pnl_bp_net for t in trades])) /
                                    max(1e-9, float(np.std([t.pnl_bp_net for t in trades],
                                                              ddof=1))))
                                if len(trades) > 1 else 0.0,
        }
    else:
        result.summary = {"n_trades": 0, "currency": _currency, "hit_rate": 0.0,
                            "total_pnl_bp": 0.0, "total_pnl_dollar": 0.0,
                            "avg_days_held": 0.0, "max_drawdown_dollar": 0.0,
                            "sharpe_per_trade": 0.0}

    return result


def build_empirical_hit_rates(result: BacktestResult) -> dict:
    """Convert backtest result into the dict format expected by score_conviction.

    Returns:
      {source_name: {
         "horizon_30d": {"hit_rate": ..., "mean_pnl_bp": ..., "n": ...,
                         "sharpe": ...},
         ...
      }}
    """
    out = {}
    if not result.trades:
        return out
    df = result.trades_df
    if df.empty:
        return out
    # Bucket by approximate hold horizon
    horizon_buckets = [(0, 7, "horizon_5d"), (7, 21, "horizon_14d"),
                        (21, 45, "horizon_30d"), (45, 90, "horizon_60d"),
                        (90, 365, "horizon_90d")]
    for source, sdf in df.groupby("source"):
        per_source = {}
        for lo, hi, key in horizon_buckets:
            slice_df = sdf[(sdf["days_held"] >= lo) & (sdf["days_held"] < hi)]
            if len(slice_df) == 0:
                continue
            hit = float((slice_df["pnl_bp_net"] > 0).mean())
            mean_pnl = float(slice_df["pnl_bp_net"].mean())
            std_pnl = float(slice_df["pnl_bp_net"].std(ddof=1)) if len(slice_df) > 1 else 0.0
            sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0.0
            per_source[key] = {
                "hit_rate": hit, "mean_pnl_bp": mean_pnl,
                "sharpe": sharpe, "n": int(len(slice_df)),
            }
        if per_source:
            out[str(source)] = per_source
    return out
