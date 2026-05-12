"""Aggregate trade metrics (Phase C / D).

Per-cell metric pack matches the TMIA gameplan §7.3 spec — 40+ fields:

  Counts            : trade_count, winning_trades, losing_trades, win_rate
  Dollar            : total_pnl_usd, gross_profit_usd, gross_loss_usd,
                       profit_factor, avg_pnl_per_trade, avg_win_pnl,
                       avg_loss_pnl, realized_rrr
  R-based           : avg_R_realized, avg_R_mae, avg_R_mfe, max_R_mae,
                       mfe_mae_ratio, expectancy_R
  Risk-adjusted     : sharpe (annualised × √252 from per-trade R), sortino,
                       max_drawdown_usd, avg_drawdown_usd, avg_updraw_usd,
                       consecutive_losses, consecutive_wins,
                       max_single_loss_usd, max_position_lots,
                       account_size_needed (= max_dd × 3)
  Time              : avg_holding_bars
  Bootstrap CIs (when trade_count ≥ 30):
                       bootstrap_sharpe_lo / hi,
                       bootstrap_winrate_lo / hi,
                       bootstrap_expectancy_lo / hi

Calendar slices (separate table): by-month / by-day-of-week / by-year.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lib.backtest.bootstrap import (
    bootstrap_sharpe_ci, bootstrap_winrate_ci, bootstrap_expectancy_ci,
)


def _empty_metrics() -> dict:
    """All-NaN metrics dict for empty trade lists."""
    return {
        "trade_count": 0,
        "winning_trades": 0, "losing_trades": 0, "win_rate": None,
        "total_pnl_usd": 0.0, "gross_profit_usd": 0.0, "gross_loss_usd": 0.0,
        "profit_factor": None,
        "avg_pnl_per_trade": None, "avg_win_pnl": None, "avg_loss_pnl": None,
        "realized_rrr": None,
        "avg_R_realized": None, "avg_R_mae": None, "avg_R_mfe": None,
        "max_R_mae": None, "mfe_mae_ratio": None, "expectancy_R": None,
        "sharpe": None, "sortino": None,
        "max_drawdown_usd": None, "avg_drawdown_usd": None,
        "avg_updraw_usd": None,
        "consecutive_losses": 0, "consecutive_wins": 0,
        "max_single_loss_usd": None,
        "max_position_lots": None, "account_size_needed": None,
        "avg_holding_bars": None,
        "bootstrap_sharpe_lo": None, "bootstrap_sharpe_hi": None,
        "bootstrap_winrate_lo": None, "bootstrap_winrate_hi": None,
        "bootstrap_expectancy_lo": None, "bootstrap_expectancy_hi": None,
    }


def _runs(bool_series: pd.Series) -> tuple:
    """Return (max_consecutive_True, max_consecutive_False)."""
    if bool_series.empty:
        return (0, 0)
    arr = bool_series.values.astype(bool)
    max_t = max_f = 0
    cur_t = cur_f = 0
    for v in arr:
        if v:
            cur_t += 1; cur_f = 0
            if cur_t > max_t: max_t = cur_t
        else:
            cur_f += 1; cur_t = 0
            if cur_f > max_f: max_f = cur_f
    return (int(max_t), int(max_f))


def _equity_curve(pnl_series: pd.Series) -> pd.Series:
    return pnl_series.cumsum()


def _drawdown_stats(equity: pd.Series) -> tuple:
    """Return (max_dd_usd, avg_dd_usd, avg_updraw_usd) on the equity series."""
    if equity.empty:
        return (None, None, None)
    running_max = equity.cummax()
    drawdowns = equity - running_max     # ≤ 0
    max_dd = float(abs(drawdowns.min())) if not drawdowns.empty else 0.0
    # average drawdown depth (excluding zero-DD bars)
    neg_dd = drawdowns[drawdowns < 0]
    avg_dd = float(abs(neg_dd.mean())) if len(neg_dd) > 0 else 0.0
    # average updraw — distance below running min, mirror logic
    running_min = equity.cummin()
    updraws = equity - running_min       # ≥ 0
    pos_ud = updraws[updraws > 0]
    avg_ud = float(pos_ud.mean()) if len(pos_ud) > 0 else 0.0
    return (max_dd, avg_dd, avg_ud)


def compute_metrics(trades: pd.DataFrame,
                       n_bootstrap: int = 1000) -> dict:
    """Compute the full per-cell metric pack from a trade DataFrame.

    Returns a flat dict of all gameplan §7.3 metrics.
    """
    if trades is None or trades.empty:
        return _empty_metrics()
    out = {}
    n = len(trades)
    out["trade_count"] = int(n)

    realized_R = trades["realized_R"].astype(float).values
    pnl = trades["realized_pnl_usd"].astype(float).values
    is_win = realized_R > 0
    is_loss = realized_R < 0
    out["winning_trades"] = int(is_win.sum())
    out["losing_trades"] = int(is_loss.sum())
    out["win_rate"] = float(is_win.mean())

    # Dollar
    out["total_pnl_usd"] = float(np.sum(pnl))
    out["gross_profit_usd"] = float(np.sum(pnl[pnl > 0]))
    out["gross_loss_usd"] = float(abs(np.sum(pnl[pnl < 0])))
    out["profit_factor"] = (out["gross_profit_usd"] / out["gross_loss_usd"]
                            if out["gross_loss_usd"] > 0 else None)
    out["avg_pnl_per_trade"] = float(np.mean(pnl))
    out["avg_win_pnl"] = float(np.mean(pnl[pnl > 0])) if (pnl > 0).any() else None
    out["avg_loss_pnl"] = float(np.mean(pnl[pnl < 0])) if (pnl < 0).any() else None
    out["realized_rrr"] = ((out["avg_win_pnl"] / abs(out["avg_loss_pnl"]))
                            if (out["avg_win_pnl"] is not None
                                and out["avg_loss_pnl"] is not None
                                and out["avg_loss_pnl"] != 0) else None)

    # R-based
    out["avg_R_realized"] = float(np.mean(realized_R))
    mae_R = trades["mae_R"].astype(float).values
    mfe_R = trades["mfe_R"].astype(float).values
    out["avg_R_mae"] = float(np.mean(mae_R))
    out["avg_R_mfe"] = float(np.mean(mfe_R))
    out["max_R_mae"] = float(np.min(mae_R))
    out["mfe_mae_ratio"] = (float(np.mean(mfe_R) / abs(np.mean(mae_R)))
                            if np.mean(mae_R) != 0 else None)
    out["expectancy_R"] = out["avg_R_realized"]

    # Risk-adjusted
    if np.std(realized_R) > 0:
        out["sharpe"] = float(np.mean(realized_R) / np.std(realized_R) * np.sqrt(252))
        downside = realized_R[realized_R < 0]
        out["sortino"] = (float(np.mean(realized_R) / np.std(downside) * np.sqrt(252))
                          if len(downside) > 0 and np.std(downside) > 0 else None)
    else:
        out["sharpe"] = None
        out["sortino"] = None

    # Drawdowns on equity curve (cumulative pnl)
    equity = pd.Series(pnl).cumsum()
    max_dd, avg_dd, avg_ud = _drawdown_stats(equity)
    out["max_drawdown_usd"] = max_dd
    out["avg_drawdown_usd"] = avg_dd
    out["avg_updraw_usd"] = avg_ud

    # Run lengths
    win_runs, loss_runs = _runs(pd.Series(is_win))
    out["consecutive_wins"] = win_runs
    out["consecutive_losses"] = loss_runs

    out["max_single_loss_usd"] = float(np.min(pnl)) if len(pnl) > 0 else None
    out["max_position_lots"] = int(trades["lots"].max()) if "lots" in trades.columns else None
    out["account_size_needed"] = (float(max_dd * 3) if max_dd is not None else None)

    out["avg_holding_bars"] = float(trades["holding_bars"].mean())

    # Bootstrap CIs (only if n>=30)
    if n >= 30:
        s_lo, s_hi = bootstrap_sharpe_ci(realized_R, n_bootstrap=n_bootstrap)
        w_lo, w_hi = bootstrap_winrate_ci(realized_R, n_bootstrap=n_bootstrap)
        e_lo, e_hi = bootstrap_expectancy_ci(realized_R, n_bootstrap=n_bootstrap)
        out["bootstrap_sharpe_lo"] = s_lo
        out["bootstrap_sharpe_hi"] = s_hi
        out["bootstrap_winrate_lo"] = w_lo
        out["bootstrap_winrate_hi"] = w_hi
        out["bootstrap_expectancy_lo"] = e_lo
        out["bootstrap_expectancy_hi"] = e_hi
    else:
        for k in ("bootstrap_sharpe_lo", "bootstrap_sharpe_hi",
                   "bootstrap_winrate_lo", "bootstrap_winrate_hi",
                   "bootstrap_expectancy_lo", "bootstrap_expectancy_hi"):
            out[k] = None

    return out


def compute_calendar_slices(trades: pd.DataFrame) -> dict:
    """Return calendar slices: by month / by day-of-week / by year.

    Returns ``{"by_month": DataFrame, "by_dow": DataFrame, "by_year": DataFrame}``.
    Each slice has: trades, wins, win_pct, pnl_usd, drawdown_usd, pnl_per_trade.
    """
    if trades is None or trades.empty:
        empty = pd.DataFrame(columns=["bucket", "trades", "wins", "win_pct",
                                          "pnl_usd", "pnl_per_trade"])
        return {"by_month": empty, "by_dow": empty, "by_year": empty}
    df = trades.copy()
    df["entry_dt"] = pd.to_datetime(df["entry_dt"])
    df["month"] = df["entry_dt"].dt.month
    df["dow"] = df["entry_dt"].dt.day_name()
    df["year"] = df["entry_dt"].dt.year
    df["is_win"] = df["realized_R"] > 0

    def _agg(grouped):
        return pd.DataFrame({
            "trades": grouped.size(),
            "wins": grouped["is_win"].sum(),
            "win_pct": grouped["is_win"].mean() * 100,
            "pnl_usd": grouped["realized_pnl_usd"].sum(),
            "pnl_per_trade": grouped["realized_pnl_usd"].mean(),
        }).reset_index().rename(columns={grouped.keys[0]: "bucket"})

    return {
        "by_month": _agg(df.groupby("month")),
        "by_dow":   _agg(df.groupby("dow")),
        "by_year":  _agg(df.groupby("year")),
    }
