"""Bar-level backtest engine + metrics + aggregator (Phases C/D/E).

Public API:
    from lib.backtest import simulate_trades, compute_metrics, build_metrics_grid
    from lib.backtest.bootstrap import bootstrap_ci
    from lib.backtest.cycle import ensure_backtest_fresh
"""
from lib.backtest.engine import (
    simulate_trades, ExitPolicy, TradeRecord,
)
from lib.backtest.metrics import compute_metrics, compute_calendar_slices
from lib.backtest.bootstrap import bootstrap_ci

__all__ = [
    "simulate_trades", "ExitPolicy", "TradeRecord",
    "compute_metrics", "compute_calendar_slices",
    "bootstrap_ci",
]
