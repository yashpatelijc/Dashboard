"""10-day backtest recompute cycle (Phase E).

Mirrors the architecture of ``lib/prewarm.py``: a module-level lock + flag
guarantee the recompute daemon thread spawns at most once per process. On
app start, ``ensure_backtest_fresh()`` checks the mtime of the persistent
backtest DuckDB; if it's older than 10 days (or missing entirely), it
spawns the recompute thread.

Mechanics:
  - Recompute every 10 days (configurable).
  - Results stored at ``D:\\STIRS_DASHBOARD\\.backtest_cache\\tmia.duckdb``.
  - Each successful recompute also drops a dated snapshot at
    ``.backtest_cache/snapshots/tmia_<YYYY-MM-DD>.duckdb`` for forensics
    (keeping the 6 most recent).
  - Status banner data via ``get_backtest_status()``.
  - ``force_recompute_now()`` for the Settings-tab manual override button.

The full per-detector replay across 5y × 22 CMC nodes is implemented by
:func:`recompute_full_backtest`. Cold compute time is roughly 5-15 min per
scope on a typical workstation; subsequent app loads are instant
(metrics + trades read from disk).
"""
from __future__ import annotations

import threading
import time
import traceback
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd


# =============================================================================
# Config
# =============================================================================

RECOMPUTE_INTERVAL_DAYS = 10

_BACKTEST_CACHE_DIR = Path("D:\\STIRS_DASHBOARD\\.backtest_cache")
_BACKTEST_CACHE_DIR.mkdir(exist_ok=True)
_SNAPSHOT_DIR = _BACKTEST_CACHE_DIR / "snapshots"
_SNAPSHOT_DIR.mkdir(exist_ok=True)
_TMIA_DUCKDB = _BACKTEST_CACHE_DIR / "tmia.duckdb"
N_SNAPSHOTS_RETAINED = 6


# =============================================================================
# Module-level state (mirrors lib/prewarm.py pattern)
# =============================================================================

_LOCK = threading.Lock()
_RECOMPUTE_STARTED = False
_RECOMPUTE_THREAD: Optional[threading.Thread] = None
_STATUS: dict = {
    "started_at": None,
    "completed_at": None,
    "errors": [],
    "n_trades": None,
    "n_cells": None,
    "asof_date": None,
    "running": False,
}


# =============================================================================
# Public API
# =============================================================================

def is_backtest_fresh(now: Optional[datetime] = None,
                          interval_days: int = RECOMPUTE_INTERVAL_DAYS) -> bool:
    """True if ``tmia.duckdb`` exists AND its mtime is within the last
    ``interval_days``."""
    if not _TMIA_DUCKDB.exists():
        return False
    if now is None:
        now = datetime.now()
    age = now - datetime.fromtimestamp(_TMIA_DUCKDB.stat().st_mtime)
    return age < timedelta(days=interval_days)


def get_backtest_status() -> dict:
    """Snapshot of current recompute status (read-only)."""
    out = dict(_STATUS)
    out["fresh"] = is_backtest_fresh()
    if _TMIA_DUCKDB.exists():
        mtime = datetime.fromtimestamp(_TMIA_DUCKDB.stat().st_mtime)
        out["last_run"] = mtime.isoformat(timespec="seconds")
        out["next_run_after"] = (mtime + timedelta(days=RECOMPUTE_INTERVAL_DAYS)).isoformat(timespec="seconds")
        out["file_size_mb"] = round(_TMIA_DUCKDB.stat().st_size / 1024**2, 2)
    else:
        out["last_run"] = None
        out["next_run_after"] = "as soon as data is available"
        out["file_size_mb"] = 0.0
    return out


def ensure_backtest_fresh(asof_date: Optional[date] = None) -> None:
    """Idempotent. If the cache is stale or missing, spawn the recompute
    daemon thread. Returns immediately. Mirrors the
    ``ensure_prewarm()`` pattern."""
    global _RECOMPUTE_STARTED, _RECOMPUTE_THREAD
    if is_backtest_fresh():
        return
    with _LOCK:
        if _RECOMPUTE_STARTED:
            return
        _RECOMPUTE_STARTED = True
        _STATUS["started_at"] = time.time()
        _STATUS["running"] = True
        _RECOMPUTE_THREAD = threading.Thread(
            target=_recompute_worker,
            args=(asof_date,),
            name="backtest-recompute",
            daemon=True,
        )
        _RECOMPUTE_THREAD.start()


def force_recompute_now(asof_date: Optional[date] = None) -> None:
    """Manual override: kick off a recompute regardless of cache freshness.
    Used by the Settings-tab 'Force recompute' button. Asynchronous —
    returns immediately."""
    global _RECOMPUTE_STARTED, _RECOMPUTE_THREAD
    with _LOCK:
        if _RECOMPUTE_STARTED and _STATUS["running"]:
            return  # already running; don't double-start
        _RECOMPUTE_STARTED = True
        _STATUS["started_at"] = time.time()
        _STATUS["running"] = True
        _STATUS["errors"] = []
        _RECOMPUTE_THREAD = threading.Thread(
            target=_recompute_worker,
            args=(asof_date, True),
            name="backtest-recompute-forced",
            daemon=True,
        )
        _RECOMPUTE_THREAD.start()


# =============================================================================
# Worker — full recompute pipeline
# =============================================================================

def _recompute_worker(asof_date: Optional[date] = None,
                          force: bool = False) -> None:
    """Daemon-thread body. Best-effort: errors are logged to ``_STATUS``
    but never re-raised."""
    try:
        from lib.cmc import build_cmc_nodes, load_cmc_panel
        from lib.sra_data import get_sra_snapshot_latest_date

        if asof_date is None:
            snap = get_sra_snapshot_latest_date()
            if snap is None:
                _STATUS["errors"].append("snapshot date unavailable")
                _STATUS["running"] = False
                return
            asof_date = snap

        _STATUS["asof_date"] = str(asof_date)

        # 1. Make sure CMC layer is built for this asof
        for scope in ("outright", "spread", "fly"):
            try:
                build_cmc_nodes(scope, asof_date)
            except Exception as e:
                _STATUS["errors"].append(f"CMC {scope}: {e}")

        # 2. Run the full backtest pipeline
        try:
            n_trades, n_cells = recompute_full_backtest(asof_date)
            _STATUS["n_trades"] = n_trades
            _STATUS["n_cells"] = n_cells
        except Exception as e:
            _STATUS["errors"].append(f"recompute: {e}")
            _STATUS["errors"].append(traceback.format_exc()[:500])

        # 3. Drop a dated snapshot
        try:
            _drop_dated_snapshot(asof_date)
        except Exception as e:
            _STATUS["errors"].append(f"snapshot copy: {e}")

        _STATUS["completed_at"] = time.time()

    except Exception as e:
        _STATUS["errors"].append(f"worker: {e}")
        _STATUS["errors"].append(traceback.format_exc()[:500])
    finally:
        _STATUS["running"] = False


def recompute_full_backtest(asof_date: date) -> tuple:
    """Full pipeline: CMC panels → detector replay → trade simulation →
    aggregation → write to tmia.duckdb.

    Returns ``(n_trades, n_cells)``.
    """
    from lib.backtest.replay import replay_all_detectors_on_cmc
    from lib.backtest.engine import simulate_trades
    from lib.backtest.aggregator import build_metrics_grid

    # Per-scope replay → fires per (setup × node × direction)
    all_trades = []
    for scope in ("outright", "spread", "fly"):
        try:
            fires_by_cell = replay_all_detectors_on_cmc(scope, asof_date)
        except Exception as e:
            _STATUS["errors"].append(f"replay {scope}: {e}")
            continue
        for cell_key, payload in fires_by_cell.items():
            setup_id = payload["setup_id"]
            cmc_node = payload["cmc_node"]
            fires = payload["fires"]
            panel = payload["panel"]
            bp_per_unit = payload["bp_per_unit"]
            if fires.empty:
                continue
            try:
                trades = simulate_trades(panel, fires, setup_id, cmc_node,
                                            bp_per_unit=bp_per_unit)
                if not trades.empty:
                    all_trades.append(trades)
            except Exception as e:
                _STATUS["errors"].append(f"simulate {cell_key}: {e}")

    if not all_trades:
        # Write empty tables so loaders don't crash
        _write_empty_tmia_duckdb()
        return (0, 0)

    trades_df = pd.concat(all_trades, ignore_index=True)

    # Aggregate
    metrics_df, calendar_df, trend_df = build_metrics_grid(
        trades_df, asof_date, n_bootstrap=200)

    # Persist
    _write_tmia_duckdb(trades_df, metrics_df, calendar_df, trend_df)
    return (len(trades_df), len(metrics_df))


def _write_tmia_duckdb(trades_df: pd.DataFrame,
                          metrics_df: pd.DataFrame,
                          calendar_df: pd.DataFrame,
                          trend_df: pd.DataFrame) -> None:
    """Persist all four output tables to ``tmia.duckdb``."""
    import duckdb
    if _TMIA_DUCKDB.exists():
        _TMIA_DUCKDB.unlink()
    con = duckdb.connect(str(_TMIA_DUCKDB))
    try:
        con.register("trades_df", trades_df)
        con.execute("CREATE TABLE tmia_backtest_trades AS SELECT * FROM trades_df")
        con.register("metrics_df", metrics_df)
        con.execute("CREATE TABLE tmia_backtest_metrics AS SELECT * FROM metrics_df")
        con.register("calendar_df", calendar_df)
        con.execute("CREATE TABLE tmia_backtest_calendar_slices AS SELECT * FROM calendar_df")
        con.register("trend_df", trend_df)
        con.execute("CREATE TABLE tmia_backtest_trend AS SELECT * FROM trend_df")
    finally:
        con.close()


def _write_empty_tmia_duckdb() -> None:
    """Create the duckdb with empty stubs of each table — used when no
    trades are produced (degenerate edge case so downstream readers don't
    blow up)."""
    import duckdb
    if _TMIA_DUCKDB.exists():
        _TMIA_DUCKDB.unlink()
    con = duckdb.connect(str(_TMIA_DUCKDB))
    try:
        con.execute("CREATE TABLE tmia_backtest_trades (setup_id VARCHAR)")
        con.execute("CREATE TABLE tmia_backtest_metrics (setup_id VARCHAR)")
        con.execute("CREATE TABLE tmia_backtest_calendar_slices (setup_id VARCHAR)")
        con.execute("CREATE TABLE tmia_backtest_trend (setup_id VARCHAR)")
    finally:
        con.close()


def _drop_dated_snapshot(asof_date: date) -> None:
    """Copy ``tmia.duckdb`` → ``snapshots/tmia_<YYYY-MM-DD>.duckdb`` and
    prune older snapshots, keeping the last ``N_SNAPSHOTS_RETAINED``."""
    if not _TMIA_DUCKDB.exists():
        return
    target = _SNAPSHOT_DIR / f"tmia_{asof_date.isoformat()}.duckdb"
    shutil.copy2(_TMIA_DUCKDB, target)
    # Prune older snapshots beyond the retention window
    snapshots = sorted(_SNAPSHOT_DIR.glob("tmia_*.duckdb"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
    for old in snapshots[N_SNAPSHOTS_RETAINED:]:
        try:
            old.unlink()
        except OSError:
            pass


# =============================================================================
# Loaders
# =============================================================================

def load_metrics_grid() -> pd.DataFrame:
    """Read the tmia_backtest_metrics table from disk. Empty DF if not present."""
    return _read_table("tmia_backtest_metrics")


def load_trades() -> pd.DataFrame:
    """Read the tmia_backtest_trades table from disk."""
    return _read_table("tmia_backtest_trades")


def load_calendar_slices() -> pd.DataFrame:
    """Read the tmia_backtest_calendar_slices table from disk."""
    return _read_table("tmia_backtest_calendar_slices")


def load_trend_table() -> pd.DataFrame:
    """Read the tmia_backtest_trend table from disk."""
    return _read_table("tmia_backtest_trend")


def _read_table(name: str) -> pd.DataFrame:
    if not _TMIA_DUCKDB.exists():
        return pd.DataFrame()
    import duckdb
    con = duckdb.connect(str(_TMIA_DUCKDB), read_only=True)
    try:
        return con.execute(f"SELECT * FROM {name}").fetchdf()
    except Exception:
        return pd.DataFrame()
    finally:
        con.close()
