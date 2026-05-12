"""Background-thread daemon for the Phase 3 turn-adjuster.

Mirrors :mod:`lib.prewarm` exactly: module-level lock + idempotent flag +
``threading.Thread(daemon=True)``. Best-effort; errors are captured into the
status dict but never propagate to the UI thread.

Freshness rule:
    The turn-residuals cache is rebuilt iff ANY of:
      (a) ``turn_residuals_<asof>.parquet`` is missing for the latest CMC asof; OR
      (b) the residuals parquet's mtime is OLDER than the matching CMC parquet
          (CMC was rebuilt; residuals are now stale).

Boot ordering (in app.py):
    ensure_prewarm()            # technicals universe scan (existing)
    ensure_backtest_fresh()     # 10-day backtest cycle (existing)
    ensure_turn_residuals_fresh()  # Phase 3 — runs after CMC is on disk

This call is non-blocking; the build itself is ~1-2 minutes cold but does
not block the Streamlit UI.
"""
from __future__ import annotations

import os
import threading
import time
import traceback
from datetime import date
from pathlib import Path
from typing import Optional


_LOCK = threading.Lock()
_DAEMON_STARTED = False
_DAEMON_THREAD: "threading.Thread | None" = None
_DAEMON_STATUS: dict = {
    "started_at": None,
    "asof_date": None,
    "completed_at": None,
    "n_nodes_total": None,
    "n_nodes_skipped": None,
    "skipped_reason": None,    # 'fresh' | 'no_cmc' | None
    "errors": [],
}


def _resolve_latest_cmc_asof() -> Optional[date]:
    """Newest CMC manifest_<asof>.json date in .cmc_cache/, or None."""
    cache = Path(__file__).resolve().parent.parent / ".cmc_cache"
    cands = sorted(cache.glob("manifest_*.json"),
                       key=os.path.getmtime, reverse=True)
    if not cands:
        return None
    stem = cands[0].stem.replace("manifest_", "")
    try:
        return date.fromisoformat(stem)
    except ValueError:
        return None


def is_turn_residuals_fresh(asof_date: Optional[date] = None) -> bool:
    """True iff turn_residuals_<asof>.parquet exists for ``asof_date`` AND
    its mtime is at least as new as the matching CMC outright parquet.

    If ``asof_date`` is None, resolves the latest CMC asof.
    """
    cache = Path(__file__).resolve().parent.parent / ".cmc_cache"
    if asof_date is None:
        asof_date = _resolve_latest_cmc_asof()
        if asof_date is None:
            return False
    res_path = cache / f"turn_residuals_{asof_date.isoformat()}.parquet"
    cmc_path = cache / f"sra_outright_{asof_date.isoformat()}.parquet"
    if not res_path.exists():
        return False
    if not cmc_path.exists():
        # CMC missing — residuals can't be "fresh" against a non-existent base.
        return False
    return os.path.getmtime(res_path) >= os.path.getmtime(cmc_path)


def _turn_residuals_worker() -> None:
    """Background body — locate latest CMC asof + build residuals. Errors
    captured into status dict; never re-raised."""
    try:
        cmc_asof = _resolve_latest_cmc_asof()
        if cmc_asof is None:
            _DAEMON_STATUS["skipped_reason"] = "no_cmc"
            _DAEMON_STATUS["errors"].append(
                "no CMC manifest_*.json found in .cmc_cache/")
            return
        _DAEMON_STATUS["asof_date"] = cmc_asof.isoformat()

        if is_turn_residuals_fresh(cmc_asof):
            _DAEMON_STATUS["skipped_reason"] = "fresh"
            _DAEMON_STATUS["completed_at"] = time.time()
            return

        from lib.analytics.turn_adjuster import build_turn_residuals
        manifest = build_turn_residuals(cmc_asof)
        _DAEMON_STATUS["n_nodes_total"] = manifest.get("n_nodes_total")
        _DAEMON_STATUS["n_nodes_skipped"] = len(manifest.get("missing_nodes", []))
        _DAEMON_STATUS["completed_at"] = time.time()

    except Exception as e:    # noqa: BLE001 — outer safety net
        _DAEMON_STATUS["errors"].append(f"worker: {e}")
        _DAEMON_STATUS["errors"].append(traceback.format_exc()[:500])


def ensure_turn_residuals_fresh() -> None:
    """Idempotent spawn of the background turn-adjuster build. Returns
    immediately. Mirrors :func:`lib.prewarm.ensure_prewarm`."""
    global _DAEMON_STARTED, _DAEMON_THREAD
    with _LOCK:
        if _DAEMON_STARTED:
            return
        _DAEMON_STARTED = True
        _DAEMON_STATUS["started_at"] = time.time()
        _DAEMON_THREAD = threading.Thread(
            target=_turn_residuals_worker,
            name="turn-adjuster-daemon",
            daemon=True,
        )
        _DAEMON_THREAD.start()


def get_turn_adjuster_status() -> dict:
    """Snapshot for Settings → System Health."""
    return dict(_DAEMON_STATUS)


def is_turn_adjuster_done() -> bool:
    """True iff the residual build has completed (or was skipped as fresh)."""
    return _DAEMON_STATUS.get("completed_at") is not None
