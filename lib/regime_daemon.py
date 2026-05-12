"""Background-thread daemon for the Phase 4 A1 regime classifier.

Mirrors :mod:`lib.turn_adjuster_daemon` exactly. Spawns after
``ensure_turn_residuals_fresh()`` since regime fits depend on Phase 3 output.

Freshness rule:
    Regime cache is rebuilt iff residuals parquet is newer than regime_states
    parquet OR regime cache is missing.
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
    "K": None,
    "n_active_states": None,
    "skipped_reason": None,
    "errors": [],
}


def _resolve_latest_residuals_asof() -> Optional[date]:
    cache = Path(__file__).resolve().parent.parent / ".cmc_cache"
    cands = sorted(cache.glob("turn_residuals_manifest_*.json"),
                       key=os.path.getmtime, reverse=True)
    if not cands:
        return None
    try:
        return date.fromisoformat(
            cands[0].stem.replace("turn_residuals_manifest_", ""))
    except ValueError:
        return None


def is_regime_fresh(asof: Optional[date] = None) -> bool:
    cache = Path(__file__).resolve().parent.parent / ".cmc_cache"
    if asof is None:
        asof = _resolve_latest_residuals_asof()
        if asof is None:
            return False
    reg = cache / f"regime_states_{asof.isoformat()}.parquet"
    res = cache / f"turn_residuals_{asof.isoformat()}.parquet"
    if not reg.exists() or not res.exists():
        return False
    return os.path.getmtime(reg) >= os.path.getmtime(res)


def _regime_worker() -> None:
    try:
        asof = _resolve_latest_residuals_asof()
        if asof is None:
            _DAEMON_STATUS["skipped_reason"] = "no_residuals"
            _DAEMON_STATUS["errors"].append("Phase 3 residuals not available")
            return
        _DAEMON_STATUS["asof_date"] = asof.isoformat()
        if is_regime_fresh(asof):
            _DAEMON_STATUS["skipped_reason"] = "fresh"
            _DAEMON_STATUS["completed_at"] = time.time()
            return
        from lib.analytics.regime_a1 import build_regime
        manifest = build_regime(asof)
        _DAEMON_STATUS["K"] = manifest.get("K")
        # Count active states from diagnostics file
        import pandas as pd
        cache = Path(__file__).resolve().parent.parent / ".cmc_cache"
        diag = pd.read_parquet(cache / f"regime_diagnostics_{asof.isoformat()}.parquet")
        _DAEMON_STATUS["n_active_states"] = int((diag["n_bars"] > 0).sum())
        _DAEMON_STATUS["completed_at"] = time.time()
    except Exception as e:
        _DAEMON_STATUS["errors"].append(f"worker: {e}")
        _DAEMON_STATUS["errors"].append(traceback.format_exc()[:500])


def ensure_regime_fresh() -> None:
    global _DAEMON_STARTED, _DAEMON_THREAD
    with _LOCK:
        if _DAEMON_STARTED:
            return
        _DAEMON_STARTED = True
        _DAEMON_STATUS["started_at"] = time.time()
        _DAEMON_THREAD = threading.Thread(
            target=_regime_worker, name="regime-daemon", daemon=True,
        )
        _DAEMON_THREAD.start()


def get_regime_status() -> dict:
    return dict(_DAEMON_STATUS)


def is_regime_done() -> bool:
    return _DAEMON_STATUS.get("completed_at") is not None
