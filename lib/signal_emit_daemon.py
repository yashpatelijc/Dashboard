"""Daemon for Phase 7 signal_emit. Mirrors event_impact_daemon."""
from __future__ import annotations

import os, threading, time, traceback
from datetime import date
from pathlib import Path
from typing import Optional

_LOCK = threading.Lock()
_STARTED = False
_THREAD: "threading.Thread | None" = None
_STATUS: dict = {"started_at": None, "asof_date": None, "completed_at": None,
                       "skipped_reason": None, "errors": [],
                       "n_emissions": None}


def _resolve_latest_event_impact_asof() -> Optional[date]:
    cache = Path(__file__).resolve().parent.parent / ".cmc_cache"
    cands = sorted(cache.glob("event_impact_manifest_*.json"),
                       key=os.path.getmtime, reverse=True)
    if not cands:
        return None
    try:
        return date.fromisoformat(cands[0].stem.replace("event_impact_manifest_", ""))
    except ValueError:
        return None


def is_signal_emit_fresh(asof: Optional[date] = None) -> bool:
    cache_se = Path(__file__).resolve().parent.parent / ".signal_cache"
    cache_ei = Path(__file__).resolve().parent.parent / ".cmc_cache"
    if asof is None:
        asof = _resolve_latest_event_impact_asof()
        if asof is None:
            return False
    se = cache_se / f"signal_emit_{asof.isoformat()}.parquet"
    ei = cache_ei / f"event_impact_{asof.isoformat()}.parquet"
    if not se.exists() or not ei.exists():
        return False
    return os.path.getmtime(se) >= os.path.getmtime(ei)


def _worker():
    try:
        asof = _resolve_latest_event_impact_asof()
        if asof is None:
            _STATUS["skipped_reason"] = "no_event_impact"
            return
        _STATUS["asof_date"] = asof.isoformat()
        if is_signal_emit_fresh(asof):
            _STATUS["skipped_reason"] = "fresh"
            _STATUS["completed_at"] = time.time()
            return
        from lib.signal_emit import build_signal_emit
        manifest = build_signal_emit(asof)
        _STATUS["n_emissions"] = manifest["n_emissions"]
        _STATUS["completed_at"] = time.time()
    except Exception as e:
        _STATUS["errors"].append(f"worker: {e}")
        _STATUS["errors"].append(traceback.format_exc()[:500])


def ensure_signal_emit_fresh() -> None:
    global _STARTED, _THREAD
    with _LOCK:
        if _STARTED:
            return
        _STARTED = True
        _STATUS["started_at"] = time.time()
        _THREAD = threading.Thread(target=_worker, name="signal-emit-daemon", daemon=True)
        _THREAD.start()


def get_signal_emit_status() -> dict:
    return dict(_STATUS)
