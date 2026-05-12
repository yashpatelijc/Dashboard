"""Daemon for Phase 6 A11 event-impact. Mirrors policy_path_daemon."""
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
                       "n_tickers_built": None, "n_rows": None}


def _resolve_latest_regime_asof() -> Optional[date]:
    cache = Path(__file__).resolve().parent.parent / ".cmc_cache"
    cands = sorted(cache.glob("regime_manifest_*.json"),
                       key=os.path.getmtime, reverse=True)
    if not cands:
        return None
    try:
        return date.fromisoformat(cands[0].stem.replace("regime_manifest_", ""))
    except ValueError:
        return None


def is_event_impact_fresh(asof: Optional[date] = None) -> bool:
    cache = Path(__file__).resolve().parent.parent / ".cmc_cache"
    if asof is None:
        asof = _resolve_latest_regime_asof()
        if asof is None:
            return False
    ei = cache / f"event_impact_{asof.isoformat()}.parquet"
    rg = cache / f"regime_states_{asof.isoformat()}.parquet"
    if not ei.exists() or not rg.exists():
        return False
    return os.path.getmtime(ei) >= os.path.getmtime(rg)


def _worker():
    try:
        asof = _resolve_latest_regime_asof()
        if asof is None:
            _STATUS["skipped_reason"] = "no_regime"
            return
        _STATUS["asof_date"] = asof.isoformat()
        if is_event_impact_fresh(asof):
            _STATUS["skipped_reason"] = "fresh"
            _STATUS["completed_at"] = time.time()
            return
        from lib.analytics.event_impact_a11 import build_event_impact
        manifest = build_event_impact(asof)
        _STATUS["n_tickers_built"] = manifest["n_tickers_built"]
        _STATUS["n_rows"] = manifest["n_rows"]
        _STATUS["completed_at"] = time.time()
    except Exception as e:
        _STATUS["errors"].append(f"worker: {e}")
        _STATUS["errors"].append(traceback.format_exc()[:500])


def ensure_event_impact_fresh() -> None:
    global _STARTED, _THREAD
    with _LOCK:
        if _STARTED:
            return
        _STARTED = True
        _STATUS["started_at"] = time.time()
        _THREAD = threading.Thread(target=_worker, name="event-impact-daemon", daemon=True)
        _THREAD.start()


def get_event_impact_status() -> dict:
    return dict(_STATUS)
