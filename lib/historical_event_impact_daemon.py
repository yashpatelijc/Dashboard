"""Background-thread daemon for the Historical Event Impact module.

Idempotent. Rebuilds when CMC parquet is newer than HEI outputs OR when
HEI is missing entirely. Mirrors the established daemon pattern.
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
    "skipped_reason": None,
    "n_tickers_built": None,
    "n_events_total": None,
    "errors": [],
}


def _resolve_latest_cmc_asof() -> Optional[date]:
    cache = Path(__file__).resolve().parent.parent / ".cmc_cache"
    cands = sorted(cache.glob("manifest_*.json"),
                       key=os.path.getmtime, reverse=True)
    if not cands:
        return None
    try:
        return date.fromisoformat(cands[0].stem.replace("manifest_", ""))
    except ValueError:
        return None


def is_hei_fresh(asof: Optional[date] = None) -> bool:
    cache = Path(__file__).resolve().parent.parent / ".cmc_cache"
    if asof is None:
        asof = _resolve_latest_cmc_asof()
        if asof is None:
            return False
    hei = cache / f"hei_manifest_{asof.isoformat()}.json"
    cmc = cache / f"sra_outright_{asof.isoformat()}.parquet"
    if not hei.exists() or not cmc.exists():
        return False
    return os.path.getmtime(hei) >= os.path.getmtime(cmc)


def _hei_worker() -> None:
    try:
        asof = _resolve_latest_cmc_asof()
        if asof is None:
            _DAEMON_STATUS["skipped_reason"] = "no_cmc"
            _DAEMON_STATUS["errors"].append("no CMC manifest in .cmc_cache/")
            return
        _DAEMON_STATUS["asof_date"] = asof.isoformat()
        if is_hei_fresh(asof):
            _DAEMON_STATUS["skipped_reason"] = "fresh"
            _DAEMON_STATUS["completed_at"] = time.time()
            return
        from lib.historical_event_impact import build_historical_event_impact
        manifest = build_historical_event_impact(asof)
        _DAEMON_STATUS["n_tickers_built"] = manifest.get("n_tickers_built")
        _DAEMON_STATUS["n_events_total"] = manifest.get("n_events_total")
        _DAEMON_STATUS["completed_at"] = time.time()
    except Exception as e:
        _DAEMON_STATUS["errors"].append(f"worker: {e}")
        _DAEMON_STATUS["errors"].append(traceback.format_exc()[:500])


def ensure_hei_fresh() -> None:
    """Idempotent boot-spawn. Returns immediately; build runs in background.

    Cold compute is heavy (~3-8 min depending on event count × 46
    instruments × 1H bar queries). Daemon thread is daemon=True, so it
    dies with the streamlit process.
    """
    global _DAEMON_STARTED, _DAEMON_THREAD
    with _LOCK:
        if _DAEMON_STARTED:
            return
        _DAEMON_STARTED = True
        _DAEMON_STATUS["started_at"] = time.time()
        _DAEMON_THREAD = threading.Thread(
            target=_hei_worker, name="hei-daemon", daemon=True,
        )
        _DAEMON_THREAD.start()


def get_hei_status() -> dict:
    return dict(_DAEMON_STATUS)


def is_hei_done() -> bool:
    return _DAEMON_STATUS.get("completed_at") is not None
