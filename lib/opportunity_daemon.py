"""Phase 9 opportunity-modules daemon."""
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
                       "modules_built": None}


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


def is_opportunity_fresh(asof: Optional[date] = None) -> bool:
    cache = Path(__file__).resolve().parent.parent / ".cmc_cache"
    if asof is None:
        asof = _resolve_latest_regime_asof()
        if asof is None:
            return False
    opp = cache / f"opp_manifest_{asof.isoformat()}.json"
    rg = cache / f"regime_states_{asof.isoformat()}.parquet"
    if not opp.exists() or not rg.exists():
        return False
    return os.path.getmtime(opp) >= os.path.getmtime(rg)


def _worker():
    try:
        asof = _resolve_latest_regime_asof()
        if asof is None:
            _STATUS["skipped_reason"] = "no_regime"
            return
        _STATUS["asof_date"] = asof.isoformat()
        if is_opportunity_fresh(asof):
            _STATUS["skipped_reason"] = "fresh"
            _STATUS["completed_at"] = time.time()
            return
        from lib.analytics.opportunity_modules import build_opportunity_modules
        manifest = build_opportunity_modules(asof)
        _STATUS["modules_built"] = manifest["modules_built"]
        _STATUS["completed_at"] = time.time()
    except Exception as e:
        _STATUS["errors"].append(f"worker: {e}")
        _STATUS["errors"].append(traceback.format_exc()[:500])


def ensure_opportunity_fresh() -> None:
    global _STARTED, _THREAD
    with _LOCK:
        if _STARTED:
            return
        _STARTED = True
        _STATUS["started_at"] = time.time()
        _THREAD = threading.Thread(target=_worker, name="opportunity-daemon", daemon=True)
        _THREAD.start()


def get_opportunity_status() -> dict:
    return dict(_STATUS)
