"""Phase 11 cross-cutting daemon."""
from __future__ import annotations
import os, threading, time, traceback
from datetime import date
from pathlib import Path
from typing import Optional

_LOCK = threading.Lock()
_STARTED = False
_STATUS: dict = {"started_at": None, "asof_date": None, "completed_at": None,
                       "skipped_reason": None, "errors": [],
                       "risk_regime_label": None}


def _resolve_asof() -> Optional[date]:
    cache = Path(__file__).resolve().parent.parent / ".cmc_cache"
    cands = sorted(cache.glob("regime_manifest_*.json"),
                       key=os.path.getmtime, reverse=True)
    if not cands:
        return None
    try:
        return date.fromisoformat(cands[0].stem.replace("regime_manifest_", ""))
    except ValueError:
        return None


def is_cross_cutting_fresh(asof: Optional[date] = None) -> bool:
    cache = Path(__file__).resolve().parent.parent / ".cmc_cache"
    if asof is None:
        asof = _resolve_asof()
        if asof is None:
            return False
    cc = cache / f"cross_cutting_manifest_{asof.isoformat()}.json"
    rg = cache / f"regime_states_{asof.isoformat()}.parquet"
    if not cc.exists() or not rg.exists():
        return False
    return os.path.getmtime(cc) >= os.path.getmtime(rg)


def _worker():
    try:
        asof = _resolve_asof()
        if asof is None:
            _STATUS["skipped_reason"] = "no_regime"; return
        _STATUS["asof_date"] = asof.isoformat()
        if is_cross_cutting_fresh(asof):
            _STATUS["skipped_reason"] = "fresh"; _STATUS["completed_at"] = time.time(); return
        from lib.analytics.cross_cutting import build_cross_cutting
        m = build_cross_cutting(asof)
        _STATUS["risk_regime_label"] = m.get("risk_regime_label")
        _STATUS["completed_at"] = time.time()
    except Exception as e:
        _STATUS["errors"].append(f"worker: {e}")
        _STATUS["errors"].append(traceback.format_exc()[:500])


def ensure_cross_cutting_fresh() -> None:
    global _STARTED
    with _LOCK:
        if _STARTED: return
        _STARTED = True
        _STATUS["started_at"] = time.time()
        threading.Thread(target=_worker, name="cross-cutting-daemon", daemon=True).start()


def get_cross_cutting_status() -> dict:
    return dict(_STATUS)
