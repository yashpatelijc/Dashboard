"""Daemon for Phase 5 A4 policy-path. Mirrors regime_daemon."""
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
                       "terminal_rate_bp": None, "cycle_label": None}


def _resolve_latest_cmc_asof() -> Optional[date]:
    cache = Path(__file__).resolve().parent.parent / ".cmc_cache"
    cands = sorted(cache.glob("manifest_*.json"), key=os.path.getmtime, reverse=True)
    if not cands:
        return None
    try:
        return date.fromisoformat(cands[0].stem.replace("manifest_", ""))
    except ValueError:
        return None


def is_policy_path_fresh(asof: Optional[date] = None) -> bool:
    cache = Path(__file__).resolve().parent.parent / ".cmc_cache"
    if asof is None:
        asof = _resolve_latest_cmc_asof()
        if asof is None:
            return False
    pp = cache / f"policy_path_{asof.isoformat()}.parquet"
    cmc = cache / f"sra_outright_{asof.isoformat()}.parquet"
    if not pp.exists() or not cmc.exists():
        return False
    return os.path.getmtime(pp) >= os.path.getmtime(cmc)


def _worker():
    try:
        asof = _resolve_latest_cmc_asof()
        if asof is None:
            _STATUS["skipped_reason"] = "no_cmc"
            return
        _STATUS["asof_date"] = asof.isoformat()
        if is_policy_path_fresh(asof):
            _STATUS["skipped_reason"] = "fresh"
            _STATUS["completed_at"] = time.time()
            return
        from lib.analytics.policy_path_a4 import build_policy_path
        manifest = build_policy_path(asof)
        _STATUS["terminal_rate_bp"] = manifest.get("terminal_rate_bp")
        _STATUS["cycle_label"] = manifest.get("cycle_label")
        _STATUS["completed_at"] = time.time()
    except Exception as e:
        _STATUS["errors"].append(f"worker: {e}")
        _STATUS["errors"].append(traceback.format_exc()[:500])


def ensure_policy_path_fresh() -> None:
    global _STARTED, _THREAD
    with _LOCK:
        if _STARTED:
            return
        _STARTED = True
        _STATUS["started_at"] = time.time()
        _THREAD = threading.Thread(target=_worker, name="policy-path-daemon", daemon=True)
        _THREAD.start()


def get_policy_path_status() -> dict:
    return dict(_STATUS)
