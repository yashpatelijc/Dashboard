"""Phase 12 alerts daemon — file-based throttled FIRED-state-change tracker.

Per spec: throttle 5/h per setup. Slack/desktop bindings deferred (require
user webhook setup).

API:
    record_fire(setup_id, asof) -> bool        # True if recorded (not throttled)
    recent_fires(window_hours=24) -> pd.DataFrame
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

_USER_STATE = Path(__file__).resolve().parent.parent / ".user_state"
_USER_STATE.mkdir(exist_ok=True)
_LOG = _USER_STATE / "alerts_log.parquet"

THROTTLE_PER_HOUR = 5


def _load() -> pd.DataFrame:
    if not _LOG.exists():
        return pd.DataFrame(columns=["setup_id", "fired_at", "asof"])
    try:
        return pd.read_parquet(_LOG)
    except Exception:
        return pd.DataFrame(columns=["setup_id", "fired_at", "asof"])


def _save(df: pd.DataFrame) -> None:
    df.to_parquet(_LOG, index=False)


def record_fire(setup_id: str, asof: date) -> bool:
    log = _load()
    now = datetime.now()
    one_hour_ago = now - timedelta(hours=1)
    recent = log[(log["setup_id"] == setup_id)
                    & (pd.to_datetime(log["fired_at"]) >= one_hour_ago)]
    if len(recent) >= THROTTLE_PER_HOUR:
        return False
    new_row = pd.DataFrame([{
        "setup_id": setup_id, "fired_at": now.isoformat(),
        "asof": asof.isoformat(),
    }])
    _save(pd.concat([log, new_row], ignore_index=True))
    return True


def recent_fires(window_hours: int = 24) -> pd.DataFrame:
    log = _load()
    if log.empty:
        return log
    cutoff = datetime.now() - timedelta(hours=window_hours)
    log["fired_at"] = pd.to_datetime(log["fired_at"])
    return log[log["fired_at"] >= cutoff].sort_values(
        "fired_at", ascending=False).reset_index(drop=True)


def clear_log() -> None:
    if _LOG.exists():
        _LOG.unlink()
