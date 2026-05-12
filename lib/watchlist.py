"""Phase 12 watchlist persistence — named views saved to
.user_state/watchlists.json. Per §15: per-user only (no multi-tenant locks).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

_USER_STATE = Path(__file__).resolve().parent.parent / ".user_state"
_USER_STATE.mkdir(exist_ok=True)
_FILE = _USER_STATE / "watchlists.json"


def _load() -> dict:
    if not _FILE.exists():
        return {}
    try:
        return json.loads(_FILE.read_text())
    except Exception:
        return {}


def _save(d: dict) -> None:
    _FILE.write_text(json.dumps(d, indent=2, default=str))


def list_watchlists() -> list[str]:
    return sorted(_load().keys())


def save_watchlist(name: str, payload: dict) -> None:
    """Save a named view. payload is an arbitrary dict (filters,
    selected scopes, drill-in selections)."""
    data = _load()
    data[name] = payload
    _save(data)


def load_watchlist(name: str) -> Optional[dict]:
    return _load().get(name)


def delete_watchlist(name: str) -> bool:
    data = _load()
    if name in data:
        del data[name]
        _save(data)
        return True
    return False
