"""ECB Governing Council monetary policy meetings (Eurozone).

Thin shim over `lib/central_banks.py` to provide the same API as `lib/fomc.py`.
Same schema as `load_fomc_meetings()` so the PCA engine can swap CB context
without other code changes.
"""
from __future__ import annotations

from datetime import date
from typing import List, Optional

import pandas as pd

from lib.central_banks import (load_meetings, get_dates_in_range,
                                    next_meeting_date, previous_meeting_date,
                                    get_blackout_window)


def load_ecb_meetings() -> pd.DataFrame:
    """ECB Governing Council monetary-policy meetings. Returns DataFrame with
    columns: decision_date, press_conf, sep (same shape as load_fomc_meetings)."""
    return load_meetings("ecb")


def get_ecb_dates_in_range(start: date, end: date) -> List[date]:
    return get_dates_in_range("ecb", start, end)


def next_ecb_date(asof: date) -> Optional[date]:
    return next_meeting_date("ecb", asof)


def previous_ecb_date(asof: date) -> Optional[date]:
    return previous_meeting_date("ecb", asof)


def ecb_blackout_window(meeting_date: date) -> tuple:
    return get_blackout_window("ecb", meeting_date)
