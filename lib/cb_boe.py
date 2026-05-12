"""Bank of England MPC monetary policy meetings (UK).

Thin shim over `lib/central_banks.py`. Same schema as `load_fomc_meetings()`.
"""
from __future__ import annotations

from datetime import date
from typing import List, Optional

import pandas as pd

from lib.central_banks import (load_meetings, get_dates_in_range,
                                    next_meeting_date, previous_meeting_date,
                                    get_blackout_window)


def load_boe_meetings() -> pd.DataFrame:
    """BoE MPC monetary-policy meetings."""
    return load_meetings("boe")


def get_boe_dates_in_range(start: date, end: date) -> List[date]:
    return get_dates_in_range("boe", start, end)


def next_boe_date(asof: date) -> Optional[date]:
    return next_meeting_date("boe", asof)


def previous_boe_date(asof: date) -> Optional[date]:
    return previous_meeting_date("boe", asof)


def boe_blackout_window(meeting_date: date) -> tuple:
    return get_blackout_window("boe", meeting_date)
