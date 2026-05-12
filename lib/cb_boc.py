"""Bank of Canada (BoC) policy-rate meetings (Canada)."""
from __future__ import annotations

from datetime import date
from typing import List, Optional

import pandas as pd

from lib.central_banks import (load_meetings, get_dates_in_range,
                                    next_meeting_date, previous_meeting_date,
                                    get_blackout_window)


def load_boc_meetings() -> pd.DataFrame:
    return load_meetings("boc")


def get_boc_dates_in_range(start: date, end: date) -> List[date]:
    return get_dates_in_range("boc", start, end)


def next_boc_date(asof: date) -> Optional[date]:
    return next_meeting_date("boc", asof)


def previous_boc_date(asof: date) -> Optional[date]:
    return previous_meeting_date("boc", asof)


def boc_blackout_window(meeting_date: date) -> tuple:
    return get_blackout_window("boc", meeting_date)
