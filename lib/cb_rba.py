"""Reserve Bank of Australia (RBA) cash-rate meetings (Australia)."""
from __future__ import annotations

from datetime import date
from typing import List, Optional

import pandas as pd

from lib.central_banks import (load_meetings, get_dates_in_range,
                                    next_meeting_date, previous_meeting_date,
                                    get_blackout_window)


def load_rba_meetings() -> pd.DataFrame:
    return load_meetings("rba")


def get_rba_dates_in_range(start: date, end: date) -> List[date]:
    return get_dates_in_range("rba", start, end)


def next_rba_date(asof: date) -> Optional[date]:
    return next_meeting_date("rba", asof)


def previous_rba_date(asof: date) -> Optional[date]:
    return previous_meeting_date("rba", asof)


def rba_blackout_window(meeting_date: date) -> tuple:
    return get_blackout_window("rba", meeting_date)
