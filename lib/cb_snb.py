"""Swiss National Bank (SNB) quarterly monetary policy assessments (Switzerland)."""
from __future__ import annotations

from datetime import date
from typing import List, Optional

import pandas as pd

from lib.central_banks import (load_meetings, get_dates_in_range,
                                    next_meeting_date, previous_meeting_date,
                                    get_blackout_window)


def load_snb_meetings() -> pd.DataFrame:
    return load_meetings("snb")


def get_snb_dates_in_range(start: date, end: date) -> List[date]:
    return get_dates_in_range("snb", start, end)


def next_snb_date(asof: date) -> Optional[date]:
    return next_meeting_date("snb", asof)


def previous_snb_date(asof: date) -> Optional[date]:
    return previous_meeting_date("snb", asof)


def snb_blackout_window(meeting_date: date) -> tuple:
    return get_blackout_window("snb", meeting_date)
