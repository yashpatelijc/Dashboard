"""Central-bank meeting calendar loader + realized-outcome derivation
(Phase 1).

Loads ``config/cb_meetings.yaml`` (FOMC + future cross-bank meetings) and
derives realized policy outcomes (hike/hold/cut + bp magnitude) from
``rates_drivers/FDTRMID_Index.parquet`` step changes around each meeting
date.

Used by:
  - A4 Heitfield-Park sequential bootstrap (per-meeting probability vectors)
  - A11-event Tier 1 (FOMC + statement)
  - §3.7 FOMC blackouts on every plotly chart
  - §3.7 Fed-dots-vs-OIS chart anchor dates
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml


_MEETINGS_PATH = (Path(__file__).resolve().parent.parent
                     / "config" / "cb_meetings.yaml")


_MEETINGS_CACHE: Optional[dict] = None


@dataclass
class FOMCMeeting:
    date: date
    has_sep: bool
    statement_only: bool
    emergency: bool = False
    realized_change_bp: Optional[float] = None    # populated from FDTRMID
    pre_target_bp: Optional[float] = None
    post_target_bp: Optional[float] = None
    direction: Optional[str] = None              # 'HIKE' | 'CUT' | 'HOLD'
    magnitude_bp: Optional[float] = None


def load_cb_meetings(force_reload: bool = False) -> dict:
    """Load + parse ``cb_meetings.yaml``. Returns nested dict per central
    bank with parsed dates."""
    global _MEETINGS_CACHE
    if not force_reload and _MEETINGS_CACHE is not None:
        return _MEETINGS_CACHE
    if not _MEETINGS_PATH.exists():
        raise FileNotFoundError(f"cb_meetings.yaml not found at {_MEETINGS_PATH}")
    raw = yaml.safe_load(_MEETINGS_PATH.read_text())
    out = {}
    for cb_code, cb_data in raw.items():
        meetings_raw = cb_data.get("meetings", [])
        meetings = [
            FOMCMeeting(
                date=date.fromisoformat(str(m["date"])),
                has_sep=bool(m.get("has_sep", False)),
                statement_only=bool(m.get("statement_only", False)),
                emergency=bool(m.get("emergency", False)),
            )
            for m in meetings_raw
        ]
        out[cb_code] = {
            "blackout_rule": cb_data.get("blackout_rule", {}),
            "meetings": meetings,
        }
    _MEETINGS_CACHE = out
    return out


def fomc_meetings() -> list[FOMCMeeting]:
    """All FOMC meetings (past + future) sorted by date."""
    data = load_cb_meetings()
    fed = data.get("fed", {})
    return sorted(fed.get("meetings", []), key=lambda m: m.date)


def fomc_meetings_in_range(start: date, end: date) -> list[FOMCMeeting]:
    """All FOMC meetings between (inclusive) ``start`` and ``end``."""
    return [m for m in fomc_meetings() if start <= m.date <= end]


def next_fomc(asof: date) -> Optional[FOMCMeeting]:
    """The next FOMC meeting on or after ``asof``."""
    upcoming = [m for m in fomc_meetings() if m.date >= asof]
    return upcoming[0] if upcoming else None


def fomc_blackout_window(meeting_date: date) -> tuple[date, date]:
    """Per the Fed's blackout rule: starts on the second Saturday before
    the meeting, ends on the Thursday after the meeting."""
    data = load_cb_meetings()
    rule = data.get("fed", {}).get("blackout_rule", {})
    pre_days = int(rule.get("pre_days", 12))
    post_days = int(rule.get("post_days", 1))
    return (meeting_date - timedelta(days=pre_days),
              meeting_date + timedelta(days=post_days))


def is_in_fomc_blackout(d: date) -> Optional[FOMCMeeting]:
    """Return the meeting whose blackout window contains ``d``, or None."""
    for m in fomc_meetings():
        start, end = fomc_blackout_window(m.date)
        if start <= d <= end:
            return m
    return None


# =============================================================================
# Realized-outcome derivation from FDTRMID
# =============================================================================

_FDTRMID_PATH = Path(r"D:\BBG data\parquet\rates_drivers\FDTRMID_Index.parquet")


def _load_fdtr_series() -> Optional[pd.Series]:
    """Load FDTRMID daily series. Returns None if file missing."""
    if not _FDTRMID_PATH.exists():
        return None
    try:
        import duckdb
        path = str(_FDTRMID_PATH).replace("\\", "/")
        con = duckdb.connect(":memory:")
        df = con.execute(
            f"SELECT date, PX_LAST FROM read_parquet('{path}') ORDER BY date"
        ).fetchdf()
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df.set_index("date")["PX_LAST"].astype(float).dropna()
    except Exception:
        return None


def populate_realized_outcomes(meetings: list[FOMCMeeting]) -> list[FOMCMeeting]:
    """For each past meeting, fill in realized_change_bp / pre_target_bp /
    post_target_bp / direction / magnitude_bp from FDTRMID step changes
    around the meeting date.

    For FUTURE meetings the realized_change_bp stays None.
    For meetings before FDTRMID's earliest date the same.
    """
    series = _load_fdtr_series()
    if series is None or series.empty:
        return meetings

    today = date.today()
    out = []
    for m in meetings:
        if m.date >= today:
            out.append(m)
            continue
        # Find the closest available pre/post observations
        try:
            pre_idx = max(d for d in series.index if d < m.date)
            post_idx = min(d for d in series.index if d >= m.date)
        except ValueError:
            out.append(m)
            continue
        pre_pct = float(series[pre_idx])
        post_pct = float(series[post_idx])
        delta_pct = post_pct - pre_pct
        delta_bp = delta_pct * 100.0   # pct → bp
        m.pre_target_bp = pre_pct * 100.0
        m.post_target_bp = post_pct * 100.0
        m.realized_change_bp = delta_bp
        m.magnitude_bp = abs(delta_bp)
        if abs(delta_bp) < 5:
            m.direction = "HOLD"
        elif delta_bp > 0:
            m.direction = "HIKE"
        else:
            m.direction = "CUT"
        out.append(m)
    return out


def fomc_meetings_with_outcomes() -> list[FOMCMeeting]:
    """All FOMC meetings, with realized outcomes populated where derivable.
    Use this for A3 path-conditional FV bucketing + A11-event regression
    on FOMC outcome × surprise × CMC node."""
    return populate_realized_outcomes(fomc_meetings())
