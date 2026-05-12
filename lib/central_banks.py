"""Unified central-bank calendar loader.

Provides a generic `load_meetings(cb_code)` function that returns a DataFrame
with the same shape as `lib/fomc.py:load_fomc_meetings()` — columns:
  decision_date · press_conf · sep

Used by:
  - Heitfield-Park step-path bootstrap (`lib/pca_step_path.py`)
  - A11-event ranking
  - All Historical Event Impact tabs
  - `lib/pca.py:build_full_pca_panel` for fomc_calendar_dates

Each per-CB wrapper (lib/cb_ecb.py, lib/cb_boe.py, lib/cb_rba.py, lib/cb_boc.py,
lib/cb_snb.py) is a thin shim that calls `load_meetings("ecb")` etc.
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st
import yaml


_MEETINGS_PATH = (Path(__file__).resolve().parent.parent
                    / "config" / "cb_meetings.yaml")


CB_CODES = ("fed", "ecb", "boe", "rba", "boc", "snb")


@st.cache_data(show_spinner=False, ttl=86400)
def load_meetings(cb_code: str) -> pd.DataFrame:
    """Load meetings for a central bank ('fed'|'ecb'|'boe'|'rba'|'boc'|'snb').

    Returns DataFrame indexed sequentially with columns:
      decision_date (date), press_conf (bool — inverse of statement_only),
      sep (bool — has_sep).

    Same schema as lib/fomc.py:load_fomc_meetings(), so downstream consumers
    can be parameterized by cb_code with no other changes.
    """
    if not _MEETINGS_PATH.exists():
        return pd.DataFrame(columns=["decision_date", "press_conf", "sep"])
    with open(_MEETINGS_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cb_data = data.get(cb_code, {}) or {}
    meetings_raw = cb_data.get("meetings", []) or []
    if not meetings_raw:
        return pd.DataFrame(columns=["decision_date", "press_conf", "sep"])
    rows = []
    for m in meetings_raw:
        try:
            rows.append({
                "decision_date": pd.to_datetime(m["date"]).date(),
                "press_conf": not bool(m.get("statement_only", False)),
                "sep": bool(m.get("has_sep", False)),
            })
        except Exception:
            continue
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("decision_date").reset_index(drop=True)


def get_dates_in_range(cb_code: str, start: date, end: date) -> List[date]:
    """Return list of meeting decision dates within [start, end] for cb_code."""
    df = load_meetings(cb_code)
    if df.empty:
        return []
    mask = (df["decision_date"] >= start) & (df["decision_date"] <= end)
    return list(df.loc[mask, "decision_date"])


def next_meeting_date(cb_code: str, asof: date) -> Optional[date]:
    df = load_meetings(cb_code)
    upcoming = df[df["decision_date"] > asof]
    if upcoming.empty:
        return None
    return upcoming.iloc[0]["decision_date"]


def previous_meeting_date(cb_code: str, asof: date) -> Optional[date]:
    df = load_meetings(cb_code)
    past = df[df["decision_date"] <= asof]
    if past.empty:
        return None
    return past.iloc[-1]["decision_date"]


def get_blackout_window(cb_code: str, meeting_date: date) -> tuple:
    """Return (blackout_start, blackout_end) dates around a given meeting,
    per the per-CB rule. Falls back to (meeting-7d, meeting) if rule absent."""
    if not _MEETINGS_PATH.exists():
        return (meeting_date - pd.Timedelta(days=7), meeting_date)
    with open(_MEETINGS_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cb_data = data.get(cb_code, {}) or {}
    rule = cb_data.get("blackout_rule", {})
    pre_days = int(rule.get("pre_days", 7))
    post_days = int(rule.get("post_days", 0))
    return (
        meeting_date - pd.Timedelta(days=pre_days),
        meeting_date + pd.Timedelta(days=post_days),
    )
