"""Hard 2023-08-01 LIBOR cutover gate (Phase 1, plan §15).

Single source of truth for the SOFR-First / Eurodollar→SOFR conversion
boundary. Pre-cutover SRA bars are tagged ``pre_transition`` and excluded
from:
  - A1 regime classification training
  - A2 KNN matchers (analog pool excludes pre_transition)
  - A3 path-conditional FV (analog pool excludes pre_transition)
  - A6 OU calibration
  - A11-event regressions
  - CPCV out-of-sample claim windows

Pre-cutover bars are RETAINED for legacy_diagnostic visual inspection only.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd

from lib.product_spec import libor_cutover_date


def cutover_date_for(product_code: str = "SRA") -> Optional[date]:
    """Return the LIBOR cutover date for a product. None if not applicable.

    For SRA: 2023-08-01 (CFTC SOFR-First migration completion + CME
    Eurodollar→SOFR conversion). Per plan §15 this is non-negotiable.
    """
    return libor_cutover_date(product_code)


def filter_post_cutover(df: pd.DataFrame,
                            date_col: str = "bar_date",
                            product_code: str = "SRA") -> pd.DataFrame:
    """Return rows whose ``date_col`` is on or after the cutover date.

    Used by every analytic that needs a clean post-transition sample. If
    the product has no cutover (e.g. ER, SON), returns the input
    unchanged.
    """
    cutover = cutover_date_for(product_code)
    if cutover is None or df is None or df.empty:
        return df
    if date_col not in df.columns:
        # Maybe indexed by date
        if isinstance(df.index, pd.DatetimeIndex):
            return df[df.index >= pd.Timestamp(cutover)]
        return df
    col = pd.to_datetime(df[date_col])
    cutover_ts = pd.Timestamp(cutover)
    return df.loc[col >= cutover_ts].copy()


def tag_pre_transition(df: pd.DataFrame,
                          date_col: str = "bar_date",
                          product_code: str = "SRA") -> pd.DataFrame:
    """Tag each row as ``pre_transition`` (bool) by the cutover date.
    Returns a copy; original untouched.
    """
    out = df.copy()
    cutover = cutover_date_for(product_code)
    if cutover is None or df.empty:
        out["pre_transition"] = False
        return out
    if date_col in df.columns:
        col = pd.to_datetime(df[date_col])
        out["pre_transition"] = (col < pd.Timestamp(cutover)).values
    elif isinstance(df.index, pd.DatetimeIndex):
        out["pre_transition"] = (df.index < pd.Timestamp(cutover))
    else:
        out["pre_transition"] = False
    return out


def is_post_cutover(d: date, product_code: str = "SRA") -> bool:
    """True if a single date is on or after the cutover."""
    cutover = cutover_date_for(product_code)
    if cutover is None:
        return True
    return d >= cutover
