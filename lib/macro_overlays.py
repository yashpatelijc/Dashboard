"""Phase 12 macro overlays + ECO calendar (plan §12).

Builds the next-7-days economic-release calendar from BBG eco/ files and
provides FOMC blackout vertical-band annotations for any Plotly figure.

Per §15 / D2: ECO calendar reads from local eco/ parquets only (no scrapers).
Per §15 / D2: Fed-dots chart uses aggregate SEP only (individual-member
dispersion deferred — no per-member scrape).

API:
    eco_calendar_next_7_days(asof) -> pd.DataFrame
    add_fomc_blackout_bands(fig, start, end) -> Figure (modifies in place)
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_BBG_ECO = Path(r"D:\BBG data\parquet\eco")

# Curated set of high-impact tickers for the calendar widget. Reuses the
# Phase 6 list but is presented in user-friendly form.
HIGH_IMPACT_TICKERS = (
    ("NFP_TCH_Index",      "NFP — Nonfarm Payrolls Change"),
    ("USURTOT_Index",      "Unemployment Rate"),
    ("AHE_YOY_Index",      "Avg Hourly Earnings YoY"),
    ("INJCJC_Index",       "Initial Jobless Claims"),
    ("CPI_CHNG_Index",     "CPI MoM"),
    ("CPI_YOY_Index",      "CPI YoY"),
    ("CPI_XYOY_Index",     "Core CPI YoY"),
    ("PPI_FINL_Index",     "PPI Final Demand MoM"),
    ("PCEDEFY_Index",      "PCE Deflator YoY (core)"),
    ("GDP_CQOQ_Index",     "GDP Annualised QoQ"),
    ("INDPRO_Index",       "Industrial Production MoM"),
    ("RSTAMOM_Index",      "Retail Sales MoM"),
    ("NAPM_PMI_Index",     "ISM Manufacturing"),
    ("NAPMNMI_Index",      "ISM Services"),
    ("CONCCONF_Index",     "Conf Board Consumer Confidence"),
    ("CONSSENT_Index",     "UMich Consumer Sentiment"),
    ("ECI_QOQ_Index",      "Employment Cost Index QoQ"),
    ("DGNOCHNG_Index",     "Durable Goods Orders MoM"),
    ("USHBHIME_Index",     "Housing Starts"),
    ("HMSI_INDX_Index",    "NAHB Housing Index"),
)


def _load_release_dates(ticker: str) -> Optional[pd.DataFrame]:
    p = _BBG_ECO / f"{ticker}.parquet"
    if not p.exists():
        return None
    try:
        import duckdb
        con = duckdb.connect(":memory:")
        path_str = str(p).replace("\\", "/")
        df = con.execute(
            f"SELECT date, ECO_RELEASE_DT, ACTUAL_RELEASE, "
            f"PX_LAST, BN_SURVEY_MEDIAN "
            f"FROM read_parquet('{path_str}')"
        ).fetchdf()
        return df
    except Exception:
        return None


def eco_calendar_next_7_days(asof: Optional[date] = None) -> pd.DataFrame:
    """Best-effort next-7-day ECO calendar. Returns empty DF if eco/ files
    haven't been refreshed with future ECO_RELEASE_DT entries (BBG terminal
    pre-publishes upcoming releases on subscription)."""
    if asof is None:
        asof = date.today()
    end = asof + timedelta(days=7)
    rows = []
    for ticker, label in HIGH_IMPACT_TICKERS:
        df = _load_release_dates(ticker)
        if df is None or df.empty:
            continue
        # Parse ECO_RELEASE_DT as YYYYMMDD
        for _, row in df.iterrows():
            eco_dt = row.get("ECO_RELEASE_DT")
            if pd.isna(eco_dt):
                continue
            try:
                s = str(int(eco_dt))
                rd = date(int(s[:4]), int(s[4:6]), int(s[6:8]))
            except (ValueError, TypeError):
                continue
            if asof <= rd <= end:
                rows.append({
                    "release_date": rd,
                    "ticker": ticker,
                    "label": label,
                    "actual": row.get("ACTUAL_RELEASE"),
                    "survey_median": row.get("BN_SURVEY_MEDIAN"),
                    "released": pd.notna(row.get("ACTUAL_RELEASE")),
                })
    if not rows:
        # Fallback: surface most-recent prints (last 7 days) for legacy view
        for ticker, label in HIGH_IMPACT_TICKERS:
            df = _load_release_dates(ticker)
            if df is None or df.empty:
                continue
            df = df.dropna(subset=["ECO_RELEASE_DT"]).sort_values(
                "ECO_RELEASE_DT", ascending=False).head(1)
            for _, row in df.iterrows():
                eco_dt = row.get("ECO_RELEASE_DT")
                try:
                    s = str(int(eco_dt))
                    rd = date(int(s[:4]), int(s[4:6]), int(s[6:8]))
                except (ValueError, TypeError):
                    continue
                if rd >= asof - timedelta(days=14):
                    rows.append({
                        "release_date": rd,
                        "ticker": ticker,
                        "label": label,
                        "actual": row.get("ACTUAL_RELEASE"),
                        "survey_median": row.get("BN_SURVEY_MEDIAN"),
                        "released": pd.notna(row.get("ACTUAL_RELEASE")),
                    })
    if not rows:
        return pd.DataFrame()
    cal = pd.DataFrame(rows).sort_values("release_date")
    # Surprise sign + magnitude
    cal["surprise"] = cal.apply(
        lambda r: (float(r["actual"]) - float(r["survey_median"]))
                      if (pd.notna(r["actual"]) and pd.notna(r["survey_median"]))
                      else None,
        axis=1,
    )
    return cal.reset_index(drop=True)


def add_fomc_blackout_bands(fig, start: date, end: date):
    """Add vertical FOMC blackout bands to a plotly figure between start/end."""
    try:
        from lib.cb_meetings import fomc_meetings_in_range, fomc_blackout_window
        meetings = fomc_meetings_in_range(start, end)
        for m in meetings:
            bo_start, bo_end = fomc_blackout_window(m.date)
            fig.add_vrect(
                x0=str(bo_start), x1=str(bo_end),
                fillcolor="rgba(232,183,93,0.08)", opacity=0.4,
                line_width=0, layer="below",
                annotation_text="FOMC", annotation_position="top left",
                annotation_font_size=9,
            )
    except Exception:
        pass
    return fig
