"""Historical Event Impact engine — 1H-resolution, same-day pre/post analysis.

Per event (one of 13 major US releases post-2024-07-15), measures hourly-bar
behavior across 46 instruments (outrights / spreads / flies / packs /
pack-flies) over the T-1 / T / T+1 window. Classifies each event into one of
8 states (expected-direction × surprise-sign × surprise-magnitude), aggregates
per-state per-instrument per-measurement, and detects drift across rolling
3m / 6m / 12m / full windows.

Per design spec §3 (release-time mapping):
  - 08:30 ET: NFP, CPI, AHE, Retail Sales, GDP, Initial Claims, Unemployment,
              Durable Goods, PCE-via-NFP (subset)
  - 10:00 ET: ISM Manufacturing, ISM Services, UMich Sentiment, Conf Board

Per §4 (bar anchors):
  B_tm1_open / B_tm1_close: first / last 1H bar on date(release)-1 (ET)
  B_t_open / B_t_pre / B_t_release / B_t_post_{1h,2h,4h} / B_t_close
  B_tp1_open / B_tp1_close

Per §5 (11 measurements) — all ATR_1H_20 normalized.
Per §6 (46 instruments) — outrights + spreads + flies + packs + pack-flies.
Per §7 (8 states) — UP/DOWN × POS/NEG × small/large.
Per §8 (4 windows) — 3m / 6m / 12m / full.

Outputs 5 parquets + 1 manifest per asof.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from lib.contract_units import is_already_bp, load_catalog as _load_units_catalog

# =============================================================================
# Configuration (locked per spec)
# =============================================================================

_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cmc_cache"
_CACHE_DIR.mkdir(exist_ok=True)
_ECO_DIR = Path(r"D:\BBG data\parquet\eco")
_OHLC_SNAP_DIR = Path(r"D:\Python Projects\QH_API_APPS\analytics_kit\db_snapshot")

BUILDER_VERSION = "1.0.0"
EVENT_CUTOFF_DATE = date(2024, 7, 15)   # 1H bar history starts here

NY_TZ = ZoneInfo("America/New_York")
UTC_TZ = ZoneInfo("UTC")

# Curated 13 tickers — validated to have ≥8 events with consensus+actual post-cutoff
# This is the SRA / US-default list. Per-market lists below in EVENTS_BY_MARKET.
TICKERS_13: tuple[str, ...] = (
    "NFP_TCH_Index", "CPI_XYOY_Index", "CPI_CHNG_Index", "USURTOT_Index",
    "AHE_YOY_Index", "NAPMPMI_Index", "NAPMNMI_Index", "RSTAMOM_Index",
    "INJCJC_Index", "GDP_CQOQ_Index", "CONSSENT_Index", "CONCCONF_Index",
    "DGNOCHNG_Index",
)

TICKER_LABELS: dict[str, str] = {
    "NFP_TCH_Index":   "Nonfarm Payrolls Change",
    "CPI_XYOY_Index":  "Core CPI YoY",
    "CPI_CHNG_Index":  "Headline CPI MoM",
    "USURTOT_Index":   "Unemployment Rate",
    "AHE_YOY_Index":   "Avg Hourly Earnings YoY",
    "NAPMPMI_Index":   "ISM Manufacturing",
    "NAPMNMI_Index":   "ISM Services",
    "RSTAMOM_Index":   "Retail Sales MoM",
    "INJCJC_Index":    "Initial Jobless Claims",
    "GDP_CQOQ_Index":  "GDP QoQ Annualised",
    "CONSSENT_Index":  "UMich Consumer Sentiment",
    "CONCCONF_Index":  "Conf Board Consumer Conf",
    "DGNOCHNG_Index":  "Durable Goods Orders MoM",
}

# ============================================================================
# Per-market event ticker registry
# ============================================================================
# Each market (SRA/ER/FSR/FER/SON/YBA/CRA) has its own list of high-impact
# economic releases curated for the local central bank's reaction function.
# The HEI tab uses this list to populate the ticker dropdown — so the user
# always sees domestically-relevant indicators for the market they're viewing.

EVENTS_BY_MARKET: dict[str, tuple[str, ...]] = {
    # US — Fed reaction function (existing canonical list)
    "SRA": TICKERS_13,

    # Eurozone — ECB watches HICP, wages, German/French growth, services PMI
    "ER": (
        "ECCPEMUY_Index",   # Eurozone HICP YoY (flash)
        "CPEXEMUY_Index",   # Core HICP YoY
        "CPEXEMUM_Index",   # Core HICP MoM
        "ECCPGEY_Index",    # Germany CPI YoY
        "FRCPIYOY_Index",   # France CPI YoY
        "UMRTEMU_Index",    # Eurozone Unemployment Rate
        "EUWGRYY_Index",    # Eurozone Negotiated Wages YoY
        "EUGNEMUY_Index",   # Eurozone GDP YoY
        "MPMIEZMA_Index",   # Eurozone Manufacturing PMI
        "MPMIEZSA_Index",   # Eurozone Services PMI
        "MPMIEZCA_Index",   # Eurozone Composite PMI
        "GRZEWI_Index",     # Germany ZEW Economic Sentiment
        "GRIFPBUS_Index",   # Germany Ifo Business Climate
    ),

    # SARON / Switzerland — SNB watches CHF inflation, KOF leading
    "FSR": (
        "ECCPEMUY_Index",   # Eurozone HICP YoY (drives EUR/CHF, indirectly SNB)
        "CPEXEMUY_Index",   # Core HICP YoY
        "UMRTEMU_Index",    # Eurozone Unemployment Rate
        "EUGNEMUY_Index",   # Eurozone GDP YoY
        "MPMIEZMA_Index",   # Eurozone Manufacturing PMI
        "MPMIEZSA_Index",   # Eurozone Services PMI
        "GRZEWI_Index",     # Germany ZEW
        "GRIFPBUS_Index",   # Germany Ifo
    ),

    # FER (Eurozone €STR-equivalent) — same set as ER
    "FER": (
        "ECCPEMUY_Index", "CPEXEMUY_Index", "CPEXEMUM_Index",
        "ECCPGEY_Index", "FRCPIYOY_Index", "UMRTEMU_Index",
        "EUWGRYY_Index", "EUGNEMUY_Index",
        "MPMIEZMA_Index", "MPMIEZSA_Index", "MPMIEZCA_Index",
        "GRZEWI_Index", "GRIFPBUS_Index",
    ),

    # UK — BoE reaction function (CPI, wages, services PMI)
    "SON": (
        "UKRPCJYR_Index",   # UK CPI YoY
        "CPIYCH_Index",     # UK Core CPI YoY
        "UKRPCJMM_Index",   # UK CPI MoM
        "UKRP_Index",       # UK RPI YoY
        "UKUEILOR_Index",   # UK ILO Unemployment Rate
        "UKAVE3MN_Index",   # UK Avg Weekly Earnings 3M YoY
        "UKAWEX3R_Index",   # UK AWE ex-bonus 3M YoY
        "UKGRYBYY_Index",   # UK GDP YoY
        "UKGRYBQQ_Index",   # UK GDP QoQ
        "UKMOIPMC_Index",   # UK Manufacturing PMI
        "UKMOIPSC_Index",   # UK Services PMI
        "UKMOIPCC_Index",   # UK Composite PMI
        "UKRSAYOY_Index",   # UK Retail Sales YoY
    ),

    # Australia — RBA reaction function (CPI, wages, employment, RBA cash rate)
    "YBA": (
        "AUCPIYOY_Index",   # Australia CPI YoY
        "AUCPIQOQ_Index",   # Australia CPI QoQ
        "AUCPITRM_Index",   # Australia Trimmed Mean CPI YoY (RBA preferred core)
        "AUCPMWMM_Index",   # Australia Weighted Median CPI YoY
        "AUCPIMOM_Index",   # Australia Monthly CPI YoY
        "AULFRTE_Index",    # Australia Unemployment Rate
        "AUEMC_Index",      # Australia Employment Change
        "AULPCPYR_Index",   # Australia Wage Price Index YoY
        "AUNAGDPC_Index",   # Australia GDP YoY
        "AUNAGDPQ_Index",   # Australia GDP QoQ
        "ANZAIPMA_Index",   # Australia Manufacturing PMI
        "AURSTOTL_Index",   # Australia Retail Sales MoM
        "AUTBTBNT_Index",   # Australia Trade Balance
    ),

    # Canada — BoC reaction function (CPI, jobs, GDP, CORRA)
    "CRA": (
        "CACPIYOY_Index",   # Canada CPI YoY
        "CACPIMOM_Index",   # Canada CPI MoM
        "CACPIMCO_Index",   # Canada Core CPI YoY (CPI-Median, BoC preferred)
        "CACPIMTR_Index",   # Canada Core CPI YoY (CPI-Trim)
        "CACPIMCM_Index",   # Canada Core CPI YoY (CPI-Common)
        "CANLNETJ_Index",   # Canada Net Employment Change
        "CANLOOR_Index",    # Canada Unemployment Rate
        "CALFEMHE_Index",   # Canada Avg Hourly Wages YoY (perm employees)
        "CGE9YOY_Index",    # Canada GDP YoY
        "CGE9MOM_Index",    # Canada GDP MoM
        "CPMICAMA_Index",   # Canada Manufacturing PMI
        "CARSCONS_Index",   # Canada Retail Sales MoM
        "CATBTOTB_Index",   # Canada Trade Balance
    ),
}


# Human-readable labels per market — used by the HEI tab dropdown
LABELS_BY_MARKET: dict[str, dict[str, str]] = {
    "SRA": TICKER_LABELS,
    "ER": {
        "ECCPEMUY_Index": "Eurozone HICP YoY (flash)",
        "CPEXEMUY_Index": "Eurozone Core HICP YoY",
        "CPEXEMUM_Index": "Eurozone Core HICP MoM",
        "ECCPGEY_Index":  "Germany CPI YoY",
        "FRCPIYOY_Index": "France CPI YoY",
        "UMRTEMU_Index":  "Eurozone Unemployment Rate",
        "EUWGRYY_Index":  "Eurozone Negotiated Wages YoY",
        "EUGNEMUY_Index": "Eurozone GDP YoY",
        "MPMIEZMA_Index": "Eurozone Manufacturing PMI",
        "MPMIEZSA_Index": "Eurozone Services PMI",
        "MPMIEZCA_Index": "Eurozone Composite PMI",
        "GRZEWI_Index":   "Germany ZEW Sentiment",
        "GRIFPBUS_Index": "Germany Ifo Business Climate",
    },
    "FSR": {
        "ECCPEMUY_Index": "Eurozone HICP YoY (flash, EUR/CHF channel)",
        "CPEXEMUY_Index": "Eurozone Core HICP YoY",
        "UMRTEMU_Index":  "Eurozone Unemployment Rate",
        "EUGNEMUY_Index": "Eurozone GDP YoY",
        "MPMIEZMA_Index": "Eurozone Manufacturing PMI",
        "MPMIEZSA_Index": "Eurozone Services PMI",
        "GRZEWI_Index":   "Germany ZEW Sentiment",
        "GRIFPBUS_Index": "Germany Ifo Business Climate",
    },
    "SON": {
        "UKRPCJYR_Index": "UK CPI YoY",
        "CPIYCH_Index":   "UK Core CPI YoY",
        "UKRPCJMM_Index": "UK CPI MoM",
        "UKRP_Index":     "UK RPI YoY",
        "UKUEILOR_Index": "UK ILO Unemployment Rate",
        "UKAVE3MN_Index": "UK Avg Weekly Earnings 3M YoY",
        "UKAWEX3R_Index": "UK AWE ex-bonus 3M YoY",
        "UKGRYBYY_Index": "UK GDP YoY",
        "UKGRYBQQ_Index": "UK GDP QoQ",
        "UKMOIPMC_Index": "UK Manufacturing PMI",
        "UKMOIPSC_Index": "UK Services PMI",
        "UKMOIPCC_Index": "UK Composite PMI",
        "UKRSAYOY_Index": "UK Retail Sales YoY",
    },
    "YBA": {
        "AUCPIYOY_Index": "Australia CPI YoY",
        "AUCPIQOQ_Index": "Australia CPI QoQ",
        "AUCPITRM_Index": "Australia Trimmed Mean CPI YoY",
        "AUCPMWMM_Index": "Australia Weighted Median CPI YoY",
        "AUCPIMOM_Index": "Australia Monthly CPI YoY",
        "AULFRTE_Index":  "Australia Unemployment Rate",
        "AUEMC_Index":    "Australia Employment Change",
        "AULPCPYR_Index": "Australia Wage Price Index YoY",
        "AUNAGDPC_Index": "Australia GDP YoY",
        "AUNAGDPQ_Index": "Australia GDP QoQ",
        "ANZAIPMA_Index": "Australia Manufacturing PMI",
        "AURSTOTL_Index": "Australia Retail Sales MoM",
        "AUTBTBNT_Index": "Australia Trade Balance",
    },
    "CRA": {
        "CACPIYOY_Index": "Canada CPI YoY",
        "CACPIMOM_Index": "Canada CPI MoM",
        "CACPIMCO_Index": "Canada Core CPI YoY (CPI-Median)",
        "CACPIMTR_Index": "Canada Core CPI YoY (CPI-Trim)",
        "CACPIMCM_Index": "Canada Core CPI YoY (CPI-Common)",
        "CANLNETJ_Index": "Canada Net Employment Change",
        "CANLOOR_Index":  "Canada Unemployment Rate",
        "CALFEMHE_Index": "Canada Avg Hourly Wages YoY",
        "CGE9YOY_Index":  "Canada GDP YoY",
        "CGE9MOM_Index":  "Canada GDP MoM",
        "CPMICAMA_Index": "Canada Manufacturing PMI",
        "CARSCONS_Index": "Canada Retail Sales MoM",
        "CATBTOTB_Index": "Canada Trade Balance",
    },
}
LABELS_BY_MARKET["FER"] = LABELS_BY_MARKET["ER"]    # FER reuses ER set


def get_tickers_for_market(base_product: str) -> tuple:
    """Return the list of economic-release tickers relevant to a market."""
    return EVENTS_BY_MARKET.get(base_product, TICKERS_13)


def get_labels_for_market(base_product: str) -> dict:
    """Return ticker→label dict for a market."""
    return LABELS_BY_MARKET.get(base_product, TICKER_LABELS)

# Release-time mapping (US Eastern). All standard publishing times.
RELEASE_TIMES_ET: dict[str, time] = {
    "NFP_TCH_Index":   time(8, 30),
    "CPI_XYOY_Index":  time(8, 30),
    "CPI_CHNG_Index":  time(8, 30),
    "USURTOT_Index":   time(8, 30),
    "AHE_YOY_Index":   time(8, 30),
    "RSTAMOM_Index":   time(8, 30),
    "INJCJC_Index":    time(8, 30),
    "GDP_CQOQ_Index":  time(8, 30),
    "DGNOCHNG_Index":  time(8, 30),
    "NAPMPMI_Index":   time(10, 0),
    "NAPMNMI_Index":   time(10, 0),
    "CONSSENT_Index":  time(10, 0),
    "CONCCONF_Index":  time(10, 0),
}

# Instrument definitions (46 total: 22 + 8 + 8 + 5 + 3)
OUTRIGHT_TENORS: tuple[int, ...] = (
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 21, 24, 30, 36, 42, 48, 54, 60,
)
SPREAD_PAIRS: tuple[tuple[int, int], ...] = (
    (0, 3), (3, 6), (6, 9), (9, 12),
    (0, 6), (0, 12), (6, 12), (12, 24),
)
FLY_TRIPLETS: tuple[tuple[int, int, int], ...] = (
    (0, 3, 6), (3, 6, 9), (6, 9, 12), (0, 6, 12),
    (3, 9, 15), (6, 12, 18), (12, 18, 24), (18, 24, 30),
)
PACK_RANGES: dict[str, range] = {
    "white": range(0, 12),
    "red":   range(12, 24),
    "green": range(24, 36),
    "blue":  range(36, 48),
    "gold":  range(48, 60),
}
PACK_FLY_TRIPLETS: tuple[tuple[str, str, str], ...] = (
    ("white", "red", "green"),
    ("red",   "green", "blue"),
    ("green", "blue", "gold"),
)

# 11 measurements (see §5)
MEASUREMENTS: tuple[str, ...] = (
    "anticipation_full",       # B_t_pre − B_tm1_open
    "anticipation_overnight",  # B_t_open − B_tm1_close
    "anticipation_morning",    # B_t_pre − B_t_open
    "release_impulse",         # B_t_post_1h − B_t_pre
    "release_immediate_2h",    # B_t_post_2h − B_t_pre
    "release_short_4h",        # B_t_post_4h − B_t_pre
    "release_day_end",         # B_t_close − B_t_pre
    "next_day_open_gap",       # B_tp1_open − B_t_close
    "next_day_full",           # B_tp1_close − B_t_close
    "full_t_to_tp1",           # B_tp1_close − B_t_pre
    "full_window",             # B_tp1_close − B_tm1_open
)

# 8 states + FLAT sentinel for tie events
STATES_8: tuple[str, ...] = (
    "UP_POS_small", "UP_POS_large",
    "UP_NEG_small", "UP_NEG_large",
    "DOWN_POS_small", "DOWN_POS_large",
    "DOWN_NEG_small", "DOWN_NEG_large",
)

WINDOWS: tuple[str, ...] = ("3m", "6m", "12m", "full")

SURPRISE_SIZE_THRESHOLD_Z = 1.0   # |surprise_z| ≥ 1.0 → 'large'
ATR_LOOKBACK_HOURS = 20            # ATR_1H_20
ATR_FLOOR_PRICE = 0.005             # = 0.5 bp; prevents normalization explosion
                                       # on thin overnight ATR for deep-tenor contracts

# Drift detection thresholds
DRIFT_FADING_THRESHOLD = 0.5
DRIFT_GROWING_THRESHOLD = 1.0

# Low-sample threshold
LOW_SAMPLE_MIN_OBS = 3

# =============================================================================
# Time helpers
# =============================================================================

def release_datetime_utc(release_date: date, release_time_et: time) -> datetime:
    """Convert (ET date, ET time-of-day) → UTC datetime, DST-aware."""
    naive_et = datetime.combine(release_date, release_time_et)
    aware_et = naive_et.replace(tzinfo=NY_TZ)
    return aware_et.astimezone(UTC_TZ)


def epoch_ms(dt_utc: datetime) -> int:
    """UTC datetime → epoch milliseconds (matches mde2_timeseries.time)."""
    return int(dt_utc.timestamp() * 1000)


def floor_to_hour_ms(ms: int) -> int:
    """Floor an epoch_ms to its 1H bar start."""
    return (ms // 3600000) * 3600000


def et_date_to_utc_ms_range(d: date) -> tuple[int, int]:
    """All bars on an ET calendar day → UTC [start_ms, end_ms_exclusive)."""
    start_et = datetime(d.year, d.month, d.day, 0, 0, 0).replace(tzinfo=NY_TZ)
    end_et = start_et + timedelta(days=1)
    return (epoch_ms(start_et.astimezone(UTC_TZ)),
              epoch_ms(end_et.astimezone(UTC_TZ)))


# =============================================================================
# OHLC snapshot connection
# =============================================================================

def _resolve_ohlc_db() -> Optional[Path]:
    """Find newest market_data_v2_*.duckdb."""
    if not _OHLC_SNAP_DIR.exists():
        return None
    cands = sorted(_OHLC_SNAP_DIR.glob("market_data_v2_*.duckdb"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def _open_ohlc_con():
    """Open thread-local read-only DuckDB connection to OHLC snapshot."""
    import duckdb
    p = _resolve_ohlc_db()
    if p is None:
        raise RuntimeError("OHLC snapshot DB not found")
    return duckdb.connect(str(p), read_only=True)


# =============================================================================
# Event catalog
# =============================================================================

def _parse_eco_release_dt(eco_dt: float) -> Optional[date]:
    """ECO_RELEASE_DT stored as YYYYMMDD float (e.g. 20250905.0)."""
    if eco_dt is None or pd.isna(eco_dt):
        return None
    try:
        s = str(int(eco_dt))
        if len(s) != 8:
            return None
        return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    except (ValueError, TypeError):
        return None


def load_eco_events(ticker: str,
                          cutoff: date = EVENT_CUTOFF_DATE) -> pd.DataFrame:
    """Load all post-cutoff events for a ticker with consensus + actual."""
    p = _ECO_DIR / f"{ticker}.parquet"
    if not p.exists():
        return pd.DataFrame()
    try:
        import duckdb
        con = duckdb.connect(":memory:")
        path = str(p).replace("\\", "/")
        df = con.execute(
            f"SELECT date, ECO_RELEASE_DT, ACTUAL_RELEASE, PX_LAST, "
            f"BN_SURVEY_MEDIAN "
            f"FROM read_parquet('{path}')"
        ).fetchdf()
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    df["release_date"] = df["ECO_RELEASE_DT"].apply(_parse_eco_release_dt)
    df = df.dropna(subset=["release_date"]).copy()
    # Prefer ACTUAL_RELEASE, fall back to PX_LAST (back-revised)
    df["actual"] = df["ACTUAL_RELEASE"].where(
        df["ACTUAL_RELEASE"].notna(), df["PX_LAST"])
    df["consensus"] = df["BN_SURVEY_MEDIAN"]
    df = df.dropna(subset=["actual", "consensus"]).copy()
    df = df[df["release_date"] >= cutoff].copy()
    df = df.sort_values("release_date").reset_index(drop=True)
    return df[["release_date", "consensus", "actual"]]


def build_event_catalog(asof: Optional[date] = None) -> pd.DataFrame:
    """Build per-event catalog across all 13 tickers with state classification."""
    rows = []
    for ticker in TICKERS_13:
        df = load_eco_events(ticker)
        if df.empty or len(df) < 3:
            continue
        # Compute per-ticker historical std of expected_change + surprise
        # for z-score normalization
        df["prior_actual"] = df["actual"].shift(1)
        df = df.dropna(subset=["prior_actual"]).copy()
        if df.empty:
            continue
        df["expected_change"] = df["consensus"] - df["prior_actual"]
        df["surprise"] = df["actual"] - df["consensus"]
        exp_std = float(df["expected_change"].std(ddof=0))
        sur_std = float(df["surprise"].std(ddof=0))
        df["expected_z"] = (df["expected_change"]
                                 / max(exp_std, 1e-9))
        df["surprise_z"] = (df["surprise"]
                                / max(sur_std, 1e-9))

        # Direction + size classification
        def _direction(x: float, eps: float = 1e-12) -> str:
            if x > eps:
                return "UP"
            if x < -eps:
                return "DOWN"
            return "FLAT"

        def _sign(x: float, eps: float = 1e-12) -> str:
            if x > eps:
                return "POS"
            if x < -eps:
                return "NEG"
            return "ZERO"

        df["expected_direction"] = df["expected_change"].apply(_direction)
        df["surprise_sign"] = df["surprise"].apply(_sign)
        df["surprise_size"] = df["surprise_z"].abs().apply(
            lambda z: "large" if z >= SURPRISE_SIZE_THRESHOLD_Z else "small")

        # 8-state classification (FLAT events tagged separately)
        def _state(row):
            if row["expected_direction"] == "FLAT" or row["surprise_sign"] == "ZERO":
                return "FLAT"
            return (f"{row['expected_direction']}_"
                      f"{row['surprise_sign']}_{row['surprise_size']}")

        df["state_8"] = df.apply(_state, axis=1)
        df["ticker"] = ticker

        # Release datetime UTC
        rel_t = RELEASE_TIMES_ET.get(ticker, time(8, 30))
        df["release_datetime_utc"] = df["release_date"].apply(
            lambda d: release_datetime_utc(d, rel_t))
        df["is_post_cutoff"] = df["release_date"] >= EVENT_CUTOFF_DATE
        rows.append(df)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    # Reorder columns
    cols = ["ticker", "release_date", "release_datetime_utc",
              "prior_actual", "consensus", "actual",
              "expected_change", "expected_z", "expected_direction",
              "surprise", "surprise_z", "surprise_sign", "surprise_size",
              "state_8", "is_post_cutoff"]
    return out[cols]


# =============================================================================
# Contract-symbol mapping per event_date
# =============================================================================

def _resolve_latest_cmc_asof() -> Optional[date]:
    cands = sorted(_CACHE_DIR.glob("manifest_*.json"),
                       key=os.path.getmtime, reverse=True)
    if not cands:
        return None
    try:
        return date.fromisoformat(cands[0].stem.replace("manifest_", ""))
    except ValueError:
        return None


def build_contract_lookup(asof: Optional[date] = None) -> dict:
    """For each (event_date × Mn tenor) → SR3 outright symbol.

    Reads from the CMC outright parquet which stores `c1_sym` per
    (bar_date, cmc_node). For each event, we look up the c1_sym at the
    event_date (or closest prior bar_date if exact missing).

    Returns: {(release_date_iso, tenor_months): symbol_or_None}
    """
    if asof is None:
        asof = _resolve_latest_cmc_asof()
        if asof is None:
            raise RuntimeError("No CMC manifest found in .cmc_cache/")
    path = _CACHE_DIR / f"sra_outright_{asof.isoformat()}.parquet"
    if not path.exists():
        raise RuntimeError(f"CMC outright parquet missing: {path}")
    panel = pd.read_parquet(path)
    panel["bar_date"] = pd.to_datetime(panel["bar_date"]).dt.date
    # Pivot to (bar_date × cmc_node) → c1_sym
    return panel.set_index(["bar_date", "cmc_node"])["c1_sym"].to_dict()


def lookup_contract(contract_table: dict, event_date: date,
                          tenor_months: int) -> Optional[str]:
    """Get c1_sym for (event_date, tenor_months). Falls back to most recent
    prior bar_date if exact missing."""
    node = f"M{int(tenor_months)}"
    key = (event_date, node)
    if key in contract_table:
        return contract_table[key]
    # Walk back up to 14 days
    for back in range(1, 15):
        prior = event_date - timedelta(days=back)
        if (prior, node) in contract_table:
            return contract_table[(prior, node)]
    return None


# =============================================================================
# 1H bar loader + window slicing
# =============================================================================

def load_1h_bars_for_event(con, symbol: str, event_date: date,
                                pad_hours_before: int = 36,
                                pad_hours_after: int = 36) -> pd.DataFrame:
    """Pull 1H bars for `symbol` spanning [release_date - 1d - pad, release_date + 1d + pad].

    pad ensures we cover B_tm1_open AND have ≥20 prior bars for ATR_1H_20.
    """
    # Compute UTC range covering T-1 and T+1 with padding
    et_start = datetime(event_date.year, event_date.month, event_date.day,
                              0, 0, 0).replace(tzinfo=NY_TZ) - timedelta(days=1, hours=pad_hours_before)
    et_end = datetime(event_date.year, event_date.month, event_date.day,
                          0, 0, 0).replace(tzinfo=NY_TZ) + timedelta(days=2, hours=pad_hours_after)
    start_ms = epoch_ms(et_start.astimezone(UTC_TZ))
    end_ms = epoch_ms(et_end.astimezone(UTC_TZ))
    try:
        df = con.execute(
            "SELECT time, open, high, low, close, volume "
            "FROM mde2_timeseries "
            "WHERE base_product='SRA' AND interval='1H' "
            "  AND symbol = ? AND time >= ? AND time < ? "
            "ORDER BY time",
            [symbol, start_ms, end_ms],
        ).fetchdf()
    except Exception:
        return pd.DataFrame()
    return df


def compute_atr_1h_20(bars: pd.DataFrame, anchor_ms: int) -> float:
    """Compute 20-hour ATR anchored at `anchor_ms` (Wilder-smoothed).

    `bars` is the 1H bar DataFrame for one symbol. Anchor_ms = B_tm1_open start.
    Uses bars strictly before anchor_ms (no look-ahead).

    Returns NaN if fewer than 5 prior bars available.
    """
    pre = bars[bars["time"] < anchor_ms].tail(ATR_LOOKBACK_HOURS + 1)
    if len(pre) < 5:
        return float("nan")
    pre = pre.sort_values("time").reset_index(drop=True)
    closes = pre["close"].values
    highs = pre["high"].values
    lows = pre["low"].values
    n = len(pre)
    tr = np.zeros(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
    # Simple mean of last 20 TRs (clean + sufficient for normalization)
    valid_tr = tr[1:]   # drop first which has no prior_close reference
    if len(valid_tr) == 0:
        return float("nan")
    return float(np.mean(valid_tr[-ATR_LOOKBACK_HOURS:]))


def find_anchor_bars(bars: pd.DataFrame, event: pd.Series) -> dict:
    """Locate the 11 anchor bars in `bars` for one event.

    Returns dict: {anchor_key: close_price_or_None}.
    Also returns 'bar_indices' dict mapping anchor_key → bar position in
    `bars` (useful for per-bar trajectory plotting in UI later).
    """
    event_date = event["release_date"]
    release_dt_utc = event["release_datetime_utc"]
    if isinstance(release_dt_utc, str):
        release_dt_utc = datetime.fromisoformat(release_dt_utc)
    release_ms = epoch_ms(release_dt_utc) if release_dt_utc.tzinfo else epoch_ms(
        release_dt_utc.replace(tzinfo=UTC_TZ))

    out = {k: None for k in (
        "B_tm1_open", "B_tm1_close", "B_t_open", "B_t_pre",
        "B_t_release", "B_t_post_1h", "B_t_post_2h", "B_t_post_4h",
        "B_t_close", "B_tp1_open", "B_tp1_close",
    )}

    if bars.empty:
        return out

    # Compute key ms boundaries
    release_bar_start = floor_to_hour_ms(release_ms)

    # Anchor by-time bars
    by_time: dict[int, dict] = {}
    for _, b in bars.iterrows():
        by_time[int(b["time"])] = {
            "open": float(b["open"]), "high": float(b["high"]),
            "low": float(b["low"]), "close": float(b["close"]),
            "volume": float(b["volume"]) if pd.notna(b["volume"]) else 0.0,
        }

    # Find ET dates of T-1, T, T+1
    tm1 = event_date - timedelta(days=1)
    tp1 = event_date + timedelta(days=1)

    def _bars_on_et_date(target_date: date) -> list[int]:
        lo_ms, hi_ms = et_date_to_utc_ms_range(target_date)
        ts = [t for t in by_time if lo_ms <= t < hi_ms]
        return sorted(ts)

    tm1_bars = _bars_on_et_date(tm1)
    t_bars = _bars_on_et_date(event_date)
    tp1_bars = _bars_on_et_date(tp1)

    # Fallback for holiday/weekend T-1: walk back up to 4 days
    if not tm1_bars:
        for back in range(2, 6):
            tm1_alt = event_date - timedelta(days=back)
            tm1_bars = _bars_on_et_date(tm1_alt)
            if tm1_bars:
                break
    # Similar for T+1
    if not tp1_bars:
        for fwd in range(2, 6):
            tp1_alt = event_date + timedelta(days=fwd)
            tp1_bars = _bars_on_et_date(tp1_alt)
            if tp1_bars:
                break

    if tm1_bars:
        out["B_tm1_open"] = by_time[tm1_bars[0]]["close"]
        out["B_tm1_close"] = by_time[tm1_bars[-1]]["close"]
    if t_bars:
        out["B_t_open"] = by_time[t_bars[0]]["close"]
        out["B_t_close"] = by_time[t_bars[-1]]["close"]
    if tp1_bars:
        out["B_tp1_open"] = by_time[tp1_bars[0]]["close"]
        out["B_tp1_close"] = by_time[tp1_bars[-1]]["close"]

    # Pre-release: last 1H bar BEFORE release_bar_start (= release_bar_start - 1h)
    pre_ms = release_bar_start - 3600000
    if pre_ms in by_time:
        out["B_t_pre"] = by_time[pre_ms]["close"]

    # Release bar (the bar containing release_dt)
    if release_bar_start in by_time:
        out["B_t_release"] = by_time[release_bar_start]["close"]

    # Post-release bars (+1h, +2h, +4h from release_bar_start)
    for label, off in (("B_t_post_1h", 1), ("B_t_post_2h", 2),
                              ("B_t_post_4h", 4)):
        ms = release_bar_start + off * 3600000
        if ms in by_time:
            out[label] = by_time[ms]["close"]

    return out


# =============================================================================
# Instrument-panel builder (per event × instrument)
# =============================================================================

def _build_outright_series(con, symbol: str, event_date: date) -> tuple[pd.DataFrame, str]:
    """Wrapper around load_1h_bars_for_event for an outright. Returns
    (bars_df, data_source) where data_source = 'actual_contract'."""
    return load_1h_bars_for_event(con, symbol, event_date), "actual_contract"


def _normalize_bp_to_price_diff(bars: pd.DataFrame, symbol: str,
                                  units_catalog: Optional[pd.DataFrame]) -> pd.DataFrame:
    """If the contract is stored in bp convention (e.g. SRA spreads/flies =
    actual price-diff × 100), divide OHLC by 100 to convert back to price-diff
    units. This matches the scale of synthesized series (which are computed
    as price_leg1 - price_leg2 etc.), so the downstream measurement code
    `(close_end - close_start) * 100.0` produces correct bp deltas regardless
    of which source path produced the series.

    Per `lib/contract_units.py` autodetection: all SRA spreads (1666) and
    flies (1620) are stored bp; outrights are stored price.
    """
    if bars.empty:
        return bars
    try:
        if not is_already_bp(symbol, base_product="SRA", catalog=units_catalog):
            return bars
    except Exception:
        # Catalog unavailable — keep bars as-is (safer than mis-scaling)
        return bars
    out = bars.copy()
    for col in ("open", "high", "low", "close"):
        if col in out.columns:
            out[col] = out[col] / 100.0
    return out


def _construct_spread_symbol(sym_front: str, sym_back: str) -> str:
    """(SRAU26, SRAZ26) → 'SRAU26-Z26'. Strips 'SRA' from back leg."""
    if sym_back.startswith("SRA"):
        return f"{sym_front}-{sym_back[3:]}"
    return f"{sym_front}-{sym_back}"


def _construct_fly_symbol(syms: list[str]) -> str:
    """[SRAU26, SRAZ26, SRAH27] → 'SRAU26-Z26-H27'. Strips 'SRA' from non-front legs."""
    if not syms:
        return ""
    legs = [s[3:] if s.startswith("SRA") and i > 0 else s
              for i, s in enumerate(syms)]
    return "-".join(legs)


def _build_spread_series(con, sym_front: str, sym_back: str,
                                event_date: date,
                                units_catalog: Optional[pd.DataFrame] = None,
                                ) -> tuple[pd.DataFrame, str]:
    """Spread = front − back. Returns (bars, data_source).

    Phase: prefer actual spread contract bars when available (≥12 bars in
    the event window — covers T-1 through T+1). Fall back to synthesis
    from underlying outright legs.

    Scale: actual SRA spread contracts are stored in bp convention
    (close × 100 vs the implied price-diff). We normalize bp → price-diff
    so both paths return the same scale and downstream measurement code
    (× 100 to bp) stays correct.
    """
    # Try actual spread contract first
    spread_sym = _construct_spread_symbol(sym_front, sym_back)
    actual = load_1h_bars_for_event(con, spread_sym, event_date)
    if len(actual) >= 12:
        # Actual contract has sufficient bars in window — normalize to price-diff scale
        return _normalize_bp_to_price_diff(actual, spread_sym, units_catalog), "actual_contract"

    # Fallback: synthesize from underlying outright legs
    f_bars = load_1h_bars_for_event(con, sym_front, event_date)
    b_bars = load_1h_bars_for_event(con, sym_back, event_date)
    if f_bars.empty or b_bars.empty:
        return pd.DataFrame(), "missing"
    merged = f_bars[["time", "open", "high", "low", "close"]].merge(
        b_bars[["time", "open", "high", "low", "close"]],
        on="time", suffixes=("_f", "_b"))
    if merged.empty:
        return pd.DataFrame(), "missing"
    out = pd.DataFrame({"time": merged["time"]})
    out["open"] = merged["open_f"] - merged["open_b"]
    out["high"] = merged["high_f"] - merged["low_b"]
    out["low"] = merged["low_f"] - merged["high_b"]
    out["close"] = merged["close_f"] - merged["close_b"]
    out["volume"] = 0.0
    return out, "synthesized"


def _build_fly_series(con, syms: list[str],
                            event_date: date,
                            units_catalog: Optional[pd.DataFrame] = None,
                            ) -> tuple[pd.DataFrame, str]:
    """Fly = wing − 2·mid + wing. Returns (bars, data_source).

    Prefers actual fly contract bars when available; falls back to
    synthesis from three underlying outright legs.

    Scale: actual SRA fly contracts are stored in bp convention; we
    normalize bp → price-diff so the downstream × 100 measurement code
    produces correct bp deltas regardless of source.
    """
    if len(syms) != 3:
        return pd.DataFrame(), "missing"

    # Try actual fly contract first
    fly_sym = _construct_fly_symbol(syms)
    actual = load_1h_bars_for_event(con, fly_sym, event_date)
    if len(actual) >= 12:
        return _normalize_bp_to_price_diff(actual, fly_sym, units_catalog), "actual_contract"

    # Fallback: synthesize
    legs = [load_1h_bars_for_event(con, s, event_date) for s in syms]
    if any(b.empty for b in legs):
        return pd.DataFrame(), "missing"
    cols = ["time", "open", "high", "low", "close"]
    m1 = legs[0][cols].rename(columns={c: f"{c}_1" if c != "time" else c for c in cols})
    m2 = legs[1][cols].rename(columns={c: f"{c}_2" if c != "time" else c for c in cols})
    m3 = legs[2][cols].rename(columns={c: f"{c}_3" if c != "time" else c for c in cols})
    merged = m1.merge(m2, on="time").merge(m3, on="time")
    if merged.empty:
        return pd.DataFrame(), "missing"
    out = pd.DataFrame({"time": merged["time"]})
    out["close"] = merged["close_1"] - 2 * merged["close_2"] + merged["close_3"]
    out["open"] = merged["open_1"] - 2 * merged["open_2"] + merged["open_3"]
    out["high"] = merged["high_1"] - 2 * merged["low_2"] + merged["high_3"]
    out["low"] = merged["low_1"] - 2 * merged["high_2"] + merged["low_3"]
    out["volume"] = 0.0
    return out, "synthesized"


def _build_pack_series(con, pack_name: str, event_date: date,
                              contract_table: dict) -> tuple[pd.DataFrame, str]:
    """Pack = mean of 12 underlying contracts. No actual pack-contract
    exists for SRA — packs are inherently synthetic."""
    rng = PACK_RANGES[pack_name]
    syms = [lookup_contract(contract_table, event_date, m) for m in rng]
    syms = [s for s in syms if s]
    if len(syms) < 6:
        return pd.DataFrame(), "missing"
    legs = [load_1h_bars_for_event(con, s, event_date) for s in syms]
    legs = [b for b in legs if not b.empty]
    if len(legs) < 6:
        return pd.DataFrame(), "missing"
    cols = ["time", "open", "high", "low", "close"]
    base = legs[0][cols].rename(columns={c: f"{c}_0" if c != "time" else c for c in cols})
    for i, b in enumerate(legs[1:], start=1):
        base = base.merge(b[cols].rename(
            columns={c: f"{c}_{i}" if c != "time" else c for c in cols}), on="time")
    if base.empty:
        return pd.DataFrame(), "missing"
    n = len(legs)
    out = pd.DataFrame({"time": base["time"]})
    out["open"] = base[[f"open_{i}" for i in range(n)]].mean(axis=1)
    out["high"] = base[[f"high_{i}" for i in range(n)]].mean(axis=1)
    out["low"] = base[[f"low_{i}" for i in range(n)]].mean(axis=1)
    out["close"] = base[[f"close_{i}" for i in range(n)]].mean(axis=1)
    out["volume"] = 0.0
    return out, "synthesized"   # packs are always synthesized


def _build_pack_fly_series(con, packs: tuple[str, str, str],
                                  event_date: date,
                                  contract_table: dict) -> tuple[pd.DataFrame, str]:
    """Pack-fly = pack1 − 2·pack2 + pack3. Always synthesized (composed
    from pack averages which are themselves synthesized)."""
    p1, _ = _build_pack_series(con, packs[0], event_date, contract_table)
    p2, _ = _build_pack_series(con, packs[1], event_date, contract_table)
    p3, _ = _build_pack_series(con, packs[2], event_date, contract_table)
    if p1.empty or p2.empty or p3.empty:
        return pd.DataFrame(), "missing"
    cols = ["time", "open", "high", "low", "close"]
    m1 = p1[cols].rename(columns={c: f"{c}_1" if c != "time" else c for c in cols})
    m2 = p2[cols].rename(columns={c: f"{c}_2" if c != "time" else c for c in cols})
    m3 = p3[cols].rename(columns={c: f"{c}_3" if c != "time" else c for c in cols})
    merged = m1.merge(m2, on="time").merge(m3, on="time")
    if merged.empty:
        return pd.DataFrame(), "missing"
    out = pd.DataFrame({"time": merged["time"]})
    out["close"] = merged["close_1"] - 2 * merged["close_2"] + merged["close_3"]
    out["open"] = merged["open_1"] - 2 * merged["open_2"] + merged["open_3"]
    out["high"] = merged["high_1"] - 2 * merged["low_2"] + merged["high_3"]
    out["low"] = merged["low_1"] - 2 * merged["high_2"] + merged["low_3"]
    out["volume"] = 0.0
    return out, "synthesized"


def list_instruments() -> list[dict]:
    """Enumerate all 46 instruments."""
    items = []
    for m in OUTRIGHT_TENORS:
        items.append({"id": f"M{m}", "type": "outright", "tenors": (m,)})
    for a, b in SPREAD_PAIRS:
        items.append({"id": f"M{a}-M{b}", "type": "spread", "tenors": (a, b)})
    for a, b, c in FLY_TRIPLETS:
        items.append({"id": f"M{a}-M{b}-M{c}", "type": "fly", "tenors": (a, b, c)})
    for name in PACK_RANGES:
        items.append({"id": f"pack_{name}", "type": "pack", "pack_name": name})
    for p1, p2, p3 in PACK_FLY_TRIPLETS:
        items.append({"id": f"packfly_{p1}_{p2}_{p3}",
                          "type": "pack_fly", "packs": (p1, p2, p3)})
    return items


def get_instrument_series(con, instrument: dict, event_date: date,
                                contract_table: dict,
                                units_catalog: Optional[pd.DataFrame] = None,
                                ) -> tuple[pd.DataFrame, str]:
    """Build the 1H bar series for any instrument type.

    Returns (bars_df, data_source) where data_source ∈
    {'actual_contract', 'synthesized', 'missing'}.

    `units_catalog` is the lib.contract_units catalog used to normalize
    actual bp-stored spread/fly contracts back to price-diff units so
    they share a scale with the synthesized fallback. Pass None on
    call-paths that don't need the correction (catalog will be loaded
    lazily inside the per-symbol normalizer).
    """
    t = instrument["type"]
    if t == "outright":
        sym = lookup_contract(contract_table, event_date,
                                  instrument["tenors"][0])
        if sym is None:
            return pd.DataFrame(), "missing"
        return _build_outright_series(con, sym, event_date)
    if t == "spread":
        a, b = instrument["tenors"]
        sa = lookup_contract(contract_table, event_date, a)
        sb = lookup_contract(contract_table, event_date, b)
        if sa is None or sb is None:
            return pd.DataFrame(), "missing"
        return _build_spread_series(con, sa, sb, event_date, units_catalog)
    if t == "fly":
        a, b, c = instrument["tenors"]
        syms = [lookup_contract(contract_table, event_date, m) for m in (a, b, c)]
        if any(s is None for s in syms):
            return pd.DataFrame(), "missing"
        return _build_fly_series(con, syms, event_date, units_catalog)
    if t == "pack":
        return _build_pack_series(con, instrument["pack_name"], event_date,
                                          contract_table)
    if t == "pack_fly":
        return _build_pack_fly_series(con, instrument["packs"], event_date,
                                              contract_table)
    return pd.DataFrame(), "missing"


# =============================================================================
# Per-event measurement engine
# =============================================================================

def compute_event_instrument_row(con, event: pd.Series, instrument: dict,
                                          contract_table: dict,
                                          units_catalog: Optional[pd.DataFrame] = None,
                                          ) -> dict:
    """Compute all 11 measurements for one (event × instrument) pair.

    Returns dict with raw_bp + normalized columns + atr_1h_20 + flags.
    """
    out = {
        "ticker": event["ticker"],
        "release_date": event["release_date"],
        "state_8": event["state_8"],
        "instrument_id": instrument["id"],
        "instrument_type": instrument["type"],
        "atr_1h_20": float("nan"),
        "partial_session_t": False,
        "partial_session_tp1": False,
        "data_source": "missing",
    }
    # Add all measurement columns up front (NaN)
    for m in MEASUREMENTS:
        out[f"{m}_bp"] = float("nan")
        out[f"{m}_norm"] = float("nan")

    bars, data_source = get_instrument_series(
        con, instrument, event["release_date"], contract_table, units_catalog)
    out["data_source"] = data_source
    if bars.empty:
        return out

    anchors = find_anchor_bars(bars, event)
    # Compute ATR_1H_20 anchored at B_tm1_open's start_ms
    if anchors["B_tm1_open"] is not None:
        # Get the bar_start_ms for B_tm1_open from bars
        et_tm1 = event["release_date"] - timedelta(days=1)
        lo_ms, hi_ms = et_date_to_utc_ms_range(et_tm1)
        tm1_bars_filt = bars[(bars["time"] >= lo_ms) & (bars["time"] < hi_ms)]
        # Walk back if holiday
        for back in range(1, 6):
            if not tm1_bars_filt.empty:
                break
            alt_et = event["release_date"] - timedelta(days=back + 1)
            lo_ms, hi_ms = et_date_to_utc_ms_range(alt_et)
            tm1_bars_filt = bars[(bars["time"] >= lo_ms) & (bars["time"] < hi_ms)]
        if not tm1_bars_filt.empty:
            anchor_ms = int(tm1_bars_filt["time"].iloc[0])
            atr = compute_atr_1h_20(bars, anchor_ms)
            out["atr_1h_20"] = atr

    # Flag partial sessions
    et_t = event["release_date"]; et_tp1 = et_t + timedelta(days=1)
    lo_t, hi_t = et_date_to_utc_ms_range(et_t)
    lo_tp1, hi_tp1 = et_date_to_utc_ms_range(et_tp1)
    n_t_bars = int(((bars["time"] >= lo_t) & (bars["time"] < hi_t)).sum())
    n_tp1_bars = int(((bars["time"] >= lo_tp1) & (bars["time"] < hi_tp1)).sum())
    out["partial_session_t"] = n_t_bars < 12
    out["partial_session_tp1"] = n_tp1_bars < 12

    # Compute the 11 measurements (close-to-close, in raw price-bp * 100)
    def _diff(end_key: str, start_key: str) -> float:
        e = anchors.get(end_key); s = anchors.get(start_key)
        if e is None or s is None:
            return float("nan")
        return (float(e) - float(s)) * 100.0   # price diff in bp

    measurement_specs = (
        ("anticipation_full",      "B_t_pre",       "B_tm1_open"),
        ("anticipation_overnight", "B_t_open",      "B_tm1_close"),
        ("anticipation_morning",   "B_t_pre",       "B_t_open"),
        ("release_impulse",        "B_t_post_1h",   "B_t_pre"),
        ("release_immediate_2h",   "B_t_post_2h",   "B_t_pre"),
        ("release_short_4h",       "B_t_post_4h",   "B_t_pre"),
        ("release_day_end",        "B_t_close",     "B_t_pre"),
        ("next_day_open_gap",      "B_tp1_open",    "B_t_close"),
        ("next_day_full",          "B_tp1_close",   "B_t_close"),
        ("full_t_to_tp1",          "B_tp1_close",   "B_t_pre"),
        ("full_window",            "B_tp1_close",   "B_tm1_open"),
    )
    # Apply ATR floor (0.5 bp = 0.005 in price units) to prevent
    # normalization explosion on thin overnight ATR for deep-tenor contracts
    raw_atr = out["atr_1h_20"]
    if not np.isnan(raw_atr) and raw_atr > 0:
        atr_floored = max(raw_atr, ATR_FLOOR_PRICE)
        out["atr_1h_20_floored"] = atr_floored
        if atr_floored > raw_atr:
            out["atr_floor_applied"] = True
        else:
            out["atr_floor_applied"] = False
        atr_bp = atr_floored * 100.0
    else:
        out["atr_1h_20_floored"] = float("nan")
        out["atr_floor_applied"] = False
        atr_bp = float("nan")
    for name, end_k, start_k in measurement_specs:
        bp = _diff(end_k, start_k)
        out[f"{name}_bp"] = bp
        if np.isnan(bp) or np.isnan(atr_bp):
            out[f"{name}_norm"] = float("nan")
        else:
            out[f"{name}_norm"] = bp / atr_bp
    return out


def build_instrument_panel(catalog: pd.DataFrame,
                                contract_table: dict,
                                max_events: Optional[int] = None) -> pd.DataFrame:
    """Build the full event × instrument table (rows = #events × 46 instruments)."""
    if catalog.empty:
        return pd.DataFrame()
    instruments = list_instruments()
    rows = []
    # Load the unit-convention catalog once for the whole build — used to
    # re-scale actual bp-stored spread/fly contracts back to price-diff units.
    try:
        units_catalog = _load_units_catalog()
        if units_catalog is None or units_catalog.empty:
            units_catalog = None
    except Exception:
        units_catalog = None
    con = _open_ohlc_con()
    try:
        events_iter = catalog.itertuples(index=False)
        for i, ev_tuple in enumerate(events_iter):
            if max_events and i >= max_events:
                break
            event = pd.Series(ev_tuple._asdict())
            for instr in instruments:
                row = compute_event_instrument_row(con, event, instr,
                                                                contract_table,
                                                                units_catalog)
                rows.append(row)
    finally:
        con.close()
    return pd.DataFrame(rows)


# =============================================================================
# Aggregation per (ticker × instrument × state × measurement × window)
# =============================================================================

def aggregate_per_state(panel: pd.DataFrame, asof: date,
                              measurements: tuple = MEASUREMENTS) -> pd.DataFrame:
    """Build the per-cell aggregate table for all 4 windows."""
    if panel.empty:
        return pd.DataFrame()
    panel = panel.copy()
    panel["release_date"] = pd.to_datetime(panel["release_date"]).dt.date

    rows = []
    for window in WINDOWS:
        if window == "full":
            window_mask = pd.Series(True, index=panel.index)
        else:
            n_months = int(window.replace("m", ""))
            cutoff = asof - timedelta(days=n_months * 30)
            window_mask = pd.to_datetime(panel["release_date"]).dt.date >= cutoff
        sub = panel[window_mask]
        if sub.empty:
            continue
        for (ticker, instr_id, state), g in sub.groupby(
                ["ticker", "instrument_id", "state_8"], dropna=True, sort=False):
            for m in measurements:
                col_norm = f"{m}_norm"; col_bp = f"{m}_bp"
                if col_norm not in g.columns:
                    continue
                valid = g.dropna(subset=[col_norm])
                if valid.empty:
                    continue
                vals = valid[col_norm].astype(float).values
                bps = valid[col_bp].astype(float).values
                n = int(len(vals))
                rows.append({
                    "ticker": ticker, "instrument_id": instr_id,
                    "instrument_type": valid["instrument_type"].iloc[0],
                    "state_8": state, "measurement": m, "window": window,
                    "n_obs": n,
                    "median_norm": float(np.median(vals)),
                    "iqr_low": float(np.percentile(vals, 25)),
                    "iqr_high": float(np.percentile(vals, 75)),
                    "median_bp": float(np.median(bps)) if len(bps) else float("nan"),
                    "pct_above_1ATR": float(np.mean(np.abs(vals) > 1.0)),
                    "last_3_event_dates": ";".join(
                        valid.sort_values("release_date")["release_date"]
                            .astype(str).tolist()[-3:]),
                    "low_sample": n < LOW_SAMPLE_MIN_OBS,
                    "fallback_count_based": False,
                })
    return pd.DataFrame(rows)


# =============================================================================
# Drift detection (3m / 6m / 12m vs full)
# =============================================================================

def compute_drift_flags(agg: pd.DataFrame) -> pd.DataFrame:
    """Per (ticker × instrument × state × measurement): recency_index + tag."""
    if agg.empty:
        return pd.DataFrame()
    pivot = agg.pivot_table(
        index=["ticker", "instrument_id", "instrument_type", "state_8", "measurement"],
        columns="window",
        values=["median_norm", "iqr_low", "iqr_high", "n_obs"],
        aggfunc="first",
    )
    # Flatten multi-index columns
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    pivot = pivot.reset_index()

    out_rows = []
    for _, r in pivot.iterrows():
        full_median = r.get("median_norm_full")
        full_iqr_lo = r.get("iqr_low_full")
        full_iqr_hi = r.get("iqr_high_full")
        if pd.isna(full_median):
            continue
        iqr_full = (
            (full_iqr_hi - full_iqr_lo)
            if pd.notna(full_iqr_hi) and pd.notna(full_iqr_lo) else float("nan")
        )
        row = {
            "ticker": r["ticker"],
            "instrument_id": r["instrument_id"],
            "instrument_type": r["instrument_type"],
            "state_8": r["state_8"],
            "measurement": r["measurement"],
            "median_3m": r.get("median_norm_3m"),
            "median_6m": r.get("median_norm_6m"),
            "median_12m": r.get("median_norm_12m"),
            "median_full": full_median,
            "n_obs_3m": r.get("n_obs_3m"),
            "n_obs_6m": r.get("n_obs_6m"),
            "n_obs_12m": r.get("n_obs_12m"),
            "n_obs_full": r.get("n_obs_full"),
        }
        # Drift indices
        for w, full_pair in (("3m", "median_3m"),
                                  ("6m", "median_6m"),
                                  ("12m", "median_12m")):
            window_median = row.get(full_pair)
            if pd.isna(window_median) or pd.isna(iqr_full) or iqr_full <= 0:
                row[f"recency_index_{w}_vs_full"] = float("nan")
                continue
            row[f"recency_index_{w}_vs_full"] = float(
                (window_median - full_median) / iqr_full)
        # Tag (using 3m vs full)
        ri3 = row.get("recency_index_3m_vs_full")
        m3 = row.get("median_3m")
        tag = "STABLE"
        if pd.notna(ri3) and pd.notna(m3) and pd.notna(full_median):
            sign_3m = np.sign(m3); sign_full = np.sign(full_median)
            if sign_3m != 0 and sign_full != 0 and sign_3m != sign_full:
                tag = "REVERSING"
            elif abs(ri3) > DRIFT_GROWING_THRESHOLD:
                tag = "GROWING"
            elif abs(ri3) < DRIFT_FADING_THRESHOLD and abs(full_median) > 0.1:
                tag = "FADING"
        row["drift_tag"] = tag
        out_rows.append(row)
    return pd.DataFrame(out_rows)


# =============================================================================
# Persistence
# =============================================================================

def _paths(asof: date) -> dict:
    stamp = asof.isoformat()
    return {
        "catalog":      _CACHE_DIR / f"hei_event_catalog_{stamp}.parquet",
        "panel":        _CACHE_DIR / f"hei_instrument_panel_1h_{stamp}.parquet",
        "aggregates":   _CACHE_DIR / f"hei_aggregates_{stamp}.parquet",
        "drift":        _CACHE_DIR / f"hei_drift_flags_{stamp}.parquet",
        "manifest":     _CACHE_DIR / f"hei_manifest_{stamp}.json",
    }


def write_outputs(asof: date, catalog: pd.DataFrame, panel: pd.DataFrame,
                      aggregates: pd.DataFrame, drift: pd.DataFrame) -> dict:
    """Atomic write of all 5 outputs."""
    paths = _paths(asof)
    tmp = {k: v.with_suffix(v.suffix + ".tmp") for k, v in paths.items()}

    # Convert datetime columns to ISO strings for parquet compatibility
    if not catalog.empty:
        cat_out = catalog.copy()
        cat_out["release_date"] = cat_out["release_date"].astype(str)
        cat_out["release_datetime_utc"] = cat_out["release_datetime_utc"].apply(
            lambda d: d.isoformat() if hasattr(d, "isoformat") else str(d))
        cat_out.to_parquet(tmp["catalog"], index=False)
    if not panel.empty:
        p_out = panel.copy()
        if "release_date" in p_out.columns:
            p_out["release_date"] = p_out["release_date"].astype(str)
        p_out.to_parquet(tmp["panel"], index=False)
    if not aggregates.empty:
        aggregates.to_parquet(tmp["aggregates"], index=False)
    if not drift.empty:
        drift.to_parquet(tmp["drift"], index=False)

    # Data-source counts per instrument type
    data_source_counts = {}
    if not panel.empty and "data_source" in panel.columns:
        for inst_type, grp in panel.groupby("instrument_type"):
            data_source_counts[inst_type] = grp["data_source"].value_counts().to_dict()

    manifest = {
        "builder_version": BUILDER_VERSION,
        "asof_date": asof.isoformat(),
        "event_cutoff_date": EVENT_CUTOFF_DATE.isoformat(),
        "n_tickers_attempted": len(TICKERS_13),
        "n_tickers_built": int(catalog["ticker"].nunique()) if not catalog.empty else 0,
        "n_events_total": int(len(catalog)) if not catalog.empty else 0,
        "n_post_cutoff": int(catalog["is_post_cutoff"].sum()) if not catalog.empty else 0,
        "release_times_used": {t: str(rt) for t, rt in RELEASE_TIMES_ET.items()},
        "instruments_count": len(list_instruments()),
        "states_count": len(STATES_8),
        "windows_computed": list(WINDOWS),
        "drift_tag_counts": (drift["drift_tag"].value_counts().to_dict()
                                  if not drift.empty else {}),
        "data_source_counts_by_type": data_source_counts,
        "atr_lookback_hours": ATR_LOOKBACK_HOURS,
        "surprise_size_threshold_z": SURPRISE_SIZE_THRESHOLD_Z,
        "atr_floor_price": ATR_FLOOR_PRICE,
    }
    tmp["manifest"].write_text(json.dumps(manifest, indent=2, default=str))

    # Atomic rename
    for k in ("catalog", "panel", "aggregates", "drift", "manifest"):
        if tmp[k].exists():
            os.replace(tmp[k], paths[k])
    return manifest


# =============================================================================
# Top-level driver
# =============================================================================

def build_historical_event_impact(asof: Optional[date] = None) -> dict:
    """Top-level: build catalog → panel → aggregates → drift → persist."""
    if asof is None:
        asof = _resolve_latest_cmc_asof()
        if asof is None:
            asof = date.today()
    catalog = build_event_catalog(asof)
    if catalog.empty:
        raise RuntimeError("event catalog empty — check eco/ data + cutoff")
    contract_table = build_contract_lookup(asof)
    panel = build_instrument_panel(catalog, contract_table)
    if panel.empty:
        raise RuntimeError("instrument panel empty — check OHLC 1H data")
    aggregates = aggregate_per_state(panel, asof)
    drift = compute_drift_flags(aggregates)
    manifest = write_outputs(asof, catalog, panel, aggregates, drift)
    return manifest


# =============================================================================
# CLI
# =============================================================================

def main(argv=None):
    import sys
    args = list(sys.argv[1:] if argv is None else argv)
    asof = None
    if args:
        try:
            asof = date.fromisoformat(args[0])
        except ValueError:
            print("usage: python -m lib.historical_event_impact [YYYY-MM-DD]")
            return 2
    print(f"[hei] building (asof={asof or 'latest'})")
    m = build_historical_event_impact(asof)
    print(f"[hei] tickers built: {m['n_tickers_built']}/{m['n_tickers_attempted']}")
    print(f"[hei] events: {m['n_events_total']} (post-cutoff: {m['n_post_cutoff']})")
    print(f"[hei] instruments: {m['instruments_count']}")
    print(f"[hei] drift tags: {m['drift_tag_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
