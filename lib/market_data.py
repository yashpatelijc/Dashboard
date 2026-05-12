"""Generic market data layer — works for any base_product in lib/markets.py.

Mirrors `lib/sra_data.py`'s public API but accepts a `base_product` parameter
(SRA / ER / FSR / FER / SON / YBA / CRA). SRA continues to flow through the
original module for backwards-compat; other markets use this generic layer.

Reads the same `mde2_contracts_catalog` + `mde2_timeseries` tables — the
catalog convention is that `base_product` column holds the market code.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import pandas as pd
import streamlit as st

from lib.connections import get_ohlc_connection, get_bbg_inmemory_connection
from lib.markets import (MARKETS, QUARTERLY_MONTH_CODES, MONTH_CODE_TO_NUM,
                              parse_outright_symbol, get_market)


# Liveness threshold — match SRA convention
LIVENESS_DAYS = 7


# =============================================================================
# Snapshot date
# =============================================================================
@st.cache_data(show_spinner=False, ttl=600)
def get_snapshot_latest_date(base_product: str) -> Optional[date]:
    """Latest 1D bar date in OHLC DB for this market's outright/spread/fly contracts."""
    con = get_ohlc_connection()
    if con is None:
        return None
    row = con.execute(f"""
        SELECT MAX(DATE(to_timestamp(time/1000.0))) AS d
        FROM mde2_timeseries
        WHERE base_product=? AND "interval"='1D' AND calc_method='api' AND is_continuous=FALSE
    """, [base_product]).fetchone()
    return row[0] if row and row[0] else None


# =============================================================================
# Live symbol catalog
# =============================================================================
@st.cache_data(show_spinner=False, ttl=600)
def get_live_symbols(base_product: str, strategy: str,
                       tenor_months: Optional[int] = None) -> pd.DataFrame:
    """Live symbols for (base_product, strategy[, tenor_months])."""
    con = get_ohlc_connection()
    if con is None:
        return pd.DataFrame()

    latest = get_snapshot_latest_date(base_product)
    if latest is None:
        return pd.DataFrame()
    cutoff = latest - timedelta(days=LIVENESS_DAYS)

    where_tenor = "" if tenor_months is None else f"AND cc.tenor_months = {int(tenor_months)}"
    sql = f"""
    WITH last_bar AS (
        SELECT symbol, MAX(time)/1000 AS last_unix
        FROM mde2_timeseries
        WHERE "interval"='1D' AND calc_method='api'
        GROUP BY symbol
    )
    SELECT cc.symbol, cc.expiry_year, cc.expiry_month, cc.tenor_months,
           cc.contract_tag, cc.curve_bucket
    FROM mde2_contracts_catalog cc
    JOIN last_bar lb USING (symbol)
    WHERE cc.is_inter = FALSE
      AND cc.base_product = '{base_product}'
      AND cc.strategy = '{strategy}'
      {where_tenor}
      AND lb.last_unix >= EPOCH(DATE '{cutoff.isoformat()}')
    ORDER BY cc.expiry_year, cc.expiry_month, cc.symbol
    """
    return con.execute(sql).fetchdf()


def get_outrights(base_product: str) -> pd.DataFrame:
    return get_live_symbols(base_product, "outright")


def get_spreads(base_product: str, tenor_months: int) -> pd.DataFrame:
    return get_live_symbols(base_product, "spread", tenor_months)


def get_flies(base_product: str, tenor_months: int) -> pd.DataFrame:
    return get_live_symbols(base_product, "fly", tenor_months)


@st.cache_data(show_spinner=False, ttl=600)
def get_available_tenors(base_product: str, strategy: str) -> list[int]:
    """Distinct tenor_months for live contracts of given strategy."""
    con = get_ohlc_connection()
    if con is None:
        return []
    latest = get_snapshot_latest_date(base_product)
    if latest is None:
        return []
    cutoff = latest - timedelta(days=LIVENESS_DAYS)
    sql = f"""
    WITH last_bar AS (
        SELECT symbol, MAX(time)/1000 AS last_unix
        FROM mde2_timeseries WHERE "interval"='1D' AND calc_method='api' GROUP BY symbol
    )
    SELECT DISTINCT cc.tenor_months
    FROM mde2_contracts_catalog cc
    JOIN last_bar lb USING (symbol)
    WHERE cc.is_inter = FALSE
      AND cc.base_product = '{base_product}'
      AND cc.strategy = '{strategy}'
      AND cc.tenor_months IS NOT NULL
      AND lb.last_unix >= EPOCH(DATE '{cutoff.isoformat()}')
    ORDER BY cc.tenor_months
    """
    rows = con.execute(sql).fetchall()
    return [int(r[0]) for r in rows]


# =============================================================================
# OHLC panel loading
# =============================================================================
@st.cache_data(show_spinner=False, ttl=600)
def load_curve_panel(base_product: str, strategy: str,
                       tenor_months: Optional[int],
                       start_date: date, end_date: date) -> pd.DataFrame:
    """Pull all daily closes for live symbols of given strategy/tenor, between two dates."""
    syms = get_live_symbols(base_product, strategy, tenor_months)
    if syms.empty:
        return pd.DataFrame()
    sym_list = ",".join(f"'{s}'" for s in syms["symbol"])
    con = get_ohlc_connection()
    sql = f"""
    SELECT DATE(to_timestamp(t.time/1000.0)) AS bar_date,
           t.symbol, cc.expiry_year, cc.expiry_month,
           t.open, t.high, t.low, t.close, t.volume
    FROM mde2_timeseries t
    JOIN mde2_contracts_catalog cc USING (symbol)
    WHERE t.symbol IN ({sym_list})
      AND t."interval" = '1D'
      AND t.calc_method = 'api'
      AND DATE(to_timestamp(t.time/1000.0))
          BETWEEN DATE '{start_date.isoformat()}' AND DATE '{end_date.isoformat()}'
    ORDER BY bar_date, cc.expiry_year, cc.expiry_month
    """
    df = con.execute(sql).fetchdf()
    return df.rename(columns={"bar_date": "asof"})


def pivot_curve_panel(panel: pd.DataFrame, contract_order: list,
                        field: str = "close") -> pd.DataFrame:
    """Wide-format pivot — same as sra_data.pivot_curve_panel."""
    if panel.empty:
        return pd.DataFrame()
    wide = panel.pivot_table(index="asof", columns="symbol", values=field, aggfunc="first")
    wide = wide.reindex(columns=contract_order)
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


# =============================================================================
# Listed spread/fly panel
# =============================================================================
def load_listed_spread_fly_panel(base_product: str, start_date: date,
                                   end_date: date, *,
                                   max_tenor_months: int = 12) -> dict:
    """Load CLOSE panels for all live listed spreads + flies in the date range."""
    out = {"spread_close": pd.DataFrame(), "fly_close": pd.DataFrame(),
            "spread_catalog": pd.DataFrame(), "fly_catalog": pd.DataFrame()}
    for strat in ("spread", "fly"):
        catalog_rows = []
        for tenor in range(1, max_tenor_months + 1):
            syms = get_live_symbols(base_product, strat, tenor)
            if not syms.empty:
                catalog_rows.append(syms)
        if not catalog_rows:
            continue
        catalog = pd.concat(catalog_rows, ignore_index=True).drop_duplicates(subset=["symbol"])
        out[f"{strat}_catalog"] = catalog
        if catalog.empty:
            continue
        sym_list = ",".join(f"'{s}'" for s in catalog["symbol"])
        con = get_ohlc_connection()
        sql = f"""
        SELECT DATE(to_timestamp(t.time/1000.0)) AS bar_date,
               t.symbol, t.close
        FROM mde2_timeseries t
        WHERE t.symbol IN ({sym_list})
          AND t."interval" = '1D'
          AND t.calc_method = 'api'
          AND DATE(to_timestamp(t.time/1000.0))
              BETWEEN DATE '{start_date.isoformat()}' AND DATE '{end_date.isoformat()}'
        """
        df = con.execute(sql).fetchdf()
        if df.empty:
            continue
        wide = df.pivot_table(index="bar_date", columns="symbol", values="close",
                                aggfunc="first")
        wide.index = pd.to_datetime(wide.index)
        wide = wide.sort_index()
        out[f"{strat}_close"] = wide
    return out


# =============================================================================
# Reference rate panel — market-specific BBG tickers
# =============================================================================
@st.cache_data(show_spinner=False, ttl=3600)
def load_reference_rate_panel(base_product: str, start_date: date,
                                end_date: date) -> pd.DataFrame:
    """Load central-bank policy rates + overnight reference rate for this market.

    Returns DataFrame indexed by date with columns:
      - rate_upper:  upper bound of policy corridor (or main rate)
      - rate_lower:  lower bound (or same as upper for single-rate banks)
      - overnight:   the overnight market rate (SOFR/€STR/SONIA/CORRA/SARON/cash-rate)
    """
    import os
    from lib.connections import BBG_PARQUET_ROOT
    cfg = get_market(base_product)
    upper_ticker = cfg.get("reference_rate_upper")
    lower_ticker = cfg.get("reference_rate_lower")
    overnight_ticker = cfg.get("reference_rate_ticker")
    lookup_dir = cfg.get("reference_rate_lookup_dir", "rates_drivers")

    con = get_bbg_inmemory_connection()
    series = {}
    candidates = [
        ("rate_upper", upper_ticker),
        ("rate_lower", lower_ticker),
        ("overnight", overnight_ticker),
    ]
    for label, ticker in candidates:
        if not ticker:
            continue
        # Try the configured directory first, then a couple of common fallbacks
        candidate_paths = [
            os.path.join(BBG_PARQUET_ROOT, lookup_dir, f"{ticker}.parquet"),
            os.path.join(BBG_PARQUET_ROOT, "eco", f"{ticker}.parquet"),
            os.path.join(BBG_PARQUET_ROOT, "rates_drivers", f"{ticker}.parquet"),
        ]
        path = next((p for p in candidate_paths if __import__("os").path.exists(p)), None)
        if path is None:
            continue
        try:
            duck_path = path.replace("\\", "/")
            df = con.execute(
                f"SELECT date, PX_LAST FROM read_parquet('{duck_path}')"
            ).fetchdf()
            df["date"] = pd.to_datetime(df["date"])
            df = df.dropna(subset=["PX_LAST"]).set_index("date").sort_index()
            series[label] = df["PX_LAST"]
        except Exception:
            continue

    if not series:
        return pd.DataFrame()

    out = pd.concat(series, axis=1)
    full_idx = pd.date_range(start=pd.Timestamp(start_date),
                                end=pd.Timestamp(end_date), freq="B")
    out = out.reindex(out.index.union(full_idx)).sort_index().ffill()
    out = out.loc[(out.index >= pd.Timestamp(start_date))
                    & (out.index <= pd.Timestamp(end_date))]
    return out


def get_reference_rates_at(base_product: str, asof_date: date,
                             ref_panel: pd.DataFrame) -> dict:
    """Return {rate_upper, rate_lower, rate_mid, overnight} at a given date."""
    out = {"rate_upper": None, "rate_lower": None, "rate_mid": None, "overnight": None}
    if ref_panel is None or ref_panel.empty:
        return out
    ts = pd.Timestamp(asof_date)
    idx = ref_panel.index[ref_panel.index <= ts]
    if len(idx) == 0:
        return out
    row = ref_panel.loc[idx[-1]]
    for col in out.keys():
        if col == "rate_mid":
            continue
        if col in ref_panel.columns:
            v = row.get(col)
            out[col] = float(v) if pd.notna(v) else None
    if out["rate_upper"] is not None and out["rate_lower"] is not None:
        out["rate_mid"] = (out["rate_upper"] + out["rate_lower"]) / 2
    return out


# =============================================================================
# Pack groupings — generalized
# =============================================================================
def get_quarterly_outright_indices(symbols_df: pd.DataFrame,
                                       base_product: str) -> list:
    """Return positional indices of quarterly outrights (H/M/U/Z) for this base product.

    Symbol format: '{base_product}{month_code}{year_2d}' → char index len(base_product)
    is the month code.
    """
    if symbols_df.empty:
        return []
    prefix_len = len(base_product)
    indices = []
    for i, sym in enumerate(symbols_df["symbol"].tolist()):
        if (len(sym) >= prefix_len + 3
                and sym[prefix_len].upper() in QUARTERLY_MONTH_CODES):
            indices.append(i)
    return indices


def compute_pack_groups(symbols_df: pd.DataFrame, base_product: str):
    """Group quarterlies into packs of 4. Pack names from MARKETS[base_product]."""
    qidx = get_quarterly_outright_indices(symbols_df, base_product)
    if not qidx:
        return []
    quarterly_syms = symbols_df.iloc[qidx]["symbol"].tolist()
    cfg = get_market(base_product)
    pack_names = cfg["pack_names"]
    out = []
    for i in range(0, len(quarterly_syms), 4):
        chunk = quarterly_syms[i:i + 4]
        if not chunk:
            break
        pack_idx = i // 4
        name = pack_names[pack_idx] if pack_idx < len(pack_names) else f"Pack{pack_idx + 1}"
        out.append((name, chunk))
    return out


# =============================================================================
# Symbol parsing — generalized
# =============================================================================
def parse_outright_year_month(symbol: str, base_product: str) -> tuple:
    """Same API as lib.fomc.parse_sra_outright but for any market."""
    return parse_outright_symbol(symbol, base_product)


# =============================================================================
# Spread/Fly symbol derivation — generalized
# =============================================================================
def _outright_sort_key(sym: str) -> tuple:
    """Sort key for outrights: (year, month_num) extracted from the trailing 3 chars."""
    if len(sym) < 3:
        return (9999, 99)
    month_code = sym[-3].upper()
    try:
        year_2d = int(sym[-2:])
    except ValueError:
        return (9999, 99)
    year_4d = 2000 + year_2d if year_2d < 50 else 1900 + year_2d
    return (year_4d, MONTH_CODE_TO_NUM.get(month_code, 99))


def derive_listed_spread_symbol(leg_symbols: list,
                                   base_product: str) -> Optional[str]:
    """Derive listed CME-style spread symbol from 2 outright legs.

    e.g. ['ERH26', 'ERM26'] → 'ERH26-M26' (front_full + '-' + back_short).
    """
    if len(leg_symbols) != 2:
        return None
    syms = sorted(leg_symbols, key=_outright_sort_key)
    if not all(s.startswith(base_product) for s in syms):
        return None
    back = syms[1][len(base_product):]
    return f"{syms[0]}-{back}"


def derive_listed_fly_symbol(leg_symbols: list,
                                base_product: str) -> Optional[str]:
    """Derive listed butterfly symbol from 3 outright legs."""
    if len(leg_symbols) != 3:
        return None
    syms = sorted(leg_symbols, key=_outright_sort_key)
    if not all(s.startswith(base_product) for s in syms):
        return None
    mid = syms[1][len(base_product):]
    back = syms[2][len(base_product):]
    return f"{syms[0]}-{mid}-{back}"


# =============================================================================
# Curve sections (front / mid / back) — uses MARKETS config
# =============================================================================
def compute_section_split(n_contracts: int, base_product: str = "SRA"):
    """Return (front_indices, mid_indices, back_indices). Section bounds from market config."""
    cfg = get_market(base_product)
    front_end = int(cfg.get("default_front_end", 6))
    mid_end = int(cfg.get("default_mid_end", 14))
    if n_contracts <= 0:
        return range(0), range(0), range(0)
    fe = min(front_end, n_contracts)
    me = min(mid_end, n_contracts)
    if me <= fe:
        me = min(fe + 1, n_contracts)
    return range(0, fe), range(fe, me), range(me, n_contracts)


def section_label_for_index(idx: int, base_product: str = "SRA") -> str:
    cfg = get_market(base_product)
    front_end = int(cfg.get("default_front_end", 6))
    mid_end = int(cfg.get("default_mid_end", 14))
    if idx < front_end:
        return "front"
    if idx < mid_end:
        return "mid"
    return "back"
