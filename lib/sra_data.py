"""SRA-specific data layer (queries against OHLC DB)."""
from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Optional

import duckdb
import numpy as np
import pandas as pd
import streamlit as st

from lib.connections import get_bbg_inmemory_connection, get_ohlc_connection


# Liveness threshold: contract is "live" if its last 1D bar is within this many days of latest snapshot.
LIVENESS_DAYS = 7


@st.cache_data(show_spinner=False, ttl=600)
def get_sra_snapshot_latest_date() -> Optional[date]:
    """Latest 1D bar date in OHLC DB for SRA outright/spread/fly contracts (excludes continuous)."""
    con = get_ohlc_connection()
    if con is None:
        return None
    row = con.execute("""
        SELECT MAX(DATE(to_timestamp(time/1000.0))) AS d
        FROM mde2_timeseries
        WHERE base_product='SRA' AND "interval"='1D' AND calc_method='api' AND is_continuous=FALSE
    """).fetchone()
    return row[0] if row and row[0] else None


@st.cache_data(show_spinner=False, ttl=600)
def get_sra_live_symbols(strategy: str, tenor_months: Optional[int] = None) -> pd.DataFrame:
    """Live SRA symbols of given strategy/tenor, ordered by left-leg expiry.

    Live = last 1D bar within LIVENESS_DAYS of latest snapshot.
    """
    con = get_ohlc_connection()
    if con is None:
        return pd.DataFrame()

    latest = get_sra_snapshot_latest_date()
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
      AND cc.base_product = 'SRA'
      AND cc.strategy = '{strategy}'
      {where_tenor}
      AND lb.last_unix >= EPOCH(DATE '{cutoff.isoformat()}')
    ORDER BY cc.expiry_year, cc.expiry_month, cc.symbol
    """
    return con.execute(sql).fetchdf()


@st.cache_data(show_spinner=False, ttl=600)
def load_sra_curve_panel(
    strategy: str,
    tenor_months: Optional[int],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Pull all daily closes for live SRA symbols of given strategy/tenor, between two dates.

    Returns long-format DataFrame: asof (date), symbol, expiry_year, expiry_month, close.
    """
    syms = get_sra_live_symbols(strategy, tenor_months)
    if syms.empty:
        return pd.DataFrame()

    sym_list = ",".join(f"'{s}'" for s in syms["symbol"])
    con = get_ohlc_connection()
    # NOTE: 'asof' is a DuckDB reserved word â€” use bar_date alias instead.
    sql = f"""
    SELECT DATE(to_timestamp(t.time/1000.0)) AS bar_date,
           t.symbol, cc.expiry_year, cc.expiry_month,
           t.open, t.high, t.low, t.close, t.volume
    FROM mde2_timeseries t
    JOIN mde2_contracts_catalog cc USING (symbol)
    WHERE t.symbol IN ({sym_list})
      AND t."interval" = '1D'
      AND t.calc_method = 'api'
      AND DATE(to_timestamp(t.time/1000.0)) BETWEEN DATE '{start_date.isoformat()}' AND DATE '{end_date.isoformat()}'
    ORDER BY bar_date, cc.expiry_year, cc.expiry_month
    """
    df = con.execute(sql).fetchdf()
    return df.rename(columns={"bar_date": "asof"})


def pivot_curve_panel(panel: pd.DataFrame, contract_order: list[str], field: str = "close") -> pd.DataFrame:
    """Wide-format pivot: index = asof, columns = symbol (in given order), values = `field`.

    `field` may be any column from load_sra_curve_panel (open / high / low / close / volume).
    Missing (date, symbol) cells become NaN.
    """
    if panel.empty:
        return pd.DataFrame()
    wide = panel.pivot_table(index="asof", columns="symbol", values=field, aggfunc="first")
    wide = wide.reindex(columns=contract_order)
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


# Convenience wrappers for common strategies
def get_outrights() -> pd.DataFrame:
    return get_sra_live_symbols("outright")


def get_spreads(tenor_months: int) -> pd.DataFrame:
    return get_sra_live_symbols("spread", tenor_months)


def get_flies(tenor_months: int) -> pd.DataFrame:
    return get_sra_live_symbols("fly", tenor_months)


# Available tenors for spreads / flies, derived from catalog (cached separately)
def contract_range_str(symbols_df: pd.DataFrame) -> str:
    """e.g. 'SRAJ26 â†’ SRAH32'."""
    if symbols_df.empty:
        return "â€”"
    syms = symbols_df["symbol"].tolist()
    return f"{syms[0]} â†’ {syms[-1]}"


def tenor_breakdown(strategy: str) -> list[tuple[int, int, str]]:
    """Return [(tenor, count, range_str), ...] for given strategy."""
    out = []
    for t in get_available_tenors(strategy):
        df = get_sra_live_symbols(strategy, t)
        out.append((t, len(df), contract_range_str(df)))
    return out


@st.cache_data(show_spinner=False, ttl=600)
def get_available_tenors(strategy: str) -> list[int]:
    """Distinct tenor_months for live SRA contracts of given strategy."""
    con = get_ohlc_connection()
    if con is None:
        return []
    latest = get_sra_snapshot_latest_date()
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
    WHERE cc.is_inter = FALSE AND cc.base_product = 'SRA'
      AND cc.strategy = '{strategy}'
      AND cc.tenor_months IS NOT NULL
      AND lb.last_unix >= EPOCH(DATE '{cutoff.isoformat()}')
    ORDER BY cc.tenor_months
    """
    rows = con.execute(sql).fetchall()
    return [int(r[0]) for r in rows]


# =============================================================================
# Reference rates (FDTR / SOFR) â€” for overlays
# =============================================================================

_BBG_RATES_DRIVERS = r"D:\BBG data\parquet\rates_drivers"
_BBG_ECO = r"D:\BBG data\parquet\eco"


@st.cache_data(show_spinner=False, ttl=3600)
def load_reference_rate_panel(start_date: date, end_date: date) -> pd.DataFrame:
    """Load FDTR upper, FDTR lower, SOFR rate over date range.

    Returns DataFrame indexed by date with columns: fdtr_upper, fdtr_lower, sofr.
    Forward-filled to cover all business days in range.
    """
    files = {
        "fdtr_upper": (os.path.join(_BBG_RATES_DRIVERS, "FDTR_Index.parquet"), "PX_LAST"),
        "fdtr_lower": (os.path.join(_BBG_ECO, "FDTRFTRL_Index.parquet"), "PX_LAST"),
        "sofr": (os.path.join(_BBG_RATES_DRIVERS, "SOFRRATE_Index.parquet"), "PX_LAST"),
    }
    con = get_bbg_inmemory_connection()
    series = {}
    for label, (path, valcol) in files.items():
        if not os.path.exists(path):
            continue
        try:
            duck_path = path.replace("\\", "/")
            df = con.execute(
                f"SELECT date, {valcol} FROM read_parquet('{duck_path}')"
            ).fetchdf()
            df["date"] = pd.to_datetime(df["date"])
            df = df.dropna(subset=[valcol]).set_index("date").sort_index()
            series[label] = df[valcol]
        except Exception:
            continue

    if not series:
        return pd.DataFrame()

    out = pd.concat(series, axis=1)
    full_idx = pd.date_range(start=pd.Timestamp(start_date), end=pd.Timestamp(end_date), freq="B")
    out = out.reindex(out.index.union(full_idx)).sort_index().ffill()
    out = out.loc[(out.index >= pd.Timestamp(start_date)) & (out.index <= pd.Timestamp(end_date))]
    return out


def get_reference_rates_at(asof_date: date, ref_panel: pd.DataFrame) -> dict:
    """Return {fdtr_upper, fdtr_lower, fdtr_mid, sofr} at a given date (forward-fill)."""
    out = {"fdtr_upper": None, "fdtr_lower": None, "fdtr_mid": None, "sofr": None}
    if ref_panel is None or ref_panel.empty:
        return out
    ts = pd.Timestamp(asof_date)
    idx = ref_panel.index[ref_panel.index <= ts]
    if len(idx) == 0:
        return out
    row = ref_panel.loc[idx[-1]]
    upper = row.get("fdtr_upper") if "fdtr_upper" in ref_panel.columns else None
    lower = row.get("fdtr_lower") if "fdtr_lower" in ref_panel.columns else None
    sofr = row.get("sofr") if "sofr" in ref_panel.columns else None
    upper = float(upper) if pd.notna(upper) else None
    lower = float(lower) if pd.notna(lower) else None
    sofr = float(sofr) if pd.notna(sofr) else None
    mid = (upper + lower) / 2 if (upper is not None and lower is not None) else None
    return {"fdtr_upper": upper, "fdtr_lower": lower, "fdtr_mid": mid, "sofr": sofr}


# =============================================================================
# Curve sections (front / mid / back)
# =============================================================================

DEFAULT_FRONT_END = 6
DEFAULT_MID_END = 14


def compute_section_split(n_contracts: int, front_end: int = DEFAULT_FRONT_END,
                          mid_end: int = DEFAULT_MID_END):
    """Return (front_indices, mid_indices, back_indices) given contract count.

    Adapts when n_contracts < mid_end so each section gets at least 1.
    """
    if n_contracts <= 0:
        return range(0), range(0), range(0)
    fe = min(front_end, n_contracts)
    me = min(mid_end, n_contracts)
    if me <= fe:
        me = min(fe + 1, n_contracts)
    return range(0, fe), range(fe, me), range(me, n_contracts)


def section_label_for_index(idx: int, front_end: int = DEFAULT_FRONT_END,
                            mid_end: int = DEFAULT_MID_END) -> str:
    if idx < front_end:
        return "front"
    if idx < mid_end:
        return "mid"
    return "back"


# =============================================================================
# Pack groupings (Whites / Reds / Greens / Blues / Golds / ...)
# =============================================================================

PACK_NAMES = ["Whites", "Reds", "Greens", "Blues", "Golds", "Purples", "Browns"]
PACK_COLORS = ["#f0f4fa", "#f87171", "#4ade80", "#60a5fa", "#e8b75d", "#a78bfa", "#a16207"]
QUARTERLY_MONTH_CODES = {"H", "M", "U", "Z"}


def get_quarterly_outright_indices(symbols_df: pd.DataFrame) -> list:
    """Return positional indices of quarterly outrights (H/M/U/Z) within symbols_df.

    SRA outright symbols look like 'SRAH26' â€” char index 3 is the month code.
    """
    if symbols_df.empty:
        return []
    indices = []
    for i, sym in enumerate(symbols_df["symbol"].tolist()):
        if len(sym) >= 6 and sym[3].upper() in QUARTERLY_MONTH_CODES:
            indices.append(i)
    return indices


def compute_pack_groups(symbols_df: pd.DataFrame):
    """Return [(pack_name, [symbols]), ...] grouping quarterlies into packs of 4."""
    qidx = get_quarterly_outright_indices(symbols_df)
    if not qidx:
        return []
    quarterly_syms = symbols_df.iloc[qidx]["symbol"].tolist()
    out = []
    for i in range(0, len(quarterly_syms), 4):
        chunk = quarterly_syms[i:i + 4]
        if not chunk:
            break
        pack_idx = i // 4
        name = PACK_NAMES[pack_idx] if pack_idx < len(PACK_NAMES) else f"Pack{pack_idx + 1}"
        out.append((name, chunk))
    return out


# =============================================================================
# Curve change & decomposition
# =============================================================================

def compute_curve_change(close_today: list, close_compare: list,
                         in_bp_units: bool = True,
                         contracts: Optional[list[str]] = None,
                         base_product: str = "SRA") -> list:
    """Per-contract change (today âˆ’ compare).

    When `contracts` is given, the per-contract unit convention is looked up in
    the contract-units catalog so that already-bp spreads/flies are NOT
    re-multiplied by 100. When `contracts` is None or the catalog is empty,
    falls back to the global `in_bp_units` switch (Ã—100 for all if True).
    """
    out = []
    if contracts is not None:
        from lib.contract_units import bp_multipliers_for, load_catalog
        cat = load_catalog()
        if not cat.empty:
            mults = bp_multipliers_for(contracts, base_product, cat) if in_bp_units \
                    else [1.0] * len(contracts)
            for t, c, m in zip(close_today, close_compare, mults):
                if t is None or c is None or pd.isna(t) or pd.isna(c):
                    out.append(None)
                else:
                    out.append((t - c) * m)
            return out
    # Fallback (no catalog or no contract list)
    mult = 100 if in_bp_units else 1
    for t, c in zip(close_today, close_compare):
        if t is None or c is None or pd.isna(t) or pd.isna(c):
            out.append(None)
        else:
            out.append((t - c) * mult)
    return out


def compute_decomposition(changes: list) -> dict:
    """Decompose curve change into parallel + slope + curvature components.

    Regress changes against [1, x, x^2] where x is normalized contract index.
    Returns dict with keys parallel / slope / curvature / residual_rmse.
    """
    valid = [(i, c) for i, c in enumerate(changes) if c is not None and not pd.isna(c)]
    if len(valid) < 3:
        return {"parallel": None, "slope": None, "curvature": None, "residual_rmse": None}

    indices = np.array([i for i, _ in valid], dtype=float)
    vals = np.array([v for _, v in valid], dtype=float)
    n = len(vals)
    denom = max(1.0, indices.max() - indices.mean())
    x = (indices - indices.mean()) / denom

    X = np.column_stack([np.ones(n), x, x * x])
    try:
        beta, *_ = np.linalg.lstsq(X, vals, rcond=None)
    except Exception:
        return {"parallel": None, "slope": None, "curvature": None, "residual_rmse": None}

    parallel, slope, curvature = float(beta[0]), float(beta[1]), float(beta[2])
    residuals = vals - X @ beta
    rmse = float(np.sqrt((residuals ** 2).mean()))
    return {"parallel": parallel, "slope": slope, "curvature": curvature, "residual_rmse": rmse}


# =============================================================================
# Per-contract z-scores and percentile bands (for Ribbon mode)
# =============================================================================

def compute_per_contract_zscores(wide_close: pd.DataFrame, asof_date: date,
                                 lookback: int = 252) -> dict:
    """Z-score of as-of-date close vs the prior `lookback` trading days, per contract.

    The history window EXCLUDES the as-of-date itself â€” we compare today against
    the prior N days (no look-ahead bias). This matters for short lookbacks (5d, 15d)
    where including today would contaminate the sample.
    """
    out = {}
    if wide_close.empty:
        return out
    ts = pd.Timestamp(asof_date)
    history = wide_close.loc[wide_close.index < ts].tail(lookback)
    if history.empty:
        return out
    try:
        today_row = wide_close.loc[wide_close.index.date == asof_date].iloc[0]
    except Exception:
        return out
    # Use ddof=1 sample std; needs at least 2 observations
    means = history.mean()
    stds = history.std(ddof=1)
    for col in wide_close.columns:
        v = today_row.get(col)
        if v is None or pd.isna(v):
            out[col] = None
            continue
        m, s = means.get(col), stds.get(col)
        # Need a real value, real std, std > 0, AND at least 2 valid prior observations
        valid_count = history[col].dropna().shape[0] if col in history.columns else 0
        if pd.isna(m) or pd.isna(s) or s == 0 or valid_count < 2:
            out[col] = None
        else:
            out[col] = float((v - m) / s)
    return out


def compute_percentile_bands(wide_close: pd.DataFrame, asof_date: date,
                             lookback: int = 252) -> dict:
    """Per-contract percentile bands {p05, p25, p50, p75, p95, mean} from PRIOR window.

    Excludes the as-of-date so today's value isn't in the band sample.
    """
    if wide_close.empty:
        return {}
    ts = pd.Timestamp(asof_date)
    history = wide_close.loc[wide_close.index < ts].tail(lookback)
    if history.empty:
        return {}
    # quantile uses linear interpolation by default â€” fine for n>=5; for n<5 it
    # collapses toward min/max which is the natural behavior.
    return {
        "p05": history.quantile(0.05).to_dict(),
        "p25": history.quantile(0.25).to_dict(),
        "p50": history.quantile(0.50).to_dict(),
        "p75": history.quantile(0.75).to_dict(),
        "p95": history.quantile(0.95).to_dict(),
        "mean": history.mean().to_dict(),
    }


def get_contract_full_panel(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Pull OHLCV + every indicator column from `v_mde2_timeseries_with_indicators`
    for a single contract. Returns a DataFrame indexed by date.

    Used by the Technicals subtab â€” every detector needs OHLCV + indicators
    in one place.
    """
    con = get_ohlc_connection()
    if con is None:
        return pd.DataFrame()
    sql = f"""
    SELECT *
    FROM v_mde2_timeseries_with_indicators t
    WHERE t.symbol = '{symbol}'
      AND t."interval" = '1D' AND t.calc_method = 'api'
      AND DATE(to_timestamp(t.time/1000.0)) BETWEEN DATE '{start_date.isoformat()}' AND DATE '{end_date.isoformat()}'
    ORDER BY t.time
    """
    try:
        df = con.execute(sql).fetchdf()
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["time"], unit="ms")
        df = df.set_index("date").sort_index()
        return df
    except Exception:
        return pd.DataFrame()


def get_contract_history(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Pull full daily OHLC + volume history for a single contract."""
    con = get_ohlc_connection()
    if con is None:
        return pd.DataFrame()
    sql = f"""
    SELECT DATE(to_timestamp(t.time/1000.0)) AS bar_date,
           t.open, t.high, t.low, t.close, t.volume
    FROM mde2_timeseries t
    WHERE t.symbol = '{symbol}'
      AND t."interval" = '1D' AND t.calc_method = 'api'
      AND DATE(to_timestamp(t.time/1000.0)) BETWEEN DATE '{start_date.isoformat()}' AND DATE '{end_date.isoformat()}'
    ORDER BY bar_date
    """
    try:
        df = con.execute(sql).fetchdf()
        df = df.rename(columns={"bar_date": "date"})
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception:
        return pd.DataFrame()


def compute_pairwise_spread_matrix(symbols_df: pd.DataFrame,
                                    wide_close: pd.DataFrame,
                                    asof_date: date,
                                    value_mode: str = "value",
                                    lookback: int = 60) -> pd.DataFrame:
    """For SPREADS or FLIES, build a 2D matrix indexed by left-leg expiry vs tenor.

    Returns DataFrame: rows = left-leg symbol (the front contract of the spread),
    cols = tenor_months (3, 6, 9, 12 for spreads; 3, 6, 9, 12 for flies).
    Values:
        value_mode="value"      â†’ today's price
        value_mode="zscore"     â†’ z-score over lookback
        value_mode="rank"       â†’ percentile rank over lookback
    """
    if symbols_df.empty:
        return pd.DataFrame()
    rows = sorted(symbols_df["expiry_year"].astype(str)
                   + "-" + symbols_df["expiry_month"].astype(int).astype(str).str.zfill(2))
    rows = sorted(set(rows))
    tenors = sorted(set(symbols_df["tenor_months"].dropna().astype(int)))

    out = pd.DataFrame(index=rows, columns=tenors, dtype=float)

    try:
        today_row = wide_close.loc[wide_close.index.date == asof_date].iloc[0]
    except Exception:
        return out

    ts = pd.Timestamp(asof_date)
    history = wide_close.loc[wide_close.index < ts].tail(lookback)

    for _, row in symbols_df.iterrows():
        sym = row["symbol"]
        tenor = int(row["tenor_months"]) if pd.notna(row["tenor_months"]) else None
        if tenor is None:
            continue
        ym = f"{int(row['expiry_year'])}-{int(row['expiry_month']):02d}"
        v = today_row.get(sym)
        if v is None or pd.isna(v):
            continue
        if value_mode == "value":
            out.loc[ym, tenor] = float(v)
        elif value_mode == "zscore":
            series = history[sym].dropna() if sym in history.columns else pd.Series(dtype=float)
            if len(series) < 2:
                continue
            mean = series.mean()
            std = series.std(ddof=1)
            if std == 0:
                continue
            out.loc[ym, tenor] = float((v - mean) / std)
        elif value_mode == "rank":
            series = history[sym].dropna() if sym in history.columns else pd.Series(dtype=float)
            if series.empty:
                continue
            out.loc[ym, tenor] = float((series < v).sum() / len(series) * 100)

    return out


def compute_percentile_rank(wide_close: pd.DataFrame, asof_date: date,
                            lookback: int = 252) -> dict:
    """Per-contract percentile rank (0-100) of as-of value vs the PRIOR window.

    Excludes the as-of-date from the sample. Uses (#prior_below / N) * 100 â€” so
    100% means today's value is strictly above every prior observation in the window.
    """
    out = {}
    if wide_close.empty:
        return out
    ts = pd.Timestamp(asof_date)
    history = wide_close.loc[wide_close.index < ts].tail(lookback)
    try:
        today_row = wide_close.loc[wide_close.index.date == asof_date].iloc[0]
    except Exception:
        return out
    for col in wide_close.columns:
        v = today_row.get(col)
        if v is None or pd.isna(v):
            out[col] = None
            continue
        series = history[col].dropna() if col in history.columns else pd.Series(dtype=float)
        if series.empty:
            out[col] = None
        else:
            out[col] = float((series < v).sum() / len(series) * 100)
    return out


# =============================================================================
# CMC (constant-maturity-curve) accessor layer  -- Phase A.7
# =============================================================================
# Thin wrappers over ``lib.cmc`` that hand back panels in the same shape as
# ``load_sra_curve_panel`` consumers expect. Setup detectors ((trend / MR /
# stir).py) take an OHLCV panel indexed by datetime; CMC nodes can be passed
# in unchanged via these accessors.

@st.cache_data(show_spinner=False, ttl=600)
def list_cmc_node_ids(scope: str) -> list[str]:
    """Wrapper over ``lib.cmc.list_cmc_nodes`` for parity with the existing
    ``get_available_tenors`` style of accessor."""
    from lib.cmc import list_cmc_nodes
    return list_cmc_nodes(scope)


@st.cache_data(show_spinner=False, ttl=600)
def load_cmc_node_panel(scope: str, node_id: str,
                          asof_date: date) -> pd.DataFrame:
    """Return a single CMC node's daily panel indexed by datetime, with
    OHLCV columns plus the Carver-correction ``raw_close_anchor``.

    Triggers a CMC build (~17s) on first call for ``asof_date``; cached
    thereafter.

    Output columns:
        open, high, low, close, volume, raw_close_anchor
    Index: pd.DatetimeIndex of bar dates.

    Suitable for direct consumption by setup detectors:
        ``r = detect_a1(panel, asof_date, bp_mult, scope='cmc_outright')``
    """
    from lib.cmc import load_cmc_panel
    long = load_cmc_panel(scope, asof_date)
    if long is None or long.empty:
        return pd.DataFrame()
    sub = long[long["cmc_node"] == node_id]
    if sub.empty:
        return pd.DataFrame()
    sub = sub[sub["has_data"]].copy() if "has_data" in sub.columns else sub.copy()
    sub["bar_date"] = pd.to_datetime(sub["bar_date"])
    panel = sub[["bar_date", "open", "high", "low", "close",
                  "volume_combined", "raw_close_anchor"]].copy()
    panel = panel.rename(columns={"volume_combined": "volume"})
    panel = panel.set_index("bar_date").sort_index()
    return panel


@st.cache_data(show_spinner=False, ttl=600)
def load_cmc_wide_panel(scope: str, asof_date: date,
                          field: str = "close") -> pd.DataFrame:
    """Wide-format pivot of a CMC scope: index=date, columns=cmc_node,
    values=``field``.

    Mirrors :func:`pivot_curve_panel` but for the CMC node universe. Useful
    for cross-node curve analysis and composite scoring.
    """
    from lib.cmc import load_cmc_panel, list_cmc_nodes
    long = load_cmc_panel(scope, asof_date)
    if long is None or long.empty:
        return pd.DataFrame()
    valid = long[long["has_data"]] if "has_data" in long.columns else long
    if field == "volume":
        field = "volume_combined"
    if field not in valid.columns:
        return pd.DataFrame()
    wide = valid.pivot_table(index="bar_date", columns="cmc_node",
                                values=field, aggfunc="first")
    wide = wide.reindex(columns=list_cmc_nodes(scope))
    wide.index = pd.to_datetime(wide.index)
    return wide.sort_index()


# =============================================================================
# Turn-adjuster (Phase 3) accessor layer
# =============================================================================
# Reader API for the residuals/diagnostics produced by
# ``lib.analytics.turn_adjuster``. Phase 4's regime classifier consumes
# ``residual_change`` instead of raw ``cmc_close.diff()`` so calendar artefacts
# (QE/YE/FOMC/NFP/holiday) don't pollute regime labels.

@st.cache_data(show_spinner=False, ttl=600)
def load_turn_residuals_panel(scope: str, asof_date: date) -> pd.DataFrame:
    """Load ``.cmc_cache/turn_residuals_<asof>.parquet`` filtered to ``scope``.

    Output columns:
        scope, cmc_node, bar_date, raw_change, residual_change,
        fitted_change, has_data

    Raises ``FileNotFoundError`` if the residuals parquet has not been built
    yet â€” the daemon (:mod:`lib.turn_adjuster_daemon`) builds it on app boot.
    """
    from pathlib import Path
    cache = Path(__file__).resolve().parent.parent / ".cmc_cache"
    path = cache / f"turn_residuals_{asof_date.isoformat()}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"turn_residuals parquet missing for asof={asof_date.isoformat()}; "
            f"build via lib.analytics.turn_adjuster.build_turn_residuals or "
            f"trigger lib.turn_adjuster_daemon.ensure_turn_residuals_fresh().")
    long = pd.read_parquet(path)
    return long[long["scope"] == scope].reset_index(drop=True)


@st.cache_data(show_spinner=False, ttl=600)
def load_turn_residual_change_wide_panel(scope: str,
                                                 asof_date: date) -> pd.DataFrame:
    """Wide DataFrame: index=bar_date, columns=cmc_node, values=residual_change.

    The canonical input to Phase 4's A1 regime classifier.
    """
    long = load_turn_residuals_panel(scope, asof_date)
    if long.empty:
        return pd.DataFrame()
    wide = long.pivot_table(index="bar_date", columns="cmc_node",
                                values="residual_change", aggfunc="first")
    wide.index = pd.to_datetime(wide.index)
    # Preserve canonical column order from lib.cmc.list_cmc_nodes
    from lib.cmc import list_cmc_nodes
    canonical = list_cmc_nodes(scope)
    wide = wide.reindex(columns=[c for c in canonical if c in wide.columns])
    return wide.sort_index()


@st.cache_data(show_spinner=False, ttl=600)
def load_turn_diagnostics(asof_date: date) -> pd.DataFrame:
    """Load ``.cmc_cache/turn_diagnostics_<asof>.parquet``.

    Per-node regression stats: n_obs, dof, r_squared, raw_var, residual_var,
    var_reduction_pct, beta_*, se_*, p_*, eff_n_*, low_sample_dummies.
    """
    from pathlib import Path
    cache = Path(__file__).resolve().parent.parent / ".cmc_cache"
    path = cache / f"turn_diagnostics_{asof_date.isoformat()}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"turn_diagnostics parquet missing for asof={asof_date.isoformat()}")
    return pd.read_parquet(path)


# =============================================================================
# Regime classifier (Phase 4) accessor layer
# =============================================================================

@st.cache_data(show_spinner=False, ttl=600)
def load_regime_states(asof_date: date) -> pd.DataFrame:
    """Load ``.cmc_cache/regime_states_<asof>.parquet``: bar_date Ã— state_id Ã—
    state_name Ã— posterior columns."""
    from pathlib import Path
    cache = Path(__file__).resolve().parent.parent / ".cmc_cache"
    path = cache / f"regime_states_{asof_date.isoformat()}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"regime_states parquet missing for asof={asof_date.isoformat()}")
    df = pd.read_parquet(path)
    df["bar_date"] = pd.to_datetime(df["bar_date"])
    return df


@st.cache_data(show_spinner=False, ttl=600)
def load_regime_diagnostics(asof_date: date) -> pd.DataFrame:
    from pathlib import Path
    cache = Path(__file__).resolve().parent.parent / ".cmc_cache"
    path = cache / f"regime_diagnostics_{asof_date.isoformat()}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"regime_diagnostics parquet missing for asof={asof_date.isoformat()}")
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False, ttl=600)
def get_current_regime(asof_date: date) -> dict:
    """Return the current (latest bar) regime as a dict for the header chip.

    Output: {state_id, state_name, top_state_posterior, bar_date}
    """
    states = load_regime_states(asof_date)
    if states.empty:
        return {}
    last = states.sort_values("bar_date").iloc[-1]
    return {
        "state_id": int(last["state_id"]),
        "state_name": str(last["state_name"]),
        "top_state_posterior": float(last["top_state_posterior"]),
        "bar_date": last["bar_date"].date(),
    }


# =============================================================================
# Policy path (Phase 5) accessor layer
# =============================================================================

@st.cache_data(show_spinner=False, ttl=600)
def load_policy_path(asof_date: date) -> pd.DataFrame:
    from pathlib import Path
    cache = Path(__file__).resolve().parent.parent / ".cmc_cache"
    path = cache / f"policy_path_{asof_date.isoformat()}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"policy_path missing for asof={asof_date.isoformat()}")
    df = pd.read_parquet(path)
    df["meeting_date"] = pd.to_datetime(df["meeting_date"]).dt.date
    return df


@st.cache_data(show_spinner=False, ttl=600)
def load_signal_emissions(asof_date: date) -> pd.DataFrame:
    """Phase 7 signal_emit canonical table (24 cols)."""
    from pathlib import Path
    cache = Path(__file__).resolve().parent.parent / ".signal_cache"
    path = cache / f"signal_emit_{asof_date.isoformat()}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"signal_emit missing for asof={asof_date.isoformat()}")
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False, ttl=600)
def top_recommended_signals(asof_date: date, n: int = 10) -> pd.DataFrame:
    """Plan Â§3.2 ranked feed: filter gate_quality==CLEAN AND eff_n>=30 AND
    not transition_flag AND not conflict_flag; sort by
    (percentile_rank desc, eff_n desc, regime_stability desc)."""
    df = load_signal_emissions(asof_date)
    if df.empty:
        return df
    filt = df[(df["gate_quality"] == "CLEAN")
                  & (df["eff_n"] >= 30)
                  & (~df["transition_flag"])
                  & (~df["conflict_flag"])]
    return filt.sort_values(
        by=["percentile_rank", "eff_n", "regime_stability"],
        ascending=[False, False, False]).head(n)


@st.cache_data(show_spinner=False, ttl=600)
def load_event_impact(asof_date: date) -> pd.DataFrame:
    """Phase 6 event-impact regression results."""
    from pathlib import Path
    cache = Path(__file__).resolve().parent.parent / ".cmc_cache"
    path = cache / f"event_impact_{asof_date.isoformat()}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"event_impact missing for asof={asof_date.isoformat()}")
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False, ttl=600)
def top_event_signals(asof_date: date, n: int = 10,
                          regime_filter: str = "ALL") -> pd.DataFrame:
    """Top-N event-impact signals by score_full (or score_recency_weighted
    if available). Filter by regime ('ALL' = full-sample only)."""
    df = load_event_impact(asof_date)
    if regime_filter == "ALL":
        df = df[df["scope"] == "full"]
    else:
        df = df[df["regime"] == regime_filter]
    return df.nlargest(n, "score_full")


@st.cache_data(show_spinner=False, ttl=600)
def get_policy_path_summary(asof_date: date) -> dict:
    """Compact summary for header chip: terminal_rate_bp, cycle_label,
    next_meeting_date, next_expected_change_bp."""
    from pathlib import Path
    import json as _json
    cache = Path(__file__).resolve().parent.parent / ".cmc_cache"
    mf = cache / f"policy_path_manifest_{asof_date.isoformat()}.json"
    if not mf.exists():
        return {}
    m = _json.loads(mf.read_text())
    out = {
        "terminal_rate_bp": m.get("terminal_rate_bp"),
        "floor_rate_bp": m.get("floor_rate_bp"),
        "current_rate_bp": m.get("current_rate_bp"),
        "cycle_label": m.get("cycle_label"),
        "n_meetings": m.get("n_meetings"),
    }
    try:
        df = load_policy_path(asof_date)
        if not df.empty:
            row = df.iloc[0]
            out["next_meeting_date"] = row["meeting_date"]
            out["next_expected_change_bp"] = float(row["expected_change_bp"])
    except Exception:
        pass
    return out


# ====================================================================
# PCA-engine additions: listed-spread/fly symbol derivation + loader
# ====================================================================

def derive_listed_spread_symbol(leg_symbols: list) -> Optional[str]:
    """Given 2 outright leg symbols (e.g. ['SRAH26', 'SRAM26']), derive the
    listed CME calendar-spread symbol ('SRAH26-M26'). Returns None if not
    exactly 2 SR3 outright symbols.

    Catalog convention: '<front_full_symbol>-<back_month_letter><back_year>'.
    """
    if len(leg_symbols) != 2:
        return None
    syms = sorted(leg_symbols, key=_outright_sort_key)
    if any(not s.startswith("SRA") or len(s) < 6 for s in syms):
        return None
    front = syms[0]
    back_short = syms[1][3:]    # strip "SRA" prefix
    return f"{front}-{back_short}"


def derive_listed_fly_symbol(leg_symbols: list) -> Optional[str]:
    """Given 3 outright leg symbols, derive the listed butterfly symbol
    ('SRAH26-M26-U26'). Returns None if not exactly 3."""
    if len(leg_symbols) != 3:
        return None
    syms = sorted(leg_symbols, key=_outright_sort_key)
    if any(not s.startswith("SRA") or len(s) < 6 for s in syms):
        return None
    front = syms[0]
    mid_short = syms[1][3:]
    back_short = syms[2][3:]
    return f"{front}-{mid_short}-{back_short}"


_MONTH_CODE_TO_NUM = {"F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
                        "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12}


def _outright_sort_key(sym: str) -> tuple:
    """Sort outrights by (year, month). Symbol fmt: SRA<month-code><yy>."""
    if not sym.startswith("SRA") or len(sym) < 6:
        return (9999, 99)
    month_code = sym[3]
    year_str = sym[4:6]
    return (int(year_str), _MONTH_CODE_TO_NUM.get(month_code, 99))


def load_listed_spread_fly_panel(start_date: date, end_date: date,
                                       *, max_tenor_months: int = 12) -> dict:
    """Load CLOSE panels for all live listed SRA spreads + flies in the date
    range. Returns dict with keys:
      "spread_close": wide DataFrame (date Ã— symbol)
      "fly_close":    wide DataFrame (date Ã— symbol)
      "spread_catalog": DataFrame of (symbol, expiry_year, expiry_month, tenor_months)
      "fly_catalog":    DataFrame of (symbol, expiry_year, expiry_month, tenor_months)

    Loads tenor_months â‰¤ max_tenor_months (default 12) â€” captures the universe
    of listed calendar spreads and flies that the engine actually trades.
    """
    out = {"spread_close": pd.DataFrame(), "fly_close": pd.DataFrame(),
            "spread_catalog": pd.DataFrame(), "fly_catalog": pd.DataFrame()}
    for strat in ("spread", "fly"):
        catalog_rows = []
        for tenor in range(1, max_tenor_months + 1):
            syms = get_sra_live_symbols(strat, tenor)
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


# Available tenors for spreads / flies, derived from catalog (cached separately)
def contract_range_str(symbols_df: pd.DataFrame) -> str:
    """e.g. 'SRAJ26 â†’ SRAH32'."""
    if symbols_df.empty:
        return "â€”"
    syms = symbols_df["symbol"].tolist()
    return f"{syms[0]} â†’ {syms[-1]}"


def tenor_breakdown(strategy: str) -> list[tuple[int, int, str]]:
    """Return [(tenor, count, range_str), ...] for given strategy."""
    out = []
    for t in get_available_tenors(strategy):
        df = get_sra_live_symbols(strategy, t)
        out.append((t, len(df), contract_range_str(df)))
    return out


@st.cache_data(show_spinner=False, ttl=600)
def get_available_tenors(strategy: str) -> list[int]:
    """Distinct tenor_months for live SRA contracts of given strategy."""
    con = get_ohlc_connection()
    if con is None:
        return []
    latest = get_sra_snapshot_latest_date()
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
    WHERE cc.is_inter = FALSE AND cc.base_product = 'SRA'
      AND cc.strategy = '{strategy}'
      AND cc.tenor_months IS NOT NULL
      AND lb.last_unix >= EPOCH(DATE '{cutoff.isoformat()}')
    ORDER BY cc.tenor_months
    """
    rows = con.execute(sql).fetchall()
    return [int(r[0]) for r in rows]


# =============================================================================
# Reference rates (FDTR / SOFR) â€” for overlays
# =============================================================================

_BBG_RATES_DRIVERS = r"D:\BBG data\parquet\rates_drivers"
_BBG_ECO = r"D:\BBG data\parquet\eco"


@st.cache_data(show_spinner=False, ttl=3600)
def load_reference_rate_panel(start_date: date, end_date: date) -> pd.DataFrame:
    """Load FDTR upper, FDTR lower, SOFR rate over date range.

    Returns DataFrame indexed by date with columns: fdtr_upper, fdtr_lower, sofr.
    Forward-filled to cover all business days in range.
    """
    files = {
        "fdtr_upper": (os.path.join(_BBG_RATES_DRIVERS, "FDTR_Index.parquet"), "PX_LAST"),
        "fdtr_lower": (os.path.join(_BBG_ECO, "FDTRFTRL_Index.parquet"), "PX_LAST"),
        "sofr": (os.path.join(_BBG_RATES_DRIVERS, "SOFRRATE_Index.parquet"), "PX_LAST"),
    }
    con = get_bbg_inmemory_connection()
    series = {}
    for label, (path, valcol) in files.items():
        if not os.path.exists(path):
            continue
        try:
            duck_path = path.replace("\\", "/")
            df = con.execute(
                f"SELECT date, {valcol} FROM read_parquet('{duck_path}')"
            ).fetchdf()
            df["date"] = pd.to_datetime(df["date"])
            df = df.dropna(subset=[valcol]).set_index("date").sort_index()
            series[label] = df[valcol]
        except Exception:
            continue

    if not series:
        return pd.DataFrame()

    out = pd.concat(series, axis=1)
    full_idx = pd.date_range(start=pd.Timestamp(start_date), end=pd.Timestamp(end_date), freq="B")
    out = out.reindex(out.index.union(full_idx)).sort_index().ffill()
    out = out.loc[(out.index >= pd.Timestamp(start_date)) & (out.index <= pd.Timestamp(end_date))]
    return out


def get_reference_rates_at(asof_date: date, ref_panel: pd.DataFrame) -> dict:
    """Return {fdtr_upper, fdtr_lower, fdtr_mid, sofr} at a given date (forward-fill)."""
    out = {"fdtr_upper": None, "fdtr_lower": None, "fdtr_mid": None, "sofr": None}
    if ref_panel is None or ref_panel.empty:
        return out
    ts = pd.Timestamp(asof_date)
    idx = ref_panel.index[ref_panel.index <= ts]
    if len(idx) == 0:
        return out
    row = ref_panel.loc[idx[-1]]
    upper = row.get("fdtr_upper") if "fdtr_upper" in ref_panel.columns else None
    lower = row.get("fdtr_lower") if "fdtr_lower" in ref_panel.columns else None
    sofr = row.get("sofr") if "sofr" in ref_panel.columns else None
    upper = float(upper) if pd.notna(upper) else None
    lower = float(lower) if pd.notna(lower) else None
    sofr = float(sofr) if pd.notna(sofr) else None
    mid = (upper + lower) / 2 if (upper is not None and lower is not None) else None
    return {"fdtr_upper": upper, "fdtr_lower": lower, "fdtr_mid": mid, "sofr": sofr}


# =============================================================================
# Curve sections (front / mid / back)
# =============================================================================

DEFAULT_FRONT_END = 6
DEFAULT_MID_END = 14


def compute_section_split(n_contracts: int, front_end: int = DEFAULT_FRONT_END,
                          mid_end: int = DEFAULT_MID_END):
    """Return (front_indices, mid_indices, back_indices) given contract count.

    Adapts when n_contracts < mid_end so each section gets at least 1.
    """
    if n_contracts <= 0:
        return range(0), range(0), range(0)
    fe = min(front_end, n_contracts)
    me = min(mid_end, n_contracts)
    if me <= fe:
        me = min(fe + 1, n_contracts)
    return range(0, fe), range(fe, me), range(me, n_contracts)


def section_label_for_index(idx: int, front_end: int = DEFAULT_FRONT_END,
                            mid_end: int = DEFAULT_MID_END) -> str:
    if idx < front_end:
        return "front"
    if idx < mid_end:
        return "mid"
    return "back"


# =============================================================================
# Pack groupings (Whites / Reds / Greens / Blues / Golds / ...)
# =============================================================================

PACK_NAMES = ["Whites", "Reds", "Greens", "Blues", "Golds", "Purples", "Browns"]
PACK_COLORS = ["#f0f4fa", "#f87171", "#4ade80", "#60a5fa", "#e8b75d", "#a78bfa", "#a16207"]
QUARTERLY_MONTH_CODES = {"H", "M", "U", "Z"}


def get_quarterly_outright_indices(symbols_df: pd.DataFrame) -> list:
    """Return positional indices of quarterly outrights (H/M/U/Z) within symbols_df.

    SRA outright symbols look like 'SRAH26' â€” char index 3 is the month code.
    """
    if symbols_df.empty:
        return []
    indices = []
    for i, sym in enumerate(symbols_df["symbol"].tolist()):
        if len(sym) >= 6 and sym[3].upper() in QUARTERLY_MONTH_CODES:
            indices.append(i)
    return indices


def compute_pack_groups(symbols_df: pd.DataFrame):
    """Return [(pack_name, [symbols]), ...] grouping quarterlies into packs of 4."""
    qidx = get_quarterly_outright_indices(symbols_df)
    if not qidx:
        return []
    quarterly_syms = symbols_df.iloc[qidx]["symbol"].tolist()
    out = []
    for i in range(0, len(quarterly_syms), 4):
        chunk = quarterly_syms[i:i + 4]
        if not chunk:
            break
        pack_idx = i // 4
        name = PACK_NAMES[pack_idx] if pack_idx < len(PACK_NAMES) else f"Pack{pack_idx + 1}"
        out.append((name, chunk))
    return out


# =============================================================================
# Curve change & decomposition
# =============================================================================

def compute_curve_change(close_today: list, close_compare: list,
                         in_bp_units: bool = True,
                         contracts: Optional[list[str]] = None,
                         base_product: str = "SRA") -> list:
    """Per-contract change (today âˆ’ compare).

    When `contracts` is given, the per-contract unit convention is looked up in
    the contract-units catalog so that already-bp spreads/flies are NOT
    re-multiplied by 100. When `contracts` is None or the catalog is empty,
    falls back to the global `in_bp_units` switch (Ã—100 for all if True).
    """
    out = []
    if contracts is not None:
        from lib.contract_units import bp_multipliers_for, load_catalog
        cat = load_catalog()
        if not cat.empty:
            mults = bp_multipliers_for(contracts, base_product, cat) if in_bp_units \
                    else [1.0] * len(contracts)
            for t, c, m in zip(close_today, close_compare, mults):
                if t is None or c is None or pd.isna(t) or pd.isna(c):
                    out.append(None)
                else:
                    out.append((t - c) * m)
            return out
    # Fallback (no catalog or no contract list)
    mult = 100 if in_bp_units else 1
    for t, c in zip(close_today, close_compare):
        if t is None or c is None or pd.isna(t) or pd.isna(c):
            out.append(None)
        else:
            out.append((t - c) * mult)
    return out


def compute_decomposition(changes: list) -> dict:
    """Decompose curve change into parallel + slope + curvature components.

    Regress changes against [1, x, x^2] where x is normalized contract index.
    Returns dict with keys parallel / slope / curvature / residual_rmse.
    """
    valid = [(i, c) for i, c in enumerate(changes) if c is not None and not pd.isna(c)]
    if len(valid) < 3:
        return {"parallel": None, "slope": None, "curvature": None, "residual_rmse": None}

    indices = np.array([i for i, _ in valid], dtype=float)
    vals = np.array([v for _, v in valid], dtype=float)
    n = len(vals)
    denom = max(1.0, indices.max() - indices.mean())
    x = (indices - indices.mean()) / denom

    X = np.column_stack([np.ones(n), x, x * x])
    try:
        beta, *_ = np.linalg.lstsq(X, vals, rcond=None)
    except Exception:
        return {"parallel": None, "slope": None, "curvature": None, "residual_rmse": None}

    parallel, slope, curvature = float(beta[0]), float(beta[1]), float(beta[2])
    residuals = vals - X @ beta
    rmse = float(np.sqrt((residuals ** 2).mean()))
    return {"parallel": parallel, "slope": slope, "curvature": curvature, "residual_rmse": rmse}


# =============================================================================
# Per-contract z-scores and percentile bands (for Ribbon mode)
# =============================================================================

def compute_per_contract_zscores(wide_close: pd.DataFrame, asof_date: date,
                                 lookback: int = 252) -> dict:
    """Z-score of as-of-date close vs the prior `lookback` trading days, per contract.

    The history window EXCLUDES the as-of-date itself â€” we compare today against
    the prior N days (no look-ahead bias). This matters for short lookbacks (5d, 15d)
    where including today would contaminate the sample.
    """
    out = {}
    if wide_close.empty:
        return out
    ts = pd.Timestamp(asof_date)
    history = wide_close.loc[wide_close.index < ts].tail(lookback)
    if history.empty:
        return out
    try:
        today_row = wide_close.loc[wide_close.index.date == asof_date].iloc[0]
    except Exception:
        return out
    # Use ddof=1 sample std; needs at least 2 observations
    means = history.mean()
    stds = history.std(ddof=1)
    for col in wide_close.columns:
        v = today_row.get(col)
        if v is None or pd.isna(v):
            out[col] = None
            continue
        m, s = means.get(col), stds.get(col)
        # Need a real value, real std, std > 0, AND at least 2 valid prior observations
        valid_count = history[col].dropna().shape[0] if col in history.columns else 0
        if pd.isna(m) or pd.isna(s) or s == 0 or valid_count < 2:
            out[col] = None
        else:
            out[col] = float((v - m) / s)
    return out


def compute_percentile_bands(wide_close: pd.DataFrame, asof_date: date,
                             lookback: int = 252) -> dict:
    """Per-contract percentile bands {p05, p25, p50, p75, p95, mean} from PRIOR window.

    Excludes the as-of-date so today's value isn't in the band sample.
    """
    if wide_close.empty:
        return {}
    ts = pd.Timestamp(asof_date)
    history = wide_close.loc[wide_close.index < ts].tail(lookback)
    if history.empty:
        return {}
    # quantile uses linear interpolation by default â€” fine for n>=5; for n<5 it
    # collapses toward min/max which is the natural behavior.
    return {
        "p05": history.quantile(0.05).to_dict(),
        "p25": history.quantile(0.25).to_dict(),
        "p50": history.quantile(0.50).to_dict(),
        "p75": history.quantile(0.75).to_dict(),
        "p95": history.quantile(0.95).to_dict(),
        "mean": history.mean().to_dict(),
    }


def get_contract_history(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Pull full daily OHLC + volume history for a single contract."""
    con = get_ohlc_connection()
    if con is None:
        return pd.DataFrame()
    sql = f"""
    SELECT DATE(to_timestamp(t.time/1000.0)) AS bar_date,
           t.open, t.high, t.low, t.close, t.volume
    FROM mde2_timeseries t
    WHERE t.symbol = '{symbol}'
      AND t."interval" = '1D' AND t.calc_method = 'api'
      AND DATE(to_timestamp(t.time/1000.0)) BETWEEN DATE '{start_date.isoformat()}' AND DATE '{end_date.isoformat()}'
    ORDER BY bar_date
    """
    try:
        df = con.execute(sql).fetchdf()
        df = df.rename(columns={"bar_date": "date"})
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception:
        return pd.DataFrame()


def compute_pairwise_spread_matrix(symbols_df: pd.DataFrame,
                                    wide_close: pd.DataFrame,
                                    asof_date: date,
                                    value_mode: str = "value",
                                    lookback: int = 60) -> pd.DataFrame:
    """For SPREADS or FLIES, build a 2D matrix indexed by left-leg expiry vs tenor.

    Returns DataFrame: rows = left-leg symbol (the front contract of the spread),
    cols = tenor_months (3, 6, 9, 12 for spreads; 3, 6, 9, 12 for flies).
    Values:
        value_mode="value"      â†’ today's price
        value_mode="zscore"     â†’ z-score over lookback
        value_mode="rank"       â†’ percentile rank over lookback
    """
    if symbols_df.empty:
        return pd.DataFrame()
    rows = sorted(symbols_df["expiry_year"].astype(str)
                   + "-" + symbols_df["expiry_month"].astype(int).astype(str).str.zfill(2))
    rows = sorted(set(rows))
    tenors = sorted(set(symbols_df["tenor_months"].dropna().astype(int)))

    out = pd.DataFrame(index=rows, columns=tenors, dtype=float)

    try:
        today_row = wide_close.loc[wide_close.index.date == asof_date].iloc[0]
    except Exception:
        return out

    ts = pd.Timestamp(asof_date)
    history = wide_close.loc[wide_close.index < ts].tail(lookback)

    for _, row in symbols_df.iterrows():
        sym = row["symbol"]
        tenor = int(row["tenor_months"]) if pd.notna(row["tenor_months"]) else None
        if tenor is None:
            continue
        ym = f"{int(row['expiry_year'])}-{int(row['expiry_month']):02d}"
        v = today_row.get(sym)
        if v is None or pd.isna(v):
            continue
        if value_mode == "value":
            out.loc[ym, tenor] = float(v)
        elif value_mode == "zscore":
            series = history[sym].dropna() if sym in history.columns else pd.Series(dtype=float)
            if len(series) < 2:
                continue
            mean = series.mean()
            std = series.std(ddof=1)
            if std == 0:
                continue
            out.loc[ym, tenor] = float((v - mean) / std)
        elif value_mode == "rank":
            series = history[sym].dropna() if sym in history.columns else pd.Series(dtype=float)
            if series.empty:
                continue
            out.loc[ym, tenor] = float((series < v).sum() / len(series) * 100)

    return out


def compute_percentile_rank(wide_close: pd.DataFrame, asof_date: date,
                            lookback: int = 252) -> dict:
    """Per-contract percentile rank (0-100) of as-of value vs the PRIOR window.

    Excludes the as-of-date from the sample. Uses (#prior_below / N) * 100 â€” so
    100% means today's value is strictly above every prior observation in the window.
    """
    out = {}
    if wide_close.empty:
        return out
    ts = pd.Timestamp(asof_date)
    history = wide_close.loc[wide_close.index < ts].tail(lookback)
    try:
        today_row = wide_close.loc[wide_close.index.date == asof_date].iloc[0]
    except Exception:
        return out
    for col in wide_close.columns:
        v = today_row.get(col)
        if v is None or pd.isna(v):
            out[col] = None
            continue
        series = history[col].dropna() if col in history.columns else pd.Series(dtype=float)
        if series.empty:
            out[col] = None
        else:
            out[col] = float((series < v).sum() / len(series) * 100)
    return out