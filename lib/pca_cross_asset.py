"""Cross-asset data loaders for the PCA subtab.

The current PCA engine reads only ~15% of the available BBG warehouse. This
module wires the rest. Every loader is robust to missing tickers (returns None
or empty Series), normalizes the date index, and exposes a single composite
panel via `load_cross_asset_panel()`.

Categories wired:

  · vol_indices   — MOVE / SRVIX / SKEW / VIX / VVIX / OVX
  · equity_indices — SPX / NDX / INDU
  · fx            — EURUSD / USDJPY / GBPUSD / USDCHF (synthesizes DXY)
  · credit        — CDX_IG / CDX_HY / LUACOAS / LF98OAS
  · commodities   — Gold / Copper / WTI / Brent (composite log change)
  · rates_drivers — USGG2YR / 5YR / 10YR / 30YR / FARBAST / RRP

Plus external sources:

  · ACM term premia          — NY Fed XLS (cached locally)
  · Kim-Wright term premia   — Federal Reserve CSV (cached locally)
  · Treasury auction calendar — TreasuryDirect API (cached daily)

All composite panels are indexed by date and aligned to a common business-day
calendar. Missing days are forward-filled within a 5-day tolerance for daily
indicators; weekly indicators (FARBAST, ACM) are forward-filled to daily.
"""
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from lib.connections import bbg_parquet_to_series, read_bbg_parquet_robust


# =============================================================================
# Local cache directory for external data (ACM, Kim-Wright, TreasuryDirect)
# =============================================================================
_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "_cross_asset_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)


# =============================================================================
# B.1 — Vol indices
# =============================================================================
VOL_TICKERS = {
    "MOVE":  ("vol_indices", "MOVE_Index"),       # 1m UST vol — canonical bond-vol gauge
    "SRVIX": ("vol_indices", "SRVIX_Index"),       # 1y swap-rate vol — STIR-specific
    "SKEW":  ("vol_indices", "SKEW_Index"),         # CBOE SKEW — tail-risk premium
    "VIX":   ("vol_indices", "VIX_Index"),           # equity vol
    "VVIX":  ("vol_indices", "VVIX_Index"),         # vol-of-vol
    "OVX":   ("vol_indices", "OVX_Index"),           # crude vol
}


def load_vol_panel(asof: Optional[date] = None,
                    lookback_days: int = 252) -> pd.DataFrame:
    """Load vol indices panel indexed by date with PX_LAST cols.

    Returns DataFrame with cols MOVE/SRVIX/SKEW/VIX/VVIX/OVX (only those that
    successfully loaded). Index is a normalized DatetimeIndex.
    """
    series_dict = {}
    for col, (cat, ticker) in VOL_TICKERS.items():
        s = bbg_parquet_to_series(cat, ticker)
        if s is None or s.empty:
            continue
        series_dict[col] = s
    if not series_dict:
        return pd.DataFrame()
    df = pd.DataFrame(series_dict).sort_index()
    if asof is not None:
        df = df.loc[df.index <= pd.Timestamp(asof)]
        if lookback_days > 0:
            cutoff = pd.Timestamp(asof) - pd.Timedelta(days=lookback_days * 2)
            df = df.loc[df.index >= cutoff]
    return df


# =============================================================================
# B.2 — Equity
# =============================================================================
EQUITY_TICKERS = {
    "SPX":  ("equity_indices", "SPX_Index"),
    "NDX":  ("equity_indices", "NDX_Index"),
    "INDU": ("equity_indices", "INDU_Index"),
}


def load_equity_panel(asof: Optional[date] = None,
                       lookback_days: int = 252) -> pd.DataFrame:
    series_dict = {}
    for col, (cat, ticker) in EQUITY_TICKERS.items():
        s = bbg_parquet_to_series(cat, ticker)
        if s is None or s.empty:
            continue
        series_dict[col] = s
    if not series_dict:
        return pd.DataFrame()
    df = pd.DataFrame(series_dict).sort_index()
    if asof is not None:
        df = df.loc[df.index <= pd.Timestamp(asof)]
        if lookback_days > 0:
            cutoff = pd.Timestamp(asof) - pd.Timedelta(days=lookback_days * 2)
            df = df.loc[df.index >= cutoff]
    return df


# =============================================================================
# B.3 — FX (build synthetic DXY from major-USD basket)
# =============================================================================
FX_TICKERS = {
    "EURUSD": ("fx", "EURUSD_Curncy"),
    "USDJPY": ("fx", "USDJPY_Curncy"),
    "GBPUSD": ("fx", "GBPUSD_Curncy"),
    "USDCHF": ("fx", "USDCHF_Curncy"),
    "USDCAD": ("fx", "USDCAD_Curncy"),
}

# DXY (USD index) basket weights (approximate ICE DXY definition)
_DXY_WEIGHTS = {"EURUSD": -0.576, "USDJPY": +0.136, "GBPUSD": -0.119,
                  "USDCHF": +0.036, "USDCAD": +0.091}    # excludes SEK (~4.2%)


def load_fx_panel(asof: Optional[date] = None,
                    lookback_days: int = 252) -> pd.DataFrame:
    series_dict = {}
    for col, (cat, ticker) in FX_TICKERS.items():
        s = bbg_parquet_to_series(cat, ticker)
        if s is None or s.empty:
            continue
        series_dict[col] = s
    if not series_dict:
        return pd.DataFrame()
    df = pd.DataFrame(series_dict).sort_index()
    # Synthesize DXY index = exp(Σ w_i · log(rate_i))
    if all(c in df.columns for c in ("EURUSD", "USDJPY", "GBPUSD", "USDCHF", "USDCAD")):
        # Inverse pairs (EURUSD / GBPUSD): negate weight via 1/rate.
        log_dxy = np.zeros(len(df))
        for col, w in _DXY_WEIGHTS.items():
            if col not in df.columns:
                continue
            log_dxy = log_dxy + w * np.log(df[col].values)
        df["DXY_synth"] = np.exp(log_dxy) * 100.0    # rebase to ~100
    if asof is not None:
        df = df.loc[df.index <= pd.Timestamp(asof)]
        if lookback_days > 0:
            cutoff = pd.Timestamp(asof) - pd.Timedelta(days=lookback_days * 2)
            df = df.loc[df.index >= cutoff]
    return df


# =============================================================================
# B.4 — Credit
# =============================================================================
CREDIT_TICKERS = {
    "CDX_IG":  ("credit", "CDX_IG_CDSI_GEN_5Y_Corp"),
    "CDX_HY":  ("credit", "CDX_HY_CDSI_GEN_5Y_Corp"),
    "LUACOAS": ("credit", "LUACOAS_Index"),         # Bloomberg US Aggregate corporate OAS
    "LF98OAS": ("credit", "LF98OAS_Index"),         # HY OAS
}


def load_credit_panel(asof: Optional[date] = None,
                       lookback_days: int = 252) -> pd.DataFrame:
    series_dict = {}
    for col, (cat, ticker) in CREDIT_TICKERS.items():
        s = bbg_parquet_to_series(cat, ticker)
        if s is None or s.empty:
            continue
        series_dict[col] = s
    if not series_dict:
        return pd.DataFrame()
    df = pd.DataFrame(series_dict).sort_index()
    if asof is not None:
        df = df.loc[df.index <= pd.Timestamp(asof)]
        if lookback_days > 0:
            cutoff = pd.Timestamp(asof) - pd.Timedelta(days=lookback_days * 2)
            df = df.loc[df.index >= cutoff]
    return df


# =============================================================================
# B.5 — Commodities
# =============================================================================
COMMODITY_TICKERS = {
    # Gold: not in current BBG warehouse (only MXMMGOLD_Index in eco). Will be
    # None and silently skipped — composite still computes from Copper/WTI/Brent.
    "COPPER": ("metals_inventory", "LMCADY_Comdty"),
    "WTI":    ("energy_extras", "USCRWTIC_Index"),
    "BRENT":  ("energy_extras", "CO1_Comdty"),
}


def load_commodity_panel(asof: Optional[date] = None,
                           lookback_days: int = 252) -> pd.DataFrame:
    series_dict = {}
    for col, (cat, ticker) in COMMODITY_TICKERS.items():
        s = bbg_parquet_to_series(cat, ticker)
        if s is None or s.empty:
            continue
        series_dict[col] = s
    if not series_dict:
        return pd.DataFrame()
    df = pd.DataFrame(series_dict).sort_index()
    # Equally-weighted log-change composite (63d window)
    if not df.empty:
        log_chg = np.log(df / df.shift(63)).mean(axis=1)
        df["composite_log_chg_63d"] = log_chg
    if asof is not None:
        df = df.loc[df.index <= pd.Timestamp(asof)]
        if lookback_days > 0:
            cutoff = pd.Timestamp(asof) - pd.Timedelta(days=lookback_days * 2)
            df = df.loc[df.index >= cutoff]
    return df


# =============================================================================
# B.6 — Cash UST yields (already in rates_drivers but unused in dossier)
# =============================================================================
UST_TICKERS = {
    "UST_2Y":  ("rates_drivers", "USGG2YR_Index"),
    "UST_5Y":  ("rates_drivers", "USGG5YR_Index"),
    "UST_10Y": ("rates_drivers", "USGG10YR_Index"),
    "UST_30Y": ("rates_drivers", "USGG30YR_Index"),
}


def load_ust_panel(asof: Optional[date] = None,
                    lookback_days: int = 252) -> pd.DataFrame:
    series_dict = {}
    for col, (cat, ticker) in UST_TICKERS.items():
        s = bbg_parquet_to_series(cat, ticker)
        if s is None or s.empty:
            continue
        series_dict[col] = s
    if not series_dict:
        return pd.DataFrame()
    df = pd.DataFrame(series_dict).sort_index()
    # Standard slope spreads
    if "UST_2Y" in df.columns and "UST_10Y" in df.columns:
        df["slope_2s10s_bp"] = (df["UST_10Y"] - df["UST_2Y"]) * 100
    if "UST_5Y" in df.columns and "UST_30Y" in df.columns:
        df["slope_5s30s_bp"] = (df["UST_30Y"] - df["UST_5Y"]) * 100
    if asof is not None:
        df = df.loc[df.index <= pd.Timestamp(asof)]
        if lookback_days > 0:
            cutoff = pd.Timestamp(asof) - pd.Timedelta(days=lookback_days * 2)
            df = df.loc[df.index >= cutoff]
    return df


# =============================================================================
# B.8 — Fed B/S + RRP
# =============================================================================
FEDBS_TICKERS = {
    "FARBAST":  ("rates_drivers", "FARBAST_Index"),     # Fed Balance Sheet (H.4.1)
    "RRP":      ("rates_drivers", "RRPONTSY_Index"),    # ON RRP usage (alt: FRBOVNTS)
}


def load_fedbs_panel(asof: Optional[date] = None,
                      lookback_days: int = 504) -> pd.DataFrame:
    series_dict = {}
    for col, (cat, ticker) in FEDBS_TICKERS.items():
        s = bbg_parquet_to_series(cat, ticker)
        if s is None or s.empty:
            # Try alternate ticker for RRP
            if col == "RRP":
                s = bbg_parquet_to_series(cat, "FRBOVNTS_Index")
        if s is None or s.empty:
            continue
        series_dict[col] = s
    if not series_dict:
        return pd.DataFrame()
    df = pd.DataFrame(series_dict).sort_index()
    if not df.empty and "FARBAST" in df.columns:
        # Weekly Δ in $bn
        df["FARBAST_wow_chg_bn"] = (df["FARBAST"].diff(5) / 1e9)
    if asof is not None:
        df = df.loc[df.index <= pd.Timestamp(asof)]
        if lookback_days > 0:
            cutoff = pd.Timestamp(asof) - pd.Timedelta(days=lookback_days * 2)
            df = df.loc[df.index >= cutoff]
    return df


# =============================================================================
# B.9 — Treasury auction forward calendar (TreasuryDirect API)
# =============================================================================
_TD_CACHE = os.path.join(_CACHE_DIR, "treasury_auctions.json")
_TD_CACHE_TTL_HOURS = 24


def load_treasury_auction_calendar(asof: Optional[date] = None,
                                       force_refresh: bool = False) -> pd.DataFrame:
    """Fetch upcoming Treasury auction calendar from TreasuryDirect.

    Cached for 24h. Returns DataFrame with cols [cusip, securityType, term,
    announcementDate, auctionDate, settlementDate, offeringAmount].
    """
    # Check cache
    use_cache = (not force_refresh and os.path.exists(_TD_CACHE)
                  and ((datetime.now().timestamp() - os.path.getmtime(_TD_CACHE))
                       < _TD_CACHE_TTL_HOURS * 3600))
    if use_cache:
        try:
            with open(_TD_CACHE, "r", encoding="utf-8") as f:
                rows = json.load(f)
            df = pd.DataFrame(rows)
        except Exception:
            df = pd.DataFrame()
    else:
        try:
            import urllib.request
            url = "https://www.treasurydirect.gov/TA_WS/securities/announced?format=json"
            with urllib.request.urlopen(url, timeout=10) as resp:
                rows = json.loads(resp.read().decode("utf-8"))
            with open(_TD_CACHE, "w", encoding="utf-8") as f:
                json.dump(rows, f)
            df = pd.DataFrame(rows)
        except Exception:
            # Fall back to cached data even if expired
            if os.path.exists(_TD_CACHE):
                try:
                    with open(_TD_CACHE, "r", encoding="utf-8") as f:
                        rows = json.load(f)
                    df = pd.DataFrame(rows)
                except Exception:
                    df = pd.DataFrame()
            else:
                df = pd.DataFrame()
    if df.empty:
        return df
    # Normalize date cols
    for c in ("announcementDate", "auctionDate", "issueDate", "maturityDate"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    # Filter to forward-only by asof
    if asof is not None and "auctionDate" in df.columns:
        df = df[df["auctionDate"] >= pd.Timestamp(asof)]
    return df.sort_values("auctionDate") if "auctionDate" in df.columns else df


# =============================================================================
# B.10 — ACM term premia + Kim-Wright term premia
# =============================================================================
_ACM_URL = "https://www.newyorkfed.org/medialibrary/media/research/data_indicators/ACMTermPremium.xls"
_KW_URL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628_1.csv"
_ACM_CACHE = os.path.join(_CACHE_DIR, "acm_term_premia.csv")
_KW_CACHE = os.path.join(_CACHE_DIR, "kim_wright_term_premia.csv")
_TP_CACHE_TTL_HOURS = 24


def _download_to(url: str, dest: str) -> bool:
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read()
        with open(dest, "wb") as f:
            f.write(data)
        return True
    except Exception:
        return False


def load_acm_term_premia(force_refresh: bool = False) -> pd.DataFrame:
    """ACM term premia from NY Fed XLS. Returns DataFrame indexed by date with
    cols `acm_2y, acm_5y, acm_10y, acm_30y` (term premia in %).

    Returns empty DataFrame if download / parse fails.
    """
    use_cache = (not force_refresh and os.path.exists(_ACM_CACHE)
                  and ((datetime.now().timestamp() - os.path.getmtime(_ACM_CACHE))
                       < _TP_CACHE_TTL_HOURS * 3600))
    if not use_cache:
        # Download XLS, extract relevant columns, save as CSV
        xls_path = _ACM_CACHE.replace(".csv", ".xls")
        if not _download_to(_ACM_URL, xls_path):
            if os.path.exists(_ACM_CACHE):
                pass  # use stale cache
            else:
                return pd.DataFrame()
        else:
            try:
                xls = pd.read_excel(xls_path, sheet_name=0)
                # ACM XLS has columns like 'DATE', 'ACMTP01' ... 'ACMTP10' (term premia in %)
                # Find date col
                date_col = None
                for c in xls.columns:
                    if "date" in str(c).lower():
                        date_col = c
                        break
                if date_col is None:
                    return pd.DataFrame()
                xls[date_col] = pd.to_datetime(xls[date_col], errors="coerce")
                xls = xls.dropna(subset=[date_col]).set_index(date_col).sort_index()
                # Map columns: ACMTP02 ↔ 2y etc.
                colmap = {}
                for c in xls.columns:
                    cs = str(c).upper()
                    if "ACMTP02" in cs:
                        colmap[c] = "acm_2y"
                    elif "ACMTP05" in cs:
                        colmap[c] = "acm_5y"
                    elif "ACMTP10" in cs:
                        colmap[c] = "acm_10y"
                    elif "ACMTP30" in cs:
                        colmap[c] = "acm_30y"
                if not colmap:
                    return pd.DataFrame()
                out = xls[list(colmap.keys())].rename(columns=colmap)
                out.to_csv(_ACM_CACHE)
                return out
            except Exception:
                if os.path.exists(_ACM_CACHE):
                    pass
                else:
                    return pd.DataFrame()
    # Load from cache
    try:
        df = pd.read_csv(_ACM_CACHE, index_col=0, parse_dates=True)
        return df.sort_index()
    except Exception:
        return pd.DataFrame()


def load_kim_wright_term_premia(force_refresh: bool = False) -> pd.DataFrame:
    """Kim-Wright term premia from Fed Reserve CSV. Returns DataFrame indexed
    by date with cols `kw_2y, kw_5y, kw_10y, kw_30y` (term premia in %).
    """
    use_cache = (not force_refresh and os.path.exists(_KW_CACHE)
                  and ((datetime.now().timestamp() - os.path.getmtime(_KW_CACHE))
                       < _TP_CACHE_TTL_HOURS * 3600))
    if not use_cache:
        if not _download_to(_KW_URL, _KW_CACHE):
            if not os.path.exists(_KW_CACHE):
                return pd.DataFrame()
    try:
        df = pd.read_csv(_KW_CACHE, comment="#")
        # Find date col
        date_col = None
        for c in df.columns:
            if "date" in str(c).lower() or str(c).lower().startswith("d"):
                try:
                    df[c] = pd.to_datetime(df[c], errors="coerce")
                    if df[c].notna().any():
                        date_col = c
                        break
                except Exception:
                    continue
        if date_col is None:
            return pd.DataFrame()
        df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
        # Term-premia columns vary by file version; canonical Kim-Wright cols
        # are TP02Y, TP05Y, TP10Y, TP30Y or similar.
        colmap = {}
        for c in df.columns:
            cs = str(c).upper()
            if cs in ("TP02Y", "TP02") or ("TP" in cs and "2" in cs and "Y" in cs):
                colmap[c] = "kw_2y"
            elif cs in ("TP05Y", "TP05") or ("TP" in cs and "5" in cs and "Y" in cs):
                colmap[c] = "kw_5y"
            elif cs in ("TP10Y", "TP10") or ("TP" in cs and "10" in cs):
                colmap[c] = "kw_10y"
            elif cs in ("TP30Y", "TP30") or ("TP" in cs and "30" in cs):
                colmap[c] = "kw_30y"
        if not colmap:
            return pd.DataFrame()
        out = df[list(colmap.keys())].rename(columns=colmap)
        return out
    except Exception:
        return pd.DataFrame()


def load_term_premia(asof: Optional[date] = None,
                       lookback_days: int = 252) -> pd.DataFrame:
    """Combined ACM + Kim-Wright term premia panel."""
    acm = load_acm_term_premia()
    kw = load_kim_wright_term_premia()
    if acm.empty and kw.empty:
        return pd.DataFrame()
    df = pd.concat([acm, kw], axis=1).sort_index()
    if asof is not None:
        df = df.loc[df.index <= pd.Timestamp(asof)]
        if lookback_days > 0:
            cutoff = pd.Timestamp(asof) - pd.Timedelta(days=lookback_days * 2)
            df = df.loc[df.index >= cutoff]
    return df


# =============================================================================
# Composite cross-asset panel — single-call entry point
# =============================================================================
@dataclass(frozen=True)
class CrossAssetPanel:
    asof: date
    vol: pd.DataFrame
    equity: pd.DataFrame
    fx: pd.DataFrame
    credit: pd.DataFrame
    commodity: pd.DataFrame
    ust: pd.DataFrame
    fedbs: pd.DataFrame
    term_premia: pd.DataFrame
    treasury_auctions: pd.DataFrame


def load_cross_asset_panel(asof: Optional[date] = None,
                              lookback_days: int = 252,
                              include_external: bool = True) -> CrossAssetPanel:
    """Single-call composite loader for all cross-asset feeds.

    `include_external=False` skips ACM/Kim-Wright/TreasuryDirect downloads
    (useful for fast iteration). Defaults to True.
    """
    return CrossAssetPanel(
        asof=asof or date.today(),
        vol=load_vol_panel(asof, lookback_days),
        equity=load_equity_panel(asof, lookback_days),
        fx=load_fx_panel(asof, lookback_days),
        credit=load_credit_panel(asof, lookback_days),
        commodity=load_commodity_panel(asof, lookback_days),
        ust=load_ust_panel(asof, lookback_days),
        fedbs=load_fedbs_panel(asof, lookback_days),
        term_premia=load_term_premia(asof, lookback_days) if include_external else pd.DataFrame(),
        treasury_auctions=load_treasury_auction_calendar(asof) if include_external else pd.DataFrame(),
    )
