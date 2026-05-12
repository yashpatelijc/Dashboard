"""Per-contract unit-convention detection and catalog.

Spread/fly contracts in the OHLC database may be stored either as:
  - "price"  — raw price difference in the underlying's price units
               (e.g. SRAH26 - SRAM26 = 96.345 - 96.37 = -0.025)
  - "bp"     — already scaled to basis points
               (e.g. SRAM26-U26 close = -2.5  →  -0.025 × 100)

Outrights are always in price units.

This module:
  1. AUTO-DETECTS the convention per contract by comparing each spread/fly's
     stored close against the implied price-difference computed from its
     underlying outright legs.
  2. PERSISTS the result in a parquet catalog at data/contract_units.parquet
     so it can be reused without re-running detection on every page load.
  3. Provides helpers — to_bp(), is_already_bp() — for analysis code to
     normalise values regardless of stored convention.

Catalog schema:
    base_product   str
    symbol         str
    strategy       str   ('outright' | 'spread' | 'fly')
    convention     str   ('bp' | 'price' | 'unknown')
    median_ratio   float (close_stored / close_implied; ~1.0 → price, ~100 → bp)
    n_samples      int   number of (date, leg-set) joined samples used
    legs           str   comma-joined leg symbols (or '' for outrights)
    built_at       str   ISO timestamp
"""
from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Optional

import duckdb
import numpy as np
import pandas as pd
import streamlit as st

from lib.connections import get_ohlc_connection


CATALOG_PATH = r"D:\STIRS_DASHBOARD\data\contract_units.parquet"

# Detection thresholds
_RATIO_BP_LOW   = 30.0      # |median_ratio| in [30, 300]  → bp convention
_RATIO_BP_HIGH  = 300.0
_RATIO_PRICE_LOW  = 0.3     # |median_ratio| in [0.3, 3.0] → price convention
_RATIO_PRICE_HIGH = 3.0
_MIN_SAMPLES = 5            # need at least N joined samples to decide
_MIN_RAW_ABS = 1e-3         # skip near-zero raw differences (in price units)
_MIN_STORED_ABS = 1e-6      # skip exact-zero stored closes (no print / off-day)
_RECENT_N = 120             # use last N joined samples for detection (recency bias)


# -----------------------------------------------------------------------------
# Symbol parsing — covers SRA-style symbols like 'SRAH26', 'SRAH26-M26', 'SRAH26-M26-U26'
# -----------------------------------------------------------------------------
_LEG_RE = re.compile(r"^([A-Z]+)([FGHJKMNQUVXZ])(\d{2})$")  # base + month-letter + 2-digit year
_SUFFIX_RE = re.compile(r"^([FGHJKMNQUVXZ])(\d{2})$")


def parse_legs(symbol: str, base_product: str) -> list[str]:
    """Return the underlying outright leg symbols for a spread/fly symbol.

    Supports the common 'BASEMy-My-...' format used by SRA/FF/S1R/etc:
      'SRAH26'         → ['SRAH26']
      'SRAH26-M26'     → ['SRAH26', 'SRAM26']
      'SRAH26-M26-U26' → ['SRAH26', 'SRAM26', 'SRAU26']
    Returns [] if the symbol can't be parsed.
    """
    if not symbol:
        return []
    parts = symbol.split("-")
    if not parts:
        return []
    head = parts[0]
    m = _LEG_RE.match(head)
    if not m:
        return [symbol]  # outright (or unknown shape) — return as-is
    base, _, _ = m.group(1), m.group(2), m.group(3)
    if base != base_product:
        # Different convention — bail safely
        return [symbol]
    legs = [head]
    for suffix in parts[1:]:
        sm = _SUFFIX_RE.match(suffix)
        if not sm:
            return []
        legs.append(f"{base}{sm.group(1)}{sm.group(2)}")
    return legs


# -----------------------------------------------------------------------------
# Detection — compute per-contract unit convention from data
# -----------------------------------------------------------------------------
def _join_panel_for_legs(con, legs: list[str]) -> pd.DataFrame:
    """Pull all (date, symbol → close) for the given leg symbols. Long-form."""
    if not legs:
        return pd.DataFrame()
    leg_list = ",".join(f"'{s}'" for s in legs)
    sql = f"""
    SELECT DATE(to_timestamp(t.time/1000.0)) AS bar_date, t.symbol, t.close
    FROM mde2_timeseries t
    WHERE t.symbol IN ({leg_list})
      AND t."interval"='1D' AND t.calc_method='api'
    """
    df = con.execute(sql).fetchdf()
    if df.empty:
        return df
    return df.pivot_table(index="bar_date", columns="symbol", values="close", aggfunc="first")


def _expected_raw_diff(strategy: str, leg_prices: pd.DataFrame, legs: list[str]) -> pd.Series:
    """Compute the implied raw price-difference series from leg prices.

    spread legs A,B   →  price_A - price_B
    fly    legs A,B,C →  price_A - 2*price_B + price_C
    Returns NaN where any leg is missing.
    """
    cols = [c for c in legs if c in leg_prices.columns]
    if len(cols) != len(legs):
        return pd.Series(dtype=float)
    if strategy == "spread" and len(legs) == 2:
        return leg_prices[legs[0]] - leg_prices[legs[1]]
    if strategy == "fly" and len(legs) == 3:
        return leg_prices[legs[0]] - 2.0 * leg_prices[legs[1]] + leg_prices[legs[2]]
    return pd.Series(dtype=float)


def detect_convention(con, base_product: str, symbol: str, strategy: str) -> dict:
    """Detect 'bp' vs 'price' for a single spread/fly contract.

    Returns dict with keys:
      base_product, symbol, strategy, convention, median_ratio, n_samples, legs
    """
    if strategy == "outright":
        # Outrights are always in price units of the underlying
        return {
            "base_product": base_product, "symbol": symbol, "strategy": "outright",
            "convention": "price", "median_ratio": 1.0, "n_samples": 0, "legs": "",
        }

    legs = parse_legs(symbol, base_product)
    if (strategy == "spread" and len(legs) != 2) or (strategy == "fly" and len(legs) != 3):
        return {
            "base_product": base_product, "symbol": symbol, "strategy": strategy,
            "convention": "unknown", "median_ratio": float("nan"), "n_samples": 0,
            "legs": ",".join(legs),
        }

    # Pull closes for legs + the contract itself
    panel = _join_panel_for_legs(con, legs + [symbol])
    if panel.empty or symbol not in panel.columns:
        return {
            "base_product": base_product, "symbol": symbol, "strategy": strategy,
            "convention": "unknown", "median_ratio": float("nan"), "n_samples": 0,
            "legs": ",".join(legs),
        }

    raw = _expected_raw_diff(strategy, panel, legs)
    stored = panel[symbol]
    df = pd.concat({"raw": raw, "stored": stored}, axis=1).dropna()
    # Filter out near-zero raw OR stored — these are off-days that distort the ratio.
    df = df[(df["raw"].abs() >= _MIN_RAW_ABS) & (df["stored"].abs() >= _MIN_STORED_ABS)]
    # Recency bias — old data may be sparse / settlement-imprecise; trust last N samples.
    if len(df) > _RECENT_N:
        df = df.iloc[-_RECENT_N:]
    if len(df) < _MIN_SAMPLES:
        return {
            "base_product": base_product, "symbol": symbol, "strategy": strategy,
            "convention": "unknown", "median_ratio": float("nan"),
            "n_samples": int(len(df)), "legs": ",".join(legs),
        }

    ratios = (df["stored"] / df["raw"]).to_numpy(dtype=float)
    med = float(np.median(ratios))
    a = abs(med)

    if _RATIO_BP_LOW <= a <= _RATIO_BP_HIGH:
        conv = "bp"
    elif _RATIO_PRICE_LOW <= a <= _RATIO_PRICE_HIGH:
        conv = "price"
    else:
        conv = "unknown"

    return {
        "base_product": base_product, "symbol": symbol, "strategy": strategy,
        "convention": conv, "median_ratio": med, "n_samples": int(len(df)),
        "legs": ",".join(legs),
    }


# -----------------------------------------------------------------------------
# Catalog — build, save, load
# -----------------------------------------------------------------------------
def list_market_symbols(con, base_product: str) -> pd.DataFrame:
    """Return all (symbol, strategy) for a base_product from the catalog table."""
    sql = f"""
    SELECT DISTINCT cc.symbol, cc.strategy
    FROM mde2_contracts_catalog cc
    WHERE cc.is_inter = FALSE
      AND cc.base_product = '{base_product}'
      AND cc.strategy IN ('outright','spread','fly')
    ORDER BY cc.strategy, cc.symbol
    """
    return con.execute(sql).fetchdf()


def _market_dominant_convention(rows: list[dict], strategy: str) -> Optional[str]:
    """Return the strategy's dominant convention within `rows`, or None if unclear."""
    counts = {}
    for r in rows:
        if r["strategy"] != strategy:
            continue
        c = r["convention"]
        if c in ("bp", "price"):
            counts[c] = counts.get(c, 0) + 1
    if not counts:
        return None
    top = max(counts.items(), key=lambda kv: kv[1])
    total = sum(counts.values())
    # Require at least 70% dominance and at least 5 confidently-classified rows
    if total >= 5 and top[1] / total >= 0.7:
        return top[0]
    return None


def build_catalog(base_products: Optional[list[str]] = None,
                  progress_cb=None) -> pd.DataFrame:
    """Build (or rebuild) the unit-convention catalog for given markets.

    Per-contract detection is attempted first. If a spread/fly contract can't be
    classified individually (insufficient samples or unstable ratios), it inherits
    the dominant convention of its (base_product, strategy) cohort — but only when
    that cohort shows ≥70% agreement.

    If base_products is None, uses ['SRA'] by default. Returns the new catalog DF.
    """
    if base_products is None:
        base_products = ["SRA"]

    con = get_ohlc_connection()
    if con is None:
        return pd.DataFrame()

    rows = []
    for bp in base_products:
        syms = list_market_symbols(con, bp)
        n = len(syms)
        market_rows = []
        for i, r in enumerate(syms.itertuples(index=False)):
            rec = detect_convention(con, bp, r.symbol, r.strategy)
            rec["built_at"] = datetime.utcnow().isoformat(timespec="seconds")
            rec["inferred_from"] = "direct"
            market_rows.append(rec)
            if progress_cb is not None and (i % 50 == 0 or i == n - 1):
                progress_cb(bp, i + 1, n)

        # Cohort fallback: fill in 'unknown' rows with market-strategy dominant convention.
        for strat in ("spread", "fly"):
            dom = _market_dominant_convention(market_rows, strat)
            if dom is None:
                continue
            for rec in market_rows:
                if rec["strategy"] == strat and rec["convention"] == "unknown":
                    rec["convention"] = dom
                    rec["inferred_from"] = f"cohort:{strat}"
        rows.extend(market_rows)

    df = pd.DataFrame(rows)
    if not df.empty:
        os.makedirs(os.path.dirname(CATALOG_PATH), exist_ok=True)
        df.to_parquet(CATALOG_PATH, index=False)
    return df


@st.cache_data(show_spinner=False, ttl=3600)
def load_catalog() -> pd.DataFrame:
    """Load the persisted catalog. Returns empty DF if not built yet."""
    if not os.path.exists(CATALOG_PATH):
        return pd.DataFrame()
    try:
        return pd.read_parquet(CATALOG_PATH)
    except Exception:
        try:
            con = duckdb.connect(":memory:")
            return con.execute(f"SELECT * FROM read_parquet('{CATALOG_PATH}')").fetchdf()
        except Exception:
            return pd.DataFrame()


def get_convention(symbol: str, base_product: str = "SRA",
                    catalog: Optional[pd.DataFrame] = None) -> str:
    """Return 'bp' / 'price' / 'unknown' for a symbol. Falls back to 'price' for outrights
    and 'unknown' for unparseable spreads/flies.
    """
    if catalog is None:
        catalog = load_catalog()
    if catalog.empty:
        # No catalog → safe defaults
        if symbol and "-" not in symbol:
            return "price"
        return "unknown"
    row = catalog[(catalog["base_product"] == base_product) & (catalog["symbol"] == symbol)]
    if row.empty:
        if symbol and "-" not in symbol:
            return "price"
        return "unknown"
    return str(row.iloc[0]["convention"])


def is_already_bp(symbol: str, base_product: str = "SRA",
                   catalog: Optional[pd.DataFrame] = None) -> bool:
    """True when the contract's stored close is already in basis points."""
    return get_convention(symbol, base_product, catalog) == "bp"


def bp_multiplier(symbol: str, base_product: str = "SRA",
                   catalog: Optional[pd.DataFrame] = None) -> float:
    """Multiplier to convert a stored value (or its delta) to basis points.

      convention 'price'   → ×100  (price diff → bp)
      convention 'bp'      → ×1    (already bp)
      convention 'unknown' → ×100  (treat conservatively as price; safer for outrights)
    """
    conv = get_convention(symbol, base_product, catalog)
    if conv == "bp":
        return 1.0
    return 100.0


def bp_multipliers_for(contracts: list[str], base_product: str = "SRA",
                        catalog: Optional[pd.DataFrame] = None) -> list[float]:
    """Vector of bp multipliers aligned to `contracts`."""
    if catalog is None:
        catalog = load_catalog()
    return [bp_multiplier(c, base_product, catalog) for c in contracts]
