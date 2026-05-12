"""Generic Fundamentals tab — renders any region's BBG ticker inventory.

Accepts a region code ('eurozone' | 'uk' | 'australia' | 'canada') and loads
the corresponding lib/{region}_fundamentals_inventory.py module, then renders
the same UI as `tabs/us/fundamentals.py`.
"""
from __future__ import annotations

import importlib

import pandas as pd
import streamlit as st


_REGION_MODULES = {
    "us": ("lib.us_fundamentals_inventory", "US_TICKERS", "United States"),
    "eurozone": ("lib.eurozone_fundamentals_inventory", "EUROZONE_TICKERS", "Eurozone"),
    "uk": ("lib.uk_fundamentals_inventory", "UK_TICKERS", "United Kingdom"),
    "australia": ("lib.australia_fundamentals_inventory", "AUSTRALIA_TICKERS", "Australia"),
    "canada": ("lib.canada_fundamentals_inventory", "CANADA_TICKERS", "Canada"),
}


def _to_dataframe(tickers: list) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame(columns=["ticker", "bucket", "category", "subcategory",
                                       "name", "analysis"])
    return pd.DataFrame(tickers)


def render(region: str) -> None:
    """Render the BBG-fundamentals inventory for `region`."""
    if region not in _REGION_MODULES:
        st.error(f"Unknown region: {region}. Known: {list(_REGION_MODULES.keys())}")
        return
    mod_name, ticker_attr, label = _REGION_MODULES[region]
    try:
        mod = importlib.import_module(mod_name)
    except Exception as e:
        st.error(f"Could not import {mod_name}: {e}")
        return
    tickers = getattr(mod, ticker_attr, [])
    categories = getattr(mod, "CATEGORIES", [])
    df = _to_dataframe(tickers)

    st.markdown(
        f"<div style='margin:0.25rem 0 0.5rem;'>"
        f"<span style='font-size:1.0rem; font-weight:600; color:var(--text-heading);'>"
        f"{label} Fundamentals</span> "
        f"<span style='color:var(--text-muted); font-size:0.78rem; "
        f"margin-left:0.5rem;'>{len(df)} BBG tickers · {len(categories)} categories</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if df.empty:
        st.info("No tickers in inventory.")
        return

    # Search box
    q = st.text_input("Filter (substring match on name/ticker/analysis)",
                          key=f"{region}_fund_q")
    filtered = df.copy()
    if q and q.strip():
        ql = q.strip().lower()
        mask = (filtered["name"].str.lower().str.contains(ql, na=False)
                | filtered["ticker"].str.lower().str.contains(ql, na=False)
                | filtered["analysis"].str.lower().str.contains(ql, na=False))
        filtered = filtered[mask]

    # Category sections
    cat_iter = categories if categories else sorted(filtered["category"].unique())
    for cat in cat_iter:
        sub = filtered[filtered["category"] == cat]
        if sub.empty:
            continue
        n = len(sub)
        sub_groups = sub["subcategory"].unique().tolist()
        st.markdown(
            f"<div style='margin-top:0.7rem;'>"
            f"<span style='color:var(--accent); font-size:0.95rem; font-weight:600;'>{cat}</span>"
            f" <span style='color:var(--text-dim); font-size:0.78rem; "
            f"font-family:JetBrains Mono, monospace;'>{n} tickers · {len(sub_groups)} sub-groups</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        for subcat in sub_groups:
            rows = sub[sub["subcategory"] == subcat]
            with st.expander(f"{subcat}  ({len(rows)})", expanded=False):
                display = rows[["ticker", "name", "analysis", "bucket"]].copy()
                display.columns = ["Ticker", "Name", "Analysis", "BBG bucket"]
                st.dataframe(display, use_container_width=True, hide_index=True)
