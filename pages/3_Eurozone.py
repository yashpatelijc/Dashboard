"""Eurozone — main page with 4 sub-tabs (ER · FSR · FER · Fundamentals).

Mirrors the US Economy page layout. Each market sub-tab uses the generic
market dispatcher to surface the same 4 inner views (Curve / Analysis /
PCA / Historical Event Impact) parameterized for the market code.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datetime import datetime

import streamlit as st

from lib.components import page_header
from lib.css import inject_global_css, render_sidebar_brand
from lib.freshness import freshness_header_chip
from lib.prewarm import ensure_prewarm
from tabs.markets import market_dispatcher, fundamentals_tab


st.set_page_config(
    page_title="Eurozone — STIRS",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_global_css()
render_sidebar_brand()

# Pre-warm scan caches for every Eurozone market on first page load — runs
# in background daemon threads, idempotent. By the time the user clicks
# any market sub-tab the PCA panel + Technicals scans are already cached.
for _code in ("ER", "FSR", "FER"):
    ensure_prewarm(_code)


page_header(
    title="🇪🇺 Eurozone",
    subtitle=("STIR analytics across Euribor (ER), SARON (FSR), and €STR-style "
                "FER markets. Mode toggle, dynamic exits, and the full PCA "
                "trade screener available inside each market sub-tab."),
    meta=datetime.now().strftime("%a · %Y-%m-%d %H:%M"),
)


# Freshness indicator
st.markdown(
    f"<div style='margin: -0.4rem 0 1rem 0;'>{freshness_header_chip()} "
    f"<span style='color:var(--text-dim); font-size:0.65rem; margin-left:0.5rem;'>"
    f"see Settings → System Health for full report</span></div>",
    unsafe_allow_html=True,
)


tab_er, tab_fsr, tab_fer, tab_fund = st.tabs([
    "ER",
    "FSR",
    "FER",
    "Fundamentals",
])

with tab_er:
    market_dispatcher.render("ER")

with tab_fsr:
    market_dispatcher.render("FSR")

with tab_fer:
    market_dispatcher.render("FER")

with tab_fund:
    fundamentals_tab.render("eurozone")
