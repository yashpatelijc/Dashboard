"""Canada Economy — main page with 2 sub-tabs (CORRA + Fundamentals)."""
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
    page_title="Canada Economy — STIRS",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_global_css()
render_sidebar_brand()

# Pre-warm CORRA caches in background — runs once per server process.
ensure_prewarm("CRA")


page_header(
    title="🇨🇦 Canada",
    subtitle=("BoC policy analytics via 3-Month CORRA futures (CRA). "
                "Full PCA trade screener with mode toggle and dynamic exits."),
    meta=datetime.now().strftime("%a · %Y-%m-%d %H:%M"),
)

st.markdown(
    f"<div style='margin: -0.4rem 0 1rem 0;'>{freshness_header_chip()} "
    f"<span style='color:var(--text-dim); font-size:0.65rem; margin-left:0.5rem;'>"
    f"see Settings → System Health for full report</span></div>",
    unsafe_allow_html=True,
)


tab_cra, tab_fund = st.tabs(["CORRA", "Fundamentals"])

with tab_cra:
    market_dispatcher.render("CRA")

with tab_fund:
    fundamentals_tab.render("canada")
