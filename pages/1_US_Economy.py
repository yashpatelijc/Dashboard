"""US Economy — main page with 4 sub-tabs."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from datetime import datetime

import streamlit as st

from lib.components import page_header
from lib.backtest.cycle import ensure_backtest_fresh
from lib.css import inject_global_css, render_sidebar_brand
from lib.freshness import freshness_header_chip
from lib.prewarm import ensure_prewarm
from tabs.us import bonds, ff_s1r, fundamentals, sra

st.set_page_config(
    page_title="US Economy — STIRS",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_global_css()
render_sidebar_brand()

# Kick off Technicals pre-warm if not already running (idempotent).
ensure_prewarm()
# Phase E: 10-day backtest recompute (idempotent; daemon thread; non-blocking).
ensure_backtest_fresh()

page_header(
    title="🇺🇸 United States",
    subtitle="STIR + Bond analytics. Switch markets via the sub-tabs.",
    meta=datetime.now().strftime("%a · %Y-%m-%d %H:%M"),
)

# Phase 0.C: data-freshness traffic light below header
st.markdown(
    f"<div style='margin: -0.4rem 0 1rem 0;'>{freshness_header_chip()} "
    f"<span style='color:var(--text-dim); font-size:0.65rem; margin-left:0.5rem;'>"
    f"see Settings → System Health for full report</span></div>",
    unsafe_allow_html=True,
)

tab_sra, tab_ff, tab_bonds, tab_fund = st.tabs([
    "SRA",
    "FF + S1R",
    "Bonds",
    "Fundamentals",
])

with tab_sra:
    sra.render()

with tab_ff:
    ff_s1r.render()

with tab_bonds:
    bonds.render()

with tab_fund:
    fundamentals.render()
