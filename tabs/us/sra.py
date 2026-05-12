"""SRA — main router with 4 sub-sub-tabs (Curve, Analysis, PCA, Historical Event Impact).

Parameterized via `base_product` — every market (SRA / ER / FSR / FER / SON / YBA / CRA)
re-uses the SAME UI, only the data source swaps. Each sub-tab module accepts
`base_product` and sets its module-level active-market shim accordingly.
"""
from __future__ import annotations

import streamlit as st

from lib.connections import get_ohlc_connection
from lib.markets import get_market
from tabs.us import sra_analysis, sra_curve, sra_event_impact, sra_pca


def render(base_product: str = "SRA") -> None:
    con = get_ohlc_connection()
    if con is None:
        st.error("OHLC database not available — see Settings → OHLC DB Viewer.")
        return

    market_cfg = get_market(base_product)
    description = market_cfg.get("description", base_product)

    # Market header — same compact style across all markets
    st.markdown(
        f"""
        <div style="display:flex; align-items:baseline; gap:0.75rem; margin-top:0.25rem;">
            <span style="font-size:1.1rem; font-weight:600; color:var(--text-heading);">{base_product}</span>
            <span style="color:var(--text-dim); font-size:0.85rem;">·</span>
            <span style="color:var(--text-muted); font-size:0.85rem;">
                {description} — outrights · calendar spreads · butterflies
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")

    # Sub-sub tabs — identical structure across all 7 markets
    sub_curve, sub_analysis, sub_pca, sub_event_impact = st.tabs([
        "Curve", "Analysis", "PCA", "Historical Event Impact",
    ])
    with sub_curve:
        try:
            sra_curve.render(base_product)
        except Exception as e:
            st.error(f"Curve subtab failed: {e}")
            with st.expander("Traceback"):
                import traceback as _tb
                st.code(_tb.format_exc())
    with sub_analysis:
        try:
            sra_analysis.render(base_product)
        except Exception as e:
            st.error(f"Analysis subtab failed: {e}")
            with st.expander("Traceback"):
                import traceback as _tb
                st.code(_tb.format_exc())
    with sub_pca:
        try:
            sra_pca.render(base_product=base_product)
        except Exception as e:
            st.error(f"PCA subtab failed: {e}")
            with st.expander("Traceback"):
                import traceback as _tb
                st.code(_tb.format_exc())
    with sub_event_impact:
        try:
            sra_event_impact.render(base_product=base_product)
        except Exception as e:
            st.error(f"Historical Event Impact subtab failed: {e}")
            with st.expander("Traceback"):
                import traceback as _tb
                st.code(_tb.format_exc())
