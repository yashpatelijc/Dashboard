"""SRA — Analysis subtab. Sub-router for derived analytics:

  · Proximity        — N-day high/low proximity, ATR-flagged ranking, confluence,
                        cluster signals, fresh/failed breaks.
  · Z-score & MR     — multi-lookback Z, OU half-life, Hurst, ADF, reversion-
                        candidate ranking, mean-reversion drill.
  · Technicals       — 26 setup detectors (TMIA-curated): A1/A2/A3/A4/A5/A6/A8/
                        A10/A11/A12a/A12b/A15 (trend), B1/B3/B5/B6/B10/B11/B13
                        (mean-reversion), C3/C4/C5/C8(12/24/36)/C9a/C9b
                        (STIR-specific), plus three composite scores (TREND/MR/
                        FINAL) re-tuned per scope (outright / spread / fly).
"""
from __future__ import annotations

import streamlit as st

from tabs.us import (
    sra_analysis_proximity,
    sra_analysis_technicals,
    sra_analysis_zscore,
)


# Tab labels — only three sub-subtabs now (placeholders removed)
_TAB_LABELS = [
    "Proximity",
    "Z-score & MR",
    "Technicals",
]


def render(base_product: str = "SRA") -> None:
    # Subtab header — same compact style as the SRA tab header
    st.markdown(
        f"""
        <div style="display:flex; align-items:baseline; gap:0.75rem; margin-top:0.25rem;">
            <span style="font-size:1.05rem; font-weight:600; color:var(--text-heading);">Analysis</span>
            <span style="color:var(--text-dim); font-size:0.85rem;">·</span>
            <span style="color:var(--text-muted); font-size:0.85rem;">
                Derived analytics from the {base_product} curve — proximity / mean-reversion / TMIA technical setups
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")

    tabs = st.tabs(_TAB_LABELS)
    with tabs[0]:
        try:
            sra_analysis_proximity.render(base_product)
        except Exception as e:
            st.error(f"Proximity subtab failed: {e}")
            with st.expander("Traceback"):
                import traceback as _tb
                st.code(_tb.format_exc())
    with tabs[1]:
        try:
            sra_analysis_zscore.render(base_product)
        except Exception as e:
            st.error(f"Z-score & MR subtab failed: {e}")
            with st.expander("Traceback"):
                import traceback as _tb
                st.code(_tb.format_exc())
    with tabs[2]:
        try:
            sra_analysis_technicals.render(base_product)
        except Exception as e:
            st.error(f"Technicals subtab failed: {e}")
            with st.expander("Traceback"):
                import traceback as _tb
                st.code(_tb.format_exc())
