"""US — Bonds (TU, FV, TY, AUL) sub-tab."""
from __future__ import annotations

import streamlit as st

from lib.connections import get_ohlc_connection


def render() -> None:
    st.markdown(
        """
        <div style="display:flex; align-items:baseline; gap:0.75rem; margin-top:0.25rem;">
            <span style="font-size:1.1rem; font-weight:600; color:var(--text-heading);">Bonds</span>
            <span style="color:var(--text-dim); font-size:0.85rem;">·</span>
            <span style="color:var(--text-muted); font-size:0.85rem;">US Treasury futures — 2Y (TU) · 5Y (FV) · 10Y (TY) · 30Y Ultra (AUL)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")

    con = get_ohlc_connection()
    if con is None:
        st.error("OHLC database not available.")
        return

    try:
        cov = con.execute("""
            SELECT base_product, "interval",
                   COUNT(DISTINCT symbol) AS symbols,
                   COUNT(*) AS bars,
                   to_timestamp(MIN("time")/1000.0) AS oldest,
                   to_timestamp(MAX("time")/1000.0) AS newest
            FROM mde2_timeseries
            WHERE base_product IN ('TU','FV','TY','AUL') AND calc_method='api'
            GROUP BY 1, 2 ORDER BY 1, 2
        """).fetchdf()
        st.dataframe(cov, use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Coverage query failed: {e}")

    st.info("Bond analytics — to be built after the STIR tabs.")
