"""US — Fundamentals (BBG macro panel) sub-tab.

Comprehensive inventory of US-relevant BBG tickers, organized into categories
and sub-categories with one-line analytical use cases for STIR/Bond decisions.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from lib.components import status_strip
from lib.us_fundamentals_inventory import (
    CATEGORIES,
    COT_FIELD_LEGEND,
    US_TICKERS,
    get_category_summary,
    get_inventory_dataframe,
)


def _render_category(df: pd.DataFrame, category: str) -> None:
    """Render a single category section — sub-grouped by subcategory."""
    sub = df[df["category"] == category].copy()
    if sub.empty:
        st.info(f"No tickers in `{category}` yet.")
        return

    n = len(sub)
    sub_groups = sub["subcategory"].unique().tolist()
    st.markdown(
        f"<div style='margin-top:0.5rem;'>"
        f"<span style='color:var(--accent); font-size:0.95rem; font-weight:600;'>{category}</span>"
        f" &nbsp;<span style='color:var(--text-dim); font-size:0.78rem; "
        f"font-family:JetBrains Mono, monospace;'>{n} tickers · {len(sub_groups)} sub-groups</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    for subcat in sub_groups:
        rows = sub[sub["subcategory"] == subcat]
        with st.expander(f"{subcat}  ({len(rows)})", expanded=False):
            display = rows[["ticker", "name", "analysis", "bucket"]].copy()
            display.columns = ["Ticker", "Name", "Analysis", "BBG bucket"]
            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Ticker": st.column_config.TextColumn(width="medium"),
                    "Name": st.column_config.TextColumn(width="large"),
                    "Analysis": st.column_config.TextColumn(width="large"),
                    "BBG bucket": st.column_config.TextColumn(width="small"),
                },
            )


def render() -> None:
    # Tab title
    st.markdown(
        """
        <div style="display:flex; align-items:baseline; gap:0.75rem; margin-top:0.25rem;">
            <span style="font-size:1.1rem; font-weight:600; color:var(--text-heading);">Fundamentals</span>
            <span style="color:var(--text-dim); font-size:0.85rem;">·</span>
            <span style="color:var(--text-muted); font-size:0.85rem;">
                US BBG fundamental data — every ticker categorized with analytical use case for STIR / Bond decisions
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")

    df = get_inventory_dataframe()
    summary = get_category_summary()

    # Status strip — totals
    status_strip([
        ("Tickers", str(len(df)), "accent"),
        ("Categories", str(len(summary)), None),
        ("Sub-groups", str(df["subcategory"].nunique()), None),
        ("BBG buckets", str(df["bucket"].nunique()), None),
    ])

    st.markdown("")

    # ---- Search + filter row ----
    c1, c2, c3 = st.columns([3, 2, 1])
    with c1:
        search = st.text_input(
            "Search (ticker, name, or analysis)",
            placeholder="e.g. CPI, FDTR, auction, JOLTS…",
            key="fund_search",
            label_visibility="collapsed",
        ).strip().lower()
    with c2:
        bucket_filter = st.multiselect(
            "BBG bucket filter",
            options=sorted(df["bucket"].unique().tolist()),
            default=[],
            key="fund_bucket_filter",
            label_visibility="collapsed",
            placeholder="Filter by BBG bucket",
        )
    with c3:
        view_mode = st.selectbox(
            "View",
            ["By category", "Flat table"],
            key="fund_view_mode",
            label_visibility="collapsed",
        )

    # Apply filters
    fdf = df.copy()
    if search:
        mask = (
            fdf["ticker"].str.lower().str.contains(search, na=False)
            | fdf["name"].str.lower().str.contains(search, na=False)
            | fdf["analysis"].str.lower().str.contains(search, na=False)
            | fdf["category"].str.lower().str.contains(search, na=False)
            | fdf["subcategory"].str.lower().str.contains(search, na=False)
        )
        fdf = fdf[mask]
    if bucket_filter:
        fdf = fdf[fdf["bucket"].isin(bucket_filter)]

    if fdf.empty:
        st.warning("No tickers match the current filters.")
        return

    if search or bucket_filter:
        st.caption(
            f"Showing **{len(fdf)}** of {len(df)} tickers across "
            f"**{fdf['category'].nunique()}** categories."
        )

    st.markdown("")

    # ---- Category quick-jump pills ----
    if view_mode == "By category":
        # Category nav as clickable buttons (filter to a single category)
        present_cats = [c for c in CATEGORIES if c in fdf["category"].unique()]
        cat_pick_default = "All categories"
        cat_options = [cat_pick_default] + present_cats
        col_a, col_b = st.columns([1, 4])
        with col_a:
            picked = st.selectbox(
                "Category",
                options=cat_options,
                key="fund_cat_pick",
                label_visibility="collapsed",
            )
        with col_b:
            counts = " · ".join(
                f"<span style='color:var(--text-dim);'>{c}</span> "
                f"<span style='color:var(--accent); font-family:JetBrains Mono, monospace;'>{summary.get(c, 0)}</span>"
                for c in present_cats
            )
            st.markdown(
                f"<div style='font-size:0.75rem; line-height:1.6; padding-top:0.4rem;'>{counts}</div>",
                unsafe_allow_html=True,
            )

        st.markdown("")
        if picked == cat_pick_default:
            for cat in present_cats:
                _render_category(fdf, cat)
                st.markdown("")
        else:
            _render_category(fdf, picked)

    else:  # Flat table
        display = fdf[["ticker", "category", "subcategory", "name", "analysis", "bucket"]].copy()
        display.columns = ["Ticker", "Category", "Sub-category", "Name", "Analysis", "Bucket"]
        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
            height=600,
            column_config={
                "Ticker": st.column_config.TextColumn(width="medium"),
                "Category": st.column_config.TextColumn(width="medium"),
                "Sub-category": st.column_config.TextColumn(width="medium"),
                "Name": st.column_config.TextColumn(width="large"),
                "Analysis": st.column_config.TextColumn(width="large"),
                "Bucket": st.column_config.TextColumn(width="small"),
            },
        )

    # ---- COT field legend ----
    st.markdown("")
    with st.expander("CFTC COT — field code legend", expanded=False):
        legend_df = pd.DataFrame(COT_FIELD_LEGEND, columns=["Field code", "Meaning"])
        st.dataframe(legend_df, use_container_width=True, hide_index=True)
        st.caption(
            "COT data is weekly (Tuesday cut, released Friday). Each Treasury futures product has the "
            "11 field codes above; each options-on-futures product has 10 (no NCN). To pull a specific "
            "ticker, append the field code to the prefix — e.g. `CBT4TNCN Index` = 10Y T-Note Non-Commercial Net."
        )
