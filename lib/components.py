"""Reusable UI components — page header, status strip, summary row, KPI chips."""
from __future__ import annotations

from typing import Optional

import streamlit as st


def page_header(title: str, subtitle: str = "", meta: str = "") -> None:
    """Slim page header with title, subtitle, and right-aligned meta.

    Renders as a single horizontal bar with bottom border."""
    st.markdown(
        f"""
        <div class="page-header">
            <div>
                <div class="page-title">{title}</div>
                {('<div class="page-subtitle">' + subtitle + '</div>') if subtitle else ''}
            </div>
            <div class="page-meta">{meta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_strip(items: list[tuple[str, str, Optional[str]]]) -> None:
    """Horizontal status strip of label/value pairs.

    items: list of (label, value, color_class) where color_class ∈ {None, "accent", "green", "red"}.
    """
    parts = []
    for tup in items:
        label, value = tup[0], tup[1]
        color = tup[2] if len(tup) > 2 and tup[2] else ""
        parts.append(
            f'<div class="item">'
            f'<span class="label">{label}</span>'
            f'<span class="value {color}">{value}</span>'
            f'</div>'
        )
    st.markdown(f'<div class="status-strip">{"".join(parts)}</div>', unsafe_allow_html=True)


def status_strip_with_dot(items: list[tuple[str, str, Optional[str]]],
                          dot_label: str = "Live") -> None:
    """Status strip with a pulsing live dot at the start."""
    parts = [
        '<div class="item">'
        f'<span class="live-dot"></span>'
        f'<span class="value green">{dot_label}</span>'
        '</div>'
    ]
    for tup in items:
        label, value = tup[0], tup[1]
        color = tup[2] if len(tup) > 2 and tup[2] else ""
        parts.append(
            f'<div class="item">'
            f'<span class="label">{label}</span>'
            f'<span class="value {color}">{value}</span>'
            f'</div>'
        )
    st.markdown(f'<div class="status-strip">{"".join(parts)}</div>', unsafe_allow_html=True)


def kpi_chip(label: str, value: str, color: str = "") -> str:
    """Return HTML for a KPI chip (compose into status_strip or markdown)."""
    cls = f"kpi-chip {color}" if color else "kpi-chip"
    return f'<span class="{cls}"><span class="label">{label}</span><span class="val">{value}</span></span>'


def summary_row(stats: list[tuple[str, str, Optional[str], Optional[str]]]) -> None:
    """Render a summary row of stats.

    stats: list of (label, value, color_class | None, sub_text | None)
    """
    parts = []
    for tup in stats:
        label, value = tup[0], tup[1]
        color = tup[2] if len(tup) > 2 and tup[2] else ""
        sub = tup[3] if len(tup) > 3 and tup[3] else ""
        sub_html = f'<span class="sub">{sub}</span>' if sub else ""
        parts.append(
            f'<div class="stat">'
            f'<span class="lbl">{label}</span>'
            f'<span class="val {color}">{value}</span>'
            f'{sub_html}'
            f'</div>'
        )
    st.markdown(f'<div class="summary-row">{"".join(parts)}</div>', unsafe_allow_html=True)


def section_card(title: str, body_html: str = "", meta: str = "") -> None:
    """Compact section card — title + optional meta + arbitrary body HTML."""
    meta_html = f'<div class="meta">{meta}</div>' if meta else ""
    st.markdown(
        f"""
        <div class="section-card">
            <div class="head">
                <div class="title">{title}</div>
                {meta_html}
            </div>
            {body_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
