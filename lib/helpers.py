"""Common helpers."""
from __future__ import annotations

import streamlit as st


def page_header(title: str, subtitle: str | None = None) -> None:
    """Standard page header."""
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def placeholder(name: str, note: str = "") -> None:
    """Placeholder block for tabs not yet built."""
    st.info(f"**{name}** — placeholder. {note}")
