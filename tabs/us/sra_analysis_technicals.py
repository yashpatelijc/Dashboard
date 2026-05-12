"""SRA → Analysis → Technicals subtab.

Five views + 12-block side panel + drill-down. Setup catalog: 11 trend (A1, A2,
A3, A4, A5, A6, A8, A10, A11, A12a, A12b, A15) + 7 mean-reversion (B1, B3, B5,
B6, B10, B11, B13) + STIR-specific (C3, C4, C5, C8 ×3, C9a, C9b) + three
composite scores (TREND/MR/FINAL) re-tuned per scope.

All numeric cells carry threshold tooltips with ✓ markers on the matching
bucket — same pattern locked in Proximity / Z-score & MR subtabs. Reuses
``lib.mean_reversion.metric_tooltip()`` with the per-scope guide built by
``lib.setups.interpretation_guide.get_full_guide_for_scope()``.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.components import status_strip_with_dot
from lib.mean_reversion import metric_tooltip
from lib.prewarm import ensure_prewarm, get_prewarm_status, is_prewarm_done
from lib.setups.base import fmt_price_for_scope, round_to_tick
from lib.setups.composite import compute_composites
from lib.setups.interpretation_guide import (
    get_composite_guide_fly,
    get_composite_guide_outright,
    get_composite_guide_spread,
    get_full_guide_for_scope,
    get_technicals_interpretation_guide,
)
from lib.setups.registry import (
    ALL_SETUP_IDS, COMPOSITE_IDS,
    FAMILY_COMPOSITE, FAMILY_MR, FAMILY_STIR, FAMILY_TREND,
    MR_IDS, STIR_IDS, TREND_IDS,
    SETUP_REGISTRY, applies_to, display_name, display_name_short,
    get_registry_entry, setups_for_scope,
)
from lib.setups.tooltips import (
    setup_six_section_tooltip, composite_cell_tooltip,
    composite_header_tooltip, regime_header_tooltip, take_header_tooltip,
)
from lib.setups.scan import scan_universe
from lib.sra_data import (
    DEFAULT_FRONT_END, DEFAULT_MID_END, LIVENESS_DAYS,
    compute_section_split,
    contract_range_str,
    get_available_tenors as _sra_get_available_tenors,
    get_contract_history,
    get_flies as _sra_get_flies,
    get_outrights as _sra_get_outrights,
    get_spreads as _sra_get_spreads,
    get_sra_snapshot_latest_date,
)
from lib import market_data as _md


# ─── Active-market shim ─────────────────────────────────────────────────────
_BP = "SRA"


def _set_market(base_product: str) -> None:
    global _BP
    _BP = base_product


def get_outrights():
    return _sra_get_outrights() if _BP == "SRA" else _md.get_outrights(_BP)


def get_spreads(t):
    return _sra_get_spreads(t) if _BP == "SRA" else _md.get_spreads(_BP, t)


def get_flies(t):
    return _sra_get_flies(t) if _BP == "SRA" else _md.get_flies(_BP, t)


def get_available_tenors(strategy):
    return (_sra_get_available_tenors(strategy) if _BP == "SRA"
              else _md.get_available_tenors(_BP, strategy))


def get_sra_snapshot_latest_date():
    if _BP == "SRA":
        from lib.sra_data import get_sra_snapshot_latest_date as _f
        return _f()
    return _md.get_snapshot_latest_date(_BP)
from lib.theme import (
    ACCENT, AMBER, BLUE, GREEN, PURPLE, RED, TEXT_BODY, TEXT_DIM, TEXT_HEADING, TEXT_MUTED,
)


_SCOPE_PREFIX = "tech"


# =============================================================================
# CSS injection — state pills, cards, mini-bar chart for composites
# =============================================================================
_TECHNICALS_CSS = """
<style>
/* ---------- state-pills (used across views) ---------- */
.tech-state {
    display: inline-block; padding: 1px 7px; border-radius: 4px;
    font-family: 'JetBrains Mono', monospace; font-weight: 600; font-size: 0.65rem;
    line-height: 1.4; letter-spacing: 0.02em; text-align: center;
    border: 1px solid transparent;
}
.tech-state.fired-long  { background:#1b4d2c; color:#86efac; border-color:#22c55e; }
.tech-state.fired-short { background:#5d1818; color:#fca5a5; border-color:#ef4444; }
.tech-state.near-long   { background:#1b3522; color:#86efac; border-color:#22c55e80; }
.tech-state.near-short  { background:#3d1818; color:#fca5a5; border-color:#ef444480; }
.tech-state.appr        { background:#1a2a44; color:#93c5fd; border-color:#60a5fa80; }
.tech-state.far         { background:transparent; color:#5e6975; border-color:transparent; }
.tech-state.na          { background:transparent; color:#3d454f; border-color:transparent; }

/* ---------- matrix cells (very compact heatmap-style) ---------- */
.tech-cell {
    display: inline-block; min-width: 22px; padding: 2px 4px;
    text-align: center; border-radius: 3px;
    font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 0.65rem;
    line-height: 1.3; cursor: help; border: 1px solid transparent;
}
.tech-cell.fired-long  { background: #22c55e; color: #0a0e14; border-color:#22c55e; }
.tech-cell.fired-short { background: #ef4444; color: #0a0e14; border-color:#ef4444; }
.tech-cell.near-long   { background: rgba(34,197,94,0.20); color: #4ade80; border-color: rgba(34,197,94,0.45); }
.tech-cell.near-short  { background: rgba(239,68,68,0.20); color: #f87171; border-color: rgba(239,68,68,0.45); }
.tech-cell.appr        { background: rgba(96,165,250,0.16); color: #60a5fa; border-color: rgba(96,165,250,0.40); }
.tech-cell.far         { color: #3d454f; }
.tech-cell.na          { color: #2a2d3a; }

/* ---------- fire cards (View 1) ---------- */
.fire-panel-header {
    font-size: 0.85rem; font-weight: 600; color: var(--text-heading);
    padding: 8px 10px; background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-top-left-radius: 8px; border-top-right-radius: 8px;
}
.fire-panel-header .count {
    color: var(--text-dim); font-weight: 400; font-size: 0.78rem; margin-left: 0.4rem;
}
.fire-setup-card {
    background: var(--bg-elevated); border: 1px solid var(--border-subtle);
    border-top: 1px solid var(--border-default);
    padding: 8px 10px; margin-top: 0;
}
.fire-setup-card .setup-id {
    color: var(--accent); font-weight: 600; font-size: 0.82rem;
}
.fire-setup-card .setup-name {
    color: var(--text-muted); font-size: 0.7rem; margin-left: 0.3rem;
}
.fire-setup-card .setup-tr {
    font-size: 0.62rem; font-style: italic; margin-top: 2px;
}
.fire-setup-card .setup-tr.green { color: var(--green); }
.fire-setup-card .setup-tr.red   { color: var(--red); }
.fire-setup-card .setup-tr.amber { color: var(--amber); }

.fire-row {
    padding: 6px 10px; background: var(--bg-base);
    border-bottom: 1px solid var(--border-subtle);
    font-family: 'JetBrains Mono', monospace; font-size: 0.7rem;
    line-height: 1.3; cursor: help;
}
.fire-row:hover { background: var(--bg-elevated); }
.fire-row .fire-head {
    display: flex; justify-content: space-between; align-items: baseline;
    margin-bottom: 2px;
}
.fire-row .fire-head .left {
    font-weight: 600;
}
.fire-row .fire-head .left.long  { color: var(--green); }
.fire-row .fire-head .left.short { color: var(--red); }
.fire-row .fire-head .sym { color: var(--accent); margin-left: 0.4rem; }
.fire-row .fire-inputs {
    color: var(--text-dim); font-size: 0.62rem;
}
.fire-row .fire-trade {
    color: var(--text-body); font-size: 0.65rem; margin-top: 2px;
    display: flex; gap: 0.4rem; flex-wrap: wrap;
}
.fire-row .fire-trade .trade-cell {
    background: var(--bg-elevated); padding: 1px 5px; border-radius: 3px;
    border: 1px solid var(--border-subtle);
}
.fire-row .fire-trade .trade-cell.lots {
    background: rgba(232,183,93,0.12); border-color: rgba(232,183,93,0.35);
    color: var(--accent);
}

/* ---------- composite mini-bar (View 3) ---------- */
.comp-bar-track {
    position: relative; width: 100%; height: 14px;
    background: var(--bg-elevated); border-radius: 2px;
    overflow: hidden; cursor: help;
}
.comp-bar-track::before {
    /* center line at 0 */
    content: ''; position: absolute; top: 0; bottom: 0; left: 50%;
    width: 1px; background: rgba(255,255,255,0.18);
}
.comp-bar-fill {
    position: absolute; top: 0; bottom: 0;
    border-radius: 2px;
}
.comp-bar-fill.pos { background: linear-gradient(to right, rgba(74,222,128,0.4), #22c55e); }
.comp-bar-fill.neg { background: linear-gradient(to left, rgba(248,113,113,0.4), #ef4444); }
.comp-bar-label {
    position: absolute; top: 1px; left: 6px; font-size: 0.62rem;
    font-family: 'JetBrains Mono', monospace; color: var(--text-heading);
    font-weight: 600; text-shadow: 0 0 3px rgba(0,0,0,0.6);
}

/* ---------- universal table styles ---------- */
.tech-table { width: 100%; border-collapse: collapse; font-family: 'JetBrains Mono', monospace; }
.tech-table th {
    text-align: left; padding: 5px 8px; color: var(--text-dim);
    font-weight: 500; font-size: 0.65rem; text-transform: uppercase;
    letter-spacing: 0.05em; border-bottom: 1px solid var(--border-default);
}
.tech-table td {
    padding: 4px 8px; border-bottom: 1px solid var(--border-subtle);
    font-size: 0.7rem; vertical-align: middle;
}
.tech-table tr:hover td { background: var(--bg-elevated); }

/* ---------- view section headers ---------- */
.tech-view-header {
    display: flex; justify-content: space-between; align-items: baseline;
    margin: 1.0rem 0 0.4rem 0; padding-bottom: 4px;
    border-bottom: 1px solid var(--border-subtle);
}
.tech-view-header .title {
    font-size: 0.92rem; font-weight: 600; color: var(--text-heading);
}
.tech-view-header .meta {
    font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: var(--text-dim);
}

/* ---------- multi-leg trade table (View 1 + drill-down) ---------- */
.trade-legs {
    margin-top: 4px; padding: 4px 6px;
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    border-left: 2px solid var(--accent);
    border-radius: 3px;
}
.trade-legs .legs-title {
    color: var(--text-dim); font-size: 0.58rem; letter-spacing: 0.05em;
    text-transform: uppercase; margin-bottom: 2px;
}
.trade-legs .leg-row {
    display: grid;
    grid-template-columns: 38px 1fr 38px 56px 1fr;
    gap: 6px; align-items: center;
    font-family: 'JetBrains Mono', monospace; font-size: 0.65rem;
    padding: 1px 0;
}
.trade-legs .leg-side {
    font-weight: 700; text-align: center; padding: 1px 0;
    border-radius: 2px; font-size: 0.6rem; letter-spacing: 0.04em;
}
.trade-legs .leg-side.buy  { background: rgba(34,197,94,0.18);  color: var(--green); border:1px solid rgba(34,197,94,0.45); }
.trade-legs .leg-side.sell { background: rgba(239,68,68,0.18);  color: var(--red);   border:1px solid rgba(239,68,68,0.45); }
.trade-legs .leg-sym {
    color: var(--accent); font-weight: 600;
}
.trade-legs .leg-role {
    color: var(--text-dim); text-align: center; font-size: 0.55rem;
    text-transform: uppercase; letter-spacing: 0.04em;
}
.trade-legs .leg-lots {
    color: var(--text-heading); text-align: right; font-weight: 600;
}
.trade-legs .leg-dv01 {
    color: var(--text-muted); font-size: 0.6rem; text-align: right;
}
</style>
"""


# =============================================================================
# Pill / badge / tooltip helpers
# =============================================================================
def _tooltip_attr(text: str) -> str:
    return (str(text)
            .replace('"', '&quot;')
            .replace("'", "&#39;")
            .replace("\n", "&#10;"))


def _state_pill(state: str, direction: Optional[str] = None) -> str:
    if state == "FIRED":
        if direction == "LONG":
            color = "var(--green)"; bg = "rgba(74,222,128,0.18)"; text = "▲ FIRED LONG"
        elif direction == "SHORT":
            color = "var(--red)"; bg = "rgba(248,113,113,0.18)"; text = "▼ FIRED SHORT"
        else:
            color = "var(--accent)"; bg = "rgba(232,183,93,0.18)"; text = "● FIRED"
    elif state == "NEAR":
        color = "var(--amber)"; bg = "rgba(251,191,36,0.14)"; text = f"~ NEAR {direction or ''}"
    elif state == "APPROACHING":
        color = "var(--blue)"; bg = "rgba(96,165,250,0.12)"; text = f"⌒ APPR {direction or ''}"
    elif state == "FAR":
        color = "var(--text-dim)"; bg = "rgba(94,105,117,0.10)"; text = "FAR"
    else:
        color = "var(--text-muted)"; bg = "rgba(94,105,117,0.05)"; text = "N/A"
    return (f"<span style='display:inline-block; padding:1px 6px; "
            f"background:{bg}; border:1px solid {color}; color:{color}; "
            f"font-size:0.65rem; font-family:JetBrains Mono, monospace; "
            f"font-weight:600; border-radius:4px; line-height:1;'>{text}</span>")


def _family_pill(family: str) -> str:
    color_map = {FAMILY_TREND: "var(--blue)", FAMILY_MR: "var(--green)",
                 FAMILY_STIR: "var(--purple, #a78bfa)", FAMILY_COMPOSITE: "var(--accent)"}
    color = color_map.get(family, "var(--text-muted)")
    label = {FAMILY_TREND: "TREND", FAMILY_MR: "MR",
             FAMILY_STIR: "STIR", FAMILY_COMPOSITE: "COMP"}.get(family, family)
    return (f"<span style='display:inline-block; padding:0 5px; "
            f"border:1px solid {color}; color:{color}; font-size:0.6rem; "
            f"font-family:JetBrains Mono, monospace; font-weight:600; "
            f"border-radius:3px; line-height:1.3;'>{label}</span>")


def _legs_block_html(legs: list) -> str:
    """Render the multi-leg trade table for a single fire.

    Empty legs → empty string (single-contract fires that haven't been
    expanded). Single-leg legs (outright role='single') → still render so
    the trader sees explicit BUY/SELL with lot count and DV01.

    If a leg carries ``entry_price`` (populated for C9 pair legs and any
    other multi-leg setup that knows per-leg execution levels), render it
    inline next to the symbol so the trader has a price to lift/hit.
    """
    if not legs:
        return ""
    rows = []
    for leg in legs:
        side = (leg.get("side") or "").upper()
        side_cls = "buy" if side == "BUY" else ("sell" if side == "SELL" else "")
        sym = leg.get("symbol") or "—"
        role = leg.get("role") or ""
        # Suppress role label for single-leg outrights (visual noise)
        role_html = (f"<div class='leg-role'>{role}</div>"
                     if role and role != "single" else "<div class='leg-role'>&nbsp;</div>")
        lots = leg.get("lots") or 0
        ratio = leg.get("ratio") or 1
        ratio_str = f"×{ratio}" if ratio and ratio != 1 else ""
        dv01 = leg.get("dv01_per_bp") or 0
        # Per-leg entry price — leg symbols are individual outrights, so use
        # outright bp_mult=100.0 for tick-rounded display.
        ep = leg.get("entry_price")
        ep_html = ""
        if ep is not None:
            try:
                ep_str = fmt_price_for_scope(float(ep), 100.0)
                ep_html = (f" <span style='color:var(--text-dim); "
                           f"font-size:0.6rem; font-weight:400;'>"
                           f"@ {ep_str}</span>")
            except Exception:
                ep_html = ""
        rows.append(
            f"<div class='leg-row'>"
            f"  <div class='leg-side {side_cls}'>{side}</div>"
            f"  <div class='leg-sym'>{sym}{ep_html}</div>"
            f"  <div class='leg-role'>{ratio_str}</div>"
            f"  <div class='leg-lots'>{int(lots)}</div>"
            f"  <div class='leg-dv01'>${dv01:,.0f}/bp · {role}</div>"
            f"</div>"
        )
    return (
        "<div class='trade-legs'>"
        "<div class='legs-title'>Trade legs</div>"
        + "".join(rows)
        + "</div>"
    )


def _composite_color(value: float) -> str:
    if value is None or pd.isna(value):
        return "var(--text-dim)"
    if value >= 0.7:
        return "var(--green)"
    if value >= 0.5:
        return "rgba(74,222,128,0.85)"
    if value >= 0.3:
        return "rgba(74,222,128,0.55)"
    if value <= -0.7:
        return "var(--red)"
    if value <= -0.5:
        return "rgba(248,113,113,0.85)"
    if value <= -0.3:
        return "rgba(248,113,113,0.55)"
    return "var(--text-muted)"


def _composite_label(v: float) -> str:
    if v is None or pd.isna(v):
        return "—"
    if v >= 0.7:    return "STRONG SIGNAL ↑"
    if v >= 0.5:    return "moderate ↑"
    if v >= 0.3:    return "soft ↑"
    if v <= -0.7:   return "STRONG SIGNAL ↓"
    if v <= -0.5:   return "moderate ↓"
    if v <= -0.3:   return "soft ↓"
    return "neutral · no edge"


def _section_header(text: str, tooltip: str = "") -> None:
    if tooltip:
        attr = _tooltip_attr(tooltip)
        st.markdown(
            f"<div title=\"{attr}\" "
            f"style='font-size:0.7rem; color:var(--text-dim); text-transform:uppercase; "
            f"letter-spacing:0.06em; margin: 0.6rem 0 0.3rem 0; font-weight:600; cursor:help;'>"
            f"{text} <span style='color:var(--accent-dim);'>ⓘ</span></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='font-size:0.7rem; color:var(--text-dim); text-transform:uppercase; "
            f"letter-spacing:0.06em; margin: 0.6rem 0 0.3rem 0; font-weight:600;'>{text}</div>",
            unsafe_allow_html=True,
        )


def _kv_block(rows: list) -> None:
    """Tuples: (label, value, color_class, sub_text, tooltip_text)."""
    parts = []
    for tup in rows:
        label = tup[0]
        value = tup[1]
        cls = tup[2] if len(tup) > 2 and tup[2] else ""
        sub = tup[3] if len(tup) > 3 and tup[3] else ""
        tooltip = tup[4] if len(tup) > 4 and tup[4] else ""
        sub_html = (f"<span style='color:var(--text-dim); font-size:0.7rem; "
                    f"font-family:JetBrains Mono, monospace; margin-left:0.4rem;'>{sub}</span>"
                    if sub else "")
        color_var = (
            "var(--accent)" if cls == "accent"
            else "var(--green)" if cls == "green"
            else "var(--red)" if cls == "red"
            else "var(--amber)" if cls == "amber"
            else "var(--text-body)"
        )
        title_attr = f'title="{_tooltip_attr(tooltip)}"' if tooltip else ""
        cursor = "cursor:help;" if tooltip else ""
        parts.append(
            f"<div {title_attr} "
            f"style='display:flex; justify-content:space-between; align-items:center; "
            f"padding:4px 0; border-bottom:1px solid var(--border-subtle); {cursor}'>"
            f"<span style='color:var(--text-muted); font-size:0.78rem;'>{label}</span>"
            f"<span style='color:{color_var}; font-family:JetBrains Mono, monospace; "
            f"font-size:0.78rem; font-weight:500; text-align:right;'>{value}{sub_html}</span>"
            f"</div>"
        )
    st.markdown("".join(parts), unsafe_allow_html=True)


def _reading(text: str) -> None:
    """Italic 'Reading:' paragraph beneath a side-panel block."""
    st.markdown(
        f"<div style='padding:4px 6px; margin-top:4px; "
        f"background:rgba(232,183,93,0.05); border-left:2px solid var(--accent-dim); "
        f"font-size:0.7rem; color:var(--text-body); font-style:italic; line-height:1.4;'>"
        f"<b style='color:var(--accent); font-style:normal;'>Reading:</b> {text}"
        f"</div>",
        unsafe_allow_html=True,
    )


def _build_setup_tooltip(setup_id: str, scope: str, result: dict) -> str:
    """Per-cell tooltip — Phase G 6-section format with research-backed
    NEAR/APPROACHING citation + result-injected current state, inputs,
    and trade levels.

    Delegates to ``lib.setups.tooltips.setup_six_section_tooltip``.
    """
    state = result.get("state", "—")
    direction = result.get("direction") or "—"
    label = (
        ("FIRED " + direction) if state == "FIRED"
        else f"{state} {direction}" if state in ("NEAR", "APPROACHING")
        else state
    )
    return setup_six_section_tooltip(setup_id, result=result, state_label=label)


# =============================================================================
# Interpretation guide expander (top of subtab)
# =============================================================================
def _render_interpretation_guide_technicals(scope: str) -> None:
    guide = get_full_guide_for_scope(scope)
    with st.expander("📖 Interpretation guide — click to see what every setup means + thresholds",
                      expanded=False):
        st.markdown(
            "<div style='color:var(--text-muted); font-size:0.78rem; margin-bottom:0.5rem;'>"
            "Every setup detector + composite is paired with a plain-English label so "
            "you don't have to memorise the conditions. This guide lists each setup, "
            "its logic, stop/T1/T2 rules, and the bucket boundaries used."
            "</div>",
            unsafe_allow_html=True,
        )
        for sid, info in guide.items():
            st.markdown(
                f"<div style='margin-top:0.6rem; padding:6px 8px; "
                f"background:var(--bg-surface); border:1px solid var(--border-subtle); "
                f"border-radius:6px;'>"
                f"<div style='color:var(--accent); font-weight:600; font-size:0.85rem; "
                f"margin-bottom:0.2rem;'>{display_name(sid)}</div>"
                f"<div style='color:var(--text-body); font-size:0.75rem; line-height:1.4;'>"
                f"{info.get('what', '')}</div>"
                f"<div style='color:var(--text-dim); font-size:0.7rem; "
                f"font-family:JetBrains Mono, monospace; margin-top:0.2rem; white-space:pre-wrap;'>"
                f"{info.get('formula', '')}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            buckets = info.get("buckets", [])
            if buckets:
                rows_html = []
                for cond, label, meaning in buckets:
                    rows_html.append(
                        f"<div style='display:grid; grid-template-columns: 1fr 1.2fr 2.2fr; "
                        f"gap:8px; padding:3px 8px; font-size:0.72rem; "
                        f"border-bottom:1px solid var(--border-subtle);'>"
                        f"<span style='color:var(--text-muted); "
                        f"font-family:JetBrains Mono, monospace;'>{cond}</span>"
                        f"<span style='color:var(--accent-bright); font-weight:500;'>{label}</span>"
                        f"<span style='color:var(--text-body);'>{meaning}</span>"
                        f"</div>"
                    )
                st.markdown("".join(rows_html), unsafe_allow_html=True)


# =============================================================================
# View 1 — Today's fires (3 panels: TREND / MR / STIR)
# =============================================================================
def _render_view_today_fires(scan: dict) -> None:
    fires = scan.get("fires_today", [])
    if not fires:
        st.info("No fires today across the selected scope.")
        return

    # Group by family
    by_family = {FAMILY_TREND: {}, FAMILY_MR: {}, FAMILY_STIR: {}}
    for f in fires:
        fam = f.get("family")
        if fam not in by_family:
            continue
        by_family[fam].setdefault(f["setup_id"], []).append(f)

    track_records = scan.get("track_records", {})
    scope = scan.get("scope", "outright")
    bp_mult = 100.0 if scope == "outright" else 1.0

    cols = st.columns(3)
    for i, (fam, label, color_var) in enumerate(
        [(FAMILY_TREND, "TREND fires", "var(--blue)"),
         (FAMILY_MR, "MEAN-REVERSION fires", "var(--green)"),
         (FAMILY_STIR, "STIR-specific fires", "#a78bfa")]
    ):
        with cols[i]:
            n_total = sum(len(v) for v in by_family[fam].values())
            n_setups = len(by_family[fam])
            st.markdown(
                f"<div class='fire-panel-header' style='border-left:3px solid {color_var};'>"
                f"{label}<span class='count'>· {n_total} fire(s) across {n_setups} setup(s)</span></div>",
                unsafe_allow_html=True,
            )
            if not by_family[fam]:
                st.markdown(
                    "<div style='padding:10px; background:var(--bg-elevated); "
                    "border:1px solid var(--border-subtle); border-top:none; "
                    "color:var(--text-dim); font-size:0.72rem; text-align:center;'>None today.</div>",
                    unsafe_allow_html=True,
                )
                continue
            # Sort setup ids by 60d track record (best mean +5-bar return first).
            # Setups without a track record (or 0 fires_60d) go last, alphabetically.
            def _track_sort_key(sid: str):
                # Strip C9a_NM / C9b_NM variant suffix so it inherits the base
                # track-record bucket (C9 itself isn't in PER_CONTRACT_DETECTORS;
                # all C9 variants will tie at "untracked").
                base_id = sid.split("_")[0] if sid.startswith(("C9a_", "C9b_")) else sid
                tr_for = track_records.get(sid) or track_records.get(base_id) or {}
                fires_n = int(tr_for.get("fires_60d") or 0)
                mr_bp = tr_for.get("mean_5bar_return_bp")
                if fires_n == 0 or mr_bp is None:
                    return (1, 0.0, sid)             # untracked → after tracked
                return (0, -float(mr_bp), sid)       # tracked, descending mean return

            sorted_sids = sorted(by_family[fam].keys(), key=_track_sort_key)
            for rank, sid in enumerate(sorted_sids, start=1):
                entries = by_family[fam][sid]
                reg = get_registry_entry(sid)
                # Resolve track record (use stripped base id for C9 variants)
                _base = sid.split("_")[0] if sid.startswith(("C9a_", "C9b_")) else sid
                tr = track_records.get(sid) or track_records.get(_base) or {}
                tr_html = ""
                rank_color = "var(--accent)"
                if tr and tr.get("fires_60d", 0) > 0:
                    wr = tr.get("win_rate_5bar")
                    mr = tr.get("mean_5bar_return_bp")
                    quality = tr.get("sample_quality", "")
                    if wr is not None and mr is not None:
                        tr_cls = ("green" if mr > 0 else "red")
                        rank_color = "var(--green)" if mr > 0 else "var(--red)"
                        tr_html = (
                            f"<div class='setup-tr {tr_cls}'>"
                            f"60d track: {tr['fires_60d']} fires · {wr:.0f}% win · "
                            f"{mr:+.1f}bp avg ({quality})</div>"
                        )
                else:
                    rank_color = "var(--text-dim)"  # untracked → muted rank pill
                rank_html = (
                    f"<span style='display:inline-block; min-width:22px; padding:0 4px; "
                    f"margin-right:5px; border:1px solid {rank_color}; color:{rank_color}; "
                    f"font-size:0.6rem; font-family:JetBrains Mono, monospace; "
                    f"font-weight:700; border-radius:3px; line-height:1.3; "
                    f"text-align:center;' title='Rank within family by 60d mean +5-bar return "
                    f"(best first; untracked setups follow alphabetically)'>"
                    f"#{rank}</span>"
                )
                st.markdown(
                    f"<div class='fire-setup-card'>"
                    f"<div style='display:flex; justify-content:space-between; align-items:baseline;'>"
                    f"<div>{rank_html}<span class='setup-name' style='font-weight:600; color:var(--accent);'>"
                    f"{display_name(sid)}</span></div>"
                    f"<span style='color:var(--text-dim); font-size:0.7rem;'>{len(entries)} fire(s)</span>"
                    f"</div>{tr_html}</div>",
                    unsafe_allow_html=True,
                )
                # Per-fire compact rows
                rows_html = []
                for f in entries[:5]:
                    sym = f["symbol"]; direction = f.get("direction") or "—"
                    e = f.get("entry"); s = f.get("stop")
                    t1 = f.get("t1"); t2 = f.get("t2"); lots = f.get("lots_at_10k_risk")
                    inputs = f.get("key_inputs", {}) or {}
                    inputs_compact = " · ".join(
                        f"{k}={v:.2f}" if isinstance(v, (int, float)) and not isinstance(v, bool)
                        else f"{k}={v}"
                        for k, v in list(inputs.items())[:3]
                    )
                    arrow = "▲" if direction == "LONG" else ("▼" if direction == "SHORT" else "●")
                    side_cls = "long" if direction == "LONG" else ("short" if direction == "SHORT" else "")
                    # C9 fires use bp_mult=1 for entry/stop display (slope is in bp)
                    fire_bp_mult = 1.0 if sid.startswith(("C9a_", "C9b_")) else bp_mult
                    trade_cells = ""
                    if e is not None:
                        trade_cells += f"<span class='trade-cell'>entry <b>{fmt_price_for_scope(e, fire_bp_mult)}</b></span>"
                    if s is not None:
                        trade_cells += f"<span class='trade-cell'>stop <b>{fmt_price_for_scope(s, fire_bp_mult)}</b></span>"
                    if t1 is not None:
                        trade_cells += f"<span class='trade-cell'>T1 <b>{fmt_price_for_scope(t1, fire_bp_mult)}</b></span>"
                    if t2 is not None:
                        trade_cells += f"<span class='trade-cell'>T2 <b>{fmt_price_for_scope(t2, fire_bp_mult)}</b></span>"
                    if lots:
                        trade_cells += f"<span class='trade-cell lots'><b>{lots}</b> lots @ $10K</span>"
                    legs_html = _legs_block_html(f.get("legs", []) or [])
                    tt = _build_setup_tooltip(sid, scope, f)
                    rows_html.append(
                        f"<div class='fire-row' title='{_tooltip_attr(tt)}'>"
                        f"  <div class='fire-head'>"
                        f"    <span class='left {side_cls}'>{arrow} {direction}<span class='sym'>{sym}</span></span>"
                        f"  </div>"
                        f"  <div class='fire-inputs'>{inputs_compact}</div>"
                        + (f"  <div class='fire-trade'>{trade_cells}</div>" if trade_cells else "")
                        + legs_html
                        + "</div>"
                    )
                if entries[5:]:
                    rows_html.append(
                        f"<div style='padding:4px 8px; background:var(--bg-base); "
                        f"font-size:0.65rem; color:var(--text-dim); text-align:center; "
                        f"border-bottom:1px solid var(--border-subtle);'>"
                        f"+ {len(entries) - 5} more fire(s) — see View 2 (state matrix)</div>"
                    )
                st.markdown("".join(rows_html), unsafe_allow_html=True)


# =============================================================================
# View 2 — Setup state matrix
# =============================================================================
def _state_to_css(state: str, direction: Optional[str]) -> tuple:
    """Map (state, direction) → (CSS class, cell text)."""
    if state == "FIRED" and direction == "LONG":
        return ("fired-long", "L")
    if state == "FIRED" and direction == "SHORT":
        return ("fired-short", "S")
    if state == "NEAR" and direction == "LONG":
        return ("near-long", "L")
    if state == "NEAR" and direction == "SHORT":
        return ("near-short", "S")
    if state == "APPROACHING":
        if direction == "LONG":
            return ("appr", "↑")
        if direction == "SHORT":
            return ("appr", "↓")
        return ("appr", "·")
    if state == "FAR":
        return ("far", "·")
    return ("na", "—")


def _render_view_state_matrix(scan: dict, top_n: int = 25) -> None:
    """Compact CSS-pill heatmap. Rows = top contracts by signal density,
    columns = setups, cells = colored state pills with direction letter."""
    by_contract = scan.get("by_contract", {})
    if not by_contract:
        st.caption("No contracts in scope.")
        return
    scope = scan.get("scope", "outright")
    setup_ids = setups_for_scope(scope)

    def _score(sym):
        n = 0
        for sid, r in by_contract.get(sym, {}).items():
            if sid.startswith("_") or not isinstance(r, dict):
                continue
            st_ = r.get("state")
            if st_ == "FIRED": n += 3
            elif st_ == "NEAR": n += 2
            elif st_ == "APPROACHING": n += 1
        return n
    ranked = sorted(by_contract.keys(), key=_score, reverse=True)[:top_n]
    if not ranked:
        st.caption("No contracts.")
        return

    parts = [
        '<div style="overflow-x:auto;">',
        '<table class="tech-table">',
        '<thead><tr>',
        '<th style="position:sticky; left:0; background:var(--bg-base); z-index:2; min-width:75px;">SYM</th>',
    ]
    for sid in setup_ids:
        # Title-Case short label (e.g. "Trend Continuation Breakout" -> "TCB").
        # Full name + formula appears via the column-header tooltip (Phase G).
        parts.append(
            f'<th title="{_tooltip_attr(display_name(sid))}" '
            f'style="text-align:center; min-width:34px; cursor:help;">'
            f'{display_name_short(sid, max_len=6)}</th>'
        )
    parts.append('</tr></thead><tbody>')

    for sym in ranked:
        parts.append('<tr>')
        parts.append(
            f'<td style="position:sticky; left:0; background:var(--bg-base); z-index:1; '
            f'color:var(--accent); font-weight:600;">{sym}</td>'
        )
        for sid in setup_ids:
            r = by_contract.get(sym, {}).get(sid)
            if not isinstance(r, dict):
                parts.append('<td style="text-align:center;">'
                             '<span class="tech-cell na">—</span></td>')
                continue
            css, txt = _state_to_css(r.get("state"), r.get("direction"))
            tt = _build_setup_tooltip(sid, scope, r)
            parts.append(
                f'<td style="text-align:center; padding:2px 4px;">'
                f'<span class="tech-cell {css}" title="{_tooltip_attr(tt)}">{txt}</span></td>'
            )
        parts.append('</tr>')
    parts.append('</tbody></table></div>')
    st.markdown("".join(parts), unsafe_allow_html=True)
    # Legend
    st.markdown(
        '<div style="display:flex; gap:0.5rem; margin-top:6px; flex-wrap:wrap; '
        'font-size:0.62rem; align-items:center;">'
        '<span class="tech-cell fired-long">L</span>'
        '<span style="color:var(--text-muted);">FIRED LONG</span>'
        '<span class="tech-cell fired-short">S</span>'
        '<span style="color:var(--text-muted);">FIRED SHORT</span>'
        '<span class="tech-cell near-long">L</span>'
        '<span style="color:var(--text-muted);">NEAR LONG</span>'
        '<span class="tech-cell near-short">S</span>'
        '<span style="color:var(--text-muted);">NEAR SHORT</span>'
        '<span class="tech-cell appr">·</span>'
        '<span style="color:var(--text-muted);">APPROACHING</span>'
        '<span class="tech-cell far">·</span>'
        '<span style="color:var(--text-muted);">FAR</span>'
        '<span class="tech-cell na">—</span>'
        '<span style="color:var(--text-muted);">N/A (no data)</span>'
        '</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# View 3 — Composite scoring (gauges + factor breakdown)
# =============================================================================
def _comp_bar_html(value: float, tooltip: str = "") -> str:
    """Render a horizontal bar visualization of a composite score in [-1, +1]."""
    v = max(-1.0, min(1.0, value if (value is not None and np.isfinite(value)) else 0.0))
    width_pct = abs(v) * 50.0    # 0..50% of track width on either side of centre
    if v >= 0:
        # Bar fills from centre to right
        bar = (f'<div class="comp-bar-fill pos" '
                f'style="left:50%; width:{width_pct:.1f}%;"></div>')
    else:
        bar = (f'<div class="comp-bar-fill neg" '
                f'style="right:50%; width:{width_pct:.1f}%;"></div>')
    label = f'<span class="comp-bar-label">{v:+.2f}</span>'
    return (f'<div class="comp-bar-track" title="{_tooltip_attr(tooltip)}">{bar}{label}</div>')


def _render_view_composites(scan: dict) -> None:
    by_contract = scan.get("by_contract", {})
    if not by_contract:
        st.caption("No contracts.")
        return
    scope = scan.get("scope", "outright")
    rows = []
    for sym, results in by_contract.items():
        comp = results.get("_composites", {}) or {}
        rows.append({
            "symbol": sym,
            "trend": comp.get("trend_score", 0.0) or 0.0,
            "mr":    comp.get("mr_score", 0.0) or 0.0,
            "final": comp.get("final_score", 0.0) or 0.0,
            "regime": comp.get("regime", "NEUTRAL"),
            "interpretation": comp.get("interpretation", "—"),
            "trend_factors": comp.get("trend_factors", {}),
            "mr_factors":    comp.get("mr_factors", {}),
            "weights":       comp.get("weights", {}),
        })
    rows.sort(key=lambda r: -abs(r["final"]))

    if scope == "outright":
        guide_for_tt = get_composite_guide_outright()
    elif scope == "spread":
        guide_for_tt = get_composite_guide_spread()
    else:
        guide_for_tt = get_composite_guide_fly()
    scope_keys = list(guide_for_tt.keys())
    trend_key = next((k for k in scope_keys if k.startswith("TREND_")), None)
    mr_key    = next((k for k in scope_keys if k.startswith("MR_")), None)
    final_key = next((k for k in scope_keys if k.startswith("FINAL_")), None)

    # Phase G: rich column-header tooltips for composite ranges + regime + take
    tt_trend_hdr = _tooltip_attr(composite_header_tooltip("TREND_COMPOSITE"))
    tt_mr_hdr    = _tooltip_attr(composite_header_tooltip("MR_COMPOSITE"))
    tt_final_hdr = _tooltip_attr(composite_header_tooltip("FINAL_COMPOSITE"))
    tt_regime_hdr = _tooltip_attr(regime_header_tooltip())
    tt_take_hdr   = _tooltip_attr(take_header_tooltip())
    parts = [
        '<div style="overflow-x:auto;">',
        '<table class="tech-table">',
        '<thead><tr>',
        '<th>SYMBOL</th>',
        f'<th title="{tt_trend_hdr}" style="min-width:140px; cursor:help;">TREND  [-1 ··· +1] ⓘ</th>',
        f'<th title="{tt_mr_hdr}" style="min-width:140px; cursor:help;">MR  [-1 ··· +1] ⓘ</th>',
        f'<th title="{tt_final_hdr}" style="min-width:140px; cursor:help;">FINAL  [-1 ··· +1] ⓘ</th>',
        f'<th title="{tt_regime_hdr}" style="text-align:center; cursor:help;">REGIME ⓘ</th>',
        f'<th title="{tt_take_hdr}" style="text-align:center; cursor:help;">TAKE ⓘ</th>',
        '</tr></thead><tbody>',
    ]
    for r in rows:
        sym = r["symbol"]
        t = r["trend"]; m = r["mr"]; fv = r["final"]
        # Phase G: composite cell tooltips with range table + factor breakdown
        tt_trend = composite_cell_tooltip("TREND_COMPOSITE", t, r.get("trend_factors"))
        tt_mr    = composite_cell_tooltip("MR_COMPOSITE",    m, r.get("mr_factors"))
        tt_final = composite_cell_tooltip("FINAL_COMPOSITE", fv,
                                            {**(r.get("weights") or {}),
                                             "TREND_score": t, "MR_score": m})
        final_color = _composite_color(fv)
        final_label = _composite_label(fv)
        parts.append(
            f'<tr>'
            f'<td style="color:var(--accent); font-weight:600;">{sym}</td>'
            f'<td>{_comp_bar_html(t, tt_trend)}</td>'
            f'<td>{_comp_bar_html(m, tt_mr)}</td>'
            f'<td>{_comp_bar_html(fv, tt_final)}</td>'
            f'<td style="text-align:center; font-size:0.65rem; color:var(--text-muted);">'
            f'{r["regime"]}</td>'
            f'<td style="text-align:center; color:{final_color}; font-style:italic; '
            f'font-size:0.7rem;">{final_label}</td>'
            f'</tr>'
        )
    parts.append('</tbody></table></div>')
    st.markdown("".join(parts), unsafe_allow_html=True)
    # Threshold reference line
    st.markdown(
        '<div style="font-size:0.6rem; color:var(--text-muted); margin-top:4px; text-align:center;">'
        'Bar centred at 0. Reference thresholds: ±0.3 SOFT · ±0.5 MODERATE · ±0.7 STRONG. '
        'Hover any bar for full factor breakdown + weights.'
        '</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# View 4 — Proximity-to-fire
# =============================================================================
def _fmt_or_dash(v, fmt: str = ":.2f") -> str:
    if v is None or pd.isna(v) or not np.isfinite(v):
        return "—"
    return ("{0" + fmt + "}").format(v)


def _render_view_near(scan: dict, top_n: int = 25) -> None:
    near = scan.get("near_today", [])[:top_n]
    if not near:
        st.markdown(
            "<div style='padding:8px 10px; background:var(--bg-elevated); "
            "border:1px solid var(--border-subtle); border-radius:4px; "
            "color:var(--text-muted); font-size:0.75rem; text-align:center;'>"
            "No NEAR signals — the universe is quiet at this scope.</div>",
            unsafe_allow_html=True,
        )
        return
    scope = scan.get("scope", "outright")
    by_contract = scan.get("by_contract", {})

    parts = [
        '<div style="overflow-x:auto;">',
        '<table class="tech-table">',
        '<thead><tr>',
        '<th>SETUP</th><th>SYMBOL</th>',
        '<th style="text-align:center;">DIR</th>',
        '<th style="text-align:right;">DIST</th>',
        '<th style="text-align:right;">ETA</th>',
        '<th>READING</th>',
        '</tr></thead><tbody>',
    ]
    for n in near:
        sid = n["setup_id"]; sym = n["symbol"]
        direction = n.get("direction") or "—"
        dist = n.get("distance_to_fire")
        eta = n.get("eta_bars")
        reading = n.get("interpretation", "")
        full_r = by_contract.get(sym, {}).get(sid, {})
        tt = _build_setup_tooltip(sid, scope, full_r) if full_r else ""
        dir_cls = ("near-long" if direction == "LONG"
                    else "near-short" if direction == "SHORT"
                    else "appr")
        dir_label = direction[:1] if direction in ("LONG", "SHORT") else direction
        parts.append(
            f'<tr title="{_tooltip_attr(tt)}" style="cursor:help;">'
            f'<td style="color:var(--text-body);">{display_name(sid)}</td>'
            f'<td style="color:var(--accent);">{sym}</td>'
            f'<td style="text-align:center;"><span class="tech-cell {dir_cls}">{dir_label}</span></td>'
            f'<td style="text-align:right;">{_fmt_or_dash(dist, ":.2f")}</td>'
            f'<td style="text-align:right; color:var(--text-dim);">'
            f'{(_fmt_or_dash(eta, ":.0f") + "d") if eta is not None else "—"}</td>'
            f'<td style="color:var(--text-muted); font-size:0.65rem;">{reading}</td>'
            f'</tr>'
        )
    parts.append('</tbody></table></div>')
    st.markdown("".join(parts), unsafe_allow_html=True)
    if len(scan.get("near_today", [])) > top_n:
        st.markdown(
            f'<div style="font-size:0.65rem; color:var(--text-dim); '
            f'text-align:center; margin-top:4px;">'
            f'Showing top {top_n} of {len(scan["near_today"])} NEAR signals (closest first).'
            f'</div>',
            unsafe_allow_html=True,
        )


# =============================================================================
# View 5 — Drill-down (per-setup or per-contract)
# =============================================================================
def _render_drill_down(scan: dict, asof_date: date) -> None:
    contracts = scan.get("contracts", [])
    scope = scan.get("scope", "outright")
    if not contracts:
        return
    with st.expander("🔍 Drill into a setup or contract", expanded=False):
        cc1, cc2, cc3 = st.columns([1, 2, 1])
        with cc1:
            valid_setups = [s for s in setups_for_scope(scope) if s in scan.get("by_contract", {}).get(contracts[0], {})]
            valid_setups = sorted(set(valid_setups))
            setup_pick = st.selectbox(
                "Setup", options=["—"] + valid_setups,
                format_func=lambda s: "—" if s == "—" else display_name(s),
                key=f"{_SCOPE_PREFIX}_drill_setup")
        with cc2:
            contract_pick = st.selectbox("Contract", options=["—"] + list(contracts),
                                           key=f"{_SCOPE_PREFIX}_drill_contract")
        with cc3:
            history_lb = st.selectbox(
                "History", ["30d", "60d", "90d", "180d", "252d"],
                index=2, key=f"{_SCOPE_PREFIX}_drill_lb",
            )
        if contract_pick == "—":
            return

        # Per-contract chart with applicable indicators
        days = {"30d": 30, "60d": 60, "90d": 90, "180d": 180, "252d": 252}[history_lb]
        start = asof_date - timedelta(days=int(days * 1.5) + 7)
        history = get_contract_history(contract_pick, start, asof_date)
        if history.empty:
            st.info(f"No history for {contract_pick}")
            return
        history = history.tail(days)

        # Setup state for the picked contract / setup
        setup_state = scan.get("by_contract", {}).get(contract_pick, {}).get(setup_pick, {})
        if isinstance(setup_state, dict) and setup_state and setup_pick != "—":
            reg = get_registry_entry(setup_pick)
            state = setup_state.get("state", "—")
            direction = setup_state.get("direction", "—")
            interp = setup_state.get("interpretation", "")
            color = "var(--green)" if direction == "LONG" else "var(--red)" if direction == "SHORT" else "var(--accent)"
            tt = _build_setup_tooltip(setup_pick, scope, setup_state)
            st.markdown(
                f"<div title='{_tooltip_attr(tt)}' "
                f"style='padding:8px 10px; background:var(--bg-surface); cursor:help; "
                f"border:1px solid var(--border-subtle); border-radius:6px; margin-bottom:6px;'>"
                f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
                f"<span style='color:var(--accent); font-weight:600;'>"
                f"{display_name(setup_pick)}</span>"
                f"<span style='color:{color}; font-weight:600;'>{state} {direction}</span>"
                f"</div>"
                f"<div style='color:var(--text-body); font-size:0.78rem; margin-top:4px;'>{interp}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            # Trade levels
            e = setup_state.get("entry"); s = setup_state.get("stop")
            t1 = setup_state.get("t1"); t2 = setup_state.get("t2")
            lots = setup_state.get("lots_at_10k_risk")
            # C9 fires use bp_mult=1 (slope is in bp); other outright fires use 100
            if setup_pick.startswith(("C9a_", "C9b_")):
                bp_mult_drill = 1.0
            else:
                bp_mult_drill = 100.0 if scope == "outright" else 1.0
            if e is not None:
                cm1, cm2, cm3, cm4, cm5 = st.columns(5)
                cm1.metric("Entry", fmt_price_for_scope(e, bp_mult_drill))
                cm2.metric("Stop", fmt_price_for_scope(s, bp_mult_drill))
                cm3.metric("T1", fmt_price_for_scope(t1, bp_mult_drill))
                cm4.metric("T2", fmt_price_for_scope(t2, bp_mult_drill))
                cm5.metric("Lots @ $10K", str(lots) if lots else "—")
            # Multi-leg trade breakdown (always render if legs are populated —
            # makes the actual execution explicit for spreads, flies, and C9 pairs).
            legs = setup_state.get("legs") or []
            if legs:
                st.markdown(_legs_block_html(legs), unsafe_allow_html=True)
            # Track record
            tr = scan.get("track_records", {}).get(setup_pick, {})
            if tr.get("fires_60d", 0) > 0:
                st.markdown(
                    f"<div style='padding:6px 10px; margin-top:6px; "
                    f"background:rgba(232,183,93,0.05); border-left:2px solid var(--accent); "
                    f"font-size:0.78rem;'>"
                    f"<b style='color:var(--accent);'>Track record (60d):</b> "
                    f"<b>{tr['fires_60d']}</b> fires · <b>{tr['win_rate_5bar']:.0f}%</b> win at +5 bars · "
                    f"<b>{tr['mean_5bar_return_bp']:+.1f} bp</b> avg return · "
                    f"sample = <b>{tr['sample_quality']}</b>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.caption(f"No fires for {display_name(setup_pick)} in last 60d (no track record).")

        # OHLC chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=history["date"], y=history["close"], mode="lines+markers",
                                   line=dict(color=ACCENT, width=1.8),
                                   marker=dict(size=4, color=ACCENT),
                                   name="close",
                                   hovertemplate="<b>%{x|%Y-%m-%d}</b><br>%{y:.4f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=history["date"], y=history["high"],
                                   line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=history["date"], y=history["low"],
                                   line=dict(width=0), fill="tonexty",
                                   fillcolor="rgba(232,183,93,0.10)",
                                   name="H–L band",
                                   hovertemplate="<b>%{x|%Y-%m-%d}</b><br>L: %{y:.4f}<extra></extra>"))
        fig.update_layout(
            xaxis=dict(title=None), yaxis=dict(title="Price", tickformat=".4f"),
            height=320, hovermode="x unified", showlegend=False,
            margin=dict(l=55, r=20, t=20, b=40),
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)


# =============================================================================
# Side-panel blocks (12)
# =============================================================================
def _block_methodology() -> None:
    text = (
        "Technicals subtab — TMIA-curated setup detectors.\n\n"
        "Setups: 11 trend (A1, A2*, A3*, A4, A5*, A6, A8, A10, A11, A12a, A12b, A15)\n"
        "       + 7 mean-reversion (B1, B3, B5, B6, B10*, B11, B13)\n"
        "       + STIR-specific (C3, C4, C5, C8 12/24/36, C9a, C9b)\n"
        "       + 3 composites (TREND/MR/FINAL) re-tuned per scope.\n\n"
        "Customizations: A2 EMA_100 · A3 90d percentile · A5 also flies (skip-N/A <55 bars)\n"
        "                · A12 split a/b (EMA_20 / EMA_50) · B10 90d windows\n"
        "                · C8 three variants (12/24/36 M, primary 24M)\n"
        "                · C9 split a/b (slope crossing / 5d trend)\n\n"
        "Each setup detector returns FIRED / NEAR / APPROACHING / FAR / N/A,\n"
        "with formula + thresholds + computed inputs in the per-cell hover tooltip.\n"
        "Trade levels (entry / stop / T1 / T2 / lots @ $10K risk) computed where applicable.\n"
        "60-day track record per setup as a quality filter."
    )
    attr = _tooltip_attr(text)
    st.markdown(
        f"<div title=\"{attr}\" style='cursor:help; padding:6px 8px; "
        f"background:var(--bg-elevated); border:1px dashed var(--border-default); "
        f"border-radius:6px; margin: 0.6rem 0 0.3rem 0;'>"
        f"<div style='font-size:0.7rem; color:var(--text-dim); "
        f"text-transform:uppercase; letter-spacing:0.06em; font-weight:600;'>"
        f"Methodology / setup catalog <span style='color:var(--accent-dim);'>ⓘ</span>"
        f"</div>"
        f"<div style='font-size:0.7rem; color:var(--text-muted); margin-top:2px;'>"
        f"26 setups · hover for full text</div></div>",
        unsafe_allow_html=True,
    )


def _block_scan_summary(scan: dict) -> None:
    _section_header("Universe scan summary",
                     tooltip="Live counts. Hover any block below for per-bucket reading.")
    fires = scan.get("fires_today", [])
    near = scan.get("near_today", [])
    n_long = sum(1 for f in fires if f.get("direction") == "LONG")
    n_short = sum(1 for f in fires if f.get("direction") == "SHORT")
    rows = [
        ("# contracts in scope", str(len(scan.get("contracts", []))), None, None, None),
        ("total fires today",     str(len(fires)),
            "red" if len(fires) > 5 else "amber" if len(fires) else None, None, None),
        ("# LONG fires",          str(n_long), "green" if n_long else None, None, None),
        ("# SHORT fires",         str(n_short), "red" if n_short else None, None, None),
        ("# NEAR (about to fire)", str(len(near)), "amber" if near else None, None, None),
        ("# errors",              str(len(scan.get("errors", []))),
            "red" if scan.get("errors") else None, None,
            ("\n".join(scan.get("errors", [])[:5]) if scan.get("errors") else None)),
    ]
    _kv_block(rows)


def _block_fire_density(scan: dict) -> None:
    _section_header("Fire density · Family × Direction",
                     tooltip=("Counts of FIRED signals by family and direction.\n"
                               "Helps spot regime concentration: 'all trend long' vs 'all MR short'."))
    cnt = {(FAMILY_TREND, "LONG"): 0, (FAMILY_TREND, "SHORT"): 0,
           (FAMILY_MR, "LONG"): 0, (FAMILY_MR, "SHORT"): 0,
           (FAMILY_STIR, "LONG"): 0, (FAMILY_STIR, "SHORT"): 0}
    for f in scan.get("fires_today", []):
        key = (f.get("family"), f.get("direction"))
        if key in cnt:
            cnt[key] += 1
    rows = [
        ("Trend  LONG / SHORT", f"{cnt[(FAMILY_TREND, 'LONG')]} / {cnt[(FAMILY_TREND, 'SHORT')]}",
            "accent", None, None),
        ("MR     LONG / SHORT", f"{cnt[(FAMILY_MR, 'LONG')]} / {cnt[(FAMILY_MR, 'SHORT')]}",
            "accent", None, None),
        ("STIR   LONG / SHORT", f"{cnt[(FAMILY_STIR, 'LONG')]} / {cnt[(FAMILY_STIR, 'SHORT')]}",
            "accent", None, None),
    ]
    _kv_block(rows)


def _block_hot_setups(scan: dict) -> None:
    _section_header("Hot setups · most fires today",
                     tooltip="Setup IDs with the most fires today across the scope.")
    fires = scan.get("fires_today", [])
    if not fires:
        st.caption("No fires.")
        return
    by_sid = {}
    for f in fires:
        by_sid.setdefault(f["setup_id"], 0)
        by_sid[f["setup_id"]] += 1
    rows = []
    for sid, n in sorted(by_sid.items(), key=lambda kv: -kv[1])[:6]:
        rows.append((display_name(sid), str(n), "red" if n >= 3 else "amber", None,
                       display_name(sid)))
    _kv_block(rows)


def _block_quiet_setups(scan: dict) -> None:
    _section_header("Quiet setups · zero fires today",
                     tooltip="Setups in scope with zero fires today. Useful for sanity-checking the engine.")
    fires = scan.get("fires_today", [])
    fired_sids = set(f["setup_id"] for f in fires)
    scope = scan.get("scope", "outright")
    quiet = [s for s in setups_for_scope(scope) if s not in fired_sids]
    if not quiet:
        st.caption("All setups have at least one fire.")
        return
    st.markdown(
        f"<div style='font-family:JetBrains Mono, monospace; font-size:0.7rem; "
        f"color:var(--text-dim); padding:4px 0;'>"
        f"{' · '.join(display_name(s) for s in quiet[:18])}</div>",
        unsafe_allow_html=True,
    )


def _block_confluent_contracts(scan: dict) -> None:
    _section_header("Confluent contracts · ≥3 setups same direction",
                     tooltip=("Contracts with ≥3 FIRED setups on the SAME direction. "
                               "Highest-conviction setups in the scope today."))
    by_contract = scan.get("by_contract", {})
    confluent = []
    for sym, results in by_contract.items():
        n_long = 0; n_short = 0
        for sid, r in results.items():
            if sid.startswith("_") or not isinstance(r, dict):
                continue
            if r.get("state") == "FIRED":
                if r.get("direction") == "LONG":
                    n_long += 1
                elif r.get("direction") == "SHORT":
                    n_short += 1
        if n_long >= 3:
            confluent.append((sym, n_long, "LONG"))
        if n_short >= 3:
            confluent.append((sym, n_short, "SHORT"))
    confluent.sort(key=lambda x: -x[1])
    if not confluent:
        st.caption("No confluent (≥3 same-direction) contracts.")
        return
    for sym, n, direction in confluent[:6]:
        color = "var(--green)" if direction == "LONG" else "var(--red)"
        arrow = "▲" if direction == "LONG" else "▼"
        st.markdown(
            f"<div style='font-family:JetBrains Mono, monospace; font-size:0.7rem; "
            f"padding:3px 0; color:var(--text-body); "
            f"border-bottom:1px solid var(--border-subtle); "
            f"display:flex; justify-content:space-between; align-items:center;'>"
            f"<span><span style='color:var(--accent);'>{sym}</span> · "
            f"<b style='color:{color};'>{arrow} {n} setups {direction}</b></span>"
            f"<span style='color:var(--text-dim); font-size:0.65rem;'>STRONG ↑↓</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


def _block_regime_conflicts(scan: dict) -> None:
    _section_header("Regime conflicts · trend ↑ + MR ↓ (sit-out list)",
                     tooltip=("Contracts where TREND fires LONG AND MR fires SHORT, "
                               "or vice versa. Mixed signals — sit out."))
    by_contract = scan.get("by_contract", {})
    conflicts = []
    for sym, results in by_contract.items():
        trend_long = trend_short = mr_long = mr_short = False
        for sid, r in results.items():
            if not isinstance(r, dict) or r.get("state") != "FIRED":
                continue
            fam = r.get("family")
            d = r.get("direction")
            if fam == FAMILY_TREND and d == "LONG":  trend_long = True
            elif fam == FAMILY_TREND and d == "SHORT": trend_short = True
            elif fam == FAMILY_MR and d == "LONG":     mr_long = True
            elif fam == FAMILY_MR and d == "SHORT":    mr_short = True
        if (trend_long and mr_short) or (trend_short and mr_long):
            conflicts.append(sym)
    if not conflicts:
        st.caption("No regime conflicts — universe is directionally consistent.")
        return
    for sym in conflicts[:6]:
        st.markdown(
            f"<div style='font-family:JetBrains Mono, monospace; font-size:0.7rem; "
            f"padding:3px 0; border-bottom:1px solid var(--border-subtle);'>"
            f"<span style='color:var(--accent);'>{sym}</span> · "
            f"<span style='color:var(--amber);'>trend ↑↓ + MR ↓↑ — CONFLICT</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


def _block_composite_distribution(scan: dict) -> None:
    _section_header("Composite distribution · FINAL across universe",
                     tooltip=("Mini-histogram of FINAL_COMPOSITE values. Tail markers at "
                               "±0.3 / ±0.5 / ±0.7 — the soft / moderate / strong thresholds."))
    by_contract = scan.get("by_contract", {})
    values = []
    for sym, results in by_contract.items():
        comp = results.get("_composites", {}) or {}
        v = comp.get("final_score")
        if v is not None and np.isfinite(v):
            values.append(v)
    if not values:
        st.caption("No composite data.")
        return
    n_strong_up = sum(1 for v in values if v >= 0.7)
    n_strong_dn = sum(1 for v in values if v <= -0.7)
    n_mod_up = sum(1 for v in values if 0.5 <= v < 0.7)
    n_mod_dn = sum(1 for v in values if -0.7 < v <= -0.5)
    rows = [
        ("FINAL ≥ +0.7 (STRONG ↑)", str(n_strong_up), "green" if n_strong_up else None, None, None),
        ("+0.5 ≤ FINAL < +0.7",      str(n_mod_up),    "green" if n_mod_up else None, None, None),
        ("|FINAL| < 0.3 (neutral)",   str(sum(1 for v in values if abs(v) < 0.3)),
            None, None, None),
        ("-0.7 < FINAL ≤ -0.5",      str(n_mod_dn),    "red" if n_mod_dn else None, None, None),
        ("FINAL ≤ -0.7 (STRONG ↓)", str(n_strong_dn), "red" if n_strong_dn else None, None, None),
    ]
    _kv_block(rows)


def _block_regime_distribution(scan: dict) -> None:
    _section_header("Regime distribution (gameplan §7.5 labels)",
                     tooltip=("ADX-state × volatility-state regime classification.\n"
                               "TRENDING_HIGH_VOL / TRENDING_LOW_VOL / RANGING_HIGH_VOL / "
                               "RANGING_LOW_VOL / NEUTRAL"))
    by_contract = scan.get("by_contract", {})
    counts = {}
    for sym, results in by_contract.items():
        comp = results.get("_composites", {}) or {}
        rg = comp.get("regime", "NEUTRAL")
        counts[rg] = counts.get(rg, 0) + 1
    rows = [(k, str(v), None, None, None) for k, v in sorted(counts.items(), key=lambda kv: -kv[1])]
    if not rows:
        st.caption("No regime data.")
    else:
        _kv_block(rows)


def _block_setup_track_records(scan: dict) -> None:
    _section_header("60d track records (top setups)",
                     tooltip=("For each setup, count of fires in last 60 trading days × +5-bar "
                               "win rate × mean +5-bar return in bp.\n"
                               "Quality: STRONG ≥10 fires · OK 3-9 · WEAK 1-2 · NO_TRACK_RECORD 0\n"
                               "Use as a quality filter — only act on fires from setups with "
                               "positive recent expectancy."))
    track = scan.get("track_records", {})
    if not track:
        st.caption("No track records.")
        return
    rows = []
    # Sort by mean_5bar_return_bp descending
    sortable = [(sid, tr) for sid, tr in track.items()
                 if tr.get("mean_5bar_return_bp") is not None]
    sortable.sort(key=lambda x: -(x[1].get("mean_5bar_return_bp") or 0))
    for sid, tr in sortable[:8]:
        n = tr.get("fires_60d", 0); wr = tr.get("win_rate_5bar")
        mr = tr.get("mean_5bar_return_bp")
        cls = "green" if (mr is not None and mr > 0) else ("red" if (mr is not None and mr < 0) else None)
        val = (f"{n} fires · {wr:.0f}% win · {mr:+.1f}bp"
               if wr is not None and mr is not None else "—")
        sub = f"({tr.get('sample_quality', '')})"
        rows.append((sid, val, cls, sub, None))
    if rows:
        _kv_block(rows)


def _block_pattern_legend() -> None:
    with st.expander("Setup-pattern legend", expanded=False):
        legend = {
            "FIRED":       "all conditions met for the setup — actionable signal today",
            "NEAR":        "1 condition missing — about to fire (≤ 1 day typically)",
            "APPROACHING": "≥ half of conditions met — multiple gates still open",
            "FAR":         "few/no conditions met — no setup",
            "N/A":         "data unavailable (insufficient history, missing column, runtime error)",
            "Confluent ↑↓": "≥3 setups firing same direction on same contract — high-conviction",
            "Regime conflict": "trend ↑ + MR ↓ on same contract — mixed signal, sit out",
        }
        for k, v in legend.items():
            st.markdown(
                f"<div style='padding:4px 0; border-bottom:1px solid var(--border-subtle);'>"
                f"<span style='color:var(--accent); font-family:JetBrains Mono, monospace; "
                f"font-weight:600; font-size:0.78rem;'>{k}</span>"
                f"<span style='color:var(--text-body); margin-left:0.5rem;'>{v}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )


def _render_combined_side_panel(scan: dict) -> None:
    st.markdown(
        f"<div style='display:flex; align-items:center; gap:0.5rem; "
        f"padding-bottom:0.4rem; border-bottom:1px solid var(--border-subtle); margin-bottom:0.4rem;'>"
        f"<span style='color:var(--accent); font-size:0.85rem; font-weight:600;'>📊 Analysis · all blocks (scroll)</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    _block_methodology()
    _block_scan_summary(scan)
    _block_fire_density(scan)
    _block_hot_setups(scan)
    _block_quiet_setups(scan)
    _block_confluent_contracts(scan)
    _block_regime_conflicts(scan)
    _block_composite_distribution(scan)
    _block_regime_distribution(scan)
    _block_setup_track_records(scan)
    _block_trend_movers(scan)               # Phase H: 12th block (parity with Proximity/Z-score)
    _block_pattern_legend()


# Phase H: 12th block — improving / declining setups from Phase D trend table.
def _block_trend_movers(scan: dict) -> None:
    """Read the Phase D ``tmia_backtest_trend`` table and surface the top
    'improving' and 'declining' setups based on the cross-window
    Spearman-style trend interpretation. Only renders if the backtest
    cycle has populated ``tmia.duckdb``; otherwise shows a friendly
    placeholder pointing to the manual-recompute button.
    """
    _section_header("Trend movers · improving vs declining (5y → 30d)",
                     tooltip=("From the 10-day backtest grid: per setup × node, "
                               "we compare the metric across 5 windows (ALL / 2Y / "
                               "1Y / 6M / 30D) and tag each cell as Improving / "
                               "Declining / Stable / Mixed. Top movers shown here."))
    try:
        from lib.backtest.cycle import load_trend_table, get_backtest_status
    except Exception:
        st.caption("Backtest cycle module not available.")
        return
    status = get_backtest_status()
    if not status.get("fresh"):
        st.markdown(
            "<div style='padding:6px 8px; background:var(--bg-surface); "
            "border:1px solid var(--border-subtle); border-radius:4px; "
            "color:var(--text-dim); font-size:0.7rem;'>"
            "10-day backtest cache not yet built. The recompute daemon spawns "
            "on app start; first run takes ~5-15 min. Status: "
            f"{status.get('last_run') or '— no prior run —'}"
            "</div>", unsafe_allow_html=True)
        return
    trend = load_trend_table()
    if trend is None or trend.empty:
        st.caption("Trend table empty — backtest produced no trades.")
        return
    # Pick L1-level rows (most granular: setup × node × direction)
    l1 = trend[trend["level"] == "L1"].copy()
    if l1.empty:
        return
    improving = l1[l1["trend_text"].str.startswith("Improving", na=False)]
    declining = l1[l1["trend_text"].str.startswith("Declining", na=False)]
    rows = []
    for _, r in improving.head(4).iterrows():
        rows.append((display_name(r["setup_id"]) + f" · {r['cmc_node']} {r['direction']}",
                       "▲", "green", r["trend_text"][:80] + "…", r["trend_text"]))
    for _, r in declining.head(4).iterrows():
        rows.append((display_name(r["setup_id"]) + f" · {r['cmc_node']} {r['direction']}",
                       "▼", "red", r["trend_text"][:80] + "…", r["trend_text"]))
    if not rows:
        st.caption("No improving / declining cells detected — all stable or insufficient data.")
        return
    _kv_block(rows)


# =============================================================================
# Top-level render
# =============================================================================
def render(base_product: str = "SRA") -> None:
    _set_market(base_product)
    # Inject local CSS once per render (idempotent at the DOM level)
    st.markdown(_TECHNICALS_CSS, unsafe_allow_html=True)

    snap_date = get_sra_snapshot_latest_date()
    if snap_date is None:
        st.error("OHLC database / SRA snapshot unavailable.")
        return

    n_outright = len(get_outrights())
    spread_tenors = get_available_tenors("spread")
    fly_tenors = get_available_tenors("fly")
    n_spread = sum(len(get_spreads(t)) for t in spread_tenors)
    n_fly = sum(len(get_flies(t)) for t in fly_tenors)
    status_strip_with_dot([
        ("Snapshot", f"{snap_date}", "accent"),
        ("Outrights", f"{n_outright}", None),
        ("Spreads",   f"{n_spread} · {len(spread_tenors)} tenors", None),
        ("Flies",     f"{n_fly} · {len(fly_tenors)} tenors", None),
        ("Liveness",  f"≤ {LIVENESS_DAYS}d", None),
    ], dot_label="Live")

    # Phase H: align with Proximity / Z-score subtabs (was [1.0, 1.4, 2.0, 1.4])
    cc1, cc2, cc3, cc4 = st.columns([1.0, 1.4, 2.4, 1.4])
    with cc1:
        strategy_label = st.segmented_control(
            "Strategy", options=["Outright", "Spread", "Fly"],
            default="Outright", key=f"{_SCOPE_PREFIX}_strat",
        )
    with cc2:
        if strategy_label == "Outright":
            tenor_months = None
            st.markdown(
                "<div style='padding-top:1.7rem; color:var(--text-dim); font-size:0.75rem;'>"
                "All quarterly + serial outrights</div>", unsafe_allow_html=True)
        elif strategy_label == "Spread":
            tenor_months = st.selectbox(
                "Tenor", options=spread_tenors, index=min(1, max(0, len(spread_tenors) - 1)),
                key=f"{_SCOPE_PREFIX}_tenor_spread", format_func=lambda t: f"{t}M",
            )
        else:
            tenor_months = st.selectbox(
                "Tenor", options=fly_tenors, index=0,
                key=f"{_SCOPE_PREFIX}_tenor_fly", format_func=lambda t: f"{t}M",
            )
    with cc3:
        family_filter = st.multiselect(
            "Family filter",
            options=["Trend", "MR", "STIR", "Composites"],
            default=["Trend", "MR", "STIR", "Composites"],
            key=f"{_SCOPE_PREFIX}_family",
        )
    with cc4:
        analysis_on = st.toggle("Analysis side panel", value=True,
                                  key=f"{_SCOPE_PREFIX}_analysis_on")

    # View toggles
    vc1, vc2, vc3, vc4 = st.columns(4)
    with vc1:
        show_fires = st.checkbox("Today's fires", value=True, key=f"{_SCOPE_PREFIX}_v_fires")
    with vc2:
        show_matrix = st.checkbox("State matrix", value=True, key=f"{_SCOPE_PREFIX}_v_matrix")
    with vc3:
        show_comp = st.checkbox("Composite scoring", value=True, key=f"{_SCOPE_PREFIX}_v_comp")
    with vc4:
        show_near = st.checkbox("Proximity-to-fire", value=True, key=f"{_SCOPE_PREFIX}_v_near")

    # Interpretation guide
    scope = strategy_label.lower()
    _render_interpretation_guide_technicals(scope)

    # Make sure pre-warm has been kicked off (idempotent — no-op if already running).
    # If app.py wasn't the entry point, this kicks it off now.
    ensure_prewarm()

    # Show pre-warm status if relevant
    pw_status = get_prewarm_status()
    if pw_status.get("started_at") and not pw_status.get("outright_done_at"):
        # Pre-warm in progress — pop a small banner
        elapsed = max(0.0, (
            __import__("time").time() - pw_status["started_at"]
        ))
        st.markdown(
            f"<div style='padding:6px 10px; margin: 0.4rem 0; "
            f"background:rgba(232,183,93,0.10); border:1px solid rgba(232,183,93,0.35); "
            f"border-radius:6px; color:var(--text-body); font-size:0.78rem;'>"
            f"⏳ <b style='color:var(--accent);'>Pre-warm in progress</b> "
            f"(~{elapsed:.0f}s elapsed). Background scan still computing — "
            f"first navigation completes the cache. Subsequent loads instant."
            f"</div>",
            unsafe_allow_html=True,
        )

    # Run scan (cached). If pre-warm has already populated the cache for this
    # (scope, tenor, asof) tuple, this returns instantly. Otherwise it
    # blocks until the scan completes and populates the cache.
    spinner_text = (
        f"Scanning SRA {strategy_label.lower()} universe — using cached pre-warm "
        if is_prewarm_done() else
        f"Scanning SRA {strategy_label.lower()} universe (first run ~30-60s; cached after)..."
    )
    with st.spinner(spinner_text):
        try:
            scan = scan_universe(scope, tenor_months, snap_date.isoformat(),
                                    history_days=280, base_product=_BP)
        except Exception as e:
            st.error(f"Scan failed: {e}")
            with st.expander("Traceback"):
                import traceback as _tb
                st.code(_tb.format_exc())
            return

    contracts = scan.get("contracts", []) or []
    if not contracts:
        st.warning("No live contracts in scope.")
        return

    scope_label = (f"SRA · {strategy_label} · {tenor_months}M"
                    if tenor_months else f"SRA · {strategy_label}")
    range_str = contract_range_str(
        get_outrights() if scope == "outright"
        else get_spreads(tenor_months) if scope == "spread"
        else get_flies(tenor_months)
    )
    n_fires = len(scan.get("fires_today", []))
    n_near = len(scan.get("near_today", []))
    status_strip_with_dot([
        ("Scope", scope_label, "accent"),
        ("Contracts", f"{len(contracts)}", None),
        ("Range", range_str, None),
        ("Fires today", f"{n_fires}", "red" if n_fires > 5 else "amber" if n_fires else None),
        ("Near", f"{n_near}", "amber" if n_near else None),
    ], dot_label="Live")

    # Helper to render a view section header with title + meta
    def _view_header(title: str, meta: str = "") -> None:
        st.markdown(
            f"<div class='tech-view-header'>"
            f"<span class='title'>{title}</span>"
            f"<span class='meta'>{meta}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    fire_count = len(scan.get("fires_today", []))
    near_count = len(scan.get("near_today", []))

    # Layout
    chart_col, side_col = st.columns([3, 1.1])
    with chart_col:
        if show_fires:
            _view_header("View 1 · Today's fires",
                          f"{fire_count} fires · grouped by family")
            try:
                _render_view_today_fires(scan)
            except Exception as e:
                st.error(f"View 1 failed: {e}")

        if show_matrix:
            _view_header("View 2 · Setup state matrix",
                          f"top 25 contracts · {len(setups_for_scope(scope))} setups · hover any cell for thresholds + computed inputs")
            try:
                _render_view_state_matrix(scan, top_n=25)
            except Exception as e:
                st.error(f"View 2 failed: {e}")

        if show_comp:
            _view_header("View 3 · Composite scoring",
                          f"TREND / MR / FINAL · re-tuned for {scope} scope · sorted by |FINAL|")
            try:
                _render_view_composites(scan)
            except Exception as e:
                st.error(f"View 3 failed: {e}")

        if show_near:
            _view_header("View 4 · Proximity-to-fire",
                          f"{near_count} NEAR signals · sorted by closeness ascending")
            try:
                _render_view_near(scan, top_n=25)
            except Exception as e:
                st.error(f"View 4 failed: {e}")

        try:
            _render_drill_down(scan, snap_date)
        except Exception as e:
            st.error(f"Drill-down failed: {e}")

    with side_col:
        if analysis_on:
            # Phase H: visible side-panel border for parity with sibling subtabs
            with st.container(height=720, border=True):
                try:
                    _render_combined_side_panel(scan)
                except Exception as e:
                    st.error(f"Side panel failed: {e}")
                    with st.expander("Traceback"):
                        import traceback as _tb
                        st.code(_tb.format_exc())
