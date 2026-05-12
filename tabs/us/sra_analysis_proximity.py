"""SRA → Analysis → Proximity subtab.

UI (left → right):

  status strip + section boundaries expander
  strategy / tenor / lookback / top-K / view toggles / side-panel switch
  ┌──────────────────────────────────┬───────────────────────┐
  │ View 1: Section Ribbons (3 cols) │ Side panel (scroll)   │
  │ View 2: Confluence Matrix         │  · 12 analytics blocks │
  │ View 3: Cluster / Density Heatmap │   each with tooltips   │
  │ View 4: Drill-down expander       │                       │
  └──────────────────────────────────┴───────────────────────┘

Every numeric cell carries a 4-line tooltip (title / formula / inputs /
interpretation). Distance metric defaults to ATR-normalised. Bp distances
respect the catalog (no double-×100 on stored-bp spreads/flies).
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from lib.charts import (
    make_confluence_matrix_chart,
    make_density_heatmap_chart,
    make_proximity_drill_chart,
    make_proximity_ribbon_chart,
)
from lib.components import status_strip_with_dot
from lib.proximity import (
    APPROACH_ATR, AT_ATR, FAILED_BREAK_LOOKBACK_BARS, NEAR_ATR, ATR_PERIOD,
    classify_proximity_pattern,
    compute_cluster_signal,
    compute_proximity_panel,
    get_pattern_descriptions,
    get_proximity_interpretation_guide,
    get_proximity_thresholds_text,
    proximity_section_regime,
    rank_closest_to_extreme,
)
from lib.mean_reversion import compute_zscore_panel, metric_tooltip
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
    load_sra_curve_panel,
    pivot_curve_panel,
    section_label_for_index,
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


def load_sra_curve_panel(strategy, tenor, start, end):
    if _BP == "SRA":
        from lib.sra_data import load_sra_curve_panel as _f
        return _f(strategy, tenor, start, end)
    return _md.load_curve_panel(_BP, strategy, tenor, start, end)


def get_sra_snapshot_latest_date():
    if _BP == "SRA":
        from lib.sra_data import get_sra_snapshot_latest_date as _f
        return _f()
    return _md.get_snapshot_latest_date(_BP)
from lib.theme import (
    ACCENT, AMBER, BLUE, GREEN, PURPLE, RED, TEXT_BODY, TEXT_DIM, TEXT_HEADING, TEXT_MUTED,
)


# Locked at 90d max for analysis (per HANDOFF §8.8)
_LOOKBACK_OPTIONS = [5, 15, 30, 60, 90]
_SCOPE_PREFIX = "prox"


# =============================================================================
# Helpers — formatting, tooltips, side-panel atoms
# =============================================================================
def _flag_color(flag: str) -> str:
    return {"AT": "var(--red)", "NEAR": "var(--amber)",
            "APPROACHING": "var(--blue)"}.get(flag, "var(--text-dim)")


def _flag_pill(flag: str) -> str:
    """HTML pill for a proximity flag (AT/NEAR/APPROACHING/FAR)."""
    color = _flag_color(flag)
    return (f"<span style='display:inline-block; padding:1px 6px; "
            f"border:1px solid {color}; color:{color}; "
            f"font-size:0.65rem; font-family:JetBrains Mono, monospace; "
            f"font-weight:600; border-radius:4px; line-height:1;'>{flag}</span>")


def _pattern_pill(pat: str) -> str:
    """HTML pill for a confluence pattern."""
    color_map = {
        "PERSISTENT": "var(--red)", "ACCELERATING": "var(--red)",
        "DECELERATING": "var(--amber)", "FRESH": "var(--blue)",
        "DRIFTED": "var(--amber)", "REVERTING": "var(--green)",
        "DIVERGENT": "var(--purple, #a78bfa)", "STABLE": "var(--text-dim)",
        "MIXED": "var(--text-muted)",
    }
    color = color_map.get(pat, "var(--text-body)")
    return (f"<span style='display:inline-block; padding:1px 6px; "
            f"border:1px solid {color}; color:{color}; "
            f"font-size:0.65rem; font-family:JetBrains Mono, monospace; "
            f"font-weight:600; border-radius:4px; line-height:1;'>{pat}</span>")


def _tooltip_attr(text: str) -> str:
    """Sanitise a multi-line string for an HTML title-attribute tooltip."""
    return (str(text)
            .replace('"', '&quot;')
            .replace("'", "&#39;")
            .replace("\n", "&#10;"))


def _kv_block(rows: list) -> None:
    """Tuples: (label, value, color_class, sub_text, tooltip_text). Last 3 are optional.
    A non-empty tooltip wraps the row with a title-attribute hover."""
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


def _mt(metric_key: str, current_label: Optional[str] = None,
         computed: Optional[str] = None) -> str:
    """Build a per-metric tooltip from the proximity guide."""
    return metric_tooltip(get_proximity_interpretation_guide(), metric_key,
                           current_label=current_label, computed=computed)


def _flag_label_for_dist(dist_atr) -> Optional[str]:
    """Map a numeric distance-in-ATR to its bucket label (matches the guide)."""
    if dist_atr is None or pd.isna(dist_atr):
        return None
    if dist_atr <= AT_ATR:
        return "AT"
    if dist_atr <= NEAR_ATR:
        return "NEAR"
    if dist_atr <= APPROACH_ATR:
        return "APPROACHING"
    return "FAR"


def _streak_label(n: int) -> str:
    if n >= 5:
        return "very persistent"
    if n >= 3:
        return "persistent"
    if n >= 1:
        return "fresh"
    return "no streak"


def _velocity_label(v) -> Optional[str]:
    if v is None or pd.isna(v):
        return None
    if v <= -0.10:
        return "rapidly closing in"
    if v < -0.02:
        return "drifting toward extreme"
    if v <= 0.02:
        return "stable"
    if v <= 0.10:
        return "backing off"
    return "rapidly moving away"


def _touch_label(n: int) -> str:
    if n >= 5:
        return "heavy coiling"
    if n >= 3:
        return "coiling"
    if n >= 1:
        return "tested"
    return "untested"


def _pir_label(pir) -> Optional[str]:
    if pir is None or pd.isna(pir):
        return None
    if pir >= 0.90:
        return "near top of range"
    if pir >= 0.70:
        return "upper third"
    if pir >= 0.30:
        return "middle of range"
    if pir >= 0.10:
        return "lower third"
    return "near bottom of range"


def _render_interpretation_guide_proximity() -> None:
    """Top-of-subtab expander explaining every proximity metric and bucket."""
    guide = get_proximity_interpretation_guide()
    with st.expander("📖 Interpretation guide — click to see what every proximity metric means",
                      expanded=False):
        st.markdown(
            "<div style='color:var(--text-muted); font-size:0.78rem; margin-bottom:0.5rem;'>"
            "Every metric on this subtab is paired with a plain-English label so you don't "
            "have to memorise the bucket boundaries. This guide lists each metric, its formula, "
            "and the threshold ranges that drive the interpretation."
            "</div>",
            unsafe_allow_html=True,
        )
        for metric_name, info in guide.items():
            st.markdown(
                f"<div style='margin-top:0.6rem; padding:6px 8px; "
                f"background:var(--bg-surface); border:1px solid var(--border-subtle); "
                f"border-radius:6px;'>"
                f"<div style='color:var(--accent); font-weight:600; font-size:0.85rem; "
                f"margin-bottom:0.2rem;'>{metric_name}</div>"
                f"<div style='color:var(--text-body); font-size:0.75rem; line-height:1.4;'>"
                f"{info.get('what', '')}</div>"
                f"<div style='color:var(--text-dim); font-size:0.7rem; "
                f"font-family:JetBrains Mono, monospace; margin-top:0.2rem;'>"
                f"{info.get('formula', '')}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            buckets = info.get("buckets", [])
            if buckets:
                rows_html = []
                for cond, label, meaning in buckets:
                    rows_html.append(
                        f"<div style='display:grid; "
                        f"grid-template-columns: 1fr 1.2fr 2.2fr; gap:8px; "
                        f"padding:3px 8px; font-size:0.72rem; "
                        f"border-bottom:1px solid var(--border-subtle);'>"
                        f"<span style='color:var(--text-muted); "
                        f"font-family:JetBrains Mono, monospace;'>{cond}</span>"
                        f"<span style='color:var(--accent-bright); font-weight:500;'>{label}</span>"
                        f"<span style='color:var(--text-body);'>{meaning}</span>"
                        f"</div>"
                    )
                st.markdown("".join(rows_html), unsafe_allow_html=True)


def _section_header(text: str, tooltip: str = "") -> None:
    """Sub-section header with optional ⓘ tooltip via title attribute."""
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


# =============================================================================
# Cached panel build — one (strategy, tenor, asof) → engine result
# =============================================================================
@st.cache_data(show_spinner=False, ttl=600)
def _build_proximity_data(strategy: str, tenor_months: Optional[int],
                           asof_str: str, lookbacks_tuple: tuple,
                           history_days: int = 252,
                           base_product: str = "SRA") -> dict:
    """Build the full proximity result + Z-score result for a single strategy bucket.

    Returns a dict with ``contracts``, ``wide_close``, ``wide_high``, ``wide_low``,
    ``proximity``, ``zscore``, ``info`` (range strings, etc).

    Cache key includes base_product so each market gets its own panel (previously
    omitted, causing cross-market data bleeding).
    """
    _set_market(base_product)
    asof_date = pd.Timestamp(asof_str).date()
    if strategy == "outright":
        syms_df = get_outrights()
    elif strategy == "spread":
        syms_df = get_spreads(int(tenor_months)) if tenor_months else pd.DataFrame()
    elif strategy == "fly":
        syms_df = get_flies(int(tenor_months)) if tenor_months else pd.DataFrame()
    else:
        syms_df = pd.DataFrame()

    out = {
        "asof": asof_date,
        "strategy": strategy,
        "tenor_months": tenor_months,
        "base_product": base_product,
        "contracts": [],
        "wide_close": pd.DataFrame(),
        "wide_high": pd.DataFrame(),
        "wide_low": pd.DataFrame(),
        "proximity": {},
        "zscore": {},
        "info": {"range_str": "—", "n_total": 0},
    }
    if syms_df is None or syms_df.empty:
        return out
    contracts = list(syms_df["symbol"])
    out["contracts"] = contracts
    out["info"]["range_str"] = contract_range_str(syms_df)
    out["info"]["n_total"] = len(contracts)

    end_date = asof_date
    start_date = asof_date - timedelta(days=int(history_days * 1.6) + 30)
    panel = load_sra_curve_panel(strategy, tenor_months, start_date, end_date)
    if panel is None or panel.empty:
        return out
    wide_close = pivot_curve_panel(panel, contracts, "close")
    wide_high = pivot_curve_panel(panel, contracts, "high")
    wide_low = pivot_curve_panel(panel, contracts, "low")
    out["wide_close"] = wide_close
    out["wide_high"] = wide_high
    out["wide_low"] = wide_low

    lookbacks = list(lookbacks_tuple)
    out["proximity"] = compute_proximity_panel(
        wide_close, wide_high, wide_low, contracts,
        asof_date, lookbacks, base_product=base_product, atr_period=ATR_PERIOD,
    )
    # Pair with Z-score panel so the side-panel cross-check block can run
    out["zscore"] = compute_zscore_panel(
        wide_close, contracts, asof_date, lookbacks,
        base_product=base_product, long_lookback_for_tests=max(lookbacks),
    )
    return out


# =============================================================================
# Universe summary — KPIs at the top of the chart column
# =============================================================================
def _proximity_kpis(panel: dict, contracts: list[str], lookback: int) -> dict:
    n_total = len(contracts)
    n_at_high = n_at_low = n_near_high = n_near_low = 0
    n_appr_high = n_appr_low = 0
    pirs = []
    for sym in contracts:
        rec = panel.get("per_contract", {}).get(sym) or {}
        m = rec.get("by_lookback", {}).get(lookback) or {}
        fh = m.get("flag_high"); fl = m.get("flag_low")
        if fh == "AT": n_at_high += 1
        elif fh == "NEAR": n_near_high += 1
        elif fh == "APPROACHING": n_appr_high += 1
        if fl == "AT": n_at_low += 1
        elif fl == "NEAR": n_near_low += 1
        elif fl == "APPROACHING": n_appr_low += 1
        pir = m.get("position_in_range")
        if pir is not None and not pd.isna(pir):
            pirs.append(pir)
    return {
        "n_total": n_total,
        "n_at_high": n_at_high, "n_at_low": n_at_low,
        "n_near_high": n_near_high, "n_near_low": n_near_low,
        "n_appr_high": n_appr_high, "n_appr_low": n_appr_low,
        "avg_pir": float(np.mean(pirs)) if pirs else None,
        "median_pir": float(np.median(pirs)) if pirs else None,
    }


# =============================================================================
# View 1 — Section Ribbons (top-K closest to HIGH / LOW per section)
# =============================================================================
def _render_section_ribbons(panel: dict, contracts: list[str], front_end: int,
                              mid_end: int, lookback: int, top_k: int) -> None:
    fr_rng, mr_rng, br_rng = compute_section_split(len(contracts), front_end, mid_end)
    sections = [
        ("FRONT", fr_rng, "rgba(251, 191, 36, 0.12)"),
        ("MID",   mr_rng, "rgba(34, 211, 238, 0.10)"),
        ("BACK",  br_rng, "rgba(167, 139, 250, 0.10)"),
    ]
    cols = st.columns(3)
    for col_idx, (name, rng, _) in enumerate(sections):
        sec_contracts = [contracts[i] for i in rng]
        with cols[col_idx]:
            n_count = len(sec_contracts)
            st.markdown(
                f"<div style='font-size:0.78rem; font-weight:600; color:var(--text-heading); "
                f"padding:6px 8px; background:var(--bg-surface); border:1px solid var(--border-subtle); "
                f"border-radius:6px 6px 0 0; margin-bottom:0;'>"
                f"{name} <span style='color:var(--text-dim); font-weight:400;'>· {n_count} contracts</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            if n_count == 0:
                st.caption("Section empty.")
                continue
            for side, side_color in [("high", "var(--red)"), ("low", "var(--green)")]:
                rows = rank_closest_to_extreme(panel, sec_contracts, lookback, side, top_k=top_k)
                title = f"Closest to {lookback}d {side.upper()}"
                st.markdown(
                    f"<div style='font-size:0.7rem; color:var(--text-muted); "
                    f"padding:6px 8px 4px 8px; border-left:2px solid {side_color}; "
                    f"margin-top:6px;'>{title}</div>",
                    unsafe_allow_html=True,
                )
                if not rows:
                    st.caption("No data.")
                    continue
                # Use a horizontal-bar chart for compactness
                fig = make_proximity_ribbon_chart(rows, side=side, title="",
                                                   height=max(180, 32 * len(rows) + 60))
                st.plotly_chart(fig, use_container_width=True, theme=None)
                # Optional terse hover-able row dump (table)
                _render_ribbon_table(rows, side, lookback)


def _render_ribbon_table(rows: list[dict], side: str, lookback: int) -> None:
    """Compact table beneath each ribbon with full tooltipped values."""
    if not rows:
        return
    parts = ['<div style="font-family:JetBrains Mono, monospace; font-size:0.68rem; '
             'margin-top:4px;">']
    parts.append(
        '<div style="display:grid; grid-template-columns: 1.1fr 0.8fr 0.8fr 0.7fr 0.6fr 0.7fr; '
        'gap:4px; padding:4px 6px; color:var(--text-dim); '
        'border-bottom:1px solid var(--border-subtle);">'
        '<span>SYMBOL</span><span style="text-align:right">CUR (bp)</span>'
        f'<span style="text-align:right">{side.upper()} (bp)</span>'
        '<span style="text-align:right">Δ ATR</span>'
        '<span style="text-align:center">FLAG</span>'
        '<span style="text-align:center">PATTERN</span>'
        '</div>'
    )
    def _fmt(v, fmt=":+.2f"):
        if v is None or pd.isna(v):
            return "—"
        return ("{0" + fmt + "}").format(v)

    for r in rows:
        sym = r.get("symbol", "")
        cur = r.get("current_bp")
        ext = r.get("extreme_bp")
        d_atr = r.get("dist_atr")
        d_bp = r.get("dist_bp")
        flag = r.get("flag", "—")
        pattern = r.get("pattern", "MIXED")
        streak = r.get("streak") or 0
        velocity = r.get("velocity_atr_per_day")
        # Tooltip per row — built defensively
        tt_lines = [f"{sym} · {lookback}d {side.upper()} proximity", ""]
        tt_lines.append(
            f"Formula:  flag = AT if dist≤{AT_ATR:.2f}·ATR; "
            f"NEAR≤{NEAR_ATR:.2f}; APPROACHING≤{APPROACH_ATR:.2f}; else FAR"
        )
        tt_lines.append(
            f"Inputs:   current {_fmt(cur)}bp · {side} {_fmt(ext)}bp · "
            f"dist {_fmt(d_bp)}bp / {_fmt(d_atr, ':.3f')} ATR · streak {streak}d"
        )
        if velocity is not None:
            tt_lines.append(f"          velocity {velocity:+.3f} ATR/day")
        if r.get("fresh_break"):
            tt_lines.append("          FRESH BREAK — today exceeded the prior N-day extreme")
        if r.get("failed_break"):
            tt_lines.append("          FAILED BREAK — touched within last 3 bars, now ≥0.5 ATR away")
        if r.get("touch_count") and r.get("touch_count") > 1:
            tt_lines.append(f"          touch count: {r['touch_count']} (re-tests)")
        if r.get("range_expansion_ratio"):
            tt_lines.append(f"          range expansion: ×{r['range_expansion_ratio']:.2f} vs prior {lookback}d")
        means = (
            "Trend confirmation candidate." if pattern in ("PERSISTENT", "ACCELERATING")
            else "Reversal / fade candidate." if pattern in ("DECELERATING", "REVERTING")
            else "Coiling pattern — watch for direction." if pattern == "DRIFTED"
            else "Event-driven move — needs longer-window confirmation." if pattern == "FRESH"
            else "Timeframe disagreement — range-bound." if pattern == "DIVERGENT"
            else "Quiet contract." if pattern == "STABLE"
            else "Neutral catch-all."
        )
        tt_lines.append(f"Means:    {means}")
        tt = "\n".join(tt_lines)
        # Per-metric tooltips
        flag_label = _flag_label_for_dist(d_atr) or flag
        tt_dist = _mt("Distance to extreme (ATR-normalised)", flag_label,
                       f"dist = {(d_bp is not None and f'{d_bp:+.2f}bp') or '—'}  /  "
                       f"ATR = {(d_atr is not None and f'{d_atr:.3f} ATR') or '—'}")
        tt_pat  = _mt("Confluence pattern (multi-window)", pattern,
                       f"computed across the focus + selected lookbacks")
        tt_cur  = ("Current close in basis points (bp), per the per-contract unit-conventions "
                   "catalog. Outrights ×100 from price; spreads/flies stored already in bp.")
        tt_ext  = (f"Rolling N-day {'high' if side=='high' else 'low'} (in bp). "
                   f"Computed strictly over the prior {lookback} bars (excludes today).")

        parts.append(
            f'<div title="{_tooltip_attr(tt)}" '
            f'style="display:grid; grid-template-columns: 1.1fr 0.8fr 0.8fr 0.7fr 0.6fr 0.7fr; '
            f'gap:4px; padding:3px 6px; align-items:center; cursor:help; '
            f'border-bottom:1px solid var(--border-subtle);">'
            f'<span style="color:var(--accent);">{sym}</span>'
            f'<span title="{_tooltip_attr(tt_cur)}" style="text-align:right; cursor:help;">{_fmt(cur)}</span>'
            f'<span title="{_tooltip_attr(tt_ext)}" style="text-align:right; cursor:help;">{_fmt(ext)}</span>'
            f'<span title="{_tooltip_attr(tt_dist)}" style="text-align:right; cursor:help;">{_fmt(d_atr, ":.2f")}</span>'
            f'<span title="{_tooltip_attr(tt_dist)}" style="text-align:center; cursor:help;">{_flag_pill(flag)}</span>'
            f'<span title="{_tooltip_attr(tt_pat)}" style="text-align:center; cursor:help;">{_pattern_pill(pattern)}</span>'
            f'</div>'
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


# =============================================================================
# View 2 — Confluence Matrix
# =============================================================================
def _render_confluence_matrix(panel: dict, contracts: list[str], lookbacks: list[int],
                               top_n: int = 25) -> None:
    """Confluence matrix — rows = top-N contracts by overall extreme proximity (any window).

    Cells = position-in-range (0=at low, 1=at high), colored on a diverging palette
    around 0.5. A pattern column is rendered as text on the right side.
    """
    # Score each contract by min(nearest_dist_atr) across selected lookbacks
    rows = []
    for sym in contracts:
        rec = panel.get("per_contract", {}).get(sym) or {}
        scores = []
        for n in lookbacks:
            m = rec.get("by_lookback", {}).get(n) or {}
            d = m.get("nearest_dist_atr")
            if d is not None and not pd.isna(d):
                scores.append(d)
        if not scores:
            continue
        rows.append((sym, min(scores), rec))
    rows.sort(key=lambda x: x[1])
    rows = rows[:top_n]

    if not rows:
        st.caption("Nothing in range.")
        return

    syms = [r[0] for r in rows]
    matrix = pd.DataFrame(index=syms, columns=lookbacks, dtype=float)
    pattern_col = pd.Series(index=syms, dtype=object)
    for sym, _, rec in rows:
        for n in lookbacks:
            m = rec.get("by_lookback", {}).get(n) or {}
            pir = m.get("position_in_range")
            matrix.loc[sym, n] = pir if pir is not None else np.nan
        pattern_col.loc[sym] = rec.get("pattern", "MIXED")

    fig = make_confluence_matrix_chart(
        matrix, pattern_col=pattern_col,
        title=f"Confluence — top {len(rows)} by extreme proximity (cells: position-in-range; 0=low / 1=high)",
        height=max(360, 22 * len(rows) + 80),
        z_mode=False,
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)


# =============================================================================
# View 3 — Cluster / Density Heatmap (Section × Lookback OR Section × Tenor)
# =============================================================================
def _render_cluster_heatmap(panel: dict, contracts: list[str], front_end: int,
                              mid_end: int, lookbacks: list[int]) -> None:
    """Two side-by-side heatmaps: HIGH-side density and LOW-side density,
    rows = sections (Front/Mid/Back), cols = lookbacks, cells = % AT/NEAR.
    """
    fr_rng, mr_rng, br_rng = compute_section_split(len(contracts), front_end, mid_end)
    sec_for: dict[str, str] = {}
    for i, sym in enumerate(contracts):
        if i in fr_rng:
            sec_for[sym] = "Front"
        elif i in mr_rng:
            sec_for[sym] = "Mid"
        else:
            sec_for[sym] = "Back"

    rows_order = ["Front", "Mid", "Back"]
    high_mat = pd.DataFrame(index=rows_order, columns=lookbacks, dtype=float)
    low_mat = pd.DataFrame(index=rows_order, columns=lookbacks, dtype=float)
    for n in lookbacks:
        cl = compute_cluster_signal(panel, contracts, sec_for, n)
        for sec in rows_order:
            stats = cl.get(sec, {})
            total = stats.get("n_total", 0)
            high_mat.loc[sec, n] = stats.get("pct_at_high", 0.0) if total else np.nan
            low_mat.loc[sec, n] = stats.get("pct_at_low", 0.0) if total else np.nan

    cols = st.columns(2)
    with cols[0]:
        fig = make_density_heatmap_chart(high_mat,
                                          title="HIGH-side density · % AT/NEAR per (Section × Lookback)",
                                          height=260, max_pct=1.0)
        st.plotly_chart(fig, use_container_width=True, theme=None)
    with cols[1]:
        fig = make_density_heatmap_chart(low_mat,
                                          title="LOW-side density · % AT/NEAR per (Section × Lookback)",
                                          height=260, max_pct=1.0)
        st.plotly_chart(fig, use_container_width=True, theme=None)


# =============================================================================
# View 4 — Drill-down expander (per-contract H-L bands, multiple lookbacks)
# =============================================================================
def _render_drill_down(scope_id: str, contracts: list[str], asof_date: date,
                        lookbacks: list[int]) -> None:
    with st.expander("🔍 Drill into a contract", expanded=False):
        cc1, cc2 = st.columns([2, 1])
        with cc1:
            sym = st.selectbox("Contract", options=["—"] + list(contracts),
                                key=f"{scope_id}_drill_pick")
        with cc2:
            history_lb = st.selectbox("History",
                                       ["30d", "60d", "90d", "180d", "252d"],
                                       index=2, key=f"{scope_id}_drill_lb")
        if sym == "—" or not sym:
            return
        days = {"30d": 30, "60d": 60, "90d": 90, "180d": 180, "252d": 252}[history_lb]
        start = asof_date - timedelta(days=int(days * 1.5) + 14)
        history = get_contract_history(sym, start, asof_date)
        if history.empty:
            st.info(f"No history available for {sym}.")
            return
        history = history.tail(days + 14)
        # Stats
        last = history["close"].iloc[-1] if not history.empty else None
        atr = None
        try:
            prev_c = history["close"].shift(1)
            tr = pd.concat([
                history["high"] - history["low"],
                (history["high"] - prev_c).abs(),
                (history["low"] - prev_c).abs(),
            ], axis=1).max(axis=1)
            atr = float(tr.rolling(ATR_PERIOD,
                                   min_periods=max(2, ATR_PERIOD // 2)).mean().iloc[-2])
        except Exception:
            atr = None
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Last close", f"{last:.4f}" if last is not None else "—")
        col2.metric(f"ATR({ATR_PERIOD})", f"{atr:.4f}" if atr is not None else "—")
        # Compute live N-day high/low for the displayed history at user's lookbacks
        # against the as-of date
        roll_max = history["close"].iloc[:-1].rolling(min(60, len(history) - 1)).max().iloc[-1]
        roll_min = history["close"].iloc[:-1].rolling(min(60, len(history) - 1)).min().iloc[-1]
        col3.metric("60d high (excl today)",
                     f"{roll_max:.4f}" if roll_max is not None and not pd.isna(roll_max) else "—")
        col4.metric("60d low (excl today)",
                     f"{roll_min:.4f}" if roll_min is not None and not pd.isna(roll_min) else "—")

        # Layered H-L bands
        fig = make_proximity_drill_chart(history.reset_index() if "date" not in history.columns else history,
                                          lookbacks=lookbacks,
                                          title=f"{sym} · close + rolling H-L bands ({', '.join(str(n)+'d' for n in lookbacks[:3])})",
                                          height=360)
        st.plotly_chart(fig, use_container_width=True, theme=None)


# =============================================================================
# Side panel — 12 analytics blocks, all rendered, scrollable
# =============================================================================
def _block_universe_glance(kpis: dict, lookback: int) -> None:
    _section_header(
        f"Universe at-a-glance ({lookback}d)",
        tooltip=(
            "Counts of contracts whose distance to the chosen N-day extreme falls in each band.\n"
            "AT≤0.25·ATR · NEAR≤0.5 · APPROACHING≤1.0 · else FAR.\n"
            "avg PIR / median PIR = position-in-range across the universe (0=at low, 1=at high)."
        ),
    )
    dist_tt = _mt("Distance to extreme (ATR-normalised)")
    pir_tt  = _mt("Position-in-range (PIR)")
    rows = [
        ("# AT (HIGH)",  f"{kpis['n_at_high']}",  "red" if kpis['n_at_high'] > 0 else "", None,
            _mt("Distance to extreme (ATR-normalised)", "AT",
                 f"{kpis['n_at_high']} contract(s) within {AT_ATR:.2f}·ATR of the {lookback}d high")),
        ("# AT (LOW)",   f"{kpis['n_at_low']}",   "green" if kpis['n_at_low'] > 0 else "", None,
            _mt("Distance to extreme (ATR-normalised)", "AT",
                 f"{kpis['n_at_low']} contract(s) within {AT_ATR:.2f}·ATR of the {lookback}d low")),
        ("# NEAR (HIGH)", f"{kpis['n_near_high']}", "amber", None,
            _mt("Distance to extreme (ATR-normalised)", "NEAR",
                 f"{kpis['n_near_high']} contract(s) within {NEAR_ATR:.2f}·ATR of the {lookback}d high")),
        ("# NEAR (LOW)",  f"{kpis['n_near_low']}",  "amber", None,
            _mt("Distance to extreme (ATR-normalised)", "NEAR",
                 f"{kpis['n_near_low']} contract(s) within {NEAR_ATR:.2f}·ATR of the {lookback}d low")),
        ("# APPR (HIGH)", f"{kpis['n_appr_high']}", None, None,
            _mt("Distance to extreme (ATR-normalised)", "APPROACHING",
                 f"{kpis['n_appr_high']} contract(s) within {APPROACH_ATR:.2f}·ATR of the {lookback}d high")),
        ("# APPR (LOW)",  f"{kpis['n_appr_low']}",  None, None,
            _mt("Distance to extreme (ATR-normalised)", "APPROACHING",
                 f"{kpis['n_appr_low']} contract(s) within {APPROACH_ATR:.2f}·ATR of the {lookback}d low")),
        ("avg PIR",       f"{kpis['avg_pir']:.2f}" if kpis['avg_pir'] is not None else "—",
            "accent", "0=at low / 1=at high",
            _mt("Position-in-range (PIR)", _pir_label(kpis['avg_pir']),
                 f"avg PIR across universe = " +
                 (f"{kpis['avg_pir']:.3f}" if kpis['avg_pir'] is not None else "—"))),
        ("median PIR",    f"{kpis['median_pir']:.2f}" if kpis['median_pir'] is not None else "—",
            None, None,
            _mt("Position-in-range (PIR)", _pir_label(kpis['median_pir']),
                 f"median PIR across universe = " +
                 (f"{kpis['median_pir']:.3f}" if kpis['median_pir'] is not None else "—"))),
    ]
    _kv_block(rows)


def _block_extreme_density(panel: dict, contracts: list[str], front_end: int,
                            mid_end: int, lookback: int) -> None:
    _section_header(
        f"Extreme density · Section × Side ({lookback}d)",
        tooltip=(
            "% of contracts in each curve section flagged AT or NEAR a N-day extreme.\n"
            "Density ≥ 50% in a single (section, side) cell signals a regime move concentrated\n"
            "in that part of the curve (front-end rally, back-end sell, belly squeeze)."
        ),
    )
    fr_rng, mr_rng, br_rng = compute_section_split(len(contracts), front_end, mid_end)
    sec_for: dict[str, str] = {}
    for i, sym in enumerate(contracts):
        if i in fr_rng: sec_for[sym] = "Front"
        elif i in mr_rng: sec_for[sym] = "Mid"
        else: sec_for[sym] = "Back"
    cl = compute_cluster_signal(panel, contracts, sec_for, lookback)
    rows = []
    for sec in ("Front", "Mid", "Back"):
        s = cl.get(sec)
        if not s or s["n_total"] == 0:
            rows.append((sec, "—", None, None))
            continue
        ph = s["pct_at_high"] * 100
        pl = s["pct_at_low"] * 100
        cls = "red" if ph >= 40 else ("green" if pl >= 40 else None)
        # Determine dominant side and bucket
        sec_label = ("AT" if max(ph, pl) >= 50
                      else "NEAR" if max(ph, pl) >= 25
                      else "APPROACHING" if max(ph, pl) >= 10
                      else "FAR")
        sec_tt = _mt("Distance to extreme (ATR-normalised)", sec_label,
                      f"{sec} section: H={s['n_at_high']}/{s['n_total']} ({ph:.0f}% AT/NEAR HIGH) · "
                      f"L={s['n_at_low']}/{s['n_total']} ({pl:.0f}% AT/NEAR LOW)")
        rows.append((
            f"{sec}",
            f"H {ph:.0f}% · L {pl:.0f}%",
            cls,
            f"{s['n_at_high']}/{s['n_at_low']}/{s['n_total']}",
            sec_tt,
        ))
    _kv_block(rows)


def _block_pattern_catalog(panel: dict, contracts: list[str]) -> None:
    _section_header(
        "Confluence patterns · contract counts",
        tooltip=(
            "Counts of contracts in each multi-lookback pattern. See pattern legend below.\n"
            "PERSISTENT/ACCELERATING — trend confirmations\n"
            "DECELERATING/REVERTING — reversal / mean-reversion candidates\n"
            "FRESH — event-driven; DRIFTED — coiling pattern\n"
            "DIVERGENT/MIXED/STABLE — neutral or no clear shape"
        ),
    )
    counts: dict = {}
    for sym in contracts:
        rec = panel.get("per_contract", {}).get(sym) or {}
        p = rec.get("pattern", "MIXED")
        counts[p] = counts.get(p, 0) + 1
    order = ["PERSISTENT", "ACCELERATING", "DECELERATING", "FRESH",
             "DRIFTED", "REVERTING", "DIVERGENT", "STABLE", "MIXED"]
    rows = []
    for p in order:
        if p not in counts:
            continue
        cls_map = {
            "PERSISTENT": "red", "ACCELERATING": "red", "DECELERATING": "amber",
            "FRESH": "accent", "DRIFTED": "amber", "REVERTING": "green",
        }
        cls = cls_map.get(p)
        rows.append((p, str(counts[p]), cls, None,
                      _mt("Confluence pattern (multi-window)", p,
                           f"{counts[p]} contract(s) classified as {p}")))
    if not rows:
        st.caption("No patterns classified.")
    else:
        _kv_block(rows)


def _block_streak_velocity(panel: dict, contracts: list[str], lookback: int) -> None:
    _section_header(
        f"Streak & velocity ({lookback}d)",
        tooltip=(
            f"Streak = consecutive prior bars within 0.5·ATR of the rolling {lookback}-day extreme.\n"
            "Velocity = avg daily Δ(dist) in ATR units over the last 5 bars on the nearest side.\n"
            "Negative velocity = approaching the extreme; positive = backing off."
        ),
    )
    rows_streak = []
    rows_velocity = []
    for sym in contracts:
        rec = panel.get("per_contract", {}).get(sym) or {}
        m = rec.get("by_lookback", {}).get(lookback) or {}
        streak = m.get("streak_at_extreme") or 0
        vel = m.get("velocity_atr_per_day")
        if streak >= 2:
            rows_streak.append((sym, streak, m.get("nearest_extreme")))
        if vel is not None and vel < 0:
            rows_velocity.append((sym, vel, m.get("nearest_extreme")))
    rows_streak.sort(key=lambda x: -x[1])
    rows_velocity.sort(key=lambda x: x[1])    # most negative = fastest approaching

    n_streak3 = sum(1 for r in rows_streak if r[1] >= 3)
    rows = []
    rows.append(("# streak ≥ 3d", str(n_streak3), "red" if n_streak3 else None, None,
                  _mt("Streak at extreme", "persistent",
                       f"{n_streak3} contract(s) with ≥3 consecutive prior bars within "
                       f"{NEAR_ATR}·ATR of the {lookback}d extreme")))
    rows.append(("# fastest closing", str(len(rows_velocity)), "accent", None,
                  _mt("Velocity to extreme (ATR/day)", "drifting toward extreme",
                       f"{len(rows_velocity)} contract(s) with negative velocity (closing in)")))
    _kv_block(rows)
    if rows_streak[:5]:
        for sym, n_streak, side in rows_streak[:5]:
            tt = _mt("Streak at extreme", _streak_label(n_streak),
                      f"{sym}: {n_streak} consecutive prior bars within "
                      f"{NEAR_ATR}·ATR of the {lookback}d {side}")
            st.markdown(
                f"<div title='{_tooltip_attr(tt)}' "
                f"style='font-family:JetBrains Mono, monospace; font-size:0.7rem; "
                f"padding:2px 0; color:var(--text-body); cursor:help;'>"
                f"<span style='color:var(--accent);'>{sym}</span> · "
                f"streak <b style='color:var(--red);'>{n_streak}d</b> at {side}"
                f"</div>",
                unsafe_allow_html=True,
            )
    if rows_velocity[:5]:
        st.markdown(
            "<div style='font-size:0.65rem; color:var(--text-dim); margin:4px 0 2px 0;'>"
            "Fastest closing in:</div>",
            unsafe_allow_html=True,
        )
        for sym, vel, side in rows_velocity[:5]:
            tt = _mt("Velocity to extreme (ATR/day)", _velocity_label(vel),
                      f"{sym}: avg Δ(dist)/day = {vel:+.4f} ATR/day toward {side}")
            st.markdown(
                f"<div title='{_tooltip_attr(tt)}' "
                f"style='font-family:JetBrains Mono, monospace; font-size:0.7rem; "
                f"padding:2px 0; color:var(--text-body); cursor:help;'>"
                f"<span style='color:var(--accent);'>{sym}</span> · "
                f"<b style='color:var(--green);'>{vel:+.3f}</b> ATR/day → {side}"
                f"</div>",
                unsafe_allow_html=True,
            )


def _block_fresh_breaks(panel: dict, lookback: int) -> None:
    _section_header(
        f"Fresh breakouts / breakdowns ({lookback}d)",
        tooltip=(
            "Today's close is strictly above the prior N-day high (FRESH HIGH) or below\n"
            "the prior N-day low (FRESH LOW). The bp-through column is signed: positive\n"
            "= bp above prior high; negative = bp below prior low. Catalog-aware bp scaling."
        ),
    )
    fbs = [b for b in panel.get("fresh_breaks_today", []) if b["lookback"] == lookback]
    if not fbs:
        st.caption("No fresh breaks at this lookback today.")
        return
    fbs.sort(key=lambda b: -abs(b.get("bp_through") or 0))
    for b in fbs[:10]:
        side = b["side"]
        bp = b.get("bp_through")
        color = "var(--green)" if side == "HIGH" else "var(--red)"
        side_label = "▲ NEW HIGH" if side == "HIGH" else "▼ NEW LOW"
        bp_str = (f" · {bp:+.2f}bp through"
                  if bp is not None and not pd.isna(bp) else "")
        st.markdown(
            f"<div style='font-family:JetBrains Mono, monospace; font-size:0.7rem; "
            f"padding:3px 0; color:var(--text-body); "
            f"border-bottom:1px solid var(--border-subtle);'>"
            f"<span style='color:{color}; font-weight:600;'>{side_label}</span> · "
            f"<span style='color:var(--accent);'>{b['symbol']}</span>{bp_str}"
            f"</div>",
            unsafe_allow_html=True,
        )


def _block_failed_extremes(panel: dict, lookback: int) -> None:
    _section_header(
        f"Failed extremes — reversal setups ({lookback}d)",
        tooltip=(
            "Within the last 3 prior bars, the close came within 0.25·ATR of the rolling\n"
            f"{lookback}-day extreme — but today is ≥0.5·ATR away. Touch-and-reverse pattern;\n"
            "classic mean-reversion entry zone."
        ),
    )
    fbs = [b for b in panel.get("failed_breaks", []) if b["lookback"] == lookback]
    if not fbs:
        st.caption("No failed extremes at this lookback.")
        return
    fbs.sort(key=lambda b: -(b.get("dist_atr_now") or 0))
    for b in fbs[:8]:
        side = b["side"]
        dist = b.get("dist_atr_now") or 0
        color = "var(--green)" if side == "HIGH" else "var(--red)"
        tt = _mt("Distance to extreme (ATR-normalised)", _flag_label_for_dist(dist),
                  f"{b['symbol']}: touched {side} extreme within last "
                  f"{FAILED_BREAK_LOOKBACK_BARS} bars; today is {dist:.2f} ATR away "
                  f"(≥ {NEAR_ATR}·ATR threshold required for failed break)")
        st.markdown(
            f"<div title='{_tooltip_attr(tt)}' "
            f"style='font-family:JetBrains Mono, monospace; font-size:0.7rem; "
            f"padding:3px 0; color:var(--text-body); cursor:help; "
            f"border-bottom:1px solid var(--border-subtle);'>"
            f"<span style='color:{color}; font-weight:600;'>"
            f"{'↘ FAILED HIGH' if side == 'HIGH' else '↗ FAILED LOW'}</span> · "
            f"<span style='color:var(--accent);'>{b['symbol']}</span> · now {dist:.2f} ATR away"
            f"</div>",
            unsafe_allow_html=True,
        )


def _block_cross_tenor_stretch(strategy: str, lookback: int, asof: date,
                                lookbacks_tuple: tuple) -> None:
    """For SPREADS / FLIES: median PIR per tenor across the curve. Tells the user
    which tenor is rich vs cheap, suggesting roll-trade ideas. (Skipped for outrights.)
    """
    if strategy == "outright":
        return
    _section_header(
        f"Cross-tenor stretch ({lookback}d)",
        tooltip=(
            "Median position-in-range across all live contracts of each tenor.\n"
            "PIR ≈ 1 → tenor is rich (near recent highs). PIR ≈ 0 → cheap.\n"
            "A tenor materially richer or cheaper than its neighbours suggests a roll trade:\n"
            "long the cheap tenor, short the rich tenor (same front-leg expiry where possible)."
        ),
    )
    tenors = get_available_tenors(strategy)
    rows = []
    for t in tenors:
        try:
            data = _build_proximity_data(strategy, t, asof.isoformat(),
                                              lookbacks_tuple, base_product=_BP)
            cs = data.get("contracts") or []
            if not cs:
                continue
            pirs = []
            for sym in cs:
                rec = data.get("proximity", {}).get("per_contract", {}).get(sym) or {}
                m = rec.get("by_lookback", {}).get(lookback) or {}
                pir = m.get("position_in_range")
                if pir is not None and not pd.isna(pir):
                    pirs.append(pir)
            if not pirs:
                continue
            med = float(np.median(pirs))
            n_at = sum(1 for sym in cs
                       if (data.get("proximity", {}).get("per_contract", {}).get(sym, {})
                                .get("by_lookback", {}).get(lookback, {}) or {})
                                .get("flag_high") == "AT"
                          or (data.get("proximity", {}).get("per_contract", {}).get(sym, {})
                                .get("by_lookback", {}).get(lookback, {}) or {})
                                .get("flag_low") == "AT")
            cls = "red" if med >= 0.75 else ("green" if med <= 0.25 else None)
            rows.append((f"{t}M tenor",
                          f"{med:.2f}",
                          cls,
                          f"{n_at}/{len(cs)} AT",
                          _mt("Position-in-range (PIR)", _pir_label(med),
                               f"{t}M tenor: median PIR = {med:.3f}  "
                               f"({n_at}/{len(cs)} contracts AT extreme)")))
        except Exception:
            continue
    if rows:
        _kv_block(rows)
    else:
        st.caption("No tenor data.")


def _block_section_regime(panel: dict, contracts: list[str], front_end: int,
                           mid_end: int, lookback: int) -> None:
    _section_header(
        f"Section regime via proximity ({lookback}d)",
        tooltip=(
            "Curve-level regime classification computed non-parametrically from average\n"
            "position-in-range per section.\n"
            "  BULL: all sections ≥0.70 · BEAR: all ≤0.30\n"
            "  STEEPENING: back ≥ front + 0.30 · FLATTENING: front ≥ back + 0.30\n"
            "  BELLY-BID: mid > avg(front,back) + 0.25 · BELLY-OFFERED: mid < avg − 0.25\n"
            "Confirms / contradicts the regression-based regime engine in the Curve subtab."
        ),
    )
    sr = proximity_section_regime(panel, contracts, front_end, mid_end, lookback)
    label = sr.get("label", "—")
    color = (
        "green" if "BULL" in label else
        "red" if "BEAR" in label else
        "accent"
    )
    # Extract bare label (BULL/BEAR/STEEPENING/...) from formatted string for guide lookup
    bare = next((tag for tag in ("BULL", "BEAR", "STEEPENING", "FLATTENING",
                                  "BELLY-BID", "BELLY-OFFERED", "MIXED")
                  if tag in label), None)
    rows = [(f"Curve label", label, color, None,
              _mt("Section regime (proximity)", bare,
                   f"computed at {lookback}d focus lookback"))]
    for sec in ("front", "mid", "back"):
        s = sr.get("sections", {}).get(sec, {})
        avg_pir = s.get("avg_pir")
        bias = s.get("bias", "—")
        cls = "red" if bias == "HIGH" else ("green" if bias == "LOW" else None)
        sec_tt = _mt("Position-in-range (PIR)", _pir_label(avg_pir),
                      f"{sec.title()} section: avg PIR = "
                      f"{(avg_pir is not None and f'{avg_pir:.3f}') or '—'}, bias = {bias}")
        rows.append((sec.title(),
                      f"PIR {avg_pir:.2f} · {bias}" if avg_pir is not None else "—",
                      cls, None, sec_tt))
    _kv_block(rows)


def _block_z_cross_check(prox_panel: dict, z_panel: dict, contracts: list[str],
                          lookback: int) -> None:
    _section_header(
        f"Z-score cross-check ({lookback}d)",
        tooltip=(
            "Pairs each AT/NEAR contract's proximity flag with its Z-score on the same lookback.\n"
            "  UNTESTED-EXTREME    AT/NEAR but |Z|<1     — uncrowded breakout candidate\n"
            "  STRETCHED-EXTREME   AT/NEAR and |Z|≥2     — exhausted, fade with confluence\n"
            "  COILED              FAR and |Z|≥1.5      — rich/cheap but not at recent extreme\n"
            "  NORMAL              FAR and |Z|<1         — quiet"
        ),
    )
    untested, stretched, coiled = [], [], []
    for sym in contracts:
        prec = prox_panel.get("per_contract", {}).get(sym) or {}
        pm = prec.get("by_lookback", {}).get(lookback) or {}
        zrec = z_panel.get("per_contract", {}).get(sym) or {}
        zm = zrec.get("by_lookback", {}).get(lookback) or {}
        flag_h = pm.get("flag_high"); flag_l = pm.get("flag_low")
        z = zm.get("z")
        if z is None or pd.isna(z):
            continue
        is_extreme = flag_h in ("AT", "NEAR") or flag_l in ("AT", "NEAR")
        if is_extreme and abs(z) < 1.0:
            untested.append((sym, z, flag_h if flag_h in ("AT", "NEAR") else flag_l))
        elif is_extreme and abs(z) >= 2.0:
            stretched.append((sym, z, flag_h if flag_h in ("AT", "NEAR") else flag_l))
        elif (not is_extreme) and abs(z) >= 1.5:
            coiled.append((sym, z))
    rows = [
        ("# UNTESTED-EXTREME", str(len(untested)), "accent", "AT/NEAR · |z|<1",
            _mt("Z-score cross-check", "UNTESTED-EXTREME",
                 f"{len(untested)} contract(s) — at extreme but z hasn't caught up; "
                 "uncrowded breakout candidate")),
        ("# STRETCHED-EXTREME", str(len(stretched)), "red", "AT/NEAR · |z|≥2",
            _mt("Z-score cross-check", "STRETCHED-EXTREME",
                 f"{len(stretched)} contract(s) — at extreme AND |z|≥2; "
                 "exhausted, fade-with-confluence")),
        ("# COILED", str(len(coiled)), "amber", "FAR · |z|≥1.5",
            _mt("Z-score cross-check", "COILED",
                 f"{len(coiled)} contract(s) — rich/cheap on z but FAR from extreme; "
                 "mean-reversion in process")),
    ]
    _kv_block(rows)
    if untested:
        for sym, z, fl in untested[:4]:
            tt = _mt("Z-score cross-check", "UNTESTED-EXTREME",
                      f"{sym}: proximity flag = {fl}, z = {z:+.2f}σ")
            st.markdown(
                f"<div title='{_tooltip_attr(tt)}' "
                f"style='font-family:JetBrains Mono, monospace; font-size:0.69rem; "
                f"padding:2px 0; color:var(--text-body); cursor:help;'>"
                f"<span style='color:var(--accent);'>{sym}</span> · "
                f"<span style='color:var(--blue);'>UNTESTED</span> · "
                f"flag {fl} · z {z:+.2f}σ"
                f"</div>",
                unsafe_allow_html=True,
            )
    if stretched:
        for sym, z, fl in stretched[:4]:
            tt = _mt("Z-score cross-check", "STRETCHED-EXTREME",
                      f"{sym}: proximity flag = {fl}, z = {z:+.2f}σ")
            st.markdown(
                f"<div title='{_tooltip_attr(tt)}' "
                f"style='font-family:JetBrains Mono, monospace; font-size:0.69rem; "
                f"padding:2px 0; color:var(--text-body); cursor:help;'>"
                f"<span style='color:var(--accent);'>{sym}</span> · "
                f"<span style='color:var(--red);'>STRETCHED</span> · "
                f"flag {fl} · z {z:+.2f}σ"
                f"</div>",
                unsafe_allow_html=True,
            )


def _block_touch_and_range(panel: dict, contracts: list[str], lookback: int) -> None:
    _section_header(
        f"Touch frequency & range expansion ({lookback}d)",
        tooltip=(
            f"Touch count = # bars in the last {lookback} that came within 0.25·ATR of the rolling\n"
            "extreme on the nearest side. ≥3 = retesting; coiled accumulation.\n"
            "Range expansion = current N-day range / prior N-day range. >1.3 = volatility regime up."
        ),
    )
    high_touch = []
    low_touch = []
    expanding = []
    contracting = []
    for sym in contracts:
        rec = panel.get("per_contract", {}).get(sym) or {}
        m = rec.get("by_lookback", {}).get(lookback) or {}
        th = m.get("touch_count_high") or 0
        tl = m.get("touch_count_low") or 0
        rer = m.get("range_expansion_ratio")
        if th >= 3:
            high_touch.append((sym, th))
        if tl >= 3:
            low_touch.append((sym, tl))
        if rer is not None and not pd.isna(rer):
            if rer >= 1.3:
                expanding.append((sym, rer))
            elif rer <= 0.75:
                contracting.append((sym, rer))
    rows = [
        ("# retest HIGH (≥3)", str(len(high_touch)), "red" if high_touch else None, None,
            _mt("Touch count (re-tests)", "coiling",
                 f"{len(high_touch)} contract(s) with ≥3 prior bars within "
                 f"{AT_ATR}·ATR of the {lookback}d high")),
        ("# retest LOW (≥3)",  str(len(low_touch)),  "green" if low_touch else None, None,
            _mt("Touch count (re-tests)", "coiling",
                 f"{len(low_touch)} contract(s) with ≥3 prior bars within "
                 f"{AT_ATR}·ATR of the {lookback}d low")),
        ("# range expanding",  str(len(expanding)),  "amber", "ratio≥1.3",
            _mt("Range expansion ratio", "expanding",
                 f"{len(expanding)} contract(s) with current {lookback}d range / prior {lookback}d range ≥ 1.3")),
        ("# range contracting", str(len(contracting)), "accent", "ratio≤0.75",
            _mt("Range expansion ratio", "contracting",
                 f"{len(contracting)} contract(s) with current {lookback}d range / prior {lookback}d range ≤ 0.75")),
    ]
    _kv_block(rows)


def _block_concurrent_clusters(panel: dict, contracts: list[str], lookback: int,
                                front_end: int, mid_end: int) -> None:
    _section_header(
        f"Concurrent extremes — cluster signal ({lookback}d)",
        tooltip=(
            "Counts how many of today's AT/NEAR contracts share the same Section. The biggest\n"
            "cluster tells you where the regime move is concentrated — single-name vs market-wide."
        ),
    )
    fr_rng, mr_rng, br_rng = compute_section_split(len(contracts), front_end, mid_end)
    sec_for: dict[str, str] = {}
    for i, sym in enumerate(contracts):
        if i in fr_rng: sec_for[sym] = "Front"
        elif i in mr_rng: sec_for[sym] = "Mid"
        else: sec_for[sym] = "Back"
    cl = compute_cluster_signal(panel, contracts, sec_for, lookback)
    rows = []
    for sec in ("Front", "Mid", "Back"):
        s = cl.get(sec)
        if not s or s["n_total"] == 0:
            continue
        h = s["n_at_high"]; l = s["n_at_low"]; n = s["n_total"]
        if h + l == 0:
            continue
        side = "HIGH" if h > l else ("LOW" if l > h else "MIXED")
        cls = "red" if side == "HIGH" else ("green" if side == "LOW" else "amber")
        rows.append((sec, f"{h+l}/{n} {side}", cls, f"{(h+l)/n*100:.0f}% cluster"))
    if rows:
        _kv_block(rows)
    else:
        st.caption("No clusters at this lookback.")


def _block_methodology() -> None:
    text = get_proximity_thresholds_text()
    attr = _tooltip_attr(text)
    st.markdown(
        f"<div title=\"{attr}\" style='cursor:help; padding:6px 8px; "
        f"background:var(--bg-elevated); border:1px dashed var(--border-default); "
        f"border-radius:6px; margin: 0.6rem 0 0.3rem 0;'>"
        f"<div style='font-size:0.7rem; color:var(--text-dim); "
        f"text-transform:uppercase; letter-spacing:0.06em; font-weight:600;'>"
        f"Methodology / thresholds <span style='color:var(--accent-dim);'>ⓘ</span>"
        f"</div>"
        f"<div style='font-size:0.7rem; color:var(--text-muted); margin-top:2px;'>"
        f"AT≤{AT_ATR:.2f}·ATR · NEAR≤{NEAR_ATR:.2f} · APPROACHING≤{APPROACH_ATR:.2f} · ATR period {ATR_PERIOD}d · "
        f"hover for full text</div></div>",
        unsafe_allow_html=True,
    )


def _block_pattern_legend() -> None:
    """Expandable legend explaining each confluence pattern + trader implication."""
    with st.expander("Pattern legend (click for trader implications)", expanded=False):
        descs = get_pattern_descriptions()
        order = ["PERSISTENT", "ACCELERATING", "DECELERATING", "FRESH",
                 "DRIFTED", "REVERTING", "DIVERGENT", "STABLE", "MIXED"]
        for p in order:
            d = descs.get(p, "")
            st.markdown(
                f"<div style='padding:4px 0; border-bottom:1px solid var(--border-subtle);'>"
                f"<div style='display:flex; align-items:center; gap:0.4rem;'>"
                f"{_pattern_pill(p)}"
                f"<span style='color:var(--text-body); font-size:0.78rem;'>{d}</span>"
                f"</div></div>",
                unsafe_allow_html=True,
            )


def _render_combined_side_panel(data: dict, contracts: list[str], lookback: int,
                                  front_end: int, mid_end: int, kpis: dict,
                                  strategy: str, asof: date,
                                  lookbacks_tuple: tuple) -> None:
    panel = data.get("proximity", {})
    z_panel = data.get("zscore", {})
    st.markdown(
        f"<div style='display:flex; align-items:center; gap:0.5rem; "
        f"padding-bottom:0.4rem; border-bottom:1px solid var(--border-subtle); margin-bottom:0.4rem;'>"
        f"<span style='color:var(--accent); font-size:0.85rem; font-weight:600;'>📊 Analysis · all blocks (scroll)</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    _block_methodology()
    _block_universe_glance(kpis, lookback)
    _block_extreme_density(panel, contracts, front_end, mid_end, lookback)
    _block_pattern_catalog(panel, contracts)
    _block_streak_velocity(panel, contracts, lookback)
    _block_fresh_breaks(panel, lookback)
    _block_failed_extremes(panel, lookback)
    _block_cross_tenor_stretch(strategy, lookback, asof, lookbacks_tuple)
    _block_section_regime(panel, contracts, front_end, mid_end, lookback)
    _block_z_cross_check(panel, z_panel, contracts, lookback)
    _block_touch_and_range(panel, contracts, lookback)
    _block_concurrent_clusters(panel, contracts, lookback, front_end, mid_end)
    _block_pattern_legend()


# =============================================================================
# Top-level render
# =============================================================================
def render(base_product: str = "SRA") -> None:
    _set_market(base_product)
    snap_date = get_sra_snapshot_latest_date()
    if snap_date is None:
        st.error("OHLC database / SRA snapshot unavailable.")
        return

    # Status strip up top
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

    # Interpretation guide — full reference for every proximity metric and bucket
    _render_interpretation_guide_proximity()

    # Section boundaries — own to this subtab so user can tune for proximity
    with st.expander("Curve section boundaries (Front / Mid / Back)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            front_end = st.number_input(
                "Front-end (last index)", min_value=2, max_value=60,
                value=int(st.session_state.get(f"{_SCOPE_PREFIX}_fe", DEFAULT_FRONT_END)),
                step=1, key=f"{_SCOPE_PREFIX}_fe",
                help="First N contracts are FRONT.",
            )
        with c2:
            mid_end = st.number_input(
                "Mid-end (last index)", min_value=front_end + 1, max_value=120,
                value=int(st.session_state.get(f"{_SCOPE_PREFIX}_me", DEFAULT_MID_END)),
                step=1, key=f"{_SCOPE_PREFIX}_me",
                help="Front+Mid spans first M contracts; remainder is BACK.",
            )

    # Top controls row
    cc1, cc2, cc3, cc4 = st.columns([1.0, 1.4, 2.4, 1.4])
    with cc1:
        strategy_label = st.segmented_control(
            "Strategy", options=["Outright", "Spread", "Fly"],
            default="Outright", key=f"{_SCOPE_PREFIX}_strat",
            help="Pick a strategy bucket.")
    with cc2:
        if strategy_label == "Outright":
            tenor_months = None
            st.markdown(
                "<div style='padding-top:1.7rem; color:var(--text-dim); font-size:0.75rem;'>"
                "All quarterly + serial outrights</div>", unsafe_allow_html=True)
        elif strategy_label == "Spread":
            tenor_months = st.selectbox(
                "Tenor", options=spread_tenors, index=min(1, max(0, len(spread_tenors) - 1)),
                key=f"{_SCOPE_PREFIX}_tenor_spread",
                format_func=lambda t: f"{t}M",
                help="Spread tenor (calendar spread length).",
            )
        else:
            tenor_months = st.selectbox(
                "Tenor", options=fly_tenors,
                index=0, key=f"{_SCOPE_PREFIX}_tenor_fly",
                format_func=lambda t: f"{t}M",
                help="Fly tenor (legs spaced by N months).",
            )
    with cc3:
        lookbacks_picked = st.multiselect(
            "Lookbacks (multi-select)",
            options=_LOOKBACK_OPTIONS, default=_LOOKBACK_OPTIONS,
            key=f"{_SCOPE_PREFIX}_lbs",
            help="Lookback windows used to compute high/low extremes. "
                 "All blocks operate on the SELECTED set; the Confluence Matrix uses all.",
        )
        if not lookbacks_picked:
            lookbacks_picked = _LOOKBACK_OPTIONS
    with cc4:
        focus_lookback = st.selectbox(
            "Focus lookback", options=lookbacks_picked,
            index=lookbacks_picked.index(30) if 30 in lookbacks_picked else 0,
            key=f"{_SCOPE_PREFIX}_focus",
            help="The Section Ribbons, Cluster Heatmap, and side-panel blocks "
                 "all use this lookback as the focal window.",
        )

    # View toggles + side-panel
    vc1, vc2, vc3, vc4, vc5 = st.columns([1, 1, 1, 1, 1.4])
    with vc1:
        top_k = st.selectbox("Top K", options=[3, 5, 10], index=1,
                              key=f"{_SCOPE_PREFIX}_topk")
    with vc2:
        show_ribbons = st.checkbox("Section ribbons", value=True,
                                    key=f"{_SCOPE_PREFIX}_show_ribbons")
    with vc3:
        show_confluence = st.checkbox("Confluence matrix", value=True,
                                       key=f"{_SCOPE_PREFIX}_show_conf")
    with vc4:
        show_cluster = st.checkbox("Cluster heatmap", value=True,
                                    key=f"{_SCOPE_PREFIX}_show_cluster")
    with vc5:
        analysis_on = st.toggle("Analysis side panel", value=True,
                                 key=f"{_SCOPE_PREFIX}_analysis_on")

    # Build engine
    strategy_db = strategy_label.lower()
    lookbacks_tuple = tuple(sorted(set(lookbacks_picked)))
    with st.spinner(f"Computing proximity for {strategy_label.lower()}s..."):
        try:
            data = _build_proximity_data(
                strategy_db, tenor_months, snap_date.isoformat(),
                lookbacks_tuple, base_product=_BP)
        except Exception as e:
            st.error(f"Failed to build proximity panel: {e}")
            with st.expander("Traceback"):
                import traceback as _tb
                st.code(_tb.format_exc())
            return

    contracts = data.get("contracts") or []
    if not contracts:
        st.warning(f"No live contracts for SRA {strategy_label} "
                    f"{f'{tenor_months}M' if tenor_months else ''}.")
        return

    panel = data.get("proximity") or {}
    if not panel:
        st.warning("Proximity panel could not be built (data may be sparse).")
        return

    # Per-scope info strip
    scope_label = (f"SRA · {strategy_label} · {tenor_months}M"
                    if tenor_months else f"SRA · {strategy_label}")
    range_str = data.get("info", {}).get("range_str", "—")
    status_strip_with_dot([
        ("Scope", scope_label, "accent"),
        ("Contracts", f"{len(contracts)}", None),
        ("Range", range_str, None),
        ("Lookbacks", " · ".join(f"{n}d" for n in lookbacks_tuple), None),
        ("Focus", f"{focus_lookback}d", "accent"),
    ], dot_label="Live")

    kpis = _proximity_kpis(panel, contracts, focus_lookback)

    # ──────────────────────────────────────────────────────────────
    # Main layout: chart column + scrollable side panel
    # ──────────────────────────────────────────────────────────────
    chart_col, side_col = st.columns([3, 1.1])
    with chart_col:
        if show_ribbons:
            st.markdown(
                f"<div style='margin-top:0.4rem; font-size:0.85rem; font-weight:600; "
                f"color:var(--text-heading);'>Section ribbons · top-{top_k} closest to {focus_lookback}d extremes</div>",
                unsafe_allow_html=True,
            )
            try:
                _render_section_ribbons(panel, contracts,
                                         int(front_end), int(mid_end),
                                         focus_lookback, int(top_k))
            except Exception as e:
                st.error(f"Section ribbons failed: {e}")

        if show_confluence:
            st.markdown(
                "<div style='margin-top:0.8rem; font-size:0.85rem; font-weight:600; "
                "color:var(--text-heading);'>Confluence matrix · multi-window proximity</div>",
                unsafe_allow_html=True,
            )
            try:
                _render_confluence_matrix(panel, contracts, list(lookbacks_tuple),
                                            top_n=25)
            except Exception as e:
                st.error(f"Confluence matrix failed: {e}")

        if show_cluster:
            st.markdown(
                "<div style='margin-top:0.8rem; font-size:0.85rem; font-weight:600; "
                "color:var(--text-heading);'>Cluster heatmap · density per (Section × Lookback)</div>",
                unsafe_allow_html=True,
            )
            try:
                _render_cluster_heatmap(panel, contracts,
                                          int(front_end), int(mid_end),
                                          list(lookbacks_tuple))
            except Exception as e:
                st.error(f"Cluster heatmap failed: {e}")

        try:
            _render_drill_down(_SCOPE_PREFIX, contracts, snap_date,
                                list(lookbacks_tuple))
        except Exception as e:
            st.error(f"Drill-down failed: {e}")

    with side_col:
        if analysis_on:
            # Phase H: visible side-panel border for parity across subtabs
            with st.container(height=720, border=True):
                try:
                    _render_combined_side_panel(
                        data, contracts, focus_lookback,
                        int(front_end), int(mid_end), kpis,
                        strategy_db, snap_date, lookbacks_tuple,
                    )
                except Exception as e:
                    st.error(f"Side panel failed: {e}")
                    with st.expander("Traceback"):
                        import traceback as _tb
                        st.code(_tb.format_exc())
