"""SRA — Curve subtab. Modes: Standard / Multi-date / Ribbon / Pack / Volume Δ /
Heatmap / Matrix. Plus regime badge, FOMC overlay, carry coloring, click-to-drill.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from lib.carry import (
    compute_carry_table,
    compute_per_contract_carry,
    compute_white_pack_carry,
    implied_rate,
)
from lib.charts import (
    make_animated_curve_chart,
    make_calendar_matrix_chart,
    make_carry_colored_curve,
    make_change_bar_chart,
    make_curve_chart,
    make_heatmap_chart,
    make_multi_date_curve_chart,
    make_pack_chart,
    make_ribbon_chart,
    make_volume_delta_chart,
    make_zscore_curve_chart,
)
from lib.components import status_strip, status_strip_with_dot
from lib.fomc import (
    annotate_contracts_with_fomcs,
    decompose_implied_rates,
    get_fomc_dates_in_range,
    get_methodology_text,
    is_quarterly,
    next_fomc_date,
    reference_period,
)
from lib.regime import classify_regime_multi_lookback, get_regime_thresholds_text
from lib.sra_data import (
    DEFAULT_FRONT_END,
    DEFAULT_MID_END,
    LIVENESS_DAYS,
    PACK_COLORS,
    compute_curve_change,
    compute_decomposition,
    compute_pack_groups as _sra_compute_pack_groups,
    compute_pairwise_spread_matrix,
    compute_per_contract_zscores,
    compute_percentile_bands,
    compute_percentile_rank,
    compute_section_split,
    contract_range_str,
    get_contract_history,
    get_flies as _sra_get_flies,
    get_outrights as _sra_get_outrights,
    get_reference_rates_at as _sra_get_refrates_at,
    get_spreads as _sra_get_spreads,
    get_sra_snapshot_latest_date as _sra_snapshot_latest_date,
    load_reference_rate_panel as _sra_load_ref_rate,
    load_sra_curve_panel as _sra_load_curve_panel,
    pivot_curve_panel,
    tenor_breakdown as _sra_tenor_breakdown,
)
from lib import market_data as _md
from lib.markets import get_market as _get_market


# ─── Active-market shim (set by render()) ───────────────────────────────────
_BP = "SRA"


def _set_market(base_product: str) -> None:
    global _BP
    _BP = base_product


def get_sra_snapshot_latest_date():
    """Returns latest snapshot for the active market (set via _set_market)."""
    return _sra_snapshot_latest_date() if _BP == "SRA" else _md.get_snapshot_latest_date(_BP)


def get_outrights():
    return _sra_get_outrights() if _BP == "SRA" else _md.get_outrights(_BP)


def get_spreads(t):
    return _sra_get_spreads(t) if _BP == "SRA" else _md.get_spreads(_BP, t)


def get_flies(t):
    return _sra_get_flies(t) if _BP == "SRA" else _md.get_flies(_BP, t)


def load_sra_curve_panel(strategy, tenor, start, end):
    if _BP == "SRA":
        return _sra_load_curve_panel(strategy, tenor, start, end)
    return _md.load_curve_panel(_BP, strategy, tenor, start, end)


def load_reference_rate_panel(start, end):
    if _BP == "SRA":
        return _sra_load_ref_rate(start, end)
    return _md.load_reference_rate_panel(_BP, start, end)


def get_reference_rates_at(asof, ref_panel):
    if _BP == "SRA":
        return _sra_get_refrates_at(asof, ref_panel)
    return _md.get_reference_rates_at(_BP, asof, ref_panel)


def compute_pack_groups(symbols_df):
    if _BP == "SRA":
        return _sra_compute_pack_groups(symbols_df)
    return _md.compute_pack_groups(symbols_df, _BP)


def tenor_breakdown(strategy):
    if _BP == "SRA":
        return _sra_tenor_breakdown(strategy)
    out = []
    for t in _md.get_available_tenors(_BP, strategy):
        df = _md.get_live_symbols(_BP, strategy, t)
        if df.empty:
            continue
        syms = df["symbol"].tolist()
        out.append((t, len(df), f"{syms[0]} → {syms[-1]}" if syms else "—"))
    return out
from lib.theme import (
    ACCENT, AMBER, BLUE, GREEN, RED, TEXT_BODY, TEXT_DIM, TEXT_HEADING, TEXT_MUTED,
)


MODE_OUTRIGHTS = ["Standard", "Multi-date", "Ribbon", "Pack", "Volume Δ", "Heatmap"]
MODE_SPREADS_FLIES = ["Standard", "Multi-date", "Ribbon", "Volume Δ", "Heatmap", "Matrix"]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _quick_compare_offset(available_dates: list, asof_date, days_back: int):
    target = asof_date - timedelta(days=days_back)
    earlier = [d for d in available_dates if d <= target]
    return max(earlier) if earlier else available_dates[0]


def _z_color(z):
    if z is None or pd.isna(z):
        return TEXT_DIM
    az = abs(z)
    if az >= 2.0:
        return RED
    if az >= 1.0:
        return AMBER
    return TEXT_BODY


def _delta_color(d):
    if d is None or pd.isna(d):
        return TEXT_DIM
    return GREEN if d > 0 else (RED if d < 0 else TEXT_BODY)


def _delta_class(v):
    if v is None or pd.isna(v):
        return ""
    return "green" if v > 0 else ("red" if v < 0 else "")


def _kv_block(rows: list) -> None:
    parts = []
    for tup in rows:
        label = tup[0]
        value = tup[1]
        cls = tup[2] if len(tup) > 2 and tup[2] else ""
        sub = tup[3] if len(tup) > 3 and tup[3] else ""
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
        parts.append(
            f"<div style='display:flex; justify-content:space-between; align-items:center; "
            f"padding:4px 0; border-bottom:1px solid var(--border-subtle);'>"
            f"<span style='color:var(--text-muted); font-size:0.78rem;'>{label}</span>"
            f"<span style='color:{color_var}; font-family:JetBrains Mono, monospace; "
            f"font-size:0.78rem; font-weight:500; text-align:right;'>{value}{sub_html}</span>"
            f"</div>"
        )
    st.markdown("".join(parts), unsafe_allow_html=True)


def _side_section_header(text: str) -> None:
    st.markdown(
        f"<div style='font-size:0.7rem; color:var(--text-dim); text-transform:uppercase; "
        f"letter-spacing:0.06em; margin: 0.6rem 0 0.3rem 0; font-weight:600;'>{text}</div>",
        unsafe_allow_html=True,
    )


def _section_slope(rng_, contracts, current_y):
    section_vals = [(contracts[i], current_y[i]) for i in rng_
                    if i < len(current_y) and current_y[i] is not None
                    and not pd.isna(current_y[i])]
    if len(section_vals) < 2:
        return None
    return section_vals[-1][1] - section_vals[0][1]


def _side_curve_summary(asof_date, contracts, current_y, front_end, mid_end) -> None:
    _side_section_header("Curve summary")
    valid = [(c, v) for c, v in zip(contracts, current_y) if v is not None and not pd.isna(v)]
    if len(valid) < 2:
        st.caption("Not enough valid points.")
        return
    front_c, front_v = valid[0]
    back_c, back_v = valid[-1]
    slope = back_v - front_v
    high_c, high_v = max(valid, key=lambda x: x[1])
    low_c, low_v = min(valid, key=lambda x: x[1])
    rng = high_v - low_v
    fr, mr, br = compute_section_split(len(contracts), front_end, mid_end)
    fr_slope = _section_slope(fr, contracts, current_y)
    mr_slope = _section_slope(mr, contracts, current_y)
    br_slope = _section_slope(br, contracts, current_y)

    rows = [
        ("Front", f"{front_c}", "accent", f"{front_v:.4f}"),
        ("Back",  f"{back_c}",  "accent", f"{back_v:.4f}"),
        ("Slope (back−front)", f"{slope:+.4f}",
         "green" if slope > 0 else ("red" if slope < 0 else ""), None),
        ("Range", f"{rng:.4f}", "accent", f"hi {high_c} · lo {low_c}"),
    ]
    _kv_block(rows)

    _side_section_header("Section slopes")
    sec_rows = [
        ("Front", f"{fr_slope:+.4f}" if fr_slope is not None else "—", _delta_class(fr_slope), None),
        ("Mid",   f"{mr_slope:+.4f}" if mr_slope is not None else "—", _delta_class(mr_slope), None),
        ("Back",  f"{br_slope:+.4f}" if br_slope is not None else "—", _delta_class(br_slope), None),
    ]
    _kv_block(sec_rows)


# -----------------------------------------------------------------------------
# Regime badge (always-on, top of each scope)
# -----------------------------------------------------------------------------
def _render_regime_badge(wide_close, asof_date, contracts, front_end, mid_end,
                          lookbacks=(1, 5, 30)) -> None:
    try:
        multi = classify_regime_multi_lookback(wide_close, asof_date, contracts,
                                                front_end, mid_end, lookbacks=lookbacks)
    except Exception:
        return
    if not multi:
        return

    # Build tooltip — use HTML entity &#10; for newlines so it survives HTML parsing
    tooltip_text = get_regime_thresholds_text()
    tooltip_attr = (tooltip_text
                    .replace('"', '&quot;')
                    .replace("'", "&#39;")
                    .replace("\n", "&#10;"))

    parts = []
    for label_key, regime in multi.items():
        lbl = regime["label"]
        dirc = regime.get("direction", "—")
        accent = "var(--green)" if dirc == "BULL" else (
                 "var(--red)" if dirc == "BEAR" else "var(--text-muted)")
        prior_dt = regime.get("prior_date")
        prior_str = prior_dt.strftime("%Y-%m-%d") if prior_dt else "—"
        full_title = f"Lookback: {label_key} (vs {prior_str})&#10;&#10;{tooltip_attr}"
        parts.append(
            f"<div title=\"{full_title}\" "
            f"style='padding: 0.25rem 0.65rem; border:1px solid var(--border-subtle); "
            f"border-radius:6px; background-color:var(--bg-surface); display:inline-flex; "
            f"flex-direction:column; gap:1px; min-width:140px; cursor:help;'>"
            f"<span style='font-size:0.62rem; color:var(--text-dim); "
            f"text-transform:uppercase; letter-spacing:0.06em;'>{label_key} regime ⓘ</span>"
            f"<span style='font-size:0.78rem; font-family:JetBrains Mono, monospace; "
            f"font-weight:600; color:{accent};'>{lbl}</span>"
            f"<span style='font-size:0.62rem; color:var(--text-dim); "
            f"font-family:JetBrains Mono, monospace;'>"
            f"par {regime.get('parallel_bp', 0) or 0:+.2f} · "
            f"slope {regime.get('slope_bp', 0) or 0:+.2f} · "
            f"curv {regime.get('curvature_bp', 0) or 0:+.2f} bp</span>"
            f"</div>"
        )
    st.markdown(
        f"<div style='display:flex; gap:0.5rem; flex-wrap:wrap; margin: 0.25rem 0 0.5rem 0;'>{''.join(parts)}</div>",
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Click-to-drill panel
# -----------------------------------------------------------------------------
def _render_drill_down(scope_id: str, contracts: list, asof_date: date,
                        wide_close: pd.DataFrame, panel_history_days: int = 252) -> None:
    """Per-contract drill-down with full daily history, stats, neighbors."""
    with st.expander("🔍 Drill into a contract", expanded=False):
        cc1, cc2 = st.columns([2, 1])
        with cc1:
            sym = st.selectbox("Contract", options=["—"] + list(contracts),
                                key=f"{scope_id}_drill_pick")
        with cc2:
            lookback = st.selectbox("Lookback",
                                     ["30d", "60d", "90d", "252d", "Max"],
                                     index=2, key=f"{scope_id}_drill_lb")
        if sym == "—" or not sym:
            return

        lookback_days = {"30d": 30, "60d": 60, "90d": 90, "252d": 252,
                         "Max": panel_history_days}[lookback]
        start = asof_date - timedelta(days=lookback_days * 2)
        history = get_contract_history(sym, start, asof_date)
        if history.empty:
            st.info(f"No history available for {sym}.")
            return
        history = history.tail(lookback_days)

        # Stats
        last_close = history["close"].iloc[-1] if not history.empty else None
        rp = reference_period(sym)
        d2e = ((rp[0] - asof_date).days if rp else None)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Last close", f"{last_close:.4f}" if last_close is not None else "—")
        col2.metric("Implied rate",
                     f"{implied_rate(last_close):.3f}%" if last_close is not None else "—")
        col3.metric("Days to ref", f"{d2e}" if d2e is not None else "—")
        col4.metric("Bars", str(len(history)))

        # Stats table
        stats = history["close"].describe()
        try:
            today_z = (history["close"].iloc[-1] - history["close"].iloc[:-1].mean()) / max(
                1e-9, history["close"].iloc[:-1].std(ddof=1))
            pct_rank = (history["close"].iloc[:-1] < history["close"].iloc[-1]).sum() \
                       / max(1, len(history) - 1) * 100
        except Exception:
            today_z = None
            pct_rank = None
        st1, st2, st3, st4, st5 = st.columns(5)
        st1.metric("Mean", f"{stats['mean']:.4f}")
        st2.metric("Std", f"{stats['std']:.4f}")
        st3.metric("Min", f"{stats['min']:.4f}")
        st4.metric("Max", f"{stats['max']:.4f}")
        st5.metric("Z (vs prior)", f"{today_z:+.2f}σ" if today_z is not None else "—",
                    delta=f"p{pct_rank:.0f}" if pct_rank is not None else None)

        # OHLC time-series chart
        import plotly.graph_objects as go
        from lib.theme import BG_BASE
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history["date"], y=history["close"],
            mode="lines", line=dict(color=ACCENT, width=2), name="close",
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>%{y:.4f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=history["date"], y=history["high"],
            mode="lines", line=dict(color="rgba(232,183,93,0)", width=0),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=history["date"], y=history["low"],
            mode="lines", line=dict(color="rgba(232,183,93,0)", width=0),
            fill="tonexty", fillcolor="rgba(232,183,93,0.18)",
            name="H–L band", hovertemplate="<b>%{x|%Y-%m-%d}</b><br>L: %{y:.4f}<extra></extra>",
        ))
        fig.update_layout(
            xaxis=dict(title=None),
            yaxis=dict(title="Price", tickformat=".4f"),
            height=320,
            hovermode="x unified",
            showlegend=False,
            margin=dict(l=55, r=20, t=20, b=40),
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)

        # Neighbors comparison
        try:
            ci = contracts.index(sym)
            neighbors = []
            if ci - 1 >= 0:
                neighbors.append(("Prev", contracts[ci - 1]))
            if ci + 1 < len(contracts):
                neighbors.append(("Next", contracts[ci + 1]))
            if neighbors:
                _side_section_header("Neighbors")
                rows = []
                for lbl, ns in neighbors:
                    try:
                        nval = wide_close.loc[wide_close.index.date == asof_date, ns].iloc[0]
                        spread = (last_close - nval) if (last_close is not None and not pd.isna(nval)) else None
                        rows.append((lbl + " " + ns,
                                     f"{nval:.4f}" if not pd.isna(nval) else "—",
                                     None,
                                     f"Δ {spread:+.4f}" if spread is not None else None))
                    except Exception:
                        pass
                if rows:
                    _kv_block(rows)
        except ValueError:
            pass


# =============================================================================
# MODE: STANDARD
# =============================================================================
def _chart_standard(scope_id, contracts, wide_close, wide_high, wide_low, available_dates,
                     ref_panel, allow_implied_rate, allow_carry, allow_fomc,
                     compare_on, animate_on, front_end, mid_end) -> dict:
    side_data = {"mode": "Standard"}

    if not animate_on:
        c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.4, 0.9])
        with c1:
            asof = st.selectbox("As-of date", options=available_dates[::-1], index=0,
                                key=f"{scope_id}_std_asof",
                                format_func=lambda d: d.strftime("%Y-%m-%d (%a)"))
        with c2:
            jump_options = ["—"] + list(contracts)
            jump_pick = st.selectbox(
                "Jump to contract", options=jump_options, index=0,
                key=f"{scope_id}_std_jump",
                help="Highlights the selected contract on the curve.",
            )
            jump_match = jump_pick if jump_pick and jump_pick != "—" else None
        with c3:
            if compare_on:
                qc = st.columns(5)
                quick_picked = None
                for i, (lbl, dys) in enumerate([("1D", 1), ("1W", 7), ("1M", 30),
                                                 ("3M", 90), ("YTD", -1)]):
                    if qc[i].button(lbl, key=f"{scope_id}_qc_{lbl}", use_container_width=True):
                        if dys == -1:
                            ytd_target = date(asof.year, 1, 1)
                            earlier = [d for d in available_dates if d <= ytd_target]
                            quick_picked = max(earlier) if earlier else available_dates[0]
                        else:
                            quick_picked = _quick_compare_offset(available_dates, asof, dys)
                if quick_picked:
                    st.session_state[f"{scope_id}_cmp_date_val"] = quick_picked
            else:
                st.markdown("&nbsp;", unsafe_allow_html=True)
        with c4:
            y_mode = "Price"
            if allow_implied_rate:
                y_mode = st.selectbox("Y axis", ["Price", "Implied rate"],
                                      key=f"{scope_id}_std_ymode")

        compare_date = None
        if compare_on:
            default_cmp = st.session_state.get(
                f"{scope_id}_cmp_date_val",
                _quick_compare_offset(available_dates, asof, 30),
            )
            try:
                idx = available_dates[::-1].index(default_cmp)
            except ValueError:
                idx = 0
            cc1, _ = st.columns([1.5, 5])
            with cc1:
                compare_date = st.selectbox(
                    "Compare date", options=available_dates[::-1], index=idx,
                    key=f"{scope_id}_std_cmp_date",
                    format_func=lambda d: d.strftime("%Y-%m-%d (%a)"),
                )

        # Overlay toggles + carry color (outrights only)
        n_overlays = 5 if allow_carry else 4
        oc = st.columns(n_overlays)
        from lib.markets import overnight_rate_label as _orl_lbl, get_market as _gm_lbl
        _cb = _gm_lbl(_BP).get("central_bank", "Fed")
        _on = _orl_lbl(_BP)
        with oc[0]:
            show_fed_band = st.checkbox(
                f"{_cb} band", value=True, key=f"{scope_id}_std_fed",
                disabled=not allow_implied_rate,
                help=f"{_cb} upper/lower bound. Plotted as horizontal band — "
                     "shown in implied-rate space AND in price space "
                     "(converted via 100 − rate).",
            )
        with oc[1]:
            show_sofr = st.checkbox(
                f"{_on} line", value=True, key=f"{scope_id}_std_sofr",
                disabled=not allow_implied_rate,
                help=f"Effective {_on} rate as horizontal line. "
                     f"In price-axis it shows as 100 − {_on}.",
            )
        with oc[2]:
            show_sections = st.checkbox("Section shading", value=True,
                                        key=f"{scope_id}_std_sect",
                                        help="Subtle background bands for FRONT / MID / BACK sections.")
        with oc[3]:
            show_delta = st.checkbox("Δ panel", value=False, key=f"{scope_id}_std_delta",
                                      help="Show per-contract bp change vs T-1 (or vs Compare date) as bar chart below.")
        if allow_carry:
            with oc[4]:
                carry_color = st.checkbox(
                    "Carry coloring", value=False, key=f"{scope_id}_std_carry",
                    help=(
                        "Markers tinted by single-contract roll-down carry (bp/day):\n\n"
                        "  carry_i (bp/day) = (rate_{i+1} − rate_i) × 100 / "
                        "days_between_expiries(i, i+1)\n\n"
                        "where rate_i = 100 − close_i (IMM convention) and the days "
                        "are calendar days between adjacent contract reference-period "
                        "starts (3rd Wed of contract month).\n\n"
                        "Positive carry (warm color) = curve slopes up from this contract "
                        "to the next, you earn bp/day rolling forward.\n"
                        "Negative carry (cool color) = curve slopes down, you lose bp/day.\n\n"
                        "The last contract has no successor → no carry."
                    ),
                )
        else:
            carry_color = False

        try:
            current_y = wide_close.loc[wide_close.index.date == asof].iloc[0].tolist()
            high_y = wide_high.loc[wide_high.index.date == asof].iloc[0].tolist()
            low_y = wide_low.loc[wide_low.index.date == asof].iloc[0].tolist()
        except Exception:
            current_y, high_y, low_y = [], [], []

        compare_y = None
        compare_label = None
        if compare_on and compare_date is not None:
            try:
                compare_y = wide_close.loc[wide_close.index.date == compare_date].iloc[0].tolist()
                compare_label = compare_date.strftime("%Y-%m-%d")
            except Exception:
                compare_y = None

        ref_rates = get_reference_rates_at(asof, ref_panel)
        is_implied = (allow_implied_rate and y_mode == "Implied rate")

        # Compute carry per contract (price-space, regardless of display mode)
        close_today = {c: (current_y[i] if i < len(current_y) else None)
                       for i, c in enumerate(contracts)}
        carry_per_day = compute_per_contract_carry(close_today, contracts) if allow_carry else {}

        if is_implied:
            current_y_disp = [(100 - v) if v is not None and not pd.isna(v) else None for v in current_y]
            high_disp = [(100 - v) if v is not None and not pd.isna(v) else None for v in low_y]
            low_disp = [(100 - v) if v is not None and not pd.isna(v) else None for v in high_y]
            compare_y_disp = ([(100 - v) if v is not None and not pd.isna(v) else None
                               for v in compare_y] if compare_y is not None else None)
            y_title = "Implied rate (%)"
        else:
            current_y_disp = current_y
            high_disp = high_y
            low_disp = low_y
            compare_y_disp = compare_y
            y_title = "Price"

        # Overlay specs — work in BOTH axis modes (rate or price-via-conversion).
        # Market-aware: SRA uses fdtr_upper/lower/sofr keys; non-SRA uses
        # rate_upper/rate_lower/overnight keys (set by the shim).
        from lib.markets import overnight_rate_label as _orl, get_market as _gm
        _cb_name = _gm(_BP).get("central_bank", "Fed")
        _on_label = _orl(_BP)
        h_lines = []
        h_bands = []
        if show_fed_band and allow_implied_rate:
            if _BP == "SRA":
                up = ref_rates.get("fdtr_upper")
                lo = ref_rates.get("fdtr_lower")
            else:
                up = ref_rates.get("rate_upper")
                lo = ref_rates.get("rate_lower")
            if up is not None and lo is not None:
                if is_implied:
                    band_lo, band_hi = lo, up
                    band_label = f"{_cb_name} {lo:.2f}–{up:.2f}%"
                else:  # price axis: convert rate → price (100 − rate)
                    band_lo, band_hi = 100 - up, 100 - lo
                    band_label = f"{_cb_name} {100-up:.3f}–{100-lo:.3f} (= {lo:.2f}–{up:.2f}%)"
                # Very subtle — barely perceptible band
                h_bands.append({"lower": band_lo, "upper": band_hi,
                                "color": "rgba(167, 139, 250, 0.07)",
                                "label": band_label})
        if show_sofr and allow_implied_rate:
            on_rate = ref_rates.get("sofr") if _BP == "SRA" else ref_rates.get("overnight")
            if on_rate is not None:
                if is_implied:
                    line_y, line_label = on_rate, f"{_on_label} {on_rate:.3f}%"
                else:  # price axis
                    line_y = 100 - on_rate
                    line_label = f"{_on_label} {line_y:.4f} (= {on_rate:.3f}%)"
                h_lines.append({"y": line_y, "color": "#22d3ee",
                                "label": line_label, "dash": "dash"})

        sect_arg = (front_end, mid_end) if show_sections else None

        # CB meeting overlay — vertical lines on x-axis at contracts spanning meetings.
        # SRA-only for now (annotate_contracts_with_fomcs is hardcoded to FOMC).
        fomc_annotations = []
        if allow_fomc and _BP == "SRA":
            contract_to_fomcs = annotate_contracts_with_fomcs(contracts)
            for i, c in enumerate(contracts):
                meets = contract_to_fomcs.get(c, [])
                if meets and any(m >= asof for m in meets):
                    fomc_annotations.append((i, c))

        if carry_color:
            carry_list = [carry_per_day.get(c) for c in contracts]
            fig = make_carry_colored_curve(
                contracts=contracts,
                current_y=current_y_disp,
                carry_per_day=carry_list,
                y_title=y_title, height=480,
                section_shading=sect_arg,
                horizontal_lines=h_lines,
                horizontal_bands=h_bands,
            )
        else:
            fig = make_curve_chart(
                contracts=contracts,
                current_y=current_y_disp,
                current_label=asof.strftime("%Y-%m-%d"),
                compare_y=compare_y_disp,
                compare_label=compare_label,
                high_y=high_disp if not compare_on else None,
                low_y=low_disp if not compare_on else None,
                y_title=y_title, height=480, value_decimals=4,
                section_shading=sect_arg,
                horizontal_lines=h_lines,
                horizontal_bands=h_bands,
                highlight_contract=jump_match,
            )

        # Add FOMC vertical-marker annotations
        if fomc_annotations:
            for idx, contract in fomc_annotations:
                fig.add_annotation(
                    x=idx, y=1, yref="paper",
                    text="🏛", showarrow=False,
                    font=dict(size=12, color=AMBER),
                    yshift=4,
                )
        st.plotly_chart(fig, use_container_width=True, theme=None)

        # Δ panel
        delta_active = show_delta or (compare_on and compare_date is not None)
        decomp = None
        changes_bp = None
        if delta_active:
            ref_compare_y = compare_y if compare_on else None
            ref_compare_label = compare_label
            if ref_compare_y is None:
                t_minus = available_dates[max(0, available_dates.index(asof) - 1)]
                try:
                    ref_compare_y = wide_close.loc[wide_close.index.date == t_minus].iloc[0].tolist()
                    ref_compare_label = t_minus.strftime("%Y-%m-%d")
                except Exception:
                    ref_compare_y = None
            if ref_compare_y is not None:
                changes_bp = compute_curve_change(current_y, ref_compare_y, in_bp_units=True,
                                                  contracts=contracts)
                fig_d = make_change_bar_chart(contracts, changes_bp,
                                              y_title=f"Δ vs {ref_compare_label} (bp)",
                                              height=200)
                st.plotly_chart(fig_d, use_container_width=True, theme=None)
                decomp = compute_decomposition(changes_bp)

        side_data.update({
            "asof": asof, "contracts": contracts, "current_y": current_y_disp,
            "ref_rates": ref_rates, "compare_label": compare_label,
            "jump_match": jump_match, "decomp": decomp, "changes_bp": changes_bp,
            "y_title": y_title, "carry": carry_per_day, "is_implied": is_implied,
            "current_y_price": current_y, "allow_fomc": allow_fomc,
        })
        return side_data

    # ANIMATION MODE
    ac1, ac2, ac3, ac4 = st.columns([1.2, 2, 1, 1.2])
    with ac1:
        anim_window = st.selectbox(
            "Window", ["Last 30d", "Last 60d", "Last 6mo", "Last 1y", "Custom"],
            index=2, key=f"{scope_id}_std_anim_win",
        )
    with ac2:
        if anim_window == "Custom":
            anim_start = st.date_input(
                "Start", value=available_dates[max(0, len(available_dates) - 60)],
                min_value=available_dates[0], max_value=available_dates[-1],
                key=f"{scope_id}_std_anim_start",
            )
        else:
            window_map = {"Last 30d": 30, "Last 60d": 60, "Last 6mo": 180, "Last 1y": 365}
            back = window_map[anim_window]
            anim_start = available_dates[max(0, len(available_dates) - back)]
            st.markdown(
                f"<div style='font-size:0.78rem; color:var(--text-dim); padding-top:1.7rem;'>"
                f"From <b style='color:var(--text-muted)'>{anim_start}</b> "
                f"→ <b style='color:var(--text-muted)'>{available_dates[-1]}</b></div>",
                unsafe_allow_html=True,
            )
    with ac3:
        speed = st.selectbox("Speed", ["Slow", "Medium", "Fast"], index=1,
                             key=f"{scope_id}_std_speed")
    with ac4:
        y_mode = "Price"
        if allow_implied_rate:
            y_mode = st.selectbox("Y axis", ["Price", "Implied rate"],
                                  key=f"{scope_id}_std_anim_ymode")

    speed_map = {"Slow": 250, "Medium": 120, "Fast": 50}
    sub = wide_close.loc[wide_close.index.date >= anim_start]
    if sub.empty:
        st.warning("No data in window.")
        return side_data

    if allow_implied_rate and y_mode == "Implied rate":
        frames_data = {idx.strftime("%Y-%m-%d"): [
            (100 - v) if v is not None and not pd.isna(v) else None for v in row.tolist()
        ] for idx, row in sub.iterrows()}
        y_title = "Implied rate (%)"
    else:
        frames_data = {idx.strftime("%Y-%m-%d"): row.tolist() for idx, row in sub.iterrows()}
        y_title = "Price"

    fig = make_animated_curve_chart(contracts=contracts, frames_data=frames_data,
                                    title="", y_title=y_title, height=500,
                                    frame_duration_ms=speed_map[speed])
    st.plotly_chart(fig, use_container_width=True, theme=None)
    st.caption(f"▶ Play / ⏸ Pause buttons or drag the slider beneath. {len(frames_data)} frames.")

    final_y = list(sub.iloc[-1].values)
    if allow_implied_rate and y_mode == "Implied rate":
        final_y = [(100 - v) if v is not None and not pd.isna(v) else None for v in final_y]
    side_data.update({"asof": available_dates[-1], "contracts": contracts,
                       "current_y": final_y, "ref_rates": {},
                       "y_title": y_title, "decomp": None, "changes_bp": None,
                       "compare_label": None, "jump_match": None, "carry": {},
                       "is_implied": False, "allow_fomc": False})
    return side_data


def _side_standard(side_data: dict, front_end: int, mid_end: int,
                    wide_close: pd.DataFrame, available_dates: list) -> None:
    _side_curve_summary(side_data["asof"], side_data["contracts"],
                        side_data["current_y"], front_end, mid_end)

    rr = side_data.get("ref_rates", {}) or {}
    # Market-aware reference-rate panel. For SRA the dict has fdtr_*/sofr keys
    # (from lib.sra_data.get_reference_rates_at). For non-SRA the shim's
    # _md.get_reference_rates_at returns rate_upper / rate_mid / rate_lower /
    # overnight — different field names — so we dispatch on _BP.
    if rr:
        from lib.markets import overnight_rate_label as _orl, get_market as _gm
        cb_name = _gm(_BP).get("central_bank", "Fed")
        on_label = _orl(_BP)
        if _BP == "SRA":
            _side_section_header(f"{cb_name} band & {on_label}")
            rows = [
                ("FDTR upper", f"{rr['fdtr_upper']:.3f}" if rr.get("fdtr_upper") is not None else "—",
                 "accent", None),
                ("FDTR mid", f"{rr['fdtr_mid']:.3f}" if rr.get("fdtr_mid") is not None else "—",
                 None, None),
                ("FDTR lower", f"{rr['fdtr_lower']:.3f}" if rr.get("fdtr_lower") is not None else "—",
                 "accent", None),
                (f"{on_label} effective",
                 f"{rr['sofr']:.3f}" if rr.get("sofr") is not None else "—",
                 "accent", None),
            ]
        else:
            _side_section_header(f"{cb_name} band & {on_label}")
            rows = [
                (f"{cb_name} upper",
                 f"{rr['rate_upper']:.3f}" if rr.get("rate_upper") is not None else "—",
                 "accent", None),
                (f"{cb_name} mid",
                 f"{rr['rate_mid']:.3f}" if rr.get("rate_mid") is not None else "—",
                 None, None),
                (f"{cb_name} lower",
                 f"{rr['rate_lower']:.3f}" if rr.get("rate_lower") is not None else "—",
                 "accent", None),
                (f"{on_label} effective",
                 f"{rr['overnight']:.3f}" if rr.get("overnight") is not None else "—",
                 "accent", None),
            ]
        _kv_block(rows)

    # CB step-path / implied-rate decomposition — works for every market.
    if side_data.get("allow_fomc"):
        from lib.markets import (cb_code_for as _ccf, cb_step_bp as _csb,
                                      settlement_convention as _sc,
                                      get_market as _gmk,
                                      overnight_rate_label as _orl_inline)
        _cb_code = _ccf(_BP)
        _step = _csb(_BP)
        _conv = _sc(_BP)
        _cb_name_inline = _gmk(_BP).get("central_bank", "Fed")
        _on_inline = _orl_inline(_BP)
        contracts = side_data["contracts"]
        current_y_price = side_data.get("current_y_price", [])
        contracts_with_rates = [
            (c, 100 - v) for c, v in zip(contracts, current_y_price)
            if v is not None and not pd.isna(v)
        ]
        if _BP == "SRA":
            anchor_v = (rr.get("sofr") if rr else None)
        else:
            anchor_v = (rr.get("overnight") if rr else None)
        sofr_anchor = anchor_v
        fomc_df = decompose_implied_rates(
            contracts_with_rates, side_data["asof"],
            anchor_rate_pct=anchor_v,
            horizon_months=18,
            ridge_lambda=0.5,
            cb_code=_cb_code,
            step_size_bp=_step,
        )
        if not fomc_df.empty:
            tooltip_attr = (get_methodology_text()
                            .replace('"', '&quot;')
                            .replace("'", "&#39;")
                            .replace("\n", "&#10;"))
            st.markdown(
                f"<div title=\"{tooltip_attr}\" style='cursor:help;'>",
                unsafe_allow_html=True,
            )
            if _conv == "forward_3m_fixing":
                _side_section_header(
                    f"Implied forward {_on_inline} path ({_cb_name_inline}) ⓘ")
            else:
                _side_section_header(
                    f"Implied policy path ({_cb_name_inline}) ⓘ")
            anchor_lbl_inline = ("policy rate" if _conv == "compounded_overnight"
                                    else _on_inline)
            st.caption(
                f"Anchored {anchor_lbl_inline}={anchor_v:.3f}% · "
                f"horizon 18 mo · ridge λ=0.5 · {_step:.0f}bp step · "
                "least-squares fit (hover the ⓘ for full methodology)"
                if anchor_v is not None else
                f"Auto-anchored · 18mo horizon · ridge λ=0.5 · {_step:.0f}bp step"
                " (hover ⓘ for methodology)"
            )
            for _, r in fomc_df.head(8).iterrows():
                d = r["meeting_date"]
                post = r["post_rate_pct"]
                delta = r["delta_bp"]
                cum = r["cum_from_front_bp"]
                p_h = r["prob_hike_25"]
                p_c = r["prob_cut_25"]
                p_hold = r["prob_hold"]
                if abs(delta) < 5:
                    badge = (f"<span style='color:var(--text-muted); font-weight:600;'>"
                             f"{p_hold*100:.0f}% hold</span>")
                elif delta > 0:
                    badge = (f"<span style='color:var(--red); font-weight:600;'>"
                             f"{p_h*100:.0f}% hike</span>")
                else:
                    badge = (f"<span style='color:var(--green); font-weight:600;'>"
                             f"{p_c*100:.0f}% cut</span>")
                st.markdown(
                    f"<div style='font-family:JetBrains Mono, monospace; font-size:0.72rem; "
                    f"padding:4px 0; border-bottom:1px solid var(--border-subtle);'>"
                    f"<div style='display:flex; justify-content:space-between; align-items:baseline;'>"
                    f"<span style='color:var(--accent);'>{d}</span>"
                    f"<span style='color:var(--text-body);'>{post:.3f}%</span></div>"
                    f"<div style='display:flex; justify-content:space-between; "
                    f"font-size:0.65rem; color:var(--text-dim); margin-top:2px;'>"
                    f"<span>Δ {delta:+.1f} bp · cum {cum:+.1f}</span>"
                    f"<span>{badge}</span></div></div>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

    # Carry summary
    carry = side_data.get("carry", {}) or {}
    if any(v is not None for v in carry.values()):
        _side_section_header("Carry (bp/day)")
        carry_pairs = [(c, v) for c, v in carry.items() if v is not None]
        # White-pack carry
        wp = compute_white_pack_carry(
            {c: side_data["current_y_price"][i] for i, c in enumerate(side_data["contracts"])
             if i < len(side_data["current_y_price"])},
            side_data["contracts"]
        )
        rows = []
        if wp is not None:
            rows.append(("White-pack avg", f"{wp:+.3f}", "accent", None))
        # Top-3 best & worst
        carry_pairs.sort(key=lambda x: x[1])
        for c, v in carry_pairs[:3]:
            rows.append((f"Worst · {c}", f"{v:+.3f}", "red", None))
        for c, v in carry_pairs[-3:][::-1]:
            rows.append((f"Best · {c}", f"{v:+.3f}", "green", None))
        if rows:
            _kv_block(rows)

    # Decomposition
    decomp = side_data.get("decomp")
    if decomp:
        _side_section_header("Move decomposition (bp)")
        p = decomp.get("parallel"); s = decomp.get("slope"); cv = decomp.get("curvature")
        rmse = decomp.get("residual_rmse")
        rows = [
            ("Parallel", f"{p:+.3f}" if p is not None else "—", _delta_class(p), "level shift"),
            ("Slope", f"{s:+.3f}" if s is not None else "—", _delta_class(s), "back vs front"),
            ("Curvature", f"{cv:+.3f}" if cv is not None else "—", _delta_class(cv), "belly bulge"),
            ("Fit RMSE", f"{rmse:.3f}" if rmse is not None else "—", "", None),
        ]
        _kv_block(rows)

        # Top movers
        changes = side_data.get("changes_bp", []) or []
        contracts = side_data["contracts"]
        valid = [(c, v) for c, v in zip(contracts, changes)
                 if v is not None and not pd.isna(v)]
        if valid:
            valid.sort(key=lambda x: -abs(x[1]))
            _side_section_header("Top movers")
            for c, v in valid[:6]:
                st.markdown(
                    f"<div style='font-family:JetBrains Mono, monospace; font-size:0.74rem; "
                    f"display:flex; justify-content:space-between; padding:3px 0;'>"
                    f"<span style='color:var(--text-body);'>{c}</span>"
                    f"<span style='color:{_delta_color(v)};'>{v:+.2f} bp</span>"
                    f"</div>", unsafe_allow_html=True)


# =============================================================================
# MODE: MULTI-DATE
# =============================================================================
def _chart_multi_date(scope_id, contracts, wide_close, available_dates,
                       allow_implied_rate, front_end, mid_end) -> dict:
    presets = {"Today": 0, "T-1": 1, "T-1W": 7, "T-1M": 30,
               "T-3M": 90, "T-6M": 180, "T-1Y": 365, "YTD start": -1}
    cc1, cc2 = st.columns([4, 1])
    with cc1:
        chosen = st.multiselect(
            "Dates to overlay (oldest → newest)", options=list(presets.keys()),
            default=["T-3M", "T-1M", "T-1W", "Today"],
            key=f"{scope_id}_md_picks",
        )
    with cc2:
        y_mode = "Price"
        if allow_implied_rate:
            y_mode = st.selectbox("Y axis", ["Price", "Implied rate"],
                                  key=f"{scope_id}_md_ymode")

    if not chosen:
        st.info("Pick at least one date.")
        return {"mode": "Multi-date"}

    today = available_dates[-1]
    date_pairs = []
    for label in chosen:
        offset = presets[label]
        if label == "Today":
            d = today
        elif offset == -1:
            target = date(today.year, 1, 1)
            earlier = [x for x in available_dates if x <= target]
            d = max(earlier) if earlier else available_dates[0]
        else:
            d = _quick_compare_offset(available_dates, today, offset)
        date_pairs.append((label, d))
    date_pairs.sort(key=lambda x: x[1])

    series_data, date_labels = [], []
    for lbl, d in date_pairs:
        try:
            row = wide_close.loc[wide_close.index.date == d].iloc[0].tolist()
        except Exception:
            row = [None] * len(contracts)
        if allow_implied_rate and y_mode == "Implied rate":
            row = [(100 - v) if v is not None and not pd.isna(v) else None for v in row]
        series_data.append(row)
        date_labels.append(f"{lbl} ({d.strftime('%Y-%m-%d')})")

    y_title = "Implied rate (%)" if (allow_implied_rate and y_mode == "Implied rate") else "Price"
    fig = make_multi_date_curve_chart(
        contracts=contracts, date_labels=date_labels, series_data=series_data,
        y_title=y_title, height=500, section_shading=(front_end, mid_end),
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)

    return {"mode": "Multi-date", "asof": today, "contracts": contracts,
            "current_y": series_data[-1] if series_data else [],
            "date_labels": date_labels, "series_data": series_data}


def _side_multi_date(side_data, front_end, mid_end) -> None:
    _side_curve_summary(side_data["asof"], side_data["contracts"],
                        side_data["current_y"], front_end, mid_end)
    date_labels = side_data.get("date_labels", [])
    series_data = side_data.get("series_data", [])
    contracts = side_data.get("contracts", [])
    if len(series_data) < 2:
        return
    _side_section_header("Cumulative Δ vs latest (bp)")
    from lib.contract_units import bp_multipliers_for, load_catalog
    cat = load_catalog()
    mults = bp_multipliers_for(contracts, _BP, cat) if not cat.empty \
            else [100.0] * len(contracts)
    latest_y = series_data[-1]
    rows = []
    for label, ys in zip(date_labels[:-1], series_data[:-1]):
        diffs = [(t - c) * m if (t is not None and c is not None
                                  and not pd.isna(t) and not pd.isna(c)) else None
                 for t, c, m in zip(latest_y, ys, mults)]
        valid_diffs = [d for d in diffs if d is not None]
        mean_d = sum(valid_diffs) / len(valid_diffs) if valid_diffs else None
        rows.append((label.split(" (")[0],
                     f"{mean_d:+.2f}" if mean_d is not None else "—",
                     _delta_class(mean_d), "avg across curve"))
    _kv_block(rows)


# =============================================================================
# MODE: RIBBON
# =============================================================================
def _chart_ribbon(scope_id, contracts, wide_close, available_dates,
                   allow_implied_rate, front_end, mid_end) -> dict:
    cc1, cc2, cc3, cc4 = st.columns([1.2, 1.2, 1.2, 1.2])
    with cc1:
        asof = st.selectbox("As-of", options=available_dates[::-1], index=0,
                            key=f"{scope_id}_rb_asof",
                            format_func=lambda d: d.strftime("%Y-%m-%d"))
    with cc2:
        lookback_label = st.selectbox(
            "Lookback",
            ["5d", "15d", "30d", "60d", "90d", "252d", "Max available"],
            index=3, key=f"{scope_id}_rb_lb",
        )
        lookback_map = {"5d": 5, "15d": 15, "30d": 30, "60d": 60,
                        "90d": 90, "252d": 252,
                        "Max available": len(available_dates)}
        lookback = lookback_map[lookback_label]
    with cc3:
        view_mode = st.selectbox("View", ["Ribbon", "Z-score bars"],
                                  key=f"{scope_id}_rb_view")
    with cc4:
        y_mode = "Price"
        if allow_implied_rate:
            y_mode = st.selectbox("Y axis", ["Price", "Implied rate"],
                                  key=f"{scope_id}_rb_ymode")

    if view_mode == "Ribbon":
        oc1, oc2 = st.columns(2)
        with oc1:
            show_mean = st.checkbox("Mean line", value=False, key=f"{scope_id}_rb_mean")
        with oc2:
            show_recent_avg = st.checkbox("Recent 5d avg", value=False,
                                           key=f"{scope_id}_rb_ravg")
    else:
        show_mean = show_recent_avg = False

    bands = compute_percentile_bands(wide_close, asof, lookback=lookback)
    zscores = compute_per_contract_zscores(wide_close, asof, lookback=lookback)
    pcts = compute_percentile_rank(wide_close, asof, lookback=lookback)

    try:
        today_y = wide_close.loc[wide_close.index.date == asof].iloc[0].tolist()
    except Exception:
        today_y = []

    if allow_implied_rate and y_mode == "Implied rate":
        today_y = [(100 - v) if v is not None and not pd.isna(v) else None for v in today_y]
        if bands:
            bands = {
                "p05": {c: (100 - bands["p95"].get(c)) if bands["p95"].get(c) is not None else None
                        for c in contracts},
                "p25": {c: (100 - bands["p75"].get(c)) if bands["p75"].get(c) is not None else None
                        for c in contracts},
                "p50": {c: (100 - bands["p50"].get(c)) if bands["p50"].get(c) is not None else None
                        for c in contracts},
                "p75": {c: (100 - bands["p25"].get(c)) if bands["p25"].get(c) is not None else None
                        for c in contracts},
                "p95": {c: (100 - bands["p05"].get(c)) if bands["p05"].get(c) is not None else None
                        for c in contracts},
                "mean": {c: (100 - bands["mean"].get(c)) if bands["mean"].get(c) is not None else None
                         for c in contracts},
            }
        y_title = "Implied rate (%)"
    else:
        y_title = "Price"

    if view_mode == "Z-score bars":
        z_list = [zscores.get(c) for c in contracts]
        fig = make_zscore_curve_chart(
            contracts=contracts, zscores=z_list,
            title=f"Z-score · {lookback_label} lookback",
            height=500, section_shading=(front_end, mid_end),
        )
    else:
        recent_avg = None
        if show_recent_avg:
            recent = wide_close.loc[wide_close.index <= pd.Timestamp(asof)].tail(5)
            if not recent.empty:
                recent_avg = recent.mean().tolist()
                if allow_implied_rate and y_mode == "Implied rate":
                    recent_avg = [(100 - v) if v is not None and not pd.isna(v) else None
                                   for v in recent_avg]
        fig = make_ribbon_chart(
            contracts=contracts, today_y=today_y,
            today_label=f"{asof.strftime('%Y-%m-%d')} · {lookback_label}",
            bands=bands, y_title=y_title, height=500,
            section_shading=(front_end, mid_end),
            show_extras={"mean": show_mean, "recent_avg": recent_avg},
        )

    st.plotly_chart(fig, use_container_width=True, theme=None)

    return {"mode": "Ribbon", "asof": asof, "contracts": contracts,
            "current_y": today_y, "zscores": zscores, "pcts": pcts,
            "lookback_label": lookback_label}


# =============================================================================
# UNIFIED ANALYSIS BLOCKS — render-functions called by the combined side panel
# =============================================================================

def _block_fed_sofr(ref_rates: dict) -> None:
    """Market-aware reference-rate panel.

    For SRA: SOFR + FDTR upper/mid/lower (legacy dict keys).
    For non-SRA: market's CB upper/mid/lower + overnight rate
    (e.g. ECB upper/lower + €STR for ER, BoE Bank Rate + SONIA for SON).
    """
    if not ref_rates:
        return
    from lib.markets import overnight_rate_label as _orl, get_market as _gm
    cb_name = _gm(_BP).get("central_bank", "Fed")
    on_label = _orl(_BP)
    _side_section_header(f"{cb_name} band & {on_label}")
    if _BP == "SRA":
        rows = [
            ("FDTR upper", f"{ref_rates['fdtr_upper']:.3f}%" if ref_rates.get("fdtr_upper") is not None else "—",
             "accent", None),
            ("FDTR mid", f"{ref_rates['fdtr_mid']:.3f}%" if ref_rates.get("fdtr_mid") is not None else "—",
             None, None),
            ("FDTR lower", f"{ref_rates['fdtr_lower']:.3f}%" if ref_rates.get("fdtr_lower") is not None else "—",
             "accent", None),
            (f"{on_label} effective",
             f"{ref_rates['sofr']:.3f}%" if ref_rates.get("sofr") is not None else "—",
             "accent", None),
        ]
    else:
        rows = [
            (f"{cb_name} upper",
             f"{ref_rates['rate_upper']:.3f}%" if ref_rates.get("rate_upper") is not None else "—",
             "accent", None),
            (f"{cb_name} mid",
             f"{ref_rates['rate_mid']:.3f}%" if ref_rates.get("rate_mid") is not None else "—",
             None, None),
            (f"{cb_name} lower",
             f"{ref_rates['rate_lower']:.3f}%" if ref_rates.get("rate_lower") is not None else "—",
             "accent", None),
            (f"{on_label} effective",
             f"{ref_rates['overnight']:.3f}%" if ref_rates.get("overnight") is not None else "—",
             "accent", None),
        ]
    _kv_block(rows)


def _block_fomc_path(contracts, current_y_price, asof, sofr_anchor) -> None:
    """Implied policy path — works for every market.

    For SRA: SR3 contracts → FOMC meetings → implied policy rate per meeting
    For ER : Euribor → ECB meetings → implied forward 3M Euribor (= ECB + basis)
    For FSR: SARON → SNB meetings → implied SARON-compounded rate per meeting
    For FER: €STR → ECB meetings → implied €STR-compounded rate per meeting
    For SON: SONIA → BoE meetings → implied SONIA-compounded rate per meeting
    For YBA: Bank Bill → RBA meetings → implied forward 3M BBSW (= RBA + basis)
    For CRA: CORRA → BoC meetings → implied CORRA-compounded rate per meeting

    The decomposition math is identical across markets (day-weighted average
    of inter-meeting overnight rate, regularised least-squares). The CB
    calendar, step size, and anchor rate switch per market.
    """
    from lib.markets import (cb_code_for, cb_step_bp, settlement_convention,
                                  get_market as _gm,
                                  overnight_rate_label as _orl)
    cb_code = cb_code_for(_BP)
    step_bp = cb_step_bp(_BP)
    conv = settlement_convention(_BP)
    cb_name = _gm(_BP).get("central_bank", "Fed")
    on_label = _orl(_BP)
    contracts_with_rates = [
        (c, 100 - v) for c, v in zip(contracts, current_y_price)
        if v is not None and not pd.isna(v)
    ]
    fomc_df = decompose_implied_rates(
        contracts_with_rates, asof,
        anchor_rate_pct=sofr_anchor,
        horizon_months=18,
        ridge_lambda=0.5,
        cb_code=cb_code,
        step_size_bp=step_bp,
    )
    if fomc_df.empty:
        return
    tooltip_attr = (get_methodology_text()
                    .replace('"', '&quot;')
                    .replace("'", "&#39;")
                    .replace("\n", "&#10;"))
    st.markdown(f"<div title=\"{tooltip_attr}\" style='cursor:help;'>",
                unsafe_allow_html=True)
    # Section header — explicit about what's being decomposed:
    #   compounded_overnight markets → "Implied policy path ({CB})"
    #   forward_3m_fixing markets    → "Implied forward {rate} path ({CB} reaction)"
    if conv == "forward_3m_fixing":
        _side_section_header(f"Implied forward {on_label} path ({cb_name}) ⓘ")
    else:
        _side_section_header(f"Implied policy path ({cb_name}) ⓘ")
    anchor_lbl = "policy rate" if conv == "compounded_overnight" else f"{on_label}"
    if sofr_anchor is not None:
        st.caption(f"Anchored {anchor_lbl}={sofr_anchor:.3f}% · "
                       f"18mo · ridge λ=0.5 · {step_bp:.0f}bp step · LSQ fit")
    else:
        st.caption(f"Auto-anchored · 18mo · ridge λ=0.5 · {step_bp:.0f}bp step (hover ⓘ)")
    for _, r in fomc_df.head(8).iterrows():
        d = r["meeting_date"]
        post = r["post_rate_pct"]
        delta = r["delta_bp"]
        cum = r["cum_from_front_bp"]
        p_h = r["prob_hike_25"]
        p_c = r["prob_cut_25"]
        p_hold = r["prob_hold"]
        if abs(delta) < 5:
            badge = (f"<span style='color:var(--text-muted); font-weight:600;'>"
                     f"{p_hold*100:.0f}% hold</span>")
        elif delta > 0:
            badge = f"<span style='color:var(--red); font-weight:600;'>{p_h*100:.0f}% hike</span>"
        else:
            badge = f"<span style='color:var(--green); font-weight:600;'>{p_c*100:.0f}% cut</span>"
        st.markdown(
            f"<div style='font-family:JetBrains Mono, monospace; font-size:0.72rem; "
            f"padding:4px 0; border-bottom:1px solid var(--border-subtle);'>"
            f"<div style='display:flex; justify-content:space-between; align-items:baseline;'>"
            f"<span style='color:var(--accent);'>{d}</span>"
            f"<span style='color:var(--text-body);'>{post:.3f}%</span></div>"
            f"<div style='display:flex; justify-content:space-between; "
            f"font-size:0.65rem; color:var(--text-dim); margin-top:2px;'>"
            f"<span>Δ {delta:+.1f} bp · cum {cum:+.1f}</span>"
            f"<span>{badge}</span></div></div>",
            unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _block_carry(contracts, current_y_price) -> None:
    close_today = {c: (current_y_price[i] if i < len(current_y_price) else None)
                   for i, c in enumerate(contracts)}
    carry_per_day = compute_per_contract_carry(close_today, contracts)
    if not any(v is not None for v in carry_per_day.values()):
        return
    _side_section_header("Carry (bp/day)")
    wp = compute_white_pack_carry(close_today, contracts)
    rows = []
    if wp is not None:
        rows.append(("White-pack avg", f"{wp:+.3f}", "accent", None))
    carry_pairs = [(c, v) for c, v in carry_per_day.items() if v is not None]
    carry_pairs.sort(key=lambda x: x[1])
    for c, v in carry_pairs[:3]:
        rows.append((f"Worst · {c}", f"{v:+.3f}", "red", None))
    for c, v in carry_pairs[-3:][::-1]:
        rows.append((f"Best · {c}", f"{v:+.3f}", "green", None))
    if rows:
        _kv_block(rows)


def _block_decomposition(decomp, contracts, changes_bp, compare_label) -> None:
    if not decomp or not changes_bp:
        return
    label = f"Move decomposition (bp) · vs {compare_label}" if compare_label else \
            "Move decomposition (bp)"
    _side_section_header(label)
    p = decomp.get("parallel"); s = decomp.get("slope"); cv = decomp.get("curvature")
    rmse = decomp.get("residual_rmse")
    rows = [
        ("Parallel", f"{p:+.3f}" if p is not None else "—", _delta_class(p), "level shift"),
        ("Slope", f"{s:+.3f}" if s is not None else "—", _delta_class(s), "back vs front"),
        ("Curvature", f"{cv:+.3f}" if cv is not None else "—", _delta_class(cv), "belly bulge"),
        ("Fit RMSE", f"{rmse:.3f}" if rmse is not None else "—", "", None),
    ]
    _kv_block(rows)
    valid = [(c, v) for c, v in zip(contracts, changes_bp)
             if v is not None and not pd.isna(v)]
    if valid:
        valid.sort(key=lambda x: -abs(x[1]))
        _side_section_header("Top movers")
        for c, v in valid[:6]:
            st.markdown(
                f"<div style='font-family:JetBrains Mono, monospace; font-size:0.74rem; "
                f"display:flex; justify-content:space-between; padding:3px 0;'>"
                f"<span style='color:var(--text-body);'>{c}</span>"
                f"<span style='color:{_delta_color(v)};'>{v:+.2f} bp</span>"
                f"</div>", unsafe_allow_html=True)


def _block_z_multi_lookback(wide_close, asof, contracts) -> None:
    """Compute z-scores at 5/15/30/60/90 days and render comparison + pattern."""
    if wide_close is None or wide_close.empty:
        return
    multi_lb_labels = ["5d", "15d", "30d", "60d", "90d"]
    multi_lb_days = [5, 15, 30, 60, 90]
    multi_z = {}
    for lbl, days in zip(multi_lb_labels, multi_lb_days):
        multi_z[lbl] = compute_per_contract_zscores(wide_close, asof, lookback=days)

    # Quick regime counts using the 30d lookback (a stable mid-term reference)
    z30 = multi_z["30d"]
    extreme = [c for c in contracts if z30.get(c) is not None and abs(z30[c]) >= 2]
    elevated = [c for c in contracts if z30.get(c) is not None and 1 <= abs(z30[c]) < 2]
    normal = [c for c in contracts if z30.get(c) is not None and abs(z30[c]) < 1]
    _side_section_header("Z-score regime (30d ref)")
    _kv_block([
        ("|z| ≥ 2 extreme", str(len(extreme)), "red" if extreme else "", None),
        ("|z| 1–2 elevated", str(len(elevated)), "amber" if elevated else "", None),
        ("|z| < 1 normal", str(len(normal)), "green", None),
    ])

    # Top contracts by max |z| across all 5 lookbacks
    max_abs_z = {}
    for c in contracts:
        zs = [multi_z[lbl].get(c) for lbl in multi_lb_labels]
        zs_valid = [abs(z) for z in zs if z is not None and not pd.isna(z)]
        if zs_valid:
            max_abs_z[c] = max(zs_valid)
    ranked = sorted(max_abs_z.items(), key=lambda x: -x[1])[:8]
    if not ranked:
        return

    _side_section_header("Z across lookbacks · top 8")
    # Header
    hdr = ("<div style='display:grid; "
           "grid-template-columns: 64px repeat(5, 1fr) 96px; "
           "gap: 3px; padding:3px 0; "
           "font-size:0.58rem; color:var(--text-dim); "
           "text-transform:uppercase; letter-spacing:0.04em; "
           "border-bottom:1px solid var(--border-subtle); margin-bottom:2px;'>"
           "<span>Contract</span>"
           "<span style='text-align:right;'>5d</span>"
           "<span style='text-align:right;'>15d</span>"
           "<span style='text-align:right;'>30d</span>"
           "<span style='text-align:right;'>60d</span>"
           "<span style='text-align:right;'>90d</span>"
           "<span style='text-align:right;'>Pattern</span>"
           "</div>")
    st.markdown(hdr, unsafe_allow_html=True)
    for c, _max in ranked:
        z5 = multi_z["5d"].get(c)
        z15 = multi_z["15d"].get(c)
        z30 = multi_z["30d"].get(c)
        z60 = multi_z["60d"].get(c)
        z90 = multi_z["90d"].get(c)

        def _zspan(z):
            if z is None or pd.isna(z):
                return "<span style='color:var(--text-dim); text-align:right;'>—</span>"
            return (f"<span style='color:{_z_color(z)}; text-align:right; "
                    f"font-family:JetBrains Mono, monospace;'>{z:+.2f}</span>")

        pattern = _interpret_z_pattern_5d(z5, z15, z30, z60, z90)
        pat_color = pattern.get("color", "var(--text-muted)")
        pat_text = pattern.get("label", "—")
        row = (
            f"<div style='display:grid; "
            f"grid-template-columns: 64px repeat(5, 1fr) 96px; "
            f"gap: 3px; padding:3px 0; align-items:center; "
            f"border-bottom:1px solid var(--border-subtle); "
            f"font-size:0.7rem;'>"
            f"<span style='color:var(--text-body); "
            f"font-family:JetBrains Mono, monospace;'>{c}</span>"
            f"{_zspan(z5)}{_zspan(z15)}{_zspan(z30)}{_zspan(z60)}{_zspan(z90)}"
            f"<span style='color:{pat_color}; text-align:right; "
            f"font-size:0.62rem; font-weight:500;'>{pat_text}</span>"
            f"</div>"
        )
        st.markdown(row, unsafe_allow_html=True)

    # Pattern legend
    with st.expander("Pattern legend (z 5d vs 15d/30d/60d/90d)", expanded=False):
        st.markdown(
            """
            <div style='font-size:0.72rem; color:var(--text-muted); line-height:1.65;'>
            <b>FRESH</b> &nbsp;|z₅| ≥ 1.5, others &lt; 1.0 — pure short-term move.
            <i>Event-driven (CPI / NFP / FOMC). Often reverts within days.</i><br>
            <b>ACCELERATING</b> &nbsp;same sign, |z₅| &gt; |z₁₅| &gt; |z₃₀| with longer-term elevated —
            momentum building. <i>Trend-follow with caution; could exhaust.</i><br>
            <b>DECELERATING</b> &nbsp;same sign, |z₅| &lt; |z₃₀| with longer still elevated —
            momentum fading. <i>Watch for reversal — partial exit candidate.</i><br>
            <b>PERSISTENT</b> &nbsp;all 5 lookbacks elevated and same sign with similar magnitudes —
            sustained directional regime. <i>Trend-follow if confirmed.</i><br>
            <b>REVERTING</b> &nbsp;z₅ and z₉₀ have <i>opposite signs</i> with both ≥ 1 in magnitude —
            recent bounce against secular trend. <i>Watch — could be turn or temporary fade.</i><br>
            <b>DRIFTED</b> &nbsp;|z₆₀| or |z₉₀| ≥ 1.5 but |z₅| &lt; 1 — slow drift, recent stabilisation.
            <i>"Coiled" — next move imminent; watch for catalyst.</i><br>
            <b>STABLE</b> &nbsp;all |z| &lt; 1 — no signal.<br>
            <b>MIXED</b> — none of the above patterns matches cleanly.
            </div>
            """,
            unsafe_allow_html=True,
        )


def _interpret_z_pattern_5d(z5, z15, z30, z60, z90) -> dict:
    """Pattern label using 5d compared with each longer lookback."""
    def safe(z):
        return z if (z is not None and not pd.isna(z)) else 0.0

    z5_, z15_, z30_, z60_, z90_ = safe(z5), safe(z15), safe(z30), safe(z60), safe(z90)
    abs_zs = [abs(z) for z in (z5_, z15_, z30_, z60_, z90_)]
    z_long = (z5_, z15_, z30_, z60_, z90_)

    if all(a < 1.0 for a in abs_zs):
        return {"label": "STABLE", "color": "var(--text-dim)"}

    # FRESH: 5d only is extreme
    if abs(z5_) >= 1.5 and all(a < 1.0 for a in abs_zs[1:]):
        return {"label": "FRESH", "color": "var(--blue)"}

    # DRIFTED: 60d or 90d extreme but 5d quiet
    if (abs(z60_) >= 1.5 or abs(z90_) >= 1.5) and abs(z5_) < 1.0 and abs(z15_) < 1.0:
        return {"label": "DRIFTED", "color": "var(--amber)"}

    # REVERTING: 5d opposite sign from 90d, both magnitude >= 1
    if z5_ * z90_ < 0 and abs(z5_) >= 1.0 and abs(z90_) >= 1.0:
        return {"label": "REVERTING", "color": "var(--green)"}

    # All same sign?
    signs = [1 if z > 0 else (-1 if z < 0 else 0) for z in z_long]
    nonzero_signs = [s for s in signs if s != 0]
    if nonzero_signs and all(s == nonzero_signs[0] for s in nonzero_signs):
        # PERSISTENT: all elevated similar magnitude
        elevated_count = sum(1 for a in abs_zs if a >= 1.0)
        if elevated_count >= 4:
            spread = max(abs_zs) - min(abs_zs)
            if spread <= max(abs_zs) * 0.4:
                return {"label": "PERSISTENT", "color": "var(--red)"}
        # ACCELERATING: |z5| >= |z15| >= |z30| with z60/z90 also elevated
        if (abs(z5_) >= abs(z15_) and abs(z15_) >= abs(z30_)
                and abs(z30_) >= 1.0 and abs(z60_) >= 0.5):
            return {"label": "ACCELERATING", "color": "var(--red)"}
        # DECELERATING: |z5| <= |z15| <= |z30| with longer elevated
        if (abs(z5_) <= abs(z15_) and abs(z15_) <= abs(z30_)
                and abs(z30_) >= 1.0):
            return {"label": "DECELERATING", "color": "var(--amber)"}

    return {"label": "MIXED", "color": "var(--text-muted)"}


def _block_pack(pack_groups, pack_means, pack_slopes, pack_ranges) -> None:
    if not pack_groups:
        return
    _side_section_header("Pack metrics")
    for i, (pack_name, syms) in enumerate(pack_groups):
        color = PACK_COLORS[i] if i < len(PACK_COLORS) else BLUE
        mean_v = pack_means.get(pack_name)
        slope_v = pack_slopes.get(pack_name)
        rng_v = pack_ranges.get(pack_name)
        if mean_v is None:
            continue
        st.markdown(
            f"<div style='border-left:3px solid {color}; padding:0.3rem 0.5rem; "
            f"margin-bottom:0.4rem; background:var(--bg-base); border-radius:0 4px 4px 0;'>"
            f"<div style='display:flex; justify-content:space-between; align-items:baseline;'>"
            f"<span style='color:{color}; font-weight:600; font-size:0.78rem;'>{pack_name}</span>"
            f"<span style='color:var(--text-dim); font-family:JetBrains Mono, monospace; "
            f"font-size:0.65rem;'>{', '.join(syms)}</span></div>"
            f"<div style='font-family:JetBrains Mono, monospace; font-size:0.72rem; "
            f"color:var(--text-muted); margin-top:2px;'>"
            f"mean <span style='color:var(--accent);'>{mean_v:.4f}</span> · "
            f"slope <span style='color:{GREEN if (slope_v or 0) > 0 else (RED if (slope_v or 0) < 0 else TEXT_BODY)};'>"
            f"{slope_v:+.4f}</span> · range <span style='color:var(--text-body);'>{rng_v:.4f}</span>"
            f"</div></div>", unsafe_allow_html=True)
    if len(pack_groups) >= 2:
        _side_section_header("Pack spreads")
        spread_rows = []
        for i in range(len(pack_groups) - 1):
            n1, n2 = pack_groups[i][0], pack_groups[i + 1][0]
            v1 = pack_means.get(n1); v2 = pack_means.get(n2)
            if v1 is not None and v2 is not None:
                d = v2 - v1
                spread_rows.append((f"{n2} − {n1}", f"{d:+.4f}", _delta_class(d), None))
        if spread_rows:
            _kv_block(spread_rows)


def _block_volume_delta(contracts, wide_volume, asof, available_dates) -> None:
    if wide_volume is None or wide_volume.empty:
        return
    if asof not in available_dates:
        return
    idx = available_dates.index(asof)
    if idx <= 0:
        return
    prior = available_dates[idx - 1]
    try:
        vol_today = wide_volume.loc[wide_volume.index.date == asof].iloc[0]
        vol_prior = wide_volume.loc[wide_volume.index.date == prior].iloc[0]
    except Exception:
        return
    pairs = []
    for c in contracts:
        t = vol_today.get(c)
        p = vol_prior.get(c)
        if (t is None or p is None or pd.isna(t) or pd.isna(p)):
            continue
        d = t - p
        if d == 0:
            continue
        pairs.append((c, d))
    if not pairs:
        return
    pairs.sort(key=lambda x: -abs(x[1]))
    _side_section_header(f"Volume Δ vs {prior}")
    for c, v in pairs[:8]:
        st.markdown(
            f"<div style='font-family:JetBrains Mono, monospace; font-size:0.74rem; "
            f"display:flex; justify-content:space-between; padding:3px 0;'>"
            f"<span style='color:var(--text-body);'>{c}</span>"
            f"<span style='color:{_delta_color(v)};'>{v:+.0f}</span></div>",
            unsafe_allow_html=True)


def _render_combined_side_panel(
    asof, contracts, current_y_disp, current_y_price, ref_rates,
    wide_close, wide_volume, front_end, mid_end, strategy,
    allow_implied_rate, available_dates, symbols_df=None,
    decomp=None, changes_bp=None, compare_label=None,
) -> None:
    """One panel that renders ALL applicable analytical blocks regardless of mode."""
    # 1. Curve summary + Section slopes
    _side_curve_summary(asof, contracts, current_y_disp, front_end, mid_end)

    # 2. Fed band & SOFR (outrights only)
    if allow_implied_rate and ref_rates:
        _block_fed_sofr(ref_rates)

    # 3. FOMC implied policy path (outrights only)
    if strategy == "outright":
        _block_fomc_path(contracts, current_y_price, asof,
                          ref_rates.get("sofr") if ref_rates else None)

    # 4. Carry analysis (outrights only)
    if strategy == "outright":
        _block_carry(contracts, current_y_price)

    # 5. Move decomposition + Top movers (auto vs T-1, or vs compare)
    if decomp:
        _block_decomposition(decomp, contracts, changes_bp, compare_label)

    # 6. Z-score multi-lookback comparison
    _block_z_multi_lookback(wide_close, asof, contracts)

    # 7. Pack metrics (outrights only, when there are quarterlies)
    if strategy == "outright" and symbols_df is not None:
        pack_groups = compute_pack_groups(symbols_df)
        if pack_groups:
            pack_means, pack_slopes, pack_ranges = {}, {}, {}
            for pack_name, syms in pack_groups:
                vals = []
                for s in syms:
                    if s in contracts:
                        idx = contracts.index(s)
                        v = current_y_price[idx] if idx < len(current_y_price) else None
                        if v is not None and not pd.isna(v):
                            vals.append(v)
                if vals:
                    pack_means[pack_name] = sum(vals) / len(vals)
                    pack_slopes[pack_name] = vals[-1] - vals[0] if len(vals) >= 2 else 0
                    pack_ranges[pack_name] = max(vals) - min(vals)
                else:
                    pack_means[pack_name] = None
                    pack_slopes[pack_name] = None
                    pack_ranges[pack_name] = None
            _block_pack(pack_groups, pack_means, pack_slopes, pack_ranges)

    # 8. Volume Δ (vs T-1)
    _block_volume_delta(contracts, wide_volume, asof, available_dates)


def _side_ribbon(side_data, front_end, mid_end, wide_close=None) -> None:
    _side_curve_summary(side_data["asof"], side_data["contracts"],
                        side_data["current_y"], front_end, mid_end)
    zscores = side_data.get("zscores", {})
    pcts = side_data.get("pcts", {})
    contracts = side_data["contracts"]
    lb = side_data.get("lookback_label", "")

    _side_section_header(f"Z-score regime ({lb})")
    extreme = [c for c in contracts if zscores.get(c) is not None and abs(zscores[c]) >= 2]
    elevated = [c for c in contracts if zscores.get(c) is not None and 1 <= abs(zscores[c]) < 2]
    normal = [c for c in contracts if zscores.get(c) is not None and abs(zscores[c]) < 1]
    _kv_block([
        ("|z| ≥ 2 extreme", str(len(extreme)), "red" if extreme else "", None),
        ("|z| 1–2 elevated", str(len(elevated)), "amber" if elevated else "", None),
        ("|z| < 1 normal", str(len(normal)), "green", None),
    ])

    # ----- Z across multiple lookbacks (5/30/60/90) — analytical block -----
    if wide_close is not None and not wide_close.empty:
        asof = side_data["asof"]
        multi_lb_labels = ["5d", "30d", "60d", "90d"]
        multi_lb_days = [5, 30, 60, 90]
        multi_z = {}
        for lbl, days in zip(multi_lb_labels, multi_lb_days):
            multi_z[lbl] = compute_per_contract_zscores(wide_close, asof, lookback=days)

        # Top 8 contracts by max |z| across lookbacks
        max_abs_z = {}
        for c in contracts:
            zs = [multi_z[lbl].get(c) for lbl in multi_lb_labels]
            zs_valid = [abs(z) for z in zs if z is not None and not pd.isna(z)]
            if zs_valid:
                max_abs_z[c] = max(zs_valid)
        ranked = sorted(max_abs_z.items(), key=lambda x: -x[1])[:8]

        if ranked:
            _side_section_header("Z across lookbacks · top 8")
            # header row
            hdr = ("<div style='display:grid; "
                   "grid-template-columns: 70px repeat(4, 1fr) 80px; "
                   "gap: 4px; padding:3px 0; "
                   "font-size:0.6rem; color:var(--text-dim); "
                   "text-transform:uppercase; letter-spacing:0.04em; "
                   "border-bottom:1px solid var(--border-subtle); margin-bottom:2px;'>"
                   "<span>Contract</span>"
                   "<span style='text-align:right;'>5d</span>"
                   "<span style='text-align:right;'>30d</span>"
                   "<span style='text-align:right;'>60d</span>"
                   "<span style='text-align:right;'>90d</span>"
                   "<span style='text-align:right;'>Pattern</span>"
                   "</div>")
            st.markdown(hdr, unsafe_allow_html=True)
            for c, _max in ranked:
                z5 = multi_z["5d"].get(c)
                z30 = multi_z["30d"].get(c)
                z60 = multi_z["60d"].get(c)
                z90 = multi_z["90d"].get(c)

                def _zspan(z):
                    if z is None or pd.isna(z):
                        return "<span style='color:var(--text-dim); text-align:right;'>—</span>"
                    return (f"<span style='color:{_z_color(z)}; text-align:right; "
                            f"font-family:JetBrains Mono, monospace;'>{z:+.2f}</span>")

                pattern = _interpret_z_pattern(z5, z30, z60, z90)
                pat_color = pattern.get("color", "var(--text-muted)")
                pat_text = pattern.get("label", "—")

                row = (
                    f"<div style='display:grid; "
                    f"grid-template-columns: 70px repeat(4, 1fr) 80px; "
                    f"gap: 4px; padding:3px 0; align-items:center; "
                    f"border-bottom:1px solid var(--border-subtle); "
                    f"font-size:0.7rem;'>"
                    f"<span style='color:var(--text-body); "
                    f"font-family:JetBrains Mono, monospace;'>{c}</span>"
                    f"{_zspan(z5)}"
                    f"{_zspan(z30)}"
                    f"{_zspan(z60)}"
                    f"{_zspan(z90)}"
                    f"<span style='color:{pat_color}; text-align:right; "
                    f"font-size:0.62rem; font-weight:500;'>{pat_text}</span>"
                    f"</div>"
                )
                st.markdown(row, unsafe_allow_html=True)

            # Pattern legend
            with st.expander("Pattern interpretations", expanded=False):
                st.markdown(
                    """
                    <div style='font-size:0.72rem; color:var(--text-muted); line-height:1.6;'>
                    <b>HIST EXTREME</b> — large |z| at long lookback (90d) — rich/cheap vs ~4 months of history.
                    Trade direction: <i>fade</i> if mean-reverting market, <i>follow</i> if regime shift confirmed.<br>
                    <b>RECENT MOVE</b> — short-term z (5d/30d) extreme but long-term (90d) normal — fresh repricing event.
                    Likely event-driven (CPI, NFP, FOMC). Often reverts within 1–2 weeks.<br>
                    <b>DRIFTED</b> — long-term z extreme but short-term normal — slow secular drift.
                    The distance has accumulated; momentum may continue.<br>
                    <b>MEAN-REVERTING</b> — 5d and 90d have <i>opposite signs</i> — recent bounce against the trend.
                    Watch for confirmation: either continuation (trend reversal) or fade back to extreme.<br>
                    <b>PERSISTENT</b> — 5d and 90d same sign and elevated — sustained directional move.
                    Trend-following candidate.<br>
                    <b>STABLE</b> — all lookbacks |z| &lt; 1 — no signal.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    if extreme:
        _side_section_header("⚠ Current-lookback extremes")
        for c in extreme[:8]:
            z = zscores[c]; p = pcts.get(c)
            p_str = f"p{int(p)}" if p is not None else "—"
            st.markdown(
                f"<div style='font-family:JetBrains Mono, monospace; font-size:0.74rem; "
                f"display:flex; justify-content:space-between; padding:3px 0;'>"
                f"<span style='color:var(--text-body);'>{c}</span>"
                f"<span style='color:{_z_color(z)};'>{z:+.2f}σ · {p_str}</span></div>",
                unsafe_allow_html=True)


def _interpret_z_pattern(z5, z30, z60, z90) -> dict:
    """Interpret a (z5d, z30d, z60d, z90d) tuple into a trader-relevant pattern label."""
    def safe(z):
        return z if (z is not None and not pd.isna(z)) else 0.0

    z5_, z30_, z60_, z90_ = safe(z5), safe(z30), safe(z60), safe(z90)

    if all(abs(z) < 1.0 for z in (z5_, z30_, z60_, z90_)):
        return {"label": "STABLE", "color": "var(--text-dim)"}

    long_extreme = abs(z90_) >= 2.0
    short_extreme = abs(z5_) >= 2.0 or abs(z30_) >= 2.0
    long_elevated = abs(z90_) >= 1.0
    short_elevated = abs(z5_) >= 1.0 or abs(z30_) >= 1.0

    # Sign comparison (non-zero check)
    same_sign_5_90 = (z5_ * z90_) > 0
    opp_sign_5_90 = (z5_ * z90_) < 0 and abs(z5_) > 0.5 and abs(z90_) > 0.5

    if long_extreme and short_extreme and same_sign_5_90:
        return {"label": "PERSISTENT", "color": "var(--red)"}
    if long_extreme and not short_elevated:
        return {"label": "DRIFTED", "color": "var(--amber)"}
    if short_extreme and not long_elevated:
        return {"label": "RECENT MOVE", "color": "var(--blue)"}
    if opp_sign_5_90:
        return {"label": "MEAN-REVERTING", "color": "var(--green)"}
    if long_elevated and same_sign_5_90:
        return {"label": "HIST EXTREME", "color": "var(--red)"}
    return {"label": "MIXED", "color": "var(--text-muted)"}


# =============================================================================
# MODE: PACK (Outrights only)
# =============================================================================
def _chart_pack(scope_id, contracts, symbols_df, wide_close, available_dates,
                 allow_implied_rate, front_end, mid_end) -> dict:
    cc1, cc2 = st.columns([2, 1])
    with cc1:
        asof = st.selectbox("As-of", options=available_dates[::-1], index=0,
                            key=f"{scope_id}_pk_asof",
                            format_func=lambda d: d.strftime("%Y-%m-%d (%a)"))
    with cc2:
        y_mode = "Price"
        if allow_implied_rate:
            y_mode = st.selectbox("Y axis", ["Price", "Implied rate"],
                                  key=f"{scope_id}_pk_ymode")

    pack_groups = compute_pack_groups(symbols_df)
    if not pack_groups:
        st.info("Pack view requires quarterly contracts (H/M/U/Z).")
        return {"mode": "Pack"}

    try:
        current_y = wide_close.loc[wide_close.index.date == asof].iloc[0].tolist()
    except Exception:
        current_y = []

    if allow_implied_rate and y_mode == "Implied rate":
        current_y = [(100 - v) if v is not None and not pd.isna(v) else None for v in current_y]
        y_title = "Implied rate (%)"
    else:
        y_title = "Price"

    pack_means, pack_slopes, pack_ranges = {}, {}, {}
    for pack_name, syms in pack_groups:
        vals = []
        for s in syms:
            if s in contracts:
                idx = contracts.index(s)
                v = current_y[idx] if idx < len(current_y) else None
                if v is not None and not pd.isna(v):
                    vals.append(v)
        if not vals:
            pack_means[pack_name] = None
            pack_slopes[pack_name] = None
            pack_ranges[pack_name] = None
        else:
            pack_means[pack_name] = sum(vals) / len(vals)
            pack_slopes[pack_name] = vals[-1] - vals[0] if len(vals) >= 2 else 0
            pack_ranges[pack_name] = max(vals) - min(vals)

    fig = make_pack_chart(
        contracts=contracts, current_y=current_y, pack_groups=pack_groups,
        pack_means=pack_means, pack_colors=PACK_COLORS,
        y_title=y_title, height=500, section_shading=(front_end, mid_end),
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)

    return {"mode": "Pack", "asof": asof, "contracts": contracts,
            "current_y": current_y, "pack_groups": pack_groups,
            "pack_means": pack_means, "pack_slopes": pack_slopes,
            "pack_ranges": pack_ranges}


def _side_pack(side_data, front_end, mid_end) -> None:
    _side_curve_summary(side_data["asof"], side_data["contracts"],
                        side_data["current_y"], front_end, mid_end)
    pack_groups = side_data.get("pack_groups", []) or []
    pack_means = side_data.get("pack_means", {}) or {}
    pack_slopes = side_data.get("pack_slopes", {}) or {}
    pack_ranges = side_data.get("pack_ranges", {}) or {}
    if pack_groups:
        _side_section_header("Pack metrics")
        for i, (pack_name, syms) in enumerate(pack_groups):
            color = PACK_COLORS[i] if i < len(PACK_COLORS) else BLUE
            mean_v = pack_means.get(pack_name)
            slope_v = pack_slopes.get(pack_name)
            rng_v = pack_ranges.get(pack_name)
            if mean_v is None:
                continue
            st.markdown(
                f"<div style='border-left:3px solid {color}; padding:0.3rem 0.5rem; "
                f"margin-bottom:0.4rem; background:var(--bg-base); border-radius:0 4px 4px 0;'>"
                f"<div style='display:flex; justify-content:space-between; align-items:baseline;'>"
                f"<span style='color:{color}; font-weight:600; font-size:0.78rem;'>{pack_name}</span>"
                f"<span style='color:var(--text-dim); font-family:JetBrains Mono, monospace; "
                f"font-size:0.65rem;'>{', '.join(syms)}</span></div>"
                f"<div style='font-family:JetBrains Mono, monospace; font-size:0.72rem; "
                f"color:var(--text-muted); margin-top:2px;'>"
                f"mean <span style='color:var(--accent);'>{mean_v:.4f}</span> · "
                f"slope <span style='color:{GREEN if (slope_v or 0) > 0 else (RED if (slope_v or 0) < 0 else TEXT_BODY)};'>"
                f"{slope_v:+.4f}</span> · range <span style='color:var(--text-body);'>{rng_v:.4f}</span>"
                f"</div></div>", unsafe_allow_html=True)
        if len(pack_groups) >= 2:
            _side_section_header("Pack spreads")
            spread_rows = []
            for i in range(len(pack_groups) - 1):
                n1, n2 = pack_groups[i][0], pack_groups[i + 1][0]
                v1 = pack_means.get(n1); v2 = pack_means.get(n2)
                if v1 is not None and v2 is not None:
                    d = v2 - v1
                    spread_rows.append((f"{n2} − {n1}", f"{d:+.4f}", _delta_class(d), None))
            if spread_rows:
                _kv_block(spread_rows)


# =============================================================================
# MODE: VOLUME / OI Δ
# =============================================================================
def _chart_volume(scope_id, contracts, wide_close, wide_volume, available_dates,
                   allow_implied_rate, front_end, mid_end) -> dict:
    cc1, cc2, cc3 = st.columns([1.2, 1.2, 2])
    with cc1:
        asof = st.selectbox("As-of", options=available_dates[::-1], index=0,
                            key=f"{scope_id}_v_asof",
                            format_func=lambda d: d.strftime("%Y-%m-%d (%a)"))
    with cc2:
        compare_back = st.selectbox("Δ vs", ["T-1", "T-1W", "T-1M"], index=0,
                                     key=f"{scope_id}_v_back")
        offset_map = {"T-1": 1, "T-1W": 7, "T-1M": 30}
        compare_dt = _quick_compare_offset(available_dates, asof, offset_map[compare_back])
    with cc3:
        view_choice = st.segmented_control(
            "View", ["Volume Δ", "Volume Δ + Curve"], default="Volume Δ + Curve",
            key=f"{scope_id}_v_view", label_visibility="collapsed",
        )

    try:
        vol_today = wide_volume.loc[wide_volume.index.date == asof].iloc[0].tolist()
        vol_compare = wide_volume.loc[wide_volume.index.date == compare_dt].iloc[0].tolist()
        vol_delta = [(t - c) if (t is not None and c is not None
                                  and not pd.isna(t) and not pd.isna(c)) else None
                     for t, c in zip(vol_today, vol_compare)]
    except Exception:
        vol_delta = [None] * len(contracts)
    oi_delta = [None] * len(contracts)

    current_y = []
    if view_choice == "Volume Δ + Curve":
        try:
            current_y = wide_close.loc[wide_close.index.date == asof].iloc[0].tolist()
        except Exception:
            current_y = []
        if allow_implied_rate:
            y_mode = st.selectbox("Y axis", ["Price", "Implied rate"], key=f"{scope_id}_v_ymode")
            if y_mode == "Implied rate":
                current_y = [(100 - v) if v is not None and not pd.isna(v) else None for v in current_y]
                y_title = "Implied rate (%)"
            else:
                y_title = "Price"
        else:
            y_title = "Price"
        fig = make_curve_chart(
            contracts=contracts, current_y=current_y,
            current_label=asof.strftime("%Y-%m-%d"),
            y_title=y_title, height=380, value_decimals=4,
            section_shading=(front_end, mid_end),
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)

    fig_v = make_volume_delta_chart(contracts, vol_delta,
                                    label=f"Volume Δ vs {compare_back}", height=240)
    st.plotly_chart(fig_v, use_container_width=True, theme=None)

    return {"mode": "Volume", "asof": asof, "contracts": contracts,
            "current_y": current_y, "vol_delta": vol_delta, "oi_delta": oi_delta,
            "compare_back": compare_back}


def _side_volume(side_data, front_end, mid_end) -> None:
    contracts = side_data["contracts"]
    if side_data.get("current_y"):
        _side_curve_summary(side_data["asof"], contracts,
                            side_data["current_y"], front_end, mid_end)
    vol_delta = side_data.get("vol_delta", []) or []
    pairs = [(c, v) for c, v in zip(contracts, vol_delta)
             if v is not None and not pd.isna(v) and v != 0]
    if pairs:
        _side_section_header(f"Top Volume Δ vs {side_data.get('compare_back', 'T-1')}")
        pairs.sort(key=lambda x: -abs(x[1]))
        for c, v in pairs[:8]:
            st.markdown(
                f"<div style='font-family:JetBrains Mono, monospace; font-size:0.74rem; "
                f"display:flex; justify-content:space-between; padding:3px 0;'>"
                f"<span style='color:var(--text-body);'>{c}</span>"
                f"<span style='color:{_delta_color(v)};'>{v:+.0f}</span></div>",
                unsafe_allow_html=True)


# =============================================================================
# MODE: HEATMAP
# =============================================================================
def _chart_heatmap(scope_id, contracts, wide_close, available_dates,
                    allow_implied_rate, front_end, mid_end) -> dict:
    cc1, cc2, cc3, cc4 = st.columns([1.2, 1.2, 1.2, 1.2])
    with cc1:
        # Spreads/flies don't have an "Implied rate" interpretation — values are spread/fly prices
        value_options = ["Price", "Δ vs T-1 (bp)", "Δ vs T-1W (bp)",
                         "Δ vs T-1M (bp)", "Z-score", "Percentile rank"]
        if allow_implied_rate:
            value_options.insert(1, "Implied rate")
        default_idx = (2 if allow_implied_rate else 1)  # default to "Δ vs T-1 (bp)" in both
        value_mode = st.selectbox(
            "Value", value_options, index=default_idx, key=f"{scope_id}_hm_value",
        )
    with cc2:
        time_window = st.selectbox(
            "Time window", ["Last 30d", "Last 60d", "Last 6mo", "Last 1y", "All"],
            index=2, key=f"{scope_id}_hm_window",
        )
    with cc3:
        z_lookback_label = st.selectbox(
            "Z/rank lookback",
            ["5d", "15d", "30d", "60d", "90d", "252d"],
            index=3, key=f"{scope_id}_hm_zlb",
            help="Used only for Z-score / Percentile rank value modes.",
        )
        z_lookback = {"5d": 5, "15d": 15, "30d": 30, "60d": 60,
                      "90d": 90, "252d": 252}[z_lookback_label]
    with cc4:
        show_values = st.checkbox("Show values", value=False, key=f"{scope_id}_hm_show")

    window_map = {"Last 30d": 30, "Last 60d": 60, "Last 6mo": 180,
                  "Last 1y": 365, "All": len(available_dates)}
    n_days = window_map[time_window]
    panel = wide_close.tail(n_days)

    # Build matrix according to value_mode
    if value_mode == "Price":
        matrix = panel.copy()
        colorscale = "Viridis"; zmid = None
    elif value_mode == "Implied rate":
        matrix = 100 - panel
        colorscale = "Viridis"; zmid = None
    elif value_mode.startswith("Δ vs"):
        offset_map = {"Δ vs T-1 (bp)": 1, "Δ vs T-1W (bp)": 5, "Δ vs T-1M (bp)": 21}
        offset = offset_map[value_mode]
        # Per-contract bp multiplier — outrights ×100, already-bp spreads/flies ×1
        from lib.contract_units import bp_multipliers_for, load_catalog
        _cat = load_catalog()
        _mults = (bp_multipliers_for(list(panel.columns), _BP, _cat)
                  if not _cat.empty else [100.0] * len(panel.columns))
        _mult_row = pd.Series(_mults, index=panel.columns)
        matrix = (panel - panel.shift(offset)).multiply(_mult_row, axis=1)
        matrix = matrix.dropna(how="all")
        colorscale = "RdBu_r"; zmid = 0
    elif value_mode == "Z-score":
        matrix = pd.DataFrame(index=panel.index, columns=panel.columns, dtype=float)
        for i, dt in enumerate(panel.index):
            asof_d = dt.date()
            zs = compute_per_contract_zscores(wide_close, asof_d, lookback=z_lookback)
            for c in panel.columns:
                matrix.at[dt, c] = zs.get(c)
        colorscale = "RdBu_r"; zmid = 0
    elif value_mode == "Percentile rank":
        matrix = pd.DataFrame(index=panel.index, columns=panel.columns, dtype=float)
        for i, dt in enumerate(panel.index):
            asof_d = dt.date()
            ps = compute_percentile_rank(wide_close, asof_d, lookback=z_lookback)
            for c in panel.columns:
                matrix.at[dt, c] = ps.get(c)
        colorscale = "RdBu_r"; zmid = 50
    else:
        matrix = panel
        colorscale = "Viridis"; zmid = None

    fig = make_heatmap_chart(
        matrix=matrix,
        title="",
        height=max(400, min(800, len(matrix) * 6 + 200)),
        colorscale=colorscale, zmid=zmid,
        show_values=show_values,
        value_format=".2f" if value_mode.startswith(("Δ", "Z", "P")) else ".4f",
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)

    try:
        latest_y = wide_close.iloc[-1].tolist()
    except Exception:
        latest_y = []
    return {"mode": "Heatmap", "asof": available_dates[-1], "contracts": contracts,
            "current_y": latest_y, "matrix": matrix, "value_mode": value_mode}


def _side_heatmap(side_data, front_end, mid_end) -> None:
    _side_curve_summary(side_data["asof"], side_data["contracts"],
                        side_data["current_y"], front_end, mid_end)
    matrix = side_data.get("matrix")
    vm = side_data.get("value_mode", "")
    if matrix is None or matrix.empty:
        return

    # Latest row stats
    _side_section_header(f"Latest row · {vm}")
    last_row = matrix.iloc[-1]
    valid = last_row.dropna()
    if not valid.empty:
        _kv_block([
            ("Min", f"{valid.min():+.3f}", "red", f"{valid.idxmin()}"),
            ("Max", f"{valid.max():+.3f}", "green", f"{valid.idxmax()}"),
            ("Mean", f"{valid.mean():+.3f}", "accent", None),
            ("Std", f"{valid.std(ddof=1):.3f}", None, None),
        ])

    # Top movers (most extreme contracts in the latest row)
    if vm.startswith(("Δ", "Z")):
        _side_section_header("Most extreme today")
        sortable = sorted(valid.items(), key=lambda x: -abs(x[1]))
        for c, v in sortable[:6]:
            st.markdown(
                f"<div style='font-family:JetBrains Mono, monospace; font-size:0.74rem; "
                f"display:flex; justify-content:space-between; padding:3px 0;'>"
                f"<span style='color:var(--text-body);'>{c}</span>"
                f"<span style='color:{_delta_color(v) if vm.startswith('Δ') else _z_color(v)};'>"
                f"{v:+.2f}</span></div>",
                unsafe_allow_html=True)


# =============================================================================
# MODE: MATRIX (Spreads / Flies cross-tenor)
# =============================================================================
def _chart_matrix(scope_id, all_tenor_breakdown, strategy, wide_close_all,
                   asof_dates_all, allow_implied_rate, front_end, mid_end,
                   wide_close_full=None) -> dict:
    """Render a 2D matrix: rows = front leg, cols = tenor, values = chosen mode."""
    cc1, cc2, cc3 = st.columns([1.2, 1.5, 1.2])
    with cc1:
        # Use latest available date as default
        asof = asof_dates_all[-1] if asof_dates_all else None
        st.markdown(
            f"<div style='padding-top:1.7rem;'><span style='color:var(--text-dim); "
            f"font-size:0.78rem;'>As of </span>"
            f"<span style='color:var(--accent); font-family:JetBrains Mono, monospace;'>"
            f"{asof.strftime('%Y-%m-%d')}</span></div>",
            unsafe_allow_html=True,
        ) if asof else None
    with cc2:
        value_mode = st.selectbox(
            "Value mode", ["Current value", "Z-score", "Percentile rank"],
            key=f"{scope_id}_mx_vmode",
        )
    with cc3:
        lookback_label = st.selectbox(
            "Lookback (z / rank)",
            ["5d", "15d", "30d", "60d", "90d", "252d"],
            index=3, key=f"{scope_id}_mx_lb",
        )
        lookback = {"5d": 5, "15d": 15, "30d": 30, "60d": 60,
                    "90d": 90, "252d": 252}[lookback_label]

    # Build the union symbols_df across all tenors (so rows/cols span everything)
    all_syms = pd.concat([
        symbols_df.assign(tenor=tenor) for tenor, _, _, symbols_df in all_tenor_breakdown
    ], ignore_index=True) if all_tenor_breakdown else pd.DataFrame()

    if all_syms.empty or wide_close_full is None or wide_close_full.empty:
        st.info("No matrix data available.")
        return {"mode": "Matrix"}

    vmode_arg = {"Current value": "value",
                 "Z-score": "zscore",
                 "Percentile rank": "rank"}[value_mode]
    matrix = compute_pairwise_spread_matrix(
        all_syms, wide_close_full, asof,
        value_mode=vmode_arg, lookback=lookback,
    )
    if matrix.empty:
        st.info("Matrix is empty.")
        return {"mode": "Matrix"}

    rows_label = list(matrix.index)
    cols_label = [f"{int(c)}M" for c in matrix.columns]
    z = matrix.values.tolist()

    if value_mode == "Current value":
        cs = "Viridis"; zm = None; vfmt = ".4f"
    elif value_mode == "Z-score":
        cs = "RdBu_r"; zm = 0; vfmt = ".2f"
    else:  # rank
        cs = "RdBu_r"; zm = 50; vfmt = ".0f"

    title = f"{strategy.capitalize()}s × Tenor — {value_mode} ({lookback_label} lookback)" \
            if value_mode != "Current value" else f"{strategy.capitalize()}s × Tenor — Current"
    fig = make_calendar_matrix_chart(
        rows_label=rows_label, cols_label=cols_label, z=z,
        title=title, height=max(400, min(700, len(rows_label) * 22 + 120)),
        colorscale=cs, zmid=zm, value_format=vfmt, show_values=True,
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)

    return {"mode": "Matrix", "asof": asof, "matrix": matrix, "value_mode": value_mode,
            "lookback_label": lookback_label}


def _side_matrix(side_data, front_end, mid_end) -> None:
    matrix = side_data.get("matrix")
    if matrix is None or matrix.empty:
        return
    vm = side_data.get("value_mode", "")
    lb = side_data.get("lookback_label", "")
    _side_section_header(f"Matrix overview ({vm} · {lb})")

    flat = matrix.stack(dropna=True)
    if flat.empty:
        return

    _kv_block([
        ("Min", f"{flat.min():+.3f}", "red", f"{flat.idxmin()}"),
        ("Max", f"{flat.max():+.3f}", "green", f"{flat.idxmax()}"),
        ("Mean", f"{flat.mean():+.3f}", "accent", None),
        ("Std", f"{flat.std(ddof=1):.3f}", None, None),
    ])

    _side_section_header("Most extreme cells")
    sortable = flat.copy()
    if vm == "Z-score":
        sortable = sortable.abs().sort_values(ascending=False)
    else:
        sortable = sortable.sort_values(ascending=False)
    for (row_lbl, tenor), v in sortable.head(8).items():
        actual_v = flat.loc[(row_lbl, tenor)]
        col_class = _z_color(actual_v) if vm == "Z-score" else (
                    _delta_color(actual_v) if vm == "Current value" else TEXT_BODY)
        st.markdown(
            f"<div style='font-family:JetBrains Mono, monospace; font-size:0.74rem; "
            f"display:flex; justify-content:space-between; padding:3px 0;'>"
            f"<span style='color:var(--text-body);'>{row_lbl} · {int(tenor)}M</span>"
            f"<span style='color:{col_class};'>{actual_v:+.3f}</span></div>",
            unsafe_allow_html=True)


# =============================================================================
# Main scope
# =============================================================================
def _curve_scope(scope_id, symbols_df, strategy, tenor_months,
                  allow_implied_rate, front_end, mid_end,
                  matrix_breakdown=None, wide_close_full_for_matrix=None) -> None:
    if symbols_df.empty:
        st.info(f"No live {strategy} contracts.")
        return

    contracts = symbols_df["symbol"].tolist()
    latest = get_sra_snapshot_latest_date()
    if latest is None:
        st.error(f"No {_BP} data available.")
        return

    history_max_days = 365
    start_date = latest - timedelta(days=history_max_days)
    panel = load_sra_curve_panel(strategy, tenor_months, start_date, latest)
    if panel.empty:
        st.warning("No data in window.")
        return

    wide_close = pivot_curve_panel(panel, contracts, "close")
    wide_high = pivot_curve_panel(panel, contracts, "high")
    wide_low = pivot_curve_panel(panel, contracts, "low")
    wide_volume = pivot_curve_panel(panel, contracts, "volume")
    available_dates = list(wide_close.index.date)
    if not available_dates:
        st.warning("No bars available.")
        return

    ref_panel = load_reference_rate_panel(start_date, latest)

    # ---- Top status strip ----
    status_strip_with_dot([
        ("Contracts", str(len(contracts)), "accent"),
        ("Range", contract_range_str(symbols_df), "accent"),
        ("First", available_dates[0].strftime("%Y-%m-%d"), None),
        ("Latest", available_dates[-1].strftime("%Y-%m-%d"), None),
        ("Bars", str(len(available_dates)), None),
    ])

    # ---- Regime classifier badges (multi-lookback) ----
    _render_regime_badge(wide_close, available_dates[-1], contracts,
                          front_end, mid_end, lookbacks=(1, 5, 30))

    # ---- Mode picker + global toggles ----
    mode_options = MODE_OUTRIGHTS if strategy == "outright" else MODE_SPREADS_FLIES
    head_l, head_r = st.columns([5, 2])
    with head_l:
        mode = st.segmented_control(
            "Visualization mode", options=mode_options, default=mode_options[0],
            key=f"{scope_id}_mode", label_visibility="collapsed",
        )
        if mode is None:
            mode = mode_options[0]
    with head_r:
        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            compare_on = st.toggle("Compare", key=f"{scope_id}_cmp_on")
        with bc2:
            animate_on = st.toggle("Animate", key=f"{scope_id}_anim_on")
        with bc3:
            side_open = st.toggle("Analysis", key=f"{scope_id}_side_on", value=True,
                                   help="Show / hide the analysis side panel.")

    # ---- Layout: chart vs chart+side ----
    if side_open:
        chart_col, side_col = st.columns([3.2, 1])
    else:
        chart_col = st.container()
        side_col = None

    side_data = {}
    allow_carry = (strategy == "outright")
    allow_fomc = (strategy == "outright")

    with chart_col:
        try:
            if mode == "Standard":
                side_data = _chart_standard(
                    scope_id, contracts, wide_close, wide_high, wide_low,
                    available_dates, ref_panel, allow_implied_rate, allow_carry, allow_fomc,
                    compare_on, animate_on, front_end, mid_end,
                )
            elif mode == "Multi-date":
                side_data = _chart_multi_date(scope_id, contracts, wide_close, available_dates,
                                               allow_implied_rate, front_end, mid_end)
            elif mode == "Ribbon":
                side_data = _chart_ribbon(scope_id, contracts, wide_close, available_dates,
                                           allow_implied_rate, front_end, mid_end)
            elif mode == "Pack":
                side_data = _chart_pack(scope_id, contracts, symbols_df, wide_close,
                                         available_dates, allow_implied_rate, front_end, mid_end)
            elif mode == "Volume Δ":
                side_data = _chart_volume(scope_id, contracts, wide_close, wide_volume,
                                           available_dates, allow_implied_rate, front_end, mid_end)
            elif mode == "Heatmap":
                side_data = _chart_heatmap(scope_id, contracts, wide_close, available_dates,
                                            allow_implied_rate, front_end, mid_end)
            elif mode == "Matrix":
                side_data = _chart_matrix(scope_id, matrix_breakdown, strategy,
                                           wide_close_full_for_matrix, available_dates,
                                           allow_implied_rate, front_end, mid_end,
                                           wide_close_full=wide_close_full_for_matrix)
        except Exception as e:
            st.error(f"⚠️ {mode} mode failed to render: `{type(e).__name__}: {e}`")
            st.caption("Try a different mode or reset toggles. The rest of the page will still work.")

    # ---- Side panel — UNIFIED: renders ALL analytical blocks regardless of mode ----
    if side_col is not None and side_data:
        with side_col:
            st.markdown(
                "<div style='font-size:0.78rem; color:var(--accent); font-weight:600; "
                "letter-spacing:0.04em; text-transform:uppercase; padding-bottom:0.4rem; "
                "border-bottom:1px solid var(--border-default); margin-bottom:0.5rem;'>"
                "📊 Analysis · all blocks (scroll)</div>",
                unsafe_allow_html=True,
            )
            with st.container(height=720, border=False):
                try:
                    asof = side_data.get("asof", available_dates[-1])
                    # current_y_disp: in current mode's display space
                    current_y_disp = side_data.get("current_y", []) or []
                    # current_y_price: always in price space (for carry / FOMC math)
                    current_y_price = side_data.get("current_y_price")
                    if current_y_price is None:
                        # If chart didn't supply price-space (non-Standard modes), compute it
                        try:
                            current_y_price = (
                                wide_close.loc[wide_close.index.date == asof].iloc[0].tolist()
                            )
                        except Exception:
                            current_y_price = current_y_disp

                    # Auto-compute decomposition vs T-1 if not already provided
                    decomp = side_data.get("decomp")
                    changes_bp = side_data.get("changes_bp")
                    compare_label = side_data.get("compare_label")
                    if not decomp:
                        if asof in available_dates:
                            idx = available_dates.index(asof)
                            if idx > 0:
                                prior = available_dates[idx - 1]
                                try:
                                    prior_close = (
                                        wide_close.loc[wide_close.index.date == prior]
                                        .iloc[0].tolist()
                                    )
                                    changes_bp = compute_curve_change(
                                        current_y_price, prior_close, in_bp_units=True,
                                        contracts=contracts,
                                    )
                                    decomp = compute_decomposition(changes_bp)
                                    compare_label = prior.strftime("%Y-%m-%d (T-1)")
                                except Exception:
                                    pass

                    ref_rates = side_data.get("ref_rates") or get_reference_rates_at(asof, ref_panel)

                    _render_combined_side_panel(
                        asof=asof,
                        contracts=contracts,
                        current_y_disp=current_y_disp,
                        current_y_price=current_y_price,
                        ref_rates=ref_rates,
                        wide_close=wide_close,
                        wide_volume=wide_volume,
                        front_end=front_end,
                        mid_end=mid_end,
                        strategy=strategy,
                        allow_implied_rate=allow_implied_rate,
                        available_dates=available_dates,
                        symbols_df=symbols_df,
                        decomp=decomp,
                        changes_bp=changes_bp,
                        compare_label=compare_label,
                    )
                except Exception as e:
                    st.warning(f"Side panel error: `{type(e).__name__}: {e}`")

    # ---- Drill-down panel (always available) ----
    try:
        _render_drill_down(scope_id, contracts, available_dates[-1], wide_close,
                            panel_history_days=history_max_days)
    except Exception as e:
        st.caption(f"Drill-down unavailable: {e}")


# =============================================================================
# Public render
# =============================================================================
def _build_matrix_breakdown(strategy: str) -> tuple:
    """For Spreads/Flies Matrix mode — pull all tenors and combine into one wide_close.

    Returns (tenor_breakdown_with_df, combined_wide_close_df).
    """
    breakdown_with_df = []
    combined_panels = []
    latest = get_sra_snapshot_latest_date()
    if latest is None:
        return [], None
    start = latest - timedelta(days=365)
    for t, c, rng in tenor_breakdown(strategy):
        df_syms = (get_spreads(t) if strategy == "spread" else get_flies(t))
        if df_syms.empty:
            continue
        breakdown_with_df.append((t, c, rng, df_syms))
        panel = load_sra_curve_panel(strategy, t, start, latest)
        if not panel.empty:
            wide = pivot_curve_panel(panel, df_syms["symbol"].tolist(), "close")
            combined_panels.append(wide)
    if not combined_panels:
        return breakdown_with_df, None
    combined = pd.concat(combined_panels, axis=1)
    return breakdown_with_df, combined


def render(base_product: str = "SRA") -> None:
    _set_market(base_product)
    latest = get_sra_snapshot_latest_date()
    outrights_df = get_outrights()
    spread_breakdown = tenor_breakdown("spread")
    fly_breakdown = tenor_breakdown("fly")
    n_spreads = sum(c for _, c, _ in spread_breakdown)
    n_flies = sum(c for _, c, _ in fly_breakdown)
    spread_summary = " · ".join(f"{t}M:{c}" for t, c, _ in spread_breakdown)
    fly_summary = " · ".join(f"{t}M:{c}" for t, c, _ in fly_breakdown)

    status_strip([
        ("Snapshot", latest.strftime("%Y-%m-%d (%a)") if latest else "—", "accent"),
        ("Outrights", f"{len(outrights_df)}  {contract_range_str(outrights_df)}", None),
        ("Spreads", f"{n_spreads} [{spread_summary}]", None),
        ("Flies", f"{n_flies} [{fly_summary}]", None),
        ("Liveness", f"≤ {LIVENESS_DAYS}d", None),
    ])

    with st.expander("Curve section boundaries", expanded=False):
        sb1, sb2, sb3 = st.columns(3)
        with sb1:
            front_end = st.number_input("Front ends at #", min_value=1, max_value=30,
                                         value=DEFAULT_FRONT_END, step=1,
                                         key="sra_curve_front_end")
        with sb2:
            mid_end = st.number_input("Mid ends at #", min_value=2, max_value=40,
                                       value=DEFAULT_MID_END, step=1,
                                       key="sra_curve_mid_end")
        with sb3:
            st.caption(
                f"Defaults: Front 1–{DEFAULT_FRONT_END} · "
                f"Mid {DEFAULT_FRONT_END+1}–{DEFAULT_MID_END} · "
                f"Back {DEFAULT_MID_END+1}+"
            )

    st.markdown("")
    section_tabs = st.tabs(["Outrights", "Spreads", "Flies"])

    with section_tabs[0]:
        _curve_scope("outrights", outrights_df, "outright", None,
                      allow_implied_rate=True, front_end=front_end, mid_end=mid_end)

    with section_tabs[1]:
        if not spread_breakdown:
            st.info("No live spreads.")
        else:
            tenor_labels = [f"{t}M ({c})" for t, c, _ in spread_breakdown]
            tenor_values = [t for t, _, _ in spread_breakdown]
            picked = st.segmented_control(
                "Tenor", options=tenor_labels,
                default=tenor_labels[1] if len(tenor_labels) > 1 else tenor_labels[0],
                key="sra_curve_spread_tenor", label_visibility="collapsed",
            )
            if picked is None:
                picked = tenor_labels[0]
            tenor = tenor_values[tenor_labels.index(picked)]
            df = get_spreads(tenor)
            # Pre-compute matrix data (all tenors) for the Matrix mode
            mb, combined_wide = _build_matrix_breakdown("spread")
            _curve_scope(f"spread_{tenor}", df, "spread", tenor,
                          allow_implied_rate=False, front_end=front_end, mid_end=mid_end,
                          matrix_breakdown=mb, wide_close_full_for_matrix=combined_wide)

    with section_tabs[2]:
        if not fly_breakdown:
            st.info("No live flies.")
        else:
            tenor_labels = [f"{t}M ({c})" for t, c, _ in fly_breakdown]
            tenor_values = [t for t, _, _ in fly_breakdown]
            picked = st.segmented_control(
                "Tenor", options=tenor_labels, default=tenor_labels[0],
                key="sra_curve_fly_tenor", label_visibility="collapsed",
            )
            if picked is None:
                picked = tenor_labels[0]
            tenor = tenor_values[tenor_labels.index(picked)]
            df = get_flies(tenor)
            mb, combined_wide = _build_matrix_breakdown("fly")
            _curve_scope(f"fly_{tenor}", df, "fly", tenor,
                          allow_implied_rate=False, front_end=front_end, mid_end=mid_end,
                          matrix_breakdown=mb, wide_close_full_for_matrix=combined_wide)
