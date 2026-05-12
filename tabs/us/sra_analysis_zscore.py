"""SRA → Analysis → Z-score & Mean Reversion subtab.

UI mirrors the Proximity subtab but built around stretch-and-reversion math:

  · Section ribbons — top-K most stretched HIGH / LOW per section (by Z)
  · Z confluence matrix — multi-lookback Z heatmap with 8-pattern column
  · Cluster heatmap — median |Z| per (Section × Lookback)
  · Composite reversion-candidate ranking
  · Composite trend-confirmation ranking
  · Universe Z-distribution histogram
  · Drill-down expander — close + mean ± σ bands + OU half-life annotation

Side panel — 12 analytics blocks:
  1.  Universe stretch summary (counts, skew, kurtosis)
  2.  Multi-window Z confluence (8-pattern catalog)
  3.  Section / tenor stretch (median Z per cell)
  4.  Half-life ranking (top fastest reverters)
  5.  Hurst regime summary (TRENDING / RANDOM / REVERTING counts)
  6.  ADF + Z gate (stationary + |z|≥2 = strongest reversion)
  7.  Composite reversion candidates (top 5)
  8.  Trend confirmations (anti-reversion list)
  9.  Velocity / pace of reversion
  10. Distribution + tail mini-histogram (rendered above chart column)
  11. Cross-check vs proximity (UNTESTED / STRETCHED / COILED / NORMAL)
  12. Methodology / thresholds tooltip
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
    make_distribution_histogram,
    make_mean_bands_chart,
    make_proximity_ribbon_chart,    # also used here as a generic horiz-bar with ATR-flag colors
    make_score_bar_chart,
)
from lib.components import status_strip_with_dot
from lib.mean_reversion import (
    Z_ELEVATED, Z_FRESH,
    adf_interpretation,
    classify_z_pattern,
    cluster_signal_z,
    compute_zscore_panel,
    get_interpretation_guide,
    get_z_pattern_descriptions,
    half_life_interpretation,
    hurst_interpretation,
    metric_tooltip,
    overall_setup_interpretation,
    pct_rank_interpretation,
    rank_by_score,
    rank_by_z,
    reversion_score_interpretation,
    section_regime_z,
    trend_score_interpretation,
    velocity_to_mean_interpretation,
    z_interpretation,
)
from lib.proximity import compute_proximity_panel, ATR_PERIOD
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
)
from lib import market_data as _md
from lib.markets import get_market as _get_market


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


_LOOKBACK_OPTIONS = [5, 15, 30, 60, 90]
_SCOPE_PREFIX = "zscore_mr"


# =============================================================================
# Helpers — formatting
# =============================================================================
def _z_color(z) -> str:
    if z is None or pd.isna(z):
        return "var(--text-dim)"
    az = abs(z)
    if az >= 2:
        return "var(--red)" if z > 0 else "var(--green)"
    if az >= 1:
        return "var(--amber)"
    return "var(--text-body)"


def _z_pill(z) -> str:
    if z is None or pd.isna(z):
        return ("<span style='display:inline-block; padding:1px 6px; "
                "border:1px solid var(--text-dim); color:var(--text-dim); "
                "font-size:0.65rem; font-family:JetBrains Mono, monospace; "
                "font-weight:600; border-radius:4px; line-height:1;'>—</span>")
    color = _z_color(z)
    return (f"<span style='display:inline-block; padding:1px 6px; "
            f"border:1px solid {color}; color:{color}; "
            f"font-size:0.65rem; font-family:JetBrains Mono, monospace; "
            f"font-weight:600; border-radius:4px; line-height:1;'>{z:+.2f}σ</span>")


def _pattern_pill(pat: str) -> str:
    color_map = {
        "PERSISTENT": "var(--red)", "ACCELERATING": "var(--red)",
        "DECELERATING": "var(--amber)", "FRESH": "var(--blue)",
        "DRIFTED": "var(--amber)", "REVERTING": "var(--green)",
        "STABLE": "var(--text-dim)",
        "MIXED": "var(--text-muted)",
    }
    color = color_map.get(pat, "var(--text-body)")
    return (f"<span style='display:inline-block; padding:1px 6px; "
            f"border:1px solid {color}; color:{color}; "
            f"font-size:0.65rem; font-family:JetBrains Mono, monospace; "
            f"font-weight:600; border-radius:4px; line-height:1;'>{pat}</span>")


def _hurst_pill(h, label):
    if h is None or pd.isna(h):
        return ("<span style='display:inline-block; padding:1px 6px; "
                "border:1px solid var(--text-dim); color:var(--text-dim); "
                "font-size:0.65rem; font-family:JetBrains Mono, monospace; "
                "font-weight:600; border-radius:4px; line-height:1;'>—</span>")
    color = ("var(--red)" if label == "TRENDING"
             else "var(--green)" if label == "REVERTING"
             else "var(--text-muted)")
    return (f"<span style='display:inline-block; padding:1px 6px; "
            f"border:1px solid {color}; color:{color}; "
            f"font-size:0.65rem; font-family:JetBrains Mono, monospace; "
            f"font-weight:600; border-radius:4px; line-height:1;'>{label} {h:.2f}</span>")


def _interp_label(text: str, color: str = "var(--text-muted)") -> str:
    """Render a small italic interpretation tag below or beside a numeric value."""
    return (f"<span style='display:inline-block; font-size:0.62rem; "
            f"color:{color}; font-family:Inter, sans-serif; font-style:italic; "
            f"line-height:1.1;'>{text}</span>")


def _value_with_interp(value_html: str, interp_text: str, interp_color: str,
                        align: str = "right") -> str:
    """Two-line cell: numeric value on top, italic interpretation underneath.

    Used in the section-ribbon and composite-score tables so every metric is
    paired with plain-English meaning, not just a number.
    """
    return (f"<div style='text-align:{align}; line-height:1.15;'>"
            f"{value_html}"
            f"<div>{_interp_label(interp_text, interp_color)}</div>"
            f"</div>")


def _take_pill(label: str, color: str) -> str:
    """Larger 'TAKE' pill — the trader-action recommendation per row."""
    return (f"<span style='display:inline-block; padding:2px 8px; "
            f"background:{color}; color:var(--bg-base); "
            f"font-size:0.65rem; font-family:JetBrains Mono, monospace; "
            f"font-weight:700; border-radius:4px; line-height:1.2; "
            f"letter-spacing:0.02em;'>{label}</span>")


def _take_pill_outline(label: str, color: str) -> str:
    """Outline variant of TAKE pill for less aggressive emphasis."""
    return (f"<span style='display:inline-block; padding:2px 8px; "
            f"border:1px solid {color}; color:{color}; "
            f"font-size:0.65rem; font-family:JetBrains Mono, monospace; "
            f"font-weight:700; border-radius:4px; line-height:1.2;'>{label}</span>")


def _render_interpretation_guide() -> None:
    """Top-of-subtab expander: explains every metric and its bucket boundaries."""
    guide = get_interpretation_guide()
    with st.expander("📖 Interpretation guide — click to see what every metric means",
                      expanded=False):
        st.markdown(
            "<div style='color:var(--text-muted); font-size:0.78rem; margin-bottom:0.5rem;'>"
            "Every metric on this subtab is paired with a plain-English label so you "
            "don't have to memorise the number ranges. This guide lists each metric, "
            "its formula, and the bucket boundaries used."
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


def _tooltip_attr(text: str) -> str:
    return (str(text)
            .replace('"', '&quot;')
            .replace("'", "&#39;")
            .replace("\n", "&#10;"))


def _kv_block(rows: list) -> None:
    """Tuples: (label, value, color_class, sub_text, tooltip_text).
    Last 3 elements are optional. When a tooltip_text is provided the entire row
    becomes hoverable with a multi-line title-attribute tooltip.
    """
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


# =============================================================================
# Metric-tooltip helper — wraps mean_reversion.metric_tooltip with the local guide
# =============================================================================
def _mt(metric_key: str, current_label: Optional[str] = None,
         computed: Optional[str] = None) -> str:
    """Shorthand for building a per-metric tooltip from the Z guide."""
    return metric_tooltip(get_interpretation_guide(), metric_key,
                           current_label=current_label, computed=computed)


def _reading(text: str) -> None:
    """Render a small italic 'Reading:' paragraph beneath a side-panel block."""
    st.markdown(
        f"<div style='padding:4px 6px; margin-top:4px; "
        f"background:rgba(232,183,93,0.05); border-left:2px solid var(--accent-dim); "
        f"font-size:0.7rem; color:var(--text-body); font-style:italic; line-height:1.4;'>"
        f"<b style='color:var(--accent); font-style:normal;'>Reading:</b> {text}"
        f"</div>",
        unsafe_allow_html=True,
    )


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


def _methodology_text() -> str:
    return (
        "MEAN-REVERSION METHODOLOGY (constant across all blocks)\n"
        "\n"
        f"Z-score:        z = (today − μ_N) / σ_N    where μ/σ from prior {','.join(str(n) for n in _LOOKBACK_OPTIONS)} bars\n"
        "                  history EXCLUDES today;  σ uses ddof=1\n"
        "Percentile rank: empirical-CDF rank of today vs prior N bars (0..100)\n"
        "\n"
        "OU half-life:   AR(1) fit  x_t = a + b·x_{t−1};  half_life = −ln(2)/ln(b)  for 0<b<1\n"
        "                  in trading days; estimated on the prior 90 bars; None if not reverting.\n"
        "\n"
        "Hurst (R/S):    log-log regression of rescaled-range R/S vs window size,\n"
        "                  on log-returns over the prior 90 bars. Range clipped to [0, 1].\n"
        "                  H>0.55 = TRENDING · H<0.45 = REVERTING · else RANDOM\n"
        "\n"
        "ADF (lag-1):    t-stat from  Δx_t = α + γ·x_{t−1} + φ·Δx_{t−1} + ε\n"
        "                  rejects H0 (unit root) when t ≤ −2.86  (5% critical, no trend)\n"
        f"\n"
        f"Stretch flags:  |z| ≥ {Z_ELEVATED:.1f} = elevated · |z| ≥ {Z_FRESH:.1f} = fresh\n"
        "Pattern catalogue:  PERSISTENT / ACCELERATING / DECELERATING / FRESH /\n"
        "                    DRIFTED / REVERTING / STABLE / MIXED\n"
        "Composite scores:\n"
        "  reversion = 0.5·|z|/2 + 0.25·(1−H) + 0.15·10/max(1,half-life) + 0.10·1{ADF rejects}\n"
        "  trend     = 0.55·|z|/2 + 0.35·H        + 0.10·1{ADF can't reject}\n"
        "BP scaling for spreads/flies via lib.contract_units (no double-×100)."
    )


# =============================================================================
# Cached engine call — same key as proximity subtab's so we hit the same cache.
# =============================================================================
@st.cache_data(show_spinner=False, ttl=600)
def _build_zscore_data(strategy: str, tenor_months: Optional[int],
                        asof_str: str, lookbacks_tuple: tuple,
                        history_days: int = 252,
                        base_product: str = "SRA") -> dict:
    """Cache key includes base_product so each market gets its own panel.

    Previously this function hardcoded base_product="SRA" inside the
    compute_zscore_panel / compute_proximity_panel calls AND omitted
    base_product from the cache key — so once any market populated the cache,
    every other market saw SRA data. Now keyed per-market.
    """
    # Set the shimmed _BP so the local get_outrights/get_spreads/get_flies/
    # load_sra_curve_panel calls below dispatch to the correct market.
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
        "asof": asof_date, "strategy": strategy, "tenor_months": tenor_months,
        "base_product": base_product,
        "contracts": [], "wide_close": pd.DataFrame(), "wide_high": pd.DataFrame(),
        "wide_low": pd.DataFrame(), "zscore": {}, "proximity": {},
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
    out["zscore"] = compute_zscore_panel(
        wide_close, contracts, asof_date, lookbacks,
        base_product=base_product, long_lookback_for_tests=max(lookbacks),
    )
    out["proximity"] = compute_proximity_panel(
        wide_close, wide_high, wide_low, contracts, asof_date, lookbacks,
        base_product=base_product, atr_period=ATR_PERIOD,
    )
    return out


# =============================================================================
# Universe stretch summary
# =============================================================================
def _universe_stretch(panel: dict, contracts: list[str], lookback: int) -> dict:
    zs = []
    n_pos2 = n_neg2 = n_pos1 = n_neg1 = 0
    for sym in contracts:
        rec = panel.get("per_contract", {}).get(sym) or {}
        m = rec.get("by_lookback", {}).get(lookback) or {}
        z = m.get("z")
        if z is None or pd.isna(z):
            continue
        zs.append(z)
        if z >= 2: n_pos2 += 1
        elif z >= 1: n_pos1 += 1
        if z <= -2: n_neg2 += 1
        elif z <= -1: n_neg1 += 1
    if not zs:
        return {"n_pos2": 0, "n_neg2": 0, "n_pos1": 0, "n_neg1": 0,
                "mean_z": None, "median_z": None, "std_z": None,
                "skew": None, "kurt": None, "all_zs": []}
    arr = np.array(zs)
    mu = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else None
    med = float(np.median(arr))
    skew = None
    kurt = None
    if sd and sd > 0 and len(arr) >= 3:
        m3 = float(np.mean((arr - mu) ** 3))
        m4 = float(np.mean((arr - mu) ** 4))
        skew = m3 / (sd ** 3)
        kurt = m4 / (sd ** 4) - 3.0
    return {"n_pos2": n_pos2, "n_neg2": n_neg2, "n_pos1": n_pos1, "n_neg1": n_neg1,
            "mean_z": mu, "median_z": med, "std_z": sd,
            "skew": skew, "kurt": kurt, "all_zs": zs}


# =============================================================================
# View 1 — Section Ribbons (top-K most stretched HIGH/LOW per section)
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
            for side, side_color, side_title in [
                ("high", "var(--red)", f"Most stretched +Z ({lookback}d)"),
                ("low",  "var(--green)", f"Most stretched −Z ({lookback}d)"),
            ]:
                rows = rank_by_z(panel, sec_contracts, lookback, top_k=top_k, side=side)
                st.markdown(
                    f"<div style='font-size:0.7rem; color:var(--text-muted); "
                    f"padding:6px 8px 4px 8px; border-left:2px solid {side_color}; "
                    f"margin-top:6px;'>{side_title}</div>",
                    unsafe_allow_html=True,
                )
                if not rows:
                    st.caption("No data.")
                    continue
                # Reuse the proximity ribbon chart by mapping z→dist_atr (abs) + flag
                bar_rows = []
                for r in rows:
                    z = r.get("z") or 0
                    az = abs(z)
                    flag = ("AT" if az >= 2 else
                            "NEAR" if az >= 1.5 else
                            "APPROACHING" if az >= 1 else "FAR")
                    bar_rows.append({
                        "symbol": r["symbol"],
                        "dist_atr": az,
                        "dist_bp": r.get("dist_to_mean_bp"),
                        "flag": flag,
                        "streak": 0,
                    })
                fig = make_proximity_ribbon_chart(bar_rows, side=side, title="",
                                                   height=max(180, 32 * len(rows) + 60))
                st.plotly_chart(fig, use_container_width=True, theme=None)
                _render_z_table(rows, side, lookback)


def _render_z_table(rows: list[dict], side: str, lookback: int) -> None:
    if not rows:
        return
    grid_cols = "1.0fr 1.0fr 0.7fr 0.9fr 0.9fr 1.0fr 1.0fr 1.4fr"
    parts = ['<div style="font-family:JetBrains Mono, monospace; font-size:0.68rem; '
             'margin-top:4px;">']
    parts.append(
        f'<div style="display:grid; grid-template-columns: {grid_cols}; '
        'gap:4px; padding:4px 6px; color:var(--text-dim); '
        'border-bottom:1px solid var(--border-subtle);">'
        '<span>SYMBOL</span>'
        '<span style="text-align:right">Z + reading</span>'
        '<span style="text-align:right">μ-Δ (bp)</span>'
        '<span style="text-align:right">PCT + reading</span>'
        '<span style="text-align:center">PATTERN</span>'
        '<span style="text-align:center">HURST + reading</span>'
        '<span style="text-align:center">HALF-LIFE + reading</span>'
        '<span style="text-align:center">TAKE — what to do</span>'
        '</div>'
    )

    def _fmt_signed(v, decimals=2, pct=False):
        if v is None or pd.isna(v):
            return "—"
        if pct:
            return f"{v:.0f}%"
        return f"{v:+.{decimals}f}"

    for r in rows:
        sym = r.get("symbol", "")
        z = r.get("z")
        dtm = r.get("dist_to_mean_bp")
        pct = r.get("pct_rank")
        pattern = r.get("pattern", "MIXED")
        h = r.get("hurst")
        h_label = r.get("hurst_label", "—")
        hl = r.get("ou_half_life")
        adf_p = r.get("adf_pvalue") or "—"
        adf_reject = r.get("adf_reject_5pct", False)
        adf_t = r.get("adf_tstat")

        # Resolve interpretations
        z_label, z_color = z_interpretation(z)
        pct_label, pct_color = pct_rank_interpretation(pct)
        h_label_interp, h_color = hurst_interpretation(h)
        hl_label, hl_color = half_life_interpretation(hl)
        adf_label, adf_color = adf_interpretation(adf_reject, adf_p)
        take_label, take_color = overall_setup_interpretation(z, h, hl, adf_reject)

        # Tooltip with full numbers + plain-English meanings
        tt_lines = [f"{sym} · {lookback}d Z-score & MR diagnostics", ""]
        z_str = f"{z:+.2f}σ" if z is not None else "—"
        dtm_str = f"{dtm:+.2f}bp" if dtm is not None else "—"
        pct_str = f"{pct:.0f}%" if pct is not None else "—"
        tt_lines.append(f"Z         {z_str}  ({z_label})")
        tt_lines.append(f"μ-Δ       {dtm_str}")
        tt_lines.append(f"Pct       {pct_str}  ({pct_label})")
        tt_lines.append(f"Pattern   {pattern}")
        if h is not None:
            tt_lines.append(f"Hurst     {h:.3f}  ({h_label_interp})")
        if hl is not None and np.isfinite(hl):
            tt_lines.append(f"Half-life {hl:.1f}d  ({hl_label})")
        if adf_t is not None:
            tt_lines.append(f"ADF       t={adf_t:+.2f}, p {adf_p}  ({adf_label})")
        tt_lines.append("")
        tt_lines.append(f"TAKE      {take_label}")
        tt = "\n".join(tt_lines)

        # HALF-LIFE cell: value above, interpretation below
        if hl is not None and np.isfinite(hl):
            hl_value_html = f'<span style="color:{hl_color};">{hl:.1f}d</span>'
        else:
            hl_value_html = '<span style="color:var(--text-dim);">—</span>'

        # Per-metric tooltips (each shows formula + thresholds + computed value, with current bucket marked ✓)
        tt_z   = _mt("Z-score (z)", z_label,
                      f"{(z is not None and f'{z:+.3f}σ') or '—'}  "
                      f"(μ_{lookback} = {(mu := r.get('mean')) is not None and f'{mu:.4f}' or '—'},  "
                      f"σ_{lookback} = {(sd := r.get('std')) is not None and f'{sd:.4f}' or '—'})")
        tt_pct = _mt("Percentile rank", pct_label,
                      f"{(pct is not None and f'{pct:.1f}%') or '—'}  (vs prior {lookback} bars)")
        tt_h   = _mt("Hurst exponent (H)", h_label_interp,
                      f"H = {(h is not None and f'{h:.3f}') or '—'}  (R/S on prior 90 bars of log-returns)")
        tt_hl  = _mt("OU half-life", hl_label,
                      f"h = {(hl is not None and np.isfinite(hl) and f'{hl:.1f}d') or 'no fit'}  "
                      "(AR(1) fit on prior 90 bars)")
        tt_take = _mt("Overall TAKE", take_label,
                       f"|z|={(z is not None and f'{abs(z):.2f}') or '—'},  "
                       f"H={(h is not None and f'{h:.2f}') or '—'},  "
                       f"hl={(hl is not None and np.isfinite(hl) and f'{hl:.1f}d') or '—'},  "
                       f"ADF rejects 5%: {adf_reject}")
        tt_pat = _mt("Multi-window confluence pattern", pattern,
                      f"pattern across {lookback}d focus + selected lookbacks")

        # Wrap each numeric cell in its own title attribute so cell-hover wins over row-hover
        row_html = (
            f'<div title="{_tooltip_attr(tt)}" '
            f'style="display:grid; grid-template-columns: {grid_cols}; '
            f'gap:4px; padding:5px 6px; align-items:center; cursor:help; '
            f'border-bottom:1px solid var(--border-subtle);">'
            f'<span style="color:var(--accent);">{sym}</span>'
            f'<div title="{_tooltip_attr(tt_z)}" style="cursor:help;">'
            f'  {_value_with_interp(_z_pill(z), z_label, z_color, align="right")}'
            f'</div>'
            f'<span style="text-align:right;">{_fmt_signed(dtm)}</span>'
            f'<div title="{_tooltip_attr(tt_pct)}" style="cursor:help;">'
            f'  {_value_with_interp(_fmt_signed(pct, pct=True), pct_label, pct_color, align="right")}'
            f'</div>'
            f'<span title="{_tooltip_attr(tt_pat)}" style="text-align:center; cursor:help;">'
            f'  {_pattern_pill(pattern)}'
            f'</span>'
            f'<div title="{_tooltip_attr(tt_h)}" style="text-align:center; line-height:1.15; cursor:help;">'
            f'  {_hurst_pill(h, h_label)}'
            f'  <div>{_interp_label(h_label_interp, h_color)}</div>'
            f'</div>'
            f'<div title="{_tooltip_attr(tt_hl)}" style="text-align:center; line-height:1.15; cursor:help;">'
            f'  {hl_value_html}'
            f'  <div>{_interp_label(hl_label, hl_color)}</div>'
            f'</div>'
            f'<div title="{_tooltip_attr(tt_take)}" style="text-align:center; cursor:help;">'
            f'  {_take_pill_outline(take_label, take_color)}'
            f'</div>'
            f'</div>'
        )
        parts.append(row_html)
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


# =============================================================================
# View 2 — Confluence Z Matrix
# =============================================================================
def _render_z_confluence(panel: dict, contracts: list[str], lookbacks: list[int],
                          top_n: int = 25) -> None:
    rows = []
    for sym in contracts:
        rec = panel.get("per_contract", {}).get(sym) or {}
        zs = []
        for n in lookbacks:
            m = rec.get("by_lookback", {}).get(n) or {}
            z = m.get("z")
            if z is not None and not pd.isna(z):
                zs.append(abs(z))
        if not zs:
            continue
        rows.append((sym, max(zs), rec))
    rows.sort(key=lambda x: -x[1])
    rows = rows[:top_n]
    if not rows:
        st.caption("Nothing stretched.")
        return
    syms = [r[0] for r in rows]
    matrix = pd.DataFrame(index=syms, columns=lookbacks, dtype=float)
    pattern_col = pd.Series(index=syms, dtype=object)
    for sym, _, rec in rows:
        for n in lookbacks:
            m = rec.get("by_lookback", {}).get(n) or {}
            z = m.get("z")
            matrix.loc[sym, n] = z if z is not None else np.nan
        pattern_col.loc[sym] = rec.get("pattern", "MIXED")
    fig = make_confluence_matrix_chart(
        matrix, pattern_col=pattern_col,
        title=f"Z confluence — top {len(rows)} by max |z| (cells: z-score; reds = +z, blues = −z)",
        height=max(360, 22 * len(rows) + 80),
        z_mode=True,
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)


# =============================================================================
# View 3 — Cluster Heatmap (median |Z| per Section × Lookback)
# =============================================================================
def _render_cluster_heatmap(panel: dict, contracts: list[str], front_end: int,
                              mid_end: int, lookbacks: list[int]) -> None:
    fr_rng, mr_rng, br_rng = compute_section_split(len(contracts), front_end, mid_end)
    sec_for: dict[str, str] = {}
    for i, sym in enumerate(contracts):
        if i in fr_rng: sec_for[sym] = "Front"
        elif i in mr_rng: sec_for[sym] = "Mid"
        else: sec_for[sym] = "Back"
    rows_order = ["Front", "Mid", "Back"]
    abs_z_mat = pd.DataFrame(index=rows_order, columns=lookbacks, dtype=float)
    extreme_count_mat = pd.DataFrame(index=rows_order, columns=lookbacks, dtype=float)
    for n in lookbacks:
        cs = cluster_signal_z(panel, contracts, sec_for, n)
        for sec in rows_order:
            stats = cs.get(sec, {})
            tot = stats.get("n_total", 0)
            if tot == 0:
                abs_z_mat.loc[sec, n] = np.nan
                extreme_count_mat.loc[sec, n] = np.nan
                continue
            mz = stats.get("median_z")
            abs_z_mat.loc[sec, n] = abs(mz) if (mz is not None and not pd.isna(mz)) else np.nan
            extreme_count = (stats.get("n_pos_extreme", 0) + stats.get("n_neg_extreme", 0))
            extreme_count_mat.loc[sec, n] = extreme_count / tot
    cols = st.columns(2)
    with cols[0]:
        # Custom diverging based on |median z|, max 2 (rough cap)
        # Use existing density chart but rescale so 0..2 maps to 0..1 visually
        z_max = float(np.nanmax(abs_z_mat.values)) if abs_z_mat.size else 1.0
        z_max = max(1.0, z_max)
        norm_mat = (abs_z_mat / z_max).clip(0, 1)
        fig = make_density_heatmap_chart(
            norm_mat,
            title=f"|median z| per (Section × Lookback)  (full scale = {z_max:.2f}σ)",
            height=260, max_pct=1.0, show_values=True,
        )
        # Override the colorbar to show actual z-scale
        st.plotly_chart(fig, use_container_width=True, theme=None)
    with cols[1]:
        fig = make_density_heatmap_chart(
            extreme_count_mat,
            title="% of section with |z|≥2 per Lookback",
            height=260, max_pct=1.0, show_values=True,
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)


# =============================================================================
# View 4 — Composite scoring (reversion + trend candidates)
# =============================================================================
def _render_composite_rankings(panel: dict, contracts: list[str], top_k: int) -> None:
    rev = rank_by_score(panel, contracts, score_field="reversion_score", top_k=top_k)
    trend = rank_by_score(panel, contracts, score_field="trend_score", top_k=top_k)
    cols = st.columns(2)
    with cols[0]:
        st.markdown(
            "<div style='font-size:0.78rem; font-weight:600; color:var(--text-heading); "
            "padding:6px 8px; background:var(--bg-surface); border:1px solid var(--border-subtle); "
            "border-radius:6px 6px 0 0; margin-bottom:0;'>"
            "Top reversion candidates "
            "<span style='color:var(--text-dim); font-weight:400;'>· short half-life · low Hurst · ADF stationary</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        if rev:
            fig = make_score_bar_chart(
                [{"symbol": r["symbol"], "score": r["score"]} for r in rev],
                score_field="score",
                title="",
                height=max(180, 32 * len(rev) + 60),
                color=GREEN,
            )
            st.plotly_chart(fig, use_container_width=True, theme=None)
            _render_score_table(rev, score_label="reversion")
        else:
            st.caption("No candidates.")
    with cols[1]:
        st.markdown(
            "<div style='font-size:0.78rem; font-weight:600; color:var(--text-heading); "
            "padding:6px 8px; background:var(--bg-surface); border:1px solid var(--border-subtle); "
            "border-radius:6px 6px 0 0; margin-bottom:0;'>"
            "Top trend confirmations "
            "<span style='color:var(--text-dim); font-weight:400;'>· high Hurst · ADF can't reject · DON'T fade</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        if trend:
            fig = make_score_bar_chart(
                [{"symbol": r["symbol"], "score": r["score"]} for r in trend],
                score_field="score",
                title="",
                height=max(180, 32 * len(trend) + 60),
                color=RED,
            )
            st.plotly_chart(fig, use_container_width=True, theme=None)
            _render_score_table(trend, score_label="trend")
        else:
            st.caption("No candidates.")


def _render_score_table(rows: list[dict], score_label: str = "reversion") -> None:
    if not rows:
        return
    grid_cols = "1.0fr 1.0fr 0.9fr 1.0fr 1.0fr 0.9fr 1.4fr"
    parts = ['<div style="font-family:JetBrains Mono, monospace; font-size:0.68rem; '
             'margin-top:4px;">']
    parts.append(
        f'<div style="display:grid; grid-template-columns: {grid_cols}; '
        'gap:4px; padding:4px 6px; color:var(--text-dim); '
        'border-bottom:1px solid var(--border-subtle);">'
        '<span>SYMBOL</span>'
        '<span style="text-align:right">SCORE + reading</span>'
        '<span style="text-align:right">Z + reading</span>'
        '<span style="text-align:center">HURST + reading</span>'
        '<span style="text-align:center">HALF-LIFE + reading</span>'
        '<span style="text-align:center">ADF + reading</span>'
        '<span style="text-align:center">TAKE — what to do</span>'
        '</div>'
    )
    for r in rows:
        sym = r.get("symbol")
        score = r.get("score") or 0.0
        z = r.get("z")
        h = r.get("hurst")
        h_label = r.get("hurst_label", "—")
        hl = r.get("ou_half_life")
        adf_p = r.get("adf_pvalue") or "—"
        adf_reject = r.get("adf_reject_5pct", False)
        adf_t = r.get("adf_tstat")

        # Interpretations
        if score_label == "reversion":
            score_interp_label, score_color = reversion_score_interpretation(score)
        else:
            score_interp_label, score_color = trend_score_interpretation(score)
        z_label, z_color = z_interpretation(z)
        h_interp_label, h_color = hurst_interpretation(h)
        hl_interp_label, hl_color = half_life_interpretation(hl)
        adf_interp_label, adf_color = adf_interpretation(adf_reject, adf_p)
        take_label, take_color = overall_setup_interpretation(z, h, hl, adf_reject)

        tt_lines = [f"{sym} · composite {score_label} score = {score:.3f}",
                     f"Score interpretation: {score_interp_label}", ""]
        if z is not None:
            tt_lines.append(f"Z         {z:+.2f}σ  ({z_label})")
        if h is not None:
            tt_lines.append(f"Hurst     {h:.3f}  ({h_interp_label})")
        if hl is not None and np.isfinite(hl):
            tt_lines.append(f"Half-life {hl:.1f}d  ({hl_interp_label})")
        if adf_t is not None:
            tt_lines.append(f"ADF       t={adf_t:+.2f}, p {adf_p}  ({adf_interp_label})")
        tt_lines.append("")
        tt_lines.append(f"TAKE      {take_label}")
        if score_label == "reversion":
            tt_lines.append("")
            tt_lines.append("Strongest fade candidates — favour when ADF rejects.")
        else:
            tt_lines.append("")
            tt_lines.append("Trend regime — DO NOT fade these names solely on |z|.")
        tt = "\n".join(tt_lines)

        # Score cell with interpretation underneath
        score_html = f'<span style="color:var(--accent-bright); font-weight:600;">{score:.2f}</span>'

        # ADF cell with color
        if adf_reject:
            adf_html = f"<span style='color:var(--green); font-weight:600;'>{adf_p}</span>"
        else:
            adf_html = f"<span style='color:var(--text-muted);'>{adf_p}</span>"

        # HALF-LIFE cell
        if hl is not None and np.isfinite(hl):
            hl_value_html = f'<span style="color:{hl_color};">{hl:.1f}d</span>'
        else:
            hl_value_html = '<span style="color:var(--text-dim);">—</span>'

        # Per-metric tooltips
        score_metric_key = ("Composite reversion score" if score_label == "reversion"
                             else "Composite trend score")
        tt_score = _mt(score_metric_key, score_interp_label,
                        f"score = {score:.3f}")
        tt_z   = _mt("Z-score (z)", z_label,
                      f"{(z is not None and f'{z:+.3f}σ') or '—'}")
        tt_h   = _mt("Hurst exponent (H)", h_interp_label,
                      f"H = {(h is not None and f'{h:.3f}') or '—'}")
        tt_hl  = _mt("OU half-life", hl_interp_label,
                      f"h = {(hl is not None and np.isfinite(hl) and f'{hl:.1f}d') or 'no fit'}")
        tt_adf = _mt("ADF (lag-1, no trend)", adf_interp_label,
                      f"t = {(adf_t is not None and f'{adf_t:+.3f}') or '—'},  p {adf_p}")
        tt_take = _mt("Overall TAKE", take_label,
                       f"|z|={(z is not None and f'{abs(z):.2f}') or '—'},  "
                       f"H={(h is not None and f'{h:.2f}') or '—'},  "
                       f"hl={(hl is not None and np.isfinite(hl) and f'{hl:.1f}d') or '—'},  "
                       f"ADF rejects 5%: {adf_reject}")

        parts.append(
            f'<div title="{_tooltip_attr(tt)}" '
            f'style="display:grid; grid-template-columns: {grid_cols}; '
            f'gap:4px; padding:5px 6px; align-items:center; cursor:help; '
            f'border-bottom:1px solid var(--border-subtle);">'
            f'<span style="color:var(--accent);">{sym}</span>'
            f'<div title="{_tooltip_attr(tt_score)}" style="cursor:help;">'
            f'  {_value_with_interp(score_html, score_interp_label, score_color, align="right")}'
            f'</div>'
            f'<div title="{_tooltip_attr(tt_z)}" style="cursor:help;">'
            f'  {_value_with_interp(_z_pill(z), z_label, z_color, align="right")}'
            f'</div>'
            f'<div title="{_tooltip_attr(tt_h)}" style="text-align:center; line-height:1.15; cursor:help;">'
            f'  {_hurst_pill(h, h_label)}'
            f'  <div>{_interp_label(h_interp_label, h_color)}</div>'
            f'</div>'
            f'<div title="{_tooltip_attr(tt_hl)}" style="text-align:center; line-height:1.15; cursor:help;">'
            f'  {hl_value_html}'
            f'  <div>{_interp_label(hl_interp_label, hl_color)}</div>'
            f'</div>'
            f'<div title="{_tooltip_attr(tt_adf)}" style="text-align:center; line-height:1.15; cursor:help;">'
            f'  {adf_html}'
            f'  <div>{_interp_label(adf_interp_label, adf_color)}</div>'
            f'</div>'
            f'<div title="{_tooltip_attr(tt_take)}" style="text-align:center; cursor:help;">'
            f'  {_take_pill_outline(take_label, take_color)}'
            f'</div>'
            f'</div>'
        )
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


# =============================================================================
# View 5 — Distribution histogram
# =============================================================================
def _render_distribution(panel: dict, contracts: list[str], lookback: int) -> None:
    zs = []
    for sym in contracts:
        rec = panel.get("per_contract", {}).get(sym) or {}
        m = rec.get("by_lookback", {}).get(lookback) or {}
        z = m.get("z")
        if z is not None and not pd.isna(z):
            zs.append(z)
    if not zs:
        st.caption("No data.")
        return
    threshold_lines = [
        {"x": -2, "color": "rgba(255,255,255,0.45)", "label": "−2σ", "dash": "dash"},
        {"x": -1, "color": "rgba(255,255,255,0.30)", "label": "−1σ", "dash": "dot"},
        {"x":  0, "color": "rgba(255,255,255,0.55)", "label": "μ",   "dash": "solid"},
        {"x":  1, "color": "rgba(255,255,255,0.30)", "label": "+1σ", "dash": "dot"},
        {"x":  2, "color": "rgba(255,255,255,0.45)", "label": "+2σ", "dash": "dash"},
    ]
    fig = make_distribution_histogram(
        zs, title=f"Universe Z-distribution ({lookback}d)",
        height=220, bin_size=0.25, x_title="z-score",
        threshold_lines=threshold_lines,
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)


# =============================================================================
# View 6 — Drill-down with mean ± σ bands and OU half-life annotation
# =============================================================================
def _render_drill_down(scope_id: str, contracts: list[str], asof_date: date,
                        panel: dict, lookback: int) -> None:
    with st.expander("🔍 Drill into a contract — mean ± σ bands + OU annotation", expanded=False):
        cc1, cc2 = st.columns([2, 1])
        with cc1:
            sym = st.selectbox("Contract", options=["—"] + list(contracts),
                                key=f"{scope_id}_drill_pick")
        with cc2:
            history_lb = st.selectbox(
                "History", ["30d", "60d", "90d", "180d", "252d"],
                index=2, key=f"{scope_id}_drill_lb",
            )
        if sym == "—" or not sym:
            return
        days = {"30d": 30, "60d": 60, "90d": 90, "180d": 180, "252d": 252}[history_lb]
        start = asof_date - timedelta(days=int(days * 1.5) + 14)
        history = get_contract_history(sym, start, asof_date)
        if history.empty:
            st.info(f"No history available for {sym}.")
            return
        history = history.tail(days)
        rec = panel.get("per_contract", {}).get(sym) or {}
        m = rec.get("by_lookback", {}).get(lookback) or {}
        mu = m.get("mean")
        sd = m.get("std")
        z = m.get("z")
        pct = m.get("pct_rank")
        hl = rec.get("ou_half_life")
        h = rec.get("hurst")
        h_label = rec.get("hurst_label", "—")
        adf_t = rec.get("adf_tstat")
        adf_p = rec.get("adf_pvalue", "—")
        adf_reject = rec.get("adf_reject_5pct", False)

        z_label, _z_color_v = z_interpretation(z)
        pct_label, _ = pct_rank_interpretation(pct)
        h_label_interp, _ = hurst_interpretation(h)
        hl_label_interp, _ = half_life_interpretation(hl)
        adf_label_interp, _ = adf_interpretation(adf_reject, adf_p)
        take_label, take_color = overall_setup_interpretation(z, h, hl, adf_reject)

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric(f"Z ({lookback}d)", f"{z:+.2f}σ" if z is not None else "—",
                     delta=z_label, delta_color="off",
                     help=_mt("Z-score (z)", z_label,
                               f"{(z is not None and f'{z:+.3f}σ') or '—'}  "
                               f"(μ = {(mu is not None and f'{mu:.4f}') or '—'}, "
                               f"σ = {(sd is not None and f'{sd:.4f}') or '—'})"))
        col2.metric("μ", f"{mu:.4f}" if mu is not None else "—",
                     help=f"Sample mean over the prior {lookback} bars (excludes today).")
        col3.metric("σ", f"{sd:.4f}" if sd is not None else "—",
                     help=f"Sample standard deviation over the prior {lookback} bars "
                          "(ddof=1; excludes today).")
        col4.metric("Pct rank", f"{pct:.0f}%" if pct is not None else "—",
                     delta=pct_label, delta_color="off",
                     help=_mt("Percentile rank", pct_label,
                               f"{(pct is not None and f'{pct:.1f}%') or '—'}  "
                               f"(vs prior {lookback} bars)"))
        col5.metric(
            "OU half-life",
            f"{hl:.1f}d" if hl is not None and np.isfinite(hl) else "—",
            delta=hl_label_interp,
            delta_color="off",
            help=_mt("OU half-life", hl_label_interp,
                      f"h = {(hl is not None and np.isfinite(hl) and f'{hl:.2f}d') or 'no fit'}  "
                      "(AR(1) on prior 90 bars)"),
        )
        # Two more cards for Hurst and ADF, since they don't appear in the metric row
        h_str = f"{h:.3f}" if h is not None else "—"
        adf_t_str = f"{adf_t:+.2f}" if adf_t is not None else "—"
        cc1, cc2 = st.columns(2)
        cc1.metric(
            "Hurst exponent (H)", h_str, delta=h_label_interp, delta_color="off",
            help=_mt("Hurst exponent (H)", h_label_interp,
                      f"H = {h_str}  (R/S analysis on prior 90 bars of log-returns)"),
        )
        cc2.metric(
            "ADF (lag-1)",
            f"t = {adf_t_str}",
            delta=adf_label_interp, delta_color="off",
            help=_mt("ADF (lag-1, no trend)", adf_label_interp,
                      f"t-stat = {adf_t_str},  p-value {adf_p}  "
                      f"({'rejects unit root at 5%' if adf_reject else 'cannot reject unit root'})"),
        )
        # Detailed interpretation paragraph
        h_str = f"{h:.3f}" if h is not None else "—"
        adf_str = f"t {adf_t:+.2f}, p {adf_p}" if adf_t is not None else "—"
        hl_str = f"{hl:.1f}d" if (hl is not None and np.isfinite(hl)) else "no fit"
        st.markdown(
            f"<div style='padding:8px 10px; background:var(--bg-surface); "
            f"border:1px solid var(--border-subtle); border-radius:6px; "
            f"margin: 0.5rem 0; font-size:0.78rem; line-height:1.5;'>"
            f"<div style='display:flex; align-items:center; gap:0.6rem; margin-bottom:0.4rem;'>"
            f"<span style='color:var(--text-dim); text-transform:uppercase; "
            f"letter-spacing:0.06em; font-size:0.65rem;'>Reading for {sym}</span>"
            f"<span style='display:inline-block; padding:2px 8px; "
            f"background:{take_color}; color:var(--bg-base); "
            f"font-family:JetBrains Mono, monospace; font-weight:700; "
            f"font-size:0.7rem; border-radius:4px;'>{take_label}</span>"
            f"</div>"
            f"<div style='color:var(--text-body);'>"
            f"<b>Z-score</b> {(z is not None and f'{z:+.2f}σ') or '—'} → <i>{z_label}</i>. "
            f"<b>Pct rank</b> {(pct is not None and f'{pct:.0f}%') or '—'} → <i>{pct_label}</i>. "
            f"<b>Hurst</b> {h_str} → <i>{h_label_interp}</i> "
            f"({'momentum dominant — fading is risky' if h_label == 'TRENDING' else 'anti-persistent — fade-friendly' if h_label == 'REVERTING' else 'no memory — pure noise'}). "
            f"<b>OU half-life</b> {hl_str} → <i>{hl_label_interp}</i> "
            f"({'a deviation from the mean halves in roughly this many trading days; shorter = stronger reversion' if hl is not None and np.isfinite(hl) else 'AR(1) coefficient is outside (0,1) so this contract is NOT mean-reverting in the last 90 bars'}). "
            f"<b>ADF</b> {adf_str} → <i>{adf_label_interp}</i> "
            f"({'series rejects unit-root hypothesis at 5% — reversion is statistically supported' if adf_reject else 'cannot reject unit root — series may be trending; fade with caution'}). "
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        fig = make_mean_bands_chart(
            history, mean=mu if mu is not None else 0.0,
            std=sd if sd is not None else 0.0,
            title=f"{sym} · close + μ ± σ ± 2σ ({lookback}d sample)",
            height=360,
            half_life=hl,
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)


# =============================================================================
# Side panel — 12 analytics blocks
# =============================================================================
def _block_universe_stretch(stretch: dict, lookback: int) -> None:
    _section_header(
        f"Universe stretch ({lookback}d)",
        tooltip=(
            "Counts of contracts at each |z| band on the focus lookback.\n"
            "Skew = third-moment shape (positive = right-tail rich, negative = left-tail rich).\n"
            "Excess kurtosis (kurt − 3) > 0 = fatter tails than normal."
        ),
    )
    z_tt = _mt("Z-score (z)")    # generic Z guide (no current bucket)
    rows = [
        ("# z ≥ +2", str(stretch["n_pos2"]), "red" if stretch["n_pos2"] else None, None,
            _mt("Z-score (z)", "stretched HIGH (>+2σ)", "count of contracts with z ≥ +2")),
        ("# z ≤ −2", str(stretch["n_neg2"]), "green" if stretch["n_neg2"] else None, None,
            _mt("Z-score (z)", "stretched LOW (<−2σ)", "count of contracts with z ≤ −2")),
        ("# +1 ≤ z < +2", str(stretch["n_pos1"]), "amber", None,
            _mt("Z-score (z)", "elevated HIGH (+1 to +2σ)", "count of contracts with +1 ≤ z < +2")),
        ("# −2 < z ≤ −1", str(stretch["n_neg1"]), "amber", None,
            _mt("Z-score (z)", "elevated LOW (−1 to −2σ)", "count of contracts with −2 < z ≤ −1")),
        ("median z", f"{stretch['median_z']:+.2f}" if stretch["median_z"] is not None else "—",
            "accent", None, z_tt),
        ("std z",    f"{stretch['std_z']:.2f}" if stretch.get("std_z") is not None else "—",
            None, None,
            "Standard deviation of the universe's Z-scores at this lookback.\n"
            "≈ 1.0 = textbook · > 1.5 = wider stretch dispersion · < 0.7 = bunched"),
        ("skew",     f"{stretch['skew']:+.2f}" if stretch.get("skew") is not None else "—",
            None, None,
            "Third-moment skewness of universe Z-scores.\n"
            "  > +1 → right-tail heavy (more extreme HIGHs)\n"
            "  < −1 → left-tail heavy (more extreme LOWs)\n"
            "  ~0 → symmetric"),
        ("excess kurt", f"{stretch['kurt']:+.2f}" if stretch.get("kurt") is not None else "—",
            None, None,
            "Excess kurtosis (kurt − 3) of universe Z-scores.\n"
            "  > +1 → fat tails (more extreme prints than Gaussian)\n"
            "  < −1 → thin tails (less extreme than Gaussian)\n"
            "  ~0 → roughly Gaussian"),
    ]
    _kv_block(rows)

    # Reading paragraph — interpret the universe stretch
    parts = []
    pos2 = stretch["n_pos2"]; neg2 = stretch["n_neg2"]
    med = stretch.get("median_z")
    skew = stretch.get("skew"); kurt = stretch.get("kurt")
    if (pos2 + neg2) >= 5:
        parts.append(f"<b>{pos2 + neg2}</b> contracts in the |z|≥2 zone — universe is stretched.")
    elif (pos2 + neg2) == 0:
        parts.append("No |z|≥2 extremes — universe is calm.")
    if med is not None:
        if med >= 0.5:
            parts.append(f"Median z {med:+.2f}σ → most contracts trade <b>rich</b> vs their {lookback}d mean.")
        elif med <= -0.5:
            parts.append(f"Median z {med:+.2f}σ → most contracts trade <b>cheap</b> vs their {lookback}d mean.")
        else:
            parts.append(f"Median z {med:+.2f}σ → universe centred near its {lookback}d mean.")
    if skew is not None:
        if abs(skew) >= 1:
            direction = "right" if skew > 0 else "left"
            parts.append(f"Skew {skew:+.2f} → distribution has a <b>{direction}-tail bias</b> (more extreme on that side).")
    if kurt is not None and kurt >= 1:
        parts.append(f"Excess kurtosis {kurt:+.2f} → <b>fatter tails than normal</b> — more frequent extreme prints than a Gaussian would predict.")
    if not parts:
        parts.append("Universe stretch metrics are within typical ranges.")
    _reading(" ".join(parts))


def _block_z_pattern_catalog(panel: dict, contracts: list[str]) -> None:
    _section_header(
        "Multi-window Z confluence patterns",
        tooltip=(
            "8-pattern classifier across all selected lookbacks.\n"
            "PERSISTENT/ACCELERATING — durable trend\n"
            "DECELERATING/REVERTING — reversal candidates\n"
            "FRESH — event-driven; DRIFTED — coiling\n"
            "STABLE/MIXED — neutral / no clear shape"
        ),
    )
    counts: dict = {}
    for sym in contracts:
        rec = panel.get("per_contract", {}).get(sym) or {}
        p = rec.get("pattern", "MIXED")
        counts[p] = counts.get(p, 0) + 1
    order = ["PERSISTENT", "ACCELERATING", "DECELERATING", "FRESH",
             "DRIFTED", "REVERTING", "STABLE", "MIXED"]
    rows = []
    cls_map = {"PERSISTENT": "red", "ACCELERATING": "red", "DECELERATING": "amber",
                "FRESH": "accent", "DRIFTED": "amber", "REVERTING": "green"}
    for p in order:
        if p in counts:
            rows.append((p, str(counts[p]), cls_map.get(p), None,
                          _mt("Multi-window confluence pattern", p,
                               f"{counts[p]} contract(s) classified as {p}")))
    if not rows:
        st.caption("No patterns classified.")
        return
    _kv_block(rows)

    # Reading paragraph — pick the dominant trade-relevant pattern
    trend_count = counts.get("PERSISTENT", 0) + counts.get("ACCELERATING", 0)
    fade_count = counts.get("DECELERATING", 0) + counts.get("REVERTING", 0)
    coil_count = counts.get("DRIFTED", 0)
    fresh_count = counts.get("FRESH", 0)
    parts = []
    if trend_count >= 3:
        parts.append(f"<b>{trend_count}</b> contracts in trend regimes (PERSISTENT/ACCELERATING) — the dominant theme is <b>directional persistence</b>.")
    if fade_count >= 3:
        parts.append(f"<b>{fade_count}</b> contracts in reversal regimes (DECELERATING/REVERTING) — fade setups are clustering.")
    if coil_count >= 3:
        parts.append(f"<b>{coil_count}</b> in DRIFTED → coiling pattern after prior thrust; watch for breakout direction.")
    if fresh_count >= 3:
        parts.append(f"<b>{fresh_count}</b> FRESH names — recent event-driven moves not yet confirmed by longer windows.")
    if not parts:
        parts.append("No clear regime concentration — the universe is mixed.")
    _reading(" ".join(parts))


def _block_section_stretch(panel: dict, contracts: list[str], front_end: int,
                            mid_end: int, lookback: int) -> None:
    _section_header(
        f"Section stretch ({lookback}d)",
        tooltip=(
            "Median z per curve section. Identifies which section of the curve is rich/cheap.\n"
            "Curve label combinations:\n"
            "  RICH/CHEAP — all sections same sign ≥1σ\n"
            "  STEEPENING — back rich vs front · FLATTENING — front rich vs back\n"
            "  BELLY-BID — mid > wings · BELLY-OFFERED — mid < wings"
        ),
    )
    sr = section_regime_z(panel, contracts, front_end, mid_end, lookback)
    label = sr.get("label", "—")
    color = (
        "red" if "RICH" in label else
        "green" if "CHEAP" in label else
        "accent"
    )
    curve_tt = (
        "Curve-level Z regime\n\n"
        "Built from each section's median z-score:\n"
        "  RICH: all sections median z ≥ +1σ\n"
        "  CHEAP: all sections median z ≤ −1σ\n"
        "  STEEPENING: back median z − front median z ≥ +1σ\n"
        "  FLATTENING: front median z − back median z ≥ +1σ\n"
        "  BELLY-BID: mid z > avg(front, back) by ≥ +1σ\n"
        "  BELLY-OFFERED: mid z < avg(front, back) by ≥ +1σ\n"
        "  MIXED: none of the above"
    )
    rows = [(f"Curve label", label, color, None, curve_tt)]
    for sec in ("front", "mid", "back"):
        s = sr.get("sections", {}).get(sec) or {}
        med = s.get("median_z")
        n_ext = s.get("n_extreme", 0)
        n_tot = s.get("n_total", 0)
        cls = "red" if med is not None and med >= 1 else ("green" if med is not None and med <= -1 else None)
        bucket_label = ("rich (med z ≥ +1σ)" if med is not None and med >= 1
                         else "cheap (med z ≤ −1σ)" if med is not None and med <= -1
                         else "neutral (|med z| < 1σ)" if med is not None
                         else None)
        sec_tt = _mt("Z-score (z)", bucket_label,
                      f"section median z = {(med is not None and f'{med:+.3f}σ') or '—'};  "
                      f"|z|≥2 in section: {n_ext}/{n_tot}")
        rows.append((sec.title(),
                      f"med {med:+.2f}σ" if med is not None else "—",
                      cls,
                      f"{n_ext}/{n_tot} |z|≥2",
                      sec_tt))
    _kv_block(rows)

    # Reading paragraph — translate curve label into trader action
    label_lc = label.lower()
    reading_text = (
        f"All sections trade rich (>+1σ) — universe-wide overshoot up; risk skewed to mean-reversion lower." if "rich" in label_lc else
        f"All sections trade cheap (<−1σ) — universe-wide overshoot down; risk skewed to mean-reversion higher." if "cheap" in label_lc else
        f"Back is rich vs front — <b>steepener</b> regime; back-end may revert lower OR front may catch up." if "steepening" in label_lc else
        f"Front is rich vs back — <b>flattener</b> regime; front may revert lower OR back may catch up." if "flattening" in label_lc else
        f"Mid (belly) is bid up vs the wings — possible body squeeze; watch for belly to revert." if "belly-bid" in label_lc else
        f"Mid (belly) is offered vs the wings — possible body sell-off; watch for wings to compress." if "belly-offered" in label_lc else
        "No clear cross-section divergence — sections are within ±1σ of each other; no strong roll signal."
    )
    _reading(reading_text)


def _block_half_life_ranking(panel: dict, contracts: list[str], lookback: int) -> None:
    _section_header(
        "Half-life ranking · top fastest reverters",
        tooltip=(
            "OU half-life from AR(1) fit on prior 90 bars: h = −ln(2)/ln(b)\n"
            "Shorter half-life = stronger mean-reversion. None = AR(1) coefficient\n"
            "outside (0,1) → not mean-reverting in this window."
        ),
    )
    rows = []
    for sym in contracts:
        rec = panel.get("per_contract", {}).get(sym) or {}
        hl = rec.get("ou_half_life")
        if hl is None or not np.isfinite(hl):
            continue
        z = (rec.get("by_lookback", {}).get(lookback, {}) or {}).get("z")
        rows.append((sym, hl, z))
    rows.sort(key=lambda r: r[1])
    if not rows:
        st.caption("No reverters detected — no contracts have an AR(1) coefficient in (0,1).")
        _reading("None of the contracts in this scope show OU-style mean reversion in the last 90 bars. "
                  "Either they're trending, near random walk, or noise dominates — fade setups are not "
                  "well-supported here.")
        return
    n_fast = sum(1 for _, hl_, _ in rows if hl_ <= 10)
    n_med = sum(1 for _, hl_, _ in rows if 10 < hl_ <= 30)
    for sym, hl, z in rows[:6]:
        z_str = f"{z:+.2f}σ" if z is not None else "—"
        z_color = _z_color(z)
        hl_label, hl_color = half_life_interpretation(hl)
        tt = _mt("OU half-life", hl_label, f"h = {hl:.2f}d  (AR(1) on prior 90 bars)")
        st.markdown(
            f"<div title='{_tooltip_attr(tt)}' "
            f"style='font-family:JetBrains Mono, monospace; font-size:0.7rem; "
            f"padding:3px 0; color:var(--text-body); cursor:help; "
            f"border-bottom:1px solid var(--border-subtle); "
            f"display:flex; justify-content:space-between; align-items:center; gap:6px;'>"
            f"<span><span style='color:var(--accent);'>{sym}</span> · "
            f"<b style='color:{hl_color};'>{hl:.1f}d</b> "
            f"<span style='color:{hl_color}; font-style:italic; font-size:0.62rem;'>"
            f"({hl_label})</span></span>"
            f"<span style='color:{z_color};'>z {z_str}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    # Reading
    parts = []
    if n_fast >= 1:
        parts.append(f"<b>{n_fast}</b> contracts have a half-life ≤10d (fast reverters) — high-quality fade candidates when paired with stretched z.")
    if n_med >= 3:
        parts.append(f"<b>{n_med}</b> have a half-life of 10–30d (medium) — workable but slower reversions.")
    if not parts:
        parts.append("Reverters detected are slow (half-life >30d) — wait for stronger setups.")
    _reading(" ".join(parts))


def _block_hurst_summary(panel: dict, contracts: list[str]) -> None:
    _section_header(
        "Hurst regime summary",
        tooltip=(
            "Hurst exponent via R/S analysis on prior 90 bars of log-returns.\n"
            "  H > 0.55 — TRENDING (persistent moves)\n"
            "  H < 0.45 — REVERTING (anti-persistent)\n"
            "  ≈ 0.5    — RANDOM (no memory)\n"
            "Trending names with high |z| are NOT good fade candidates."
        ),
    )
    counts = {"TRENDING": 0, "RANDOM": 0, "REVERTING": 0, "—": 0}
    for sym in contracts:
        rec = panel.get("per_contract", {}).get(sym) or {}
        lbl = rec.get("hurst_label", "—")
        counts[lbl] = counts.get(lbl, 0) + 1
    h_tt_trending  = _mt("Hurst exponent (H)", "trending (H≥0.55)", f"{counts['TRENDING']} contract(s) have H ≥ 0.55")
    h_tt_random    = _mt("Hurst exponent (H)", "random walk (H≈0.5)", f"{counts['RANDOM']} contract(s) have 0.45 ≤ H < 0.55")
    h_tt_reverting = _mt("Hurst exponent (H)", "reverting (H≤0.45)",  f"{counts['REVERTING']} contract(s) have H ≤ 0.45")
    rows = [
        ("# TRENDING (H>0.55)",  str(counts["TRENDING"]), "red", None, h_tt_trending),
        ("# RANDOM (H≈0.5)",      str(counts["RANDOM"]),    None,   None, h_tt_random),
        ("# REVERTING (H<0.45)", str(counts["REVERTING"]), "green", None, h_tt_reverting),
        ("# unfit",               str(counts["—"]),          None,   None,
            "Contracts where Hurst could not be estimated (insufficient or degenerate data)."),
    ]
    _kv_block(rows)

    # Reading paragraph — characterise the universe's memory regime
    total_classified = counts["TRENDING"] + counts["RANDOM"] + counts["REVERTING"]
    if total_classified == 0:
        _reading("Insufficient data to estimate Hurst exponents.")
        return
    trending_pct = counts["TRENDING"] / total_classified * 100
    reverting_pct = counts["REVERTING"] / total_classified * 100
    parts = []
    if trending_pct >= 50:
        parts.append(f"<b>{trending_pct:.0f}%</b> of contracts have <b>trending memory</b> (H&gt;0.55) — moves persist; "
                     "fading high |z| names is risky in this regime.")
    elif reverting_pct >= 50:
        parts.append(f"<b>{reverting_pct:.0f}%</b> of contracts are <b>mean-reverting</b> (H&lt;0.45) — "
                     "fade setups are well-supported by the universe's memory structure.")
    elif trending_pct >= 30 and reverting_pct >= 30:
        parts.append("Mixed memory regime — trending and reverting names coexist; treat each contract on its own merits.")
    else:
        parts.append("Most names sit near random-walk H≈0.5 — pure noise; rely on Z-score + ADF gate to pick fades.")
    _reading(" ".join(parts))


def _block_adf_z_gate(panel: dict, contracts: list[str], lookback: int) -> None:
    _section_header(
        f"ADF + Z gate ({lookback}d)",
        tooltip=(
            "Pairs |z|≥2 contracts with their ADF stationarity result.\n"
            "  STATIONARY (5%) — ADF rejects unit root → mean-reversion is trustworthy\n"
            "  cannot reject — series might be trending → fade with caution\n"
            "Best reversion entries: |z|≥2 AND ADF rejects."
        ),
    )
    n_z2_stat = n_z2_nonstat = 0
    examples_stat = []
    examples_nonstat = []
    for sym in contracts:
        rec = panel.get("per_contract", {}).get(sym) or {}
        z = (rec.get("by_lookback", {}).get(lookback, {}) or {}).get("z")
        if z is None or pd.isna(z) or abs(z) < 2:
            continue
        if rec.get("adf_reject_5pct"):
            n_z2_stat += 1
            if len(examples_stat) < 3:
                examples_stat.append((sym, z, rec.get("adf_pvalue", "—")))
        else:
            n_z2_nonstat += 1
            if len(examples_nonstat) < 3:
                examples_nonstat.append((sym, z, rec.get("adf_pvalue", "—")))
    rows = [
        ("|z|≥2 · STATIONARY",      str(n_z2_stat),    "green" if n_z2_stat else None, None,
            _mt("ADF (lag-1, no trend)", "stationary",
                 f"{n_z2_stat} contract(s) with |z|≥2 AND ADF rejects H₀ at 5%")),
        ("|z|≥2 · NON-STATIONARY", str(n_z2_nonstat), "red" if n_z2_nonstat else None, None,
            _mt("ADF (lag-1, no trend)", "non-stationary",
                 f"{n_z2_nonstat} contract(s) with |z|≥2 BUT ADF cannot reject H₀")),
    ]
    _kv_block(rows)
    if examples_stat:
        st.markdown(
            "<div style='font-size:0.65rem; color:var(--text-dim); margin:4px 0 2px 0;'>"
            "Stationary + stretched (top fade candidates):</div>",
            unsafe_allow_html=True,
        )
        for sym, z, p in examples_stat:
            tt = _mt("ADF (lag-1, no trend)", "stationary",
                      f"|z| ≥ 2 ({z:+.2f}σ),  ADF p {p},  rejects unit-root H₀")
            st.markdown(
                f"<div title='{_tooltip_attr(tt)}' "
                f"style='font-family:JetBrains Mono, monospace; font-size:0.7rem; "
                f"padding:2px 0; cursor:help;'>"
                f"<span style='color:var(--accent);'>{sym}</span> · "
                f"z {z:+.2f}σ · ADF {p} · "
                f"<span style='color:var(--green); font-style:italic;'>fade-trustworthy</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
    if examples_nonstat:
        st.markdown(
            "<div style='font-size:0.65rem; color:var(--text-dim); margin:4px 0 2px 0;'>"
            "Non-stationary + stretched (DON'T fade alone):</div>",
            unsafe_allow_html=True,
        )
        for sym, z, p in examples_nonstat:
            tt = _mt("ADF (lag-1, no trend)", "non-stationary",
                      f"|z| ≥ 2 ({z:+.2f}σ),  ADF p {p},  cannot reject unit-root H₀")
            st.markdown(
                f"<div title='{_tooltip_attr(tt)}' "
                f"style='font-family:JetBrains Mono, monospace; font-size:0.7rem; "
                f"padding:2px 0; cursor:help;'>"
                f"<span style='color:var(--accent);'>{sym}</span> · "
                f"z {z:+.2f}σ · ADF {p} · "
                f"<span style='color:var(--red); font-style:italic;'>caution fading</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # Reading paragraph
    parts = []
    if n_z2_stat >= 1:
        parts.append(f"<b>{n_z2_stat}</b> stretched names test as stationary — these are the <b>highest-quality fade setups</b> in the universe right now.")
    if n_z2_nonstat >= 1:
        parts.append(f"<b>{n_z2_nonstat}</b> stretched names are non-stationary — could keep extending; <b>do not fade these on |z| alone</b>.")
    if not parts:
        parts.append("No |z|≥2 names — universe is calm; no Z-stretch to gate.")
    _reading(" ".join(parts))


def _block_top_reversion(panel: dict, contracts: list[str]) -> None:
    _section_header(
        "Top composite reversion candidates",
        tooltip=(
            "Composite score weights:\n"
            "  0.50 · |z|/2  (capped at 1)\n"
            "  0.25 · (1 − Hurst)\n"
            "  0.15 · 10 / max(1, half-life)\n"
            "  0.10 · 1 if ADF rejects (else 0)\n"
            "Higher = stronger reversion setup."
        ),
    )
    rev = rank_by_score(panel, contracts, score_field="reversion_score", top_k=5)
    if not rev:
        st.caption("No reversion candidates.")
        _reading("No contracts pass the composite reversion-score threshold — wait for stronger setups.")
        return
    n_strong = sum(1 for r in rev if (r.get("score") or 0) >= 0.7)
    n_mod = sum(1 for r in rev if 0.5 <= (r.get("score") or 0) < 0.7)
    for r in rev:
        z = r.get("z")
        z_str = f"{z:+.2f}σ" if z is not None else "—"
        hl = r.get("ou_half_life")
        hl_str = (f"hl {hl:.0f}d"
                  if hl is not None and np.isfinite(hl) else "hl —")
        score_label, score_color = reversion_score_interpretation(r.get("score"))
        right_text = f"z {z_str} · {hl_str}"
        tt = _mt("Composite reversion score", score_label,
                  f"score = {r['score']:.3f}  ·  z {z_str}  ·  {hl_str}")
        st.markdown(
            f"<div title='{_tooltip_attr(tt)}' "
            f"style='font-family:JetBrains Mono, monospace; font-size:0.7rem; "
            f"padding:3px 0; color:var(--text-body); cursor:help; "
            f"border-bottom:1px solid var(--border-subtle); "
            f"display:flex; justify-content:space-between; align-items:center; gap:6px;'>"
            f"<span><span style='color:var(--accent);'>{r['symbol']}</span> · "
            f"<b style='color:{score_color};'>{r['score']:.2f}</b> "
            f"<span style='color:{score_color}; font-size:0.62rem; font-style:italic;'>"
            f"({score_label})</span></span>"
            f"<span style='color:var(--text-muted);'>{right_text}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    # Reading paragraph
    if n_strong:
        _reading(f"<b>{n_strong}</b> contracts score ≥0.70 — these are <b>high-quality fade setups</b> with stretched z, "
                 "low Hurst, fast half-life, and ADF stationarity all aligned.")
    elif n_mod:
        _reading(f"Best candidates score 0.50–0.70 — moderate setups; pair with sizing discipline and confirmation.")
    else:
        _reading("All scores are weak (<0.50) — no high-quality fades right now; wait for stronger setups.")


def _block_top_trend(panel: dict, contracts: list[str]) -> None:
    _section_header(
        "Top trend confirmations · DON'T fade",
        tooltip=(
            "Anti-reversion list. Composite trend score weights:\n"
            "  0.55 · |z|/2  ·  0.35 · Hurst  ·  0.10 · 1 if ADF can't reject (else 0)\n"
            "These names have stretched z + trending memory + non-stationarity ⇒ ride, don't fade."
        ),
    )
    tr = rank_by_score(panel, contracts, score_field="trend_score", top_k=5)
    if not tr:
        st.caption("No trend candidates.")
        _reading("No contracts pass the trend-score threshold — the universe isn't strongly trending.")
        return
    n_strong = sum(1 for r in tr if (r.get("score") or 0) >= 0.7)
    for r in tr:
        z = r.get("z")
        z_str = f"{z:+.2f}σ" if z is not None else "—"
        score_label, score_color = trend_score_interpretation(r.get("score"))
        tt = _mt("Composite trend score", score_label,
                  f"score = {r['score']:.3f}  ·  z {z_str}  ·  Hurst {r.get('hurst_label', '—')}")
        st.markdown(
            f"<div title='{_tooltip_attr(tt)}' "
            f"style='font-family:JetBrains Mono, monospace; font-size:0.7rem; "
            f"padding:3px 0; color:var(--text-body); cursor:help; "
            f"border-bottom:1px solid var(--border-subtle); "
            f"display:flex; justify-content:space-between; align-items:center; gap:6px;'>"
            f"<span><span style='color:var(--accent);'>{r['symbol']}</span> · "
            f"<b style='color:{score_color};'>{r['score']:.2f}</b> "
            f"<span style='color:{score_color}; font-size:0.62rem; font-style:italic;'>"
            f"({score_label})</span></span>"
            f"<span style='color:var(--text-muted);'>"
            f"z {z_str} · {r.get('hurst_label', '—')}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    if n_strong:
        _reading(f"<b>{n_strong}</b> names score ≥0.70 — these have strong trending memory + stretched z + "
                 "non-stationarity. <b>Do NOT fade these on |z| alone</b> — let the trend run.")
    else:
        _reading("Trend scores are moderate or weak — no strong 'don't-fade' signal. Z-based fades are workable "
                 "with the usual confluence checks.")


def _block_velocity_to_mean(panel: dict, contracts: list[str], lookback: int) -> None:
    _section_header(
        f"Velocity / pace of reversion ({lookback}d)",
        tooltip=(
            "Avg daily Δ|z| over last 5 bars. Negative = |z| shrinking (reverting toward mean).\n"
            "Positive = |z| growing (still extending). Distinguishes 'already reverting' from 'still extending'."
        ),
    )
    reverting = []
    extending = []
    for sym in contracts:
        rec = panel.get("per_contract", {}).get(sym) or {}
        m = rec.get("by_lookback", {}).get(lookback) or {}
        v = m.get("velocity_to_mean")
        z = m.get("z")
        if v is None or pd.isna(v) or z is None or pd.isna(z) or abs(z) < 1:
            continue
        if v < 0:
            reverting.append((sym, v, z))
        else:
            extending.append((sym, v, z))
    reverting.sort(key=lambda r: r[1])
    extending.sort(key=lambda r: -r[1])
    vel_tt = _mt("Velocity to mean (Δ|z|/d)")
    rows = [
        ("# reverting (Δ|z|<0, |z|≥1)",  str(len(reverting)), "green", None, vel_tt),
        ("# still extending (Δ|z|>0, |z|≥1)", str(len(extending)), "red", None, vel_tt),
    ]
    _kv_block(rows)
    for sym, v, z in reverting[:3]:
        v_label, v_color = velocity_to_mean_interpretation(v)
        tt = _mt("Velocity to mean (Δ|z|/d)", v_label,
                  f"Δ|z|/day = {v:+.4f}  ·  z {z:+.2f}σ")
        st.markdown(
            f"<div title='{_tooltip_attr(tt)}' "
            f"style='font-family:JetBrains Mono, monospace; font-size:0.69rem; "
            f"padding:2px 0; cursor:help;'>"
            f"<span style='color:var(--accent);'>{sym}</span> · "
            f"<span style='color:{v_color};'>Δ|z|/d {v:+.3f}</span> "
            f"<span style='color:{v_color}; font-size:0.62rem; font-style:italic;'>"
            f"({v_label})</span> · "
            f"z {z:+.2f}σ"
            f"</div>",
            unsafe_allow_html=True,
        )
    # Reading paragraph
    if len(reverting) >= 3 and len(reverting) > len(extending):
        _reading(f"More contracts are <b>reverting</b> than extending ({len(reverting)} vs {len(extending)}) — the "
                 "universe is generally pulling back toward its mean. Existing fade trades likely have momentum.")
    elif len(extending) >= 3 and len(extending) > len(reverting):
        _reading(f"More contracts are <b>still extending</b> than reverting ({len(extending)} vs {len(reverting)}) — "
                 "moves are not yet exhausted. Premature fades will continue to bleed.")
    elif len(reverting) + len(extending) == 0:
        _reading("No contracts have meaningful |z|≥1 — universe is quiet.")
    else:
        _reading("Mixed velocity — reversions and extensions are roughly balanced; treat each setup individually.")


def _block_z_proximity_xcheck(panel: dict, prox_panel: dict, contracts: list[str],
                                lookback: int) -> None:
    _section_header(
        f"Cross-check vs Proximity ({lookback}d)",
        tooltip=(
            "Pairs Z-stretch with the proximity flag from the Proximity engine.\n"
            "  UNTESTED-EXTREME    AT/NEAR but |z|<1     — uncrowded breakout\n"
            "  STRETCHED-EXTREME   AT/NEAR and |z|≥2     — exhausted, fade with confluence\n"
            "  COILED              FAR and |z|≥1.5      — rich/cheap but not at recent extreme\n"
            "  NORMAL              FAR and |z|<1         — quiet"
        ),
    )
    untested, stretched, coiled = [], [], []
    for sym in contracts:
        zrec = panel.get("per_contract", {}).get(sym) or {}
        zm = zrec.get("by_lookback", {}).get(lookback) or {}
        z = zm.get("z")
        prec = prox_panel.get("per_contract", {}).get(sym) or {}
        pm = prec.get("by_lookback", {}).get(lookback) or {}
        flag_h = pm.get("flag_high"); flag_l = pm.get("flag_low")
        if z is None or pd.isna(z):
            continue
        is_extreme = flag_h in ("AT", "NEAR") or flag_l in ("AT", "NEAR")
        if is_extreme and abs(z) < 1.0:
            untested.append((sym, z))
        elif is_extreme and abs(z) >= 2.0:
            stretched.append((sym, z))
        elif (not is_extreme) and abs(z) >= 1.5:
            coiled.append((sym, z))
    rows = [
        ("# UNTESTED-EXTREME", str(len(untested)), "accent", "AT/NEAR · |z|<1",
            "UNTESTED-EXTREME\n   Current: AT/NEAR proximity flag with |z|<1\n\n"
            "What:    Contract sits at the recent N-day high or low BUT its z-score "
            "vs the N-day mean is small. The level was reached without a major "
            "departure from typical levels — the move has 'room to run'.\n\n"
            "Inputs:  proximity flag ∈ {AT, NEAR}, |z| < 1\n"
            "Action:  UNCROWDED BREAKOUT — extension candidate; do not fade."),
        ("# STRETCHED-EXTREME", str(len(stretched)), "red", "AT/NEAR · |z|≥2",
            "STRETCHED-EXTREME\n   Current: AT/NEAR proximity flag with |z|≥2\n\n"
            "What:    Contract is at the recent extreme AND is statistically far "
            "from its mean — exhausted move. Both engines (proximity + z) agree.\n\n"
            "Inputs:  proximity flag ∈ {AT, NEAR}, |z| ≥ 2\n"
            "Action:  FADE WITH CONFLUENCE — high-quality reversal entry zone."),
        ("# COILED",            str(len(coiled)),    "amber", "FAR · |z|≥1.5",
            "COILED\n   Current: FAR from extreme but |z|≥1.5\n\n"
            "What:    Contract is rich/cheap on the z-score (statistically stretched) "
            "but sitting away from the recent N-day extreme — mean-reversion is "
            "already in process.\n\n"
            "Inputs:  proximity flag = FAR, |z| ≥ 1.5\n"
            "Action:  REVERSION IN PROCESS — late entries; verify with velocity."),
    ]
    _kv_block(rows)
    # Reading paragraph
    parts = []
    if untested:
        parts.append(f"<b>{len(untested)}</b> UNTESTED — sitting at extremes but z is small. <b>Uncrowded breakouts</b>; the move has room to extend.")
    if stretched:
        parts.append(f"<b>{len(stretched)}</b> STRETCHED — at extremes AND |z|≥2. <b>Exhausted</b>; fade-with-confluence candidates.")
    if coiled:
        parts.append(f"<b>{len(coiled)}</b> COILED — rich/cheap on z but FAR from recent highs/lows. <b>Mean-reversion in process</b>.")
    if not parts:
        parts.append("No clear cross-check signals — universe is in normal regime.")
    _reading(" ".join(parts))


def _block_methodology() -> None:
    text = _methodology_text()
    attr = _tooltip_attr(text)
    st.markdown(
        f"<div title=\"{attr}\" style='cursor:help; padding:6px 8px; "
        f"background:var(--bg-elevated); border:1px dashed var(--border-default); "
        f"border-radius:6px; margin: 0.6rem 0 0.3rem 0;'>"
        f"<div style='font-size:0.7rem; color:var(--text-dim); "
        f"text-transform:uppercase; letter-spacing:0.06em; font-weight:600;'>"
        f"Methodology / formulas <span style='color:var(--accent-dim);'>ⓘ</span>"
        f"</div>"
        f"<div style='font-size:0.7rem; color:var(--text-muted); margin-top:2px;'>"
        f"Z · OU · Hurst · ADF · composite scores — hover for full text"
        f"</div></div>",
        unsafe_allow_html=True,
    )


def _block_pattern_legend() -> None:
    with st.expander("Z pattern legend", expanded=False):
        descs = get_z_pattern_descriptions()
        order = ["PERSISTENT", "ACCELERATING", "DECELERATING", "FRESH",
                 "DRIFTED", "REVERTING", "STABLE", "MIXED"]
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
                                  front_end: int, mid_end: int) -> None:
    z_panel = data.get("zscore", {})
    prox_panel = data.get("proximity", {})
    st.markdown(
        f"<div style='display:flex; align-items:center; gap:0.5rem; "
        f"padding-bottom:0.4rem; border-bottom:1px solid var(--border-subtle); margin-bottom:0.4rem;'>"
        f"<span style='color:var(--accent); font-size:0.85rem; font-weight:600;'>📊 Analysis · all blocks (scroll)</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    _block_methodology()
    stretch = _universe_stretch(z_panel, contracts, lookback)
    _block_universe_stretch(stretch, lookback)
    _block_z_pattern_catalog(z_panel, contracts)
    _block_section_stretch(z_panel, contracts, front_end, mid_end, lookback)
    _block_half_life_ranking(z_panel, contracts, lookback)
    _block_hurst_summary(z_panel, contracts)
    _block_adf_z_gate(z_panel, contracts, lookback)
    _block_top_reversion(z_panel, contracts)
    _block_top_trend(z_panel, contracts)
    _block_velocity_to_mean(z_panel, contracts, lookback)
    _block_z_proximity_xcheck(z_panel, prox_panel, contracts, lookback)
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

    # Interpretation guide — full reference for every metric and bucket
    _render_interpretation_guide()

    with st.expander("Curve section boundaries (Front / Mid / Back)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            front_end = st.number_input(
                "Front-end (last index)", min_value=2, max_value=60,
                value=int(st.session_state.get(f"{_SCOPE_PREFIX}_fe", DEFAULT_FRONT_END)),
                step=1, key=f"{_SCOPE_PREFIX}_fe",
            )
        with c2:
            mid_end = st.number_input(
                "Mid-end (last index)", min_value=front_end + 1, max_value=120,
                value=int(st.session_state.get(f"{_SCOPE_PREFIX}_me", DEFAULT_MID_END)),
                step=1, key=f"{_SCOPE_PREFIX}_me",
            )

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
                key=f"{_SCOPE_PREFIX}_tenor_spread",
                format_func=lambda t: f"{t}M",
            )
        else:
            tenor_months = st.selectbox(
                "Tenor", options=fly_tenors, index=0,
                key=f"{_SCOPE_PREFIX}_tenor_fly",
                format_func=lambda t: f"{t}M",
            )
    with cc3:
        lookbacks_picked = st.multiselect(
            "Lookbacks (multi-select)", options=_LOOKBACK_OPTIONS,
            default=_LOOKBACK_OPTIONS, key=f"{_SCOPE_PREFIX}_lbs",
            help="Z-scores computed for every selected lookback. Confluence matrix "
                 "uses all selected.",
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
        show_composite = st.checkbox("Composite ranks", value=True,
                                      key=f"{_SCOPE_PREFIX}_show_comp")
    with vc5:
        analysis_on = st.toggle("Analysis side panel", value=True,
                                 key=f"{_SCOPE_PREFIX}_analysis_on")

    strategy_db = strategy_label.lower()
    lookbacks_tuple = tuple(sorted(set(lookbacks_picked)))
    with st.spinner(f"Computing Z + tests for {strategy_label.lower()}s..."):
        try:
            data = _build_zscore_data(
                strategy_db, tenor_months, snap_date.isoformat(),
                lookbacks_tuple, base_product=_BP)
        except Exception as e:
            st.error(f"Failed to build Z panel: {e}")
            with st.expander("Traceback"):
                import traceback as _tb
                st.code(_tb.format_exc())
            return

    contracts = data.get("contracts") or []
    if not contracts:
        st.warning(f"No live contracts for SRA {strategy_label} "
                    f"{f'{tenor_months}M' if tenor_months else ''}.")
        return

    panel = data.get("zscore") or {}
    if not panel:
        st.warning("Z panel could not be built (data may be sparse).")
        return

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

    chart_col, side_col = st.columns([3, 1.1])
    with chart_col:
        if show_ribbons:
            st.markdown(
                f"<div style='margin-top:0.4rem; font-size:0.85rem; font-weight:600; "
                f"color:var(--text-heading);'>Section ribbons · top-{top_k} most stretched at {focus_lookback}d</div>",
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
                "color:var(--text-heading);'>Z confluence matrix · multi-window stretch</div>",
                unsafe_allow_html=True,
            )
            try:
                _render_z_confluence(panel, contracts, list(lookbacks_tuple), top_n=25)
            except Exception as e:
                st.error(f"Z confluence matrix failed: {e}")

        st.markdown(
            "<div style='margin-top:0.8rem; font-size:0.85rem; font-weight:600; "
            "color:var(--text-heading);'>Cluster heatmap · |median z| and % extreme per (Section × Lookback)</div>",
            unsafe_allow_html=True,
        )
        try:
            _render_cluster_heatmap(panel, contracts, int(front_end), int(mid_end),
                                     list(lookbacks_tuple))
        except Exception as e:
            st.error(f"Cluster heatmap failed: {e}")

        st.markdown(
            "<div style='margin-top:0.8rem; font-size:0.85rem; font-weight:600; "
            "color:var(--text-heading);'>Universe Z-distribution</div>",
            unsafe_allow_html=True,
        )
        try:
            _render_distribution(panel, contracts, focus_lookback)
        except Exception as e:
            st.error(f"Distribution failed: {e}")

        if show_composite:
            st.markdown(
                "<div style='margin-top:0.8rem; font-size:0.85rem; font-weight:600; "
                "color:var(--text-heading);'>Composite candidate rankings</div>",
                unsafe_allow_html=True,
            )
            try:
                _render_composite_rankings(panel, contracts, int(top_k))
            except Exception as e:
                st.error(f"Composite rankings failed: {e}")

        try:
            _render_drill_down(_SCOPE_PREFIX, contracts, snap_date, panel, focus_lookback)
        except Exception as e:
            st.error(f"Drill-down failed: {e}")

    with side_col:
        if analysis_on:
            # Phase H: visible side-panel border for parity across subtabs
            with st.container(height=720, border=True):
                try:
                    _render_combined_side_panel(data, contracts, focus_lookback,
                                                  int(front_end), int(mid_end))
                except Exception as e:
                    st.error(f"Side panel failed: {e}")
                    with st.expander("Traceback"):
                        import traceback as _tb
                        st.code(_tb.format_exc())
