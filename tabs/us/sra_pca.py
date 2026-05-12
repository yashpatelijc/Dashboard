"""SRA › PCA — Trade Screener (v3, trade-first design).

Single-screen layout:
    1. Status bar (one line: snapshot · #tradeable · alerts)
    2. Filter strip (search · sort · gate)
    3. Ranked trade table — one row per idea, click row → trade drilldown
    4. Contract symbols throughout are CLICKABLE → opens deep contract dossier
    5. Collapsed "engine status" expander at the bottom

PCA mechanics power everything underneath; the trader never sees PC charts,
loadings tables, regime IDs, or variance ratios on the home screen. The 14-section
contract dossier is the deepest drill-in: every analysis the engine has run on
that one contract, presented as a trader's dossier.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.charts import (
    BG_BASE, BORDER_DEFAULT, BORDER_SUBTLE,
)
from lib.fomc import (
    is_quarterly, load_fomc_meetings, parse_sra_outright,
    reference_period, third_wednesday,
)
from lib.pca import (
    build_full_pca_panel, eigenspectrum_gap, variance_ratio_regime,
    pc1_loading_asymmetry, _pchip_curve,
)
from lib.pca_analogs import (
    analog_fv_band, batch_analog_fv, knn_analog_search, ledoit_wolf_shrinkage,
)
from lib.pca_dossier import contract_dossier_data
from lib.pca_events import calendar_dummy_regression, event_drift_table
from lib.pca_regimes import fit_regime_stack
from lib.pca_trades import (
    SOURCE_NAMES, generate_all_trade_ideas, generate_execution_ticket,
    load_positions,
)
from lib.sra_data import (
    DEFAULT_FRONT_END, DEFAULT_MID_END, compute_pack_groups,
    get_outrights, get_sra_snapshot_latest_date, load_sra_curve_panel,
    pivot_curve_panel,
)
from lib.theme import (
    ACCENT, AMBER, BLUE, CYAN, GREEN, PURPLE, RED,
    TEXT_BODY, TEXT_DIM, TEXT_HEADING, TEXT_MUTED,
)


_SCOPE = "sra_pca_v3"


# =============================================================================
# Helpers — pills, formatting, click handlers
# =============================================================================
def _pill(text: str, color: str = "var(--text-body)", *, bold: bool = True) -> str:
    weight = "600" if bold else "400"
    return (f"<span style='display:inline-block; padding:1px 6px; "
            f"border:1px solid {color}; color:{color}; "
            f"font-size:0.65rem; font-family:JetBrains Mono, monospace; "
            f"font-weight:{weight}; border-radius:4px; line-height:1.2;'>{text}</span>")


def _conv_bar(c: float) -> str:
    """7-block conviction bar."""
    n_full = int(round(c * 7))
    full = "▰" * n_full
    empty = "▱" * (7 - n_full)
    color = ("var(--green)" if c >= 0.7
              else "var(--amber)" if c >= 0.5
              else "var(--text-dim)")
    return f"<span style='color:{color}; font-family:monospace;'>{full}{empty}</span>"


def _fmt_bp(x: Optional[float], decimals: int = 2) -> str:
    if x is None or pd.isna(x) or not np.isfinite(x):
        return "—"
    return f"{x:+.{decimals}f}"


def _fmt_dollar(x: Optional[float]) -> str:
    if x is None or pd.isna(x) or not np.isfinite(x):
        return "—"
    return f"${x:+,.0f}"


def _set_contract(symbol: str) -> None:
    st.session_state[f"{_SCOPE}_contract"] = symbol


def _clear_contract() -> None:
    st.session_state.pop(f"{_SCOPE}_contract", None)


def _section_header(text: str, concept_key: str = None) -> None:
    """Section header. If `concept_key` is provided, adds a ⓘ hover-tooltip
    linking to the concept glossary."""
    tooltip_attr = ""
    if concept_key:
        try:
            from lib.pca_concepts import concept_tooltip_html
            tt = concept_tooltip_html(concept_key)
            if tt:
                tooltip_attr = (f" title='{tt}'"
                                .replace("'", "&#39;"))
        except Exception:
            pass
    icon = (f"<span style='color:var(--blue); margin-left:0.4rem; "
              f"cursor:help; font-size:0.65rem;'{tooltip_attr}>ⓘ</span>"
              if concept_key else "")
    st.markdown(
        f"<div style='font-size:0.7rem; color:var(--text-dim); text-transform:uppercase; "
        f"letter-spacing:0.06em; margin: 0.5rem 0 0.25rem 0; font-weight:600;'>"
        f"{text}{icon}</div>",
        unsafe_allow_html=True,
    )


def _kv_block(rows: list) -> None:
    parts = []
    for tup in rows:
        label = tup[0]
        value = tup[1]
        cls = tup[2] if len(tup) > 2 else None
        sub = tup[3] if len(tup) > 3 else None
        color = (
            "var(--accent)" if cls == "accent" else
            "var(--green)" if cls == "green" else
            "var(--red)" if cls == "red" else
            "var(--amber)" if cls == "amber" else
            "var(--blue)" if cls == "blue" else
            "var(--purple)" if cls == "purple" else
            "var(--text-body)"
        )
        sub_html = (f"<span style='color:var(--text-dim); font-size:0.7rem; "
                     f"font-family:JetBrains Mono, monospace; margin-left:0.4rem;'>{sub}</span>"
                     if sub else "")
        parts.append(
            f"<div style='display:flex; justify-content:space-between; align-items:center; "
            f"padding:3px 0; border-bottom:1px solid var(--border-subtle);'>"
            f"<span style='color:var(--text-muted); font-size:0.78rem;'>{label}</span>"
            f"<span style='color:{color}; font-family:JetBrains Mono, monospace; "
            f"font-size:0.78rem; font-weight:500; text-align:right;'>{value}{sub_html}</span>"
            f"</div>"
        )
    st.markdown("".join(parts), unsafe_allow_html=True)


def _no_title(fig: go.Figure) -> go.Figure:
    fig.update_layout(title=dict(text="", x=0.0, xanchor="left"))
    return fig


# =============================================================================
# Cached engine panel
# =============================================================================
@st.cache_data(show_spinner=False, ttl=600)
def _build_engine_panel(asof_str: str, mode: str = "positional",
                          weekly_smooth: bool = False,
                          history_days: Optional[int] = None,
                          base_product: str = "SRA") -> dict:
    """Build the full PCA panel with mode-aware horizon parameters.

    Mode (``"intraday" | "swing" | "positional"``) controls residual_lookback,
    triple_gate_lookback, z_threshold, hold horizons, and detrend window. See
    ``lib.pca.MODE_PARAMS``.

    `weekly_smooth=True` enables weekly resampling of residuals — used by the
    "Weekly smoothing" toggle in positional mode for less noisy signals.
    """
    asof = pd.Timestamp(asof_str).date()
    resample = "W" if weekly_smooth else "D"
    panel = build_full_pca_panel(asof, mode=mode, resample=resample,
                                    history_days=history_days,
                                    base_product=base_product)
    if panel.get("pca_fit_static") is None:
        return panel

    # Phase 2: regime stack
    try:
        pc_panel = panel.get("pc_panel", pd.DataFrame()).dropna()
        anchor = panel.get("anchor_series", pd.Series(dtype=float)).dropna()
        if not pc_panel.empty and not anchor.empty and {"PC1", "PC2", "PC3"}.issubset(pc_panel.columns):
            pc1_vol = pc_panel["PC1"].rolling(20, min_periods=5).std()
            features_df = pd.DataFrame({
                "PC1": pc_panel["PC1"], "PC2": pc_panel["PC2"], "PC3": pc_panel["PC3"],
                "Anchor": anchor.reindex(pc_panel.index),
                "sigma_PC1_20d": pc1_vol,
            }).dropna()
            if len(features_df) >= 60:
                panel["regime_stack"] = fit_regime_stack(
                    features_df, k=6, n_restarts=10, max_iter=100, hmm_iter=3,
                )
                panel["features_df"] = features_df
    except Exception:
        panel["regime_stack"] = {}

    # Central-bank meeting calendar — market-aware (Fed/ECB/BoE/RBA/BoC/SNB)
    try:
        if base_product == "SRA":
            fomc_df = load_fomc_meetings()
        else:
            from importlib import import_module
            from lib.markets import get_market as _gm
            _cfg = _gm(base_product)
            _cb_mod = import_module(_cfg.get("central_bank_module", "lib.fomc"))
            _cb_fn = getattr(_cb_mod, _cfg.get("central_bank_decision_fn",
                                                  "load_fomc_meetings"))
            fomc_df = _cb_fn()
        panel["fomc_calendar_dates"] = (list(fomc_df["decision_date"])
                                          if not fomc_df.empty else [])
    except Exception:
        panel["fomc_calendar_dates"] = []

    # A12d event drift on Anchor (representative)
    try:
        anchor = panel.get("anchor_series")
        if anchor is not None and not anchor.empty and panel["fomc_calendar_dates"]:
            panel["event_drift_table"] = event_drift_table(
                anchor, event_dates=panel["fomc_calendar_dates"],
                event_class="FOMC", fdr_alpha=0.10,
            )
    except Exception:
        panel["event_drift_table"] = pd.DataFrame()

    # Seasonality
    try:
        anchor = panel.get("anchor_series")
        if anchor is not None and not anchor.empty and panel["fomc_calendar_dates"]:
            panel["seasonality_results"] = {
                "Anchor": calendar_dummy_regression(
                    anchor, fomc_dates=panel["fomc_calendar_dates"],
                    fdr_alpha=0.05, min_beta_bp=1.0,
                )
            }
    except Exception:
        panel["seasonality_results"] = {}

    panel["positions"] = load_positions()

    # Phase 5 — empirical hit rates from cached backtest (if available).
    # The backtest writes to D:\STIRS_DASHBOARD\cache\backtest_empirical_hit_rates_{mode}.json
    # If the file exists and is < 7 days old, load it; else leave as empty dict.
    try:
        import json
        from pathlib import Path
        cache_dir = Path("D:/STIRS_DASHBOARD/cache")
        cache_file = cache_dir / f"backtest_empirical_hit_rates_{panel.get('mode', 'positional')}.json"
        if cache_file.exists():
            mtime = pd.Timestamp(cache_file.stat().st_mtime, unit="s")
            if (pd.Timestamp.now() - mtime).days < 7:
                with cache_file.open("r", encoding="utf-8") as f:
                    panel["empirical_hit_rates"] = json.load(f)
    except Exception:
        pass

    return panel


# =============================================================================
# Main render
# =============================================================================
def render(base_product: str = "SRA") -> None:
    """Render the Trade Screener for any market. Defaults to SRA for back-compat.

    Multi-market wrappers (Eurozone/UK/Australia/Canada pages) pass their own
    base_product code (ER/FSR/FER/SON/YBA/CRA) and the whole engine retargets
    to that market — same UI, same generators, market-specific data + central
    bank + Bloomberg indicators.
    """
    from lib.markets import get_market as _get_market
    market_cfg = _get_market(base_product)
    market_label = market_cfg.get("code", base_product)
    market_desc = market_cfg.get("description", base_product)
    # Header
    st.markdown(
        f"""
        <div style="display:flex; align-items:baseline; gap:0.75rem; margin-top:0.25rem;">
            <span style="font-size:1.05rem; font-weight:600; color:var(--text-heading);">
                {market_label} · Trade Screener</span>
            <span style="color:var(--text-dim); font-size:0.85rem;">·</span>
            <span style="color:var(--text-muted); font-size:0.85rem;">
                {market_desc} — ranked tradeable opportunities · click any contract for deep dossier</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Market-aware snapshot lookup
    if base_product == "SRA":
        snapshot = get_sra_snapshot_latest_date()
    else:
        from lib.market_data import get_snapshot_latest_date as _gsl
        snapshot = _gsl(base_product)
    if snapshot is None:
        st.error("OHLC database not available — see Settings → OHLC DB Viewer.")
        return

    # ─── Trade Horizon Mode toggle ─────────────────────────────────────────
    # Positional default (user is a positional STIRS trader). User may switch
    # to Swing or Intraday on demand. All horizon-sensitive engine params
    # (residual lookback, OU sweet-spot, z-threshold, hold caps) are derived
    # from this single selector via lib.pca.MODE_PARAMS.
    # Use the widget key as the single source of truth (Streamlit pattern —
    # avoid the index=/value= + key= double-binding that causes warnings).
    mode_key = f"{_SCOPE}_mode_radio"
    smooth_key = f"{_SCOPE}_weekly_check"
    if mode_key not in st.session_state:
        st.session_state[mode_key] = "positional"
    if smooth_key not in st.session_state:
        st.session_state[smooth_key] = False

    mode_col, smooth_col, info_col = st.columns([2, 1.5, 4])
    with mode_col:
        mode = st.radio(
            "Trade Horizon",
            options=["intraday", "swing", "positional"],
            format_func=lambda m: {
                "intraday": "Intraday (HL 0.5-7d, 1σ)",
                "swing": "Swing (HL 3-30d, 1.5σ)",
                "positional": "Positional (HL 15-90d, 2σ)",
            }[m],
            horizontal=True,
            key=mode_key,
        )
    with smooth_col:
        weekly = st.checkbox(
            "Weekly smoothing",
            help=("Resample residuals to weekly before fitting OU/z. Reduces "
                  "noise for positional view. Only useful in positional mode."),
            key=smooth_key,
            disabled=(mode != "positional"),
        )
    with info_col:
        from lib.pca import mode_params as _mp
        mp = _mp(mode)
        st.markdown(
            f"<div style='font-size:0.78rem; color:var(--text-muted); padding-top:0.3rem;'>"
            f"Residual lookback: <b>{mp['residual_lookback']}d</b> · "
            f"OU sweet spot: <b>{mp['sweet_spot_full'][0]:.0f}-{mp['sweet_spot_full'][1]:.0f}d</b> · "
            f"Hold: <b>{mp['hold_floor']}-{mp['hold_cap']}d</b> · "
            f"History: <b>{mp['history_days']}d</b>"
            f"</div>",
            unsafe_allow_html=True,
        )

    effective_weekly = bool(weekly) if mode == "positional" else False
    with st.spinner(f"Running engine ({base_product} · {mode} mode · PCA + generators)..."):
        panel = _build_engine_panel(snapshot.isoformat(), mode=mode,
                                       weekly_smooth=effective_weekly,
                                       base_product=base_product)
    if panel is None or panel.get("pca_fit_static") is None:
        st.warning("Engine not ready — likely insufficient post-2023-08-01 sample.")
        return

    with st.spinner("Generating trade ideas..."):
        ideas = generate_all_trade_ideas(panel, snapshot)

    # Status row (one line, minimal)
    _render_status_row(snapshot, panel, ideas)

    # Phase G.2 — Engine Health pill strip at TOP (was bottom expander)
    _render_engine_health_strip(panel, ideas)

    # Phase G.3 — Cross-asset context strip at TOP
    _render_cross_asset_strip(panel)

    # If a contract is selected, show its dossier (top of the page)
    selected_contract = st.session_state.get(f"{_SCOPE}_contract")
    if selected_contract:
        _render_contract_dossier(selected_contract, panel, ideas)
        st.markdown("---")

    # Filter / sort strip
    filtered = _render_filter_strip_and_filter(ideas)

    # Main trade table
    _render_trade_table(filtered, panel)

    # Phase E.5 — Probabilistic FOMC card surfaces from existing A4 step-path
    _render_fomc_probability_card(panel)

    # Engine state details (collapsed — supplementary to Engine Health strip above)
    with st.expander("🔧 detailed diagnostics + engine-state signals",
                       expanded=False):
        _render_engine_status(panel, ideas)
        # Phase A.2 — engine-state signals (informational overlays)
        _render_engine_state_signals(panel)

    # Phase 5 / Part C — engine backtest (collapsed)
    _render_backtest_section(panel)


# =============================================================================
# Status row
# =============================================================================
def _render_status_row(snapshot: date, panel: dict, ideas: list) -> None:
    n_ideas = len(ideas)
    n_clean = sum(1 for i in ideas if i.gate_quality == "clean")
    n_alerts = sum(1 for i in ideas if i.gate_quality == "regime_unstable")
    pct_today = panel.get("reconstruction_pct_today")
    pct_str = f"{pct_today * 100:.0f}%" if pct_today is not None else "—"
    pct_color = ("var(--green)" if pct_today is not None and pct_today >= 0.95
                  else "var(--amber)" if pct_today is not None and pct_today >= 0.9
                  else "var(--red)")
    chips = [
        f"<span style='color:var(--green); font-size:0.95rem;'>●</span> live",
        f"snapshot <b style='color:var(--text-heading);'>{snapshot}</b>",
        f"tradeable <b style='color:var(--green);'>{n_clean}</b>/{n_ideas}",
        f"alerts <b style='color:var(--amber);'>{n_alerts}</b>" if n_alerts else
            "alerts <b style='color:var(--green);'>0</b>",
        f"engine <b style='color:{pct_color};'>{pct_str}</b>",
    ]
    st.markdown(
        f"<div style='display:flex; gap:1.5rem; align-items:center; "
        f"font-family:JetBrains Mono, monospace; font-size:0.78rem; "
        f"color:var(--text-muted); padding:0.4rem 0.6rem; "
        f"border:1px solid var(--border-subtle); border-radius:4px; "
        f"margin: 0.5rem 0;'>"
        f"{' · '.join(chips)}</div>",
        unsafe_allow_html=True,
    )


# =============================================================================
# Concept Glossary expander (pedagogical disclosure for new users)
# =============================================================================
def _render_concept_glossary() -> None:
    """Top-of-page expander linking to every concept the engine uses."""
    with st.expander("📚 What do these numbers mean? · Concept glossary "
                       "(z-score, PC1/2/3, ADF/KPSS, half-life, analog FV, ...)",
                       expanded=False):
        from lib.pca_concepts import CONCEPTS, concept_disclosure_html
        st.markdown(
            "<div style='font-size:0.78rem; color:var(--text-muted); "
            "margin-bottom:0.6rem; line-height:1.6;'>"
            "Every metric in the engine has a precise definition and a "
            "specific interpretation for trading. This glossary covers each "
            "concept used in the trade screener + dossier — math, plain-English, "
            "and academic source.  <b>Open any one to read the full disclosure.</b>"
            "</div>",
            unsafe_allow_html=True,
        )
        # Group concepts by category
        cats = {}
        for k, c in CONCEPTS.items():
            cat = c.get("category", "misc")
            cats.setdefault(cat, []).append((k, c))
        cat_labels = {
            "factor": "1. Factor concepts (PC1, PC2, PC3, residuals)",
            "test": "2. Statistical tests (ADF, KPSS, VR)",
            "statistic": "3. Statistics (z-score, half-life, Hurst, eff_n, …)",
            "model": "4. Models (analog FV, path FV, HMM, ACM, H-P step path)",
            "framework": "5. Composite frameworks (triple-gate, cycle, vol/risk regimes, conviction)",
        }
        for cat in ("factor", "test", "statistic", "model", "framework"):
            entries = cats.get(cat, [])
            if not entries:
                continue
            st.markdown(
                f"<div style='font-size:0.75rem; color:var(--accent); "
                f"text-transform:uppercase; letter-spacing:0.06em; "
                f"font-weight:600; margin: 0.6rem 0 0.3rem 0; "
                f"padding-top:0.3rem; border-top:1px solid var(--border-subtle);'>"
                f"{cat_labels.get(cat, cat)}</div>",
                unsafe_allow_html=True,
            )
            for k, c in entries:
                with st.expander(f"• {c['name']} — {c['one_liner']}",
                                   expanded=False):
                    st.markdown(concept_disclosure_html(k),
                                 unsafe_allow_html=True)


# =============================================================================
# Phase G.2 — Engine Health pill strip (top of page)
# =============================================================================
def _render_engine_health_strip(panel: dict, ideas: list) -> None:
    """Compact pill strip showing today's engine state. Replaces bottom-expander."""
    fit = panel.get("pca_fit_static")
    rs = panel.get("regime_stack", {}) or {}
    hmm = rs.get("hmm_fit") if rs else None
    pct_today = panel.get("reconstruction_pct_today")
    cross_corr = panel.get("cross_pc_corr", pd.DataFrame())
    print_alerts = panel.get("print_quality_alerts", []) or []
    asof = panel.get("asof", date.today())

    chips = []
    # 3-PC explained
    if pct_today is not None:
        col = ("var(--green)" if pct_today >= 0.95
                else "var(--amber)" if pct_today >= 0.90 else "var(--red)")
        chips.append(_pill(f"3-PC explained: {pct_today * 100:.0f}%", col))
    # Cross-PC corr today
    if not cross_corr.empty:
        try:
            last = cross_corr.iloc[-1].dropna()
            if not last.empty:
                m = float(last.abs().max())
                col = "var(--green)" if m <= 0.30 else "var(--amber)" if m <= 0.50 else "var(--red)"
                chips.append(_pill(f"cross-PC corr: {m:.2f}", col))
        except Exception:
            pass
    # Regime confidence
    if hmm is not None and hmm.dominant_confidence is not None and len(hmm.dominant_confidence) > 0:
        conf = float(hmm.dominant_confidence[-1])
        col = "var(--green)" if conf >= 0.6 else "var(--amber)"
        chips.append(_pill(f"regime: {conf*100:.0f}% conf", col))
    # Print quality
    today_print_flagged = any(pd.Timestamp(d).date() == asof for d in print_alerts)
    if today_print_flagged:
        chips.append(_pill("⚠ print quality flagged", "var(--red)"))
    else:
        chips.append(_pill("print: clean", "var(--green)"))
    # # ideas
    chips.append(_pill(f"trades: {len(ideas)}", "var(--text-body)"))
    n_clean = sum(1 for i in ideas if i.gate_quality == "clean")
    chips.append(_pill(f"clean: {n_clean}", "var(--green)"))
    # # engine-state signals
    es_count = len(panel.get("engine_state_signals", []) or [])
    if es_count:
        chips.append(_pill(f"engine signals: {es_count}", "var(--blue)"))
    # Refit age
    rolling = panel.get("rolling_fits", {}) or {}
    if rolling:
        latest_refit = max(rolling.keys())
        if asof is not None:
            try:
                age = (asof - latest_refit).days
                col = "var(--green)" if age <= 5 else "var(--amber)"
                chips.append(_pill(f"refit: {age}d old", col))
            except Exception:
                pass

    st.markdown(
        f"<div style='display:flex; gap:0.4rem; flex-wrap:wrap; "
        f"padding:0.4rem 0.6rem; margin: 0.25rem 0; "
        f"background:rgba(255,255,255,0.02); "
        f"border:1px solid var(--border-subtle); border-radius:4px; "
        f"font-family:JetBrains Mono, monospace; font-size:0.72rem;'>"
        f"<span style='color:var(--text-dim); margin-right:0.5rem; align-self:center;'>"
        f"🔧 ENGINE</span>"
        f"{' '.join(chips)}</div>",
        unsafe_allow_html=True,
    )


# =============================================================================
# Phase G.3 — Cross-asset context strip (top of page)
# =============================================================================
def _render_cross_asset_strip(panel: dict) -> None:
    """Tape state strip: vol regime, risk state, credit cycle, key cross-assets."""
    ca = panel.get("cross_asset_analysis")
    if ca is None:
        # Trigger lazy compute via dossier path — only happens once per session
        try:
            from lib.pca_dossier import _section_cross_asset
            cross = _section_cross_asset(panel)
        except Exception:
            cross = {}
    else:
        cross = {
            "vol_regime": ca.vol_regime,
            "risk_state": ca.risk_state,
            "credit_cycle": ca.credit_cycle,
        }

    chips = []
    vol = cross.get("vol_regime") or {}
    risk = cross.get("risk_state") or {}
    cred = cross.get("credit_cycle") or {}

    # Vol regime
    if vol.get("regime") and vol["regime"] != "unknown":
        regime = vol["regime"]
        z = vol.get("composite_z", 0)
        col = ({"quiet": "var(--green)", "normal": "var(--text-body)",
                "stressed": "var(--amber)", "crisis": "var(--red)"}.get(regime, "var(--text-body)"))
        chips.append(_pill(f"vol {regime} (z={z:+.2f})", col))
    # Risk state
    if risk.get("state") and risk["state"] != "neutral":
        state = risk["state"]
        col = ({"risk_on": "var(--green)", "neutral": "var(--text-body)",
                "risk_off": "var(--amber)", "panic": "var(--red)"}.get(state, "var(--text-body)"))
        score = risk.get("score", 0)
        chips.append(_pill(f"risk {state} (z={score:+.2f})", col))
    # Recession prob
    if cred.get("recession_prob_4w") is not None:
        rp = cred["recession_prob_4w"]
        col = "var(--red)" if rp > 0.4 else "var(--amber)" if rp > 0.2 else "var(--green)"
        chips.append(_pill(f"recession prob 4w: {rp*100:.0f}%", col))
    # IG/HY divergence flag
    if cred.get("ig_hy_diverging"):
        chips.append(_pill("⚠ IG/HY diverging", "var(--amber)"))

    # Individual cross-asset chips from panel cross_asset_panel
    cap = panel.get("cross_asset_panel")
    if cap is not None:
        if not cap.equity.empty and "SPX" in cap.equity.columns:
            try:
                spx_now = float(cap.equity["SPX"].dropna().iloc[-1])
                spx_5d = float(cap.equity["SPX"].dropna().iloc[-6])
                chg = (spx_now / spx_5d - 1.0) * 100
                col = "var(--green)" if chg > 0.5 else "var(--red)" if chg < -0.5 else "var(--text-body)"
                chips.append(_pill(f"SPX {chg:+.1f}%/5d", col))
            except Exception:
                pass
        if not cap.vol.empty and "MOVE" in cap.vol.columns:
            try:
                m = float(cap.vol["MOVE"].dropna().iloc[-1])
                chips.append(_pill(f"MOVE {m:.0f}", "var(--text-body)"))
            except Exception:
                pass
        if not cap.fx.empty and "DXY_synth" in cap.fx.columns:
            try:
                d = float(cap.fx["DXY_synth"].dropna().iloc[-1])
                chips.append(_pill(f"DXY {d:.1f}", "var(--text-body)"))
            except Exception:
                pass

    if not chips:
        return
    st.markdown(
        f"<div style='display:flex; gap:0.4rem; flex-wrap:wrap; "
        f"padding:0.4rem 0.6rem; margin: 0.25rem 0; "
        f"background:rgba(94,180,255,0.04); "
        f"border:1px solid var(--blue); border-radius:4px; "
        f"font-family:JetBrains Mono, monospace; font-size:0.72rem;'>"
        f"<span style='color:var(--blue); margin-right:0.5rem; align-self:center; font-weight:500;'>"
        f"🌐 TAPE</span>"
        f"{' '.join(chips)}</div>",
        unsafe_allow_html=True,
    )


# =============================================================================
# Phase A.2 — Engine state signals panel (in expander)
# =============================================================================
def _render_engine_state_signals(panel: dict) -> None:
    """Render the engine-state signals (overlay only, non-tradeable)."""
    signals = panel.get("engine_state_signals", []) or []
    if not signals:
        st.markdown(
            "<div style='color:var(--text-dim); font-size:0.78rem;'>"
            "No engine-state signals firing today.</div>",
            unsafe_allow_html=True,
        )
        return
    _section_header("Engine-state signals (informational, no executable legs)")
    for s in signals:
        # rationale_html is populated; just render it inline
        st.markdown(
            f"<div style='padding:6px 8px; margin: 4px 0; "
            f"border-left:3px solid var(--blue); "
            f"background:rgba(94,180,255,0.04); "
            f"border-radius:0 4px 4px 0; font-size:0.78rem;'>"
            f"<div style='color:var(--blue); font-size:0.7rem; "
            f"text-transform:uppercase; font-weight:500; margin-bottom:0.3rem;'>"
            f"{s.primary_source}</div>"
            f"{s.rationale_html}</div>",
            unsafe_allow_html=True,
        )


# =============================================================================
# Phase E.5 — Probabilistic FOMC card
# =============================================================================
def _render_fomc_probability_card(panel: dict) -> None:
    """Surface A4 step-path probabilities for the next FOMC."""
    asof = panel.get("asof", date.today())
    fomc_dates = panel.get("fomc_calendar_dates", []) or []
    upcoming_fomc = sorted(pd.Timestamp(d).date() for d in fomc_dates
                            if pd.Timestamp(d).date() > asof)
    if not upcoming_fomc:
        return

    # Try to compute step-path probs from cmc_panel — reuse existing fit
    try:
        from lib.pca_step_path import fit_step_path_bootstrap
        step_path = fit_step_path_bootstrap(
            cmc_panel=panel.get("cmc_panel", pd.DataFrame()),
            fomc_dates=fomc_dates,
            asof=asof, n_meetings=3, n_draws=10, kernel_sigma_bp=37.5,
        )
        meetings = step_path.get("meetings", [])
    except Exception:
        meetings = []

    if not meetings:
        return

    # Market-aware central-bank name for the card title
    _cb = str((panel.get("market") or {}).get("central_bank", "Fed"))
    st.markdown(
        f"<div style='font-size:0.72rem; color:var(--text-dim); "
        f"text-transform:uppercase; letter-spacing:0.06em; margin: 0.6rem 0 0.2rem 0;"
        f"font-weight:600;'>"
        f"🎯 {_cb} step-path probabilities (A4 Heitfield-Park)"
        f"</div>",
        unsafe_allow_html=True,
    )

    cols = st.columns(min(len(meetings), 3))
    for i, m in enumerate(meetings[:3]):
        d = m["decision_date"]
        offset_d = (d - asof).days
        probs = m["probs"]
        impl = m["implied_bp_mean"]
        with cols[i]:
            bars = []
            for b in ("large_cut", "cut", "hold", "hike", "large_hike"):
                pv = float(probs.get(b, 0.0))
                n_full = max(0, min(7, int(round(pv * 7))))
                bar = "▰" * n_full + "▱" * (7 - n_full)
                col = ("var(--green)" if b in ("large_cut", "cut")
                        else "var(--red)" if b in ("hike", "large_hike")
                        else "var(--text-body)")
                bars.append(
                    f"<div style='display:flex; justify-content:space-between; "
                    f"font-size:0.7rem; padding:1px 0;'>"
                    f"<span style='color:var(--text-muted);'>{b.replace('_', ' '):<11}</span>"
                    f"<span style='color:{col}; font-family:monospace;'>{bar} {pv*100:5.1f}%</span></div>"
                )
            st.markdown(
                f"<div style='padding:0.6rem 0.8rem; "
                f"border:1px solid var(--border-subtle); border-radius:6px; "
                f"background:rgba(255,255,255,0.015);'>"
                f"<div style='font-size:0.78rem; color:var(--accent); font-weight:500; "
                f"margin-bottom:0.3rem;'>{d} <span style='color:var(--text-dim); "
                f"font-size:0.7rem; font-weight:400;'>(in {offset_d}d)</span></div>"
                f"<div style='font-size:0.65rem; color:var(--text-dim); margin-bottom:0.4rem;'>"
                f"implied Δ = {impl:+.2f} bp</div>"
                f"{''.join(bars)}</div>",
                unsafe_allow_html=True,
            )


# =============================================================================
# Filter strip
# =============================================================================
def _render_filter_strip_and_filter(ideas: list) -> list:
    # Row 1 — Structure type chips (multi-select) — outright / spread / fly / pack / basket
    available_structures = sorted({i.structure_type or "other" for i in ideas})
    structure_label = {"outright": "📊 Outrights", "spread": "📐 Spreads",
                          "fly": "🦋 Flies", "pack": "📦 Packs",
                          "basket": "🎯 Baskets", "other": "✨ Other"}
    st.markdown(
        "<div style='font-size:0.7rem; color:var(--text-dim); "
        "text-transform:uppercase; letter-spacing:0.06em; margin: 0.6rem 0 0.2rem 0; "
        "font-weight:600;'>📂 Trade type filter</div>",
        unsafe_allow_html=True,
    )
    pinned_structures = st.multiselect(
        "structure types to show",
        options=available_structures,
        default=available_structures,
        format_func=lambda s: structure_label.get(s, s),
        key=f"{_SCOPE}_structures",
        label_visibility="collapsed",
    )
    # Row 2 — secondary filters
    cols = st.columns([1.4, 1.2, 1.4, 1.0, 0.6])
    with cols[0]:
        search = st.text_input("🔍 filter by contract symbol",
                                  value="", placeholder="e.g. SRAM26",
                                  key=f"{_SCOPE}_search")
    with cols[1]:
        sort_by = st.selectbox(
            "sort by",
            options=["conviction", "|z|", "expected $", "half-life", "expected hold"],
            index=0, key=f"{_SCOPE}_sort",
        )
    with cols[2]:
        gates = st.multiselect(
            "gates",
            options=["clean", "low_n", "non_stationary", "regime_unstable", "circular_proxy"],
            default=["clean"], key=f"{_SCOPE}_gates",
        )
    with cols[3]:
        min_conv = st.slider("min conviction", 0.0, 1.0, 0.4, 0.05,
                                 key=f"{_SCOPE}_min_conv")
    with cols[4]:
        max_show = st.number_input("top N (per group)", 1, 200, 5, step=1,
                                       key=f"{_SCOPE}_max",
                                       help="Each card is fully expanded so keep this small")

    out = [
        i for i in ideas
        if i.conviction >= min_conv
        and (not gates or i.gate_quality in gates)
        and ((i.structure_type or "other") in pinned_structures)
    ]
    if search.strip():
        s = search.strip().upper()
        out = [i for i in out if any(s in leg.symbol.upper() for leg in i.legs)
                or s in i.primary_source.upper()]

    if sort_by == "conviction":
        out.sort(key=lambda i: -i.conviction)
    elif sort_by == "|z|":
        out.sort(key=lambda i: -abs(i.z_score) if i.z_score is not None else 0)
    elif sort_by == "expected $":
        out.sort(key=lambda i: -(i.expected_pnl_dollar or 0))
    elif sort_by == "half-life":
        out.sort(key=lambda i: i.half_life_d or 999)
    elif sort_by == "expected hold":
        out.sort(key=lambda i: i.expected_revert_d or 999)

    # Top-N per structure group (so user gets a representative card of each type)
    grouped_top = []
    seen_per_group = {}
    for idea in out:
        gkey = idea.structure_type or "other"
        seen_per_group.setdefault(gkey, 0)
        if seen_per_group[gkey] < int(max_show):
            grouped_top.append(idea)
            seen_per_group[gkey] += 1
    return grouped_top


# =============================================================================
# Trade table
# =============================================================================
def _trade_label(idea) -> str:
    """Compact human label for a trade."""
    syms = [leg.symbol for leg in idea.legs]
    if not syms:
        return idea.primary_source
    if idea.structure_type == "fly":
        return f"{'/'.join(syms[:3])} fly"
    if idea.structure_type == "spread":
        return f"{'-'.join(syms[:2])} spread"
    if idea.structure_type == "outright":
        return f"{syms[0]} outright"
    if idea.structure_type == "pack":
        return f"{syms[0]} pack"
    if idea.structure_type == "basket":
        return f"{syms[0]}+ basket"
    return "/".join(syms[:3])


def _render_trade_table(filtered: list, panel: dict) -> None:
    """Card-based trade list — grouped by structure type, ranked within group,
    with conviction-driven visual prominence."""
    if not filtered:
        st.markdown(
            "<div style='color:var(--text-dim); padding:1rem; "
            "border:1px dashed var(--border-default); border-radius:6px; "
            "text-align:center;'>"
            "No trades pass current filters. Lower conviction threshold or relax gates."
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # Group by structure type for visual organization
    groups = {}
    for idea in filtered:
        key = idea.structure_type or "other"
        groups.setdefault(key, []).append(idea)

    group_order = ["outright", "spread", "fly", "pack", "basket", "other"]
    group_labels = {
        "outright": "📊 Outrights — single-contract residual fades",
        "spread":   "📐 Spreads — slope / calendar relative value",
        "fly":      "🦋 Flies — curvature / butterfly trades",
        "pack":     "📦 Packs — 4-contract bundle RV",
        "basket":   "🎯 Baskets — PCA-isolated factor exposures",
        "other":    "✨ Other structures",
    }

    # Global rank counter
    rank_counter = 0
    for gkey in group_order:
        if gkey not in groups:
            continue
        gideas = sorted(groups[gkey], key=lambda i: -i.conviction)
        st.markdown(
            f"<div style='font-size:0.8rem; color:var(--accent); "
            f"text-transform:uppercase; letter-spacing:0.06em; font-weight:600; "
            f"margin: 1rem 0 0.4rem 0; padding-bottom:0.3rem; "
            f"border-bottom:1px solid var(--accent);'>"
            f"{group_labels.get(gkey, gkey)} "
            f"<span style='color:var(--text-dim); font-size:0.7rem; font-weight:400;'>"
            f"· {len(gideas)} trade{'s' if len(gideas) != 1 else ''}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        for idea in gideas:
            rank_counter += 1
            _render_trade_card(rank_counter, idea, panel)


def _render_trade_card(rank: int, idea, panel: dict) -> None:
    """Each trade is rendered as a visually prominent CARD with:
      - Top strip: rank, label, action chip, conviction bar
      - Mini sparkline of recent residual (always visible)
      - Actual SR3 prices panel (entry/target/stop/$/c/RR)
      - 1-line headline italic
      - Expand button → full Trade Card with chart + narrative + factor breakdown
    """
    from lib.pca_trade_interpretation import convert_to_sr3_prices

    # Cache SR3 prices for this idea
    sr3_cache_key = f"_sr3_{idea.idea_id}"
    if sr3_cache_key not in panel:
        try:
            panel[sr3_cache_key] = convert_to_sr3_prices(idea, panel)
        except Exception:
            panel[sr3_cache_key] = None
    sr3 = panel[sr3_cache_key]

    # Conviction-driven visual prominence
    conv = idea.conviction
    if conv >= 0.70:
        border_color = "var(--green)"
        bg = "rgba(53,169,81,0.04)"
        tier_badge = "HIGH"
        tier_bg = "var(--green)"
    elif conv >= 0.55:
        border_color = "var(--accent)"
        bg = "rgba(232,183,93,0.04)"
        tier_badge = "MED"
        tier_bg = "var(--accent)"
    elif conv >= 0.40:
        border_color = "var(--amber)"
        bg = "rgba(232,141,60,0.03)"
        tier_badge = "LOW"
        tier_bg = "var(--amber)"
    else:
        border_color = "var(--text-dim)"
        bg = "rgba(255,255,255,0.015)"
        tier_badge = "•"
        tier_bg = "var(--text-dim)"

    label = _trade_label(idea)
    n_full = max(0, min(7, int(round(conv * 7))))
    conv_bar = "▰" * n_full + "▱" * (7 - n_full)

    # Resolve display strings from SR3 prices
    if sr3 is not None and sr3.is_outright and sr3.entry_price is not None:
        action_str = sr3.contract_action
        action_color = "var(--green)" if sr3.contract_action == "BUY" else "var(--red)"
        entry_str = f"{sr3.entry_price:.4f}"
        target_str = f"{sr3.target_price:.4f}"
        stop_str = f"{sr3.stop_price:.4f}"
        pnl_str = f"${sr3.pnl_per_contract_dollar:.0f}"
        risk_str = f"${sr3.risk_per_contract_dollar:.0f}"
        rr_str = f"{sr3.risk_reward:.2f}:1" if sr3.risk_reward else "—"
    elif sr3 is not None and not sr3.is_outright:
        # Multi-leg row preview — show simple BUY/SELL chip + actual CME contract
        # price (in bp-of-price-differential, the unit CME uses for spreads/flies).
        if sr3.price_source == "listed":
            # BUY/SELL chip (the package-level action — same semantics as outright)
            action_str = sr3.package_action or "—"
            action_color = "var(--green)" if action_str == "BUY" else "var(--red)"
            entry_str = f"{sr3.listed_entry_price:+.2f} bp"
            target_str = f"{sr3.listed_target_price:+.2f} bp"
            stop_str = f"{sr3.listed_stop_price:+.2f} bp"
            pnl_str = (f"${sr3.listed_pnl_dollar:.0f}"
                         if sr3.listed_pnl_dollar is not None else "—")
            risk_str = (f"${sr3.listed_risk_dollar:.0f}"
                          if sr3.listed_risk_dollar is not None else "—")
            rr_str = (f"{sr3.listed_risk_reward:.2f}:1"
                        if sr3.listed_risk_reward is not None else "—")
        elif sr3.price_source == "synthetic":
            action_str = sr3.package_action or "—"
            action_color = "var(--green)" if action_str == "BUY" else "var(--red)"
            entry_str = f"{sr3.synth_canonical_bp:+.2f} bp"
            target_str = f"{sr3.synth_target_bp:+.2f} bp"
            stop_str = f"{sr3.synth_stop_bp:+.2f} bp"
            pnl_str = (f"${sr3.synth_pnl_dollar:.0f}"
                         if sr3.synth_pnl_dollar is not None else "—")
            risk_str = (f"${sr3.synth_risk_dollar:.0f}"
                          if sr3.synth_risk_dollar is not None else "—")
            rr_str = (f"{sr3.synth_risk_reward:.2f}:1"
                        if sr3.synth_risk_reward is not None else "—")
        else:
            # per_leg — engine multi-leg structure with no listed product
            # equivalent (e.g. PC2-spread with 3 legs, PC1-basket with 4 legs).
            # Show BUY/SELL chip (derived from direction) so user has a
            # consistent action label across ALL trade types.
            action_str = sr3.package_action or f"{len(idea.legs)}-LEG"
            action_color = ("var(--green)" if action_str == "BUY"
                              else "var(--red)" if action_str == "SELL"
                              else "var(--text-muted)")
            entry_str = (f"{sr3.net_entry_bp:+.2f} bp/u"
                          if sr3.net_entry_bp is not None else "—")
            target_str = (f"{sr3.net_target_bp:+.2f} bp/u"
                           if sr3.net_target_bp is not None else "0.00 bp/u")
            stop_str = (f"{sr3.net_stop_bp:+.2f} bp/u"
                         if sr3.net_stop_bp is not None else "—")
            pnl_str = (f"${sr3.pnl_per_contract_dollar:.0f}"
                         if sr3.pnl_per_contract_dollar else "—")
            risk_str = (f"${sr3.risk_per_contract_dollar:.0f}"
                          if sr3.risk_per_contract_dollar else "—")
            rr_str = (f"{sr3.risk_reward:.2f}:1"
                        if sr3.risk_reward else "—")
    else:
        action_str = idea.direction[:4].upper()
        action_color = ("var(--green)" if idea.direction == "long"
                          else "var(--red)" if idea.direction == "short"
                          else "var(--text-body)")
        entry_str = _fmt_bp(idea.entry_bp)
        target_str = _fmt_bp(idea.target_bp)
        stop_str = _fmt_bp(idea.stop_bp)
        pnl_str = _fmt_dollar(idea.expected_pnl_dollar)
        risk_str = "—"
        rr_str = "—"

    # Build the visual card top strip (always-expanded mode — no toggle)
    st.markdown(
        f"<div style='border-left:4px solid {border_color}; "
        f"background:{bg}; padding:0.7rem 0.85rem; "
        f"margin: 0.5rem 0 0.2rem 0; border-radius:0 6px 6px 0;'>"
        # Top row: rank · tier badge · label · action chip · prices · conviction
        f"<div style='display:grid; "
        f"grid-template-columns: 0.4fr 0.5fr 1.7fr 0.7fr 0.85fr 0.85fr 0.85fr 0.65fr 0.65fr 0.65fr 1.0fr; "
        f"gap:0.5rem; align-items:center; font-family:JetBrains Mono, monospace;'>"
        # Rank
        f"<div style='font-size:0.78rem; color:var(--text-muted);'>#{rank}</div>"
        # Tier badge
        f"<div style='font-size:0.65rem; color:white; font-weight:700; "
        f"background:{tier_bg}; padding:0.15rem 0.3rem; border-radius:3px; "
        f"text-align:center;'>{tier_badge}</div>"
        # Label
        f"<div style='font-size:0.85rem; color:var(--text-heading); font-weight:500;'>"
        f"{label} <span style='color:var(--text-dim); font-size:0.7rem; font-weight:400;'>"
        f"({idea.primary_source})</span></div>"
        # Action chip
        f"<div style='font-size:0.78rem; color:{action_color}; font-weight:700; text-align:center;'>"
        f"{action_str}</div>"
        # Entry
        f"<div style='font-size:0.78rem; color:var(--text-body); text-align:right;'>"
        f"<span style='color:var(--text-dim); font-size:0.62rem; display:block;'>ENTRY</span>"
        f"{entry_str}</div>"
        # Target
        f"<div style='font-size:0.78rem; color:var(--green); text-align:right;'>"
        f"<span style='color:var(--text-dim); font-size:0.62rem; display:block;'>TARGET</span>"
        f"{target_str}</div>"
        # Stop
        f"<div style='font-size:0.78rem; color:var(--red); text-align:right;'>"
        f"<span style='color:var(--text-dim); font-size:0.62rem; display:block;'>STOP</span>"
        f"{stop_str}</div>"
        # P&L
        f"<div style='font-size:0.78rem; color:var(--green); text-align:right;'>"
        f"<span style='color:var(--text-dim); font-size:0.62rem; display:block;'>P&L/c</span>"
        f"{pnl_str}</div>"
        # Risk
        f"<div style='font-size:0.78rem; color:var(--red); text-align:right;'>"
        f"<span style='color:var(--text-dim); font-size:0.62rem; display:block;'>RISK/c</span>"
        f"{risk_str}</div>"
        # R:R
        f"<div style='font-size:0.78rem; color:var(--text-body); text-align:right;'>"
        f"<span style='color:var(--text-dim); font-size:0.62rem; display:block;'>R:R</span>"
        f"{rr_str}</div>"
        # Conviction
        f"<div style='font-size:0.72rem; color:var(--text-body); text-align:right;'>"
        f"<span style='color:var(--text-dim); font-size:0.62rem; display:block;'>CONVICTION</span>"
        f"<span style='color:{border_color};'>{conv_bar}</span> "
        f"<span style='color:var(--text-dim);'>{conv:.2f}</span></div>"
        f"</div>"
        + (
            (f"<div style='font-size:0.72rem; color:var(--text-muted); "
              f"font-style:italic; margin-top:0.3rem; padding-top:0.3rem; "
              f"border-top:1px solid var(--border-subtle);'>"
              f"{idea.headline_html}</div>")
            if idea.headline_html else ""
        )
        # Per-leg execution subtitle — shows ACTUAL CONTRACT PRICES for every
        # leg so user can verify the trade against the visible outright market.
        + _build_per_leg_execution_subtitle(idea, sr3)
        + "</div>",
        unsafe_allow_html=True,
    )

    # Per-card expand toggle — collapsed by default. Click the button under
    # each card to reveal: SR3 contract prices · 3 charts · dynamic narrative ·
    # per-factor analysis.
    expand_key = f"{_SCOPE}_exp_{idea.idea_id}"
    is_expanded = st.session_state.get(expand_key, False)
    if st.button(
        ("▾ Hide full analysis" if is_expanded
            else "▸ Expand full analysis (prices · charts · narrative · factors)"),
        key=f"{_SCOPE}_btn_{idea.idea_id}",
        use_container_width=True,
    ):
        st.session_state[expand_key] = not is_expanded
        st.rerun()
    if is_expanded:
        _render_trade_drilldown(idea, panel, sr3)
        # Visual separator between cards
        st.markdown(
            "<div style='margin: 1.0rem 0; "
            "border-bottom: 1px solid var(--border-subtle);'></div>",
            unsafe_allow_html=True,
        )


def _build_per_leg_execution_subtitle(idea, sr3) -> str:
    """For multi-leg trades, build a compact per-leg execution line showing
    each leg's BUY/SELL action and ACTUAL SR3 contract price. This is what
    the trader actually executes at the exchange (outright contracts).

    Example output: "Execute: BUY SRAH27 @ 96.2600 · SELL SRAH28 @ 96.4000"
    """
    if sr3 is None or sr3.is_outright or not sr3.per_leg_prices:
        return ""
    parts = []
    for lp in sr3.per_leg_prices:
        if lp.get("entry_price") is None:
            continue
        act = lp.get("action", "")
        sym = lp.get("symbol", "")
        price = lp.get("entry_price")
        act_color = "var(--green)" if act == "BUY" else "var(--red)"
        parts.append(
            f"<span style='color:{act_color}; font-weight:600;'>{act}</span> "
            f"<span style='color:var(--text-body);'>{sym}</span> "
            f"<span style='color:var(--text-muted);'>@ {price:.4f}</span>"
        )
    if not parts:
        return ""
    return (
        f"<div style='font-size:0.7rem; color:var(--text-muted); "
        f"margin-top:0.3rem; padding-top:0.3rem; "
        f"border-top:1px solid var(--border-subtle); "
        f"font-family:JetBrains Mono, monospace;'>"
        f"<span style='color:var(--text-dim); font-size:0.62rem; "
        f"text-transform:uppercase; letter-spacing:0.06em; margin-right:0.5rem;'>"
        f"EXECUTE LEGS:</span>"
        + " · ".join(parts) +
        f"</div>"
    )


def _render_trade_minichart(idea, panel: dict, sr3) -> None:
    """Always-visible compact sparkline showing the contract's recent price
    + FV path + entry/target/stop horizontal lines. ~80px tall."""
    from lib.pca_trade_interpretation import build_recent_chart_data
    try:
        import plotly.graph_objects as go
    except Exception:
        return
    primary_sym = idea.legs[0].symbol if idea.legs else None
    if not primary_sym:
        return
    cache_key = f"_chart90d_{primary_sym}"
    if cache_key not in panel:
        try:
            panel[cache_key] = build_recent_chart_data(primary_sym, panel, lookback_days=90)
        except Exception:
            panel[cache_key] = None
    chart = panel.get(cache_key)
    if not chart or not chart.get("dates"):
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart["dates"], y=chart["prices"], mode="lines",
        line=dict(color="#e8b75d", width=1.5), name="close",
        showlegend=False, hoverinfo="x+y",
    ))
    if chart.get("fv_prices") and any(v is not None for v in chart["fv_prices"]):
        fig.add_trace(go.Scatter(
            x=chart["dates"], y=chart["fv_prices"], mode="lines",
            line=dict(color="#5eb4ff", width=1, dash="dot"),
            name="FV", showlegend=False, hoverinfo="x+y",
        ))
    # Entry/target/stop horizontal lines if outright
    if sr3 is not None and sr3.is_outright and sr3.entry_price:
        for y, c in [
            (sr3.entry_price,  "#e8b75d"),
            (sr3.target_price, "#35a951"),
            (sr3.stop_price,   "#e15a5a"),
        ]:
            fig.add_hline(y=y, line=dict(color=c, width=0.7, dash="dash"))
    fig.update_layout(
        height=80, margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono", color="#a8a8b3", size=9),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
        hovermode="x",
    )
    st.plotly_chart(fig, use_container_width=True,
                     key=f"mini_{primary_sym}_{idea.idea_id}",
                     config={"displayModeBar": False})


# =============================================================================
# Trade drilldown (expanded inline below a row)
# =============================================================================
def _render_trade_drilldown(idea, panel: dict, sr3=None) -> None:
    """Rich Trade Card — what the user actually wanted on every trade row.

    Layout (top → bottom):
      1. Actual SR3 contract prices panel (ACTION + ENTRY/TARGET/STOP + R:R + $/c)
      2. 90-day chart of the contract with FV path + entry/target/stop overlays
      3. DYNAMIC 5-paragraph narrative (specific to this trade)
      4. Per-factor interpretation (each factor with what-it-means-for-this-trade)
      5. Execution + sizing controls
    """
    from lib.pca_trade_interpretation import (
        convert_to_sr3_prices, interpret_factors,
        build_recent_chart_data, build_dynamic_narrative,
    )
    # Resolve sr3 + factors + chart data (cached on panel)
    if sr3 is None:
        sr3 = convert_to_sr3_prices(idea, panel)
    factor_cache = f"_factors_{idea.idea_id}"
    if factor_cache not in panel:
        try:
            panel[factor_cache] = interpret_factors(idea, panel)
        except Exception:
            panel[factor_cache] = []
    factors = panel[factor_cache]

    primary_symbol = idea.legs[0].symbol if idea.legs else None
    chart_cache = f"_chart90d_{primary_symbol}"
    if primary_symbol and chart_cache not in panel:
        try:
            panel[chart_cache] = build_recent_chart_data(primary_symbol, panel,
                                                            lookback_days=90)
        except Exception:
            panel[chart_cache] = None
    chart_data = panel.get(chart_cache)

    border = (
        "var(--green)" if idea.conviction >= 0.7
        else "var(--amber)" if idea.conviction >= 0.5
        else "var(--text-dim)"
    )
    st.markdown(
        f"<div style='border-left:3px solid {border}; padding:0.7rem 0.9rem; "
        f"margin: 0.4rem 0 1rem 2rem; background:rgba(255,255,255,0.02); "
        f"border-radius:0 6px 6px 0;'>",
        unsafe_allow_html=True,
    )

    # ============================================================
    # 1. ACTUAL SR3 CONTRACT PRICES PANEL
    # ============================================================
    if sr3 is not None and sr3.is_outright and sr3.entry_price is not None:
        action_color = "var(--green)" if sr3.contract_action == "BUY" else "var(--red)"
        rr_str = f"{sr3.risk_reward:.2f}:1" if sr3.risk_reward else "—"
        st.markdown(
            f"<div style='padding:0.7rem 0.9rem; margin-bottom:0.6rem; "
            f"background:rgba(255,255,255,0.03); "
            f"border:1px solid var(--border-default); border-radius:6px;'>"
            f"<div style='font-size:0.7rem; color:var(--text-dim); "
            f"text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.4rem; "
            f"font-weight:600;'>"
            f"💵 Actual SR3 contract prices — {primary_symbol}</div>"
            f"<div style='display:grid; grid-template-columns: 1fr 1.2fr 1.2fr 1.2fr 1fr 1fr; "
            f"gap:0.8rem; font-family:JetBrains Mono, monospace; font-size:0.78rem;'>"
            f"<div>"
            f"<div style='color:var(--text-dim); font-size:0.65rem;'>ACTION</div>"
            f"<div style='color:{action_color}; font-weight:700; font-size:1.05rem; margin-top:0.15rem;'>"
            f"{sr3.contract_action}</div></div>"
            f"<div>"
            f"<div style='color:var(--text-dim); font-size:0.65rem;'>ENTRY PRICE</div>"
            f"<div style='color:var(--text-heading); font-size:0.95rem; margin-top:0.15rem;'>"
            f"{sr3.entry_price:.4f}</div></div>"
            f"<div>"
            f"<div style='color:var(--text-dim); font-size:0.65rem;'>TARGET PRICE</div>"
            f"<div style='color:var(--green); font-size:0.95rem; margin-top:0.15rem;'>"
            f"{sr3.target_price:.4f}</div></div>"
            f"<div>"
            f"<div style='color:var(--text-dim); font-size:0.65rem;'>STOP PRICE</div>"
            f"<div style='color:var(--red); font-size:0.95rem; margin-top:0.15rem;'>"
            f"{sr3.stop_price:.4f}</div></div>"
            f"<div>"
            f"<div style='color:var(--text-dim); font-size:0.65rem;'>P&L/c</div>"
            f"<div style='color:var(--green); font-size:0.95rem; margin-top:0.15rem;'>"
            f"${sr3.pnl_per_contract_dollar:.0f}</div></div>"
            f"<div>"
            f"<div style='color:var(--text-dim); font-size:0.65rem;'>R:R</div>"
            f"<div style='color:var(--text-body); font-size:0.95rem; margin-top:0.15rem;'>"
            f"{rr_str}</div></div>"
            f"</div>"
            f"<div style='margin-top:0.45rem; padding-top:0.4rem; "
            f"border-top:1px solid var(--border-subtle); "
            f"font-size:0.7rem; color:var(--text-dim); font-family:JetBrains Mono, monospace;'>"
            f"YIELD-SPACE: entry {sr3.entry_yield_bp:.2f} bp · "
            f"target {sr3.target_yield_bp:.2f} bp · "
            f"stop {sr3.stop_yield_bp:.2f} bp"
            f"</div></div>",
            unsafe_allow_html=True,
        )
    elif sr3 is not None and not sr3.is_outright:
        # Multi-leg trade — show per-leg execution table with FULL prices
        # (entry / target / stop) for each leg + synthetic spread/fly quote.
        leg_rows = []
        for lp in sr3.per_leg_prices:
            act_color = "var(--green)" if lp["action"] == "BUY" else "var(--red)"
            entry_p = (f"{lp['entry_price']:.4f}" if lp.get("entry_price") is not None else "—")
            target_p = (f"{lp['target_price']:.4f}" if lp.get("target_price") is not None else "—")
            stop_p = (f"{lp['stop_price']:.4f}" if lp.get("stop_price") is not None else "—")
            pnl_p = f"${lp.get('pnl_per_contract_dollar', 0):.0f}"
            risk_p = f"${lp.get('risk_per_contract_dollar', 0):.0f}"
            leg_rows.append(
                f"<div style='display:grid; "
                f"grid-template-columns: 1.0fr 0.5fr 0.85fr 0.85fr 0.85fr 0.6fr 0.6fr 0.55fr; "
                f"gap:0.4rem; padding:4px 0; font-size:0.78rem; "
                f"font-family:JetBrains Mono, monospace;'>"
                f"<span style='color:var(--text-body);'>{lp['symbol']}</span>"
                f"<span style='color:{act_color}; font-weight:700;'>{lp['action']}</span>"
                f"<span style='color:var(--text-heading); text-align:right;'>{entry_p}</span>"
                f"<span style='color:var(--green); text-align:right;'>{target_p}</span>"
                f"<span style='color:var(--red); text-align:right;'>{stop_p}</span>"
                f"<span style='color:var(--green); text-align:right;'>{pnl_p}</span>"
                f"<span style='color:var(--red); text-align:right;'>{risk_p}</span>"
                f"<span style='color:var(--text-muted); text-align:right;'>{lp['pv01_wt']:+.2f}</span>"
                f"</div>"
            )

        # PRICE block — LISTED / SYNTHETIC / PER-LEG depending on what's available.
        # ALWAYS produces a block (never empty).
        listed_block = ""
        cw = "(1, -1)" if len(idea.legs) == 2 else "(1, -2, 1)"
        if sr3.price_source == "per_leg":
            # No listed product and no canonical synthetic available.
            # Engine-generated multi-leg structure (e.g. PC2-spread with 3
            # legs, PC1-basket with 4 legs). Show package P&L from per-leg.
            pkg_action = sr3.package_action or "—"
            pkg_color = ("var(--green)" if pkg_action == "BUY"
                            else "var(--red)" if pkg_action == "SELL"
                            else "var(--text-dim)")
            listed_block = (
                f"<div style='margin-top:0.55rem; padding:0.6rem 0.7rem; "
                f"background:rgba(255,255,255,0.025); "
                f"border:1.5px solid var(--text-dim); border-radius:6px; "
                f"font-family:JetBrains Mono, monospace; font-size:0.78rem;'>"
                f"<div style='font-size:0.7rem; color:var(--text-muted); "
                f"text-transform:uppercase; font-weight:700; margin-bottom:0.3rem;'>"
                f"🔗 {len(idea.legs)}-LEG ENGINE STRUCTURE "
                f"<span style='color:var(--text-dim); font-size:0.65rem; font-weight:400; "
                f"text-transform:none;'>· no listed CME product · trade each leg below</span>"
                f"<div style='display:inline-block; padding:0.1rem 0.4rem; "
                f"border-radius:3px; background:rgba(255,255,255,0.04); "
                f"color:var(--text-body); font-size:0.62rem; margin-left:0.5rem;'>"
                f"PER-LEG</div></div>"
                f"<div style='display:grid; grid-template-columns: 0.7fr 1.2fr 1fr 1fr 0.9fr 0.9fr 0.7fr; gap:0.6rem;'>"
                f"<div><span style='color:var(--text-dim); font-size:0.62rem; display:block;'>ACTION (package)</span>"
                f"<span style='color:{pkg_color}; font-size:1.1rem; font-weight:700;'>"
                f"{pkg_action}</span></div>"
                f"<div><span style='color:var(--text-dim); font-size:0.62rem; display:block;'>NET RESIDUAL</span>"
                f"<span style='color:var(--text-heading); font-size:1.0rem;'>"
                f"{sr3.net_entry_bp:+.2f} bp/u-pv01</span></div>"
                f"<div><span style='color:var(--text-dim); font-size:0.62rem; display:block;'>TARGET</span>"
                f"<span style='color:var(--green); font-size:1.0rem;'>"
                f"{sr3.net_target_bp:+.2f} bp/u</span></div>"
                f"<div><span style='color:var(--text-dim); font-size:0.62rem; display:block;'>STOP</span>"
                f"<span style='color:var(--red); font-size:1.0rem;'>"
                f"{sr3.net_stop_bp:+.2f} bp/u</span></div>"
                f"<div><span style='color:var(--text-dim); font-size:0.62rem; display:block;'>PKG P&L</span>"
                f"<span style='color:var(--green);'>${sr3.pnl_per_contract_dollar:.0f}</span></div>"
                f"<div><span style='color:var(--text-dim); font-size:0.62rem; display:block;'>PKG RISK</span>"
                f"<span style='color:var(--red);'>${sr3.risk_per_contract_dollar:.0f}</span></div>"
                f"<div><span style='color:var(--text-dim); font-size:0.62rem; display:block;'>R:R</span>"
                f"<span style='color:var(--text-body);'>{sr3.risk_reward:.2f}:1</span></div>"
                f"</div>"
                f"<div style='margin-top:0.35rem; padding-top:0.3rem; "
                f"border-top:1px solid var(--border-subtle); "
                f"font-size:0.66rem; color:var(--text-muted); line-height:1.55;'>"
                f"<b>This is a {len(idea.legs)}-leg engine-generated structure.</b> "
                f"It does not map 1:1 to a CME-listed spread or fly contract — execute "
                f"each leg individually (see the per-leg execution table below). "
                f"<b>Net residual</b> is the PV01-weighted-bp sum from the engine; package "
                f"P&L = Σ leg-P&L (sum of expected per-leg moves × $25/bp/contract)."
                f"</div>"
                f"</div>"
            )
        elif sr3.price_source == "synthetic":
            # No listed product matches the leg pattern (e.g. PC2-spread with
            # non-standard gap, or PC1-basket); use synthetic-from-legs.
            spnl = (f"${sr3.synth_pnl_dollar:.2f}"
                      if sr3.synth_pnl_dollar is not None else "—")
            srisk = (f"${sr3.synth_risk_dollar:.2f}"
                       if sr3.synth_risk_dollar is not None else "—")
            srr = (f"{sr3.synth_risk_reward:.2f}:1"
                     if sr3.synth_risk_reward is not None else "—")
            pkg_action = sr3.package_action or "—"
            pkg_color = "var(--green)" if pkg_action == "BUY" else "var(--red)"
            listed_block = (
                f"<div style='margin-top:0.55rem; padding:0.6rem 0.7rem; "
                f"background:rgba(94,180,255,0.06); "
                f"border:2px solid var(--blue); border-radius:6px; "
                f"font-family:JetBrains Mono, monospace; font-size:0.78rem;'>"
                f"<div style='font-size:0.7rem; color:var(--blue); "
                f"text-transform:uppercase; font-weight:700; margin-bottom:0.3rem;'>"
                f"📐 SYNTHETIC {idea.structure_type.upper()} — reconstructed from legs "
                f"<span style='color:var(--text-dim); font-size:0.65rem; font-weight:400; "
                f"text-transform:none;'>· canonical {cw} · no listed CME product matches</span>"
                f"<div style='display:inline-block; padding:0.1rem 0.4rem; "
                f"border-radius:3px; background:rgba(255,255,255,0.04); "
                f"color:var(--blue); font-size:0.62rem; margin-left:0.5rem;'>"
                f"SYNTH</div></div>"
                f"<div style='display:grid; grid-template-columns: 0.7fr 1.2fr 1fr 1fr 0.9fr 0.9fr 0.7fr; gap:0.6rem;'>"
                f"<div><span style='color:var(--text-dim); font-size:0.62rem; display:block;'>ACTION</span>"
                f"<span style='color:{pkg_color}; font-size:1.1rem; font-weight:700;'>"
                f"{pkg_action}</span></div>"
                f"<div><span style='color:var(--text-dim); font-size:0.62rem; display:block;'>ENTRY (Σ cw·p×100)</span>"
                f"<span style='color:var(--text-heading); font-size:1.0rem;'>"
                f"{sr3.synth_canonical_bp:+.2f} bp</span></div>"
                f"<div><span style='color:var(--text-dim); font-size:0.62rem; display:block;'>TARGET (FV=0)</span>"
                f"<span style='color:var(--green); font-size:1.0rem;'>"
                f"{sr3.synth_target_bp:+.2f} bp</span></div>"
                f"<div><span style='color:var(--text-dim); font-size:0.62rem; display:block;'>STOP (3.5×)</span>"
                f"<span style='color:var(--red); font-size:1.0rem;'>"
                f"{sr3.synth_stop_bp:+.2f} bp</span></div>"
                f"<div><span style='color:var(--text-dim); font-size:0.62rem; display:block;'>P&L/c</span>"
                f"<span style='color:var(--green);'>{spnl}</span></div>"
                f"<div><span style='color:var(--text-dim); font-size:0.62rem; display:block;'>RISK/c</span>"
                f"<span style='color:var(--red);'>{srisk}</span></div>"
                f"<div><span style='color:var(--text-dim); font-size:0.62rem; display:block;'>R:R</span>"
                f"<span style='color:var(--text-body);'>{srr}</span></div>"
                f"</div>"
                f"<div style='margin-top:0.35rem; padding-top:0.3rem; "
                f"border-top:1px solid var(--border-subtle); "
                f"font-size:0.66rem; color:var(--text-muted); line-height:1.55;'>"
                f"<b>Units:</b> bp of price-differential (Σ cw·leg_price × 100). "
                f"FV = 0 (canonical Σcw=0). Target = 0; stop = 3.5×entry; "
                f"P&L = |move_bp| × $25/contract. "
                f"<b>No listed CME calendar/fly product matches this exact leg pattern</b> "
                f"— trade by executing each leg individually."
                f"</div>"
                f"</div>"
            )
        elif sr3.price_source == "listed":
            pnl_str = (f"${sr3.listed_pnl_dollar:.2f}"
                         if sr3.listed_pnl_dollar is not None else "—")
            risk_str = (f"${sr3.listed_risk_dollar:.2f}"
                          if sr3.listed_risk_dollar is not None else "—")
            rr_str = (f"{sr3.listed_risk_reward:.2f}:1"
                        if sr3.listed_risk_reward is not None else "—")
            # Cross-check chip — listed vs reconstructed-from-legs
            check_chip = ""
            if sr3.synth_canonical_bp is not None and sr3.listed_vs_synth_diff_bp is not None:
                d = abs(sr3.listed_vs_synth_diff_bp)
                if d <= 1.0:
                    check_color = "var(--green)"
                    check_label = "✓ matches legs"
                elif d <= 5.0:
                    check_color = "var(--amber)"
                    check_label = f"⚠ mild Δ {d:.1f} bp"
                else:
                    check_color = "var(--red)"
                    check_label = f"⚠ Δ {d:.1f} bp — verify"
                check_chip = (
                    f"<div style='display:inline-block; padding:0.1rem 0.4rem; "
                    f"border-radius:3px; background:rgba(255,255,255,0.04); "
                    f"color:{check_color}; font-size:0.62rem; margin-left:0.5rem;'>"
                    f"{check_label}</div>"
                )
            # Cross-check line (only if synth_canonical_bp is computed)
            xcheck_line = ""
            if sr3.synth_canonical_bp is not None and sr3.listed_vs_synth_diff_bp is not None:
                explain = ("sub-bp settlement rounding"
                            if abs(sr3.listed_vs_synth_diff_bp) <= 1
                            else "real disagreement — verify the leg pattern matches")
                xcheck_line = (
                    f"<br><b>Cross-check:</b> reconstructed from legs = "
                    f"<b style='color:var(--text-body);'>{sr3.synth_canonical_bp:+.2f} bp</b> "
                    f"⟶ Δ = <b>{sr3.listed_vs_synth_diff_bp:+.2f} bp</b> ({explain})."
                )
            # Big BUY/SELL chip for the package
            pkg_action = sr3.package_action or "—"
            pkg_color = "var(--green)" if pkg_action == "BUY" else "var(--red)"
            listed_block = (
                f"<div style='margin-top:0.55rem; padding:0.6rem 0.7rem; "
                f"background:rgba(232,183,93,0.08); "
                f"border:2px solid var(--accent); border-radius:6px; "
                f"font-family:JetBrains Mono, monospace; font-size:0.78rem;'>"
                f"<div style='font-size:0.7rem; color:var(--accent); "
                f"text-transform:uppercase; font-weight:700; margin-bottom:0.3rem;'>"
                f"🎯 LISTED CME {idea.structure_type.upper()} — "
                f"<span style='color:var(--text-heading); font-size:0.85rem;'>"
                f"{sr3.listed_symbol}</span> "
                f"<span style='color:var(--text-dim); font-size:0.65rem; font-weight:400; "
                f"text-transform:none;'>· canonical {cw} · units = bp of price-diff</span>"
                f"{check_chip}</div>"
                f"<div style='display:grid; grid-template-columns: 0.7fr 1.2fr 1fr 1fr 0.9fr 0.9fr 0.7fr; gap:0.6rem;'>"
                f"<div><span style='color:var(--text-dim); font-size:0.62rem; display:block;'>ACTION</span>"
                f"<span style='color:{pkg_color}; font-size:1.1rem; font-weight:700;'>"
                f"{pkg_action}</span></div>"
                f"<div><span style='color:var(--text-dim); font-size:0.62rem; display:block;'>ENTRY (CME quote)</span>"
                f"<span style='color:var(--text-heading); font-size:1.0rem;'>"
                f"{sr3.listed_entry_price:+.2f} bp</span></div>"
                f"<div><span style='color:var(--text-dim); font-size:0.62rem; display:block;'>TARGET (FV=0)</span>"
                f"<span style='color:var(--green); font-size:1.0rem;'>"
                f"{sr3.listed_target_price:+.2f} bp</span></div>"
                f"<div><span style='color:var(--text-dim); font-size:0.62rem; display:block;'>STOP (3.5×)</span>"
                f"<span style='color:var(--red); font-size:1.0rem;'>"
                f"{sr3.listed_stop_price:+.2f} bp</span></div>"
                f"<div><span style='color:var(--text-dim); font-size:0.62rem; display:block;'>P&L/c</span>"
                f"<span style='color:var(--green);'>{pnl_str}</span></div>"
                f"<div><span style='color:var(--text-dim); font-size:0.62rem; display:block;'>RISK/c</span>"
                f"<span style='color:var(--red);'>{risk_str}</span></div>"
                f"<div><span style='color:var(--text-dim); font-size:0.62rem; display:block;'>R:R</span>"
                f"<span style='color:var(--text-body);'>{rr_str}</span></div>"
                f"</div>"
                f"<div style='margin-top:0.35rem; padding-top:0.3rem; "
                f"border-top:1px solid var(--border-subtle); "
                f"font-size:0.66rem; color:var(--text-muted); line-height:1.55;'>"
                f"<b>Units:</b> CME-listed spread/fly quoted in bp of price-differential "
                f"(1 unit = 1 bp = $25/contract). Listed = Σ cw·legprice × 100. "
                f"FV = 0 (since Σcw=0). Target = 0; stop = 3.5×entry; tick = 0.25 bp."
                f"{xcheck_line}"
                f"</div>"
                f"</div>"
            )

        # Synthetic spread/fly price block — secondary, for verification
        synth_block = ""
        if (sr3.synth_entry_unit is not None and sr3.synth_target_unit is not None
                and sr3.synth_stop_unit is not None):
            synth_block = (
                f"<div style='margin-top:0.55rem; padding:0.5rem 0.6rem; "
                f"background:rgba(94,180,255,0.06); border-left:2px solid var(--blue); "
                f"border-radius:0 4px 4px 0; "
                f"font-family:JetBrains Mono, monospace; font-size:0.75rem;'>"
                f"<div style='font-size:0.65rem; color:var(--blue); "
                f"text-transform:uppercase; font-weight:600; margin-bottom:0.2rem;'>"
                f"📐 Synthetic {idea.structure_type} quote (derived from leg combination — verification)</div>"
                f"<div style='display:grid; grid-template-columns: 1fr 1fr 1fr; gap:0.6rem;'>"
                f"<div><span style='color:var(--text-dim); font-size:0.62rem; display:block;'>ENTRY</span>"
                f"<span style='color:var(--text-heading);'>{sr3.synth_entry_unit:+.4f}</span></div>"
                f"<div><span style='color:var(--text-dim); font-size:0.62rem; display:block;'>TARGET</span>"
                f"<span style='color:var(--green);'>{sr3.synth_target_unit:+.4f}</span></div>"
                f"<div><span style='color:var(--text-dim); font-size:0.62rem; display:block;'>STOP</span>"
                f"<span style='color:var(--red);'>{sr3.synth_stop_unit:+.4f}</span></div>"
                f"</div></div>"
            )

        st.markdown(
            f"<div style='padding:0.7rem 0.9rem; margin-bottom:0.6rem; "
            f"background:rgba(255,255,255,0.03); "
            f"border:1px solid var(--border-default); border-radius:6px;'>"
            f"<div style='font-size:0.7rem; color:var(--text-dim); "
            f"text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.4rem; "
            f"font-weight:600;'>"
            f"💵 Per-leg execution ({idea.structure_type}) — execute all legs simultaneously</div>"
            f"<div style='display:grid; "
            f"grid-template-columns: 1.0fr 0.5fr 0.85fr 0.85fr 0.85fr 0.6fr 0.6fr 0.55fr; "
            f"gap:0.4rem; padding:3px 0; font-size:0.62rem; color:var(--text-dim); "
            f"font-family:JetBrains Mono, monospace; "
            f"border-bottom:1px solid var(--border-subtle); margin-bottom:0.2rem; "
            f"text-transform:uppercase; letter-spacing:0.05em;'>"
            f"<span>LEG</span><span>ACTION</span>"
            f"<span style='text-align:right;'>ENTRY $</span>"
            f"<span style='text-align:right;'>TARGET $</span>"
            f"<span style='text-align:right;'>STOP $</span>"
            f"<span style='text-align:right;'>P&L/c</span>"
            f"<span style='text-align:right;'>RISK/c</span>"
            f"<span style='text-align:right;'>PV01 W</span></div>"
            + "".join(leg_rows)
            + listed_block
            + synth_block
            + f"<div style='margin-top:0.5rem; padding-top:0.4rem; "
            f"border-top:1px solid var(--border-subtle); font-family:JetBrains Mono, monospace; "
            f"font-size:0.72rem;'>"
            f"<span style='color:var(--text-dim);'>NET residual (per unit PV01):</span> "
            f"<span style='color:var(--text-body);'>"
            f"entry {sr3.net_entry_bp:+.2f} bp · "
            f"target {sr3.net_target_bp:+.2f} bp · "
            f"stop {sr3.net_stop_bp:+.2f} bp · "
            f"<b>Package P&L ${sr3.pnl_per_contract_dollar:.0f}</b> · "
            f"<b>Risk ${sr3.risk_per_contract_dollar:.0f}</b> · "
            f"<b>R:R {(f'{sr3.risk_reward:.2f}:1' if sr3.risk_reward else '—')}</b>"
            f"</span></div></div>",
            unsafe_allow_html=True,
        )

    # ============================================================
    # 2. CHART PANEL — 3 charts side-by-side (price+FV, residual+OU band, conviction)
    # ============================================================
    if chart_data and chart_data.get("dates"):
        chart_cols = st.columns([1.4, 1.0, 1.0])
        with chart_cols[0]:
            st.markdown(
                "<div style='font-size:0.68rem; color:var(--text-dim); "
                "text-transform:uppercase; letter-spacing:0.06em; "
                "margin-bottom:0.2rem; font-weight:600;'>"
                "💹 Price + PCA fair value</div>",
                unsafe_allow_html=True,
            )
            _render_trade_chart_inline(chart_data, sr3, primary_symbol, idea.idea_id)
        with chart_cols[1]:
            st.markdown(
                "<div style='font-size:0.68rem; color:var(--text-dim); "
                "text-transform:uppercase; letter-spacing:0.06em; "
                "margin-bottom:0.2rem; font-weight:600;'>"
                "📈 Residual + OU bands</div>",
                unsafe_allow_html=True,
            )
            _render_trade_chart_residual(chart_data, idea, panel)
        with chart_cols[2]:
            st.markdown(
                "<div style='font-size:0.68rem; color:var(--text-dim); "
                "text-transform:uppercase; letter-spacing:0.06em; "
                "margin-bottom:0.2rem; font-weight:600;'>"
                "🎯 Conviction breakdown</div>",
                unsafe_allow_html=True,
            )
            _render_trade_chart_conviction(idea)

    # ============================================================
    # 2.5 — POSITIONAL OUTLOOK + EXIT PLAN (new)
    # ============================================================
    _render_positional_outlook_and_exit_plan(idea, panel)

    # ============================================================
    # 3. DYNAMIC PER-TRADE NARRATIVE (the heart of this rebuild)
    # ============================================================
    try:
        narrative = build_dynamic_narrative(idea, sr3, factors, panel)
    except Exception as e:
        narrative = f"<div style='color:var(--amber);'>Narrative generation failed: {e}</div>"
    st.markdown(
        f"<div style='padding:0.7rem 0.9rem; margin-bottom:0.6rem; "
        f"background:rgba(232,183,93,0.04); "
        f"border-left:3px solid var(--accent); border-radius:0 6px 6px 0;'>"
        f"<div style='font-size:0.7rem; color:var(--accent); "
        f"text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.5rem; "
        f"font-weight:600;'>"
        f"🧠 How the engine arrived at THIS trade</div>"
        f"{narrative}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ============================================================
    # 4. PER-FACTOR INTERPRETATION (always visible, not in expander)
    # ============================================================
    if factors:
        st.markdown(
            "<div style='font-size:0.7rem; color:var(--text-dim); "
            "text-transform:uppercase; letter-spacing:0.06em; "
            "margin: 0.6rem 0 0.3rem 0; font-weight:600;'>"
            "📊 Per-factor analysis — every signal contributing to the conviction"
            "</div>",
            unsafe_allow_html=True,
        )
        from lib.pca_concepts import get_concept
        for f in factors:
            tier_color = {"supportive": "var(--green)",
                            "neutral": "var(--text-body)",
                            "caveat": "var(--amber)"}.get(f.tier, "var(--text-body)")
            tier_icon = {"supportive": "✓", "neutral": "·", "caveat": "⚠"}.get(f.tier, "·")
            concept = get_concept(f.key) or {}
            concept_name = concept.get("name", f.key)
            wt_str = (f"<span style='color:var(--text-dim); margin-left:0.5rem;'>"
                       f"(+{f.weight_in_conviction:.3f} to conviction)</span>"
                       if f.weight_in_conviction > 0
                       else (f"<span style='color:var(--red); margin-left:0.5rem;'>"
                              f"({f.weight_in_conviction:+.3f})</span>"
                              if f.weight_in_conviction < 0 else ""))
            # Lookup the concept's "what is it" for inline disclosure (under detail)
            concept_explainer = ""
            if concept.get("one_liner"):
                concept_explainer = (
                    f"<div style='margin-top:0.3rem; padding:0.3rem 0.5rem; "
                    f"background:rgba(94,180,255,0.04); border-radius:3px; "
                    f"font-size:0.7rem; color:var(--text-muted); line-height:1.5;'>"
                    f"<b style='color:var(--blue);'>What is {concept_name}?</b> "
                    f"{concept.get('one_liner')}"
                    f"</div>"
                )
            st.markdown(
                f"<div style='padding:0.5rem 0.7rem; margin: 0.25rem 0; "
                f"border-left:3px solid {tier_color}; "
                f"background:rgba(255,255,255,0.015); border-radius:0 4px 4px 0;'>"
                f"<div style='display:flex; justify-content:space-between; align-items:baseline;'>"
                f"<div>"
                f"<span style='color:{tier_color}; font-weight:600;'>{tier_icon} {concept_name}</span> "
                f"<span style='color:var(--text-dim); font-size:0.72rem; "
                f"font-family:JetBrains Mono, monospace;'>= {f.value_display}</span>{wt_str}</div>"
                f"</div>"
                f"<div style='font-size:0.78rem; color:var(--text-body); margin-top:0.25rem;'>"
                f"<b>{f.headline}</b></div>"
                f"<div style='font-size:0.72rem; color:var(--text-muted); margin-top:0.2rem; "
                f"line-height:1.55;'>{f.detail}</div>"
                f"{concept_explainer}"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ============================================================
    # 5. EXECUTION + LEG NAV (kept from old drilldown)
    # ============================================================
    exec_col, leg_col = st.columns([1.2, 1.0])
    with exec_col:
        st.markdown(
            "<div style='font-size:0.7rem; color:var(--text-dim); "
            "text-transform:uppercase; letter-spacing:0.06em; "
            "margin: 0.5rem 0 0.3rem 0; font-weight:600;'>🎫 Generate ticket</div>",
            unsafe_allow_html=True,
        )
        size = st.number_input(
            "$ size", 10000, 1000000, 100000, 10000,
            key=f"{_SCOPE}_size_{idea.idea_id}", label_visibility="collapsed",
        )
        if st.button("Generate ticket CSV", key=f"{_SCOPE}_tkt_{idea.idea_id}",
                          use_container_width=True):
            ticket = generate_execution_ticket(idea, position_size_dollar=size)
            if ticket["legs"]:
                st.code(ticket["csv_text"], language="csv")
                st.download_button(
                    "⬇️ Download CSV", data=ticket["csv_text"],
                    file_name=f"ticket_{idea.idea_id}.csv",
                    mime="text/csv",
                    key=f"{_SCOPE}_dl_{idea.idea_id}",
                )
    with leg_col:
        if idea.legs:
            st.markdown(
                "<div style='font-size:0.7rem; color:var(--text-dim); "
                "text-transform:uppercase; letter-spacing:0.06em; "
                "margin: 0.5rem 0 0.3rem 0; font-weight:600;'>"
                "🔍 Open contract dossier</div>",
                unsafe_allow_html=True,
            )
            leg_cols = st.columns(min(4, len(idea.legs)))
            for i, leg in enumerate(idea.legs[:4]):
                with leg_cols[i % len(leg_cols)]:
                    side_color = "🟢" if leg.side == "buy" else "🔴"
                    if st.button(
                        f"{side_color} {leg.symbol}",
                        key=f"{_SCOPE}_leg_{idea.idea_id}_{i}",
                        help=f"weight: {leg.weight_pv01:+.4f}",
                        use_container_width=True,
                    ):
                        _set_contract(leg.symbol)
                        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def _render_trade_chart_inline(chart_data: dict, sr3, symbol: str,
                                  idea_id: str) -> None:
    """Render the 90-day SR3 price chart with FV path + entry/target/stop markers
    inline in the trade card."""
    try:
        import plotly.graph_objects as go
    except Exception:
        return
    dates = chart_data.get("dates", [])
    prices = chart_data.get("prices", [])
    fv_prices = chart_data.get("fv_prices", [])
    if not dates or not prices:
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=prices, mode="lines", name=f"{symbol} close",
        line=dict(color="#e8b75d", width=2),
    ))
    # FV time series intentionally NOT plotted — the engine's PCA is fit on
    # yield CHANGES (lib.pca:506) so a level-space FV reconstruction is
    # mathematically wrong. Today's FV = sr3.target_price (entry − residual)
    # and is shown as the TARGET horizontal line below.
    if sr3 and sr3.is_outright and sr3.entry_price:
        for label, y, color in [
            ("ENTRY",   sr3.entry_price,  "#e8b75d"),
            ("FV/TARGET", sr3.target_price, "#35a951"),
            ("STOP",    sr3.stop_price,   "#e15a5a"),
        ]:
            fig.add_hline(y=y, line=dict(color=color, width=1, dash="dash"),
                            annotation_text=f"{label} {y:.4f}",
                            annotation_position="right",
                            annotation_font_color=color)
    fig.update_layout(
        height=240, margin=dict(l=10, r=85, t=10, b=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono", color="#a8a8b3", size=10),
        xaxis=dict(gridcolor="#2a2a35", showgrid=True, gridwidth=0.5),
        yaxis=dict(gridcolor="#2a2a35", showgrid=True, gridwidth=0.5,
                     title=f"{symbol} price"),
        showlegend=True,
        legend=dict(orientation="h", x=0, y=1.15, font=dict(size=9)),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True,
                     key=f"chart_{symbol}_{idea_id}")


def _render_trade_chart_residual(chart_data: dict, idea, panel: dict) -> None:
    """Plot the residual time series with ±1σ / ±2σ horizontal bands
    + entry / target (0) / stop residual lines.

    Uses build_residual_series (CHANGE-space residual replicating the engine's
    per_traded_outright_residuals logic) — NOT chart_data.residuals_bp
    (which was reconstructed in a broken level-space).
    """
    try:
        import plotly.graph_objects as go
    except Exception:
        return
    # Try to use the engine's actual residual series for this contract
    primary_sym = idea.legs[0].symbol if idea.legs else None
    dates = []
    resids = []
    mean_resid = 0.0
    sigma_resid = 1.0
    if primary_sym:
        from lib.pca_trade_interpretation import build_residual_series
        res_cache_key = f"_resid_series_{primary_sym}"
        if res_cache_key not in panel:
            try:
                panel[res_cache_key] = build_residual_series(primary_sym, panel, lookback_days=90)
            except Exception:
                panel[res_cache_key] = None
        rs = panel.get(res_cache_key) or {}
        dates = rs.get("dates", [])
        resids = rs.get("residuals_bp", [])
        mean_resid = rs.get("mean") or 0.0
        sigma_resid = rs.get("sigma") or 1.0
    if not dates or not resids:
        st.markdown("<div style='color:var(--text-dim); font-size:0.7rem; "
                     "padding:1rem 0; text-align:center;'>"
                     "Residual series unavailable (engine fit missing).</div>", unsafe_allow_html=True)
        return
    import numpy as np_local

    fig = go.Figure()
    # ±1σ and ±2σ shaded bands
    fig.add_hrect(y0=mean_resid - 2*sigma_resid, y1=mean_resid + 2*sigma_resid,
                    line_width=0, fillcolor="rgba(232,141,60,0.08)")
    fig.add_hrect(y0=mean_resid - 1*sigma_resid, y1=mean_resid + 1*sigma_resid,
                    line_width=0, fillcolor="rgba(94,180,255,0.08)")
    # The residual series
    fig.add_trace(go.Scatter(
        x=dates, y=resids, mode="lines", name="residual (bp)",
        line=dict(color="#e8b75d", width=1.5), showlegend=False,
    ))
    # Mean line (= FV)
    fig.add_hline(y=mean_resid, line=dict(color="#5eb4ff", width=1, dash="dot"))
    # Entry / target / stop in residual space
    if idea.entry_bp is not None:
        fig.add_hline(y=idea.entry_bp, line=dict(color="#e8b75d", width=1, dash="dash"),
                        annotation_text="ENTRY", annotation_position="right",
                        annotation_font_color="#e8b75d")
    if idea.target_bp is not None:
        fig.add_hline(y=idea.target_bp, line=dict(color="#35a951", width=1, dash="dash"),
                        annotation_text="TARGET", annotation_position="right",
                        annotation_font_color="#35a951")
    if idea.stop_bp is not None:
        fig.add_hline(y=idea.stop_bp, line=dict(color="#e15a5a", width=1, dash="dash"),
                        annotation_text="STOP", annotation_position="right",
                        annotation_font_color="#e15a5a")
    fig.update_layout(
        height=240, margin=dict(l=10, r=80, t=10, b=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono", color="#a8a8b3", size=10),
        xaxis=dict(gridcolor="#2a2a35", showgrid=True, gridwidth=0.5),
        yaxis=dict(gridcolor="#2a2a35", showgrid=True, gridwidth=0.5,
                     title="residual (bp)"),
        showlegend=False,
        hovermode="x",
    )
    st.plotly_chart(fig, use_container_width=True,
                     key=f"resid_{idea.idea_id}",
                     config={"displayModeBar": False})


def _render_positional_outlook_and_exit_plan(idea, panel: dict) -> None:
    """Render Positional Outlook + Exit Plan cards inside an expanded trade card.

    Positional Outlook shows: estimated hold days, regime stability over hold
    window, FOMC events in window, carry, convexity warning, z-score path.

    Exit Plan shows the 4 target tiers + 6 stop tiers with their target levels
    (residual / day / z) so the user knows the engine's full exit roadmap.
    """
    from lib.trade_exits import entry_state_from_idea, planned_levels
    from lib.pca import mode_params

    mode = panel.get("mode") or "positional"
    mp = mode_params(mode)
    asof = panel.get("asof")

    # Build EntryState from idea (treat current state as "entry" for display)
    es = entry_state_from_idea(idea, asof if asof else date.today())
    levels = planned_levels(idea, es, mode=mode)

    # ── Positional Outlook ──
    hold_days = idea.expected_revert_d
    fomc = panel.get("fomc_calendar_dates", []) or []
    horizon_days = int(round((hold_days or mp["hold_floor"]) * 1.5))
    upcoming_fomc = []
    if asof:
        upcoming_fomc = sorted([pd.Timestamp(d).date() for d in fomc
                                  if pd.Timestamp(d).date() > asof
                                  and (pd.Timestamp(d).date() - asof).days <= horizon_days])
    n_fomc = len(upcoming_fomc)
    fomc_text = ", ".join(d.strftime("%Y-%m-%d") for d in upcoming_fomc[:3]) or "none"

    # Regime stability (HMM transition matrix → P(regime unchanged in N days))
    regime_stack = panel.get("regime_stack", {})
    hmm = regime_stack.get("hmm_fit") if regime_stack else None
    regime_conf_str = "—"
    if hmm is not None and getattr(hmm, "dominant_confidence", None) is not None:
        try:
            confs = list(hmm.dominant_confidence)
            if confs:
                regime_conf_str = f"{float(confs[-1])*100:.0f}%"
        except Exception:
            pass

    convexity_str = ("⚠ flagged" if idea.convexity_warning
                      else "✓ within bounds")

    # Market-aware central bank label (Fed/ECB/BoE/RBA/BoC/SNB)
    cb_name_upper = str(market_cfg.get("central_bank", "Fed")).upper()
    cb_name = str(market_cfg.get("central_bank", "Fed"))

    # Build the optional CB-meeting footer separately so the outer f-string
    # doesn't have to contain backslash-escaped HTML (forbidden in Python 3.12+).
    fomc_footer_html = ""
    if upcoming_fomc:
        fomc_footer_html = (
            "<div style='margin-top:0.4rem; padding-top:0.35rem; "
            "border-top:1px solid var(--border-subtle); "
            "font-size:0.68rem; color:var(--text-muted);'>"
            f"{cb_name} meeting dates: {fomc_text}"
            "</div>"
        )
    st.markdown(
        f"<div style='padding:0.6rem 0.9rem; margin-bottom:0.5rem; "
        f"background:rgba(94,180,255,0.04); border-left:3px solid var(--blue); "
        f"border-radius:0 6px 6px 0;'>"
        f"<div style='font-size:0.7rem; color:var(--blue); "
        f"text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.4rem; "
        f"font-weight:600;'>"
        f"📅 Positional outlook — {mode} mode</div>"
        f"<div style='display:grid; grid-template-columns: repeat(5, 1fr); "
        f"gap:0.8rem; font-family:JetBrains Mono, monospace; font-size:0.74rem;'>"
        f"<div><div style='color:var(--text-dim); font-size:0.65rem;'>HOLD DAYS</div>"
        f"<div style='color:var(--text-heading); font-size:0.85rem;'>"
        f"{int(round(hold_days or 0))}d "
        f"<span style='color:var(--text-dim); font-size:0.65rem;'>"
        f"(cap {int(mp['hold_cap'])}d)</span></div></div>"
        f"<div><div style='color:var(--text-dim); font-size:0.65rem;'>REGIME CONF</div>"
        f"<div style='color:var(--text-heading); font-size:0.85rem;'>{regime_conf_str}</div></div>"
        f"<div><div style='color:var(--text-dim); font-size:0.65rem;'>{cb_name_upper} IN WINDOW</div>"
        f"<div style='color:var(--text-heading); font-size:0.85rem;'>"
        f"{n_fomc} event{'s' if n_fomc != 1 else ''}</div></div>"
        f"<div><div style='color:var(--text-dim); font-size:0.65rem;'>CONVEXITY</div>"
        f"<div style='color:var(--text-heading); font-size:0.85rem;'>{convexity_str}</div></div>"
        f"<div><div style='color:var(--text-dim); font-size:0.65rem;'>EFF SAMPLE</div>"
        f"<div style='color:var(--text-heading); font-size:0.85rem;'>n = {idea.eff_n}</div></div>"
        f"</div>"
        f"{fomc_footer_html}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Exit Plan ──
    def _fmt_d(x):
        return f"{int(round(x))}d" if x is not None and np.isfinite(x) else "—"
    def _fmt_bp(x):
        return f"{x:+.2f} bp" if x is not None and np.isfinite(x) else "—"
    def _fmt_z(x):
        return f"{x:+.2f}σ" if x is not None and np.isfinite(x) else "—"

    st.markdown(
        f"<div style='padding:0.6rem 0.9rem; margin-bottom:0.6rem; "
        f"background:rgba(53,169,81,0.03); border-left:3px solid var(--green); "
        f"border-radius:0 6px 6px 0;'>"
        f"<div style='font-size:0.7rem; color:var(--green); "
        f"text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.4rem; "
        f"font-weight:600;'>"
        f"🎯 Exit plan — dynamic, re-evaluated daily</div>"
        f"<div style='display:grid; grid-template-columns: 1fr 1fr; gap:1.0rem; "
        f"font-family:JetBrains Mono, monospace; font-size:0.72rem;'>"
        # Targets column
        f"<div>"
        f"<div style='color:var(--green); font-weight:600; margin-bottom:0.3rem;'>TARGETS</div>"
        f"<div>T1 full revert — z={_fmt_z(levels['target_t1_z'])}, "
        f"resid {_fmt_bp(levels['target_t1_residual_bp'])}, "
        f"est. day {_fmt_d(levels['target_t1_day_estimate'])} · close 100%</div>"
        f"<div>T2 partial — z={_fmt_z(levels['target_t2_z'])}, "
        f"resid {_fmt_bp(levels['target_t2_residual_bp'])} · close 33%</div>"
        f"<div>T3 time stop — day {_fmt_d(levels['target_t3_day'])} "
        f"(1.5×HL, clipped {int(mp['hold_floor'])}-{int(mp['hold_cap'])}d) · close 100%</div>"
        f"<div>T4 signal decay — if confirming sources drop ≥ 50% after 5d · close 50%</div>"
        f"</div>"
        # Stops column
        f"<div>"
        f"<div style='color:var(--red); font-weight:600; margin-bottom:0.3rem;'>STOPS</div>"
        f"<div>S1 adverse breakout — z={_fmt_z(levels['stop_s1_z'])}, "
        f"resid {_fmt_bp(levels['stop_s1_residual_bp'])} · close 100%</div>"
        f"<div>S2 triple-gate fail — stationarity breaks · close 100%</div>"
        f"<div>S3 HL extension — current HL > {_fmt_d(levels['stop_s3_hl_threshold_d'])} · close 100%</div>"
        f"<div>S4 convexity warn — Piterbarg bias fires · close 100%</div>"
        f"<div>S5 regime transition — HMM regime shifts hostile · close 100%</div>"
        f"<div>S6 hard P&L — realized ≤ {_fmt_bp(levels['stop_s6_pnl_bp'])} · close 100%</div>"
        f"</div>"
        f"</div>"
        f"<div style='margin-top:0.45rem; padding-top:0.4rem; "
        f"border-top:1px solid var(--border-subtle); "
        f"font-size:0.66rem; color:var(--text-muted); line-height:1.5;'>"
        f"All tiers checked daily; first to fire wins. Targets ≠ static — "
        f"S3 (HL extension) and S2 (gate failure) react to live model degradation, "
        f"not just price. See <code>lib.trade_exits.evaluate_dynamic_exit</code>."
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_backtest_section(panel: dict) -> None:
    """Standalone backtest section — collapsible expander at bottom of PCA tab.

    Lets the user run the engine backtest with selectable date range / mode and
    surfaces the per-source performance table + equity curve. Writes empirical
    hit rates to disk cache for score_conviction to consume on next refresh.
    """
    from lib.pca_backtest import run_engine_backtest, build_empirical_hit_rates

    with st.expander("📊 Backtest the engine — empirical hit rates + equity curve",
                       expanded=False):
        st.markdown(
            "<div style='font-size:0.78rem; color:var(--text-muted); "
            "margin-bottom:0.5rem; line-height:1.6;'>"
            "Replays the engine over historical data using dynamic 4-target / "
            "6-stop tiered exits with slippage + commission costs. Results "
            "feed back into <code>score_conviction.empirical_hit_rate</code> "
            "on next page refresh. <b>Walk-forward mode is unbiased but slow</b> "
            "(re-fits PCA at each step); fast mode reuses one panel."
            "</div>",
            unsafe_allow_html=True,
        )
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            n_days = st.number_input("Backtest length (trading days)",
                                       min_value=30, max_value=2000, value=250,
                                       step=10, key=f"{_SCOPE}_bt_ndays")
        with c2:
            bt_mode = st.selectbox("Backtest mode",
                                     ["intraday", "swing", "positional"],
                                     index=2,
                                     key=f"{_SCOPE}_bt_mode")
        with c3:
            min_conv = st.number_input("Min conviction",
                                         min_value=0.0, max_value=1.0,
                                         value=0.50, step=0.05,
                                         key=f"{_SCOPE}_bt_minconv")
        with c4:
            wf = st.checkbox("Walk-forward (slow but unbiased)",
                              value=False, key=f"{_SCOPE}_bt_wf")
        run_btn = st.button("Run backtest", key=f"{_SCOPE}_bt_run")
        if run_btn:
            asof = panel.get("asof")
            if asof is None:
                st.error("No snapshot date available.")
                return
            start = asof - timedelta(days=int(n_days * 1.5))    # bday rough
            with st.spinner(f"Running {bt_mode} backtest over ~{n_days}d…"):
                progress = st.progress(0.0)
                def _cb(i, total, d):
                    if total > 0:
                        progress.progress(min(1.0, (i + 1) / total))
                result = run_engine_backtest(
                    start_date=start, end_date=asof, mode=bt_mode,
                    min_conviction=float(min_conv),
                    walk_forward=bool(wf), walk_step_days=5,
                    max_trades_per_day=3,
                    progress_callback=_cb,
                )
                progress.empty()

            summ = result.summary
            st.markdown(
                f"<div style='display:grid; "
                f"grid-template-columns:repeat(5,1fr); gap:0.8rem; "
                f"margin-top:0.4rem; font-family:JetBrains Mono,monospace; "
                f"font-size:0.78rem;'>"
                f"<div><div style='color:var(--text-dim); font-size:0.66rem;'>"
                f"N TRADES</div><div>{summ.get('n_trades',0)}</div></div>"
                f"<div><div style='color:var(--text-dim); font-size:0.66rem;'>"
                f"HIT RATE</div><div>{summ.get('hit_rate',0):.1%}</div></div>"
                f"<div><div style='color:var(--text-dim); font-size:0.66rem;'>"
                f"TOTAL PNL ($)</div><div>${summ.get('total_pnl_dollar',0):,.0f}</div></div>"
                f"<div><div style='color:var(--text-dim); font-size:0.66rem;'>"
                f"MAX DD ($)</div><div>${summ.get('max_drawdown_dollar',0):,.0f}</div></div>"
                f"<div><div style='color:var(--text-dim); font-size:0.66rem;'>"
                f"SHARPE/TRADE</div><div>{summ.get('sharpe_per_trade',0):.2f}</div></div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            if not result.equity_curve.empty:
                try:
                    import plotly.graph_objects as _go
                    fig = _go.Figure()
                    fig.add_trace(_go.Scatter(
                        x=result.equity_curve.index,
                        y=result.equity_curve.values,
                        mode="lines", name="Cum P&L ($)",
                    ))
                    fig.update_layout(height=220, margin=dict(l=4, r=4, t=4, b=4),
                                        plot_bgcolor="rgba(0,0,0,0)",
                                        paper_bgcolor="rgba(0,0,0,0)",
                                        font=dict(size=10),
                                        xaxis=dict(showgrid=False),
                                        yaxis=dict(showgrid=True,
                                                    gridcolor="rgba(255,255,255,0.06)"))
                    st.plotly_chart(fig, use_container_width=True,
                                     key=f"{_SCOPE}_bt_eq",
                                     config={"displayModeBar": False})
                except Exception:
                    pass

            if not result.by_source.empty:
                st.markdown("**Per-source performance** (Sharpe-sorted)")
                st.dataframe(result.by_source.style.format({
                    "hit_rate": "{:.1%}", "avg_pnl_bp": "{:+.2f}",
                    "median_pnl_bp": "{:+.2f}", "total_pnl_dollar": "${:,.0f}",
                    "avg_days_held": "{:.1f}", "std_pnl_bp": "{:.2f}",
                    "sharpe": "{:.2f}",
                }), use_container_width=True, height=240)

            if not result.by_conviction_bucket.empty:
                st.markdown("**Calibration** — hit rate per conviction bucket")
                st.dataframe(result.by_conviction_bucket.style.format({
                    "hit_rate": "{:.1%}", "avg_pnl_bp": "{:+.2f}",
                }), use_container_width=True)

            # Cache empirical hit rates for score_conviction
            try:
                import json
                from pathlib import Path
                cache_dir = Path("D:/STIRS_DASHBOARD/cache")
                cache_dir.mkdir(parents=True, exist_ok=True)
                ehr = build_empirical_hit_rates(result)
                with (cache_dir / f"backtest_empirical_hit_rates_{bt_mode}.json").open(
                        "w", encoding="utf-8") as f:
                    json.dump(ehr, f, indent=2)
                st.success(f"Empirical hit rates cached for {bt_mode} mode — "
                            f"score_conviction will use them on next refresh.")
            except Exception as e:
                st.warning(f"Could not cache hit rates: {e}")


def _render_trade_chart_conviction(idea) -> None:
    """Horizontal bar chart of the conviction-breakdown components."""
    try:
        import plotly.graph_objects as go
    except Exception:
        return
    cb = idea.conviction_breakdown or ()
    if not cb:
        st.markdown("<div style='color:var(--text-dim); font-size:0.7rem; "
                     "padding:1rem 0; text-align:center;'>"
                     "No conviction breakdown available.</div>",
                     unsafe_allow_html=True)
        return
    # Sort by absolute contribution
    sorted_cb = sorted(cb, key=lambda x: x[1])
    names = [c[0] for c in sorted_cb]
    values = [c[1] for c in sorted_cb]
    colors = ["#35a951" if v > 0.001 else "#e15a5a" if v < -0.001 else "#444"
                for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in values],
        textposition="outside",
        textfont=dict(size=9, family="JetBrains Mono", color="#a8a8b3"),
    ))
    fig.add_vline(x=0, line=dict(color="#666", width=1))
    fig.update_layout(
        height=240, margin=dict(l=10, r=10, t=10, b=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono", color="#a8a8b3", size=9),
        xaxis=dict(gridcolor="#2a2a35", showgrid=True, gridwidth=0.5,
                     title=f"contribution → conviction = {idea.conviction:.3f}"),
        yaxis=dict(gridcolor="#2a2a35", showgrid=False, tickfont=dict(size=8)),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True,
                     key=f"conv_{idea.idea_id}",
                     config={"displayModeBar": False})


# =============================================================================
# Per-CONTRACT deep dossier — full 18-section drill-in
# Sections: Identity → Pricing → Position-on-curve → Factor sensitivity →
#           Recent stretch → Stationarity → 3 FV views → Active ideas →
#           Event sensitivity → Calendar → Momentum → Cycle/regime →
#           Hedge candidates → Positioning → Charts (4) → Execution profile →
#           Quick actions → Provenance footer
# =============================================================================
def _render_contract_dossier(symbol: str, panel: dict, ideas: list) -> None:
    """Deep dossier for a single contract — every analysis the engine has run."""
    asof = panel.get("asof", date.today())
    # Build the data once
    try:
        D = contract_dossier_data(symbol, panel, ideas=ideas)
    except Exception as e:
        st.error(f"Dossier build failed for {symbol}: {e}")
        return

    # Header with close button
    cols = st.columns([5, 1])
    with cols[0]:
        st.markdown(
            f"<div style='display:flex; align-items:baseline; gap:0.75rem; "
            f"padding:0.6rem 0.8rem; "
            f"background:rgba(232,183,93,0.08); border:1px solid var(--accent); "
            f"border-radius:6px; margin: 0.5rem 0;'>"
            f"<span style='font-size:1.1rem; font-weight:600; "
            f"font-family:JetBrains Mono, monospace; color:var(--accent);'>"
            f"📋 {symbol}</span>"
            f"<span style='color:var(--text-dim); font-size:0.85rem;'>"
            f"deep contract dossier — every analysis the engine has run on this contract</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with cols[1]:
        if st.button("✕ close", key=f"{_SCOPE}_close_dossier", use_container_width=True):
            _clear_contract()
            st.rerun()

    strategy = D["identity"].get("kind", "outright")
    outright_close = panel.get("outright_close_panel", pd.DataFrame())
    outrights_df = panel.get("outrights_df", pd.DataFrame())

    # ============================================================================
    # 🎯 PHASE F.1 — TRADE THESIS CARD (hero, above all sections)
    # ============================================================================
    _render_trade_thesis_card(D, symbol)

    # ============================================================================
    # 🌐 PHASE F.7 — Cross-asset context one-liner (right under thesis)
    # ============================================================================
    pe = D.get("plain_english", {})
    if pe.get("cross_asset"):
        st.markdown(
            f"<div style='font-size:0.78rem; color:var(--text-body); "
            f"padding:0.4rem 0.8rem; margin: 0.2rem 0 0.5rem 0; "
            f"background:rgba(94,180,255,0.04); border-left:2px solid var(--blue); "
            f"border-radius:0 4px 4px 0;'>"
            f"🌐 <b>Cross-asset tape</b>: {pe['cross_asset']}</div>",
            unsafe_allow_html=True,
        )

    # === Two-column main layout ===
    main_col, side_col = st.columns([2.2, 1.0])

    # ---------------------------- LEFT COLUMN ----------------------------
    with main_col:
        # 1. Identity ----------------------------------------------------
        _section_header("1 · Identity")
        _render_identity(symbol, D["identity"])

        # 2. Pricing -----------------------------------------------------
        _section_header("2 · Current pricing")
        _render_pricing(D["pricing"])

        # 3. Position on curve (CMC chart with this contract dotted) -----
        _section_header("3 · Position on the curve")
        _render_position_on_curve(D["position_on_curve"], symbol)

        # 4. Curve-factor sensitivity (plain English) --------------------
        _section_header("4 · Curve-factor sensitivity", concept_key="PC1")
        _render_factor_sensitivity(D["factor_sensitivity"])

        # 5. Recent stretch (multi-lookback z) ---------------------------
        _section_header("5 · Recent stretch", concept_key="residual_z")
        _render_recent_stretch(D["recent_stretch"])

        # 6. Stationarity diagnostics ------------------------------------
        _section_header("6 · Stationarity diagnostics")
        # Phase F.2 — plain-English layer
        pe_st = D.get("plain_english", {}).get("stationarity")
        if pe_st:
            st.markdown(
                f"<div style='font-size:0.85rem; padding:0.4rem 0.6rem; margin-bottom:0.3rem; "
                f"background:rgba(255,255,255,0.025); border-radius:4px;'>{pe_st}</div>",
                unsafe_allow_html=True,
            )
        _render_stationarity(D["stationarity"])

        # 7. Three FV views ---------------------------------------------
        _section_header("7 · Fair-value views")
        _render_fv_views(D["fv_views"])

        # 8. Active trade ideas involving this contract ------------------
        _section_header(f"8 · Active trade ideas involving {symbol}")
        _render_active_ideas(D["active_ideas"], symbol)

        # 9. Event sensitivity (per ticker) ------------------------------
        _section_header("9 · Event sensitivity (per macro release)")
        _render_event_sensitivity(D["event_sensitivity"])

        # 15. Charts (4-panel) -------------------------------------------
        _section_header("15 · Charts")
        _render_charts(symbol, panel, D)

    # --------------------------- RIGHT COLUMN ---------------------------
    with side_col:
        # 10. Calendar ---------------------------------------------------
        _section_header("10 · Calendar")
        _render_calendar(D["calendar"])

        # 11. Momentum ---------------------------------------------------
        _section_header("11 · Momentum (TSM signs)")
        _render_momentum(D["momentum"])

        # 12. Cycle / regime --------------------------------------------
        _section_header("12 · Cycle / regime")
        # Phase F.3 — plain-English regime layer
        pe_rg = D.get("plain_english", {}).get("regime")
        if pe_rg:
            st.markdown(
                f"<div style='font-size:0.82rem; padding:0.35rem 0.5rem; "
                f"margin-bottom:0.3rem; background:rgba(255,255,255,0.025); "
                f"border-radius:4px;'>{pe_rg}</div>",
                unsafe_allow_html=True,
            )
        _render_regime(D["regime"])

        # 13. Hedge candidates -------------------------------------------
        _section_header("13 · Hedge candidates (PC1-isolated)")
        _render_hedge_candidates(D["hedge_candidates"])

        # 14. Positioning (CFTC COT proxy) ------------------------------
        _section_header("14 · Positioning (CFTC COT)")
        # Phase F.4 — plain-English positioning interpretation
        pe_pos = D.get("plain_english", {}).get("positioning")
        if pe_pos:
            st.markdown(
                f"<div style='font-size:0.82rem; padding:0.35rem 0.5rem; "
                f"margin-bottom:0.3rem; background:rgba(255,255,255,0.025); "
                f"border-radius:4px;'>{pe_pos}</div>",
                unsafe_allow_html=True,
            )
        _render_positioning(D["positioning"])

        # 16. Execution profile ------------------------------------------
        _section_header("16 · Execution profile")
        _render_execution(D["execution"])

        # 17. Quick actions / Compare neighbors --------------------------
        _section_header("17 · Quick actions")
        _render_quick_actions(symbol, panel, strategy)

        # 18. Provenance footer -----------------------------------------
        _section_header("18 · Engine provenance")
        # Phase F.5 — plain-English one-liner; raw table behind disclosure
        pe_pv = D.get("plain_english", {}).get("provenance")
        if pe_pv:
            st.markdown(
                f"<div style='font-size:0.82rem; padding:0.4rem 0.6rem; "
                f"margin-bottom:0.3rem;'>{pe_pv}</div>",
                unsafe_allow_html=True,
            )
        with st.expander("Show raw diagnostics", expanded=False):
            _render_provenance(D["provenance"])


# =============================================================================
# Phase F.1 — Trade Thesis card (hero card at top of dossier)
# =============================================================================
def _render_trade_thesis_card(D: dict, symbol: str) -> None:
    """3-line plain-English thesis synthesizing all signals — top of dossier."""
    thesis = D.get("trade_thesis") or {}
    side = thesis.get("side", "neutral")
    headline = thesis.get("headline", "")
    supporting = thesis.get("supporting_signals", []) or []
    caveats = thesis.get("caveats", []) or []
    confidence = thesis.get("confidence", 0.0)

    # Color by side
    border_color = ("var(--green)" if side == "long"
                     else "var(--red)" if side == "short"
                     else "var(--text-dim)")
    bg_color = ("rgba(53,169,81,0.08)" if side == "long"
                 else "rgba(225,90,90,0.08)" if side == "short"
                 else "rgba(255,255,255,0.02)")

    # Confidence bar (7-block)
    n_full = max(0, min(7, int(round(confidence * 7))))
    conf_bar = "▰" * n_full + "▱" * (7 - n_full)

    sup_html = ""
    if supporting:
        sup_html = (
            "<div style='font-size:0.75rem; margin-top:0.4rem; line-height:1.6;'>"
            "<span style='color:var(--text-dim); font-size:0.68rem; "
            "text-transform:uppercase; letter-spacing:0.05em; margin-right:0.4rem;'>"
            "Supporting:</span>"
            "<span style='color:var(--green);'>"
            + " · ".join(f"✓ {s}" for s in supporting) +
            "</span></div>"
        )
    cav_html = ""
    if caveats:
        cav_html = (
            "<div style='font-size:0.75rem; margin-top:0.25rem; line-height:1.6;'>"
            "<span style='color:var(--text-dim); font-size:0.68rem; "
            "text-transform:uppercase; letter-spacing:0.05em; margin-right:0.4rem;'>"
            "Caveats:</span>"
            "<span style='color:var(--amber);'>"
            + " · ".join(f"⚠ {c}" for c in caveats) +
            "</span></div>"
        )

    st.markdown(
        f"<div style='padding:0.7rem 1rem; margin: 0.4rem 0 0.5rem 0; "
        f"background:{bg_color}; "
        f"border-left:4px solid {border_color}; border-radius:0 6px 6px 0;'>"
        f"<div style='display:flex; justify-content:space-between; align-items:baseline;'>"
        f"<div style='font-size:0.95rem; color:var(--text-heading);'>"
        f"🎯 <b>TRADE THESIS — {symbol}</b></div>"
        f"<div style='font-family:JetBrains Mono, monospace; font-size:0.72rem; color:var(--text-dim);'>"
        f"confidence {conf_bar} {confidence:.2f}</div>"
        f"</div>"
        f"<div style='font-size:0.95rem; margin-top:0.3rem; color:var(--text-body);'>"
        f"{headline}</div>"
        f"{sup_html}{cav_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


# =============================================================================
# Per-section render helpers — each consumes a slice of the dossier dict
# =============================================================================
def _render_identity(symbol: str, idn: dict) -> None:
    rows = [("symbol", f"<b>{symbol}</b>", "accent", None),
            ("kind",   idn.get("kind", "—"), None, None)]
    if idn.get("kind") == "outright":
        if "month_name" in idn and "year" in idn:
            rows.append(("delivery", f"{idn['month_name']} {idn['year']}", None, None))
        if "ref_start" in idn and "ref_end" in idn:
            rows.append(("ref quarter", f"{idn['ref_start']} → {idn['ref_end']}",
                          None, f"{idn.get('ref_days','—')}d"))
        if "tenor_months" in idn:
            rows.append(("tenor (M from asof)", f"{idn['tenor_months']:.1f}M", None, None))
        rows.append(("quarterly?", "✓ H/M/U/Z" if idn.get("quarterly") else "serial",
                      "green" if idn.get("quarterly") else None, None))
        if idn.get("pack_name"):
            rows.append(("pack", idn["pack_name"], "blue",
                          f"position {idn.get('pack_position','—')}/4"))
    _kv_block(rows)


def _render_pricing(p: dict) -> None:
    if not p:
        st.markdown("<div style='color:var(--text-dim); font-size:0.78rem;'>"
                     "No pricing data.</div>", unsafe_allow_html=True)
        return
    rows = []
    if "last_close" in p:
        rows.append(("last close", f"{p['last_close']:.4f}", "accent", None))
    if "implied_3m_fwd_yield_pct" in p:
        rows.append(("implied 3M fwd yield", f"{p['implied_3m_fwd_yield_pct']:.4f}%", None, None))
    if "day_change_bp" in p:
        ch = p["day_change_bp"]
        rows.append(("today's Δ", f"{ch:+.2f} bp",
                      "red" if ch > 0 else "green" if ch < 0 else None,
                      f"${p.get('day_change_dollar', 0):+,.0f}/contract"))
    if "atr14_bp" in p:
        rows.append(("14d ATR", f"{p['atr14_bp']:.2f} bp", None, None))
    if "realized_vol_252d_bp" in p:
        rows.append(("realized σ (252d)", f"{p['realized_vol_252d_bp']:.2f} bp/d",
                      None, None))
    _kv_block(rows)


def _render_position_on_curve(p: dict, symbol: str) -> None:
    if not p or not p.get("tenors_months"):
        st.markdown("<div style='color:var(--text-dim); font-size:0.78rem;'>"
                     "Position-on-curve data unavailable.</div>", unsafe_allow_html=True)
        return
    tenors = p["tenors_months"]
    yields = p["yields_bp"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tenors, y=yields, mode="lines+markers",
        line=dict(color=ACCENT, width=1.4),
        marker=dict(color=ACCENT, size=4),
        name="CMC",
    ))
    # Highlight this contract's tenor(s)
    for leg, tau, y in (p.get("leg_yields") or []):
        if y is None:
            continue
        fig.add_trace(go.Scatter(
            x=[tau], y=[y], mode="markers",
            marker=dict(color=AMBER, size=14, symbol="circle",
                          line=dict(color=BG_BASE, width=2)),
            name=leg, showlegend=False,
        ))
        fig.add_annotation(x=tau, y=y, text=f"<b>{leg}</b>",
                            yshift=14, font=dict(color=AMBER, size=10),
                            showarrow=False)
    fig.update_layout(
        height=220, margin=dict(l=50, r=15, t=10, b=30),
        xaxis=dict(title="tenor (months)", showgrid=True, gridcolor=BORDER_SUBTLE),
        yaxis=dict(title="yield (bp)", showgrid=True, gridcolor=BORDER_SUBTLE),
        showlegend=False,
    )
    st.plotly_chart(_no_title(fig), use_container_width=True,
                      key=f"{_SCOPE}_dossier_curve_{symbol}")


def _render_factor_sensitivity(p: dict) -> None:
    if not p or "loadings" not in p:
        st.markdown("<div style='color:var(--text-dim); font-size:0.78rem;'>"
                     "No factor-sensitivity data.</div>", unsafe_allow_html=True)
        return
    L = p["loadings"]
    pl = p.get("plain", {})
    _kv_block([
        ("level (PC1)",      f"{pl.get('level','—')}",     "blue",
          f"{L.get('PC1', 0):+.3f}"),
        ("slope (PC2)",      f"{pl.get('slope','—')}",     None,
          f"{L.get('PC2', 0):+.3f}"),
        ("curvature (PC3)",  f"{pl.get('curvature','—')}", "purple",
          f"{L.get('PC3', 0):+.3f}"),
    ])


def _render_recent_stretch(p: dict) -> None:
    if not p or not p.get("by_lookback"):
        st.markdown("<div style='color:var(--text-dim); font-size:0.78rem;'>"
                     "Insufficient history for z-score multi-lookback.</div>",
                     unsafe_allow_html=True)
        return
    rows = []
    for n in (5, 15, 30, 60, 90, 252):
        if n in p["by_lookback"]:
            zv = p["by_lookback"][n]
            color = ("red" if abs(zv) >= 2 and zv > 0
                      else "green" if abs(zv) >= 2 and zv < 0
                      else "amber" if abs(zv) >= 1 else None)
            rows.append((f"{n}d z", f"{zv:+.2f}σ", color, None))
    if "pct_252d" in p:
        rows.append(("1y percentile", f"{p['pct_252d']:.0f}%", None, None))
    _kv_block(rows)


def _render_stationarity(p: dict) -> None:
    if not p:
        st.markdown("<div style='color:var(--text-dim); font-size:0.78rem;'>"
                     "Insufficient history for stationarity tests.</div>",
                     unsafe_allow_html=True)
        return
    rows = []
    adf = p.get("adf", {})
    if adf and adf.get("tstat") is not None:
        rows.append(("ADF (no trend, 1 lag)",
                      "✓ stationary @5%" if adf.get("reject_5pct")
                      else "✗ unit root",
                      "green" if adf.get("reject_5pct") else "red",
                      f"t={adf.get('tstat'):.2f}, p={adf.get('pvalue','—')}"))
    kpss = p.get("kpss", {})
    if kpss and kpss.get("tstat") is not None:
        rows.append(("KPSS (level)",
                      "✗ rejects level-stationary" if kpss.get("reject_5pct")
                      else "✓ level-stationary",
                      "red" if kpss.get("reject_5pct") else "green",
                      f"t={kpss.get('tstat'):.3f} (crit 0.463)"))
    vr = p.get("vr_q4", {})
    if vr and vr.get("vr") is not None:
        verdict = ("trending" if vr["vr"] > 1.10
                    else "reverting" if vr["vr"] < 0.90 else "random walk")
        col = ("red" if vr["vr"] > 1.10 else "green" if vr["vr"] < 0.90 else None)
        rows.append(("Variance Ratio q=4",
                      f"{vr['vr']:.3f} ({verdict})", col,
                      f"z={vr.get('z_stat', 0):.2f}"))
    vr12 = p.get("vr_q12", {})
    if vr12 and vr12.get("vr") is not None:
        rows.append(("Variance Ratio q=12",
                      f"{vr12['vr']:.3f}", None,
                      f"z={vr12.get('z_stat', 0):.2f}"))
    if p.get("hurst") is not None:
        rows.append(("Hurst exponent",
                      f"{p['hurst']:.2f} ({p.get('hurst_label','—')})",
                      ("green" if p.get("hurst_label") == "REVERTING"
                       else "red" if p.get("hurst_label") == "TRENDING" else None),
                      None))
    if p.get("half_life_d") is not None:
        hl = p["half_life_d"]
        col = "green" if 5 <= hl <= 30 else "amber" if 3 <= hl <= 60 else "red"
        rows.append(("OU half-life", f"{hl:.1f}d", col, "in band [3,60]"))
    ou = p.get("ou", {})
    if ou and ou.get("kappa") is not None:
        rows.append(("OU κ (mean-reversion speed)",
                      f"{ou['kappa']:.4f}", None, None))
        rows.append(("OU μ (long-run mean)",
                      f"{ou.get('mu', 0):.4f}", "accent", None))
        rows.append(("OU σ (instant vol)",
                      f"{ou.get('sigma', 0):.4f}", None, None))
    _kv_block(rows)


def _render_fv_views(p: dict) -> None:
    if not p:
        st.markdown("<div style='color:var(--text-dim); font-size:0.78rem;'>"
                     "Fair-value views unavailable.</div>", unsafe_allow_html=True)
        return
    # 7a — historical close-z
    if "historical_z_60d" in p:
        zv = p["historical_z_60d"]
        col = ("red" if abs(zv) >= 2 and zv > 0
                else "green" if abs(zv) >= 2 and zv < 0
                else "amber" if abs(zv) >= 1 else None)
        _kv_block([
            ("(a) HISTORICAL · close z (60d)", f"{zv:+.2f}σ", col,
              f"pct {p.get('historical_pct_252d','—'):.0f}%"
              if isinstance(p.get('historical_pct_252d'), (int, float)) else None),
        ])
    # 7b — analog FV (A2)
    afv = p.get("analog_fv")
    if afv:
        rows = [
            ("(b) ANALOG · FV (A2)", f"{_fmt_bp(afv.get('fv_bp'))} bp", "accent", None),
            ("    band 25-75%",
              f"[{_fmt_bp(afv.get('lo_bp'))}, {_fmt_bp(afv.get('hi_bp'))}] bp",
              None, None),
            ("    current deviation",
              f"{_fmt_bp(afv.get('today_bp'))} bp", None, None),
            ("    z vs analog FV",
              f"{_fmt_bp(afv.get('z'), 2)}σ",
              ("red" if afv.get('z', 0) > 1.5 else
               "green" if afv.get('z', 0) < -1.5 else None),
              f"pct {afv.get('pct', 0)*100:.0f}%"
              if afv.get('pct') is not None else None),
            ("    eff_n / n_analogs",
              f"{afv.get('eff_n', 0):.1f} / {afv.get('n_analogs', 0)}",
              ("green" if afv.get('gate') == 'clean'
                else "amber" if afv.get('gate') == 'low_n' else "red"),
              afv.get('gate')),
        ]
        _kv_block(rows)
    # 7c — path-conditional FV (A3 + A4 Heitfield-Park)
    pfv = p.get("path_fv")
    if pfv:
        z_val = pfv.get("z")
        sparse_note = ("sparse-bucket fallback"
                        if pfv.get("sparse_bucket_fallback") else None)
        rows = [
            ("(c) PATH · FV (A3 + A4)",
              f"{_fmt_bp(pfv.get('fv_bp'))} bp", "accent",
              pfv.get("method", "")),
            ("    band 25-75%",
              f"[{_fmt_bp(pfv.get('lo_bp'))}, {_fmt_bp(pfv.get('hi_bp'))}] bp",
              None, None),
            ("    today's residual",
              f"{_fmt_bp(pfv.get('today_bp'))} bp", None, None),
            ("    z vs path FV",
              f"{_fmt_bp(z_val, 2)}σ" if z_val is not None else "—",
              ("red" if (z_val or 0) > 1.5 else
               "green" if (z_val or 0) < -1.5 else None),
              None),
            ("    eff_n (overall)",
              f"{pfv.get('eff_n', 0):.1f}",
              ("green" if pfv.get('gate') == 'clean'
                else "amber" if pfv.get('gate') == 'low_n' else "red"),
              sparse_note or pfv.get('gate')),
        ]
        _kv_block(rows)
        # A4 step-path bucket probabilities (sparkline-ish)
        probs = pfv.get("today_path_probs") or pfv.get("bucket_probs") or {}
        if probs:
            bucket_order = ["large_cut", "cut", "hold", "hike", "large_hike"]
            bars = []
            for b in bucket_order:
                pv = float(probs.get(b, 0.0))
                # 7-block bar width
                n_full = max(0, min(7, int(round(pv * 7))))
                bar = "▰" * n_full + "▱" * (7 - n_full)
                col = ("var(--green)" if b in ("large_cut", "cut")
                        else "var(--red)" if b in ("hike", "large_hike")
                        else "var(--text-body)")
                bars.append(
                    f"<div style='display:flex; justify-content:space-between; "
                    f"font-size:0.7rem; padding:1px 0; "
                    f"font-family:JetBrains Mono, monospace;'>"
                    f"<span style='color:var(--text-muted);'>"
                    f"&nbsp;&nbsp;&nbsp;&nbsp;{b.replace('_', ' '):>11}</span>"
                    f"<span style='color:{col};'>{bar} {pv*100:5.1f}%</span></div>"
                )
            st.markdown(
                "<div style='font-size:0.7rem; color:var(--text-dim); "
                "margin: 0.3rem 0 0.1rem 0;'>A4 step-path probabilities (today, weighted across 8 future FOMCs):</div>"
                + "".join(bars), unsafe_allow_html=True,
            )
    elif p.get("path_fv_error"):
        st.markdown(
            f"<div style='font-size:0.7rem; color:var(--text-dim);'>"
            f"path FV unavailable: {p['path_fv_error']}</div>",
            unsafe_allow_html=True,
        )
    # 7d — carry/roll
    cr = p.get("carry_roll")
    if cr:
        rows = []
        if "carry_3m_bp" in cr:
            rows.append(("(d) CARRY · 3M",
                          f"{cr['carry_3m_bp']:+.2f} bp",
                          "green" if cr['carry_3m_bp'] > 0 else "red", None))
        if "roll_3m_bp" in cr:
            rows.append(("    ROLL · 3M (into next quarterly)",
                          f"{cr['roll_3m_bp']:+.2f} bp",
                          "green" if cr['roll_3m_bp'] > 0 else "red",
                          cr.get("roll_into_symbol")))
        if "carry_plus_roll_3m_bp" in cr:
            tot = cr['carry_plus_roll_3m_bp']
            rows.append(("    CARRY + ROLL TOTAL",
                          f"<b>{tot:+.2f} bp</b>",
                          "green" if tot > 0 else "red", "per 3M hold"))
        if rows:
            _kv_block(rows)
    # 7e — A10 turn / QE / YE adjuster
    ta = p.get("turn_adjustment")
    if ta and ta.get("applies"):
        rows = []
        types_str = ", ".join(ta.get("turn_types", []))
        rows.append(("(e) A10 TURN · spans",
                      types_str, "amber",
                      f"{len(ta.get('turn_dates', []))} turn(s)"))
        if "premium_bp" in ta and abs(ta["premium_bp"]) > 0.01:
            adj = ta["premium_bp"]
            rows.append(("    estimated turn premium",
                          f"{adj:+.2f} bp",
                          "red" if adj > 0 else "green",
                          f"{ta.get('n_historical_turns', 0)} turns histo"))
            rows.append(("    de-turn fair-yield adj",
                          f"−{adj:+.2f} bp",
                          None, "subtract from raw fwd yield"))
        if rows:
            _kv_block(rows)


def _render_active_ideas(ideas_data: list, symbol: str) -> None:
    """Rich Trade-Card rendering — one card per active idea involving `symbol`.

    Each card shows:
      - Headline with conviction pill + side
      - ACTUAL SR3 contract prices (entry / target / stop / RR / $/contract)
      - 90-day mini chart with FV path + entry/target/stop markers
      - Plain-English explanation of WHY this action is what it is
      - Per-factor interpretation (10+ factors, each with what-it-means-here)
      - Full rationale chain (4-part narrative from generator)
    """
    if not ideas_data:
        st.markdown("<div style='color:var(--text-dim); font-size:0.78rem; "
                     "padding:0.5rem 0;'>"
                     f"No active trade ideas currently include <b>{symbol}</b>. "
                     f"Either it's not statistically stretched (|z| < 1.5), or "
                     f"its trades are filtered out by current gate settings.</div>",
                     unsafe_allow_html=True)
        return

    for i, item in enumerate(ideas_data[:6]):    # cap at 6 per contract
        _render_one_trade_card(item, symbol, idx=i)


def _render_one_trade_card(item: dict, symbol: str, idx: int = 0) -> None:
    """Render one rich Trade Card. `item` is one element of active_ideas section."""
    idea = item["idea"]
    sr3 = item["sr3_prices"]
    factors = item["factor_interpretations"] or []
    chart_data = item["chart_data_90d"] or {}

    # ---- Header strip ----
    conv = idea.conviction
    conv_color = ("var(--green)" if conv >= 0.7
                    else "var(--amber)" if conv >= 0.5
                    else "var(--text-dim)")
    side_color = ("var(--green)" if idea.direction == "long"
                    else "var(--red)" if idea.direction == "short"
                    else "var(--text-dim)")
    n_full = max(0, min(7, int(round(conv * 7))))
    conv_bar = "▰" * n_full + "▱" * (7 - n_full)

    sources_html = " · ".join(idea.sources)
    sources_chip = (
        f"<div style='font-size:0.7rem; color:var(--text-muted); margin-top:0.2rem;'>"
        f"⊕ {sources_html}</div>"
        if idea.n_confirming_sources > 1 else ""
    )
    st.markdown(
        f"<div style='padding:0.55rem 0.75rem; margin: 0.6rem 0 0.25rem 0; "
        f"background:rgba(232,183,93,0.05); border-left:3px solid {side_color}; "
        f"border-radius:0 6px 6px 0;'>"
        f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
        f"<div style='font-size:0.85rem; color:var(--text-heading);'>"
        f"<b>{_trade_label(idea)}</b> · "
        f"<span style='color:{side_color}; text-transform:uppercase;'>"
        f"{idea.direction}</span> "
        f"<span style='color:var(--text-dim); font-size:0.7rem;'>"
        f"({idea.primary_source})</span></div>"
        f"<div style='font-family:JetBrains Mono, monospace; font-size:0.72rem;'>"
        f"<span style='color:{conv_color};'>{conv_bar} conv {conv:.2f}</span>"
        f"</div></div>"
        f"{sources_chip}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ---- Plain-English headline ----
    if idea.headline_html:
        st.markdown(
            f"<div style='font-size:0.78rem; color:var(--text-body); "
            f"padding:0.3rem 0.75rem; margin-bottom:0.3rem; font-style:italic;'>"
            f"{idea.headline_html}</div>",
            unsafe_allow_html=True,
        )

    # ---- ACTUAL SR3 contract prices (the key user request) ----
    if sr3 is not None and sr3.is_outright:
        action_color = "var(--green)" if sr3.contract_action == "BUY" else "var(--red)"
        rr_str = f"{sr3.risk_reward:.2f}" if sr3.risk_reward else "—"
        st.markdown(
            f"<div style='padding:0.6rem 0.8rem; margin: 0.3rem 0 0.5rem 0; "
            f"background:rgba(255,255,255,0.025); border:1px solid var(--border-default); "
            f"border-radius:6px;'>"
            f"<div style='font-size:0.7rem; color:var(--text-dim); "
            f"text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.4rem;'>"
            f"💵 Actual SR3 contract prices</div>"
            f"<div style='display:grid; grid-template-columns: 1fr 1fr 1fr 1fr 1fr; "
            f"gap:0.6rem; font-family:JetBrains Mono, monospace; font-size:0.78rem;'>"
            f"<div>"
            f"<div style='color:var(--text-dim); font-size:0.65rem;'>ACTION</div>"
            f"<div style='color:{action_color}; font-weight:600; font-size:0.95rem;'>"
            f"{sr3.contract_action}</div></div>"
            f"<div>"
            f"<div style='color:var(--text-dim); font-size:0.65rem;'>ENTRY</div>"
            f"<div style='color:var(--text-heading); font-size:0.9rem;'>"
            f"{sr3.entry_price:.4f}</div></div>"
            f"<div>"
            f"<div style='color:var(--text-dim); font-size:0.65rem;'>TARGET</div>"
            f"<div style='color:var(--green); font-size:0.9rem;'>"
            f"{sr3.target_price:.4f}</div></div>"
            f"<div>"
            f"<div style='color:var(--text-dim); font-size:0.65rem;'>STOP</div>"
            f"<div style='color:var(--red); font-size:0.9rem;'>"
            f"{sr3.stop_price:.4f}</div></div>"
            f"<div>"
            f"<div style='color:var(--text-dim); font-size:0.65rem;'>R:R · $/c</div>"
            f"<div style='color:var(--text-body); font-size:0.85rem;'>"
            f"{rr_str} · ${sr3.pnl_per_contract_dollar:.0f}</div></div>"
            f"</div>"
            f"<div style='margin-top:0.4rem; padding-top:0.4rem; "
            f"border-top:1px solid var(--border-subtle); "
            f"font-size:0.72rem; color:var(--text-muted); line-height:1.5;'>"
            f"{sr3.explanation}"
            f"</div>"
            f"<div style='margin-top:0.3rem; font-size:0.65rem; color:var(--text-dim); "
            f"font-family:JetBrains Mono, monospace;'>"
            f"YIELD-SPACE: entry {sr3.entry_yield_bp:.2f} bp · "
            f"target {sr3.target_yield_bp:.2f} bp · "
            f"stop {sr3.stop_yield_bp:.2f} bp"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    elif sr3 is not None and not sr3.is_outright:
        # Multi-leg trade: show per-leg prices + net residual
        leg_rows = []
        for lp in sr3.per_leg_prices:
            act_color = "var(--green)" if lp["action"] == "BUY" else "var(--red)"
            price_str = (f"{lp['entry_price']:.4f}" if lp["entry_price"] is not None
                          else "—")
            leg_rows.append(
                f"<div style='display:grid; grid-template-columns: 1.5fr 0.6fr 0.7fr 0.5fr; "
                f"gap:0.3rem; padding:2px 0; font-size:0.73rem; "
                f"font-family:JetBrains Mono, monospace;'>"
                f"<span style='color:var(--text-body);'>{lp['symbol']}</span>"
                f"<span style='color:{act_color}; font-weight:600;'>{lp['action']}</span>"
                f"<span style='color:var(--text-heading); text-align:right;'>{price_str}</span>"
                f"<span style='color:var(--text-muted); text-align:right;'>"
                f"w {lp['pv01_wt']:+.2f}</span>"
                f"</div>"
            )
        st.markdown(
            f"<div style='padding:0.5rem 0.7rem; margin: 0.3rem 0 0.5rem 0; "
            f"background:rgba(255,255,255,0.025); border:1px solid var(--border-default); "
            f"border-radius:6px;'>"
            f"<div style='font-size:0.7rem; color:var(--text-dim); "
            f"text-transform:uppercase; letter-spacing:0.06em; margin-bottom:0.4rem;'>"
            f"💵 Per-leg execution ({idea.structure_type})</div>"
            + "".join(leg_rows)
            + f"<div style='margin-top:0.4rem; padding-top:0.4rem; "
            f"border-top:1px solid var(--border-subtle); font-family:JetBrains Mono, monospace; "
            f"font-size:0.72rem;'>"
            f"<span style='color:var(--text-dim);'>NET residual:</span> "
            f"<span style='color:var(--text-body);'>"
            f"entry {sr3.net_entry_bp:+.2f} bp · "
            f"target {sr3.net_target_bp:+.2f} bp · "
            f"stop {sr3.net_stop_bp:+.2f} bp</span></div>"
            f"<div style='margin-top:0.3rem; font-size:0.7rem; color:var(--text-muted); line-height:1.5;'>"
            f"{sr3.explanation}</div></div>",
            unsafe_allow_html=True,
        )

    # ---- 90-day chart with entry/target/stop overlays ----
    if chart_data.get("dates"):
        _render_trade_chart_90d(chart_data, sr3, symbol, idx)

    # ---- Factor-by-factor interpretation (the deep "why") ----
    with st.expander(f"📊 Factor-by-factor interpretation ({len(factors)} factors analyzed)",
                       expanded=False):
        if not factors:
            st.markdown("<div style='color:var(--text-dim);'>No factor interpretations computed.</div>",
                          unsafe_allow_html=True)
        else:
            for f in factors:
                tier_color_map = {"supportive": "var(--green)",
                                     "neutral": "var(--text-body)",
                                     "caveat": "var(--amber)"}
                tcolor = tier_color_map.get(f.tier, "var(--text-body)")
                tier_icon = {"supportive": "✓", "neutral": "·", "caveat": "⚠"}.get(f.tier, "·")
                # Concept tooltip (full explanation)
                from lib.pca_concepts import get_concept
                concept = get_concept(f.key)
                concept_name = concept.get("name", f.key) if concept else f.key
                wt_str = (f" (+{f.weight_in_conviction:.3f})"
                           if f.weight_in_conviction > 0
                           else (f" ({f.weight_in_conviction:+.3f})"
                                  if f.weight_in_conviction < 0 else ""))
                st.markdown(
                    f"<div style='padding:0.5rem 0.6rem; margin: 0.25rem 0; "
                    f"border-left:3px solid {tcolor}; "
                    f"background:rgba(255,255,255,0.02); border-radius:0 4px 4px 0;'>"
                    f"<div style='display:flex; justify-content:space-between; align-items:baseline;'>"
                    f"<div>"
                    f"<span style='color:{tcolor}; font-weight:500;'>{tier_icon} {concept_name}</span> "
                    f"<span style='color:var(--text-dim); font-size:0.7rem;'>"
                    f"= {f.value_display}{wt_str}</span></div>"
                    f"</div>"
                    f"<div style='font-size:0.78rem; color:var(--text-body); margin-top:0.2rem;'>"
                    f"<b>{f.headline}</b></div>"
                    f"<div style='font-size:0.72rem; color:var(--text-muted); margin-top:0.2rem; line-height:1.55;'>"
                    f"{f.detail}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                # Concept disclosure (full definition)
                if concept and concept.get("what_it_measures"):
                    with st.expander(f"  ↳ What is {concept_name}?", expanded=False):
                        from lib.pca_concepts import concept_disclosure_html
                        st.markdown(concept_disclosure_html(f.key),
                                     unsafe_allow_html=True)

    # ---- Full rationale (4-part narrative) ----
    if idea.rationale_html:
        with st.expander("📝 Full rationale (4-part narrative)", expanded=False):
            st.markdown(
                f"<div style='font-size:0.78rem; padding:0.5rem; "
                f"background:rgba(232,183,93,0.04); border-radius:4px;'>"
                f"{idea.rationale_html}</div>",
                unsafe_allow_html=True,
            )


def _render_trade_chart_90d(chart_data: dict, sr3, symbol: str, idx: int) -> None:
    """Render the 90-day SR3 price chart with FV path + entry/target/stop markers."""
    try:
        import plotly.graph_objects as go
    except Exception:
        return
    dates = chart_data.get("dates", [])
    prices = chart_data.get("prices", [])
    fv_prices = chart_data.get("fv_prices", [])
    if not dates or not prices:
        return
    fig = go.Figure()
    # Actual close price
    fig.add_trace(go.Scatter(
        x=dates, y=prices, mode="lines", name=f"{symbol} close",
        line=dict(color="#e8b75d", width=2),
    ))
    # FV implied price
    if fv_prices and any(v is not None for v in fv_prices):
        fig.add_trace(go.Scatter(
            x=dates, y=fv_prices, mode="lines", name="PCA-implied FV",
            line=dict(color="#5eb4ff", width=1, dash="dot"),
        ))
    # Entry / target / stop markers (only if outright)
    if sr3 and sr3.is_outright and sr3.entry_price:
        for label, y, color in [
            ("ENTRY",  sr3.entry_price,  "#e8b75d"),
            ("TARGET", sr3.target_price, "#35a951"),
            ("STOP",   sr3.stop_price,   "#e15a5a"),
        ]:
            fig.add_hline(y=y, line=dict(color=color, width=1, dash="dash"),
                            annotation_text=f"{label} {y:.4f}",
                            annotation_position="right",
                            annotation_font_color=color)
    fig.update_layout(
        height=240, margin=dict(l=10, r=80, t=10, b=20),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono", color="#a8a8b3", size=10),
        xaxis=dict(gridcolor="#2a2a35", showgrid=True, gridwidth=0.5),
        yaxis=dict(gridcolor="#2a2a35", showgrid=True, gridwidth=0.5,
                     title="SR3 price"),
        showlegend=True,
        legend=dict(orientation="h", x=0, y=1.15, font=dict(size=9)),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True,
                     key=f"trade_chart_{symbol}_{idx}_{id(sr3)}")


def _render_event_sensitivity(p: dict) -> None:
    if not p:
        st.markdown("<div style='color:var(--text-dim); font-size:0.78rem;'>"
                     "No macro-event sensitivity (release dates outside contract sample, "
                     "or contract too new).</div>", unsafe_allow_html=True)
        return
    # Header row
    st.markdown(
        "<div style='display:grid; grid-template-columns: 1.0fr 0.7fr 0.6fr 0.6fr 0.7fr 0.7fr 0.6fr; "
        "gap:8px; padding:5px 6px; border-bottom:1px solid var(--border-default); "
        "font-size:0.7rem; color:var(--text-muted); text-transform:uppercase; "
        "font-weight:600; font-family:JetBrains Mono, monospace;'>"
        "<span>event</span>"
        "<span style='text-align:right;'>β bp/σ</span>"
        "<span style='text-align:right;'>t</span>"
        "<span style='text-align:right;'>R²</span>"
        "<span style='text-align:right;'>5d drift</span>"
        "<span style='text-align:right;'>20d drift</span>"
        "<span style='text-align:right;'>n</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    for evt, d in p.items():
        if evt == "FOMC":
            # FOMC has no surprise series — show mean dy and post-drift
            mean_dy = d.get("mean_dy_bp")
            std_dy = d.get("std_dy_bp")
            n = d.get("n_events", 0)
            d5 = d.get("post_drift_5d_bp")
            d20 = d.get("post_drift_20d_bp")
            beta_str = (f"{mean_dy:+.2f} (μ)"
                          if mean_dy is not None else "—")
            t_str = (f"{mean_dy/std_dy:+.2f}"
                      if mean_dy is not None and std_dy and std_dy > 0 else "—")
            r2 = "—"
        else:
            beta = d.get("beta_bp_per_sigma")
            t = d.get("tstat")
            r2 = d.get("r2")
            n = d.get("n_events", 0)
            d5 = d.get("post_drift_5d_bp")
            d20 = d.get("post_drift_20d_bp")
            beta_str = f"{beta:+.2f}" if beta is not None else "—"
            t_str = f"{t:+.2f}" if t is not None else "—"
            r2 = f"{r2*100:.1f}%" if r2 is not None else "—"
        d5_str = f"{d5:+.1f}" if d5 is not None else "—"
        d20_str = f"{d20:+.1f}" if d20 is not None else "—"
        # Significance highlight
        beta_color = ("var(--text-body)")
        if evt != "FOMC" and isinstance(d.get("tstat"), (int, float)):
            if abs(d["tstat"]) >= 2.0:
                beta_color = ("var(--red)" if d["beta_bp_per_sigma"] > 0
                                else "var(--green)")
            elif abs(d["tstat"]) >= 1.5:
                beta_color = "var(--amber)"
        st.markdown(
            f"<div style='display:grid; grid-template-columns: 1.0fr 0.7fr 0.6fr 0.6fr 0.7fr 0.7fr 0.6fr; "
            f"gap:8px; padding:3px 6px; border-bottom:1px solid var(--border-subtle); "
            f"font-family:JetBrains Mono, monospace; font-size:0.72rem;'>"
            f"<span style='color:var(--text-body);'>{evt}</span>"
            f"<span style='color:{beta_color}; text-align:right; font-weight:500;'>{beta_str}</span>"
            f"<span style='color:var(--text-muted); text-align:right;'>{t_str}</span>"
            f"<span style='color:var(--text-muted); text-align:right;'>{r2}</span>"
            f"<span style='color:var(--text-muted); text-align:right;'>{d5_str}</span>"
            f"<span style='color:var(--text-muted); text-align:right;'>{d20_str}</span>"
            f"<span style='color:var(--text-dim); text-align:right;'>{n}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


def _render_calendar(p: dict) -> None:
    if not p:
        st.markdown("<div style='color:var(--text-dim); font-size:0.78rem;'>"
                     "No upcoming calendar.</div>", unsafe_allow_html=True)
        return
    rows = []
    events = p.get("events", [])
    for e in events[:6]:
        dout = e.get("days_out", 0)
        col = "red" if dout <= 7 else "amber" if dout <= 21 else None
        rows.append((e.get("event","—"), str(e.get("date","—")),
                      col, f"{dout}d"))
    if "days_to_expiry" in p:
        dte = p["days_to_expiry"]
        col = "red" if dte < 14 else "amber" if dte < 45 else None
        rows.append(("days to last-trade", f"{dte}d", col,
                      str(p.get("last_trade_date","—"))))
    if rows:
        _kv_block(rows)
    else:
        st.markdown("<div style='color:var(--text-dim); font-size:0.78rem;'>"
                     "No upcoming calendar items.</div>", unsafe_allow_html=True)


def _render_momentum(p: dict) -> None:
    if not p or not p.get("by_horizon"):
        st.markdown("<div style='color:var(--text-dim); font-size:0.78rem;'>"
                     "Insufficient history for momentum signs.</div>",
                     unsafe_allow_html=True)
        return
    rows = []
    for h in (21, 63, 126, 252):
        if h in p["by_horizon"]:
            v = p["by_horizon"][h]
            sign_glyph = "▲" if v["sign"] > 0 else "▼" if v["sign"] < 0 else "◇"
            col = ("green" if v["sign"] > 0
                    else "red" if v["sign"] < 0 else None)
            rows.append((f"{h}d", f"{sign_glyph} {v['ret_bp']:+.1f} bp",
                          col, None))
    cons = p.get("consensus", "—")
    cons_col = ("green" if cons == "long"
                 else "red" if cons == "short" else "amber")
    rows.append(("consensus", cons.upper(), cons_col, None))
    _kv_block(rows)


def _render_regime(p: dict) -> None:
    if not p:
        st.markdown("<div style='color:var(--text-dim); font-size:0.78rem;'>"
                     "Regime stack unavailable.</div>", unsafe_allow_html=True)
        return
    rows = []
    rows.append(("cycle phase", p.get("cycle_phase","—"), "accent", None))
    if "regime_label" in p:
        conf = p.get("regime_confidence", 0)
        rows.append(("HMM regime",
                      f"R{p['regime_label']}",
                      "green" if conf >= 0.6 else "amber",
                      f"{conf*100:.0f}% conf"))
    if "days_since_break" in p:
        ds = p["days_since_break"]
        rows.append(("days since regime break",
                      f"{ds}d", None,
                      str(p.get("last_break_date","—"))))
    if "bocpd_p_today" in p:
        rows.append(("BOCPD p(short-run)",
                      f"{p['bocpd_p_today']*100:.1f}%",
                      "red" if p["bocpd_p_today"] > 0.20 else None,
                      "elevated" if p["bocpd_p_today"] > 0.20 else "stable"))
    _kv_block(rows)


def _render_hedge_candidates(items: list) -> None:
    if not items:
        st.markdown("<div style='color:var(--text-dim); font-size:0.78rem;'>"
                     "No PC1-isolated 4-leg basket for this contract.</div>",
                     unsafe_allow_html=True)
        return
    for h in items[:4]:
        z = h.get("z")
        z_str = f"{z:+.2f}σ" if z is not None and np.isfinite(z) else "—"
        z_color = ("var(--red)" if z is not None and z > 1.5
                    else "var(--green)" if z is not None and z < -1.5
                    else "var(--text-body)")
        side = h.get("side_for_target", "—").upper()
        adf_glyph = "✓" if h.get("adf_pass") else "✗"
        adf_color = "var(--green)" if h.get("adf_pass") else "var(--text-dim)"
        legs_str = " · ".join(h.get("legs", []))
        weights_str = " ".join(f"{w:+.2f}" for w in h.get("weights_normalized", []))
        st.markdown(
            f"<div style='padding:5px 0; border-bottom:1px solid var(--border-subtle); "
            f"font-family:JetBrains Mono, monospace; font-size:0.7rem;'>"
            f"<div style='color:var(--text-body); font-weight:500;'>"
            f"<span style='color:{adf_color};'>{adf_glyph}</span> {legs_str}</div>"
            f"<div style='display:flex; justify-content:space-between; "
            f"color:var(--text-muted); font-size:0.66rem; margin-top:2px;'>"
            f"<span>w: {weights_str}</span>"
            f"<span style='color:{z_color};'>z={z_str}</span>"
            f"<span style='color:var(--accent);'>side: {side}</span>"
            f"</div></div>",
            unsafe_allow_html=True,
        )


def _render_positioning(p: dict) -> None:
    if not p:
        st.markdown("<div style='color:var(--text-dim); font-size:0.78rem;'>"
                     "CFTC COT proxy unavailable.</div>", unsafe_allow_html=True)
        return
    rows = []
    if "last_value" in p:
        rows.append(("UST 2Y net non-comm",
                      f"{p['last_value']:+,.0f}", None,
                      f"{p.get('age_days','—')}d old"))
    if "z_2y" in p and p["z_2y"] is not None:
        z = p["z_2y"]
        col = "red" if abs(z) >= 2 else "amber" if abs(z) >= 1 else None
        rows.append(("z-score (2y)", f"{z:+.2f}σ", col, None))
    if "pct_2y" in p:
        pc = p["pct_2y"]
        col = "red" if pc <= 5 else "green" if pc >= 95 else None
        rows.append(("percentile (2y)", f"{pc:.0f}%", col, None))
    if "wow_change" in p:
        rows.append(("week-on-week Δ",
                      f"{p['wow_change']:+,.0f}", None, None))
    if "pct_of_oi" in p:
        rows.append(("% of OI",
                      f"{p['pct_of_oi']:+.1f}%", None,
                      f"{p.get('open_interest','—'):,.0f}"
                      if isinstance(p.get('open_interest'), (int, float))
                      else None))
    _kv_block(rows)
    if "proxy_used" in p:
        st.markdown(
            f"<div style='font-size:0.65rem; color:var(--text-dim); "
            f"font-style:italic; margin-top:0.25rem;'>"
            f"{p['proxy_used']}</div>",
            unsafe_allow_html=True,
        )


def _render_charts(symbol: str, panel: dict, D: dict) -> None:
    """Charts panel — 1y price+events, residual+OU bands, event-β bar, z-heatmap."""
    # 4 small panels in a 2×2 grid
    c1, c2 = st.columns(2)
    with c1:
        _render_chart_price_events(symbol, panel)
    with c2:
        _render_chart_residual_ou(symbol, D["charts"], panel)
    c3, c4 = st.columns(2)
    with c3:
        _render_chart_event_betas(symbol, D["event_sensitivity"])
    with c4:
        _render_chart_z_heatmap(symbol, D["z_heatmap"])


def _render_chart_price_events(symbol: str, panel: dict) -> None:
    closes = panel.get("outright_close_panel", pd.DataFrame())
    if symbol not in closes.columns:
        st.markdown("<div style='color:var(--text-dim); font-size:0.7rem;'>"
                     "no price history</div>", unsafe_allow_html=True)
        return
    cs = closes[symbol].dropna().tail(252)
    if cs.empty:
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cs.index, y=cs.values, mode="lines",
                              line=dict(color=ACCENT, width=1.4), name=symbol))
    fomc = panel.get("fomc_calendar_dates", []) or []
    for d in fomc:
        ts = pd.Timestamp(d)
        if cs.index.min() <= ts <= cs.index.max():
            fig.add_vline(x=ts, line_color="rgba(232,183,93,0.18)",
                            line_dash="dot", line_width=1)
    fig.update_layout(
        height=210, margin=dict(l=42, r=8, t=22, b=22),
        xaxis=dict(showgrid=True, gridcolor=BORDER_SUBTLE, title=None),
        yaxis=dict(showgrid=True, gridcolor=BORDER_SUBTLE, title="close",
                      tickformat=".4f"),
        showlegend=False,
    )
    st.markdown(
        "<div style='font-size:0.7rem; color:var(--text-dim); "
        "text-align:center; margin-bottom:-0.6rem;'>"
        f"close · 1y · FOMC dotted</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(_no_title(fig), use_container_width=True,
                      key=f"{_SCOPE}_dossier_chart_price_{symbol}")


def _render_chart_residual_ou(symbol: str, charts: dict, panel: dict) -> None:
    if not charts or not charts.get("residual_dates"):
        st.markdown("<div style='color:var(--text-dim); font-size:0.7rem;'>"
                     "no residual series</div>", unsafe_allow_html=True)
        return
    dates = charts["residual_dates"]
    vals = charts["residual_values"]
    ou = charts.get("residual_ou", {}) or {}
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=vals, mode="lines",
                              line=dict(color=BLUE, width=1.2),
                              name="Δ_residual"))
    mu = ou.get("mu")
    sig = ou.get("sigma")
    if mu is not None and sig is not None:
        x_band = [dates[0], dates[-1]]
        for k_sigma, color in [(2, "rgba(225,90,90,0.10)"),
                                (1, "rgba(225,90,90,0.18)")]:
            fig.add_trace(go.Scatter(
                x=x_band + x_band[::-1],
                y=[mu + k_sigma * sig, mu + k_sigma * sig,
                   mu - k_sigma * sig, mu - k_sigma * sig],
                fill="toself", fillcolor=color, line=dict(width=0),
                hoverinfo="skip", showlegend=False,
            ))
        fig.add_hline(y=mu, line_color=AMBER, line_dash="dash", line_width=1)
    fig.update_layout(
        height=210, margin=dict(l=42, r=8, t=22, b=22),
        xaxis=dict(showgrid=True, gridcolor=BORDER_SUBTLE, title=None),
        yaxis=dict(showgrid=True, gridcolor=BORDER_SUBTLE,
                      title="residual (bp)"),
        showlegend=False,
    )
    hl_str = (f"HL {ou.get('halflife_d', 0):.0f}d"
                if ou.get("halflife_d") is not None else "HL —")
    st.markdown(
        f"<div style='font-size:0.7rem; color:var(--text-dim); "
        f"text-align:center; margin-bottom:-0.6rem;'>"
        f"per-contract residual · OU bands ±1σ ±2σ · {hl_str}</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(_no_title(fig), use_container_width=True,
                      key=f"{_SCOPE}_dossier_chart_resid_{symbol}")


def _render_chart_event_betas(symbol: str, ev: dict) -> None:
    if not ev:
        st.markdown("<div style='color:var(--text-dim); font-size:0.7rem;'>"
                     "no event sensitivity</div>", unsafe_allow_html=True)
        return
    rows = []
    for k, v in ev.items():
        if k == "FOMC":
            beta = v.get("mean_dy_bp")
            std = v.get("std_dy_bp", 0) or 0
            ci = 1.96 * std / np.sqrt(max(v.get("n_events", 1), 1))
            tstat = v.get("mean_dy_bp", 0) / max(std, 1e-9) if std else 0
        else:
            beta = v.get("beta_bp_per_sigma")
            ci = 1.96 * (v.get("se_bp", 0) or 0)
            tstat = v.get("tstat", 0) or 0
        if beta is None:
            continue
        rows.append((k, beta, ci, tstat))
    if not rows:
        st.markdown("<div style='color:var(--text-dim); font-size:0.7rem;'>"
                     "all betas null</div>", unsafe_allow_html=True)
        return
    rows.sort(key=lambda r: -abs(r[1]))
    rows = rows[:10]
    names = [r[0] for r in rows]
    betas = [r[1] for r in rows]
    cis = [r[2] for r in rows]
    tstats = [r[3] for r in rows]
    colors = [(RED if (b > 0 and abs(t) >= 2)
               else GREEN if (b < 0 and abs(t) >= 2)
               else AMBER if abs(t) >= 1.5 else "rgba(180,180,180,0.5)")
              for b, t in zip(betas, tstats)]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=betas, y=names, orientation="h",
        marker_color=colors,
        error_x=dict(type="data", array=cis, color="rgba(255,255,255,0.4)",
                       thickness=1.5, width=4),
        hovertemplate="<b>%{y}</b><br>β=%{x:.2f} bp/σ ±%{error_x.array:.2f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="rgba(255,255,255,0.3)", line_width=1)
    fig.update_layout(
        height=210, margin=dict(l=80, r=8, t=22, b=22),
        xaxis=dict(showgrid=True, gridcolor=BORDER_SUBTLE,
                      title="bp / σ surprise"),
        yaxis=dict(showgrid=False, autorange="reversed",
                      tickfont=dict(size=9)),
        showlegend=False,
    )
    st.markdown(
        "<div style='font-size:0.7rem; color:var(--text-dim); "
        "text-align:center; margin-bottom:-0.6rem;'>"
        "event β (bp / σ-surprise) · 95% CI bars</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(_no_title(fig), use_container_width=True,
                      key=f"{_SCOPE}_dossier_chart_betas_{symbol}")


def _render_chart_z_heatmap(symbol: str, zh: dict) -> None:
    if not zh or not zh.get("rows"):
        st.markdown("<div style='color:var(--text-dim); font-size:0.7rem;'>"
                     "no z-heatmap data</div>", unsafe_allow_html=True)
        return
    rows = zh["rows"]
    lookbacks = [r["lookback"] for r in rows]
    z_close = [r.get("close_z", np.nan) for r in rows]
    z_resid = [r.get("residual_z", np.nan) for r in rows]
    fig = go.Figure(data=[go.Heatmap(
        z=[z_close, z_resid],
        x=[f"{l}d" for l in lookbacks],
        y=["close", "residual"],
        colorscale=[
            [0.0, "rgb(53, 169, 81)"],
            [0.5, "rgb(48, 53, 70)"],
            [1.0, "rgb(225, 90, 90)"],
        ],
        zmid=0, zmin=-3, zmax=3,
        colorbar=dict(title="z", thickness=8, len=0.8),
        hovertemplate="<b>%{y}</b> @ %{x}<br>z = %{z:.2f}σ<extra></extra>",
        text=[[f"{v:+.1f}" if v is not None and np.isfinite(v) else ""
                for v in z_close],
               [f"{v:+.1f}" if v is not None and np.isfinite(v) else ""
                for v in z_resid]],
        texttemplate="%{text}",
        textfont=dict(size=10, color="white"),
    )])
    fig.update_layout(
        height=210, margin=dict(l=60, r=8, t=22, b=22),
        xaxis=dict(side="bottom"), yaxis=dict(autorange="reversed"),
    )
    st.markdown(
        "<div style='font-size:0.7rem; color:var(--text-dim); "
        "text-align:center; margin-bottom:-0.6rem;'>"
        "z-score heatmap · close + residual × 6 lookbacks</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(_no_title(fig), use_container_width=True,
                      key=f"{_SCOPE}_dossier_chart_zheat_{symbol}")


def _render_execution(p: dict) -> None:
    if not p:
        st.markdown("<div style='color:var(--text-dim); font-size:0.78rem;'>"
                     "Execution profile unavailable.</div>", unsafe_allow_html=True)
        return
    rows = []
    if "avg_vol_60d" in p:
        v = p["avg_vol_60d"]
        col = "green" if v > 50000 else "amber" if v > 5000 else "red"
        rows.append(("avg vol (60d)", f"{v:,.0f}", col,
                      f"{p.get('n_days_60','—')} days"))
    if "today_vol" in p:
        rows.append(("today vol", f"{p['today_vol']:,.0f}", None, None))
    if "slippage_bp_by_size" in p:
        for sz, slip in sorted(p["slippage_bp_by_size"].items()):
            col = "green" if slip < 0.1 else "amber" if slip < 0.5 else "red"
            rows.append((f"slip @ {sz}ct", f"~{slip:.2f} bp", col, None))
    _kv_block(rows)


def _render_quick_actions(symbol: str, panel: dict, strategy: str) -> None:
    if st.button("📊 Show all trades w/ this contract",
                       key=f"{_SCOPE}_filter_to_{symbol}",
                       use_container_width=True):
        st.session_state[f"{_SCOPE}_search"] = symbol
        _clear_contract()
        st.rerun()
    asof = panel.get("asof", date.today())
    outright_close = panel.get("outright_close_panel", pd.DataFrame())
    if strategy == "outright" and not outright_close.empty:
        st.markdown(
            "<div style='font-size:0.7rem; color:var(--text-dim); "
            "margin: 0.4rem 0 0.2rem 0;'>compare to neighbors:</div>",
            unsafe_allow_html=True,
        )
        syms_sorted = []
        for s in outright_close.columns:
            t = _outright_tenor_months(s, asof)
            if t is not None:
                syms_sorted.append((s, t))
        syms_sorted.sort(key=lambda x: x[1])
        cur_idx = next((i for i, (s, _) in enumerate(syms_sorted)
                          if s == symbol), None)
        if cur_idx is not None:
            neighbors = []
            for off in (-2, -1, 1, 2):
                j = cur_idx + off
                if 0 <= j < len(syms_sorted):
                    neighbors.append(syms_sorted[j][0])
            if neighbors:
                n_cols = st.columns(len(neighbors))
                for i, n in enumerate(neighbors):
                    with n_cols[i]:
                        if st.button(n, key=f"{_SCOPE}_nbr_{symbol}_{n}",
                                          use_container_width=True):
                            _set_contract(n)
                            st.rerun()


def _render_provenance(p: dict) -> None:
    if not p:
        st.markdown("<div style='color:var(--text-dim); font-size:0.78rem;'>"
                     "Provenance unavailable.</div>", unsafe_allow_html=True)
        return
    rows = []
    if "recon_pct_today" in p:
        v = p["recon_pct_today"] * 100
        col = "green" if v >= 95 else "amber" if v >= 90 else "red"
        rows.append(("3-PC explained today",
                      f"{v:.1f}%", col, None))
    if "recon_err_percentile" in p:
        v = p["recon_err_percentile"]
        col = "red" if v >= 95 else "amber" if v >= 90 else None
        rows.append(("recon-err percentile",
                      f"{v:.0f}%", col,
                      f"{p.get('recon_err_today',0):.2f} bp"))
    if "max_cross_pc_corr_today" in p:
        v = p["max_cross_pc_corr_today"]
        col = "amber" if v > 0.3 else "green"
        rows.append(("max cross-PC corr today",
                      f"{v:.2f}", col, None))
    if "refit_age_days" in p:
        rows.append(("refit age", f"{p['refit_age_days']}d", None,
                      str(p.get("latest_refit_date","—"))))
    if "n_rolling_refits" in p:
        rows.append(("# rolling refits",
                      str(p["n_rolling_refits"]), None, None))
    if "pc1_var_share" in p:
        v = p["pc1_var_share"] * 100
        rows.append(("PC1 var share",
                      f"{v:.1f}%", "blue", None))
    if "pc2_var_share" in p:
        v = p["pc2_var_share"] * 100
        rows.append(("PC2 var share",
                      f"{v:.1f}%", None, None))
    if "pc3_var_share" in p:
        v = p["pc3_var_share"] * 100
        rows.append(("PC3 var share",
                      f"{v:.1f}%", "purple", None))
    if "pca_n_obs" in p:
        rows.append(("PCA fit n_obs",
                      str(p["pca_n_obs"]), None, None))
    if "print_quality_today" in p:
        rows.append(("print quality today",
                      "✓ clean" if p["print_quality_today"] else "⚠ flagged",
                      "green" if p["print_quality_today"] else "red",
                      None))
    if "convexity_warning" in p:
        rows.append(("convexity warning",
                      "⚠ back-end" if p["convexity_warning"] else "—",
                      "amber" if p["convexity_warning"] else None, None))
    if "outlier_today" in p:
        rows.append(("outlier today",
                      "⚠ yes" if p["outlier_today"] else "no",
                      "red" if p["outlier_today"] else "green", None))
    _kv_block(rows)


def _outright_tenor_months(symbol: str, asof: date) -> Optional[float]:
    rp = reference_period(symbol)
    if rp is None:
        return None
    start, end = rp
    mid_days = (start - asof).days + ((end - start).days / 2.0)
    return float(mid_days) / 30.4375


# =============================================================================
# Engine status (collapsed at the bottom)
# =============================================================================
def _render_engine_status(panel: dict, ideas: list) -> None:
    fit = panel.get("pca_fit_static")
    rs = panel.get("regime_stack", {})
    hmm = rs.get("hmm_fit") if rs else None
    rolling = panel.get("rolling_fits", {})

    # Health chip strip
    chips = []
    pct_today = panel.get("reconstruction_pct_today")
    if pct_today is not None:
        col = ("var(--green)" if pct_today >= 0.95
                else "var(--amber)" if pct_today >= 0.9
                else "var(--red)")
        chips.append(_pill(f"3-PC explained today: {pct_today * 100:.0f}%", col))
    cross_corr = panel.get("cross_pc_corr", pd.DataFrame())
    if not cross_corr.empty:
        last_corr = float(cross_corr.abs().iloc[-1].max()) if not cross_corr.iloc[-1].dropna().empty else 0
        col = "var(--green)" if last_corr <= 0.3 else "var(--amber)"
        chips.append(_pill(f"cross-PC corr: {last_corr:.2f}", col))
    if hmm is not None and hmm.dominant_confidence is not None and len(hmm.dominant_confidence) > 0:
        pct = float((hmm.dominant_confidence > 0.6).sum()) / len(hmm.dominant_confidence) * 100
        col = "var(--green)" if pct >= 70 else "var(--amber)"
        chips.append(_pill(f"regime confidence: {pct:.0f}% days", col))
    print_alerts = panel.get("print_quality_alerts", [])
    asof = panel.get("asof", date.today())
    today_print = any(pd.Timestamp(d).date() == asof for d in print_alerts)
    chips.append(_pill("print quality: clean" if not today_print else "⚠ print quality flagged",
                          "var(--green)" if not today_print else "var(--red)"))
    bp_breaks = rs.get("bai_perron_breaks", []) if rs else []
    if bp_breaks:
        last_bp = bp_breaks[-1]
        days_since = (asof - last_bp).days if hasattr(last_bp, "year") else 999
        col = "var(--green)" if days_since > 30 else "var(--amber)"
        chips.append(_pill(f"last regime break: {days_since}d ago", col))
    chips.append(_pill(f"# ideas: {len(ideas)}", "var(--text-body)"))
    chips.append(_pill(f"# clean: {sum(1 for i in ideas if i.gate_quality == 'clean')}",
                          "var(--green)"))

    st.markdown(
        f"<div style='display:flex; gap:0.5rem; flex-wrap:wrap; "
        f"padding:0.4rem 0.6rem; "
        f"font-family:JetBrains Mono, monospace; font-size:0.75rem;'>"
        f"{' '.join(chips)}</div>",
        unsafe_allow_html=True,
    )

    # Source distribution mini-table
    from collections import Counter
    src_counts = Counter(i.primary_source for i in ideas)
    if src_counts:
        _section_header("Active trade-source distribution")
        rows = []
        for src, n in src_counts.most_common(15):
            rows.append((src, str(n), None, None))
        _kv_block(rows)

    # Regime stack info
    gmm = rs.get("gmm_fit") if rs else None
    if gmm is not None:
        _section_header("Regime stack info")
        _kv_block([
            ("GMM K", str(gmm.means.shape[0]), None, None),
            ("GMM n_obs", str(gmm.n_obs), None, None),
            ("GMM log-lik", f"{gmm.log_likelihood:.2f}", None, None),
            ("# rolling refits", str(len(rolling)), None, None),
            ("# Bai-Perron breaks (all-time)",
              str(len(bp_breaks)), None, None),
            ("# outlier days flagged",
              str(len(panel.get('outlier_days', []))), None, None),
            ("# print-quality alerts",
              str(len(print_alerts)),
              "red" if print_alerts else "green", None),
        ])
