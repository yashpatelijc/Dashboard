"""STIRS_DASHBOARD — Streamlit landing page."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from lib.components import page_header, status_strip
from lib.connections import (
    bbg_warehouse_path_info,
    list_bbg_categories,
    list_ohlc_tables,
    ohlc_db_path_info,
)
from lib.backtest.cycle import ensure_backtest_fresh
from lib.css import inject_global_css, render_sidebar_brand
from lib.freshness import freshness_header_chip
from lib.prewarm import ensure_prewarm, ensure_prewarm_all_markets
from lib.turn_adjuster_daemon import ensure_turn_residuals_fresh
from lib.regime_daemon import ensure_regime_fresh
from lib.policy_path_daemon import ensure_policy_path_fresh
from lib.event_impact_daemon import ensure_event_impact_fresh
from lib.signal_emit_daemon import ensure_signal_emit_fresh
from lib.opportunity_daemon import ensure_opportunity_fresh
from lib.knn_daemon import ensure_knn_fresh
from lib.cross_cutting_daemon import ensure_cross_cutting_fresh
from lib.historical_event_impact_daemon import ensure_hei_fresh

st.set_page_config(
    page_title="STIRS / Bonds — Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_global_css()
render_sidebar_brand()

# Kick off the Technicals + PCA scans in background daemon threads for EVERY
# market (SRA, ER, FSR, FER, SON, YBA, CRA). Idempotent across re-runs and
# page navigations. By the time the user navigates to any market sub-tab the
# Technicals scan + PCA panel are already cached.
ensure_prewarm_all_markets()

# Phase E: also kick off the 10-day backtest recompute if cache is stale or
# missing. Mirrors prewarm — non-blocking daemon thread; cold compute ~5-15 min.
ensure_backtest_fresh()

# Phase 3: A10 turn / QE / YE adjuster — rebuilds residuals when the CMC
# parquets are newer than the last residual build. Non-blocking; ~1-2 min cold.
ensure_turn_residuals_fresh()

# Phase 4: A1 regime classifier — rebuilds when residuals refresh.
# Non-blocking; ~5-10s cold for K=6 GMM + HMM smoothing.
ensure_regime_fresh()

# Phase 5: A4 Heitfield-Park policy-path PMFs — rebuilds with CMC.
# Non-blocking; ~3-5s cold for SLSQP on 14-meeting horizon.
ensure_policy_path_fresh()

# Phase 6: A11 event-impact regression. Non-blocking; ~10-30s cold for
# 13 curated tickers × 61 nodes × N_regimes.
ensure_event_impact_fresh()

# Phase 7: signal_emit canonical snapshot — pulls from Phases 4/5/6 + tmia
# detector fires. Non-blocking; ~1-2s cold.
ensure_signal_emit_fresh()

# Phase 9: opportunity modules (A6 OU / A4m TSM / A1c carry / A12d event-drift /
# A9 regime transitions; A2p/A6s/A7c stubbed per plan §15). Non-blocking; ~10-20s cold.
ensure_opportunity_fresh()

# Phase 10: A2 KNN bands + A3 path-conditional FV. Non-blocking; ~5s cold.
# Per §15 A2: SRA-only forever; emissions tagged low_sample.
ensure_knn_fresh()

# Phase 11: cross-cutting [+] additions (cointegration + cross-asset risk
# regime + Bauer-Swanson). Non-blocking; ~10s cold.
ensure_cross_cutting_fresh()

# Historical Event Impact: 1H-resolution intraday pre/post analysis around 13
# major US releases (T-1/T/T+1 windows × 46 instruments × 8 states × 4 windows).
# Non-blocking; ~3-8 min cold compute (one-time per CMC refresh).
ensure_hei_fresh()

# ---------- Header ----------
ohlc_info = ohlc_db_path_info()
meta_str = ""
if ohlc_info["path"]:
    meta_str = f"OHLC snapshot · {ohlc_info['modified']:%Y-%m-%d}"
page_header(
    title="STIRS / Bonds Analytics",
    subtitle="Multi-economy short-rate and bond analytics. Use the sidebar to navigate.",
    meta=meta_str,
)

# Phase 0.C: data-freshness traffic light below header
# Phase 4: regime chip alongside.
def _regime_chip_html() -> str:
    try:
        from pathlib import Path as _P
        from datetime import date as _D
        cache = _P(__file__).resolve().parent / ".cmc_cache"
        cands = sorted(cache.glob("regime_manifest_*.json"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands:
            return ""
        from lib.sra_data import get_current_regime
        asof = _D.fromisoformat(cands[0].stem.replace("regime_manifest_", ""))
        cur = get_current_regime(asof)
        if not cur:
            return ""
        post = cur["top_state_posterior"]
        return (
            f"<span style='display:inline-flex; align-items:center; gap:0.35rem; "
            f"padding:2px 8px; border:1px solid #e8b75d; border-radius:999px; "
            f"font-family: JetBrains Mono, monospace; font-size:0.65rem; "
            f"color:#e8b75d; background:rgba(232,183,93,0.08); margin-left:0.5rem;'>"
            f"<span>REGIME · {cur['state_name']} · "
            f"stab={post:.2f}</span></span>"
        )
    except Exception:
        return ""


def _policy_chip_html() -> str:
    try:
        from pathlib import Path as _P
        from datetime import date as _D
        cache = _P(__file__).resolve().parent / ".cmc_cache"
        cands = sorted(cache.glob("policy_path_manifest_*.json"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands:
            return ""
        from lib.sra_data import get_policy_path_summary
        asof = _D.fromisoformat(cands[0].stem.replace("policy_path_manifest_", ""))
        s = get_policy_path_summary(asof)
        if not s or s.get("terminal_rate_bp") is None:
            return ""
        terminal = s["terminal_rate_bp"] / 100.0
        cycle = s["cycle_label"]
        color = {"LATE_HIKE": "#ef4444", "LATE_CUT": "#22d3ee",
                    "NEUTRAL": "#94a3b8"}.get(cycle, "#5e6975")
        return (
            f"<span style='display:inline-flex; align-items:center; gap:0.35rem; "
            f"padding:2px 8px; border:1px solid {color}; border-radius:999px; "
            f"font-family: JetBrains Mono, monospace; font-size:0.65rem; "
            f"color:{color}; background:rgba(0,0,0,0.2); margin-left:0.5rem;'>"
            f"<span>TERMINAL · {terminal:.3f}% · {cycle}</span></span>"
        )
    except Exception:
        return ""


def _risk_regime_chip_html() -> str:
    """Phase 12 cross-asset risk-regime header chip (BULL_FLATTENER /
    BEAR_STEEPENER / FLIGHT_TO_QUALITY / RISK_ON / GOLDILOCKS / VOL_BACKTEST)."""
    try:
        from pathlib import Path as _P
        cache = _P(__file__).resolve().parent / ".cmc_cache"
        cands = sorted(cache.glob("cross_cutting_manifest_*.json"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands:
            return ""
        import json as _json
        m = _json.loads(cands[0].read_text())
        label = m.get("risk_regime_label", "")
        if not label or label == "UNKNOWN":
            return ""
        color = {
            "RISK_ON": "#22c55e", "GOLDILOCKS": "#84cc16",
            "FLIGHT_TO_QUALITY": "#fbbf24", "VOL_BACKTEST": "#ef4444",
            "BULL_FLATTENER": "#3b82f6", "BEAR_STEEPENER": "#f97316",
        }.get(label, "#94a3b8")
        return (
            f"<span style='display:inline-flex; align-items:center; gap:0.35rem; "
            f"padding:2px 8px; border:1px solid {color}; border-radius:999px; "
            f"font-family: JetBrains Mono, monospace; font-size:0.65rem; "
            f"color:{color}; background:rgba(0,0,0,0.2); margin-left:0.5rem;'>"
            f"<span>RISK · {label}</span></span>"
        )
    except Exception:
        return ""

st.markdown(
    f"<div style='margin: -0.4rem 0 1rem 0;'>{freshness_header_chip()}"
    f"{_regime_chip_html()}{_policy_chip_html()}{_risk_regime_chip_html()} "
    f"<span style='color:var(--text-dim); font-size:0.65rem; margin-left:0.5rem;'>"
    f"see Settings → System Health for full report</span></div>",
    unsafe_allow_html=True,
)

# ---------- Status strip ----------
cats = list_bbg_categories()
n_files = int(cats["files"].sum()) if not cats.empty else 0
size_mb = round(cats["size_mb"].sum(), 1) if not cats.empty else 0

try:
    n_tables = len(list_ohlc_tables()) if ohlc_info["path"] else 0
except Exception:
    n_tables = 0

status_strip([
    ("OHLC", Path(ohlc_info["path"]).name if ohlc_info["path"] else "—", "accent"),
    ("Size", f"{ohlc_info['size_gb']} GB" if ohlc_info["path"] else "—", None),
    ("Tables", str(n_tables), None),
    ("BBG categories", str(len(cats)), None),
    ("BBG files", f"{n_files:,}", None),
    ("BBG size", f"{size_mb} MB", None),
])

st.markdown("")

# ---------- Pages overview ----------
st.markdown("### Pages")

col_l, col_r = st.columns(2)

with col_l:
    st.markdown(
        """
        <div class="section-card">
            <div class="head">
                <div class="title">🇺🇸 US Economy</div>
                <div class="meta">SRA · FF+S1R · Bonds · Fundamentals</div>
            </div>
            <div style="color:var(--text-body); font-size:0.85rem; line-height:1.6;">
                Curve charts (outrights / spreads / flies), term structure,
                positioning, fundamentals overlay.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_r:
    st.markdown(
        """
        <div class="section-card">
            <div class="head">
                <div class="title">⚙️ Settings</div>
                <div class="meta">General · OHLC Viewer · BBG Viewer</div>
            </div>
            <div style="color:var(--text-body); font-size:0.85rem; line-height:1.6;">
                Detailed data viewers + custom SQL playgrounds for both data sources.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    "<div style='color:var(--text-dim); font-size:0.75rem; margin-top:1.5rem; "
    "font-family: JetBrains Mono, monospace;'>"
    "More economies (EU · UK · JP · CH · CA · AU · NZ) added once the US page is feature-complete."
    "</div>",
    unsafe_allow_html=True,
)
