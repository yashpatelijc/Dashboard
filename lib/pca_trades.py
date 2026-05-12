"""Master trade-idea engine for PCA-driven SRA opportunities.

This module sits on top of lib/pca.py + lib/pca_regimes.py + lib/pca_analogs.py +
lib/pca_events.py and produces a unified ranked feed of TradeIdea objects.

Every TradeIdea carries:
  * A complete trade structure (legs, weights, direction)
  * Economics (entry, fair value, target, stop, expected P&L)
  * Statistics (z-score, half-life, ADF, eff_n)
  * Cross-confirmation strip (which sources agree on these legs)
  * Lifecycle state (NEW/MATURING/PEAK/FADING/REVERTED/FAILED/TIMED_OUT)
  * Conviction breakdown (17-input blend)
  * Risk overlay (factor exposure, margin, worst-case 1d loss)
  * Execution gates (slippage, convexity, event-in-window, print-quality)
  * Context (regime, cycle phase, seasonality, event drift expectation)

The 38 trade-opportunity sources are organized in tiers; each `gen_*_ideas`
function emits raw TradeIdea objects which are then clustered by leg
fingerprint, scored, lifecycle-tagged, and returned by the master orchestrator.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field, asdict, replace
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from lib.pca import (
    PCAFit, SparsePCAFit, StructureCandidate,
    DEFAULT_TENOR_GRID_MONTHS, EFF_N_FLOOR, LIBOR_SOFR_CUTOVER,
    pc1_loading_asymmetry, eigenspectrum_gap, variance_ratio_regime,
    sparse_dense_divergence, cross_pc_corr_breakdown_signal,
    SEGMENT_BOUNDS,
)
from lib.contract_units import bp_multipliers_for, load_catalog


# =============================================================================
# Constants
# =============================================================================
LEDGER_PATH = "data/pca_trade_ledger.parquet"
SRA_TICK_VALUE_USD = 25.0          # $25/bp per contract for SRA (CME spec)
DEFAULT_PV01_PER_LEG = 25.0        # SRA outright PV01 in $/bp


def _panel_mode_params(panel: dict) -> dict:
    """Resolve mode params for a generator, given the panel.

    Generators read mode from panel["mode"] (populated by build_full_pca_panel).
    Falls back to DEFAULT_MODE if not present.
    """
    from lib.pca import mode_params
    return mode_params(panel.get("mode"))


def _bounded_hold_days(hl: Optional[float], panel: dict) -> Optional[float]:
    """Apply mode-aware floor/cap to the 1.5×HL hold-window heuristic.

    Returns None if hl is None. Otherwise:
      hold = clip(hold_mult × hl, hold_floor, hold_cap)
    """
    if hl is None or not np.isfinite(hl) or hl <= 0:
        return None
    mp = _panel_mode_params(panel)
    raw = float(mp["hold_mult"]) * float(hl)
    return float(max(mp["hold_floor"], min(raw, mp["hold_cap"])))
LIFECYCLE_STATES = ("NEW", "MATURING", "PEAK", "FADING",
                     "REVERTED", "FAILED", "TIMED_OUT")
SOURCE_NAMES = {
    1: "PC3-fly", 2: "PC2-spread", 3: "PC1-basket",
    4: "anchor", 5: "cross-PC-corr-breakdown", 6: "sparse-dense-divergence",
    7: "variance-regime", 8: "eigenspectrum-gap", 9: "PC1-asymmetry",
    10: "front-PCA", 11: "belly-PCA", 12: "back-PCA",
    13: "outright-fade", 14: "spread-fade", 15: "fly-arb", 16: "pack",
    17: "analog-FV", 18: "path-FV",
    19: "GMM-HMM-regime", 20: "posterior-degradation", 21: "BOCPD", 22: "Bai-Perron",
    23: "seasonality", 24: "event-drift", 25: "days-since-FOMC",
    26: "pack-fly", 27: "bundle-RV",
    28: "PC-momentum", 29: "cycle-phase", 30: "outlier-reversal", 31: "carry-fly",
    32: "pnl-attribution", 33: "position-factor-exposure",
    34: "slippage-gate", 35: "convexity-flag", 36: "print-quality-filter",
    37: "hard-calendar-block",
    38: "event-impact-ranking",
}


# =============================================================================
# Types
# =============================================================================
@dataclass(frozen=True)
class TradeLeg:
    symbol: str
    weight_pv01: float            # PV01-normalized weight
    side: str                     # "buy" | "sell"
    contracts: int = 1


@dataclass(frozen=True)
class TradeIdea:
    # Identity
    idea_id: str
    leg_fingerprint: str
    primary_source: str            # e.g. "PC3-fly"
    sources: tuple = ()             # all sources that confirm this idea
    source_id: int = 1             # numeric source ID

    # Structure
    direction: str = "long"        # "long" | "short" | "flatten" | "steepen" | "neutral"
    structure_type: str = "fly"    # "outright" | "spread" | "fly" | "pack" | "basket"
    legs: tuple = ()                # tuple of TradeLeg
    pv01_sum: float = 0.0

    # Economics (in bp)
    entry_bp: Optional[float] = None
    fair_value_bp: Optional[float] = None
    target_bp: Optional[float] = None
    stop_bp: Optional[float] = None
    expected_revert_d: Optional[float] = None
    expected_pnl_bp: Optional[float] = None
    expected_pnl_dollar: Optional[float] = None

    # Statistics
    z_score: Optional[float] = None
    half_life_d: Optional[float] = None
    adf_pass: bool = False
    triple_gate_pass: bool = False
    eff_n: int = 0

    # Cross-confirmation (filled by clustering engine)
    n_confirming_sources: int = 1
    confirming_pills: tuple = ()

    # Conditional FV cross-checks (analog A2 / path A3)
    analog_fv_bp: Optional[float] = None
    analog_fv_z: Optional[float] = None
    analog_fv_eff_n: Optional[float] = None
    path_fv_bp: Optional[float] = None
    path_fv_z: Optional[float] = None

    # Conviction
    conviction: float = 0.0
    conviction_breakdown: tuple = ()    # tuple of (component_name, value) pairs

    # Lifecycle
    state: str = "NEW"
    days_alive: int = 0
    max_conviction_to_date: float = 0.0
    entry_residual_at_birth: Optional[float] = None

    # Context
    regime_label: Optional[str] = None
    cycle_phase: Optional[str] = None
    cycle_alignment: str = "neutral"     # "favoured" | "neutral" | "counter"
    seasonality_tag: Optional[str] = None
    event_proximity_d: Optional[int] = None

    # Risk (filled by risk overlay)
    factor_exposure: tuple = ()           # tuple of (PC_name, σ-loading)
    idiosyncratic_var: Optional[float] = None
    margin_estimate_dollar: Optional[float] = None
    worst_case_1d_loss_dollar: Optional[float] = None

    # What-if (filled later)
    what_if_table: tuple = ()             # tuple of (scenario, $pnl)

    # Execution
    slippage_estimate_bp: Optional[float] = None
    print_quality_today: bool = True
    convexity_warning: bool = False
    risk_flags: tuple = ()
    gate_quality: str = "clean"           # "clean" | "low_n" | "non_stationary"
                                            # "regime_unstable" | "event_in_window"

    # Track record
    source_hit_rate_30: Optional[float] = None
    source_hit_rate_30_in_regime: Optional[float] = None

    rationale_html: str = ""

    # 1-line caption for the trade-table row (NEW). Distinct from rationale_html
    # which is the 4-part full narrative shown on click.
    # Example: "SRAM27 fade · +1.8σ rich · OU 14d · 21d mom supports · clean"
    headline_html: str = ""


# =============================================================================
# Rationale builders — 4-part narrative chain per generator family
# =============================================================================
# Every TradeIdea gets a populated rationale via these builders. The 4 parts:
#   1. WHAT THE ENGINE SAW   — the analysis output (z-score, FV deviation, etc.)
#   2. WHY THIS MATTERS       — economic interpretation
#   3. THE TRADE              — entry/target/stop/PV01 + structure
#   4. HOW IT RESOLVES        — expected window, events, slippage, caveats
#
# Each returns HTML — usable directly in the drilldown panel.
# =============================================================================

def _hl_zone_label(hl_d: Optional[float]) -> str:
    """Plain-English OU half-life zone."""
    if hl_d is None or not np.isfinite(hl_d):
        return "no half-life (random walk)"
    if hl_d < 3:
        return f"too fast ({hl_d:.1f}d, hard to size)"
    if hl_d <= 30:
        return f"sweet spot ({hl_d:.1f}d)"
    if hl_d <= 60:
        return f"slow ({hl_d:.1f}d, watch overnight risk)"
    return f"very slow ({hl_d:.1f}d, marginal edge)"


def _gate_label(gate_quality: str) -> str:
    return {
        "clean": "✓ clean (ADF+KPSS+VR pass)",
        "low_n": "⚠ low effective sample (eff_n < 30)",
        "non_stationary": "⚠ ADF fails (random-walk risk)",
        "drift": "⚠ KPSS rejects (drift component)",
        "random_walk": "✗ no mean-reversion edge (VR ≈ 1)",
        "regime_unstable": "⚠ regime in transition (BOCPD/Bai-Perron firing)",
        "stale": "⚠ stale (fired 3+ times in 14d)",
        "event_in_window": "⚠ event in hold window",
    }.get(gate_quality, gate_quality)


def _direction_word(direction: str, structure_type: str) -> str:
    if structure_type == "fly":
        return "SHORT the fly (fade)" if direction == "short" else "LONG the fly"
    if structure_type == "spread":
        return "SHORT the spread (flatten)" if direction == "short" else "LONG the spread (steepen)"
    if structure_type == "outright":
        return "SHORT outright" if direction == "short" else "LONG outright"
    if structure_type == "basket":
        return "SHORT basket" if direction == "short" else "LONG basket"
    return direction.upper()


def _fmt_legs_inline(legs: tuple, max_n: int = 4) -> str:
    if not legs:
        return "(meta-signal, no legs)"
    pieces = []
    for leg in legs[:max_n]:
        glyph = "+" if leg.weight_pv01 > 0 else "−" if leg.weight_pv01 < 0 else "·"
        pieces.append(f"{glyph}{leg.symbol}")
    if len(legs) > max_n:
        pieces.append(f"+{len(legs) - max_n} more")
    return " ".join(pieces)


def _events_in_window_str(panel: dict, expected_revert_d: Optional[float]) -> str:
    """Return a phrase describing FOMC/CPI/NFP events inside the hold window."""
    asof = panel.get("asof")
    if asof is None or not expected_revert_d or not np.isfinite(expected_revert_d):
        return "no event window data"
    fomc = panel.get("fomc_calendar_dates", []) or []
    horizon_d = max(1, int(round(expected_revert_d * 1.5)))
    upcoming = [pd.Timestamp(d).date() for d in fomc
                  if pd.Timestamp(d).date() > asof
                  and (pd.Timestamp(d).date() - asof).days <= horizon_d]
    if not upcoming:
        return f"no FOMC in next {horizon_d}d"
    days = (upcoming[0] - asof).days
    return f"FOMC in {days}d (within {horizon_d}d window)"


def _build_rationale(what: str, why: str, trade: str, resolves: str) -> str:
    """Render the 4-part rationale as HTML."""
    return (
        "<div style='line-height:1.65;'>"
        f"<div style='margin-bottom:0.25rem;'>"
        f"<span style='color:var(--text-dim); font-size:0.7rem; text-transform:uppercase; "
        f"letter-spacing:0.05em; margin-right:0.5rem; min-width:11rem; display:inline-block;'>"
        f"WHAT THE ENGINE SAW</span>"
        f"<span style='color:var(--text-body);'>{what}</span></div>"
        f"<div style='margin-bottom:0.25rem;'>"
        f"<span style='color:var(--text-dim); font-size:0.7rem; text-transform:uppercase; "
        f"letter-spacing:0.05em; margin-right:0.5rem; min-width:11rem; display:inline-block;'>"
        f"WHY THIS MATTERS</span>"
        f"<span style='color:var(--text-body);'>{why}</span></div>"
        f"<div style='margin-bottom:0.25rem;'>"
        f"<span style='color:var(--accent); font-size:0.7rem; text-transform:uppercase; "
        f"letter-spacing:0.05em; margin-right:0.5rem; min-width:11rem; display:inline-block;'>"
        f"THE TRADE</span>"
        f"<span style='color:var(--text-heading); font-weight:500;'>{trade}</span></div>"
        f"<div>"
        f"<span style='color:var(--text-dim); font-size:0.7rem; text-transform:uppercase; "
        f"letter-spacing:0.05em; margin-right:0.5rem; min-width:11rem; display:inline-block;'>"
        f"HOW IT RESOLVES</span>"
        f"<span style='color:var(--text-body);'>{resolves}</span></div>"
        "</div>"
    )


def _rationale_pc_isolated(c, source_id: int, panel: dict,
                              direction: str, pnl_bp: Optional[float],
                              pnl_dollar: Optional[float], legs: tuple,
                              revert_d: Optional[float]) -> tuple:
    """Rationale for PC1 / PC2 / PC3 isolated structures (Tier 1).

    Returns (headline, full_rationale_html).
    """
    pc_name = {1: "level (PC1)", 2: "slope (PC2)", 3: "curvature (PC3)"}.get(c.target_pc, f"PC{c.target_pc}")
    structure_word = {1: "PC1-basket", 2: "PC2-spread", 3: "PC3-fly"}.get(c.target_pc, f"PC{c.target_pc}")
    legs_str = _fmt_legs_inline(legs)
    z_str = f"{c.residual_z:+.2f}σ" if c.residual_z is not None else "—"
    res_str = f"{c.residual_today_bp:+.2f} bp" if c.residual_today_bp is not None else "—"
    hl_str = _hl_zone_label(c.half_life_d)
    gate_str = _gate_label(c.gate_quality)
    eff_n = c.eff_n or 0
    side_word = _direction_word(direction, "fly" if c.target_pc == 3 else "spread" if c.target_pc == 2 else "basket")
    events_str = _events_in_window_str(panel, revert_d)

    fade_word = {3: "Curvature dislocation", 2: "Slope dislocation", 1: "Level dislocation"}.get(c.target_pc, "Dislocation")
    why_word = {3: "1Y belly curvature", 2: "front-vs-back slope", 1: "outright level"}.get(c.target_pc, "factor")

    what = (f"<b>{structure_word}</b> residual is <b>{z_str}</b> "
             f"({res_str}) vs analog window ; eff_n={eff_n}, gate {gate_str}")
    why = (f"{fade_word} in {why_word} — the {pc_name}-isolated basket "
            f"is priced richer than analog windows with similar 3-PC state.")
    pnl_str = f"~${pnl_dollar:+,.0f}/contract" if pnl_dollar else "~$/contract n/a"
    trade = (f"{side_word}: {legs_str} · entry {res_str} · target 0bp · "
              f"stop ±2.5σ · {pnl_str}")
    resolves = (f"OU half-life {hl_str}; {events_str}")

    headline = (f"<b>{structure_word}</b> {direction} · "
                 f"<b>{z_str}</b> · OU {hl_str.split(' ')[0] if hl_str else '—'} · {gate_str}")
    return headline, _build_rationale(what, why, trade, resolves)


def _rationale_residual_fade(row, source_id: int, panel: dict,
                                structure_kind: str, direction: str,
                                pnl_bp: Optional[float], pnl_dollar: Optional[float],
                                legs: tuple, revert_d: Optional[float]) -> tuple:
    """Rationale for outright/spread/fly/pack residual fades."""
    instrument = row.get("instrument", "?")
    z = row.get("residual_z")
    res_today = row.get("residual_today_bp")
    hl = row.get("half_life")
    gate = row.get("gate_quality", "clean")
    eff_n = row.get("eff_n", 0)

    z_str = f"{z:+.2f}σ" if z is not None and np.isfinite(z) else "—"
    res_str = f"{res_today:+.2f} bp" if res_today is not None and np.isfinite(res_today) else "—"
    hl_str = _hl_zone_label(hl)
    gate_str = _gate_label(gate)
    side_word = _direction_word(direction, structure_kind)
    events_str = _events_in_window_str(panel, revert_d)

    kind_word = {
        "outright": "outright", "spread": "calendar spread",
        "fly": "butterfly", "pack": "pack",
    }.get(structure_kind, structure_kind)

    what = (f"{instrument} {kind_word} residual is <b>{z_str}</b> "
             f"({res_str}) vs 60d distribution ; eff_n={eff_n}, gate {gate_str}")
    why = (f"Today's {kind_word} change-space residual exceeds the 3-PC factor model — "
            f"unexplained idiosyncratic move that historically reverts.")
    pnl_str = f"~${pnl_dollar:+,.0f}/contract" if pnl_dollar else "~$/contract n/a"
    trade = (f"{side_word}: {_fmt_legs_inline(legs)} · entry {res_str} · "
              f"target 0bp · stop ±2.5σ · {pnl_str}")
    resolves = f"OU half-life {hl_str}; {events_str}"

    headline = (f"<b>{instrument} {kind_word} fade</b> · {z_str} · "
                 f"OU {hl_str.split(' ')[0] if hl_str else '—'} · {gate_str}")
    return headline, _build_rationale(what, why, trade, resolves)


def _rationale_analog_fv(row, source_id: int, panel: dict,
                           direction: str, fv_bp: Optional[float],
                           legs: tuple, revert_d: Optional[float]) -> tuple:
    """Rationale for A2 analog-FV-divergence ideas."""
    instrument = row.get("instrument", "?")
    z = row.get("residual_z")
    res = row.get("residual_today_bp")
    eff_n = row.get("eff_n", 0)
    gate = row.get("gate_quality", "clean")

    z_str = f"{z:+.2f}σ" if z is not None and np.isfinite(z) else "—"
    fv_str = f"{fv_bp:+.2f} bp" if fv_bp is not None and np.isfinite(fv_bp) else "—"
    res_str = f"{res:+.2f} bp" if res is not None and np.isfinite(res) else "—"
    gate_str = _gate_label(gate)
    side_word = _direction_word(direction, "outright")
    events_str = _events_in_window_str(panel, revert_d)

    what = (f"<b>A2 analog-FV</b>: {instrument} today {res_str} "
             f"vs analog FV {fv_str} ({z_str} divergence). eff_n={eff_n}, gate {gate_str}")
    why = ("KNN-Mahalanobis matched against historical PC-state analogs "
            "(Ledoit-Wolf shrunk cov + ±60d temporal exclusion + 250d half-life decay).")
    trade = (f"{side_word} {instrument} · entry {res_str} · target {fv_str} · "
              f"stop ±2.5σ from FV")
    resolves = f"Analog FV reverts within trade-horizon; {events_str}"

    headline = f"<b>{instrument} analog-FV {direction}</b> · {z_str} · {gate_str}"
    return headline, _build_rationale(what, why, trade, resolves)


def _rationale_path_fv(row, source_id: int, panel: dict,
                         direction: str, fv_bp: Optional[float],
                         path_probs: dict, legs: tuple,
                         revert_d: Optional[float]) -> tuple:
    """Rationale for A3+A4 path-conditional FV ideas."""
    instrument = row.get("instrument", "?")
    z = row.get("residual_z")
    res = row.get("residual_today_bp")

    z_str = f"{z:+.2f}σ" if z is not None and np.isfinite(z) else "—"
    fv_str = f"{fv_bp:+.2f} bp" if fv_bp is not None and np.isfinite(fv_bp) else "—"
    res_str = f"{res:+.2f} bp" if res is not None and np.isfinite(res) else "—"

    if path_probs:
        cut = (path_probs.get("large_cut", 0) + path_probs.get("cut", 0)) * 100
        hold = path_probs.get("hold", 0) * 100
        hike = (path_probs.get("hike", 0) + path_probs.get("large_hike", 0)) * 100
        path_phrase = f"A4 step-path: <b>{cut:.0f}% cut · {hold:.0f}% hold · {hike:.0f}% hike</b>"
    else:
        path_phrase = "A4 step-path unavailable"

    side_word = _direction_word(direction, "outright")
    events_str = _events_in_window_str(panel, revert_d)

    what = (f"<b>A3 path-FV</b>: {instrument} today {res_str} vs path-conditional FV "
             f"{fv_str} ({z_str}). {path_phrase}")
    why = ("Heitfield-Park bootstrap on FOMC step probabilities + bucket-conditional "
            "analog FV — reflects today's policy-path expectations, not just curve state.")
    trade = (f"{side_word} {instrument} · entry {res_str} · target {fv_str}")
    resolves = f"Mean-revert as policy path resolves; {events_str}"

    headline = f"<b>{instrument} path-FV {direction}</b> · {z_str}"
    return headline, _build_rationale(what, why, trade, resolves)


def _rationale_carry_fly(idea_label: str, panel: dict, direction: str,
                            carry_plus_roll_bp: Optional[float], legs: tuple,
                            extra_note: str = "") -> tuple:
    """Rationale for carry-fly + fly-arb ideas."""
    cr_str = (f"{carry_plus_roll_bp:+.2f} bp/3M"
                if carry_plus_roll_bp is not None and np.isfinite(carry_plus_roll_bp)
                else "—")
    side_word = _direction_word(direction, "fly")
    what = f"<b>{idea_label}</b> — carry+roll over 3M hold = {cr_str} · {extra_note}"
    why = ("Time-decay return from holding the fly while curve shape stays constant. "
            "Positive carry+roll trades pay even without mean-reversion.")
    trade = f"{side_word}: {_fmt_legs_inline(legs)} · enter at market"
    resolves = "Time-pass-through; revisit if curve shape changes"
    headline = f"<b>{idea_label}</b> · {cr_str} · carry-positive"
    return headline, _build_rationale(what, why, trade, resolves)


def _rationale_event_anticipation(idea_label: str, panel: dict, direction: str,
                                       event_class: str, expected_drift_bp: Optional[float],
                                       legs: tuple) -> tuple:
    """Rationale for event-driven trades (FOMC, CPI, NFP, auction anticipation)."""
    drift_str = (f"{expected_drift_bp:+.2f} bp" if expected_drift_bp is not None
                  and np.isfinite(expected_drift_bp) else "—")
    side_word = _direction_word(direction, "outright")
    what = f"<b>{event_class} event drift</b>: A12d expects {drift_str} post-event in this segment"
    why = (f"Historical {event_class} events have produced systematic post-event drift "
            f"in this curve segment (Newey-West HAC s.e., BH-FDR retained).")
    trade = f"{side_word}: {_fmt_legs_inline(legs)} · pre-position before event"
    resolves = f"Drift resolves over T+5d to T+20d window post-{event_class}"
    headline = f"<b>{idea_label}</b> · {event_class} drift {drift_str}"
    return headline, _build_rationale(what, why, trade, resolves)


def _rationale_pack_rv(idea_label: str, panel: dict, direction: str,
                          z_score: Optional[float], legs: tuple,
                          extra_note: str = "") -> tuple:
    """Rationale for pack/bundle/pack-fly RV ideas."""
    z_str = f"{z_score:+.2f}σ" if z_score is not None and np.isfinite(z_score) else "—"
    side_word = _direction_word(direction, "basket")
    what = f"<b>{idea_label}</b> z = {z_str} · {extra_note}"
    why = ("Pack/bundle synthetic captures a basket of consecutive quarterlies — "
            "smooths individual contract noise, exposes regime-level dislocation.")
    trade = f"{side_word}: {_fmt_legs_inline(legs)}"
    resolves = "Pack RV reverts as quarterly noise averages out"
    headline = f"<b>{idea_label}</b> · {z_str}"
    return headline, _build_rationale(what, why, trade, resolves)


def _rationale_meta(source_label: str, payload: str) -> tuple:
    """Rationale for engine-state signals (meta-signals, no executable legs)."""
    what = f"<b>{source_label}</b> — {payload}"
    why = ("Engine state signal — affects conviction of trades downstream. "
            "Not directly tradeable on its own.")
    trade = "(no executable legs — informational overlay)"
    resolves = "Resolves when engine state normalizes"
    headline = f"<b>{source_label}</b> · {payload[:50]}"
    return headline, _build_rationale(what, why, trade, resolves)


# =============================================================================
# Helpers
# =============================================================================
def _stable_idea_id(source: str, leg_fingerprint: str) -> str:
    h = hashlib.md5(f"{source}|{leg_fingerprint}".encode("utf-8")).hexdigest()
    return f"id_{h[:12]}"


def _fingerprint_legs(legs: tuple, direction: str) -> str:
    """Stable fingerprint = sorted(symbol|weight) + direction."""
    if not legs:
        return f"empty|{direction}"
    parts = sorted(f"{leg.symbol}@{round(leg.weight_pv01, 4)}" for leg in legs)
    return "|".join(parts) + f"|{direction}"


def _make_legs_from_arrays(symbols: list, weights: np.ndarray) -> tuple:
    """Convert (symbols, weights) into TradeLeg tuple. Side derived from sign."""
    out = []
    for sym, w in zip(symbols, weights):
        side = "buy" if w >= 0 else "sell"
        out.append(TradeLeg(symbol=sym, weight_pv01=float(w),
                              side=side, contracts=int(np.sign(w)) if w != 0 else 0))
    return tuple(out)


def _direction_from_z(z: Optional[float]) -> str:
    """Map z-score to trade direction (sign rule)."""
    if z is None or pd.isna(z):
        return "neutral"
    return "short" if z > 0 else "long"


def _compute_pnl(entry_bp: Optional[float], fv_bp: Optional[float],
                  pv01_dollar_per_leg: float, n_legs: int) -> tuple:
    """Expected P&L from full mean revert: (bp, dollar)."""
    if entry_bp is None or fv_bp is None or not np.isfinite(entry_bp) or not np.isfinite(fv_bp):
        return None, None
    pnl_bp = abs(entry_bp - fv_bp)
    pnl_dollar = pnl_bp * pv01_dollar_per_leg * max(1, n_legs)
    return float(pnl_bp), float(pnl_dollar)


# =============================================================================
# Tier 1 — PCA-isolated structures (PC3-fly, PC2-spread, PC1-basket)
# =============================================================================
def _idea_from_structure(c: StructureCandidate, source_id: int,
                           panel: dict) -> Optional[TradeIdea]:
    """Convert a StructureCandidate from lib.pca into a TradeIdea — with full
    4-part rationale + headline populated."""
    if c is None or not c.symbols:
        return None
    if c.target_pc not in (1, 2, 3):
        return None
    legs = _make_legs_from_arrays(c.symbols, c.weights)
    direction = _direction_from_z(c.residual_z)
    fp = _fingerprint_legs(legs, direction)
    source = SOURCE_NAMES.get(source_id, f"PC{c.target_pc}")
    iid = _stable_idea_id(source, fp)
    fv_bp = 0.0    # PC-isolated structures mean-revert to 0
    target_bp = 0.0
    stop_bp = (c.residual_today_bp or 0.0) + (
        2.5 * abs(c.residual_today_bp or 0.0) if c.residual_today_bp else 5.0
    ) * (1 if direction == "long" else -1)
    pnl_bp, pnl_dollar = _compute_pnl(c.residual_today_bp, fv_bp,
                                          DEFAULT_PV01_PER_LEG, len(legs))
    revert_d = _bounded_hold_days(c.half_life_d, panel)
    risk_flags = []
    if c.gate_quality == "low_n":
        risk_flags.append("low_n")
    if c.gate_quality == "non_stationary":
        risk_flags.append("non_stationary")
    if c.gate_quality == "regime_unstable":
        risk_flags.append("regime_unstable")
    structure_type = {3: "fly", 2: "spread", 1: "basket"}.get(c.target_pc, "basket")
    # Build 4-part rationale + 1-line headline
    headline_html, rationale_html = _rationale_pc_isolated(
        c, source_id, panel, direction, pnl_bp, pnl_dollar, legs, revert_d,
    )
    return TradeIdea(
        idea_id=iid,
        leg_fingerprint=fp,
        primary_source=source,
        sources=(source,),
        source_id=source_id,
        direction=direction,
        structure_type=structure_type,
        legs=legs,
        pv01_sum=float(c.pv01_sum),
        entry_bp=c.residual_today_bp,
        fair_value_bp=fv_bp,
        target_bp=target_bp,
        stop_bp=stop_bp,
        expected_revert_d=revert_d,
        expected_pnl_bp=pnl_bp,
        expected_pnl_dollar=pnl_dollar,
        z_score=c.residual_z,
        half_life_d=c.half_life_d,
        adf_pass=c.adf_pass,
        triple_gate_pass=c.adf_pass,    # StructureCandidate's adf_pass IS the triple-gate verdict
        eff_n=c.eff_n,
        gate_quality=c.gate_quality,
        risk_flags=tuple(risk_flags),
        rationale_html=rationale_html,
        headline_html=headline_html,
    )


def gen_pc3_fly_ideas(panel: dict) -> list:
    """Tier-1 source #1 — PC3 curvature flies."""
    cands = [c for c in panel.get("structure_candidates", []) if c.target_pc == 3]
    return [i for i in (_idea_from_structure(c, 1, panel) for c in cands) if i is not None]


def gen_pc2_spread_ideas(panel: dict) -> list:
    """Tier-1 source #2 — PC2 slope spreads."""
    cands = [c for c in panel.get("structure_candidates", []) if c.target_pc == 2]
    return [i for i in (_idea_from_structure(c, 2, panel) for c in cands) if i is not None]


def gen_pc1_basket_ideas(panel: dict) -> list:
    """Tier-1 source #3 — PC1 directional baskets."""
    cands = [c for c in panel.get("structure_candidates", []) if c.target_pc == 1]
    return [i for i in (_idea_from_structure(c, 3, panel) for c in cands) if i is not None]


# =============================================================================
# Tier 2 — Cross-factor structure trades
# =============================================================================
def gen_anchor_ideas(panel: dict) -> list:
    """Source #4 — Anchor (12M − 24M) slope mean-reversion."""
    out = []
    anchor = panel.get("anchor_series")
    diag = panel.get("pc_diagnostics", {}).get("Anchor", {})
    if anchor is None or anchor.empty or not diag:
        return out
    mp = _panel_mode_params(panel)
    z = diag.get("z")
    if z is None or abs(z) < float(mp["z_threshold"]):
        return out
    today_v = float(anchor.iloc[-1])
    fv_v = float(anchor.tail(252).mean()) if len(anchor) >= 30 else float(anchor.mean())
    direction = "short" if z > 0 else "long"
    fp = f"ANCHOR_24M-12M|{direction}"
    iid = _stable_idea_id("anchor", fp)
    pnl_bp = abs(today_v - fv_v)
    hl_anchor = diag.get("ou_half_life")
    revert_d = _bounded_hold_days(hl_anchor, panel) or float(mp["hold_floor"])
    legs = (TradeLeg("CMC_24M", 1.0, "buy" if direction == "long" else "sell"),
             TradeLeg("CMC_12M", -1.0, "sell" if direction == "long" else "buy"))
    side_word = _direction_word(direction, "spread")
    z_str = f"{z:+.2f}σ"
    hl_str = _hl_zone_label(diag.get("ou_half_life"))
    events_str = _events_in_window_str(panel, revert_d)
    headline = f"<b>Anchor (24M-12M)</b> {direction} · {z_str} · OU {hl_str.split(' ')[0]}"
    rationale = _build_rationale(
        what=f"<b>Anchor (12M-24M slope)</b> z = {z_str} · today {today_v:+.2f} bp vs 1y mean {fv_v:+.2f} bp",
        why=("12M-24M segment slope is the cleanest reading on cycle-belly steepness. "
              "Extreme z's tend to mean-revert as the policy path crystalizes."),
        trade=f"{side_word}: 24M leg {'+' if direction == 'long' else '−'} 12M leg · entry {today_v:+.2f} bp · target {fv_v:+.2f} bp",
        resolves=f"OU half-life {hl_str}; {events_str}",
    )
    out.append(TradeIdea(
        idea_id=iid, leg_fingerprint=fp,
        primary_source="anchor", sources=("anchor",), source_id=4,
        direction=direction, structure_type="basket",
        legs=legs,
        pv01_sum=0.0, entry_bp=today_v, fair_value_bp=fv_v,
        target_bp=fv_v, stop_bp=today_v + 2.5 * (today_v - fv_v),
        expected_revert_d=revert_d, expected_pnl_bp=pnl_bp,
        expected_pnl_dollar=pnl_bp * DEFAULT_PV01_PER_LEG * 2,
        z_score=z, half_life_d=diag.get("ou_half_life"),
        adf_pass=diag.get("adf_reject_5pct", False),
        eff_n=diag.get("adf_n_obs", 0),
        gate_quality="clean" if diag.get("adf_reject_5pct", False) else "non_stationary",
        rationale_html=rationale,
        headline_html=headline,
    ))
    return out


def gen_cross_pc_corr_breakdown_ideas(panel: dict) -> list:
    """Source #5 — Cross-PC corr breakdown overlay (decorates existing structures)."""
    out = []
    cross_corr = panel.get("cross_pc_corr")
    if cross_corr is None or cross_corr.empty:
        return out
    sig = cross_pc_corr_breakdown_signal(cross_corr, threshold=0.3)
    if sig.empty or not bool(sig.iloc[-1]):
        return out
    # Today's orthogonality is broken — emit a META "fade-the-breakdown" trade
    # by constructing a basket that's PC1-loaded with small PC2 counter-exposure.
    fp = "CROSS_PC_CORR_FADE"
    iid = _stable_idea_id("cross-PC-corr-breakdown", fp)
    out.append(TradeIdea(
        idea_id=iid, leg_fingerprint=fp,
        primary_source="cross-PC-corr-breakdown", sources=("cross-PC-corr-breakdown",),
        source_id=5,
        direction="neutral", structure_type="basket",
        legs=(),
        rationale_html="PCA orthogonality broken (corr > 0.3 today). "
                         "Slope/curvature trades systematically biased. "
                         "Reduce sizing on PC-isolated trades; consider fade.",
        gate_quality="regime_unstable",
        risk_flags=("regime_unstable",),
    ))
    return out


def gen_sparse_dense_divergence_ideas(panel: dict) -> list:
    """Source #6 — Sparse vs dense loadings divergence (informational overlay)."""
    out = []
    dense = panel.get("pca_fit_static")
    sparse = panel.get("sparse_pca_fit")
    if dense is None or sparse is None:
        return out
    div = sparse_dense_divergence(dense, sparse, threshold=0.05)
    flagged_tenors = []
    for k in (1, 2, 3):
        for tenor, info in div.get(f"PC{k}", {}).items():
            if info.get("flagged"):
                flagged_tenors.append((k, tenor))
    if not flagged_tenors:
        return out
    fp = f"SPARSE_DENSE_DIV_{len(flagged_tenors)}"
    iid = _stable_idea_id("sparse-dense-divergence", fp)
    out.append(TradeIdea(
        idea_id=iid, leg_fingerprint=fp,
        primary_source="sparse-dense-divergence",
        sources=("sparse-dense-divergence",),
        source_id=6, direction="neutral", structure_type="basket",
        legs=(),
        rationale_html=f"{len(flagged_tenors)} tenors with dense-vs-sparse "
                         f"loadings divergence > 0.05 — likely held up by noise. "
                         f"Fade individual tenor exposure on those.",
        gate_quality="clean",
    ))
    return out


def gen_variance_regime_ideas(panel: dict) -> list:
    """Source #7 — Variance regime (conviction overlay, not a trade)."""
    out = []
    fit = panel.get("pca_fit_static")
    if fit is None:
        return out
    label = variance_ratio_regime(fit)
    if label == "no_fit":
        return out
    fp = f"VARIANCE_REGIME_{label}"
    iid = _stable_idea_id("variance-regime", fp)
    pc1_pct = fit.variance_ratio[0] * 100 if len(fit.variance_ratio) > 0 else 0
    out.append(TradeIdea(
        idea_id=iid, leg_fingerprint=fp,
        primary_source="variance-regime",
        sources=("variance-regime",), source_id=7,
        direction="neutral", structure_type="basket",
        legs=(),
        rationale_html=f"Variance regime: {label} (PC1 share={pc1_pct:.1f}%). "
                         f"In low_pc1 regime, curvature flies pay best; "
                         f"in elevated_pc1, level trades dominate.",
        gate_quality="clean",
    ))
    return out


def gen_eigenspectrum_gap_ideas(panel: dict) -> list:
    """Source #8 — Eigenspectrum gap monitor (warns when PC2/PC3 indistinguishable)."""
    out = []
    rolling = panel.get("rolling_fits")
    if not rolling:
        return out
    gap = eigenspectrum_gap(rolling)
    if gap.empty:
        return out
    today_alert = bool(gap["gap_alert"].iloc[-1])
    if not today_alert:
        return out
    ratio = float(gap["gap_ratio"].iloc[-1])
    fp = "EIGEN_GAP_ALERT"
    iid = _stable_idea_id("eigenspectrum-gap", fp)
    out.append(TradeIdea(
        idea_id=iid, leg_fingerprint=fp,
        primary_source="eigenspectrum-gap", sources=("eigenspectrum-gap",),
        source_id=8, direction="neutral", structure_type="basket",
        legs=(),
        rationale_html=f"PC2/PC3 eigenvalue gap = {ratio:.2f} (< 1.5 threshold). "
                         f"Components statistically indistinguishable — pause PC3-fly emissions.",
        gate_quality="regime_unstable",
        risk_flags=("regime_unstable",),
    ))
    return out


def gen_pc1_asymmetry_ideas(panel: dict) -> list:
    """Source #9 — PC1 loadings asymmetry (front-led vs back-led regime tag)."""
    out = []
    fit = panel.get("pca_fit_static")
    if fit is None:
        return out
    asym = pc1_loading_asymmetry(fit)
    if asym["regime"] == "balanced" or asym["regime"] == "no_fit":
        return out
    fp = f"PC1_ASYM_{asym['regime']}"
    iid = _stable_idea_id("PC1-asymmetry", fp)
    out.append(TradeIdea(
        idea_id=iid, leg_fingerprint=fp,
        primary_source="PC1-asymmetry", sources=("PC1-asymmetry",),
        source_id=9, direction="neutral", structure_type="basket",
        legs=(),
        rationale_html=f"PC1 regime: {asym['regime']} "
                         f"(front_mean={asym['front_mean']:.3f}, back_mean={asym['back_mean']:.3f}). "
                         f"Hedge effectiveness shifts.",
        gate_quality="clean",
    ))
    return out


# =============================================================================
# Tier 3 — Multi-resolution PCA (front/belly/back) — wraps existing structures
#                                                       with model-source tag
# =============================================================================
def _gen_multires_ideas(panel: dict, segment: str, source_id: int) -> list:
    """Multi-resolution PCA structures from a specific segment fit.

    NOTE: Multi-resolution PCA fits live in `panel["multi_res_pca"]` once Phase 1
    extension is wired into the master orchestrator. For now this returns empty
    if the panel doesn't carry segment fits — graceful degradation.
    """
    out = []
    mr = panel.get("multi_res_pca", {})
    fit = mr.get(segment)
    if fit is None:
        return out
    # This is a meta-source: emits an informational marker that the
    # segment-specific PCA produced different loadings than full-curve.
    # Real generators would enumerate structures off this fit; for compactness
    # we surface it as a context idea consumed by the cross-confirm engine.
    var = fit.variance_ratio
    pc1_pct = float(var[0] * 100) if len(var) > 0 else 0
    fp = f"MULTIRES_{segment}_PC1_{pc1_pct:.0f}"
    iid = _stable_idea_id(SOURCE_NAMES[source_id], fp)
    out.append(TradeIdea(
        idea_id=iid, leg_fingerprint=fp,
        primary_source=SOURCE_NAMES[source_id],
        sources=(SOURCE_NAMES[source_id],),
        source_id=source_id, direction="neutral", structure_type="basket",
        legs=(),
        rationale_html=f"{segment.title()}-segment PCA: PC1 share={pc1_pct:.1f}%. "
                         f"Differs from full-curve fit; segment-specific dislocations possible.",
        gate_quality="clean",
    ))
    return out


def gen_front_pca_ideas(panel: dict) -> list:
    return _gen_multires_ideas(panel, "front", 10)


def gen_belly_pca_ideas(panel: dict) -> list:
    return _gen_multires_ideas(panel, "belly", 11)


def gen_back_pca_ideas(panel: dict) -> list:
    return _gen_multires_ideas(panel, "back", 12)


# =============================================================================
# Tier 4 — Residual fade trades (outright / spread / fly / pack)
# =============================================================================
def _residual_row_to_idea(row, source_id: int, structure_type: str,
                              panel: Optional[dict] = None) -> Optional[TradeIdea]:
    inst = str(row.get("instrument", ""))
    z = row.get("residual_z")
    today = row.get("residual_today_bp")
    mp = _panel_mode_params(panel or {})
    if z is None or not np.isfinite(z) or abs(z) < float(mp["z_threshold"]):
        return None
    # Phase A.4 — A6 triple-gate: suppress random-walk emissions
    upstream_gate = str(row.get("gate_quality", "clean"))
    if upstream_gate == "random_walk":
        return None    # gameplan §A6 — suppress random-walk emissions entirely
    direction = _direction_from_z(z)
    legs = (TradeLeg(symbol=inst, weight_pv01=1.0,
                       side="sell" if z > 0 else "buy", contracts=1),)
    fp = _fingerprint_legs(legs, direction)
    iid = _stable_idea_id(SOURCE_NAMES[source_id], fp)
    fv_bp = 0.0    # level-residual reverts to 0 after detrend
    pnl_bp = abs(today) if today is not None else 0.0
    hl_v = row.get("half_life")
    revert_d = _bounded_hold_days(hl_v, panel or {})
    # Build 4-part rationale + 1-line headline
    headline_html, rationale_html = _rationale_residual_fade(
        row, source_id, panel or {}, structure_type, direction,
        pnl_bp, pnl_bp * DEFAULT_PV01_PER_LEG, legs, revert_d,
    )
    return TradeIdea(
        idea_id=iid, leg_fingerprint=fp,
        primary_source=SOURCE_NAMES[source_id],
        sources=(SOURCE_NAMES[source_id],), source_id=source_id,
        direction=direction, structure_type=structure_type, legs=legs,
        pv01_sum=DEFAULT_PV01_PER_LEG,
        entry_bp=today, fair_value_bp=fv_bp, target_bp=fv_bp,
        stop_bp=(today or 0) + 2.5 * (today or 0),
        expected_revert_d=revert_d,
        expected_pnl_bp=pnl_bp,
        expected_pnl_dollar=pnl_bp * DEFAULT_PV01_PER_LEG,
        z_score=z, half_life_d=row.get("half_life"),
        adf_pass=bool(row.get("adf_pass", False)),
        triple_gate_pass=bool(row.get("triple_gate_pass", row.get("adf_pass", False))),
        eff_n=int(row.get("eff_n", 0)),
        gate_quality=str(row.get("gate_quality", "clean")),
        rationale_html=rationale_html,
        headline_html=headline_html,
    )


def gen_outright_fade_ideas(panel: dict) -> list:
    """Source #13 — Single-outright residual fade with PC1-isolated hedge."""
    df = panel.get("residual_outrights")
    if df is None or df.empty:
        return []
    out = []
    for _, row in df.iterrows():
        idea = _residual_row_to_idea(row, 13, "outright", panel)
        if idea is not None:
            out.append(idea)
    return out


def gen_spread_fade_ideas(panel: dict) -> list:
    """Source #14 — Traded-spread residual fade."""
    df = panel.get("residual_traded_spreads")
    if df is None or df.empty:
        return []
    return [i for i in (_residual_row_to_idea(row, 14, "spread", panel)
                          for _, row in df.iterrows()) if i is not None]


def gen_fly_arb_ideas(panel: dict) -> list:
    """Source #15 — Traded-fly residual vs PCA-fly arb."""
    df = panel.get("residual_traded_flies")
    if df is None or df.empty:
        return []
    return [i for i in (_residual_row_to_idea(row, 15, "fly", panel)
                          for _, row in df.iterrows()) if i is not None]


def gen_pack_ideas(panel: dict) -> list:
    """Source #16 — Pack rich/cheap (synthetic pack vs PCA model)."""
    df = panel.get("residual_packs")
    if df is None or df.empty:
        return []
    return [i for i in (_residual_row_to_idea(row, 16, "pack", panel)
                          for _, row in df.iterrows()) if i is not None]


# =============================================================================
# Tier 5 — Conditional FV (analog A2 / path A3)
# Note: A2/A3 are usually consumed as conviction inputs to existing ideas via
# the cross-confirmation engine, not as standalone idea sources. We surface a
# small set of high-conviction A2 disagreements as standalone ideas (when A2
# strongly disagrees with PCA-residual sign).
# =============================================================================
def gen_analog_fv_ideas(panel: dict) -> list:
    """Source #17 — Mahalanobis-LW analog FV strong disagreements."""
    out = []
    analog_panel = panel.get("analog_fv_results", {})
    if not analog_panel:
        return out
    for sid, fv in analog_panel.items():
        if fv.gate_quality != "clean":
            continue
        if fv.residual_z is None or abs(fv.residual_z) < 2.0:
            continue
        direction = _direction_from_z(fv.residual_z)
        legs = (TradeLeg(symbol=str(sid), weight_pv01=1.0,
                          side="sell" if fv.residual_z > 0 else "buy", contracts=1),)
        fp = _fingerprint_legs(legs, direction)
        iid = _stable_idea_id("analog-FV", fp)
        out.append(TradeIdea(
            idea_id=iid, leg_fingerprint=fp,
            primary_source="analog-FV", sources=("analog-FV",), source_id=17,
            direction=direction, structure_type="basket", legs=legs,
            entry_bp=fv.residual_today_bp, fair_value_bp=fv.fv_bp,
            target_bp=fv.fv_bp,
            expected_pnl_bp=abs(fv.residual_today_bp - fv.fv_bp),
            z_score=fv.residual_z, eff_n=int(fv.eff_n),
            analog_fv_bp=fv.fv_bp, analog_fv_z=fv.residual_z,
            analog_fv_eff_n=fv.eff_n,
            gate_quality=fv.gate_quality,
        ))
    return out


def gen_path_fv_ideas(panel: dict) -> list:
    """Source #18 — Path-conditional FV strong signals."""
    out = []
    path_panel = panel.get("path_fv_results", {})
    if not path_panel:
        return out
    for sid, fv in path_panel.items():
        if fv.gate_quality not in ("clean", "low_n"):
            continue
        if not np.isfinite(fv.fv_bp):
            continue
        # Surface when the path-conditional FV materially differs from raw FV
        legs = (TradeLeg(symbol=str(sid), weight_pv01=1.0, side="buy", contracts=1),)
        fp = _fingerprint_legs(legs, "neutral")
        iid = _stable_idea_id("path-FV", fp)
        out.append(TradeIdea(
            idea_id=iid, leg_fingerprint=fp,
            primary_source="path-FV", sources=("path-FV",), source_id=18,
            direction="neutral", structure_type="basket", legs=legs,
            fair_value_bp=fv.fv_bp,
            path_fv_bp=fv.fv_bp,
            eff_n=int(fv.eff_n_overall),
            gate_quality=fv.gate_quality,
        ))
    return out


# =============================================================================
# Tier 6 — Regime / change-point (gameplan §A1.5–§A9)
# =============================================================================
def _emit_regime_meta(label: str, source_id: int, src_name: str,
                       extra: str = "") -> TradeIdea:
    fp = f"REGIME_META_{src_name}"
    iid = _stable_idea_id(src_name, fp)
    return TradeIdea(
        idea_id=iid, leg_fingerprint=fp,
        primary_source=src_name, sources=(src_name,), source_id=source_id,
        direction="neutral", structure_type="basket", legs=(),
        rationale_html=f"{label} {extra}".strip(),
        gate_quality="regime_unstable",
        risk_flags=("regime_unstable",),
    )


def gen_gmm_regime_ideas(panel: dict) -> list:
    """Source #19 — GMM K=6 + HMM regime label (informational)."""
    regime_stack = panel.get("regime_stack", {})
    hmm = regime_stack.get("hmm_fit")
    if hmm is None:
        return []
    labels = hmm.smoothed_labels
    if labels is None or len(labels) == 0:
        return []
    today_label = int(labels[-1])
    today_conf = float(hmm.dominant_confidence[-1])
    fp = f"GMM_REGIME_{today_label}"
    iid = _stable_idea_id("GMM-HMM-regime", fp)
    return [TradeIdea(
        idea_id=iid, leg_fingerprint=fp,
        primary_source="GMM-HMM-regime", sources=("GMM-HMM-regime",),
        source_id=19, direction="neutral", structure_type="basket", legs=(),
        regime_label=f"R{today_label}",
        rationale_html=f"Current regime: R{today_label} (HMM posterior {today_conf:.2f}). "
                         f"{'Stable' if today_conf >= 0.6 else 'Unstable'} — "
                         f"{'reversion bias' if today_conf >= 0.6 else 'momentum bias'}.",
        gate_quality="clean" if today_conf >= 0.6 else "regime_unstable",
    )]


def gen_posterior_degradation_ideas(panel: dict) -> list:
    """Source #20 — GMM posterior degradation alert."""
    deg = panel.get("regime_stack", {}).get("posterior_degradation")
    if deg is None or deg.empty or not bool(deg.iloc[-1]):
        return []
    return [_emit_regime_meta("HMM posterior degradation fired today.",
                                20, "posterior-degradation",
                                "Reduce mean-reversion sizing.")]


def gen_bocpd_ideas(panel: dict) -> list:
    """Source #21 — BOCPD online change-point."""
    boc = panel.get("regime_stack", {}).get("bocpd_short_run")
    if boc is None or boc.empty or float(boc.iloc[-1]) <= 0.5:
        return []
    return [_emit_regime_meta(
        f"BOCPD P(short run) = {float(boc.iloc[-1]):.2f} > 0.5 today.",
        21, "BOCPD",
        "Imminent change-point — de-risk reversion trades.")]


def gen_bai_perron_ideas(panel: dict) -> list:
    """Source #22 — Bai-Perron offline structural breaks (recent only)."""
    bp_breaks = panel.get("regime_stack", {}).get("bai_perron_breaks", [])
    if not bp_breaks:
        return []
    last_bp = bp_breaks[-1] if bp_breaks else None
    today = panel.get("asof", date.today())
    if last_bp is None:
        return []
    days_since = (today - last_bp).days if hasattr(last_bp, "year") else 999
    if days_since > 30:
        return []
    return [_emit_regime_meta(
        f"Bai-Perron break detected at {last_bp} ({days_since}d ago).",
        22, "Bai-Perron",
        "Recent structural break — confirm regime stability before sizing up.")]


# =============================================================================
# Tier 7 — Calendar / event-aware (§A6s + §A12d + §A11-event)
# =============================================================================
def gen_seasonality_ideas(panel: dict) -> list:
    """Source #23 — Seasonality calendar effects."""
    out = []
    seasonality = panel.get("seasonality_results", {})
    for struct_id, results in seasonality.items():
        for effect, info in results.items():
            if not info.get("retained"):
                continue
            beta = info["beta_bp"]
            direction = "short" if beta > 0 else "long"
            fp = f"SEASONALITY_{struct_id}_{effect}_{direction}"
            iid = _stable_idea_id("seasonality", fp)
            out.append(TradeIdea(
                idea_id=iid, leg_fingerprint=fp,
                primary_source="seasonality", sources=("seasonality",),
                source_id=23, direction=direction, structure_type="basket",
                legs=(TradeLeg(symbol=str(struct_id), weight_pv01=1.0,
                                 side="sell" if beta > 0 else "buy", contracts=1),),
                seasonality_tag=effect,
                rationale_html=f"Seasonality: {effect} on {struct_id} → "
                                 f"β = {beta:+.2f} bp (BH-FDR retained).",
                expected_pnl_bp=abs(beta),
                gate_quality="clean",
            ))
    return out


def gen_event_drift_ideas(panel: dict) -> list:
    """Source #24 — Pre/post-event drift (currently uses B2 daily zscore proxy)."""
    out = []
    drift_df = panel.get("event_drift_table")
    if drift_df is None or drift_df.empty:
        return out
    retained = drift_df[drift_df.get("retained", False)] if "retained" in drift_df.columns else pd.DataFrame()
    for _, row in retained.iterrows():
        mean = row.get("drift_bp_mean", 0.0) or 0.0
        ev_class = row.get("event_class", "Event")
        window = row.get("window", "[0,5]")
        direction = "short" if mean > 0 else "long"
        fp = f"EVENT_DRIFT_{ev_class}_{window}_{direction}"
        iid = _stable_idea_id("event-drift", fp)
        out.append(TradeIdea(
            idea_id=iid, leg_fingerprint=fp,
            primary_source="event-drift", sources=("event-drift",), source_id=24,
            direction=direction, structure_type="basket", legs=(),
            rationale_html=f"{ev_class} drift in {window}: "
                             f"mean={mean:+.2f}bp (BH-retained). "
                             f"Position for typical drift.",
            expected_pnl_bp=abs(mean), gate_quality="circular_proxy",
            risk_flags=("circular_proxy",),
        ))
    return out


def gen_days_since_fomc_ideas(panel: dict) -> list:
    """Source #25 — Days-since-FOMC trade-timing overlay (informational)."""
    out = []
    fomc_dates = panel.get("fomc_calendar_dates", [])
    asof = panel.get("asof", date.today())
    if not fomc_dates:
        return out
    past = [d for d in fomc_dates if d <= asof]
    if not past:
        return out
    days_since = (asof - past[-1]).days
    fp = f"DAYS_SINCE_FOMC_{days_since}"
    iid = _stable_idea_id("days-since-FOMC", fp)
    out.append(TradeIdea(
        idea_id=iid, leg_fingerprint=fp,
        primary_source="days-since-FOMC", sources=("days-since-FOMC",),
        source_id=25, direction="neutral", structure_type="basket", legs=(),
        rationale_html=f"{days_since} trading days since last FOMC ({past[-1]}). "
                         f"Trade timing context: PC trajectories tend to mean-revert "
                         f"5-10d post-meeting.",
        gate_quality="clean",
    ))
    return out


def gen_event_impact_ranking_ideas(panel: dict) -> list:
    """Source #38 — A11-event-impact dynamic ticker importance ranking."""
    out = []
    ranking = panel.get("event_impact_ranking")
    if ranking is None or ranking.empty:
        return out
    # Surface the top-3 most important tickers as calendar trade primers
    top = ranking.nsmallest(3, "importance_rank")
    for _, row in top.iterrows():
        ticker = row["ticker"]
        importance = row["importance_score"]
        seg = row["dominant_segment"]
        h = row["dominant_horizon_d"]
        beta = abs(row["abs_beta_dom"])
        becoming = bool(row["becoming_more_important"])
        fp = f"IMPORTANCE_{ticker}"
        iid = _stable_idea_id("event-impact-ranking", fp)
        out.append(TradeIdea(
            idea_id=iid, leg_fingerprint=fp,
            primary_source="event-impact-ranking",
            sources=("event-impact-ranking",), source_id=38,
            direction="neutral", structure_type="basket", legs=(),
            rationale_html=f"Top-importance: {ticker} (rank {row['importance_rank']}, "
                             f"score {importance:+.2f}). β={beta:.2f} bp/σ on {seg} "
                             f"at T+{h}d. {'🔺 BECOMING MORE IMPORTANT' if becoming else ''}".strip(),
            gate_quality="clean" if becoming else "low_n",
        ))
    return out


# =============================================================================
# Tier 8 — Pack and bundle structures (§A2p)
# =============================================================================
def gen_pack_fly_ideas(panel: dict) -> list:
    """Source #26 — Pack-fly trades (white − 2·red + green)."""
    df = panel.get("residual_packs")
    if df is None or df.empty or len(df) < 3:
        return []
    out = []
    # Compute pack-fly from white/red/green pack residuals
    packs = df.set_index("instrument") if "instrument" in df.columns else df
    if "Whites" in packs.index and "Reds" in packs.index and "Greens" in packs.index:
        white_z = packs.loc["Whites", "residual_z"]
        red_z = packs.loc["Reds", "residual_z"]
        green_z = packs.loc["Greens", "residual_z"]
        if not (np.isfinite(white_z) and np.isfinite(red_z) and np.isfinite(green_z)):
            return out
        # Pack-fly z = white - 2*red + green (combined)
        fly_z = white_z - 2 * red_z + green_z
        if abs(fly_z) < float(_panel_mode_params(panel)["z_threshold"]):
            return out
        direction = _direction_from_z(fly_z)
        legs = (
            TradeLeg(symbol="Whites", weight_pv01=1.0,
                       side="buy" if direction == "long" else "sell"),
            TradeLeg(symbol="Reds", weight_pv01=-2.0,
                       side="sell" if direction == "long" else "buy"),
            TradeLeg(symbol="Greens", weight_pv01=1.0,
                       side="buy" if direction == "long" else "sell"),
        )
        fp = _fingerprint_legs(legs, direction)
        iid = _stable_idea_id("pack-fly", fp)
        out.append(TradeIdea(
            idea_id=iid, leg_fingerprint=fp,
            primary_source="pack-fly", sources=("pack-fly",),
            source_id=26, direction=direction, structure_type="fly", legs=legs,
            z_score=fly_z, gate_quality="clean",
            rationale_html=f"Pack-fly W-2R+G z={fly_z:+.2f}σ. "
                             f"Curvature on packs (more liquid than outright flies).",
        ))
    return out


def gen_bundle_rv_ideas(panel: dict) -> list:
    """Source #27 — Bundle relative value (informational)."""
    df = panel.get("residual_packs")
    if df is None or df.empty:
        return []
    # Bundle = Whites + Reds + Greens + Blues (4y duration)
    out = []
    packs = df.set_index("instrument") if "instrument" in df.columns else df
    needed = ["Whites", "Reds", "Greens", "Blues"]
    if all(p in packs.index for p in needed):
        bundle_z = packs.loc[needed, "residual_z"].mean()
        if abs(bundle_z) < float(_panel_mode_params(panel)["z_threshold"]):
            return out
        direction = _direction_from_z(bundle_z)
        legs = tuple(
            TradeLeg(symbol=p, weight_pv01=0.25, side="buy" if direction == "long" else "sell")
            for p in needed
        )
        fp = _fingerprint_legs(legs, direction)
        iid = _stable_idea_id("bundle-RV", fp)
        out.append(TradeIdea(
            idea_id=iid, leg_fingerprint=fp,
            primary_source="bundle-RV", sources=("bundle-RV",),
            source_id=27, direction=direction, structure_type="basket", legs=legs,
            z_score=bundle_z, gate_quality="clean",
            rationale_html=f"Bundle (W+R+G+B) z={bundle_z:+.2f}σ. "
                             f"4y synthetic duration play.",
        ))
    return out


# =============================================================================
# Tier 9 — Momentum / cycle / outlier
# =============================================================================
def gen_pc_momentum_ideas(panel: dict) -> list:
    """Source #28 — PC momentum continuation (TSM agreement at multiple horizons)."""
    out = []
    pc_panel = panel.get("pc_panel")
    if pc_panel is None or pc_panel.empty:
        return out
    horizons = [21, 63, 126, 252]
    for col in ("PC1", "PC2", "PC3"):
        if col not in pc_panel.columns:
            continue
        s = pc_panel[col].dropna()
        if len(s) < max(horizons) + 1:
            continue
        signs = []
        for h in horizons:
            if len(s) <= h:
                continue
            change = s.iloc[-1] - s.iloc[-1 - h]
            signs.append(np.sign(change))
        if not signs:
            continue
        # Require at least 3 horizons agreeing in sign
        agree = sum(1 for x in signs if x == signs[-1])
        if agree < 3:
            continue
        direction = "long" if signs[-1] > 0 else "short"
        fp = f"PC_MOMENTUM_{col}_{direction}"
        iid = _stable_idea_id("PC-momentum", fp)
        out.append(TradeIdea(
            idea_id=iid, leg_fingerprint=fp,
            primary_source="PC-momentum", sources=("PC-momentum",),
            source_id=28, direction=direction, structure_type="basket", legs=(),
            rationale_html=f"{col} momentum: {agree}/{len(signs)} horizons agree on "
                             f"{direction} bias. Trend-continuation candidate.",
            gate_quality="clean",
        ))
    return out


def gen_cycle_phase_ideas(panel: dict) -> list:
    """Source #29 — Cycle-phase aligned bias (favoured-trade table)."""
    out = []
    cycle = panel.get("cycle_phase")
    if not cycle:
        return out
    phase, inputs = cycle if isinstance(cycle, tuple) else (cycle, {})
    fp = f"CYCLE_PHASE_{phase}"
    iid = _stable_idea_id("cycle-phase", fp)
    favoured_map = {
        "early-cut": "Long-belly fly (PC3 curvature short)",
        "mid-cut": "Steepener (PC2-isolated)",
        "late-cut": "Flattener (PC2-isolated)",
        "trough": "Aggressive sizing on any reversion",
        "early-hike": "Short-belly fly (PC3 curvature long)",
        "mid-hike": "Slope-flattener with caution",
        "late-hike": "Curvature flies (PC3 pays best historically)",
        "peak": "Aggressive mean-reversion sizing",
    }
    favoured = favoured_map.get(phase, "Mixed bias")
    out.append(TradeIdea(
        idea_id=iid, leg_fingerprint=fp,
        primary_source="cycle-phase", sources=("cycle-phase",),
        source_id=29, direction="neutral", structure_type="basket", legs=(),
        cycle_phase=phase, cycle_alignment="favoured",
        rationale_html=f"Cycle phase: {phase}. Favoured: {favoured}",
        gate_quality="low_n",
    ))
    return out


def gen_outlier_reversal_ideas(panel: dict) -> list:
    """Source #30 — Outlier-day reversal (fade days with recon error > 99th pct)."""
    out = []
    outliers = panel.get("outlier_days", [])
    if not outliers:
        return out
    asof = panel.get("asof", date.today())
    today_ts = pd.Timestamp(asof)
    # Check if today is an outlier
    is_today_outlier = any(pd.Timestamp(d).date() == asof for d in outliers)
    if not is_today_outlier:
        return out
    fp = "OUTLIER_DAY_FADE"
    iid = _stable_idea_id("outlier-reversal", fp)
    out.append(TradeIdea(
        idea_id=iid, leg_fingerprint=fp,
        primary_source="outlier-reversal", sources=("outlier-reversal",),
        source_id=30, direction="neutral", structure_type="basket", legs=(),
        rationale_html="Today is a 99th-pct reconstruction-error outlier. "
                         "Fade tomorrow's residual move conditional on regime stability.",
        gate_quality="clean",
    ))
    return out


def gen_carry_fly_ideas(panel: dict) -> list:
    """Source #31 — Carry-positive PCA-fly (composite PC3 + carry)."""
    out = []
    cands = [c for c in panel.get("structure_candidates", [])
              if c.target_pc == 3 and c.gate_quality == "clean"]
    # Without a dedicated carry calc, approximate carry from CMC slope at fly tenors
    # (carry = expected drift if curve doesn't change). Place positive-carry flies
    # at higher conviction; negative-carry flies tagged.
    for c in cands[:20]:
        if c.composite_score is None:
            continue
        if abs(c.composite_score) < 0.5:
            continue
        # Approximate carry via the structure's PV01-weighted forward yield
        # (a tiny bias; real impl would compute the carry roll explicitly)
        idea = _idea_from_structure(c, 31, panel)
        if idea is None:
            continue
        idea = replace(idea, primary_source="carry-fly", sources=("carry-fly",))
        out.append(idea)
    return out


# =============================================================================
# Tier 10 — Position-aware overlays (not generators)
# =============================================================================
def attribute_pnl(positions: pd.DataFrame,
                    panel: dict) -> dict:
    """Decompose today's portfolio P&L into PC1/PC2/PC3/residual contributions.

    `positions` — DataFrame with columns [symbol, contracts, side]
                  (side="long" → positive contracts; "short" → negative).
    """
    out = {"total_pnl_dollar": 0.0,
           "pc1_pnl": 0.0, "pc2_pnl": 0.0, "pc3_pnl": 0.0,
           "residual_pnl": 0.0, "by_position": []}
    if positions is None or positions.empty:
        return out
    fit = panel.get("pca_fit_static")
    if fit is None:
        return out
    delta_decomp = panel.get("delta_decomposition", {})
    pc_contribs = delta_decomp.get("pc1_contrib_bp", []), \
                    delta_decomp.get("pc2_contrib_bp", []), \
                    delta_decomp.get("pc3_contrib_bp", [])
    residual = delta_decomp.get("residual_bp", [])
    tenors = delta_decomp.get("tenors_months", [])
    if not tenors or not pc_contribs[0]:
        return out

    # Per-position attribution: for each position, find its tenor and sum the contribs
    for _, row in positions.iterrows():
        sym = row.get("symbol")
        contracts = float(row.get("contracts", 0))
        side = row.get("side", "long")
        sign = 1 if side == "long" else -1
        # Position's tenor index (approximate)
        # In real use, look up via reference_period; here just use first tenor
        # as fallback if we can't find symbol
        tenor_idx = 0
        for i, t in enumerate(tenors):
            if str(sym).startswith("SRA") and i < len(pc_contribs[0]):
                tenor_idx = i
                break
        pc1_pnl = float(pc_contribs[0][tenor_idx] or 0) * contracts * sign * SRA_TICK_VALUE_USD
        pc2_pnl = float(pc_contribs[1][tenor_idx] or 0) * contracts * sign * SRA_TICK_VALUE_USD
        pc3_pnl = float(pc_contribs[2][tenor_idx] or 0) * contracts * sign * SRA_TICK_VALUE_USD
        res_pnl = float(residual[tenor_idx] or 0) * contracts * sign * SRA_TICK_VALUE_USD
        out["pc1_pnl"] += pc1_pnl
        out["pc2_pnl"] += pc2_pnl
        out["pc3_pnl"] += pc3_pnl
        out["residual_pnl"] += res_pnl
        out["by_position"].append({
            "symbol": sym, "contracts": contracts, "side": side,
            "pc1_pnl": pc1_pnl, "pc2_pnl": pc2_pnl, "pc3_pnl": pc3_pnl,
            "residual_pnl": res_pnl,
        })
    out["total_pnl_dollar"] = out["pc1_pnl"] + out["pc2_pnl"] + out["pc3_pnl"] + out["residual_pnl"]
    return out


def position_factor_exposure(positions: pd.DataFrame, fit: PCAFit) -> dict:
    """Net PC1/PC2/PC3 exposure per σ-move."""
    out = {"PC1_per_sigma_dollar": 0.0,
           "PC2_per_sigma_dollar": 0.0,
           "PC3_per_sigma_dollar": 0.0}
    if positions is None or positions.empty or fit is None:
        return out
    L = fit.loadings
    sigma_pc = np.sqrt(fit.eigenvalues) if len(fit.eigenvalues) >= 3 else np.array([1.0, 1.0, 1.0])
    for _, row in positions.iterrows():
        sym = row.get("symbol")
        contracts = float(row.get("contracts", 0))
        side = row.get("side", "long")
        sign = 1 if side == "long" else -1
        # Approximate: average PC loading across tenors
        for k in range(min(3, L.shape[0])):
            avg_loading = float(np.mean(L[k]))
            out[f"PC{k+1}_per_sigma_dollar"] += avg_loading * sigma_pc[k] * contracts * sign * SRA_TICK_VALUE_USD
    return out


def load_positions(path: str = "positions.csv") -> Optional[pd.DataFrame]:
    """Load optional positions.csv (columns: symbol, contracts, side)."""
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


# =============================================================================
# Tier 11 — Execution gates
# =============================================================================
def estimate_slippage(idea: TradeIdea, volume_panel: Optional[pd.DataFrame] = None) -> float:
    """Volume-conditioned slippage estimate in bp.

    Heuristic: slippage = base + size_factor / liquidity. Without volume_panel,
    return a base estimate based on leg count.
    """
    n_legs = len(idea.legs)
    if n_legs == 0:
        return 0.0
    base_per_leg = 0.1   # 0.1 bp per leg base slippage
    if volume_panel is None or volume_panel.empty:
        return base_per_leg * n_legs
    # Per-leg volume lookup
    total = base_per_leg * n_legs
    for leg in idea.legs:
        if leg.symbol in volume_panel.columns:
            avg_vol = float(volume_panel[leg.symbol].tail(60).mean() or 1000)
            # Higher vol → less slippage. Simple inverse scaling.
            total += 1000 / max(avg_vol, 100) * 0.05
    return float(total)


def convexity_flag(idea: TradeIdea, max_tenor_months: int = 36) -> bool:
    """True if any leg is at back-end tenor where convexity adjustment matters."""
    for leg in idea.legs:
        sym = leg.symbol
        # Heuristic: SRA outright tenors past 3y carry convexity bias
        if sym.startswith("SRA") and len(sym) >= 5:
            try:
                yy = int(sym[4:6])
                # Fairly approximate: anything more than 3y forward is back-end
                # (better implementation would use reference_period from lib.fomc)
                if 26 <= yy <= 35:    # 2026-2035 contracts
                    pass    # need asof to compute precise tenor; pass for now
            except ValueError:
                continue
    return False


def event_in_window_flag(idea: TradeIdea,
                           fomc_dates: list,
                           asof: date) -> bool:
    """True if expected revert period (1.5×HL) spans a Tier-1 event."""
    rd = idea.expected_revert_d
    if rd is None or not np.isfinite(rd) or rd <= 0:
        return False
    end_date = asof + timedelta(days=int(rd * 1.5))
    for d in fomc_dates or []:
        ts = pd.Timestamp(d).date() if hasattr(d, "year") else d
        if asof <= ts <= end_date:
            return True
    return False


def hard_block_filter(ideas: list, panel: dict) -> list:
    """Apply execution-quality gates: print quality + event-in-window."""
    out = []
    print_alerts = panel.get("print_quality_alerts", [])
    asof = panel.get("asof", date.today())
    fomc_dates = panel.get("fomc_calendar_dates", [])
    today_print_bad = any(pd.Timestamp(d).date() == asof for d in print_alerts)

    for idea in ideas:
        flags = list(idea.risk_flags)
        gate = idea.gate_quality
        new_print_quality = idea.print_quality_today
        # Print quality filter
        if today_print_bad:
            new_print_quality = False
            if "print_quality" not in flags:
                flags.append("print_quality")
        # Event-in-window filter
        if event_in_window_flag(idea, fomc_dates, asof):
            if "event_in_window" not in flags:
                flags.append("event_in_window")
        new_idea = replace(idea, risk_flags=tuple(flags),
                            print_quality_today=new_print_quality,
                            convexity_warning=convexity_flag(idea))
        out.append(new_idea)
    return out


# =============================================================================
# Cross-confirmation engine
# =============================================================================
def cluster_by_fingerprint(raw_ideas: list) -> list:
    """Group ideas by leg_fingerprint. Returns list of clusters (lists)."""
    by_fp: dict = {}
    for idea in raw_ideas:
        by_fp.setdefault(idea.leg_fingerprint, []).append(idea)
    return list(by_fp.values())


def merge_cluster(cluster: list) -> TradeIdea:
    """Merge confirming ideas into a primary. Cross-source FV fields (analog_fv_*,
    path_fv_*) propagate from ANY cluster member to the primary, so a residual-fade
    idea backed by an analog-FV agreement gets credit even if the primary itself
    doesn't carry analog data.
    """
    if not cluster:
        raise ValueError("merge_cluster called with empty cluster")
    if len(cluster) == 1:
        return cluster[0]
    primary = max(cluster, key=lambda i: i.conviction or 0.0)
    sources = tuple(sorted(set(s for i in cluster for s in i.sources)))
    pills = sources

    # Backfill cross-source FV fields when primary doesn't carry them
    def _first_set(field: str):
        for i in cluster:
            v = getattr(i, field, None)
            if v is not None:
                return v
        return None

    afv_bp = primary.analog_fv_bp if primary.analog_fv_bp is not None else _first_set("analog_fv_bp")
    afv_z = primary.analog_fv_z if primary.analog_fv_z is not None else _first_set("analog_fv_z")
    afv_n = primary.analog_fv_eff_n if primary.analog_fv_eff_n is not None else _first_set("analog_fv_eff_n")
    pfv_bp = primary.path_fv_bp if primary.path_fv_bp is not None else _first_set("path_fv_bp")
    pfv_z = primary.path_fv_z if primary.path_fv_z is not None else _first_set("path_fv_z")

    return replace(primary,
                     sources=sources,
                     n_confirming_sources=len(cluster),
                     confirming_pills=pills,
                     analog_fv_bp=afv_bp, analog_fv_z=afv_z, analog_fv_eff_n=afv_n,
                     path_fv_bp=pfv_bp, path_fv_z=pfv_z)


# =============================================================================
# Lifecycle ledger
# =============================================================================
def load_ledger(path: str = LEDGER_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=[
            "asof_date", "idea_id", "source", "leg_fingerprint",
            "z_score", "conviction", "state", "entry_residual",
            "current_residual", "days_alive", "max_conviction",
        ])
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


def update_ledger(asof: date, ideas: list,
                    path: str = LEDGER_PATH) -> pd.DataFrame:
    ledger = load_ledger(path)
    today_str = pd.Timestamp(asof).date().isoformat()
    rows = []
    for idea in ideas:
        rows.append({
            "asof_date": today_str, "idea_id": idea.idea_id,
            "source": idea.primary_source,
            "leg_fingerprint": idea.leg_fingerprint,
            "z_score": idea.z_score, "conviction": idea.conviction,
            "state": idea.state, "entry_residual": idea.entry_bp,
            "current_residual": idea.entry_bp,
            "days_alive": idea.days_alive,
            "max_conviction": idea.max_conviction_to_date,
        })
    new_rows = pd.DataFrame(rows)
    if not ledger.empty:
        # Drop today's duplicates if re-running
        ledger = ledger[ledger["asof_date"] != today_str]
    out = pd.concat([ledger, new_rows], ignore_index=True)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        out.to_parquet(path, index=False)
    except Exception:
        pass
    return out


def assign_lifecycle_state(idea: TradeIdea, ledger: pd.DataFrame) -> str:
    """Determine lifecycle state from ledger history."""
    if ledger.empty or "leg_fingerprint" not in ledger.columns:
        return "NEW"
    history = ledger[ledger["leg_fingerprint"] == idea.leg_fingerprint]
    if history.empty:
        return "NEW"
    n = len(history)
    if n == 1:
        return "NEW"
    if n <= 5:
        # MATURING if conviction rising
        recent = history.tail(3)["conviction"].values
        if len(recent) >= 2 and recent[-1] > recent[0]:
            return "MATURING"
        return "FADING"
    # Track max-conviction to detect peak/fade
    max_idx = history["conviction"].idxmax()
    last_idx = history.index[-1]
    if max_idx == last_idx:
        return "PEAK"
    # Conviction declining 2+ days from peak
    declining_days = (history.index > max_idx).sum()
    if declining_days >= 2:
        return "FADING"
    return "MATURING"


def compute_track_record(ledger: pd.DataFrame, *, lookback_n: int = 30) -> dict:
    """Per-source REVERTED / FAILED / TIMED-OUT counts for last `lookback_n` ideas."""
    out = {}
    if ledger.empty:
        return out
    for source in ledger["source"].unique():
        sub = ledger[ledger["source"] == source].tail(lookback_n)
        states = sub["state"].value_counts().to_dict()
        n_total = len(sub)
        out[source] = {
            "n_total": n_total,
            "revert_rate": states.get("REVERTED", 0) / max(1, n_total),
            "fail_rate": states.get("FAILED", 0) / max(1, n_total),
            "timeout_rate": states.get("TIMED_OUT", 0) / max(1, n_total),
            "active": states.get("NEW", 0) + states.get("MATURING", 0) + states.get("PEAK", 0),
        }
    return out


# =============================================================================
# 17-input conviction scorer
# =============================================================================
def _sweet_spot_score(hl: Optional[float], mode_p: dict) -> float:
    """Return the OU half-life sweet-spot score [0,1] based on the mode's bands.

    Mode params define `sweet_spot_full` (full credit window), `sweet_spot_mid_band`
    (partial credit just above the full band), and the partial-below/above thresholds.
    """
    if hl is None or not np.isfinite(hl) or hl <= 0:
        return 0.0
    full_lo, full_hi = mode_p["sweet_spot_full"]
    mid_lo, mid_hi = mode_p["sweet_spot_mid_band"]
    below = mode_p["sweet_spot_partial_below"]
    above = mode_p["sweet_spot_partial_above"]
    if hl < below:
        return 0.2
    if hl < full_lo:
        return 0.5
    if hl <= full_hi:
        return 1.0
    if hl <= mid_hi:
        return 0.6
    if hl <= above:
        return 0.3
    return 0.1


def _seasonality_score(idea: TradeIdea, panel: dict) -> float:
    """Activated seasonality input — returns [0,1] credit.

    Two paths:
      (a) If idea has a `seasonality_tag` already populated by a generator, map
          "favoured" → 1.0, "neutral" → 0.5, "counter" → 0.0.
      (b) Otherwise, derive from cycle_phase × direction. In late_easing,
          flatteners and front-end longs are favoured; in mid_hiking, the
          reverse.
    """
    tag = (idea.seasonality_tag or "").lower()
    if tag == "favoured":
        return 1.0
    if tag == "counter":
        return 0.0
    if tag == "neutral":
        return 0.5
    # Derive from cycle_phase
    phase = (idea.cycle_phase or panel.get("cycle_phase", ("neutral", {}))[0] or "neutral").lower()
    direction = (idea.direction or "neutral").lower()
    # Late easing — front-end longs favoured (rates falling)
    if "easing" in phase and direction in ("long", "steepen"):
        return 0.7
    if "easing" in phase and direction in ("short", "flatten"):
        return 0.3
    # Mid hiking — front-end shorts favoured
    if "hiking" in phase and direction in ("short", "flatten"):
        return 0.7
    if "hiking" in phase and direction in ("long", "steepen"):
        return 0.3
    return 0.5    # neutral


def _event_drift_score(idea: TradeIdea, panel: dict) -> float:
    """Activated event-drift input — returns [0,1] credit.

    Considers proximity to next FOMC. Trades held over an event get penalized
    (event risk); trades resolving well before the next event get a bonus.
    Uses idea.expected_revert_d and panel["fomc_calendar_dates"].
    """
    asof = panel.get("asof")
    revert_d = idea.expected_revert_d
    if asof is None or revert_d is None or not np.isfinite(revert_d):
        return 0.5
    fomc = panel.get("fomc_calendar_dates", []) or []
    upcoming = sorted([pd.Timestamp(d).date() for d in fomc
                        if pd.Timestamp(d).date() > asof])
    if not upcoming:
        return 0.5
    days_to_next = (upcoming[0] - asof).days
    horizon = max(1, int(round(revert_d * 1.5)))
    # Trade resolves >5d before next event — bonus
    if days_to_next > horizon + 5:
        return 0.85
    # Trade resolves just before event (1-5d) — neutral plus
    if days_to_next > horizon:
        return 0.65
    # Event sits inside the hold window — penalty
    if days_to_next <= horizon:
        return 0.25
    return 0.5


def _empirical_hit_rate_score(idea: TradeIdea, panel: dict, ledger: pd.DataFrame) -> float:
    """Empirical hit rate — prefers backtested per-source rates when available,
    falls back to ledger track-record (which exists earlier in the day).

    Returns the hit rate normalized to [0,1] where 0.5 = coin-flip neutral.
    """
    # Phase 5 wiring: panel["empirical_hit_rates"] gets populated by backtest.
    ehr = panel.get("empirical_hit_rates") or {}
    src_key = idea.primary_source
    revert_d = idea.expected_revert_d or 30.0
    if src_key in ehr:
        # Pick the closest horizon bucket
        per_h = ehr[src_key]
        if isinstance(per_h, dict) and per_h:
            horizons = []
            for h_key in per_h.keys():
                # Keys like "horizon_30d" → extract integer
                try:
                    if isinstance(h_key, str) and h_key.startswith("horizon_"):
                        h_int = int(h_key.replace("horizon_", "").replace("d", ""))
                    else:
                        h_int = int(h_key)
                    horizons.append((h_int, h_key))
                except Exception:
                    continue
            if horizons:
                horizons.sort(key=lambda hh: abs(hh[0] - float(revert_d)))
                best_key = horizons[0][1]
                rec = per_h[best_key]
                if isinstance(rec, dict) and "hit_rate" in rec:
                    return float(rec["hit_rate"])
    # Fallback: live ledger track-record
    try:
        track = compute_track_record(ledger, lookback_n=30)
        src_record = track.get(idea.primary_source, {})
        return float(src_record.get("revert_rate", 0.5))
    except Exception:
        return 0.5


def _exit_clarity_score(idea: TradeIdea, panel: dict) -> float:
    """Exit-clarity input — credit if T1/S1 are at reasonable distances from
    entry, so the trade has clear target/stop separation rather than being cramped.

    1.0 if both target and stop are present and stop-distance >= 0.5 of expected
    PnL distance (room to breathe).
    0.5 if either missing.
    0.0 if stop distance is smaller than expected PnL (cramped).
    """
    pnl_bp = idea.expected_pnl_bp
    entry = idea.entry_bp
    target = idea.target_bp
    stop = idea.stop_bp
    if pnl_bp is None or entry is None or target is None or stop is None:
        return 0.5
    if not (np.isfinite(pnl_bp) and np.isfinite(entry) and np.isfinite(target) and np.isfinite(stop)):
        return 0.5
    target_dist = abs(target - entry)
    stop_dist = abs(stop - entry)
    if target_dist <= 0:
        return 0.0
    ratio = stop_dist / target_dist
    if ratio < 0.3:
        return 0.0    # cramped: stop too tight
    if ratio < 0.5:
        return 0.5
    if ratio <= 1.5:
        return 1.0    # well-balanced
    return 0.7        # very wide stop — okay but worse R:R


def score_conviction(idea: TradeIdea, panel: dict, ledger: pd.DataFrame,
                      mode: Optional[str] = None) -> tuple:
    """Compute composite conviction (0..1) and breakdown.

    Phase-4-recalibrated 19-input blend (mode-aware sweet-spot, activated
    placeholders for seasonality / event_drift / empirical_hit_rate, new
    exit_clarity input). Weights sum to 1.00 exactly.

    Mode (`"intraday" | "swing" | "positional"`) re-keys the OU sweet-spot
    bands. Defaults to `panel["mode"]` if present, else `DEFAULT_MODE`.
    """
    # Resolve mode
    from lib.pca import mode_params, DEFAULT_MODE
    eff_mode = mode if mode in ("intraday", "swing", "positional") else panel.get("mode") or DEFAULT_MODE
    mp = mode_params(eff_mode)

    components = []

    # 1. z-significance (0.12) — cumulative deviation magnitude
    z_v = idea.z_score
    sig_z = min(1.0, abs(z_v) / 3.0) if z_v is not None else 0.0
    components.append(("z_significance", 0.12 * sig_z))

    # 2. OU sweet spot (0.10) — mode-aware HL band
    hl_score = _sweet_spot_score(idea.half_life_d, mp)
    components.append(("ou_sweet_spot", 0.10 * hl_score))

    # 3. Triple-gate pass (0.08) — replaces ADF-only check
    # Fall back to adf_pass when triple_gate_pass not populated by generator.
    gate_ok = bool(idea.triple_gate_pass or idea.adf_pass)
    components.append(("triple_gate_pass", 0.08 if gate_ok else 0.0))

    # 4. eff_n (0.05) — sample size
    eff_score = min(1.0, idea.eff_n / 100.0)
    components.append(("eff_n", 0.05 * eff_score))

    # 5. Model fit (0.05) — 3-PC explained today
    pct_explained = panel.get("reconstruction_pct_today") or 0.0
    components.append(("model_fit", 0.05 * float(pct_explained)))

    # 6. Cycle alignment (0.07)
    align_map = {"favoured": 1.0, "neutral": 0.5, "counter": 0.0}
    components.append(("cycle_alignment", 0.07 * align_map.get(idea.cycle_alignment, 0.5)))

    # 7. Regime stable (0.05) — HMM dominant confidence
    regime_stack = panel.get("regime_stack", {})
    hmm = regime_stack.get("hmm_fit") if regime_stack else None
    regime_stable = bool(hmm and hmm.dominant_confidence is not None
                            and len(hmm.dominant_confidence) > 0
                            and float(hmm.dominant_confidence[-1]) >= 0.6)
    components.append(("regime_stable", 0.05 if regime_stable else 0.0))

    # 8. Cross-PC orthogonality (0.04)
    cross_corr = panel.get("cross_pc_corr")
    if cross_corr is not None and not cross_corr.empty:
        last_corr = float(cross_corr.abs().iloc[-1].max() if not cross_corr.iloc[-1].dropna().empty else 0)
        components.append(("cross_pc_orth", 0.04 if last_corr <= 0.3 else 0.0))
    else:
        components.append(("cross_pc_orth", 0.02))

    # 9. Variance regime (0.04)
    fit = panel.get("pca_fit_static")
    if fit is not None:
        regime_label = variance_ratio_regime(fit)
        favours = (
            (regime_label == "low_pc1" and idea.source_id in (1, 26))
            or (regime_label == "elevated_pc1" and idea.source_id in (3, 13))
            or (regime_label == "normal")
        )
        components.append(("variance_regime", 0.04 if favours else 0.0))
    else:
        components.append(("variance_regime", 0.02))

    # 10. Analog FV agreement (0.05)
    if idea.analog_fv_z is not None and idea.z_score is not None:
        agree = (np.sign(idea.analog_fv_z) == np.sign(idea.z_score)
                  if abs(idea.analog_fv_z) > 0.5 else True)
        components.append(("analog_fv_agree", 0.05 if agree else 0.0))
    else:
        components.append(("analog_fv_agree", 0.025))

    # 11. Path FV agreement (0.04)
    if idea.path_fv_z is not None and idea.z_score is not None:
        agree = np.sign(idea.path_fv_z) == np.sign(idea.z_score)
        components.append(("path_fv_agree", 0.04 if agree else 0.0))
    else:
        components.append(("path_fv_agree", 0.02))

    # 12. Seasonality (0.03) — ACTIVATED
    components.append(("seasonality", 0.03 * _seasonality_score(idea, panel)))

    # 13. Event drift (0.03) — ACTIVATED
    components.append(("event_drift", 0.03 * _event_drift_score(idea, panel)))

    # 14. Empirical hit rate (0.06) — ACTIVATED with backtest priority
    components.append(("empirical_hit_rate",
                        0.06 * _empirical_hit_rate_score(idea, panel, ledger)))

    # 15. Lifecycle (0.04)
    if idea.state in ("NEW", "MATURING", "PEAK"):
        components.append(("lifecycle", 0.04))
    elif idea.state == "FADING":
        components.append(("lifecycle", -0.02))
    else:
        components.append(("lifecycle", 0.0))

    # 16. Cross-confirmation (0.05)
    n_conf = max(1, idea.n_confirming_sources)
    components.append(("cross_confirm", 0.05 * np.log(1 + n_conf) / np.log(6)))

    # 17. Slippage acceptable (0.03)
    pnl_bp = idea.expected_pnl_bp or 0.0
    slip = idea.slippage_estimate_bp or 0.5
    if pnl_bp > 0 and slip / pnl_bp < 0.30:
        components.append(("slippage_ok", 0.03))
    else:
        components.append(("slippage_ok", 0.0))

    # 18. Convexity safe (0.03)
    components.append(("convexity_safe", 0.0 if idea.convexity_warning else 0.03))

    # 19. Exit clarity (0.04) — NEW — clear target/stop separation
    components.append(("exit_clarity", 0.04 * _exit_clarity_score(idea, panel)))

    total = sum(v for _, v in components)
    total = max(0.0, min(1.0, total))
    return total, tuple(components)


# =============================================================================
# Risk + what-if
# =============================================================================
def compute_factor_exposure(idea: TradeIdea, fit: PCAFit) -> dict:
    """Per-PC σ-loading of the trade."""
    out = {"PC1_sigma": 0.0, "PC2_sigma": 0.0, "PC3_sigma": 0.0}
    if fit is None or not idea.legs:
        return out
    L = fit.loadings
    sigma_pc = np.sqrt(fit.eigenvalues) if len(fit.eigenvalues) >= 3 else np.array([1.0, 1.0, 1.0])
    # Approximate per-leg loading via average loading across tenors
    for k in range(min(3, L.shape[0])):
        avg_load = float(np.mean(L[k]))
        net = sum(leg.weight_pv01 * avg_load for leg in idea.legs)
        out[f"PC{k+1}_sigma"] = net * float(sigma_pc[k]) * SRA_TICK_VALUE_USD
    return out


def what_if_pc_shock(idea: TradeIdea, fit: PCAFit, pc: int, sigma: float) -> float:
    """P&L impact ($) of a PC-σ shock."""
    if fit is None or pc < 1 or pc > fit.loadings.shape[0]:
        return 0.0
    L = fit.loadings
    sigma_pc = np.sqrt(fit.eigenvalues[pc - 1]) if len(fit.eigenvalues) >= pc else 1.0
    avg_load = float(np.mean(L[pc - 1]))
    net = sum(leg.weight_pv01 * avg_load for leg in idea.legs)
    return float(net * sigma_pc * sigma * SRA_TICK_VALUE_USD)


def what_if_event(idea: TradeIdea, event_class: str, surprise_magnitude: float,
                    panel: dict) -> float:
    """P&L impact of a fundamental-surprise event ($)."""
    ranking = panel.get("event_impact_ranking")
    if ranking is None or ranking.empty:
        # Generic fallback: 1.5 bp/σ across legs
        return 1.5 * surprise_magnitude * SRA_TICK_VALUE_USD * len(idea.legs)
    # Lookup event_class average β
    sub = ranking[ranking.get("event_class", "") == event_class]
    if sub.empty:
        return 1.5 * surprise_magnitude * SRA_TICK_VALUE_USD * len(idea.legs)
    avg_beta = float(sub["abs_beta_dom"].mean())
    return float(avg_beta * surprise_magnitude * SRA_TICK_VALUE_USD * max(1, len(idea.legs)))


def compute_what_if_table(idea: TradeIdea, panel: dict) -> tuple:
    fit = panel.get("pca_fit_static")
    rows = []
    for pc in (1, 2, 3):
        for sig in (-1, 1):
            label = f"PC{pc} {'+' if sig > 0 else '-'}1σ"
            pnl = what_if_pc_shock(idea, fit, pc, sig)
            rows.append((label, pnl))
    rows.append(("FOMC +25bp surprise", what_if_event(idea, "FOMC", 1.0, panel)))
    rows.append(("FOMC -25bp surprise", -what_if_event(idea, "FOMC", 1.0, panel)))
    rows.append(("CPI +1σ surprise", what_if_event(idea, "Inflation", 1.0, panel)))
    rows.append(("NFP +1σ surprise", what_if_event(idea, "Employment", 1.0, panel)))
    return tuple(rows)


# =============================================================================
# Master orchestrator — split into TRADE vs ENGINE_STATE registries (gameplan
# Phase A.2). TRADE_GENERATORS produce executable legs. ENGINE_STATE_GENERATORS
# emit informational overlays (regime/orthogonality/seasonality). The trade
# screener home table shows only TRADE_GENERATORS output. ENGINE_STATE output
# goes into a separate "Engine Health" panel + still bumps trade conviction
# via cross-confirmation.
# =============================================================================
TRADE_GENERATORS = [
    gen_pc3_fly_ideas, gen_pc2_spread_ideas, gen_pc1_basket_ideas,        # 1-3 PCA-isolated
    gen_anchor_ideas,                                                      # 4 anchor
    gen_front_pca_ideas, gen_belly_pca_ideas, gen_back_pca_ideas,         # 10-12 segment
    gen_outright_fade_ideas, gen_spread_fade_ideas,                       # 13-14 residual
    gen_fly_arb_ideas, gen_pack_ideas,                                    # 15-16
    gen_analog_fv_ideas, gen_path_fv_ideas,                               # 17-18 conditional FV
    gen_event_drift_ideas, gen_event_impact_ranking_ideas,                # 24, 38 event
    gen_pack_fly_ideas, gen_bundle_rv_ideas,                              # 26-27 pack RV
    gen_outlier_reversal_ideas, gen_carry_fly_ideas,                      # 30-31
]

ENGINE_STATE_GENERATORS = [
    gen_cross_pc_corr_breakdown_ideas,                                    # 5 orthogonality
    gen_sparse_dense_divergence_ideas, gen_variance_regime_ideas,         # 6-7
    gen_eigenspectrum_gap_ideas, gen_pc1_asymmetry_ideas,                 # 8-9
    gen_gmm_regime_ideas, gen_posterior_degradation_ideas,                # 19-20
    gen_bocpd_ideas, gen_bai_perron_ideas,                                # 21-22 break-detect
    gen_seasonality_ideas, gen_days_since_fomc_ideas,                     # 23, 25
    gen_pc_momentum_ideas, gen_cycle_phase_ideas,                         # 28-29
]

# Backwards-compat — older code still references ALL_GENERATORS.
ALL_GENERATORS = TRADE_GENERATORS + ENGINE_STATE_GENERATORS


def _finalize_narrative(idea: TradeIdea) -> TradeIdea:
    """Post-process pass — fill any missing rationale_html or headline_html with
    a sensible default so no idea ever leaves the engine without narrative.
    """
    rationale = idea.rationale_html
    headline = idea.headline_html

    needs_rationale = not rationale or len(rationale.strip()) < 20
    needs_headline = not headline or len(headline.strip()) < 5

    # Build defaults from the existing fields if needed.
    if needs_rationale:
        z_str = (f"{idea.z_score:+.2f}σ" if idea.z_score is not None
                  and np.isfinite(idea.z_score) else "—")
        entry_str = (f"{idea.entry_bp:+.2f} bp" if idea.entry_bp is not None
                       and np.isfinite(idea.entry_bp) else "—")
        target_str = (f"{idea.target_bp:+.2f} bp" if idea.target_bp is not None
                        and np.isfinite(idea.target_bp) else "—")
        hl_str = _hl_zone_label(idea.half_life_d)
        gate_str = _gate_label(idea.gate_quality)
        legs_str = _fmt_legs_inline(idea.legs)
        side_word = _direction_word(idea.direction, idea.structure_type)
        rationale = _build_rationale(
            what=f"<b>{idea.primary_source}</b> · z = {z_str} · gate {gate_str}",
            why=f"Engine source: {idea.primary_source} · structure: {idea.structure_type}",
            trade=f"{side_word}: {legs_str} · entry {entry_str} · target {target_str}",
            resolves=f"OU half-life {hl_str}; events not analyzed for this source",
        )

    if needs_headline:
        z_str = (f"{idea.z_score:+.2f}σ" if idea.z_score is not None
                  and np.isfinite(idea.z_score) else "—")
        gate_str = _gate_label(idea.gate_quality)
        first_leg = idea.legs[0].symbol if idea.legs else idea.primary_source
        headline = f"<b>{first_leg} {idea.structure_type} {idea.direction}</b> · {z_str} · {gate_str}"

    if not needs_rationale and not needs_headline:
        return idea

    new_rationale = rationale if needs_rationale else idea.rationale_html
    new_headline = headline if needs_headline else idea.headline_html
    return replace(idea, rationale_html=new_rationale, headline_html=new_headline)


def generate_all_trade_ideas(panel: dict, asof: date,
                                *,
                                positions: Optional[pd.DataFrame] = None,
                                filters: Optional[dict] = None) -> list:
    """Master orchestrator. Runs TRADE_GENERATORS for the executable trade feed,
    ENGINE_STATE_GENERATORS into `panel["engine_state_signals"]`, then clusters,
    scores, lifecycle-tags, filters, sorts the trade feed.

    Engine-state signals still feed cross-confirmation (so a regime-stable
    overlay still bumps a trade's conviction), but they don't appear AS rows
    in the trade table.
    """
    panel.setdefault("asof", asof)

    # 1a. Engine-state generators → separate panel field
    engine_state_signals = []
    for gen in ENGINE_STATE_GENERATORS:
        try:
            engine_state_signals.extend(gen(panel) or [])
        except Exception as e:
            import sys
            print(f"WARN: engine-state generator {gen.__name__} failed: {e}", file=sys.stderr)
            continue
    # Finalize narrative on engine-state signals too (they appear in Engine Health panel)
    engine_state_signals = [_finalize_narrative(i) for i in engine_state_signals]
    panel["engine_state_signals"] = engine_state_signals

    # 1b. Trade generators → trade feed
    raw_ideas = []
    for gen in TRADE_GENERATORS:
        try:
            raw_ideas.extend(gen(panel) or [])
        except Exception as e:
            import sys
            print(f"WARN: trade generator {gen.__name__} failed: {e}", file=sys.stderr)
            continue

    # Engine-state signals also flow into cross-confirmation (they bump conviction
    # of trades whose direction agrees with the engine-state regime). We append
    # them to raw_ideas so the cluster/merge step picks them up, then strip after.
    engine_fingerprints = {s.leg_fingerprint for s in engine_state_signals}
    raw_ideas.extend(engine_state_signals)

    # 2. Cross-confirmation clustering
    clusters = cluster_by_fingerprint(raw_ideas)
    merged = [merge_cluster(c) for c in clusters]

    # 3. Lifecycle assignment
    ledger = load_ledger()
    lifecycle_tagged = []
    for idea in merged:
        state = assign_lifecycle_state(idea, ledger)
        # Compute days_alive
        history = ledger[ledger["leg_fingerprint"] == idea.leg_fingerprint] if not ledger.empty else pd.DataFrame()
        days_alive = len(history)
        max_conv = float(history["conviction"].max()) if not history.empty else 0.0
        lifecycle_tagged.append(replace(idea, state=state, days_alive=days_alive,
                                           max_conviction_to_date=max_conv))

    # 4. Apply hard-block filters (print quality, event-in-window, convexity)
    filtered = hard_block_filter(lifecycle_tagged, panel)

    # 5. Compute slippage estimates (TODO: pass volume_panel from data layer)
    with_slippage = []
    for idea in filtered:
        slip = estimate_slippage(idea, None)
        new_idea = replace(idea, slippage_estimate_bp=slip)
        with_slippage.append(new_idea)

    # 6. Compute factor exposure + what-if per idea
    fit = panel.get("pca_fit_static")
    enriched = []
    for idea in with_slippage:
        if fit is not None and idea.legs:
            fe = compute_factor_exposure(idea, fit)
            wt = compute_what_if_table(idea, panel)
            new_idea = replace(idea,
                                 factor_exposure=tuple(fe.items()),
                                 what_if_table=wt)
        else:
            new_idea = idea
        enriched.append(new_idea)

    # 7. Score conviction (after lifecycle + cross-confirm + risk)
    scored = []
    for idea in enriched:
        conv, breakdown = score_conviction(idea, panel, ledger)
        scored.append(replace(idea, conviction=conv, conviction_breakdown=breakdown))

    # 8. Apply user filters
    if filters:
        scored = _apply_user_filters(scored, filters)

    # 8b. Strip engine-state signals from the trade feed (they live in panel["engine_state_signals"])
    scored = [i for i in scored if i.leg_fingerprint not in engine_fingerprints]

    # 8c. Finalize narrative — fill any remaining blank rationales/headlines
    scored = [_finalize_narrative(i) for i in scored]

    # 9. Sort by conviction descending
    scored.sort(key=lambda i: -i.conviction)

    # 10. Update ledger (persist today's emissions for tomorrow's lifecycle calc)
    try:
        update_ledger(asof, scored)
    except Exception:
        pass

    return scored


def _apply_user_filters(ideas: list, filters: dict) -> list:
    """Apply user filter dict {min_conviction, sources, directions, gates}."""
    out = ideas
    if "min_conviction" in filters:
        thresh = float(filters["min_conviction"])
        out = [i for i in out if i.conviction >= thresh]
    if "sources" in filters and filters["sources"]:
        allowed = set(filters["sources"])
        out = [i for i in out if i.primary_source in allowed]
    if "directions" in filters and filters["directions"]:
        allowed = set(filters["directions"])
        out = [i for i in out if i.direction in allowed]
    if "gates" in filters and filters["gates"]:
        allowed = set(filters["gates"])
        out = [i for i in out if i.gate_quality in allowed]
    if "lifecycle" in filters and filters["lifecycle"]:
        allowed = set(filters["lifecycle"])
        out = [i for i in out if i.state in allowed]
    return out


def generate_execution_ticket(idea: TradeIdea,
                                position_size_dollar: float = 50000) -> dict:
    """Generate broker-format execution ticket for an idea.

    Returns dict with leg-formatted contracts and a CSV-ready string.
    """
    out = {"idea_id": idea.idea_id, "primary_source": idea.primary_source,
           "direction": idea.direction, "structure": idea.structure_type,
           "legs": [], "estimated_margin_dollar": idea.margin_estimate_dollar,
           "csv_text": ""}
    if not idea.legs:
        return out
    # Compute contracts per leg from total $ size
    total_pv01 = sum(abs(leg.weight_pv01) for leg in idea.legs)
    if total_pv01 == 0:
        return out
    pv01_per_dollar = position_size_dollar / (total_pv01 * SRA_TICK_VALUE_USD)
    csv_lines = ["symbol,side,contracts,weight_pv01"]
    for leg in idea.legs:
        contracts = max(1, int(round(abs(leg.weight_pv01) * pv01_per_dollar)))
        side = leg.side
        out["legs"].append({
            "symbol": leg.symbol, "side": side,
            "contracts": contracts, "weight_pv01": leg.weight_pv01,
        })
        csv_lines.append(f"{leg.symbol},{side},{contracts},{leg.weight_pv01:.4f}")
    out["csv_text"] = "\n".join(csv_lines)
    return out
