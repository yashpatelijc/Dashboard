"""NEAR / APPROACHING / FAR threshold dispatcher (Phase B).

Per-category methodology (research-backed, see HANDOFF §5F or the plan file
for full citations). Each technical setup falls into one of nine categories;
this module:

  1. Maps each setup_id → category (``CATEGORY_MAP``).
  2. Provides ``near_state(setup_id, key_inputs, context, calibrated)``
     that returns a ``(state, distance_to_fire, missing_condition_text,
     rule_text, citation_key)`` tuple. The state is one of
     ``"FIRED" | "NEAR" | "APPROACHING" | "FAR" | "N/A"``.
  3. Exposes ``METHODOLOGY_TABLE`` (the literature-derived constants per
     category) so tooltips can show the trader exactly which rule was
     applied.

Design note: detectors typically already compute their own FIRED gate
(it's what makes them detectors). What this layer adds is a UNIFIED
NEAR / APPROACHING grader keyed on the setup category. Detectors can
optionally call ``near_state(...)`` to set ``res.state`` consistently;
existing detectors that compute state inline keep working unchanged
(this layer is additive, not destructive).

For categories where literature defines no canonical threshold (multi-
condition n-of-m and slope/trend), an empirical bootstrap-calibrated
threshold can be loaded via :func:`load_calibrated_thresholds`. If no
calibration exists, the literature default applies.
"""
from __future__ import annotations

from typing import Optional

from lib.setups._NEAR_CITATIONS import cite


# =============================================================================
# Category constants — research-backed from the plan's §5 table
# =============================================================================

CAT_DISTANCE_TO_LEVEL = "a"   # ATR-fraction proximity (Donchian, BB touch, S-R)
CAT_OSCILLATOR_NEUTRAL = "b"  # RSI/Stochastic neutral regime (Wilder)
CAT_OSCILLATOR_BULL = "b'"    # RSI bull-market range (Brown)
CAT_CONNORS_RSI = "b''"       # Connors RSI (10/90 default)
CAT_ZSCORE = "c"              # Stat-arb |z| (Avellaneda & Lee)
CAT_MULTI_CONDITION = "d"     # n-of-m gates (DeMark TD analogue + empirical)
CAT_MA_CROSSOVER = "e"        # EMA / DI crossovers (ATR-normalised)
CAT_SLOPE_TREND = "f"         # Slope thresholds (empirical bootstrap)
CAT_VOL_SQUEEZE = "g"         # BBWP / TTM squeeze (Bollinger / Carter)
CAT_COMPOSITE_THRESHOLD = "h" # Trend / MR / Final composite over threshold (Carver)


# =============================================================================
# METHODOLOGY_TABLE — the constants per category
# =============================================================================
# Each row: (NEAR rule text, APPROACHING rule text, citation key, near_param,
#            approach_param). Param semantics depend on category — see
#            ``_apply_<category>(...)`` for how each is consumed.
METHODOLOGY_TABLE: dict = {
    CAT_DISTANCE_TO_LEVEL: {
        "near_text":       "(level − close) / ATR_14 ∈ (0, 0.5]",
        "approach_text":   "(level − close) / ATR_14 ∈ (0.5, 1.5]",
        "citation":        "wilder_atr",
        "near_atr_frac":   0.5,
        "approach_atr_frac": 1.5,
    },
    CAT_OSCILLATOR_NEUTRAL: {
        "near_text":       "RSI 30-40 (long) / 60-70 (short)",
        "approach_text":   "RSI 40-50 (long) / 50-60 (short)",
        "citation":        "wilder_rsi",
        "near_low":        30.0, "near_high": 40.0,
        "approach_low":    40.0, "approach_high": 50.0,
        "trigger_default": 30.0,
    },
    CAT_OSCILLATOR_BULL: {
        "near_text":       "RSI 40-50 (pullback long; bull regime)",
        "approach_text":   "RSI 50-55",
        "citation":        "brown_rsi",
        "near_low":        40.0, "near_high": 50.0,
        "approach_low":    50.0, "approach_high": 55.0,
        "trigger_default": 40.0,
    },
    CAT_CONNORS_RSI: {
        "near_text":       "ConnorsRSI 10-20 (long) / 80-90 (short)",
        "approach_text":   "CRSI 20-30 (long) / 70-80 (short)",
        "citation":        "connors_crsi",
        "near_low":        10.0, "near_high": 20.0,
        "approach_low":    20.0, "approach_high": 30.0,
        "trigger_default": 10.0,
    },
    CAT_ZSCORE: {
        "near_text":       "|z| ∈ [1.5, 2.0)",
        "approach_text":   "|z| ∈ [1.0, 1.5)",
        "citation":        "avellaneda_lee",
        "near_z_low":      1.5, "near_z_high": 2.0,
        "approach_z_low":  1.0, "approach_z_high": 1.5,
        "trigger_z":       2.0,
    },
    CAT_MULTI_CONDITION: {
        "near_text":       "(n−1) of n conditions met AND missing condition's "
                            "gap < 25% of its trigger distance",
        "approach_text":   "(n−2) of n conditions met (only if ≥ ⌈n/2⌉)",
        "citation":        "demark_td",
        "near_min_met_frac": 0.0,    # set per-setup based on n; computed dynamically
        "approach_min_met_frac": 0.0,
    },
    CAT_MA_CROSSOVER: {
        "near_text":       "|EMA_fast − EMA_slow| / ATR_14 ≤ 0.5  AND  spread closing",
        "approach_text":   "… ≤ 1.5  AND  spread closing",
        "citation":        "wilder_atr",
        "near_atr_frac":   0.5,
        "approach_atr_frac": 1.5,
    },
    CAT_SLOPE_TREND: {
        "near_text":       "slope ≥ 60% of trigger AND slope rising",
        "approach_text":   "slope ≥ 30% of trigger",
        "citation":        "empirical_bootstrap",
        "near_frac":       0.60,
        "approach_frac":   0.30,
    },
    CAT_VOL_SQUEEZE: {
        "near_text":       "BBWP percentile ≤ 25 (rising from < 15) "
                            "OR BB barely outside Keltner",
        "approach_text":   "BBWP percentile ≤ 35",
        "citation":        "bollinger_bandwidth",
        "near_pctile":     25.0,
        "approach_pctile": 35.0,
    },
    CAT_COMPOSITE_THRESHOLD: {
        "near_text":       "composite within 0.1 of threshold AND moving toward it",
        "approach_text":   "composite within 0.2 of threshold",
        "citation":        "carver_continuous",
        "near_band":       0.1,
        "approach_band":   0.2,
    },
}


# =============================================================================
# CATEGORY_MAP — every detector setup_id → category
# =============================================================================
# Each setup_id maps to its primary NEAR-threshold category. Setups whose
# inline detector logic already produces NEAR / APPROACHING via more
# specialised math are listed too, so tooltips can report the methodology
# they're using even if `near_state(...)` isn't called.
#
# For C9a / C9b, the suffixed variants (e.g. C9a_12M) inherit from the
# base id via the ``_strip_variant`` helper.
CATEGORY_MAP: dict[str, str] = {
    # TREND setups
    "A1":   CAT_MULTI_CONDITION,    # 4-condition breakout
    "A2":   CAT_MULTI_CONDITION,    # 3-condition pullback
    "A3":   CAT_VOL_SQUEEZE,        # VCP — BB-bandwidth percentile gate
    "A4":   CAT_VOL_SQUEEZE,        # squeeze fire (BB inside Keltner)
    "A5":   CAT_DISTANCE_TO_LEVEL,  # 55d Donchian breakout
    "A6":   CAT_MA_CROSSOVER,       # Golden / Death cross
    "A8":   CAT_MULTI_CONDITION,    # Ichimoku cloud breakout (3 conditions)
    "A10":  CAT_MULTI_CONDITION,    # MA ribbon alignment (5+ conditions)
    "A11":  CAT_MULTI_CONDITION,    # Supertrend directional + ADX
    "A12a": CAT_MULTI_CONDITION,    # ADX/DI cross + EMA_20
    "A12b": CAT_MULTI_CONDITION,    # ADX/DI cross + EMA_50
    "A15":  CAT_DISTANCE_TO_LEVEL,  # Triangle/wedge — distance to apex

    # MR setups
    "B1":   CAT_MULTI_CONDITION,    # BB 2σ + ADX + RSI divergence
    "B3":   CAT_MULTI_CONDITION,    # RSI divergence + 20d low + reversal candle
    "B5":   CAT_ZSCORE,             # ZPRICE_20 < -2 (and ZPRICE_50 secondary)
    "B6":   CAT_ZSCORE,             # ZRET_5 < -2.5
    "B10":  CAT_MULTI_CONDITION,    # Volume climax fade (4 conditions)
    "B11":  CAT_ZSCORE,             # OU half-life < 30 + z-score < -2
    "B13":  CAT_ZSCORE,             # Hurst < 0.45 + ZPRICE_20 < -2

    # STIR-specific
    "C3":   CAT_DISTANCE_TO_LEVEL,  # Carry-rank intra-curve (top decile)
    "C4":   CAT_ZSCORE,             # Traditional fly MR (|z| > 2)
    "C5":   CAT_ZSCORE,             # Calendar spread MR (|z| > 2)
    "C8":   CAT_DISTANCE_TO_LEVEL,  # Terminal rate ≥ 25bp move over 5d
    "C9a":  CAT_DISTANCE_TO_LEVEL,  # Slope crossing (distance to zero)
    "C9b":  CAT_SLOPE_TREND,        # 5d slope trend ≥ 5bp

    # Composites
    "TREND_COMPOSITE":  CAT_COMPOSITE_THRESHOLD,
    "MR_COMPOSITE":     CAT_COMPOSITE_THRESHOLD,
    "FINAL_COMPOSITE":  CAT_COMPOSITE_THRESHOLD,
}


def _strip_variant(setup_id: str) -> str:
    """Strip the trailing offset variant for C9a_12M / C9b_24M / C8_12M
    style ids → returns the base id (C9a / C9b / C8)."""
    if not isinstance(setup_id, str):
        return setup_id
    for prefix in ("C9a_", "C9b_", "C8_"):
        if setup_id.startswith(prefix):
            return prefix.rstrip("_")
    return setup_id


def category_for(setup_id: str) -> Optional[str]:
    """Return the NEAR-threshold category for a setup id, or None if unmapped.

    Handles offset-suffixed variants (C9a_12M / C9b_24M / C8_12M) by
    stripping to the base id."""
    base = _strip_variant(setup_id)
    return CATEGORY_MAP.get(base)


def methodology_for(setup_id: str) -> dict:
    """Return the full methodology row for a setup_id (NEAR text, APPROACHING
    text, citation, params). Empty dict if the setup has no mapped category."""
    cat = category_for(setup_id)
    if cat is None:
        return {}
    row = dict(METHODOLOGY_TABLE.get(cat, {}))
    row["category"] = cat
    return row


# =============================================================================
# near_state dispatcher — single entry point
# =============================================================================

def near_state(setup_id: str,
                 key_inputs: dict,
                 context: Optional[dict] = None,
                 calibrated: Optional[dict] = None) -> tuple:
    """Classify a setup as FIRED / NEAR / APPROACHING / FAR / N/A.

    This is the canonical proximity grader used by the watchlist UI and the
    Phase G tooltips. Each detector that implements its own FIRED gate can
    optionally call this to set NEAR / APPROACHING consistently.

    Parameters
    ----------
    setup_id : str
        The setup id (e.g. ``"A1"``, ``"B5"``, ``"C9b_12M"``).
    key_inputs : dict
        The detector's already-computed input values. Expected keys depend
        on category:

          - DISTANCE_TO_LEVEL: ``"close"``, ``"trigger_level"``, ``"atr"``
          - OSCILLATOR_*: ``"rsi"``
          - ZSCORE:       ``"z"``
          - MULTI_CONDITION: ``"n_met"``, ``"n_total"``,
                             ``"missing_gap_frac"`` (optional, per-cond)
          - MA_CROSSOVER: ``"fast"``, ``"slow"``, ``"atr"``
          - SLOPE_TREND:  ``"slope"``, ``"trigger"``, ``"is_rising"``
          - VOL_SQUEEZE:  ``"bbwp_pctile"``
          - COMPOSITE:    ``"value"``, ``"threshold"``
    context : dict, optional
        Free-form per-call context (e.g. regime). Currently unused but
        reserved for the calibrated-threshold loader.
    calibrated : dict, optional
        Per-(setup_id, direction) override of the literature thresholds.
        Loaded by Phase E from ``near_thresholds_calibrated.parquet``.

    Returns
    -------
    tuple
        ``(state, distance_to_fire, missing_condition_text,
            rule_text, citation_key)`` —
        * state: ``"FIRED" | "NEAR" | "APPROACHING" | "FAR" | "N/A"``.
        * distance_to_fire: float in [0, +inf]. 0 = fired; smaller = closer.
        * missing_condition_text: human-readable description of what's
          missing for FIRED (used by tooltips).
        * rule_text: the methodology applied (used by tooltips).
        * citation_key: the source for that rule.
    """
    cat = category_for(setup_id)
    if cat is None:
        return ("N/A", float("inf"), f"no NEAR-category mapping for {setup_id}",
                "no methodology", "")

    methodology = METHODOLOGY_TABLE.get(cat, {})
    if calibrated:
        # Calibrated overrides take precedence
        methodology = {**methodology, **calibrated}

    rule_text = methodology.get("near_text", "")
    citation = methodology.get("citation", "")

    # Dispatch to per-category logic
    state, dist, missing = _apply_category(cat, key_inputs, methodology)
    return (state, dist, missing, rule_text, citation)


# ---------- per-category logic ----------

def _apply_category(cat: str, ki: dict, m: dict) -> tuple:
    """Return ``(state, distance_to_fire, missing_text)`` for the given
    category and key_inputs. State per the literature thresholds in ``m``."""
    handlers = {
        CAT_DISTANCE_TO_LEVEL:    _apply_distance,
        CAT_OSCILLATOR_NEUTRAL:   _apply_oscillator_neutral,
        CAT_OSCILLATOR_BULL:      _apply_oscillator_bull,
        CAT_CONNORS_RSI:          _apply_connors_rsi,
        CAT_ZSCORE:               _apply_zscore,
        CAT_MULTI_CONDITION:      _apply_multi_condition,
        CAT_MA_CROSSOVER:         _apply_ma_crossover,
        CAT_SLOPE_TREND:          _apply_slope,
        CAT_VOL_SQUEEZE:          _apply_squeeze,
        CAT_COMPOSITE_THRESHOLD:  _apply_composite,
    }
    handler = handlers.get(cat)
    if handler is None:
        return ("N/A", float("inf"), f"no handler for category {cat}")
    try:
        return handler(ki, m)
    except (KeyError, TypeError, ValueError) as e:
        return ("N/A", float("inf"), f"missing key for {cat}: {e}")


def _apply_distance(ki: dict, m: dict) -> tuple:
    """ATR-fraction distance to a trigger level."""
    close = float(ki["close"])
    trigger = float(ki["trigger_level"])
    atr = float(ki.get("atr") or 0)
    direction = ki.get("direction", "LONG")
    if atr <= 0:
        return ("N/A", float("inf"), "atr unavailable")
    if direction == "LONG":
        # Trigger above close (e.g. breakout above resistance)
        gap_atr = (trigger - close) / atr
    else:
        gap_atr = (close - trigger) / atr
    if gap_atr <= 0:
        return ("FIRED", 0.0, "")
    if gap_atr <= m["near_atr_frac"]:
        return ("NEAR", gap_atr, f"{gap_atr:.2f} ATR away from trigger")
    if gap_atr <= m["approach_atr_frac"]:
        return ("APPROACHING", gap_atr, f"{gap_atr:.2f} ATR away from trigger")
    return ("FAR", gap_atr, f"{gap_atr:.2f} ATR away from trigger")


def _apply_oscillator_neutral(ki: dict, m: dict) -> tuple:
    """RSI-style with Wilder's 30/70 trigger."""
    rsi = float(ki["rsi"])
    direction = ki.get("direction", "LONG")
    if direction == "LONG":
        trigger = m.get("trigger_default", 30.0)
        if rsi <= trigger:
            return ("FIRED", 0.0, "")
        if rsi <= m["near_high"]:
            return ("NEAR", rsi - trigger, f"RSI {rsi:.1f} (trigger {trigger:.0f})")
        if rsi <= m["approach_high"]:
            return ("APPROACHING", rsi - trigger, f"RSI {rsi:.1f}")
        return ("FAR", rsi - trigger, f"RSI {rsi:.1f}")
    else:
        # Mirror for SHORT: trigger at 70, watch zones at 60-70 / 50-60
        trigger = 100 - m.get("trigger_default", 30.0)
        near_low = 100 - m["near_high"]; near_high = 100 - m["near_low"]
        approach_low = 100 - m["approach_high"]; approach_high = 100 - m["approach_low"]
        if rsi >= trigger:
            return ("FIRED", 0.0, "")
        if rsi >= near_low:
            return ("NEAR", trigger - rsi, f"RSI {rsi:.1f} (trigger {trigger:.0f})")
        if rsi >= approach_low:
            return ("APPROACHING", trigger - rsi, f"RSI {rsi:.1f}")
        return ("FAR", trigger - rsi, f"RSI {rsi:.1f}")


def _apply_oscillator_bull(ki: dict, m: dict) -> tuple:
    """Brown's bull-market RSI band (40-50 = pullback support)."""
    rsi = float(ki["rsi"])
    if rsi <= m.get("trigger_default", 40.0):
        return ("FIRED", 0.0, "")
    if rsi <= m["near_high"]:
        return ("NEAR", rsi - m["trigger_default"], f"RSI {rsi:.1f}")
    if rsi <= m["approach_high"]:
        return ("APPROACHING", rsi - m["trigger_default"], f"RSI {rsi:.1f}")
    return ("FAR", rsi - m["trigger_default"], f"RSI {rsi:.1f}")


def _apply_connors_rsi(ki: dict, m: dict) -> tuple:
    """Connors RSI 10/90 trigger."""
    crsi = float(ki["rsi"])   # generic key name
    direction = ki.get("direction", "LONG")
    if direction == "LONG":
        trigger = m.get("trigger_default", 10.0)
        if crsi <= trigger:
            return ("FIRED", 0.0, "")
        if crsi <= m["near_high"]:
            return ("NEAR", crsi - trigger, f"CRSI {crsi:.1f}")
        if crsi <= m["approach_high"]:
            return ("APPROACHING", crsi - trigger, f"CRSI {crsi:.1f}")
        return ("FAR", crsi - trigger, f"CRSI {crsi:.1f}")
    else:
        trigger = 100 - m.get("trigger_default", 10.0)
        near_low = 100 - m["near_high"]
        approach_low = 100 - m["approach_high"]
        if crsi >= trigger:
            return ("FIRED", 0.0, "")
        if crsi >= near_low:
            return ("NEAR", trigger - crsi, f"CRSI {crsi:.1f}")
        if crsi >= approach_low:
            return ("APPROACHING", trigger - crsi, f"CRSI {crsi:.1f}")
        return ("FAR", trigger - crsi, f"CRSI {crsi:.1f}")


def _apply_zscore(ki: dict, m: dict) -> tuple:
    """|z| extremes per Avellaneda & Lee."""
    z = float(ki["z"])
    abs_z = abs(z)
    trigger = m.get("trigger_z", 2.0)
    if abs_z >= trigger:
        return ("FIRED", 0.0, "")
    if abs_z >= m["near_z_low"]:
        return ("NEAR", trigger - abs_z, f"|z|={abs_z:.2f} (trigger {trigger:.1f})")
    if abs_z >= m["approach_z_low"]:
        return ("APPROACHING", trigger - abs_z, f"|z|={abs_z:.2f}")
    return ("FAR", trigger - abs_z, f"|z|={abs_z:.2f}")


def _apply_multi_condition(ki: dict, m: dict) -> tuple:
    """n-of-m gates: (n−1)/n = NEAR; (n−2)/n = APPROACHING."""
    n_met = int(ki["n_met"])
    n_total = int(ki["n_total"])
    if n_total <= 0:
        return ("N/A", float("inf"), "n_total <= 0")
    if n_met >= n_total:
        return ("FIRED", 0.0, "")
    missing_gap = ki.get("missing_gap_frac", None)   # 0..1, fraction of trigger distance
    if n_met == n_total - 1:
        # Optional gating: if the missing condition's gap is < 25% of trigger, it's NEAR
        if missing_gap is None or missing_gap < 0.25:
            return ("NEAR", n_total - n_met,
                    f"{n_met} of {n_total} conditions met")
        return ("APPROACHING", n_total - n_met,
                f"{n_met}/{n_total} met but missing condition gap > 25%")
    if n_met >= (n_total + 1) // 2:
        return ("APPROACHING", n_total - n_met,
                f"{n_met}/{n_total} conditions met")
    return ("FAR", n_total - n_met, f"{n_met}/{n_total} conditions met")


def _apply_ma_crossover(ki: dict, m: dict) -> tuple:
    """ATR-fraction proximity of MA crossover."""
    fast = float(ki["fast"])
    slow = float(ki["slow"])
    atr = float(ki.get("atr") or 0)
    direction = ki.get("direction", "LONG")
    if atr <= 0:
        return ("N/A", float("inf"), "atr unavailable")
    spread = (fast - slow) if direction == "LONG" else (slow - fast)
    spread_atr = spread / atr
    if spread_atr >= 0:
        return ("FIRED", 0.0, "")  # already crossed
    abs_spread_atr = abs(spread_atr)
    if abs_spread_atr <= m["near_atr_frac"]:
        return ("NEAR", abs_spread_atr,
                f"|EMA_fast - EMA_slow| = {abs_spread_atr:.2f} ATR, closing")
    if abs_spread_atr <= m["approach_atr_frac"]:
        return ("APPROACHING", abs_spread_atr,
                f"|EMA_fast - EMA_slow| = {abs_spread_atr:.2f} ATR")
    return ("FAR", abs_spread_atr, f"{abs_spread_atr:.2f} ATR away")


def _apply_slope(ki: dict, m: dict) -> tuple:
    """Slope-trend gate as fraction of trigger."""
    slope = float(ki["slope"])
    trigger = float(ki["trigger"])
    is_rising = bool(ki.get("is_rising", False))
    if trigger == 0:
        return ("N/A", float("inf"), "trigger == 0")
    abs_frac = abs(slope) / abs(trigger)
    if abs_frac >= 1.0:
        return ("FIRED", 0.0, "")
    if abs_frac >= m["near_frac"] and is_rising:
        return ("NEAR", 1.0 - abs_frac, f"slope = {abs_frac:.0%} of trigger, rising")
    if abs_frac >= m["approach_frac"]:
        return ("APPROACHING", 1.0 - abs_frac, f"slope = {abs_frac:.0%} of trigger")
    return ("FAR", 1.0 - abs_frac, f"slope = {abs_frac:.0%} of trigger")


def _apply_squeeze(ki: dict, m: dict) -> tuple:
    """BBWP percentile-based squeeze."""
    bbwp = float(ki["bbwp_pctile"])
    if bbwp <= 15:
        return ("FIRED", 0.0, "")  # in the squeeze
    if bbwp <= m["near_pctile"]:
        return ("NEAR", bbwp - 15, f"BBWP pct = {bbwp:.0f}")
    if bbwp <= m["approach_pctile"]:
        return ("APPROACHING", bbwp - 15, f"BBWP pct = {bbwp:.0f}")
    return ("FAR", bbwp - 15, f"BBWP pct = {bbwp:.0f}")


def _apply_composite(ki: dict, m: dict) -> tuple:
    """Composite-threshold proximity (Carver continuous-forecast)."""
    value = float(ki["value"])
    trigger = float(ki["threshold"])
    abs_v = abs(value)
    abs_t = abs(trigger)
    if abs_v >= abs_t:
        return ("FIRED", 0.0, "")
    gap = abs_t - abs_v
    if gap <= m["near_band"]:
        return ("NEAR", gap, f"|composite|={abs_v:.2f}, threshold {abs_t:.1f}")
    if gap <= m["approach_band"]:
        return ("APPROACHING", gap, f"|composite|={abs_v:.2f}")
    return ("FAR", gap, f"|composite|={abs_v:.2f}")


# =============================================================================
# Calibrated-threshold loader (Phase E plumbing — for now reads parquet
# if present, returns empty dict otherwise)
# =============================================================================

def load_calibrated_thresholds() -> dict:
    """Load per-(setup_id, direction) calibrated thresholds from
    ``data/near_thresholds_calibrated.parquet`` if present.

    Returns ``{setup_id: {param: value}}`` to be merged into METHODOLOGY_TABLE
    rows by ``near_state(...)``.

    Empty dict when no calibration has been run yet — literature defaults
    apply uniformly. Phase E's recompute cycle populates this.
    """
    from pathlib import Path
    import pandas as pd
    path = Path(__file__).resolve().parent.parent.parent / "data" / "near_thresholds_calibrated.parquet"
    if not path.exists():
        return {}
    try:
        df = pd.read_parquet(path)
        return {row["setup_id"]: row.drop("setup_id").to_dict()
                  for _, row in df.iterrows()}
    except Exception:
        return {}
