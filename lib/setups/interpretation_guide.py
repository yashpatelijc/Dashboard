"""Plain-English interpretation guide for the Technicals subtab.

Mirrors the structure of ``lib.mean_reversion.get_interpretation_guide``
so ``lib.mean_reversion.metric_tooltip()`` can be reused on every cell —
producing a multi-line tooltip with formula + threshold table + ✓ marker
on the matching bucket.

The guide is built from ``lib.setups.registry.SETUP_REGISTRY`` so any change
to a setup's logic / buckets propagates automatically.
"""
from __future__ import annotations

from lib.setups.registry import SETUP_REGISTRY, get_registry_entry


def _bucket_meaning(label: str, default: str) -> str:
    """Empty-meaning fallback; UI shows the label by itself."""
    return default if default else "—"


def get_technicals_interpretation_guide() -> dict:
    """Return interpretation guide keyed by setup ID. Each value:

        {
          "what":    one-line description,
          "formula": logic_long (or formula_text),
          "buckets": list of (condition_str, label, meaning_str),
        }
    """
    out = {}
    for sid, entry in SETUP_REGISTRY.items():
        if not entry:
            continue
        long_desc = (
            f"{entry.get('name', sid)} — family {entry.get('family')} · scopes {entry.get('scopes')}"
        )
        formula_lines = []
        if entry.get("logic_long"):
            formula_lines.append(f"LONG:  {entry['logic_long']}")
        if entry.get("logic_short"):
            formula_lines.append(f"SHORT: {entry['logic_short']}")
        if entry.get("stop_rule") and entry["stop_rule"] != "—":
            formula_lines.append(f"Stop:  {entry['stop_rule']}")
        t1 = entry.get("t1"); t2 = entry.get("t2")
        if t1 and t1 != "—":
            tts = f"T1: {t1}"
            if t2 and t2 != "—":
                tts += f"  ·  T2: {t2}"
            formula_lines.append(tts)
        if entry.get("time_stop_bars") is not None:
            formula_lines.append(f"Time stop: {entry['time_stop_bars']} bars")
        if entry.get("trail") and entry["trail"] not in (None, "none"):
            formula_lines.append(f"Trail: {entry['trail']}")
        if entry.get("partial") and entry["partial"] not in (None, "none"):
            formula_lines.append(f"Partial: {entry['partial']}")
        if entry.get("note"):
            formula_lines.append(f"Note: {entry['note']}")
        formula = "\n".join(formula_lines) if formula_lines else ""
        buckets = [(cond, label, _bucket_meaning(label, meaning))
                   for cond, label, meaning in entry.get("buckets", [])]
        out[sid] = {
            "what": long_desc,
            "formula": formula,
            "buckets": buckets,
        }
    return out


# =============================================================================
# Composite-specific guides used by the gauge view
# =============================================================================
def get_composite_guide_outright() -> dict:
    """Interpretation guide entries for the OUTRIGHT composites.

    Each composite entry mirrors the registry buckets but adds the per-scope
    factor list — the trader sees exactly which factors went into the score.
    """
    return {
        "TREND_COMPOSITE · OUTRIGHT": {
            "what": "Mean of 8 factors clipped to [-1, +1]. Positive = bullish trend posture.",
            "formula": (
                "f1 = clip((ADX_14 / 25) × sign(DI+ − DI−), -1, +1)\n"
                "f2 = (count of EMA stack pairs 5>10>20>30>50>100) / 5\n"
                "f3 = clip((close − EMA_100) / ATR_14, -3, +3) / 3\n"
                "f4 = clip(rolling 252d z of (close.shift(21)/close.shift(252) − 1), -2, +2) / 2\n"
                "f5 = clip(MACD_hist / ATR_14, -2, +2) / 2\n"
                "f6 = clip(2 × (close − (High_55+Low_55)/2) / (High_55−Low_55), -1, +1)\n"
                "f7 = AroonOsc_14 / 100\n"
                "f8 = sign(carry_per_day_bp) × min(|carry|/1.0, 1.0)   (rate-aware tilt)\n"
                "TREND = clip(mean(f1..f8), -1, +1)"
            ),
            "buckets": SETUP_REGISTRY["TREND_COMPOSITE"]["buckets"],
        },
        "MR_COMPOSITE · OUTRIGHT": {
            "what": "Mean of 6 factors clipped to [-1, +1]. Positive = stretched LOW (fade up).",
            "formula": (
                "g1 = clip(-ZPRICE_20 / 2, -1, +1)\n"
                "g2 = clip(-ZPRICE_50 / 2, -1, +1)\n"
                "g3 = clip(-(RSI_14 − 50) / 50, -1, +1)\n"
                "g4 = clip(-(BB_pctB_20_2 − 0.5) × 2, -1, +1)\n"
                "g5 = clip(-ZRET_5 / 2.5, -1, +1)\n"
                "g6 = clip(-(close − KC_MID_20) / (KC_UP_20_2 − KC_MID_20), -1, +1)\n"
                "MR = clip(mean(g1..g6), -1, +1)\n"
                "(IBS / CCI / range·close-pos factors dropped — too noisy on STIR outrights)"
            ),
            "buckets": SETUP_REGISTRY["MR_COMPOSITE"]["buckets"],
        },
        "FINAL_COMPOSITE · OUTRIGHT": {
            "what": "Regime-weighted blend with carry tilt. Sign convention: + = bullish.",
            "formula": (
                "w_trend       = clip(ADX_14 / 25, 0, 1)\n"
                "w_mr          = 1 − w_trend\n"
                "hurst_factor  = 1.0 if H<0.45 else 0.5 if H<0.55 else 0.0\n"
                "carry_tilt    = sign(carry) × min(|carry|/1.0, 1.0) × 0.10\n"
                "FINAL = clip(w_trend × TREND − w_mr × hurst_factor × MR + carry_tilt, -1, +1)"
            ),
            "buckets": SETUP_REGISTRY["FINAL_COMPOSITE"]["buckets"],
        },
    }


def get_composite_guide_spread() -> dict:
    return {
        "TREND_COMPOSITE · SPREAD": {
            "what": ("Mean of 5 factors. Spreads rarely trend; this composite is computed for "
                     "transparency and capped at ≤ 40% weight in FINAL."),
            "formula": (
                "f1 = clip((ADX_14 / 22) × sign(DI+ − DI−), -1, +1)   (denom 22 for spreads)\n"
                "f2 = (count of EMA_5>20, 20>50, 50>100) / 3\n"
                "f3 = clip(((close − EMA_50) × bp_mult) / typical_daily_bp, -3, +3) / 3\n"
                "f4 = clip(linear_slope_5d × bp_mult / typical_daily_bp, -3, +3) / 3\n"
                "f5 = Donchian-30 position (smaller window than outrights)\n"
                "TREND = clip(mean(f1..f5), -1, +1)"
            ),
            "buckets": SETUP_REGISTRY["TREND_COMPOSITE"]["buckets"],
        },
        "MR_COMPOSITE · SPREAD": {
            "what": "Mean of 7 factors. Dominant composite for spreads.",
            "formula": (
                "g1 = clip(-Z(close, 60d) / 2, -1, +1)\n"
                "g2 = clip(-Z(close, 30d) / 2, -1, +1)\n"
                "g3 = clip(-Z(close, 15d) / 2.5, -1, +1)\n"
                "g4 = clip(-(BB_pctB_20_2 − 0.5) × 2, -1, +1)\n"
                "g5 = clip(-ZRET_5 / 2.5, -1, +1)\n"
                "g6 = clip(-2 × (percentile_rank(close, 90d)/100 − 0.5), -1, +1)\n"
                "g7 = OU half-life factor: +1 if HL<30, 0 if 30-60, -0.3 if HL>60 / no fit\n"
                "MR = clip(mean(g1..g7), -1, +1)"
            ),
            "buckets": SETUP_REGISTRY["MR_COMPOSITE"]["buckets"],
        },
        "FINAL_COMPOSITE · SPREAD": {
            "what": "MR-tilted regime-weighted blend with ADF + Hurst quality gates.",
            "formula": (
                "w_trend       = clip(ADX_14 / 30, 0, 0.4)              # capped at 40%\n"
                "w_mr          = 1 − w_trend                              # ≥ 60%\n"
                "adf_factor    = 1.0 if ADF rejects p<0.05 else 0.6\n"
                "hurst_factor  = 1.0 if H<0.45 else 0.6 if H<0.55 else 0.3\n"
                "FINAL = clip(w_trend × TREND − w_mr × adf_factor × hurst_factor × MR, -1, +1)"
            ),
            "buckets": SETUP_REGISTRY["FINAL_COMPOSITE"]["buckets"],
        },
    }


def get_composite_guide_fly() -> dict:
    return {
        "TREND_COMPOSITE · FLY": {
            "what": ("Mean of 3 factors. Flies are essentially never trending; this composite "
                     "is computed for transparency and weighted only 10% in FINAL."),
            "formula": (
                "f1 = clip(linear_slope_5d × bp_mult / typical_daily_bp, -3, +3) / 3\n"
                "f2 = clip((close − mean_60d) / std_60d, -3, +3) / 3\n"
                "f3 = clip((ADX_14 / 18) × sign(DI+ − DI−), -1, +1)\n"
                "TREND = clip(mean(f1..f3), -1, +1)"
            ),
            "buckets": SETUP_REGISTRY["TREND_COMPOSITE"]["buckets"],
        },
        "MR_COMPOSITE · FLY": {
            "what": "Mean of 7 factors. Dominant composite for flies — tighter thresholds.",
            "formula": (
                "g1 = clip(-Z(close, 60d) / 2, -1, +1)\n"
                "g2 = clip(-Z(close, 30d) / 2, -1, +1)\n"
                "g3 = clip(-Z(close, 15d) / 2.5, -1, +1)\n"
                "g4 = clip(-(BB_pctB_20_2 − 0.5) × 2, -1, +1)\n"
                "g5 = clip(-2 × (percentile_rank(close, 90d)/100 − 0.5), -1, +1)\n"
                "g6 = OU multiplier: +1 if HL<15, +0.7 if 15-30, +0.4 if 30-60, 0 if >60\n"
                "g7 = clip(touch_count/lookback, 0, 1) × sign(z_60d)\n"
                "MR = clip(mean(g1..g7), -1, +1)"
            ),
            "buckets": SETUP_REGISTRY["MR_COMPOSITE"]["buckets"],
        },
        "FINAL_COMPOSITE · FLY": {
            "what": "Extreme MR-weighted (10/90) with ADF + Hurst + OU triple-gate.",
            "formula": (
                "w_trend       = 0.10                                    # near-zero by design\n"
                "w_mr          = 0.90\n"
                "adf_factor    = 1.0 if ADF rejects p<0.05 else 0.4\n"
                "hurst_factor  = 1.0 if H<0.35 else 0.7 if H<0.45 else 0.4 if H<0.55 else 0.2\n"
                "ou_factor     = 1.0 if HL<15 else 0.8 if HL<30 else 0.5 if HL<60 else 0.2\n"
                "FINAL = clip(w_trend × TREND − w_mr × adf_factor × hurst_factor × ou_factor × MR, -1, +1)"
            ),
            "buckets": SETUP_REGISTRY["FINAL_COMPOSITE"]["buckets"],
        },
    }


def get_full_guide_for_scope(scope: str) -> dict:
    """Return guide combining setup-detector entries + scope-specific composite entries."""
    base = get_technicals_interpretation_guide()
    if scope == "outright":
        base.update(get_composite_guide_outright())
    elif scope == "spread":
        base.update(get_composite_guide_spread())
    elif scope == "fly":
        base.update(get_composite_guide_fly())
    return base
