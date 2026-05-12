"""Composite scoring per scope: TREND_COMPOSITE · MR_COMPOSITE · FINAL_COMPOSITE.

Three flavors of each — OUTRIGHT / SPREAD / FLY — because the statistical
character of these instruments differs (see HANDOFF §5C and the proposal
document for the full reasoning).

Returns a uniform dict for any scope:
    {
      "trend_score": float,
      "mr_score": float,
      "final_score": float,
      "trend_factors": {f1: ..., f2: ...},
      "mr_factors":    {g1: ..., g2: ...},
      "weights":       {w_trend: ..., w_mr: ..., adf_factor: ..., hurst_factor: ..., ...},
      "regime":        "TRENDING_HIGH_VOL" | "RANGING_LOW_VOL" | ...
      "interpretation": str,
      "scope": str,
    }
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from lib.carry import compute_per_contract_carry
from lib.mean_reversion import (
    adf_test as mr_adf_test,
    hurst_exponent as mr_hurst,
    ou_half_life as mr_ou_half_life,
    zscore_value as mr_zscore_value,
)
from lib.runtime_indicators import (
    ema_200,
    linear_slope,
    rolling_percentile,
)
from lib.setups.base import safe_float
from lib.setups.trend import _ind, _get_today_and_history


# =============================================================================
# Universal helpers
# =============================================================================
def _clip(x, lo: float = -1.0, hi: float = +1.0) -> float:
    if x is None or not np.isfinite(x):
        return 0.0
    return float(np.clip(x, lo, hi))


def _typical_daily_bp(close_series: pd.Series, asof_date: date,
                      lookback: int = 20, bp_mult: float = 100.0,
                      floor_bp: float = 0.5) -> float:
    """Median of |Δclose| × bp_mult over the prior ``lookback`` bars."""
    if close_series is None or close_series.empty:
        return floor_bp
    ts = pd.Timestamp(asof_date)
    diffs = close_series.diff().abs().loc[close_series.index < ts].tail(lookback).dropna()
    if diffs.empty:
        return floor_bp
    typ = float(diffs.median()) * bp_mult
    return max(typ, floor_bp)


def _regime_label(adx: Optional[float], atr_pct: Optional[float]) -> str:
    """gameplan §7.5 regime classification."""
    if adx is None:
        return "NEUTRAL"
    if adx >= 25:
        adx_state = "TRENDING"
    elif adx < 20:
        adx_state = "RANGING"
    else:
        adx_state = "NEUTRAL"
    if atr_pct is None:
        vol_state = "NEUTRAL"
    elif atr_pct > 0.75:
        vol_state = "HIGH_VOL"
    elif atr_pct < 0.25:
        vol_state = "LOW_VOL"
    else:
        vol_state = "NEUTRAL"
    if adx_state == "NEUTRAL" or vol_state == "NEUTRAL":
        if adx_state == "NEUTRAL" and vol_state == "NEUTRAL":
            return "NEUTRAL"
        return "NEUTRAL"
    return f"{adx_state}_{vol_state}"


def _final_to_label(final: float) -> str:
    if final >= 0.7:
        return "STRONG SIGNAL ↑"
    if final >= 0.5:
        return "moderate ↑"
    if final >= 0.3:
        return "soft ↑"
    if final <= -0.7:
        return "STRONG SIGNAL ↓"
    if final <= -0.5:
        return "moderate ↓"
    if final <= -0.3:
        return "soft ↓"
    return "neutral · no edge"


# =============================================================================
# OUTRIGHT composites — 8 trend factors / 6 MR factors
# =============================================================================
def compute_outright_composites(panel: pd.DataFrame, asof_date: date,
                                  bp_mult: float = 100.0,
                                  carry_per_day_bp: Optional[float] = None) -> dict:
    """For SRA outrights. Returns the standard composite dict."""
    out = {"trend_score": 0.0, "mr_score": 0.0, "final_score": 0.0,
           "trend_factors": {}, "mr_factors": {}, "weights": {}, "regime": "NEUTRAL",
           "interpretation": "neutral · no edge", "scope": "outright", "error": None}
    try:
        today, history = _get_today_and_history(panel, asof_date, min_history=60)
    except ValueError as e:
        out["error"] = str(e); return out

    close = safe_float(today.get("close"))
    adx = safe_float(_ind(today, "ADX_14"))
    di_p = safe_float(_ind(today, "DIplus_14", ["DIplus_14"]))
    di_m = safe_float(_ind(today, "DIminus_14", ["DIminus_14"]))
    ema100 = safe_float(_ind(today, "EMA_100"))
    atr = safe_float(_ind(today, "ATR14", ["ATR20"]))
    macd_hist = safe_float(_ind(today, "MACD_12_26_9_hist", ["MACD_12_26_9_hist", "MACD_12_26_9_hist"]))
    aroon_osc = safe_float(_ind(today, "AroonOsc_14", ["AROONOSC_14"]))

    # EMA stack pairs
    pairs = []
    for p_a, p_b in [(5, 10), (10, 20), (20, 30), (30, 50), (50, 100)]:
        a = safe_float(_ind(today, f"EMA_{p_a}"))
        b = safe_float(_ind(today, f"EMA_{p_b}"))
        if a is not None and b is not None:
            pairs.append(1 if a > b else -1)
    f2 = (sum(pairs) / max(1, len(pairs))) if pairs else 0.0

    # f1
    f1 = 0.0
    if adx is not None and di_p is not None and di_m is not None:
        f1 = _clip((adx / 25.0) * np.sign(di_p - di_m))

    # f3
    f3 = 0.0
    if close is not None and ema100 is not None and atr is not None and atr > 0:
        f3 = _clip(((close - ema100) / atr) / 3.0)

    # f4: 12-1 momentum z-score (rolling 252d z of (close.shift(21)/close.shift(252) - 1))
    f4 = 0.0
    if "close" in panel.columns and len(panel) > 270:
        ts = pd.Timestamp(asof_date)
        c = panel["close"].loc[panel["close"].index <= ts]
        if len(c) > 270:
            mom = c.shift(21) / c.shift(252) - 1.0
            mom_252 = mom.tail(252).dropna()
            if len(mom_252) > 30:
                today_mom = safe_float(mom.iloc[-1])
                mu = float(mom_252.mean()); sd = float(mom_252.std(ddof=1))
                if today_mom is not None and sd > 0:
                    f4 = _clip(((today_mom - mu) / sd) / 2.0)

    # f5: MACD_hist / ATR
    f5 = 0.0
    if macd_hist is not None and atr is not None and atr > 0:
        f5 = _clip((macd_hist / atr) / 2.0)

    # f6: Donchian-55 position from CLOSE
    f6 = 0.0
    if "close" in panel.columns and len(history) >= 55:
        last55 = history["close"].tail(55)
        h55 = float(last55.max()); l55 = float(last55.min())
        rng = h55 - l55
        if rng > 0 and close is not None:
            f6 = _clip(2.0 * (close - (h55 + l55) / 2.0) / rng)

    # f7: AroonOsc
    f7 = 0.0
    if aroon_osc is not None:
        f7 = _clip(aroon_osc / 100.0)

    # f8: Carry direction (bp/day) — rate-aware tilt
    f8 = 0.0
    if carry_per_day_bp is not None and np.isfinite(carry_per_day_bp):
        f8 = _clip(np.sign(carry_per_day_bp) * min(abs(carry_per_day_bp) / 1.0, 1.0))

    trend_factors = {"f1_ADX_DI": f1, "f2_EMA_stack": f2, "f3_dist_EMA100": f3,
                      "f4_12_1_z": f4, "f5_MACD_ATR": f5, "f6_Donchian_55": f6,
                      "f7_AroonOsc": f7, "f8_carry_dir": f8}
    trend_score = _clip(np.mean(list(trend_factors.values())))

    # =====================================================================
    # MR factors (6)
    # =====================================================================
    z20 = safe_float(_ind(today, "ZPRICE_20"))
    z50 = safe_float(_ind(today, "ZPRICE_50"))
    rsi14 = safe_float(_ind(today, "RSI_14"))
    bb_pctB = safe_float(_ind(today, "BB_pctB_20_2.0x", ["BB_pctB_20_2", "BB_pctB_20"]))
    zret5 = safe_float(_ind(today, "ZRET_5"))
    kc_mid = safe_float(_ind(today, "KC_MID_20", ["BB_MID_20", "SMA_20"]))
    kc_up = safe_float(_ind(today, "KC_UP_20_2.0x_ATR20", ["KC_UP_20_2", "KC_UP_20"]))

    g1 = _clip(-z20 / 2.0) if z20 is not None else 0.0
    g2 = _clip(-z50 / 2.0) if z50 is not None else 0.0
    g3 = _clip(-(rsi14 - 50.0) / 50.0) if rsi14 is not None else 0.0
    g4 = _clip(-(bb_pctB - 0.5) * 2.0) if bb_pctB is not None else 0.0
    g5 = _clip(-zret5 / 2.5) if zret5 is not None else 0.0
    g6 = 0.0
    if close is not None and kc_mid is not None and kc_up is not None and (kc_up - kc_mid) > 0:
        g6 = _clip(-(close - kc_mid) / (kc_up - kc_mid))

    mr_factors = {"g1_-ZPRICE_20/2": g1, "g2_-ZPRICE_50/2": g2,
                   "g3_-(RSI-50)/50": g3, "g4_-(BB_pctB-0.5)x2": g4,
                   "g5_-ZRET_5/2.5": g5, "g6_-KC_pos": g6}
    mr_score = _clip(np.mean(list(mr_factors.values())))

    # FINAL — regime-weighted blend with carry tilt
    w_trend = float(np.clip(adx / 25.0, 0, 1)) if adx is not None else 0.0
    w_mr = 1.0 - w_trend
    h = mr_hurst(panel["close"], asof_date, lookback=90) if "close" in panel.columns else None
    if h is None:
        hurst_factor = 0.5
    elif h < 0.45:
        hurst_factor = 1.0
    elif h < 0.55:
        hurst_factor = 0.5
    else:
        hurst_factor = 0.0

    carry_tilt = 0.0
    if carry_per_day_bp is not None and np.isfinite(carry_per_day_bp):
        carry_tilt = float(np.sign(carry_per_day_bp)
                            * min(abs(carry_per_day_bp) / 1.0, 1.0) * 0.10)

    final = _clip(w_trend * trend_score - w_mr * hurst_factor * mr_score + carry_tilt)

    # Regime label
    atr_full = _ind(panel, "ATR14", ["ATR20"])
    atr_pct = None
    if isinstance(atr_full, pd.Series) and not atr_full.empty:
        rp = rolling_percentile(atr_full, window=252)
        rp_today = safe_float(rp.loc[rp.index <= pd.Timestamp(asof_date)].iloc[-1])
        atr_pct = (rp_today / 100.0) if rp_today is not None else None
    regime = _regime_label(adx, atr_pct)

    out["trend_score"] = trend_score
    out["mr_score"] = mr_score
    out["final_score"] = final
    out["trend_factors"] = trend_factors
    out["mr_factors"] = mr_factors
    out["weights"] = {"w_trend": w_trend, "w_mr": w_mr,
                       "hurst": h, "hurst_factor": hurst_factor,
                       "carry_per_day_bp": carry_per_day_bp,
                       "carry_tilt": carry_tilt,
                       "atr_pct_252d": atr_pct}
    out["regime"] = regime
    out["interpretation"] = _final_to_label(final)
    return out


# =============================================================================
# SPREAD composites — 5 trend / 7 MR (MR-tilted)
# =============================================================================
def compute_spread_composites(panel: pd.DataFrame, asof_date: date,
                                bp_mult: float = 1.0) -> dict:
    """For SRA spreads. ``bp_mult`` is 1.0 because spread closes are stored in bp."""
    out = {"trend_score": 0.0, "mr_score": 0.0, "final_score": 0.0,
           "trend_factors": {}, "mr_factors": {}, "weights": {}, "regime": "NEUTRAL",
           "interpretation": "neutral · no edge", "scope": "spread", "error": None}
    try:
        today, history = _get_today_and_history(panel, asof_date, min_history=60)
    except ValueError as e:
        out["error"] = str(e); return out

    close = safe_float(today.get("close"))
    adx = safe_float(_ind(today, "ADX_14"))
    di_p = safe_float(_ind(today, "DIplus_14", ["DIplus_14"]))
    di_m = safe_float(_ind(today, "DIminus_14", ["DIminus_14"]))
    ema50 = safe_float(_ind(today, "EMA_50"))
    bb_pctB = safe_float(_ind(today, "BB_pctB_20_2.0x", ["BB_pctB_20_2", "BB_pctB_20"]))
    zret5 = safe_float(_ind(today, "ZRET_5"))

    typ_bp = _typical_daily_bp(panel["close"], asof_date, lookback=20, bp_mult=bp_mult, floor_bp=0.1)

    # f1 ADX·DI
    f1 = 0.0
    if adx is not None and di_p is not None and di_m is not None:
        f1 = _clip((adx / 22.0) * np.sign(di_p - di_m))

    # f2 EMA stack 5>20>50>100 (3 pairs)
    pairs = []
    for p_a, p_b in [(5, 20), (20, 50), (50, 100)]:
        a = safe_float(_ind(today, f"EMA_{p_a}"))
        b = safe_float(_ind(today, f"EMA_{p_b}"))
        if a is not None and b is not None:
            pairs.append(1 if a > b else -1)
    f2 = (sum(pairs) / max(1, len(pairs))) if pairs else 0.0

    # f3 distance EMA_50 in bp
    f3 = 0.0
    if close is not None and ema50 is not None and typ_bp > 0:
        f3 = _clip(((close - ema50) * bp_mult / typ_bp) / 3.0)

    # f4 5d linear slope in bp
    f4 = 0.0
    if "close" in panel.columns:
        slope_series = linear_slope(panel["close"], window=5)
        s_today = safe_float(slope_series.loc[slope_series.index <= pd.Timestamp(asof_date)].iloc[-1])
        if s_today is not None and typ_bp > 0:
            f4 = _clip((s_today * bp_mult / typ_bp) / 3.0)

    # f5 Donchian-30 position
    f5 = 0.0
    if len(history) >= 30 and close is not None:
        last30 = history["close"].tail(30)
        h30 = float(last30.max()); l30 = float(last30.min())
        rng = h30 - l30
        if rng > 0:
            f5 = _clip(2.0 * (close - (h30 + l30) / 2.0) / rng)

    trend_factors = {"f1_ADX_DI": f1, "f2_EMA_stack_3": f2,
                      "f3_dist_EMA50_bp": f3, "f4_slope_5d": f4,
                      "f5_Donchian_30": f5}
    trend_score = _clip(np.mean(list(trend_factors.values())))

    # =====================================================================
    # MR factors (7)
    # =====================================================================
    z60 = mr_zscore_value(panel["close"], asof_date, lookback=60)
    z30 = mr_zscore_value(panel["close"], asof_date, lookback=30)
    z15 = mr_zscore_value(panel["close"], asof_date, lookback=15)

    g1 = _clip(-z60 / 2.0) if z60 is not None else 0.0
    g2 = _clip(-z30 / 2.0) if z30 is not None else 0.0
    g3 = _clip(-z15 / 2.5) if z15 is not None else 0.0
    g4 = _clip(-(bb_pctB - 0.5) * 2.0) if bb_pctB is not None else 0.0
    g5 = _clip(-zret5 / 2.5) if zret5 is not None else 0.0

    # g6 percentile_rank inverted
    g6 = 0.0
    rp = rolling_percentile(panel["close"], window=90)
    rp_today = safe_float(rp.loc[rp.index <= pd.Timestamp(asof_date)].iloc[-1])
    if rp_today is not None:
        g6 = _clip(-2.0 * (rp_today / 100.0 - 0.5))

    # g7 OU half-life factor
    hl = mr_ou_half_life(panel["close"], asof_date, lookback=90)
    if hl is not None and np.isfinite(hl) and hl < 30.0:
        g7 = +1.0
    elif hl is not None and np.isfinite(hl) and hl <= 60.0:
        g7 = 0.0
    else:
        g7 = -0.3

    mr_factors = {"g1_-Z(60d)/2": g1, "g2_-Z(30d)/2": g2, "g3_-Z(15d)/2.5": g3,
                   "g4_-(BB_pctB-0.5)x2": g4, "g5_-ZRET_5/2.5": g5,
                   "g6_pct_rank_90d": g6, "g7_OU_HL_factor": g7}
    mr_score = _clip(np.mean(list(mr_factors.values())))

    # FINAL with ADF + Hurst gates
    w_trend = float(np.clip(adx / 30.0, 0, 0.4)) if adx is not None else 0.0
    w_mr = 1.0 - w_trend
    adf = mr_adf_test(panel["close"], asof_date, lookback=90, n_lags=1)
    adf_factor = 1.0 if adf.get("reject_5pct") else 0.6
    h = mr_hurst(panel["close"], asof_date, lookback=90)
    if h is None:
        hurst_factor = 0.5
    elif h < 0.45:
        hurst_factor = 1.0
    elif h < 0.55:
        hurst_factor = 0.6
    else:
        hurst_factor = 0.3

    final = _clip(w_trend * trend_score - w_mr * adf_factor * hurst_factor * mr_score)

    atr_full = _ind(panel, "ATR14", ["ATR20"])
    atr_pct = None
    if isinstance(atr_full, pd.Series) and not atr_full.empty:
        rp_atr = rolling_percentile(atr_full, window=252)
        atr_pct_today = safe_float(rp_atr.loc[rp_atr.index <= pd.Timestamp(asof_date)].iloc[-1])
        atr_pct = (atr_pct_today / 100.0) if atr_pct_today is not None else None
    regime = _regime_label(adx, atr_pct)

    out["trend_score"] = trend_score
    out["mr_score"] = mr_score
    out["final_score"] = final
    out["trend_factors"] = trend_factors
    out["mr_factors"] = mr_factors
    out["weights"] = {"w_trend": w_trend, "w_mr": w_mr,
                       "adf_p": adf.get("pvalue"), "adf_reject": adf.get("reject_5pct"),
                       "adf_factor": adf_factor,
                       "hurst": h, "hurst_factor": hurst_factor,
                       "ou_half_life": hl, "atr_pct_252d": atr_pct}
    out["regime"] = regime
    out["interpretation"] = _final_to_label(final)
    return out


# =============================================================================
# FLY composites — 3 trend / 7 MR (extreme MR-tilted)
# =============================================================================
def compute_fly_composites(panel: pd.DataFrame, asof_date: date,
                             bp_mult: float = 1.0) -> dict:
    """For SRA flies. Stored already in bp."""
    out = {"trend_score": 0.0, "mr_score": 0.0, "final_score": 0.0,
           "trend_factors": {}, "mr_factors": {}, "weights": {}, "regime": "NEUTRAL",
           "interpretation": "neutral · no edge", "scope": "fly", "error": None}
    try:
        today, history = _get_today_and_history(panel, asof_date, min_history=60)
    except ValueError as e:
        out["error"] = str(e); return out

    close = safe_float(today.get("close"))
    adx = safe_float(_ind(today, "ADX_14"))
    di_p = safe_float(_ind(today, "DIplus_14", ["DIplus_14"]))
    di_m = safe_float(_ind(today, "DIminus_14", ["DIminus_14"]))
    bb_pctB = safe_float(_ind(today, "BB_pctB_20_2.0x", ["BB_pctB_20_2", "BB_pctB_20"]))

    typ_bp = _typical_daily_bp(panel["close"], asof_date, lookback=20, bp_mult=bp_mult, floor_bp=0.05)

    # f1 5d slope
    f1 = 0.0
    slope_series = linear_slope(panel["close"], window=5)
    s_today = safe_float(slope_series.loc[slope_series.index <= pd.Timestamp(asof_date)].iloc[-1])
    if s_today is not None and typ_bp > 0:
        f1 = _clip((s_today * bp_mult / typ_bp) / 3.0)

    # f2 (close - mean_60d) / std_60d
    f2 = 0.0
    ts = pd.Timestamp(asof_date)
    hist60 = panel["close"].loc[panel["close"].index < ts].tail(60).dropna()
    if not hist60.empty and close is not None:
        mu = float(hist60.mean()); sd = float(hist60.std(ddof=1))
        if sd > 0:
            f2 = _clip(((close - mu) / sd) / 3.0)

    # f3 ADX·DI (denom 18)
    f3 = 0.0
    if adx is not None and di_p is not None and di_m is not None:
        f3 = _clip((adx / 18.0) * np.sign(di_p - di_m))

    trend_factors = {"f1_slope_5d": f1, "f2_z_60d": f2, "f3_ADX_DI_18": f3}
    trend_score = _clip(np.mean(list(trend_factors.values())))

    # =====================================================================
    # MR factors (7)
    # =====================================================================
    z60 = mr_zscore_value(panel["close"], asof_date, lookback=60)
    z30 = mr_zscore_value(panel["close"], asof_date, lookback=30)
    z15 = mr_zscore_value(panel["close"], asof_date, lookback=15)

    g1 = _clip(-z60 / 2.0) if z60 is not None else 0.0
    g2 = _clip(-z30 / 2.0) if z30 is not None else 0.0
    g3 = _clip(-z15 / 2.5) if z15 is not None else 0.0
    g4 = _clip(-(bb_pctB - 0.5) * 2.0) if bb_pctB is not None else 0.0

    g5 = 0.0
    rp = rolling_percentile(panel["close"], window=90)
    rp_today = safe_float(rp.loc[rp.index <= pd.Timestamp(asof_date)].iloc[-1])
    if rp_today is not None:
        g5 = _clip(-2.0 * (rp_today / 100.0 - 0.5))

    # g6 OU multiplier (signed boost)
    hl = mr_ou_half_life(panel["close"], asof_date, lookback=90)
    if hl is not None and np.isfinite(hl) and hl < 15.0:
        g6_mag = 1.0
    elif hl is not None and np.isfinite(hl) and hl < 30.0:
        g6_mag = 0.7
    elif hl is not None and np.isfinite(hl) and hl < 60.0:
        g6_mag = 0.4
    else:
        g6_mag = 0.0
    # signed by z direction (positive g6 = boost LONG fade signal)
    if z60 is not None:
        g6 = g6_mag * (-np.sign(z60))   # if z>0 (rich), boost shorting → MR negative for short
        g6 = _clip(g6)
    else:
        g6 = 0.0

    # g7 touch_count proxy via percentile_rank (no separate touch_count engine here)
    # When rp ≈ 0 or 100, the close is at recent extreme → strong fade signal
    g7 = 0.0
    if rp_today is not None and z60 is not None:
        ext_pull = max(0.0, abs(rp_today / 100.0 - 0.5) - 0.4) * 5.0   # 0..0.5 mapped 0..1
        g7 = _clip(ext_pull * (-np.sign(z60)))

    mr_factors = {"g1_-Z(60d)/2": g1, "g2_-Z(30d)/2": g2, "g3_-Z(15d)/2.5": g3,
                   "g4_-(BB_pctB-0.5)x2": g4, "g5_pct_rank_90d": g5,
                   "g6_OU_signed": g6, "g7_extreme_pull": g7}
    mr_score = _clip(np.mean(list(mr_factors.values())))

    # FINAL — extreme MR weighting
    w_trend = 0.10
    w_mr = 0.90
    adf = mr_adf_test(panel["close"], asof_date, lookback=90, n_lags=1)
    adf_factor = 1.0 if adf.get("reject_5pct") else 0.4
    h = mr_hurst(panel["close"], asof_date, lookback=90)
    if h is None:
        hurst_factor = 0.5
    elif h < 0.35:
        hurst_factor = 1.0
    elif h < 0.45:
        hurst_factor = 0.7
    elif h < 0.55:
        hurst_factor = 0.4
    else:
        hurst_factor = 0.2

    if hl is None or not np.isfinite(hl):
        ou_factor = 0.2
    elif hl < 15:
        ou_factor = 1.0
    elif hl < 30:
        ou_factor = 0.8
    elif hl < 60:
        ou_factor = 0.5
    else:
        ou_factor = 0.2

    final = _clip(w_trend * trend_score - w_mr * adf_factor * hurst_factor * ou_factor * mr_score)

    atr_full = _ind(panel, "ATR14", ["ATR20"])
    atr_pct = None
    if isinstance(atr_full, pd.Series) and not atr_full.empty:
        rp_atr = rolling_percentile(atr_full, window=252)
        atr_pct_today = safe_float(rp_atr.loc[rp_atr.index <= pd.Timestamp(asof_date)].iloc[-1])
        atr_pct = (atr_pct_today / 100.0) if atr_pct_today is not None else None
    regime = _regime_label(adx, atr_pct)

    out["trend_score"] = trend_score
    out["mr_score"] = mr_score
    out["final_score"] = final
    out["trend_factors"] = trend_factors
    out["mr_factors"] = mr_factors
    out["weights"] = {"w_trend": w_trend, "w_mr": w_mr,
                       "adf_p": adf.get("pvalue"), "adf_reject": adf.get("reject_5pct"),
                       "adf_factor": adf_factor, "hurst": h, "hurst_factor": hurst_factor,
                       "ou_half_life": hl, "ou_factor": ou_factor,
                       "atr_pct_252d": atr_pct}
    out["regime"] = regime
    out["interpretation"] = _final_to_label(final)
    return out


# =============================================================================
# Dispatcher
# =============================================================================
def compute_composites(panel: pd.DataFrame, asof_date: date,
                        scope: str, bp_mult: float,
                        carry_per_day_bp: Optional[float] = None) -> dict:
    """Dispatch to the right per-scope composite computer."""
    if scope == "outright":
        return compute_outright_composites(panel, asof_date, bp_mult, carry_per_day_bp)
    if scope == "spread":
        return compute_spread_composites(panel, asof_date, bp_mult)
    if scope == "fly":
        return compute_fly_composites(panel, asof_date, bp_mult)
    return {"error": f"unknown scope {scope}"}
