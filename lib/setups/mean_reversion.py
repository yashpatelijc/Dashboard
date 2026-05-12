"""Mean-reversion setup detectors — B1, B3, B5, B6, B10, B11, B13.

Same return contract as the trend detectors (SetupResult). B11 / B13 reuse the
existing ``lib.mean_reversion`` engine for OU half-life / Hurst / ADF so the
math is identical to what the Z-score & MR subtab already shows.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from lib.mean_reversion import (
    adf_test as mr_adf_test,
    hurst_exponent as mr_hurst,
    ou_half_life as mr_ou_half_life,
    zscore_value as mr_zscore_value,
)
from lib.runtime_indicators import (
    rsi_divergence_today,
    tr_zscore,
    true_range,
    vol_zscore,
)
from lib.setups.base import (
    SetupResult,
    compute_R_levels,
    distance_to_threshold_atr,
    lots_at_10k_risk,
    normalize_distance,
    round_to_tick,
    safe_float,
    state_from_conditions_met,
)
from lib.setups.registry import FAMILY_MR
from lib.setups.trend import _bool_to_n, _get_today_and_history, _ind, _set_levels_long, _set_levels_short


# =============================================================================
# B1 · BB_2SIGMA_REVERSION
# =============================================================================
def detect_b1(panel: pd.DataFrame, asof_date: date,
              bp_mult: float, scope: str = "outright") -> SetupResult:
    res = SetupResult(setup_id="B1", name="BB_2SIGMA_REVERSION",
                       family=FAMILY_MR, scope=scope)
    try:
        today, history = _get_today_and_history(panel, asof_date, min_history=22)
    except ValueError as e:
        res.state = "N/A"; res.error = str(e); return res

    close = safe_float(today.get("close"))
    bb_dn_2 = safe_float(_ind(today, "BB_DN_20_2.0x", ["BB_DN_20_2", "BB_DN_20"]))
    bb_dn_3 = safe_float(_ind(today, "BB_DN_20_3.0x", ["BB_DN_20_3", "BB_DN_20_2.5"]))
    bb_up_2 = safe_float(_ind(today, "BB_UP_20_2.0x", ["BB_UP_20_2", "BB_UP_20"]))
    bb_up_3 = safe_float(_ind(today, "BB_UP_20_3.0x", ["BB_UP_20_3", "BB_UP_20_2.5"]))
    bb_mid = safe_float(_ind(today, "BB_MID_20", ["SMA_20"]))
    adx = safe_float(_ind(today, "ADX_14"))

    if any(v is None for v in (close, bb_dn_2, bb_up_2, bb_mid, adx)):
        res.state = "N/A"; res.error = "missing BB / ADX"; return res

    # RSI divergence on close + RSI_14 series over last 20 bars
    rsi_full = _ind(panel, "RSI_14")
    if isinstance(rsi_full, pd.Series) and not rsi_full.empty:
        ts = pd.Timestamp(asof_date)
        prior_close_inc = pd.concat([history["close"], pd.Series([close], index=[ts])])
        prior_rsi_inc = pd.concat([rsi_full.loc[rsi_full.index < ts],
                                    pd.Series([safe_float(rsi_full.loc[rsi_full.index <= ts].iloc[-1])
                                               if not rsi_full.empty else None],
                                              index=[ts])])
        div = rsi_divergence_today(prior_close_inc, prior_rsi_inc, lookback_bars=20)
    else:
        div = {"bullish": False, "bearish": False}

    # LONG conditions
    c1_long = close <= bb_dn_2
    c2 = adx < 20.0
    c3_long = bool(div.get("bullish"))
    long_n = _bool_to_n(c1_long, c2, c3_long)
    # SHORT conditions
    c1_short = close >= bb_up_2
    c3_short = bool(div.get("bearish"))
    short_n = _bool_to_n(c1_short, c2, c3_short)

    n_met = max(long_n, short_n)
    state = state_from_conditions_met(n_met, 3)
    res.state = state
    res.confidence = n_met / 3.0
    if state == "FIRED" and long_n == 3:
        res.fired_long = True; res.direction = "LONG"
    elif state == "FIRED" and short_n == 3:
        res.fired_short = True; res.direction = "SHORT"
    elif state in ("NEAR", "APPROACHING"):
        res.direction = "LONG" if long_n >= short_n else "SHORT"

    res.key_inputs = {
        "close": close, "BB_DN_2σ": bb_dn_2, "BB_UP_2σ": bb_up_2, "BB_MID_20": bb_mid,
        "ADX_14": adx, "bullish_divergence": c3_long, "bearish_divergence": c3_short,
    }
    res.thresholds = {
        "close vs BB_2σ": "≤ BB_DN (LONG) or ≥ BB_UP (SHORT)",
        "ADX_14":          "< 20 (ranging regime)",
        "RSI divergence":  "bullish HL (LONG) / bearish LH (SHORT) over 20d",
    }
    if res.fired_long and bb_dn_3 is not None:
        _set_levels_long(res, close, bb_dn_3, bp_mult, 1.0, None)
        res.t1 = round_to_tick(bb_mid, bp_mult); res.t2 = round_to_tick(bb_up_2, bp_mult)
    elif res.fired_short and bb_up_3 is not None:
        _set_levels_short(res, close, bb_up_3, bp_mult, 1.0, None)
        res.t1 = round_to_tick(bb_mid, bp_mult); res.t2 = round_to_tick(bb_dn_2, bp_mult)

    if state in ("NEAR", "APPROACHING"):
        res.distance_to_fire = float(3 - n_met)
        res.eta_bars = res.distance_to_fire * 3.0

    res.interpretation = (
        f"FIRED {res.direction} — BB extreme + ranging + divergence" if state == "FIRED" else
        f"NEAR — {n_met}/3" if state == "NEAR" else
        f"APPROACHING — {n_met}/3" if state == "APPROACHING" else
        "no setup"
    )
    return res


# =============================================================================
# B3 · RSI_DIVERGENCE_REVERSAL
# =============================================================================
def detect_b3(panel: pd.DataFrame, asof_date: date,
              bp_mult: float, scope: str = "outright") -> SetupResult:
    res = SetupResult(setup_id="B3", name="RSI_DIVERGENCE_REVERSAL",
                       family=FAMILY_MR, scope=scope)
    try:
        today, history = _get_today_and_history(panel, asof_date, min_history=22)
    except ValueError as e:
        res.state = "N/A"; res.error = str(e); return res

    close = safe_float(today.get("close"))
    open_ = safe_float(today.get("open"))
    high_today = safe_float(today.get("high"))
    low_today = safe_float(today.get("low"))
    atr = safe_float(_ind(today, "ATR14", ["ATR20"]))
    if any(v is None for v in (close, atr, high_today, low_today)):
        res.state = "N/A"; res.error = "missing OHLC/ATR"; return res

    # New 20d low (LONG) / high (SHORT)
    last20 = history["close"].tail(20)
    new_20_low = bool(close <= safe_float(last20.min()))
    new_20_high = bool(close >= safe_float(last20.max()))

    # RSI divergence
    rsi_full = _ind(panel, "RSI_14")
    if not isinstance(rsi_full, pd.Series) or rsi_full.empty:
        res.state = "N/A"; res.error = "no RSI_14"; return res
    ts = pd.Timestamp(asof_date)
    rsi_today = safe_float(rsi_full.loc[rsi_full.index <= ts].iloc[-1])
    prior_close_inc = pd.concat([history["close"], pd.Series([close], index=[ts])])
    prior_rsi_inc = pd.concat([rsi_full.loc[rsi_full.index < ts],
                                pd.Series([rsi_today], index=[ts])])
    div = rsi_divergence_today(prior_close_inc, prior_rsi_inc, lookback_bars=20)

    # Bullish reversal candle: close > open AND body ≥ 0.5 × range
    body = abs(close - open_) if open_ is not None else 0.0
    rng = max(1e-9, high_today - low_today)
    bullish_candle = (open_ is not None and close > open_ and (body / rng) >= 0.5)
    bearish_candle = (open_ is not None and close < open_ and (body / rng) >= 0.5)

    long_n = _bool_to_n(new_20_low, div.get("bullish", False), bullish_candle)
    short_n = _bool_to_n(new_20_high, div.get("bearish", False), bearish_candle)
    n_met = max(long_n, short_n)
    state = state_from_conditions_met(n_met, 3)
    res.state = state
    res.confidence = n_met / 3.0
    if state == "FIRED" and long_n == 3:
        res.fired_long = True; res.direction = "LONG"
    elif state == "FIRED" and short_n == 3:
        res.fired_short = True; res.direction = "SHORT"
    elif state in ("NEAR", "APPROACHING"):
        res.direction = "LONG" if long_n >= short_n else "SHORT"

    res.key_inputs = {
        "close": close, "open": open_, "high": high_today, "low": low_today,
        "20d low touched": new_20_low, "20d high touched": new_20_high,
        "bullish_div": bool(div.get("bullish", False)), "bearish_div": bool(div.get("bearish", False)),
        "bullish_candle": bullish_candle, "bearish_candle": bearish_candle,
        "body/range": (body / rng) if rng > 0 else 0.0,
    }
    res.thresholds = {
        "20d extreme":    "new 20d low (LONG) or new 20d high (SHORT)",
        "RSI divergence": "bullish HL / bearish LH over 20d",
        "reversal candle": "body ≥ 0.5 × range AND closes opposite to direction of new extreme",
    }
    if res.fired_long:
        _set_levels_long(res, close, close - 1.5 * atr, bp_mult, 1.0, None)
    elif res.fired_short:
        _set_levels_short(res, close, close + 1.5 * atr, bp_mult, 1.0, None)

    if state in ("NEAR", "APPROACHING"):
        res.distance_to_fire = float(3 - n_met)
        res.eta_bars = res.distance_to_fire * 3.0

    res.interpretation = (
        f"FIRED {res.direction} — RSI divergence reversal at swing extreme" if state == "FIRED" else
        f"NEAR — {n_met}/3" if state == "NEAR" else
        f"APPROACHING — {n_met}/3" if state == "APPROACHING" else
        "no divergence setup"
    )
    return res


# =============================================================================
# B5 · ZPRICE_REVERSION
# =============================================================================
def detect_b5(panel: pd.DataFrame, asof_date: date,
              bp_mult: float, scope: str = "outright") -> SetupResult:
    res = SetupResult(setup_id="B5", name="ZPRICE_REVERSION",
                       family=FAMILY_MR, scope=scope)
    try:
        today, history = _get_today_and_history(panel, asof_date, min_history=52)
    except ValueError as e:
        res.state = "N/A"; res.error = str(e); return res

    close = safe_float(today.get("close"))
    z20 = safe_float(_ind(today, "ZPRICE_20"))
    z50 = safe_float(_ind(today, "ZPRICE_50"))
    adx = safe_float(_ind(today, "ADX_14"))
    sma20 = safe_float(_ind(today, "SMA_20", ["BB_MID_20"]))
    if any(v is None for v in (close, z20, z50, adx, sma20)):
        res.state = "N/A"; res.error = "missing ZPRICE_20/50, ADX, SMA_20"; return res

    # Compute std_20 from history for the dynamic stop
    std_20 = safe_float(history["close"].tail(20).std(ddof=1))
    if std_20 is None or std_20 <= 0:
        std_20 = safe_float(_ind(today, "BB_BW_20_2.0x", ["BB_BW_20_2"]))
        std_20 = (std_20 / 4.0) if std_20 else None    # rough fallback
    if std_20 is None or std_20 <= 0:
        res.state = "N/A"; res.error = "std_20 unavailable"; return res

    # LONG conditions
    c1_long = z20 < -2.0
    c2_long = z50 < -1.5
    c3 = adx < 25.0
    long_n = _bool_to_n(c1_long, c2_long, c3)
    # SHORT
    c1_short = z20 > +2.0
    c2_short = z50 > +1.5
    short_n = _bool_to_n(c1_short, c2_short, c3)

    n_met = max(long_n, short_n)
    state = state_from_conditions_met(n_met, 3)
    res.state = state
    res.confidence = n_met / 3.0
    if state == "FIRED" and long_n == 3:
        res.fired_long = True; res.direction = "LONG"
    elif state == "FIRED" and short_n == 3:
        res.fired_short = True; res.direction = "SHORT"
    elif state in ("NEAR", "APPROACHING"):
        res.direction = "LONG" if long_n >= short_n else "SHORT"

    res.key_inputs = {"close": close, "ZPRICE_20": z20, "ZPRICE_50": z50,
                       "ADX_14": adx, "SMA_20": sma20, "std_20": std_20}
    res.thresholds = {
        "ZPRICE_20": "< -2 (LONG) / > +2 (SHORT)",
        "ZPRICE_50": "< -1.5 (LONG) / > +1.5 (SHORT)",
        "ADX_14":     "< 25 (non-trending regime)",
    }
    if res.fired_long:
        _set_levels_long(res, close, sma20 - 3 * std_20, bp_mult, t1_r=1.0, t2_r=None)
        res.t1 = round_to_tick(sma20, bp_mult); res.t2 = None
    elif res.fired_short:
        _set_levels_short(res, close, sma20 + 3 * std_20, bp_mult, t1_r=1.0, t2_r=None)
        res.t1 = round_to_tick(sma20, bp_mult); res.t2 = None

    if state in ("NEAR", "APPROACHING"):
        if long_n >= short_n:
            dist = []
            if not c1_long: dist.append(max(0, abs(z20 + 2.0)))
            if not c2_long: dist.append(max(0, abs(z50 + 1.5)))
            if not c3: dist.append(max(0, (adx - 25.0) / 5.0))
        else:
            dist = []
            if not c1_short: dist.append(max(0, abs(z20 - 2.0)))
            if not c2_short: dist.append(max(0, abs(z50 - 1.5)))
            if not c3: dist.append(max(0, (adx - 25.0) / 5.0))
        res.distance_to_fire = normalize_distance(dist)
        res.eta_bars = res.distance_to_fire * 3.0 if np.isfinite(res.distance_to_fire) else None

    res.interpretation = (
        f"FIRED {res.direction} — double-window ZPRICE stretch in non-trending regime" if state == "FIRED" else
        f"NEAR — {n_met}/3" if state == "NEAR" else
        f"APPROACHING — {n_met}/3" if state == "APPROACHING" else
        "no setup"
    )
    return res


# =============================================================================
# B6 · ZRET_5D_REVERSAL
# =============================================================================
def detect_b6(panel: pd.DataFrame, asof_date: date,
              bp_mult: float, scope: str = "outright") -> SetupResult:
    res = SetupResult(setup_id="B6", name="ZRET_5D_REVERSAL",
                       family=FAMILY_MR, scope=scope)
    try:
        today, history = _get_today_and_history(panel, asof_date, min_history=10)
    except ValueError as e:
        res.state = "N/A"; res.error = str(e); return res

    close = safe_float(today.get("close"))
    zret5 = safe_float(_ind(today, "ZRET_5"))
    atr = safe_float(_ind(today, "ATR14", ["ATR20"]))
    if any(v is None for v in (close, zret5, atr)):
        res.state = "N/A"; res.error = "missing close/ZRET_5/ATR"; return res

    if zret5 < -2.5:
        res.state = "FIRED"; res.fired_long = True; res.direction = "LONG"
    elif zret5 > 2.5:
        res.state = "FIRED"; res.fired_short = True; res.direction = "SHORT"
    elif abs(zret5) >= 2.0:
        res.state = "NEAR"; res.direction = "LONG" if zret5 < 0 else "SHORT"
    elif abs(zret5) >= 1.5:
        res.state = "APPROACHING"; res.direction = "LONG" if zret5 < 0 else "SHORT"
    else:
        res.state = "FAR"

    res.confidence = min(1.0, abs(zret5) / 2.5) if zret5 is not None else 0.0
    res.key_inputs = {"close": close, "ZRET_5": zret5, "ATR14": atr}
    res.thresholds = {"|ZRET_5|": "≥ 2.5 → FIRED · ≥ 2 → NEAR · ≥ 1.5 → APPROACHING"}
    if res.fired_long:
        _set_levels_long(res, close, close - 2 * atr, bp_mult, 1.0, None)
    elif res.fired_short:
        _set_levels_short(res, close, close + 2 * atr, bp_mult, 1.0, None)

    if res.state in ("NEAR", "APPROACHING"):
        res.distance_to_fire = max(0, 2.5 - abs(zret5))
        res.eta_bars = res.distance_to_fire * 1.5

    res.interpretation = (
        f"FIRED {res.direction} — 5d return outlier (ZRET_5={zret5:+.2f}σ)" if res.state == "FIRED" else
        f"NEAR (ZRET_5={zret5:+.2f}σ)" if res.state == "NEAR" else
        f"APPROACHING (ZRET_5={zret5:+.2f}σ)" if res.state == "APPROACHING" else
        "no signal"
    )
    return res


# =============================================================================
# B10 · VOLUME_CLIMAX_FADE  (90d windows per Yash customization)
# =============================================================================
def detect_b10(panel: pd.DataFrame, asof_date: date,
               bp_mult: float, scope: str = "outright") -> SetupResult:
    res = SetupResult(setup_id="B10", name="VOLUME_CLIMAX_FADE",
                       family=FAMILY_MR, scope=scope)
    try:
        today, history = _get_today_and_history(panel, asof_date, min_history=92)
    except ValueError as e:
        res.state = "N/A"; res.error = str(e); return res

    close = safe_float(today.get("close"))
    high_t = safe_float(today.get("high"))
    low_t = safe_float(today.get("low"))
    vol_today = safe_float(today.get("volume"))
    bb_dn = safe_float(_ind(today, "BB_DN_20_2.0x", ["BB_DN_20_2", "BB_DN_20"]))
    bb_up = safe_float(_ind(today, "BB_UP_20_2.0x", ["BB_UP_20_2", "BB_UP_20"]))
    ema20 = safe_float(_ind(today, "EMA_20"))
    if any(v is None for v in (close, high_t, low_t, bb_dn, bb_up, ema20)):
        res.state = "N/A"; res.error = "missing OHLC/BB/EMA"; return res

    if vol_today is None:
        # STIR contracts may have sparse volume — treat as quiet
        res.state = "N/A"; res.error = "no volume data"; return res

    # Vol z-score over 90d
    vol_full = panel["volume"] if "volume" in panel.columns else None
    if vol_full is None or vol_full.empty:
        res.state = "N/A"; res.error = "no volume series"; return res
    ts = pd.Timestamp(asof_date)
    vol_z_series = vol_zscore(vol_full, window=90)
    vol_z_today = safe_float(vol_z_series.loc[vol_z_series.index <= ts].iloc[-1])

    # TR z-score over 90d
    tr_full = true_range(panel["high"], panel["low"], panel["close"])
    tr_z_series = tr_zscore(tr_full, window=90)
    tr_z_today = safe_float(tr_z_series.loc[tr_z_series.index <= ts].iloc[-1])

    if vol_z_today is None or tr_z_today is None:
        res.state = "N/A"; res.error = "vol_z / tr_z NaN"; return res

    # Position in bar's range
    bar_rng = high_t - low_t
    if bar_rng <= 0:
        bar_pos = 0.5
    else:
        bar_pos = (close - low_t) / bar_rng    # 0=at low, 1=at high

    c1 = vol_z_today >= 3.0
    c2 = tr_z_today >= 2.0
    c3_long = bar_pos <= 0.25
    c3_short = bar_pos >= 0.75
    c4_long = close < bb_dn
    c4_short = close > bb_up

    long_n = _bool_to_n(c1, c2, c3_long, c4_long)
    short_n = _bool_to_n(c1, c2, c3_short, c4_short)

    n_met = max(long_n, short_n)
    state = state_from_conditions_met(n_met, 4)
    res.state = state
    res.confidence = n_met / 4.0
    if state == "FIRED" and long_n == 4:
        res.fired_long = True; res.direction = "LONG"
    elif state == "FIRED" and short_n == 4:
        res.fired_short = True; res.direction = "SHORT"
    elif state in ("NEAR", "APPROACHING"):
        res.direction = "LONG" if long_n >= short_n else "SHORT"

    res.key_inputs = {
        "close": close, "high": high_t, "low": low_t,
        "vol_z(90d)": vol_z_today, "TR_z(90d)": tr_z_today,
        "bar position": bar_pos, "BB_DN_2σ": bb_dn, "BB_UP_2σ": bb_up,
    }
    res.thresholds = {
        "vol_z(90d)":          "≥ 3",
        "TR_z(90d)":           "≥ 2",
        "bar position":        "≤ 0.25 (LONG fade) / ≥ 0.75 (SHORT fade)",
        "close vs BB_2σ":      "< BB_DN (LONG) / > BB_UP (SHORT)",
    }
    if res.fired_long:
        _set_levels_long(res, close, low_t, bp_mult, t1_r=1.0, t2_r=None)
        res.t1 = round_to_tick(ema20, bp_mult); res.t2 = None
    elif res.fired_short:
        _set_levels_short(res, close, high_t, bp_mult, t1_r=1.0, t2_r=None)
        res.t1 = round_to_tick(ema20, bp_mult); res.t2 = None

    if state in ("NEAR", "APPROACHING"):
        res.distance_to_fire = float(4 - n_met)
        res.eta_bars = res.distance_to_fire * 2.0

    res.interpretation = (
        f"FIRED {res.direction} — capitulation candle / climactic extreme" if state == "FIRED" else
        f"NEAR — {n_met}/4" if state == "NEAR" else
        f"APPROACHING — {n_met}/4" if state == "APPROACHING" else
        "no climax"
    )
    return res


# =============================================================================
# B11 · OU_HALF_LIFE_REVERSION  (uses lib.mean_reversion engine)
# =============================================================================
def detect_b11(panel: pd.DataFrame, asof_date: date,
               bp_mult: float, scope: str = "outright") -> SetupResult:
    res = SetupResult(setup_id="B11", name="OU_HALF_LIFE_REVERSION",
                       family=FAMILY_MR, scope=scope)
    try:
        today, history = _get_today_and_history(panel, asof_date, min_history=62)
    except ValueError as e:
        res.state = "N/A"; res.error = str(e); return res

    close = safe_float(today.get("close"))
    sma20 = safe_float(_ind(today, "SMA_20", ["BB_MID_20"]))
    if close is None or sma20 is None:
        res.state = "N/A"; res.error = "missing close/SMA_20"; return res

    series = panel["close"]
    z60 = mr_zscore_value(series, asof_date, lookback=60)
    hl = mr_ou_half_life(series, asof_date, lookback=90)
    if z60 is None:
        res.state = "N/A"; res.error = "z(60d) unavailable"; return res

    # std_60 for dynamic stop
    ts = pd.Timestamp(asof_date)
    history_60 = series.loc[series.index < ts].tail(60).dropna()
    std_60 = safe_float(history_60.std(ddof=1)) if not history_60.empty else None
    mean_60 = safe_float(history_60.mean()) if not history_60.empty else None

    hl_ok = (hl is not None and np.isfinite(hl) and hl < 30.0)
    z_long = z60 < -2.0
    z_short = z60 > +2.0
    z_long_near = -2.0 <= z60 <= -1.5
    z_short_near = 1.5 <= z60 <= 2.0
    z_long_appr = -1.5 < z60 < -1.0
    z_short_appr = 1.0 < z60 < 1.5

    if hl_ok and z_long:
        res.state = "FIRED"; res.fired_long = True; res.direction = "LONG"
    elif hl_ok and z_short:
        res.state = "FIRED"; res.fired_short = True; res.direction = "SHORT"
    elif hl_ok and (z_long_near or z_short_near):
        res.state = "NEAR"; res.direction = "LONG" if z_long_near else "SHORT"
    elif hl_ok and (z_long_appr or z_short_appr):
        res.state = "APPROACHING"; res.direction = "LONG" if z_long_appr else "SHORT"
    else:
        res.state = "FAR"

    res.confidence = min(1.0, abs(z60) / 2.5) if (hl_ok and z60 is not None) else 0.0
    res.key_inputs = {"close": close, "z(60d)": z60, "OU half-life": hl,
                       "mean_60": mean_60, "std_60": std_60}
    res.thresholds = {"OU half-life": "< 30 trading days",
                       "|z(60d)|":      "≥ 2 → FIRED · ≥ 1.5 → NEAR · ≥ 1 → APPROACHING"}
    if res.fired_long and mean_60 is not None and std_60 is not None and std_60 > 0:
        _set_levels_long(res, close, mean_60 - 3 * std_60, bp_mult, t1_r=1.0, t2_r=None)
        res.t1 = round_to_tick(mean_60, bp_mult); res.t2 = None
    elif res.fired_short and mean_60 is not None and std_60 is not None and std_60 > 0:
        _set_levels_short(res, close, mean_60 + 3 * std_60, bp_mult, t1_r=1.0, t2_r=None)
        res.t1 = round_to_tick(mean_60, bp_mult); res.t2 = None

    if res.state in ("NEAR", "APPROACHING"):
        # Distance to FIRED: |z| needs to reach 2
        res.distance_to_fire = float(max(0, 2.0 - abs(z60)))
        res.eta_bars = res.distance_to_fire * 3.0

    res.interpretation = (
        f"FIRED {res.direction} — high-quality OU mean-reversion (HL={hl:.1f}d)" if res.state == "FIRED" else
        f"NEAR (z={z60:+.2f}σ, HL={hl:.1f}d)" if res.state == "NEAR" else
        f"APPROACHING (z={z60:+.2f}σ)" if res.state == "APPROACHING" else
        "no setup (HL≥30d or |z|<1)"
    )
    return res


# =============================================================================
# B13 · HURST_FILTERED_MEAN_REVERSION
# =============================================================================
def detect_b13(panel: pd.DataFrame, asof_date: date,
               bp_mult: float, scope: str = "outright") -> SetupResult:
    res = SetupResult(setup_id="B13", name="HURST_FILTERED_MEAN_REVERSION",
                       family=FAMILY_MR, scope=scope)
    try:
        today, history = _get_today_and_history(panel, asof_date, min_history=62)
    except ValueError as e:
        res.state = "N/A"; res.error = str(e); return res

    close = safe_float(today.get("close"))
    z20 = safe_float(_ind(today, "ZPRICE_20"))
    sma20 = safe_float(_ind(today, "SMA_20", ["BB_MID_20"]))
    if any(v is None for v in (close, z20, sma20)):
        res.state = "N/A"; res.error = "missing ZPRICE_20 / SMA_20"; return res

    h = mr_hurst(panel["close"], asof_date, lookback=90)
    h_ok = (h is not None and h < 0.45)
    z_long = z20 < -2.0
    z_short = z20 > +2.0
    z_long_near = -2.0 <= z20 <= -1.5
    z_short_near = 1.5 <= z20 <= 2.0
    z_long_appr = -1.5 < z20 < -1.0
    z_short_appr = 1.0 < z20 < 1.5

    if h_ok and z_long:
        res.state = "FIRED"; res.fired_long = True; res.direction = "LONG"
    elif h_ok and z_short:
        res.state = "FIRED"; res.fired_short = True; res.direction = "SHORT"
    elif h_ok and (z_long_near or z_short_near):
        res.state = "NEAR"; res.direction = "LONG" if z_long_near else "SHORT"
    elif h_ok and (z_long_appr or z_short_appr):
        res.state = "APPROACHING"; res.direction = "LONG" if z_long_appr else "SHORT"
    else:
        res.state = "FAR"

    # Std for dynamic stop
    ts = pd.Timestamp(asof_date)
    history_20 = panel["close"].loc[panel["close"].index < ts].tail(20).dropna()
    std_20 = safe_float(history_20.std(ddof=1)) if not history_20.empty else None

    res.confidence = min(1.0, abs(z20) / 2.5) if h_ok else 0.0
    res.key_inputs = {"close": close, "ZPRICE_20": z20, "Hurst_60d": h, "SMA_20": sma20}
    res.thresholds = {"Hurst_60d": "< 0.45 (anti-persistent / mean-reverting regime)",
                       "|ZPRICE_20|": "≥ 2 → FIRED · ≥ 1.5 → NEAR · ≥ 1 → APPROACHING"}
    if res.fired_long and std_20 is not None and std_20 > 0:
        _set_levels_long(res, close, sma20 - 3 * std_20, bp_mult, t1_r=1.0, t2_r=None)
        res.t1 = round_to_tick(sma20, bp_mult); res.t2 = None
    elif res.fired_short and std_20 is not None and std_20 > 0:
        _set_levels_short(res, close, sma20 + 3 * std_20, bp_mult, t1_r=1.0, t2_r=None)
        res.t1 = round_to_tick(sma20, bp_mult); res.t2 = None

    if res.state in ("NEAR", "APPROACHING"):
        res.distance_to_fire = float(max(0, 2.0 - abs(z20)))
        res.eta_bars = res.distance_to_fire * 3.0

    res.interpretation = (
        f"FIRED {res.direction} — Hurst-filtered MR (H={h:.2f}, z={z20:+.2f}σ)" if res.state == "FIRED" else
        f"NEAR (H={h:.2f}, z={z20:+.2f}σ)" if res.state == "NEAR" else
        f"APPROACHING (H={h:.2f}, z={z20:+.2f}σ)" if res.state == "APPROACHING" else
        "no setup (Hurst≥0.45 or |z|<1)"
    )
    return res
