"""Trend setup detectors — A1, A2, A3, A4, A5, A6, A8, A10, A11, A12a, A12b, A15.

Each detector is a pure function:
    detect_xx(panel: pd.DataFrame, asof_date: date, bp_mult: float, scope: str)
        → SetupResult

``panel`` must contain (at minimum) the OHLCV columns plus the indicators
referenced by the detector. The scan layer (lib.setups.scan) is responsible
for fetching panels via the ``v_mde2_timeseries_with_indicators`` view so
the panel covers everything in one pull.

Locked conventions:
  · history excludes today (``panel.loc[panel.index < ts]``)
  · ``today_row`` extracted by date-equality
  · all detectors return SetupResult — never raise; on error, ``state="N/A"`` + ``error``
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

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
from lib.setups.registry import FAMILY_TREND
from lib.runtime_indicators import (
    bb_inside_kc_counter,
    donchian_high_low,
    ema_200,
    ichimoku_full,
    polyfit_pattern_lines,
    rolling_percentile,
    swing_detect,
    wilder_ema,
)


# =============================================================================
# Internal helpers (shared across detectors)
# =============================================================================
def _get_today_and_history(panel: pd.DataFrame, asof_date: date,
                            min_history: int = 1) -> tuple:
    """Return (today_row Series, history DataFrame). Raises ValueError on bad data."""
    if panel is None or panel.empty:
        raise ValueError("no panel")
    ts = pd.Timestamp(asof_date)
    history = panel.loc[panel.index < ts]
    if len(history) < min_history:
        raise ValueError(f"insufficient history (<{min_history} bars)")
    today_match = panel.loc[panel.index.normalize() == ts.normalize()]
    if today_match.empty:
        raise ValueError(f"no row for asof_date {asof_date}")
    return today_match.iloc[-1], history


def _ind(row_or_panel, name: str, fallbacks: Optional[list] = None):
    """Resolve indicator value/series from a row or a panel.

    Tries primary, then case-insensitive, then fallbacks (if any).
    Returns None for a row when not found; empty Series for a panel.
    """
    if isinstance(row_or_panel, pd.Series):
        cols = row_or_panel.index
    else:
        cols = row_or_panel.columns
    if name in cols:
        return row_or_panel[name]
    cl = {str(c).lower(): c for c in cols}
    if name.lower() in cl:
        return row_or_panel[cl[name.lower()]]
    if fallbacks:
        for fb in fallbacks:
            if fb in cols:
                return row_or_panel[fb]
            if fb.lower() in cl:
                return row_or_panel[cl[fb.lower()]]
    if isinstance(row_or_panel, pd.Series):
        return None
    return pd.Series(dtype=float)


def _set_levels_long(res: SetupResult, entry: float, stop: float,
                     bp_mult: float, t1_r: float = 1.0, t2_r: Optional[float] = 2.0) -> None:
    """Entry / stop / T1 / T2 are all rounded to the half-bp tick for display."""
    res.entry = round_to_tick(float(entry), bp_mult)
    res.stop = round_to_tick(float(stop), bp_mult)
    if res.entry is None or res.stop is None or res.entry <= res.stop:
        return
    t1, t2 = compute_R_levels(res.entry, res.stop, "LONG", t1_r, t2_r)
    res.t1 = round_to_tick(t1, bp_mult)
    res.t2 = round_to_tick(t2, bp_mult)
    sd_bp = (res.entry - res.stop) * bp_mult
    res.lots_at_10k_risk = lots_at_10k_risk(sd_bp)


def _set_levels_short(res: SetupResult, entry: float, stop: float,
                      bp_mult: float, t1_r: float = 1.0, t2_r: Optional[float] = 2.0) -> None:
    res.entry = round_to_tick(float(entry), bp_mult)
    res.stop = round_to_tick(float(stop), bp_mult)
    if res.entry is None or res.stop is None or res.stop <= res.entry:
        return
    t1, t2 = compute_R_levels(res.entry, res.stop, "SHORT", t1_r, t2_r)
    res.t1 = round_to_tick(t1, bp_mult)
    res.t2 = round_to_tick(t2, bp_mult)
    sd_bp = (res.stop - res.entry) * bp_mult
    res.lots_at_10k_risk = lots_at_10k_risk(sd_bp)


def _bool_to_n(*conds) -> int:
    return int(sum(bool(c) for c in conds))


# =============================================================================
# A1 · TREND_CONTINUATION_BREAKOUT
# =============================================================================
def detect_a1(panel: pd.DataFrame, asof_date: date,
              bp_mult: float, scope: str = "outright") -> SetupResult:
    res = SetupResult(setup_id="A1", name="TREND_CONTINUATION_BREAKOUT",
                       family=FAMILY_TREND, scope=scope)
    try:
        today, history = _get_today_and_history(panel, asof_date, min_history=22)
    except ValueError as e:
        res.state = "N/A"; res.error = str(e); return res

    close = safe_float(today.get("close"))
    adx = safe_float(_ind(today, "ADX_14"))
    ema50 = safe_float(_ind(today, "EMA_50"))
    vol = safe_float(today.get("volume"))
    high_20 = safe_float(history["high"].tail(20).max() if "high" in history.columns else None)
    low_20 = safe_float(history["low"].tail(20).min() if "low" in history.columns else None)
    sma_vol = safe_float(history["volume"].tail(20).mean() if "volume" in history.columns else None)
    atr = safe_float(_ind(today, "ATR14", ["ATR20", "ATR10", "ATR5"]))

    # Validate critical inputs
    if close is None or adx is None or ema50 is None or atr is None:
        res.state = "N/A"; res.error = "missing indicator(s)"; return res
    if vol is None or sma_vol is None or sma_vol <= 0:
        # volume might be 0/missing on STIR contracts — treat vol filter as "ok"
        vol = vol if vol is not None else 0.0
        sma_vol = sma_vol if sma_vol and sma_vol > 0 else 1.0

    # LONG conditions
    c1_long = (high_20 is not None) and (close > high_20)
    c2 = adx > 20.0
    c3_long = close > ema50
    c4 = vol > 1.5 * sma_vol if sma_vol > 0 else False
    long_n = _bool_to_n(c1_long, c2, c3_long, c4)

    # SHORT conditions
    c1_short = (low_20 is not None) and (close < low_20)
    c3_short = close < ema50
    short_n = _bool_to_n(c1_short, c2, c3_short, c4)

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
        "close": close, "High_20": high_20, "Low_20": low_20,
        "ADX_14": adx, "EMA_50": ema50, "vol": vol, "vol/SMA20": vol / sma_vol if sma_vol > 0 else None,
    }
    res.thresholds = {
        "close vs High_20/Low_20": "> for LONG / < for SHORT",
        "ADX_14": "> 20",
        "close vs EMA_50": "> for LONG / < for SHORT",
        "vol/SMA20": "> 1.5",
    }

    # Trade levels
    if res.fired_long:
        _set_levels_long(res, close, close - 2 * atr, bp_mult, 1.0, 2.0)
    elif res.fired_short:
        _set_levels_short(res, close, close + 2 * atr, bp_mult, 1.0, 2.0)

    # Distance to fire (for NEAR/APPROACHING)
    if state in ("NEAR", "APPROACHING") and atr > 0:
        # use the side with more met
        if long_n >= short_n:
            dist_atr = []
            if not c1_long and high_20 is not None:
                dist_atr.append(max(0, (high_20 - close) / atr))
            if not c2:
                dist_atr.append(max(0, (20.0 - adx) / 5.0))
            if not c3_long:
                dist_atr.append(max(0, (ema50 - close) / atr))
            if not c4 and sma_vol > 0:
                dist_atr.append(max(0, (1.5 * sma_vol - vol) / sma_vol))
        else:
            dist_atr = []
            if not c1_short and low_20 is not None:
                dist_atr.append(max(0, (close - low_20) / atr))
            if not c2:
                dist_atr.append(max(0, (20.0 - adx) / 5.0))
            if not c3_short:
                dist_atr.append(max(0, (close - ema50) / atr))
            if not c4 and sma_vol > 0:
                dist_atr.append(max(0, (1.5 * sma_vol - vol) / sma_vol))
        res.distance_to_fire = normalize_distance(dist_atr)
        # ETA: assume each condition closes 1 unit per typical bar
        res.eta_bars = res.distance_to_fire * 2.0 if np.isfinite(res.distance_to_fire) else None

    res.interpretation = (
        f"FIRED {res.direction} ({n_met}/4 conditions)" if state == "FIRED" else
        f"NEAR — {n_met}/4 conditions ({res.direction or '—'})" if state == "NEAR" else
        f"APPROACHING — {n_met}/4 conditions" if state == "APPROACHING" else
        "no setup"
    )
    return res


# =============================================================================
# A2 · PULLBACK_IN_TREND  (EMA_100 instead of EMA_200 per Yash customization)
# =============================================================================
def detect_a2(panel: pd.DataFrame, asof_date: date,
              bp_mult: float, scope: str = "outright") -> SetupResult:
    res = SetupResult(setup_id="A2", name="PULLBACK_IN_TREND",
                       family=FAMILY_TREND, scope=scope)
    try:
        today, history = _get_today_and_history(panel, asof_date, min_history=110)
    except ValueError as e:
        res.state = "N/A"; res.error = str(e); return res

    close = safe_float(today.get("close"))
    ema100 = safe_float(_ind(today, "EMA_100"))
    ema20 = safe_float(_ind(today, "EMA_20"))
    rsi14 = safe_float(_ind(today, "RSI_14"))
    atr = safe_float(_ind(today, "ATR14", ["ATR20"]))

    if close is None or ema100 is None or ema20 is None or rsi14 is None or atr is None:
        res.state = "N/A"; res.error = "missing indicator(s)"; return res

    # LONG: in uptrend (close>EMA100), pulling back (close<EMA20), RSI<45
    c1_long = close > ema100
    c2_long = close < ema20
    c3_long = rsi14 < 45.0
    long_n = _bool_to_n(c1_long, c2_long, c3_long)
    # SHORT mirrors
    c1_short = close < ema100
    c2_short = close > ema20
    c3_short = rsi14 > 55.0
    short_n = _bool_to_n(c1_short, c2_short, c3_short)

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

    res.key_inputs = {"close": close, "EMA_100": ema100, "EMA_20": ema20, "RSI_14": rsi14}
    res.thresholds = {
        "close vs EMA_100": "> for LONG / < for SHORT",
        "close vs EMA_20":  "< for LONG / > for SHORT (pullback condition)",
        "RSI_14":            "< 45 for LONG / > 55 for SHORT",
    }
    if res.fired_long:
        _set_levels_long(res, close, close - 1.5 * atr, bp_mult, 1.0, None)
    elif res.fired_short:
        _set_levels_short(res, close, close + 1.5 * atr, bp_mult, 1.0, None)

    if state in ("NEAR", "APPROACHING") and atr > 0:
        if long_n >= short_n:
            dist = []
            if not c1_long: dist.append(max(0, (ema100 - close) / atr))
            if not c2_long: dist.append(max(0, (close - ema20) / atr))
            if not c3_long: dist.append(max(0, (rsi14 - 45.0) / 10.0))
        else:
            dist = []
            if not c1_short: dist.append(max(0, (close - ema100) / atr))
            if not c2_short: dist.append(max(0, (ema20 - close) / atr))
            if not c3_short: dist.append(max(0, (55.0 - rsi14) / 10.0))
        res.distance_to_fire = normalize_distance(dist)
        res.eta_bars = res.distance_to_fire * 2.0 if np.isfinite(res.distance_to_fire) else None

    res.interpretation = (
        f"FIRED {res.direction} ({n_met}/3) — buy/sell the dip in trend" if state == "FIRED" else
        f"NEAR — {n_met}/3 ({res.direction or '—'})" if state == "NEAR" else
        f"APPROACHING — {n_met}/3" if state == "APPROACHING" else
        "no setup"
    )
    return res


# =============================================================================
# A3 · VCP_BREAKOUT  (90d percentile per Yash customization)
# =============================================================================
def detect_a3(panel: pd.DataFrame, asof_date: date,
              bp_mult: float, scope: str = "outright") -> SetupResult:
    res = SetupResult(setup_id="A3", name="VCP_BREAKOUT",
                       family=FAMILY_TREND, scope=scope)
    try:
        today, history = _get_today_and_history(panel, asof_date, min_history=92)
    except ValueError as e:
        res.state = "N/A"; res.error = str(e); return res

    close = safe_float(today.get("close"))
    bb_bw_today = safe_float(_ind(today, "BB_BW_20_2.0x", ["BB_BW_20_2", "BB_BW_20"]))
    bb_up = _ind(panel, "BB_UP_20_2.0x", ["BB_UP_20_2", "BB_UP_20"])
    bb_dn = _ind(panel, "BB_DN_20_2.0x", ["BB_DN_20_2", "BB_DN_20"])
    kc_up = _ind(panel, "KC_UP_20_2.0x_ATR20", ["KC_UP_20_2", "KC_UP_20"])
    kc_dn = _ind(panel, "KC_DN_20_2.0x_ATR20", ["KC_DN_20_2", "KC_DN_20"])
    bb_bw_full = _ind(panel, "BB_BW_20_2.0x", ["BB_BW_20_2", "BB_BW_20"])
    vol = safe_float(today.get("volume"))
    sma_vol = safe_float(history["volume"].tail(20).mean() if "volume" in history.columns else None)
    atr = safe_float(_ind(today, "ATR14", ["ATR20"]))

    if close is None or atr is None or bb_bw_today is None:
        res.state = "N/A"; res.error = "missing BB / ATR"; return res

    # 90d percentile of BB_BW
    if not isinstance(bb_bw_full, pd.Series) or bb_bw_full.empty:
        res.state = "N/A"; res.error = "BB_BW series unavailable"; return res
    bb_bw_pct = rolling_percentile(bb_bw_full, window=90)
    pct_today = safe_float(bb_bw_pct.loc[bb_bw_pct.index < pd.Timestamp(asof_date)].iloc[-1]
                             if not bb_bw_pct.empty else None)

    # BB-inside-KC count over last 10 bars (ending at today, exclusive)
    inside_count = 0
    if (isinstance(bb_up, pd.Series) and isinstance(kc_up, pd.Series)
            and isinstance(bb_dn, pd.Series) and isinstance(kc_dn, pd.Series)):
        cnt_series = bb_inside_kc_counter(bb_up, bb_dn, kc_up, kc_dn, window=10)
        inside_count = safe_float(cnt_series.loc[cnt_series.index < pd.Timestamp(asof_date)].iloc[-1]
                                    if not cnt_series.empty else None) or 0
    inside_count = int(inside_count or 0)

    # 5-bar consolidation: high/low of close over last 5 prior bars
    last5 = history["close"].tail(5)
    cons_high = safe_float(last5.max() if not last5.empty else None)
    cons_low = safe_float(last5.min() if not last5.empty else None)

    # Volume thrust
    vol_ratio = (vol / sma_vol) if (vol is not None and sma_vol and sma_vol > 0) else None

    # Conditions LONG
    c1_pct = pct_today is not None and pct_today < 25.0
    c2_squeeze = inside_count >= 5
    c3_long = (cons_high is not None and close > cons_high)
    c4 = vol_ratio is not None and vol_ratio >= 2.0
    long_n = _bool_to_n(c1_pct, c2_squeeze, c3_long, c4)

    c3_short = (cons_low is not None and close < cons_low)
    short_n = _bool_to_n(c1_pct, c2_squeeze, c3_short, c4)

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
        "close": close, "BB_BW_pct90d": pct_today, "BB-inside-KC count(10)": inside_count,
        "5d cons_high": cons_high, "5d cons_low": cons_low, "vol/SMA20": vol_ratio,
    }
    res.thresholds = {
        "BB_BW percentile_90d": "< 25",
        "BB-inside-KC count(10)": "≥ 5",
        "close vs 5d consolidation": "> high (LONG) / < low (SHORT)",
        "vol/SMA20": "≥ 2.0",
    }
    if res.fired_long and cons_low is not None:
        _set_levels_long(res, close, cons_low, bp_mult, 2.0, 3.0)
    elif res.fired_short and cons_high is not None:
        _set_levels_short(res, close, cons_high, bp_mult, 2.0, 3.0)

    if state in ("NEAR", "APPROACHING") and atr > 0:
        if long_n >= short_n:
            dist = []
            if not c1_pct and pct_today is not None: dist.append(max(0, (pct_today - 25.0) / 25.0))
            if not c2_squeeze: dist.append(max(0, (5 - inside_count) / 5.0))
            if not c3_long and cons_high is not None: dist.append(max(0, (cons_high - close) / atr))
            if not c4 and vol_ratio is not None: dist.append(max(0, (2.0 - vol_ratio) / 2.0))
        else:
            dist = []
            if not c1_pct and pct_today is not None: dist.append(max(0, (pct_today - 25.0) / 25.0))
            if not c2_squeeze: dist.append(max(0, (5 - inside_count) / 5.0))
            if not c3_short and cons_low is not None: dist.append(max(0, (close - cons_low) / atr))
            if not c4 and vol_ratio is not None: dist.append(max(0, (2.0 - vol_ratio) / 2.0))
        res.distance_to_fire = normalize_distance(dist)
        res.eta_bars = res.distance_to_fire * 3.0 if np.isfinite(res.distance_to_fire) else None

    res.interpretation = (
        f"FIRED {res.direction} — VCP fire" if state == "FIRED" else
        f"NEAR — {n_met}/4 conditions" if state == "NEAR" else
        f"APPROACHING — {n_met}/4" if state == "APPROACHING" else
        "no setup"
    )
    return res


# =============================================================================
# A4 · SQUEEZE_FIRE_DIRECTIONAL
# =============================================================================
def detect_a4(panel: pd.DataFrame, asof_date: date,
              bp_mult: float, scope: str = "outright") -> SetupResult:
    res = SetupResult(setup_id="A4", name="SQUEEZE_FIRE_DIRECTIONAL",
                       family=FAMILY_TREND, scope=scope)
    try:
        today, history = _get_today_and_history(panel, asof_date, min_history=20)
    except ValueError as e:
        res.state = "N/A"; res.error = str(e); return res

    close = safe_float(today.get("close"))
    bb_up_t = safe_float(_ind(today, "BB_UP_20_2.0x", ["BB_UP_20_2", "BB_UP_20"]))
    bb_dn_t = safe_float(_ind(today, "BB_DN_20_2.0x", ["BB_DN_20_2", "BB_DN_20"]))
    kc_up_t = safe_float(_ind(today, "KC_UP_20_2.0x_ATR20", ["KC_UP_20_2", "KC_UP_20"]))
    kc_dn_t = safe_float(_ind(today, "KC_DN_20_2.0x_ATR20", ["KC_DN_20_2", "KC_DN_20"]))
    macd_hist = _ind(panel, "MACD_12_26_9_hist", ["MACD_12_26_9_hist", "MACD_12_26_9_hist"])
    atr = safe_float(_ind(today, "ATR14", ["ATR20"]))

    if close is None or atr is None or bb_up_t is None or kc_up_t is None:
        res.state = "N/A"; res.error = "missing BB / KC / ATR"; return res

    # Prior squeeze: BB inside KC for ≥3 bars in last 20
    bb_up = _ind(panel, "BB_UP_20_2.0x", ["BB_UP_20_2", "BB_UP_20"])
    bb_dn = _ind(panel, "BB_DN_20_2.0x", ["BB_DN_20_2", "BB_DN_20"])
    kc_up = _ind(panel, "KC_UP_20_2.0x_ATR20", ["KC_UP_20_2", "KC_UP_20"])
    kc_dn = _ind(panel, "KC_DN_20_2.0x_ATR20", ["KC_DN_20_2", "KC_DN_20"])
    prior_squeeze = False
    if (isinstance(bb_up, pd.Series) and isinstance(kc_up, pd.Series)
            and isinstance(bb_dn, pd.Series) and isinstance(kc_dn, pd.Series)):
        cnt = bb_inside_kc_counter(bb_up, bb_dn, kc_up, kc_dn, window=20)
        prior = cnt.loc[cnt.index < pd.Timestamp(asof_date)]
        prior_squeeze = bool(prior.tail(1).iloc[0] >= 3) if not prior.empty else False

    # BB exits KC today (BB_UP > KC_UP OR BB_DN < KC_DN)
    bb_exits_kc = (bb_up_t > kc_up_t) or (bb_dn_t < kc_dn_t)

    # MACD_hist >0 rising (LONG) or <0 falling (SHORT)
    if isinstance(macd_hist, pd.Series) and not macd_hist.empty:
        ts = pd.Timestamp(asof_date)
        prior_hist = macd_hist.loc[macd_hist.index < ts]
        prev_hist = safe_float(prior_hist.iloc[-1] if not prior_hist.empty else None)
        today_hist = safe_float(_ind(today, "MACD_12_26_9_hist", ["MACD_12_26_9_hist", "MACD_12_26_9_hist"]))
    else:
        prev_hist = None; today_hist = None

    macd_long = (today_hist is not None and prev_hist is not None
                  and today_hist > 0 and today_hist > prev_hist)
    macd_short = (today_hist is not None and prev_hist is not None
                   and today_hist < 0 and today_hist < prev_hist)

    long_n = _bool_to_n(prior_squeeze, bb_exits_kc, macd_long)
    short_n = _bool_to_n(prior_squeeze, bb_exits_kc, macd_short)

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
        "close": close, "BB_UP_2σ": bb_up_t, "BB_DN_2σ": bb_dn_t,
        "KC_UP_2σ": kc_up_t, "KC_DN_2σ": kc_dn_t,
        "prior_squeeze (≥3 bars BB⊂KC)": prior_squeeze,
        "BB exits KC today": bb_exits_kc,
        "MACD_hist today": today_hist, "MACD_hist prev": prev_hist,
    }
    res.thresholds = {
        "prior squeeze": "BB ⊂ KC for ≥3 bars in last 20",
        "BB exits KC": "today's BB band breaks outside KC band",
        "MACD_12_26_9_hist": "> 0 rising (LONG) / < 0 falling (SHORT)",
    }
    if res.fired_long:
        # Stop = max(2×ATR, opposite KC)
        stop = max(close - 2 * atr, kc_dn_t)
        _set_levels_long(res, close, stop, bp_mult, 1.0, 2.0)
    elif res.fired_short:
        stop = min(close + 2 * atr, kc_up_t)
        _set_levels_short(res, close, stop, bp_mult, 1.0, 2.0)

    if state in ("NEAR", "APPROACHING"):
        # crude distance — count of unmet conditions
        res.distance_to_fire = float(3 - n_met)
        res.eta_bars = res.distance_to_fire * 3.0

    res.interpretation = (
        f"FIRED {res.direction} — squeeze fire" if state == "FIRED" else
        f"NEAR — {n_met}/3 conditions" if state == "NEAR" else
        f"APPROACHING — {n_met}/3" if state == "APPROACHING" else
        "no squeeze setup"
    )
    return res


# =============================================================================
# A5 · DONCHIAN_55D_BREAKOUT  (extended to flies; skip-N/A when bars<55)
# =============================================================================
def detect_a5(panel: pd.DataFrame, asof_date: date,
              bp_mult: float, scope: str = "outright") -> SetupResult:
    res = SetupResult(setup_id="A5", name="DONCHIAN_55D_BREAKOUT",
                       family=FAMILY_TREND, scope=scope)
    try:
        today, history = _get_today_and_history(panel, asof_date, min_history=56)
    except ValueError as e:
        res.state = "N/A"; res.error = "INSUFFICIENT_HISTORY (<56 bars)"; return res

    if len(history) < 55:
        res.state = "N/A"; res.error = "INSUFFICIENT_HISTORY (<55 bars)"; return res

    close = safe_float(today.get("close"))
    atr = safe_float(_ind(today, "ATR14", ["ATR20"]))
    if close is None or atr is None:
        res.state = "N/A"; res.error = "missing close/ATR"; return res

    # Donchian-55 from CLOSE (per turtle System 2 spec)
    last55 = history["close"].tail(55)
    high_55 = safe_float(last55.max())
    low_55 = safe_float(last55.min())
    if high_55 is None or low_55 is None:
        res.state = "N/A"; res.error = "Donchian-55 unavailable"; return res

    # Opposite 20d extreme for T1
    last20 = history["close"].tail(20)
    high_20 = safe_float(last20.max())
    low_20 = safe_float(last20.min())

    fired_long = close > high_55
    fired_short = close < low_55
    state = "FIRED" if (fired_long or fired_short) else (
        "NEAR" if (atr > 0 and (
            (high_55 - close) <= 0.5 * atr or (close - low_55) <= 0.5 * atr
        )) else "FAR"
    )
    res.state = state
    res.confidence = 1.0 if state == "FIRED" else (0.6 if state == "NEAR" else 0.0)
    if fired_long:
        res.fired_long = True; res.direction = "LONG"
    elif fired_short:
        res.fired_short = True; res.direction = "SHORT"
    elif state == "NEAR":
        res.direction = "LONG" if (high_55 - close) <= (close - low_55) else "SHORT"

    res.key_inputs = {"close": close, "High_55 (close)": high_55, "Low_55 (close)": low_55}
    res.thresholds = {"FIRED LONG": "close > High_55", "FIRED SHORT": "close < Low_55",
                       "NEAR": "within 0.5 × ATR14"}
    if res.fired_long:
        _set_levels_long(res, close, close - 2 * atr, bp_mult,
                          t1_r=(low_20 - close) / max(1e-9, close - (close - 2 * atr))
                                if low_20 is not None else 1.0,
                          t2_r=None)
        # Cleaner: hardcoded T1 = low_20, T2 = None per registry spec
        res.t1 = round_to_tick(low_20, bp_mult) if low_20 is not None else res.t1
        res.t2 = None
    elif res.fired_short:
        _set_levels_short(res, close, close + 2 * atr, bp_mult, t1_r=1.0, t2_r=None)
        res.t1 = round_to_tick(high_20, bp_mult) if high_20 is not None else res.t1
        res.t2 = None

    if state == "NEAR" and atr > 0:
        d = min((high_55 - close), (close - low_55)) / atr
        res.distance_to_fire = float(max(0, d))
        res.eta_bars = res.distance_to_fire * 2.0

    res.interpretation = (
        f"FIRED {res.direction} — turtle System 2 break" if state == "FIRED" else
        "NEAR breakout (within 0.5 ATR)" if state == "NEAR" else
        "in range"
    )
    return res


# =============================================================================
# A6 · GOLDEN_DEATH_CROSS  (EMA_50 × EMA_200; EMA_200 runtime-computed)
# =============================================================================
def detect_a6(panel: pd.DataFrame, asof_date: date,
              bp_mult: float, scope: str = "outright") -> SetupResult:
    res = SetupResult(setup_id="A6", name="GOLDEN_DEATH_CROSS",
                       family=FAMILY_TREND, scope=scope)
    try:
        today, history = _get_today_and_history(panel, asof_date, min_history=205)
    except ValueError as e:
        res.state = "N/A"; res.error = str(e); return res

    close = safe_float(today.get("close"))
    ema50_today = safe_float(_ind(today, "EMA_50"))
    adx = safe_float(_ind(today, "ADX_14"))
    atr = safe_float(_ind(today, "ATR14", ["ATR20"]))
    if close is None or ema50_today is None or adx is None or atr is None:
        res.state = "N/A"; res.error = "missing indicator(s)"; return res

    # Compute EMA_200 at runtime over the panel
    if "close" not in panel.columns:
        res.state = "N/A"; res.error = "no close column"; return res
    ema200_full = ema_200(panel["close"])
    ts = pd.Timestamp(asof_date)
    prev_idx = ema200_full.index < ts
    if prev_idx.sum() < 2:
        res.state = "N/A"; res.error = "EMA_200 history insufficient"; return res
    ema200_today = safe_float(ema200_full.loc[prev_idx | (ema200_full.index.normalize() == ts.normalize())].iloc[-1])
    ema200_prev = safe_float(ema200_full.loc[prev_idx].iloc[-1])
    if ema200_today is None or ema200_prev is None:
        res.state = "N/A"; res.error = "EMA_200 NaN"; return res

    ema50_prior = _ind(panel, "EMA_50")
    if not isinstance(ema50_prior, pd.Series) or ema50_prior.empty:
        res.state = "N/A"; res.error = "EMA_50 series unavailable"; return res
    ema50_prev = safe_float(ema50_prior.loc[prev_idx].iloc[-1])
    if ema50_prev is None:
        res.state = "N/A"; res.error = "EMA_50 prev NaN"; return res

    cross_up = (ema50_prev <= ema200_prev) and (ema50_today > ema200_today)
    cross_dn = (ema50_prev >= ema200_prev) and (ema50_today < ema200_today)
    adx_ok = adx > 18.0

    long_n = _bool_to_n(cross_up, adx_ok)
    short_n = _bool_to_n(cross_dn, adx_ok)

    if cross_up and adx_ok:
        res.fired_long = True; res.direction = "LONG"; res.state = "FIRED"
    elif cross_dn and adx_ok:
        res.fired_short = True; res.direction = "SHORT"; res.state = "FIRED"
    elif cross_up or cross_dn:
        res.state = "APPROACHING"
        res.direction = "LONG" if cross_up else "SHORT"
    else:
        # NEAR if EMAs converging within 0.5 × ATR
        gap = abs(ema50_today - ema200_today)
        if gap <= 0.5 * atr:
            res.state = "NEAR"
            res.direction = "LONG" if ema50_today > ema200_today else "SHORT"
        else:
            res.state = "FAR"

    res.confidence = 1.0 if res.state == "FIRED" else (0.6 if res.state == "NEAR" else 0.0)
    res.key_inputs = {
        "close": close, "EMA_50": ema50_today, "EMA_200": ema200_today,
        "EMA_50 prev": ema50_prev, "EMA_200 prev": ema200_prev, "ADX_14": adx,
    }
    res.thresholds = {
        "EMA_50 cross EMA_200": "above (LONG) / below (SHORT) today",
        "ADX_14": "> 18",
    }
    if res.fired_long:
        _set_levels_long(res, close, close - 3 * atr, bp_mult, 1.5, None)
    elif res.fired_short:
        _set_levels_short(res, close, close + 3 * atr, bp_mult, 1.5, None)

    if res.state == "NEAR" and atr > 0:
        res.distance_to_fire = float(abs(ema50_today - ema200_today) / atr)
        res.eta_bars = res.distance_to_fire * 5.0

    res.interpretation = (
        f"FIRED {res.direction} — golden/death cross" if res.state == "FIRED" else
        f"APPROACHING — cross today, ADX≤18 (weak)" if res.state == "APPROACHING" else
        f"NEAR — EMAs converging (within 0.5 ATR)" if res.state == "NEAR" else
        "no cross"
    )
    return res


# =============================================================================
# A8 · ICHIMOKU_CLOUD_BREAKOUT
# =============================================================================
def detect_a8(panel: pd.DataFrame, asof_date: date,
              bp_mult: float, scope: str = "outright") -> SetupResult:
    res = SetupResult(setup_id="A8", name="ICHIMOKU_CLOUD_BREAKOUT",
                       family=FAMILY_TREND, scope=scope)
    try:
        today, history = _get_today_and_history(panel, asof_date, min_history=80)
    except ValueError as e:
        res.state = "N/A"; res.error = str(e); return res

    close = safe_float(today.get("close"))
    atr = safe_float(_ind(today, "ATR14", ["ATR20"]))
    if close is None or atr is None:
        res.state = "N/A"; res.error = "missing close/ATR"; return res

    # Compute Ichimoku at runtime
    if "high" not in panel.columns or "low" not in panel.columns:
        res.state = "N/A"; res.error = "no high/low"; return res
    ich = ichimoku_full(panel["high"], panel["low"], panel["close"])
    ts = pd.Timestamp(asof_date)
    try:
        tenkan_today = safe_float(ich["tenkan"].loc[ich["tenkan"].index <= ts].iloc[-1])
        kijun_today = safe_float(ich["kijun"].loc[ich["kijun"].index <= ts].iloc[-1])
        sa_today = safe_float(ich["senkou_a"].loc[ich["senkou_a"].index <= ts].iloc[-1])
        sb_today = safe_float(ich["senkou_b"].loc[ich["senkou_b"].index <= ts].iloc[-1])
    except IndexError:
        res.state = "N/A"; res.error = "Ichimoku NaN"; return res

    if any(v is None for v in (tenkan_today, kijun_today, sa_today, sb_today)):
        res.state = "N/A"; res.error = "Ichimoku NaN"; return res

    cloud_max = max(sa_today, sb_today)
    cloud_min = min(sa_today, sb_today)

    # Chikou: today's close vs price 26 bars ago (Chikou is close shifted -26;
    # for "today" comparison: close[today] vs close[today-26])
    prior26 = history["close"].tail(26)
    if len(prior26) < 26:
        res.state = "N/A"; res.error = "<26 prior bars for Chikou"; return res
    price_26_ago = safe_float(prior26.iloc[0])

    # LONG: close > cloud_max  AND  Tenkan > Kijun  AND  Chikou clears (close > price_26_ago)
    c1_long = close > cloud_max
    c2_long = tenkan_today > kijun_today
    c3_long = close > price_26_ago
    long_n = _bool_to_n(c1_long, c2_long, c3_long)

    c1_short = close < cloud_min
    c2_short = tenkan_today < kijun_today
    c3_short = close < price_26_ago
    short_n = _bool_to_n(c1_short, c2_short, c3_short)

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
        "close": close, "Tenkan_9": tenkan_today, "Kijun_26": kijun_today,
        "Senkou_A": sa_today, "Senkou_B": sb_today, "price_26_ago": price_26_ago,
    }
    res.thresholds = {
        "close vs Kumo (cloud)": "above max(SA,SB) for LONG; below min for SHORT",
        "Tenkan vs Kijun":        "Tenkan > Kijun (LONG)",
        "Chikou":                 "close > price_26_ago (LONG)",
    }
    if res.fired_long:
        _set_levels_long(res, close, kijun_today, bp_mult, 1.0, 2.0)
        # Custom T1/T2: half/full Kumo height
        kumo_h = max(0, cloud_max - cloud_min)
        res.t1 = round_to_tick(close + 0.5 * kumo_h, bp_mult)
        res.t2 = round_to_tick(close + kumo_h, bp_mult)
    elif res.fired_short:
        _set_levels_short(res, close, kijun_today, bp_mult, 1.0, 2.0)
        kumo_h = max(0, cloud_max - cloud_min)
        res.t1 = round_to_tick(close - 0.5 * kumo_h, bp_mult)
        res.t2 = round_to_tick(close - kumo_h, bp_mult)

    if state in ("NEAR", "APPROACHING") and atr > 0:
        if long_n >= short_n:
            dist = []
            if not c1_long: dist.append(max(0, (cloud_max - close) / atr))
            if not c2_long: dist.append(max(0, (kijun_today - tenkan_today) / atr))
            if not c3_long: dist.append(max(0, (price_26_ago - close) / atr))
        else:
            dist = []
            if not c1_short: dist.append(max(0, (close - cloud_min) / atr))
            if not c2_short: dist.append(max(0, (tenkan_today - kijun_today) / atr))
            if not c3_short: dist.append(max(0, (close - price_26_ago) / atr))
        res.distance_to_fire = normalize_distance(dist)
        res.eta_bars = res.distance_to_fire * 3.0

    res.interpretation = (
        f"FIRED {res.direction} — Ichimoku breakout" if state == "FIRED" else
        f"NEAR — {n_met}/3 conditions" if state == "NEAR" else
        f"APPROACHING — {n_met}/3" if state == "APPROACHING" else
        "no setup"
    )
    return res


# =============================================================================
# A10 · MA_RIBBON_ALIGNMENT
# =============================================================================
def detect_a10(panel: pd.DataFrame, asof_date: date,
               bp_mult: float, scope: str = "outright") -> SetupResult:
    res = SetupResult(setup_id="A10", name="MA_RIBBON_ALIGNMENT",
                       family=FAMILY_TREND, scope=scope)
    try:
        today, history = _get_today_and_history(panel, asof_date, min_history=110)
    except ValueError as e:
        res.state = "N/A"; res.error = str(e); return res

    emas = {p: safe_float(_ind(today, f"EMA_{p}")) for p in (5, 10, 20, 30, 50, 100)}
    if any(v is None for v in emas.values()):
        res.state = "N/A"; res.error = "missing EMA(s)"; return res
    close = safe_float(today.get("close"))
    atr = safe_float(_ind(today, "ATR14", ["ATR20"]))

    # 5 stack pairs
    pairs_long = [
        emas[5] > emas[10], emas[10] > emas[20], emas[20] > emas[30],
        emas[30] > emas[50], emas[50] > emas[100],
    ]
    pairs_short = [
        emas[5] < emas[10], emas[10] < emas[20], emas[20] < emas[30],
        emas[30] < emas[50], emas[50] < emas[100],
    ]
    long_n = _bool_to_n(*pairs_long)
    short_n = _bool_to_n(*pairs_short)

    # "First bar" condition: today is fully aligned but prior bar was NOT
    if long_n == 5 or short_n == 5:
        prior_emas = {p: safe_float(_ind(panel, f"EMA_{p}")) for p in (5, 10, 20, 30, 50, 100)}
        prior_today = {p: None for p in prior_emas}
        ts = pd.Timestamp(asof_date)
        for p, ser in prior_emas.items():
            if isinstance(ser, pd.Series) and not ser.empty:
                pri = ser.loc[ser.index < ts]
                prior_today[p] = safe_float(pri.iloc[-1] if not pri.empty else None)
        prior_aligned_long = all(
            prior_today[a] is not None and prior_today[b] is not None and prior_today[a] > prior_today[b]
            for a, b in [(5, 10), (10, 20), (20, 30), (30, 50), (50, 100)]
        ) if all(v is not None for v in prior_today.values()) else False
        prior_aligned_short = all(
            prior_today[a] is not None and prior_today[b] is not None and prior_today[a] < prior_today[b]
            for a, b in [(5, 10), (10, 20), (20, 30), (30, 50), (50, 100)]
        ) if all(v is not None for v in prior_today.values()) else False
        first_bar_long = (long_n == 5) and (not prior_aligned_long)
        first_bar_short = (short_n == 5) and (not prior_aligned_short)
    else:
        first_bar_long = first_bar_short = False

    n_met = max(long_n, short_n)
    state = state_from_conditions_met(n_met, 5)
    res.state = state
    res.confidence = n_met / 5.0
    if first_bar_long:
        res.fired_long = True; res.direction = "LONG"
    elif first_bar_short:
        res.fired_short = True; res.direction = "SHORT"
    elif n_met == 5:
        res.state = "APPROACHING"   # already aligned, not "first bar"
        res.direction = "LONG" if long_n == 5 else "SHORT"
    elif state in ("NEAR", "APPROACHING"):
        res.direction = "LONG" if long_n >= short_n else "SHORT"

    res.key_inputs = {**{f"EMA_{p}": emas[p] for p in (5, 10, 20, 30, 50, 100)},
                       "long stack pairs met": long_n, "short stack pairs met": short_n,
                       "first bar (LONG)": first_bar_long, "first bar (SHORT)": first_bar_short}
    res.thresholds = {
        "stack alignment (5 pairs)": "EMA_5>10>20>30>50>100 (LONG) or reverse (SHORT)",
        "first bar": "today aligned AND prior bar NOT aligned",
    }
    if res.fired_long and atr is not None:
        _set_levels_long(res, close, emas[50], bp_mult, 1.5, None)
    elif res.fired_short and atr is not None:
        _set_levels_short(res, close, emas[50], bp_mult, 1.5, None)

    if state in ("NEAR", "APPROACHING"):
        res.distance_to_fire = float(5 - n_met)
        res.eta_bars = res.distance_to_fire * 3.0

    res.interpretation = (
        f"FIRED {res.direction} — fresh full ribbon alignment" if state == "FIRED" else
        f"already aligned ({long_n}/{short_n} pairs) — APPROACHING" if state == "APPROACHING" else
        f"NEAR — {n_met}/5 stack pairs" if state == "NEAR" else
        "no alignment"
    )
    return res


# =============================================================================
# A11 · SUPERTREND_DIRECTIONAL
# =============================================================================
def detect_a11(panel: pd.DataFrame, asof_date: date,
               bp_mult: float, scope: str = "outright") -> SetupResult:
    res = SetupResult(setup_id="A11", name="SUPERTREND_DIRECTIONAL",
                       family=FAMILY_TREND, scope=scope)
    try:
        today, history = _get_today_and_history(panel, asof_date, min_history=20)
    except ValueError as e:
        res.state = "N/A"; res.error = str(e); return res

    close = safe_float(today.get("close"))
    st_lb = safe_float(_ind(today, "ST_LB_ATR10_2.0x", ["ST_LB_ATR10_2", "ST_LB_2.0", "ST_LB_2"]))
    st_ub = safe_float(_ind(today, "ST_UB_ATR10_2.0x", ["ST_UB_ATR10_2", "ST_UB_2.0", "ST_UB_2"]))
    adx = safe_float(_ind(today, "ADX_14"))
    atr = safe_float(_ind(today, "ATR14", ["ATR20"]))
    if close is None or st_lb is None or st_ub is None or adx is None or atr is None:
        res.state = "N/A"; res.error = "missing ST/ADX/ATR"; return res

    # Cross detection — need prior bar
    ts = pd.Timestamp(asof_date)
    st_lb_full = _ind(panel, "ST_LB_ATR10_2.0x", ["ST_LB_ATR10_2", "ST_LB_2.0"])
    st_ub_full = _ind(panel, "ST_UB_ATR10_2.0x", ["ST_UB_ATR10_2", "ST_UB_2.0"])
    close_full = panel["close"]
    prior = close_full.index < ts
    if prior.sum() < 1:
        res.state = "N/A"; res.error = "no prior bar"; return res
    prev_close = safe_float(close_full.loc[prior].iloc[-1])
    prev_lb = safe_float(st_lb_full.loc[prior].iloc[-1] if isinstance(st_lb_full, pd.Series) and not st_lb_full.empty else None)
    prev_ub = safe_float(st_ub_full.loc[prior].iloc[-1] if isinstance(st_ub_full, pd.Series) and not st_ub_full.empty else None)

    cross_up = (prev_close is not None and prev_lb is not None
                and prev_close <= prev_lb and close > st_lb)
    cross_dn = (prev_close is not None and prev_ub is not None
                and prev_close >= prev_ub and close < st_ub)
    adx_ok = adx > 18.0

    if cross_up and adx_ok:
        res.fired_long = True; res.direction = "LONG"; res.state = "FIRED"
    elif cross_dn and adx_ok:
        res.fired_short = True; res.direction = "SHORT"; res.state = "FIRED"
    elif cross_up or cross_dn:
        res.state = "APPROACHING"
        res.direction = "LONG" if cross_up else "SHORT"
    else:
        # NEAR if close within 0.3 ATR of either ST band
        d_long = max(0, st_lb - close)
        d_short = max(0, close - st_ub)
        d_min = min(d_long, d_short)
        if atr > 0 and d_min / atr <= 0.5:
            res.state = "NEAR"
            res.direction = "LONG" if d_long < d_short else "SHORT"
        else:
            res.state = "FAR"

    res.confidence = 1.0 if res.state == "FIRED" else (0.6 if res.state == "NEAR" else 0.0)
    res.key_inputs = {"close": close, "ST_LB_2σ": st_lb, "ST_UB_2σ": st_ub,
                       "prev_close": prev_close, "prev_LB": prev_lb, "prev_UB": prev_ub,
                       "ADX_14": adx}
    res.thresholds = {
        "close vs ST band": "cross above LB (LONG) / below UB (SHORT)",
        "ADX_14":            "> 18",
    }
    if res.fired_long:
        _set_levels_long(res, close, st_ub, bp_mult, 2.0, None)
    elif res.fired_short:
        _set_levels_short(res, close, st_lb, bp_mult, 2.0, None)

    if res.state == "NEAR" and atr > 0:
        res.distance_to_fire = float(min(max(0, st_lb - close), max(0, close - st_ub)) / atr)
        res.eta_bars = res.distance_to_fire * 3.0

    res.interpretation = (
        f"FIRED {res.direction} — Supertrend regime change" if res.state == "FIRED" else
        f"APPROACHING — cross today, ADX≤18 (weak)" if res.state == "APPROACHING" else
        f"NEAR ST band" if res.state == "NEAR" else
        "no setup"
    )
    return res


# =============================================================================
# A12a · ADX_DI_CROSS · EMA_20 variant
# A12b · ADX_DI_CROSS · EMA_50 variant
# =============================================================================
def _detect_a12_variant(panel: pd.DataFrame, asof_date: date,
                         bp_mult: float, scope: str,
                         setup_id: str, name: str, ema_period: int) -> SetupResult:
    res = SetupResult(setup_id=setup_id, name=name, family=FAMILY_TREND, scope=scope)
    try:
        today, history = _get_today_and_history(panel, asof_date, min_history=ema_period + 5)
    except ValueError as e:
        res.state = "N/A"; res.error = str(e); return res

    close = safe_float(today.get("close"))
    di_p = safe_float(_ind(today, "DIplus_14", ["DIplus_14", "DIp14", "DI_PLUS_14"]))
    di_m = safe_float(_ind(today, "DIminus_14", ["DIminus_14", "DIn14", "DI_MINUS_14"]))
    adx = safe_float(_ind(today, "ADX_14"))
    ema = safe_float(_ind(today, f"EMA_{ema_period}"))
    atr = safe_float(_ind(today, "ATR14", ["ATR20"]))

    if any(v is None for v in (close, di_p, di_m, adx, ema, atr)):
        res.state = "N/A"; res.error = "missing indicator(s)"; return res

    # Crossing — need prior DI+/DI-
    ts = pd.Timestamp(asof_date)
    di_p_full = _ind(panel, "DIplus_14", ["DIplus_14", "DIp14"])
    di_m_full = _ind(panel, "DIminus_14", ["DIminus_14", "DIn14"])
    prior = di_p_full.index < ts if isinstance(di_p_full, pd.Series) else None
    prev_p = safe_float(di_p_full.loc[prior].iloc[-1]) if prior is not None and prior.any() else None
    prev_m = safe_float(di_m_full.loc[prior].iloc[-1]) if prior is not None and prior.any() else None

    cross_up = prev_p is not None and prev_m is not None and prev_p <= prev_m and di_p > di_m
    cross_dn = prev_p is not None and prev_m is not None and prev_p >= prev_m and di_p < di_m
    adx_ok = adx > 20.0

    long_n = _bool_to_n(cross_up, adx_ok, close > ema)
    short_n = _bool_to_n(cross_dn, adx_ok, close < ema)
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
        "close": close, "DI+": di_p, "DI-": di_m, "ADX_14": adx, f"EMA_{ema_period}": ema,
        "prev DI+": prev_p, "prev DI-": prev_m,
    }
    res.thresholds = {
        "DI cross today": "DI+ crosses above DI- (LONG) / below (SHORT)",
        "ADX_14":          "> 20",
        f"close vs EMA_{ema_period}": "> for LONG / < for SHORT",
    }
    if res.fired_long:
        _set_levels_long(res, close, close - 2 * atr, bp_mult, 1.5, None)
    elif res.fired_short:
        _set_levels_short(res, close, close + 2 * atr, bp_mult, 1.5, None)

    if state in ("NEAR", "APPROACHING"):
        res.distance_to_fire = float(3 - n_met)
        res.eta_bars = res.distance_to_fire * 3.0

    res.interpretation = (
        f"FIRED {res.direction} — DI cross + ADX + MA confirmation" if state == "FIRED" else
        f"NEAR — {n_met}/3" if state == "NEAR" else
        f"APPROACHING — {n_met}/3" if state == "APPROACHING" else
        "no setup"
    )
    return res


def detect_a12a(panel: pd.DataFrame, asof_date: date,
                bp_mult: float, scope: str = "outright") -> SetupResult:
    return _detect_a12_variant(panel, asof_date, bp_mult, scope,
                                "A12a", "ADX_DI_CROSS · EMA_20 variant", 20)


def detect_a12b(panel: pd.DataFrame, asof_date: date,
                bp_mult: float, scope: str = "outright") -> SetupResult:
    return _detect_a12_variant(panel, asof_date, bp_mult, scope,
                                "A12b", "ADX_DI_CROSS · EMA_50 variant", 50)


# =============================================================================
# A15 · TRIANGLE_WEDGE
# =============================================================================
def detect_a15(panel: pd.DataFrame, asof_date: date,
               bp_mult: float, scope: str = "outright") -> SetupResult:
    res = SetupResult(setup_id="A15", name="TRIANGLE_WEDGE",
                       family=FAMILY_TREND, scope=scope)
    try:
        today, history = _get_today_and_history(panel, asof_date, min_history=40)
    except ValueError as e:
        res.state = "N/A"; res.error = str(e); return res

    close = safe_float(today.get("close"))
    atr = safe_float(_ind(today, "ATR14", ["ATR20"]))
    if close is None or atr is None:
        res.state = "N/A"; res.error = "missing close/ATR"; return res
    if "high" not in panel.columns or "low" not in panel.columns:
        res.state = "N/A"; res.error = "no high/low"; return res

    # Detect swings on the entire panel (excluding today)
    sw = swing_detect(panel["high"], panel["low"], lookback=5)
    sw_prior = sw.loc[sw.index < pd.Timestamp(asof_date)]
    swing_high_idx_dt = sw_prior.index[sw_prior["swing_high"] > 0]
    swing_low_idx_dt = sw_prior.index[sw_prior["swing_low"] > 0]
    if len(swing_high_idx_dt) < 4 or len(swing_low_idx_dt) < 4:
        res.state = "APPROACHING"; res.error = "<4 swings each side — pattern setup forming"
        res.key_inputs = {"swing_highs": int(len(swing_high_idx_dt)),
                           "swing_lows": int(len(swing_low_idx_dt))}
        res.interpretation = "pattern setup forming — insufficient swings"
        return res

    # Map dates to integer x-axis
    full_dates = panel.index
    date_to_idx = {d: i for i, d in enumerate(full_dates)}
    h_idx = [date_to_idx[d] for d in swing_high_idx_dt]
    l_idx = [date_to_idx[d] for d in swing_low_idx_dt]
    h_val = panel.loc[swing_high_idx_dt, "high"].astype(float).tolist()
    l_val = panel.loc[swing_low_idx_dt, "low"].astype(float).tolist()
    fit = polyfit_pattern_lines(h_idx, h_val, l_idx, l_val, min_points=4)
    if fit["upper_slope"] is None or fit["lower_slope"] is None:
        res.state = "N/A"; res.error = "polyfit failed"; return res

    today_idx = len(full_dates) - 1   # last bar in panel
    upper_proj = fit["upper_slope"] * today_idx + fit["upper_intercept"]
    lower_proj = fit["lower_slope"] * today_idx + fit["lower_intercept"]

    converging = bool(fit["is_converging"])
    breakout_long = converging and close > upper_proj
    breakout_short = converging and close < lower_proj

    if breakout_long:
        res.fired_long = True; res.direction = "LONG"; res.state = "FIRED"
    elif breakout_short:
        res.fired_short = True; res.direction = "SHORT"; res.state = "FIRED"
    elif converging:
        res.state = "NEAR"
        # which side closer
        d_up = max(0, upper_proj - close)
        d_dn = max(0, close - lower_proj)
        res.direction = "LONG" if d_up < d_dn else "SHORT"
    else:
        res.state = "APPROACHING"

    pattern_height = max(0, upper_proj - lower_proj)
    res.confidence = 1.0 if res.state == "FIRED" else (0.5 if res.state == "NEAR" else 0.2)
    res.key_inputs = {"close": close, "upper trendline @ today": upper_proj,
                       "lower trendline @ today": lower_proj,
                       "n upper swings": fit["n_upper"], "n lower swings": fit["n_lower"],
                       "is converging": converging, "pattern height": pattern_height,
                       "upper R²": fit["upper_r2"], "lower R²": fit["lower_r2"]}
    res.thresholds = {
        "converging": "upper slope < 0 AND lower slope > 0 (or strictly opposite)",
        "breakout":   "close > upper (LONG) or close < lower (SHORT)",
    }
    if res.fired_long:
        _set_levels_long(res, close, lower_proj, bp_mult,
                          t1_r=(0.5 * pattern_height) / max(1e-9, close - lower_proj) if pattern_height > 0 else 1.0,
                          t2_r=pattern_height / max(1e-9, close - lower_proj) if pattern_height > 0 else 2.0)
    elif res.fired_short:
        _set_levels_short(res, close, upper_proj, bp_mult,
                          t1_r=(0.5 * pattern_height) / max(1e-9, upper_proj - close) if pattern_height > 0 else 1.0,
                          t2_r=pattern_height / max(1e-9, upper_proj - close) if pattern_height > 0 else 2.0)

    if res.state == "NEAR" and atr > 0:
        res.distance_to_fire = float(min(max(0, upper_proj - close),
                                          max(0, close - lower_proj)) / atr)
        res.eta_bars = res.distance_to_fire * 3.0

    res.interpretation = (
        f"FIRED {res.direction} — triangle/wedge breakout" if res.state == "FIRED" else
        "NEAR — converging, await breakout" if res.state == "NEAR" else
        "pattern setup forming"
    )
    return res
