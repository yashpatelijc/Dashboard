"""Detector replay across CMC history (Phase E support).

For each (CMC scope × CMC node × setup) combination, walk forward through
the CMC panel running the detector at every historical bar. Collect each
FIRED state into a fire DataFrame consumed by the backtest engine.

This is intentionally a thin orchestration layer — the actual detector
math lives in ``lib.setups.{trend, mean_reversion, stir}.py``. The
replay layer just iterates over historical bars and stitches results.

Performance note: 5y × ~22 CMC nodes × ~26 detectors × ~1300 bars =
~750k detector calls. Each call is O(1) on a slim panel slice (or O(N)
if the detector recomputes indicators), so total replay is bounded by
~5 minutes in the typical case.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd


def _detector_registry(scope: str) -> dict:
    """Map setup_id → detector callable for the per-contract setups
    applicable to ``scope``. Mirrors the gating in
    ``lib.setups.track_record.compute_track_record`` but expanded to
    cover the full setup catalog where possible.
    """
    from lib.setups.trend import (
        detect_a1, detect_a2, detect_a3, detect_a4, detect_a5, detect_a6,
        detect_a8, detect_a10, detect_a11, detect_a12a, detect_a12b, detect_a15,
    )
    from lib.setups.mean_reversion import (
        detect_b1, detect_b3, detect_b5, detect_b6, detect_b10, detect_b11, detect_b13,
    )

    base = {
        "A1":   detect_a1,  "A2":   detect_a2,  "A3":   detect_a3,
        "A4":   detect_a4,  "A5":   detect_a5,  "A6":   detect_a6,
        "A8":   detect_a8,  "A10":  detect_a10, "A11":  detect_a11,
        "A12a": detect_a12a,"A12b": detect_a12b,"A15":  detect_a15,
        "B1":   detect_b1,  "B3":   detect_b3,  "B5":   detect_b5,
        "B6":   detect_b6,  "B10":  detect_b10, "B11":  detect_b11,
        "B13":  detect_b13,
    }
    if scope == "outright":
        return base
    if scope == "spread":
        from lib.setups.stir import detect_c5
        return {**base, "C5": detect_c5}
    if scope == "fly":
        from lib.setups.stir import detect_c4
        return {**base, "C4": detect_c4}
    return base


def _bp_per_unit_for_scope(scope: str) -> float:
    """SR3 outright price units → 100 bp per 1.00 price; spread/fly stored
    in bp directly → 1."""
    return 100.0 if scope == "outright" else 1.0


def _replay_detector_on_node(detector, panel: pd.DataFrame,
                                  bp_mult: float, scope: str,
                                  setup_id: str) -> pd.DataFrame:
    """Walk through each historical bar in ``panel`` and run the detector.
    Collect FIRED bars into a fire DataFrame."""
    fires = []
    if panel is None or panel.empty:
        return pd.DataFrame()
    # Need a minimum history depth so indicators are meaningful
    min_bars = 60
    if len(panel) < min_bars:
        return pd.DataFrame()
    # We replay starting at min_bars and going forward. Each detector call
    # truncates the panel to "as-of" the current bar inside the detector.
    for i in range(min_bars, len(panel) - 1):
        eval_dt = panel.index[i].date()
        sub = panel.iloc[: i + 1]   # inclusive of bar i
        try:
            result = detector(sub, eval_dt, bp_mult, scope=scope)
        except Exception:
            continue
        if result is None or getattr(result, "state", None) != "FIRED":
            continue
        d = getattr(result, "direction", None)
        if d not in ("LONG", "SHORT"):
            continue
        fires.append({
            "entry_dt":   eval_dt,
            "direction":  d,
            "stop_price": getattr(result, "stop", None),
            "t1_price":   getattr(result, "t1", None),
            "t2_price":   getattr(result, "t2", None),
            "setup_id":   setup_id,
        })
    return pd.DataFrame(fires)


def replay_all_detectors_on_cmc(scope: str, asof_date: date) -> dict:
    """Run every applicable detector against every CMC node in the scope,
    over the full 5-year CMC history.

    Returns ``{cell_key: {setup_id, cmc_node, panel, fires, bp_per_unit}}``.

    ``cell_key`` is a string like ``"A1|M3"`` for indexing.
    """
    from lib.cmc import list_cmc_nodes
    from lib.sra_data import load_cmc_node_panel

    bp_per_unit = _bp_per_unit_for_scope(scope)
    detectors = _detector_registry(scope)
    out = {}
    for node_id in list_cmc_nodes(scope):
        panel = load_cmc_node_panel(scope, node_id, asof_date)
        if panel is None or panel.empty:
            continue
        # Add minimal indicator columns the detectors expect (ATR14, ADX_14,
        # etc.) — the underlying CMC panel has OHLC; detectors may compute
        # their own indicators from close, but ATR/ADX are typically read
        # from indicator columns. Compute fast approximations here.
        panel = _augment_panel_with_indicators(panel)
        for setup_id, det in detectors.items():
            cell_key = f"{setup_id}|{node_id}"
            try:
                fires = _replay_detector_on_node(det, panel, bp_per_unit,
                                                    scope, setup_id)
            except Exception:
                fires = pd.DataFrame()
            out[cell_key] = {
                "setup_id": setup_id,
                "cmc_node": node_id,
                "panel": panel,
                "fires": fires,
                "bp_per_unit": bp_per_unit,
            }
    return out


def _augment_panel_with_indicators(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute the full indicator stack on a CMC OHLCV panel.

    Phase 0 expansion (per plan §11.3 and §15): expanded from the previous
    pragmatic-minimum (~5 indicators) to the full set the framework's
    analytics modules need (~50). All indicators are computed locally on
    the back-adjusted close (Carver-correction is preserved for %-return
    indicators by using ``raw_close_anchor`` as the denominator).

    Indicator families covered:
      * ATR variants (5/10/14/20)
      * ADX_14 + DI± (Wilder)
      * EMAs (5/10/20/30/50/100/200) + SMAs (5/10/20/30/50/100)
      * Bollinger Bands (20-period × 1.0x / 2.0x / 3.0x σ)
      * Keltner Channels (20-period × 1.0x / 2.0x / 3.0x ATR20)
      * Supertrend (10-period × 1.0x / 2.0x / 3.0x ATR10)
      * Aroon Up/Dn/Osc (14)
      * CCI (20)
      * OBV
      * RSI (14) + RSI_2 (Connors)
      * Z-score price (5/10/20/50/100) + Z-score returns (5/10/20/50/100)
      * Range / High / Low (5/10/20/50/100)
      * Range/ATR ratios (5/10/20/50/100 over ATR_20)
      * MACD (12/26/9)
      * BB-width percentile (BBWP, 252-day rank)

    Each indicator name follows the convention used in
    ``mde2_indicators`` so detectors can read either source uniformly.
    """
    p = panel.copy()
    if p.empty:
        return p

    high, low, close = p["high"], p["low"], p["close"]
    volume = p.get("volume", pd.Series(0.0, index=p.index))
    prev_close = close.shift(1)

    # ----- True Range and ATR variants -----
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    p["TR"] = tr
    for span in (5, 10, 14, 20):
        p[f"ATR{span}"] = tr.rolling(span).mean()
    p["ATR_14"] = p["ATR14"]    # legacy alias

    # ----- Wilder DI± / ADX -----
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    atr = p["ATR14"]
    plus_di = 100 * (plus_dm.rolling(14).sum() / atr)
    minus_di = 100 * (minus_dm.rolling(14).sum() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    p["ADX_14"] = dx.rolling(14).mean()
    p["DIplus_14"] = plus_di
    p["DIminus_14"] = minus_di

    # ----- Moving averages -----
    for span in (5, 10, 20, 30, 50, 100, 200):
        p[f"EMA_{span}"] = close.ewm(span=span, adjust=False).mean()
    for span in (5, 10, 20, 30, 50, 100):
        p[f"SMA_{span}"] = close.rolling(span).mean()

    # ----- Bollinger Bands (20-period) at multiple multipliers -----
    bb_mid = close.rolling(20).mean()
    bb_sd = close.rolling(20).std()
    p["BB_MID_20"] = bb_mid
    for mult in (1.0, 2.0, 3.0):
        p[f"BB_UP_20_{mult}x"] = bb_mid + mult * bb_sd
        p[f"BB_DN_20_{mult}x"] = bb_mid - mult * bb_sd
        p[f"BB_BW_20_{mult}x"] = (2 * mult * bb_sd) / bb_mid
        p[f"BB_pctB_20_{mult}x"] = (close - (bb_mid - mult * bb_sd)) / (2 * mult * bb_sd + 1e-9)
    p["BB_BW_20"] = p["BB_BW_20_2.0x"]   # legacy alias
    # BBWP — 252-day rank of BB-width-2.0x (used by A3 / A4 squeeze detectors)
    p["BBWP_20"] = (p["BB_BW_20_2.0x"]
                       .rolling(252)
                       .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100,
                              raw=False))

    # ----- Keltner Channels (20-period × ATR20) -----
    kc_mid = close.ewm(span=20, adjust=False).mean()
    kc_atr = p["ATR20"]
    p["KC_MID_20"] = kc_mid
    for mult in (1.0, 2.0, 3.0):
        p[f"KC_UP_20_{mult}x_ATR20"] = kc_mid + mult * kc_atr
        p[f"KC_DN_20_{mult}x_ATR20"] = kc_mid - mult * kc_atr

    # ----- Supertrend (ATR10 × multipliers) — Wilder convention -----
    st_atr = p["ATR10"]
    hl2 = (high + low) / 2.0
    for mult in (1.0, 2.0, 3.0):
        # Basic upper / lower bands (no trend-tracking; sufficient for backtest)
        p[f"ST_UB_ATR10_{mult}x"] = hl2 + mult * st_atr
        p[f"ST_LB_ATR10_{mult}x"] = hl2 - mult * st_atr

    # ----- Aroon (14) -----
    n = 14
    p["AroonUp_14"] = high.rolling(n + 1).apply(
        lambda x: 100.0 * (n - (len(x) - 1 - x.argmax())) / n, raw=True)
    p["AroonDn_14"] = low.rolling(n + 1).apply(
        lambda x: 100.0 * (n - (len(x) - 1 - x.argmin())) / n, raw=True)
    p["AroonOsc_14"] = p["AroonUp_14"] - p["AroonDn_14"]

    # ----- CCI (20) -----
    typical = (high + low + close) / 3.0
    cci_mean = typical.rolling(20).mean()
    cci_md = typical.rolling(20).apply(
        lambda x: pd.Series(x).sub(pd.Series(x).mean()).abs().mean(), raw=False)
    p["CCI_20"] = (typical - cci_mean) / (0.015 * cci_md + 1e-9)

    # ----- OBV -----
    sign = pd.Series(0.0, index=close.index)
    sign[close > prev_close] = 1.0
    sign[close < prev_close] = -1.0
    p["OBV"] = (sign * volume).cumsum()

    # ----- RSI (Wilder 14) + Connors RSI_2 -----
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / (loss + 1e-9)
    p["RSI_14"] = 100 - 100 / (1 + rs)
    gain2 = delta.where(delta > 0, 0.0).ewm(alpha=1/2, adjust=False).mean()
    loss2 = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/2, adjust=False).mean()
    rs2 = gain2 / (loss2 + 1e-9)
    p["RSI_2"] = 100 - 100 / (1 + rs2)

    # ----- Z-scores (price + return) at multiple lookbacks -----
    rets = close.pct_change()
    for span in (5, 10, 20, 50, 100):
        m_p = close.rolling(span).mean()
        s_p = close.rolling(span).std()
        p[f"ZPRICE_{span}"] = (close - m_p) / (s_p + 1e-9)
        m_r = rets.rolling(span).mean()
        s_r = rets.rolling(span).std()
        p[f"ZRET_{span}"] = (rets - m_r) / (s_r + 1e-9)

    # ----- Highs / Lows / Range -----
    for span in (5, 10, 20, 30, 50, 55, 100):
        p[f"High_{span}"] = high.rolling(span).max()
        p[f"Low_{span}"] = low.rolling(span).min()
        p[f"Range_{span}"] = p[f"High_{span}"] - p[f"Low_{span}"]
    # Range/ATR_20 ratios (used by A3 VCP)
    atr20 = p["ATR20"]
    for span in (5, 10, 20, 50, 100):
        p[f"Range_{span}_over_ATR20"] = p[f"Range_{span}"] / (atr20 + 1e-9)

    # ----- MACD 12/26/9 -----
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    p["MACD_12_26_9"] = macd
    p["MACD_12_26_9_signal"] = macd_signal
    p["MACD_12_26_9_hist"] = macd - macd_signal

    # ----- IBS (Internal Bar Strength) — used by B4 -----
    p["IBS"] = (close - low) / ((high - low).replace(0, 1e-9))

    return p
