"""Runtime indicators not present in ``mde2_indicators``.

Every helper is pure-function over a pandas Series / DataFrame and returns
the indicator series aligned to the input index. NaNs are propagated where
insufficient observations exist (no synthetic fill).

Locked conventions (match the rest of the project):
  · history excludes today (caller responsibility — these helpers compute on
    whatever series is passed in)
  · ``ddof=1`` for sample stats
  · numpy/pandas only — no scipy / statsmodels in the hot path
  · all functions tolerate short series gracefully
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# =============================================================================
# Wilder smoothing (used by EMA_200 fallback and friends)
# =============================================================================
def wilder_ema(series: pd.Series, period: int) -> pd.Series:
    """Wilder smoothing — equivalent to EMA with alpha = 1/period.

    Used because the ``mde2_indicators`` table stops at EMA_100; Wilder EMA_200
    is needed by setup A6 (Golden / Death cross with EMA_200).
    """
    if series is None or series.empty:
        return pd.Series(dtype=float)
    alpha = 1.0 / max(1, period)
    return series.ewm(alpha=alpha, adjust=False, min_periods=max(2, period // 2)).mean()


def ema_200(close: pd.Series) -> pd.Series:
    """EMA-200 used by A6. Wilder smoothing on close."""
    return wilder_ema(close, period=200)


# =============================================================================
# RSI / RSI_2
# =============================================================================
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI. Used as a fallback when the DB column isn't present at the
    requested period (e.g. RSI_2 for Connors)."""
    if close is None or close.empty:
        return pd.Series(dtype=float)
    diff = close.diff()
    up = diff.clip(lower=0.0)
    dn = (-diff.clip(upper=0.0))
    avg_up = wilder_ema(up, period)
    avg_dn = wilder_ema(dn, period)
    rs = avg_up / avg_dn.replace(0.0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def rsi_2(close: pd.Series) -> pd.Series:
    """Connors RSI(2)."""
    return rsi(close, period=2)


# =============================================================================
# Ichimoku (full)
# =============================================================================
def ichimoku_full(high: pd.Series, low: pd.Series, close: pd.Series,
                  tenkan_period: int = 9, kijun_period: int = 26,
                  senkou_b_period: int = 52,
                  chikou_displacement: int = 26) -> dict:
    """Full Ichimoku Kinko Hyo:

    - tenkan = (max(high, 9) + min(low, 9)) / 2
    - kijun  = (max(high, 26) + min(low, 26)) / 2
    - senkou_a = (tenkan + kijun) / 2  shifted +26
    - senkou_b = (max(high, 52) + min(low, 52)) / 2  shifted +26
    - chikou = close shifted -26
    """
    out = {"tenkan": pd.Series(dtype=float), "kijun": pd.Series(dtype=float),
           "senkou_a": pd.Series(dtype=float), "senkou_b": pd.Series(dtype=float),
           "chikou": pd.Series(dtype=float)}
    if high is None or low is None or close is None or close.empty:
        return out
    n = len(close)
    if n < max(tenkan_period, kijun_period, senkou_b_period):
        return out
    h = high; l = low
    tenkan = (h.rolling(tenkan_period).max() + l.rolling(tenkan_period).min()) / 2.0
    kijun = (h.rolling(kijun_period).max() + l.rolling(kijun_period).min()) / 2.0
    senkou_a = ((tenkan + kijun) / 2.0).shift(chikou_displacement)
    senkou_b = ((h.rolling(senkou_b_period).max()
                  + l.rolling(senkou_b_period).min()) / 2.0).shift(chikou_displacement)
    chikou = close.shift(-chikou_displacement)
    return {"tenkan": tenkan, "kijun": kijun,
            "senkou_a": senkou_a, "senkou_b": senkou_b,
            "chikou": chikou}


# =============================================================================
# Internal Bar Strength
# =============================================================================
def ibs(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """IBS = (close − low) / (high − low). NaN where high == low (degenerate bar)."""
    if high is None or low is None or close is None or close.empty:
        return pd.Series(dtype=float)
    rng = (high - low).replace(0.0, np.nan)
    return ((close - low) / rng).clip(0.0, 1.0)


# =============================================================================
# Volume / TR z-scores (rolling)
# =============================================================================
def vol_zscore(volume: pd.Series, window: int = 90) -> pd.Series:
    """Rolling z-score of volume over ``window`` bars."""
    if volume is None or volume.empty:
        return pd.Series(dtype=float)
    mean = volume.rolling(window, min_periods=max(5, window // 4)).mean()
    std = volume.rolling(window, min_periods=max(5, window // 4)).std(ddof=1)
    return (volume - mean) / std.replace(0.0, np.nan)


def tr_zscore(tr: pd.Series, window: int = 90) -> pd.Series:
    """Rolling z-score of true range over ``window`` bars."""
    if tr is None or tr.empty:
        return pd.Series(dtype=float)
    mean = tr.rolling(window, min_periods=max(5, window // 4)).mean()
    std = tr.rolling(window, min_periods=max(5, window // 4)).std(ddof=1)
    return (tr - mean) / std.replace(0.0, np.nan)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True range = max(high-low, |high-prev_close|, |low-prev_close|)."""
    if high is None or low is None or close is None or close.empty:
        return pd.Series(dtype=float)
    prev_c = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_c).abs()
    tr3 = (low - prev_c).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


# =============================================================================
# Donchian channels
# =============================================================================
def donchian_high_low(close: pd.Series, period: int = 55) -> tuple:
    """Donchian channel: rolling max/min of close over ``period`` bars.

    Returns (high_n, low_n). Per setup A5 spec, computed from CLOSE — not
    high/low — to match the gameplan §6.2 'turtle System 2' definition which
    uses close-of-bar comparisons.
    """
    if close is None or close.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    high_n = close.rolling(period, min_periods=max(2, period // 2)).max()
    low_n = close.rolling(period, min_periods=max(2, period // 2)).min()
    return high_n, low_n


# =============================================================================
# Swing detection (fractal)
# =============================================================================
def swing_detect(high: pd.Series, low: pd.Series, lookback: int = 5) -> pd.DataFrame:
    """Strict fractal swing detection.

    A bar is a swing-HIGH if its high is strictly greater than the highs of
    the ``lookback`` bars on either side. Symmetric for swing-LOW.

    Returns DataFrame indexed same as input with columns:
      ``swing_high`` (1 where bar is a swing-high, 0 otherwise)
      ``swing_low``  (1 where bar is a swing-low, 0 otherwise)
    """
    out = pd.DataFrame(index=high.index, columns=["swing_high", "swing_low"], dtype=float)
    out[:] = 0.0
    if high is None or low is None or len(high) < 2 * lookback + 1:
        return out
    n = len(high)
    h_arr = high.values
    l_arr = low.values
    for i in range(lookback, n - lookback):
        center_h = h_arr[i]
        center_l = l_arr[i]
        left_h = h_arr[i - lookback:i]
        right_h = h_arr[i + 1:i + 1 + lookback]
        left_l = l_arr[i - lookback:i]
        right_l = l_arr[i + 1:i + 1 + lookback]
        if (center_h > left_h).all() and (center_h > right_h).all():
            out.iat[i, 0] = 1.0
        if (center_l < left_l).all() and (center_l < right_l).all():
            out.iat[i, 1] = 1.0
    return out


# =============================================================================
# RSI divergence (bullish / bearish) — single-bar detection in last K bars
# =============================================================================
def rsi_divergence_today(close: pd.Series, rsi_series: pd.Series,
                          lookback_bars: int = 20,
                          swing_lookback: int = 3) -> dict:
    """Bullish / bearish RSI divergence detection at today's bar.

    Bullish divergence: most-recent close is the lowest in ``lookback_bars``,
                        AND most-recent RSI is HIGHER than RSI at the prior
                        local-low close in the lookback window.
    Bearish divergence: most-recent close is the highest in ``lookback_bars``,
                        AND most-recent RSI is LOWER than RSI at the prior
                        local-high close in the lookback window.

    Returns ``{bullish: bool, bearish: bool, prior_close_idx, today_idx}``.
    """
    out = {"bullish": False, "bearish": False,
           "prior_close": None, "today_close": None,
           "prior_rsi": None, "today_rsi": None}
    if close is None or rsi_series is None or close.empty or rsi_series.empty:
        return out
    if len(close) < lookback_bars + 1:
        return out
    window_close = close.tail(lookback_bars)
    window_rsi = rsi_series.tail(lookback_bars)
    today_c = window_close.iloc[-1]
    today_r = window_rsi.iloc[-1]
    if pd.isna(today_c) or pd.isna(today_r):
        return out
    # prior bars (excluding today)
    prior_c = window_close.iloc[:-1]
    prior_r = window_rsi.iloc[:-1]
    if prior_c.dropna().empty:
        return out
    # Bullish: today is lowest close, but RSI is higher than RSI at the *previous lowest*
    prev_low_idx = prior_c.idxmin()
    if pd.notna(prev_low_idx):
        prev_low_c = prior_c.loc[prev_low_idx]
        prev_low_r = prior_r.loc[prev_low_idx] if prev_low_idx in prior_r.index else np.nan
        if (today_c <= prev_low_c) and (not pd.isna(prev_low_r)) and (today_r > prev_low_r):
            out["bullish"] = True
            out["prior_close"] = float(prev_low_c)
            out["today_close"] = float(today_c)
            out["prior_rsi"] = float(prev_low_r)
            out["today_rsi"] = float(today_r)
    # Bearish: today is highest close, but RSI is lower than RSI at the *previous highest*
    prev_high_idx = prior_c.idxmax()
    if pd.notna(prev_high_idx):
        prev_high_c = prior_c.loc[prev_high_idx]
        prev_high_r = prior_r.loc[prev_high_idx] if prev_high_idx in prior_r.index else np.nan
        if (today_c >= prev_high_c) and (not pd.isna(prev_high_r)) and (today_r < prev_high_r):
            out["bearish"] = True
            out["prior_close"] = float(prev_high_c)
            out["today_close"] = float(today_c)
            out["prior_rsi"] = float(prev_high_r)
            out["today_rsi"] = float(today_r)
    return out


# =============================================================================
# BB-inside-KC counter (rolling)
# =============================================================================
def bb_inside_kc_counter(bb_up: pd.Series, bb_dn: pd.Series,
                         kc_up: pd.Series, kc_dn: pd.Series,
                         window: int = 10) -> pd.Series:
    """Count of bars in last ``window`` where BB band is fully inside KC band.

    Used by setups A3 (VCP) and A4 (Squeeze). When BB ⊂ KC, volatility is
    contracted relative to true range — a "squeeze" condition.

    Returns a rolling count series 0..window.
    """
    if bb_up is None or bb_dn is None or kc_up is None or kc_dn is None:
        return pd.Series(dtype=float)
    inside = ((bb_up <= kc_up) & (bb_dn >= kc_dn)).astype(float)
    return inside.rolling(window, min_periods=max(2, window // 2)).sum()


# =============================================================================
# Rolling percentile rank (used by A3 — 90d window per Yash customization)
# =============================================================================
def rolling_percentile(series: pd.Series, window: int = 90) -> pd.Series:
    """For each bar, the percentile rank (0..100) of its value within the
    prior ``window`` observations (inclusive of self at the rolling tail)."""
    if series is None or series.empty:
        return pd.Series(dtype=float)

    def _pct(arr):
        if len(arr) < 2:
            return np.nan
        v = arr[-1]
        if not np.isfinite(v):
            return np.nan
        return float(((arr[:-1] < v).sum()) / max(1, len(arr) - 1) * 100.0)

    return series.rolling(window, min_periods=max(5, window // 4)).apply(_pct, raw=True)


# =============================================================================
# Linear-regression slope (used in spread/fly trend factors)
# =============================================================================
def linear_slope(series: pd.Series, window: int = 5) -> pd.Series:
    """Rolling OLS slope of ``series`` on a 0..window-1 x-axis. Units = same as
    series per bar. Used by A14 (range expansion timing — though we removed A14,
    keeping for spread/fly trend factor f4)."""
    if series is None or series.empty:
        return pd.Series(dtype=float)
    n = window
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()
    if x_var == 0:
        return pd.Series(np.nan, index=series.index)

    def _slope(arr):
        if len(arr) < n or not np.isfinite(arr).all():
            return np.nan
        y_mean = arr.mean()
        return ((x - x_mean) * (arr - y_mean)).sum() / x_var

    return series.rolling(window, min_periods=window).apply(_slope, raw=True)


# =============================================================================
# Triangle / wedge polyfit on swings (A15)
# =============================================================================
def polyfit_pattern_lines(swings_high_idx: list, swings_high_val: list,
                           swings_low_idx: list, swings_low_val: list,
                           min_points: int = 4) -> dict:
    """Fit linear trendlines to the most recent N swing-highs and swing-lows
    (separately) and characterise the pattern.

    Returns:
      {
        upper_slope, upper_intercept, upper_r2,
        lower_slope, lower_intercept, lower_r2,
        n_upper, n_lower,
        is_converging  (True iff slopes have opposite signs and lines intersect ahead),
        breakout_above_at, breakout_below_at  (values projected at last bar+1),
      }
    Insufficient swings → all None.
    """
    out = {"upper_slope": None, "upper_intercept": None, "upper_r2": None,
           "lower_slope": None, "lower_intercept": None, "lower_r2": None,
           "n_upper": 0, "n_lower": 0,
           "is_converging": False,
           "breakout_above_at": None, "breakout_below_at": None}
    if len(swings_high_idx) < min_points or len(swings_low_idx) < min_points:
        return out
    # Use the most recent min_points to 6 swings of each type
    n_use = min(6, max(min_points, min(len(swings_high_idx), len(swings_low_idx))))
    h_idx = np.array(swings_high_idx[-n_use:], dtype=float)
    h_val = np.array(swings_high_val[-n_use:], dtype=float)
    l_idx = np.array(swings_low_idx[-n_use:], dtype=float)
    l_val = np.array(swings_low_val[-n_use:], dtype=float)
    out["n_upper"] = int(len(h_idx))
    out["n_lower"] = int(len(l_idx))
    try:
        u_slope, u_int = np.polyfit(h_idx, h_val, 1)
        u_pred = u_slope * h_idx + u_int
        u_ss_res = ((h_val - u_pred) ** 2).sum()
        u_ss_tot = ((h_val - h_val.mean()) ** 2).sum()
        u_r2 = 1.0 - (u_ss_res / u_ss_tot) if u_ss_tot > 0 else 0.0
    except Exception:
        return out
    try:
        l_slope, l_int = np.polyfit(l_idx, l_val, 1)
        l_pred = l_slope * l_idx + l_int
        l_ss_res = ((l_val - l_pred) ** 2).sum()
        l_ss_tot = ((l_val - l_val.mean()) ** 2).sum()
        l_r2 = 1.0 - (l_ss_res / l_ss_tot) if l_ss_tot > 0 else 0.0
    except Exception:
        return out
    last_idx = max(h_idx.max(), l_idx.max())
    out["upper_slope"] = float(u_slope)
    out["upper_intercept"] = float(u_int)
    out["upper_r2"] = float(u_r2)
    out["lower_slope"] = float(l_slope)
    out["lower_intercept"] = float(l_int)
    out["lower_r2"] = float(l_r2)
    out["breakout_above_at"] = float(u_slope * (last_idx + 1) + u_int)
    out["breakout_below_at"] = float(l_slope * (last_idx + 1) + l_int)
    # Converging when upper slope is ≤0 AND lower slope is ≥0 (or strictly opposite)
    if (u_slope < 0 and l_slope > 0) or (u_slope <= 0 and l_slope > 0) \
            or (u_slope < 0 and l_slope >= 0):
        out["is_converging"] = True
    return out
