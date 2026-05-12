"""Proximity engine — N-day high/low proximity, ATR-normalised distance, streaks,
velocities, fresh/failed breaks, confluence-pattern catalogue, and cluster
signals — for the SRA → Analysis → Proximity subtab.

Design principles (locked):
  · ALL spread/fly bp scaling goes through ``lib.contract_units.bp_multipliers_for``
    so already-bp contracts are NOT re-multiplied by 100.
  · History windows EXCLUDE the as-of date itself (``< ts``, not ``<=``) — same
    convention as the Z-score block in ``lib.sra_data``.
  · Distances reported in three flavours:
      raw bp           — close differences ×bp_multiplier
      ATR-normalised   — dist / ATR(14), unit-agnostic
      % of N-day range — current's position in [low_n, high_n]

Threshold convention (matches PIA):
  AT          dist_to_extreme ≤ 0.25 · ATR
  NEAR                       ≤ 0.50 · ATR
  APPROACHING                ≤ 1.00 · ATR
  FAR         otherwise
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from lib.contract_units import bp_multipliers_for, load_catalog


# =============================================================================
# Constants
# =============================================================================
ATR_PERIOD = 14

AT_ATR        = 0.25      # dist ≤ 0.25 ATR  → AT
NEAR_ATR      = 0.50      # dist ≤ 0.50 ATR  → NEAR
APPROACH_ATR  = 1.00      # dist ≤ 1.00 ATR  → APPROACHING

FAILED_BREAK_LOOKBACK_BARS = 3   # was AT within last K bars, now ≥0.5 ATR away
VELOCITY_WINDOW_BARS       = 5   # average daily Δdist over last K bars


def get_proximity_interpretation_guide() -> dict:
    """Structured per-metric guide for the Proximity subtab.

    Mirrors the shape used by ``lib.mean_reversion.get_interpretation_guide``
    so ``lib.mean_reversion.metric_tooltip`` can be reused on either subtab.
    """
    return {
        "Distance to extreme (ATR-normalised)": {
            "what":  "Distance from current close to the rolling N-day high/low, "
                     "normalised by ATR(14). Unit-agnostic so spreads/flies compare to outrights.",
            "formula": "dist_atr = max(0, extreme − current) / ATR(14)   (extreme = high_n or low_n)",
            "buckets": [
                (f"≤ {AT_ATR:.2f}",          "AT",                 "at the extreme — touched/tested"),
                (f"≤ {NEAR_ATR:.2f}",        "NEAR",               "within half-ATR of extreme — close call"),
                (f"≤ {APPROACH_ATR:.2f}",    "APPROACHING",        "within one ATR — building pressure"),
                (f"> {APPROACH_ATR:.2f}",    "FAR",                "neutral / out of range — no edge"),
            ],
        },
        "Position-in-range (PIR)": {
            "what":  "Where the current close sits within [low_n, high_n] of the lookback window. "
                     "0 = at low, 1 = at high.",
            "formula": "PIR = (current − low_n) / (high_n − low_n)   (clipped to [0, 1])",
            "buckets": [
                ("≥ 0.90",        "near top of range",   "within 10% of N-day high — extension-leaning"),
                ("0.70–0.90",     "upper third",         "leans rich"),
                ("0.30–0.70",     "middle of range",     "neutral"),
                ("0.10–0.30",     "lower third",         "leans cheap"),
                ("≤ 0.10",        "near bottom of range", "within 10% of N-day low — exhaustion-leaning"),
            ],
        },
        "Streak at extreme": {
            "what":  "Consecutive prior bars within 0.5·ATR of the rolling N-day extreme on the nearest side. "
                     "Persistence indicator — long streaks = the regime has legs.",
            "formula": "walk back through prior bars; count while dist_atr ≤ 0.5; stop on first miss",
            "buckets": [
                ("≥ 5d",          "very persistent",      "regime committed — fading is risky"),
                ("3–4d",          "persistent",           "trend confirmed across multiple sessions"),
                ("1–2d",          "fresh",                "recent or single-bar touch"),
                ("0d",            "no streak",            "today's flag is the first touch"),
            ],
        },
        "Velocity to extreme (ATR/day)": {
            "what":  "Average daily Δ(distance-to-extreme in ATR units) over the last 5 prior bars. "
                     "Negative = closing in on the extreme; positive = backing off.",
            "formula": "vel = mean(diff(dist_atr_t))  for t in last 6 bars",
            "buckets": [
                ("≤ −0.10",       "rapidly closing in",   "tightening fast — momentum building toward extreme"),
                ("−0.10 to −0.02", "drifting toward extreme", "slow approach"),
                ("−0.02 to +0.02", "stable",              "no velocity — coiled at level"),
                ("+0.02 to +0.10", "backing off",         "moving away from extreme — losing momentum"),
                ("> +0.10",       "rapidly moving away",  "extension fading"),
            ],
        },
        "Touch count (re-tests)": {
            "what":  f"Number of prior bars within {AT_ATR:.2f}·ATR of the rolling N-day extreme on the nearest side. "
                     "Multiple touches indicate coiling / re-test pattern.",
            "formula": f"# bars in window with dist_atr ≤ {AT_ATR:.2f}",
            "buckets": [
                ("≥ 5",           "heavy coiling",        "level repeatedly tested — accumulation/distribution"),
                ("3–4",           "coiling",              "level being re-tested; breakout candidate"),
                ("1–2",           "tested",               "level touched recently"),
                ("0",             "untested",             "extreme hasn't been touched within the window"),
            ],
        },
        "Range expansion ratio": {
            "what":  "Current N-day range divided by the prior N-day range (the [-2N : -N] window). "
                     "Tells you whether volatility is expanding or contracting.",
            "formula": "range_n / range_n_prior",
            "buckets": [
                ("≥ 1.50",        "strong expansion",     "volatility regime spike — sustainability check needed"),
                ("1.20–1.50",     "expanding",            "vol regime up — trends extend further"),
                ("0.80–1.20",     "stable",               "consistent volatility regime"),
                ("0.50–0.80",     "contracting",          "vol shrinking — coiling pattern"),
                ("< 0.50",        "strong contraction",   "deep volatility compression — breakout pending"),
            ],
        },
        "Confluence pattern (multi-window)": {
            "what":  "Classifies the cross-lookback (5/15/30/60/90d) proximity structure into one of 9 named regimes.",
            "formula": "see lib.proximity.classify_proximity_pattern",
            "buckets": [
                ("PERSISTENT",    "all elevated, same side",  "trend regime — ride don't fade"),
                ("ACCELERATING",  "tighter as window shortens", "momentum building — trend continuation"),
                ("DECELERATING",  "looser as window shortens", "momentum fading — reversal candidate"),
                ("FRESH",         "only shortest elevated",    "event-driven — needs longer-window confirmation"),
                ("DRIFTED",       "long-window elevated, 5d FAR", "coiling near old extreme"),
                ("REVERTING",     "5d & 90d opposite sides",   "caught in the turn — fade entry"),
                ("DIVERGENT",     "multiple sides flagged",    "timeframe disagreement — range-bound"),
                ("STABLE",        "nothing elevated",          "quiet — no signal"),
                ("MIXED",         "no clean shape",            "neutral catch-all"),
            ],
        },
        "Section regime (proximity)": {
            "what":  "Curve-level regime label derived from average position-in-range per section. "
                     "Confirms / contradicts the regression-based regime engine.",
            "formula": "BULL/BEAR if all sections >0.7 / <0.3 · STEEPENING/FLATTENING if back/front PIR exceeds the other by 0.3 · BELLY-BID/OFFERED if mid-PIR vs avg(wings) ±0.25",
            "buckets": [
                ("BULL",                "all sections at recent highs", "universe-wide rally — ride longs"),
                ("BEAR",                "all sections at recent lows",  "universe-wide sell — ride shorts"),
                ("STEEPENING",          "back high, front low",         "back rich vs front — possible flatten unwind"),
                ("FLATTENING",          "front high, back low",         "front rich vs back — possible steepen unwind"),
                ("BELLY-BID",           "mid bulged up vs wings",       "body squeeze — fade belly long"),
                ("BELLY-OFFERED",       "mid sagged vs wings",          "body sell — fade belly short"),
                ("MIXED",               "no clear divergence",          "no strong cross-section signal"),
            ],
        },
        "Z-score cross-check": {
            "what":  "Pairs proximity flag with Z-score on the same lookback to characterise the extreme.",
            "formula": "AT/NEAR + |z|<1 → UNTESTED · AT/NEAR + |z|≥2 → STRETCHED · FAR + |z|≥1.5 → COILED · FAR + |z|<1 → NORMAL",
            "buckets": [
                ("UNTESTED-EXTREME",  "AT/NEAR but |z|<1",       "uncrowded breakout — move has room to extend"),
                ("STRETCHED-EXTREME", "AT/NEAR and |z|≥2",       "exhausted — fade with confluence"),
                ("COILED",            "FAR and |z|≥1.5",         "rich/cheap on z but not at recent extreme — mean-reversion in process"),
                ("NORMAL",            "FAR and |z|<1",           "quiet — no edge"),
            ],
        },
    }


def get_proximity_thresholds_text() -> str:
    """Plain-text summary of proximity thresholds for HTML title-attribute tooltips."""
    return (
        "PROXIMITY THRESHOLDS (constant across all lookbacks)\n"
        "\n"
        f"  AT          dist ≤ {AT_ATR:.2f} · ATR(14)\n"
        f"  NEAR        dist ≤ {NEAR_ATR:.2f} · ATR(14)\n"
        f"  APPROACHING dist ≤ {APPROACH_ATR:.2f} · ATR(14)\n"
        "  FAR         otherwise\n"
        "\n"
        "Distances reported in:\n"
        "  bp                — close diff × bp_multiplier (catalog-aware)\n"
        "  ATR-normalised    — dist / ATR(14)\n"
        "  % of N-day range  — position_in_range ∈ [0,1]; 0=at low, 1=at high\n"
        "\n"
        "History windows EXCLUDE the as-of date — today vs the prior N days.\n"
        "ATR is computed on stored-unit OHLC and converted to bp via the same\n"
        "per-contract multiplier (1× for stored-bp spreads/flies, 100× for outrights).\n"
        "\n"
        f"FRESH break    today's close > prior N-day high (strict) or < prior N-day low\n"
        f"FAILED break   AT within last {FAILED_BREAK_LOOKBACK_BARS} bars, now ≥0.5 ATR away\n"
        f"VELOCITY       avg daily Δ(dist) in ATR units over last {VELOCITY_WINDOW_BARS} bars (negative = closing in)\n"
    )


# =============================================================================
# ATR computation (per-contract, in stored units; bp via catalog multiplier)
# =============================================================================
def _atr_series_for_contract(high: pd.Series, low: pd.Series, close: pd.Series,
                             period: int = ATR_PERIOD) -> pd.Series:
    """ATR(period) Series in the contract's stored-price units. Robust to NaNs."""
    df = pd.concat({"h": high, "l": low, "c": close}, axis=1).dropna()
    if len(df) < 2:
        return pd.Series(dtype=float)
    prev_c = df["c"].shift(1)
    tr = pd.concat([
        df["h"] - df["l"],
        (df["h"] - prev_c).abs(),
        (df["l"] - prev_c).abs(),
    ], axis=1).max(axis=1)
    min_p = max(2, period // 2)
    return tr.rolling(period, min_periods=min_p).mean()


def compute_atr_panel(wide_high: pd.DataFrame, wide_low: pd.DataFrame,
                      wide_close: pd.DataFrame, asof_date: date,
                      period: int = ATR_PERIOD) -> dict:
    """Latest ATR per contract (strictly before asof_date), in stored-price units.

    Returns ``{symbol: atr_value}``.  Symbols with insufficient history → omitted.
    """
    out: dict[str, float] = {}
    if wide_close is None or wide_close.empty:
        return out
    ts = pd.Timestamp(asof_date)
    for sym in wide_close.columns:
        try:
            high = wide_high[sym] if sym in wide_high.columns else None
            low = wide_low[sym] if sym in wide_low.columns else None
            close = wide_close[sym]
            if high is None or low is None or close is None:
                continue
            atr_series = _atr_series_for_contract(high, low, close, period=period)
            if atr_series.empty:
                continue
            history = atr_series.loc[atr_series.index < ts].dropna()
            if not history.empty:
                v = float(history.iloc[-1])
                if np.isfinite(v) and v > 0:
                    out[sym] = v
        except Exception:
            continue
    return out


# =============================================================================
# Per-contract proximity over a single lookback
# =============================================================================
def _flag_for_atr(dist_atr: Optional[float]) -> str:
    """Map a normalised distance to AT/NEAR/APPROACHING/FAR."""
    if dist_atr is None or pd.isna(dist_atr):
        return "—"
    if dist_atr <= AT_ATR:
        return "AT"
    if dist_atr <= NEAR_ATR:
        return "NEAR"
    if dist_atr <= APPROACH_ATR:
        return "APPROACHING"
    return "FAR"


def _streak_at_extreme(close_series: pd.Series, side: str, asof_date: date,
                       lookback: int, atr_stored: float) -> int:
    """Consecutive prior bars within 0.5·ATR of the rolling N-day extreme on ``side``.

    side: 'high' or 'low'. We walk backward from the most recent prior bar; for each
    bar we recompute the N-day high/low EXCLUDING that bar and check if the bar
    sits within 0.5·ATR of that extreme. Stop at the first bar that breaks.
    """
    if atr_stored is None or atr_stored <= 0 or close_series.empty:
        return 0
    ts = pd.Timestamp(asof_date)
    history = close_series.loc[close_series.index < ts].dropna()
    if history.empty:
        return 0
    streak = 0
    for i in range(len(history) - 1, -1, -1):
        # Window of N bars BEFORE bar i (so bar i is not in the window)
        start = max(0, i - lookback)
        window = history.iloc[start:i]
        if len(window) < 2:
            break
        bar_val = float(history.iloc[i])
        if side == "high":
            ext = float(window.max())
            dist = max(0.0, ext - bar_val)
        else:
            ext = float(window.min())
            dist = max(0.0, bar_val - ext)
        if dist / atr_stored <= NEAR_ATR:
            streak += 1
        else:
            break
    return streak


def _velocity_atr_per_day(close_series: pd.Series, side: str, asof_date: date,
                          lookback: int, atr_stored: float,
                          window_bars: int = VELOCITY_WINDOW_BARS) -> Optional[float]:
    """Average daily Δ(dist-to-extreme in ATR units) over the last ``window_bars``.

    Negative = approaching the extreme (good for momentum). Positive = backing off.
    """
    if atr_stored is None or atr_stored <= 0:
        return None
    ts = pd.Timestamp(asof_date)
    history = close_series.loc[close_series.index < ts].dropna()
    if len(history) < window_bars + lookback:
        # Not enough history; relax slightly
        if len(history) < window_bars + 2:
            return None
    dists = []
    for i in range(len(history) - window_bars - 1, len(history)):
        if i < 0:
            continue
        start = max(0, i - lookback + 1)
        window = history.iloc[start:i + 1]
        if len(window) < 2:
            continue
        bar_val = float(history.iloc[i])
        if side == "high":
            ext = float(window.max())
            d = max(0.0, ext - bar_val) / atr_stored
        else:
            ext = float(window.min())
            d = max(0.0, bar_val - ext) / atr_stored
        dists.append(d)
    if len(dists) < 2:
        return None
    deltas = np.diff(dists)
    return float(np.mean(deltas))


def _touch_count_in_window(close_series: pd.Series, asof_date: date,
                           lookback: int, atr_stored: float, side: str) -> int:
    """Count of prior bars in the lookback window that came within ``AT_ATR`` of the
    rolling N-day extreme (computed bar-by-bar, exclusive). Indicates re-tests."""
    if atr_stored is None or atr_stored <= 0:
        return 0
    ts = pd.Timestamp(asof_date)
    history = close_series.loc[close_series.index < ts].dropna()
    if len(history) < 3:
        return 0
    # The lookback window relative to today: most-recent N bars
    window = history.tail(lookback)
    if window.empty or atr_stored <= 0:
        return 0
    if side == "high":
        ext = float(window.max())
        diffs = (ext - window).abs()
    else:
        ext = float(window.min())
        diffs = (window - ext).abs()
    return int((diffs / atr_stored <= AT_ATR).sum())


def _failed_break_check(close_series: pd.Series, asof_date: date, lookback: int,
                        atr_stored: float, side: str,
                        recent_k: int = FAILED_BREAK_LOOKBACK_BARS) -> bool:
    """True iff within the last ``recent_k`` prior bars the close came within
    ``AT_ATR`` of the rolling N-day extreme on ``side``, AND today is ≥``NEAR_ATR``
    away from that extreme — i.e. touch-and-reverse setup.
    """
    if atr_stored is None or atr_stored <= 0:
        return False
    ts = pd.Timestamp(asof_date)
    prior = close_series.loc[close_series.index < ts].dropna()
    if len(prior) < recent_k + 2:
        return False
    # Today's close = the as-of bar (could be in series or not)
    try:
        today_close = float(close_series.loc[close_series.index.date == asof_date].iloc[0])
    except Exception:
        return False
    # Today's distance to the N-day extreme (excluding today)
    window_today = prior.tail(lookback)
    if len(window_today) < 2:
        return False
    if side == "high":
        ext_today = float(window_today.max())
        today_dist = max(0.0, ext_today - today_close) / atr_stored
    else:
        ext_today = float(window_today.min())
        today_dist = max(0.0, today_close - ext_today) / atr_stored
    if today_dist < NEAR_ATR:
        return False     # not actually away → not a failed break
    # Did the recent K bars touch?
    for i in range(max(0, len(prior) - recent_k), len(prior)):
        bar_val = float(prior.iloc[i])
        start = max(0, i - lookback + 1)
        window = prior.iloc[start:i + 1]
        if len(window) < 2:
            continue
        if side == "high":
            ext = float(window.max())
            d = max(0.0, ext - bar_val) / atr_stored
        else:
            ext = float(window.min())
            d = max(0.0, bar_val - ext) / atr_stored
        if d <= AT_ATR:
            return True
    return False


def _range_expansion_ratio(close_series: pd.Series, asof_date: date,
                           lookback: int) -> Optional[float]:
    """Ratio of current N-day range to the prior N-day range (i.e. range_n / range_2n−n).

    >1 → range expanding (volatility increasing); <1 → range contracting.
    """
    ts = pd.Timestamp(asof_date)
    prior = close_series.loc[close_series.index < ts].dropna()
    if len(prior) < lookback * 2:
        return None
    recent = prior.tail(lookback)
    earlier = prior.iloc[-(2 * lookback):-lookback]
    rng_recent = float(recent.max() - recent.min())
    rng_earlier = float(earlier.max() - earlier.min())
    if rng_earlier <= 1e-12:
        return None
    return rng_recent / rng_earlier


def compute_contract_proximity(close_series: pd.Series, asof_date: date,
                                lookback: int, atr_stored: float,
                                bp_mult: float) -> dict:
    """Per-contract proximity metrics over a single lookback.

    All BP-flavoured outputs are catalog-aware via ``bp_mult`` (1× for stored-bp
    spreads/flies, 100× for outrights).
    """
    out = {
        "high_n": None, "low_n": None,
        "high_n_bp": None, "low_n_bp": None,
        "current": None, "current_bp": None,
        "dist_high_bp": None, "dist_low_bp": None,
        "dist_high_atr": None, "dist_low_atr": None,
        "position_in_range": None,
        "flag_high": "—", "flag_low": "—",
        "nearest_extreme": "—", "nearest_dist_atr": None,
        "streak_at_extreme": 0,
        "velocity_atr_per_day": None,
        "fresh_break_high": False, "fresh_break_low": False,
        "failed_break_high": False, "failed_break_low": False,
        "touch_count_high": 0, "touch_count_low": 0,
        "range_n_bp": None, "range_expansion_ratio": None,
        "n_observations": 0,
    }

    if close_series is None or close_series.empty:
        return out

    ts = pd.Timestamp(asof_date)
    history = close_series.loc[close_series.index < ts].dropna()
    if history.empty:
        return out

    window = history.tail(lookback)
    out["n_observations"] = int(len(window))
    if len(window) < 2:
        return out

    try:
        current = float(close_series.loc[close_series.index.date == asof_date].iloc[0])
    except Exception:
        return out
    if pd.isna(current):
        return out

    high_n = float(window.max())
    low_n = float(window.min())
    rng = high_n - low_n

    out["high_n"] = high_n
    out["low_n"] = low_n
    out["high_n_bp"] = high_n * bp_mult
    out["low_n_bp"] = low_n * bp_mult
    out["current"] = current
    out["current_bp"] = current * bp_mult
    out["range_n_bp"] = rng * bp_mult
    out["dist_high_bp"] = max(0.0, high_n - current) * bp_mult
    out["dist_low_bp"] = max(0.0, current - low_n) * bp_mult
    if rng > 1e-12:
        out["position_in_range"] = max(0.0, min(1.0, (current - low_n) / rng))
    else:
        out["position_in_range"] = 0.5

    if atr_stored is not None and atr_stored > 0:
        out["dist_high_atr"] = max(0.0, high_n - current) / atr_stored
        out["dist_low_atr"] = max(0.0, current - low_n) / atr_stored
        out["flag_high"] = _flag_for_atr(out["dist_high_atr"])
        out["flag_low"] = _flag_for_atr(out["dist_low_atr"])
        if out["dist_high_atr"] <= out["dist_low_atr"]:
            out["nearest_extreme"] = "HIGH"
            out["nearest_dist_atr"] = out["dist_high_atr"]
            side = "high"
        else:
            out["nearest_extreme"] = "LOW"
            out["nearest_dist_atr"] = out["dist_low_atr"]
            side = "low"
        out["streak_at_extreme"] = _streak_at_extreme(
            close_series, side, asof_date, lookback, atr_stored)
        out["velocity_atr_per_day"] = _velocity_atr_per_day(
            close_series, side, asof_date, lookback, atr_stored)
        out["touch_count_high"] = _touch_count_in_window(
            close_series, asof_date, lookback, atr_stored, "high")
        out["touch_count_low"] = _touch_count_in_window(
            close_series, asof_date, lookback, atr_stored, "low")
        out["failed_break_high"] = _failed_break_check(
            close_series, asof_date, lookback, atr_stored, "high")
        out["failed_break_low"] = _failed_break_check(
            close_series, asof_date, lookback, atr_stored, "low")

    # Fresh breaks — strict comparisons against prior N-day extreme
    out["fresh_break_high"] = bool(current > high_n + 1e-12)
    out["fresh_break_low"] = bool(current < low_n - 1e-12)
    # When fresh break occurs, the corresponding distance is 0 (touched/exceeded)
    if out["fresh_break_high"]:
        out["dist_high_bp"] = 0.0
        out["dist_high_atr"] = 0.0
        out["flag_high"] = "AT"
    if out["fresh_break_low"]:
        out["dist_low_bp"] = 0.0
        out["dist_low_atr"] = 0.0
        out["flag_low"] = "AT"

    out["range_expansion_ratio"] = _range_expansion_ratio(close_series, asof_date, lookback)

    return out


# =============================================================================
# Confluence-pattern classifier across multiple lookbacks
# =============================================================================
def _is_elevated(metrics: dict) -> bool:
    """A contract is 'elevated' on a given lookback when nearest-dist ≤ NEAR_ATR (AT or NEAR)."""
    d = metrics.get("nearest_dist_atr")
    return d is not None and d <= NEAR_ATR


def _side(metrics: dict) -> Optional[str]:
    """'HIGH' / 'LOW' / None for an elevated reading."""
    if not _is_elevated(metrics):
        return None
    return metrics.get("nearest_extreme")


def classify_proximity_pattern(by_lookback: dict, lookbacks: list) -> str:
    """8-pattern classifier across an ordered list of lookbacks (5/15/30/60/90).

    PERSISTENT  · all elevated, same side
    FRESH       · only shortest elevated; longer windows FAR
    DRIFTED     · longer-windows elevated, shortest FAR  (coiling near old extreme)
    ACCELERATING· monotonically tighter (smaller dist) as window shrinks, all same side, all elevated
    DECELERATING· monotonically looser, all same side, all elevated
    REVERTING   · shortest HIGH and longest LOW (or vice versa), both elevated
    DIVERGENT   · mixed sides without a clear shape
    STABLE      · nothing elevated in any window
    MIXED       · catch-all
    """
    if not lookbacks:
        return "MIXED"
    ordered = sorted(lookbacks)
    lows = [by_lookback.get(k, {}) for k in ordered]
    if not lows or all(not lo for lo in lows):
        return "STABLE"
    elev = [_is_elevated(m) for m in lows]
    sides = [_side(m) for m in lows]
    dists = [m.get("nearest_dist_atr") for m in lows]

    if not any(elev):
        return "STABLE"

    # PERSISTENT
    if all(elev) and len(set(s for s in sides if s)) == 1:
        # Need similar magnitudes for "persistent" (no monotonic ordering)
        if all(d is not None for d in dists):
            mono_decreasing = all(dists[i] <= dists[i - 1] + 1e-9 for i in range(1, len(dists)))
            mono_increasing = all(dists[i] >= dists[i - 1] - 1e-9 for i in range(1, len(dists)))
            if not mono_decreasing and not mono_increasing:
                return "PERSISTENT"
            # Strictly monotonic with same side AND elevated → ACCELERATING/DECELERATING
            if mono_decreasing and not mono_increasing:
                return "ACCELERATING"
            if mono_increasing and not mono_decreasing:
                return "DECELERATING"
            return "PERSISTENT"

    # FRESH — only shortest is elevated
    if elev[0] and not any(elev[1:]):
        return "FRESH"

    # DRIFTED — longer-windows elevated, shortest not
    if not elev[0] and any(elev[1:]):
        # If all longer windows elevated and same side → drifted
        longer_sides = [s for e, s in zip(elev[1:], sides[1:]) if e and s]
        if longer_sides and len(set(longer_sides)) == 1:
            return "DRIFTED"

    # REVERTING — shortest and longest opposite sides, both elevated
    if elev[0] and elev[-1] and sides[0] and sides[-1] and sides[0] != sides[-1]:
        return "REVERTING"

    # DIVERGENT — multiple sides flagged
    elevated_sides = set(s for e, s in zip(elev, sides) if e and s)
    if len(elevated_sides) > 1:
        return "DIVERGENT"

    return "MIXED"


def get_pattern_descriptions() -> dict:
    """Tooltip text per pattern — used in the legend expander."""
    return {
        "PERSISTENT":   "All selected windows AT/NEAR the same extreme — strong directional regime; trend is your friend, fade only with confluence of weakness.",
        "FRESH":        "Only the shortest window is at extreme; longer windows still FAR — event-driven move that has yet to be confirmed by longer horizons.",
        "DRIFTED":      "Long-horizon windows AT/NEAR but the shortest is FAR — coiling pattern after a prior thrust; watch for breakout direction.",
        "ACCELERATING": "Same-side proximity tightens monotonically as the window shrinks — momentum building into the move; trend continuation likely.",
        "DECELERATING": "Same-side proximity loosens monotonically as the window shrinks — momentum fading; reversal/exhaustion candidate.",
        "REVERTING":    "Shortest window at one extreme, longest at the opposite — caught in the turn; classic mean-reversion entry zone.",
        "DIVERGENT":    "Multiple windows AT/NEAR but on different sides — timeframe disagreement; range-bound or no-clear-trend regime.",
        "STABLE":       "No window is AT/NEAR an extreme — quiet contract; no proximity-based signal here.",
        "MIXED":        "Pattern doesn't fit any of the above clean shapes — neutral catch-all.",
    }


# =============================================================================
# Engine — compute everything for the full universe in one go
# =============================================================================
def compute_proximity_panel(wide_close: pd.DataFrame, wide_high: pd.DataFrame,
                             wide_low: pd.DataFrame, contracts: list[str],
                             asof_date: date, lookbacks: list[int],
                             base_product: str = "SRA",
                             atr_period: int = ATR_PERIOD) -> dict:
    """Build the full proximity result for the entire universe.

    Output structure::

        {
          "asof": date,
          "lookbacks": [5, 15, 30, 60, 90],
          "atr_period": 14,
          "n_contracts": int,
          "per_contract": {
            symbol: {
              "convention": "bp"|"price",
              "bp_multiplier": 1.0|100.0,
              "atr_stored": float, "atr_bp": float,
              "current": float, "current_bp": float,
              "by_lookback": {n: {...metrics...}},
              "pattern": str,
            },
            ...
          },
          "fresh_breaks_today": [{symbol, lookback, side, bp_through}, ...],
          "failed_breaks": [{symbol, lookback, side, dist_atr_now}, ...],
        }
    """
    out = {
        "asof": asof_date,
        "lookbacks": list(lookbacks),
        "atr_period": atr_period,
        "n_contracts": len(contracts),
        "per_contract": {},
        "fresh_breaks_today": [],
        "failed_breaks": [],
    }

    if wide_close is None or wide_close.empty or not contracts:
        return out

    cat = load_catalog()
    mults = bp_multipliers_for(contracts, base_product, cat) if not cat.empty \
            else [100.0] * len(contracts)
    sym_mult = dict(zip(contracts, mults))
    sym_conv = {c: ("bp" if abs(sym_mult[c] - 1.0) < 1e-6 else "price")
                for c in contracts}

    atr_panel = compute_atr_panel(wide_high, wide_low, wide_close, asof_date,
                                  period=atr_period)

    for sym in contracts:
        atr_stored = atr_panel.get(sym)
        bp_mult = sym_mult.get(sym, 100.0)
        try:
            close_series = wide_close[sym] if sym in wide_close.columns else pd.Series(dtype=float)
        except Exception:
            close_series = pd.Series(dtype=float)

        record = {
            "convention": sym_conv.get(sym, "unknown"),
            "bp_multiplier": float(bp_mult),
            "atr_stored": atr_stored,
            "atr_bp": (atr_stored * bp_mult) if atr_stored is not None else None,
            "current": None,
            "current_bp": None,
            "by_lookback": {},
            "pattern": "STABLE",
        }
        for n in lookbacks:
            m = compute_contract_proximity(close_series, asof_date, n, atr_stored, bp_mult)
            record["by_lookback"][n] = m
            # populate scalar current values from any non-empty lookback
            if record["current"] is None and m.get("current") is not None:
                record["current"] = m["current"]
                record["current_bp"] = m["current_bp"]
            # collect global lists
            if m.get("fresh_break_high"):
                out["fresh_breaks_today"].append({
                    "symbol": sym, "lookback": n, "side": "HIGH",
                    "bp_through": (m["current"] - m["high_n"]) * bp_mult
                                   if (m.get("current") is not None and m.get("high_n") is not None)
                                   else None,
                })
            if m.get("fresh_break_low"):
                out["fresh_breaks_today"].append({
                    "symbol": sym, "lookback": n, "side": "LOW",
                    "bp_through": (m["low_n"] - m["current"]) * bp_mult
                                   if (m.get("current") is not None and m.get("low_n") is not None)
                                   else None,
                })
            if m.get("failed_break_high"):
                out["failed_breaks"].append({
                    "symbol": sym, "lookback": n, "side": "HIGH",
                    "dist_atr_now": m.get("dist_high_atr"),
                })
            if m.get("failed_break_low"):
                out["failed_breaks"].append({
                    "symbol": sym, "lookback": n, "side": "LOW",
                    "dist_atr_now": m.get("dist_low_atr"),
                })
        record["pattern"] = classify_proximity_pattern(record["by_lookback"], lookbacks)
        out["per_contract"][sym] = record

    return out


# =============================================================================
# Ranking helpers — feeds the section ribbons
# =============================================================================
def rank_closest_to_extreme(panel: dict, contracts: list[str], lookback: int,
                            side: str, top_k: int = 5) -> list[dict]:
    """Return top-K contracts closest to N-day extreme on ``side`` ('high'|'low'),
    ranked by ATR-normalised distance ascending. Each row has the full metric dict
    plus ``symbol`` for convenience. Ties broken by raw bp distance.
    """
    rows = []
    side = side.lower()
    dist_key_atr = "dist_high_atr" if side == "high" else "dist_low_atr"
    dist_key_bp = "dist_high_bp" if side == "high" else "dist_low_bp"
    for sym in contracts:
        rec = panel.get("per_contract", {}).get(sym)
        if not rec:
            continue
        m = rec.get("by_lookback", {}).get(lookback) or {}
        d_atr = m.get(dist_key_atr)
        if d_atr is None or pd.isna(d_atr):
            continue
        rows.append({
            "symbol": sym,
            "convention": rec.get("convention"),
            "bp_multiplier": rec.get("bp_multiplier"),
            "atr_stored": rec.get("atr_stored"),
            "atr_bp": rec.get("atr_bp"),
            "current_bp": m.get("current_bp"),
            "extreme_bp": m.get("high_n_bp" if side == "high" else "low_n_bp"),
            "dist_bp": m.get(dist_key_bp),
            "dist_atr": d_atr,
            "flag": m.get("flag_high" if side == "high" else "flag_low"),
            "position_in_range": m.get("position_in_range"),
            "streak": m.get("streak_at_extreme"),
            "velocity_atr_per_day": m.get("velocity_atr_per_day"),
            "touch_count": m.get("touch_count_high" if side == "high" else "touch_count_low"),
            "fresh_break": m.get("fresh_break_high" if side == "high" else "fresh_break_low"),
            "failed_break": m.get("failed_break_high" if side == "high" else "failed_break_low"),
            "range_n_bp": m.get("range_n_bp"),
            "range_expansion_ratio": m.get("range_expansion_ratio"),
            "pattern": rec.get("pattern"),
        })
    rows.sort(key=lambda r: (r["dist_atr"] if r["dist_atr"] is not None else 1e9,
                             r["dist_bp"] if r["dist_bp"] is not None else 1e12))
    return rows[:max(0, int(top_k))]


# =============================================================================
# Cluster signal — % AT/NEAR per (group_by) bucket
# =============================================================================
def compute_cluster_signal(panel: dict, contracts: list[str],
                           group_for: dict[str, str], lookback: int) -> dict:
    """For each group label in ``group_for`` (sym → label), compute:

        n_total       contracts in the group
        n_at_high     # AT/NEAR HIGH side
        n_at_low      # AT/NEAR LOW side
        pct_at_high   share at HIGH (0..1)
        pct_at_low    share at LOW
        avg_pir       average position_in_range (None if insufficient)
        median_pir    median position_in_range
    Returns ``{label: stats}``.
    """
    buckets: dict[str, dict] = {}
    for sym in contracts:
        label = group_for.get(sym)
        if label is None:
            continue
        rec = panel.get("per_contract", {}).get(sym) or {}
        m = rec.get("by_lookback", {}).get(lookback) or {}
        b = buckets.setdefault(label, {
            "n_total": 0, "n_at_high": 0, "n_at_low": 0,
            "pirs": [], "members": [],
        })
        b["n_total"] += 1
        b["members"].append(sym)
        flag_h = m.get("flag_high")
        flag_l = m.get("flag_low")
        if flag_h in ("AT", "NEAR"):
            b["n_at_high"] += 1
        if flag_l in ("AT", "NEAR"):
            b["n_at_low"] += 1
        pir = m.get("position_in_range")
        if pir is not None and not pd.isna(pir):
            b["pirs"].append(pir)
    out: dict = {}
    for label, b in buckets.items():
        n = b["n_total"]
        pirs = b["pirs"]
        out[label] = {
            "n_total": n,
            "n_at_high": b["n_at_high"],
            "n_at_low": b["n_at_low"],
            "pct_at_high": (b["n_at_high"] / n) if n else 0.0,
            "pct_at_low": (b["n_at_low"] / n) if n else 0.0,
            "avg_pir": float(np.mean(pirs)) if pirs else None,
            "median_pir": float(np.median(pirs)) if pirs else None,
            "members": b["members"],
        }
    return out


# =============================================================================
# Section regime classification from proximity (non-parametric)
# =============================================================================
def proximity_section_regime(panel: dict, contracts: list[str],
                              front_end: int, mid_end: int,
                              lookback: int) -> dict:
    """Build a non-parametric section-level regime from proximity / position-in-range.

    For each section we compute:
        avg_pir            (0..1; 0=at low, 1=at high)
        n_at_high / low    counts of AT/NEAR flags
        bias               "HIGH" / "LOW" / "MIXED" / "QUIET"
    Then a curve-level summary label:
        BULL                 — front+mid+back avg_pir all > 0.7
        BEAR                 — all < 0.3
        STEEPENING           — back high, front low
        FLATTENING           — front high, back low
        BELLY-BID            — mid PIR > avg(front,back) by ≥ 0.25
        BELLY-OFFERED        — mid PIR < avg(front,back) by ≥ 0.25
        MIXED                — none of above
    """
    n = len(contracts)
    if n == 0:
        return {"sections": {}, "label": "—"}
    fe = min(front_end, n)
    me = min(mid_end, n)
    if me <= fe:
        me = min(fe + 1, n)
    sec_ranges = {"front": range(0, fe), "mid": range(fe, me), "back": range(me, n)}
    sections: dict[str, dict] = {}
    for name, rng in sec_ranges.items():
        pirs, n_high, n_low, n_total = [], 0, 0, 0
        for i in rng:
            sym = contracts[i]
            rec = panel.get("per_contract", {}).get(sym) or {}
            m = rec.get("by_lookback", {}).get(lookback) or {}
            n_total += 1
            pir = m.get("position_in_range")
            if pir is not None and not pd.isna(pir):
                pirs.append(pir)
            if m.get("flag_high") in ("AT", "NEAR"):
                n_high += 1
            if m.get("flag_low") in ("AT", "NEAR"):
                n_low += 1
        avg_pir = float(np.mean(pirs)) if pirs else None
        bias = "QUIET"
        if n_total > 0:
            high_pct = n_high / n_total
            low_pct = n_low / n_total
            if high_pct >= 0.4 and low_pct < 0.2:
                bias = "HIGH"
            elif low_pct >= 0.4 and high_pct < 0.2:
                bias = "LOW"
            elif (high_pct + low_pct) >= 0.4:
                bias = "MIXED"
        sections[name] = {
            "n_total": n_total, "n_at_high": n_high, "n_at_low": n_low,
            "avg_pir": avg_pir, "bias": bias,
        }

    # Curve-level label
    def _pir(name): return sections.get(name, {}).get("avg_pir")
    fr_p, md_p, bk_p = _pir("front"), _pir("mid"), _pir("back")
    label = "MIXED"
    if fr_p is not None and md_p is not None and bk_p is not None:
        all_high = all(p >= 0.70 for p in (fr_p, md_p, bk_p))
        all_low = all(p <= 0.30 for p in (fr_p, md_p, bk_p))
        steepening = bk_p - fr_p >= 0.30
        flattening = fr_p - bk_p >= 0.30
        belly_bid = md_p - 0.5 * (fr_p + bk_p) >= 0.25
        belly_offered = 0.5 * (fr_p + bk_p) - md_p >= 0.25
        if all_high:
            label = "BULL (curve at recent highs)"
        elif all_low:
            label = "BEAR (curve at recent lows)"
        elif steepening:
            label = "STEEPENING (back high, front low)"
        elif flattening:
            label = "FLATTENING (front high, back low)"
        elif belly_bid:
            label = "BELLY-BID (mid bulged up)"
        elif belly_offered:
            label = "BELLY-OFFERED (mid sagged)"
        else:
            label = "MIXED"

    return {"sections": sections, "label": label}
