"""Catalog of every setup in the Technicals subtab — id → metadata.

Used by the UI to render setup cards (name, formula, applies-to, threshold info)
without each detector having to repeat its own description text.

Customizations applied per Yash 2026-05-07:
  · A2 — uses EMA_100 (not EMA_200)
  · A3 — percentile_90d (not 252d)
  · A5 — also applies to flies (skip-with-N/A when bars<55)
  · A6 — keeps EMA_50 × EMA_200 (EMA_200 computed at runtime)
  · A12 — split into A12a (EMA_20 variant) and A12b (EMA_50 variant)
  · B10 — vol_z(90d) and TR_z(90d)  (not 252d / 60d)
  · C8 — three variants emit separately: C8_12M, C8_24M (PRIMARY), C8_36M
  · C9 — split into C9a (slope crossing) and C9b (5d trend in slope)
"""
from __future__ import annotations

from typing import Optional


# =============================================================================
# Family + scope tags
# =============================================================================
FAMILY_TREND = "trend"
FAMILY_MR = "mean_reversion"
FAMILY_STIR = "stir"
FAMILY_COMPOSITE = "composite"

SCOPE_OUTRIGHT = "outright"
SCOPE_SPREAD = "spread"
SCOPE_FLY = "fly"
SCOPE_ALL = (SCOPE_OUTRIGHT, SCOPE_SPREAD, SCOPE_FLY)


# =============================================================================
# Registry — one dict per setup id
# =============================================================================
SETUP_REGISTRY: dict = {

    # =====================================================================
    # TREND setups (12 — A12 split into a/b)
    # =====================================================================
    "A1": {
        "name": "TREND_CONTINUATION_BREAKOUT",
        "family": FAMILY_TREND,
        "scopes": SCOPE_ALL,
        "logic_long": "close > High_20.shift(1)  AND  ADX_14 > 20  AND  close > EMA_50  AND  vol > 1.5 × SMA(vol,20)",
        "logic_short": "close < Low_20.shift(1)   AND  ADX_14 > 20  AND  close < EMA_50  AND  vol > 1.5 × SMA(vol,20)",
        "stop_rule": "entry − 2 × ATR_14  (LONG; SHORT mirrors)",
        "t1": "+1R",  "t2": "+2R",
        "trail": "max(EMA_10, prior_close − 3×ATR_14)",
        "time_stop_bars": 30,
        "partial": "50% at T1, stop → BE",
        "buckets": [
            ("4 of 4 conditions met",  "FIRED",       "all conditions met — high-quality breakout"),
            ("3 of 4 met",             "NEAR",        "1 condition missing — about to fire"),
            ("2 of 4 met",             "APPROACHING", "halfway"),
            ("≤1 of 4 met",            "FAR",         "no setup"),
        ],
    },

    "A2": {
        "name": "PULLBACK_IN_TREND",
        "family": FAMILY_TREND,
        "scopes": SCOPE_ALL,
        "logic_long": "close > EMA_100  AND  close < EMA_20  AND  RSI_14 < 45",
        "logic_short": "close < EMA_100  AND  close > EMA_20  AND  RSI_14 > 55",
        "stop_rule": "1.5 × ATR_14",
        "t1": "EMA_20 touch",  "t2": "—",
        "trail": "none", "time_stop_bars": 15, "partial": "none",
        "buckets": [
            ("3 of 3 met", "FIRED",       "trend-pullback in motion"),
            ("2 of 3 met", "NEAR",        "1 condition missing"),
            ("1 of 3 met", "APPROACHING", ""),
            ("0 met",      "FAR",         ""),
        ],
        "note": "EMA_100 (not 200) per Yash customization 2026-05-07",
    },

    "A3": {
        "name": "VCP_BREAKOUT",
        "family": FAMILY_TREND,
        "scopes": SCOPE_ALL,
        "logic_long": ("BB_BW_20_2σ percentile_90d < 25  AND  BB-inside-KC count(10) ≥ 5  "
                       "AND  close > 5d-consol-high  AND  vol ≥ 2 × SMA(vol,20)"),
        "logic_short": "mirror with 5d-consol-low",
        "stop_rule": "consolidation low/high",
        "t1": "+2R",  "t2": "+3R",
        "trail": "EMA_20", "time_stop_bars": 20, "partial": "50% at T1",
        "buckets": [
            ("4 of 4 met", "FIRED",       "Minervini-style VCP breakout"),
            ("3 of 4 met", "NEAR",        "tight base, awaiting volume / break"),
            ("2 of 4 met", "APPROACHING", ""),
            ("≤1 met",     "FAR",         ""),
        ],
        "note": "percentile window = 90d (not 252d) per Yash customization 2026-05-07",
    },

    "A4": {
        "name": "SQUEEZE_FIRE_DIRECTIONAL",
        "family": FAMILY_TREND,
        "scopes": SCOPE_ALL,
        "logic_long": "prior squeeze (BB ⊂ KC ≥3 bars)  AND  BB exits KC today  AND  MACD_hist > 0 rising",
        "logic_short": "prior squeeze  AND  BB exits KC today  AND  MACD_hist < 0 falling",
        "stop_rule": "2 × ATR_14 capped @ opposite KC band",
        "t1": "+1×ATR",  "t2": "+2×ATR",
        "trail": "none", "time_stop_bars": 10, "partial": "50% at T1",
        "buckets": [
            ("3 of 3 met", "FIRED",       "TTM-style squeeze fire — directional thrust"),
            ("2 of 3 met", "NEAR",        ""),
            ("1 of 3 met", "APPROACHING", "squeeze present but no thrust yet"),
            ("0 met",      "FAR",         ""),
        ],
    },

    "A5": {
        "name": "DONCHIAN_55D_BREAKOUT",
        "family": FAMILY_TREND,
        "scopes": SCOPE_ALL,
        "logic_long": "close > High_55.shift(1)  (turtle System 2)",
        "logic_short": "close < Low_55.shift(1)",
        "stop_rule": "2 × ATR_14",
        "t1": "opposite 20d extreme",  "t2": "—",
        "trail": "none", "time_stop_bars": None, "partial": "none",
        "buckets": [
            ("close > High_55", "FIRED-LONG",  "turtle long"),
            ("close < Low_55",  "FIRED-SHORT", "turtle short"),
            ("within 0.5 ATR",  "NEAR",        "approaching breakout"),
            ("else",            "FAR",         "in range"),
        ],
        "note": "extended to flies per Yash customization 2026-05-07; "
                "skipped with state=N/A when fewer than 55 daily bars available",
    },

    "A6": {
        "name": "GOLDEN_DEATH_CROSS",
        "family": FAMILY_TREND,
        "scopes": (SCOPE_OUTRIGHT,),
        "logic_long": "EMA_50 crosses above EMA_200 today  AND  ADX_14 > 18",
        "logic_short": "EMA_50 crosses below EMA_200 today  AND  ADX_14 > 18",
        "stop_rule": "3 × ATR_14",
        "t1": "+1.5R",  "t2": "—",
        "trail": "EMA_50 ± 0.5 × ATR_14", "time_stop_bars": 252,
        "partial": "50% at T1",
        "buckets": [
            ("cross today + ADX>18", "FIRED",       "regime change"),
            ("cross today, ADX≤18",  "APPROACHING", "weak ADX — wait"),
            ("EMAs converging",      "NEAR",        "cross imminent (within 0.5×ATR)"),
            ("else",                 "FAR",         ""),
        ],
        "note": "EMA_200 computed at runtime (not in mde2_indicators)",
    },

    "A8": {
        "name": "ICHIMOKU_CLOUD_BREAKOUT",
        "family": FAMILY_TREND,
        "scopes": SCOPE_ALL,
        "logic_long": "close > Senkou_max  AND  Tenkan > Kijun  AND  Chikou > price 26 bars ago",
        "logic_short": "close < Senkou_min  AND  Tenkan < Kijun  AND  Chikou < price 26 bars ago",
        "stop_rule": "Kijun_26 line",
        "t1": "half-projected cloud",  "t2": "full cloud",
        "trail": "none", "time_stop_bars": 20, "partial": "50% at T1",
        "buckets": [
            ("3 of 3 met", "FIRED",       "Ichimoku cloud breakout"),
            ("2 of 3 met", "NEAR",        ""),
            ("1 of 3 met", "APPROACHING", ""),
            ("0 met",      "FAR",         ""),
        ],
    },

    "A10": {
        "name": "MA_RIBBON_ALIGNMENT",
        "family": FAMILY_TREND,
        "scopes": SCOPE_ALL,
        "logic_long": "first bar of: EMA_5 > EMA_10 > EMA_20 > EMA_30 > EMA_50 > EMA_100",
        "logic_short": "first bar of: EMA_5 < EMA_10 < EMA_20 < EMA_30 < EMA_50 < EMA_100",
        "stop_rule": "EMA_50",
        "t1": "+1.5R",  "t2": "—",
        "trail": "EMA_20", "time_stop_bars": 30, "partial": "50% at T1",
        "buckets": [
            ("5 of 5 stack steps met", "FIRED",       "fresh full alignment"),
            ("4 of 5 met",             "NEAR",        ""),
            ("3 of 5 met",             "APPROACHING", ""),
            ("≤2 met",                 "FAR",         ""),
        ],
    },

    "A11": {
        "name": "SUPERTREND_DIRECTIONAL",
        "family": FAMILY_TREND,
        "scopes": SCOPE_ALL,
        "logic_long": "close crosses above ST_LB_ATR10_2.0  AND  ADX_14 > 18",
        "logic_short": "close crosses below ST_UB_ATR10_2.0  AND  ADX_14 > 18",
        "stop_rule": "opposite ST band",
        "t1": "+2R",  "t2": "—",
        "trail": "ST band ratchet", "time_stop_bars": None,
        "partial": "50% at T1",
        "buckets": [
            ("cross + ADX>18", "FIRED",       "Supertrend regime change"),
            ("cross, ADX≤18",  "APPROACHING", ""),
            ("close near ST",  "NEAR",        ""),
            ("else",           "FAR",         ""),
        ],
    },

    "A12a": {
        "name": "ADX_DI_CROSS · EMA_20 variant",
        "family": FAMILY_TREND,
        "scopes": SCOPE_ALL,
        "logic_long": "DI+ crosses above DI-  AND  ADX_14 > 20  AND  close > EMA_20",
        "logic_short": "DI- crosses above DI+  AND  ADX_14 > 20  AND  close < EMA_20",
        "stop_rule": "2 × ATR_14",
        "t1": "+1.5R",  "t2": "—",
        "trail": "none", "time_stop_bars": 20, "partial": "50% at T1",
        "buckets": [
            ("3 of 3 met", "FIRED",       "directional dominance shift confirmed by short MA"),
            ("2 of 3 met", "NEAR",        ""),
            ("1 of 3 met", "APPROACHING", ""),
            ("0 met",      "FAR",         ""),
        ],
        "note": "fast-MA confirmation (EMA_20)",
    },

    "A12b": {
        "name": "ADX_DI_CROSS · EMA_50 variant",
        "family": FAMILY_TREND,
        "scopes": SCOPE_ALL,
        "logic_long": "DI+ crosses above DI-  AND  ADX_14 > 20  AND  close > EMA_50",
        "logic_short": "DI- crosses above DI+  AND  ADX_14 > 20  AND  close < EMA_50",
        "stop_rule": "2 × ATR_14",
        "t1": "+1.5R",  "t2": "—",
        "trail": "none", "time_stop_bars": 20, "partial": "50% at T1",
        "buckets": [
            ("3 of 3 met", "FIRED",       "directional dominance shift with medium-trend confirmation"),
            ("2 of 3 met", "NEAR",        ""),
            ("1 of 3 met", "APPROACHING", ""),
            ("0 met",      "FAR",         ""),
        ],
        "note": "medium-MA confirmation (EMA_50)",
    },

    "A15": {
        "name": "TRIANGLE_WEDGE",
        "family": FAMILY_TREND,
        "scopes": SCOPE_ALL,
        "logic_long": "polyfit on last 4-6 swing-highs & swing-lows: lines converging "
                       "(slopes opposite-signed)  AND  close breaks above upper trendline",
        "logic_short": "lines converging  AND  close breaks below lower trendline",
        "stop_rule": "opposite trendline",
        "t1": "half pattern-height",  "t2": "full pattern-height",
        "trail": "none", "time_stop_bars": 30, "partial": "50% at T1",
        "buckets": [
            ("converging + breakout",        "FIRED",       "classical pattern fire"),
            ("converging, no breakout yet",  "NEAR",        "watch for break"),
            ("partial swing data (<4 each)", "APPROACHING", "pattern setup forming"),
            ("else",                          "FAR",         ""),
        ],
    },

    # =====================================================================
    # MEAN-REVERSION setups (7)
    # =====================================================================
    "B1": {
        "name": "BB_2SIGMA_REVERSION",
        "family": FAMILY_MR,
        "scopes": SCOPE_ALL,
        "logic_long": "close ≤ BB_DN_20_2.0  AND  ADX_14 < 20  AND  RSI bullish divergence (20d)",
        "logic_short": "close ≥ BB_UP_20_2.0  AND  ADX_14 < 20  AND  RSI bearish divergence (20d)",
        "stop_rule": "BB_DN_20_3.0  (LONG) / BB_UP_20_3.0  (SHORT)",
        "t1": "BB_MID_20",  "t2": "opposite BB_2σ band",
        "trail": "none", "time_stop_bars": 8, "partial": "50% at T1",
        "buckets": [
            ("3 of 3 met", "FIRED",       "BB extreme + ranging + divergence — high-quality fade"),
            ("2 of 3 met", "NEAR",        ""),
            ("1 of 3 met", "APPROACHING", ""),
            ("0 met",      "FAR",         ""),
        ],
    },

    "B3": {
        "name": "RSI_DIVERGENCE_REVERSAL",
        "family": FAMILY_MR,
        "scopes": SCOPE_ALL,
        "logic_long": "new 20d low close  AND  RSI HL  AND  bullish reversal candle (close>open, body≥0.5×range)",
        "logic_short": "new 20d high close  AND  RSI LH  AND  bearish reversal candle",
        "stop_rule": "1.5 × ATR_14",
        "t1": "+1R",  "t2": "RSI = 50",
        "trail": "none", "time_stop_bars": 10, "partial": "50% at T1",
        "buckets": [
            ("3 of 3 met", "FIRED",       "classical RSI divergence reversal at swing extreme"),
            ("2 of 3 met", "NEAR",        ""),
            ("1 of 3 met", "APPROACHING", ""),
            ("0 met",      "FAR",         ""),
        ],
    },

    "B5": {
        "name": "ZPRICE_REVERSION",
        "family": FAMILY_MR,
        "scopes": SCOPE_ALL,
        "logic_long": "ZPRICE_20 < -2  AND  ZPRICE_50 < -1.5  AND  ADX_14 < 25",
        "logic_short": "ZPRICE_20 > +2  AND  ZPRICE_50 > +1.5  AND  ADX_14 < 25",
        "stop_rule": "dynamic — z = ±3 (price = SMA_20 ± 3·std_20)",
        "t1": "SMA_20 (z=0)",  "t2": "—",
        "trail": "none", "time_stop_bars": 20, "partial": "50% at T1",
        "buckets": [
            ("3 of 3 met", "FIRED",       "double-window stretch in non-trending regime"),
            ("2 of 3 met", "NEAR",        ""),
            ("1 of 3 met", "APPROACHING", ""),
            ("0 met",      "FAR",         ""),
        ],
    },

    "B6": {
        "name": "ZRET_5D_REVERSAL",
        "family": FAMILY_MR,
        "scopes": SCOPE_ALL,
        "logic_long": "ZRET_5 < -2.5",
        "logic_short": "ZRET_5 > +2.5",
        "stop_rule": "2 × ATR_14",
        "t1": "ZRET_5 = 0",  "t2": "—",
        "trail": "none", "time_stop_bars": 5, "partial": "none",
        "buckets": [
            ("|ZRET_5| ≥ 2.5",     "FIRED",       "5d return outlier — fade short-cycle exhaust"),
            ("2.0 ≤ |ZRET_5| < 2.5", "NEAR",      "approaching outlier band"),
            ("1.5 ≤ |ZRET_5| < 2.0", "APPROACHING", ""),
            ("|ZRET_5| < 1.5",     "FAR",         "no signal"),
        ],
    },

    "B10": {
        "name": "VOLUME_CLIMAX_FADE",
        "family": FAMILY_MR,
        "scopes": SCOPE_ALL,
        "logic_long": ("vol_z(90d) ≥ 3  AND  TR_z(90d) ≥ 2  AND  close in bottom 25% of bar  "
                       "AND  close < BB_DN_20_2.0"),
        "logic_short": ("vol_z(90d) ≥ 3  AND  TR_z(90d) ≥ 2  AND  close in top 25% of bar  "
                        "AND  close > BB_UP_20_2.0"),
        "stop_rule": "bar low (LONG) / bar high (SHORT)",
        "t1": "EMA_20",  "t2": "—",
        "trail": "none", "time_stop_bars": 3, "partial": "none",
        "buckets": [
            ("4 of 4 met", "FIRED",       "capitulation candle — climactic extreme"),
            ("3 of 4 met", "NEAR",        ""),
            ("2 of 4 met", "APPROACHING", ""),
            ("≤1 met",     "FAR",         ""),
        ],
        "note": "windows = 90d (not 252d / 60d) per Yash customization 2026-05-07",
    },

    "B11": {
        "name": "OU_HALF_LIFE_REVERSION",
        "family": FAMILY_MR,
        "scopes": SCOPE_ALL,
        "logic_long": "OU half-life < 30  AND  z(close, 60d) < -2",
        "logic_short": "OU half-life < 30  AND  z(close, 60d) > +2",
        "stop_rule": "dynamic — z = ±3",
        "t1": "z = 0",  "t2": "—",
        "trail": "none", "time_stop_bars": "HL × 2 (dynamic)", "partial": "50% at T1",
        "buckets": [
            ("HL<30 & |z|≥2", "FIRED",       "high-quality OU mean-reversion"),
            ("HL<30 & |z|≥1.5", "NEAR",       ""),
            ("HL<30 & |z|<1.5", "APPROACHING", "fast reverter, not yet stretched"),
            ("HL≥30 or no fit", "FAR",         "not OU-style mean-reverting"),
        ],
    },

    "B13": {
        "name": "HURST_FILTERED_MEAN_REVERSION",
        "family": FAMILY_MR,
        "scopes": SCOPE_ALL,
        "logic_long": "Hurst_60 < 0.45  AND  ZPRICE_20 < -2",
        "logic_short": "Hurst_60 < 0.45  AND  ZPRICE_20 > +2",
        "stop_rule": "dynamic — z = ±3",
        "t1": "SMA_20 (z = 0)",  "t2": "—",
        "trail": "none", "time_stop_bars": 20, "partial": "50% at T1",
        "buckets": [
            ("Hurst<0.45 & |z|≥2", "FIRED",       "anti-persistent regime + stretched — fade"),
            ("Hurst<0.45 & |z|≥1.5", "NEAR",      ""),
            ("Hurst<0.45 & |z|<1.5", "APPROACHING", "reverting regime, not yet stretched"),
            ("Hurst≥0.45",         "FAR",         "not anti-persistent — fade risky"),
        ],
    },

    # =====================================================================
    # STIR-specific setups (5 base; C8 has 3 variants, C9 has 2)
    # =====================================================================
    "C3": {
        "name": "INTRA_CURVE_CARRY_RANK",
        "family": FAMILY_STIR,
        "scopes": (SCOPE_OUTRIGHT,),
        "logic_long": "rank by carry/day within market: long top decile + close > EMA_20",
        "logic_short": "short bottom decile + close < EMA_20",
        "stop_rule": "2 × ATR_14",
        "t1": "+1R",  "t2": "—",
        "trail": "none", "time_stop_bars": 21, "partial": "none",
        "buckets": [
            ("top decile + EMA20 ✓",   "FIRED-LONG",  "long top-carry contract aligned with trend"),
            ("bottom decile + EMA20 ✗", "FIRED-SHORT", "short bottom-carry contract"),
            ("near top/bottom decile",  "NEAR",        ""),
            ("middle of carry rank",    "FAR",         ""),
        ],
    },

    "C4": {
        "name": "TRADITIONAL_FLY_MR",
        "family": FAMILY_STIR,
        "scopes": (SCOPE_FLY,),
        "logic_long": "raw fly z(60d) < -2",
        "logic_short": "raw fly z(60d) > +2",
        "stop_rule": "z = ±3",
        "t1": "z = 0",  "t2": "—",
        "trail": "none", "time_stop_bars": 30, "partial": "50% at T1",
        "buckets": [
            ("|z|≥2",   "FIRED",       "raw fly z extreme — fade"),
            ("|z|≥1.5", "NEAR",        ""),
            ("|z|≥1",   "APPROACHING", ""),
            ("|z|<1",   "FAR",         ""),
        ],
    },

    "C5": {
        "name": "CALENDAR_SPREAD_MR",
        "family": FAMILY_STIR,
        "scopes": (SCOPE_SPREAD,),
        "logic_long": "spread z(60d) < -2",
        "logic_short": "spread z(60d) > +2",
        "stop_rule": "z = ±3",
        "t1": "z = 0",  "t2": "—",
        "trail": "none", "time_stop_bars": 20, "partial": "50% at T1",
        "buckets": [
            ("|z|≥2",   "FIRED",       "spread z extreme — fade"),
            ("|z|≥1.5", "NEAR",        ""),
            ("|z|≥1",   "APPROACHING", ""),
            ("|z|<1",   "FAR",         ""),
        ],
    },

    "C8_12M": {
        "name": "TERMINAL_RATE_REPRICE · 12M",
        "family": FAMILY_STIR,
        "scopes": (SCOPE_OUTRIGHT,),
        "logic_long": "terminal_12m moves ≥ +25 bp over 5 days",
        "logic_short": "terminal_12m moves ≤ -25 bp over 5 days",
        "stop_rule": "—",
        "t1": "—",  "t2": "—",
        "trail": "none", "time_stop_bars": 21, "partial": "none",
        "buckets": [
            ("|Δ5d| ≥ 25bp",         "FIRED",       "near-term cycle revaluation"),
            ("|Δ5d| ≥ 15bp",         "NEAR",        ""),
            ("|Δ5d| ≥ 10bp",         "APPROACHING", ""),
            ("|Δ5d| < 10bp",         "FAR",         ""),
        ],
    },

    "C8_24M": {
        "name": "TERMINAL_RATE_REPRICE · 24M (PRIMARY)",
        "family": FAMILY_STIR,
        "scopes": (SCOPE_OUTRIGHT,),
        "logic_long": "terminal_24m moves ≥ +25 bp over 5 days",
        "logic_short": "terminal_24m moves ≤ -25 bp over 5 days",
        "stop_rule": "—",
        "t1": "—",  "t2": "—",
        "trail": "none", "time_stop_bars": 21, "partial": "none",
        "buckets": [
            ("|Δ5d| ≥ 25bp",         "FIRED",       "primary cycle revaluation — most actionable"),
            ("|Δ5d| ≥ 15bp",         "NEAR",        ""),
            ("|Δ5d| ≥ 10bp",         "APPROACHING", ""),
            ("|Δ5d| < 10bp",         "FAR",         ""),
        ],
    },

    "C8_36M": {
        "name": "TERMINAL_RATE_REPRICE · 36M",
        "family": FAMILY_STIR,
        "scopes": (SCOPE_OUTRIGHT,),
        "logic_long": "terminal_36m moves ≥ +25 bp over 5 days",
        "logic_short": "terminal_36m moves ≤ -25 bp over 5 days",
        "stop_rule": "—",
        "t1": "—",  "t2": "—",
        "trail": "none", "time_stop_bars": 21, "partial": "none",
        "buckets": [
            ("|Δ5d| ≥ 25bp",         "FIRED",       "long-horizon revaluation"),
            ("|Δ5d| ≥ 15bp",         "NEAR",        ""),
            ("|Δ5d| ≥ 10bp",         "APPROACHING", ""),
            ("|Δ5d| < 10bp",         "FAR",         ""),
        ],
    },

    "C9a": {
        "name": "CURVE_STEEPENER / FLATTENER · slope crossing",
        "family": FAMILY_STIR,
        "scopes": (SCOPE_OUTRIGHT,),
        "logic_long": "slope crosses above 0 (front rate falls below back rate today)",
        "logic_short": "slope crosses below 0",
        "stop_rule": "—",
        "t1": "—",  "t2": "—",
        "trail": "none", "time_stop_bars": 30, "partial": "none",
        "buckets": [
            ("slope cross today",    "FIRED",       "curve-shape regime change today"),
            ("|slope| < 1bp",        "NEAR",        "approaching cross"),
            ("|slope| < 3bp",        "APPROACHING", ""),
            ("else",                 "FAR",         ""),
        ],
        "note": "slope = back_rate − front_rate; default front = nearest quarterly, back = +12 months",
    },

    "C9b": {
        "name": "CURVE_STEEPENER / FLATTENER · 5d trend",
        "family": FAMILY_STIR,
        "scopes": (SCOPE_OUTRIGHT,),
        "logic_long": "slope 5d trend ≥ +5 bp (steepening)",
        "logic_short": "slope 5d trend ≤ -5 bp (flattening)",
        "stop_rule": "—",
        "t1": "—",  "t2": "—",
        "trail": "none", "time_stop_bars": 30, "partial": "none",
        "buckets": [
            ("|Δ5d slope| ≥ 5bp",   "FIRED",       "directional curve trend"),
            ("|Δ5d slope| ≥ 3bp",   "NEAR",        ""),
            ("|Δ5d slope| ≥ 1.5bp", "APPROACHING", ""),
            ("else",                "FAR",         ""),
        ],
    },

    # =====================================================================
    # COMPOSITE scoring — registry entries (math lives in lib/setups/composite.py)
    # =====================================================================
    "TREND_COMPOSITE": {
        "name": "TREND_COMPOSITE",
        "family": FAMILY_COMPOSITE,
        "scopes": SCOPE_ALL,
        "logic_long": "scope-specific factor mean clipped [-1,+1]; positive = bullish trend posture",
        "logic_short": "negative = bearish trend posture",
        "stop_rule": "—", "t1": "—", "t2": "—", "trail": None, "time_stop_bars": None, "partial": None,
        "buckets": [
            ("≥ +0.7", "STRONG ↑",     "strong bullish trend posture — high-conviction"),
            ("≥ +0.5", "moderate ↑",   "moderate trending bias up"),
            ("≥ +0.3", "soft ↑",       "soft tilt up"),
            ("|x|<0.3","neutral",      "no trend posture"),
            ("≤ -0.3", "soft ↓",       "soft tilt down"),
            ("≤ -0.5", "moderate ↓",   "moderate trending bias down"),
            ("≤ -0.7", "STRONG ↓",     "strong bearish trend posture"),
        ],
    },
    "MR_COMPOSITE": {
        "name": "MR_COMPOSITE",
        "family": FAMILY_COMPOSITE,
        "scopes": SCOPE_ALL,
        "logic_long": "scope-specific MR factor mean. Positive = stretched LOW (fade up); negative = stretched HIGH (fade down)",
        "stop_rule": "—", "t1": "—", "t2": "—", "trail": None, "time_stop_bars": None, "partial": None,
        "buckets": [
            ("≥ +0.7", "STRETCHED LOW · fade up",   "strong long-fade signal — highest-quality MR"),
            ("≥ +0.5", "moderate fade up",          ""),
            ("|x|<0.3","neutral",                   ""),
            ("≤ -0.5", "moderate fade down",        ""),
            ("≤ -0.7", "STRETCHED HIGH · fade down", "strong short-fade signal"),
        ],
    },
    "FINAL_COMPOSITE": {
        "name": "FINAL_COMPOSITE",
        "family": FAMILY_COMPOSITE,
        "scopes": SCOPE_ALL,
        "logic_long": ("regime-weighted blend: w_trend·TREND − w_mr·{ADF·Hurst·OU gate}·MR + carry_tilt(outright). "
                       "weights and gates differ per scope — see lib/setups/composite.py"),
        "stop_rule": "—", "t1": "—", "t2": "—", "trail": None, "time_stop_bars": None, "partial": None,
        "buckets": [
            ("≥ +0.7", "STRONG SIGNAL ↑",  "high-conviction long — combined with ≥2 trend setups same direction"),
            ("≥ +0.5", "moderate ↑",       ""),
            ("≥ +0.3", "soft ↑",           ""),
            ("|x|<0.3","neutral · no edge",""),
            ("≤ -0.3", "soft ↓",           ""),
            ("≤ -0.5", "moderate ↓",       ""),
            ("≤ -0.7", "STRONG SIGNAL ↓",  "high-conviction short"),
        ],
    },
}


# =============================================================================
# Convenience: ordered ID lists per family
# =============================================================================
TREND_IDS = ["A1", "A2", "A3", "A4", "A5", "A6", "A8", "A10", "A11", "A12a", "A12b", "A15"]
MR_IDS = ["B1", "B3", "B5", "B6", "B10", "B11", "B13"]
STIR_IDS = ["C3", "C4", "C5", "C8_12M", "C8_24M", "C8_36M", "C9a", "C9b"]
COMPOSITE_IDS = ["TREND_COMPOSITE", "MR_COMPOSITE", "FINAL_COMPOSITE"]
ALL_SETUP_IDS = TREND_IDS + MR_IDS + STIR_IDS  # composites are computed separately


def get_registry_entry(setup_id: str) -> dict:
    """Return the registry dict for a setup id; empty dict if unknown.

    Falls back to a base id for offset-suffixed variants like
    ``C9a_6M`` / ``C9a_12M`` / ``C9b_24M`` — these resolve to the ``C9a`` /
    ``C9b`` registry entry, with the offset rendered into the displayed name.
    """
    if setup_id in SETUP_REGISTRY:
        return SETUP_REGISTRY[setup_id]
    # Strip a trailing _NM offset suffix (e.g. "C9a_12M" → "C9a")
    if "_" in setup_id:
        head, _, tail = setup_id.rpartition("_")
        if tail.endswith("M") and tail[:-1].isdigit() and head in SETUP_REGISTRY:
            base = SETUP_REGISTRY[head].copy()
            # Append offset to the displayed name for clarity in the UI
            if "name" in base:
                base["name"] = f"{base['name']} · {tail} slope"
            base["_variant_offset"] = int(tail[:-1])
            return base
    return {}


# =============================================================================
# Phase F — Title-Case display names (no code names anywhere in user UI)
# =============================================================================

_DISPLAY_NAME_CACHE: dict[str, str] = {}

# Acronyms to preserve as ALL-CAPS in display names. Order matters — longer
# strings first to prevent partial matches (e.g. "BB" before "B" alone).
_PRESERVE_ACRONYMS = (
    "ADX", "ADF", "ATR", "ADR", "BBWP", "BB", "CCI", "CRSI", "DI", "DV01",
    "EMA", "FF", "FOMC", "FX", "IBS", "MACD", "MR", "NPS", "OHLC", "OU",
    "PCA", "PNL", "RSI", "SMA", "SOFR", "SR", "STIR", "TR", "TC",
    "TBF3", "VCP", "VIX",
)
_PRESERVE_LOWER = ("d",)   # e.g. "5d" stays "5d"


def _titlecase_word(w: str) -> str:
    """Title-case a single word with acronym preservation.

    - "rsi" → "RSI"
    - "5d"  → "5d"
    - "12M" → "12M"
    - "ema_20" → "EMA 20"
    - "z-price" → "Z-Price"
    """
    if not w:
        return w
    upper = w.upper()
    if upper in _PRESERVE_ACRONYMS:
        return upper
    if w in _PRESERVE_LOWER:
        return w
    # Pure-numeric tokens (e.g. "20") stay literal
    if w.isdigit():
        return w
    # Trailing 'M' on a number (e.g. "12M", "24M") — preserve uppercase
    if len(w) >= 2 and w[:-1].isdigit() and w[-1].upper() == "M":
        return w.upper()
    # Number-letter combos like "5d" — preserve lowercase suffix
    if len(w) >= 2 and w[:-1].isdigit() and w[-1].lower() in _PRESERVE_LOWER:
        return w.lower()
    # Hyphenated words: title-case each part
    if "-" in w:
        return "-".join(_titlecase_word(p) for p in w.split("-"))
    return w[0].upper() + w[1:].lower()


def display_name(setup_id: str) -> str:
    """Convert the registry name to Title Case with spaces.

    ``TREND_CONTINUATION_BREAKOUT`` → ``Trend Continuation Breakout``
    ``ADX_DI_CROSS · EMA_20`` → ``ADX DI Cross · EMA 20``
    ``CURVE_STEEPENER / FLATTENER · slope crossing · 12M slope`` →
        ``Curve Steepener / Flattener · Slope Crossing · 12M Slope``

    Memoised for speed.
    """
    if setup_id in _DISPLAY_NAME_CACHE:
        return _DISPLAY_NAME_CACHE[setup_id]
    reg = get_registry_entry(setup_id)
    raw = reg.get("name") if reg else None
    if not raw:
        out = setup_id
    else:
        # First, replace underscores with spaces; then iterate token by token
        words = []
        for tok in raw.replace("_", " ").split(" "):
            if not tok:
                continue
            if any(ch.isalnum() for ch in tok):
                words.append(_titlecase_word(tok))
            else:
                words.append(tok)
        out = " ".join(words)
    _DISPLAY_NAME_CACHE[setup_id] = out
    return out


def display_name_short(setup_id: str, max_len: int = 14) -> str:
    """For dense matrix column headers — abbreviate to ``max_len`` chars
    by taking the first letter of each title-cased word; padding with the
    full name's prefix if there are too few words.

    ``Trend Continuation Breakout`` → ``TCB``
    ``Bb 2sigma Reversion`` → ``B2R``
    """
    full = display_name(setup_id)
    if len(full) <= max_len:
        return full
    initials = "".join(w[0].upper() for w in full.split() if w and w[0].isalpha())
    if 2 <= len(initials) <= max_len:
        return initials
    # Fall back to truncation
    return full[:max_len].rstrip()


def applies_to(setup_id: str, scope: str) -> bool:
    """Whether a setup applies to a given scope ('outright' / 'spread' / 'fly')."""
    entry = SETUP_REGISTRY.get(setup_id, {})
    return scope in entry.get("scopes", ())


def setups_for_scope(scope: str, families: Optional[tuple] = None) -> list:
    """Return list of setup ids applicable to a scope (optionally filtered by family)."""
    out = []
    for sid in ALL_SETUP_IDS:
        e = SETUP_REGISTRY.get(sid, {})
        if scope not in e.get("scopes", ()):
            continue
        if families is not None and e.get("family") not in families:
            continue
        out.append(sid)
    return out
