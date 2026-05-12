"""Citations for the NEAR / APPROACHING threshold methodology.

Each setup category in lib/setups/near_thresholds.py is anchored to one of
these literature sources. The citation strings are surfaced via
lib/setups/tooltips.py so the trader can audit *why* a NEAR threshold was
chosen.

Full citation list lives in:
  C:\\Users\\yash.patel\\.claude\\plans\\c-users-yash-patel-downloads-tmia-v13-s-magical-mitten-agent-a1bd94828a312a17f.md

and HANDOFF.md (§5E for CMC; Phase B section once added).
"""
from __future__ import annotations


CITATIONS: dict[str, str] = {
    "wilder_atr":           "Wilder 1978, *New Concepts in Technical Trading "
                              "Systems* — ATR / N-based proximity (Turtle convention).",
    "wilder_rsi":           "Wilder 1978, *New Concepts in Technical Trading "
                              "Systems* — RSI 30/70 + 50 midline + failure-swing reversals.",
    "wilder_adx":           "Wilder 1978 — ADX 25 = strong trend; 20-25 = "
                              "emerging-trend watch zone.",
    "brown_rsi":            "Brown 1999 (2nd ed. 2011), *Technical Analysis "
                              "for the Trading Professional* — bull-market RSI "
                              "range 40-80 (40-50 = pullback support); "
                              "bear-market 20-60 (50-60 = retracement resistance).",
    "connors_crsi":         "Connors / Connors Research, ConnorsRSI — 10/90 "
                              "oversold/overbought; 5/95 for high-volatility names.",
    "carver_continuous":    "Carver 2015, *Systematic Trading* + pysystemtrade. "
                              "Continuous forecast on [-20,+20]; "
                              "forecast = 40·(price − roll_mean) / (roll_max − roll_min).",
    "carver_breakout":      "Carver, qoppac.blogspot.com — breakout rule with "
                              "continuous strength score.",
    "demark_td":            "DeMark TD Sequential — 1-9 setup count (bars 6-8 "
                              "are the practitioner-accepted anticipation states).",
    "bollinger_bandwidth":  "Bollinger 2010 BandWidth — *BandWidth is narrow when "
                              "less than 4% of price*; squeeze = 6-month low or "
                              "BBWP < 15th percentile.",
    "carter_ttm_squeeze":   "Carter, TTM Squeeze — Bollinger inside Keltner, "
                              "fires on expansion.",
    "avellaneda_lee":       "Avellaneda & Lee 2010, *Statistical Arbitrage in "
                              "the U.S. Equities Market*, Quantitative Finance — "
                              "OU-based s-scores; entry at |z| ∈ [1.25, 2.0].",
    "macrosynergy":         "Macrosynergy research blog — signal-quality "
                              "framework: 54-58% directional accuracy is realistic; "
                              "winsorise at ±2σ.",
    "aronson":              "Aronson 2007, *Evidence-Based Technical Analysis* — "
                              "bootstrap / Monte Carlo for thresholds; "
                              "data-mining-bias correction.",
    "turtle":               "Turtle Trading rules — ATR-units (N) for proximity, "
                              "stops, and pyramiding.",
    "empirical_bootstrap":  "Empirical bootstrap-quantile calibration on the "
                              "5-year CMC history (Aronson 2007 method, "
                              "Macrosynergy ±2σ winsor).",
}


def cite(key: str) -> str:
    """Look up a citation. Returns ``"(citation missing: <key>)"`` if not found
    so the caller doesn't fail silently."""
    return CITATIONS.get(key, f"(citation missing: {key})")
