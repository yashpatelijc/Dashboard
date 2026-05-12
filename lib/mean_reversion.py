"""Mean-reversion engine â€” Z-score, OU half-life, Hurst exponent (R/S), ADF
stationarity, composite reversion-candidate score â€” for the SRA â†’ Analysis â†’
Z-score / Mean-Reversion subtab.

All math respects the LOCKED conventions:

  Â· History windows EXCLUDE the as-of date itself (`< ts`).
  Â· ``ddof=1`` for sample std (matches lib.sra_data.compute_per_contract_zscores).
  Â· Catalog-aware bp scaling via ``lib.contract_units.bp_multipliers_for``.
  Â· Tests are *robust* â€” short series, NaNs, and degenerate cases return ``None``
    rather than raising.

All implementations are SCIPY-FREE (only numpy + pandas) so the module works
in the locked qh_data env without adding deps.
"""
from __future__ import annotations

from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from lib.contract_units import bp_multipliers_for, load_catalog


# =============================================================================
# Interpretation helpers â€” plain-English labels for every metric.
# Each helper returns (label, color_var) so the UI can render colored badges.
# Color vars match the locked theme (red=risky/extreme, green=favorable,
# amber=transitional, text-muted=neutral, text-dim=missing).
# =============================================================================
def z_interpretation(z) -> tuple[str, str]:
    """Z-score â†’ interpretation."""
    if z is None or pd.isna(z):
        return ("â€”", "var(--text-dim)")
    if z >= 2:
        return ("stretched HIGH (>+2Ïƒ)", "var(--red)")
    if z >= 1:
        return ("elevated HIGH (+1 to +2Ïƒ)", "var(--amber)")
    if z <= -2:
        return ("stretched LOW (<âˆ’2Ïƒ)", "var(--red)")
    if z <= -1:
        return ("elevated LOW (âˆ’1 to âˆ’2Ïƒ)", "var(--amber)")
    return ("neutral (|z|<1Ïƒ)", "var(--text-muted)")


def pct_rank_interpretation(pct) -> tuple[str, str]:
    """Percentile rank (0..100) â†’ interpretation."""
    if pct is None or pd.isna(pct):
        return ("â€”", "var(--text-dim)")
    if pct >= 90:
        return ("top decile (rich)", "var(--red)")
    if pct >= 75:
        return ("upper quartile", "var(--amber)")
    if pct >= 25:
        return ("middle range", "var(--text-muted)")
    if pct >= 10:
        return ("lower quartile", "var(--amber)")
    return ("bottom decile (cheap)", "var(--green)")


def half_life_interpretation(hl) -> tuple[str, str]:
    """OU half-life (trading days) â†’ interpretation.

    Buckets:
      â‰¤ 5d  very fast â€” strong reversion, intra-week
      â‰¤10d  fast
      â‰¤30d  medium
      â‰¤60d  slow
      >60d  very slow â€” barely reverting
      None  no reversion fit (AR(1) coef outside (0,1))
    """
    if hl is None or pd.isna(hl) or not np.isfinite(hl):
        return ("no reversion fit", "var(--text-dim)")
    if hl <= 5:
        return ("very fast (â‰¤5d)", "var(--green)")
    if hl <= 10:
        return ("fast (5-10d)", "var(--green)")
    if hl <= 30:
        return ("medium (10-30d)", "var(--amber)")
    if hl <= 60:
        return ("slow (30-60d)", "var(--text-muted)")
    return ("very slow (>60d)", "var(--text-muted)")


def hurst_interpretation(h) -> tuple[str, str]:
    """Hurst exponent â†’ interpretation.

    Buckets:
      â‰¥0.65   strongly trending â€” momentum dominant
      â‰¥0.55   trending â€” persistent moves
       ~0.5   random walk â€” no memory
      â‰¤0.45   reverting â€” anti-persistent
      â‰¤0.35   strongly reverting â€” fade-friendly
    """
    if h is None or pd.isna(h):
        return ("â€”", "var(--text-dim)")
    if h >= 0.65:
        return ("strongly trending (Hâ‰¥0.65)", "var(--red)")
    if h >= 0.55:
        return ("trending (Hâ‰¥0.55)", "var(--red)")
    if h >= 0.45:
        return ("random walk (Hâ‰ˆ0.5)", "var(--text-muted)")
    if h >= 0.35:
        return ("reverting (Hâ‰¤0.45)", "var(--green)")
    return ("strongly reverting (Hâ‰¤0.35)", "var(--green)")


def adf_interpretation(reject_5pct: bool, pvalue) -> tuple[str, str]:
    """ADF result â†’ interpretation.

    STATIONARY (reject)  â†’ series mean-reverts; Z-score signal trustworthy.
    BORDERLINE (p<0.10)  â†’ suggestive but not conclusive.
    NON-STATIONARY        â†’ series may be trending; high |z| could keep extending.
    """
    if pvalue is None:
        return ("â€”", "var(--text-dim)")
    if reject_5pct:
        return ("stationary â€” fade-trustworthy", "var(--green)")
    if pvalue == "<0.10":
        return ("borderline â€” weak evidence", "var(--amber)")
    return ("non-stationary â€” caution fading", "var(--red)")


def velocity_to_mean_interpretation(vel) -> tuple[str, str]:
    """Avg daily Î”|z| over last 5 bars â†’ interpretation.

    Negative = |z| shrinking (returning to mean).
    """
    if vel is None or pd.isna(vel):
        return ("â€”", "var(--text-dim)")
    if vel <= -0.10:
        return ("rapidly reverting", "var(--green)")
    if vel < -0.02:
        return ("drifting back to mean", "var(--green)")
    if vel <= 0.02:
        return ("stable (no velocity)", "var(--text-muted)")
    if vel <= 0.10:
        return ("still extending", "var(--amber)")
    return ("rapidly extending", "var(--red)")


def reversion_score_interpretation(score) -> tuple[str, str]:
    """Composite reversion score (0..1) â†’ interpretation."""
    if score is None or pd.isna(score):
        return ("â€”", "var(--text-dim)")
    if score >= 0.70:
        return ("strong reversion setup", "var(--green)")
    if score >= 0.50:
        return ("moderate setup", "var(--amber)")
    if score >= 0.30:
        return ("weak setup", "var(--text-muted)")
    return ("very weak", "var(--text-dim)")


def trend_score_interpretation(score) -> tuple[str, str]:
    """Composite trend-confirmation score (0..1) â†’ interpretation."""
    if score is None or pd.isna(score):
        return ("â€”", "var(--text-dim)")
    if score >= 0.70:
        return ("strong trend â€” DON'T fade", "var(--red)")
    if score >= 0.50:
        return ("moderate trend", "var(--amber)")
    if score >= 0.30:
        return ("weak trend", "var(--text-muted)")
    return ("very weak", "var(--text-dim)")


def overall_setup_interpretation(z, hurst, half_life, adf_reject) -> tuple[str, str]:
    """Compose a single TAKE label combining z-magnitude, Hurst, half-life and ADF.

    The TAKE answers 'what should the trader actually do?' in plain language.
    """
    if z is None or pd.isna(z):
        return ("no signal", "var(--text-dim)")
    az = abs(z)
    direction = "HIGH" if z > 0 else "LOW"
    is_trending = hurst is not None and not pd.isna(hurst) and hurst >= 0.55
    is_reverting = hurst is not None and not pd.isna(hurst) and hurst <= 0.45
    is_stationary = bool(adf_reject)
    fast = (half_life is not None and not pd.isna(half_life)
            and np.isfinite(half_life) and half_life <= 15)

    if az < 1:
        return ("neutral Â· no edge", "var(--text-muted)")
    if az >= 2 and is_stationary and (is_reverting or fast):
        return (f"FADE {direction} â€” high-quality setup", "var(--green)")
    if az >= 2 and is_trending and not is_stationary:
        return (f"RIDE TREND {direction} â€” don't fade", "var(--red)")
    if az >= 2 and is_stationary:
        return (f"FADE {direction} â€” stationary candidate", "var(--green)")
    if az >= 2 and is_trending:
        return (f"RIDE TREND {direction} â€” trending memory", "var(--red)")
    if az >= 2:
        return (f"stretched {direction} â€” context-dependent", "var(--amber)")
    if 1 <= az < 2 and is_trending and is_stationary is False:
        return (f"early trend {direction}", "var(--amber)")
    if 1 <= az < 2 and is_reverting:
        return (f"early fade {direction}", "var(--green)")
    return (f"elevated {direction}", "var(--amber)")


def metric_tooltip(guide: dict, metric_key: str, current_label: Optional[str] = None,
                   computed: Optional[str] = None) -> str:
    """Build a multi-line tooltip string for a single metric.

    Includes:
      - Title (metric name)
      - Current bucket marker (if `current_label` matches one of the buckets)
      - What the metric is + formula
      - Threshold table â€” every bucket condition â†’ label, with the matching
        bucket marked with âœ“ when known
      - Computed value (if `computed` provided) â€” e.g. "z = (96.345 âˆ’ 96.319) / 0.033 = +0.77Ïƒ"

    Used by tooltips on every numeric cell so the trader sees the formula, the
    inputs that produced this value, AND the bucket boundaries that drove the
    interpretation â€” without leaving the cell.
    """
    info = guide.get(metric_key)
    if not info:
        return ""
    lines = [metric_key.upper()]
    if current_label:
        lines.append(f"   Current: {current_label}")
    lines.append("")
    what = info.get("what", "")
    formula = info.get("formula", "")
    if what:
        lines.append(f"What:    {what}")
    if formula:
        lines.append(f"Formula: {formula}")
    buckets = info.get("buckets", [])
    if buckets:
        lines.append("")
        lines.append("Thresholds:")
        for cond, label, meaning in buckets:
            marker = "âœ“ " if (current_label and label == current_label) else "  "
            lines.append(f"  {marker}{cond}  â†’  {label}")
            lines.append(f"        {meaning}")
    if computed:
        lines.append("")
        lines.append(f"Computed: {computed}")
    return "\n".join(lines)


def get_interpretation_guide() -> dict:
    """Return all interpretation buckets as a structured dict.

    Used by the UI to render an 'Interpretation guide' expander listing
    every metric, its formula sketch, and the bucket boundaries with
    plain-English meanings.
    """
    return {
        "Z-score (z)": {
            "what":  "How many sample-standard-deviations the current close is from the prior N-day mean.",
            "formula": "z = (today âˆ’ Î¼_N) / Ïƒ_N   (history excludes today; ddof=1)",
            "buckets": [
                ("|z| < 1",          "neutral",                  "no edge â€” quiet contract"),
                ("+1 â‰¤ z < +2",      "elevated HIGH",            "modest stretch up"),
                ("z â‰¥ +2",           "stretched HIGH",           "extreme â€” potential fade"),
                ("âˆ’2 < z â‰¤ âˆ’1",      "elevated LOW",             "modest stretch down"),
                ("z â‰¤ âˆ’2",           "stretched LOW",            "extreme â€” potential fade"),
            ],
        },
        "Percentile rank": {
            "what":  "Where today sits in the empirical distribution of the prior N days (0..100%).",
            "formula": "pct = (#prior_below / N) Ã— 100",
            "buckets": [
                ("â‰¥ 90%",            "top decile (rich)",        "current above 90% of prior days"),
                ("75â€“90%",           "upper quartile",           "leaning rich"),
                ("25â€“75%",           "middle range",             "neutral"),
                ("10â€“25%",           "lower quartile",           "leaning cheap"),
                ("â‰¤ 10%",            "bottom decile (cheap)",    "current below 90% of prior days"),
            ],
        },
        "OU half-life": {
            "what":  "Time (in trading days) for a deviation from the mean to halve, "
                     "assuming Ornstein-Uhlenbeck dynamics. Shorter = stronger reversion.",
            "formula": "h = âˆ’ln(2) / ln(b)   where b is the AR(1) coefficient on x_{t-1}; "
                       "fit on the last 90 bars",
            "buckets": [
                ("â‰¤ 5d",             "very fast",                "strong intra-week reversion"),
                ("5â€“10d",            "fast",                     "reliable mean-reverter"),
                ("10â€“30d",           "medium",                   "typical"),
                ("30â€“60d",           "slow",                     "weak reversion"),
                ("> 60d",            "very slow",                "barely reverting"),
                ("None",             "no reversion fit",         "AR(1) coef outside (0,1) â€” series is trending or random"),
            ],
        },
        "Hurst exponent (H)": {
            "what":  "Long-memory exponent from R/S analysis on log-returns. "
                     "Tells you whether moves persist (trend) or reverse (mean-revert).",
            "formula": "log(R/S) â‰ˆ c + H Â· log(n);  H from log-log regression across "
                       "non-overlapping sub-windows; clipped to [0,1]",
            "buckets": [
                ("H â‰¥ 0.65",         "strongly trending",        "momentum dominant â€” fade with extreme caution"),
                ("0.55 â‰¤ H < 0.65",  "trending",                 "persistent moves â€” fading is risky"),
                ("0.45 < H < 0.55",  "random walk",              "no memory â€” pure noise"),
                ("0.35 < H â‰¤ 0.45",  "reverting",                "anti-persistent â€” fade-friendly"),
                ("H â‰¤ 0.35",         "strongly reverting",       "high-quality fade environment"),
            ],
        },
        "ADF (lag-1, no trend)": {
            "what":  "Augmented Dickey-Fuller stationarity test. Rejects unit-root Hâ‚€ when t-stat â‰¤ âˆ’2.86 (5% critical). "
                     "Stationary series mean-revert; non-stationary may be trending.",
            "formula": "Î”x_t = Î± + Î³Â·x_{t-1} + Ï†Â·Î”x_{t-1} + Îµ;  test on Î³. p-value mapped from MacKinnon table.",
            "buckets": [
                ("p < 0.05 (rejects)", "stationary",             "Z-score signal is trustworthy â€” reversion expected"),
                ("0.05 â‰¤ p < 0.10",   "borderline",              "weak evidence of stationarity"),
                ("p â‰¥ 0.10",          "non-stationary",          "could be trending â€” DO NOT fade on z alone"),
            ],
        },
        "Velocity to mean (Î”|z|/d)": {
            "what":  "Average daily change in |z-score| over the last 5 prior bars. "
                     "Tells you if the contract is reverting or still extending in real time.",
            "formula": "vel = mean(diff(|z_t|))  for t in last 6 bars, prior to today",
            "buckets": [
                ("â‰¤ âˆ’0.10",          "rapidly reverting",        "|z| shrinking fast â€” already moving back"),
                ("âˆ’0.10 to âˆ’0.02",   "drifting back to mean",    "modest reversion"),
                ("âˆ’0.02 to +0.02",   "stable",                   "no velocity â€” coiled at level"),
                ("+0.02 to +0.10",   "still extending",          "|z| growing â€” move not done"),
                ("> +0.10",          "rapidly extending",        "|z| accelerating â€” fade dangerous"),
            ],
        },
        "Composite reversion score": {
            "what":  "Weighted blend of |z|, Hurst, half-life, and ADF â€” ranks fade candidates.",
            "formula": "0.50Â·|z|/2 + 0.25Â·(1âˆ’H) + 0.15Â·10/max(1,h) + 0.10Â·1{ADF rejects}",
            "buckets": [
                ("â‰¥ 0.70",           "strong reversion setup",   "high-quality fade â€” multiple confirmations"),
                ("0.50â€“0.70",        "moderate setup",           "fade with sizing discipline"),
                ("0.30â€“0.50",        "weak setup",               "marginal â€” wait for better"),
                ("< 0.30",           "very weak",                "no fade edge"),
            ],
        },
        "Composite trend score": {
            "what":  "Weighted blend favouring high |z| + high Hurst + non-stationarity. "
                     "Identifies names you should NOT fade â€” let the trend run.",
            "formula": "0.55Â·|z|/2 + 0.35Â·H + 0.10Â·1{ADF can't reject}",
            "buckets": [
                ("â‰¥ 0.70",           "strong trend â€” DON'T fade", "trending memory + stretched |z| + non-stationary"),
                ("0.50â€“0.70",        "moderate trend",            "trend exists but caution warranted"),
                ("0.30â€“0.50",        "weak trend",                "lukewarm trend signal"),
                ("< 0.30",           "very weak",                 "no trend edge"),
            ],
        },
        "Multi-window confluence pattern": {
            "what":  "Classifies the cross-lookback (5/15/30/60/90d) Z structure into one of 8 named regimes.",
            "formula": "see lib.mean_reversion.classify_z_pattern",
            "buckets": [
                ("PERSISTENT",    "all elevated, same sign",       "durable directional regime â€” trend's friend"),
                ("ACCELERATING",  "|z| grows as window shortens",  "momentum building â€” trend continuation"),
                ("DECELERATING",  "|z| shrinks as window shortens", "momentum fading â€” reversal candidate"),
                ("FRESH",         "only shortest window elevated", "event-driven â€” needs longer-window confirmation"),
                ("DRIFTED",       "long-window elevated, 5d quiet", "coiling near old extreme"),
                ("REVERTING",     "5d & 90d opposite sides",        "caught in the turn â€” classic fade entry"),
                ("STABLE",        "all |z| < 1",                     "no signal"),
                ("MIXED",         "no clean shape",                  "neutral catch-all"),
            ],
        },
        "Overall TAKE": {
            "what":  "Plain-English action recommendation combining all the above for the contract.",
            "formula": "z + Hurst + half-life + ADF â†’ decision tree; see code",
            "buckets": [
                ("FADE â€” high-quality",  "stationary + reverting + |z|â‰¥2", "best fade setups"),
                ("RIDE TREND â€” don't fade", "trending + non-stationary + |z|â‰¥2", "ride, do not fade"),
                ("early fade",            "1 â‰¤ |z| < 2 + reverting",        "anticipate reversion"),
                ("early trend",           "1 â‰¤ |z| < 2 + trending",         "early trend signal"),
                ("elevated",              "1 â‰¤ |z| < 2",                     "monitor"),
                ("stretched (context-dep)", "|z|â‰¥2 mixed signals",          "decide based on context"),
                ("neutral",               "|z| < 1",                         "no edge"),
            ],
        },
    }


# =============================================================================
# Per-contract Z-score & percentile rank (multi-lookback) â€” already exists in
# lib.sra_data; here we provide a per-symbol helper that mirrors the convention
# but is callable on a single column.
# =============================================================================
def zscore_value(series: pd.Series, asof_date: date, lookback: int) -> Optional[float]:
    """Today's Z-score vs the prior ``lookback`` days (excluding today)."""
    if series is None or series.empty:
        return None
    ts = pd.Timestamp(asof_date)
    history = series.loc[series.index < ts].tail(lookback).dropna()
    if len(history) < 2:
        return None
    try:
        today = float(series.loc[series.index.date == asof_date].iloc[0])
    except Exception:
        return None
    if pd.isna(today):
        return None
    mu = float(history.mean())
    sd = float(history.std(ddof=1))
    if sd == 0 or not np.isfinite(sd):
        return None
    return (today - mu) / sd


def percentile_rank_value(series: pd.Series, asof_date: date,
                           lookback: int) -> Optional[float]:
    """Empirical-CDF rank (0..100) of today vs prior ``lookback`` days."""
    if series is None or series.empty:
        return None
    ts = pd.Timestamp(asof_date)
    history = series.loc[series.index < ts].tail(lookback).dropna()
    if history.empty:
        return None
    try:
        today = float(series.loc[series.index.date == asof_date].iloc[0])
    except Exception:
        return None
    return float((history < today).sum() / len(history) * 100.0)


def mean_value(series: pd.Series, asof_date: date, lookback: int) -> Optional[float]:
    ts = pd.Timestamp(asof_date)
    history = series.loc[series.index < ts].tail(lookback).dropna()
    if history.empty:
        return None
    return float(history.mean())


def std_value(series: pd.Series, asof_date: date, lookback: int) -> Optional[float]:
    ts = pd.Timestamp(asof_date)
    history = series.loc[series.index < ts].tail(lookback).dropna()
    if len(history) < 2:
        return None
    return float(history.std(ddof=1))


# =============================================================================
# OU half-life â€” fit ``Î”x_t = -Î» (x_{t-1} - Î¼) + Îµ``  â†’  half_life = ln(2) / Î»
#
# We use OLS on the AR(1) form  x_t = a + bÂ·x_{t-1} + Îµ  where b = (1 - Î»).
# half_life = -ln(2) / ln(b)  when 0 < b < 1.   b â‰¥ 1 â†’ not mean-reverting â†’ None.
# =============================================================================
def ou_half_life(series: pd.Series, asof_date: date,
                 lookback: int = 90) -> Optional[float]:
    """OU half-life in trading days, computed on the prior ``lookback`` bars.

    Returns None if not mean-reverting in this window or insufficient data.
    """
    if series is None or series.empty:
        return None
    ts = pd.Timestamp(asof_date)
    history = series.loc[series.index < ts].tail(lookback).dropna()
    if len(history) < 20:
        return None
    x = history.values
    x_prev = x[:-1]
    x_curr = x[1:]
    if len(x_prev) < 5:
        return None
    # OLS:  x_curr = a + b*x_prev
    A = np.column_stack([np.ones(len(x_prev)), x_prev])
    try:
        coefs, *_ = np.linalg.lstsq(A, x_curr, rcond=None)
    except Exception:
        return None
    a, b = float(coefs[0]), float(coefs[1])
    if not np.isfinite(b) or b <= 0 or b >= 1.0:
        return None
    try:
        hl = -np.log(2.0) / np.log(b)
    except Exception:
        return None
    if not np.isfinite(hl) or hl <= 0:
        return None
    return float(hl)


# =============================================================================
# Hurst exponent â€” Rescaled Range (R/S) analysis on log-returns.
#
# R/S(n) ~ c Â· n^H. Estimate H by linear regression of log(R/S) on log(n)
# across a small set of sub-window sizes.
#
#   H â‰ˆ 0.5  â†’ random walk (no memory)
#   H > 0.5  â†’ trending / persistent
#   H < 0.5  â†’ mean-reverting / anti-persistent
# =============================================================================
def hurst_exponent(series: pd.Series, asof_date: date,
                   lookback: int = 90) -> Optional[float]:
    """Hurst exponent via R/S, computed on the prior ``lookback`` bars."""
    if series is None or series.empty:
        return None
    ts = pd.Timestamp(asof_date)
    history = series.loc[series.index < ts].tail(lookback).dropna()
    if len(history) < 30:
        return None
    rets = np.diff(history.values)
    if len(rets) < 20:
        return None
    n_total = len(rets)
    # Pick a sensible set of sub-window sizes between ~8 and n/2 (log-spaced)
    max_n = max(8, n_total // 2)
    if max_n < 8:
        return None
    log_n = np.linspace(np.log(8), np.log(max_n), num=8)
    sizes = sorted(set(int(round(np.exp(v))) for v in log_n))
    sizes = [s for s in sizes if 8 <= s <= n_total]
    if len(sizes) < 4:
        return None
    rs_values = []
    valid_sizes = []
    for n in sizes:
        # Split rets into non-overlapping chunks of size n
        n_chunks = n_total // n
        if n_chunks < 1:
            continue
        rs_list = []
        for k in range(n_chunks):
            chunk = rets[k * n:(k + 1) * n]
            mean = chunk.mean()
            dev = chunk - mean
            cum = np.cumsum(dev)
            R = cum.max() - cum.min()
            S = chunk.std(ddof=0)
            if S > 0 and np.isfinite(R) and np.isfinite(S):
                rs_list.append(R / S)
        if rs_list:
            rs_values.append(np.mean(rs_list))
            valid_sizes.append(n)
    if len(rs_values) < 4:
        return None
    log_sizes = np.log(np.array(valid_sizes))
    log_rs = np.log(np.array(rs_values))
    if not np.isfinite(log_rs).all():
        return None
    try:
        slope, _ = np.polyfit(log_sizes, log_rs, 1)
    except Exception:
        return None
    if not np.isfinite(slope):
        return None
    # Clip to a sensible range
    return float(max(0.0, min(1.0, slope)))


def hurst_label(h: Optional[float]) -> str:
    if h is None or pd.isna(h):
        return "â€”"
    if h >= 0.55:
        return "TRENDING"
    if h <= 0.45:
        return "REVERTING"
    return "RANDOM"


# =============================================================================
# Augmented Dickey-Fuller â€” lag-1 ADF without trend.
#
# Test  Î”x_t = Î± + Î³Â·x_{t-1} + Î£ Ï†_iÂ·Î”x_{t-i} + Îµ  with Hâ‚€: Î³ = 0 (unit root).
# Approximated p-value via MacKinnon's surface critical values
# (constant-term, no trend) â€” see Hamilton (1994), Table B.6.
#
# We compute the t-stat for Î³. Then map to a coarse p-value by the standard
# critical values  (1%: -3.43, 5%: -2.86, 10%: -2.57). For dashboard purposes a
# coarse p-bucket is sufficient â€” we expose both the t-stat and the bucket.
# =============================================================================
def adf_test(series: pd.Series, asof_date: date,
             lookback: int = 90, n_lags: int = 1) -> dict:
    """Lag-N ADF (no trend, with constant). Returns ``{tstat, pvalue, reject_5pct}``.

    ``pvalue`` is a coarse bucket from MacKinnon critical values (None if can't
    compute). ``reject_5pct`` = True iff t-stat â‰¤ -2.86.
    """
    out = {"tstat": None, "pvalue": None, "reject_5pct": False, "n_obs": 0}
    if series is None or series.empty:
        return out
    ts = pd.Timestamp(asof_date)
    history = series.loc[series.index < ts].tail(lookback).dropna()
    if len(history) < max(20, 4 * (n_lags + 1)):
        return out
    x = history.values.astype(float)
    dx = np.diff(x)
    n_lags = max(0, int(n_lags))
    # Regression:  dx_t = Î± + Î³Â·x_{t-1} + Î£ Ï†_iÂ·dx_{t-i} + Îµ
    # Build aligned arrays (drop the first n_lags rows of dx)
    if n_lags >= len(dx):
        return out
    y = dx[n_lags:]
    n = len(y)
    if n < 10:
        return out
    cols = [np.ones(n), x[n_lags:-1]]   # constant + x_{t-1}
    for i in range(1, n_lags + 1):
        cols.append(dx[n_lags - i:-i if i > 0 else None])
    # Trim/align: ensure each col has length n
    cols = [c[-n:] for c in cols]
    X = np.column_stack(cols)
    if X.shape[1] >= X.shape[0]:
        return out
    try:
        coefs, residuals, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    except Exception:
        return out
    if X.shape[0] - X.shape[1] <= 0:
        return out
    yhat = X @ coefs
    resid = y - yhat
    sigma2 = float((resid ** 2).sum() / (n - X.shape[1]))
    if sigma2 <= 0 or not np.isfinite(sigma2):
        return out
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return out
    se_gamma = float(np.sqrt(sigma2 * XtX_inv[1, 1]))
    if se_gamma <= 0 or not np.isfinite(se_gamma):
        return out
    gamma_hat = float(coefs[1])
    tstat = gamma_hat / se_gamma
    out["tstat"] = float(tstat)
    out["n_obs"] = int(n)
    out["reject_5pct"] = bool(tstat <= -2.86)
    # Coarse p-bucket
    if tstat <= -3.43:
        out["pvalue"] = "<0.01"
    elif tstat <= -2.86:
        out["pvalue"] = "<0.05"
    elif tstat <= -2.57:
        out["pvalue"] = "<0.10"
    else:
        out["pvalue"] = ">0.10"
    return out


# =============================================================================
# Composite reversion-candidate score
# =============================================================================
def composite_reversion_score(z: Optional[float], half_life: Optional[float],
                               hurst: Optional[float],
                               adf_reject: bool) -> Optional[float]:
    """Heuristic composite score for ranking mean-reversion candidates.

    score = 0.5 Â· |z|/2          (capped 1.0)
          + 0.25 Â· (1 - hurst)/0.5
          + 0.15 Â· 1 / max(1, half_life/10)
          + 0.10 Â· {1 if ADF rejects 5% else 0}
    Each term in [0, 1]; total âˆˆ [0, 1].
    """
    if z is None or not np.isfinite(z):
        return None
    z_term = min(1.0, abs(z) / 2.0)
    if hurst is None or not np.isfinite(hurst):
        h_term = 0.5 / 0.5    # neutral
    else:
        h_term = max(0.0, min(1.0, (0.5 - min(0.5, hurst)) / 0.5 + 0.5 - 0.5))
        # Simpler & more linear:
        h_term = max(0.0, min(1.0, (1.0 - hurst) / 1.0))
    if half_life is None or not np.isfinite(half_life):
        hl_term = 0.0
    else:
        hl_term = max(0.0, min(1.0, 10.0 / max(1.0, half_life)))
    adf_term = 1.0 if adf_reject else 0.0
    return float(0.5 * z_term + 0.25 * h_term + 0.15 * hl_term + 0.10 * adf_term)


def composite_trend_score(z: Optional[float], hurst: Optional[float],
                           adf_reject: bool) -> Optional[float]:
    """Inverse â€” high |z| + high Hurst + ADF cannot reject = trend candidate."""
    if z is None or not np.isfinite(z):
        return None
    z_term = min(1.0, abs(z) / 2.0)
    if hurst is None or not np.isfinite(hurst):
        h_term = 0.5
    else:
        h_term = max(0.0, min(1.0, hurst))
    adf_term = 0.0 if adf_reject else 1.0    # inverted
    return float(0.55 * z_term + 0.35 * h_term + 0.10 * adf_term)


# =============================================================================
# Engine â€” compute everything for the full universe in one go
# =============================================================================
def compute_zscore_panel(wide_close: pd.DataFrame, contracts: list[str],
                         asof_date: date, lookbacks: list[int],
                         base_product: str = "SRA",
                         long_lookback_for_tests: int = 90) -> dict:
    """Build a complete Z-score / mean-reversion panel for the universe.

    Output structure::

        {
          "asof": date,
          "lookbacks": [5, 15, 30, 60, 90],
          "test_lookback": 90,
          "n_contracts": int,
          "per_contract": {
            symbol: {
              "convention": "bp"|"price",
              "bp_multiplier": float,
              "current": float, "current_bp": float,
              "by_lookback": {n: {z, mean, std, dist_to_mean_bp,
                                   pct_rank, velocity_to_mean}},
              "pattern": str,
              "ou_half_life": float|None,
              "hurst": float|None,
              "hurst_label": str,
              "adf_tstat": float|None,
              "adf_pvalue": str|None,
              "adf_reject_5pct": bool,
              "reversion_score": float|None,
              "trend_score": float|None,
            }
          }
        }
    """
    out = {
        "asof": asof_date,
        "lookbacks": list(lookbacks),
        "test_lookback": long_lookback_for_tests,
        "n_contracts": len(contracts),
        "per_contract": {},
    }
    if wide_close is None or wide_close.empty or not contracts:
        return out

    cat = load_catalog()
    mults = bp_multipliers_for(contracts, base_product, cat) if not cat.empty \
            else [100.0] * len(contracts)
    sym_mult = dict(zip(contracts, mults))
    sym_conv = {c: ("bp" if abs(sym_mult[c] - 1.0) < 1e-6 else "price")
                for c in contracts}

    for sym in contracts:
        if sym not in wide_close.columns:
            continue
        series = wide_close[sym]
        bp_mult = sym_mult.get(sym, 100.0)
        try:
            current = float(series.loc[series.index.date == asof_date].iloc[0])
        except Exception:
            current = None
        rec = {
            "convention": sym_conv.get(sym, "unknown"),
            "bp_multiplier": float(bp_mult),
            "current": current,
            "current_bp": (current * bp_mult) if (current is not None and not pd.isna(current)) else None,
            "by_lookback": {},
        }
        for n in lookbacks:
            z = zscore_value(series, asof_date, n)
            mu = mean_value(series, asof_date, n)
            sd = std_value(series, asof_date, n)
            pct = percentile_rank_value(series, asof_date, n)
            dtm_bp = None
            if current is not None and mu is not None:
                dtm_bp = (current - mu) * bp_mult
            # Velocity to mean: avg per-day Î”|z| over the last 5 bars (negative = approaching mean)
            vel = None
            try:
                ts = pd.Timestamp(asof_date)
                hist = series.loc[series.index < ts].dropna()
                if len(hist) >= 7:
                    recent = hist.tail(6)
                    z_seq = []
                    for i in range(1, len(recent) + 1):
                        sub = hist.iloc[:hist.index.get_loc(recent.index[i - 1]) + 1]
                        if len(sub) < n + 1:
                            continue
                        win = sub.tail(n + 1).iloc[:-1]
                        if len(win) < 2 or win.std(ddof=1) == 0:
                            continue
                        z_seq.append(abs((float(recent.iloc[i - 1]) - float(win.mean())) / float(win.std(ddof=1))))
                    if len(z_seq) >= 2:
                        vel = float(np.mean(np.diff(z_seq)))
            except Exception:
                vel = None
            rec["by_lookback"][n] = {
                "z": z, "mean": mu, "std": sd, "pct_rank": pct,
                "dist_to_mean_bp": dtm_bp,
                "velocity_to_mean": vel,
            }
        # Tests (use long lookback so we have enough sample)
        rec["ou_half_life"] = ou_half_life(series, asof_date, long_lookback_for_tests)
        rec["hurst"] = hurst_exponent(series, asof_date, long_lookback_for_tests)
        rec["hurst_label"] = hurst_label(rec["hurst"])
        adf = adf_test(series, asof_date, long_lookback_for_tests, n_lags=1)
        rec["adf_tstat"] = adf.get("tstat")
        rec["adf_pvalue"] = adf.get("pvalue")
        rec["adf_reject_5pct"] = bool(adf.get("reject_5pct", False))
        # Pattern (8-pattern z-classifier â€” same as proximity but on Z values)
        rec["pattern"] = classify_z_pattern(rec["by_lookback"], lookbacks)
        # Composite scores using the longest selected lookback's z
        z_for_score = rec["by_lookback"].get(max(lookbacks), {}).get("z")
        rec["reversion_score"] = composite_reversion_score(
            z_for_score, rec["ou_half_life"], rec["hurst"], rec["adf_reject_5pct"])
        rec["trend_score"] = composite_trend_score(
            z_for_score, rec["hurst"], rec["adf_reject_5pct"])
        out["per_contract"][sym] = rec
    return out


# =============================================================================
# Z-pattern classifier across multiple lookbacks
#
# Mirrors the 8-pattern catalogue locked in lib.sra_curve._block_z_multi_lookback,
# adapted here to operate on a single-contract dict-of-z-values keyed by lookback.
# =============================================================================
Z_ELEVATED = 1.0
Z_FRESH = 1.5


def _z_at(by_lookback: dict, n: int) -> Optional[float]:
    m = by_lookback.get(n) or {}
    return m.get("z")


def classify_z_pattern(by_lookback: dict, lookbacks: list[int]) -> str:
    """Return one of 8 named patterns describing the multi-lookback Z structure.

    STABLE       â€” all |z| < 1
    FRESH        â€” |z[shortest]| â‰¥ 1.5; longer windows |z|<1
    DRIFTED      â€” long-window |z| â‰¥ 1.5, shortest |z|<1
    REVERTING    â€” z[shortest] and z[longest] opposite signs, both |z|â‰¥1
    PERSISTENT   â€” all |z|â‰¥1, same sign, similar magnitudes
    ACCELERATING â€” |z| monotonically increases as window shortens, same sign, all elevated
    DECELERATING â€” |z| monotonically decreases as window shortens, same sign, all elevated
    MIXED        â€” none of the above
    """
    if not lookbacks:
        return "MIXED"
    ordered = sorted(lookbacks)
    zs = [_z_at(by_lookback, n) for n in ordered]
    if any(z is None or pd.isna(z) for z in zs):
        valid = [z for z in zs if z is not None and not pd.isna(z)]
        if not valid:
            return "MIXED"
        if all(abs(z) < Z_ELEVATED for z in valid):
            return "STABLE"
        return "MIXED"
    abs_zs = [abs(z) for z in zs]
    signs = [int(np.sign(z)) for z in zs]

    if all(a < Z_ELEVATED for a in abs_zs):
        return "STABLE"

    # FRESH â€” only shortest is fresh-magnitude elevated; longer windows quiet
    if abs_zs[0] >= Z_FRESH and all(a < Z_ELEVATED for a in abs_zs[1:]):
        return "FRESH"

    # DRIFTED â€” long-window elevated, shortest quiet
    if abs_zs[0] < Z_ELEVATED and all(a >= Z_FRESH for a in abs_zs[-2:]):
        # Same sign across the longer windows
        long_signs = [s for s, a in zip(signs[-2:], abs_zs[-2:]) if a >= Z_FRESH]
        if long_signs and len(set(long_signs)) == 1:
            return "DRIFTED"

    # REVERTING â€” shortest and longest both elevated, opposite signs
    if (abs_zs[0] >= Z_ELEVATED and abs_zs[-1] >= Z_ELEVATED
            and signs[0] * signs[-1] < 0):
        return "REVERTING"

    # PERSISTENT / ACCELERATING / DECELERATING â€” all elevated, same sign
    if all(a >= Z_ELEVATED for a in abs_zs) and len(set(signs)) == 1:
        # Sort by lookback ascending; |z| monotonic?
        # ordered = ascending lookback; abs_zs[0] is shortest
        mono_dec_short_to_long = all(abs_zs[i] >= abs_zs[i + 1] - 1e-9
                                      for i in range(len(abs_zs) - 1))
        mono_inc_short_to_long = all(abs_zs[i] <= abs_zs[i + 1] + 1e-9
                                      for i in range(len(abs_zs) - 1))
        # ACCELERATING: shortest has the largest |z|
        if mono_dec_short_to_long and not mono_inc_short_to_long:
            return "ACCELERATING"
        if mono_inc_short_to_long and not mono_dec_short_to_long:
            return "DECELERATING"
        # Otherwise â€” magnitudes similar across windows
        spread = max(abs_zs) - min(abs_zs)
        if spread < 0.75:
            return "PERSISTENT"
        return "PERSISTENT"
    return "MIXED"


def get_z_pattern_descriptions() -> dict:
    return {
        "PERSISTENT":   "All selected windows |z|â‰¥1, same sign â€” durable directional regime; trend is your friend, fade only with confluence.",
        "FRESH":        "Only the shortest window |z|â‰¥1.5; longer windows quiet â€” event-driven move that has yet to be confirmed by longer horizons.",
        "DRIFTED":      "Long-horizon windows |z|â‰¥1.5 but the shortest is quiet â€” coiling pattern; watch for breakout direction.",
        "REVERTING":    "Shortest and longest windows on opposite sides of the mean â€” caught in the turn; classic mean-reversion entry zone.",
        "ACCELERATING": "|z| monotonically grows as window shortens, same sign across all windows â€” momentum building; trend continuation likely.",
        "DECELERATING": "|z| monotonically shrinks as window shortens, same sign â€” momentum fading; reversal/exhaustion candidate.",
        "STABLE":       "All windows |z|<1 â€” no Z-stretch in any window; no signal.",
        "MIXED":        "Pattern doesn't fit any clean shape â€” neutral catch-all.",
    }


# =============================================================================
# Ranking helpers â€” feeds the section ribbons
# =============================================================================
def rank_by_z(panel: dict, contracts: list[str], lookback: int,
              top_k: int = 5, side: str = "high") -> list[dict]:
    """Return top-K contracts by Z-score on ``side`` ('high' = most positive,
    'low' = most negative).
    """
    rows = []
    side = side.lower()
    for sym in contracts:
        rec = panel.get("per_contract", {}).get(sym)
        if not rec:
            continue
        m = rec.get("by_lookback", {}).get(lookback) or {}
        z = m.get("z")
        if z is None or pd.isna(z):
            continue
        rows.append({
            "symbol": sym,
            "convention": rec.get("convention"),
            "bp_multiplier": rec.get("bp_multiplier"),
            "current_bp": rec.get("current_bp"),
            "z": z,
            "mean": m.get("mean"),
            "std": m.get("std"),
            "pct_rank": m.get("pct_rank"),
            "dist_to_mean_bp": m.get("dist_to_mean_bp"),
            "velocity_to_mean": m.get("velocity_to_mean"),
            "ou_half_life": rec.get("ou_half_life"),
            "hurst": rec.get("hurst"),
            "hurst_label": rec.get("hurst_label"),
            "adf_tstat": rec.get("adf_tstat"),
            "adf_pvalue": rec.get("adf_pvalue"),
            "adf_reject_5pct": rec.get("adf_reject_5pct"),
            "pattern": rec.get("pattern"),
            "reversion_score": rec.get("reversion_score"),
            "trend_score": rec.get("trend_score"),
        })
    if side == "high":
        rows.sort(key=lambda r: -r["z"])
    else:
        rows.sort(key=lambda r: r["z"])
    return rows[:max(0, int(top_k))]


def rank_by_score(panel: dict, contracts: list[str], score_field: str,
                   top_k: int = 5) -> list[dict]:
    """Top-K by ``reversion_score`` or ``trend_score`` (descending)."""
    rows = []
    for sym in contracts:
        rec = panel.get("per_contract", {}).get(sym) or {}
        s = rec.get(score_field)
        if s is None or pd.isna(s):
            continue
        # Use the longest selected lookback's z for display
        lbs = list(rec.get("by_lookback", {}).keys())
        long_lb = max(lbs) if lbs else None
        z = (rec.get("by_lookback", {}).get(long_lb, {}) or {}).get("z") if long_lb else None
        rows.append({
            "symbol": sym, "score": s, "z": z,
            "ou_half_life": rec.get("ou_half_life"),
            "hurst": rec.get("hurst"), "hurst_label": rec.get("hurst_label"),
            "adf_tstat": rec.get("adf_tstat"), "adf_pvalue": rec.get("adf_pvalue"),
            "adf_reject_5pct": rec.get("adf_reject_5pct"),
            "pattern": rec.get("pattern"),
            "current_bp": rec.get("current_bp"),
        })
    rows.sort(key=lambda r: -r["score"])
    return rows[:max(0, int(top_k))]


# =============================================================================
# Cluster / section regime via Z
# =============================================================================
def cluster_signal_z(panel: dict, contracts: list[str],
                     group_for: dict[str, str], lookback: int) -> dict:
    """Per-group {n_total, n_pos_extreme, n_neg_extreme, mean_z, median_z}.
    'pos extreme' = z â‰¥ 2; 'neg extreme' = z â‰¤ -2.
    """
    buckets: dict[str, dict] = {}
    for sym in contracts:
        label = group_for.get(sym)
        if label is None:
            continue
        rec = panel.get("per_contract", {}).get(sym) or {}
        m = rec.get("by_lookback", {}).get(lookback) or {}
        z = m.get("z")
        b = buckets.setdefault(label, {
            "n_total": 0, "n_pos_extreme": 0, "n_neg_extreme": 0,
            "zs": [], "members": [],
        })
        b["n_total"] += 1
        b["members"].append(sym)
        if z is not None and not pd.isna(z):
            b["zs"].append(z)
            if z >= 2:
                b["n_pos_extreme"] += 1
            elif z <= -2:
                b["n_neg_extreme"] += 1
    out: dict = {}
    for label, b in buckets.items():
        zs = b["zs"]
        out[label] = {
            "n_total": b["n_total"],
            "n_pos_extreme": b["n_pos_extreme"],
            "n_neg_extreme": b["n_neg_extreme"],
            "mean_z": float(np.mean(zs)) if zs else None,
            "median_z": float(np.median(zs)) if zs else None,
            "members": b["members"],
        }
    return out


def section_regime_z(panel: dict, contracts: list[str], front_end: int,
                      mid_end: int, lookback: int) -> dict:
    """Section-level z summary:  avg/median z, # |z|â‰¥2, label."""
    n = len(contracts)
    if n == 0:
        return {"sections": {}, "label": "â€”"}
    fe = min(front_end, n)
    me = min(mid_end, n)
    if me <= fe:
        me = min(fe + 1, n)
    sec_ranges = {"front": range(0, fe), "mid": range(fe, me), "back": range(me, n)}
    sections: dict[str, dict] = {}
    for name, rng in sec_ranges.items():
        zs, n_extreme, n_total = [], 0, 0
        for i in rng:
            sym = contracts[i]
            rec = panel.get("per_contract", {}).get(sym) or {}
            m = rec.get("by_lookback", {}).get(lookback) or {}
            n_total += 1
            z = m.get("z")
            if z is not None and not pd.isna(z):
                zs.append(z)
                if abs(z) >= 2:
                    n_extreme += 1
        sections[name] = {
            "n_total": n_total, "n_extreme": n_extreme,
            "mean_z": float(np.mean(zs)) if zs else None,
            "median_z": float(np.median(zs)) if zs else None,
        }
    fr = sections.get("front", {}).get("median_z")
    md = sections.get("mid", {}).get("median_z")
    bk = sections.get("back", {}).get("median_z")
    label = "MIXED"
    if fr is not None and md is not None and bk is not None:
        all_pos = all(v >= 1.0 for v in (fr, md, bk))
        all_neg = all(v <= -1.0 for v in (fr, md, bk))
        steepening = bk - fr >= 1.0
        flattening = fr - bk >= 1.0
        belly_bid = md - 0.5 * (fr + bk) >= 1.0
        belly_offered = 0.5 * (fr + bk) - md >= 1.0
        if all_pos:
            label = "RICH (all sections > +1Ïƒ)"
        elif all_neg:
            label = "CHEAP (all sections < âˆ’1Ïƒ)"
        elif steepening:
            label = "STEEPENING (back rich vs front)"
        elif flattening:
            label = "FLATTENING (front rich vs back)"
        elif belly_bid:
            label = "BELLY-BID (mid rich vs wings)"
        elif belly_offered:
            label = "BELLY-OFFERED (mid cheap vs wings)"
    return {"sections": sections, "label": label}


# ====================================================================
# PCA-engine additions: KPSS + Variance-Ratio + triple-stationarity gate
# ====================================================================

# =============================================================================
# KPSS test â€” Kwiatkowski-Phillips-Schmidt-Shin level-stationarity (1992)
#
# Null: series IS level-stationary. Reject (test stat > critical) means series
# has a unit root or drift. Hand-rolled with Bartlett kernel.
#
# Critical values (Kwiatkowski 1992, Table 1, level-stationarity):
#   10%: 0.347   5%: 0.463   2.5%: 0.574   1%: 0.739
# =============================================================================
def kpss_test(series: pd.Series, asof_date: date,
                lookback: int = 90, lag: int = 10) -> dict:
    """KPSS test for level-stationarity.

    Returns ``{tstat, reject_5pct, n_obs}``. ``reject_5pct=True`` means the
    series is NOT level-stationary (drift / unit root present).
    """
    out = {"tstat": None, "reject_5pct": None, "n_obs": 0}
    if series is None or series.empty:
        return out
    ts = pd.Timestamp(asof_date)
    history = series.loc[series.index < ts].tail(lookback).dropna()
    n = len(history)
    if n < 20:
        return out
    x = history.values.astype(float)
    mu = x.mean()
    e = x - mu
    s = np.cumsum(e)
    s2 = float((s ** 2).sum() / (n ** 2))
    # Long-run variance ÏƒÌ‚Â²(l) using Bartlett kernel
    gamma0 = float((e ** 2).sum() / n)
    lr_var = gamma0
    for h in range(1, min(lag, n) + 1):
        gamma_h = float((e[h:] * e[:-h]).sum() / n)
        w = 1.0 - h / (lag + 1.0)
        lr_var += 2.0 * w * gamma_h
    if lr_var <= 0 or not np.isfinite(lr_var):
        return out
    tstat = s2 / lr_var
    out["tstat"] = float(tstat)
    out["reject_5pct"] = bool(tstat > 0.463)
    out["n_obs"] = int(n)
    return out


# =============================================================================
# Variance Ratio test â€” Lo & MacKinlay (1988)
#
# VR(q) = Var[r_q] / (q Â· Var[r_1]). Random walk â†’ VR(q) â‰ˆ 1.
# VR < 1 â†’ mean reverting; VR > 1 â†’ trending.
#
# Homoscedastic asymptotic z-statistic under RW null:
#   z = (VR âˆ’ 1) / sqrt(2(2qâˆ’1)(qâˆ’1)/(3qN))
# =============================================================================
def variance_ratio_test(series: pd.Series, asof_date: date,
                          lookback: int = 90, q: int = 4) -> dict:
    """Lo-MacKinlay variance ratio test at horizon ``q``.

    Returns ``{vr, z_stat, reject_5pct, n_obs}``. ``reject_5pct=True`` means
    the series departs from random walk at 5%.
    """
    out = {"vr": None, "z_stat": None, "reject_5pct": None, "n_obs": 0}
    if series is None or series.empty:
        return out
    ts = pd.Timestamp(asof_date)
    history = series.loc[series.index < ts].tail(lookback).dropna()
    n = len(history)
    if n < max(40, 2 * q + 2):
        return out
    x = history.values.astype(float)
    rets = np.diff(x)
    if len(rets) < q + 5:
        return out
    var_1 = float(rets.var(ddof=1))
    if var_1 <= 0 or not np.isfinite(var_1):
        return out
    q_rets = np.array([rets[i:i + q].sum() for i in range(len(rets) - q + 1)])
    var_q = float(q_rets.var(ddof=1))
    vr = var_q / (q * var_1)
    out["vr"] = float(vr)
    nq = len(rets)
    if nq <= q:
        return out
    var_vr = (2.0 * (2.0 * q - 1.0) * (q - 1.0)) / (3.0 * q * nq)
    if var_vr <= 0 or not np.isfinite(var_vr):
        return out
    z = (vr - 1.0) / np.sqrt(var_vr)
    out["z_stat"] = float(z)
    out["reject_5pct"] = bool(abs(z) > 1.96)
    out["n_obs"] = int(nq)
    return out


# =============================================================================
# Triple stationarity gate â€” gameplan Â§A6 emission requirement
#
# A trade emits "clean" only if all three gates pass:
#   Â· ADF rejects unit root @ 5% (series is stationary, NOT random walk)
#   Â· KPSS does NOT reject level-stationarity @ 5% (no drift component)
#   Â· Variance Ratio shows |VR(q=4) âˆ’ 1| > some threshold (mean-reverting OR
#     trending â€” not random walk; directionality determined by VR sign)
#
# When all three pass, the OU half-life is meaningful and the mean-reversion
# trade has statistical support. When ADF passes but KPSS rejects, a drift
# component is present â†’ don't size up. When VR â‰ˆ 1, no edge.
# =============================================================================
def triple_stationarity_gate(series: pd.Series, asof_date: date,
                                lookback: int = 90,
                                vr_q: int = 4,
                                vr_threshold: float = 0.05) -> dict:
    """Run the three gates and return a verdict.

    Returns:
      {
        "adf_pass": bool, "kpss_pass": bool, "vr_pass": bool,
        "all_three": bool,        # True only if all gates pass cleanly
        "drift_present": bool,    # ADF passes but KPSS rejects â†’ drift
        "random_walk": bool,      # VR â‰ˆ 1 â†’ no mean-reversion edge
        "verdict": str,           # one of {clean, drift, random_walk, non_stationary}
        "adf": dict,              # full ADF output
        "kpss": dict,             # full KPSS output
        "vr": dict,               # full VR output
      }

    `verdict` mapping:
      - "clean":          ADF rejects + KPSS passes + |VR-1| > threshold
      - "drift":          ADF rejects + KPSS rejects (drift component)
      - "random_walk":    |VR - 1| â‰¤ threshold (no mean-reversion edge)
      - "non_stationary": ADF fails (unit root present)
    """
    adf = adf_test(series, asof_date, lookback=lookback, n_lags=1)
    kpss = kpss_test(series, asof_date, lookback=lookback, lag=10)
    vr = variance_ratio_test(series, asof_date, lookback=lookback, q=vr_q)

    adf_pass = bool(adf.get("reject_5pct", False))
    kpss_passes_level = (kpss.get("reject_5pct") is False)    # NOT rejecting = level-stationary
    vr_val = vr.get("vr")
    vr_pass = (vr_val is not None and np.isfinite(vr_val)
                and abs(vr_val - 1.0) > float(vr_threshold))

    drift_present = bool(adf_pass and (kpss.get("reject_5pct") is True))
    random_walk = (vr_val is not None and np.isfinite(vr_val)
                    and abs(vr_val - 1.0) <= float(vr_threshold))
    all_three = bool(adf_pass and kpss_passes_level and vr_pass)

    if all_three:
        verdict = "clean"
    elif random_walk:
        verdict = "random_walk"
    elif drift_present:
        verdict = "drift"
    elif not adf_pass:
        verdict = "non_stationary"
    else:
        verdict = "inconclusive"

    return {
        "adf_pass": adf_pass,
        "kpss_pass": kpss_passes_level,
        "vr_pass": vr_pass,
        "all_three": all_three,
        "drift_present": drift_present,
        "random_walk": random_walk,
        "verdict": verdict,
        "adf": adf,
        "kpss": kpss,
        "vr": vr,
    }


# =============================================================================
# Composite reversion-candidate score
# =============================================================================
def composite_reversion_score(z: Optional[float], half_life: Optional[float],
                               hurst: Optional[float],
                               adf_reject: bool) -> Optional[float]:
    """Heuristic composite score for ranking mean-reversion candidates.

    score = 0.5 Â· |z|/2          (capped 1.0)
          + 0.25 Â· (1 - hurst)/0.5
          + 0.15 Â· 1 / max(1, half_life/10)
          + 0.10 Â· {1 if ADF rejects 5% else 0}
    Each term in [0, 1]; total âˆˆ [0, 1].
    """
    if z is None or not np.isfinite(z):
        return None
    z_term = min(1.0, abs(z) / 2.0)
    if hurst is None or not np.isfinite(hurst):
        h_term = 0.5 / 0.5    # neutral
    else:
        h_term = max(0.0, min(1.0, (0.5 - min(0.5, hurst)) / 0.5 + 0.5 - 0.5))
        # Simpler & more linear:
        h_term = max(0.0, min(1.0, (1.0 - hurst) / 1.0))
    if half_life is None or not np.isfinite(half_life):
        hl_term = 0.0
    else:
        hl_term = max(0.0, min(1.0, 10.0 / max(1.0, half_life)))
    adf_term = 1.0 if adf_reject else 0.0
    return float(0.5 * z_term + 0.25 * h_term + 0.15 * hl_term + 0.10 * adf_term)


def composite_trend_score(z: Optional[float], hurst: Optional[float],
                           adf_reject: bool) -> Optional[float]:
    """Inverse â€” high |z| + high Hurst + ADF cannot reject = trend candidate."""
    if z is None or not np.isfinite(z):
        return None
    z_term = min(1.0, abs(z) / 2.0)
    if hurst is None or not np.isfinite(hurst):
        h_term = 0.5
    else:
        h_term = max(0.0, min(1.0, hurst))
    adf_term = 0.0 if adf_reject else 1.0    # inverted
    return float(0.55 * z_term + 0.35 * h_term + 0.10 * adf_term)


# =============================================================================
# Engine â€” compute everything for the full universe in one go
# =============================================================================
def compute_zscore_panel(wide_close: pd.DataFrame, contracts: list[str],
                         asof_date: date, lookbacks: list[int],
                         base_product: str = "SRA",
                         long_lookback_for_tests: int = 90) -> dict:
    """Build a complete Z-score / mean-reversion panel for the universe.

    Output structure::

        {
          "asof": date,
          "lookbacks": [5, 15, 30, 60, 90],
          "test_lookback": 90,
          "n_contracts": int,
          "per_contract": {
            symbol: {
              "convention": "bp"|"price",
              "bp_multiplier": float,
              "current": float, "current_bp": float,
              "by_lookback": {n: {z, mean, std, dist_to_mean_bp,
                                   pct_rank, velocity_to_mean}},
              "pattern": str,
              "ou_half_life": float|None,
              "hurst": float|None,
              "hurst_label": str,
              "adf_tstat": float|None,
              "adf_pvalue": str|None,
              "adf_reject_5pct": bool,
              "reversion_score": float|None,
              "trend_score": float|None,
            }
          }
        }
    """
    out = {
        "asof": asof_date,
        "lookbacks": list(lookbacks),
        "test_lookback": long_lookback_for_tests,
        "n_contracts": len(contracts),
        "per_contract": {},
    }
    if wide_close is None or wide_close.empty or not contracts:
        return out

    cat = load_catalog()
    mults = bp_multipliers_for(contracts, base_product, cat) if not cat.empty \
            else [100.0] * len(contracts)
    sym_mult = dict(zip(contracts, mults))
    sym_conv = {c: ("bp" if abs(sym_mult[c] - 1.0) < 1e-6 else "price")
                for c in contracts}

    for sym in contracts:
        if sym not in wide_close.columns:
            continue
        series = wide_close[sym]
        bp_mult = sym_mult.get(sym, 100.0)
        try:
            current = float(series.loc[series.index.date == asof_date].iloc[0])
        except Exception:
            current = None
        rec = {
            "convention": sym_conv.get(sym, "unknown"),
            "bp_multiplier": float(bp_mult),
            "current": current,
            "current_bp": (current * bp_mult) if (current is not None and not pd.isna(current)) else None,
            "by_lookback": {},
        }
        for n in lookbacks:
            z = zscore_value(series, asof_date, n)
            mu = mean_value(series, asof_date, n)
            sd = std_value(series, asof_date, n)
            pct = percentile_rank_value(series, asof_date, n)
            dtm_bp = None
            if current is not None and mu is not None:
                dtm_bp = (current - mu) * bp_mult
            # Velocity to mean: avg per-day Î”|z| over the last 5 bars (negative = approaching mean)
            vel = None
            try:
                ts = pd.Timestamp(asof_date)
                hist = series.loc[series.index < ts].dropna()
                if len(hist) >= 7:
                    recent = hist.tail(6)
                    z_seq = []
                    for i in range(1, len(recent) + 1):
                        sub = hist.iloc[:hist.index.get_loc(recent.index[i - 1]) + 1]
                        if len(sub) < n + 1:
                            continue
                        win = sub.tail(n + 1).iloc[:-1]
                        if len(win) < 2 or win.std(ddof=1) == 0:
                            continue
                        z_seq.append(abs((float(recent.iloc[i - 1]) - float(win.mean())) / float(win.std(ddof=1))))
                    if len(z_seq) >= 2:
                        vel = float(np.mean(np.diff(z_seq)))
            except Exception:
                vel = None
            rec["by_lookback"][n] = {
                "z": z, "mean": mu, "std": sd, "pct_rank": pct,
                "dist_to_mean_bp": dtm_bp,
                "velocity_to_mean": vel,
            }
        # Tests (use long lookback so we have enough sample)
        rec["ou_half_life"] = ou_half_life(series, asof_date, long_lookback_for_tests)
        rec["hurst"] = hurst_exponent(series, asof_date, long_lookback_for_tests)
        rec["hurst_label"] = hurst_label(rec["hurst"])
        adf = adf_test(series, asof_date, long_lookback_for_tests, n_lags=1)
        rec["adf_tstat"] = adf.get("tstat")
        rec["adf_pvalue"] = adf.get("pvalue")
        rec["adf_reject_5pct"] = bool(adf.get("reject_5pct", False))
        # Pattern (8-pattern z-classifier â€” same as proximity but on Z values)
        rec["pattern"] = classify_z_pattern(rec["by_lookback"], lookbacks)
        # Composite scores using the longest selected lookback's z
        z_for_score = rec["by_lookback"].get(max(lookbacks), {}).get("z")
        rec["reversion_score"] = composite_reversion_score(
            z_for_score, rec["ou_half_life"], rec["hurst"], rec["adf_reject_5pct"])
        rec["trend_score"] = composite_trend_score(
            z_for_score, rec["hurst"], rec["adf_reject_5pct"])
        out["per_contract"][sym] = rec
    return out


# =============================================================================
# Z-pattern classifier across multiple lookbacks
#
# Mirrors the 8-pattern catalogue locked in lib.sra_curve._block_z_multi_lookback,
# adapted here to operate on a single-contract dict-of-z-values keyed by lookback.
# =============================================================================
Z_ELEVATED = 1.0
Z_FRESH = 1.5


def _z_at(by_lookback: dict, n: int) -> Optional[float]:
    m = by_lookback.get(n) or {}
    return m.get("z")


def classify_z_pattern(by_lookback: dict, lookbacks: list[int]) -> str:
    """Return one of 8 named patterns describing the multi-lookback Z structure.

    STABLE       â€” all |z| < 1
    FRESH        â€” |z[shortest]| â‰¥ 1.5; longer windows |z|<1
    DRIFTED      â€” long-window |z| â‰¥ 1.5, shortest |z|<1
    REVERTING    â€” z[shortest] and z[longest] opposite signs, both |z|â‰¥1
    PERSISTENT   â€” all |z|â‰¥1, same sign, similar magnitudes
    ACCELERATING â€” |z| monotonically increases as window shortens, same sign, all elevated
    DECELERATING â€” |z| monotonically decreases as window shortens, same sign, all elevated
    MIXED        â€” none of the above
    """
    if not lookbacks:
        return "MIXED"
    ordered = sorted(lookbacks)
    zs = [_z_at(by_lookback, n) for n in ordered]
    if any(z is None or pd.isna(z) for z in zs):
        valid = [z for z in zs if z is not None and not pd.isna(z)]
        if not valid:
            return "MIXED"
        if all(abs(z) < Z_ELEVATED for z in valid):
            return "STABLE"
        return "MIXED"
    abs_zs = [abs(z) for z in zs]
    signs = [int(np.sign(z)) for z in zs]

    if all(a < Z_ELEVATED for a in abs_zs):
        return "STABLE"

    # FRESH â€” only shortest is fresh-magnitude elevated; longer windows quiet
    if abs_zs[0] >= Z_FRESH and all(a < Z_ELEVATED for a in abs_zs[1:]):
        return "FRESH"

    # DRIFTED â€” long-window elevated, shortest quiet
    if abs_zs[0] < Z_ELEVATED and all(a >= Z_FRESH for a in abs_zs[-2:]):
        # Same sign across the longer windows
        long_signs = [s for s, a in zip(signs[-2:], abs_zs[-2:]) if a >= Z_FRESH]
        if long_signs and len(set(long_signs)) == 1:
            return "DRIFTED"

    # REVERTING â€” shortest and longest both elevated, opposite signs
    if (abs_zs[0] >= Z_ELEVATED and abs_zs[-1] >= Z_ELEVATED
            and signs[0] * signs[-1] < 0):
        return "REVERTING"

    # PERSISTENT / ACCELERATING / DECELERATING â€” all elevated, same sign
    if all(a >= Z_ELEVATED for a in abs_zs) and len(set(signs)) == 1:
        # Sort by lookback ascending; |z| monotonic?
        # ordered = ascending lookback; abs_zs[0] is shortest
        mono_dec_short_to_long = all(abs_zs[i] >= abs_zs[i + 1] - 1e-9
                                      for i in range(len(abs_zs) - 1))
        mono_inc_short_to_long = all(abs_zs[i] <= abs_zs[i + 1] + 1e-9
                                      for i in range(len(abs_zs) - 1))
        # ACCELERATING: shortest has the largest |z|
        if mono_dec_short_to_long and not mono_inc_short_to_long:
            return "ACCELERATING"
        if mono_inc_short_to_long and not mono_dec_short_to_long:
            return "DECELERATING"
        # Otherwise â€” magnitudes similar across windows
        spread = max(abs_zs) - min(abs_zs)
        if spread < 0.75:
            return "PERSISTENT"
        return "PERSISTENT"
    return "MIXED"


def get_z_pattern_descriptions() -> dict:
    return {
        "PERSISTENT":   "All selected windows |z|â‰¥1, same sign â€” durable directional regime; trend is your friend, fade only with confluence.",
        "FRESH":        "Only the shortest window |z|â‰¥1.5; longer windows quiet â€” event-driven move that has yet to be confirmed by longer horizons.",
        "DRIFTED":      "Long-horizon windows |z|â‰¥1.5 but the shortest is quiet â€” coiling pattern; watch for breakout direction.",
        "REVERTING":    "Shortest and longest windows on opposite sides of the mean â€” caught in the turn; classic mean-reversion entry zone.",
        "ACCELERATING": "|z| monotonically grows as window shortens, same sign across all windows â€” momentum building; trend continuation likely.",
        "DECELERATING": "|z| monotonically shrinks as window shortens, same sign â€” momentum fading; reversal/exhaustion candidate.",
        "STABLE":       "All windows |z|<1 â€” no Z-stretch in any window; no signal.",
        "MIXED":        "Pattern doesn't fit any clean shape â€” neutral catch-all.",
    }


# =============================================================================
# Ranking helpers â€” feeds the section ribbons
# =============================================================================
def rank_by_z(panel: dict, contracts: list[str], lookback: int,
              top_k: int = 5, side: str = "high") -> list[dict]:
    """Return top-K contracts by Z-score on ``side`` ('high' = most positive,
    'low' = most negative).
    """
    rows = []
    side = side.lower()
    for sym in contracts:
        rec = panel.get("per_contract", {}).get(sym)
        if not rec:
            continue
        m = rec.get("by_lookback", {}).get(lookback) or {}
        z = m.get("z")
        if z is None or pd.isna(z):
            continue
        rows.append({
            "symbol": sym,
            "convention": rec.get("convention"),
            "bp_multiplier": rec.get("bp_multiplier"),
            "current_bp": rec.get("current_bp"),
            "z": z,
            "mean": m.get("mean"),
            "std": m.get("std"),
            "pct_rank": m.get("pct_rank"),
            "dist_to_mean_bp": m.get("dist_to_mean_bp"),
            "velocity_to_mean": m.get("velocity_to_mean"),
            "ou_half_life": rec.get("ou_half_life"),
            "hurst": rec.get("hurst"),
            "hurst_label": rec.get("hurst_label"),
            "adf_tstat": rec.get("adf_tstat"),
            "adf_pvalue": rec.get("adf_pvalue"),
            "adf_reject_5pct": rec.get("adf_reject_5pct"),
            "pattern": rec.get("pattern"),
            "reversion_score": rec.get("reversion_score"),
            "trend_score": rec.get("trend_score"),
        })
    if side == "high":
        rows.sort(key=lambda r: -r["z"])
    else:
        rows.sort(key=lambda r: r["z"])
    return rows[:max(0, int(top_k))]


def rank_by_score(panel: dict, contracts: list[str], score_field: str,
                   top_k: int = 5) -> list[dict]:
    """Top-K by ``reversion_score`` or ``trend_score`` (descending)."""
    rows = []
    for sym in contracts:
        rec = panel.get("per_contract", {}).get(sym) or {}
        s = rec.get(score_field)
        if s is None or pd.isna(s):
            continue
        # Use the longest selected lookback's z for display
        lbs = list(rec.get("by_lookback", {}).keys())
        long_lb = max(lbs) if lbs else None
        z = (rec.get("by_lookback", {}).get(long_lb, {}) or {}).get("z") if long_lb else None
        rows.append({
            "symbol": sym, "score": s, "z": z,
            "ou_half_life": rec.get("ou_half_life"),
            "hurst": rec.get("hurst"), "hurst_label": rec.get("hurst_label"),
            "adf_tstat": rec.get("adf_tstat"), "adf_pvalue": rec.get("adf_pvalue"),
            "adf_reject_5pct": rec.get("adf_reject_5pct"),
            "pattern": rec.get("pattern"),
            "current_bp": rec.get("current_bp"),
        })
    rows.sort(key=lambda r: -r["score"])
    return rows[:max(0, int(top_k))]


# =============================================================================
# Cluster / section regime via Z
# =============================================================================
def cluster_signal_z(panel: dict, contracts: list[str],
                     group_for: dict[str, str], lookback: int) -> dict:
    """Per-group {n_total, n_pos_extreme, n_neg_extreme, mean_z, median_z}.
    'pos extreme' = z â‰¥ 2; 'neg extreme' = z â‰¤ -2.
    """
    buckets: dict[str, dict] = {}
    for sym in contracts:
        label = group_for.get(sym)
        if label is None:
            continue
        rec = panel.get("per_contract", {}).get(sym) or {}
        m = rec.get("by_lookback", {}).get(lookback) or {}
        z = m.get("z")
        b = buckets.setdefault(label, {
            "n_total": 0, "n_pos_extreme": 0, "n_neg_extreme": 0,
            "zs": [], "members": [],
        })
        b["n_total"] += 1
        b["members"].append(sym)
        if z is not None and not pd.isna(z):
            b["zs"].append(z)
            if z >= 2:
                b["n_pos_extreme"] += 1
            elif z <= -2:
                b["n_neg_extreme"] += 1
    out: dict = {}
    for label, b in buckets.items():
        zs = b["zs"]
        out[label] = {
            "n_total": b["n_total"],
            "n_pos_extreme": b["n_pos_extreme"],
            "n_neg_extreme": b["n_neg_extreme"],
            "mean_z": float(np.mean(zs)) if zs else None,
            "median_z": float(np.median(zs)) if zs else None,
            "members": b["members"],
        }
    return out


def section_regime_z(panel: dict, contracts: list[str], front_end: int,
                      mid_end: int, lookback: int) -> dict:
    """Section-level z summary:  avg/median z, # |z|â‰¥2, label."""
    n = len(contracts)
    if n == 0:
        return {"sections": {}, "label": "â€”"}
    fe = min(front_end, n)
    me = min(mid_end, n)
    if me <= fe:
        me = min(fe + 1, n)
    sec_ranges = {"front": range(0, fe), "mid": range(fe, me), "back": range(me, n)}
    sections: dict[str, dict] = {}
    for name, rng in sec_ranges.items():
        zs, n_extreme, n_total = [], 0, 0
        for i in rng:
            sym = contracts[i]
            rec = panel.get("per_contract", {}).get(sym) or {}
            m = rec.get("by_lookback", {}).get(lookback) or {}
            n_total += 1
            z = m.get("z")
            if z is not None and not pd.isna(z):
                zs.append(z)
                if abs(z) >= 2:
                    n_extreme += 1
        sections[name] = {
            "n_total": n_total, "n_extreme": n_extreme,
            "mean_z": float(np.mean(zs)) if zs else None,
            "median_z": float(np.median(zs)) if zs else None,
        }
    fr = sections.get("front", {}).get("median_z")
    md = sections.get("mid", {}).get("median_z")
    bk = sections.get("back", {}).get("median_z")
    label = "MIXED"
    if fr is not None and md is not None and bk is not None:
        all_pos = all(v >= 1.0 for v in (fr, md, bk))
        all_neg = all(v <= -1.0 for v in (fr, md, bk))
        steepening = bk - fr >= 1.0
        flattening = fr - bk >= 1.0
        belly_bid = md - 0.5 * (fr + bk) >= 1.0
        belly_offered = 0.5 * (fr + bk) - md >= 1.0
        if all_pos:
            label = "RICH (all sections > +1Ïƒ)"
        elif all_neg:
            label = "CHEAP (all sections < âˆ’1Ïƒ)"
        elif steepening:
            label = "STEEPENING (back rich vs front)"
        elif flattening:
            label = "FLATTENING (front rich vs back)"
        elif belly_bid:
            label = "BELLY-BID (mid rich vs wings)"
        elif belly_offered:
            label = "BELLY-OFFERED (mid cheap vs wings)"
    return {"sections": sections, "label": label}