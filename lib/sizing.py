"""Position sizing calculators (Phase 8, plan §3.5).

Three side-by-side methods: Kelly / vol-target / fixed-fractional.
Per §15 D4=No (no book integration), all sizers assume a flat $10k base
notional per trade. The trader picks one in the UI.

API:
    kelly_size(win_rate, avg_win_R, avg_loss_R, account_usd) -> dict
    vol_target_size(target_vol_pct, expected_vol_pct, account_usd) -> dict
    fixed_fractional_size(fraction, stop_R, account_usd) -> dict
    compare_sizing_methods(setup_stats, account_usd=10_000) -> pd.DataFrame
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_ACCOUNT_USD = 10_000.0


@dataclass
class SizerResult:
    method: str
    notional_usd: float
    n_lots: float
    rationale: str
    capped: bool = False
    inputs: Optional[dict] = None


def kelly_size(win_rate: float, avg_win_R: float, avg_loss_R: float,
                  account_usd: float = DEFAULT_ACCOUNT_USD,
                  cap_fraction: float = 0.25) -> SizerResult:
    """Kelly fraction = (p*b - q) / b where b = avg_win_R / avg_loss_R, p=win_rate, q=1-p.
    Capped at ``cap_fraction`` (default 25% — quarter Kelly is the common practitioner cap)."""
    p = float(np.clip(win_rate, 0.0, 1.0))
    if avg_loss_R <= 0 or avg_win_R <= 0:
        return SizerResult("Kelly", 0.0, 0.0,
                                 "Invalid R: need both avg_win_R > 0 and avg_loss_R > 0",
                                 capped=False, inputs={"win_rate": p})
    b = avg_win_R / avg_loss_R
    q = 1 - p
    f = (p * b - q) / b
    capped = False
    if f > cap_fraction:
        f = cap_fraction; capped = True
    if f < 0:
        f = 0.0
    notional = account_usd * f
    return SizerResult(
        method="Kelly",
        notional_usd=float(notional),
        n_lots=float(notional / 25.0),   # SR3 DV01 ≈ $25/lot/bp
        rationale=f"Kelly f = ({p:.2f}·{b:.2f} - {q:.2f})/{b:.2f} = {f:.3f}"
                      + (" (capped at quarter-Kelly)" if capped else ""),
        capped=capped,
        inputs={"win_rate": p, "avg_win_R": avg_win_R, "avg_loss_R": avg_loss_R},
    )


def vol_target_size(target_vol_pct: float, expected_vol_pct: float,
                       account_usd: float = DEFAULT_ACCOUNT_USD) -> SizerResult:
    """Notional = account * (target_vol / expected_vol). Standard CTA scaling."""
    if expected_vol_pct <= 0:
        return SizerResult("Vol-target", 0.0, 0.0,
                                 "expected_vol_pct must be > 0",
                                 inputs={"target_vol_pct": target_vol_pct})
    scale = target_vol_pct / expected_vol_pct
    notional = account_usd * scale
    return SizerResult(
        method="Vol-target",
        notional_usd=float(notional),
        n_lots=float(notional / 25.0),
        rationale=f"scale = {target_vol_pct:.2f}% / {expected_vol_pct:.2f}% = {scale:.3f}",
        inputs={"target_vol_pct": target_vol_pct,
                  "expected_vol_pct": expected_vol_pct},
    )


def fixed_fractional_size(fraction: float, stop_R: float,
                              account_usd: float = DEFAULT_ACCOUNT_USD) -> SizerResult:
    """Risk a fixed fraction of account per trade.
    notional = (account * fraction) / stop_R_in_dollars
    For SR3, stop_R_in_dollars = stop_bp × $25/lot."""
    if stop_R <= 0:
        return SizerResult("Fixed-fractional", 0.0, 0.0,
                                 "stop_R must be > 0", inputs={"fraction": fraction})
    risk_dollars = account_usd * fraction
    dollars_per_unit = stop_R * 25.0   # SR3 DV01
    n_lots = risk_dollars / max(dollars_per_unit, 1e-9)
    notional = n_lots * 25.0 * 100.0   # 100 = price scale-up to notional
    return SizerResult(
        method="Fixed-fractional",
        notional_usd=float(notional),
        n_lots=float(n_lots),
        rationale=f"risk = ${risk_dollars:.0f} / (stop {stop_R} bp × $25/bp) = {n_lots:.2f} lots",
        inputs={"fraction": fraction, "stop_R": stop_R},
    )


def compare_sizing_methods(setup_stats: dict,
                                  account_usd: float = DEFAULT_ACCOUNT_USD,
                                  ) -> pd.DataFrame:
    """Side-by-side comparison of the three methods given a setup's stats.

    setup_stats keys (any subset; missing → method skipped):
        win_rate, avg_win_R, avg_loss_R    (Kelly)
        target_vol_pct, expected_vol_pct   (Vol-target)
        fixed_fraction, stop_R             (Fixed-fractional)
    """
    rows = []
    if all(k in setup_stats for k in ("win_rate", "avg_win_R", "avg_loss_R")):
        r = kelly_size(setup_stats["win_rate"], setup_stats["avg_win_R"],
                          setup_stats["avg_loss_R"], account_usd)
        rows.append(asdict(r))
    if all(k in setup_stats for k in ("target_vol_pct", "expected_vol_pct")):
        r = vol_target_size(setup_stats["target_vol_pct"],
                                  setup_stats["expected_vol_pct"], account_usd)
        rows.append(asdict(r))
    if all(k in setup_stats for k in ("fixed_fraction", "stop_R")):
        r = fixed_fractional_size(setup_stats["fixed_fraction"],
                                          setup_stats["stop_R"], account_usd)
        rows.append(asdict(r))
    return pd.DataFrame(rows)
