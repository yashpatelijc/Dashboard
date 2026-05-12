"""Setup result dataclass + helpers shared by every detector."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import numpy as np
import pandas as pd


# =============================================================================
# Tick-rounding helpers — entry/stop/T1/T2 must display in round-tick units
# =============================================================================
# SR3 minimum tick: 0.005 in price units (back-month) = 0.5 bp
# Front-month override is 0.0025 = 0.25 bp but we use the back-month tick
# uniformly for setup-level price levels (good enough for trader display;
# the trader rounds the front-month tick downstream).
TICK_BP = 0.5     # 0.5 basis points = 0.005 price units (one half-tick on SR3 back-month)


def round_to_tick(price: Optional[float], bp_mult: float = 100.0,
                  tick_bp: float = TICK_BP) -> Optional[float]:
    """Round a stored-units price to the nearest tick.

    For SRA outrights (``bp_mult == 100``):  tick_stored = 0.5 / 100 = 0.005
    For SRA spreads/flies (``bp_mult == 1``): tick_stored = 0.5 / 1   = 0.5

    The result is always quantized to ``tick_bp`` basis points exactly,
    so display matches what the exchange would actually fill.
    """
    if price is None or pd.isna(price) or not np.isfinite(price):
        return None
    if bp_mult is None or bp_mult <= 0:
        return float(price)
    tick_stored = tick_bp / bp_mult
    return float(round(float(price) / tick_stored) * tick_stored)


def fmt_price_for_scope(price: Optional[float], bp_mult: float,
                         signed: bool = False) -> str:
    """Format a tick-rounded price for display.

    Outrights (bp_mult=100, stored in price units 96.x): 4 decimals → '96.3850'
    Spreads/flies (bp_mult=1, stored in bp): 2 decimals → '-2.50bp' or '+1.50bp'
    """
    rounded = round_to_tick(price, bp_mult)
    if rounded is None:
        return "—"
    if bp_mult and bp_mult >= 50:
        return f"{rounded:+.4f}" if signed else f"{rounded:.4f}"
    # spread/fly path — show with bp suffix
    return (f"{rounded:+.2f}bp" if signed or rounded < 0 else f"{rounded:.2f}bp")


# =============================================================================
# Dataclass returned by every detector
# =============================================================================
@dataclass
class SetupResult:
    """Per-(symbol, setup) signal state and trade-level metadata.

    A detector always returns a SetupResult — never raises. On bad data it
    sets ``state = "N/A"`` and ``error`` with a short message.
    """
    setup_id: str
    name: str
    family: str                                 # 'trend' | 'mean_reversion' | 'stir' | 'composite'
    scope: str                                  # 'outright' | 'spread' | 'fly'
    fired_long: bool = False
    fired_short: bool = False
    state: str = "FAR"                          # 'FIRED' | 'NEAR' | 'APPROACHING' | 'FAR' | 'N/A'
    direction: Optional[str] = None             # 'LONG' | 'SHORT' | None
    confidence: float = 0.0                     # 0..1 — softer signal strength
    key_inputs: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)
    distance_to_fire: Optional[float] = None    # 0 if FIRED; otherwise normalized "how far" (smaller = closer)
    eta_bars: Optional[float] = None            # rough "days until fire" given typical daily Δ
    missing_condition: Optional[str] = None     # plain-English description of what's missing
    entry: Optional[float] = None
    stop: Optional[float] = None
    t1: Optional[float] = None
    t2: Optional[float] = None
    lots_at_10k_risk: Optional[int] = None
    # Multi-leg trade breakdown — populated by scan.py / detectors after entry/stop set.
    # Each leg dict: {"role", "symbol", "side", "lots", "ratio", "dv01_per_bp"}.
    # Outright single contract = 1 leg; spread = 2 legs; fly = 3 legs (1:2:1);
    # C9 outright pair = 2 legs (DV01-neutral, equal lots).
    legs: list = field(default_factory=list)
    interpretation: str = ""
    notes: str = ""
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# Helpers shared by every detector
# =============================================================================
def safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Convert to float, return default on failure / NaN / inf."""
    try:
        v = float(value)
        if not np.isfinite(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def state_from_conditions_met(n_met: int, n_total: int) -> str:
    """Map (#conditions met, total) → state label.

    Convention used by all multi-condition setups:
      all met       → FIRED
      n_total - 1   → NEAR (one gate away)
      ≥ ⌈n/2⌉       → APPROACHING (about half met)
      else          → FAR
    """
    if n_total <= 0:
        return "N/A"
    if n_met >= n_total:
        return "FIRED"
    if n_met == n_total - 1:
        return "NEAR"
    if n_met >= (n_total + 1) // 2:
        return "APPROACHING"
    return "FAR"


def fmt_inputs_str(inputs: dict) -> str:
    """Compact 'k=v' string for tooltips."""
    parts = []
    for k, v in inputs.items():
        if v is None:
            parts.append(f"{k}=—")
        elif isinstance(v, bool):
            parts.append(f"{k}={'✓' if v else '✗'}")
        elif isinstance(v, (int, np.integer)):
            parts.append(f"{k}={int(v)}")
        elif isinstance(v, (float, np.floating)):
            parts.append(f"{k}={float(v):.4g}")
        else:
            parts.append(f"{k}={v}")
    return " · ".join(parts)


# =============================================================================
# Sizing helper: $10K risk → integer lots for SRA
# =============================================================================
SRA_DOLLARS_PER_BP_PER_LOT = 25.0     # SR3 DV01 ≈ $25 per bp per contract (gameplan §5)


def lots_at_10k_risk(stop_distance_bp: Optional[float],
                     risk_dollars: float = 10_000.0,
                     dollars_per_bp_per_lot: float = SRA_DOLLARS_PER_BP_PER_LOT,
                     max_lots: int = 1000) -> Optional[int]:
    """Compute integer lots that risk ≈ ``risk_dollars`` per trade given a
    stop distance in basis points.

    Returns None if stop_distance_bp is None / non-positive / non-finite.
    """
    sd = safe_float(stop_distance_bp)
    if sd is None or sd <= 0:
        return None
    lots = risk_dollars / (sd * dollars_per_bp_per_lot)
    if not np.isfinite(lots):
        return None
    return int(max(1, min(max_lots, lots)))


# =============================================================================
# Trade-level helpers
# =============================================================================
def compute_R_levels(entry: Optional[float], stop: Optional[float],
                     direction: str = "LONG",
                     r_multiple_t1: float = 1.0,
                     r_multiple_t2: Optional[float] = 2.0) -> tuple:
    """From (entry, stop) compute T1 and T2 at +R / +2R for LONG (mirror for SHORT).

    Returns (t1, t2). Either may be None if entry/stop missing.
    """
    e = safe_float(entry); s = safe_float(stop)
    if e is None or s is None:
        return (None, None)
    risk = e - s if direction == "LONG" else s - e
    if risk <= 0:
        return (None, None)
    if direction == "LONG":
        t1 = e + r_multiple_t1 * risk
        t2 = e + r_multiple_t2 * risk if r_multiple_t2 is not None else None
    else:
        t1 = e - r_multiple_t1 * risk
        t2 = e - r_multiple_t2 * risk if r_multiple_t2 is not None else None
    return (safe_float(t1), safe_float(t2))


# =============================================================================
# Distance helpers — produce a normalized "distance to fire" used by NEAR ranking
# =============================================================================
def distance_to_threshold_atr(value: Optional[float], threshold: float,
                              direction: str, atr: Optional[float]) -> Optional[float]:
    """Distance from value to threshold expressed in ATR units. Direction
    determines which side the threshold is on.

    For 'gt' direction: distance = (threshold − value) / atr (positive = value below threshold)
    For 'lt' direction: distance = (value − threshold) / atr
    """
    v = safe_float(value); t = safe_float(threshold); a = safe_float(atr)
    if v is None or t is None or a is None or a <= 0:
        return None
    if direction == "gt":
        return (t - v) / a
    if direction == "lt":
        return (v - t) / a
    return None


def normalize_distance(distance_components: list) -> float:
    """Combine multiple distance components (each ≥0, smaller = closer) into
    a single scalar for ranking. Uses the L2 norm.
    """
    arr = [d for d in distance_components if d is not None and np.isfinite(d) and d >= 0]
    if not arr:
        return float("inf")
    return float(np.sqrt(np.sum([d ** 2 for d in arr])))


# =============================================================================
# Multi-leg trade breakdown
# =============================================================================
# Catalog convention (verified empirically on `mde2_contracts_catalog`):
#   spread "SRAH27-M27"      stored close (bp) ≈ (left_close − right_close) × 100
#   fly    "SRAH27-M27-U27"  stored close (bp) ≈ (left − 2·mid + right) × 100
#
# So a LONG signal on the spread/fly aggregate decomposes into legs as:
#   spread LONG  → BUY left + SELL right
#   spread SHORT → SELL left + BUY right
#   fly    LONG  → BUY left + SELL 2·mid + BUY right     (1:2:1 ratio)
#   fly    SHORT → SELL left + BUY 2·mid + SELL right
#
# C9 (curve steepener / flattener) is a slope between two outrights:
#   slope = front_close − back_close (in price units; ×100 for bp)
#   slope LONG  (steepener) → BUY front + SELL back   (DV01-neutral, equal lots)
#   slope SHORT (flattener) → SELL front + BUY back
def _parse_legs_string(legs_str: Optional[str]) -> list:
    """Split the catalog ``legs`` column ('SRAH27,SRAM27,SRAU27' or 'SRAH27,SRAM27'
    or '') into a list of leg symbols. Empty string → []."""
    if legs_str is None or not isinstance(legs_str, str):
        return []
    parts = [p.strip() for p in legs_str.split(",") if p and p.strip()]
    return parts


def build_legs(symbol: str, scope: str, direction: Optional[str],
                lots: Optional[int], legs_from_catalog: Optional[list] = None,
                dv01_per_lot: float = SRA_DOLLARS_PER_BP_PER_LOT) -> list:
    """Build the per-leg trade breakdown.

    Parameters
    ----------
    symbol : str
        The aggregate contract symbol (single outright, spread, or fly).
    scope : str
        ``'outright'`` | ``'spread'`` | ``'fly'``.
    direction : str | None
        ``'LONG'`` | ``'SHORT'``. If None, returns [] (no actionable trade).
    lots : int | None
        Aggregate lot count (wing lots for fly, spread lots for spread,
        outright lots for outright). If None or ≤0, returns [].
    legs_from_catalog : list | None
        Pre-parsed leg symbols from the catalog ``legs`` column.
        For spread: ``[left, right]``; for fly: ``[left, mid, right]``.
        Ignored for outright.
    dv01_per_lot : float
        $ DV01 per single contract (default $25 for SRA).

    Returns
    -------
    list[dict]
        Each leg has keys: ``role, symbol, side, lots, ratio, dv01_per_bp``.
    """
    if direction not in ("LONG", "SHORT"):
        return []
    n = safe_float(lots)
    if n is None or n <= 0:
        return []
    n = int(n)

    if scope == "outright":
        side = "BUY" if direction == "LONG" else "SELL"
        return [{
            "role": "single", "symbol": symbol, "side": side,
            "lots": n, "ratio": 1, "dv01_per_bp": dv01_per_lot * n,
        }]

    legs = legs_from_catalog or []
    if scope == "spread" and len(legs) >= 2:
        left, right = legs[0], legs[1]
        if direction == "LONG":
            sides = ("BUY", "SELL")
        else:
            sides = ("SELL", "BUY")
        return [
            {"role": "left",  "symbol": left,  "side": sides[0],
             "lots": n, "ratio": 1, "dv01_per_bp": dv01_per_lot * n},
            {"role": "right", "symbol": right, "side": sides[1],
             "lots": n, "ratio": 1, "dv01_per_bp": dv01_per_lot * n},
        ]
    if scope == "fly" and len(legs) >= 3:
        left, mid, right = legs[0], legs[1], legs[2]
        if direction == "LONG":
            sides = ("BUY", "SELL", "BUY")
        else:
            sides = ("SELL", "BUY", "SELL")
        return [
            {"role": "left",  "symbol": left,  "side": sides[0],
             "lots": n, "ratio": 1, "dv01_per_bp": dv01_per_lot * n},
            {"role": "mid",   "symbol": mid,   "side": sides[1],
             "lots": 2 * n, "ratio": 2, "dv01_per_bp": dv01_per_lot * 2 * n},
            {"role": "right", "symbol": right, "side": sides[2],
             "lots": n, "ratio": 1, "dv01_per_bp": dv01_per_lot * n},
        ]
    # Fallback — single-leg representation
    side = "BUY" if direction == "LONG" else "SELL"
    return [{"role": "single", "symbol": symbol, "side": side,
              "lots": n, "ratio": 1, "dv01_per_bp": dv01_per_lot * n}]


def build_pair_legs(front_sym: str, back_sym: str, direction: Optional[str],
                     lots: Optional[int],
                     dv01_per_lot: float = SRA_DOLLARS_PER_BP_PER_LOT,
                     front_close: Optional[float] = None,
                     back_close: Optional[float] = None) -> list:
    """Build the leg breakdown for a 2-outright slope/curve pair (C9a/C9b).

    LONG  (steepener) → BUY front + SELL back  (equal lots → DV01-neutral)
    SHORT (flattener) → SELL front + BUY back

    If ``front_close`` / ``back_close`` are supplied (today's closes of each
    outright in **price** units), each emitted leg dict carries an
    ``entry_price`` — the actual price the trader would lift/hit to open the
    leg. The slope-space entry on SetupResult.entry remains in bp.
    """
    if direction not in ("LONG", "SHORT") or not front_sym or not back_sym:
        return []
    n = safe_float(lots)
    if n is None or n <= 0:
        # Default to 1 lot per side if no risk-sized count is available
        n = 1
    n = int(n)
    if direction == "LONG":
        front_side, back_side = "BUY", "SELL"
    else:
        front_side, back_side = "SELL", "BUY"
    fc = safe_float(front_close)
    bc = safe_float(back_close)
    return [
        {"role": "front", "symbol": front_sym, "side": front_side,
         "lots": n, "ratio": 1, "dv01_per_bp": dv01_per_lot * n,
         "entry_price": fc},
        {"role": "back",  "symbol": back_sym,  "side": back_side,
         "lots": n, "ratio": 1, "dv01_per_bp": dv01_per_lot * n,
         "entry_price": bc},
    ]
