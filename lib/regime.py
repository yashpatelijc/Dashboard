"""Curve regime classifier.

Classifies today's curve move (vs prior period) into a multi-dimensional regime:

  Direction:        BULL (rates ↓ / prices ↑) | BEAR (rates ↑ / prices ↓) | FLAT
  Shape:            STEEPENING | FLATTENING | BUTTERFLIED (belly bid/offered) | PARALLEL
  Magnitude:        MINOR | MODERATE | MAJOR (based on largest decomposition component)
  Section-led:      FRONT-LED | MID-LED | BACK-LED

Multiple-lookback view: the same regime computed across short / medium / long
windows so you can see if it's a one-day move or part of a multi-week regime.

Thresholds are MODULE-LEVEL CONSTANTS — identical across every lookback so the
classification is consistent. See get_regime_thresholds() for the values.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from lib.sra_data import compute_decomposition, compute_section_split


# =============================================================================
# THRESHOLDS — module-level constants, identical across all lookbacks.
# Units: basis points (bp) on the parallel/slope/curvature decomposition.
# =============================================================================
DIRECTION_THRESHOLD_BP = 1.0       # |parallel| > this → BULL or BEAR; else FLAT
SLOPE_THRESHOLD_BP     = 0.5       # |slope|    > this → STEEPENER / FLATTENER
CURV_THRESHOLD_BP      = 0.5       # |curvature|> this → BELLY-OFFERED / BELLY-BID
MAGNITUDE_MINOR_BP     = 1.0       # max component < this → MINOR
MAGNITUDE_MAJOR_BP     = 5.0       # max component ≥ this → MAJOR (otherwise MODERATE)


def get_regime_thresholds() -> dict:
    """Return the regime-classification thresholds (constant across lookbacks)."""
    return {
        "direction_bp": DIRECTION_THRESHOLD_BP,
        "slope_bp": SLOPE_THRESHOLD_BP,
        "curvature_bp": CURV_THRESHOLD_BP,
        "magnitude_minor_bp": MAGNITUDE_MINOR_BP,
        "magnitude_major_bp": MAGNITUDE_MAJOR_BP,
    }


def get_regime_thresholds_text() -> str:
    """Plain-text summary suitable for HTML 'title' tooltips. Same for every lookback."""
    return (
        "REGIME THRESHOLDS (constant across all lookbacks)\n"
        "\n"
        "DIRECTION (parallel = level shift, in bp):\n"
        f"  BULL  if parallel > +{DIRECTION_THRESHOLD_BP:.1f} bp  (rates fell)\n"
        f"  BEAR  if parallel < -{DIRECTION_THRESHOLD_BP:.1f} bp  (rates rose)\n"
        "  FLAT  otherwise\n"
        "\n"
        "SHAPE (slope = back vs front, curvature = belly bulge, in bp):\n"
        f"  STEEPENER     if slope     > +{SLOPE_THRESHOLD_BP:.1f} bp\n"
        f"  FLATTENER     if slope     < -{SLOPE_THRESHOLD_BP:.1f} bp\n"
        f"  BELLY-OFFERED if curvature > +{CURV_THRESHOLD_BP:.1f} bp  (ends up vs middle)\n"
        f"  BELLY-BID     if curvature < -{CURV_THRESHOLD_BP:.1f} bp  (middle up vs ends)\n"
        "  PARALLEL otherwise\n"
        "\n"
        "MAGNITUDE (max |component| across parallel / slope / curvature):\n"
        f"  MINOR    if max < {MAGNITUDE_MINOR_BP:.1f} bp\n"
        f"  MODERATE if {MAGNITUDE_MINOR_BP:.1f} ≤ max < {MAGNITUDE_MAJOR_BP:.1f} bp\n"
        f"  MAJOR    if max ≥ {MAGNITUDE_MAJOR_BP:.1f} bp\n"
        "\n"
        "SECTION-LED (which curve section moved most by avg |Δ|):\n"
        "  FRONT-LED / MID-LED / BACK-LED  — sections defined by Front/Mid boundary settings."
    )


def _classify_direction(parallel_bp) -> str:
    """Bull = rates fell (price up so parallel positive in PRICE-change space)."""
    if parallel_bp is None:
        return "—"
    if parallel_bp > DIRECTION_THRESHOLD_BP:
        return "BULL"
    if parallel_bp < -DIRECTION_THRESHOLD_BP:
        return "BEAR"
    return "FLAT"


def _classify_shape(slope_bp, curvature_bp) -> str:
    """Shape from slope + curvature."""
    parts = []
    if slope_bp is not None:
        if slope_bp > SLOPE_THRESHOLD_BP:
            parts.append("STEEPENER")
        elif slope_bp < -SLOPE_THRESHOLD_BP:
            parts.append("FLATTENER")
    if curvature_bp is not None:
        if curvature_bp > CURV_THRESHOLD_BP:
            parts.append("BELLY-OFFERED")
        elif curvature_bp < -CURV_THRESHOLD_BP:
            parts.append("BELLY-BID")
    return " · ".join(parts) if parts else "PARALLEL"


def _classify_magnitude(parallel_bp, slope_bp, curvature_bp) -> str:
    """Magnitude from the absolute largest component."""
    vals = [abs(v) for v in (parallel_bp, slope_bp, curvature_bp) if v is not None]
    if not vals:
        return "—"
    big = max(vals)
    if big < MAGNITUDE_MINOR_BP:
        return "MINOR"
    if big < MAGNITUDE_MAJOR_BP:
        return "MODERATE"
    return "MAJOR"


def _classify_section_led(changes: list, contracts: list, front_end: int,
                          mid_end: int) -> str:
    """Identify which curve section had the largest absolute move."""
    if not changes or len(changes) != len(contracts):
        return "—"
    fr, mr, br = compute_section_split(len(contracts), front_end, mid_end)

    def _sec_avg_abs(rng_):
        vals = [abs(changes[i]) for i in rng_
                if i < len(changes) and changes[i] is not None and not pd.isna(changes[i])]
        return sum(vals) / len(vals) if vals else 0

    f = _sec_avg_abs(fr)
    m = _sec_avg_abs(mr)
    b = _sec_avg_abs(br)
    sec = max(("FRONT", f), ("MID", m), ("BACK", b), key=lambda x: x[1])
    if sec[1] == 0:
        return "—"
    return f"{sec[0]}-LED"


def classify_regime(changes: list, contracts: list,
                    front_end: int, mid_end: int) -> dict:
    """Return regime classification dict from per-contract changes (in bp)."""
    decomp = compute_decomposition(changes)
    parallel = decomp.get("parallel")
    slope = decomp.get("slope")
    curvature = decomp.get("curvature")
    direction = _classify_direction(parallel)
    shape = _classify_shape(slope, curvature)
    magnitude = _classify_magnitude(parallel, slope, curvature)
    sec_led = _classify_section_led(changes, contracts, front_end, mid_end)

    label = magnitude
    if direction != "—" and direction != "FLAT":
        label += f" {direction}"
    if shape != "PARALLEL":
        label += f"-{shape.replace(' · ', '/')}"
    if sec_led != "—":
        label += f" ({sec_led})"
    return {
        "label": label,
        "direction": direction,
        "shape": shape,
        "magnitude": magnitude,
        "section_led": sec_led,
        "parallel_bp": parallel,
        "slope_bp": slope,
        "curvature_bp": curvature,
        "residual_rmse": decomp.get("residual_rmse"),
    }


def classify_regime_multi_lookback(wide_close: pd.DataFrame, asof_date: date,
                                    contracts: list, front_end: int, mid_end: int,
                                    lookbacks: tuple = (1, 5, 30),
                                    base_product: str = "SRA") -> dict:
    """Run regime classifier across multiple comparison horizons.

    Returns dict {lookback_label: regime_dict} for each lookback.
    Per-contract bp scaling comes from the contract-units catalog so that
    already-bp spread/fly closes aren't re-multiplied by 100.
    """
    out = {}
    ts = pd.Timestamp(asof_date)
    try:
        today_row = wide_close.loc[wide_close.index.date == asof_date].iloc[0].tolist()
    except Exception:
        return out

    # Resolve per-contract bp multipliers from the catalog (fallback ×100 for all).
    from lib.contract_units import bp_multipliers_for, load_catalog
    _cat = load_catalog()
    mults = bp_multipliers_for(contracts, base_product, _cat) if not _cat.empty \
            else [100.0] * len(contracts)

    available_dates = list(wide_close.index.date)
    for n in lookbacks:
        # Find date n trading days before asof_date
        try:
            asof_idx = available_dates.index(asof_date)
            prior_idx = max(0, asof_idx - n)
            prior_date = available_dates[prior_idx]
            prior_row = wide_close.loc[wide_close.index.date == prior_date].iloc[0].tolist()
        except (ValueError, IndexError):
            continue

        changes_bp = []
        for t, p, m in zip(today_row, prior_row, mults):
            if t is None or p is None or pd.isna(t) or pd.isna(p):
                changes_bp.append(None)
            else:
                changes_bp.append((t - p) * m)
        regime = classify_regime(changes_bp, contracts, front_end, mid_end)
        regime["lookback_days"] = n
        regime["prior_date"] = prior_date
        out[f"{n}d"] = regime
    return out
