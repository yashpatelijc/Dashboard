"""FOMC meeting data + meeting-impact decomposition from SOFR futures.

Methodology (CME-style, from CME white paper "Decoding Implied Policy Rates from
SOFR Futures" plus Standard textbook chapter on STIR):

1. Each CME 3-Month SOFR (SR3) contract settles to the COMPOUNDED daily SOFR rate
   over its 3-month "reference period" running from the 3rd Wednesday of the contract
   month to the 3rd Wednesday three months later.

2. Daily SOFR is closely tied to the Fed Funds target. Between FOMC meetings, daily
   SOFR is approximately constant at the prevailing target. After each FOMC meeting
   that changes the target, SOFR jumps by that amount.

3. So the contract's implied rate (100 - price) ≈ a day-weighted average of the
   prevailing target rates within its reference period:

        implied_rate_c  ≈  (1/N_c) × Σ_i d_i × r_i

    where r_i is the rate prevailing during sub-segment i (between meetings)
    and d_i is the number of business days in that sub-segment.

4. Given M consecutive (non-overlapping) quarterly contracts and K FOMC meetings
   within the modelled window, we get M equations and K+1 unknowns (one rate
   per inter-meeting segment). When M >= K+1 the system is solvable; otherwise
   we use ridge regression.

5. Solving the system gives us, for each FOMC meeting, the market-implied
   POST-MEETING rate. The difference between consecutive implied rates is the
   meeting's expected move in basis points.

6. Probability of hike/cut/hold uses the standard binary mapping:
        Δ_bp = post_meeting − pre_meeting
        P(hike 25bp) ≈ max(0, Δ_bp / 25)
        P(cut 25bp)  ≈ max(0, -Δ_bp / 25)
        P(hold)      ≈ 1 - |Δ_bp / 25|
   (clamped to [0, 1])
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import yaml


_FOMC_FILE = os.path.join(os.path.dirname(__file__), "..", "config", "fomc_meetings.yaml")


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=86400)
def load_fomc_meetings() -> pd.DataFrame:
    """Load FOMC schedule from YAML. Columns: decision_date · press_conf · sep."""
    if not os.path.exists(_FOMC_FILE):
        return pd.DataFrame(columns=["decision_date", "press_conf", "sep"])
    with open(_FOMC_FILE, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    rows = data.get("meetings", []) or []
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["decision_date"] = pd.to_datetime(df["decision_date"]).dt.date
    df = df.sort_values("decision_date").reset_index(drop=True)
    return df


def get_fomc_dates_in_range(start: date, end: date) -> List[date]:
    """Return list of FOMC decision dates within [start, end]."""
    df = load_fomc_meetings()
    if df.empty:
        return []
    mask = (df["decision_date"] >= start) & (df["decision_date"] <= end)
    return list(df.loc[mask, "decision_date"])


def next_fomc_date(asof: date) -> Optional[date]:
    df = load_fomc_meetings()
    upcoming = df[df["decision_date"] > asof]
    if upcoming.empty:
        return None
    return upcoming.iloc[0]["decision_date"]


def previous_fomc_date(asof: date) -> Optional[date]:
    df = load_fomc_meetings()
    past = df[df["decision_date"] <= asof]
    if past.empty:
        return None
    return past.iloc[-1]["decision_date"]


# -----------------------------------------------------------------------------
# Contract reference-period helpers
# -----------------------------------------------------------------------------
_MONTH_CODE_TO_NUM = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}


def parse_sra_outright(symbol: str) -> Optional[tuple]:
    """Parse '{BASE}{month_code}{year_2d}' → (year_4d, month_1to12) or None.

    Originally SRA-only; now generalized to all STIR markets registered in
    `lib.markets.MARKETS` (SRA, ER, FSR, FER, SON, YBA, CRA). Probes each
    known base-product prefix and parses the trailing month-code + 2-digit
    year. SR3 outrights still work; ER / SON / YBA / CRA / FSR / FER all
    parse correctly.
    """
    if not symbol or len(symbol) < 4:
        return None
    # Probe known base-product prefixes (longest first so 'SRA' beats 'SR')
    try:
        from lib.markets import MARKETS as _MK
        prefixes = sorted(_MK.keys(), key=len, reverse=True)
    except Exception:
        prefixes = ["SRA", "FSR", "FER", "YBA", "CRA", "ER", "SON"]
    for base in prefixes:
        if not symbol.startswith(base):
            continue
        remainder = symbol[len(base):]
        if len(remainder) < 3:
            continue
        code = remainder[0].upper()
        if code not in _MONTH_CODE_TO_NUM:
            continue
        try:
            yy = int(remainder[1:3])
        except ValueError:
            continue
        year = 2000 + yy if yy < 70 else 1900 + yy
        return year, _MONTH_CODE_TO_NUM[code]
    return None


def third_wednesday(year: int, month: int) -> date:
    """Return the 3rd Wednesday of (year, month)."""
    d = date(year, month, 1)
    # Mon=0 ... Sun=6 ; we want Wed (=2)
    first_wed_offset = (2 - d.weekday()) % 7
    first_wed = d + timedelta(days=first_wed_offset)
    return first_wed + timedelta(days=14)


def reference_period(symbol: str) -> Optional[tuple]:
    """Return (start_date, end_date) — the 3-month reference period for an SR3 contract.

    Reference period: 3rd Wednesday of contract month → 3rd Wednesday of contract
    month + 3 months. (CME spec for SR3 / 3-Month SOFR Futures.)
    """
    parsed = parse_sra_outright(symbol)
    if not parsed:
        return None
    year, month = parsed
    start = third_wednesday(year, month)
    # End = 3rd Wednesday of month+3
    new_month = month + 3
    new_year = year
    if new_month > 12:
        new_month -= 12
        new_year += 1
    end = third_wednesday(new_year, new_month)
    return start, end


def is_quarterly(symbol: str) -> bool:
    """True if the symbol's month code is a quarterly: H/M/U/Z."""
    p = parse_sra_outright(symbol)
    if not p:
        return False
    _, month = p
    return month in (3, 6, 9, 12)


# -----------------------------------------------------------------------------
# Meeting-impact decomposition
# -----------------------------------------------------------------------------
def decompose_implied_rates(
    contracts_with_rates: list,
    asof: date,
    anchor_rate_pct: float = None,
    horizon_months: int = 18,
    ridge_lambda: float = 0.5,
    cb_code: str = "fed",
    step_size_bp: float = 25.0,
) -> pd.DataFrame:
    """Decompose contract implied rates into a per-meeting implied rate path.

    Works for any STIR market in the registry: SR3 (Fed), Euribor (ECB),
    SARON (SNB), €STR (ECB), SONIA (BoE), Bank Bill (RBA), CORRA (BoC).

    METHODOLOGY (CME-aligned + numerical-stability hardening):

    1. Each contract settles to the COMPOUNDED daily overnight rate over its 3-month
       reference period (3rd Wed of contract month → 3rd Wed of contract month + 3).
       We approximate compounding with a day-weighted arithmetic average — accurate
       to within a few bp for typical overnight rate levels.

       For Euribor / Bank Bill (single 3M IBOR fixing settlement), the same
       reference-period approximation gives the implied FORWARD 3M rate path —
       deltas between consecutive meetings still approximate the expected policy
       move (assuming the credit/term basis is roughly stable across meetings).

    2. Daily overnight rate closely tracks the policy rate target. Between meetings
       it's ≈ constant; at each meeting it jumps by the policy change. So the daily-rate
       path is a step function with K steps (one per CB meeting in window).

    3. Variables: r_0 (rate BEFORE the first in-window FOMC), r_1 (after FOMC #1),
       ..., r_K (after FOMC #K). K+1 unknowns.

    4. Each contract gives an equation:
            implied_rate_c × N_c  =  Σ_i  d_{c,i} × r_i
       where N_c = total business days in contract ref period and d_{c,i} = business
       days of segment i (between FOMCs) that overlap contract c's ref period.

    5. ANCHORING: r_0 is anchored to `anchor_rate_pct` (current SOFR effective)
       via a soft prior with weight `ridge_lambda` × n_contracts. This stabilises
       the system when underdetermined (fewer contracts than unknowns) and ensures
       the front-end implied rate matches reality.

    6. RIDGE REGULARISATION: small L2 penalty on Δ-between-segments encourages
       a smooth rate path (avoiding the wild oscillations a pure least-squares
       solution exhibits when M < K).

    7. We solve the regularized normal equations:
            (A^T A + λ_anchor × E_0 + λ_smooth × D^T D) r = A^T b + λ_anchor × E_0 anchor
       where E_0 is the "anchor" matrix that pins r_0 and D^T D penalises
       differences between adjacent rates.

    8. PROBABILITY MAPPING: assumes the next FOMC moves by integer multiples of
       25 bp. If implied Δ = +18 bp → P(hike 25) = 18/25 = 72%, P(hold) = 28%.
       For |Δ| > 25 bp we map proportionally up to 100% with overflow accounting
       for multi-step moves.

    LIMITATIONS:
    - Day-weighted average ≠ exact compounding (small bias for high vol)
    - Assumes rates only change at FOMCs (ignores BS-decisions / non-FOMC moves)
    - Ridge λ trades off responsiveness vs noise — default 0.5 is a balanced choice
    """
    if not contracts_with_rates:
        return pd.DataFrame()

    # Filter to quarterlies (non-overlapping ref periods → cleaner system)
    quarterly = [(s, r) for s, r in contracts_with_rates
                 if is_quarterly(s) and r is not None and not pd.isna(r)]
    if len(quarterly) < 2:
        return pd.DataFrame()

    # Limit to contracts within `horizon_months` from asof
    horizon_end = asof + timedelta(days=int(horizon_months * 30.4))
    quarterly = [(s, r) for s, r in quarterly
                 if reference_period(s) is not None
                 and reference_period(s)[0] <= horizon_end]
    if len(quarterly) < 2:
        return pd.DataFrame()

    quarterly_sorted = sorted(quarterly, key=lambda x: reference_period(x[0])[0])
    earliest = reference_period(quarterly_sorted[0][0])[0]
    latest = reference_period(quarterly_sorted[-1][0])[1]

    # Load the meeting calendar for the requested central bank.
    if cb_code == "fed":
        fomcs = get_fomc_dates_in_range(earliest, latest)
    else:
        try:
            from lib.central_banks import get_dates_in_range as _gdir
            fomcs = _gdir(cb_code, earliest, latest)
        except Exception:
            fomcs = []
    if not fomcs:
        return pd.DataFrame()

    seg_boundaries = [earliest] + list(fomcs) + [latest]
    K = len(fomcs)
    n_segs = K + 1
    M = len(quarterly_sorted)

    # Build A (M × n_segs) and b
    A = np.zeros((M, n_segs))
    b = np.zeros(M)
    for ci, (sym, rate) in enumerate(quarterly_sorted):
        ref_start, ref_end = reference_period(sym)
        N_c = pd.bdate_range(ref_start, ref_end - timedelta(days=1)).size
        if N_c == 0:
            continue
        for si in range(n_segs):
            seg_start = seg_boundaries[si]
            seg_end = seg_boundaries[si + 1]
            lo = max(seg_start, ref_start)
            hi = min(seg_end, ref_end)
            if hi <= lo:
                continue
            d = pd.bdate_range(lo, hi - timedelta(days=1)).size
            A[ci, si] = d / N_c
        b[ci] = rate

    # Anchor the first segment to current SOFR (or first contract's rate if not given)
    if anchor_rate_pct is None:
        anchor_rate_pct = float(quarterly_sorted[0][1])

    # Build smoothness penalty matrix D (first-difference operator) — penalises |r_i - r_{i-1}|
    if n_segs >= 2:
        D = np.zeros((n_segs - 1, n_segs))
        for i in range(n_segs - 1):
            D[i, i] = 1
            D[i, i + 1] = -1
    else:
        D = np.zeros((1, n_segs))

    # Anchor matrix: pins r_0 with strong weight
    E0 = np.zeros((1, n_segs))
    E0[0, 0] = 1.0

    # Form the augmented system: minimise || [A; λ_a E0; λ_s D] r - [b; λ_a anchor; 0] ||
    lambda_anchor = float(M) * 5.0   # strong anchor (5× per equation effective weight)
    lambda_smooth = float(ridge_lambda) * float(M)

    A_aug = np.vstack([A,
                        lambda_anchor * E0,
                        lambda_smooth * D])
    b_aug = np.concatenate([b,
                             [lambda_anchor * anchor_rate_pct],
                             np.zeros(D.shape[0])])

    try:
        r_segs, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=None)
    except Exception:
        return pd.DataFrame()

    # Build result
    rows = []
    for i, m_date in enumerate(fomcs):
        if m_date < asof:
            continue
        post = float(r_segs[i + 1])
        prior = float(r_segs[i])
        delta_bp = (post - prior) * 100.0
        front_rate = float(r_segs[0])
        cum_from_front_bp = (post - front_rate) * 100.0

        # Probability mapping using the market's step size (default 25bp).
        # Maps implied Δ to {hike, cut, hold} probs.
        step = float(step_size_bp) if step_size_bp and step_size_bp > 0 else 25.0
        if abs(delta_bp) <= step:
            p_hike = max(0.0, delta_bp / step)
            p_cut = max(0.0, -delta_bp / step)
            p_hold = max(0.0, 1.0 - p_hike - p_cut)
        else:
            p_hike = min(1.0, max(0.0, delta_bp) / step)
            p_cut = min(1.0, max(0.0, -delta_bp) / step)
            p_hold = max(0.0, 1.0 - p_hike - p_cut)

        rows.append({
            "meeting_date": m_date,
            "post_rate_pct": post,
            "prior_rate_pct": prior,
            "delta_bp": delta_bp,
            "cum_from_front_bp": cum_from_front_bp,
            "prob_hike_25": p_hike,
            "prob_cut_25": p_cut,
            "prob_hold": p_hold,
        })

    return pd.DataFrame(rows)


def get_methodology_text() -> str:
    """Plain-text methodology summary for tooltips / side-panel display."""
    return (
        "FOMC IMPLIED POLICY PATH — METHODOLOGY\n\n"
        "Source contracts: SR3 (3-Month SOFR) quarterly outrights H/M/U/Z within 18 months.\n\n"
        "Each contract's implied rate (= 100 − close) ≈ day-weighted average of the daily\n"
        "SOFR rate over its 3-month reference period [3rd Wed of contract month → 3rd Wed of\n"
        "contract month + 3]. Daily SOFR is approximately constant between FOMC meetings, so\n"
        "the rate path is a step function with one jump per FOMC.\n\n"
        "Variables: r_0 (rate before first in-window FOMC), r_1...r_K (after each FOMC).\n"
        "Equation per contract: implied_rate × N = Σ d_i × r_i (intersect contract & segments).\n\n"
        "Solved via regularised least-squares with:\n"
        "  • ANCHOR — r_0 is pinned to current SOFR effective (front-end ≈ market reality)\n"
        "  • SMOOTHNESS prior — small ridge on |r_i − r_{i−1}| stabilises the underdetermined\n"
        "    system and dampens noise (default λ = 0.5 × n_contracts)\n\n"
        "Probability mapping (25 bp step assumption):\n"
        "  P(hike 25) = max(0, Δ/25),  P(cut 25) = max(0, −Δ/25),  P(hold) = 1 − the above\n"
        "  (overflow above 25 bp implies multi-step move; capped at 100%).\n\n"
        "LIMITATIONS:\n"
        "  • Day-weighted average is a small approximation to compounded SOFR\n"
        "  • Assumes rates only change AT FOMCs (no inter-meeting moves)\n"
        "  • Ridge λ trades responsiveness for stability"
    )


def annotate_contracts_with_fomcs(contracts: list) -> dict:
    """For each contract, return the list of FOMC meeting dates in its reference period."""
    out = {}
    for sym in contracts:
        rp = reference_period(sym)
        if rp is None:
            out[sym] = []
            continue
        start, end = rp
        out[sym] = get_fomc_dates_in_range(start, end)
    return out
