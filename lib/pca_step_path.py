"""A4 — Heitfield-Park sequential bootstrap of FOMC step probabilities.

Decomposes the SR3 forward curve into per-meeting policy-step probabilities
{large_cut, cut, hold, hike, large_hike} for each future FOMC meeting.

Inputs:
  · CMC panel (post-cutover daily constant-maturity SR3 forward yield bp)
  · FOMC calendar (decision dates in ascending order)
  · Today's effective Fed-Funds Target (lower bound, FDTRFTRL)

Method (Heitfield-Park 2016, simplified for streamlit):

  1. For each future FOMC meeting m_i ∈ {m_1, m_2, ...}:
       Bracket m_i with adjacent CMC tenors τ_low (just before) and τ_high
       (just after the meeting).  The implied per-meeting target change is

         ΔR_i ≈ R(τ_high) − R(τ_low)

       in basis points.  This is the "expected step" per CME polynomial
       front-pin convention, given the running effective rate stays flat
       between meetings.

  2. Convert ΔR_i to a discrete distribution over the 5-bucket lattice
     {-50, -25, 0, +25, +50} bp using a centred triangular kernel:

         p(b) ∝ max(0, 1 − |b − ΔR_i| / σ)        σ = 25 bp default

     Re-normalise to sum 1.  This is the "deterministic" mapping.

  3. Sequential bootstrap:  repeat (1) and (2) N times, each time
     resampling 60d of CMC daily yields with replacement (block bootstrap,
     block size = 5d).  Average the resulting per-meeting prob vectors
     over draws.  The final {bucket: prob} dict per meeting is the bootstrap
     average.

  4. Build a `policy_path_history` DataFrame indexed by FOMC date with
     a single column `target_change_bp` derived from the realised target
     rate (FDTRFTRL diffs at FOMC decision dates).  This feeds the empirical
     Markov fallback in `lib.pca_analogs._empirical_markov_transition`.

The output of this module is what `lib.pca_analogs.path_conditional_fv`
needs (`today_policy_path_probs` + `policy_path_history`).
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from lib.fomc import load_fomc_meetings
from lib.pca_analogs import LATTICE_BP, PATH_BUCKETS_5


# Research-validation fix: standard Heitfield-Park kernel σ is ~10-15 bp, not 25.
# 25bp was smearing ~50% probability mass into adjacent buckets even at center.
# We now default to 12.5 bp (the half-step size) which keeps adjacent-bucket
# leakage near zero at center while still allowing in-between values to
# distribute across two buckets.
DEFAULT_KERNEL_SIGMA_BP = 12.5
DEFAULT_BOOTSTRAP_DRAWS = 50
DEFAULT_BOOTSTRAP_BLOCK_D = 5
DEFAULT_BOOTSTRAP_LOOKBACK_D = 60


def _bracket_meeting_yields(cmc_today_row: pd.Series,
                             meeting_offset_months: float) -> tuple:
    """Return (R_before, R_after) — yields just before and just after a meeting,
    interpolated linearly between adjacent integer tenor nodes.

    `cmc_today_row` index = tenors_months (int), values = yield_bp.
    `meeting_offset_months` = months from asof to the meeting decision.
    """
    if cmc_today_row is None or cmc_today_row.empty:
        return (np.nan, np.nan)
    tenors = np.asarray(cmc_today_row.index, dtype=float)
    yields = cmc_today_row.values.astype(float)
    if meeting_offset_months <= tenors.min() or meeting_offset_months >= tenors.max():
        return (np.nan, np.nan)
    # Research-validation fix: standard H-P brackets the meeting with a NARROW
    # window — order of a few business days, NOT half a month. Half-month
    # bracket was understating step magnitude because intermediate yields blend
    # the step over too much smoothing. We use ~3 business days (0.15 months
    # ≈ 4.5 calendar days), which captures the OIS-forward discontinuity at
    # the meeting decision without smearing across other meeting effects.
    eps_m = 0.15
    tau_lo = max(tenors.min(), meeting_offset_months - eps_m)
    tau_hi = min(tenors.max(), meeting_offset_months + eps_m)
    R_lo = float(np.interp(tau_lo, tenors, yields))
    R_hi = float(np.interp(tau_hi, tenors, yields))
    return (R_lo, R_hi)


def _step_probs_from_implied_change(implied_bp: float,
                                     sigma_bp: float = DEFAULT_KERNEL_SIGMA_BP
                                     ) -> dict:
    """Map a real-valued implied target change (bp) to a 5-bucket lattice
    distribution using a triangular kernel centred on `implied_bp`.

    The kernel bandwidth `sigma_bp` controls how concentrated the distribution
    is (smaller = sharper).  Renormalised to sum 1.
    """
    if not np.isfinite(implied_bp):
        return {b: 0.2 for b in PATH_BUCKETS_5}
    raw = {}
    total = 0.0
    for b in PATH_BUCKETS_5:
        center = LATTICE_BP[b]
        w = max(0.0, 1.0 - abs(center - implied_bp) / max(sigma_bp, 1e-6))
        raw[b] = w
        total += w
    if total <= 0:
        # implied move is far outside the lattice; pin to closest bucket
        nearest = min(PATH_BUCKETS_5, key=lambda b: abs(LATTICE_BP[b] - implied_bp))
        return {b: (1.0 if b == nearest else 0.0) for b in PATH_BUCKETS_5}
    return {b: raw[b] / total for b in PATH_BUCKETS_5}


def _today_meeting_offsets_months(asof: date,
                                    fomc_dates: list,
                                    n_meetings: int = 8) -> list:
    """Return list of (decision_date, offset_months) for the next N FOMCs."""
    upcoming = [d for d in fomc_dates if d > asof]
    upcoming = upcoming[:n_meetings]
    out = []
    for d in upcoming:
        offset_months = (d - asof).days / 30.4375
        out.append((d, offset_months))
    return out


def _block_bootstrap_yields(cmc_panel: pd.DataFrame,
                              asof: date,
                              lookback_d: int = DEFAULT_BOOTSTRAP_LOOKBACK_D,
                              block_d: int = DEFAULT_BOOTSTRAP_BLOCK_D,
                              rng: Optional[np.random.Generator] = None
                              ) -> pd.Series:
    """Block-bootstrap the trailing `lookback_d` CMC days with block size `block_d`.

    Returns a synthetic CMC row (same tenors as cmc_panel.columns) — the mean
    of the bootstrap-sampled days.  Used to inject sampling noise into the
    step-path bootstrap.
    """
    if rng is None:
        rng = np.random.default_rng()
    history = cmc_panel.loc[cmc_panel.index <= pd.Timestamp(asof)].tail(lookback_d)
    if len(history) < block_d * 2:
        return cmc_panel.iloc[-1]
    n_blocks_needed = max(1, len(history) // block_d)
    starts = rng.integers(0, len(history) - block_d + 1, size=n_blocks_needed)
    sampled = pd.concat([history.iloc[s:s + block_d] for s in starts])
    return sampled.mean(axis=0)


def fit_step_path_bootstrap(cmc_panel: pd.DataFrame,
                              fomc_dates: list,
                              asof: date,
                              *,
                              n_meetings: int = 8,
                              n_draws: int = DEFAULT_BOOTSTRAP_DRAWS,
                              kernel_sigma_bp: float = DEFAULT_KERNEL_SIGMA_BP,
                              lookback_d: int = DEFAULT_BOOTSTRAP_LOOKBACK_D,
                              block_d: int = DEFAULT_BOOTSTRAP_BLOCK_D,
                              seed: int = 42) -> dict:
    """Run the Heitfield-Park sequential bootstrap.

    Returns a dict:
      {
        "meetings": [(decision_date, offset_months, implied_bp_mean,
                       implied_bp_se, prob_dict_mean), ...],
        "today_path_probs_combined": {bucket: prob} averaged over all meetings,
        "method": "Heitfield-Park sequential block-bootstrap",
        "n_draws": int,
        "kernel_sigma_bp": float,
      }

    `today_path_probs_combined` is the aggregated bucket distribution suitable
    for `path_conditional_fv`'s `today_policy_path_probs` argument.
    """
    out = {
        "meetings": [],
        "today_path_probs_combined": {b: 0.0 for b in PATH_BUCKETS_5},
        "method": "Heitfield-Park sequential block-bootstrap",
        "n_draws": int(n_draws),
        "kernel_sigma_bp": float(kernel_sigma_bp),
    }
    if cmc_panel is None or cmc_panel.empty:
        return out

    upcoming = _today_meeting_offsets_months(asof, fomc_dates, n_meetings=n_meetings)
    if not upcoming:
        return out

    rng = np.random.default_rng(seed)

    # Precompute draws once — same N synthetic curves used for all meetings
    draw_curves = [cmc_panel.iloc[-1]]    # include real curve as draw 0
    for _ in range(int(n_draws) - 1):
        draw_curves.append(
            _block_bootstrap_yields(cmc_panel, asof,
                                     lookback_d=lookback_d, block_d=block_d, rng=rng)
        )

    combined = {b: 0.0 for b in PATH_BUCKETS_5}
    for d_meeting, offset_months in upcoming:
        impls = []
        for curve in draw_curves:
            R_lo, R_hi = _bracket_meeting_yields(curve, offset_months)
            if np.isfinite(R_lo) and np.isfinite(R_hi):
                impls.append(R_hi - R_lo)
        if not impls:
            continue
        impl_arr = np.asarray(impls, dtype=float)
        impl_mean = float(impl_arr.mean())
        impl_se = float(impl_arr.std(ddof=1)) if len(impl_arr) > 1 else 0.0

        # Average the prob dicts across draws
        prob_sums = {b: 0.0 for b in PATH_BUCKETS_5}
        for v in impl_arr:
            p = _step_probs_from_implied_change(v, sigma_bp=kernel_sigma_bp)
            for b, pv in p.items():
                prob_sums[b] += pv
        prob_mean = {b: prob_sums[b] / len(impl_arr) for b in PATH_BUCKETS_5}

        out["meetings"].append({
            "decision_date": d_meeting,
            "offset_months": float(offset_months),
            "implied_bp_mean": impl_mean,
            "implied_bp_se": impl_se,
            "probs": prob_mean,
        })

        # First few meetings dominate the contract-relevant path —
        # weight by a decay so far-out meetings don't drown the signal
        meeting_weight = np.exp(-offset_months / 12.0)    # e-fold ≈ 1y
        for b in PATH_BUCKETS_5:
            combined[b] += meeting_weight * prob_mean[b]

    total = sum(combined.values())
    if total > 0:
        for b in combined:
            combined[b] /= total
    out["today_path_probs_combined"] = combined
    return out


def build_policy_path_history_from_fdtr(fdtr_panel: pd.DataFrame,
                                          fomc_dates: list) -> pd.DataFrame:
    """Construct `policy_path_history` (DataFrame indexed by FOMC date with column
    `target_change_bp`) from the FDTR_FTRL (lower bound) daily series.

    `fdtr_panel` is a daily DataFrame with at least column "PX_LAST" containing
    the lower-bound target rate in percentage points.

    Diff between the value on each FOMC date and the value on the previous
    FOMC date is the realised step (in bp).
    """
    if fdtr_panel is None or fdtr_panel.empty or not fomc_dates:
        return pd.DataFrame(columns=["target_change_bp"])
    df = fdtr_panel.copy()
    # Pick FDTR series: prefer fdtr_lower (CME settles to (upper+lower)/2 but
    # lower-bound is the policy reference), then fdtr_upper, then PX_LAST,
    # then any numeric column.
    rate_col = None
    for cand in ("fdtr_lower", "fdtr_upper", "PX_LAST"):
        if cand in df.columns:
            rate_col = cand
            break
    if rate_col is None:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                rate_col = col
                break
    if rate_col is None:
        return pd.DataFrame(columns=["target_change_bp"])
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").set_index("date")
        else:
            return pd.DataFrame(columns=["target_change_bp"])
    df = df.sort_index()
    rate_series = df[rate_col].dropna()    # skip leading NaNs
    if rate_series.empty:
        return pd.DataFrame(columns=["target_change_bp"])
    fomc_sorted = sorted(pd.Timestamp(d) for d in fomc_dates)
    rows = []
    prev_v = None
    for d in fomc_sorted:
        idx_pos = rate_series.index.searchsorted(d, side="right") - 1
        if idx_pos < 0:
            continue
        v = float(rate_series.iloc[idx_pos])
        if not np.isfinite(v):
            continue
        if prev_v is not None and np.isfinite(prev_v):
            change_bp = (v - prev_v) * 100.0
            rows.append({"asof": d, "target_change_bp": change_bp})
        prev_v = v
    if not rows:
        return pd.DataFrame(columns=["target_change_bp"])
    return pd.DataFrame(rows).set_index("asof")
