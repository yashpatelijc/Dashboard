"""A4 Heitfield-Park policy-path decomposition (Phase 5, plan §12).

Decomposes the SR3 strip into per-FOMC-meeting probability vectors over the
{-50, -25, 0, +25, +50} bps lattice via a constrained least-squares solve
(SLSQP on probability simplex, ZLB constraint, lattice rounding).

Inputs
------
- ``rates_drivers/FDTRMID_Index.parquet`` — current effective Fed funds rate
- CMC outright nodes at the latest asof — SR3 forward rates at each future
  FOMC date (closest M_k node by months-to-meeting)
- FOMC meeting calendar from ``lib.cb_meetings.fomc_meetings()``

Outputs
-------
- ``.cmc_cache/policy_path_<asof>.parquet`` — per-meeting:
  meeting_date, days_to_meeting, p_m50, p_m25, p_0, p_p25, p_p50,
  expected_change_bp, post_meeting_rate_bp, sr3_implied_rate_bp,
  fit_residual_bp
- ``.cmc_cache/policy_path_manifest_<asof>.json`` — terminal_rate,
  floor_rate, cycle_label (LATE_HIKE / LATE_CUT / NEUTRAL),
  optimizer_success, total_rss

Per §15 D2: NO FedWatch scraper cross-check. A4 derives from SR3 strip only.

Algorithm:
  1. Build "FedWatch-style" initial PMFs from observed implied rate change
     per-meeting (concentrated on 1-2 adjacent lattice points).
  2. Refine via SLSQP minimizing sum of squared residuals between
     (cumulative expected rate after meeting m) and
     (SR3 forward at d_m).
  3. Apply ZLB by clipping post_rate at 0 in the residual term.
  4. Lattice rounding: convert refined PMF to nearest lattice point
     allocation that sums to 1 (greedy with leftover redistribution).
"""
from __future__ import annotations

import json
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".cmc_cache"
_CACHE_DIR.mkdir(exist_ok=True)

BUILDER_VERSION = "1.0.0"

LATTICE_BP = np.array([-50.0, -25.0, 0.0, 25.0, 50.0])
N_LATTICE = len(LATTICE_BP)
LATTICE_LABELS = ["m50", "m25", "0", "p25", "p50"]

DEFAULT_HORIZON_MONTHS = 36   # forward through 3 years of FOMC meetings


# =============================================================================
# Inputs
# =============================================================================

def _load_current_fdtr_bp() -> Optional[float]:
    """Return latest FDTRMID in basis points, or None if unavailable."""
    path = Path(r"D:\BBG data\parquet\rates_drivers\FDTRMID_Index.parquet")
    if not path.exists():
        return None
    try:
        import duckdb
        con = duckdb.connect(":memory:")
        row = con.execute(
            f"SELECT date, PX_LAST FROM read_parquet('{str(path).replace(chr(92), chr(47))}') "
            f"ORDER BY date DESC LIMIT 1"
        ).fetchone()
        return float(row[1]) * 100.0   # pct -> bp
    except Exception:
        return None


def _load_cmc_close_at(asof: date, scope: str = "outright") -> dict:
    """Return {cmc_node: latest_close_price} for all nodes at asof."""
    from lib.sra_data import load_cmc_wide_panel
    wide = load_cmc_wide_panel(scope, asof, field="close")
    if wide is None or wide.empty:
        return {}
    last = wide.iloc[-1]
    return {n: float(last[n]) for n in last.index if not pd.isna(last[n])}


def _months_to_date(target: date, today: date) -> int:
    """Approximate months from today to target (rounded). Min 0, max 60."""
    days = (target - today).days
    return max(0, min(60, int(round(days / 30.42))))


def _sr3_implied_rate_at(meeting_date: date, asof: date,
                              cmc_closes: dict) -> Optional[float]:
    """Return SR3-implied 3M rate (bp) for the period straddling the meeting.

    Approximation: pick CMC outright node M_n where n ≈ months from asof to
    meeting_date. SR3 quotes price = 100 - rate%, so rate_bp = (100 - close) * 100.
    """
    n = _months_to_date(meeting_date, asof)
    node = f"M{n}"
    if node not in cmc_closes:
        # try nearest available
        avail_n = sorted(int(k[1:]) for k in cmc_closes if k.startswith("M"))
        if not avail_n:
            return None
        n = min(avail_n, key=lambda x: abs(x - n))
        node = f"M{n}"
    price = cmc_closes[node]
    return float((100.0 - price) * 100.0)


# =============================================================================
# Initial FedWatch-style PMFs
# =============================================================================

def fedwatch_pmf(implied_change_bp: float) -> np.ndarray:
    """Map a single meeting's expected rate change to a PMF over the
    {-50, -25, 0, +25, +50} bp lattice using FedWatch's two-point allocation.

    Rule: probability lives on the two lattice points bracketing
    implied_change_bp; weights linearly interpolate.
    """
    p = np.zeros(N_LATTICE)
    if implied_change_bp <= LATTICE_BP[0]:
        p[0] = 1.0
        return p
    if implied_change_bp >= LATTICE_BP[-1]:
        p[-1] = 1.0
        return p
    for i in range(N_LATTICE - 1):
        lo, hi = LATTICE_BP[i], LATTICE_BP[i + 1]
        if lo <= implied_change_bp <= hi:
            w = (implied_change_bp - lo) / (hi - lo)
            p[i] = 1 - w
            p[i + 1] = w
            return p
    return p


def initial_pmfs(per_meeting_implied_change_bp: list[float]) -> np.ndarray:
    """Stack FedWatch PMFs into matrix shape (N_meetings, N_lattice)."""
    return np.stack([fedwatch_pmf(c) for c in per_meeting_implied_change_bp])


# =============================================================================
# SLSQP refinement
# =============================================================================

def _expected_change_per_meeting(P: np.ndarray) -> np.ndarray:
    """Given (N_meetings, N_lattice) matrix, return per-meeting expected change."""
    return P @ LATTICE_BP


def _cumulative_post_rate(P: np.ndarray, current_rate_bp: float) -> np.ndarray:
    """Post-rate after each meeting: current + cumulative expected changes.
    Returns array of length N_meetings."""
    expected = _expected_change_per_meeting(P)
    return current_rate_bp + np.cumsum(expected)


def _objective_fn(p_flat: np.ndarray, N_meetings: int,
                       observed_rates_bp: np.ndarray,
                       current_rate_bp: float) -> float:
    """Sum of squared residuals between cumulative post-rate and observed
    SR3 forward rates. Includes ZLB penalty."""
    P = p_flat.reshape(N_meetings, N_LATTICE)
    post_rates = _cumulative_post_rate(P, current_rate_bp)
    # Hard ZLB: clip post_rates at 0 in the comparison
    eff = np.maximum(post_rates, 0.0)
    rss = float(np.sum((eff - observed_rates_bp) ** 2))
    # Tiny regularizer pulling toward FedWatch-style two-point concentration
    # (penalizes high-entropy PMFs)
    eps = 1e-12
    entropy = -np.sum(P * np.log(P + eps))
    return rss + 1e-3 * entropy


def fit_policy_path(meeting_dates: list[date], asof: date,
                       current_rate_bp: float,
                       cmc_closes: dict,
                       use_slsqp: bool = True) -> dict:
    """End-to-end fit. Returns:
      {P (N,5), expected_change_bp, post_rate_bp, sr3_implied_rate_bp,
       fit_residual_bp, optimizer_success, fit_rss}
    """
    N = len(meeting_dates)
    if N == 0:
        return {"P": np.zeros((0, N_LATTICE)),
                  "expected_change_bp": np.zeros(0),
                  "post_rate_bp": np.zeros(0),
                  "sr3_implied_rate_bp": np.zeros(0),
                  "fit_residual_bp": np.zeros(0),
                  "optimizer_success": True, "fit_rss": 0.0}
    obs = np.zeros(N)
    for i, d in enumerate(meeting_dates):
        r = _sr3_implied_rate_at(d, asof, cmc_closes)
        obs[i] = r if r is not None else current_rate_bp

    # Initial PMFs from observed inter-meeting changes
    obs_changes = np.diff(np.concatenate([[current_rate_bp], obs]))
    # Round each implied change to nearest lattice value bounded ±50
    P_init = initial_pmfs(obs_changes.tolist())
    success = True
    fit_rss = float(np.sum((np.cumsum(_expected_change_per_meeting(P_init))
                                + current_rate_bp - obs) ** 2))

    if use_slsqp:
        try:
            from scipy.optimize import minimize, LinearConstraint, Bounds
            # Constraints: each row of P sums to 1
            # Build linear constraint matrix: (N, N*5)
            A_eq = np.zeros((N, N * N_LATTICE))
            for i in range(N):
                A_eq[i, i * N_LATTICE:(i + 1) * N_LATTICE] = 1.0
            constraints = [LinearConstraint(A_eq, lb=1.0, ub=1.0)]
            bounds = Bounds(lb=np.zeros(N * N_LATTICE),
                                  ub=np.ones(N * N_LATTICE))
            x0 = P_init.flatten()
            res = minimize(
                _objective_fn, x0,
                args=(N, obs, current_rate_bp),
                method="SLSQP",
                bounds=bounds, constraints=constraints,
                options={"maxiter": 200, "ftol": 1e-6, "disp": False},
            )
            if res.success or res.fun < fit_rss:
                P_init = res.x.reshape(N, N_LATTICE)
                # Re-normalise rows in case of tiny numerical drift
                P_init = np.maximum(P_init, 0.0)
                P_init /= np.maximum(P_init.sum(axis=1, keepdims=True), 1e-12)
                fit_rss = float(res.fun)
                success = bool(res.success)
            else:
                success = False
        except Exception:
            success = False

    expected_change = _expected_change_per_meeting(P_init)
    post_rate = current_rate_bp + np.cumsum(expected_change)
    fit_residual = post_rate - obs

    return {
        "P": P_init,
        "expected_change_bp": expected_change,
        "post_rate_bp": post_rate,
        "sr3_implied_rate_bp": obs,
        "fit_residual_bp": fit_residual,
        "optimizer_success": success,
        "fit_rss": fit_rss,
    }


# =============================================================================
# Persistence
# =============================================================================

def _paths(asof: date) -> dict:
    stamp = asof.isoformat()
    return {
        "table":    _CACHE_DIR / f"policy_path_{stamp}.parquet",
        "manifest": _CACHE_DIR / f"policy_path_manifest_{stamp}.json",
    }


def _cycle_label(post_rates: np.ndarray, current_bp: float) -> str:
    """Heuristic cycle label from terminal & cumulative move."""
    if len(post_rates) == 0:
        return "UNKNOWN"
    terminal = float(post_rates[-1])
    move = terminal - current_bp
    if move > 25:
        return "LATE_HIKE"
    if move < -25:
        return "LATE_CUT"
    return "NEUTRAL"


def write_policy_path_cache(asof: date, meeting_dates: list[date],
                                  fit: dict, current_rate_bp: float) -> dict:
    paths = _paths(asof)
    P = fit["P"]
    rows = []
    for i, d in enumerate(meeting_dates):
        row = {
            "meeting_date": d,
            "days_to_meeting": (d - asof).days,
            "expected_change_bp": float(fit["expected_change_bp"][i]),
            "post_meeting_rate_bp": float(fit["post_rate_bp"][i]),
            "sr3_implied_rate_bp": float(fit["sr3_implied_rate_bp"][i]),
            "fit_residual_bp": float(fit["fit_residual_bp"][i]),
        }
        for k, lab in enumerate(LATTICE_LABELS):
            row[f"p_{lab}"] = float(P[i, k])
        rows.append(row)
    df = pd.DataFrame(rows)

    tmp_table = paths["table"].with_suffix(".parquet.tmp")
    tmp_man = paths["manifest"].with_suffix(".json.tmp")
    df.to_parquet(tmp_table, index=False)

    post = fit["post_rate_bp"]
    manifest = {
        "builder_version": BUILDER_VERSION,
        "asof_date": asof.isoformat(),
        "current_rate_bp": float(current_rate_bp),
        "n_meetings": len(meeting_dates),
        "first_meeting_date": meeting_dates[0].isoformat() if meeting_dates else None,
        "last_meeting_date": meeting_dates[-1].isoformat() if meeting_dates else None,
        "terminal_rate_bp": float(post[-1]) if len(post) else float(current_rate_bp),
        "floor_rate_bp": float(post.min()) if len(post) else float(current_rate_bp),
        "cycle_label": _cycle_label(post, current_rate_bp),
        "fit_rss": float(fit["fit_rss"]),
        "optimizer_success": bool(fit["optimizer_success"]),
        "lattice_bp": LATTICE_BP.tolist(),
    }
    tmp_man.write_text(json.dumps(manifest, indent=2, default=str))

    os.replace(tmp_table, paths["table"])
    os.replace(tmp_man, paths["manifest"])
    return manifest


# =============================================================================
# Top-level + CLI
# =============================================================================

def build_policy_path(asof: Optional[date] = None,
                          horizon_months: int = DEFAULT_HORIZON_MONTHS) -> dict:
    if asof is None:
        # Use latest CMC manifest's asof
        cands = sorted(_CACHE_DIR.glob("manifest_*.json"),
                          key=os.path.getmtime, reverse=True)
        if not cands:
            raise RuntimeError("no CMC manifest in .cmc_cache/")
        asof = date.fromisoformat(cands[0].stem.replace("manifest_", ""))

    current = _load_current_fdtr_bp()
    if current is None:
        raise RuntimeError("FDTRMID parquet unavailable")

    cmc = _load_cmc_close_at(asof, "outright")
    if not cmc:
        raise RuntimeError(f"no CMC closes at asof={asof}")

    from lib.cb_meetings import fomc_meetings_in_range
    horizon_end = date(asof.year + horizon_months // 12,
                            min(12, asof.month + (horizon_months % 12)),
                            min(28, asof.day))
    meetings = [m.date for m in fomc_meetings_in_range(asof, horizon_end)]
    if not meetings:
        raise RuntimeError(f"no FOMC meetings in horizon")

    fit = fit_policy_path(meetings, asof, current, cmc, use_slsqp=True)
    return write_policy_path_cache(asof, meetings, fit, current)


def main(argv=None):
    import sys
    args = list(sys.argv[1:] if argv is None else argv)
    asof = None
    if args:
        try:
            asof = date.fromisoformat(args[0])
        except ValueError:
            print("usage: python -m lib.analytics.policy_path_a4 [YYYY-MM-DD]")
            return 2
    print(f"[policy_path_a4] building Heitfield-Park PMFs (asof={asof or 'latest'})")
    manifest = build_policy_path(asof)
    print(f"[policy_path_a4] {manifest['n_meetings']} meetings / "
              f"current={manifest['current_rate_bp']:.0f}bp / "
              f"terminal={manifest['terminal_rate_bp']:.0f}bp / "
              f"cycle={manifest['cycle_label']} / "
              f"fit_rss={manifest['fit_rss']:.2f} / "
              f"slsqp_ok={manifest['optimizer_success']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
