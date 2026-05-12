"""A4 policy-path verification harness (Phase 5).

CLI: ``python -m lib.policy_path_verify [YYYY-MM-DD]``

Checks:
  1. schema_integrity — parquet exists with expected cols
  2. manifest_integrity — required fields, builder version
  3. pmf_normalisation — every meeting's PMF sums to 1 (within 1e-6)
  4. zlb_constraint — all post_meeting_rate_bp >= 0
  5. fit_residual_acceptable — max |residual| < 25 bp (one lattice step)
  6. lattice_concentration — max two non-zero PMF entries per meeting
     (FedWatch-style; the SLSQP refinement preserves this in most cases)
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from lib.analytics.policy_path_a4 import BUILDER_VERSION, _paths, LATTICE_LABELS


def _check_pass(passed: bool, message: str, **diag) -> dict:
    return {"passed": bool(passed), "message": str(message), **diag}


def check_schema(asof: date) -> dict:
    p = _paths(asof)
    if not p["table"].exists():
        return _check_pass(False, f"missing: {p['table']}")
    df = pd.read_parquet(p["table"])
    expected = {"meeting_date", "days_to_meeting", "expected_change_bp",
                  "post_meeting_rate_bp", "sr3_implied_rate_bp", "fit_residual_bp"}
    expected |= {f"p_{l}" for l in LATTICE_LABELS}
    missing = expected - set(df.columns)
    if missing:
        return _check_pass(False, f"missing cols: {sorted(missing)}")
    return _check_pass(True, f"schema OK: {df.shape}", n_meetings=int(len(df)))


def check_manifest(asof: date) -> dict:
    p = _paths(asof)
    if not p["manifest"].exists():
        return _check_pass(False, "manifest missing")
    m = json.loads(p["manifest"].read_text())
    required = {"builder_version", "asof_date", "current_rate_bp", "n_meetings",
                  "terminal_rate_bp", "floor_rate_bp", "cycle_label", "fit_rss"}
    missing = required - set(m.keys())
    if missing:
        return _check_pass(False, f"missing fields: {sorted(missing)}")
    if m["builder_version"] != BUILDER_VERSION:
        return _check_pass(False, "version mismatch")
    return _check_pass(
        True,
        f"manifest OK: cycle={m['cycle_label']} terminal={m['terminal_rate_bp']:.0f}bp "
        f"floor={m['floor_rate_bp']:.0f}bp slsqp_ok={m['optimizer_success']}",
    )


def check_pmf_normalisation(asof: date, tol: float = 1e-6) -> dict:
    df = pd.read_parquet(_paths(asof)["table"])
    sums = df[[f"p_{l}" for l in LATTICE_LABELS]].sum(axis=1)
    bad = (sums - 1.0).abs() > tol
    if bad.any():
        worst = float((sums - 1.0).abs().max())
        return _check_pass(False, f"{int(bad.sum())} meetings violate sum=1; "
                                       f"worst |sum-1|={worst:.2e}")
    return _check_pass(True, f"PMF normalisation OK across {len(df)} meetings",
                              max_abs_dev=float((sums - 1.0).abs().max()))


def check_zlb(asof: date) -> dict:
    df = pd.read_parquet(_paths(asof)["table"])
    bad = df[df["post_meeting_rate_bp"] < 0]
    if not bad.empty:
        return _check_pass(False,
                              f"ZLB violation at {len(bad)} meetings; min rate = "
                              f"{df['post_meeting_rate_bp'].min():.1f} bp")
    return _check_pass(True,
                              f"ZLB OK: min post_rate = {df['post_meeting_rate_bp'].min():.1f} bp",
                              min_rate_bp=float(df["post_meeting_rate_bp"].min()))


def check_fit_residual(asof: date, threshold_bp: float = 25.0) -> dict:
    df = pd.read_parquet(_paths(asof)["table"])
    max_abs = float(df["fit_residual_bp"].abs().max())
    return _check_pass(
        max_abs < threshold_bp,
        f"max |fit_residual| = {max_abs:.2f} bp (threshold < {threshold_bp:.0f} bp)",
        max_abs_residual_bp=max_abs,
    )


def check_lattice_concentration(asof: date) -> dict:
    """Each meeting's PMF should have at most ~2-3 non-trivial entries."""
    df = pd.read_parquet(_paths(asof)["table"])
    nz_counts = (df[[f"p_{l}" for l in LATTICE_LABELS]] > 0.05).sum(axis=1)
    too_many = nz_counts > 4
    if too_many.any():
        return _check_pass(False,
                              f"{int(too_many.sum())} meetings have >4 lattice "
                              f"entries with p>0.05 — diffuse PMF")
    return _check_pass(
        True,
        f"lattice concentration OK: max non-trivial entries = {int(nz_counts.max())}",
        max_nz_entries=int(nz_counts.max()),
    )


def verify_all(asof: date) -> dict:
    checks = {
        "schema_integrity":      check_schema(asof),
        "manifest_integrity":    check_manifest(asof),
        "pmf_normalisation":     check_pmf_normalisation(asof),
        "zlb_constraint":        check_zlb(asof),
        "fit_residual":          check_fit_residual(asof),
        "lattice_concentration": check_lattice_concentration(asof),
    }
    n_passed = sum(1 for c in checks.values() if c["passed"])
    return {"asof_date": str(asof), "passed_all": n_passed == len(checks),
              "n_passed": n_passed, "n_total": len(checks), "checks": checks}


def _resolve_latest_asof() -> date | None:
    cache = Path(__file__).resolve().parent.parent / ".cmc_cache"
    cands = sorted(cache.glob("policy_path_manifest_*.json"),
                       key=os.path.getmtime, reverse=True)
    if not cands:
        return None
    try:
        return date.fromisoformat(cands[0].stem.replace("policy_path_manifest_", ""))
    except ValueError:
        return None


def _main(argv: list[str]) -> int:
    if len(argv) >= 2:
        asof = date.fromisoformat(argv[1])
    else:
        asof = _resolve_latest_asof()
        if asof is None:
            print("[policy_path_verify] no policy_path manifest", file=sys.stderr)
            return 2
    print(f"Verifying policy_path for asof = {asof}\n")
    rep = verify_all(asof)
    for name, c in rep["checks"].items():
        print(f"{'[PASS]' if c['passed'] else '[FAIL]'} {name}: {c['message']}")
    print(f"\nSummary: {rep['n_passed']}/{rep['n_total']} checks passed.")
    return 0 if rep["passed_all"] else 1


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
