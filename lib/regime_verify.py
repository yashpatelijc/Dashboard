"""Regime classifier verification harness (Phase 4).

Run interactively: ``python -m lib.regime_verify [YYYY-MM-DD]``

KPIs from plan §12 Phase 4:
  1. dominant posterior > 0.6 on >= 70% of days
  2. mean run-length >= 10 days
  3. < 5% of days flip after refit (Hungarian-relabel KPI; N/A on first fit)

Plus structural checks: schema integrity, manifest integrity, n_active_states.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from lib.analytics.regime_a1 import (
    BUILDER_VERSION, _regime_paths,
    KPI_DOMINANT_POST, KPI_DOMINANT_FRAC,
    KPI_MEAN_RUN_LENGTH, KPI_REFIT_FLIP_FRAC,
)


def _check_pass(passed: bool, message: str, **diagnostics) -> dict:
    return {"passed": bool(passed), "message": str(message), **diagnostics}


def check_schema_integrity(asof: date) -> dict:
    paths = _regime_paths(asof)
    if not paths["states"].exists():
        return _check_pass(False, f"regime_states parquet missing: {paths['states']}")
    if not paths["diagnostics"].exists():
        return _check_pass(False, f"regime_diagnostics parquet missing: {paths['diagnostics']}")
    states = pd.read_parquet(paths["states"])
    diag = pd.read_parquet(paths["diagnostics"])
    expected_cols = {"bar_date", "state_id", "state_name", "top_state_posterior"}
    missing = expected_cols - set(states.columns)
    if missing:
        return _check_pass(False, f"states missing cols: {sorted(missing)}")
    diag_expected = {"state_id", "state_name", "n_bars", "frac_bars",
                          "mean_run_length", "max_run_length"}
    missing_diag = diag_expected - set(diag.columns)
    if missing_diag:
        return _check_pass(False, f"diagnostics missing cols: {sorted(missing_diag)}")
    return _check_pass(True,
                              f"schema OK: states {states.shape}, diagnostics {diag.shape}",
                              n_states=int(diag.shape[0]))


def check_manifest_integrity(asof: date) -> dict:
    paths = _regime_paths(asof)
    if not paths["manifest"].exists():
        return _check_pass(False, f"manifest missing: {paths['manifest']}")
    manifest = json.loads(paths["manifest"].read_text())
    required = {"builder_version", "asof_date", "scope", "n_bars", "K",
                  "pca_components", "pca_explained_variance_ratio",
                  "state_names", "state_centroids", "transition_matrix"}
    missing = required - set(manifest.keys())
    if missing:
        return _check_pass(False, f"manifest missing fields: {sorted(missing)}")
    if manifest.get("builder_version") != BUILDER_VERSION:
        return _check_pass(False, f"builder_version mismatch")
    return _check_pass(
        True,
        f"manifest OK: K={manifest['K']} / n_bars={manifest['n_bars']} / "
        f"BIC={manifest.get('gmm_bic', '?')} / "
        f"PCA exp_var={[f'{v:.1%}' for v in manifest['pca_explained_variance_ratio']]}",
    )


def check_kpi_dominant_posterior(asof: date) -> dict:
    """KPI 1: posterior dominant state > 0.6 on >= 70% of days."""
    paths = _regime_paths(asof)
    states = pd.read_parquet(paths["states"])
    n = len(states)
    if n == 0:
        return _check_pass(False, "no states rows")
    dominant_n = int((states["top_state_posterior"] > KPI_DOMINANT_POST).sum())
    frac = dominant_n / n
    return _check_pass(
        frac >= KPI_DOMINANT_FRAC,
        f"KPI1 dominant_post>{KPI_DOMINANT_POST}: {dominant_n}/{n} = {frac:.1%} "
        f"(target >= {KPI_DOMINANT_FRAC:.0%})",
        dominant_frac=float(frac),
    )


def check_kpi_run_length(asof: date) -> dict:
    """KPI 2: mean run-length >= 10 days (averaged over ACTIVE states)."""
    paths = _regime_paths(asof)
    diag = pd.read_parquet(paths["diagnostics"])
    active = diag[diag["n_bars"] > 0]
    if active.empty:
        return _check_pass(False, "no active states")
    mean_run = float(active["mean_run_length"].mean())
    return _check_pass(
        mean_run >= KPI_MEAN_RUN_LENGTH,
        f"KPI2 mean(mean_run_length over active states) = {mean_run:.2f} "
        f"(target >= {KPI_MEAN_RUN_LENGTH})",
        mean_run_length=mean_run,
        n_active_states=int(len(active)),
    )


def check_kpi_refit_stability(asof: date) -> dict:
    """KPI 3: < 5% of days flip after refit (Hungarian-relabel KPI).

    Compares this fit against the most recent prior fit (asof < current).
    On the FIRST fit (no prior), reports SKIPPED.
    """
    paths = _regime_paths(asof)
    cache_dir = paths["states"].parent
    cur_states = pd.read_parquet(paths["states"])
    cur_states["bar_date"] = pd.to_datetime(cur_states["bar_date"]).dt.date

    prior_cands = sorted(cache_dir.glob("regime_states_*.parquet"),
                              key=os.path.getmtime, reverse=True)
    prior_path = None
    for p in prior_cands:
        try:
            d = date.fromisoformat(p.stem.replace("regime_states_", ""))
        except ValueError:
            continue
        if d < asof:
            prior_path = p
            break
    if prior_path is None:
        return _check_pass(True, "KPI3 SKIPPED: no prior fit to compare against",
                                 skipped=True)

    prior = pd.read_parquet(prior_path)
    prior["bar_date"] = pd.to_datetime(prior["bar_date"]).dt.date
    merged = pd.merge(
        cur_states[["bar_date", "state_id"]],
        prior[["bar_date", "state_id"]],
        on="bar_date", suffixes=("_cur", "_prior"),
    )
    if merged.empty:
        return _check_pass(True, "KPI3 SKIPPED: no overlapping dates with prior",
                                 skipped=True)
    flip_n = int((merged["state_id_cur"] != merged["state_id_prior"]).sum())
    flip_frac = flip_n / len(merged)
    return _check_pass(
        flip_frac < KPI_REFIT_FLIP_FRAC,
        f"KPI3 refit flips: {flip_n}/{len(merged)} = {flip_frac:.2%} "
        f"(threshold < {KPI_REFIT_FLIP_FRAC:.0%})",
        flip_frac=float(flip_frac),
        n_overlapping_dates=int(len(merged)),
    )


def check_active_states(asof: date, min_active: int = 2) -> dict:
    """Soft check: at least N states have observations.

    K=6 is over-parameterized for short post-cutover samples. As long as
    >= 2 states are active the classifier is doing useful work; below 2
    the model is collapsed and unusable.
    """
    paths = _regime_paths(asof)
    diag = pd.read_parquet(paths["diagnostics"])
    n_active = int((diag["n_bars"] > 0).sum())
    K = int(diag.shape[0])
    return _check_pass(
        n_active >= min_active,
        f"n_active_states = {n_active}/{K} (threshold >= {min_active})",
        n_active=n_active, K=K,
    )


def verify_all(asof: date) -> dict:
    checks = {
        "schema_integrity":        check_schema_integrity(asof),
        "manifest_integrity":      check_manifest_integrity(asof),
        "kpi1_dominant_posterior": check_kpi_dominant_posterior(asof),
        "kpi2_run_length":         check_kpi_run_length(asof),
        "kpi3_refit_stability":    check_kpi_refit_stability(asof),
        "n_active_states":         check_active_states(asof),
    }
    n_passed = sum(1 for c in checks.values() if c["passed"])
    return {
        "asof_date": str(asof),
        "passed_all": n_passed == len(checks),
        "n_passed": n_passed,
        "n_total": len(checks),
        "checks": checks,
    }


def _resolve_latest_regime_asof() -> date | None:
    cache = Path(__file__).resolve().parent.parent / ".cmc_cache"
    cands = sorted(cache.glob("regime_manifest_*.json"),
                       key=os.path.getmtime, reverse=True)
    if not cands:
        return None
    try:
        return date.fromisoformat(cands[0].stem.replace("regime_manifest_", ""))
    except ValueError:
        return None


def _main(argv: list[str]) -> int:
    if len(argv) >= 2:
        asof = date.fromisoformat(argv[1])
    else:
        asof = _resolve_latest_regime_asof()
        if asof is None:
            print("[regime_verify] no regime cache; build first", file=sys.stderr)
            return 2
    print(f"Verifying regime classifier for asof = {asof}\n")
    report = verify_all(asof)
    for name, c in report["checks"].items():
        marker = "[PASS]" if c["passed"] else "[FAIL]"
        print(f"{marker} {name}: {c['message']}")
    print(f"\nSummary: {report['n_passed']}/{report['n_total']} checks passed.")
    return 0 if report["passed_all"] else 1


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
