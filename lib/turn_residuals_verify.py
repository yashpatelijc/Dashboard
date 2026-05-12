"""Turn-adjuster verification harness (Phase 3).

Run interactively:
    python -m lib.turn_residuals_verify [YYYY-MM-DD]

or programmatically:
    from lib.turn_residuals_verify import verify_all
    report = verify_all(asof_date=date(2026, 4, 27))

Returns a dict with one entry per check. Each entry is a dict with
``passed: bool``, ``message: str``, plus check-specific diagnostics.

Phase 3 checks (6 total):
    1. schema_integrity — both parquets exist with expected columns + dtypes.
    2. manifest_integrity — manifest JSON has all required fields, version matches.
    3. null_residual_orthogonality — re-regress residual_change on the same
       dummies; assert every dummy p-value > 0.5 (the verification gate from
       plan §12 Phase 3).
    4. mean_preservation — |mean(residual) − mean(raw)| < 0.001 bp per node.
    5. variance_reduction — var(residual) ≤ var(raw) × 1.01 per node.
    6. node_coverage — every node has ≥ 30 obs OR is recorded in
       manifest.missing_nodes.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from lib.analytics.turn_adjuster import (
    DUMMY_NAMES,
    BUILDER_VERSION,
    MIN_OBS_PER_NODE,
    _residuals_paths,
    build_calendar_dummies,
    fit_turn_adjustment,
)


# ---------- helper ----------

def _check_pass(passed: bool, message: str, **diagnostics) -> dict:
    return {"passed": bool(passed), "message": str(message), **diagnostics}


def _resolve_paths(asof_date: date) -> dict:
    return _residuals_paths(asof_date)


# ---------- individual checks ----------

EXPECTED_RESIDUAL_COLS = ["scope", "cmc_node", "bar_date",
                              "raw_change", "residual_change",
                              "fitted_change", "has_data"]

EXPECTED_DIAG_BASE_COLS = ["scope", "cmc_node", "n_obs", "dof", "r_squared",
                                  "raw_var", "residual_var", "var_reduction_pct",
                                  "low_sample_dummies"]
EXPECTED_DIAG_REG_COLS = (
    [f"beta_{r}" for r in ["intercept", *DUMMY_NAMES]]
    + [f"se_{r}" for r in ["intercept", *DUMMY_NAMES]]
    + [f"p_{r}" for r in ["intercept", *DUMMY_NAMES]]
    + [f"eff_n_{d}" for d in DUMMY_NAMES]
)


def check_schema_integrity(asof_date: date) -> dict:
    """Both parquets exist; expected columns present; dtype checks."""
    paths = _resolve_paths(asof_date)
    if not paths["residuals"].exists():
        return _check_pass(False, f"residuals parquet missing: {paths['residuals']}")
    if not paths["diagnostics"].exists():
        return _check_pass(False, f"diagnostics parquet missing: {paths['diagnostics']}")

    res = pd.read_parquet(paths["residuals"])
    diag = pd.read_parquet(paths["diagnostics"])

    missing_res = [c for c in EXPECTED_RESIDUAL_COLS if c not in res.columns]
    missing_diag = [c for c in EXPECTED_DIAG_BASE_COLS + EXPECTED_DIAG_REG_COLS
                          if c not in diag.columns]
    if missing_res:
        return _check_pass(False, f"residuals missing cols: {missing_res}",
                                 cols_found=res.columns.tolist())
    if missing_diag:
        return _check_pass(False, f"diagnostics missing cols: {missing_diag}",
                                 cols_found=diag.columns.tolist())

    # dtype sanity
    if not pd.api.types.is_bool_dtype(res["has_data"]):
        return _check_pass(False, f"residuals.has_data dtype != bool "
                                       f"(got {res['has_data'].dtype})")
    n_total = res["cmc_node"].nunique()
    return _check_pass(
        True,
        f"schema_integrity: residuals shape={res.shape} ({n_total} nodes), "
        f"diagnostics shape={diag.shape}",
        n_residual_rows=int(res.shape[0]),
        n_diagnostic_rows=int(diag.shape[0]),
        n_unique_nodes=int(n_total),
    )


def check_manifest_integrity(asof_date: date) -> dict:
    """Manifest JSON exists, parses, has required fields, builder_version matches."""
    paths = _resolve_paths(asof_date)
    if not paths["manifest"].exists():
        return _check_pass(False, f"manifest missing: {paths['manifest']}")
    try:
        manifest = json.loads(paths["manifest"].read_text())
    except json.JSONDecodeError as e:
        return _check_pass(False, f"manifest is not valid JSON: {e}")

    required_fields = {
        "builder_version", "asof_date", "cmc_asof_date", "history_start",
        "post_cutover_only", "product_code", "hac_lag",
        "n_nodes_total", "n_nodes_outright", "n_nodes_spread", "n_nodes_fly",
        "dummy_definitions", "dummy_fire_counts", "node_coverage",
        "missing_nodes",
    }
    missing = required_fields - set(manifest.keys())
    if missing:
        return _check_pass(False, f"manifest missing fields: {sorted(missing)}",
                                 fields_found=sorted(manifest.keys()))

    if manifest.get("builder_version") != BUILDER_VERSION:
        return _check_pass(False, f"builder_version mismatch: "
                                       f"manifest={manifest.get('builder_version')!r} "
                                       f"vs current={BUILDER_VERSION!r}")

    return _check_pass(
        True,
        f"manifest_integrity: builder={manifest['builder_version']} "
        f"hac_lag={manifest['hac_lag']} "
        f"n_nodes={manifest['n_nodes_total']} "
        f"history_start={manifest['history_start']}",
        manifest_summary={
            "n_nodes_total": manifest["n_nodes_total"],
            "n_nodes_outright": manifest["n_nodes_outright"],
            "n_nodes_spread": manifest["n_nodes_spread"],
            "n_nodes_fly": manifest["n_nodes_fly"],
            "dummy_fire_counts": manifest["dummy_fire_counts"],
        },
    )


def check_null_residual_orthogonality(asof_date: date,
                                            p_threshold: float = 0.5
                                            ) -> dict:
    """Re-regress residual_change on the same calendar dummies for each node;
    every dummy's two-sided HAC p-value MUST exceed ``p_threshold``.

    This is the verification gate from plan §12 Phase 3.
    """
    paths = _resolve_paths(asof_date)
    res = pd.read_parquet(paths["residuals"])

    failures: list = []
    n_checked = 0
    worst = (None, None, 1.0)   # (node, dummy, p_value)

    for node, sub in res.groupby("cmc_node", sort=True):
        sub = sub.sort_values("bar_date")
        idx = pd.to_datetime(sub["bar_date"])
        s = pd.Series(sub["residual_change"].astype(float).values, index=idx)
        if s.dropna().shape[0] < MIN_OBS_PER_NODE:
            continue
        dummies = build_calendar_dummies(idx)
        fit = fit_turn_adjustment(s, dummies, cmc_node=node)
        n_checked += 1
        for d in DUMMY_NAMES:
            p = fit.p_values.get(d, float("nan"))
            if not np.isfinite(p):
                continue
            if p < worst[2]:
                worst = (node, d, p)
            if p < p_threshold:
                failures.append((node, d, p))

    if failures:
        sample = failures[:10]
        return _check_pass(
            False,
            f"orthogonality FAIL: {len(failures)} (node, dummy) pairs have "
            f"p < {p_threshold}. Worst: {worst[0]}/{worst[1]} p={worst[2]:.4f}. "
            f"First 10 failures: "
            + ", ".join(f"{n}/{d}=p{p:.3f}" for n, d, p in sample),
            n_nodes_checked=n_checked,
            n_failures=len(failures),
            worst_p=float(worst[2]),
        )
    return _check_pass(
        True,
        f"orthogonality PASS: re-regressed {n_checked} nodes; "
        f"min dummy p-value = {worst[2]:.4f} (threshold {p_threshold}, "
        f"worst at {worst[0]}/{worst[1]})",
        n_nodes_checked=n_checked,
        worst_p=float(worst[2]),
    )


def check_mean_preservation(asof_date: date,
                                 tolerance_bp: float = 1e-3) -> dict:
    """|mean(residual_change) - mean(raw_change)| < ``tolerance_bp`` per node."""
    paths = _resolve_paths(asof_date)
    res = pd.read_parquet(paths["residuals"])
    failures: list = []
    deltas: list = []
    for node, sub in res.groupby("cmc_node", sort=True):
        df = sub.dropna(subset=["raw_change", "residual_change"])
        if df.empty:
            continue
        delta = float(df["residual_change"].mean() - df["raw_change"].mean())
        deltas.append((node, delta))
        if abs(delta) > tolerance_bp:
            failures.append((node, delta))

    max_abs = max((abs(d) for _, d in deltas), default=0.0)
    if failures:
        sample = failures[:10]
        return _check_pass(
            False,
            f"mean_preservation FAIL: {len(failures)} nodes with "
            f"|delta_mean| > {tolerance_bp} bp. "
            f"Worst |delta|={max_abs:.6f} bp. "
            f"First 10: " + ", ".join(f"{n}={d:+.4f}" for n, d in sample),
            n_failures=len(failures),
            max_abs_delta_bp=float(max_abs),
        )
    return _check_pass(
        True,
        f"mean_preservation PASS: max |delta_mean| = {max_abs:.2e} bp "
        f"(< {tolerance_bp} bp tolerance) across {len(deltas)} nodes",
        n_nodes_checked=len(deltas),
        max_abs_delta_bp=float(max_abs),
    )


def check_variance_reduction(asof_date: date,
                                  ratio_tolerance: float = 1.01) -> dict:
    """var(residual_change) ≤ var(raw_change) * ``ratio_tolerance`` per node.

    Removing a regressor should never INCREASE variance (modulo numerical
    noise). The 1% tolerance catches floating-point edge cases.
    """
    paths = _resolve_paths(asof_date)
    diag = pd.read_parquet(paths["diagnostics"])
    bad = diag[diag["residual_var"] > diag["raw_var"] * ratio_tolerance]
    if not bad.empty:
        sample = bad.head(10)[["cmc_node", "raw_var", "residual_var"]].to_dict("records")
        return _check_pass(
            False,
            f"variance_reduction FAIL: {len(bad)} nodes have "
            f"var(residual) > var(raw) * {ratio_tolerance}. "
            f"First 10: {sample}",
            n_failures=int(len(bad)),
        )
    var_red_min = float(diag["var_reduction_pct"].min())
    var_red_med = float(diag["var_reduction_pct"].median())
    var_red_max = float(diag["var_reduction_pct"].max())
    return _check_pass(
        True,
        f"variance_reduction PASS: across {len(diag)} nodes, "
        f"var_reduction_pct min={var_red_min:.3f}%, median={var_red_med:.3f}%, "
        f"max={var_red_max:.3f}%",
        n_nodes_checked=int(len(diag)),
        var_reduction_pct_min=var_red_min,
        var_reduction_pct_median=var_red_med,
        var_reduction_pct_max=var_red_max,
    )


def check_node_coverage(asof_date: date,
                            min_obs: int = MIN_OBS_PER_NODE) -> dict:
    """Every node has ≥ ``min_obs`` observations, OR is recorded in
    manifest.missing_nodes."""
    paths = _resolve_paths(asof_date)
    diag = pd.read_parquet(paths["diagnostics"])
    manifest = json.loads(paths["manifest"].read_text())
    declared_missing = set(manifest.get("missing_nodes", []))

    fitted_low = diag[diag["n_obs"] < min_obs]
    fitted_low_ids = (fitted_low["scope"] + "/" + fitted_low["cmc_node"]).tolist()
    silent_drops = [n for n in fitted_low_ids if n not in declared_missing]

    if silent_drops:
        return _check_pass(
            False,
            f"node_coverage FAIL: {len(silent_drops)} nodes have n_obs < "
            f"{min_obs} but are NOT declared in manifest.missing_nodes. "
            f"Silent: {silent_drops[:10]}",
            n_silent_drops=len(silent_drops),
        )
    return _check_pass(
        True,
        f"node_coverage PASS: all {len(diag)} fitted nodes have "
        f"n_obs >= {min_obs}; manifest declares {len(declared_missing)} "
        f"missing nodes.",
        n_nodes_fitted=int(len(diag)),
        n_nodes_declared_missing=len(declared_missing),
    )


# ---------- aggregator ----------

def verify_all(asof_date: date) -> dict:
    """Run every check and return a structured report. Mirrors
    :func:`lib.cmc_verify.verify_all`."""
    checks = {
        "schema_integrity":             check_schema_integrity(asof_date),
        "manifest_integrity":           check_manifest_integrity(asof_date),
        "null_residual_orthogonality":  check_null_residual_orthogonality(asof_date),
        "mean_preservation":            check_mean_preservation(asof_date),
        "variance_reduction":           check_variance_reduction(asof_date),
        "node_coverage":                check_node_coverage(asof_date),
    }
    n_passed = sum(1 for c in checks.values() if c["passed"])
    return {
        "asof_date": str(asof_date),
        "passed_all": n_passed == len(checks),
        "n_passed": n_passed,
        "n_total": len(checks),
        "checks": checks,
    }


# ---------- CLI ----------

def _resolve_latest_residuals_asof() -> date | None:
    cache = Path(__file__).resolve().parent.parent / ".cmc_cache"
    cands = sorted(cache.glob("turn_residuals_manifest_*.json"),
                       key=os.path.getmtime, reverse=True)
    if not cands:
        return None
    stem = cands[0].stem.replace("turn_residuals_manifest_", "")
    try:
        return date.fromisoformat(stem)
    except ValueError:
        return None


def _main(argv: list[str]) -> int:
    if len(argv) >= 2:
        try:
            asof = date.fromisoformat(argv[1])
        except ValueError:
            print(f"Bad date: {argv[1]}; expected YYYY-MM-DD", file=sys.stderr)
            return 2
    else:
        asof = _resolve_latest_residuals_asof()
        if asof is None:
            print("[turn_residuals_verify] no turn_residuals_manifest_*.json "
                    "in .cmc_cache/; build first via "
                    "`python -m lib.analytics.turn_adjuster`",
                    file=sys.stderr)
            return 2
    print(f"Verifying turn-adjuster outputs for asof = {asof}")
    print()
    report = verify_all(asof)
    for name, c in report["checks"].items():
        marker = "[PASS]" if c["passed"] else "[FAIL]"
        print(f"{marker} {name}: {c['message']}")
    print()
    print(f"Summary: {report['n_passed']}/{report['n_total']} checks passed.")
    return 0 if report["passed_all"] else 1


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
