"""signal_emit verification (Phase 7)."""
from __future__ import annotations

import json, os, sys
from datetime import date
from pathlib import Path
import pandas as pd

from lib.signal_emit import (
    BUILDER_VERSION, _paths, VALID_FAMILIES, VALID_SCOPES,
    VALID_TYPES, VALID_DIRECTIONS,
)


def _check_pass(passed: bool, message: str, **diag) -> dict:
    return {"passed": bool(passed), "message": str(message), **diag}


def check_schema(asof: date) -> dict:
    p = _paths(asof)
    if not p["table"].exists():
        return _check_pass(False, "signal_emit parquet missing")
    df = pd.read_parquet(p["table"])
    expected = {"emit_id", "asof_date", "detector_id", "detector_family",
                  "scope", "cmc_node", "signal_type", "direction", "raw_value",
                  "percentile_rank", "confidence_stoplight", "eff_n",
                  "regime_id", "regime_stability", "transition_flag",
                  "conflict_flag", "gate_quality", "expected_horizon_days",
                  "rationale", "sources", "tags", "created_at",
                  "builder_version", "version"}
    missing = expected - set(df.columns)
    if missing:
        return _check_pass(False, f"missing cols: {sorted(missing)}")
    return _check_pass(True, f"schema OK: 24 cols, {len(df)} rows",
                              n_rows=int(len(df)))


def check_manifest(asof: date) -> dict:
    p = _paths(asof)
    if not p["manifest"].exists():
        return _check_pass(False, "manifest missing")
    m = json.loads(p["manifest"].read_text())
    if m.get("builder_version") != BUILDER_VERSION:
        return _check_pass(False, "version mismatch")
    return _check_pass(
        True,
        f"manifest OK: {m['n_emissions']} emissions; clean={m['n_clean']} "
        f"low_sample={m['n_low_sample']} conflict={m['n_conflict']}",
        **m,
    )


def check_enum_validity(asof: date) -> dict:
    df = pd.read_parquet(_paths(asof)["table"])
    if df.empty:
        return _check_pass(True, "no rows to validate", skipped=True)
    bad_fam = df[~df["detector_family"].isin(VALID_FAMILIES)]
    bad_scope = df[~df["scope"].isin(VALID_SCOPES)]
    bad_type = df[~df["signal_type"].isin(VALID_TYPES)]
    bad_dir = df[~df["direction"].isin(VALID_DIRECTIONS)]
    if any(len(x) for x in (bad_fam, bad_scope, bad_type, bad_dir)):
        return _check_pass(False,
                              f"invalid enums: family={len(bad_fam)} scope={len(bad_scope)} "
                              f"type={len(bad_type)} direction={len(bad_dir)}")
    return _check_pass(True, "all enum values valid",
                              n_families=int(df["detector_family"].nunique()))


def check_score_bounds(asof: date) -> dict:
    df = pd.read_parquet(_paths(asof)["table"])
    if df.empty:
        return _check_pass(True, "no rows", skipped=True)
    bad_pct = df[(df["percentile_rank"] < 0) | (df["percentile_rank"] > 100)]
    bad_conf = df[(df["confidence_stoplight"] < 0) | (df["confidence_stoplight"] > 100)]
    if not bad_pct.empty or not bad_conf.empty:
        return _check_pass(False, f"out-of-bounds: pct={len(bad_pct)} conf={len(bad_conf)}")
    return _check_pass(True, "all scores in [0, 100]")


def check_gate_quality_consistency(asof: date) -> dict:
    """gate_quality should be CLEAN/LOW_SAMPLE/TRANSITION/CONFLICT and
    consistent with the corresponding flags."""
    df = pd.read_parquet(_paths(asof)["table"])
    if df.empty:
        return _check_pass(True, "no rows", skipped=True)
    bad_conflict = df[(df["gate_quality"] != "CONFLICT") & (df["conflict_flag"])]
    bad_trans = df[(df["gate_quality"] != "TRANSITION") & (df["transition_flag"])]
    if not bad_conflict.empty or not bad_trans.empty:
        return _check_pass(
            False,
            f"flag-vs-gate inconsistency: conflict_mismatch={len(bad_conflict)} "
            f"transition_mismatch={len(bad_trans)}",
        )
    return _check_pass(True,
                              f"gate_quality consistent with conflict/transition flags",
                              n_clean=int((df["gate_quality"]=="CLEAN").sum()))


def check_regime_id_sanity(asof: date) -> dict:
    """All emissions should have regime_id in [-1, 5] (K=6 from Phase 4)."""
    df = pd.read_parquet(_paths(asof)["table"])
    if df.empty:
        return _check_pass(True, "no rows", skipped=True)
    bad = df[(df["regime_id"] < -1) | (df["regime_id"] > 5)]
    if not bad.empty:
        return _check_pass(False, f"{len(bad)} rows with regime_id outside [-1, 5]")
    return _check_pass(True, f"regime_id in [-1, 5] for all {len(df)} rows")


def verify_all(asof: date) -> dict:
    checks = {
        "schema_integrity":          check_schema(asof),
        "manifest_integrity":        check_manifest(asof),
        "enum_validity":             check_enum_validity(asof),
        "score_bounds":              check_score_bounds(asof),
        "gate_quality_consistency":  check_gate_quality_consistency(asof),
        "regime_id_sanity":          check_regime_id_sanity(asof),
    }
    n_passed = sum(1 for c in checks.values() if c["passed"])
    return {"asof_date": str(asof), "passed_all": n_passed == len(checks),
              "n_passed": n_passed, "n_total": len(checks), "checks": checks}


def _resolve_latest_asof() -> date | None:
    cache = Path(__file__).resolve().parent.parent / ".signal_cache"
    cands = sorted(cache.glob("signal_emit_manifest_*.json"),
                       key=os.path.getmtime, reverse=True)
    if not cands:
        return None
    try:
        return date.fromisoformat(cands[0].stem.replace("signal_emit_manifest_", ""))
    except ValueError:
        return None


def _main(argv: list[str]) -> int:
    asof = (date.fromisoformat(argv[1]) if len(argv) >= 2
                else _resolve_latest_asof())
    if asof is None:
        print("[signal_emit_verify] no signal_emit cache", file=sys.stderr)
        return 2
    print(f"Verifying signal_emit for asof = {asof}\n")
    rep = verify_all(asof)
    for n, c in rep["checks"].items():
        print(f"{'[PASS]' if c['passed'] else '[FAIL]'} {n}: {c['message']}")
    print(f"\nSummary: {rep['n_passed']}/{rep['n_total']} checks passed.")
    return 0 if rep["passed_all"] else 1


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
