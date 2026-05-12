"""Phase 9 opportunity-modules verification."""
from __future__ import annotations

import json, os, sys
from datetime import date
from pathlib import Path
import pandas as pd

from lib.analytics.opportunity_modules import _paths, BUILDER_VERSION


def _check_pass(passed: bool, message: str, **diag) -> dict:
    return {"passed": bool(passed), "message": str(message), **diag}


def check_manifest(asof: date) -> dict:
    p = _paths(asof)
    if not p["manifest"].exists():
        return _check_pass(False, "opp manifest missing")
    m = json.loads(p["manifest"].read_text())
    if m.get("builder_version") != BUILDER_VERSION:
        return _check_pass(False, "version mismatch")
    return _check_pass(True,
                              f"manifest OK: built={m['modules_built']}, "
                              f"stubbed={m['modules_stubbed']}")


def check_a6_outputs(asof: date) -> dict:
    p = _paths(asof)
    if not p["a6"].exists():
        return _check_pass(False, "a6 parquet missing")
    df = pd.read_parquet(p["a6"])
    if df.empty:
        return _check_pass(False, "a6 empty")
    n_passes = int(df["passes_gate"].sum())
    return _check_pass(True,
                              f"A6 OU fitted on {len(df)} nodes; "
                              f"{n_passes} pass HL in [3d,60d] AND ADF<0.05 gate",
                              n_pass=n_passes, n_total=int(len(df)))


def check_a4m_outputs(asof: date) -> dict:
    p = _paths(asof)
    if not p["a4m"].exists():
        return _check_pass(False, "a4m parquet missing")
    df = pd.read_parquet(p["a4m"])
    if df.empty:
        return _check_pass(False, "a4m empty")
    return _check_pass(True,
                              f"A4m TSM rows = {len(df)} "
                              f"(3 PCs × {df['lookback_bars'].nunique()} lookbacks); "
                              f"max |z| = {df['tsm_zscore'].abs().max():.2f}")


def check_a1c_outputs(asof: date) -> dict:
    p = _paths(asof)
    if not p["a1c"].exists():
        return _check_pass(False, "a1c parquet missing")
    df = pd.read_parquet(p["a1c"])
    return _check_pass(True,
                              f"A1c carry decomposition on {len(df)} nodes; "
                              f"mean carry = {df['carry'].mean():.3f}")


def check_a12d_outputs(asof: date) -> dict:
    p = _paths(asof)
    if not p["a12d"].exists():
        return _check_pass(False, "a12d parquet missing")
    df = pd.read_parquet(p["a12d"])
    if df.empty:
        return _check_pass(False, "a12d empty")
    return _check_pass(True,
                              f"A12d event-drift rows = {len(df)} "
                              f"({df['ticker'].nunique()} tickers × {df['window'].nunique()} windows); "
                              f"max |median_drift| = {df['median_drift_bp'].abs().max():.2f} bp")


def check_a9_outputs(asof: date) -> dict:
    p = _paths(asof)
    if not p["a9"].exists():
        return _check_pass(False, "a9 parquet missing")
    df = pd.read_parquet(p["a9"])
    if df.empty:
        return _check_pass(False, "a9 empty")
    n_signals = int(df["transition_signal"].sum())
    return _check_pass(True,
                              f"A9 transitions tracked across {len(df)} bars; "
                              f"{n_signals} bars currently flagged transitioning",
                              n_signals=n_signals)


def verify_all(asof: date) -> dict:
    checks = {
        "manifest_integrity": check_manifest(asof),
        "a6_ou":              check_a6_outputs(asof),
        "a4m_tsm":            check_a4m_outputs(asof),
        "a1c_carry":          check_a1c_outputs(asof),
        "a12d_event_drift":   check_a12d_outputs(asof),
        "a9_regime_transitions": check_a9_outputs(asof),
    }
    n_passed = sum(1 for c in checks.values() if c["passed"])
    return {"asof_date": str(asof), "passed_all": n_passed == len(checks),
              "n_passed": n_passed, "n_total": len(checks), "checks": checks}


def _resolve_latest_asof() -> date | None:
    cache = Path(__file__).resolve().parent.parent / ".cmc_cache"
    cands = sorted(cache.glob("opp_manifest_*.json"),
                       key=os.path.getmtime, reverse=True)
    if not cands:
        return None
    try:
        return date.fromisoformat(cands[0].stem.replace("opp_manifest_", ""))
    except ValueError:
        return None


def _main(argv):
    asof = (date.fromisoformat(argv[1]) if len(argv) >= 2
                else _resolve_latest_asof())
    if asof is None:
        print("[opportunity_verify] no opportunity cache", file=sys.stderr)
        return 2
    print(f"Verifying opportunity modules for asof = {asof}\n")
    rep = verify_all(asof)
    for n, c in rep["checks"].items():
        print(f"{'[PASS]' if c['passed'] else '[FAIL]'} {n}: {c['message']}")
    print(f"\nSummary: {rep['n_passed']}/{rep['n_total']} checks passed.")
    return 0 if rep["passed_all"] else 1


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
