"""A11 event-impact verification harness (Phase 6).

CLI: ``python -m lib.event_impact_verify [YYYY-MM-DD]``
"""
from __future__ import annotations

import json, os, sys
from datetime import date
from pathlib import Path
import pandas as pd

from lib.analytics.event_impact_a11 import BUILDER_VERSION, _paths


def _check_pass(passed: bool, message: str, **diag) -> dict:
    return {"passed": bool(passed), "message": str(message), **diag}


def check_schema(asof: date) -> dict:
    p = _paths(asof)
    if not p["table"].exists():
        return _check_pass(False, "event_impact parquet missing")
    df = pd.read_parquet(p["table"])
    expected = {"ticker", "axis", "cmc_node", "regime", "n_obs",
                  "beta", "se", "r_squared", "hit_rate",
                  "score_full", "score_recency_weighted",
                  "becoming_more_important", "scope"}
    missing = expected - set(df.columns)
    if missing:
        return _check_pass(False, f"missing cols: {sorted(missing)}")
    return _check_pass(True, f"schema OK: {df.shape}",
                              n_tickers=int(df["ticker"].nunique()))


def check_manifest(asof: date) -> dict:
    p = _paths(asof)
    if not p["manifest"].exists():
        return _check_pass(False, "manifest missing")
    m = json.loads(p["manifest"].read_text())
    req = {"builder_version", "asof_date", "n_rows", "n_tickers_built",
              "axes_used", "halflife_days", "n_becoming_more_important"}
    missing = req - set(m.keys())
    if missing:
        return _check_pass(False, f"manifest missing fields: {sorted(missing)}")
    if m["builder_version"] != BUILDER_VERSION:
        return _check_pass(False, "version mismatch")
    return _check_pass(True,
                              f"manifest OK: {m['n_tickers_built']} tickers / "
                              f"{m['n_rows']} rows / axes={m['axes_used']}",
                              **{k: m[k] for k in ["n_tickers_built","n_rows","axes_used",
                                                          "n_becoming_more_important"]})


def check_score_bounds(asof: date) -> dict:
    df = pd.read_parquet(_paths(asof)["table"])
    bad_full = df[df["score_full"].between(0, 2.0) == False]
    bad_rec = df.loc[df["score_recency_weighted"].notna() &
                          ~df["score_recency_weighted"].between(0, 2.0)]
    if len(bad_full) > 0 or len(bad_rec) > 0:
        return _check_pass(False, f"out-of-bounds scores: full={len(bad_full)}, "
                                       f"recency={len(bad_rec)}")
    return _check_pass(True,
                              f"all scores in [0, 2.0]: max_full={df['score_full'].max():.3f}",
                              max_score_full=float(df["score_full"].max()))


def check_r_squared_bounds(asof: date) -> dict:
    df = pd.read_parquet(_paths(asof)["table"])
    bad = df[(df["r_squared"] < 0) | (df["r_squared"] > 1)]
    if not bad.empty:
        return _check_pass(False, f"{len(bad)} rows with R² outside [0, 1]")
    return _check_pass(True, f"R² in [0, 1] for all {len(df)} rows",
                              max_r2=float(df["r_squared"].max()))


def check_hit_rate_bounds(asof: date) -> dict:
    df = pd.read_parquet(_paths(asof)["table"])
    bad = df[(df["hit_rate"] < 0) | (df["hit_rate"] > 1)]
    if not bad.empty:
        return _check_pass(False, f"{len(bad)} rows with hit_rate outside [0, 1]")
    return _check_pass(True,
                              f"hit_rate in [0, 1] for all {len(df)} rows; "
                              f"mean = {df['hit_rate'].mean():.3f}",
                              mean_hit_rate=float(df["hit_rate"].mean()))


def check_top_signals_present(asof: date) -> dict:
    """At least one ticker × node should have score_full > 0.20 (otherwise
    something is broken in the regression)."""
    df = pd.read_parquet(_paths(asof)["table"])
    full = df[df["scope"] == "full"]
    top = full[full["score_full"] > 0.20]
    if top.empty:
        return _check_pass(False, "no rows with score_full > 0.20 — degenerate")
    return _check_pass(True,
                              f"{len(top)} rows have score_full > 0.20; "
                              f"top: {top.nlargest(1, 'score_full').iloc[0]['ticker']}/"
                              f"{top.nlargest(1, 'score_full').iloc[0]['cmc_node']} "
                              f"({top['score_full'].max():.3f})",
                              n_top_signals=int(len(top)))


def verify_all(asof: date) -> dict:
    checks = {
        "schema_integrity":   check_schema(asof),
        "manifest_integrity": check_manifest(asof),
        "score_bounds":       check_score_bounds(asof),
        "r_squared_bounds":   check_r_squared_bounds(asof),
        "hit_rate_bounds":    check_hit_rate_bounds(asof),
        "top_signals_present": check_top_signals_present(asof),
    }
    n_passed = sum(1 for c in checks.values() if c["passed"])
    return {"asof_date": str(asof), "passed_all": n_passed == len(checks),
              "n_passed": n_passed, "n_total": len(checks), "checks": checks}


def _resolve_latest_asof() -> date | None:
    cache = Path(__file__).resolve().parent.parent / ".cmc_cache"
    cands = sorted(cache.glob("event_impact_manifest_*.json"),
                       key=os.path.getmtime, reverse=True)
    if not cands:
        return None
    try:
        return date.fromisoformat(cands[0].stem.replace("event_impact_manifest_", ""))
    except ValueError:
        return None


def _main(argv: list[str]) -> int:
    asof = (date.fromisoformat(argv[1]) if len(argv) >= 2
                else _resolve_latest_asof())
    if asof is None:
        print("[event_impact_verify] no event_impact cache", file=sys.stderr)
        return 2
    print(f"Verifying event_impact for asof = {asof}\n")
    rep = verify_all(asof)
    for n, c in rep["checks"].items():
        print(f"{'[PASS]' if c['passed'] else '[FAIL]'} {n}: {c['message']}")
    print(f"\nSummary: {rep['n_passed']}/{rep['n_total']} checks passed.")
    return 0 if rep["passed_all"] else 1


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
