"""Historical Event Impact verify (10 checks per spec §11)."""
from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from lib.historical_event_impact import (
    BUILDER_VERSION, EVENT_CUTOFF_DATE, TICKERS_13,
    STATES_8, MEASUREMENTS, WINDOWS, _paths, list_instruments,
    LOW_SAMPLE_MIN_OBS,
)


def _check_pass(passed: bool, message: str, **diag) -> dict:
    return {"passed": bool(passed), "message": str(message), **diag}


def _resolve_latest_asof() -> date | None:
    cache = Path(__file__).resolve().parent.parent / ".cmc_cache"
    cands = sorted(cache.glob("hei_manifest_*.json"),
                       key=os.path.getmtime, reverse=True)
    if not cands:
        return None
    try:
        return date.fromisoformat(cands[0].stem.replace("hei_manifest_", ""))
    except ValueError:
        return None


# =============================================================================
# 10 verify checks
# =============================================================================

def check_catalog_schema_integrity(asof: date) -> dict:
    p = _paths(asof)
    if not p["catalog"].exists():
        return _check_pass(False, "catalog parquet missing")
    df = pd.read_parquet(p["catalog"])
    expected = {"ticker", "release_date", "release_datetime_utc",
                  "prior_actual", "consensus", "actual",
                  "expected_change", "expected_z", "expected_direction",
                  "surprise", "surprise_z", "surprise_sign", "surprise_size",
                  "state_8", "is_post_cutoff"}
    missing = expected - set(df.columns)
    if missing:
        return _check_pass(False, f"missing cols: {sorted(missing)}")
    return _check_pass(True, f"schema OK: {df.shape}",
                              n_events=int(len(df)),
                              n_tickers=int(df["ticker"].nunique()))


def check_catalog_post_cutoff_filter(asof: date) -> dict:
    df = pd.read_parquet(_paths(asof)["catalog"])
    df["release_date"] = pd.to_datetime(df["release_date"]).dt.date
    bad = df[df["release_date"] < EVENT_CUTOFF_DATE]
    if len(bad) > 0:
        return _check_pass(False,
                              f"{len(bad)} events before cutoff {EVENT_CUTOFF_DATE}")
    return _check_pass(True,
                              f"all {len(df)} events on/after {EVENT_CUTOFF_DATE}",
                              cutoff=str(EVENT_CUTOFF_DATE))


def check_state_8_assignment_exhaustive(asof: date) -> dict:
    df = pd.read_parquet(_paths(asof)["catalog"])
    valid = set(STATES_8) | {"FLAT"}
    bad = df[~df["state_8"].isin(valid)]
    if len(bad) > 0:
        return _check_pass(False,
                              f"{len(bad)} events with invalid state_8 value: "
                              f"{sorted(bad['state_8'].unique())[:5]}")
    n_flat = int((df["state_8"] == "FLAT").sum())
    return _check_pass(
        True,
        f"all {len(df)} events validly classified; FLAT count = {n_flat}",
        states_used=sorted(df["state_8"].unique()),
    )


def check_ticker_coverage_sufficient(asof: date) -> dict:
    df = pd.read_parquet(_paths(asof)["catalog"])
    counts = df.groupby("ticker").size()
    sufficient = counts[counts >= 15]
    if len(sufficient) < 10:
        return _check_pass(
            False,
            f"only {len(sufficient)} of 13 tickers have >=15 events "
            f"(threshold for sub-tab usefulness)",
            ticker_counts=counts.to_dict(),
        )
    return _check_pass(
        True,
        f"{len(sufficient)} of 13 tickers have >=15 events post-cutoff",
        ticker_counts=counts.to_dict(),
    )


def check_instrument_panel_completeness(asof: date) -> dict:
    p = _paths(asof)
    if not p["panel"].exists():
        return _check_pass(False, "panel parquet missing")
    panel = pd.read_parquet(p["panel"])
    expected_n_instruments = len(list_instruments())
    if expected_n_instruments == 0:
        return _check_pass(False, "list_instruments() returned 0")
    # Per event, count instrument rows
    counts = panel.groupby(["ticker", "release_date"]).size()
    # Some events may be missing some instruments (e.g. deep gold pack at very
    # early events with thin coverage). Tolerance: >=75% of instruments per event.
    min_per_event = int(0.75 * expected_n_instruments)
    bad = counts[counts < min_per_event]
    if len(bad) > 0:
        return _check_pass(
            False,
            f"{len(bad)} events have <{min_per_event}/{expected_n_instruments} "
            f"instruments populated",
            worst_event=bad.idxmin(),
        )
    return _check_pass(
        True,
        f"all {len(counts)} events have >={min_per_event}/"
        f"{expected_n_instruments} instruments populated",
        n_events=int(len(counts)),
        instruments_expected=expected_n_instruments,
    )


def check_normalized_moves_bounded(asof: date) -> dict:
    """Verify normalized moves are bounded with a tiered policy:
      - Hard fail if any value > 100 ATR (true ATR-explosion)
      - Hard fail if > 0.5% of values > 50 ATR (rampant outliers)
      - Soft fail if > 5% of values > 10 ATR (questionable distribution)
      - PASS if outliers are rare extreme events (deep-tenor instruments
        on thin overnight hours legitimately can hit 50-100 ATR)
    """
    panel = pd.read_parquet(_paths(asof)["panel"])
    norm_cols = [f"{m}_norm" for m in MEASUREMENTS]
    norm_cols = [c for c in norm_cols if c in panel.columns]
    if not norm_cols:
        return _check_pass(False, "no normalized columns found")
    all_norm = panel[norm_cols].to_numpy().flatten()
    finite_norm = all_norm[np.isfinite(all_norm)]
    if len(finite_norm) == 0:
        return _check_pass(False, "no finite normalized values")
    n_total_finite = len(finite_norm)
    abs_norm = np.abs(finite_norm)
    n_above_100 = int(np.sum(abs_norm > 100))
    n_above_50 = int(np.sum(abs_norm > 50))
    n_above_10 = int(np.sum(abs_norm > 10))
    max_abs = float(np.max(abs_norm))
    if n_above_100 > 0:
        return _check_pass(False,
                              f"{n_above_100} values exceed 100 ATR — true "
                              f"ATR-explosion bug. Max |norm| = {max_abs:.1f}")
    pct_above_50 = n_above_50 / n_total_finite
    if pct_above_50 > 0.005:
        return _check_pass(False,
                              f"{pct_above_50:.2%} of values exceed 50 ATR (>0.5% "
                              f"tolerance — too many extreme outliers)")
    pct_above_10 = n_above_10 / n_total_finite
    if pct_above_10 > 0.05:
        return _check_pass(False,
                              f"{pct_above_10:.1%} of values exceed 10 ATR (>5% "
                              f"tolerance)")
    return _check_pass(True,
                              f"{n_total_finite} normalized values bounded; max "
                              f"|norm| = {max_abs:.2f}, {n_above_50} "
                              f"({pct_above_50:.3%}) above 50 ATR, {n_above_10} "
                              f"({pct_above_10:.2%}) above 10 ATR",
                              max_abs_norm=max_abs,
                              n_above_10=n_above_10,
                              n_above_50=n_above_50,
                              n_total_finite=n_total_finite)


def check_atr_1h_20_finite(asof: date) -> dict:
    panel = pd.read_parquet(_paths(asof)["panel"])
    if "atr_1h_20" not in panel.columns:
        return _check_pass(False, "atr_1h_20 column missing")
    # Allow some NaN (early events, missing data) — gate on majority
    atr_vals = panel["atr_1h_20"]
    n_total = len(atr_vals)
    n_finite_positive = int(((atr_vals > 0) & atr_vals.notna()).sum())
    frac_ok = n_finite_positive / n_total if n_total > 0 else 0
    if frac_ok < 0.70:
        return _check_pass(False,
                              f"only {frac_ok:.1%} of rows have finite positive ATR "
                              f"({n_finite_positive}/{n_total})")
    if (atr_vals < 0).any():
        return _check_pass(False, "negative ATR values present (impossible)")
    return _check_pass(True,
                              f"ATR_1H_20 finite & positive for {frac_ok:.1%} of rows "
                              f"({n_finite_positive}/{n_total}); median = "
                              f"{float(atr_vals.median()):.4f}",
                              median_atr=float(atr_vals.median()),
                              frac_finite_positive=frac_ok)


def check_aggregate_window_consistency(asof: date) -> dict:
    p = _paths(asof)
    if not p["aggregates"].exists():
        return _check_pass(False, "aggregates parquet missing")
    agg = pd.read_parquet(p["aggregates"])
    windows_present = set(agg["window"].unique())
    expected = set(WINDOWS)
    missing = expected - windows_present
    if missing:
        return _check_pass(False, f"missing windows: {sorted(missing)}")
    # Per window, count rows
    by_window = agg.groupby("window").size().to_dict()
    return _check_pass(True,
                              f"all 4 windows present: {by_window}",
                              window_row_counts=by_window)


def check_drift_flag_consistency(asof: date) -> dict:
    p = _paths(asof)
    if not p["drift"].exists():
        return _check_pass(False, "drift parquet missing")
    drift = pd.read_parquet(p["drift"])
    valid_tags = {"FADING", "GROWING", "REVERSING", "STABLE"}
    bad = drift[~drift["drift_tag"].isin(valid_tags)]
    if len(bad) > 0:
        return _check_pass(False,
                              f"{len(bad)} rows with invalid drift_tag: "
                              f"{sorted(bad['drift_tag'].unique())[:5]}")
    counts = drift["drift_tag"].value_counts().to_dict()
    return _check_pass(True, f"drift_tag valid; counts = {counts}",
                              tag_counts=counts)


def check_manifest_integrity(asof: date) -> dict:
    p = _paths(asof)
    if not p["manifest"].exists():
        return _check_pass(False, "manifest missing")
    m = json.loads(p["manifest"].read_text())
    if m.get("builder_version") != BUILDER_VERSION:
        return _check_pass(False,
                              f"version mismatch: manifest={m.get('builder_version')} "
                              f"vs current={BUILDER_VERSION}")
    required = {"asof_date", "event_cutoff_date", "n_tickers_built",
                  "n_events_total", "instruments_count", "windows_computed"}
    missing = required - set(m.keys())
    if missing:
        return _check_pass(False, f"manifest missing fields: {sorted(missing)}")
    return _check_pass(True,
                              f"manifest OK: {m['n_tickers_built']}/{m['n_events_total']}"
                              f" tickers/events, {m['instruments_count']} instruments",
                              **{k: m[k] for k in
                                 ["n_tickers_built", "n_events_total",
                                  "instruments_count", "drift_tag_counts"]})


def verify_all(asof: date) -> dict:
    checks = {
        "catalog_schema_integrity":         check_catalog_schema_integrity(asof),
        "catalog_post_cutoff_filter":       check_catalog_post_cutoff_filter(asof),
        "state_8_assignment_exhaustive":    check_state_8_assignment_exhaustive(asof),
        "ticker_coverage_sufficient":       check_ticker_coverage_sufficient(asof),
        "instrument_panel_completeness":    check_instrument_panel_completeness(asof),
        "normalized_moves_bounded":         check_normalized_moves_bounded(asof),
        "atr_1h_20_finite":                 check_atr_1h_20_finite(asof),
        "aggregate_window_consistency":     check_aggregate_window_consistency(asof),
        "drift_flag_consistency":           check_drift_flag_consistency(asof),
        "manifest_integrity":               check_manifest_integrity(asof),
    }
    n_passed = sum(1 for c in checks.values() if c["passed"])
    return {
        "asof_date": str(asof),
        "passed_all": n_passed == len(checks),
        "n_passed": n_passed,
        "n_total": len(checks),
        "checks": checks,
    }


def _main(argv: list[str]) -> int:
    if len(argv) >= 2:
        asof = date.fromisoformat(argv[1])
    else:
        asof = _resolve_latest_asof()
        if asof is None:
            print("[hei_verify] no HEI manifest", file=sys.stderr)
            return 2
    print(f"Verifying Historical Event Impact for asof = {asof}\n")
    rep = verify_all(asof)
    for n, c in rep["checks"].items():
        print(f"{'[PASS]' if c['passed'] else '[FAIL]'} {n}: {c['message']}")
    print(f"\nSummary: {rep['n_passed']}/{rep['n_total']} checks passed.")
    return 0 if rep["passed_all"] else 1


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
