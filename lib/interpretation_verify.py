"""Phase 8 interpretation-layer smoke checks (no parquet output to verify).

CLI: ``python -m lib.interpretation_verify``

Exercises sizing / hedge_calc / counterfactual public APIs to confirm
they import + return well-formed structures.
"""
from __future__ import annotations

import sys
import numpy as np


def _check_pass(passed: bool, message: str, **diag) -> dict:
    return {"passed": bool(passed), "message": str(message), **diag}


def check_sizing_compare() -> dict:
    from lib.sizing import compare_sizing_methods
    df = compare_sizing_methods({"win_rate": 0.55, "avg_win_R": 1.5,
                                       "avg_loss_R": 1.0,
                                       "target_vol_pct": 0.5, "expected_vol_pct": 1.0,
                                       "fixed_fraction": 0.01, "stop_R": 8.0})
    if len(df) != 3:
        return _check_pass(False, f"expected 3 sizing methods, got {len(df)}")
    if (df["notional_usd"] < 0).any():
        return _check_pass(False, "negative notional in sizing output")
    return _check_pass(True, f"3 sizing methods ok; total_notional="
                              f"${df['notional_usd'].sum():.0f}",
                              n_methods=int(len(df)))


def check_kelly_caps() -> dict:
    """Quarter-Kelly cap should fire for high-edge setup."""
    from lib.sizing import kelly_size
    r = kelly_size(0.90, 3.0, 1.0)   # crazy edge -> definitely capped
    if not r.capped:
        return _check_pass(False, "Kelly should cap at quarter-Kelly for high edge")
    return _check_pass(True, "Kelly quarter-cap fires correctly")


def check_dv01_neutral_fly() -> dict:
    from lib.hedge_calc import (
        dv01_neutral_one_two_one, build_legs_table, basket_dv01_check,
    )
    lots = dv01_neutral_one_two_one()
    legs = build_legs_table(["M3","M6","M9"], lots, [96.2, 96.5, 96.7])
    bk = basket_dv01_check(legs)
    if not bk["is_dv01_neutral"]:
        return _check_pass(False,
                              f"1:2:1 fly not DV01-neutral: ${bk['total_dv01_usd_per_bp']}")
    return _check_pass(True, "1:2:1 fly is DV01-neutral exactly")


def check_pca_fly_orthogonality() -> dict:
    from lib.hedge_calc import pca_fly_weights
    L1 = np.array([1.0, 1.0, 1.0])
    L2 = np.array([-1.0, 0.0, 1.0])
    L3 = np.array([1.0, -2.0, 1.0])
    w = pca_fly_weights(L1, L2, L3)
    if abs(L1 @ w) > 1e-10 or abs(L2 @ w) > 1e-10:
        return _check_pass(False, f"PCA-fly not orthogonal: L1.w={L1@w} L2.w={L2@w}")
    if abs(L3 @ w) < 0.5:
        return _check_pass(False, f"PCA-fly has trivial PC3 exposure: {L3@w}")
    return _check_pass(True,
                              f"PCA fly orthogonal to PC1/PC2 (L1·w={L1@w:.1e} L2·w={L2@w:.1e}); "
                              f"PC3 exposure = {L3@w:.3f}")


def check_counterfactual_graceful_empty() -> dict:
    from lib.counterfactual import analog_outcomes
    out = analog_outcomes("__nonexistent_setup_id__")
    if out.get("sufficient_sample"):
        return _check_pass(False, "should report insufficient sample for unknown setup")
    if "reason" not in out:
        return _check_pass(False, "should include 'reason' for insufficient sample")
    return _check_pass(True,
                              "counterfactual gracefully handles unknown setup id "
                              "(no crash, structured response)")


def verify_all() -> dict:
    checks = {
        "sizing_compare":              check_sizing_compare(),
        "kelly_quarter_cap":           check_kelly_caps(),
        "dv01_neutral_fly":            check_dv01_neutral_fly(),
        "pca_fly_orthogonality":       check_pca_fly_orthogonality(),
        "counterfactual_empty_handle": check_counterfactual_graceful_empty(),
    }
    n_passed = sum(1 for c in checks.values() if c["passed"])
    return {"passed_all": n_passed == len(checks),
              "n_passed": n_passed, "n_total": len(checks), "checks": checks}


def _main(argv):
    print("Verifying Phase 8 interpretation modules\n")
    rep = verify_all()
    for n, c in rep["checks"].items():
        print(f"{'[PASS]' if c['passed'] else '[FAIL]'} {n}: {c['message']}")
    print(f"\nSummary: {rep['n_passed']}/{rep['n_total']} checks passed.")
    return 0 if rep["passed_all"] else 1


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
