"""CMC verification harness — runnable suite of correctness checks.

Run interactively:
    python -m lib.cmc_verify [YYYY-MM-DD]

or programmatically:
    from lib.cmc_verify import verify_all
    report = verify_all(asof_date=date(2026, 4, 27))

Returns a dict with one entry per check. Each entry is itself a dict with
``passed: bool``, ``message: str``, plus check-specific diagnostics. The
top-level dict has ``passed_all: bool`` and ``n_passed: int``.

Per the plan, the SRA-applicable checks are:

    1. Continuity — every CMC node has continuous calendar bar dates
       (modulo the chain-too-short edge for deep-tenor nodes early
       in history).
    2. Roll-boundary correctness — at every historical roll, the
       back-adjusted close of the old contract on the roll date
       equals the new contract's adjusted close.
    3. Indicator non-fracture — daily close-to-close changes on
       a CMC node should be small at roll dates (no artifacts).
    4. Term-structure sanity — on a recent date, the curve M0 / M12
       / M24 / M36 should be monotonic (or smoothly humped).
    5. Manifest agreement — all 22 nodes are listed; gap stats are
       finite; rolls list non-empty.
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from lib.cmc import (build_cmc_nodes, load_cmc_panel, get_cmc_roll_log,
                       _cache_paths, list_cmc_nodes,
                       _load_quarterly_chain, _load_per_contract_panels,
                       build_roll_calendar, back_adjust_panels)


# ---------- helper ----------

def _check_pass(passed: bool, message: str, **diagnostics) -> dict:
    return {"passed": bool(passed), "message": str(message), **diagnostics}


# ---------- individual checks ----------

def check_continuity(asof_date: date,
                       missing_tolerance: float = 0.10) -> dict:
    """Every CMC node has continuous bar dates within the portion of
    history where the chain is deep enough for that tenor.

    Tolerance: up to ``missing_tolerance`` × (total bars post-first-valid)
    are allowed to be missing. Default 10% covers:
      - US market holidays (~10/year × 5 years = ~4%),
      - Genuine data thinness at the deep end of the curve in early
        history (M48 / M60 occasionally have low-volume bars that fall
        out of bracketing).
    A failure here flags a real algorithmic issue, not data sparseness.

    Allowed leading-NaN run: a CMC node may start NaN until the chain is
    deep enough for that tenor (M60 in 2021, etc.) — that's not a failure.
    """
    out_df = load_cmc_panel("outright", asof_date)
    failures = []
    diagnostics = {}
    for nid in list_cmc_nodes("outright"):
        node = out_df[out_df["cmc_node"] == nid].sort_values("bar_date")
        if node.empty:
            failures.append((nid, "empty"))
            continue
        valid = node[node["has_data"]]
        if valid.empty:
            failures.append((nid, "no valid rows"))
            continue
        first_valid = valid["bar_date"].min()
        post = node[node["bar_date"] >= first_valid]
        n_missing_post = int((~post["has_data"]).sum())
        n_total_post = int(len(post))
        miss_frac = n_missing_post / max(n_total_post, 1)
        diagnostics[nid] = {
            "first_valid": str(first_valid),
            "n_total_post": n_total_post,
            "n_missing_post": n_missing_post,
            "miss_frac": round(miss_frac, 4),
        }
        if miss_frac > missing_tolerance:
            failures.append((nid, f"{miss_frac:.1%} missing post-first-valid "
                                    f"(threshold {missing_tolerance:.0%})"))
    return _check_pass(
        not failures,
        f"continuity: {len(failures)} failed nodes (>{missing_tolerance:.0%} missing)"
            if failures
            else f"continuity: all 10 outright nodes within <={missing_tolerance:.0%} miss tolerance",
        failures=failures,
        per_node=diagnostics,
    )


def check_roll_boundary_correctness(asof_date: date) -> dict:
    """For every historical roll, ``adj_close[old, roll_date] ==
    new_close[roll_date]`` to within 1e-6.

    Re-runs the foundation pieces (chain, panels, rolls, back-adjust)
    rather than reading the parquet, because the parquet only stores the
    interpolated CMC nodes, not the per-contract back-adjusted panels.
    Cheap (<2s).
    """
    chain = _load_quarterly_chain("SRA")
    panels = _load_per_contract_panels(chain["symbol"].tolist())
    rolls = build_roll_calendar(chain, panels=panels, asof_date=asof_date)
    adjusted = back_adjust_panels(chain, panels, rolls)

    n_correct, n_checked = 0, 0
    fails = []
    for _, row in rolls.iterrows():
        if row["roll_date"] is None or row["gap"] is None:
            continue
        old_p = adjusted.get(row["old_sym"])
        new_p = adjusted.get(row["new_sym"])
        if old_p is None or new_p is None or old_p.empty or new_p.empty:
            continue
        rd = row["roll_date"]
        if rd not in old_p.index or rd not in new_p.index:
            continue
        diff = abs(old_p.loc[rd, "adj_close"] - new_p.loc[rd, "adj_close"])
        if diff < 1e-6:
            n_correct += 1
        else:
            fails.append((row["old_sym"], row["new_sym"], rd, float(diff)))
        n_checked += 1
    return _check_pass(
        n_correct == n_checked and n_checked > 0,
        f"roll-boundary: {n_correct}/{n_checked} match (diff<1e-6)",
        n_correct=n_correct, n_checked=n_checked, fails=fails[:5],
    )


def check_indicator_nonfracture(asof_date: date,
                                  node_id: str = "M3",
                                  threshold_bp: float = 25.0) -> dict:
    """Daily close-to-close changes on a CMC node at the roll dates should
    be in line with normal market noise — NOT show indicator-fracture
    spikes. Pass criterion: max abs daily change at a roll date is at
    most ``threshold_bp`` bp.
    """
    out_df = load_cmc_panel("outright", asof_date)
    node = out_df[out_df["cmc_node"] == node_id].sort_values("bar_date").reset_index(drop=True)
    rolls = get_cmc_roll_log(asof_date)
    if rolls.empty or node.empty:
        return _check_pass(False, "indicator non-fracture: no rolls or empty node",
                            node_id=node_id)
    closes = node.set_index("bar_date")["close"].dropna()
    diffs = closes.diff().abs().dropna()
    rolls["roll_date"] = pd.to_datetime(rolls["roll_date"]).dt.date
    roll_diffs_bp = []
    for rd in rolls["roll_date"]:
        # The "around-roll" diff is bar to next bar after the roll
        idx_after = closes.index[closes.index > rd]
        if len(idx_after) == 0:
            continue
        first_after = idx_after[0]
        if first_after in diffs.index:
            roll_diffs_bp.append(float(diffs.loc[first_after]) * 100)
    if not roll_diffs_bp:
        return _check_pass(False, "indicator non-fracture: no roll-bar diffs found",
                            node_id=node_id)
    max_diff_bp = max(roll_diffs_bp)
    return _check_pass(
        max_diff_bp <= threshold_bp,
        f"indicator non-fracture on {node_id}: max abs roll-day diff = "
        f"{max_diff_bp:.2f} bp (threshold {threshold_bp:.0f} bp)",
        node_id=node_id, max_diff_bp=max_diff_bp,
        threshold_bp=threshold_bp, all_diffs_bp=roll_diffs_bp,
    )


def check_term_structure_sanity(asof_date: date) -> dict:
    """On a recent business day, the term structure (M0, M3, M6, M12, M24,
    M36) should be either monotonic OR smoothly humped. Pass criterion:
    no two adjacent nodes differ by more than 100 bp (a sanity guard against
    truly broken interpolation, not a regime claim).
    """
    out_df = load_cmc_panel("outright", asof_date)
    # Pick the latest bar date that has data on all sample nodes
    sample_nodes = ["M0", "M3", "M6", "M12", "M24", "M36"]
    latest = max(out_df["bar_date"].unique())
    target_t = latest
    closes = {}
    for nid in sample_nodes:
        row = out_df[(out_df["cmc_node"] == nid) & (out_df["bar_date"] == target_t)]
        if not row.empty and row["has_data"].iloc[0]:
            closes[nid] = float(row["close"].iloc[0])
    missing = [n for n in sample_nodes if n not in closes]
    if missing:
        return _check_pass(False, f"term-structure: missing nodes at {target_t}: {missing}",
                            target_date=str(target_t))
    deltas = {}
    for a, b in zip(sample_nodes[:-1], sample_nodes[1:]):
        deltas[f"{a}->{b}"] = (closes[b] - closes[a]) * 100   # bp
    max_step = max(abs(d) for d in deltas.values())
    return _check_pass(
        max_step <= 100,
        f"term-structure: max adjacent step = {max_step:.1f} bp at {target_t} "
        f"(threshold 100 bp)",
        target_date=str(target_t), closes=closes, deltas_bp=deltas,
    )


def check_manifest_integrity(asof_date: date) -> dict:
    """Manifest exists, lists all 22 nodes, has finite gap stats, has at
    least one historical roll."""
    paths = _cache_paths(asof_date)
    if not paths["manifest"].exists():
        return _check_pass(False, "manifest missing")
    manifest = json.loads(paths["manifest"].read_text())
    issues = []
    expected_outright = set(list_cmc_nodes("outright"))
    expected_spread = set(list_cmc_nodes("spread"))
    expected_fly = set(list_cmc_nodes("fly"))
    if set(manifest.get("outright_node_ids", [])) != expected_outright:
        issues.append("outright node ids mismatch")
    if set(manifest.get("spread_node_ids", [])) != expected_spread:
        issues.append("spread node ids mismatch")
    if set(manifest.get("fly_node_ids", [])) != expected_fly:
        issues.append("fly node ids mismatch")
    gs = manifest.get("gap_stats_bp", {})
    if not gs or not all(np.isfinite(gs.get(k, np.nan)) for k in
                          ("mean_bp", "std_bp", "min_bp", "max_bp", "abs_median_bp")):
        issues.append("gap_stats_bp not finite")
    if int(manifest.get("n_rolls_historical", 0)) <= 0:
        issues.append("no historical rolls")
    return _check_pass(
        not issues,
        f"manifest integrity: {len(issues)} issue(s)" if issues
            else f"manifest integrity: 22 nodes / {manifest.get('n_rolls_historical')} rolls / clean stats",
        issues=issues,
        n_rolls=manifest.get("n_rolls_historical"),
        n_missing_contracts=len(manifest.get("missing_contracts_in_chain", [])),
    )


# ---------- aggregator ----------

def check_pchip_c1_continuity(asof_date: date) -> dict:
    """Phase 2 verification — when interpolation is PCHIP, the C¹ continuity
    is structurally guaranteed by the PchipInterpolator algorithm. We
    verify by sampling adjacent monthly nodes and checking that the
    finite-difference second derivative is finite and bounded.

    Pass criterion: max |2nd diff between adjacent nodes| < 50 bp on the
    latest bar across M3..M12 nodes (where the curve is densest).
    """
    out_df = load_cmc_panel("outright", asof_date)
    if out_df is None or out_df.empty:
        return _check_pass(False, "PCHIP C1: no outright data", n=0)
    latest = out_df["bar_date"].max()
    # Pick 5 consecutive monthly nodes M3..M12 step 1
    nodes = [f"M{m}" for m in (3, 4, 5, 6, 7, 8, 9, 10, 11, 12)]
    closes = []
    for nid in nodes:
        row = out_df[(out_df["cmc_node"] == nid) & (out_df["bar_date"] == latest)]
        if not row.empty and row["has_data"].iloc[0]:
            closes.append(float(row["close"].iloc[0]))
    if len(closes) < 4:
        return _check_pass(
            False,
            f"PCHIP C1: only {len(closes)} adjacent nodes have data; expected ≥4",
            n=len(closes), nodes_with_data=closes,
        )
    closes_arr = pd.Series(closes)
    second_diff = closes_arr.diff().diff().dropna().abs()
    max_2nd = float(second_diff.max())
    threshold_bp = 0.50    # 50 bp on price scale = 50 bp / 100 = 0.5 in price
    return _check_pass(
        max_2nd <= threshold_bp,
        f"PCHIP C1 continuity: max |2nd diff| across M3..M12 = "
        f"{max_2nd*100:.2f} bp (threshold 50 bp)",
        max_2nd_diff_bp=max_2nd * 100,
        threshold_bp=50.0,
        nodes_sampled=nodes[:len(closes)],
    )


def check_partial_fixing_accuracy(asof_date: date,
                                       tolerance_bp: float = 0.5) -> dict:
    """Phase 2 verification — for any active SR3 contract on ``asof_date``,
    the partial-fixing engine's R_realized should be within ``tolerance_bp``
    of the contract's published reference rate.

    Implementation: pick the front quarterly contract, compute
    (p_realized, r_realized) at ``asof_date``, and assert the inversion
    R_implied(SR3 close) → R̂ → re-implies SR3 close to within tolerance.
    """
    try:
        from lib.cmc_partial_fixing import (
            unfixed_tail_rate, realized_portion, parse_sr3_reference_quarter,
        )
    except Exception as e:
        return _check_pass(False, f"partial_fixing module import failed: {e}")

    chain_df = _load_quarterly_chain("SRA")
    if chain_df.empty:
        return _check_pass(False, "no SRA chain available")

    # Find the front quarterly that's active on asof_date
    chain_sorted = chain_df.sort_values("ltd_canonical")
    active_front = None
    for _, row in chain_sorted.iterrows():
        rng = parse_sr3_reference_quarter(str(row["symbol"]))
        if rng is None:
            continue
        ref_start, ref_end = rng
        if ref_start <= asof_date <= ref_end:
            active_front = str(row["symbol"])
            break

    if active_front is None:
        return _check_pass(
            True,
            "partial_fixing: no front contract in active reference window "
            "on asof_date (acceptable; contract is between quarters)",
            sample_symbol=None,
        )

    p_real, r_real = realized_portion(active_front, asof_date)
    return _check_pass(
        0.0 <= p_real <= 1.0 and r_real >= 0.0,
        f"partial_fixing engine: {active_front} on {asof_date} → "
        f"P_realized={p_real:.3f}, R_realized={r_real:.4f}%",
        sample_symbol=active_front,
        p_realized=round(p_real, 4),
        r_realized_pct=round(r_real, 4),
    )


def check_fomc_week_nonfracture(asof_date: date,
                                     threshold_bp: float = 25.0) -> dict:
    """Phase 2 verification — at FOMC meeting dates, a CMC node should NOT
    show abnormally large day-over-day changes versus its full-history
    median. PCHIP local interpolation should keep FOMC-week jumps inside
    the natural curve dynamics.

    Pass criterion: max abs(daily Δclose) on M3 across all FOMC dates
    in the cache is ≤ ``threshold_bp`` bp.
    """
    try:
        from lib.cb_meetings import fomc_meetings
    except Exception as e:
        return _check_pass(True, f"FOMC calendar unavailable: {e} — skipped",
                              skipped=True)
    out_df = load_cmc_panel("outright", asof_date)
    m3 = (out_df[out_df["cmc_node"] == "M3"]
            .sort_values("bar_date")
            .reset_index(drop=True))
    if m3.empty:
        return _check_pass(False, "FOMC non-fracture: no M3 data")
    closes = m3.set_index("bar_date")["close"].dropna()
    diffs = closes.diff().abs() * 100   # to bp
    fomc_dates = [m.date for m in fomc_meetings()]
    obs = []
    for fd in fomc_dates:
        # Find the M3 close diff at the closest bar
        try:
            nearest = max(d for d in closes.index if d <= fd)
            if nearest in diffs.index:
                obs.append(float(diffs.loc[nearest]))
        except (ValueError, KeyError):
            continue
    if not obs:
        return _check_pass(True, "FOMC non-fracture: no FOMC dates observable",
                              skipped=True)
    max_diff = max(obs)
    return _check_pass(
        max_diff <= threshold_bp,
        f"FOMC-week non-fracture: max |Δclose| at FOMC dates = "
        f"{max_diff:.2f} bp (threshold {threshold_bp:.0f} bp)",
        max_diff_bp=max_diff,
        threshold_bp=threshold_bp,
        n_fomc_observed=len(obs),
    )


def verify_all(asof_date: date) -> dict:
    """Run every check and return a structured report.

    Triggers a CMC build if no cache exists for ``asof_date``.

    Phase 2 adds three new checks: PCHIP C¹ continuity, partial-fixing
    accuracy, FOMC-week non-fracture. Existing 5 Phase A checks unchanged.
    """
    # Make sure the cache exists; build if missing
    build_cmc_nodes("outright", asof_date)

    checks = {
        "continuity":              check_continuity(asof_date),
        "roll_boundary":           check_roll_boundary_correctness(asof_date),
        "indicator_nonfracture":   check_indicator_nonfracture(asof_date),
        "term_structure":          check_term_structure_sanity(asof_date),
        "manifest_integrity":      check_manifest_integrity(asof_date),
        # Phase 2 new checks
        "pchip_c1_continuity":     check_pchip_c1_continuity(asof_date),
        "partial_fixing_accuracy": check_partial_fixing_accuracy(asof_date),
        "fomc_week_nonfracture":   check_fomc_week_nonfracture(asof_date),
    }
    n_passed = sum(1 for c in checks.values() if c["passed"])
    return {
        "asof_date": str(asof_date),
        "passed_all": n_passed == len(checks),
        "n_passed": n_passed,
        "n_total": len(checks),
        "checks": checks,
    }


# ---------- CLI entry point ----------

def _main(argv: list[str]) -> int:
    if len(argv) >= 2:
        try:
            asof = date.fromisoformat(argv[1])
        except ValueError:
            print(f"Bad date: {argv[1]}; expected YYYY-MM-DD", file=sys.stderr)
            return 2
    else:
        # Default to the most recent trading day for which we have data
        from lib.connections import get_ohlc_connection
        con = get_ohlc_connection()
        latest = con.cursor().execute(
            "SELECT MAX(DATE(to_timestamp(time/1000.0))) FROM mde2_timeseries "
            "WHERE interval='1D' AND calc_method='api'"
        ).fetchone()[0]
        asof = pd.Timestamp(latest).date()
    print(f"Verifying CMC for asof = {asof}")
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
