"""A2 (KNN bands) + A3 (path-conditional FV) — Phase 10 (plan §12).

A2: Mahalanobis-Ledoit-Wolf KNN matcher with ±60d exclusion + 250d
half-life decay weights. Picks K nearest historical bars to today's
state and returns their forward N-day returns.

A3: Bucket A2 analogs by realised FOMC outcomes inside the structure
window (next 90 BD by default), compute conditional FV per bucket on the
{-50, -25, 0, +25, +50} lattice; Markov-chain marginal fallback for
sparse buckets.

Per §15 A2: SRA-only permanently. All emissions tagged `low_sample` until
cross-product (out of scope) is added.

Output:
    .cmc_cache/knn_a2_a3_<asof>.parquet — per-target-node:
        target_date, target_node, n_analogs, mean_R_fwd_5,
        mean_R_fwd_20, std_R_fwd_5, std_R_fwd_20,
        analog_top_dates (semicolon-sep), bucket_outcomes (semicolon-sep)
    .cmc_cache/knn_a2_a3_manifest_<asof>.json
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".cmc_cache"
_CACHE_DIR.mkdir(exist_ok=True)

BUILDER_VERSION = "1.0.0"

DEFAULT_K = 30
DEFAULT_EXCLUSION_DAYS = 60
DEFAULT_HALFLIFE_DAYS = 250
DEFAULT_FORWARD_HORIZONS = (5, 20)
DEFAULT_STRUCTURE_WINDOW_DAYS = 90


def ledoit_wolf_cov(X: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrunk covariance (sklearn convenience)."""
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf().fit(X)
    return lw.covariance_


def mahalanobis_distances(target: np.ndarray, candidates: np.ndarray,
                              cov_inv: np.ndarray) -> np.ndarray:
    """Vector of Mahalanobis distances from target to each row of candidates."""
    diff = candidates - target
    return np.sqrt(np.einsum("ij,jk,ik->i", diff, cov_inv, diff))


@dataclass
class A2KnnResult:
    target_date: date
    target_node: str
    n_analogs: int
    analog_dates: list
    analog_distances: list
    weights: list
    forward_R_5d: list
    forward_R_20d: list
    bucket_outcomes_at_next_fomc: list


def fit_a2_a3_for_node(wide_residual: pd.DataFrame, target_date: date,
                            node: str, fomc_dates_in_horizon: list[date],
                            K: int = DEFAULT_K,
                            exclusion_days: int = DEFAULT_EXCLUSION_DAYS,
                            halflife_days: int = DEFAULT_HALFLIFE_DAYS,
                            ) -> Optional[A2KnnResult]:
    """KNN match for a single target_date × node."""
    if node not in wide_residual.columns:
        return None
    wide_residual = wide_residual.sort_index()
    target_ts = pd.Timestamp(target_date)
    if target_ts not in wide_residual.index:
        return None

    # Use full residual_change row as feature vector
    target_vec = wide_residual.loc[target_ts].to_numpy(dtype=float)
    if np.isnan(target_vec).all():
        return None

    # Candidates: all bars EXCLUDING ±60d window
    excl_lo = target_ts - pd.Timedelta(days=exclusion_days)
    excl_hi = target_ts + pd.Timedelta(days=exclusion_days)
    cand_mask = (wide_residual.index < excl_lo) | (wide_residual.index > excl_hi)
    cand_dates = wide_residual.index[cand_mask]
    if len(cand_dates) < K:
        return None
    cand_X = wide_residual.loc[cand_dates].to_numpy(dtype=float)

    # Drop rows with any NaN
    keep = ~np.isnan(cand_X).any(axis=1)
    cand_X = cand_X[keep]; cand_dates = cand_dates[keep]
    if len(cand_X) < K:
        return None
    if np.isnan(target_vec).any():
        # Fill missing target dims with 0 (rare — residual rows generally complete)
        target_vec = np.where(np.isnan(target_vec), 0.0, target_vec)

    # Ledoit-Wolf cov on candidates
    cov = ledoit_wolf_cov(cand_X)
    try:
        cov_inv = np.linalg.pinv(cov)
    except np.linalg.LinAlgError:
        return None

    distances = mahalanobis_distances(target_vec, cand_X, cov_inv)

    # Top-K
    knn_idx = np.argsort(distances)[:K]
    analog_dates = cand_dates[knn_idx]
    analog_distances = distances[knn_idx]

    # Weights: 250d half-life decay relative to target_date
    age_days = np.array([(target_ts - d).days for d in analog_dates], dtype=float)
    weights = np.exp(-np.abs(age_days) / max(halflife_days, 1))
    weights = weights / max(weights.sum(), 1e-9)

    # Forward N-day returns at this node for each analog
    fwd_5, fwd_20 = [], []
    for d in analog_dates:
        d_ts = pd.Timestamp(d)
        try:
            v_now = float(wide_residual.loc[d_ts, node])
            d5 = wide_residual.index[wide_residual.index > d_ts][:5]
            d20 = wide_residual.index[wide_residual.index > d_ts][:20]
            fwd_5.append(float(wide_residual.loc[d5, node].sum()) if len(d5) > 0 else np.nan)
            fwd_20.append(float(wide_residual.loc[d20, node].sum()) if len(d20) > 0 else np.nan)
        except Exception:
            fwd_5.append(np.nan); fwd_20.append(np.nan)

    # A3 bucket: for each analog, did the next FOMC inside structure window
    # produce a hike/cut/hold?
    bucket_outcomes = []
    for d in analog_dates:
        d_obj = d.date() if hasattr(d, "date") else d
        next_fomc = next(
            (m for m in fomc_dates_in_horizon
             if m > d_obj
             and (m - d_obj).days <= DEFAULT_STRUCTURE_WINDOW_DAYS),
            None,
        )
        if next_fomc is None:
            bucket_outcomes.append("none")
        else:
            # Bucket by sign of post-FOMC residual move
            d_idx = pd.Timestamp(next_fomc)
            try:
                v = float(wide_residual.loc[d_idx, node])
                if v > 5:
                    bucket_outcomes.append("p25")
                elif v < -5:
                    bucket_outcomes.append("m25")
                else:
                    bucket_outcomes.append("0")
            except Exception:
                bucket_outcomes.append("unknown")

    return A2KnnResult(
        target_date=target_date,
        target_node=node,
        n_analogs=K,
        analog_dates=[str(d.date() if hasattr(d, 'date') else d) for d in analog_dates],
        analog_distances=[float(x) for x in analog_distances],
        weights=[float(w) for w in weights],
        forward_R_5d=[float(x) for x in fwd_5],
        forward_R_20d=[float(x) for x in fwd_20],
        bucket_outcomes_at_next_fomc=bucket_outcomes,
    )


def build_knn_a2_a3(asof: Optional[date] = None,
                          K: int = DEFAULT_K) -> dict:
    if asof is None:
        cands = sorted(_CACHE_DIR.glob("regime_manifest_*.json"),
                          key=os.path.getmtime, reverse=True)
        if not cands:
            raise RuntimeError("no regime cache; run Phase 4 first")
        asof = date.fromisoformat(cands[0].stem.replace("regime_manifest_", ""))

    from lib.sra_data import load_turn_residual_change_wide_panel
    from lib.cb_meetings import fomc_meetings_in_range
    wide = load_turn_residual_change_wide_panel("outright", asof)
    if wide is None or wide.empty:
        raise RuntimeError("no Phase 3 residual_change panel")
    wide.index = pd.to_datetime(wide.index)
    target_ts = wide.index.max()

    fomcs = [m.date for m in fomc_meetings_in_range(
        wide.index.min().date(), wide.index.max().date() + timedelta(days=180))]

    rows = []
    # Limit to a focused subset of nodes (front of curve, 1-3y, deep) for compute
    target_nodes = [n for n in ("M0", "M3", "M6", "M12", "M24", "M36", "M48", "M60")
                          if n in wide.columns]
    for node in target_nodes:
        result = fit_a2_a3_for_node(wide, target_ts.date(), node, fomcs, K=K)
        if result is None:
            continue
        # A3 bucketed forward FV per outcome class
        bucket_means = {}
        bucket_counts = {}
        for b in set(result.bucket_outcomes_at_next_fomc):
            mask = [bo == b for bo in result.bucket_outcomes_at_next_fomc]
            ws = np.array(result.weights)[mask]
            r5 = np.array(result.forward_R_5d)[mask]
            ws = ws / max(ws.sum(), 1e-9)
            bucket_means[b] = float(np.nansum(ws * r5))
            bucket_counts[b] = int(np.array(mask).sum())
        # Markov-marginal fallback (if every bucket has count < 5, use overall mean)
        if all(c < 5 for c in bucket_counts.values()):
            for b in list(bucket_means.keys()):
                bucket_means[b] = float(np.nanmean(result.forward_R_5d))
        weights_arr = np.array(result.weights)
        rows.append({
            "target_date": str(target_ts.date()),
            "target_node": node,
            "n_analogs": result.n_analogs,
            "weighted_mean_R_5d": float(np.nansum(weights_arr * np.array(result.forward_R_5d))),
            "weighted_mean_R_20d": float(np.nansum(weights_arr * np.array(result.forward_R_20d))),
            "std_R_5d": float(np.nanstd(result.forward_R_5d, ddof=0)),
            "std_R_20d": float(np.nanstd(result.forward_R_20d, ddof=0)),
            "analog_top_dates": ";".join(result.analog_dates[:5]),
            "bucket_p25_mean": bucket_means.get("p25", np.nan),
            "bucket_0_mean":   bucket_means.get("0", np.nan),
            "bucket_m25_mean": bucket_means.get("m25", np.nan),
            "bucket_p25_count": bucket_counts.get("p25", 0),
            "bucket_0_count":   bucket_counts.get("0", 0),
            "bucket_m25_count": bucket_counts.get("m25", 0),
            "low_sample_flag": True,   # SRA-only forever per §15 A2
        })

    df = pd.DataFrame(rows)
    paths = _paths(asof)
    if not df.empty:
        df.to_parquet(paths["table"], index=False)

    manifest = {
        "builder_version": BUILDER_VERSION,
        "asof_date": asof.isoformat(),
        "n_target_nodes": int(len(df)),
        "K": int(K),
        "exclusion_days": DEFAULT_EXCLUSION_DAYS,
        "halflife_days": DEFAULT_HALFLIFE_DAYS,
        "structure_window_days": DEFAULT_STRUCTURE_WINDOW_DAYS,
        "all_emissions_low_sample": True,
        "low_sample_reason": "SRA-only per plan §15 A2; cross-product OOS",
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def _paths(asof: date) -> dict:
    stamp = asof.isoformat()
    return {
        "table":    _CACHE_DIR / f"knn_a2_a3_{stamp}.parquet",
        "manifest": _CACHE_DIR / f"knn_a2_a3_manifest_{stamp}.json",
    }


def main(argv=None):
    import sys
    args = list(sys.argv[1:] if argv is None else argv)
    asof = (date.fromisoformat(args[0]) if args else None)
    print(f"[knn_a2_a3] building (asof={asof or 'latest'})")
    manifest = build_knn_a2_a3(asof)
    print(f"[knn_a2_a3] {manifest['n_target_nodes']} target nodes; "
              f"K={manifest['K']}; "
              f"all_low_sample={manifest['all_emissions_low_sample']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
