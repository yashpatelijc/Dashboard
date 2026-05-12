"""A1 regime classifier (Phase 4, plan §12).

Pipeline: PCA on Δ_CMC residuals (Phase 3 output) → GMM(K=6, full Σ) →
HMM smoothing (stand-alone Baum-Welch — no hmmlearn dependency) →
Hungarian relabeling against the previous fit so state IDs are stable
across refits.

Inputs
------
- Wide residual_change panel from
  ``lib.sra_data.load_turn_residual_change_wide_panel('outright', asof)``
  (Phase 3 output — 61 nodes × ~684 post-cutover bars).

Outputs
-------
- ``.cmc_cache/regime_states_<asof>.parquet`` — long: bar_date, state_id,
  state_name, posterior, top_state_posterior, posterior_S0..S5.
- ``.cmc_cache/regime_diagnostics_<asof>.parquet`` — per-state stats:
  state_id, state_name, n_bars, mean_run_length, mean_PC1, mean_PC2,
  mean_PC3, std_PC1, ...
- ``.cmc_cache/regime_manifest_<asof>.json``

Verification gates (plan §12 Phase 4):
- posterior dominant state >0.6 on ≥70% of days
- mean run-length ≥10 days
- <5% of days flip after refit (Hungarian-relabel KPI)

State naming convention (semantic, derived from PC sign/cycle context):
- S0..S5 by GMM order, then human-readable label assigned via PC1/PC2/PC3
  centroid signs:
  PC1 (level): + = HIGHER_RATES, - = LOWER_RATES
  PC2 (slope): + = STEEPENER,    - = FLATTENER
  PC3 (curv) : + = HUMP,         - = INVERSE_HUMP
  Combined into 6 names: TRENDING_HIKE_LATE / TRENDING_CUT_LATE /
  STEEPENING_RISK_ON / FLATTENING_FLIGHT / RANGE_BOUND / VOL_BREAKOUT
  (the human labels are deterministic given centroid signs + variance).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".cmc_cache"
_CACHE_DIR.mkdir(exist_ok=True)

BUILDER_VERSION = "1.0.0"
DEFAULT_K = 6
DEFAULT_PCA_COMPONENTS = 3
DEFAULT_HMM_ITER = 30
DEFAULT_HMM_TOL = 1e-4
RANDOM_STATE = 42

# KPI thresholds (plan §12 Phase 4)
KPI_DOMINANT_POST = 0.60       # posterior > 0.6
KPI_DOMINANT_FRAC = 0.70       # on >= 70% of days
KPI_MEAN_RUN_LENGTH = 10        # mean run-length >= 10 days
KPI_REFIT_FLIP_FRAC = 0.05     # < 5% of days flip after refit


# =============================================================================
# Data prep
# =============================================================================

def _load_residual_panel(scope: str, asof: date) -> pd.DataFrame:
    from lib.sra_data import load_turn_residual_change_wide_panel
    return load_turn_residual_change_wide_panel(scope, asof)


def _drop_low_coverage_cols(wide: pd.DataFrame, min_frac: float = 0.7) -> pd.DataFrame:
    """Drop CMC nodes with <``min_frac`` non-NaN coverage."""
    keep = wide.columns[wide.notna().mean(axis=0) >= min_frac]
    return wide[list(keep)]


# =============================================================================
# PCA
# =============================================================================

def fit_pca(X: np.ndarray, n_components: int = DEFAULT_PCA_COMPONENTS,
                use_sparse: bool = False) -> dict:
    """Standardise then PCA. Optionally SparsePCA on top.

    Returns: {'mean': (D,), 'std': (D,), 'components': (n, D),
                'scores': (T, n), 'explained_variance_ratio': (n,)}
    """
    from sklearn.decomposition import PCA, SparsePCA
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd_safe = np.where(sd > 1e-12, sd, 1.0)
    Z = (X - mu) / sd_safe
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    scores = pca.fit_transform(Z)
    out = {
        "mean": mu, "std": sd_safe,
        "components": pca.components_,
        "scores": scores,
        "explained_variance_ratio": pca.explained_variance_ratio_,
    }
    if use_sparse:
        # Refit a SparsePCA on the standardised matrix; project Z onto it
        # for "interpretable" loading scores. We KEEP the PCA scores as the
        # downstream input (sparse adds interpretability, not ML accuracy).
        try:
            spca = SparsePCA(n_components=n_components, alpha=1.0,
                                  random_state=RANDOM_STATE, max_iter=100)
            spca.fit(Z)
            out["sparse_components"] = spca.components_
        except Exception:
            out["sparse_components"] = None
    return out


# =============================================================================
# GMM
# =============================================================================

def fit_gmm(scores: np.ndarray, K: int = DEFAULT_K) -> dict:
    """Gaussian mixture, full covariance. Returns label vector + posteriors."""
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(
        n_components=K, covariance_type="full",
        random_state=RANDOM_STATE, max_iter=200, n_init=3,
        reg_covar=1e-4,
    )
    gmm.fit(scores)
    posteriors = gmm.predict_proba(scores)   # (T, K)
    labels = posteriors.argmax(axis=1)
    return {
        "model": gmm,
        "labels": labels,
        "posteriors": posteriors,
        "means": gmm.means_,
        "covariances": gmm.covariances_,
        "weights": gmm.weights_,
        "bic": gmm.bic(scores),
        "aic": gmm.aic(scores),
    }


# =============================================================================
# HMM smoothing — stand-alone Baum-Welch (no hmmlearn dependency)
# =============================================================================

def _initial_transition_from_labels(labels: np.ndarray, K: int,
                                          smoothing: float = 1.0) -> np.ndarray:
    """Build initial transition matrix from observed GMM-label transitions
    (with Laplace smoothing for unseen pairs)."""
    A = np.full((K, K), smoothing)
    for t in range(1, len(labels)):
        A[labels[t-1], labels[t]] += 1.0
    A /= A.sum(axis=1, keepdims=True)
    return A


def _gaussian_logpdf(X: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Multivariate normal log-pdf. X: (T, D); mu: (D,); cov: (D, D)."""
    D = X.shape[1]
    diff = X - mu
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        # Add small ridge for numerical safety
        cov = cov + 1e-6 * np.eye(D)
        sign, logdet = np.linalg.slogdet(cov)
    inv = np.linalg.inv(cov)
    quad = np.einsum("td,de,te->t", diff, inv, diff)
    return -0.5 * (D * np.log(2 * np.pi) + logdet + quad)


def baum_welch_smooth(scores: np.ndarray,
                          gmm_means: np.ndarray, gmm_covariances: np.ndarray,
                          gmm_weights: np.ndarray, init_labels: np.ndarray,
                          n_iter: int = DEFAULT_HMM_ITER,
                          tol: float = DEFAULT_HMM_TOL) -> dict:
    """HMM forward-backward SMOOTHING with FIXED parameters from GMM.

    Design choice: we deliberately do NOT re-estimate the transition matrix
    via Baum-Welch on small samples (K=6, full covariance, ~700 bars). On
    short post-LIBOR-cutover history, EM re-estimation collapses to absorbing
    states because rare states get Laplace-dominated transition rows that
    iterations push toward zero. The standard reference (Murphy 2012, ch.17)
    notes this failure mode for short series.

    Instead, we:
      1. Build the transition matrix from observed GMM-label transition counts
         + Laplace smoothing (so rare-state rows stay valid).
      2. Run forward-backward ONCE to get smoothed posteriors over states.
      3. Take argmax(posterior) as the labels.

    This preserves the spec's K=6 + GMM(full covariance) but avoids the
    EM-collapse failure mode. KPIs (posterior dominance, run-length, refit
    flip-rate) still apply and are checked by the verification harness.
    """
    T, _D = scores.shape
    K = gmm_means.shape[0]

    # Pre-compute log-emission probabilities
    log_B = np.zeros((T, K))
    for k in range(K):
        log_B[:, k] = _gaussian_logpdf(scores, gmm_means[k],
                                              gmm_covariances[k])

    # Fixed transition matrix from observed GMM transitions + Laplace
    # smoothing (smoothing alpha=2.0 for stronger uniform prior on rare states).
    A = _initial_transition_from_labels(init_labels, K, smoothing=2.0)

    # Initial state probs: GMM weights (the unconditional state distribution)
    pi = gmm_weights.copy()
    pi = pi / pi.sum()

    # Single forward pass
    log_alpha = np.zeros((T, K))
    log_alpha[0] = np.log(pi + 1e-300) + log_B[0]
    for t in range(1, T):
        log_alpha[t] = log_B[t] + _logsumexp_rows(
            log_alpha[t-1, :, None] + np.log(A + 1e-300))
    ll = _logsumexp(log_alpha[-1])

    # Single backward pass
    log_beta = np.zeros((T, K))
    log_beta[-1] = 0.0
    for t in range(T - 2, -1, -1):
        log_beta[t] = _logsumexp_rows(
            np.log(A + 1e-300) + log_B[t+1, None, :] + log_beta[t+1, None, :])

    # Smoothed posteriors
    log_gamma = log_alpha + log_beta
    log_gamma -= _logsumexp_rows(log_gamma)[:, None]
    gamma = np.exp(log_gamma)
    labels_smoothed = gamma.argmax(axis=1)

    return {
        "transition_matrix": A,
        "initial_probs": pi,
        "posteriors": gamma,
        "labels": labels_smoothed,
        "log_likelihood": float(ll),
        "n_iter_run": 1,
    }


def _logsumexp(x: np.ndarray) -> float:
    m = np.max(x)
    return float(m + np.log(np.exp(x - m).sum()))


def _logsumexp_rows(X: np.ndarray) -> np.ndarray:
    """logsumexp along axis=-1, returning a vector of shape (...,)"""
    m = X.max(axis=-1, keepdims=True)
    return (m + np.log(np.exp(X - m).sum(axis=-1, keepdims=True))).squeeze(-1)


# =============================================================================
# Hungarian relabel (state-id stability across refits)
# =============================================================================

def hungarian_relabel(new_means: np.ndarray,
                          previous_means: Optional[np.ndarray]
                          ) -> np.ndarray:
    """Map ``new_means[k]`` rows to the closest previous-mean row via
    Hungarian assignment. Returns a permutation array such that
    ``new_means[perm]`` aligns to ``previous_means``. If no previous fit,
    returns identity (sorted by mean PC1 descending — deterministic seed).
    """
    K = new_means.shape[0]
    if previous_means is None or previous_means.shape != new_means.shape:
        # Deterministic seed ordering: sort by PC1 desc, then PC2 desc
        order = np.lexsort((-new_means[:, 1] if new_means.shape[1] > 1
                                else np.zeros(K),
                               -new_means[:, 0]))
        return np.array(order, dtype=int)
    from scipy.optimize import linear_sum_assignment
    cost = np.linalg.norm(
        new_means[:, None, :] - previous_means[None, :, :], axis=-1
    )
    row_ind, col_ind = linear_sum_assignment(cost)
    # Build permutation: new index i should be remapped to col_ind[i]
    perm = np.zeros(K, dtype=int)
    for r, c in zip(row_ind, col_ind):
        perm[c] = r
    return perm


def _apply_perm_to_labels(labels: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Re-label each entry: new_label = inverse(perm)[old_label]."""
    inv = np.zeros_like(perm)
    for new_id, old_id in enumerate(perm):
        inv[old_id] = new_id
    return inv[labels]


# =============================================================================
# State naming
# =============================================================================

REGIME_NAMES = (
    "TRENDING_HIKE_LATE",
    "TRENDING_CUT_LATE",
    "STEEPENING_RISK_ON",
    "FLATTENING_FLIGHT",
    "RANGE_BOUND",
    "VOL_BREAKOUT",
)


def name_states(centroids: np.ndarray, posteriors: np.ndarray) -> list[str]:
    """Heuristic deterministic naming from PC1 / PC2 / PC3 centroid signs +
    each state's ambient variance (proxy from the entropy of its posterior
    column)."""
    K = centroids.shape[0]
    pc_dim = centroids.shape[1]
    # Per-state mean entropy of posterior column = "uncertainty"
    eps = 1e-12
    H = -np.nanmean(posteriors * np.log(posteriors + eps), axis=0)
    # Sort states deterministically: PC1 desc, then PC2 desc
    order = np.lexsort((
        -(centroids[:, 1] if pc_dim > 1 else np.zeros(K)),
        -centroids[:, 0],
    ))
    # Assign the 6 canonical names by ranking
    names = ["UNKNOWN"] * K
    if K == 6:
        # Map sorted positions to semantic names
        # 0 (highest PC1+PC2): TRENDING_HIKE_LATE
        # 1: STEEPENING_RISK_ON
        # 2: VOL_BREAKOUT (high entropy near the middle)
        # 3: RANGE_BOUND   (low entropy near origin)
        # 4: FLATTENING_FLIGHT
        # 5 (lowest PC1+PC2): TRENDING_CUT_LATE
        # We override 2/3 by entropy: highest H = VOL_BREAKOUT, lowest H = RANGE_BOUND
        rank_names = [
            "TRENDING_HIKE_LATE", "STEEPENING_RISK_ON",
            None, None,
            "FLATTENING_FLIGHT", "TRENDING_CUT_LATE",
        ]
        for rank, k in enumerate(order):
            if rank_names[rank] is not None:
                names[k] = rank_names[rank]
        # Middle two: assign by entropy
        middle = [k for k, n in zip(order[2:4], rank_names[2:4]) if True]
        if len(middle) >= 2:
            mid_entropies = [(H[k], k) for k in middle]
            mid_entropies.sort()
            names[mid_entropies[0][1]] = "RANGE_BOUND"
            names[mid_entropies[-1][1]] = "VOL_BREAKOUT"
    else:
        for rank, k in enumerate(order):
            names[k] = f"S{rank}"
    return names


# =============================================================================
# Top-level driver
# =============================================================================

def fit_regime(asof: date,
                  scope: str = "outright",
                  K: int = DEFAULT_K,
                  n_pca_components: int = DEFAULT_PCA_COMPONENTS,
                  ) -> dict:
    """End-to-end fit. Returns the full result dict (in memory; persisted
    by ``write_regime_cache``)."""
    wide = _load_residual_panel(scope, asof)
    if wide is None or wide.empty:
        raise RuntimeError(f"empty residual panel for asof={asof}")
    wide = _drop_low_coverage_cols(wide)
    # Drop rows with any remaining NaN
    full = wide.dropna(how="any")
    if len(full) < 60:
        raise RuntimeError(f"not enough complete rows after NaN drop: {len(full)}")
    X = full.to_numpy(dtype=float)

    # PCA
    pca = fit_pca(X, n_components=n_pca_components, use_sparse=True)
    Z = pca["scores"]

    # GMM
    gmm = fit_gmm(Z, K=K)

    # HMM smoothing
    hmm = baum_welch_smooth(
        Z, gmm["means"], gmm["covariances"], gmm["weights"], gmm["labels"],
    )

    # Hungarian relabel against previous fit
    prev_means = _read_previous_means(asof)
    perm = hungarian_relabel(gmm["means"], prev_means)
    smoothed_labels = _apply_perm_to_labels(hmm["labels"], perm)
    # Re-index posteriors columns by the same permutation
    inv = np.zeros_like(perm)
    for new_id, old_id in enumerate(perm):
        inv[old_id] = new_id
    posteriors = hmm["posteriors"][:, np.argsort(inv)]
    means_relabeled = gmm["means"][np.argsort(inv)]

    state_names = name_states(means_relabeled, posteriors)

    return {
        "asof": asof,
        "scope": scope,
        "bar_dates": pd.DatetimeIndex(full.index),
        "X_index": full.index,
        "X_cols": full.columns.tolist(),
        "pca": pca,
        "gmm_bic": gmm["bic"],
        "gmm_aic": gmm["aic"],
        "hmm_log_likelihood": hmm["log_likelihood"],
        "hmm_n_iter": hmm["n_iter_run"],
        "labels": smoothed_labels,
        "posteriors": posteriors,
        "transition_matrix": hmm["transition_matrix"],
        "means": means_relabeled,
        "state_names": state_names,
        "previous_means": prev_means,
        "permutation": perm,
    }


def _read_previous_means(asof: date) -> Optional[np.ndarray]:
    """Locate the previous regime fit (most-recent regime_manifest_*.json
    BEFORE the current asof) and return its means matrix. None if first fit."""
    cands = sorted(_CACHE_DIR.glob("regime_manifest_*.json"),
                       key=os.path.getmtime, reverse=True)
    for p in cands:
        try:
            d = date.fromisoformat(p.stem.replace("regime_manifest_", ""))
        except ValueError:
            continue
        if d >= asof:
            continue
        manifest = json.loads(p.read_text())
        means = manifest.get("state_centroids")
        if means:
            return np.array(means, dtype=float)
    return None


# =============================================================================
# Persistence
# =============================================================================

def _regime_paths(asof: date) -> dict:
    stamp = asof.isoformat()
    return {
        "states":      _CACHE_DIR / f"regime_states_{stamp}.parquet",
        "diagnostics": _CACHE_DIR / f"regime_diagnostics_{stamp}.parquet",
        "manifest":    _CACHE_DIR / f"regime_manifest_{stamp}.json",
    }


def write_regime_cache(result: dict) -> dict:
    paths = _regime_paths(result["asof"])
    bar_dates = result["bar_dates"]
    labels = result["labels"]
    posteriors = result["posteriors"]
    state_names = result["state_names"]
    K = posteriors.shape[1]

    states_long = pd.DataFrame({
        "bar_date": pd.Index(bar_dates).date,
        "state_id": labels.astype(int),
        "state_name": [state_names[int(k)] for k in labels],
        "top_state_posterior": posteriors.max(axis=1),
    })
    for k in range(K):
        states_long[f"posterior_S{k}"] = posteriors[:, k]

    # Diagnostics per state
    diag_rows = []
    # Compute run-lengths
    n = len(labels)
    state_run_lengths: dict[int, list] = {k: [] for k in range(K)}
    cur_state = labels[0]; cur_len = 1
    for t in range(1, n):
        if labels[t] == cur_state:
            cur_len += 1
        else:
            state_run_lengths[cur_state].append(cur_len)
            cur_state = labels[t]; cur_len = 1
    state_run_lengths[cur_state].append(cur_len)

    means = result["means"]
    pca_comp_dim = means.shape[1]
    for k in range(K):
        n_bars = int((labels == k).sum())
        runs = state_run_lengths[k]
        mean_run = float(np.mean(runs)) if runs else 0.0
        max_run = int(max(runs)) if runs else 0
        diag = {
            "state_id": k,
            "state_name": state_names[k],
            "n_bars": n_bars,
            "frac_bars": n_bars / n if n > 0 else 0.0,
            "mean_run_length": mean_run,
            "max_run_length": max_run,
        }
        for d in range(pca_comp_dim):
            diag[f"centroid_PC{d+1}"] = float(means[k, d])
        diag_rows.append(diag)
    diagnostics = pd.DataFrame(diag_rows)

    # Atomic writes
    tmp_states = paths["states"].with_suffix(".parquet.tmp")
    tmp_diag = paths["diagnostics"].with_suffix(".parquet.tmp")
    tmp_manifest = paths["manifest"].with_suffix(".json.tmp")

    states_long.to_parquet(tmp_states, index=False)
    diagnostics.to_parquet(tmp_diag, index=False)

    manifest = {
        "builder_version": BUILDER_VERSION,
        "asof_date": result["asof"].isoformat(),
        "scope": result["scope"],
        "n_bars": int(n),
        "K": int(K),
        "pca_components": int(means.shape[1]),
        "pca_explained_variance_ratio": [float(x) for x in
                                                  result["pca"]["explained_variance_ratio"]],
        "gmm_bic": float(result["gmm_bic"]),
        "gmm_aic": float(result["gmm_aic"]),
        "hmm_log_likelihood": result["hmm_log_likelihood"],
        "hmm_n_iter": result["hmm_n_iter"],
        "state_names": state_names,
        "state_centroids": means.tolist(),
        "transition_matrix": result["transition_matrix"].tolist(),
        "permutation_from_previous": result["permutation"].tolist(),
        "had_previous_fit": result["previous_means"] is not None,
    }
    tmp_manifest.write_text(json.dumps(manifest, indent=2, default=str))

    os.replace(tmp_states, paths["states"])
    os.replace(tmp_diag, paths["diagnostics"])
    os.replace(tmp_manifest, paths["manifest"])
    return manifest


def build_regime(asof: Optional[date] = None,
                    scope: str = "outright",
                    K: int = DEFAULT_K) -> dict:
    """Top-level: fit the regime classifier + persist."""
    if asof is None:
        asof = _resolve_latest_residuals_asof()
        if asof is None:
            raise RuntimeError("no residuals manifest found")
    result = fit_regime(asof, scope=scope, K=K)
    return write_regime_cache(result)


def _resolve_latest_residuals_asof() -> Optional[date]:
    cands = sorted(_CACHE_DIR.glob("turn_residuals_manifest_*.json"),
                       key=os.path.getmtime, reverse=True)
    if not cands:
        return None
    stem = cands[0].stem.replace("turn_residuals_manifest_", "")
    try:
        return date.fromisoformat(stem)
    except ValueError:
        return None


# =============================================================================
# CLI
# =============================================================================

def main(argv=None):
    import sys
    args = list(sys.argv[1:] if argv is None else argv)
    if args:
        try:
            asof = date.fromisoformat(args[0])
        except ValueError:
            print("usage: python -m lib.analytics.regime_a1 [YYYY-MM-DD]")
            return 2
    else:
        asof = _resolve_latest_residuals_asof()
        if asof is None:
            print("[regime_a1] no residual cache found; run Phase 3 first")
            return 2
    print(f"[regime_a1] fitting regime classifier for asof={asof}")
    manifest = build_regime(asof)
    print(f"[regime_a1] K={manifest['K']} states / "
              f"n_bars={manifest['n_bars']} / "
              f"BIC={manifest['gmm_bic']:.1f} / "
              f"HMM log-L={manifest['hmm_log_likelihood']:.1f} / "
              f"explained_var={[f'{x:.2%}' for x in manifest['pca_explained_variance_ratio']]}")
    print(f"[regime_a1] state names: {manifest['state_names']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
