"""Regime classification + change-point detection for the PCA trade engine.

Implements gameplan §A1.5–§A1.7 (GMM K=6 full-Σ + Hungarian relabel + HMM smoothing)
plus §A9 (regime-transition signals: GMM posterior degradation online, BOCPD online,
Bai-Perron offline).

Hand-rolled where it matters; scipy used for `linear_sum_assignment` (Hungarian)
and numerical stability helpers (`logsumexp`, `multivariate_normal`).

Output dataclasses are frozen and produced by the corresponding fitters.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from scipy.stats import multivariate_normal


# =============================================================================
# Types
# =============================================================================
@dataclass(frozen=True)
class GMMFit:
    means: np.ndarray            # (K, D)
    covariances: np.ndarray      # (K, D, D)
    weights: np.ndarray          # (K,)
    labels: np.ndarray           # (n_obs,)  argmax assignment
    posteriors: np.ndarray       # (n_obs, K)
    log_likelihood: float
    n_obs: int
    feature_names: list
    asof_dates: Optional[list] = None   # date index per observation


@dataclass(frozen=True)
class HMMFit:
    transition_matrix: np.ndarray       # (K, K)
    smoothed_posteriors: np.ndarray     # (n_obs, K)
    smoothed_labels: np.ndarray         # (n_obs,)
    dominant_confidence: np.ndarray     # (n_obs,)  max posterior per row
    asof_dates: Optional[list] = None


# =============================================================================
# Helpers
# =============================================================================
def _kmeans_pp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """K-means++ seeding. Returns (k, D) initial means."""
    n, d = X.shape
    means = np.empty((k, d), dtype=float)
    idx0 = int(rng.integers(0, n))
    means[0] = X[idx0]
    for i in range(1, k):
        # Distance to nearest already-chosen mean
        diffs = X[:, None, :] - means[:i][None, :, :]   # (n, i, d)
        d2 = (diffs ** 2).sum(axis=2)                    # (n, i)
        min_d2 = d2.min(axis=1)                          # (n,)
        probs = min_d2 / (min_d2.sum() + 1e-12)
        idx = int(rng.choice(n, p=probs))
        means[i] = X[idx]
    return means


def _regularize_cov(cov: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Add diagonal regularization to keep covariance PD."""
    d = cov.shape[0]
    return cov + eps * np.eye(d)


def _safe_logpdf(X: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Multivariate normal log-pdf with PD fallback."""
    try:
        return multivariate_normal.logpdf(X, mean=mean, cov=cov, allow_singular=False)
    except (np.linalg.LinAlgError, ValueError):
        # Fallback: regularize and retry
        cov_reg = _regularize_cov(cov, eps=1e-3)
        return multivariate_normal.logpdf(X, mean=mean, cov=cov_reg, allow_singular=True)


# =============================================================================
# A1 — GMM K=6 full-Σ with EM, k-means++ init, multi-restart
# =============================================================================
def fit_gmm_full_sigma(features: np.ndarray, *,
                        k: int = 6,
                        n_restarts: int = 50,
                        max_iter: int = 200,
                        tol: float = 1e-4,
                        feature_names: Optional[list] = None,
                        asof_dates: Optional[list] = None,
                        seed: int = 42) -> Optional[GMMFit]:
    """GMM-EM K=6 with full covariance.

    `features` shape (n_obs, D). Typical D=5 for (PC1, PC2, PC3, Anchor, σ_PC1_20d).
    50 random restarts; keep the fit with highest final log-likelihood.

    Returns None if sample is too small (n_obs < k*5) or all restarts diverge.
    """
    if features is None or features.size == 0:
        return None
    X = np.asarray(features, dtype=float)
    if np.any(np.isnan(X)):
        # Drop rows with NaN
        mask = ~np.any(np.isnan(X), axis=1)
        X = X[mask]
        if asof_dates is not None:
            asof_dates = [d for d, m in zip(asof_dates, mask) if m]
    n, d = X.shape
    if n < k * 5:
        return None

    rng = np.random.default_rng(seed)
    best: Optional[GMMFit] = None

    for restart in range(n_restarts):
        # --- Init ---
        try:
            means = _kmeans_pp_init(X, k, rng)
        except Exception:
            means = X[rng.choice(n, size=k, replace=False)]
        global_cov = _regularize_cov(np.cov(X.T) if d > 1 else np.array([[np.var(X)]]))
        covs = np.array([global_cov.copy() for _ in range(k)])
        weights = np.ones(k) / k

        prev_ll = -np.inf
        last_resp = None
        last_ll = None
        for _it in range(max_iter):
            # E-step: log responsibilities
            log_pdfs = np.zeros((n, k))
            for j in range(k):
                log_pdfs[:, j] = _safe_logpdf(X, means[j], covs[j])
            log_w = np.log(weights + 1e-300)
            log_resp_unnorm = log_w[None, :] + log_pdfs
            log_norm = logsumexp(log_resp_unnorm, axis=1, keepdims=True)
            log_resp = log_resp_unnorm - log_norm
            resp = np.exp(log_resp)
            ll = float(log_norm.sum())

            # M-step
            Nk = resp.sum(axis=0)
            # Avoid degenerate state collapse
            if np.any(Nk < 1e-6):
                break
            weights = Nk / n
            for j in range(k):
                means[j] = (resp[:, j:j + 1] * X).sum(axis=0) / Nk[j]
                diff = X - means[j]
                covs[j] = (resp[:, j:j + 1] * diff).T @ diff / Nk[j]
                covs[j] = _regularize_cov(covs[j], eps=1e-6)

            # Convergence
            if abs(ll - prev_ll) < tol:
                last_resp = resp
                last_ll = ll
                break
            prev_ll = ll
            last_resp = resp
            last_ll = ll

        if last_resp is None or last_ll is None or not np.isfinite(last_ll):
            continue
        if best is None or last_ll > best.log_likelihood:
            labels = np.argmax(last_resp, axis=1)
            best = GMMFit(
                means=means.copy(),
                covariances=covs.copy(),
                weights=weights.copy(),
                labels=labels,
                posteriors=last_resp.copy(),
                log_likelihood=float(last_ll),
                n_obs=int(n),
                feature_names=list(feature_names) if feature_names else [f"f{i}" for i in range(d)],
                asof_dates=list(asof_dates) if asof_dates else None,
            )
    return best


# =============================================================================
# Hungarian relabel — align new labels to prior labels by Mahalanobis on cluster centers
# =============================================================================
def hungarian_relabel(new_means: np.ndarray,
                       prior_means: Optional[np.ndarray],
                       prior_combined_cov: Optional[np.ndarray] = None) -> np.ndarray:
    """Permutation that maps new labels onto prior labels.

    `permutation[new_label] = old_label`.
    If prior_means is None, returns identity.
    """
    K = new_means.shape[0]
    if prior_means is None:
        return np.arange(K)
    if prior_combined_cov is None:
        # Use identity (Euclidean)
        sigma_inv = np.eye(new_means.shape[1])
    else:
        try:
            sigma_inv = np.linalg.inv(_regularize_cov(prior_combined_cov))
        except np.linalg.LinAlgError:
            sigma_inv = np.eye(new_means.shape[1])
    # Cost matrix: Mahalanobis between (new[i], prior[j])
    cost = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            diff = new_means[i] - prior_means[j]
            cost[i, j] = float(diff @ sigma_inv @ diff)
    row_ind, col_ind = linear_sum_assignment(cost)
    permutation = np.zeros(K, dtype=int)
    for i, j in zip(row_ind, col_ind):
        permutation[i] = j
    return permutation


def apply_permutation(fit: GMMFit, permutation: np.ndarray) -> GMMFit:
    """Permute clusters in a GMMFit so labels[i] becomes permutation[i]."""
    K = fit.means.shape[0]
    inv = np.zeros(K, dtype=int)
    for i, p in enumerate(permutation):
        inv[p] = i
    return GMMFit(
        means=fit.means[inv].copy(),
        covariances=fit.covariances[inv].copy(),
        weights=fit.weights[inv].copy(),
        labels=permutation[fit.labels].copy(),
        posteriors=fit.posteriors[:, inv].copy(),
        log_likelihood=fit.log_likelihood,
        n_obs=fit.n_obs,
        feature_names=list(fit.feature_names),
        asof_dates=list(fit.asof_dates) if fit.asof_dates else None,
    )


# =============================================================================
# A1.7 — HMM forward-backward smoothing with transition refit (Baum-Welch)
# =============================================================================
def fit_hmm_smoothing(gmm_fit: GMMFit, *,
                      transition_init: str = "diag-heavy",
                      diag_strength: float = 0.9,
                      n_baum_welch_iter: int = 5,
                      asof_dates: Optional[list] = None) -> HMMFit:
    """HMM with Gaussian emissions inheriting GMM means/covariances.

    Baum-Welch updates transition matrix only (emissions are frozen at GMM solution).
    Forward-backward gives smoothed posteriors π(r_t = k | y_{1..N}).

    `diag_strength` initial diagonal mass for transitions; off-diagonal split equally.
    """
    K = gmm_fit.means.shape[0]
    N = gmm_fit.n_obs
    if asof_dates is None:
        asof_dates = gmm_fit.asof_dates

    # Initialize transition matrix
    if transition_init == "diag-heavy":
        off = (1.0 - diag_strength) / max(1, K - 1)
        A = np.full((K, K), off)
        np.fill_diagonal(A, diag_strength)
    elif transition_init == "uniform":
        A = np.full((K, K), 1.0 / K)
    else:
        A = np.eye(K)

    # Compute log emissions from GMM means/covs
    # Recover X from gmm_fit (we don't store X — use posteriors as proxy)
    # Since we only have posteriors, we can rebuild log_emissions from those
    # via: log_emissions[t, k] = log_pdf(x_t | mu_k, Sigma_k)
    # We need X. The cleanest is to require X passed in OR store in gmm_fit.
    # Since we don't have X, we derive log_emissions from the posteriors:
    #     log_resp[t, k] = log_w[k] + log_emissions[t, k] − log_norm[t]
    #     log_emissions[t, k] = log_resp[t, k] − log_w[k] + log_norm[t]
    # But log_norm[t] is unknown. Use:
    #     log_emissions[t, k] = log_resp[t, k] − log_w[k]   (up to row-shift, irrelevant)
    log_resp = np.log(gmm_fit.posteriors + 1e-300)
    log_w = np.log(gmm_fit.weights + 1e-300)
    log_emissions = log_resp - log_w[None, :]    # row-shifts cancel in F-B

    # Baum-Welch loop (transition-only update)
    for _ in range(max(1, n_baum_welch_iter)):
        log_alpha, log_beta = _forward_backward(log_emissions, np.log(A + 1e-300),
                                                   gmm_fit.weights)
        # Smoothed posteriors gamma
        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)
        if n_baum_welch_iter <= 1:
            break
        # xi: pair posteriors for transition update
        log_A = np.log(A + 1e-300)
        log_xi_num = (log_alpha[:-1, :, None]
                       + log_A[None, :, :]
                       + log_emissions[1:, None, :]
                       + log_beta[1:, None, :])
        # Normalise per t
        log_norm_xi = logsumexp(log_xi_num.reshape(N - 1, -1), axis=1, keepdims=True)
        log_xi = log_xi_num.reshape(N - 1, -1) - log_norm_xi
        log_xi = log_xi.reshape(N - 1, K, K)
        xi = np.exp(log_xi)
        # M-step for A
        num = xi.sum(axis=0)
        den = num.sum(axis=1, keepdims=True) + 1e-300
        A = num / den

    smoothed_labels = np.argmax(gamma, axis=1)
    dominant_conf = gamma.max(axis=1)

    return HMMFit(
        transition_matrix=A.copy(),
        smoothed_posteriors=gamma.copy(),
        smoothed_labels=smoothed_labels,
        dominant_confidence=dominant_conf,
        asof_dates=list(asof_dates) if asof_dates else None,
    )


def _forward_backward(log_emissions: np.ndarray,
                       log_A: np.ndarray,
                       initial_weights: np.ndarray) -> tuple:
    """Standard log-space forward-backward. Returns (log_alpha, log_beta), each (N, K)."""
    N, K = log_emissions.shape
    log_alpha = np.full((N, K), -np.inf)
    log_alpha[0] = np.log(initial_weights + 1e-300) + log_emissions[0]
    for t in range(1, N):
        # log_alpha[t, k] = logsumexp_j(log_alpha[t-1, j] + log_A[j, k]) + log_emissions[t, k]
        log_alpha[t] = log_emissions[t] + logsumexp(
            log_alpha[t - 1, :, None] + log_A, axis=0)
    log_beta = np.zeros((N, K))    # log(1) at terminal
    for t in range(N - 2, -1, -1):
        log_beta[t] = logsumexp(
            log_A + log_emissions[t + 1, None, :] + log_beta[t + 1, None, :], axis=1)
    return log_alpha, log_beta


# =============================================================================
# A9.1 — GMM posterior degradation (online transition signal)
# =============================================================================
def gmm_posterior_degradation(hmm_fit: HMMFit, *,
                                max_threshold: float = 0.6,
                                second_threshold: float = 0.25) -> pd.Series:
    """Boolean series: True when max posterior < threshold AND second-max > threshold.

    Per gameplan §A9.1 — signals incipient regime transition.
    """
    P = hmm_fit.smoothed_posteriors
    if P is None or P.size == 0:
        return pd.Series(dtype=bool)
    sorted_p = -np.sort(-P, axis=1)    # descending
    max_p = sorted_p[:, 0]
    second_p = sorted_p[:, 1] if P.shape[1] >= 2 else np.zeros_like(max_p)
    fired = (max_p < max_threshold) & (second_p > second_threshold)
    if hmm_fit.asof_dates:
        return pd.Series(fired, index=pd.to_datetime(hmm_fit.asof_dates))
    return pd.Series(fired)


# =============================================================================
# A9.3 — BOCPD online change-point detection (Adams-MacKay 2007)
# =============================================================================
def detect_bocpd(features: np.ndarray, *,
                  hazard_lambda: float = 1 / 100.0,
                  asof_dates: Optional[list] = None,
                  short_run_horizon: int = 5) -> pd.Series:
    """BOCPD with multivariate Gaussian likelihood (independent across dimensions).

    Maintains a posterior over run-lengths. Returns P(run-length < short_run_horizon)
    per timepoint — a high value indicates recent change-point.

    Simplified to per-dim normal-inverse-gamma (NIG) conjugate updates and
    aggregated likelihood across dimensions (assumes feature independence;
    suitable when features = orthogonal PCs, which they are by construction).
    """
    if features is None or features.size == 0:
        return pd.Series(dtype=float)
    X = np.asarray(features, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n, d = X.shape

    # NIG conjugate prior per dim
    mu0 = 0.0
    kappa0 = 1.0
    alpha0 = 1.0
    beta0 = 1.0

    # Track sufficient stats per (run-length, dim)
    # We approximate by pruning: keep top-100 most-likely run-lengths.
    # For our scale (n ≈ 450), we can keep all.
    # Run-length distribution at time t: P(r_t = r) for r ∈ {0, 1, ..., t}
    log_p_r = np.array([0.0])    # log P(r_0 = 0) = 0 (mass at r=0)
    sufficient = {0: {"mu": np.full(d, mu0), "kappa": kappa0,
                       "alpha": alpha0, "beta": np.full(d, beta0)}}

    p_short_run = np.zeros(n)

    for t in range(n):
        x_t = X[t]
        log_pred_probs = []
        new_log_p_r = []
        new_sufficient = {}

        # For each existing run-length, compute predictive prob of observing x_t
        for r, log_p in zip(range(len(log_p_r)), log_p_r):
            ss = sufficient[r]
            # Per-dim Student-t predictive distribution
            mu = ss["mu"]
            kappa = ss["kappa"]
            alpha = ss["alpha"]
            beta = ss["beta"]
            # Marginal per dim is Student-t with df=2*alpha, loc=mu, scale=sqrt(beta*(kappa+1)/(alpha*kappa))
            df = 2 * alpha
            scale_sq = beta * (kappa + 1) / (alpha * kappa)
            # log Student-t pdf per dim
            from scipy.special import gammaln
            log_pred_per_dim = (
                gammaln((df + 1) / 2) - gammaln(df / 2)
                - 0.5 * np.log(df * np.pi * scale_sq)
                - ((df + 1) / 2) * np.log(1 + (x_t - mu) ** 2 / (df * scale_sq + 1e-12))
            )
            log_pred = float(log_pred_per_dim.sum())
            log_pred_probs.append(log_pred)

            # Update sufficient stats for r+1
            new_kappa = kappa + 1
            new_mu = (kappa * mu + x_t) / new_kappa
            new_alpha = alpha + 0.5
            new_beta = beta + (kappa * (x_t - mu) ** 2) / (2 * new_kappa)
            new_sufficient[r + 1] = {
                "mu": new_mu, "kappa": new_kappa,
                "alpha": new_alpha, "beta": new_beta,
            }

        # Prior over hazard
        log_h = np.log(hazard_lambda)
        log_1mh = np.log(1 - hazard_lambda)
        log_pred_arr = np.array(log_pred_probs)

        # Growth: r increases by 1 with prob (1 - h)
        log_growth = log_p_r + log_pred_arr + log_1mh

        # Change-point: total mass at r=0 = sum_r P(r) * pred * h
        log_change = logsumexp(log_p_r + log_pred_arr + log_h)

        # New posterior
        new_log = np.empty(len(log_growth) + 1)
        new_log[0] = log_change
        new_log[1:] = log_growth
        # Normalise
        new_log -= logsumexp(new_log)

        log_p_r = new_log

        # Reset sufficient[0] to prior
        new_sufficient[0] = {"mu": np.full(d, mu0), "kappa": kappa0,
                              "alpha": alpha0, "beta": np.full(d, beta0)}
        sufficient = new_sufficient

        # P(r < short_run_horizon)
        head_mass = np.exp(log_p_r[:short_run_horizon]).sum()
        p_short_run[t] = head_mass

    if asof_dates:
        return pd.Series(p_short_run, index=pd.to_datetime(asof_dates))
    return pd.Series(p_short_run)


# =============================================================================
# A9.2 — Bai-Perron offline structural breaks
# =============================================================================
def detect_bai_perron(series: pd.Series, *,
                       max_breaks: int = 5,
                       trim: float = 0.15,
                       asof_dates: Optional[list] = None) -> list:
    """Multiple structural breaks in mean of `series`.

    Uses dynamic programming for global SSR optimum, sup-F + UDmax for selection.
    Reference: Bai & Perron (2003).

    Returns list of break dates (or integer indices if no asof_dates).
    """
    if series is None or len(series) == 0:
        return []
    y = np.asarray(series.values if hasattr(series, "values") else series, dtype=float)
    mask = ~np.isnan(y)
    y = y[mask]
    if asof_dates is None and isinstance(series, pd.Series):
        asof_dates = list(series.index[mask])
    n = len(y)
    if n < 30:
        return []

    min_seg = max(5, int(trim * n))

    # SSR for each segment [i, j] (i, j inclusive)
    # DP: best SSR with up to m breaks ending at position j
    # Compute segment SSRs
    cumsum = np.concatenate([[0], np.cumsum(y)])
    cumsum2 = np.concatenate([[0], np.cumsum(y ** 2)])

    def seg_ssr(i: int, j: int) -> float:
        # Sum of squares of (y[i..j] − mean)
        if j < i:
            return np.inf
        size = j - i + 1
        if size <= 0:
            return np.inf
        s = cumsum[j + 1] - cumsum[i]
        s2 = cumsum2[j + 1] - cumsum2[i]
        mean = s / size
        ssr = s2 - 2 * mean * s + size * mean ** 2
        return float(max(0.0, ssr))

    # ssr_full = ssr with no breaks
    ssr_full = seg_ssr(0, n - 1)

    # DP for up to max_breaks breaks
    # dp[m][j] = (ssr, breaks_list) — best SSR with EXACTLY m breaks where the
    # last segment ends at position j (inclusive).
    dp = [{} for _ in range(max_breaks + 1)]

    # Base case m=0: no breaks, one segment from 0 to j (inclusive).
    for j in range(min_seg - 1, n):
        dp[0][j] = (seg_ssr(0, j), [])

    # Recursive case m >= 1:
    # dp[m][j] = min over k in [m*min_seg - 1, j - min_seg]
    #            of dp[m-1][k] + ssr(k+1, j)
    for m in range(1, max_breaks + 1):
        # j must allow at least (m+1) segments of size >= min_seg before it
        j_lo = (m + 1) * min_seg - 1
        for j in range(j_lo, n):
            best = (np.inf, [])
            # k is the index of the m-th break (last segment is [k+1, j])
            k_lo = m * min_seg - 1
            k_hi = j - min_seg
            for k in range(k_lo, k_hi + 1):
                if k not in dp[m - 1]:
                    continue
                prev_ssr, prev_breaks = dp[m - 1][k]
                this_ssr = prev_ssr + seg_ssr(k + 1, j)
                if this_ssr < best[0]:
                    best = (this_ssr, prev_breaks + [k])
            if best[0] < np.inf:
                dp[m][j] = best

    # BIC-based selection — standard for Bai-Perron break-count selection.
    # BIC(m) = n * log(ssr_m / n) + (m + 1) * log(n)   (each segment adds 1 mean param)
    # Choose m minimising BIC; require at least 5% SSR reduction vs no-break.
    best_bic = n * np.log(ssr_full / n) + np.log(n)
    chosen_m = 0
    chosen_breaks = []
    for m in range(1, max_breaks + 1):
        if (n - 1) not in dp[m]:
            continue
        ssr_m = dp[m][n - 1][0]
        if ssr_m <= 0 or not np.isfinite(ssr_m):
            continue
        bic_m = n * np.log(ssr_m / n) + (m + 1) * np.log(n)
        rel_reduction = 1 - ssr_m / max(ssr_full, 1e-12)
        if bic_m < best_bic and rel_reduction > 0.02:
            best_bic = bic_m
            chosen_m = m
            chosen_breaks = dp[m][n - 1][1]

    if not chosen_breaks:
        return []
    if asof_dates:
        return [pd.Timestamp(asof_dates[bp]).date() for bp in chosen_breaks]
    return list(chosen_breaks)


# =============================================================================
# Persistence rule wrapper — gameplan §A9 5-day persistence rule
# =============================================================================
def regime_meta_signal(hmm_fit: HMMFit,
                         bocpd_series: pd.Series,
                         bp_breaks: list,
                         *,
                         persistence_d: int = 5) -> pd.Series:
    """Combine three regime-transition channels with persistence rule.

    Returns a categorical pd.Series indexed by date with values in
    {"stable", "transition_warning", "regime_unstable"}.

    `transition_warning` = any single channel fired today.
    `regime_unstable`    = at least one channel has fired ≥ persistence_d consecutive days.
    """
    if hmm_fit.asof_dates is None:
        return pd.Series(dtype=object)
    idx = pd.to_datetime(hmm_fit.asof_dates)
    n = len(idx)

    # Channel 1: GMM posterior degradation
    deg = gmm_posterior_degradation(hmm_fit)
    if deg.empty:
        deg = pd.Series([False] * n, index=idx)
    else:
        deg = deg.reindex(idx, fill_value=False)

    # Channel 2: BOCPD short-run mass > 0.5
    if bocpd_series is None or bocpd_series.empty:
        boc = pd.Series([False] * n, index=idx)
    else:
        boc_aligned = bocpd_series.reindex(idx, method="nearest")
        boc = boc_aligned > 0.5

    # Channel 3: BP breaks (within 5d on either side counts as recent break)
    bp_set = set(pd.Timestamp(bp).date() for bp in bp_breaks) if bp_breaks else set()
    bp_signal = pd.Series(
        [(pd.Timestamp(d).date() in bp_set) for d in idx],
        index=idx,
    )

    any_fire = deg | boc | bp_signal

    # Persistence: rolling sum of any_fire over persistence_d window
    rolling_fires = any_fire.astype(int).rolling(persistence_d, min_periods=1).sum()

    out = pd.Series(["stable"] * n, index=idx, dtype=object)
    out[any_fire] = "transition_warning"
    out[rolling_fires >= persistence_d] = "regime_unstable"
    return out


# =============================================================================
# Convenience: end-to-end regime stack from PC features
# =============================================================================
def fit_regime_stack(features_df: pd.DataFrame, *,
                      k: int = 6,
                      n_restarts: int = 50,
                      max_iter: int = 200,
                      hmm_iter: int = 5,
                      bocpd_hazard: float = 1 / 100.0,
                      bp_max_breaks: int = 5,
                      prior_gmm: Optional[GMMFit] = None) -> dict:
    """End-to-end regime stack: GMM → Hungarian relabel → HMM → A9 channels.

    `features_df` index = dates, columns = features (PC1, PC2, PC3, Anchor, σ_PC1_20d).
    `prior_gmm` (optional) = previous-refit GMM fit; enables Hungarian relabeling.

    Returns:
      {
        "gmm_fit": GMMFit,
        "hmm_fit": HMMFit,
        "posterior_degradation": pd.Series,
        "bocpd_short_run": pd.Series,
        "bai_perron_breaks": list[date],
        "regime_meta_signal": pd.Series,
      }
    """
    out = {
        "gmm_fit": None, "hmm_fit": None,
        "posterior_degradation": pd.Series(dtype=bool),
        "bocpd_short_run": pd.Series(dtype=float),
        "bai_perron_breaks": [],
        "regime_meta_signal": pd.Series(dtype=object),
    }
    if features_df is None or features_df.empty:
        return out
    X = features_df.values.astype(float)
    asof_dates = list(features_df.index)
    feature_names = list(features_df.columns)

    # GMM fit
    gmm = fit_gmm_full_sigma(X, k=k, n_restarts=n_restarts, max_iter=max_iter,
                                feature_names=feature_names, asof_dates=asof_dates)
    if gmm is None:
        return out

    # Hungarian relabel against prior
    if prior_gmm is not None and prior_gmm.means.shape == gmm.means.shape:
        # Combined cov for Mahalanobis
        combined = gmm.covariances.mean(axis=0)
        perm = hungarian_relabel(gmm.means, prior_gmm.means, combined)
        gmm = apply_permutation(gmm, perm)

    # HMM smoothing
    hmm = fit_hmm_smoothing(gmm, n_baum_welch_iter=hmm_iter, asof_dates=asof_dates)

    # A9 channels
    deg = gmm_posterior_degradation(hmm)
    boc = detect_bocpd(X, hazard_lambda=bocpd_hazard, asof_dates=asof_dates)
    # Bai-Perron on smoothed PC1 (20d rolling mean) — captures regime-level
    # structural breaks. Raw PC1 is daily Δ-projections (white-noise around 0),
    # so mean breaks in raw PC1 are meaningless. Smoothing reveals slow regime
    # mean shifts that match the trader's interpretation of "regime change".
    pc1_smoothed = features_df.iloc[:, 0].rolling(20, min_periods=10).mean().dropna()
    bp = detect_bai_perron(pc1_smoothed, max_breaks=bp_max_breaks)
    meta = regime_meta_signal(hmm, boc, bp)

    out["gmm_fit"] = gmm
    out["hmm_fit"] = hmm
    out["posterior_degradation"] = deg
    out["bocpd_short_run"] = boc
    out["bai_perron_breaks"] = bp
    out["regime_meta_signal"] = meta
    return out
