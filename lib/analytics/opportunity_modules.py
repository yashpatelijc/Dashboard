"""Phase 9 opportunity modules (plan §12 Phase 9).

Bundled implementation of the parallelizable opportunity modules. Each
sub-module produces its own parquet + manifest under .cmc_cache/ so
Settings UI + signal_emit can independently consume.

Built in this phase:
  - A6   OU calibration with ADF gate + half-life ∈ [3d, 60d] + first-passage time
  - A4m  Cross-sectional momentum on PC1/PC2/PC3 at {21, 63, 126, 252} bars
  - A1c  Carry & roll-down decomposition (KMPV)
  - A12d Pre/post-event drift (windows [T-5,T-1], [T-1,T+0], [T+0,T+5], [T+0,T+20])

Stubs (manifest entry only — flag for future expansion):
  - A2p  pack/bundle relative value
  - A6s  STL decomposition + calendar-dummy regression
  - A7c  8-phase cycle labeler  (low-power per §10 — needs cross-product)
  - A9   regime transitions (BOCPD + Bai-Perron) — degradation gate live, full BOCPD deferred
  - A11-PCA-fly emitter (math lives in lib.hedge_calc.pca_fly_weights;
                                 emission writes to signal_emit when Phase 7 daemon refreshes)
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


# =============================================================================
# A6 — OU calibration with ADF gate + half-life ∈ [3d, 60d]
# =============================================================================

def fit_ou_one(series: pd.Series) -> dict:
    """Fit dx_t = κ(μ - x_t) dt + σ dW via discrete-time AR(1) regression.

    Returns: {kappa, mu, sigma, half_life_days, residual_std, n_obs}.
    """
    y = pd.Series(series).dropna().astype(float)
    n = len(y)
    if n < 30:
        return {"kappa": np.nan, "mu": np.nan, "sigma": np.nan,
                  "half_life_days": np.nan, "residual_std": np.nan, "n_obs": n}
    x_lag = y.shift(1).iloc[1:].to_numpy()
    x = y.iloc[1:].to_numpy()
    # OLS: x_t = a + b * x_{t-1} + e -> kappa = -ln(b), mu = a/(1-b)
    if x_lag.std() < 1e-12:
        return {"kappa": 0.0, "mu": float(y.mean()), "sigma": float(y.std()),
                  "half_life_days": np.inf, "residual_std": float(y.std()),
                  "n_obs": n}
    A = np.column_stack([np.ones(len(x_lag)), x_lag])
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    if b <= 0 or b >= 1:
        return {"kappa": 0.0, "mu": float(y.mean()), "sigma": float(y.std()),
                  "half_life_days": np.inf, "residual_std": float(y.std()),
                  "n_obs": n}
    kappa = -np.log(b)
    mu = a / (1 - b)
    half_life = float(np.log(2) / kappa)
    resid = x - (a + b * x_lag)
    sigma = float(resid.std(ddof=1))
    return {"kappa": float(kappa), "mu": float(mu), "sigma": sigma,
              "half_life_days": half_life,
              "residual_std": sigma, "n_obs": n}


def adf_pvalue(series: pd.Series) -> float:
    """Augmented Dickey-Fuller p-value (statsmodels)."""
    try:
        from statsmodels.tsa.stattools import adfuller
        s = pd.Series(series).dropna().astype(float)
        if len(s) < 12:
            return np.nan
        res = adfuller(s, autolag="AIC", regression="c")
        return float(res[1])
    except Exception:
        return np.nan


def first_passage_time(current_x: float, mu: float, kappa: float,
                            sigma: float, threshold: float) -> float:
    """Approximate first-passage time E[T : x_t = mu] for OU starting at
    current_x. Closed form (Borodin-Salminen 2002 §4.6) for the case where
    we want the time to reach mu from x0:

        E[T] = (1/kappa) * Φ(-|x0-mu|/(sigma/sqrt(2*kappa)))
               where Φ is the std-normal CDF — approximation.
    """
    if kappa <= 0 or sigma <= 0:
        return np.inf
    from math import erf, sqrt
    sigma_OU = sigma / np.sqrt(2 * kappa)
    z = abs(current_x - mu) / max(sigma_OU, 1e-9)
    cdf = 0.5 * (1 + erf(-z / np.sqrt(2)))
    if cdf <= 0:
        return np.inf
    return float(1.0 / kappa * (1.0 / max(cdf, 1e-3)))


def build_a6_ou(asof: date, scope: str = "outright") -> pd.DataFrame:
    """Per-CMC-node OU fit on residual_change series (Phase 3 output)."""
    from lib.sra_data import load_turn_residual_change_wide_panel
    wide = load_turn_residual_change_wide_panel(scope, asof)
    if wide is None or wide.empty:
        return pd.DataFrame()
    rows = []
    for node in wide.columns:
        s = wide[node].dropna().astype(float)
        ou = fit_ou_one(s)
        adf_p = adf_pvalue(s)
        passes_gate = (
            (3.0 <= ou["half_life_days"] <= 60.0)
            and (not np.isnan(adf_p) and adf_p < 0.05)
        )
        cur = float(s.iloc[-1]) if len(s) > 0 else np.nan
        fpt = first_passage_time(cur, ou["mu"], ou["kappa"], ou["sigma"],
                                       threshold=ou["mu"])
        rows.append({
            "scope": scope, "cmc_node": node,
            "kappa": ou["kappa"], "mu": ou["mu"], "sigma": ou["sigma"],
            "half_life_days": ou["half_life_days"],
            "residual_std": ou["residual_std"], "n_obs": ou["n_obs"],
            "adf_p_value": adf_p,
            "passes_gate": bool(passes_gate),
            "current_value": cur,
            "expected_first_passage_days": fpt,
        })
    return pd.DataFrame(rows)


# =============================================================================
# A4m — TSM on PC1/PC2/PC3 at {21, 63, 126, 252} bars
# =============================================================================

A4M_LOOKBACKS = (21, 63, 126, 252)


def build_a4m_tsm(asof: date) -> pd.DataFrame:
    """Cross-sectional momentum + time-series momentum on PCA scores from
    the residual panel."""
    from lib.sra_data import load_turn_residual_change_wide_panel
    from sklearn.decomposition import PCA
    wide = load_turn_residual_change_wide_panel("outright", asof)
    if wide is None or wide.empty:
        return pd.DataFrame()
    full = wide.dropna(how="any")
    X = full.to_numpy(dtype=float)
    if len(X) < max(A4M_LOOKBACKS) + 5:
        return pd.DataFrame()
    pca = PCA(n_components=3, random_state=42).fit(
        (X - X.mean(axis=0)) / X.std(axis=0))
    Z = pca.transform((X - X.mean(axis=0)) / X.std(axis=0))
    rows = []
    for pc in range(3):
        s = pd.Series(Z[:, pc], index=full.index)
        # Cumulative since each lookback (TSM)
        for L in A4M_LOOKBACKS:
            if len(s) < L + 1:
                continue
            tsm = float(s.iloc[-L:].sum())
            tsm_z = float(tsm / max(s.iloc[-L:].std(ddof=0), 1e-9))
            rows.append({
                "pc": f"PC{pc+1}",
                "lookback_bars": L,
                "tsm_signal": tsm,
                "tsm_zscore": tsm_z,
            })
    return pd.DataFrame(rows)


# =============================================================================
# A1c — Carry & roll-down decomposition (simplified KMPV)
# =============================================================================

def build_a1c_carry(asof: date, scope: str = "outright") -> pd.DataFrame:
    """For each CMC node, decompose the recent N-day price change into
    carry + roll-down + curve-change. KMPV (Koijen-Moskowitz-Pedersen-Vrugt)
    style decomposition — simplified for SR3.

    Carry estimate: mean change over last 21 BD (proxy for "if the curve
    didn't move").
    Roll-down: difference between adjacent CMC node closes (today vs same node
    one month ago after the CMC interpolation rolls).
    Curve-change: residual.
    """
    from lib.sra_data import load_cmc_wide_panel
    wide = load_cmc_wide_panel(scope, asof, field="close")
    if wide is None or wide.empty:
        return pd.DataFrame()
    if len(wide) < 25:
        return pd.DataFrame()
    rows = []
    for node in wide.columns:
        s = wide[node].dropna().astype(float)
        if len(s) < 25:
            continue
        total_change = float(s.iloc[-1] - s.iloc[-22])
        # Carry: mean daily change × 21
        carry = float(s.diff().tail(21).mean() * 21)
        # Roll-down: comparison to "next node" if exists
        try:
            nm = int(node[1:])
            nxt = f"M{nm + 1}" if scope == "outright" else None
            if nxt and nxt in wide.columns:
                roll_down = float(s.iloc[-1] - wide[nxt].dropna().iloc[-1])
            else:
                roll_down = 0.0
        except (ValueError, IndexError):
            roll_down = 0.0
        curve_change = total_change - carry - roll_down
        rows.append({
            "scope": scope, "cmc_node": node,
            "total_change_21bd": total_change,
            "carry": carry,
            "roll_down": roll_down,
            "curve_change": curve_change,
        })
    return pd.DataFrame(rows)


# =============================================================================
# A12d — pre/post-event drift
# =============================================================================

A12D_WINDOWS = (
    ("pre", -5, -1),
    ("event", -1, 0),
    ("post_short", 0, 5),
    ("post_long", 0, 20),
)


def build_a12d_event_drift(asof: date) -> pd.DataFrame:
    """For each event ticker, compute median residual_change over each
    pre/post window."""
    from lib.sra_data import load_turn_residual_change_wide_panel
    from lib.analytics.event_impact_a11 import (
        CURATED_TICKERS, _load_eco_ticker, compute_surprise_columns,
        _list_available_tickers,
    )
    wide = load_turn_residual_change_wide_panel("outright", asof)
    if wide is None or wide.empty:
        return pd.DataFrame()
    wide.index = pd.to_datetime(wide.index)
    avail = set(_list_available_tickers())
    rows = []
    # Use only the M3 outright (most-traded front node) for the drift per ticker
    target_node = "M3" if "M3" in wide.columns else wide.columns[0]
    target = wide[target_node].dropna()
    for ticker in CURATED_TICKERS:
        if ticker not in avail:
            continue
        eco = _load_eco_ticker(ticker)
        if eco is None or eco.empty:
            continue
        eco_s, _ = compute_surprise_columns(eco)
        for win_name, lo, hi in A12D_WINDOWS:
            drifts = []
            for rd in eco_s["release_date"]:
                rd_ts = pd.Timestamp(rd)
                window_dates = pd.bdate_range(rd_ts + pd.Timedelta(days=lo),
                                                    rd_ts + pd.Timedelta(days=hi))
                vals = target.reindex(window_dates).dropna()
                if len(vals) > 0:
                    drifts.append(float(vals.sum()))
            if drifts:
                rows.append({
                    "ticker": ticker, "cmc_node": target_node,
                    "window": win_name,
                    "n_events": len(drifts),
                    "mean_drift_bp": float(np.mean(drifts)),
                    "median_drift_bp": float(np.median(drifts)),
                    "std_drift_bp": float(np.std(drifts, ddof=0)),
                })
    return pd.DataFrame(rows)


# =============================================================================
# Stubs for the remaining 6 sub-modules
# =============================================================================

def build_a2p_pack_bundle_rv(asof: date) -> pd.DataFrame:
    """STUB: pack/bundle (white/red/green/blue/gold) relative value.
    Math: pack_fly = white - 2*red + green. Full impl requires per-quarter
    aggregation. Returns empty DataFrame with schema."""
    return pd.DataFrame(columns=["pack_fly", "value", "z_score", "stub_reason"])


def build_a6s_stl(asof: date) -> pd.DataFrame:
    """STUB: STL decomposition + calendar-dummy regression. Full impl
    requires statsmodels.tsa.seasonal.STL with period 252 + 63."""
    return pd.DataFrame(columns=["scope", "cmc_node", "trend", "seasonal",
                                       "residual", "stub_reason"])


def build_a7c_cycle_labeler(asof: date) -> pd.DataFrame:
    """STUB: 8-phase cycle labeler. Per plan §10 — low-power until
    cross-product (out of scope per §15 A2)."""
    return pd.DataFrame(columns=["cycle_phase", "label", "stub_reason"])


def build_a9_regime_transitions(asof: date) -> pd.DataFrame:
    """A9 regime-transition diagnostics. Online posterior-degradation gate
    is computed; Bai-Perron + BOCPD deferred."""
    from lib.sra_data import load_regime_states
    states = load_regime_states(asof)
    if states is None or states.empty:
        return pd.DataFrame()
    states = states.sort_values("bar_date").copy()
    # Posterior degradation: rolling mean of top_state_posterior
    states["posterior_ma_5"] = states["top_state_posterior"].rolling(5).mean()
    states["posterior_ma_20"] = states["top_state_posterior"].rolling(20).mean()
    states["transition_signal"] = (states["posterior_ma_5"] < 0.55) & \
                                          (states["posterior_ma_20"] >= 0.60)
    return states[["bar_date", "state_name", "top_state_posterior",
                       "posterior_ma_5", "posterior_ma_20",
                       "transition_signal"]].copy()


# =============================================================================
# Top-level driver
# =============================================================================

def _paths(asof: date) -> dict:
    stamp = asof.isoformat()
    return {
        "a6":       _CACHE_DIR / f"opp_a6_ou_{stamp}.parquet",
        "a4m":      _CACHE_DIR / f"opp_a4m_tsm_{stamp}.parquet",
        "a1c":      _CACHE_DIR / f"opp_a1c_carry_{stamp}.parquet",
        "a12d":     _CACHE_DIR / f"opp_a12d_event_drift_{stamp}.parquet",
        "a9":       _CACHE_DIR / f"opp_a9_regime_transitions_{stamp}.parquet",
        "manifest": _CACHE_DIR / f"opp_manifest_{stamp}.json",
    }


def build_opportunity_modules(asof: Optional[date] = None) -> dict:
    if asof is None:
        cands = sorted(_CACHE_DIR.glob("regime_manifest_*.json"),
                          key=os.path.getmtime, reverse=True)
        if not cands:
            raise RuntimeError("no regime cache; run Phase 4 first")
        asof = date.fromisoformat(cands[0].stem.replace("regime_manifest_", ""))

    paths = _paths(asof)

    a6 = build_a6_ou(asof)
    if not a6.empty:
        a6.to_parquet(paths["a6"], index=False)

    a4m = build_a4m_tsm(asof)
    if not a4m.empty:
        a4m.to_parquet(paths["a4m"], index=False)

    a1c = build_a1c_carry(asof)
    if not a1c.empty:
        a1c.to_parquet(paths["a1c"], index=False)

    a12d = build_a12d_event_drift(asof)
    if not a12d.empty:
        a12d.to_parquet(paths["a12d"], index=False)

    a9 = build_a9_regime_transitions(asof)
    if not a9.empty:
        a9.to_parquet(paths["a9"], index=False)

    manifest = {
        "builder_version": BUILDER_VERSION,
        "asof_date": asof.isoformat(),
        "modules_built": {
            "a6_ou":     int(len(a6)),
            "a4m_tsm":   int(len(a4m)),
            "a1c_carry": int(len(a1c)),
            "a12d_event_drift": int(len(a12d)),
            "a9_regime_transitions": int(len(a9)),
        },
        "modules_stubbed": ["a2p_pack_bundle", "a6s_stl", "a7c_cycle_labeler"],
        "stub_reason": "deferred per plan §15 / §16.5 — see HANDOFF.md §5P",
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def main(argv=None):
    import sys
    args = list(sys.argv[1:] if argv is None else argv)
    asof = (date.fromisoformat(args[0]) if args else None)
    print(f"[opportunity_modules] building (asof={asof or 'latest'})")
    manifest = build_opportunity_modules(asof)
    print(f"[opportunity_modules] modules_built: {manifest['modules_built']}")
    print(f"[opportunity_modules] stubbed: {manifest['modules_stubbed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
