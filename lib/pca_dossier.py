"""Per-contract dossier computations for the SRA › PCA › Trade Screener.

`contract_dossier_data(symbol, panel)` is the single public entry point. It assembles
every per-contract analysis the engine has run on one outright (or — where the
analysis applies — one spread/fly) into a single dict that the renderer consumes.

The 18 sections are (numbered to match the renderer):

  1. Identity                  — contract symbol, parsed delivery, ref period, pack
  2. Pricing                   — last close, day Δ, ATR proxy, implied 3M fwd yield
  3. Position on curve         — CMC chart with this contract's tenor highlighted
  4. Curve-factor sensitivity  — plain-English level/slope/curvature exposure
  5. Recent stretch            — multi-lookback z, percentile, ADF, Hurst, half-life
  6. Stationarity diagnostics  — ADF, KPSS, variance-ratio, OU κ/μ/σ, Hurst, HL
  7. Three FV views            — historical z, analog FV (A2), path FV (A3 — fallback
                                  to analog when policy-path history sparse), carry/roll
  8. Active trade ideas        — every TradeIdea whose legs include this contract
  9. Event sensitivity         — per-event β on (CPI, NFP, FOMC, ISM) + post-event drift
 10. Calendar                  — next FOMC + days-to-expiry + season-week impact
 11. Momentum                  — TSM signs at 21/63/126/252d horizons
 12. Cycle / regime            — phase, GMM/HMM regime, days since last break
 13. Hedge candidates          — PC1-isolated 4-leg baskets including this leg
 14. Positioning               — CFTC COT proxy (UST 2Y net non-comm = STIR proxy)
 15. Charts                    — 4 panels: 1y price+events, residual+OU bands,
                                  event-β bar with 95% CI, z-heatmap
 16. Execution profile         — avg volume, slippage at multiple sizes, holiday cal
 17. Quick actions             — filter-to-contract, neighbor comparison
 18. Provenance                — eff_n, refit age, recon%, print quality, cross-PC corr,
                                  convexity flag

All computations are deliberately defensive: every data fetch is wrapped, every
result returns either a usable value or an explicit None — so the renderer can
gray-out missing sections without bombing.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from lib.contract_units import parse_legs
from lib.fomc import (
    is_quarterly,
    next_fomc_date,
    parse_sra_outright,
    previous_fomc_date,
    reference_period,
)
from lib.mean_reversion import (
    adf_test,
    hurst_exponent,
    hurst_label,
    ou_half_life,
    percentile_rank_value,
    zscore_value,
    composite_reversion_score,
)
from lib.pca import (
    EFF_N_FLOOR,
    PCAFit,
    SEGMENT_BOUNDS,
    _instrument_feature_mean,
    _instrument_loadings,
    _outright_tenor_months,
    _outright_yield_bp_panel,
    _pchip_curve,
    enumerate_pc1_synthetics,
    solve_isolated_pc_weights,
)
from lib.pca_analogs import (
    analog_fv_band,
    knn_analog_search,
    ledoit_wolf_shrinkage,
    path_bucket_assignment,
    path_conditional_fv,
    PATH_BUCKETS_5,
)
from lib.pca_step_path import (
    build_policy_path_history_from_fdtr,
    fit_step_path_bootstrap,
)
from lib.pca_turn_adjuster import turn_adjustment_for_contract
from lib.pca_cross_asset import load_cross_asset_panel
from lib.pca_cross_asset_analysis import run_all_cross_asset_analyses
from lib.sra_data import compute_pack_groups, load_reference_rate_panel


# =============================================================================
# constants
# =============================================================================

# UST-2Y CFTC NCN proxy. IMM13 (Eurodollar legacy) is empty in the warehouse —
# Eurodollars were delisted in 2023 and BBG no longer publishes COT for them.
# UST-2Y net non-commercial captures the same speculator front-end positioning.
_COT_PROXY_TICKER = "CBT42NCN_Index"
_COT_PROXY_OI_TICKER = "CBT42OIN_Index"
_BBG_PARQUET_ROOT = r"D:\BBG data\parquet"

# Tier-1/2 macro surprise tickers used for per-contract event sensitivity.
# Each entry: (event_class, eco_ticker_filename_without_ext)
# Phase B.7 — expanded from 8 to 25+ tickers per gameplan §3.1
_SURPRISE_TICKERS = [
    # Inflation
    ("CPI YoY",         "CPI_YOY_Index"),
    ("CPI MoM",         "CPI_CHNG_Index"),
    ("Core CPI YoY",    "CPI_XYOY_Index"),
    ("Core CPI MoM",    "CPUPXCHG_Index"),
    # Employment
    ("NFP",             "NFP_TCH_Index"),
    ("NFP Prior Rev",   "NFP_PCH_Index"),
    ("ADP",             "ADP_CHNG_Index"),
    ("AHE MoM",         "AHE_MOM_Index"),
    ("AHE YoY",         "AHE_YOY_Index"),
    ("Unemp",           "USURTOT_Index"),
    ("Initial Claims",  "INJCJC_Index"),
    ("Continuing",      "INJCSP_Index"),
    ("JOLTS Open",      "JOLTOPEN_Index"),
    # ISM / PMI
    ("ISM Mfg",         "NAPMPMI_Index"),
    ("ISM Svcs",        "NAPMNMI_Index"),
    ("ISM Mfg Prices",  "NAPMPRIC_Index"),
    # Activity
    ("Retail Sales",    "RSTAMOM_Index"),
    ("Industrial Prod", "IP_CHNG_Index"),
    ("Durable Goods",   "DGNOCHNG_Index"),
    ("GDP QoQ",         "GDP_CQOQ_Index"),
    # Sentiment / inflation expectations
    ("UMich Sent",      "CONSSENT_Index"),
    ("UMich 1y Inflation",  "CONSPXMD_Index"),
    ("UMich 5y Inflation",  "CONSP5MD_Index"),
    ("Conf Board",      "CONCCONF_Index"),
    # Regional Fed
    ("Empire State",    "EMPRGBCI_Index"),
    ("Philly Fed",      "OUTFGAF_Index"),
    ("Chicago Fed Nat", "CFNAI_Index"),
    # Housing
    ("Housing Starts",  "NHSPATOT_Index"),
    ("Existing Home Sales", "ETSLTOTL_Index"),
]

# Treasury auction tickers — bid-to-cover ratios; "surprise" = deviation from
# trailing-12-auction rolling mean (no BBG survey median available for these).
# Each entry: (event_class, eco_ticker_filename_without_ext)
_AUCTION_TICKERS = [
    ("UST 2Y Auction (BTC)",  "USB2YBC_Index"),
    ("UST 5Y Auction (BTC)",  "USB5YBC_Index"),
    ("UST 7Y Auction (BTC)",  "USB7YBC_Index"),
    ("UST 10Y Auction (BTC)", "USN10YBC_Index"),
    ("UST 30Y Auction (BTC)", "USBD30YB_Index"),
]

# Standard event-window horizons for post-event drift (trading days)
_DRIFT_HORIZONS_D = (1, 5, 20)


# =============================================================================
# 1-2. Identity + Pricing
# =============================================================================
def _section_identity(symbol: str, panel: dict) -> dict:
    asof = panel.get("asof")
    out = {"symbol": symbol, "kind": "outright"}
    if "-" in symbol:
        out["kind"] = "fly" if symbol.count("-") == 2 else "spread"
        return out
    parsed = parse_sra_outright(symbol)
    if parsed:
        year, month = parsed
        out["year"] = year
        out["month"] = month
        out["month_name"] = ["Jan","Feb","Mar","Apr","May","Jun",
                              "Jul","Aug","Sep","Oct","Nov","Dec"][month - 1]
    rp = reference_period(symbol)
    if rp:
        start, end = rp
        out["ref_start"] = start
        out["ref_end"] = end
        out["ref_days"] = (end - start).days
        if asof:
            mid_days = (start - asof).days + (end - start).days / 2.0
            out["tenor_months"] = float(mid_days) / 30.4375
    out["quarterly"] = is_quarterly(symbol)
    # pack membership
    odf = panel.get("outrights_df")
    if odf is not None and not odf.empty:
        for pack_name, leg_syms in compute_pack_groups(odf):
            if symbol in leg_syms:
                out["pack_name"] = pack_name
                out["pack_legs"] = list(leg_syms)
                out["pack_position"] = leg_syms.index(symbol) + 1
                break
    return out


def _section_pricing(symbol: str, panel: dict) -> dict:
    out = {}
    closes = panel.get("outright_close_panel", pd.DataFrame())
    if symbol not in closes.columns:
        return out
    cs = closes[symbol].dropna()
    if cs.empty:
        return out
    last_close = float(cs.iloc[-1])
    out["last_close"] = last_close
    out["implied_3m_fwd_yield_pct"] = 100.0 - last_close
    if len(cs) >= 2:
        prev_close = float(cs.iloc[-2])
        out["day_change_bp"] = (last_close - prev_close) * 100.0
        out["day_change_dollar"] = (last_close - prev_close) * 100.0 * 25.0  # $25/bp/contract
    diffs_bp = (cs.diff() * 100.0).dropna()
    if len(diffs_bp) >= 14:
        out["atr14_bp"] = float(diffs_bp.tail(14).abs().mean())
    if len(diffs_bp) >= 252:
        out["realized_vol_252d_bp"] = float(diffs_bp.tail(252).std(ddof=1))
    return out


# =============================================================================
# 3. Position on curve
# =============================================================================
def _section_position_on_curve(symbol: str, panel: dict) -> dict:
    """Returns the data needed to render a CMC chart with this contract's tenor
    highlighted: the latest CMC row + the contract's tenor in months.

    Spreads/flies have no single tenor — return contract list of tenors instead.
    """
    out = {}
    cmc = panel.get("cmc_panel", pd.DataFrame())
    asof = panel.get("asof")
    if cmc is None or cmc.empty or asof is None:
        return out
    # Latest CMC row by tenor
    last_row = cmc.iloc[-1].dropna()
    out["tenors_months"] = [int(t) for t in last_row.index]
    out["yields_bp"] = last_row.values.astype(float).tolist()
    out["asof"] = pd.Timestamp(cmc.index[-1]).date()
    # This contract's tenor on the curve
    legs = parse_legs(symbol, "SRA") or [symbol]
    leg_taus = []
    for leg in legs:
        t = _outright_tenor_months(leg, asof)
        if t is not None:
            leg_taus.append((leg, float(t)))
    out["leg_tenors"] = leg_taus
    # Yield at each leg's tenor by PCHIP-interpolating today's CMC
    if leg_taus and out["tenors_months"]:
        tenors_arr = np.asarray(out["tenors_months"], dtype=float)
        yields_arr = np.asarray(out["yields_bp"], dtype=float)
        leg_yields = []
        for leg, tau in leg_taus:
            try:
                y = float(_pchip_curve(tenors_arr, yields_arr, np.array([tau]))[0])
                leg_yields.append((leg, tau, y))
            except Exception:
                leg_yields.append((leg, tau, None))
        out["leg_yields"] = leg_yields
    return out


# =============================================================================
# 4. Curve-factor sensitivity
# =============================================================================
def _plain_english(loading: float, *, hi: float, mid: float) -> str:
    a = abs(loading)
    if a > hi:
        lvl = "HIGH"
    elif a > mid:
        lvl = "MODERATE"
    else:
        lvl = "LOW"
    direction = "+" if loading >= 0 else "-"
    return f"{lvl} ({direction})"


def _section_factor_sensitivity(symbol: str, panel: dict) -> dict:
    out = {}
    fit = panel.get("pca_fit_static")
    asof = panel.get("asof")
    if fit is None or asof is None:
        return out
    strategy = "outright" if "-" not in symbol else (
        "fly" if symbol.count("-") == 2 else "spread")
    load_inst = _instrument_loadings(symbol, strategy, fit, asof)
    if load_inst is None:
        return out
    out["loadings"] = {f"PC{k+1}": float(load_inst[k]) for k in range(min(3, len(load_inst)))}
    out["plain"] = {
        "level":     _plain_english(load_inst[0], hi=0.30, mid=0.15),
        "slope":     _plain_english(load_inst[1], hi=0.20, mid=0.10),
        "curvature": _plain_english(load_inst[2], hi=0.10, mid=0.05),
    }
    return out


# =============================================================================
# 5. Recent stretch
# =============================================================================
def _section_recent_stretch(symbol: str, panel: dict) -> dict:
    out = {"by_lookback": {}}
    closes = panel.get("outright_close_panel", pd.DataFrame())
    asof = panel.get("asof")
    if symbol not in closes.columns or asof is None:
        return out
    cs = closes[symbol].dropna()
    if len(cs) <= 30:
        return out
    for n in (5, 15, 30, 60, 90, 252):
        if len(cs) > n:
            zv = zscore_value(cs, asof, int(n))
            if zv is not None and np.isfinite(zv):
                out["by_lookback"][int(n)] = float(zv)
    pct = percentile_rank_value(cs, asof, 252)
    if pct is not None and np.isfinite(pct):
        out["pct_252d"] = float(pct)
    return out


# =============================================================================
# 6. Stationarity diagnostics — ADF + KPSS + VR + OU κ/μ/σ + Hurst + HL
# =============================================================================
def _kpss_test(series: pd.Series, asof: date,
                lookback: int = 90, lag: int = 10) -> dict:
    """Hand-rolled KPSS test for level stationarity (no trend).

    Null: series is level-stationary. Reject if test stat > critical 5% (≈0.463).
    Critical values (Kwiatkowski 1992): 10%=0.347, 5%=0.463, 2.5%=0.574, 1%=0.739.

    Returns {tstat, reject_5pct (means non-stationary), n_obs}.
    """
    out = {"tstat": None, "reject_5pct": None, "n_obs": 0}
    if series is None or series.empty:
        return out
    ts = pd.Timestamp(asof)
    history = series.loc[series.index < ts].tail(lookback).dropna()
    n = len(history)
    if n < 20:
        return out
    x = history.values.astype(float)
    mu = x.mean()
    e = x - mu
    s = np.cumsum(e)
    s2 = float((s ** 2).sum() / (n ** 2))
    # Long-run variance σ̂²(l) using Bartlett kernel
    gamma0 = float((e ** 2).sum() / n)
    lr_var = gamma0
    for h in range(1, min(lag, n) + 1):
        gamma_h = float((e[h:] * e[:-h]).sum() / n)
        w = 1.0 - h / (lag + 1.0)
        lr_var += 2.0 * w * gamma_h
    if lr_var <= 0 or not np.isfinite(lr_var):
        return out
    tstat = s2 / lr_var
    out["tstat"] = float(tstat)
    out["reject_5pct"] = bool(tstat > 0.463)
    out["n_obs"] = int(n)
    return out


def _variance_ratio_test(series: pd.Series, asof: date,
                          lookback: int = 90, q: int = 4) -> dict:
    """Lo-MacKinlay variance ratio test. VR(q) = Var[r_q] / (q · Var[r_1]).

    Random walk → VR(q) ≈ 1. VR < 1 → mean reverting; VR > 1 → trending.
    Returns {vr, z_homoscedastic, reject_5pct (means non-RW)}.
    """
    out = {"vr": None, "z_stat": None, "reject_5pct": None, "n_obs": 0}
    if series is None or series.empty:
        return out
    ts = pd.Timestamp(asof)
    history = series.loc[series.index < ts].tail(lookback).dropna()
    n = len(history)
    if n < max(40, 2 * q + 2):
        return out
    x = history.values.astype(float)
    rets = np.diff(x)
    if len(rets) < q + 5:
        return out
    var_1 = float(rets.var(ddof=1))
    if var_1 <= 0 or not np.isfinite(var_1):
        return out
    # q-period overlapping returns:  R_q[t] = sum r[t..t+q-1]
    q_rets = np.array([rets[i:i + q].sum() for i in range(len(rets) - q + 1)])
    var_q = float(q_rets.var(ddof=1))
    vr = var_q / (q * var_1)
    out["vr"] = float(vr)
    # Homoscedastic asymptotic z under RW null:
    nq = len(rets)
    if nq <= q:
        return out
    var_vr = (2.0 * (2.0 * q - 1.0) * (q - 1.0)) / (3.0 * q * nq)
    if var_vr <= 0 or not np.isfinite(var_vr):
        return out
    z = (vr - 1.0) / np.sqrt(var_vr)
    out["z_stat"] = float(z)
    out["reject_5pct"] = bool(abs(z) > 1.96)
    out["n_obs"] = int(nq)
    return out


def _ou_kappa_mu_sigma(series: pd.Series, asof: date,
                        lookback: int = 90) -> dict:
    """OU calibration: dx = κ(μ - x)dt + σ dW.

    Discrete AR(1):  x_t = a + b·x_{t-1} + ε,  with b = e^{-κΔt}, a = μ(1 - b),
    σ²_ε = σ² · (1 - b²) / (2κ).  Δt = 1 trading day.
    Returns {kappa, mu, sigma, halflife_d, b_ar1, var_resid}.
    """
    out = {"kappa": None, "mu": None, "sigma": None,
           "halflife_d": None, "b_ar1": None, "var_resid": None}
    if series is None or series.empty:
        return out
    ts = pd.Timestamp(asof)
    history = series.loc[series.index < ts].tail(lookback).dropna()
    if len(history) < 20:
        return out
    x = history.values.astype(float)
    x_prev = x[:-1]
    x_curr = x[1:]
    A = np.column_stack([np.ones(len(x_prev)), x_prev])
    try:
        coefs, *_ = np.linalg.lstsq(A, x_curr, rcond=None)
    except Exception:
        return out
    a = float(coefs[0])
    b = float(coefs[1])
    if not np.isfinite(b) or b <= 0 or b >= 1:
        return out
    resid = x_curr - (a + b * x_prev)
    var_resid = float(resid.var(ddof=2))
    kappa = -np.log(b)
    mu = a / (1.0 - b)
    sigma2 = var_resid * 2.0 * kappa / (1.0 - b * b)
    if sigma2 <= 0 or not np.isfinite(sigma2):
        sigma = float("nan")
    else:
        sigma = float(np.sqrt(sigma2))
    halflife = -np.log(2.0) / np.log(b)
    out.update({
        "kappa": float(kappa), "mu": float(mu), "sigma": float(sigma),
        "halflife_d": float(halflife), "b_ar1": float(b),
        "var_resid": float(var_resid),
    })
    return out


def _section_stationarity(symbol: str, panel: dict) -> dict:
    out = {}
    closes = panel.get("outright_close_panel", pd.DataFrame())
    asof = panel.get("asof")
    if symbol not in closes.columns or asof is None:
        return out
    cs = closes[symbol].dropna()
    if len(cs) < 30:
        return out
    out["adf"] = adf_test(cs, asof, 90, 1)
    out["kpss"] = _kpss_test(cs, asof, 90, lag=10)
    out["vr_q4"] = _variance_ratio_test(cs, asof, 90, q=4)
    out["vr_q12"] = _variance_ratio_test(cs, asof, 90, q=12)
    h = hurst_exponent(cs, asof, 90)
    out["hurst"] = float(h) if h is not None and np.isfinite(h) else None
    out["hurst_label"] = hurst_label(h)
    hl = ou_half_life(cs, asof, 90)
    out["half_life_d"] = float(hl) if hl is not None and np.isfinite(hl) else None
    out["ou"] = _ou_kappa_mu_sigma(cs, asof, 90)
    return out


# =============================================================================
# 7. Three FV views — historical / analog / path / carry-roll
# =============================================================================
def _section_fv_views(symbol: str, panel: dict) -> dict:
    """Per-contract fair-value views.

    historical_z : z of last_close on the trailing-252d rolling distribution
    analog_fv    : A2 KNN-analog FV band on per-outright residual (change-space)
    path_fv      : A3 path-conditional restriction (uses # FOMC moves in window
                    as policy proxy — simple and resilient when policy_path is sparse)
    carry_roll   : back-of-envelope per-contract carry vs. roll into next quarterly
    """
    out = {}
    closes = panel.get("outright_close_panel", pd.DataFrame())
    asof = panel.get("asof")
    fit = panel.get("pca_fit_static")
    pc_panel = panel.get("pc_panel", pd.DataFrame())
    if asof is None:
        return out
    # ---- 7.1 historical z on raw close
    if symbol in closes.columns:
        cs = closes[symbol].dropna()
        if len(cs) > 30:
            zv = zscore_value(cs, asof, 60)
            pct = percentile_rank_value(cs, asof, 252)
            if zv is not None and np.isfinite(zv):
                out["historical_z_60d"] = float(zv)
            if pct is not None and np.isfinite(pct):
                out["historical_pct_252d"] = float(pct)

    # ---- 7.2 analog FV on per-outright residual series (change-space)
    if (fit is not None and not pc_panel.empty
            and symbol in closes.columns and "PC1" in pc_panel.columns):
        try:
            res_series = _per_outright_residual_series(symbol, panel)
            if res_series is not None and len(res_series) > 60:
                today_res = float(res_series.iloc[-1]) if len(res_series) > 0 else None
                feats = pc_panel[["PC1", "PC2", "PC3"]].dropna()
                # Align residual & features histories (drop today's row)
                feats_hist = feats.loc[feats.index < pd.Timestamp(asof)]
                if len(feats_hist) > 60:
                    X_hist = feats_hist.values.astype(float)
                    feats_today = feats.iloc[-1].values.astype(float) if not feats.empty else None
                    if feats_today is not None:
                        sigma_lw, _, _ = ledoit_wolf_shrinkage(X_hist)
                        sigma_inv = np.linalg.pinv(sigma_lw)
                        knn_res = knn_analog_search(
                            features_today=feats_today,
                            features_history=X_hist,
                            asof_today=asof,
                            asof_history=list(feats_hist.index.date),
                            k=50, time_decay_h=250.0,
                            exclusion_d=60, sigma_inv=sigma_inv,
                        )
                        fv = analog_fv_band(
                            structure_residual_series=res_series,
                            knn_result=knn_res,
                            asof_history=list(feats_hist.index.date),
                            today_residual=today_res,
                            eff_n_floor=EFF_N_FLOOR,
                        )
                        out["analog_fv"] = {
                            "fv_bp": fv.fv_bp, "lo_bp": fv.band_lo_bp,
                            "hi_bp": fv.band_hi_bp,
                            "today_bp": fv.residual_today_bp,
                            "z": fv.residual_z, "pct": fv.percentile_rank,
                            "eff_n": fv.eff_n, "n_analogs": fv.n_analogs,
                            "gate": fv.gate_quality,
                        }
        except Exception as e:
            out["analog_fv_error"] = str(e)

    # ---- 7.3 path FV — full A3 5-bucket policy lattice driven by A4
    # Heitfield-Park step-path bootstrap on the live CMC.  Falls back via
    # the empirical-Markov path inside `path_conditional_fv` when the
    # policy_path_history is too sparse.
    fomc_dates = panel.get("fomc_calendar_dates", []) or []
    if fomc_dates and "analog_fv" in out:
        try:
            res_series = _per_outright_residual_series(symbol, panel)
            cmc = panel.get("cmc_panel", pd.DataFrame())
            sofr = panel.get("sofr_panel", pd.DataFrame())
            pc_panel = panel.get("pc_panel", pd.DataFrame())
            if (res_series is not None and len(res_series) > 90
                    and cmc is not None and not cmc.empty
                    and pc_panel is not None and not pc_panel.empty):
                fomc_dates_clean = sorted(
                    pd.Timestamp(d).date() if not isinstance(d, date) else d
                    for d in fomc_dates
                )
                # ---- A4 Heitfield-Park step-path probabilities for today
                # σ=37.5bp = 1.5 × standard step → kernel spans into adjacent
                # buckets so probs aren't all-or-nothing in low-conviction regimes
                step_path = fit_step_path_bootstrap(
                    cmc_panel=cmc, fomc_dates=fomc_dates_clean,
                    asof=asof, n_meetings=8, n_draws=20,
                    kernel_sigma_bp=37.5,
                )
                today_probs = step_path["today_path_probs_combined"]
                # ---- Build policy_path_history for A3's empirical Markov fallback
                policy_hist = build_policy_path_history_from_fdtr(
                    sofr, fomc_dates_clean,
                )
                # ---- KNN over (PC1, PC2, PC3) features
                feats = pc_panel[["PC1", "PC2", "PC3"]].dropna()
                feats_hist = feats.loc[feats.index < pd.Timestamp(asof)]
                if len(feats_hist) > 60:
                    X_hist = feats_hist.values.astype(float)
                    feats_today = feats.iloc[-1].values.astype(float)
                    sigma_lw, _, _ = ledoit_wolf_shrinkage(X_hist)
                    sigma_inv = np.linalg.pinv(sigma_lw)
                    knn_res = knn_analog_search(
                        features_today=feats_today,
                        features_history=X_hist,
                        asof_today=asof,
                        asof_history=list(feats_hist.index.date),
                        k=50, time_decay_h=250.0,
                        exclusion_d=60, sigma_inv=sigma_inv,
                    )
                    today_residual = (float(res_series.iloc[-1])
                                       if len(res_series) else None)
                    # path_conditional_fv expects fomc_calendar as a DataFrame
                    # with `decision_date` column — wrap our list:
                    fomc_cal_df = pd.DataFrame({"decision_date": fomc_dates_clean})
                    pcfv = path_conditional_fv(
                        analog_result=knn_res,
                        structure_residual_series=res_series,
                        asof_history=list(feats_hist.index.date),
                        structure_window_end=asof,
                        fomc_calendar=fomc_cal_df,
                        policy_path_history=policy_hist,
                        today_policy_path_probs=today_probs,
                        today_residual=today_residual,
                    )
                    # Compute today's z & percentile vs path-FV: PathConditionalFV
                    # exposes the bucket-mixture FV but not the today-z directly.
                    fv_val = pcfv.fv_bp
                    band_w = (pcfv.band_hi_bp - pcfv.band_lo_bp) / 2.0
                    z_today = ((today_residual - fv_val) / band_w
                                 if (today_residual is not None and band_w > 0
                                     and np.isfinite(today_residual)
                                     and np.isfinite(fv_val))
                                 else None)
                    out["path_fv"] = {
                        "method": step_path["method"],
                        "fv_bp": pcfv.fv_bp,
                        "lo_bp": pcfv.band_lo_bp,
                        "hi_bp": pcfv.band_hi_bp,
                        "today_bp": today_residual,
                        "z": z_today,
                        "eff_n": pcfv.eff_n_overall,
                        "gate": pcfv.gate_quality,
                        "sparse_bucket_fallback": pcfv.sparse_bucket_fallback_used,
                        "today_path_probs": today_probs,
                        "bucket_probs": pcfv.bucket_probs,
                        "bucket_fv": pcfv.bucket_fv,
                        "bucket_eff_n": pcfv.bucket_eff_n,
                        "step_path_meetings": step_path["meetings"][:6],
                    }
        except Exception as e:
            out["path_fv_error"] = str(e)

    # ---- 7.4 carry-roll: carry-into-next-quarterly per-contract
    out["carry_roll"] = _carry_roll_for_outright(symbol, panel)

    # ---- 7.5 A10 turn / QE / YE adjuster: per-contract turn premium
    sofr_panel = panel.get("sofr_panel", pd.DataFrame())
    if sofr_panel is not None and not sofr_panel.empty and "-" not in symbol:
        try:
            adj = turn_adjustment_for_contract(symbol, sofr_panel,
                                                  lookback_n_turns=5)
            out["turn_adjustment"] = adj
        except Exception as e:
            out["turn_adjustment_error"] = str(e)

    return out


def _per_outright_residual_series(symbol: str, panel: dict) -> Optional[pd.Series]:
    """Reconstruct the per-outright change-space residual series.

    Same construction as `lib.pca.per_outright_residuals` but returns the full
    series instead of just today's stats. Used by FV-view analog computations.
    """
    fit = panel.get("pca_fit_static")
    closes = panel.get("outright_close_panel", pd.DataFrame())
    pc_panel = panel.get("pc_panel", pd.DataFrame())
    asof = panel.get("asof")
    if (fit is None or pc_panel is None or pc_panel.empty
            or symbol not in closes.columns or asof is None):
        return None
    yield_panel = _outright_yield_bp_panel(closes, [symbol])
    if yield_panel.empty or symbol not in yield_panel.columns:
        return None
    delta_yield = yield_panel[symbol].diff()
    pcs_aligned = pc_panel.reindex(yield_panel.index)
    load_inst = _instrument_loadings(symbol, "outright", fit, asof)
    if load_inst is None:
        return None
    tau = _outright_tenor_months(symbol, asof)
    if tau is None:
        mean_inst = 0.0
    else:
        mean_inst = float(_pchip_curve(
            np.asarray(fit.tenors_months, dtype=float),
            np.asarray(fit.feature_mean, dtype=float),
            np.array([tau]))[0])
    recon = np.full(len(yield_panel), float(mean_inst))
    for k in range(min(3, fit.loadings.shape[0])):
        col = f"PC{k + 1}"
        if col in pcs_aligned.columns:
            vals = pcs_aligned[col].values * load_inst[k]
            recon = recon + np.where(np.isnan(vals), 0.0, vals)
    residual = delta_yield.values - recon
    return pd.Series(residual, index=yield_panel.index).dropna()


def _carry_roll_for_outright(symbol: str, panel: dict) -> dict:
    """Per-contract carry-into-next-quarterly proxy.

    For an outright at tenor τ:
      carry_3m_bp = - (yield_at_τ - yield_at_(τ-3m))   [next contract is 3m closer]
      roll_3m_bp = current spread (next quarterly − this contract) in bp
      carry+roll = sum

    Sign convention: positive = trade earns yield holding it.
    """
    out = {}
    asof = panel.get("asof")
    cmc = panel.get("cmc_panel", pd.DataFrame())
    closes = panel.get("outright_close_panel", pd.DataFrame())
    if asof is None or cmc is None or cmc.empty or "-" in symbol:
        return out
    tau = _outright_tenor_months(symbol, asof)
    if tau is None:
        return out
    last_row = cmc.iloc[-1].dropna()
    if last_row.empty:
        return out
    tenors_arr = np.asarray([int(t) for t in last_row.index], dtype=float)
    yields_arr = last_row.values.astype(float)
    try:
        y_now = float(_pchip_curve(tenors_arr, yields_arr, np.array([tau]))[0])
        y_3m = float(_pchip_curve(tenors_arr, yields_arr, np.array([max(tau - 3.0, tenors_arr.min())]))[0])
        out["carry_3m_bp"] = -(y_now - y_3m)
    except Exception:
        pass
    # roll: spread to next-quarterly outright (3 months later)
    if symbol in closes.columns:
        # Find next quarterly contract
        parsed = parse_sra_outright(symbol)
        if parsed:
            year, month = parsed
            new_month = month + 3
            new_year = year
            if new_month > 12:
                new_month -= 12
                new_year += 1
            month_letter = {3: "H", 6: "M", 9: "U", 12: "Z"}.get(new_month)
            if month_letter:
                next_sym = f"SRA{month_letter}{str(new_year)[-2:]}"
                if next_sym in closes.columns:
                    cur_close = float(closes[symbol].dropna().iloc[-1])
                    nxt_close = float(closes[next_sym].dropna().iloc[-1])
                    # roll = price gap → yield gap. price↑ = yield↓; convention: + = roll-down (good)
                    out["roll_3m_bp"] = (nxt_close - cur_close) * 100.0
                    out["roll_into_symbol"] = next_sym
    if "carry_3m_bp" in out and "roll_3m_bp" in out:
        out["carry_plus_roll_3m_bp"] = out["carry_3m_bp"] + out["roll_3m_bp"]
    return out


# =============================================================================
# 8. Trade ideas referencing this contract
# =============================================================================
def _section_active_ideas(symbol: str, panel: dict, ideas: list) -> list:
    """Return a list of dicts — one per active idea involving this contract —
    each containing the idea + SR3-price conversion + factor interpretations
    + 90-day chart data. The renderer consumes this directly."""
    from lib.pca_trade_interpretation import (
        convert_to_sr3_prices,
        interpret_factors,
        build_recent_chart_data,
    )
    out = []
    # 90-day chart for the CONTRACT itself (used as the canvas for all trades
    # involving this contract). Cache on panel to avoid re-computation per idea.
    chart_cache_key = f"_chart90d_{symbol}"
    if chart_cache_key not in panel:
        try:
            panel[chart_cache_key] = build_recent_chart_data(symbol, panel,
                                                                lookback_days=90)
        except Exception:
            panel[chart_cache_key] = {"dates": [], "prices": [],
                                         "fv_prices": [], "residuals_bp": []}
    chart_data = panel[chart_cache_key]
    for idea in ideas or []:
        involves = any(leg.symbol == symbol for leg in idea.legs)
        if not involves:
            continue
        try:
            sr3 = convert_to_sr3_prices(idea, panel)
        except Exception as e:
            sr3 = None
        try:
            factors = interpret_factors(idea, panel)
        except Exception as e:
            factors = []
        out.append({
            "idea": idea,
            "sr3_prices": sr3,
            "factor_interpretations": factors,
            "chart_data_90d": chart_data,
        })
    return out


# =============================================================================
# 9. Per-contract event sensitivity
# =============================================================================
def _read_bbg_parquet_robust(category: str, ticker_filename: str) -> Optional[pd.DataFrame]:
    """Compatibility shim — delegates to lib.connections.read_bbg_parquet_robust.

    Kept under the underscore-prefixed name for backwards compatibility with
    callers inside this module. New callers should import directly from
    lib.connections.
    """
    from lib.connections import read_bbg_parquet_robust
    return read_bbg_parquet_robust(category, ticker_filename)


def _load_eco_surprise_series(eco_filename: str) -> Optional[pd.DataFrame]:
    """Load one ECO ticker parquet and produce a clean (release_date, surprise) frame.

    surprise = ACTUAL_RELEASE - BN_SURVEY_MEDIAN, indexed by ECO_RELEASE_DT.
    """
    df = _read_bbg_parquet_robust("eco", eco_filename)
    if df is None or df.empty:
        return None
    if "ACTUAL_RELEASE" not in df.columns or "BN_SURVEY_MEDIAN" not in df.columns:
        return None
    if "ECO_RELEASE_DT" not in df.columns:
        return None
    df = df.dropna(subset=["ACTUAL_RELEASE", "BN_SURVEY_MEDIAN", "ECO_RELEASE_DT"])
    if df.empty:
        return None
    # ECO_RELEASE_DT is YYYYMMDD float
    df = df.copy()
    df["release_date"] = pd.to_datetime(df["ECO_RELEASE_DT"].astype(int).astype(str),
                                          format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["release_date"])
    df["surprise"] = df["ACTUAL_RELEASE"] - df["BN_SURVEY_MEDIAN"]
    return df[["release_date", "surprise", "ACTUAL_RELEASE", "BN_SURVEY_MEDIAN"]]


def _safe_ols(x: np.ndarray, y: np.ndarray) -> dict:
    """OLS with closed-form CIs. Returns dict with {beta, se, tstat, r2, n}."""
    out = {"beta": None, "se": None, "tstat": None, "r2": None, "n": int(len(x))}
    n = len(x)
    if n < 5:
        return out
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return out
    x = x[mask]
    y = y[mask]
    sx = x.std(ddof=1)
    sy = y.std(ddof=1)
    if sx <= 0 or sy <= 0:
        return out
    A = np.column_stack([np.ones(len(x)), x])
    try:
        coefs, *_ = np.linalg.lstsq(A, y, rcond=None)
    except Exception:
        return out
    beta = float(coefs[1])
    yhat = A @ coefs
    resid = y - yhat
    sse = float((resid ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - sse / max(sst, 1e-12)
    sigma2 = sse / max(len(y) - 2, 1)
    XtX_inv = np.linalg.pinv(A.T @ A)
    se = float(np.sqrt(sigma2 * XtX_inv[1, 1]))
    tstat = beta / se if se > 0 else float("nan")
    return {"beta": beta, "se": se, "tstat": tstat, "r2": float(r2),
            "n": int(len(x))}


def _section_event_sensitivity(symbol: str, panel: dict) -> dict:
    """For each macro surprise: regress next-day Δ_close (bp) on standardized surprise.

    Returns {event_class: {beta_bp_per_sigma, se_bp, tstat, r2, n_events,
                              ci95_lo, ci95_hi, post_drift_5d_bp, post_drift_20d_bp}}
    """
    out = {}
    closes = panel.get("outright_close_panel", pd.DataFrame())
    asof = panel.get("asof")
    if symbol not in closes.columns or asof is None:
        return out
    cs = closes[symbol].dropna()
    if len(cs) < 60:
        return out
    diffs_bp = (cs.diff() * 100.0).rename("dy_bp")
    # FOMC sensitivity uses panel's fomc_calendar_dates (no surprise series — use 0/1 dummy + sign)
    fomc_dates = panel.get("fomc_calendar_dates", []) or []
    fomc_in_sample = sorted(pd.Timestamp(d).date() for d in fomc_dates
                            if d <= asof and d >= cs.index.min().date())
    if fomc_in_sample:
        # Just summarize: avg next-day Δ (bp) on FOMC days, std, and post-event drift
        rows = []
        for d in fomc_in_sample:
            d_ts = pd.Timestamp(d)
            try:
                # Next trading-day return
                pos = cs.index.searchsorted(d_ts)
                if pos + 1 < len(cs):
                    next_dy = (cs.iloc[pos + 1] - cs.iloc[pos]) * 100.0
                    rows.append(float(next_dy))
            except Exception:
                pass
        if rows:
            arr = np.asarray(rows, dtype=float)
            out["FOMC"] = {
                "n_events": int(len(arr)),
                "mean_dy_bp": float(arr.mean()),
                "std_dy_bp": float(arr.std(ddof=1)) if len(arr) > 1 else None,
                "abs_mean_dy_bp": float(np.abs(arr).mean()),
            }
            # 5d, 20d post-drift
            for h in (5, 20):
                drifts = []
                for d in fomc_in_sample:
                    d_ts = pd.Timestamp(d)
                    pos = cs.index.searchsorted(d_ts)
                    if pos + h < len(cs):
                        drifts.append(float((cs.iloc[pos + h] - cs.iloc[pos]) * 100.0))
                if drifts:
                    out["FOMC"][f"post_drift_{h}d_bp"] = float(np.mean(drifts))

    for event_class, eco_file in _SURPRISE_TICKERS:
        sf = _load_eco_surprise_series(eco_file)
        if sf is None or len(sf) < 5:
            continue
        # Pair (release_date, surprise) → next-day Δ_close
        pairs = []
        for _, row in sf.iterrows():
            rd = row["release_date"].date()
            try:
                pos = cs.index.searchsorted(pd.Timestamp(rd))
                if 0 <= pos < len(cs) and pos + 1 < len(cs):
                    next_dy = (cs.iloc[pos + 1] - cs.iloc[pos]) * 100.0
                    pairs.append((float(row["surprise"]), float(next_dy)))
            except Exception:
                pass
        if len(pairs) < 5:
            continue
        x_arr = np.array([p[0] for p in pairs])
        y_arr = np.array([p[1] for p in pairs])
        # Standardize surprise to σ-units so β has units bp / σ
        sx = x_arr.std(ddof=1)
        if sx <= 0:
            continue
        x_std = x_arr / sx
        reg = _safe_ols(x_std, y_arr)
        if reg["beta"] is None:
            continue
        ci_half = 1.96 * (reg["se"] or 0.0)
        out[event_class] = {
            "beta_bp_per_sigma": reg["beta"],
            "se_bp": reg["se"],
            "tstat": reg["tstat"],
            "r2": reg["r2"],
            "n_events": reg["n"],
            "ci95_lo": reg["beta"] - ci_half,
            "ci95_hi": reg["beta"] + ci_half,
            "abs_mean_dy_bp": float(np.abs(y_arr).mean()),
        }
        # post-drift
        for h in (5, 20):
            drifts = []
            for _, row in sf.iterrows():
                rd = row["release_date"].date()
                try:
                    pos = cs.index.searchsorted(pd.Timestamp(rd))
                    if 0 <= pos < len(cs) and pos + h < len(cs):
                        drifts.append(float((cs.iloc[pos + h] - cs.iloc[pos]) * 100.0))
                except Exception:
                    pass
            if drifts:
                out[event_class][f"post_drift_{h}d_bp"] = float(np.mean(drifts))

    # Treasury auctions — surprise = bid-to-cover deviation from trailing-12 rolling mean
    for event_class, eco_file in _AUCTION_TICKERS:
        sf = _load_auction_surprise_series(eco_file)
        if sf is None or len(sf) < 5:
            continue
        pairs = []
        for _, row in sf.iterrows():
            rd = row["release_date"].date()
            try:
                pos = cs.index.searchsorted(pd.Timestamp(rd))
                if 0 <= pos < len(cs) and pos + 1 < len(cs):
                    next_dy = (cs.iloc[pos + 1] - cs.iloc[pos]) * 100.0
                    pairs.append((float(row["surprise"]), float(next_dy)))
            except Exception:
                pass
        if len(pairs) < 5:
            continue
        x_arr = np.array([p[0] for p in pairs])
        y_arr = np.array([p[1] for p in pairs])
        sx = x_arr.std(ddof=1)
        if sx <= 0:
            continue
        x_std = x_arr / sx
        reg = _safe_ols(x_std, y_arr)
        if reg["beta"] is None:
            continue
        ci_half = 1.96 * (reg["se"] or 0.0)
        out[event_class] = {
            "beta_bp_per_sigma": reg["beta"],
            "se_bp": reg["se"],
            "tstat": reg["tstat"],
            "r2": reg["r2"],
            "n_events": reg["n"],
            "ci95_lo": reg["beta"] - ci_half,
            "ci95_hi": reg["beta"] + ci_half,
            "abs_mean_dy_bp": float(np.abs(y_arr).mean()),
        }
        for h in (5, 20):
            drifts = []
            for _, row in sf.iterrows():
                rd = row["release_date"].date()
                try:
                    pos = cs.index.searchsorted(pd.Timestamp(rd))
                    if 0 <= pos < len(cs) and pos + h < len(cs):
                        drifts.append(float((cs.iloc[pos + h] - cs.iloc[pos]) * 100.0))
                except Exception:
                    pass
            if drifts:
                out[event_class][f"post_drift_{h}d_bp"] = float(np.mean(drifts))
    return out


def _load_auction_surprise_series(eco_filename: str) -> Optional[pd.DataFrame]:
    """Load a Treasury auction parquet and return (release_date, surprise) frame.

    Auction "surprise" = bid-to-cover deviation from trailing-12-auction rolling mean
    (no Bloomberg survey series exists for auction bid-to-cover).  Larger positive
    surprise = stronger demand than recent average.
    """
    df = _read_bbg_parquet_robust("eco", eco_filename)
    if df is None or df.empty:
        return None
    if "ACTUAL_RELEASE" not in df.columns or "ECO_RELEASE_DT" not in df.columns:
        return None
    df = df.dropna(subset=["ACTUAL_RELEASE", "ECO_RELEASE_DT"]).copy()
    if df.empty:
        return None
    df["release_date"] = pd.to_datetime(df["ECO_RELEASE_DT"].astype(int).astype(str),
                                          format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["release_date"]).sort_values("release_date").reset_index(drop=True)
    if len(df) < 6:
        return None
    # Rolling mean of bid-to-cover (trailing-12 max, min-periods 4 for early sample),
    # surprise = current - mean.  Want at least ~6 valid surprises.
    rolling_mean = df["ACTUAL_RELEASE"].rolling(12, min_periods=4).mean().shift(1)
    df["surprise"] = df["ACTUAL_RELEASE"] - rolling_mean
    df = df.dropna(subset=["surprise"])
    if len(df) < 4:
        return None
    return df[["release_date", "surprise", "ACTUAL_RELEASE"]]


# =============================================================================
# 10. Calendar — next events + days to expiry + week-of-year context
# =============================================================================
def _section_calendar(symbol: str, panel: dict) -> dict:
    out = {"events": []}
    asof = panel.get("asof")
    if asof is None:
        return out
    fomc = panel.get("fomc_calendar_dates", []) or []
    upcoming_fomc = sorted(pd.Timestamp(d).date() for d in fomc
                            if pd.Timestamp(d).date() > asof)[:3]
    for d in upcoming_fomc:
        out["events"].append({"event": "FOMC", "date": d, "days_out": (d - asof).days})
    # CPI release dates from CPI_YOY ECO_RELEASE_DT
    cpi_rel = _load_eco_surprise_series("CPI_YOY_Index")
    if cpi_rel is not None:
        upcoming = cpi_rel[cpi_rel["release_date"].dt.date > asof].head(2)
        for _, row in upcoming.iterrows():
            d = row["release_date"].date()
            out["events"].append({"event": "CPI YoY", "date": d, "days_out": (d - asof).days})
    nfp_rel = _load_eco_surprise_series("NFP_TCH_Index")
    if nfp_rel is not None:
        upcoming = nfp_rel[nfp_rel["release_date"].dt.date > asof].head(2)
        for _, row in upcoming.iterrows():
            d = row["release_date"].date()
            out["events"].append({"event": "NFP", "date": d, "days_out": (d - asof).days})
    out["events"].sort(key=lambda x: x["days_out"])
    out["events"] = out["events"][:6]

    # Days to last-trading
    if "-" not in symbol:
        rp = reference_period(symbol)
        if rp:
            start, _ = rp
            last_trade = start - timedelta(days=1)
            out["days_to_expiry"] = (last_trade - asof).days
            out["last_trade_date"] = last_trade
    # Seasonality context
    season = panel.get("seasonality_results", {})
    anchor_season = season.get("Anchor", {}) if isinstance(season, dict) else {}
    if isinstance(anchor_season, dict) and "betas" in anchor_season:
        # week-of-asof context
        wk = pd.Timestamp(asof).isocalendar().week
        out["asof_week_of_year"] = int(wk)
    return out


# =============================================================================
# 11. Momentum — TSM signs at 21/63/126/252d
# =============================================================================
def _section_momentum(symbol: str, panel: dict) -> dict:
    out = {"by_horizon": {}}
    closes = panel.get("outright_close_panel", pd.DataFrame())
    asof = panel.get("asof")
    if symbol not in closes.columns or asof is None:
        return out
    cs = closes[symbol].dropna()
    if cs.empty:
        return out
    last = float(cs.iloc[-1])
    for h in (21, 63, 126, 252):
        if len(cs) > h:
            past = float(cs.iloc[-(h + 1)])
            ret_bp = (last - past) * 100.0
            sign = 1 if ret_bp > 0 else -1 if ret_bp < 0 else 0
            out["by_horizon"][h] = {"ret_bp": ret_bp, "sign": sign}
    if out["by_horizon"]:
        signs = [v["sign"] for v in out["by_horizon"].values()]
        out["consensus"] = "long" if all(s > 0 for s in signs) else (
            "short" if all(s < 0 for s in signs) else "mixed")
    return out


# =============================================================================
# 12. Cycle / regime context
# =============================================================================
def _section_regime(symbol: str, panel: dict) -> dict:
    out = {}
    asof = panel.get("asof")
    cycle = panel.get("cycle_phase", ("—", {}))
    out["cycle_phase"] = cycle[0] if isinstance(cycle, tuple) else str(cycle)
    rs = panel.get("regime_stack", {}) or {}
    hmm = rs.get("hmm_fit") if rs else None
    if hmm is not None and hmm.smoothed_labels is not None and len(hmm.smoothed_labels):
        out["regime_label"] = int(hmm.smoothed_labels[-1])
        out["regime_confidence"] = float(hmm.dominant_confidence[-1])
    bp_breaks = rs.get("bai_perron_breaks", []) if rs else []
    if bp_breaks and asof is not None:
        last_bp = bp_breaks[-1]
        try:
            out["days_since_break"] = (asof - last_bp).days
            out["last_break_date"] = last_bp
        except Exception:
            pass
    bocpd_series = rs.get("bocpd_p_change_series") if rs else None
    if bocpd_series is not None and len(bocpd_series):
        try:
            out["bocpd_p_today"] = float(bocpd_series.iloc[-1])
        except Exception:
            pass
    return out


# =============================================================================
# 13. Hedge candidates — PC1-isolated 4-leg baskets that include this contract
# =============================================================================
def _section_hedge_candidates(symbol: str, panel: dict, max_n: int = 6) -> list:
    out = []
    asof = panel.get("asof")
    fit = panel.get("pca_fit_static")
    closes = panel.get("outright_close_panel", pd.DataFrame())
    outright_symbols = panel.get("outright_symbols", [])
    if (asof is None or fit is None or "-" in symbol or
            symbol not in outright_symbols):
        return out
    # Enumerate full PC1 basket set, then filter to baskets that include this contract
    all_baskets = enumerate_pc1_synthetics(outright_symbols, asof)
    relevant = [b for b in all_baskets if symbol in b]
    if not relevant:
        return out
    yield_panel = _outright_yield_bp_panel(closes, outright_symbols)
    for basket in relevant[:max_n]:
        load_subset = []
        for sym in basket:
            li = _instrument_loadings(sym, "outright", fit, asof)
            if li is None:
                load_subset = []
                break
            load_subset.append(li)
        if not load_subset:
            continue
        L_sub = np.vstack(load_subset).T
        w = solve_isolated_pc_weights(L_sub, target_pc=1,
                                       pv01_legs=np.array([25.0] * len(basket)))
        if w is None:
            continue
        try:
            sub = yield_panel[list(basket)]
        except KeyError:
            continue
        res = (sub.values * w.reshape(1, -1)).sum(axis=1)
        res_series = pd.Series(res, index=sub.index).dropna()
        if len(res_series) < 30:
            continue
        z = zscore_value(res_series, asof, 60)
        adf = adf_test(res_series, asof, 60, 1)
        try:
            today_val = float(res_series.iloc[-1])
        except Exception:
            today_val = None
        # Normalize weights to max(|w|)=1 for human readability
        w_arr = np.asarray(w, dtype=float)
        max_abs_w = float(np.abs(w_arr).max()) if len(w_arr) else 1.0
        w_norm = w_arr / max_abs_w if max_abs_w > 0 else w_arr
        this_idx = basket.index(symbol)
        this_w_norm = float(w_norm[this_idx])
        out.append({
            "legs": list(basket),
            "weights_normalized": [float(x) for x in w_norm],
            "this_leg_idx": this_idx,
            "this_leg_weight_norm": this_w_norm,
            # Side: positive normalized weight on this leg ↔ "long the basket means buy this leg"
            "side_for_target": "buy" if this_w_norm > 0 else "sell",
            "z": float(z) if z is not None and np.isfinite(z) else None,
            "adf_pass": bool(adf.get("reject_5pct", False)),
        })
    # Sort by |z| descending so most-stretched baskets float to top
    out.sort(key=lambda x: -abs(x["z"] or 0.0))
    return out


# =============================================================================
# 14. Positioning — CFTC COT proxy (UST 2Y NCN)
# =============================================================================
def _section_positioning(panel: dict) -> dict:
    """UST 2Y net non-commercial — front-end speculator positioning proxy.

    Eurodollar legacy IMM13 series is empty in the warehouse (Eurodollars delisted
    Aug 2023, BBG no longer publishes COT for them). UST 2Y NCN is the cleanest
    available STIR proxy with full daily series.
    """
    out = {}
    asof = panel.get("asof")
    if asof is None:
        return out
    df = _read_bbg_parquet_robust("xcot", _COT_PROXY_TICKER)
    if df is None or df.empty or "PX_LAST" not in df.columns:
        return out
    df = df.dropna(subset=["PX_LAST"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")["PX_LAST"]
    if df.empty:
        return out
    df = df.loc[df.index <= pd.Timestamp(asof)]
    if df.empty:
        return out
    last_v = float(df.iloc[-1])
    last_dt = df.index[-1].date()
    out["last_value"] = last_v
    out["last_date"] = last_dt
    out["age_days"] = (asof - last_dt).days
    if len(df) >= 53:
        wk = df.tail(105)  # ~2y of weekly
        out["pct_2y"] = float((wk <= last_v).sum() / len(wk) * 100.0)
        out["z_2y"] = float((last_v - wk.mean()) / wk.std(ddof=1)) if wk.std(ddof=1) > 0 else None
    if len(df) >= 4:
        out["wow_change"] = float(last_v - df.iloc[-2])
    # Open Interest for context
    oi_df = _read_bbg_parquet_robust("xcot", _COT_PROXY_OI_TICKER)
    if oi_df is not None and not oi_df.empty and "PX_LAST" in oi_df.columns:
        try:
            oi = oi_df.dropna(subset=["PX_LAST"]).copy()
            oi["date"] = pd.to_datetime(oi["date"])
            oi = oi.sort_values("date").set_index("date")["PX_LAST"]
            oi = oi.loc[oi.index <= pd.Timestamp(asof)]
            if not oi.empty:
                out["open_interest"] = float(oi.iloc[-1])
                if out["open_interest"] > 0:
                    out["pct_of_oi"] = (last_v / out["open_interest"]) * 100.0
        except Exception:
            pass
    out["proxy_used"] = "UST 2Y (CBT42NCN) — Eurodollar IMM13 unavailable post-delisting"
    return out


# =============================================================================
# 15-16. Charts data + Execution profile
# =============================================================================
def _section_charts(symbol: str, panel: dict) -> dict:
    """Return raw arrays for the 4 charts. Actual rendering happens in tab layer."""
    out = {}
    closes = panel.get("outright_close_panel", pd.DataFrame())
    asof = panel.get("asof")
    if symbol in closes.columns:
        cs = closes[symbol].dropna().tail(252)
        if not cs.empty:
            out["price_series_dates"] = list(cs.index)
            out["price_series_values"] = cs.values.astype(float).tolist()
    # residual series for chart
    res_series = _per_outright_residual_series(symbol, panel)
    if res_series is not None and len(res_series) > 5:
        rs = res_series.tail(252)
        out["residual_dates"] = list(rs.index)
        out["residual_values"] = rs.values.astype(float).tolist()
        # OU bands on residual (κ·μ·σ)
        if asof is not None:
            ou = _ou_kappa_mu_sigma(res_series, asof, lookback=90)
            out["residual_ou"] = ou
    return out


def _section_execution_profile(symbol: str, panel: dict) -> dict:
    out = {}
    asof = panel.get("asof")
    if asof is None or "-" in symbol:
        return out
    try:
        from lib.connections import get_ohlc_connection
        con = get_ohlc_connection()
        if con is None:
            return out
        # Average daily volume over trailing 60 days
        sql_60 = f"""
            SELECT AVG(volume) AS avg_vol_60,
                   COUNT(*) AS n_days
            FROM mde2_timeseries
            WHERE symbol = '{symbol}' AND "interval" = '1D'
              AND DATE(to_timestamp(time/1000.0))
                  BETWEEN DATE '{(asof - timedelta(days=60)).isoformat()}'
                  AND DATE '{asof.isoformat()}'
        """
        row = con.execute(sql_60).fetchone()
        if row and row[0]:
            out["avg_vol_60d"] = float(row[0])
            out["n_days_60"] = int(row[1])
        # Today's volume
        sql_today = f"""
            SELECT volume FROM mde2_timeseries
            WHERE symbol = '{symbol}' AND "interval" = '1D'
              AND DATE(to_timestamp(time/1000.0)) = DATE '{asof.isoformat()}'
            LIMIT 1
        """
        row2 = con.execute(sql_today).fetchone()
        if row2 and row2[0]:
            out["today_vol"] = float(row2[0])
    except Exception:
        pass
    # Slippage estimates at multiple sizes
    avg_v = out.get("avg_vol_60d", 0)
    if avg_v > 0:
        sizes = [100, 500, 1000, 2500]
        out["slippage_bp_by_size"] = {}
        for s in sizes:
            # Linear-impact proxy: slip ≈ 0.05bp + 0.5 * (size / avg_vol) bp
            slip = 0.05 + 0.5 * (s / max(avg_v, 100))
            out["slippage_bp_by_size"][s] = float(round(slip, 3))
    return out


# =============================================================================
# 17. Quick actions: neighbor list (computed in renderer)
# 18. Provenance footer
# =============================================================================
def _section_provenance(symbol: str, panel: dict) -> dict:
    out = {}
    asof = panel.get("asof")
    fit = panel.get("pca_fit_static")
    rolling = panel.get("rolling_fits", {}) or {}
    pct_today = panel.get("reconstruction_pct_today")
    if pct_today is not None:
        out["recon_pct_today"] = float(pct_today)
    recon_err = panel.get("reconstruction_error", pd.Series(dtype=float))
    if isinstance(recon_err, pd.Series) and not recon_err.empty:
        try:
            today_err = float(recon_err.iloc[-1])
            pctile = float((recon_err <= today_err).sum() / len(recon_err) * 100.0)
            out["recon_err_today"] = today_err
            out["recon_err_percentile"] = pctile
        except Exception:
            pass
    if rolling:
        latest_refit = max(rolling.keys())
        out["latest_refit_date"] = latest_refit
        if asof is not None:
            try:
                out["refit_age_days"] = (asof - latest_refit).days
            except Exception:
                pass
        out["n_rolling_refits"] = len(rolling)
    cross = panel.get("cross_pc_corr", pd.DataFrame())
    if isinstance(cross, pd.DataFrame) and not cross.empty:
        try:
            last = cross.iloc[-1].dropna()
            if not last.empty:
                out["max_cross_pc_corr_today"] = float(last.abs().max())
        except Exception:
            pass
    print_alerts = panel.get("print_quality_alerts", []) or []
    today_print_flagged = False
    if asof is not None and print_alerts:
        today_print_flagged = any(pd.Timestamp(d).date() == asof for d in print_alerts)
    out["print_quality_today"] = not today_print_flagged
    if fit is not None:
        out["pc1_var_share"] = float(fit.variance_ratio[0]) if len(fit.variance_ratio) else None
        out["pc2_var_share"] = float(fit.variance_ratio[1]) if len(fit.variance_ratio) > 1 else None
        out["pc3_var_share"] = float(fit.variance_ratio[2]) if len(fit.variance_ratio) > 2 else None
        out["pca_n_obs"] = int(fit.n_obs)
    # Convexity warning: if symbol's tenor > 36M and PC3 loading large
    if "-" not in symbol and asof is not None and fit is not None:
        tau = _outright_tenor_months(symbol, asof)
        if tau is not None and tau > 36:
            li = _instrument_loadings(symbol, "outright", fit, asof)
            if li is not None and abs(li[2]) > 0.05:
                out["convexity_warning"] = True
            else:
                out["convexity_warning"] = False
        else:
            out["convexity_warning"] = False
    # Outlier today?
    outlier_days = panel.get("outlier_days", [])
    if asof is not None:
        out["outlier_today"] = any(pd.Timestamp(d).date() == asof for d in outlier_days)
    return out


# =============================================================================
# Z-heatmap matrix — raw close + residual z across 6 lookbacks
# =============================================================================
def _section_z_heatmap(symbol: str, panel: dict) -> dict:
    out = {}
    closes = panel.get("outright_close_panel", pd.DataFrame())
    asof = panel.get("asof")
    if symbol not in closes.columns or asof is None:
        return out
    cs = closes[symbol].dropna()
    if len(cs) <= 5:
        return out
    rows = []
    lookbacks = (5, 15, 30, 60, 90, 252)
    for lookback in lookbacks:
        r = {"lookback": lookback}
        if len(cs) > lookback:
            zv = zscore_value(cs, asof, int(lookback))
            r["close_z"] = float(zv) if zv is not None and np.isfinite(zv) else None
            pct = percentile_rank_value(cs, asof, int(lookback))
            r["close_pct"] = float(pct) if pct is not None and np.isfinite(pct) else None
        rows.append(r)
    out["rows"] = rows
    res_series = _per_outright_residual_series(symbol, panel)
    if res_series is not None and len(res_series) > 5:
        for r in out["rows"]:
            lookback = r["lookback"]
            if len(res_series) > lookback:
                zv = zscore_value(res_series, asof, int(lookback))
                r["residual_z"] = float(zv) if zv is not None and np.isfinite(zv) else None
    return out


# =============================================================================
# Top-level entry
# =============================================================================
def _section_cross_asset(panel: dict) -> dict:
    """Cross-asset overlay (vol regime, risk state, credit cycle, lead-lag, term-premia,
    convexity bias). Cached on panel as panel['cross_asset_analysis'] to avoid
    recomputation per dossier call.
    """
    cached = panel.get("cross_asset_analysis")
    if cached is not None:
        ca = cached
    else:
        try:
            asof = panel.get("asof", date.today())
            cap = load_cross_asset_panel(asof=asof, lookback_days=252,
                                            include_external=False)
            pc_panel = panel.get("pc_panel", pd.DataFrame())
            # Realized PC1 σ in bp (last 60d)
            sigma_pc1 = None
            if not pc_panel.empty and "PC1" in pc_panel.columns:
                pc1_last_60 = pc_panel["PC1"].dropna().tail(60)
                if len(pc1_last_60) >= 20:
                    sigma_pc1 = float(pc1_last_60.std(ddof=1))
            tenor_grid = panel.get("tenor_grid_months", [])
            # Front-yield panel for convexity bias (research fix:
            # use front-SR3 yield σ, not PC1 score σ).
            yield_panel = panel.get("yield_bp_panel", pd.DataFrame())
            front_yield_panel = (yield_panel.iloc[:, :1]
                                  if isinstance(yield_panel, pd.DataFrame)
                                      and not yield_panel.empty
                                  else None)
            ca = run_all_cross_asset_analyses(
                cap, pc_panel=pc_panel,
                realized_pc1_sigma_bp=sigma_pc1,
                tenor_grid_months=tenor_grid,
                front_yield_panel=front_yield_panel,
            )
            panel["cross_asset_analysis"] = ca
            panel["cross_asset_panel"] = cap
        except Exception as e:
            return {"error": str(e)}
    out = {
        "vol_regime": ca.vol_regime,
        "risk_state": ca.risk_state,
        "credit_cycle": ca.credit_cycle,
        "lead_lag_matrix": (ca.lead_lag.to_dict("records") if hasattr(ca.lead_lag, "to_dict") else []),
        "eq_rates_corr": ca.eq_rates_corr,
        "fx_differential": ca.fx_differential,
        "term_premia_decomp": ca.term_premia_decomp,
        "convexity_bias_today_bp": (float(ca.convexity_bias.iloc[-1]) if hasattr(ca.convexity_bias, "iloc") and len(ca.convexity_bias) else None),
    }
    return out


def _compute_trade_thesis(symbol: str, D: dict, panel: dict) -> dict:
    """Phase F.1 — synthesize a 3-line plain-English Trade Thesis from the dossier.

    Returns:
      {
        "side": "long" | "short" | "neutral",
        "headline": "SHORT candidate — +1.8σ rich vs analog FV ...",
        "supporting_signals": ["+1.8σ stretch on 60d", "OU 14d sweet spot", ...],
        "caveats": ["KPSS rejects level-stationary", "FOMC 12d out", ...],
        "confidence": 0.0..1.0,
      }
    """
    rs = D.get("recent_stretch", {})
    stat = D.get("stationarity", {})
    fv = D.get("fv_views", {})
    momentum = D.get("momentum", {})
    regime = D.get("regime", {})
    cal = D.get("calendar", {})
    cross = D.get("cross_asset", {})
    positioning = D.get("positioning", {})

    # Figure out side
    z_252 = rs.get("by_lookback", {}).get(252)
    z_60 = rs.get("by_lookback", {}).get(60)
    primary_z = z_252 if z_252 is not None else z_60

    side = "neutral"
    if primary_z is not None and abs(primary_z) > 1.0:
        side = "short" if primary_z > 0 else "long"

    # Build supporting signals + caveats
    supporting = []
    caveats = []

    if primary_z is not None and abs(primary_z) > 1.0:
        supporting.append(f"{primary_z:+.2f}σ stretched (252d basis)")

    # Stationarity
    if stat:
        adf_pass = stat.get("adf", {}).get("reject_5pct", False)
        kpss_reject = stat.get("kpss", {}).get("reject_5pct", None)
        hl = stat.get("half_life_d")
        if adf_pass and kpss_reject is False:
            supporting.append("ADF + KPSS confirm reverting")
        elif adf_pass and kpss_reject:
            caveats.append("KPSS rejects → drift component (don't size up)")
        elif not adf_pass:
            caveats.append("ADF fails → no mean-reversion edge")
        if hl is not None and 5 <= hl <= 30:
            supporting.append(f"OU half-life {hl:.0f}d (sweet spot)")
        elif hl is not None and hl > 60:
            caveats.append(f"OU very slow ({hl:.0f}d half-life)")

    # Momentum
    if momentum:
        cons = momentum.get("consensus", "mixed")
        if cons == "long" and side == "long":
            supporting.append("momentum supports LONG (all 4 horizons)")
        elif cons == "short" and side == "short":
            supporting.append("momentum supports SHORT (all 4 horizons)")
        elif cons in ("long", "short") and side != cons:
            caveats.append(f"momentum is {cons} (against the trade)")

    # Regime
    if regime:
        conf = regime.get("regime_confidence", 0)
        if conf >= 0.6:
            supporting.append(f"regime stable ({conf*100:.0f}% conf)")
        elif conf > 0:
            caveats.append(f"regime transitioning ({conf*100:.0f}% conf)")

    # Calendar — events in window
    events = cal.get("events", []) or []
    if events:
        first = events[0]
        d_out = first.get("days_out", 999)
        if d_out <= 14:
            caveats.append(f"{first.get('event','event')} in {d_out}d (event risk)")

    # Cross-asset context
    vol_regime = (cross.get("vol_regime") or {}).get("regime")
    risk_state = (cross.get("risk_state") or {}).get("state")
    if vol_regime == "stressed":
        caveats.append("vol regime: stressed (band ×1.4)")
    elif vol_regime == "crisis":
        caveats.append("vol regime: crisis (band ×2.0)")
    if risk_state == "risk_off":
        supporting.append("risk-off tape supports STIR rally bias")
    elif risk_state == "panic":
        caveats.append("panic tape — reduce sizing")

    # Positioning
    pct_2y = positioning.get("pct_2y") if positioning else None
    if pct_2y is not None:
        if pct_2y < 5 or pct_2y > 95:
            supporting.append(f"positioning extreme (pct {pct_2y:.0f}) — contrarian setup")

    # Build headline
    if side == "short":
        headline = f"<b>SHORT candidate</b> — {primary_z:+.2f}σ rich on 252d"
    elif side == "long":
        headline = f"<b>LONG candidate</b> — {primary_z:+.2f}σ cheap on 252d"
    else:
        headline = "<b>Neutral</b> — no statistically meaningful stretch"

    # Confidence: count of supporting vs caveats, normalized
    n_sup = len(supporting)
    n_cav = len(caveats)
    if side == "neutral":
        confidence = 0.0
    else:
        confidence = max(0.0, min(1.0, 0.4 + 0.1 * (n_sup - n_cav)))

    return {
        "side": side,
        "headline": headline,
        "supporting_signals": supporting,
        "caveats": caveats,
        "confidence": float(confidence),
        "primary_z": primary_z,
    }


def plain_english_stationarity(D: dict) -> str:
    """Phase F.2 — 1-line plain-English summary of section 6 stationarity."""
    s = D.get("stationarity", {})
    if not s:
        return "⚪ Stationarity diagnostics not available"
    adf_pass = s.get("adf", {}).get("reject_5pct", False)
    kpss_reject = s.get("kpss", {}).get("reject_5pct", None)
    vr_q4 = s.get("vr_q4", {})
    vr = vr_q4.get("vr") if vr_q4 else None
    hurst = s.get("hurst")
    hl = s.get("half_life_d")

    if adf_pass and kpss_reject is False and vr is not None and abs(vr - 1.0) > 0.10:
        hl_str = f"OU half-life {hl:.1f}d" if hl is not None else ""
        return f"🟢 <b>Reverts cleanly</b> — ADF+KPSS+VR all confirm mean-reverting · {hl_str}"
    if adf_pass and kpss_reject:
        return "🟡 <b>Mixed signal</b> — ADF passes but KPSS rejects → drift component present"
    if not adf_pass and vr is not None and abs(vr - 1.0) <= 0.05:
        return "🔴 <b>Random walk</b> — no mean-reversion edge, momentum trades only"
    if hurst is not None and hurst > 0.55:
        return f"📈 <b>Trending</b> — Hurst {hurst:.2f} > 0.55, persistent regime"
    return "⚪ <b>Inconclusive</b> — diagnostic mix"


def plain_english_regime(D: dict) -> str:
    """Phase F.3 — 1-line plain-English regime summary."""
    r = D.get("regime", {})
    if not r:
        return "Regime data unavailable"
    phase = r.get("cycle_phase", "—")
    label_map = {
        "early-cut": "Early-cut, accelerating",
        "mid-cut": "Mid-cut, decelerating",
        "late-cut": "Late-cut, near-terminal",
        "trough": "Trough, awaiting hike cycle",
        "early-hike": "Early-hike, ramping",
        "mid-hike": "Mid-hike, mature",
        "late-hike": "Late-hike, terminal nearing",
        "peak": "Peak, hold-pause",
    }
    phase_label = label_map.get(phase, phase)
    conf = r.get("regime_confidence", 0)
    stability = "stable" if conf >= 0.8 else "transitioning"
    days = r.get("days_since_break")
    days_str = f"{days}d run" if days is not None else "?"
    return f"<b>{phase_label}</b> · {stability} ({conf*100:.0f}% conf) · {days_str}"


def plain_english_provenance(D: dict) -> str:
    """Phase F.5 — 1-line provenance health summary."""
    p = D.get("provenance", {})
    if not p:
        return "Provenance unavailable"
    issues = []
    if p.get("recon_pct_today") is not None and p["recon_pct_today"] < 0.92:
        issues.append(f"recon {p['recon_pct_today']*100:.0f}% (< 92%)")
    if p.get("max_cross_pc_corr_today") is not None and p["max_cross_pc_corr_today"] > 0.30:
        issues.append(f"cross-PC corr {p['max_cross_pc_corr_today']:.2f}")
    if p.get("refit_age_days") is not None and p["refit_age_days"] > 5:
        issues.append(f"refit {p['refit_age_days']}d old")
    if p.get("outlier_today"):
        issues.append("outlier day flagged")
    if p.get("convexity_warning"):
        issues.append("convexity warning")
    if not issues:
        return (f"🟢 <b>Engine clean today</b>: 3-PC={p.get('recon_pct_today',0)*100:.0f}% · "
                 f"refit {p.get('refit_age_days', 0)}d · cross-PC {p.get('max_cross_pc_corr_today', 0):.2f}")
    return "🟡 <b>Engine caveats</b>: " + " · ".join(issues)


def plain_english_positioning(D: dict, trade_side: Optional[str] = None) -> str:
    """Phase F.4 — bullish/bearish interpretation of CFTC positioning."""
    p = D.get("positioning", {})
    if not p:
        return "Positioning data unavailable"
    pct = p.get("pct_2y")
    z = p.get("z_2y")
    if pct is None:
        return "Positioning history insufficient"
    if pct < 5:
        contrarian = "(contrarian setup ✓)" if trade_side == "long" else "(crowded short)"
        return f"<b>Spec NCN extreme short</b> · pct {pct:.0f} · z {z:+.2f}σ {contrarian}"
    if pct > 95:
        contrarian = "(contrarian setup ✓)" if trade_side == "short" else "(crowded long)"
        return f"<b>Spec NCN extreme long</b> · pct {pct:.0f} · z {z:+.2f}σ {contrarian}"
    return f"Spec NCN moderate · pct {pct:.0f} · z {z:+.2f}σ"


def plain_english_cross_asset(D: dict) -> str:
    """Phase F.7 — 1-line cross-asset tape summary."""
    cross = D.get("cross_asset", {})
    if not cross:
        return "Cross-asset data unavailable"
    vol = cross.get("vol_regime", {})
    risk = cross.get("risk_state", {})
    cred = cross.get("credit_cycle", {})

    parts = []
    if vol.get("regime") and vol["regime"] != "unknown":
        parts.append(f"vol <b>{vol['regime']}</b> (composite z={vol.get('composite_z', 0):.2f})")
    if risk.get("state") and risk["state"] != "neutral":
        parts.append(f"risk <b>{risk['state']}</b>")
    if cred.get("recession_prob_4w") is not None:
        rp = cred["recession_prob_4w"]
        if rp > 0.30:
            parts.append(f"recession prob 4w <b>{rp*100:.0f}%</b>")
    if not parts:
        return "Cross-asset tape: balanced"
    return " · ".join(parts)


def contract_dossier_data(symbol: str, panel: dict, ideas: list = None) -> dict:
    """Build the full 18-section dossier data dict for one contract.

    NEW (Phase A.4 + F): adds `cross_asset` and `trade_thesis` sections.
    """
    if ideas is None:
        ideas = []
    D = {
        "identity":          _section_identity(symbol, panel),
        "pricing":           _section_pricing(symbol, panel),
        "position_on_curve": _section_position_on_curve(symbol, panel),
        "factor_sensitivity": _section_factor_sensitivity(symbol, panel),
        "recent_stretch":    _section_recent_stretch(symbol, panel),
        "stationarity":      _section_stationarity(symbol, panel),
        "fv_views":          _section_fv_views(symbol, panel),
        "active_ideas":      _section_active_ideas(symbol, panel, ideas),
        "event_sensitivity": _section_event_sensitivity(symbol, panel),
        "calendar":          _section_calendar(symbol, panel),
        "momentum":          _section_momentum(symbol, panel),
        "regime":            _section_regime(symbol, panel),
        "hedge_candidates":  _section_hedge_candidates(symbol, panel),
        "positioning":       _section_positioning(panel),
        "charts":            _section_charts(symbol, panel),
        "execution":         _section_execution_profile(symbol, panel),
        "z_heatmap":         _section_z_heatmap(symbol, panel),
        "provenance":        _section_provenance(symbol, panel),
        "cross_asset":       _section_cross_asset(panel),
    }
    # Phase F.1 — Trade Thesis card data (synthesizes from above sections)
    D["trade_thesis"] = _compute_trade_thesis(symbol, D, panel)
    # Plain-English layers (Phase F.2-F.7)
    D["plain_english"] = {
        "stationarity":  plain_english_stationarity(D),
        "regime":        plain_english_regime(D),
        "provenance":    plain_english_provenance(D),
        "positioning":   plain_english_positioning(D, D["trade_thesis"]["side"]),
        "cross_asset":   plain_english_cross_asset(D),
    }
    return D
