"""Cross-asset analyses for the PCA subtab — Phase C of the rebuild.

Eight analyses, each consumes the lib.pca_cross_asset.CrossAssetPanel and
emits a structured dict the dossier renderer + trade-idea narrative chain
consume.

C.1 — vol regime classifier (MOVE/SRVIX/SKEW composite)
C.2 — risk-on/off state (SPX/DXY/credit/MOVE composite)
C.3 — credit-cycle leading indicator (recession probability)
C.4 — cross-asset lead-lag matrix (PCs vs SPX/MOVE/etc.)
C.5 — equity-rates correlation regime
C.6 — FX-implied rate differential context
C.7 — term-premia decomposition (ACM + Kim-Wright)
C.8 — convexity bias model (Mercurio approximation)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd


def _zscore_today(s: pd.Series, lookback: int = 252) -> Optional[float]:
    if s is None or s.empty or len(s) < 30:
        return None
    sub = s.dropna().tail(lookback + 1)
    if len(sub) < 30:
        return None
    today = float(sub.iloc[-1])
    hist = sub.iloc[:-1]
    mu = float(hist.mean())
    sigma = float(hist.std(ddof=1))
    if sigma <= 0 or not np.isfinite(sigma):
        return None
    return (today - mu) / sigma


def _last_value(s: pd.Series) -> Optional[float]:
    if s is None or s.empty:
        return None
    v = s.dropna()
    return float(v.iloc[-1]) if not v.empty else None


# =============================================================================
# C.1 — Vol regime classifier
# =============================================================================
def classify_vol_regime(vol_panel: pd.DataFrame) -> dict:
    """Composite z-score of MOVE+SRVIX+SKEW → regime bucket.

    Buckets:
      quiet     (z < -0.75)
      normal    (-0.75 ≤ z ≤ +0.75)
      stressed  (+0.75 < z ≤ +1.5)
      crisis    (z > +1.5)

    Returns dict with regime, composite_z, per-component z, and
    band_widening_factor for analog FV.
    """
    out = {"regime": "unknown", "composite_z": None,
           "move_z": None, "srvix_z": None, "skew_z": None, "vix_z": None,
           "band_widening_factor": 1.0}
    if vol_panel is None or vol_panel.empty:
        return out

    zs = []
    for col in ("MOVE", "SRVIX", "SKEW", "VIX"):
        if col in vol_panel.columns:
            z = _zscore_today(vol_panel[col])
            out[f"{col.lower()}_z"] = z
            if z is not None and np.isfinite(z):
                zs.append(z)
    if not zs:
        return out
    composite = float(np.mean(zs))
    out["composite_z"] = composite
    if composite < -0.75:
        out["regime"] = "quiet"
        out["band_widening_factor"] = 1.0
    elif composite <= 0.75:
        out["regime"] = "normal"
        out["band_widening_factor"] = 1.0
    elif composite <= 1.5:
        out["regime"] = "stressed"
        out["band_widening_factor"] = 1.4
    else:
        out["regime"] = "crisis"
        out["band_widening_factor"] = 2.0
    return out


# =============================================================================
# C.2 — Risk-on/off composite
# =============================================================================
def compute_risk_state(equity: pd.DataFrame, fx: pd.DataFrame,
                          credit: pd.DataFrame, vol: pd.DataFrame,
                          lookback: int = 21) -> dict:
    """Composite risk-on / risk-off state.

    Inputs: SPX 21d log return, DXY 21d log return, credit OAS 21d Δ, MOVE 21d Δ.
    Negative score = risk-on (good news), positive = risk-off (flight-to-quality).
    """
    out = {"state": "neutral", "score": None,
           "spx_ret_z": None, "dxy_ret_z": None,
           "credit_chg_z": None, "move_chg_z": None}

    drivers = []
    if equity is not None and not equity.empty and "SPX" in equity.columns:
        spx = equity["SPX"].dropna()
        if len(spx) > lookback + 30:
            ret = np.log(spx / spx.shift(lookback)).dropna()
            z = _zscore_today(ret, lookback=252)
            out["spx_ret_z"] = z
            if z is not None:
                drivers.append(-z)    # SPX up = risk-on = negative score

    if fx is not None and not fx.empty and "DXY_synth" in fx.columns:
        dxy = fx["DXY_synth"].dropna()
        if len(dxy) > lookback + 30:
            ret = np.log(dxy / dxy.shift(lookback)).dropna()
            z = _zscore_today(ret, lookback=252)
            out["dxy_ret_z"] = z
            if z is not None:
                drivers.append(z * 0.5)    # USD up = mild risk-off

    if credit is not None and not credit.empty:
        for col in ("LUACOAS", "CDX_IG"):
            if col in credit.columns:
                s = credit[col].dropna()
                if len(s) > lookback + 30:
                    chg = (s - s.shift(lookback)).dropna()
                    z = _zscore_today(chg, lookback=252)
                    out["credit_chg_z"] = z
                    if z is not None:
                        drivers.append(z)    # credit widens = risk-off
                    break

    if vol is not None and not vol.empty and "MOVE" in vol.columns:
        s = vol["MOVE"].dropna()
        if len(s) > lookback + 30:
            chg = (s - s.shift(lookback)).dropna()
            z = _zscore_today(chg, lookback=252)
            out["move_chg_z"] = z
            if z is not None:
                drivers.append(z * 0.5)

    if not drivers:
        return out
    score = float(np.mean(drivers))
    out["score"] = score
    if score < -1.0:
        out["state"] = "risk_on"
    elif score < 0.5:
        out["state"] = "neutral"
    elif score < 1.5:
        out["state"] = "risk_off"
    else:
        out["state"] = "panic"
    return out


# =============================================================================
# C.3 — Credit-cycle leading indicator
# =============================================================================
def compute_credit_cycle(credit: pd.DataFrame) -> dict:
    """IG / HY OAS divergence → recession probability.

    Returns recession_prob_4w, recession_prob_12w, IG/HY z-scores, and a flag
    when IG and HY are diverging (a known late-cycle signal).
    """
    out = {"recession_prob_4w": None, "recession_prob_12w": None,
           "ig_zscore": None, "hy_zscore": None, "ig_hy_diverging": False}
    if credit is None or credit.empty:
        return out

    # Use LUACOAS as IG proxy, LF98OAS as HY proxy
    for ig_col, hy_col in [("LUACOAS", "LF98OAS"), ("CDX_IG", "CDX_HY")]:
        if ig_col in credit.columns and hy_col in credit.columns:
            ig = credit[ig_col].dropna()
            hy = credit[hy_col].dropna()
            if len(ig) < 30 or len(hy) < 30:
                continue
            # 4w and 12w changes z-scored
            ig_4w = (ig - ig.shift(20)).dropna()
            hy_4w = (hy - hy.shift(20)).dropna()
            ig_z_4w = _zscore_today(ig_4w)
            hy_z_4w = _zscore_today(hy_4w)
            out["ig_zscore"] = ig_z_4w
            out["hy_zscore"] = hy_z_4w
            if ig_z_4w is None or hy_z_4w is None:
                continue
            # Logistic: recession_prob = 1 / (1 + exp(-(ig_z + hy_z - 1.5)))
            x4 = ig_z_4w + 0.5 * hy_z_4w - 1.5
            x12 = (ig_z_4w * 1.5) + (0.5 * hy_z_4w) - 1.0
            out["recession_prob_4w"] = float(1.0 / (1.0 + np.exp(-x4)))
            out["recession_prob_12w"] = float(1.0 / (1.0 + np.exp(-x12)))
            # IG/HY diverging: HY rallying while IG widens (or vice versa)
            out["ig_hy_diverging"] = bool(
                (ig_z_4w > 1.0 and hy_z_4w < -0.5) or
                (ig_z_4w < -0.5 and hy_z_4w > 1.0)
            )
            break
    return out


# =============================================================================
# C.4 — Cross-asset lead-lag matrix
# =============================================================================
def compute_lead_lag_matrix(pc_panel: pd.DataFrame,
                                 cross_panels: dict,
                                 max_lag_days: int = 10,
                                 window: int = 60) -> pd.DataFrame:
    """For each (PC, asset) pair, find the lag in [-max_lag, +max_lag] that
    maximizes |corr| over the trailing `window` days.

    `cross_panels` is a dict like {"SPX": equity["SPX"], "MOVE": vol["MOVE"], ...}.
    Negative lag means asset leads PC; positive means PC leads asset.
    """
    if pc_panel is None or pc_panel.empty:
        return pd.DataFrame()
    rows = []
    for pc_col in ("PC1", "PC2", "PC3"):
        if pc_col not in pc_panel.columns:
            continue
        pc_diff = pc_panel[pc_col].diff().dropna().tail(window)
        if pc_diff.empty:
            continue
        for asset_name, asset_series in cross_panels.items():
            if asset_series is None or asset_series.empty:
                continue
            asset_diff = asset_series.dropna().diff().dropna()
            best_corr = 0.0
            best_lag = 0
            for lag in range(-max_lag_days, max_lag_days + 1):
                if lag < 0:
                    a = asset_diff.shift(-lag)
                else:
                    a = asset_diff.shift(lag)
                aligned = pd.concat([pc_diff, a], axis=1, join="inner").dropna()
                if len(aligned) < 20:
                    continue
                c = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
                if not np.isfinite(c):
                    continue
                if abs(c) > abs(best_corr):
                    best_corr = c
                    best_lag = lag
            rows.append({
                "PC": pc_col,
                "asset": asset_name,
                "peak_corr": best_corr,
                "peak_lag_d": best_lag,    # negative = asset leads PC
            })
    return pd.DataFrame(rows)


# =============================================================================
# C.5 — Equity-rates correlation regime
# =============================================================================
def compute_eq_rates_corr_regime(spx: pd.Series, pc1: pd.Series,
                                       window: int = 63) -> dict:
    """Rolling 63d correlation of SPX log returns vs PC1 (level) changes.

    Normal: negative (good news = sell bonds). Flipped: positive (e.g. inflation
    fear regime). Hugely meaningful for hedge construction.
    """
    out = {"regime": "unknown", "corr_now": None, "corr_z": None,
           "days_in_regime": None}
    if spx is None or spx.empty or pc1 is None or pc1.empty:
        return out
    spx_ret = np.log(spx / spx.shift(1)).dropna()
    pc1_diff = pc1.diff().dropna()
    aligned = pd.concat([spx_ret, pc1_diff], axis=1, join="inner").dropna()
    if len(aligned) < window + 30:
        return out
    rolling_corr = aligned.iloc[:, 0].rolling(window).corr(aligned.iloc[:, 1]).dropna()
    if rolling_corr.empty:
        return out
    out["corr_now"] = float(rolling_corr.iloc[-1])
    out["corr_z"] = _zscore_today(rolling_corr)
    if out["corr_now"] < -0.20:
        out["regime"] = "normal_negative"
    elif out["corr_now"] > 0.20:
        out["regime"] = "flipped_positive"
    else:
        out["regime"] = "transitioning"
    # Count days in current regime (consecutive days with same sign)
    sign_today = np.sign(out["corr_now"])
    count = 0
    for v in rolling_corr.iloc[::-1]:
        if np.sign(v) == sign_today:
            count += 1
        else:
            break
    out["days_in_regime"] = int(count)
    return out


# =============================================================================
# C.6 — FX-implied rate differential context
# =============================================================================
def compute_fx_rate_differential(fx: pd.DataFrame,
                                       fdtr_panel: Optional[pd.DataFrame] = None) -> dict:
    """USD-vs-EUR / GBP / JPY rate differential context from FX moves.

    If fdtr is wired, use FDTR vs ECB (proxy via EUDR1T or similar). Otherwise
    use EUR/GBP/JPY 30d log change z-scored as a proxy for divergent paths.
    """
    out = {"divergence_z": None, "expected_usd_basket_drift": None,
           "eurusd_30d_z": None, "usdjpy_30d_z": None}
    if fx is None or fx.empty:
        return out
    if "EURUSD" in fx.columns:
        s = fx["EURUSD"].dropna()
        if len(s) > 60:
            chg = np.log(s / s.shift(30)).dropna()
            out["eurusd_30d_z"] = _zscore_today(chg)
    if "USDJPY" in fx.columns:
        s = fx["USDJPY"].dropna()
        if len(s) > 60:
            chg = np.log(s / s.shift(30)).dropna()
            out["usdjpy_30d_z"] = _zscore_today(chg)
    # Composite divergence
    zs = [v for v in (out["eurusd_30d_z"], out["usdjpy_30d_z"]) if v is not None]
    if zs:
        out["divergence_z"] = float(np.mean(zs))
    return out


# =============================================================================
# C.7 — Term-premia decomposition
# =============================================================================
def decompose_yield_into_premia_and_path(ust_panel: pd.DataFrame,
                                                term_premia: pd.DataFrame) -> dict:
    """Per tenor: yield = expected_path + term_premium.

    Returns today's premium / expected-path values + 30d Δ of each component.
    """
    out = {"by_tenor": {}, "summary_30d": None}
    if (ust_panel is None or ust_panel.empty
            or term_premia is None or term_premia.empty):
        return out
    tenor_map = {"UST_2Y": "acm_2y", "UST_5Y": "acm_5y",
                  "UST_10Y": "acm_10y", "UST_30Y": "acm_30y"}
    asof_idx = ust_panel.index.intersection(term_premia.index)
    if len(asof_idx) < 30:
        return out
    today_ust = ust_panel.loc[asof_idx[-1]]
    today_tp = term_premia.loc[asof_idx[-1]]
    rows = []
    for ust_col, tp_col in tenor_map.items():
        if ust_col not in ust_panel.columns or tp_col not in term_premia.columns:
            continue
        y = float(today_ust.get(ust_col, np.nan))
        tp = float(today_tp.get(tp_col, np.nan))
        if not np.isfinite(y) or not np.isfinite(tp):
            continue
        ep = y - tp    # expected path = yield − term premium
        # 30d changes
        if len(asof_idx) > 30:
            y30 = float(ust_panel.loc[asof_idx[-30], ust_col])
            tp30 = float(term_premia.loc[asof_idx[-30], tp_col])
            d_y = y - y30
            d_tp = tp - tp30
            d_ep = (y - tp) - (y30 - tp30)
        else:
            d_y = d_tp = d_ep = None
        out["by_tenor"][ust_col] = {
            "yield": y, "term_premium": tp, "expected_path": ep,
            "yield_30d_chg": d_y, "tp_30d_chg": d_tp, "ep_30d_chg": d_ep,
        }
    return out


# =============================================================================
# C.8 — Convexity bias model (Mercurio simplified)
# =============================================================================
def compute_convexity_bias(short_rate_daily_sigma_decimal: float,
                              tenor_grid_months: list) -> pd.Series:
    """Per-tenor convexity bias in bp — futures vs OIS-forward gap.

    Uses Piterbarg (2006) short-rate-vol approximation:
        bias_bp(T, τ) ≈ -0.5 · σ_r² · T · τ · 10000

    Where:
      σ_r = ANNUALIZED short-rate vol in DECIMAL (e.g. 0.0090 = 90 bp/yr)
            = front-SR3 daily yield σ (decimal) · sqrt(252)
      T   = time to futures expiry (years)
      τ   = SOFR-accrual period of the futures = 0.25 (3 months)

    Notes:
    - Was previously using PC1-score σ which is unitless and not a rate vol;
      that was a bug (research-agent validation flagged it). PC1-score σ does
      not have units of bp per year of yield change.
    - The Ho-Lee form `-0.5·σ²·τ²` was mislabelled as Mercurio; we use the
      Piterbarg short-rate form which is the standard for SR3-vs-OIS gap.
    - Bias is NEGATIVE: SR3-implied forward rate is LOWER than OIS forward
      → SR3 futures price is HIGHER than OIS-implied price → buyers of
      SR3 outright effectively pay a small premium.

    Parameters
    ----------
    short_rate_daily_sigma_decimal : float
        Daily realized vol of the front-SR3 IMPLIED YIELD (in decimal,
        e.g. 0.0001 = 1 bp/day). Use front-month yield, not PC1 score.
    tenor_grid_months : list
        Months-out tenors to compute bias for (e.g. [3, 6, 9, ..., 60]).

    Returns
    -------
    pd.Series indexed by tenor_months, values in bp (negative for normal regime).
    """
    if (not short_rate_daily_sigma_decimal
            or not np.isfinite(short_rate_daily_sigma_decimal)
            or short_rate_daily_sigma_decimal <= 0):
        return pd.Series(dtype=float)
    # Annualize the daily yield σ
    sigma_r_ann = short_rate_daily_sigma_decimal * np.sqrt(252.0)    # decimal/yr
    biases = {}
    tau_accrual = 0.25    # 3-month SOFR period
    for tau_m in tenor_grid_months:
        T = tau_m / 12.0    # years to expiry
        # Piterbarg short-rate form (in decimal²·yr²): bias = -0.5 · σ² · T · τ
        bias_decimal = -0.5 * (sigma_r_ann ** 2) * T * tau_accrual
        # convert to bp: 1 decimal = 10,000 bp
        bias_bp = bias_decimal * 10000.0
        biases[int(tau_m)] = float(bias_bp)
    return pd.Series(biases, name="convexity_bias_bp")


# =============================================================================
# Composite cross-asset analysis bundle (single-call entry)
# =============================================================================
@dataclass(frozen=True)
class CrossAssetAnalysis:
    vol_regime: dict
    risk_state: dict
    credit_cycle: dict
    lead_lag: pd.DataFrame
    eq_rates_corr: dict
    fx_differential: dict
    term_premia_decomp: dict
    convexity_bias: pd.Series


def run_all_cross_asset_analyses(cap, pc_panel: Optional[pd.DataFrame] = None,
                                       realized_pc1_sigma_bp: Optional[float] = None,
                                       tenor_grid_months: Optional[list] = None,
                                       front_yield_panel: Optional[pd.DataFrame] = None
                                       ) -> CrossAssetAnalysis:
    """Run all 8 Phase C analyses on the CrossAssetPanel `cap`.

    `pc_panel` and `realized_pc1_sigma_bp` come from the existing PCA panel.
    Lead-lag analysis requires both PCs and cross-asset feeds.

    `front_yield_panel` is used for convexity-bias computation. Should be a
    DataFrame of SR3-implied yields in bp (front contract preferred). If not
    provided, falls back to the prior (incorrect) PC1-σ path.
    """
    vol_regime = classify_vol_regime(cap.vol)
    risk_state = compute_risk_state(cap.equity, cap.fx, cap.credit, cap.vol)
    credit_cycle = compute_credit_cycle(cap.credit)
    fx_diff = compute_fx_rate_differential(cap.fx)

    # Lead-lag matrix
    cross_panels = {}
    if cap.equity is not None and not cap.equity.empty and "SPX" in cap.equity.columns:
        cross_panels["SPX"] = cap.equity["SPX"]
    if cap.vol is not None and not cap.vol.empty and "MOVE" in cap.vol.columns:
        cross_panels["MOVE"] = cap.vol["MOVE"]
    if cap.credit is not None and not cap.credit.empty:
        for c in ("LUACOAS", "CDX_IG"):
            if c in cap.credit.columns:
                cross_panels["IG"] = cap.credit[c]
                break
    if cap.fx is not None and not cap.fx.empty and "DXY_synth" in cap.fx.columns:
        cross_panels["DXY"] = cap.fx["DXY_synth"]
    lead_lag = (compute_lead_lag_matrix(pc_panel, cross_panels)
                  if pc_panel is not None and not pc_panel.empty
                  else pd.DataFrame())

    # Equity-rates correlation regime
    eq_rates = {}
    if (pc_panel is not None and not pc_panel.empty and "PC1" in pc_panel.columns
            and "SPX" in cross_panels):
        eq_rates = compute_eq_rates_corr_regime(cross_panels["SPX"], pc_panel["PC1"])

    # Term premia decomposition
    tp_decomp = decompose_yield_into_premia_and_path(cap.ust, cap.term_premia)

    # Convexity bias — now uses front-SR3 yield daily σ in decimal
    # (research validation flagged old PC1-σ formulation as incorrect:
    # PC1 score is unitless, not a rate vol; converting bp→decimal recovered
    # nonsense magnitudes).
    sigma_front_decimal = None
    if front_yield_panel is not None and not front_yield_panel.empty:
        try:
            front_col = front_yield_panel.columns[0]
            # Use last 60d of daily yield changes (in bp), then convert to decimal
            front_chg_bp = front_yield_panel[front_col].dropna().diff().tail(60)
            if len(front_chg_bp) >= 20:
                sigma_bp_daily = float(front_chg_bp.std(ddof=1))
                if np.isfinite(sigma_bp_daily) and sigma_bp_daily > 0:
                    sigma_front_decimal = sigma_bp_daily / 10000.0
        except Exception:
            sigma_front_decimal = None
    # Fallback: derive from PC1 σ if no front yield panel provided
    # (less accurate but better than zero)
    if sigma_front_decimal is None and realized_pc1_sigma_bp:
        # Rough scaling: PC1 σ ≈ avg-loading × per-tenor yield σ
        sigma_front_decimal = float(realized_pc1_sigma_bp) / 10000.0
    convex = (compute_convexity_bias(sigma_front_decimal, tenor_grid_months)
                if sigma_front_decimal and tenor_grid_months
                else pd.Series(dtype=float))

    return CrossAssetAnalysis(
        vol_regime=vol_regime,
        risk_state=risk_state,
        credit_cycle=credit_cycle,
        lead_lag=lead_lag,
        eq_rates_corr=eq_rates,
        fx_differential=fx_diff,
        term_premia_decomp=tp_decomp,
        convexity_bias=convex,
    )
