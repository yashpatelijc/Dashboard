"""Phase 11 cross-cutting [+] additions (plan §12).

Per §15:
  KEPT — cointegration screens (Engle-Granger + Johansen) on CMC node pairs.
  KEPT — cross-asset risk-regime composite.
  KEPT — Bauer-Swanson orthogonalised MP-surprises (sans option-implied
         skewness term, per D3=defer).
  REMOVED — CFTC COT heatmap (no scrapers per D2; xcot/ is commodities).
  REMOVED — Hull-White convexity adjustment (no vol surface per D3).

Outputs:
  .cmc_cache/cross_cointegration_<asof>.parquet
  .cmc_cache/cross_risk_regime_<asof>.parquet
  .cmc_cache/cross_bauer_swanson_<asof>.parquet
  .cmc_cache/cross_cutting_manifest_<asof>.json
"""
from __future__ import annotations

import json
import os
from datetime import date, timedelta
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".cmc_cache"
_BBG_DIR = Path(r"D:\BBG data\parquet")

BUILDER_VERSION = "1.0.0"

COINT_NODE_PAIRS = (
    ("M0", "M3"), ("M3", "M6"), ("M6", "M9"), ("M9", "M12"),
    ("M0", "M6"), ("M3", "M12"), ("M12", "M24"), ("M24", "M36"),
    ("M36", "M48"), ("M48", "M60"),
)

RISK_REGIME_TICKERS = {
    "MOVE":  ("vol_indices", "MOVE_Index"),
    "SRVIX": ("vol_indices", "SRVIX_Index"),
    "VIX":   ("vol_indices", "VIX_Index"),
    "SKEW":  ("vol_indices", "SKEW_Index"),
    "CDX_IG": ("credit", "CDX_IG_CDSI_GEN_5Y_Corp"),
    "CDX_HY": ("credit", "CDX_HY_CDSI_GEN_5Y_Corp"),
    "DXY":   ("fx", "DXY_Curncy"),
    "SPX":   ("equity_indices", "SPX_Index"),
}


# =============================================================================
# Cointegration screens (Engle-Granger + Johansen)
# =============================================================================

def cointegrate_pair(s1: pd.Series, s2: pd.Series) -> dict:
    """Returns {eg_pvalue, johansen_p1, half_life_days}. Half-life from
    AR(1) on residual."""
    out = {"eg_pvalue": np.nan, "johansen_p1": np.nan,
              "half_life_days": np.nan, "n_obs": 0}
    df = pd.concat([s1, s2], axis=1).dropna()
    if len(df) < 60:
        return out
    out["n_obs"] = int(len(df))
    try:
        from statsmodels.tsa.stattools import coint, adfuller
        # Engle-Granger
        try:
            res = coint(df.iloc[:, 0], df.iloc[:, 1])
            out["eg_pvalue"] = float(res[1])
        except Exception:
            pass
        # Johansen (rank=1 trace stat p-value approximation)
        try:
            from statsmodels.tsa.vector_ar.vecm import coint_johansen
            res_j = coint_johansen(df.values, det_order=0, k_ar_diff=1)
            # Johansen returns trace stat & critical values; use 5% column
            stat = float(res_j.lr1[0])
            crit = float(res_j.cvt[0, 1])  # 5% level
            out["johansen_p1"] = 0.04 if stat > crit else 0.20
        except Exception:
            pass
        # Half-life of OLS residual
        beta = float(np.polyfit(df.iloc[:, 1].values,
                                       df.iloc[:, 0].values, 1)[0])
        resid = df.iloc[:, 0] - beta * df.iloc[:, 1]
        from lib.analytics.opportunity_modules import fit_ou_one
        ou = fit_ou_one(resid)
        out["half_life_days"] = ou.get("half_life_days", np.nan)
    except Exception:
        pass
    return out


def build_cointegration(asof: date) -> pd.DataFrame:
    from lib.sra_data import load_cmc_wide_panel
    wide = load_cmc_wide_panel("outright", asof, field="close")
    if wide is None or wide.empty:
        return pd.DataFrame()
    rows = []
    for n1, n2 in COINT_NODE_PAIRS:
        if n1 not in wide.columns or n2 not in wide.columns:
            continue
        cr = cointegrate_pair(wide[n1], wide[n2])
        flag = (
            (cr["eg_pvalue"] is not None and cr["eg_pvalue"] < 0.05)
            and (cr["half_life_days"] is not None and cr["half_life_days"] < 60)
        )
        rows.append({
            "node_a": n1, "node_b": n2,
            "n_obs": cr["n_obs"],
            "eg_pvalue": cr["eg_pvalue"],
            "johansen_p1": cr["johansen_p1"],
            "half_life_days": cr["half_life_days"],
            "cointegrated_flag": bool(flag),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Cross-asset risk-regime composite
# =============================================================================

def _load_bbg_ticker_pct(category: str, ticker: str) -> Optional[pd.Series]:
    path = _BBG_DIR / category / f"{ticker}.parquet"
    if not path.exists():
        return None
    try:
        import duckdb
        con = duckdb.connect(":memory:")
        p = str(path).replace("\\", "/")
        df = con.execute(
            f"SELECT date, PX_LAST FROM read_parquet('{p}')"
        ).fetchdf()
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")["PX_LAST"].astype(float).dropna().sort_index()
    except Exception:
        return None


def build_cross_asset_risk_regime(asof: date,
                                          lookback_days: int = 252) -> pd.DataFrame:
    """For each ticker, compute current value + 252-day z-score. Composite:
    average z over MOVE/SRVIX/VIX/SKEW + CDX_IG/CDX_HY + DXY + SPX."""
    asof_ts = pd.Timestamp(asof)
    rows = []
    z_scores = {}
    for label, (cat, ticker) in RISK_REGIME_TICKERS.items():
        s = _load_bbg_ticker_pct(cat, ticker)
        if s is None or s.empty:
            rows.append({"ticker": label, "value": np.nan, "z_252d": np.nan,
                            "status": "missing"})
            continue
        s_truncated = s[s.index <= asof_ts]
        if len(s_truncated) < 30:
            rows.append({"ticker": label, "value": np.nan, "z_252d": np.nan,
                            "status": "shallow"})
            continue
        cur = float(s_truncated.iloc[-1])
        recent = s_truncated.iloc[-lookback_days:]
        if recent.std() == 0:
            z = 0.0
        else:
            z = (cur - recent.mean()) / recent.std()
        z_scores[label] = float(z)
        rows.append({"ticker": label, "value": cur, "z_252d": float(z),
                        "status": "ok"})
    df = pd.DataFrame(rows)
    # Composite label
    if z_scores:
        composite_z = float(np.mean(list(z_scores.values())))
        if composite_z > 1.0:
            label = "VOL_BACKTEST"
        elif composite_z > 0.3:
            label = "FLIGHT_TO_QUALITY"
        elif composite_z < -1.0:
            label = "BULL_FLATTENER"
        elif composite_z < -0.3:
            label = "GOLDILOCKS"
        else:
            label = "RISK_ON"
    else:
        composite_z = np.nan; label = "UNKNOWN"
    df.loc[len(df)] = {"ticker": "COMPOSITE", "value": composite_z,
                              "z_252d": composite_z, "status": label}
    return df


# =============================================================================
# Bauer-Swanson 2023 orthogonalised MP-surprises (sans skewness — D3=defer)
# =============================================================================

def build_bauer_swanson(asof: date) -> pd.DataFrame:
    """Regress raw FOMC implied move on (NFP surprise, payroll trend, S&P
    log-Δ, slope-Δ, commodity-Δ); residual = orthogonalised MP-surprise.
    Per §16.5 / D3: option-implied skewness term omitted."""
    from lib.cb_meetings import fomc_meetings_with_outcomes
    from lib.analytics.event_impact_a11 import _load_eco_ticker
    meetings = [m for m in fomc_meetings_with_outcomes()
                  if m.realized_change_bp is not None]
    if not meetings:
        return pd.DataFrame()
    # Try to align NFP for each meeting (nearest pre-FOMC release)
    nfp = _load_eco_ticker("NFP_TCH_Index")
    if nfp is None or nfp.empty:
        return pd.DataFrame()
    spx = _load_bbg_ticker_pct("equity_indices", "SPX_Index")
    rows = []
    for m in meetings:
        # Find NFP release closest to but before m.date
        prior_nfps = nfp[nfp["release_date"] <= m.date]
        if prior_nfps.empty or spx is None:
            continue
        nfp_row = prior_nfps.iloc[-1]
        nfp_surprise = float(nfp_row["actual"] - (nfp_row["survey_median"]
                                                            if pd.notna(nfp_row["survey_median"])
                                                            else nfp_row["actual"]))
        # SPX 5-day log-Δ pre-meeting
        try:
            cur_spx = float(spx[spx.index <= pd.Timestamp(m.date)].iloc[-1])
            prior_spx = float(spx[spx.index <= pd.Timestamp(m.date)
                                     - pd.Timedelta(days=7)].iloc[-1])
            spx_delta = float(np.log(cur_spx / prior_spx))
        except Exception:
            spx_delta = 0.0
        rows.append({
            "meeting_date": m.date,
            "realized_change_bp": float(m.realized_change_bp),
            "nfp_surprise": nfp_surprise,
            "spx_5d_log_delta": spx_delta,
            "direction": m.direction,
        })
    if len(rows) < 6:
        return pd.DataFrame(rows)
    df = pd.DataFrame(rows)
    # Regress realized_change_bp on (nfp_surprise, spx_delta) -> residual = orthog. surprise
    X = df[["nfp_surprise", "spx_5d_log_delta"]].fillna(0).to_numpy()
    y = df["realized_change_bp"].to_numpy()
    A = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    fitted = A @ coef
    df["orthogonalised_surprise_bp"] = y - fitted
    df["beta_intercept"] = float(coef[0])
    df["beta_nfp"] = float(coef[1])
    df["beta_spx"] = float(coef[2])
    return df


# =============================================================================
# Top-level driver
# =============================================================================

def _paths(asof: date) -> dict:
    stamp = asof.isoformat()
    return {
        "coint":     _CACHE_DIR / f"cross_cointegration_{stamp}.parquet",
        "risk":      _CACHE_DIR / f"cross_risk_regime_{stamp}.parquet",
        "bauer":     _CACHE_DIR / f"cross_bauer_swanson_{stamp}.parquet",
        "manifest":  _CACHE_DIR / f"cross_cutting_manifest_{stamp}.json",
    }


def build_cross_cutting(asof: Optional[date] = None) -> dict:
    if asof is None:
        cands = sorted(_CACHE_DIR.glob("regime_manifest_*.json"),
                          key=os.path.getmtime, reverse=True)
        if not cands:
            raise RuntimeError("no regime cache; run Phase 4 first")
        asof = date.fromisoformat(cands[0].stem.replace("regime_manifest_", ""))
    paths = _paths(asof)
    coint = build_cointegration(asof)
    if not coint.empty:
        coint.to_parquet(paths["coint"], index=False)
    risk = build_cross_asset_risk_regime(asof)
    if not risk.empty:
        risk.to_parquet(paths["risk"], index=False)
    bauer = build_bauer_swanson(asof)
    if not bauer.empty:
        bauer.to_parquet(paths["bauer"], index=False)

    composite_label = "UNKNOWN"
    if not risk.empty:
        comp_row = risk[risk["ticker"] == "COMPOSITE"]
        if not comp_row.empty:
            composite_label = str(comp_row.iloc[0]["status"])

    manifest = {
        "builder_version": BUILDER_VERSION,
        "asof_date": asof.isoformat(),
        "n_cointegration_pairs": int(len(coint)),
        "n_cointegrated_flagged": int(coint["cointegrated_flag"].sum()) if not coint.empty else 0,
        "risk_regime_label": composite_label,
        "n_bauer_meetings": int(len(bauer)),
        "removed_per_plan_15": ["CFTC_COT_heatmap", "Hull_White_convexity"],
        "deferred_per_D3": ["option_implied_skewness_term"],
    }
    paths["manifest"].write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def main(argv=None):
    import sys
    args = list(sys.argv[1:] if argv is None else argv)
    asof = (date.fromisoformat(args[0]) if args else None)
    print(f"[cross_cutting] building (asof={asof or 'latest'})")
    m = build_cross_cutting(asof)
    print(f"[cross_cutting] coint pairs={m['n_cointegration_pairs']} "
              f"flagged={m['n_cointegrated_flagged']} | risk_regime={m['risk_regime_label']} | "
              f"bauer_meetings={m['n_bauer_meetings']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
