"""A11 event-impact module (Phase 6, plan §12 — single largest deliverable).

Per (ticker × CMC-node × regime) regression of same-day CMC residual_change
on standardised event surprise. Builds an importance score (composite of
|β|, R², hit rate) + recency-weighted variant + "becoming more important"
flag.

Inputs
------
- Phase 3 ``residual_change`` panel (post-LIBOR-cutover) via
  ``lib.sra_data.load_turn_residual_change_wide_panel``.
- Phase 4 regime states via ``lib.sra_data.load_regime_states``.
- BBG eco parquets: ``D:\BBG data\parquet\eco\<TICKER>.parquet``
  with columns [date, ECO_RELEASE_DT, ACTUAL_RELEASE, PX_LAST,
  BN_SURVEY_MEDIAN, ...].

Surprise definition (per ticker)
--------------------------------
- B0 axis (preferred): standardised (ACTUAL_RELEASE - BN_SURVEY_MEDIAN).
  When BN_SURVEY_MEDIAN is available, use it; tag ``axis=B0_consensus``.
- B2 axis (fallback, ``circular_proxy`` tag): z-score of ACTUAL_RELEASE
  vs trailing 12-release rolling mean/std. Used when consensus data
  unavailable for that ticker.

Per the spec, B1 (intraday-jump) is deferred to Tier 1 sub-minute data
which we don't have; not built in Phase 6.

Composite importance score
--------------------------
    score = 0.40 * |beta_norm| + 0.30 * R² + 0.30 * hit_rate
where
    beta_norm = |beta| / std(residual_change)  (unitless 0..1+)
    R²        = 1 - SS_resid / SS_total       (clipped 0..1)
    hit_rate  = fraction of releases where sign(beta * surprise) ==
                sign(residual_change)         (0..1)

Recency-weighted variant
------------------------
Same regression with exponential weights w_i = exp(-Δt_i / halflife).
Halflife = 2 years (730 days) per spec. Released as
``score_recency_weighted``.

"becoming_more_important" flag
------------------------------
Set to True iff ``score_recency_weighted > score_full * 1.50``.

Per §15 D2 (no scrapers): Tier 3 Treasury auctions = STUB; Tier 1/2/4
fully built from eco/ files.
"""
from __future__ import annotations

import json
import os
import warnings
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".cmc_cache"
_CACHE_DIR.mkdir(exist_ok=True)
_ECO_DIR = Path(r"D:\BBG data\parquet\eco")

BUILDER_VERSION = "1.0.0"

DEFAULT_HALFLIFE_DAYS = 730       # 2 years per spec
MIN_RELEASES_PER_TICKER = 8        # below this, ticker dropped
RECENCY_FLAG_THRESHOLD = 1.50      # recency_weighted > full × 1.5

# Curated high-impact ticker list. The engine accepts any subset that
# exists on disk; missing tickers are silently skipped. Adding a ticker
# requires no code change — drop it in this list.
CURATED_TICKERS: tuple = (
    # Employment
    "NFP_TCH_Index", "USURTOT_Index", "AHE_YOY_Index", "INJCJC_Index",
    "ECI_QOQ_Index",
    # Inflation
    "CPI_CHNG_Index", "CPI_YOY_Index", "CPI_XYOY_Index",
    "PPI_FINL_Index", "PPI_YOY_Index", "PCEDEFY_Index",
    # Activity / output
    "GDP_CQOQ_Index", "INDPRO_Index", "DGNOCHNG_Index", "DGNOXAI_Index",
    "RSTAMOM_Index", "RSTAEXAG_Index",
    # Surveys / sentiment
    "NAPM_PMI_Index", "NAPMNMI_Index", "CONCCONF_Index", "CONSSENT_Index",
    # Housing
    "USHBHIME_Index", "HMSI_INDX_Index", "COMRSALE_Index",
    # Other macro
    "EHSLTOTL_Index", "MWINCNG_Index",
)


# =============================================================================
# Eco data loader
# =============================================================================

def _list_available_tickers() -> list[str]:
    if not _ECO_DIR.exists():
        return []
    files = list(_ECO_DIR.glob("*.parquet"))
    return [f.stem for f in files]


def _load_eco_ticker(ticker: str) -> Optional[pd.DataFrame]:
    """Load a single eco ticker. Returns DF with [release_date, actual,
    survey_median, surprise_B0]. None if file missing/unreadable."""
    path = _ECO_DIR / f"{ticker}.parquet"
    if not path.exists():
        return None
    try:
        import duckdb
        con = duckdb.connect(":memory:")
        p = str(path).replace("\\", "/")
        df = con.execute(
            f"SELECT date, ECO_RELEASE_DT, ACTUAL_RELEASE, "
            f"PX_LAST, BN_SURVEY_MEDIAN "
            f"FROM read_parquet('{p}')"
        ).fetchdf()
    except Exception:
        return None
    if df.empty:
        return None

    # Build release_date: prefer ECO_RELEASE_DT (YYYYMMDD float), else use date.
    def _parse_release(row):
        eco_dt = row.get("ECO_RELEASE_DT")
        if eco_dt is not None and not pd.isna(eco_dt):
            try:
                s = str(int(eco_dt))
                return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
            except (ValueError, TypeError):
                pass
        d = row.get("date")
        if pd.isna(d):
            return None
        return pd.to_datetime(d).date()

    df["release_date"] = df.apply(_parse_release, axis=1)
    df = df.dropna(subset=["release_date"]).copy()

    # Use ACTUAL_RELEASE if available else PX_LAST (back-revised)
    df["actual"] = df["ACTUAL_RELEASE"].where(
        df["ACTUAL_RELEASE"].notna(), df["PX_LAST"])
    df["survey_median"] = df["BN_SURVEY_MEDIAN"]
    df = df.dropna(subset=["actual"]).copy()
    return df[["release_date", "actual", "survey_median"]].sort_values("release_date")


def compute_surprise_columns(eco_df: pd.DataFrame
                                  ) -> tuple[pd.DataFrame, str]:
    """Compute standardised surprise. Returns (df_with_surprise, axis_used).

    axis_used: 'B0_consensus' if survey_median populated for ≥ 50% of rows,
    else 'B2_z_history' (the circular_proxy fallback tag).
    """
    df = eco_df.copy()
    n = len(df)
    consensus_frac = float(df["survey_median"].notna().sum() / max(n, 1))
    if consensus_frac >= 0.5:
        # B0 axis: standardised (actual - survey_median)
        diff = df["actual"] - df["survey_median"]
        diff_std = float(diff.std(ddof=0))
        if diff_std > 0:
            df["surprise"] = (diff - diff.mean()) / diff_std
        else:
            df["surprise"] = 0.0
        axis = "B0_consensus"
    else:
        # B2 z-score proxy
        rolling_mean = df["actual"].rolling(12, min_periods=4).mean()
        rolling_std = df["actual"].rolling(12, min_periods=4).std(ddof=0)
        df["surprise"] = (df["actual"] - rolling_mean) / rolling_std.replace(0, np.nan)
        axis = "B2_z_history"
    df = df.dropna(subset=["surprise"]).copy()
    return df, axis


# =============================================================================
# Per (ticker × node × regime) regression
# =============================================================================

def _hit_rate(beta: float, surprise: np.ndarray, response: np.ndarray) -> float:
    """Fraction of releases where sign(beta * surprise) == sign(response)."""
    if beta == 0 or len(response) == 0:
        return 0.0
    pred_sign = np.sign(beta * surprise)
    obs_sign = np.sign(response)
    return float((pred_sign == obs_sign).mean())


def regress_one(surprise: np.ndarray, response: np.ndarray,
                    weights: Optional[np.ndarray] = None) -> dict:
    """Weighted OLS of response ~ surprise (no intercept; pre-centered).

    Returns: {beta, se, r_squared, hit_rate, n_obs, beta_norm}.
    """
    if len(surprise) < MIN_RELEASES_PER_TICKER:
        return {"beta": np.nan, "se": np.nan, "r_squared": np.nan,
                  "hit_rate": np.nan, "n_obs": int(len(surprise)),
                  "beta_norm": np.nan}
    x = np.asarray(surprise, dtype=float)
    y = np.asarray(response, dtype=float)
    # Standardise both for unit-free comparison
    x_c = x - x.mean(); y_c = y - y.mean()
    if weights is None:
        w = np.ones_like(x_c)
    else:
        w = np.asarray(weights, dtype=float)
        w = w / max(w.sum(), 1e-12) * len(w)
    sxx = float(np.sum(w * x_c * x_c))
    sxy = float(np.sum(w * x_c * y_c))
    syy = float(np.sum(w * y_c * y_c))
    if sxx <= 1e-12:
        return {"beta": 0.0, "se": np.nan, "r_squared": 0.0,
                  "hit_rate": 0.0, "n_obs": int(len(x)),
                  "beta_norm": 0.0}
    beta = sxy / sxx
    resid = y_c - beta * x_c
    rss = float(np.sum(w * resid * resid))
    r2 = max(0.0, 1.0 - rss / max(syy, 1e-12))
    # SE via residual variance
    n = len(x)
    sigma2 = rss / max(n - 1, 1)
    se = float(np.sqrt(sigma2 / max(sxx, 1e-12)))
    hr = _hit_rate(beta, x, y)
    y_std = float(np.std(y, ddof=0))
    beta_norm = float(abs(beta) * float(np.std(x, ddof=0)) / max(y_std, 1e-12))
    return {"beta": float(beta), "se": float(se), "r_squared": float(r2),
              "hit_rate": float(hr), "n_obs": int(n),
              "beta_norm": float(min(beta_norm, 5.0))}


def composite_score(beta_norm: float, r_squared: float,
                        hit_rate: float) -> float:
    """40% beta_norm + 30% R² + 30% hit_rate (clipped 0..1)."""
    if any(np.isnan([beta_norm, r_squared, hit_rate])):
        return 0.0
    bn = min(max(beta_norm, 0.0), 1.0)
    return float(0.40 * bn + 0.30 * r_squared + 0.30 * hit_rate)


# =============================================================================
# Top-level driver
# =============================================================================

def build_event_impact(asof: Optional[date] = None,
                            tickers: Optional[list[str]] = None,
                            halflife_days: int = DEFAULT_HALFLIFE_DAYS,
                            ) -> dict:
    """For each ticker × CMC node × regime cell, fit two regressions
    (full-sample + recency-weighted) and persist a ranked table."""
    if asof is None:
        cands = sorted(_CACHE_DIR.glob("regime_manifest_*.json"),
                          key=os.path.getmtime, reverse=True)
        if not cands:
            raise RuntimeError("no regime cache; run Phase 4 first")
        asof = date.fromisoformat(cands[0].stem.replace("regime_manifest_", ""))

    if tickers is None:
        tickers = list(CURATED_TICKERS)
    available = set(_list_available_tickers())
    tickers = [t for t in tickers if t in available]
    if not tickers:
        raise RuntimeError("no curated tickers available in eco/")

    # Load Phase 3 wide residual_change panel + Phase 4 regimes
    from lib.sra_data import (
        load_turn_residual_change_wide_panel, load_regime_states,
    )
    resid = load_turn_residual_change_wide_panel("outright", asof)
    regimes = load_regime_states(asof)
    if resid is None or resid.empty or regimes is None or regimes.empty:
        raise RuntimeError("Phase 3 residuals or Phase 4 regimes missing")
    resid.index = pd.to_datetime(resid.index).tz_localize(None)
    regimes["bar_date"] = pd.to_datetime(regimes["bar_date"]).dt.tz_localize(None)
    regime_lookup = regimes.set_index("bar_date")["state_name"].to_dict()

    cmc_nodes = list(resid.columns)
    rows = []
    skipped_reasons = {}

    for ticker in tickers:
        eco = _load_eco_ticker(ticker)
        if eco is None or eco.empty:
            skipped_reasons[ticker] = "no_data"
            continue
        eco_with_surprise, axis = compute_surprise_columns(eco)
        if len(eco_with_surprise) < MIN_RELEASES_PER_TICKER:
            skipped_reasons[ticker] = f"only_{len(eco_with_surprise)}_obs"
            continue

        # Filter to post-cutover dates that intersect with residual panel
        resid_min = resid.index.min().date()
        resid_max = resid.index.max().date()
        eco_in_range = eco_with_surprise[
            (eco_with_surprise["release_date"] >= resid_min)
            & (eco_with_surprise["release_date"] <= resid_max)
        ].copy()
        if len(eco_in_range) < MIN_RELEASES_PER_TICKER:
            skipped_reasons[ticker] = f"only_{len(eco_in_range)}_post_cutover"
            continue

        eco_in_range["release_date"] = pd.to_datetime(
            eco_in_range["release_date"])

        for node in cmc_nodes:
            # Same-day residual change at each release date
            same_day = (eco_in_range
                            .merge(
                                pd.DataFrame({"release_date": resid.index,
                                                  "response": resid[node].values}),
                                on="release_date", how="left")
                            .dropna(subset=["response"]))
            if len(same_day) < MIN_RELEASES_PER_TICKER:
                continue
            x = same_day["surprise"].to_numpy(dtype=float)
            y = same_day["response"].to_numpy(dtype=float)

            # Full-sample regression
            full = regress_one(x, y)
            full_score = composite_score(
                full["beta_norm"], full["r_squared"], full["hit_rate"])

            # Recency weights: exp(-(asof - release_date) / halflife)
            asof_ts = pd.Timestamp(asof)
            ages = (asof_ts - same_day["release_date"]).dt.days.to_numpy(dtype=float)
            w = np.exp(-ages / max(halflife_days, 1))
            recency = regress_one(x, y, weights=w)
            recency_score = composite_score(
                recency["beta_norm"], recency["r_squared"], recency["hit_rate"])

            becoming_more_important = bool(
                recency_score > full_score * RECENCY_FLAG_THRESHOLD
                and recency_score > 0.10
            )

            # Also: per-regime (only if a regime has >= MIN_RELEASES_PER_TICKER releases)
            regime_dates = same_day["release_date"].dt.date
            regime_for_release = regime_dates.apply(
                lambda d: regime_lookup.get(pd.Timestamp(d), None))
            for regime_name in set(regime_for_release.dropna().unique()):
                mask = regime_for_release == regime_name
                if int(mask.sum()) < MIN_RELEASES_PER_TICKER:
                    continue
                xr = x[mask]; yr = y[mask]
                rfit = regress_one(xr, yr)
                rscore = composite_score(rfit["beta_norm"], rfit["r_squared"],
                                              rfit["hit_rate"])
                rows.append({
                    "ticker": ticker, "axis": axis, "cmc_node": node,
                    "regime": regime_name,
                    "n_obs": rfit["n_obs"],
                    "beta": rfit["beta"], "se": rfit["se"],
                    "r_squared": rfit["r_squared"],
                    "hit_rate": rfit["hit_rate"],
                    "score_full": rscore, "score_recency_weighted": np.nan,
                    "becoming_more_important": False, "scope": "by_regime",
                })

            rows.append({
                "ticker": ticker, "axis": axis, "cmc_node": node,
                "regime": "ALL",
                "n_obs": full["n_obs"],
                "beta": full["beta"], "se": full["se"],
                "r_squared": full["r_squared"],
                "hit_rate": full["hit_rate"],
                "score_full": full_score,
                "score_recency_weighted": recency_score,
                "becoming_more_important": becoming_more_important,
                "scope": "full",
            })

    if not rows:
        raise RuntimeError("no rows produced — check eco/ ticker availability")

    df = pd.DataFrame(rows)

    return write_event_impact_cache(asof, df, tickers, axis_used_set={
        r["axis"] for r in rows
    }, skipped=skipped_reasons, halflife_days=halflife_days)


def write_event_impact_cache(asof: date, df: pd.DataFrame,
                                  tickers: list[str], axis_used_set: set,
                                  skipped: dict, halflife_days: int) -> dict:
    paths = _paths(asof)
    tmp_table = paths["table"].with_suffix(".parquet.tmp")
    tmp_man = paths["manifest"].with_suffix(".json.tmp")
    df.to_parquet(tmp_table, index=False)

    manifest = {
        "builder_version": BUILDER_VERSION,
        "asof_date": asof.isoformat(),
        "n_rows": int(len(df)),
        "n_tickers_attempted": len(tickers),
        "n_tickers_built": int(df["ticker"].nunique()),
        "axes_used": sorted(axis_used_set),
        "halflife_days": int(halflife_days),
        "min_releases_per_ticker": int(MIN_RELEASES_PER_TICKER),
        "n_becoming_more_important": int(
            df.loc[df["scope"] == "full", "becoming_more_important"].sum()),
        "skipped_tickers": skipped,
    }
    tmp_man.write_text(json.dumps(manifest, indent=2, default=str))

    os.replace(tmp_table, paths["table"])
    os.replace(tmp_man, paths["manifest"])
    return manifest


def _paths(asof: date) -> dict:
    stamp = asof.isoformat()
    return {
        "table":    _CACHE_DIR / f"event_impact_{stamp}.parquet",
        "manifest": _CACHE_DIR / f"event_impact_manifest_{stamp}.json",
    }


def main(argv=None):
    import sys
    args = list(sys.argv[1:] if argv is None else argv)
    asof = None
    if args:
        try:
            asof = date.fromisoformat(args[0])
        except ValueError:
            print("usage: python -m lib.analytics.event_impact_a11 [YYYY-MM-DD]")
            return 2
    print(f"[event_impact_a11] building event-impact table (asof={asof or 'latest'})")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        manifest = build_event_impact(asof)
    print(f"[event_impact_a11] {manifest['n_tickers_built']}/"
              f"{manifest['n_tickers_attempted']} tickers built; "
              f"{manifest['n_rows']} rows; "
              f"axes_used={manifest['axes_used']}; "
              f"becoming_more_important={manifest['n_becoming_more_important']}")
    if manifest["skipped_tickers"]:
        print(f"  skipped: {list(manifest['skipped_tickers'].items())[:5]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
