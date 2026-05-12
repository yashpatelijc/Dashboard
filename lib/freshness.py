"""Unified freshness reporting (Phase 0.B + 0.C).

Aggregates every data source's manifest / mtime into a single freshness
struct that every page header + the Settings "System Health" sub-tab
consumes. Any source older than its budget is flagged amber/red.

Architecture: each source has a freshness budget (max-age before flagged).
On query, we walk each source, read its mtime / manifest, compute age,
and return a structured report.

Sources covered:
  - OHLC snapshot DB (rotates daily; budget 36h)
  - BBG parquet warehouse (refreshed weekly; budget 14d)
  - CMC parquet cache (rebuilt daily on first launch; budget 36h)
  - tmia.duckdb backtest cache (10-day cycle; budget 11d)
  - prewarm cache (in-process; report status only)
  - Per-BBG-category depth (counts rows in a sample ticker; flags <2y as low_sample)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


# =============================================================================
# Freshness budgets — locked
# =============================================================================

BUDGETS_HOURS: dict = {
    "ohlc_snapshot":   36,        # daily rotation; 36h tolerates weekend
    "bbg_warehouse":   14 * 24,   # weekly refresh; 14d tolerates short outages
    "cmc_cache":       36,        # rebuilt daily; 36h tolerates weekend
    "backtest_cache":  11 * 24,   # 10-day recompute cycle; 11d tolerates one-day slip
    "prewarm":         12,        # in-process; should always be fresh
    "turn_residuals":  36,        # rebuilt with CMC; same budget
    "regime":          36,        # rebuilt with residuals; same budget
    "policy_path":     36,        # rebuilt with CMC; same budget
    "event_impact":    36,        # rebuilt with regimes; same budget
    "signal_emit":     36,        # rebuilt with all upstream; same budget
    "hei":             36,        # historical event impact; rebuilt with CMC
}


# Per-BBG-category sample tickers used to measure depth. The sample's
# row-count is the proxy for the category's effective depth.
BBG_DEPTH_SAMPLES: dict = {
    "rates_drivers":   "FDTRMID_Index",
    "vol_indices":     "MOVE_Index",
    "credit":          "CDX_HY_CDSI_GEN_5Y_Corp",
    "fx":              "EURUSD_Curncy",
    "eco":             "CPI_CHNG_Index",
    "equity_indices":  "SPX_Index",
    "energy_extras":   "CO1_Comdty",
    "macro_economies": "AUCPIYOY_Index",
    "xcot":            "CFTCNGAS_Index",
}


@dataclass
class SourceFreshness:
    """One entry in the freshness report."""
    source: str             # human-readable
    last_modified: Optional[str]   # ISO datetime or None
    age_hours: Optional[float]
    budget_hours: float
    status: str             # 'green' | 'amber' | 'red' | 'missing'
    detail: str = ""


def _classify(age_hours: Optional[float], budget_hours: float) -> str:
    if age_hours is None:
        return "missing"
    if age_hours <= budget_hours:
        return "green"
    if age_hours <= budget_hours * 1.5:
        return "amber"
    return "red"


def _file_freshness(name: str, path: Path,
                       budget_hours: float) -> SourceFreshness:
    """Generic file-mtime-based freshness check."""
    if not path.exists():
        return SourceFreshness(
            source=name, last_modified=None, age_hours=None,
            budget_hours=budget_hours, status="missing",
            detail=f"file not found: {path}")
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    age = (datetime.now() - mtime).total_seconds() / 3600
    return SourceFreshness(
        source=name,
        last_modified=mtime.isoformat(timespec="seconds"),
        age_hours=round(age, 2),
        budget_hours=budget_hours,
        status=_classify(age, budget_hours),
        detail=f"mtime: {mtime:%Y-%m-%d %H:%M}",
    )


# =============================================================================
# Per-source checkers
# =============================================================================

def check_ohlc_snapshot() -> SourceFreshness:
    """OHLC DuckDB snapshot freshness — finds the latest snapshot file."""
    import glob
    pattern = r"D:\Python Projects\QH_API_APPS\analytics_kit\db_snapshot\market_data_v2_*.duckdb"
    files = glob.glob(pattern)
    if not files:
        return SourceFreshness(
            source="OHLC snapshot DB",
            last_modified=None, age_hours=None,
            budget_hours=BUDGETS_HOURS["ohlc_snapshot"],
            status="missing", detail="no snapshot files found")
    latest = max(files, key=os.path.getmtime)
    return _file_freshness("OHLC snapshot DB", Path(latest),
                              BUDGETS_HOURS["ohlc_snapshot"])


def check_bbg_warehouse() -> SourceFreshness:
    """BBG parquet warehouse — uses any rates_drivers ticker as proxy."""
    sample = Path(r"D:\BBG data\parquet\rates_drivers\FDTRMID_Index.parquet")
    return _file_freshness("BBG parquet warehouse", sample,
                              BUDGETS_HOURS["bbg_warehouse"])


def check_cmc_cache(asof_date: Optional[datetime] = None) -> SourceFreshness:
    """CMC parquet cache — finds latest sra_outright_*.parquet."""
    cache_dir = Path(r"D:\STIRS_DASHBOARD\.cmc_cache")
    if not cache_dir.exists():
        return SourceFreshness(
            source="CMC parquet cache", last_modified=None, age_hours=None,
            budget_hours=BUDGETS_HOURS["cmc_cache"], status="missing",
            detail="cache directory absent")
    candidates = sorted(cache_dir.glob("sra_outright_*.parquet"),
                          key=os.path.getmtime, reverse=True)
    if not candidates:
        return SourceFreshness(
            source="CMC parquet cache", last_modified=None, age_hours=None,
            budget_hours=BUDGETS_HOURS["cmc_cache"], status="missing",
            detail="no parquets in .cmc_cache/")
    return _file_freshness("CMC parquet cache", candidates[0],
                              BUDGETS_HOURS["cmc_cache"])


def check_backtest_cache() -> SourceFreshness:
    """tmia.duckdb backtest cache — 10-day recompute cycle."""
    path = Path(r"D:\STIRS_DASHBOARD\.backtest_cache\tmia.duckdb")
    f = _file_freshness("Backtest tmia.duckdb", path,
                          BUDGETS_HOURS["backtest_cache"])
    if f.status == "missing":
        f.detail = "not yet computed; daemon spawns on next app launch"
    return f


def check_turn_residuals() -> SourceFreshness:
    """Phase 3 turn-adjuster residuals — rebuilt when CMC parquets refresh."""
    cache_dir = Path(r"D:\STIRS_DASHBOARD\.cmc_cache")
    if not cache_dir.exists():
        return SourceFreshness(
            source="Turn-adjuster residuals", last_modified=None, age_hours=None,
            budget_hours=BUDGETS_HOURS["turn_residuals"], status="missing",
            detail="cache directory absent")
    candidates = sorted(cache_dir.glob("turn_residuals_*.parquet"),
                          key=os.path.getmtime, reverse=True)
    if not candidates:
        return SourceFreshness(
            source="Turn-adjuster residuals", last_modified=None, age_hours=None,
            budget_hours=BUDGETS_HOURS["turn_residuals"], status="missing",
            detail="not yet computed; daemon spawns on next app launch")
    return _file_freshness("Turn-adjuster residuals", candidates[0],
                              BUDGETS_HOURS["turn_residuals"])


def check_regime() -> SourceFreshness:
    """Phase 4 regime classifier states — rebuilt when residuals refresh."""
    cache_dir = Path(r"D:\STIRS_DASHBOARD\.cmc_cache")
    if not cache_dir.exists():
        return SourceFreshness(
            source="Regime classifier", last_modified=None, age_hours=None,
            budget_hours=BUDGETS_HOURS["regime"], status="missing",
            detail="cache directory absent")
    candidates = sorted(cache_dir.glob("regime_states_*.parquet"),
                          key=os.path.getmtime, reverse=True)
    if not candidates:
        return SourceFreshness(
            source="Regime classifier", last_modified=None, age_hours=None,
            budget_hours=BUDGETS_HOURS["regime"], status="missing",
            detail="not yet computed; daemon spawns on next app launch")
    return _file_freshness("Regime classifier", candidates[0],
                              BUDGETS_HOURS["regime"])


def check_policy_path() -> SourceFreshness:
    """Phase 5 A4 policy-path PMFs — rebuilt with CMC."""
    cache_dir = Path(r"D:\STIRS_DASHBOARD\.cmc_cache")
    if not cache_dir.exists():
        return SourceFreshness(
            source="Policy path (A4)", last_modified=None, age_hours=None,
            budget_hours=BUDGETS_HOURS["policy_path"], status="missing",
            detail="cache directory absent")
    candidates = sorted(cache_dir.glob("policy_path_*.parquet"),
                          key=os.path.getmtime, reverse=True)
    if not candidates:
        return SourceFreshness(
            source="Policy path (A4)", last_modified=None, age_hours=None,
            budget_hours=BUDGETS_HOURS["policy_path"], status="missing",
            detail="not yet computed; daemon spawns on next app launch")
    return _file_freshness("Policy path (A4)", candidates[0],
                              BUDGETS_HOURS["policy_path"])


def check_event_impact() -> SourceFreshness:
    """Phase 6 A11 event-impact table — rebuilt with regimes."""
    cache_dir = Path(r"D:\STIRS_DASHBOARD\.cmc_cache")
    if not cache_dir.exists():
        return SourceFreshness(
            source="Event impact (A11)", last_modified=None, age_hours=None,
            budget_hours=BUDGETS_HOURS["event_impact"], status="missing",
            detail="cache directory absent")
    candidates = sorted(cache_dir.glob("event_impact_*.parquet"),
                          key=os.path.getmtime, reverse=True)
    if not candidates:
        return SourceFreshness(
            source="Event impact (A11)", last_modified=None, age_hours=None,
            budget_hours=BUDGETS_HOURS["event_impact"], status="missing",
            detail="not yet computed; daemon spawns on next app launch")
    return _file_freshness("Event impact (A11)", candidates[0],
                              BUDGETS_HOURS["event_impact"])


def check_hei() -> SourceFreshness:
    """Historical Event Impact cache — rebuilt with CMC."""
    cache_dir = Path(r"D:\STIRS_DASHBOARD\.cmc_cache")
    if not cache_dir.exists():
        return SourceFreshness(
            source="Historical Event Impact",
            last_modified=None, age_hours=None,
            budget_hours=BUDGETS_HOURS["hei"], status="missing",
            detail="cache directory absent")
    candidates = sorted(cache_dir.glob("hei_manifest_*.json"),
                          key=os.path.getmtime, reverse=True)
    if not candidates:
        return SourceFreshness(
            source="Historical Event Impact",
            last_modified=None, age_hours=None,
            budget_hours=BUDGETS_HOURS["hei"], status="missing",
            detail="not yet computed; daemon spawns on next app launch")
    return _file_freshness("Historical Event Impact", candidates[0],
                              BUDGETS_HOURS["hei"])


def check_signal_emit() -> SourceFreshness:
    """Phase 7 signal_emit canonical table — rebuilt with all upstream."""
    cache_dir = Path(r"D:\STIRS_DASHBOARD\.signal_cache")
    if not cache_dir.exists():
        return SourceFreshness(
            source="signal_emit", last_modified=None, age_hours=None,
            budget_hours=BUDGETS_HOURS["signal_emit"], status="missing",
            detail="signal_cache directory absent")
    candidates = sorted(cache_dir.glob("signal_emit_*.parquet"),
                          key=os.path.getmtime, reverse=True)
    if not candidates:
        return SourceFreshness(
            source="signal_emit", last_modified=None, age_hours=None,
            budget_hours=BUDGETS_HOURS["signal_emit"], status="missing",
            detail="not yet computed; daemon spawns on next app launch")
    return _file_freshness("signal_emit", candidates[0],
                              BUDGETS_HOURS["signal_emit"])


def check_prewarm_status() -> SourceFreshness:
    """Pre-warm daemon — in-process state, not file-based."""
    try:
        from lib.prewarm import get_prewarm_status, is_prewarm_done
    except Exception as e:
        return SourceFreshness(
            source="Pre-warm daemon", last_modified=None, age_hours=None,
            budget_hours=BUDGETS_HOURS["prewarm"], status="missing",
            detail=f"import failed: {e}")
    status = get_prewarm_status()
    started = status.get("started_at")
    completed = status.get("outright_done_at") or status.get("completed_at")
    if completed:
        mtime = datetime.fromtimestamp(completed)
        age = (datetime.now() - mtime).total_seconds() / 3600
        return SourceFreshness(
            source="Pre-warm daemon",
            last_modified=mtime.isoformat(timespec="seconds"),
            age_hours=round(age, 2),
            budget_hours=BUDGETS_HOURS["prewarm"],
            status=_classify(age, BUDGETS_HOURS["prewarm"]),
            detail="outright leg complete")
    if started:
        elapsed = datetime.now().timestamp() - started
        return SourceFreshness(
            source="Pre-warm daemon", last_modified=None,
            age_hours=elapsed / 3600,
            budget_hours=BUDGETS_HOURS["prewarm"],
            status="amber",
            detail=f"running ({elapsed:.0f}s elapsed)")
    return SourceFreshness(
        source="Pre-warm daemon", last_modified=None, age_hours=None,
        budget_hours=BUDGETS_HOURS["prewarm"], status="missing",
        detail="not started this session")


# =============================================================================
# BBG category depth (data sufficiency) — Phase 0.D
# =============================================================================

@dataclass
class CategoryDepth:
    category: str
    sample_ticker: str
    rows: Optional[int]
    first_date: Optional[str]
    last_date: Optional[str]
    years_of_history: Optional[float]
    sufficiency: str   # 'sufficient' | 'shallow' | 'missing'


def check_bbg_category_depth(category: str,
                                  sample_ticker: Optional[str] = None) -> CategoryDepth:
    """Count rows + date range of a BBG parquet category, using a sample
    ticker as proxy. Used to detect 'shallow data' regimes that gate
    sample-bound analytics (regime classifier, A11-event recency-weighting,
    A6 OU calibration).
    """
    if sample_ticker is None:
        sample_ticker = BBG_DEPTH_SAMPLES.get(category, "")
    if not sample_ticker:
        return CategoryDepth(category=category, sample_ticker="",
                                rows=None, first_date=None, last_date=None,
                                years_of_history=None, sufficiency="missing")
    path = Path(rf"D:\BBG data\parquet\{category}\{sample_ticker}.parquet")
    if not path.exists():
        return CategoryDepth(category=category, sample_ticker=sample_ticker,
                                rows=None, first_date=None, last_date=None,
                                years_of_history=None, sufficiency="missing")
    try:
        import duckdb
        con = duckdb.connect(":memory:")
        row = con.execute(
            f"SELECT COUNT(*), MIN(date), MAX(date) "
            f"FROM read_parquet('{str(path).replace(chr(92), chr(47))}')"
        ).fetchone()
        n, lo, hi = int(row[0]), str(row[1]), str(row[2])
        try:
            years = (datetime.fromisoformat(hi) -
                       datetime.fromisoformat(lo)).days / 365.25
        except (ValueError, TypeError):
            years = None
        # Sufficiency: 5y is "sufficient" for regime / event-impact recency
        if years is not None and years >= 5:
            suff = "sufficient"
        elif years is not None and years >= 1:
            suff = "shallow"
        else:
            suff = "shallow"
        return CategoryDepth(
            category=category, sample_ticker=sample_ticker,
            rows=n, first_date=lo, last_date=hi,
            years_of_history=round(years, 2) if years is not None else None,
            sufficiency=suff,
        )
    except Exception as e:
        return CategoryDepth(category=category, sample_ticker=sample_ticker,
                                rows=None, first_date=None, last_date=None,
                                years_of_history=None,
                                sufficiency="missing")


# =============================================================================
# Public aggregator — single call returns the full freshness picture
# =============================================================================

def freshness_report() -> dict:
    """Return the full freshness + depth report.

    Structure:
        {
          "sources": [SourceFreshness, ...],
          "category_depth": [CategoryDepth, ...],
          "overall_status": "green" | "amber" | "red" | "missing",
          "as_of": ISO datetime
        }
    """
    sources = [
        check_ohlc_snapshot(),
        check_bbg_warehouse(),
        check_cmc_cache(),
        check_backtest_cache(),
        check_turn_residuals(),
        check_regime(),
        check_policy_path(),
        check_event_impact(),
        check_signal_emit(),
        check_hei(),
        check_prewarm_status(),
    ]
    depths = [check_bbg_category_depth(cat) for cat in BBG_DEPTH_SAMPLES]

    # Overall: worst across all source statuses
    rank = {"green": 0, "amber": 1, "red": 2, "missing": 3}
    worst = max(rank.get(s.status, 0) for s in sources)
    overall = next(k for k, v in rank.items() if v == worst)

    return {
        "sources": [asdict(s) for s in sources],
        "category_depth": [asdict(d) for d in depths],
        "overall_status": overall,
        "as_of": datetime.now().isoformat(timespec="seconds"),
    }


# =============================================================================
# UI helpers
# =============================================================================

def freshness_traffic_light_html(status: str, label: str = "") -> str:
    """Render an inline traffic-light dot for a header chip.

    Status 'green' / 'amber' / 'red' / 'missing'. Returns HTML string.
    """
    color_map = {
        "green":   ("#4ade80", "FRESH"),
        "amber":   ("#fbbf24", "STALE"),
        "red":     ("#f87171", "OLD"),
        "missing": ("#5e6975", "NONE"),
    }
    color, default_label = color_map.get(status, ("#5e6975", "NONE"))
    text = label or default_label
    return (
        f"<span style='display:inline-flex; align-items:center; gap:0.35rem; "
        f"padding:2px 8px; border:1px solid {color}; border-radius:999px; "
        f"font-family: JetBrains Mono, monospace; font-size:0.65rem; "
        f"color:{color}; background:rgba(0,0,0,0.2);'>"
        f"<span style='display:inline-block; width:6px; height:6px; "
        f"background:{color}; border-radius:50%;'></span>"
        f"<span>{text}</span></span>"
    )


def freshness_header_chip() -> str:
    """One-line header chip showing the worst-of-all freshness status."""
    rep = freshness_report()
    overall = rep["overall_status"]
    n_red = sum(1 for s in rep["sources"] if s["status"] == "red")
    n_amber = sum(1 for s in rep["sources"] if s["status"] == "amber")
    n_missing = sum(1 for s in rep["sources"] if s["status"] == "missing")
    if overall == "green":
        label = "DATA FRESH"
    elif n_red > 0:
        label = f"{n_red} STALE"
    elif n_missing > 0:
        label = f"{n_missing} MISSING"
    else:
        label = f"{n_amber} AMBER"
    return freshness_traffic_light_html(overall, label)
