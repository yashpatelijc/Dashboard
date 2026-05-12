"""Universe scanner — single cached entry point for the Technicals subtab.

``scan_universe(strategy, tenor_months, asof_str, ...)`` returns:

  {
    "asof": date,
    "scope": "outright"|"spread"|"fly",
    "tenor_months": int|None,
    "contracts": [...],
    "by_contract": {
        symbol: {
            setup_id: SetupResult-as-dict,
            "_composites": {trend_score, mr_score, final_score, factors, weights, regime, ...},
            "_meta": {bp_multiplier, convention, atr_bp, ...},
            "_carry": float | None,
        },
        ...
    },
    "fires_today": [ {symbol, setup_id, direction, ...}, ... ],
    "near_today":  [ ... ],
    "track_records": {setup_id: {fires_60d, win_rate_5bar, mean_5bar_return_bp}},
    "errors": [ ... ],
  }

All math is delegated to detectors / composite.py. This module is the
performance-critical layer (cached, parallelisable).
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from lib.carry import compute_per_contract_carry, implied_rate
from lib.contract_units import bp_multiplier, load_catalog
from lib.fomc import is_quarterly, third_wednesday
from lib.setups.base import SetupResult, build_legs, _parse_legs_string
from lib.setups.composite import compute_composites
from lib.setups.mean_reversion import (
    detect_b1, detect_b3, detect_b5, detect_b6, detect_b10, detect_b11, detect_b13,
)
from lib.setups.registry import (
    FAMILY_MR, FAMILY_STIR, FAMILY_TREND,
    setups_for_scope,
    SETUP_REGISTRY,
)
from lib.setups.stir import (
    C9_DEFAULT_OFFSET_MONTHS,
    compute_c3_universe,
    detect_c3_for_contract,
    detect_c4,
    detect_c5,
    detect_c8_variant,
    detect_c9a,
    detect_c9b,
    pick_c9_contract_pair,
    pick_c9_contract_pairs,
)
from lib.setups.track_record import compute_track_record
from lib.setups.trend import (
    detect_a1, detect_a2, detect_a3, detect_a4, detect_a5, detect_a6,
    detect_a8, detect_a10, detect_a11, detect_a12a, detect_a12b, detect_a15,
)
from lib.sra_data import (
    get_contract_full_panel,
    get_flies as _sra_get_flies,
    get_outrights as _sra_get_outrights,
    get_spreads as _sra_get_spreads,
    get_sra_snapshot_latest_date,
)
from lib import market_data as _md


# Active-market shim for setup scans
_SCAN_BP = "SRA"


def _set_scan_market(base_product: str) -> None:
    global _SCAN_BP
    _SCAN_BP = base_product


def get_outrights():
    return _sra_get_outrights() if _SCAN_BP == "SRA" else _md.get_outrights(_SCAN_BP)


def get_spreads(t):
    return _sra_get_spreads(t) if _SCAN_BP == "SRA" else _md.get_spreads(_SCAN_BP, t)


def get_flies(t):
    return _sra_get_flies(t) if _SCAN_BP == "SRA" else _md.get_flies(_SCAN_BP, t)


# =============================================================================
# Detector registry — id → callable
# =============================================================================
TREND_DETECTORS = {
    "A1": detect_a1, "A2": detect_a2, "A3": detect_a3, "A4": detect_a4,
    "A5": detect_a5, "A6": detect_a6, "A8": detect_a8, "A10": detect_a10,
    "A11": detect_a11, "A12a": detect_a12a, "A12b": detect_a12b, "A15": detect_a15,
}
MR_DETECTORS = {
    "B1": detect_b1, "B3": detect_b3, "B5": detect_b5, "B6": detect_b6,
    "B10": detect_b10, "B11": detect_b11, "B13": detect_b13,
}


# =============================================================================
# Universe symbol loader (per scope/tenor)
# =============================================================================
def _load_symbols(strategy: str, tenor_months: Optional[int]) -> pd.DataFrame:
    if strategy == "outright":
        return get_outrights()
    if strategy == "spread":
        return get_spreads(int(tenor_months)) if tenor_months else pd.DataFrame()
    if strategy == "fly":
        return get_flies(int(tenor_months)) if tenor_months else pd.DataFrame()
    return pd.DataFrame()


# =============================================================================
# Per-contract panel cache (panel = full OHLCV + 159 indicators)
# =============================================================================
def _load_panels(contracts: list, asof_date: date,
                  history_days: int = 280) -> dict:
    """Fetch the OHLCV + indicator panel per contract. Returns ``{sym: DataFrame}``."""
    end = asof_date
    start = asof_date - timedelta(days=int(history_days * 1.5) + 30)
    out = {}
    for sym in contracts:
        try:
            df = get_contract_full_panel(sym, start, end)
            out[sym] = df
        except Exception:
            out[sym] = pd.DataFrame()
    return out


# =============================================================================
# Main scanner — cached
# =============================================================================
@st.cache_data(show_spinner=False, ttl=600)
def scan_universe(strategy: str, tenor_months: Optional[int],
                   asof_str: str, history_days: int = 280,
                   base_product: str = "SRA") -> dict:
    """Run all applicable detectors over the live universe of (strategy, tenor).

    Cache key = (strategy, tenor_months, asof_str, history_days). Re-running
    on the same key returns cached result (TTL 600s).
    """
    _set_scan_market(base_product)
    asof_date = pd.Timestamp(asof_str).date()
    out = {
        "asof": asof_date,
        "base_product": base_product,
        "scope": strategy,
        "tenor_months": tenor_months,
        "contracts": [],
        "by_contract": {},
        "fires_today": [],
        "near_today": [],
        "track_records": {},
        "errors": [],
        "n_total": 0,
    }
    syms_df = _load_symbols(strategy, tenor_months)
    if syms_df is None or syms_df.empty:
        out["errors"].append("no live contracts in scope")
        return out
    contracts = list(syms_df["symbol"])
    out["contracts"] = contracts
    out["n_total"] = len(contracts)
    panels = _load_panels(contracts, asof_date, history_days)

    cat = load_catalog()
    # Auto-detect base_product per symbol so non-SRA markets get the right tick.
    try:
        from lib.markets import parse_symbol_to_base_product as _psbp
    except Exception:
        _psbp = lambda s: "SRA"
    bp_mults = {sym: bp_multiplier(sym, _psbp(sym) or "SRA", cat) for sym in contracts}

    # Per-symbol legs lookup from the catalog 'legs' column. Outright = []
    # (single-leg fallback inside build_legs); spread = [left, right];
    # fly = [left, mid, right].
    legs_lookup: dict = {}
    if not cat.empty and "legs" in cat.columns:
        cat_idx = cat.set_index("symbol")["legs"]
        for sym in contracts:
            legs_lookup[sym] = _parse_legs_string(cat_idx.get(sym, ""))

    def _fill_legs(r_dict: dict, sym: str) -> None:
        """Populate r_dict['legs'] in-place if entry/lots indicate a sized
        trade. Idempotent — does nothing if r_dict already has legs or no
        actionable direction. Does NOT mutate setups that set their own
        legs (e.g. C9a/C9b which know their pair internally)."""
        if not isinstance(r_dict, dict):
            return
        if r_dict.get("legs"):
            return
        direction = r_dict.get("direction")
        if direction not in ("LONG", "SHORT"):
            return
        if r_dict.get("entry") is None:
            return
        lots = r_dict.get("lots_at_10k_risk")
        legs = build_legs(sym, strategy, direction, lots,
                            legs_from_catalog=legs_lookup.get(sym, []))
        if legs:
            r_dict["legs"] = legs

    # =====================================================================
    # C3 — universe-level pre-compute (only outrights)
    # =====================================================================
    c3_universe = None
    if strategy == "outright":
        close_today = {}
        ts = pd.Timestamp(asof_date)
        for sym in contracts:
            p = panels.get(sym)
            if p is None or p.empty:
                close_today[sym] = None; continue
            row = p.loc[p.index.normalize() == ts.normalize()]
            close_today[sym] = float(row["close"].iloc[-1]) if not row.empty else None
        c3_universe = compute_c3_universe(close_today, panels, contracts, asof_date)

    # =====================================================================
    # C9 — pick front/back PAIRS (multi-offset curve ladder, only outrights)
    # =====================================================================
    c9_pairs: list = []   # list of (front_sym, back_sym, offset_months)
    if strategy == "outright":
        c9_pairs = pick_c9_contract_pairs(syms_df, asof_date,
                                            offsets_months=C9_DEFAULT_OFFSET_MONTHS)

    # =====================================================================
    # C8 — universe-level pre-compute (only outrights)
    # =====================================================================
    c8_results = {}
    if strategy == "outright":
        close_history = {sym: panels[sym]["close"] if "close" in panels[sym].columns else pd.Series(dtype=float)
                          for sym in contracts}
        for h in (12, 24, 36):
            try:
                c8_results[h] = detect_c8_variant(
                    close_history, syms_df, asof_date, horizon_months=h,
                    lookback_days=5, fire_bp_threshold=25.0,
                )
            except Exception as e:
                out["errors"].append(f"C8_{h}M: {e}")

        # Mark PRIMARY = strongest fire by |Δ|
        if c8_results:
            primary = max(c8_results.keys(),
                           key=lambda h: abs(c8_results[h].key_inputs.get("Δ over 5d (bp)", 0))
                                          if c8_results[h].state != "N/A" else -1)
            for h in c8_results:
                if h == primary and c8_results[h].state == "FIRED":
                    c8_results[h].notes = (c8_results[h].notes + " · PRIMARY").strip(" ·")

    # =====================================================================
    # Per-contract detectors
    # =====================================================================
    for sym in contracts:
        panel = panels.get(sym)
        if panel is None or panel.empty:
            out["by_contract"][sym] = {"_meta": {"error": "no panel"}}
            continue
        bp_mult = bp_mults.get(sym, 100.0)
        contract_results = {}

        # Trend (always applicable to all 3 scopes per registry)
        for sid in setups_for_scope(strategy, families=(FAMILY_TREND,)):
            if sid not in TREND_DETECTORS:
                continue
            try:
                r = TREND_DETECTORS[sid](panel, asof_date, bp_mult, scope=strategy)
                rd = r.to_dict()
                _fill_legs(rd, sym)
                contract_results[sid] = rd
            except Exception as e:
                contract_results[sid] = SetupResult(
                    setup_id=sid, name=sid, family=FAMILY_TREND, scope=strategy,
                    state="N/A", error=f"detector exception: {e}",
                ).to_dict()

        # Mean-reversion (all 3 scopes per registry)
        for sid in setups_for_scope(strategy, families=(FAMILY_MR,)):
            if sid not in MR_DETECTORS:
                continue
            try:
                r = MR_DETECTORS[sid](panel, asof_date, bp_mult, scope=strategy)
                rd = r.to_dict()
                _fill_legs(rd, sym)
                contract_results[sid] = rd
            except Exception as e:
                contract_results[sid] = SetupResult(
                    setup_id=sid, name=sid, family=FAMILY_MR, scope=strategy,
                    state="N/A", error=f"detector exception: {e}",
                ).to_dict()

        # STIR-specific
        if strategy == "outright":
            # C3
            if c3_universe is not None:
                try:
                    r = detect_c3_for_contract(
                        sym, asof_date, bp_mult,
                        c3_universe["carry"], c3_universe["ema20"], c3_universe["close"],
                        c3_universe["top_decile"], c3_universe["bottom_decile"],
                    )
                    rd = r.to_dict()
                    _fill_legs(rd, sym)
                    contract_results["C3"] = rd
                except Exception as e:
                    contract_results["C3"] = SetupResult(
                        setup_id="C3", name="INTRA_CURVE_CARRY_RANK",
                        family=FAMILY_STIR, scope="outright",
                        state="N/A", error=f"detector exception: {e}",
                    ).to_dict()
        elif strategy == "fly":
            try:
                r = detect_c4(panel, asof_date, bp_mult, scope="fly")
                rd = r.to_dict()
                _fill_legs(rd, sym)
                contract_results["C4"] = rd
            except Exception as e:
                contract_results["C4"] = SetupResult(
                    setup_id="C4", name="TRADITIONAL_FLY_MR",
                    family=FAMILY_STIR, scope="fly",
                    state="N/A", error=f"detector exception: {e}",
                ).to_dict()
        elif strategy == "spread":
            try:
                r = detect_c5(panel, asof_date, bp_mult, scope="spread")
                rd = r.to_dict()
                _fill_legs(rd, sym)
                contract_results["C5"] = rd
            except Exception as e:
                contract_results["C5"] = SetupResult(
                    setup_id="C5", name="CALENDAR_SPREAD_MR",
                    family=FAMILY_STIR, scope="spread",
                    state="N/A", error=f"detector exception: {e}",
                ).to_dict()

        # Composites
        carry_bp = None
        if c3_universe is not None and sym in c3_universe.get("carry", {}):
            carry_bp = c3_universe["carry"].get(sym)
        try:
            comp = compute_composites(panel, asof_date, scope=strategy,
                                        bp_mult=bp_mult, carry_per_day_bp=carry_bp)
        except Exception as e:
            comp = {"error": str(e), "trend_score": 0.0, "mr_score": 0.0, "final_score": 0.0,
                     "trend_factors": {}, "mr_factors": {}, "weights": {}, "regime": "NEUTRAL",
                     "interpretation": "no signal", "scope": strategy}
        contract_results["_composites"] = comp
        contract_results["_meta"] = {"bp_multiplier": bp_mult,
                                      "convention": "bp" if abs(bp_mult - 1.0) < 1e-6 else "price",
                                      "carry_per_day_bp": carry_bp}
        out["by_contract"][sym] = contract_results

    # =====================================================================
    # Inject C8/C9 universe-level results into the FIRST outright contract
    # so the UI can pick them up alongside per-contract setups.
    # =====================================================================
    if strategy == "outright":
        # C8 — attach to terminal contract (most actionable label)
        for h, c8_res in c8_results.items():
            if c8_res is None:
                continue
            target_sym = c8_res.notes.split(" ·")[0] if c8_res.notes else (contracts[0] if contracts else None)
            if target_sym and target_sym in out["by_contract"]:
                rd = c8_res.to_dict()
                _fill_legs(rd, target_sym)
                out["by_contract"][target_sym][c8_res.setup_id] = rd
            elif contracts:
                rd = c8_res.to_dict()
                _fill_legs(rd, contracts[0])
                out["by_contract"][contracts[0]][c8_res.setup_id] = rd

        # C9a / C9b — multi-pair curve ladder. Each pair generates its own
        # offset-suffixed setup id (C9a_6M / C9a_12M / C9a_18M / C9a_24M, same for C9b).
        # Legs are populated inside the detector since the pair (front, back)
        # is known internally — no need to call _fill_legs.
        for (front_sym, back_sym, off) in c9_pairs:
            if not (front_sym and back_sym and front_sym in panels and back_sym in panels):
                continue
            close_history = {
                s: panels[s]["close"] if "close" in panels[s].columns else pd.Series(dtype=float)
                for s in (front_sym, back_sym)
            }
            try:
                r9a = detect_c9a(close_history, front_sym, back_sym, asof_date,
                                  offset_months=off)
                out["by_contract"].setdefault(front_sym, {})[r9a.setup_id] = r9a.to_dict()
            except Exception as e:
                out["errors"].append(f"C9a_{off}M: {e}")
            try:
                r9b = detect_c9b(close_history, front_sym, back_sym, asof_date,
                                  offset_months=off)
                out["by_contract"].setdefault(front_sym, {})[r9b.setup_id] = r9b.to_dict()
            except Exception as e:
                out["errors"].append(f"C9b_{off}M: {e}")

    # =====================================================================
    # Aggregate fires/near lists
    # =====================================================================
    for sym, results in out["by_contract"].items():
        for sid, r in results.items():
            if sid.startswith("_"):
                continue
            if not isinstance(r, dict):
                continue
            state = r.get("state")
            if state == "FIRED":
                out["fires_today"].append({
                    "symbol": sym, "setup_id": sid,
                    "direction": r.get("direction"),
                    "name": r.get("name", sid),
                    "family": r.get("family"),
                    "key_inputs": r.get("key_inputs", {}),
                    "interpretation": r.get("interpretation", ""),
                    "entry": r.get("entry"), "stop": r.get("stop"),
                    "t1": r.get("t1"), "t2": r.get("t2"),
                    "lots_at_10k_risk": r.get("lots_at_10k_risk"),
                    "legs": r.get("legs", []) or [],
                    "notes": r.get("notes", ""),
                })
            elif state in ("NEAR", "APPROACHING"):
                out["near_today"].append({
                    "symbol": sym, "setup_id": sid,
                    "direction": r.get("direction"),
                    "state": state,
                    "name": r.get("name", sid),
                    "family": r.get("family"),
                    "distance_to_fire": r.get("distance_to_fire"),
                    "eta_bars": r.get("eta_bars"),
                    "missing_condition": r.get("missing_condition"),
                    "interpretation": r.get("interpretation", ""),
                    "notes": r.get("notes", ""),
                })

    # Sort near_today by distance ascending
    out["near_today"].sort(key=lambda x: (
        x.get("distance_to_fire") if x.get("distance_to_fire") is not None
        and np.isfinite(x.get("distance_to_fire")) else 1e9))

    # =====================================================================
    # Track records (60d)
    # =====================================================================
    try:
        out["track_records"] = compute_track_record(panels, asof_date, strategy,
                                                      window_days=60,
                                                      forward_bars=5)
    except Exception as e:
        out["errors"].append(f"track record: {e}")
        out["track_records"] = {}

    return out
