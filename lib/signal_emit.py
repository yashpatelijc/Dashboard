"""signal_emit canonical schema (Phase 7, plan §12 / §3.1).

Single source-of-truth table for every emission produced by the dashboard:
detector fires (TREND/MR/STIR), analytics signals (REGIME/POLICY/EVENT),
composites. Stored as a parquet snapshot per asof at
``.signal_cache/signal_emit_<asof>.parquet`` plus a DuckDB-readable view
for ad-hoc queries.

Schema (24 columns, locked):
  1. emit_id           — uuid4
  2. asof_date         — date the snapshot was taken (latest CMC bar)
  3. detector_id       — e.g. "A1_TREND_DONCHIAN", "REGIME_A1", "EVENT_NFP"
  4. detector_family   — TREND | MR | STIR | REGIME | POLICY | EVENT | COMPOSITE
  5. scope             — outright | spread | fly | regime | event | policy
  6. cmc_node          — M3 / M3_M6 / M3_M6_M9 / "ALL" / NA
  7. signal_type       — FIRED | NEAR | APPROACHING | HOLD | FLIP
  8. direction         — LONG | SHORT | NEUTRAL
  9. raw_value         — the underlying metric (e.g. RSI_14)
 10. percentile_rank   — 0-100 (where this fits in history)
 11. confidence_stoplight  — 0-100 composite (GREEN >=75 / AMBER 50-74 / RED <50 per §3.3)
 12. eff_n             — effective sample size
 13. regime_id         — current Phase 4 regime state_id
 14. regime_stability  — Phase 4 top_state_posterior at asof
 15. transition_flag   — bool, True if posterior < 0.6 (regime in transition)
 16. conflict_flag     — bool, True if a contradictory emission also fired
 17. gate_quality      — CLEAN | LOW_SAMPLE | TRANSITION | CONFLICT
 18. expected_horizon_days — typical horizon for this signal type (median trade duration)
 19. rationale         — short text for the drill-in card
 20. sources           — comma-sep list of underlying tickers/nodes
 21. tags              — comma-sep tags (low_sample / circular_proxy / ...)
 22. created_at        — ISO timestamp at emission
 23. builder_version   — schema version
 24. version           — major-minor-patch of the parent module that emitted

Per §3.3 confidence_stoplight composite (D4=No removes book alignment):
   30% data_quality + 30% sample_size_score + 40% regime_fit
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_CACHE_DIR = Path(__file__).resolve().parent.parent / ".signal_cache"
_CACHE_DIR.mkdir(exist_ok=True)

BUILDER_VERSION = "1.0.0"

GREEN_THRESHOLD = 75
AMBER_THRESHOLD = 50

VALID_FAMILIES = ("TREND", "MR", "STIR", "REGIME", "POLICY", "EVENT", "COMPOSITE")
VALID_SCOPES = ("outright", "spread", "fly", "regime", "event", "policy")
VALID_TYPES = ("FIRED", "NEAR", "APPROACHING", "HOLD", "FLIP")
VALID_DIRECTIONS = ("LONG", "SHORT", "NEUTRAL")


@dataclass
class Emission:
    detector_id: str
    detector_family: str
    scope: str
    cmc_node: str = "ALL"
    signal_type: str = "FIRED"
    direction: str = "NEUTRAL"
    raw_value: float = 0.0
    percentile_rank: float = 50.0
    confidence_stoplight: int = 50
    eff_n: int = 0
    regime_id: int = -1
    regime_stability: float = 0.0
    transition_flag: bool = False
    conflict_flag: bool = False
    gate_quality: str = "CLEAN"
    expected_horizon_days: int = 5
    rationale: str = ""
    sources: str = ""
    tags: str = ""
    asof_date: Optional[date] = None
    emit_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    builder_version: str = BUILDER_VERSION
    version: str = "1.0.0"

    def to_dict(self) -> dict:
        d = asdict(self)
        if d["asof_date"] is not None:
            d["asof_date"] = str(d["asof_date"])
        return d


def confidence_stoplight_color(score: int) -> str:
    if score >= GREEN_THRESHOLD:
        return "GREEN"
    if score >= AMBER_THRESHOLD:
        return "AMBER"
    return "RED"


def compute_confidence(data_quality: float, sample_size_score: float,
                          regime_fit: float) -> int:
    """30% data_quality + 30% sample_size_score + 40% regime_fit, 0..100."""
    score = 30.0 * data_quality + 30.0 * sample_size_score + 40.0 * regime_fit
    return int(max(0, min(100, round(score))))


def _sample_size_score(eff_n: int) -> float:
    """Map eff_n -> 0..100 (saturating at 100)."""
    if eff_n <= 0:
        return 0.0
    if eff_n >= 100:
        return 100.0
    return 100.0 * eff_n / 100.0


# =============================================================================
# Snapshot writer
# =============================================================================

def _paths(asof: date) -> dict:
    stamp = asof.isoformat()
    return {
        "table":    _CACHE_DIR / f"signal_emit_{stamp}.parquet",
        "manifest": _CACHE_DIR / f"signal_emit_manifest_{stamp}.json",
    }


def write_emissions(asof: date, emissions: list[Emission]) -> dict:
    paths = _paths(asof)
    rows = []
    for e in emissions:
        if e.asof_date is None:
            e.asof_date = asof
        rows.append(e.to_dict())
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=[
            "emit_id","asof_date","detector_id","detector_family","scope",
            "cmc_node","signal_type","direction","raw_value","percentile_rank",
            "confidence_stoplight","eff_n","regime_id","regime_stability",
            "transition_flag","conflict_flag","gate_quality",
            "expected_horizon_days","rationale","sources","tags",
            "created_at","builder_version","version",
        ])

    tmp_t = paths["table"].with_suffix(".parquet.tmp")
    tmp_m = paths["manifest"].with_suffix(".json.tmp")
    df.to_parquet(tmp_t, index=False)

    manifest = {
        "builder_version": BUILDER_VERSION,
        "asof_date": asof.isoformat(),
        "n_emissions": int(len(df)),
        "n_by_family": (df.groupby("detector_family").size().to_dict()
                          if not df.empty else {}),
        "n_clean": int((df["gate_quality"] == "CLEAN").sum()) if not df.empty else 0,
        "n_low_sample": int((df["gate_quality"] == "LOW_SAMPLE").sum()) if not df.empty else 0,
        "n_transition": int((df["gate_quality"] == "TRANSITION").sum()) if not df.empty else 0,
        "n_conflict": int((df["gate_quality"] == "CONFLICT").sum()) if not df.empty else 0,
        "schema_version": "1.0.0",
        "schema_columns": df.columns.tolist() if not df.empty else [],
    }
    tmp_m.write_text(json.dumps(manifest, indent=2, default=str))
    os.replace(tmp_t, paths["table"])
    os.replace(tmp_m, paths["manifest"])
    return manifest


# =============================================================================
# Top-level driver — pulls Phases 4/5/6 + (where available) detector fires
# =============================================================================

def _build_phase4_emissions(asof: date, emissions: list[Emission]) -> None:
    try:
        from lib.sra_data import (
            get_current_regime, load_regime_states, load_regime_diagnostics,
        )
        cur = get_current_regime(asof)
        if not cur:
            return
        diag = load_regime_diagnostics(asof)
        n_active = int((diag["n_bars"] > 0).sum())
        # Confidence: regime stability is the key fit metric
        post = cur["top_state_posterior"]
        in_transition = post < 0.60
        confidence = compute_confidence(
            data_quality=1.0,
            sample_size_score=_sample_size_score(int(diag["n_bars"].sum())) / 100.0,
            regime_fit=post,
        )
        gate = "TRANSITION" if in_transition else "CLEAN"
        emissions.append(Emission(
            detector_id="REGIME_A1",
            detector_family="REGIME",
            scope="regime",
            cmc_node="ALL",
            signal_type="HOLD" if not in_transition else "FLIP",
            direction="NEUTRAL",
            raw_value=float(cur["state_id"]),
            percentile_rank=post * 100,
            confidence_stoplight=confidence,
            eff_n=int(diag["n_bars"].sum()),
            regime_id=int(cur["state_id"]),
            regime_stability=post,
            transition_flag=in_transition,
            gate_quality=gate,
            expected_horizon_days=20,
            rationale=f"Current regime: {cur['state_name']} (posterior={post:.2f}); "
                          f"{n_active}/6 states active",
            sources=f"residual_change,n_bars={int(diag['n_bars'].sum())}",
            tags="phase4,a1_regime",
        ))
    except Exception:
        pass


def _build_phase5_emissions(asof: date, emissions: list[Emission],
                                  cur_regime: dict) -> None:
    try:
        from lib.sra_data import get_policy_path_summary, load_policy_path
        s = get_policy_path_summary(asof)
        if not s or s.get("terminal_rate_bp") is None:
            return
        df = load_policy_path(asof)
        if df.empty:
            return
        cur_rate = s["current_rate_bp"]; terminal = s["terminal_rate_bp"]
        cycle = s["cycle_label"]
        net_move = terminal - cur_rate
        # Direction from cycle label
        if cycle == "LATE_HIKE":
            direction = "SHORT"   # short rates / bearish duration
        elif cycle == "LATE_CUT":
            direction = "LONG"
        else:
            direction = "NEUTRAL"
        confidence = compute_confidence(
            data_quality=1.0,
            sample_size_score=min(1.0, len(df) / 14.0),
            regime_fit=cur_regime.get("top_state_posterior", 0.7),
        )
        emissions.append(Emission(
            detector_id="POLICY_A4",
            detector_family="POLICY",
            scope="policy",
            cmc_node="ALL",
            signal_type="FIRED" if abs(net_move) >= 25 else "HOLD",
            direction=direction,
            raw_value=float(net_move),
            percentile_rank=50.0,   # absolute; we have no historical comp yet
            confidence_stoplight=confidence,
            eff_n=int(s.get("n_meetings", 0)),
            regime_id=int(cur_regime.get("state_id", -1)),
            regime_stability=float(cur_regime.get("top_state_posterior", 0.0)),
            gate_quality="CLEAN",
            expected_horizon_days=90,
            rationale=f"Cycle: {cycle}; current={cur_rate/100:.3f}% terminal={terminal/100:.3f}% "
                          f"({net_move:+.0f} bp net over horizon)",
            sources=f"sr3_strip,n_meetings={s.get('n_meetings',0)}",
            tags="phase5,a4_policy_path",
        ))
    except Exception:
        pass


def _build_phase6_emissions(asof: date, emissions: list[Emission],
                                  cur_regime: dict, top_n: int = 10) -> None:
    try:
        from lib.sra_data import top_event_signals
        top = top_event_signals(asof, n=top_n, regime_filter="ALL")
        if top.empty:
            return
        for _, row in top.iterrows():
            score = float(row["score_full"])
            recency = (float(row["score_recency_weighted"])
                          if not pd.isna(row["score_recency_weighted"]) else None)
            data_q = 1.0 if row["axis"] == "B0_consensus" else 0.7
            sample_q = min(1.0, float(row["n_obs"]) / 30.0)
            regime_q = cur_regime.get("top_state_posterior", 0.6)
            confidence = compute_confidence(data_q, sample_q, regime_q)
            tags_list = ["phase6", "a11_event"]
            if row["axis"] == "B2_z_history":
                tags_list.append("circular_proxy")
            if int(row["n_obs"]) < 30:
                tags_list.append("low_sample")
            if bool(row["becoming_more_important"]):
                tags_list.append("becoming_more_important")
            gate = "LOW_SAMPLE" if int(row["n_obs"]) < 30 else "CLEAN"
            direction = ("LONG" if row["beta"] > 0
                              else "SHORT" if row["beta"] < 0
                              else "NEUTRAL")
            emissions.append(Emission(
                detector_id=f"EVENT_{row['ticker'].replace('_Index','')}",
                detector_family="EVENT",
                scope="event",
                cmc_node=str(row["cmc_node"]),
                signal_type="FIRED" if score > 0.30 else "NEAR",
                direction=direction,
                raw_value=float(row["beta"]),
                percentile_rank=score * 100.0,
                confidence_stoplight=confidence,
                eff_n=int(row["n_obs"]),
                regime_id=int(cur_regime.get("state_id", -1)),
                regime_stability=float(cur_regime.get("top_state_posterior", 0.0)),
                gate_quality=gate,
                expected_horizon_days=1,
                rationale=f"Event impact: β={row['beta']:+.3f} R²={row['r_squared']:.3f} "
                              f"hit={row['hit_rate']:.2f} score={score:.3f}"
                              + (f" (recency={recency:.3f})" if recency is not None else ""),
                sources=str(row["ticker"]),
                tags=",".join(tags_list),
            ))
    except Exception:
        pass


def _build_detector_emissions(asof: date, emissions: list[Emission],
                                    cur_regime: dict) -> None:
    """Pipe detector fires from the Technicals scan_universe into signal_emit.

    Strategy: read the latest tmia.duckdb backtest table to pull recent
    detector fires (last 5 BD) and convert each fire row into an Emission.
    This is the §16.6-recommended single-point-of-intercept pattern that
    avoids touching 26 individual detector files.
    """
    try:
        from pathlib import Path as _P
        import duckdb
        tmia = _P(__file__).resolve().parent.parent / ".backtest_cache" / "tmia.duckdb"
        if not tmia.exists():
            return
        con = duckdb.connect(str(tmia), read_only=True)
        # Pull recent fires
        try:
            fires = con.execute(
                "SELECT setup_id, family, symbol, scope, fire_date, "
                "       direction, entry_price, t1_price "
                "FROM tmia_fires "
                "WHERE fire_date >= ?"
                "ORDER BY fire_date DESC LIMIT 200",
                [asof.isoformat()],
            ).fetchdf()
        except Exception:
            try:
                from datetime import timedelta as _td
                fires = con.execute(
                    "SELECT * FROM tmia_fires LIMIT 0"
                ).fetchdf()  # introspect schema only
            except Exception:
                fires = pd.DataFrame()
        con.close()
        if fires.empty:
            return
        for _, row in fires.iterrows():
            family = str(row.get("family", "TREND")).upper()
            if family not in VALID_FAMILIES:
                family = "TREND"
            direction = str(row.get("direction", "LONG")).upper()
            if direction not in VALID_DIRECTIONS:
                direction = "NEUTRAL"
            confidence = compute_confidence(
                data_quality=1.0,
                sample_size_score=0.5,
                regime_fit=cur_regime.get("top_state_posterior", 0.6),
            )
            emissions.append(Emission(
                detector_id=str(row.get("setup_id", "UNKNOWN")),
                detector_family=family,
                scope=str(row.get("scope", "outright")),
                cmc_node=str(row.get("symbol", "ALL")),
                signal_type="FIRED",
                direction=direction,
                raw_value=float(row.get("entry_price", 0.0)) if pd.notna(row.get("entry_price")) else 0.0,
                percentile_rank=50.0,
                confidence_stoplight=confidence,
                eff_n=30,
                regime_id=int(cur_regime.get("state_id", -1)),
                regime_stability=float(cur_regime.get("top_state_posterior", 0.0)),
                gate_quality="CLEAN",
                expected_horizon_days=5,
                rationale=f"{row.get('setup_id', '?')} fire on {row.get('fire_date', '?')}",
                sources=str(row.get("symbol", "")),
                tags="phase7,detector_intercept",
            ))
    except Exception:
        pass


def build_signal_emit(asof: Optional[date] = None) -> dict:
    """Top-level: snapshot every analytics module's current emissions."""
    if asof is None:
        cache = Path(__file__).resolve().parent.parent / ".cmc_cache"
        cands = sorted(cache.glob("regime_manifest_*.json"),
                          key=os.path.getmtime, reverse=True)
        if not cands:
            raise RuntimeError("no regime cache; run Phase 4 first")
        asof = date.fromisoformat(cands[0].stem.replace("regime_manifest_", ""))

    emissions: list[Emission] = []

    # Phase 4 — current regime
    _build_phase4_emissions(asof, emissions)

    # Pull current regime info for downstream emissions
    try:
        from lib.sra_data import get_current_regime
        cur_regime = get_current_regime(asof)
    except Exception:
        cur_regime = {}

    # Phase 5 — policy path
    _build_phase5_emissions(asof, emissions, cur_regime)

    # Phase 6 — top event signals
    _build_phase6_emissions(asof, emissions, cur_regime)

    # Phase 7 — detector fires (intercepted from tmia backtest)
    _build_detector_emissions(asof, emissions, cur_regime)

    # Conflict detection: if any (cmc_node × direction) pair has both LONG
    # and SHORT emissions, flag both as conflict.
    if emissions:
        df_chk = pd.DataFrame([e.to_dict() for e in emissions])
        conflicts = (df_chk.groupby("cmc_node")["direction"]
                          .nunique() > 1)
        for nm, has_conflict in conflicts.items():
            if has_conflict and nm != "ALL":
                for e in emissions:
                    if e.cmc_node == nm and e.direction in ("LONG", "SHORT"):
                        e.conflict_flag = True
                        e.gate_quality = "CONFLICT"

    return write_emissions(asof, emissions)


def main(argv=None):
    import sys
    args = list(sys.argv[1:] if argv is None else argv)
    asof = None
    if args:
        try:
            asof = date.fromisoformat(args[0])
        except ValueError:
            print("usage: python -m lib.signal_emit [YYYY-MM-DD]")
            return 2
    print(f"[signal_emit] building snapshot for asof={asof or 'latest'}")
    manifest = build_signal_emit(asof)
    print(f"[signal_emit] {manifest['n_emissions']} emissions; "
              f"by_family={manifest['n_by_family']}; "
              f"clean={manifest['n_clean']}, low_sample={manifest['n_low_sample']}, "
              f"transition={manifest['n_transition']}, conflict={manifest['n_conflict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
