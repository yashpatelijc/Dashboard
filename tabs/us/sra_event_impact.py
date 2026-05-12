"""Historical Event Impact sub-tab (5th in SRA router).

7 sections per design spec §10:
  Header: ticker / window / measurement / instrument-filter / asof selectors
  Section 1: event characteristics snapshot
  Section 2: pre-release behavior heatmap (anticipation)
  Section 3: 8-state post-release reaction matrix (centerpiece)
  Section 4: top-5 most-reactive per state (8 ranked tables)
  Section 5: 1H bar-by-bar trajectory (this instrument, 48 bars by state)
  Section 6: per-instrument event timeline scatter
  Section 7: drift comparison panel (3m / 6m / 12m / Full)
  Section 8: export footer
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st


# =============================================================================
# Data loaders (cached)
# =============================================================================

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".cmc_cache"


@st.cache_data(show_spinner=False, ttl=600)
def _resolve_latest_hei_asof() -> Optional[str]:
    cands = sorted(_CACHE_DIR.glob("hei_manifest_*.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        return None
    return cands[0].stem.replace("hei_manifest_", "")


@st.cache_data(show_spinner=False, ttl=600)
def load_hei_catalog(asof_iso: str) -> pd.DataFrame:
    p = _CACHE_DIR / f"hei_event_catalog_{asof_iso}.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["release_date"] = pd.to_datetime(df["release_date"]).dt.date
    return df


@st.cache_data(show_spinner=False, ttl=600)
def load_hei_panel(asof_iso: str) -> pd.DataFrame:
    p = _CACHE_DIR / f"hei_instrument_panel_1h_{asof_iso}.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["release_date"] = pd.to_datetime(df["release_date"]).dt.date
    return df


@st.cache_data(show_spinner=False, ttl=600)
def load_hei_aggregates(asof_iso: str) -> pd.DataFrame:
    p = _CACHE_DIR / f"hei_aggregates_{asof_iso}.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


@st.cache_data(show_spinner=False, ttl=600)
def load_hei_drift(asof_iso: str) -> pd.DataFrame:
    p = _CACHE_DIR / f"hei_drift_flags_{asof_iso}.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


@st.cache_data(show_spinner=False, ttl=600)
def load_hei_manifest(asof_iso: str) -> dict:
    p = _CACHE_DIR / f"hei_manifest_{asof_iso}.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


# =============================================================================
# Color palettes
# =============================================================================

STATE_COLORS = {
    "UP_POS_small":   "#84cc16",
    "UP_POS_large":   "#22c55e",
    "UP_NEG_small":   "#fbbf24",
    "UP_NEG_large":   "#f59e0b",
    "DOWN_POS_small": "#a78bfa",
    "DOWN_POS_large": "#8b5cf6",
    "DOWN_NEG_small": "#fb923c",
    "DOWN_NEG_large": "#ef4444",
    "FLAT":           "#94a3b8",
}

INSTRUMENT_TYPE_COLORS = {
    "outright": "#60a5fa", "spread": "#f59e0b", "fly": "#a855f7",
    "pack": "#22c55e", "pack_fly": "#ef4444",
}


# =============================================================================
# Narrative builder — auto-generated text interpretation of every analysis
# section for the selected ticker. Returns Markdown.
# =============================================================================

def _fmt_sigma(v: float) -> str:
    return f"{v:+.2f}σ_ATR" if pd.notna(v) else "n/a"


def _fmt_bp(v: float) -> str:
    return f"{v:+.1f} bp" if pd.notna(v) else "n/a"


def _segment_of_outright(instrument_id: str, pack_ranges: dict) -> Optional[str]:
    if not isinstance(instrument_id, str) or not instrument_id.startswith("M"):
        return None
    try:
        m = int(instrument_id[1:])
    except ValueError:
        return None
    for name, rng in pack_ranges.items():
        if m in rng:
            return name
    return None


def _resolve_current_symbol(instrument_id: str, contract_table: dict,
                                  asof_date, pack_ranges: dict) -> str:
    """Render the current contract symbol(s) for a generic instrument_id.

    Examples (with asof = 2026-04-27):
      'M15'                       → 'SRAU27'
      'M3-M6'                     → 'SRAU26-Z26'
      'M3-M6-M9'                  → 'SRAU26-Z26-H27'
      'pack_red'                  → 'SRAH27..SRAZ27'  (range over the 12 legs)
      'packfly_white_red_green'   → 'white-red-green' (composite — not expanded)
    Returns '' if instrument_id is unrecognised, 'n/a' if a leg is missing.
    """
    from lib.historical_event_impact import (
        lookup_contract, _construct_spread_symbol, _construct_fly_symbol,
    )
    if not isinstance(instrument_id, str) or not instrument_id:
        return ""
    # Outright 'M15'
    if instrument_id.startswith("M") and "-" not in instrument_id:
        try:
            tenor = int(instrument_id[1:])
        except ValueError:
            return ""
        sym = lookup_contract(contract_table, asof_date, tenor)
        return sym or "n/a"
    # Spread 'M3-M6' or fly 'M3-M6-M9'
    if instrument_id.startswith("M") and "-" in instrument_id:
        parts = instrument_id.split("-")
        try:
            tenors = [int(p[1:]) for p in parts]
        except ValueError:
            return ""
        syms = [lookup_contract(contract_table, asof_date, t) for t in tenors]
        if any(s is None for s in syms):
            return "n/a"
        if len(syms) == 2:
            return _construct_spread_symbol(syms[0], syms[1])
        if len(syms) == 3:
            return _construct_fly_symbol(syms)
        return "-".join(syms)
    # Pack 'pack_white' → first..last underlying outright
    if instrument_id.startswith("pack_"):
        name = instrument_id.replace("pack_", "")
        rng = pack_ranges.get(name)
        if rng is None:
            return ""
        s_lo = lookup_contract(contract_table, asof_date, rng.start)
        s_hi = lookup_contract(contract_table, asof_date, rng.stop - 1)
        if s_lo and s_hi:
            return f"{s_lo}..{s_hi}"
        return name
    # Pack-fly 'packfly_white_red_green' — too verbose to expand into 36 legs
    if instrument_id.startswith("packfly_"):
        return instrument_id.replace("packfly_", "").replace("_", "-")
    return ""


def _fmt_instr(instrument_id: str, contract_table: dict,
                  asof_date, pack_ranges: dict) -> str:
    """Render `instrument_id` followed by its current contract symbol in
    brackets, e.g. '`M15` [SRAU27]' or '`M3-M6` [SRAU26-Z26]'."""
    current = _resolve_current_symbol(instrument_id, contract_table,
                                              asof_date, pack_ranges)
    if not current or current == "n/a":
        return f"`{instrument_id}`"
    return f"`{instrument_id}` [{current}]"


def build_event_narrative(
    ticker: str,
    ticker_label: str,
    cat_t: pd.DataFrame,
    aggregates_full: pd.DataFrame,
    drift_full: pd.DataFrame,
    panel_t: pd.DataFrame,
    measurement: str,
    window: str,
    states_8: tuple,
    pack_ranges: dict,
    contract_table: Optional[dict] = None,
    current_asof: Optional[date] = None,
) -> str:
    """Build a detailed Markdown narrative interpreting every analytic
    section for the currently-selected ticker. Synthesizes:
       - event-level characteristics (counts, surprise dynamics, state mix)
       - most-recent release spotlight
       - pre-release positioning (anticipation by quadrant)
       - per-state post-release reactions (with named instruments)
       - trajectory shape (release impulse vs full-day vs T+1)
       - drift trends (FADING / GROWING / REVERSING / STABLE)
       - curve reactivity (family, segment, evolution across windows)
       - actionable trade-relevance takeaways
    """
    if cat_t.empty:
        return "_No events to interpret._"

    # Build a current-contract lookup lazily so every generic instrument_id
    # (M15 / M3-M6 / pack_red / packfly_white_red_green) can be annotated with
    # its current tradeable symbol in brackets. Asof = most recent release
    # in catalog (matches what the user is seeing in Section 6 timeline).
    # Normalize release_date dtype to plain `date` on both cat_t and panel_t
    # so downstream merges and date arithmetic work regardless of upstream
    # Timestamp/datetime/string inputs.
    cat_t = cat_t.copy()
    cat_t["release_date"] = pd.to_datetime(cat_t["release_date"]).dt.date
    if isinstance(panel_t, pd.DataFrame) and not panel_t.empty and "release_date" in panel_t.columns:
        panel_t = panel_t.copy()
        panel_t["release_date"] = pd.to_datetime(panel_t["release_date"]).dt.date

    if contract_table is None:
        try:
            from lib.historical_event_impact import build_contract_lookup
            contract_table = build_contract_lookup(current_asof)
        except Exception:
            contract_table = {}
    if current_asof is None:
        try:
            current_asof = cat_t["release_date"].max()
        except Exception:
            current_asof = date.today()

    def fmt(instrument_id: str) -> str:
        return _fmt_instr(instrument_id, contract_table, current_asof, pack_ranges)

    agg_tw = aggregates_full[
        (aggregates_full["ticker"] == ticker)
        & (aggregates_full["window"] == window)
        & (aggregates_full["measurement"] == measurement)
    ]
    agg_all_w = aggregates_full[(aggregates_full["ticker"] == ticker)]
    drift_t = drift_full[drift_full["ticker"] == ticker] if not drift_full.empty else pd.DataFrame()

    lines: list[str] = []

    # -----------------------------------------------------------------
    # §A. Event-level snapshot
    # -----------------------------------------------------------------
    n_events = int(len(cat_t))
    last_date = cat_t["release_date"].max()
    first_date = cat_t["release_date"].min()
    consensus_acc = float((cat_t["surprise_z"].abs() < 0.5).mean()) if n_events else 0.0
    avg_abs_surprise = float(cat_t["surprise_z"].abs().mean())
    avg_abs_expected = float(cat_t["expected_z"].abs().mean())
    pos_surprise_rate = float((cat_t["surprise_sign"] == "POS").mean())
    state_counts = cat_t["state_8"].value_counts()
    if not state_counts.empty:
        top_state = state_counts.idxmax()
        top_state_n = int(state_counts.max())
    else:
        top_state, top_state_n = "n/a", 0
    n_flat = int((cat_t["state_8"] == "FLAT").sum())

    lines.append(f"### §A · Event-level snapshot — {ticker_label} (`{ticker}`)")
    lines.append(
        f"- **Coverage:** {n_events} historical releases between **{first_date}** and **{last_date}**."
    )
    lines.append(
        f"- **Consensus quality:** {consensus_acc:.0%} of releases came in within ±0.5σ of expectations "
        f"(avg `|surprise z|` = {avg_abs_surprise:.2f}, avg `|expected z|` = {avg_abs_expected:.2f}). "
        f"Positive surprises occurred {pos_surprise_rate:.0%} of the time — "
        f"{'a hawkish skew' if pos_surprise_rate > 0.55 else 'a dovish skew' if pos_surprise_rate < 0.45 else 'roughly symmetric'}."
    )
    lines.append(
        f"- **Dominant state:** `{top_state}` ({top_state_n}/{n_events} releases, "
        f"{top_state_n / n_events:.0%}). FLAT count: {n_flat}."
    )
    # Full state distribution table
    lines.append(
        "- **Full state distribution:** "
        + " · ".join(
            f"`{s}`={int(state_counts.get(s, 0))}" for s in states_8
        )
        + (f" · `FLAT`={n_flat}" if n_flat > 0 else "")
    )
    # Expected-direction split
    exp_up_rate = float((cat_t["expected_direction"] == "UP").mean())
    surprise_size_large_rate = float((cat_t["surprise_size"] == "large").mean())
    lines.append(
        f"- **Expectation regime:** {exp_up_rate:.0%} of releases were preceded by an UP-rate "
        f"expectation (consensus > prior); {1 - exp_up_rate:.0%} preceded by DOWN. "
        f"{surprise_size_large_rate:.0%} of releases produced a large-magnitude surprise (|z|≥1)."
    )
    if not state_counts.empty:
        thin = [s for s in states_8 if state_counts.get(s, 0) < 3]
        if thin:
            lines.append(
                f"- **Sample-thin states (n<3, treat with caution):** "
                f"{', '.join(f'`{s}`' for s in thin)}."
            )

    # -----------------------------------------------------------------
    # §B. Most-recent release in spotlight
    # -----------------------------------------------------------------
    last_row = cat_t.sort_values("release_date").iloc[-1]
    last_state = last_row["state_8"]
    lines.append("")
    lines.append("### §B · Most-recent release spotlight")
    lines.append(
        f"- **Date:** {last_row['release_date']} · "
        f"prior = {last_row['prior_actual']:.2f}, "
        f"consensus = {last_row['consensus']:.2f}, "
        f"actual = {last_row['actual']:.2f}."
    )
    lines.append(
        f"- **Expectation:** {last_row['expected_direction']} "
        f"(`expected_z` = {last_row['expected_z']:+.2f}). "
        f"**Surprise:** `{last_row['surprise_sign']}` / `{last_row['surprise_size']}` "
        f"(`surprise_z` = {last_row['surprise_z']:+.2f})."
    )
    lines.append(
        f"- **State classification:** `{last_state}`."
    )
    # Pull most-reactive instrument for that state from agg_tw if available
    if not agg_tw.empty:
        st_slice = agg_tw[(agg_tw["state_8"] == last_state) & (agg_tw["n_obs"] >= 3)]
        if not st_slice.empty:
            top_row = st_slice.reindex(
                st_slice["median_norm"].abs().sort_values(ascending=False).index
            ).iloc[0]
            lines.append(
                f"- **Historical playbook for `{last_state}` (`{measurement}`, {window}):** "
                f"the most-reactive instrument was **{fmt(top_row['instrument_id'])}** "
                f"({top_row['instrument_type']}), median move "
                f"{_fmt_sigma(top_row['median_norm'])} "
                f"({_fmt_bp(top_row['median_bp'])}) over n={int(top_row['n_obs'])} prior fires."
            )

    # -----------------------------------------------------------------
    # §C. Pre-release positioning (anticipation_full quadrants)
    # -----------------------------------------------------------------
    lines.append("")
    lines.append("### §C · Pre-release positioning")
    pre_agg = agg_all_w[
        (agg_all_w["measurement"] == "anticipation_full")
        & (agg_all_w["window"] == window)
        & (agg_all_w["n_obs"] >= 3)
    ]
    if pre_agg.empty:
        lines.append(
            "- No sufficient-sample anticipation rows for this window. "
            "Pre-release behavior insufficient for inference."
        )
    else:
        # Group by expected direction × surprise size from state name
        # state_8 format: '{UP|DOWN}_{POS|NEG}_{small|large}'
        quad_lines = []
        for exp_dir in ("UP", "DOWN"):
            for size in ("small", "large"):
                states_in_quad = [s for s in states_8
                                          if s.startswith(exp_dir) and s.endswith(size)]
                quad = pre_agg[pre_agg["state_8"].isin(states_in_quad)]
                if quad.empty:
                    continue
                # Most-anticipated instrument in this quadrant (max |median_norm|)
                quad = quad.copy()
                quad["abs_norm"] = quad["median_norm"].abs()
                top = quad.sort_values("abs_norm", ascending=False).iloc[0]
                direction_word = ("rates UP" if exp_dir == "UP" else "rates DOWN")
                # In SR3 outright: price up = rates down, so sign meaning depends
                sign_word = (
                    "price down (rates up)" if top["median_norm"] < 0
                    else "price up (rates down)"
                )
                quad_lines.append(
                    f"  - **Expecting {direction_word}, surprise mag = {size}:** "
                    f"{fmt(top['instrument_id'])} ({top['instrument_type']}) pre-prices "
                    f"{_fmt_sigma(top['median_norm'])} on average — {sign_word}."
                )
        if quad_lines:
            lines.append(
                "- Strongest anticipation-window movers (T-1 open → T pre-release) per quadrant:"
            )
            lines.extend(quad_lines)
        else:
            lines.append(
                "- No clear pre-release positioning emerged across the 4 expected×size quadrants."
            )

    # -----------------------------------------------------------------
    # §D. Post-release reaction per state (selected measurement, window)
    # -----------------------------------------------------------------
    lines.append("")
    lines.append(f"### §D · Post-release reaction per state — `{measurement}` · {window}")
    if agg_tw.empty:
        lines.append("- No sufficient-sample rows for this measurement / window combination.")
    else:
        n_state_lines_before = len(lines)
        for st_name in states_8:
            st_slice = agg_tw[(agg_tw["state_8"] == st_name) & (agg_tw["n_obs"] >= 3)]
            if st_slice.empty:
                continue
            st_slice = st_slice.copy()
            st_slice["abs_norm"] = st_slice["median_norm"].abs()
            # Top-3 (was 2) — more depth
            top3 = st_slice.sort_values("abs_norm", ascending=False).head(3)
            mean_n_obs = int(top3["n_obs"].mean())
            # IQR width = (iqr_high - iqr_low) for the top instrument — proxy for noise
            top_row = top3.iloc[0]
            iqr_width = float(top_row["iqr_high"] - top_row["iqr_low"]) if pd.notna(top_row["iqr_high"]) else float("nan")
            consistency = (
                "high" if abs(top_row["median_norm"]) > iqr_width
                else "moderate" if abs(top_row["median_norm"]) * 2 > iqr_width
                else "low"
            ) if pd.notna(iqr_width) and iqr_width > 0 else "n/a"
            top_names = []
            for _, r in top3.iterrows():
                top_names.append(
                    f"{fmt(r['instrument_id'])} ({_fmt_sigma(r['median_norm'])}, "
                    f"{_fmt_bp(r['median_bp'])})"
                )
            lines.append(
                f"- **`{st_name}`** (n~{mean_n_obs}, signal-consistency: {consistency}, "
                f"top-instrument IQR width = {iqr_width:.2f}σ): most-reactive — "
                + " · ".join(top_names)
            )
        if len(lines) == n_state_lines_before:
            lines.append(
                "- All 8 states have n<3 for this measurement / window — "
                "most events resolved into `FLAT` (see §A state distribution)."
            )

        # High-level pattern: aligned vs counter-trend strength
        aligned_states = ["UP_POS_large", "UP_POS_small",
                              "DOWN_NEG_large", "DOWN_NEG_small"]
        counter_states = ["UP_NEG_large", "UP_NEG_small",
                              "DOWN_POS_large", "DOWN_POS_small"]
        aligned = agg_tw[(agg_tw["state_8"].isin(aligned_states))
                                  & (agg_tw["n_obs"] >= 3)]["median_norm"].abs().mean()
        counter = agg_tw[(agg_tw["state_8"].isin(counter_states))
                                  & (agg_tw["n_obs"] >= 3)]["median_norm"].abs().mean()
        if pd.notna(aligned) and pd.notna(counter):
            verdict = ("**stronger** when surprise aligns with expectation"
                          if aligned > counter * 1.15
                          else "**stronger** when surprise contradicts expectation"
                          if counter > aligned * 1.15
                          else "**symmetric** between aligned and counter-trend states")
            lines.append(
                f"- **Reaction asymmetry:** average reaction magnitude is "
                f"{aligned:.2f}σ (aligned) vs {counter:.2f}σ (counter-trend) → {verdict}."
            )

    # -----------------------------------------------------------------
    # §E. Trajectory dynamics (release impulse → full window)
    # -----------------------------------------------------------------
    lines.append("")
    lines.append("### §E · Trajectory dynamics")
    if panel_t.empty:
        lines.append("- Per-event panel unavailable.")
    else:
        # Average normalized moves across each measurement for this ticker
        cols_to_check = ["release_impulse_norm", "release_immediate_2h_norm",
                              "release_day_end_norm", "next_day_full_norm",
                              "full_t_to_tp1_norm", "anticipation_full_norm"]
        avg = {c: float(panel_t[c].dropna().abs().median())
                  if c in panel_t.columns and panel_t[c].notna().any()
                  else float("nan")
                  for c in cols_to_check}
        lines.append(
            f"- **Anticipation (T-1 open → T pre):** median |move| = "
            f"{avg.get('anticipation_full_norm', float('nan')):.2f}σ_ATR."
        )
        lines.append(
            f"- **Release impulse (T pre → T+1h):** median |move| = "
            f"{avg.get('release_impulse_norm', float('nan')):.2f}σ_ATR."
        )
        lines.append(
            f"- **2-hour reaction (T pre → T+2h):** median |move| = "
            f"{avg.get('release_immediate_2h_norm', float('nan')):.2f}σ_ATR."
        )
        lines.append(
            f"- **End-of-day (T pre → T close):** median |move| = "
            f"{avg.get('release_day_end_norm', float('nan')):.2f}σ_ATR."
        )
        lines.append(
            f"- **T+1 close (continuation):** median |move| over full T pre→T+1 close = "
            f"{avg.get('full_t_to_tp1_norm', float('nan')):.2f}σ_ATR."
        )
        # Persistence vs reversal
        if (pd.notna(avg["release_impulse_norm"])
                and pd.notna(avg["release_day_end_norm"])):
            ratio = avg["release_day_end_norm"] / avg["release_impulse_norm"]
            if ratio > 1.20:
                shape_verdict = ("**continues into the close** — initial impulse "
                                          "tends to extend; consider holding through end-of-day.")
            elif ratio < 0.80:
                shape_verdict = ("**fades into the close** — initial impulse tends "
                                          "to partially reverse; mean-reversion opportunities post-impulse.")
            else:
                shape_verdict = "**stabilizes** after the 1h impulse (close ≈ T+1h)."
            lines.append(f"- **Shape verdict:** the day-end move {shape_verdict}")
        if (pd.notna(avg["full_t_to_tp1_norm"])
                and pd.notna(avg["release_day_end_norm"])):
            if avg["full_t_to_tp1_norm"] > avg["release_day_end_norm"] * 1.20:
                lines.append(
                    "- **T+1 continuation:** move grows through next day — directional "
                    "drift is sticky beyond the release session."
                )
            elif avg["full_t_to_tp1_norm"] < avg["release_day_end_norm"] * 0.80:
                lines.append(
                    "- **T+1 fade:** move retraces overnight — close-of-day positions "
                    "are at risk of giving back gains."
                )

    # -----------------------------------------------------------------
    # §F. Drift trends — what's evolving
    # -----------------------------------------------------------------
    lines.append("")
    lines.append("### §F · Drift trends (3m vs full history)")
    if drift_t.empty:
        lines.append("- No drift records available.")
    else:
        dt = drift_t[drift_t["measurement"] == measurement].copy()
        if dt.empty:
            lines.append(f"- No drift rows for `{measurement}` on this ticker.")
        else:
            tag_counts = dt["drift_tag"].value_counts().to_dict()
            n_dt = int(len(dt))
            lines.append(
                f"- **Drift mix (`{measurement}`):** "
                + ", ".join(f"`{k}` = {v} ({v / n_dt:.0%})"
                              for k, v in tag_counts.items())
                + f" across {n_dt} (state × instrument) rows."
            )
            # Highlight the strongest GROWING and FADING signals
            dt["abs_recency"] = dt["recency_index_3m_vs_full"].abs()
            growing = dt[dt["drift_tag"] == "GROWING"].sort_values(
                "abs_recency", ascending=False).head(3)
            fading = dt[dt["drift_tag"] == "FADING"].sort_values(
                "abs_recency", ascending=False).head(3)
            reversing = dt[dt["drift_tag"] == "REVERSING"].sort_values(
                "abs_recency", ascending=False).head(3)
            if not growing.empty:
                names = " · ".join(
                    f"{fmt(r['instrument_id'])} in `{r['state_8']}` "
                    f"({r['median_3m']:+.2f}σ vs {r['median_full']:+.2f}σ full)"
                    for _, r in growing.iterrows()
                )
                lines.append(f"- **Strongest GROWING signals:** {names}.")
            if not fading.empty:
                names = " · ".join(
                    f"{fmt(r['instrument_id'])} in `{r['state_8']}` "
                    f"({r['median_3m']:+.2f}σ vs {r['median_full']:+.2f}σ full)"
                    for _, r in fading.iterrows()
                )
                lines.append(f"- **Strongest FADING signals:** {names}.")
            if not reversing.empty:
                names = " · ".join(
                    f"{fmt(r['instrument_id'])} in `{r['state_8']}` "
                    f"({r['median_3m']:+.2f}σ vs {r['median_full']:+.2f}σ full)"
                    for _, r in reversing.iterrows()
                )
                lines.append(
                    f"- **REVERSING (sign-flip) signals — recent behavior contradicts history:** {names}."
                )

    # -----------------------------------------------------------------
    # §G. Curve reactivity (family, segment, evolution)
    # -----------------------------------------------------------------
    lines.append("")
    lines.append("### §G · Curve reactivity — where the action concentrates")
    if agg_tw.empty:
        lines.append("- No reactivity data for this measurement / window.")
    else:
        agg_tw_pos = agg_tw[(agg_tw["n_obs"] >= 3)].copy()
        agg_tw_pos["abs_norm"] = agg_tw_pos["median_norm"].abs()

        # Family ranking
        fam_rank = agg_tw_pos.groupby("instrument_type")["abs_norm"].mean().sort_values(ascending=False)
        if not fam_rank.empty:
            top_fam = fam_rank.idxmax()
            lines.append(
                f"- **Most-reactive family** (mean |σ_ATR| across all states & instruments): "
                f"`{top_fam}` ({fam_rank.max():.2f}σ). "
                f"Ranking: " + " > ".join(
                    f"`{fam}` ({v:.2f})" for fam, v in fam_rank.items()
                ) + "."
            )
        # Outright segment ranking
        outrights = agg_tw_pos[agg_tw_pos["instrument_type"] == "outright"].copy()
        if not outrights.empty:
            outrights["segment"] = outrights["instrument_id"].map(
                lambda x: _segment_of_outright(x, pack_ranges))
            outrights = outrights.dropna(subset=["segment"])
            if not outrights.empty:
                seg_rank = outrights.groupby("segment")["abs_norm"].mean()
                seg_ordered = [s for s in ["white", "red", "green", "blue", "gold"]
                                          if s in seg_rank.index]
                seg_rank = seg_rank.reindex(seg_ordered)
                top_seg = seg_rank.idxmax()
                seg_meaning = {
                    "white": "front (M0-M11, near-Fed)",
                    "red":   "M12-M23 (1-2y forward)",
                    "green": "M24-M35 (2-3y forward)",
                    "blue":  "M36-M47 (3-4y forward)",
                    "gold":  "M48-M59 (4-5y forward)",
                }
                lines.append(
                    f"- **Curve segment heat:** strongest reactivity in **{top_seg}** "
                    f"({seg_meaning[top_seg]}) at {seg_rank.max():.2f}σ. "
                    f"Full ranking: " + " > ".join(
                        f"{s} ({seg_rank[s]:.2f})" for s in seg_ordered
                    ) + "."
                )
                if top_seg == "white":
                    lines.append(
                        "- **Implication:** the front of the curve absorbs most of "
                        "the release — Fed-near sensitivity dominates; back-end is anchored."
                    )
                elif top_seg in ("blue", "gold"):
                    lines.append(
                        "- **Implication:** the back of the curve is reactive — "
                        "term-premium / cycle expectations are doing the work; front-end is sticky."
                    )

        # Family evolution across windows
        evo = aggregates_full[
            (aggregates_full["ticker"] == ticker)
            & (aggregates_full["measurement"] == measurement)
            & (aggregates_full["n_obs"] >= 3)
        ].copy()
        if not evo.empty:
            evo["abs_norm"] = evo["median_norm"].abs()
            evo_grp = evo.groupby(["instrument_type", "window"])["abs_norm"].mean().unstack()
            if {"3m", "full"}.issubset(evo_grp.columns):
                deltas = (evo_grp["3m"] - evo_grp["full"]) / evo_grp["full"]
                rising = deltas[deltas > 0.20].sort_values(ascending=False)
                falling = deltas[deltas < -0.20].sort_values()
                if not rising.empty:
                    parts = " · ".join(
                        f"`{fam}` (+{rel * 100:.0f}%)" for fam, rel in rising.items()
                    )
                    lines.append(
                        f"- **Reactivity RISING (3m vs full):** {parts} — these families "
                        f"are reacting more strongly in the recent regime."
                    )
                if not falling.empty:
                    parts = " · ".join(
                        f"`{fam}` ({rel * 100:.0f}%)" for fam, rel in falling.items()
                    )
                    lines.append(
                        f"- **Reactivity FALLING (3m vs full):** {parts} — historically-active "
                        f"families are quieter recently."
                    )
                if rising.empty and falling.empty:
                    lines.append(
                        "- **Reactivity evolution:** all families within ±20% of historical "
                        "average (no material 3m-vs-full drift in reactivity strength)."
                    )

    # -----------------------------------------------------------------
    # §H. Trade-relevance takeaways
    # -----------------------------------------------------------------
    lines.append("")
    lines.append("### §H · Trade-relevance takeaways")
    takeaways: list[str] = []
    if not agg_tw.empty:
        # Build a generic per-state playbook (n_obs >= 3, mark <5 as low-confidence)
        for st_name in states_8:
            st_slice = agg_tw[(agg_tw["state_8"] == st_name) & (agg_tw["n_obs"] >= 3)]
            if st_slice.empty:
                continue
            st_slice = st_slice.copy()
            st_slice["abs_norm"] = st_slice["median_norm"].abs()
            top = st_slice.sort_values("abs_norm", ascending=False).iloc[0]
            direction_word = ("LONG" if top["median_norm"] > 0 else "SHORT")
            n_top = int(top["n_obs"])
            confidence_tag = " · _low_sample_" if n_top < 5 else ""
            takeaways.append(
                f"- When `{ticker_label}` resolves into **`{st_name}`** "
                f"(n={n_top} prior fires{confidence_tag}), the playbook: "
                f"**{direction_word} {fmt(top['instrument_id'])}** — "
                f"median {_fmt_sigma(top['median_norm'])} "
                f"({_fmt_bp(top['median_bp'])}) over `{measurement}`."
            )
    if not drift_t.empty:
        dt = drift_t[drift_t["measurement"] == measurement]
        reversing = dt[dt["drift_tag"] == "REVERSING"]
        if len(reversing) > 0:
            takeaways.append(
                f"- **Caveat:** {len(reversing)} (state × instrument) cells flagged "
                f"`REVERSING` — recent behavior contradicts the historical sign. "
                f"De-rate conviction on directional plays where the historical edge "
                f"sits on a reversed signal."
            )
    if takeaways:
        lines.extend(takeaways[:8])  # cap at 8 to keep readable
    else:
        lines.append("- Insufficient sample to extract directional playbooks for this combination.")

    # -----------------------------------------------------------------
    # §I. Surprise → Reaction sensitivity (correlation analysis)
    # -----------------------------------------------------------------
    lines.append("")
    lines.append("### §I · Surprise → Reaction sensitivity")
    if panel_t.empty or measurement + "_norm" not in panel_t.columns:
        lines.append("- Per-event panel unavailable for sensitivity calculation.")
    else:
        # Merge per-event measurement values with cat_t surprise_z by release_date.
        # Average over instruments (per-event) to get a single reaction-value per release.
        meas_col = f"{measurement}_norm"
        per_event = (
            panel_t.dropna(subset=[meas_col])
            .groupby("release_date")
            .agg(
                mean_abs_norm=(meas_col, lambda s: float(s.abs().mean())),
                signed_norm=(meas_col, "mean"),
                n_instruments=(meas_col, "count"),
            )
            .reset_index()
        )
        sur = (
            cat_t[["release_date", "surprise_z", "expected_z", "state_8"]]
            .drop_duplicates(subset="release_date", keep="last")
        )
        merged = per_event.merge(sur, on="release_date", how="inner")
        if len(merged) < 3:
            lines.append("- Not enough joined (event × measurement) rows to compute sensitivity.")
        else:
            corr_abs = float(merged["surprise_z"].abs().corr(merged["mean_abs_norm"]))
            corr_signed = float(merged["surprise_z"].corr(merged["signed_norm"]))
            lines.append(
                f"- **|surprise| → |reaction| correlation (Pearson):** {corr_abs:+.2f} "
                f"(n={len(merged)} events). "
                + ("Strong — bigger surprises ⇒ bigger moves." if abs(corr_abs) > 0.5
                      else "Moderate — surprise magnitude is a useful signal."
                      if abs(corr_abs) > 0.25
                      else "Weak — reaction magnitude is largely independent of surprise size.")
            )
            lines.append(
                f"- **Signed surprise → signed reaction:** {corr_signed:+.2f}. "
                + ("Reactions follow surprise sign cleanly." if corr_signed > 0.4
                      else "Reactions invert surprise sign (counter-intuitive — possible mean-reversion)."
                      if corr_signed < -0.4
                      else "No consistent directional pass-through.")
            )
            # Largest historical reactions
            top_by_abs = merged.reindex(
                merged["mean_abs_norm"].abs().sort_values(ascending=False).index
            ).head(3)
            for _, r in top_by_abs.iterrows():
                lines.append(
                    f"  - {r['release_date']}: surprise_z = {r['surprise_z']:+.2f}, "
                    f"reaction = {r['signed_norm']:+.2f}σ_ATR averaged across "
                    f"{int(r['n_instruments'])} instruments, state `{r['state_8']}`."
                )

    # -----------------------------------------------------------------
    # §J. Window-by-window breakdown (3m / 6m / 12m / full)
    # -----------------------------------------------------------------
    lines.append("")
    lines.append(f"### §J · Window-by-window breakdown — `{measurement}`")
    evo_per_state = aggregates_full[
        (aggregates_full["ticker"] == ticker)
        & (aggregates_full["measurement"] == measurement)
        & (aggregates_full["n_obs"] >= 3)
    ].copy()
    if evo_per_state.empty:
        lines.append("- No data across windows for this measurement.")
    else:
        # For each state, build a "median across instruments per window" snapshot.
        evo_per_state["abs_norm"] = evo_per_state["median_norm"].abs()
        for st_name in states_8:
            st_evo = evo_per_state[evo_per_state["state_8"] == st_name]
            if st_evo.empty:
                continue
            win_med = st_evo.groupby("window")["abs_norm"].mean()
            wins_with_data = [w for w in ["3m", "6m", "12m", "full"] if w in win_med.index]
            if not wins_with_data:
                continue
            parts = [f"{w}={win_med[w]:.2f}σ" for w in wins_with_data]
            # Highlight the strongest window
            strongest_window = max(wins_with_data, key=lambda w: win_med[w])
            lines.append(
                f"- **`{st_name}`**: " + " · ".join(parts)
                + f" — strongest in **{strongest_window}** window."
            )
        # Window-level summary across all states
        win_overall = evo_per_state.groupby("window")["abs_norm"].mean()
        win_order = [w for w in ["3m", "6m", "12m", "full"] if w in win_overall.index]
        if win_order:
            best_overall = max(win_order, key=lambda w: win_overall[w])
            lines.append(
                "- **Window-level summary (all states):** "
                + " · ".join(f"{w}={win_overall[w]:.2f}σ" for w in win_order)
                + f". Strongest overall reactivity in **{best_overall}**."
            )

    # -----------------------------------------------------------------
    # §K. Cross-measurement signal strength (which measurement is best?)
    # -----------------------------------------------------------------
    lines.append("")
    lines.append("### §K · Cross-measurement signal strength")
    cross = aggregates_full[
        (aggregates_full["ticker"] == ticker)
        & (aggregates_full["window"] == window)
        & (aggregates_full["n_obs"] >= 3)
    ].copy()
    if cross.empty:
        lines.append("- No data for cross-measurement comparison.")
    else:
        cross["abs_norm"] = cross["median_norm"].abs()
        meas_strength = cross.groupby("measurement")["abs_norm"].mean().sort_values(ascending=False)
        top_meas_lines = []
        for meas_name, val in meas_strength.items():
            tag = " **(current)**" if meas_name == measurement else ""
            top_meas_lines.append(f"`{meas_name}` = {val:.2f}σ{tag}")
        lines.append(
            "- **Mean |σ_ATR| reaction across instruments × states, by measurement type:**"
        )
        lines.append("  - " + " · ".join(top_meas_lines))
        best_meas = meas_strength.idxmax()
        if best_meas != measurement:
            lines.append(
                f"- The strongest signal for this ticker is **`{best_meas}`** "
                f"({meas_strength.max():.2f}σ avg), not the currently-selected "
                f"`{measurement}` ({meas_strength.get(measurement, float('nan')):.2f}σ avg). "
                f"Consider switching the measurement selector to surface stronger setups."
            )
        else:
            lines.append(
                f"- The current measurement `{measurement}` is the strongest "
                f"signal for this ticker — well-chosen."
            )
        # Anticipation vs reaction comparison
        ant_meas = ["anticipation_full", "anticipation_overnight", "anticipation_morning"]
        rea_meas = ["release_impulse", "release_immediate_2h", "release_short_4h",
                          "release_day_end"]
        ant_avg = float(meas_strength.reindex(
            [m for m in ant_meas if m in meas_strength.index]).mean())
        rea_avg = float(meas_strength.reindex(
            [m for m in rea_meas if m in meas_strength.index]).mean())
        if pd.notna(ant_avg) and pd.notna(rea_avg):
            if rea_avg > ant_avg * 1.3:
                lines.append(
                    f"- **Anticipation vs reaction split:** post-release moves "
                    f"({rea_avg:.2f}σ avg) clearly dominate pre-release positioning "
                    f"({ant_avg:.2f}σ avg) — this is a **release-driven** ticker."
                )
            elif ant_avg > rea_avg * 1.3:
                lines.append(
                    f"- **Anticipation vs reaction split:** pre-release positioning "
                    f"({ant_avg:.2f}σ avg) outweighs post-release reaction "
                    f"({rea_avg:.2f}σ avg) — this is an **anticipated** ticker "
                    f"(consensus is well-priced before print)."
                )
            else:
                lines.append(
                    f"- **Anticipation vs reaction split:** anticipation ({ant_avg:.2f}σ) "
                    f"and reaction ({rea_avg:.2f}σ) are roughly comparable — "
                    f"the curve digests this ticker in both phases."
                )

    # -----------------------------------------------------------------
    # §L. Per-historical-event roll-call (last 15 events)
    # -----------------------------------------------------------------
    lines.append("")
    lines.append("### §L · Per-historical-event roll-call")
    if cat_t.empty or panel_t.empty:
        lines.append("- No per-event data available.")
    else:
        meas_col = f"{measurement}_norm"
        bp_col = f"{measurement}_bp"
        # Build per-event summary: top-instrument by |meas| per release_date
        if meas_col in panel_t.columns:
            roll = []
            sorted_cat = cat_t.sort_values("release_date", ascending=False).head(15)
            for _, ev in sorted_cat.iterrows():
                ev_panel = panel_t[(panel_t["release_date"] == ev["release_date"])
                                              & (panel_t[meas_col].notna())].copy()
                if ev_panel.empty:
                    continue
                ev_panel["abs_norm"] = ev_panel[meas_col].abs()
                top_instr = ev_panel.sort_values("abs_norm", ascending=False).iloc[0]
                roll.append(
                    f"  - **{ev['release_date']}** · state `{ev['state_8']}` · "
                    f"surprise_z={ev['surprise_z']:+.2f} · "
                    f"top reactor {fmt(top_instr['instrument_id'])} "
                    f"({top_instr[meas_col]:+.2f}σ, "
                    f"{top_instr[bp_col]:+.1f} bp)"
                )
            if roll:
                lines.append(
                    f"- Most recent 15 events (chronological, newest first) — "
                    f"each shows: date · state · surprise_z · top-reactor in `{measurement}`:"
                )
                lines.extend(roll)
            else:
                lines.append("- No events have populated reaction data in the panel.")
        else:
            lines.append(f"- `{measurement}_norm` column not present in panel.")

    lines.append("")
    lines.append(
        f"_Generated from {n_events} historical events, measurement = `{measurement}`, "
        f"window = `{window}`. Threshold: n_obs ≥ 3 (bullets with n_obs < 5 are tagged `low_sample`)._"
    )

    return "\n".join(lines)


# =============================================================================
# Main render function (called from sra.py router)
# =============================================================================

def render(asof_input: Optional[date] = None, base_product: str = "SRA") -> None:
    from lib.historical_event_impact import (
        TICKERS_13, TICKER_LABELS, MEASUREMENTS, STATES_8, WINDOWS,
        PACK_RANGES, get_tickers_for_market, get_labels_for_market,
    )
    from lib.historical_event_impact_daemon import get_hei_status
    from lib.markets import get_market as _gm

    market_cfg = _gm(base_product)
    region_label = market_cfg.get("description", base_product)
    cb_name = market_cfg.get("central_bank", "?")

    # Market-specific event ticker registry (CPI / wages / GDP / etc. relevant
    # to THIS market's central bank reaction function — not US releases).
    market_tickers = list(get_tickers_for_market(base_product))
    market_labels = dict(get_labels_for_market(base_product))

    st.markdown(f"## 📅 Historical Event Impact — {base_product}")
    if base_product == "SRA":
        st.caption(
            "1H-resolution intraday analysis of how each SR3 instrument behaves "
            "around 13 major US economic releases. Same-day pre/post "
            "(T-1 / T / T+1) with 8-state outcome bucketing + 3m/6m/12m drift "
            "detection. Coverage: events from 2024-07-15 onwards."
        )
    else:
        st.caption(
            f"1H-resolution intraday analysis of how each {base_product} instrument "
            f"behaves around {len(market_tickers)} {cb_name}-relevant economic releases "
            f"({region_label}). Same-day pre/post (T-1 / T / T+1) with 8-state "
            f"outcome bucketing + 3m/6m/12m drift detection."
        )

    asof_iso = _resolve_latest_hei_asof()
    if asof_iso is None:
        status = get_hei_status()
        if status.get("errors"):
            st.error(f"HEI build errors:\n{status['errors'][-1][:400]}")
        elif status.get("started_at") and not status.get("completed_at"):
            st.info("HEI is being built in background. Refresh in a few minutes.")
        else:
            st.warning("No HEI cache available. Run "
                          "`python -m lib.historical_event_impact` to build.")
        if base_product != "SRA":
            # Show the relevant ticker list even when no cache exists
            st.markdown(
                f"<div style='margin-top:0.7rem; font-size:0.8rem; "
                f"color:var(--text-muted);'>"
                f"<b>Relevant indicators for {base_product} ({cb_name}):</b><br>"
                + "<br>".join(
                    f"&nbsp;· <code style='color:var(--accent-bright);'>{t}</code> — "
                    f"{market_labels.get(t, t)}"
                    for t in market_tickers
                )
                + "</div>",
                unsafe_allow_html=True,
            )
        return

    catalog = load_hei_catalog(asof_iso)
    aggregates = load_hei_aggregates(asof_iso)
    drift = load_hei_drift(asof_iso)
    manifest = load_hei_manifest(asof_iso)

    if catalog.empty or aggregates.empty:
        st.warning("HEI parquets present but empty. May still be building.")
        return

    # =========================================================================
    # HEADER STRIP (selectors)
    # =========================================================================

    # Filter catalog to market-relevant tickers. For non-SRA markets the HEI
    # cache might not contain these tickers — fall back to whatever's in the
    # catalog with a clear note.
    catalog_tickers = sorted(catalog["ticker"].unique())
    intersection = [t for t in market_tickers if t in catalog_tickers]
    if base_product != "SRA" and not intersection:
        st.info(
            f"ℹ️ The HEI cache currently holds US economic releases. "
            f"{base_product}-relevant releases ({cb_name}: "
            f"{', '.join(market_tickers[:4])}...) are not yet in the cache. "
            f"Showing global risk-on/risk-off effect of US releases on "
            f"{base_product} instruments — these still move because of "
            f"cross-currency funding & USD-driven risk-asset flow."
        )
        avail_tickers = catalog_tickers
        active_labels = TICKER_LABELS
    elif base_product != "SRA" and intersection:
        avail_tickers = intersection
        active_labels = market_labels
    else:
        avail_tickers = catalog_tickers
        active_labels = TICKER_LABELS

    ticker_count = catalog.groupby("ticker").size().to_dict()

    h1, h2, h3, h4 = st.columns([3, 2, 2, 2])
    ticker_options = [
        f"{active_labels.get(t, t)} — {t} ({ticker_count.get(t, 0)} events)"
        for t in avail_tickers
    ]
    sel_idx = h1.selectbox("Event ticker",
                                range(len(ticker_options)),
                                format_func=lambda i: ticker_options[i],
                                key="hei_ticker_sel")
    ticker = avail_tickers[sel_idx]

    window = h2.selectbox(
        "Window", ["full", "12m", "6m", "3m"], key="hei_window_sel",
    )
    measurement = h3.selectbox(
        "Measurement", list(MEASUREMENTS), index=3,   # default release_impulse
        key="hei_measurement_sel",
    )
    instr_filter = h4.multiselect(
        "Instrument types",
        ["outright", "spread", "fly", "pack", "pack_fly"],
        default=["outright", "spread", "fly", "pack", "pack_fly"],
        key="hei_instr_filter_sel",
    )

    # Filter catalog to selected ticker
    cat_t = catalog[catalog["ticker"] == ticker]
    if cat_t.empty:
        st.warning(f"No events for {ticker}.")
        return

    # Filter aggregates to selected (ticker, window) — used by Sections 3/4/7
    agg_t_window = aggregates[
        (aggregates["ticker"] == ticker)
        & (aggregates["window"] == window)
    ]

    # =========================================================================
    # SECTION 0: Detailed narrative interpretation (placed first per user pref)
    # Auto-generated synthesis of every analysis below — useful as a TL;DR
    # and as a written report for trading decisions.
    # =========================================================================

    st.markdown("##### Section 0 — Detailed narrative interpretation")
    st.caption(
        "Auto-generated, data-driven synthesis of every analysis section below. "
        "All numbers reflect the current ticker / measurement / window selectors."
    )
    ticker_panel_for_narrative = load_hei_panel(asof_iso)
    if isinstance(ticker_panel_for_narrative, pd.DataFrame) and not ticker_panel_for_narrative.empty:
        ticker_panel_for_narrative = ticker_panel_for_narrative[
            ticker_panel_for_narrative["ticker"] == ticker
        ]
    try:
        narrative_md = build_event_narrative(
            ticker=ticker,
            ticker_label=active_labels.get(ticker, TICKER_LABELS.get(ticker, ticker)),
            cat_t=cat_t,
            aggregates_full=aggregates,
            drift_full=drift,
            panel_t=ticker_panel_for_narrative,
            measurement=measurement,
            window=window,
            states_8=STATES_8,
            pack_ranges=PACK_RANGES,
        )
        with st.expander("Read full narrative", expanded=True):
            st.markdown(narrative_md)
    except Exception as e:
        st.error(f"Narrative builder failed: {e}")

    # =========================================================================
    # SECTION 1: Event characteristics snapshot
    # =========================================================================

    st.markdown("##### Section 1 — Event characteristics")
    s1_cols = st.columns(5)
    s1_cols[0].metric("Total events", len(cat_t),
                          help="Post-2024-07-15 events with consensus + actual")
    accuracy = float((cat_t["surprise_z"].abs() < 0.5).mean()) if not cat_t.empty else 0
    s1_cols[1].metric("Consensus accuracy",
                          f"{accuracy:.0%}",
                          help="% of events with |surprise_z| < 0.5")
    avg_abs_surprise = float(cat_t["surprise_z"].abs().mean())
    s1_cols[2].metric("Avg |surprise z|",
                          f"{avg_abs_surprise:.2f}")
    avg_abs_expected = float(cat_t["expected_z"].abs().mean())
    s1_cols[3].metric("Avg |expected z|",
                          f"{avg_abs_expected:.2f}")
    last_date = cat_t["release_date"].max()
    s1_cols[4].metric("Last release", str(last_date))

    # State distribution
    state_counts = cat_t["state_8"].value_counts().reindex(STATES_8, fill_value=0)
    state_counts = pd.concat([state_counts,
                                    pd.Series({"FLAT": int((cat_t["state_8"] == "FLAT").sum())})])
    import plotly.graph_objects as go
    fig_sd = go.Figure(data=go.Bar(
        x=list(state_counts.index), y=state_counts.values,
        marker_color=[STATE_COLORS.get(s, "#94a3b8") for s in state_counts.index],
    ))
    fig_sd.update_layout(
        height=200,
        margin=dict(l=40, r=20, t=10, b=80),
        xaxis=dict(title=None, tickangle=-45),
        yaxis=dict(title="count"),
        title="Historical state distribution for this ticker",
        title_font_size=11,
    )
    st.plotly_chart(fig_sd, use_container_width=True, theme=None)

    # =========================================================================
    # SECTION 2: Pre-release behavior — 4-quadrant by Expected dir × |z| bucket
    # =========================================================================

    st.markdown("##### Section 2 — Pre-release behavior (anticipation)")
    st.caption(
        "How each instrument moves leading up to the release, "
        "split into 4 quadrants by (expected direction × |expected z|). "
        "Top-10 instruments per quadrant by absolute median normalized move."
    )

    # Per-event pre-release split by quadrant
    pre_measurement = st.selectbox(
        "Pre-release measurement",
        ["anticipation_full", "anticipation_overnight", "anticipation_morning"],
        index=0,
        key="hei_pre_measurement_sel",
        help=(
            "anticipation_full = T-1 open to T pre · "
            "overnight = T-1 close to T open · "
            "morning = T open to T pre"
        ),
    )

    # Restrict panel to this ticker + instrument types
    panel_t = load_hei_panel(asof_iso)
    panel_t = panel_t[(panel_t["ticker"] == ticker)
                            & (panel_t["instrument_type"].isin(instr_filter))]
    pre_col = f"{pre_measurement}_norm"

    if panel_t.empty or pre_col not in panel_t.columns:
        st.caption("No pre-release data available.")
    else:
        # Join with catalog to get expected_direction + expected_z
        cat_min = cat_t[["release_date", "expected_direction", "expected_z"]].copy()
        cat_min["release_date"] = pd.to_datetime(cat_min["release_date"]).dt.date
        panel_t_joined = panel_t.merge(cat_min, on="release_date", how="left")

        # Bucket: small / large based on |expected_z|
        panel_t_joined["expected_z_bucket"] = panel_t_joined["expected_z"].abs().apply(
            lambda z: "large" if z >= 1.0 else "small")
        # Drop FLAT direction events
        panel_t_joined = panel_t_joined[
            panel_t_joined["expected_direction"].isin(["UP", "DOWN"])].copy()

        if panel_t_joined.empty:
            st.caption("No directional events (all expected_change=0).")
        else:
            # Window filter (apply same window logic as aggregates)
            window_cutoff = None
            if window != "full":
                n_months = int(window.replace("m", ""))
                from datetime import timedelta
                asof = date.fromisoformat(asof_iso)
                window_cutoff = asof - timedelta(days=n_months * 30)
            if window_cutoff is not None:
                panel_t_joined = panel_t_joined[
                    panel_t_joined["release_date"] >= window_cutoff]

            # 4 quadrants: (UP/DOWN) × (small/large)
            q_grid = st.columns(2)
            quadrants = [
                ("UP × small |z|", "UP", "small", q_grid[0]),
                ("UP × large |z|", "UP", "large", q_grid[1]),
                ("DOWN × small |z|", "DOWN", "small", q_grid[0]),
                ("DOWN × large |z|", "DOWN", "large", q_grid[1]),
            ]
            for q_title, q_dir, q_size, q_col in quadrants:
                with q_col:
                    sub = panel_t_joined[
                        (panel_t_joined["expected_direction"] == q_dir)
                        & (panel_t_joined["expected_z_bucket"] == q_size)
                    ]
                    if sub.empty:
                        st.markdown(f"**{q_title}** — no events")
                        continue
                    n_events = sub["release_date"].nunique()
                    # Median per instrument
                    per_instr = sub.groupby(
                        ["instrument_id", "instrument_type"]
                    )[pre_col].agg(["median", "count"]).reset_index()
                    per_instr.columns = ["instrument_id", "instrument_type",
                                              "median", "n"]
                    per_instr = per_instr.dropna(subset=["median"])
                    if per_instr.empty:
                        st.markdown(f"**{q_title}** — n_events={n_events}, no data")
                        continue
                    per_instr["abs_median"] = per_instr["median"].abs()
                    top10 = per_instr.nlargest(10, "abs_median")
                    fig_q = go.Figure(data=go.Bar(
                        x=top10["median"].round(3),
                        y=top10["instrument_id"],
                        orientation="h",
                        marker_color=[
                            INSTRUMENT_TYPE_COLORS.get(t, "#94a3b8")
                            for t in top10["instrument_type"]
                        ],
                        hovertemplate=(
                            "<b>%{y}</b><br>"
                            "median: %{x:+.3f} σ_ATR<br>"
                            "n: %{customdata}<extra></extra>"
                        ),
                        customdata=top10["n"],
                    ))
                    fig_q.add_vline(x=0, line_dash="dot",
                                          line_color="#94a3b8")
                    fig_q.update_layout(
                        title=f"{q_title} (n_events={n_events})",
                        title_font_size=11,
                        height=260,
                        xaxis=dict(title="σ_ATR"),
                        yaxis=dict(autorange="reversed"),
                        margin=dict(l=80, r=15, t=35, b=30),
                    )
                    st.plotly_chart(fig_q, use_container_width=True,
                                          theme=None)

    # =========================================================================
    # SECTION 3: 8-state post-release reaction matrix (the centerpiece)
    # =========================================================================

    st.markdown(f"##### Section 3 — 8-state reaction matrix · `{measurement}` · {window} window")
    st.caption(
        "Cell color: median normalized move (ATR units). "
        "Red = price down (rates up). Green = price up (rates down). "
        f"Cells with n_obs < 3 are sample-thin and shown faded."
    )
    matrix_agg = agg_t_window[
        (agg_t_window["measurement"] == measurement)
        & (agg_t_window["instrument_type"].isin(instr_filter))
        & (agg_t_window["state_8"].isin(STATES_8))
    ]
    if matrix_agg.empty:
        st.caption("No data for this combination.")
    else:
        # Pivot: rows = instrument, cols = state
        matrix = matrix_agg.pivot_table(
            index=["instrument_id", "instrument_type"],
            columns="state_8", values="median_norm", aggfunc="first",
        ).reindex(columns=list(STATES_8))
        # n_obs pivot for low_sample masking
        nobs = matrix_agg.pivot_table(
            index=["instrument_id", "instrument_type"],
            columns="state_8", values="n_obs", aggfunc="first",
        ).reindex(columns=list(STATES_8)).fillna(0).astype(int)
        # Sort instruments by type then id
        type_order_map = {"outright": 0, "spread": 1, "fly": 2,
                              "pack": 3, "pack_fly": 4}
        matrix = matrix.reset_index()
        matrix["sort_key"] = matrix["instrument_type"].map(type_order_map)
        matrix = matrix.sort_values(["sort_key", "instrument_id"])
        instrument_labels = matrix["instrument_id"].tolist()
        matrix_data = matrix[list(STATES_8)].to_numpy(dtype=float)
        nobs = nobs.reset_index()
        nobs["sort_key"] = nobs["instrument_type"].map(type_order_map)
        nobs = nobs.sort_values(["sort_key", "instrument_id"])
        nobs_data = nobs[list(STATES_8)].to_numpy(dtype=int)

        # Build hover text
        hover_text = []
        for i, instr in enumerate(instrument_labels):
            row_text = []
            for j, st_name in enumerate(STATES_8):
                v = matrix_data[i, j]
                n = nobs_data[i, j]
                bp_row = matrix_agg[
                    (matrix_agg["instrument_id"] == instr)
                    & (matrix_agg["state_8"] == st_name)
                ]
                bp = (float(bp_row["median_bp"].iloc[0])
                          if not bp_row.empty else float("nan"))
                pct_atr = (float(bp_row["pct_above_1ATR"].iloc[0])
                              if not bp_row.empty else float("nan"))
                if np.isnan(v):
                    row_text.append(
                        f"{instr} · {st_name}<br>NO DATA")
                else:
                    flag = " (LOW_SAMPLE)" if n < 3 else ""
                    row_text.append(
                        f"{instr} · {st_name}<br>"
                        f"median_norm = {v:+.2f}σ_ATR · {bp:+.2f} bp<br>"
                        f"n_obs = {n}{flag} · pct>1ATR = {pct_atr:.0%}"
                    )
            hover_text.append(row_text)

        # Mask low-sample cells (faded by adjusting opacity in colorscale logic)
        z_display = matrix_data.copy()
        z_display[nobs_data < 3] = np.nan   # leaves cell empty in heatmap

        fig_m = go.Figure(data=go.Heatmap(
            z=z_display,
            x=list(STATES_8),
            y=instrument_labels,
            colorscale="RdYlGn",
            zmid=0,
            hoverinfo="text",
            text=hover_text,
            colorbar=dict(title="σ ATR"),
        ))
        # Annotate low-sample cells with "·"
        annotations = []
        for i in range(matrix_data.shape[0]):
            for j in range(matrix_data.shape[1]):
                if nobs_data[i, j] < 3:
                    annotations.append(dict(
                        x=STATES_8[j], y=instrument_labels[i],
                        text="·", showarrow=False,
                        font=dict(color="#94a3b8", size=10),
                    ))
        fig_m.update_layout(
            height=max(420, 18 * len(instrument_labels)),
            xaxis=dict(side="top", tickangle=-45),
            yaxis=dict(autorange="reversed"),
            annotations=annotations,
            margin=dict(l=110, r=20, t=80, b=20),
        )
        st.plotly_chart(fig_m, use_container_width=True, theme=None)

    # =========================================================================
    # SECTION 4: Top-5 most reactive per state
    # =========================================================================

    st.markdown("##### Section 4 — Top-5 most-reactive instruments per state")
    s4_cols = st.columns(4)
    for idx, st_name in enumerate(STATES_8):
        with s4_cols[idx % 4]:
            st.markdown(
                f'<span style="color:{STATE_COLORS[st_name]}; font-weight:600;">'
                f'{st_name}</span>',
                unsafe_allow_html=True,
            )
            state_agg = agg_t_window[
                (agg_t_window["measurement"] == measurement)
                & (agg_t_window["instrument_type"].isin(instr_filter))
                & (agg_t_window["state_8"] == st_name)
                & (agg_t_window["n_obs"] >= 3)
            ]
            if state_agg.empty:
                st.caption("low_sample")
                continue
            state_agg = state_agg.copy()
            state_agg["abs_median"] = state_agg["median_norm"].abs()
            top5 = state_agg.nlargest(5, "abs_median")
            top5_view = top5[["instrument_id", "median_norm",
                                  "median_bp", "n_obs"]].rename(
                columns={"median_norm": "σATR",
                            "median_bp": "bp",
                            "n_obs": "n"})
            top5_view["σATR"] = top5_view["σATR"].round(2)
            top5_view["bp"] = top5_view["bp"].round(2)
            st.dataframe(top5_view, hide_index=True,
                              use_container_width=True, height=210)

    # =========================================================================
    # SECTION 5: 1H bar-by-bar trajectory (per instrument)
    # =========================================================================

    st.markdown("##### Section 5 — 1H bar-by-bar trajectory")
    panel = load_hei_panel(asof_iso)
    panel_t = panel[panel["ticker"] == ticker]
    instruments_available = sorted(panel_t["instrument_id"].unique())
    default_instr = "M3" if "M3" in instruments_available else (
        instruments_available[0] if instruments_available else None)
    if default_instr is None:
        st.caption("No instrument data.")
    else:
        sel_instr = st.selectbox(
            "Instrument",
            instruments_available,
            index=instruments_available.index(default_instr),
            key="hei_instr_trajectory_sel",
        )
        st.caption(
            "Median cumulative normalized move at each anchor bar, by 8-state. "
            "States with n_obs < 3 are hidden. T = release moment."
        )
        # For each state, compute the median measurement at each "bar offset"
        # Bar offsets (in same order as anchor labels)
        anchor_measurements_in_order = [
            ("T-1 open", "B_tm1_open", "anticipation_full",   "inverted"),
            ("T-1 close", "B_tm1_close", "anticipation_overnight", "via_t_open"),
            ("T open", "B_t_open", "anticipation_morning", "via_t_pre"),
            ("T pre",  "B_t_pre",  "release_impulse",   "anchor"),
            ("T+1h",  "B_t_post_1h", "release_impulse", "directly"),
            ("T+2h",  "B_t_post_2h", "release_immediate_2h", "directly"),
            ("T+4h",  "B_t_post_4h", "release_short_4h", "directly"),
            ("T close", "B_t_close", "release_day_end", "directly"),
            ("T+1 open", "B_tp1_open", "next_day_open_gap", "via_t_close"),
            ("T+1 close", "B_tp1_close", "next_day_full", "via_t_close"),
        ]

        # Simpler approach: for each event × state, compute cumulative
        # normalized move from B_tm1_open through each anchor. Anchor moves
        # already in panel as `<m>_norm`. Sum/cumulative is from the panel's
        # per-event raw values.

        instr_panel = panel_t[panel_t["instrument_id"] == sel_instr]
        if instr_panel.empty:
            st.caption(f"No data for {sel_instr}")
        else:
            # Trajectory framework: build a position at each anchor bar.
            # Reference point: B_tm1_open (== 0 normalized).
            # Each subsequent anchor's value = cum normalized move from B_tm1_open.
            anchor_keys = ["B_tm1_open", "B_tm1_close", "B_t_open", "B_t_pre",
                                "B_t_post_1h", "B_t_post_2h", "B_t_post_4h",
                                "B_t_close", "B_tp1_open", "B_tp1_close"]
            anchor_labels = ["T-1 open", "T-1 close", "T open", "T pre",
                                  "T+1h", "T+2h", "T+4h", "T close",
                                  "T+1 open", "T+1 close"]
            # Compose: use 'full_window' which is B_tp1_close − B_tm1_open
            # plus partial anchors expressed as deltas off B_tm1_open
            # — use measurements we have:
            #   anchor B_tm1_open → 0
            #   anchor B_tm1_close → +(anchor delta from open) [no direct measure;
            #                                                   = -anticipation_full from another angle]
            # Simplest: compute per-event raw price at each anchor, normalize by event's ATR.
            # We need the bars themselves for this. To avoid re-querying OHLC,
            # we approximate using available measurement bp values, reconstructing
            # the trajectory from individual segments.

            # Each event has measurement bp values. We rebuild anchor values
            # relative to B_tm1_open = 0:
            #   B_tm1_close = (B_tm1_close - B_tm1_open) — not directly measured.
            #     Use: B_tm1_close ≈ B_tm1_open + (B_t_open - B_tm1_close)^-1 — too convoluted.
            #
            # Simpler: just plot the median measurement values vs their END anchor:
            #   B_t_pre @ "anticipation_full_norm" (relative to B_tm1_open)
            #   B_t_open @ "anticipation_full_norm - anticipation_morning_norm" — too noisy.
            #
            # Cleanest: present the 8 simple anchor measurements as
            # "cumulative move from B_t_pre" (since release_* measurements all
            # anchor on B_t_pre) plus anticipation_full for the lead-up:
            traj_anchors = [
                ("T-1 open",   "neg_anticipation_full",  "lead_up"),
                ("T pre",      "zero",                     "anchor_release"),
                ("T+1h",       "release_impulse_norm",   "post"),
                ("T+2h",       "release_immediate_2h_norm","post"),
                ("T+4h",       "release_short_4h_norm",  "post"),
                ("T close",    "release_day_end_norm",   "post"),
                ("T+1 close",  "full_t_to_tp1_norm",     "post"),
            ]

            # For each state, get per-event series
            traj_rows = []
            for st_name in STATES_8:
                state_events = instr_panel[instr_panel["state_8"] == st_name]
                if len(state_events) < 3:
                    continue
                for label, col, _ in traj_anchors:
                    if col == "zero":
                        values = pd.Series([0.0] * len(state_events))
                    elif col == "neg_anticipation_full":
                        if "anticipation_full_norm" in state_events.columns:
                            values = -state_events["anticipation_full_norm"]
                        else:
                            continue
                    else:
                        if col not in state_events.columns:
                            continue
                        values = state_events[col]
                    values = values.dropna()
                    if values.empty:
                        continue
                    traj_rows.append({
                        "state": st_name,
                        "anchor": label,
                        "median": float(values.median()),
                        "iqr_lo": float(values.quantile(0.25)),
                        "iqr_hi": float(values.quantile(0.75)),
                        "n": int(len(values)),
                    })
            if traj_rows:
                traj_df = pd.DataFrame(traj_rows)
                fig_traj = go.Figure()
                anchor_order = [a[0] for a in traj_anchors]
                for st_name in STATES_8:
                    sub = traj_df[traj_df["state"] == st_name]
                    if sub.empty:
                        continue
                    sub = sub.set_index("anchor").reindex(anchor_order).reset_index()
                    fig_traj.add_trace(go.Scatter(
                        x=sub["anchor"], y=sub["median"],
                        mode="lines+markers",
                        line=dict(color=STATE_COLORS[st_name], width=2),
                        name=f"{st_name} (n={int(sub['n'].max())})",
                        hovertemplate=("anchor: %{x}<br>"
                                          "median: %{y:+.2f}σ_ATR<extra>" + st_name + "</extra>"),
                    ))
                # add_vline with annotation_text breaks on categorical x-axes
                # (plotly tries to average start+end which fails on strings).
                # Draw the line + label separately.
                fig_traj.add_shape(
                    type="line", xref="x", yref="paper",
                    x0="T pre", x1="T pre", y0=0, y1=1,
                    line=dict(color="#fb923c", dash="dash", width=1),
                )
                fig_traj.add_annotation(
                    x="T pre", y=1.0, xref="x", yref="paper",
                    text="release", showarrow=False,
                    font=dict(color="#fb923c", size=10),
                    yshift=8,
                )
                fig_traj.update_layout(
                    height=380,
                    xaxis=dict(title="anchor bar"),
                    yaxis=dict(title="σ_ATR relative to T-pre baseline (= 0)"),
                    margin=dict(l=55, r=20, t=20, b=40),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  xanchor="right", x=1.0),
                )
                st.plotly_chart(fig_traj, use_container_width=True, theme=None)
            else:
                st.caption("All states have <3 events — trajectory unavailable.")

    # =========================================================================
    # SECTION 6: Per-instrument event timeline scatter
    # =========================================================================

    st.markdown("##### Section 6 — Event timeline scatter")
    if default_instr is not None and not instr_panel.empty:
        m_col = f"{measurement}_norm"
        bp_col = f"{measurement}_bp"
        if m_col in instr_panel.columns:
            scat_df = instr_panel.dropna(subset=[m_col]).copy()
            if not scat_df.empty:
                fig_s = go.Figure()
                # Per-release surprise_z, dedup'd by release_date (a single
                # ticker may have duplicate release_date rows after revisions —
                # keep the most recent and drop dupes so reindex stays valid).
                surprise_lookup = (
                    cat_t[["release_date", "surprise_z"]]
                    .drop_duplicates(subset="release_date", keep="last")
                    .set_index("release_date")["surprise_z"]
                )
                for st_name in scat_df["state_8"].unique():
                    sub = scat_df[scat_df["state_8"] == st_name]
                    surp_z = surprise_lookup.reindex(sub["release_date"]).fillna(0).to_numpy()
                    marker_sizes = np.clip(8 + np.abs(surp_z) * 4, 8, 30)
                    fig_s.add_trace(go.Scatter(
                        x=sub["release_date"],
                        y=sub[m_col],
                        mode="markers",
                        marker=dict(
                            color=STATE_COLORS.get(st_name, "#94a3b8"),
                            size=marker_sizes,
                            line=dict(color="#0a0e14", width=0.5),
                        ),
                        name=st_name,
                        hovertemplate=(
                            "<b>%{x}</b> · " + st_name +
                            "<br>norm: %{y:+.2f}σ<br>"
                            "<extra>" + sel_instr + "</extra>"
                        ),
                    ))
                fig_s.add_hline(y=0, line_dash="dot", line_color="#94a3b8")
                fig_s.update_layout(
                    height=300,
                    xaxis=dict(title=None),
                    yaxis=dict(title=f"{measurement} (σ_ATR)"),
                    margin=dict(l=55, r=20, t=10, b=40),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                  xanchor="right", x=1.0),
                )
                st.plotly_chart(fig_s, use_container_width=True, theme=None)

    # =========================================================================
    # SECTION 7: Drift comparison — 4-window heatmap grid + drift table
    # =========================================================================

    st.markdown("##### Section 7 — Drift comparison (3m / 6m / 12m / full)")
    st.caption(
        "Each heatmap shows the 8-state reaction matrix for that window. "
        "Drift table on the right ranks cells with biggest 3m-vs-full change."
    )

    drift_t = drift[(drift["ticker"] == ticker)
                          & (drift["measurement"] == measurement)
                          & (drift["instrument_type"].isin(instr_filter))]

    # Build 4 mini-heatmaps using the aggregates table (one per window)
    grid_layout = st.columns([2, 2, 2, 2, 3])
    for idx, win in enumerate(WINDOWS):
        with grid_layout[idx]:
            agg_w = aggregates[
                (aggregates["ticker"] == ticker)
                & (aggregates["measurement"] == measurement)
                & (aggregates["window"] == win)
                & (aggregates["instrument_type"].isin(instr_filter))
                & (aggregates["state_8"].isin(STATES_8))
            ]
            if agg_w.empty:
                st.markdown(f"**{win}** — no data")
                continue
            mat = agg_w.pivot_table(
                index=["instrument_id", "instrument_type"],
                columns="state_8", values="median_norm", aggfunc="first",
            ).reindex(columns=list(STATES_8))
            nobs = agg_w.pivot_table(
                index=["instrument_id", "instrument_type"],
                columns="state_8", values="n_obs", aggfunc="first",
            ).reindex(columns=list(STATES_8)).fillna(0).astype(int)
            type_order_m = {"outright": 0, "spread": 1, "fly": 2,
                              "pack": 3, "pack_fly": 4}
            mat = mat.reset_index()
            mat["sort_key"] = mat["instrument_type"].map(type_order_m)
            mat = mat.sort_values(["sort_key", "instrument_id"])
            nobs = nobs.reset_index()
            nobs["sort_key"] = nobs["instrument_type"].map(type_order_m)
            nobs = nobs.sort_values(["sort_key", "instrument_id"])
            labels = mat["instrument_id"].tolist()
            z = mat[list(STATES_8)].to_numpy(dtype=float)
            n_arr = nobs[list(STATES_8)].to_numpy(dtype=int)
            z[n_arr < 3] = np.nan
            fig_g = go.Figure(data=go.Heatmap(
                z=z,
                x=[s.replace("_", "\n") for s in STATES_8],
                y=labels,
                colorscale="RdYlGn",
                zmid=0,
                showscale=(idx == 3),
                hoverinfo="text",
                text=[
                    [
                        (f"{labels[i]} · {STATES_8[j]}<br>"
                          f"median {z[i, j]:+.2f}σ · n={n_arr[i, j]}")
                        if not np.isnan(z[i, j])
                        else f"{labels[i]} · {STATES_8[j]}<br>LOW_SAMPLE n={n_arr[i, j]}"
                        for j in range(len(STATES_8))
                    ]
                    for i in range(len(labels))
                ],
            ))
            fig_g.update_layout(
                title=f"<b>{win}</b>",
                title_font_size=11,
                height=max(360, 14 * len(labels)),
                xaxis=dict(side="top", tickangle=-30, tickfont=dict(size=8)),
                yaxis=dict(autorange="reversed",
                              tickfont=dict(size=8)),
                margin=dict(l=70, r=10, t=40, b=10),
            )
            st.plotly_chart(fig_g, use_container_width=True, theme=None)

    # Right column: drift ranking
    with grid_layout[4]:
        st.markdown("**Top-15 drift cells**")
        if drift_t.empty:
            st.caption("No drift data.")
        else:
            drift_t_s = drift_t.copy()
            drift_t_s["abs_drift"] = drift_t_s["recency_index_3m_vs_full"].abs()
            drift_t_s = drift_t_s.sort_values("abs_drift", ascending=False)
            drift_view = drift_t_s.head(15)[[
                "instrument_id", "state_8", "median_3m", "median_full",
                "recency_index_3m_vs_full", "drift_tag",
                "n_obs_3m", "n_obs_full",
            ]].copy()
            for c in ["median_3m", "median_full", "recency_index_3m_vs_full"]:
                drift_view[c] = drift_view[c].astype(float).round(2)
            drift_view = drift_view.rename(columns={
                "median_3m": "m3m", "median_full": "mF",
                "recency_index_3m_vs_full": "Δ_idx",
                "drift_tag": "tag",
                "n_obs_3m": "n3", "n_obs_full": "nF",
            })
            st.dataframe(drift_view, hide_index=True, use_container_width=True,
                              height=520)
            tag_counts = drift_t_s["drift_tag"].value_counts().to_dict()
            tag_strs = [f"`{tag}`: {count}" for tag, count in tag_counts.items()]
            st.caption(" · ".join(tag_strs))

    # =========================================================================
    # SECTION 7.5: Curve-reactivity analysis
    #   (a) instrument-family × state-8 reactivity heatmap
    #   (b) outright tenor-segment × state reactivity heatmap (which part of curve)
    #   (c) family reactivity evolution across 3m / 6m / 12m / full
    # =========================================================================

    st.markdown(
        f"##### Section 7.5 — Curve reactivity by family · `{measurement}`"
    )
    st.caption(
        "Where on the curve and in which family the move is concentrated. "
        "Cells are mean |median σ_ATR| across instruments in that bucket "
        "(low_sample cells with n_obs<3 excluded). Higher = more reactive."
    )

    # Reuse the (ticker × window)-filtered slice already built above and
    # pre-filter to selected measurement / instrument filter.
    agg_meas = agg_t_window[
        (agg_t_window["measurement"] == measurement)
        & (agg_t_window["instrument_type"].isin(instr_filter))
        & (agg_t_window["n_obs"] >= 3)
    ].copy()

    if agg_meas.empty:
        st.caption("No sufficient-sample rows for this combination.")
    else:
        agg_meas["abs_norm"] = agg_meas["median_norm"].abs()

        # -- (a) Family × State reactivity ------------------------------------
        st.markdown(
            "**(a) Instrument family × state reactivity** — "
            "rows = family, cols = state, values = mean |σ_ATR|"
        )
        fam_matrix = agg_meas.pivot_table(
            index="instrument_type", columns="state_8",
            values="abs_norm", aggfunc="mean",
        ).reindex(columns=list(STATES_8))
        n_fam = agg_meas.pivot_table(
            index="instrument_type", columns="state_8",
            values="n_obs", aggfunc="sum",
        ).reindex(columns=list(STATES_8)).fillna(0).astype(int)
        fam_order = [t for t in ["outright", "spread", "fly", "pack", "pack_fly"]
                              if t in fam_matrix.index]
        fam_matrix = fam_matrix.reindex(fam_order)
        n_fam = n_fam.reindex(fam_order)

        # Build hover text for family heatmap
        fam_hover = []
        for fam in fam_order:
            row_text = []
            for st_name in STATES_8:
                v = fam_matrix.loc[fam, st_name]
                n = int(n_fam.loc[fam, st_name])
                row_text.append(
                    f"family: {fam}<br>state: {st_name}<br>"
                    f"mean |σ_ATR|: {v:.2f}<br>cumulative n: {n}"
                    if pd.notna(v) else
                    f"family: {fam}<br>state: {st_name}<br>(no data)"
                )
            fam_hover.append(row_text)

        fig_fam = go.Figure(data=go.Heatmap(
            z=fam_matrix.to_numpy(dtype=float),
            x=list(STATES_8), y=fam_order,
            colorscale="Viridis",
            colorbar=dict(title="|σ_ATR|"),
            hoverinfo="text", text=fam_hover,
            zmin=0,
        ))
        fig_fam.update_layout(
            height=240, margin=dict(l=80, r=15, t=20, b=80),
            xaxis=dict(title=None, tickangle=-45),
            yaxis=dict(title=None, autorange="reversed"),
        )
        st.plotly_chart(fig_fam, use_container_width=True, theme=None)

        # -- (b) Outright tenor-segment × state reactivity --------------------
        if "outright" in instr_filter:
            st.markdown(
                "**(b) Curve segment × state reactivity** — "
                "outrights split by canonical pack ranges "
                "(white M0-11 · red M12-23 · green M24-35 · blue M36-47 · gold M48-59)"
            )

            def _segment_of(instrument_id: str) -> Optional[str]:
                # outright IDs look like 'M3', 'M12', 'M24'...
                if not isinstance(instrument_id, str) or not instrument_id.startswith("M"):
                    return None
                try:
                    m = int(instrument_id[1:])
                except ValueError:
                    return None
                for name, rng in PACK_RANGES.items():
                    if m in rng:
                        return name
                return None

            outright_agg = agg_meas[agg_meas["instrument_type"] == "outright"].copy()
            outright_agg["segment"] = outright_agg["instrument_id"].map(_segment_of)
            outright_agg = outright_agg.dropna(subset=["segment"])

            if outright_agg.empty:
                st.caption("No outright rows after filtering.")
            else:
                seg_matrix = outright_agg.pivot_table(
                    index="segment", columns="state_8",
                    values="abs_norm", aggfunc="mean",
                ).reindex(columns=list(STATES_8))
                n_seg = outright_agg.pivot_table(
                    index="segment", columns="state_8",
                    values="n_obs", aggfunc="sum",
                ).reindex(columns=list(STATES_8)).fillna(0).astype(int)
                seg_order = [s for s in ["white", "red", "green", "blue", "gold"]
                                      if s in seg_matrix.index]
                seg_matrix = seg_matrix.reindex(seg_order)
                n_seg = n_seg.reindex(seg_order)

                seg_hover = []
                for seg in seg_order:
                    row_text = []
                    for st_name in STATES_8:
                        v = seg_matrix.loc[seg, st_name]
                        n = int(n_seg.loc[seg, st_name])
                        row_text.append(
                            f"segment: {seg}<br>state: {st_name}<br>"
                            f"mean |σ_ATR|: {v:.2f}<br>cumulative n: {n}"
                            if pd.notna(v) else
                            f"segment: {seg}<br>state: {st_name}<br>(no data)"
                        )
                    seg_hover.append(row_text)

                fig_seg = go.Figure(data=go.Heatmap(
                    z=seg_matrix.to_numpy(dtype=float),
                    x=list(STATES_8), y=seg_order,
                    colorscale="Plasma",
                    colorbar=dict(title="|σ_ATR|"),
                    hoverinfo="text", text=seg_hover,
                    zmin=0,
                ))
                fig_seg.update_layout(
                    height=240, margin=dict(l=80, r=15, t=20, b=80),
                    xaxis=dict(title=None, tickangle=-45),
                    yaxis=dict(title=None, autorange="reversed"),
                )
                st.plotly_chart(fig_seg, use_container_width=True, theme=None)

                # Rank table: which segment reacts most overall
                seg_overall = (
                    outright_agg.groupby("segment")["abs_norm"]
                    .agg(["mean", "max", "count"])
                    .reindex(seg_order)
                    .rename(columns={"mean": "mean_abs_σATR",
                                              "max":  "max_abs_σATR",
                                              "count": "n_cells"})
                    .round(2)
                    .reset_index()
                )
                st.markdown("Curve-segment overall ranking (this window):")
                st.dataframe(seg_overall, hide_index=True,
                                  use_container_width=True, height=210)

        # -- (c) Family reactivity evolution across windows -------------------
        st.markdown(
            "**(c) Evolution across 3m / 6m / 12m / full** — "
            "how each family's average reactivity has changed (current measurement)."
        )
        evo_src = aggregates[
            (aggregates["ticker"] == ticker)
            & (aggregates["measurement"] == measurement)
            & (aggregates["instrument_type"].isin(instr_filter))
            & (aggregates["n_obs"] >= 3)
        ].copy()
        if evo_src.empty:
            st.caption("No sufficient-sample rows across windows.")
        else:
            evo_src["abs_norm"] = evo_src["median_norm"].abs()
            evo = (
                evo_src.groupby(["instrument_type", "window"])["abs_norm"]
                .mean()
                .reset_index()
            )
            window_order = ["3m", "6m", "12m", "full"]
            evo["window"] = pd.Categorical(
                evo["window"], categories=window_order, ordered=True,
            )
            evo = evo.sort_values(["instrument_type", "window"])
            fam_colors = {
                "outright": "#22d3ee", "spread": "#a78bfa", "fly": "#f97316",
                "pack": "#84cc16", "pack_fly": "#ef4444",
            }
            fig_evo = go.Figure()
            for fam in fam_order:
                sub = evo[evo["instrument_type"] == fam]
                if sub.empty:
                    continue
                fig_evo.add_trace(go.Scatter(
                    x=sub["window"].astype(str), y=sub["abs_norm"],
                    mode="lines+markers",
                    line=dict(color=fam_colors.get(fam, "#94a3b8"), width=2),
                    name=fam,
                    hovertemplate=("window: %{x}<br>"
                                      "mean |σ_ATR|: %{y:.2f}"
                                      "<extra>" + fam + "</extra>"),
                ))
            fig_evo.update_layout(
                height=280, margin=dict(l=55, r=20, t=20, b=40),
                xaxis=dict(title="window (recent → all)"),
                yaxis=dict(title="mean |σ_ATR| across instruments"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                              xanchor="right", x=1.0),
            )
            st.plotly_chart(fig_evo, use_container_width=True, theme=None)

            # Per-family delta (3m vs full) — explicit "evolution" call-out
            delta_rows = []
            for fam in fam_order:
                sub = evo[evo["instrument_type"] == fam].set_index("window")
                if "3m" in sub.index and "full" in sub.index:
                    v3 = float(sub.loc["3m", "abs_norm"])
                    vf = float(sub.loc["full", "abs_norm"])
                    rel = (v3 - vf) / vf if vf > 0 else float("nan")
                    tag = ("RISING" if rel > 0.20
                              else "FALLING" if rel < -0.20 else "STABLE")
                    delta_rows.append({
                        "family": fam, "3m": round(v3, 2),
                        "full": round(vf, 2),
                        "Δ (3m − full)": round(v3 - vf, 2),
                        "rel_Δ %": round(rel * 100, 1),
                        "tag": tag,
                    })
            if delta_rows:
                delta_df = pd.DataFrame(delta_rows)
                st.markdown("Family reactivity Δ (3m vs full):")
                st.dataframe(delta_df, hide_index=True,
                                  use_container_width=True, height=210)

    # =========================================================================
    # SECTION 8: Export footer
    # =========================================================================

    st.markdown("##### Section 8 — Exports")
    foot = st.columns([2, 2, 2, 6])
    # Per-ticker aggregates CSV
    ticker_agg = aggregates[aggregates["ticker"] == ticker]
    if not ticker_agg.empty:
        csv_data = ticker_agg.to_csv(index=False)
        foot[0].download_button(
            "📋 Aggregates CSV",
            csv_data,
            file_name=f"hei_aggregates_{ticker}_{asof_iso}.csv",
            mime="text/csv", key="hei_export_agg",
        )
    ticker_panel = panel_t if 'panel_t' in dir() else load_hei_panel(asof_iso)
    if not ticker_panel.empty:
        csv_panel = ticker_panel.to_csv(index=False)
        foot[1].download_button(
            "📋 Per-event panel CSV",
            csv_panel,
            file_name=f"hei_panel_{ticker}_{asof_iso}.csv",
            mime="text/csv", key="hei_export_panel",
        )
    ticker_drift = drift[drift["ticker"] == ticker]
    if not ticker_drift.empty:
        csv_drift = ticker_drift.to_csv(index=False)
        foot[2].download_button(
            "📋 Drift CSV",
            csv_drift,
            file_name=f"hei_drift_{ticker}_{asof_iso}.csv",
            mime="text/csv", key="hei_export_drift",
        )

    with st.expander("Manifest"):
        st.json(manifest)
