"""STIR-specific setup detectors — C3, C4, C5, C8 (12/24/36 variants), C9a, C9b.

These detectors operate on SRA universe-level data:
  · C3 (intra-curve carry rank): needs ALL outright closes + their carry/day
  · C4 (fly z): needs the fly's own close history (single contract)
  · C5 (spread z): needs the spread's own close history (single contract)
  · C8 (terminal rate reprice): needs the full outright chain (12M / 24M / 36M projection)
  · C9 (slope steep/flat): needs front × back outright pair

Detectors take richer context than A/B detectors. The scan layer prepares
the data and routes per-contract calls to each detector.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from lib.carry import compute_per_contract_carry
from lib.contract_units import bp_multiplier, load_catalog
from lib.fomc import (
    get_fomc_dates_in_range,
    is_quarterly,
    next_fomc_date,
    reference_period,
    third_wednesday,
)
from lib.mean_reversion import zscore_value as mr_zscore_value
from lib.setups.base import (
    SetupResult,
    build_pair_legs,
    compute_R_levels,
    lots_at_10k_risk,
    round_to_tick,
    safe_float,
    state_from_conditions_met,
)
from lib.setups.registry import FAMILY_STIR


# =============================================================================
# C3 · INTRA_CURVE_CARRY_RANK  (operates over ALL outrights in market)
# =============================================================================
def detect_c3_for_contract(symbol: str, asof_date: date, bp_mult: float,
                            carry_per_contract: dict, ema20_per_contract: dict,
                            close_per_contract: dict,
                            top_decile_threshold: Optional[float],
                            bottom_decile_threshold: Optional[float]) -> SetupResult:
    """Detect C3 fire for a single contract.

    Caller must compute carry_per_contract and the decile thresholds across the
    entire universe of OUTRIGHTS first, then call this once per contract.
    """
    res = SetupResult(setup_id="C3", name="INTRA_CURVE_CARRY_RANK",
                       family=FAMILY_STIR, scope="outright")
    carry = safe_float(carry_per_contract.get(symbol))
    ema20 = safe_float(ema20_per_contract.get(symbol))
    close = safe_float(close_per_contract.get(symbol))
    if any(v is None for v in (carry, ema20, close)):
        res.state = "N/A"; res.error = "missing carry/EMA_20/close"; return res
    if top_decile_threshold is None or bottom_decile_threshold is None:
        res.state = "N/A"; res.error = "no universe deciles"; return res

    in_top = carry >= top_decile_threshold
    in_bot = carry <= bottom_decile_threshold
    above_ema = close > ema20
    below_ema = close < ema20

    if in_top and above_ema:
        res.state = "FIRED"; res.fired_long = True; res.direction = "LONG"
    elif in_bot and below_ema:
        res.state = "FIRED"; res.fired_short = True; res.direction = "SHORT"
    elif (in_top and not above_ema) or (in_bot and not below_ema):
        res.state = "APPROACHING"
        res.direction = "LONG" if in_top else "SHORT"
    elif (carry >= top_decile_threshold * 0.85) or (carry <= bottom_decile_threshold * 0.85):
        res.state = "NEAR"
        res.direction = "LONG" if carry > 0 else "SHORT"
    else:
        res.state = "FAR"

    res.confidence = 1.0 if res.state == "FIRED" else (0.5 if res.state == "NEAR" else 0.0)
    res.key_inputs = {"close": close, "carry/day (bp)": carry,
                       "EMA_20": ema20, "in_top_decile": in_top, "in_bottom_decile": in_bot,
                       "top_decile_threshold (bp)": top_decile_threshold,
                       "bottom_decile_threshold (bp)": bottom_decile_threshold}
    res.thresholds = {"carry rank":          "top decile (LONG) / bottom decile (SHORT)",
                       "close vs EMA_20":     "> for LONG / < for SHORT"}
    # No specific stop/target — sized by 2×ATR convention on caller's side if desired
    res.interpretation = (
        f"FIRED {res.direction} — top-carry contract aligned with trend"
        if res.state == "FIRED" else
        f"APPROACHING — carry rank ✓ but close on wrong side of EMA_20"
        if res.state == "APPROACHING" else
        f"NEAR — carry near decile threshold" if res.state == "NEAR" else
        "no signal"
    )
    return res


def compute_c3_universe(close_today: dict, panel_per_contract: dict,
                          contracts: list, asof_date: date) -> dict:
    """Compute carry/day per outright contract and return universe-level
    carry stats + per-contract decile thresholds.

    Returns dict with keys ``carry``, ``ema20``, ``close``, ``top_decile``,
    ``bottom_decile``.
    """
    carry = compute_per_contract_carry(close_today, contracts)
    ema20 = {}
    for sym in contracts:
        p = panel_per_contract.get(sym)
        if p is None or p.empty:
            ema20[sym] = None; continue
        try:
            row = p.loc[p.index.normalize() == pd.Timestamp(asof_date).normalize()]
            ema20[sym] = safe_float(row.iloc[-1].get("EMA_20")) if not row.empty else None
        except Exception:
            ema20[sym] = None
    valid = [v for v in carry.values() if v is not None and np.isfinite(v)]
    if not valid:
        return {"carry": carry, "ema20": ema20, "close": close_today,
                "top_decile": None, "bottom_decile": None}
    arr = np.array(valid)
    return {"carry": carry, "ema20": ema20, "close": close_today,
             "top_decile": float(np.percentile(arr, 90)),
             "bottom_decile": float(np.percentile(arr, 10))}


# =============================================================================
# C4 · TRADITIONAL_FLY_MR  (per-fly z-score)
# =============================================================================
def detect_c4(panel: pd.DataFrame, asof_date: date,
              bp_mult: float, scope: str = "fly") -> SetupResult:
    res = SetupResult(setup_id="C4", name="TRADITIONAL_FLY_MR",
                       family=FAMILY_STIR, scope=scope)
    if panel is None or panel.empty:
        res.state = "N/A"; res.error = "no panel"; return res
    close_series = panel["close"] if "close" in panel.columns else None
    if close_series is None or close_series.empty:
        res.state = "N/A"; res.error = "no close"; return res
    z = mr_zscore_value(close_series, asof_date, lookback=60)
    if z is None:
        res.state = "N/A"; res.error = "z(60d) unavailable"; return res

    # Standard deviations for stops
    ts = pd.Timestamp(asof_date)
    hist60 = close_series.loc[close_series.index < ts].tail(60).dropna()
    if hist60.empty:
        res.state = "N/A"; res.error = "no 60d history"; return res
    mean60 = float(hist60.mean()); std60 = float(hist60.std(ddof=1))
    try:
        close = float(close_series.loc[close_series.index.normalize() == ts.normalize()].iloc[-1])
    except IndexError:
        res.state = "N/A"; res.error = "no close at asof"; return res

    az = abs(z)
    if az >= 2:
        res.state = "FIRED"
        if z > 0:
            res.fired_short = True; res.direction = "SHORT"
        else:
            res.fired_long = True; res.direction = "LONG"
    elif az >= 1.5:
        res.state = "NEAR"
        res.direction = "SHORT" if z > 0 else "LONG"
    elif az >= 1.0:
        res.state = "APPROACHING"
        res.direction = "SHORT" if z > 0 else "LONG"
    else:
        res.state = "FAR"

    res.confidence = min(1.0, az / 2.0)
    res.key_inputs = {"close": close, "z(60d)": z, "mean_60": mean60, "std_60": std60}
    res.thresholds = {"|z(60d)|": "≥ 2 → FIRED · ≥ 1.5 → NEAR · ≥ 1 → APPROACHING"}
    if res.fired_long and std60 > 0:
        stop = mean60 - 3 * std60
        if close > stop:
            res.entry = round_to_tick(close, bp_mult); res.stop = round_to_tick(stop, bp_mult)
            res.t1 = round_to_tick(mean60, bp_mult); res.t2 = None
            sd_bp = (close - stop) * bp_mult
            res.lots_at_10k_risk = lots_at_10k_risk(sd_bp)
    elif res.fired_short and std60 > 0:
        stop = mean60 + 3 * std60
        if close < stop:
            res.entry = round_to_tick(close, bp_mult); res.stop = round_to_tick(stop, bp_mult)
            res.t1 = round_to_tick(mean60, bp_mult); res.t2 = None
            sd_bp = (stop - close) * bp_mult
            res.lots_at_10k_risk = lots_at_10k_risk(sd_bp)

    if res.state in ("NEAR", "APPROACHING"):
        res.distance_to_fire = float(max(0, 2.0 - az))
        res.eta_bars = res.distance_to_fire * 3.0

    res.interpretation = (
        f"FIRED {res.direction} — fly z extreme ({z:+.2f}σ)" if res.state == "FIRED" else
        f"NEAR (z={z:+.2f}σ)" if res.state == "NEAR" else
        f"APPROACHING (z={z:+.2f}σ)" if res.state == "APPROACHING" else
        "no signal"
    )
    return res


# =============================================================================
# C5 · CALENDAR_SPREAD_MR  (per-spread z-score)
# =============================================================================
def detect_c5(panel: pd.DataFrame, asof_date: date,
              bp_mult: float, scope: str = "spread") -> SetupResult:
    res = SetupResult(setup_id="C5", name="CALENDAR_SPREAD_MR",
                       family=FAMILY_STIR, scope=scope)
    if panel is None or panel.empty:
        res.state = "N/A"; res.error = "no panel"; return res
    close_series = panel["close"] if "close" in panel.columns else None
    if close_series is None or close_series.empty:
        res.state = "N/A"; res.error = "no close"; return res
    z = mr_zscore_value(close_series, asof_date, lookback=60)
    if z is None:
        res.state = "N/A"; res.error = "z(60d) unavailable"; return res

    ts = pd.Timestamp(asof_date)
    hist60 = close_series.loc[close_series.index < ts].tail(60).dropna()
    if hist60.empty:
        res.state = "N/A"; res.error = "no 60d history"; return res
    mean60 = float(hist60.mean()); std60 = float(hist60.std(ddof=1))
    try:
        close = float(close_series.loc[close_series.index.normalize() == ts.normalize()].iloc[-1])
    except IndexError:
        res.state = "N/A"; res.error = "no close at asof"; return res

    az = abs(z)
    if az >= 2:
        res.state = "FIRED"
        if z > 0:
            res.fired_short = True; res.direction = "SHORT"
        else:
            res.fired_long = True; res.direction = "LONG"
    elif az >= 1.5:
        res.state = "NEAR"
        res.direction = "SHORT" if z > 0 else "LONG"
    elif az >= 1.0:
        res.state = "APPROACHING"
        res.direction = "SHORT" if z > 0 else "LONG"
    else:
        res.state = "FAR"

    res.confidence = min(1.0, az / 2.0)
    res.key_inputs = {"close": close, "z(60d)": z, "mean_60": mean60, "std_60": std60}
    res.thresholds = {"|z(60d)|": "≥ 2 → FIRED · ≥ 1.5 → NEAR · ≥ 1 → APPROACHING"}
    if res.fired_long and std60 > 0:
        stop = mean60 - 3 * std60
        if close > stop:
            res.entry = round_to_tick(close, bp_mult); res.stop = round_to_tick(stop, bp_mult)
            res.t1 = round_to_tick(mean60, bp_mult); res.t2 = None
            sd_bp = (close - stop) * bp_mult
            res.lots_at_10k_risk = lots_at_10k_risk(sd_bp)
    elif res.fired_short and std60 > 0:
        stop = mean60 + 3 * std60
        if close < stop:
            res.entry = round_to_tick(close, bp_mult); res.stop = round_to_tick(stop, bp_mult)
            res.t1 = round_to_tick(mean60, bp_mult); res.t2 = None
            sd_bp = (stop - close) * bp_mult
            res.lots_at_10k_risk = lots_at_10k_risk(sd_bp)

    if res.state in ("NEAR", "APPROACHING"):
        res.distance_to_fire = float(max(0, 2.0 - az))
        res.eta_bars = res.distance_to_fire * 3.0

    res.interpretation = (
        f"FIRED {res.direction} — spread z extreme ({z:+.2f}σ)" if res.state == "FIRED" else
        f"NEAR (z={z:+.2f}σ)" if res.state == "NEAR" else
        f"APPROACHING (z={z:+.2f}σ)" if res.state == "APPROACHING" else
        "no signal"
    )
    return res


# =============================================================================
# C8 · TERMINAL_RATE_REPRICE  (12M / 24M / 36M variants)
# =============================================================================
def compute_terminal_rate(close_today_per_contract: dict, contract_metadata: pd.DataFrame,
                           asof_date: date, horizon_months: int) -> dict:
    """Compute terminal rate as per gameplan §6.5 — for a given horizon (12/24/36).

    contract_metadata: DataFrame with columns ``symbol``, ``expiry_year``, ``expiry_month``.
    Returns dict: ``{terminal_max, terminal_min, front_implied, cycle, terminal,
                     contract_at_terminal, days_to_terminal}``.
    """
    out = {"terminal_max": None, "terminal_min": None, "front_implied": None,
            "cycle": None, "terminal": None, "contract_at_terminal": None,
            "days_to_terminal": None, "horizon_months": horizon_months}
    if contract_metadata is None or contract_metadata.empty:
        return out
    horizon_end = asof_date + timedelta(days=horizon_months * 30)
    eligible = []
    for _, r in contract_metadata.iterrows():
        sym = r["symbol"]
        try:
            exp_y = int(r["expiry_year"]); exp_m = int(r["expiry_month"])
        except (TypeError, ValueError):
            continue
        try:
            exp_dt = third_wednesday(exp_y, exp_m)
        except Exception:
            continue
        if exp_dt < asof_date or exp_dt > horizon_end:
            continue
        close = safe_float(close_today_per_contract.get(sym))
        if close is None:
            continue
        # IMM convention: implied = 100 - close (SRA is IMM)
        implied = 100.0 - close
        eligible.append({"symbol": sym, "implied": implied, "expiry": exp_dt})
    if not eligible:
        return out
    eligible.sort(key=lambda x: x["expiry"])
    front = eligible[0]
    out["front_implied"] = float(front["implied"])
    implieds = [e["implied"] for e in eligible]
    max_imp = max(implieds); min_imp = min(implieds)
    out["terminal_max"] = float(max_imp)
    out["terminal_min"] = float(min_imp)
    diff_max = max_imp - out["front_implied"]
    diff_min = out["front_implied"] - min_imp
    if diff_max >= 0.25 and diff_min < 0.10:
        out["cycle"] = "HIKING"; out["terminal"] = max_imp
        target = max(eligible, key=lambda e: e["implied"])
    elif diff_min >= 0.25 and diff_max < 0.10:
        out["cycle"] = "CUTTING"; out["terminal"] = min_imp
        target = min(eligible, key=lambda e: e["implied"])
    elif abs(diff_max) < 0.10 and abs(diff_min) < 0.10:
        out["cycle"] = "ON_HOLD"; out["terminal"] = float(out["front_implied"])
        target = front
    else:
        out["cycle"] = "MIXED"
        if diff_max >= diff_min:
            out["terminal"] = max_imp
            target = max(eligible, key=lambda e: e["implied"])
        else:
            out["terminal"] = min_imp
            target = min(eligible, key=lambda e: e["implied"])
    out["contract_at_terminal"] = target["symbol"]
    out["days_to_terminal"] = (target["expiry"] - asof_date).days
    return out


def detect_c8_variant(close_history_per_contract: dict, contract_metadata: pd.DataFrame,
                       asof_date: date, horizon_months: int,
                       lookback_days: int = 5,
                       fire_bp_threshold: float = 25.0) -> SetupResult:
    """Detect C8 fire at a given horizon (12/24/36 months).

    close_history_per_contract: {sym: pd.Series of closes indexed by date}
    Returns one SetupResult — the fire is universe-level (not per-contract) but
    we tag the ``contract_at_terminal`` as the principal contract.
    """
    setup_id = f"C8_{horizon_months}M"
    name = f"TERMINAL_RATE_REPRICE · {horizon_months}M" + (
        " (PRIMARY)" if horizon_months == 24 else ""
    )
    res = SetupResult(setup_id=setup_id, name=name, family=FAMILY_STIR, scope="outright")
    # Today and lookback_days ago closes
    close_today = {sym: safe_float(s.loc[s.index.normalize() == pd.Timestamp(asof_date).normalize()].iloc[-1])
                    if (s is not None and not s.empty
                        and (s.index.normalize() == pd.Timestamp(asof_date).normalize()).any())
                    else None
                    for sym, s in close_history_per_contract.items()}
    lookback_dt = pd.Timestamp(asof_date) - pd.Timedelta(days=int(lookback_days * 1.5))
    close_then = {}
    for sym, s in close_history_per_contract.items():
        if s is None or s.empty:
            close_then[sym] = None; continue
        prior = s.loc[s.index <= lookback_dt]
        close_then[sym] = safe_float(prior.iloc[-1] if not prior.empty else None)

    today_terminal = compute_terminal_rate(close_today, contract_metadata, asof_date, horizon_months)
    then_terminal = compute_terminal_rate(close_then, contract_metadata,
                                            (pd.Timestamp(asof_date) - pd.Timedelta(days=lookback_days)).date(),
                                            horizon_months)

    if today_terminal["terminal"] is None or then_terminal["terminal"] is None:
        res.state = "N/A"; res.error = "terminal not computable"; return res

    delta = (today_terminal["terminal"] - then_terminal["terminal"]) * 100.0  # to bp
    abs_delta = abs(delta)

    if abs_delta >= fire_bp_threshold:
        res.state = "FIRED"
        if delta > 0:
            res.fired_long = True   # implied rate up = price down — for outrights this is SHORT
            # But "fire LONG/SHORT" here represents direction of the rate move, not a trade
            # Re-tag: rate up → bearish for outrights (price down)
            res.fired_long = False
            res.fired_short = True; res.direction = "SHORT"
        else:
            res.fired_long = True; res.direction = "LONG"
    elif abs_delta >= 15.0:
        res.state = "NEAR"
        res.direction = "SHORT" if delta > 0 else "LONG"
    elif abs_delta >= 10.0:
        res.state = "APPROACHING"
        res.direction = "SHORT" if delta > 0 else "LONG"
    else:
        res.state = "FAR"

    res.confidence = min(1.0, abs_delta / fire_bp_threshold)
    res.key_inputs = {
        "terminal_today": today_terminal["terminal"],
        "terminal_then":  then_terminal["terminal"],
        "Δ over 5d (bp)":  delta,
        "cycle":           today_terminal["cycle"],
        "terminal contract":  today_terminal["contract_at_terminal"],
        "days_to_terminal":  today_terminal["days_to_terminal"],
    }
    res.thresholds = {
        "|Δ over 5d (bp)|": (
            f"≥ {fire_bp_threshold} → FIRED · ≥ 15 → NEAR · ≥ 10 → APPROACHING"
        ),
        "horizon": f"{horizon_months} months",
    }
    if res.state in ("NEAR", "APPROACHING"):
        res.distance_to_fire = float(max(0, fire_bp_threshold - abs_delta))
        res.eta_bars = res.distance_to_fire / 5.0   # 5bp/day implied move
    res.notes = today_terminal["contract_at_terminal"] or ""
    res.interpretation = (
        f"FIRED — terminal_{horizon_months}M moved {delta:+.1f}bp / 5d ({today_terminal['cycle']})"
        if res.state == "FIRED" else
        f"NEAR ({delta:+.1f}bp / 5d)" if res.state == "NEAR" else
        f"APPROACHING ({delta:+.1f}bp / 5d)" if res.state == "APPROACHING" else
        f"quiet ({delta:+.1f}bp / 5d)"
    )
    return res


# =============================================================================
# C9a / C9b · CURVE_STEEPENER / FLATTENER
# =============================================================================
def _slope_history(close_history_per_contract: dict, front_sym: str, back_sym: str,
                    asof_date: date, lookback_days: int = 30) -> Optional[pd.Series]:
    """Compute slope = (100 - close_back) - (100 - close_front) = close_front - close_back
    in IMM convention. (Higher implied rate at back → positive slope = steepener.)

    Returns slope series in PRICE units (which equals -ΔRate).
    Caller multiplies by 100 to get bp.
    """
    if front_sym not in close_history_per_contract or back_sym not in close_history_per_contract:
        return None
    front_s = close_history_per_contract[front_sym]
    back_s = close_history_per_contract[back_sym]
    if front_s is None or back_s is None or front_s.empty or back_s.empty:
        return None
    # Slope_bp = (back_implied - front_implied) × 100
    #          = ((100 - close_back) - (100 - close_front)) × 100
    #          = (close_front - close_back) × 100
    df = pd.concat({"f": front_s, "b": back_s}, axis=1).dropna()
    df = df.loc[df.index <= pd.Timestamp(asof_date)]
    if len(df) < 6:
        return None
    return ((df["f"] - df["b"]) * 100.0)   # bp


# Default stop distances (bp on slope) — used to compute lots_at_10k_risk
# and the entry/stop/T1/T2 levels for the slope itself.
_C9A_STOP_BP = 5.0    # cross trade — slope must travel 5bp adverse to invalidate
_C9B_STOP_BP = 5.0    # trend trade — same magnitude, looser horizon


def _pair_today_closes(close_history_per_contract: dict, front_sym: str,
                         back_sym: str, asof_date: date) -> tuple:
    """Return (front_close, back_close) at the latest <= asof_date bar of each
    leg's close series. Either may be None if the series is empty / missing."""
    def _last_close(sym):
        s = close_history_per_contract.get(sym)
        if s is None or s.empty:
            return None
        s = s.loc[s.index <= pd.Timestamp(asof_date)]
        return float(s.iloc[-1]) if not s.empty else None
    return (_last_close(front_sym), _last_close(back_sym))


def _c9_setup_id(base: str, offset_months: int) -> str:
    return f"{base}_{int(offset_months)}M"


def _c9_name(base_text: str, offset_months: int) -> str:
    return f"{base_text} · {int(offset_months)}M slope"


def _c9_legs_and_lots(direction: Optional[str], stop_bp: float,
                       front_sym: str, back_sym: str,
                       front_close: Optional[float] = None,
                       back_close: Optional[float] = None) -> tuple:
    """Compute (lots_per_side, legs) for a C9 pair given the chosen stop in bp.
    DV01-neutral pair: equal lots both legs, $25/bp DV01 each. Slope DV01 of
    the pair = $25 × N. So lots = floor(10000 / (stop_bp × 25)).

    If ``front_close`` / ``back_close`` are supplied (today's outright closes
    in price units), the emitted legs carry per-leg ``entry_price`` so the
    trader can place orders at known price levels on each individual leg.
    """
    lots = lots_at_10k_risk(stop_bp)
    legs = build_pair_legs(front_sym, back_sym, direction, lots,
                            front_close=front_close, back_close=back_close)
    return (lots, legs)


def detect_c9a(close_history_per_contract: dict,
                front_sym: str, back_sym: str, asof_date: date,
                offset_months: int = 12,
                bp_mult_front: float = 100.0,
                bp_mult_back: float = 100.0) -> SetupResult:
    """C9a · slope CROSSING for a (front, back) pair separated by ``offset_months``.

    Fires when slope = (front − back) crosses zero today (sign flip).
    Setup id is suffixed with the offset, e.g. ``C9a_12M``.
    """
    sid = _c9_setup_id("C9a", offset_months)
    res = SetupResult(
        setup_id=sid,
        name=_c9_name("CURVE_STEEPENER / FLATTENER · slope crossing", offset_months),
        family=FAMILY_STIR, scope="outright",
    )
    slope = _slope_history(close_history_per_contract, front_sym, back_sym, asof_date)
    if slope is None or len(slope) < 2:
        res.state = "N/A"; res.error = "slope unavailable"; return res
    today_s = float(slope.iloc[-1])
    prev_s = float(slope.iloc[-2])

    cross_up = prev_s <= 0 and today_s > 0
    cross_dn = prev_s >= 0 and today_s < 0
    if cross_up:
        res.state = "FIRED"; res.fired_long = True; res.direction = "LONG"
    elif cross_dn:
        res.state = "FIRED"; res.fired_short = True; res.direction = "SHORT"
    elif abs(today_s) < 1.0:
        res.state = "NEAR"; res.direction = "LONG" if today_s >= 0 else "SHORT"
    elif abs(today_s) < 3.0:
        res.state = "APPROACHING"; res.direction = "LONG" if today_s >= 0 else "SHORT"
    else:
        res.state = "FAR"

    res.confidence = 1.0 if res.state == "FIRED" else (0.5 if res.state == "NEAR" else 0.0)
    res.key_inputs = {
        "slope today (bp)": today_s, "slope prev (bp)": prev_s,
        "offset (months)":  offset_months,
        "front contract":   front_sym, "back contract": back_sym,
    }
    res.thresholds = {
        "FIRED": "slope crosses zero today (sign flip)",
        "NEAR":  "|slope| < 1 bp",
        "APPROACHING": "|slope| < 3 bp",
    }
    res.notes = f"front={front_sym} · back={back_sym} · offset={offset_months}M"

    # Trade levels — slope-relative, in bp. For LONG (steepener): entry=today_s,
    # stop=today_s - 5bp, T1=today_s + 5bp, T2=today_s + 10bp. Mirror for SHORT.
    # Slope is stored in bp; round to half-bp so display matches exchange granularity.
    if res.direction in ("LONG", "SHORT"):
        sign = 1.0 if res.direction == "LONG" else -1.0
        res.entry = round_to_tick(today_s,                                bp_mult=1.0)
        res.stop  = round_to_tick(today_s - sign * _C9A_STOP_BP,          bp_mult=1.0)
        res.t1    = round_to_tick(today_s + sign * _C9A_STOP_BP,          bp_mult=1.0)
        res.t2    = round_to_tick(today_s + sign * 2.0 * _C9A_STOP_BP,    bp_mult=1.0)
        front_c, back_c = _pair_today_closes(
            close_history_per_contract, front_sym, back_sym, asof_date)
        res.lots_at_10k_risk, res.legs = _c9_legs_and_lots(
            res.direction, _C9A_STOP_BP, front_sym, back_sym,
            front_close=front_c, back_close=back_c,
        )

    res.interpretation = (
        f"FIRED {res.direction} {offset_months}M slope — crossed zero today "
        f"({prev_s:+.2f} → {today_s:+.2f} bp)"
        if res.state == "FIRED" else
        f"NEAR cross (slope {today_s:+.2f} bp · {offset_months}M)"
        if res.state == "NEAR" else
        f"APPROACHING (slope {today_s:+.2f} bp · {offset_months}M)"
        if res.state == "APPROACHING" else
        f"slope = {today_s:+.2f} bp · {offset_months}M"
    )
    return res


def detect_c9b(close_history_per_contract: dict,
                front_sym: str, back_sym: str, asof_date: date,
                offset_months: int = 12) -> SetupResult:
    """C9b · slope 5d TREND for a (front, back) pair separated by ``offset_months``.

    Fires when slope's 5-day cumulative trend ≥ +5bp (steepening) or ≤ -5bp (flattening).
    Setup id is suffixed with the offset, e.g. ``C9b_12M``.
    """
    sid = _c9_setup_id("C9b", offset_months)
    res = SetupResult(
        setup_id=sid,
        name=_c9_name("CURVE_STEEPENER / FLATTENER · 5d trend", offset_months),
        family=FAMILY_STIR, scope="outright",
    )
    slope = _slope_history(close_history_per_contract, front_sym, back_sym, asof_date)
    if slope is None or len(slope) < 6:
        res.state = "N/A"; res.error = "slope unavailable"; return res
    delta_5d = float(slope.iloc[-1] - slope.iloc[-6])
    today_s = float(slope.iloc[-1])
    abs_d = abs(delta_5d)
    if abs_d >= 5.0:
        res.state = "FIRED"
        if delta_5d > 0:
            res.fired_long = True; res.direction = "LONG"   # steepener
        else:
            res.fired_short = True; res.direction = "SHORT" # flattener
    elif abs_d >= 3.0:
        res.state = "NEAR"; res.direction = "LONG" if delta_5d >= 0 else "SHORT"
    elif abs_d >= 1.5:
        res.state = "APPROACHING"; res.direction = "LONG" if delta_5d >= 0 else "SHORT"
    else:
        res.state = "FAR"

    res.confidence = min(1.0, abs_d / 5.0)
    res.key_inputs = {
        "slope today (bp)": today_s,
        "slope 5d ago (bp)": float(slope.iloc[-6]),
        "Δ over 5d (bp)":   delta_5d,
        "offset (months)":  offset_months,
        "front contract":   front_sym, "back contract": back_sym,
    }
    res.thresholds = {
        "|Δ over 5d|": "≥ 5 bp → FIRED · ≥ 3 → NEAR · ≥ 1.5 → APPROACHING",
    }
    if res.state in ("NEAR", "APPROACHING"):
        res.distance_to_fire = float(max(0, 5.0 - abs_d))
        res.eta_bars = res.distance_to_fire / 1.0
    res.notes = f"front={front_sym} · back={back_sym} · offset={offset_months}M"

    if res.direction in ("LONG", "SHORT"):
        sign = 1.0 if res.direction == "LONG" else -1.0
        res.entry = round_to_tick(today_s,                                bp_mult=1.0)
        res.stop  = round_to_tick(today_s - sign * _C9B_STOP_BP,          bp_mult=1.0)
        res.t1    = round_to_tick(today_s + sign * _C9B_STOP_BP,          bp_mult=1.0)
        res.t2    = round_to_tick(today_s + sign * 2.0 * _C9B_STOP_BP,    bp_mult=1.0)
        front_c, back_c = _pair_today_closes(
            close_history_per_contract, front_sym, back_sym, asof_date)
        res.lots_at_10k_risk, res.legs = _c9_legs_and_lots(
            res.direction, _C9B_STOP_BP, front_sym, back_sym,
            front_close=front_c, back_close=back_c,
        )

    res.interpretation = (
        f"FIRED {'steepener' if delta_5d > 0 else 'flattener'} {offset_months}M — "
        f"Δslope/5d = {delta_5d:+.2f} bp"
        if res.state == "FIRED" else
        f"NEAR (Δslope/5d = {delta_5d:+.2f} bp · {offset_months}M)"
        if res.state == "NEAR" else
        f"APPROACHING (Δslope/5d = {delta_5d:+.2f} bp · {offset_months}M)"
        if res.state == "APPROACHING" else
        f"slope flat (Δ5d = {delta_5d:+.2f} bp · {offset_months}M)"
    )
    return res


# =============================================================================
# Helpers — pick default front/back contracts for C9
# =============================================================================
# Default offsets in months for C9 multi-pair scanning (curve ladder).
# 6M  = "near slope"   12M = "1Y slope" (canonical) · 18M = "longer slope" · 24M = "2Y slope"
C9_DEFAULT_OFFSET_MONTHS = (6, 12, 18, 24)


def pick_c9_contract_pairs(contracts_meta: pd.DataFrame, asof_date: date,
                             offsets_months: tuple = C9_DEFAULT_OFFSET_MONTHS) -> list:
    """Pick (front, back, offset_months) tuples — one per requested offset.

    Front is always the nearest quarterly outright that has not yet expired.
    Back is the quarterly closest to ``front_expiry + offset_months × ~30d``.
    If a given offset has no distinct back contract (e.g. universe too short),
    that offset is silently dropped.

    Returns a list of ``(front_sym, back_sym, offset_months)`` tuples,
    possibly empty if no front contract is selectable.
    """
    if contracts_meta is None or contracts_meta.empty:
        return []
    qm = contracts_meta.copy()
    if "symbol" not in qm.columns:
        return []
    def _is_q(sym):
        try:
            return is_quarterly(sym)
        except Exception:
            return True
    qm["_q"] = qm["symbol"].apply(_is_q)
    qm = qm[qm["_q"] == True]
    if qm.empty:
        return []
    qm = qm.copy()
    qm["expiry_dt"] = qm.apply(
        lambda r: third_wednesday(int(r["expiry_year"]), int(r["expiry_month"]))
        if pd.notna(r["expiry_year"]) and pd.notna(r["expiry_month"]) else None,
        axis=1,
    )
    qm = qm.dropna(subset=["expiry_dt"])
    qm = qm[qm["expiry_dt"] >= asof_date]
    if qm.empty:
        return []
    qm = qm.sort_values("expiry_dt").reset_index(drop=True)
    front = qm.iloc[0]
    front_sym = front["symbol"]
    out = []
    seen_backs = set()
    for off in offsets_months:
        target_back_dt = front["expiry_dt"] + timedelta(days=int(off) * 30)
        diffs = (qm["expiry_dt"] - target_back_dt).abs()
        back = qm.iloc[diffs.argmin()]
        back_sym = back["symbol"]
        if back_sym == front_sym or back_sym in seen_backs:
            # Skip degenerate cases (front, or duplicate of an earlier offset's back)
            continue
        seen_backs.add(back_sym)
        out.append((front_sym, back_sym, int(off)))
    return out


def pick_c9_contract_pair(contracts_meta: pd.DataFrame, asof_date: date,
                            offset_months: int = 12) -> tuple:
    """Pick (front, back) outright pair: nearest quarterly + back contract
    ~``offset_months`` later. Both must exist in the meta and be live.

    Returns (front_sym, back_sym) or (None, None) if not pickable.
    Kept for backward compatibility with the prior single-pair API.
    """
    if contracts_meta is None or contracts_meta.empty:
        return (None, None)
    # Filter to quarterly outrights
    qm = contracts_meta.copy()
    if "symbol" not in qm.columns:
        return (None, None)
    # Quarterly = last char of base sym minus year code is in {H, M, U, Z}
    def _is_q(sym):
        try:
            return is_quarterly(sym)
        except Exception:
            return True
    qm["_q"] = qm["symbol"].apply(_is_q)
    qm = qm[qm["_q"] == True]
    if qm.empty:
        return (None, None)
    # Sort by expiry
    qm = qm.copy()
    qm["expiry_dt"] = qm.apply(
        lambda r: third_wednesday(int(r["expiry_year"]), int(r["expiry_month"]))
        if pd.notna(r["expiry_year"]) and pd.notna(r["expiry_month"]) else None,
        axis=1,
    )
    qm = qm.dropna(subset=["expiry_dt"])
    qm = qm[qm["expiry_dt"] >= asof_date]
    if qm.empty:
        return (None, None)
    qm = qm.sort_values("expiry_dt").reset_index(drop=True)
    front = qm.iloc[0]
    target_back_dt = front["expiry_dt"] + timedelta(days=offset_months * 30)
    qm["_diff"] = (qm["expiry_dt"] - target_back_dt).abs()
    back = qm.iloc[qm["_diff"].argmin()]
    if back["symbol"] == front["symbol"]:
        # No further-out contract — try last available
        if len(qm) >= 2:
            back = qm.iloc[-1]
        else:
            return (front["symbol"], None)
    return (front["symbol"], back["symbol"])
