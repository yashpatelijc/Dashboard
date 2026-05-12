"""Dynamic rule-based target/stop mechanism for PCA trade ideas.

Replaces static entry-time target/stop with a daily re-evaluated tiered system.
Each day, given an idea's entry state and the current panel, walks through
4 target tiers and 6 stop tiers in priority order; first hit wins.

Used by:
  - `lib/backtest.py` to simulate exits during historical replay
  - `tabs/us/sra_pca.py` Exit Plan card section to display live tier status
  - `lib/pca_trades.py` `score_conviction.exit_clarity` to credit clear setups

Tier philosophy
---------------
Targets (in priority order — first hit closes the trade or partial):
  T1 — Full revert    : |z| ≤ 0.5 with sign-agreement → exit 100%
  T2 — Partial revert : |z| ≤ entry_|z| × 0.67       → exit 33%
  T3 — Time stop      : days_held ≥ hold_mult × current_HL → exit 100%
  T4 — Signal decay   : n_confirming < entry_n × 0.5 → exit 50%

Stops (in priority order — first hit closes 100%):
  S1 — Adverse breakout : |z| > max(3.5, entry|z| × 1.5) in adverse direction
  S2 — Triple-gate fail : triple_gate.all_three == False (model broken)
  S3 — HL extension     : current_HL > hl_ext_mult × entry_HL or > hold_cap
  S4 — Convexity warning: Piterbarg convexity bias fires (mode-aware threshold)
  S5 — Regime transition: HMM dominant regime changed AND hostile to structure
  S6 — Hard P&L stop    : realized_PnL ≤ -1.5 × expected_PnL (always-on backstop)

The stop checks consider sign-agreement: a residual moving in the favourable
direction is NOT a stop event even if its magnitude exceeds the breakout band.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExitState:
    """Result of evaluating dynamic exit rules for a single open position."""
    target_tier: Optional[int]      # 1..4 if a target fired, else None
    stop_tier: Optional[int]        # 1..6 if a stop fired, else None
    exit_now: bool                  # True if any tier fired
    exit_size_frac: float           # 1.0 = full, 0.33 = partial Tier-2, 0.5 = Tier-4
    reason: str                     # Human-readable reason (one sentence)
    severity: str                   # "target" | "soft_stop" | "hard_stop"


@dataclass(frozen=True)
class EntryState:
    """Snapshot of trade state at entry. Frozen at fill, used for delta checks."""
    entry_date: date
    entry_residual_bp: float        # The residual level when we entered
    entry_z: float                  # Z-score at entry
    entry_hl_d: float               # OU half-life fit at entry
    entry_n_confirming: int         # Cross-confirmation count at entry
    entry_regime_label: Optional[str] = None
    entry_triple_gate_pass: bool = True
    expected_pnl_bp: float = 0.0    # Per-unit expected $ pnl at entry
    direction: str = "long"         # "long" | "short" | "neutral"


def _current_signal_state(panel: dict, leg_fingerprint: str,
                            instrument_symbol: str) -> dict:
    """Extract the latest z, HL, residual, gate state for an instrument from panel.

    Tries (in order): outright residuals, spread residuals, fly residuals, packs.
    Returns dict with keys: z, hl_d, residual_bp, gate, triple_gate_pass, regime
    """
    out = {"z": None, "hl_d": None, "residual_bp": None,
            "gate": "low_n", "triple_gate_pass": False, "regime": None}
    for key in ("residual_outrights", "residual_traded_spreads",
                  "residual_traded_flies", "residual_packs"):
        df = panel.get(key)
        if df is None or df.empty or "instrument" not in df.columns:
            continue
        row = df[df["instrument"] == instrument_symbol]
        if row.empty:
            continue
        r = row.iloc[0]
        out["z"] = r.get("residual_z")
        out["hl_d"] = r.get("half_life")
        out["residual_bp"] = r.get("residual_today_bp")
        out["gate"] = r.get("gate_quality", "low_n")
        out["triple_gate_pass"] = bool(r.get("triple_gate_pass",
                                                 r.get("adf_pass", False)))
        break

    # Pull regime label if available
    regime_stack = panel.get("regime_stack", {})
    hmm = regime_stack.get("hmm_fit") if regime_stack else None
    if hmm is not None and getattr(hmm, "dominant_states", None) is not None:
        try:
            dom = hmm.dominant_states
            out["regime"] = str(dom[-1]) if hasattr(dom, "__getitem__") and len(dom) > 0 else None
        except Exception:
            pass
    return out


def _z_sign_for_revert(direction: str, current_z: Optional[float]) -> int:
    """Return +1 if current_z indicates revert toward target, -1 if away, 0 if unclear.

    For a LONG position (we bought because residual was negative ↔ instrument cheap):
        revert direction is residual moving UP toward 0 (less negative).
        i.e. current_z > entry_z (toward 0 from below).
    For a SHORT position (residual was positive ↔ rich): revert is residual moving
        DOWN toward 0 (less positive), current_z < entry_z.
    """
    if current_z is None or not np.isfinite(current_z):
        return 0
    return 0   # this helper is informational; actual logic lives in tier checks below


def evaluate_dynamic_exit(
    idea,                               # TradeIdea-like (only fields read; duck-typed)
    entry_state: EntryState,
    current_panel: dict,
    days_held: int,
    mode: Optional[str] = None,
    current_pnl_bp: Optional[float] = None,
) -> ExitState:
    """Walk all target and stop tiers in priority order; first hit wins.

    Returns ExitState. Target Tier 2 (partial 33%) and Target Tier 4 (signal decay
    50%) are partial exits; all others are full closes.
    """
    from lib.pca import mode_params
    mp = mode_params(mode or current_panel.get("mode"))
    hold_mult = float(mp["hold_mult"])
    hold_cap = float(mp["hold_cap"])
    hl_ext_mult = float(mp.get("hl_extension_stop_mult", 3.0))

    # Resolve current state for the primary instrument
    legs = getattr(idea, "legs", ()) or ()
    primary_sym = legs[0].symbol if legs else ""
    sig = _current_signal_state(current_panel, getattr(idea, "leg_fingerprint", ""), primary_sym)
    current_z = sig.get("z")
    current_hl = sig.get("hl_d")
    current_triple_pass = bool(sig.get("triple_gate_pass", False))
    current_regime = sig.get("regime")

    # =========================================================================
    # TARGETS (check in priority order; first hit wins)
    # =========================================================================
    # T1 — Full revert: |z| ≤ 0.5 AND z has moved toward 0 from entry
    if current_z is not None and np.isfinite(current_z):
        if abs(current_z) <= 0.5:
            # Confirm direction of movement (sign change or magnitude reduction)
            if abs(current_z) < abs(entry_state.entry_z):
                return ExitState(target_tier=1, stop_tier=None, exit_now=True,
                                   exit_size_frac=1.0,
                                   reason=f"Full revert: |z|={abs(current_z):.2f}σ ≤ 0.5σ "
                                          f"(entry was {entry_state.entry_z:+.2f}σ). Take 100%.",
                                   severity="target")

        # T2 — Partial revert: |z| ≤ entry_|z| × 0.67 (one-third closer to zero)
        if (entry_state.entry_z != 0 and
                abs(current_z) <= abs(entry_state.entry_z) * 0.67 and
                abs(current_z) < abs(entry_state.entry_z)):
            return ExitState(target_tier=2, stop_tier=None, exit_now=True,
                               exit_size_frac=0.33,
                               reason=f"Partial revert: |z|={abs(current_z):.2f}σ ≤ "
                                      f"{abs(entry_state.entry_z) * 0.67:.2f}σ "
                                      f"(67% of entry). Take 33% off, hold rest for T1.",
                               severity="target")

    # T3 — Time target: days held ≥ hold_mult × current_HL
    if current_hl is not None and np.isfinite(current_hl) and current_hl > 0:
        time_threshold = max(1.0, hold_mult * float(current_hl))
        if days_held >= time_threshold:
            return ExitState(target_tier=3, stop_tier=None, exit_now=True,
                               exit_size_frac=1.0,
                               reason=f"Time stop: held {days_held}d ≥ {time_threshold:.0f}d "
                                      f"({hold_mult:.1f}×HL={float(current_hl):.1f}d). "
                                      f"Exit at market — half-life elapsed.",
                               severity="target")

    # T4 — Signal decay: cross-confirmation dropped to half of entry AND held ≥5d
    n_conf_current = getattr(idea, "n_confirming_sources", 1)
    if (entry_state.entry_n_confirming > 1 and
            n_conf_current < entry_state.entry_n_confirming * 0.5 and
            days_held >= 5):
        return ExitState(target_tier=4, stop_tier=None, exit_now=True,
                           exit_size_frac=0.5,
                           reason=f"Signal decay: confirming sources {n_conf_current} "
                                  f"< 50% of entry ({entry_state.entry_n_confirming}). "
                                  f"Take half off, monitor remainder.",
                           severity="target")

    # =========================================================================
    # STOPS (check in priority order; first hit wins; all stops close 100%)
    # =========================================================================
    # S1 — Adverse z-score breakout
    # If current_z has same sign as entry_z (residual hasn't reverted) AND
    # magnitude has expanded beyond max(3.5, entry|z|×1.5), it's a regime break.
    if current_z is not None and np.isfinite(current_z):
        breakout_threshold = max(3.5, abs(entry_state.entry_z) * 1.5)
        same_sign = (np.sign(current_z) == np.sign(entry_state.entry_z)
                       if entry_state.entry_z != 0 else False)
        if same_sign and abs(current_z) > breakout_threshold:
            return ExitState(target_tier=None, stop_tier=1, exit_now=True,
                               exit_size_frac=1.0,
                               reason=f"Adverse breakout: |z|={abs(current_z):.2f}σ > "
                                      f"{breakout_threshold:.2f}σ in same direction. "
                                      f"Regime break — exit immediately.",
                               severity="hard_stop")

    # S2 — Triple-gate failure (model assumptions violated)
    if entry_state.entry_triple_gate_pass and not current_triple_pass:
        return ExitState(target_tier=None, stop_tier=2, exit_now=True,
                           exit_size_frac=1.0,
                           reason="Triple-gate failure: residual no longer stationary "
                                  "(ADF/KPSS/VR test failed since entry). Model invalid.",
                           severity="hard_stop")

    # S3 — Half-life extension
    entry_hl_safe = (float(entry_state.entry_hl_d)
                      if entry_state.entry_hl_d is not None
                      and np.isfinite(float(entry_state.entry_hl_d))
                      and float(entry_state.entry_hl_d) > 0 else None)
    if (current_hl is not None and np.isfinite(current_hl) and current_hl > 0
            and entry_hl_safe is not None):
        if (current_hl > hl_ext_mult * entry_hl_safe
                or current_hl > hold_cap):
            return ExitState(target_tier=None, stop_tier=3, exit_now=True,
                               exit_size_frac=1.0,
                               reason=f"HL extension: current={current_hl:.1f}d > "
                                      f"{hl_ext_mult:.0f}×entry "
                                      f"({entry_hl_safe:.1f}d) or > mode-cap "
                                      f"{hold_cap:.0f}d. Mean-reversion broken — exit.",
                               severity="soft_stop")

    # S4 — Convexity warning fires (re-evaluated each day)
    if bool(getattr(idea, "convexity_warning", False)) and days_held > 5:
        return ExitState(target_tier=None, stop_tier=4, exit_now=True,
                           exit_size_frac=1.0,
                           reason="Convexity warning: Piterbarg bias exceeded threshold. "
                                  "Rate move has made convexity material.",
                           severity="soft_stop")

    # S5 — Regime transition AGAINST trade
    if (entry_state.entry_regime_label is not None
            and current_regime is not None
            and current_regime != entry_state.entry_regime_label):
        # Hostile = if entry was rangebound and now trending (any trending state)
        hostile = (("trend" in str(current_regime).lower() or
                     "break" in str(current_regime).lower())
                     and "range" in str(entry_state.entry_regime_label).lower())
        if hostile:
            return ExitState(target_tier=None, stop_tier=5, exit_now=True,
                               exit_size_frac=1.0,
                               reason=f"Regime transition: '{entry_state.entry_regime_label}' "
                                      f"→ '{current_regime}'. New regime hostile to mean-reversion trade.",
                               severity="soft_stop")

    # S6 — Hard P&L stop (always-on backstop)
    if (current_pnl_bp is not None and np.isfinite(current_pnl_bp)
            and entry_state.expected_pnl_bp > 0):
        loss_threshold = -1.5 * abs(entry_state.expected_pnl_bp)
        if current_pnl_bp <= loss_threshold:
            return ExitState(target_tier=None, stop_tier=6, exit_now=True,
                               exit_size_frac=1.0,
                               reason=f"Hard P&L stop: realized={current_pnl_bp:+.2f} bp "
                                      f"≤ {loss_threshold:+.2f} bp "
                                      f"(-1.5× expected {entry_state.expected_pnl_bp:.2f}). "
                                      f"Maximum acceptable loss hit.",
                               severity="hard_stop")

    # No tier fired — keep position open
    return ExitState(target_tier=None, stop_tier=None, exit_now=False,
                       exit_size_frac=0.0, reason="In-trade — no exit triggered.",
                       severity="target")


def planned_levels(idea, entry_state: EntryState, mode: Optional[str] = None) -> dict:
    """Compute planned target/stop levels at entry for UI display.

    Returns dict with keys:
      target_t1_z, target_t1_residual_bp, target_t1_day_estimate
      target_t2_z, target_t2_residual_bp
      target_t3_day
      stop_s1_z, stop_s1_residual_bp
      stop_s3_hl_threshold_d
      stop_s6_pnl_bp
    """
    from lib.pca import mode_params
    mp = mode_params(mode)
    hold_mult = float(mp["hold_mult"])
    hold_floor = float(mp["hold_floor"])
    hold_cap = float(mp["hold_cap"])
    hl_ext_mult = float(mp.get("hl_extension_stop_mult", 3.0))

    entry_z = float(entry_state.entry_z) if entry_state.entry_z is not None else 0.0
    entry_resid = float(entry_state.entry_residual_bp or 0.0)
    raw_hl = entry_state.entry_hl_d
    entry_hl = float(raw_hl) if (raw_hl is not None and np.isfinite(float(raw_hl)) and float(raw_hl) > 0) else None

    # T1 — z target at sign-flipped 0.5σ
    t1_z = np.sign(entry_z) * 0.5 if entry_z != 0 else 0.0
    # Scale residual by ratio of z's (residual is proportional to z within window)
    t1_resid = entry_resid * (0.5 / abs(entry_z)) if abs(entry_z) > 0.001 else 0.0
    t1_day = (hold_mult * entry_hl * 0.7) if entry_hl is not None else None    # ~70% of hold to hit T1

    # T2 — z reduced to 67% of entry
    t2_z = entry_z * 0.67
    t2_resid = entry_resid * 0.67

    # T3 — time stop
    t3_day = (hold_mult * entry_hl) if entry_hl is not None else None
    if t3_day is not None:
        t3_day = max(hold_floor, min(t3_day, hold_cap))

    # S1 — adverse breakout level
    breakout_threshold = max(3.5, abs(entry_z) * 1.5)
    s1_z = np.sign(entry_z) * breakout_threshold
    s1_resid = entry_resid * (breakout_threshold / abs(entry_z)) if abs(entry_z) > 0.001 else 0.0

    # S3 — HL extension threshold
    s3_hl = (hl_ext_mult * entry_hl) if entry_hl is not None else hold_cap

    # S6 — hard P&L stop
    s6_pnl = -1.5 * float(entry_state.expected_pnl_bp or 0.0)

    return {
        "target_t1_z": float(t1_z),
        "target_t1_residual_bp": float(t1_resid),
        "target_t1_day_estimate": float(t1_day) if t1_day is not None else None,
        "target_t2_z": float(t2_z),
        "target_t2_residual_bp": float(t2_resid),
        "target_t3_day": float(t3_day) if t3_day is not None else None,
        "stop_s1_z": float(s1_z),
        "stop_s1_residual_bp": float(s1_resid),
        "stop_s3_hl_threshold_d": float(s3_hl),
        "stop_s6_pnl_bp": float(s6_pnl),
    }


def format_exit_state_for_ui(es: ExitState) -> dict:
    """Format an ExitState into UI-friendly fields (label, color, badge)."""
    if not es.exit_now:
        return {"label": "OPEN", "color": "neutral",
                 "detail": "Trade open — no exit triggered.",
                 "badge": "·"}
    if es.target_tier is not None:
        color = "green"
        label = f"TARGET T{es.target_tier}"
        if es.exit_size_frac < 1.0:
            label += f" ({int(es.exit_size_frac * 100)}%)"
    else:
        color = "red" if es.severity == "hard_stop" else "amber"
        label = f"STOP S{es.stop_tier}"
    return {"label": label, "color": color,
             "detail": es.reason, "badge": label}


def entry_state_from_idea(idea, entry_date: date) -> EntryState:
    """Build EntryState from a TradeIdea at the moment of fill.

    None / NaN HL is preserved (not coerced to 0) so the T3 time-stop check
    correctly skips this tier rather than firing immediately.
    """
    hl_raw = getattr(idea, "half_life_d", None)
    try:
        hl_v = float(hl_raw) if hl_raw is not None and np.isfinite(float(hl_raw)) else float("nan")
    except (TypeError, ValueError):
        hl_v = float("nan")
    return EntryState(
        entry_date=entry_date,
        entry_residual_bp=float(getattr(idea, "entry_bp", 0.0) or 0.0),
        entry_z=float(getattr(idea, "z_score", 0.0) or 0.0),
        entry_hl_d=hl_v,
        entry_n_confirming=int(getattr(idea, "n_confirming_sources", 1) or 1),
        entry_regime_label=getattr(idea, "regime_label", None),
        entry_triple_gate_pass=bool(getattr(idea, "triple_gate_pass",
                                                 getattr(idea, "adf_pass", False))),
        expected_pnl_bp=float(getattr(idea, "expected_pnl_bp", 0.0) or 0.0),
        direction=str(getattr(idea, "direction", "long") or "long"),
    )
