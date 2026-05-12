"""Phase G — tooltip system for the Technicals subtab.

Three tooltip types:

  G1. Setup name hover  — 6 sections: TITLE / FORMULA / FIRED / NEAR /
      APPROACHING / INTERPRET. Used in Today's Fires cards, state-matrix
      column headers, drill-down headers.
  G2. Composite scoring cells (Trend / MR / Final) — range → interpretation
      mapping with the cell's value highlighted.
  G3. Regime + Take column headers — definitions of every label/pill.

All builders return plain strings ready for HTML attribute escaping by the
caller (use `_tooltip_attr(...)` in technicals.py).
"""
from __future__ import annotations

from typing import Optional

from lib.setups.registry import display_name, get_registry_entry
from lib.setups.near_thresholds import methodology_for
from lib.setups._NEAR_CITATIONS import cite


# =============================================================================
# G1 — setup name 6-section tooltip
# =============================================================================

def setup_six_section_tooltip(setup_id: str,
                                  result: Optional[dict] = None,
                                  state_label: Optional[str] = None) -> str:
    """Build the 6-section tooltip text:

        TITLE     · <Display Name>  (<setup_id>)
        FORMULA   · <logic_long / logic_short>
        FIRED     · <FIRED bucket text from registry>
        NEAR      · <NEAR rule from research-backed methodology + citation>
        APPROACH. · <APPROACHING rule + citation>
        INTERPRET · <plain-English plain language>

    If ``result`` (a dict from the detector) is supplied, the FIRED row is
    augmented with the current ``state_label`` and key inputs. Otherwise
    the tooltip is purely descriptive.
    """
    reg = get_registry_entry(setup_id)
    name = display_name(setup_id) if reg else setup_id

    # FORMULA — concatenate long + short logic
    logic_long = reg.get("logic_long", "")
    logic_short = reg.get("logic_short", "")
    if logic_long and logic_short and logic_long != logic_short:
        formula = f"LONG:  {logic_long}\nSHORT: {logic_short}"
    elif logic_long:
        formula = logic_long
    elif logic_short:
        formula = logic_short
    else:
        formula = "(see registry — no formula text)"

    # FIRED bucket
    buckets = reg.get("buckets", []) or []
    fired_text = next(
        (b[0] for b in buckets if (len(b) >= 2 and b[1] == "FIRED")), None)
    if fired_text is None and buckets:
        fired_text = buckets[0][0]
    if fired_text is None:
        fired_text = "(all setup conditions met)"

    # NEAR / APPROACHING — research-backed methodology
    methodology = methodology_for(setup_id)
    near_text = methodology.get("near_text", "(not classified)")
    approach_text = methodology.get("approach_text", "(not classified)")
    citation_key = methodology.get("citation", "")
    citation = cite(citation_key) if citation_key else ""

    # INTERPRETATION — registry note OR result-injected interpretation
    interp = reg.get("note", "")
    if result and result.get("interpretation"):
        interp = result["interpretation"]
    if not interp:
        interp = "(no plain-English interpretation registered)"

    # Build the tooltip
    lines = [
        f"TITLE      · {name}  ({setup_id})",
        f"FORMULA    · {formula}",
        f"FIRED      · {fired_text}",
        f"NEAR       · {near_text}",
        f"APPROACH.  · {approach_text}",
    ]
    if citation:
        lines.append(f"           · Source: {citation}")
    lines.append(f"INTERPRET  · {interp}")

    if state_label:
        lines.append(f"CURRENT    · {state_label}")

    if result:
        ki = result.get("key_inputs", {})
        if ki:
            ki_str = " · ".join(
                f"{k}={v:+.3g}" if isinstance(v, (int, float)) and not isinstance(v, bool)
                else f"{k}={v}"
                for k, v in list(ki.items())[:6]
            )
            lines.append(f"INPUTS     · {ki_str}")
        if result.get("entry") is not None:
            e = result.get("entry"); s = result.get("stop")
            t1 = result.get("t1"); t2 = result.get("t2")
            lots = result.get("lots_at_10k_risk")
            bits = []
            if e is not None: bits.append(f"entry {e:.4f}")
            if s is not None: bits.append(f"stop {s:.4f}")
            if t1 is not None: bits.append(f"T1 {t1:.4f}")
            if t2 is not None: bits.append(f"T2 {t2:.4f}")
            if lots: bits.append(f"{lots} lots @ $10K")
            if bits:
                lines.append(f"TRADE      · " + " · ".join(bits))
    return "\n".join(lines)


# =============================================================================
# G2 — composite scoring cell tooltip
# =============================================================================

COMPOSITE_RANGES = (
    (+0.7, "STRONG SIGNAL ↑   (highest-conviction trend long)"),
    (+0.5, "moderate ↑        (trend long, secondary confirmation helpful)"),
    (+0.3, "soft ↑            (weak signal, treat as bias not trade)"),
    (-0.3, "neutral           (no edge, sit out)"),
    (-0.5, "soft ↓"),
    (-0.7, "moderate ↓"),
    (-1.0, "STRONG SIGNAL ↓"),
)


def composite_cell_tooltip(composite_name: str, value: Optional[float],
                              factor_breakdown: Optional[dict] = None) -> str:
    """6-line tooltip for a composite cell (Trend / MR / Final).

    Shows the cell's current value, the 7-band range → interpretation
    mapping, the band the cell falls into, and an optional factor breakdown.
    """
    if value is None:
        return f"{composite_name}: N/A"
    band_label = "neutral"
    for thresh, label in COMPOSITE_RANGES:
        if value >= thresh:
            band_label = label
            break
        if value <= thresh:
            band_label = label
    lines = [f"{composite_name} = {value:+.2f}", "", "Range → interpretation:"]
    for thresh, label in COMPOSITE_RANGES[:4]:
        marker = "→ " if (value >= thresh and "current" not in lines[-1]) else "   "
        lines.append(f"  ≥ {thresh:+.1f}   {label}")
    for thresh, label in COMPOSITE_RANGES[4:]:
        lines.append(f"  ≤ {thresh:+.1f}   {label}")
    lines.append("")
    lines.append(f"This cell: {value:+.2f} → {band_label.split('  ')[0].strip()}")
    if factor_breakdown:
        items = list(factor_breakdown.items())[:8]
        lines.append("")
        lines.append("Factor breakdown:")
        for k, v in items:
            sign = "+" if (isinstance(v, (int, float)) and v > 0) else "−" if (isinstance(v, (int, float)) and v < 0) else " "
            lines.append(f"  {k}: {v:+.3f}" if isinstance(v, (int, float)) else f"  {k}: {v}")
    return "\n".join(lines)


# =============================================================================
# G3 — Regime & Take column-header tooltips
# =============================================================================

REGIME_DEFINITIONS = (
    ("TRENDING_HIGH_VOL",  "ADX_14 ≥ 25  AND  ATR pctile > 0.75 — strong trend with elevated volatility"),
    ("TRENDING_LOW_VOL",   "ADX_14 ≥ 25  AND  ATR pctile < 0.25 — trend on quiet vol; favors breakouts"),
    ("TRENDING_NEUTRAL",   "ADX_14 ≥ 25  AND  ATR pctile in [0.25, 0.75]"),
    ("RANGING_HIGH_VOL",   "ADX_14 < 20  AND  ATR pctile > 0.75 — choppy / volatile range; mean-reversion bias"),
    ("RANGING_LOW_VOL",    "ADX_14 < 20  AND  ATR pctile < 0.25 — quiet range; squeeze precursor"),
    ("RANGING_NEUTRAL",    "ADX_14 < 20  AND  ATR pctile in [0.25, 0.75]"),
    ("NEUTRAL",            "ADX_14 in [20, 25)  OR  ATR pctile in [0.25, 0.75] — no clear regime"),
)


def regime_header_tooltip() -> str:
    """Tooltip for the Regime column header — definitions of every regime label."""
    lines = ["REGIME labels (gameplan §7.5)",
             "",
             "Conditions for each:"]
    for label, defn in REGIME_DEFINITIONS:
        lines.append(f"  {label:<22} {defn}")
    return "\n".join(lines)


TAKE_DEFINITIONS = (
    ("STRONG SIGNAL ↑", "FINAL ≥ +0.7 — highest conviction; act on it"),
    ("moderate ↑",      "FINAL +0.5..+0.7 — directional bias; size moderately"),
    ("soft ↑",          "FINAL +0.3..+0.5 — weak; treat as filter, not entry"),
    ("neutral",         "FINAL −0.3..+0.3 — no edge; sit out"),
    ("soft ↓",          "FINAL −0.5..−0.3 — weak short bias"),
    ("moderate ↓",      "FINAL −0.7..−0.5 — directional short bias"),
    ("STRONG SIGNAL ↓", "FINAL ≤ −0.7 — highest conviction short"),
)


def take_header_tooltip() -> str:
    """Tooltip for the TAKE column header — definitions of every TAKE pill."""
    lines = ["TAKE pill (FINAL_COMPOSITE bucket)",
             "",
             "Conditions for each:"]
    for label, defn in TAKE_DEFINITIONS:
        lines.append(f"  {label:<18} {defn}")
    return "\n".join(lines)


def composite_header_tooltip(composite_name: str = "Composite") -> str:
    """Generic tooltip for a Trend/MR/Final composite column header."""
    lines = [f"{composite_name} score on [-1, +1]",
             "",
             "Range → interpretation:"]
    for thresh, label in COMPOSITE_RANGES[:4]:
        lines.append(f"  ≥ {thresh:+.1f}   {label}")
    for thresh, label in COMPOSITE_RANGES[4:]:
        lines.append(f"  ≤ {thresh:+.1f}   {label}")
    return "\n".join(lines)
