"""Per-trade interpretation engine.

For every TradeIdea, produces:
  1. `interpret_factors(idea, panel) -> list[FactorInterpretation]`
     A list of structured per-factor interpretations — what each signal
     SAYS for THIS specific trade (not generic). Used in the dossier
     "factor interpretation" expander.

  2. `convert_to_sr3_prices(idea, panel) -> SR3Prices`
     Translates the bp-residual entry/target/stop into actual SR3
     contract prices (handles the yield↔price flip explicitly so users
     can verify the contract action without re-deriving sign convention).

  3. `build_recent_chart_data(symbol, panel, lookback=90) -> dict`
     Pulls the last 90 calendar days of OHLC + PCA-implied FV path
     for charting. Cached on panel.

  4. `chartable_overlay_for_trade(idea, panel) -> dict`
     For a multi-leg trade, returns the synthetic spread/fly price
     series to overlay on the chart.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

# Public helpers exported from this module:
#   - build_dynamic_narrative(idea, sr3, factors, panel) → str
#       The trade-specific narrative paragraph that walks the user through
#       what analysis ran, what the numbers came out to, and WHY the engine
#       therefore recommends this specific entry/target/stop.
#   - convert_to_sr3_prices(idea, panel) → SR3Prices
#   - interpret_factors(idea, panel) → list[FactorInterpretation]
#   - build_recent_chart_data(symbol, panel, lookback=90) → dict
from typing import Optional

import numpy as np
import pandas as pd

from lib.pca_concepts import CONCEPTS, get_concept


# =============================================================================
# Data classes
# =============================================================================
@dataclass(frozen=True)
class FactorInterpretation:
    """One factor's interpretation FOR THIS SPECIFIC TRADE.

    `key` matches a CONCEPTS entry (for tooltips); `value` is the actual
    number observed; `tier` is one of "supportive" / "neutral" / "caveat";
    `headline` is a 1-line plain-English summary; `detail` is a paragraph.
    """
    key: str
    value_display: str            # e.g. "+1.97σ" or "12.4 days"
    tier: str                     # "supportive" | "neutral" | "caveat"
    headline: str                 # 1-line "what this MEANS for this trade"
    detail: str                   # 1-paragraph explanation
    weight_in_conviction: float   # 0..1 contribution to total conviction


@dataclass(frozen=True)
class SR3Prices:
    """Actual SR3 contract prices for a trade.

    All prices in SR3 quote units (e.g., 96.0250).

    For an outright SR3 trade:
      - contract_action = "BUY" or "SELL" (the actual order side)
      - entry_price     = current close
      - target_price    = the price the residual implies it'll move TO
      - stop_price      = 2.5× wider against
      - pnl_per_contract_dollar  = abs(target - entry) × $25 / 0.01

    For a spread/fly:
      - per_leg_prices = list of (symbol, action, entry, target, stop)
      - net_entry_bp / net_target_bp / net_stop_bp = residual-space numbers
      - The renderer shows BOTH the per-leg contract prices AND the net bp
        residual for the trader to verify.
    """
    is_outright: bool
    contract_action: Optional[str]    # "BUY" / "SELL" for outright; None for multi-leg
    entry_price: Optional[float]
    target_price: Optional[float]
    stop_price: Optional[float]
    pnl_per_contract_dollar: Optional[float]
    risk_per_contract_dollar: Optional[float]
    risk_reward: Optional[float]
    # Multi-leg
    per_leg_prices: tuple             # tuple of dicts: {symbol, action, entry, target, stop, pv01_wt}
    net_entry_bp: Optional[float]
    net_target_bp: Optional[float]
    net_stop_bp: Optional[float]
    # Yield-space (for verification + chart annotations)
    entry_yield_bp: Optional[float]
    target_yield_bp: Optional[float]
    stop_yield_bp: Optional[float]
    # Synthetic spread/fly price — sign-only combination (what spread/fly traders quote)
    synth_entry_unit: Optional[float] = None
    synth_target_unit: Optional[float] = None
    synth_stop_unit: Optional[float] = None
    # LISTED CME spread/fly contract (when the leg pattern matches a tradable product)
    listed_symbol: Optional[str] = None
    listed_entry_price: Optional[float] = None
    listed_target_price: Optional[float] = None
    listed_stop_price: Optional[float] = None
    # LISTED contract P&L + R:R (CME convention: 1 listed unit = 1 bp = $25/contract)
    listed_pnl_dollar: Optional[float] = None
    listed_risk_dollar: Optional[float] = None
    listed_risk_reward: Optional[float] = None
    # Synthetic-from-legs canonical reconstruction (always computed for 2/3 legs)
    synth_canonical_bp: Optional[float] = None        # Σ cw × leg_price × 100
    synth_target_bp: Optional[float] = None           # = 0 (FV by construction)
    synth_stop_bp: Optional[float] = None             # = 3.5 × synth_canonical_bp
    synth_pnl_dollar: Optional[float] = None
    synth_risk_dollar: Optional[float] = None
    synth_risk_reward: Optional[float] = None
    listed_vs_synth_diff_bp: Optional[float] = None
    # Provenance: "listed" / "synthetic" / "per_leg" / "none"
    price_source: str = "none"
    # Package-level action (for spreads/flies that trade as a single CME contract):
    # "BUY" if listed/synth entry < 0 (cheap vs FV=0, expect rise);
    # "SELL" if > 0 (rich vs FV=0, expect fall). None for outright (use contract_action).
    package_action: Optional[str] = None
    # Plain-English explanation of WHY the action is what it is
    explanation: str = ""


# =============================================================================
# SR3 contract price conversion
# =============================================================================
def _sr3_price_today(symbol: str, panel: dict) -> Optional[float]:
    """Today's close price for an SR3 contract."""
    close = panel.get("outright_close_panel", pd.DataFrame())
    if close is None or close.empty or symbol not in close.columns:
        return None
    s = close[symbol].dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


def _bp_to_price_delta(bp: float) -> float:
    """1 bp of SR3 yield = 0.01 in SR3 price units."""
    return bp * 0.01


def _bp_to_dollars_per_contract(bp: float, panel: Optional[dict] = None) -> float:
    """Convert |bp| → local-currency-per-contract using the panel's market config.

    For SRA: 1 bp = $25. For ER/FER/FSR: €25 (or CHF25). For SON: £12.50.
    For YBA: AUD24. For CRA: CAD25. Falls back to $25 if panel is None or has
    no market key (legacy callers).
    """
    if panel is not None:
        return abs(bp) * _market_conventions(panel)["local_per_bp"]
    return abs(bp) * 25.0


SR3_TICK = 0.0025    # outright SR3 tick = 0.0025 = ¼ bp = $6.25 / contract

# CME-listed spread/fly contracts are quoted in BP of price-differential
# (×100 of raw price). 1 unit of listed = 1 bp = $25 / contract.
# Tick = 0.25 bp = $6.25 / contract (same dollar tick value as outright).
LISTED_BP_TICK = 0.25
USD_PER_LISTED_BP = 25.0    # $/bp/contract for CME-listed spread or fly (SRA default)


def _market_conventions(panel: dict) -> dict:
    """Look up tick + $/bp from panel's market config. Falls back to SRA."""
    market = (panel or {}).get("market") or {}
    return {
        "outright_tick": float(market.get("tick_bp", 0.25)) / 100.0,    # in price units
        "listed_tick_bp": float(market.get("tick_bp", 0.25)),
        "local_per_bp": float(market.get("dollar_per_bp_local", 25.0)),
        "usd_per_bp": float(market.get("usd_per_bp", 25.0)),
        "currency": market.get("currency", "USD"),
        "currency_symbol": market.get("currency_symbol", "$"),
    }


def _round_to_tick(price: Optional[float], panel: Optional[dict] = None) -> Optional[float]:
    """Round an outright price to the nearest tradable tick. SRA default 0.0025."""
    if price is None or not np.isfinite(price):
        return price
    tick = SR3_TICK
    if panel is not None:
        tick = _market_conventions(panel)["outright_tick"]
    return round(price / tick) * tick


def _round_to_listed_tick(price_bp: Optional[float],
                            panel: Optional[dict] = None) -> Optional[float]:
    """Round a listed spread/fly price (in bp units) to the nearest tick."""
    if price_bp is None or not np.isfinite(price_bp):
        return price_bp
    tick = LISTED_BP_TICK
    if panel is not None:
        tick = _market_conventions(panel)["listed_tick_bp"]
    return round(price_bp / tick) * tick


def convert_to_sr3_prices(idea, panel: dict) -> SR3Prices:
    """Translate bp-residual entry/target/stop into actual SR3 contract prices.

    Conventions:
      - residual_today_bp > 0  →  YIELD is RICH (above PCA-implied FV)
                                  → SR3 PRICE is LOW (below its FV)
                                  → Mean-reversion: yield falls → price RISES
                                  → CORRECT ACTION = BUY (long the futures)

      - residual_today_bp < 0  →  YIELD is CHEAP (below FV)
                                  → SR3 PRICE is HIGH
                                  → Mean-reversion: yield rises → price FALLS
                                  → CORRECT ACTION = SELL (short the futures)

    This handles the known labelling inconsistency in `lib.pca_trades`:
    `TradeLeg.side="sell"` when `residual_z > 0` is technically labelling
    the YIELD-space action ("sell the rich yield") — but for an outright
    SR3 contract, the actual futures action is the OPPOSITE.

    For SPREADS / FLIES: the leg-side conventions in pca_trades are correctly
    in CONTRACT-space (the weights are PV01-balanced and sides map directly).
    We pass them through.
    """
    legs = idea.legs
    is_outright = (idea.structure_type == "outright" and len(legs) == 1)

    # ===== Single-leg LISTED SPREAD or FLY trade =====
    # `gen_spread_fade_ideas` / `gen_fly_arb_ideas` emit single-leg trades
    # where the leg's symbol is the LISTED contract name (e.g. "SRAU28-U29"
    # or "SRAH28-M28-U28"). The leg list has length 1 but the symbol contains
    # hyphens (vs outright symbols which don't). Look up the listed close
    # directly from spread_close_panel / fly_close_panel and compute prices
    # in CME-listed bp units.
    if not is_outright and len(legs) == 1 and "-" in legs[0].symbol:
        sym = legs[0].symbol
        sp_panel = panel.get("spread_close_panel", pd.DataFrame())
        fp_panel = panel.get("fly_close_panel", pd.DataFrame())
        listed_close = None
        if isinstance(sp_panel, pd.DataFrame) and sym in sp_panel.columns:
            ser = sp_panel[sym].dropna()
            if not ser.empty:
                listed_close = float(ser.iloc[-1])
        elif isinstance(fp_panel, pd.DataFrame) and sym in fp_panel.columns:
            ser = fp_panel[sym].dropna()
            if not ser.empty:
                listed_close = float(ser.iloc[-1])
        residual_bp = float(idea.entry_bp or 0.0)
        if listed_close is not None:
            # Engine residual (in CHANGE space, level-cumulated) = listed_today - FV
            # → FV = listed_today - residual ; target = FV (= listed - residual)
            # → stop = FV + 3.5×residual = listed_today + 2.5×residual
            #   (residual moves 2.5× further from 0 in same direction → listed
            #    moves 2.5× further from FV in the opposite direction of the trade)
            listed_target_calc = listed_close - residual_bp
            listed_stop_calc = listed_close + 2.5 * residual_bp
            listed_target_calc = _round_to_listed_tick(listed_target_calc)
            listed_stop_calc = _round_to_listed_tick(listed_stop_calc)
            _lpb = _market_conventions(panel)["local_per_bp"]
            listed_pnl_calc = abs(listed_target_calc - listed_close) * _lpb
            listed_risk_calc = abs(listed_stop_calc - listed_close) * _lpb
            listed_rr_calc = (listed_pnl_calc / listed_risk_calc
                              if listed_risk_calc > 0 else None)
            pkg_action = "BUY" if listed_target_calc > listed_close else "SELL"
            return SR3Prices(
                is_outright=False,
                contract_action=None,
                entry_price=None, target_price=None, stop_price=None,
                pnl_per_contract_dollar=listed_pnl_calc,
                risk_per_contract_dollar=listed_risk_calc,
                risk_reward=listed_rr_calc,
                per_leg_prices=(),
                net_entry_bp=residual_bp,
                net_target_bp=0.0,
                net_stop_bp=3.5 * residual_bp,
                entry_yield_bp=None, target_yield_bp=None, stop_yield_bp=None,
                listed_symbol=sym,
                listed_entry_price=listed_close,
                listed_target_price=listed_target_calc,
                listed_stop_price=listed_stop_calc,
                listed_pnl_dollar=listed_pnl_calc,
                listed_risk_dollar=listed_risk_calc,
                listed_risk_reward=listed_rr_calc,
                price_source="listed",
                package_action=pkg_action,
                explanation=(
                    f"Single-leg listed-product trade. Engine residual = "
                    f"{residual_bp:+.2f} bp implies FV = listed − residual = "
                    f"{listed_close:.2f} − ({residual_bp:+.2f}) = "
                    f"{listed_target_calc:.2f} bp. Execute as one CME ticket."
                ),
            )

    if is_outright:
        sym = legs[0].symbol
        price_today = _sr3_price_today(sym, panel)
        if price_today is None:
            return SR3Prices(
                is_outright=True, contract_action=None,
                entry_price=None, target_price=None, stop_price=None,
                pnl_per_contract_dollar=None,
                risk_per_contract_dollar=None,
                risk_reward=None,
                per_leg_prices=(),
                net_entry_bp=idea.entry_bp,
                net_target_bp=idea.target_bp,
                net_stop_bp=idea.stop_bp,
                entry_yield_bp=None, target_yield_bp=None, stop_yield_bp=None,
                explanation=f"No close price available for {sym}.",
            )

        # Yield-space (in bp)
        # yield_bp = (100 - price) × 100
        entry_yield = (100.0 - price_today) * 100.0
        residual_bp = float(idea.entry_bp or 0.0)
        # Target yield: residual reverts to 0 → yield = entry_yield - residual_bp
        target_yield = entry_yield - residual_bp
        # Stop yield: residual widens 2.5× → yield = entry + 2.5 × residual_bp
        stop_yield = entry_yield + 2.5 * residual_bp

        # Now flip to price-space (price = 100 - yield/100), then round to tick
        entry_price = _round_to_tick(price_today)
        target_price = _round_to_tick(100.0 - target_yield / 100.0)
        stop_price = _round_to_tick(100.0 - stop_yield / 100.0)

        # Determine action from price direction
        contract_action = "BUY" if target_price > entry_price else "SELL"

        # P&L per contract (use rounded prices for what user actually executes)
        actual_target_move_bp = abs(target_price - entry_price) * 100.0
        actual_stop_move_bp = abs(stop_price - entry_price) * 100.0
        pnl_dollar = _bp_to_dollars_per_contract(actual_target_move_bp, panel)
        risk_dollar = _bp_to_dollars_per_contract(actual_stop_move_bp, panel)
        rr = (pnl_dollar / risk_dollar) if risk_dollar > 0 else None

        # Plain-English explanation
        if residual_bp > 0:
            explanation = (
                f"<b>{sym}</b> residual is <b>+{abs(residual_bp):.2f} bp</b> rich today "
                f"(yield is above PCA-implied fair value). When the yield reverts to "
                f"FV, the futures PRICE rises by ~{abs(residual_bp):.2f} bp "
                f"(=${abs(residual_bp)*25:.0f}/contract). "
                f"The correct execution is to <b style='color:var(--green);'>BUY</b> "
                f"at {entry_price:.4f}, target {target_price:.4f}, stop {stop_price:.4f}."
            )
        elif residual_bp < 0:
            explanation = (
                f"<b>{sym}</b> residual is <b>{residual_bp:.2f} bp</b> cheap today "
                f"(yield is below PCA-implied fair value). When the yield reverts to "
                f"FV, the futures PRICE falls by ~{abs(residual_bp):.2f} bp "
                f"(=${abs(residual_bp)*25:.0f}/contract). "
                f"The correct execution is to <b style='color:var(--red);'>SELL</b> "
                f"at {entry_price:.4f}, target {target_price:.4f}, stop {stop_price:.4f}."
            )
        else:
            explanation = "Residual is zero — no trade."

        return SR3Prices(
            is_outright=True,
            contract_action=contract_action,
            entry_price=entry_price,
            target_price=target_price,
            stop_price=stop_price,
            pnl_per_contract_dollar=pnl_dollar,
            risk_per_contract_dollar=risk_dollar,
            risk_reward=rr,
            per_leg_prices=(),
            net_entry_bp=idea.entry_bp,
            net_target_bp=idea.target_bp,
            net_stop_bp=idea.stop_bp,
            entry_yield_bp=entry_yield,
            target_yield_bp=target_yield,
            stop_yield_bp=stop_yield,
            explanation=explanation,
        )

    # ===== Multi-leg trade (spread / fly / pack / basket) =====
    #
    # FIRST: try to match this trade against a LISTED CME spread/fly contract.
    # The DB has actual OHLC for listed calendar spreads (e.g. 'SRAH26-M26')
    # and butterflies (e.g. 'SRAH26-M26-U26'). When the leg pattern matches
    # a listed contract, we use its ACTUAL traded price rather than computing
    # a synthetic combination from the outrights.
    listed_symbol = None
    listed_price = None
    listed_panel = None
    leg_symbols_only = [leg.symbol for leg in legs]
    try:
        from lib.sra_data import (derive_listed_spread_symbol,
                                       derive_listed_fly_symbol)
        if len(legs) == 2:
            listed_symbol = derive_listed_spread_symbol(leg_symbols_only)
            listed_panel = panel.get("spread_close_panel", pd.DataFrame())
        elif len(legs) == 3:
            listed_symbol = derive_listed_fly_symbol(leg_symbols_only)
            listed_panel = panel.get("fly_close_panel", pd.DataFrame())
        if (listed_symbol is not None and isinstance(listed_panel, pd.DataFrame)
                and not listed_panel.empty
                and listed_symbol in listed_panel.columns):
            last = listed_panel[listed_symbol].dropna()
            if not last.empty:
                listed_price = float(last.iloc[-1])
    except Exception:
        listed_price = None

    # Compute per-leg target/stop in ACTUAL SR3 PRICE space, not just entry.
    #
    # Logic: the package residual today is `idea.entry_bp` in weighted-bp units
    # (i.e., w₁·y₁ + w₂·y₂ + ... where y are bp yields). When residual reverts
    # to 0, EACH leg's yield is expected to contribute its proportional share
    # of the reversion. For PV01-balanced structures, the per-leg implied
    # yield move ≈ residual_per_pv01_unit × sign(w_i).
    sum_abs_w = sum(abs(leg.weight_pv01) for leg in legs) or 1.0
    n_legs = len(legs)
    # Normalize so per-leg implied move is in raw bp
    norm_residual_bp = float(idea.entry_bp or 0.0) / sum_abs_w
    # For target (residual → 0), each leg's yield should move by:
    #   per_leg_yield_move_bp = -norm_residual_bp × sign(w_i)
    # (negative sign because positive residual means yields above FV → fall)
    # For stop, residual moves to (stop_bp = 3.5×entry_bp), each leg moves
    #   per_leg_yield_move_stop_bp = -2.5 × norm_residual_bp × sign(w_i)
    #   (against the trade direction)

    per_leg = []
    pkg_pnl_dollar = 0.0
    pkg_risk_dollar = 0.0
    for leg in legs:
        p = _sr3_price_today(leg.symbol, panel)
        if p is None:
            per_leg.append({
                "symbol": leg.symbol, "action": leg.side.upper(),
                "entry_price": None, "target_price": None, "stop_price": None,
                "pv01_wt": leg.weight_pv01,
                "implied_yield_move_target_bp": None,
                "implied_yield_move_stop_bp": None,
            })
            continue
        w_sign = 1.0 if leg.weight_pv01 >= 0 else -1.0
        # Per-leg yield moves on target / stop
        leg_yield_move_target_bp = -norm_residual_bp * w_sign
        leg_yield_move_stop_bp = 2.5 * norm_residual_bp * w_sign
        # Convert yield moves → SR3 price moves (1 bp = 0.01 price)
        leg_price_move_target = -leg_yield_move_target_bp * 0.01    # yield down → price up
        leg_price_move_stop = -leg_yield_move_stop_bp * 0.01
        leg_entry_price = _round_to_tick(float(p))
        leg_target_price = _round_to_tick(float(p) + leg_price_move_target)
        leg_stop_price = _round_to_tick(float(p) + leg_price_move_stop)
        # Per-leg P&L per contract = (price_move) × $25/0.01 = price_move × $2500
        leg_pnl_dollar = abs(leg_target_price - leg_entry_price) * 2500.0
        leg_risk_dollar = abs(leg_stop_price - leg_entry_price) * 2500.0
        pkg_pnl_dollar += leg_pnl_dollar
        pkg_risk_dollar += leg_risk_dollar
        per_leg.append({
            "symbol": leg.symbol,
            "action": "BUY" if leg.weight_pv01 > 0 else "SELL",
            "entry_price": leg_entry_price,
            "target_price": leg_target_price,
            "stop_price": leg_stop_price,
            "pv01_wt": leg.weight_pv01,
            "implied_yield_move_target_bp": leg_yield_move_target_bp,
            "implied_yield_move_stop_bp": leg_yield_move_stop_bp,
            "pnl_per_contract_dollar": leg_pnl_dollar,
            "risk_per_contract_dollar": leg_risk_dollar,
        })

    # SYNTHETIC SPREAD/FLY PRICE — the package as a single quotable number.
    # For PV01-balanced structures: spread/fly price = Σ_i sign(w_i) · p_i
    # (the sign-only combination — gives the natural quoting convention)
    # We provide BOTH:
    #   - synthetic_unit_price: sign-only combination (the "quotable" spread)
    #   - synthetic_pv01_price: PV01-weighted combination (the "true" exposure)
    synth_entry_unit = 0.0
    synth_target_unit = 0.0
    synth_stop_unit = 0.0
    synth_entry_pv01 = 0.0
    synth_target_pv01 = 0.0
    synth_stop_pv01 = 0.0
    for lp in per_leg:
        if lp.get("entry_price") is None:
            continue
        w_sign = 1.0 if lp["pv01_wt"] >= 0 else -1.0
        w_full = lp["pv01_wt"]
        synth_entry_unit += w_sign * lp["entry_price"]
        synth_target_unit += w_sign * (lp.get("target_price") or lp["entry_price"])
        synth_stop_unit += w_sign * (lp.get("stop_price") or lp["entry_price"])
        synth_entry_pv01 += w_full * lp["entry_price"]
        synth_target_pv01 += w_full * (lp.get("target_price") or lp["entry_price"])
        synth_stop_pv01 += w_full * (lp.get("stop_price") or lp["entry_price"])

    # Aggregate "net entry / target / stop" — show net residual movement
    norm_entry = norm_residual_bp                       # weighted-bp / sum|w|
    norm_target = 0.0                                   # reverts to 0
    norm_stop = -2.5 * norm_residual_bp                 # 2.5× wider
    rr = (pkg_pnl_dollar / pkg_risk_dollar) if pkg_risk_dollar > 0 else None

    # ===== LISTED contract pricing (when CME spread/fly product available) =====
    # CME canonical weights and the listed price's relationship to canonical
    # yield residual:
    #   Spread (1,-1):  listed = p1 - p2 = -(y1 - y2)/100 = -canonical_residual/100
    #   Fly (1,-2,1):  listed = p1 - 2p2 + p3 = -(y1 - 2y2 + y3)/100 = -canonical_residual/100
    # Both have Σ cw_i = 0, so listed price at FV (residual=0) = 0.
    #
    # The synthetic price I compute uses SIGN-ONLY weights (1, -1, 1 for a fly)
    # which is DIFFERENT from the canonical (1, -2, 1). For a 3-leg fly, the
    # synthetic move is 3× per-leg-move while the listed move is 4× per-leg-move
    # — different scaling.
    #
    # The correct listed target/stop math:
    #   listed_target = 0                         (FV by construction)
    #   listed_stop   = 3.5 × listed_entry        (2.5× further from FV, same direction)
    listed_target = None
    listed_stop = None
    listed_pnl_dollar = None
    listed_risk_dollar = None
    listed_rr = None
    # SYNTHETIC reconstruction — same bp-unit scale as listed (×100 of raw
    # price-diff). Only meaningful when the leg pattern corresponds to a
    # canonical CME-listed structure: 2-leg calendar spread (1,-1) OR 3-leg
    # butterfly (1,-2,1). For other multi-leg combos (e.g. PC2-spreads with
    # 3+ legs, PC1-baskets) there's no single canonical quote — those trades
    # are executed leg-by-leg from the per-leg execution table.
    synth_canonical_bp = None
    synth_target_bp = None
    synth_stop_bp = None
    synth_pnl_dollar = None
    synth_risk_dollar = None
    synth_rr = None
    listed_vs_synth_diff_bp = None
    # Determine if this leg pattern matches a CME-canonical structure
    # Spread = 2 legs always (one BUY, one SELL — canonical (1,-1))
    # Fly    = 3 legs, target_pc == 3 (PC3 curvature isolated)
    is_canonical_spread = (len(legs) == 2)
    is_canonical_fly = (len(legs) == 3 and idea.structure_type == "fly")
    if is_canonical_spread or is_canonical_fly:
        from lib.sra_data import _outright_sort_key as _sort_key
        sorted_idx = sorted(range(len(legs)), key=lambda i: _sort_key(legs[i].symbol))
        canonical_w = (1, -1) if is_canonical_spread else (1, -2, 1)
        all_prices = []
        for rank, idx in enumerate(sorted_idx):
            p = _sr3_price_today(legs[idx].symbol, panel)
            if p is None:
                all_prices = None
                break
            all_prices.append(p)
        if all_prices is not None:
            synth_canonical_bp = sum(canonical_w[i] * all_prices[i]
                                          for i in range(len(canonical_w))) * 100.0
            synth_target_bp = 0.0
            synth_stop_bp = 3.5 * synth_canonical_bp
            _lpb = _market_conventions(panel)["local_per_bp"]
            synth_pnl_dollar = abs(synth_target_bp - synth_canonical_bp) * _lpb
            synth_risk_dollar = abs(synth_stop_bp - synth_canonical_bp) * _lpb
            synth_rr = (synth_pnl_dollar / synth_risk_dollar
                          if synth_risk_dollar > 0 else None)
            if listed_price is not None:
                listed_vs_synth_diff_bp = listed_price - synth_canonical_bp

    listed_target = None
    listed_stop = None
    listed_pnl_dollar = None
    listed_risk_dollar = None
    listed_rr = None
    if listed_price is not None and len(legs) in (2, 3):
        # IMPORTANT — units: CME-listed spread/fly contracts are quoted in
        # BP-of-price-differential (×100 of raw price units). Verified by
        # cross-checking DB close vs reconstructed Σ cwᵢ × leg_priceᵢ × 100:
        #   SRAH27-H28 listed = -13.5 bp  vs  (96.26 − 96.40) × 100 = -14.0 bp
        #   SRAH27-M27 listed = -1.0 bp   vs  (96.26 − 96.27) × 100 = -1.0 bp
        # 1 unit of listed price = 1 bp of price-diff = $25/contract.
        listed_target = _round_to_listed_tick(0.0)
        listed_stop = _round_to_listed_tick(3.5 * listed_price)
        _lpb = _market_conventions(panel)["local_per_bp"]
        listed_pnl_dollar = abs(listed_target - listed_price) * _lpb
        listed_risk_dollar = abs(listed_stop - listed_price) * _lpb
        listed_rr = (listed_pnl_dollar / listed_risk_dollar
                       if listed_risk_dollar > 0 else None)
    leg_summary = " · ".join(
        f"{leg['action']} {leg['symbol']} @ {leg['entry_price']:.4f}"
        if leg['entry_price'] is not None else f"{leg['action']} {leg['symbol']}"
        for leg in per_leg
    )
    listed_line = ""
    if listed_price is not None and listed_target is not None:
        canonical = "(1, -1)" if len(legs) == 2 else "(1, -2, 1)"
        listed_line = (
            f"<br><b>LISTED CME {idea.structure_type} contract {listed_symbol}</b>: "
            f"<b>{listed_price:+.4f}</b> → "
            f"<b style='color:var(--green);'>{listed_target:+.4f}</b> "
            f"(stop <b style='color:var(--red);'>{listed_stop:+.4f}</b>). "
            f"<br>This uses CME canonical weights <b>{canonical}</b>: "
            f"listed price = Σ cw·leg_price, so fair value = 0 when residual = 0. "
            f"Target = 0; stop = 3.5× entry (2.5× wider from FV). "
            f"<i>Note: synthetic quote below uses sign-only weights (different scaling) — "
            f"the LISTED block is the actionable one.</i>"
        )
    explanation = (
        f"<b>Multi-leg {idea.structure_type}</b> — execute all legs simultaneously: "
        f"{leg_summary}. <br>"
        f"Package residual = <b>{norm_entry:+.2f} bp</b> per unit PV01; "
        f"target <b>{norm_target:+.2f} bp</b> (reverts to 0); "
        f"stop <b>{norm_stop:+.2f} bp</b> (2.5× wider). <br>"
        f"Approximate P&L per package set (1 contract per leg): "
        f"<b>${pkg_pnl_dollar:.0f}</b>; risk <b>${pkg_risk_dollar:.0f}</b>; "
        f"R:R <b>{(f'{rr:.2f}:1' if rr else '—')}</b>.<br>"
        f"<b>Synthetic {idea.structure_type} price</b> (sign-only combination, "
        f"the natural quote): <b>{synth_entry_unit:+.4f}</b> → "
        f"<b style='color:var(--green);'>{synth_target_unit:+.4f}</b> "
        f"(stop <b style='color:var(--red);'>{synth_stop_unit:+.4f}</b>)."
        + listed_line
    )

    return SR3Prices(
        is_outright=False,
        contract_action=None,
        entry_price=None,
        target_price=None,
        stop_price=None,
        pnl_per_contract_dollar=pkg_pnl_dollar,    # per package set (1 contract per leg)
        risk_per_contract_dollar=pkg_risk_dollar,
        risk_reward=rr,
        per_leg_prices=tuple(per_leg),
        net_entry_bp=norm_entry,
        net_target_bp=norm_target,
        net_stop_bp=norm_stop,
        entry_yield_bp=None, target_yield_bp=None, stop_yield_bp=None,
        synth_entry_unit=synth_entry_unit,
        synth_target_unit=synth_target_unit,
        synth_stop_unit=synth_stop_unit,
        listed_symbol=listed_symbol if listed_price is not None else None,
        listed_entry_price=listed_price,
        listed_target_price=listed_target,
        listed_stop_price=listed_stop,
        listed_pnl_dollar=listed_pnl_dollar,
        listed_risk_dollar=listed_risk_dollar,
        listed_risk_reward=listed_rr,
        synth_canonical_bp=synth_canonical_bp,
        synth_target_bp=synth_target_bp,
        synth_stop_bp=synth_stop_bp,
        synth_pnl_dollar=synth_pnl_dollar,
        synth_risk_dollar=synth_risk_dollar,
        synth_risk_reward=synth_rr,
        listed_vs_synth_diff_bp=listed_vs_synth_diff_bp,
        price_source=(
            "listed" if (listed_price is not None and abs(listed_price) > 1e-6)
            else "synthetic" if synth_canonical_bp is not None
            else "per_leg"    # multi-leg engine structure, no listed product;
                              # use per-leg execution table for trading
        ),
        # package_action — a simple BUY/SELL label for EVERY multi-leg trade.
        # For listed/synthetic: sign of entry vs FV=0 (entry < 0 → BUY to bet on
        # rise to 0; entry > 0 → SELL). For per_leg structures (no single CME
        # quote): translate engine YIELD-space direction to package action.
        # direction = "long" (cheap residual, expect rise)  → BUY  the package
        # direction = "short" (rich residual, expect fall) → SELL the package
        package_action=(
            ("BUY" if listed_price < 0 else "SELL")
                if (listed_price is not None and abs(listed_price) > 1e-6)
            else ("BUY" if synth_canonical_bp < 0 else "SELL")
                if (synth_canonical_bp is not None and abs(synth_canonical_bp) > 1e-3)
            else ("BUY" if idea.direction == "long"
                   else "SELL" if idea.direction == "short"
                   else None)
        ),
        explanation=explanation,
    )


# =============================================================================
# Per-trade factor interpretation
# =============================================================================
def _tier_color(tier: str) -> str:
    return {
        "supportive": "var(--green)",
        "neutral":    "var(--text-body)",
        "caveat":     "var(--amber)",
    }.get(tier, "var(--text-body)")


def _interpret_z(idea) -> Optional[FactorInterpretation]:
    z = idea.z_score
    if z is None or not np.isfinite(z):
        return None
    abs_z = abs(z)
    if abs_z >= 2.5:
        tier = "supportive"
        head = f"Strong stretch: |z| = {abs_z:.2f}σ (top 1% historical extremity)"
        detail = (
            f"The cumulative deviation is {abs_z:.2f} standard deviations from its mean "
            f"over the residual lookback window. "
            f"In a normal distribution, this happens roughly 1 day in 100. "
            f"Strong mean-reversion candidates typically fire at |z| > 2.0."
        )
    elif abs_z >= 1.5:
        tier = "supportive"
        head = f"Tradeable stretch: |z| = {abs_z:.2f}σ"
        detail = (
            f"|z| above 1.5 is the engine's threshold for emitting a fade idea. "
            f"At {abs_z:.2f}σ, you're in the top ~7% of historical residual values."
        )
    elif abs_z >= 1.0:
        tier = "neutral"
        head = f"Mild stretch: |z| = {abs_z:.2f}σ (watch-list level)"
        detail = "Not yet a high-conviction signal; engine generally requires |z| ≥ 1.5."
    else:
        tier = "caveat"
        head = f"Weak signal: |z| = {abs_z:.2f}σ"
        detail = "Residual is within 1σ of its mean — no statistical edge."
    return FactorInterpretation(
        key="residual_z",
        value_display=f"{z:+.2f}σ",
        tier=tier,
        headline=head,
        detail=detail,
        weight_in_conviction=0.13 * min(1.0, abs_z / 3.0),
    )


def _interpret_half_life(idea) -> Optional[FactorInterpretation]:
    hl = idea.half_life_d
    if hl is None or not np.isfinite(hl) or hl <= 0:
        return None
    if 3 <= hl <= 30:
        tier = "supportive"
        head = f"OU half-life = {hl:.1f}d (in sweet spot [3, 30]d)"
        detail = (
            f"The mean-reversion speed implies the residual will decay halfway "
            f"to its mean in {hl:.1f} trading days. Combined with the 1.5× rule, "
            f"expected hold time is ~{1.5*hl:.0f}d. This is the ideal range — "
            f"long enough to capture the move, short enough to avoid drift."
        )
        wt = 0.09
    elif 30 < hl <= 60:
        tier = "neutral"
        head = f"OU half-life = {hl:.1f}d (slow but tradeable)"
        detail = (
            f"Half-life > 30d means slower reversion. Hold time will be "
            f"~{1.5*hl:.0f}d — longer capital commitment for the same edge."
        )
        wt = 0.054
    elif hl < 3:
        tier = "caveat"
        head = f"OU half-life = {hl:.1f}d (too fast — risk of over-fit)"
        detail = (
            f"Sub-3-day half-life often reflects noise rather than true "
            f"mean-reversion. Tighten the stop or pass."
        )
        wt = 0.027
    else:
        tier = "caveat"
        head = f"OU half-life = {hl:.1f}d (too slow)"
        detail = (
            f"Mean-reversion takes too long ({hl:.0f}d half-life). "
            f"Capital tied up; better trades likely available."
        )
        wt = 0.018
    return FactorInterpretation(
        key="OU_half_life", value_display=f"{hl:.1f} d",
        tier=tier, headline=head, detail=detail, weight_in_conviction=wt,
    )


def _interpret_adf(idea) -> FactorInterpretation:
    passes = bool(getattr(idea, "adf_pass", False))
    if passes:
        return FactorInterpretation(
            key="ADF",
            value_display="✓ rejects unit root @ 5%",
            tier="supportive",
            headline="ADF: stationarity confirmed",
            detail=(
                "The Augmented Dickey-Fuller test rejected the null of "
                "non-stationarity at the 5% level. This is necessary (but not "
                "sufficient) for a mean-reversion trade. ADF can over-reject "
                "on small samples, which is why we also run KPSS and the "
                "variance ratio (the triple-gate)."
            ),
            weight_in_conviction=0.08,
        )
    return FactorInterpretation(
        key="ADF",
        value_display="✗ cannot reject unit root",
        tier="caveat",
        headline="ADF: stationarity NOT confirmed",
        detail=(
            "ADF could not reject the null hypothesis of a unit root. The "
            "series may be a random walk or have a slow trend. Mean-reversion "
            "edge is unjustified — trade is suppressed by the triple-gate "
            "unless other tests overrule."
        ),
        weight_in_conviction=0.0,
    )


def _interpret_gate(idea) -> FactorInterpretation:
    g = idea.gate_quality
    if g == "clean":
        return FactorInterpretation(
            key="triple_gate",
            value_display="✓ clean (ADF+KPSS+VR all agree)",
            tier="supportive",
            headline="Triple-gate verdict: CLEAN — high confidence in mean-reversion",
            detail=(
                "All three independent stationarity tests confirm the residual is "
                "mean-reverting: ADF rejects unit root, KPSS does not reject "
                "stationarity, and the Lo-MacKinlay variance ratio differs from 1 "
                "in the mean-reverting direction. This is the engine's highest tier "
                "of statistical confidence."
            ),
            weight_in_conviction=0.0,
        )
    if g == "drift":
        return FactorInterpretation(
            key="triple_gate",
            value_display="⚠ drift present (ADF✓ but KPSS rejects)",
            tier="caveat",
            headline="Drift component detected",
            detail=(
                "ADF passes but KPSS rejects stationarity. The series has a slow "
                "drift overlaid on the mean-reverting signal. Trade smaller; "
                "consider hedging with a slow-momentum position."
            ),
            weight_in_conviction=0.0,
        )
    if g == "random_walk":
        return FactorInterpretation(
            key="triple_gate", value_display="✗ random walk — should not be shown",
            tier="caveat",
            headline="Random walk detected (BUG: shouldn't be in the table)",
            detail="The triple-gate marked this as a random walk. The engine should suppress these — flag this to the engineer.",
            weight_in_conviction=0.0,
        )
    if g == "regime_unstable":
        return FactorInterpretation(
            key="triple_gate", value_display="⚠ regime unstable",
            tier="caveat",
            headline="Regime unstable — cross-PC correlation high",
            detail="Cross-PC correlation > 0.30 today — the PCA basis has drifted. The 'isolated' structure is contaminated with other factors.",
            weight_in_conviction=0.0,
        )
    return FactorInterpretation(
        key="triple_gate", value_display=g,
        tier="neutral",
        headline=f"Gate verdict: {g}",
        detail="See the triple_stationarity_gate documentation in lib/mean_reversion.py.",
        weight_in_conviction=0.0,
    )


def _interpret_eff_n(idea) -> FactorInterpretation:
    n = idea.eff_n
    if n >= 100:
        return FactorInterpretation(
            key="eff_n", value_display=str(n),
            tier="supportive",
            headline=f"Sample size = {n} (statistically robust)",
            detail=f"With {n} observations, all test statistics have high statistical power.",
            weight_in_conviction=0.06,
        )
    if n >= 30:
        wt = 0.06 * (n / 100.0)
        return FactorInterpretation(
            key="eff_n", value_display=str(n),
            tier="neutral",
            headline=f"Sample size = {n} (moderate)",
            detail=f"{n} observations is above the eff_n floor. Statistical inference is reliable but conviction is scaled by n/100.",
            weight_in_conviction=wt,
        )
    return FactorInterpretation(
        key="eff_n", value_display=str(n),
        tier="caveat",
        headline=f"Sample size = {n} (below floor)",
        detail=f"Only {n} observations — below the eff_n floor of 30. Test statistics may be unreliable; trade is flagged low_n.",
        weight_in_conviction=0.0,
    )


def _interpret_regime(idea, panel) -> Optional[FactorInterpretation]:
    rs = panel.get("regime_stack", {}) or {}
    hmm = rs.get("hmm_fit")
    if not hmm or hmm.dominant_confidence is None or len(hmm.dominant_confidence) == 0:
        return None
    conf = float(hmm.dominant_confidence[-1])
    if conf >= 0.8:
        return FactorInterpretation(
            key="HMM_regime", value_display=f"dominant = {conf*100:.0f}%",
            tier="supportive",
            headline=f"Regime stable ({conf*100:.0f}% dominant)",
            detail=f"HMM dominant-regime posterior is {conf*100:.0f}% — today belongs cleanly to one regime. Signals derived from historical analogs in that regime are reliable.",
            weight_in_conviction=0.05,
        )
    if conf >= 0.6:
        return FactorInterpretation(
            key="HMM_regime", value_display=f"dominant = {conf*100:.0f}%",
            tier="neutral",
            headline=f"Regime moderately stable ({conf*100:.0f}% dominant)",
            detail=f"Posterior {conf*100:.0f}% is above the 0.60 threshold but not high. Some risk of regime transition.",
            weight_in_conviction=0.05,
        )
    return FactorInterpretation(
        key="HMM_regime", value_display=f"dominant = {conf*100:.0f}%",
        tier="caveat",
        headline=f"Regime transitioning ({conf*100:.0f}% dominant)",
        detail=f"Dominant-regime posterior {conf*100:.0f}% below 0.60 — today is between regimes. Analog FV less reliable; reduce sizing.",
        weight_in_conviction=0.0,
    )


def _interpret_cycle(idea) -> Optional[FactorInterpretation]:
    align = idea.cycle_alignment
    phase = idea.cycle_phase
    if align == "favoured":
        return FactorInterpretation(
            key="cycle_phase",
            value_display=f"{phase or 'unknown'} (favoured)",
            tier="supportive",
            headline=f"Cycle phase '{phase}' FAVOURS this trade",
            detail=f"The trade's direction aligns with the historically-favoured side during '{phase}' phases of the policy cycle. Adds +0.07 to conviction.",
            weight_in_conviction=0.07,
        )
    if align == "counter":
        return FactorInterpretation(
            key="cycle_phase",
            value_display=f"{phase or 'unknown'} (counter)",
            tier="caveat",
            headline=f"Cycle phase '{phase}' COUNTERS this trade",
            detail=f"The trade is against the historically-favoured direction during '{phase}'. The trade can still work, but historical odds are worse.",
            weight_in_conviction=0.0,
        )
    return FactorInterpretation(
        key="cycle_phase",
        value_display=f"{phase or 'unknown'} (neutral)",
        tier="neutral",
        headline=f"Cycle phase '{phase}' is neutral for this trade",
        detail="No strong directional bias from the cycle phase.",
        weight_in_conviction=0.035,
    )


def _interpret_cross_confirmation(idea) -> Optional[FactorInterpretation]:
    n = max(1, idea.n_confirming_sources)
    if n == 1:
        return FactorInterpretation(
            key="conviction", value_display=f"1 source",
            tier="neutral",
            headline=f"Single source ({idea.primary_source})",
            detail="Only one of the 19 trade generators emitted this idea. Other independent analyses didn't confirm — proceed with normal caution.",
            weight_in_conviction=0.0,
        )
    bonus = 0.05 * np.log(1 + n) / np.log(6)
    if n >= 4:
        tier = "supportive"
        head = f"⊕ {n} INDEPENDENT sources confirm — high confluence"
    else:
        tier = "supportive"
        head = f"⊕ {n} sources confirm"
    detail = (
        f"{n} of the 19 trade generators agree on this leg+direction combination: "
        f"{' · '.join(idea.sources)}. Cross-confirmation is the strongest "
        f"single conviction boost — independent analyses converging is much "
        f"stronger evidence than a single-source signal."
    )
    return FactorInterpretation(
        key="conviction", value_display=f"{n} sources",
        tier=tier, headline=head, detail=detail,
        weight_in_conviction=bonus,
    )


def _interpret_cross_asset(idea, panel) -> Optional[FactorInterpretation]:
    ca = panel.get("cross_asset_analysis")
    if ca is None:
        return None
    vol_regime = (ca.vol_regime or {}).get("regime")
    risk_state = (ca.risk_state or {}).get("state")
    if vol_regime in ("stressed", "crisis"):
        return FactorInterpretation(
            key="vol_regime", value_display=vol_regime,
            tier="caveat",
            headline=f"Vol regime: {vol_regime} — widen stops",
            detail=(
                f"MOVE/SKEW/VIX composite is in '{vol_regime}' territory. Analog "
                f"FV bands widen by ×{ca.vol_regime.get('band_widening_factor', 1.0):.1f}. "
                f"Historical mean-reversion edge is less reliable in this vol "
                f"regime — consider smaller size."
            ),
            weight_in_conviction=0.0,
        )
    if risk_state in ("risk_off", "panic"):
        head = f"Risk state: {risk_state} — supports STIR rally"
        detail = "Risk-off tape tends to drive STIRs higher (yields lower). For trades long-yield/short-future, this is HEADWIND. For trades short-yield/long-future, it's TAILWIND."
        return FactorInterpretation(
            key="risk_state", value_display=risk_state,
            tier="neutral", headline=head, detail=detail,
            weight_in_conviction=0.0,
        )
    if vol_regime == "quiet" and risk_state in ("risk_on", "neutral"):
        return FactorInterpretation(
            key="vol_regime", value_display=vol_regime,
            tier="supportive",
            headline=f"Vol regime: {vol_regime} — fades work best here",
            detail=(
                "Vol composite low and risk state benign — classic environment "
                "for relative-value fades. Bands are tight; expected mean-reversion "
                "is most reliable."
            ),
            weight_in_conviction=0.0,
        )
    return None


def _interpret_lifecycle(idea) -> Optional[FactorInterpretation]:
    state = idea.state
    days_alive = idea.days_alive
    if state == "NEW":
        return FactorInterpretation(
            key="conviction", value_display=f"NEW (day 1)",
            tier="supportive",
            headline="Fresh signal — first appearance today",
            detail="Just emerged today; not yet rolled-over or faded. Highest priority for fresh entry.",
            weight_in_conviction=0.04,
        )
    if state == "MATURING":
        return FactorInterpretation(
            key="conviction", value_display=f"MATURING (day {days_alive})",
            tier="supportive",
            headline=f"Maturing — {days_alive} days alive, conviction still rising",
            detail=f"Has appeared for {days_alive} days with rising conviction. Still in the right phase of the trade.",
            weight_in_conviction=0.04,
        )
    if state == "PEAK":
        return FactorInterpretation(
            key="conviction", value_display=f"PEAK (day {days_alive})",
            tier="supportive",
            headline=f"At peak conviction — entry quality high",
            detail=f"Conviction is at its maximum-to-date over the {days_alive} days this signal has been alive.",
            weight_in_conviction=0.04,
        )
    if state == "FADING":
        return FactorInterpretation(
            key="conviction", value_display=f"FADING (day {days_alive})",
            tier="caveat",
            headline=f"Fading — conviction declining (day {days_alive})",
            detail=f"Signal has been alive for {days_alive} days and conviction is declining from its peak. Either the move has already played out or the setup is breaking down.",
            weight_in_conviction=-0.02,
        )
    return None


def interpret_factors(idea, panel: dict) -> list:
    """Build the full list of factor interpretations for one trade."""
    out = []
    for fn in (_interpret_z, _interpret_half_life, _interpret_adf,
                _interpret_gate, _interpret_eff_n,
                lambda i: _interpret_regime(i, panel),
                _interpret_cycle, _interpret_cross_confirmation,
                lambda i: _interpret_cross_asset(i, panel),
                _interpret_lifecycle):
        try:
            res = fn(idea)
        except Exception:
            res = None
        if res is not None:
            out.append(res)
    return out


# =============================================================================
# 90-day chart data
# =============================================================================
def build_recent_chart_data(symbol: str, panel: dict,
                              lookback_days: int = 90) -> dict:
    """Last 90d of price + implied FV + residual + entry/target/stop markers.

    Returns a dict the renderer can pass straight to plotly:
      {
        "dates":       list[str],         # ISO
        "prices":      list[float],       # SR3 close
        "fv_prices":   list[float],       # PCA-implied FV in price units
        "residuals_bp":list[float],       # residual in bp
        "entry_price": float | None,
        "target_price":float | None,
        "stop_price":  float | None,
        "ohlc":        None,              # placeholder — only close used today
      }
    """
    close = panel.get("outright_close_panel", pd.DataFrame())
    if close is None or close.empty or symbol not in close.columns:
        return {"dates": [], "prices": [], "fv_prices": [], "residuals_bp": []}
    s = close[symbol].dropna().tail(lookback_days + 30)
    if s.empty:
        return {"dates": [], "prices": [], "fv_prices": [], "residuals_bp": []}
    s = s.tail(lookback_days)

    # FV reconstruction:
    # The engine's `pca_fit_static` is fit on CMC yield CHANGES (lib.pca:506
    # `delta = cmc_panel.diff().dropna()`). PCA-in-change-space gives loadings
    # and per-day PC scores that describe how the curve MOVES day-to-day —
    # not where it SITS. Reconstructing a LEVEL via
    #   fv_y = mean_inst + Σ_k PC_k(t)·load_k
    # is meaningless because `mean_inst` and the loadings are in change units
    # (bp/day). That gives `fv_y ≈ 0`, then `fv_price = 100 − 0/100 ≈ 100`
    # which is what was rendering on the chart.
    #
    # CORRECT FV (level-space): would require fitting a SEPARATE level-space
    # PCA on yields (not changes). That's statistically problematic (yields are
    # non-stationary) and is a different model from what the engine actually
    # uses for trade generation.
    #
    # For the chart, the only honest FV is TODAY's FV which we can derive
    # directly: today's residual_bp = idea.entry_bp (from the change-space
    # accumulation). today's FV_yield = today's actual_yield − residual_bp.
    # today's FV_price = 100 − FV_yield/100.
    #
    # So instead of plotting a misleading FV time series, we:
    #   1) Plot only the actual close price (the line)
    #   2) Let the renderer add a single horizontal FV line at today's FV_price
    #      using the SR3 target_price (= FV_price by construction)
    return {
        "dates": [d.strftime("%Y-%m-%d") for d in s.index],
        "prices": [float(v) for v in s.values],
        "fv_prices": [None] * len(s),    # intentionally unused — see comment above
        "residuals_bp": [None] * len(s),    # populated by build_residual_series instead
    }


def build_residual_series(symbol: str, panel: dict,
                              lookback_days: int = 90) -> dict:
    """Build the CORRECT residual time series for one contract over `lookback_days`.

    Replicates lib.pca.per_traded_outright_residuals' inner logic but keeps the
    full residual series (instead of just today's value). This is what the
    residual chart panel actually plots.

    Returns: { "dates": list, "residuals_bp": list, "mean": float, "sigma": float }
    """
    fit = panel.get("pca_fit_static")
    close = panel.get("outright_close_panel", pd.DataFrame())
    pc_panel = panel.get("pc_panel", pd.DataFrame())
    if (fit is None or close is None or close.empty
            or symbol not in close.columns
            or pc_panel is None or pc_panel.empty):
        return {"dates": [], "residuals_bp": [], "mean": None, "sigma": None}
    asof = panel.get("asof", date.today())
    try:
        from lib.pca import _instrument_loadings, _instrument_feature_mean
        load_inst = _instrument_loadings(symbol, "outright", fit, asof)
        mean_inst = _instrument_feature_mean(symbol, "outright", fit, asof)
        if load_inst is None or mean_inst is None:
            return {"dates": [], "residuals_bp": [], "mean": None, "sigma": None}
        # Build yield-change series for this symbol
        s_price = close[symbol].dropna()
        s_yield_bp = (100.0 - s_price) * 100.0
        s_dyield = s_yield_bp.diff()
        # Subset to the lookback window
        sub_idx = s_dyield.tail(lookback_days).index
        residuals = []
        dates_out = []
        for dt in sub_idx:
            if dt not in pc_panel.index or pd.isna(s_dyield.loc[dt]):
                continue
            pcs = pc_panel.loc[dt]
            recon = float(mean_inst)
            for k in range(min(3, len(load_inst))):
                col = f"PC{k+1}"
                if col in pcs.index and np.isfinite(pcs[col]):
                    recon += pcs[col] * load_inst[k]
            res = float(s_dyield.loc[dt]) - recon
            residuals.append(res)
            dates_out.append(dt.strftime("%Y-%m-%d"))
        if not residuals:
            return {"dates": [], "residuals_bp": [], "mean": None, "sigma": None}
        arr = np.array(residuals)
        return {
            "dates": dates_out,
            "residuals_bp": residuals,
            "mean": float(arr.mean()),
            "sigma": float(arr.std(ddof=1)) if len(arr) > 1 else None,
        }
    except Exception:
        return {"dates": [], "residuals_bp": [], "mean": None, "sigma": None}


# =============================================================================
# Dynamic per-trade narrative — the centrepiece of the user request
# =============================================================================
def build_dynamic_narrative(idea, sr3, factors: list, panel: dict) -> str:
    """Generate a trade-specific multi-paragraph narrative that walks the user
    through how the engine arrived at this exact recommendation.

    The narrative is structured into 5 numbered paragraphs:
      1. WHAT WE OBSERVED    — the specific raw signal that triggered this idea
      2. WHY IT'S TRADEABLE  — statistical tests that say "this is real"
      3. WHY THIS DIRECTION  — sign chain: residual sign → yield action → contract action
      4. THE TRADE           — actual contract prices + sizing + R:R
      5. WHAT KILLS IT       — risk factors, caveats, watch items

    Each paragraph cites the actual numbers for THIS trade — no generic copy.
    """
    sym = idea.legs[0].symbol if idea.legs else "?"
    z = idea.z_score
    entry_bp = idea.entry_bp
    hl = idea.half_life_d
    n = idea.eff_n
    gate = idea.gate_quality
    source = idea.primary_source
    n_conf = max(1, idea.n_confirming_sources)
    structure = idea.structure_type
    direction = idea.direction
    is_outright = structure == "outright" and len(idea.legs) == 1

    # Resolve mode + lookback for narrative phrasing
    mode = (panel.get("mode") or "positional")
    mode_p = panel.get("mode_params") or {}
    res_lb = int(mode_p.get("residual_lookback", 252 if mode == "positional" else 60))
    mode_label = {"intraday": "intraday", "swing": "swing", "positional": "positional"}.get(mode, "swing")

    # ---- paragraph 1: WHAT WE OBSERVED ----
    p1_parts = []
    if is_outright:
        if z is not None and entry_bp is not None:
            rich_cheap = "rich (yield above fair value)" if entry_bp > 0 else "cheap (yield below fair value)"
            pct_str = (str(int(round((1.0 - 0.5 * (1.0 + _erf_approx(abs(z)/1.4142))) * 100))) + '%'
                        if z else 'tail')
            p1_parts.append(
                f"Today's <b>{sym}</b> has a <b>cumulative deviation from PCA fair value</b> "
                f"of <b>{entry_bp:+.2f} bp</b> — {rich_cheap}. "
                f"This level-residual is built by integrating daily change-residuals and "
                f"detrending; standardized over its trailing <b>{res_lb}-day</b> distribution "
                f"this is <b>{z:+.2f}σ</b> "
                f"(top <b>{pct_str}</b> under a normal assumption). "
                f"Mode: <b>{mode_label}</b>."
            )
        elif entry_bp is not None:
            p1_parts.append(
                f"<b>{sym}</b> current cumulative deviation from PCA fair value: "
                f"<b>{entry_bp:+.2f} bp</b>."
            )
    else:
        leg_names = [l.symbol for l in idea.legs]
        struct_word = {"fly": "butterfly", "spread": "spread",
                         "pack": "pack", "basket": "basket"}.get(structure, structure)
        sum_abs_w = sum(abs(l.weight_pv01) for l in idea.legs) or 1.0
        norm_bp = (entry_bp or 0) / sum_abs_w
        if z is not None:
            p1_parts.append(
                f"This is a <b>{struct_word}</b> on legs <b>{' · '.join(leg_names)}</b> "
                f"whose PV01-weighted <b>cumulative deviation from PCA fair value</b> is "
                f"<b>{norm_bp:+.2f} bp per unit-PV01</b> "
                f"({z:+.2f}σ vs its own {res_lb}-day distribution). Mode: <b>{mode_label}</b>."
            )
        else:
            p1_parts.append(
                f"This is a <b>{struct_word}</b> on legs <b>{' · '.join(leg_names)}</b> "
                f"with package cumulative deviation {norm_bp:+.2f} bp per unit-PV01."
            )

    # Source-specific description (market-aware central bank name)
    _cb_for_blurb = (panel.get("market") or {}).get("central_bank", "Fed")
    source_blurbs = {
        "PC3-fly":     "It came from the <b>PC3 fly enumerator</b> (source #1): a butterfly with weights solved to load purely on the PC3 curvature factor, zeroing PC1 (level) and PC2 (slope).",
        "PC2-spread":  "It came from the <b>PC2 spread enumerator</b> (source #2): a calendar spread weighted to load on PC2 (slope) with zero level and curvature drift.",
        "PC1-basket":  "It came from the <b>PC1 basket enumerator</b> (source #3): a directional basket isolated to the level factor, hedging slope and curvature.",
        "anchor":      "It came from the <b>12M − 24M anchor-slope mean-reversion overlay</b> (source #4).",
        "front-PCA":   "It came from the <b>front-segment PCA</b> (source #10) — a multi-resolution PCA fitted to just the front of the curve.",
        "belly-PCA":   "It came from the <b>belly-segment PCA</b> (source #11).",
        "back-PCA":    "It came from the <b>back-segment PCA</b> (source #12).",
        "outright-fade":"It came from the <b>per-contract residual-fade scanner</b> (source #13): each contract is watched for individual stretching from its model-implied yield.",
        "spread-fade": "It came from the <b>spread residual-fade scanner</b> (source #14).",
        "fly-arb":     "It came from the <b>fly arbitrage scanner</b> (source #15).",
        "pack-rv":     "It came from the <b>pack-RV scanner</b> (source #16).",
        "analog-FV":   "It came from the <b>Mahalanobis k-NN analog FV engine</b> (source #17): today's k=30 most-similar historical days were averaged to estimate where this contract should be — and the answer disagrees with where it is now.",
        "path-FV":     f"It came from the <b>policy-path conditional FV</b> (source #18): bucketed by the implied 4-quarter {_cb_for_blurb}-path, today's bucket has a historical residual avg that disagrees with the current price.",
    }
    if source in source_blurbs:
        p1_parts.append(source_blurbs[source])

    # Detail on analog FV neighbors (when this trade has analog data)
    if idea.analog_fv_bp is not None and np.isfinite(idea.analog_fv_bp):
        agree = (idea.analog_fv_z is not None and z is not None
                   and np.sign(idea.analog_fv_z) == np.sign(z)
                   and abs(idea.analog_fv_z) > 0.5)
        agree_txt = ("<b style='color:var(--green);'>agrees</b>"
                       if agree else
                       "<b style='color:var(--amber);'>disagrees</b>"
                       if idea.analog_fv_z is not None and z is not None
                            and abs(idea.analog_fv_z) > 0.5
                       else "<b>neutral</b>")
        p1_parts.append(
            f"<b>Analog FV cross-check:</b> the k-NN engine (eff_n="
            f"{idea.analog_fv_eff_n or '?'} good neighbors) puts FV at "
            f"<b>{idea.analog_fv_bp:+.2f} bp</b> "
            f"(z = {idea.analog_fv_z:+.2f}σ). This {agree_txt} with the raw PCA-residual sign."
        )

    # Detail on path-conditional FV
    if idea.path_fv_bp is not None and np.isfinite(idea.path_fv_bp):
        p1_parts.append(
            f"<b>Path-conditional FV:</b> today's policy-path bucket suggests FV = "
            f"<b>{idea.path_fv_bp:+.2f} bp</b>"
            + (f" (z = {idea.path_fv_z:+.2f}σ)" if idea.path_fv_z is not None else "")
            + "."
        )

    # ---- paragraph 2: WHY IT'S TRADEABLE (statistical) ----
    p2_parts = []
    import numpy as _np
    hl_ok = hl is not None and _np.isfinite(hl) and hl > 0
    # Mode-aware sweet-spot band + clipped hold horizon
    sw_lo, sw_hi = mode_p.get("sweet_spot_full", (3.0, 30.0))
    hold_floor = float(mode_p.get("hold_floor", 3))
    hold_cap = float(mode_p.get("hold_cap", 30))
    hold_mult = float(mode_p.get("hold_mult", 1.5))
    if hl_ok and sw_lo <= hl <= sw_hi:
        raw_hold = hold_mult * hl
        clipped_hold = max(hold_floor, min(raw_hold, hold_cap))
        p2_parts.append(
            f"The Ornstein-Uhlenbeck fit on the level residual gives a half-life of "
            f"<b>{hl:.1f} trading days</b> — inside the <b>{mode_label}</b>-mode "
            f"sweet-spot band [{sw_lo:.0f}, {sw_hi:.0f}]d. "
            f"Expected mean-revert over <b>~{raw_hold:.0f}d</b>; "
            f"recommended hold window for this mode: "
            f"<b>{int(round(clipped_hold))}d</b> (floor {int(hold_floor)}d, cap {int(hold_cap)}d)."
        )
    elif hl_ok:
        zone = ("slower than" if hl > sw_hi else "faster than")
        raw_hold = hold_mult * hl
        clipped_hold = max(hold_floor, min(raw_hold, hold_cap))
        p2_parts.append(
            f"OU half-life is <b>{hl:.1f}d</b> ({zone} the {mode_label}-mode "
            f"sweet-spot [{sw_lo:.0f}, {sw_hi:.0f}]d — expect a "
            f"{'slower' if hl > sw_hi else 'noisier'} fade). "
            f"Clipped hold window: <b>{int(round(clipped_hold))}d</b>."
        )
    elif hl is None or not _np.isfinite(hl):
        p2_parts.append(
            "Ornstein-Uhlenbeck half-life could not be fitted on this level residual "
            "(insufficient signal or non-reverting series). Treat sizing conservatively."
        )
    if gate == "clean":
        p2_parts.append(
            f"The triple-stationarity gate (ADF + KPSS + Lo-MacKinlay variance "
            f"ratio) returned <b style='color:var(--green);'>CLEAN</b> — all three "
            f"independent tests agree the residual is mean-reverting. This is the "
            f"engine's highest statistical-confidence tier."
        )
    elif gate == "drift":
        p2_parts.append(
            f"Triple-gate verdict is <b style='color:var(--amber);'>DRIFT</b> — "
            f"ADF passes but KPSS rejects, meaning there's a slow drift component "
            f"on top of the mean-reverting signal. Trade smaller than usual."
        )
    elif gate == "non_stationary":
        p2_parts.append(
            f"Triple-gate verdict is <b style='color:var(--amber);'>NON-STATIONARY</b> — "
            f"the strict tests didn't fully confirm. The signal is still surfaced "
            f"but with reduced confidence."
        )
    if n is not None:
        if n >= 100:
            p2_parts.append(
                f"Sample size <b>{n} observations</b> — statistically robust; "
                f"every test stat has high power."
            )
        elif n >= 30:
            p2_parts.append(
                f"Sample size <b>{n} observations</b> — above the engine floor "
                f"of 30 but conviction is scaled by n/100."
            )
        else:
            p2_parts.append(
                f"Sample size <b>{n} observations</b> — <b style='color:var(--amber);'>"
                f"below the eff_n floor</b>, so the trade is flagged low_n and "
                f"conviction is capped."
            )
    if n_conf > 1:
        srcs_listed = "<br>".join(
            f"  • <b>{s}</b>" for s in idea.sources
        )
        p2_parts.append(
            f"<b>{n_conf} of the 19 trade generators independently emitted this same "
            f"leg + direction.</b> The confirming sources are:<br>"
            f"{srcs_listed}<br>"
            f"Independent analyses converging on the same leg/direction is the "
            f"strongest single conviction boost (log-scaled bonus +0.05 × ln(1+n)/ln(6))."
        )

    # HMM regime label (specific name, not just "stable")
    rs = panel.get("regime_stack", {}) if panel else {}
    hmm = rs.get("hmm_fit") if rs else None
    if hmm and hmm.dominant_confidence is not None and len(hmm.dominant_confidence) > 0:
        conf = float(hmm.dominant_confidence[-1])
        regime_id = None
        if hasattr(hmm, "dominant_state") and hmm.dominant_state is not None:
            try:
                regime_id = int(hmm.dominant_state[-1])
            except Exception:
                regime_id = None
        regime_label = f"Regime #{regime_id}" if regime_id is not None else "current regime"
        conf_color = "var(--green)" if conf >= 0.8 else "var(--amber)" if conf >= 0.6 else "var(--red)"
        p2_parts.append(
            f"<b>Regime context</b>: today is in <b>{regime_label}</b> with HMM dominant "
            f"posterior <b style='color:{conf_color};'>{conf*100:.0f}%</b> "
            f"({'stable' if conf >= 0.8 else 'moderate' if conf >= 0.6 else 'transitioning'}). "
            f"Analog FV relies on historical days in the SAME regime — high stability means analog FV is trustworthy."
        )

    # Cycle phase
    if idea.cycle_phase:
        align_color = ("var(--green)" if idea.cycle_alignment == "favoured"
                         else "var(--red)" if idea.cycle_alignment == "counter"
                         else "var(--text-body)")
        p2_parts.append(
            f"<b>Cycle phase</b>: <b>{idea.cycle_phase}</b>. This trade's direction is "
            f"<b style='color:{align_color};'>{idea.cycle_alignment}</b> the historically-favoured "
            f"side for this phase."
        )

    # ---- paragraph 3: WHY THIS DIRECTION ----
    p3_parts = []
    if is_outright and entry_bp is not None and sr3 is not None and sr3.is_outright:
        if entry_bp > 0:
            p3_parts.append(
                f"The residual sign is <b>positive</b>, meaning the contract's "
                f"implied yield is <b>above</b> its model fair value. Mean reversion "
                f"says the yield should fall back. <br><br>"
                f"For an SR3 outright, falling yield = <b>rising price</b> "
                f"(SR3 price = 100 − yield/100). Therefore the correct trade is "
                f"to <b style='color:var(--green);'>BUY</b> the futures and profit "
                f"as the price climbs back toward fair value."
            )
        else:
            p3_parts.append(
                f"The residual sign is <b>negative</b>, meaning the contract's "
                f"implied yield is <b>below</b> its model fair value. Mean reversion "
                f"says the yield should rise back. <br><br>"
                f"For an SR3 outright, rising yield = <b>falling price</b>. "
                f"Therefore the correct trade is to <b style='color:var(--red);'>SELL</b> "
                f"the futures and profit as the price drops back toward fair value."
            )
    elif not is_outright:
        leg_actions = []
        for l in idea.legs:
            act = "BUY" if l.weight_pv01 > 0 else "SELL"
            leg_actions.append(f"<b>{act} {l.symbol}</b> (weight {l.weight_pv01:+.2f})")
        p3_parts.append(
            f"This is a relative-value structure — the bet is that the "
            f"combination of legs converges back to its model fair value, "
            f"regardless of where outright rates go. <br><br>"
            f"Execution: {' / '.join(leg_actions)}."
        )

    # ---- paragraph 4: THE TRADE — with full derivation math ----
    p4_parts = []
    if sr3 is not None and sr3.is_outright and sr3.entry_price is not None:
        rr_str = f"{sr3.risk_reward:.2f}:1" if sr3.risk_reward else "—"
        # DERIVATION MATH — show the user exactly how the prices were derived
        derivation = (
            f"<div style='font-family:JetBrains Mono, monospace; font-size:0.72rem; "
            f"background:rgba(255,255,255,0.04); padding:0.4rem 0.6rem; "
            f"border-radius:4px; margin: 0.4rem 0; line-height:1.7; color:var(--text-body);'>"
            f"<span style='color:var(--text-dim);'># SR3 price = 100 − implied_yield/100  ·  1 bp yield = 0.01 price  ·  $25/0.01</span><br>"
            f"<span style='color:var(--text-dim);'>entry_yield  </span>= (100 − {sr3.entry_price:.4f}) × 100 = <b>{sr3.entry_yield_bp:.2f} bp</b><br>"
            f"<span style='color:var(--text-dim);'>residual     </span>= entry_yield − PCA_FV_yield = <b>{entry_bp:+.2f} bp</b><br>"
            f"<span style='color:var(--text-dim);'>target_yield </span>= entry_yield − residual = <b>{sr3.target_yield_bp:.2f} bp</b> (when residual = 0)<br>"
            f"<span style='color:var(--text-dim);'>target_price </span>= 100 − target_yield/100 = <b>{sr3.target_price:.4f}</b> "
            f"(rounded to {SR3_TICK} tick)<br>"
            f"<span style='color:var(--text-dim);'>stop_yield   </span>= entry_yield + 2.5 × residual = <b>{sr3.stop_yield_bp:.2f} bp</b><br>"
            f"<span style='color:var(--text-dim);'>stop_price   </span>= 100 − stop_yield/100 = <b>{sr3.stop_price:.4f}</b><br>"
            f"<span style='color:var(--text-dim);'>P&L/c        </span>= |target − entry| × $2,500/price = <b>${sr3.pnl_per_contract_dollar:.0f}</b><br>"
            f"<span style='color:var(--text-dim);'>risk/c       </span>= |stop − entry| × $2,500/price = <b>${sr3.risk_per_contract_dollar:.0f}</b>"
            f"</div>"
        )
        p4_parts.append(
            f"<b>Execute:</b> <b style='color:{'var(--green)' if sr3.contract_action == 'BUY' else 'var(--red)'};'>"
            f"{sr3.contract_action}</b> {sym} at <b>{sr3.entry_price:.4f}</b> · "
            f"target <b style='color:var(--green);'>{sr3.target_price:.4f}</b> · "
            f"stop <b style='color:var(--red);'>{sr3.stop_price:.4f}</b> · "
            f"R:R <b>{rr_str}</b>."
            + derivation
        )
        if (idea.expected_revert_d is not None
                and np.isfinite(idea.expected_revert_d)
                and idea.expected_revert_d > 0):
            if hl_ok:
                p4_parts.append(
                    f"Expected hold time: ~<b>{idea.expected_revert_d:.0f} trading days</b> "
                    f"(1.5× the OU half-life of {hl:.1f}d)."
                )
            else:
                p4_parts.append(
                    f"Expected hold time: ~<b>{idea.expected_revert_d:.0f} trading days</b>."
                )
        # Slippage
        if idea.slippage_estimate_bp is not None and idea.expected_pnl_bp:
            slip_pct = (idea.slippage_estimate_bp / max(idea.expected_pnl_bp, 0.01)) * 100
            slip_color = "var(--green)" if slip_pct < 20 else "var(--amber)" if slip_pct < 40 else "var(--red)"
            p4_parts.append(
                f"Estimated round-trip slippage: <b style='color:{slip_color};'>"
                f"{idea.slippage_estimate_bp:.2f} bp</b> "
                f"({slip_pct:.0f}% of expected P&L)."
            )
    elif sr3 is not None and not sr3.is_outright:
        # Multi-leg with per-leg actual prices
        leg_rows = []
        for lp in sr3.per_leg_prices:
            if lp.get("entry_price") is not None and lp.get("target_price") is not None:
                act_color = "var(--green)" if lp["action"] == "BUY" else "var(--red)"
                leg_rows.append(
                    f"<tr><td style='color:var(--text-body); padding-right:1rem;'>{lp['symbol']}</td>"
                    f"<td style='color:{act_color}; padding-right:1rem;'><b>{lp['action']}</b></td>"
                    f"<td style='color:var(--text-heading); padding-right:1rem; text-align:right;'>"
                    f"{lp['entry_price']:.4f}</td>"
                    f"<td style='color:var(--green); padding-right:1rem; text-align:right;'>"
                    f"{lp['target_price']:.4f}</td>"
                    f"<td style='color:var(--red); padding-right:1rem; text-align:right;'>"
                    f"{lp['stop_price']:.4f}</td>"
                    f"<td style='color:var(--text-muted); text-align:right;'>"
                    f"${lp.get('pnl_per_contract_dollar', 0):.0f} / ${lp.get('risk_per_contract_dollar', 0):.0f}</td>"
                    f"</tr>"
                )
        p4_parts.append(
            f"<b>Execute all legs simultaneously</b> at the prices shown.<br>"
            f"<table style='font-family:JetBrains Mono, monospace; font-size:0.72rem; "
            f"margin:0.3rem 0; border-collapse:collapse;'>"
            f"<thead><tr style='color:var(--text-dim); font-size:0.65rem;'>"
            f"<th style='text-align:left; padding-right:1rem;'>LEG</th>"
            f"<th style='text-align:left; padding-right:1rem;'>ACTION</th>"
            f"<th style='text-align:right; padding-right:1rem;'>ENTRY</th>"
            f"<th style='text-align:right; padding-right:1rem;'>TARGET</th>"
            f"<th style='text-align:right; padding-right:1rem;'>STOP</th>"
            f"<th style='text-align:right;'>P&L / RISK</th>"
            f"</tr></thead><tbody>"
            + "".join(leg_rows) +
            f"</tbody></table>"
            f"<b>Package totals:</b> P&L per set <b style='color:var(--green);'>"
            f"${sr3.pnl_per_contract_dollar:.0f}</b>, risk per set "
            f"<b style='color:var(--red);'>${sr3.risk_per_contract_dollar:.0f}</b>, "
            f"R:R <b>{(f'{sr3.risk_reward:.2f}:1' if sr3.risk_reward else '—')}</b>."
        )

    # ---- paragraph 4b: CONVICTION BREAKDOWN (numerical) ----
    if idea.conviction_breakdown:
        cb_rows = []
        sorted_cb = sorted(idea.conviction_breakdown, key=lambda x: -abs(x[1]))
        for component, value in sorted_cb[:8]:    # top 8 contributors
            color = ("var(--text-dim)" if abs(value) < 0.001
                      else "var(--green)" if value > 0
                      else "var(--red)")
            cb_rows.append(
                f"<tr>"
                f"<td style='color:var(--text-muted); padding-right:1rem;'>{component}</td>"
                f"<td style='color:{color}; text-align:right; font-family:JetBrains Mono, monospace;'>"
                f"{value:+.3f}</td></tr>"
            )
        p4_parts.append(
            f"<br><b>Conviction = {idea.conviction:.3f}</b> built from these top components:"
            f"<table style='font-size:0.72rem; margin-top:0.3rem; border-collapse:collapse;'>"
            + "".join(cb_rows) +
            f"</table>"
        )

    # ---- paragraph 5: WHAT KILLS IT (caveats + risk panel) ----
    p5_parts = []
    caveats_from_factors = [f for f in factors if f.tier == "caveat"]
    for f in caveats_from_factors[:3]:
        p5_parts.append(f"• <b>{f.headline}</b> — {f.detail}")
    # Convexity warning
    if idea.convexity_warning:
        p5_parts.append(
            "• <b>Convexity caveat</b> — this is a back-end contract where the "
            "SR3-vs-OIS-forward gap exceeds 3 bp. SR3-implied yields need to be "
            "OIS-adjusted before benchmarking; the residual signal here may be "
            "partly mechanical convexity."
        )
    # Slippage
    if idea.slippage_estimate_bp is not None and idea.expected_pnl_bp:
        slip_pct = (idea.slippage_estimate_bp / max(idea.expected_pnl_bp, 0.01)) * 100
        if slip_pct > 30:
            p5_parts.append(
                f"• <b>High slippage relative to edge</b> — round-trip slip "
                f"{idea.slippage_estimate_bp:.2f} bp = {slip_pct:.0f}% of "
                f"expected P&L. Net edge is thin; use limit orders."
            )
    # Engine-state caveats from panel
    if panel:
        ca = panel.get("cross_asset_analysis")
        if ca:
            vol = (ca.vol_regime or {}).get("regime")
            if vol in ("stressed", "crisis"):
                p5_parts.append(
                    f"• Vol regime is <b>{vol}</b> — analog FV bands widen "
                    f"×{ca.vol_regime.get('band_widening_factor', 1.0):.1f}. "
                    f"Mean-reversion edge less reliable in this environment."
                )
        # Print quality alerts today
        asof = panel.get("asof")
        if asof:
            try:
                pq = panel.get("print_quality_alerts", []) or []
                if any(pd.Timestamp(d).date() == asof for d in pq):
                    p5_parts.append(
                        "• <b>Print-quality alert today</b> — at least one contract "
                        "in the surface had a suspicious print on this date. "
                        "PCA basis may have been contaminated."
                    )
            except Exception:
                pass
    if not p5_parts:
        p5_parts.append(
            "• No major caveats flagged. Standard risk: residual fails to revert "
            "and stops out at 2.5× wider; or a fresh data print / event causes "
            "a regime shift that invalidates the historical statistics."
        )

    return _format_narrative(p1_parts, p2_parts, p3_parts, p4_parts, p5_parts)


def _format_narrative(p1, p2, p3, p4, p5) -> str:
    """Assemble the 5-paragraph HTML narrative."""
    def block(num: str, label: str, parts: list) -> str:
        if not parts:
            return ""
        body = " ".join(parts)
        return (
            f"<div style='margin-bottom:0.75rem;'>"
            f"<div style='font-size:0.68rem; color:var(--accent); "
            f"text-transform:uppercase; letter-spacing:0.08em; font-weight:600; "
            f"margin-bottom:0.25rem;'>{num} · {label}</div>"
            f"<div style='font-size:0.78rem; color:var(--text-body); "
            f"line-height:1.65;'>{body}</div></div>"
        )
    return (
        block("①", "What we observed", p1) +
        block("②", "Why it's tradeable", p2) +
        block("③", "Why this direction", p3) +
        block("④", "The trade", p4) +
        block("⑤", "What kills it", p5)
    )


def _erf_approx(x: float) -> float:
    """Abramowitz-Stegun erf approximation. For converting |z| → tail percentile."""
    import math
    sign = 1 if x >= 0 else -1
    x = abs(x)
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return sign * y

