"""Market registry — central configuration for all STIRS markets.

Every market-specific parameter (currency, tick value, central bank, Bloomberg
data sources, PCA history cutover, symbol prefix) lives here. Tabs and library
modules read from MARKETS[code] rather than hardcoding US/SRA conventions.

Supported markets:
  SRA — US SOFR (3M, CME)        ← default / canonical
  ER  — Euribor 3M (ICE)
  FSR — SARON 3M (Eurex)
  FER — Eurozone €STR equivalent
  SON — SONIA 3M (ICE/LSE)
  YBA — Australian 90-day Bank Bills (ASX)
  CRA — CORRA (Canadian Overnight Repo Rate Average, MX)

Each entry carries:
  - region: page bucket ("us" | "eurozone" | "uk" | "australia" | "canada")
  - description: human-readable
  - currency, tick_bp, usd_per_bp, dollar_per_bp_local
  - central_bank: ECB/BoE/BoC/RBA/SNB/Fed
  - central_bank_module: which lib/cb_*.py module holds the calendar
  - reference_rate_ticker: BBG ticker for the overnight policy rate
  - reference_rate_upper/lower: tickers for the policy corridor
  - rate_history_cutover: date before which we exclude (LIBOR transitions, etc.)
  - bbg_economy_inventory_module: which lib/*_fundamentals_inventory.py
  - mode_overrides: dict for any market-specific mode-param tweaks
  - pack_names: ordered pack labels (most are Whites/Reds/Greens/Blues)
"""
from __future__ import annotations

from datetime import date


# Generic month codes — same across all CME-style products
MONTH_CODE_TO_NUM = {"F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
                      "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12}
NUM_TO_MONTH_CODE = {v: k for k, v in MONTH_CODE_TO_NUM.items()}
QUARTERLY_MONTH_CODES = {"H", "M", "U", "Z"}


# Short human label for each market's overnight reference rate — used in
# Curve-tab band/line legends, captions, and tooltips.
OVERNIGHT_RATE_LABELS = {
    "SRA": "SOFR",
    "ER":  "Euribor 3M",
    "FSR": "SARON",
    "FER": "€STR",
    "SON": "SONIA",
    "YBA": "BBSW 3M",
    "CRA": "CORRA",
}


def overnight_rate_label(base_product: str) -> str:
    """Short label for the market's overnight rate. Falls back to base_product."""
    return OVERNIGHT_RATE_LABELS.get(base_product, base_product)


# Settlement convention per market:
#   "compounded_overnight" — contract settles to compounded daily overnight rate
#                            over the 3M reference period (Fed SR3 style).
#                            decompose_implied_rates works directly.
#   "forward_3m_fixing"    — contract settles to a single forward 3M IBOR fixing
#                            on the last trading day (Euribor / Bank Bill style).
#                            The decomposition still works mathematically but the
#                            interpretation is "implied 3M forward fixing path"
#                            rather than "implied policy rate path"; deltas
#                            between meetings still approximate policy moves
#                            (assuming the credit/term basis is roughly stable).
SETTLEMENT_CONVENTION = {
    "SRA": "compounded_overnight",
    "ER":  "forward_3m_fixing",      # 3M Euribor fixing
    "FSR": "compounded_overnight",   # SARON 3M compounded
    "FER": "compounded_overnight",   # €STR 3M compounded
    "SON": "compounded_overnight",   # SONIA 3M compounded
    "YBA": "forward_3m_fixing",      # BBSW 3M fixing
    "CRA": "compounded_overnight",   # CORRA 3M compounded
}


# Map base_product → cb_code used by lib.central_banks.load_meetings()
CB_CODE_BY_PRODUCT = {
    "SRA": "fed",
    "ER":  "ecb",
    "FSR": "snb",
    "FER": "ecb",
    "SON": "boe",
    "YBA": "rba",
    "CRA": "boc",
}


# Typical CB step size in bp (for hike/cut/hold probability mapping).
# All major CBs default to 25bp; SNB has used 50bp in 2022-23 but 25bp is
# still the standard probability unit.
CB_STEP_SIZE_BP = {
    "SRA": 25.0,
    "ER":  25.0,
    "FSR": 25.0,
    "FER": 25.0,
    "SON": 25.0,
    "YBA": 25.0,
    "CRA": 25.0,
}


def cb_code_for(base_product: str) -> str:
    return CB_CODE_BY_PRODUCT.get(base_product, "fed")


def cb_step_bp(base_product: str) -> float:
    return CB_STEP_SIZE_BP.get(base_product, 25.0)


def settlement_convention(base_product: str) -> str:
    return SETTLEMENT_CONVENTION.get(base_product, "compounded_overnight")


MARKETS = {
    # =========================================================================
    # SRA — US SOFR 3M (CME) — canonical reference, all SRA code paths
    # =========================================================================
    "SRA": {
        "code": "SRA",
        "region": "us",
        "description": "CME 3-Month SOFR Futures (SR3) — USD",
        "currency": "USD",
        "currency_symbol": "$",
        "tick_bp": 0.25,
        "usd_per_bp": 25.0,
        "dollar_per_bp_local": 25.0,
        "contract_size_notional": 1_000_000.0,
        "central_bank": "Fed",
        "central_bank_module": "lib.fomc",
        "central_bank_decision_fn": "load_fomc_meetings",
        "reference_rate_ticker": "SOFRRATE_Index",
        "reference_rate_upper": "FDTR_Index",
        "reference_rate_lower": "FDTRFTRL_Index",
        "reference_rate_lookup_dir": "rates_drivers",
        "rate_history_cutover": date(2023, 8, 1),    # post-LIBOR transition
        "bbg_economy_inventory_module": "lib.us_fundamentals_inventory",
        "pack_names": ["Whites", "Reds", "Greens", "Blues", "Golds", "Purples", "Browns"],
        "pack_colors": ["#f0f4fa", "#f87171", "#4ade80", "#60a5fa", "#e8b75d", "#a78bfa", "#a16207"],
        "default_front_end": 6,
        "default_mid_end": 14,
        "anchor_tenor_short_m": 12,
        "anchor_tenor_long_m": 24,
        "mode_overrides": {},
        "yield_convention": "100_minus_price",   # standard CME STIR
    },

    # =========================================================================
    # ER — Euribor 3M (ICE)
    # =========================================================================
    "ER": {
        "code": "ER",
        "region": "eurozone",
        "description": "ICE 3-Month Euribor Futures — EUR",
        "currency": "EUR",
        "currency_symbol": "€",
        "tick_bp": 0.5,
        "usd_per_bp": 27.5,                  # approximate, FX-sensitive
        "dollar_per_bp_local": 25.0,         # €25 per bp in local currency
        "contract_size_notional": 1_000_000.0,
        "central_bank": "ECB",
        "central_bank_module": "lib.cb_ecb",
        "central_bank_decision_fn": "load_ecb_meetings",
        "reference_rate_ticker": "ESTRON_Index",       # €STR
        "reference_rate_upper": "EUMRMRD_Index",       # ECB main refi rate
        "reference_rate_lower": "EUDR1T_Index",        # ECB deposit facility
        "reference_rate_lookup_dir": "rates_drivers",
        "rate_history_cutover": date(2022, 1, 1),
        "bbg_economy_inventory_module": "lib.eurozone_fundamentals_inventory",
        "pack_names": ["Whites", "Reds", "Greens", "Blues", "Golds", "Purples", "Browns"],
        "pack_colors": ["#f0f4fa", "#f87171", "#4ade80", "#60a5fa", "#e8b75d", "#a78bfa", "#a16207"],
        "default_front_end": 6,
        "default_mid_end": 14,
        "anchor_tenor_short_m": 12,
        "anchor_tenor_long_m": 24,
        "mode_overrides": {},
        "yield_convention": "100_minus_price",
    },

    # =========================================================================
    # FSR — SARON 3M (Eurex)
    # =========================================================================
    "FSR": {
        "code": "FSR",
        "region": "eurozone",
        "description": "Eurex 3-Month SARON Futures — CHF",
        "currency": "CHF",
        "currency_symbol": "CHF ",
        "tick_bp": 0.5,
        "usd_per_bp": 28.5,                  # approximate, FX-sensitive
        "dollar_per_bp_local": 25.0,         # CHF25 per bp
        "contract_size_notional": 1_000_000.0,
        "central_bank": "SNB",
        "central_bank_module": "lib.cb_snb",
        "central_bank_decision_fn": "load_snb_meetings",
        "reference_rate_ticker": "SSARON_Index",       # SARON
        "reference_rate_upper": "SZLTPRT_Index",       # SNB policy rate
        "reference_rate_lower": "SZLTPRT_Index",
        "reference_rate_lookup_dir": "rates_drivers",
        "rate_history_cutover": date(2022, 1, 1),
        "bbg_economy_inventory_module": "lib.eurozone_fundamentals_inventory",
        "pack_names": ["Whites", "Reds", "Greens", "Blues", "Golds", "Purples", "Browns"],
        "pack_colors": ["#f0f4fa", "#f87171", "#4ade80", "#60a5fa", "#e8b75d", "#a78bfa", "#a16207"],
        "default_front_end": 6,
        "default_mid_end": 14,
        "anchor_tenor_short_m": 12,
        "anchor_tenor_long_m": 24,
        "mode_overrides": {},
        "yield_convention": "100_minus_price",
    },

    # =========================================================================
    # FER — Eurozone €STR equivalent (CME/Eurex)
    # =========================================================================
    "FER": {
        "code": "FER",
        "region": "eurozone",
        "description": "3-Month FER Futures (€STR equivalent) — EUR",
        "currency": "EUR",
        "currency_symbol": "€",
        "tick_bp": 0.5,
        "usd_per_bp": 27.5,
        "dollar_per_bp_local": 25.0,
        "contract_size_notional": 1_000_000.0,
        "central_bank": "ECB",
        "central_bank_module": "lib.cb_ecb",
        "central_bank_decision_fn": "load_ecb_meetings",
        "reference_rate_ticker": "ESTRON_Index",
        "reference_rate_upper": "EUMRMRD_Index",
        "reference_rate_lower": "EUDR1T_Index",
        "reference_rate_lookup_dir": "rates_drivers",
        "rate_history_cutover": date(2022, 1, 1),
        "bbg_economy_inventory_module": "lib.eurozone_fundamentals_inventory",
        "pack_names": ["Whites", "Reds", "Greens", "Blues", "Golds", "Purples", "Browns"],
        "pack_colors": ["#f0f4fa", "#f87171", "#4ade80", "#60a5fa", "#e8b75d", "#a78bfa", "#a16207"],
        "default_front_end": 6,
        "default_mid_end": 14,
        "anchor_tenor_short_m": 12,
        "anchor_tenor_long_m": 24,
        "mode_overrides": {},
        "yield_convention": "100_minus_price",
    },

    # =========================================================================
    # SON — SONIA 3M (ICE/LSE)
    # =========================================================================
    "SON": {
        "code": "SON",
        "region": "uk",
        "description": "ICE 3-Month SONIA Futures — GBP",
        "currency": "GBP",
        "currency_symbol": "£",
        "tick_bp": 0.25,
        "usd_per_bp": 15.6,                  # approximate
        "dollar_per_bp_local": 12.50,        # £12.50 per bp
        "contract_size_notional": 500_000.0,
        "central_bank": "BoE",
        "central_bank_module": "lib.cb_boe",
        "central_bank_decision_fn": "load_boe_meetings",
        "reference_rate_ticker": "SONIO_Index",        # SONIA
        "reference_rate_upper": "UKBRBASE_Index",      # BoE Bank Rate
        "reference_rate_lower": "UKBRBASE_Index",
        "reference_rate_lookup_dir": "rates_drivers",
        "rate_history_cutover": date(2018, 4, 23),     # SONIA reform
        "bbg_economy_inventory_module": "lib.uk_fundamentals_inventory",
        "pack_names": ["Whites", "Reds", "Greens", "Blues", "Golds", "Purples", "Browns"],
        "pack_colors": ["#f0f4fa", "#f87171", "#4ade80", "#60a5fa", "#e8b75d", "#a78bfa", "#a16207"],
        "default_front_end": 6,
        "default_mid_end": 14,
        "anchor_tenor_short_m": 12,
        "anchor_tenor_long_m": 24,
        "mode_overrides": {},
        "yield_convention": "100_minus_price",
    },

    # =========================================================================
    # YBA — Australian 90-Day Bank Bills (ASX)
    # =========================================================================
    "YBA": {
        "code": "YBA",
        "region": "australia",
        "description": "ASX 90-Day Bank Bill Futures — AUD",
        "currency": "AUD",
        "currency_symbol": "AUD ",
        "tick_bp": 1.0,                      # 0.01 yield = 1bp
        "usd_per_bp": 16.0,                  # approximate
        "dollar_per_bp_local": 24.0,         # AUD24 per bp at 90d
        "contract_size_notional": 1_000_000.0,
        "central_bank": "RBA",
        "central_bank_module": "lib.cb_rba",
        "central_bank_decision_fn": "load_rba_meetings",
        "reference_rate_ticker": "RBACTRD_Index",      # RBA cash rate target
        "reference_rate_upper": "RBACTRD_Index",
        "reference_rate_lower": "RBACTRD_Index",
        "reference_rate_lookup_dir": "rates_drivers",
        "rate_history_cutover": date(2018, 1, 1),
        "bbg_economy_inventory_module": "lib.australia_fundamentals_inventory",
        "pack_names": ["Whites", "Reds", "Greens", "Blues", "Golds", "Purples", "Browns"],
        "pack_colors": ["#f0f4fa", "#f87171", "#4ade80", "#60a5fa", "#e8b75d", "#a78bfa", "#a16207"],
        "default_front_end": 4,              # YBA has shorter curve
        "default_mid_end": 8,
        "mode_overrides": {},
        "yield_convention": "100_minus_price",
    },

    # =========================================================================
    # CRA — CORRA 3M (MX, Bourse de Montréal)
    # =========================================================================
    "CRA": {
        "code": "CRA",
        "region": "canada",
        "description": "MX 3-Month CORRA Futures — CAD",
        "currency": "CAD",
        "currency_symbol": "CAD ",
        "tick_bp": 0.5,
        "usd_per_bp": 18.5,                  # approximate
        "dollar_per_bp_local": 25.0,         # CAD25 per bp
        "contract_size_notional": 1_000_000.0,
        "central_bank": "BoC",
        "central_bank_module": "lib.cb_boc",
        "central_bank_decision_fn": "load_boc_meetings",
        "reference_rate_ticker": "CAONREPO_Index",     # CORRA
        "reference_rate_upper": "CABROVER_Index",      # BoC overnight rate
        "reference_rate_lower": "CABROVER_Index",
        "reference_rate_lookup_dir": "rates_drivers",
        "rate_history_cutover": date(2020, 6, 22),     # CORRA refresh
        "bbg_economy_inventory_module": "lib.canada_fundamentals_inventory",
        "pack_names": ["Whites", "Reds", "Greens", "Blues", "Golds", "Purples", "Browns"],
        "pack_colors": ["#f0f4fa", "#f87171", "#4ade80", "#60a5fa", "#e8b75d", "#a78bfa", "#a16207"],
        "default_front_end": 6,
        "default_mid_end": 14,
        "anchor_tenor_short_m": 12,
        "anchor_tenor_long_m": 24,
        "mode_overrides": {},
        "yield_convention": "100_minus_price",
    },
}


# Per-region multi-market sub-tabs (matches pages → subtabs structure)
REGION_MARKETS = {
    "us":         ["SRA"],
    "eurozone":   ["ER", "FSR", "FER"],
    "uk":         ["SON"],
    "australia":  ["YBA"],
    "canada":     ["CRA"],
}


def get_market(code: str) -> dict:
    """Return market config dict for `code`. Falls back to SRA if unknown."""
    if code in MARKETS:
        return MARKETS[code]
    raise KeyError(f"Unknown market code: {code}. Known: {list(MARKETS.keys())}")


def list_markets() -> list[str]:
    """Return all market codes."""
    return list(MARKETS.keys())


def markets_in_region(region: str) -> list[str]:
    """Return market codes belonging to a region (us / eurozone / uk / australia / canada)."""
    return list(REGION_MARKETS.get(region, []))


def parse_symbol_to_base_product(symbol: str) -> str:
    """Extract the base product code from a symbol (e.g. 'SRAH26' → 'SRA',
    'ERH26' → 'ER'). Returns None if no known prefix matches."""
    if not symbol:
        return None
    # Check 3-char prefixes first (SRA, FSR, FER, SON, YBA, CRA)
    for code in MARKETS:
        if symbol.startswith(code) and len(symbol) >= len(code) + 3:
            return code
    return None


def parse_outright_symbol(symbol: str, base_product: str) -> tuple[int, int]:
    """Parse 'SRAH26' or 'ERH26' → (year_4d, month_num_1to12).

    Symbol format: {base_product}{month_code}{year_2d}. Year < 50 → 20xx,
    else 19xx (handles wrap).
    """
    if not symbol or not symbol.startswith(base_product):
        return (None, None)
    remainder = symbol[len(base_product):]
    if len(remainder) < 3:
        return (None, None)
    month_code = remainder[0].upper()
    if month_code not in MONTH_CODE_TO_NUM:
        return (None, None)
    try:
        year_2d = int(remainder[1:3])
    except ValueError:
        return (None, None)
    year_4d = 2000 + year_2d if year_2d < 50 else 1900 + year_2d
    return (year_4d, MONTH_CODE_TO_NUM[month_code])
