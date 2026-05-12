"""TMIA-curated setup detectors for the SRA → Analysis → Technicals subtab.

Module structure:
  - ``base``                  Setup result dataclass + helpers
  - ``registry``              Catalog of all setup definitions (id → metadata)
  - ``interpretation_guide``  Plain-English guide for tooltips (mirrors lib.mean_reversion.get_interpretation_guide)
  - ``trend``                 A1, A2, A3, A4, A5, A6, A8, A10, A11, A12a, A12b, A15
  - ``mean_reversion``        B1, B3, B5, B6, B10, B11, B13
  - ``stir``                  C3, C4, C5, C8 (12/24/36 variants), C9a, C9b
  - ``composite``             TREND/MR/FINAL composites per scope (outright/spread/fly)
  - ``scan``                  scan_universe(strategy, tenor, asof) — cached entry point
  - ``track_record``          60d lightweight track record per setup

Locked conventions:
  · history excludes today (``< asof_date``)
  · ddof=1 for sample std
  · catalog-aware bp scaling via lib.contract_units.bp_multipliers_for
  · all detectors return SetupResult — never raise on bad data, set ``error`` field instead
"""
from lib.setups.base import (
    SetupResult,
    fmt_inputs_str,
    safe_float,
    state_from_conditions_met,
)

__all__ = ["SetupResult", "fmt_inputs_str", "safe_float", "state_from_conditions_met"]
