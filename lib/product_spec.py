"""Per-product specification loader (Phase 1).

Reads ``config/product_spec.yaml`` and returns a typed mapping. Used by
every analytics module that needs product-specific conventions
(tick size, DV01, calendar, CMC node list, LIBOR cutover, etc.).

The CMC layer in lib/cmc.py + lib/sra_data.py was previously parameterised
in-line with hard-coded SRA constants. As of Phase 1 those constants come
from this loader, allowing cross-product extension as a config change
rather than a refactor (per plan §4).

Validation happens at load time:
  - YAML must parse
  - Required fields present per product
  - Enabled products must have all calendar/cmc fields
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import yaml


_SPEC_PATH = Path(__file__).resolve().parent.parent / "config" / "product_spec.yaml"


_REQUIRED_FIELDS_ENABLED = {
    "tick_size", "bp_multiplier",
    "quarterly_month_codes", "last_trade_rule",
    "earliest_listing", "naming_convention",
    "trading_calendar",
}


_PRODUCT_SPEC_CACHE: Optional[dict] = None


def load_product_spec(force_reload: bool = False) -> dict:
    """Load + validate ``product_spec.yaml``. Returns the products mapping
    keyed by product code (e.g. ``"SRA"``).
    """
    global _PRODUCT_SPEC_CACHE
    if not force_reload and _PRODUCT_SPEC_CACHE is not None:
        return _PRODUCT_SPEC_CACHE
    if not _SPEC_PATH.exists():
        raise FileNotFoundError(f"product_spec.yaml not found at {_SPEC_PATH}")
    raw = yaml.safe_load(_SPEC_PATH.read_text())
    if not isinstance(raw, dict) or "products" not in raw:
        raise ValueError("product_spec.yaml must have a top-level 'products' mapping")
    products = raw["products"]
    if not isinstance(products, dict):
        raise ValueError("'products' must be a dict")

    # Validate enabled products
    for code, spec in products.items():
        if not spec.get("enabled", False):
            continue
        missing = _REQUIRED_FIELDS_ENABLED - set(spec)
        if missing:
            raise ValueError(
                f"Enabled product {code!r} is missing required fields: {missing}")

    _PRODUCT_SPEC_CACHE = products
    return products


def get_product(code: str) -> dict:
    """Return the spec dict for a single product code."""
    spec = load_product_spec()
    if code not in spec:
        raise KeyError(f"Unknown product code: {code!r} (known: {list(spec)})")
    return spec[code]


def is_enabled(code: str) -> bool:
    """True iff product ``code`` is enabled in the spec."""
    try:
        return bool(get_product(code).get("enabled", False))
    except KeyError:
        return False


def enabled_products() -> list[str]:
    """List codes of all enabled products."""
    return [c for c, s in load_product_spec().items() if s.get("enabled", False)]


def libor_cutover_date(code: str) -> Optional[date]:
    """Return the LIBOR cutover date for a product, or None.

    For SRA, used to filter analog pools / KNN matchers / OU calibration
    /  event-impact regressions to bars >= cutover (per plan §4 +
    §15.1 D1).
    """
    spec = get_product(code)
    raw = spec.get("libor_cutover_date")
    if raw is None:
        return None
    return date.fromisoformat(str(raw))


def cmc_outright_nodes(code: str) -> list[int]:
    """Return the CMC outright node tenor list (months) for a product."""
    return list(get_product(code).get("cmc_outright_nodes", []))


def cmc_spread_nodes(code: str) -> list[tuple]:
    """Return the CMC spread node pairs as list of (front_M, back_M)."""
    return [tuple(p) for p in get_product(code).get("cmc_spread_nodes", [])]


def cmc_fly_nodes(code: str) -> list[tuple]:
    """Return the CMC fly node triples as list of (left_M, mid_M, right_M)."""
    return [tuple(t) for t in get_product(code).get("cmc_fly_nodes", [])]


def dv01_per_lot(code: str) -> float:
    """Return DV01 per lot in the product's local currency."""
    spec = get_product(code)
    for k in ("dv01_per_lot_usd", "dv01_per_lot_eur", "dv01_per_lot_gbp"):
        v = spec.get(k)
        if v is not None:
            return float(v)
    return 25.0   # SR3 default


def tick_size(code: str) -> float:
    return float(get_product(code).get("tick_size", 0.005))


def bp_multiplier(code: str) -> float:
    return float(get_product(code).get("bp_multiplier", 100.0))
