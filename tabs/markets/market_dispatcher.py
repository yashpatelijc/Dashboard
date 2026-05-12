"""Market dispatcher — every non-SRA market routes to the SAME canonical SRA
tab module (tabs.us.sra), passing its base_product code so the full UI
(Curve · Analysis · PCA · Historical Event Impact, each with all sub-views)
renders identically with only the data source swapped.

This guarantees zero UI drift between SRA and other markets — same charts,
same panels, same dynamic exits, same backtest expander.
"""
from __future__ import annotations

from tabs.us import sra as _sra_router


def render(base_product: str) -> None:
    """Render the canonical 4-sub-sub-tab SRA UI for the given market code."""
    _sra_router.render(base_product=base_product)
