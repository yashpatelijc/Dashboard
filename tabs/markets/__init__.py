"""Multi-market wrappers — all non-SRA markets render via the same canonical
SRA tab modules (tabs.us.sra.render(base_product=...)).

Modules:
  - market_dispatcher.render(base_product)  — thin delegate to tabs.us.sra
  - fundamentals_tab.render(region)          — regional BBG inventory viewer
"""
