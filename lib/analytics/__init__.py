"""Analytics namespace (Phase 3+).

Hosts the framework-spec analytics modules referenced by the senior-trader
roadmap §12:
- ``turn_adjuster``  (Phase 3, A10) — calendar-dummy regression of CMC change
- ``regime_a1``      (Phase 4, A1)  — PCA → GMM(K=6) → HMM → Hungarian
- ``policy_path_a4`` (Phase 5, A4)  — Heitfield-Park sequential bootstrap
- ``event_impact_a11`` (Phase 6, A11) — Tier 1-4 event surprise regression
"""
