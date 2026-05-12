"""Background-thread pre-warming for expensive cached computations.

The Technicals subtab's first scan takes ~30-60s cold (29 SRA outrights ×
22 detectors + composites + 60d track record). Once cached the same call
returns instantly. This module starts the scan in background daemon
threads on app/page first-load so by the time the user clicks Technicals
the cache is populated and the UI is instant.

Multi-market support:
  - ``ensure_prewarm(base_product='SRA')`` warms one specific market's caches
  - ``ensure_prewarm_all_markets()`` spawns parallel daemon threads for
    every market in ``lib.markets.MARKETS`` so all region pages are warm
    by the time the user navigates to any of them.

How it works:
  · ``ensure_prewarm`` is idempotent per (base_product) — uses module-level
    locks so each market's warm runs exactly once per server process.
  · Spawns ``threading.Thread(daemon=True)`` so the Streamlit UI is never
    blocked. Daemon threads die when the server shuts down.
  · The background thread calls ``scan_universe`` with the same arg
    signature the UI will use (including ``base_product``), populating
    Streamlit's process-level ``MemoryCacheStorageManager``.
  · Pre-warms OUTRIGHT scope first (most-likely first click). Then SPREAD
    and FLY tenors with the smallest universes.

Call sites:
  - ``app.py``                          (landing page) — warms ALL markets
  - ``pages/1_US_Economy.py``           — warms SRA explicitly
  - ``pages/3_Eurozone.py``             — warms ER + FSR + FER
  - ``pages/4_UK_Economy.py``           — warms SON
  - ``pages/5_Australia_Economy.py``    — warms YBA
  - ``pages/6_Canada_Economy.py``       — warms CRA
"""
from __future__ import annotations

import threading
import time
import traceback


_LOCK = threading.Lock()
_PREWARM_STARTED: dict = {}    # base_product → bool
_PREWARM_THREADS: dict = {}    # base_product → Thread
_PREWARM_STATUS: dict = {}     # base_product → status dict


def _empty_status() -> dict:
    return {
        "started_at": None,
        "outright_done_at": None,
        "spread_3m_done_at": None,
        "fly_3m_done_at": None,
        "errors": [],
    }


def _prewarm_worker(base_product: str) -> None:
    """Background-thread body. Pre-warms scan_universe for the most common
    scopes / tenors. Errors are logged to status['errors'] but never propagated."""
    status = _PREWARM_STATUS.setdefault(base_product, _empty_status())
    try:
        from lib.setups.scan import scan_universe
        if base_product == "SRA":
            from lib.sra_data import (get_available_tenors,
                                          get_sra_snapshot_latest_date)
            snap = get_sra_snapshot_latest_date()
            tenor_fn = get_available_tenors
        else:
            from lib.market_data import (get_available_tenors as _gat,
                                              get_snapshot_latest_date)
            snap = get_snapshot_latest_date(base_product)
            tenor_fn = lambda s: _gat(base_product, s)

        if snap is None:
            status["errors"].append(f"{base_product}: snapshot unavailable")
            return
        asof_str = snap.isoformat()

        # 1) Outright scope first
        try:
            scan_universe("outright", None, asof_str, history_days=280,
                            base_product=base_product)
            status["outright_done_at"] = time.time()
        except Exception as e:
            status["errors"].append(f"{base_product} outright: {e}")
            status["errors"].append(traceback.format_exc()[:500])

        # 2) Spread tenors (3M first)
        try:
            spread_tenors = tenor_fn("spread") or []
            preferred = sorted(spread_tenors, key=lambda t: (t != 3, t))
            for t in preferred[:2]:
                try:
                    scan_universe("spread", int(t), asof_str, history_days=280,
                                    base_product=base_product)
                    if int(t) == 3:
                        status["spread_3m_done_at"] = time.time()
                except Exception as e:
                    status["errors"].append(f"{base_product} spread {t}M: {e}")
        except Exception as e:
            status["errors"].append(f"{base_product} spread tenors: {e}")

        # 3) Fly tenors
        try:
            fly_tenors = tenor_fn("fly") or []
            preferred = sorted(fly_tenors, key=lambda t: (t != 3, t))
            for t in preferred[:2]:
                try:
                    scan_universe("fly", int(t), asof_str, history_days=280,
                                    base_product=base_product)
                    if int(t) == 3:
                        status["fly_3m_done_at"] = time.time()
                except Exception as e:
                    status["errors"].append(f"{base_product} fly {t}M: {e}")
        except Exception as e:
            status["errors"].append(f"{base_product} fly tenors: {e}")

        # 4) PCA panel — also pre-warm the heavy build_full_pca_panel for this market
        try:
            from lib.pca import build_full_pca_panel
            build_full_pca_panel(snap, base_product=base_product, mode="positional",
                                    sparse_enabled=False)
            status["pca_done_at"] = time.time()
        except Exception as e:
            status["errors"].append(f"{base_product} pca: {e}")

    except Exception as e:    # noqa: BLE001 — outer safety net
        status["errors"].append(f"{base_product} worker: {e}")
        status["errors"].append(traceback.format_exc()[:500])


def ensure_prewarm(base_product: str = "SRA") -> None:
    """Call from app.py / pages — triggers the background pre-warm for
    `base_product` exactly once per server process. Returns immediately."""
    with _LOCK:
        if _PREWARM_STARTED.get(base_product):
            return
        _PREWARM_STARTED[base_product] = True
        _PREWARM_STATUS[base_product] = _empty_status()
        _PREWARM_STATUS[base_product]["started_at"] = time.time()
        thr = threading.Thread(
            target=_prewarm_worker, args=(base_product,),
            name=f"prewarm-{base_product}", daemon=True,
        )
        _PREWARM_THREADS[base_product] = thr
        thr.start()


def ensure_prewarm_all_markets() -> None:
    """Spawn one prewarm thread per registered market. Each is independent
    and idempotent — safe to call from app.py landing page."""
    try:
        from lib.markets import list_markets
        for code in list_markets():
            ensure_prewarm(code)
    except Exception:
        # Fall back to SRA-only if registry unavailable
        ensure_prewarm("SRA")


def get_prewarm_status(base_product: str = "SRA") -> dict:
    """Read-only snapshot of pre-warm status for `base_product`."""
    return dict(_PREWARM_STATUS.get(base_product, _empty_status()))


def get_all_prewarm_status() -> dict:
    """Per-market prewarm status dict."""
    return {code: dict(s) for code, s in _PREWARM_STATUS.items()}


def is_prewarm_done(base_product: str = "SRA") -> bool:
    """True iff the outright scan has been pre-warmed for this market."""
    return (_PREWARM_STATUS.get(base_product, {})
              .get("outright_done_at") is not None)
