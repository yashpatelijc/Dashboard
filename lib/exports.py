"""Phase 12 exports — CSV + JSON for trades / fires / recommendations."""
from __future__ import annotations

from datetime import date
from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd


def export_signal_emit_csv(asof: date) -> Optional[str]:
    """Return signal_emit snapshot as CSV string."""
    p = Path(__file__).resolve().parent.parent / ".signal_cache" / f"signal_emit_{asof.isoformat()}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    return df.to_csv(index=False)


def export_top_recommendations_csv(asof: date, n: int = 10) -> Optional[str]:
    from lib.sra_data import top_recommended_signals
    df = top_recommended_signals(asof, n=n)
    if df is None or df.empty:
        return None
    return df.to_csv(index=False)


def export_event_impact_csv(asof: date) -> Optional[str]:
    p = Path(__file__).resolve().parent.parent / ".cmc_cache" / f"event_impact_{asof.isoformat()}.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p).to_csv(index=False)


def export_regime_states_csv(asof: date) -> Optional[str]:
    p = Path(__file__).resolve().parent.parent / ".cmc_cache" / f"regime_states_{asof.isoformat()}.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p).to_csv(index=False)


def export_policy_path_csv(asof: date) -> Optional[str]:
    p = Path(__file__).resolve().parent.parent / ".cmc_cache" / f"policy_path_{asof.isoformat()}.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p).to_csv(index=False)
