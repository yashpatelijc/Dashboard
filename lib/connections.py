"""Database connections for STIRS_DASHBOARD.

Two data sources:
1. OHLC market data â€” DuckDB snapshot at D:\\Python Projects\\QH_API_APPS\\analytics_kit\\db_snapshot\\
2. BBG fundamentals â€” Parquet files at D:\\BBG data\\parquet\\

Threading model â€” each connection getter is **thread-local**, not
``@st.cache_resource``. DuckDB's Python client is NOT safe to call from
multiple threads on the same connection object (it crashes with
``InternalException: Attempted to dereference unique_ptr that is NULL``
under contention). But DuckDB IS happy with multiple read-only
connections to the same file. So:

- The Streamlit UI thread gets its own connection (created lazily on
  first access, reused across re-runs of the same session).
- The Technicals pre-warm daemon thread (``lib/prewarm.py``) gets its
  own connection.
- Any other helper threads get their own.

Each connection lives for the lifetime of its thread â€” daemon threads
die at process exit, session threads die when the session closes.
"""
from __future__ import annotations

import glob
import os
import threading
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
import streamlit as st


OHLC_SNAPSHOT_DIR = r"D:\Python Projects\QH_API_APPS\analytics_kit\db_snapshot"
BBG_PARQUET_ROOT = r"D:\BBG data\parquet"
BBG_WAREHOUSE_DUCKDB = r"D:\BBG data\warehouse.duckdb"


_thread_local = threading.local()


def resolve_latest_ohlc_snapshot() -> Optional[str]:
    """Find the most recent market_data_v2_*.duckdb snapshot."""
    pattern = os.path.join(OHLC_SNAPSHOT_DIR, "market_data_v2_*.duckdb")
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def get_ohlc_connection() -> Optional[duckdb.DuckDBPyConnection]:
    """Per-thread read-only DuckDB connection to the latest OHLC snapshot.

    NOT cached via ``@st.cache_resource`` â€” that would share a single
    connection across all threads, which DuckDB's client cannot
    tolerate. Instead each thread builds its own connection on first
    access and reuses it thereafter.
    """
    con = getattr(_thread_local, "ohlc_con", None)
    if con is not None:
        return con
    path = resolve_latest_ohlc_snapshot()
    if not path:
        return None
    con = duckdb.connect(path, read_only=True)
    _thread_local.ohlc_con = con
    return con


def get_bbg_warehouse_connection() -> Optional[duckdb.DuckDBPyConnection]:
    """Per-thread read-only DuckDB connection to BBG warehouse (note: views
    are broken â€” read parquets directly)."""
    con = getattr(_thread_local, "bbg_warehouse_con", None)
    if con is not None:
        return con
    if not os.path.exists(BBG_WAREHOUSE_DUCKDB):
        return None
    con = duckdb.connect(BBG_WAREHOUSE_DUCKDB, read_only=True)
    _thread_local.bbg_warehouse_con = con
    return con


def get_bbg_inmemory_connection() -> duckdb.DuckDBPyConnection:
    """Per-thread in-memory DuckDB for ad-hoc parquet queries against BBG warehouse."""
    con = getattr(_thread_local, "bbg_inmem_con", None)
    if con is not None:
        return con
    con = duckdb.connect(":memory:")
    _thread_local.bbg_inmem_con = con
    return con


@st.cache_data(show_spinner=False, ttl=3600)
def list_bbg_categories() -> pd.DataFrame:
    """List BBG parquet categories with file counts and total size."""
    rows = []
    if not os.path.isdir(BBG_PARQUET_ROOT):
        return pd.DataFrame(columns=["category", "files", "size_mb"])
    for d in sorted(os.listdir(BBG_PARQUET_ROOT)):
        full = os.path.join(BBG_PARQUET_ROOT, d)
        if not os.path.isdir(full):
            continue
        files = glob.glob(os.path.join(full, "*.parquet"))
        size_mb = sum(os.path.getsize(f) for f in files) / (1024 * 1024)
        rows.append({"category": d, "files": len(files), "size_mb": round(size_mb, 2)})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, ttl=3600)
def list_bbg_tickers(category: str) -> pd.DataFrame:
    """List parquet files (tickers) inside a BBG category."""
    cat_dir = os.path.join(BBG_PARQUET_ROOT, category)
    if not os.path.isdir(cat_dir):
        return pd.DataFrame(columns=["ticker", "size_kb", "modified"])
    rows = []
    for f in sorted(glob.glob(os.path.join(cat_dir, "*.parquet"))):
        st_info = os.stat(f)
        rows.append({
            "ticker": Path(f).stem,
            "size_kb": round(st_info.st_size / 1024, 1),
            "modified": pd.Timestamp(st_info.st_mtime, unit="s"),
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, ttl=3600)
def read_bbg_parquet(category: str, ticker: str) -> pd.DataFrame:
    """Read a single BBG parquet (ticker) into a DataFrame.

    Uses DuckDB's parquet reader â€” pyarrow's reader hits
    `Repetition level histogram size mismatch` on some BBG files
    written with a newer parquet format. DuckDB handles them fine.
    """
    path = os.path.join(BBG_PARQUET_ROOT, category, f"{ticker}.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    duck_path = path.replace("\\", "/")
    con = get_bbg_inmemory_connection()
    try:
        return con.execute(f"SELECT * FROM read_parquet('{duck_path}')").fetchdf()
    except Exception:
        # Fallback to pandas/pyarrow only if DuckDB also fails
        return pd.read_parquet(path)


@st.cache_data(show_spinner=False, ttl=600)
def list_ohlc_tables() -> pd.DataFrame:
    """List tables and views in the OHLC snapshot with row counts."""
    con = get_ohlc_connection()
    if con is None:
        return pd.DataFrame(columns=["table_type", "table_name", "rows"])
    objs = con.execute(
        "SELECT table_type, table_name FROM information_schema.tables "
        "WHERE table_schema='main' ORDER BY table_type, table_name"
    ).fetchdf()
    rows = []
    for _, r in objs.iterrows():
        try:
            n = con.execute(f'SELECT COUNT(*) FROM "{r.table_name}"').fetchone()[0]
        except Exception:
            n = None
        rows.append({"table_type": r.table_type, "table_name": r.table_name, "rows": n})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, ttl=600)
def describe_ohlc_table(table_name: str) -> pd.DataFrame:
    """Return column schema of an OHLC table."""
    con = get_ohlc_connection()
    if con is None:
        return pd.DataFrame()
    return con.execute(f'DESCRIBE "{table_name}"').fetchdf()


def run_ohlc_sql(sql: str, limit: int = 1000) -> pd.DataFrame:
    """Execute arbitrary read-only SQL against the OHLC DB. Caller must trust input source (we only run from Settings tab)."""
    con = get_ohlc_connection()
    if con is None:
        raise RuntimeError("OHLC database not available")
    return con.execute(sql).fetchdf().head(limit)


def ohlc_db_path_info() -> dict:
    """Metadata about the resolved OHLC snapshot."""
    path = resolve_latest_ohlc_snapshot()
    if not path or not os.path.exists(path):
        return {"path": None, "size_gb": None, "modified": None}
    s = os.stat(path)
    return {
        "path": path,
        "size_gb": round(s.st_size / (1024**3), 2),
        "modified": pd.Timestamp(s.st_mtime, unit="s"),
    }


def bbg_warehouse_path_info() -> dict:
    """Metadata about BBG warehouse.duckdb."""
    p = BBG_WAREHOUSE_DUCKDB
    if not os.path.exists(p):
        return {"path": None, "size_kb": None, "modified": None}
    s = os.stat(p)
    return {
        "path": p,
        "size_kb": round(s.st_size / 1024, 1),
        "modified": pd.Timestamp(s.st_mtime, unit="s"),
    }


# ====================================================================
# PCA-engine additions: BBG parquet readers
# ====================================================================

def read_bbg_parquet_robust(category: str, ticker_filename: str) -> Optional[pd.DataFrame]:
    """Read a BBG parquet using DuckDB first (handles newer parquet formats), fall
    back to pandas/pyarrow if DuckDB unavailable.

    BBG warehouse files were written with a parquet format pyarrow chokes on with
    "Repetition level histogram size mismatch" â€” DuckDB reads them fine. This is
    the non-Streamlit-cached version used by analytics modules outside the UI.

    Args:
      category: BBG category (e.g. 'eco', 'xcot', 'rates_drivers', 'vol_indices')
      ticker_filename: filename WITHOUT extension (e.g. 'MOVE_Index')

    Returns:
      DataFrame or None if the file doesn't exist / read fails.
    """
    path = os.path.join(BBG_PARQUET_ROOT, category, f"{ticker_filename}.parquet")
    if not os.path.exists(path):
        return None
    try:
        import duckdb
        con = duckdb.connect(":memory:")
        duck_path = path.replace("\\", "/")
        return con.execute(f"SELECT * FROM read_parquet('{duck_path}')").fetchdf()
    except Exception:
        try:
            return pd.read_parquet(path)
        except Exception:
            return None


def bbg_parquet_to_series(category: str, ticker_filename: str,
                            value_col: str = "PX_LAST",
                            date_col: str = "date") -> Optional[pd.Series]:
    """Load a BBG parquet and return it as a clean pandas Series indexed by date.

    Returns None if the file doesn't exist, has no rows, or lacks the value/date columns.
    """
    df = read_bbg_parquet_robust(category, ticker_filename)
    if df is None or df.empty:
        return None
    if value_col not in df.columns or date_col not in df.columns:
        return None
    df = df.dropna(subset=[value_col, date_col]).copy()
    if df.empty:
        return None
    df[date_col] = pd.to_datetime(df[date_col])
    s = df.sort_values(date_col).set_index(date_col)[value_col].astype(float)
    return s.dropna()


@st.cache_data(show_spinner=False, ttl=600)
def list_ohlc_tables() -> pd.DataFrame:
    """List tables and views in the OHLC snapshot with row counts."""
    con = get_ohlc_connection()
    if con is None:
        return pd.DataFrame(columns=["table_type", "table_name", "rows"])
    objs = con.execute(
        "SELECT table_type, table_name FROM information_schema.tables "
        "WHERE table_schema='main' ORDER BY table_type, table_name"
    ).fetchdf()
    rows = []
    for _, r in objs.iterrows():
        try:
            n = con.execute(f'SELECT COUNT(*) FROM "{r.table_name}"').fetchone()[0]
        except Exception:
            n = None
        rows.append({"table_type": r.table_type, "table_name": r.table_name, "rows": n})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False, ttl=600)
def describe_ohlc_table(table_name: str) -> pd.DataFrame:
    """Return column schema of an OHLC table."""
    con = get_ohlc_connection()
    if con is None:
        return pd.DataFrame()
    return con.execute(f'DESCRIBE "{table_name}"').fetchdf()


def run_ohlc_sql(sql: str, limit: int = 1000) -> pd.DataFrame:
    """Execute arbitrary read-only SQL against the OHLC DB. Caller must trust input source (we only run from Settings tab)."""
    con = get_ohlc_connection()
    if con is None:
        raise RuntimeError("OHLC database not available")
    return con.execute(sql).fetchdf().head(limit)


def ohlc_db_path_info() -> dict:
    """Metadata about the resolved OHLC snapshot."""
    path = resolve_latest_ohlc_snapshot()
    if not path or not os.path.exists(path):
        return {"path": None, "size_gb": None, "modified": None}
    s = os.stat(path)
    return {
        "path": path,
        "size_gb": round(s.st_size / (1024**3), 2),
        "modified": pd.Timestamp(s.st_mtime, unit="s"),
    }


def bbg_warehouse_path_info() -> dict:
    """Metadata about BBG warehouse.duckdb."""
    p = BBG_WAREHOUSE_DUCKDB
    if not os.path.exists(p):
        return {"path": None, "size_kb": None, "modified": None}
    s = os.stat(p)
    return {
        "path": p,
        "size_kb": round(s.st_size / 1024, 1),
        "modified": pd.Timestamp(s.st_mtime, unit="s"),
    }