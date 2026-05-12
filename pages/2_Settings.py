"""Settings — General / OHLC DB Viewer / BBG Parquet Viewer."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

from lib.connections import (
    BBG_PARQUET_ROOT,
    OHLC_SNAPSHOT_DIR,
    bbg_warehouse_path_info,
    describe_ohlc_table,
    get_bbg_inmemory_connection,
    get_bbg_warehouse_connection,
    get_ohlc_connection,
    list_bbg_categories,
    list_bbg_tickers,
    list_ohlc_tables,
    ohlc_db_path_info,
    read_bbg_parquet,
    run_ohlc_sql,
)
from lib.components import page_header
from lib.css import inject_global_css, render_sidebar_brand

st.set_page_config(
    page_title="Settings — STIRS",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_global_css()
render_sidebar_brand()

page_header(
    title="⚙️ Settings",
    subtitle="Configuration · data viewers · custom SQL",
)

tab_general, tab_health, tab_units, tab_ohlc, tab_bbg = st.tabs(
    ["General", "System Health", "Unit conventions", "OHLC DB Viewer", "BBG Parquet Viewer"]
)


# -----------------------------------------------------------------------------
# GENERAL
# -----------------------------------------------------------------------------
with tab_general:
    st.subheader("General settings")
    st.caption("Knobs persist for the current session only (no file write yet).")

    col1, col2 = st.columns(2)
    with col1:
        st.text_input("OHLC snapshot directory", value=OHLC_SNAPSHOT_DIR, disabled=True,
                      help="Auto-resolves to the most recent market_data_v2_*.duckdb file.")
        st.text_input("BBG parquet root", value=BBG_PARQUET_ROOT, disabled=True)

    with col2:
        st.number_input("Default sample-row limit (DB viewer)", min_value=10, max_value=10_000, value=100, step=10,
                        key="row_limit_default")
        st.checkbox("Auto-refresh data caches every load", value=False, key="auto_refresh")

    st.divider()
    st.markdown("##### About")
    st.markdown(
        "- **OHLC DB**: read-only DuckDB snapshot of `market_data_v2`.\n"
        "- **BBG warehouse**: parquet files per ticker, accessed via DuckDB `read_parquet` or pyarrow.\n"
        "- The `warehouse.duckdb` file has views with hardcoded source paths from a previous install — they fail to query. "
        "Use the **BBG Parquet Viewer** which reads parquets directly."
    )


# -----------------------------------------------------------------------------
# SYSTEM HEALTH — Phase 0.D — freshness + data-depth + pipeline status
# -----------------------------------------------------------------------------
with tab_health:
    from lib.freshness import (
        freshness_report, freshness_traffic_light_html, BUDGETS_HOURS,
    )
    st.subheader("System Health · data freshness, depth, and pipeline status")
    st.caption(
        "Every data source the dashboard reads is freshness-budgeted; "
        "any source older than its budget shows red. Use this panel before "
        "trusting any analytic emission."
    )

    rep = freshness_report()
    overall = rep["overall_status"]
    chip = freshness_traffic_light_html(overall, label=f"OVERALL · {overall.upper()}")
    st.markdown(f"<div style='margin-bottom:0.8rem;'>{chip}</div>",
                unsafe_allow_html=True)

    # ---------- Data sources freshness table ----------
    st.markdown("##### Data sources")
    src_rows = []
    for s in rep["sources"]:
        chip_html = freshness_traffic_light_html(s["status"], label=s["status"].upper())
        age_str = f"{s['age_hours']:.1f}h" if s["age_hours"] is not None else "—"
        src_rows.append({
            "Status": s["status"],
            "Source": s["source"],
            "Last modified": s["last_modified"] or "—",
            "Age (h)": age_str,
            "Budget (h)": s["budget_hours"],
            "Detail": s["detail"],
        })
    import pandas as _pd
    src_df = _pd.DataFrame(src_rows)
    st.dataframe(src_df, hide_index=True, use_container_width=True)

    # ---------- Category depth table ----------
    st.markdown("##### BBG category depth (data sufficiency for analytics)")
    st.caption(
        "An analytic that needs ≥5y of history will be sample-bound on any "
        "category flagged 'shallow'. The dashboard auto-deepens when the "
        "external backfill program lands deeper history."
    )
    depth_rows = []
    for d in rep["category_depth"]:
        suff_color = {"sufficient": "green", "shallow": "amber", "missing": "red"}[d["sufficiency"]]
        suff_chip = freshness_traffic_light_html(suff_color, label=d["sufficiency"].upper())
        depth_rows.append({
            "Sufficiency": d["sufficiency"],
            "Category": d["category"],
            "Sample ticker": d["sample_ticker"],
            "Rows": d["rows"] if d["rows"] is not None else "—",
            "First date": d["first_date"] or "—",
            "Last date": d["last_date"] or "—",
            "Years": f"{d['years_of_history']:.1f}y" if d["years_of_history"] is not None else "—",
        })
    depth_df = _pd.DataFrame(depth_rows)
    st.dataframe(depth_df, hide_index=True, use_container_width=True)

    # ---------- Pipeline status ----------
    st.markdown("##### Pipelines")
    try:
        from lib.prewarm import get_prewarm_status, is_prewarm_done
        prewarm_status = get_prewarm_status()
    except Exception:
        prewarm_status = {}
    try:
        from lib.backtest.cycle import get_backtest_status
        backtest_status = get_backtest_status()
    except Exception:
        backtest_status = {}

    cp1, cp2 = st.columns(2)
    with cp1:
        st.markdown("**Prewarm daemon**")
        if prewarm_status.get("started_at"):
            done = "✓ outright" if prewarm_status.get("outright_done_at") else "running"
            st.markdown(f"- Status: `{done}`")
            st.markdown(f"- Errors: `{len(prewarm_status.get('errors', []))}`")
        else:
            st.markdown("- Status: `not started this session`")
    with cp2:
        st.markdown("**Backtest daemon (10-day cycle)**")
        if backtest_status.get("running"):
            st.markdown("- Status: `running`")
        elif backtest_status.get("fresh"):
            st.markdown(f"- Status: `fresh` (last run {backtest_status.get('last_run')})")
        elif backtest_status.get("last_run"):
            st.markdown(f"- Status: `stale` (last run {backtest_status.get('last_run')})")
        else:
            st.markdown("- Status: `not yet computed`")
        if backtest_status.get("file_size_mb"):
            st.markdown(f"- tmia.duckdb size: `{backtest_status['file_size_mb']} MB`")
        if backtest_status.get("n_trades") is not None:
            st.markdown(f"- Last trade count: `{backtest_status['n_trades']:,}`")

    st.markdown("##### Manifests + caches")
    c1, c2, c3 = st.columns(3)
    from pathlib import Path as _P
    cmc_dir = _P(r"D:\STIRS_DASHBOARD\.cmc_cache")
    bk_dir = _P(r"D:\STIRS_DASHBOARD\.backtest_cache")
    log_dir = _P(r"D:\STIRS_DASHBOARD\.logs")
    c1.metric("CMC parquets", str(len(list(cmc_dir.glob("*.parquet")))) if cmc_dir.exists() else "—")
    c2.metric("Backtest snapshots", str(len(list((bk_dir / "snapshots").glob("*.duckdb")))) if (bk_dir / "snapshots").exists() else "—")
    c3.metric("Log files", str(len(list(log_dir.glob("*.log")))) if log_dir.exists() else "—")

    # ---------- Manifest viewer ----------
    # ---------- Per-contract data-quality table (Phase 1 deliverable) ----------
    st.markdown("##### Per-contract data quality")
    st.caption(
        "Bar count, first/last bar, and missing-bar % per SRA contract. "
        "Rows highlighted red if last_bar > 7d behind the snapshot date "
        "(staleness threshold)."
    )
    with st.expander("Load per-contract quality table (queries OHLC DB; ~2-3s for 760+ rows)",
                       expanded=False):
        if st.button("Run scan", key="btn_run_dq_scan"):
            try:
                from lib.connections import get_ohlc_connection
                con = get_ohlc_connection()
                dq = con.cursor().execute("""
                    SELECT
                        cc.symbol, cc.base_product, cc.strategy, cc.tenor_months,
                        MIN(DATE(to_timestamp(t.time/1000.0))) AS first_bar,
                        MAX(DATE(to_timestamp(t.time/1000.0))) AS last_bar,
                        COUNT(*) AS bars,
                        DATE_DIFF('day',
                            MAX(DATE(to_timestamp(t.time/1000.0))),
                            CURRENT_DATE) AS days_since_last_bar
                    FROM mde2_timeseries t
                    JOIN mde2_contracts_catalog cc ON cc.symbol = t.symbol
                    WHERE cc.is_continuous=FALSE
                      AND t.interval='1D'
                      AND t.calc_method='api'
                      AND cc.base_product='SRA'
                    GROUP BY cc.symbol, cc.base_product, cc.strategy, cc.tenor_months
                    ORDER BY days_since_last_bar, cc.symbol
                """).fetchdf()
                # Style: red rows with days_since_last_bar > 7
                stale = dq[dq["days_since_last_bar"] > 7]
                fresh = dq[dq["days_since_last_bar"] <= 7]
                cdq1, cdq2, cdq3 = st.columns(3)
                cdq1.metric("Total contracts", f"{len(dq):,}")
                cdq2.metric("Live (≤ 7d)", f"{len(fresh):,}")
                cdq3.metric("Stale (> 7d)", f"{len(stale):,}",
                              delta=None,
                              delta_color="inverse" if len(stale) > 0 else "off")
                tab_fresh, tab_stale = st.tabs([f"Live ({len(fresh)})",
                                                    f"Stale ({len(stale)})"])
                with tab_fresh:
                    st.dataframe(fresh, hide_index=True, use_container_width=True,
                                  height=400)
                with tab_stale:
                    if len(stale) > 0:
                        st.dataframe(stale, hide_index=True, use_container_width=True,
                                      height=400)
                    else:
                        st.success("No stale contracts.")
            except Exception as e:
                st.error(f"Data-quality scan failed: {e}")

    # ---------- FOMC meeting calendar (Phase 1 deliverable) ----------
    st.markdown("##### FOMC meeting calendar (`config/cb_meetings.yaml`)")
    st.caption(
        "All known FOMC meetings past + future ≥2y, with realized outcomes "
        "derived from FDTRMID step changes (where data is available). "
        "Used by A4 Heitfield-Park, A11-event Tier 1, FOMC blackouts, "
        "and Fed-dots-vs-OIS chart."
    )
    try:
        from lib.cb_meetings import fomc_meetings_with_outcomes
        from datetime import date as _date
        meetings = fomc_meetings_with_outcomes()
        today = _date.today()
        past_with_outcome = [m for m in meetings if m.realized_change_bp is not None]
        future = [m for m in meetings if m.date >= today]
        cm1, cm2, cm3 = st.columns(3)
        cm1.metric("Total meetings", f"{len(meetings)}")
        cm2.metric("Past with outcome", f"{len(past_with_outcome)}")
        cm3.metric("Future", f"{len(future)}")
        # Render upcoming + recent
        upcoming = sorted(future, key=lambda m: m.date)[:6]
        recent_with_outcome = sorted(past_with_outcome, key=lambda m: m.date,
                                          reverse=True)[:6]
        with st.expander("Upcoming meetings (next 6) + recent outcomes (last 6)",
                           expanded=False):
            up_df = _pd.DataFrame([
                {"Date": str(m.date), "SEP": m.has_sep, "Statement-only": m.statement_only}
                for m in upcoming
            ])
            st.markdown("**Upcoming**")
            st.dataframe(up_df, hide_index=True, use_container_width=True)
            re_df = _pd.DataFrame([
                {"Date": str(m.date), "Direction": m.direction,
                 "Magnitude (bp)": f"{m.magnitude_bp:.1f}" if m.magnitude_bp else "—",
                 "Pre target (bp)": f"{m.pre_target_bp:.1f}" if m.pre_target_bp else "—",
                 "Post target (bp)": f"{m.post_target_bp:.1f}" if m.post_target_bp else "—"}
                for m in recent_with_outcome
            ])
            st.markdown("**Recent (with derived outcomes from FDTRMID)**")
            st.dataframe(re_df, hide_index=True, use_container_width=True)
    except Exception as e:
        st.warning(f"FOMC calendar unavailable: {e}")

    # ---------- Product spec (Phase 1 deliverable) ----------
    st.markdown("##### Product spec (`config/product_spec.yaml`)")
    st.caption(
        "Per-product conventions (tick size, DV01, calendar, CMC nodes, "
        "LIBOR cutover). SRA-only enabled per plan §15."
    )
    try:
        from lib.product_spec import load_product_spec
        sp = load_product_spec()
        prod_rows = []
        for code, s in sp.items():
            prod_rows.append({
                "Code": code,
                "Enabled": s.get("enabled", False),
                "Description": s.get("description", ""),
                "Tick size": s.get("tick_size", "—"),
                "DV01/lot": (s.get("dv01_per_lot_usd")
                              or s.get("dv01_per_lot_eur")
                              or s.get("dv01_per_lot_gbp") or "—"),
                "BP mult": s.get("bp_multiplier", "—"),
                "LIBOR cutover": s.get("libor_cutover_date") or "—",
            })
        st.dataframe(_pd.DataFrame(prod_rows),
                       hide_index=True, use_container_width=True)
    except Exception as e:
        st.warning(f"Product spec unavailable: {e}")

    st.markdown("##### CMC manifest (most recent)")
    if cmc_dir.exists():
        manifests = sorted(cmc_dir.glob("manifest_*.json"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if manifests:
            import json as _json
            data = _json.loads(manifests[0].read_text())
            st.markdown(f"- File: `{manifests[0].name}`")
            st.markdown(f"- Builder version: `{data.get('builder_version', '?')}`")
            st.markdown(f"- Asof: `{data.get('asof_date', '?')}`")
            st.markdown(f"- Contracts in chain: `{data.get('n_contracts_in_chain', '?')}`")
            st.markdown(f"- Historical rolls: `{data.get('n_rolls_historical', '?')}`")
            gs = data.get("gap_stats_bp", {})
            if gs:
                st.markdown(
                    f"- Gap stats: mean `{gs.get('mean_bp', 0):.2f}` bp · "
                    f"std `{gs.get('std_bp', 0):.2f}` bp · "
                    f"abs median `{gs.get('abs_median_bp', 0):.2f}` bp"
                )
            missing = data.get("missing_contracts_in_chain", [])
            if missing:
                st.markdown(f"- ⚠️ Missing contracts: `{', '.join(missing[:8])}{' ...' if len(missing) > 8 else ''}`")
            with st.expander("Full manifest JSON"):
                st.json(data)
        else:
            st.info("No CMC manifests yet — daemon hasn't run.")
    else:
        st.info("No `.cmc_cache/` directory yet.")

    # ---------- Turn-adjuster diagnostics (Phase 3 deliverable) ----------
    st.markdown("##### Turn-adjuster diagnostics (Phase 3)")
    st.caption(
        "Per-CMC-node Newey-West HAC regression of daily close change on "
        "calendar dummies (QE / YE / FOMC-week / NFP-week / holiday-week). "
        "Residuals (`residual_change`) are the input to Phase 4's regime "
        "classifier — they preserve drift and orthogonality to all dummies."
    )
    try:
        from datetime import date as _date_ta
        from lib.turn_residuals_verify import verify_all as _verify_turn
        from lib.turn_adjuster_daemon import (
            get_turn_adjuster_status as _ta_status,
            is_turn_residuals_fresh as _ta_fresh,
        )
        from lib.sra_data import (
            load_turn_diagnostics as _load_diag,
            load_turn_residuals_panel as _load_resid,
        )
        from lib.cmc import _cache_paths as _cmc_paths
        from lib.freshness import freshness_traffic_light_html as _light

        # Resolve latest residuals asof
        ta_files = sorted(cmc_dir.glob("turn_residuals_manifest_*.json"),
                              key=lambda p: p.stat().st_mtime, reverse=True)
        if not ta_files:
            st.info(
                "No turn-residuals cache yet — daemon spawns on next app "
                "launch. To force-build: `python -m lib.analytics.turn_adjuster`."
            )
        else:
            ta_stem = ta_files[0].stem.replace("turn_residuals_manifest_", "")
            ta_asof = _date_ta.fromisoformat(ta_stem)

            # Verification pill
            report = _verify_turn(ta_asof)
            overall_ok = report["passed_all"]
            chip_status = "green" if overall_ok else "red"
            chip_label = (f"{report['n_passed']}/{report['n_total']} CHECKS PASS"
                              if overall_ok
                              else f"{report['n_total']-report['n_passed']} FAILED")
            st.markdown(
                f"<div style='margin-bottom:0.6rem;'>{_light(chip_status, chip_label)}</div>",
                unsafe_allow_html=True,
            )

            # Per-check results
            with st.expander("Verification check details", expanded=not overall_ok):
                for name, c in report["checks"].items():
                    icon = "✓" if c["passed"] else "✗"
                    st.markdown(f"- `{icon}` **{name}**: {c['message']}")

            # Daemon status
            ds = _ta_status()
            ds_msg = []
            if ds.get("started_at"):
                ds_msg.append(f"started {pd.Timestamp(ds['started_at'], unit='s'):%Y-%m-%d %H:%M}")
            if ds.get("skipped_reason") == "fresh":
                ds_msg.append("**status: cache fresh, no rebuild needed**")
            elif ds.get("completed_at"):
                ds_msg.append(f"completed {pd.Timestamp(ds['completed_at'], unit='s'):%Y-%m-%d %H:%M} "
                                f"(n_nodes={ds.get('n_nodes_total','?')}, "
                                f"skipped={ds.get('n_nodes_skipped','?')})")
            if ds.get("errors"):
                ds_msg.append(f"errors: {ds['errors'][-1][:200]}")
            if ds_msg:
                st.caption(" · ".join(ds_msg))

            # Per-node diagnostics table
            st.markdown("**Per-node regression diagnostics**")
            diag = _load_diag(ta_asof)
            if not diag.empty:
                # Compute "top significant dummy" per node (smallest p among the 5)
                p_cols = [f"p_{d}" for d in ("QE","YE","FOMC","NFP","HOL")]
                def _top_sig(row):
                    pairs = [(d, row.get(f"p_{d}")) for d in ("QE","YE","FOMC","NFP","HOL")]
                    pairs = [(d, p) for d, p in pairs if pd.notna(p)]
                    if not pairs:
                        return "—"
                    d, p = min(pairs, key=lambda x: x[1])
                    return f"{d} (p={p:.3f})"
                diag_view = pd.DataFrame({
                    "scope":           diag["scope"],
                    "node":            diag["cmc_node"],
                    "n_obs":           diag["n_obs"].astype(int),
                    "R²":              diag["r_squared"].round(4),
                    "var_red %":       diag["var_reduction_pct"].round(3),
                    "top sig dummy":   diag.apply(_top_sig, axis=1),
                    "low_sample":      diag["low_sample_dummies"].fillna("").replace("", "—"),
                })
                st.dataframe(diag_view, hide_index=True, use_container_width=True,
                                height=360)

                # Drill-in: select a node and chart raw vs residual
                st.markdown("**Drill-in — raw vs residual change**")
                cols = st.columns([2, 2, 2])
                scope_sel = cols[0].selectbox("scope", ["outright","spread","fly"],
                                                  key="ta_scope_sel")
                nodes_for_scope = diag[diag["scope"] == scope_sel]["cmc_node"].tolist()
                node_sel = cols[1].selectbox("node", nodes_for_scope, key="ta_node_sel")
                show_n = cols[2].selectbox("window",
                                                ["last 60d","last 180d","last 1y","all post-cutover"],
                                                key="ta_window_sel")
                window_map = {"last 60d": 60, "last 180d": 180, "last 1y": 252,
                                  "all post-cutover": None}
                tail_n = window_map.get(show_n)

                resid_long = _load_resid(scope_sel, ta_asof)
                node_df = (resid_long[resid_long["cmc_node"] == node_sel]
                                .sort_values("bar_date").copy())
                if tail_n is not None:
                    node_df = node_df.tail(tail_n)
                if not node_df.empty:
                    import plotly.graph_objects as _go
                    fig = _go.Figure()
                    fig.add_trace(_go.Scatter(
                        x=node_df["bar_date"], y=node_df["raw_change"],
                        mode="lines", line=dict(color="#60a5fa", width=1.4),
                        name="raw Δ (bp)",
                        hovertemplate="<b>%{x|%Y-%m-%d}</b> raw=%{y:+.3f} bp<extra></extra>",
                    ))
                    fig.add_trace(_go.Scatter(
                        x=node_df["bar_date"], y=node_df["residual_change"],
                        mode="lines", line=dict(color="#fb923c", width=1.4),
                        name="residual Δ (bp)",
                        hovertemplate="<b>%{x|%Y-%m-%d}</b> resid=%{y:+.3f} bp<extra></extra>",
                    ))
                    fig.update_layout(
                        xaxis=dict(title=None),
                        yaxis=dict(title="Daily change (price-bp)"),
                        height=320,
                        hovermode="x unified",
                        legend=dict(orientation="h", yanchor="bottom", y=1.0,
                                       xanchor="right", x=1.0),
                        margin=dict(l=55, r=20, t=10, b=40),
                    )
                    st.plotly_chart(fig, use_container_width=True, theme=None)

                    # Inline summary
                    valid = node_df.dropna(subset=["raw_change","residual_change"])
                    if not valid.empty:
                        corr = valid["raw_change"].corr(valid["residual_change"])
                        st.caption(
                            f"window n={len(valid)} bars · "
                            f"mean(raw)={valid['raw_change'].mean():+.4f} bp · "
                            f"mean(resid)={valid['residual_change'].mean():+.4f} bp · "
                            f"var(raw)={valid['raw_change'].var():.4f} · "
                            f"var(resid)={valid['residual_change'].var():.4f} · "
                            f"corr(raw, resid)={corr:.4f}"
                        )

            # Manifest viewer
            with st.expander("Full turn-residuals manifest JSON"):
                import json as _json_ta
                st.json(_json_ta.loads(ta_files[0].read_text()))
    except Exception as e:
        st.warning(f"Turn-adjuster diagnostics unavailable: {e}")

    # ---------- Regime classifier diagnostics (Phase 4 deliverable) ----------
    st.markdown("##### Regime classifier diagnostics (Phase 4 — A1)")
    st.caption(
        "PCA → GMM(K=6, full Σ) → HMM smoothing → Hungarian relabel on the "
        "Phase 3 residual_change panel. State-id stability across refits "
        "guaranteed by Hungarian assignment to previous-fit centroids."
    )
    try:
        from datetime import date as _date_rg
        from lib.regime_verify import verify_all as _verify_rg
        from lib.regime_daemon import (
            get_regime_status as _rg_status,
            is_regime_fresh as _rg_fresh,
        )
        from lib.sra_data import (
            load_regime_states as _load_states,
            load_regime_diagnostics as _load_rg_diag,
            get_current_regime as _cur_rg,
        )
        from lib.freshness import freshness_traffic_light_html as _light_rg

        rg_files = sorted(cmc_dir.glob("regime_manifest_*.json"),
                              key=lambda p: p.stat().st_mtime, reverse=True)
        if not rg_files:
            st.info(
                "No regime cache yet — daemon spawns on next app launch. "
                "To force-build: `python -m lib.analytics.regime_a1`."
            )
        else:
            rg_stem = rg_files[0].stem.replace("regime_manifest_", "")
            rg_asof = _date_rg.fromisoformat(rg_stem)

            report = _verify_rg(rg_asof)
            ok = report["passed_all"]
            chip_label = (f"{report['n_passed']}/{report['n_total']} CHECKS PASS"
                              if ok
                              else f"{report['n_total']-report['n_passed']} FAILED")
            st.markdown(
                f"<div style='margin-bottom:0.6rem;'>"
                f"{_light_rg('green' if ok else 'red', chip_label)}</div>",
                unsafe_allow_html=True,
            )

            with st.expander("Verification check details", expanded=not ok):
                for name, c in report["checks"].items():
                    icon = "✓" if c["passed"] else "✗"
                    st.markdown(f"- `{icon}` **{name}**: {c['message']}")

            ds = _rg_status()
            ds_msg = []
            if ds.get("started_at"):
                ds_msg.append(f"started {pd.Timestamp(ds['started_at'], unit='s'):%Y-%m-%d %H:%M}")
            if ds.get("skipped_reason") == "fresh":
                ds_msg.append("**status: cache fresh, no rebuild needed**")
            elif ds.get("completed_at"):
                ds_msg.append(f"completed {pd.Timestamp(ds['completed_at'], unit='s'):%Y-%m-%d %H:%M} "
                                f"(K={ds.get('K','?')}, active={ds.get('n_active_states','?')})")
            if ds.get("errors"):
                ds_msg.append(f"errors: {ds['errors'][-1][:200]}")
            if ds_msg:
                st.caption(" · ".join(ds_msg))

            # Current regime + state distribution
            cur = _cur_rg(rg_asof)
            if cur:
                st.markdown(
                    f"**Current regime ({cur['bar_date']})**: "
                    f"`{cur['state_name']}` (S{cur['state_id']}) "
                    f"posterior = {cur['top_state_posterior']:.3f}"
                )

            diag = _load_rg_diag(rg_asof)
            if not diag.empty:
                diag_view = diag[["state_id","state_name","n_bars","frac_bars",
                                       "mean_run_length","max_run_length"]].copy()
                diag_view["frac_bars"] = (diag_view["frac_bars"] * 100).round(1)
                diag_view = diag_view.rename(columns={"frac_bars": "frac %"})
                st.markdown("**State distribution + run-length stats**")
                st.dataframe(diag_view, hide_index=True, use_container_width=True)

            # Regime timeline plot (last N days)
            states = _load_states(rg_asof)
            if not states.empty:
                tail_n = st.slider("Timeline window (bars)", 60, len(states),
                                          min(252, len(states)), key="rg_timeline_window")
                states_tail = states.sort_values("bar_date").tail(tail_n)
                import plotly.graph_objects as _go_rg
                # Color per state name (deterministic palette)
                palette = {
                    "TRENDING_HIKE_LATE": "#ef4444",
                    "TRENDING_CUT_LATE":  "#22d3ee",
                    "STEEPENING_RISK_ON": "#22c55e",
                    "FLATTENING_FLIGHT":  "#f59e0b",
                    "RANGE_BOUND":        "#94a3b8",
                    "VOL_BREAKOUT":       "#a855f7",
                }
                fig = _go_rg.Figure()
                # Draw vertical-band style: scatter with colors
                colors = [palette.get(s, "#5e6975") for s in states_tail["state_name"]]
                fig.add_trace(_go_rg.Scatter(
                    x=states_tail["bar_date"], y=states_tail["top_state_posterior"],
                    mode="markers", marker=dict(color=colors, size=5),
                    hovertemplate="<b>%{x|%Y-%m-%d}</b><br>"
                                       "regime: %{text}<br>"
                                       "posterior: %{y:.3f}<extra></extra>",
                    text=states_tail["state_name"],
                ))
                fig.update_layout(
                    xaxis=dict(title=None),
                    yaxis=dict(title="Top-state posterior", range=[0, 1.05]),
                    height=260, margin=dict(l=55, r=20, t=10, b=40),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True, theme=None)

            # Transition matrix viewer
            import json as _json_rg
            mf = _json_rg.loads(rg_files[0].read_text())
            with st.expander("Transition matrix + manifest"):
                tm = pd.DataFrame(mf["transition_matrix"],
                                       index=[f"S{i}" for i in range(mf["K"])],
                                       columns=[f"S{i}" for i in range(mf["K"])])
                st.markdown("**Transition matrix (row = from, col = to)**")
                st.dataframe(tm.round(3), use_container_width=True)
                st.markdown("**Manifest JSON**")
                st.json(mf)
    except Exception as e:
        st.warning(f"Regime classifier diagnostics unavailable: {e}")

    # ---------- Policy-path diagnostics (Phase 5 — A4) ----------
    st.markdown("##### Heitfield-Park policy path (Phase 5 — A4)")
    st.caption(
        "SLSQP-refined per-FOMC PMFs over the {-50, -25, 0, +25, +50} bp lattice. "
        "Decomposes the SR3 strip; ZLB-constrained; SR3-only (no FedWatch scrape per D2)."
    )
    try:
        from datetime import date as _date_pp
        from lib.policy_path_verify import verify_all as _verify_pp
        from lib.policy_path_daemon import get_policy_path_status as _pp_status
        from lib.sra_data import (
            load_policy_path as _load_pp,
            get_policy_path_summary as _pp_summary,
        )
        from lib.freshness import freshness_traffic_light_html as _light_pp

        pp_files = sorted(cmc_dir.glob("policy_path_manifest_*.json"),
                              key=lambda p: p.stat().st_mtime, reverse=True)
        if not pp_files:
            st.info("No policy-path cache yet — `python -m lib.analytics.policy_path_a4`.")
        else:
            pp_asof = _date_pp.fromisoformat(
                pp_files[0].stem.replace("policy_path_manifest_", ""))
            rep = _verify_pp(pp_asof)
            ok = rep["passed_all"]
            n_pass = rep["n_passed"]; n_total = rep["n_total"]
            chip_label_pp = (f"{n_pass}/{n_total} CHECKS PASS" if ok
                                  else f"{n_total - n_pass} FAILED")
            st.markdown(
                f"<div style='margin-bottom:0.6rem;'>"
                f"{_light_pp('green' if ok else 'red', chip_label_pp)}"
                f"</div>",
                unsafe_allow_html=True,
            )
            with st.expander("Verification check details", expanded=not ok):
                for name, c in rep["checks"].items():
                    st.markdown(f"- `{'✓' if c['passed'] else '✗'}` **{name}**: {c['message']}")

            ds = _pp_status()
            ds_msg = []
            if ds.get("started_at"):
                ds_msg.append(f"started {pd.Timestamp(ds['started_at'], unit='s'):%Y-%m-%d %H:%M}")
            if ds.get("skipped_reason") == "fresh":
                ds_msg.append("**status: cache fresh, no rebuild needed**")
            elif ds.get("completed_at"):
                ds_msg.append(f"completed {pd.Timestamp(ds['completed_at'], unit='s'):%Y-%m-%d %H:%M}")
            if ds_msg:
                st.caption(" · ".join(ds_msg))

            summary = _pp_summary(pp_asof)
            if summary:
                cols_pp = st.columns(4)
                cols_pp[0].metric("Current rate",
                                       f"{summary.get('current_rate_bp', 0)/100:.3f}%")
                cols_pp[1].metric("Terminal rate",
                                       f"{summary.get('terminal_rate_bp', 0)/100:.3f}%",
                                       delta=f"{(summary.get('terminal_rate_bp', 0) - summary.get('current_rate_bp', 0)):+.0f} bp")
                cols_pp[2].metric("Floor rate",
                                       f"{summary.get('floor_rate_bp', 0)/100:.3f}%")
                cols_pp[3].metric("Cycle", summary.get("cycle_label", "?"))

            df_pp = _load_pp(pp_asof)
            if not df_pp.empty:
                st.markdown("**Per-meeting PMFs + expected change**")
                view = df_pp[["meeting_date", "days_to_meeting",
                                  "expected_change_bp", "post_meeting_rate_bp",
                                  "sr3_implied_rate_bp", "fit_residual_bp",
                                  "p_m50", "p_m25", "p_0", "p_p25", "p_p50"]].copy()
                view["expected_change_bp"] = view["expected_change_bp"].round(1)
                view["post_meeting_rate_bp"] = view["post_meeting_rate_bp"].round(1)
                view["sr3_implied_rate_bp"] = view["sr3_implied_rate_bp"].round(1)
                view["fit_residual_bp"] = view["fit_residual_bp"].round(2)
                for col in ["p_m50", "p_m25", "p_0", "p_p25", "p_p50"]:
                    view[col] = view[col].round(3)
                st.dataframe(view, hide_index=True, use_container_width=True)

                # Plot: post-meeting rate path vs SR3 implied
                import plotly.graph_objects as _go_pp
                fig_pp = _go_pp.Figure()
                fig_pp.add_trace(_go_pp.Scatter(
                    x=df_pp["meeting_date"], y=df_pp["sr3_implied_rate_bp"]/100,
                    mode="lines+markers", line=dict(color="#60a5fa", width=2),
                    name="SR3-implied",
                    hovertemplate="<b>%{x}</b><br>SR3=%{y:.3f}%<extra></extra>",
                ))
                fig_pp.add_trace(_go_pp.Scatter(
                    x=df_pp["meeting_date"], y=df_pp["post_meeting_rate_bp"]/100,
                    mode="lines+markers", line=dict(color="#fb923c", width=2),
                    name="A4 post-meeting",
                    hovertemplate="<b>%{x}</b><br>A4=%{y:.3f}%<extra></extra>",
                ))
                fig_pp.update_layout(
                    xaxis=dict(title=None), yaxis=dict(title="Rate %"),
                    height=300, hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.0,
                                  xanchor="right", x=1.0),
                    margin=dict(l=55, r=20, t=10, b=40),
                )
                st.plotly_chart(fig_pp, use_container_width=True, theme=None)

            with st.expander("Manifest JSON"):
                import json as _json_pp
                st.json(_json_pp.loads(pp_files[0].read_text()))
    except Exception as e:
        st.warning(f"Policy-path diagnostics unavailable: {e}")

    # ---------- Event-impact diagnostics (Phase 6 — A11) ----------
    st.markdown("##### Event-impact regression (Phase 6 — A11)")
    st.caption(
        "Per (ticker × CMC node × regime) regression of same-day residual_change "
        "on standardised event surprise. Composite score = 40% |β_norm| + 30% R² + "
        "30% hit_rate. Recency-weighted variant uses halflife = 2y; flagged "
        "'becoming_more_important' if recency_score > full_score × 1.5."
    )
    try:
        from datetime import date as _date_ei
        from lib.event_impact_verify import verify_all as _verify_ei
        from lib.event_impact_daemon import get_event_impact_status as _ei_status
        from lib.sra_data import (
            load_event_impact as _load_ei,
            top_event_signals as _top_ei,
        )
        from lib.freshness import freshness_traffic_light_html as _light_ei

        ei_files = sorted(cmc_dir.glob("event_impact_manifest_*.json"),
                              key=lambda p: p.stat().st_mtime, reverse=True)
        if not ei_files:
            st.info("No event-impact cache yet — `python -m lib.analytics.event_impact_a11`.")
        else:
            ei_asof = _date_ei.fromisoformat(
                ei_files[0].stem.replace("event_impact_manifest_", ""))
            rep = _verify_ei(ei_asof)
            ok = rep["passed_all"]
            n_p = rep["n_passed"]; n_t = rep["n_total"]
            chip_lab = (f"{n_p}/{n_t} CHECKS PASS" if ok
                            else f"{n_t - n_p} FAILED")
            st.markdown(
                f"<div style='margin-bottom:0.6rem;'>"
                f"{_light_ei('green' if ok else 'red', chip_lab)}</div>",
                unsafe_allow_html=True,
            )
            with st.expander("Verification check details", expanded=not ok):
                for n, c in rep["checks"].items():
                    st.markdown(f"- `{'✓' if c['passed'] else '✗'}` **{n}**: {c['message']}")

            ds = _ei_status()
            ds_msg = []
            if ds.get("started_at"):
                ds_msg.append(f"started {pd.Timestamp(ds['started_at'], unit='s'):%Y-%m-%d %H:%M}")
            if ds.get("skipped_reason") == "fresh":
                ds_msg.append("**status: cache fresh**")
            elif ds.get("completed_at"):
                ds_msg.append(f"completed {pd.Timestamp(ds['completed_at'], unit='s'):%Y-%m-%d %H:%M} "
                                f"(tickers={ds.get('n_tickers_built','?')}, rows={ds.get('n_rows','?')})")
            if ds_msg:
                st.caption(" · ".join(ds_msg))

            ei_df = _load_ei(ei_asof)
            if not ei_df.empty:
                # Top-10 by score_full
                st.markdown("**Top 10 event signals (full-sample, ALL regimes)**")
                top10 = _top_ei(ei_asof, n=10, regime_filter="ALL")
                view = top10[["ticker", "cmc_node", "axis", "n_obs",
                                  "beta", "r_squared", "hit_rate",
                                  "score_full", "score_recency_weighted",
                                  "becoming_more_important"]].copy()
                for c in ["beta","r_squared","hit_rate","score_full","score_recency_weighted"]:
                    view[c] = view[c].astype(float).round(4)
                st.dataframe(view, hide_index=True, use_container_width=True)

                # "Becoming more important" subset
                bmi = ei_df[(ei_df["scope"]=="full") & (ei_df["becoming_more_important"])]
                if not bmi.empty:
                    st.markdown(f"**Becoming more important** ({len(bmi)} signals)")
                    st.dataframe(bmi[["ticker","cmc_node","axis","score_full",
                                            "score_recency_weighted"]].round(4),
                                       hide_index=True, use_container_width=True)
                else:
                    st.caption("No signals currently flagged 'becoming_more_important'.")

                # Per-ticker × node heatmap (top 8 tickers by max score)
                top_tickers = (ei_df[ei_df["scope"]=="full"]
                                  .groupby("ticker")["score_full"].max()
                                  .nlargest(8).index.tolist())
                heat_data = (ei_df[(ei_df["scope"]=="full")
                                          & (ei_df["ticker"].isin(top_tickers))]
                                  .pivot_table(index="ticker", columns="cmc_node",
                                                  values="score_full", aggfunc="first"))
                # Reorder columns to canonical M0..M60
                from lib.cmc import list_cmc_nodes
                col_order = [c for c in list_cmc_nodes("outright") if c in heat_data.columns]
                heat_data = heat_data.reindex(columns=col_order)
                st.markdown("**Score heatmap: top 8 tickers × CMC nodes**")
                import plotly.graph_objects as _go_ei
                fig_h = _go_ei.Figure(data=_go_ei.Heatmap(
                    z=heat_data.values, x=heat_data.columns, y=heat_data.index,
                    colorscale="Viridis", colorbar=dict(title="score"),
                    hovertemplate="%{y} × %{x}: %{z:.3f}<extra></extra>",
                ))
                fig_h.update_layout(height=300, margin=dict(l=120, r=20, t=10, b=40))
                st.plotly_chart(fig_h, use_container_width=True, theme=None)

            with st.expander("Manifest JSON"):
                import json as _json_ei
                st.json(_json_ei.loads(ei_files[0].read_text()))
    except Exception as e:
        st.warning(f"Event-impact diagnostics unavailable: {e}")

    # ---------- signal_emit canonical feed (Phase 7) ----------
    st.markdown("##### signal_emit canonical feed (Phase 7)")
    st.caption(
        "Single-source-of-truth for every emission (TREND / MR / STIR / REGIME / "
        "POLICY / EVENT). 24-column schema; conflict detection across direction "
        "× cmc_node; gate_quality bucketing CLEAN / LOW_SAMPLE / TRANSITION / CONFLICT."
    )
    try:
        from datetime import date as _date_se
        from pathlib import Path as _Path_se
        from lib.signal_emit_verify import verify_all as _verify_se
        from lib.signal_emit_daemon import get_signal_emit_status as _se_status
        from lib.sra_data import (
            load_signal_emissions as _load_se,
            top_recommended_signals as _top_se,
        )
        from lib.freshness import freshness_traffic_light_html as _light_se

        sig_dir = _Path_se(r"D:\STIRS_DASHBOARD\.signal_cache")
        se_files = (sorted(sig_dir.glob("signal_emit_manifest_*.json"),
                              key=lambda p: p.stat().st_mtime, reverse=True)
                       if sig_dir.exists() else [])
        if not se_files:
            st.info("No signal_emit cache yet — `python -m lib.signal_emit`.")
        else:
            se_asof = _date_se.fromisoformat(
                se_files[0].stem.replace("signal_emit_manifest_", ""))
            rep = _verify_se(se_asof)
            ok = rep["passed_all"]
            n_p = rep["n_passed"]; n_t = rep["n_total"]
            chip_lab = (f"{n_p}/{n_t} CHECKS PASS" if ok
                            else f"{n_t - n_p} FAILED")
            st.markdown(
                f"<div style='margin-bottom:0.6rem;'>"
                f"{_light_se('green' if ok else 'red', chip_lab)}</div>",
                unsafe_allow_html=True,
            )
            with st.expander("Verification check details", expanded=not ok):
                for n, c in rep["checks"].items():
                    st.markdown(f"- `{'✓' if c['passed'] else '✗'}` **{n}**: {c['message']}")

            ds = _se_status()
            if ds.get("started_at") and ds.get("completed_at"):
                st.caption(f"daemon completed {pd.Timestamp(ds['completed_at'], unit='s'):%Y-%m-%d %H:%M} "
                              f"({ds.get('n_emissions','?')} emissions)")

            df_se = _load_se(se_asof)
            if not df_se.empty:
                # Top recommended (gate=CLEAN, eff_n>=30, no flags)
                st.markdown("**Top 10 recommended (gate=CLEAN, eff_n≥30, no flags)**")
                top = _top_se(se_asof, n=10)
                if top.empty:
                    st.info("No emissions pass the §3.2 ranked-feed gate yet.")
                else:
                    view = top[["detector_id", "detector_family", "scope",
                                  "cmc_node", "signal_type", "direction",
                                  "percentile_rank", "confidence_stoplight",
                                  "eff_n", "regime_stability", "rationale"]].copy()
                    for c in ["percentile_rank", "regime_stability"]:
                        view[c] = view[c].astype(float).round(3)
                    st.dataframe(view, hide_index=True, use_container_width=True)

                # Full table breakdown
                st.markdown("**All emissions (current snapshot)**")
                full_view = df_se[["detector_id", "detector_family", "scope",
                                       "cmc_node", "signal_type", "direction",
                                       "percentile_rank", "confidence_stoplight",
                                       "eff_n", "gate_quality", "tags"]].copy()
                full_view["percentile_rank"] = full_view["percentile_rank"].round(2)
                st.dataframe(full_view, hide_index=True, use_container_width=True)

                # Conflicts sub-feed (per plan §3.2)
                conflicts = df_se[df_se["conflict_flag"]]
                if not conflicts.empty:
                    st.markdown(f"**Conflicts** ({len(conflicts)} emissions)")
                    cv = conflicts[["detector_id","cmc_node","direction","rationale"]]
                    st.dataframe(cv, hide_index=True, use_container_width=True)

            with st.expander("Manifest JSON"):
                st.json(__import__("json").loads(se_files[0].read_text()))
    except Exception as e:
        st.warning(f"signal_emit diagnostics unavailable: {e}")

    # ---------- Trader tools (Phase 8 — interpretation layer) ----------
    st.markdown("##### Trader tools (Phase 8 — sizer / hedge calc / counterfactual)")
    st.caption(
        "Position sizer, DV01-neutral hedge calc, counterfactual analog-fires "
        "histogram. Per §15 D4=No: sizer assumes flat $10k base notional, "
        "hedge calc generic, no book context."
    )
    try:
        from lib.sizing import compare_sizing_methods, DEFAULT_ACCOUNT_USD
        from lib.hedge_calc import (
            dv01_neutral_one_two_one, dv01_neutral_butterfly,
            build_legs_table, basket_dv01_check,
        )
        from lib.counterfactual import analog_outcomes, histogram_data

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown("**Position sizer (compare 3 methods)**")
            wr = st.number_input("win_rate", value=0.55, min_value=0.0,
                                       max_value=1.0, key="ps_wr")
            awr = st.number_input("avg_win_R", value=1.5, min_value=0.0,
                                        key="ps_awr")
            alr = st.number_input("avg_loss_R", value=1.0, min_value=0.0,
                                        key="ps_alr")
            tvol = st.number_input("target_vol % daily", value=0.5,
                                          min_value=0.01, key="ps_tvol")
            evol = st.number_input("expected_vol % daily", value=1.0,
                                          min_value=0.01, key="ps_evol")
            ff = st.number_input("fixed_fraction", value=0.01,
                                       min_value=0.0001, max_value=0.10,
                                       key="ps_ff")
            stp = st.number_input("stop_R (bp)", value=8.0, min_value=0.1,
                                        key="ps_stp")
            acct = st.number_input("Account USD",
                                          value=float(DEFAULT_ACCOUNT_USD),
                                          min_value=100.0, key="ps_acct")
            stats = {"win_rate": wr, "avg_win_R": awr, "avg_loss_R": alr,
                       "target_vol_pct": tvol, "expected_vol_pct": evol,
                       "fixed_fraction": ff, "stop_R": stp}
            sizer_df = compare_sizing_methods(stats, account_usd=acct)
            view_s = sizer_df[["method","notional_usd","n_lots","capped",
                                  "rationale"]].copy()
            view_s["notional_usd"] = view_s["notional_usd"].round(0)
            view_s["n_lots"] = view_s["n_lots"].round(2)
            st.dataframe(view_s, hide_index=True, use_container_width=True)

        with col_b:
            st.markdown("**DV01-neutral hedge calc**")
            structure = st.selectbox("structure",
                                            ["1:2:1 fly", "Butterfly (custom DV01)",
                                              "Pair (calendar spread)"],
                                            key="hc_struct")
            if structure == "1:2:1 fly":
                lots = dv01_neutral_one_two_one()
                syms = ["M3 (front)", "M6 (mid)", "M9 (back)"]
                prices = [96.20, 96.50, 96.70]
                df_legs = build_legs_table(syms, lots, prices)
                st.dataframe(df_legs, hide_index=True, use_container_width=True)
                bk = basket_dv01_check(df_legs)
                st.caption(f"basket DV01 = ${bk['total_dv01_usd_per_bp']:+.2f}/bp · "
                              f"is_neutral = {bk['is_dv01_neutral']}")
            elif structure == "Butterfly (custom DV01)":
                d1 = st.number_input("d1 ($/bp)", value=25.0, key="hc_d1")
                d2 = st.number_input("d2 ($/bp)", value=25.0, key="hc_d2")
                d3 = st.number_input("d3 ($/bp)", value=25.0, key="hc_d3")
                target = st.number_input("PC3 weight", value=1.0, key="hc_pc3")
                lots = dv01_neutral_butterfly([d1, d2, d3], target)
                df_legs = build_legs_table(
                    ["L1", "L2", "L3"], lots, [96.20, 96.50, 96.70])
                st.dataframe(df_legs, hide_index=True, use_container_width=True)
                bk = basket_dv01_check(df_legs)
                st.caption(f"basket DV01 = ${bk['total_dv01_usd_per_bp']:+.2f}/bp · "
                              f"is_neutral = {bk['is_dv01_neutral']}")
            else:
                from lib.hedge_calc import dv01_neutral_pair
                df_d = st.number_input("front DV01 ($/bp)", value=25.0,
                                              key="hc_pf")
                db_d = st.number_input("back DV01 ($/bp)", value=25.0,
                                              key="hc_pb")
                lots = dv01_neutral_pair(df_d, db_d)
                df_legs = build_legs_table(
                    ["front", "back"], lots, [96.20, 96.30])
                st.dataframe(df_legs, hide_index=True, use_container_width=True)
                bk = basket_dv01_check(df_legs)
                st.caption(f"basket DV01 = ${bk['total_dv01_usd_per_bp']:+.2f}/bp · "
                              f"is_neutral = {bk['is_dv01_neutral']}")

        with col_c:
            st.markdown("**Counterfactual analog fires**")
            sid = st.text_input("setup_id", value="A1_TREND_DONCHIAN",
                                      key="cf_sid")
            n_min = st.number_input("min sample", value=30, min_value=1,
                                          key="cf_nmin")
            outcomes = analog_outcomes(sid, n_min=n_min)
            if not outcomes.get("sufficient_sample", False):
                st.warning(f"Insufficient sample: {outcomes.get('reason', '')}")
            else:
                st.metric("Win rate", f"{outcomes['win_rate']:.1%}")
                st.metric("Expectancy R", f"{outcomes['expectancy_R']:+.3f}")
                qs = outcomes["quantiles"]
                st.markdown(
                    f"R quantiles: 10%={qs['q10']:+.2f} · 25%={qs['q25']:+.2f} · "
                    f"50%={qs['q50']:+.2f} · 75%={qs['q75']:+.2f} · 90%={qs['q90']:+.2f}"
                )
                # Histogram
                import plotly.graph_objects as _go_cf
                hd = histogram_data(np.array(outcomes["R_realized"]))
                if hd["bin_edges"]:
                    centers = [
                        (hd["bin_edges"][i] + hd["bin_edges"][i+1]) / 2
                        for i in range(len(hd["counts"]))
                    ]
                    fig_cf = _go_cf.Figure(data=_go_cf.Bar(
                        x=centers, y=hd["counts"], marker_color="#fb923c",
                    ))
                    fig_cf.update_layout(
                        xaxis=dict(title="R_realized"),
                        yaxis=dict(title="count"),
                        height=200, margin=dict(l=40, r=20, t=10, b=40),
                    )
                    st.plotly_chart(fig_cf, use_container_width=True, theme=None)
    except Exception as e:
        st.warning(f"Trader-tools unavailable: {e}")

    # ---------- Opportunity modules (Phase 9) ----------
    st.markdown("##### Opportunity modules (Phase 9)")
    st.caption(
        "A6 OU calibration · A4m PC-momentum · A1c carry decomposition · "
        "A12d pre/post-event drift · A9 regime-transition gate. "
        "Stubs (per plan §15): A2p pack/bundle RV, A6s STL, A7c 8-phase cycle."
    )
    try:
        from datetime import date as _date_op
        from lib.opportunity_verify import verify_all as _verify_op
        from lib.opportunity_daemon import get_opportunity_status as _op_status
        from lib.freshness import freshness_traffic_light_html as _light_op
        from pathlib import Path as _Path_op
        cmc_dir_op = _Path_op(r"D:\STIRS_DASHBOARD\.cmc_cache")
        op_files = sorted(cmc_dir_op.glob("opp_manifest_*.json"),
                              key=lambda p: p.stat().st_mtime, reverse=True)
        if not op_files:
            st.info("No opportunity-modules cache yet — `python -m lib.analytics.opportunity_modules`.")
        else:
            op_asof = _date_op.fromisoformat(
                op_files[0].stem.replace("opp_manifest_", ""))
            rep_op = _verify_op(op_asof)
            ok = rep_op["passed_all"]
            n_p = rep_op["n_passed"]; n_t = rep_op["n_total"]
            chip_lab_op = (f"{n_p}/{n_t} CHECKS PASS" if ok
                                else f"{n_t - n_p} FAILED")
            st.markdown(
                f"<div style='margin-bottom:0.6rem;'>"
                f"{_light_op('green' if ok else 'red', chip_lab_op)}</div>",
                unsafe_allow_html=True,
            )
            with st.expander("Verification check details", expanded=not ok):
                for n, c in rep_op["checks"].items():
                    st.markdown(f"- `{'✓' if c['passed'] else '✗'}` **{n}**: {c['message']}")

            ds_op = _op_status()
            if ds_op.get("started_at") and ds_op.get("completed_at"):
                st.caption(f"daemon completed {pd.Timestamp(ds_op['completed_at'], unit='s'):%Y-%m-%d %H:%M} "
                              f"(modules: {ds_op.get('modules_built','?')})")

            # A6 OU table — top 10 by 1/half_life (most reverting)
            try:
                a6 = pd.read_parquet(cmc_dir_op / f"opp_a6_ou_{op_asof.isoformat()}.parquet")
                st.markdown("**A6 OU — top 10 by ADF<0.05 + half-life ∈ [3d, 60d]**")
                a6_pass = a6[a6["passes_gate"]].sort_values("half_life_days").head(10)
                if a6_pass.empty:
                    st.info("No nodes currently pass the OU gate (residual_change is dominantly random-walk on this sample).")
                else:
                    st.dataframe(a6_pass[["cmc_node","kappa","mu","half_life_days",
                                                "adf_p_value","current_value",
                                                "expected_first_passage_days"]].round(3),
                                       hide_index=True, use_container_width=True)

                # A4m TSM table
                a4m = pd.read_parquet(cmc_dir_op / f"opp_a4m_tsm_{op_asof.isoformat()}.parquet")
                st.markdown("**A4m TSM (PC1/PC2/PC3 × {21,63,126,252} bars)**")
                st.dataframe(a4m.round(4), hide_index=True, use_container_width=True)

                # A1c carry decomposition — top 10 by |total_change|
                a1c = pd.read_parquet(cmc_dir_op / f"opp_a1c_carry_{op_asof.isoformat()}.parquet")
                a1c_top = a1c.assign(abs_chg=a1c["total_change_21bd"].abs()).nlargest(10, "abs_chg")
                st.markdown("**A1c carry decomposition — top 10 |21bd change|**")
                st.dataframe(a1c_top[["cmc_node","total_change_21bd","carry","roll_down","curve_change"]].round(4),
                                   hide_index=True, use_container_width=True)

                # A12d event drift — heatmap of mean_drift by ticker × window
                a12d = pd.read_parquet(cmc_dir_op / f"opp_a12d_event_drift_{op_asof.isoformat()}.parquet")
                st.markdown("**A12d event-drift (mean drift in bp per window, M3 outright)**")
                heat = a12d.pivot_table(index="ticker", columns="window",
                                              values="mean_drift_bp", aggfunc="first")
                heat = heat.reindex(columns=["pre","event","post_short","post_long"])
                st.dataframe(heat.round(3), use_container_width=True)
            except Exception as eee:
                st.caption(f"(some opportunity tables unavailable: {eee})")
    except Exception as e:
        st.warning(f"Opportunity-modules diagnostics unavailable: {e}")

    # ---------- A2 KNN + A3 path-conditional FV (Phase 10) ----------
    st.markdown("##### A2 KNN bands + A3 path-conditional FV (Phase 10)")
    st.caption(
        "Mahalanobis-Ledoit-Wolf KNN matcher (K=30) + ±60d exclusion + 250d "
        "halflife decay + per-FOMC-bucket conditional FV. Per §15 A2: SRA-only "
        "permanently — all emissions tagged `low_sample`."
    )
    try:
        from datetime import date as _date_kn
        from pathlib import Path as _Path_kn
        cmc_dir_kn = _Path_kn(r"D:\STIRS_DASHBOARD\.cmc_cache")
        kn_files = sorted(cmc_dir_kn.glob("knn_a2_a3_manifest_*.json"),
                              key=lambda p: p.stat().st_mtime, reverse=True)
        if not kn_files:
            st.info("No KNN cache yet — `python -m lib.analytics.knn_a2_a3`.")
        else:
            kn_asof = _date_kn.fromisoformat(
                kn_files[0].stem.replace("knn_a2_a3_manifest_", ""))
            kn_df = pd.read_parquet(cmc_dir_kn / f"knn_a2_a3_{kn_asof.isoformat()}.parquet")
            st.markdown(f"**KNN per-target-node forward FV (today = {kn_asof})**")
            view = kn_df[["target_node","n_analogs","weighted_mean_R_5d",
                              "weighted_mean_R_20d","std_R_5d","std_R_20d",
                              "bucket_p25_mean","bucket_0_mean","bucket_m25_mean",
                              "bucket_p25_count","bucket_0_count","bucket_m25_count",
                              "low_sample_flag"]].copy()
            for c in ["weighted_mean_R_5d","weighted_mean_R_20d","std_R_5d","std_R_20d",
                          "bucket_p25_mean","bucket_0_mean","bucket_m25_mean"]:
                if c in view.columns:
                    view[c] = view[c].astype(float).round(3)
            st.dataframe(view, hide_index=True, use_container_width=True)
            st.caption("Top analog dates per node (first 5 nearest):")
            for _, row in kn_df.iterrows():
                st.markdown(f"- **{row['target_node']}**: {row['analog_top_dates']}")
            with st.expander("Manifest JSON"):
                st.json(__import__("json").loads(kn_files[0].read_text()))
    except Exception as e:
        st.warning(f"KNN A2/A3 diagnostics unavailable: {e}")

    # ---------- Cross-cutting [+] (Phase 11) ----------
    st.markdown("##### Cross-cutting [+] (Phase 11)")
    st.caption(
        "Cointegration screens (Engle-Granger + Johansen) on CMC pairs · "
        "cross-asset risk-regime composite (MOVE/SRVIX/VIX/SKEW/CDX/DXY/SPX) · "
        "Bauer-Swanson 2023 orthogonalised MP surprises (no skewness — D3=defer). "
        "REMOVED per §15: CFTC COT heatmap, Hull-White convexity."
    )
    try:
        from datetime import date as _date_cc
        from pathlib import Path as _Path_cc
        cmc_dir_cc = _Path_cc(r"D:\STIRS_DASHBOARD\.cmc_cache")
        cc_files = sorted(cmc_dir_cc.glob("cross_cutting_manifest_*.json"),
                              key=lambda p: p.stat().st_mtime, reverse=True)
        if not cc_files:
            st.info("No cross-cutting cache yet — `python -m lib.analytics.cross_cutting`.")
        else:
            cc_asof = _date_cc.fromisoformat(
                cc_files[0].stem.replace("cross_cutting_manifest_", ""))
            mf_cc = __import__("json").loads(cc_files[0].read_text())
            cols_cc = st.columns(3)
            cols_cc[0].metric("Cointegrated pairs",
                                  f"{mf_cc.get('n_cointegrated_flagged','?')}/"
                                  f"{mf_cc.get('n_cointegration_pairs','?')}")
            cols_cc[1].metric("Risk regime", mf_cc.get("risk_regime_label", "?"))
            cols_cc[2].metric("Bauer-Swanson meetings",
                                  mf_cc.get("n_bauer_meetings", 0))

            try:
                coint_df = pd.read_parquet(cmc_dir_cc / f"cross_cointegration_{cc_asof.isoformat()}.parquet")
                st.markdown("**Cointegration screens**")
                st.dataframe(coint_df.round(4), hide_index=True, use_container_width=True)
                risk_df = pd.read_parquet(cmc_dir_cc / f"cross_risk_regime_{cc_asof.isoformat()}.parquet")
                st.markdown("**Cross-asset risk-regime composite**")
                st.dataframe(risk_df.round(4), hide_index=True, use_container_width=True)
                bauer_df = pd.read_parquet(cmc_dir_cc / f"cross_bauer_swanson_{cc_asof.isoformat()}.parquet")
                st.markdown("**Bauer-Swanson orthogonalised MP-surprises**")
                st.dataframe(bauer_df.round(3), hide_index=True, use_container_width=True)
            except Exception as eee:
                st.caption(f"(some cross-cutting tables unavailable: {eee})")

            with st.expander("Manifest JSON"):
                st.json(mf_cc)
    except Exception as e:
        st.warning(f"Cross-cutting diagnostics unavailable: {e}")

    # ---------- Macro overlays + ECO calendar (Phase 12) ----------
    st.markdown("##### Macro overlays + ECO calendar (Phase 12)")
    st.caption(
        "Next-7-day economic-release calendar (or last-14-day fallback if BBG "
        "ECO_RELEASE_DT future-dating not available); FOMC blackout vertical-band "
        "annotation helper for any plotly chart. Per §15: Fed-dots chart uses "
        "aggregate SEP only; CFTC COT heatmap removed."
    )
    try:
        from datetime import date as _date_mc
        from lib.macro_overlays import eco_calendar_next_7_days
        cal = eco_calendar_next_7_days()
        if cal is None or cal.empty:
            st.info("No ECO calendar entries — eco/ files lack future ECO_RELEASE_DT timestamps.")
        else:
            st.markdown("**ECO calendar (next 7 days, or recent 14 days as fallback)**")
            view = cal.copy()
            view["actual"] = view["actual"].astype(str)
            view["surprise_dir"] = view["surprise"].apply(
                lambda v: ("↑" if pd.notna(v) and v > 0
                              else "↓" if pd.notna(v) and v < 0
                              else "—"))
            st.dataframe(view, hide_index=True, use_container_width=True)
    except Exception as e:
        st.warning(f"Macro overlays unavailable: {e}")

    # ---------- Exports (Phase 12) ----------
    st.markdown("##### Exports (Phase 12)")
    st.caption("CSV downloads for current snapshots (signal_emit / event_impact / "
                  "regime / policy_path / top recommendations).")
    try:
        from datetime import date as _date_ex
        from pathlib import Path as _P_ex
        from lib.exports import (
            export_signal_emit_csv, export_event_impact_csv,
            export_regime_states_csv, export_policy_path_csv,
            export_top_recommendations_csv,
        )
        cmc_dir_ex = _P_ex(r"D:\STIRS_DASHBOARD\.cmc_cache")
        # Use latest CMC asof
        cmc_cands = sorted(cmc_dir_ex.glob("manifest_*.json"),
                                key=lambda p: p.stat().st_mtime, reverse=True)
        if cmc_cands:
            ex_asof = _date_ex.fromisoformat(
                cmc_cands[0].stem.replace("manifest_", ""))
            cols_ex = st.columns(5)
            csvs = {
                "signal_emit": export_signal_emit_csv(ex_asof),
                "event_impact": export_event_impact_csv(ex_asof),
                "regime_states": export_regime_states_csv(ex_asof),
                "policy_path": export_policy_path_csv(ex_asof),
                "top_recommendations": export_top_recommendations_csv(ex_asof, n=10),
            }
            for i, (name, csv_data) in enumerate(csvs.items()):
                with cols_ex[i]:
                    if csv_data:
                        st.download_button(
                            label=name,
                            data=csv_data,
                            file_name=f"{name}_{ex_asof.isoformat()}.csv",
                            mime="text/csv", key=f"dl_{name}",
                        )
                    else:
                        st.caption(f"{name}: n/a")
    except Exception as e:
        st.warning(f"Exports unavailable: {e}")

    # ---------- Watchlists (Phase 12) ----------
    st.markdown("##### Watchlists (Phase 12)")
    st.caption("Save/load named views. Stored in `.user_state/watchlists.json`.")
    try:
        from lib.watchlist import list_watchlists, save_watchlist, load_watchlist, delete_watchlist
        existing = list_watchlists()
        cwl1, cwl2, cwl3 = st.columns([2, 2, 1])
        wl_name = cwl1.text_input("Watchlist name", value="default", key="wl_name")
        wl_payload = cwl2.text_input("Payload (JSON)", value='{"focus": ["M3","M6"]}', key="wl_payload")
        if cwl3.button("Save", key="wl_save"):
            try:
                payload = __import__("json").loads(wl_payload)
                save_watchlist(wl_name, payload)
                st.success(f"Saved watchlist '{wl_name}'.")
            except Exception as e:
                st.error(f"Save failed: {e}")
        if existing:
            st.markdown(f"**Existing watchlists** ({len(existing)})")
            for w in existing:
                payload = load_watchlist(w)
                col1, col2 = st.columns([4, 1])
                col1.markdown(f"- **{w}**: `{payload}`")
                if col2.button("Delete", key=f"wl_del_{w}"):
                    delete_watchlist(w)
                    st.rerun()
    except Exception as e:
        st.warning(f"Watchlists unavailable: {e}")

    # ---------- Alerts (Phase 12) ----------
    st.markdown("##### Alerts (Phase 12)")
    st.caption("File-based throttled alert log (5/h per setup_id). Slack/desktop "
                  "bindings deferred (require user webhook setup).")
    try:
        from lib.alerts import recent_fires, record_fire, clear_log, THROTTLE_PER_HOUR
        rec = recent_fires(window_hours=24)
        st.markdown(f"**Recent fires (last 24h)** — throttle = {THROTTLE_PER_HOUR}/h per setup")
        if rec.empty:
            st.caption("No fires logged yet.")
        else:
            st.dataframe(rec, hide_index=True, use_container_width=True)
        cala1, cala2 = st.columns(2)
        if cala1.button("Test fire (record A1_TREND_DONCHIAN)", key="al_test"):
            from datetime import date as _date_al
            ok = record_fire("A1_TREND_DONCHIAN", _date_al.today())
            if ok:
                st.success("Fire recorded.")
            else:
                st.warning("Throttled — already 5 fires this hour.")
            st.rerun()
        if cala2.button("Clear alert log", key="al_clear"):
            clear_log()
            st.rerun()
    except Exception as e:
        st.warning(f"Alerts unavailable: {e}")


# -----------------------------------------------------------------------------
# UNIT CONVENTIONS — auto-detected per-contract bp/price scale
# -----------------------------------------------------------------------------
with tab_units:
    st.subheader("Per-contract unit conventions")
    st.caption(
        "Spread and fly contracts may store their close as a raw price difference "
        "or already scaled to basis points. This catalog auto-detects the convention "
        "by comparing each contract's stored close against the implied price-difference "
        "of its underlying outright legs. Analysis uses this to scale Δ correctly."
    )

    from lib.contract_units import (
        CATALOG_PATH, build_catalog, load_catalog,
    )

    cat = load_catalog()
    cu1, cu2, cu3, cu4 = st.columns([1, 1, 1, 2])
    cu1.metric("Catalog rows", f"{len(cat):,}" if not cat.empty else "—")
    if not cat.empty and "built_at" in cat.columns:
        cu2.metric("Last built", str(cat["built_at"].iloc[0])[:16])
    else:
        cu2.metric("Last built", "—")
    if not cat.empty:
        bp_n = int((cat["convention"] == "bp").sum())
        price_n = int((cat["convention"] == "price").sum())
        unk_n = int((cat["convention"] == "unknown").sum())
        cu3.metric("bp / price / unk", f"{bp_n} / {price_n} / {unk_n}")
    else:
        cu3.metric("bp / price / unk", "—")
    with cu4:
        st.text_input("Catalog file", value=CATALOG_PATH, disabled=True)

    rebuild = st.button("Rebuild catalog (SRA)", key="units_rebuild")
    if rebuild:
        ph = st.empty()
        bar = st.progress(0)

        def cb(mkt, i, n):
            ph.caption(f"Detecting {mkt} … {i}/{n}")
            bar.progress(min(1.0, i / max(1, n)))

        new_cat = build_catalog(["SRA"], progress_cb=cb)
        ph.success(f"Built {len(new_cat):,} rows.")
        bar.empty()
        st.cache_data.clear()
        cat = load_catalog()

    if cat.empty:
        st.info("No catalog yet — click **Rebuild catalog (SRA)** to build it.")
    else:
        st.markdown("##### Breakdown")
        bk = cat.groupby(["base_product", "strategy", "convention"]).size().reset_index(name="count")
        st.dataframe(bk, use_container_width=True, hide_index=True, height=240)

        st.markdown("##### Filter & inspect")
        f1, f2, f3 = st.columns(3)
        with f1:
            mkts = ["(all)"] + sorted(cat["base_product"].unique().tolist())
            mkt = st.selectbox("Market", mkts, key="u_mkt")
        with f2:
            strats = ["(all)", "outright", "spread", "fly"]
            strat = st.selectbox("Strategy", strats, key="u_strat")
        with f3:
            convs = ["(all)", "bp", "price", "unknown"]
            conv = st.selectbox("Convention", convs, key="u_conv")
        view = cat.copy()
        if mkt != "(all)": view = view[view["base_product"] == mkt]
        if strat != "(all)": view = view[view["strategy"] == strat]
        if conv != "(all)": view = view[view["convention"] == conv]
        search = st.text_input("Search symbol", key="u_search").strip().upper()
        if search:
            view = view[view["symbol"].str.upper().str.contains(search)]
        st.caption(f"{len(view):,} rows")
        st.dataframe(view, use_container_width=True, hide_index=True, height=420)


# -----------------------------------------------------------------------------
# OHLC DB VIEWER
# -----------------------------------------------------------------------------
with tab_ohlc:
    st.subheader("OHLC DB Viewer")

    info = ohlc_db_path_info()
    if not info["path"]:
        st.error(f"No `market_data_v2_*.duckdb` snapshot found in {OHLC_SNAPSHOT_DIR}.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Resolved snapshot", Path(info["path"]).name)
        c2.metric("Size (GB)", info["size_gb"])
        c3.metric("Last modified", info["modified"].strftime("%Y-%m-%d %H:%M"))

    con = get_ohlc_connection()
    if con is None:
        st.stop()

    st.markdown("##### Tables and views")
    tables_df = list_ohlc_tables()
    st.dataframe(tables_df, use_container_width=True, hide_index=True, height=320)

    st.markdown("##### Schema explorer")
    if not tables_df.empty:
        sel = st.selectbox("Pick a table or view", tables_df["table_name"].tolist(), key="ohlc_table_pick")
        if sel:
            cdesc = describe_ohlc_table(sel)
            st.dataframe(cdesc, use_container_width=True, hide_index=True, height=240)
            n = st.number_input("Rows to preview", 1, 1000, 20, key="ohlc_preview_rows")
            try:
                preview = con.execute(f'SELECT * FROM "{sel}" LIMIT {int(n)}').fetchdf()
                st.dataframe(preview, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Preview failed: {e}")

    st.markdown("##### Custom SQL (read-only)")
    sql = st.text_area(
        "SQL",
        value="SELECT base_product, COUNT(DISTINCT symbol) AS symbols\nFROM mde2_contracts_catalog\nGROUP BY 1 ORDER BY symbols DESC LIMIT 20;",
        height=140,
        key="ohlc_sql",
    )
    limit = st.number_input("Result limit", 10, 100_000, 1000, key="ohlc_sql_limit")
    if st.button("Run query", key="ohlc_sql_run"):
        try:
            df = run_ohlc_sql(sql, limit=int(limit))
            st.success(f"{len(df)} rows returned")
            st.dataframe(df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Query failed: {e}")


# -----------------------------------------------------------------------------
# BBG PARQUET VIEWER
# -----------------------------------------------------------------------------
with tab_bbg:
    st.subheader("BBG Parquet Viewer")

    info = bbg_warehouse_path_info()
    c1, c2, c3 = st.columns(3)
    c1.metric("Parquet root", BBG_PARQUET_ROOT)
    if info["path"]:
        c2.metric("warehouse.duckdb", f"{info['size_kb']} KB")
        c3.metric("Last modified", info["modified"].strftime("%Y-%m-%d %H:%M"))
    else:
        c2.metric("warehouse.duckdb", "missing")

    cats = list_bbg_categories()
    if cats.empty:
        st.error(f"No categories found at {BBG_PARQUET_ROOT}.")
        st.stop()

    total_files = int(cats["files"].sum())
    total_size = round(cats["size_mb"].sum(), 1)
    st.caption(f"**{len(cats)}** categories · **{total_files:,}** parquet files · **{total_size} MB** total")

    st.markdown("##### Categories")
    st.dataframe(cats, use_container_width=True, hide_index=True, height=280)

    st.markdown("##### Ticker explorer")
    cat_sel = st.selectbox("Category", cats["category"].tolist(), key="bbg_cat_pick")
    if cat_sel:
        tickers = list_bbg_tickers(cat_sel)
        st.caption(f"{len(tickers):,} tickers in `{cat_sel}/`")
        search = st.text_input("Search ticker", key="bbg_ticker_search").strip().upper()
        view = tickers
        if search:
            view = view[view["ticker"].str.upper().str.contains(search)]
        st.dataframe(view, use_container_width=True, hide_index=True, height=320)

        if not view.empty:
            ticker_sel = st.selectbox("Inspect a ticker", view["ticker"].tolist(), key="bbg_ticker_pick")
            if ticker_sel:
                df = read_bbg_parquet(cat_sel, ticker_sel)
                if df.empty:
                    st.warning("Empty parquet.")
                else:
                    st.markdown("**Schema**")
                    schema_df = pd.DataFrame({"column": df.columns, "dtype": [str(t) for t in df.dtypes]})
                    st.dataframe(schema_df, use_container_width=True, hide_index=True, height=200)

                    st.markdown(f"**First & last rows** (total {len(df):,})")
                    head_n = st.number_input("Head/tail rows each", 1, 50, 5, key="bbg_headtail")
                    cc1, cc2 = st.columns(2)
                    with cc1:
                        st.caption("Head")
                        st.dataframe(df.head(int(head_n)), use_container_width=True, hide_index=True)
                    with cc2:
                        st.caption("Tail")
                        st.dataframe(df.tail(int(head_n)), use_container_width=True, hide_index=True)

                    st.markdown("**Numeric summary**")
                    try:
                        st.dataframe(df.describe().T, use_container_width=True)
                    except Exception:
                        st.caption("(no numeric columns)")

    st.markdown("##### Custom DuckDB query against BBG parquets")
    st.caption(
        "Use `read_parquet('D:/BBG data/parquet/<category>/*.parquet', union_by_name=true)`. "
        "DuckDB needs forward-slashes in the path string."
    )
    sql = st.text_area(
        "SQL",
        value=(
            "SELECT _ticker, MIN(date) AS first_date, MAX(date) AS last_date, COUNT(*) AS rows\n"
            "FROM read_parquet('D:/BBG data/parquet/eco/*.parquet', union_by_name=true)\n"
            "WHERE _ticker IN ('CPI YOY Index','NFP TCH Index','FDTR Index')\n"
            "GROUP BY 1 ORDER BY 1;"
        ),
        height=160,
        key="bbg_sql",
    )
    if st.button("Run query", key="bbg_sql_run"):
        try:
            con_mem = get_bbg_inmemory_connection()
            df = con_mem.execute(sql).fetchdf()
            st.success(f"{len(df)} rows returned")
            st.dataframe(df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Query failed: {e}")
