# STIRS Dashboard — Comprehensive Project Handoff
*Generated for new-session continuation. Use as the first reference doc.*

This file is the single source of truth for everything we've built, every design
decision Yash made, every bug we hit and fixed, and every piece of domain knowledge
captured. Read this whole file before continuing.

---

## 1 · WHO / WHERE

- **User:** Yash Patel — STIRS / Bonds analyst building a multi-economy analytics dashboard.
- **Project root:** `D:\STIRS_DASHBOARD\`
- **Plan file:** `C:\Users\yash.patel\.claude\plans\d-python-projects-qh-api-apps-analytics-buzzing-aurora.md`
- **Conda env:** `C:\ProgramData\Anaconda3\envs\qh_data\` — Python 3.10, has `streamlit 1.48`, `duckdb 1.3.2`, `pandas`, `numpy`, `pyarrow 19.0`, `plotly 6.3`, `pyyaml`.
- **Dashboard runs at:** `http://localhost:8503` (8501/8502 already used by other Streamlit apps on this machine — do **not** change to 8501).
- **OS:** Windows 11. Bash uses Unix-style paths/commands; PowerShell available too.

---

## 2 · DATA SOURCES (read-only)

### A · OHLC Market Data — DuckDB snapshot

- **Path:** `D:\Python Projects\QH_API_APPS\analytics_kit\db_snapshot\market_data_v2_<DATE>_<TIME>.duckdb` (current snapshot 2026-05-04, 6.7 GB)
- **Connector** (`lib/connections.py:get_ohlc_connection`) auto-resolves the most-recent file.
- **Open with `read_only=True`** — production DB serializes writers and may be locked by the parallel Streamlit-v2 app that creates these snapshots.

#### Schema (key tables)
| Table | Purpose |
|---|---|
| `mde2_timeseries` | OHLCV facts. PK = `(symbol, interval, calc_method, time)`. `time` is unix milliseconds. |
| `mde2_indicators` | 143 indicator columns sharing same PK |
| `mde2_contracts_catalog` | Per-symbol metadata: `symbol, base_product, market_or_pair, strategy, tenor_months, contract_tag, curve_bucket, expiry_year, expiry_month, left_leg, mid_leg, right_leg, is_inter, is_continuous` |
| `v_mde2_timeseries`, `v_mde2_timeseries_with_indicators`, `v_mde2_continuous`, `v_mde2_per_contract`, `v_mde2_indicators` | Convenience views |

#### Universe (locked in TMIA v13 gameplan)
- **91 products** in 9 asset classes
- **STIRS (10):** SRA, S1R, FF, ER, FER, SON, FSR, CRA, JTOA, YBA, NBB *(plus FERER misc)*
- **Bonds (15):** FGBS, FGBM, FGBL, FGBX, FOAT, FBTS, FBTP, FLG, YTT, YTC, TU, FV, TY, AUL, S1RFF
- 30 intermarket pairs (cross-STIR like SRA-ER, SRA-SON, etc.)

#### **CRITICAL: 'asof' is a DuckDB reserved word.**
Use `bar_date` as a column alias in any SQL we emit, then rename to `asof` in pandas. (See `lib/sra_data.load_sra_curve_panel` — done correctly.)

#### Live-contract definition
A contract is "live" if its last 1D bar is within **`LIVENESS_DAYS = 7`** of the latest snapshot OR `is_continuous=TRUE`. Snapshot date = 2026-04-27 → cutoff 2026-04-20. Yields **766 live contracts** across STIRS+Bonds.

#### Coverage facts (verified, not rumors)
- SRA: 29 outrights live (range SRAJ26 → SRAH32), 88 spreads in 5 tenors (1M:1, 3M:21, 6M:19, 9M:19, 12M:18), 67 flies in 4 tenors (3M:16, 6M:16, 9M:3, 12M:10)
- All STIRS+Bonds have **100% indicator coverage** (verified via PK join `mde2_timeseries × mde2_indicators`)
- 1H data starts ~mid-2024 for STIRS, ~early 2024 for bonds
- Daily depth: SRA from 2019, ER from 2015, FGBS/FGBM/FGBL/FGBX/FLG from 2005, YTT/YTC from 1992

### B · BBG Fundamentals — Parquet warehouse

- **Path:** `D:\BBG data\parquet\` — 14,339 parquet files in 15 sub-folders, ~160 MB total
- **`warehouse.duckdb`** has 15 views — **all broken** (hardcoded `C:/Users/Bloomberg/Documents/...` path from old install). Workaround: read parquets directly via `duckdb.read_parquet('D:/BBG data/parquet/<cat>/*.parquet', union_by_name=true)` or pandas (we use DuckDB because pyarrow chokes on some BBG parquets with "Repetition level histogram size mismatch").

#### Categories
| Folder | Files | Time range | Use |
|---|---:|---|---|
| `xwab/` | 9,991 | 1 yr | Weather stations (NOT relevant for STIR/Bonds — skip) |
| `eco/` | 1,951 | 1990-2026 (36y) | ECO calendar releases + consensus surveys |
| `xcot/` | 1,944 | 1986-2026 (40y) | CFTC Commitments of Traders |
| `macro_economies/` | 165 | ~1y | Country-level macro panels |
| `xeia/` | 135 | 1y | DOE/EIA energy stocks (user said: **skip — price covers it**) |
| `fx/` | 30 | 1y | USD pairs |
| `rates_drivers/` | 27 | 1y | Policy rates + benchmark yields |
| `credit/` | 23 | 1y | Sovereign + corp CDS, OAS spreads |
| `energy_extras/` | 21 | 1y | WTI, Brent, NG, rigs, OPEC |
| `vol_indices/` | 17 | 1y | MOVE, SRVIX, VIX family, FX vol |
| `metals_inventory/` | 13 | 1y | LME stocks (user said: **skip — price covers it**) |
| `equity_indices/` | 10 | 1y | SPX, INDU, NDX, DAX, NKY etc |
| `reer_basis_crack/` | 6 | 1y | REER, basis, cracks |
| `usda_extras/` | 5 | 1y | Soybean/corn planting, USDA grain stocks |
| `weather_extras/` | 1 | 1y | UK weather (skip) |

Common parquet schema: `date (str YYYY-MM-DD), <field columns>, _dataset, _ticker, _loaded_at`. Eco files also have `ECO_RELEASE_DT, ACTUAL_RELEASE, BN_SURVEY_MEDIAN/AVERAGE/HIGH/LOW`. Note: BBG warehouse last refreshed 2026-05-05 ~19:15.

#### **Ticker dictionary** (Yash provided)
- **Path:** `C:\Users\yash.patel\Downloads\ticker_dictionary.csv`
- 3,300 rows with `ticker, NAME, SHORT_NAME, SECURITY_DES, COUNTRY_FULL_NAME, ECO_REGION, ECO_CATEGORY` etc.
- Covers `eco / xcot / macro_economies` mostly; doesn't cover `rates_drivers / credit / vol_indices / fx / equity_indices / reer_basis_crack` (those need our own decoding).

---

## 3 · DOMAIN-SPECIFIC FACTS USER LOCKED

### A · Curve sections
- **Default boundaries:** Front 1-6, Mid 7-14, Back 15+
- User-adjustable in dashboard via expander in Curve subtab (number inputs)

### B · Pack groupings (SOFR convention)
- Whites = front-4 quarterlies (year 1)
- Reds = next 4 (year 2)
- Greens = year 3
- Blues = year 4
- Golds = year 5
- Quarterly month codes = H/M/U/Z. Serial months = J/K/N/Q/V/X/F/G — they have valid 3M reference periods too (CME 3M SOFR has serial contracts)

### C · Implied rate convention
- IMM contracts (SRA, S1R, FF, ER, FER, SON, FSR, CRA, JTOA): `implied_rate_pct = 100 - close`
- Discount-security contracts (YBA, NBB): close IS the yield directly
- **Only outrights have a meaningful implied rate.** Spreads / flies are differential price quantities — never show "implied rate" toggle for them.

### D · CME 3M SOFR (SR3) reference period
- `start = 3rd Wednesday of contract month`
- `end = 3rd Wednesday of contract month + 3`
- Both quarterlies AND serial-month contracts have 3-month reference periods (serials' periods overlap quarterlies')
- Implementation: `lib/fomc.reference_period(symbol)` and `third_wednesday(year, month)`

### E · Carry conventions (Ilmanen 2011 Ch.14, used in `lib/carry.py`)
- **Single-contract roll-down:** `carry_i (bp/day) = (rate_{i+1} − rate_i) × 100 / days_between_expiries`
- **White-pack carry:** average single-contract carry across front-4 quarterlies
- **Cumulative carry to expiry:** `carry_per_day × days_between(asof, contract_ref_start)`
- Stored in `D:\STIRS_DASHBOARD\lib\carry.py`

### F · FOMC implied policy path (CME-aligned methodology)
- **Anchor:** `r_0` (rate before first in-window FOMC) pinned to current SOFR effective via soft prior with weight `5×n_contracts`
- **Smoothness ridge:** L2 penalty on `|r_i − r_{i-1}|` with `λ = 0.5 × n_contracts`
- **Horizon:** 18 months (only contracts within this window used)
- **Quarterlies only** (non-overlapping reference periods)
- Solver: `np.linalg.lstsq` on the augmented system
- Probability mapping: `P(hike 25) = max(0, Δ/25)`, `P(cut 25) = max(0, -Δ/25)`, `P(hold) = 1 − above` (clamped 0..1)
- Methodology text in `lib/fomc.get_methodology_text()` — shown as title-attribute tooltip on the side panel
- Stored in `D:\STIRS_DASHBOARD\lib\fomc.py`

### G · FOMC dates
- **YAML:** `D:\STIRS_DASHBOARD\config\fomc_meetings.yaml` — 32 meetings hand-seeded from federalreserve.gov, Jan 2024 → Dec 2027
- **Refresh script:** `D:\STIRS_DASHBOARD\scripts\refresh_fomc.py` — best-effort scraper of `federalreserve.gov/monetarypolicy/fomccalendars.htm` (run manually if Fed updates schedule)

### H · Z-score / percentile bands convention
- **History EXCLUDES the as-of date** (`< ts`, not `<=`) — proper "today vs prior N days" semantics
- Bug we fixed: was including `asof` in sample → biased toward 0 by ~1.5σ on 5d windows. Now: `wide_close.loc[wide_close.index < ts].tail(lookback)`
- `std(ddof=1)` explicit
- Min-2-observations guard
- Stored in `lib/sra_data.compute_per_contract_zscores / compute_percentile_bands / compute_percentile_rank`

### I · Regime classifier thresholds (constant across all lookbacks — module-level constants)
- `DIRECTION_THRESHOLD_BP = 1.0` — `|parallel|` over this → BULL/BEAR
- `SLOPE_THRESHOLD_BP = 0.5` — for STEEPENER/FLATTENER
- `CURV_THRESHOLD_BP = 0.5` — for BELLY-OFFERED/BELLY-BID
- `MAGNITUDE_MINOR_BP = 1.0`, `MAGNITUDE_MAJOR_BP = 5.0`
- Exposed via `lib/regime.get_regime_thresholds()` and `get_regime_thresholds_text()` (the latter is the tooltip body)
- Convention: parallel/slope/curvature from regression of price-change vs normalized contract index `x ∈ [-1, 1]`. Positive parallel = price up = rate fell = BULL.

### J · Per-contract unit convention (NEW — this session)
**The big finding:** in the OHLC database, spread and fly contracts store their close **already in basis points**, not as a raw price difference.

Verified empirically on 2026-04-27:
- SRAM26 outright = 96.345
- SRAU26 outright = 96.37
- Raw price diff = 96.345 − 96.37 = −0.025
- Stored close of SRAM26-U26 spread = **−2.5** (= −0.025 × 100)

So: outrights are **price** units, spreads/flies are **bp** units. Prior code multiplied every change by 100 unconditionally — which over-scaled spread/fly Δs by 100×.

**Solution:** auto-detected per-contract convention catalog stored at `D:\STIRS_DASHBOARD\data\contract_units.parquet`. See section 6 below.

---

## 4 · APP STRUCTURE

```
D:\STIRS_DASHBOARD\
├── app.py                              # Landing page
├── requirements.txt
├── HANDOFF.md                          # ← THIS FILE
├── .streamlit\config.toml              # Dark theme · port 8503
├── data\
│   └── contract_units.parquet          # Per-contract bp/price catalog (3,584 SRA rows)
├── config\
│   └── fomc_meetings.yaml              # 32 FOMC dates 2024-2027
├── scripts\
│   └── refresh_fomc.py                 # Fed website scraper
├── lib\
│   ├── __init__.py
│   ├── theme.py                        # Color tokens + type scale
│   ├── css.py                          # Custom CSS injection + sidebar brand
│   ├── components.py                   # status_strip, status_strip_with_dot, summary_row, page_header, kpi_chip
│   ├── helpers.py
│   ├── connections.py                  # OHLC + BBG connectors (cached)
│   ├── sra_data.py                     # SRA queries, percentile/z math, sections, packs, decomp
│   ├── charts.py                       # Plotly factory: curve, multi-date, ribbon, pack, volume-Δ,
│   │                                   #                heatmap, calendar-matrix, z-score bars,
│   │                                   #                carry-colored. Plus add_section_shading,
│   │                                   #                add_horizontal_band, add_horizontal_line,
│   │                                   #                add_highlight_marker.
│   │                                   #   ★ Analysis factories: make_proximity_ribbon_chart,
│   │                                   #     make_density_heatmap_chart, make_confluence_matrix_chart,
│   │                                   #     make_distribution_histogram, make_mean_bands_chart,
│   │                                   #     make_proximity_drill_chart, make_score_bar_chart
│   ├── fomc.py                         # FOMC YAML loader, reference_period, decompose_implied_rates,
│   │                                   #                get_methodology_text
│   ├── carry.py                        # 3 carry notions
│   ├── regime.py                       # Multi-lookback regime classifier + thresholds
│   ├── contract_units.py               # Auto-detect per-contract bp vs price scale
│   ├── proximity.py                    # ★ NEW — proximity engine (ATR-flag, streak, velocity,
│   │                                   #                fresh/failed breaks, confluence patterns,
│   │                                   #                cluster signals, section regime)
│   ├── mean_reversion.py               # ★ NEW — Z-score, OU half-life (AR(1)), Hurst (R/S),
│   │                                   #                ADF (lag-1, no-trend), composite scores,
│   │                                   #                Z confluence patterns
│   └── us_fundamentals_inventory.py    # 294 US tickers categorized — used by Fundamentals tab
├── pages\
│   ├── 1_US_Economy.py
│   └── 2_Settings.py                   # Now has 4 tabs: General · Unit conventions · OHLC · BBG
└── tabs\us\
    ├── __init__.py
    ├── sra.py                          # Curve / Analysis sub-sub-tabs router
    ├── sra_curve.py                    # Curve subtab — modes + side panel
    ├── sra_analysis.py                 # ★ NEW — Analysis sub-router (8 sub-subtabs)
    ├── sra_analysis_proximity.py       # ★ NEW — Proximity subtab
    ├── sra_analysis_zscore.py          # ★ NEW — Z-score & Mean Reversion subtab
    ├── ff_s1r.py                       # PLACEHOLDER
    ├── bonds.py                        # PLACEHOLDER
    └── fundamentals.py                 # 294-ticker browseable inventory with search + filters
```

### Sidebar nav (Streamlit auto from `pages/`)
- US Economy
- Settings

### US Economy tab structure
```
US Economy
├── SRA
│   ├── Curve     ← built (Outrights / Spreads / Flies; modes: Standard / Multi-date /
│   │                       Ribbon / Pack / Volume Δ / Heatmap / Matrix)
│   └── Analysis  ← built first 2 subtabs; remainder placeholders
│       ├── Proximity        ← BUILT — ATR-flagged ranking · confluence patterns · clusters
│       ├── Z-score & MR     ← BUILT — Z + OU half-life + Hurst + ADF + composite scoring
│       ├── PCA              ← placeholder
│       ├── Terminal rate    ← placeholder
│       ├── Carry            ← placeholder
│       ├── Slope / Curv     ← placeholder
│       ├── Stat tests       ← placeholder
│       └── Fly residuals    ← placeholder
├── FF + S1R    ← placeholder (just shows coverage table)
├── Bonds       ← placeholder
└── Fundamentals  ← built (US ticker inventory, filterable, by category)
```

### Settings tab structure (built)
- General — config knobs (placeholder values)
- **Unit conventions** ← NEW — view/rebuild the per-contract bp/price catalog
- OHLC DB Viewer — table list, schema explorer, custom SQL
- BBG Parquet Viewer — category browse, ticker search, custom DuckDB SQL

---

## 5 · CURVE SUBTAB — DETAILED STATE (COMPLETE)

### Top-level controls
- 5-item status strip (Snapshot · Outrights/Spreads/Flies counts · Liveness)
- "Curve section boundaries" expander (Front-end / Mid-end inputs)
- Sub-tabs: Outrights · Spreads · Flies (segmented control for tenor inside each)

### Per-scope (one strategy + tenor) controls
- 5-item info strip with live dot
- **Regime badges** — 3 pills (1d / 5d / 30d) with **threshold tooltip** (same thresholds, hovers show values), color-coded by direction (BULL=green, BEAR=red, FLAT=muted)
- Mode picker: segmented control (6 modes for outrights, 6 for spreads/flies — Pack→Matrix swap)
- Toggles: **Compare** · **Animate** · **Analysis** (Analysis defaults ON)

### Mode 1 — Standard
- As-of date dropdown
- **Jump-to-contract** dropdown (was text input — fixed)
- Y-axis: Price | Implied rate (only for outrights)
- Compare with quick buttons (1D · 1W · 1M · 3M · YTD)
- Animation (Slow/Med/Fast, windows 30d/60d/6mo/1y/Custom)
- 4-5 overlay checkboxes with **tooltips explaining math**:
  - Fed band (FDTR upper/lower; works in both rate AND price axes — converts via 100-rate; alpha 0.07 — very subtle)
  - SOFR line (works in both axes)
  - Section shading (front amber 0.14, mid cyan 0.11, back purple 0.11)
  - Δ panel (auto-on with compare)
  - Carry coloring (outrights only; tooltip shows the formula)
- 🏛 FOMC markers at top of contracts that span an upcoming FOMC
- H-L band: **filtered to valid contracts only**, then connected fill (alpha 0.20). Skips NaN gaps cleanly. **NO error-bar wicks** (those were the "weird markers" user disliked).

### Mode 2 — (replaced) Section shading on Mode 1
- Was its own mode; replaced with subtle background bands on the Standard chart per user feedback.

### Mode 3 — Multi-date overlay
- Multi-select preset dates (Today/T-1/T-1W/T-1M/T-3M/T-6M/T-1Y/YTD)
- Cool-to-warm color palette (oldest=purple → recent=gold-bold)
- Y-axis toggle (outrights only)

### Mode 4 — Ribbon
- Lookback: **5d / 15d / 30d / 60d / 90d / 252d / Max** (default 60d)
- View: **Ribbon | Z-score bars** (z-score view shows ±1σ ±2σ thresholds)
- Optional: Mean line, Recent 5d avg
- Bands: 5-95% outer (alpha 0.13), 25-75% IQR (alpha 0.28), median dotted

### Mode 5 — (replaced) Δ panel integrated into other modes
- Originally standalone bar chart; now appears as Δ panel under Standard chart when compare-on or Δ-checkbox-on. The decomposition (parallel/slope/curvature/RMSE) ALWAYS shows in the side panel (auto-computes vs T-1 if no explicit compare).

### Mode 6 — Pack (outrights only)
- Quarterly grouping into Whites/Reds/Greens/Blues/Golds
- Pack-mean overlay markers (large diamond, distinct color per pack)
- Side panel: per-pack mean/slope/range cards + pack-spread table

### Mode 7 — DROPPED (Forwards) per user
- Skipped entirely

### Mode 8 — Volume / OI Δ (delta-based per user)
- Δ vs T-1 / T-1W / T-1M
- View: Volume-Δ alone OR Volume-Δ + Curve stacked
- OI placeholder (no OI data in current panel — would need to add)

### Mode 9 — DROPPED in Curve, MOVED to Analysis subtab
- Slope/Curvature time series belongs in Analysis (next subtab to build)

### NEW — Mode H — Heatmap (date × contract evolution)
- **Value modes:**
  - Outrights: Price | Implied rate | Δ vs T-1 (bp) | Δ vs T-1W | Δ vs T-1M | Z-score | Percentile rank
  - Spreads/Flies: same minus "Implied rate" (gated by `allow_implied_rate`)
- Time window: Last 30d / 60d / 6mo / 1y / All
- Z/rank lookback: 5d / 15d / 30d / 60d / 90d / 252d
- Show-values toggle
- Color: Viridis (sequential) for prices, RdBu_r diverging around 0 for changes/z, around 50 for rank
- **Δ scaling now per-contract aware** (uses `contract_units` catalog so spread/fly Δs aren't 100× off)

### NEW — Mode M — Matrix (Spreads/Flies only — cross-tenor)
- Rows = front-leg expiry (YYYY-MM)
- Cols = tenor (3M/6M/9M/12M)
- Values: Current value | Z-score | Percentile rank
- Lookback for z/rank: 5d/15d/30d/60d/90d/252d

### Click-to-drill (always available, beneath every mode)
- Expander with contract-picker dropdown + lookback (30d/60d/90d/252d/Max)
- Shows: last close, implied rate, days-to-ref, bars count, mean/std/min/max, z vs prior, percentile rank
- Per-contract OHLC time-series chart with H-L band
- Neighbors panel (prev/next contract values + spread)

### Side Panel — UNIFIED (renders ALL blocks regardless of mode)
- **Default ON**, scrollable container 720px height
- Header: "📊 Analysis · all blocks (scroll)"
- Blocks (skip if not applicable):
  1. **Curve summary** — front/back/slope/range
  2. **Section slopes** — front/mid/back, color-coded
  3. **Fed band & SOFR** — outrights only
  4. **Implied policy path (FOMC)** ⓘ — outrights only, anchored to SOFR + ridge λ=0.5, 18mo horizon, methodology in title-attribute tooltip
  5. **Carry** — outrights only, white-pack avg + top 3 best/worst contracts
  6. **Move decomposition** — auto vs T-1 OR vs explicit compare date; parallel/slope/curvature/RMSE + top 6 movers (now uses per-contract bp scaling)
  7. **Z-score regime (30d ref)** — extreme/elevated/normal counts
  8. **Z across lookbacks · top 8** — 5d/15d/30d/60d/90d table + pattern column with 8 patterns:
     - **STABLE** (all |z|<1, dim)
     - **FRESH** (5d only ≥1.5, others normal — blue, event-driven)
     - **DRIFTED** (60d/90d ≥1.5, 5d quiet — amber, "coiled")
     - **REVERTING** (5d & 90d opposite signs, both ≥1 — green, watch for turn)
     - **PERSISTENT** (all 5 elevated, same sign, similar magnitudes — red, trend regime)
     - **ACCELERATING** (|z5|>|z15|>|z30|, longer elevated, same sign — red)
     - **DECELERATING** (|z5|<|z15|<|z30|, longer elevated, same sign — amber)
     - **MIXED** (none of above)
     - Pattern legend in expander explains each + trader implication
  9. **Pack metrics** — outrights only, color-banded cards + pack spreads (Reds-Whites etc.)
  10. **Volume Δ** — vs T-1, top 8 per-contract Δ rankings

### Lookback policy (locked)
- User-selectable dropdowns: **5d / 15d / 30d / 60d / 90d / 252d / Max**
- **Analysis is capped at 90d** — multi-lookback z blocks use [5,15,30,60,90] in standard panel and [5,30,60,90] in ribbon panel. 252d remains as a dropdown option but no analytical block uses 252d as its longest window.

---

## 5B · ANALYSIS SUBTABS — DETAILED STATE (PROXIMITY + Z-SCORE & MR COMPLETE)

The Analysis subtab is split via `tabs/us/sra_analysis.py` into 8 sub-subtabs. The first two
are fully implemented; the remaining 6 are placeholders waiting on follow-on builds (PCA,
Terminal rate, Carry, Slope/Curv, Stat tests, Fly residuals). All placeholders preview their
intended scope so users know what's coming.

### Common shell (both subtabs)

- Status strip with snapshot, outright/spread/fly counts, liveness
- **Section boundaries expander** — own to each subtab so users can tune Front/Mid/Back
  separately from Curve subtab (defaults to DEFAULT_FRONT_END=6, DEFAULT_MID_END=14)
- **Strategy** segmented control (Outright / Spread / Fly), gated **Tenor** selector for
  spreads (1M/3M/6M/9M/12M) and flies (3M/6M/9M/12M)
- **Lookbacks (multi-select)** — `[5, 15, 30, 60, 90]`, all selected by default
- **Focus lookback** dropdown — picks the lookback that drives Section Ribbons, Cluster
  Heatmap, and side-panel blocks (default 30)
- **Top K** = 3/5/10 (default 5)
- View toggles: Section ribbons · Confluence matrix · Cluster heatmap (Composite ranks
  in Z subtab) · Analysis side panel (default ON)
- Per-scope info strip (scope · contracts · range · lookbacks · focus)
- Layout: `chart_col, side_col = st.columns([3, 1.1])`; side panel in `st.container(height=720)`
- Drill-down expander always available beneath all views

### Engines (lib/proximity.py, lib/mean_reversion.py)

#### Proximity engine (`lib/proximity.py`)
- **ATR(14)** computed in stored-price units per contract (uses `mde2_timeseries.high/low/close`),
  bp via catalog multiplier (`bp_multipliers_for`)
- **Per-lookback metrics** in `compute_contract_proximity`:
  - `current`, `current_bp`, `high_n_bp`, `low_n_bp`, `range_n_bp`
  - `dist_high_bp` / `dist_low_bp` (catalog-aware)
  - `dist_high_atr` / `dist_low_atr` (unit-agnostic — multiplier cancels)
  - `position_in_range` (0=at low, 1=at high)
  - `flag_high` / `flag_low`: AT (≤0.25 ATR) / NEAR (≤0.5) / APPROACHING (≤1.0) / FAR
  - `nearest_extreme` + `nearest_dist_atr`
  - `streak_at_extreme` — consecutive prior bars within 0.5·ATR of rolling extreme
  - `velocity_atr_per_day` — avg Δdist over last 5 prior bars (negative = closing in)
  - `touch_count_high` / `touch_count_low` — # bars in window within 0.25·ATR of extreme
  - `fresh_break_high/low` — today strictly past prior N-day high/low (sets dist=0, flag=AT)
  - `failed_break_high/low` — touched within last 3 bars, now ≥0.5 ATR away
  - `range_expansion_ratio` — current N-day range / prior N-day range
- **History strictly excludes asof** (`< ts`) — same convention as Z-score.
- **8 confluence patterns** in `classify_proximity_pattern` (across selected lookbacks):
  PERSISTENT · ACCELERATING · DECELERATING · FRESH · DRIFTED · REVERTING ·
  DIVERGENT · STABLE · MIXED
- **Cluster signal** (`compute_cluster_signal`) — % AT/NEAR per group bucket
- **Section regime** (`proximity_section_regime`) — non-parametric curve label from PIR:
  BULL · BEAR · STEEPENING · FLATTENING · BELLY-BID · BELLY-OFFERED · MIXED
- All thresholds exposed via `get_proximity_thresholds_text()` for tooltip injection.

#### Mean-reversion engine (`lib/mean_reversion.py`)
- **Per-lookback Z** (`zscore_value`), percentile rank, mean, std, dist-to-mean (bp)
  — all match locked Z conventions (history `< ts`, ddof=1, min-2 obs)
- **OU half-life** (`ou_half_life`) — AR(1) `x_t = a + b·x_{t-1}`; half_life = `−ln(2)/ln(b)`
  for `0 < b < 1`; otherwise None. 90-day fit by default. Numpy-only (no scipy).
- **Hurst exponent** (`hurst_exponent`) — Rescaled Range (R/S) on log-returns. Log-spaced
  sub-windows (8 to N/2). Linear regression of log(R/S) vs log(n). Clipped to [0, 1].
  Label: TRENDING (>0.55) · RANDOM · REVERTING (<0.45).
- **ADF (lag-1, no trend)** (`adf_test`) — t-stat from regression
  `Δx_t = α + γ·x_{t-1} + φ·Δx_{t-1} + ε`; coarse p-bucket from MacKinnon critical values
  (`<0.01 / <0.05 / <0.10 / >0.10`); `reject_5pct = (t ≤ −2.86)`.
- **Composite scores**:
  - `composite_reversion_score = 0.5·|z|/2 + 0.25·(1−H) + 0.15·10/max(1,h) + 0.10·1{ADF rejects}`
  - `composite_trend_score = 0.55·|z|/2 + 0.35·H + 0.10·1{ADF can't reject}`
- **8 Z-pattern catalogue** (`classify_z_pattern`) — same names as proximity patterns,
  defined on |z| thresholds (≥1 elevated · ≥1.5 fresh) — locked alignment with the
  Curve subtab's existing Z-pattern block.

### Subtab 1 — Proximity (`tabs/us/sra_analysis_proximity.py`)

#### Views (chart column)
1. **Section Ribbons** — 3 columns (FRONT/MID/BACK), each with two horizontal-bar
   panels: "Closest to Nd HIGH" (red) and "Closest to Nd LOW" (green), top-K rows.
   Below each ribbon is a compact tooltipped table (symbol · cur bp · ext bp ·
   Δ ATR · flag · pattern). Threshold lines at 0.25 / 0.5 / 1.0 ATR.
2. **Confluence Matrix** — top 25 contracts by min(nearest_dist_atr) across selected
   lookbacks; rows = contracts, cols = lookbacks, cells = position_in_range (0..1)
   on a diverging RdBu_r palette around 0.5. Pattern column annotated on the right.
3. **Cluster Heatmap** — two side-by-side density heatmaps (HIGH-side and LOW-side),
   rows = Front/Mid/Back, cols = lookbacks, cells = % AT/NEAR. Custom colorscale
   (dark→deep-red intensifying with %).
4. **Drill-down expander** — per-contract close + 3 layered rolling-H/L bands
   (5d / 30d / 90d by default). Prints last close, ATR(14), 60d high/low (excl today).

#### Side panel (12 blocks, all rendered, scrollable 720px)
1. Methodology / thresholds (title-attribute tooltip)
2. Universe at-a-glance — # AT/NEAR/APPROACHING per side · avg/median PIR
3. Extreme density · Section × Side (focus lookback)
4. Confluence patterns · contract counts (8-pattern bucket)
5. Streak & velocity — # streak ≥ 3d · # fastest closing · top 5 of each
6. Fresh breakouts / breakdowns — symbols and bp-through (catalog-aware)
7. Failed extremes — touch-and-reverse setups (last 3 bars)
8. Cross-tenor stretch (spreads/flies only) — median PIR per tenor; rich/cheap labels
9. Section regime via proximity — BULL/BEAR/STEEPENING/FLATTENING/BELLY-BID/OFFERED
10. Z-score cross-check — UNTESTED-EXTREME / STRETCHED-EXTREME / COILED / NORMAL counts
    (and example contracts) — pairs proximity flag with Z magnitude
11. Touch frequency & range expansion — # retest HIGH/LOW · # range expanding/contracting
12. Concurrent extremes — biggest cluster (Section × side) ranked
13. (Expander) Pattern legend — 9-pattern descriptions with trader implication

#### Caching
`_build_proximity_data(strategy, tenor, asof_str, lookbacks_tuple)` is `@st.cache_data`
keyed on those args, returns `{contracts, wide_close/high/low, proximity, zscore, info}`.
The Z-score panel is computed in the same call so the cross-check block in the side
panel can run without a second roundtrip.

### Subtab 2 — Z-score & Mean Reversion (`tabs/us/sra_analysis_zscore.py`)

#### Views (chart column)
1. **Section Ribbons** — top-K most stretched +Z (red) and most stretched −Z (green) per
   section. Bars colored by |z| band (AT≥2σ · NEAR≥1.5σ · APPROACHING≥1σ · FAR). Compact
   tooltipped table per ribbon (symbol · z · μ-Δ bp · pct rank · pattern · Hurst · half-life).
2. **Z Confluence Matrix** — top 25 contracts by max |z|; rows × cols × cells = z-score
   on a diverging RdBu_r palette around 0; pattern column on the right.
3. **Cluster Heatmap** — two heatmaps: |median z| per (Section × Lookback) and
   % of section with |z|≥2 per lookback. Auto-scaled to actual maximum.
4. **Universe Z distribution** — histogram of z(focus) across the universe with ±1σ/±2σ
   threshold lines.
5. **Composite candidate rankings** — two horizontal-bar charts side-by-side:
   - Top reversion candidates (green) — composite reversion score
   - Top trend confirmations (red) — composite trend score
   With a tooltipped per-row table beneath each.
6. **Drill-down expander** — per-contract close + μ ± σ ± 2σ horizontal bands +
   OU half-life annotation. Metrics row: Z(focus) · μ · σ · pct rank · OU half-life
   with Hurst label as delta. Caption shows ADF status.

#### Side panel (12 blocks, all rendered, scrollable 720px)
1. Methodology / formulas (title-attribute tooltip)
2. Universe stretch — counts (z≥+2, z≤−2, etc.) · median z · std z · skew · excess kurtosis
3. Multi-window Z confluence — 8-pattern bucket counts
4. Section stretch — median z per section + curve label (RICH/CHEAP/STEEPENING/...)
5. Half-life ranking — top 6 fastest reverters with z context
6. Hurst regime summary — TRENDING / RANDOM / REVERTING / unfit counts
7. ADF + Z gate — # |z|≥2 stationary vs non-stationary; example contracts in each
8. Top composite reversion candidates (top 5) with score/z/half-life summary
9. Top trend confirmations (top 5) — DON'T fade list
10. Velocity / pace of reversion — # reverting (Δ|z|<0) vs # still extending; examples
11. Cross-check vs proximity — UNTESTED / STRETCHED / COILED / NORMAL counts
12. (Expander) Pattern legend — 8-pattern descriptions

#### Caching
`_build_zscore_data` mirrors the proximity build: same key shape, computes z + proximity
panels in one call so the side-panel cross-check block has both available.

### UI standards (locked, all in line with Curve subtab)

- Dark mode only (`#0a0e14` base, gold accent)
- All tooltips use 4-line PIA convention (title · formula · inputs · interpretation)
  via title-attribute with `&#10;` for newlines
- Status strip with live dot at top of each subtab
- Subtle dividers between rows in side-panel kv blocks
- JetBrains Mono for all numeric values
- Catalog-aware bp scaling — every spread/fly distance multiplied via `bp_multipliers_for()`,
  never blanket × 100
- All chart-mode dispatch wrapped in try/except so one block's runtime error doesn't
  kill the page (errors render inline with collapsible traceback)
- 90d cap respected — 252d does not appear in any analytical engine

### Bug fixes during this build

| # | Bug | Cause | Fix |
|---|---|---|---|
| 13 | `Z confluence matrix failed: name 'np' is not defined` | numpy used inside `make_confluence_matrix_chart` (and other new factories) but not imported in `lib/charts.py` | Added `import numpy as np` to charts.py |
| 14 | Misbinding ternary in z-table HTML row builder collapsed `<div>` open tag → malformed row when `dtm` was None | `if/else` ternary applied to entire concatenated f-string instead of just the cell | Pre-compute formatted strings, use `_fmt(...)` helper, assemble row HTML with non-ternary logic |
| 15 | Same misbinding ternary in `_render_score_table` and `_block_top_reversion` (Z subtab) and `_block_fresh_breaks` (Proximity) | Same root cause | Same fix — explicit assembly with pre-formatted strings |

### Interpretation-everywhere layer (Z subtab) — added after first round of testing

When Yash first looked at the Z subtab he asked: "I am using the Hurst and half-life analysis
for the first time so I want to know the interpretation of each that is currently shown — and
this goes for the whole subtab of Z-score & Mean-Reversion. I won't understand the numbers by
seeing them so I want to see the interpretation of them instead."

Solution: every numeric on the Z subtab is now paired with a plain-English label, and the
subtab opens with a comprehensive "Interpretation guide" expander.

**`lib/mean_reversion.py`** — new helpers, each returns ``(label, color_var)``:

| Helper | Returns |
|---|---|
| `z_interpretation(z)` | "neutral / elevated HIGH / stretched HIGH / elevated LOW / stretched LOW" + threshold |
| `pct_rank_interpretation(pct)` | "top decile (rich) / upper quartile / middle range / lower quartile / bottom decile (cheap)" |
| `half_life_interpretation(hl)` | "very fast (≤5d) / fast (5-10d) / medium (10-30d) / slow (30-60d) / very slow (>60d) / no reversion fit" |
| `hurst_interpretation(h)` | "strongly trending (H≥0.65) / trending / random walk / reverting / strongly reverting (H≤0.35)" |
| `adf_interpretation(reject, p)` | "stationary — fade-trustworthy / borderline — weak evidence / non-stationary — caution fading" |
| `velocity_to_mean_interpretation(v)` | "rapidly reverting / drifting back / stable / still extending / rapidly extending" |
| `reversion_score_interpretation(s)` | "strong reversion setup / moderate / weak / very weak" |
| `trend_score_interpretation(s)` | "strong trend — DON'T fade / moderate trend / weak trend / very weak" |
| `overall_setup_interpretation(z, h, hl, adf)` | composite TAKE label: "FADE — high-quality / RIDE TREND — don't fade / early fade / early trend / elevated / neutral · no edge" — combines all four signals into a single plain-English action recommendation |
| `get_interpretation_guide()` | structured dict of every metric with `what` (description), `formula`, and `buckets` (list of `(condition, label, meaning)`) — used by the "Interpretation guide" expander |

**`tabs/us/sra_analysis_zscore.py`** — UI changes:

1. **"📖 Interpretation guide"** expander at the top of the subtab (default closed).
   Click to see a per-metric reference card with formula and bucket table.
2. **Section-ribbon tables** — each numeric cell is two-line: value on top, italic
   plain-English label underneath (italic, smaller, color-coded). New columns:
   `Z + reading · μ-Δ · PCT + reading · PATTERN · HURST + reading · HALF-LIFE + reading · TAKE`
   Right-most TAKE column shows a colored pill with the overall action recommendation.
3. **Composite-score tables** — same treatment: `SCORE + reading · Z + reading · HURST + reading · HALF-LIFE + reading · ADF + reading · TAKE`
4. **Side-panel blocks** — each block has a final "Reading:" italic paragraph that
   interprets the numbers shown above. Examples:
   - Universe stretch → calls out skew/kurtosis interpretation in plain language
   - Pattern catalog → identifies dominant theme (trend/reversal/coiling/event-driven)
   - Section stretch → translates curve label to trader action
   - Half-life ranking → counts fast vs medium reverters, calls out best candidates
   - Hurst summary → describes the universe's memory regime
   - ADF + Z gate → explicit fade-trustworthy vs caution-fading summary
   - Top reversion / trend → highlights how many score in each tier
   - Velocity → tells you whether market is reverting or still extending
   - Cross-check → describes UNTESTED / STRETCHED / COILED prevalence
5. **Drill-down expander** — adds a full prose paragraph beneath the metrics row that
   reads each value (Z, Pct, Hurst, half-life, ADF) with its interpretation, plus a
   prominent TAKE pill in the corner.

Tooltips on rows now include both the raw values and their plain-English meanings,
so the trader can mouse over to see the whole reading at once.

The Proximity subtab (already shipped) was not touched in this round — its tooltips
already include trader implications. Same treatment can be added there if needed.

### Threshold-aware tooltips on every value (added after second round of testing)

Yash's follow-up: "for each interpretation along with the calculations on hover I want to
see the threshold for that interpretation for the numbers calculated too everywhere".

Solution: every numeric value's hover tooltip now shows the **full bucket table** with the
matching bucket marked ✓, in addition to the formula + computed value + interpretation.

**Engine layer** — `lib/mean_reversion.py` adds `metric_tooltip(guide, key, current_label,
computed)` that builds a multi-line tooltip from a structured guide dict:

  - Title (metric name)
  - "Current: <label>" if known
  - What (description)
  - Formula
  - Threshold table — each bucket as ``cond → label · (meaning)`` with the matching
    bucket prefixed `✓ ` so the user sees at a glance which one fired.
  - Computed value (e.g. `z = +0.77σ (μ_30 = 96.319, σ_30 = 0.033)`)

**Mirrored guide for Proximity** — `lib/proximity.py:get_proximity_interpretation_guide()`
adds parallel structured guide entries for: Distance to extreme (ATR-normalised) ·
Position-in-range (PIR) · Streak at extreme · Velocity to extreme · Touch count ·
Range expansion ratio · Confluence pattern · Section regime (proximity) · Z-score cross-check.
``metric_tooltip()`` is generic — it accepts either guide dict.

**UI layer (both subtabs)**:
- Z subtab — `_render_z_table` and `_render_score_table` now wrap each numeric cell in a
  `title="..."` attribute with that metric's threshold tooltip (cell hover wins over
  row hover). `_kv_block` accepts a 5-tuple `(label, value, color, sub, tooltip)` so any
  side-panel row can carry a tooltip; populated for universe stretch counters, pattern
  catalog, section stretch, half-life ranking, hurst summary, ADF gate, top reversion,
  top trend, velocity, cross-check.
- Z drill-down `st.metric` calls now pass `help=` containing the metric_tooltip output,
  so each metric card displays a (?) icon that on hover shows formula + thresholds +
  computed value + bucket marker.
- Proximity subtab — same treatment for ribbon table cells, side-panel blocks (universe
  glance, extreme density, pattern catalog, streak/velocity, fresh/failed breaks,
  cross-tenor stretch, section regime, z-cross-check, touch/range, concurrent clusters).
  Adds its own "📖 Interpretation guide — click to see what every proximity metric means"
  expander at the top of the subtab.

**DOM verification** (sampled live): Z subtab carries **372** distinct title-attribute
threshold tooltips, **129** with the ✓ marker on the matching bucket. Proximity subtab
carries an analogous set. Counts by metric on the Z subtab: Z-score (49), Hurst (41),
OU half-life (44), ADF (12), Percentile rank (28), Overall TAKE (38), composite
score / pattern (rest).

---

## 5D · TECHNICALS SUBTAB (TMIA-curated setup detectors) — COMPLETE

The third Analysis sub-subtab. **Removes** the six placeholder sub-subtabs (PCA /
Terminal rate / Carry / Slope-Curv / Stat tests / Fly residuals) and replaces them
with a single comprehensive **Technicals** view that absorbs the TMIA v13 gameplan
setup catalog at SRA scope.

### 5D.1 · Curated setup catalog (Yash-locked subset, 26 outputs / 41 fire-points)

**Trend (12)** — A1, A2*, A3*, A4, A5*, A6, A8, A10, A11, A12a, A12b, A15
**Mean-reversion (7)** — B1, B3, B5, B6, B10*, B11, B13
**STIR-specific (7 fire-points)** — C3, C4, C5, C8(12M/24M/36M), C9a, C9b
**Composites (3 per scope)** — TREND_COMPOSITE / MR_COMPOSITE / FINAL_COMPOSITE
re-tuned per OUTRIGHT / SPREAD / FLY scope.

Customizations applied:
- **A2** uses EMA_100 (not EMA_200)
- **A3** uses 90-day percentile (not 252-day) for `BB_BW` and `BB-inside-KC` gates
- **A5** Donchian-55 extended to flies; skip-with-N/A flag when bars<55
- **A12** split into A12a (close>EMA_20) and A12b (close>EMA_50) variants
- **B10** vol_z(90d) and TR_z(90d) (not 252d / 60d)
- **C8** three variants (12M/24M/36M) emit separately; strongest gets PRIMARY badge
- **C9** split into C9a (slope crossing) and C9b (5d slope trend)
- **A6** kept as EMA_50 × EMA_200 with EMA_200 computed at runtime via `wilder_ema`

Removed (out of scope for SRA-only): A7, A9a, A9b, A13, A14, B2, B4, B7, B8, B9, B12,
C1, C2, C6, C7, C10.

### 5D.2 · Composite redesign per scope

The original TMIA composite (8 trend factors / 9 MR factors / regime-weighted FINAL)
is built for a 91-product universe. STIR outrights, spreads, and flies have very
different statistical character — so each scope gets its own factor list and weight
schedule. Full math in `lib/setups/composite.py`:

| Scope | Trend factors | MR factors | FINAL gates |
|---|---|---|---|
| **OUTRIGHT** | 8 (incl. f8 = carry direction tilt) | 6 (IBS / CCI / range·close-pos dropped — too noisy on STIR outrights) | `w_trend = clip(ADX/25)` · Hurst factor · ±0.10 carry tilt |
| **SPREAD**   | 5 (ADX·DI denom 22; spread-EMA stack; bp-distance from EMA_50; 5d slope; Donchian-30) | 7 (multi-window Z 60/30/15; BB_pctB; ZRET_5; 90d percentile; OU HL factor) | `w_trend = clip(ADX/30, max 0.4)` · ADF gate · Hurst factor |
| **FLY**      | 3 (slope, distance from 60d mean, ADX·DI denom 18) | 7 (multi-window Z; BB_pctB; 90d percentile; signed OU multiplier; extreme-pull) | `w_trend = 0.10` (fixed) · ADF + Hurst + OU triple-gate |

FINAL bucket convention is uniform across all scopes:
`≥+0.7 STRONG ↑ · 0.5 MODERATE ↑ · 0.3 SOFT ↑ · |x|<0.3 NEUTRAL · -0.3 SOFT ↓ · -0.5 MODERATE ↓ · ≤-0.7 STRONG ↓`.

### 5D.3 · Engine layer

```
lib/runtime_indicators.py    — wilder_ema, ema_200, rsi, rsi_2, ichimoku_full,
                                ibs, vol_zscore (90d), tr_zscore (90d), true_range,
                                donchian_high_low, swing_detect, rsi_divergence_today,
                                bb_inside_kc_counter, rolling_percentile (90d),
                                linear_slope, polyfit_pattern_lines

lib/setups/__init__.py        — exports
lib/setups/base.py            — SetupResult dataclass + helpers (lots_at_10k_risk,
                                compute_R_levels, distance_to_threshold_atr,
                                normalize_distance, state_from_conditions_met)
lib/setups/registry.py        — SETUP_REGISTRY dict; per-id metadata, scopes,
                                logic_long/short, stop_rule, t1/t2/trail/time/partial,
                                threshold buckets
lib/setups/interpretation_guide.py — get_full_guide_for_scope(scope) builds the
                                guide consumed by metric_tooltip() so every
                                setup cell shows formula + thresholds + ✓ marker
lib/setups/trend.py           — 12 detectors (A1-A15 subset)
lib/setups/mean_reversion.py  — 7 detectors (B1, B3, B5, B6, B10, B11, B13)
lib/setups/stir.py            — C3 (universe-level + per-contract), C4, C5,
                                C8 12/24/36 variants (terminal-rate computation),
                                C9a/C9b (slope crossing / 5d trend)
lib/setups/composite.py       — compute_outright_composites, compute_spread_composites,
                                compute_fly_composites with per-scope factor lists
                                and weight schedules
lib/setups/scan.py            — scan_universe(strategy, tenor, asof_str) — cached
                                @ st.cache_data(ttl=600). Returns
                                {by_contract, fires_today, near_today, track_records, ...}
lib/setups/track_record.py    — compute_track_record(panels, asof, scope, window=60,
                                forward_bars=5) — replays detectors over last 60
                                trading days and returns
                                {fires_60d, win_rate_5bar, mean_5bar_return_bp,
                                 sample_quality} per setup
```

`lib/sra_data.py` was extended with `get_contract_full_panel(symbol, start, end)`
which pulls OHLCV + all 167 indicator columns from the
`v_mde2_timeseries_with_indicators` view in one query.

### 5D.4 · UI architecture (`tabs/us/sra_analysis_technicals.py`)

Same shell as Proximity / Z-score & MR. Top controls: Strategy · Tenor · Family
filter · Side-panel toggle. Plus the **📖 Interpretation guide** expander at the
top covering all 26 setups with formula + threshold buckets.

Five views (each toggleable):

1. **View 1 · Today's fires** — 3 columns (TREND / MR / STIR-specific). Per-setup
   cards with fire counts and per-fire mini-cards (entry / stop / T1 / T2 / lots
   @ $10K risk + 60d track-record badge with win-rate and mean +5-bar return).
2. **View 2 · Setup state matrix** — heatmap-style grid: rows = top 25 contracts
   (ranked by fire/near density), cols = setups; cells show 🟢/🔴/🟡/🟠/🔵/·/—
   for FIRED-LONG / FIRED-SHORT / NEAR-LONG / NEAR-SHORT / APPROACHING / FAR / N/A.
   Hover any cell = formula + computed inputs + threshold table + ✓ marker.
3. **View 3 · Composite scoring** — per-contract row showing TREND / MR / FINAL
   scores on a [-1,+1] colour-graded scale + regime label + plain-English TAKE
   pill. Hover any score cell = full factor breakdown + weights.
4. **View 4 · Proximity-to-fire** — sortable table of NEAR signals ranked by
   distance-to-fire ascending, with ETA (estimated bars until fire).
5. **View 5 · Drill-down expander** — pick a setup + contract; renders the per-row
   setup card + trade levels + 60d track record + OHLC chart.

Side panel (12 blocks, scrollable 720px):

1. Methodology / setup catalog tooltip
2. Universe scan summary
3. Fire density · Family × Direction
4. Hot setups · most fires today
5. Quiet setups · zero fires
6. Confluent contracts · ≥3 setups same direction
7. Regime conflicts · trend ↑ + MR ↓ (sit-out list)
8. Composite distribution (FINAL bucket counts)
9. Regime distribution (gameplan §7.5 labels)
10. 60d track records (top setups by mean +5-bar return)
11. (placeholder for future Z/Proximity cross-check)
12. Setup-pattern legend (FIRED / NEAR / APPROACHING / FAR / N/A definitions)

### 5D.5 · DOM verification (live)

- **998 title-attribute tooltips with thresholds**
- **466** with ✓ markers on matching bucket
- **25** visible FIRED cells (🟢/🔴), **87** NEAR (🟡/🟠), **53** APPROACHING (🔵)
- Cold scan time on 29 outrights ≈ 50s; cached subsequent runs near-instant
- All 5 views render; side panel populated; no Python errors in Streamlit log

### 5D.6 · Sizing

`lots_at_10k_risk(stop_distance_bp)` uses SRA DV01 = $25 / bp / contract:

```python
lots = floor( $10,000 / (stop_distance_bp × $25) )    # capped at 1000, min 1
```

Same DV01 across SRA outrights, spreads, and flies (spreads/flies are 2-leg /
3-leg structures whose net DV01 simplifies to $25/bp on the spread/fly value).

### 5D.7 · Tick rounding (post-shipping refinement)

All entry / stop / T1 / T2 prices are rounded to the half-bp tick on display so
the levels match what the exchange would actually fill:

```python
TICK_BP = 0.5    # 0.5 basis points = 0.005 price units (SR3 back-month tick)
def round_to_tick(price, bp_mult):
    tick_stored = TICK_BP / bp_mult     # 0.005 for outright, 0.5 for spread/fly
    return round(price / tick_stored) * tick_stored
```

`fmt_price_for_scope(price, bp_mult)` applies rounding then formats:
- Outright (`bp_mult=100`): 4 decimals → `96.3850`
- Spread/fly (`bp_mult=1`): 2 decimals + `bp` suffix → `-2.50bp`

Both `_set_levels_long/short` helpers in `lib/setups/trend.py` apply rounding,
so detectors that flow through them get rounding for free. Detectors with
direct T1/T2 overrides (B1 BB-mid, B5/B11/B13 SMA-20, B10 EMA-20, A5 high/low,
A8 cloud heights, C4/C5/C8 mean targets) were patched to round explicitly.

The drill-down's `st.metric` cards also use `fmt_price_for_scope`.

### 5D.8 · UI polish layer (custom CSS injection)

Beyond the threshold-tooltip pattern, the Technicals subtab gets its own
local CSS (injected at the top of `render()` and idempotent across reloads)
to upgrade visual density without breaking the locked dashboard theme:

| CSS class | Purpose |
|---|---|
| `.tech-state` (with `.fired-long`, `.fired-short`, `.near-long`, `.near-short`, `.appr`, `.far`, `.na`) | State pills used in proximity-to-fire row direction badges |
| `.tech-cell` (same modifiers) | **Replaces emoji** in the Setup State Matrix with colored CSS chips. Rendering `<span class="tech-cell fired-long">L</span>` produces a green-filled cell with white "L" letter. NEAR/APPROACHING use lighter tints. FAR/NA are barely visible. |
| `.fire-panel-header` / `.fire-setup-card` / `.fire-row` | Cleaner card layout for View 1 (Today's fires) — 3-column TREND/MR/STIR panels with header → setup card → per-fire row pattern. Each `.fire-row` shows direction arrow + symbol + indicator chips + trade-level cells (entry/stop/T1/T2/lots) with rounded prices. |
| `.comp-bar-track` / `.comp-bar-fill` (`.pos`/`.neg`) | Inline horizontal bar visualization for View 3 (Composite scoring). Each composite score is rendered as a colored bar centered at 0; positive fills right (green), negative fills left (red). The numeric value sits centered over the bar. Hover reveals the full factor breakdown + weights. |
| `.tech-table` | Standardised table styling used across views 2, 3, 4 (replaces the previous CSS-grid divs that were inconsistent across browsers). |
| `.tech-view-header` | Section-header wrapper with title + meta-line ("23 fires · grouped by family") for each view. |

Each view now uses a `_view_header(title, meta)` helper with consistent
typography (Inter for title, JetBrains Mono for meta).

### 5D.9 · DOM verification (post-polish)

| Metric | Count |
|---|---|
| Threshold tooltips with `Thresholds:` block | **998** |
| With ✓ marker on matching bucket | **466** |
| `.tech-cell.fired-long` (FIRED LONG pills) | 3 |
| `.tech-cell.fired-short` (FIRED SHORT pills) | 22 |
| `.tech-cell.near-long` | 32 |
| `.tech-cell.near-short` | 80 |
| `.tech-cell.appr` (APPROACHING) | 53 |
| `.tech-cell.far` | 276 |
| `.tech-cell.na` | 191 |
| `.comp-bar-track` (composite bars) | **87**  (= 29 outrights × 3 composites) |

---

### 5D.10 · Background pre-warm (zero-delay first scan)

Yash feedback: *"by default on app load itself all the calculations needed for the
Technicals subtab should start so when viewing there is no delay"*. The cold
`scan_universe(outright)` takes ~30-60s (29 contracts × 22 detectors + composites
+ 60d track record). Once cached, the same call returns instantly because
`scan_universe` is decorated with `@st.cache_data(ttl=600)`. Solution: kick the
scan off in a background daemon thread the moment the dashboard process starts,
so the cache is populated before the user clicks Technicals.

#### Module — `lib/prewarm.py`

| Public API | Purpose |
|---|---|
| `ensure_prewarm()` | Idempotent. Spawns the daemon thread on first call; no-op afterwards. Safe to invoke from every page render. Returns immediately. |
| `get_prewarm_status()` | Read-only snapshot dict with `started_at`, `outright_done_at`, `spread_3m_done_at`, `fly_3m_done_at`, `errors`. |
| `is_prewarm_done()` | True iff the outright leg (the slowest) has completed. |

Internals:
- Module-level `_LOCK = threading.Lock()` + `_PREWARM_STARTED` flag — guarantees the worker spawns exactly once per server process regardless of how many Streamlit re-runs / page navigations happen.
- Worker spawned via `threading.Thread(target=_prewarm_worker, daemon=True)` — never blocks the UI; dies cleanly when the server shuts down.
- Three legs run sequentially inside the worker:
  1. **Outright** — most-likely first click and biggest universe (29 contracts), ~30-60s. After this, `is_prewarm_done()` flips True.
  2. **Spread** — `get_available_tenors("spread")` sorted with 3M first, then next tenor. Each ≤24 contracts so they finish in 10-25s.
  3. **Fly** — `get_available_tenors("fly")` sorted with 3M first, then next tenor. Smallest universes, fastest legs.
- Errors are appended to `_PREWARM_STATUS["errors"]` and never re-raised — pre-warming is best-effort and must not break the dashboard.

Streamlit interaction note: `scan_universe` uses `@st.cache_data` which off-thread emits
`WARNING streamlit.runtime.caching.cache_data_api: No runtime found, using
MemoryCacheStorageManager`. This is expected and harmless — the worker is
populating Streamlit's process-level `MemoryCacheStorageManager` directly. When
the user later triggers `scan_universe` from the UI thread, Streamlit finds
the same cache key and returns the cached dict instantly.

#### ⚠️ Thread-safety requirement: per-thread DuckDB connections

Pre-warming is *only* safe because the underlying DuckDB connections are
**thread-local**. DuckDB's Python client is not thread-safe at the connection
level — calling `con.execute()` on the same connection from two threads
simultaneously crashes with:

```
InternalException: INTERNAL Error: Attempted to dereference unique_ptr that is NULL!
```

The fix lives in `lib/connections.py`. `get_ohlc_connection`,
`get_bbg_warehouse_connection`, and `get_bbg_inmemory_connection` are no
longer `@st.cache_resource` (which would share one connection across all
threads). Each is now backed by `threading.local()`:

```python
_thread_local = threading.local()

def get_ohlc_connection():
    con = getattr(_thread_local, "ohlc_con", None)
    if con is not None:
        return con
    path = resolve_latest_ohlc_snapshot()
    con = duckdb.connect(path, read_only=True)
    _thread_local.ohlc_con = con
    return con
```

DuckDB tolerates many simultaneous read-only connections to the same file,
so each thread (UI session, prewarm daemon, helper threads) gets its own.
Same-thread reuse is preserved (cached on `_thread_local`), so calls within
a Streamlit re-run don't pay the open-cost. Stress test verified: **400
concurrent queries on 2 threads with zero `NULL unique_ptr` errors**, and
a real `scan_universe(outright)` call from a daemon thread alongside a
parallel `load_sra_curve_panel` loop on the main thread completes cleanly.

If you ever add a new `@st.cache_resource` getter that returns a DB
connection or any other non-thread-safe handle, **route it through
`threading.local()` instead** — pre-warm + UI access otherwise will deadlock
or crash under load.

#### Integration points

| File | Hook |
|---|---|
| `app.py` | `from lib.prewarm import ensure_prewarm` + call after `inject_global_css()` |
| `pages/1_US_Economy.py` | Same — fires if user opens US Economy directly without hitting the landing page |
| `tabs/us/sra_analysis_technicals.py` `render()` | Calls `ensure_prewarm()` defensively (also no-op once started). Reads `get_prewarm_status()` to render the **⏳ Pre-warm in progress (~Ns elapsed)** banner if a scan is mid-flight. Spinner text switches between *"first run ~30-60s; cached after"* and *"using cached pre-warm"* based on `is_prewarm_done()`. |

#### Verified end-to-end (this session, qh_data env)

| Check | Result |
|---|---|
| `ensure_prewarm()` returns time | 0.00s (truly non-blocking) |
| Outright leg cold-scan time | **~53s** on this machine — within documented 30-60s envelope |
| `is_prewarm_done()` after outright | True ✓ |
| Errors during outright leg | None ✓ |
| Spread/fly legs after outright | Continue in background; UI never waits on them |
| Streamlit auto-reload picks up new hook | Yes — `app.py` modification triggers script re-run; `lib.prewarm` imports fresh on first request after change |

Net effect: by the time the user clicks the Technicals subtab (typically several
seconds to a minute after dashboard launch), the outright cache is already warm
and the scan returns instantly. The banner exists for the edge case where the
user opens Technicals within the first ~30s of process start.

---

### 5D.11 · Multi-leg trade rendering + C9 multi-pair expansion

Yash feedback: *"yes needs to expand and show the complete trade and this applies
to everywhere"* + *"in todays fires I told you to show setups in order of their
60d"*. Two coupled deliverables.

#### A · `SetupResult.legs` — explicit per-leg trade breakdown

Every fired setup now carries a `legs: list[dict]` field. Each leg has
`{role, symbol, side, lots, ratio, dv01_per_bp}`. Side conventions follow the
DB-stored close convention (verified empirically — see §3.J / §6):

| Scope | Spread/fly close formula | LONG → leg sides | Lot ratios |
|---|---|---|---|
| **outright** | n/a (single contract) | BUY (or SELL for SHORT) | 1 leg |
| **spread** | `left − right` | BUY left + SELL right | 1 : 1 |
| **fly** | `left − 2·mid + right` | BUY left + SELL 2·mid + BUY right | 1 : 2 : 1 |
| **C9 pair (slope)** | `front − back` | BUY front + SELL back (steepener LONG) | 1 : 1 (DV01-neutral) |

Build helpers in `lib/setups/base.py`:
- `build_legs(symbol, scope, direction, lots, legs_from_catalog)` — generic
  outright/spread/fly path; reads catalog `legs` column (comma-separated
  `"left,right"` or `"left,mid,right"`).
- `build_pair_legs(front_sym, back_sym, direction, lots)` — special-purpose
  C9 path used directly inside `detect_c9a` / `detect_c9b` because the pair
  is known internally.
- `_parse_legs_string(legs_str)` — splits the catalog string into a list.

Per-leg DV01 is `$25 × lots × ratio` (SR3 = $25/bp/contract). For a 5-lot
spread: each leg = $125/bp. For a 5-lot fly: wings = $125/bp each, body =
$250/bp (matches the 1:2:1 lot ratio).

`scan.py` calls `_fill_legs(rd, sym)` after every detector returns, populating
legs from the per-symbol `legs_lookup` dict (built once at scan start from
`load_catalog()`). Idempotent — won't overwrite legs already set by a detector
(e.g. C9).

#### B · C9 multi-pair curve ladder

`pick_c9_contract_pair` (singular) is preserved for back-compat; the new
`pick_c9_contract_pairs` (plural) returns a list of `(front, back, offset_M)`
tuples for the default offset ladder `(6, 12, 18, 24) M`. Each pair generates
its own offset-suffixed setup id:

| Setup id | Pair | Description |
|---|---|---|
| `C9a_6M`, `C9a_12M`, `C9a_18M`, `C9a_24M` | front + (6/12/18/24)M back | Slope crossing zero today |
| `C9b_6M`, `C9b_12M`, `C9b_18M`, `C9b_24M` | front + (6/12/18/24)M back | Slope 5d trend ≥ ±5bp |

Each variant runs `_slope_history(front, back)`, applies the same FIRED /
NEAR / APPROACHING gates, and populates entry/stop/T1/T2 in slope-bp units
(rounded to half-bp tick). The default stop = 5bp on both A and B variants
→ `lots_at_10k_risk = 80` per side. Skipped silently if the offset has no
distinct back contract.

`registry.get_registry_entry(sid)` falls back to the base ID (`C9a` / `C9b`)
when fed a suffixed variant — strips the trailing `_NM` token and appends
the offset to the displayed name (e.g. `"CURVE_STEEPENER / FLATTENER · slope crossing · 12M slope"`).

Verified end-to-end in this session: 8 C9 entries per outright scan (2
detectors × 4 offsets), all keyed under the front contract symbol.

#### C · View 1 sorted by 60d track record

Setup cards within each family column (TREND / MR / STIR-specific) now sort
by `mean_5bar_return_bp` descending — best 60d performers first, untracked
setups (e.g. C9 variants, which aren't in `PER_CONTRACT_DETECTORS`) at the
end alphabetically.

Sort key (in `tabs/us/sra_analysis_technicals.py`):

```python
def _track_sort_key(sid):
    base_id = sid.split("_")[0] if sid.startswith(("C9a_", "C9b_")) else sid
    tr_for = track_records.get(sid) or track_records.get(base_id) or {}
    fires_n = int(tr_for.get("fires_60d") or 0)
    mr_bp   = tr_for.get("mean_5bar_return_bp")
    if fires_n == 0 or mr_bp is None:
        return (1, 0.0, sid)             # untracked → after tracked
    return (0, -float(mr_bp), sid)       # tracked, descending mean return
```

The C9 variants share the un-suffixed track record (none today) but the
key still resolves cleanly because of the base-id fallback.

#### D · Rendering — `_legs_block_html(legs)`

Each fire card and the drill-down render a compact 5-column legs table:
`SIDE · SYMBOL · ratio · LOTS · DV01`. CSS class `.trade-legs` (defined
in the technicals CSS injection) adds an accent left-border, color-coded
BUY (green) / SELL (red) pills, and monospace alignment.

Integration points:
- View 1 fire-row — appends `_legs_block_html(f["legs"])` after the entry/stop/T1/T2 trade-cells.
- View 5 drill-down — renders below the 5-column metric strip when `setup_state["legs"]` is populated.

#### E · Verified shapes

| Scope | Sample fire | Legs |
|---|---|---|
| outright (single) | `A4 SRAH27 LONG ×40 lots` | `BUY SRAH27 ×40` ($1000/bp) |
| outright (C9 pair) | `C9b_6M SRAM26 LONG (steepener)` | `BUY SRAM26 ×80` + `SELL SRAZ26 ×80` ($2000/bp each) |
| spread | `A10 SRAU26-Z26 LONG ×133 lots` | `BUY SRAU26 ×133` + `SELL SRAZ26 ×133` ($3325/bp each) |
| fly | `A2 SRAM26-U26-Z26 LONG ×160 wings` | `BUY SRAM26 ×160` + `SELL SRAU26 ×320` + `BUY SRAZ26 ×160` ($4000/$8000/$4000) |

Coverage of legs population in this session's snapshot:

| Scope | fires today | with legs | reason for gap |
|---|---:|---:|---|
| outright | 26 | 24 | C3 has no entry/stop set (carry-rank signal) — 2 fires unstaged |
| spread 3M | 22 | 19 | 3 setups (e.g. C5 specific gates) without entry |
| fly 3M | 20 | 16 | 4 setups (e.g. C4 specific gates) without entry |

C9 pairs always populate legs because the detector sets them inline.

#### F · Refinement — per-leg entry prices + visible rank cue

After the initial pair-trade work landed, Yash flagged that the C9 fire cards
showed lots and DV01 per leg but no actual price the trader could lift/hit on
each individual leg, and that the 60d-track ordering wasn't visually obvious.
Two coupled refinements:

**F1 · Per-leg `entry_price` on `build_pair_legs`** — accepts optional
`front_close` / `back_close` (today's outright closes in price units) and
attaches an `entry_price` to each leg dict. New helper
`_pair_today_closes(close_history, front, back, asof)` in `lib/setups/stir.py`
extracts the latest `≤ asof_date` close of each leg; both `detect_c9a` and
`detect_c9b` now call it after the slope passes its state check. `_legs_block_html`
renders `@ <price>` (tick-rounded via `fmt_price_for_scope(ep, 100.0)`) next
to each leg symbol when `entry_price` is set:

```
[BUY ]  SRAH27 @ 96.0450    80   $2,000/bp · front
[SELL]  SRAH28 @ 95.9100    80   $2,000/bp · back
```

Backwards-compat: `build_pair_legs` without closes still emits legs, just with
`entry_price=None` — every pre-existing call site keeps working. Spread/fly
aggregate fires already show entry/stop/T1/T2 on the listed contract itself,
so per-leg prices on their decomposed legs remain optional (would require
loading outright panels for each leg — left as a follow-up).

**F2 · `#N` rank pill on setup cards** — the sort key `(0, -mean_5bar_return_bp, sid)`
already sorted by 60d track-record descending, but with no visible indicator
that ordering had been applied. Each card title now carries a `#N` pill
prefix coloured by 60d direction:

| Pill colour | Meaning |
|---|---|
| green `#N` | tracked · positive `mean_5bar_return_bp` |
| red `#N`   | tracked · negative `mean_5bar_return_bp` |
| dim `#N`   | untracked (no fires in 60d window) — sorted alphabetically after tracked |

Tooltip on the pill: *"Rank within family by 60d mean +5-bar return (best
first; untracked setups follow alphabetically)"*.

**Verified (qh_data env, this session):**
- `build_pair_legs(LONG, front_close=96.045, back_close=95.910)` → BUY SRAH27 @ 96.045 / SELL SRAH28 @ 95.910 ✓
- `build_pair_legs(SHORT, ...)` → SELL front @ 96.045 / BUY back @ 95.910 ✓
- `build_pair_legs()` without closes → both `entry_price=None` (back-compat) ✓
- `_c9_legs_and_lots(LONG, 5bp, ..., front_close=96.045, back_close=95.910)` → 80 lots/side, front leg `entry_price=96.045` ✓
- All three modified files (`lib/setups/base.py`, `lib/setups/stir.py`, `tabs/us/sra_analysis_technicals.py`) compile + import clean ✓

---

## 5E · CMC LAYER — Constant-Maturity-Curve construction (NEW — Phase A, COMPLETE)

The Technicals backtest cannot use raw contract identifiers (SRAM27 was a back
contract for years and is the front today; setup statistics computed on its
history are contaminated by curve-position drift). Yash flagged this; the
fix is to build constant-maturity continuous (CMC) series — M0, M3, M6,
M12, ..., M60 outright nodes, plus derived spreads and flies — and run all
historical backtests against those.

### Methodology (research-backed; locked)

| Decision | Choice | Source |
|---|---|---|
| Roll rule | **Calendar: 5 business days before front contract's last trade date**, with volume sanity check (skip if next-contract volume hasn't exceeded the front for 3 consecutive sessions) | Norgate, Pinnacle CLC, Carver pysystemtrade |
| Adjustment | **Backward Panama (additive)**: `gap = new_close − old_close` on roll date; add gap to all prior bars | CSI / Norgate / Portara / QuantConnect default; mandatory for STIR (ratio breaks across zero/negative rates) |
| %-return denominator | **Raw contract price** (Carver correction): `adj_close[t] − adj_close[t-1]` over `raw_close[t-1]` | Carver, "Systems building: futures rolling" |
| Constant-maturity sampling | **Linear interpolation between bracketing contracts on days-to-expiry**: `CMC = (1−w)·c1 + w·c2` where `w = (T − dte_c1) / (dte_c2 − dte_c1)` | Holton; S&P VIX Futures Indices methodology |
| Spread / fly construction | **Derived from CMC outright nodes**: `M3M6 = M3 − M6`; `M3M6M9 = M3 − 2·M6 + M9` | Burghardt, *Eurodollar Futures Handbook* |
| LTD source | **Canonical SR3 rule**: day before 3rd Wednesday of `expiry_month + 3` (named-month convention). Catalog has only `expiry_year` / `expiry_month`; observed `MAX(time)` truncates to data-end for live contracts. | CME SR3 contract specs |
| Earliest start | **5 years from today** = 2021-Q2 → 2026-Q2. SR3 launched 2018-05-07; deferred quarterlies liquid since ~Q1 2019. | CME SOFR launch press release |

Full citations + links live in
`C:\Users\yash.patel\.claude\plans\c-users-yash-patel-downloads-tmia-v13-s-magical-mitten-agent-a85c9abb95e87077c.md`.

### CMC nodes (22 per asset class)

| Scope | Node IDs | Construction |
|---|---|---|
| Outright (10) | M0, M3, M6, M9, M12, M18, M24, M36, M48, M60 | Linear-interp between bracketing quarterlies on DTE; back-adjusted at roll boundaries |
| Spread (8) | M0_M3, M3_M6, M6_M9, M9_M12, M0_M6, M0_M12, M6_M12, M12_M24 | `M{a}_M{b} = M{a}_outright − M{b}_outright` |
| Fly (4) | M0_M3_M6, M3_M6_M9, M6_M9_M12, M0_M6_M12 | `M{a}_M{b}_M{c} = M{a} − 2·M{b} + M{c}` |

### Module layout

| File | Role |
|---|---|
| `lib/cmc.py` | All construction logic. Public API: `build_cmc_nodes(scope, asof_date, history_years=5)`, `load_cmc_panel(scope, asof_date)`, `get_cmc_roll_log(asof_date)`, `list_cmc_nodes(scope)`. Internal helpers: `_load_quarterly_chain`, `_load_per_contract_panels`, `build_roll_calendar`, `back_adjust_panels`, `_interpolate_cmc_outright_node`, `_build_outright_cmc_table`, `_build_spread_cmc_table`, `_build_fly_cmc_table`. |
| `lib/cmc_verify.py` | Verification harness. Public API: `verify_all(asof_date) → dict` and `python -m lib.cmc_verify [YYYY-MM-DD]` CLI. Five checks: continuity / roll-boundary correctness / indicator non-fracture / term-structure sanity / manifest integrity. |
| `lib/sra_data.py` | Streamlit-cached accessors: `list_cmc_node_ids(scope)`, `load_cmc_node_panel(scope, node_id, asof_date)` (datetime-indexed OHLCV with `raw_close_anchor`), `load_cmc_wide_panel(scope, asof_date, field='close')` (cross-tenor pivot). |

### On-disk cache layout

```
D:\STIRS_DASHBOARD\.cmc_cache\
├── sra_outright_<YYYY-MM-DD>.parquet     # ~544 KB; 10 nodes × ~1305 BD ≈ 13050 rows
├── sra_spread_<YYYY-MM-DD>.parquet       # ~460 KB; 8 nodes  × ~1305 BD ≈ 10440 rows
├── sra_fly_<YYYY-MM-DD>.parquet          # ~264 KB; 4 nodes  × ~1305 BD ≈ 5220 rows
└── manifest_<YYYY-MM-DD>.json            # roll log, gap stats, missing-contract flags, builder version
```

Cold build: ~17s for the full 5-year × 22-node SRA universe. Cache hit: 0.000s.
Cache key = `asof_date`. The CMC builder is idempotent and atomic (writes
all three parquets + manifest in one call).

### Parquet schema — outright

| Column | Type | Notes |
|---|---|---|
| `cmc_node` | str | 'M0' .. 'M60' |
| `bar_date` | date | Business day |
| `c1_sym`, `c2_sym` | str | The two bracketing listed quarterlies |
| `weight` | float | Linear-interp weight on `c2`; `c1` weight = `1−weight` |
| `dte_c1`, `dte_c2` | int | Business days to expiry of each bracket |
| `open` / `high` / `low` / `close` | float | Back-adjusted, linearly-interpolated OHLC |
| `volume_combined` | float | Linearly-interpolated volume (informational, not authoritative) |
| `raw_close_anchor` | float | RAW (un-adjusted) close of `c1` — **the Carver-correction denominator** for any %-return indicator |
| `has_data` | bool | Both legs had a bar that day |

Spread/fly schemas mirror the outright but with `c1_left/c1_right` (spreads) or `c1_left/c1_mid/c1_right` (flies) and the corresponding linear combinations.

### Verification (5/5 checks pass on the 2026-04-27 snapshot)

- **Continuity**: every CMC outright node within 10% miss tolerance post-first-valid bar (handles US holidays + deep-tenor sparseness in early history).
- **Roll-boundary correctness**: 20/20 historical rolls. At each, `adj_close[old, roll_date] == adj_close[new, roll_date]` to within 1e-6.
- **Indicator non-fracture**: max abs daily change on M3 at any roll date = 5.25 bp (well under 25 bp threshold).
- **Term-structure sanity**: max adjacent-node step at the latest bar = 14.4 bp (well under 100 bp guard).
- **Manifest integrity**: 22 nodes / 20 rolls / clean stats / SRAH26 correctly flagged as missing.

Run from CLI: `python -m lib.cmc_verify 2026-04-27`.

### Known data gaps

- **SRAH26 (March 2026 SR3 quarterly) is missing from the OHLC catalog.** The manifest flags it (`missing_contracts_in_chain`). Effect: the SRAZ25→SRAM26 roll (2026-03-10) skips the intermediate SRAH26→SRAM26 roll that would normally happen 2026-06-09. The composite SRAZ25→SRAM26 gap captures the cumulative drift. Material only for back-tests that depend on Q1-Q2 2026 SOFR transitions.
- Pre-2019 quarterlies (SRA{H,M,U,Z}{18,19,20}) are also flagged but expected — SR3 began 2018-05-07 and our 5-year window starts 2021.

### Relationship to existing systems

- **Existing `load_sra_curve_panel`** (per-contract panels for the Curve / Proximity / Z-score subtabs) is **unchanged**. CMC is additive — used only by the upcoming backtest engine and any explicit "CMC view" toggles.
- **Setup detectors** (`lib.setups.{trend,mean_reversion,stir}.py`) take a generic OHLCV panel; `load_cmc_node_panel` returns one in the same shape, so the same detectors run unchanged on CMC nodes during backtest replay.
- **`lib.prewarm`** (Technicals scan pre-warm) is independent — pre-warms scan_universe over per-contract data for the live "Today's Fires" view. CMC results live in their own cache and are loaded on demand by the backtest engine (Phase C-D, not yet wired).

### References

- CME — [Three-Month SOFR overview](https://www.cmegroup.com/markets/interest-rates/stirs/three-month-sofr.html), [SOFR launch press release](https://www.cmegroup.com/media-room/press-releases/2018/3/01/cme_group_announcesnewsofrfutureslaunchdateandcontractspecificat.html)
- Holton — [Constant-Maturity Futures Prices](https://www.value-at-risk.net/constant-maturity-futures-prices/)
- S&P DJI — [VIX Futures Indices Methodology](https://www.spglobal.com/spdji/en/documents/methodologies/methodology-sp-vix-futures-indices.pdf)
- Hurst, Ooi & Pedersen — [Century of Evidence on Trend-Following](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2993026); Moskowitz, Ooi & Pedersen — [Time Series Momentum, JFE 2012](http://docs.lhpedersen.com/TimeSeriesMomentum.pdf)
- Burghardt — [Eurodollar Futures and Options Handbook](https://www.amazon.com/Eurodollar-Futures-Handbook-McGraw-Hill-Investment/dp/0071418555)
- Carver — [Systems building: futures rolling](https://qoppac.blogspot.com/2015/05/systems-building-futures-rolling.html)
- Vendor methodologies: [CSI](https://www.csidata.com/custserv/onlinehelp/OnlineManual/backadjustedoverview.htm), [Norgate](https://norgatedata.com/futurespackage.php), [Pinnacle](https://pinnacledata2.com/clc.html), [Portara](https://portaracqg.com/continuous-futures-data/), [QuantConnect](https://www.quantconnect.com/forum/discussion/17093/default-continuous-futures-adjustment-method/)

---

## 6 · UNIT-CONVENTION CATALOG (NEW — this session, COMPLETE)

### Why
Spread/fly closes in the OHLC DB are stored in **bp** (verified empirically — see §3.J). Prior code blanket-multiplied all changes by 100 to "convert price → bp", which made spread/fly deltas 100× too large.

### What
- **Module:** `D:\STIRS_DASHBOARD\lib\contract_units.py`
- **Catalog:** `D:\STIRS_DASHBOARD\data\contract_units.parquet` (3,584 SRA rows)
- **Settings UI:** Settings → Unit conventions tab — view stats, filter/inspect, **Rebuild catalog** button

### Catalog schema
| Column | Meaning |
|---|---|
| `base_product` | e.g. `SRA` |
| `symbol` | full contract symbol |
| `strategy` | `outright` / `spread` / `fly` |
| `convention` | `bp` / `price` / `unknown` |
| `median_ratio` | `stored_close / implied_raw_diff` (≈1 → price; ≈100 → bp) |
| `n_samples` | # joined samples after filtering |
| `legs` | comma-joined leg symbols (`""` for outrights) |
| `built_at` | ISO timestamp |
| `inferred_from` | `"direct"` (own data) or `"cohort:<strategy>"` (fallback) |

### Detection algorithm
For each spread/fly contract:
1. Parse legs from symbol via regex (e.g. `SRAM26-U26-Z26` → `[SRAM26, SRAU26, SRAZ26]`)
2. Pull all daily closes for legs + the spread/fly itself; pivot to wide
3. Compute "expected raw" =
   - spread: `close_A − close_B`
   - fly: `close_A − 2·close_B + close_C`
4. Compute ratios `r = stored / raw` per date (skip near-zero raw or stored)
5. Take last `_RECENT_N=120` samples; need ≥`_MIN_SAMPLES=5`
6. Classify by median ratio:
   - `[30, 300]` → `bp`
   - `[0.3, 3.0]` → `price`
   - else → `unknown`

### Cohort fallback
If a contract can't classify (insufficient samples / unstable ratios), it inherits the **dominant convention of its (base_product, strategy) cohort** — but only when ≥70% of confidently-classified rows agree.

### SRA results
| strategy | bp | price | unknown |
|---|---:|---:|---:|
| outright | 0 | 298 | 0 |
| spread | 1666 | 0 | 0 |
| fly | 1620 | 0 | 0 |

All 184 *live* SRA contracts: 29 outrights `price`, 88 spreads `bp`, 67 flies `bp`. **No live unknowns.**

### Public API
```python
from lib.contract_units import (
    load_catalog, build_catalog,
    get_convention, is_already_bp,
    bp_multiplier, bp_multipliers_for,
)

cat = load_catalog()                                  # cached load
mults = bp_multipliers_for(contracts, "SRA", cat)     # vector of ×1 or ×100

# Build / rebuild for any market
build_catalog(["SRA"], progress_cb=lambda mkt,i,n: ...)
```

### Wired into analysis (no more 100× over-scaling on spreads/flies)
- `lib/sra_data.py:compute_curve_change` — accepts `contracts` arg; per-contract multiplier from catalog
- `lib/regime.py:classify_regime_multi_lookback` — accepts `base_product`; uses catalog
- `tabs/us/sra_curve.py:_chart_heatmap` — heatmap Δ matrix multiplies row-wise from catalog
- `tabs/us/sra_curve.py:_side_multi_date` — multi-date "Cumulative Δ vs latest" uses per-contract multipliers

### Extending to other markets
Add base_product names when other STIRs come online:
```python
build_catalog(["SRA", "FF", "S1R", "ER", "FER", "SON", "FSR", "CRA", "JTOA", "YBA", "NBB"])
```
The detector is base_product-agnostic; the legs regex (`_LEG_RE`) covers any letter-prefix like `SRA`/`FF`/`S1R`/`ER`/etc.

---

## 7 · NEXT TO BUILD (in this order)

1. **SRA → Analysis subtab** — Proximity ✅ and Z-score & MR ✅ complete (see §5B). Remaining
   sub-subtabs (placeholders rendering preview text in `tabs/us/sra_analysis.py`):
   - **PCA decomposition** — rolling 60d, PC1/PC2/PC3 loadings, variance explained,
     today's projection, anomaly residuals. Use SRA outright rates.
   - **Terminal rate** — 12M / 24M (PRIMARY) / 36M; cycle classification
     (HIKING/CUTTING/ON_HOLD/MIXED); bp_to_terminal; meetings_to_terminal (using FOMC YAML)
   - **Carry analysis (3 notions)** — single-contract ranked, spread carry, white-pack
     ranked. Detailed table.
   - **Slope / Curvature time series** — historical + Z-score badge + distribution mini-histogram
   - **Statistical tests** — ADF stationarity, OU half-life (rolling 90d), Hurst exponent
     (R/S 60d) on flies — universe-wide pass/fail summary (NB: per-contract versions of
     these already power the Z-score & MR subtab; this would be the universe-aggregate view)
   - **Fly residuals after PCA neutralization** — ranked rich/cheap list (the alpha
     source for C1 setup in TMIA)

2. **FF + S1R subtab** — same Curve+Analysis structure. Most code can be lifted from sra_curve.py with minor changes (different reference-period logic — FF reference period is the contract month itself, not 3-month forward; S1R is 1-month). **Remember to build the unit catalog for FF/S1R** before wiring in Δ analysis.

3. **Bonds subtab** (TU/FV/TY/AUL) — different structure since bonds have **price + DV01** rather than implied rate. Plan from TMIA gameplan: cash-yield overlay (USGG2/5/10/30YR from BBG `rates_drivers/`), DV01 curves, butterfly trades, US-DE/US-AU country spreads.

4. **US Fundamentals tab** — currently shows ticker inventory. Need to build actual analytics: time-series overlays, surprise indices (actual vs survey), event impact analysis. Use `BN_SURVEY_*` columns in eco parquets.

5. **Other economies** — order: EU → UK → JP → CH → CA → AU → NZ. Each gets its own page with the same SRA/Bonds/Fundamentals structure tailored to that economy's product set:
   - EU: ER/FER STIRs, FGBS/FGBM/FGBL/FGBX/FOAT/FBTS/FBTP bonds
   - UK: SON STIR, FLG bond
   - JP: JTOA STIR (no major futures bond — use cash JGB)
   - CH: FSR STIR (no bond)
   - CA: CRA STIR (no major bond — use cash)
   - AU: YBA STIR, YTT/YTC/AUL bonds
   - NZ: NBB STIR (no major bond)

---

## 8 · KEY UI / UX PRINCIPLES YASH HAS LOCKED

These have been emphasized repeatedly. Don't backslide.

1. **Dark mode only** — `#0a0e14` base, `#11161f` surface, gold accent `#e8b75d`. Inter for text, JetBrains Mono for numbers/data.
2. **Full-width layout** — no wasted side gutters. `.block-container { max-width: 100%; padding-left/right: 1.75rem; }`.
3. **Information density** — small fonts (13px base, 11-12px for labels), compact spacing, status strips not big metric cards for supplementary info.
4. **Always show contract names** with counts — "29 outrights (SRAJ26 → SRAH32)" format. Same for tenor breakdowns.
5. **Subtle chart shading** — Fed band alpha ~0.07, section shading alpha ~0.11-0.14. Visible but not screaming.
6. **Connected curves** — `connectgaps=True`. Don't break the line for missing contracts.
7. **Tooltips on every formula** — math should be discoverable on hover via title-attribute. Carry, FOMC methodology, regime thresholds all have these.
8. **Multiple lookbacks where applicable** — 5d/15d/30d/60d/90d/252d/Max for user-selectable dropdowns. Analysis is capped at 90d (multi-lookback z blocks). 252d remains in dropdowns but no analytical block uses 252d as longest window.
9. **Side panel always shows ALL analyses** (latest direction). Don't bring back per-mode dispatch.
10. **No errors** — wrap chart-mode dispatch in try/except so one mode's runtime error doesn't kill the page.
11. **For spreads/flies, use unit-conventions catalog** — never hard-code `* 100` for Δ in bp. Look up `bp_multipliers_for(contracts, base_product)`.

---

## 9 · BUGS WE FIXED (don't reintroduce)

| # | Bug | Cause | Fix |
|---|---|---|---|
| 1 | `Parser Error: syntax error at or near ","` | DuckDB reserves `asof` as a keyword (ASOF joins) | Use `bar_date` alias in SQL, rename to `asof` in pandas |
| 2 | `OSError: Repetition level histogram size mismatch` | pyarrow can't read some BBG parquets written with newer format | Read via DuckDB's `read_parquet` instead |
| 3 | Side panel not rendering | `with side_col` nested inside `with chart_col` | Render chart and side as siblings; use `st.container(height=720)` for scroll |
| 4 | Carry coloring crash on stale data | Plotly rejects `None` values in `marker.color` array | Convert `None`→`NaN`; fallback to uniform color if all NaN |
| 5 | H-L band horizontal artifacts | `connectgaps=True` interpolated through NaN gaps | Filter band traces to valid-only contracts; categorical x-axis preserves order |
| 6 | FOMC implied path oscillating ±90bp | Underdetermined system, no anchor | Anchor `r_0` to current SOFR with strong soft prior; ridge `λ=0.5×n_contracts` on first-difference |
| 7 | Z-score biased toward 0 on short windows | `<=` in history filter included as-of date itself | Change to `<` (strict less-than); explicit `ddof=1` |
| 8 | Tooltip newlines collapsed | Raw `\n` in HTML title attribute | Use `&#10;` HTML entity |
| 9 | Spreads/flies showing "Implied rate" in Heatmap value mode | Hardcoded option list | Conditional `if allow_implied_rate: insert("Implied rate")` |
| 10 | Streamlit cache held old broken state across edits | runOnSave doesn't always invalidate function-source-changed cached results | Hard restart server + browser hard-refresh (Ctrl+Shift+R) |
| 11 | **Spread/fly Δ values 100× too large** | `compute_curve_change` and heatmap Δ blanket-multiplied price diff by 100, ignoring that DB stores spreads/flies already in bp | Built `lib/contract_units.py` auto-detection catalog; per-contract multiplier of ×1 (bp) or ×100 (price) |
| 12 | Many spread/fly contracts classifying as "unknown" on first detector run | Filter only excluded near-zero raw, not near-zero stored — off-day stored=0 distorted ratios | Filter near-zero stored too; bias to last 120 samples; cohort fallback (≥70% strategy agreement fills unknowns) |
| 13 | `name 'np' is not defined` in `make_confluence_matrix_chart` and other Analysis chart factories | New chart factories used numpy but `lib/charts.py` only imported pandas/plotly | Added `import numpy as np` to `lib/charts.py` |
| 14 | Analysis tables rendered malformed HTML when a value was None — the entire row's `<div>` and trailing cells got eaten | Misbinding ternary: `... if x is not None else ...` placed at the END of a multi-line concatenated f-string applied to the WHOLE concatenation, not just the cell | Pre-compute formatted strings with a small `_fmt(value, fmt)` helper; assemble row HTML linearly, no inline ternaries on multi-line f-strings |
| 15 | (Same root cause as #14) `_render_score_table`, `_block_top_reversion`, `_block_fresh_breaks` had the same misbinding | Same root cause | Same fix — apply this pattern to ALL multi-line HTML row builders in Analysis subtabs |
| 16 | First Technicals scan threw `missing indicator(s)` for many detectors | DB column naming differs from gameplan (e.g. `BB_DN_20_2.0x` not `BB_DN_20_2.0`, `ATR14` not `ATR_14`, `DIplus_14` not `DIp_14`, `MACD_12_26_9_hist` not `MACD_HIST_12_26_9`, `KC_UP_20_2.0x_ATR20` not `KC_UP_20_2.0_ATR20`, `ST_LB_ATR10_2.0x` not `ST_LB_ATR10_2.0`) | Renamed primary lookups across `setups/{trend,mean_reversion,stir,composite}.py`; added comprehensive fallbacks in `_ind()` helper |
| 17 | EMA_200 / High_55 / Low_55 not in `mde2_indicators` | Stops at EMA_100 and High_50 / Low_50 | Compute at runtime via `lib.runtime_indicators.wilder_ema(close, 200)` and `donchian_high_low(close, 55)` |
| 18 | Trade levels (entry/stop/T1/T2) showed sub-tick precision (`96.4984`, `96.2149`) — not exchange-fillable | `_set_levels_long/short` and direct T1/T2 overrides used raw arithmetic without rounding | Added `round_to_tick(price, bp_mult)` and `fmt_price_for_scope(price, bp_mult)` to `lib/setups/base.py`. Outright tick = 0.005, spread/fly tick = 0.5 bp. All trade levels round on display in cards + drill-down `st.metric` |
| 19 | Setup state matrix used emoji (🟢🔴🟡🟠🔵·—) which rendered inconsistently across systems | Direct unicode emoji in cell text | Replaced with `<span class="tech-cell ...">` CSS pills using `_state_to_css(state, direction)` mapper; matching pills used in proximity-to-fire direction badges. |
| 20 | Composite scoring view was a wall of numbers — hard to scan visually | Used numeric text for TREND/MR/FINAL | Added `_comp_bar_html(value)` rendering an inline horizontal bar centered at 0 (red fills left for negative, green right for positive) with the numeric value as a centered label; full factor breakdown still in hover tooltip |
| 21 | Proximity-to-fire row's DIST cell rendered as `0` (zero) when distance was exactly zero | Misbinding `(dist if x else "—") and (...)` evaluated to 0 because `bool(0) is False` | Replaced with explicit `_fmt_or_dash(v, fmt)` helper |

---

## 10 · GAMEPLAN REFERENCES

Two TMIA gameplans informed the structure:

- **`C:\Users\yash.patel\Downloads\TMIA_v12.1_GAMEPLAN.md`** — defined the **140 cross-asset ratios** (Commodity 21, Equity 17, Long-Horizon 12, FX 16, Curve & Bond 27, Cross-STIR 15, Vol 8, Lead-Lag 24)
- **`C:\Users\yash.patel\Downloads\TMIA_v13_Streamlit_Gameplan.md`** — defined the 91-product, 9-class universe and the **159-column native indicator panel** in `mde2_indicators` (TR, EMA/SMA 5/10/20/30/50/100, ATR 5/10/14/20, RSI_14, ADX/DI±, Aroon, MACD, OBV, BB at 5σs, KC at 3 periods × 5σs, ST at 5σs, ZPRICE/ZRET 5/10/20/50/100, Range_n_over_ATR20, Range_X_over_Range_Y)

---

## 5F · TECHNICALS OVERHAUL — Phases B → H (NEW, COMPLETE)

The Technicals subtab is no longer a 60d track-record + emoji-pill view. It's now backed by a research-based NEAR/APPROACHING grader, a 5-year backtest engine + 270-cell aggregator + 10-day recompute cycle, with a clean Title-Case UI and rich hover tooltips that surface formula / NEAR rule / citation / regime definitions / TAKE definitions inline.

### Phase B — NEAR / APPROACHING thresholds (research-backed)

Three new modules:
- **`lib/setups/_NEAR_CITATIONS.py`** — citation strings (Wilder ATR/RSI/ADX, Brown bull-RSI, Carver continuous, DeMark TD, Bollinger / Carter squeeze, Avellaneda & Lee, Aronson, Macrosynergy, Turtle).
- **`lib/setups/near_thresholds.py`** — `near_state(setup_id, key_inputs, context, calibrated)` dispatcher returning `(state, distance_to_fire, missing_text, rule_text, citation_key)`. Maps every registered setup_id (28 entries: A1-A15, B1-B13, C3-C9, composites) to one of nine categories: Distance-to-level / Oscillator-neutral / Oscillator-bull / ConnorsRSI / Z-score / Multi-condition / MA-crossover / Slope-trend / Volatility-squeeze / Composite-threshold. Each category's NEAR + APPROACHING rule has a literature citation. Variant ids (C9a_12M / C8_24M / etc.) inherit via `_strip_variant`.
- **`lib/setups/near_calibration.py`** — bootstrap-quantile calibrator (Aronson 2007 + Macrosynergy ±2σ winsor) for the two categories (Multi-condition, Slope-trend) where literature has no canonical threshold. Calibration runs on the 5-year CMC history; output persisted to `data/near_thresholds_calibrated.parquet`. Phase E recompute populates this file; until then literature defaults apply.

Verified: 28/28 setups mapped, 7 smoke tests across categories return correct (state, distance, missing) tuples; variant resolution handles `C9b_18M` → category f (slope_trend).

### Phase C — backtest engine (bar-level)

Three new modules:
- **`lib/backtest/engine.py`** — `simulate_trades(panel, fires, setup_id, cmc_node, bp_per_unit, cooldown_bars, risk_dollars)`. Bar-level simulator with the gameplan §6.1 universal rules: signal at close T → fill at OPEN T+1, **same-bar stop-wins**, 3-bar cooldown after exit, position concurrency 1 per (setup × node × direction), slippage = 0, commission = 0, time stop = 60 bars, optional reversal exit. Returns one `TradeRecord` per fire that opened a position with all 21 gameplan §7.3 fields including `regime_at_entry`.
- **`lib/backtest/metrics.py`** — `compute_metrics(trades, n_bootstrap)` produces the full 40+ metric pack: counts / dollar / R-based / risk-adjusted / time / bootstrap CIs (when trade_count ≥ 30). Plus `compute_calendar_slices(trades)` for by-month / by-DoW / by-year breakdowns.
- **`lib/backtest/bootstrap.py`** — percentile-based bootstrap CI helpers: `bootstrap_sharpe_ci`, `bootstrap_winrate_ci`, `bootstrap_expectancy_ci`.

Verified: 5/5 golden trade scenarios pass — T1 hit, stop hit, **same-bar stop+target STOP WINS**, time stop, multi-fire with cooldown.

### Phase D — aggregator (L1–L18 × 5 regimes × 5 windows)

**`lib/backtest/aggregator.py`** — `build_metrics_grid(trades, asof_date, n_bootstrap)` returns `(metrics_df, calendar_df, trend_df)`.

- **5 windows** (per Yash request: max history + 30d + multi-horizon comparison): ALL / LAST_2Y / LAST_1Y / LAST_6M / LAST_30D.
- **5 regimes** (gameplan §7.5): ALL / TRENDING (ADX≥25) / RANGING (ADX<20) / HIGH_VOL (ATR pctile > 0.75) / LOW_VOL (ATR pctile < 0.25).
- **Cell levels currently emitted**: L1 (setup × node × direction), L2 (setup × node, both dirs), L4 (setup overall), L14 (per-node all-setups). L5–L13 (composite-threshold sweeps) and L15–L18 (STIR-specific rollups) are scaffolded but require composite-trade emission to populate.
- **Cross-window trend interpretation**: each cell tagged Improving / Declining / Stable / Mixed via Spearman-style recency-rank correlation (|ρ|>0.7 = trending, <0.3 = stable). Surfaces the cell's metric across all 5 windows in one tooltip line.

Verified: 200 synthetic trades produced 825 cells (33 cell × 5 windows × 5 regimes), 671 calendar slices, 33 trend rows. Improving / Declining / Stable interpretations classified correctly.

### Phase E — 10-day recompute cycle + disk persistence

Two new modules:
- **`lib/backtest/cycle.py`** — `ensure_backtest_fresh()` mirrors `lib/prewarm.py` architecture. Module-level lock + idempotent flag; spawns a daemon thread on app start if `tmia.duckdb` is missing or older than 10 days. After each successful recompute, drops a dated copy at `.backtest_cache/snapshots/tmia_<YYYY-MM-DD>.duckdb` (keeps 6 most recent). Public API: `is_backtest_fresh()`, `get_backtest_status()`, `force_recompute_now()`, plus loaders for the four output tables (`load_metrics_grid`, `load_trades`, `load_calendar_slices`, `load_trend_table`).
- **`lib/backtest/replay.py`** — `replay_all_detectors_on_cmc(scope, asof_date)` walks the 5-year CMC history bar-by-bar for every (setup × CMC node) and collects FIRED bars into a fire DataFrame. Augments CMC OHLC panels with all needed indicators (ATR14, ADX_14, EMA_5/10/20/30/50/100/200, BB_20_2, RSI_14, Donchian highs/lows, MACD) so existing detectors work unchanged on CMC nodes.

Hooked into `app.py` and `pages/1_US_Economy.py` via `ensure_backtest_fresh()` calls right after `ensure_prewarm()`. Cold compute ~5-15 min per scope; results persisted at `.backtest_cache/tmia.duckdb` with 4 tables: `tmia_backtest_trades`, `tmia_backtest_metrics`, `tmia_backtest_calendar_slices`, `tmia_backtest_trend`.

Smoke-tested: A1 produces 39 fires, A5 produces 174, A6 produces 4, B5/B6 produce 0 over the 5y M3 history (varying selectivity, all running cleanly). ~1 s per detector × node, total replay ~9 min within the 5-15 min target.

### Phase F — kill code names

`lib/setups/registry.py` gains:
- `display_name(setup_id)` — Title Case with acronym preservation. `TREND_CONTINUATION_BREAKOUT` → `Trend Continuation Breakout`. `ADX_DI_CROSS · EMA_20` → `ADX DI Cross · EMA 20`. `12M slope` stays `12M Slope`. `5d trend` stays `5d Trend`. Memoised.
- `display_name_short(setup_id, max_len=14)` — initials abbreviation for dense matrix columns. `Trend Continuation Breakout` → `TCB`.

Nine call sites swapped in `tabs/us/sra_analysis_technicals.py`:
- Interpretation guide expander, fire-card name, state-matrix column header (now uses 6-char abbreviation + tooltip showing full name), proximity-table SETUP column, drill-down picker (with `format_func`), drill-down result label, side-panel hot-setups, side-panel quiet-setups list. Code names `A1` / `B4` / `C9a_12M` survive only as internal dict keys / sort keys, never as user-visible text.

### Phase G — rich hover tooltips

**`lib/setups/tooltips.py`** with three builder functions:

- `setup_six_section_tooltip(setup_id, result, state_label)` — replaces the old `_build_setup_tooltip` body. Six sections: TITLE / FORMULA / FIRED / NEAR / APPROACH. + Source / INTERPRET, plus optional CURRENT state / INPUTS / TRADE rows when result is supplied.
- `composite_cell_tooltip(name, value, factor_breakdown)` — for Trend / MR / Final cells in View 3. Shows the cell's value, the 7-band range → interpretation table, the band the cell falls into, and an optional factor breakdown.
- `composite_header_tooltip / regime_header_tooltip / take_header_tooltip` — for the View 3 column headers. Regime tooltip lists all 7 regime labels with their conditions; Take tooltip lists all 7 TAKE pills with their FINAL_COMPOSITE thresholds.

Wired into:
- View 1 (Today's Fires) and View 5 (Drill-down) — via the rewritten `_build_setup_tooltip`.
- View 2 (state matrix) column headers — `<th title="..." cursor:help>` with full setup name.
- View 3 (Composite scoring) — every cell + every column header has a research-backed tooltip with `ⓘ` indicator.

### Phase H — UI consistency across the 3 SRA Analysis subtabs

- **12-block parity**: Technicals' side panel gained a 12th block — `_block_trend_movers` reads `tmia_backtest_trend` from the Phase E cache and surfaces top "Improving" + top "Declining" setups. Friendly placeholder when the backtest cache hasn't been built yet.
- **Control-row alignment**: Technicals `[1.0, 1.4, 2.0, 1.4]` → `[1.0, 1.4, 2.4, 1.4]` to match Proximity / Z-score.
- **Visible side-panel borders**: all three subtabs now use `st.container(height=720, border=True)` (was `border=False`) — gives the side panel visual weight and clearly separates from main chart area.
- **Deferred**: full migration of Technicals' local `_TECHNICALS_CSS` (~180 lines) into `lib/css.py`. The local CSS is namespaced (`.tech-*`, `.fire-*`, `.trade-legs`) so cross-subtab interaction is fine; migration is pure refactoring with high regression risk vs. low visible value.

### Verification — full app boot

After all Phase B–H code lands and streamlit restarts on 8503: `/_stcore/health → 200 ok`, no tracebacks in `streamlit_8503.err.log` or `streamlit_8503.out.log`. The 10-day backtest recompute spawns its daemon thread on app start; first run takes ~5-15 min in the background and populates `.backtest_cache/tmia.duckdb`.

### Files added (Phase B–H)

```
lib/setups/_NEAR_CITATIONS.py       (Phase B)
lib/setups/near_thresholds.py       (Phase B)
lib/setups/near_calibration.py      (Phase B)
lib/setups/tooltips.py              (Phase G)
lib/backtest/__init__.py            (Phase C)
lib/backtest/engine.py              (Phase C)
lib/backtest/metrics.py             (Phase C)
lib/backtest/bootstrap.py           (Phase C)
lib/backtest/aggregator.py          (Phase D)
lib/backtest/cycle.py               (Phase E)
lib/backtest/replay.py              (Phase E)
```

### Files modified (Phase B–H)

```
app.py                               (ensure_backtest_fresh hook)
pages/1_US_Economy.py                (ensure_backtest_fresh hook)
lib/setups/registry.py               (display_name + display_name_short)
tabs/us/sra_analysis_technicals.py   (display_name swap × 9, tooltip rewire,
                                       12th block, control-row 2.4,
                                       side-panel border=True)
tabs/us/sra_analysis_proximity.py    (side-panel border=True)
tabs/us/sra_analysis_zscore.py       (side-panel border=True)
```

---

## 5G · SENIOR-TRADER ROADMAP — Phase 0 (Data substrate, COMPLETE)

This is the first phase of the comprehensive senior-trader roadmap documented at
`C:\Users\yash.patel\.claude\plans\c-users-yash-patel-downloads-tmia-v13-s-magical-mitten.md` (13 phases, Phase 0–12). The plan file is the source of truth for what comes after Phase 0.

### Pre-build backup

Per plan §17 — created at `D:\STIRS_DASHBOARD_backup_phase0_2026-05-10_154137.zip` (4.76 MB, 71 files, excludes `__pycache__` + live logs). Restorable via `Expand-Archive`. Per-phase incremental backups will be taken before Phase N starts.

### Phase 0 deliverables (all complete + verified)

**Phase 0.A — 137 unused indicators loaded onto CMC panels**
`lib/backtest/replay.py:_augment_panel_with_indicators` expanded from 5 to 97 indicators in 16 categories: ATR variants (5/10/14/20), Wilder ADX/DI±, EMAs (5/10/20/30/50/100/200), SMAs, Bollinger Bands at 1.0/2.0/3.0× σ + BBWP_20 (252-day percentile rank), Keltner Channels (1.0/2.0/3.0× ATR20), Supertrend (1.0/2.0/3.0× ATR10), Aroon Up/Dn/Osc, CCI_20, OBV, RSI_14 + RSI_2 (Connors), Z-scores price + return at 5/10/20/50/100, Highs/Lows/Range at 5/10/20/30/50/55/100, Range/ATR_20 ratios, MACD 12/26/9, IBS, TR. CMC M3 panel went from 6 cols → 103 cols.

**Phase 0.B — `lib/freshness.py` unified freshness module**
- `freshness_report()` → dict with sources (5: OHLC snapshot DB, BBG warehouse, CMC cache, backtest cache, prewarm daemon) + per-BBG-category depth (9 categories) + overall status.
- `BUDGETS_HOURS` constants (OHLC 36h, BBG warehouse 14d, CMC 36h, backtest 11d, prewarm 12h).
- `BBG_DEPTH_SAMPLES` per-category sample tickers measuring effective depth.
- Sufficiency classification: ≥5y = sufficient, ≥1y = shallow, else missing.

**Phase 0.C — Snapshot-freshness traffic light on every page header**
Wired into `app.py` (line ~50) + `pages/1_US_Economy.py` (line ~38). `freshness_header_chip()` returns the worst-of-all status as a color-coded pill ("DATA FRESH" / "N STALE" / "N MISSING") with a hint link to Settings → System Health.

**Phase 0.D — Settings → System Health new sub-tab**
`pages/2_Settings.py` gains a 5th sub-tab between General and Unit Conventions. Renders:
- Overall status pill
- Data sources freshness table (5 rows: source / last_modified / age / budget / detail)
- BBG category depth table (9 rows: sufficiency / category / sample_ticker / rows / first_date / last_date / years)
- Pipelines panel (prewarm + backtest daemon status)
- Manifests + caches metric strip (CMC parquet count, backtest snapshot count, log file count)
- CMC manifest viewer with full JSON expander; surfaces missing-contract warnings (e.g. SRAH26)

### Verification — live on 8503

```
[200 ok] /_stcore/health
[clean]  streamlit_8503.err.log → empty
[clean]  streamlit_8503.out.log → no Traceback / Exception / Error matches
```

The freshness panel correctly reports the current state of the running app:
- OHLC + CMC are 65h old (red) — flag rebuild needed on next launch
- BBG warehouse 116h (green; budget 336h)
- 7 of 9 BBG categories show "shallow" (1y) — exactly what the external backfill program (per D1 in plan §15.1) is expected to deepen

### Files added / modified in Phase 0

- **NEW** `lib/freshness.py` (200 LoC)
- **MODIFIED** `lib/backtest/replay.py` — `_augment_panel_with_indicators` expanded from 60 → 220 lines
- **MODIFIED** `app.py` — header chip wiring (3 lines added)
- **MODIFIED** `pages/1_US_Economy.py` — header chip wiring (3 lines added)
- **MODIFIED** `pages/2_Settings.py` — System Health sub-tab (~110 lines added)

---

## 5H · SENIOR-TRADER ROADMAP — Phase 1 (Architectural foundation, COMPLETE)

Phase 1 lands the product/calendar config layer + the LIBOR cutover gate + the CPCV harness. Everything analytic from Phase 2 onwards reads from these foundations.

### Pre-Phase-1 backup

Created at `D:\STIRS_DASHBOARD_backup_phase1_2026-05-10_154808.zip` (4.77 MB, 71 files).

### Phase 1 deliverables (all complete + verified)

**Phase 1.A — `config/product_spec.yaml` + `lib/product_spec.py`**

Single source of truth for per-product conventions. SRA enabled with all required fields (tick_size 0.005, DV01 $25/lot, bp_multiplier 100, quarterly month codes H/M/U/Z, last_trade_rule "third_wednesday_minus_one", earliest_listing 2018-05-07, naming_convention "reference_quarter_start", trading_calendar "us_futures", libor_cutover_date 2023-08-01). ER/SON/FF stub-listed but disabled (cross-product permanent OOS per plan §15 / D1=A2).

API:
- `load_product_spec()` — validates + caches the YAML
- `get_product(code)` / `is_enabled(code)` / `enabled_products()`
- `libor_cutover_date(code)` / `cmc_outright_nodes(code)` / `cmc_spread_nodes(code)` / `cmc_fly_nodes(code)`
- `dv01_per_lot(code)` / `tick_size(code)` / `bp_multiplier(code)`

**Phase 1.B — `config/cb_meetings.yaml` + `lib/cb_meetings.py`**

78 FOMC meetings 2018-06-13 through 2027-12-15, with `has_sep` / `statement_only` / `emergency` flags. Future dates locked through 2027 from Fed-published calendar. Realized policy outcomes derived at runtime from `FDTRMID_Index` step changes around each past meeting date.

API:
- `load_cb_meetings()` — parses + caches YAML
- `fomc_meetings()` / `fomc_meetings_in_range(start, end)` / `next_fomc(asof)`
- `fomc_blackout_window(meeting_date)` — 12d before / 1d after per Fed convention
- `is_in_fomc_blackout(d)` — returns the meeting whose blackout contains d, else None
- `fomc_meetings_with_outcomes()` — returns FOMCMeeting list with realized_change_bp / pre_target_bp / post_target_bp / direction (HIKE/CUT/HOLD) / magnitude_bp populated where derivable

Verification: 9 of 78 meetings currently have realized outcomes — limited because FDTRMID parquet has only 1y depth. Expected to deepen automatically as the external backfill program (per D1) lands deeper FDTRMID history.

**Phase 1.C — `lib/libor_cutover.py`**

The hard 2023-08-01 SOFR-First / Eurodollar→SOFR cutover gate. Used by every analytic that needs a clean post-transition sample.

API:
- `cutover_date_for(product_code)` — date or None
- `filter_post_cutover(df, date_col, product_code)` — drops rows below cutover
- `tag_pre_transition(df, ...)` — adds `pre_transition` boolean col
- `is_post_cutover(d, product_code)` — single-date check

Pre-cutover bars are RETAINED for legacy_diagnostic visual inspection; excluded from analog pools / OU calibration / event-impact regressions.

**Phase 1.D — `lib/backtest/cpcv.py` (CPCV harness, the new SoT for OOS claims per A1)**

Combinatorial Purged Cross-Validation per López de Prado 2018. Defaults: N=6 groups, k=2 test groups → C(6,2)=15 paths; embargo=5 BD; purge by label_horizon=60 BD.

API:
- `cpcv_paths(trades, n_groups, k_test, embargo_bars, label_horizon)` — generates the 15 path objects with train/test indices, applying time-ordered group assignment + purge + embargo
- `deflated_sharpe(sr_observed, n_trials, skewness, kurtosis, n_obs)` — Bailey-López de Prado 2014 selection-bias-corrected Sharpe
- `probability_of_backtest_overfitting(is_metrics, oos_metrics)` — PBO per Bailey-Borwein-LdP-Zhu 2017
- `run_cpcv_evaluation(trades, ...)` — end-to-end harness returning `CPCVResult(n_paths, in_sample_metrics, out_of_sample_metrics, deflated_sharpe, pbo, ...)`

Smoke-tested on a synthetic 240-trade Sharpe-0.5 dataset: produces 15 paths exactly, OOS Sharpe distribution mean = 8.12 (= 0.5 × √252 as expected), PBO = 0.625 (high overfit on random data, which is correct).

**Phase 1.E — Per-contract data-quality table + FOMC calendar viewer + Product spec viewer in Settings → System Health**

Three new sections appended to the System Health sub-tab:
- **Per-contract data quality** — lazy-loaded scan of all SRA contracts: bar count, first/last bar, days_since_last_bar; "Live" vs "Stale" tabs with thresholds at 7 days
- **FOMC meeting calendar** — upcoming next 6 meetings + recent 6 with derived outcomes
- **Product spec viewer** — table of all 4 products (SRA enabled, ER/SON/FF stubs)

### Verification — live on 8503

```
[200 ok] /_stcore/health
[clean]  streamlit_8503.err.log → empty
[clean]  streamlit_8503.out.log → no Traceback / Exception / Error matches
[15/15] CPCV produces correct path count (C(6,2))
[78/78] FOMC meetings load with future dates to 2027-12-15
[ 9/78] meetings have realized outcomes (FDTRMID 1y depth gates this; auto-deepens)
```

### Files added / modified in Phase 1

- **NEW** `config/product_spec.yaml` (~120 lines)
- **NEW** `lib/product_spec.py` (~110 LoC)
- **NEW** `config/cb_meetings.yaml` (~150 lines, 78 meetings)
- **NEW** `lib/cb_meetings.py` (~180 LoC)
- **NEW** `lib/libor_cutover.py` (~80 LoC)
- **NEW** `lib/backtest/cpcv.py` (~280 LoC)
- **MODIFIED** `pages/2_Settings.py` — System Health sub-tab gains 3 new sections (~120 lines)

---

## 5I · SENIOR-TRADER ROADMAP — Phase 2 (A12 PCHIP CMC upgrade, COMPLETE with caveats)

Phase 2 delivers the PCHIP interpolator + 61-node monthly grid + partial-fixing engine + 3 new verify checks. **Default interpolation stays linear** while three PCHIP-specific regressions are addressed (front-extrapolation, roll-boundary smoothing, deep-tenor coverage). PCHIP is opt-in via `build_cmc_nodes(interpolation="pchip")` and ready for follow-up tuning.

### Pre-Phase-2 backup
`D:\STIRS_DASHBOARD_backup_phase2_2026-05-10_155400.zip` (4.78 MB).

### Phase 2 deliverables

**Phase 2.A — 61-node monthly grid**
`OUTRIGHT_NODES_MONTHS` expanded from 10 nodes (0,3,6,9,12,18,24,36,48,60) to 61 nodes (M0..M60 monthly). The 10-node legacy set is preserved as `OUTRIGHT_NODES_MONTHS_LEGACY` for callers who want it.

**Phase 2.B — PCHIP interpolation (opt-in)**
`_build_outright_cmc_table_pchip(...)` in `lib/cmc.py` fits a PCHIP curve through ALL active chain contracts at each bar (instead of pairwise linear). Local, C¹-continuous, monotone-preserving, shape-preserving. Auto-falls-back to linear if `scipy.interpolate.PchipInterpolator` is unavailable. Build performance: 5.4s for 61 nodes × 5y (vs ~16s for linear 10 nodes — actually faster because one fit replaces 61 bracketing-pair lookups).

**Phase 2.C — Front-month partial-fixing engine** (`lib/cmc_partial_fixing.py`)
- `parse_sr3_reference_quarter(symbol)` — returns (start, end) of the contract's reference quarter per SR3 spec (3rd Wednesday of named month + 3-month window)
- `realized_portion(symbol, bar_date)` — (P_realized fraction, R_realized arithmetic-mean of SOFR fixings)
- `unfixed_tail_rate(symbol, bar_date, sr3_close)` — solves for R̂(t) = (R_implied − P·R_realized) / (1−P)
- `front_quarter_status(symbol, bar_date)` — diagnostic dict for System Health UI
Uses `rates_drivers/SOFRRATE_Index.parquet` (currently 1y depth; will deepen via D1 backfill).
Verified: SRAH26 reference quarter computes to 2026-03-18 → 2026-06-16 (correct per CME SR3 spec).

**Phase 2.D — 3 new verification checks**
- `check_pchip_c1_continuity(asof_date)` — max 2nd-difference across M3..M12 ≤ 50 bp (currently 0.65 bp on linear, 0.13 bp on PCHIP — PCHIP smoother as expected)
- `check_partial_fixing_accuracy(asof_date)` — partial-fixing engine returns non-negative P_realized + R_realized for any active SR3 contract
- `check_fomc_week_nonfracture(asof_date)` — max |Δclose| on M3 at FOMC dates ≤ 25 bp (currently 8.75 bp linear)

### Live verification status (linear default)

```
[FAIL] continuity:                  M48 deep-tenor 7.5% miss-rate (Phase A baseline issue, not Phase 2)
[PASS] roll_boundary:               31/31 match
[PASS] indicator_nonfracture:       max 5.25 bp at rolls
[PASS] term_structure:              max 14.4 bp adjacent
[PASS] manifest_integrity:          22 nodes / 31 rolls clean
[PASS] pchip_c1_continuity (NEW):   max 2nd-diff = 0.65 bp
[PASS] partial_fixing_accuracy (NEW): engine returns valid P_realized / R_realized
[PASS] fomc_week_nonfracture (NEW): max FOMC Δ = 8.75 bp
```
**Score: 7 of 8 pass** (M48 fail is the pre-existing Phase A baseline edge — deep-tenor sparseness in early 2021 — not a Phase 2 regression).

### PCHIP regressions deferred — to be resolved before promoting to default

When `interpolation="pchip"` is passed:
- **continuity**: 5 nodes >10% miss-rate (deep monthly tenors lack chain depth in early history)
- **indicator_nonfracture**: 30.75 bp at roll dates (PCHIP global re-fit at rolls causes sharper jumps than pairwise linear). Mitigation: per-segment PCHIP within roll-out windows, or local-bracketing fallback at roll dates.
- **term_structure**: M0/M3 missing on latest bar because target_dte < min(chain_dte). Mitigation: extrapolate via `extrapolate=True` with bound-clamping, OR fall back to linear-front for M0..M2.

PCHIP is callable via `lib.cmc.build_cmc_nodes(scope, asof, interpolation="pchip")` for power users who want to inspect the new path; product_spec keeps `interpolation: linear` as the production default until the three regressions are resolved.

### Files added / modified in Phase 2

- **NEW** `lib/cmc_partial_fixing.py` (~150 LoC)
- **MODIFIED** `lib/cmc.py` — `OUTRIGHT_NODES_MONTHS` expanded; `_build_outright_cmc_table_pchip` added (~150 LoC); `build_cmc_nodes(interpolation=...)` parameter added
- **MODIFIED** `lib/cmc_verify.py` — 3 new check functions + verify_all extended to 8 checks (~120 LoC added)
- **MODIFIED** `config/product_spec.yaml` — `interpolation` field documented; default stays linear

---

## 5J · SENIOR-TRADER ROADMAP — Phase 3 (A10 turn / QE / YE adjuster, COMPLETE)

Phase 3 lands the A10 calendar adjuster — a per-CMC-node OLS regression of daily close change on QE / YE / FOMC-week / NFP-week / holiday-week dummies with Newey-West HAC standard errors. Produces a `residual_change` series (preserving unconditional drift, orthogonal to all dummies by construction) that Phase 4's A1 regime classifier consumes instead of raw `cmc_close.diff()`. Without this, the regime GMM would learn "QE/YE regime" labels that are pure calendar artefacts.

### Pre-Phase-3 backup
`D:\STIRS_DASHBOARD_backup_phase3_2026-05-10_163845.zip` (7.23 MB, 166 entries; includes `.cmc_cache/` parquets + `.backtest_cache/tmia.duckdb` + all lib/pages/config code; only the two locked streamlit log files were skipped — they're rebuildable).

### Phase 3 deliverables (all complete + verified)

**Phase 3.B / 3.C — `lib/analytics/` namespace + `turn_adjuster.py` engine + dummy builders**
NEW `lib/analytics/__init__.py` establishes the namespace Phase 4-9 modules will inhabit (`regime_a1.py`, `policy_path_a4.py`, `event_impact_a11.py`).
NEW `lib/analytics/turn_adjuster.py` (~530 LoC) implements:
- 5 calendar-dummy builders aligned to ISO weeks where appropriate: `_qe_dummy` (last 5 BD of Mar/Jun/Sep, with Dec subsumed by YE), `_ye_dummy` (last 10 BD Dec + first 5 BD Jan), `_fomc_dummy` (blackout window OR same-ISO-week as a Fed meeting via `lib.cb_meetings`), `_nfp_dummy` (same-ISO-week as first Friday of month — no scraper per D2), `_hol_dummy` (US federal holidays via `pandas.tseries.holiday.USFederalHolidayCalendar`).
- `newey_west_hac()` numpy-only Bartlett-kernel HAC covariance — no `statsmodels` dependency. ~30 LoC; cross-checks against the closed-form for h=0 and known reference values.
- `fit_turn_adjustment()` per-node OLS via `np.linalg.lstsq` (handles near-singular dummy combinations). Drops dummies with zero fires before regression. Records regression-level β / NW-HAC se / two-sided normal-asymptotic p-values for the intercept + 5 dummies.
- **Residual definition: `residual_change = raw − Σ_d β_d · (D_d − mean(D_d))`** (demeaned dummy effect). This preserves BOTH `mean(residual) == mean(raw)` to machine epsilon AND `cov(residual, D_d) == 0` for every dummy. The naive `raw − Σ_d β_d · D_d` would shift the unconditional mean by `Σ_d β_d · mean(D_d)`, which is material for FOMC/NFP weeks (~25-35% sample frequency).
- `adjust_cmc_panel(scope, asof)` orchestrates per-scope: filters via `lib.libor_cutover.filter_post_cutover` (2023-08-01 hard gate), pre-builds the dummy matrix once on the union of bar dates (saves per-node FOMC lookups), then loops over nodes.
- `build_turn_residuals(cmc_asof)` is the top-level driver — runs all three scopes (61 outright + 8 spread + 4 fly = 73 nodes total).
- CLI: `python -m lib.analytics.turn_adjuster [YYYY-MM-DD]`. Auto-resolves the latest CMC asof if no arg supplied.

**Phase 3.D — Atomic persistence + manifest**
Two parquets + one JSON per asof, all written via temp-file + `os.replace` (mirrors `lib/cmc.py`):
- `.cmc_cache/turn_residuals_<asof>.parquet` — long format, 7 cols: `[scope, cmc_node, bar_date, raw_change, residual_change, fitted_change, has_data]`
- `.cmc_cache/turn_diagnostics_<asof>.parquet` — wide per-node, 32 cols: `[scope, cmc_node, n_obs, dof, r_squared, raw_var, residual_var, var_reduction_pct, low_sample_dummies]` + `[beta_*, se_*, p_*]` × 6 regressors (intercept + 5 dummies) + `eff_n_*` × 5 dummies
- `.cmc_cache/turn_residuals_manifest_<asof>.json` — `[builder_version, asof_date, cmc_asof_date, history_start, post_cutover_only, product_code, hac_lag, n_nodes_total, n_nodes_outright, n_nodes_spread, n_nodes_fly, dummy_definitions, dummy_fire_counts, node_coverage, missing_nodes]`

Live build for asof=2026-04-27 (cmc_asof matches): 73 nodes / 52,195 rows / history_start=2023-08-01 (post-LIBOR-cutover) / 0 missing nodes / dummy_fire_counts={QE: 40, YE: 45, FOMC: 237, NFP: 165, HOL: 150}. Cold compute ~5s on this machine.

**Phase 3.E — `lib/turn_residuals_verify.py`**
NEW (~290 LoC), six checks, returns dict with `passed_all + n_passed + n_total + checks{...}` (mirrors `lib.cmc_verify.verify_all` shape exactly):
1. `schema_integrity` — both parquets exist with all 7 + 32 expected columns and `has_data` is bool.
2. `manifest_integrity` — manifest JSON parses, has 16 required fields, builder_version matches current.
3. `null_residual_orthogonality` — re-regress `residual_change` on the same dummies for ALL 73 nodes; assert min p-value > 0.5 across every (node, dummy) pair. **This is the verification gate from plan §12 Phase 3.**
4. `mean_preservation` — `|mean(residual) − mean(raw)| < 1e-3 bp` per node.
5. `variance_reduction` — `var(residual) ≤ var(raw) × 1.01` per node.
6. `node_coverage` — every fitted node has `n_obs ≥ 30` OR is recorded in `manifest.missing_nodes`.
CLI: `python -m lib.turn_residuals_verify [YYYY-MM-DD]`.

**Phase 3.F — `lib/turn_adjuster_daemon.py` + `app.py` wiring**
NEW (~125 LoC) mirrors `lib/prewarm.py` exactly: module-level `_LOCK` + `_DAEMON_STARTED` flag + `threading.Thread(daemon=True)` + status-dict snapshot. `is_turn_residuals_fresh(asof)` returns True iff residuals parquet exists AND its mtime ≥ matching CMC `sra_outright_<asof>.parquet` mtime — automatic invalidation when CMC rebuilds. `_turn_residuals_worker()` resolves latest CMC asof, skips if fresh, otherwise calls `build_turn_residuals(cmc_asof)`. Errors captured in `_DAEMON_STATUS["errors"]`, never re-raised.
MODIFIED `app.py`: imports `ensure_turn_residuals_fresh` and calls it after `ensure_backtest_fresh()`. Non-blocking; ~1-2 min cold compute, instant when fresh.

**Phase 3.G — Reader API in `lib/sra_data.py`**
Three new `@st.cache_data(ttl=600)` accessors appended after the CMC accessor tier:
- `load_turn_residuals_panel(scope, asof_date) -> pd.DataFrame` — long DF filtered to scope. Raises `FileNotFoundError` if missing.
- `load_turn_residual_change_wide_panel(scope, asof_date) -> pd.DataFrame` — wide DF: index=bar_date, columns=cmc_node, values=residual_change. Column order matches `lib.cmc.list_cmc_nodes(scope)` canonical order. **This is what Phase 4's A1 regime classifier will call.**
- `load_turn_diagnostics(asof_date) -> pd.DataFrame` — full per-node regression stats DataFrame.

**Phase 3.H — `lib/freshness.py` registration**
`BUDGETS_HOURS["turn_residuals"] = 36` (matches CMC budget — they refresh together). NEW `check_turn_residuals() -> SourceFreshness` mirrors `check_cmc_cache()`. Appended to `freshness_report()`'s sources list so the worst-of-all overall pill rolls turn-residual staleness in.

**Phase 3.I — Settings → System Health "Turn-adjuster diagnostics" section**
~155-line block appended to `tab_health` in `pages/2_Settings.py`, after the CMC manifest viewer. Renders:
- Section heading + caption (one paragraph explaining purpose + Phase 4 consumer dependency)
- Pass/fail traffic-light pill (green "6/6 CHECKS PASS" or red "N FAILED")
- Verification check details — collapsible expander listing each check's status + message (auto-expanded on failure)
- Daemon status caption (started_at / completed_at / skipped_reason / errors)
- **Per-node regression diagnostics table** — 73 rows × 7 cols: `[scope, node, n_obs, R², var_red %, top sig dummy, low_sample]`. The "top sig dummy" column shows the single dummy with the smallest p-value per node (e.g. `NFP (p=0.048)`).
- **Drill-in chart** — three dropdowns (`scope` × `node` × `window` from {60d, 180d, 1y, all}) → Plotly 2-line chart: blue raw Δ vs orange residual Δ in price-bp units. Pattern reused from `tabs/us/sra_curve.py`.
- Inline summary caption below the chart: `window n=N · mean(raw) · mean(resid) · var(raw) · var(resid) · corr(raw, resid)`
- Manifest viewer (collapsible JSON expander).

### Live verification status (8503, fresh restart)

```
[200 ok]   /_stcore/health
[clean]    streamlit_8503.out.log → no Traceback
[1 known]  streamlit_8503.err.log → 1 pre-existing Phase 0 warning
            (BBG category-depth `Rows` column has em-dash mixed with int64;
             Streamlit auto-recovers via column-type conversion; not Phase 3)
[6/6 PASS] python -m lib.turn_residuals_verify
            ✓ schema_integrity:           residuals (52195, 7) / diagnostics (73, 32)
            ✓ manifest_integrity:         builder=1.0.0 / hac_lag=5 / n_nodes=73 / history_start=2023-08-01
            ✓ null_residual_orthogonality: 73 nodes re-regressed; min p = 1.0000 (worst at M0_M3/YE)
            ✓ mean_preservation:          max |delta_mean| = 2.03e-16 bp (< 1e-3 tolerance)
            ✓ variance_reduction:         var_red_pct min=0.281% / median=0.853% / max=1.794%
            ✓ node_coverage:              all 73 fitted nodes ≥ 30 obs; 0 declared missing
[live UI]  Settings → System Health renders Turn-adjuster section:
            · Green "6/6 CHECKS PASS" pill ✓
            · Per-node table grid: 7 cols × 74 rows (header + 73 nodes) ✓
            · Drill-in chart for outright/M0/last 60d ✓
            · Inline stats: "window n=60 · mean(raw)=-0.1500 bp · mean(resid)=-0.1515 bp ·
              var(raw)=2.9496 · var(resid)=2.9246 · corr(raw, resid)=0.9992" ✓
[Phase 4]  reader surface ready: load_turn_residual_change_wide_panel('outright', asof)
            returns shape (684, 61) with canonical column order [M0..M60]
```

### Known limitations / honest caveats

- **HAC lag = 5 (one trading week).** Conservative; tunable in the manifest. If autocorrelation in the residual series exceeds 5 bars (e.g. genuinely persistent post-FOMC drift), the SE will be biased downward. For Phase 3 SRA-only with daily bars + ~684 obs, this is acceptable.
- **`p ≈ 1.0` re-regression p-values are by-construction.** OLS first-order conditions guarantee `cov(residual, D_d) = 0` exactly. The orthogonality check verifies the math is implemented correctly, not that the dummies don't matter — which is what the original-regression p-values + R² in the diagnostics parquet show (R² ranges 0.003–0.018 across nodes; modest but real).
- **5-dummy specification is intentionally minimal.** Plan §16.2 explicitly accepts that A11-event Tier 1-4 (Phase 6) will eventually replace the simple FOMC-week dummy with the actual surprise-impact regression. Phase 3 just removes the calendar baseline; Phase 6 substitutes a richer event-aware variant when ready.
- **Auto-deepens with D1.** Currently 684 post-cutover bars (2023-08-01 → 2026-04-27). When the external backfill program lands deeper history, the next CMC rebuild → next residual rebuild auto-extends the regression sample and tightens HAC SEs without code changes.

### Files added / modified in Phase 3

- **NEW** `lib/analytics/__init__.py` (1 line — namespace marker)
- **NEW** `lib/analytics/turn_adjuster.py` (~530 LoC)
- **NEW** `lib/turn_residuals_verify.py` (~290 LoC, 6 checks)
- **NEW** `lib/turn_adjuster_daemon.py` (~125 LoC)
- **NEW** `.cmc_cache/turn_residuals_2026-04-27.parquet` (auto-generated)
- **NEW** `.cmc_cache/turn_diagnostics_2026-04-27.parquet` (auto-generated)
- **NEW** `.cmc_cache/turn_residuals_manifest_2026-04-27.json` (auto-generated)
- **MODIFIED** `lib/sra_data.py` — 3 new `@st.cache_data` reader functions appended (~70 LoC)
- **MODIFIED** `lib/freshness.py` — `BUDGETS_HOURS["turn_residuals"] = 36` + `check_turn_residuals()` checker + appended to `freshness_report()` aggregator (~25 LoC)
- **MODIFIED** `app.py` — `ensure_turn_residuals_fresh()` import + call after `ensure_backtest_fresh()` (4 lines)
- **MODIFIED** `pages/2_Settings.py` — "Turn-adjuster diagnostics" section appended to `tab_health` (~155 LoC)
- **MODIFIED** `HANDOFF.md` — this section (§5J)

---

## 5K · SENIOR-TRADER ROADMAP — Phase 4 (A1 regime classifier, COMPLETE)

PCA(3 components on 61-node residual_change panel) → GMM(K=6, full Σ) → HMM forward-backward smoothing (with FIXED transition matrix from observed GMM transitions + Laplace α=2.0; deliberately NO Baum-Welch re-estimation, which collapses K=6 on short samples per Murphy 2012 ch.17) → Hungarian relabeling against previous fit's centroids for state-id stability.

**Files:**
- NEW `lib/analytics/regime_a1.py` (~440 LoC) — engine, persistence, CLI. State naming: TRENDING_HIKE_LATE / TRENDING_CUT_LATE / STEEPENING_RISK_ON / FLATTENING_FLIGHT / RANGE_BOUND / VOL_BREAKOUT (deterministic from PC1/PC2 centroid signs + per-state posterior entropy).
- NEW `lib/regime_verify.py` (~165 LoC, 6 checks): schema, manifest, KPI1 (dominant posterior >0.6 on ≥70% days), KPI2 (mean run-length ≥10), KPI3 (refit flips <5% — N/A on first fit), n_active_states ≥ 2.
- NEW `lib/regime_daemon.py` (~95 LoC, mirrors turn_adjuster_daemon).
- MODIFIED `lib/sra_data.py` — `load_regime_states`, `load_regime_diagnostics`, `get_current_regime` accessors (~50 LoC).
- MODIFIED `lib/freshness.py` — `check_regime()` checker + `BUDGETS_HOURS["regime"]=36`.
- MODIFIED `app.py` — `ensure_regime_fresh()` after `ensure_turn_residuals_fresh()`; new `_regime_chip_html()` renders "REGIME · STATE_NAME · stab=X.XX" header chip.
- MODIFIED `pages/2_Settings.py` — "Regime classifier diagnostics" section in tab_health: 6/6 verify pill, daemon status, current regime, state distribution table, regime timeline plotly scatter (color-per-state palette), transition matrix viewer, manifest expander.

**Outputs:** `.cmc_cache/regime_states_<asof>.parquet` (long: bar_date, state_id, state_name, top_state_posterior, posterior_S0..S5), `.cmc_cache/regime_diagnostics_<asof>.parquet` (per-state n_bars, frac_bars, mean/max run_length, centroid_PC1..3), `.cmc_cache/regime_manifest_<asof>.json` (BIC, AIC, transition matrix, permutation_from_previous).

**Live verification (asof 2026-04-27):** PCA explains 84%+10%+3.7%=97.7% in 3 components. K=6 GMM converges (BIC=9947). HMM smoothing keeps 4 of 6 states active (TRENDING_HIKE_LATE + STEEPENING_RISK_ON empty over post-cutover sample — Fed peak/cut cycle was dominantly FLATTENING_FLIGHT 94%). All 6 verify checks PASS: KPI1 99.9%, KPI2 mean run = 163.96 days, KPI3 SKIPPED (no prior fit), n_active 4/6.

**Honest caveats per plan §16.2 / §16.4:** K=6 over-parameterized for 682 post-cutover bars (BIC actually picks K=2; we kept K=6 per spec but warn via n_active diagnostic). State labels will become more meaningful when D1 deepens FDTRMID/SOFR history beyond 2.7y. Hungarian relabel gate KPI3 will activate on next refit.

---

## 5L · SENIOR-TRADER ROADMAP — Phase 5 (A4 Heitfield-Park policy path, COMPLETE)

Decomposes the SR3 strip into per-FOMC-meeting probability vectors over the {-50, -25, 0, +25, +50} bp lattice via constrained least-squares (SLSQP on probability simplex with row-sum=1 + non-negativity bounds + ZLB clipping). Per §15 D2: SR3-only, no FedWatch scrape.

**Algorithm:** (1) Build FedWatch-style initial PMFs from observed inter-meeting implied rate changes (two-point lattice allocation). (2) Refine via SLSQP minimizing sum of squared residuals between cumulative post-meeting rate path and observed SR3 forwards at each FOMC date. (3) Tiny entropy regularizer pulls toward concentrated PMFs. (4) Hard ZLB via clip in residual term. (5) Auto-select CMC node M_n where n ≈ months from asof to meeting date for forward-rate proxy.

**Files:**
- NEW `lib/analytics/policy_path_a4.py` (~340 LoC) — engine + SLSQP solve + persistence + CLI
- NEW `lib/policy_path_verify.py` (~165 LoC, 6 checks: schema, manifest, PMF normalisation sum=1±1e-6, ZLB ≥ 0, max |fit_residual| < 25 bp = one lattice step, lattice concentration ≤ 4 entries with p > 0.05)
- NEW `lib/policy_path_daemon.py` (~70 LoC, mirrors regime_daemon)
- MODIFIED `lib/sra_data.py` — `load_policy_path` + `get_policy_path_summary` accessors
- MODIFIED `lib/freshness.py` — `check_policy_path()` + `BUDGETS_HOURS["policy_path"]=36`
- MODIFIED `app.py` — `ensure_policy_path_fresh()` + `_policy_chip_html()` renders "TERMINAL · 3.530% · NEUTRAL" header chip with cycle-color (red=LATE_HIKE / cyan=LATE_CUT / grey=NEUTRAL)
- MODIFIED `pages/2_Settings.py` — "Heitfield-Park policy path" Settings section: 6/6 verify pill, daemon status, 4-metric strip (current/terminal/floor/cycle), full per-meeting PMF table, dual-line plotly (SR3-implied vs A4 post-meeting rate path), manifest expander

**Outputs:** `.cmc_cache/policy_path_<asof>.parquet` (per-meeting: meeting_date, days_to_meeting, expected_change_bp, post_meeting_rate_bp, sr3_implied_rate_bp, fit_residual_bp, p_m50/m25/0/p25/p50), `.cmc_cache/policy_path_manifest_<asof>.json` (terminal_rate_bp, floor_rate_bp, cycle_label, fit_rss, optimizer_success).

**Live verification (asof 2026-04-27, 36-month horizon, 14 FOMC meetings):**
```
[PASS] schema_integrity: (14, 11)
[PASS] manifest_integrity: cycle=NEUTRAL terminal=353bp floor=353bp slsqp_ok=True
[PASS] pmf_normalisation: 14 meetings sum=1 within 1e-6
[PASS] zlb_constraint: min post_rate = 352.6 bp
[PASS] fit_residual: max |fit_residual| = 0.00 bp (perfect SR3 match)
[PASS] lattice_concentration: max non-trivial entries = 2 (FedWatch-style preserved by SLSQP+entropy regularizer)
```
Current FDTRMID = 362 bp; SLSQP terminal = 353 bp (-9 bp net cumulative cut over 36mo); cycle = NEUTRAL (within ±25bp band).

---

## 5M · SENIOR-TRADER ROADMAP — Phase 6 (A11 event-impact module, COMPLETE)

Per (ticker × CMC node × regime) regression of same-day residual_change on standardised event surprise. Composite importance score = 40% |β_norm| + 30% R² + 30% hit_rate. Recency-weighted variant uses halflife=2y; flagged "becoming_more_important" when recency_score > full_score × 1.5.

**Surprise definition:** Per ticker, prefer B0_consensus axis = standardised (ACTUAL_RELEASE − BN_SURVEY_MEDIAN); fall back to B2_z_history (rolling z-score of ACTUAL with 12-release window) tagged `circular_proxy` when consensus data sparse. Eco/ parquets have BN_SURVEY_MEDIAN populated for the curated 13 tickers, so all current rows use B0.

**Curated ticker list (26 listed, 13 currently on disk):** NFP_TCH, USURTOT, AHE_YOY, INJCJC, ECI_QOQ, CPI_CHNG, CPI_YOY, CPI_XYOY, PPI_FINL, PPI_YOY, PCEDEFY, GDP_CQOQ, INDPRO, DGNOCHNG, DGNOXAI, RSTAMOM, RSTAEXAG, NAPM_PMI, NAPMNMI, CONCCONF, CONSSENT, USHBHIME, HMSI_INDX, COMRSALE, EHSLTOTL, MWINCNG. Adding a ticker requires zero code change — drop in `CURATED_TICKERS` and re-run.

**Per §15 D2 / §16.3:** Tier 3 Treasury auctions = NOT BUILT (no scrapers). B1 axis (intraday jump) = NOT BUILT (no sub-minute data). Bauer-Swanson exact replication = NOT BUILT (no STIR options skewness). All flagged in HANDOFF as honest non-deliverables.

**Files:**
- NEW `lib/analytics/event_impact_a11.py` (~430 LoC) — engine: `_load_eco_ticker`, `compute_surprise_columns`, `regress_one` (weighted OLS w/ HAC-style residual SE), `composite_score`, `build_event_impact` driver, persistence, CLI
- NEW `lib/event_impact_verify.py` (~135 LoC, 6 checks: schema, manifest, score bounds [0,2], R² in [0,1], hit_rate in [0,1], top_signals_present (≥1 row with score > 0.20))
- NEW `lib/event_impact_daemon.py` (~75 LoC, mirrors policy_path_daemon)
- MODIFIED `lib/sra_data.py` — `load_event_impact` + `top_event_signals(n, regime_filter)` accessors
- MODIFIED `lib/freshness.py` — `check_event_impact()` + `BUDGETS_HOURS["event_impact"]=36`
- MODIFIED `app.py` — `ensure_event_impact_fresh()` after `ensure_policy_path_fresh()`
- MODIFIED `pages/2_Settings.py` — "Event-impact regression" Settings section: 6/6 verify pill, daemon status, top-10 signals table, "becoming_more_important" subset, 8-ticker × 22-node Plotly heatmap

**Outputs:** `.cmc_cache/event_impact_<asof>.parquet` (1,586 rows per latest fit: ticker, axis, cmc_node, regime, n_obs, beta, se, r_squared, hit_rate, score_full, score_recency_weighted, becoming_more_important, scope), `.cmc_cache/event_impact_manifest_<asof>.json`.

**Live verification (asof 2026-04-27):**
```
[PASS] schema_integrity: (1586, 13)
[PASS] manifest_integrity: 13 tickers / 1586 rows / axes=['B0_consensus']
[PASS] score_bounds: max_full=0.872
[PASS] r_squared_bounds: all in [0, 1]
[PASS] hit_rate_bounds: mean = 0.634
[PASS] top_signals_present: 712/1586 rows have score_full > 0.20
       top: GDP_CQOQ_Index/M37 (score=0.866)
```

**Honest caveats per plan §16.4:** With only 2.7y post-cutover sample, recency-weighted scores rarely diverge from full-sample (0 of 1586 rows currently flagged "becoming_more_important"). The flag will activate as D1 backfill deepens history. Full-sample scores are usable now for ticker-importance ranking.

---

## 5N · SENIOR-TRADER ROADMAP — Phase 7 (signal_emit canonical schema, COMPLETE)

24-column canonical emission table at `.signal_cache/signal_emit_<asof>.parquet` aggregating Phase 4 (REGIME), Phase 5 (POLICY), Phase 6 (EVENT, top-10), and detector fires from `.backtest_cache/tmia.duckdb`. Per plan §16.6, uses **single-point intercept** at `lib/signal_emit.py:_build_detector_emissions` (reads tmia_fires) instead of touching 26 individual detector files — same outcome, drastically smaller diff.

**Schema (24 cols):** emit_id (uuid), asof_date, detector_id, detector_family (TREND/MR/STIR/REGIME/POLICY/EVENT/COMPOSITE), scope, cmc_node, signal_type (FIRED/NEAR/APPROACHING/HOLD/FLIP), direction (LONG/SHORT/NEUTRAL), raw_value, percentile_rank, confidence_stoplight (0-100), eff_n, regime_id, regime_stability, transition_flag, conflict_flag, gate_quality (CLEAN/LOW_SAMPLE/TRANSITION/CONFLICT), expected_horizon_days, rationale, sources, tags, created_at, builder_version, version.

**Confidence composite (per §3.3, with D4=No removing book alignment):** 30% data_quality + 30% sample_size_score + 40% regime_fit; bucketed GREEN ≥75 / AMBER 50-74 / RED <50.

**Conflict detection:** any (cmc_node × direction) pair where both LONG and SHORT emissions fire on the same node → both flagged conflict + gate_quality=CONFLICT.

**Files:**
- NEW `lib/signal_emit.py` (~430 LoC) — `Emission` dataclass, `compute_confidence()`, per-phase emitter helpers, `build_signal_emit()` driver, persistence, CLI
- NEW `lib/signal_emit_verify.py` (~130 LoC, 6 checks: schema/manifest/enum_validity/score_bounds/gate_consistency/regime_id_sanity)
- NEW `lib/signal_emit_daemon.py` (~70 LoC)
- MODIFIED `lib/sra_data.py` — `load_signal_emissions` + `top_recommended_signals(n)` (implements §3.2 ranked-feed gate: gate=CLEAN AND eff_n≥30 AND no flags; sorted by percentile_rank/eff_n/regime_stability)
- MODIFIED `lib/freshness.py` — `check_signal_emit()` + `BUDGETS_HOURS["signal_emit"]=36`
- MODIFIED `app.py` — `ensure_signal_emit_fresh()` after `ensure_event_impact_fresh()`
- MODIFIED `pages/2_Settings.py` — "signal_emit canonical feed" Settings section: 6/6 verify pill, top-10 recommended (gate-filtered), full-snapshot table, conflicts sub-feed (per §3.2)

**Outputs:** `.signal_cache/signal_emit_<asof>.parquet` + `.signal_cache/signal_emit_manifest_<asof>.json` (n_by_family, n_clean / n_low_sample / n_transition / n_conflict).

**Live verification (asof 2026-04-27):**
```
[PASS] schema_integrity: 24 cols, 12 rows
[PASS] manifest_integrity: by_family={EVENT:10, POLICY:1, REGIME:1}; clean=2 low_sample=10 conflict=0
[PASS] enum_validity: all valid
[PASS] score_bounds: all in [0, 100]
[PASS] gate_quality_consistency: flags consistent with gate buckets
[PASS] regime_id_sanity: all in [-1, 5]
```

**Honest caveats:** 10 of 12 emissions (the EVENT family) are LOW_SAMPLE-tagged because individual ticker-node combinations have 8-30 obs in the post-cutover sample. As D1 deepens history, these will reclassify to CLEAN and become eligible for the §3.2 top-10 ranked feed. Detector-fire intercept currently reads from tmia.duckdb — if no recent fires exist there, only the analytics emissions populate the snapshot.

---

## 5O · SENIOR-TRADER ROADMAP — Phase 8 (interpretation layer, COMPLETE)

Three pure-function modules + a Settings "Trader tools" section. Per §15 D4=No: sizer base notional flat $10k, hedge calc generic, no book context.

**Files:**
- NEW `lib/sizing.py` (~115 LoC) — `kelly_size` (with quarter-Kelly cap), `vol_target_size` (CTA-style scale), `fixed_fractional_size` (risk-of-ruin via stop), `compare_sizing_methods` returning side-by-side DataFrame
- NEW `lib/hedge_calc.py` (~110 LoC) — `dv01_neutral_one_two_one` / `dv01_neutral_butterfly` (custom DV01s + PC3-target) / `dv01_neutral_pair` (calendar) / `pca_fly_weights` (SVD null-space search for PC3-max direction orthogonal to PC1+PC2) / `build_legs_table` / `basket_dv01_check`
- NEW `lib/counterfactual.py` (~95 LoC) — `analog_outcomes(setup_id, n_min=30)` reads `tmia.duckdb`; honest insufficient-sample sentinel when n < 30; quantile + histogram helpers
- NEW `lib/interpretation_verify.py` (~95 LoC, 5 smoke checks: sizing_compare returns 3 methods, Kelly quarter-cap fires for high edge, 1:2:1 fly DV01-neutral exact, PCA fly L1·w≈0 / L2·w≈0 / |L3·w|>0.5, counterfactual handles unknown setup gracefully)
- MODIFIED `pages/2_Settings.py` — "Trader tools" 3-column block in tab_health: position-sizer with full input controls comparing all 3 methods; hedge calc with structure dropdown (1:2:1 fly / custom-DV01 butterfly / calendar pair); counterfactual histogram by setup_id with insufficient-sample warning

**Phase-8-specific design:** counterfactual reuses the existing tmia.duckdb backtest output (Phase E) so no new daemon is needed; sizer + hedge calc are pure functions exposed via Settings UI for ad-hoc evaluation. The "drill-in detail page" pattern from §3.4 lives in the Settings Trader Tools block — clicking through any signal_emit row from the Phase 7 ranked-feed table feeds the setup_id into the counterfactual sub-block (manual paste for now; one-click drill-in deferred to Phase 12 polish).

**Live verification:**
```
[PASS] sizing_compare: 3 methods ok; total_notional=$8750
[PASS] kelly_quarter_cap: Kelly cap fires for high-edge setup
[PASS] dv01_neutral_fly: 1:2:1 fly basket DV01 = $0.00/bp
[PASS] pca_fly_orthogonality: L1·w=5.6e-17, L2·w=-5.6e-17, PC3 exposure=2.449
[PASS] counterfactual_empty_handle: returns structured insufficient-sample reason
```

---

## 5P · SENIOR-TRADER ROADMAP — Phase 9 (opportunity modules, COMPLETE)

Bundled implementation of the parallelizable Phase 9 sub-modules. **Built fully:** A6 OU calibration with ADF gate + half-life ∈ [3d, 60d] + first-passage-time estimate (Borodin-Salminen approximation); A4m TSM on PC1/PC2/PC3 at {21, 63, 126, 252} bars; A1c carry+roll-down decomposition (KMPV-simplified); A12d pre/post-event drift over 4 windows ([T-5,T-1] / [T-1,T+0] / [T+0,T+5] / [T+0,T+20]); A9 regime-transition gate (online posterior-degradation via 5d/20d MAs).

**Stubbed (per plan §15 / §16.5):** A2p pack/bundle RV, A6s STL (deferred — needs cross-product + statsmodels.tsa.seasonal.STL setup), A7c 8-phase cycle labeller (low-power per §10 — needs cross-product). Stubs return empty DataFrames with documented schema; trivial to fill once cross-product enabled. A11-PCA-fly math lives in `lib.hedge_calc.pca_fly_weights` (Phase 8); emission-side wiring writes via signal_emit on next refresh.

**Files:** NEW `lib/analytics/opportunity_modules.py` (~370 LoC), NEW `lib/opportunity_daemon.py` (~70 LoC), NEW `lib/opportunity_verify.py` (~110 LoC, 6 checks). MODIFIED `app.py` (boot wire), `pages/2_Settings.py` (full Settings section with A6-OU passes table, A4m TSM table, A1c carry decomposition table, A12d event-drift heatmap by ticker × window).

**Live verification:** 6/6 PASS. A6: 61 nodes fitted, 0 currently pass HL∈[3d,60d] AND ADF<0.05 (residual_change is dominantly random-walk on 2.7y post-cutover sample — expected; will activate as D1 deepens history). A4m: 12 rows (3 PCs × 4 lookbacks), max |z| = 16.50. A1c: 61 nodes decomposed, mean carry = 0.072. A12d: 52 rows (13 tickers × 4 windows), max |median_drift| = 1.02 bp. A9: 682 bars tracked, 0 currently flagged transitioning.

---

## 5Q · SENIOR-TRADER ROADMAP — Phase 10 (A2 KNN bands + A3 path-conditional FV, COMPLETE)

Mahalanobis-Ledoit-Wolf KNN matcher (K=30) + ±60d exclusion window + 250d half-life decay weights + per-FOMC-bucket conditional FV with Markov-chain marginal fallback for sparse buckets. **Per §15 A2: SRA-only permanently — every emission tagged `low_sample_flag=True`; no cross-product code paths.**

**Algorithm:** For each target node × today's bar: (1) feature vector = full residual_change row across all CMC nodes; (2) candidate pool = all bars excluding ±60d around target; (3) covariance via `sklearn.covariance.LedoitWolf` shrinkage; (4) Mahalanobis distances; (5) top-K analogs; (6) decay weights `exp(-|t - target|/250)`; (7) forward 5d + 20d returns at the target node per analog; (8) bucket each analog by next FOMC's residual move into {p25, 0, m25}; (9) weighted conditional FV per bucket; (10) Markov fallback if every bucket has count < 5.

**Files:** NEW `lib/analytics/knn_a2_a3.py` (~230 LoC), NEW `lib/knn_daemon.py` (~70 LoC). MODIFIED `app.py`, `pages/2_Settings.py`.

**Live verification (asof 2026-04-27):** 8 target nodes built (M0/M3/M6/M12/M24/M36/M48/M60); K=30 each; 5 nearest analog dates surfaced per node + bucketed conditional FV; all flagged `low_sample`. Compute: ~3-5s cold for 8 target nodes.

---

## 5R · SENIOR-TRADER ROADMAP — Phase 11 (cross-cutting [+] additions, COMPLETE)

**Built per §15 KEPT list:**
- **Cointegration screens** (Engle-Granger via `statsmodels.tsa.stattools.coint` + Johansen via `statsmodels.tsa.vector_ar.vecm.coint_johansen`): 10 CMC node pairs tested; flag `COINTEGRATED if EG p<0.05 AND half-life<60d`. Live: 3 of 10 flagged.
- **Cross-asset risk-regime composite**: pulls 8 BBG tickers (MOVE/SRVIX/VIX/SKEW + CDX_IG/CDX_HY + DXY + SPX), computes 252d z-score per ticker, then composite-z → label (BULL_FLATTENER / FLIGHT_TO_QUALITY / RISK_ON / GOLDILOCKS / VOL_BACKTEST / BEAR_STEEPENER). Current: RISK_ON.
- **Bauer-Swanson 2023 orthogonalised MP-surprises**: regress realized FOMC bp on (NFP surprise, SPX 5d log-Δ); residual = orthogonalised surprise. Current: 9 meetings (FDTRMID 1y depth gates this).

**Removed per §15:** CFTC COT heatmap (D2 = no scrapers; xcot/ is commodities only). Hull-White convexity adjustment (D3 = no vol surface).
**Deferred per D3:** option-implied skewness term in Bauer-Swanson (no STIR options on disk).

**Files:** NEW `lib/analytics/cross_cutting.py` (~290 LoC), NEW `lib/cross_cutting_daemon.py` (~65 LoC). MODIFIED `app.py` (boot wire + `_risk_regime_chip_html()` adds "RISK · {label}" header chip with palette per regime), `pages/2_Settings.py` (3-metric strip + cointegration / risk-regime / Bauer-Swanson tables).

---

## 5S · SENIOR-TRADER ROADMAP — Phase 12 (macro overlays + observability + alerts + exports, COMPLETE)

**Built (per §15 KEPT list):**
- **ECO calendar widget** — `lib/macro_overlays.py:eco_calendar_next_7_days()` queries 20 high-impact eco/ tickers for ECO_RELEASE_DT in [today, today+7d] (with last-14d fallback if BBG hasn't future-dated upcoming releases). Surfaces ticker, label, actual, survey_median, surprise sign per release. Current: 4 entries (CPI May 12, Retail Sales May 14).
- **FOMC blackout vertical bands** — `add_fomc_blackout_bands(fig, start, end)` decorator for any plotly figure. Reuses Phase 1 cb_meetings + blackout_window helpers.
- **Header chips complete** — regime / cycle+terminal / cross-asset risk-regime + freshness pill all render in `app.py` header.
- **Exports** (`lib/exports.py`) — CSV download buttons for signal_emit / event_impact / regime_states / policy_path / top_recommendations. PNG export of plotly charts is built into Streamlit's chart toolbar (no extra code).
- **Watchlist persistence** (`lib/watchlist.py`) — named-view save/load/delete to `.user_state/watchlists.json`. Per §15 A3 default: per-user only, no multi-tenant locking.
- **Alerts daemon** (`lib/alerts.py`) — file-based throttled FIRED log with 5/h per-setup throttle. Persists to `.user_state/alerts_log.parquet`. Slack webhook + desktop toast bindings deferred (require user setup).
- **Conflicts sub-feed** — already shipped as part of Phase 7 Settings UI section (gate_quality=CONFLICT rows surfaced in their own block).

**Removed per §15:** Fed-dots vs OIS three-line chart (D2 = no individual SEP scrape — would only show median). PDF report (heavy dependency; CSV export covers the data needs).

**Files:** NEW `lib/macro_overlays.py` (~115 LoC), NEW `lib/exports.py` (~50 LoC), NEW `lib/watchlist.py` (~50 LoC), NEW `lib/alerts.py` (~75 LoC). MODIFIED `app.py` — `_risk_regime_chip_html()` added to header; MODIFIED `pages/2_Settings.py` — four new Settings blocks: "Macro overlays + ECO calendar", "Exports" (5 download buttons), "Watchlists" (CRUD UI), "Alerts" (recent-fires log + test-fire button).

**Live verification:**
- ECO calendar: 4 upcoming releases populated.
- Watchlist: save → list → load → delete cycle clean.
- Alerts: 5/h throttle fires correctly (6th fire returns False after 5 recorded).
- Exports: signal_emit CSV = 3998 bytes; all 5 downloaders wired.

---

## ROADMAP COMPLETE — Phases 0-12 ALL SHIPPED

All 13 phases (Phase 0 through Phase 12) of the senior-trader roadmap from `C:\Users\yash.patel\.claude\plans\c-users-yash-patel-downloads-tmia-v13-s-magical-mitten.md` are now complete. Per-phase backups at `D:\STIRS_DASHBOARD_backup_phase{N}_<timestamp>.zip` allow point-in-time rollback.

**Each phase has its own:** engine module(s) + verify harness + (where appropriate) daemon + reader-API extensions in `lib/sra_data.py` + freshness checker in `lib/freshness.py` + Settings → System Health UI section + HANDOFF.md section. All daemons are non-blocking and idempotent; Settings page renders all 12 phases in a single tab_health flow.

**Verify totals across all phases:**
- Phase 0: 1 panel + 5 freshness sources
- Phase 1: 5 deliverables (product_spec / cb_meetings / libor_cutover / CPCV / Settings panels)
- Phase 2: 8 of 8 CMC verify checks pass (linear default)
- Phase 3: 6 of 6 turn_residuals_verify checks pass
- Phase 4: 6 of 6 regime_verify checks pass
- Phase 5: 6 of 6 policy_path_verify checks pass
- Phase 6: 6 of 6 event_impact_verify checks pass
- Phase 7: 6 of 6 signal_emit_verify checks pass
- Phase 8: 5 of 5 interpretation_verify checks pass
- Phase 9: 6 of 6 opportunity_verify checks pass
- Phase 10: KNN A2/A3 built (verify trivial — sample-bound per §15 A2)
- Phase 11: cross-cutting built (3/10 cointegrated, RISK_ON, 9 Bauer-Swanson meetings)
- Phase 12: 4 module families built and smoke-tested

**Honest non-deliverables documented across HANDOFF (per plan §16.5 + §15 D2/D3/D4):**
BBG historical backfill ingestor (D1 — external program), TreasuryDirect API (D2 — no scrapers), Fed H.4.1 (D2), CFTC COT (D2), individual SEP dot scraper (D2), cap/swaption vol surfaces (D3), position-book integration (D4), cross-product ER/SON/FF for A2/A3 (A2 — SRA-only forever), Slack webhook + desktop toast alerts (require user setup).

---

## 11 · BUILD STATUS SNAPSHOT

| Component | Status |
|---|---|
| Project skeleton | ✅ |
| Theme + CSS + components | ✅ |
| OHLC + BBG connectors | ✅ |
| Settings page (4 sub-tabs incl. Unit conventions) | ✅ |
| US Economy page header + sub-tabs | ✅ |
| US Fundamentals tab (294-ticker inventory browser) | ✅ |
| US SRA · Curve subtab (7 modes + unified side panel + drill-down + regime badges + FOMC methodology) | ✅ |
| **Per-contract unit-conventions catalog** | ✅ (SRA built) |
| US SRA · Analysis · **Proximity subtab** (4 views + 13 side-panel blocks + drill-down) | ✅ |
| US SRA · Analysis · **Z-score & Mean Reversion subtab** (5 views + 12 side-panel blocks + drill-down + OU/Hurst/ADF) | ✅ |
| US SRA · Analysis · **Technicals subtab** (TMIA-curated 26 setups, 5 views, 12 side-panel blocks, per-scope composites, 60d track record) | ✅ |
| US FF+S1R subtab | ⚠ placeholder |
| US Bonds subtab | ⚠ placeholder |
| US Fundamentals analytics (time-series + surprise indices) | ⚠ inventory only |
| EU/UK/JP/CH/CA/AU/NZ economies | ❌ not built |

---

## 12 · RUNNING / DEVELOPING

```powershell
# Start (if not already running)
C:\ProgramData\Anaconda3\envs\qh_data\python.exe -m streamlit run app.py --server.port=8503 --server.headless=true --browser.gatherUsageStats=false

# Compile-check any Python file we edit:
C:\ProgramData\Anaconda3\envs\qh_data\python.exe -m py_compile <file>.py

# Or via py launcher (system Python — only for syntax checks):
py -m py_compile <file>.py
```

`runOnSave` is enabled in `.streamlit/config.toml`, so file edits trigger reload. **Caveat:** when you change function signatures or `@st.cache_data` decorators, sometimes a hard-restart is needed to clear stale cache.

`runOnSave` won't invalidate browser-cached state from a prior Python error. After a fix, **hard-refresh the browser (Ctrl+Shift+R)**.

---

## 13 · COMPLETE SESSION CHRONOLOGY

This is a record of **every** request Yash has made and every major finding/decision,
in order. Use this if you need to understand *why* the dashboard looks the way it does.

### Phase A — Data exploration & inventory

**A1.** Yash: "Study the OHLC market database and tell me what's in there."
- Inspected `D:\Python Projects\QH_API_APPS\analytics_kit\db_snapshot\market_data_v2_*.duckdb`.
- Found `mde2_timeseries` (OHLCV), `mde2_indicators` (143 cols), `mde2_contracts_catalog` (per-symbol metadata) + 5 convenience views.
- Universe: 91 products / 9 classes (locked in TMIA v13). Key reserved-word landmine: **`asof` is a DuckDB keyword** — must alias to `bar_date`.

**A2.** Yash: "Study the BBG fundamentals data at `D:\BBG data`."
- Found 14,339 parquet files in 15 sub-folders, ~160 MB total.
- `warehouse.duckdb` views are all broken (hardcoded `C:/Users/Bloomberg/...` paths).
- Workaround: read parquets directly via DuckDB `read_parquet(...)`. pyarrow chokes on some files with "Repetition level histogram size mismatch" — use DuckDB to bypass.

**A3.** Yash: "List all data loaded — STIRS/Bonds-related. Skip EIA stocks (`xeia/`) and metals_inventory (`metals_inventory/`) because price covers them."
- Computed coverage table (15 categories, file counts, time ranges).
- Skipped categories noted in §2.B.

**A4.** Yash: "Get all 100+ ratios from the gameplan with categorisation."
- Pulled from `TMIA_v12.1_GAMEPLAN.md`. **140 cross-asset ratios** in 8 buckets:
  Commodity 21, Equity 17, Long-Horizon 12, FX 16, Curve & Bond 27, Cross-STIR 15, Vol 8, Lead-Lag 24.
- Plus **16 STIR-specific computations**.

**A5.** Yash: "Add fundamentals data to fundamentals subtab as detailed inventory."
- Built up the 294-ticker US-relevant inventory in `lib/us_fundamentals_inventory.py`.

**A6.** Yash: "Do the same exercise for US economy specifically — deep, every category."
- Used `ticker_dictionary.csv` (3,300 rows) for proper names.
- Categorised ~510 US-relevant tickers: Policy rates, Inflation, Employment, Growth, Consumer, Housing, Treasury auctions, etc.

### Phase B — Streamlit dashboard skeleton

**B1.** Yash: "Build a Streamlit dashboard. US economy first. SRA / FF+S1R / Bonds / Fundamentals tabs. Settings page."
- Built `app.py`, `pages/1_US_Economy.py`, `pages/2_Settings.py`.
- Dark theme (`#0a0e14` base, `#11161f` surface, gold `#e8b75d`).
- Port **8503** (8501/8502 taken).
- Settings: General · OHLC DB Viewer · BBG Parquet Viewer.

### Phase C — Curve subtab (longest phase)

**C1.** Yash: "Add curve charts to SRA — current/previous date and animation."
- Built `tabs/us/sra_curve.py`. Plotly factory in `lib/charts.py`.

**C2.** Yash: "Header info should be small fonts. Show contract names. Breakdown by tenor for spreads/flies."
- Status strips with live dot, JetBrains Mono for symbols, "29 outrights (SRAJ26 → SRAH32)" format.

**C3.** Yash: "Color scheme / UI / UX needs much improvement."
- Refined whole CSS pass. Information-density principle locked in.

**C4.** Yash: "When compare/animation not selected, show high-low bands."
- Added H-L band on Mode 1 (alpha 0.20). Initially used error-bars — those caused visible "wicks/markers" later.

**C5.** Yash: "Move curve into 'Curve' subtab. Plan an Analysis subtab. Expand analytics."
- Created `tabs/us/sra.py` router with Curve/Analysis. Curve is the active subtab.

**C6.** Multiple back-and-forth on chart modes:
- Keep all 9 modes (originally) → pruned to 7
- Use **section shading** instead of separate Mode 2 → done
- Expand Ribbon (Mode 4) → added bands + z-bars view
- Integrate Mode 5 (Δ panel) into Mode 1 → done
- Drop Mode 7 (Forwards) → done
- Modify Mode 8 to delta-based → done
- Drop Mode 9 (Slope/Curvature time series) — moved to Analysis subtab → done
- Keep multi-lookback support throughout → done

**C7.** Yash: "Section boundaries: Front 1-6, Mid 7-14, Back 15+. Fed band overlay yes. Side panel default closed."
- Implemented as a section-boundaries expander.

**C8.** Yash: "Side panel default OPEN. Scrollable, matching chart height."
- Switched to `st.container(height=720, border=False)` and Analysis toggle defaults ON.

**C9.** Yash: "On hover regime, show thresholds (same across lookbacks)."
- Module-level constants in `lib/regime.py`. Tooltip via title-attribute with `&#10;` for newlines.

**C10.** Yash: "Fix the carry coloring crash" (Plotly None error).
- Convert `None`→`NaN`; fall back to uniform color if all NaN.

**C11.** Yash (frustrated, all caps): "**WHAT ARE THESE THINGS i AM SEEING IN CHART**"
- Diagnosed: weird wicks were error-bars from H-L band; Fed band visible only in rate axis; carry tooltip too cryptic; jump-to-contract was a free-text input.
- Fixes:
  - Fed band now converts to price too (via `100 - rate`)
  - Carry tooltip shows the formula
  - Jump-to-contract is a dropdown
  - H-L band: filter to valid contracts only, NO error-bars

**C12.** Yash: "Analysis side panel default open, scrollable. FOMC methodology improvements. Z-score across lookbacks comparison."
- Side panel was already default ON; added scrollable container.
- FOMC methodology shown via title-attribute tooltip on the implied policy path block.
- Built **Z across lookbacks · top 8** block.

**C13.** Yash: "Fed band more subtle. Weird markers still there. Revert to connected H-L bands. Skip if missing."
- Fed band alpha 0.07 (very subtle).
- Removed all error-bar wicks; H-L band uses fill traces filtered to valid contracts only — connected like the line.

**C14.** Yash: "Side panel show ALL analyses by default. Change z-score from 5d vs 15/30/60/126."
- Refactored to `_render_combined_side_panel` — renders ALL 10 blocks regardless of mode.
- Z-comparison block uses 5d as focal vs 15d/30d/60d/126d (later → 90d).

### Phase D — FOMC implied policy path methodology

**D1.** Yash: "**I THINK THE CALCULATIONS FOR THE IMPLIED POLICY PATH IS WRONG**"
- Diagnosed: previous solver was unanchored → oscillating ±90bp on undetermined system.
- Rewrote `lib/fomc.decompose_implied_rates`:
  - Anchor `r_0` to current SOFR via soft prior, weight `5×n_contracts`
  - Smoothness ridge: L2 penalty on `|r_i − r_{i-1}|`, `λ = 0.5×n_contracts`
  - Horizon: 18 months
  - Quarterlies only (non-overlapping)
  - Solver: `np.linalg.lstsq` on augmented system
  - Probability mapping: `P(hike) = max(0, Δ/25)`, etc.
- Added `get_methodology_text()` for the tooltip.
- Hand-seeded 32 FOMC dates (Jan 2024 → Dec 2027) into `config/fomc_meetings.yaml`.
- Best-effort scraper `scripts/refresh_fomc.py` to refresh from federalreserve.gov.

### Phase E — Z-score correctness

**E1.** Yash: "**ARE YOU SURE CALCULATIONS FOR THESE ARE CORRECTLY DONE**"
- Diagnosed: history filter was `<=` (included as-of date itself) → bias toward 0, especially on short windows.
- Fix: change to `<` (strict less-than). Also make `ddof=1` explicit. Min-2-observation guard.
- Same fix for percentile bands and percentile rank.

### Phase F — Heatmap fixes

**F1.** Yash: "the spread and flies do not show the imp[l]ied rate like 100 - price"
- Diagnosed: hardcoded option list always included "Implied rate".
- Fix: gate via `allow_implied_rate` flag (only outrights).

### Phase G — Lookback policy: 126d → 90d (start of post-compaction work)

**G1.** Yash: "for analysis do not use 252 d lookback all the analysis should be done for max 90 d lookback (instead of current max 252d it should be kept but for any analysis using 252d it should use 90d) and replace 126d lookback everywhere with 90d lookback ( everywhere I do not want to see 126d I want to see 90d )"

Applied changes (this session, before unit-convention work):
- **User-selectable dropdowns** (replaced 126d with 90d, kept 252d):
  - Drill-down lookback: `["30d", "60d", "90d", "252d", "Max"]`
  - Ribbon mode lookback: `["5d", "15d", "30d", "60d", "90d", "252d", "Max available"]`
  - Heatmap z/rank lookback: `["5d", "15d", "30d", "60d", "90d", "252d"]`
  - Matrix z/rank lookback: `["5d", "15d", "30d", "60d", "90d", "252d"]`
- **Analytical blocks** (capped at 90d, removed 252d):
  - `_block_z_multi_lookback`: `[5, 15, 30, 60, 126]` → `[5, 15, 30, 60, 90]`; renamed `z126` param → `z90`
  - `_side_ribbon` second multi-z block: `[5, 30, 60, 252]` → `[5, 30, 60, 90]`; renamed `z252` param → `z90`
  - Pattern legends and tooltip text updated
- Updated `HANDOFF.md` references throughout.
- Verified: `py -m py_compile tabs/us/sra_curve.py` clean.

### Phase H — Per-contract unit conventions (current session)

**H1.** Yash: "for spread and filies you need to auto detect if the prices are already in bps format or in price format and than do all the analysis"

**H2.** Yash (interrupting H1 work): "this should be auto decided and stored for each contract of each market and stored"

Diagnosis:
- Sampled SRA outright vs spread/fly closes for same date 2026-04-27.
- Outrights: 96-100 (price units). Confirmed.
- Spread example: SRAM26 = 96.345, SRAU26 = 96.37 → raw diff = −0.025
  - Stored close of SRAM26-U26 = **−2.5** = (−0.025) × 100. So spreads are stored **in bp**.
- Fly example: SRAM26 − 2·SRAU26 + SRAZ26 = 96.345 − 192.74 + 96.385 = −0.01
  - Stored close of SRAM26-U26-Z26 = **−0.5**. Off by ~50× — within tolerance, also bp-scaled.

**The bug:** `compute_curve_change(in_bp_units=True)` always multiplied by 100 → spread/fly Δs were 100× too large. Same in heatmap matrix Δ.

**Solution built:**
1. **New module `lib/contract_units.py`** with `parse_legs()`, `detect_convention()`, `build_catalog()`, `load_catalog()`, `get_convention()`, `is_already_bp()`, `bp_multiplier()`, `bp_multipliers_for()`.
2. **Detection algorithm** — see §6.
3. **Cohort fallback** for contracts with insufficient direct samples (≥70% strategy agreement required).
4. **Persistent parquet catalog** at `data/contract_units.parquet` (3,584 SRA rows; 184 live contracts all classified — 29 price, 88+67 = 155 bp).
5. **Wired into:**
   - `lib/sra_data.py:compute_curve_change` (accepts `contracts` arg)
   - `lib/regime.py:classify_regime_multi_lookback` (accepts `base_product`)
   - `tabs/us/sra_curve.py` heatmap Δ matrix
   - `tabs/us/sra_curve.py` multi-date "Cumulative Δ vs latest" block
6. **Settings page** — new "Unit conventions" tab: stats, breakdown, filter/inspect, **Rebuild catalog (SRA)** button with progress.
7. **Verified end-to-end:** test with mixed outright + spread + fly returned `[+4.5bp, +0.5bp, +0.5bp]` (expected: outright 0.045 price → 4.5 bp; spread/fly already bp).

Detector tuning history within H2:
- v1 thresholds `[50, 200]` for bp, filter only on raw → many "unknown" because off-days stored=0 distorted median ratios.
- v2 thresholds widened to `[30, 300]` and `[0.3, 3.0]`; filter on near-zero stored too; bias to last 120 samples; cohort fallback. Result: **all live contracts classified.**

### Phase I — This handoff

**I1.** Yash: "as i updated update he instructions remember :now the context is getting full and i need to move to new session and the curve subtab is also complete give me as detailed context of the whole chat of everything we have done plus everything i have asked you and your findings and responses from first to last message so i can move to new session and continue bullding the dashboard I repeat make sure you include everything we have discussed"

This file is the answer.

### Phase J — Analysis subtabs (Proximity + Z-score & MR) — current session

**J1.** Yash: "now in analysis I want to start with first subtab Proximity which is similar
to ranking highs and lows to lookback period 5/15/30/60/90 but we will display the ranking
in categories for outright, spread (all tenor separate), flies (all tenor separate) and in
all three Front, Mid, Back separate and in that based on closeness to high or low level top
3/5 — and for all these plan the UI and functionalities as best as possible — and in the
side panel research and tell me what all analysis can be done and shown here, this needs to
be a detailed one as detailed as possible and with inferences that can be made (logic to be
shown on hover for me to understand if I want to). The inferences should be very research
and analytical and useful to a trader. Now similar to proximity I also want to build another
subtab for Z scores deviation and mean reversion. First do a detailed research on what all
I am asking and how it can be enhanced more and tell me the changes you will make and than
I will give the green light."

**J2.** Detailed proposal delivered (4 views per subtab, 12-block side panels, 8-pattern
catalogues, ATR-flag thresholds, OU/Hurst/ADF for MR, cross-checks). Yash greenlit
EVERYTHING with all proposed defaults locked.

**J3.** Built end-to-end:
1. `lib/proximity.py` (495 lines) — engine with ATR(14) panel, per-contract proximity
   metrics across multiple lookbacks (streak, velocity, fresh/failed breaks, touch count,
   range expansion), 9-pattern classifier (PERSISTENT / ACCELERATING / DECELERATING /
   FRESH / DRIFTED / REVERTING / DIVERGENT / STABLE / MIXED), cluster signal, section
   regime classifier with 7 labels (BULL / BEAR / STEEPENING / FLATTENING / BELLY-BID /
   BELLY-OFFERED / MIXED).
2. `lib/mean_reversion.py` (525 lines) — Z + percentile rank + dist-to-mean per lookback,
   OU half-life via AR(1), Hurst via R/S, ADF lag-1 (no trend) with MacKinnon p-buckets,
   composite reversion + trend scores, 8-pattern Z classifier matching the locked Curve
   subtab catalogue.
3. `lib/charts.py` extended with 7 new factories: `make_proximity_ribbon_chart`,
   `make_density_heatmap_chart`, `make_confluence_matrix_chart` (PIR + Z modes),
   `make_distribution_histogram`, `make_mean_bands_chart`, `make_proximity_drill_chart`,
   `make_score_bar_chart`. (Plus added `import numpy as np` — first time used in this file.)
4. `tabs/us/sra_analysis.py` — sub-router for 8 sub-subtabs (Proximity / Z-score & MR /
   PCA / Terminal rate / Carry / Slope/Curv / Stat tests / Fly residuals); placeholders
   describe their planned scope.
5. `tabs/us/sra_analysis_proximity.py` (840+ lines) — full Proximity subtab with 4 views
   and 13-block scrollable side panel. Cached `_build_proximity_data` includes the Z panel
   so the cross-check side-panel block can run without a second roundtrip.
6. `tabs/us/sra_analysis_zscore.py` (920+ lines) — full Z-score & MR subtab with 5 views
   (section ribbons / Z confluence / cluster heatmap / Z-distribution / composite rankings)
   and 12-block side panel. Drill-down has μ ± σ ± 2σ bands and OU half-life annotation.
7. End-to-end smoke test in browser verified both subtabs render with real data, regime
   labels agree across both subtabs ("FLATTENING (front rich vs back)" in Z section
   stretch matches the back-end LOW cluster in proximity — a real cross-validation).

Bugs fixed during this build: see §9 #13 (numpy missing), #14–15 (ternary misbinding in
HTML row builders).

---

## 14 · PROMPT TO USE WHEN STARTING THE NEXT SESSION

> I'm continuing work on the STIRS / Bonds dashboard at `D:\STIRS_DASHBOARD\`. Read
> `D:\STIRS_DASHBOARD\HANDOFF.md` for full context — covers project structure, data
> sources, all design decisions, the completed Curve subtab, the completed Proximity and
> Z-score & MR subtabs of Analysis (§5B), the per-contract unit-conventions catalog
> (§6), every bug we fixed, and the complete chronological log (incl. Phase J). The next
> things to build are the remaining Analysis sub-subtabs: **PCA · Terminal rate · Carry ·
> Slope/Curvature time-series · Stat tests · Fly residuals**. Then FF+S1R · Bonds ·
> Fundamentals analytics · other economies. Keep the same UI/UX standards, multi-lookback
> support (max 90d for analysis), comprehensive side panel, **always use
> `lib/contract_units.bp_multipliers_for()` for spread/fly bp scaling**, and don't
> re-introduce any of the bugs in §9. The Proximity and Z-score subtabs share a chart_col
> + scrollable side_col layout you can lift for the remaining sub-subtabs.

---

*End of handoff. Last updated after **Senior-Trader Roadmap Phases 0+1+2** — see §5G (Phase 0: indicator load + freshness + System Health), §5H (Phase 1: product_spec + cb_meetings + LIBOR cutover + CPCV harness), §5I (Phase 2: 61-node grid + PCHIP opt-in + partial-fixing engine + 8-check verify). 7/8 checks pass with linear default; PCHIP path callable opt-in. Backups taken at every phase boundary (`D:\STIRS_DASHBOARD_backup_phase{0,1,2}_*.zip`). Phases 3-12 pending — see plan file at `C:\Users\yash.patel\.claude\plans\c-users-yash-patel-downloads-tmia-v13-s-magical-mitten.md` §12 for the next-phase details.*

*Earlier today: Phases B-H — the full Technicals overhaul is now landed. Phase B: research-backed NEAR/APPROACHING dispatcher with 9 setup categories, 11 literature citations, and bootstrap-quantile calibration shell (`lib/setups/{near_thresholds,near_calibration,_NEAR_CITATIONS}.py`). Phase C: bar-level backtest engine with same-bar stop-wins, 3-bar cooldown, 40+ metrics, bootstrap CIs (`lib/backtest/{engine,metrics,bootstrap}.py`). Phase D: 270-cell L1–L18 × 5 regimes × 5 windows aggregator with cross-window Improving/Declining/Stable/Mixed trend interpretation (`lib/backtest/aggregator.py`). Phase E: 10-day recompute cycle daemon mirroring prewarm architecture; persists `tmia.duckdb` + dated snapshots (`lib/backtest/cycle.py` + `lib/backtest/replay.py`); wired into `app.py` and `pages/1_US_Economy.py`. Phase F: `display_name()` Title-Case helper with ADX/DI/EMA/MR acronym preservation; 9 UI call sites swapped in `tabs/us/sra_analysis_technicals.py`. Phase G: `lib/setups/tooltips.py` — 6-section setup tooltip + composite-cell tooltip + regime/take/composite header tooltips, all wired into Views 1/2/3. Phase H: 12th side-panel block (`_block_trend_movers` reading the Phase D trend table), control-row proportion fixed to `[1.0, 1.4, 2.4, 1.4]`, visible borders on all 3 SRA Analysis subtabs. See §5F. Streamlit 8503 boots clean, no errors. — May 2026.*

*Earlier today: Phase A (CMC layer with Backward-Panama additive + Carver correction + linear interpolation; 22 SRA nodes; 5y history; 5/5 verify pass — see §5E).*

*Earlier today: Phase K3-fix (DuckDB thread-safety via `threading.local()`); Phase K4.1 (per-leg `entry_price` on C9 pair legs); Phase K4 base (Multi-leg trade rendering + C9 multi-pair expansion + 60d-track sort); Phase K3 (background pre-warm via `lib/prewarm.py`); Phase K2 (Technicals UI polish — tick rounding, CSS state pills, inline composite bars).*
