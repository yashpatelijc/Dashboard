# PCA Trade Screener — Canonical Reference

This is the complete reference for the PCA tab (`tabs/us/sra_pca.py`) — a relative-value trade screener for **CME 3-Month SOFR futures (SR3)**.

The engine fits a 3-factor PCA on the SR3 yield curve, scans every instrument (outright, listed spread, listed fly, plus synthetic baskets) for stretched residuals, gates the candidates through three independent stationarity tests, ranks the survivors by a 17-input conviction score, and translates each idea into actual SR3 (or CME-listed spread/fly) contract prices that you can execute directly.

Every section below covers **what it does · the math · how to read the numbers · why it matters · where to find it in the code**.

---

## Contents

**Part I — Architecture & data**
1. [Architecture & data flow](#1-architecture--data-flow)
2. [Data sources](#2-data-sources)
3. [The panel dictionary — single source of truth](#3-the-panel-dictionary)

**Part II — The factor model**
4. [The PCA model (Litterman-Scheinkman)](#4-the-pca-model)
5. [CMC panel — interpolated yield surface](#5-cmc-panel)
6. [PC sign normalization](#6-pc-sign-normalization)
7. [Loadings, scores, and reconstruction](#7-loadings-scores-reconstruction)

**Part III — Residuals & statistics**
8. [Residual computation](#8-residual-computation)
9. [Mean-reversion statistics](#9-mean-reversion-statistics)
10. [The triple-stationarity gate](#10-triple-stationarity-gate)
11. [Hurst exponent + composite reversion score](#11-hurst-composite)

**Part IV — Trade ideas**
12. [The 19 trade generators](#12-trade-generators)
13. [The 13 engine-state generators (overlays)](#13-engine-state-generators)
14. [Cross-confirmation clustering](#14-cross-confirmation-clustering)
15. [Lifecycle states (NEW/MATURING/PEAK/FADING)](#15-lifecycle-states)
16. [Hard-block filters](#16-hard-block-filters)
17. [The 17-input conviction score](#17-conviction-score)

**Part V — Price translation (the hardest part)**
18. [Sign convention — yield vs contract space](#18-sign-convention)
19. [Outright SR3 price conversion](#19-outright-sr3-conversion)
20. [Multi-leg price conversion — three modes](#20-multi-leg-conversion)
21. [CME-listed spread/fly contracts](#21-listed-spread-fly)
22. [Tick rounding and dollar conventions](#22-tick-rounding)
23. [Cross-check methodology (the trust chip)](#23-cross-check)

**Part VI — Cross-validation models**
24. [Analog FV (Mahalanobis k-NN + Ledoit-Wolf)](#24-analog-fv)
25. [Path-conditional FV (Heitfield-Park bootstrap)](#25-path-fv)
26. [Carry + roll-down](#26-carry-roll)

**Part VII — Regime & cycle**
27. [HMM + GMM regime classification](#27-regime-classification)
28. [Cycle phase mapping](#28-cycle-phase)
29. [Cross-asset overlays (8 analyses)](#29-cross-asset)
30. [Convexity bias (Piterbarg short-rate)](#30-convexity)

**Part VIII — Interpretation & UI**
31. [Dynamic per-trade narrative](#31-dynamic-narrative)
32. [Per-factor interpretation](#32-factor-interpretation)
33. [Trade Card layout](#33-trade-card-layout)
34. [Filter strip + grouping](#34-filter-strip)
35. [Engine Health + TAPE pill strips](#35-pill-strips)
36. [Concept tooltips + glossary](#36-glossary)

**Part IX — Validation, bugs, and references**
37. [Research validation against standard methods](#37-research-validation)
38. [Bugs found and fixed](#38-bugs-fixed)
39. [Module reference](#39-module-reference)
40. [Key data structures](#40-data-structures)
41. [Quick-reference index](#41-quick-ref)

---

<a name="1-architecture--data-flow"></a>
## 1. Architecture & data flow

```
                                ┌──────────────────────────────────────┐
                                │  market_data_v2_*.duckdb             │
                                │  (CME SR3 OHLC + listed spread/fly   │
                                │   + BBG eco + UST + FX + credit)     │
                                └────────────┬─────────────────────────┘
                                             │
        ┌────────────────────────────────────┴───────────────────────────┐
        │                                                                │
        ▼                                                                ▼
┌───────────────────┐                                       ┌────────────────────────┐
│ load_sra_curve_   │                                       │ load_listed_spread_fly_│
│ panel("outright") │                                       │ panel                  │
│ → outright_close  │                                       │ → spread_close_panel   │
│                   │                                       │ → fly_close_panel      │
└─────────┬─────────┘                                       └───────────┬────────────┘
          │                                                             │
          ▼                                                             │
┌───────────────────┐    ┌────────────────────┐                         │
│ _outright_yield_  │    │ build_cmc_panel    │                         │
│ bp_panel          │    │ (interpolated      │                         │
│ y = (100-p) × 100 │    │  yields at fixed   │                         │
│                   │    │  tenors 1m..60m)   │                         │
└─────────┬─────────┘    └──────────┬─────────┘                         │
          │                         │                                   │
          │                         ▼                                   │
          │           ┌──────────────────────────────┐                  │
          │           │ fit_pca_static(cmc.diff())   │                  │
          │           │ → loadings (3 × 58)          │                  │
          │           │ → eigenvalues               │                  │
          │           │ → variance_ratio (PC1-PC3)   │                  │
          │           │ → PCAFit dataclass           │                  │
          │           └──────────┬───────────────────┘                  │
          │                      │                                      │
          ▼                      ▼                                      │
┌─────────────────────────────────────────────────────────────────────┐ │
│                              panel dict                              │ │
│  outright_close_panel · yield_bp_panel · cmc_panel · pc_panel       │ │
│  pca_fit_static · spread_close_panel · fly_close_panel              │ │
│  spread_catalog · fly_catalog · sofr_panel · cross_pc_corr          │ │
│  regime_stack · fomc_calendar_dates · …                             │ │
└─────────────────────────────────┬───────────────────────────────────┘ │
                                  │                                     │
                                  ▼                                     │
        ┌─────────────────────────────────────────────────┐             │
        │  per_traded_*_residuals + structure_candidates  │             │
        │  (yield-bp space, change-space residuals,       │             │
        │   triple-gate verdicts, z-scores, half-lives)   │             │
        └────────────────────────┬────────────────────────┘             │
                                 ▼                                      │
        ┌─────────────────────────────────────────────────┐             │
        │  19 TRADE_GENERATORS + 13 ENGINE_STATE_GENERATORS │           │
        └────────────────────────┬────────────────────────┘             │
                                 ▼                                      │
        ┌─────────────────────────────────────────────────┐             │
        │  cluster_by_fingerprint → merge duplicates     │             │
        │  + count cross-confirming sources              │             │
        └────────────────────────┬────────────────────────┘             │
                                 ▼                                      │
        ┌─────────────────────────────────────────────────┐             │
        │  hard_block_filter (print quality, event-in-    │             │
        │  window, convexity warning)                     │             │
        └────────────────────────┬────────────────────────┘             │
                                 ▼                                      │
        ┌─────────────────────────────────────────────────┐             │
        │  score_conviction (17 inputs) → conviction ∈[0,1]│             │
        └────────────────────────┬────────────────────────┘             │
                                 ▼                                      │
        ┌─────────────────────────────────────────────────┐             │
        │  convert_to_sr3_prices                          │ ◄───────────┘
        │  → outright price OR listed spread/fly OR       │
        │    synthetic OR per-leg                         │
        │  → SR3Prices dataclass                          │
        └────────────────────────┬────────────────────────┘
                                 ▼
        ┌─────────────────────────────────────────────────┐
        │  Trade Card UI (collapsed by default)           │
        │  + dynamic narrative + per-factor interp +      │
        │  cross-check chip + 90-day charts + concepts    │
        └─────────────────────────────────────────────────┘
```

**Files**:
- `lib/sra_data.py` — DuckDB loaders + listed-symbol derivation helpers
- `lib/pca.py` — PCA fit + CMC panel + structure enumeration + residual scanners
- `lib/pca_trades.py` — 19 trade generators + 13 engine-state generators + clustering + conviction scoring
- `lib/mean_reversion.py` — ADF, KPSS, VR, Hurst, OU half-life, composite reversion score, triple-gate
- `lib/pca_analogs.py` — Mahalanobis k-NN + Ledoit-Wolf shrinkage + path-bucket assignment
- `lib/pca_step_path.py` — Heitfield-Park step-path bootstrap
- `lib/pca_regimes.py` — GMM + HMM fitting
- `lib/pca_dossier.py` — per-contract 18-section dossier
- `lib/pca_trade_interpretation.py` — SR3 price conversion + factor interpretation + dynamic narrative
- `lib/pca_concepts.py` — concept glossary (26 entries)
- `lib/pca_cross_asset.py` + `lib/pca_cross_asset_analysis.py` — vol/equity/credit/FX/UST/term-premia/convexity overlays
- `tabs/us/sra_pca.py` — Streamlit UI

---

<a name="2-data-sources"></a>
## 2. Data sources

| Source | What | Where | Used for |
|---|---|---|---|
| `mde2_timeseries` (DuckDB) | 1D OHLC for every SR3 outright, listed spread, listed fly | `lib/sra_data.py:load_sra_curve_panel`, `load_listed_spread_fly_panel` | Building yield panels, listed-contract entry prices |
| `mde2_contracts_catalog` (DuckDB) | Contract metadata: symbol, expiry_year/month, tenor_months, strategy, curve_bucket | `lib/sra_data.py:get_sra_live_symbols` | Symbol enumeration, listed-product matching |
| BBG eco parquets | Macro economic surprises (CPI, NFP, ADP, ISM, etc.) | `lib/connections.py:read_bbg_parquet_robust` | Event sensitivity in dossier |
| BBG vol indices | MOVE, SRVIX, SKEW, VIX, VVIX, OVX | `lib/pca_cross_asset.py:load_volatility_indices` | Vol regime overlay |
| BBG equity indices | SPX, NDX, INDU | `lib/pca_cross_asset.py:load_equity_indices` | Risk-on/off state |
| BBG FX | EURUSD, GBPUSD, USDJPY, USDCHF | `lib/pca_cross_asset.py:load_fx_panel` | FX-implied rate differentials |
| BBG credit | CDX IG, CDX HY, Bloomberg LUACOAS (IG OAS), LF98OAS (HY OAS) | `lib/pca_cross_asset.py:load_credit_panel` | Credit-cycle recession probability |
| BBG UST yields | 2Y, 5Y, 10Y, 30Y on-the-run | `lib/pca_cross_asset.py:load_ust_panel` | Term-premia decomposition |
| NY Fed ACM | Adrian-Crump-Moench term premia (2/5/10/30Y) | `lib/pca_cross_asset.py:load_acm_term_premia` | Yield decomposition into expected-path + term-premium |
| FOMC calendar | Meeting dates + decisions | `lib/fomc.py:load_fomc_meetings` | Days-to-FOMC, step-path bootstrap, event sensitivity |
| TreasuryDirect API | Upcoming UST auction schedule (cached) | `lib/pca_cross_asset.py:load_treasury_auction_calendar` | Calendar overlay in dossier |
| `sofr_panel` (DuckDB) | Daily SOFR fixings | Loaded inside `build_full_pca_panel` | Front-pin for CMC interpolation |

The DuckDB files are pre-built snapshots at `D:\Python Projects\QH_API_APPS\analytics_kit\db_snapshot\market_data_v2_*.duckdb` — the dashboard reads the most recently modified one (read-only).

---

<a name="3-the-panel-dictionary"></a>
## 3. The panel dictionary — single source of truth

`build_full_pca_panel(asof, history_days=700)` in `lib/pca.py` returns a dict with all derived state. Every downstream function reads from this one place. Key entries:

| Key | Type | What |
|---|---|---|
| `asof` | `date` | The snapshot date |
| `info` | dict | Counts of outrights/spreads/flies + history window |
| `outrights_df` | DataFrame | Live SR3 outright metadata (symbol, expiry, curve_bucket) |
| `outright_symbols` | list | Ordered SR3 outright symbols |
| `outright_close_panel` | DataFrame | wide: date × outright_symbol → close price |
| `yield_bp_panel` | DataFrame | wide: date × outright → yield in bp = (100−price)×100 |
| `spread_close_panel` | DataFrame | wide: date × listed_spread_symbol → close (in bp of price-diff) |
| `fly_close_panel` | DataFrame | wide: date × listed_fly_symbol → close (in bp of price-diff) |
| `spread_catalog`, `fly_catalog` | DataFrame | Metadata for listed spread/fly contracts |
| `sofr_panel` | DataFrame | Daily SOFR fixings (front-pin) |
| `cmc_panel` | DataFrame | Constant-Maturity-Curve: date × tenor_months (1..60) → yield bp |
| `pca_fit_static` | PCAFit | Full-sample PCA fit |
| `pca_fit_sparse` | SparsePCAFit | Sparse-PCA cross-check |
| `rolling_fits` | dict | date → PCAFit (rolling 60d window) |
| `pc_panel` | DataFrame | date × {PC1, PC2, PC3} scores |
| `cross_pc_corr` | DataFrame | rolling corr(PCᵢ, PCⱼ) — basis-health diagnostic |
| `reconstruction_pct_today` | float | today's 3-PC explained-variance fraction |
| `anchor_series` | Series | 12M − 24M anchor slope |
| `structure_candidates` | list[StructureCandidate] | PCA-isolated baskets (PC1/PC2/PC3) |
| `residual_outrights_df` | DataFrame | per-outright residuals + gate verdicts |
| `residual_spreads_df`, `residual_flies_df` | DataFrame | per-traded-spread / per-traded-fly residuals |
| `print_quality_alerts` | list[date] | Dates flagged by `cmc_print_quality_alert` |
| `regime_stack` | dict | GMM + HMM fits |
| `fomc_calendar_dates` | list[date] | FOMC meeting dates |
| `cross_asset_analysis` | CrossAssetAnalysis | 8 cross-asset overlays (vol regime, risk state, etc.) |
| `cross_asset_panel` | CrossAssetPanel | Raw loaders for vol/equity/fx/credit/ust/etc. |
| `engine_state_signals` | list[TradeIdea] | Output of the 13 engine-state generators (overlay only) |
| `tenor_grid_months` | list | The 58 tenor points used for CMC interpolation |

Every per-trade computation (residual, conviction, SR3 prices, factor interpretation) reads exclusively from this dict — no global state.

---

<a name="4-the-pca-model"></a>
## 4. The PCA model — Litterman-Scheinkman (1991)

**Reference**: Litterman, R. & Scheinkman, J. (1991), *Common Factors Affecting Bond Returns*, J. Fixed Income 1(1), 54-61.

### The premise

The yield curve has thousands of points but only 3 statistical degrees of freedom that matter. Those 3 — PC1 (level), PC2 (slope), PC3 (curvature) — explain 95%+ of daily yield-change variance. Everything else is noise or instrument-specific idiosyncratic move (= "residual").

### Implementation (`lib/pca.py:fit_pca_static` at L499)

```python
delta = cmc_panel.diff().dropna()          # daily CMC yield changes
X = delta.values                           # shape (n_days, 58_tenors)
feature_mean = X.mean(axis=0)              # ≈ 0 bp/day per tenor
X_c = X - feature_mean                     # centered
U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
loadings = Vt[:3]                          # 3 × 58 eigenvectors
eigenvalues = S**2 / (n_obs - 1)           # variance per PC
variance_ratio = eigenvalues / eigenvalues.sum()
```

### Critical design choices

| Choice | Value | Reasoning |
|---|---|---|
| Input data | **yield CHANGES** (Δy) | L-S framework. Yields are non-stationary; changes are. Required for ADF/KPSS to be valid. |
| Basis | **covariance** (not correlation) | All tenors share units (bp), standardizing distorts loadings. |
| Rotation | **none** | Varimax destroys orthogonality. L-S's natural rotation already gives interpretable factors. |
| Sign convention | enforced via `_normalize_pc_signs` | PC1 positive at front, PC2 monotone front→back, PC3 belly-rule at 24M. |
| Refit cadence | static fit + rolling 60d | Static is the anchor; rolling tracks structural shifts. |
| Cross-PC corr trigger | > 0.30 | Basis-broken flag → PC-isolated trades marked `regime_unstable`. |

### What each PC means physically

| PC | Typical loadings | Interpretation | Variance share |
|---|---|---|---|
| PC1 (Level) | all positive, ~equal | Parallel shift. +1σ PC1 = entire curve up by ~same bp. Reflects Fed policy direction. | ~80-90% |
| PC2 (Slope) | positive front, negative back | Front-back rotation. +1σ = curve steepens (or flattens, depending on sign convention). | ~5-10% |
| PC3 (Curvature) | wave shape (+−+ or −+−) | Belly vs wings. +1σ = belly cheap (high yield) relative to wings, or vice versa. | ~1-3% |

### Implications for the chart's FV reconstruction

**The PCA is fit on yield CHANGES.** The PC scores `PCₖ(t)` are projections of `(Δy_t − mean_Δy)` onto the eigenvectors — they have units of **bp/day**, NOT yield level. Reconstructing a yield LEVEL by `mean + Σ PCₖ(t)·loadingₖ` gives a bp/day quantity, not a yield. Converting via `100 − bp_per_day / 100 ≈ 100` is meaningless.

→ **The price-chart "FV time series" in the Trade Card therefore does NOT plot a reconstructed FV path.** Only today's FV (= entry_price + price-equivalent-of-residual) is shown, as the dashed FV/TARGET horizontal line. This is the only mathematically correct quantity derivable from a change-space PCA.

A level-space PCA would capture trend (yields are non-stationary). That's intentionally NOT done — it would defeat the L-S framework.

---

<a name="5-cmc-panel"></a>
## 5. CMC panel — interpolated constant-maturity yield surface

`build_cmc_panel` in `lib/pca.py` interpolates outright SR3 yields to **fixed-tenor** grid points (1m, 2m, 3m, ..., 60m). The result is a 58-column panel where column 12 always represents "yield 12 months out", regardless of which contract that maps to on any given day.

The interpolation uses PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) for the bulk of the curve, with a SOFR-pin at the front for the 0-1 month region.

**Why interpolate?** Outright contracts roll quarterly (March/June/Sept/Dec), so the front contract's *tenor-from-today* shrinks every day. PCA needs a stable tenor structure to be comparable across days. The CMC panel provides that.

**Output**: a DataFrame indexed by date with 58 columns (1..60 months) of bp yields.

---

<a name="6-pc-sign-normalization"></a>
## 6. PC sign normalization

`_normalize_pc_signs` in `lib/pca.py` enforces canonical sign conventions on the eigenvectors after SVD. Without this, sign flips across days would invalidate cross-day comparisons of PC scores.

| PC | Rule |
|---|---|
| PC1 | Force first loading (front tenor) to be **positive** |
| PC2 | Force loadings to be **monotone front-to-back** (positive at front, negative at back, or vice-versa with a consistent sign at the front) |
| PC3 | **Belly rule** — force the loading at the 24-month tenor to be negative (or positive, depending on convention) so that "+1σ PC3" consistently means "belly cheap" |

---

<a name="7-loadings-scores-reconstruction"></a>
## 7. Loadings, scores, and reconstruction

For day t and tenor τ:
- **Loading** `Lₖ(τ)`: how strongly factor k moves tenor τ per unit of factor change. Set by the fit, time-invariant.
- **Score** `PCₖ(t)`: today's value of factor k. Projection of `(Δy_t − mean)` onto the k-th eigenvector.
- **Reconstruction** of Δy at tenor τ: `Σ_k PCₖ(t) · Lₖ(τ)`.
- **Residual** at (t, τ): `Δy_actual(t, τ) − reconstruction(t, τ)`.

For an instrument that isn't a single CMC tenor (e.g., a contract with tenor changing daily as it ages, or a spread), we interpolate the loading at the contract's current tenor using `_instrument_loadings`.

---

<a name="8-residual-computation"></a>
## 8. Residual computation

For each tradable instrument (outright, listed spread, listed fly), the residual is the part of today's yield change that the 3-PC model **cannot explain**.

```
                     ┌─────────────────┐
                     │  Δyield_today   │ ← actual yield change (bp)
                     └────────┬────────┘
                              │
                              ▼
                     ┌────────────────────────┐
                     │  Σₖ PCₖ(t)·Lₖ + mean  │ ← model-implied change
                     └────────┬───────────────┘
                              │
                              ▼
                  residual = actual − model
                              │
                              ▼
                     ┌─────────────────┐
                     │ residual_today  │ ← today's value of the residual
                     │ in bp           │   series (= idea.entry_bp later)
                     └─────────────────┘
```

**Functions**:
- `per_traded_outright_residuals` — one residual series per outright contract
- `per_traded_spread_residuals` — one per listed spread
- `per_traded_fly_residuals` — one per listed fly

Each returns a DataFrame with: `instrument`, `residual_today_bp`, `residual_z`, `half_life`, `hurst`, `adf_pass`, `gate_quality`, `eff_n`, `composite_score`.

### Sign convention

- **residual_bp > 0** ↔ yield is ABOVE model FV ↔ "rich" yield ↔ SR3 price is BELOW model FV
- **residual_bp < 0** ↔ yield is BELOW model FV ↔ "cheap" yield ↔ SR3 price is ABOVE model FV

The trade hypothesis is **mean reversion**: residual decays back to 0. See §18 for the full sign chain to contract action.

---

<a name="9-mean-reversion-statistics"></a>
## 9. Mean-reversion statistics

### OU half-life (`ou_half_life` in `lib/mean_reversion.py`)

Fits a discrete-time Ornstein-Uhlenbeck process via OLS regression of `x_t` on `x_{t-1}`:

```
x_t = a + b · x_{t-1} + ε_t

→ half_life = −ln(2) / ln(b)
```

Mathematically equivalent to fitting `Δx = κ(μ − x_{t-1}) + σ·dW` with `b = 1−κ` and `HL = ln(2)/κ`. The exact form `−ln(2)/ln(b)` is more accurate than the small-κ approximation `ln(2)/κ`.

**Interpretation thresholds**:
- HL ∈ [3, 30]d — **sweet spot**, ideal for STIR trades
- HL ∈ [30, 60]d — slower but tradeable
- HL < 3d — too fast, often noise
- HL > 60d — too slow, capital tied up
- HL negative — **explosive**, do NOT trade

Expected hold time: **1.5 × HL** trading days.

### ADF (Augmented Dickey-Fuller)

Reference: Said & Dickey (1984). Implemented in `lib/mean_reversion.py:adf_test`.

```
Δx_t = α + γ · x_{t-1} + Σᵢ βᵢ · Δx_{t-i} + ε_t

H₀: γ = 0 (unit root, non-stationary)
Test stat: t-stat on γ
5% critical value: −2.86 (MacKinnon constant-no-trend)
```

**Reject H₀** (test stat < −2.86) → series IS stationary (mean-reverting around a constant).

### KPSS (Kwiatkowski-Phillips-Schmidt-Shin)

Reference: Kwiatkowski, Phillips, Schmidt & Shin (1992). Implemented in `lib/mean_reversion.py:kpss_test`.

```
LM = Σ_t S_t² / (T² · σ̂²(l))

where S_t = Σᵢ≤t eᵢ (partial sum of residuals)
      σ̂²(l) = Bartlett-kernel long-run variance with lag l

H₀: stationary around a level
5% critical value: 0.463
```

KPSS has the **opposite null** of ADF — rejecting KPSS means stationarity is NOT supported.

### Variance Ratio (Lo-MacKinlay 1988)

```
VR(q) = Var(x_t − x_{t−q}) / (q · Var(Δx_t))

For i.i.d. random walk: VR(q) = 1
VR < 1 → mean-reversion (negative autocorrelation)
VR > 1 → trending / momentum (positive autocorrelation)

z-stat (hetero-robust): z = (VR − 1) / √(2(2q−1)(q−1) / (3qN))
```

We use q = 4. Implementation in `lib/mean_reversion.py:lo_mackinlay_variance_ratio`.

---

<a name="10-triple-stationarity-gate"></a>
## 10. The triple-stationarity gate (gameplan §A6)

A residual is only "clean" if three independent stationarity tests agree. This catches false positives that ADF alone misses (ADF over-rejects ~15% of the time on small samples).

`triple_stationarity_gate` in `lib/mean_reversion.py`:

```
clean   ⟺  (ADF rejects @ 5%) AND (KPSS does NOT reject @ 5%) AND |VR − 1| > 0.10
drift   ⟺  ADF rejects   AND KPSS rejects   (drift component present)
random_walk ⟺  ADF fails AND VR ≈ 1.0       (suppressed, idea never reaches table)
non_stationary ⟺  ADF fails AND VR > 1.15   (trending)
```

| Verdict | Action | Conviction impact |
|---|---|---|
| `clean` | Full surface, highest tier | adf_pass adds +0.08 |
| `drift` | Trade smaller, hedge with momentum | adf_pass adds +0.08, but cycle alignment matters more |
| `non_stationary` | Surface with reduced conviction | adf_pass = 0 |
| `random_walk` | **SUPPRESSED** — never reaches the table | n/a |
| `low_n` | eff_n < 30 — conviction capped | eff_n component caps |
| `regime_unstable` | cross-PC corr > 0.30 today | PC-isolated trades flagged |

---

<a name="11-hurst-composite"></a>
## 11. Hurst exponent + composite reversion score

### Hurst (`hurst_exponent` in `lib/mean_reversion.py`)

Classic R/S analysis:
```
For sub-window size n ∈ {8, 16, 32, ..., N/2}:
    R = max(cum_dev) − min(cum_dev)
    S = std(returns)
H = slope of log(R/S) vs log(n)
```

- **H < 0.40** → strong mean-reversion (anti-persistent)
- **H ≈ 0.50** → random walk
- **H > 0.55** → trending / persistent

### Composite reversion score (`composite_reversion_score`)

A single 0..1 number combining z, half-life, Hurst, and ADF:

```python
score = (
    0.40 * sigmoid((|z| − 1.5) / 0.5)        # how stretched
  + 0.25 * (1 if HL ∈ [3,30]d else 0.4 if HL ∈ [30,60] else 0)
  + 0.20 * sigmoid((0.5 − H) / 0.05)         # anti-persistence
  + 0.15 * (1 if adf_pass else 0)
)
```

Used internally as a tiebreaker when ranking residuals.

---

<a name="12-trade-generators"></a>
## 12. The 19 trade generators

Defined as `TRADE_GENERATORS` in `lib/pca_trades.py:1864`. These are the **only** generators whose output reaches the trade table.

### Tier 1 — PCA-isolated structures

These solve the constrained system "find weights such that the basket loads on ONE PC and zeros the others".

| Source ID | Name | Structure | Math |
|---|---|---|---|
| 1 | `gen_pc3_fly_ideas` | butterfly | `min ||w||² s.t. w·L₁=0, w·L₂=0, w·L₃=1` |
| 2 | `gen_pc2_spread_ideas` | spread or basket | `w·L₁=0, w·L₂=1, w·L₃=0` |
| 3 | `gen_pc1_basket_ideas` | basket | `w·L₁=1, w·L₂=0, w·L₃=0` |

Solver: `solve_isolated_pc_weights` in `lib/pca.py`.

### Tier 2 — Cross-factor

| 4 | `gen_anchor_ideas` | 12M − 24M anchor slope, mean-reverts when \|z\| > 1.5 |

### Tier 3 — Segment-specific (multi-resolution)

| 10 | `gen_front_pca_ideas` | PCA fit on just the front (1-12M) of the curve |
| 11 | `gen_belly_pca_ideas` | PCA fit on belly (9-30M) |
| 12 | `gen_back_pca_ideas` | PCA fit on back (24-60M) |

These catch front-specific or back-specific moves that the full-curve PCA might smooth over.

### Tier 4 — Residual fades (per-instrument)

Scan `residual_outrights_df`, `residual_spreads_df`, `residual_flies_df` for `|z| > 1.5`.

| 13 | `gen_outright_fade_ideas` | single outright |
| 14 | `gen_spread_fade_ideas` | listed spread |
| 15 | `gen_fly_arb_ideas` | listed fly |
| 16 | `gen_pack_ideas` | 4-contract pack |
| 26 | `gen_pack_fly_ideas` | spread of packs |
| 27 | `gen_bundle_rv_ideas` | bundle RV |
| 30 | `gen_outlier_reversal_ideas` | extreme residual reversal (\|z\| > 3) |
| 31 | `gen_carry_fly_ideas` | fly weighted to maximize carry while remaining PC3-isolated |

### Tier 5 — Conditional FV

| 17 | `gen_analog_fv_ideas` | Mahalanobis k-NN analog FV (fires when \|analog z\| > 2 AND disagrees with raw residual) |
| 18 | `gen_path_fv_ideas` | policy-path conditional FV (Heitfield-Park bucket) |

### Tier 6 — Event-aware

| 24 | `gen_event_drift_ideas` | post-FOMC/CPI/NFP drift |
| 38 | `gen_event_impact_ranking_ideas` | event-conditional structures |

### What each generator emits

Each returns `list[TradeIdea]` with fields:

```python
@dataclass
class TradeIdea:
    idea_id, leg_fingerprint
    primary_source                # generator name (e.g. "PC3-fly")
    sources                       # tuple of all confirming sources after clustering
    source_id
    direction                     # "long" / "short" — in YIELD space
    structure_type                # "outright" / "spread" / "fly" / "pack" / "basket"
    legs                          # tuple of TradeLeg(symbol, weight_pv01, side)
    pv01_sum
    entry_bp                      # today's residual (= residual_today_bp)
    fair_value_bp                 # = 0 for PC-isolated; analog/path FV for tier 5; mean for anchor
    target_bp                     # = fair_value_bp
    stop_bp                       # entry + sign × 2.5 × |entry|
    expected_revert_d             # = 1.5 × OU half-life
    expected_pnl_bp, expected_pnl_dollar
    z_score, half_life_d, adf_pass, eff_n, gate_quality
    risk_flags
    rationale_html                # 4-part narrative chain
    headline_html                 # 1-line caption
    cycle_phase, cycle_alignment
    analog_fv_bp, analog_fv_z, analog_fv_eff_n
    path_fv_bp, path_fv_z
    conviction, conviction_breakdown    # filled after clustering + scoring
    state, days_alive, max_conviction_to_date
    slippage_estimate_bp, convexity_warning
    factor_exposure, what_if_table
```

---

<a name="13-engine-state-generators"></a>
## 13. The 13 engine-state generators (overlays, not trades)

`ENGINE_STATE_GENERATORS` in `lib/pca_trades.py:1876`. These produce **diagnostic** signals about the engine itself — they go into `panel["engine_state_signals"]` and surface only in the "🔧 detailed diagnostics" expander. They do NOT appear in the trade table.

| Source ID | Name | Watches |
|---|---|---|
| 5 | `gen_cross_pc_corr_breakdown_ideas` | Cross-PC corr > 0.30 (orthogonality lost) |
| 6 | `gen_sparse_dense_divergence_ideas` | Sparse vs dense PCA loadings disagree |
| 7 | `gen_variance_regime_ideas` | PC1 variance unusual vs its rolling distribution |
| 8 | `gen_eigenspectrum_gap_ideas` | Gap between λ₃ and λ₄ shrinking |
| 9 | `gen_pc1_asymmetry_ideas` | PC1 skew (up vs down asymmetric) |
| 19 | `gen_gmm_regime_ideas` | GMM regime probability change |
| 20 | `gen_posterior_degradation_ideas` | HMM dominant-regime posterior collapsing |
| 21 | `gen_bocpd_ideas` | Bayesian online change-point detection |
| 22 | `gen_bai_perron_ideas` | Bai-Perron structural break |
| 23 | `gen_seasonality_ideas` | Day-of-month or week-of-quarter seasonal bias |
| 25 | `gen_days_since_fomc_ideas` | FOMC drift signal |
| 28 | `gen_pc_momentum_ideas` | PC level momentum |
| 29 | `gen_cycle_phase_ideas` | Fed-cycle phase transition |

These are **overlays** that boost or dampen conviction of TRADE_GENERATORS via the conviction-score cross-confirmation bonus.

---

<a name="14-cross-confirmation-clustering"></a>
## 14. Cross-confirmation clustering

`cluster_by_fingerprint` in `lib/pca_trades.py`:

After all 19 generators run, ideas with the same **leg-fingerprint** (sorted leg symbols + direction + sign of each weight) are merged into a single `TradeIdea`. The `sources` tuple accumulates all confirming source names.

This is critical because the same trade often comes out of multiple generators:
- A 1y/2y curve flattener might be emitted by: `gen_pc2_spread_ideas`, `gen_spread_fade_ideas`, `gen_analog_fv_ideas`, `gen_path_fv_ideas`. All four agree → `n_confirming_sources = 4`.

The conviction-score `cross_confirm` component gets `0.05 × ln(1+n_sources) / ln(6)`:
- 1 source → 0.000
- 2 sources → 0.028
- 3 sources → 0.039
- 4 sources → 0.045
- 6+ sources → 0.050 (capped)

In practice this is the single biggest swing factor. An outright fade with z = +1.9 confirmed by 4 generators usually beats a PC3 fly with z = +2.8 confirmed by 1.

---

<a name="15-lifecycle-states"></a>
## 15. Lifecycle states

`assign_lifecycle_state` in `lib/pca_trades.py` tags each idea based on its ledger history:

| State | Definition | Conviction adjustment |
|---|---|---|
| `NEW` | First appearance today | +0.04 |
| `MATURING` | Appeared for 2-5 days, conviction rising | +0.04 |
| `PEAK` | Conviction at maximum-to-date | +0.04 |
| `FADING` | Conviction declining from peak | −0.02 |

The ledger is a SQLite/CSV file recording every (date, idea_id, conviction) tuple historically. Loaded via `load_ledger`.

---

<a name="16-hard-block-filters"></a>
## 16. Hard-block filters

`hard_block_filter` in `lib/pca_trades.py` drops trades that meet ANY of:

- **Print-quality alert today** — `cmc_print_quality_alert` flagged this date (suspicious CMC interpolation, e.g. an outright with a stale settle that contaminated the curve).
- **Event-in-window** — a tier-1 macro release (CPI, NFP, FOMC) falls within `expected_revert_d`. Trade would resolve on the event, not on mean reversion.
- **Convexity warning** — the contract's leg has a convexity bias > 3 bp. SR3-implied rates need OIS adjustment before benchmarking.

The "low_n" and "non_stationary" gate verdicts do NOT trigger a hard-block — those trades are still shown, just with reduced conviction.

---

<a name="17-conviction-score"></a>
## 17. The 17-input conviction score

`score_conviction` in `lib/pca_trades.py:1672`. Each idea gets `conviction ∈ [0, 1]` as the sum of weighted contributions.

| # | Component | Max wt | Computation |
|---|---|---:|---|
| 1 | `z_significance` | 0.130 | `min(1, \|z\|/3)` |
| 2 | `ou_sweet_spot` | 0.090 | HL ∈ [3,30] → 1.0; [30,60] → 0.6; outside → 0.2-0.3 |
| 3 | `adf_pass` | 0.080 | binary |
| 4 | `eff_n` | 0.060 | `min(1, eff_n/100)` |
| 5 | `model_fit` | 0.060 | today's 3-PC explained-variance fraction |
| 6 | `cycle_alignment` | 0.070 | favoured = 1.0, neutral = 0.5, counter = 0.0 |
| 7 | `regime_stable` | 0.050 | HMM dominant conf ≥ 0.6 → 1.0 |
| 8 | `cross_pc_orth` | 0.050 | cross-PC corr ≤ 0.3 → 1.0 |
| 9 | `variance_regime` | 0.050 | regime favours this trade type |
| 10 | `analog_fv_agree` | 0.050 | analog FV sign matches PCA sign |
| 11 | `path_fv_agree` | 0.040 | path FV sign matches PCA sign |
| 12 | `seasonality` | 0.020 | (placeholder) |
| 13 | `event_drift` | 0.015 | (placeholder) |
| 14 | `empirical_hit_rate` | 0.040 | trailing 30-trade revert rate for this source |
| 15 | `lifecycle` | ±0.040 | NEW/MATURING/PEAK = +0.04 ; FADING = −0.02 |
| 16 | `cross_confirm` | 0.050 | `0.05 × ln(1+n_sources)/ln(6)` |
| 17 | `slippage_ok` | 0.030 | slippage/pnl < 0.30 |
| — | `convexity_safe` | 0.030 | not flagged → 1.0 |

**Total** clipped to `[0, 1]`.

### Tier thresholds (visible as the colored border on each card)

- **HIGH (green)**: ≥ 0.70
- **MED (accent)**: 0.55 – 0.70
- **LOW (amber)**: 0.40 – 0.55
- Below 0.40: filtered out by default min-conviction slider

---

<a name="18-sign-convention"></a>
## 18. Sign convention — yield space vs contract space

This is the most subtle part of the engine and was the source of the historical "side" labeling confusion. The chain:

```
residual_bp > 0
  ↔ yield is ABOVE PCA fair value
  ↔ yield is "rich" (high)
  ↔ SR3 price (= 100 − yield/100) is BELOW its model FV
  ↔ price is "cheap"

  Mean reversion expectation:
    yield → FV  (falls)
  ⟺ price → FV (rises)

  Correct CONTRACT action:
    BUY the futures (profit when price rises)
```

```
residual_bp < 0
  ↔ yield is BELOW FV
  ↔ yield "cheap" / price "rich"

  Mean reversion:
    yield → FV (rises)
  ⟺ price → FV (falls)

  Correct CONTRACT action:
    SELL the futures (profit when price falls)
```

### The `TradeLeg.side` field caveat

In `lib/pca_trades.py`, `TradeLeg.side = "sell" if residual_z > 0 else "buy"`. This is in **YIELD space** — "sell the rich yield" — NOT contract space. For an outright SR3 contract, the actual futures order is the OPPOSITE.

`convert_to_sr3_prices` in `lib/pca_trade_interpretation.py` makes this explicit by emitting a `contract_action` field that's always in CONTRACT space ("BUY" or "SELL" — the actual order to send).

### For multi-leg trades

The per-leg `weight_pv01` already encodes the direction. Convention:
- `weight_pv01 > 0` → BUY this leg
- `weight_pv01 < 0` → SELL this leg

This applies to the SR3 contract (price space) for each leg. For a calendar spread (sell front, buy back), the front-leg weight is negative.

---

<a name="19-outright-sr3-conversion"></a>
## 19. Outright SR3 price conversion

`convert_to_sr3_prices(idea, panel)` in `lib/pca_trade_interpretation.py:131` for outrights:

### The math (line-by-line)

```python
price_today      = outright_close_panel[symbol][-1]              # e.g. 96.3400
entry_yield_bp   = (100 − price_today) × 100                     # 366.00 bp
residual_bp      = idea.entry_bp                                 # +1.62 bp (the engine's
                                                                 #            residual_today_bp)
target_yield_bp  = entry_yield_bp − residual_bp                  # 364.38 bp  ← reverts to FV
stop_yield_bp    = entry_yield_bp + 2.5 × residual_bp            # 370.04 bp  ← 2.5× wider
                                                                 #              (against direction)
target_price     = 100 − target_yield_bp/100                     # 96.3562
stop_price       = 100 − stop_yield_bp/100                       # 96.2996

# Round all to tradable tick (0.0025 = ¼ bp = $6.25/contract)
target_price = round(target_price / 0.0025) * 0.0025
stop_price   = round(stop_price   / 0.0025) * 0.0025

contract_action = "BUY" if target_price > entry_price else "SELL"

# P&L per outright contract: 1 bp = $25, so $25/bp = $2,500 per 0.01 of price
pnl_per_contract_dollar  = |target_price − entry_price| × $2,500
risk_per_contract_dollar = |stop_price   − entry_price| × $2,500
risk_reward = pnl / risk
```

### Worked example (live data)

`SRAH29` outright fade, residual = +1.62 bp:
- entry_price = **96.3400**
- target_price = 100 − (366 − 1.62)/100 = **96.3562** (BUY → price expected to rise)
- stop_price = 100 − (366 + 4.05)/100 = **96.2996**
- P&L per contract = (96.3562 − 96.3400) × $2,500 = **$40.50** (rounded to $40)
- Risk per contract = (96.3400 − 96.2996) × $2,500 = **$100**
- R:R = 40 / 100 = **0.40:1**

### Card display

The Trade Card's "💵 Actual SR3 contract prices — SRAH29" block shows ACTION · ENTRY · TARGET · STOP · P&L/c · R:R with yield-space verification row.

---

<a name="20-multi-leg-conversion"></a>
## 20. Multi-leg price conversion — three modes

Multi-leg trades (spreads, flies, packs, baskets) have three possible price-source modes. `convert_to_sr3_prices` picks one based on the leg pattern:

```
                    ┌─────────────────────────────────────┐
                    │  Multi-leg TradeIdea                │
                    └────────────────┬────────────────────┘
                                     │
                  ┌──────────────────┴───────────────────┐
                  │                                      │
                  ▼                                      ▼
        ┌─────────────────────┐                  ┌────────────────┐
        │ Leg pattern matches │                  │ Otherwise      │
        │ a listed CME        │                  │ (engine-       │
        │ contract?           │                  │  generated)    │
        │                     │                  │                │
        │ • 2 legs OR         │                  │                │
        │ • 3 legs with       │                  │                │
        │   target_pc == 3    │                  │                │
        └──────────┬──────────┘                  └───────┬────────┘
                   │                                     │
        ┌──────────┴──────────┐                          │
        │                     │                          │
        ▼                     ▼                          ▼
┌───────────────┐   ┌────────────────┐      ┌─────────────────────┐
│  LISTED       │   │  SYNTHETIC     │      │  PER_LEG            │
│  (gold block) │   │  (blue block)  │      │  (gray block)       │
│               │   │                │      │                     │
│  CME quote ≠ 0│   │  CME quote = 0 │      │  PC2-spread with    │
│  AND in DB    │   │  OR not in DB  │      │  3 legs, PC1-basket │
│               │   │                │      │  with 4 legs, etc.  │
└───────┬───────┘   └────────┬───────┘      └──────────┬──────────┘
        │                    │                         │
        ▼                    ▼                         ▼
  Use actual CME      Reconstruct via            Use per-leg P&L
  market price       Σ cwᵢ · pᵢ × 100             from engine's
                                                  PV01-weighted
                                                  residual
```

### Mode 1: LISTED

For 2-leg combos matching `SRAH26-M26` or 3-leg fly combos matching `SRAH26-M26-U26` in `spread_close_panel` / `fly_close_panel` (and close ≠ 0):

```
listed_entry_bp  = actual CME market quote (units: bp of price-differential)
listed_target_bp = 0                                  ← FV by construction
listed_stop_bp   = 3.5 × listed_entry_bp              ← 2.5× wider from FV
P&L per contract = |target − entry| × $25/bp/contract
tick             = 0.25 bp
```

See §21 for why target = 0 and the unit derivation.

### Mode 2: SYNTHETIC

Same canonical-weight math but reconstructed from the leg prices (when CME quote is 0 or contract is missing):

```
synth_canonical_bp = Σ cwᵢ · leg_priceᵢ × 100   where cw = (1,-1) or (1,-2,1)
synth_target_bp    = 0
synth_stop_bp      = 3.5 × synth_canonical_bp
```

### Mode 3: PER_LEG

For engine-generated multi-leg structures that don't map to a listed product (PC2-spreads with 3 legs, PC1-baskets with 4 legs, packs, custom baskets):

- The "package quote" doesn't exist — each leg trades separately.
- Show **net residual** in PV01-weighted bp (the engine's `idea.entry_bp / Σ|wᵢ|`).
- Show **package P&L** = `Σ leg-P&L` (sum of expected per-leg moves × $25).
- The expanded view shows the full per-leg execution table.

### `price_source` field

Set by `convert_to_sr3_prices`:
```python
price_source = (
    "listed"    if (listed_price is not None AND |listed_price| > 1e-6)
    "synthetic" if synth_canonical_bp is not None
    "per_leg"   otherwise
)
```

The Trade Card UI dispatches on this field to render the correct block (gold / blue / gray).

---

<a name="21-listed-spread-fly"></a>
## 21. CME-listed spread/fly contracts

### The catalog naming

Listed SR3 spreads and flies are stored in `mde2_contracts_catalog` with strategy `'spread'` or `'fly'`. The symbol convention:

- **Spread**: `SRA<front-month-code><yy>-<back-month-code><yy>` → e.g. `SRAH26-M26`
- **Fly**: `SRA<front><yy>-<mid><yy>-<back><yy>` → e.g. `SRAH26-M26-U26`

`tenor_months` in the catalog = the **gap** between adjacent legs (in months). Supported: 1, 2, 3, 6, 9, 12.

`lib/sra_data.py:derive_listed_spread_symbol(leg_symbols)` / `derive_listed_fly_symbol(leg_symbols)` build the listed symbol from outright leg names. Legs are sorted by (year, month) via the standard CME month-code map:

```python
_MONTH_CODE_TO_NUM = {"F":1, "G":2, "H":3, "J":4, "K":5, "M":6,
                        "N":7, "Q":8, "U":9, "V":10, "X":11, "Z":12}
```

### The units convention (the bug I fixed)

CME-listed SR3 spread and fly contracts are quoted in **basis points of price-differential** (×100 of raw price units), not in raw price units like outrights. Verified empirically by reconstructing each listed close from the underlying outright legs:

```
SRAH27-H28 listed_close = -13.5      (CME quote in bp)
SRAH27_price − SRAH28_price = -0.14  (reconstructed from outrights, in price units)
-0.14 × 100 = -14                    (matches listed within 0.5 bp settlement rounding)
```

So:
- **1 unit of listed = 1 bp of price-differential = $25 per CME contract**
- **Tick = 0.25 bp = $6.25 per CME contract** (same dollar tick value as outright)

Constants in `lib/pca_trade_interpretation.py`:
```python
LISTED_BP_TICK = 0.25
USD_PER_LISTED_BP = 25.0
```

### Why target = 0 (FV by construction)

For canonical weights `cw` (=(1,-1) for spread or (1,-2,1) for fly), `Σ cwᵢ = 0`. Then:
```
listed_price = Σ cwᵢ · pᵢ                            [definition]
             = Σ cwᵢ · (100 − yᵢ/100)
             = 100 · Σ cwᵢ  −  (1/100) Σ cwᵢ · yᵢ
             = 0  −  canonical_yield_residual / 100
```

So **listed_price = −canonical_yield_residual / 100**. When canonical residual = 0 (mean-reversion target), listed_price = 0. The FV listed price for ANY canonical CME spread or fly is exactly **0**.

Therefore `listed_target_bp = 0` is exact, not an approximation.

### Why stop = 3.5 × entry

The engine's standard "2.5× wider" rule:
```
stop_residual = entry_residual + 2.5 × (entry_residual − target_residual)
             = entry_residual + 2.5 × (entry_residual − 0)
             = 3.5 × entry_residual
```

Same sign as entry — the stop is 2.5× further from FV (= 0) in the same direction.

For a fly currently at −0.5 bp (slightly cheap):
- target = 0 (move up toward FV)
- stop = −1.75 bp (move down further from FV)

---

<a name="22-tick-rounding"></a>
## 22. Tick rounding and dollar conventions

| Instrument | Tick | $/tick | $/0.01 (1 bp) | Notes |
|---|---|---|---|---|
| SR3 outright | 0.0025 | $6.25 | $25 | Quoted price = 100 − yield/100 |
| Listed calendar spread | 0.25 bp | $6.25 | $25 | Quoted in bp-of-price-diff |
| Listed butterfly | 0.25 bp | $6.25 | $25 | Quoted in bp-of-price-diff |

`_round_to_tick(p)` in `lib/pca_trade_interpretation.py` rounds to 0.0025 (outright).
`_round_to_listed_tick(bp)` rounds to 0.25 (listed spread/fly).

P&L formulas:
- **Outright**: `|target_price − entry_price| × $2,500` (since 1 bp = 0.01 price = $25)
- **Listed**: `|target_bp − entry_bp| × $25` (since 1 listed unit = 1 bp = $25)

The renderer uses the appropriate constant based on `is_outright`.

---

<a name="23-cross-check"></a>
## 23. Cross-check methodology (the trust chip)

Every LISTED block carries a cross-check between the actual CME quote and the reconstruction-from-legs:

```python
synth_canonical_bp = Σ cwᵢ · leg_priceᵢ × 100
diff = listed_price − synth_canonical_bp
```

If the engine's leg combination correctly maps to the listed product, `|diff|` should be ≤ 1 bp (only sub-bp settlement rounding remains). Larger Δ means either:
- The CME quote and underlying outright closes are timed differently (a stale settle)
- OR the engine's leg pattern doesn't actually correspond to the listed contract (a code bug — would be a red flag)

| Δ (bp) | Chip | Verdict |
|---|---|---|
| **≤ 1.0** | 🟢 `✓ matches legs` | Sub-bp settlement rounding only — math is consistent |
| **1.0 – 5.0** | 🟡 `⚠ mild Δ` | Minor data drift, trade still actionable, inspect |
| **> 5.0** | 🔴 `⚠ verify` | Real disagreement — leg pattern may not map, do NOT trade until verified |

The chip is rendered in the top-right of the LISTED block. The detailed Δ value appears in the cross-check line below the price grid.

This is your **end-to-end consistency gauge**: green = engine, DB, and canonical math all agree.

---

<a name="24-analog-fv"></a>
## 24. Analog FV — Mahalanobis k-NN with Ledoit-Wolf shrinkage

Reference: Mahalanobis (1936); Ledoit & Wolf (2004), *J. Multivariate Analysis* 88, 365-411.

### The premise

"Given today's state (PC1/PC2/PC3/anchor/vol/etc.), what residual does each contract carry on AVERAGE across historically similar days?"

### State vector

```python
x_t = [PC1(t), PC2(t), PC3(t), Anchor(t), σ_PC1_20d(t), regime_dummies, ...]
```

### Mahalanobis distance

```
d²(x, y) = (x − y)ᵀ · Σ_LW⁻¹ · (x − y)
```

`Σ_LW` is the **Ledoit-Wolf shrunk** covariance:
```
Σ_LW = (1−α) · Σ_sample + α · F             where F = scaled identity target
α    = min(1, π̂ / γ̂ / n)                  optimal shrinkage by Ledoit-Wolf 2004
π̂    = sum of squared deviations            (asymptotic variance estimator)
γ̂    = Frobenius distance: ||Σ_sample − F||²
```

Shrinkage handles the small-sample inversion problem — pure sample covariance has unbounded estimation error when dimensions ≈ observations.

### Soft-weighted k-NN

Standard k-NN takes the k closest neighbors and averages. We use exponentially-weighted k-NN with adaptive bandwidth:

```python
weight_i = exp(−d²ᵢ / h) · exp(−|days|/H_time)
FV       = Σᵢ weightᵢ · contract_residualᵢ
```

- `h = median(d²)` (adaptive bandwidth — k=30 effective neighbors)
- `H_time` (typically 252 trading days) down-weights stale neighbors

The deviation from textbook k-NN is intentional: a discrete cutoff creates a step-function FV; soft weighting smooths transitions.

### `eff_n`

Number of "good" neighbors (weight > threshold). If eff_n < 30 → `gate_quality = "low_n"` — surfaced with reduced conviction.

### Implementation

`lib/pca_analogs.py:knn_analog_search` (line 120). Returns `AnalogFVResult` dataclass with `fv_bp`, `residual_z`, `eff_n`, `gate_quality`.

---

<a name="25-path-fv"></a>
## 25. Path-conditional FV — Heitfield-Park bootstrap

Reference: Heitfield & Park (2014), Federal Reserve Working Paper.

### The 5-bucket lattice

Each FOMC meeting's implied step is bucketed using ±12.5 bp and ±37.5 bp thresholds:

| Bucket | Implied step (bp) | Lattice center |
|---|---|---|
| `large_cut` | < −37.5 | −50 |
| `cut` | [−37.5, −12.5] | −25 |
| `hold` | [−12.5, +12.5] | 0 |
| `hike` | [+12.5, +37.5] | +25 |
| `large_hike` | > +37.5 | +50 |

### Block bootstrap

```python
n_draws = 50
block_size = 5 days     # preserves daily autocorrelation
lookback = 60 days
```

For each draw, block-bootstrap the trailing 60-day CMC panel and recompute the implied step at each upcoming FOMC. Kernel-weight each draw's bucket assignment:

```
P(bucket | meeting_m) ∝ Σ_draws max(0, 1 − |center_b − implied_draw| / σ)
                       σ = 12.5 bp     # half-step size
```

### Why σ = 12.5 bp (the fix)

Original implementation used σ = 25 bp which smeared ~50% mass into adjacent buckets even at the center. Fixed to 12.5 bp so the at-center kernel value is 1.0 with zero leakage to adjacent centers (their `|center − implied|` = 25 bp ≥ σ).

### Why bracket window = 0.15 months (the fix)

`_bracket_meeting_yields` interpolates the CMC at narrow windows around the FOMC date to extract the implied step. Original `eps_m = 0.5` (half month) was too wide — smeared the step magnitude across other meeting effects. Fixed to 0.15 months (~4 business days).

### Path-conditional FV

Each historical day is bucketed by its implied 4-quarter Fed path. Today's path-FV for a contract = mean residual on historical days in the same bucket. If today's PCA-residual sign disagrees with path-FV sign AND |path-FV z| > 0.5, that's a conviction kill.

Implementation: `lib/pca_step_path.py:fit_step_path_bootstrap`.

---

<a name="26-carry-roll"></a>
## 26. Carry + roll-down

Implemented in `lib/pca_dossier.py:_section_fv_views`.

```
carry_bp     = front_yield − next_yield                (annualized differential)
roll_down_bp = (∂y/∂τ) × (Δτ / 365)                    (slide down the curve)
total        = carry + roll
```

For each contract, the third FV view shows expected P&L over the holding period purely from the curve's shape — no rate moves assumed.

- **Positive carry + roll**: contract earns yield just by sitting (good for long positions)
- **Negative**: contract bleeds value if curve stays flat (need a fast move to overcome)

Used to discount conviction: a trade with +5 bp expected mean-reversion but −3 bp/month negative carry has a much narrower net edge.

---

<a name="27-regime-classification"></a>
## 27. HMM + GMM regime classification

`lib/pca_regimes.py:fit_regime_stack`.

### GMM (Gaussian Mixture Model)

K=6 components, full covariance, EM with 50 k-means++ random restarts (best log-likelihood wins). Features:
```
[PC1, PC2, PC3, Anchor, σ_PC1_20d]
```

Each day gets a posterior probability per component → "soft" regime classification.

### HMM (Hidden Markov Model)

Initial state distribution + transition matrix learned via Baum-Welch. Emissions inherit from the GMM fit (Gaussian, same parameters).

For each day:
- `posterior[t, k]` (γ in HMM notation): P(state = k | observations) via forward-backward
- `dominant_state[t]` = argmax γ
- `dominant_confidence[t]` = max γ

### Conviction impact

`regime_stable` component = 0.05 if `dominant_confidence_today ≥ 0.60`, else 0.

### Caveat (noted in §38)

The implementation reconstructs emissions from posteriors instead of storing X — fine for posterior γ but introduces a per-t shift in transition M-step ξ. Worth a future refactor.

---

<a name="28-cycle-phase"></a>
## 28. Cycle phase mapping

Heuristic Fed-cycle classification from FDTR trajectory + days-since-last-break + HMM regime:

| Phase | Definition |
|---|---|
| `early-cut` | First 1-3 cuts in a new easing cycle |
| `mid-cut` | 4+ cuts, still actively easing |
| `late-cut` | Near terminal, decelerating |
| `trough` | Holding low, awaiting hike cycle |
| `early-hike` | First 1-3 hikes |
| `mid-hike` | Active hiking cycle |
| `late-hike` | Near terminal, decelerating |
| `peak` | Holding high (current regime as of 2025-2026) |

Each phase has a historically favored trade direction:
- `early-cut` → long front rates (curve bull-flattens)
- `mid-hike` → short front, long back (bear-flattener)
- `peak` → fade extremes (high vol-of-vol)

### Conviction impact

`cycle_alignment` component:
- `favoured` (trade matches phase-favored side) → +0.07
- `neutral` → +0.035
- `counter` → 0

---

<a name="29-cross-asset"></a>
## 29. Cross-asset overlays (8 analyses)

`lib/pca_cross_asset.py` + `lib/pca_cross_asset_analysis.py`. Eight analyses bias the conviction of every trade:

### 29.1 Vol regime
Composite z-score of MOVE, SRVIX, SKEW, VIX over 252d. Buckets:
- `quiet` (z < −0.75) → fades work best
- `normal` (−0.75 .. 0.75) → standard
- `stressed` (0.75 .. 1.5) → analog FV bands widen ×1.4
- `crisis` (> 1.5) → bands widen ×2.0

### 29.2 Risk-on/off state
```
score = −z(SPX_21d_ret) + 0.5·z(DXY_21d_ret) + z(credit_chg) + 0.5·z(MOVE_chg)
```
Buckets: `risk_on / neutral / risk_off / panic`.

### 29.3 Credit-cycle recession probability
```
recession_prob_4w  = logistic(0.5·z_IG + 0.5·z_HY − 1.5)
recession_prob_12w = logistic(1.5·z_IG + 0.5·z_HY − 1.0)
```
Plus "IG/HY diverging" flag (late-cycle warning).

### 29.4 Cross-asset lead-lag matrix
For each (PC, asset ∈ {SPX, MOVE, IG, DXY}) pair, find the lag in [−10, +10]d that maximizes |corr|. Negative lag = asset leads PC.

### 29.5 Equity-rates correlation regime
Rolling 63d corr(Δlog SPX, ΔPC1). Buckets:
- `normal_negative` (< −0.20)
- `flipped_positive` (> +0.20) — inflation regime
- `transitioning` (between)

### 29.6 Term-premia decomposition (Adrian-Crump-Moench)
For each UST tenor: `yield = expected_path + term_premium`. Uses published ACM series. Today's 30d Δ on each component identifies whether yield moves are expectations or premium driven.

### 29.7 FX-implied rate differential
EUR/JPY 30d z-score composite. Signals FX divergence.

### 29.8 Convexity bias (Piterbarg short-rate form)
See §30.

### Surfaces in UI

- "🌐 TAPE" pill strip at the top of the screener shows vol regime / risk state / recession prob / SPX 5d % / MOVE / DXY
- Each Trade Card's narrative paragraph 5 (WHAT KILLS IT) flags adverse cross-asset state

---

<a name="30-convexity"></a>
## 30. Convexity bias (Piterbarg short-rate form)

Reference: Piterbarg (2006), *Risk Magazine*, 79-84.

### The math (corrected formula)

```
bias_bp(T) = −0.5 · σ_r² · T · τ · 10⁴

where σ_r          = annualized short-rate vol (decimal)
                   = realized_front_yield_daily_σ × √252
      T            = years to futures expiry
      τ            = SOFR accrual period = 0.25 years
```

Negative bias → SR3-implied forward rate is LOWER than OIS-implied forward → SR3 futures price is HIGHER than OIS-implied price.

### The fix

Original formula used the standard PC1-score σ as if it were a yield vol. That's incorrect — PC1 scores are unitless projections, not bp/year. Fixed to use realized front-SR3 yield σ (in decimal) at `lib/pca_cross_asset_analysis.py:381`.

Original formula form was `-0.5 · σ² · τ²` which is Ho-Lee (not Mercurio as labeled). Fixed to the canonical Piterbarg form above.

### Magnitude

For σ_r ≈ 90 bp/yr and τ = 0.25y:
- T = 1y → bias ≈ −0.10 bp (negligible)
- T = 2y → bias ≈ −0.20 bp
- T = 5y → bias ≈ −0.50 bp (still small)
- T = 10y → bias ≈ −1.00 bp (start to matter for OIS benchmarking)

`convexity_warning` fires when bias > 3 bp on any leg of a trade → trade flagged + conviction docked 0.03.

---

<a name="31-dynamic-narrative"></a>
## 31. Dynamic per-trade narrative

`build_dynamic_narrative(idea, sr3, factors, panel)` in `lib/pca_trade_interpretation.py:781`. Generates a 5-paragraph HTML narrative using THIS trade's actual numbers.

| Paragraph | Contents |
|---|---|
| ① **What we observed** | Today's residual, z-score, source description (which generator, what it does), analog FV cross-check, path-FV bucket |
| ② **Why it's tradeable** | OU half-life, triple-gate verdict, sample size, named confirming sources, HMM regime label + dominant confidence, cycle phase + alignment |
| ③ **Why this direction** | Residual-sign → yield-action → contract-action chain. For multi-leg: per-leg BUY/SELL with PV01 weights |
| ④ **The trade** | Execute string + **derivation math monospace block** + slippage estimate + **numerical conviction breakdown table** (top 8 components) |
| ⑤ **What kills it** | Caveats from per-factor analysis (gate failures, low_n, regime transition), convexity warning, slippage > 30% of edge, vol regime stressed/crisis, print-quality alert today |

Each number cited in the narrative is the trade's actual computed value — no generic copy. Branches dynamically by:
- Source (PC3-fly / PC2-spread / PC1-basket / outright-fade / analog-FV / anchor / segment-PCA / fly-arb)
- Gate verdict (clean / drift / non_stationary)
- Half-life bucket (sweet spot / too fast / too slow)
- Direction (long / short)
- Cross-confirm count (1 / 2 / 3+)

The derivation math block (paragraph ④) shows the exact entry_yield → residual → target_yield → target_price → P&L chain so the user can audit the conversion.

---

<a name="32-factor-interpretation"></a>
## 32. Per-factor interpretation

`interpret_factors(idea, panel)` in `lib/pca_trade_interpretation.py:688`. Returns a list of `FactorInterpretation` objects, each with:

```python
key                      # links to CONCEPTS dict in lib/pca_concepts.py
value_display            # e.g. "+1.74σ" or "5.6 d"
tier                     # "supportive" | "neutral" | "caveat"
headline                 # 1-line "what this means for THIS trade"
detail                   # 1-paragraph explanation
weight_in_conviction     # contribution to total conviction (matches score_conviction's weights)
```

10 interpreters cover: residual z, OU half-life, ADF, triple-gate, eff_n, HMM regime, cycle alignment, cross-confirmation, cross-asset overlay, lifecycle.

Rendered as a stack of color-coded chips inside the expanded Trade Card. Each chip has a nested "↳ What is X?" disclosure linking to the concept glossary.

---

<a name="33-trade-card-layout"></a>
## 33. Trade Card layout

Each trade renders as a collapsed card by default. The collapsed view shows:

```
┌──────────────────────────────────────────────────────────────────────────┐
│ ▌ #1 [HIGH] SRAM27 outright outright-fade   BUY  96.3400  96.3562        │
│                                                  96.2996  $40    $100    │
│                                                  0.40:1  ▰▰▰▰▱▱▱ 0.65   │
│ "outright fade · +1.94σ · OU 12d (sweet spot) · clean (ADF+KPSS+VR)"     │
│                                                                          │
│ ▸ Expand full analysis (prices · charts · narrative · factors)           │
└──────────────────────────────────────────────────────────────────────────┘
```

Clicking "▸ Expand" reveals the full drilldown:

```
┌──────────────────────────────────────────────────────────────────────────┐
│ 💵 ACTUAL SR3 CONTRACT PRICES — SRAM27                                    │
│   ACTION · ENTRY PRICE · TARGET PRICE · STOP PRICE · P&L/c · R:R         │
│   BUY    · 96.3400     · 96.3562      · 96.2996    · $40   · 0.40:1     │
│   YIELD-SPACE: entry 366.00 bp · target 364.38 bp · stop 370.04 bp       │
│                                                                          │
│ 💹 Price + FV   📈 Residual + OU band   🎯 Conviction breakdown          │
│ ┌─────────────┐ ┌─────────────────────┐ ┌────────────────────────────┐  │
│ │ Price line  │ │ Residual series + │ │ Horizontal bar chart of    │  │
│ │ + ENTRY/    │ │ ±1σ / ±2σ bands +  │ │ all 17 conviction          │  │
│ │ TARGET/STOP │ │ entry/target/stop  │ │ components sorted by       │  │
│ │ overlays    │ │ residual lines     │ │ contribution               │  │
│ └─────────────┘ └─────────────────────┘ └────────────────────────────┘  │
│                                                                          │
│ 🧠 HOW THE ENGINE ARRIVED AT THIS TRADE                                   │
│                                                                          │
│ ① WHAT WE OBSERVED — Today's SRAM27 closed with an implied yield...      │
│ ② WHY IT'S TRADEABLE — OU half-life 12.3d ... triple-gate CLEAN ...      │
│ ③ WHY THIS DIRECTION — residual +1.94 σ ... BUY the futures              │
│ ④ THE TRADE — Execute BUY ... at 96.3400 ... target 96.3562 ... +$40    │
│   ┌─ DERIVATION ──────────────────────────────────────────────────┐      │
│   │ entry_yield  = (100 − 96.3400) × 100 = 366.00 bp              │      │
│   │ residual     = entry − PCA_FV = +1.62 bp                      │      │
│   │ target_yield = 366 − 1.62 = 364.38 bp                         │      │
│   │ target_price = 100 − 364.38/100 = 96.3562 (tick 0.0025)       │      │
│   │ stop_yield   = 366 + 4.05 = 370.04 bp                         │      │
│   │ stop_price   = 96.2996                                         │      │
│   │ P&L/c        = $40, risk/c = $100                              │      │
│   └────────────────────────────────────────────────────────────────┘      │
│   Conviction = 0.616 from these top contributors:                       │
│     adf_pass        +0.080                                              │
│     z_significance  +0.076                                              │
│     ...                                                                 │
│                                                                          │
│ ⑤ WHAT KILLS IT — caveats from per-factor + convexity + slippage         │
│                                                                          │
│ 📊 PER-FACTOR ANALYSIS                                                   │
│   ✓ Residual z-score = +1.94σ  (+0.084)                                  │
│   ✓ ADF rejects unit root @ 5%  (+0.080)                                 │
│   ✓ Triple-Stationarity Gate: CLEAN  (+0.000)                            │
│   ✓ Sample size = 660  (+0.060)                                          │
│   ...                                                                   │
└──────────────────────────────────────────────────────────────────────────┘
```

For multi-leg trades, the price block is replaced with the LISTED / SYNTHETIC / PER-LEG block (see §20).

State per-card is keyed by `idea.idea_id` so each can be independently expanded.

---

<a name="34-filter-strip"></a>
## 34. Filter strip + grouping

### Top filter strip (above the trade list)

**Row 1 — 📂 Trade type filter** (multi-select chips):
- 📊 Outrights
- 📐 Spreads
- 🦋 Flies
- 📦 Packs
- 🎯 Baskets

Deselect any type to hide all its cards.

**Row 2 — secondary filters**:
- 🔍 search (by contract symbol)
- sort by (conviction / |z| / expected $ / half-life / expected hold)
- gates (clean / low_n / non_stationary / regime_unstable / circular_proxy)
- min conviction (slider, 0.0..1.0, default 0.40)
- top N per group (default 5 — each group shows top-N independently)

### Group headers

Cards are grouped by `structure_type` with section headers:
- 📊 Outrights — single-contract residual fades
- 📐 Spreads — slope / calendar relative value
- 🦋 Flies — curvature / butterfly trades
- 📦 Packs — 4-contract bundle RV
- 🎯 Baskets — PCA-isolated factor exposures

Within each group, cards are sorted by conviction (descending). The top-N filter applies per group, so the user sees up to N of each type rather than N total dominated by one type.

---

<a name="35-pill-strips"></a>
## 35. Engine Health + TAPE pill strips

Two compact strips at the top of the page show real-time engine + tape state.

### 🔧 ENGINE strip

| Chip | Meaning | Threshold |
|---|---|---|
| `3-PC explained: N%` | today's reconstruction_pct (variance share) | green ≥ 95%, amber 90-95%, red < 90% |
| `cross-PC corr: X.XX` | max abs(cross-PC correlation) today | green ≤ 0.30, amber 0.30-0.50, red > 0.50 |
| `regime: N% conf` | HMM dominant confidence today | green ≥ 60% |
| `print: clean / ⚠ flagged` | print quality alert today | green/red |
| `trades: N` | total active trades |  |
| `clean: N` | trades with gate_quality = "clean" |  |
| `engine signals: N` | count of engine_state_signals overlays |  |
| `refit: Nd old` | days since latest rolling fit | green ≤ 5d, amber > 5d |

### 🌐 TAPE strip

| Chip | Meaning |
|---|---|
| `vol normal/stressed/crisis (z=X.XX)` | composite vol regime |
| `risk_on / risk_off / panic (z=X.XX)` | composite risk state |
| `recession prob 4w: N%` | from credit cycle module |
| `⚠ IG/HY diverging` | (when flag fires) |
| `SPX ±X.X%/5d` | equity 5-day change |
| `MOVE NN` | MOVE bond-option vol index |
| `DXY NN.N` | synthetic DXY level |

---

<a name="36-glossary"></a>
## 36. Concept tooltips + glossary

`lib/pca_concepts.py:CONCEPTS` — single dict with 26 entries. Each entry:

```python
{
    "name":              "Ornstein-Uhlenbeck Half-Life",
    "category":          "statistic",
    "one_liner":         "Time it takes for a stretched residual to decay halfway back to its mean.",
    "what_it_measures":  "<paragraph>",
    "math":              "Δx_t = κ(μ − x_{t-1}) + σ·ε_t   →   HL = ln(2) / κ",
    "interpretation":    "<thresholds with what each means>",
    "for_trade":         "<why a trader cares>",
    "source":            "Ornstein & Uhlenbeck (1930); fitting via Vasicek (1977) discretization",
    "display_units":     "days",
}
```

Surfaces in the UI:
- **Section headers** in the dossier have a ⓘ icon → hover tooltip with the one-liner + math + interpretation
- **Per-factor analysis** chips have a nested "↳ What is X?" expander → full disclosure HTML

Helpers:
- `concept_tooltip_html(key)` → compact hover tooltip
- `concept_disclosure_html(key)` → full HTML for expanders

---

<a name="37-research-validation"></a>
## 37. Research validation against standard methods

Methodology: an independent reviewer agent read every quant module and compared the implementation to the canonical published methods. Results:

| # | Method | Source | Verdict |
|---|---|---|---|
| 1 | Litterman-Scheinkman PCA | L&S 1991 J. Fixed Income | ✅ Matches (covariance basis, sign-normalized) |
| 2 | ACM term premia decomp | Adrian-Crump-Moench 2013 | ✅ Matches (subtraction identity, NY Fed published series) |
| 3 | Ledoit-Wolf shrinkage | L&W 2004 | ✅ Matches (correct π̂/γ̂/n estimator) |
| 4 | OU half-life | Vasicek 1977 discretization | ✅ Matches (exact `−ln 2 / ln b` form) |
| 5 | ADF / KPSS / Variance-Ratio | Said-Dickey 1984 / KPSS 1992 / Lo-MacKinlay 1988 | ✅ Matches (correct critical values, hetero-robust z) |
| 6 | Hurst R/S | Hurst 1951 | ✅ Matches |
| 7 | Mahalanobis k-NN | Mahalanobis 1936 | ⚠️ Soft-weighted (justified — smoother FV) |
| 8 | HMM Baum-Welch | Rabiner 1989 | ⚠️ Emissions-reconstruction may bias A-matrix |
| 9 | **Heitfield-Park step path** | H&P 2014 | ❌ **Kernel σ 25 bp too wide, bracket 0.5m too wide → FIXED** |
| 10 | **Mercurio/Piterbarg convexity** | Mercurio 2018 / Piterbarg 2006 | ❌ **PC1-σ used as rate vol, Ho-Lee form mislabeled → FIXED** |

---

<a name="38-bugs-fixed"></a>
## 38. Bugs found and fixed

### Critical (changed engine output)

| # | File | Bug | Fix |
|---|---|---|---|
| 1 | `lib/pca_cross_asset_analysis.py:381` | Convexity bias used PC1-score σ (unitless) as if it were short-rate σ; formula `−0.5·σ²·τ²` mislabeled as Mercurio (actually Ho-Lee). | Use front-SR3 yield daily σ in decimal; Piterbarg short-rate form `−0.5·σ_r²·T·τ_accrual`. |
| 2 | `lib/pca_step_path.py:57` | Kernel σ = 25 bp smeared ~50% mass into adjacent buckets at center. Bracket window `eps_m = 0.5 months` understated step magnitude. | Kernel σ → 12.5 bp. Bracket → 0.15 months (~4 business days). |
| 3 | `lib/pca_trade_interpretation.py:_render_trade_chart_inline` | "PCA-implied FV" time-series in price chart was reconstructed from change-space PCA, producing flat near-100. | Drop the misleading FV line. Today's FV = `target_price` (dashed horizontal annotation only). |
| 4 | `lib/pca_trade_interpretation.py:convert_to_sr3_prices` (multi-leg) | Listed CME spread/fly prices treated as raw price units when they're actually in **bp of price-differential** (×100 of raw). P&L was off by 100×. | Recognized units: 1 listed unit = 1 bp = $25 / contract. Tick = 0.25 bp. P&L formula scaled correctly. |
| 5 | `lib/pca_trade_interpretation.py:convert_to_sr3_prices` (multi-leg) | Sign-only weights (1, −1, 1) for flies gave 25%-too-small synthetic move vs canonical (1, −2, 1). | Use canonical CME weights for spread (1, −1) and fly (1, −2, 1). Listed FV = 0 (since Σcw = 0). Target = 0, stop = 3.5×entry. |
| 6 | `lib/pca_trade_interpretation.py:convert_to_sr3_prices` (multi-leg) | Trades not matching a listed product showed "see card" placeholders. | Three-mode price-source classifier: LISTED / SYNTHETIC / PER_LEG. Every multi-leg trade now gets actionable numbers. |

### Sign convention clarification

`TradeLeg.side` in `lib/pca_trades.py` is in YIELD space (`side = "sell"` when `residual_z > 0` means "sell the rich yield"). For an outright SR3 contract, the actual futures order is the OPPOSITE. `convert_to_sr3_prices` emits an explicit `contract_action` field that's always in CONTRACT space — fixed labeling end-to-end.

### Other observations (not yet refactored)

- HMM Baum-Welch reconstructs emissions from posteriors instead of storing X — fine for posterior γ but introduces a per-t shift in transition M-step ξ. Worth refactoring to store X in `GMMFit`.
- Soft-weighted k-NN deviates from textbook (intentional, justified for smoother FV).

---

<a name="39-module-reference"></a>
## 39. Module reference

### Core engine

| File | Lines | Purpose |
|---|---:|---|
| `lib/pca.py` | ~2,100 | PCA fit + CMC panel + structure enumeration + per-instrument residuals |
| `lib/pca_trades.py` | ~2,050 | 19 trade generators + 13 engine-state generators + clustering + conviction scoring |
| `lib/pca_step_path.py` | ~250 | Heitfield-Park step-path bootstrap |
| `lib/pca_analogs.py` | ~400 | Mahalanobis k-NN + Ledoit-Wolf + path-bucket assignment |
| `lib/pca_regimes.py` | ~350 | GMM + HMM fitting |
| `lib/pca_turn_adjuster.py` | ~150 | Turn-of-year/quarter/month adjustments |
| `lib/mean_reversion.py` | ~500 | ADF, KPSS, VR, Hurst, OU half-life, composite reversion score, triple-stationarity gate |
| `lib/contract_units.py` | ~200 | Contract leg parsing |
| `lib/fomc.py` | ~200 | FOMC calendar |
| `lib/sra_data.py` | ~500 | DuckDB loaders + listed spread/fly symbol derivation |
| `lib/connections.py` | ~150 | Bloomberg parquet loaders |

### Cross-asset

| File | Lines | Purpose |
|---|---:|---|
| `lib/pca_cross_asset.py` | ~700 | Loaders for vol/equity/FX/credit/commodities/UST/Fed BS/auctions/ACM |
| `lib/pca_cross_asset_analysis.py` | ~500 | 8 cross-asset analyses (vol regime, risk state, credit cycle, lead-lag, eq-rates corr, FX diff, term premia, convexity) |

### Interpretation + UI

| File | Lines | Purpose |
|---|---:|---|
| `lib/pca_dossier.py` | ~1,700 | Per-contract 18-section dossier data |
| `lib/pca_trade_interpretation.py` | ~1,100 | SR3 price conversion + factor interpretation + dynamic narrative + chart data + cross-check |
| `lib/pca_concepts.py` | ~450 | Concept glossary (26 entries with math + plain-English + citations) |
| `tabs/us/sra_pca.py` | ~2,700 | Trade Screener tab — filter strip, card layout, charts, drilldown |

### Documentation

| File | Lines | Purpose |
|---|---:|---|
| `PCA_TAB_DOCUMENTATION.md` | (this file) | Canonical reference |

---

<a name="40-data-structures"></a>
## 40. Key data structures

```python
# lib/pca.py
@dataclass(frozen=True)
class PCAFit:
    asof: date
    tenors_months: list                 # e.g. [1, 2, 3, ..., 60]
    loadings: np.ndarray                # shape (n_pc, n_tenors)
    eigenvalues: np.ndarray
    variance_ratio: np.ndarray
    feature_mean: np.ndarray            # mean Δyield per tenor (bp/day)
    n_obs: int
    fit_window: tuple                   # (start_date, end_date)

@dataclass(frozen=True)
class StructureCandidate:
    target_pc: int                      # 1, 2, or 3
    symbols: list
    weights: np.ndarray                 # PV01 weights
    pv01_sum: float
    residual_today_bp: Optional[float]
    residual_z: Optional[float]
    half_life_d: Optional[float]
    adf_pass: bool
    composite_score: Optional[float]
    eff_n: int
    gate_quality: str                   # "clean" / "low_n" / "non_stationary" / "drift" / "random_walk" / "regime_unstable"

# lib/pca_trades.py
@dataclass(frozen=True)
class TradeLeg:
    symbol: str
    weight_pv01: float                  # signed PV01 weight
    side: str                           # "buy" / "sell" — YIELD space
    contracts: int = 0

@dataclass(frozen=True)
class TradeIdea:
    idea_id: str
    leg_fingerprint: str
    primary_source: str                 # e.g. "PC3-fly"
    sources: tuple                      # all confirming generator names after clustering
    source_id: int
    direction: str                      # "long" / "short" — YIELD space
    structure_type: str                 # "outright" / "spread" / "fly" / "pack" / "basket"
    legs: tuple                         # tuple of TradeLeg
    pv01_sum: float
    entry_bp: Optional[float]           # residual_today_bp
    fair_value_bp: Optional[float]
    target_bp: Optional[float]
    stop_bp: Optional[float]
    expected_revert_d: Optional[float]  # 1.5 × half_life
    expected_pnl_bp: Optional[float]
    expected_pnl_dollar: Optional[float]
    z_score: Optional[float]
    half_life_d: Optional[float]
    adf_pass: bool
    eff_n: int
    gate_quality: str
    risk_flags: tuple
    rationale_html: str
    headline_html: str
    cycle_phase: Optional[str]
    cycle_alignment: str                # "favoured" / "neutral" / "counter"
    analog_fv_bp: Optional[float]
    analog_fv_z: Optional[float]
    analog_fv_eff_n: Optional[int]
    path_fv_bp: Optional[float]
    path_fv_z: Optional[float]
    conviction: float = 0.0
    conviction_breakdown: tuple = ()
    state: str = "NEW"                  # NEW / MATURING / PEAK / FADING
    days_alive: int = 0
    max_conviction_to_date: float = 0.0
    slippage_estimate_bp: Optional[float] = None
    convexity_warning: bool = False
    factor_exposure: tuple = ()
    what_if_table: tuple = ()
    n_confirming_sources: int = 1

# lib/pca_trade_interpretation.py
@dataclass(frozen=True)
class SR3Prices:
    is_outright: bool
    contract_action: Optional[str]      # "BUY" / "SELL" — CONTRACT space
    entry_price: Optional[float]        # SR3 quote, tick-rounded
    target_price: Optional[float]
    stop_price: Optional[float]
    pnl_per_contract_dollar: Optional[float]
    risk_per_contract_dollar: Optional[float]
    risk_reward: Optional[float]
    per_leg_prices: tuple               # tuple of per-leg dicts
    net_entry_bp: Optional[float]       # PV01-weighted net residual (per-leg mode)
    net_target_bp: Optional[float]
    net_stop_bp: Optional[float]
    entry_yield_bp: Optional[float]     # outright only
    target_yield_bp: Optional[float]
    stop_yield_bp: Optional[float]
    # Synthetic spread/fly canonical reconstruction
    synth_canonical_bp: Optional[float] = None    # Σ cw × leg_price × 100
    synth_target_bp: Optional[float] = None        # = 0 (FV by canonical construction)
    synth_stop_bp: Optional[float] = None
    synth_pnl_dollar: Optional[float] = None
    synth_risk_dollar: Optional[float] = None
    synth_risk_reward: Optional[float] = None
    # LISTED CME contract data
    listed_symbol: Optional[str] = None            # e.g. "SRAH26-M26-U26"
    listed_entry_price: Optional[float] = None     # in bp of price-diff
    listed_target_price: Optional[float] = None    # = 0 (FV)
    listed_stop_price: Optional[float] = None      # = 3.5 × entry
    listed_pnl_dollar: Optional[float] = None
    listed_risk_dollar: Optional[float] = None
    listed_risk_reward: Optional[float] = None
    listed_vs_synth_diff_bp: Optional[float] = None
    # Provenance
    price_source: str = "none"          # "listed" / "synthetic" / "per_leg" / "none"
    explanation: str = ""

@dataclass(frozen=True)
class FactorInterpretation:
    key: str                            # links to pca_concepts.CONCEPTS
    value_display: str
    tier: str                           # "supportive" / "neutral" / "caveat"
    headline: str
    detail: str
    weight_in_conviction: float
```

---

<a name="41-quick-ref"></a>
## 41. Quick-reference index — where to find ...

| Question | Where |
|---|---|
| The 3-PC PCA fit | `lib/pca.py:499` (`fit_pca_static`) |
| The 19 trade generators registry | `lib/pca_trades.py:1864` (`TRADE_GENERATORS`) |
| The 13 engine-state generators | `lib/pca_trades.py:1876` (`ENGINE_STATE_GENERATORS`) |
| The conviction-score formula (17 inputs) | `lib/pca_trades.py:1672` (`score_conviction`) |
| The triple-stationarity gate | `lib/mean_reversion.py:triple_stationarity_gate` |
| ADF / KPSS / VR / Hurst / OU implementation | `lib/mean_reversion.py` |
| Outright SR3 contract-price conversion | `lib/pca_trade_interpretation.py:131` (`convert_to_sr3_prices`, outright branch) |
| Multi-leg price conversion (3 modes) | `lib/pca_trade_interpretation.py:240+` (multi-leg branch) |
| Listed CME symbol derivation | `lib/sra_data.py:derive_listed_spread_symbol` / `derive_listed_fly_symbol` |
| Listed spread/fly close panel loader | `lib/sra_data.py:load_listed_spread_fly_panel` |
| Dynamic narrative builder | `lib/pca_trade_interpretation.py:781` (`build_dynamic_narrative`) |
| Per-factor interpretation | `lib/pca_trade_interpretation.py:688` (`interpret_factors`) |
| Mahalanobis k-NN | `lib/pca_analogs.py:120` (`knn_analog_search`) |
| Heitfield-Park step path | `lib/pca_step_path.py:147` (`fit_step_path_bootstrap`) |
| Convexity bias | `lib/pca_cross_asset_analysis.py:381` (`compute_convexity_bias`) |
| Cross-asset overlays | `lib/pca_cross_asset_analysis.py:run_all_cross_asset_analyses` |
| GMM + HMM regime fit | `lib/pca_regimes.py:fit_regime_stack` |
| Concept glossary dict | `lib/pca_concepts.py:CONCEPTS` |
| The Trade Card renderer | `tabs/us/sra_pca.py:_render_trade_card` |
| The 3-chart panel | `tabs/us/sra_pca.py:_render_trade_drilldown` |
| Filter strip + grouping | `tabs/us/sra_pca.py:_render_filter_strip_and_filter` + `_render_trade_table` |
| Engine Health pill strip | `tabs/us/sra_pca.py:_render_engine_health_strip` |
| TAPE pill strip | `tabs/us/sra_pca.py:_render_cross_asset_strip` |

---

## Workflow — how to verify any trade end-to-end

When you see a recommendation and want to audit the math:

1. **Open the card**: click `▸ Expand full analysis`.
2. **Read paragraph ④ "The trade"** — the derivation monospace block shows the exact entry_yield → residual → target_yield → target_price chain. Verify each line manually if desired.
3. **Check the cross-check chip** (for spreads/flies):
   - 🟢 `✓ matches legs` → engine, DB, and canonical math agree end-to-end
   - 🟡 `⚠ mild Δ` → sub-bp drift, OK for trading
   - 🔴 `⚠ verify` → leg pattern may not map to that listed product, do NOT trade
4. **Check the conviction breakdown** — see which inputs contributed. If conviction comes mostly from `cross_confirm` (multiple sources agreeing) + `adf_pass` (statistical gate), the trade has high methodological backing.
5. **Check paragraph ⑤ "What kills it"** — any caveats listed. Trade smaller if any are present.
6. **Look at the residual chart (middle of 3)** — is the residual currently outside ±1σ bands? Is it stretched (good for fade) or trending (bad)?
7. **Look at the conviction-breakdown bar chart (right)** — verify the dominant components are statistical (z, ADF, eff_n) not just lifecycle/cross-confirm.

This 7-step workflow lets you audit any recommendation without re-running the engine.

---

<a name="42-mode-system"></a>
## 42. Trade Horizon Mode System (positional pivot)

Added 2026-05-11. Every horizon-sensitive parameter is now keyed off a single `mode` selector — `"intraday" | "swing" | "positional"`. The default is **`"positional"`** because the user is a positional STIRS trader.

### What mode controls

| Parameter | Intraday | Swing | **Positional** |
|---|---|---|---|
| `residual_lookback` (z-score window) | 30d | 60d | **252d** |
| `triple_gate_lookback` | 60d | 90d | **120d** |
| `history_days` (PCA history) | 250 | 700 | **1000** (≈4y) |
| `z_threshold` (generator filter) | 1.0σ | 1.5σ | **2.0σ** |
| `hold_mult` × HL | 1.0× | 1.5× | **1.5×** |
| `hold_floor` (min hold days) | 1 | 3 | **21** |
| `hold_cap` (max hold days) | 14 | 30 | **120** |
| `min_eff_n` | 30 | 50 | **80** |
| OU sweet spot (full credit) | [0.5, 7]d | [3, 30]d | **[15, 90]d** |
| OU sweet spot (partial mid) | (7, 15]d | (30, 60]d | **(90, 180]d** |
| HL extension stop multiplier | 2.0× | 3.0× | **3.0×** |
| Detrend window | 90d | 252d | **252d** |

### Single source of truth

```python
# lib/pca.py
MODE_PARAMS = {
    "intraday": {...},
    "swing": {...},
    "positional": {...},  # see table above
}
DEFAULT_MODE = "positional"

def mode_params(mode: Optional[str]) -> dict:
    """Return parameter dict for `mode`, falling back to DEFAULT_MODE."""
```

Every horizon-sensitive function (residual computation, generators, conviction
scoring, exit logic, narrative) reads from `mode_params(panel.get("mode"))`.

### Where mode flows

1. UI `tabs/us/sra_pca.py` → radio selector → `st.session_state.pca_horizon_mode`
2. `_build_engine_panel(mode=...)` → `build_full_pca_panel(mode=...)`
3. `_build_pca_panel_internal(mode=...)` writes `out["mode"]` and `out["mode_params"]`
4. Per-residual functions read mode-derived `residual_lookback` + `triple_gate_lookback`
5. `score_structure`, all 19 generators, `score_conviction` read from `panel["mode"]`
6. Exit-rule evaluation (`lib/trade_exits.py`) and backtest (`lib/pca_backtest.py`)
   also read mode from panel

### Weekly resampling option

Positional mode supports an optional `resample="W"` flag (UI checkbox "Weekly
smoothing") that resamples the level-residual to weekly (Friday close) before
computing z / HL / triple-gate. Reduces noise for positional decision cadence.

---

<a name="43-dynamic-exits"></a>
## 43. Dynamic rule-based target/stop tiers

Added 2026-05-11. Static entry-time target/stop replaced with a daily re-evaluated
4-target + 6-stop tier system in [lib/trade_exits.py](lib/trade_exits.py).

### Why dynamic

Static stops set at fill are wrong because:
- Market regime can change (HMM transitions) — stop should re-evaluate
- Residual half-life can extend or contract — time stop should adapt
- Cross-confirmation count can drop — signal degradation should trigger early exit
- New data should update FV target, not freeze at entry

### Target tiers (priority order — first hit wins)

| Tier | Condition | Action |
|---|---|---|
| **T1** Full revert | `|current_z| ≤ 0.5σ` AND magnitude reduced from entry | Close 100% |
| **T2** Partial revert | `|current_z| ≤ entry_|z| × 0.67` | Close 33%, hold rest for T1 |
| **T3** Time target | `days_held ≥ hold_mult × current_HL` | Close 100% (HL re-fit daily) |
| **T4** Signal decay | `n_confirming < entry × 0.5` AND held ≥ 5d | Close 50% |

### Stop tiers (priority order — all close 100%)

| Tier | Condition | Severity |
|---|---|---|
| **S1** Adverse breakout | `|current_z| > max(3.5, entry_|z| × 1.5)` in adverse direction | Hard |
| **S2** Triple-gate fail | `triple_gate.all_three == False` after entry | Hard |
| **S3** HL extension | `current_HL > hl_ext_mult × entry_HL` or `> hold_cap` | Soft |
| **S4** Convexity warning | Piterbarg bias fires after 5d held | Soft |
| **S5** Regime transition | HMM regime shifts AND new regime hostile to structure | Soft |
| **S6** Hard P&L stop | `realized_PnL ≤ -1.5 × expected_PnL` | Hard |

### Key API

```python
from lib.trade_exits import (EntryState, evaluate_dynamic_exit,
                                 entry_state_from_idea, planned_levels)

es = entry_state_from_idea(idea, entry_date)
levels = planned_levels(idea, es, mode="positional")  # for UI display

# Each day after entry:
exit_state = evaluate_dynamic_exit(
    idea, es, current_panel, days_held=N, mode="positional",
    current_pnl_bp=...
)
if exit_state.exit_now:
    # exit_state.target_tier or exit_state.stop_tier names the trigger
    close_position(fraction=exit_state.exit_size_frac)
```

The trade card UI shows the full Exit Plan with all 10 tier levels and their
target dates. See `_render_positional_outlook_and_exit_plan` in
[tabs/us/sra_pca.py](tabs/us/sra_pca.py).

---

<a name="44-backtest"></a>
## 44. Backtest framework

Added 2026-05-11. Located at [lib/pca_backtest.py](lib/pca_backtest.py).

### Purpose

Replay the engine through history. For each emitted idea above conviction
threshold, simulate the trade with dynamic exits (Section 43) and realistic
slippage/commission costs. Aggregate per-source / per-conviction-bucket
performance into hit-rate tables that feed back into `score_conviction`'s
`empirical_hit_rate` input (Phase 5 wiring).

### Cost model

| Component | Value | Per |
|---|---|---|
| Slippage (listed contracts) | 0.25 bp × 1.5 (proxy) × 2 sides × n_legs | Round-trip |
| Commission | $1.50 × 2 sides × n_legs | Round-trip |
| Spread proxy | Half-tick outright, full-tick spread, 1.5× fly | One side |

### Two execution modes

- **Fast mode** (`walk_forward=False`): single full-history PCA fit at panel build,
  walks decision days with that fixed PCA. Fast but technically forward-looking.
- **Walk-forward** (`walk_forward=True`): re-fits PCA at each decision date
  (with `walk_step_days` cadence). Unbiased — the only correct way to compute
  empirical Sharpe. ~6× slower.

### Empirical hit rates → score_conviction

After each backtest run, results are persisted to
`D:\STIRS_DASHBOARD\cache\backtest_empirical_hit_rates_{mode}.json` keyed by
source name and horizon bucket. On next engine refresh, `_build_engine_panel`
loads them into `panel["empirical_hit_rates"]`, where `score_conviction`'s
`_empirical_hit_rate_score` picks the closest-horizon hit rate for each trade's
source and weights it at 0.06 (up from 0.04 placeholder).

### Key API

```python
from lib.pca_backtest import run_engine_backtest, build_empirical_hit_rates

result = run_engine_backtest(
    start_date=date(2024, 1, 1),
    end_date=date(2026, 5, 11),
    mode="positional",
    min_conviction=0.50,
    walk_forward=False,
    walk_step_days=5,
    max_trades_per_day=3,
)
ehr = build_empirical_hit_rates(result)
# ehr is the dict to write to cache
```

`BacktestResult` fields: `trades_df`, `equity_curve`, `by_source`,
`by_conviction_bucket`, `summary`. See dataclass for full schema.

### UI

`_render_backtest_section` in [tabs/us/sra_pca.py](tabs/us/sra_pca.py) renders
an expander at the bottom of the PCA tab with: date range / mode / conviction /
walk-forward controls + Run button. After run: summary card, equity curve,
per-source table, conviction-bucket calibration table. One-click "Cache empirical
hit rates" persists them for score_conviction.

---

<a name="45-phase456"></a>
## 45. Phase 4-6 changelog

### Phase 4 — Conviction recalibration

Rebalanced 17 inputs → 19 inputs with explicit weights summing to 1.0. Changes
in [lib/pca_trades.py:1672](lib/pca_trades.py:1672) `score_conviction`:

| Component | Old weight | New weight | Notes |
|---|---|---|---|
| z_significance | 0.13 | 0.12 | Slight reduction |
| ou_sweet_spot | 0.09 | 0.10 | Mode-aware band (was hardcoded [3,30]) |
| adf_pass | 0.08 | 0.08 (renamed to **triple_gate_pass**) | Reads `idea.triple_gate_pass` |
| eff_n | 0.06 | 0.05 |  |
| model_fit | 0.06 | 0.05 |  |
| cycle_alignment | 0.07 | 0.07 | unchanged |
| regime_stable | 0.05 | 0.05 | unchanged |
| cross_pc_orth | 0.05 | 0.04 |  |
| variance_regime | 0.05 | 0.04 |  |
| analog_fv_agree | 0.05 | 0.05 | unchanged |
| path_fv_agree | 0.04 | 0.04 | unchanged |
| **seasonality** | 0.02 (placeholder) | **0.03 (activated)** | Reads cycle phase × direction |
| **event_drift** | 0.015 (placeholder) | **0.03 (activated)** | Reads days-to-next-FOMC vs hold-window |
| **empirical_hit_rate** | 0.04 (ledger only) | **0.06 (backtest-fed)** | Reads `panel["empirical_hit_rates"]` |
| lifecycle | 0.04 | 0.04 | unchanged |
| cross_confirm | 0.05 | 0.05 | unchanged |
| slippage_ok | 0.03 | 0.03 | unchanged |
| convexity_safe | 0.03 | 0.03 | unchanged |
| **exit_clarity** | — | **0.04 (NEW)** | Credit when target/stop separation is balanced |
| **Total** | 0.97 | **1.00** | Now sums exactly to 1.0 |

### Phase 5 — Historical mean-reversion cross-validation

Wired via the new backtest framework (Section 44). `score_conviction`'s
`empirical_hit_rate` input now reads from `panel["empirical_hit_rates"]` which
is populated from cached backtest results. Falls back to live ledger
`compute_track_record` when no cache is present.

### Phase 6 — Narrative final pass

Updated [lib/pca_trade_interpretation.py:1156](lib/pca_trade_interpretation.py:1156) `build_dynamic_narrative`:

- "Today's residual" → "Current cumulative deviation from PCA fair value"
- "60-day distribution" → "{mode-derived lookback}-day distribution" (dynamic)
- OU half-life sentence reframed: "expected mean-revert over X trading days;
  recommended hold window for {mode} mode: max(floor, min(X, cap)) days"
- Added explicit mode label in paragraph 1 of every narrative
- UI: "today's residual" → "current deviation" throughout sra_pca.py

---

<a name="46-mode-glossary"></a>
## 46. Quick reference — positional trader's workflow

The intended workflow for a positional STIRS trader:

1. **Default mode is Positional** (15-90d HL sweet spot, 21-120d hold)
2. **Filter trades** by `min conviction ≥ 0.55` and `min hold days ≥ 21d`
3. **Read the Positional Outlook card** for each candidate:
   - Hold days vs cap
   - Regime confidence (≥60% preferred)
   - FOMC events in window (zero or one is fine; two+ is event-cluster risk)
   - Convexity status
4. **Read the Exit Plan card** to know your T1/T2/T3 levels and S1-S6 stops
5. **Size by S1 distance** — `n_contracts = portfolio_risk_$ / S1_distance_bp_$`
6. **Periodically run the backtest** at bottom of tab to refresh empirical hit
   rates; check Sharpe per source — if a generator's Sharpe is < 0.2, lower
   weight on its emissions
7. **Don't trade if Exit Plan S3 threshold (HL extension) is already close** —
   means the model is fragile to small HL changes

---

*This document is the canonical reference for the PCA tab as of 2026-05-11. For the actual code, always defer to the source files — this doc captures intent and design choices, but implementation details may evolve.*
