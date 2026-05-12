"""Plain-English glossary of every concept used in the PCA Trade Screener.

Each entry is a dict with:
  - `name`            : short human label
  - `category`        : "factor" | "test" | "statistic" | "model" | "framework"
  - `one_liner`       : 1-sentence summary (for tooltips)
  - `what_it_measures`: 1-paragraph plain-English explanation
  - `math`            : the formula in LaTeX-friendly plaintext
  - `interpretation`  : how to read the number ("z > 2 means ...", "HL < 5d means ...")
  - `for_trade`       : why a trader cares
  - `source`          : academic citation (year, authors, journal/working paper)
  - `display_units`   : "bp" | "%" | "ratio" | "days" | "score" | "dimensionless"

This module is the single source of truth — both the renderer (for tooltips
and disclosure boxes) and the per-trade interpretation engine read from
CONCEPTS dict. To add a new concept, add it once here.
"""
from __future__ import annotations

CONCEPTS: dict = {
    # =========================================================================
    # PCA / factor concepts
    # =========================================================================
    "PC1": {
        "name": "Principal Component 1 (Level)",
        "category": "factor",
        "one_liner": "The single direction that explains the most variance across the yield curve — typically a parallel shift up/down.",
        "what_it_measures": (
            "PC1 is the first eigenvector of the covariance matrix of daily "
            "yield changes across the SR3 strip. Loadings are usually all the same "
            "sign and roughly equal — so a +1σ move in PC1 means every tenor moves "
            "up by about the same amount. PC1 typically explains 80-90% of total "
            "variance and corresponds to the Fed-policy / front-end-rate direction."
        ),
        "math": "PC1 = first eigenvector of Cov(ΔY), where Y is the yield panel",
        "interpretation": (
            "PC1 level today (in σ-units) tells you where the curve sits in its "
            "PC1 distribution. PC1 > +1σ means rates are stretched high; "
            "PC1 < -1σ means rates are stretched low."
        ),
        "for_trade": (
            "A PC1 trade is a directional bet — buying or selling rates. "
            "PC1-isolated baskets give you pure directional exposure with no "
            "slope or curvature drift."
        ),
        "source": "Litterman & Scheinkman (1991), 'Common Factors Affecting Bond Returns', J. Fixed Income",
        "display_units": "σ",
    },
    "PC2": {
        "name": "Principal Component 2 (Slope)",
        "category": "factor",
        "one_liner": "Steepening/flattening of the curve — front-end moves opposite to back-end.",
        "what_it_measures": (
            "PC2 typically has positive loadings on the front contracts and "
            "negative on the back (or vice-versa). A +1σ PC2 move corresponds "
            "to a steepening (front yields rise, back yields fall) or flattening "
            "depending on sign convention. PC2 explains ~5-10% of variance."
        ),
        "math": "PC2 = second eigenvector of Cov(ΔY); orthogonal to PC1",
        "interpretation": (
            "PC2 = +1σ means the curve is more steeply sloped than usual; "
            "PC2 = -1σ means flatter than usual."
        ),
        "for_trade": (
            "A PC2 trade is a steepener or flattener. PC2-isolated spreads "
            "(weights solved to zero PC1 and PC3) give you pure slope exposure "
            "with no level or curvature drift."
        ),
        "source": "Litterman & Scheinkman (1991)",
        "display_units": "σ",
    },
    "PC3": {
        "name": "Principal Component 3 (Curvature)",
        "category": "factor",
        "one_liner": "Belly vs wings — middle of the curve moves opposite to front and back.",
        "what_it_measures": (
            "PC3 has loadings shaped like a wave — positive at the wings (front and "
            "back) and negative in the belly, or vice-versa. A +1σ PC3 move "
            "corresponds to the belly cheapening relative to the wings. PC3 "
            "explains ~1-3% of variance."
        ),
        "math": "PC3 = third eigenvector of Cov(ΔY); orthogonal to PC1, PC2",
        "interpretation": (
            "PC3 = +1σ means the belly is cheap (high yield) vs wings; "
            "PC3 = -1σ means belly is rich (low yield) vs wings."
        ),
        "for_trade": (
            "A PC3 trade is a butterfly (-1 +2 -1 or +1 -2 +1). PC3-isolated "
            "flies give you pure curvature exposure — buying the rich/cheap belly "
            "with the wings paid for by the inverse position."
        ),
        "source": "Litterman & Scheinkman (1991)",
        "display_units": "σ",
    },
    "residual": {
        "name": "Residual",
        "category": "statistic",
        "one_liner": "How far an instrument's yield deviates from its PCA-implied fair value, in basis points.",
        "what_it_measures": (
            "Given a PCA fit with 3 PCs, every instrument has a model-implied "
            "yield = mean + PC1·loading₁ + PC2·loading₂ + PC3·loading₃. The "
            "residual is the actual yield minus this model implied yield. "
            "A positive residual means the instrument is yielding higher than "
            "the model says it should (= rich rate = cheap price)."
        ),
        "math": "residual_bp = y_actual - (mean + Σ_k PC_k · load_k)",
        "interpretation": (
            "+5 bp residual means the contract's yield is 5 bp above where "
            "the curve says it should sit. Mean reversion implies it will "
            "drift back to 0 over time."
        ),
        "for_trade": (
            "Residuals are the primary signal for relative-value fades. A "
            "stretched residual (|z| > 1.5) is a candidate trade; the entry "
            "is today's residual, the target is 0, and the stop is 2.5× wider."
        ),
        "source": "Standard PCA decomposition (Litterman-Scheinkman framework)",
        "display_units": "bp",
    },
    "residual_z": {
        "name": "Residual z-score",
        "category": "statistic",
        "one_liner": "How many standard deviations stretched the residual is, relative to its own history.",
        "what_it_measures": (
            "Residual_z = (residual_today − mean_residual_lookback) / std_residual_lookback. "
            "Lookback is typically 60 days. A z of +2 means today's residual is "
            "2 standard deviations above its rolling mean — historically an unusual "
            "level."
        ),
        "math": "z = (r_t - μ_lookback) / σ_lookback",
        "interpretation": (
            "|z| < 1.0   → normal range, no trade  \n"
            "|z| ∈ [1.0, 1.5] → stretched, watch list  \n"
            "|z| ∈ [1.5, 2.0] → trade candidate  \n"
            "|z| > 2.0   → extreme, high-confidence fade (subject to stationarity gate)"
        ),
        "for_trade": (
            "z-score is the headline conviction input. Conviction component "
            "z_significance = min(1, |z|/3), so |z| = 3 is the max contribution. "
            "The sign of z determines direction: +z → fade-down → 'short the yield'."
        ),
        "source": "Standard normalization for mean-reverting signals",
        "display_units": "σ",
    },

    # =========================================================================
    # Mean-reversion tests
    # =========================================================================
    "OU_half_life": {
        "name": "Ornstein-Uhlenbeck Half-Life",
        "category": "statistic",
        "one_liner": "Time it takes for a stretched residual to decay halfway back to its mean.",
        "what_it_measures": (
            "We fit the discrete-time OU process Δx_t = κ(μ − x_{t-1}) + σ·ε_t "
            "via OLS regression of Δx on x_{t-1}. κ is the speed of mean-reversion "
            "(positive = reverts to μ, negative = explosive). "
            "Half-life = ln(2) / κ ≈ 0.693 / κ. Measured in trading days."
        ),
        "math": "Δx_t = κ(μ - x_{t-1}) + σ·ε_t   →   HL = ln(2) / κ",
        "interpretation": (
            "HL < 3d  → too fast, noisy / over-fit  \n"
            "HL ∈ [3, 30]d  → sweet spot for a 1-4 week hold  \n"
            "HL ∈ [30, 60]d  → slow but tradeable  \n"
            "HL > 60d → too slow, ties up capital  \n"
            "HL negative → mean-explosive, do NOT trade"
        ),
        "for_trade": (
            "Half-life sets the expected hold time (≈ 1.5 × HL). Conviction "
            "score peaks in the [3, 30]d band. A trade with z = +2 and HL = 12d "
            "is worth ~10× more than the same trade with HL = 80d."
        ),
        "source": "Ornstein & Uhlenbeck (1930); fitting via Vasicek (1977) discretization",
        "display_units": "days",
    },
    "ADF": {
        "name": "Augmented Dickey-Fuller test",
        "category": "test",
        "one_liner": "Tests the null hypothesis that the series has a unit root (i.e., is non-stationary).",
        "what_it_measures": (
            "Regress Δx_t on x_{t-1} plus lagged Δx terms. The test statistic is "
            "the t-stat on x_{t-1}. Critical values are from Dickey-Fuller (1979). "
            "Rejecting the null (p < 0.05) means the series IS stationary "
            "(mean-reverting around a constant level)."
        ),
        "math": "Δx_t = α + ρ·x_{t-1} + Σ_i γ_i Δx_{t-i} + ε_t;  H₀: ρ = 0",
        "interpretation": (
            "ADF p < 0.01 → strong evidence of stationarity (high-confidence mean-revert)  \n"
            "ADF p < 0.05 → standard threshold for accepting stationarity  \n"
            "ADF p > 0.05 → can't reject unit-root; series may be random walk  \n"
            "Always combine with KPSS — they test opposite null hypotheses."
        ),
        "for_trade": (
            "ADF pass is a necessary (not sufficient) condition for a fade trade. "
            "If ADF fails, the series is either trending or has a unit root — "
            "betting on mean-reversion is unjustified."
        ),
        "source": "Dickey & Fuller (1979); Said & Dickey (1984) for augmented version",
        "display_units": "p-value",
    },
    "KPSS": {
        "name": "Kwiatkowski-Phillips-Schmidt-Shin test",
        "category": "test",
        "one_liner": "Tests the null hypothesis that the series IS stationary (the OPPOSITE of ADF).",
        "what_it_measures": (
            "KPSS uses an LM statistic on the sum of partial residuals. Null = "
            "stationarity, alternative = unit root. Rejecting the null (p < 0.05) "
            "means stationarity is REJECTED — series likely has a trend or unit root."
        ),
        "math": "LM = Σ S_t² / (T² · σ²);  H₀: stationary around a level (or trend)",
        "interpretation": (
            "KPSS p > 0.05 (DO NOT reject) → consistent with stationarity ✓  \n"
            "KPSS p < 0.05 (REJECT)    → drift / trend present, not pure stationary  \n"
            "Combine with ADF: ADF rejects AND KPSS does not reject → clean mean-revert  \n"
            "ADF rejects AND KPSS rejects → drift present (trade smaller)"
        ),
        "for_trade": (
            "KPSS catches drift that ADF misses. A series can pass ADF but have a "
            "slow trend — KPSS flags this. Required for the triple-gate."
        ),
        "source": "Kwiatkowski, Phillips, Schmidt & Shin (1992), J. Econometrics",
        "display_units": "p-value",
    },
    "variance_ratio": {
        "name": "Variance Ratio test (Lo-MacKinlay)",
        "category": "test",
        "one_liner": "Ratio of q-period variance to q × 1-period variance — should equal 1 for a random walk.",
        "what_it_measures": (
            "VR(q) = Var(x_t - x_{t-q}) / (q · Var(x_t - x_{t-1})). For an i.i.d. "
            "random walk, VR(q) = 1.0. VR > 1 indicates positive autocorrelation "
            "(trending / momentum). VR < 1 indicates negative autocorrelation "
            "(mean-reverting). We use q = 4."
        ),
        "math": "VR(q) = Var(x_t - x_{t-q}) / (q · Var(Δx_t))",
        "interpretation": (
            "VR ≈ 1.0    → random walk, no edge  \n"
            "VR < 0.85   → strong mean-reversion ✓ ('rich/cheap' fades work)  \n"
            "VR > 1.15   → trending / momentum (fades fail, trends work)  \n"
            "Heteroskedasticity-robust z-stat tests significance"
        ),
        "for_trade": (
            "VR is the strongest single test for mean-reversion vs random-walk. "
            "Required leg of the A6 triple-stationarity gate: ADF+KPSS+VR all "
            "must agree for the 'clean' verdict."
        ),
        "source": "Lo & MacKinlay (1988), Review of Financial Studies",
        "display_units": "ratio",
    },
    "Hurst": {
        "name": "Hurst Exponent (H)",
        "category": "statistic",
        "one_liner": "Long-memory parameter: H < 0.5 = mean-reverting, H ≈ 0.5 = random walk, H > 0.5 = trending.",
        "what_it_measures": (
            "Estimated via R/S analysis (rescaled range): the range of partial sums "
            "scales as range ~ n^H. We use detrended fluctuation analysis (DFA) "
            "or simple R/S. H ∈ [0, 1] in theory; practical values 0.3-0.7."
        ),
        "math": "log(R/S) = H · log(n) + c  →  estimate H via slope",
        "interpretation": (
            "H < 0.40 → strong mean-reversion (anti-persistent)  \n"
            "H ∈ [0.40, 0.55] → close to random walk  \n"
            "H > 0.55 → trending / persistent (momentum regime)"
        ),
        "for_trade": (
            "Hurst is the orthogonal test to ADF/KPSS. A series can pass ADF "
            "but have Hurst = 0.65 (mean-reverting but on a trend). For "
            "intraday-to-multi-day STIR trades, we want H < 0.50."
        ),
        "source": "Hurst (1951); Mandelbrot & Wallis (1969); Peng et al. (1994) for DFA",
        "display_units": "dimensionless",
    },
    "triple_gate": {
        "name": "Triple-Stationarity Gate (ADF + KPSS + VR)",
        "category": "framework",
        "one_liner": "Three independent tests must all confirm stationarity before a trade is classified 'clean'.",
        "what_it_measures": (
            "We run ADF (null = unit root), KPSS (null = stationary), and "
            "Lo-MacKinlay VR (1.0 = random walk) on the same residual series. "
            "All three must agree on stationarity for the highest-tier verdict. "
            "This is much stricter than ADF alone, which is known to over-reject "
            "the null on small samples."
        ),
        "math": "clean ⟺ (ADF rejects @ 5%) AND (KPSS does NOT reject @ 5%) AND |VR - 1| > 0.10",
        "interpretation": (
            "clean → all three agree, highest conviction  \n"
            "drift → ADF passes but KPSS rejects (trade smaller, hedge with momentum)  \n"
            "random_walk → all three suggest random walk, idea SUPPRESSED  \n"
            "non_stationary → ADF fails, idea flagged but shown with low conviction"
        ),
        "for_trade": (
            "Gameplan §A6. The gate prevents the engine from emitting fades on "
            "series that are genuinely random walks. ADF alone over-rejects ~15% "
            "of the time on noisy samples; the triple-gate reduces false fades."
        ),
        "source": "Combination of Said-Dickey (1984), KPSS (1992), Lo-MacKinlay (1988)",
        "display_units": "verdict",
    },

    # =========================================================================
    # Analog / fair-value methods
    # =========================================================================
    "analog_FV": {
        "name": "Mahalanobis k-NN Analog Fair Value",
        "category": "model",
        "one_liner": "Fair value derived from the k historical days that were most similar to today across the full state vector.",
        "what_it_measures": (
            "For each day, build a state vector = [PC1, PC2, PC3, anchor, vol_pc1, "
            "regime_dummies, ...]. Mahalanobis distance d²(x,y) = (x-y)ᵀΣ⁻¹(x-y) "
            "measures similarity in covariance-aware geometry. Take the k=30 "
            "nearest historical days; weight by exp(-d²/h); fair value = "
            "weighted-mean of those days' contract residuals. The covariance Σ "
            "is Ledoit-Wolf-shrunk to handle small samples."
        ),
        "math": "d²(x,y) = (x-y)ᵀ Σ_shrunk⁻¹ (x-y);  FV = Σ_i w_i · y_i  where w_i ∝ exp(-d²_i/h)",
        "interpretation": (
            "When analog FV agrees with raw PCA residual sign → high-conviction trade  \n"
            "When they disagree → regime change suspected, lower conviction  \n"
            "When eff_n < 30 (too few good analogs) → analog FV is unreliable"
        ),
        "for_trade": (
            "Analog FV is the 'data-driven' second opinion on top of the parametric "
            "PCA model. If today looks like 30 historical days where the residual "
            "averaged +2 bp, that's where the FV should be — not at 0."
        ),
        "source": "Mahalanobis (1936); Ledoit & Wolf (2004) for shrinkage; standard k-NN regression",
        "display_units": "bp",
    },
    "path_FV": {
        "name": "Policy-Path-Conditional Fair Value",
        "category": "model",
        "one_liner": "Fair value conditioned on the implied Fed policy path bucket (rapid cut / cut / hold / hike / rapid hike).",
        "what_it_measures": (
            "Each day is bucketed by the implied 4-quarter Fed-policy path: "
            "{rapid_cut, cut, hold, hike, rapid_hike} using ±12.5 bp and ±37.5 bp "
            "thresholds on cumulative path. Within each bucket, compute the "
            "average residual. Today's path-conditional FV = mean residual of "
            "historical days in the same bucket."
        ),
        "math": "FV_path(today_bucket) = mean(residual_t | t ∈ historical days in bucket)",
        "interpretation": (
            "If we're in 'cut' bucket and the contract's path-FV is +3 bp, that's "
            "where it has historically sat during cut cycles. If the residual today "
            "is -2 bp, we expect convergence to +3 bp (a 5 bp move)."
        ),
        "for_trade": (
            "Path-FV captures regime-specific positioning that PCA misses. Boosts "
            "conviction when path-FV and PCA-residual agree."
        ),
        "source": "Heitfield & Park (2014); custom bucket thresholds calibrated to recent data",
        "display_units": "bp",
    },
    "carry_roll": {
        "name": "Carry + Roll-Down",
        "category": "statistic",
        "one_liner": "Expected P&L over the holding period from the curve's shape alone (no rate moves).",
        "what_it_measures": (
            "Carry = front-contract yield minus its successor contract's yield. "
            "Roll-down = the change in fair-value yield as the contract ages "
            "and slides down/up the curve. Combined carry+roll is the 'free' "
            "yield earned from just holding."
        ),
        "math": "carry_bp = y_T1 - y_T2 (annualized);  roll_bp = (∂y/∂τ) × (Δτ / 365)",
        "interpretation": (
            "+2 bp/month carry+roll → contract earns 2 bp just by sitting  \n"
            "negative carry+roll → contract bleeds value if curve stays flat  \n"
            "Use to size: trade needs to overcome negative carry to be profitable"
        ),
        "for_trade": (
            "Third FV view (after historical-z and analog-FV). A trade with +5 bp "
            "expected mean-reversion but -3 bp/month carry has a much narrower edge."
        ),
        "source": "Standard fixed-income carry decomposition (Tuckman & Serrat, 2012)",
        "display_units": "bp/month",
    },

    # =========================================================================
    # Regime / cycle
    # =========================================================================
    "HMM_regime": {
        "name": "Hidden Markov Model Regime",
        "category": "model",
        "one_liner": "Classifies each day into one of K hidden regimes based on PC1/PC2/PC3/anchor/vol — fitted via Baum-Welch EM.",
        "what_it_measures": (
            "HMM with Gaussian emissions on the feature vector. K=6 typically. "
            "Each day gets a posterior probability of belonging to each regime. "
            "'Dominant confidence' = max posterior probability — measures how "
            "cleanly the day belongs to one regime."
        ),
        "math": "P(s_t = k | features_1:T) via forward-backward; transitions A_{ij}; emissions N(μ_k, Σ_k)",
        "interpretation": (
            "dominant_conf ≥ 0.80 → regime stable, trades have predictable behavior  \n"
            "dominant_conf ∈ [0.60, 0.80] → moderate stability  \n"
            "dominant_conf < 0.60 → transitioning between regimes (reduce sizing)"
        ),
        "for_trade": (
            "Conviction +0.05 if dominant_conf ≥ 0.6. Regime-transition periods "
            "see signal breakdown — analog FV becomes less reliable because the "
            "historical neighbors may belong to the OLD regime."
        ),
        "source": "Rabiner (1989) HMM tutorial; Hamilton (1989) for regime-switching",
        "display_units": "probability",
    },
    "cycle_phase": {
        "name": "Cycle Phase",
        "category": "framework",
        "one_liner": "Where we are in the Fed easing/tightening cycle — early-hike, mid-hike, peak, early-cut, mid-cut, trough, etc.",
        "what_it_measures": (
            "Heuristic classification from FDTR trajectory + days-since-last-meeting "
            "+ HMM regime. Buckets: early-cut, mid-cut, late-cut, trough, "
            "early-hike, mid-hike, late-hike, peak."
        ),
        "math": "Rule-based: phase = f(rolling_d FDTR change sign, days_since_break, HMM regime)",
        "interpretation": (
            "Each phase has a known 'favoured' trade direction:  \n"
            "early-cut → long front rates (curve bull-flattens)  \n"
            "mid-hike → short front, long back (curve bear-flattens)  \n"
            "peak → fade extremes (high vol of vol)"
        ),
        "for_trade": (
            "Conviction +0.07 if trade direction aligns with phase-favoured side. "
            "Penalty if it counters the phase."
        ),
        "source": "Custom; standard macro-cycle taxonomy",
        "display_units": "label",
    },

    # =========================================================================
    # Cross-asset
    # =========================================================================
    "vol_regime": {
        "name": "Vol Regime",
        "category": "framework",
        "one_liner": "Composite z-score of MOVE/SRVIX/SKEW/VIX classifying the rates-vol environment.",
        "what_it_measures": (
            "Average z-score (252d lookback) of MOVE, SRVIX, SKEW, VIX. "
            "Buckets: quiet (z < -0.75), normal (-0.75..0.75), stressed (0.75..1.5), "
            "crisis (> 1.5). Drives the analog-FV band widening factor."
        ),
        "math": "vol_composite_z = mean(z_MOVE, z_SRVIX, z_SKEW, z_VIX) over 252d",
        "interpretation": (
            "quiet → tight bands, fades work well  \n"
            "normal → standard band width  \n"
            "stressed → bands widen ×1.4 (analog FV uncertainty up)  \n"
            "crisis → bands widen ×2.0 (most signals unreliable)"
        ),
        "for_trade": (
            "Vol regime stressed/crisis → reduce sizing, widen stops. Quiet → "
            "tighten stops, increase size."
        ),
        "source": "Custom composite; MOVE = Merrill bond-option vol index, SRVIX = swap rate vol",
        "display_units": "regime label",
    },
    "risk_state": {
        "name": "Risk-On/Off State",
        "category": "framework",
        "one_liner": "Composite of SPX, DXY, credit, MOVE classifying risk appetite — negative = risk-on, positive = risk-off.",
        "what_it_measures": (
            "score = -z(SPX_21d_ret) + 0.5·z(DXY_21d_ret) + z(credit_chg) + 0.5·z(MOVE_chg). "
            "Buckets: risk_on (<-1), neutral (-1..0.5), risk_off (0.5..1.5), panic (>1.5)."
        ),
        "math": "score = weighted z-composite of SPX/DXY/credit/MOVE 21d changes",
        "interpretation": (
            "risk_on → equities up, credit tight, vol low → STIRs follow growth  \n"
            "risk_off → flight-to-quality → STIRs rally (front-end yields fall)  \n"
            "panic → extreme — historically all signals widen, fades fail"
        ),
        "for_trade": (
            "Risk-off tape supports STIR rally bias — adds conviction to "
            "long-yield-fade trades (i.e., buy SR3 contracts). Panic → reduce size."
        ),
        "source": "Custom composite based on standard risk-state literature",
        "display_units": "score",
    },
    "credit_cycle": {
        "name": "Credit Cycle / Recession Probability",
        "category": "framework",
        "one_liner": "4-week and 12-week recession probability from IG/HY OAS divergence.",
        "what_it_measures": (
            "recession_prob = logistic(IG_z + 0.5·HY_z - 1.5). "
            "IG_z = z-score of Bloomberg IG OAS 20d change. "
            "HY_z = z-score of HY OAS 20d change. "
            "Divergence (IG widens but HY tight, or vice-versa) is a late-cycle warning."
        ),
        "math": "recession_prob = 1 / (1 + exp(-(z_IG + 0.5·z_HY - 1.5)))",
        "interpretation": (
            "< 10% → benign credit  \n"
            "10-30% → caution, watch hard data  \n"
            "30-50% → recession risk elevated, STIRs price cuts  \n"
            ">50% → recession likely"
        ),
        "for_trade": (
            "Elevated recession prob biases STIRs to rally (Fed cuts priced in). "
            "Increases conviction of long-SR3 fades."
        ),
        "source": "Standard credit-cycle literature; logistic mapping is custom-calibrated",
        "display_units": "probability",
    },

    # =========================================================================
    # Convexity / term-premia
    # =========================================================================
    "convexity_bias": {
        "name": "Convexity Bias (Mercurio / Piterbarg)",
        "category": "statistic",
        "one_liner": "Gap between SR3-implied forwards and true OIS forwards due to the futures vs forward distinction.",
        "what_it_measures": (
            "SR3 futures settle to compounded SOFR (not a single forward rate), and "
            "their daily mark-to-market induces a small convexity adjustment vs the "
            "OIS-implied forward. Simplified: bias_bp(τ) ≈ -0.5 · σ²_ann · τ². "
            "Negative = SR3 undervalued vs OIS forward."
        ),
        "math": "bias_bp(τ) = -0.5 · σ²_ann · τ²,  τ in years, σ_ann = σ_realized · √252",
        "interpretation": (
            "Bias becomes large for back-end contracts (τ² grows quickly).  \n"
            "Front-end (τ < 0.5y) bias is < 0.5 bp — ignorable.  \n"
            "5-year-out contracts: bias can be 5-10 bp — material."
        ),
        "for_trade": (
            "Convexity warning fires when bias > 3 bp — flag the trade so user "
            "knows SR3-implied rates need to be adjusted before benchmarking to "
            "OIS curves."
        ),
        "source": "Mercurio (2018), Risk Magazine; Piterbarg (2006), Wilmott; standard quant convention",
        "display_units": "bp",
    },
    "ACM_term_premium": {
        "name": "ACM (Adrian-Crump-Moench) Term Premium",
        "category": "model",
        "one_liner": "Decomposes UST yields into expected-path + term-premium components via affine 5-factor model.",
        "what_it_measures": (
            "Standard NY Fed term-premium estimates. Yield = expected average "
            "short rate over the tenor (expected_path) + term premium (extra "
            "yield demanded for bearing duration risk). Updated monthly by NY Fed."
        ),
        "math": "y(τ) = (1/τ)·E[∫_0^τ r_s ds] + TP(τ)",
        "interpretation": (
            "TP > 0 → market demands premium for duration risk (bond-bear regime)  \n"
            "TP ≈ 0 → no premium (mid-cycle)  \n"
            "TP < 0 → 'flight-to-quality' premium NEGATIVE (rare; late-cycle/recession)"
        ),
        "for_trade": (
            "For STIRs, the expected-path component is the actionable signal. "
            "If expected_path falls but term_premium rises (yield flat), the "
            "Fed-cut narrative is strengthening — bullish for SR3."
        ),
        "source": "Adrian, Crump & Moench (2013), J. Financial Economics; data: NY Fed",
        "display_units": "bp",
    },
    "FOMC_step_path": {
        "name": "FOMC Step-Path (Heitfield-Park bootstrap)",
        "category": "model",
        "one_liner": "Probability distribution over Fed decisions {large_cut, cut, hold, hike, large_hike} per meeting.",
        "what_it_measures": (
            "For each upcoming FOMC, bootstrap the CMC panel to draw possible "
            "step paths. Bucket each draw's per-meeting move using ±12.5 bp and "
            "±37.5 bp thresholds. Probabilities = frequency in each bucket "
            "across draws. Kernel σ = 37.5 bp for distribution shape."
        ),
        "math": "P(decision_k | meeting_m) via bootstrap; thresholds ±12.5 bp / ±37.5 bp",
        "interpretation": (
            "P(hold) > 0.65 → strongly priced for no change  \n"
            "P(cut) > 0.40 → cut likely priced in  \n"
            "Distribution skew is more informative than the mean"
        ),
        "for_trade": (
            "If market prices P(cut)=0.7 but your view says 0.4, fade the "
            "rich SR3 contracts whose value depends on that cut materializing."
        ),
        "source": "Heitfield & Park (2014) FOMC bootstrap; custom kernel calibration",
        "display_units": "probability",
    },

    # =========================================================================
    # Engine plumbing
    # =========================================================================
    "eff_n": {
        "name": "Effective Sample Size (eff_n)",
        "category": "statistic",
        "one_liner": "Number of independent observations behind a residual — sets the lower bound on conviction.",
        "what_it_measures": (
            "Number of non-NaN observations in the residual series, adjusted for "
            "the trading-day frequency. eff_n < 30 means we don't have a reliable "
            "statistical base for the test stats."
        ),
        "math": "eff_n = count(non-NaN observations in residual series)",
        "interpretation": (
            "eff_n < 30  → 'low_n' flag, conviction capped  \n"
            "eff_n ∈ [30, 100] → moderate confidence  \n"
            "eff_n > 100 → robust statistics"
        ),
        "for_trade": "Conviction component eff_n score = min(1, eff_n/100). Caps the |z|-based component for small samples.",
        "source": "Standard practice for small-sample statistical inference",
        "display_units": "count",
    },
    "cross_PC_corr": {
        "name": "Cross-PC Correlation",
        "category": "statistic",
        "one_liner": "Correlation between supposedly-orthogonal PCs (should be near 0).",
        "what_it_measures": (
            "Rolling correlation between PC1·PC2, PC1·PC3, PC2·PC3. PCA constructs "
            "PCs to be orthogonal by definition, BUT this only holds on the "
            "in-sample period. Out-of-sample correlations leak — when |corr| > 0.3, "
            "the basis has drifted and the model needs a refit."
        ),
        "math": "ρ(PC_i, PC_j) over rolling window — should be ≈ 0 in-sample",
        "interpretation": (
            "|ρ| ≤ 0.20 → basis is clean ✓  \n"
            "|ρ| ∈ [0.20, 0.30] → mild drift, OK  \n"
            "|ρ| > 0.30 → basis broken, refit needed; trades flagged 'regime_unstable'"
        ),
        "for_trade": "If cross-PC corr is high, a 'pure PC1' trade is no longer pure — it has unintended slope/curvature exposure. Conviction docks 0.05.",
        "source": "Implicit in any rolling-window PCA application",
        "display_units": "correlation",
    },
    "reconstruction_pct": {
        "name": "3-PC Reconstruction Pct",
        "category": "statistic",
        "one_liner": "Fraction of today's yield-change variance explained by the first 3 PCs.",
        "what_it_measures": (
            "(Var_total - Var_residual) / Var_total, for today's yield-change vector "
            "reconstructed from 3 PCs. Should be ≥ 0.95 if the curve is well-behaved."
        ),
        "math": "1 - (Σ residual_i²) / (Σ yield_change_i²)",
        "interpretation": (
            "≥ 0.95 → 3 PCs capture the curve, model is sound  \n"
            "0.90-0.95 → mild idiosyncratic moves, some residuals real, some noise  \n"
            "< 0.90 → big idiosyncratic move today (e.g. one contract trading on a story); reduce conviction"
        ),
        "for_trade": "Conviction component model_fit = 0.06 · pct. A 95% day adds 0.057, a 90% day adds 0.054.",
        "source": "Standard PCA fit diagnostic",
        "display_units": "%",
    },
    "conviction": {
        "name": "Conviction Score (0..1)",
        "category": "framework",
        "one_liner": "Composite of 17 weighted inputs deciding how strongly to back this trade.",
        "what_it_measures": (
            "Sum of weighted contributions from: z-significance (0.13), OU sweet-spot "
            "(0.09), ADF pass (0.08), eff_n (0.06), model_fit (0.06), cycle_alignment "
            "(0.07), regime_stable (0.05), cross_PC_orth (0.05), variance_regime (0.05), "
            "analog_FV_agree (0.05), path_FV_agree (0.04), seasonality (0.02), "
            "event_drift (0.015), empirical_hit_rate (0.04), lifecycle (±0.04), "
            "cross_confirm (0.05), slippage_ok (0.03), convexity_safe (0.03)."
        ),
        "math": "conviction = clip(Σ w_i · score_i, 0, 1)",
        "interpretation": (
            "conviction ≥ 0.70 → high-conviction (green pill)  \n"
            "conviction ∈ [0.50, 0.70] → moderate (amber)  \n"
            "conviction < 0.50 → low (gray) — filtered out by default ≥ 0.40 threshold"
        ),
        "for_trade": "Sort key for the trade table. Cross-confirmation (>1 source) typically the biggest single boost.",
        "source": "Custom — see lib/pca_trades.py:score_conviction()",
        "display_units": "score",
    },
}


def get_concept(key: str) -> dict:
    """Lookup. Returns empty dict if not found rather than raising."""
    return CONCEPTS.get(key, {})


def concept_tooltip_html(key: str) -> str:
    """Build a compact tooltip HTML for use in `help=` attributes / inline hover."""
    c = get_concept(key)
    if not c:
        return ""
    lines = []
    lines.append(c["name"])
    lines.append("")
    lines.append(c["one_liner"])
    if c.get("math"):
        lines.append("")
        lines.append("MATH:  " + c["math"])
    if c.get("interpretation"):
        lines.append("")
        lines.append("READ:")
        lines.append(c["interpretation"])
    if c.get("source"):
        lines.append("")
        lines.append("REF:  " + c["source"])
    return "\n".join(lines)


def concept_disclosure_html(key: str) -> str:
    """Full plain-English disclosure HTML for use inside an expander.

    More verbose than `concept_tooltip_html` — used in glossary view."""
    c = get_concept(key)
    if not c:
        return f"<div>No glossary entry for '{key}'.</div>"
    return f"""
<div style='line-height:1.6; font-size:0.85rem;'>
<div style='font-weight:600; color:var(--accent); margin-bottom:0.3rem;'>{c['name']}</div>
<div style='color:var(--text-muted); font-style:italic; margin-bottom:0.5rem;'>{c['one_liner']}</div>

<div style='font-size:0.7rem; color:var(--text-dim); text-transform:uppercase; margin-top:0.6rem;'>What it measures</div>
<div>{c['what_it_measures']}</div>

<div style='font-size:0.7rem; color:var(--text-dim); text-transform:uppercase; margin-top:0.6rem;'>Math</div>
<div style='font-family:JetBrains Mono, monospace; color:var(--text-body); background:rgba(255,255,255,0.03);
            padding:0.3rem 0.5rem; border-radius:4px;'>{c['math']}</div>

<div style='font-size:0.7rem; color:var(--text-dim); text-transform:uppercase; margin-top:0.6rem;'>How to read the number</div>
<div style='white-space:pre-line;'>{c['interpretation']}</div>

<div style='font-size:0.7rem; color:var(--text-dim); text-transform:uppercase; margin-top:0.6rem;'>Why a trader cares</div>
<div>{c['for_trade']}</div>

<div style='font-size:0.7rem; color:var(--text-dim); margin-top:0.6rem;'>Reference: <span style='color:var(--text-muted);'>{c['source']}</span></div>
</div>
"""
