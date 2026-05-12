# PCA Trade Screener — A Guide in Plain English

*Written for someone who has never heard of PCA, residuals, half-life, or any of the other jargon. By the end of this guide, every number on every trade card will make sense.*

---

## Table of contents

1. [What does this tool actually do?](#1-what-does-this-tool-do)
2. [The big idea: the yield curve moves in 3 ways](#2-the-big-idea)
3. [What does the engine measure?](#3-what-does-the-engine-measure)
4. [The 3 factors, explained with everyday analogies](#4-three-factors)
5. [What is a "residual"?](#5-what-is-a-residual)
6. [What does "stretched" mean? — the z-score](#6-z-score)
7. [Why fade a stretched residual? — mean reversion](#7-mean-reversion)
8. [The statistical tests, in plain English](#8-stat-tests)
9. [Half-life — how long the trade takes to work](#9-half-life)
10. [The three trade structures: outrights, spreads, flies](#10-trade-structures)
11. [The 19 ways the engine finds trades](#11-the-19-ways)
12. [How conviction is scored](#12-conviction)
13. [Three opinions on fair value](#13-three-fv)
14. [Regime, cycle, and the wider market](#14-regime-cycle)
15. [Reading a trade card line by line](#15-reading-a-card)
16. [Frequently asked questions](#16-faq)
17. [Glossary](#17-glossary)

---

<a name="1-what-does-this-tool-do"></a>
## 1. What does this tool actually do?

This is a **trade-idea scanner** for short-term interest rate (STIR) futures. Specifically, it watches the **CME SR3 futures contracts** — these are bets on what the 3-month US lending rate (SOFR) will be at various dates in the future.

Imagine you have a row of about 30 contracts, each one settling on a different date over the next several years. Each contract's price tells you what the market thinks the 3-month rate will be at that date.

The screener's job: **find places where the market has temporarily got the price wrong**, so you can bet on it returning to the "right" price and make money.

That's it in a sentence. Everything else in this guide is detail on **how** the engine identifies "wrong" prices and **how confident** it is.

---

<a name="2-the-big-idea"></a>
## 2. The big idea: the yield curve moves in 3 ways

Imagine you draw a curve plotting today's expected interest rate for every future date — that's the **yield curve**.

Every day, this curve moves. But it doesn't wiggle randomly — it moves in **three predictable patterns**:

### Pattern 1: The whole curve shifts up or down

When the Fed signals tightening, ALL rates go up — the curve shifts up like an elevator. When they signal easing, the curve shifts down. **This is the biggest type of move** — about 85% of all daily curve action.

### Pattern 2: Front and back move in opposite directions

Sometimes the short rates rise but long rates fall (or vice versa). The curve "steepens" or "flattens" like a see-saw pivoting in the middle. **About 10% of the action.**

### Pattern 3: The middle bulges or sags

Sometimes the very front and very back are stable but the middle moves on its own. The curve develops a "belly" or a "hollow". **About 3% of the action.**

A statistical technique called **PCA (Principal Component Analysis)** is the formal way to discover these three patterns from historical data. The technique was first applied to yield curves by Litterman and Scheinkman at Goldman Sachs in 1991, and it's been the foundation of bond relative-value trading ever since.

You don't need to understand the math — just remember: **PCA finds the 3 dominant patterns of curve movement automatically**. We call them PC1 (level), PC2 (slope), and PC3 (curvature). Together they explain ~98% of all daily curve action.

---

<a name="3-what-does-the-engine-measure"></a>
## 3. What does the engine measure?

Every day, the engine asks: **for each contract, how much of today's price move can I explain using just the 3 dominant patterns?**

If the model explains the move perfectly, the contract is behaving "as expected" — no trade.

If the model misses by a noticeable amount, the contract has done something **idiosyncratic** — moved on its own story, not in tune with the curve. This unexplained movement is called the **residual**.

> Think of it like predicting the temperature in your city. You have a model that says "today should be 18°C". If the actual temperature is 18°C, your model worked. If it's 25°C, your model missed by 7°C — and that 7°C is the *residual*. Maybe a heat wave came in, maybe a sensor malfunction. Either way, the model didn't see it coming.

When a contract has a big positive residual, it's "rich" — priced too high relative to the model. When negative, it's "cheap". The bet is that big residuals are temporary anomalies that will close back up.

---

<a name="4-three-factors"></a>
## 4. The three factors, explained with everyday analogies

The 3 PCA factors aren't mysterious — they correspond to things you'd identify if you just looked at thousands of yield-curve charts side-by-side.

### PC1 — The Level (the elevator)

Every contract gets approximately the same weight. When PC1 moves, EVERY rate moves in the same direction by roughly the same amount.

**Analogy**: Imagine the entire yield curve is a flat sheet of paper, and PC1 lifts or lowers the whole sheet. When the Fed surprises with a hawkish statement, PC1 jumps positive — the whole curve moves up together.

### PC2 — The Slope (the see-saw)

Front contracts get positive weight, back contracts get negative weight (or vice versa). When PC2 moves, the front and back move in opposite directions.

**Analogy**: Imagine the curve is a see-saw pivoting around the 2-year point. PC2 positive = front end up, back end down = "bear flattener". PC2 negative = "bull steepener". This is the typical move during a Fed cycle transition.

### PC3 — The Curvature (the belly bulge)

The middle contracts get one sign of weight; the front and back get the opposite sign. When PC3 moves, the belly moves relative to the wings.

**Analogy**: Imagine pinching the middle of a string and pulling up — the middle rises while the ends stay put. PC3 positive = belly cheap (high yield) relative to wings = "long the wings, short the belly" is the canonical trade.

### Why does this matter?

Because most "trade ideas" in STIRs are bets on **one of these three patterns**. The engine constructs special baskets of contracts where:
- A **PC1 basket** only loads on the level (you bet on a parallel rate move)
- A **PC2 spread** only loads on the slope (you bet on steepening / flattening)
- A **PC3 fly** only loads on the curvature (you bet on belly cheapness)

These are called "PC-isolated structures" — they let you bet on ONE factor at a time without accidental exposure to the others.

---

<a name="5-what-is-a-residual"></a>
## 5. What is a "residual"?

A residual is **the part of today's price move that the 3 dominant factors can't explain**.

### Walking through a worked example

Imagine the model says: based on today's PC1, PC2, PC3 moves, contract SRAU28's yield should move +3 bp today.

But actually, SRAU28's yield moved +5 bp today.

The **residual** is 5 − 3 = **+2 bp**. SRAU28 moved 2 bp more than the curve patterns explain. Some kind of story specific to this contract pushed it up extra.

Now if you watch this every day, you build a **residual series** for SRAU28 — one number per day. Most days it's near zero (the model gets it about right). Occasionally it spikes — those are the trade opportunities.

### The key insight

If you ADD UP all the daily residuals over time, you get the **cumulative deviation from fair value**. That's the "level residual" — how far away from the model-implied price the contract has drifted in total.

> Think of it like this: every day the model makes a small mistake. Some days too high, some days too low. Over weeks, those mistakes might mostly cancel out (cumulative deviation ≈ 0 = at fair value), OR they might pile up in one direction (cumulative deviation ≠ 0 = the contract has drifted to "rich" or "cheap" territory).

The engine's trade signals come from large cumulative deviations — when a contract has piled up its mistakes in one direction enough to look stretched.

---

<a name="6-z-score"></a>
## 6. What does "stretched" mean? — the z-score

OK so we have a "level residual" for each contract — a number in basis points telling us how far above or below fair value the contract sits.

But what counts as "big"? +1 bp deviation is tiny — happens every day. +50 bp is huge — might happen once a year.

The **z-score** normalizes this. It tells you **how unusual** today's deviation is **compared to its own history**.

```
z = (today's residual − historical average of residuals) / historical standard deviation
```

- **z = 0** → completely typical
- **z = +1** → one standard deviation above the norm. Happens about 16% of the time. Not much.
- **z = +2** → two standard deviations. Happens about 2.3% of the time. Unusual.
- **z = +3** → three standard deviations. Happens about 0.1% of the time. Very rare.

The engine looks for **|z| > 1.5** (about a 7% tail event) as the threshold to even consider a trade. **|z| > 2** means high-conviction stretch.

> Analogy: If someone in your city is 7 feet tall, that's roughly +3σ above average height. You'd notice. The engine "notices" contracts that are >1.5σ away from their typical state.

---

<a name="7-mean-reversion"></a>
## 7. Why fade a stretched residual? — mean reversion

The whole point of fading a stretched residual is the bet that **it will come back**. This is called **mean reversion**: stretched-away things tend to return to where they came from.

### Why should we believe this?

For STIR contracts specifically, three reasons:

1. **The 3 PCA factors really do explain most of the variance.** If a contract drifts away from PC-implied fair value, there's no fundamental reason it should stay drifted — the market will eventually fix it.

2. **Other traders are watching the same residuals.** When something looks rich/cheap to you, it looks rich/cheap to dozens of other relative-value desks too. They'll trade against the drift and pull it back.

3. **Most "stretches" come from temporary causes** — a single big order that moved one contract, a stale settle, a position unwind, a one-off news event affecting only one delivery date. None of these have lasting effects.

### What can go wrong?

Sometimes a "stretch" is real. Maybe a specific contract has genuinely changed risk (regulatory change, supply shock, etc.). Then fading it loses money. The statistical tests below try to filter these out.

---

<a name="8-stat-tests"></a>
## 8. The statistical tests, in plain English

The engine doesn't just look at z-score. It runs three independent statistical tests before declaring a residual "tradeable":

### Test 1: ADF (Augmented Dickey-Fuller)

**What it asks**: Is this residual series stationary (does it stay in a bounded range), or does it wander off forever like a drunk's path home (a "random walk")?

**Why it matters**: You can only fade something if it RETURNS. A random walk doesn't return — it just keeps walking.

**Verdict**: If ADF "rejects" (= "yes, it's stationary, it does return"), the series is fadeable.

### Test 2: KPSS (Kwiatkowski-Phillips-Schmidt-Shin)

**What it asks**: Is this residual oscillating around a fixed level, or is there a slow drift (trend) baked in?

**Why it matters**: A "trending" residual won't revert cleanly — it'll keep drifting. You want a residual that wobbles around a flat line.

**Verdict**: If KPSS does NOT reject, the series is consistent with flat oscillation.

### Test 3: Lo-MacKinlay Variance Ratio

**What it asks**: Compare the 4-day variance to 4× the 1-day variance. For a true random walk, they should equal. Mean-reverting series have LESS variance over longer horizons (because reversions cancel out).

**Why it matters**: This catches what ADF misses on smallish samples.

**Verdict**: If the variance ratio is below 1, it's mean-reverting.

### The triple-stationarity gate

When **all three tests agree** that the residual is mean-reverting → the trade is tagged **"clean"** (highest confidence).

When some tests pass but not all → tagged **"drift"** or **"non_stationary"** (trade smaller).

When all three say "random walk" → trade is **suppressed** entirely.

> Analogy: Like getting a second and third opinion on a medical diagnosis before surgery. ADF alone over-diagnoses ~15% of the time on small samples. The 3-test combination has much lower error.

---

<a name="9-half-life"></a>
## 9. Half-life — how long the trade takes to work

If a contract is +5 bp rich and the residual mean-reverts, how long does it take to drift back to fair value? That's the **half-life**: the number of trading days for the residual to **decay halfway** back to zero.

The engine fits a mathematical model (Ornstein-Uhlenbeck, an old physics model for diffusing particles) to the residual series and extracts the decay rate.

### Reading the number

- **Half-life = 3 days** → very fast reverter. By day 4-5 you've captured half the move. Aggressive day-trade.
- **Half-life = 10 days** → typical for STIRs. Hold for 2-3 weeks, capture most of the move.
- **Half-life = 30 days** → slow. Hold for a month or more. Capital tied up; opportunity cost.
- **Half-life = 60+ days** → too slow to be useful. The engine penalises these in the conviction score.
- **Half-life < 3 days** → too fast, often just noise. Penalised.
- **Half-life = NaN (undefined)** → no detectable mean reversion in this residual. Avoid.

### Expected holding period

The engine reports `expected_revert_d = 1.5 × half_life`. After 1.5 half-lives, about **65% of the move** has been captured — a good target for closing.

> Analogy: Half-life is like a hot cup of coffee cooling down. The first 5 minutes it loses 50% of its excess heat (above room temperature). Another 5 minutes, 50% of what's left. And so on. The "half-life" tells you how fast the system equilibrates.

---

<a name="10-trade-structures"></a>
## 10. The three trade structures

Every trade idea is one of three structural types:

### Outright — single contract

You're just betting on ONE contract. Example: "Buy 1 SRAH27 because its residual is −1.94σ (cheap)."

- Simplest to execute (1 order)
- Most directional — your P&L depends on this one contract's price
- Larger risk exposure to overall curve moves

### Spread — two contracts, opposite sides

You're betting on the **difference** between two contracts. Example: "Buy SRAH27 + sell SRAH28 = bet that the curve flattens between Mar 2027 and Mar 2028."

- The two sides offset each other's directional risk
- You're betting on **relative** mispricing, not absolute direction
- Less risky in big curve moves
- CME lists these as standalone contracts (so you can trade them as a single ticket)

### Fly (butterfly) — three contracts, "1, -2, 1" pattern

You're betting on **curvature**. Example: "Buy SRAH27 + sell 2×SRAM27 + buy SRAU27 = bet that the belly (M27) is too rich relative to the wings (H27, U27)."

- Pure curvature bet — completely hedged for level and slope moves
- Smallest absolute P&L per move (but also smallest risk)
- CME lists popular flies as standalone contracts

### Pack — 4 contracts averaged

A "pack" is 4 consecutive quarterly contracts averaged together. Whites = first 4 = nearest year. Reds = next 4. Then Greens, Blues, Golds, etc.

Used for: betting on the "average" expectation over a 1-year period. Less granular than individual contracts but easier to trade and less noisy.

### Basket — 4+ contracts with custom weights

The engine sometimes constructs custom weighted combinations of 4+ contracts that isolate ONE PC factor (most commonly PC1 = level). These are called PC1 baskets. They're not listed CME products — you execute each leg separately.

---

<a name="11-the-19-ways"></a>
## 11. The 19 ways the engine finds trades

The engine has 19 different "generators" — each one scans for a specific kind of opportunity. Here they are in plain English:

### Tier 1 — PCA-isolated structures

| Generator | What it does (plain English) |
|---|---|
| **PC1 basket** | Looks for moments when a contract's level (parallel-shift) component is off. Builds a 4-leg basket that captures pure level exposure with zero slope or curvature drift. |
| **PC2 spread** | Looks for moments when the curve's slope is off. Builds a spread that captures pure slope exposure. |
| **PC3 fly** | Looks for moments when the curve's curvature is off. Builds a fly (1,−2,1) that captures pure curvature exposure. |

These are the **purest** factor bets — you isolate exactly which PCA factor you're betting on.

### Tier 2 — Cross-factor combinations

| **Anchor** | Watches the 12-month vs 24-month yield spread. When it stretches beyond ±1.5σ, fade it. |

### Tier 3 — Segment-specific PCAs

| **Front PCA** | Fits a separate PCA on JUST the 1-12 month contracts. Catches dislocations that the full-curve PCA might smooth over. |
| **Belly PCA** | Same for 9-30 month contracts. |
| **Back PCA** | Same for 24-60 month contracts. |

### Tier 4 — Per-contract residual fades

| **Outright fade** | Scans every individual contract for stretched residuals (|z| > 1.5). |
| **Spread fade** | Scans every listed CME calendar spread for stretched residuals. |
| **Fly arb** | Scans every listed CME butterfly for stretched residuals. |
| **Pack** | Scans the 4-contract pack residuals (Whites, Reds, Greens, Blues). |
| **Pack fly** | A fly built on packs: "Whites − 2×Reds + Greens" for ultra-stable curvature. |
| **Bundle RV** | A bundle (4 packs) relative-value play. |
| **Outlier reversal** | Extreme stretches (|z| > 3) for sharp reversion bets. |
| **Carry fly** | A fly weighted to maximize "carry" while staying curvature-isolated. |

### Tier 5 — Cross-validation methods

| **Analog FV** | Uses k-nearest-neighbours to find the 30 historical days most similar to today (across many features), and asks "where did the residual sit on those days?" If today's residual disagrees with the analog FV by > 2σ → trade. |
| **Path FV** | Same idea but conditioned on the implied Fed policy path. "On historical days when the Fed was in a 'hold' regime, where did this contract typically sit?" |

### Tier 6 — Event-aware

| **Event drift** | After FOMC/CPI/NFP releases, there's often a 1-3 day drift. Captures that. |
| **Event impact ranking** | Identifies trades that are especially likely to perform on certain events. |

### How they work together

A given day, some generators fire (find a trade) and others don't. When MULTIPLE generators fire on the SAME leg combination + direction, that's strong evidence — the engine treats this as **cross-confirmation** and boosts the conviction score (see next section).

---

<a name="12-conviction"></a>
## 12. How conviction is scored

Every trade idea gets a **conviction score** from 0 to 1. This is the engine's overall "should I trust this?" verdict.

It's the weighted sum of 17 separate signals. Here they are with their max contributions:

| Signal | Max weight | What it's asking |
|---|---:|---|
| **z significance** | 0.13 | How stretched is the residual? (z=3 maxes this out) |
| **OU sweet spot** | 0.09 | Is half-life in the [3, 30]d ideal range? |
| **ADF pass** | 0.08 | Did the stationarity test pass? |
| **Sample size** | 0.06 | Is there enough history for the stats to be reliable? |
| **Model fit** | 0.06 | What fraction of today's curve variance does the 3-PC model explain? |
| **Cycle alignment** | 0.07 | Does this trade go WITH or AGAINST the historical-favored direction for the current Fed cycle phase? |
| **Regime stable** | 0.05 | Is the market in a "stable" regime (not transitioning)? |
| **Cross-PC orthogonality** | 0.05 | Are the 3 PCs still cleanly separate today (no leakage between them)? |
| **Variance regime** | 0.05 | Does the current volatility regime favor this trade type? |
| **Analog FV agreement** | 0.05 | Does the k-NN historical-similarity FV agree with the raw residual? |
| **Path FV agreement** | 0.04 | Does the policy-path-conditional FV agree? |
| **Seasonality** | 0.02 | Does the calendar (day of month, week of quarter) favor this? |
| **Event drift** | 0.015 | Are we in a window where event-drift is statistically likely? |
| **Empirical hit rate** | 0.04 | Has this generator made money on its trailing 30 trades? |
| **Lifecycle** | ±0.04 | Is this idea NEW/MATURING/PEAK (+) or FADING (−)? |
| **Cross-confirm** | 0.05 | How many of the 19 generators independently came up with this same trade? |
| **Slippage** | 0.03 | Is the bid-ask cost small relative to expected P&L? |
| **Convexity safe** | 0.03 | Is the back-of-curve futures-vs-OIS gap small? |

The contributions add up. Clipped to [0, 1].

### Tier thresholds (what you see as colored badges)

- **HIGH (green)**: ≥ 0.70 — high conviction, trade with size
- **MED (gold)**: 0.55 to 0.70 — moderate, normal size
- **LOW (amber)**: 0.40 to 0.55 — speculative, small size
- **Below 0.40**: filtered out by default

### Why cross-confirmation is the most important single signal

The cross-confirm bonus only gives 0.05, but **practically it's the biggest swing factor**. Here's why: most trades come out of 1-2 generators. When 4-5 generators independently identify the SAME trade, it tells you the same dislocation is visible through multiple analytical lenses. That convergence is much stronger evidence than a single statistical test.

So: a trade with `|z|=1.9` confirmed by 4 generators typically beats a trade with `|z|=2.8` confirmed by 1.

---

<a name="13-three-fv"></a>
## 13. Three opinions on "fair value"

For every contract, the engine produces THREE independent estimates of fair value:

### FV #1: PCA fair value

What the 3-factor PCA model predicts today's yield should be. This is the engine's PRIMARY signal — the residual is measured against this.

### FV #2: Analog FV (Mahalanobis k-NN)

Looks at the 30 historical days most similar to today across many features (PC1/2/3 levels, anchor slope, vol regime, etc.). Asks: "where did this contract's residual sit on those days?" The weighted average is the analog FV.

> Analogy: Like asking 30 historical days "where was this contract in your time?" and taking a weighted vote.

If today's residual is, say, +2 bp and the analog FV is +1 bp, the analog is consistent with PCA. If analog says −3 bp, there's a disagreement — the model might be missing something.

### FV #3: Path-conditional FV (Heitfield-Park)

Looks at where the contract sat on historical days when the **implied Fed policy path** looked like today's. Specifically buckets days into 5 categories (large_cut, cut, hold, hike, large_hike) based on the implied path through the next 4 quarters.

> "On historical days when the market was pricing a 'hold' regime, where did this contract sit on average?"

### Why three?

Because no single estimate is reliable on its own. When all three agree → very high confidence trade. When they disagree → caution.

---

<a name="14-regime-cycle"></a>
## 14. Regime, cycle, and the wider market

The engine doesn't just look at residuals in isolation. It also tracks the **broader market state** because mean-reversion edges work differently in different regimes.

### Regime classification (HMM + GMM)

Every day is classified into one of 6 "hidden" market regimes based on features like PC1 level, PC2 level, PC3 level, anchor slope, recent volatility. Using statistical methods called Gaussian Mixture Model (GMM) and Hidden Markov Model (HMM), the engine assigns probabilities to which regime today is in.

> Why? Because the SAME trade idea works differently in different regimes. A short-the-rich-belly fly might work great in a stable regime but fail in a panic regime.

If the dominant regime probability is ≥ 0.6 → regime is "stable" → trade with confidence.
If < 0.6 → regime is "transitioning" → reduce sizing.

### Cycle phase

The engine also tracks which phase of the Fed cycle we're in:
- early-cut / mid-cut / late-cut / trough
- early-hike / mid-hike / late-hike / peak

Each phase has historically favored trade directions. A trade aligned with the cycle phase gets +0.07 conviction.

### Cross-asset overlays (8 inputs)

Beyond the curve itself, 8 other markets are tracked daily and feed into the conviction:

1. **Vol regime** — composite z-score of MOVE / SRVIX / SKEW / VIX. "Quiet" vs "stressed" vs "crisis" — affects band widening for analog FV.
2. **Risk-on/off state** — SPX, DXY, credit, MOVE composite. "Risk-on" or "risk-off".
3. **Credit cycle recession probability** — derived from IG/HY OAS divergence.
4. **Cross-asset lead-lag matrix** — which assets are leading the PCs.
5. **Equity-rates correlation regime** — is the SPX-vs-Treasury correlation positive (inflation regime) or negative (normal)?
6. **Term-premia decomposition** — using NY Fed's ACM model to split yields into expected-path + risk-premium components.
7. **FX rate-differential** — divergence in USD strength.
8. **Convexity bias** — gap between SR3-implied forwards and true OIS forwards.

If the tape is "panic" or "stressed", conviction is docked — the engine becomes more cautious because mean-reversion edges fade in crisis.

---

<a name="15-reading-a-card"></a>
## 15. Reading a trade card line by line

Every trade in the screener renders as a **card**. Here's what each piece means in plain English:

### The collapsed (header) view

```
#5 [MED] SRAU28-U29 spread (spread-fade)   BUY   8.00 bp    0.00 bp    -1.75 bp    $37   $94   0.40:1   ▰▰▰▱▱▱▱ 0.42
```

| Element | Meaning |
|---|---|
| `#5` | Rank in the trade list, sorted by conviction |
| `[MED]` | Conviction tier badge (HIGH/MED/LOW) — color-coded |
| `SRAU28-U29 spread` | The actual CME contract being traded (this case: a calendar spread) |
| `(spread-fade)` | Which of the 19 generators identified this trade |
| `BUY` | Whether to buy or sell. For listed spreads/flies, this means buying that listed CME contract |
| `8.00 bp` | ENTRY: the current quoted price (for spreads/flies, in bp differential) |
| `0.00 bp` | TARGET: where the engine expects it to move (fair value) |
| `-1.75 bp` | STOP: where to cut losses if it goes wrong |
| `$37` | Expected P&L per contract |
| `$94` | Risk per contract (stop-out loss) |
| `0.40:1` | Reward-to-risk ratio. (e.g., 0.40 means you risk $1 to make $0.40 on average) |
| `▰▰▰▱▱▱▱ 0.42` | Conviction bar (more filled = higher conviction). 0.42 = the score itself |

### The subtitle

```
EXECUTE LEGS: BUY SRAU28 @ 96.3800 · SELL SRAU29 @ 96.3000
```

For multi-leg trades, this shows the actual outright contracts you'd execute and at what prices. Lets you verify the trade against the visible market.

### The expanded view (click ▸ to open)

When expanded, the card shows:

1. **A LARGE prices block** — same ACTION/ENTRY/TARGET/STOP/P&L/Risk in big text
2. **Three side-by-side charts**:
   - Price + PCA fair value over the last 90 days
   - Residual time series with ±1σ / ±2σ bands
   - A horizontal bar chart showing each of the 17 conviction inputs and how much they contributed
3. **A dynamic 5-paragraph narrative** explaining:
   - ① What the engine observed
   - ② Why it's tradeable (statistical tests passing)
   - ③ Why this direction
   - ④ The exact derivation math
   - ⑤ Risk factors that could derail the trade
4. **Per-factor analysis** — every input to the conviction with color-coded chips (green = supportive, amber = caveat)

---

<a name="16-faq"></a>
## 16. Frequently asked questions

### Q: I see prices like "8.00 bp" or "-1.75 bp" for spreads. Aren't prices supposed to be like 96.34?

For **outright contracts** (single SR3 contract), yes — prices are like 96.34.

For **listed CME calendar spreads and butterflies**, no — CME quotes them as the **price differential** in basis points. So "8.00 bp" means the front contract is 8 bp HIGHER in price than the back contract (in bp-of-price-diff units, which is ×100 of raw price units).

This is a CME quoting convention. The engine handles the conversion. If you trade the spread as a single CME ticket, the price you see is the bp number.

### Q: Why is the target always 0 for spreads and flies?

Because of the math: for a CME-quoted spread/fly, the "fair value" (where the underlying yield-differential is at PCA-implied curve fair value) is mathematically **zero by construction**. The current quote (e.g., +8 bp) is the deviation from FV. Mean reversion sends it to 0.

### Q: My half-life is 0.5 days. Should I trade this?

Probably not. Half-lives below 3 days usually indicate noise rather than real mean-reversion. The engine penalizes these in the conviction score. If you see HL = 0.5d AND conviction > 0.7 — there's probably noise in the inputs; double-check.

### Q: All my trades show "non_stationary" gate. What does that mean?

The triple-stationarity gate has three components. "non_stationary" means at least one of the three tests didn't agree the residual is stationary. The trade is still surfaced (because |z| > 1.5), but with reduced statistical confidence. Filter by `gate = "clean"` to see only trades where all 3 tests agree.

### Q: What's the difference between SR3 outright and SR3 spread contracts?

- **Outright** = bet on one delivery date's rate (e.g., SRAH27 = Mar 2027 SOFR 3M rate)
- **Spread** = bet on the DIFFERENCE between two outright rates (e.g., SRAH27-H28 = Mar 2027 rate minus Mar 2028 rate)

Both are listed on CME. The spread is its own contract — you buy 1 spread, you get long the front leg + short the back leg automatically.

### Q: Why does the engine reject some trades as "random_walk"?

Random walks (like a drunk's path home) don't return to any "average" — they just wander. Trading mean-reversion on a random walk is statistically the same as flipping a coin and hoping. The variance ratio test (one of the 3 statistical tests) catches these and rejects the trade.

### Q: What's the difference between a residual and a z-score?

- **Residual** = absolute deviation from fair value, in basis points
- **z-score** = the same deviation, but normalized by how variable this residual usually is

A residual of +5 bp might be huge for one contract and tiny for another. The z-score normalizes — z=+2 means "twice the typical deviation for THIS specific contract".

### Q: I see the same trade emitted by 4 different generators. Is that a coincidence?

No, it's a feature. When multiple analytical lenses (different generators) all identify the same dislocation, the convergence is strong evidence. The "cross-confirmation" bonus in the conviction score rewards this.

### Q: What's the difference between "outright fade" and "PC1 basket"?

- **Outright fade**: bets on ONE contract's residual reverting
- **PC1 basket**: bets on a basket of 4 contracts whose weighted combination loads PURELY on the level factor (no slope/curvature contamination)

The basket is cleaner statistically but harder to execute (4 legs vs 1).

### Q: My screen shows trades with stop > entry (for a BUY trade). Isn't that the wrong direction?

Check that the trade is actually a BUY at the entry. For a listed spread quoted at −1.5 bp with target +0.0 bp and stop +5.0 bp: the BUY direction makes sense because you're betting the spread RISES from −1.5 to 0 (profit) but stops if it falls to +5 (loss). Wait that doesn't sound right.

Actually let me re-check: for entry < 0 and target = 0, the spread is below FV. You'd BUY to bet on it rising. Stop would be ABOVE entry but BELOW target. For entry = −1.5 and target = 0: stop should be at... 3.5 × −1.5 = −5.25 (further below). So the stop is more negative than entry. That's "if it falls further from FV, cut losses".

If you see a "BUY" trade with stop ABOVE entry (e.g. entry 96.34, stop 96.30), that's correct — the stop is below entry (so we'd cut losses if price falls). The display should make this clear.

If anything looks backwards, it's a UI display bug — flag it.

---

<a name="17-glossary"></a>
## 17. Glossary

| Term | Plain English |
|---|---|
| **Anchor slope** | The 12M vs 24M yield spread. A favorite signal for STIR mean-reversion. |
| **Basis points (bp)** | 1/100th of 1 percent. Yields are typically quoted in bp. |
| **Bundle** | 4 packs averaged together. ~4 years of duration. |
| **CME** | Chicago Mercantile Exchange. Where SR3 futures are listed. |
| **CMC** | Constant Maturity Curve. An interpolated yield surface at fixed tenors (1m, 2m, ..., 60m). Anchors the PCA fit. |
| **Conviction** | The engine's overall confidence score for a trade, 0 to 1. |
| **Cross-confirmation** | When multiple independent generators identify the same trade. Strong signal. |
| **Cycle phase** | Where we are in the Fed's tightening/easing cycle. Affects which trade direction is "favored". |
| **DV01 / PV01** | Dollar value of 1 basis point. For SR3 outrights, ~$25/bp/contract. |
| **Fair value (FV)** | What the model says the price/yield "should" be, given the 3-PC factor model. |
| **Fly (butterfly)** | A 3-leg trade with 1,−2,1 weights. Bets on curvature (the belly vs the wings). |
| **Generator** | One of the 19 analytical methods the engine uses to find trades. |
| **Half-life** | Number of trading days for a residual to decay halfway back to zero. |
| **HMM** | Hidden Markov Model. Used to classify days into regimes. |
| **k-NN** | k-Nearest-Neighbours. Used in analog FV — find the 30 most similar historical days. |
| **Level residual** | The cumulated daily change-residual — the actual deviation from FV. |
| **Mean reversion** | The tendency of a stretched value to drift back toward its average. |
| **Outright** | A single SR3 contract for one delivery date. |
| **Pack** | 4 consecutive quarterly SR3 contracts averaged. Whites, Reds, Greens, Blues, Golds. |
| **PCA** | Principal Component Analysis. Discovers the dominant patterns of co-movement in the curve. |
| **PC1/PC2/PC3** | The three factors PCA finds: level (parallel shift), slope (front vs back), curvature (belly vs wings). |
| **Random walk** | A series whose changes are completely uncorrelated; like flipping a coin each day. Doesn't mean-revert. |
| **Residual** | The unexplained portion of today's price move — the part the 3-PC model didn't predict. |
| **SR3** | CME 3-Month SOFR Futures. The instrument this engine trades. |
| **SOFR** | Secured Overnight Financing Rate. The underlying interest rate. |
| **Spread (calendar)** | A 2-leg trade buying one date, selling another. Bets on slope. |
| **Stretched** | A residual with |z| > 1.5. The engine's threshold for trade-worthy deviation. |
| **Tenor** | The time until a contract delivers (in months). |
| **Tick** | The minimum price increment. For SR3: 0.0025 (¼ bp = $6.25 per contract). |
| **Triple-stationarity gate** | The 3-test combination (ADF + KPSS + VR). All 3 must agree for highest-confidence "clean" trade. |
| **Yield curve** | The plot of expected interest rates vs delivery date. |
| **z-score** | How unusual today's value is, in standard-deviation units. |

---

## Closing thought

The engine's job is to find places where the market got the curve wrong. It does this by:

1. Identifying the 3 dominant patterns of curve motion (PC1/PC2/PC3)
2. Measuring how each contract's price differs from the PC-implied fair value
3. Running 3 statistical tests to confirm the residual mean-reverts (isn't just noise)
4. Combining 17 signals into a single 0-1 conviction score
5. Surfacing trades ranked by that score

The trade card walks you through ALL of the above for each specific trade. The narrative paragraph in the expanded view explains, in plain English, exactly why THIS trade was generated.

If a number ever looks wrong to you, the engine's cross-check chip (on every listed spread/fly card) independently reconstructs the price from leg outrights and compares — green = math is consistent end-to-end, red = something to verify.

---

## 18. Trade Horizon Modes — picking your time scale

The screener has three modes. Pick the one that matches **how long you intend to hold a trade**:

| Mode | Hold window | Reversion speed | When to use |
|---|---|---|---|
| **Intraday** | 1-14 days | Very fast (0.5-7d HL) | You scalp or day-trade |
| **Swing** | 3-30 days | Medium (3-30d HL) | You hold for a couple weeks |
| **Positional** ⭐ | **21-120 days** | **Slow (15-90d HL)** | **You hold 1-3 months** ← default for you |

The radio button at the top of the tab switches every parameter at once: how much history we look at, what counts as a "stretched" residual (z-threshold), what counts as a good half-life ("sweet spot"), and how long trades are held. **Default is Positional** because you're a positional trader.

### Why Positional uses 252-day history

A 60-day z-score asks "is this stretched compared to the last quarter?" — useful for swing trades that revert in a couple weeks.

A 252-day z-score asks "is this stretched compared to the last year?" — useful for positional trades that revert over a couple months. The signal needs to persist long enough for you to actually act on it.

### Weekly smoothing (optional, positional only)

If you tick this checkbox, the engine resamples residuals to weekly closes before fitting the z-score / half-life. Reduces noise. Useful if you only re-evaluate positions once a week anyway — daily fluctuations become irrelevant.

---

## 19. Dynamic Exits — what closes your trade

Old approach: you pick an entry, a target, and a stop. They're frozen. Maybe you trail the stop.

New approach: every day, the engine re-evaluates **10 possible exit triggers** (4 targets + 6 stops) and tells you which one (if any) fires today.

### The 4 targets (first one hit closes 100% — or partial — of the trade)

| Tier | When it fires | What happens |
|---|---|---|
| **T1 Full revert** | Residual is back within 0.5σ of mean | Close 100% — full profit taken |
| **T2 Partial revert** | Residual is 33% closer to zero than entry | Close 33% — let the rest ride to T1 |
| **T3 Time target** | Days held ≥ 1.5 × current half-life | Close 100% at market — your time is up |
| **T4 Signal decay** | Confirming sources dropped by half AND held ≥ 5d | Close 50% — signal is fading |

### The 6 stops (any one fires → close 100%)

| Tier | When it fires | Severity |
|---|---|---|
| **S1 Adverse breakout** | Residual moved AGAINST you past 3.5σ | Hard stop — regime break |
| **S2 Triple-gate fail** | Statistical tests now say "no longer mean-reverting" | Hard stop — model broken |
| **S3 HL extension** | Mean-reversion speed has slowed 3× from entry | Soft stop — signal fragile |
| **S4 Convexity warning** | Rate move made convexity bias material | Soft stop |
| **S5 Regime transition** | HMM says we just shifted from "rangebound" to "trending" | Soft stop |
| **S6 Hard P&L stop** | Realized loss ≥ 1.5× expected profit | Always-on backstop |

### Why dynamic > static

A static stop at entry assumes nothing changes. In reality:
- Mean-reversion speed (HL) drifts day-to-day — your time target shifts with it
- New economic data can break the model assumption — triple-gate detects this
- The same loss looks different if the signal degraded vs if the market just whipsawed

The Exit Plan card on every expanded trade shows ALL 10 tiers with their target dates and levels — so you can see the engine's full exit roadmap before opening.

---

## 20. The Backtest — empirical validation

Bottom of the PCA tab is an expandable section: **"📊 Backtest the engine"**.

Click "Run backtest" and the engine replays itself across history:
- For every past trading day, it would have generated some trade ideas
- For each idea above conviction threshold, simulate the trade
- Walk it day-by-day applying the same dynamic exit tiers above
- Record: did it hit T1? T3? S1? How much P&L?

### What you see

- **Summary card**: Total P&L $, hit rate %, Sharpe per trade, max drawdown
- **Equity curve**: cumulative P&L over time — should slope up; flat = no edge; down = engine is wrong
- **Per-source table**: which of the 19 generators actually make money? Sort by Sharpe
- **Conviction calibration**: does higher conviction → higher returns? (it should — if conviction 0.7 trades don't outperform 0.5 trades, the conviction model is broken)

### Two backtest modes

- **Fast** (default): single panel build, walks days quickly. Slightly forward-looking (PCA was fit on data you wouldn't have had).
- **Walk-forward**: rebuilds the PCA at every step. Slow but **the only correct way** to compute true Sharpe. Use it if you actually care about the numbers.

### Why this matters

The backtest's per-source hit rates **feed back into the conviction score** for future trades. After you run a backtest, the `empirical_hit_rate` input in the conviction breakdown for every new trade uses YOUR engine's real historical performance — not a placeholder, not theory.

So the workflow is:
1. Run a backtest weekly (or after big regime changes)
2. The engine learns which generators are working in the current regime
3. New trades get weighted accordingly

---

## 21. Positional Workflow — putting it all together

1. **Open the PCA tab** — mode is already on "Positional"
2. **Filter**: min conviction ≥ 0.55, min hold days ≥ 21d
3. **Look at the top 3 trades** — expand each card and read:
   - The 5-paragraph narrative (paragraph 4 has your contract prices)
   - The **Positional Outlook** card: hold days, FOMC in window, regime confidence
   - The **Exit Plan** card: your T1/T2/T3 targets and S1-S6 stops
4. **Size your position**: `n_contracts = portfolio_risk_$ / S1_distance_$_per_contract` (the engine pre-fills this for you with a default 0.5% portfolio risk)
5. **Execute** the listed contract at the entry price shown
6. **Re-check daily** — the dynamic exit re-evaluates with new data
7. **Periodically (~weekly)** run the backtest to refresh empirical hit rates

You're not trading the engine's output blindly — you're using the engine to:
- **Surface** trades you'd otherwise miss
- **Quantify** risk/reward at entry
- **Tell you when to exit** dynamically

---

*This document is for general orientation. The technical reference is in `PCA_TAB_DOCUMENTATION.md` (the engineering-oriented version). For day-to-day use of the screener, this guide is enough.*
