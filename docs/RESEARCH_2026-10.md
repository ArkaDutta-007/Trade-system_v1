# How to make the system better — research, October 2026

*Arka's question (2026-10-09): research how to make the system better by performance, gain and ability.*
*Everything below was either measured on our own data under the leak-free protocol, or is cited.*

## 1. Bottom line

> **Correction (2026-10-09, 22:00 — read §12 first).** Once the point-in-time data gap (§7) was filled,
> the clean test changed from 13.4%/yr to **−1.3%/yr** (SPY +16.4%). The earlier clean-test result was
> an accident of missing data on stocks that later dropped out of the universe; the model's book has
> **no demonstrated edge** on honest data. Item 1 below is superseded.

1. ~~**The model has a small, real edge**~~ *(superseded by §12)* **The model has a small, real edge — about +3.5–4% a year of alpha over SPY (beta-adjusted), in both
   the 18-year debiased history and the clean 22-month point-in-time test.** Its risk-adjusted return
   (Sharpe ≈ 0.9–1.0) is modestly above SPY's. Against an equal-weight basket of the *same* liquid stocks it
   trails on raw return (14.0% vs 16.4% a year, 2009 →) and wins on risk (Sharpe 0.97 vs 0.88, worst drop
   −27% vs −39%) — and that basket is itself inflated by survivorship. That modest edge is what the
   decisions below protect.
2. **None of the dozen model-level alternatives beats the current model** — debiased training, three published
   anomalies, insider-trading features, sector- or beta-neutral labels, LightGBM, a linear model and
   ensembles. On the clean 22-month test they all sit inside the noise; over 18 debiased years the current
   model has the highest return, and debiased training (−1.4%/yr), sector-neutral labels (−2.4%), LightGBM
   (−1.8%) and insider features (−1.5%) are significantly worse (95% CIs exclude zero). Only the linear
   ridge model comes close (best Sharpe 1.06, lower return, P(better) 0.30) — a candidate for a future
   ensemble test, not an adoption. Two independent tests agreeing makes this a solid "keep what we have".
3. **The book's tuning knobs don't matter, and tuning them would overfit.** Across 324 configurations
   the Probability of Backtest Overfitting is 0.51 — picking the best-looking settings is a coin flip
   out of sample. Production sits mid-plateau.
4. **The safety overlays are worth keeping.** Over 2004–2026 (incl. 2008 and 2022) the production
   overlay has the best Sharpe (0.89) and Calmar (0.42) and the smallest drawdown (−30% vs −54% without);
   its worst stretch was the 2025 V-shaped rebound, which is why the last two years made it look costly.
5. **The biggest remaining lever is data honesty, not modelling.** Our long backtest can't be trusted
   for absolute numbers because the universe is today's survivors. One month of Massive's Developer plan
   (~$79) would provide whole-market history back to 2016 and turn every future decision into a clean test.
6. **"Ability" gains we did deliver today:** the crawler no longer burns a CPU core and 15 GB of RAM
   (rebuild storm fixed), insider-trading data (3.4M trades, 2006 →) and SEC fundamentals are now in the
   system, and LLM calls are 8× faster.
7. **Follow-up the same day (§6):** the legacy engine is retired (daily run ~36 → ~5 min, Saturday ~4 h →
   ~10 min); no weighting rule adds return — a covariance-aware optimiser trades ~20% less volatility for
   the same long-run return but lagged the 2024–26 AI rally, so it stays an option, not a change; the
   fundamental-vs-story news split does not replicate in our liquid names ($0.34 to test); and live
   evidence is now checked daily against rules fixed in advance — a tripwire for a broken model, since
   confirming an edge this size takes years of live data.
8. **Data audit, local models, small RL (§7–§10):** the news history is complete and the features are
   computed correctly, but data depth followed today's top 1,000, so names that later dropped out had
   thin data in point-in-time tests — fixed (crawl of all 1,433 past members running). A 4B local
   decision model (tev1) on our 8 GB GPU labels news as well as the cloud models, but no news score
   predicts subsequent returns, and small RL policies don't beat the simple rules they would replace.

## 2. How everything was tested

Two weeks ago we learned the hard way that a backtest on *today's* 1,000 most-traded stocks rewards a
model for knowing which small stocks would later become big (the ranking objective "won" that test and
lost the honest one). So every idea here was judged twice, on identical folds:

| Test | Universe | Period | What it removes | What it can't |
|---|---|---|---|---|
| **PIT** (point-in-time) | top-1,000 by trailing $volume *that day*, rebuilt daily from whole-market bars (1,441 names incl. those that later faded) | Nov 2024 → Oct 2026 | all selection look-ahead | it's short: small effects are invisible |
| **LONG** (debiased) | our 1,000 names, but only the top-500 by liquidity *that day* | 2008/09 → Oct 2026 | the "small then, big now" look-ahead | names that died before 2024 (no free prices) |

Models: causal walk-forward, refit every 126 sessions on labels that had matured, 21- and 63-day
horizons. Books: production rules (20 names, 8% cap, gates, sector cap, 25% vol brake, 200-day trend
overlay, 35% partial trading) in the research simulator with its cost model. Statistics: per-date rank IC,
paired t on non-overlapping dates, stationary-bootstrap CIs, deflated Sharpe, PBO.

## 3. Results

### 3.1 Model ideas

**PIT test (Nov 2024 → Oct 2026)**

| Variant | IC 21d | IC 63d | Book CAGR | Sharpe | Max DD | Excess vs base (95% CI) |
|---|--:|--:|--:|--:|--:|---|
| **current model** | 0.047 | 0.051 | 13.4% | 0.90 | −12.0% | — |
| debiased training (top-500 that day) | 0.055 | 0.048 | 6.7% | 0.49 | −15.9% | −6.0% [−13.9, +2.1] |
| debiased + new price features | 0.053 | 0.050 | 9.6% | 0.61 | −16.9% | −3.1% [−11.6, +6.0] |
| debiased + sector-neutral label | 0.053 | 0.049 | 8.0% | 0.61 | −13.1% | −5.0% [−10.3, +0.6] |
| debiased + beta-neutral label | 0.017 | 0.015 | 5.6% | 0.88 | −7.6% | −8.0% [−25.5, +7.7] |
| debiased, LightGBM | 0.050 | 0.048 | 5.6% | 0.43 | −16.5% | −7.1% [−14.6, +0.1] |
| debiased, ridge (linear) | 0.041 | 0.073 | 4.0% | 0.57 | −7.4% | −9.5% [−22.2, +3.3] |
| current + new price features | 0.050 | 0.047 | 10.8% | 0.72 | −11.7% | −2.2% [−7.4, +3.6] |
| **current + insider features** | 0.030 | 0.028 | **14.4%** | **0.94** | −11.3% | **+0.9% [−3.5, +6.7]** |
| current + all candidates | 0.034 | 0.031 | 13.7% | 0.85 | −11.9% | +0.5% [−6.2, +7.5] |
| current + ridge ensemble | 0.044 | 0.058 | 7.3% | 0.72 | −9.1% | −6.2% [−15.6, +2.1] |
| *SPY / QQQ / equal-weight S&P (RSP)* | | | 16.4 / 24.0 / 8.8% | 0.99 / 1.09 / 0.64 | | |

**LONG debiased test (2009 → Oct 2026)**

| Variant | IC 21d (paired t) | IC 63d (paired t) | CAGR | Sharpe (95% CI) | Max DD | Excess vs current (95% CI) |
|---|--:|--:|--:|--:|--:|---|
| **current model** | 0.024 | 0.036 | 14.0% | 0.97 [0.50, 1.40] | -26.6% | — |
| debiased training (top-500 that day) | 0.022 (-2.2) | 0.031 (-1.5) | 12.4% | 0.87 [0.43, 1.32] | -27.2% | -1.4% [-2.7%, -0.3%] |
| debiased + new price features | 0.018 (-2.8) | 0.030 (-0.7) | 12.5% | 0.89 [0.48, 1.33] | -28.3% | -1.3% [-2.8%, +0.2%] |
| debiased + sector-neutral label | 0.016 (-2.8) | 0.023 (-2.1) | 11.3% | 0.81 [0.36, 1.29] | -32.4% | -2.4% [-4.2%, -0.7%] |
| debiased + beta-neutral label | 0.016 (-0.8) | 0.010 (-1.0) | 10.2% | 0.98 [0.60, 1.43] | -25.3% | -3.9% [-8.3%, +0.0%] |
| debiased, LightGBM | 0.022 (-1.9) | 0.031 (-1.6) | 12.0% | 0.86 [0.42, 1.29] | -27.0% | -1.8% [-3.1%, -0.5%] |
| debiased, ridge (linear) | 0.024 (+0.1) | 0.041 (+0.9) | 13.4% | 1.06 [0.58, 1.51] | -28.4% | -0.8% [-4.2%, +2.0%] |
| debiased ensemble (XGB+LGBM+ridge) | 0.025 (-0.2) | 0.038 (+0.6) | 13.0% | 0.95 [0.52, 1.39] | -27.9% | -0.9% [-2.5%, +0.5%] |
| current + new price features | 0.021 (-1.4) | 0.034 (+0.1) | 13.8% | 0.95 [0.51, 1.43] | -27.4% | -0.1% [-1.1%, +0.9%] |
| current + insider features | 0.012 (-2.1) | 0.022 (-1.4) | 12.2% | 0.84 [0.39, 1.27] | -28.1% | -1.5% [-2.9%, -0.1%] |
| current + all candidates | 0.009 (-3.2) | 0.020 (-1.0) | 12.1% | 0.84 [0.38, 1.28] | -31.0% | -1.7% [-3.2%, -0.1%] |
| *equal weight of the same eligible names* |  |  | 16.4% | 0.88 [0.42, 1.29] | -38.9% | — |
| *SPY* |  |  | 14.9% | 0.87 [0.43, 1.28] | -33.7% | — |

### 3.2 What the published anomalies look like in our data (2005/10 → 2026, liquid names)

| Signal (source) | Our result | Literature |
|---|---|---|
| Residual momentum (Blitz, Huij & Martens 2011) | 63d IC +0.005, t 0.3 | contested by [Ehsani & Linnainmaa](https://www.nber.org/system/files/working_papers/w25551/w25551.pdf); taper after the late 2000s ([FAJ 2025](https://www.tandfonline.com/doi/full/10.1080/0015198X.2025.2562790)) |
| Industry momentum (Moskowitz & Grinblatt 1999) | IC +0.004, t 0.4 | subsumed by factor momentum (Ehsani & Linnainmaa) |
| Same-month seasonality (Heston & Sadka 2008) | IC −0.016, t −1.5 | — |
| Earnings surprise / SUE (Bernard & Thomas) | rejected 2026-09-26 (paired t −0.3) | post-earnings drift largely arbitraged in large caps |
| Earnings-announcement premium ([Frazzini & Lamont](https://www.nber.org/digest/mar08/stocks-rise-around-earnings-announcements)) | +0.12%/month, t 1.1 | ~0.6%/month in 1972–2004 |
| **Opportunistic insider buying ([Cohen, Malloy & Pomorski 2012](https://www.nber.org/digest/apr11/decoding-inside-information))** | **officer/director buy → +0.35% over 63d (t 3.0); ≥2 buys → +0.45%** | 82 bp/month value-weighted, all stocks |

The pattern matches McLean & Pontiff's finding — US factor returns are 26% lower out of sample and 58% lower
after publication (as summarised in [Jensen, Kelly & Pedersen 2023](https://www.nber.org/system/files/working_papers/w28432/w28432.pdf),
who also show most factors do replicate, just smaller) — and the decay is strongest in the large, liquid names we trade.

### 3.3 Portfolio construction (same signal, different books)

| Book | PIT CAGR / Sharpe / DD | LONG 2009→ CAGR / Sharpe / DD | Alpha (LONG) |
|---|---|---|---|
| **top-20 (production)** | 13.4% / 0.90 / −12.0% | 14.0% / 0.97 / −26.6% | +3.7% |
| top-30 | 9.7% / 0.78 / −10.7% | 12.3% / 0.94 / −25.3% | +2.7% |
| top-50 | 7.7% / 0.71 / −8.4% | 10.7% / 0.92 / −24.1% | +2.1% |
| size tilt (√ADV) | 11.4% / 0.71 / −15.2% | 14.8% / 0.96 / −29.2% | +4.1% |
| no overlays | 26.6% / 1.27 / −13.5% | 17.6% / 0.96 / −34.7% | +3.6% |
| 50% SPY + 50% book | 15.1% / 1.03 / −13.4% | 14.6% / 0.96 / −30.0% | +1.8% |
| long-short 20/20 | −2.2% / 0.10 / −34.5% | 5.6% / 0.34 / −52.6% | +5.5% |
| *SPY* | 16.4% / 0.99 / −18.8% | 14.9% / 0.87 / −33.7% | — |

- **Concentration helps:** the edge lives in the top names (cf. [Cohen, Polk & Silli, "Best Ideas"](https://eprints.lse.ac.uk/24471/1/Best%20ideas(published).pdf);
  the 2026 [agentic-AI nowcasting](https://arxiv.org/abs/2601.11958) paper finds the same: skill "only for identifying top winners").
- **Market-neutral long-short fails:** shorting high-volatility names in a rising market gets squeezed.

### 3.4 Overlays, 2004 → 2026 (includes 2008)

| Overlay | CAGR | Sharpe | Max DD | Calmar | 2008 | 2022 | 2025 tariff + rebound |
|---|--:|--:|--:|--:|--:|--:|--:|
| none | 14.8% | 0.78 | −53.6% | 0.28 | −50.6% | −29.5% | +7.5% |
| SPY 200-day trend, ½ exposure | 13.3% | 0.86 | −37.1% | 0.36 | −29.2% | −28.1% | +1.3% |
| … with −10% drawdown confirmation | 13.7% | 0.84 | −37.1% | 0.37 | −31.6% | −28.1% | +3.5% |
| … with fast (50-day) re-entry | 13.4% | 0.84 | −37.1% | 0.36 | −35.0% | −28.1% | +1.3% |
| graded trend (½ at −10%) | 14.1% | 0.84 | −38.2% | 0.37 | −35.7% | −29.0% | +5.1% |
| SPY realised-vol target 18% | 12.7% | 0.78 | −42.1% | 0.30 | −40.0% | −28.9% | +2.3% |
| **production (EW trend ½ + 25% vol brake)** | **12.8%** | **0.89** | **−30.2%** | **0.42** | **−28.1%** | **−19.7%** | −0.6% |
| *SPY* | 10.9% | 0.65 | −55.2% | 0.20 | −54.8% | −24.1% | +4.0% |

Consistent with the literature: trend filters cut drawdowns at a real cost in V-shaped markets
([60/40 + 200-day filter, 2011–2026: drawdown −21.8% → −17.4%, growth 9.5% → 3.1%](https://www.chat2invest.com/research/60-40-trend-filter)),
and volatility timing alone does not survive out of sample ([Cederburg et al. 2020](https://www.sciencedirect.com/science/article/abs/pii/S0304405X2030132X)).

### 3.5 Book parameter grid (324 configurations, 2004 → 2026)

top-k {10, 15, 20, 30} × cap {6, 8, 12%} × rebalance {10, 21, 42 days} × partial rate {0.2, 0.35, 0.6, 1.0}
× eligible universe {300, 500, 800}. **PBO = 0.51**, median out-of-sample rank of the in-sample winner 0.48.
Sharpe median 0.90 (IQR 0.85–0.96); production 0.885 (43rd percentile). Knobs move CAGR and drawdown
along the same risk line, not Sharpe. The one consistent slope (wider universe → better) is the direction
that admits more future winners from today's survivor list, so it isn't trusted.

## 4. Data and "ability"

| Item | Status | Note |
|---|---|---|
| SEC insider trades (Forms 4, 2006 →) | **added** — `ts data insiders`, 3.4M open-market trades, 17k tickers | CMP routine rule + 10b5-1 flag (2023 →); features in the panel as candidates |
| SEC fundamentals (XBRL, as first reported) | added 2026-10-08 | replaces Massive's retired endpoint |
| News sentiment (Massive LLM "insights") | exists only from mid-2024 | generated near publication → little LLM look-ahead ([Glasserman & Lin](https://arxiv.org/pdf/2309.17322)) |
| News *volume* | 2k articles (2020) → 443k (2022) | a coverage break, not a market change — count features can learn it |
| News *type* | **tested 2026-10-09, not adopted** (§6.3) | 88,700 articles tagged for $0.34; the paper's drift/reversal split adds nothing in our liquid universe |
| Crawler efficiency | **fixed today** | was rebuilding all tables every ~3 min (1 CPU core, 15 GB); now 0.5 GB |
| LLM calls | fixed 2026-10-08 | thinking off: 12 s → 1.5 s, 30× fewer tokens |

## 5. What the evidence says to do

| # | Action | Why | Cost |
|---|---|---|---|
| 1 | **Keep the current model, book and overlays unchanged** | nothing beat it; PBO 0.51 says tuning would fit noise | — |
| 2 | **Wait for live evidence before any further change** — now automated with pre-registered rules (§6.4) | backtest Sharpe barely predicts live Sharpe ([Quantopian, 888 algos: R² < 0.025](https://www.researchgate.net/publication/307553701_All_That_Glitters_Is_Not_Gold_Comparing_Backtest_and_Out-of-Sample_Performance_on_a_Large_Cohort_of_Trading_Algorithms)); first 21d forecasts mature Oct 15, 63d in mid-December | time |
| 3 | **Buy one month of Massive Developer (~$79) for whole-market history 2016 →** | turns the 22-month clean test into a 10-year one; makes debiased training possible | $79 once |
| 4 | ~~Retire the legacy engine from the daily pipeline~~ **done 2026-10-09** (§6.1) | ~34 of the pipeline's 36 minutes and 4 h every Saturday; its books trailed SPY by 7–8% | — |
| 5 | Use insider data as an *event filter or report item*, not as model features | as raw features it made the model worse (−1.5%/yr, CI excludes 0) despite predicting returns on its own (t ≈ 3) | — |
| 6 | ~~Pilot news-type classification (fundamental vs soft)~~ **done 2026-10-09** (§6.3) | most promising new information source in the 2026 literature | $0.34 |
| 7 | Long term: learn portfolio weights net of costs directly ([Jensen, Kelly, Malamud & Pedersen](https://www.aqr.com/Insights/Research/Working-Paper/Machine-Learning-and-the-Implementable-Efficient-Frontier)) | the frontier for implementable ML portfolios | research |

**Don't:** short (long-short lost money), widen the book, tune parameters, add volatility timing, or adopt
any change that hasn't passed the PIT test. Expect live results below backtest.


## 6. Follow-up, 2026-10-09 — acting on §5

### 6.1 Legacy engine retired

Removed from the daily run: `ts daily`, `ts ledger --resolve`, `ts features -u liquid --deep`, the ADR
date trim, BUY signals, future-predict and paper status, the ML-model backtest, `picks_v2` and the raw
`ts picks` ranking. Removed from the Saturday job: the 14-model ensemble retrain (`research/deploy.py`,
4–5 h). The daily run goes from ~36 to ~5 minutes; the weekly from ~4 h to ~10 min. What the legacy
run still did for everyone else — rebuild the liquid-universe price file the books, flag board and regime
layer read — is now `ts ingest -u liquid --no-news` (~20 s; the legacy news + LLM "apprehension" fetch
fed only legacy features).

Paper books: `spy_benchmark`, `momentum` and `alpha_v2` stay active. Momentum is now computed straight
from prices instead of the retired gold features (identical ranking on 2026-10-08: same 357 eligible
names, same top 10, zero difference in the momentum values). The four legacy-model books are frozen at
their last mark (2026-10-08) and listed under the scoreboard: ml_raw −7.9%, ml_v2 −6.9%, ml_v2_gp −7.3%
vs SPY over their own dates; blend (half momentum) +1.9%. The scoreboard's "vs SPY" now uses SPY over
each book's own dates (alpha_v2 started two weeks after the others). The code is still in the repo.

### 6.2 Can smarter weighting gain anything? (same picks, different weights)

Production holds the top 20 at ½ equal + ½ inverse-vol, capped at 8%. Nine weighting rules were
pre-registered (`ops/research/weighting_2026_10.py`) and run with the identical signal, picks, cap,
vol brake and trend overlay — only the split of capital changes. CAGR / Sharpe / max drawdown:

| Weighting | PIT 2024-11 → | LONG 2009 → | FULL 2004 → | Sharpe vs production, LONG · FULL (P>0) |
|---|---|---|---|---|
| **production: ½ equal + ½ inverse-vol** | 13.4% / 0.90 / −12.0% | 14.0% / 0.97 / −26.6% | 12.8% / 0.89 / −30.2% | — |
| equal weight | 13.6% / 0.86 / −12.8% | 14.9% / 0.97 / −27.2% | 14.1% / 0.91 / −30.6% | +0.00 (0.54) · +0.03 (0.89) |
| inverse-vol | 12.7% / 0.90 / −11.5% | 13.0% / 0.95 / −26.9% | 11.7% / 0.85 / −30.6% | −0.02 (0.22) · −0.03 (0.07) |
| inverse-variance | 12.4% / 0.92 / −11.3% | 12.4% / 0.94 / −27.3% | 11.1% / 0.83 / −30.1% | −0.03 (0.25) · −0.05 (0.08) |
| conviction tilt (top pick 2×) | 12.4% / 0.83 / −12.3% | 14.2% / 0.96 / −29.1% | 13.3% / 0.90 / −31.0% | −0.01 (0.33) · +0.01 (0.79) |
| Grinold α/σ² (z/σ) | 13.2% / 0.88 / −12.2% | 14.2% / 0.95 / −29.9% | 13.3% / 0.89 / −31.6% | −0.01 (0.29) · +0.01 (0.68) |
| equal risk contribution (Ledoit–Wolf Σ) | 12.8% / 0.91 / −11.2% | 13.2% / 0.97 / −25.5% | 12.5% / 0.91 / −29.4% | +0.00 (0.58) · +0.02 (0.86) |
| hierarchical risk parity | 11.6% / 0.87 / −11.2% | 12.5% / 0.95 / −25.3% | 11.6% / 0.88 / −30.0% | −0.02 (0.37) · −0.01 (0.40) |
| mean-variance optimiser, top-40, λ = 5 | 9.9% / 0.92 / −8.3% | 12.3% / 1.05 / −24.4% | 12.7% / 1.07 / −27.3% | +0.08 (0.84) · +0.19 (1.00) |

- **No weighting rule adds return.** Equal weight is ~1%/yr ahead at the same Sharpe (more volatility),
  inside the noise; the score-based tilts (conviction, Grinold) change nothing — at an IC of 0.03 the
  score carries too little information to size on, the classic 1/N result
  ([DeMiguel, Garlappi & Uppal 2009](https://academic.oup.com/rfs/article-abstract/22/5/1915/1592901)).
- **A risk-model optimiser buys a smoother ride, not a bigger one.** Mean-variance with a shrunk
  covariance ([Ledoit & Wolf 2004](https://doi.org/10.1016/S0047-259X%2803%2900096-4)) cuts volatility ~20% (14.8% → 11.8%) and
  drawdowns 2–4 points at about the same long-run return; Sharpe +0.08 (2009 →) to +0.19 (2004 →). The
  diagnostics agree in direction for every setting tried (candidate set 20–60, λ 2.5–10: Sharpe +0.01 to
  +0.21 on the long windows). But in the 2024–26 AI-led market it would have cost 3.5–5.5%/yr by
  diversifying away from the concentrated winners, and it misses the pre-registered bar on 2009 →
  (P = 0.84 < 0.95). **Not adopted.** It is a legitimate *preference* — same return, less risk over
  20 years — rather than an improvement; choose it only if a smoother ride matters more than keeping up
  in momentum-led markets.
- The best-looking diagnostic settings (top-60, or λ = 2.5) are not candidates: picking them after
  seeing the results is the overfitting the grid study warned about (PBO 0.25–0.36 across them).

### 6.3 News type: does fundamental news drift while story news reverses?

Every Massive article since July 2024 (88,700 articles, 1,000-name point-in-time universe) was tagged
by qwen3.7-flash with one of 14 event types plus quantified / first-report / rumour flags, following
[Kargarzadeh et al. 2026, "Buy the Rumor, Sell the News"](https://arxiv.org/abs/2608.14014)
(`ops/research/newstype_2026_10.py`; 3,539 calls, **$0.34**). Direction comes from Massive's sentiment,
generated at publication time, so the LLM judges only the article *type* — little room for look-ahead.
Groups were fixed in advance from the paper: HARD = earnings, guidance, dividends/buybacks, analyst
actions (the paper: drifts); SOFT = launches, macro commentary, leadership (the paper: reverses).

Market-adjusted returns in the news direction, 62,164 ticker-day events (t-stats Newey–West by day):

| Group | events | reaction day | days 1–5 | days 2–21 (tradable) | persistence (day 0–20 ÷ day 0) |
|---|--:|--:|--:|--:|--:|
| all news | 62,164 | +0.59% (t 18) | +0.00% | +0.05% (t 0.7) | 1.07 |
| HARD | 8,452 | +1.08% (t 13) | −0.09% | +0.36% (t 1.5) | 1.32 |
| HARD, quantified + first report | 6,729 | +1.32% (t 13) | −0.16% | +0.34% (t 2.1) | — |
| SOFT | 15,200 | +0.59% (t 9.5) | +0.02% | +0.02% (t 0.5) | 1.11 |
| other corporate (M&A, deals, legal, financing) | 11,932 | +0.35% (t 6.5) | −0.02% | −0.35% (t −1.2) | −0.10 |
| recaps, opinion, routine releases | 26,580 | +0.55% (t 14) | +0.03% | +0.15% (t 1.2) | 1.22 |

As cross-sectional signals (net news direction by group over the trailing month, IC vs the next 21
days, PIT window), nothing adds to the model: HARD IC +0.017 (t 1.1), partial IC after the model score
and existing sentiment +0.012 (t 0.7); SOFT +0.019 (t 1.1; positive, not reversing); HARD − SOFT −0.004.

- **The paper's headline does not replicate here.** Its pooled news move is 2.8× larger on the day than
  20 days later (persistence ≈ 0.36); in our liquid names news moves *stick* (1.07). Soft news does
  not reverse. Hard news keeps drifting a little — positive hard news +0.43% over the next month — the
  direction the paper predicts, but at t 1.4–2.1 and too small to move a 20-name book.
- Likely reasons: the 1,000 most liquid US names are where news is priced fastest (the paper's 3,000
  include small caps); ~100× fewer articles; two years of data.
- **Not adopted; no book test** (the pre-registered next step required a significant incremental IC).
  The labels are kept (`data/silver/newstype/labels.parquet`); tagging new articles costs ~$0.01/day if
  the question is ever reopened with a longer history.

### 6.4 Live evidence — automated, with rules fixed in advance

`ts alpha live` (daily digest section; Arka gets an ops alert email once per checkpoint and if the
verdict turns RED or AMBER). Live IC per horizon with overlap-aware standard errors (consecutive daily
21-day forecasts are ~one observation), compared with the research expectation (5d 0.015, 21d 0.030,
63d 0.040). **RED**: live IC ≥ 2 SE below zero over ≥ 2 effective periods → something is broken.
**AMBER**: ≥ 2 SE below the expectation over ≥ 3 effective periods → decay, re-research. **GREEN** otherwise.

What it can and cannot decide. A single date's IC swings by ±0.11–0.15, so telling an IC of 0.03 from
zero at 2σ needs ~70 independent months — **about six years of live data**; the paper book against SPY
needs longer still. Live results therefore cannot *confirm* the edge this year; they can catch a broken
model or data pipeline within weeks, and by March 2027 show whether live skill is grossly below the
research. Checkpoints: first 21-day tally ≈ 2026-10-16 (pipeline check), 2026-11-16, first 63-day
tally ≈ 2026-12-16, **2027-03-15 (first consistency verdict)**, 2027-09-15. As of 2026-10-09: GREEN —
5-day live IC +0.014 vs +0.015 expected (12 matured dates); nothing at 21/63 days has matured yet.

## 7. Is the news data properly ingested? (audit, 2026-10-09)

| Check | Result |
|---|---|
| History per stock | complete — every article Massive has since 2017 for today's top 1,000 (no truncation: AAPL 22,513, NVDA 24,466 articles) |
| Freshness | rolling 14-day refresh per stock + the whole-market daily feed (~500–570 articles a day, far under its 6,000 cap) |
| Feature correctness | the panel's news counts and sentiment equal recounts from the raw articles (e.g. 2026-09-30: AAPL 169 articles / +0.30, NVDA 398 / +0.51), using only articles published before the date |
| Provider coverage | Massive's sources changed: Zacks 65k articles (2023) → ~0 (2025); MarketWatch 8k → 0; Benzinga 18k → 4k. Article *counts* are not comparable across years; sentiment averages much less affected |
| Sentiment | Massive's per-ticker LLM sentiment exists from 2024-07 only (92% of rows since then) |
| **Point-in-time coverage** | **gap found and fixed** (below) |

**The gap.** Data depth followed *today's* top 1,000. The 422 names that were in the point-in-time
universe at some point since late 2024 but have since dropped out had: news sentiment on 38% of days
(vs 72%), fundamentals 64% (vs 89%), earnings-surprise features 46% (vs 78%), short interest 1.7% (vs
99.7%). Inside the PIT universe, *being in today's crawl* predicts the next month's return with IC +0.063
(t 7.4) — so any feature that exists mostly for survivors leaks hindsight into a point-in-time test.

**What it did to earlier results.** It did *not* flatter the model: the model scored the poorly covered
names higher (mean z +0.25 vs −0.03), 17% of its top-20 picks came from them, and its IC inside the
well-covered names is +0.056 vs +0.044 overall — the honest test, if anything, understated it. The news
feature tests (§6.3) were mildly flattered (a bare "has news" indicator has IC +0.023, t 1.8); their
negative conclusions stand.

**Fix (commit 705bf7d).** `massive.pit_members()` = every name that was top-1000 on any day of the bar
window (1,433 names). The crawler now gives all of them full depth (deep news, short interest/volume,
corporate actions, events; ~3,400 calls, ~6 h at 10/min, running), and `ts data fundamentals` fetches
their SEC filings (done: 4,517 tickers). **To do once the crawl finishes:** rebuild the PIT panel and
re-run the clean test; re-test Massive's sentiment as an overlay (partial IC +0.020, t 1.8 beyond the
model score today — borderline and possibly coverage-driven).

## 8. Local "System-1" models on our 8 GB GPU

Setup (no sudo, nothing in the system service changed): Ollama 0.40.2 in user space on port 11435
(`~/opt/ollama-0.40.2`, models in `~/opt/ollama-models`) next to the system Ollama 0.30.11, serving
Together AI's **tev1** decision models (Qwen3.5 fine-tunes, MIT licence) through `/v1/systemone`: send
an article plus typed questions, get back choices with probabilities in one pass. Plus **FinBERT**
(110M encoder, 2019 — no LLM look-ahead possible) and the existing qwen3.7-flash API as a reference.
GPU: RTX 2070 SUPER, 7.6 GiB usable; tev1:4b (4.5 GB) and FinBERT fit together.

| Model | Speed on our GPU | Cost | Notes |
|---|---|---|---|
| FinBERT | **707 articles/s** (all 88,723 in 2 min) | free | sentiment only |
| tev1:0.8b | 4.2–4.9 articles/s (4 questions) | free | routing 25/28 |
| tev1:4b | 1.5 articles/s (4 questions); 0.76/s at 3 questions per stock | free | routing 27/28 |
| qwen3.7-flash (API) | 11,785 pairs in ~4 min, batched | $0.04 | reference |

The decision template is ~2,100 tokens per request around a ~100-token article and requests run one
at a time (parallel requests don't help), so tev1 suits *live* scoring (~500 new articles a day ≈ 6–11
minutes) far better than re-scoring history (88.7k articles ≈ 16–32 hours).

Results (tev1:4b on 2,500 of the 6,987 sample articles = 3,914 article–stock pairs; the other models on
all of them; Massive's sentiment is the reference because it is generated at publication time):

| Sentiment toward the tagged stock | Agrees with Massive (κ) | ρ with the day's move, earnings news (n = 3,166) | pos − neg day-0 move | ρ with days 2–21 |
|---|--:|--:|--:|--:|
| Massive (reference, cloud LLM) | — | +0.189 | +4.04% | +0.014 |
| **tev1:4b (local)** | **70% (0.50)** | **+0.215** | +3.29% | +0.003 |
| tev1:0.8b (local) | 63% (0.38) | — (random sample only) | — | — |
| FinBERT (local) | 55% (0.24) | +0.157 | +2.22% | +0.015 |
| qwen3.7-flash (API) | 45% (0.17) | +0.143 | +2.85% | +0.012 |

- **The local 4B model reads news as well as the cloud models** — best agreement with Massive and the
  tightest fit to the day's move on earnings news. It is also better at *which* company a fact belongs
  to: most of its disagreements with qwen on beat/miss are articles that merely mention a stock (KVUE in
  a J&J story, AAPL in a Goldman piece), where tev1 correctly answers "no results reported for this
  company". The 0.8B is clearly weaker (κ 0.38; its materiality score is near useless).
- **No score predicts what happens next.** Correlation with days 2–21 is ~0 for every model. As a
  monthly feature, FinBERT sentiment (all 88.7k articles) has IC +0.013 (t 1.4) and +0.005 (t 0.8) after
  the model score and Massive's sentiment.
- **No post-earnings drift.** Beats vs misses: day 0 +3.8% (tev1) / +2.5% (qwen), then days 2–21 −0.9%
  (t −1.0) / +0.2% (t 0.2), days 2–63 +1.7% (t 1.3) / +0.2%; guidance raised vs cut likewise
  insignificant. Matches [Martineau 2022](https://cfr.ivo-welch.info/published/papers/martineau2021rest.pdf):
  no drift in large stocks since 2006.
- **Materiality** (tev1 impact score) tracks the size of the day's move (ρ +0.14; Massive's |sentiment|
  +0.16) but not the next week's volatility (ρ ≈ 0.01) — no use for risk sizing.

**Verdict:** a local model can replace the *labelling* (free, private, per-company precise) but adds no
*information* the system lacks. Don't feed local scores into the forecaster. Worth keeping as (1) a
backup if Massive drops its sentiment field, (2) structured per-holding facts for the brief, and (3) the
agent router in §10. The sample run was stopped at 3,481 of 6,987 articles to free the GPU for §11; it is
resumable (`python3 ops/research/system1_2026_10.py tev1`, then `analyze`).


## 9. Small reinforcement-learning models

Our book makes ~12 decisions a year on a weak signal, so the only RL worth testing is small: a few
parameters, trained walk-forward, judged against the simple rule it would replace
(`ops/research/rl_2026_10.py`; all constants fixed before the first run).

**A. Learned exposure policy** — a linear policy or a 5-seed ensemble of 8-unit networks sets the
book's gross exposure (35–100%) each day from market state at the prior close (SPY trend, volatility,
drawdown, VIX, credit spread, yield curve, the book's own recent return/volatility). Exposure doesn't
move future states except through costs, so this is a contextual bandit with a known reward; trained by
deterministic policy gradient on the Sharpe of the exposed book (10 bp per unit of exposure change),
refit every January on all prior years, out of sample 2009 → 2026:

| Exposure rule (2009 → 2026) | CAGR | Sharpe | Max DD | 2022 bear | 2020 crash+rebound |
|---|--:|--:|--:|--:|--:|
| always invested | 17.2% | 0.92 | −38.8% | −24.1% | +4.9% |
| **200-day trend rule (½ below)** | 14.3% | **0.96** | −34.5% | −24.1% | −0.6% |
| learned linear policy (avg gross 0.81) | 11.9% | 0.94 | −26.3% | −15.7% | −4.0% |
| learned 8-unit network ensemble | 12.3% | 0.87 | −31.1% | −9.5% | −5.2% |

Neither beats the trend rule on Sharpe (−0.02 [−0.18, +0.11] and −0.09 [−0.26, +0.10]). They behave
like a more cautious overlay — shallower drawdowns, less return — the same trade the optimiser in §6.2
offers, at more complexity.

**B. Online learning over horizon blends** — six fixed blends of the 5/21/63-day forecasts run as
separate books; each month Hedge (exponential weights), discounted Hedge, follow-the-leader on 36-month
Sharpe, or a contextual leader (same market-trend state) picks the mix from past months only. 2009 →
2026: all within −0.04 to +0.01 Sharpe of the fixed production blend (CIs ±0.05–0.1 around zero). The
blends are too similar (Sharpe 0.84–1.09) for any learner to exploit.

**Where RL already is, and where it could still fit.** The partial-trading rule (35% of the gap per
month) is the closed-form solution of Gârleanu & Pedersen's linear-quadratic dynamic trading problem —
exactly what an RL execution agent would learn under those assumptions; and the calibrator's IC-weighted
horizon blend is a simple online learner. Execution RL only pays with real fills and size (a 2025 review
of 167 studies finds RL's clearest wins in market making, modest ones in portfolio allocation, and that
implementation quality matters more than the algorithm — [Reinforcement Learning in Financial Decision
Making](https://arxiv.org/html/2512.10913v1)). RL fine-tuning of a small LLM on return labels would
bake in look-ahead and overfit at our sample size. **Verdict: no RL component is worth adding now.**

## 10. Where small fast models could fit — map and recommendations

| Pipeline stage | Small-model use | Evidence (this round) | Verdict |
|---|---|---|---|
| News sentiment (direction) | score each article and stock locally | see §8: local scores match the day's move about as well as Massive's; no score predicts the following month | keep Massive's in the model; a local tev1 score is a free backup if Massive drops its sentiment field (as it dropped financials) |
| Earnings beat/miss, guidance | structured extraction from news | no post-earnings drift in our liquid names 2024–26, by any labeller — consistent with [Martineau 2022](https://cfr.ivo-welch.info/published/papers/martineau2021rest.pdf) (no drift in large stocks since 2006) and the [microcap-only revival](https://anderson-review.ucla.edu/is-post-earnings-announcement-drift-a-thing-again/) | report item for holdings, not alpha |
| Materiality / novelty | flag big news | predicts the size of the day's move weakly; not the next week's volatility | no |
| Agent routing (Silas on Telegram) | message → fixed tool, with probabilities | tev1:0.8b 25/28 correct in 0.12 s, tev1:4b 27/28; the prompt-injection message went to "other"; low-confidence cases can fall back to Qwen | **most promising** — reliability, privacy, zero cost; needs OpenClaw wiring and your OK (it brings back a local Ollama dependency) |
| Fidelity screenshot reading | local vision model instead of the cloud API | not tested | worth testing for privacy — holdings screenshots currently go to a cloud API |
| Ops alerts | triage noise | the channel-level de-duplication already fixed the July duplicate-alert problem | no |
| Regime layer | count oil-shock / war / AI-capex headlines | not tested; FRED already supplies the state | low |
| Filings (8-K / 10-K text) | event classification, change detection | "Lazy Prices" replications are weak; needs a new data pipeline | not now |

## 11. tev1:4b as a decision-maker next to the algorithms (2026-10-09, evening)

The quant models rank stocks; several other decisions in the system are qualitative and were made by hand
or not at all. Each pilot gave tev1:4b (local, on the 8 GB GPU) the relevant text and a typed question,
and checked the answer against something objective (`ops/research/tev1_decisions_2026_10.py`).

| Decision | Checked against | Result |
|---|---|---|
| **FOMC decision** (cut / hold / hike) from that day's headlines | FRED DFEDTARU | **18 / 18** since 2024-07, incl. the Sep-2026 hike; tone calls plausible (Dec-2024 "hawkish cut") |
| **Playbook earnings switches** from the company's own release (SEC 8-K item 2.02) | the release text | facts right: META revenue +28% (+27% constant currency) → "≥25%" yes 0.99, bear case no 0.08; CRM/NOW beat + raise. "vs expectations" switches (UBER weak guide, MU gap-up) need the price reaction too — the release says what was raised, not whether it beat consensus |
| same switches from news articles | the release / the move | unreliable — the per-ticker news is mostly opinion pieces (META judged from previews) |
| **Hormuz switch** (oil-supply monitor over daily macro headlines) | the headlines | fired 2026-03-02 "Iran war shuts the Strait of Hormuz" (p 0.99), again 03-19/03-25 and 07-23 (tanker attacks). **The playbook's `hormuz_closed` (forces O=RED) stayed false by hand** |
| **CRWV equity-raise kill switch** over CRWV news since June | the headlines | fired once: 2026-09-17, "at-the-market offering of up to 35 million shares" + $3B converts (p 0.93; stock −5.3% vs SPY). **The rule said SELL same day; the switch stayed false by hand** |
| chip-export-control monitor | the headlines | top days are real: 2025-01-13 (AI diffusion rule), 2025-11-09 (Blackwell exports blocked); fires on 2.6% of days |
| Taiwan-escalation monitor | — | fired on 1 of 806 days (p 0.57) — quiet, as the period was |
| **Prompt-injection screen** for untrusted headlines | 15 planted attacks + 400 real headlines | **15 / 15 caught, 0 / 400 false alarms** |
| **Pre-buy screen** of the clean test's 420 picks (last 30 days of the stock's news) | next 21 days' return | see below |

Pre-buy screen ("ok / caution / avoid to open a position now", plus takeover / distress / binary-event /
guidance-cut flags), clean test Nov 2024 → Sep 2026:

| | ok | caution | avoid | book with "avoid" vetoed |
|---|--:|--:|--:|---|
| company named | +3.6% (175) | +2.1% (56) | −0.3% (39) | 13.4% → 16.0%/yr, Sharpe 0.90 → 1.07 (P 0.88) |
| **company hidden** (look-ahead check) | +3.1% (172) | +2.8% (58) | +1.2% (40) | 13.4% → 14.6%/yr, Sharpe 0.90 → 0.99 (P 0.71) |

Hiding the company's name halves the effect — a 2025-26 model partly *remembers* these companies
([Glasserman & Lin](https://arxiv.org/pdf/2309.17322)). What survives masking is the fact-based flag:
picks with a guidance cut in the prior 30 days returned −2.7% vs +3.4%. So: promising, unproven, and only
live use can settle it, because live the model cannot know the outcome.

**Integration (informational).** `ts alpha judge` runs every morning in the digest ("🧠 tev1 judgments",
~25 s): playbook switches and thesis-break monitors from `configs/tev1_switches.yaml` (editable; keyword
pre-filter for recall, tev1 for precision), and the pre-buy / holding screen of today's picks, the alpha_v2
book and the real portfolio if present. It never changes a book or a config — switches are *proposals* —
and logs every answer to `data/ledger/tev1_judgments.parquet` for live scoring. The Telegram brief lists
any FIRING switch/monitor and any "avoid" on a held name. Served by the system Ollama, upgraded to
0.40.2 on 2026-10-10 (`~/ops/bin/upgrade-ollama`); the temporary user-space copy on :11435 was removed.

**Not yet built, worth considering:** Fed-day decision/tone feeding the F flag automatically; earnings
switches from 8-K releases when the playbook gets new events (the June playbook's horizon ended
2026-09-30); a shadow paper book that vetoes "avoid" picks (live A/B of the screen); the Telegram router
(§10) and screening of untrusted text before the agent reads it.

## 12. Correction: with complete point-in-time data the clean test fails (2026-10-09, 22:00)

After the crawler had filled the data of the 422 names that were in the point-in-time universe but have
since dropped out (§7), the clean test (Nov 2024 → Oct 2026, whole-market point-in-time top-1000) was
re-run. Same code and protocol; only the data of those names changed (short interest 1.7% → ~97% of
their days; news history complete; fundamentals from SEC EDGAR instead of Massive).

| Clean test, production book (top-20, overlays, partial trading) | CAGR | Sharpe | Max DD | IC21 / IC63 |
|---|--:|--:|--:|--:|
| published (old panel, old models) | 13.4% | 0.90 | −12.0% | 0.047 / 0.051 |
| old panel, today's models | 11.5% | 0.79 | −10.7% | 0.049 / 0.052 |
| **complete data, today's models** | **−1.3%** | **−0.01** | −18.5% | 0.039 / 0.038 |
| … dropped-out names' old fundamentals restored | 0.2% | 0.09 | −18.8% | 0.039 / 0.040 |
| … dropped-out names' old (missing) short + news restored | 10.2% | 0.71 | −11.7% | 0.048 / 0.049 |
| … model trained without short features | 0.5% | 0.10 | −17.0% | 0.029 / 0.029 |
| … without short and news features | 1.0% | 0.14 | −13.8% | 0.031 / 0.036 |
| complete data, book limited to the 300 most liquid names that day | 8.3% | 0.65 | −14.2% | — |
| *SPY / RSP (equal-weight S&P 500)* | *16.4% / 8.8%* | *0.99 / 0.64* | | |

**What happened.** The model likes beaten-down names (cheap, falling, heavily shorted). It learned that
from a training panel of *today's survivors* (1998 →), in which beaten-down names that did not recover
are absent — so "beaten down" looks like a buying opportunity. In the clean test the names that later
failed had missing short and news data, and the trees' routing of missing values happened to push their
scores down. With the data filled in, those names score high (mean z +0.23 → +0.66), make up 43% of the
picks instead of 15%, and lose. The IC barely moves (0.047 → 0.039: the bulk of the ranking is still
mildly right) but the top of the ranking — where the book lives — is dominated by future failures.
Ruled out as the cause: wrong SEC company mapping (321 of 391 names mapped, names agree), the EDGAR
fundamentals themselves (restoring the old ones changes nothing), and the book's gates.

**Consequences.**
- The production book has **no demonstrated edge on honest data**. Every absolute clean-test number
  in §3–§6 was measured on the gappy panel and is too high; relative comparisons there (model variants,
  weightings, overlays) were all made under the same artifact and should be re-checked before reuse.
- The root cause is the training universe, not the code: a model trained only on survivors learns
  survivor relationships. The fix is training on point-in-time universes that include the failures —
  whole-market history, which Massive's free tier has for 2 years only. **The ~$79 one-month Developer
  plan (10 years of whole-market bars) moves from "useful" to "required"** for any honest model.
- A point-in-time *large-cap* restriction (the 300 most liquid names that day) avoids much of the damage
  (8.3%/yr) — but it was chosen after seeing this result, and it is still below SPY.
- Live, the alpha_v2 book is exposed to the same mechanism (today's names, complete data).

## 13. Free fixes, no data budget (2026-10-10)

Whole-market history costs money (Massive Developer ~$79; MarketParquet's archive $79 one-time; Norgate
$35/month with a 21-day trial). Free routes: **RIT's Saunders College lists CRSP through WRDS** — the
academic survivorship-free standard (every US stock since 1925, delisting returns included); ask RIT Libraries
(Jennifer Freer, jlfwml@rit.edu) and check that WRDS's academic licence covers this use. Meanwhile three
fixes that cost nothing, settings fixed before the run (`ops/research/pit_fix_2026_10.py`), honest panel:

| Clean test, Nov 2024 → Oct 2026 | production universe | 300 most liquid that day |
|---|---|---|
| production model | −1.2% / Sharpe −0.01 | 8.4% / 0.66 |
| monotone constraints from published anomaly signs | −1.1% / −0.00 | 7.3% / 0.60 |
| **point-in-time training where it exists** (Nov 2024 →, failures included) | **6.8% / 0.48** (+8.0%/yr vs production [+1.6, +16.4]) | **10.6% / 0.75** |
| both | 5.4% / 0.40 | 7.0% / 0.53 |
| 12-1 momentum, no model | 1.3% / 0.16 | 4.6% / 0.37 |
| low volatility, no model | 2.4% / 0.49 (max DD −5.4%) | 2.0% / 0.31 |
| *SPY / QQQ / RSP* | *16.7% / 24.3% / 9.1%* | |

- **The fix that works is the training data, not the model.** Letting the model learn from the
  point-in-time universe — including the names that later failed — where we have it (our own free
  whole-market bars, Sep 2024 →) recovers +8%/yr against production, significantly, and the gain grows
  through the window as that data accumulates. The crawler adds whole-market bars every day, so this
  training set grows for free.
- Forcing literature signs on the model doesn't help: the damage is in which names look "cheap and
  beaten down", not in a few features' signs.
- **Nothing here beats the index in this window.** 2024-26 was an AI mega-cap market (QQQ +24%/yr); the
  best honest variant (point-in-time training, large caps) made 10.6% vs SPY 16.7% and RSP 9.1%.
- Recommendation: switch the production training to point-in-time rows where they exist and restrict the
  book to the 300 most liquid names that day — tested here after the failure was seen, so it must earn its
  keep live (shadow first) before anyone relies on it.

*Scripts: `ops/research/lab_2026_10.py`, `construction_2026_10.py`, `overlay_2026_10.py`, `grid_2026_10.py`,
`weighting_2026_10.py`, `newstype_2026_10.py`, `system1_2026_10.py`, `rl_2026_10.py`, `tev1_decisions_2026_10.py`, `pit_fix_2026_10.py`. Raw results:
`reports/alpha/lab_2026_10/*.json`.*
