# How to make the system better — research, October 2026

*Arka's question (2026-10-09): research how to make the system better by performance, gain and ability.*
*Everything below was either measured on our own data under the leak-free protocol, or is cited.*

## 1. Bottom line

1. **The model has a small, real edge — about +3.5–4% a year of alpha over SPY (beta-adjusted), in both
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
| News *type* | not used | [fundamental news drifts for weeks, soft news reverses](https://arxiv.org/abs/2608.14014) — classifiable with qwen3.7-flash for ~$5 |
| Crawler efficiency | **fixed today** | was rebuilding all tables every ~3 min (1 CPU core, 15 GB); now 0.5 GB |
| LLM calls | fixed 2026-10-08 | thinking off: 12 s → 1.5 s, 30× fewer tokens |

## 5. What the evidence says to do

| # | Action | Why | Cost |
|---|---|---|---|
| 1 | **Keep the current model, book and overlays unchanged** | nothing beat it; PBO 0.51 says tuning would fit noise | — |
| 2 | **Wait for live evidence before any further change** | backtest Sharpe barely predicts live Sharpe ([Quantopian, 888 algos: R² < 0.025](https://www.researchgate.net/publication/307553701_All_That_Glitters_Is_Not_Gold_Comparing_Backtest_and_Out-of-Sample_Performance_on_a_Large_Cohort_of_Trading_Algorithms)); first 21d forecasts mature Oct 15, 63d in mid-December | time |
| 3 | **Buy one month of Massive Developer (~$79) for whole-market history 2016 →** | turns the 22-month clean test into a 10-year one; makes debiased training possible | $79 once |
| 4 | Retire the legacy engine from the daily pipeline | ~45–60 of the pipeline's minutes; its paper books trail (ml_raw −6%, ml_v2 −5%) | your call |
| 5 | Use insider data as an *event filter or report item*, not as model features | as raw features it made the model worse (−1.5%/yr, CI excludes 0) despite predicting returns on its own (t ≈ 3) | — |
| 6 | Pilot news-type classification (fundamental vs soft) on 2022 → | most promising new information source in the 2026 literature | ~$5–10 |
| 7 | Long term: learn portfolio weights net of costs directly ([Jensen, Kelly, Malamud & Pedersen](https://www.aqr.com/Insights/Research/Working-Paper/Machine-Learning-and-the-Implementable-Efficient-Frontier)) | the frontier for implementable ML portfolios | research |

**Don't:** short (long-short lost money), widen the book, tune parameters, add volatility timing, or adopt
any change that hasn't passed the PIT test. Expect live results below backtest.

*Scripts: `ops/research/lab_2026_10.py`, `construction_2026_10.py`, `overlay_2026_10.py`, `grid_2026_10.py`.
Raw results: `reports/alpha/lab_2026_10/*.json`.*
