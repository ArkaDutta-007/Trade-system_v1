# How to make the system better — second research pass, November 2026

*Arka's request (2026-11-04): another pass on the latest literature and computing methods, and on how more
resources and better algorithms could make the system much better.*
*Everything here was measured on our own data with settings fixed before the first run, or is cited. Paper
books only; nothing in this report changes production. Code: `ops/research/*_2026_11.py`. Results:
`reports/alpha/lab_2026_11/`.*

## 1. Bottom line

1. **More compute is not what is holding us back.** All 13 model comparisons ran on the existing RTX 2070 in
   27 minutes. A five-seed neural network ensemble did not beat XGBoost in the book, and blending four engines
   made the book worse. The limits are two honest years of data and a weak signal.
2. **The best new idea is holding the market minus the worst names, not picking 20 winners.** Cap-weighting the
   300 most liquid names and dropping the worst 30% by model score beat the same index without the drop on all
   four rebalance days tested, by +3.0 to +6.7%/yr, at about 4× turnover a year. Nearly all of that gain came in
   year 1: year 2 was only +0.5%. It deserves a paper trial, not real money.
3. **The best training recipe is now known, and it corrects October.** Point-in-time rows plus training on that
   day's top 500 lifted the production-universe top-20 book from −0.9% to +8.7%/yr (median across start days).
   Point-in-time training alone did not: October's +6.8% was mostly rebalance-timing luck (§2).
4. **No top-20 book beat SPY or QQQ over the full two years.** SPY made +14.2%/yr and QQQ +17.6%/yr. The
   XGBoost top-20 books in §3 all lost money in year 2 (−3.6% to −11.4%) while SPY made +11.8%.
5. **30%/yr has no honest route here.** The highest returns in these long-history tests needed 3× leverage:
   16.9%/yr on the US market and 20.0%/yr on the Nasdaq, with drawdowns of −85% and −94%. Even then, a 10-year
   stretch reached 30%/yr only 13–14% of the time.
6. **The biggest free upgrade is data.** Open Source Asset Pricing now publishes 209 stock-level signals and all
   its portfolio returns under CC BY 4.0. Pair those signals with CRSP returns through RIT's WRDS and we could run
   an honest test over decades instead of two years. Whether RIT's licence covers this use still has to be checked.

## 2. Corrections to earlier results

| what | before | after | why |
|---|---|---|---|
| Whole-market price table | rolled old days off; started 2024-10-11 | keeps every cached day; starts 2024-09-27 and accumulates | `build_raw_bars` trimmed to the plan's 2-year window, so days would have become unreachable. Raw JSON was intact, so nothing was lost |
| Point-in-time training alone (RESEARCH_2026-10 §13) | +6.8%/yr | **median −0.8%** (range −2.5% … +3.6%) | Moving the start date by 0–18 sessions changes the answer. The 21-day book is sensitive to its rebalance day |
| "Debiasing did not help" (October batch 1) | no gain | **best model** | That verdict came from the gappy panel |
| Literature composite (this pass) | +12.5%, best book | **+3.8%** (median +5.3%) | A single ±inf in `leverage` made 87% of the scores NaN, so the book picked names in arbitrary order. Caught by a suspiciously identical result, fixed in `panel_2026_11.z` |
| Distress filter (this pass) | no effect | **no help** (§3) | Same bug |

From now on any book result should be quoted as a median across start days, with the range. The new
`panel_2026_11.py phase` does this.

## 3. Stock-level results on the refreshed honest panel

Panel: 1,434 names, failures included, 2024-11-21 → 2026-11-03. Walk-forward with 63-session refits. S = trained
on today's survivors (production). P = survivors before the panel starts, then point-in-time rows. Debias = train
only on that day's top 500. Year 1 ends 2025-10-31.

### Production book (top 20), seven start days

| model · universe | median CAGR | range | year 1 | year 2 |
|---|---|---|---|---|
| S-xgb · production universe (**today's production**) | −0.9% | −3.6 … +5.5 | +9.0% | −11.4% |
| S-xgb · top-300 | +3.7% | +1.9 … +5.1 | +14.0% | −3.6% |
| P-xgb · production universe | −0.8% | −2.5 … +3.6 | +19.2% | −10.7% |
| P-xgb · top-300 | +9.2% | +7.6 … +10.2 | +24.4% | −4.3% |
| **P-xgb-debias · production universe** | **+8.7%** | +4.9 … +10.6 | +28.3% | −7.1% |
| **P-xgb-debias · top-300** | **+11.3%** | +9.7 … +13.4 | +26.2% | −5.3% |
| literature composite · top-300 | +5.3% | +3.2 … +6.1 | −1.5% | +9.0% |
| SPY / QQQ / RSP (equal-weight S&P) | +14.2 / +17.6 / +10.4% | | | |

The 36 top-20 books fail the overfitting checks: probability of backtest overfitting 0.82, and the deflated
Sharpe of the best one is 0.61. Choosing the winner from that table would not be evidence of skill. What is
consistent is the universe: all 16 trained models and blends did better in the top-300 than in the production
universe.

### Index minus the worst names (monthly, 10 bps, 300 most liquid names, cap-weighted)

| book | CAGR | by rebalance day | year 1 | year 2 | Sharpe | vs control (95% CI) |
|---|---|---|---|---|---|---|
| Control: all 300 (correlation with SPY .98) | +16.1% | +14.2 … +16.1 | +21.8% | +9.4% | .91 | |
| **Drop worst 30% by P-xgb-debias** | **+22.8%** | +17.2 … +22.8 | +34.9% | +9.9% | 1.18 | **+5.7% (+1.3, +10.5)** |
| Drop worst 30% by P-xgb | +20.0% | +18.3 … +20.0 | +30.9% | +8.4% | 1.07 | +3.4% (−1.1, +8.0) |
| Drop worst 30% by S-xgb | +18.5% | +16.5 … +19.3 | +27.1% | +9.1% | 1.03 | +2.1% (−1.5, +5.8) |
| Drop worst 30% by ML + literature blend (post hoc) | +18.9% | +14.8 … +18.9 | +25.8% | +10.9% | 1.07 | +2.3% (−0.7, +5.5) |
| Keep best 30% by P-xgb-debias | +21.1% | +15.8 … +22.0 | +42.0% | +1.7% | .97 | +5.1% (−7.2, +14.9) |
| SPY | +13.9% | | +14.9% | +11.8% | .87 | |

Reading this table:

- **Dropping the worst is more stable than picking the best.** Keeping the best 30% had a much wider range and
  fell apart in year 2 (+1.7% vs the control's +9.4%). Dropping the worst stayed level with the control in year 2.
- **Drawdown did not improve much:** −18.5% vs −20.5% for the control and −18.8% for SPY.
- **Year 2 is the warning:** +9.9% trails SPY's +11.8%. Most of the excess is year 1.
- **The long record agrees** (§4): since 2015 the short legs of published signals have lagged the market more
  reliably than the long legs have beaten it.

### Signal strength by year (rank IC)

| model | 21d year 1 | 21d year 2 | 63d year 1 | 63d year 2 |
|---|---|---|---|---|
| S-xgb (production) | +.057 | +.010 | +.084 | −.033 |
| P-xgb-debias | +.105 | +.018 | +.120 | −.012 |
| P-mlp (five-seed network) | +.048 | +.020 | +.105 | +.005 |
| Literature composite (no training) | −.004 | +.044 | +.012 | +.069 |
| ML + literature blend (post hoc) | +.058 | +.034 | +.087 | +.048 |

Every ML model decayed sharply in year 2. The blend is the only score that stayed clearly positive at both
horizons in both years (smallest value +.034), but it was added after seeing the results, so it is a hypothesis
to watch live, not a finding.

### Where the signal sits (21-day return vs universe, by score decile, %)

| | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---|---|---|---|---|---|---|---|---|---|
| P-xgb, top-300, year 1 | −2.8 | −2.4 | −1.3 | −.5 | +.1 | −.1 | .0 | +1.1 | +2.3 | +3.5 |
| P-xgb, top-300, year 2 | .0 | +.4 | +.4 | +.2 | −.7 | −.2 | .0 | +.1 | +.7 | −.9 |

The production model's top decile leans towards names about to drop out: 22% have left the point-in-time
universe within 63 sessions, against 2% in the bottom decile. Debiased point-in-time training cuts that to 13%. This
matches [Avramov, Cheng & Metzker](https://papers.ssrn.com/abstract=3450322), who find ML profits concentrate in
hard-to-arbitrage and distressed names and shrink once those are excluded.

### What did not help

- **Neural networks:** the five-seed MLP held up better on 63-day IC in year 2 (+.005 vs −.032 for P-xgb) but
  gave no better book (+6.9% vs +9.5% in the top-300).
- **Engine blends:** P-all4 returned −0.4%/yr in the production universe.
- **Failure-inclusive signal weights:** weighting the literature composite by trailing Open Source Asset Pricing
  Sharpes (`lit-osap`) was worse than equal weights on IC (+.006 vs +.020) and in the top-300 book (+0.4% vs +3.8%),
  though better in the production universe (+6.7% vs +3.4%). Mixed at best.
- **Distress filter:** dropping the worst 10–20% by a Campbell-Hilscher-Szilagyi style rank trimmed drawdown by
  2–4 points and did not help return. S-xgb stayed at −1.8%; P-xgb fell from +3.3% to −1.9%.

## 4. What about 100 years of data that includes the failures?

Sources: Ken French daily data 1926 → 2026-09 and [Open Source Asset Pricing](https://huggingface.co/datasets/OpenSourceAP/CrossSection)
release 2026.10, both built from CRSP with failed stocks included. Code: `longrun_2026_11.py`.

### Published signals have stopped paying on the long side

Value-weighted decile portfolios, 179 signals, annual excess over the market, median signal:

| era | long leg | share beating market | short leg | share lagging market |
|---|---|---|---|---|
| 1926–2004 | +2.9% | 87% | −1.6% | 81% |
| 2015–2025 | −1.0% | 43% | −2.8% | 74% |
| 2021–2025 | +0.2% | 52% | −2.7% | 68% |
| After each signal's publication | −0.1% | 49% | −1.7% | 76% |

Timing the signals (hold the 10 long legs with the best trailing year, only among signals published by then)
made 13.2%/yr vs the market's 10.8%, but with a lower Sharpe (.68 vs .74). Since 2015: 14.3% vs 13.4%, Sharpe .72
vs .89. Published factor momentum is stronger ([Gupta & Kelly](https://www.institutionalinvestor.com/article/2bswk7762zwxielme1t6o/portfolio/factor-momentum-is-real-researchers-argue)),
but it is long-short. "Timing the factor zoo" (Neuhierl et al., 2023) also reports only modest out-of-sample gains.

A caution on arithmetic: in a cap-weighted portfolio each decile holds only a modest slice of the market, so
avoiding one signal's short leg is worth little. The gain in §3 comes from dropping 30% of names on a composite.

### Trend and leverage

Hold L× the market above its 200-day average, T-bills below. The signal is one day old when traded. Leverage pays
T-bills + 0.5% plus a 0.9% fee, and each change costs 5 bps. Simulated returns slightly understate the real funds
(SSO 14.9% vs real 15.7%, UPRO 30.0% vs real 32.7%).

| rule | CAGR | Sharpe | max drawdown | worst 10 yrs | 10 yrs ≥ 20%/yr | 10 yrs ≥ 30%/yr | since 2005 |
|---|---|---|---|---|---|---|---|
| US market 1× always | 9.8% | .63 | −84% | −8.5% | 0% | 0% | 11.1% |
| US market 1× + trend | 9.9% | .89 | −40% | +0.1% | 0% | 0% | 8.1% |
| US market 2× + trend | 13.7% | .68 | −69% | −3.5% | 17% | 0% | 11.1% |
| US market 3× + trend | 16.9% | .63 | −85% | −8.3% | 35% | 14% | 13.4% |
| US market 3× always | 9.7% | .44 | −100% | −44.2% | 36% | 18% | 13.8% |
| Nasdaq 1× + trend (1971 →) | 12.4% | .93 | −52% | −0.1% | 1% | 0% | |
| Nasdaq 2× + trend | 16.9% | .71 | −81% | −6.3% | 32% | 2% | |
| Nasdaq 3× + trend | 20.0% | .66 | −94% | −13.9% | 45% | 13% | |

Trend at 1× keeps the return and halves the drawdown over a century, as
[Gayed & Bilello](https://www.cxoadvisory.com/volatility-effects/leveraging-the-u-s-stock-market-based-on-sma-rules/)
and [AQR's century study](https://www.TrendFollowing.com/whitepaper/Century_Evidence_Trend_Following.pdf) report.
Since 2005 it has cost 3%/yr. The −85% and −94% rows mean a leveraged trend rule is not a safe road to high returns,
and the Nasdaq rows are chosen with hindsight. They use the price index plus an assumed 1%/yr dividend.

### Rotation

| rule | full history | market | since 2015 | market |
|---|---|---|---|---|
| 49 industries, top 5 by 12-1 momentum | 16.2% (Sharpe .78) | 10.1% (.62) | 15.1% (.79) | 13.6% (.91) |
| 10 industries, top 3 | 13.0% (.76) | 10.1% (.62) | 14.9% (.85) | 13.6% (.91) |
| Nine sector ETFs, top 3 (1999 →) | 9.1% (.66) | 8.2% (.60) | 13.2% (.94) | 13.5% (.93) |
| Faber GTAA5 (after 2007 publication) | 5.1% (.79) | 11.1% (.76) | | |
| Antonacci dual momentum (after 2014 publication) | 7.5% (.63) | 13.5% (.93) | | |

Industry momentum is a real long-run effect that has given little since 2015, and 49 industries are not directly
investable. The two tactical ETF rules both lagged SPY on return after publication.

## 5. What the recent literature adds

- **LLM trading backtests shrink under honest evaluation.** [FINSABER](https://arxiv.org/html/2505.07078v5) finds
  LLM advantages shrink over 20 years and 100+ symbols, and recommends trend detection and regime-aware risk
  control. Other work documents survivorship in LLM finance evaluation
  ([2603.27539](https://arxiv.org/pdf/2603.27539), [2602.14233](https://arxiv.org/pdf/2602.14233),
  [2607.12067](https://arxiv.org/pdf/2607.12067)). Our October 13.4% → −1.3% correction is the same story.
- **Complexity is contested.** [Kelly, Malamud & Zhou](https://www.aqr.com/insights/research/journal-article/the-virtue-of-complexity-in-return-prediction?aqrPDF=1)
  argue more parameters help; [Nagel](https://ideas.repec.org/p/nbr/nberwo/34104.html) shows that the method reduces
  to volatility-timed momentum in short windows. Our neural network result gives no reason to scale model size.
- **Trading costs belong in the objective.** [Jensen, Kelly, Malamud & Pedersen](https://papers.ssrn.com/abstract=4187217)
  show that ignoring them pushes ML portfolios into fleeting small-scale characteristics. The index-minus book is a
  crude version of the same idea.
- **Tabular deep learning has caught up in general benchmarks.** [TabArena](https://arxiv.org/pdf/2506.16791v1) and
  [RealMLP](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2ee1c87245956e3eaa71aaba5f5753eb-Abstract.html)
  say tuned networks and ensembles can match boosted trees. That did not translate to our low signal-to-noise problem.
- **Foundation models are unproven for returns.** [Kronos](https://arxiv.org/abs/2508.02739) reports RankIC +93% over
  other foundation models, but I found no independent replication of that claim.
  [Re(Visiting) Time Series Foundation Models in Finance](https://arxiv.org/html/2511.18578v1) found negative
  out-of-sample R² even for the simple baselines. Kronos was **not run** here (§6).
- **LLM alpha mining.** [RD-Agent(Q)](https://arxiv.org/html/2505.15155v2) reports IC .053 on CSI 300, while
  [AlphaCrafter](https://arxiv.org/pdf/2605.05580) reports stable ICs of only about .02 and a declining RD-Agent.
  The Vibe-Trading alpha zoo decayed the same way on our panel: IC +.069 → −.015.
- **This market is hard for stock pickers.** [BofA figures reported in October](https://www.roic.ai/news/bofa-active-funds-struggle-as-megacaps-dominate-10-05-2026)
  had only 44% of large-cap funds beating benchmarks in September, with the cap-weighted S&P ahead of equal weight
  by nearly 5 points. [Allspring](https://www.allspringglobal.com/insights/articles/ai-at-a-crossroads-concentration-capex-and-what-comes-next/)
  puts the Magnificent 7 at about 32% of the index. A book that holds mega-caps at market weight avoids this bet.

## 6. Resources: what is worth getting

| resource | cost | what it would give | status |
|---|---|---|---|
| **[Open Source Asset Pricing](https://huggingface.co/datasets/OpenSourceAP/CrossSection)** portfolios | free, CC BY 4.0 | failure-inclusive portfolio returns, monthly and daily, 1926–2025 | **Monthly files downloaded and in use** |
| Open Source AP stock-level signals (209, 1.9 GB) | free | a decades-long ML training set, but it has no returns or tickers, and Price, Size and STreversal are withheld | Needs CRSP returns to be useful |
| **WRDS/CRSP through RIT** | free if granted | the same test we run now, over decades, with delisting returns | [Saunders lists CRSP](https://www.rit.edu/business/faculty-and-staff-resources). Student access and licence for this use are unconfirmed. Ask RIT Libraries first |
| [Ken French data](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) | free | daily market, factors and industries since 1926 | **In use** |
| S&P 500 membership history ([fja05680/sp500](https://github.com/fja05680/sp500)) + Yahoo | free | a longer point-in-time large-cap test | **Not good enough.** 432 of 977 past members are missing from our store and Yahoo still has 135. The rest are mostly the failures and takeovers that matter |
| [Tiingo free tier](https://www.quantstart.com/articles/evaluating-data-coverage-with-tiingo/) | free key, 500 symbols/month | might cover the missing 297 names | Delisted coverage unconfirmed; test with one failed ticker |
| Our own cached whole-market days | free | one more honest day every day | **Now kept** (§2) |
| [RIT SPORC cluster](https://research-computing.git-pages.rit.edu/docs/about_us.html) | free to request | 164 GPUs, including 100 A100s | Not needed for tabular work. Useful only for fine-tuning a large model |
| [NSF ACCESS](https://allocations.access-ci.org/allocations-policy) | credits | national clusters | Graduate students can qualify with an advisor. Not needed now |

**Kronos was not run.** The repository and its one dependency are at `~/Desktop/Kronos`, but the safety check
blocked running downloaded code that had not been requested by name. If Arka approves it, the pre-specified test is
Kronos-small on 120-session bars, 21 sessions ahead, 5 sampled paths, at each month-end from 2025-03, judged by IC
and a top-20 book, and treating only dates from 2025-09 as out of sample for its weights.

## 7. Recommendations

Ranked by value for effort. None of these has been adopted.

1. **Start a paper book for "top-300, cap-weighted, minus the worst 30%", scored by P-xgb-debias.** Run it alongside
   `alpha_v2` so live results can decide. This replaces the top-300 proposal from October.
2. **Change the training recipe in that paper book only:** point-in-time rows plus top-500 training. Keep
   production as it is until live evidence accumulates; the next checkpoint is 2026-11-16.
3. **Ask RIT Libraries about WRDS student access and whether it covers this use.** This is the one item that could
   change what we can know. Contact: Jennifer Freer, jlfwml@rit.edu.
4. **Quote books as medians across start days** and report year-by-year splits.
5. **Watch the ML + literature blend live.** It is the only score clearly positive in both years, and it was found
   after looking at the data.
6. **Don't spend effort on** larger models, more engines, hyperparameter search, tactical ETF rules, or more compute.
7. **Be realistic about the goal.** A sensible target is the market plus a few percent, with the same or lower risk.
   Leverage can raise that, at the cost of drawdowns above 80%.

## 8. Limits

- Two honest years is a very short sample. Most confidence intervals here span several percentage points.
- More than 60 books were examined, so the best number in any table is flattered by selection.
- The cap-weight control uses a size estimate where market cap was missing, and it is not SPY: it tracked SPY with
  correlation .98 and 3.8% tracking error, and it holds more mega-cap weight.
- Year 1 favoured ML and year 2 did not; we cannot yet say which is typical.
- Open Source Asset Pricing data end in December 2025, so the signal weights for 2026 are stale.

## 9. Reproduce

```bash
python ops/research/lab_2026_11.py             # refreshed panel
python ops/research/longrun_2026_11.py         # §4
python ops/research/panel_2026_11.py walk      # 13 models, 27 minutes on the RTX 2070
python ops/research/panel_2026_11.py eval      # books, IC, overfitting checks
python ops/research/panel_2026_11.py phase     # rebalance-timing check
python ops/research/tilts_2026_11.py           # deciles, tilts, distress
python ops/research/tilts_2026_11.py posthoc   # ML + literature blend
```

`data/external/` (Ken French, Open Source AP, S&P lists) is git-ignored, so download it again on another machine.
