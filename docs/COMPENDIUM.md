# Trading-System Compendium

A complete reference for the system: architecture, data flow, every feature and
its formula, the cross-domain mathematics, the forecasting methodology, and the
operational gotchas. Companion to the [README](../README.md) and the auto-generated
[command directory](COMMANDS.md).

> Research / paper-trading system. Not financial advice. Live execution should
> follow months of paper trading and slippage validation.

---

## Table of contents

1. [Philosophy & design principles](#1-philosophy--design-principles)
2. [Repository map](#2-repository-map)
3. [Data architecture (medallion)](#3-data-architecture-medallion)
4. [Ingestion layer](#4-ingestion-layer)
5. [Feature reserve — the catalog](#5-feature-reserve--the-catalog)
6. [Cross-domain nonlinear mathematics](#6-cross-domain-nonlinear-mathematics)
7. [Random Matrix Theory](#7-random-matrix-theory)
8. [Sequence models (RNN / LSTM / GRU)](#8-sequence-models-rnn--lstm--gru)
9. [Forecasting methodology](#9-forecasting-methodology)
10. [Probabilistic price bounds](#10-probabilistic-price-bounds)
11. [Decision pipeline](#11-decision-pipeline)
12. [Playbook engine](#12-playbook-engine)
13. [Backtesting & leakage tests](#13-backtesting--leakage-tests)
13a. [Alpha engine v2 (`alpha/`, V5.2)](#13a-alpha-engine-v2-alpha-v52--forecast-ledger-continuous-learning-causal-backtest)
14. [LLM layer (DeepSeek + RAG)](#14-llm-layer-deepseek--rag)
15. [Compute & hardware layer](#15-compute--hardware-layer)
16. [Parallelism map](#16-parallelism-map)
17. [Command reference](#17-command-reference)
18. [Operational gotchas](#18-operational-gotchas)
19. [Glossary](#19-glossary)

---

## 1. Philosophy & design principles

* **Local-first, free-data.** Everything runs on a laptop against free APIs
  (yfinance, FRED, Finnhub/NewsData, SEC EDGAR). No paid feeds required.
* **Leakage is the cardinal sin.** Every feature is point-in-time/causal; every
  evaluation uses purged+embargoed CV; three leakage tests gate strategy
  promotion; a label-shuffle gate guards each forecaster.
* **Interpretability first.** Tree models + SHAP before black boxes. Sequence
  models are *opt-in* and must beat trees on honest CV to be selected.
* **Calibrated uncertainty over point forecasts.** A decision ships a *band*
  (conformal 90% coverage) and a distribution, not a single number.
* **Rigorous transfer of mathematics.** Estimators are borrowed from physics,
  physiology, climate science, hydrology — each with a known-answer test so the
  borrowing is verified, not vibes.

---

## 2. Repository map

```
src/trading_system/
├── ingestion/      market_data, macro_fred, calendar_events, sec_filings,
│                   news_events (+ finnhub_news, newsdata_news, google_news_fetcher,
│                   dedup, rag), edgar_discovery (spinoffs/IPOs), llm_extractor, realtime
├── features/       technical, extended_features, regimes, macro, event_features,
│                   nonlinear, nonlinear_panel, nonlinear_report, rmt,
│                   reserve (catalog), context (macro inputs), build
├── models/         ensemble, train, validation (purged CV), forecast_train,
│                   sequence (RNN/LSTM/GRU), intervals (conformal), implied_vol,
│                   store, predict, shap_analysis, model_registry
├── decision/       analyze, bounds, explain, groundings, report,
│                   invest (budget→plan), longterm (picks), discover (moonshots)
├── flags/          O/F/I/S/C live flag board + composite regime
├── playbook/       standing rules, compliance, cycles, blotter, briefing
├── portfolio/      sizing (incl. distribution-based), risk, order_policy
├── backtesting/    vectorized, event_driven, slippage, metrics, engines/
├── execution/      paper_broker, tranches (portfolio tabs), live_broker (stub)
├── monitoring/     drift, pnl_attribution, alerts, shap_viz, ledger (decision ledger)
├── quality/        data_checks, leakage
├── pipeline/       daily flow
├── storage/        DuckDB + Parquet helpers
├── utils/          logging, compute (CPU/MPS/CUDA), progress (track/parallel_map)
└── cli.py          `ts` entry point  ·  config.py
models_store/       committed best models (forecast/<h>d + intervals + manifest)
```

---

## 3. Data architecture (medallion)

A classic bronze → silver → gold lakehouse on Parquet + DuckDB:

| Layer | Path | Contents |
| --- | --- | --- |
| **raw** | `data/raw/` | untouched API payloads (optional) |
| **bronze** | `data/bronze/ohlcv_daily.parquet` | tidy long-form OHLCV, **sanitized** (incomplete bars dropped) |
| **silver** | `data/silver/` | `events.parquet` (news), `apprehension_scores.parquet`, `macro_cache/`, `news_cache/`, `iv_cache/` |
| **gold** | `data/gold/features.parquet` | the full panel feature matrix (one row per ticker×date) |

`gold/` is the single input to training and decisions. `models_store/` (tracked
in git) holds the promoted models.

---

## 4. Ingestion layer

### OHLCV (`market_data.py`)
Threaded yfinance fetch (default 8 workers, `--workers`/`TS_INGEST_WORKERS`) with
a live progress bar. `sanitize_ohlcv()` drops today's in-progress bar (null close,
half-formed range) and any impossible/non-positive OHLC before the bronze write,
so a few API glitches can't fail the quality gate.

### Massive / ex-Polygon (`massive.py`, V5.1) — whole-market EOD + reference data
`data.source: auto` (default) makes `ingest_universe` use Massive when
`MASSIVE_API_KEY` is set *and* the local cache is backfilled, else yfinance.
The free plan is **5 req/min, EOD, 2 years of history**, so the design is:

* **Cross-process rate limiter** — a file-locked sliding window
  (`data/raw/massive/.ratelimit.json`) shared by cron and every shell; 429s
  still back off on `Retry-After`. Budget ≈ 7 200 calls/day.
* **Grouped daily bars** `/v2/aggs/grouped/…/{date}` — *every* US stock for one
  date in one call, cached **unadjusted** and forever (immutable truth).
  Backfill = ~520 calls ≈ 100 min; the daily top-up is 1–3 calls.
* **Corporate actions** (splits, dividends by month), **ticker directory**,
  per-ticker **overview** (SIC, market cap, shares), **fundamentals**
  (`/vX/reference/financials`, every `*.value` leaf flattened) and **tagged news
  with LLM sentiment** (`/v2/reference/news` insights). Bronze outputs:
  `data/bronze/massive/{ohlcv_all,splits,dividends,tickers,details,financials,news}.parquet`
  plus the long tail `events` (symbol changes), `related`, `ipos`, `exchanges`,
  `ticker_types`, `fed_*` (treasury yields, inflation, expectations),
  `short_interest`, `short_volume` — every field kept (`flat_rows`).
* **Adjustment maths, CRSP-style, recomputed at build time:**
  `s_t = ∏_{splits d>t} from/to`, `f_t = ∏_{ex-dates d>t} (1 − cash_d/close_{d−1})`;
  `close = raw·s_t` (split-adjusted = yfinance *Close*), `adj_close = close·f_t`
  (= yfinance *Adj Close*), volume ÷ `s_t`.
* **Splice** — per ticker, yfinance rows before the Massive window are rescaled
  at the first common date (`recent/deep` ratio for both price and adj series)
  so `bronze/ohlcv_daily.parquet` is continuous 1995→today; universe names
  missing from the feed (or with <20 Massive days) stay yfinance-only.
* **Survivorship-free universes** — `ts massive universe` ranks the *whole*
  market by trailing-63d median dollar volume (CS/ADRC only, no OTC) into a
  universe YAML; `ohlcv_all.parquet` keeps delisted names for research.

* **Multiple keys** — `MASSIVE_API_KEY`, `MASSIVE_API_KEY2…9` each carry their
  own 5/min at the server, so the client keeps one limiter file per key
  (`.ratelimit-<sha-fingerprint>.json`) and round-robins: N keys ≈ N×5 req/min.
  A 429 cools only that key (the next request goes out on another one), a 401
  retires the key for the process, a 403 is a `MassiveEntitlementError` (plan
  limit, not key) and the crawler parks that endpoint family for a week.
  Tickers are stored as `BRK-B` but sent as `BRK.B` (`api_ticker`).
* **Continuous crawler** (`Crawler`, `ts massive crawl`, systemd user unit
  `massive-crawler.service`) keeps the budget saturated with cache-planned work
  in priority tiers, re-checked before *every* call: 0 newest bars once published
  (D+1 01:30 UTC), current corp actions, today's news → 1 two-year grouped
  backfill (newest first) → 2 directory, holidays, exchanges/types, monthly
  splits/dividends, IPOs, Fed series, short interest/volume → 3 universe depth:
  overview, fundamentals, 2y news backfill + rolling news, ticker events, related
  companies — **at full depth**: every article Massive has (archive starts
  2017-04, 1 call/1000, 365d TTL), complete dividend + split histories
  (per ticker, 180d), short interest since 2017-12 (14d), daily FINRA short
  volume since 2024-02 (30d), fundamentals since 2009 with the SEC
  `acceptance_datetime` of every filing → 4 **universe intraday**: unadjusted
  1-minute bars (pre/regular/post, `session` + New-York `date` columns) for the
  plan's 2-year window, one immutable doc per ticker-month (`aggs_minute/{t}/
  {YYYY-MM}`, the open month refreshes daily, a month fetched before its last
  day published is re-fetched), built incrementally into
  `bronze/massive/bars_minute/{ticker}.parquet` — ≈8 200 calls once, then
  ≈330/day → 5 **extended**: the top-`extended_top`
  (1000) US stocks/ADRs by trailing-63d dollar volume get the tier-3 depth (their
  YAML lands in `bronze/massive/universe_extended.yaml`, refreshed hourly) →
  6 whole market by liquidity rank (45d/14d/30d) → 7 QA: Massive's own adjusted
  series vs our `close` (`adjustment_qa`, in `ts massive status`). A task that
  errors is backed off 6h → 24h → 7d instead of retried (a 403 on the oldest
  minute month — the plan edge — is treated the same way rather than parking
  the family); SIGTERM stops after the current call; state in
  `data/raw/massive/crawler_state.json`. Not on the free plan (probed
  2026-09-21, 403): trades/quotes, Benzinga earnings calendar / analyst ratings
  / guidance — the nearest free proxies for "event time" are news
  `published_utc`, filing `acceptance_datetime` and the minute bars.
* **Directory history** (tier 8, lowest priority, 2026-09-26): `/v3/reference/tickers?date=` on the
  first weekday of every month since 2008 (empty before ~2008 on the free plan) →
  `bronze/massive/tickers_history.parquet` (snapshot, ticker, type, exchange, CIK, FIGI). ~8.3k
  securities per 2008 snapshot incl. LEH/BSC/WM/CFC — the point-in-time universe membership needed
  to measure survivorship. Historical `tickers/{t}?date=` snapshots are NOT usable (market cap is
  today's price × old shares). Re-probed 2026-09-26: minute AND daily bars older than 2 years are
  403 on the free plan; 10y needs Stocks Developer ($79/mo), 20y+ Advanced ($199/mo), both with
  flat files (whole-market daily files — one month's subscription would be enough to backfill).
* **Deep prices** — the plan hard-caps bars at 2 years (a request from 2000
  returns 2024-09→, verified), so `ohlcv_deep.parquet` (universe + extended,
  `source` column) is built weekly in a crawler background thread from yfinance
  since `data.massive.deep_start` (1970) and spliced onto the Massive window
  with `splice_history`. Paid tiers lift the cap (Starter 5y, Developer 10y,
  Advanced 20y+); set `history_years` accordingly and the splice shrinks.

CLI: `ts massive status | crawl | backfill | update | build | universe`; the daily
pipeline runs `ts massive update -u liquid` before `ts daily` (a no-op when the
crawler has already fetched everything — they share the limiter). Tests:
`tests/unit/test_massive.py` (limiter, cache/TTL, pagination, 429/auth,
adjustment vs hand calcs, splice continuity, budget caps) — all network-free.

### News (`news_events.py`) — pluggable backends
Tried in order; the first that returns rows for a ticker wins (no double-count):

1. **Finnhub** `company-news` — *symbol-tagged* (no false positives), free 60/min.
2. **Massive** `/v2/reference/news` — symbol-tagged, whole market in ≤ 4 cached
   calls per 6 h, ships per-ticker LLM sentiment (used instead of the lexicon).
3. **NewsData.io** `/latest` — keyword (`"<ticker> stock"`), free ~200 credits/day.
4. **Google News RSS** — broad free-text breadth + full-article body extraction.
5. **NewsAPI** — headline fallback (key dead since 2026-09; not in the default list).

Articles are disk-cached, and **near-duplicate wire stories are collapsed** by
token-set Jaccard (`dedup.py`) before scoring. All timestamps coerce to UTC
datetimes regardless of source/cache representation.

### Macro (`macro_fred.py` + `flags/datafeed.py`)
Resilient three-tier FRED feed: official API → keyless `fredgraph.csv` → disk
cache. Series: 10y/2y/3m yields, 2s10s curve, fed funds, CPI, unemployment, VIX,
HY OAS. Powers both the flag board and the macro **feature levels**.

### Calendars (`calendar_events.py`)
Earnings dates (threaded yfinance, needs `lxml`) + FRED economic-release schedule
(CPI/PPI/NFP/Retail/GDP) + a hard-coded FOMC schedule. Drives `days_to_earnings`,
`days_to_fomc`, `macro_event_imminent`.

### SEC filings (`sec_filings.py`)
EDGAR submissions JSON (10-K/10-Q/8-K), serial (rate-limited) with a progress bar.

### RAG (`rag.py`)
Point-in-time retrieval over stored events: filter to `known_at ≤ as_of`
(leakage-safe), rank by relevance to a finance query (sklearn TF-IDF default,
optional BGE embeddings via `sentence-transformers`), return top-k snippets to
ground the LLM.

---

## 5. Feature reserve — the catalog

The **reserve** (`features/reserve.py`) is the single source of truth for what the
forecasters can select. `resolve_reserve(df, groups, min_non_null_frac=0.6)`
intersects the catalog with columns actually present *and* well-covered (≥60%
non-null) — so a forecaster never references a missing or mostly-empty column.
`build_feature_matrix` assembles everything in one causal pass.

| Group | Features | Notes |
| --- | --- | --- |
| **trend** | `mom_{5,10,20,60,120}d`, `mom_12m1m`, `sma_gap_{10,20,50,200}`, `mom_accel`, `dist_52w_high/low`, `bb_pctb_20`, `breakout_20`, `breakdown_20` | momentum, MA gaps, 12-1 momentum, Bollinger %b |
| **volatility** | `vol_{10,20,60}d`, `atr_14`, `downside_vol_{20,60}d`, `vol_of_vol_60`, `ret_skew_60`, `ret_kurt_60`, `max_dd_252`, `dd_from_high_60` | realised/semi vol, skew, kurtosis, drawdowns |
| **liquidity** | `rel_vol_20`, `avg_dollar_volume_20`, `amihud_illiq_20`, `volume_z_60`, `overnight_gap` | microstructure / liquidity |
| **mean_reversion** | `rsi_14` | |
| **regime** | `bull_regime`, `high_vol_regime`, `mom_20d_rank`, `excess_ret_1d`, `beta_60`, `corr_bench_60` | cross-sectional + benchmark-relative |
| **macro** | `macro_{ust_10y,yield_curve,vix,hy_oas,fed_funds}` + `_chg_20d` + `_z_252` | FRED levels, 20d change, 1y z-score |
| **events** | `event_count`, `event_sentiment_mean`, `event_magnitude_mean`, `event_novelty_max`, `risk_flag_count`, `sent_decay_{3,7,14}d`, `sent_momentum`, `apprehension_score` | news/LLM-derived |
| **calendar** | `days_to_fomc`, `days_to_earnings`, `macro_event_imminent` | event proximity |
| **fractal / entropy / chaos / tail / earlywarning** | see §6 | nonlinear dynamics |
| **rmt** | `rmt_systematic_frac`, `rmt_market_beta` | see §7 |

**Selected formulas (extended_features.py):**

* Amihud illiquidity: `amihud = mean( |r_t| / dollar_volume_t )` over 20d — price
  impact per dollar traded.
* Downside vol: `√252 · std( r_t · 1[r_t<0] )` — only the left tail's dispersion.
* Rolling beta: `β = Cov(r_i, r_b) / Var(r_b)` over 60d; `corr_bench` the
  correlation. `r_b` is the benchmark daily return joined by date.
* 12-1 momentum: `P_{t-21}/P_{t-252} − 1` — classic momentum skipping the most
  recent (mean-reverting) month.

All rolling ops shift before any forward-looking operation; targets
(`forward_return_*`, `fwd_ret_*`) are computed separately and **never** enter the
feature set.

---

## 6. Cross-domain nonlinear mathematics

The crux of the "weirdly transferable" idea: rigorous estimators from other
fields, applied causally to log-price / log-return / a 5-day vol proxy. Each lives
in `features/nonlinear.py`, has a **known-answer test**, and is registered in an
`NLFeature(name, source, window, stride, func, group)` table so the catalog can
never drift from what's computed. Two tiers: **FAST** (default) and **DEEP**
(`--deep`, the O(W²)/fit-based ones).

Panel integration (`nonlinear_panel.py`): each estimator is recomputed every
`stride` trading days on a trailing `window` and **forward-filled** in between
(strictly causal — a Hurst exponent barely moves day to day, and forward-fill
only ever carries a *past* value). Per-ticker work fans out over a spawn process
pool. Numpy `NaN` is converted to a polars **null** (`fill_nan(None)`) so coverage
gating and `drop_nulls` behave and NaN never leaks into tabular models.

### 6.1 Fractal geometry / long memory

**Hurst exponent — DFA** (`hurst_dfa`, source: chaos theory / geophysics).
Detrended Fluctuation Analysis. Integrate the mean-subtracted series
`Y(k)=Σ_{i≤k}(x_i − x̄)`; split into windows of size `n`; fit & remove a local
polynomial trend; the fluctuation `F(n)=√⟨residual²⟩` scales as `F(n) ∝ n^H`. `H`
is the slope of `log F(n)` vs `log n`.
*Reading:* `H>0.5` persistent/trending, `H=0.5` random walk, `H<0.5`
mean-reverting. Robust to non-stationary trends (its advantage over R/S).

**Hurst exponent — R/S** (`hurst_rs`, source: hydrology — Hurst's Nile-flood
study). Rescaled range: for window `n`, `R = max − min` of the cumulative
deviation, `S = std`; `⟨R/S⟩ ∝ n^H`. Same reading as DFA; kept as a
cross-check (agreement → trustworthy memory estimate).

**Higuchi fractal dimension** (`higuchi_fd`, source: signal processing). Builds
`k` interleaved sub-series, measures curve length `L(k) ∝ k^{−D}`; `D` = slope of
`log L(k)` vs `log(1/k)`, with `D∈[1,2]`. `D→1` smooth/clean trend, `D→2`
space-filling/jagged. A geometric complementary to Hurst (`D ≈ 2−H`).

**Rough-volatility Hurst** (`rough_volatility_hurst`, source: rough-volatility /
fractional Brownian motion, Gatheral–Jaisson–Rosenbaum). Scaling of the `q`-th
absolute moment of vol increments: `m(q,Δ)=⟨|σ_{t+Δ}−σ_t|^q⟩ ∝ Δ^{qH}`; `H` from
the slope of `log m` vs `log Δ` divided by `q`. Empirically `H≈0.1` for real
markets ("volatility is rough"); the system reproduces ≈0.13 on NVDA.

### 6.2 Information theory / complexity

**Permutation entropy** (`permutation_entropy`, source: dynamical systems / EEG,
Bandt–Pompe). Embed in `m` dimensions; replace each window by the *ordinal
pattern* (rank permutation) of its values; Shannon entropy of the pattern
histogram normalised by `log(m!)`. Robust to monotone transforms; `→1`
near-random, low = structured order.

**Sample entropy** (`sample_entropy`, source: cardiology — heart-rate
variability). `SampEn = −ln(A/B)`, `B` = count of length-`m` template pairs
matching within tolerance `r·σ`, `A` = same for length `m+1`. Lower = more regular
/ self-similar. Bias-free vs approximate entropy.

**Spectral entropy** (`spectral_entropy`, source: information theory). Normalise
the power spectral density to a probability distribution; Shannon entropy /
`log(n_freq)`. Low = cyclical (a dominant frequency), high = broadband/noisy.
`dominant_period` returns the peak-PSD period (in days).

**Wavelet HF ratio** (`wavelet_hf_ratio`, source: wavelet analysis). Share of
energy in the highest-frequency detail bands of a Haar decomposition —
choppiness vs smoothness without assuming stationarity.

### 6.3 Chaos / predictability

**Largest Lyapunov exponent** (`largest_lyapunov`, source: chaos theory,
Rosenstein algorithm). Time-delay embed; for each point find its nearest neighbour
(excluding a Theiler window to avoid temporal correlation); track the mean
log-divergence `⟨ln d_i(t)⟩`; `λ` = slope over the initial linear region. `λ>0` =
sensitive dependence on initial conditions (chaotic), bounding the predictability
horizon `≈1/λ`.

**Recurrence quantification (RQA)** (`recurrence_metrics`/`recurrence_determinism`,
source: nonlinear dynamics / recurrence plots). Recurrence matrix
`R_ij = Θ(ε − ‖x_i − x_j‖)` at a recurrence-rate target. **Determinism** = fraction
of recurrence points lying on diagonal lines ≥ `l_min` (predictable structure);
**laminarity** = vertical lines (intermittency/sticking). High DET = deterministic
(forecastable), low = stochastic.

**0–1 test for chaos** (`chaos01`, source: Gottwald–Melbourne). Drive
`p(n)=Σx(j)cos(jc)`, `q(n)=Σx(j)sin(jc)` for random `c`; the mean-square
displacement `M(n)` grows *linearly* for chaos and stays *bounded* for regular
dynamics; `K∈[0,1]` is its asymptotic growth rate (correlation with `n`). `K≈1`
chaotic, `K≈0` periodic. Operates directly on the scalar series (no embedding).

### 6.4 Bifurcation theory / early-warning — *the transferable gem*

**Critical slowing down** (`early_warning_score`, source: ecology & climate
tipping points). As a dynamical system approaches a bifurcation (tipping point),
it recovers ever more slowly from perturbations, so lag-1 autocorrelation `AR(1)→1`
and variance rise *together*. The score is the **Kendall-τ trend** of `AR(1)` and
rolling variance across overlapping sub-windows, combined. Borrowed from
detecting lake-eutrophication / climate regime shifts → here, rising fragility /
crash risk. (`ar1` is the standalone lag-1 autocorrelation.)

### 6.5 Extreme value theory / econophysics

**Hill tail index** (`hill_tail_index`, source: extreme value theory /
econophysics). On the `k` largest absolute returns (order statistics `X_{(1)}≥…`),
`α = [ (1/k) Σ_{i=1}^{k} ln(X_{(i)}/X_{(k+1)}) ]^{−1}`. `α` is the power-law tail
exponent: lower `α` = fatter tails. Equities sit near `α≈3` ("inverse cubic
law"); `α<2.5` flags extreme-move-prone names (infinite variance territory at
`α<2`).

### 6.6 Catastrophe / rupture physics

**LPPLS bubble confidence** (`lppls_confidence`, source: Sornette, log-periodic
power-law singularity — material rupture & earthquake physics). A bubble is
*faster-than-exponential* growth decorated with accelerating log-periodic
oscillations toward a critical time `t_c`:

```
ln p(t) = A + B (t_c − t)^m + C (t_c − t)^m cos( ω·ln(t_c − t) − φ )
```

Fits are sampled over candidate `t_c`/`ω`; **confidence is the improvement over a
constant-growth (linear-in-log-price) baseline** — not raw R² (a flexible
4-parameter model overfits anything, so raw R² is a *bad* bubble detector; this
baseline-relative score cleanly separates a real bubble ≈0.99 from a noisy trend
≈0.0). Positive = bubble signature, strongly negative = anti-bubble / crash-spike
risk.

---

## 7. Random Matrix Theory

`features/rmt.py`, source: Wigner's nuclear-physics spectral theory → finance
(Laloux–Bouchaud–Potters, Plerou et al.).

A sample correlation matrix of `N` assets over `T` days is *mostly noise*. If
entries were iid, its eigenvalues would fall inside the **Marchenko–Pastur band**

```
[λ−, λ+] = (1 ∓ √q)²,   q = N / T   (unit variance)
```

Eigenvalues poking **above** `λ+` are statistically real collective modes — the
market factor (giant top eigenvalue) and sectors. Two causal, strided
(every 5d, 252d window) features:

* **`rmt_systematic_frac`** (by date): `Σ λ_i[λ_i>λ+] / Σ λ_i` — fraction of
  cross-sectional variance in real modes. High = correlated "everything moves
  together" tape (fragile, low diversification); low = stock-pickers' market.
* **`rmt_market_beta`** (by ticker×date): loading on the dominant eigenvector,
  scaled by `√λ_top`, sign-oriented so the market mode is positive — an
  RMT-cleaned market exposure that doesn't need an index proxy.

Known-answer: a strong common factor → `systematic_frac≈0.82` with all-positive
betas; pure noise → `≈0.00`.

---

## 8. Sequence models (RNN / LSTM / GRU)

`models/sequence.py`. Where tree models treat each `(ticker,date)` row as
independent, a recurrent net reads the **ordered lookback window** and learns
temporal structure (path dependence, vol clustering, momentum exhaustion) itself.

* **Causal windowing** (`build_sequence_tensor`): for row `i`, the window is the
  `lookback` rows of the same ticker ending at `i`, edge-padded when history is
  short. Built from the same null-free, ticker/date-sorted frame as the flat
  matrix `X`, so purged-split indices slice both identically. Strictly backward.
* **Architecture** (`_make_net`): `RNN/LSTM/GRU → LayerNorm → Linear → GELU →
  Dropout → Linear(1)` on the last hidden state. Huber (SmoothL1) loss for fat
  tails; AdamW; gradient clipping; random early-stopping holdout (regularisation
  only — reported metrics come from the outer purged CV).
* **Hardware**: device from the compute profile (`cuda`/`mps`/`cpu`), all
  float32 (MPS has no float64).
* **Persistence**: pickles the best `state_dict` + architecture tuple (~10 KB),
  not the live device-bound module — portable into `models_store/`.
* **Opt-in**: `pip install -e '.[deep]'`; torch import is lazy so the rest of the
  system never depends on it.

They compete head-to-head with trees under the *same* CV/ICIR/leakage gate — no
special-casing the winner.

---

## 9. Forecasting methodology

`models/forecast_train.py`, `models/validation.py`. `ts train-forecast` trains and
honestly evaluates per-horizon forecasters {21, 63, 126, 252d}.

### 9.1 Purged + embargoed walk-forward CV
A training row at date `d` carries a label spanning `[d, d+h]`. If `d+h` reaches
into the test block, the model has seen the future. For `h=252` that overlap is a
*year* wide. Fix (López de Prado):

* **Expanding walk-forward**: train on the past, test forward in `n_splits` blocks.
* **Purge**: drop training rows with `d > test_start − h`.
* **Embargo**: drop an extra `embargo_days` buffer before each test block so
  slow-decaying autocorrelation can't bleed across the seam.

`coverage_no_overlap()` asserts (in tests) that no training label window reaches
into its test block.

### 9.2 Metrics
* **IC** (information coefficient): the **per-date cross-sectional** Spearman rank
  correlation — computed *within each date* across the universe, then averaged
  over dates. This is deliberate: pooling all `(ticker, date)` pairs conflates
  *market timing* with *stock selection* — a feature that's constant per date
  (a macro level, the RMT systematic fraction) correlates with the date-level
  forward return, so pooled IC is inflated by the common market factor (and
  survives label shuffling). Per-date IC isolates genuine cross-sectional skill.
  *(In practice the correction is large: a pooled ICIR of 3.4 at 21d collapsed to
  ~0.33 once measured cross-sectionally — the rest was market beta, not alpha.)*
* **ICIR**: `mean(IC)/std(IC)` across folds — the Sharpe-equivalent of forecasting
  skill, and the **selection criterion**.
* Plus directional **hit-rate**, MAE, R².

### 9.3 Label-shuffle leakage gate (permutation test)
Train the best family on `N` **shuffled-label** copies and build the null
distribution of `|per-date IC|` on the held-out fold. The model **passes iff its
real IC exceeds the null's 95th percentile** (`p < 0.05`). A single shuffle (the
original gate) is too noisy with a wide feature set and can't tell real skill from
chance; the distribution makes the verdict trustworthy, and it correctly fails
short horizons whose apparent edge is market-timing artifact. Recorded per horizon
as `leak_pass` with the real/null ICs. (`N=10` tabular, `3` for sequence models.)

### 9.4 Model store
The winner per horizon is refit on all fully-labeled data and written to
`models_store/forecast/<h>d/` (`model.pkl` + `metrics.json`), plus a `manifest.json`
index. Tracked in git (unlike `reports/`). Validated example: 252d → LightGBM
ICIR 1.10 / IC +0.124 / 64% hit / gate PASS.

### 9.5 Signal & honest-validation levers (V3.8)
* **Market-neutral target** (`--neutralize`): subtract each date's cross-sectional
  mean forward return so the model learns pure stock selection, not market timing.
* **Universe-weighted selection** (`--universe-weight w --priority-universe`):
  rank by `(1−w)·broad_ICIR + w·priority_ICIR` — train on breadth (`-u liquid`,
  374 names), select on relevance to the names you trade.
* **CPCV** (`--cv cpcv`, `models/validation.combinatorial_purged_splits`): C(n,k)
  purged+embargoed combinatorial paths → an ICIR *distribution*, not one fragile
  walk-forward estimate. Long-horizon paths that purge to empty are dropped.
* **Deflated ICIR** (`deflated_icir` / `expected_max_sharpe`): the winner's ICIR
  is haircut against the best-of-N null (we trial many families), so selection
  stays honest as the model zoo grows. Reported as the `Deflate` gate.
* **FinBERT text features** (`features/text_features.py`, `ts features --text`):
  causal, cached news sentiment from a finance transformer — the one orthogonal
  input (text, not prices). New `text` reserve group.
* **Futility pruning** (V4.1, default on; `--no-screen` disables): from the 2nd
  viable CV fold, a family whose running mean IC is ≤ 0 or < 35% of the
  leader's stops scheduling its remaining folds — the top-2 always keep
  running. This is where hopeless sequence families (minutes per fold) stop
  burning compute. Honesty is preserved: pruned families are **ineligible for
  selection** (1–2 lucky low-std folds would otherwise fake a huge ICIR), stay
  in the report (`pruned_after_fold`), and still count in the deflated-ICIR
  trial count. The panel is also sorted once across all horizons instead of
  per horizon.

### 9.6a Meta-labeling (`ts train-meta`, V4.3)

López de Prado's second stage. `ts train-forecast` now persists the winning
family's **out-of-sample predictions across all CV folds**
(`forecast/<h>d/oos_predictions.parquet` — the *pseudo-ledger*: thousands of
honest (prediction, outcome) pairs, no waiting for the live ledger).
`models/meta.py` trains a per-horizon LightGBM classifier on
`hit = sign(y_pred)==sign(y_true)` conditioned on **state, not alpha**: the
nonlinear fingerprint (Hurst, permutation/sample entropy, RQA determinism,
early-warning, tail index, LPPLS), RMT systematic fraction, vol/regime/macro
context, plus the signal's within-date shape (rank pct, z, sign) and
cross-sectional dispersion/breadth — "is this name currently in a
*forecastable* state?". Same purged walk-forward splitter as the primary
(labels span the same [d, d+h]); probabilities **isotonic-calibrated on
pooled OOS folds**; reported AUC/Brier-skill/decile-lift are OOS-only.
`ts invest` then sizes by **weight × P(right)** (`meta_p` on every position
and ledger record, `P✓` column in CLI/dashboard). Missing meta models degrade
gracefully to unscaled sizing.

### 9.6 Long-term decision artifact (`ts picks`)
`decision/longterm.py` packages the committed forecaster + conformal bounds into a
ranked plan: **entry** (last close), **add-on-dip** (1m lower band), **median &
stretch targets** (calibrated median / upper band), **invalidation stop**
(conformal lower band), reward:risk, and a trend-timing hint — with the model's
leakage-gate verdict attached so low-confidence horizons are flagged.

---

## 10. Probabilistic price bounds

`models/intervals.py`, `decision/bounds.py`, `models/implied_vol.py`.
Replaces the old `forecast_20d = score·4·0.65` and the symmetric ±2σ Gaussian fan.

### 10.1 Quantile regression
Per horizon {5,21,63,126,252d}, LightGBM with `objective="quantile"` at
τ∈{0.05,0.25,0.5,0.75,0.95} on realised forward returns — asymmetric, data-driven
lower/median/upper (equities don't move symmetrically).

### 10.2 Conformalized Quantile Regression (CQR)
Romano–Patterson–Candès (2019). Distribution-free coverage guarantee:

1. Fit `q_lo, q_hi` (τ=α/2, 1−α/2) on a training split.
2. On a *temporal* calibration split compute conformity scores
   `E_i = max( q_lo(x_i) − y_i ,  y_i − q_hi(x_i) )`.
3. `Q = ⌈(n+1)(1−α)⌉ / n` empirical quantile of `E`.
4. Calibrated band `[ q_lo(x) − Q , q_hi(x) + Q ]` has **≥(1−α) marginal
   coverage**, regardless of how well the quantile models fit.

Validated: ~90% out-of-sample coverage at every horizon.

### 10.3 Forward-looking width (implied vol)
`implied_vol.py` pulls the yfinance option chain, takes near-the-money IV per
expiry → an IV term structure, interpolated to each horizon. Used to widen the
band with the *market's* forward view rather than trailing realised vol.

### 10.4 Monte-Carlo fan (fallback)
When no bundle is trained: bootstrap the ticker's own daily log-returns (real
skew/fat tails), scale spread to IV when available, tilt drift gently toward the
model's 5-day view, simulate `mc_paths`, and read terminal-price quantiles.

### 10.5 Distribution-based sizing
`portfolio.sizing.distribution_sized_weights`: score each name by reward-to-downside
`edge / |5th-pct loss|` (Sortino-style), convert to fractional-Kelly weights,
cap per name, renormalise to ≤100% gross.

---

## 11. Decision pipeline

`decision/analyze.py` — `ts analyze TICKER`:

1. Load/refresh OHLCV (universe context matters for cross-sectional ranks).
2. Build features; take the ticker's latest row.
3. Score with the ensemble/registry model (or a transparent rule-based fallback).
4. 5d/20d forecast + BUY/HOLD/SELL with confidence; RSI sanity + liquidity floor.
5. **Groundings**: technical, regime, cross-section, events, **relevant_news**
   (RAG), apprehension, model+SHAP, **bounds**, **per-ticker SHAP waterfall**.
6. **Playbook overlay** (§12): triggered standing rules force action; a BUY must
   clear compliance or downgrade to HOLD; a triggered §3 rule forces the action.
7. Write markdown + JSON report; append a compact **DeepSeek narration**.

`analyze-all` runs the universe threaded (IV+LLM-bound) with a progress bar.

### 11.1 Invest planner (`ts invest <budget>`)

`decision/invest.py` — turns a dollar budget into a gated, sized,
hold-horizon-annotated buy plan:

1. **Conviction**: all committed horizons score the latest cross-section;
   per-horizon z-scores blend weighted by `horizon_reliability` =
   `max(ICIR, 0.05) × (1.0 if leak_pass else 0.35)`.
2. **Hold horizon**: `choose_hold_horizon` picks, per name, the conformal
   band's best `annualized_edge / annualized_downside × reliability`
   (edge annualized linearly, downside by √t). Hold = that horizon; the band's
   median/hi/lo become target/stretch/stop.
3. **Gates**: playbook compliance per BUY (blocked names reported with
   reasons); composite flag board scales deployable budget
   (`budget × deployment_fraction`).
4. **Sizing**: `portfolio/allocate.py` — Kelly conviction (band
   edge/downside) blended 50/50 with HRP on an **RMT-cleaned covariance**
   (Marchenko–Pastur noise eigenvalues flattened to their mean before
   single-linkage clustering + recursive bisection). Per-name cap with
   pro-rata excess redistribution; dollars → fractional/whole shares;
   sub-minimum allocations dropped and redistributed.
5. **Ledger**: each position is recorded as a falsifiable prediction (§11.2).

### 11.2 Decision ledger (`ts ledger`)

`monitoring/ledger.py` — the self-scoring feedback loop. Append-only JSONL
under `data/ledger/` (`predictions.jsonl` + `resolutions.jsonl`; a resolution
never mutates a prediction; ids are deterministic
`sha1(ticker|as_of|horizon|source)` so re-runs dedup). `resolve_ledger` scores
each prediction once its horizon has elapsed **in trading days on the ticker's
own price index**: directional hit, band coverage (the ~90% conformal promise,
checked on the system's own picks), realised vs forecast return.
`calibration_report` aggregates by (source, horizon): hit rate, coverage,
avg forecast vs realised, Spearman IC of conviction vs outcome. Surfaced as
`ts ledger [--resolve]` and the 📒 Calibration Ledger dashboard page. The
resolved ledger doubles as future training data for a meta-labeling stage.

### 11.3 Moonshot discovery (`ts discover`)

`decision/discover.py` + `ingestion/edgar_discovery.py` — a speculative
sourcing layer for asymmetric, under-the-radar names. Two candidate pools:

1. **Structural freshness** — EDGAR full-text search (`efts.sec.gov`) over a
   lookback window (default 120d) for the forms that precede big repricings:
   **10-12B / 10-12G** (spinoff registration — the SNDK signature),
   **8-A12B** (exchange listing: "about to trade"), **424B4** (final IPO
   prospectus). Filer CIKs map to tickers via the official
   `company_tickers.json`; responses are disk-cached for a day and reuse the
   SEC fair-access throttle from `sec_history`.
2. **In-universe sleepers** — the quiet end of the gold panel: a
   low-dollar-volume percentile blended with an **attention-ignition**
   z-score, `0.4·z(wiki_attention_mom) + 0.3·z(news_tone_mom) +
   0.2·z(sec_form4_90d) + 0.1·z(news_buzz)`; sleeper rank =
   `0.5·quiet + 0.5·tanh(ignition)`.

**Score.** Each candidate's own daily returns (de-meaned — no drift
extrapolation) are bootstrapped into 2 000 twelve-month paths;
`asym = upside(q95) / max(|downside(q5)|, 5%)`. Then:

```
moonshot_score = asym × category_boost × (1 + 0.25·tanh(ignition_z))
category_boost:  spinoff 1.3 · new_listing 1.15 · ipo 1.1 · sleeper 1.0
```

**Honesty rules** (design constraints, not afterthoughts):

* Names with too little history to band (< ~40 daily returns) go to a
  **watch-list** — reported, never scored.
* Sub-$1 prices and < **$2M/day** median dollar volume are excluded — an
  exit must exist before the entry is interesting.
* The 12m bands are **bootstrap fans, not conformal** — the committed models
  have no coverage of fresh listings; every position carries that warning
  (`bounds_method="mc_bootstrap_adhoc"`).
* The sleeve is **capped twice**: `ts invest --moonshot f` clips `f` to
  ≤ 0.30 of the deployable tranche and splits it equal-weight over
  playbook-cleared picks (count fitted so each position clears the $50
  minimum, ≤ 4 names), with each moonshot individually capped at **10% of
  deployable** — a lone pick can never absorb the whole sleeve.
* Every pick is ledgered under **`source="moonshot"`** so the §11.2
  calibration report scores the moonshot strategy separately from the core
  planner — the sleeve is judged by its own numbers, not by stories. Fresh
  names outside bronze OHLCV are priced **ad hoc via yfinance** at resolve
  time, so their predictions mature like any other (and their `as_of` is
  stamped from their own price history, not the gold panel's date).

Reports land in `reports/discover/discover_<date>.{md,json}`.

### 11.4 Portfolio tabs (`ts tab`)

`execution/tranches.py` — named virtual portfolios ("tabs"): the money-level
scoreboard the prediction-level ledger (§11.2) cannot provide.

* **Book format** — `data/portfolios/<name>.json`, an append-only list of
  fills (committable, so every machine shares the tabs). Each fill records
  ticker / shares / plan-close price / dollars, the plan's hold horizon and
  review levels (stop, median target), and the **SPY counterfactual
  shares** the same dollars would have bought at the same close.
  `ts invest … --portfolio p1` books the plan (core + moonshot sleeve) into
  `p1`; repeat tranches into the same name merge at mark time into a
  cost-weighted average (review levels come from the **latest** fill).
  Booking is **idempotent per plan**: each fill carries a `plan_id` (the
  plan's `generated_at`), so re-booking the same cached plan — a Streamlit
  re-render, a repeated CLI call — skips fills already present instead of
  doubling the book.
* **MTM semantics** — `ts tab` marks every book at the latest close in
  bronze OHLCV, pricing non-universe (moonshot) names **ad hoc via
  yfinance**; `ts tab p1` drills into one book with per-position P&L,
  hold-day counts and `hit_stop` / `hit_target` flags (last close vs the
  plan's review levels). All prices are **plan closes**: this measures
  *decision quality*, not execution — no slippage, no commissions,
  fractional shares assumed. A ticker that still can't be priced has its
  cost **held out of P&L entirely** (`unpriced_cost`, reported separately) —
  unknown is not a loss, and a fake $0 valuation would poison P&L% and
  alpha.
* **SPY counterfactual + alpha** — `spy_value` is what the same dollars,
  deployed into SPY on the same dates, are worth at the same mark. The
  comparison is strictly **covered-cost vs covered-cost** (no pro-rata
  extrapolation onto fills the benchmark can't match):
  `alpha_vs_spy = (pnl_pct − spy_pnl_pct) × cost`. The only fair null
  hypothesis — "you could have just bought SPY" — is baked into every
  readout.
* **Parallel strategies** — books are just names: run `p1`…`p5` with
  different planner flags (core-only vs `--moonshot 0.15`, different
  budgets) and compare tabs. The ledger scores predictions; the tabs score
  money.

---

## 12. Playbook engine

A decision-tree playbook encoded in `configs/playbook_v2.yaml`.

* **Five flags** (`flags/`): **O** oil/Iran (Brent), **F** Fed (FRED + tone), **I**
  inflation (core CPI), **S** semi tape (NDX), **C** AI capex. Each GREEN/YELLOW/RED,
  resolved concurrently with a resilient cached feed.
* **Composite**: ≥4 green/0 red → deploy 100%; YELLOW → halves; any RED →
  defensives only, 25% cap. `C=RED or S=RED → semi freeze`.
* **Compliance** (`playbook/compliance.py`): never-buy list, re-entry lockouts,
  13%/4% concentration caps, semi freeze, composite gating.
* **Blotter** (`playbook/blotter.py`): append-only fill log with realized P&L;
  `ts log-trade` records fills and stores their compliance verdict.

---

## 13. Backtesting & leakage tests

* Native vectorized engine (`backtesting/vectorized.py`) is authoritative; vectorbt
  / backtesting.py / bt adapters for cross-validation.
* **Three mandatory leakage tests** (`quality/leakage.py`): `shift_features_test`
  (peeking forward should *not* improve Sharpe), `signal_delay_test` (extra delay
  should degrade gracefully), `label_shuffle_test` (shuffled weights → Sharpe ~0).
* Golden-file backtest regression + data-quality + integration suites.

---

## 13a. Alpha engine v2 (`alpha/`, V5.2) — forecast ledger, continuous learning, causal backtest

Built 2026-09-21 because the pick path above was measurably not working: the
252d model's realised IC was 0.011, `ts picks` only ever ranked the ~54 names
with a positive neutralised score, `compute_bounds` made picks take 15 minutes,
and the HRP step produced 20% single-name weights. The engine is a separate,
self-contained package (`src/trading_system/alpha/`, `ts alpha …`) that reuses
the research simulator for its book backtest and leaves the old path in place
as a diagnostic.

* **Panel** (`panel.py`, ~10 s for 5M rows). Prices from
  `bronze/massive/ohlcv_deep.parquet` (1000 names, 1998→, daily sessions
  appended from `ohlcv_all`), 59 features as pure polars window expressions:
  31 price/volume (momentum 1–12m, 12-1, vol/downside vol/skew/kurtosis,
  max/min daily return, dollar volume, Amihud, volume ratio, 52w distances,
  SMA gaps, up-day fraction, overnight vs intraday, Parkinson vol, range
  position, beta/idio vol vs the panel's own EW market, Abdi–Ranaldo spread),
  13 fundamentals **point-in-time on `filing_date` + 1** (TTM E/P, B/M, S/P,
  CF/P, gross profitability, ROE/ROA, asset & revenue growth, accruals,
  leverage, log mcap, filing age), 5 news (counts, sentiment, count z-score —
  available the day *after* publication), 4 short (FINRA short interest +10d
  lag, short-volume ratio), 6 market/sector context. Labels `fwd_h` and
  `y_h` = per-date gaussianised rank of the forward return for h ∈ {5, 21, 63}.
* **Model** (`model.py`). Features are per-date pct-ranked; XGBoost on CUDA
  (~10 s per fit on 1.2M rows), squared error on the gaussian-rank label, 2
  seeds bagged, 10-year half-life sample decay, training dates thinned by a
  per-horizon stride so overlapping labels are not counted as independent.
  `causal_scores` walks forward: refit every 63 trading days on rows whose
  labels matured (`didx ≤ cut − h − 5`), score until the next refit.
* **Ledger** (`ledger.py`, `data/ledger/alpha_forecasts.parquet`). One row per
  (date, ticker, horizon, mode ∈ {backtest, live}): score, calibrated
  `exp_ret`, conformal `q10/q90`, entry price, and — once `horizon` sessions
  have passed — `realized_ret`, `hit`, `in_band`. `tally` fills outcomes from
  prices (a delisted name is scored at its last print). `skill_report` /
  `yearly_ic` / `rolling_ic` give rank-IC, ICIR, t-stat, hit rate, decile
  spread, band coverage and calibration slope, overall and per year.
* **Continuous learning** = `Calibrator`: refit on every run from the matured
  rows of the trailing 3 years — isotonic score → expected return per horizon,
  conformal 80% residual bands per score quintile, and **horizon blend weights
  ∝ trailing IC⁺** (a horizon whose realised skill has gone to zero loses its
  vote). Production models refit weekly (`ts alpha train`, ~1 min); the
  calibration refits daily (`ts alpha forecast`).
* **Book** (`portfolio.py`): gates (price ≥ $5, $vol ≥ $20M/d, vol ≤ 110%) →
  rank on demeaned composite / vol^0.5 → greedy top-20 with ≤ 5 per SIC sector
  → ½ inverse-vol + ½ equal weights, 8% cap → gross scaled to an 18% vol target
  (constant-correlation estimate) → Gârleanu–Pedersen partial trading (35% of
  the gap per monthly rebalance, 20% no-trade band). Optional regime overlay
  (gross × 0.5 when breadth < 40% and the EW market fell > 8% in a quarter).
  The same `target_weights` builds the backtest book and the live book.
* **Backtest** (`backtest.py`, `ts alpha backtest`): walk-forward scores →
  backtest ledger rows → tally → calibrator walked forward in 63-day blocks
  (sees only outcomes known before each block) → composite replayed through
  `research.wfbacktest.run_walk_forward` with the research cost model, against
  momentum (same book rules), the equal-weight universe and SPY; stationary
  bootstrap Sharpe CIs, deflated Sharpe over every variant tried, calendar-year
  table. Report: `reports/alpha/backtest.md` (+ json, daily curves).
  `--extend` scores only the dates after the cached run (weekly).
* **Live**: `ts alpha panel → tally → forecast --days 3` in the daily pipeline
  (~2 min), `ts alpha picks` for the brief, and the `alpha_v2` paper book in
  `ops/portfolio/portfolios.py` (GP partial trading, monthly) so a live
  track record accrues next to ml_v2_gp.

**Results (2004-01 → 2026-09, 22.7 years, after costs, `reports/alpha/backtest.md`).**
Signal: 63d rank-IC 0.073 (ICIR 0.75, t = 56, positive in 23 of 24 years;
0.11–0.15 in 2024–26), 21d 0.045, 5d 0.030; top-decile minus bottom-decile
63d spread +5.9%. Gross, the monthly top-20 earns ≈ +3.0%/month vs +1.4% for
the gated universe. Book (production rules): **CAGR 17.2%, Sharpe 1.12
[0.73, 1.56], MaxDD −31.7%, turnover 5.4×/yr, cost drag 1.5%/yr** vs
momentum with the same rules 16.7% / 0.83 / −37.6%, the equal-weight
*eligible* universe 14.6% / 0.78 / −53.9%, SPY 10.9% / 0.65 / −55.2%. Without
the overlays the same signal makes 19.9% / 1.00 / −51.6%; the 200-day trend
rule is what buys the drawdown. Two things learned the hard way and now
encoded: (1) the simulator gated on *split-adjusted* prices, which silently
excluded every future large-cap in its early years — eligibility is now
decided on raw closes and passed in as a null score; (2) the simulator
renormalised every book to full gross, so vol targets, regime scaling and
partial trading from cash had no effect — `respect_target_gross=True` makes it
honour the gross the weight function asked for (opt-in; the lab's numbers used
the old behaviour). Symbol reuse in the free price feeds (SOLS ×487,399 in a
day, CHRD, BNY…) is cut at the last series break in `sanitize_prices`.

* **Regime / fragility layer** (`regime.py`, `ts alpha regime`, 2026-09-21).
  A daily state vector since 1990 — VIX/VXN, WTI/Brent level and realised
  vol, OVX, Baa–10y credit, 10y–3m curve, real 10y, breakevens, dollar,
  Nasdaq relative strength and drawdown (the dot-com fingerprint), plus from
  the panel: market drawdown/vol/trend/breadth/dispersion, average pairwise
  correlation and an **AI basket** (semis + AI cloud + hyperscalers) vol,
  momentum and share of volume. Everything lagged a session and z-scored on
  an *expanding* window (point-in-time). FRED series are cached under
  `data/silver/regime/` and refreshed daily (`FRED_API_KEY`).
  - **Analogs**: NaN-aware distance from today's z-vector to every past day;
    rank-based weights `w ∝ exp(−rank/252)`; a similarity score per curated
    episode (Gulf War 1990, LTCM, dot-com, 9/11, Iraq, 2008-H1 oil spike,
    GFC, euro 2011, oil crash 2014-16, China 2015, Q4 2018, Covid, 2022
    inflation/Ukraine, SVB, tariff shock 2025, Israel–Iran oil 2025).
  - **Conditional evidence**: the backtest ledger's rank-IC and the book's
    next-63-day return / drawdown distribution *in the analog days*, next to
    the unconditional numbers (`conditional_skill`, `conditional_forward`).
  - **Recalibration**: `ts alpha forecast` fits the calibrator twice — on the
    trailing 3 years and on the whole matured history *weighted by analog
    similarity* (weighted isotonic + weighted conformal quantiles) — and
    blends them 50/50 (`--analog-weight`); bands take the wider side; horizon
    weights follow the blended IC. First live reading (2026-09-18: Brent
    $131, OVX 52, VIX 15, AI share of volume z +2.4): closest episodes
    Israel–Iran oil 2025, tariff shock 2025, SVB, Q4 2018, 2022; in those
    backgrounds the 21d IC was +0.008 vs +0.046 unconditional while the 63d
    IC held (+0.070 vs +0.074) → the 21d horizon's vote fell from 0.28 to
    0.23 and 63d rose to 0.63.
  - **Model context**: nine date-level macro columns (`MACRO_FEATURES`) are
    fed to the forecaster *raw* — `model.prepare` now rank-transforms only
    cross-sectional features (a within-date rank of a date-constant is 0.5
    everywhere, which is how the first version silently discarded its market
    context).
  - **Stress overlay**: `fragility_score` (stress = VIX, oil vol, OVX, credit
    Δ, correlation, market vol; imbalance = oil level, real rates, rate Δ, AI
    share, Nasdaq relative, AI vol) and `gross_multiplier` exist and are
    reported, but the overlay is **off** by default: on 2004→26 it added
    nothing over the 200-day trend rule (Sharpe 1.13 → 1.13, MaxDD −33 →
    −32%), because partial trading already lags exposure changes.

* **Model changes go through `ts alpha experiment`** (`experiment.py`, 2026-09-26): candidate
  `TrainSpec`s walked forward on identical folds (refit 126, OOS 2011→), per-date IC, ICIR,
  recent IC, decile spread, paired t on non-overlapping dates, years won. Adoption: paired t ≥ 2
  on 63d IC **or** a book-level excess (same simulator/costs/overlays) whose bootstrap CI is
  above zero with ≥70% of years won. First run:
  - **Earnings surprise features** (SUE, revenue SUE, ΔSUE, announcement-window return; PIT on
    the SEC filing date, Q4 from the 10-K at end+75d) — **rejected** (paired t −0.3). Kept in the
    panel for research; not in `MODEL_FEATURES`' effect because the ranker ignores them.
  - **Pairwise ranking objective** (`TrainSpec.objective="rank"`, per-date query groups) —
    IC flat (t +0.5) but top–bottom decile spread 4.7% → 9.4% (63d). Book test 2012→2026:
    CAGR 16.5% → 29.1%, Sharpe 1.09 → 1.42, alpha +5.2% → +13.6%/yr at beta 0.72 → 0.90,
    excess +11.0%/yr CI [+6.3%, +15.8%], 12/15 years better → **adopted** as the production
    objective. MaxDD rose −28% → −39%.
* **`ts data status`** (`datastatus.py`): every store's rows/tickers/span, sessions behind the
  last close and an ok/stale verdict, crawler liveness/parked families/calls-per-day, disk.

Tests: `tests/unit/test_alpha.py` (hand-checked features, backward-looking
guarantee, PIT joins, purge/embargo, planted-signal recovery through the
causal walk, ledger tally incl. delistings, calibrator monotonicity and band
coverage, causal composite, book gates/caps/vol target/partial trading).

---

## 14. LLM layer (DeepSeek + RAG)

`ingestion/llm_extractor.py`, `decision/explain.py`.

* **Router**: DeepSeek cloud primary, Ollama local fallback, rule-based last.
* **Cost discipline**: compact JSON prompts (not rendered markdown), a stable
  cache-friendly system-prompt prefix (DeepSeek disk cache → ~1/10 input cost on
  hits — observed 93% hit-rate), concurrent per-ticker calls (`complete_many`),
  and cache-hit telemetry.
* **Apprehension scorer**: one batched call per ticker → `apprehension_score`,
  `outlook`, drivers (a leakage-safe risk feature).
* **RAG** grounds narration in retrieved, point-in-time news (§4).

---

## 15. Compute & hardware layer

`utils/compute.py`. `get_compute_profile()` (cached) detects cores, RAM, and GPU
(CUDA via torch, MPS on Apple Silicon) and returns a `ComputeProfile` with tuned
`lgbm_params()` / `xgb_params()` (device + `n_jobs`) and BLAS thread env vars.
Overrides: `TS_DEVICE` (cpu|gpu), `TS_N_JOBS`, `TS_GPU` (0|1). It also calls
`preload_omp_runtimes()` (see §18).

---

## 16. Parallelism map

| Work | Strategy | Control |
| --- | --- | --- |
| OHLCV ingest | thread pool + progress | `--workers`, `TS_INGEST_WORKERS` |
| `analyze-all` | thread pool (IV+LLM bound) + progress | `--workers` |
| Earnings calendar | thread pool + progress | yfinance, network-bound |
| News / SEC loops | progress bar only (rate-limited) | — |
| Nonlinear features | **spawn** process pool, per ticker | `--jobs`, `--no-parallel` |
| LLM per-ticker calls | thread pool | `complete_many` |
| Tree/RNN training | intra-model (`n_jobs`/GPU) via compute profile | `TS_N_JOBS`/`TS_DEVICE` |
| RMT | strided cross-sectional (already cheap) | — |

Shared helpers: `utils/progress.py` — `track()` (progress) and `parallel_map()`
(threaded map + progress). The nonlinear pool uses **spawn** (not fork/loky)
because fork pools segfault and lingering loky pools deadlock once torch/MPS state
is live in the parent; spawn workers start clean and a `with`-block tears the pool
down immediately.

---

## 17. Command reference

Full grouped directory: [`docs/COMMANDS.md`](COMMANDS.md) (regenerate with
`ts commands --write`). The pipeline order:

```bash
ts ingest            # threaded OHLCV + news → bronze/silver
ts quality           # data-quality gate
ts features          # → gold/features.parquet (add --deep, --jobs N)
ts train             # ensemble (5d)
ts train-forecast    # long-horizon best models (+ --models lstm,gru) → models_store/
ts train-intervals   # conformal price-bound models
ts analyze TICKER    # decision + report (bounds, SHAP, narration)
ts invest 2000       # budget → gated, sized, hold-annotated buy plan (+ ledger)
ts discover          # moonshot scan: EDGAR spinoffs/listings/IPOs + sleepers (+ ledger)
ts tab [name]        # portfolio tabs: money-level P&L + alpha vs SPY counterfactual
ts ledger --resolve  # score matured predictions: hit rate, coverage, IC
ts bounds TICKER     # calibrated low/median/high per horizon
ts complexity TICKER # nonlinear fingerprint
ts daily             # full pipeline + paper rebalance + report
ts dashboard         # Streamlit desk (5 grouped sections)
```

---

## 18. Operational gotchas

* **macOS `libomp` double-init crash.** Importing torch before LightGBM/XGBoost
  segfaults the first tree fit (two OpenMP runtimes). Since `get_compute_profile()`
  imports torch, this is latent everywhere once torch is installed. Fix:
  `preload_omp_runtimes()` imports LightGBM/XGBoost's OpenMP *first*; called from
  `compute.py` and `sequence.py`.
* **torch ↔ LightGBM OpenMP deadlock.** torch CPU training hangs against
  LightGBM's pool in a shared process. Fix: `torch.set_num_threads(1)` in the
  sequence model's fit/predict (GPU/MPS math unaffected).
* **Parallelism backend.** Use a **spawn** `ProcessPoolExecutor` in a `with`
  block, *not* fork or loky — both fail (segfault / deadlock) once MPS state is
  live in the parent.
* **polars NaN vs null.** A numpy `np.nan` becomes a polars **NaN-float**, which
  `is_not_null()` counts as *present* — defeating `resolve_reserve`'s coverage gate
  and leaking NaN into tabular models. Always `fill_nan(None)` on numpy-backed
  feature columns.
* **`./venv` vs anaconda.** The project venv (`./venv/bin/python`) has torch 2.x +
  MPS and modern numpy/scipy/sklearn; a stale anaconda may shadow `python` on PATH.
  Use `./venv/bin/python` for everything.
* **News key provider.** A `pub_…` key is **NewsData.io**, not Finnhub
  (`NEWSDATA_API_KEY`). Finnhub keys are not `pub_`-prefixed.
* **`lxml` required** for yfinance `earnings_dates` (now a dependency).

### 18.1 Point-in-time & data-integrity principles
These are load-bearing — violating them silently inflates measured skill:
* **News for training vs live.** Recent-fetch backends only cover ~a week, so they
  are **not** trained features — they power the live decision layer (analyze / RAG
  / apprehension). The *trained* deep-history signals come from `ts backfill-history`:
  **GDELT** tone/attention (2017+), **SEC EDGAR** filing intensity (all-forms / 8-K /
  Form-4 counts + recency — covers the *whole* panel), and **Wikipedia** pageviews
  (retail-attention proxy, 2015+). All causal by construction (each value known by
  that day's close) and cached per ticker.
* **Coverage-driven, resumable backfill.** `ts backfill-history` retries the
  still-uncovered tickers round after round until ≥90% *ticker* coverage (not
  panel-row coverage — the pre-floor years can't be filled) or a plateau. A
  per-source progress ledger (`_progress.json`) + the per-ticker caches make it
  idempotent: a re-run or a restart of an interrupted (tmux) run does **zero**
  network for tickers already completed for that day, so it never repeats work.
* **Impute-and-indicator for sparse signals.** A signal that only exists from a
  source's coverage floor (GDELT 2017+, Wiki 2015+) or for covered names is
  *structurally* sparse — leaving it null both trips `resolve_reserve`'s ≥60%
  coverage gate (so the signal never trains) **and** would delete every uncovered
  row via the trainer's `drop_nulls`. `densify_sparse_signals` resolves this the
  standard way: add a per-source **presence flag** (`news_present` / `sec_present`
  / `wiki_present`) and neutral-fill the columns. The filled columns are dense
  (pass the gate, never poison rows); the flag lets the model separate *"neutral
  reading"* from *"no coverage"*. Fills are constants → not forward-looking. (Truly
  dense-when-present event/apprehension features stay **null-when-absent** so a
  sparse recent-only column is still honestly dropped rather than faked to 0.)
* **Earnings look-ahead cap.** `days_to_earnings` only counts a future date within
  a ~90-day announcement window (dates months out weren't knowable).
* **Survivorship bias.** The universe is *today's* constituents; delisted names are
  absent and recent IPOs have short history, so long-run ICIR/picks read
  optimistically — `ts picks` prints the caveat. (Free data can't give point-in-time
  index membership; treat it as a known discount.)
* **`known_at` on rebuild.** The RAG point-in-time filter is only fully honest if
  events were accumulated by daily runs; a from-scratch rebuild stamps `known_at`
  = now. Irrelevant to GDELT (dates are historical) and to training (keyed on the
  event date), but matters for RAG backtests.

---

## 19. Glossary

* **IC / rank-IC** — Spearman correlation of forecast vs realised return.
* **ICIR** — `mean(IC)/std(IC)`; forecasting skill's information ratio.
* **Purge / embargo** — removing train rows whose label overlaps (or sits just
  before) the test block to prevent leakage from overlapping horizons.
* **CQR** — Conformalized Quantile Regression; distribution-free calibrated bands.
* **Hurst H** — long-memory exponent; >0.5 trending, <0.5 mean-reverting.
* **Lyapunov λ** — exponential divergence rate; >0 = chaotic.
* **Marchenko–Pastur** — noise eigenvalue band of a random correlation matrix.
* **LPPLS** — log-periodic power-law singularity; bubble/crash signature.
* **Critical slowing down** — rising AR(1)+variance near a tipping point.
* **Apprehension score** — LLM-assessed market fear for a name [0,1].
* **Composite deployment** — the flag-board-driven fraction of cash to deploy.

---

*Generated as part of the V3.7 overhaul. Keep this in sync via the modules it
references; the feature catalog and command directory are the source of truth.*
