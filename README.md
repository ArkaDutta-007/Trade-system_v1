# trading-system

Local-first research and **paper-trading** system for systematic equity strategies.
Built around DuckDB + Parquet + Polars, with vectorized backtesting, walk-forward
ML training, SHAP-based interpretability, and strict leakage / data-quality tests.

> Not financial advice. This system is intended for research and paper trading.
> Live execution should only follow months of paper trading and slippage validation.

## Layout

```
trading-system/
├── data/                # raw / bronze / silver / gold (parquet)
├── src/trading_system/
│   ├── ingestion/       # market_data, sec_filings, macro_fred, news_events, calendar_events
│   ├── features/        # technical, regimes, fundamentals, sentiment, event_features, build
│   ├── strategies/      # baseline_momentum, mean_reversion, event_driven, ml_signal
│   ├── backtesting/     # vectorized, event_driven, slippage, metrics
│   ├── models/          # train (walk-forward LightGBM), predict, shap_analysis, model_registry
│   ├── portfolio/       # sizing, risk (limits + kill switch), order_policy
│   ├── execution/       # paper_broker, live_broker (stub)
│   ├── monitoring/      # drift (KS tests), pnl_attribution, alerts
│   ├── quality/         # data_checks, leakage tests
│   ├── pipeline/        # daily flow
│   ├── storage/         # DuckDB + Parquet helpers
│   ├── utils/
│   ├── cli.py           # `ts` command-line entry point
│   └── config.py
├── tests/               # unit, integration, data_quality, leakage_tests, backtest_regression
├── configs/default.yaml
├── scripts/             # dashboard.py, prefect_flow.py
└── pyproject.toml
```

## Install

```bash
cd trading-system
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Optional: `pip install -e ".[backtest]"` to add `vectorbt` and `backtesting.py` for
parameter sweeps and a second-engine sanity check.

## Configure

`configs/default.yaml` is the canonical config. Set API keys via environment:

```bash
export FRED_API_KEY=...     # macro data
export NEWSAPI_KEY=...       # news (optional)
export OPENAI_API_KEY=...    # event extraction (optional)
```

## Quick start

```bash
ts universe                       # show the 100-symbol universe
ts ingest                         # pull daily OHLCV for the configured universe
ts quality                        # OHLCV data-quality checks
ts features                       # build the feature matrix to gold/
ts train                          # walk-forward LightGBM + SHAP summary

# Single-symbol decision interface (your typical workflow)
ts analyze GOOGL                  # decision + markdown audit doc + groundings
ts analyze MSFT --no-report       # decision-only, skip writing files
ts analyze-all                    # run on the full 100-symbol universe

# Full daily flow
ts backtest momentum_rotation     # vectorized backtest with metrics
ts daily                          # full pipeline + paper rebalance + report
ts dashboard                      # Streamlit UI
```

## Single-symbol analysis (`ts analyze TICKER`)

Calls `analyze_symbol(ticker)`. For any one symbol the system:

1. Loads (or refreshes) OHLCV across the universe — context matters because
   features include cross-sectional momentum ranks vs the other 99 names.
2. Builds the feature matrix and looks up the most recent row for the ticker.
3. Loads the latest model from `reports/models/`. If no model is present yet,
   falls back to a transparent rule-based score.
4. Produces 5d and 20d return forecasts plus a **BUY / HOLD / SELL** stance
   with confidence, applying RSI sanity overrides and a liquidity floor.
5. Writes a markdown audit doc to `reports/decisions/<TICKER>_<timestamp>.md`
   and a JSON sidecar with the same content.

Each report contains:

* Stance, confidence, 5d/20d forecasts, score source (model vs rules)
* Rationale bullets
* **Technical state:** momentum (5/20/60/120/12-1m), realized vol, RSI, ATR,
  SMA gaps, breakout/drawdown, relative volume, average dollar volume
* **Regime:** bull regime (200d SMA), high-vol regime, cross-sectional
  momentum percentile, excess return vs benchmark
* **Cross-section:** the ticker's rank in the universe today, plus the top
  10 and bottom 10 names by 20d momentum
* **Recent events:** rows from `data/silver/events.parquet` if populated
  (NewsAPI / SEC EDGAR / FRED feed into this; LLM extractor optional)
* **Model groundings:** model score (5d expected return), feature columns,
  top features by mean-absolute SHAP

## Universe (100 symbols)

Defined in [`configs/universe_100.yaml`](configs/universe_100.yaml):

* **52 required** (your input): META, COIN, AMZN, INTC, SNPS, DUOL, AVGO,
  MSFT, SNAP, MRVL, SQNS, TW, TSM, ADBE, INTU, UBER, CRM, GOOG, NOW, LLY,
  SFM, SBET, MNDY, ALSN, NVDA, NFLX, AAPL, SMCI, ARM, CRWD, QCOM, DELL, IBM,
  PANW, GE, RDDT, RBLX, DASH, UNH, ABNB, ZM, DBX, CRWV, ORCL, AMD, NBIS,
  SPOT, MDB, AMAT, MU, LRCX, ASML.
* **48 additions** for breadth and benchmarks: GOOGL, TSLA, PLTR, SHOP, SNOW,
  DDOG, NET, ZS, OKTA, PYPL, SQ, DKNG, BRK-B, JPM, BAC, GS, V, MA, AXP, COST,
  WMT, HD, MCD, NKE, DIS, KO, PEP, JNJ, PFE, ABBV, MRK, TMO, PG, BA, CAT, DE,
  HON, RTX, XOM, CVX, VZ, SPY, QQQ, XLK, XLF, XLE, XLV, XLI.

`ts analyze GOOGL` works for tickers outside the configured universe too —
it fetches that symbol on the fly and runs the full decision pipeline.

## Open-source engines (cross-validation backends)

The native `run_vectorized_backtest` is the workhorse, but you can run the
same `(prices, weights)` pair through three other widely-used engines for
cross-validation. Install with `pip install -e ".[backtest]"`:

| Engine | Module | Best for |
| --- | --- | --- |
| **vectorbt** | `trading_system.backtesting.engines.vectorbt_backtest` | Massive parameter sweeps, Numba-accelerated portfolio backtests |
| **backtesting.py** | `trading_system.backtesting.engines.backtesting_py_backtest` | Quick single-asset prototypes with Bokeh charts |
| **bt** | `trading_system.backtesting.engines.bt_backtest` | Portfolio-level allocation backtests (weight schedules) |

Adapters convert Polars frames in and return the engine's native result
object so you can use its full reporting/visualization. The native engine
remains authoritative for the daily pipeline.

Run tests:

```bash
pytest tests/unit tests/data_quality
pytest tests/leakage_tests        # mandatory before promoting a strategy
pytest tests/backtest_regression  # golden-file regression
pytest -m integration             # end-to-end (slow)
```

## Daily pipeline

`run_daily_pipeline()` does, in order:

1. Ingest OHLCV (yfinance)
2. Run OHLCV quality checks; alert + abort on failure
3. Build feature matrix (technical + regime + cross-sectional, optional event features)
4. Generate signals (default: momentum rotation top-k)
5. Apply risk overlays (per-position cap, gross cap, drawdown kill switch)
6. Vectorized backtest for metrics + monitoring
7. Convert today's target weights to orders and submit to the paper broker
8. PnL attribution per ticker
9. Write JSON report to `reports/daily_YYYYMMDD.json`

To schedule, see `scripts/prefect_flow.py`.

## Strategies

| Name | Description |
| --- | --- |
| `buy_and_hold` | 100% benchmark (SPY by default) |
| `ma_crossover` | Long benchmark when fast SMA > slow SMA |
| `momentum_rotation` | Top-k cross-sectional momentum, monthly rebalance |
| `mean_reversion` | Long after a sharp 1-day drop, short hold horizon |
| `event_driven` | Sentiment + novelty tilt from extracted events |
| `ml_ranker` | Top-k by walk-forward LightGBM scores |

## ML model

Walk-forward LightGBM regression on `forward_return_5d`. SHAP runs on the latest
fold; the summary is written to `reports/shap_summary.csv`. The model registry lives
under `reports/models/`.

Critical: the feature matrix excludes `forward_return_*` from the model's feature
columns, and all rolling features are computed from past values only.

## Leakage testing — non-negotiable

`src/trading_system/quality/leakage.py` ships three mandatory tests:

1. **shift_features_test** — shifts weights *forward* (peeking at future returns).
   If Sharpe improves materially, the strategy is using future info.
2. **signal_delay_test** — adds an extra day of execution delay; performance should
   degrade but not collapse.
3. **label_shuffle_test** — randomly reorders weights; Sharpe should be ~0.

Run these before promoting any new strategy.

## What's intentionally a stub today

* `live_broker.py` raises `NotImplementedError` until live readiness is reviewed.
* `features/fundamentals.py` is a pass-through; wire it to SEC XBRL company-facts
  data when ready.
* `news_events.py` returns empty unless `NEWSAPI_KEY` is set; production-grade
  event extraction should run an LLM with structured outputs against the news +
  SEC filings.

## What's missing on purpose

This MVP avoids:

* tick-level data and full order-book simulation
* RL / deep models (use tree models first; they're easier to debug and SHAP-friendly)
* live trading infrastructure

Add those only after the vectorized loop, the leakage tests, and 3-6 months of
paper trading look healthy.
