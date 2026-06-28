# Command directory

All commands are invoked as `ts <command>`. Run `ts <command> --help` for options.

## Data & features

| Command | What it does |
| --- | --- |
| `ts ingest` | Pull daily OHLCV for the universe (yfinance) |
| `ts quality` | OHLCV data-quality checks |
| `ts features` | Build the gold feature matrix (technical+macro+events reserve) |
| `ts universe` | Show the active universe |

## Forecasting & models

| Command | What it does |
| --- | --- |
| `ts train` | Walk-forward 14-model ensemble (5d target) |
| `ts train-forecast` | ★ Long-horizon best models via purged CV → models_store/ |
| `ts train-intervals` | Conformalized quantile price-bound models (90% coverage) |
| `ts bounds` | Show lower/median/upper price bounds for a ticker |
| `ts backtest` | Vectorized backtest of a strategy with metrics |

## Decisions

| Command | What it does |
| --- | --- |
| `ts analyze` | Single-symbol decision + report (bounds, SHAP, narration) |
| `ts analyze-all` | Run analyze across the whole universe |
| `ts signals` | Cross-sectional signal table |
| `ts explain` | DeepSeek plain-English narration of a report |

## Future-predict sessions

| Command | What it does |
| --- | --- |
| `ts future-predict` | Open a dated forecast session with allocation |
| `ts future-status` | Show a session's equity + hit-rate |
| `ts future-update` | MTM + redeploy dry powder for live sessions |

## Playbook & NRA tax

| Command | What it does |
| --- | --- |
| `ts flags` | Live O/F/I/S/C flag board + composite |
| `ts brief` | One-page morning briefing |
| `ts playbook` | Which §4 cycle rules fire today |
| `ts check` | Pre-trade compliance (never-buy, caps, freeze, tax) |
| `ts log-trade` | Record a fill in the blotter |
| `ts tax` | 2026 NRA tax-shield status |

## Paper trading

| Command | What it does |
| --- | --- |
| `ts paper-trade` | Run the ML/ensemble paper portfolio |
| `ts paper-status` | Show paper-portfolio holdings + equity |
| `ts daily` | Full daily pipeline (ingest→features→signals→rebalance) |

## Agent (LLM)

| Command | What it does |
| --- | --- |
| `ts agent-analyze` | Multi-step ReAct analysis of a ticker |
| `ts agent-portfolio` | Agent review of the whole portfolio |
| `ts agent-briefing` | Agent-written morning brief |

## App

| Command | What it does |
| --- | --- |
| `ts dashboard` | Launch the Streamlit desk |
| `ts commands` | This command directory |
