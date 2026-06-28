# models_store — committed best models

Unlike `reports/models/` (gitignored, scratch), this directory **is tracked by
git** so the best forecasters travel with the repo.

Populated by `ts train-forecast`:

```
models_store/
├── manifest.json              # index: per-horizon best model, ICIR, IC, hit-rate, leakage gate
├── forecast/
│   └── <h>d/
│       ├── model.pkl          # best estimator for horizon h, refit on all labeled data
│       └── metrics.json       # purged-walk-forward OOS metrics + leakage-gate result
└── intervals/
    ├── interval_bundle.pkl    # conformalized quantile bounds (from `ts train-intervals`)
    └── meta.json              # per-horizon coverage
```

How they're chosen: each horizon is evaluated with **purged + embargoed
walk-forward CV** (overlapping labels can't leak) across LightGBM / XGBoost /
HistGBM / Ridge, ranked by **ICIR** (IC ÷ IC-std), and gated by a label-shuffle
leakage test. See `src/trading_system/models/forecast_train.py`.

Regenerate: `ts train-forecast` (add `--horizons 21,63,126,252`).
