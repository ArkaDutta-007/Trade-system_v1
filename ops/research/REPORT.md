# Model & Backtest Robustness Program — Final Report
**2026-07-14 · RIT machine · `~/trade-ops/research/` · zero repo mutations**

## What was asked
Improve models, training, and backtesting for more robust and better outcomes,
run it end-to-end, and put train+backtest into the openclaw daily schedule.

## What is now LIVE
- **Deployed model:** `reports/models/ensemble_1784069157` — the repo's full
  14-model ensemble retrained with **purged walk-forward (5td) + embargo (5td)
  + temporal validation** on the base-14 feature set (config B, the experiment
  winner). The registry auto-selects the newest ensemble; verified live:
  `ts analyze NVDA` → `score_src=ensemble:ensemble_stack_lgbm`, differentiated
  forecasts (the old flat +0.70% rule-fallback is gone).
- **Honest out-of-sample quality:** daily cross-sectional IC **+0.0108,
  t-stat 4.3** over 2,830 OOS days (2015→2026).
- **Model P&L** (top-20 equal-weight, after 4bps costs, 11.5y OOS):
  **CAGR +32.9% · Sharpe 1.03 · MaxDD −53.8%** raw;
  **+11.9% · 0.82 · −17.3% MaxDD · Calmar 0.69** with the 15% vol-target +
  drawdown-throttle overlay (the deployable configuration).
- **future-predict** seeded with the new model ($10k tiered session,
  2026-07-14: STX, MRNA, LCID, TER, AEM, ORCL, APP, LITE, …).
- **Schedule:** daily 05:15 digest now includes the model backtest + trailing-
  63d decay flag; **cron `trade-weekly-retrain` (Sat 03:00 ET, alert-wrapped)**
  retrains + deploys weekly via `~/trade-ops/weekly_retrain.sh`.

## Experiment results (3-learner proxy, identical eval, 12 folds, 2,830 OOS days)

| Variant | Features | IC | t-stat | IC-IR | verdict |
|---|---|---:|---:|---:|---|
| A current repo (leaky) | 14 | +0.0081 | 2.9 | 0.054 | baseline |
| **B purged + temporal val** | 14 | **+0.0101** | **3.7** | **0.070** | **winner** |
| C = B + wide features | 86 | +0.0090 | 2.1 | 0.040 | more ≠ better |
| D = C + xsec target | 86 | +0.0058 | 1.5 | 0.027 | worst |

(Full ensemble on config B did better than the proxy: IC +0.0108, t=4.3.)

## Findings
1. **The leakage fix IMPROVED results** (+25% IC, higher t-stat): train.py's
   row-order validation split was corrupting ensemble blend weights, and the
   last 5 train days' labels overlapped the test window. Honest > leaky, in
   P&L too — B ≥ A on every like-for-like backtest config.
2. **Feature widening failed as-is** (zero-fill + GBMs): the deep nonlinear /
   macro / news set added variance, not signal (t 3.7 → 2.1). Needs SHAP-gated
   per-feature selection before it earns a slot. Note: the event/sentiment
   columns are 99.9% null by design (event days only) — any `drop_nulls`
   across them wipes the dataset; zero-fill is the correct treatment.
3. **Cross-sectional target underperformed** at this horizon/universe;
   `xsec_neutralize` ships in the patches but OFF (it also changes score
   semantics — the ±0.5% decision thresholds would misfire).
4. **Sizing:** equal-weight top-20 beat score- and inv-vol-weighting (scores
   too noisy to size on). Robustness comes from the overlay: MaxDD −58%→−18%
   at equal Sharpe, Calmar 0.49→0.66.
5. **Cost sensitivity is the #1 open risk:** 3× costs halve Sharpe
   (0.87→0.53). Daily top-20 churn is expensive → next lever: weekly
   rebalance / trade bands / turnover penalty in the ranker.
6. **Multiple-testing honesty:** across all 36 configs tried,
   PBO(CSCV) = 33% (moderate), best-config Deflated Sharpe = 0.96 (the Sharpe
   survives correction for how many things we tried).
7. Reference: rule-based `momentum_rotation` still wins risk-adjusted
   (Sharpe 1.21 vs 1.03) — blending the ML sleeve with the momentum sleeve is
   the obvious next study, along with meta-label gating (`models/meta.py` is
   already built for it) and a 5d/21d multi-horizon blend.

## Known limitations (not fixable on this box)
- **Survivorship bias:** universe = today's constituents; absolute CAGRs are
  upper bounds. Needs point-in-time membership + delisted history (laptop
  decision — e.g. Sharadar/Norgate, or start snapshotting membership now).
- Overlay applied on the return stream (exposure-scaling trade costs not
  modeled); flat-bps costs remain optimistic for less-liquid names.

## Artifacts
- `out/summary.json`, `out/backtest_report.json`, `out/oos_*.parquet`,
  `out/harness.log`, `out/chain.log` — every number above is reproducible.
- `patches/*.patch` + `patches/README.md` — laptop-ready diffs: purge/embargo
  + temporal val (train.py), config-driven features + `ts backtest ml_ranker`
  wiring (cli.py), sqrt market-impact costs (slippage/vectorized), config keys
  (default.yaml). Apply order & evidence in the README.
- `deploy.py` / `~/trade-ops/weekly_retrain.sh` — the recurring retrain path.
- `daily_ml_backtest.py` — the digest block (also: ask Silas "how is the
  model doing").
