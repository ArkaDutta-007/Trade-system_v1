# Trade-system_v1 — model/backtest robustness patches (2026-07-14)

Generated on the RIT machine by the research harness in `~/trade-ops/research/`.
Every change here was validated on the real gold feature matrix first — numbers
in `../out/summary.json` and `../out/backtest_report.json`, prose in
`../REPORT.md`.

**Apply on the laptop** (repo root), then commit + push; the RIT machine pulls:

```bash
git apply --stat  ~/patches/train.py.patch        # preview
git apply         ~/patches/train.py.patch
git apply         ~/patches/cli.py.patch
git apply         ~/patches/slippage.py.patch
git apply         ~/patches/vectorized.py.patch
git apply         ~/patches/default.yaml.patch
pytest tests/ -x -q                                # sanity
```

(Copy the `*.patch` files over first, e.g.
`scp rit:~/trade-ops/research/patches/*.patch ~/patches/`.)

## What each patch does & why

| Patch | Change | Evidence |
|---|---|---|
| `train.py.patch` | **Purge (5 td) + embargo (5 td)** in the walk-forward: the last 5 train days' `forward_return_5d` labels are computed from test-window prices — that's leakage. **Temporal validation split** (last 20% of train *dates*) replaces the row-order 80/20 slice, which mixed dates across train/val and corrupted the IC blend weights. Adds `FeatureSpec.xsec_neutralize` (research option). | Purged+temporal-val (B) beat the current setup (A) OOS: daily IC +0.0101 vs +0.0081, t-stat 3.7 vs 2.9. Honest AND better. |
| `cli.py.patch` | `ts train` reads `model.feature_columns` from config (`base14` \| `wide` \| explicit list) instead of a hard-coded 14-column list, passes `purge_days`/`embargo_days` through, and **wires `ts backtest ml_ranker`** — backtests the trained model's OOS predictions (`data/gold/predictions.parquet`) into an actual equity curve. | The deep/macro/news features the repo already computes were unreachable by the ensemble; the model had no P&L view at all. |
| `slippage.py.patch` + `vectorized.py.patch` | Optional **square-root market-impact** term (`impact_coeff_bps`, default 0 = exactly the old behaviour). | Flat 4 bps round-trip is optimistic for anything that trades daily; sqrt-impact is the standard institutional model. |
| `default.yaml.patch` | Turns it all on: `feature_columns: wide`, `purge_days/embargo_days: 5`, `impact_coeff_bps: 10`. | See REPORT.md variant table. |

## Not patched (deliberate)

- **`xsec_neutralize` ships OFF.** Cross-sectional demeaning changes score
  semantics from "expected 5d return" to "expected relative return" — the
  ±0.5% BUY/SELL thresholds in `decision/` would silently misfire. If the
  research numbers justify it, the decision layer should move to rank-based
  thresholds in the same commit that turns this on.
- **Survivorship bias** is a data problem, not a code problem: the universe
  files list *today's* constituents, so backtests inherit survivorship. Fixing
  it needs point-in-time membership + delisted-ticker history (e.g. a Sharadar/
  Norgate subscription, or hand-maintained membership snapshots going forward).
  Until then, treat all absolute CAGRs as upper bounds.
- **Paper-portfolio PnL baseline** (`paper_portfolio.py` prints PnL off
  `initial_cash` even when the journal was created with different capital) —
  known cosmetic bug, harmless since the 2026-07-14 reset; fix at leisure.
