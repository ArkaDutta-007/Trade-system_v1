#!/usr/bin/env python3
"""Out-of-repo research harness for Trade-system_v1 model improvements.

Runs controlled walk-forward experiments on the real gold feature matrix and
backtests the resulting OOS predictions into P&L, so every proposed repo change
is justified by numbers before it lands on the laptop.

Variants (identical learners + eval everywhere; ONE thing changes at a time):
  A  base14_nopurge   current repo behaviour: 14 features, no purge, row-split val
  B  base14_purged    14 features + purged/embargoed walk-forward + temporal val
  C  wide_purged      B + wide feature set (technical/deep/macro/news, no leaks)
  D  wide_xsec        C + cross-sectionally demeaned target (relative 5d return)

Learners: LightGBM, XGBoost, HistGBM (the consistent top-IC trio in repo logs),
IC-weighted blend — a faster stand-in for the repo's 14-model EnsembleModel so
the variant deltas are measurable in hours, not days. The winning config is then
retrained with the FULL repo EnsembleModel by deploy.py.

Writes everything to ~/trade-ops/research/out/ (nothing in the repo).
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import date
from pathlib import Path

import numpy as np

REPO = Path("/home/ad2688/Desktop/Trade-system_v1")
sys.path.insert(0, str(REPO / "src"))

import polars as pl  # noqa: E402  (repo venv)
from scipy import stats  # noqa: E402

warnings.filterwarnings("ignore")

OUT = Path.home() / "trade-ops/research/out"
OUT.mkdir(parents=True, exist_ok=True)

TARGET = "forward_return_5d"
HORIZON_TD = 5          # trading days the label looks ahead
EMBARGO_TD = 5          # extra buffer between train end and test start

BASE14 = [
    "mom_5d", "mom_20d", "mom_60d", "mom_120d", "mom_12m1m",
    "vol_20d", "vol_60d", "rsi_14", "rel_vol_20",
    "sma_gap_50", "sma_gap_200", "breakout_20", "dd_from_high_60",
    "excess_ret_1d",
]

# Columns that must NEVER be features: identifiers, raw prices (non-stationary),
# targets, and same-bar fields already expressed as returns/gaps elsewhere.
NON_FEATURES = {
    "date", "ticker", "open", "high", "low", "close", "adj_close", "volume",
    "forward_return_5d", "forward_return_20d",
    "sma_10", "sma_20", "sma_50", "sma_200", "atr_14",       # raw price levels
    "avg_dollar_volume_20", "bench_ret_1d",                   # level / benchmark raw
}


def wide_columns(df: pl.DataFrame, max_null_frac: float = 0.40) -> list[str]:
    """All numeric, non-leaky columns that are populated most of the time.

    Event/sentiment columns (event_count, sent_decay_*, apprehension_score, …)
    are ~99.9% null — they only exist on event days — so requiring non-null
    across the wide set wipes the dataset. Columns above `max_null_frac` are
    excluded here; the survivors are zero-filled at variant build time
    (null == "no event/no reading" is the correct neutral for these).
    """
    n = df.height
    cols = []
    for c, dt in zip(df.columns, df.dtypes):
        if c in NON_FEATURES or not dt.is_numeric():
            continue
        if df[c].null_count() / n > max_null_frac:
            continue
        cols.append(c)
    return cols


# ── Purged walk-forward on trading-day index ─────────────────────────────────
def folds(dates: list, train_years=4, test_years=1, step_years=1,
          purge_td=0, embargo_td=0):
    """Yield (train_dates, val_dates, test_dates) with optional purge/embargo.

    Purge drops the last `purge_td` trading days of the train window (their
    5d-forward labels overlap the test window); embargo skips the first
    `embargo_td` trading days of the test window. Validation = last 20% of the
    (purged) train dates — temporal, never row-order.
    """
    d0, dN = dates[0], dates[-1]
    idx = {d: i for i, d in enumerate(dates)}
    cur = d0
    while True:
        tr_end_cal = cur.replace(year=cur.year + train_years)
        te_end_cal = tr_end_cal.replace(year=tr_end_cal.year + test_years)
        if tr_end_cal >= dN:
            return
        tr_dates = [d for d in dates if cur <= d < tr_end_cal]
        te_dates = [d for d in dates if tr_end_cal <= d < min(te_end_cal, dN)]
        if purge_td:
            tr_dates = tr_dates[:-purge_td]
        if embargo_td:
            te_dates = te_dates[embargo_td:]
        if len(tr_dates) < 260 or not te_dates:
            cur = cur.replace(year=cur.year + step_years)
            continue
        vsplit = int(len(tr_dates) * 0.8)
        yield tr_dates[:vsplit], tr_dates[vsplit:], te_dates
        cur = cur.replace(year=cur.year + step_years)


# ── Learners ─────────────────────────────────────────────────────────────────
def make_learners():
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.ensemble import HistGradientBoostingRegressor
    return {
        "lgbm": lgb.LGBMRegressor(
            num_leaves=63, learning_rate=0.05, n_estimators=400,
            min_child_samples=30, feature_fraction=0.8, bagging_fraction=0.8,
            bagging_freq=5, n_jobs=8, verbosity=-1, seed=42),
        "xgb": xgb.XGBRegressor(
            max_depth=6, learning_rate=0.05, n_estimators=400,
            subsample=0.8, colsample_bytree=0.8, n_jobs=8,
            random_state=42, verbosity=0, tree_method="hist"),
        "hist_gbm": HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=63,
            min_samples_leaf=30, random_state=42),
    }


def daily_ic(df: pl.DataFrame, score="score", target="y") -> pl.DataFrame:
    """Per-date Spearman IC."""
    return (
        df.with_columns(
            rs=pl.col(score).rank().over("date"),
            rt=pl.col(target).rank().over("date"),
        )
        .group_by("date")
        .agg(ic=pl.corr("rs", "rt"))
        .drop_nulls()
        .sort("date")
    )


def ic_summary(ics: pl.DataFrame) -> dict:
    v = ics["ic"].to_numpy()
    n = len(v)
    m = float(np.mean(v)) if n else float("nan")
    s = float(np.std(v, ddof=1)) if n > 1 else float("nan")
    return {
        "ic_mean": m, "ic_std": s,
        "ic_tstat": m / s * np.sqrt(n) if n > 1 and s > 0 else float("nan"),
        "ic_ir": m / s if n > 1 and s > 0 else float("nan"),
        "n_days": n,
        "pct_days_positive": float(np.mean(v > 0)) if n else float("nan"),
    }


# ── One experiment variant ───────────────────────────────────────────────────
def run_variant(name: str, feat: pl.DataFrame, feature_cols: list[str],
                purge: bool, xsec: bool, log) -> dict:
    t0 = time.time()
    work = feat
    tgt = TARGET
    if xsec:
        work = work.with_columns(
            (pl.col(TARGET) - pl.col(TARGET).mean().over("date")).alias("xsec_target")
        )
        tgt = "xsec_target"

    need = ["date", "ticker", tgt, TARGET] + feature_cols
    # Rows must have labels + the dense core features (nulls there = warmup —
    # same row filter for every variant, so comparisons are apples-to-apples).
    # Sparse extras (event/sentiment/nonlinear warmups) are zero-filled.
    core = list({tgt, TARGET} | {c for c in BASE14 if c in feature_cols})
    work = (work.select(sorted(set(c for c in need if c in work.columns),
                               key=need.index))
            .drop_nulls(subset=core)
            .with_columns([pl.col(c).fill_null(0.0) for c in feature_cols])
            .sort(["date", "ticker"]))
    dates = sorted(set(work["date"].to_list()))
    if not dates:
        raise RuntimeError(f"{name}: no rows survived label filtering")

    purge_td = HORIZON_TD if purge else 0
    embargo_td = EMBARGO_TD if purge else 0

    oos_frames, fold_stats = [], []
    for fi, (tr_d, val_d, te_d) in enumerate(
            folds(dates, purge_td=purge_td, embargo_td=embargo_td)):
        tr = work.filter(pl.col("date").is_in(tr_d))
        va = work.filter(pl.col("date").is_in(val_d))
        te = work.filter(pl.col("date").is_in(te_d))
        if not purge:
            # replicate current repo behaviour: row-order 80/20 split of the
            # whole train window (val leaks across dates, as in train.py)
            allrows = work.filter(pl.col("date").is_in(tr_d + val_d))
            split = int(allrows.height * 0.8)
            tr, va = allrows.head(split), allrows.tail(allrows.height - split)

        Xtr = tr.select(feature_cols).to_numpy()
        ytr = tr[tgt].to_numpy()
        Xva = va.select(feature_cols).to_numpy()
        yva = va[tgt].to_numpy()
        Xte = te.select(feature_cols).to_numpy()

        preds_va, preds_te, ics_val = {}, {}, {}
        for lname, model in make_learners().items():
            model.fit(Xtr, ytr)
            pv = model.predict(Xva)
            preds_va[lname] = pv
            preds_te[lname] = model.predict(Xte)
            ic = stats.spearmanr(pv, yva).statistic
            ics_val[lname] = 0.0 if np.isnan(ic) else max(ic, 0.0)

        wsum = sum(ics_val.values())
        wts = ({k: v / wsum for k, v in ics_val.items()} if wsum > 0
               else {k: 1 / len(ics_val) for k in ics_val})
        blend_te = sum(wts[k] * preds_te[k] for k in preds_te)

        oos_frames.append(
            te.select(["date", "ticker", TARGET])
              .with_columns(score=pl.Series(blend_te.astype(np.float64))))
        fold_stats.append({"fold": fi, "train_days": len(tr_d),
                           "test_days": len(te_d), "blend_wts": wts})
        log(f"  {name} fold {fi}: {tr_d[0]}→{te_d[0]}..{te_d[-1]} "
            f"rows tr={tr.height} te={te.height} wts={ {k: round(v,2) for k,v in wts.items()} }")

    oos = pl.concat(oos_frames)
    ics = daily_ic(oos.rename({TARGET: "y"}))
    summ = ic_summary(ics)
    summ.update({"variant": name, "n_features": len(feature_cols),
                 "purged": purge, "xsec": xsec,
                 "minutes": round((time.time() - t0) / 60, 1),
                 "n_folds": len(fold_stats)})
    oos.write_parquet(OUT / f"oos_{name}.parquet")
    ics.write_parquet(OUT / f"ic_{name}.parquet")
    (OUT / f"folds_{name}.json").write_text(json.dumps(fold_stats, default=str, indent=1))
    log(f"== {name}: IC={summ['ic_mean']:+.4f} t={summ['ic_tstat']:.1f} "
        f"IR={summ['ic_ir']:.3f} days+={summ['pct_days_positive']:.0%} "
        f"({summ['minutes']}m)")
    return summ


def main():
    logf = open(OUT / "harness.log", "a", buffering=1)

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")

    log("loading gold features…")
    feat = pl.read_parquet(REPO / "data/gold/features.parquet")
    wide = wide_columns(feat)
    log(f"rows={feat.height} wide feature set = {len(wide)} cols")

    try:  # resume: keep results from a previous partial run
        results = json.loads((OUT / "summary.json").read_text())
    except FileNotFoundError:
        results = []
    done = {r["variant"] for r in results}

    for name, cols, purge, xsec in [
        ("A_base14_nopurge", BASE14, False, False),
        ("B_base14_purged",  BASE14, True,  False),
        ("C_wide_purged",    wide,   True,  False),
        ("D_wide_xsec",      wide,   True,  True),
    ]:
        if name in done and (OUT / f"oos_{name}.parquet").exists():
            log(f"skip {name} (already done)")
            continue
        cols = [c for c in cols if c in feat.columns]
        results.append(run_variant(name, feat, cols, purge, xsec, log))
        (OUT / "summary.json").write_text(json.dumps(results, indent=2))

    log("ALL VARIANTS DONE")
    logf.close()


if __name__ == "__main__":
    main()
