#!/usr/bin/env python3
"""Retrain the repo's FULL 14-model EnsembleModel on the winning research
config and save it to reports/models/ (gitignored registry).

Why this works as a live upgrade with zero repo edits:
  - analyze.py / signals resolve their model via best_ensemble_artifact(),
    which picks the NEWEST ensemble-type artifact in reports/models/ — saving
    ours after the baseline makes it the live model.
  - The target stays ABSOLUTE forward_return_5d so the decision layer's
    buy/sell thresholds (±0.5% expected 5d return) keep their meaning.
    (The cross-sectional variant is a laptop patch, not a hot deploy.)

Differences vs `ts train` (all validated by the harness first):
  - purged walk-forward (drop last 5 trading days of train; 5-day embargo)
  - temporal validation split (last 20% of train DATES, not row order)
  - winning feature set from out/summary.json (falls back to wide-96)

Usage: python3 deploy.py [--dry-run]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/home/ad2688/Desktop/Trade-system_v1")
sys.path.insert(0, str(REPO / "src"))

import polars as pl  # noqa: E402
from trading_system.models.ensemble import EnsembleModel  # noqa: E402
from trading_system.models.model_registry import save_model  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from harness import BASE14, HORIZON_TD, EMBARGO_TD, TARGET, folds, wide_columns  # noqa: E402

OUT = Path.home() / "trade-ops/research/out"
REGISTRY = REPO / "reports/models"


def pick_winning_features(feat: pl.DataFrame) -> tuple[str, list[str]]:
    """Winner = best IC t-stat among PURGED, ABSOLUTE-target variants."""
    try:
        summ = json.loads((OUT / "summary.json").read_text())
    except FileNotFoundError:
        summ = []
    eligible = [s for s in summ if s.get("purged") and not s.get("xsec")]
    if eligible:
        best = max(eligible, key=lambda s: s.get("ic_tstat", -9))
        name = best["variant"]
        cols = BASE14 if "base14" in name else wide_columns(feat)
        return name, [c for c in cols if c in feat.columns]
    return "C_wide_purged(default)", wide_columns(feat)


def main() -> int:
    dry = "--dry-run" in sys.argv
    t0 = time.time()
    feat = pl.read_parquet(REPO / "data/gold/features.parquet")
    win_name, cols = pick_winning_features(feat)
    print(f"deploying config: {win_name} ({len(cols)} features)", flush=True)

    work = (feat.select(["date", "ticker", TARGET] + cols)
                .drop_nulls(subset=[TARGET] + cols)
                .sort(["date", "ticker"]))
    dates = sorted(set(work["date"].to_list()))

    oos_frames, fold_records = [], []
    for fi, (tr_d, val_d, te_d) in enumerate(
            folds(dates, purge_td=HORIZON_TD, embargo_td=EMBARGO_TD)):
        tr = work.filter(pl.col("date").is_in(tr_d))
        va = work.filter(pl.col("date").is_in(val_d))
        te = work.filter(pl.col("date").is_in(te_d))
        print(f"fold {fi}: train {tr_d[0]}→{tr_d[-1]} ({tr.height} rows) "
              f"test {te_d[0]}→{te_d[-1]} ({te.height} rows)", flush=True)

        ens = EnsembleModel()
        ens.fit(tr.select(cols).to_numpy(), tr[TARGET].to_numpy(),
                va.select(cols).to_numpy(), va[TARGET].to_numpy(),
                feature_names=cols)
        best = ens.best_ensemble_name()
        preds = ens.predict(te.select(cols).to_numpy())
        score = preds.get(best, preds.get("ensemble_blend", np.zeros(te.height)))
        oos_frames.append(te.select(["date", "ticker"])
                            .with_columns(score=pl.Series(score.astype(np.float64))))
        fold_records.append({"fold": fi, "best_variant": best,
                             "blend_weights": ens.blend_weights})
        last_ens = ens

    oos = pl.concat(oos_frames)
    # daily spearman IC of the full ensemble, honest (purged) OOS
    ics = (oos.join(work.select(["date", "ticker", TARGET]), on=["date", "ticker"])
              .with_columns(rs=pl.col("score").rank().over("date"),
                            rt=pl.col(TARGET).rank().over("date"))
              .group_by("date").agg(ic=pl.corr("rs", "rt")).drop_nulls())
    v = ics["ic"].to_numpy()
    ic_m, ic_s = float(np.mean(v)), float(np.std(v, ddof=1))
    tstat = ic_m / ic_s * np.sqrt(len(v))
    print(f"\nfull-ensemble purged OOS: IC={ic_m:+.4f} t={tstat:.1f} "
          f"days={len(v)} ({(time.time()-t0)/60:.0f}m)")

    if dry:
        print("[dry-run] not saving")
        return 0

    oos.write_parquet(OUT / "oos_deployed_full_ensemble.parquet")
    # also refresh the repo's gitignored predictions file (feeds ml_ranker patch)
    oos.rename({"score": "score"}).write_parquet(
        REPO / "data/gold/predictions.parquet", compression="zstd")

    stamp = int(time.time())
    name = f"ensemble_{stamp}"
    save_model(
        last_ens, name=name, feature_columns=cols, target=TARGET,
        metadata={
            "model_type": "ensemble",
            "n_folds": len(fold_records),
            "best_variant": fold_records[-1]["best_variant"],
            "oos_rows": oos.height,
            "blend_weights": fold_records[-1]["blend_weights"],
            "research": {
                "config": win_name, "purge_td": HORIZON_TD,
                "embargo_td": EMBARGO_TD, "temporal_val": True,
                "oos_ic_mean": ic_m, "oos_ic_tstat": tstat,
                "provenance": "trade-ops/research/deploy.py 2026-07-14",
            },
        },
        registry=REGISTRY,
    )
    print(f"saved {name} -> {REGISTRY} (now newest ensemble ⇒ live for "
          f"ts signals/analyze/future-predict)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
