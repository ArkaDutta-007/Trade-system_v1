#!/usr/bin/env python3
"""Multi-horizon, IC-IR-weighted composite signal (+ momentum sleeve).

WHY
---
The live pipeline ranks names on the 5-day ensemble. Measured on this system's
own walk-forward metrics (models_store/forecast/*/metrics.json) the 5d target is
the WEAKEST signal by an order of magnitude:

        horizon   IC      ICIR
        5d        0.010   ~0.05     <- what `ts signals` / picks run on
        21d       0.021   1.60
        63d       0.055   3.40
        126d      0.057   2.23
        252d      0.104   3.03

This matches the literature: short-horizon return predictability is marginal
(R² ~ 0); quarterly-to-annual horizons are where tree ensembles retain signal
(Gu-Kelly-Xiu 2020 and follow-ups). The repo already trains and commits the
longer-horizon forecasters — they are simply not blended into the live book.

WHAT
----
  1. Score every name on each stored horizon model (63d / 126d / 252d; 21d is
     included at low weight, 5d is excluded — its ICIR rounds to zero).
  2. Convert each to a cross-sectional RANK (0..1) per date — ranks are
     comparable across horizons whose raw scales differ by 10x, and are robust
     to the fat-tailed raw forecasts.
  3. Weight horizons by their measured out-of-sample ICIR, the information
     ratio of the signal — the standard "favour high-IR alphas" rule for
     combining forecasts. Weights are re-read from metrics.json, so a retrain
     that changes a horizon's skill automatically changes its influence.
  4. Add a MOMENTUM sleeve (12-1 month, the classic Jegadeesh-Titman definition)
     as an independent alpha. In this system's own backtests rule-based
     momentum_rotation (Sharpe 1.21) beat the ML model (1.03) risk-adjusted;
     "factor momentum captures most of the predictability of ML strategies" is
     the published finding. It is blended by IR like any other signal.
  5. Output: composite score in rank units, ready for picks_v2's gates,
     risk-adjustment, diversification and HRP allocation.

Usage:
    python3 signals_v3.py              # top-20 table + horizon weights
    python3 signals_v3.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/ad2688/Desktop/Trade-system_v1")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import polars as pl  # noqa: E402

HORIZONS = (21, 63, 126, 252)
MOMENTUM_ICIR = 2.0     # prior for the momentum sleeve's IR (see below)
# ^ Momentum has no walk-forward metrics.json of its own. Its blended weight
#   is set from a conservative prior: its own long-run IC on this universe
#   sits near the 63d model's, and its backtest Sharpe (1.21) exceeds the ML
#   model's (1.03). 2.0 lands it between the 126d and 63d/252d models rather
#   than letting it dominate. Revisit once the paper books have ~60 sessions.


def _features() -> pl.DataFrame:
    return pl.read_parquet(REPO / "data/gold/features.parquet")


def horizon_scores(features: pl.DataFrame) -> tuple[dict[int, pl.DataFrame], dict[int, float]]:
    """{horizon: DataFrame(ticker, score)} and {horizon: icir} from the store."""
    from trading_system.models.store import load_forecast_model
    from trading_system.models.forecast_train import forecast_scores_latest
    frames, icir = {}, {}
    for h in HORIZONS:
        loaded = load_forecast_model(h, REPO / "models_store")
        if loaded is None:
            continue
        model, meta = loaded
        try:
            tickers, scores, _ = forecast_scores_latest(model, meta, features)
        except Exception as e:  # noqa: BLE001
            print(f"  ! {h}d scoring failed: {e}", file=sys.stderr)
            continue
        frames[h] = pl.DataFrame({"ticker": list(tickers),
                                  f"s{h}": np.asarray(scores, dtype=float)})
        best = meta.get("best_model")
        pm = (meta.get("per_model") or {}).get(best) or {}
        icir[h] = float(pm.get("icir", 0.0))
    return frames, icir


def momentum_sleeve(features: pl.DataFrame) -> pl.DataFrame:
    """12-1 month momentum (skip the most recent month — short-term reversal)."""
    last = features["date"].max()
    d = features.filter(pl.col("date") == last)
    if "mom_12m1m" in d.columns:
        return d.select(["ticker", pl.col("mom_12m1m").alias("mom")])
    return d.select(["ticker", pl.col("mom_120d").alias("mom")])


def blend(frames: dict[int, pl.DataFrame], icir: dict[int, float],
          mom: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, float]]:
    """IC-IR weighted average of cross-sectional ranks."""
    df = mom
    for h, f in frames.items():
        df = df.join(f, on="ticker", how="outer_coalesce" if hasattr(pl, "__version__") else "outer")
    cols = [f"s{h}" for h in frames] + ["mom"]
    # ranks in (0,1]; nulls (name missing from a horizon) get the median rank
    # so an absent signal neither helps nor hurts.
    for c in cols:
        df = df.with_columns(
            (pl.col(c).rank(method="average") / pl.col(c).count()).alias(f"r_{c}"))
        df = df.with_columns(pl.col(f"r_{c}").fill_null(0.5))
    raw_w = {f"s{h}": max(icir.get(h, 0.0), 0.0) for h in frames}
    raw_w["mom"] = MOMENTUM_ICIR
    tot = sum(raw_w.values()) or 1.0
    w = {k: v / tot for k, v in raw_w.items()}
    expr = sum((pl.col(f"r_{c}") * w[c] for c in cols), pl.lit(0.0))
    df = df.with_columns(expr.alias("score_v3")).sort("score_v3", descending=True)
    return df, w


def build() -> dict:
    feats = _features()
    frames, icir = horizon_scores(feats)
    if not frames:
        raise SystemExit("no horizon models loaded from models_store/forecast")
    mom = momentum_sleeve(feats)
    df, w = blend(frames, icir, mom)
    return {"as_of": str(feats["date"].max()), "weights": w,
            "icir": {f"{h}d": round(v, 3) for h, v in icir.items()},
            "frame": df}


def render(res: dict, top: int = 20) -> str:
    df = res["frame"].head(top)
    L = [f"Signals v3 · as of {res['as_of']} · IC-IR-weighted blend of horizons + momentum",
         "weights: " + "  ".join(f"{k}={v:.2f}" for k, v in res["weights"].items())
         + "   (from OOS ICIR: " + ", ".join(f"{k} {v}" for k, v in res["icir"].items()) + ")",
         ""]
    hcols = [c for c in df.columns if c.startswith("r_s")]
    L.append(f"{'#':>2}  {'Ticker':<7}{'score':>7}  " + "".join(f"{c[2:]:>6}" for c in hcols) + f"{'mom':>6}")
    for i, r in enumerate(df.iter_rows(named=True), 1):
        L.append(f"{i:>2}  {r['ticker']:<7}{r['score_v3']:>7.3f}  "
                 + "".join(f"{r[c]:>6.2f}" for c in hcols) + f"{r['r_mom']:>6.2f}")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    res = build()
    print(render(res, a.top))
    if a.json:
        out = {k: v for k, v in res.items() if k != "frame"}
        out["scores"] = res["frame"].select(["ticker", "score_v3"]).to_dicts()
        a.json.write_text(json.dumps(out, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
