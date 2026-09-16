#!/usr/bin/env python3
"""Backtest the harness OOS predictions into P&L + robustness statistics.

For each variant's OOS score frame:
  - build daily top-k portfolios under three sizing schemes
      equal      equal-weight top-k (what the repo strategies do today)
      score_iv   weight ∝ score-rank × inverse 20d vol   (conviction × risk)
      inv_vol    weight ∝ inverse 20d vol                (pure risk parity)
  - run the REPO's own vectorized backtester (same costs/delay/caps as ts
    backtest) — flat costs and a 3× stressed-cost run
  - apply overlays on the daily net returns: 15% vol-target + drawdown throttle
  - compute Deflated Sharpe (Bailey & López de Prado) across ALL configurations
    tried, and PBO via CSCV (16 blocks)

Everything reads/writes ~/trade-ops/research/out/. Repo is import-only.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import add_repo_to_path, ops_root  # noqa: E402

REPO = add_repo_to_path()

import polars as pl  # noqa: E402
from scipy import stats  # noqa: E402
from trading_system.backtesting import compute_metrics, run_vectorized_backtest  # noqa: E402
from trading_system.backtesting.slippage import CostModel  # noqa: E402

OUT = ops_root() / "research" / "out"
TOP_K = 20
CAP = 0.10
ANN = 252


def build_weights(oos: pl.DataFrame, vol: pl.DataFrame, scheme: str) -> pl.DataFrame:
    df = (oos.join(vol, on=["date", "ticker"], how="left")
             .with_columns(pl.col("vol_20d").fill_null(pl.col("vol_20d").median()))
             .with_columns(rk=pl.col("score").rank(descending=True).over("date"))
             .filter(pl.col("rk") <= TOP_K))
    if scheme == "equal":
        df = df.with_columns(w=pl.lit(1.0))
    elif scheme == "inv_vol":
        df = df.with_columns(w=1.0 / (pl.col("vol_20d") + 1e-6))
    elif scheme == "score_iv":
        df = df.with_columns(
            w=(TOP_K + 1 - pl.col("rk")) / (pl.col("vol_20d") + 1e-6))
    else:
        raise ValueError(scheme)
    df = df.with_columns(weight=pl.col("w") / pl.col("w").sum().over("date"))
    df = df.with_columns(weight=pl.min_horizontal(pl.col("weight"), pl.lit(CAP)))
    df = df.with_columns(weight=pl.col("weight") / pl.col("weight").sum().over("date"))
    return df.select(["date", "ticker", "weight"])


def overlay(net: np.ndarray, vol_target=0.15, dd_throttle=0.10) -> np.ndarray:
    """Vol-target + drawdown-throttle overlay on a daily net return stream."""
    out = np.zeros_like(net)
    eq = 1.0
    peak = 1.0
    window = []
    expo = 1.0
    for i, r in enumerate(net):
        out[i] = expo * r
        eq *= 1 + out[i]
        peak = max(peak, eq)
        window.append(r)
        if len(window) > 20:
            window.pop(0)
        rv = np.std(window, ddof=1) * np.sqrt(ANN) if len(window) >= 10 else vol_target
        expo = min(vol_target / max(rv, 1e-4), 1.5)          # vol target, max 1.5x
        if eq / peak - 1 < -dd_throttle:
            expo *= 0.5                                       # halve in a >10% dd
    return out


# ── Deflated Sharpe + PBO ────────────────────────────────────────────────────
def psr(returns: np.ndarray, sr_benchmark: float) -> float:
    """Probabilistic Sharpe Ratio (non-annualized inputs)."""
    n = len(returns)
    sr = np.mean(returns) / np.std(returns, ddof=1)
    g3 = stats.skew(returns)
    g4 = stats.kurtosis(returns, fisher=False)
    denom = np.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr**2, 1e-12) / (n - 1))
    return float(stats.norm.cdf((sr - sr_benchmark) / denom))


def deflated_sharpe(returns: np.ndarray, trial_srs: list[float]) -> dict:
    """DSR: PSR against the expected max Sharpe of N random trials."""
    n_trials = len(trial_srs)
    var_sr = np.var(trial_srs, ddof=1) if n_trials > 1 else 0.0
    emc = 0.5772156649
    z1 = stats.norm.ppf(1 - 1.0 / n_trials) if n_trials > 1 else 0.0
    z2 = stats.norm.ppf(1 - 1.0 / (n_trials * np.e)) if n_trials > 1 else 0.0
    sr0 = np.sqrt(var_sr) * ((1 - emc) * z1 + emc * z2)
    return {"sr0_daily": float(sr0), "dsr": psr(returns, sr0),
            "n_trials": n_trials}


def pbo_cscv(ret_matrix: np.ndarray, n_blocks=16) -> float:
    """Probability of Backtest Overfitting via CSCV (returns T×N matrix)."""
    T, N = ret_matrix.shape
    blocks = np.array_split(np.arange(T), n_blocks)
    half = n_blocks // 2
    lambdas = []
    for combo in itertools.combinations(range(n_blocks), half):
        is_idx = np.concatenate([blocks[b] for b in combo])
        oos_idx = np.concatenate([blocks[b] for b in range(n_blocks) if b not in combo])
        m_is = ret_matrix[is_idx].mean(0) / (ret_matrix[is_idx].std(0, ddof=1) + 1e-12)
        m_oos = ret_matrix[oos_idx].mean(0) / (ret_matrix[oos_idx].std(0, ddof=1) + 1e-12)
        star = int(np.argmax(m_is))
        rank_oos = stats.rankdata(m_oos)[star] / (N + 1)      # relative OOS rank
        logit = np.log(rank_oos / (1 - rank_oos))
        lambdas.append(logit)
    lambdas = np.array(lambdas)
    return float(np.mean(lambdas <= 0))


def main():
    t0 = time.time()
    prices = pl.read_parquet(REPO / "data/bronze/ohlcv_daily.parquet")
    feat = pl.read_parquet(REPO / "data/gold/features.parquet")
    vol = feat.select(["date", "ticker", "vol_20d"])

    variants = sorted(p.stem.replace("oos_", "") for p in OUT.glob("oos_*.parquet"))
    print(f"variants found: {variants}")

    rows, ret_streams = [], {}
    for vname in variants:
        oos = pl.read_parquet(OUT / f"oos_{vname}.parquet")
        for scheme in ("equal", "score_iv", "inv_vol"):
            w = build_weights(oos, vol, scheme)
            for costs, tag in ((CostModel(1.0, 2.0, 1.0), "flat"),
                               (CostModel(3.0, 6.0, 3.0), "stress3x")):
                res = run_vectorized_backtest(
                    prices, w, cost=costs, signal_delay_days=1,
                    initial_cash=10_000.0, max_gross_exposure=1.0,
                    max_position_weight=CAP, benchmark="SPY")
                net = res.daily["net_ret"].to_numpy()
                m = compute_metrics(net, turnover=res.daily["turnover"].to_numpy())
                key = f"{vname}|{scheme}|{tag}"
                ret_streams[key] = net
                rows.append({"config": key, **{k: float(np.round(v, 4))
                             for k, v in m.items() if isinstance(v, (int, float))}})
                if tag == "flat":                             # overlay on flat only
                    ov = overlay(net)
                    mo = compute_metrics(ov)
                    key2 = f"{vname}|{scheme}|overlay"
                    ret_streams[key2] = ov
                    rows.append({"config": key2, **{k: float(np.round(v, 4))
                                 for k, v in mo.items() if isinstance(v, (int, float))}})
                print(f"  {key}: CAGR={m['CAGR']:+.2%} Sharpe={m['Sharpe']:.2f} "
                      f"MaxDD={m['MaxDrawdown']:.1%}", flush=True)

    # robustness stats across everything we tried
    keys = list(ret_streams)
    minlen = min(len(v) for v in ret_streams.values())
    mat = np.column_stack([ret_streams[k][-minlen:] for k in keys])
    trial_srs = [float(np.mean(mat[:, j]) / np.std(mat[:, j], ddof=1))
                 for j in range(mat.shape[1])]
    pbo = pbo_cscv(mat)

    for r in rows:
        net = ret_streams[r["config"]][-minlen:]
        d = deflated_sharpe(net, trial_srs)
        r["DSR"] = round(d["dsr"], 4)
    best = max(rows, key=lambda r: r.get("Sharpe", -9))

    out = {"pbo_cscv": pbo, "n_configs": len(keys),
           "best_by_sharpe": best["config"], "table": rows}
    (OUT / "backtest_report.json").write_text(json.dumps(out, indent=2))
    print(f"\nPBO(CSCV)={pbo:.2%} across {len(keys)} configs; "
          f"best={best['config']} Sharpe={best['Sharpe']} DSR={best['DSR']} "
          f"({(time.time()-t0)/60:.1f}m)")


if __name__ == "__main__":
    main()
