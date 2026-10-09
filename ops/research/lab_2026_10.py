#!/usr/bin/env python3
"""Research lab, October 2026 — candidate improvements under one leak-free protocol.

Every variant is a ``TrainSpec`` (features / label / training universe / engine). Each is judged twice:

  LONG  — causal walk-forward 2008 → 2026 on the 1000-name panel (refit every 126 sessions, 21 & 63 day
          horizons), evaluated ONLY on names that were the top-500 by trailing dollar volume on that day.
          That removes the specific look-ahead the model exploited ("small then, big now"); it does not
          remove classic survivorship (dead names have no free price history before 2024).
  PIT   — the honest test: Nov 2024 → Sep 2026 on a universe rebuilt each day from whole-market bars
          (top-1000 by trailing $volume using data to that day), every name given identical data.

Signals: per-date rank IC, decile spread, paired t vs the baseline on non-overlapping dates. Books:
the production book rules (20 names, gates, sector cap, vol brake, 200-day trend overlay, partial trading)
in the research simulator with its cost model. Results → reports/alpha/lab_2026_10/.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from trading_system.alpha import model as M  # noqa: E402
from trading_system.alpha import panel as P  # noqa: E402
from trading_system.alpha.backtest import ColumnAlpha, _weight_fn  # noqa: E402
from trading_system.alpha.portfolio import BookConfig, market_regime_on  # noqa: E402
from trading_system.config import get_config  # noqa: E402
from trading_system.research.stats import bootstrap_ci, sharpe  # noqa: E402
from trading_system.research.wfbacktest import WalkForwardConfig, build_panel, run_walk_forward  # noqa: E402

cfg = get_config(str(REPO / "configs/default.yaml"))
OUT = cfg.path("reports") / "alpha" / "lab_2026_10"
OUT.mkdir(parents=True, exist_ok=True)
H = (21, 63)
W = {21: 0.3, 63: 0.7}
LONG_START = date(2008, 1, 1)
EVAL_TOP = 500
BASE_FEATS = tuple(P.MODEL_FEATURES)
NEW_FEATS = BASE_FEATS + tuple(P.PRICE_CANDIDATES)
INS_FEATS = BASE_FEATS + tuple(P.INSIDER_FEATURES)
ALL_FEATS = BASE_FEATS + tuple(P.CANDIDATE_FEATURES)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


VARIANTS = {
    "base":            M.TrainSpec(horizons=H),
    "debias":          M.TrainSpec(horizons=H, train_top=EVAL_TOP),
    "debias+feats":    M.TrainSpec(horizons=H, train_top=EVAL_TOP, features=NEW_FEATS),
    "debias+sectorlab": M.TrainSpec(horizons=H, train_top=EVAL_TOP, label="ys"),
    "debias+betalab":  M.TrainSpec(horizons=H, train_top=EVAL_TOP, label="yr"),
    "debias+lgbm":     M.TrainSpec(horizons=H, train_top=EVAL_TOP, engine="lgbm"),
    "debias+ridge":    M.TrainSpec(horizons=H, train_top=EVAL_TOP, engine="ridge", seeds=(0,)),
    # batch 2 — additions on top of the CURRENT model (debiasing did not help on the PIT test)
    "base+feats":      M.TrainSpec(horizons=H, features=NEW_FEATS),
    "base+insider":    M.TrainSpec(horizons=H, features=INS_FEATS),
    "base+all":        M.TrainSpec(horizons=H, features=ALL_FEATS),
    "base+ridge":      M.TrainSpec(horizons=H, engine="ridge", seeds=(0,)),
}
BLENDS = {"debias·ens(xgb+lgbm+ridge)": ["debias", "debias+lgbm", "debias+ridge"],
          "debias·ens(xgb+lgbm)": ["debias", "debias+lgbm"],
          "base·ens(xgb+ridge)": ["base", "base+ridge"]}


def zcomp(sc: pl.DataFrame) -> pl.DataFrame:
    """date, ticker, comp = Σ_h W_h · z_h (per-date z-scores of each horizon's score)."""
    z = sc.with_columns(z=(pl.col("score") - pl.col("score").mean().over(["date", "horizon"]))
                        / (pl.col("score").std().over(["date", "horizon"]) + 1e-9))
    w = pl.DataFrame({"horizon": list(W), "w": list(W.values())}).with_columns(pl.col("horizon").cast(z.schema["horizon"]))
    return z.join(w, on="horizon").group_by(["date", "ticker"]).agg(comp=(pl.col("z") * pl.col("w")).sum())


def blend(frames: list[pl.DataFrame]) -> pl.DataFrame:
    zs = []
    for i, f in enumerate(frames):
        zs.append(f.with_columns(((pl.col("score") - pl.col("score").mean().over(["date", "horizon"]))
                                  / (pl.col("score").std().over(["date", "horizon"]) + 1e-9)).alias(f"z{i}"))
                  .select("date", "ticker", "horizon", f"z{i}"))
    j = zs[0]
    for f in zs[1:]:
        j = j.join(f, on=["date", "ticker", "horizon"], how="inner")
    return j.select("date", "ticker", "horizon", score=pl.mean_horizontal([f"z{i}" for i in range(len(frames))]))


# ── signal statistics ─────────────────────────────────────────────────────────

def ic_table(scores: pl.DataFrame, labels: pl.DataFrame, mask: pl.DataFrame | None = None) -> pl.DataFrame:
    """date, horizon, ic, spread (top − bottom decile mean forward return)."""
    rows = []
    for h in H:
        s = scores.filter(pl.col("horizon") == h).join(labels.select("date", "ticker", f"fwd_{h}"), on=["date", "ticker"])
        if mask is not None:
            s = s.join(mask, on=["date", "ticker"], how="semi")
        s = s.drop_nulls(f"fwd_{h}").with_columns(
            dec=((pl.col("score").rank("ordinal").over("date") - 1) * 10 // pl.col("score").count().over("date")))
        g = s.group_by("date").agg(ic=pl.corr("score", f"fwd_{h}", method="spearman"), n=pl.len(),
                                   spread=pl.col(f"fwd_{h}").filter(pl.col("dec") == 9).mean()
                                   - pl.col(f"fwd_{h}").filter(pl.col("dec") == 0).mean())
        rows.append(g.filter(pl.col("n") >= 30).with_columns(horizon=pl.lit(h)))
    return pl.concat(rows).sort(["horizon", "date"])


def ic_summary(t: pl.DataFrame, base: pl.DataFrame | None = None) -> dict:
    out = {}
    for h in H:
        g = t.filter(pl.col("horizon") == h).sort("date")
        x = g["ic"].to_numpy()
        step = h  # non-overlapping
        xs = x[::step]
        d = {"ic": float(np.nanmean(x)), "icir": float(np.nanmean(x) / (np.nanstd(x) + 1e-12)),
             "t": float(np.nanmean(xs) / (np.nanstd(xs, ddof=1) + 1e-12) * np.sqrt(len(xs))),
             "spread": float(g["spread"].mean()), "n_dates": len(x)}
        if base is not None:
            m = g.join(base.filter(pl.col("horizon") == h).select("date", ic_b="ic"), on="date")
            dd = (m["ic"] - m["ic_b"]).to_numpy()[::step]
            d["paired_t"] = float(dd.mean() / (dd.std(ddof=1) + 1e-12) * np.sqrt(len(dd))) if len(dd) > 3 else None
        out[h] = d
    return out


# ── book ──────────────────────────────────────────────────────────────────────

def book_run(pdata, comp: pl.DataFrame, gate: pl.DataFrame, sector_of: dict, regime: dict, oos: date,
             label: str, bk: BookConfig | None = None, rate: float = 0.35):
    bk = bk or replace(BookConfig(), min_price=0.0, min_dollar_volume=0.0)
    f = gate.join(comp, on=["date", "ticker"], how="left").with_columns(
        comp=pl.when(pl.col("elig")).then(pl.col("comp").cast(pl.Float64))).filter(pl.col("comp").is_not_null())
    wcfg = WalkForwardConfig(oos_start=oos, rebalance_days=21, retrain_days=10**6, min_train_days=5, horizon=21,
                             top_k=bk.top_k, max_weight=bk.max_weight, min_dollar_volume=0.0, min_price=0.0,
                             partial_trade_rate=rate, respect_target_gross=True)
    return run_walk_forward(pdata, f, ["comp"], lambda: ColumnAlpha("comp"), wcfg,
                            weight_fn=_weight_fn(bk, sector_of, regime), label=label, progress=False)


def book_stats(r: np.ndarray) -> dict:
    eq = np.cumprod(1 + r)
    y = len(r) / 252
    ci = bootstrap_ci(r, n_boot=300)
    return {"cagr": float(eq[-1] ** (1 / y) - 1), "sharpe": sharpe(r), "sharpe_ci": [ci["lo"], ci["hi"]],
            "vol": float(r.std() * 252 ** 0.5), "maxdd": float((eq / np.maximum.accumulate(eq) - 1).min())}


def excess_ci(r: np.ndarray, b: np.ndarray) -> dict:
    n = min(len(r), len(b))
    ci = bootstrap_ci(r[-n:] - b[-n:], stat=lambda x: x.mean() * 252, n_boot=300)
    return {"excess": ci["point"], "lo": ci["lo"], "hi": ci["hi"], "p_gt_0": ci["p_gt_0"]}


# ── LONG protocol ─────────────────────────────────────────────────────────────

def long_protocol(names: list[str]) -> dict:
    pn = P.load_panel(cfg)
    frame, _ = M.prepare(pn, BASE_FEATS)
    mask = frame.filter(pl.col("liq_rank") <= EVAL_TOP).select("date", "ticker")
    labels = pn.select("date", "ticker", *[f"fwd_{h}" for h in H])
    scores = {}
    for name in names:
        p = OUT / f"long_{name}.parquet"
        if p.exists():
            scores[name] = pl.read_parquet(p); log(f"LONG {name}: cached"); continue
        t0 = time.time()
        sc = M.causal_scores(pn, VARIANTS[name], refit_every=126, min_train_days=1260, oos_start=LONG_START, progress=False)
        sc.write_parquet(p)
        scores[name] = sc
        log(f"LONG {name}: {sc.height:,} scores in {(time.time() - t0) / 60:.1f} min")
    for bname, parts in BLENDS.items():
        if all(x in scores for x in parts):
            scores[bname] = blend([scores[x] for x in parts])
    ics = {k: ic_table(v, labels, mask) for k, v in scores.items()}
    res = {"ic": {k: ic_summary(v, ics.get("base") if k != "base" else None) for k, v in ics.items()}}
    # books on the debiased eligible set
    prices = P.load_prices(cfg, start="2006-01-01")
    pdata = build_panel(prices)
    sector_of = dict(pn.select("ticker", "sector").unique(subset=["ticker"]).iter_rows())
    regime = {d: market_regime_on(v) for d, v in pn.group_by("date").agg(pl.col("mkt_trend_200").first()).iter_rows()}
    gate = (frame.select("date", "ticker", "adj_close", "close", "vol_63", "liq_rank")
                 .with_columns(elig=(pl.col("close") >= 5) & (pl.col("liq_rank") <= EVAL_TOP) & (pl.col("vol_63") / 252 ** 0.5 <= 0.07))
                 .select("date", "ticker", "adj_close", "elig"))
    books = {}
    for k, v in scores.items():
        res_k = book_run(pdata, zcomp(v), gate, sector_of, regime, date(2009, 1, 1), k)
        books[k] = res_k.returns()
    dates = res_k.daily["date"].to_list()
    # passive: equal weight of the same eligible set, SPY
    el = gate.filter(pl.col("elig")).select("date", "ticker")
    pr = prices.sort(["ticker", "date"]).with_columns(r=pl.col("adj_close") / pl.col("adj_close").shift(1).over("ticker") - 1,
                                                      prev=pl.col("date").shift(1).over("ticker"))
    ew = dict(pr.join(el.rename({"date": "prev"}), on=["prev", "ticker"]).filter(pl.col("r").abs() < 5)
                .group_by("date").agg(pl.col("r").mean()).iter_rows())
    books["passive: eligible EW"] = np.array([ew.get(d, 0.0) for d in dates])
    tick = list(pdata.tickers); di = pdata.date_index()
    if "SPY" in tick:
        books["passive: SPY"] = np.array([pdata.ret[di[d], tick.index("SPY")] for d in dates])
    res["book"] = {k: book_stats(v) for k, v in books.items()}
    res["book_excess_vs_base"] = {k: excess_ci(v, books["base"]) for k, v in books.items() if k != "base" and not k.startswith("passive")}
    res["window"] = [str(dates[0]), str(dates[-1])]
    return res


# ── PIT protocol ──────────────────────────────────────────────────────────────

def pit_panel() -> tuple[pl.DataFrame, pl.DataFrame, list[str]]:
    p = OUT / "pit_panel.parquet"
    B = cfg.path("data_bronze") / "massive"
    tk = pl.read_parquet(B / "tickers.parquet", columns=["ticker", "type"])
    ok = tk.filter(pl.col("type").is_in(["CS", "ADRC"]))["ticker"].to_list()
    px = (pl.scan_parquet(B / "ohlcv_all.parquet").filter(~pl.col("otc"), pl.col("ticker").is_in(ok), pl.col("close") > 0, pl.col("adj_close") > 0)
            .select("date", "ticker", "open", "high", "low", "close", "adj_close", "volume").collect())
    uni = (px.sort(["ticker", "date"]).with_columns(dv=pl.col("close") * pl.col("volume"))
             .with_columns(med63=pl.col("dv").rolling_median(63, min_samples=40).over("ticker"))
             .filter(pl.col("close") >= 5, pl.col("med63").is_not_null())
             .with_columns(rk=pl.col("med63").rank(descending=True, method="ordinal").over("date"))
             .filter(pl.col("rk") <= 1000).select("date", "ticker"))
    names = uni["ticker"].unique().to_list()
    if p.exists():
        pw = pl.read_parquet(p)
    else:
        pw = P.build_panel(cfg, prices=px.filter(pl.col("ticker").is_in(names)), min_dollar_vol=0)
        pw = pw.join(uni, on=["date", "ticker"], how="semi")
        pw.write_parquet(p)
    spyq = (pl.scan_parquet(B / "ohlcv_all.parquet").filter(pl.col("ticker").is_in(["SPY", "QQQ", "RSP"]))
              .select("date", "ticker", "open", "high", "low", "close", "adj_close", "volume").collect())
    allpx = pl.concat([px.filter(pl.col("ticker").is_in(names)), spyq]).unique(subset=["date", "ticker"])
    return pw, allpx, names


def pit_protocol(names: list[str]) -> dict:
    pw, allpx, _ = pit_panel()
    hist = P.load_panel(cfg)
    cal = sorted(pw["date"].unique().to_list())
    cuts = cal[3::63]
    scores = {}
    for name in names:
        p = OUT / f"pit_{name}.parquet"
        if p.exists():
            scores[name] = pl.read_parquet(p); log(f"PIT {name}: cached"); continue
        spec = VARIANTS[name]
        t0 = time.time()
        hf, rcols = M.prepare(hist, spec.features)
        wf, _ = M.prepare(pw, spec.features)
        hd = hf.select("date", "didx").unique().sort("didx")
        parts = []
        for i, cut in enumerate(cuts):
            nxt = cuts[i + 1] if i + 1 < len(cuts) else date(2100, 1, 1)
            ci = int(hd.filter(pl.col("date") <= cut)["didx"][-1])
            blk = wf.filter((pl.col("date") >= cut) & (pl.col("date") < nxt))
            X = blk.select(rcols).to_numpy().astype(np.float32)
            cols = {"date": blk["date"], "ticker": blk["ticker"]}
            for h in H:
                tr = M.training_rows(hf, h, ci, spec)
                Xt, yt = M._xy(tr, rcols, h, spec.label)
                m = M.AlphaGBM(spec, h, rcols).fit(Xt, yt, M.sample_weights(tr, ci, spec), qid=tr["didx"].to_numpy())
                cols[f"s{h}"] = m.predict(X)
            parts.append(pl.DataFrame(cols))
        s = pl.concat(parts).unpivot(index=["date", "ticker"], on=[f"s{h}" for h in H], variable_name="horizon", value_name="score")
        s = s.with_columns(pl.col("horizon").str.slice(1).cast(pl.Int32))
        s.write_parquet(p)
        scores[name] = s
        log(f"PIT {name}: {s.height:,} scores in {(time.time() - t0) / 60:.1f} min")
    for bname, parts in BLENDS.items():
        if all(x in scores for x in parts):
            scores[bname] = blend([scores[x] for x in parts])
    labels = pw.select("date", "ticker", *[f"fwd_{h}" for h in H])
    ics = {k: ic_table(v, labels) for k, v in scores.items()}
    res = {"ic": {k: ic_summary(v, ics.get("base") if k != "base" else None) for k, v in ics.items()}}
    pdata = build_panel(P.sanitize_prices(allpx))
    sector_of = dict(pw.select("ticker", "sector").unique(subset=["ticker"]).iter_rows())
    regime = {d: market_regime_on(v) for d, v in pw.group_by("date").agg(pl.col("mkt_trend_200").first()).iter_rows()}
    gate = pw.select("date", "ticker", "adj_close", elig=(pl.col("close") >= 5) & (pl.col("vol_63") / 252 ** 0.5 <= 0.07)
                     & (pl.col("log_dv_21") >= np.log1p(20e6)))
    books = {}
    for k, v in scores.items():
        r = book_run(pdata, zcomp(v), gate, sector_of, regime, cuts[0], k)
        books[k] = r.returns()
    dates = r.daily["date"].to_list()
    tick = list(pdata.tickers); di = pdata.date_index()
    for t in ("SPY", "QQQ", "RSP"):
        if t in tick:
            books[f"passive: {t}"] = np.array([pdata.ret[di[d], tick.index(t)] for d in dates])
    res["book"] = {k: book_stats(v) for k, v in books.items()}
    res["book_excess_vs_base"] = {k: excess_ci(v, books["base"]) for k, v in books.items() if k != "base" and not k.startswith("passive")}
    res["window"] = [str(dates[0]), str(dates[-1])]
    return res


def main():
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else list(VARIANTS)[:7]
    phases = sys.argv[2].split(",") if len(sys.argv) > 2 else ["pit", "long"]
    tag = sys.argv[3] if len(sys.argv) > 3 else "results"
    t0 = time.time()
    results = {"variants": {k: asdict(v) | {"stride": None} for k, v in VARIANTS.items() if k in names}}
    if "pit" in phases:
        log("PIT protocol …")
        results["pit"] = pit_protocol(names)
        (OUT / f"{tag}.json").write_text(json.dumps(results, indent=1, default=str))
    if "long" in phases:
        log("LONG protocol …")
        results["long"] = long_protocol(names)
        (OUT / f"{tag}.json").write_text(json.dumps(results, indent=1, default=str))
    log(f"done in {(time.time() - t0) / 60:.0f} min → {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
