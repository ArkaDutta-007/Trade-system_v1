#!/usr/bin/env python3
"""Stock-level tests on the complete honest panel (November 2026 research pass).

The October model comparisons (RESEARCH_2026-10 §1-§5) ran on the panel with missing short and news rows for
failed names, so their verdicts did not carry over. Everything is re-run here, walk-forward with 63-session
refits, on the refreshed point-in-time panel (lab_2026_11). All settings below were fixed before the first run.

  training   S = today's survivors only (production) · P = survivors before the panel starts, then the
             point-in-time rows, failures included (RESEARCH §13)
  engines    xgb (production settings) · lgbm · ridge · mlp = five-seed three-layer network (64-32-16, SiLU,
             dropout .1, Adam 1e-3, 5 epochs, batch 4096, same labels and sample weights; Gu-Kelly-Xiu style)
  labels     y (production) · ys sector-neutral · yr beta-residual     debias = train on that day's top-500
  no-train   lit = equal-weight z-scores of published anomalies with published signs
             lit-osap = same, weighted by each anomaly's trailing 10-year large-cap Sharpe (Open Source Asset
             Pricing, failures included, weights known at the time; a negative trailing Sharpe gets weight 0)
  books      production book, in the production universe and in that day's 300 most liquid names
Results → reports/alpha/lab_2026_11/panel.json (`eval`), wf_*.parquet (`walk`)
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
import lab_2026_11 as L  # noqa: E402

LAB = L.LAB
OUT = L.OUT
from trading_system.alpha import model as M  # noqa: E402
from trading_system.alpha import panel as P  # noqa: E402
from trading_system.alpha.portfolio import market_regime_on  # noqa: E402
from trading_system.research.wfbacktest import build_panel  # noqa: E402

H = LAB.H
Y1_END = date(2025, 10, 31)
# (our feature, published sign, Open Source Asset Pricing signal)
LIT = [("mom_12_1", 1, "Mom12m"), ("ret_21", -1, "STreversal"), ("vol_63", -1, "RealizedVol"), ("idio_vol_63", -1, "IdioVol3F"),
       ("max_ret_21", -1, "MaxRet"), ("dist_52w_high", 1, "High52"), ("log_mcap", -1, "Size"), ("bm", 1, "BMdec"), ("ep_ttm", 1, "EP"),
       ("sp_ttm", 1, "SP"), ("cfp_ttm", 1, "CF"), ("gp_assets", 1, "GP"), ("roa_ttm", 1, "roaq"), ("asset_growth", -1, "AssetGrowth"),
       ("accruals", -1, "Accruals"), ("sue", 1, "EarningsSurprise"), ("ear_3d", 1, "AnnouncementReturn"), ("si_ratio", -1, "ShortInterest"),
       ("res_mom_12_1", 1, "ResidualMomentum"), ("ind_mom_126", 1, "IndMom"), ("log_price", -1, "Price")]


class MLPEns:
    def __init__(self, spec, h, feats, seeds=(0, 1, 2, 3, 4), hidden=(64, 32, 16), epochs=5, bs=4096, lr=1e-3):
        self.seeds, self.hidden, self.epochs, self.bs, self.lr = seeds, hidden, epochs, bs, lr
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"

    def _net(self, d):
        ly, p = [], d
        for k in self.hidden:
            ly += [nn.Linear(p, k), nn.SiLU(), nn.Dropout(.1)]; p = k
        return nn.Sequential(*ly, nn.Linear(p, 1)).to(self.dev)

    def _x(self, X):
        return torch.from_numpy(np.nan_to_num(X - .5, nan=0.0).astype(np.float32))

    def fit(self, X, y, w=None):
        Xt = self._x(X).to(self.dev)
        yt = torch.from_numpy(((y - y.mean()) / (y.std() + 1e-9)).astype(np.float32)).to(self.dev)
        wt = torch.from_numpy((w / w.mean()).astype(np.float32)).to(self.dev) if w is not None else torch.ones_like(yt)
        self.nets = []
        n = len(yt)
        for s in self.seeds:
            torch.manual_seed(s)
            net = self._net(Xt.shape[1])
            opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=1e-5)
            for _ in range(self.epochs):
                net.train()
                perm = torch.randperm(n, device=self.dev)
                for i in range(0, n, self.bs):
                    b = perm[i:i + self.bs]
                    loss = (wt[b] * (net(Xt[b]).squeeze(-1) - yt[b]) ** 2).mean()
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            net.eval()
            self.nets.append(net)
        return self

    @torch.no_grad()
    def predict(self, X):
        Xt = self._x(X)
        out = np.zeros(len(Xt))
        for net in self.nets:
            out += torch.cat([net(c.to(self.dev)).squeeze(-1).cpu() for c in Xt.split(65536)]).numpy().astype(np.float64)
        return out / len(self.nets)


def spec_of(engine="xgb", label="y", top=0):
    if engine in ("xgb", "mlp"):
        return M.TrainSpec(horizons=H, label=label, train_top=top)
    return M.TrainSpec(horizons=H, label=label, train_top=top, engine=engine, seeds=(0,) if engine == "ridge" else (0, 1))


VARIANTS = {  # name: (engine, label, top, point-in-time training)
    "S-xgb": ("xgb", "y", 0, False), "P-xgb": ("xgb", "y", 0, True), "S-mlp": ("mlp", "y", 0, False), "P-mlp": ("mlp", "y", 0, True),
    "P-lgbm": ("lgbm", "y", 0, True), "P-ridge": ("ridge", "y", 0, True), "S-ridge": ("ridge", "y", 0, False), "S-lgbm": ("lgbm", "y", 0, False),
    "P-xgb-debias": ("xgb", "y", 500, True), "P-xgb-sector": ("xgb", "ys", 0, True), "P-xgb-beta": ("xgb", "yr", 0, True),
    "S-xgb-debias": ("xgb", "y", 500, False), "S-xgb-sector": ("xgb", "ys", 0, False)}
BLENDS = {"P-xgb+mlp": ["P-xgb", "P-mlp"], "P-all4": ["P-xgb", "P-mlp", "P-lgbm", "P-ridge"], "S-all4": ["S-xgb", "S-mlp", "S-lgbm", "S-ridge"]}


def walk(name, pw, hist):
    path = OUT / f"wf_{name}.parquet"
    if path.exists():
        return
    engine, label, top, aug = VARIANTS[name]
    spec = spec_of(engine, label, top)
    t0 = time.time()
    cal = sorted(pw["date"].unique().to_list())
    cuts, pit_start = cal[3::63], cal[0]
    hf, rcols = M.prepare(hist, spec.features)
    wf, _ = M.prepare(pw, spec.features)
    hd = hf.select("date", "didx").unique().sort("didx")
    wd = wf.select("date", "didx").unique().sort("didx")
    parts = []
    for i, cut in enumerate(cuts):
        nxt = cuts[i + 1] if i + 1 < len(cuts) else date(2100, 1, 1)
        ci = int(hd.filter(pl.col("date") <= cut)["didx"][-1])
        wi = int(wd.filter(pl.col("date") <= cut)["didx"][-1])
        blk = wf.filter((pl.col("date") >= cut) & (pl.col("date") < nxt))
        cols = {"date": blk["date"], "ticker": blk["ticker"]}
        X = blk.select(rcols).to_numpy().astype(np.float32)
        for h in H:
            tr = M.training_rows(hf, h, ci, spec)
            if aug:
                tr = tr.filter(pl.col("date") < pit_start)
            Xt, yt = M._xy(tr, rcols, h, label)
            wt = M.sample_weights(tr, ci, spec)
            if aug:
                tw = M.training_rows(wf, h, wi, spec)
                if tw.height:
                    X2, y2 = M._xy(tw, rcols, h, label)
                    Xt, yt = np.vstack([Xt, X2]), np.concatenate([yt, y2])
                    wt = np.concatenate([wt, M.sample_weights(tw, wi, spec)])
            m = (MLPEns(spec, h, rcols) if engine == "mlp" else M.AlphaGBM(spec, h, rcols)).fit(Xt, yt, wt)
            cols[f"s{h}"] = np.asarray(m.predict(X), dtype=np.float64)
        parts.append(pl.DataFrame(cols))
        LAB.log(f"  {name} fold {i + 1}/{len(cuts)} · {(time.time() - t0) / 60:.1f} min")
    s = pl.concat(parts).unpivot(index=["date", "ticker"], on=[f"s{h}" for h in H], variable_name="horizon", value_name="score")
    s.with_columns(pl.col("horizon").str.slice(1).cast(pl.Int32)).write_parquet(path)
    LAB.log(f"{name}: done in {(time.time() - t0) / 60:.1f} min")


def z(col):
    x = pl.when(pl.col(col).is_finite()).then(pl.col(col))        # a single ±inf would turn the whole day's z-scores into NaN
    return ((x - x.mean().over("date")) / (x.std().over("date") + 1e-9)).clip(-3, 3).fill_null(0.0)


def lit_scores(pw) -> dict[str, pl.DataFrame]:
    use = [x for x in LIT if x[0] in pw.columns]
    zs = pw.select("date", "ticker", *[(z(f) * s).cast(pl.Float64).alias(f) for f, s, _ in use])
    eq = zs.select("date", "ticker", comp=pl.mean_horizontal([f for f, _, _ in use]))
    import pandas as pd
    p = pd.read_parquet(L.EXT / "osap" / "Alt_LiqME.parquet", columns=["signalname", "port", "date", "ret"])
    ls = p[p.port == "LS"].pivot_table(index="date", columns="signalname", values="ret") / 100
    ls.index = pd.to_datetime(ls.index) + pd.offsets.MonthEnd(0)
    use2 = [x for x in use if x[2] in ls.columns]
    sh = (ls[[x[2] for x in use2]].rolling(120, min_periods=60).mean() / ls[[x[2] for x in use2]].rolling(120, min_periods=60).std()).clip(lower=0)
    dates = sorted(pw["date"].unique().to_list())
    rows = []
    for d in dates:
        t = pd.Timestamp(d) - pd.offsets.MonthEnd(1)          # last completed month
        w = sh.loc[:t].iloc[-1]
        rows.append({"date": d, **{f: float(w[o]) for f, _, o in use2}})
    wdf = pl.DataFrame(rows)
    fs = [f for f, _, _ in use2]
    j = zs.select("date", "ticker", *fs).join(wdf.rename({f: f"w_{f}" for f in fs}), on="date")
    osap = j.select("date", "ticker", comp=pl.sum_horizontal([pl.col(f) * pl.col(f"w_{f}") for f in fs]) / pl.sum_horizontal([f"w_{f}" for f in fs]))
    miss = [x for x in LIT if x not in use2]
    LAB.log(f"literature composite: {len(use)} signals, {len(use2)} with Open Source AP weights; unmatched {miss}")
    last = sh.loc[:"2025-12-31"].iloc[-1].sort_values(ascending=False)
    (OUT / "lit_weights_2025-12.json").write_text(last.round(3).to_json(indent=0))
    return {"lit": eq, "lit-osap": osap}


def ic_year(sc, labels):
    t = LAB.ic_table(sc, labels)
    out = {"all": LAB.ic_summary(t)}
    for k, lo, hi in (("y1", date(2000, 1, 1), Y1_END), ("y2", date(2025, 11, 1), date(2100, 1, 1))):
        out[k] = LAB.ic_summary(t.filter((pl.col("date") >= lo) & (pl.col("date") <= hi)))
    return out


def evaluate():
    pw, allpx, _ = LAB.pit_panel()
    pdata = build_panel(P.sanitize_prices(allpx))
    cal = sorted(pw["date"].unique().to_list())
    oos = cal[3::63][0]
    sector_of = dict(pw.select("ticker", "sector").unique(subset=["ticker"]).iter_rows())
    regime = {d: market_regime_on(v) for d, v in pw.group_by("date").agg(pl.col("mkt_trend_200").first()).iter_rows()}
    base_gate = (pl.col("close") >= 5) & (pl.col("vol_63") / 252 ** 0.5 <= 0.07)
    liq = pw.with_columns(lr=pl.col("log_dv_21").rank(descending=True).over("date"))
    gates = {"prod": liq.select("date", "ticker", "adj_close", elig=base_gate & (pl.col("log_dv_21") >= np.log1p(20e6))),
             "top300": liq.select("date", "ticker", "adj_close", elig=base_gate & (pl.col("lr") <= 300))}
    labels = pw.select("date", "ticker", "fwd_21", "fwd_63")
    raw = {n: pl.read_parquet(OUT / f"wf_{n}.parquet") for n in VARIANTS if (OUT / f"wf_{n}.parquet").exists()}
    for b, parts in BLENDS.items():
        if all(x in raw for x in parts):
            raw[b] = LAB.blend([raw[x] for x in parts])
    comps = {k: LAB.zcomp(v) for k, v in raw.items()}
    lit = lit_scores(pw)
    comps.update(lit)
    res = {"ic": {k: ic_year(v, labels) for k, v in raw.items()}, "books": {}, "excess": {}}
    for k, v in lit.items():
        s = pl.concat([v.select("date", "ticker", score="comp").with_columns(horizon=pl.lit(h, dtype=pl.Int32)) for h in H])
        res["ic"][k] = ic_year(s, labels)
    rets = {}
    for g, gate in gates.items():
        for c, comp in comps.items():
            r = LAB.book_run(pdata, comp, gate, sector_of, regime, oos, c).returns()
            rets[(g, c)] = r
            res["books"][f"{c}·{g}"] = LAB.book_stats(r)
    base = rets[("prod", "S-xgb")]
    res["excess"] = {f"{c}·{g}": LAB.excess_ci(r, base) for (g, c), r in rets.items() if (g, c) != ("prod", "S-xgb")}
    tick, di = list(pdata.tickers), pdata.date_index()
    dates = [d for d in pdata.dates if d >= oos][:len(base)]
    res["window"] = [str(dates[0]), str(dates[-1])]
    for t in ("SPY", "QQQ", "RSP"):
        if t in tick:
            res["books"][f"passive:{t}"] = LAB.book_stats(np.array([pdata.ret[di[d], tick.index(t)] for d in dates]))
    from trading_system.research.stats import deflated_sharpe, pbo_cscv
    keys = list(rets)
    n = min(len(rets[k]) for k in keys)
    mat = np.column_stack([rets[k][-n:] for k in keys])
    res["pbo_all_books"] = pbo_cscv(mat, n_blocks=12)
    srs = mat.mean(0) / (mat.std(0) + 1e-12)            # per period, as deflated_sharpe expects
    best = keys[int(np.argmax(srs))]
    res["deflated_best"] = {"book": f"{best[1]}·{best[0]}", **deflated_sharpe(rets[best][-n:], srs)}
    (OUT / "panel.json").write_text(json.dumps(res, indent=1, default=str))
    np.savez(OUT / "panel_returns.npz", dates=np.array([str(d) for d in dates]), **{f"{c}|{g}": r for (g, c), r in rets.items()})
    for k, v in sorted(res["books"].items(), key=lambda kv: -kv[1]["cagr"]):
        e = res["excess"].get(k)
        LAB.log(f"{k:<26} CAGR {v['cagr']:+6.1%} Sharpe {v['sharpe']:5.2f} DD {v['maxdd']:+6.1%}"
                + (f"  vs prod {e['excess']:+6.1%} [{e['lo']:+.1%},{e['hi']:+.1%}]" if e else ""))
    for k, v in res["ic"].items():
        a, y1, y2 = v["all"], v["y1"], v["y2"]
        LAB.log(f"IC {k:<16} 21d {a[21]['ic']:+.4f} (t {a[21]['t']:+.2f}) y1 {y1[21]['ic']:+.4f} y2 {y2[21]['ic']:+.4f} | 63d {a[63]['ic']:+.4f} y1 {y1[63]['ic']:+.4f} y2 {y2[63]['ic']:+.4f}")


def phase():
    """Rebalance-timing luck: the same book started 0, 3, … 18 sessions later (the book trades every 21)."""
    pw, allpx, _ = LAB.pit_panel()
    pdata = build_panel(P.sanitize_prices(allpx))
    cal = sorted(pw["date"].unique().to_list())
    sector_of = dict(pw.select("ticker", "sector").unique(subset=["ticker"]).iter_rows())
    regime = {d: market_regime_on(v) for d, v in pw.group_by("date").agg(pl.col("mkt_trend_200").first()).iter_rows()}
    base_gate = (pl.col("close") >= 5) & (pl.col("vol_63") / 252 ** 0.5 <= 0.07)
    liq = pw.with_columns(lr=pl.col("log_dv_21").rank(descending=True).over("date"))
    gates = {"prod": liq.select("date", "ticker", "adj_close", elig=base_gate & (pl.col("log_dv_21") >= np.log1p(20e6))),
             "top300": liq.select("date", "ticker", "adj_close", elig=base_gate & (pl.col("lr") <= 300))}
    lit = lit_scores(pw)
    comps = {"S-xgb": LAB.zcomp(pl.read_parquet(OUT / "wf_S-xgb.parquet")), "P-xgb": LAB.zcomp(pl.read_parquet(OUT / "wf_P-xgb.parquet")),
             "P-xgb-debias": LAB.zcomp(pl.read_parquet(OUT / "wf_P-xgb-debias.parquet")), "lit": lit["lit"]}
    res = {}
    for (m, g) in (("S-xgb", "prod"), ("S-xgb", "top300"), ("P-xgb", "prod"), ("P-xgb", "top300"), ("P-xgb-debias", "prod"), ("P-xgb-debias", "top300"), ("lit", "top300"), ("lit", "prod")):
        cg = []
        for k in range(0, 21, 3):
            r = LAB.book_run(pdata, comps[m], gates[g], sector_of, regime, cal[3 + k], m).returns()
            cg.append(float(np.prod(1 + r) ** (252 / len(r)) - 1))
        res[f"{m}·{g}"] = {"cagr_by_start": cg, "median": float(np.median(cg)), "min": min(cg), "max": max(cg)}
        LAB.log(f"{m}·{g}: median {np.median(cg):+.1%}  range {min(cg):+.1%} … {max(cg):+.1%}")
    (OUT / "phase.json").write_text(json.dumps(res, indent=1))


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "walk"
    if cmd == "walk":
        pw, _, _ = LAB.pit_panel()
        hist = P.load_panel(LAB.cfg)
        LAB.log(f"hist {hist.height:,} rows; pit {pw.height:,} rows")
        for n in (sys.argv[2:] or VARIANTS):
            walk(n, pw, hist)
    elif cmd == "phase":
        phase()
    else:
        evaluate()


if __name__ == "__main__":
    main()
