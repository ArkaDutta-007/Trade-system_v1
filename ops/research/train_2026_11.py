#!/usr/bin/env python3
"""New training ideas, Kronos and language-model inputs on the honest panel (2026-11-13, RESEARCH_2026-11 §13).

Arka: "try Kronos; use more innovative training, any kind of LLM inclusion or post-training, and other ways".
Panel: the challenger's point-in-time panel on corrected data (failures included, 2024-11 → last close).
Base = the challenger recipe (point-in-time rows + top-500 training, XGBoost, refit every 63 sessions).
Everything below was fixed before the first run.

Training (stage `walk`)
  resid      factor-neutral target: each day's forward return is first regressed on beta, 63-day volatility,
             size, 12-1 momentum, 21-day return and sector; the model is trained on the residual's rank. §10
             found four-fifths of the year-1 signal was exposure to those; this trains on what is left.
  riskadj    risk-adjusted target: rank of forward return ÷ 63-day volatility.
  sae        supervised autoencoder + MLP, the Jane Street 2021 winning design: input noise, an encoder whose
             code feeds the classifier, reconstruction and auxiliary losses; five seeds, 5 epochs.
  tail       train for the job the book does: the index-minus book only needs to know which names will be in
             the bottom 30%, so the target is 1 unless the stock lands in that day's bottom 30% (else 0).
  dropout    a second model for "about to drop out": probability that a name leaves the point-in-time top-1000
             within 63 sessions, trained on point-in-time rows only (walk-forward, purged 68 sessions), with two
             liquidity-trend inputs added. Used two ways: base−dropout (score minus its z-score) and a gate
             that bars the riskiest 20% from the top-20 book.
Kronos and language models (stage `extra`, scores made by kronos_2026_11.py and newsprobe_2026_11.py)
  kronos     zero-shot 21-session forecasts at month-ends; alone and as base+kronos (mean of z-scores).
  news       a probe trained on language-model embeddings of headlines; alone and as base+½·news.

Verdict per variant: rank IC by year and over the last 12 months; index-minus (top-300 cap-weighted minus the
worst 30%, 4 tranches) against its control; the production top-20 book inside the top-300 (4 tranches).
Adoption rule, as in §10: IC no lower than the base in either year AND a higher index-minus excess.
Results → reports/alpha/lab_2026_11/train.json, extra.json
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
import panel_2026_11 as PP  # noqa: E402
import tilts_2026_11 as TT  # noqa: E402

LAB, OUT, P, M = PP.LAB, PP.OUT, PP.P, PP.M
from trading_system.alpha import challenger as C  # noqa: E402
from trading_system.alpha.portfolio import market_regime_on  # noqa: E402
from trading_system.research.wfbacktest import build_panel  # noqa: E402

H = LAB.H
OFFSETS = (0, 5, 10, 15)
RISK = ("beta_63", "vol_63", "log_mcap", "mom_12_1", "ret_21")
Y1_END = PP.Y1_END


# ── data ──────────────────────────────────────────────────────────────────────
def load():
    cfg = LAB.cfg
    pw = C.load_panel(cfg)
    px, _ = C.pit_prices(cfg)
    spy = (pl.scan_parquet(cfg.path("data_bronze") / "massive" / "ohlcv_all.parquet").filter(pl.col("ticker") == "SPY")
             .select("date", "ticker", "open", "high", "low", "close", "adj_close", "volume").collect())
    return pw, pl.concat([px, spy]).unique(subset=["date", "ticker"])


def add_labels(df: pl.DataFrame) -> pl.DataFrame:
    """``yv_h`` (risk-adjusted) and ``yn_h`` (factor-neutral) gaussian-rank labels next to the panel's ``y_h``."""
    from scipy.stats import rankdata
    out = df
    for h in H:
        ra = pl.col(f"fwd_{h}") / pl.col("vol_63")
        out = out.with_columns(((ra.rank("average").over("date") - 0.5) / ra.count().over("date")).alias(f"_u{h}"))
    for h in H:
        out = out.with_columns(pl.Series(f"yv_{h}", P._norm_ppf(out[f"_u{h}"].to_numpy()), dtype=pl.Float32).fill_nan(None)).drop(f"_u{h}")
    ex = out.select("date", "ticker", "sector", *[PP.z(c).alias("x_" + c) for c in RISK], *[f"fwd_{h}" for h in H])
    parts = []
    for g in ex.partition_by("date"):
        sec = g["sector"].to_numpy()
        X = np.column_stack([g.select(["x_" + c for c in RISK]).to_numpy().astype(np.float64)] + [(sec == u).astype(np.float64) for u in np.unique(sec)])
        cols = {"date": g["date"], "ticker": g["ticker"]}
        for h in H:
            y = g[f"fwd_{h}"].to_numpy().astype(np.float64)
            ok, r = np.isfinite(y), np.full(len(y), np.nan)
            if ok.sum() >= 50:
                res = y[ok] - X[ok] @ np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
                r[ok] = P._norm_ppf((rankdata(res) - 0.5) / ok.sum())
            cols[f"yn_{h}"] = r
        parts.append(pl.DataFrame(cols))
    lab = pl.concat(parts).with_columns([pl.col(f"yn_{h}").cast(pl.Float32).fill_nan(None) for h in H])
    cut = float(P._norm_ppf(np.array([0.30]))[0])                      # y_h is a gaussian rank: bottom 30% ⇔ y < −0.524
    out = out.with_columns([pl.when(pl.col(f"y_{h}").is_not_null()).then((pl.col(f"y_{h}") > cut).cast(pl.Float32)).alias(f"yb_{h}") for h in H])
    return out.join(lab, on=["date", "ticker"], how="left")


# ── models ────────────────────────────────────────────────────────────────────
class SAENet(nn.Module):
    def __init__(self, d: int, code: int = 64, hidden=(128, 64, 32), p: float = 0.2, noise: float = 0.05):
        super().__init__()
        self.noise = noise
        self.enc = nn.Sequential(nn.Linear(d, code), nn.BatchNorm1d(code), nn.SiLU())
        self.dec = nn.Sequential(nn.Dropout(p), nn.Linear(code, d))
        self.aux = nn.Sequential(nn.Linear(d, 32), nn.BatchNorm1d(32), nn.SiLU(), nn.Dropout(p), nn.Linear(32, 1))
        ly, q = [], d + code
        for k in hidden:
            ly += [nn.Linear(q, k), nn.BatchNorm1d(k), nn.SiLU(), nn.Dropout(p)]; q = k
        self.head = nn.Sequential(*ly, nn.Linear(q, 1))

    def forward(self, x):
        xn = x + self.noise * torch.randn_like(x) if self.training else x
        e = self.enc(xn)
        rec = self.dec(e)
        return self.head(torch.cat([x, e], 1)).squeeze(-1), rec, self.aux(rec).squeeze(-1)


class SAEEns:
    """Five seeds of the supervised autoencoder; loss = prediction + reconstruction + auxiliary prediction."""

    def __init__(self, seeds=(0, 1, 2, 3, 4), epochs=5, bs=4096, lr=1e-3):
        self.seeds, self.epochs, self.bs, self.lr = seeds, epochs, bs, lr
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _x(X):
        return torch.from_numpy(np.nan_to_num(X - 0.5, nan=0.0).astype(np.float32))

    def fit(self, X, y, w=None):
        Xt = self._x(X).to(self.dev)
        yt = torch.from_numpy(((y - y.mean()) / (y.std() + 1e-9)).astype(np.float32)).to(self.dev)
        wt = torch.from_numpy((w / w.mean()).astype(np.float32)).to(self.dev) if w is not None else torch.ones_like(yt)
        self.nets, n = [], len(yt)
        for s in self.seeds:
            torch.manual_seed(s)
            net = SAENet(Xt.shape[1]).to(self.dev)
            opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=1e-5)
            for _ in range(self.epochs):
                net.train()
                perm = torch.randperm(n, device=self.dev)
                for i in range(0, n - 1, self.bs):
                    b = perm[i:i + self.bs]
                    if len(b) < 2:
                        continue
                    out, rec, aux = net(Xt[b])
                    loss = (wt[b] * ((out - yt[b]) ** 2 + (aux - yt[b]) ** 2)).mean() + ((rec - Xt[b]) ** 2).mean()
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            net.eval()
            self.nets.append(net)
        return self

    @torch.no_grad()
    def predict(self, X):
        Xt, out = self._x(X), np.zeros(len(X))
        for net in self.nets:
            out += torch.cat([net(c.to(self.dev))[0].cpu() for c in Xt.split(65536)]).numpy().astype(np.float64)
        return out / len(self.nets)


WALKS = {"base": ("y", "xgb"), "resid": ("yn", "xgb"), "riskadj": ("yv", "xgb"), "sae": ("y", "sae"), "tail": ("yb", "xgb")}


def walk(name: str, pw: pl.DataFrame, hist: pl.DataFrame) -> None:
    path = OUT / f"wf2_{name}.parquet"
    if path.exists():
        return
    label, engine = WALKS[name]
    t0 = time.time()
    sp = M.TrainSpec(horizons=H, train_top=C.TRAIN_TOP, label=label)
    cal = sorted(pw["date"].unique().to_list())
    cuts = cal[3::63]
    hf, rcols = M.prepare(hist, sp.features)
    wf, _ = M.prepare(pw, sp.features)
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
            Xt, yt, wt, _ = C.training_set(hf, wf, rcols, h, ci, wi, cal[0], sp)
            m = (SAEEns() if engine == "sae" else M.AlphaGBM(sp, h, rcols)).fit(Xt, yt, wt)
            cols[f"s{h}"] = np.asarray(m.predict(X), dtype=np.float64)
        parts.append(pl.DataFrame(cols))
    s = pl.concat(parts).unpivot(index=["date", "ticker"], on=[f"s{h}" for h in H], variable_name="horizon", value_name="score")
    s.with_columns(pl.col("horizon").str.slice(1).cast(pl.Int32)).write_parquet(path)
    LAB.log(f"{name}: done in {(time.time() - t0) / 60:.1f} min")


def dropout(pw: pl.DataFrame) -> dict:
    """Walk-forward P(leaves the point-in-time universe within 63 sessions) → wf2_dropout.parquet + AUC by fold."""
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score
    path = OUT / "wf2_dropout.parquet"
    cal = sorted(pw["date"].unique().to_list())
    pos = {d: i for i, d in enumerate(cal)}
    mem = set(pw.select("date", "ticker").iter_rows())
    gone = [None if pos[d] + 63 >= len(cal) else float((cal[pos[d] + 63], t) not in mem) for d, t in pw.select("date", "ticker").iter_rows()]
    f = pw.with_columns(gone=pl.Series(gone, dtype=pl.Float32)).sort(["ticker", "date"]).with_columns(
        liq_rank=pl.col("log_dv_21").rank().over("date") / pl.col("log_dv_21").count().over("date"))
    f = f.with_columns(liq_rank_chg=pl.col("liq_rank") - pl.col("liq_rank").shift(63).over("ticker"),
                       dv_trend=pl.col("log_dv_21") - pl.col("log_dv_21").rolling_mean(126, min_samples=40).over("ticker"))
    sp = M.TrainSpec(horizons=H)
    wf, rcols = M.prepare(f, sp.features)
    extra = ["liq_rank_chg", "dv_trend"]
    wf = wf.with_columns([((pl.col(c).rank().over("date") - 0.5) / pl.col(c).count().over("date")).cast(pl.Float32).alias(c + "_r") for c in extra])
    cols = rcols + [c + "_r" for c in extra]
    cuts = cal[3::63]
    wd = wf.select("date", "didx").unique().sort("didx")
    parts, aucs = [], {}
    for i, cut in enumerate(cuts):
        nxt = cuts[i + 1] if i + 1 < len(cuts) else date(2100, 1, 1)
        wi = int(wd.filter(pl.col("date") <= cut)["didx"][-1])
        last = wi - 63 - 5
        tr = wf.filter((pl.col("didx") <= last) & pl.col("gone").is_not_null() & ((pl.col("didx") % 4) == (last % 4)))
        blk = wf.filter((pl.col("date") >= cut) & (pl.col("date") < nxt))
        if tr.height < 10_000:
            parts.append(blk.select("date", "ticker", p_gone=pl.lit(None, dtype=pl.Float64)))
            continue
        dm = xgb.DMatrix(tr.select(cols).to_numpy().astype(np.float32), label=tr["gone"].to_numpy())
        bst = xgb.train({"objective": "binary:logistic", "max_depth": 4, "eta": 0.05, "subsample": 0.7, "colsample_bytree": 0.6,
                         "min_child_weight": 50, "lambda": 10.0, "tree_method": "hist", "device": "cuda", "max_bin": 128, "verbosity": 0, "seed": 0}, dm, 300)
        p = bst.predict(xgb.DMatrix(blk.select(cols).to_numpy().astype(np.float32)))
        parts.append(blk.select("date", "ticker").with_columns(p_gone=pl.Series(p.astype(np.float64))))
        y = blk["gone"].to_numpy()
        ok = np.isfinite(y)
        if ok.sum() > 1000 and 0 < y[ok].mean() < 1:
            aucs[str(cut)] = {"auc": float(roc_auc_score(y[ok], p[ok])), "base_rate": float(y[ok].mean()), "train_rows": tr.height}
    pl.concat(parts).write_parquet(path)
    return aucs


# ── verdicts ──────────────────────────────────────────────────────────────────
def ic_periods(comp: pl.DataFrame, pw: pl.DataFrame, cal: list) -> dict:
    j = comp.join(pw.select("date", "ticker", "fwd_21", "fwd_63"), on=["date", "ticker"])
    ly, out = cal[max(0, len(cal) - 252)], {}
    for h in H:
        g = (j.drop_nulls([f"fwd_{h}", "comp"]).group_by("date").agg(ic=pl.corr("comp", f"fwd_{h}", method="spearman"), n=pl.len())
              .filter(pl.col("n") >= 30).sort("date"))
        x = g["ic"].to_numpy()
        xs = x[::h]
        m = lambda q: float(g.filter(q)["ic"].mean()) if g.filter(q).height else None
        out[h] = {"all": float(np.nanmean(x)) if len(x) else None, "y1": m(pl.col("date") <= Y1_END), "y2": m(pl.col("date") > Y1_END),
                  "last12": m(pl.col("date") >= ly), "dates": len(x),
                  "t": float(np.nanmean(xs) / (np.nanstd(xs, ddof=1) + 1e-12) * np.sqrt(len(xs))) if len(xs) > 3 else None}
    return out


def verdicts(comps: dict[str, pl.DataFrame], pw: pl.DataFrame, allpx: pl.DataFrame, gates: dict | None = None,
             offsets=OFFSETS, top20: bool = True) -> dict:
    """IC, index-minus vs control and the top-20 book for each composite (``date, ticker, comp``)."""
    cal = sorted(pw["date"].unique().to_list())
    res = {"ic": {k: ic_periods(v, pw, cal) for k, v in comps.items()}, "index_minus": {}, "top20": {}}
    r1 = allpx.sort(["ticker", "date"]).with_columns(r=pl.col("adj_close") / pl.col("adj_close").shift(1).over("ticker") - 1)
    R = {(d, t): r for d, t, r in r1.select("date", "ticker", "r").iter_rows() if r is not None}
    start = max(c["date"].min() for c in comps.values())
    days = [d for d in cal if d > start]
    rets_of = {"days": days, "R": R}
    y1 = np.array([d <= Y1_END for d in days])
    l12 = np.arange(len(days)) >= len(days) - 252
    tot = lambda r, m: float(np.prod(1 + r[m]) - 1)
    cagr = lambda r: float(np.prod(1 + r) ** (252 / len(r)) - 1)
    any_comp = next(iter(comps.values())).filter(pl.col("date") >= start)
    ctrl = []
    for off in offsets:
        first, prev, by = TT.month_book(pw, any_comp, off)
        ctrl.append(TT.run(rets_of, first, prev, by, "all", "cap")[0])
    ct = np.mean(ctrl, axis=0)
    res["index_minus"]["control"] = {**LAB.book_stats(ct), "y1": tot(ct, y1), "y2": tot(ct, ~y1), "last12": tot(ct, l12)}
    spy = np.array([R.get((d, "SPY"), 0.0) for d in days])
    res["index_minus"]["SPY"] = {**LAB.book_stats(spy), "y1": tot(spy, y1), "y2": tot(spy, ~y1), "last12": tot(spy, l12)}
    res["window"] = [str(days[0]), str(days[-1])]
    for k, c in comps.items():
        ser = []
        for off in offsets:
            first, prev, by = TT.month_book(pw, c.filter(pl.col("date") >= start), off)
            ser.append(TT.run(rets_of, first, prev, by, "drop30", "cap")[0])
        tr = np.mean(ser, axis=0)
        res["index_minus"][k] = {**LAB.book_stats(tr), "by_day": [cagr(r) for r in ser], "y1": tot(tr, y1), "y2": tot(tr, ~y1), "last12": tot(tr, l12),
                                 "vs_control": LAB.excess_ci(tr, ct), "ex_y1": tot(tr, y1) - tot(ct, y1), "ex_y2": tot(tr, ~y1) - tot(ct, ~y1),
                                 "ex_last12": tot(tr, l12) - tot(ct, l12)}
    if top20:
        pdata = build_panel(P.sanitize_prices(allpx))
        sector_of = dict(pw.select("ticker", "sector").unique(subset=["ticker"]).iter_rows())
        regime = {d: market_regime_on(v) for d, v in pw.group_by("date").agg(pl.col("mkt_trend_200").first()).iter_rows()}
        liq = pw.with_columns(lr=pl.col("log_dv_21").rank(descending=True).over("date"))
        gate0 = liq.select("date", "ticker", "adj_close", elig=(pl.col("close") >= 5) & (pl.col("vol_63") / 252 ** 0.5 <= 0.07) & (pl.col("lr") <= 300))
        runs = [(k, c, gate0) for k, c in comps.items()] + [(k, comps[src], g) for k, (src, g) in (gates or {}).items()]
        for k, c, g in runs:
            ser = [LAB.book_run(pdata, c, g, sector_of, regime, cal[3 + off], k).returns() for off in offsets]
            n = min(len(r) for r in ser)
            tr = np.mean([r[-n:] for r in ser], axis=0)
            m = min(252, n)
            res["top20"][k] = {**LAB.book_stats(tr), "by_day": [cagr(r) for r in ser], "last12": float(np.prod(1 + tr[-m:]) - 1)}
        res["_gate0"] = None
    return res


def show(res: dict) -> None:
    f = lambda x: "   n/a" if x is None else f"{x:+.3f}"
    for k, v in res["ic"].items():
        print(f"IC {k:<16} 21d y1 {f(v[21]['y1'])} y2 {f(v[21]['y2'])} last12 {f(v[21]['last12'])} | 63d y1 {f(v[63]['y1'])} y2 {f(v[63]['y2'])} last12 {f(v[63]['last12'])} | dates {v[21]['dates']}")
    for k in ("control", "SPY"):
        v = res["index_minus"][k]
        print(f"{k:<19} {v['cagr']:+.1%}/yr y1 {v['y1']:+.1%} y2 {v['y2']:+.1%} last12 {v['last12']:+.1%} Sh {v['sharpe']:.2f} DD {v['maxdd']:+.1%}")
    for k, v in res["index_minus"].items():
        if k in ("control", "SPY"):
            continue
        e = v["vs_control"]
        print(f"idx-minus {k:<16} {v['cagr']:+.1%}/yr Sh {v['sharpe']:.2f} DD {v['maxdd']:+.1%} | vs control {e['excess']:+.1%} [{e['lo']:+.1%},{e['hi']:+.1%}] "
              f"y1 {v['ex_y1']:+.1%} y2 {v['ex_y2']:+.1%} last12 {v['ex_last12']:+.1%} | days {' '.join(f'{x:+.1%}' for x in v['by_day'])}")
    for k, v in res["top20"].items():
        print(f"top-20 {k:<19} {v['cagr']:+.1%}/yr Sh {v['sharpe']:.2f} DD {v['maxdd']:+.1%} last12 {v['last12']:+.1%} | days {' '.join(f'{x:+.1%}' for x in v['by_day'])}")


def zc(col: str = "comp") -> pl.Expr:
    return (pl.col(col) - pl.col(col).mean().over("date")) / (pl.col(col).std().over("date") + 1e-9)


def stage_walk():
    pw, _ = load()
    hist = add_labels(P.load_panel(LAB.cfg))
    pwl = add_labels(pw)
    LAB.log("labels ready")
    for n in WALKS:
        walk(n, pwl, hist)
    if not (OUT / "wf2_dropout.parquet").exists():
        aucs = dropout(pw)
        (OUT / "train_dropout_auc.json").write_text(json.dumps(aucs, indent=1))
        LAB.log(f"dropout AUC by fold: {[round(v['auc'], 3) for v in aucs.values()]}")


def stage_eval():
    pw, allpx = load()
    comps = {n: LAB.zcomp(pl.read_parquet(OUT / f"wf2_{n}.parquet")) for n in WALKS}
    dp = pl.read_parquet(OUT / "wf2_dropout.parquet")
    j = comps["base"].join(dp, on=["date", "ticker"], how="left")
    comps["base-dropout"] = j.select("date", "ticker", comp=zc() - zc("p_gone").fill_null(0.0))
    liq = pw.with_columns(lr=pl.col("log_dv_21").rank(descending=True).over("date")).join(dp, on=["date", "ticker"], how="left")
    liq = liq.with_columns(pr=pl.col("p_gone").rank().over("date") / pl.col("p_gone").count().over("date"))
    gate = liq.select("date", "ticker", "adj_close", elig=(pl.col("close") >= 5) & (pl.col("vol_63") / 252 ** 0.5 <= 0.07) & (pl.col("lr") <= 300)
                      & (pl.col("pr").is_null() | (pl.col("pr") <= 0.8)))
    res = verdicts(comps, pw, allpx, gates={"base+gate": ("base", gate)})
    # the drop-out probability on its own starts with its third fold, so it is scored by IC only: putting it
    # in `comps` would cut every book's window to its dates (as the first run of this stage did)
    res["ic"]["dropout only"] = ic_periods(dp.drop_nulls("p_gone").select("date", "ticker", comp=-pl.col("p_gone")), pw,
                                           sorted(pw["date"].unique().to_list()))
    res["dropout_auc"] = json.loads((OUT / "train_dropout_auc.json").read_text())
    (OUT / "train.json").write_text(json.dumps(res, indent=1, default=str))
    show(res)


def stage_extra():
    """Kronos and the news probe: alone and blended with the base, on the dates each one has."""
    pw, allpx = load()
    base = LAB.zcomp(pl.read_parquet(OUT / "wf2_base.parquet"))
    out = {}
    kp = OUT / "kronos_scores.parquet"
    if kp.exists():
        k = pl.read_parquet(kp).select("date", "ticker", comp="score")
        b = base.join(k.select("date", "ticker"), on=["date", "ticker"], how="semi")       # same dates and names
        bl = base.join(k.rename({"comp": "kr"}), on=["date", "ticker"]).select("date", "ticker", comp=0.5 * zc() + 0.5 * zc("kr"))
        comps = {"base (same dates)": b, "kronos": k, "base+kronos": bl}
        dp, nw = OUT / "wf2_dropout.parquet", OUT / "news_scores.parquet"
        if dp.exists() and nw.exists():            # every signal we have, equal weight (news at half): one pre-set blend
            a = (base.join(k.rename({"comp": "kr"}), on=["date", "ticker"]).join(pl.read_parquet(dp), on=["date", "ticker"], how="left")
                     .join(pl.read_parquet(nw).select("date", "ticker", nw="score"), on=["date", "ticker"], how="left"))
            comps["all signals"] = a.select("date", "ticker", comp=zc() + zc("kr") - zc("p_gone").fill_null(0.0) + 0.5 * zc("nw").fill_null(0.0))
        out["kronos"] = verdicts(comps, pw, allpx, offsets=(0,), top20=False)
        cal = sorted(pw["date"].unique().to_list())
        oos = {n: ic_periods(c.filter(pl.col("date") >= date(2025, 9, 1)), pw, cal) for n, c in comps.items()}
        out["kronos"]["ic_from_2025_09"] = {n: {h: v[h]["all"] for h in H} | {"dates": v[21]["dates"]} for n, v in oos.items()}
        print("— Kronos (month-end dates; index-minus on start day 0 only) —"); show(out["kronos"])
        print("IC from 2025-09 (after its weights were published):", json.dumps(out["kronos"]["ic_from_2025_09"], default=lambda x: round(x, 4)))
    np_ = OUT / "news_scores.parquet"
    if np_.exists():
        n = pl.read_parquet(np_).select("date", "ticker", comp="score")
        j = base.join(n.rename({"comp": "nw"}), on=["date", "ticker"], how="left")
        comps = {"base": base, "base+news": j.select("date", "ticker", comp=zc() + 0.5 * zc("nw").fill_null(0.0)), "news only": n}
        out["news"] = verdicts(comps, pw, allpx)
        print("— News probe —"); show(out["news"])
    (OUT / "extra.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    {"walk": stage_walk, "eval": stage_eval, "extra": stage_extra}[sys.argv[1] if len(sys.argv) > 1 else "walk"]()
