"""Clean AI-era test: point-in-time top-1000 universe from WHOLE-MARKET bars (every name gets identical data),
models trained causally on the historical panel, scored on the PIT universe, same book/costs/overlays."""
import sys, time
from dataclasses import replace
from datetime import date, timedelta
import numpy as np, polars as pl
from trading_system.config import get_config
from trading_system.alpha import panel as P, model as M
from trading_system.alpha.backtest import ColumnAlpha, _weight_fn
from trading_system.alpha.portfolio import BookConfig, market_regime_on
from trading_system.alpha.regime import AI_THEME
from trading_system.research.runner import summarize_result
from trading_system.research.stats import bootstrap_ci, sharpe
from trading_system.research.wfbacktest import WalkForwardConfig, build_panel, run_walk_forward

cfg = get_config("configs/default.yaml")
B = cfg.path("data_bronze") / "massive"
TOP = 1000
t0 = time.time()
# 1. whole-market common stocks / ADRs, exchange-listed, incl. names that later delisted
tk = pl.read_parquet(B / "tickers.parquet", columns=["ticker", "type"])
ok_types = tk.filter(pl.col("type").is_in(["CS", "ADRC"]))["ticker"]
px = (pl.scan_parquet(B / "ohlcv_all.parquet")
        .filter(~pl.col("otc"), pl.col("ticker").is_in(ok_types.to_list()), pl.col("close") > 0, pl.col("adj_close") > 0)
        .select("date", "ticker", "open", "high", "low", "close", "adj_close", "volume").collect())
print(f"whole-market CS/ADR bars: {px.height:,} rows · {px['ticker'].n_unique():,} tickers · {px['date'].min()} → {px['date'].max()}")
# 2. point-in-time universe: top-1000 by trailing-63d median $volume among price ≥ $5, using data up to that day only
uni = (px.sort(["ticker", "date"])
         .with_columns(dv=pl.col("close") * pl.col("volume"))
         .with_columns(med63=pl.col("dv").rolling_median(63, min_samples=40).over("ticker"))
         .filter(pl.col("close") >= 5, pl.col("med63").is_not_null())
         .with_columns(rk=pl.col("med63").rank(descending=True, method="ordinal").over("date"))
         .filter(pl.col("rk") <= TOP).select("date", "ticker"))
first_day = uni["date"].min()
print(f"PIT universe from {first_day}: {uni['ticker'].n_unique():,} distinct names ever in the top-{TOP}")
biased = set(pl.read_parquet(cfg.path("data_gold") / "alpha_panel.parquet", columns=["ticker"]).unique()["ticker"].to_list())
names = set(uni["ticker"].unique().to_list())
print(f"  of which NOT in today's 1000-name research universe: {len(names - biased):,} (names that faded, got acquired or delisted)")
# 3. features for every PIT name from the SAME source (whole-market bars only — no deep history for anyone)
pw = P.build_panel(cfg, prices=px.filter(pl.col("ticker").is_in(list(names))), min_dollar_vol=0)
pw = pw.join(uni.with_columns(inu=pl.lit(True)), on=["date", "ticker"], how="left").filter(pl.col("inu")).drop("inu")
print(f"PIT panel: {pw.height:,} rows ({time.time()-t0:.0f}s)")
# 4. causal scores: refit every 63 sessions on the historical panel (labels matured before the cut), score the PIT cross-section
hist = P.load_panel(cfg)
FEATS = {"all features": tuple(P.MODEL_FEATURES), "regression objective": ("REG", tuple(P.MODEL_FEATURES)),
         "no size/liquidity": tuple(f for f in P.MODEL_FEATURES if f not in ("log_dv_21", "amihud_21", "log_mcap", "log_price", "spread_ar_21", "vol_ratio_5_63"))}
start_score = first_day + timedelta(days=5)
cal_dates = sorted(pw.filter(pl.col("date") >= start_score)["date"].unique().to_list())
cuts = cal_dates[::63]
hframe, _ = M.prepare(hist)
hdates = hframe.select("date", "didx").unique().sort("didx")
scores = {}
CACHE = cfg.path("reports") / "alpha" / "pit_scores"
CACHE.mkdir(parents=True, exist_ok=True)
for label, feats in FEATS.items():
    cp = CACHE / (label.replace("/", "_").replace(" ", "_") + ".parquet")
    if cp.exists():
        scores[label] = pl.read_parquet(cp); print(f"scored [{label}]: cached"); continue
    obj = "rank"
    if feats and feats[0] == "REG":
        obj, feats = "reg", feats[1]
    spec = M.TrainSpec(horizons=(21, 63), features=feats, objective=obj)
    hf, rcols = M.prepare(hist, feats)
    wf, _ = M.prepare(pw, feats)
    parts = []
    for i, cut in enumerate(cuts):
        nxt = cuts[i + 1] if i + 1 < len(cuts) else date(2100, 1, 1)
        cut_idx = int(hdates.filter(pl.col("date") <= cut)["didx"][-1])
        blk = wf.filter((pl.col("date") >= cut) & (pl.col("date") < nxt))
        X = blk.select(rcols).to_numpy().astype(np.float32)
        cols = {"date": blk["date"], "ticker": blk["ticker"]}
        for h in (21, 63):
            tr = M.training_rows(hf, h, cut_idx, spec)
            Xt, yt = M._xy(tr, rcols, h)
            m = M.AlphaGBM(spec, h, rcols).fit(Xt, yt, M.sample_weights(tr, cut_idx, spec), qid=tr["didx"].to_numpy())
            cols[f"s{h}"] = m.predict(X)
        parts.append(pl.DataFrame(cols))
    s = pl.concat(parts)
    z = lambda c: (pl.col(c) - pl.col(c).mean().over("date")) / (pl.col(c).std().over("date") + 1e-9)
    scores[label] = s.select("date", "ticker", comp=0.3 * z("s21") + 0.7 * z("s63"))
    scores[label].write_parquet(cp)
    print(f"scored [{label}]: {s.height:,} rows · {len(cuts)} refits ({time.time()-t0:.0f}s)")
# 5. signal skill on the PIT universe
pw_f = pw.select("date", "ticker", "fwd_21", "fwd_63")
for label, s in scores.items():
    j = s.join(pw_f, on=["date", "ticker"])
    out = []
    for h in (21, 63):
        ic = j.drop_nulls(f"fwd_{h}").group_by("date").agg(pl.corr("comp", f"fwd_{h}", method="spearman").alias("ic"))["ic"]
        out.append(f"{h}d IC {ic.mean():+.4f} (t {ic.mean()/ic.std()*np.sqrt(len(ic)/h):.1f}, non-overlapping)")
    print(f"signal [{label}] on PIT universe: " + " · ".join(out))
# 6. the book: same rules, PIT eligibility, whole-market prices for fills and delistings
spy_q = pl.scan_parquet(B / "ohlcv_all.parquet").filter(pl.col("ticker").is_in(["SPY", "QQQ", "SMH"])).select("date", "ticker", "open", "high", "low", "close", "adj_close", "volume").collect()
pdata = build_panel(P.sanitize_prices(pl.concat([px.filter(pl.col("ticker").is_in(list(names))), spy_q]).unique(subset=["date", "ticker"])))
sector_of = dict(pw.select("ticker", "sector").unique(subset=["ticker"]).iter_rows())
trend = dict(pw.group_by("date").agg(pl.col("mkt_trend_200").first()).iter_rows())
reg = {d: market_regime_on(v) for d, v in trend.items()}
gate = pw.select("date", "ticker", "adj_close", elig=(pl.col("close") >= 5) & (pl.col("vol_63") / 252 ** 0.5 <= 0.07) & (pl.col("log_dv_21") >= np.log1p(20e6)))
bk = replace(BookConfig(), min_price=0.0, min_dollar_volume=0.0)
OOS = cuts[0]
wcfg = WalkForwardConfig(oos_start=OOS, rebalance_days=21, retrain_days=10**6, min_train_days=5, horizon=21, top_k=20, max_weight=0.08,
                         min_dollar_volume=0.0, min_price=0.0, partial_trade_rate=0.35, respect_target_gross=True)
def book(label, col_df, delay=0, weight_fn=None, rg=reg):
    f = gate.join(col_df, on=["date", "ticker"], how="left").sort(["ticker", "date"])
    if delay:
        f = f.with_columns(pl.col("comp").shift(delay).over("ticker"))        # act on a stale signal
    f = f.with_columns(comp=pl.when(pl.col("elig")).then(pl.col("comp").cast(pl.Float64))).filter(pl.col("comp").is_not_null())
    res = run_walk_forward(pdata, f, ["comp"], lambda: ColumnAlpha("comp"), wcfg, weight_fn=weight_fn or _weight_fn(bk, sector_of, rg), label=label, progress=False)
    return res
runs = {}
for label, s in scores.items():
    runs[f"book · {label}"] = book(label, s)
runs["book · all features, +1 day stale signal"] = book("d1", scores["all features"], delay=1)
runs["book · all features, +5 days stale signal"] = book("d5", scores["all features"], delay=5)
rng = np.random.default_rng(3)
plc = scores["all features"].with_columns(comp=pl.Series(rng.standard_normal(scores["all features"].height)))
runs["placebo · random scores"] = book("placebo", plc)
dates = runs["book · all features"].daily["date"].to_list()
tick = list(pdata.tickers); didx = pdata.date_index(); idx = np.array([didx[d] for d in dates])
bench = {t: pdata.ret[idx, tick.index(t)] for t in ("SPY", "QQQ", "SMH") if t in tick}
G = gate.filter(pl.col("elig")).sort(["ticker", "date"]).with_columns(nxt=pl.col("date"))
pr = px.sort(["ticker", "date"]).with_columns(r=pl.col("adj_close") / pl.col("adj_close").shift(1).over("ticker") - 1, prev=pl.col("date").shift(1).over("ticker"))
ew = pr.join(G.select(prev="date", ticker="ticker"), on=["prev", "ticker"], how="inner").filter(pl.col("r").abs() < 5).group_by("date").agg(pl.col("r").mean())
m = dict(ew.iter_rows()); bench["PIT eligible EW"] = np.array([m.get(d, 0.0) for d in dates])
ai_cols = [tick.index(t) for t in AI_THEME if t in tick]
ra, al = pdata.ret[np.ix_(idx, ai_cols)], pdata.alive[np.ix_(idx, ai_cols)]
bench["AI/hardware basket EW"] = (ra * al).sum(1) / np.maximum(al.sum(1), 1)
def st(r):
    eq = np.cumprod(1 + r); y = len(r) / 252
    return eq[-1] ** (1 / y) - 1, eq[-1] - 1, sharpe(r), r.std() * 252 ** 0.5, float((eq / np.maximum.accumulate(eq) - 1).min())
print(f"\nBOOK {dates[0]} → {dates[-1]} ({len(dates)} sessions), after costs")
for k, res in runs.items():
    r = res.returns(); c, t, s_, v, dd = st(r); ci = bootstrap_ci(r, n_boot=400)
    print(f"{k:44s} CAGR {c:+7.1%} total {t:+7.1%} Sharpe {s_:5.2f} [{ci['lo']:+.2f},{ci['hi']:+.2f}] vol {v:5.1%} MaxDD {dd:+6.1%} turn {summarize_result(res)['ann_turnover']:.1f}")
for k, r in bench.items():
    c, t, s_, v, dd = st(r); print(f"{k:44s} CAGR {c:+7.1%} total {t:+7.1%} Sharpe {s_:5.2f}                 vol {v:5.1%} MaxDD {dd:+6.1%}")
ab = runs["book · all features"].returns()
for nm in ("SPY", "QQQ"):
    if nm in bench:
        bm = bench[nm]; b = np.cov(ab, bm)[0, 1] / np.var(bm)
        print(f"vs {nm}: beta {b:.2f}, alpha {(ab.mean() - b * bm.mean()) * 252:+.1%}/yr")
diff = ab - bench["PIT eligible EW"]; ci = bootstrap_ci(diff, stat=lambda x: x.mean() * 252, n_boot=400)
print(f"excess over PIT eligible EW: {ci['point']:+.1%}/yr [{ci['lo']:+.1%}, {ci['hi']:+.1%}]")
dd_ = np.array(dates)
for name, a0, a1 in (("DeepSeek shock", date(2025, 1, 24), date(2025, 2, 7)), ("Tariff crash", date(2025, 2, 19), date(2025, 4, 8)),
                     ("Post-tariff AI rebound", date(2025, 4, 8), date(2025, 7, 31)), ("Last 6 months", date(2026, 3, 18), dates[-1])):
    mk = (dd_ >= a0) & (dd_ <= a1)
    if mk.sum() > 2:
        print(f"{name:24s} book {np.prod(1+ab[mk])-1:+6.1%} · SPY {np.prod(1+bench['SPY'][mk])-1:+6.1%} · QQQ {np.prod(1+bench['QQQ'][mk])-1:+6.1%} · AI basket {np.prod(1+bench['AI/hardware basket EW'][mk])-1:+6.1%} · PIT EW {np.prod(1+bench['PIT eligible EW'][mk])-1:+6.1%}")
W = runs["book · all features"].weights
held = W.select("ticker").unique()["ticker"].to_list()
print(f"names held: {len(held)}; of them outside today's research universe: {len(set(held) - biased)}; AI/hardware weight avg {float(W.filter(pl.col('ticker').is_in(list(AI_THEME)))['weight'].sum() / W['date'].n_unique()):.1%}")
