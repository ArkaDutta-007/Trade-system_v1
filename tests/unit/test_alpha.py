"""Alpha engine v2 — network-free unit tests on synthetic panels."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from trading_system.alpha import ledger as L
from trading_system.alpha import model as M
from trading_system.alpha import panel as P
from trading_system.alpha import portfolio as B
from trading_system.alpha.backtest import causal_composite


# ── fixtures ──────────────────────────────────────────────────────────────────

def _bdays(n: int, start=date(2020, 1, 1)) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def synth_prices(n_tickers=30, n_days=700, seed=0, signal_col=None) -> pl.DataFrame:
    """Geometric random walks; each ticker has its own drift so momentum-ish features carry signal."""
    rng = np.random.default_rng(seed)
    days = _bdays(n_days)
    rows = []
    for j in range(n_tickers):
        mu = rng.normal(0.0003, 0.0006)
        sig = rng.uniform(0.01, 0.03)
        r = rng.normal(mu, sig, n_days)
        px = 50 * np.exp(np.cumsum(r))
        vol = rng.uniform(2e5, 2e6, n_days)
        for i, d in enumerate(days):
            o = px[i] * (1 + rng.normal(0, 0.002))
            rows.append({"date": d, "ticker": f"T{j:02d}", "open": o, "high": max(o, px[i]) * 1.01,
                         "low": min(o, px[i]) * 0.99, "close": px[i], "adj_close": px[i], "volume": vol[i]})
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date)).sort(["ticker", "date"])


def synth_panel(**kw) -> pl.DataFrame:
    px = synth_prices(**kw)
    pn = P.price_features(px).with_columns(sector=pl.lit("manufacturing"))
    pn = P._join_fundamentals(pn, pl.DataFrame())
    pn = P._join_news(pn, pl.DataFrame())
    pn = P._join_short(pn, None, None)
    pn = P.market_features(pn)
    pn = P.add_labels(pn, (5, 21))
    feats = [c for c in P.FEATURE_COLS if c in pn.columns]
    return pn.with_columns([pl.col(c).cast(pl.Float32) for c in feats]).select(
        [c for c in P.BASE_COLS + feats + ["fwd_5", "fwd_21", "y_5", "y_21"] if c in pn.columns]).sort(["date", "ticker"])


# ── panel ─────────────────────────────────────────────────────────────────────

def test_price_features_match_hand_calculation():
    px = synth_prices(n_tickers=2, n_days=300)
    f = P.price_features(px).filter(pl.col("ticker") == "T00").sort("date")
    ac = f["adj_close"].to_numpy()
    i = 280
    assert f["ret_21"][i] == pytest.approx(ac[i] / ac[i - 21] - 1, rel=1e-6)
    assert f["mom_12_1"][i] == pytest.approx(ac[i - 21] / ac[i - 252] - 1, rel=1e-6)
    r1 = np.log(ac[1:] / ac[:-1])
    assert f["vol_21"][i] == pytest.approx(np.std(r1[i - 21:i], ddof=1) * np.sqrt(252), rel=1e-4)
    assert f["dist_52w_high"][i] == pytest.approx(ac[i] / ac[i - 251:i + 1].max() - 1, rel=1e-6)
    assert f["log_dv_21"][i] == pytest.approx(np.log1p(np.mean((f["close"] * f["volume"]).to_numpy()[i - 20:i + 1])), rel=1e-6)
    # every feature is backward-looking: perturbing the LAST bar must not change features at i
    px2 = px.with_columns(pl.when((pl.col("ticker") == "T00") & (pl.col("date") == px["date"].max()))
                            .then(pl.col("adj_close") * 5).otherwise(pl.col("adj_close")).alias("adj_close"))
    f2 = P.price_features(px2).filter(pl.col("ticker") == "T00").sort("date")
    for c in P.PRICE_FEATURES:
        if c in f.columns:
            a, b = f[c][i], f2[c][i]
            assert (a == b) or (np.isnan(a) and np.isnan(b)), c


def test_labels_are_forward_looking_gaussian_ranks():
    pn = synth_panel(n_tickers=20, n_days=300)
    last = sorted(pn["date"].unique().to_list())
    assert pn.filter(pl.col("date") >= last[-21])["y_21"].is_null().all()          # cannot be known yet
    d = pn.filter(pl.col("date") == last[-60])
    assert abs(d["y_21"].mean()) < 0.05 and 0.8 < d["y_21"].std() < 1.2               # per-date standardised
    assert d.sort("fwd_21")["y_21"].is_sorted()                                        # monotone in the raw return
    # the raw forward return is exactly adj_close[t+h]/adj_close[t]-1
    t = pn.filter(pl.col("ticker") == "T00").sort("date")
    ac = t["adj_close"].to_numpy()
    assert t["fwd_21"][10] == pytest.approx(ac[31] / ac[10] - 1, rel=1e-6)


def test_fundamentals_join_is_point_in_time():
    px = synth_prices(n_tickers=1, n_days=400)
    fin = pl.DataFrame({
        "ticker": ["T00"] * 5, "timeframe": ["quarterly"] * 4 + ["annual"],
        "end_date": ["2019-12-31", "2020-03-31", "2020-06-30", "2020-09-30", "2020-09-30"],
        "filing_date": ["2020-02-15", "2020-05-15", "2020-08-14", "2020-11-13", "2020-11-13"],
        "income_statement__revenues": [100.0, 110.0, 120.0, 130.0, 460.0],
        "income_statement__net_income_loss": [10.0, 11.0, 12.0, 13.0, 46.0],
        "income_statement__gross_profit": [50.0] * 4 + [200.0],
        "cash_flow_statement__net_cash_flow_from_operating_activities": [12.0] * 4 + [48.0],
        "balance_sheet__assets": [1000.0, 1010.0, 1020.0, 1030.0, 1030.0],
        "balance_sheet__equity": [500.0] * 5, "balance_sheet__liabilities": [500.0] * 5,
        "income_statement__diluted_average_shares": [100.0] * 5,
    })
    fund = P.fundamental_features(fin)
    pn = P._join_fundamentals(P.price_features(px), fund).sort("date")
    before = pn.filter(pl.col("date") <= date(2020, 2, 15))
    assert before["roe_ttm"].is_null().all()                                             # nothing filed yet
    on = pn.filter(pl.col("date") == date(2020, 2, 18)).row(0, named=True)             # first session after filing+1
    assert on["roe_ttm"] is None and on["leverage"] == pytest.approx(0.5)               # levels known, TTM needs 4 quarters
    mid = pn.filter(pl.col("date") == date(2020, 11, 13)).row(0, named=True)
    assert mid["roe_ttm"] is None                                                       # filed today → usable tomorrow
    late = pn.filter(pl.col("date") == date(2020, 12, 1)).row(0, named=True)
    assert late["roe_ttm"] == pytest.approx(46.0 / 500.0)                               # 4 quarters (span ok) = 46
    assert late["ep_ttm"] == pytest.approx(46.0 / (100.0 * late["close"]))
    assert late["fund_age_days"] == (date(2020, 12, 1) - date(2020, 11, 13)).days


def test_news_join_lands_the_day_after_publication():
    px = synth_prices(n_tickers=1, n_days=60)
    days = sorted(px["date"].unique().to_list())
    pub = days[30]
    news = pl.DataFrame({"ticker": ["T00", "T00"], "published_utc": [f"{pub}T14:00:00Z", f"{pub}T22:00:00Z"],
                         "sentiment": [1.0, -1.0]}).with_columns(pl.col("published_utc").str.to_datetime("%Y-%m-%dT%H:%M:%SZ", time_zone="UTC"))
    nd = P.news_daily(news)
    pn = P._join_news(P.price_features(px), nd).sort("date")
    assert pn.filter(pl.col("date") == pub)["news_n_5"].is_null().all() or pn.filter(pl.col("date") == pub)["news_n_5"][0] == 0
    nxt = pn.filter(pl.col("date") > pub).row(0, named=True)
    assert nxt["news_n_5"] == 2 and nxt["news_sent_5"] == pytest.approx(0.0)


# ── model ─────────────────────────────────────────────────────────────────────

def test_training_rows_are_purged_and_embargoed():
    pn = synth_panel(n_tickers=10, n_days=400)
    frame, rcols = M.prepare(pn)
    spec = M.TrainSpec(stride={5: 1, 21: 1}, embargo_days=5)
    cut = 300
    tr = M.training_rows(frame, 21, cut, spec)
    assert tr["didx"].max() <= cut - 21 - 5 and tr["y_21"].is_null().sum() == 0
    assert set(rcols) == {f + "_r" for f in P.MODEL_FEATURES if f in pn.columns}
    assert not any(f in P.DATE_LEVEL_FEATURES for f in P.MODEL_FEATURES)
    # date-level columns, when explicitly requested, pass through raw instead of collapsing to a 0.5 rank
    fr2, rc2 = M.prepare(pn, ["ret_21", "mkt_vol_21"])
    assert fr2.filter(pl.col("didx") == 300)["mkt_vol_21_r"].n_unique() == 1
    assert fr2.filter(pl.col("didx") == 300)["mkt_vol_21_r"][0] == pytest.approx(float(pn.filter(pl.col("date") == fr2.filter(pl.col("didx") == 300)["date"][0])["mkt_vol_21"][0]), rel=1e-5)
    r = frame.filter(pl.col("didx") == 300)["ret_21_r"]
    assert 0 < r.min() and r.max() < 1                                                  # per-date pct ranks


def test_causal_scores_recover_a_planted_signal(tmp_path):
    pn = synth_panel(n_tickers=40, n_days=800, seed=3)
    # plant: the forward 21d rank is partly predictable from up_frac_63 (a real feature)
    rng = np.random.default_rng(0)
    u = pn["up_frac_63"].fill_null(0.5).to_numpy()
    pn = pn.with_columns(y_21=pl.Series((0.6 * (u - 0.5) * 6 + 0.8 * rng.normal(size=pn.height)).astype(np.float32))
                         .fill_nan(None))
    spec = M.TrainSpec(horizons=(21,), n_rounds=60, seeds=(0,), device="cpu", stride={21: 2}, half_life_days=0)
    sc = M.causal_scores(pn, spec, refit_every=100, min_train_days=300, progress=False)
    assert sc["date"].min() >= sorted(pn["date"].unique().to_list())[300]               # nothing scored before the first cut
    j = sc.join(pn.select("date", "ticker", "y_21"), on=["date", "ticker"]).drop_nulls("y_21")
    ic = j.group_by("date").agg(pl.corr("score", "y_21", method="spearman").alias("ic"))["ic"].mean()
    assert ic > 0.2
    assert sc["model_id"].n_unique() >= 4


def test_fit_predict_save_load_roundtrip(tmp_path):
    pn = synth_panel(n_tickers=15, n_days=400)
    spec = M.TrainSpec(horizons=(5,), n_rounds=20, seeds=(0, 1), device="cpu", stride={5: 1})
    models = M.fit_production(pn, spec, tmp_path / "m")
    loaded = M.load_production(tmp_path / "m")
    pred = M.predict_dates(pn, models)
    pred2 = M.predict_dates(pn, loaded)
    assert pred.height == pn.filter(pl.col("date") == pn["date"].max()).height
    assert np.allclose(pred["score"].to_numpy(), pred2["score"].to_numpy())
    assert (tmp_path / "m" / "manifest.json").exists() and models[5].trained_through is not None
    ex = M.explain_latest(pn, models, ["T00"], 5, top=3)
    assert len(ex["T00"]) == 3 and all(isinstance(v, float) for _, v in ex["T00"])


# ── ledger ────────────────────────────────────────────────────────────────────

def _prices_for(days: list[date], tickers: list[str], base=100.0, step=1.0) -> pl.DataFrame:
    rows = [{"date": d, "ticker": t, "adj_close": base + step * i + 10 * k} for k, t in enumerate(tickers) for i, d in enumerate(days)]
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def test_ledger_upsert_tally_and_delisting(tmp_path):
    days = _bdays(40)
    px = _prices_for(days, ["A", "B"])
    px = px.filter(~((pl.col("ticker") == "B") & (pl.col("date") > days[20])))        # B delists after day 20
    lp = tmp_path / "led.parquet"
    rows = pl.DataFrame({"date": [days[10], days[10], days[35]], "ticker": ["A", "B", "A"], "horizon": [5, 21, 5],
                         "mode": ["live"] * 3, "model_id": ["m"] * 3, "score": [0.5, -0.2, 0.1],
                         "exp_ret": [0.02, -0.01, 0.0], "q10": [-0.05, -0.1, -0.05], "q90": [0.1, 0.05, 0.05],
                         "entry_price": [110.0, 120.0, 135.0]})
    assert L.upsert(lp, rows) == 3
    assert L.upsert(lp, rows.head(1)) == 1 and L.load(lp).height == 3                  # idempotent by key
    res = L.tally(lp, px)
    led = L.load(lp).sort(["date", "ticker"])
    a = led.filter((pl.col("ticker") == "A") & (pl.col("date") == days[10])).row(0, named=True)
    assert a["realized_ret"] == pytest.approx((100 + 15) / (100 + 10) - 1) and a["realized_date"] == days[15] and a["hit"]
    b = led.filter(pl.col("ticker") == "B").row(0, named=True)
    assert b["realized_date"] == days[20] and b["realized_ret"] == pytest.approx((110 + 20) / (110 + 10) - 1)   # scored at last print
    assert not b["hit"] and b["in_band"] is False                                       # forecast −1%, realised +8% → miss, outside band
    late = led.filter(pl.col("date") == days[35]).row(0, named=True)
    assert late["realized_ret"] is None and res == {"matured": 2, "pending": 1}
    # re-upserting the same key without outcomes keeps the realised values
    L.upsert(lp, rows.head(1).with_columns(score=pl.lit(0.9)))
    a2 = L.load(lp).filter((pl.col("ticker") == "A") & (pl.col("date") == days[10])).row(0, named=True)
    assert a2["score"] == pytest.approx(0.9) and a2["realized_ret"] == pytest.approx(a["realized_ret"])


def _synthetic_ledger(n_dates=300, n_tk=60, seed=1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    days = _bdays(n_dates)
    rows = []
    for i, d in enumerate(days):
        s = rng.normal(size=n_tk)
        for h, k in ((5, 0.15), (21, 0.30)):
            r = k * 0.02 * s + rng.normal(0, 0.05, n_tk)
            for j in range(n_tk):
                rows.append({"date": d, "ticker": f"T{j}", "horizon": h, "mode": "backtest", "model_id": "m",
                             "score": float(s[j]), "exp_ret": None, "q10": None, "q90": None, "entry_price": 1.0,
                             "realized_ret": float(r[j]), "realized_date": days[min(i + h, n_dates - 1)]})
    return L._conform(pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date), pl.col("realized_date").cast(pl.Date)))


def test_skill_report_and_calibrator_learn_from_the_tally():
    led = _synthetic_ledger()
    rep = L.skill_report(led)
    r21 = rep.filter(pl.col("horizon") == 21).row(0, named=True)
    r5 = rep.filter(pl.col("horizon") == 5).row(0, named=True)
    assert r21["ic_mean"] > r5["ic_mean"] > 0 and r21["decile_spread"] > 0 and r21["t_stat"] > 5
    cal = L.Calibrator.fit(led, (5, 21), window_days=10_000)
    cal.min_rows = 100
    c = cal.horizons[21]
    assert np.all(np.diff(c.y) >= -1e-12)                                              # isotonic → monotone
    assert c.expected(np.array([2.0]))[0] > c.expected(np.array([-2.0]))[0]
    w = cal.skill_weights()
    assert w[21] > w[5] and abs(sum(w.values()) - 1) < 1e-9
    applied = cal.apply(led.select("date", "ticker", "horizon", "score"))
    cov = ((led["realized_ret"] >= applied["q10"]) & (led["realized_ret"] <= applied["q90"])).mean()
    assert 0.74 <= cov <= 0.86                                                          # conformal 80% band holds in-sample
    y = L.yearly_ic(led)
    assert set(y["horizon"].unique().to_list()) == {5, 21}
    comp = L.composite_score(led.select("date", "ticker", "horizon", "score"), w)
    assert comp.columns == ["date", "ticker", "composite"] and comp.height == led.select("date", "ticker").unique().height


def test_calibrator_round_trips_through_json(tmp_path):
    led = _synthetic_ledger(n_dates=120)
    cal = L.Calibrator.fit(led, (5, 21), window_days=10_000)
    cal.save(tmp_path / "cal.json")
    back = L.Calibrator.load(tmp_path / "cal.json")
    s = np.array([-1.5, 0.0, 0.3, 2.0])
    for h in (5, 21):
        assert np.allclose(back.horizons[h].expected(s), cal.horizons[h].expected(s))
        assert np.allclose(back.horizons[h].band(s)[1], cal.horizons[h].band(s)[1])
        assert back.horizons[h].n == cal.horizons[h].n and back.horizons[h].ic == pytest.approx(cal.horizons[h].ic)


def test_causal_composite_only_uses_matured_history():
    led = _synthetic_ledger(n_dates=200)
    comp, log = causal_composite(led, (5, 21), refit_every=50, min_rows=100)
    assert comp.height == led.select("date", "ticker").unique().height
    assert log[0]["ic"] == {}                                                             # first block: nothing matured → equal weights
    assert log[0]["weights"] == {5: 0.5, 21: 0.5}
    assert any(e["ic"] for e in log[1:])                                                # later blocks learned weights
    both = [e for e in log if set(e["ic"]) == {5, 21}]
    assert both and np.mean([e["weights"][21] for e in both]) > np.mean([e["weights"][5] for e in both])   # prefers the stronger horizon


# ── portfolio ─────────────────────────────────────────────────────────────────

def test_target_weights_respect_gates_sector_cap_and_vol_target():
    n = 40
    rng = np.random.default_rng(0)
    scores = rng.normal(size=n)
    scores[:5] += 3.0                                                                    # five strong names
    dvol = np.full(n, 0.02); dvol[0] = 0.10                                              # name 0 too volatile
    adv = np.full(n, 50e6); adv[1] = 1e6                                                 # name 1 illiquid
    price = np.full(n, 30.0); price[2] = 2.0                                             # name 2 penny
    sectors = np.array(["tech"] * 20 + ["fin"] * 20)
    cfg = B.BookConfig(top_k=10, max_weight=0.15, max_per_sector=6, vol_target=0.10, avg_corr=0.3, max_daily_vol=0.07)
    assert B.market_regime_on(0.02) and not B.market_regime_on(-0.01) and B.market_regime_on(None)
    off = B.target_weights(scores, {"dvol": dvol, "adv": adv, "price": price}, B.BookConfig(top_k=10, vol_target=None, regime_scale=0.5), sectors, regime_on=False)
    assert off.sum() == pytest.approx(0.5)
    w = B.target_weights(scores, {"dvol": dvol, "adv": adv, "price": price}, cfg, sectors)
    assert w[0] == 0 and w[1] == 0 and w[2] == 0
    assert (w > 0).sum() == 10 and (w[:20] > 0).sum() <= 6
    assert w.max() <= 0.15 + 1e-9
    pv = B.portfolio_vol(w, dvol, 0.3)
    assert pv <= 0.10 + 1e-6 and w.sum() < 1.0                                          # scaled down to the vol target
    w2 = B.target_weights(scores, {"dvol": dvol, "adv": adv, "price": price}, B.BookConfig(top_k=10, vol_target=None), sectors)
    assert w2.sum() == pytest.approx(1.0)


def test_partial_rebalance_moves_toward_target_and_exits():
    cfg = B.BookConfig(trade_rate=0.5, no_trade_band=0.2)
    cur = np.array([0.10, 0.10, 0.00, 0.004])
    tgt = np.array([0.10, 0.00, 0.10, 0.00])
    new = B.partial_rebalance(cur, tgt, cfg)
    assert new[0] == 0.10 and new[1] == pytest.approx(0.05) and new[2] == pytest.approx(0.05) and new[3] == 0.0
    small = B.partial_rebalance(np.array([0.095]), np.array([0.10]), cfg)
    assert small[0] == 0.095                                                             # inside the band → untouched


def test_build_book_end_to_end():
    n = 30
    today = pl.DataFrame({"ticker": [f"T{i}" for i in range(n)], "composite": np.linspace(-1, 1, n),
                          "dvol": [0.02] * n, "adv": [50e6] * n, "price": [40.0] * n,
                          "sector": ["a"] * 15 + ["b"] * 15})
    cfg = B.BookConfig(top_k=8, max_per_sector=5, vol_target=None, regime_scale=1.0)
    book = B.build_book(today, cfg, prev=None)
    held = book.filter(pl.col("weight") > 0)
    assert held.height == 8 and held["ticker"][0] == "T29" and held["weight"].sum() == pytest.approx(1.0)
    prev = {t: w for t, w in zip(held["ticker"], held["weight"])}
    today2 = today.with_columns(composite=-pl.col("composite"))                          # signal flips
    book2 = B.build_book(today2, cfg, prev)
    h2 = book2.filter(pl.col("weight") > 0)
    assert "T0" in h2["ticker"].to_list() and "T29" in h2["ticker"].to_list()             # partially in, partially out
    assert h2.filter(pl.col("ticker") == "T29")["weight"][0] < prev["T29"]


# ── regime layer ──────────────────────────────────────────────────────────────

def test_regime_standardize_is_expanding_and_similarity_finds_the_lookalike():
    from trading_system.alpha import regime as R
    days = _bdays(1200, start=date(2015, 1, 1))
    rng = np.random.default_rng(0)
    base = rng.normal(size=(1200, 4))
    base[600:620] += 3.0                                    # one stress episode
    base[-1] = base[610]                                    # today looks exactly like the episode
    rf = pl.DataFrame({"date": days, "vix": base[:, 0], "oil_vol_21": base[:, 1], "baa_spread": base[:, 2], "avg_corr_21": base[:, 3]})
    zf = R.standardize(rf, min_days=100)
    z = zf["vix_z"].to_numpy()
    assert np.isnan(z[:99]).all() and np.isfinite(z[100:]).all()
    # point-in-time: the z at row i must not change when later rows change
    rf2 = rf.with_columns(pl.when(pl.col("date") > days[800]).then(pl.col("vix") + 50).otherwise(pl.col("vix")).alias("vix"))
    z2 = R.standardize(rf2, min_days=100)["vix_z"].to_numpy()
    assert np.allclose(z[100:800], z2[100:800])
    an = R.similarity(zf, cols=["vix", "oil_vol_21", "baa_spread", "avg_corr_21"], k=5, exclude_recent_days=30, min_gap_days=5, tau=20)
    assert an.nearest["date"][0] == days[610] and an.weights["w"].sum() == pytest.approx(1.0)
    assert (an.weights.filter(pl.col("date") >= days[600]).filter(pl.col("date") <= days[619])["w"].sum()) > 0.5


def test_fragility_and_gross_multiplier():
    from trading_system.alpha import regime as R
    assert R.gross_multiplier(0.0) == 1.0 and R.gross_multiplier(None) == 1.0
    assert R.gross_multiplier(1.5) == pytest.approx(0.75) and R.gross_multiplier(3.0) == 0.5
    days = _bdays(30)
    zf = pl.DataFrame({"date": days, "vix_z": [2.0] * 30, "mkt_vol_21_z": [0.0] * 30, "oil_level_z_z": [1.0] * 30})
    f = R.fragility_score(zf)
    assert f["stress"][0] == pytest.approx(1.0) and f["imbalance"][0] == pytest.approx(1.0) and f["fragility"][0] == pytest.approx(1.0)
    assert 0.4 < f["crisis_like"][0] < 0.6


def test_calibrator_analog_blend_uses_the_weights():
    led = _synthetic_ledger(n_dates=300)
    dates = sorted(led["date"].unique().to_list())
    # analog weights that only see the LAST 100 days, where we make realised returns systematically higher
    boost = pl.when(pl.col("date") >= dates[200]).then(pl.col("realized_ret") + 0.05).otherwise(pl.col("realized_ret"))
    led = led.with_columns(realized_ret=boost.cast(pl.Float32))
    w = pl.DataFrame({"date": dates, "w": [0.0] * 200 + [1.0] * 100})
    plain = L.Calibrator.fit(led, (21,), window_days=10_000)
    mixed = L.Calibrator.fit(led, (21,), window_days=10_000, date_weights=w, analog_blend=1.0)
    s = np.array([0.0])
    assert mixed.horizons[21].source == "blend" and mixed.horizons[21].ess > 2000
    assert mixed.horizons[21].expected(s)[0] > plain.horizons[21].expected(s)[0] + 0.03   # analog fit sees the boosted regime
    half = L.Calibrator.fit(led, (21,), window_days=10_000, date_weights=w, analog_blend=0.5)
    assert plain.horizons[21].expected(s)[0] < half.horizons[21].expected(s)[0] < mixed.horizons[21].expected(s)[0]
    assert half.horizons[21].q_hi[2] >= plain.horizons[21].q_hi[2] - 1e-9                # bands never narrower than trailing


# ── earnings surprise / experiment harness / data status ─────────────────────

def test_earnings_events_sue_uses_prior_dispersion_and_is_point_in_time():
    ends = [date(2018 + i // 4, 3 * (i % 4) + 1, 28) for i in range(12)]
    eps = [1.0, 1.0, 1.0, 1.0, 1.1, 1.0, 1.1, 1.0, 1.2, 1.0, 1.2, 2.0]      # last quarter: a big beat
    fin = pl.DataFrame({"ticker": ["T00"] * 12, "timeframe": ["quarterly"] * 12,
                        "end_date": [str(d) for d in ends], "filing_date": [str(d + timedelta(days=35)) for d in ends],
                        "income_statement__diluted_earnings_per_share": eps,
                        "income_statement__revenues": [100.0 + i for i in range(12)]})
    ev = P.earnings_events(fin).sort("end_date")
    assert ev["sue"][:4].is_null().all()                                   # needs a year of seasonal history
    assert ev["sue"][-1] > 3 and ev["sue"][-1] > ev["sue"][-2]            # scaled by PRIOR surprises only
    px = synth_prices(n_tickers=1, n_days=1200, seed=1)
    pn = P.price_features(px)
    j = P._join_earnings(pn, ev).sort("date")
    f_last = ev["filing_date"][-1]
    assert j.filter(pl.col("date") <= f_last)["sue"].drop_nulls().max() < 3       # not visible on the filing day
    after = j.filter(pl.col("date") > f_last).head(1)
    assert after["sue"][0] > 3
    w = j.filter(pl.col("date") > f_last).head(3)["ear_3d"].to_list()
    assert w[0] is None and w[-1] is not None                              # window return only after the window closes


def test_experiment_compare_and_verdict():
    from trading_system.alpha import experiment as X
    days = _bdays(400)
    rng = np.random.default_rng(0)
    base = pl.DataFrame({"date": days, "horizon": [63] * 400, "ic": rng.normal(0.02, 0.05, 400), "spread": [0.01] * 400})
    good = base.with_columns(ic=pl.col("ic") + 0.05)
    tab = X.compare({"base": base, "good": good, "same": base})
    ok, msg = X.verdict(tab, "good")
    assert ok and "ADOPT" in msg
    assert not X.verdict(tab, "same")[0]
    a = pl.DataFrame({"date": [days[0]] * 3, "ticker": ["A", "B", "C"], "horizon": [63] * 3, "score": [1.0, 2.0, 3.0]})
    b = a.with_columns(score=pl.Series([3.0, 2.0, 1.0]))
    assert X.blend(a, b)["score"].abs().max() < 1e-6                        # opposite signals cancel


def test_data_status_freshness_rules(tmp_path):
    from trading_system import datastatus as DS
    assert DS.last_session(date(2026, 9, 26)) == date(2026, 9, 25)          # Saturday → Friday
    assert DS.sessions_between(date(2026, 9, 18), date(2026, 9, 25)) == 5
    (tmp_path / "data/bronze").mkdir(parents=True)
    pl.DataFrame({"date": [date(2026, 9, 10), date(2026, 9, 25)], "ticker": ["A", "B"]}).write_parquet(tmp_path / "data/bronze/ohlcv_daily.parquet")
    s = DS.Store("x", "data/bronze/ohlcv_daily.parquet", "date", "t", 1)
    r = DS.inspect(tmp_path, s, date(2026, 9, 26))
    assert r["status"] == "ok" and r["rows"] == 2 and r["tickers"] == 2 and r["lag"] == 0
    r2 = DS.inspect(tmp_path, s, date(2026, 10, 20))
    assert r2["status"] == "STALE"
    assert DS.inspect(tmp_path, DS.Store("y", "nope.parquet", "date", "t", 1), date(2026, 9, 26))["status"] == "missing"
