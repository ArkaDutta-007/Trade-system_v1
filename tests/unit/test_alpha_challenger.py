"""Challenger engine: index-minus targets, tranche blending, and the share-unit repair it exposed."""
from datetime import date

import polars as pl
import pytest

from trading_system.alpha import challenger as C
from trading_system.alpha import panel as P


def _xs(n=10, **over):
    d = {"date": [date(2026, 11, 12)] * n, "ticker": [f"T{i:02d}" for i in range(n)], "close": [50.0] * n,
         "log_dv_21": [20.0 - i * 0.1 for i in range(n)], "log_cap": [25.0] * n, "comp": [float(i) for i in range(n)]}
    d.update(over)
    return pl.DataFrame(d)


def test_fresh_targets_drops_the_lowest_scores_and_cap_weights_the_rest():
    t = C.fresh_targets(_xs(), top=10, drop=0.30)
    assert set(t["top300_cap"]) == {f"T{i:02d}" for i in range(10)}
    assert set(t["index_minus"]) == {f"T{i:02d}" for i in range(3, 10)}          # the three lowest scores are out
    assert sum(t["index_minus"].values()) == pytest.approx(1.0) and sum(t["top300_cap"].values()) == pytest.approx(1.0)
    assert t["index_minus"]["T05"] == pytest.approx(1 / 7)                        # equal caps → equal weights


def test_fresh_targets_weights_follow_cap_and_universe_follows_liquidity():
    xs = _xs(log_cap=[25.0] * 9 + [25.0 + 0.6931471805599453])                    # the last name is twice as big
    t = C.fresh_targets(xs, top=5, drop=0.0)
    assert set(t["top300_cap"]) == {f"T{i:02d}" for i in range(5)}                # the five most liquid only
    t = C.fresh_targets(xs, top=10, drop=0.0)
    assert t["top300_cap"]["T09"] == pytest.approx(2 / 11)


def test_fresh_targets_missing_score_is_never_dropped_and_missing_cap_is_estimated():
    xs = _xs(comp=[None] + [float(i) for i in range(1, 10)], log_cap=[None] + [25.0] * 9)
    t = C.fresh_targets(xs, top=10, drop=0.30)
    assert "T00" in t["index_minus"]                                             # median score, not last
    assert t["top300_cap"]["T00"] > 0                                            # cap estimated from dollar volume
    assert C.fresh_targets(_xs(close=[2.0] * 10), top=10) == {"index_minus": {}, "top300_cap": {}, "top20_fn": {}}   # nothing ≥ $5


def test_blend_is_the_mean_of_the_last_tranches():
    rows = []
    for k, d in enumerate([date(2026, 10, 1), date(2026, 10, 8), date(2026, 10, 15), date(2026, 10, 22), date(2026, 10, 29)]):
        rows += [{"date": d, "book": "index_minus", "ticker": "A", "w": 1.0 if k < 4 else 0.0},
                 {"date": d, "book": "index_minus", "ticker": "B", "w": 0.0 if k < 4 else 1.0}]
    snaps = pl.DataFrame(rows)
    w = C.blend(snaps, "index_minus", k=4)                                       # the oldest tranche has rolled off
    assert w["A"] == pytest.approx(0.75) and w["B"] == pytest.approx(0.25)
    young = C.blend(snaps.filter(pl.col("date") <= date(2026, 10, 8)), "index_minus", k=4)
    assert young["A"] == pytest.approx(1.0)                                      # two tranches: fully invested, not half
    assert C.blend(snaps, "top300_cap") == {}


def test_sessions_since_counts_later_sessions():
    cal = [date(2026, 11, d) for d in (2, 3, 4, 5, 6, 9)]
    assert C.sessions_since(cal, date(2026, 11, 4)) == 3
    assert C.sessions_since(cal, "2026-11-09") == 0
    assert C.sessions_since(cal, None) > 1000


def _q(shares, ni, eps, ticker="X"):
    n = len(shares)
    return pl.DataFrame({"ticker": [ticker] * n, "shares": [float(x) if x is not None else None for x in shares],
                         "ni": [float(x) for x in ni], "eps": eps}, schema_overrides={"eps": pl.Float64})


def test_repair_share_units_fixes_thousands_and_overshoot():
    # filed in thousands (NVDA 2011): 588,684 with EPS 0.29 on $171.7M → 588.7M shares
    assert P.repair_share_units(_q([588684], [1.71651e8], [0.29]))["shares"][0] == pytest.approx(5.88684e8)
    # 1,000× too many (WAT 2026): history 6.0e7, then 8.2e10 with EPS −0.87 on −$72M
    r = P.repair_share_units(_q([5.97e7, 5.96e7, 5.96e7, 8.2139e10], [1.2e8, 1.5e8, 1.5e8, -7.2e7], [2.03, 2.47, 2.5, -0.87]))
    assert r["shares"].to_list() == pytest.approx([5.97e7, 5.96e7, 5.96e7, 8.2139e7])


def test_repair_share_units_leaves_real_changes_alone():
    q = _q([1.0e8, 1.0e8, 1.0e8, 2.0e9], [1e8, 1e8, 1e8, 1e8], [1.0, 1.0, 1.0, 0.05])     # a 20-for-1 split
    assert P.repair_share_units(q)["shares"].to_list() == pytest.approx([1.0e8, 1.0e8, 1.0e8, 2.0e9])


def test_repair_share_units_trusts_the_independent_count_over_a_bad_eps():
    q = _q([8.0054e7], [-5.6454e7], [-705.2], ticker="BBOT")                    # the EPS is the mis-scaled one
    ref = pl.DataFrame({"ticker": ["BBOT"], "ref_shares": [8.0174e7]})
    assert P.repair_share_units(q, ref)["shares"][0] == pytest.approx(8.0054e7)
    assert P.repair_share_units(q)["shares"][0] == pytest.approx(8.0054e4)       # with nothing else to go on, EPS decides


def test_repair_share_units_falls_back_to_history_without_eps():
    q = _q([5.0e7, 5.1e7, 5.2e7, 5.3e10], [1e7] * 4, [None] * 4)
    assert P.repair_share_units(q)["shares"][3] == pytest.approx(5.3e7)


def test_honest_note_quotes_the_weekly_test_and_falls_back(monkeypatch):
    from trading_system.alpha import cli
    monkeypatch.setattr(C, "honest", lambda cfg: {"window": ["2024-11-26", "2026-11-12"], "production_top20": {"cagr": -0.012},
                                                  "spy": {"cagr": 0.141}})
    note = cli.honest_note(None)
    assert "−1.2%/yr vs SPY +14.1% (Nov 24 → Nov 26, re-run weekly)" in note and "no proven edge" in note
    monkeypatch.setattr(C, "honest", lambda cfg: None)
    assert cli.honest_note(None) == cli.HONEST_NOTE



def test_top20_fn_follows_the_production_book_rules():
    n = 40
    xs = pl.DataFrame({"date": [date(2026, 11, 12)] * n, "ticker": [f"T{i:02d}" for i in range(n)], "close": [50.0] * n,
                       "log_dv_21": [20.0 - i * 0.01 for i in range(n)], "log_cap": [25.0] * n, "comp": [0.0] * n,
                       "comp_fn": [float(i) for i in range(n)], "vol_63": [0.25] * n, "sector": ["a", "b", "c", "d"] * 10,
                       "mkt_trend_200": [0.02] * n})
    w = C.fresh_targets(xs, top=40)["top20_fn"]
    assert len(w) == 20 and sum(w.values()) == pytest.approx(1.0)                 # risk-on, equal vols → fully invested
    assert set(w) == {f"T{i:02d}" for i in range(20, 40)}                        # the 20 best scores (5 per sector fits)
    assert max(w.values()) <= 0.08 + 1e-9
    off = C.fresh_targets(xs.with_columns(mkt_trend_200=pl.lit(-0.05)), top=40)["top20_fn"]
    assert sum(off.values()) == pytest.approx(0.5)                               # below the 200-day trend → half gross
    assert C.fresh_targets(xs.drop("comp_fn"), top=40)["top20_fn"] == {}


def test_neutral_labels_remove_the_factor_part():
    import numpy as np
    rng = np.random.default_rng(0)
    n = 300
    beta = rng.normal(size=n)
    spec_ret = rng.normal(size=n) * 0.02
    df = pl.DataFrame({"date": [date(2026, 1, 5)] * n, "ticker": [f"T{i}" for i in range(n)], "sector": ["a", "b", "c"] * 100,
                       "beta_63": beta, "vol_63": rng.normal(size=n), "log_mcap": rng.normal(size=n), "mom_12_1": rng.normal(size=n),
                       "ret_21": rng.normal(size=n), "fwd_21": 0.10 * beta + spec_ret, "fwd_63": 0.10 * beta + spec_ret})
    out = C.neutral_labels(df)
    y = out["yn_21"].to_numpy()
    assert abs(np.corrcoef(y, beta)[0, 1]) < 0.05                                # the beta bet is gone
    assert np.corrcoef(y, spec_ret)[0, 1] > 0.9                                  # the stock-specific part remains

