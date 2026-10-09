"""Invariant tests for the pick gating/allocation and the paper books."""
import json
import numpy as np
import polars as pl
import pytest

import picks_v2 as pv
import portfolios as pf
from mathlib import effective_n


# ── gates ───────────────────────────────────────────────────────────────────
def _frame(rows):
    return pl.DataFrame(rows)


def test_gates_remove_cheap_illiquid_and_wild_names():
    df = _frame([
        {"ticker": "GOOD", "score": 0.5, "close": 100.0,
         "avg_dollar_volume_20": 5e8, "vol_20d": 0.30},
        {"ticker": "CHEAP", "score": 9.0, "close": 1.01,
         "avg_dollar_volume_20": 5e8, "vol_20d": 0.30},
        {"ticker": "THIN", "score": 9.0, "close": 100.0,
         "avg_dollar_volume_20": 9e4, "vol_20d": 0.30},
        {"ticker": "WILD", "score": 9.0, "close": 100.0,
         "avg_dollar_volume_20": 5e8, "vol_20d": 3.0},
    ])
    kept, rej = pv.apply_gates(df)
    assert kept["ticker"].to_list() == ["GOOD"]
    assert rej["price<$5"] == 1 and rej["illiquid<$20M/d"] == 1


def test_gates_clip_absurd_scores_without_dropping_them():
    rows = [{"ticker": f"T{i}", "score": 0.10 + 0.001 * i, "close": 50.0,
             "avg_dollar_volume_20": 5e8, "vol_20d": 0.3} for i in range(40)]
    rows.append({"ticker": "GLITCH", "score": 500.0, "close": 50.0,
                 "avg_dollar_volume_20": 5e8, "vol_20d": 0.3})
    kept, rej = pv.apply_gates(_frame(rows))
    assert "GLITCH" in kept["ticker"].to_list()          # kept, not dropped
    g = kept.filter(pl.col("ticker") == "GLITCH")
    assert g["score_clipped"][0] < 500.0                 # but clipped
    assert rej["score_clipped"] >= 1


# ── risk adjustment ─────────────────────────────────────────────────────────
def test_risk_adjust_prefers_the_better_risk_reward_at_equal_score():
    df = _frame([
        {"ticker": "CALM", "score_clipped": 1.0, "vol_20d": 0.20},
        {"ticker": "WILD", "score_clipped": 1.0, "vol_20d": 0.90},
    ])
    out = pv.risk_adjust(df, ic=0.10)
    assert out["ticker"][0] == "CALM"


def test_risk_adjust_is_demeaned_so_a_constant_score_is_not_a_lowvol_screen():
    """Regression: adding the cross-sectional mean back before dividing by vol
    turned the ranking into a pure 1/vol screen and discarded the model."""
    df = _frame([{"ticker": f"T{i}", "score_clipped": 1.0,
                  "vol_20d": 0.2 + 0.05 * i} for i in range(6)])
    out = pv.risk_adjust(df, ic=0.10)
    assert np.allclose(out["score_ra"].to_numpy(), 0.0, atol=1e-12)


# ── allocation ──────────────────────────────────────────────────────────────
def _alloc_frame(n=8, seed=0):
    rng = np.random.default_rng(seed)
    return _frame([{"ticker": f"T{i}", "score_ra": float(rng.normal()),
                    "vol_20d": 0.3} for i in range(n)])


def test_weights_sum_to_one_and_respect_the_cap():
    picks = _alloc_frame()
    n = picks.height
    base = np.full(n, 1.0 / n)
    sr = picks["score_ra"].to_numpy()
    z = np.clip((sr - sr.mean()) / (sr.std(ddof=1) or 1), -3, 3)
    w = base * np.exp(pv.CONVICTION_TILT * z)
    w /= w.sum()
    for _ in range(50):
        over = w > pv.MAX_WEIGHT
        if not over.any():
            break
        ex = (w[over] - pv.MAX_WEIGHT).sum()
        w[over] = pv.MAX_WEIGHT
        free = ~over
        w[free] += ex * w[free] / w[free].sum()
    assert np.isclose(w.sum(), 1.0)
    assert w.max() <= pv.MAX_WEIGHT + 1e-9


def test_effective_n_beats_naive_concentration():
    """The allocator must not be worse-diversified than a couple of names."""
    w = np.array([0.20, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.03, 0.01])
    assert effective_n(w) > 5.0


def test_theme_diversification_caps_are_enforced():
    meta = {f"T{i}": {"sector": "Technology"} for i in range(10)}
    df = _frame([{"ticker": f"T{i}", "score_ra": 1.0 - i * 0.01} for i in range(10)])
    out = pv.diversify(df, top_n=10, meta=meta)
    assert out.height <= pv.MAX_PER_SECTOR      # one sector => sector cap binds


# ── paper books ─────────────────────────────────────────────────────────────
def test_book_equity_is_cash_plus_marked_holdings():
    b = {"cash": 100.0, "holdings": {"AAA": 2.0, "BBB": 3.0}}
    assert pf.equity(b, {"AAA": 10.0, "BBB": 5.0}) == pytest.approx(135.0)


def test_book_equity_ignores_unpriced_names_rather_than_crashing():
    b = {"cash": 50.0, "holdings": {"AAA": 1.0, "GONE": 5.0}}
    assert pf.equity(b, {"AAA": 10.0}) == pytest.approx(60.0)


def test_rebalance_is_fully_invested_and_costs_reduce_equity():
    b = {"name": "t", "cash": 1000.0, "holdings": {}, "equity_log": [],
         "trades": [], "last_rebalance": None, "created": "2020-01-01"}
    px = {"AAA": 10.0, "BBB": 20.0}
    pf.rebalance_book(b, ["AAA", "BBB"], px, "2020-01-02")
    eq = pf.equity(b, px)
    assert eq < 1000.0                       # costs were charged
    assert eq > 990.0                        # but they are small
    assert b["cash"] == 0.0                  # fully invested
    assert set(b["holdings"]) == {"AAA", "BBB"}


def test_rebalance_skips_unpriced_targets():
    b = {"name": "t", "cash": 1000.0, "holdings": {}, "equity_log": [],
         "trades": [], "last_rebalance": None, "created": "2020-01-01"}
    pf.rebalance_book(b, ["AAA", "NOPRICE"], {"AAA": 10.0}, "2020-01-02")
    assert set(b["holdings"]) == {"AAA"}


def test_metrics_are_sane_on_a_known_curve():
    log = [{"date": f"d{i}", "equity": 10000.0 * (1.001 ** i)} for i in range(253)]
    m = pf.metrics(log)
    assert m["total_ret"] == pytest.approx(1.001 ** 252 - 1, rel=1e-6)
    assert m["maxdd"] == pytest.approx(0.0, abs=1e-12)   # monotonic => no dd
    assert m["sharpe"] > 5                               # noiseless uptrend


def test_metrics_detects_drawdown():
    eq = [100, 120, 60, 90]
    log = [{"date": f"d{i}", "equity": float(v)} for i, v in enumerate(eq)]
    assert pf.metrics(log)["maxdd"] == pytest.approx(-0.5)


def test_metrics_needs_two_points():
    assert pf.metrics([{"date": "d0", "equity": 100.0}]) == {}


def test_all_books_have_distinct_selection_rules():
    assert len(set(pf.BOOKS)) == len(pf.BOOKS)
    assert "spy_benchmark" in pf.BOOKS          # the bar must always exist


def test_persisted_books_are_valid_json_with_required_keys():
    for name in pf.BOOKS + list(pf.RETIRED):
        p = pf.book_path(name)
        if not p.exists():
            pytest.skip("books not initialised")
        b = json.loads(p.read_text())
        for k in ("name", "cash", "holdings", "equity_log", "trades"):
            assert k in b, f"{name} missing {k}"
        assert b["cash"] >= -1e-9, f"{name} has negative cash"


# ── book set after the legacy retirement (2026-10-09) ───────────────────────
def test_legacy_books_are_retired_not_deleted():
    """The four legacy-model books leave the active set but keep their history (shown frozen)."""
    assert pf.BOOKS == ["spy_benchmark", "momentum", "alpha_v2"]
    assert set(pf.RETIRED) == {"ml_raw", "ml_v2", "blend", "ml_v2_gp"}
    assert not set(pf.RETIRED) & set(pf.BOOKS)
    for name in pf.RETIRED:
        with pytest.raises(ValueError):
            pf.select(name) if name != "ml_v2_gp" else pf.select_weighted(name)


def test_momentum_rank_uses_the_legacy_feature_definitions():
    """120-row momentum on adj_close, 20-row mean of close×volume ≥ $20M, close ≥ $5 — from prices."""
    import datetime as dt
    days = [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(130)]
    rows = []
    for t, growth, vol, close0 in (("UP", 1.004, 1e6, 50.0), ("FLAT", 1.0, 1e6, 50.0),
                                   ("THIN", 1.01, 1e3, 50.0), ("PENNY", 1.01, 1e9, 1.0)):
        for i, d in enumerate(days):
            c = close0 * growth ** i
            rows.append({"date": d, "ticker": t, "close": c, "adj_close": c, "volume": vol})
    r = pf.momentum_rank(pl.DataFrame(rows))
    assert r["ticker"].to_list() == ["UP", "FLAT"]                  # THIN fails $volume, PENNY fails price
    assert r["mom"][0] == pytest.approx(1.004 ** 120 - 1, rel=1e-9)


def _book(name, created, log, holdings=None):
    return {"name": name, "created": created, "cash": 0.0, "holdings": holdings or {}, "trades": [],
            "last_rebalance": created, "equity_log": [{"date": d, "equity": e} for d, e in log]}


def test_retired_books_are_frozen_and_reported_separately(tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "BOOKS_DIR", tmp_path)
    pf.save_book(_book("spy_benchmark", "2026-09-04", [("2026-09-04", 10000), ("2026-09-18", 10100), ("2026-10-08", 10200)],
                       {"SPY": 10000 / 100}))
    a = _book("alpha_v2", "2026-09-18", [("2026-09-18", 10000), ("2026-10-08", 10000)])
    a["cash"] = 10000.0
    pf.save_book(a)
    frozen = _book("ml_v2", "2026-09-04", [("2026-09-04", 10000), ("2026-10-08", 9300)], {"AAA": 93.0})
    pf.save_book(frozen)
    px = pl.DataFrame({"date": [__import__("datetime").date(2026, 10, 9)] * 2, "ticker": ["SPY", "AAA"],
                       "adj_close": [103.0, 500.0]})
    pf.do_mark(px)
    assert pf.load_book("ml_v2") == json.loads(json.dumps(frozen))          # untouched by the daily mark
    assert pf.load_book("spy_benchmark")["equity_log"][-1] == {"date": "2026-10-09", "equity": 10300.0}
    rep = pf.do_report()
    assert "Retired with the legacy model" in rep and "ml_v2" in rep.split("Retired")[1]
    line = next(x for x in rep.splitlines() if x.startswith("alpha_v2"))
    assert "-2.0%" in line   # vs SPY over ITS OWN dates (09-18 → 10-09: SPY +2.0%); since 09-04 SPY made +3.0%


def test_gp_rebalance_moves_only_trade_rate_of_the_gap():
    """An invested book moves exactly trade_rate of the way to a new target in one step."""
    b = {"name": "ml_v2_gp", "cash": 0.0, "holdings": {"AAA": 1000.0}, "equity_log": [],
         "trades": [], "last_rebalance": None, "created": "2020-01-01"}
    px = {"AAA": 10.0, "BBB": 20.0}
    pf.rebalance_book_gp(b, {"AAA": 0.5, "BBB": 0.5}, px, "2020-01-02")
    eq = pf.equity(b, px)
    assert b["holdings"]["BBB"] * px["BBB"] / eq == pytest.approx(pf.GP_TRADE_RATE * 0.5, rel=1e-3)
    assert b["holdings"]["AAA"] * px["AAA"] / eq == pytest.approx(1 - pf.GP_TRADE_RATE * 0.5, rel=1e-3)
    assert b["trades"][-1]["turnover_frac"] == pytest.approx(pf.GP_TRADE_RATE, rel=1e-6)


def test_gp_seeds_fully_then_trades_partially_toward_a_new_target():
    """From cash the book goes straight to target (2026-10-08 fix: the 35%/month ramp left alpha_v2 34%
    invested after two months); afterwards each step closes 35% of the gap and turnover decays."""
    b = {"name": "ml_v2_gp", "cash": 10_000.0, "holdings": {}, "equity_log": [],
         "trades": [], "last_rebalance": None, "created": "2020-01-01"}
    px = {"AAA": 10.0, "BBB": 20.0, "CCC": 5.0}
    pf.rebalance_book_gp(b, {"AAA": 0.5, "BBB": 0.5}, px, "2020-01-02")
    eq = pf.equity(b, px)
    assert sum(q * px[t] for t, q in b["holdings"].items()) / eq > 0.99      # fully deployed at seeding
    turns = []
    for i in range(6):                                                    # the model switches BBB → CCC
        pf.rebalance_book_gp(b, {"AAA": 0.5, "CCC": 0.5}, px, f"2020-0{i+2}-02")
        turns.append(b["trades"][-1]["turnover_frac"])
    w_ccc = b["holdings"].get("CCC", 0) * px["CCC"] / pf.equity(b, px)
    assert 0.40 < w_ccc <= 0.5 and turns[0] < 1.0 * 0.5 + 1e-9               # partial, not all at once
    assert all(turns[i] > turns[i + 1] for i in range(len(turns) - 1))       # decaying
    for i in range(6):                                                    # 0.5·0.65^n falls below 0.5% at n≈11
        pf.rebalance_book_gp(b, {"AAA": 0.5, "CCC": 0.5}, px, f"2021-0{i+1}-02")
    assert "BBB" not in b["holdings"]                                      # the leftover is sold, not kept as dust


def test_gp_rebalance_exits_dropped_names():
    """A name the model dropped to 0 must be traded out, not held forever."""
    b = {"name": "ml_v2_gp", "cash": 0.0, "holdings": {"OLD": 500.0}, "equity_log": [],
         "trades": [], "last_rebalance": None, "created": "2020-01-01"}
    px = {"OLD": 10.0, "NEW": 10.0}
    for i in range(12):
        pf.rebalance_book_gp(b, {"NEW": 1.0}, px, f"2020-{i+1:02d}-02")
    assert b["holdings"].get("OLD", 0.0) * px["OLD"] / pf.equity(b, px) < 0.01
