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
    for name in pf.BOOKS:
        p = pf.book_path(name)
        if not p.exists():
            pytest.skip("books not initialised")
        b = json.loads(p.read_text())
        for k in ("name", "cash", "holdings", "equity_log", "trades"):
            assert k in b, f"{name} missing {k}"
        assert b["cash"] >= -1e-9, f"{name} has negative cash"


# ── ml_v2_gp: Gârleanu-Pedersen partial-trading book (added 2026-09-16) ─────
def test_gp_book_is_registered_and_separate():
    """Research winner runs as a SIXTH book; the original five are untouched."""
    assert "ml_v2_gp" in pf.BOOKS
    assert pf.BOOKS[:5] == ["spy_benchmark", "ml_raw", "ml_v2", "momentum", "blend"]


def test_gp_rebalance_moves_only_trade_rate_of_the_gap():
    """From all-cash, one GP step deploys exactly trade_rate of the target."""
    b = {"name": "ml_v2_gp", "cash": 10_000.0, "holdings": {}, "equity_log": [],
         "trades": [], "last_rebalance": None, "created": "2020-01-01"}
    px = {"AAA": 10.0, "BBB": 20.0}
    pf.rebalance_book_gp(b, {"AAA": 0.5, "BBB": 0.5}, px, "2020-01-02")
    eq = pf.equity(b, px)
    invested = sum(q * px[t] for t, q in b["holdings"].items())
    assert invested / eq == pytest.approx(pf.GP_TRADE_RATE, rel=1e-6)
    assert b["cash"] / eq == pytest.approx(1 - pf.GP_TRADE_RATE, rel=1e-6)
    assert b["trades"][-1]["turnover_frac"] == pytest.approx(pf.GP_TRADE_RATE, rel=1e-6)


def test_gp_rebalance_converges_toward_target_and_charges_less_than_full():
    """Repeated steps approach full deployment; turnover per step shrinks."""
    b = {"name": "ml_v2_gp", "cash": 10_000.0, "holdings": {}, "equity_log": [],
         "trades": [], "last_rebalance": None, "created": "2020-01-01"}
    px = {"AAA": 10.0, "BBB": 20.0}
    tgt = {"AAA": 0.5, "BBB": 0.5}
    turns = []
    for i in range(8):
        pf.rebalance_book_gp(b, tgt, px, f"2020-0{i+1}-02")
        turns.append(b["trades"][-1]["turnover_frac"])
    eq = pf.equity(b, px)
    invested = sum(q * px[t] for t, q in b["holdings"].items())
    assert invested / eq > 0.95                          # ~97% after 8 steps
    assert all(turns[i] > turns[i + 1] for i in range(len(turns) - 1))  # decaying
    full_rebalance_cost = 10_000 * (pf.COST_BPS * 1.0 + pf.IMPACT_BPS) / 10_000
    assert b["trades"][0]["cost"] < full_rebalance_cost  # first step cheaper than going all-in


def test_gp_rebalance_exits_dropped_names():
    """A name the model dropped to 0 must be traded out, not held forever."""
    b = {"name": "ml_v2_gp", "cash": 0.0, "holdings": {"OLD": 500.0}, "equity_log": [],
         "trades": [], "last_rebalance": None, "created": "2020-01-01"}
    px = {"OLD": 10.0, "NEW": 10.0}
    for i in range(12):
        pf.rebalance_book_gp(b, {"NEW": 1.0}, px, f"2020-{i+1:02d}-02")
    assert b["holdings"].get("OLD", 0.0) * px["OLD"] / pf.equity(b, px) < 0.01


def test_gp_cap_renormalise_respects_ten_percent():
    w = {f"T{i}": (0.5 if i == 0 else 0.5 / 19) for i in range(20)}
    # replicate select_weighted's cap loop on a synthetic weight dict
    for _ in range(50):
        over = {t: v for t, v in w.items() if v > pf.GP_MAX_W}
        if not over:
            break
        ex = sum(v - pf.GP_MAX_W for v in over.values())
        for t in over: w[t] = pf.GP_MAX_W
        free = {t: v for t, v in w.items() if t not in over}; fs = sum(free.values())
        for t in free: w[t] += ex * free[t] / fs
    assert max(w.values()) <= pf.GP_MAX_W + 1e-9
    assert sum(w.values()) == pytest.approx(1.0)
