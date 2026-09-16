"""Properties of the causal walk-forward simulator.

The point of this file is the one thing a backtester must get right: nothing
inside the loop may depend on information from after the decision date.  The
tests are therefore built around an *oracle* model that is allowed to cheat.  If
the simulator is honest, the oracle makes a fortune and a causal model does not;
if the two ever converge, something is leaking.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from trading_system.research.costs import RealisticCostModel
from trading_system.research.scores import generate_causal_scores
from trading_system.research.wfbacktest import (
    WalkForwardConfig, build_panel, default_target_weights, run_walk_forward,
)


# ── synthetic panel ──────────────────────────────────────────────────────────

def make_panel(n_days=1600, n_tickers=40, seed=0):
    """A panel where ``signal`` genuinely predicts the next 21 days' return."""
    rng = np.random.default_rng(seed)
    start = dt.date(2010, 1, 4)
    dates = [start + dt.timedelta(days=i) for i in range(n_days)]
    dates = [d for d in dates if d.weekday() < 5]

    rows = []
    for j in range(n_tickers):
        px, sig = 50.0 + j, 0.0
        for d in dates:
            sig = 0.95 * sig + rng.normal(0, 1)
            drift = 0.0004 * sig                 # the exploitable edge
            px *= np.exp(drift + rng.normal(0, 0.015))
            vol = 1_000_000 * (1 + j)
            rows.append({
                "date": d, "ticker": f"T{j:02d}",
                "open": px, "high": px * 1.01, "low": px * 0.99,
                "close": px, "adj_close": px, "volume": vol,
                "signal": sig, "noise": rng.normal(),
                "vol_20d": 0.015,
            })
    df = pl.DataFrame(rows)
    ohlcv = df.select(["date", "ticker", "open", "high", "low", "close",
                       "adj_close", "volume"])
    feats = df.select(["date", "ticker", "adj_close", "signal", "noise"])
    return ohlcv, feats, ["signal", "noise"]


class OracleAlpha:
    """Cheats: returns the realised forward return. Used to detect leakage."""

    name = "oracle"

    def __init__(self, feats: pl.DataFrame, horizon: int):
        fwd = (feats.sort(["ticker", "date"]).with_columns(
            ((pl.col("adj_close").shift(-horizon).over("ticker")
              / pl.col("adj_close")) - 1).alias("_f")))
        self.lookup = {(r["date"], r["ticker"]): r["_f"]
                       for r in fwd.select(["date", "ticker", "_f"]).iter_rows(named=True)}

    def fit(self, panel, feat_cols, target):
        return None

    def predict(self, today, feat_cols):
        return np.array([self.lookup.get((d, t)) or 0.0
                         for d, t in zip(today["date"].to_list(),
                                         today["ticker"].to_list())])


class SignalAlpha:
    """Honest: uses only the same-day feature, no fitting."""

    name = "signal"

    def fit(self, panel, feat_cols, target):
        return None

    def predict(self, today, feat_cols):
        return today["signal"].to_numpy()


class NoiseAlpha:
    name = "noise"

    def fit(self, panel, feat_cols, target):
        return None

    def predict(self, today, feat_cols):
        return today["noise"].to_numpy()


@pytest.fixture(scope="module")
def synthetic():
    ohlcv, feats, cols = make_panel()
    return ohlcv, feats, cols, build_panel(ohlcv)


def _cfg(**kw):
    base = dict(oos_start=dt.date(2013, 1, 2), rebalance_days=21, retrain_days=252,
                min_train_days=500, horizon=21, top_k=10, max_weight=0.2,
                min_dollar_volume=0.0, min_price=0.0, risk_adjust=False,
                cost=RealisticCostModel(max_participation=1.0))
    base.update(kw)
    return WalkForwardConfig(**base)


class TestLeakage:
    def test_an_oracle_beats_an_honest_model_by_a_wide_margin(self, synthetic):
        """If the simulator leaks, these two converge. They must not."""
        ohlcv, feats, cols, panel = synthetic
        oracle = run_walk_forward(panel, feats, cols,
                                  lambda: OracleAlpha(feats, 21), _cfg(), label="oracle")
        honest = run_walk_forward(panel, feats, cols, SignalAlpha, _cfg(), label="signal")
        from trading_system.research.stats import sharpe
        assert sharpe(oracle.returns()) > 3.0, "the oracle should be extremely profitable"
        assert sharpe(oracle.returns()) > 2 * sharpe(honest.returns())

    def test_a_pure_noise_signal_earns_no_risk_adjusted_edge(self, synthetic):
        ohlcv, feats, cols, panel = synthetic
        from trading_system.research.stats import sharpe
        res = run_walk_forward(panel, feats, cols, NoiseAlpha, _cfg(), label="noise")
        assert abs(sharpe(res.returns())) < 1.5

    def test_a_real_signal_beats_noise(self, synthetic):
        ohlcv, feats, cols, panel = synthetic
        from trading_system.research.stats import sharpe
        sig = run_walk_forward(panel, feats, cols, SignalAlpha, _cfg(), label="s")
        noise = run_walk_forward(panel, feats, cols, NoiseAlpha, _cfg(), label="n")
        assert sharpe(sig.returns()) > sharpe(noise.returns())

    def test_refit_cut_never_reaches_the_decision_date(self, synthetic):
        """Every training cut-off must precede the decision by the full label window."""
        ohlcv, feats, cols, panel = synthetic
        cfg = _cfg(horizon=63)
        res = run_walk_forward(panel, feats, cols, SignalAlpha, cfg, label="x")
        pad = dt.timedelta(days=cfg.purge_calendar_days())
        for r in res.refits:
            assert r["cut"] <= r["date"] - pad

    def test_scores_stage_uses_the_same_purge(self, synthetic):
        ohlcv, feats, cols, panel = synthetic
        cfg = _cfg(horizon=63)
        _, refits = generate_causal_scores(panel, feats, cols, SignalAlpha, cfg)
        pad = dt.timedelta(days=cfg.purge_calendar_days())
        for r in refits:
            assert dt.date.fromisoformat(r["cut"]) <= dt.date.fromisoformat(r["date"]) - pad


class TestAccounting:
    def test_equity_is_positive_and_finite_throughout(self, synthetic):
        ohlcv, feats, cols, panel = synthetic
        res = run_walk_forward(panel, feats, cols, SignalAlpha, _cfg(), label="s")
        eq = res.daily["equity"].to_numpy()
        assert np.isfinite(eq).all() and (eq > 0).all()

    def test_costs_are_charged_and_reduce_the_return(self, synthetic):
        ohlcv, feats, cols, panel = synthetic
        free = run_walk_forward(
            panel, feats, cols, SignalAlpha,
            _cfg(cost=RealisticCostModel(0.0, 0.0, 0.0, max_participation=1.0)),
            label="free")
        paid = run_walk_forward(
            panel, feats, cols, SignalAlpha,
            _cfg(cost=RealisticCostModel(5.0, 1.0, 1.0, max_participation=1.0)),
            label="paid")
        assert paid.daily["cost"].sum() > free.daily["cost"].sum()
        assert paid.daily["equity"][-1] < free.daily["equity"][-1]

    def test_higher_costs_never_help(self, synthetic):
        ohlcv, feats, cols, panel = synthetic
        base = RealisticCostModel(max_participation=1.0)
        eqs = []
        for m in (1.0, 3.0, 6.0):
            r = run_walk_forward(panel, feats, cols, SignalAlpha,
                                 _cfg(cost=base.scaled(m)), label=f"c{m}")
            eqs.append(float(r.daily["equity"][-1]))
        assert eqs[0] >= eqs[1] >= eqs[2]

    def test_bands_and_partial_trading_cut_turnover(self, synthetic):
        ohlcv, feats, cols, panel = synthetic
        full = run_walk_forward(panel, feats, cols, SignalAlpha, _cfg(), label="full")
        band = run_walk_forward(panel, feats, cols, SignalAlpha,
                                _cfg(band_entry=0.4, band_exit=0.2), label="band")
        part = run_walk_forward(panel, feats, cols, SignalAlpha,
                                _cfg(partial_trade_rate=0.3), label="part")
        t_full = full.daily["turnover"].sum()
        assert band.daily["turnover"].sum() < t_full
        assert part.daily["turnover"].sum() < t_full

    def test_participation_cap_produces_shortfall(self, synthetic):
        ohlcv, feats, cols, panel = synthetic
        tight = run_walk_forward(
            panel, feats, cols, SignalAlpha,
            _cfg(cost=RealisticCostModel(max_participation=1e-6),
                 initial_cash=1e9),
            label="tight")
        assert tight.trades["shortfall"].abs().sum() > 0


class TestMissingPrices:
    """A missing bar must never destroy money.

    These cover the bug that made a freshly ingested panel look like a -98%
    drawdown: the final row was a partial trading day with 3 of 362 names
    printed, and zeroing positions for every non-printing name deleted the book.
    """

    @staticmethod
    def _with_partial_last_day(ohlcv: pl.DataFrame) -> pl.DataFrame:
        last = ohlcv["date"].max()
        keep = sorted(ohlcv["ticker"].unique().to_list())[:2]
        return ohlcv.filter((pl.col("date") != last) | pl.col("ticker").is_in(keep))

    def test_build_panel_trims_a_partial_trailing_day(self, synthetic):
        ohlcv, feats, cols, panel = synthetic
        thin = self._with_partial_last_day(ohlcv)
        trimmed = build_panel(thin)
        assert len(trimmed.dates) == len(panel.dates) - 1
        assert trimmed.dates[-1] < panel.dates[-1]

    def test_a_partial_last_day_cannot_wipe_the_book(self, synthetic):
        ohlcv, feats, cols, panel = synthetic
        thin = build_panel(self._with_partial_last_day(ohlcv), trim_sparse_tail=0.0)
        res = run_walk_forward(thin, feats, cols, SignalAlpha, _cfg(), label="thin")
        r = res.returns()
        assert r.min() > -0.5, "no single day may lose half the book on missing bars"

    def test_a_delisted_name_converts_to_cash_rather_than_vanishing(self, synthetic):
        """Truncate one ticker's history; its value must survive as cash."""
        ohlcv, feats, cols, panel = synthetic
        victim = sorted(ohlcv["ticker"].unique().to_list())[0]
        cut = ohlcv["date"].max() - dt.timedelta(days=120)
        truncated = ohlcv.filter(~((pl.col("ticker") == victim) & (pl.col("date") > cut)))

        p_full = build_panel(ohlcv)
        p_cut = build_panel(truncated)
        full = run_walk_forward(p_full, feats, cols, SignalAlpha, _cfg(), label="full")
        cutr = run_walk_forward(p_cut, feats, cols, SignalAlpha, _cfg(), label="cut")

        # losing one of 40 names may change the result, but not catastrophically
        ratio = float(cutr.daily["equity"][-1]) / float(full.daily["equity"][-1])
        assert 0.5 < ratio < 2.0
        assert cutr.returns().min() > -0.5


class TestWeightConstruction:
    def test_weights_respect_the_cap_and_sum_to_the_gross(self):
        cfg = _cfg(top_k=5, max_weight=0.3, gross_exposure=1.0)
        scores = np.array([9.0, 8.0, 7.0, 6.0, 5.0, 1.0, 0.0])
        w = default_target_weights(scores, {"dvol": np.full(7, 0.02)}, cfg)
        assert w.sum() == pytest.approx(1.0)
        assert w.max() <= 0.3 + 1e-9
        assert (w > 0).sum() == 5

    def test_nan_scores_are_never_selected(self):
        cfg = _cfg(top_k=3, max_weight=1.0)
        scores = np.array([np.nan, 1.0, 2.0, 3.0, np.nan])
        w = default_target_weights(scores, {"dvol": np.full(5, 0.02)}, cfg)
        assert w[0] == 0 and w[4] == 0

    def test_risk_adjustment_demeans_before_dividing_by_vol(self):
        """Without the demean this degenerates into a pure low-volatility screen.

        Two names with identical, positive raw scores but different volatility:
        after demeaning both sit at zero, so vol cannot decide between them and
        the ranking must come from the score.
        """
        cfg = _cfg(top_k=1, max_weight=1.0, risk_adjust=True)
        scores = np.array([1.0, 1.0, 3.0])
        dvol = np.array([0.001, 0.05, 0.05])      # name 0 is by far the calmest
        w = default_target_weights(scores, {"dvol": dvol}, cfg)
        assert w[2] == pytest.approx(1.0), "the highest score must win, not the lowest vol"
