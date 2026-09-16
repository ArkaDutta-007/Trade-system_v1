"""Properties of the cost model and the execution policies."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from trading_system.research.costs import (
    RealisticCostModel, abdi_ranaldo_spread, corwin_schultz_spread,
    estimate_spread_panel,
)
from trading_system.research.execution import (
    GarleanuPedersenPolicy, apply_no_trade_bands, turnover_of,
)


def _synthetic_bars(n=400, spread=0.002, vol=0.02, seed=0):
    """Daily bars with a known embedded spread."""
    rng = np.random.default_rng(seed)
    px = 100 * np.exp(np.cumsum(rng.normal(0, vol, n)))
    rng2 = np.random.default_rng(seed + 1)
    hi = px * (1 + np.abs(rng2.normal(0, vol / 2, n)) + spread / 2)
    lo = px * (1 - np.abs(rng2.normal(0, vol / 2, n)) - spread / 2)
    close = np.clip(px * (1 + rng2.normal(0, spread, n)), lo, hi)
    return close, hi, lo


class TestSpreadEstimators:
    def test_corwin_schultz_recovers_the_right_order_of_magnitude(self):
        _, hi, lo = _synthetic_bars(spread=0.002)
        s = corwin_schultz_spread(hi, lo)
        assert np.isnan(s[0]), "first observation cannot be estimated"
        assert 5e-4 < np.nanmean(s) < 8e-3

    def test_abdi_ranaldo_is_non_negative_and_finite(self):
        c, hi, lo = _synthetic_bars()
        s = abdi_ranaldo_spread(c, hi, lo)
        finite = s[np.isfinite(s)]
        assert len(finite) > 100
        assert (finite >= 0).all()

    def test_wider_bars_give_a_wider_spread(self):
        c1, h1, l1 = _synthetic_bars(spread=0.0005, seed=3)
        c2, h2, l2 = _synthetic_bars(spread=0.01, seed=3)
        s1 = np.nanmedian(abdi_ranaldo_spread(c1, h1, l1))
        s2 = np.nanmedian(abdi_ranaldo_spread(c2, h2, l2))
        assert s2 > s1

    def test_panel_estimate_is_bounded_and_complete(self):
        """No NaN may survive: the simulator would silently floor it.

        Both null (warm-up rows) and NaN (a window whose estimate was all-NaN)
        occur in practice and Polars treats them as different values, so this
        checks the finished column rather than the estimator.
        """
        import datetime as dt
        n = 200
        rows = []
        for si, t in enumerate(("A", "B")):
            c, h, l = _synthetic_bars(n=n, seed=si)
            for i in range(n):
                rows.append({"date": dt.date(2020, 1, 1) + dt.timedelta(days=i),
                             "ticker": t, "close": c[i], "high": h[i], "low": l[i]})
        df = pl.DataFrame(rows)
        out = estimate_spread_panel(df, floor_bps=1.0, cap_bps=500.0)
        assert out.height == df.height
        v = out["spread_bps"].to_numpy()
        assert np.isfinite(v).all(), "NaN or inf leaked into the spread panel"
        assert (v >= 1.0).all() and (v <= 500.0).all()

    def test_a_name_with_no_estimable_spread_falls_back_to_the_floor(self):
        import datetime as dt
        rows = [{"date": dt.date(2020, 1, 1) + dt.timedelta(days=i), "ticker": "FLAT",
                 "close": 10.0, "high": 10.0, "low": 10.0} for i in range(60)]
        out = estimate_spread_panel(pl.DataFrame(rows), floor_bps=2.0)
        assert out["spread_bps"].to_numpy() == pytest.approx(2.0)


class TestRealisticCostModel:
    def test_illiquid_costs_more_per_dollar_than_liquid(self):
        cm = RealisticCostModel()
        q = np.array([100_000.0])
        liquid = cm.cost_notional(q, np.array([5e7]), np.array([5.0]), np.array([0.015]))
        thin = cm.cost_notional(q, np.array([2e6]), np.array([80.0]), np.array([0.05]))
        assert thin[0] > 3 * liquid[0]

    def test_impact_is_concave_in_size(self):
        """Doubling the order should less than double the *per-dollar* impact."""
        cm = RealisticCostModel(commission_bps=0.0, spread_mult=0.0)
        adv, spr, vol = np.array([1e7]), np.array([10.0]), np.array([0.02])
        c1 = cm.cost_notional(np.array([1e5]), adv, spr, vol)[0] / 1e5
        c2 = cm.cost_notional(np.array([2e5]), adv, spr, vol)[0] / 2e5
        assert c1 < c2 < c1 * 2

    def test_participation_cap_clips_and_preserves_sign(self):
        cm = RealisticCostModel(max_participation=0.05)
        adv = np.array([1e6, 1e6])
        filled = cm.fillable_notional(np.array([1e6, -1e6]), adv)
        assert filled[0] == pytest.approx(5e4)
        assert filled[1] == pytest.approx(-5e4)

    def test_small_orders_are_not_clipped(self):
        cm = RealisticCostModel(max_participation=0.05)
        want = np.array([1000.0])
        assert cm.fillable_notional(want, np.array([1e6]))[0] == pytest.approx(1000.0)

    def test_scaled_multiplies_every_rate(self):
        cm = RealisticCostModel(commission_bps=1.0, spread_mult=0.5, impact_eta=0.5)
        s = cm.scaled(3.0)
        assert (s.commission_bps, s.spread_mult, s.impact_eta) == (3.0, 1.5, 1.5)
        assert s.max_participation == cm.max_participation, "capacity is not a cost"

    def test_cost_is_monotone_in_the_multiplier(self):
        cm = RealisticCostModel()
        args = (np.array([1e5]), np.array([1e7]), np.array([20.0]), np.array([0.02]))
        assert cm.scaled(3.0).cost_notional(*args)[0] > cm.cost_notional(*args)[0]


class TestNoTradeBands:
    def test_small_drift_is_left_alone(self):
        target = np.array([0.20, 0.20])
        current = np.array([0.19, 0.20])
        out = apply_no_trade_bands(target, current, entry_hurdle=0.3, hold_hurdle=0.15)
        assert out[0] == pytest.approx(0.19), "1% drift on a 20% target is inside the band"

    def test_large_gap_is_traded(self):
        out = apply_no_trade_bands(np.array([0.20]), np.array([0.05]), 0.3, 0.15)
        assert out[0] == pytest.approx(0.20)

    def test_exits_are_never_banded(self):
        """A held name the model has dropped must be sold, not held by the band."""
        out = apply_no_trade_bands(np.array([0.0]), np.array([0.04]), 0.3, 0.15)
        assert out[0] == pytest.approx(0.0)

    def test_bands_reduce_turnover(self):
        rng = np.random.default_rng(0)
        cur = rng.dirichlet(np.ones(20))
        tgt = cur + rng.normal(0, 0.004, 20)
        banded = apply_no_trade_bands(tgt, cur, 0.3, 0.15)
        assert turnover_of(banded, cur) < turnover_of(tgt, cur)


class TestGarleanuPedersen:
    def test_partial_trading_moves_less_than_full(self):
        cur = np.array([0.5, 0.5, 0.0])
        tgt = np.array([0.0, 0.5, 0.5])
        pol = GarleanuPedersenPolicy(trade_rate=0.35, signal_decay=0.3)
        assert turnover_of(pol.step(cur, tgt), cur) < turnover_of(tgt, cur)

    def test_rate_one_with_no_decay_reaches_the_target(self):
        cur = np.array([0.6, 0.4])
        tgt = np.array([0.3, 0.7])
        out = GarleanuPedersenPolicy(trade_rate=1.0, signal_decay=0.0).step(cur, tgt)
        assert out == pytest.approx(tgt, abs=1e-9)

    def test_faster_decaying_signals_are_traded_less(self):
        """'Aim in front of the target': a signal that will have moved is under-traded."""
        tgt = np.array([1.0, 0.0])
        slow = GarleanuPedersenPolicy(trade_rate=0.5, signal_decay=0.05).aim(tgt)
        fast = GarleanuPedersenPolicy(trade_rate=0.5, signal_decay=0.8).aim(tgt)
        assert fast[0] < slow[0]

    def test_decay_estimate_separates_noise_from_persistence(self):
        noise = [np.random.default_rng(s).normal(size=60) for s in range(6)]
        persistent = [np.arange(60.0) + np.random.default_rng(s).normal(0, 1.5, 60)
                      for s in range(6)]
        assert GarleanuPedersenPolicy.fit_decay(noise) > 0.7
        assert GarleanuPedersenPolicy.fit_decay(persistent) < 0.2

    def test_weights_stay_normalised(self):
        cur = np.array([0.4, 0.4, 0.2])
        tgt = np.array([0.1, 0.1, 0.8])
        out = GarleanuPedersenPolicy(0.4, 0.3).step(cur, tgt)
        assert out.sum() == pytest.approx(1.0)
