"""Invest-planner pure helpers: hold-horizon choice + reliability weighting."""
from __future__ import annotations

import pytest

from trading_system.decision.invest import choose_hold_horizon, horizon_reliability


def _band(days, lo, med, hi):
    return {"days": days, "return": {"lo": lo, "median": med, "hi": hi},
            "price": {"lo": 100 * (1 + lo), "median": 100 * (1 + med),
                      "hi": 100 * (1 + hi)}}


class TestHorizonReliability:
    def test_leak_fail_haircut(self):
        good = horizon_reliability({"icir": 1.0, "leak_pass": True})
        bad = horizon_reliability({"icir": 1.0, "leak_pass": False})
        assert bad == pytest.approx(good * 0.35)

    def test_negative_icir_floored(self):
        assert horizon_reliability({"icir": -0.5, "leak_pass": True}) == pytest.approx(0.05)


class TestChooseHoldHorizon:
    def test_prefers_reliable_horizon(self):
        bands = {
            "1m": _band(21, -0.05, 0.02, 0.08),
            "12m": _band(252, -0.15, 0.20, 0.60),
        }
        # identical raw quality ordering aside, the 12m model is far more trusted
        rel = {21: 0.05, 252: 1.2}
        days, info = choose_hold_horizon(bands, rel)
        assert days == 252
        assert info["label"] == "12m"
        assert info["annualized_edge"] == pytest.approx(0.20, abs=1e-6)

    def test_no_positive_edge_returns_none(self):
        bands = {"1m": _band(21, -0.08, -0.01, 0.03)}
        assert choose_hold_horizon(bands, {21: 1.0}) is None

    def test_short_horizon_wins_when_long_model_is_junk(self):
        bands = {
            "1m": _band(21, -0.03, 0.03, 0.07),   # strong monthly edge
            "12m": _band(252, -0.30, 0.05, 0.40),  # weak yearly edge
        }
        rel = {21: 1.0, 252: 0.05}
        days, _ = choose_hold_horizon(bands, rel)
        assert days == 21

    def test_annualization_math(self):
        # 3m: 6% edge over 63d → ~24% annualized
        bands = {"3m": _band(63, -0.10, 0.06, 0.20)}
        days, info = choose_hold_horizon(bands, {63: 1.0})
        assert days == 63
        assert info["annualized_edge"] == pytest.approx(0.06 * 252 / 63, abs=1e-6)
