"""Distribution-based sizing from calibrated bounds."""
from trading_system.portfolio.sizing import distribution_sized_weights


def _bounds(median, lo, hi):
    return {"horizons": {"1m": {"return": {"median": median, "lo": lo, "hi": hi}}}}


def test_positive_edge_gets_weight_negative_does_not():
    bt = {
        "GOOD": _bounds(0.06, -0.04, 0.16),   # strong reward-to-downside
        "BAD": _bounds(-0.02, -0.10, 0.05),   # negative median -> excluded
    }
    w = distribution_sized_weights(bt, horizon="1m")
    assert "GOOD" in w and "BAD" not in w
    assert w["GOOD"] > 0


def test_better_reward_to_downside_gets_more_weight():
    bt = {
        "A": _bounds(0.06, -0.03, 0.15),   # edge 6%, downside 3% -> ratio 2.0
        "B": _bounds(0.06, -0.12, 0.20),   # edge 6%, downside 12% -> ratio 0.5
    }
    w = distribution_sized_weights(bt, horizon="1m", max_weight=1.0)
    assert w["A"] > w["B"]


def test_per_name_cap_and_gross_bound():
    bt = {f"T{i}": _bounds(0.10, -0.02, 0.25) for i in range(20)}
    w = distribution_sized_weights(bt, horizon="1m", max_weight=0.10)
    assert all(v <= 0.10 + 1e-9 for v in w.values())
    assert sum(w.values()) <= 1.0 + 1e-9


def test_empty_when_no_horizon():
    assert distribution_sized_weights({"X": {"horizons": {}}}, horizon="1m") == {}
