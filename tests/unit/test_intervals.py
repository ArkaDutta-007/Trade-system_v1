"""Conformalized quantile interval models — coverage, monotonicity, bounds."""
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from trading_system.models.intervals import (
    train_interval_models, bounds_for_ticker, IntervalBundle,
)


def _synthetic_features(n_days=400, n_tickers=6, seed=0):
    rng = np.random.default_rng(seed)
    start = date(2020, 1, 1)
    rows = []
    for ti in range(n_tickers):
        price = 100.0
        feat = rng.normal(0, 1)
        for di in range(n_days):
            d = start + timedelta(days=di)
            # a feature that genuinely predicts next-day drift + noise
            feat = 0.9 * feat + 0.1 * rng.normal()
            ret = 0.0008 * feat + rng.normal(0, 0.02)
            price *= (1 + ret)
            rows.append({"date": d, "ticker": f"T{ti}", "adj_close": price, "signal": feat})
    return pl.DataFrame(rows)


@pytest.fixture(scope="module")
def bundle_and_feats():
    feats = _synthetic_features()
    bundle = train_interval_models(
        feats, ["signal"], horizons=(5, 21), quantiles=(0.1, 0.5, 0.9),
        alpha=0.2, cal_fraction=0.25,
    )
    return bundle, feats


def test_coverage_meets_target(bundle_and_feats):
    bundle, _ = bundle_and_feats
    # conformal guarantees ~ (1 - alpha) = 80% coverage on calibration
    for h in bundle.horizons:
        assert bundle.coverage[h] >= 0.75


def test_quantiles_are_monotone(bundle_and_feats):
    bundle, feats = bundle_and_feats
    row = feats.tail(1).select(bundle.feature_columns).to_numpy()[0]
    dist = bundle.predict_row(row)
    for h, r in dist.items():
        assert r["q10"] <= r["median"] <= r["q90"]
        assert r["lo"] <= r["median"] <= r["hi"]          # conformal-widened envelope
        assert r["lo"] <= r["q10"] and r["hi"] >= r["q90"]  # conformal widens, never narrows


def test_bounds_widen_with_horizon(bundle_and_feats):
    bundle, feats = bundle_and_feats
    out = bounds_for_ticker(bundle, feats, "T0", last_price=100.0)
    assert out is not None
    hz = out["horizons"]
    width_5d = hz["5d"]["price"]["hi"] - hz["5d"]["price"]["lo"]
    width_1m = hz["1m"]["price"]["hi"] - hz["1m"]["price"]["lo"]
    assert width_1m > width_5d  # uncertainty grows with horizon


def test_save_load_roundtrip(bundle_and_feats, tmp_path):
    bundle, feats = bundle_and_feats
    bundle.save(tmp_path / "intervals")
    from trading_system.models.intervals import load_interval_bundle
    loaded = load_interval_bundle(tmp_path)
    assert isinstance(loaded, IntervalBundle)
    assert loaded.horizons == bundle.horizons
