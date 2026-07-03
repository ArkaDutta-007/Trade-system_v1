"""Meta-labeling — the classifier must find a planted forecastability regime.

Synthetic setup: a state feature `regime_q` controls whether the primary
forecast is right (80% hit when high, 45% when low). A working meta layer
must (1) achieve AUC well above 0.5 on purged OOS folds, (2) produce
calibrated probabilities that are monotone in the planted regime, and
(3) score today's cross-section through the same feature path.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from trading_system.models.meta import (
    build_meta_dataset,
    meta_probabilities,
    resolve_meta_features,
    train_meta_for_horizon,
)


def _panel_and_oos(n_tickers=40, n_days=360, seed=5):
    """Gold-panel slice + pseudo-ledger with a planted regime effect."""
    rng = np.random.default_rng(seed)
    d0 = dt.date(2022, 1, 3)
    dates = []
    d = d0
    while len(dates) < n_days:
        if d.weekday() < 5:
            dates.append(d)
        d += dt.timedelta(days=1)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]

    feat_rows, oos_rows = [], []
    for tk in tickers:
        regime = rng.uniform(0, 1, n_days)          # the forecastability state
        noise = rng.normal(0, 1, n_days)
        for j, day in enumerate(dates):
            y_pred = rng.normal(0, 0.05)
            p_hit = 0.80 if regime[j] > 0.5 else 0.45
            hit = rng.uniform() < p_hit
            y_true = abs(rng.normal(0.02, 0.01)) * (1 if hit else -1) * np.sign(y_pred)
            feat_rows.append({
                "ticker": tk, "date": day,
                "hurst_dfa_120": regime[j],          # planted signal
                "vol_20d": abs(noise[j]) * 0.2 + 0.1,  # noise state
                "mom_20d": rng.normal(0, 0.05),
                "adj_close": 100.0,
            })
            oos_rows.append({"ticker": tk, "date": day,
                             "y_pred": y_pred, "y_true": y_true})
    schema = {"date": pl.Date}
    return (pl.DataFrame(feat_rows).with_columns(pl.col("date").cast(pl.Date)),
            pl.DataFrame(oos_rows).with_columns(pl.col("date").cast(pl.Date)))


@pytest.fixture(scope="module")
def data():
    return _panel_and_oos()


class TestResolveAndDataset:
    def test_resolve_picks_state_columns_only(self, data):
        features, _ = data
        cols = resolve_meta_features(features)
        assert "hurst_dfa_120" in cols
        assert "vol_20d" in cols
        assert "mom_20d" not in cols        # alpha feature, not state
        assert "adj_close" not in cols

    def test_dataset_labels_and_columns(self, data):
        features, oos = data
        ds, cols = build_meta_dataset(oos, features)
        assert ds.height == oos.height
        row = ds.row(0, named=True)
        assert row["hit"] == int(np.sign(row["y_pred"]) == np.sign(row["y_true"]))
        for c in ("pred_rank_pct", "pred_z", "pred_sign",
                  "xsec_mom_dispersion", "xsec_breadth"):
            assert c in cols and c in ds.columns
        # rank pct within a date spans (0, 1]
        one_day = ds.filter(pl.col("date") == ds["date"].min())
        assert 0 < one_day["pred_rank_pct"].min() <= one_day["pred_rank_pct"].max() <= 1.0


class TestTrainMeta:
    @pytest.fixture(scope="class")
    def bundle(self, data):
        features, oos = data
        return train_meta_for_horizon(oos, features, horizon=21, n_splits=4)

    def test_finds_the_planted_regime(self, bundle):
        m = bundle["metrics"]
        assert m["auc_mean"] is not None and m["auc_mean"] > 0.60
        assert m["brier_skill"] > 0            # beats the base-rate forecast
        assert m["top_minus_bottom_decile"] > 0.10

    def test_calibration_monotone_in_regime(self, bundle, data):
        features, _ = data
        last = features.filter(pl.col("date") == features["date"].max())
        scores = {t: 0.05 for t in last["ticker"].to_list()}
        probs = meta_probabilities(bundle, last, scores)
        assert set(probs) == set(scores)
        assert all(0.0 <= p <= 1.0 for p in probs.values())
        # planted: high regime_q (hurst col) → higher P(right)
        reg = {r["ticker"]: r["hurst_dfa_120"] for r in last.to_dicts()}
        hi = [probs[t] for t in probs if reg[t] > 0.6]
        lo = [probs[t] for t in probs if reg[t] < 0.4]
        assert np.mean(hi) > np.mean(lo)

    def test_shuffled_labels_have_no_skill(self, data):
        features, oos = data
        rng = np.random.default_rng(0)
        shuffled = oos.with_columns(
            pl.Series("y_true", rng.permutation(oos["y_true"].to_numpy())))
        b = train_meta_for_horizon(shuffled, features, horizon=21, n_splits=4)
        assert abs(b["metrics"]["auc_mean"] - 0.5) < 0.05

    def test_too_small_dataset_raises(self, data):
        features, oos = data
        with pytest.raises(ValueError, match="too small"):
            train_meta_for_horizon(oos.head(200), features, horizon=21)


class TestMetaProbabilities:
    def test_missing_columns_returns_empty(self, data):
        features, oos = data
        bundle = train_meta_for_horizon(oos, features, horizon=21, n_splits=4)
        crippled = features.drop("hurst_dfa_120").filter(
            pl.col("date") == features["date"].max())
        assert meta_probabilities(
            bundle, crippled, {t: 0.1 for t in crippled["ticker"].to_list()}) == {}

    def test_empty_scores(self, data):
        features, oos = data
        bundle = train_meta_for_horizon(oos, features, horizon=21, n_splits=4)
        last = features.filter(pl.col("date") == features["date"].max())
        assert meta_probabilities(bundle, last, {}) == {}
