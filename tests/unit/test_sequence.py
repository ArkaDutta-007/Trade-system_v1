"""Sequence (RNN/LSTM/GRU) forecasters — windowing causality + estimator surface."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from trading_system.models.forecast_train import _resolve_model_names, TABULAR_MODELS
from trading_system.models.sequence import (
    SEQUENCE_MODELS, build_sequence_tensor, build_sequence_windows, SequenceWindows,
)


def _panel(n_tickers=4, n_days=120, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_tickers):
        for i in range(n_days):
            rows.append((date(2022, 1, 3) + timedelta(days=i), f"T{t}", float(rng.normal()), float(rng.normal())))
    return pl.DataFrame(rows, schema=["date", "ticker", "f0", "f1"], orient="row").sort(["ticker", "date"])


# ── model-name resolution (no torch needed) ──────────────────────────────────

def test_resolve_model_names_default_is_tabular_only():
    tab, seq = _resolve_model_names(None)
    assert tab == list(TABULAR_MODELS)
    assert seq == []


def test_resolve_model_names_splits_and_drops_unknown():
    tab, seq = _resolve_model_names(["lgbm", "lstm", "gru", "bogus"])
    assert tab == ["lgbm"]
    assert set(seq) == {"lstm", "gru"}
    assert all(s in SEQUENCE_MODELS for s in seq)


# ── causal windowing ──────────────────────────────────────────────────────────

def test_sequence_tensor_shape_and_causality():
    df = _panel()
    feat = ["f0", "f1"]
    L = 10
    X = build_sequence_tensor(df, feat, lookback=L)
    assert X.shape == (df.height, L, len(feat))
    # the LAST timestep of each window must equal that row's own features
    rowfeat = df.select(feat).to_numpy().astype("float32")
    assert np.allclose(X[:, -1, :], rowfeat, atol=1e-5)


def test_sequence_tensor_edge_pads_first_rows_per_ticker():
    df = _panel(n_tickers=3, n_days=40)
    X = build_sequence_tensor(df, ["f0", "f1"], lookback=8)
    # first row of each ticker block: all timesteps are the edge-padded same row
    tickers = df["ticker"].to_numpy()
    starts = [0] + [i for i in range(1, len(tickers)) if tickers[i] != tickers[i - 1]]
    for s in starts:
        assert np.allclose(X[s, 0, :], X[s, -1, :])


def test_sequence_windows_do_not_use_future_rows():
    # Build a per-ticker monotone marker; window max must never exceed the row's value.
    rows = []
    for t in range(2):
        for i in range(50):
            rows.append((date(2022, 1, 3) + timedelta(days=i), f"T{t}", float(i)))
    df = pl.DataFrame(rows, schema=["date", "ticker", "marker"], orient="row").sort(["ticker", "date"])
    X = build_sequence_tensor(df, ["marker"], lookback=12)
    row_vals = df["marker"].to_numpy()
    window_max = X[:, :, 0].max(axis=1)
    assert np.all(window_max <= row_vals + 1e-6)  # strictly backward-looking


# ── lazy windows must equal the dense tensor (memory-safe training path) ──────

def test_lazy_windows_match_dense_tensor():
    df = _panel(n_tickers=5, n_days=60, seed=3)
    feat = ["f0", "f1"]
    L = 10
    dense = build_sequence_tensor(df, feat, lookback=L)
    lazy = build_sequence_windows(df, feat, lookback=L)
    assert isinstance(lazy, SequenceWindows)
    assert lazy.shape == dense.shape
    # full materialisation matches
    assert np.allclose(lazy.materialize(), dense, atol=1e-6)
    # fancy-index slicing (as the CV folds do) returns a consistent sub-view
    sel = np.array([0, 5, 17, 42, 41])
    assert np.allclose(lazy[sel].materialize(), dense[sel], atol=1e-6)


def test_lazy_windows_terminal_row_is_self():
    df = _panel(n_tickers=3, n_days=40, seed=4)
    feat = ["f0", "f1"]
    lazy = build_sequence_windows(df, feat, lookback=8)
    rowfeat = df.select(feat).to_numpy().astype("float32")
    term = lazy.flat[lazy.gather_idx[:, -1]]
    assert np.allclose(term, rowfeat, atol=1e-6)


# ── torch estimator (skipped if torch missing) ────────────────────────────────

@pytest.mark.parametrize("kind", list(SEQUENCE_MODELS))
def test_estimator_fit_predict_and_pickle(kind):
    pytest.importorskip("torch")
    import pickle
    from trading_system.models.sequence import TorchSequenceRegressor

    df = _panel(n_tickers=6, n_days=120, seed=1)
    X = build_sequence_tensor(df, ["f0", "f1"], lookback=8)
    y = np.tanh(X[:, -1, 0])  # learnable from the last timestep

    m = TorchSequenceRegressor(kind=kind, hidden_size=12, num_layers=1,
                               epochs=6, batch_size=64, device="cpu", seed=0)
    m.fit(X, y)
    p = m.predict(X)
    assert p.shape == (X.shape[0],)
    assert np.isfinite(p).all()

    # pickle round-trip → identical predictions (state_dict serialised)
    m2 = pickle.loads(pickle.dumps(m))
    assert np.allclose(p[:32], m2.predict(X[:32]), atol=1e-5)


def test_fit_predict_lazy_matches_dense():
    """Training on lazy windows must be numerically identical to the dense tensor
    (same seed, same standardisation, same batches) — the memory fix changes only
    *where* windows are materialised, not the maths."""
    pytest.importorskip("torch")
    from trading_system.models.sequence import TorchSequenceRegressor

    df = _panel(n_tickers=6, n_days=100, seed=7)
    feat = ["f0", "f1"]
    dense = build_sequence_tensor(df, feat, lookback=8)
    lazy = build_sequence_windows(df, feat, lookback=8)
    y = np.tanh(dense[:, -1, 0])

    kw = dict(kind="gru", hidden_size=12, num_layers=1, epochs=5,
              batch_size=64, device="cpu", seed=0)
    p_dense = TorchSequenceRegressor(**kw).fit(dense, y).predict(dense)
    p_lazy = TorchSequenceRegressor(**kw).fit(lazy, y).predict(lazy)
    assert np.allclose(p_dense, p_lazy, atol=1e-4)


def test_parallel_nonlinear_coexists_with_torch():
    """Regression: the spawn worker-pool used by the nonlinear feature build must
    neither segfault nor deadlock when torch is live in-process (both were loky
    failure modes that broke the suite)."""
    pytest.importorskip("torch")
    import torch
    import torch.nn as nn
    from datetime import date, timedelta
    from trading_system.features.nonlinear_panel import compute_nonlinear_features

    net = nn.LSTM(2, 4, batch_first=True)
    opt = torch.optim.Adam(net.parameters())

    def _train():
        for _ in range(2):
            opt.zero_grad()
            o, _ = net(torch.rand(8, 5, 2))
            o.sum().backward()
            opt.step()

    _train()  # torch state live BEFORE the parallel pool
    rng = np.random.default_rng(0)
    rows = []
    for t in range(4):
        px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))
        for i in range(200):
            rows.append((date(2022, 1, 3) + timedelta(days=i), f"T{t}", float(px[i])))
    df = pl.DataFrame(rows, schema=["date", "ticker", "adj_close"], orient="row")

    out = compute_nonlinear_features(df, deep=False, parallel=True)  # spawn pool
    _train()  # torch must still train AFTER the pool is torn down (no deadlock)

    assert "hurst_dfa_120" in out.columns
    assert out["hurst_dfa_120"].is_nan().sum() == 0
