"""Sequence models — RNN / LSTM / GRU forecasters over per-ticker lookback windows.

Why sequence models
-------------------
The tabular families (LightGBM / XGBoost / HistGBM / Ridge) treat every
``(ticker, date)`` row as an *independent* point: any temporal structure has to
be hand-encoded as a feature (``mom_20d``, ``vol_60d`` …).  A recurrent net
instead reads the **ordered lookback window** of the last ``lookback``
observations and learns the temporal structure itself — path dependence,
momentum build-up/exhaustion, volatility clustering — that a flat feature vector
throws away.

These compete head-to-head with the tree models under the *same*
purged + embargoed walk-forward CV in :mod:`forecast_train`, ranked by ICIR and
gated by the label-shuffle leakage test.  There is no special-casing of the
winner: if an LSTM genuinely generalises better at 252d it gets selected; if it
overfits, the gate drops it.

Leakage safety
--------------
:func:`build_sequence_tensor` builds, for each row ``i``, the window of the
``lookback`` rows **ending at i within the same ticker** — strictly backward in
time.  The forward-return label is still produced by ``forecast_train`` and the
purge removes any *training* row whose label window overlaps the test block,
identical to the tabular path.  A test row's lookback window may reach back
across the purge gap into older data — that is legitimate use of history, not
leakage.

Hardware
--------
Device is taken from ``get_compute_profile().torch_device`` (``cuda`` / ``mps`` /
``cpu``) so the same code trains on an RTX box, an Apple-silicon laptop (MPS) or
CPU.  Everything is float32 — MPS has no float64.

torch is an *optional* dependency (``pip install -e '.[deep]'``).  Importing this
module never requires torch; only constructing/fitting a model does.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import polars as pl

# Names recognised as sequence (deep) families by the trainer/CLI.
SEQUENCE_MODELS: tuple[str, ...] = ("rnn", "lstm", "gru")


def _preload_omp() -> None:
    """Load LightGBM/XGBoost OpenMP before torch (macOS libomp ordering).

    See ``utils.compute.preload_omp_runtimes`` for the why. This makes the
    *standalone* sequence path (e.g. unit tests that never call
    ``get_compute_profile``) crash-safe too.
    """
    try:
        from ..utils.compute import preload_omp_runtimes
        preload_omp_runtimes()
    except Exception:
        pass


def _limit_torch_threads(torch) -> None:
    """Pin torch to a single intra-op thread.

    In a process that has also loaded LightGBM/XGBoost, torch's CPU OpenMP pool
    **deadlocks** against theirs the first time a net trains — even with the
    correct lib load order (``preload_omp_runtimes``).  One thread sidesteps the
    deadlock entirely and is plenty for these small recurrent nets; GPU (MPS/CUDA)
    compute is unaffected (this only bounds CPU host-side ops).
    """
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def torch_available() -> bool:
    """True if torch can be imported (so deep models are runnable)."""
    _preload_omp()
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


# ── Causal windowing ──────────────────────────────────────────────────────────

def build_sequence_tensor(
    sub: "pl.DataFrame",
    feat_cols: list[str],
    lookback: int,
    group_col: str = "ticker",
) -> np.ndarray:
    """Build a ``(n_rows, lookback, n_features)`` tensor aligned to ``sub``'s rows.

    ``sub`` must be sorted by ``[group_col, date]`` (the order produced by the
    trainer's ``_xy``) and free of nulls in ``feat_cols`` — so the flat design
    matrix ``X`` and this tensor share an identical row order, and the purged
    split indices slice both consistently.

    For row ``i`` the window is the ``lookback`` rows of the same ticker ending
    at ``i``, edge-padded (the earliest row repeated) when fewer than
    ``lookback`` rows of history exist.  Strictly backward-looking.
    """
    tickers = sub[group_col].to_numpy()
    F = sub.select(feat_cols).to_numpy().astype(np.float32)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)  # defensive; sub is null-free
    n, nf = F.shape
    if n == 0:
        return np.empty((0, lookback, nf), dtype=np.float32)

    out = np.empty((n, lookback, nf), dtype=np.float32)
    start = 0
    for i in range(1, n + 1):
        if i == n or tickers[i] != tickers[start]:
            block = F[start:i]                                   # (T, nf), contiguous ticker
            pad = np.repeat(block[:1], lookback - 1, axis=0)     # edge-pad with first row
            padded = np.vstack([pad, block])                     # (T + L - 1, nf)
            win = np.lib.stride_tricks.sliding_window_view(padded, lookback, axis=0)  # (T, nf, L)
            out[start:i] = win.transpose(0, 2, 1)                # (T, L, nf)
            start = i
    return out


# ── torch module ──────────────────────────────────────────────────────────────

def _make_net(kind, n_features, hidden, layers, dropout, bidirectional):
    import torch.nn as nn

    rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[kind]
    rnn_kwargs = dict(
        input_size=n_features, hidden_size=hidden, num_layers=layers,
        batch_first=True, dropout=(dropout if layers > 1 else 0.0),
        bidirectional=bidirectional,
    )
    if kind == "rnn":
        rnn_kwargs["nonlinearity"] = "tanh"

    class _SeqNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = rnn_cls(**rnn_kwargs)
            d = hidden * (2 if bidirectional else 1)
            self.head = nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, max(8, d // 2)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(max(8, d // 2), 1),
            )

        def forward(self, x):
            out, _ = self.rnn(x)        # (B, L, d) — works for RNN/LSTM/GRU
            return self.head(out[:, -1, :]).squeeze(-1)

    return _SeqNet()


# ── sklearn-style estimator ─────────────────────────────────────────────────────

class TorchSequenceRegressor:
    """RNN/LSTM/GRU regressor with a scikit-learn ``fit``/``predict`` surface.

    Inputs are 3-D ``(n_samples, lookback, n_features)`` tensors from
    :func:`build_sequence_tensor`.  Features and target are standardised
    internally on the training split; a chronology-agnostic random tail is held
    out only for early stopping (the *reported* metrics come from the outer
    purged CV).  Trained with AdamW + Huber loss (robust to fat tails).
    """

    def __init__(
        self,
        kind: str = "lstm",
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        epochs: int = 40,
        batch_size: int = 256,
        patience: int = 6,
        val_frac: float = 0.15,
        huber_beta: float = 1.0,
        grad_clip: float = 1.0,
        device: str | None = None,
        seed: int = 0,
        verbose: bool = False,
    ):
        if kind not in SEQUENCE_MODELS:
            raise ValueError(f"kind must be one of {SEQUENCE_MODELS}, got {kind!r}")
        self.kind = kind
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_frac = val_frac
        self.huber_beta = huber_beta
        self.grad_clip = grad_clip
        self.device = device
        self.seed = seed
        self.verbose = verbose
        # learned state
        self.net_ = None
        self._n_features = None
        self.mu_ = self.sd_ = None
        self.y_mu_ = self.y_sd_ = None

    # -- device -----------------------------------------------------------------
    def _resolve_device(self):
        import torch
        if self.device:
            return torch.device(self.device)
        try:
            from ..utils import get_compute_profile
            dev = get_compute_profile().torch_device
        except Exception:
            dev = "cpu"
        if dev == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if dev == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # -- fit --------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray):
        if not torch_available():
            raise ImportError(
                "Sequence models need torch. Install with: pip install -e '.[deep]'"
            )
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        _limit_torch_threads(torch)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        n, L, F = X.shape
        self._n_features = F

        # standardise features (over all timesteps) and target
        flat = X.reshape(-1, F)
        self.mu_ = np.nanmean(flat, axis=0).astype(np.float32)
        self.sd_ = (np.nanstd(flat, axis=0) + 1e-8).astype(np.float32)
        self.y_mu_ = float(np.mean(y))
        self.y_sd_ = float(np.std(y) + 1e-8)
        Xs = (X - self.mu_) / self.sd_
        ys = (y - self.y_mu_) / self.y_sd_

        device = self._resolve_device()
        net = _make_net(self.kind, F, self.hidden_size, self.num_layers,
                        self.dropout, self.bidirectional).to(device)

        # random early-stopping holdout (regularisation only — not reported)
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(n)
        n_val = max(1, int(n * self.val_frac)) if n > 20 else 0
        val_idx, tr_idx = idx[:n_val], idx[n_val:]

        to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a))
        tr_ds = TensorDataset(to_t(Xs[tr_idx]), to_t(ys[tr_idx]))
        loader = DataLoader(tr_ds, batch_size=self.batch_size, shuffle=True, drop_last=False)
        if n_val:
            Xv = to_t(Xs[val_idx]).to(device)
            yv = to_t(ys[val_idx]).to(device)

        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = torch.nn.SmoothL1Loss(beta=self.huber_beta)

        best_val, best_state, bad = np.inf, None, 0
        for ep in range(self.epochs):
            net.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = loss_fn(net(xb), yb)
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), self.grad_clip)
                opt.step()

            if n_val:
                net.eval()
                with torch.no_grad():
                    vloss = float(loss_fn(net(Xv), yv).item())
                if vloss < best_val - 1e-6:
                    best_val, bad = vloss, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                if self.verbose:
                    print(f"  [{self.kind}] epoch {ep+1}/{self.epochs} val={vloss:.4f} bad={bad}")
                if bad >= self.patience:
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        self.net_ = net
        return self

    # -- predict ----------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.net_ is None:
            raise RuntimeError("model is not fitted")
        _preload_omp()
        import torch
        _limit_torch_threads(torch)

        X = np.asarray(X, dtype=np.float32)
        Xs = (X - self.mu_) / self.sd_
        device = self._resolve_device()
        net = self.net_.to(device)
        net.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(Xs), 4096):
                xb = torch.from_numpy(np.ascontiguousarray(Xs[i:i + 4096])).to(device)
                preds.append(net(xb).cpu().numpy())
        out = np.concatenate(preds) if preds else np.zeros(0, dtype=np.float32)
        return out * self.y_sd_ + self.y_mu_

    # -- pickling (store best state_dict, not the live module/device) -----------
    def __getstate__(self):
        state = self.__dict__.copy()
        net = state.pop("net_", None)
        if net is not None:
            state["_net_state"] = {k: v.detach().cpu() for k, v in net.state_dict().items()}
            state["_net_arch"] = (
                self.kind, self._n_features, self.hidden_size,
                self.num_layers, self.dropout, self.bidirectional,
            )
        return state

    def __setstate__(self, state):
        arch = state.pop("_net_arch", None)
        sd = state.pop("_net_state", None)
        self.__dict__.update(state)
        self.net_ = None
        if arch is not None and sd is not None:
            net = _make_net(*arch)
            net.load_state_dict(sd)
            net.eval()
            self.net_ = net


def build_sequence_model(kind: str, lookback: int, prof=None, **overrides) -> TorchSequenceRegressor:
    """Factory used by the trainer. Sensible defaults, device from compute profile."""
    device = None
    if prof is not None:
        device = getattr(prof, "torch_device", None)
    params = dict(
        kind=kind, hidden_size=64, num_layers=2, dropout=0.2,
        epochs=40, batch_size=256, patience=6, device=device,
    )
    params.update(overrides)
    return TorchSequenceRegressor(**params)
