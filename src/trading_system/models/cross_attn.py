"""Cross-sectional attention forecaster — the one deep architecture with evidence.

Everything this repo trains is an *own-asset* model: the score for NVDA on a
given day is a function of NVDA's own feature row.  Tree ensembles are very good
at that and, on low-signal panels, beat deep nets at it — which is exactly what
this system's own metrics found when the RNN/LSTM/GRU families were pruned with
negative IC.  Stacking more capacity on own-asset prediction does not help.

The gap is structural, not architectural.  Kelly, Kuznetsov, Malamud & Xu
(*Artificial Intelligence Asset Pricing Models*, NBER w33351, rev. 2026) isolate
it cleanly: take a linear SDF, change nothing except adding an attention
mechanism that lets each asset see the rest of the cross-section, and the
out-of-sample Sharpe goes from 3.6 to 3.9 with an alpha t-statistic of 6.8
against the attention-free version.  Nonlinear transformers do better again, and
crucially the MLP and the transformer have significant alpha *against each
other* — own-asset and cross-asset models are complements, not substitutes.
That is the case for adding this alongside the GBMs rather than replacing them.

The same paper is the reason this model is small.  Out-of-sample performance
rises with parameter count and then flattens: their curves are flat by ~25,000
parameters, and going to a million changes nothing ("limits to learning" — there
is not enough financial data to support more).  The defaults here land near
20-30k parameters on purpose.  Three A6000s do not change that arithmetic; they
just make the seed-averaging and the walk-forward refits fast.

Design notes
------------
*The cross-section is a set, not a sequence.*  There is no positional encoding
and no causal mask across assets: attention is permutation-equivariant, so
relabelling the tickers cannot change any score.  That is the correct inductive
bias and it is free.

*Features are rank-gauss normalised within each date, plus a market-context
block.*  Raw feature levels are not comparable across regimes (a 2008 volatility
reading is not a 2017 one) and the quantity being predicted is cross-sectional,
so ranking within a date is the right representation and a strong outlier
defence, using only same-date information.

But ranking alone is not enough, and the failure mode is easy to miss: after
within-date ranking *every date has an identical feature distribution*, so the
cross-sectional mean — the market state — has been erased from the input.  A
model asked to learn a signal whose sign flips with the regime then cannot
learn it at all, however much attention it has.  That is not hypothetical; it
is what ``tests/unit/test_cross_attn.py::TestLearning`` measures, and the
rank-only version scores an IC of -0.01 on a signal it should capture easily.

So each asset's row is concatenated with a **context block**: the date's
cross-sectional mean of every raw feature, standardised with statistics frozen
from the training window.  Attention then shares information about *relative*
position across assets while the context block carries the *level* of the
market that day.  Both channels are needed; neither substitutes for the other.

*The loss is per-date rank correlation, not MSE.*  MSE on forward returns spends
its capacity fitting the market factor and the fat tails.  The objective here is
a differentiable surrogate for the within-date IC — the statistic the book
actually monetises — optionally blended with an after-cost portfolio return
term (:class:`CrossAttnConfig.sharpe_weight`), which is the end-to-end variant.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ..utils import get_logger

logger = get_logger(__name__)

__all__ = ["CrossAttnConfig", "CrossSectionalAttention", "CrossAttnAlpha",
           "rank_gauss", "rank_gauss_2d", "torch_available"]


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


@dataclass
class CrossAttnConfig:
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 64
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-2
    epochs: int = 40
    patience: int = 6
    batch_dates: int = 24
    val_frac: float = 0.15
    max_assets: int = 512
    use_context: bool = True       # concatenate the date's market-state block
    sharpe_weight: float = 0.0     # >0 blends an after-cost long-short P&L term
    turnover_penalty: float = 0.0  # bps charged on the P&L term's turnover
    top_frac: float = 0.10         # long/short fraction for the P&L term
    seed: int = 0
    device: str | None = None
    verbose: bool = True

    def n_params_estimate(self, n_features: int) -> int:
        n_features = n_features * (2 if self.use_context else 1)
        d, ff, L = self.d_model, self.d_ff, self.n_layers
        per_layer = 4 * d * d + 4 * d + 2 * d * ff + d + ff + 4 * d
        return n_features * d + d + L * per_layer + d + 1


# ── Data preparation ─────────────────────────────────────────────────────────

def rank_gauss(x: np.ndarray) -> np.ndarray:
    """Map a vector to standard-normal quantiles by rank (ties broken by order).

    ``rank/(n+1)`` then the normal inverse CDF.  Bounded, outlier-proof, and
    scale-free, so a feature whose units drift across decades still enters the
    model on a stable footing.
    """
    from scipy.stats import norm
    n = len(x)
    if n < 2:
        return np.zeros(n)
    finite = np.isfinite(x)
    out = np.zeros(n)
    if finite.sum() < 2:
        return out
    r = x[finite].argsort().argsort().astype(float)
    out[finite] = norm.ppf((r + 1.0) / (finite.sum() + 1.0))
    return out


def rank_gauss_2d(X: np.ndarray) -> np.ndarray:
    """Rank-gauss every column of ``X`` at once.

    Same result as applying :func:`rank_gauss` column by column, but one
    ``norm.ppf`` call on the whole matrix instead of one per feature.  That
    matters: the walk-forward refits this on an expanding window 22 times, and
    with ~4,300 dates and 68 features the column-wise version dominates the
    whole run — the GPU sits idle while a Python loop ranks arrays.
    """
    from scipy.stats import norm

    X = np.asarray(X, dtype=np.float64)
    n, f = X.shape
    if n < 2:
        return np.zeros_like(X)

    finite = np.isfinite(X)
    # sort NaNs to the end of each column so they never take a real rank
    order = np.argsort(np.where(finite, X, np.inf), axis=0, kind="stable")
    ranks = np.empty((n, f), dtype=np.int64)
    np.put_along_axis(ranks, order, np.arange(n)[:, None].repeat(f, axis=1), axis=0)

    cnt = finite.sum(axis=0)
    out = np.zeros((n, f))
    usable = cnt >= 2
    if usable.any():
        q = (ranks[:, usable] + 1.0) / (cnt[usable] + 1.0)
        out[:, usable] = norm.ppf(np.clip(q, 1e-9, 1 - 1e-9))
    out[~finite] = 0.0
    return out


def build_date_batches(
    panel: pl.DataFrame, feat_cols: list[str], target: str | None,
    max_assets: int = 512,
) -> tuple[list[np.ndarray], list[np.ndarray], list, np.ndarray]:
    """Group a long panel into per-date (features, target, context) arrays.

    Returns ``(X_list, y_list, dates, ctx)``:

    * ``X_list[i]`` is ``(n_assets_i, n_features)``, rank-gauss normalised
      within that date — the *relative* view.
    * ``y_list[i]`` is the matching target, also rank-gauss normalised so the
      correlation loss is well conditioned.
    * ``ctx`` is ``(n_dates, n_features)``: each date's cross-sectional mean of
      the **raw** features — the *level* view that ranking throws away.  It is
      returned unstandardised; the model freezes the scaling from its training
      window so nothing about the evaluation period leaks into it.
    """
    cols = ["date", *feat_cols] + ([target] if target else [])
    sub = panel.select(cols)
    if target:
        sub = sub.drop_nulls(subset=[target])
    sub = sub.drop_nulls(subset=feat_cols)

    Xs, ys, ds, ctx = [], [], [], []
    for (d,), g in sub.group_by(["date"], maintain_order=True):
        if g.height < 20:
            continue
        if g.height > max_assets:
            g = g.head(max_assets)
        raw = g.select(feat_cols).to_numpy().astype(np.float64)
        ctx.append(np.nanmean(raw, axis=0))
        Xs.append(rank_gauss_2d(raw).astype(np.float32))
        if target:
            ys.append(rank_gauss(g[target].to_numpy().astype(np.float64)).astype(np.float32))
        ds.append(d)
    ctx_arr = (np.asarray(ctx, dtype=np.float32) if ctx
               else np.zeros((0, len(feat_cols)), dtype=np.float32))
    return Xs, ys, ds, ctx_arr


# ── Model ────────────────────────────────────────────────────────────────────

def _build_module(n_features: int, cfg: CrossAttnConfig):
    import torch
    import torch.nn as nn

    class Block(nn.Module):
        """Pre-norm attention block over the asset axis."""

        def __init__(self):
            super().__init__()
            self.n1 = nn.LayerNorm(cfg.d_model)
            self.attn = nn.MultiheadAttention(
                cfg.d_model, cfg.n_heads, dropout=cfg.dropout, batch_first=True)
            self.n2 = nn.LayerNorm(cfg.d_model)
            self.ff = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(),
                nn.Dropout(cfg.dropout), nn.Linear(cfg.d_ff, cfg.d_model))
            self.drop = nn.Dropout(cfg.dropout)

        def forward(self, h, key_padding_mask):
            z = self.n1(h)
            a, _ = self.attn(z, z, z, key_padding_mask=key_padding_mask,
                             need_weights=False)
            h = h + self.drop(a)
            h = h + self.drop(self.ff(self.n2(h)))
            return h

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(n_features, cfg.d_model)
            self.blocks = nn.ModuleList([Block() for _ in range(cfg.n_layers)])
            self.norm = nn.LayerNorm(cfg.d_model)
            self.head = nn.Linear(cfg.d_model, 1)

        def forward(self, x, key_padding_mask):
            """x: (B, A, F); key_padding_mask: (B, A) True where padded."""
            h = self.embed(x)
            for b in self.blocks:
                h = b(h, key_padding_mask)
            return self.head(self.norm(h)).squeeze(-1)

    torch.manual_seed(cfg.seed)
    return Net()


def _ic_loss(pred, target, mask):
    """Negative mean within-date Pearson correlation of predictions and target.

    Both sides are already rank-gauss transformed, so Pearson on them is a
    smooth stand-in for Spearman — differentiable, and it is exactly the
    quantity ICIR is built from.
    """
    p = pred.masked_fill(mask, 0.0)
    t = target.masked_fill(mask, 0.0)
    n = (~mask).sum(dim=1, keepdim=True).clamp(min=2).float()
    pm = p - (p.sum(1, keepdim=True) / n)
    tm = t - (t.sum(1, keepdim=True) / n)
    pm = pm.masked_fill(mask, 0.0)
    tm = tm.masked_fill(mask, 0.0)
    num = (pm * tm).sum(1)
    den = pm.pow(2).sum(1).sqrt() * tm.pow(2).sum(1).sqrt() + 1e-8
    return -(num / den).mean()


def _pnl_loss(pred, target, mask, top_frac: float, turnover_bps: float):
    """Negative mean P&L of a soft long-short book built from ``pred``.

    Weights come from a temperature-sharpened softmax over the predicted scores
    (long) and its negative (short), which keeps the whole thing differentiable
    while behaving like a top-decile book.  ``turnover_bps`` charges the
    date-over-date weight change, so the objective can prefer a slightly weaker
    signal that trades less — the property the end-to-end literature identifies
    as the difference between a transformer that survives costs and an LSTM that
    does not.
    """
    import torch
    neg_inf = torch.finfo(pred.dtype).min
    p = pred.masked_fill(mask, neg_inf)
    tau = 1.0 / max(top_frac, 1e-3)
    wl = torch.softmax(p * tau, dim=1)
    ws = torch.softmax((-p).masked_fill(mask, neg_inf) * tau, dim=1)
    w = wl - ws
    r = target.masked_fill(mask, 0.0)
    pnl = (w * r).sum(1)
    if turnover_bps > 0 and w.shape[0] > 1:
        turn = (w[1:] - w[:-1]).abs().sum(1)
        pnl = pnl - torch.cat([turn.new_zeros(1), turn]) * (turnover_bps / 1e4)
    return -pnl.mean()


class CrossSectionalAttention:
    """Trainable cross-sectional attention model (fit/predict on numpy)."""

    def __init__(self, n_features: int, cfg: CrossAttnConfig | None = None):
        if not torch_available():
            raise ImportError("cross-attention model needs torch")
        import torch
        self.cfg = cfg or CrossAttnConfig()
        self.n_features = n_features
        self.in_features = n_features * (2 if self.cfg.use_context else 1)
        self.device = torch.device(
            self.cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net = _build_module(self.in_features, self.cfg).to(self.device)
        # frozen from the training window; see the module docstring
        self.ctx_mean = np.zeros(n_features, dtype=np.float32)
        self.ctx_std = np.ones(n_features, dtype=np.float32)
        self.n_params = sum(p.numel() for p in self.net.parameters())
        self.history: list[dict] = []

    # ── batching helpers ──
    def _with_context(self, X: np.ndarray, ctx_row: np.ndarray | None) -> np.ndarray:
        """Concatenate the standardised market-state block onto every asset row."""
        if not self.cfg.use_context:
            return X
        if ctx_row is None:
            c = np.zeros(self.n_features, dtype=np.float32)
        else:
            c = (np.asarray(ctx_row, dtype=np.float32) - self.ctx_mean) / self.ctx_std
            c = np.clip(np.nan_to_num(c, nan=0.0), -5.0, 5.0)
        return np.concatenate([X, np.tile(c, (X.shape[0], 1))], axis=1).astype(np.float32)

    def _pad(self, Xs, ys, idx, ctx=None):
        import torch
        A = max(Xs[i].shape[0] for i in idx)
        B = len(idx)
        x = np.zeros((B, A, self.in_features), dtype=np.float32)
        y = np.zeros((B, A), dtype=np.float32)
        m = np.ones((B, A), dtype=bool)
        for b, i in enumerate(idx):
            n = Xs[i].shape[0]
            x[b, :n] = self._with_context(
                Xs[i], ctx[i] if ctx is not None and len(ctx) > i else None)
            m[b, :n] = False
            if ys is not None:
                y[b, :n] = ys[i]
        t = torch.as_tensor
        return (t(x).to(self.device), t(y).to(self.device), t(m).to(self.device))

    def fit(self, Xs: list[np.ndarray], ys: list[np.ndarray], dates: list,
            ctx: np.ndarray | None = None) -> "CrossSectionalAttention":
        """Train with an inner *temporal* validation split and early stopping.

        The split is the last ``val_frac`` of dates, never a random row split:
        a random split would put neighbouring days on both sides and the
        overlapping labels would leak straight through.
        """
        import torch
        cfg = self.cfg
        n = len(Xs)
        if n < 50:
            raise ValueError(f"need >=50 dates to train, got {n}")
        cut = int(n * (1 - cfg.val_frac))
        tr_idx, va_idx = list(range(cut)), list(range(cut, n))

        # Freeze the context scaling on the *training* dates only. Standardising
        # over the whole array would let the evaluation period set the scale.
        if cfg.use_context and ctx is not None and len(ctx) >= cut > 0:
            tr_ctx = np.asarray(ctx[:cut], dtype=np.float64)
            self.ctx_mean = np.nan_to_num(tr_ctx.mean(0)).astype(np.float32)
            sd = np.nan_to_num(tr_ctx.std(0))
            self.ctx_std = np.where(sd > 1e-8, sd, 1.0).astype(np.float32)
        self._ctx = ctx

        opt = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr,
                                weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
        rng = np.random.default_rng(cfg.seed)
        best, best_state, bad = np.inf, None, 0

        for ep in range(cfg.epochs):
            self.net.train()
            order = rng.permutation(tr_idx)
            tot, nb = 0.0, 0
            for s in range(0, len(order), cfg.batch_dates):
                idx = order[s: s + cfg.batch_dates]
                if len(idx) < 2:
                    continue
                x, y, m = self._pad(Xs, ys, idx, ctx)
                loss = _ic_loss(self.net(x, m), y, m)
                if cfg.sharpe_weight > 0:
                    loss = (1 - cfg.sharpe_weight) * loss + cfg.sharpe_weight * _pnl_loss(
                        self.net(x, m), y, m, cfg.top_frac, cfg.turnover_penalty)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                tot += float(loss.detach())
                nb += 1
            sched.step()

            vl = self._eval(Xs, ys, va_idx, ctx) if va_idx else tot / max(nb, 1)
            self.history.append({"epoch": ep, "train": tot / max(nb, 1), "val": vl})
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        if cfg.verbose:
            logger.info(f"  cross-attn: {self.n_params:,} params, "
                        f"{len(self.history)} epochs, best val IC {-best:+.4f}")
        return self

    def _eval(self, Xs, ys, idx, ctx=None) -> float:
        import torch
        if not idx:
            return float("inf")
        self.net.eval()
        tot, nb = 0.0, 0
        with torch.no_grad():
            for s in range(0, len(idx), self.cfg.batch_dates):
                b = idx[s: s + self.cfg.batch_dates]
                if len(b) < 2:
                    continue
                x, y, m = self._pad(Xs, ys, b, ctx)
                tot += float(_ic_loss(self.net(x, m), y, m))
                nb += 1
        return tot / max(nb, 1)

    def predict_one(self, X: np.ndarray, ctx_row: np.ndarray | None = None) -> np.ndarray:
        """Score one date's cross-section: ``(n_assets, n_features)`` -> scores."""
        import torch
        self.net.eval()
        Xc = self._with_context(X, ctx_row)
        with torch.no_grad():
            x = torch.as_tensor(Xc[None, :, :]).to(self.device)
            m = torch.zeros((1, Xc.shape[0]), dtype=torch.bool, device=self.device)
            return self.net(x, m).squeeze(0).cpu().numpy()


# ── AlphaModel adapter for the walk-forward simulator ────────────────────────

class CrossAttnAlpha:
    """Adapter so the attention model drops into the walk-forward loop.

    ``n_seeds`` fits are averaged.  Seed variance on a low-signal panel is large
    enough that a single fit's walk-forward result is mostly a statement about
    the seed; averaging is the standard fix and it parallelises across the GPUs.
    """

    name = "cross_attn"

    def __init__(self, cfg: CrossAttnConfig | None = None, n_seeds: int = 3):
        self.cfg = cfg or CrossAttnConfig()
        self.n_seeds = n_seeds
        self.models: list[CrossSectionalAttention] = []
        self.chosen = "cross_attn"

    def fit(self, panel: pl.DataFrame, feat_cols: list[str], target: str) -> None:
        Xs, ys, ds, ctx = build_date_batches(panel, feat_cols, target,
                                             max_assets=self.cfg.max_assets)
        self.models = []
        for s in range(self.n_seeds):
            c = CrossAttnConfig(**{**self.cfg.__dict__, "seed": self.cfg.seed + s})
            m = CrossSectionalAttention(len(feat_cols), c).fit(Xs, ys, ds, ctx)
            self.models.append(m)
        self.chosen = f"cross_attn x{self.n_seeds} ({self.models[0].n_params:,}p)"

    def predict(self, today: pl.DataFrame, feat_cols: list[str]) -> np.ndarray:
        raw = today.select(feat_cols).to_numpy().astype(np.float64)
        ctx_row = np.nanmean(raw, axis=0)
        X = rank_gauss_2d(raw).astype(np.float32)
        return np.mean([m.predict_one(X, ctx_row) for m in self.models], axis=0)
