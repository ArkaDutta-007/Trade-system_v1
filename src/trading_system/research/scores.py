"""Causal score generation — computed once, reused everywhere.

Refitting an alpha model is by far the most expensive thing in this research
loop: an expanding-window GBM refit on a million-row panel takes minutes, and a
21-year walk-forward with annual refits does it twenty times.  Doing that again
for every execution policy you want to compare would make the interesting
questions — bands versus no bands, partial trading versus full, RL versus the
closed form — unaffordable.

So the expensive part is separated out.  :func:`generate_causal_scores` walks
the sample forward exactly as the simulator does, refitting on the same
schedule with the same purge, and records a dense ``(n_dates, n_tickers)``
matrix of scores.  Every cell is a number the model would have produced on that
date with only prior data.  After that, comparing ten execution policies is ten
cheap replays of the same tape.

The same matrix is what the RL environment consumes, which matters for more
than speed: it guarantees the agent trains on signals of exactly the quality the
backtest measures, rather than on in-sample scores that would teach it to trust
the model far more than it should.
"""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

from ..utils import get_logger
from .wfbacktest import AlphaModel, PanelData, WalkForwardConfig, _to_date

logger = get_logger(__name__)

__all__ = ["generate_causal_scores", "PrecomputedAlpha", "save_scores", "load_scores"]


def generate_causal_scores(
    panel: PanelData,
    features: pl.DataFrame,
    feat_cols: list[str],
    model_factory: Callable[[], AlphaModel],
    cfg: WalkForwardConfig,
    label: str = "",
) -> tuple[np.ndarray, list[dict]]:
    """Walk the sample forward, refitting on schedule, recording every score.

    Returns ``(scores, refits)`` where ``scores`` is ``(T, N)`` with NaN wherever
    the model produced nothing (before the first refit, on non-rebalance dates,
    or for a name with incomplete features).  Scores are held forward from the
    scoring date until the next one, so a daily consumer sees the signal that
    was actually live.
    """
    T, N = panel.ret.shape
    tix = {t: i for i, t in enumerate(panel.tickers)}

    tgt = f"_cs_fwd_{cfg.horizon}d"
    feats = features.sort(["ticker", "date"]).with_columns(
        ((pl.col("adj_close").shift(-cfg.horizon).over("ticker") / pl.col("adj_close")) - 1)
        .alias(tgt))
    if cfg.neutralize:
        feats = feats.with_columns(
            (pl.col(tgt) - pl.col(tgt).mean().over("date")).alias(tgt))
    feats = feats.with_columns(pl.col("date").cast(pl.Date))

    score_frame = feats.select(["date", "ticker", *feat_cols]).drop_nulls(subset=feat_cols)
    by_date = {_to_date(d): g for (d,), g in
               score_frame.group_by(["date"], maintain_order=True)}

    oos_end = cfg.oos_end or panel.dates[-1]
    idxs = [i for i, d in enumerate(panel.dates) if cfg.oos_start <= d <= oos_end]
    if not idxs:
        raise ValueError("no dates in the requested window")

    scores = np.full((T, N), np.nan)
    refits: list[dict] = []
    model: AlphaModel | None = None
    last_refit = -10 ** 9
    rebal = set(idxs[:: cfg.rebalance_days])

    for i in idxs:
        d = panel.dates[i]
        if i not in rebal:
            continue
        if model is None or (i - last_refit) >= cfg.retrain_days:
            cut = d - timedelta(days=cfg.purge_calendar_days())
            train = feats.filter((pl.col("date") <= cut) & pl.col(tgt).is_not_null())
            if cfg.train_window_days is not None:
                lo = cut - timedelta(days=int(cfg.train_window_days * 365 / 252))
                train = train.filter(pl.col("date") >= lo)
            train = train.drop_nulls(subset=feat_cols)
            if train.height >= 5000:
                model = model_factory()
                model.fit(train, feat_cols, tgt)
                last_refit = i
                refits.append({"date": str(d), "cut": str(cut), "rows": train.height,
                               "model": getattr(model, "chosen", model.name)})
                logger.info(f"{label} refit @{d} cut={cut} rows={train.height:,} "
                            f"→ {getattr(model, 'chosen', model.name)}")
        if model is None:
            continue
        today = by_date.get(d)
        if today is None or today.height < 10:
            continue
        cand = today["ticker"].to_list()
        keep = np.array([t in tix for t in cand])
        if not keep.any():
            continue
        cidx = np.array([tix[t] for t in cand if t in tix])
        scores[i, cidx] = np.asarray(model.predict(today, feat_cols)).ravel()[keep]

    # hold the last score forward so a daily consumer sees the live signal
    last = np.full(N, np.nan)
    for i in range(T):
        row = scores[i]
        have = np.isfinite(row)
        if have.any():
            last = row.copy()
        else:
            scores[i] = last
    return scores, refits


class PrecomputedAlpha:
    """Replays a score matrix produced by :func:`generate_causal_scores`.

    Nothing is fitted; ``fit`` is a no-op so the simulator's refit schedule
    costs nothing.  The scores it serves were generated under that same
    schedule, so the causality guarantee is unchanged — it has only been moved
    one step earlier in the pipeline.
    """

    def __init__(self, scores: np.ndarray, panel: PanelData, name: str = "precomputed"):
        self.scores = scores
        self.didx = panel.date_index()
        self.tix = {t: i for i, t in enumerate(panel.tickers)}
        self.name = name
        self.chosen = name

    def fit(self, panel, feat_cols, target) -> None:
        return None

    def predict(self, today: pl.DataFrame, feat_cols) -> np.ndarray:
        d = _to_date(today["date"][0])
        i = self.didx.get(d)
        if i is None:
            return np.full(today.height, np.nan)
        row = self.scores[i]
        return np.array([row[self.tix[t]] if t in self.tix else np.nan
                         for t in today["ticker"].to_list()])


def save_scores(path: Path, scores: np.ndarray, panel: PanelData, meta: dict) -> None:
    """Persist a score matrix with the axes needed to reattach it to a panel."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, scores=scores,
        dates=np.array([str(d) for d in panel.dates]),
        tickers=np.array([str(t) for t in panel.tickers]),
        meta=np.array([str(meta)]))


def load_scores(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load ``(scores, dates, tickers)`` saved by :func:`save_scores`."""
    z = np.load(path, allow_pickle=False)
    return z["scores"], z["dates"], z["tickers"]
