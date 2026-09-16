"""Cost-aware trading policies: what to actually do with a target portfolio.

A ranking model produces a *target* book.  Trading all the way to that target
every rebalance is the wrong thing to do when trading costs money, and it is
what the current pipeline does.  The published fix has two layers:

**Trade partially toward an aim portfolio** (Gârleanu & Pedersen 2013).  With
quadratic costs and mean-reverting signals the optimal dynamic policy has a
closed form and two parts: *aim in front of the target* — tilt toward where the
target is heading, because a fast-decaying signal will have moved by the time
you arrive — and *trade partially toward the aim* at a rate set by the ratio of
costs to risk.  No learning is involved and it is very hard to beat, which makes
it the honest benchmark for any RL policy rather than a naive full-rebalance.

**No-trade bands.**  Leave a position alone while it is close enough to target.
Banding reduces cost at far less signal loss than the obvious alternative of
rebalancing less often, because it keeps reacting to large signal changes while
ignoring drift.  Entry and exit hurdles differ on purpose: it should be harder
to open a position than to keep one.

Both return *target weights*; the simulator prices the fills.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["apply_no_trade_bands", "GarleanuPedersenPolicy", "turnover_of"]


def turnover_of(w_new: np.ndarray, w_old: np.ndarray) -> float:
    """One-way turnover as a fraction of equity."""
    return float(np.abs(np.asarray(w_new) - np.asarray(w_old)).sum())


def apply_no_trade_bands(
    target: np.ndarray,
    current: np.ndarray,
    entry_hurdle: float = 0.30,
    hold_hurdle: float = 0.15,
) -> np.ndarray:
    """Band the move from ``current`` toward ``target``.

    A name not currently held must clear ``entry_hurdle`` (as a fraction of its
    proposed weight) before any of it is bought; a name already held is only
    adjusted once the gap exceeds ``hold_hurdle``.  Everything else stays where
    it is, which is the whole point — drift costs nothing to leave alone.
    """
    target = np.asarray(target, dtype=float)
    current = np.asarray(current, dtype=float)
    gap = target - current
    held = np.abs(current) > 1e-9
    scale = np.maximum(np.abs(target), 1e-9)
    hurdle = np.where(held, hold_hurdle, entry_hurdle) * scale
    move = np.abs(gap) > hurdle
    # a target of zero on a held name is an exit: never band an exit
    move |= held & (np.abs(target) <= 1e-12)
    return np.where(move, target, current)


@dataclass
class GarleanuPedersenPolicy:
    """Partial trading toward an aim portfolio.

    The closed-form solution says the new position is a convex combination of
    where you are and an *aim* portfolio,

        x_t = (1 - rate) * x_{t-1} + rate * aim_t

    with the aim itself a weighted average of the current Markowitz target and
    its expected future values.  With a signal decaying at rate ``phi`` per
    period the aim collapses to a shrunk target,

        aim_t = target_t / (1 + rate * phi / (1 - phi))   (per signal)

    so a fast-decaying signal (large ``phi``) is deliberately under-traded: by
    the time the position is built the reason for it has gone.

    Parameters
    ----------
    trade_rate:
        Fraction of the gap to the aim closed each rebalance.  In the closed
        form this is increasing in risk aversion and in the signal's value, and
        decreasing in trading costs.  ``1.0`` recovers full rebalancing.
    signal_decay:
        Per-rebalance decay ``phi`` of the alpha signal, in [0, 1).  Estimate it
        from the autocorrelation of the score cross-section at the rebalance
        frequency (:meth:`fit_decay`) rather than guessing.
    """

    trade_rate: float = 0.35
    signal_decay: float = 0.30

    def aim(self, target: np.ndarray) -> np.ndarray:
        """Shrink the Markowitz target toward zero by the signal's decay."""
        phi = float(np.clip(self.signal_decay, 0.0, 0.99))
        shrink = 1.0 / (1.0 + self.trade_rate * phi / max(1.0 - phi, 1e-6))
        return np.asarray(target, dtype=float) * shrink

    def step(self, current: np.ndarray, target: np.ndarray) -> np.ndarray:
        """New target weights: move ``trade_rate`` of the way toward the aim."""
        cur = np.asarray(current, dtype=float)
        a = self.aim(target)
        w = (1.0 - self.trade_rate) * cur + self.trade_rate * a
        # never hold a name the model has dropped to exactly zero *and* that the
        # book no longer wants at all: partial trading must still allow exits
        w = np.where((np.asarray(target) == 0.0) & (np.abs(w) < 1e-4), 0.0, w)
        s = w.sum()
        return w / s if s > 1e-9 else w

    @staticmethod
    def fit_decay(score_history: list[np.ndarray]) -> float:
        """Estimate ``phi`` from the lag-1 autocorrelation of the score ranks.

        ``score_history`` is a list of equal-length, same-order score vectors on
        consecutive rebalance dates.  Returns ``1 - rho``: a signal whose ranks
        persist perfectly decays at 0, one that is fresh noise each period
        decays at 1.
        """
        rhos = []
        for a, b in zip(score_history[:-1], score_history[1:], strict=True):
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 10:
                continue
            ra = a[m].argsort().argsort().astype(float)
            rb = b[m].argsort().argsort().astype(float)
            if ra.std() > 0 and rb.std() > 0:
                rhos.append(float(np.corrcoef(ra, rb)[0, 1]))
        if not rhos:
            return 0.5
        return float(np.clip(1.0 - np.mean(rhos), 0.0, 0.99))
