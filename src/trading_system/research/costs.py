"""Realistic, point-in-time transaction costs for the walk-forward simulator.

The repo's legacy :class:`~trading_system.backtesting.slippage.CostModel` charges a
flat bps rate on turnover plus an optional ``turnover**1.5`` term applied to the
*portfolio*.  That is fine for a sanity check and wrong for a book that trades
small, illiquid names: real cost is a per-name quantity that depends on the
spread of that name on that day and on how large the order is relative to that
name's volume.  Three names at 5% of ADV cost far more than one name at 15% of a
liquid ADV, and the portfolio-level formula cannot tell them apart.

This module prices each order separately:

    cost_i = notional_i * (half_spread_i + commission) + impact_i

``half_spread_i`` comes from an **Abdi–Ranaldo** estimate built only from that
name's own daily close/high/low bars (Abdi & Ranaldo 2017, *RFS*), so it is
available at every point in the sample with no vendor quote data and no
look-ahead.  Corwin–Schultz (2012) is also implemented and was tried first; on
this panel it returns ~48 bps for every liquidity decile including AAPL, because
it reads overnight gap volatility as spread.  Abdi–Ranaldo is monotone in
liquidity as it should be — roughly 30 bps in the least liquid decile against 10
bps in the most, 7 bps for AAPL, 5 for JNJ, 147 for a microcap.  Both estimators
run high against true quoted spreads for mega caps, which errs toward charging
the book too much rather than too little; the cost ladder
(:meth:`RealisticCostModel.scaled`) is there to show how much that assumption
matters.  ``impact_i`` follows the square-root law that is standard in the execution
literature (Almgren et al. 2005; Tóth et al. 2011):

    impact_bps_i = eta * sigma_i * sqrt(q_i / ADV_i) * 1e4

with ``sigma_i`` the name's daily return volatility and ``q_i`` the traded
notional.  Both terms are *per name, per day*, so a strategy that concentrates
into microcaps is charged for it — which is exactly the failure mode the
existing raw-rank backtest hides (49.9% CAGR whose fills are fiction).

A **participation cap** enforces the other half of that reality: an order larger
than ``max_participation`` of the day's dollar volume cannot be filled today.
The simulator fills what it can and carries the rest, so an "alpha" that only
exists at impossible size shows up as a tracking shortfall instead of free P&L.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

__all__ = [
    "RealisticCostModel",
    "corwin_schultz_spread",
    "abdi_ranaldo_spread",
    "estimate_spread_panel",
]


# ── Corwin–Schultz high/low spread estimator ─────────────────────────────────

def corwin_schultz_spread(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Two-day high/low bid-ask spread estimate, as a fraction of price.

    Corwin & Schultz (2012): the high/low ratio over one day reflects both the
    true variance and the spread; over two consecutive days the variance
    component doubles while the spread component does not, so the two can be
    separated.  Returns a per-row array aligned to ``high``/``low`` (the first
    element is NaN — the estimator needs a previous day).

    Negative estimates (noise when the true spread is near zero) are floored at
    0, which is the standard treatment.
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    n = len(high)
    out = np.full(n, np.nan)
    if n < 2:
        return out

    with np.errstate(divide="ignore", invalid="ignore"):
        beta = np.log(high[1:] / low[1:]) ** 2 + np.log(high[:-1] / low[:-1]) ** 2
        h2 = np.maximum(high[1:], high[:-1])
        l2 = np.minimum(low[1:], low[:-1])
        gamma = np.log(h2 / l2) ** 2

        k = 3.0 - 2.0 * np.sqrt(2.0)
        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
        alpha = np.maximum(alpha, 0.0)
        spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))

    out[1:] = spread
    return out


def abdi_ranaldo_spread(
    close: np.ndarray, high: np.ndarray, low: np.ndarray, window: int = 21
) -> np.ndarray:  # noqa: E741 - `l` mirrors the paper's notation for the low
    """Abdi–Ranaldo close-high-low spread estimate, as a fraction of price.

    The efficient price is proxied by the mid-range ``eta = (log h + log l)/2``.
    Because the bid-ask bounce pushes the close away from the mid-range in a way
    that reverses next day, the covariance of consecutive deviations identifies
    the spread:

        s^2 = 4 * E[(c_t - eta_t)(c_t - eta_{t+1})]

    Averaged over a trailing ``window`` and floored at zero.  Uses only past and
    same-day bars at every point, so it is safe inside a walk-forward.
    """
    c = np.log(np.asarray(close, dtype=float))
    h = np.log(np.asarray(high, dtype=float))
    lo = np.log(np.asarray(low, dtype=float))
    n = len(c)
    out = np.full(n, np.nan)
    if n < window + 2:
        return out

    eta = (h + lo) / 2.0
    x = np.full(n, np.nan)
    with np.errstate(invalid="ignore"):
        x[:-1] = 4.0 * (c[:-1] - eta[:-1]) * (c[:-1] - eta[1:])
    smooth = (pl.Series(x)
              .rolling_mean(window, min_samples=max(5, window // 3))
              .to_numpy())
    return np.sqrt(np.maximum(smooth, 0.0))


def estimate_spread_panel(
    ohlcv: pl.DataFrame,
    method: str = "abdi_ranaldo",
    smooth_days: int = 21,
    floor_bps: float = 1.0,
    cap_bps: float = 500.0,
) -> pl.DataFrame:
    """Per (ticker, date) proportional spread estimate in **bps**.

    ``method`` is ``"abdi_ranaldo"`` (default) or ``"corwin_schultz"``.  Both
    estimators are noisy day to day, so the result is smoothed with a trailing
    (causal) mean and clipped into a sane band.  Output columns: ``date``,
    ``ticker``, ``spread_bps``.
    """
    need = {"date", "ticker", "high", "low", "close"}
    missing = need - set(ohlcv.columns)
    if missing:
        raise ValueError(f"estimate_spread_panel needs columns {sorted(missing)}")

    df = ohlcv.select(["date", "ticker", "close", "high", "low"]).sort(["ticker", "date"])
    parts = []
    for (_tkr,), g in df.group_by(["ticker"], maintain_order=True):
        if method == "corwin_schultz":
            s = corwin_schultz_spread(g["high"].to_numpy(), g["low"].to_numpy())
        elif method == "abdi_ranaldo":
            s = abdi_ranaldo_spread(g["close"].to_numpy(), g["high"].to_numpy(),
                                    g["low"].to_numpy(), window=smooth_days)
        else:
            raise ValueError(f"unknown spread method {method!r}")
        parts.append(g.select(["date", "ticker"]).with_columns(
            pl.Series("_raw", s * 1e4)))
    out = pl.concat(parts) if parts else df.select(["date", "ticker"]).with_columns(
        pl.lit(None, dtype=pl.Float64).alias("_raw"))

    # NaN and null are different things in Polars and both occur here: a name's
    # warm-up rows are null, while a window whose estimator output was all-NaN
    # stays NaN through the rolling mean.  Only handling nulls would leave NaN
    # spreads in the panel, which the simulator would silently floor.  Fill NaN
    # first, then carry the last known spread forward within the ticker, then
    # fall back to the floor for names that never had one.
    return (
        out.sort(["ticker", "date"])
        .with_columns(
            pl.col("_raw")
            .rolling_mean(smooth_days, min_samples=3)
            .over("ticker")
            .alias("spread_bps")
        )
        .with_columns(
            pl.col("spread_bps")
            .fill_nan(None)
            .forward_fill()
            .over("ticker")
            .alias("spread_bps")
        )
        .with_columns(
            pl.col("spread_bps").fill_null(floor_bps).clip(floor_bps, cap_bps)
        )
        .select(["date", "ticker", "spread_bps"])
    )


# ── Per-order cost model ─────────────────────────────────────────────────────

@dataclass
class RealisticCostModel:
    """Per-name execution cost.  All rates in bps of traded notional.

    Parameters
    ----------
    commission_bps:
        Broker commission / exchange fees, charged on every dollar traded.
    spread_mult:
        Fraction of the estimated *full* spread paid on a trade.  0.5 = pay the
        half-spread (a marketable limit order at the touch); >0.5 models a
        sweep.  This multiplies the Corwin–Schultz estimate, not a constant.
    impact_eta:
        Coefficient of the square-root impact law.  Empirical estimates cluster
        around 0.3–1.0 for US equities; 0.5 is the common mid-point.
    max_participation:
        Largest fraction of a name's dollar volume the book may trade in one
        day.  Orders above this are clipped and the remainder carried.
    min_spread_bps / max_spread_bps:
        Band applied to the supplied spread estimate.
    """

    commission_bps: float = 0.5
    spread_mult: float = 0.5
    impact_eta: float = 0.5
    max_participation: float = 0.05
    min_spread_bps: float = 1.0
    max_spread_bps: float = 500.0

    def fillable_notional(self, desired: np.ndarray, adv: np.ndarray) -> np.ndarray:
        """Clip each order to ``max_participation`` of that name's dollar volume."""
        cap = np.maximum(adv, 0.0) * self.max_participation
        return np.sign(desired) * np.minimum(np.abs(desired), cap)

    def cost_notional(
        self,
        traded: np.ndarray,
        adv: np.ndarray,
        spread_bps: np.ndarray,
        daily_vol: np.ndarray,
    ) -> np.ndarray:
        """Dollar cost of each filled order.

        ``traded`` is signed dollar notional; ``adv`` the name's average daily
        dollar volume; ``spread_bps`` its estimated proportional spread in bps;
        ``daily_vol`` its daily return standard deviation (not annualised).
        """
        q = np.abs(np.asarray(traded, dtype=float))
        adv = np.maximum(np.asarray(adv, dtype=float), 1.0)
        spr = np.clip(np.nan_to_num(np.asarray(spread_bps, dtype=float),
                                    nan=self.min_spread_bps),
                      self.min_spread_bps, self.max_spread_bps)
        vol = np.nan_to_num(np.asarray(daily_vol, dtype=float), nan=0.02)

        linear_bps = self.commission_bps + self.spread_mult * spr
        # square-root law: impact grows with sqrt(participation), scaled by vol
        impact_bps = self.impact_eta * vol * np.sqrt(q / adv) * 1e4
        return q * (linear_bps + impact_bps) / 1e4

    def scaled(self, factor: float) -> "RealisticCostModel":
        """A stressed copy: every cost rate multiplied by ``factor``.

        Used for the cost ladder — a strategy whose edge dies at 3x costs is not
        an edge, it is a cost-model assumption.
        """
        return RealisticCostModel(
            commission_bps=self.commission_bps * factor,
            spread_mult=self.spread_mult * factor,
            impact_eta=self.impact_eta * factor,
            max_participation=self.max_participation,
            min_spread_bps=self.min_spread_bps,
            max_spread_bps=self.max_spread_bps,
        )
