"""Measuring the survivorship bias in this universe, rather than asserting it.

The repo has always carried the caveat that the universe is *today's* liquid
names, so backtest returns are an upper bound.  A caveat is not a number, and
without a number there is no way to know whether the strategy's edge is real or
is the bias.

The measurement does not need delisted price data, which is the thing free
sources will not provide (yfinance and Stooq both return nothing for ATVI,
TWTR, FRC, SIVB, XLNX or RTN).  It only needs a *survivorship-free equivalent*
of the universe, and one exists as a tradeable product: the equal-weight S&P 500
(``RSP``, live since 2003) holds whatever was in the index at the time,
including everything that later failed or was acquired.

So:

    bias = equal-weight return of today's universe  -  equal-weight index return

Both sides are equal-weighted, long-only, US large cap, over identical dates.
The difference is what selecting the universe after the fact is worth.  On this
panel, 2005-2026, it is about **8 percentage points of CAGR and 0.36 of
Sharpe** — larger than most strategies' entire claimed edge.

The right way to read every backtest in this repo follows from that: compare a
strategy to ``universe_ew``, not to SPY.  Beating SPY on a survivorship-biased
universe is the null result, not the finding.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from ..utils import get_logger
from .stats import sharpe

logger = get_logger(__name__)

__all__ = ["BiasEstimate", "fetch_reference_returns", "estimate_survivorship_bias"]

# Equal-weight first: it is the like-for-like comparison for an equal-weighted
# book of large caps. The cap-weighted names are context, not the benchmark.
REFERENCE_TICKERS = {
    "RSP": "S&P 500 equal weight (survivorship-free, 2003+)",
    "SPY": "S&P 500 cap weight",
    "VTI": "total US market",
    "IWM": "Russell 2000 small cap",
}


@dataclass
class BiasEstimate:
    start: str
    end: str
    years: float
    universe_cagr: float
    universe_sharpe: float
    reference: str
    reference_cagr: float
    reference_sharpe: float

    @property
    def cagr_bias(self) -> float:
        return self.universe_cagr - self.reference_cagr

    @property
    def sharpe_bias(self) -> float:
        return self.universe_sharpe - self.reference_sharpe

    def render(self) -> str:
        return (
            f"Survivorship bias, {self.start} to {self.end} ({self.years:.1f}y)\n"
            f"  equal-weight of today's universe : CAGR {self.universe_cagr:+.2%}  "
            f"Sharpe {self.universe_sharpe:.2f}\n"
            f"  {self.reference:<32} : CAGR {self.reference_cagr:+.2%}  "
            f"Sharpe {self.reference_sharpe:.2f}\n"
            f"  bias                             : CAGR {self.cagr_bias:+.2%}  "
            f"Sharpe {self.sharpe_bias:+.2f}\n"
            f"  -> compare strategies to the universe, not to the index: the "
            f"universe already contains this much unearned return."
        )


def fetch_reference_returns(
    start: date, end: date, tickers: list[str] | None = None
) -> dict[str, "object"]:
    """Daily total returns for the reference index products, from yfinance.

    These are the survivorship-free side of the comparison: an index ETF held
    whatever the index held at the time, so its record includes the constituents
    that later disappeared.
    """
    import yfinance as yf

    tickers = tickers or list(REFERENCE_TICKERS)
    px = yf.download(tickers, start=str(start), end=str(end),
                     progress=False, auto_adjust=True, threads=False)["Close"]
    if hasattr(px, "to_frame") and px.ndim == 1:
        px = px.to_frame(tickers[0])
    return px.dropna()


def estimate_survivorship_bias(
    universe_returns: np.ndarray,
    dates: list,
    reference: str = "RSP",
) -> BiasEstimate | None:
    """Compare an equal-weight universe return stream to a real index product.

    ``universe_returns`` should be the daily equal-weight return of every live
    name in the configured universe — what
    :func:`trading_system.research.runner.passive_benchmarks` calls
    ``universe_ew``.  Returns ``None`` if the reference cannot be fetched.
    """
    if len(universe_returns) < 252:
        return None
    start, end = dates[0], dates[-1]
    try:
        px = fetch_reference_returns(start, end, [reference])
    except Exception as e:  # noqa: BLE001
        logger.warning(f"could not fetch {reference}: {e}")
        return None
    if px.empty or reference not in px.columns:
        return None

    ref = px[reference].pct_change().dropna().to_numpy()
    n = min(len(ref), len(universe_returns))
    if n < 252:
        return None
    ref, uni = ref[-n:], np.asarray(universe_returns)[-n:]
    yrs = n / 252

    def cagr(r):
        return float(np.prod(1 + r) ** (1 / yrs) - 1)

    return BiasEstimate(
        start=str(start), end=str(end), years=yrs,
        universe_cagr=cagr(uni), universe_sharpe=sharpe(uni),
        reference=f"{reference} — {REFERENCE_TICKERS.get(reference, '')}",
        reference_cagr=cagr(ref), reference_sharpe=sharpe(ref),
    )
