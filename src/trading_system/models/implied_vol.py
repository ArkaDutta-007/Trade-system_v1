"""Option-implied volatility from yfinance — the market's *forward* view.

Realised vol (what the old fan used) is backward-looking; option-implied vol is
the market's forecast of future movement and is the right width for a forward
price band.  This pulls the ATM IV term structure from the free yfinance option
chain and interpolates it to any horizon.

Everything is best-effort and disk-cached: no options data (ETFs, illiquid names,
network down) simply returns ``None`` and callers fall back to realised vol.
"""
from __future__ import annotations

import json
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np

from ..utils import get_logger

logger = get_logger(__name__)


def _atm_iv_for_expiry(tk, expiry: str, spot: float) -> float | None:
    """Average IV of the strikes nearest the money for one expiry."""
    try:
        chain = tk.option_chain(expiry)
    except Exception:
        return None
    ivs: list[float] = []
    for leg in (chain.calls, chain.puts):
        if leg is None or len(leg) == 0 or "impliedVolatility" not in leg:
            continue
        leg = leg.dropna(subset=["impliedVolatility", "strike"])
        if leg.empty:
            continue
        # nearest 5 strikes to spot
        leg = leg.assign(_dist=(leg["strike"] - spot).abs()).nsmallest(5, "_dist")
        vals = [float(v) for v in leg["impliedVolatility"].tolist() if 0.01 < float(v) < 5.0]
        ivs.extend(vals)
    if not ivs:
        return None
    return float(np.median(ivs))


def atm_iv_term_structure(
    ticker: str,
    spot: float | None = None,
    max_expiries: int = 6,
    cache_dir: Path | None = None,
    cache_hours: float = 12.0,
) -> list[tuple[int, float]]:
    """Return [(calendar_days_to_expiry, annualised_iv), ...] sorted by tenor.

    Empty list if no options data is available.
    """
    ticker = ticker.upper()
    cache_file = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"iv_{ticker}_{date.today().isoformat()}.json"
        if cache_file.exists():
            age_h = (time.time() - cache_file.stat().st_mtime) / 3600.0
            if age_h <= cache_hours:
                try:
                    return [tuple(x) for x in json.loads(cache_file.read_text())]
                except Exception:
                    pass

    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        if spot is None:
            hist = tk.history(period="1d")
            spot = float(hist["Close"].iloc[-1]) if len(hist) else None
        if not spot:
            return []
        expiries = list(tk.options or [])[:max_expiries]
    except Exception as e:
        logger.debug(f"IV term structure unavailable for {ticker}: {e}")
        return []

    today = date.today()
    term: list[tuple[int, float]] = []
    for exp in expiries:
        try:
            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
        except Exception:
            continue
        dte = (exp_date - today).days
        if dte <= 0:
            continue
        iv = _atm_iv_for_expiry(tk, exp, spot)
        if iv is not None:
            term.append((dte, iv))

    term.sort(key=lambda x: x[0])
    if cache_file and term:
        try:
            cache_file.write_text(json.dumps(term))
        except Exception:
            pass
    return term


def iv_for_horizon(term: list[tuple[int, float]], horizon_cal_days: int) -> float | None:
    """Interpolate annualised IV at a calendar-day horizon. None if no term data."""
    if not term:
        return None
    xs = [d for d, _ in term]
    ys = [v for _, v in term]
    if horizon_cal_days <= xs[0]:
        return ys[0]
    if horizon_cal_days >= xs[-1]:
        return ys[-1]
    return float(np.interp(horizon_cal_days, xs, ys))
