"""Unified price-bound engine: calibrated quantiles first, MC fan as fallback.

Resolution order for "how high can it go / how low can it drop" over
days → months → a year:

  1. **Conformalized quantile bundle** (``models/intervals.py``) if one is
     trained — distribution-free coverage guarantee, asymmetric, SHAP-explainable.
  2. **Monte-Carlo fan** otherwise — bootstrap the ticker's own daily returns
     (real skew/fat-tails), scale the spread to **option-implied vol** when
     available (forward-looking), and tilt the drift gently toward the model's
     5-day view.

Both paths emit the same shape so the analyzer, report, and dashboard don't care
which produced the numbers.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl

from ..utils import get_logger

logger = get_logger(__name__)

_HORIZON_LABEL = {5: "5d", 21: "1m", 63: "3m", 126: "6m", 252: "12m"}
_TRADING_DAYS_PER_CAL_DAY = 252 / 365.0


def _historical_daily_logrets(ohlcv: pl.DataFrame, ticker: str, lookback: int = 504) -> np.ndarray:
    sub = (
        ohlcv.filter(pl.col("ticker") == ticker.upper())
        .sort("date")
        .tail(lookback + 1)
        .select("adj_close")
        .drop_nulls()
    )
    px = sub["adj_close"].to_numpy().astype(np.float64)
    if len(px) < 30:
        return np.array([])
    return np.diff(np.log(px))


def _monte_carlo_bounds(
    ticker: str,
    last_price: float,
    daily_logrets: np.ndarray,
    score_5d: float,
    horizons: list[int],
    quantiles: list[float],
    iv_term: list[tuple[int, float]] | None,
    n_paths: int,
    seed: int = 7,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    base = daily_logrets - daily_logrets.mean()  # de-mean; we add our own drift
    realized_daily = float(base.std()) or 0.01

    # Gentle drift toward the model's 5d view (capped so a year doesn't explode)
    daily_drift = float(np.clip(score_5d, -0.02, 0.02)) / 5.0 * 0.5

    max_h = max(horizons)
    # Per-horizon IV scaling factor (forward-looking vol / realised vol)
    from ..models.implied_vol import iv_for_horizon

    horizons_out: dict[str, Any] = {}
    # Simulate once to max horizon, snapshot at each checkpoint
    # Resample daily shocks with replacement (bootstrap keeps skew & fat tails)
    shocks = rng.choice(base, size=(n_paths, max_h), replace=True)

    for h in horizons:
        scale = 1.0
        if iv_term:
            cal_days = int(round(h / _TRADING_DAYS_PER_CAL_DAY))
            iv = iv_for_horizon(iv_term, cal_days)
            if iv and realized_daily > 1e-6:
                iv_daily = iv / math.sqrt(252)
                scale = float(np.clip(iv_daily / realized_daily, 0.5, 3.0))
        path = shocks[:, :h] * scale + daily_drift
        terminal = last_price * np.exp(path.sum(axis=1))
        qs = np.quantile(terminal, quantiles)
        median = float(np.quantile(terminal, 0.5))
        lo = float(qs[0]); hi = float(qs[-1])
        horizons_out[_HORIZON_LABEL.get(h, f"{h}d")] = {
            "days": h,
            "price": {
                "lo": round(lo, 4),
                "median": round(median, 4),
                "hi": round(hi, 4),
                "q25": round(float(np.quantile(terminal, 0.25)), 4),
                "q75": round(float(np.quantile(terminal, 0.75)), 4),
            },
            "return": {
                "lo": round(lo / last_price - 1, 5),
                "median": round(median / last_price - 1, 5),
                "hi": round(hi / last_price - 1, 5),
            },
            "iv_scale": round(scale, 3),
        }
    return {
        "ticker": ticker.upper(),
        "last_price": round(last_price, 4),
        "method": "monte_carlo_bootstrap" + ("_iv" if iv_term else "_realized"),
        "horizons": horizons_out,
    }


def compute_bounds(
    cfg,
    ticker: str,
    features: pl.DataFrame,
    ohlcv: pl.DataFrame,
    last_price: float,
    score_5d: float,
) -> dict[str, Any] | None:
    """Compute multi-horizon price bounds for a ticker. Returns None on failure."""
    if not last_price or last_price <= 0:
        return None
    bcfg = cfg.get("bounds", {}) or {}
    horizons = list(bcfg.get("horizons_days", [5, 21, 63, 126, 252]))
    quantiles = list(bcfg.get("quantiles", [0.05, 0.25, 0.5, 0.75, 0.95]))
    use_iv = bool(bcfg.get("use_implied_vol", True))

    # IV term structure (best-effort, shared by both paths)
    iv_term = None
    if use_iv:
        try:
            from ..models.implied_vol import atm_iv_term_structure
            iv_term = atm_iv_term_structure(
                ticker, spot=last_price,
                cache_dir=cfg.path("data_silver") / "iv_cache",
            ) or None
        except Exception as e:
            logger.debug(f"IV fetch failed for {ticker}: {e}")

    # 1. Calibrated quantile bundle — prefer the committed store, then reports/
    try:
        from ..models.intervals import load_interval_bundle, bounds_for_ticker
        bundle = None
        try:
            store_intervals = cfg.project_root / "models_store"
            bundle = load_interval_bundle(store_intervals)
        except Exception:
            bundle = None
        if bundle is None:
            bundle = load_interval_bundle(cfg.path("reports") / "models")
        if bundle is not None:
            out = bounds_for_ticker(bundle, features, ticker, last_price)
            if out:
                if iv_term:
                    cal = int(round(21 / _TRADING_DAYS_PER_CAL_DAY))
                    from ..models.implied_vol import iv_for_horizon
                    out["implied_vol_1m"] = round(iv_for_horizon(iv_term, cal) or 0.0, 4)
                return out
    except Exception as e:
        logger.warning(f"quantile bounds failed for {ticker}, using MC fallback: {e}")

    # 2. Monte-Carlo fallback
    rets = _historical_daily_logrets(ohlcv, ticker)
    if rets.size < 30:
        return None
    return _monte_carlo_bounds(
        ticker, last_price, rets, score_5d, horizons, quantiles, iv_term,
        n_paths=int(bcfg.get("mc_paths", 2000)),
    )
