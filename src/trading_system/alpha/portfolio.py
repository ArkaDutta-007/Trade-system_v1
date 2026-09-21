"""Book construction: calibrated forecasts → a constrained, risk-managed long-only book.

The rules are deliberately few and each one answers a measured failure of the
old picks:

* rank on **risk-adjusted** score (demeaned score / vol^0.5) — the raw ranking
  measurably preferred volatile, illiquid names;
* **liquidity/price gates** before ranking, not after;
* **sector cap** so the book is not one theme expressed twenty ways;
* **inverse-vol × equal-weight blend**, single-name cap, so no 20% positions;
* **portfolio vol target** (gross exposure scales down when the book's
  estimated vol exceeds the target — cash is a position) and a **200-day
  trend overlay** (half gross while the market index is below its 200-day
  average; the single change that cut max drawdown from −52% to −33%);
* **partial trading** toward the target (Gârleanu–Pedersen), which the causal
  research found to be the single biggest improvement in net Sharpe.

``target_weights`` has the ``(scores, cols, cfg)`` signature the research
simulator expects, so the *same* function builds the backtest book and the
live book.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

_ANN = 252 ** 0.5


@dataclass
class BookConfig:
    top_k: int = 20
    max_weight: float = 0.08
    gross_exposure: float = 1.0
    vol_power: float = 0.5           # rank on demeaned score / vol**vol_power
    max_per_sector: int = 5
    inv_vol_blend: float = 0.5       # 1.0 = pure inverse-vol, 0.0 = equal weight
    vol_target: float | None = 0.25  # annualised crisis brake (the constant-corr estimate runs low; binds rarely)
    avg_corr: float = 0.30           # for the closed-form portfolio-vol estimate
    min_price: float = 5.0
    min_dollar_volume: float = 20e6
    max_daily_vol: float = 0.07      # ≈110% annualised
    trade_rate: float = 0.35         # partial trading toward target per rebalance
    no_trade_band: float = 0.20      # skip trades smaller than this fraction of the target weight
    regime_scale: float = 0.5        # gross multiplier when the market trend is "off" (1.0 = no overlay)
    stress_overlay: bool = False     # also scale gross by the stress score — tested 2004→26: adds nothing over the trend rule
    min_gross_mult: float = 0.35     # floor for the combined overlay multiplier


def portfolio_vol(w: np.ndarray, dvol: np.ndarray, avg_corr: float) -> float:
    """Annualised vol under a constant-correlation model (fast, no covariance to estimate)."""
    s = w * dvol
    var = (1 - avg_corr) * float((s ** 2).sum()) + avg_corr * float(s.sum()) ** 2
    return float(np.sqrt(max(var, 0.0))) * _ANN


def _cap(w: np.ndarray, cap: float, rounds: int = 10) -> np.ndarray:
    for _ in range(rounds):
        over = w > cap
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w = np.where(over, cap, w)
        free = (w > 0) & ~over
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()
    return w


def target_weights(scores: np.ndarray, cols: dict, cfg: BookConfig, sectors: np.ndarray | None = None,
                   regime_on: bool = True, gross_mult: float = 1.0) -> np.ndarray:
    """Scores + per-name context → target weights (same length as ``scores``).

    ``cols`` needs ``dvol`` (daily return vol); ``adv`` and ``price`` are used for the gates when
    present (the research simulator applies its own gates before calling). ``regime_on=False``
    multiplies the gross by ``cfg.regime_scale`` (the trend overlay) and ``gross_mult`` by the caller's
    stress/fragility multiplier; the product is floored at ``cfg.min_gross_mult``.
    """
    s = np.asarray(scores, dtype=float)
    dvol = np.asarray(cols["dvol"], dtype=float)
    n = len(s)
    ok = np.isfinite(s) & np.isfinite(dvol) & (dvol > 0)
    if "adv" in cols:
        ok &= np.asarray(cols["adv"], dtype=float) >= cfg.min_dollar_volume
    if "price" in cols:
        ok &= np.asarray(cols["price"], dtype=float) >= cfg.min_price
    ok &= dvol <= cfg.max_daily_vol
    w = np.zeros(n)
    if ok.sum() < max(3, cfg.top_k // 4):
        return w
    key = np.full(n, -np.inf)
    key[ok] = (s[ok] - s[ok].mean()) / np.maximum(dvol[ok], 1e-4) ** cfg.vol_power
    order = np.argsort(-key)
    picked: list[int] = []
    per_sector: dict = {}
    for j in order:
        if not np.isfinite(key[j]):
            break
        if sectors is not None:
            sec = sectors[j]
            if per_sector.get(sec, 0) >= cfg.max_per_sector:
                continue
            per_sector[sec] = per_sector.get(sec, 0) + 1
        picked.append(j)
        if len(picked) >= cfg.top_k:
            break
    if not picked:
        return w
    idx = np.array(picked)
    inv = 1.0 / np.maximum(dvol[idx], 1e-4)
    inv = inv / inv.sum()
    eq = np.full(len(idx), 1.0 / len(idx))
    w[idx] = cfg.inv_vol_blend * inv + (1 - cfg.inv_vol_blend) * eq
    w = _cap(w, cfg.max_weight)
    w = w / w.sum() * cfg.gross_exposure
    if cfg.vol_target:
        pv = portfolio_vol(w, dvol, cfg.avg_corr)
        if pv > cfg.vol_target:
            w = w * (cfg.vol_target / pv)
    mult = (cfg.regime_scale if not regime_on else 1.0) * float(gross_mult)
    mult = max(min(mult, 1.0), cfg.min_gross_mult) if mult < 1.0 else 1.0
    return w * mult


def partial_rebalance(current: np.ndarray, target: np.ndarray, cfg: BookConfig) -> np.ndarray:
    """Move ``trade_rate`` of the way toward ``target``; exits complete once tiny; small moves skipped."""
    cur = np.asarray(current, dtype=float)
    tgt = np.asarray(target, dtype=float)
    new = cur + cfg.trade_rate * (tgt - cur)
    small = np.abs(tgt - cur) <= cfg.no_trade_band * np.maximum(tgt, 1e-9)
    new = np.where(small & (tgt > 0), cur, new)
    new = np.where((tgt <= 0) & (new < 0.005), 0.0, new)
    return new


def market_regime_on(mkt_trend_200: float | None) -> bool:
    """The overlay's single rule (Faber-style): risk-on while the equal-weight market index sits at or
    above its 200-day average; below it the book runs at ``regime_scale`` of its gross.
    On 2004→2026 this took the book from −52% to −33% max drawdown at a 1pp CAGR cost."""
    if mkt_trend_200 is None or not np.isfinite(mkt_trend_200):
        return True
    return mkt_trend_200 >= 0.0


def build_book(today: pl.DataFrame, cfg: BookConfig, prev: dict[str, float] | None = None,
               regime_on: bool = True, gross_mult: float = 1.0) -> pl.DataFrame:
    """One date's cross-section (``ticker, composite, dvol, adv, price, sector, exp_ret?, q10?, q90?``)
    → target + traded weights. ``prev`` = current weights by ticker (for partial trading)."""
    t = today.sort("ticker")
    scores = t["composite"].to_numpy()
    cols = {"dvol": t["dvol"].to_numpy(), "adv": t["adv"].to_numpy(), "price": t["price"].to_numpy()}
    sectors = t["sector"].to_numpy() if "sector" in t.columns else None
    tgt = target_weights(scores, cols, cfg, sectors, regime_on=regime_on, gross_mult=gross_mult)
    tickers = t["ticker"].to_list()
    cur = np.array([float((prev or {}).get(tk, 0.0)) for tk in tickers])
    new = partial_rebalance(cur, tgt, cfg) if prev is not None else tgt
    out = t.with_columns(target_weight=pl.Series(tgt), weight=pl.Series(new), prev_weight=pl.Series(cur))
    out = out.filter((pl.col("target_weight") > 0) | (pl.col("weight") > 0) | (pl.col("prev_weight") > 0))
    key = ((pl.col("composite") - pl.col("composite").mean()) / pl.col("dvol").pow(cfg.vol_power))
    return out.with_columns(rank_key=key).sort(["target_weight", "rank_key"], descending=[True, True])
