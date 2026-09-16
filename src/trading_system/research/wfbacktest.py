"""Strictly-causal walk-forward portfolio simulator.

Why this exists
---------------
The repo already validates *models* honestly: ``models/validation.py`` gives
purged, embargoed walk-forward and CPCV splits with a label-shuffle gate and a
deflated-ICIR haircut.  What it does not give is an honest estimate of what the
*book* would have earned, and the gap between the two is large enough to change
every conclusion:

1. **CPCV is not causal.**  It holds out a block in 2018 and trains on 2011-2017
   *and 2019-2024*.  That is the right way to get a distribution of IC for model
   selection and the wrong way to estimate P&L — no live book has 2024 data in
   2018.  The existing P&L numbers are built from CPCV out-of-sample predictions,
   so they are an upper bound of unknown tightness.
2. **Model selection itself leaks.**  Picking "best by ICIR" using the full
   sample, then backtesting that winner on the same sample, embeds the choice in
   the result.  Here the selection happens inside each refit, on past data only.
3. **Costs were portfolio-level.**  See :mod:`trading_system.research.costs`.

This module simulates the whole stack forward in time:

    for each rebalance date t:
        if the model is stale:
            refit on rows whose labels completed on or before t - purge
        score the cross-section using only features known at t
        build target weights
        trade toward them at t+1, paying per-name spread + impact,
        clipped to a participation cap
    hold, letting positions drift with prices, until the next rebalance

Nothing in the loop can see past ``t``.  The output is a daily equity curve, a
trade blotter and a weight history, so every number downstream (Sharpe, DSR,
PBO, turnover, cost drag) is computed on a stream a live book could have
produced.

The alpha model is injected, so the same protocol measures the GBM ensemble, a
rule baseline, the cross-sectional attention model and an RL policy under
identical conditions.  That comparability is the point: a new model is only
better if it wins *here*.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Protocol

import numpy as np
import polars as pl

from ..utils import get_logger
from .costs import RealisticCostModel, estimate_spread_panel

logger = get_logger(__name__)

_CAL_PER_TD = 365 / 252

__all__ = [
    "AlphaModel",
    "WalkForwardConfig",
    "WalkForwardResult",
    "PanelData",
    "run_walk_forward",
]


# ── Alpha model protocol ─────────────────────────────────────────────────────

class AlphaModel(Protocol):
    """What the simulator needs from any signal source.

    ``fit`` sees only rows dated at or before the refit cut-off (labels already
    realised).  ``predict`` sees one date's cross-section.  Implementations live
    in :mod:`trading_system.research.alphas`.
    """

    name: str

    def fit(self, panel: pl.DataFrame, feat_cols: list[str], target: str) -> None: ...

    def predict(self, today: pl.DataFrame, feat_cols: list[str]) -> np.ndarray: ...


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class WalkForwardConfig:
    """Everything that defines one simulated book.

    Defaults describe a long-only, monthly-rebalanced, 20-name book with a 10%
    position cap — the shape the repo's `ts picks` already produces — so the
    first run is a like-for-like replacement of the leaky backtest rather than a
    different strategy.
    """

    # ── timing ──
    oos_start: date = date(2005, 1, 1)
    oos_end: date | None = None
    rebalance_days: int = 21          # trading days between rebalances
    retrain_days: int = 63            # trading days between model refits
    min_train_days: int = 756         # ~3y of dates before the first refit
    train_window_days: int | None = None   # None = expanding; else rolling
    embargo_days: int = 5             # extra calendar days purged before the cut

    # ── label ──
    horizon: int = 63                 # trading days the alpha target looks ahead
    neutralize: bool = True           # demean the target within each date

    # ── book construction ──
    top_k: int = 20
    max_weight: float = 0.10
    gross_exposure: float = 1.0
    long_only: bool = True
    risk_adjust: bool = True          # rank on demeaned score / vol**0.5
    vol_power: float = 0.5
    min_dollar_volume: float = 2_000_000.0
    min_price: float = 3.0

    # ── execution ──
    trade_delay_days: int = 1         # decide at t, trade at t+1
    band_entry: float = 0.0           # no-trade band, fraction of target weight
    band_exit: float = 0.0
    partial_trade_rate: float = 1.0   # 1.0 = go all the way to target
    min_position_weight: float = 0.005  # below this an unwanted holding is closed
    cost: RealisticCostModel = field(default_factory=RealisticCostModel)

    initial_cash: float = 1_000_000.0
    seed: int = 0

    def purge_calendar_days(self) -> int:
        return int(round(self.horizon * _CAL_PER_TD)) + self.embargo_days


# ── Prepared panel ───────────────────────────────────────────────────────────

@dataclass
class PanelData:
    """Wide, date-aligned arrays the simulator needs, built once.

    Keeping these as dense ``(n_dates, n_tickers)`` float arrays makes the daily
    loop pure NumPy.  ``alive`` marks (date, ticker) cells with a real price;
    everything else is masked out of both trading and return accrual, so a name
    that has not listed yet — or has stopped trading — can never contribute P&L.
    """

    dates: np.ndarray            # (T,) datetime.date
    tickers: np.ndarray          # (N,) str
    ret: np.ndarray              # (T, N) simple daily total return
    price: np.ndarray            # (T, N) adjusted close
    adv: np.ndarray              # (T, N) trailing 20d average dollar volume
    spread_bps: np.ndarray       # (T, N) Abdi-Ranaldo estimate
    dvol: np.ndarray             # (T, N) trailing 20d daily return vol
    alive: np.ndarray            # (T, N) bool

    def date_index(self) -> dict[date, int]:
        return {d: i for i, d in enumerate(self.dates)}


def _to_date(v) -> date:
    if isinstance(v, date):
        return v
    if hasattr(v, "date"):
        return v.date()
    return date.fromisoformat(str(v)[:10])


def build_panel(
    ohlcv: pl.DataFrame,
    spread: pl.DataFrame | None = None,
    trim_sparse_tail: float = 0.5,
) -> PanelData:
    """Pivot bronze OHLCV into the dense arrays the simulator consumes.

    ``trim_sparse_tail`` drops trailing dates whose live-name count is below
    this fraction of the recent median.  The last row of a freshly ingested
    panel is usually a partial trading day — on this data 3 of 362 names had
    printed — and a partial day is not a day the book could have traded through.
    Left in, it reads as 359 simultaneous delistings.
    """
    px = (
        ohlcv.select(["date", "ticker", "adj_close", "close", "volume", "high", "low"])
        .sort(["ticker", "date"])
        .with_columns(
            ret=pl.col("adj_close").pct_change().over("ticker"),
            dollar_vol=pl.col("close") * pl.col("volume"),
        )
        .with_columns(
            adv20=pl.col("dollar_vol").rolling_mean(20, min_samples=5).over("ticker"),
            dvol20=pl.col("ret").rolling_std(20, min_samples=5).over("ticker"),
        )
    )
    if spread is None:
        spread = estimate_spread_panel(ohlcv)
    px = px.join(spread, on=["date", "ticker"], how="left")

    def wide(col: str) -> tuple[np.ndarray, list[date], list[str]]:
        w = px.pivot(values=col, index="date", on="ticker",
                     aggregate_function="first").sort("date")
        tick = [c for c in w.columns if c != "date"]
        return w.select(tick).to_numpy(), [_to_date(d) for d in w["date"]], tick

    price, dates, tickers = wide("adj_close")
    ret, _, _ = wide("ret")
    adv, _, _ = wide("adv20")
    spr, _, _ = wide("spread_bps")
    dv, _, _ = wide("dvol20")

    alive = np.isfinite(price) & (price > 0)
    ret = np.nan_to_num(ret, nan=0.0, posinf=0.0, neginf=0.0)
    ret[~alive] = 0.0

    keep = len(dates)
    if trim_sparse_tail > 0 and keep > 80:
        n_live = alive.sum(axis=1)
        ref = np.median(n_live[max(0, keep - 60):keep])
        while keep > 1 and n_live[keep - 1] < trim_sparse_tail * ref:
            keep -= 1
        if keep < len(dates):
            logger.info(f"build_panel: dropped {len(dates) - keep} sparse trailing "
                        f"date(s) — last kept {dates[keep - 1]}")

    sl = slice(0, keep)
    return PanelData(
        dates=np.array(dates[:keep], dtype=object),
        tickers=np.array(tickers, dtype=object),
        ret=ret[sl],
        price=np.nan_to_num(price[sl], nan=0.0),
        adv=np.nan_to_num(adv[sl], nan=0.0),
        spread_bps=np.nan_to_num(spr[sl], nan=25.0),
        dvol=np.nan_to_num(dv[sl], nan=0.02),
        alive=alive[sl],
    )


# ── Result ───────────────────────────────────────────────────────────────────

@dataclass
class WalkForwardResult:
    daily: pl.DataFrame            # date, equity, net_ret, gross_ret, cost, turnover, n_pos
    trades: pl.DataFrame           # date, ticker, notional, cost, shortfall
    weights: pl.DataFrame          # date, ticker, weight (post-trade targets)
    refits: list[dict]             # one row per model refit: cut-off, rows, chosen model
    config: WalkForwardConfig
    label: str = ""

    def returns(self) -> np.ndarray:
        return self.daily["net_ret"].to_numpy()


# ── Weight construction ──────────────────────────────────────────────────────

def default_target_weights(
    scores: np.ndarray,
    cols: dict[str, np.ndarray],
    cfg: WalkForwardConfig,
) -> np.ndarray:
    """Scores → target weights for the eligible cross-section.

    The risk adjustment divides by volatility **after** demeaning the score
    across the date.  Skipping the demean is a live bug the ops tests caught:
    the cross-sectional mean of a return forecast is market beta, not alpha, so
    ranking ``score / vol**a`` un-demeaned degenerates into a pure low-vol
    screen that has nothing to do with the model.
    """
    s = np.asarray(scores, dtype=float)
    ok = np.isfinite(s)
    if ok.sum() == 0:
        return np.zeros_like(s)

    if cfg.risk_adjust:
        centred = s - np.nanmean(s[ok])
        vol = np.maximum(cols["dvol"], 1e-4)
        s = np.where(ok, centred / vol ** cfg.vol_power, -np.inf)
    else:
        s = np.where(ok, s, -np.inf)

    k = min(cfg.top_k, int(ok.sum()))
    if k <= 0:
        return np.zeros_like(s)
    pick = np.argpartition(-s, k - 1)[:k]
    pick = pick[np.isfinite(s[pick])]

    w = np.zeros_like(s)
    if len(pick) == 0:
        return w
    w[pick] = 1.0 / len(pick)

    # per-name cap with pro-rata redistribution onto the uncapped names
    for _ in range(8):
        over = w > cfg.max_weight
        if not over.any():
            break
        excess = (w[over] - cfg.max_weight).sum()
        w[over] = cfg.max_weight
        free = (w > 0) & ~over
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()

    tot = w.sum()
    return w * (cfg.gross_exposure / tot) if tot > 0 else w


# ── The simulator ────────────────────────────────────────────────────────────

def run_walk_forward(
    panel: PanelData,
    features: pl.DataFrame,
    feat_cols: list[str],
    model_factory: Callable[[], AlphaModel],
    cfg: WalkForwardConfig,
    weight_fn: Callable[..., np.ndarray] = default_target_weights,
    label: str = "",
    progress: bool = True,
) -> WalkForwardResult:
    """Run one causal walk-forward simulation.

    Parameters
    ----------
    panel:
        Price/liquidity arrays from :func:`build_panel`.
    features:
        Long feature matrix (``date``, ``ticker``, ``adj_close``, + ``feat_cols``).
    model_factory:
        Called to build a *fresh* model at every refit.  Fresh matters: reusing a
        fitted estimator across refits would carry future-trained state backwards
        on the next fold.
    weight_fn:
        ``(scores, cols, cfg) -> weights``; swap to test a different book
        construction on identical signals.
    """
    T, N = panel.ret.shape
    tix = {t: i for i, t in enumerate(panel.tickers)}

    tgt = f"_wf_fwd_{cfg.horizon}d"
    feats = features.sort(["ticker", "date"]).with_columns(
        ((pl.col("adj_close").shift(-cfg.horizon).over("ticker") / pl.col("adj_close")) - 1)
        .alias(tgt)
    )
    if cfg.neutralize:
        feats = feats.with_columns(
            (pl.col(tgt) - pl.col(tgt).mean().over("date")).alias(tgt))
    feats = feats.with_columns(pl.col("date").cast(pl.Date))

    # rows usable for scoring on a given date, indexed for fast slicing
    score_cols = ["date", "ticker", *feat_cols]
    score_frame = feats.select(score_cols).drop_nulls(subset=feat_cols)
    by_date = {
        _to_date(d): g for (d,), g in score_frame.group_by(["date"], maintain_order=True)
    }

    oos_end = cfg.oos_end or panel.dates[-1]
    all_idx = [i for i, d in enumerate(panel.dates) if cfg.oos_start <= d <= oos_end]
    if not all_idx:
        raise ValueError("no dates in the requested OOS window")
    start_i = all_idx[0]
    if start_i < cfg.min_train_days:
        raise ValueError(
            f"oos_start {cfg.oos_start} leaves only {start_i} training dates "
            f"(min_train_days={cfg.min_train_days}); move oos_start later")

    rebal_set = set(all_idx[:: cfg.rebalance_days])

    # A name is "delisted" on the first date after its last ever print — that is
    # the day its value converts to cash. Computing it once keeps the daily loop
    # from having to distinguish a missing bar from a permanent exit.
    delisted_on = np.zeros_like(panel.alive)
    for j in range(N):
        live = np.nonzero(panel.alive[:, j])[0]
        if len(live) and live[-1] + 1 < T:
            delisted_on[live[-1] + 1, j] = True

    pos = np.zeros(N)
    cash = cfg.initial_cash
    model: AlphaModel | None = None
    last_refit_i = -10**9
    refits: list[dict] = []
    trade_rows: list[dict] = []
    weight_rows: list[dict] = []
    daily_rows: list[dict] = []
    pending: np.ndarray | None = None   # target weights awaiting execution

    equity_prev = cfg.initial_cash

    for i in range(start_i, all_idx[-1] + 1):
        d = panel.dates[i]

        # 1. accrue: positions move with the day's returns.
        #    A missing bar is not a wipeout. ``ret`` is already 0 where a name did
        #    not print, so the position simply carries at its last value; only a
        #    name that never prints again is liquidated, and then into *cash* at
        #    its last mark. Zeroing the position instead — the obvious-looking
        #    line — deletes the money, and one partial trading day at the end of
        #    the panel then reads as every holding going to zero at once.
        pos = pos * (1.0 + panel.ret[i])
        gone = delisted_on[i]
        if gone.any():
            moved = pos[gone]
            if np.abs(moved).sum() > 0:
                cash += float(moved.sum())
                pos[gone] = 0.0
        equity = pos.sum() + cash
        if equity <= 0:
            logger.warning(f"{label}: equity wiped out at {d}")
            break

        gross_ret = equity / equity_prev - 1.0
        day_cost = 0.0
        turnover = 0.0

        # 2. execute anything decided on a previous day
        if pending is not None:
            cur_w = pos / equity

            # Bands and partial trading both modify the *weights*, and the
            # modified weights are then renormalised back to the gross target.
            # Doing it on dollar gaps instead (the obvious way) silently breaks
            # the budget: freezing some names inside the band while sending the
            # rest to target makes the weights stop summing to the gross, so the
            # book drifts into leverage or idle cash with no signal that
            # anything changed.
            w_adj = pending.copy()

            if cfg.band_entry > 0 or cfg.band_exit > 0:
                band = np.where(pending > 0, cfg.band_entry, cfg.band_exit)
                inside = np.abs(cur_w - pending) <= band * np.maximum(pending, 1e-9)
                # never band an exit: a held name the model dropped must be sold
                inside &= ~((np.abs(cur_w) > 1e-9) & (pending <= 1e-12))
                w_adj = np.where(inside, cur_w, w_adj)

            if cfg.partial_trade_rate < 1.0:
                w_adj = cur_w + cfg.partial_trade_rate * (w_adj - cur_w)

            # Dust cleanup. Partial trading shrinks an unwanted position
            # geometrically but never to zero, so without this the book
            # accumulates hundreds of sub-basis-point holdings — measured at 190
            # to 231 names for a nominally 20-name book — each still paying
            # spread every time it is touched.
            dust = (w_adj < cfg.min_position_weight) & (pending <= 1e-12)
            w_adj = np.where(dust, 0.0, w_adj)

            # A name with no bar today cannot be traded at all — not bought, and
            # not sold either. Its current weight is frozen and the rest of the
            # book is normalised over whatever gross is left, so an untradable
            # holding neither vanishes nor forces a fill at a price that does
            # not exist.
            tradable = panel.alive[i]
            frozen_w = np.where(tradable, 0.0, cur_w)
            budget = max(cfg.gross_exposure - frozen_w.sum(), 0.0)
            w_adj = np.where(tradable, w_adj, 0.0)
            tot = w_adj.sum()
            if tot > 1e-9:
                w_adj = w_adj * (budget / tot)
            w_adj = w_adj + frozen_w

            desired = np.where(tradable, w_adj * equity - pos, 0.0)
            filled = cfg.cost.fillable_notional(desired, panel.adv[i])
            costs = cfg.cost.cost_notional(
                filled, panel.adv[i], panel.spread_bps[i], panel.dvol[i])

            traded_mask = np.abs(filled) > 1e-9
            pos = pos + filled
            day_cost = float(costs.sum())
            cash = cash - float(filled.sum()) - day_cost
            turnover = float(np.abs(filled).sum() / equity)

            for j in np.nonzero(traded_mask)[0]:
                trade_rows.append({
                    "date": d, "ticker": panel.tickers[j],
                    "notional": float(filled[j]), "cost": float(costs[j]),
                    "shortfall": float(desired[j] - filled[j]),
                })
            equity = pos.sum() + cash
            pending = None

        # 3. decide (refit if stale, score, build targets)
        if i in rebal_set:
            if model is None or (i - last_refit_i) >= cfg.retrain_days:
                cut = d - timedelta(days=cfg.purge_calendar_days())
                train = feats.filter(
                    (pl.col("date") <= cut) & pl.col(tgt).is_not_null())
                if cfg.train_window_days is not None:
                    lo = cut - timedelta(days=int(cfg.train_window_days * _CAL_PER_TD))
                    train = train.filter(pl.col("date") >= lo)
                train = train.drop_nulls(subset=feat_cols)
                if train.height >= 5000:
                    model = model_factory()
                    model.fit(train, feat_cols, tgt)
                    last_refit_i = i
                    refits.append({
                        "date": d, "cut": cut, "rows": train.height,
                        "model": getattr(model, "chosen", model.name),
                    })
                    if progress:
                        logger.info(
                            f"{label} refit @{d} cut={cut} rows={train.height:,} "
                            f"→ {getattr(model, 'chosen', model.name)}")

            if model is not None:
                today = by_date.get(d)
                if today is not None and today.height >= cfg.top_k:
                    cand = today["ticker"].to_list()
                    cidx = np.array([tix[t] for t in cand if t in tix])
                    keep = np.array([t in tix for t in cand])
                    if cidx.size >= cfg.top_k:
                        raw = np.asarray(model.predict(today, feat_cols)).ravel()[keep]
                        liq = (panel.adv[i, cidx] >= cfg.min_dollar_volume) & \
                              (panel.price[i, cidx] >= cfg.min_price) & \
                              panel.alive[i, cidx]
                        raw = np.where(liq, raw, np.nan)
                        w_sub = weight_fn(
                            raw,
                            {"dvol": panel.dvol[i, cidx], "adv": panel.adv[i, cidx]},
                            cfg,
                        )
                        w_full = np.zeros(N)
                        w_full[cidx] = w_sub
                        pending = w_full
                        for j in np.nonzero(w_full)[0]:
                            weight_rows.append({
                                "date": d, "ticker": panel.tickers[j],
                                "weight": float(w_full[j])})

        net_ret = equity / equity_prev - 1.0
        daily_rows.append({
            "date": d, "equity": equity, "gross_ret": gross_ret,
            "net_ret": net_ret, "cost": day_cost, "turnover": turnover,
            "n_pos": int((np.abs(pos) > 1e-6).sum()),
        })
        equity_prev = equity

    daily = pl.DataFrame(daily_rows)
    return WalkForwardResult(
        daily=daily,
        trades=pl.DataFrame(trade_rows) if trade_rows else pl.DataFrame(
            schema={"date": pl.Date, "ticker": pl.Utf8, "notional": pl.Float64,
                    "cost": pl.Float64, "shortfall": pl.Float64}),
        weights=pl.DataFrame(weight_rows) if weight_rows else pl.DataFrame(
            schema={"date": pl.Date, "ticker": pl.Utf8, "weight": pl.Float64}),
        refits=refits,
        config=cfg,
        label=label,
    )
