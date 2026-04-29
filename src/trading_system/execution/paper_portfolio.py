"""PaperPortfolio — wraps PaperBroker with daily snapshotting, horizon PnL tracking,
and a backfill replayer that reconstructs a full trade history from OOS predictions.

Key features
------------
- Equal-weight position sizing capped at `max_position_pct` (default 5%)
- Persistent daily equity log at `data/gold/paper_equity_log.parquet`
- Horizon PnL: 1-month, 3-month, 6-month, 1-year from today
- Backfill: replay 2010→today using OOS model predictions + stance thresholds
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from ..execution.paper_broker import PaperBroker
from ..portfolio.order_policy import Order
from ..utils import get_logger

logger = get_logger(__name__)

_DEFAULT_JOURNAL = Path("data/gold/paper_portfolio_journal.json")
_DEFAULT_EQUITY_LOG = Path("data/gold/paper_equity_log.parquet")

# Stance thresholds (mirror decision/analyze.py defaults)
_BUY_THR = 0.005
_SELL_THR = -0.005


class PaperPortfolio:
    """Daily paper trading portfolio with backfill and horizon PnL.

    Parameters
    ----------
    journal_path    : path to JSON state file (positions + trade log)
    equity_log_path : path to parquet daily equity snapshots
    initial_cash    : starting capital (used only when creating fresh)
    max_position_pct: maximum fraction of equity per single position
    cost_bps        : round-trip commission in basis points
    """

    def __init__(
        self,
        journal_path: Path | str = _DEFAULT_JOURNAL,
        equity_log_path: Path | str = _DEFAULT_EQUITY_LOG,
        initial_cash: float = 100_000.0,
        max_position_pct: float = 0.05,
        cost_bps: float = 4.0,
    ):
        self.journal_path = Path(journal_path)
        self.equity_log_path = Path(equity_log_path)
        self.max_position_pct = max_position_pct
        self.broker = PaperBroker.from_journal(self.journal_path, cost_bps=cost_bps)
        if self.broker.cash == 100_000.0 and not self.broker.trades:
            self.broker.cash = initial_cash
        self._initial_cash = initial_cash

    # ── Core decision processing ──────────────────────────────────────────

    def process_decisions(
        self,
        decisions: list[Any],  # list[DecisionResult]
        prices: dict[str, float],
    ) -> list[Order]:
        """Convert BUY/SELL/HOLD decisions into orders and submit to broker."""
        equity = self.broker.equity(prices)
        orders: list[Order] = []

        for d in decisions:
            ticker = d.ticker
            stance = d.stance
            px = prices.get(ticker)
            if px is None or px <= 0:
                continue

            if stance == "BUY":
                # Skip if already fully sized
                cur_qty = self.broker.holdings.get(ticker, 0.0)
                cur_notional = cur_qty * px
                target_notional = min(equity * self.max_position_pct, equity * self.max_position_pct)
                if cur_notional >= target_notional * 0.95:
                    continue  # already at/near target
                delta_notional = target_notional - cur_notional
                if delta_notional < 10.0:
                    continue
                qty = delta_notional / px
                orders.append(Order(ticker=ticker, qty=qty, side="buy", notional=delta_notional))

            elif stance == "SELL":
                qty = self.broker.holdings.get(ticker, 0.0)
                if qty > 0:
                    orders.append(Order(ticker=ticker, qty=-qty, side="sell", notional=qty * px))

        self.broker.submit(orders, prices)
        return orders

    # ── Daily snapshot ────────────────────────────────────────────────────

    def snapshot(self, as_of: date | str, prices: dict[str, float]) -> dict:
        """Append a daily equity row to the parquet log."""
        as_of_str = str(as_of)
        equity = self.broker.equity(prices)
        n_pos = sum(1 for q in self.broker.holdings.values() if q > 0.001)

        # Load existing log or create
        log_path = self.equity_log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if log_path.exists():
            existing = pl.read_parquet(log_path)
        else:
            existing = pl.DataFrame(schema={
                "date": pl.Utf8, "equity": pl.Float64,
                "cash": pl.Float64, "n_positions": pl.Int32,
                "drawdown": pl.Float64,
            })

        # Drawdown from peak
        peak = float(existing["equity"].max()) if len(existing) > 0 else equity
        peak = max(peak, equity)
        drawdown = (equity - peak) / peak if peak > 0 else 0.0

        new_row = pl.DataFrame({
            "date": [as_of_str],
            "equity": [equity],
            "cash": [self.broker.cash],
            "n_positions": [n_pos],
            "drawdown": [drawdown],
        })
        pl.concat([existing, new_row]).write_parquet(log_path, compression="zstd")
        return {"date": as_of_str, "equity": equity, "cash": self.broker.cash,
                "n_positions": n_pos, "drawdown": drawdown}

    # ── Horizon PnL ───────────────────────────────────────────────────────

    def compute_pnl_horizons(self, as_of: date | None = None) -> dict[str, float | None]:
        """Return portfolio return over 1m / 3m / 6m / 1y horizons ending today."""
        if not self.equity_log_path.exists():
            return {"1m": None, "3m": None, "6m": None, "1y": None}
        log = pl.read_parquet(self.equity_log_path).sort("date")
        if log.is_empty():
            return {"1m": None, "3m": None, "6m": None, "1y": None}

        today_eq_rows = log.tail(1)["equity"]
        today_eq = float(today_eq_rows[0]) if len(today_eq_rows) else None
        if today_eq is None:
            return {"1m": None, "3m": None, "6m": None, "1y": None}

        ref_date = as_of or date.today()
        result = {}
        for label, days in [("1m", 30), ("3m", 91), ("6m", 182), ("1y", 365)]:
            cutoff = str(ref_date - timedelta(days=days))
            past = log.filter(pl.col("date") <= cutoff)
            if past.is_empty():
                result[label] = None
            else:
                past_eq = float(past.tail(1)["equity"][0])
                result[label] = (today_eq - past_eq) / past_eq if past_eq > 0 else None
        return result

    # ── Summary ───────────────────────────────────────────────────────────

    def summary(self, prices: dict[str, float] | None = None) -> dict:
        """Return a dict summary for display."""
        prices = prices or {}
        equity = self.broker.equity(prices)
        horizons = self.compute_pnl_horizons()

        # Win rate from trades
        gains = [t.qty * prices.get(t.ticker, t.price) - abs(t.notional)
                 for t in self.broker.trades if t.side == "sell" and t.ticker in prices]
        win_rate = (sum(1 for g in gains if g > 0) / len(gains)) if gains else None

        return {
            "equity": equity,
            "cash": self.broker.cash,
            "n_positions": sum(1 for q in self.broker.holdings.values() if q > 0.001),
            "n_trades": len(self.broker.trades),
            "win_rate": win_rate,
            "pnl_total": (equity - self._initial_cash) / self._initial_cash,
            "pnl_1m": horizons.get("1m"),
            "pnl_3m": horizons.get("3m"),
            "pnl_6m": horizons.get("6m"),
            "pnl_1y": horizons.get("1y"),
            "holdings": dict(self.broker.holdings),
        }

    # ── Backfill replayer ─────────────────────────────────────────────────

    def backfill_from_predictions(
        self,
        features_path: Path | str,
        predictions_path: Path | str,
        buy_threshold: float = _BUY_THR,
        sell_threshold: float = _SELL_THR,
        start_date: str = "2015-01-01",
    ) -> int:
        """Replay all trading days using OOS model predictions.

        For each date (ascending):
          1. Look up model score for each ticker
          2. Apply buy/sell/hold thresholds
          3. Use adj_close as execution price
          4. Submit orders to broker + snapshot

        Returns number of days replayed.
        """
        features = pl.read_parquet(Path(features_path))
        predictions = pl.read_parquet(Path(predictions_path))

        # Align on date + ticker
        data = (
            features
            .select(["date", "ticker", "adj_close"])
            .join(predictions.select(["date", "ticker", "score"]), on=["date", "ticker"], how="inner")
            .filter(pl.col("date") >= pl.lit(start_date).cast(pl.Date))
            .sort("date")
        )

        if data.is_empty():
            logger.warning("backfill_from_predictions: no data to replay")
            return 0

        dates = sorted(data["date"].unique().to_list())
        logger.info(f"Backfill: replaying {len(dates)} trading days from {dates[0]} to {dates[-1]}")

        replayed = 0
        for d in dates:
            day_data = data.filter(pl.col("date") == d).to_dicts()
            prices = {r["ticker"]: float(r["adj_close"]) for r in day_data if r["adj_close"]}

            # Build synthetic decision objects
            decisions = []
            for r in day_data:
                score = float(r["score"]) if r["score"] is not None else 0.0
                if score >= buy_threshold:
                    stance = "BUY"
                elif score <= sell_threshold:
                    stance = "SELL"
                else:
                    stance = "HOLD"
                decisions.append(_SyntheticDecision(ticker=r["ticker"], stance=stance))

            self.process_decisions(decisions, prices)
            self.snapshot(d, prices)
            replayed += 1

        logger.info(f"Backfill complete: {replayed} days replayed")
        return replayed


class _SyntheticDecision:
    """Minimal stand-in for DecisionResult used during backfill replay."""
    __slots__ = ("ticker", "stance")

    def __init__(self, ticker: str, stance: str):
        self.ticker = ticker
        self.stance = stance
