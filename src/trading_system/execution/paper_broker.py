"""In-memory paper broker. Persists trades and positions to JSON for daily review."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

from ..portfolio.order_policy import Order


@dataclass
class Trade:
    timestamp: str
    ticker: str
    qty: float
    price: float
    side: str
    notional: float
    cost: float


@dataclass
class PaperBroker:
    cash: float = 100_000.0
    holdings: dict[str, float] = field(default_factory=dict)
    trades: list[Trade] = field(default_factory=list)
    cost_bps: float = 4.0
    journal_path: Path | None = None
    _killed: bool = False

    def kill(self) -> None:
        self._killed = True

    @property
    def is_killed(self) -> bool:
        return self._killed

    def equity(self, prices: dict[str, float]) -> float:
        mtm = sum(self.holdings.get(t, 0.0) * prices.get(t, 0.0) for t in self.holdings)
        return self.cash + mtm

    def submit(self, orders: list[Order], prices: dict[str, float]) -> list[Trade]:
        if self._killed:
            return []
        executed: list[Trade] = []
        for o in orders:
            px = prices.get(o.ticker)
            if px is None or px <= 0:
                continue
            cost = o.notional * (self.cost_bps / 10_000.0)
            self.cash -= o.qty * px + cost
            self.holdings[o.ticker] = self.holdings.get(o.ticker, 0.0) + o.qty
            tr = Trade(
                timestamp=datetime.utcnow().isoformat(),
                ticker=o.ticker,
                qty=o.qty,
                price=px,
                side=o.side,
                notional=o.notional,
                cost=cost,
            )
            self.trades.append(tr)
            executed.append(tr)
        self._persist()
        return executed

    def _persist(self) -> None:
        if not self.journal_path:
            return
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cash": self.cash,
            "holdings": self.holdings,
            "trades": [asdict(t) for t in self.trades],
            "killed": self._killed,
        }
        self.journal_path.write_text(json.dumps(payload, indent=2, default=str))

    @classmethod
    def from_journal(cls, path: Path | str, cost_bps: float = 4.0) -> "PaperBroker":
        p = Path(path)
        if not p.exists():
            return cls(journal_path=p, cost_bps=cost_bps)
        data = json.loads(p.read_text())
        broker = cls(
            cash=data.get("cash", 100_000.0),
            holdings=data.get("holdings", {}),
            cost_bps=cost_bps,
            journal_path=p,
        )
        broker._killed = data.get("killed", False)
        broker.trades = [Trade(**t) for t in data.get("trades", [])]
        return broker
