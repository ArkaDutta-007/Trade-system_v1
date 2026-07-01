"""Load the playbook config and the live portfolio/watchlist state.

Portfolio truth comes from "portfolio and watchlist.json" (Fidelity snapshot
exported by the user). The blotter (reports/blotter.csv) layers post-snapshot
trades on top, so positions and the realized-G/L ledger stay current without
re-exporting the JSON after every fill.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..config import Config
from ..utils import get_logger

logger = get_logger(__name__)

DEFAULT_PORTFOLIO_FILE = "portfolio and watchlist.json"


@dataclass
class Holding:
    account: str
    symbol: str
    quantity: float
    last_price: float
    current_value: float
    cost_basis_total: float
    average_cost: float
    asset_type: str = "equity"

    def market_value(self, price: float | None = None) -> float:
        px = price if price is not None else self.last_price
        return self.quantity * px


@dataclass
class Portfolio:
    holdings: list[Holding]
    watchlist: dict[str, dict]          # symbol -> {starmine_score, starmine_rating, last_price, ...}
    recently_sold: list[dict]
    cash: float                          # SPAXX + pending across accounts
    as_of: str

    def position(self, symbol: str) -> Holding | None:
        """Aggregate across accounts."""
        rows = [h for h in self.holdings if h.symbol == symbol.upper()]
        if not rows:
            return None
        qty = sum(h.quantity for h in rows)
        val = sum(h.current_value for h in rows)
        basis = sum(h.cost_basis_total for h in rows)
        return Holding(
            account="ALL", symbol=symbol.upper(), quantity=qty,
            last_price=rows[0].last_price, current_value=val,
            cost_basis_total=basis,
            average_cost=basis / qty if qty else 0.0,
            asset_type=rows[0].asset_type,
        )

    @property
    def held_symbols(self) -> list[str]:
        return sorted({h.symbol for h in self.holdings})

    def invested_value(self, prices: dict[str, float] | None = None) -> float:
        prices = prices or {}
        return sum(h.market_value(prices.get(h.symbol)) for h in self.holdings)

    def total_value(self, prices: dict[str, float] | None = None) -> float:
        return self.invested_value(prices) + self.cash

    def weight_pct(self, symbol: str, prices: dict[str, float] | None = None) -> float:
        pos = self.position(symbol)
        if pos is None:
            return 0.0
        total = self.total_value(prices)
        if total <= 0:
            return 0.0
        px = (prices or {}).get(symbol.upper())
        return 100.0 * pos.market_value(px) / total

    def weights(self, prices: dict[str, float] | None = None) -> dict[str, float]:
        return {s: self.weight_pct(s, prices) for s in self.held_symbols}

    def starmine(self, symbol: str) -> float | None:
        row = self.watchlist.get(symbol.upper())
        if row is None:
            return None
        return row.get("starmine_score")


@dataclass
class Playbook:
    raw: dict
    path: Path
    overrides: dict = field(default_factory=dict)

    # ── convenience accessors ────────────────────────────────────────────
    @property
    def never_buy(self) -> dict[str, float]:
        return {k.upper(): v for k, v in (self.raw.get("never_buy") or {}).items()}

    @property
    def lockout_tickers(self) -> set[str]:
        lk = self.raw.get("reentry_lockouts") or {}
        return {t.upper() for t in lk.get("tickers", [])}

    @property
    def lockout_min_starmine(self) -> float:
        return float((self.raw.get("reentry_lockouts") or {}).get("min_starmine", 6.0))

    @property
    def defensives(self) -> set[str]:
        return {t.upper() for t in self.raw.get("defensives", [])}

    @property
    def ticker_classes(self) -> dict[str, set[str]]:
        return {k: {t.upper() for t in v} for k, v in (self.raw.get("ticker_classes") or {}).items()}

    def classes_of(self, symbol: str) -> set[str]:
        symbol = symbol.upper()
        return {cls for cls, members in self.ticker_classes.items() if symbol in members}

    def is_semi(self, symbol: str) -> bool:
        return bool(self.classes_of(symbol) & {"semi", "semicap"})

    @property
    def caps(self) -> dict:
        return self.raw.get("position_caps") or {}

    def cap_for(self, symbol: str) -> float:
        per = {k.upper(): v for k, v in (self.caps.get("per_ticker") or {}).items()}
        return float(per.get(symbol.upper(), self.caps.get("max_pct", 13.0)))

    @property
    def standing_rules(self) -> list[dict]:
        return self.raw.get("standing_rules") or []

    @property
    def cycles(self) -> list[dict]:
        return self.raw.get("cycles") or []

    @property
    def catalysts(self) -> list[dict]:
        return self.raw.get("catalysts") or []

    @property
    def events(self) -> dict:
        return self.overrides.get("events") or {}

    @property
    def baseline_invested(self) -> float:
        return float((self.raw.get("baseline") or {}).get("invested_value", 0.0))

    @property
    def drawdown_semi_freeze(self) -> float:
        return float((self.raw.get("baseline") or {}).get("drawdown_semi_freeze", 0.15))

    @property
    def monthly_contribution(self) -> float:
        return float(self.raw.get("monthly_contribution", 2500.0))


def load_playbook(cfg: Config) -> Playbook:
    pb_rel = cfg.get("playbook", {}).get("file", "configs/playbook_v2.yaml")
    ov_rel = cfg.get("playbook", {}).get("overrides", "configs/flag_overrides.yaml")
    pb_path = cfg.project_root / pb_rel
    with open(pb_path) as f:
        raw = yaml.safe_load(f) or {}
    ov_path = cfg.project_root / ov_rel
    overrides = {}
    if ov_path.exists():
        with open(ov_path) as f:
            overrides = yaml.safe_load(f) or {}
    return Playbook(raw=raw, path=pb_path, overrides=overrides)


def load_portfolio(cfg: Config) -> Portfolio:
    rel = cfg.get("playbook", {}).get("portfolio", DEFAULT_PORTFOLIO_FILE)
    path = cfg.project_root / rel
    if not path.exists():
        raise FileNotFoundError(f"portfolio file not found: {path}")
    data = json.loads(path.read_text())

    holdings = [
        Holding(
            account=h.get("account", "?"),
            symbol=h["symbol"].upper(),
            quantity=float(h.get("quantity", 0.0)),
            last_price=float(h.get("last_price", 0.0)),
            current_value=float(h.get("current_value", 0.0)),
            cost_basis_total=float(h.get("cost_basis_total", 0.0)),
            average_cost=float(h.get("average_cost", 0.0)),
            asset_type=h.get("type", "equity"),
        )
        for h in data.get("holdings", [])
        if h.get("status", "held") == "held"
    ]
    watchlist = {w["symbol"].upper(): w for w in data.get("watchlist", [])}
    cash = sum(float(c.get("value") or 0.0) for c in data.get("cash_and_other", []))
    return Portfolio(
        holdings=holdings,
        watchlist=watchlist,
        recently_sold=data.get("recently_sold", []),
        cash=cash,
        as_of=str(data.get("metadata", {}).get("generated", "?")),
    )
