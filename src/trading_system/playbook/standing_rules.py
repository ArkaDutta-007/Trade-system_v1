"""§3 standing sell/hold rules — evaluated against live prices every run.

Every rule in the playbook is definitive: a SELL fires only on its trigger.
The evaluator reports one of:
  TRIGGERED — the definitive condition is met → act
  NEAR      — within `near_pct` of a trigger → watch closely
  AWAITING  — event-conditioned rule whose event is unresolved
  OK        — nothing to do
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .loader import Playbook, Portfolio

NEAR_PCT = 0.03  # within 3% of a trigger level counts as NEAR


@dataclass
class RuleCheck:
    ticker: str
    kind: str
    status: str            # TRIGGERED | NEAR | AWAITING | OK | NO_DATA
    action: str            # what to do if/when it fires
    detail: str
    price: float | None = None
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker, "kind": self.kind, "status": self.status,
            "action": self.action, "detail": self.detail, "price": self.price,
            **({"extras": self.extras} if self.extras else {}),
        }


def _near(price: float, level: float) -> bool:
    return abs(price - level) / level <= NEAR_PCT


def _check_one(rule: dict, ticker: str, price: float | None,
               starmine: float | None, events: dict) -> RuleCheck:
    kind = rule.get("kind", "hold")
    note = rule.get("note", "")

    if kind in ("hold_winner", "hold_core", "hold_accumulate", "hold"):
        return RuleCheck(ticker, kind, "OK", "HOLD", note, price)

    if price is None and kind != "event_kill_switch":
        return RuleCheck(ticker, kind, "NO_DATA", "—", f"no live price for {ticker}", None)

    if kind == "stop_and_trim":
        stop, trim = float(rule["stop_below"]), float(rule["trim_above"])
        if price < stop:
            return RuleCheck(ticker, kind, "TRIGGERED", "SELL ALL",
                             f"close {price:.2f} < stop {stop:.2f}. {note}", price)
        if price > trim:
            return RuleCheck(ticker, kind, "TRIGGERED", "TRIM 1/2",
                             f"price {price:.2f} > blow-off {trim:.2f}. {note}", price)
        if _near(price, stop):
            return RuleCheck(ticker, kind, "NEAR", f"SELL ALL if < {stop:.0f}",
                             f"price {price:.2f} within 3% of stop {stop:.2f}", price)
        return RuleCheck(ticker, kind, "OK", f"stop {stop:.0f} / trim {trim:.0f}",
                         f"price {price:.2f} inside band", price)

    if kind in ("probation_floor", "floor"):
        floor = float(rule.get("hard_floor", rule.get("sell_below")))
        ev_key = rule.get("event_key")
        ev_val = events.get(ev_key) if ev_key else None
        if price < floor:
            return RuleCheck(ticker, kind, "TRIGGERED", "SELL (hard floor)",
                             f"close {price:.2f} < floor {floor:.2f} — sell without waiting. {note}", price)
        if ev_key and ev_val == rule.get("event_sell_value"):
            return RuleCheck(ticker, kind, "TRIGGERED", "SELL (event)",
                             f"{ev_key} = {ev_val}. {note}", price)
        if _near(price, floor):
            return RuleCheck(ticker, kind, "NEAR", f"SELL if < {floor:.0f}",
                             f"price {price:.2f} within 3% of floor {floor:.2f}", price)
        status = "AWAITING" if (ev_key and ev_val is None) else "OK"
        return RuleCheck(ticker, kind, status, f"floor {floor:.0f}",
                         f"price {price:.2f} above floor. {note}", price)

    if kind == "probation_band":
        lo = float(rule["sell_below"])
        s_lo, s_hi = (float(x) for x in rule.get("strength_exit", [None, None]) or (None, None))
        if price < lo:
            return RuleCheck(ticker, kind, "TRIGGERED", "SELL",
                             f"close {price:.2f} < {lo:.2f}. {note}", price)
        if s_lo is not None and s_lo <= price <= s_hi:
            return RuleCheck(ticker, kind, "TRIGGERED", "SELL into strength",
                             f"price {price:.2f} inside strength-exit {s_lo:.0f}–{s_hi:.0f}", price)
        ev_key = rule.get("event_key")
        if events.get(ev_key) == rule.get("event_sell_value"):
            return RuleCheck(ticker, kind, "TRIGGERED", "SELL (event)",
                             f"{ev_key} = {events.get(ev_key)}. {note}", price)
        if _near(price, lo):
            return RuleCheck(ticker, kind, "NEAR", f"SELL if < {lo:.0f}",
                             f"price {price:.2f} within 3% of {lo:.2f}", price)
        return RuleCheck(ticker, kind, "OK", f"sell <{lo:.0f} or {s_lo:.0f}–{s_hi:.0f} bounce",
                         f"price {price:.2f}. {note}", price)

    if kind == "exit_band":
        r_lo, r_hi = (float(x) for x in rule["recovery_exit"])
        cap = float(rule["capitulation_below"])
        if r_lo <= price <= r_hi or price > r_hi:
            return RuleCheck(ticker, kind, "TRIGGERED", "SELL (recovery exit)",
                             f"price {price:.2f} reached {r_lo:.0f}–{r_hi:.0f}", price)
        if price < cap:
            return RuleCheck(ticker, kind, "TRIGGERED", "SELL (capitulation exit)",
                             f"close {price:.2f} < {cap:.2f}", price)
        if _near(price, cap) or _near(price, r_lo):
            return RuleCheck(ticker, kind, "NEAR", f"exits at <{cap:.0f} or {r_lo:.0f}–{r_hi:.0f}",
                             f"price {price:.2f} approaching an exit", price)
        return RuleCheck(ticker, kind, "OK", f"exit <{cap:.0f} or {r_lo:.0f}–{r_hi:.0f}",
                         f"price {price:.2f} between exits. {note}", price)

    if kind == "event_kill_switch":
        ev_key = rule.get("event_key")
        if events.get(ev_key) is True:
            return RuleCheck(ticker, kind, "TRIGGERED", "SELL SAME DAY",
                             f"{ev_key} = true. {note}", price)
        return RuleCheck(ticker, kind, "OK", "SELL same day on equity raise",
                         note, price)

    if kind == "starmine_review":
        thr = float(rule.get("starmine_below", 2.0))
        if starmine is None:
            return RuleCheck(ticker, kind, "NO_DATA", f"review if StarMine < {thr}",
                             "no StarMine score available", price)
        if starmine < thr:
            return RuleCheck(ticker, kind, "TRIGGERED", "REVIEW position",
                             f"StarMine {starmine:.1f} < {thr:.1f}. {note}", price)
        return RuleCheck(ticker, kind, "OK", f"review if StarMine < {thr}",
                         f"StarMine {starmine:.1f}", price)

    return RuleCheck(ticker, kind, "OK", "—", note, price)


def evaluate_standing_rules(
    playbook: Playbook,
    portfolio: Portfolio,
    prices: dict[str, float],
) -> list[RuleCheck]:
    """Evaluate every §3 rule for the tickers it covers."""
    events = playbook.events
    checks: list[RuleCheck] = []
    for rule in playbook.standing_rules:
        tickers = [t.upper() for t in rule.get("tickers", [])]
        for t in tickers:
            price = prices.get(t)
            if price is None:
                pos = portfolio.position(t)
                price = pos.last_price if pos else None
            checks.append(_check_one(rule, t, price, portfolio.starmine(t), events))
    order = {"TRIGGERED": 0, "NEAR": 1, "AWAITING": 2, "NO_DATA": 3, "OK": 4}
    checks.sort(key=lambda c: (order.get(c.status, 9), c.ticker))
    return checks
