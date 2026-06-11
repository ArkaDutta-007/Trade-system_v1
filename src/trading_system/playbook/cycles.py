"""§4 cycle playbook evaluator.

Each rule's condition block is checked against:
  - today's date (window)
  - the live flag snapshot (flags_any / flags_not / composite)
  - manual event outcomes from configs/flag_overrides.yaml (event)
  - live prices (price_below + per-order price guards)
  - portfolio weights (position_below_pct, portfolio_drawdown_gt)

Statuses:
  FIRES           — all conditions met; orders pass their price guards
  PRICE_GUARD     — conditions met but a price is outside its range
                    ("re-verify price at order time; if it isn't inside the
                     stated range, the rule doesn't fire")
  AWAITING_EVENT  — gated on an event that hasn't resolved yet
  BLOCKED_FLAGS   — flag/composite condition not met
  INACTIVE        — outside the rule's date window
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from ..flags.models import FlagSnapshot
from .compliance import check_trade
from .loader import Playbook, Portfolio


@dataclass
class OrderPlan:
    ticker: str
    dollars: float
    price: float | None
    price_ok: bool
    guard: str               # human description of the price guard
    compliance: dict | None = None
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker, "dollars": self.dollars, "price": self.price,
            "price_ok": self.price_ok, "guard": self.guard, "note": self.note,
            **({"compliance": self.compliance} if self.compliance else {}),
        }


@dataclass
class CycleRuleEval:
    rule_id: str
    cycle: int
    label: str
    status: str
    reasons: list[str] = field(default_factory=list)
    orders: list[OrderPlan] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "rule_id": self.rule_id, "cycle": self.cycle, "label": self.label,
            "status": self.status, "reasons": self.reasons, "note": self.note,
            "orders": [o.to_dict() for o in self.orders],
        }


def _flag_color(snapshot: FlagSnapshot | None, flag: str) -> str | None:
    if snapshot is None:
        return None
    r = snapshot.readings.get(flag)
    return r.color.value if r else None


def _check_condition(
    cond: dict,
    snapshot: FlagSnapshot | None,
    events: dict,
    prices: dict[str, float],
    portfolio: Portfolio | None,
) -> tuple[str, list[str]]:
    """Return (status, reasons). Status in OK | AWAITING_EVENT | BLOCKED_FLAGS."""
    reasons: list[str] = []

    for flag, allowed in (cond.get("flags_any") or {}).items():
        color = _flag_color(snapshot, flag)
        if color is None or color == "UNKNOWN":
            return "AWAITING_EVENT", [f"flag {flag} unknown — refresh flags / set the override"]
        if color not in allowed:
            return "BLOCKED_FLAGS", [f"needs {flag} in {allowed}, is {color}"]
        reasons.append(f"{flag}={color} ✓")

    for flag, banned in (cond.get("flags_not") or {}).items():
        color = _flag_color(snapshot, flag)
        if color is not None and color in banned:
            return "BLOCKED_FLAGS", [f"{flag}={color} is disqualifying ({flag} must not be {banned})"]

    if cond.get("composite"):
        comp = snapshot.composite.color.value if snapshot else None
        if comp not in cond["composite"]:
            return "BLOCKED_FLAGS", [f"needs composite in {cond['composite']}, is {comp}"]
        reasons.append(f"composite={comp} ✓")

    ev = cond.get("event")
    if ev:
        val = events.get(ev["key"])
        if val is None:
            return "AWAITING_EVENT", [f"event '{ev['key']}' unresolved — set it in flag_overrides.yaml"]
        if val != ev.get("equals"):
            return "BLOCKED_FLAGS", [f"event '{ev['key']}' = {val!r}, needs {ev.get('equals')!r}"]
        reasons.append(f"event {ev['key']}={val} ✓")

    for tkr, level in (cond.get("price_below") or {}).items():
        px = prices.get(tkr.upper())
        if px is None:
            return "AWAITING_EVENT", [f"no live price for {tkr}"]
        if px >= float(level):
            return "BLOCKED_FLAGS", [f"{tkr} {px:.2f} not below {level}"]
        reasons.append(f"{tkr} {px:.2f} < {level} ✓")

    for tkr, pct in (cond.get("position_below_pct") or {}).items():
        if portfolio is None:
            continue
        w = portfolio.weight_pct(tkr, prices)
        if w >= float(pct):
            return "BLOCKED_FLAGS", [f"{tkr} weight {w:.1f}% ≥ {pct}% guard"]
        reasons.append(f"{tkr} weight {w:.1f}% < {pct}% ✓")

    dd_gt = cond.get("portfolio_drawdown_gt")
    if dd_gt is not None and portfolio is not None:
        base = portfolio.invested_value()  # snapshot baseline handled by caller's playbook
        # evaluated separately in compliance; here only as informational trigger
        reasons.append(f"(drawdown rule monitored via compliance, threshold {float(dd_gt):.0%})")

    return "OK", reasons


def _order_plan(
    o: dict,
    prices: dict[str, float],
    playbook: Playbook,
    portfolio: Portfolio | None,
    snapshot: FlagSnapshot | None,
) -> OrderPlan:
    tkr = str(o["ticker"]).upper()
    dollars = float(o.get("dollars", 0.0))
    px = prices.get(tkr)

    guard_parts: list[str] = []
    price_ok = True
    if "max_price" in o:
        guard_parts.append(f"≤ {o['max_price']}")
        price_ok = price_ok and (px is not None and px <= float(o["max_price"]))
    if "min_price" in o:
        guard_parts.append(f"≥ {o['min_price']}")
        price_ok = price_ok and (px is not None and px >= float(o["min_price"]))
    if "price_range" in o:
        lo, hi = (float(x) for x in o["price_range"])
        guard_parts.append(f"{lo}–{hi}")
        price_ok = price_ok and (px is not None and lo <= px <= hi)
    if not guard_parts:
        guard_parts.append("market")
    if px is None and ("max_price" in o or "min_price" in o or "price_range" in o):
        price_ok = False

    # placeholder tickers (SEMICAP_LAGGARD etc.) can't be compliance-checked directly
    compliance = None
    if portfolio is not None and tkr.isalpha() and not tkr.startswith(("SEMICAP", "DIVERSIF", "BEST")):
        compliance = check_trade(
            tkr, "BUY", dollars, playbook, portfolio, snapshot=snapshot, prices=prices,
        ).to_dict()

    note_bits = [str(o.get("note", ""))]
    if o.get("requires_flag"):
        note_bits.append(f"requires {o['requires_flag']}")
    if o.get("half"):
        note_bits.append(f"half {o['half']}")
    if o.get("cap_pct"):
        note_bits.append(f"hard cap {o['cap_pct']}% of portfolio")
    if o.get("choose_from"):
        note_bits.append(f"choose from {o['choose_from']} ({o.get('rule', o.get('min_starmine', ''))})")
    if o.get("substitutes_for"):
        note_bits.append(f"substitutes for the {o['substitutes_for']} leg — never both")

    return OrderPlan(
        ticker=tkr, dollars=dollars, price=px, price_ok=price_ok,
        guard=" ".join(guard_parts),
        compliance=compliance,
        note="; ".join(b for b in note_bits if b),
    )


def evaluate_cycles(
    playbook: Playbook,
    portfolio: Portfolio | None,
    snapshot: FlagSnapshot | None,
    prices: dict[str, float] | None = None,
    today: date | None = None,
    include_inactive: bool = False,
) -> list[CycleRuleEval]:
    prices = prices or {}
    today = today or date.today()
    events = playbook.events
    contribution_scale = playbook.monthly_contribution / 2500.0

    evals: list[CycleRuleEval] = []
    for rule in playbook.cycles:
        win = rule.get("window") or []
        start = date.fromisoformat(str(win[0])) if win else date.min
        end = date.fromisoformat(str(win[1])) if len(win) > 1 else date.max
        ev = CycleRuleEval(
            rule_id=str(rule.get("id")), cycle=int(rule.get("cycle", 0)),
            label=rule.get("label", ""), status="INACTIVE", note=rule.get("note", ""),
        )
        if not (start <= today <= end):
            ev.reasons.append(f"window {start}–{end} (today {today})")
            if include_inactive:
                evals.append(ev)
            continue

        status, reasons = _check_condition(rule.get("condition") or {}, snapshot, events, prices, portfolio)
        ev.reasons = reasons
        if status != "OK":
            ev.status = status
            evals.append(ev)
            continue

        orders = [
            _order_plan(
                {**o, "dollars": float(o.get("dollars", 0.0)) * contribution_scale},
                prices, playbook, portfolio, snapshot,
            )
            for o in (rule.get("orders") or [])
        ]
        ev.orders = orders
        if orders and not all(o.price_ok for o in orders):
            ev.status = "PRICE_GUARD"
            bad = [o.ticker for o in orders if not o.price_ok]
            ev.reasons.append(f"price outside guard for {bad} — the rule doesn't fire")
        else:
            ev.status = "FIRES"
        evals.append(ev)

    order = {"FIRES": 0, "PRICE_GUARD": 1, "AWAITING_EVENT": 2, "BLOCKED_FLAGS": 3, "INACTIVE": 4}
    evals.sort(key=lambda e: (order.get(e.status, 9), e.rule_id))
    return evals
