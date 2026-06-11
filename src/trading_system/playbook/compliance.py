"""Pre-trade compliance gate.

Every proposed order is screened against the playbook before it can become a
recommendation, an order, or a logged trade:

  BUY gates
    1. composite RED        → defensives only (max 25% of tranche)
    2. semi freeze (C/S=RED)→ no semi or semicap buys, period
    3. never-buy list (§5)  → blocked outright
    4. re-entry lockouts    → blocked unless StarMine > 6 AND price > 50d MA
    5. position caps (3.5)  → >13% (or per-name cap) bars new cash until <11%
    6. drawdown rule (3.4)  → invested down >15% from Jun 10 → no semi adds
  SELL gates (advisory — selling is the user's call)
    7. hold-winner names    → warn: NRA prime directive, ~36% haircut
    8. tax impact           → shield usage / tax due attached to every sell
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..flags.models import FlagSnapshot
from .loader import Playbook, Portfolio
from .tax import sell_impact, shield_status


@dataclass
class ComplianceResult:
    ticker: str
    side: str                  # BUY | SELL
    allowed: bool
    violations: list[str] = field(default_factory=list)   # hard blocks
    warnings: list[str] = field(default_factory=list)     # advisory
    tax: dict | None = None

    @property
    def verdict(self) -> str:
        return "ALLOWED" if self.allowed else "BLOCKED"

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker, "side": self.side, "verdict": self.verdict,
            "violations": self.violations, "warnings": self.warnings,
            **({"tax": self.tax} if self.tax else {}),
        }


def _hold_winner_tickers(playbook: Playbook) -> set[str]:
    out: set[str] = set()
    for rule in playbook.standing_rules:
        if rule.get("kind") in ("hold_winner", "hold_core"):
            out |= {t.upper() for t in rule.get("tickers", [])}
    return out


def check_trade(
    ticker: str,
    side: str,
    dollars: float,
    playbook: Playbook,
    portfolio: Portfolio,
    snapshot: FlagSnapshot | None = None,
    prices: dict[str, float] | None = None,
    sma50: dict[str, float] | None = None,
    blotter_realized: list[float] | None = None,
) -> ComplianceResult:
    ticker = ticker.upper()
    side = side.upper()
    prices = prices or {}
    sma50 = sma50 or {}
    res = ComplianceResult(ticker=ticker, side=side, allowed=True)

    price = prices.get(ticker)
    if price is None:
        pos = portfolio.position(ticker)
        wrow = portfolio.watchlist.get(ticker)
        price = pos.last_price if pos else (wrow or {}).get("last_price")

    if side == "BUY":
        # 1. composite RED → defensives only
        if snapshot is not None and snapshot.composite.defensives_only:
            if ticker not in playbook.defensives:
                res.allowed = False
                res.violations.append(
                    f"composite RED → defensives only ({'/'.join(sorted(playbook.defensives))}); "
                    f"{ticker} is not a defensive"
                )
            else:
                res.warnings.append("composite RED: cap this tranche at 25% and keep the rest in SPAXX")

        # 2. semi freeze
        if snapshot is not None and snapshot.composite.semi_freeze and playbook.is_semi(ticker):
            res.allowed = False
            res.violations.append(
                f"SEMI FREEZE active ({snapshot.composite.rationale.split('·')[-1].strip()}) — "
                f"{ticker} is classed {sorted(playbook.classes_of(ticker) & {'semi', 'semicap'})}"
            )

        # 3. never-buy list
        nb = playbook.never_buy
        if ticker in nb:
            res.allowed = False
            held_note = " (hold ≠ add)" if portfolio.position(ticker) else ""
            res.violations.append(f"never-buy list (§5): StarMine {nb[ticker]}{held_note}")

        # 4. re-entry lockouts
        if ticker in playbook.lockout_tickers:
            sm = portfolio.starmine(ticker)
            ma = sma50.get(ticker)
            ok_sm = sm is not None and sm > playbook.lockout_min_starmine
            ok_ma = ma is not None and price is not None and price > ma
            if not (ok_sm and ok_ma):
                res.allowed = False
                sm_s = f"StarMine {sm}" if sm is not None else "StarMine unknown"
                ma_s = (
                    f"price {price:.2f} vs 50d MA {ma:.2f}" if (ma is not None and price is not None)
                    else "50d MA unknown"
                )
                res.violations.append(
                    f"re-entry lockout: needs StarMine > {playbook.lockout_min_starmine:.0f} "
                    f"AND price > 50-day MA ({sm_s}; {ma_s})"
                )

        # 5. position caps (rule 3.5 — never sell, bar new cash)
        weight = portfolio.weight_pct(ticker, prices)
        cap = playbook.cap_for(ticker)
        resume = float(playbook.caps.get("resume_pct", 11.0))
        if weight >= cap:
            res.allowed = False
            res.violations.append(
                f"position cap: {ticker} is {weight:.1f}% of portfolio (cap {cap:.0f}%) — "
                f"new cash barred until < {resume:.0f}% (do NOT sell to fix this)"
            )
        elif weight >= cap - 1.5:
            res.warnings.append(f"{ticker} at {weight:.1f}% — approaching its {cap:.0f}% cap")

        # 6. drawdown semi-freeze (rule 3.4)
        if playbook.is_semi(ticker) and playbook.baseline_invested > 0:
            invested = portfolio.invested_value(prices)
            dd = 1.0 - invested / playbook.baseline_invested
            if dd > playbook.drawdown_semi_freeze:
                res.allowed = False
                res.violations.append(
                    f"rule 3.4: invested value down {dd:.1%} from the Jun 10 baseline "
                    f"(>{playbook.drawdown_semi_freeze:.0%}) — ALL semi adds suspended for the quarter"
                )

    elif side == "SELL":
        pos = portfolio.position(ticker)
        if pos is None:
            res.warnings.append(f"no held position in {ticker} per the portfolio snapshot")
        else:
            if ticker in _hold_winner_tickers(playbook):
                res.warnings.append(
                    "NRA prime directive: winners are not sold for rebalancing (~36% haircut). "
                    "Sell only on thesis break."
                )
            shield = shield_status(playbook, portfolio, blotter_realized)
            px = price or pos.last_price
            qty = min(pos.quantity, dollars / px) if dollars > 0 else pos.quantity
            res.tax = sell_impact(qty, px, pos.average_cost, shield)
            res.warnings.append(f"tax: {res.tax['note']}")

    return res
