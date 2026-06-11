"""NRA tax engine (§8).

Facts for this account holder:
  - Nonresident alien, 183+ days present → flat 30% federal on net US-source
    capital gains (Schedule NEC / 1040-NR). Holding period irrelevant.
  - NY resident → ~6% on top. All-in ≈ 36% on net gains.
  - Losses net against same-year gains only: no carryforward, no $3k offset.
  - THE 2026 SHIELD: net realized losses YTD mean roughly that much in
    additional 2026 gains is federally free — and it vanishes Jan 1.

Realized ledger = the portfolio JSON's `recently_sold` baseline + any SELL
rows logged in the blotter (reports/blotter.csv).
"""
from __future__ import annotations

from dataclasses import dataclass

from .loader import Playbook, Portfolio


@dataclass
class ShieldStatus:
    year: int
    realized_gains: float
    realized_losses: float          # negative number
    net_realized: float
    shield_remaining: float         # gains absorbable at $0 federal
    all_in_rate: float

    def summary(self) -> str:
        return (
            f"{self.year} realized: {self.realized_gains:+,.0f} gains / "
            f"{self.realized_losses:+,.0f} losses → net {self.net_realized:+,.0f}. "
            f"Shield remaining: ${self.shield_remaining:,.0f} of gains at ~$0 tax "
            f"(use it or lose it Dec 31). Beyond that: ~{self.all_in_rate:.0%} haircut."
        )


def all_in_rate(playbook: Playbook) -> float:
    t = playbook.tax
    return float(t.get("federal_rate", 0.30)) + float(t.get("state_rate", 0.06))


def realized_ledger(portfolio: Portfolio, blotter_realized: list[float] | None = None) -> tuple[float, float]:
    """Return (gains, losses) realized this year. Losses are negative."""
    amounts = [
        float(s.get("realized_gain_loss_est") or 0.0)
        for s in portfolio.recently_sold
    ]
    amounts += list(blotter_realized or [])
    gains = sum(a for a in amounts if a > 0)
    losses = sum(a for a in amounts if a < 0)
    return gains, losses


def shield_status(
    playbook: Playbook,
    portfolio: Portfolio,
    blotter_realized: list[float] | None = None,
) -> ShieldStatus:
    gains, losses = realized_ledger(portfolio, blotter_realized)
    net = gains + losses
    return ShieldStatus(
        year=int(playbook.tax.get("shield_year", 2026)),
        realized_gains=gains,
        realized_losses=losses,
        net_realized=net,
        shield_remaining=max(0.0, -net),
        all_in_rate=all_in_rate(playbook),
    )


def sell_impact(
    qty: float,
    price: float,
    average_cost: float,
    shield: ShieldStatus,
) -> dict:
    """Tax consequence of selling qty at price given the current shield."""
    proceeds = qty * price
    basis = qty * average_cost
    gain = proceeds - basis
    if gain <= 0:
        return {
            "proceeds": proceeds,
            "gain": gain,
            "shield_used": 0.0,
            "taxable_gain": 0.0,
            "tax_due": 0.0,
            "note": (
                "realized loss — adds to the shield only if matched by same-year gains; "
                "an unabsorbed NRA loss evaporates Dec 31"
            ),
        }
    shield_used = min(gain, shield.shield_remaining)
    taxable = gain - shield_used
    tax_due = taxable * shield.all_in_rate
    note = (
        f"gain {gain:+,.0f}: ${shield_used:,.0f} absorbed by the {shield.year} shield"
        + (f", ${taxable:,.0f} taxed at ~{shield.all_in_rate:.0%} → ${tax_due:,.0f} due" if taxable > 0 else " → $0 tax")
    )
    return {
        "proceeds": proceeds,
        "gain": gain,
        "shield_used": shield_used,
        "taxable_gain": taxable,
        "tax_due": tax_due,
        "note": note,
    }
