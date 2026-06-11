"""Composite flag logic (§1 of the playbook).

GREEN  = ≥4 flags green, 0 red  → deploy 100% of the cycle's cash
YELLOW = anything else          → deploy in 2 halves ~2 weeks apart
RED    = any flag red           → defensives only, max 25% of tranche
Special: C=RED (or S=RED) → semi freeze — no semi/semicap buys, period.

UNKNOWN readings are treated as YELLOW for the composite but surfaced as
data warnings so a broken feed can never silently turn the board green.
"""
from __future__ import annotations

from .models import FLAG_ORDER, CompositeReading, FlagColor, FlagReading

_DEFAULT_DEPLOYMENT = {"GREEN": 1.0, "YELLOW": 0.5, "RED": 0.25}


def compute_composite(
    readings: dict[str, FlagReading],
    green_min_greens: int = 4,
    deployment: dict[str, float] | None = None,
    semi_freeze_flags: tuple[str, ...] = ("C", "S"),
) -> CompositeReading:
    deployment = deployment or _DEFAULT_DEPLOYMENT
    warnings: list[str] = []

    n_green = n_yellow = n_red = 0
    for f in FLAG_ORDER:
        r = readings.get(f)
        if r is None or r.color == FlagColor.UNKNOWN:
            n_yellow += 1
            why = r.detail if r else "no reading"
            warnings.append(f"{f}: treated as YELLOW ({why})")
            continue
        if r.stale:
            warnings.append(f"{f}: override is stale ({r.as_of}) — re-verify")
        if r.color == FlagColor.GREEN:
            n_green += 1
        elif r.color == FlagColor.RED:
            n_red += 1
        else:
            n_yellow += 1

    if n_red > 0:
        color = FlagColor.RED
        rationale = f"{n_red} flag(s) RED → defensives only (NVO/PGR/SPAXX), max 25% of tranche"
    elif n_green >= green_min_greens:
        color = FlagColor.GREEN
        rationale = f"{n_green} flags GREEN, 0 RED → deploy 100% of the cycle's cash"
    else:
        color = FlagColor.YELLOW
        rationale = f"{n_green}G/{n_yellow}Y/{n_red}R → deploy in 2 halves ~2 weeks apart"

    semi_freeze = any(
        (readings.get(f) is not None and readings[f].color == FlagColor.RED)
        for f in semi_freeze_flags
    )
    if semi_freeze:
        frozen = [f for f in semi_freeze_flags if readings.get(f) and readings[f].color == FlagColor.RED]
        rationale += f" · {'/'.join(frozen)}=RED → SEMI FREEZE (no semi/semicap buys)"

    return CompositeReading(
        color=color,
        n_green=n_green,
        n_yellow=n_yellow,
        n_red=n_red,
        deployment_fraction=float(deployment.get(color.value, 0.5)),
        semi_freeze=semi_freeze,
        defensives_only=(color == FlagColor.RED),
        rationale=rationale,
        data_warnings=warnings,
    )
