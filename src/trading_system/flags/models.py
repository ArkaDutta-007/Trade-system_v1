"""Data models for the five-flag (O/F/I/S/C) regime system."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any


class FlagColor(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"
    UNKNOWN = "UNKNOWN"  # lookup failed — treated as YELLOW in the composite

    @property
    def rank(self) -> int:
        return {"GREEN": 0, "YELLOW": 1, "UNKNOWN": 1, "RED": 2}[self.value]


FLAG_ORDER = ["O", "F", "I", "S", "C"]


@dataclass
class FlagReading:
    flag: str                 # O | F | I | S | C
    name: str                 # "Oil / Iran", ...
    color: FlagColor
    value: float | str | None  # the observed metric (Brent price, NDX level, core CPI m/m %)
    detail: str               # human-readable explanation of the reading
    source: str               # "live" | "override" | "auto+override" | "error"
    as_of: str                # ISO timestamp of the observation
    stale: bool = False       # override older than max_age_days
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["color"] = self.color.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FlagReading":
        d = dict(d)
        d["color"] = FlagColor(d["color"])
        return cls(**d)


@dataclass
class CompositeReading:
    color: FlagColor              # composite GREEN / YELLOW / RED
    n_green: int
    n_yellow: int
    n_red: int
    deployment_fraction: float    # 1.0 / 0.5 / 0.25
    semi_freeze: bool             # C=RED or S=RED → zero semi/semicap buys
    defensives_only: bool         # composite RED → defensives only
    rationale: str
    data_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["color"] = self.color.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "CompositeReading":
        d = dict(d)
        d["color"] = FlagColor(d["color"])
        return cls(**d)


@dataclass
class FlagSnapshot:
    as_of: str                              # ISO timestamp of the snapshot
    readings: dict[str, FlagReading]        # keyed by flag letter
    composite: CompositeReading

    def to_dict(self) -> dict:
        return {
            "as_of": self.as_of,
            "readings": {k: v.to_dict() for k, v in self.readings.items()},
            "composite": self.composite.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FlagSnapshot":
        return cls(
            as_of=d["as_of"],
            readings={k: FlagReading.from_dict(v) for k, v in d["readings"].items()},
            composite=CompositeReading.from_dict(d["composite"]),
        )

    def age_minutes(self) -> float:
        then = datetime.fromisoformat(self.as_of)
        return (datetime.now(tz=then.tzinfo) - then).total_seconds() / 60.0

    def summary_line(self) -> str:
        seq = " ".join(f"{f}={self.readings[f].color.value[0]}" for f in FLAG_ORDER if f in self.readings)
        return (
            f"{seq} → composite {self.composite.color.value} "
            f"({self.composite.n_green}G/{self.composite.n_yellow}Y/{self.composite.n_red}R), "
            f"deploy {self.composite.deployment_fraction:.0%}"
            + (" · SEMI FREEZE" if self.composite.semi_freeze else "")
        )
