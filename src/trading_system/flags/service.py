"""Flag snapshot orchestration: live lookups + manual overrides + persistence.

`get_flag_snapshot(cfg)` is the single entry point the rest of the system
uses. It caches snapshots in data/silver/flags_latest.json so a batch run
(e.g. `ts analyze-all` over 118 tickers) hits the network once, not 118 times.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import yaml

from ..config import Config
from ..utils import get_logger
from .composite import compute_composite
from .lookups import (
    lookup_capex,
    lookup_fed,
    lookup_inflation,
    lookup_oil,
    lookup_semi_tape,
)
from .models import FlagColor, FlagReading, FlagSnapshot
from .store import load_latest, save_snapshot

logger = get_logger(__name__)

DEFAULT_MAX_AGE_MINUTES = 60.0


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_playbook_raw(cfg: Config) -> dict:
    pb_path = cfg.project_root / cfg.get("playbook", {}).get("file", "configs/playbook_v2.yaml")
    return _load_yaml(pb_path)


def load_overrides(cfg: Config) -> dict:
    ov_path = cfg.project_root / cfg.get("playbook", {}).get("overrides", "configs/flag_overrides.yaml")
    return _load_yaml(ov_path)


def _is_stale(as_of, max_age_days: int) -> bool:
    if as_of is None:
        return False
    if isinstance(as_of, str):
        as_of = date.fromisoformat(as_of[:10])
    if isinstance(as_of, datetime):
        as_of = as_of.date()
    return (date.today() - as_of).days > max_age_days


def _apply_override(reading: FlagReading, override: dict | None, max_age_days: int) -> FlagReading:
    """Manual color wins over the automatic reading; staleness is tracked."""
    if not override:
        return reading
    color = override.get("color")
    if color is None:
        return reading
    try:
        forced = FlagColor(str(color).upper())
    except ValueError:
        logger.warning(f"{reading.flag}: bad override color {color!r} — ignoring")
        return reading
    note = override.get("note") or ""
    as_of = override.get("as_of")
    stale = _is_stale(as_of, max_age_days)
    auto = f" [auto would be {reading.color.value}: {reading.detail}]" if reading.source == "live" else ""
    reading.color = forced
    reading.detail = f"override ({as_of}): {note}{auto}"
    reading.source = "override" if reading.source != "live" else "auto+override"
    reading.stale = stale
    return reading


def build_snapshot(cfg: Config) -> FlagSnapshot:
    """Run all live lookups, apply overrides, compute the composite."""
    pb = load_playbook_raw(cfg)
    ov = load_overrides(cfg)
    flag_cfg = pb.get("flags", {})
    ov_flags = ov.get("flags", {}) or {}
    events = ov.get("events", {}) or {}
    max_age_days = int(ov.get("max_age_days", 45))

    readings = {
        "O": lookup_oil(
            thresholds=flag_cfg.get("O", {}).get("thresholds"),
            hormuz_closed=bool(events.get("hormuz_closed")),
        ),
        "F": lookup_fed(lookback_days=int(flag_cfg.get("F", {}).get("lookback_days", 75))),
        "I": lookup_inflation(thresholds=flag_cfg.get("I", {}).get("thresholds")),
        "S": lookup_semi_tape(thresholds=flag_cfg.get("S", {}).get("thresholds")),
        "C": lookup_capex(),
    }
    for f, r in readings.items():
        readings[f] = _apply_override(r, ov_flags.get(f), max_age_days)

    comp_cfg = pb.get("composite", {})
    composite = compute_composite(
        readings,
        green_min_greens=int(comp_cfg.get("green_min_greens", 4)),
        deployment=comp_cfg.get("deployment"),
        semi_freeze_flags=tuple(comp_cfg.get("semi_freeze_flags", ["C", "S"])),
    )
    return FlagSnapshot(
        as_of=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        readings=readings,
        composite=composite,
    )


def get_flag_snapshot(
    cfg: Config,
    refresh: bool = False,
    max_age_minutes: float = DEFAULT_MAX_AGE_MINUTES,
) -> FlagSnapshot:
    """Return a flag snapshot, reusing the cached one when fresh.

    refresh=True forces live lookups. On lookup failure with a cache present,
    the stale cache is returned (annotated) rather than raising.
    """
    silver = cfg.path("data_silver")
    cached = load_latest(silver)
    if not refresh and cached is not None and cached.age_minutes() <= max_age_minutes:
        return cached

    try:
        snap = build_snapshot(cfg)
        save_snapshot(snap, silver)
        return snap
    except Exception as e:
        logger.warning(f"flag snapshot build failed: {e}")
        if cached is not None:
            cached.composite.data_warnings.append(
                f"refresh failed ({e}); using cached snapshot from {cached.as_of}"
            )
            return cached
        raise
