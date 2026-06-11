"""Persistence for flag snapshots.

Two artifacts under data/silver/:
  flags_latest.json     — most recent snapshot (fast reuse across commands)
  flag_history.parquet  — append-only log, one row per flag per snapshot,
                          so regime changes are auditable and plottable.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from ..utils import get_logger
from .models import FlagSnapshot

logger = get_logger(__name__)

LATEST_NAME = "flags_latest.json"
HISTORY_NAME = "flag_history.parquet"


def save_snapshot(snapshot: FlagSnapshot, silver_dir: Path) -> Path:
    silver_dir.mkdir(parents=True, exist_ok=True)
    latest = silver_dir / LATEST_NAME
    latest.write_text(json.dumps(snapshot.to_dict(), indent=2, default=str))

    rows = []
    for f, r in snapshot.readings.items():
        rows.append({
            "snapshot_at": snapshot.as_of,
            "flag": f,
            "color": r.color.value,
            "value": str(r.value) if r.value is not None else None,
            "detail": r.detail,
            "source": r.source,
            "composite": snapshot.composite.color.value,
            "deployment_fraction": snapshot.composite.deployment_fraction,
            "semi_freeze": snapshot.composite.semi_freeze,
        })
    new = pl.DataFrame(rows)
    hist_path = silver_dir / HISTORY_NAME
    if hist_path.exists():
        try:
            old = pl.read_parquet(hist_path)
            new = pl.concat([old, new], how="diagonal").unique(
                subset=["snapshot_at", "flag"], keep="last"
            ).sort(["snapshot_at", "flag"])
        except Exception as e:
            logger.warning(f"flag history merge failed, rewriting: {e}")
    new.write_parquet(hist_path, compression="zstd")
    return latest


def load_latest(silver_dir: Path) -> FlagSnapshot | None:
    p = silver_dir / LATEST_NAME
    if not p.exists():
        return None
    try:
        return FlagSnapshot.from_dict(json.loads(p.read_text()))
    except Exception as e:
        logger.warning(f"could not parse {p}: {e}")
        return None


def load_history(silver_dir: Path) -> pl.DataFrame | None:
    p = silver_dir / HISTORY_NAME
    if not p.exists():
        return None
    return pl.read_parquet(p)
