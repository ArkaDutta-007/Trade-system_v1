#!/usr/bin/env python3
"""Shared set-up for the November 2026 research pass: the point-in-time panel rebuilt on every cached
whole-market day (the bar table used to roll days off, see MassiveStore.build_raw_bars) through the latest
close, in its own folder so the October results stay reproducible. Re-uses lab_2026_10's helpers."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import lab_2026_10 as LAB  # noqa: E402

OUT = LAB.cfg.path("reports") / "alpha" / "lab_2026_11"
OUT.mkdir(parents=True, exist_ok=True)
LAB.OUT = OUT                                   # pit_panel / pit_protocol now cache here
EXT = LAB.cfg.path("data_bronze").parent / "external"

if __name__ == "__main__":
    pw, allpx, names = LAB.pit_panel()
    print(f"PIT panel {pw.height:,} rows · {pw['ticker'].n_unique():,} names · {pw['date'].min()} → {pw['date'].max()} · {pw['date'].n_unique()} dates")
