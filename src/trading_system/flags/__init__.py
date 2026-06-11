"""Five-flag (O/F/I/S/C) regime system from the v2 NRA decision tree.

O — Oil/Iran (Brent) · F — Fed (policy+tone) · I — Inflation (core CPI m/m)
S — Semi tape (NDX level) · C — AI capex (hyperscaler guidance)
"""
from .composite import compute_composite
from .lookups import (
    classify_core_cpi,
    classify_fed,
    classify_oil,
    classify_semi_tape,
    lookup_capex,
    lookup_fed,
    lookup_inflation,
    lookup_oil,
    lookup_semi_tape,
)
from .models import FLAG_ORDER, CompositeReading, FlagColor, FlagReading, FlagSnapshot
from .service import build_snapshot, get_flag_snapshot, load_overrides, load_playbook_raw
from .store import load_history, load_latest, save_snapshot

__all__ = [
    "FLAG_ORDER",
    "FlagColor",
    "FlagReading",
    "CompositeReading",
    "FlagSnapshot",
    "classify_oil",
    "classify_fed",
    "classify_core_cpi",
    "classify_semi_tape",
    "lookup_oil",
    "lookup_fed",
    "lookup_inflation",
    "lookup_semi_tape",
    "lookup_capex",
    "compute_composite",
    "build_snapshot",
    "get_flag_snapshot",
    "load_overrides",
    "load_playbook_raw",
    "save_snapshot",
    "load_latest",
    "load_history",
]
