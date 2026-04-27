"""Alert sink. Today: append to a JSONL file under reports/alerts."""
from __future__ import annotations

import json
import time
from pathlib import Path


def emit_alert(level: str, message: str, payload: dict | None = None,
               root: str | Path = "reports/alerts") -> Path:
    p = Path(root)
    p.mkdir(parents=True, exist_ok=True)
    out = p / f"alerts_{time.strftime('%Y%m%d')}.jsonl"
    record = {"ts": time.time(), "level": level, "message": message, "payload": payload or {}}
    with open(out, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
    return out
