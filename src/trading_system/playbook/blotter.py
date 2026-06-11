"""Trade blotter — the append-only record of actual fills.

reports/blotter.csv is deliberately CSV: git-diffable, hand-editable, and
importable into a spreadsheet. Every executed trade (done manually in
Fidelity) should be logged with `ts log-trade` so that:
  - realized P&L feeds the NRA tax-shield tracker
  - position drift vs the JSON snapshot is visible
  - every fill carries its compliance verdict for the audit trail
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

BLOTTER_NAME = "blotter.csv"
COLUMNS = [
    "logged_at", "trade_date", "account", "ticker", "side", "qty", "price",
    "dollars", "avg_cost_basis", "realized_pnl", "compliance_verdict", "note",
]


def blotter_path(reports_dir: Path) -> Path:
    return reports_dir / BLOTTER_NAME


def log_trade(
    reports_dir: Path,
    ticker: str,
    side: str,
    qty: float,
    price: float,
    trade_date: str | None = None,
    account: str = "Z32148892",
    avg_cost_basis: float | None = None,
    compliance_verdict: str = "",
    note: str = "",
) -> dict:
    """Append one fill. For SELLs with a known basis, realized P&L is computed."""
    side = side.upper()
    realized = None
    if side == "SELL" and avg_cost_basis is not None:
        realized = round((price - avg_cost_basis) * qty, 2)
    row = {
        "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trade_date": trade_date or datetime.now().date().isoformat(),
        "account": account,
        "ticker": ticker.upper(),
        "side": side,
        "qty": qty,
        "price": price,
        "dollars": round(qty * price, 2),
        "avg_cost_basis": avg_cost_basis if avg_cost_basis is not None else "",
        "realized_pnl": realized if realized is not None else "",
        "compliance_verdict": compliance_verdict,
        "note": note,
    }
    path = blotter_path(reports_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            w.writeheader()
        w.writerow(row)
    return row


def load_blotter(reports_dir: Path) -> pl.DataFrame | None:
    path = blotter_path(reports_dir)
    if not path.exists():
        return None
    return pl.read_csv(path, infer_schema_length=1000)


def blotter_realized(reports_dir: Path, year: int | None = None) -> list[float]:
    """Realized P&L amounts from logged SELLs (for the tax-shield ledger)."""
    df = load_blotter(reports_dir)
    if df is None or df.is_empty() or "realized_pnl" not in df.columns:
        return []
    sub = df.filter(pl.col("side") == "SELL")
    if year is not None:
        sub = sub.filter(pl.col("trade_date").str.starts_with(str(year)))
    vals = []
    for v in sub["realized_pnl"].to_list():
        try:
            if v is not None and str(v) != "":
                vals.append(float(v))
        except (TypeError, ValueError):
            continue
    return vals
