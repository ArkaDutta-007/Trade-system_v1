"""Decision ledger — every recommendation becomes a falsifiable, scored prediction.

The system's long-game asset: each ``ts invest`` / ``ts picks`` position (and
anything else that opts in) is appended to an immutable JSONL ledger with its
entry price, calibrated band, horizon and conviction. Once a prediction's
horizon has elapsed, ``resolve_ledger`` scores it against realised prices:

  * **hit** — did price move the predicted direction?
  * **in_band** — did the terminal price land inside the conformal band
    (checks the promised ~90% coverage *on the system's own picks*)?
  * **realized_return** vs the forecast median.

``calibration_report`` aggregates by horizon/source into the numbers that tell
you whether to trust the machine more or less this quarter: hit rate, band
coverage, realised-vs-forecast gap, and the rank-IC of conviction vs outcome.

Files (append-only; a resolution never mutates a prediction):
  data/ledger/predictions.jsonl   one record per recommendation
  data/ledger/resolutions.jsonl   one record per matured prediction
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ..config import Config
from ..utils import get_logger

logger = get_logger(__name__)

__all__ = [
    "record_predictions",
    "resolve_ledger",
    "load_ledger",
    "calibration_report",
]


def _ledger_dir(cfg: Config) -> Path:
    d = cfg.path("data_bronze").parent / "ledger"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("a") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(f"skipping corrupt ledger line in {path.name}")
    return out


def prediction_id(ticker: str, as_of: str, horizon_days: int, source: str) -> str:
    key = f"{ticker.upper()}|{as_of}|{horizon_days}|{source}"
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def record_predictions(cfg: Config, records: list[dict], source: str) -> int:
    """Append prediction records; duplicates (same id) are silently skipped.

    Each record needs: ticker, as_of, horizon_days, entry_price, and ideally
    band_lo/band_median/band_hi (prices), conviction, weight, dollars, model,
    icir, leak_pass, composite.
    """
    d = _ledger_dir(cfg)
    path = d / "predictions.jsonl"
    existing = {r["id"] for r in _read_jsonl(path)}
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    rows = []
    for r in records:
        rid = prediction_id(r["ticker"], str(r["as_of"]), int(r["horizon_days"]), source)
        if rid in existing:
            continue
        rows.append({"id": rid, "created_at": now, "source": source, **r})
        existing.add(rid)
    if rows:
        _append_jsonl(path, rows)
    return len(rows)


def resolve_ledger(cfg: Config, ohlcv: pl.DataFrame | None = None) -> dict[str, int]:
    """Score every matured, still-unresolved prediction against realised prices.

    Maturity is measured in *trading days* on the ticker's own price index —
    a 21d horizon resolves at the 21st bar after ``as_of``, not 21 calendar
    days. Returns counts: {resolved, pending, total}.
    """
    d = _ledger_dir(cfg)
    preds = _read_jsonl(d / "predictions.jsonl")
    if not preds:
        return {"resolved": 0, "pending": 0, "total": 0}
    done = {r["prediction_id"] for r in _read_jsonl(d / "resolutions.jsonl")}
    todo = [p for p in preds if p["id"] not in done]
    if not todo:
        return {"resolved": 0, "pending": 0, "total": len(preds)}

    if ohlcv is None:
        bronze = cfg.path("data_bronze") / "ohlcv_daily.parquet"
        if not bronze.exists():
            logger.warning("no OHLCV parquet — cannot resolve ledger")
            return {"resolved": 0, "pending": len(todo), "total": len(preds)}
        ohlcv = pl.read_parquet(bronze)

    # per-ticker sorted (dates, prices)
    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for tk in {p["ticker"] for p in todo}:
        sub = (
            ohlcv.filter(pl.col("ticker") == tk)
            .sort("date")
            .select(["date", "adj_close"])
            .drop_nulls()
        )
        if sub.height:
            series[tk] = (
                np.array(sub["date"].to_list(), dtype="datetime64[D]"),
                sub["adj_close"].to_numpy().astype(np.float64),
            )

    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    resolutions: list[dict] = []
    for p in todo:
        tk = p["ticker"]
        if tk not in series:
            continue
        dates, px = series[tk]
        as_of = np.datetime64(str(p["as_of"])[:10], "D")
        i0 = int(np.searchsorted(dates, as_of, side="right")) - 1
        if i0 < 0:
            continue
        i1 = i0 + int(p["horizon_days"])
        if i1 >= len(dates):
            continue  # not matured yet
        entry = float(p.get("entry_price") or px[i0])
        terminal = float(px[i1])
        realized = terminal / entry - 1.0
        med = p.get("band_median")
        forecast_ret = (float(med) / entry - 1.0) if med else None
        predicted_up = forecast_ret is None or forecast_ret >= 0
        lo, hi = p.get("band_lo"), p.get("band_hi")
        resolutions.append({
            "prediction_id": p["id"],
            "resolved_at": now,
            "matured_on": str(dates[i1]),
            "terminal_price": round(terminal, 4),
            "realized_return": round(realized, 5),
            "forecast_return": round(forecast_ret, 5) if forecast_ret is not None else None,
            "hit": bool(realized > 0) == predicted_up,
            "in_band": (float(lo) <= terminal <= float(hi))
                       if (lo is not None and hi is not None) else None,
        })
    if resolutions:
        _append_jsonl(d / "resolutions.jsonl", resolutions)
    return {
        "resolved": len(resolutions),
        "pending": len(todo) - len(resolutions),
        "total": len(preds),
    }


def load_ledger(cfg: Config) -> pl.DataFrame:
    """Predictions left-joined with their resolutions (empty frame if none)."""
    d = _ledger_dir(cfg)
    preds = _read_jsonl(d / "predictions.jsonl")
    if not preds:
        return pl.DataFrame()
    pf = pl.DataFrame(preds)
    res = _read_jsonl(d / "resolutions.jsonl")
    if res:
        rf = pl.DataFrame(res).rename({"prediction_id": "id"})
        pf = pf.join(rf, on="id", how="left")
    else:
        pf = pf.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("realized_return"),
            pl.lit(None, dtype=pl.Boolean).alias("hit"),
            pl.lit(None, dtype=pl.Boolean).alias("in_band"),
        )
    return pf


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 5:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    denom = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return round(float((ra * rb).sum() / denom), 4) if denom > 0 else None


def calibration_report(cfg: Config) -> dict[str, Any]:
    """How well has the system's advice actually done, by horizon and source?"""
    df = load_ledger(cfg)
    if df.is_empty():
        return {"n_predictions": 0, "n_resolved": 0, "groups": []}
    resolved = df.filter(pl.col("realized_return").is_not_null())
    groups: list[dict] = []
    if not resolved.is_empty():
        for (source, hz), g in sorted(
            resolved.group_by(["source", "horizon_days"], maintain_order=True),
            key=lambda kv: (kv[0][0], kv[0][1]),
        ):
            rr = g["realized_return"].to_numpy()
            conv = g["conviction"].to_numpy() if "conviction" in g.columns else np.array([])
            fr = g["forecast_return"].drop_nulls().to_numpy() if "forecast_return" in g.columns else np.array([])
            in_band = g["in_band"].drop_nulls()
            groups.append({
                "source": source,
                "horizon_days": int(hz),
                "n": g.height,
                "hit_rate": round(float(g["hit"].mean()), 3),
                "band_coverage": round(float(in_band.mean()), 3) if in_band.len() else None,
                "avg_realized": round(float(rr.mean()), 4),
                "avg_forecast": round(float(fr.mean()), 4) if fr.size else None,
                "conviction_ic": _spearman(conv, rr) if conv.size == rr.size and conv.size else None,
            })
    return {
        "n_predictions": df.height,
        "n_resolved": resolved.height,
        "n_pending": df.height - resolved.height,
        "groups": groups,
    }
