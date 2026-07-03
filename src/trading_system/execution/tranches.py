"""Named virtual portfolios ("tabs") — the money-level scoreboard.

``ts invest 2000 --portfolio p1`` books the plan's fills into a named virtual
portfolio; ``ts tab`` marks every portfolio to market and answers the question
the prediction-level ledger can't: *did the decisions actually make money?*

Each portfolio is a JSON book of fills under ``data/portfolios/<name>.json``
(committable, so both machines share the tabs). Alongside every fill the book
records a **SPY counterfactual** — the benchmark shares the same dollars would
have bought at the same close — so the tab always shows alpha vs "you could
have just bought SPY", not raw P&L alone.

This is paper accounting at plan entry prices (the same close the plan was
built on). It measures decision quality, not execution quality.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

import polars as pl

from ..config import Config
from ..utils import get_logger

logger = get_logger(__name__)

_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]{1,32}$")


def _portfolio_dir(cfg: Config) -> Path:
    d = cfg.path("data_bronze").parent / "portfolios"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _book_path(cfg: Config, name: str) -> Path:
    if not _NAME_RE.match(name):
        raise ValueError(
            f"portfolio name {name!r} must be 1-32 chars of letters/digits/_/- "
            "(e.g. p1, moonshot, core_2026)")
    return _portfolio_dir(cfg) / f"{name.lower()}.json"


def load_book(cfg: Config, name: str) -> dict:
    p = _book_path(cfg, name)
    if p.exists():
        return json.loads(p.read_text())
    return {"name": name.lower(), "created_at": None, "fills": []}


def save_book(cfg: Config, book: dict) -> Path:
    p = _book_path(cfg, book["name"])
    p.write_text(json.dumps(book, indent=2, default=str))
    return p


def list_portfolios(cfg: Config) -> list[str]:
    return sorted(p.stem for p in _portfolio_dir(cfg).glob("*.json"))


def book_fills(
    cfg: Config,
    name: str,
    positions: list[dict],
    as_of: str,
    spy_close: float | None,
    source: str = "invest",
    cash_reserve: float = 0.0,
    plan_id: str | None = None,
) -> dict:
    """Append an invest plan's positions to a named book.

    Each fill records ticker/shares/price/dollars, the plan's hold horizon and
    review levels, and the SPY counterfactual shares for the same dollars.

    ``plan_id`` (use the plan's ``generated_at``) makes booking idempotent:
    re-booking the same plan — a Streamlit re-render, a repeated CLI call on a
    cached plan — skips fills already present instead of doubling the book.
    """
    book = load_book(cfg, name)
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    book["created_at"] = book.get("created_at") or now
    already = {(f.get("plan_id"), f["ticker"], f.get("source"))
               for f in book["fills"] if f.get("plan_id")}
    for s in positions:
        dollars = float(s.get("dollars") or 0.0)
        price = float(s.get("entry") or s.get("price") or 0.0)
        if dollars <= 0 or price <= 0:
            continue
        if plan_id and (plan_id, s["ticker"], source) in already:
            continue
        book["fills"].append({
            "plan_id": plan_id,
            "booked_at": now,
            "as_of": str(as_of),
            "source": source,
            "ticker": s["ticker"],
            "shares": round(dollars / price, 6),
            "price": round(price, 4),
            "dollars": round(dollars, 2),
            "hold_days": s.get("hold_days"),
            "stop": s.get("stop"),
            "median_target": s.get("median_target"),
            "spy_close": round(float(spy_close), 4) if spy_close else None,
            "spy_shares": round(dollars / float(spy_close), 6) if spy_close else None,
        })
    if cash_reserve > 0:
        events = book.setdefault("cash_events", [])
        if not (plan_id and any(e.get("plan_id") == plan_id for e in events)):
            events.append({"as_of": str(as_of), "amount": round(cash_reserve, 2),
                           "plan_id": plan_id})
    save_book(cfg, book)
    return book


def _close_series(ohlcv: pl.DataFrame, tickers: set[str]) -> dict[str, tuple[list, list]]:
    out: dict[str, tuple[list, list]] = {}
    sub = (ohlcv.filter(pl.col("ticker").is_in(sorted(tickers)))
           .select(["ticker", "date", "adj_close"]).drop_nulls().sort("date"))
    for (tk,), g in sub.group_by(["ticker"], maintain_order=True):
        out[tk] = (g["date"].to_list(), g["adj_close"].to_list())
    return out


def _price_on_or_before(series: tuple[list, list], day: dt.date) -> float | None:
    dates, px = series
    import bisect
    i = bisect.bisect_right(dates, day) - 1
    return float(px[i]) if i >= 0 else None


def mark_to_market(cfg: Config, name: str, ohlcv: pl.DataFrame | None = None,
                   fetch_missing: bool = False) -> dict[str, Any]:
    """Value a book at the latest close, with the SPY counterfactual and
    per-position P&L. Returns a summary dict (empty book → n_fills=0).

    ``fetch_missing=True`` prices non-universe tickers (moonshot fills) via an
    ad-hoc yfinance fetch — off by default so tests and offline runs never
    touch the network.
    """
    book = load_book(cfg, name)
    fills = book.get("fills", [])
    if not fills:
        return {"name": name, "n_fills": 0}
    if ohlcv is None:
        ohlcv = pl.read_parquet(cfg.path("data_bronze") / "ohlcv_daily.parquet")
    tickers = {f["ticker"] for f in fills} | {"SPY"}
    series = _close_series(ohlcv, tickers)
    if fetch_missing:
        for tk in sorted(tickers - set(series) - {"SPY"}):
            try:
                import yfinance as yf
                h = yf.Ticker(tk).history(period="1y", auto_adjust=True)
                if h is not None and not h.empty:
                    series[tk] = ([d.date() for d in h.index.to_pydatetime()],
                                  [float(x) for x in h["Close"]])
            except Exception as e:
                logger.debug(f"adhoc mark price {tk} failed: {e}")
    last_date: dt.date | None = max(
        (s[0][-1] for s in series.values() if s[0]), default=None)
    if last_date is None:
        return {"name": name, "n_fills": len(fills),
                "error": "no local price history for any booked ticker — run `ts ingest`"}

    # aggregate per ticker (multiple tranches into the same name merge);
    # review levels (stop/target/hold) come from the LATEST fill — a newer
    # tranche's plan supersedes stale levels
    agg: dict[str, dict] = {}
    for f in fills:
        a = agg.setdefault(f["ticker"], {
            "shares": 0.0, "cost": 0.0, "first": f["as_of"],
        })
        a["shares"] += float(f["shares"])
        a["cost"] += float(f["dollars"])
        a["hold_days"] = f.get("hold_days")
        a["stop"] = f.get("stop")
        a["median_target"] = f.get("median_target")

    positions = []
    value = cost = unpriced_cost = 0.0
    for tk, a in sorted(agg.items()):
        s = series.get(tk)
        px = _price_on_or_before(s, last_date) if s else None
        priced = px is not None
        mv = (px or 0.0) * a["shares"]
        entry = a["cost"] / a["shares"] if a["shares"] else 0.0
        positions.append({
            "ticker": tk,
            "shares": round(a["shares"], 4),
            "avg_cost": round(entry, 4),
            "last_price": round(px, 4) if priced else None,
            "cost": round(a["cost"], 2),
            "value": round(mv, 2) if priced else None,
            "pnl": round(mv - a["cost"], 2) if priced else None,
            "pnl_pct": round(mv / a["cost"] - 1, 4) if (priced and a["cost"]) else None,
            "priced": priced,
            "first_fill": a["first"],
            "hold_days": a["hold_days"],
            "stop": a["stop"],
            "median_target": a["median_target"],
            "hit_stop": (priced and a["stop"] is not None
                         and px <= float(a["stop"])),
            "hit_target": (priced and a["median_target"] is not None
                           and px >= float(a["median_target"])),
        })
        # unpriced cost is UNKNOWN, not a loss — hold it out of P&L entirely
        if priced:
            value += mv
            cost += a["cost"]
        else:
            unpriced_cost += a["cost"]

    # SPY counterfactual over the fills whose tickers are actually priced —
    # no scaling/extrapolation onto uncovered dollars (that silently biased
    # alpha); the comparison is return-vs-return on the covered cost.
    priced_tickers = {p["ticker"] for p in positions if p["priced"]}
    spy_shares_total = sum(float(f["spy_shares"]) for f in fills
                           if f.get("spy_shares") and f["ticker"] in priced_tickers)
    spy_cost_covered = sum(float(f["dollars"]) for f in fills
                           if f.get("spy_shares") and f["ticker"] in priced_tickers)
    spy_px = _price_on_or_before(series.get("SPY", ([], [])), last_date)
    spy_value = (spy_px or 0.0) * spy_shares_total
    pnl_pct = (value / cost - 1) if cost else None
    spy_pnl_pct = (spy_value / spy_cost_covered - 1) \
        if (spy_value and spy_cost_covered) else None
    alpha = ((pnl_pct - spy_pnl_pct) * cost
             if (pnl_pct is not None and spy_pnl_pct is not None) else None)
    return {
        "name": name,
        "as_of": str(last_date),
        "n_fills": len(fills),
        "first_fill": min(f["as_of"] for f in fills),
        "cost": round(cost, 2),
        "value": round(value, 2),
        "pnl": round(value - cost, 2),
        "pnl_pct": round(pnl_pct, 4) if pnl_pct is not None else None,
        "spy_value": round(spy_value, 2) if spy_value else None,
        "spy_pnl_pct": round(spy_pnl_pct, 4) if spy_pnl_pct is not None else None,
        "alpha_vs_spy": round(alpha, 2) if alpha is not None else None,
        "unpriced": [p["ticker"] for p in positions if not p["priced"]],
        "unpriced_cost": round(unpriced_cost, 2),
        "positions": positions,
    }


def mark_all(cfg: Config, ohlcv: pl.DataFrame | None = None,
             fetch_missing: bool = False) -> list[dict]:
    names = list_portfolios(cfg)
    if ohlcv is None:
        bronze = cfg.path("data_bronze") / "ohlcv_daily.parquet"
        if not bronze.exists():
            # books exist but nothing to price them with — say so, don't
            # masquerade as "no portfolios yet"
            return [{"name": n, "n_fills": len(load_book(cfg, n).get("fills", [])),
                     "error": "no OHLCV data — run `ts ingest`"} for n in names]
        ohlcv = pl.read_parquet(bronze)
    return [mark_to_market(cfg, n, ohlcv, fetch_missing=fetch_missing)
            for n in names]
