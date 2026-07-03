"""EDGAR structural-freshness discovery — where the next SNDK comes from.

Big asymmetric winners are often *structurally fresh*: spinoffs (SNDK from
WDC), fresh exchange listings, and recent IPOs. All three leave a free,
point-in-time paper trail on EDGAR before the market has fully priced them:

  * **Form 10-12B / 10-12G** — Exchange Act registration of a *spinoff* /
    carve-out (the SNDK signature).
  * **Form 8-A12B / 8-A12A** — registration of a class of securities to list
    on an exchange: "about to trade".
  * **424B4** — final IPO prospectus.

This module queries EDGAR full-text search (``efts.sec.gov``) for recent
filings of those forms, maps filer CIKs to tickers via the official
``company_tickers.json``, and returns candidates with a freshness category.
Responses are disk-cached for a day; the SEC fair-access throttle from
``sec_history`` is reused.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import time
from pathlib import Path

import requests

from ..utils import get_logger
from .sec_history import _UA, _throttle, company_ciks

logger = get_logger(__name__)

_FTS_URL = "https://efts.sec.gov/LATEST/search-index"

# form → freshness category (order = priority when a CIK files several)
DISCOVERY_FORMS: tuple[tuple[str, str], ...] = (
    ("10-12B", "spinoff"),
    ("10-12G", "spinoff"),
    ("8-A12B", "new_listing"),
    ("424B4", "ipo"),
)


def _fts_query(form: str, start: str, end: str, sess: requests.Session,
               max_pages: int = 3) -> list[dict]:
    """Page through EDGAR full-text search for one form type."""
    hits: list[dict] = []
    for page in range(max_pages):
        _throttle()
        params = {
            "q": f'"{form}"',
            "forms": form,
            "dateRange": "custom",
            "startdt": start,
            "enddt": end,
            "from": page * 100,
        }
        try:
            r = sess.get(_FTS_URL, params=params, timeout=30)
            r.raise_for_status()
            batch = (r.json().get("hits") or {}).get("hits") or []
        except Exception as e:
            logger.warning(f"EDGAR FTS {form} page {page} failed: {e}")
            break
        hits.extend(batch)
        if len(batch) < 100:
            break
    return hits


_TICKER_IN_NAME = re.compile(r"\(([A-Z][A-Z0-9.\-]{0,9})\)")


def _parse_hit(h: dict) -> dict | None:
    src = h.get("_source") or {}
    names = src.get("display_names") or []
    if not names:
        return None
    # "Company Name  (TRAX)  (CIK 0001234567)" — ticker parenthetical optional
    disp = names[0]
    cik = None
    if "CIK" in disp:
        digits = "".join(c for c in disp.split("CIK")[-1] if c.isdigit())
        cik = str(int(digits)) if digits else None
    hint = None
    for m in _TICKER_IN_NAME.finditer(disp):
        if m.group(1) != "CIK" and not m.group(1).isdigit():
            hint = m.group(1)
            break
    tickers_field = src.get("tickers") or ""
    if not hint and tickers_field:
        hint = tickers_field.split(",")[0].strip().upper()
    return {
        "company": disp.split("(")[0].strip(),
        "cik": cik,
        "ticker_hint": hint,
        "form": src.get("file_type", ""),
        "filed": src.get("file_date"),
    }


def recent_structural_filings(
    lookback_days: int = 120,
    cache_dir: Path | None = None,
    cache_ttl_h: float = 24.0,
) -> list[dict]:
    """Recent spinoff/new-listing/IPO filers with resolved tickers.

    Returns rows: ``{ticker, company, category, form, filed, cik}`` — only
    filers whose CIK (or FTS ticker hint) resolves to a real ticker survive,
    deduped to each ticker's highest-priority category.
    """
    end = dt.date.today()
    start = end - dt.timedelta(days=lookback_days)
    cache_file = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"edgar_discovery_{lookback_days}d.json"
        if cache_file.exists():
            age_h = (time.time() - cache_file.stat().st_mtime) / 3600
            if age_h < cache_ttl_h:
                try:
                    return json.loads(cache_file.read_text())
                except Exception:
                    pass

    sess = requests.Session()
    sess.headers.update(_UA)
    try:
        # normalize to unpadded CIK strings — _parse_hit strips leading zeros,
        # while company_ciks() returns 10-digit zero-padded keys
        cik_to_ticker = {str(int(cik)): tk for tk, cik in company_ciks().items()}
    except Exception as e:
        logger.warning(f"CIK→ticker map unavailable: {e}")
        cik_to_ticker = {}

    by_ticker: dict[str, dict] = {}
    priority = {cat: i for i, (_, cat) in enumerate(DISCOVERY_FORMS)}
    for form, category in DISCOVERY_FORMS:
        for h in _fts_query(form, start.isoformat(), end.isoformat(), sess):
            row = _parse_hit(h)
            if not row:
                continue
            ticker = row["ticker_hint"] or cik_to_ticker.get(row["cik"] or "")
            if not ticker:
                continue  # unlisted filer — nothing tradeable yet
            rec = {
                "ticker": ticker.upper(),
                "company": row["company"],
                "category": category,
                "form": form,
                "filed": row["filed"],
                "cik": row["cik"],
            }
            prev = by_ticker.get(rec["ticker"])
            if prev is None or priority[category] < priority[prev["category"]]:
                by_ticker[rec["ticker"]] = rec

    out = sorted(by_ticker.values(), key=lambda r: r.get("filed") or "", reverse=True)
    # never cache an empty result — an EDGAR outage would otherwise poison
    # discovery for a full day (same principle as the wiki-ingest fix)
    if cache_file is not None and out:
        cache_file.write_text(json.dumps(out, indent=2))
    logger.info(f"EDGAR discovery: {len(out)} structurally-fresh tickers "
                f"({start} → {end})")
    return out
