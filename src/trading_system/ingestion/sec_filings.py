"""SEC EDGAR filings ingestion. Uses the public submissions JSON endpoint."""
from __future__ import annotations

import time
from typing import Iterable

import polars as pl
import requests

from ..utils import get_logger

logger = get_logger(__name__)

SEC_BASE = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"


def _ticker_to_cik(user_agent: str) -> dict[str, int]:
    r = requests.get(TICKER_MAP_URL, headers={"User-Agent": user_agent}, timeout=20)
    r.raise_for_status()
    data = r.json()
    return {row["ticker"].upper(): int(row["cik_str"]) for row in data.values()}


def fetch_recent_filings(
    tickers: Iterable[str],
    user_agent: str,
    forms: Iterable[str] = ("10-K", "10-Q", "8-K"),
    sleep_seconds: float = 0.2,
) -> pl.DataFrame:
    """Fetch recent filings metadata from SEC EDGAR."""
    forms_set = set(forms)
    try:
        cik_map = _ticker_to_cik(user_agent)
    except Exception as e:
        logger.warning(f"Could not load SEC ticker map: {e}")
        return pl.DataFrame()

    from ..utils import track

    rows = []
    for t in track(list(tickers), "SEC filings"):
        cik = cik_map.get(t.upper())
        if cik is None:
            logger.info(f"No CIK for {t}, skipping")
            continue
        url = SEC_BASE.format(cik=cik)
        try:
            r = requests.get(url, headers={"User-Agent": user_agent}, timeout=20)
            r.raise_for_status()
            recent = r.json().get("filings", {}).get("recent", {})
            n = len(recent.get("accessionNumber", []))
            for i in range(n):
                form = recent["form"][i]
                if form not in forms_set:
                    continue
                rows.append(
                    {
                        "ticker": t.upper(),
                        "cik": cik,
                        "accession": recent["accessionNumber"][i],
                        "form": form,
                        "filing_date": recent["filingDate"][i],
                        "report_date": recent.get("reportDate", [""] * n)[i],
                        "primary_document": recent.get("primaryDocument", [""] * n)[i],
                    }
                )
        except Exception as e:
            logger.warning(f"SEC fetch failed for {t}: {e}")
        time.sleep(sleep_seconds)

    if not rows:
        return pl.DataFrame()
    df = pl.DataFrame(rows)
    df = df.with_columns(
        pl.col("filing_date").str.strptime(pl.Date, "%Y-%m-%d", strict=False),
        pl.col("report_date").str.strptime(pl.Date, "%Y-%m-%d", strict=False),
    )
    return df.sort(["ticker", "filing_date"])
