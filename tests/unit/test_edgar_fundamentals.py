"""SEC EDGAR company-facts → quarterly/annual rows: YTD differencing, Q4 derivation, as-first-reported."""
from datetime import date

import polars as pl
import pytest

from trading_system.ingestion.edgar_fundamentals import facts_to_rows


def _e(start, end, val, filed, fp, form="10-Q"):
    return {"start": start, "end": end, "val": val, "filed": filed, "fp": fp, "form": form}


FACTS = {"facts": {"us-gaap": {
    "NetIncomeLoss": {"units": {"USD": [
        _e("2025-01-01", "2025-03-31", 10, "2025-05-01", "Q1"),
        _e("2025-04-01", "2025-06-30", 12, "2025-08-01", "Q2"),
        _e("2025-04-01", "2025-06-30", 99, "2026-08-01", "Q2"),          # later restatement → ignored
        _e("2025-07-01", "2025-09-30", 14, "2025-11-01", "Q3"),
        _e("2025-01-01", "2025-12-31", 50, "2026-02-15", "FY", "10-K"),
    ]}},
    "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [   # YTD only, as in real 10-Qs
        _e("2025-01-01", "2025-03-31", 20, "2025-05-01", "Q1"),
        _e("2025-01-01", "2025-06-30", 45, "2025-08-01", "Q2"),
        _e("2025-01-01", "2025-09-30", 75, "2025-11-01", "Q3"),
        _e("2025-01-01", "2025-12-31", 110, "2026-02-15", "FY", "10-K"),
    ]}},
    "Assets": {"units": {"USD": [{"end": "2025-06-30", "val": 1000, "filed": "2025-08-01", "fp": "Q2", "form": "10-Q"}]}},
    "LiabilitiesAndStockholdersEquity": {"units": {"USD": [{"end": "2025-06-30", "val": 1000, "filed": "2025-08-01", "fp": "Q2", "form": "10-Q"}]}},
    "StockholdersEquity": {"units": {"USD": [{"end": "2025-06-30", "val": 400, "filed": "2025-08-01", "fp": "Q2", "form": "10-Q"}]}},
}}}


def test_quarters_ytd_differencing_q4_and_first_reported():
    df = facts_to_rows(FACTS, "TEST")
    q = df.filter(pl.col("timeframe") == "quarterly").sort("end_date")
    ni = dict(zip(q["end_date"], q["income_statement__net_income_loss"]))
    ocf = dict(zip(q["end_date"], q["cash_flow_statement__net_cash_flow_from_operating_activities"]))
    assert ni[date(2025, 6, 30)] == 12                                  # first reported, not the 99 restatement
    assert ocf[date(2025, 6, 30)] == pytest.approx(25)                  # 45 YTD − 20
    assert ocf[date(2025, 9, 30)] == pytest.approx(30)                  # 75 − 45
    assert ni[date(2025, 12, 31)] == pytest.approx(14)                  # Q4 = 50 FY … derived from the YTD chain
    q4 = q.filter(pl.col("end_date") == date(2025, 12, 31)).row(0, named=True)
    assert q4["filing_date"] == date(2026, 2, 15)                       # usable only from the 10-K
    q2 = q.filter(pl.col("end_date") == date(2025, 6, 30)).row(0, named=True)
    assert q2["filing_date"] == date(2025, 8, 1)
    assert q2["balance_sheet__liabilities"] == 600                      # L&SE − equity when Liabilities is absent
    a = df.filter(pl.col("timeframe") == "annual").row(0, named=True)
    assert a["income_statement__net_income_loss"] == 50 and a["filing_date"] == date(2026, 2, 15)
