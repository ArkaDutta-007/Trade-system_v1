"""Fundamentals straight from SEC EDGAR XBRL (company facts) — the free, primary source.

Why: Massive retired ``/vX/reference/financials`` (sunset 2026-06-22, 410 brownout from
2026-09-28) and its replacement needs a paid plan. EDGAR is where that data came from anyway.

What it produces: ``data/bronze/edgar/financials.parquet`` in the same wide schema the alpha panel
already consumes (``income_statement__revenues``, ``balance_sheet__assets`` …, ``timeframe`` in
quarterly/annual, ``end_date``, ``filing_date``), merged with the older Massive rows for any
(ticker, period) EDGAR does not cover (mostly foreign IFRS filers).

Point-in-time rules
  * Every value is taken **as first reported**: for each (concept, period) the record with the
    earliest ``filed`` date wins, so later restatements never leak backwards.
  * 10-Q cash-flow lines are year-to-date; quarterly values are differenced within a fiscal year
    (``start`` date) chain. Q4 flows = annual − 9-month YTD, available from the 10-K's filing date.
  * Balance-sheet lines are instants at the period end.

Freshness: the first run fetches company facts for every requested ticker (~4 MB each, ≤ 8 req/s
per SEC fair-access policy). Later runs read EDGAR's daily form index and refetch only companies
that filed a 10-Q/10-K since the last update.
"""
from __future__ import annotations

import gzip
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import polars as pl
import requests

from ..utils import get_logger

logger = get_logger(__name__)

FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
DAILY_INDEX_URL = "https://www.sec.gov/Archives/edgar/daily-index/{y}/QTR{q}/form.{ymd}.idx"
FORMS = {"10-Q", "10-K", "10-Q/A", "10-K/A", "10-KT"}

I, B, C = "income_statement__", "balance_sheet__", "cash_flow_statement__"
# output column → candidate us-gaap concepts (first that has data for a period wins)
FLOW = {
    I + "revenues": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet",
                     "RevenueFromContractWithCustomerIncludingAssessedTax", "RevenuesNetOfInterestExpense",
                     "SalesRevenueGoodsNet", "SalesRevenueServicesNet"],
    I + "net_income_loss": ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    I + "gross_profit": ["GrossProfit"],
    I + "operating_income_loss": ["OperatingIncomeLoss"],
    I + "diluted_earnings_per_share": ["EarningsPerShareDiluted"],
    I + "basic_earnings_per_share": ["EarningsPerShareBasic"],
    C + "net_cash_flow_from_operating_activities": ["NetCashProvidedByUsedInOperatingActivities",
                                                    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
}
SHARES = {  # averages over the period: not additive, never differenced
    I + "diluted_average_shares": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
    I + "basic_average_shares": ["WeightedAverageNumberOfSharesOutstandingBasic"],
}
INSTANT = {
    B + "assets": ["Assets"],
    B + "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    B + "liabilities": ["Liabilities"],
    B + "current_assets": ["AssetsCurrent"],
    B + "current_liabilities": ["LiabilitiesCurrent"],
    B + "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "_lse": ["LiabilitiesAndStockholdersEquity"],
}


class _RateLimit:
    def __init__(self, per_s: float):
        self.dt, self.lock, self.next = 1.0 / per_s, threading.Lock(), 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            t = max(now, self.next)
            self.next = t + self.dt
        time.sleep(max(0.0, t - now))


def _d(s: str | None) -> date | None:
    return date.fromisoformat(s) if s else None


def _first_reported(entries: list[dict]) -> dict[tuple, dict]:
    """(start, end) → the earliest-filed 10-Q/10-K record for that period."""
    out: dict[tuple, dict] = {}
    for e in entries:
        if e.get("form") not in FORMS or e.get("val") is None or not e.get("end") or not e.get("filed"):
            continue
        k = (e.get("start"), e["end"])
        if k not in out or e["filed"] < out[k]["filed"]:
            out[k] = e
    return out


def _concept_entries(facts: dict, concept: str) -> list[dict]:
    units = (facts.get("us-gaap", {}).get(concept) or {}).get("units") or {}
    for u in ("USD", "USD/shares", "shares"):
        if u in units:
            return units[u]
    return next(iter(units.values()), [])


def _flow_series(entries: list[dict], additive: bool = True) -> tuple[dict, dict]:
    """→ (quarterly {end: (val, filed, fp)}, annual {end: (val, filed, start)}) for one concept."""
    recs = _first_reported(entries)
    q, a, chains = {}, {}, {}
    for (s, e), r in recs.items():
        if not s:
            continue
        dur = (_d(e) - _d(s)).days
        if 80 <= dur <= 100:
            q[e] = (float(r["val"]), r["filed"], r.get("fp"))
        if 350 <= dur <= 380:
            a[e] = (float(r["val"]), r["filed"], s)
        chains.setdefault(s, []).append((e, dur, r))
    if additive:
        for s, ch in chains.items():          # YTD chains: difference consecutive cumulative values
            ch.sort()
            for (e0, d0, r0), (e1, d1, r1) in zip(ch, ch[1:]):
                if e1 in q or d1 <= 100:
                    continue
                if 80 <= (_d(e1) - _d(e0)).days <= 100:
                    q[e1] = (float(r1["val"]) - float(r0["val"]), r1["filed"], r1.get("fp") if d1 < 350 else "Q4")
        for e, (v, fl, s) in a.items():       # Q4 = FY − (Q1+Q2+Q3) when only 3-month figures were filed
            if e in q:
                continue
            inside = sorted(k for k in q if _d(s) < _d(k) < _d(e))
            if len(inside) == 3 and 80 <= (_d(e) - _d(inside[-1])).days <= 100:
                q[e] = (v - sum(q[k][0] for k in inside), fl, "Q4")
    else:
        for e, (v, f, s) in a.items():       # Q4 average shares ≈ the annual average
            q.setdefault(e, (v, f, "Q4"))
    return q, a


def facts_to_rows(facts: dict, ticker: str) -> pl.DataFrame:
    """Company facts JSON → one row per (period end, timeframe) in the panel's financials schema."""
    f = facts.get("facts", {})
    qcols: dict[str, dict] = {}
    acols: dict[str, dict] = {}
    fp_of: dict[str, str] = {}
    filed_q: dict[str, str] = {}
    for col, concepts in {**FLOW, **SHARES}.items():
        for concept in concepts:
            ents = _concept_entries(f, concept)
            if not ents:
                continue
            q, a = _flow_series(ents, additive=col not in SHARES)
            for e, (v, fl, fp) in q.items():
                qcols.setdefault(col, {}).setdefault(e, v)
                fp_of.setdefault(e, fp or "")
                if col in (I + "revenues", I + "net_income_loss"):
                    filed_q[e] = min(filed_q.get(e, fl), fl)
            for e, (v, fl, s) in a.items():
                acols.setdefault(col, {}).setdefault(e, (v, fl, s))
    inst: dict[str, dict] = {}
    for col, concepts in INSTANT.items():
        for concept in concepts:
            for (s, e), r in _first_reported(_concept_entries(f, concept)).items():
                if s is None:
                    inst.setdefault(col, {}).setdefault(e, (float(r["val"]), r["filed"]))
    rows = []
    q_ends = set(qcols.get(I + "net_income_loss", {})) | set(qcols.get(I + "revenues", {}))
    a_ends = set(acols.get(I + "net_income_loss", {})) | set(acols.get(I + "revenues", {}))
    for tf, ends in (("quarterly", q_ends), ("annual", a_ends)):
        for e in ends:
            row = {"ticker": ticker, "timeframe": tf, "end_date": e}
            filed = []
            for col in list(FLOW) + list(SHARES):
                if tf == "quarterly":
                    v = qcols.get(col, {}).get(e)
                    row[col] = v
                else:
                    t = acols.get(col, {}).get(e)
                    row[col] = t[0] if t else None
                    if t:
                        filed.append(t[1]); row["start_date"] = t[2]
            for col in INSTANT:
                t = inst.get(col, {}).get(e)
                row[col] = t[0] if t else None
                if t:
                    filed.append(t[1])
            if tf == "quarterly":
                fq = filed_q.get(e)
                row["filing_date"] = fq or (min(filed) if filed else None)
                row["fiscal_period"] = fp_of.get(e) or ""
            else:
                row["filing_date"] = min(filed) if filed else None
                row["fiscal_period"] = "FY"
            rows.append(row)
    if not rows:
        return pl.DataFrame()
    df = pl.DataFrame(rows, infer_schema_length=None)
    if B + "liabilities" in df.columns and "_lse" in df.columns:
        df = df.with_columns(pl.coalesce(pl.col(B + "liabilities"), pl.col("_lse") - pl.col(B + "equity")).alias(B + "liabilities"))
    df = df.drop([c for c in ("_lse",) if c in df.columns])
    for c in ("end_date", "filing_date", "start_date"):
        if c in df.columns:
            df = df.with_columns(pl.col(c).cast(pl.Utf8).str.to_date("%Y-%m-%d", strict=False))
    df = df.with_columns(fiscal_year=pl.col("end_date").dt.year().cast(pl.Utf8), source=pl.lit("edgar"))
    return df.sort(["timeframe", "end_date"])


class EdgarFundamentals:
    def __init__(self, root: Path, user_agent: str, out_path: Path | None = None, rate_per_s: float = 8.0,
                 workers: int = 4):
        self.root = Path(root)
        self.out_path = Path(out_path) if out_path else self.root / "financials.parquet"
        self.rows_dir = self.root / "rows"
        self.rows_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.ua = {"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}
        self.rl = _RateLimit(rate_per_s)
        self.workers = workers
        self.session = requests.Session()

    # ---- state ------------------------------------------------------------------------------
    def state(self) -> dict:
        try:
            return json.loads(self.state_path.read_text())
        except Exception:
            return {"last_index_date": None, "fetched": {}}

    def save_state(self, st: dict) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=1))
        tmp.replace(self.state_path)

    def _get(self, url: str, timeout: float = 60.0) -> requests.Response:
        for attempt in range(4):
            self.rl.wait()
            try:
                r = self.session.get(url, headers=self.ua, timeout=timeout)
            except requests.RequestException:
                time.sleep(2 ** attempt); continue
            if r.status_code in (429, 503):
                time.sleep(5 * (attempt + 1)); continue
            return r
        raise RuntimeError(f"SEC request kept failing: {url}")

    def cik_map(self) -> dict[str, int]:
        p = self.root / "company_tickers.json"
        if not p.exists() or time.time() - p.stat().st_mtime > 86400:
            r = self._get(TICKERS_URL)
            r.raise_for_status()
            p.write_text(r.text)
        d = json.loads(p.read_text())
        return {v["ticker"].upper().replace(".", "-"): int(v["cik_str"]) for v in d.values()}

    # ---- fetch ------------------------------------------------------------------------------
    def fetch_company(self, cik: int, tickers: list[str]) -> int:
        r = self._get(FACTS_URL.format(cik=cik))
        if r.status_code == 404:
            return 0
        r.raise_for_status()
        facts = r.json()
        parts = [facts_to_rows(facts, t) for t in tickers]
        parts = [p for p in parts if p.height]
        if not parts:
            return 0
        df = pl.concat(parts, how="diagonal_relaxed")
        df.write_parquet(self.rows_dir / f"{cik}.parquet", compression="zstd")
        return df.height

    def filers_since(self, since: date, until: date) -> set[int]:
        """CIKs that filed a 10-Q/10-K (or amendment) on (since, until] per EDGAR's daily form index."""
        ciks: set[int] = set()
        d = since + timedelta(days=1)
        while d <= until:
            if d.weekday() < 5:
                url = DAILY_INDEX_URL.format(y=d.year, q=(d.month - 1) // 3 + 1, ymd=d.strftime("%Y%m%d"))
                r = self._get(url)
                if r.status_code == 200:
                    for line in r.text.splitlines():
                        form = line[:17].strip()
                        if form in FORMS:
                            parts = line[17:].split()
                            # columns: company name (spaces) … CIK, date, file name
                            if len(parts) >= 3 and parts[-3].isdigit():
                                ciks.add(int(parts[-3]))
            d += timedelta(days=1)
        return ciks

    def update(self, tickers: Iterable[str], full: bool = False, progress: bool = True) -> dict:
        """Fetch what is missing or newly filed. Returns counts."""
        st = self.state()
        cmap = self.cik_map()
        want: dict[int, list[str]] = {}
        unmapped = []
        for t in dict.fromkeys(x.upper().replace(".", "-") for x in tickers):
            cik = cmap.get(t)
            if cik is None:
                unmapped.append(t)
            else:
                want.setdefault(cik, []).append(t)
        have = {int(p.stem) for p in self.rows_dir.glob("*.parquet")}
        todo = {c for c in want if c not in have} if not full else set(want)
        today = date.today()
        last = st.get("last_index_date")
        if last and not full:
            filed = self.filers_since(date.fromisoformat(last), today)
            todo |= {c for c in want if c in filed}
        n_rows, fails = 0, []
        t0 = time.time()

        def job(cik):
            try:
                return cik, self.fetch_company(cik, want[cik]), None
            except Exception as e:
                return cik, 0, str(e)[:120]
        with ThreadPoolExecutor(self.workers) as ex:
            for i, (cik, n, err) in enumerate(ex.map(job, sorted(todo)), 1):
                n_rows += n
                if err:
                    fails.append((cik, err))
                else:
                    st["fetched"][str(cik)] = str(today)
                if progress and i % 100 == 0:
                    logger.info(f"edgar: {i}/{len(todo)} companies · {time.time() - t0:.0f}s")
        st["last_index_date"] = str(today)
        st["tickers"] = {str(c): v for c, v in want.items()}
        self.save_state(st)
        return {"companies_wanted": len(want), "fetched": len(todo) - len(fails), "failed": len(fails),
                "unmapped_tickers": len(unmapped), "rows": n_rows, "seconds": round(time.time() - t0, 1),
                "fail_examples": fails[:3], "unmapped_examples": unmapped[:10]}

    def build(self, massive_financials: Path | None = None) -> pl.DataFrame:
        """All cached company rows (only tickers currently requested) ⊕ Massive history where EDGAR has none."""
        st = self.state()
        keep = {t for v in st.get("tickers", {}).values() for t in v}
        parts = [pl.read_parquet(p) for p in sorted(self.rows_dir.glob("*.parquet"))]
        ed = pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()
        if keep and ed.height:
            ed = ed.filter(pl.col("ticker").is_in(list(keep)))
        if massive_financials is not None and Path(massive_financials).exists():
            ms = pl.read_parquet(massive_financials).filter(pl.col("timeframe").is_in(["quarterly", "annual"]))
            for c in ("end_date", "filing_date", "start_date"):
                if c in ms.columns and ms.schema[c] != pl.Date:
                    ms = ms.with_columns(pl.col(c).cast(pl.Utf8).str.slice(0, 10).str.to_date("%Y-%m-%d", strict=False))
            ms = ms.with_columns(source=pl.lit("massive"))
            if ed.height:
                key = ["ticker", "end_date", "timeframe"]
                # 1) blank EDGAR fields (custom extension tags are not in company facts) ← Massive's value
                fill = [c for c in ed.columns if "__" in c and c in ms.columns]
                ed = (ed.join(ms.select(key + fill).unique(subset=key, keep="last"), on=key, how="left", suffix="_ms")
                        .with_columns([pl.coalesce(pl.col(c), pl.col(c + "_ms")).alias(c) for c in fill])
                        .drop([c + "_ms" for c in fill]))
                # 2) periods EDGAR lacks entirely
                ms = ms.join(ed.select(key), on=key, how="anti")
            ed = pl.concat([ed, ms], how="diagonal_relaxed") if ed.height else ms
        out = ed.sort(["ticker", "timeframe", "end_date"]) if ed.height else ed
        p = self.out_path
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        out.write_parquet(tmp, compression="zstd")
        tmp.replace(p)
        return out


def store(cfg) -> EdgarFundamentals:
    ua = (cfg.get("api_keys", {}) or {}).get("sec_user_agent") or "research-bot research@example.com"
    return EdgarFundamentals(cfg.path("data_raw") / "edgar", ua, out_path=edgar_path(cfg))


def edgar_path(cfg) -> Path:
    return cfg.path("data_bronze") / "edgar" / "financials.parquet"


def fundamentals_path(cfg) -> Path:
    """The financials table the panel should read: EDGAR-merged if built, else the old Massive one."""
    p = edgar_path(cfg)
    return p if p.exists() else cfg.path("data_bronze") / "massive" / "financials.parquet"
