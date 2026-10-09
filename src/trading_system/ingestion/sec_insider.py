"""Insider trading from the SEC's Form 3/4/5 data sets (free, quarterly, January 2006 →).

Source: https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets — the XML
part of every Form 3/4/5, flattened into SUBMISSION / REPORTINGOWNER / NONDERIV_TRANS tables.

Kept: open-market purchases (code P) and sales (code S) of non-derivative securities on Forms 4/4A.
Each trade becomes usable the day AFTER its filing date (Form 4 is due within two business days of
the trade). Trades are tagged *routine* — carrying no information per Cohen, Malloy & Pomorski
(2012, "Decoding Inside Information") — when the insider traded in the same calendar month in each
of the previous three years, or (from 2023) when the filing flags a Rule 10b5-1 plan. The rest are
*opportunistic*: in CMP these earned ~82 bp/month (value-weighted) and carried all the predictive power.

Features (``INSIDER_FEATURES``), per ticker and trading day, over the trailing 90/180 calendar days:
  ins_opp_buys_90   number of opportunistic purchase filings
  ins_opp_sells_90  number of opportunistic sale filings
  ins_net_180       (opportunistic $ bought − $ sold) / 21-day average dollar volume
  ins_officer_buy_180  1 if an officer/director made an opportunistic purchase in 180 days
"""
from __future__ import annotations

import io
import zipfile
from datetime import timedelta
from pathlib import Path

import polars as pl

from ..utils import get_logger

logger = get_logger(__name__)

INSIDER_FEATURES = ["ins_opp_buys_90", "ins_opp_sells_90", "ins_net_180", "ins_officer_buy_180"]


def _read(z: zipfile.ZipFile, name: str, cols: list[str]) -> pl.DataFrame:
    with z.open(name) as fh:
        raw = fh.read()
    df = pl.read_csv(io.BytesIO(raw), separator="\t", quote_char=None, infer_schema_length=0, truncate_ragged_lines=True,
                     ignore_errors=True)
    return df.select([c for c in cols if c in df.columns])


def parse_quarter(path: Path) -> pl.DataFrame:
    """One quarterly zip → open-market P/S trades with filer, issuer and relationship."""
    with zipfile.ZipFile(path) as z:
        sub = _read(z, "SUBMISSION.tsv", ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK",
                                          "ISSUERTRADINGSYMBOL", "AFF10B5ONE"])
        own = _read(z, "REPORTINGOWNER.tsv", ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP"])
        tr = _read(z, "NONDERIV_TRANS.tsv", ["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES",
                                             "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD"])
    tr = tr.filter(pl.col("TRANS_CODE").is_in(["P", "S"]))
    sub = sub.filter(pl.col("DOCUMENT_TYPE").is_in(["4", "4/A"]))
    if "AFF10B5ONE" not in sub.columns:
        sub = sub.with_columns(AFF10B5ONE=pl.lit(None, dtype=pl.Utf8))
    own = own.group_by("ACCESSION_NUMBER").agg(pl.col("RPTOWNERCIK").first(), pl.col("RPTOWNER_RELATIONSHIP").first())
    df = tr.join(sub, on="ACCESSION_NUMBER", how="inner").join(own, on="ACCESSION_NUMBER", how="left")
    d = lambda c: pl.col(c).str.to_date("%d-%b-%Y", strict=False)
    return df.select(
        accession="ACCESSION_NUMBER", filing_date=d("FILING_DATE"), trans_date=d("TRANS_DATE"),
        issuer_cik=pl.col("ISSUERCIK").cast(pl.Int64, strict=False), symbol=pl.col("ISSUERTRADINGSYMBOL").str.to_uppercase(),
        owner_cik=pl.col("RPTOWNERCIK").cast(pl.Int64, strict=False), relationship="RPTOWNER_RELATIONSHIP",
        code="TRANS_CODE", shares=pl.col("TRANS_SHARES").cast(pl.Float64, strict=False),
        price=pl.col("TRANS_PRICEPERSHARE").cast(pl.Float64, strict=False),
        plan_10b5_1=pl.col("AFF10B5ONE").str.to_lowercase().is_in(["1", "true"]),
    ).drop_nulls(["filing_date", "issuer_cik", "shares"])


def build_trades(raw_dir: Path, out: Path, cik_to_ticker: dict[int, str]) -> pl.DataFrame:
    """All quarters → one deduplicated trades table with the routine/opportunistic tag."""
    parts = []
    for p in sorted(raw_dir.glob("*_form345.zip")):
        try:
            parts.append(parse_quarter(p))
        except Exception as e:  # a corrupt or partial download should not stop the build
            logger.warning(f"insider: {p.name} unreadable ({str(e)[:80]})")
    df = pl.concat(parts, how="vertical_relaxed").unique(subset=["accession", "trans_date", "code", "shares", "price"])
    df = df.with_columns(value=pl.col("shares") * pl.col("price").fill_null(0.0),
                         ticker=pl.col("issuer_cik").replace_strict(cik_to_ticker, default=None),
                         officer=pl.col("relationship").str.contains("(?i)officer|director").fill_null(False))
    df = df.with_columns(ticker=pl.coalesce(pl.col("ticker"), pl.col("symbol").str.replace_all(".", "-", literal=True)))
    # CMP routine rule: traded in the same calendar month in each of the three previous years
    td = pl.coalesce(pl.col("trans_date"), pl.col("filing_date"))
    df = df.with_columns(y=td.dt.year(), m=td.dt.month())
    seen = df.select("owner_cik", "issuer_cik", "y", "m").unique()
    for k in (1, 2, 3):
        df = df.join(seen.with_columns((pl.col("y") + k).alias("y"), pl.lit(True).alias(f"_prev{k}")),
                     on=["owner_cik", "issuer_cik", "y", "m"], how="left")
    df = df.with_columns(routine=(pl.col("_prev1") & pl.col("_prev2") & pl.col("_prev3")).fill_null(False) | pl.col("plan_10b5_1").fill_null(False))
    df = df.drop(["_prev1", "_prev2", "_prev3", "y", "m"]).sort(["ticker", "filing_date"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out, compression="zstd")
    logger.info(f"insider: {df.height:,} open-market trades · {df['ticker'].n_unique():,} tickers · "
                f"{df['filing_date'].min()} → {df['filing_date'].max()} · routine {df['routine'].mean():.0%}")
    return df


def insider_features(panel: pl.DataFrame, trades: pl.DataFrame) -> pl.DataFrame:
    """Join INSIDER_FEATURES onto (date, ticker) panel rows — counts/values over trailing calendar windows,
    each trade usable from the first trading session after filing_date + 1 day."""
    T, D = "ticker", "date"
    if trades.height == 0:
        return panel.with_columns([pl.lit(None, dtype=pl.Float64).alias(c) for c in INSIDER_FEATURES])
    opp = trades.filter(~pl.col("routine"))
    ev = opp.select(T, avail=pl.col("filing_date") + timedelta(days=1), buy=(pl.col("code") == "P").cast(pl.Int32),
                    sell=(pl.col("code") == "S").cast(pl.Int32),
                    net=pl.when(pl.col("code") == "P").then(pl.col("value")).otherwise(-pl.col("value")),
                    obuy=((pl.col("code") == "P") & pl.col("officer")).cast(pl.Int32), acc="accession")
    # one event per filing (an insider's filing often has several rows)
    ev = ev.group_by([T, "avail", "acc"]).agg(pl.col("buy").max(), pl.col("sell").max(), pl.col("net").sum(), pl.col("obuy").max())
    cal = panel.select(D).unique().sort(D)
    ev = (ev.sort("avail").join_asof(cal.with_columns(sess=pl.col(D)), left_on="avail", right_on=D, strategy="forward",
                                     check_sortedness=False)
            .drop_nulls("sess").group_by([T, "sess"]).agg(pl.col("buy").sum(), pl.col("sell").sum(), pl.col("net").sum(),
                                                          pl.col("obuy").max()).rename({"sess": D}))
    j = panel.select(T, D, "log_dv_21").join(ev, on=[T, D], how="left").with_columns(
        [pl.col(c).fill_null(0) for c in ("buy", "sell", "net", "obuy")]).sort([T, D])
    j = j.with_columns(
        ins_opp_buys_90=pl.col("buy").rolling_sum_by(D, "90d").over(T).cast(pl.Float64),
        ins_opp_sells_90=pl.col("sell").rolling_sum_by(D, "90d").over(T).cast(pl.Float64),
        ins_net_180=pl.col("net").rolling_sum_by(D, "180d").over(T) / (pl.col("log_dv_21").exp() + 1.0),
        ins_officer_buy_180=(pl.col("obuy").rolling_sum_by(D, "180d").over(T) > 0).cast(pl.Float64),
    )
    first = trades["filing_date"].min()
    j = j.with_columns([pl.when(pl.col(D) <= first + timedelta(days=180)).then(None).otherwise(pl.col(c)).alias(c)
                        for c in INSIDER_FEATURES])
    return panel.join(j.select(T, D, *INSIDER_FEATURES), on=[T, D], how="left")


def trades_path(cfg) -> Path:
    return cfg.path("data_bronze") / "sec" / "insider_trades.parquet"
