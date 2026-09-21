"""The alpha panel: one point-in-time feature matrix for every tradable name.

Design
------
* **Whole tradable universe, deep history.**  Prices come from
  ``bronze/massive/ohlcv_deep.parquet`` (1000 names, 1970→, Massive bars for the
  last two years spliced onto yfinance before that) with ``bronze/ohlcv_daily``
  as the fallback, so training sees ~26 years × ~1000 names instead of the 374
  names the old gold matrix carries.
* **Everything is a polars window expression.**  No per-ticker Python loops;
  the full panel (≈6M rows × ~60 features) builds in well under a minute, which
  is what makes a daily rebuild and a quarterly-refit walk-forward affordable.
* **Point-in-time by construction.**  Fundamentals join on the SEC *filing*
  date (+1 day), news on the day *after* publication, short interest ten days
  after settlement (FINRA's own publication lag).  Labels look strictly forward
  from ``adj_close``; features look strictly backward.
* **Features stay raw here.**  The model rank-transforms them within each date
  (see ``model.prepare``), so the panel is also usable for portfolio maths
  (vol, liquidity, price) without un-ranking anything.

Columns: ``date, ticker, close, adj_close, volume`` + ``FEATURE_COLS`` +
``fwd_{h}`` (raw forward return) and ``y_{h}`` (per-date demeaned, gaussianised
rank of ``fwd_{h}``) for each horizon in ``HORIZONS``.
"""
from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl

from ..utils import get_logger

logger = get_logger(__name__)

HORIZONS = (5, 21, 63)
_ANN = math.sqrt(252.0)

# ── feature catalogue ──────────────────────────────────────────────────────────
PRICE_FEATURES = [
    "ret_1", "ret_5", "ret_21", "ret_63", "ret_126", "ret_252", "mom_12_1",
    "vol_21", "vol_63", "vol_252", "dvol_63", "skew_63", "kurt_63", "max_ret_21", "min_ret_21",
    "log_dv_21", "amihud_21", "vol_ratio_5_63", "dist_52w_high", "dist_52w_low",
    "sma_gap_50", "sma_gap_200", "up_frac_63", "overnight_21", "intraday_21",
    "parkinson_21", "range_pos_21", "beta_63", "idio_vol_63", "log_price", "spread_ar_21",
]
FUND_FEATURES = [
    "ep_ttm", "bm", "sp_ttm", "cfp_ttm", "gp_assets", "roe_ttm", "roa_ttm",
    "asset_growth", "rev_growth", "accruals", "leverage", "log_mcap", "fund_age_days",
]
NEWS_FEATURES = ["news_n_5", "news_n_21", "news_sent_5", "news_sent_21", "news_n_z"]
SHORT_FEATURES = ["si_days_to_cover", "si_ratio", "si_chg", "sv_ratio_5"]
MARKET_FEATURES = ["mkt_ret_21", "mkt_ret_63", "mkt_vol_21", "breadth_200", "dispersion_21", "sector_rel_63"]
from .regime import MACRO_FEATURES  # noqa: E402  (date-level macro/fragility state, see regime.py)
FEATURE_COLS = PRICE_FEATURES + FUND_FEATURES + NEWS_FEATURES + SHORT_FEATURES + MARKET_FEATURES + MACRO_FEATURES
# constant within a date → they cannot be within-date ranks. Tested 2003→2026 as raw model inputs: the
# regime-aware walk-forward was WORSE (63d IC 0.074 → 0.065, ICIR 0.76 → 0.51, paired t −7.8) — the ranker
# overfits regime-specific cross-sectional patterns it has seen only a handful of times. So they stay in the
# panel for the regime layer (calibration weights, reporting) but are NOT default model features.
DATE_LEVEL_FEATURES = [c for c in MARKET_FEATURES if c != "sector_rel_63"] + MACRO_FEATURES
MODEL_FEATURES = [c for c in FEATURE_COLS if c not in DATE_LEVEL_FEATURES]
BASE_COLS = ["date", "ticker", "close", "adj_close", "volume", "sector"]
CONTEXT_COLS = ["mkt_trend_200"]          # kept for the book's regime overlay, NOT a model feature


# ── inputs ─────────────────────────────────────────────────────────────────────

def load_prices(cfg, tickers: Iterable[str] | None = None, start: str | date = "1998-01-01",
                end: date | None = None, deep_path: Path | None = None) -> pl.DataFrame:
    """Long OHLCV: the Massive deep table when present, else bronze ``ohlcv_daily``."""
    bronze = cfg.path("data_bronze")
    deep = deep_path or (bronze / "massive" / "ohlcv_deep.parquet")
    src = deep if deep.exists() else bronze / "ohlcv_daily.parquet"
    lf = pl.scan_parquet(src).select(["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"])
    lf = lf.filter(pl.col("date") >= pl.lit(start if isinstance(start, date) else date.fromisoformat(str(start))))
    if end is not None:
        lf = lf.filter(pl.col("date") <= pl.lit(end))
    if tickers is not None:
        lf = lf.filter(pl.col("ticker").is_in(list(tickers)))
    px = (lf.with_columns(pl.col("date").cast(pl.Date), pl.col("volume").cast(pl.Float64))
            .filter(pl.col("adj_close") > 0, pl.col("close") > 0)
            .unique(subset=["date", "ticker"], keep="last")
            .sort(["ticker", "date"])
            .collect())
    # the deep table is refreshed weekly; the whole-market table daily → append the newest sessions
    daily = bronze / "massive" / "ohlcv_all.parquet"
    if src == deep and daily.exists():
        dmax = px["date"].max()
        fresh = (pl.scan_parquet(daily).select(["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"])
                   .filter(pl.col("date") > pl.lit(dmax), pl.col("ticker").is_in(px["ticker"].unique().to_list()),
                           pl.col("adj_close") > 0, pl.col("close") > 0)
                   .with_columns(pl.col("date").cast(pl.Date), pl.col("volume").cast(pl.Float64)).collect())
        if end is not None:
            fresh = fresh.filter(pl.col("date") <= end)
        if fresh.height:
            px = pl.concat([px, fresh.select(px.columns)], how="vertical_relaxed").sort(["ticker", "date"])
            logger.info(f"prices: appended {fresh.height:,} rows after {dmax} from ohlcv_all → {px['date'].max()}")
    px = sanitize_prices(px)
    logger.info(f"prices: {px.height:,} rows · {px['ticker'].n_unique()} tickers · "
                f"{px['date'].min()} → {px['date'].max()} ({src.name})")
    return px


BREAK_UP, BREAK_DOWN, MIN_PRICE = 4.0, -0.85, 0.01


def sanitize_prices(px: pl.DataFrame) -> pl.DataFrame:
    """Drop the part of a ticker's history that belongs to a *different* security.

    Free price feeds reuse symbols (BNY, CHRD, SOLS…): the series jumps ×250 or ÷1000 on the day
    the new listing starts. Any single-day move above +400% or below −85% is treated as a series
    break and everything before the *last* break is discarded, so features never straddle two
    companies and the equal-weight benchmark cannot post a +487,399% day. Sub-cent prints go too.
    """
    px = px.filter(pl.col("adj_close") >= MIN_PRICE, pl.col("close") >= MIN_PRICE).sort(["ticker", "date"])
    r = pl.col("adj_close") / pl.col("adj_close").shift(1).over("ticker") - 1
    brk = px.with_columns(brk=((r > BREAK_UP) | (r < BREAK_DOWN)).fill_null(False))
    last_break = (brk.filter(pl.col("brk")).group_by("ticker").agg(pl.col("date").max().alias("segment_start")))
    if last_break.height == 0:
        return px
    out = (px.join(last_break, on="ticker", how="left")
             .filter(pl.col("segment_start").is_null() | (pl.col("date") >= pl.col("segment_start")))
             .drop("segment_start"))
    logger.info(f"prices: {last_break.height} series breaks (symbol reuse) → dropped {px.height - out.height:,} pre-break rows "
                f"({', '.join(last_break.sort('ticker')['ticker'].head(8).to_list())}…)")
    return out


def _sector_map(cfg) -> pl.DataFrame:
    """ticker → coarse sector from the Massive SIC code (2-digit division, ~10 buckets)."""
    p = cfg.path("data_bronze") / "massive" / "details.parquet"
    if not p.exists():
        return pl.DataFrame({"ticker": pl.Series([], dtype=pl.Utf8), "sector": pl.Series([], dtype=pl.Utf8)})
    d = pl.read_parquet(p, columns=["ticker", "sic_code"]).drop_nulls("sic_code")
    sic = pl.col("sic_code").cast(pl.Utf8).str.slice(0, 2).cast(pl.Int32, strict=False)
    sector = (pl.when(sic < 10).then(pl.lit("agri"))
                .when(sic < 15).then(pl.lit("mining_energy"))
                .when(sic < 18).then(pl.lit("construction"))
                .when(sic < 40).then(pl.lit("manufacturing"))
                .when(sic < 50).then(pl.lit("transport_utilities"))
                .when(sic < 52).then(pl.lit("wholesale"))
                .when(sic < 60).then(pl.lit("retail"))
                .when(sic < 68).then(pl.lit("finance"))
                .when(sic < 90).then(pl.lit("services"))
                .otherwise(pl.lit("other")))
    return d.with_columns(sector=sector).select(["ticker", "sector"]).unique(subset=["ticker"])


# ── price features ─────────────────────────────────────────────────────────────

def price_features(px: pl.DataFrame) -> pl.DataFrame:
    """Backward-looking price/volume features, per ticker (input sorted by ticker, date)."""
    T = "ticker"
    ac = pl.col("adj_close")
    r1 = (ac / ac.shift(1).over(T)).log()
    dv = pl.col("close") * pl.col("volume")

    def roll_mean(e, n, mn=None):
        return e.rolling_mean(n, min_samples=mn or max(2, n // 2)).over(T)

    def roll_std(e, n, mn=None):
        return e.rolling_std(n, min_samples=mn or max(3, n // 2)).over(T)

    out = px.with_columns(r1=r1, dollar_vol=dv,
                          ovn=(pl.col("open") / pl.col("close").shift(1).over(T)).log(),
                          intra=(pl.col("close") / pl.col("open")).log(),
                          hl2=(pl.col("high") / pl.col("low")).log().pow(2))
    # Abdi-Ranaldo spread proxy: 2*sqrt(max(E[(c - (h+l)/2)(c - (h1+l1)/2)], 0)) on log prices
    c = pl.col("close").log()
    mid = ((pl.col("high").log() + pl.col("low").log()) / 2)
    ar_term = (c - mid) * (c - mid.shift(-1).over(T))          # uses next day's mid → shift back by 1 below
    out = out.with_columns(ar_term=ar_term.shift(1).over(T))     # now strictly backward-looking
    out = out.with_columns(
        ret_1=(ac / ac.shift(1).over(T) - 1),
        ret_5=(ac / ac.shift(5).over(T) - 1),
        ret_21=(ac / ac.shift(21).over(T) - 1),
        ret_63=(ac / ac.shift(63).over(T) - 1),
        ret_126=(ac / ac.shift(126).over(T) - 1),
        ret_252=(ac / ac.shift(252).over(T) - 1),
        mom_12_1=(ac.shift(21).over(T) / ac.shift(252).over(T) - 1),
        vol_21=roll_std(pl.col("r1"), 21) * _ANN,
        vol_63=roll_std(pl.col("r1"), 63) * _ANN,
        vol_252=roll_std(pl.col("r1"), 252) * _ANN,
        dvol_63=roll_std(pl.min_horizontal(pl.col("r1"), pl.lit(0.0)), 63) * _ANN,
        skew_63=pl.col("r1").rolling_skew(63).over(T),
        kurt_63=(roll_mean(pl.col("r1").pow(4), 63) / roll_std(pl.col("r1"), 63).pow(4)),
        max_ret_21=pl.col("r1").rolling_max(21, min_samples=10).over(T),
        min_ret_21=pl.col("r1").rolling_min(21, min_samples=10).over(T),
        log_dv_21=roll_mean(pl.col("dollar_vol"), 21).log1p(),
        amihud_21=(roll_mean(pl.col("r1").abs() / (pl.col("dollar_vol") + 1.0), 21) * 1e6).log1p(),
        vol_ratio_5_63=(roll_mean(pl.col("volume"), 5, 3) / (roll_mean(pl.col("volume"), 63) + 1.0)),
        dist_52w_high=(ac / ac.rolling_max(252, min_samples=126).over(T) - 1),
        dist_52w_low=(ac / ac.rolling_min(252, min_samples=126).over(T) - 1),
        sma_gap_50=(ac / roll_mean(ac, 50) - 1),
        sma_gap_200=(ac / roll_mean(ac, 200) - 1),
        up_frac_63=roll_mean((pl.col("r1") > 0).cast(pl.Float64), 63),
        overnight_21=pl.col("ovn").rolling_sum(21, min_samples=10).over(T),
        intraday_21=pl.col("intra").rolling_sum(21, min_samples=10).over(T),
        parkinson_21=(roll_mean(pl.col("hl2"), 21) / (4 * math.log(2))).sqrt() * _ANN,
        range_pos_21=((pl.col("close") - pl.col("low").rolling_min(21, min_samples=10).over(T))
                      / (pl.col("high").rolling_max(21, min_samples=10).over(T)
                         - pl.col("low").rolling_min(21, min_samples=10).over(T) + 1e-9)),
        log_price=pl.col("close").log(),
        spread_ar_21=(2 * pl.max_horizontal(roll_mean(pl.col("ar_term"), 21), pl.lit(0.0)).sqrt()),
    )
    # beta / idiosyncratic vol vs the equal-weight market of the panel itself
    out = out.with_columns(mkt_r1=pl.col("r1").mean().over("date"))
    cov = roll_mean(pl.col("r1") * pl.col("mkt_r1"), 63) - roll_mean(pl.col("r1"), 63) * roll_mean(pl.col("mkt_r1"), 63)
    var_m = roll_mean(pl.col("mkt_r1").pow(2), 63) - roll_mean(pl.col("mkt_r1"), 63).pow(2)
    var_r = roll_mean(pl.col("r1").pow(2), 63) - roll_mean(pl.col("r1"), 63).pow(2)
    out = out.with_columns(beta_63=(cov / (var_m + 1e-12)))
    out = out.with_columns(
        idio_vol_63=pl.max_horizontal(var_r - pl.col("beta_63").pow(2) * var_m, pl.lit(0.0)).sqrt() * _ANN)
    return out.drop(["ovn", "intra", "hl2", "ar_term"])


# ── fundamentals (point-in-time on filing date) ───────────────────────────────

def fundamental_features(fin: pl.DataFrame) -> pl.DataFrame:
    """Per-filing fundamental ratios (before the price scaling), keyed by ``avail_date``.

    TTM flows = sum of the last four quarterly filings when they span ~a year, else
    the latest annual filing. Levels (assets, equity…) come from the latest filing.
    Everything is known on ``filing_date`` → available the next day.
    """
    I, B, C = "income_statement__", "balance_sheet__", "cash_flow_statement__"

    def col(name):
        return pl.col(name).cast(pl.Float64) if name in fin.columns else pl.lit(None, dtype=pl.Float64)

    q = (fin.filter(pl.col("timeframe").is_in(["quarterly", "annual"]))
            .select(ticker="ticker", timeframe="timeframe",
                    end_date=_date_col(fin, "end_date"), filing_date=_date_col(fin, "filing_date"),
                    rev=col(I + "revenues"), ni=col(I + "net_income_loss"), gp=col(I + "gross_profit"),
                    ocf=col(C + "net_cash_flow_from_operating_activities"),
                    assets=col(B + "assets"), equity=col(B + "equity"), liab=col(B + "liabilities"),
                    shares=pl.coalesce(col(I + "diluted_average_shares"), col(I + "basic_average_shares")))
            .drop_nulls(["end_date", "filing_date"])
            .sort(["ticker", "end_date", "filing_date"]))
    T = "ticker"
    qq = q.filter(pl.col("timeframe") == "quarterly")
    span = (pl.col("end_date") - pl.col("end_date").shift(3).over(T)).dt.total_days()
    ok = (span >= 240) & (span <= 310)          # four consecutive quarter-ends are ~273 days apart
    ttm = qq.with_columns(
        rev_ttm=pl.when(ok).then(pl.col("rev").rolling_sum(4).over(T)),
        ni_ttm=pl.when(ok).then(pl.col("ni").rolling_sum(4).over(T)),
        gp_ttm=pl.when(ok).then(pl.col("gp").rolling_sum(4).over(T)),
        ocf_ttm=pl.when(ok).then(pl.col("ocf").rolling_sum(4).over(T)),
        assets_yoy=pl.col("assets").shift(4).over(T), rev_yoy=pl.col("rev").rolling_sum(4).over(T).shift(4).over(T),
    )
    ann = q.filter(pl.col("timeframe") == "annual").select(
        T, "end_date", "filing_date", rev_a="rev", ni_a="ni", gp_a="gp", ocf_a="ocf",
        assets_a="assets", equity_a="equity", liab_a="liab", shares_a="shares")
    ann = ann.with_columns(assets_a_prev=pl.col("assets_a").shift(1).over(T), rev_a_prev=pl.col("rev_a").shift(1).over(T))
    # union quarterly + annual filings into one PIT stream, forward-filling within ticker
    rows = pl.concat([ttm.select(T, "end_date", "filing_date", "rev_ttm", "ni_ttm", "gp_ttm", "ocf_ttm",
                                 "assets", "equity", "liab", "shares", "assets_yoy", "rev_yoy"),
                      ann.select(T, "end_date", "filing_date",
                                 rev_ttm="rev_a", ni_ttm="ni_a", gp_ttm="gp_a", ocf_ttm="ocf_a",
                                 assets="assets_a", equity="equity_a", liab="liab_a", shares="shares_a",
                                 assets_yoy="assets_a_prev", rev_yoy="rev_a_prev")],
                     how="vertical_relaxed").sort([T, "filing_date", "end_date"])
    rows = rows.with_columns([pl.col(c).forward_fill().over(T) for c in
                              ["rev_ttm", "ni_ttm", "gp_ttm", "ocf_ttm", "assets", "equity", "liab", "shares",
                               "assets_yoy", "rev_yoy"]])
    rows = rows.with_columns(avail_date=pl.col("filing_date") + pl.duration(days=1))
    out = rows.with_columns(
        gp_assets=pl.col("gp_ttm") / pl.col("assets"),
        roe_ttm=pl.col("ni_ttm") / pl.col("equity"),
        roa_ttm=pl.col("ni_ttm") / pl.col("assets"),
        asset_growth=pl.col("assets") / pl.col("assets_yoy") - 1,
        rev_growth=pl.col("rev_ttm") / pl.col("rev_yoy") - 1,
        accruals=(pl.col("ni_ttm") - pl.col("ocf_ttm")) / pl.col("assets"),
        leverage=pl.col("liab") / pl.col("assets"),
    ).select(T, "avail_date", "filing_date", "ni_ttm", "rev_ttm", "ocf_ttm", "equity", "shares",
             "gp_assets", "roe_ttm", "roa_ttm", "asset_growth", "rev_growth", "accruals", "leverage")
    return out.unique(subset=[T, "avail_date"], keep="last").sort([T, "avail_date"])


def _join_fundamentals(panel: pl.DataFrame, fund: pl.DataFrame) -> pl.DataFrame:
    if fund.height == 0:
        return panel.with_columns([pl.lit(None, dtype=pl.Float64).alias(c) for c in FUND_FEATURES])
    j = panel.sort(["ticker", "date"]).join_asof(
        fund.sort(["ticker", "avail_date"]), left_on="date", right_on="avail_date", by="ticker",
        strategy="backward", tolerance=timedelta(days=450), check_sortedness=False)
    mcap = pl.col("shares") * pl.col("close")
    return j.with_columns(
        ep_ttm=pl.col("ni_ttm") / mcap, bm=pl.col("equity") / mcap, sp_ttm=pl.col("rev_ttm") / mcap,
        cfp_ttm=pl.col("ocf_ttm") / mcap, log_mcap=mcap.log1p(),
        fund_age_days=(pl.col("date") - pl.col("filing_date")).dt.total_days().cast(pl.Float64),
    ).drop(["avail_date", "filing_date", "ni_ttm", "rev_ttm", "ocf_ttm", "equity", "shares"])


# ── news / short interest ─────────────────────────────────────────────────────

def _date_col(df: pl.DataFrame, name: str) -> pl.Expr:
    """Date column whether stored as Date, Datetime or ISO string."""
    dt = df.schema[name]
    if dt == pl.Date:
        return pl.col(name)
    if isinstance(dt, pl.Datetime):
        return pl.col(name).dt.date()
    return pl.col(name).cast(pl.Utf8).str.slice(0, 10).str.to_date("%Y-%m-%d", strict=False)


def news_daily(news: pl.DataFrame) -> pl.DataFrame:
    """(ticker, avail_date) → article count + mean sentiment; available the day after publication."""
    d = (news.select("ticker", pub=pl.col("published_utc").dt.date(), sentiment="sentiment")
             .group_by(["ticker", "pub"]).agg(n=pl.len(), sent=pl.col("sentiment").mean())
             .with_columns(avail_date=pl.col("pub") + pl.duration(days=1)).drop("pub"))
    return d.sort(["ticker", "avail_date"])


def _join_news(panel: pl.DataFrame, nd: pl.DataFrame) -> pl.DataFrame:
    T = "ticker"
    if nd.height == 0:
        return panel.with_columns([pl.lit(None, dtype=pl.Float64).alias(c) for c in NEWS_FEATURES])
    # exact-day join then roll over trading days (articles on non-trading days land on the next session)
    nd = nd.rename({"avail_date": "date"})
    cal = panel.select("date").unique().sort("date")
    nd = nd.sort("date").join_asof(cal.with_columns(sess=pl.col("date")), on="date", strategy="forward",
                                   check_sortedness=False).drop("date").rename({"sess": "date"}).drop_nulls("date")
    nd = nd.group_by([T, "date"]).agg(n=pl.col("n").sum(), sent=(pl.col("sent") * pl.col("n")).sum() / pl.col("n").sum())
    j = panel.join(nd, on=[T, "date"], how="left").with_columns(pl.col("n").fill_null(0.0), sent_n=pl.col("sent") * pl.col("n"))
    first_news = nd["date"].min()
    j = j.with_columns(
        news_n_5=pl.col("n").rolling_sum(5, min_samples=1).over(T),
        news_n_21=pl.col("n").rolling_sum(21, min_samples=1).over(T),
        news_sent_5=(pl.col("sent_n").rolling_sum(5, min_samples=1).over(T) / (pl.col("n").rolling_sum(5, min_samples=1).over(T) + 1e-9)),
        news_sent_21=(pl.col("sent_n").rolling_sum(21, min_samples=1).over(T) / (pl.col("n").rolling_sum(21, min_samples=1).over(T) + 1e-9)),
    )
    j = j.with_columns(news_n_z=((pl.col("news_n_21") - pl.col("news_n_21").rolling_mean(252, min_samples=63).over(T))
                                 / (pl.col("news_n_21").rolling_std(252, min_samples=63).over(T) + 1e-9)))
    # before the archive starts the counts are unknown, not zero
    j = j.with_columns([pl.when(pl.col("date") < first_news).then(None).otherwise(pl.col(c)).alias(c) for c in NEWS_FEATURES])
    return j.drop(["n", "sent", "sent_n"])


def _join_short(panel: pl.DataFrame, si: pl.DataFrame | None, sv: pl.DataFrame | None) -> pl.DataFrame:
    T = "ticker"
    out = panel
    if si is not None and si.height:
        s = (si.select(T, sd=_date_col(si, "settlement_date"), si=pl.col("short_interest").cast(pl.Float64),
                       adv=pl.col("avg_daily_volume").cast(pl.Float64), dtc=pl.col("days_to_cover").cast(pl.Float64))
               .sort([T, "sd"])
               .with_columns(si_chg=pl.col("si") / pl.col("si").shift(1).over(T) - 1,
                             avail_date=pl.col("sd") + pl.duration(days=10))
               .unique(subset=[T, "avail_date"], keep="last").sort([T, "avail_date"]))
        out = out.sort([T, "date"]).join_asof(s, left_on="date", right_on="avail_date", by=T, strategy="backward",
                                              tolerance=timedelta(days=45), check_sortedness=False)
        out = out.with_columns(si_days_to_cover=pl.col("dtc"), si_ratio=pl.col("si") / (pl.col("adv") + 1.0),
                               si_chg=pl.col("si_chg")).drop(["sd", "si", "adv", "dtc", "avail_date"])
    else:
        out = out.with_columns(si_days_to_cover=pl.lit(None, dtype=pl.Float64), si_ratio=pl.lit(None, dtype=pl.Float64),
                               si_chg=pl.lit(None, dtype=pl.Float64))
    if sv is not None and sv.height:
        v = (sv.select(T, date=_date_col(sv, "date"),
                       svr=pl.col("short_volume_ratio").cast(pl.Float64))
               .drop_nulls("date").with_columns(date=pl.col("date") + pl.duration(days=1))   # published next morning
               .unique(subset=[T, "date"], keep="last"))
        out = out.join(v, on=[T, "date"], how="left")
        out = out.with_columns(sv_ratio_5=pl.col("svr").rolling_mean(5, min_samples=1).over(T)).drop("svr")
        first = v["date"].min()
        out = out.with_columns(pl.when(pl.col("date") < first).then(None).otherwise(pl.col("sv_ratio_5")).alias("sv_ratio_5"))
    else:
        out = out.with_columns(sv_ratio_5=pl.lit(None, dtype=pl.Float64))
    return out


# ── market context + labels ───────────────────────────────────────────────────

def market_features(panel: pl.DataFrame) -> pl.DataFrame:
    D = "date"
    out = panel.with_columns(
        mkt_ret_21=pl.col("ret_21").mean().over(D), mkt_ret_63=pl.col("ret_63").mean().over(D),
        breadth_200=(pl.col("sma_gap_200") > 0).cast(pl.Float64).mean().over(D),
        dispersion_21=pl.col("ret_21").std().over(D),
        sector_rel_63=pl.col("ret_63") - pl.col("ret_63").mean().over([D, "sector"]),
    )
    mkt = (out.group_by(D).agg(m=pl.col("r1").mean()).sort(D)
              .with_columns(mkt_vol_21=pl.col("m").rolling_std(21, min_samples=10) * _ANN,
                            idx=(1 + pl.col("m").fill_null(0.0)).cum_prod())
              .with_columns(mkt_trend_200=(pl.col("idx") / pl.col("idx").rolling_mean(200, min_samples=100) - 1))
              .drop("m", "idx"))
    return out.join(mkt, on=D, how="left")


def add_labels(panel: pl.DataFrame, horizons: Iterable[int] = HORIZONS) -> pl.DataFrame:
    """``fwd_h`` = raw forward return; ``y_h`` = gaussianised per-date rank of the demeaned forward return.

    The gaussian-rank label is what makes the forecaster a *ranker*: it is
    blind to the market's level, cannot be dominated by lottery-ticket tails,
    and gives every date the same weight in the loss.
    """
    T, D = "ticker", "date"
    ac = pl.col("adj_close")
    out = panel.sort([T, D])
    for h in horizons:
        out = out.with_columns((ac.shift(-h).over(T) / ac - 1).alias(f"fwd_{h}"))
    for h in horizons:
        f = pl.col(f"fwd_{h}")
        n = f.count().over(D)
        rk = f.rank(method="average").over(D)
        out = out.with_columns(((rk - 0.5) / n).alias(f"_u_{h}"))
    for h in horizons:
        u = out[f"_u_{h}"].to_numpy()
        out = out.with_columns(pl.Series(f"y_{h}", _norm_ppf(u), dtype=pl.Float32).fill_nan(None))
    return out.drop([f"_u_{h}" for h in horizons])


def _norm_ppf(u: np.ndarray) -> np.ndarray:
    from scipy.special import ndtri
    with np.errstate(invalid="ignore"):
        return ndtri(np.clip(u, 1e-6, 1 - 1e-6)).astype(np.float32)


# ── orchestration ─────────────────────────────────────────────────────────────

def build_panel(cfg, tickers: Iterable[str] | None = None, start: str | date = "1998-01-01",
                end: date | None = None, horizons: Iterable[int] = HORIZONS,
                min_dollar_vol: float = 1e6, deep_path: Path | None = None) -> pl.DataFrame:
    """Prices → features → PIT joins → labels. Returns the long panel (float32 features)."""
    import time as _t
    t0 = _t.time()
    B = cfg.path("data_bronze") / "massive"
    px = load_prices(cfg, tickers, start=start, end=end, deep_path=deep_path)
    panel = price_features(px)
    panel = panel.join(_sector_map(cfg), on="ticker", how="left").with_columns(pl.col("sector").fill_null("unknown"))
    fin_p, news_p, si_p, sv_p = B / "financials.parquet", B / "news.parquet", B / "short_interest.parquet", B / "short_volume.parquet"
    fund = fundamental_features(pl.read_parquet(fin_p)) if fin_p.exists() else pl.DataFrame()
    panel = _join_fundamentals(panel, fund)
    nd = news_daily(pl.read_parquet(news_p, columns=["ticker", "published_utc", "sentiment"])) if news_p.exists() else pl.DataFrame()
    panel = _join_news(panel, nd)
    si = pl.read_parquet(si_p) if si_p.exists() else None
    sv = pl.read_parquet(sv_p, columns=["ticker", "date", "short_volume_ratio"]) if sv_p.exists() else None
    panel = _join_short(panel, si, sv)
    panel = market_features(panel)
    try:
        from .regime import macro_features_for_panel
        panel = panel.join(macro_features_for_panel(cfg, panel), on="date", how="left")
    except Exception as e:                       # macro is an enrichment, never a hard dependency
        logger.warning(f"panel: macro/regime features unavailable ({str(e)[:100]})")
    panel = add_labels(panel, horizons)
    # tradable-ish rows only (a $1M/day floor keeps the training cross-section honest but broad)
    panel = panel.filter(pl.col("log_dv_21").is_null() | (pl.col("log_dv_21") >= math.log1p(min_dollar_vol)))
    feats = [c for c in FEATURE_COLS if c in panel.columns]
    panel = panel.with_columns([pl.col(c).cast(pl.Float32) for c in feats])
    keep = BASE_COLS + feats + CONTEXT_COLS + [f"fwd_{h}" for h in horizons] + [f"y_{h}" for h in horizons]
    panel = panel.select([c for c in keep if c in panel.columns]).sort(["date", "ticker"])
    logger.info(f"panel: {panel.height:,} rows · {panel['ticker'].n_unique()} tickers · "
                f"{len(feats)} features · {panel['date'].min()} → {panel['date'].max()} · {_t.time() - t0:.1f}s")
    return panel


def panel_path(cfg) -> Path:
    return cfg.path("data_gold") / "alpha_panel.parquet"


def save_panel(cfg, panel: pl.DataFrame) -> Path:
    p = panel_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    panel.write_parquet(tmp, compression="zstd")
    tmp.replace(p)
    return p


def load_panel(cfg) -> pl.DataFrame:
    p = panel_path(cfg)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing — run `ts alpha panel` first")
    return pl.read_parquet(p)
