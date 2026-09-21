"""Extensive, causal backtest of the alpha engine — signal and book, on the research simulator.

Two layers, both strictly out-of-sample:

1. **Signal**: the walk-forward scores (``model.causal_scores``) go into the
   ledger as ``mode="backtest"`` rows and are tallied against realised returns
   exactly like live forecasts. ``ledger.skill_report`` / ``yearly_ic`` then
   give rank-IC, ICIR, hit rate, decile spread and band coverage per horizon,
   overall and per year.
2. **Book**: the calibrator is itself walked forward (refit on matured rows
   only, every ``refit_every`` trading days) to produce the causal composite
   score, which is replayed through :func:`research.wfbacktest.run_walk_forward`
   with the *same* ``portfolio.target_weights`` the live book uses, paying the
   research cost model.  Momentum, the equal-weight universe and SPY run on
   identical dates for comparison.  Sharpe CIs are stationary-bootstrap; the
   deflated Sharpe counts every variant tried.

Outputs land in ``reports/alpha/``: ``backtest.md`` (the report),
``backtest.json`` (numbers), ``daily_<variant>.parquet`` (equity curves).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from ..utils import get_logger
from . import ledger as L
from .panel import HORIZONS
from .portfolio import BookConfig, target_weights

logger = get_logger(__name__)


# ── signal layer ──────────────────────────────────────────────────────────────

def scores_to_ledger_rows(scores: pl.DataFrame, panel: pl.DataFrame, mode: str = "backtest") -> pl.DataFrame:
    px = panel.select("date", "ticker", entry_price=pl.col("adj_close").cast(pl.Float32))
    return (scores.join(px, on=["date", "ticker"], how="left")
                  .with_columns(mode=pl.lit(mode), exp_ret=pl.lit(None, pl.Float32), q10=pl.lit(None, pl.Float32),
                                q90=pl.lit(None, pl.Float32)))


def causal_composite(led: pl.DataFrame, horizons=HORIZONS, refit_every: int = 63, window_days: int = 365 * 3,
                     min_rows: int = 5000) -> tuple[pl.DataFrame, list[dict]]:
    """Walk the calibrator forward over the backtest ledger → ``date, ticker, composite`` + calibration log.

    For each block of ``refit_every`` dates the calibrator sees only rows whose outcome was known
    before the block starts (date + horizon < block start). Blocks with no usable calibration fall
    back to equal horizon weights, which is what a live system would do on day one.
    """
    dates = led.select("date").unique().sort("date")["date"].to_list()
    blocks = list(range(0, len(dates), refit_every))
    parts, log = [], []
    for k, b in enumerate(blocks):
        d0 = dates[b]
        d1 = dates[min(b + refit_every, len(dates)) - 1]
        known = led.filter(pl.col("realized_date").is_not_null() & (pl.col("realized_date") < d0))
        cal = (L.Calibrator.fit(known, horizons, window_days=window_days, until=d0, min_rows=min_rows)
               if known.height >= min_rows else L.Calibrator())
        w = cal.skill_weights(horizons)
        block = led.filter((pl.col("date") >= d0) & (pl.col("date") <= d1)).select("date", "ticker", "horizon", "score")
        parts.append(L.composite_score(block, w))
        log.append({"block_start": str(d0), "weights": {int(h): round(v, 3) for h, v in w.items()},
                    "ic": {int(h): round(c.ic, 4) for h, c in cal.horizons.items()}})
    return pl.concat(parts).sort(["date", "ticker"]), log


# ── book layer ────────────────────────────────────────────────────────────────

class ColumnAlpha:
    """Simulator model that replays a precomputed column (composite / momentum)."""

    def __init__(self, col: str):
        self.col, self.name, self.chosen = col, col, col

    def fit(self, panel, feat_cols, target) -> None:
        return None

    def predict(self, today: pl.DataFrame, feat_cols) -> np.ndarray:
        return today[self.col].fill_null(strategy="zero").to_numpy()


def _weight_fn(book: BookConfig, sector_of: dict[str, str], regime: dict | None = None, stress: dict | None = None):
    """``regime``: date → bool (trend risk-on); ``stress``: date → gross multiplier; None = that overlay off."""
    def fn(scores, cols, cfg):
        sectors = np.array([sector_of.get(t, "unknown") for t in cols["ticker"]]) if "ticker" in cols else None
        d = cols.get("date")
        on = True if regime is None else regime.get(d, True)
        gm = 1.0 if stress is None else stress.get(d, 1.0)
        return target_weights(scores, cols, book, sectors, regime_on=on, gross_mult=gm)
    return fn


def regime_map(panel: pl.DataFrame) -> dict:
    from .portfolio import market_regime_on
    m = panel.group_by("date").agg(pl.col("mkt_trend_200").first()).sort("date")
    return {r["date"]: market_regime_on(r["mkt_trend_200"]) for r in m.iter_rows(named=True)}


def stress_map(cfg, panel: pl.DataFrame) -> dict:
    """date → gross multiplier from the point-in-time stress score (regime.gross_multiplier)."""
    from .regime import build_state, fragility_score, gross_multiplier
    _, zf = build_state(cfg, panel)
    f = fragility_score(zf)
    return {r["date"]: gross_multiplier(r["stress"]) for r in f.iter_rows(named=True)}


def eligible_mask(panel: pl.DataFrame, book: BookConfig) -> pl.Expr:
    """Live-tradable rows: raw close ≥ min_price, 21d dollar volume ≥ min_dollar_volume, vol within cap."""
    import math
    return ((pl.col("close") >= book.min_price)
            & (pl.col("log_dv_21") >= math.log1p(book.min_dollar_volume))
            & (pl.col("vol_63") / 252 ** 0.5 <= book.max_daily_vol)).fill_null(False).alias("elig")


def eligible_ew_returns(panel: pl.DataFrame, dates: list, book: BookConfig | None = None) -> np.ndarray:
    """Daily-rebalanced equal weight of the names that pass the book's gates on the previous session —
    the benchmark the book must beat (same universe, same gates, no signal)."""
    book = book or BookConfig()
    p = (panel.select("date", "ticker", "adj_close", "close", "log_dv_21", "vol_63").sort(["ticker", "date"])
              .with_columns(eligible_mask(panel, book))
              .with_columns(r=(pl.col("adj_close") / pl.col("adj_close").shift(1).over("ticker") - 1),
                            elig_prev=pl.col("elig").shift(1).over("ticker")))
    ew = (p.filter(pl.col("elig_prev") & pl.col("r").is_finite() & (pl.col("r").abs() < 5))
           .group_by("date").agg(ew=pl.col("r").mean()).sort("date"))
    m = dict(ew.iter_rows())
    return np.array([m.get(d, 0.0) or 0.0 for d in dates])


def _max_dd(eq: np.ndarray) -> float:
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1).min())


def _yearly(daily: pl.DataFrame) -> pl.DataFrame:
    return (daily.with_columns(year=pl.col("date").dt.year())
                 .group_by("year").agg(ret=((1 + pl.col("net_ret")).product() - 1), n=pl.len()).sort("year"))


def run_book_backtest(cfg, panel: pl.DataFrame, composite: pl.DataFrame, book: BookConfig | None = None,
                      oos_start: date | None = None, rebalance_days: int = 21, trade_rate: float = 0.35,
                      out_dir: Path | None = None, n_boot: int = 500, extra_variants: dict | None = None) -> dict:
    """Replay the composite through the research simulator against momentum / universe_ew / SPY."""
    from ..research.runner import passive_benchmarks, summarize_result
    from ..research.stats import bootstrap_ci, deflated_sharpe, sharpe
    from ..research.wfbacktest import WalkForwardConfig, build_panel, run_walk_forward
    from .panel import load_prices

    book = book or BookConfig()
    t0 = time.time()
    prices = load_prices(cfg, start=str(panel["date"].min()), end=panel["date"].max())
    pdata = build_panel(prices)
    sector_of = dict(panel.select("ticker", "sector").unique(subset=["ticker"]).iter_rows())
    # Eligibility is decided on the RAW close (what a live order sees), not the split-adjusted price the
    # simulator carries: an adjusted-price floor silently drops every name that later split its way to
    # a large cap — exactly the winners — in the early years. Ineligible rows get a null score.
    elig = eligible_mask(panel, book)
    feats = (panel.select("date", "ticker", "adj_close", "mom_12_1", "close", "log_dv_21", "vol_63").with_columns(elig)
                  .drop("close", "log_dv_21", "vol_63")
                  .join(composite, on=["date", "ticker"], how="left")
                  .with_columns(composite=pl.when(pl.col("elig")).then(pl.col("composite").cast(pl.Float64)),
                                mom_12_1=pl.when(pl.col("elig")).then(pl.col("mom_12_1").cast(pl.Float64))))
    first = oos_start or (composite["date"].min() + timedelta(days=400))
    base = WalkForwardConfig(oos_start=first, rebalance_days=rebalance_days, retrain_days=10**6, min_train_days=250,
                             horizon=21, top_k=book.top_k, max_weight=book.max_weight, min_dollar_volume=0.0,
                             min_price=0.0, partial_trade_rate=trade_rate, respect_target_gross=True)
    regime = regime_map(panel)
    # the simulator hands the weight function split-ADJUSTED prices, so the price/volume gates must not be
    # re-applied there (eligibility was decided above on raw closes); keep the vol cap
    sim_book = replace(book, min_price=0.0, min_dollar_volume=0.0)
    no_overlay = replace(sim_book, regime_scale=1.0, vol_target=None)
    variants = {
        "alpha|book": (ColumnAlpha("composite"), replace(base), _weight_fn(sim_book, sector_of, regime)),
        "alpha|no_overlay": (ColumnAlpha("composite"), replace(base), _weight_fn(no_overlay, sector_of)),
        "alpha|full_rebalance": (ColumnAlpha("composite"), replace(base, partial_trade_rate=1.0), _weight_fn(sim_book, sector_of, regime)),
        "momentum|book": (ColumnAlpha("mom_12_1"), replace(base), _weight_fn(sim_book, sector_of, regime)),
    }
    if extra_variants:
        variants.update(extra_variants)
    results, rows = {}, []
    for name, (model, wcfg, wfn) in variants.items():
        col = model.col
        res = run_walk_forward(pdata, feats.filter(pl.col(col).is_not_null()), [col], lambda m=model: m, wcfg,
                               weight_fn=wfn, label=name, progress=False)
        results[name] = res
        row = summarize_result(res)
        row["yearly"] = {int(r["year"]): round(float(r["ret"]), 4) for r in _yearly(res.daily).iter_rows(named=True)}
        rows.append(row)
        logger.info(f"alpha backtest {name}: CAGR {row['CAGR']:+.2%} Sharpe {row['Sharpe']} MaxDD {row['MaxDD']:.1%} "
                    f"turnover {row['ann_turnover']}x")
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            res.daily.write_parquet(out_dir / f"daily_{name.replace('|', '_')}.parquet")
    dates = next(iter(results.values())).daily["date"].to_list()
    bench = passive_benchmarks(pdata, dates)
    bench["eligible_ew"] = eligible_ew_returns(panel, dates, book)
    for bname, br in bench.items():
        eq = np.cumprod(1 + br)
        yrs = len(br) / 252
        d = pl.DataFrame({"date": dates[:len(br)], "net_ret": br})
        rows.append({"variant": bname, "years": round(yrs, 2), "CAGR": round(float(eq[-1] ** (1 / yrs) - 1), 4),
                     "Sharpe": round(sharpe(br), 3), "MaxDD": round(_max_dd(eq), 4), "ann_turnover": 0.0,
                     "cost_drag_annual": 0.0, "AnnVol": round(float(np.std(br, ddof=1) * 252 ** 0.5), 4),
                     "yearly": {int(r["year"]): round(float(r["ret"]), 4) for r in _yearly(d).iter_rows(named=True)}})
        results[bname] = br
    # honesty: bootstrap CI on the headline book and the deflated Sharpe over everything tried
    strat_rets = {k: (v.returns() if hasattr(v, "returns") else v) for k, v in results.items()}
    trials = np.array([sharpe(r, annualise=False) for r in strat_rets.values()])
    stats = {}
    for name, r in strat_rets.items():
        ci = bootstrap_ci(r, n_boot=n_boot)
        stats[name] = {"sharpe_ci": [round(ci["lo"], 3), round(ci["hi"], 3)], "p_sharpe_gt_0": round(ci["p_gt_0"], 3),
                       "dsr": round(deflated_sharpe(r, trials)["dsr"], 3)}
    # excess over the equal-weight ELIGIBLE universe (same gates, no signal — the honest benchmark)
    if "eligible_ew" in strat_rets:
        ew = strat_rets["eligible_ew"]
        for name, r in strat_rets.items():
            n = min(len(r), len(ew))
            stats[name]["excess_vs_eligible_ew"] = round(float(np.mean(r[:n] - ew[:n]) * 252), 4)
    out = {"generated": str(date.today()), "book": asdict(book), "rebalance_days": rebalance_days,
           "trade_rate": trade_rate, "table": rows, "stats": stats, "runtime_s": round(time.time() - t0, 1)}
    if out_dir:
        (out_dir / "backtest.json").write_text(json.dumps(out, indent=1, default=str))
    return out


# ── report ────────────────────────────────────────────────────────────────────

def _fmt(v, pct=False, nd=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v * 100:+.{nd}f}%" if pct else f"{v:.{nd}f}"


def write_report(out_dir: Path, signal: pl.DataFrame, yearly: pl.DataFrame, book: dict | None,
                 calib_log: list[dict] | None = None, notes: list[str] | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# Alpha engine v2 — causal backtest ({date.today()})", ""]
    lines += ["Every number is out-of-sample: models refit only on labels that had matured, the calibrator "
              "walked forward on matured ledger rows, the book traded a day after the decision paying "
              "per-name spread + impact.", ""]
    lines += ["## Signal (rank-IC of the raw score vs realised return, per horizon)", "",
              "| horizon | dates | IC | ICIR | t | hit | decile spread | 80% band cov. | calib slope | MAE |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in signal.iter_rows(named=True):
        lines.append(f"| {r['horizon']}d | {r.get('n_dates')} | {_fmt(r.get('ic_mean'), nd=4)} | {_fmt(r.get('icir'))} | "
                     f"{_fmt(r.get('t_stat'), nd=1)} | {_fmt(r.get('hit_rate'), pct=True, nd=1)} | "
                     f"{_fmt(r.get('decile_spread'), pct=True)} | {_fmt(r.get('coverage_80'), pct=True, nd=1)} | "
                     f"{_fmt(r.get('calib_slope'))} | {_fmt(r.get('mae'), pct=True)} |")
    lines += ["", "### IC by year", ""]
    if yearly.height:
        hs = sorted(yearly["horizon"].unique().to_list())
        lines.append("| year | " + " | ".join(f"{h}d IC" for h in hs) + " |")
        lines.append("|---:|" + "---:|" * len(hs))
        for (y,), g in yearly.group_by(["year"], maintain_order=True):
            m = {int(r["horizon"]): r["ic"] for r in g.iter_rows(named=True)}
            lines.append(f"| {y} | " + " | ".join(_fmt(m.get(h), nd=3) for h in hs) + " |")
    if book:
        lines += ["", "## Book (research simulator, after costs)", "",
                  "| variant | years | CAGR | Sharpe | Sharpe 95% CI | DSR | MaxDD | ann. vol | turnover/yr | cost drag | excess vs eligible EW |",
                  "|:--|---:|---:|---:|:--:|---:|---:|---:|---:|---:|---:|"]
        st = book.get("stats", {})
        for r in sorted(book["table"], key=lambda r: -r["Sharpe"]):
            s = st.get(r["variant"], {})
            ci = s.get("sharpe_ci")
            lines.append(f"| {r['variant']} | {r['years']} | {_fmt(r['CAGR'], pct=True)} | {r['Sharpe']} | "
                         f"{'[%+.2f, %+.2f]' % tuple(ci) if ci else '—'} | {s.get('dsr', '—')} | {_fmt(r['MaxDD'], pct=True, nd=1)} | "
                         f"{_fmt(r.get('AnnVol'), pct=True, nd=1)} | {r['ann_turnover']} | {_fmt(r.get('cost_drag_annual'), pct=True)} | "
                         f"{_fmt(s.get('excess_vs_eligible_ew'), pct=True)} |")
        years = sorted({y for r in book["table"] for y in r.get("yearly", {})})
        if years:
            names = [r["variant"] for r in book["table"]]
            lines += ["", "### Calendar-year returns", "", "| year | " + " | ".join(names) + " |", "|---:|" + "---:|" * len(names)]
            for y in years:
                lines.append(f"| {y} | " + " | ".join(_fmt(r.get("yearly", {}).get(y), pct=True, nd=1) for r in book["table"]) + " |")
    if calib_log:
        lines += ["", "## Continuous learning — horizon weights the calibrator chose over time", ""]
        step = max(1, len(calib_log) // 12)
        for e in calib_log[::step]:
            lines.append(f"- {e['block_start']}: weights {e['weights']} · trailing IC {e['ic']}")
    if notes:
        lines += ["", "## Read this before believing any of it", ""] + [f"- {n}" for n in notes]
    p = out_dir / "backtest.md"
    p.write_text("\n".join(lines) + "\n")
    return p


DEFAULT_NOTES = [
    "Universe = today's top-1000 by liquidity with yfinance history before 2024-09: names that died before then are "
    "missing, so every CAGR is an upper bound (the repo's `ts bias-check` measures the size of that bias; the "
    "equal-weight universe row already contains it — beat *that*, not SPY).",
    "Costs are modelled (Abdi–Ranaldo spread + square-root impact, participation-capped), not quoted.",
    "Fundamentals start 2009, news 2016, short interest 2017: earlier years are price/volume-only, so the model's "
    "recent skill is not the same model as its early skill.",
    "Long-only, no leverage, daily bars. Nothing here places trades.",
]
