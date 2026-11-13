"""Challenger engine: point-in-time training and the index-minus book, run beside production (paper only).

Why (docs/RESEARCH_2026-11.md §3 and §10). On the honest point-in-time test — every stock that traded, failures
included — the production recipe's top-20 book made about −1%/yr while SPY made +14%. Two changes held up there:

* **training**: use the point-in-time rows where we have them (the whole-market history kept since 2024-09) and
  train only on each day's 500 most liquid names, instead of today's survivors only;
* **construction**: hold the 300 most liquid names at market-cap weight and drop the worst 30% by score, instead
  of betting on 20 names. Avoiding losers was steadier than picking winners, in our data and in a century of
  published-signal portfolios.

The book is held as four sub-books rebalanced a week apart (Newfound's fix for rebalance-timing luck), so the
result does not depend on which day of the month we trade. Designs from trading-firm competitions were tried on
top of this (RESEARCH_2026-11 §10): refitting every 5 or 21 sessions instead of 63, a 2-year sample-weight
half-life, and Numerai-style neutralisation of the scores. None passed the rule set in advance, so the model is
refit every 63 sessions with the production weighting, exactly as tested.

A second model, ``fn`` (factor-neutral, RESEARCH_2026-11 §13), is trained the same way on a different target: the
part of each stock's forward return that beta, volatility, size, momentum, last month's return and sector do not
explain. On the honest test it is the first score with a positive IC in both years (the base model's fell to zero
in the second), and through the production top-20 rules inside the top-300 it gave the steadiest book. It feeds
a third paper book, ``top20_fn``.

Nothing here touches the production model, its ledger or the picks card. The engine keeps its own panel, models,
scores and weekly target snapshots under ``data/gold/challenger`` and feeds three paper books: ``index_minus``,
its control ``top300_cap`` (the same 300 names with nothing dropped; the control, not SPY, is the comparison)
and ``top20_fn``.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from ..utils import get_logger
from . import model as M
from . import panel as P
from .portfolio import BookConfig, market_regime_on, target_weights

logger = get_logger(__name__)

H = (21, 63)
W = {21: 0.3, 63: 0.7}
PIT_TOP = 1000            # the point-in-time universe: each day's top names by trailing 63-day median $ volume
TRAIN_TOP = 500           # train on that day's 500 most liquid names
UNIVERSE_TOP = 300        # the book's universe
DROP = 0.30               # share of the universe dropped (lowest scores)
TRANCHES = 4              # sub-books …
TRANCHE_GAP = 5           # … rebalanced this many sessions apart
RETRAIN_EVERY = 63        # sessions between refits (5 and 21 were tested and were no better)
BOOKS = ("index_minus", "top300_cap", "top20_fn")
MODELS = {"base": "y", "fn": "yn"}          # model → label column (y = rank of the return, yn = factor-neutral)
RISK = ("beta_63", "vol_63", "log_mcap", "mom_12_1", "ret_21")


def root(cfg) -> Path:
    d = cfg.path("data_gold") / "challenger"
    d.mkdir(parents=True, exist_ok=True)
    return d


def panel_path(cfg) -> Path:
    return root(cfg) / "pit_panel.parquet"


# ── point-in-time panel ───────────────────────────────────────────────────────

def pit_prices(cfg, top: int = PIT_TOP) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Whole-market bars for every common stock / ADR that was ever in a day's top ``top`` by trailing 63-day
    median dollar volume (close ≥ $5), and that membership as ``date, ticker``. Failed names stay in."""
    B = cfg.path("data_bronze") / "massive"
    tk = pl.read_parquet(B / "tickers.parquet", columns=["ticker", "type"])
    ok = tk.filter(pl.col("type").is_in(["CS", "ADRC"]))["ticker"].to_list()
    px = (pl.scan_parquet(B / "ohlcv_all.parquet")
            .filter(~pl.col("otc"), pl.col("ticker").is_in(ok), pl.col("close") > 0, pl.col("adj_close") > 0)
            .select("date", "ticker", "open", "high", "low", "close", "adj_close", "volume").collect())
    uni = (px.sort(["ticker", "date"]).with_columns(dv=pl.col("close") * pl.col("volume"))
             .with_columns(med63=pl.col("dv").rolling_median(63, min_samples=40).over("ticker"))
             .filter(pl.col("close") >= 5, pl.col("med63").is_not_null())
             .with_columns(rk=pl.col("med63").rank(descending=True, method="ordinal").over("date"))
             .filter(pl.col("rk") <= top).select("date", "ticker"))
    names = uni["ticker"].unique().to_list()
    return px.filter(pl.col("ticker").is_in(names)), uni


def build_panel(cfg) -> pl.DataFrame:
    px, uni = pit_prices(cfg)
    pw = P.build_panel(cfg, prices=px, min_dollar_vol=0).join(uni, on=["date", "ticker"], how="semi")
    pw.write_parquet(panel_path(cfg), compression="zstd")
    return pw


def load_panel(cfg) -> pl.DataFrame:
    p = panel_path(cfg)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing — run `ts alpha challenger`")
    return pl.read_parquet(p)


# ── model ─────────────────────────────────────────────────────────────────────

def spec(label: str = "y") -> M.TrainSpec:
    return M.TrainSpec(horizons=H, train_top=TRAIN_TOP, label=label)


def models_dir(cfg, name: str = "base") -> Path:
    return root(cfg) / "models" if name == "base" else root(cfg) / "models" / name


def neutral_labels(df: pl.DataFrame) -> pl.DataFrame:
    """Add ``yn_h``: the gaussian rank of each day's forward return after removing what beta, 63-day volatility,
    size, 12-1 momentum, the last month's return and sector explain (a per-day least-squares fit)."""
    from scipy.stats import rankdata

    def z(c: str) -> pl.Expr:
        x = pl.when(pl.col(c).is_finite()).then(pl.col(c))
        return ((x - x.mean().over("date")) / (x.std().over("date") + 1e-9)).clip(-3, 3).fill_null(0.0)

    ex = df.select("date", "ticker", "sector", *[z(c).alias("x_" + c) for c in RISK], *[f"fwd_{h}" for h in H])
    parts = []
    for g in ex.partition_by("date"):
        sec = g["sector"].to_numpy()
        X = np.column_stack([g.select(["x_" + c for c in RISK]).to_numpy().astype(np.float64)]
                            + [(sec == u).astype(np.float64) for u in np.unique(sec)])
        cols = {"date": g["date"], "ticker": g["ticker"]}
        for h in H:
            y = g[f"fwd_{h}"].to_numpy().astype(np.float64)
            ok, r = np.isfinite(y), np.full(len(y), np.nan)
            if ok.sum() >= 50:
                res = y[ok] - X[ok] @ np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
                r[ok] = P._norm_ppf((rankdata(res) - 0.5) / ok.sum())
            cols[f"yn_{h}"] = r
        parts.append(pl.DataFrame(cols))
    lab = pl.concat(parts).with_columns([pl.col(f"yn_{h}").cast(pl.Float32).fill_nan(None) for h in H])
    return df.drop([c for c in df.columns if c.startswith("yn_")]).join(lab, on=["date", "ticker"], how="left")


def training_set(hf: pl.DataFrame, wf: pl.DataFrame, rcols: list[str], h: int, ci: int, wi: int, pit_start: date,
                 sp: M.TrainSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Survivor rows before the point-in-time history starts, point-in-time rows after — each purged and
    embargoed against its own calendar (``ci`` / ``wi`` = the as-of day index in each frame)."""
    tr = M.training_rows(hf, h, ci, sp).filter(pl.col("date") < pit_start)
    X, y = M._xy(tr, rcols, h, sp.label)
    w = M.sample_weights(tr, ci, sp)
    tw = M.training_rows(wf, h, wi, sp)
    if tw.height:
        X2, y2 = M._xy(tw, rcols, h, sp.label)
        X, y = np.vstack([X, X2]), np.concatenate([y, y2])
        w = np.concatenate([w, M.sample_weights(tw, wi, sp)])
    return X, y, w, tw.height


def train(cfg, pw: pl.DataFrame | None = None) -> dict:
    pw = neutral_labels(pw if pw is not None else load_panel(cfg))
    hist = neutral_labels(P.load_panel(cfg))
    hf, rcols = M.prepare(hist, spec().features)
    wf, _ = M.prepare(pw, spec().features)
    last = pw["date"].max()
    ci = int(hf.filter(pl.col("date") <= last)["didx"].max())
    wi = int(wf["didx"].max())
    out = {"horizons": list(H), "models": list(MODELS), "trained_through": str(last), "features": len(rcols), "rows": {}, "pit_rows": {}}
    for name, label in MODELS.items():
        sp = spec(label)
        for h in H:
            X, y, w, n_pit = training_set(hf, wf, rcols, h, ci, wi, pw["date"].min(), sp)
            m = M.AlphaGBM(sp, h, rcols).fit(X, y, w)
            m.trained_through, m.n_rows = str(last), int(len(y))
            m.save(models_dir(cfg, name) / f"{h}d")
            if name == "base":
                out["rows"][h], out["pit_rows"][h] = int(len(y)), int(n_pit)
    (root(cfg) / "models" / "manifest.json").write_text(json.dumps(out, indent=1))
    return out


def manifest(cfg) -> dict | None:
    p = root(cfg) / "models" / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else None


def sessions_since(cal: list[date], d: date | str | None) -> int:
    """Sessions in ``cal`` strictly after ``d`` (a large number when ``d`` is None)."""
    if d is None:
        return 10 ** 6
    d = date.fromisoformat(str(d)[:10])
    return sum(1 for x in cal if x > d)


def class_shares(cfg) -> pl.DataFrame:
    """``ticker, cls_shares``: shares of each listed share class, for index weights.

    The panel's market cap is company-wide and comes from SEC filings, so it double-counts companies with two
    listings (GOOG and GOOGL each carried all of Alphabet) and is missing for foreign filers (ASML, TSM, AZN …).
    Massive's ticker details give shares per class; where they don't, the company's weighted shares are split
    equally across its listed classes. These are current counts, used for weights only — never as a signal."""
    p = cfg.path("data_bronze") / "massive" / "details.parquet"
    if not p.exists():
        return pl.DataFrame(schema={"ticker": pl.Utf8, "cls_shares": pl.Float64})
    d = pl.read_parquet(p, columns=["ticker", "cik", "share_class_shares_outstanding", "weighted_shares_outstanding"])
    n_cls = pl.when(pl.col("cik").is_not_null()).then(pl.len().over("cik")).otherwise(1)
    return d.select("ticker", cls_shares=pl.coalesce(pl.col("share_class_shares_outstanding").cast(pl.Float64),
                                                     pl.col("weighted_shares_outstanding").cast(pl.Float64) / n_cls)).unique(subset=["ticker"])


def with_log_cap(cfg, frame: pl.DataFrame) -> pl.DataFrame:
    """Add ``log_cap`` (the weight proxy) to rows that have ``ticker, close, log_mcap``: per-class shares × that
    day's close, else the panel's filing-based market cap, else null (``fresh_targets`` then estimates it)."""
    f = frame.join(class_shares(cfg), on="ticker", how="left")
    cap = pl.when(pl.col("cls_shares") > 0).then(pl.col("cls_shares") * pl.col("close"))
    pan = pl.when(pl.col("log_mcap").is_finite()).then(pl.col("log_mcap").cast(pl.Float64))
    return f.with_columns(log_cap=pl.coalesce(cap.log(), pan)).drop("cls_shares")


def score(cfg, pw: pl.DataFrame, days: int = 1) -> pl.DataFrame:
    """Score the last ``days`` panel dates and upsert them into ``scores.parquet``."""
    man = manifest(cfg)
    if man is None:
        raise FileNotFoundError("challenger models missing — run `ts alpha challenger --retrain`")
    models = {name: {h: M.AlphaGBM.load(models_dir(cfg, name) / f"{h}d") for h in H} for name in MODELS}
    wf, rcols = M.prepare(pw, models["base"][H[0]].spec.features)
    dates = sorted(wf["date"].unique().to_list())[-days:]
    sub = wf.filter(pl.col("date").is_in(dates))
    X = sub.select(rcols).to_numpy().astype(np.float32)
    col = lambda name, h: f"s{h}" if name == "base" else f"s{h}_{name}"
    out = sub.select("date", "ticker", "sector", "close", "log_dv_21", "log_mcap", "vol_63", "mkt_trend_200").with_columns(
        [pl.Series(col(name, h), np.asarray(models[name][h].predict(X), dtype=np.float64)) for name in MODELS for h in H])
    z = lambda c: (pl.col(c) - pl.col(c).mean().over("date")) / (pl.col(c).std().over("date") + 1e-9)
    out = with_log_cap(cfg, out.with_columns(comp=sum(W[h] * z(col("base", h)) for h in H), comp_fn=sum(W[h] * z(col("fn", h)) for h in H),
                                             model=pl.lit(man["trained_through"])))
    p = root(cfg) / "scores.parquet"
    if p.exists():
        out = pl.concat([pl.read_parquet(p).filter(~pl.col("date").is_in(dates)), out], how="diagonal_relaxed")
    out.sort(["date", "ticker"]).write_parquet(p, compression="zstd")
    return out.filter(pl.col("date") == dates[-1])


# ── book ──────────────────────────────────────────────────────────────────────

def fresh_targets(xs: pl.DataFrame, top: int = UNIVERSE_TOP, drop: float = DROP) -> dict[str, dict[str, float]]:
    """One day's cross-section (``ticker, close, log_dv_21, log_cap, comp``) → target weights for both books.

    Universe: the ``top`` most liquid names that day at ≥ $5. Weights: market cap of the share class (see
    ``with_log_cap``; a missing cap is estimated from dollar volume through the day's median ratio).
    ``index_minus`` drops the lowest ``drop`` share by score — a missing score counts as the median, so a name
    is never dropped for lack of data — and re-weights the rest by cap. ``top300_cap`` keeps everything."""
    cap = pl.when(pl.col("log_cap").is_finite()).then(pl.col("log_cap"))
    u = (xs.with_columns(lr=pl.col("log_dv_21").rank(descending=True)).filter(pl.col("lr") <= top, pl.col("close") >= 5)
           .with_columns(mc=pl.coalesce(cap, pl.col("log_dv_21") + (cap - pl.col("log_dv_21")).median()),
                         comp=pl.col("comp").fill_null(pl.col("comp").median())))
    u = u.filter(pl.col("mc").is_finite()).sort(["comp", "ticker"])
    n = u.height
    if n == 0:
        return {b: {} for b in BOOKS}

    def cap_w(g: pl.DataFrame) -> dict[str, float]:
        v = np.exp(g["mc"].to_numpy().astype(np.float64))
        return dict(zip(g["ticker"].to_list(), (v / v.sum()).tolist()))

    return {"index_minus": cap_w(u.tail(n - int(n * drop))), "top300_cap": cap_w(u), "top20_fn": top20_targets(xs, top)}


def top20_targets(xs: pl.DataFrame, top: int = UNIVERSE_TOP) -> dict[str, float]:
    """The production top-20 rules (risk-adjusted rank, 5 per sector, ½ equal + ½ inverse-vol, 8% cap, vol brake,
    half gross below the 200-day trend) on the factor-neutral score, inside the ``top`` most liquid names. Empty
    when the cross-section carries no ``comp_fn``."""
    need = {"comp_fn", "vol_63", "sector"}
    if not need <= set(xs.columns):
        return {}
    u = (xs.with_columns(lr=pl.col("log_dv_21").rank(descending=True), dvol=pl.col("vol_63") / 252 ** 0.5)
           .filter(pl.col("lr") <= top, pl.col("close") >= 5, pl.col("dvol") <= 0.07).drop_nulls(["comp_fn", "dvol"]).sort("ticker"))
    if u.height == 0:
        return {}
    trend = xs["mkt_trend_200"][0] if "mkt_trend_200" in xs.columns else None
    bk = BookConfig(min_price=0.0, min_dollar_volume=0.0)          # the gates are the filter above, as in the backtest
    w = target_weights(u["comp_fn"].to_numpy(), {"dvol": u["dvol"].to_numpy()}, bk, sectors=u["sector"].to_numpy(),
                       regime_on=market_regime_on(trend))
    return {t: float(x) for t, x in zip(u["ticker"].to_list(), w) if x > 0}


def targets_path(cfg) -> Path:
    return root(cfg) / "targets.parquet"


def load_snapshots(cfg) -> pl.DataFrame:
    p = targets_path(cfg)
    if p.exists():
        return pl.read_parquet(p)
    return pl.DataFrame(schema={"date": pl.Date, "book": pl.Utf8, "ticker": pl.Utf8, "w": pl.Float64})


def snapshot(cfg, xs: pl.DataFrame, cal: list[date], force: bool = False) -> bool:
    """Record today's fresh targets as one tranche if ``TRANCHE_GAP`` sessions have passed since the last one."""
    snaps = load_snapshots(cfg)
    d = xs["date"].max()
    last = snaps["date"].max() if snaps.height else None
    if not force and sessions_since(cal, last) < TRANCHE_GAP:
        return False
    rows = [{"date": d, "book": b, "ticker": t, "w": w} for b, tw in fresh_targets(xs).items() for t, w in tw.items()]
    if not rows:
        return False
    new = pl.DataFrame(rows, schema=snaps.schema)
    pl.concat([snaps.filter(pl.col("date") != d), new]).sort(["date", "book", "ticker"]).write_parquet(targets_path(cfg))
    return True


def blend(snaps: pl.DataFrame, book: str, k: int = TRANCHES) -> dict[str, float]:
    """The held target: the mean of the last ``k`` tranche snapshots (fewer while the book is young)."""
    s = snaps.filter(pl.col("book") == book)
    dates = sorted(s["date"].unique().to_list())[-k:]
    if not dates:
        return {}
    g = s.filter(pl.col("date").is_in(dates)).group_by("ticker").agg(w=pl.col("w").sum() / len(dates))
    return dict(zip(g["ticker"].to_list(), g["w"].to_list()))


def book_weights(cfg, book: str) -> dict[str, float]:
    return blend(load_snapshots(cfg), book)


def last_snapshot(cfg) -> date | None:
    s = load_snapshots(cfg)
    return s["date"].max() if s.height else None


# ── daily entry point and reporting ───────────────────────────────────────────

def update(cfg, retrain: bool = False, days: int = 1, force_snapshot: bool = False) -> dict:
    """Refresh the panel, refit when due, score the latest date and record a tranche when due."""
    pw = build_panel(cfg)
    cal = sorted(pw["date"].unique().to_list())
    man = manifest(cfg)
    trained = False
    if retrain or man is None or set(MODELS) - set(man.get("models", ["base"])) or sessions_since(cal, man["trained_through"]) >= RETRAIN_EVERY:
        man, trained = train(cfg, pw), True
    xs = score(cfg, pw, days=days)
    snapped = snapshot(cfg, xs, cal, force=force_snapshot)
    return {"as_of": str(cal[-1]), "names": xs.height, "trained": trained, "trained_through": man["trained_through"],
            "pit_rows": man.get("pit_rows", {}), "snapshot": snapped, "last_snapshot": str(last_snapshot(cfg))}


def live_ic(cfg, pw: pl.DataFrame | None = None, h: int = 21) -> dict:
    """Rank IC of the recorded composite against the ``h``-day returns that have matured since."""
    p = root(cfg) / "scores.parquet"
    if not p.exists():
        return {"dates": 0}
    pw = pw if pw is not None else load_panel(cfg)
    j = (pl.read_parquet(p).select("date", "ticker", "comp")
           .join(pw.select("date", "ticker", f"fwd_{h}"), on=["date", "ticker"]).drop_nulls([f"fwd_{h}", "comp"]))
    g = j.group_by("date").agg(ic=pl.corr("comp", f"fwd_{h}", method="spearman"), n=pl.len()).filter(pl.col("n") >= 30).sort("date")
    if g.height == 0:
        return {"dates": 0}
    x = g["ic"].to_numpy()
    return {"dates": g.height, "ic": float(np.nanmean(x)), "independent": max(1, g.height // h),
            "first": str(g["date"][0]), "last": str(g["date"][-1])}


def avoid_list(cfg, n: int = 12) -> list[tuple[str, float]]:
    """The largest names (by control weight) that the latest tranche dropped: ``(ticker, control weight)``."""
    s = load_snapshots(cfg)
    if s.height == 0:
        return []
    d = s["date"].max()
    s = s.filter(pl.col("date") == d)
    kept = set(s.filter(pl.col("book") == "index_minus")["ticker"].to_list())
    c = s.filter((pl.col("book") == "top300_cap") & ~pl.col("ticker").is_in(list(kept))).sort("w", descending=True)
    return list(zip(c["ticker"].to_list()[:n], c["w"].to_list()[:n]))


def honest_path(cfg) -> Path:
    return cfg.path("reports") / "alpha" / "honest_latest.json"


def honest(cfg) -> dict | None:
    p = honest_path(cfg)
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except ValueError:
        return None
