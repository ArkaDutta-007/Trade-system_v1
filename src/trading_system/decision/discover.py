"""Moonshot discovery — under-the-radar names with asymmetric upside.

"Find the next SNDK" decomposed into things free data can actually measure:

  1. **Structural freshness** (`ingestion/edgar_discovery`): spinoffs,
     new listings and IPOs from EDGAR — names the market hasn't finished
     pricing because they barely have a price history.
  2. **Under-covered sleepers**: the quiet end of the configured universe —
     bottom-decile dollar volume / news coverage, but with attention
     *igniting* (Wikipedia pageview momentum, news-tone momentum, insider
     filing activity already computed in the gold panel).
  3. **Asymmetry**: the calibrated (or MC-fan) 12m band's upside-to-downside —
     a moonshot is only rational when the band is genuinely lopsided.

Every component of the score is reported (no black-box "AI pick"), every
candidate carries honest caveats (short history, no model coverage), and the
sleeve is ledgered under ``source="moonshot"`` so `ts ledger` eventually
answers the only question that matters: *do your moonshots pay?*

This is a speculative sleeve, capped in ``ts invest --moonshot`` — never the
core allocation.
"""
from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import polars as pl

from ..config import Config, get_config
from ..utils import get_logger

logger = get_logger(__name__)

_MIN_PRICE = 1.0          # no sub-$1 zombies
_MIN_DOLLAR_VOL = 2e6     # ≥ $2M/day median traded value — exit must exist


def _pct_rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(x))
    return order / max(len(x) - 1, 1)


def asymmetry_from_returns(daily_logrets: np.ndarray, horizon: int = 252,
                           n_paths: int = 2000, seed: int = 7) -> dict | None:
    """Bootstrap the name's own daily **log**-returns to a 12m terminal
    distribution and read upside(95th)/|downside(5th)| — the lopsidedness of
    its future. De-meaned, so asymmetry comes from the return *shape* (skew,
    fat right tail), not trend extrapolation."""
    r = np.asarray(daily_logrets, dtype=np.float64)
    r = r[np.isfinite(r)]
    if r.size < 40:
        return None
    rng = np.random.default_rng(seed)
    base = r - r.mean()
    shocks = rng.choice(base, size=(n_paths, horizon), replace=True)
    term = np.exp(shocks.sum(axis=1)) - 1.0
    lo, med, hi = (float(np.quantile(term, q)) for q in (0.05, 0.5, 0.95))
    if not all(np.isfinite(v) for v in (lo, med, hi)):
        return None
    dn = max(0.05, -lo)
    return {"lo": round(lo, 4), "median": round(med, 4), "hi": round(hi, 4),
            "asym": round(max(hi, 0.0) / dn, 3)}


def _fetch_adhoc_history(tickers: list[str], period: str = "2y") -> dict[str, pl.DataFrame]:
    """Daily closes for names outside bronze (structurally fresh listings)."""
    out: dict[str, pl.DataFrame] = {}
    if not tickers:
        return out
    try:
        import yfinance as yf
    except ImportError:
        return out
    for tk in tickers:
        try:
            h = yf.Ticker(tk).history(period=period, auto_adjust=True)
            if h is None or h.empty:
                continue
            df = pl.DataFrame({
                "date": [d.date() for d in h.index.to_pydatetime()],
                "adj_close": h["Close"].to_numpy().astype(float),
                "volume": h["Volume"].to_numpy().astype(float),
            })
            out[tk] = df
        except Exception as e:
            logger.debug(f"adhoc history {tk} failed: {e}")
    return out


def _sleepers_from_panel(features: pl.DataFrame, exclude: set[str],
                         top_k: int = 25) -> list[dict]:
    """Quiet-but-igniting names already inside the gold panel."""
    last = features.filter(pl.col("date") == features["date"].max())
    need = [c for c in ("ticker", "adj_close", "avg_dollar_volume_20") if c in last.columns]
    if len(need) < 3 or last.is_empty():
        return []
    cols = {c: (last[c].to_numpy() if c in last.columns else None)
            for c in ("avg_dollar_volume_20", "wiki_attention_mom", "wiki_attention_z",
                      "news_tone_mom", "news_buzz", "sec_form4_90d", "mom_60d",
                      "adj_close")}
    tickers = last["ticker"].to_list()
    dv = np.nan_to_num(cols["avg_dollar_volume_20"].astype(float), nan=np.inf)
    quiet = _pct_rank(-dv)  # high = low dollar volume (under-covered)

    def _z(name):
        v = cols.get(name)
        if v is None:
            return np.zeros(len(tickers))
        v = np.nan_to_num(np.asarray(v, dtype=float), nan=0.0)
        s = v.std()
        return (v - v.mean()) / s if s > 1e-12 else np.zeros(len(tickers))

    ignition = 0.4 * _z("wiki_attention_mom") + 0.3 * _z("news_tone_mom") \
        + 0.2 * _z("sec_form4_90d") + 0.1 * _z("news_buzz")
    score = quiet * 0.5 + np.tanh(ignition) * 0.5
    out = []
    for i in np.argsort(score)[::-1]:
        tk = tickers[i]
        px = float(cols["adj_close"][i] or 0)
        dvi = float(dv[i])
        if tk in exclude or px < _MIN_PRICE or not np.isfinite(dvi) or dvi < _MIN_DOLLAR_VOL:
            continue
        out.append({
            "ticker": tk, "category": "sleeper",
            "quiet_pct": round(float(quiet[i]), 3),
            "ignition_z": round(float(ignition[i]), 3),
            "last_price": px,
        })
        if len(out) >= top_k:
            break
    return out


def build_discovery(
    cfg: Config | None = None,
    top_n: int = 12,
    lookback_days: int = 120,
    include_fresh: bool = True,
    record: bool = True,
) -> dict[str, Any]:
    """Rank moonshot candidates across the fresh + sleeper pools."""
    cfg = cfg or get_config()
    from .analyze import _load_or_build_features
    ohlcv, features = _load_or_build_features(cfg)
    as_of = features["date"].max()
    universe = set(cfg["universe"]["tickers"])
    held_or_known = universe

    # pool 1: EDGAR structural freshness -------------------------------------
    fresh: list[dict] = []
    if include_fresh:
        try:
            from ..ingestion.edgar_discovery import recent_structural_filings
            fresh = recent_structural_filings(
                lookback_days=lookback_days,
                cache_dir=cfg.path("data_silver") / "edgar_discovery",
            )
        except Exception as e:
            logger.warning(f"EDGAR discovery unavailable: {e}")

    # pool 2: in-panel sleepers ----------------------------------------------
    sleepers = _sleepers_from_panel(features, exclude=set())

    # price/asymmetry for every candidate ------------------------------------
    fresh_tickers = [f["ticker"] for f in fresh if f["ticker"] not in held_or_known]
    adhoc = _fetch_adhoc_history(fresh_tickers[:30])

    candidates: list[dict] = []
    seen: set[str] = set()
    for f in fresh:
        tk = f["ticker"]
        if tk in seen:
            continue
        seen.add(tk)
        hist = adhoc.get(tk)
        in_panel = tk in universe
        if hist is None and in_panel:
            sub = ohlcv.filter(pl.col("ticker") == tk).sort("date")
            if sub.height:
                hist = sub.select(["date", "adj_close"])
        if hist is None or hist.height < 5:
            candidates.append({**f, "status": "watch — not yet tradeable/priced"})
            continue
        px = float(hist["adj_close"][-1])
        closes = hist["adj_close"].to_numpy().astype(float)
        med_dollar_vol = (float(np.nanmedian(hist["volume"].to_numpy() * closes))
                          if "volume" in hist.columns else np.inf)
        if px < _MIN_PRICE or med_dollar_vol < _MIN_DOLLAR_VOL:
            candidates.append({**f, "last_price": round(px, 2),
                               "status": "watch — fails price/liquidity floor"})
            continue
        rets = np.diff(np.log(closes[closes > 0]))
        asym = asymmetry_from_returns(rets)
        candidates.append({
            **f,
            "last_price": round(px, 2),
            # a fresh name's price is from its own (runtime) history, not the
            # gold panel — stamp its prediction with the date that price is from
            "priced_as_of": str(hist["date"][-1]),
            "days_listed": hist.height,
            "asym": asym,
            "status": "ok" if asym else "watch — history too short to band",
        })
    for s in sleepers:
        tk = s["ticker"]
        if tk in seen:
            continue
        seen.add(tk)
        # date ≤ as_of: bronze can be fresher than the gold panel, and the
        # ledgered prediction is stamped as_of — no post-as_of returns allowed
        rets = (ohlcv.filter((pl.col("ticker") == tk) & (pl.col("date") <= as_of))
                .sort("date").select("adj_close").drop_nulls())
        r = np.diff(np.log(rets["adj_close"].to_numpy().astype(float)))[-504:]
        asym = asymmetry_from_returns(r)
        candidates.append({**s, "asym": asym, "status": "ok" if asym else "watch"})

    # score: asymmetry × freshness/ignition ----------------------------------
    cat_boost = {"spinoff": 1.3, "new_listing": 1.15, "ipo": 1.1, "sleeper": 1.0}
    scored, watch = [], []
    for c in candidates:
        if c.get("status", "").startswith("watch") or not c.get("asym"):
            watch.append(c)
            continue
        a = c["asym"]["asym"]
        ign = 1.0 + 0.25 * float(np.tanh(c.get("ignition_z", 0.0)))
        c["moonshot_score"] = round(a * cat_boost.get(c["category"], 1.0) * ign, 3)
        scored.append(c)
    scored.sort(key=lambda c: c["moonshot_score"], reverse=True)
    picks = scored[:top_n]

    plan = {
        "as_of": str(as_of),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "n_fresh": len(fresh),
        "n_sleepers": len(sleepers),
        "picks": picks,
        "watchlist": watch[:20],
        "note": (
            "SPECULATIVE sleeve. Fresh names have short histories — bands are "
            "bootstrap fans, not conformal; models have no coverage of them. "
            "Sleepers are in-universe but thinly traded. Cap the sleeve, size "
            "small, and judge it by its own ledger calibration "
            "(source='moonshot'), not by stories."
        ),
    }
    if record and picks:
        try:
            from ..monitoring.ledger import record_predictions
            n = record_predictions(cfg, [
                {
                    "ticker": p["ticker"],
                    "as_of": p.get("priced_as_of") or str(as_of),
                    "horizon_days": 252,
                    "entry_price": p.get("last_price"),
                    "band_lo": round(p["last_price"] * (1 + p["asym"]["lo"]), 4),
                    "band_median": round(p["last_price"] * (1 + p["asym"]["median"]), 4),
                    "band_hi": round(p["last_price"] * (1 + p["asym"]["hi"]), 4),
                    "conviction": p["moonshot_score"],
                    "model": f"discover/{p['category']}",
                } for p in picks if p.get("last_price")
            ], source="moonshot")
            plan["ledger_recorded"] = n
        except Exception as e:
            logger.warning(f"moonshot ledger recording failed: {e}")
    return plan


def render_discovery_markdown(plan: dict) -> str:
    out = [
        "# Moonshot discovery — structurally fresh + under-covered",
        "",
        f"**As of:** {plan['as_of']} · EDGAR fresh: {plan['n_fresh']} · "
        f"sleepers screened: {plan['n_sleepers']}",
        "",
        f"_{plan['note']}_",
        "",
        "| # | Ticker | Cat | Score | Price | 12m band (lo/med/hi) | Asym | Filed/Signal |",
        "| --: | --- | --- | --: | --: | --- | --: | --- |",
    ]
    for i, p in enumerate(plan["picks"], 1):
        a = p["asym"]
        sig = p.get("filed") or f"ignition z={p.get('ignition_z')}"
        out.append(
            f"| {i} | **{p['ticker']}** | {p['category']} | {p['moonshot_score']} | "
            f"{p.get('last_price', '—')} | {a['lo']:+.0%} / {a['median']:+.0%} / "
            f"{a['hi']:+.0%} | {a['asym']} | {sig} |"
        )
    if plan.get("watchlist"):
        out += ["", "**Watch (not yet band-able):**", ""]
        out += [f"- `{w['ticker']}` ({w['category']}, {w.get('filed', '')}) — {w['status']}"
                for w in plan["watchlist"]]
    return "\n".join(out)


def write_discovery(cfg: Config, plan: dict):
    import json
    out_dir = cfg.path("reports") / "discover"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.date.today().isoformat()
    md = out_dir / f"discover_{stamp}.md"
    md.write_text(render_discovery_markdown(plan))
    (out_dir / f"discover_{stamp}.json").write_text(json.dumps(plan, indent=2, default=str))
    return md
