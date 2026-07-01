"""Long-term decision artifact — what to buy, when, at what price.

This is the packaging layer over the trained long-horizon forecaster + the
calibrated conformal bounds. For a chosen horizon (default 252d — the one that
carried genuine cross-sectional edge) it:

  1. ranks the universe by the horizon model's **cross-sectional** score,
  2. attaches each top name's calibrated price band (lower / median / upper),
  3. turns the band into an actionable plan — entry zone, add-on-dip level,
     target (upper quantile), invalidation/stop (conformal lower), and the
     reward-to-risk — plus a timing hint from where price sits in its trend.

Not fast trading: these are accumulate-over-time, long-horizon decisions.
"""
from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import polars as pl

from ..config import Config, get_config
from ..utils import get_logger

logger = get_logger(__name__)

_HZLABEL = {5: "5d", 21: "1m", 63: "3m", 126: "6m", 252: "12m"}


def _timing_hint(rsi, sma_gap_50) -> str:
    """A coarse entry-timing read for a long-horizon accumulator."""
    rsi = rsi if rsi is not None else 50.0
    gap = sma_gap_50 if sma_gap_50 is not None else 0.0
    if rsi > 70 or gap > 0.15:
        return "extended — wait for a pullback"
    if rsi < 45 or gap < -0.03:
        return "pullback — accumulate now"
    return "neutral — scale in"


def build_longterm_picks(
    cfg: Config | None = None,
    horizon: int = 252,
    top_n: int = 20,
    min_score: float = 0.0,
) -> dict[str, Any]:
    """Return a ranked, actionable long-term buy plan for the universe.

    Uses the committed ``models_store`` forecast model for ``horizon`` to rank
    names, and the conformal interval bundle for each name's price band.
    """
    cfg = cfg or get_config()
    from .analyze import _load_or_build_features
    from ..models.store import load_forecast_model
    from .bounds import compute_bounds

    ohlcv, features = _load_or_build_features(cfg)
    last_date = features["date"].max()

    loaded = load_forecast_model(horizon, cfg.project_root / "models_store")
    if loaded is None:
        raise FileNotFoundError(
            f"No {horizon}d model in models_store/ — run `ts train-forecast` first."
        )
    model, meta = loaded
    feat_cols = [c for c in meta["feature_columns"] if c in features.columns]

    today = features.filter(pl.col("date") == last_date).drop_nulls(subset=feat_cols)
    if today.is_empty():
        raise ValueError("no complete feature rows for the latest date")
    X = today.select(feat_cols).to_numpy().astype(np.float64)
    scores = np.asarray(model.predict(X)).ravel()
    tickers = today["ticker"].to_list()

    # price + a couple of timing features for the latest row
    def _col(name):
        return {r["ticker"]: r.get(name) for r in today.select(["ticker", name]).to_dicts()} \
            if name in today.columns else {}
    px = _col("adj_close")
    rsi = _col("rsi_14")
    gap50 = _col("sma_gap_50")

    ranked = sorted(zip(tickers, scores), key=lambda t: t[1], reverse=True)
    leak = (meta.get("leakage_gate") or {}).get("pass")

    picks: list[dict] = []
    for tk, sc in ranked:
        if sc <= min_score or len(picks) >= top_n:
            if len(picks) >= top_n:
                break
            continue
        last_price = float(px.get(tk) or 0.0)
        if last_price <= 0:
            continue
        b = compute_bounds(cfg, tk, features, ohlcv, last_price, float(sc))
        horizons_b = (b or {}).get("horizons", {})
        hz = horizons_b.get(_HZLABEL.get(horizon, f"{horizon}d"))
        if not hz:
            continue
        p = hz["price"]
        entry = last_price
        target = p["hi"]
        stop = p["lo"]
        median = p["median"]
        # "Add on a dip" = a realistic near-term pullback price (1-month lower band),
        # capped below the entry so it's genuinely a better accumulation level.
        near = horizons_b.get("1m", {}).get("price", {})
        add_dip = min(entry, near.get("lo", entry * 0.92))
        downside = entry - stop
        rr = round((target - entry) / downside, 2) if downside > 1e-9 else None
        picks.append({
            "ticker": tk,
            "score": round(float(sc), 5),
            "entry": round(entry, 2),
            "add_on_dip_below": round(add_dip, 2),
            "median_target": round(median, 2),
            "stretch_target": round(target, 2),
            "invalidation_stop": round(stop, 2),
            "upside_pct": round(target / entry - 1, 3),
            "downside_pct": round(stop / entry - 1, 3),
            "reward_risk": rr,
            "timing": _timing_hint(rsi.get(tk), gap50.get(tk)),
        })

    return {
        "as_of": str(last_date),
        "horizon_days": horizon,
        "horizon_label": _HZLABEL.get(horizon, f"{horizon}d"),
        "model": meta.get("best_model"),
        "model_leak_pass": leak,
        "coverage": (b or {}).get("target_coverage") if picks else None,
        "n_ranked": len(ranked),
        "picks": picks,
    }


def render_picks_markdown(plan: dict) -> str:
    out = [
        f"# Long-term picks — {plan['horizon_label']} horizon",
        "",
        f"**As of:** {plan['as_of']} · **Model:** {plan['model']} "
        f"(leakage gate: {plan['model_leak_pass']}) · **Ranked:** {plan['n_ranked']} names",
        "",
        "_Accumulate-over-time, long-horizon decisions. Entry = last close; "
        "add-on-dip = 25th-pct price; targets are the calibrated median / upper band; "
        "stop = conformal lower band. Not financial advice._",
        "",
        "| # | Ticker | Score | Entry | Add ≤ | Median | Stretch | Stop | Up% | Dn% | R/R | Timing |",
        "| --: | --- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --- |",
    ]
    for i, p in enumerate(plan["picks"], 1):
        out.append(
            f"| {i} | **{p['ticker']}** | {p['score']:+.4f} | {p['entry']:.2f} | "
            f"{p['add_on_dip_below']:.2f} | {p['median_target']:.2f} | {p['stretch_target']:.2f} | "
            f"{p['invalidation_stop']:.2f} | {p['upside_pct']*100:+.0f}% | {p['downside_pct']*100:+.0f}% | "
            f"{p['reward_risk'] if p['reward_risk'] is not None else '—'} | {p['timing']} |"
        )
    out += [
        "",
        "> **Caveats.** The universe is *today's* constituents, so the ranking is "
        "**survivorship-biased** (delisted/failed names are absent, recent IPOs have "
        "short history) — read the numbers with that discount. The forecast is "
        "model-implied, not a guarantee; size to the invalidation stop, not the target.",
    ]
    return "\n".join(out)


def write_picks(cfg: Config, plan: dict) -> "Path":  # type: ignore[name-defined]
    from pathlib import Path
    out_dir = cfg.path("reports") / "picks"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    md = out_dir / f"picks_{plan['horizon_label']}_{stamp}.md"
    js = out_dir / f"picks_{plan['horizon_label']}_{stamp}.json"
    md.write_text(render_picks_markdown(plan))
    import json
    js.write_text(json.dumps(plan, indent=2, default=str))
    return md
