"""Live glue: the latest calibrated cross-section → today's target book (library entry point).

Used by ``ts alpha picks`` and by the ops paper-book runner, so both see the
same names with the same weights.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from . import ledger as L
from .model import load_production, models_dir, predict_dates
from .panel import load_panel
from .portfolio import BookConfig, build_book, market_regime_on


def latest_cross_section(cfg, panel: pl.DataFrame | None = None) -> tuple[pl.DataFrame, dict]:
    """Composite score + calibrated expectations for the panel's last date.

    Prefers the live ledger rows written by ``ts alpha forecast`` (so the book is built from the
    forecasts that were *recorded*); predicts on the fly only when none exist for that date.
    """
    pn = panel if panel is not None else load_panel(cfg)
    last = pn["date"].max()
    led = L.load(L.ledger_path(cfg))
    live = led.filter((pl.col("mode") == "live") & (pl.col("date") == last))
    cal_p = models_dir(cfg) / "calibration.json"
    cal = L.Calibrator.load(cal_p) if cal_p.exists() else L.Calibrator()
    source = "ledger"
    if live.height == 0:
        models = load_production(models_dir(cfg))
        live = cal.apply(predict_dates(pn, models, [last]))
        source = "model"
    horizons = sorted(int(h) for h in live["horizon"].unique().to_list())
    w = cal.skill_weights(horizons)
    comp = L.composite_score(live.select("date", "ticker", "horizon", "score"), w)
    wide = live.pivot(values=["exp_ret", "q10", "q90"], index="ticker", on="horizon")
    today = pn.filter(pl.col("date") == last).select(
        "ticker", "sector", price=pl.col("close"), adv=(pl.col("log_dv_21").exp() - 1),
        dvol=pl.col("vol_63") / 252 ** 0.5, vol=pl.col("vol_63"), mom=pl.col("mom_12_1"), ret_21="ret_21",
        mkt_trend_200="mkt_trend_200")
    xs = today.join(comp, on="ticker", how="inner").join(wide, on="ticker", how="left")
    trend = float(xs["mkt_trend_200"][0]) if xs.height and xs["mkt_trend_200"][0] is not None else None
    meta = {"as_of": str(last), "weights": {int(h): round(v, 3) for h, v in w.items()}, "source": source,
            "calibrated": bool(cal.horizons), "n": xs.height, "mkt_trend_200": trend,
            "regime_on": market_regime_on(trend)}
    return xs, meta


def latest_targets(cfg, top_k: int = 20, book: BookConfig | None = None, prev: dict[str, float] | None = None,
                   panel: pl.DataFrame | None = None) -> tuple[pl.DataFrame, dict]:
    """Today's book (``target_weight`` and, with ``prev``, the partially-traded ``weight``)."""
    xs, meta = latest_cross_section(cfg, panel)
    bcfg = book or BookConfig(top_k=top_k)
    out = build_book(xs, bcfg, prev, regime_on=meta["regime_on"] or bcfg.regime_scale >= 1.0)
    meta["book"] = {"top_k": bcfg.top_k, "max_weight": bcfg.max_weight, "vol_target": bcfg.vol_target,
                    "trade_rate": bcfg.trade_rate, "max_per_sector": bcfg.max_per_sector}
    return out, meta


def targets_as_dict(cfg, top_k: int = 20) -> dict[str, float]:
    out, _ = latest_targets(cfg, top_k)
    return {r["ticker"]: float(r["target_weight"]) for r in out.filter(pl.col("target_weight") > 0).iter_rows(named=True)}


def write_targets_json(cfg, path: Path, top_k: int = 20) -> Path:
    out, meta = latest_targets(cfg, top_k)
    rows = [{k: (float(v) if isinstance(v, (int, float)) and k not in ("ticker", "sector") else v)
             for k, v in r.items()} for r in out.filter(pl.col("target_weight") > 0).iter_rows(named=True)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta, "targets": rows}, indent=1, default=str))
    return path
