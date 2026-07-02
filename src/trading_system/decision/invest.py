"""Budget-aware invest planner — "I have $X right now: what do I buy,
how much of each, and how long do I hold it?"

The full decision stack in one artifact:

  1. **Rank** — every committed ``models_store`` horizon (21/63/126/252d)
     scores the latest cross-section; per-horizon scores are z-scored and
     blended into one conviction, weighted by each horizon's honest quality
     (ICIR, haircut hard when its leakage gate failed).
  2. **Hold horizon** — per name, the calibrated conformal band picks the
     horizon with the best *annualized* reward-to-downside × model quality.
     That horizon is the recommended holding period, with the band's median /
     stretch / stop as review levels.
  3. **Gate** — every BUY clears playbook compliance (never-buy, lockouts,
     caps, semi freeze); the composite flag board scales how much of the
     budget deploys at all. Blocked names are reported, not hidden.
  4. **Size** — conviction (Kelly on band edge/downside) blended with
     RMT-cleaned Hierarchical Risk Parity over trailing correlations, capped
     per name, converted to dollars and shares within the budget.
  5. **Learn** — each position is appended to the decision ledger as a
     falsifiable prediction, so ``ts ledger`` can later score whether this
     advice was any good. The system evolves or it is nothing.

Not fast trading: this is a deploy-a-tranche, hold-for-months plan.
"""
from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import polars as pl

from ..config import Config, get_config
from ..utils import get_logger

logger = get_logger(__name__)

_HZLABEL = {5: "5d", 21: "1m", 63: "3m", 126: "6m", 252: "12m"}
_HOLD_LABEL = {5: "~1 week", 21: "~1 month", 63: "~3 months",
               126: "~6 months", 252: "~12 months"}
_DOWNSIDE_FLOOR = 0.02


def horizon_reliability(meta: dict) -> float:
    """Honest model quality → blend weight: ICIR, haircut if the leakage gate failed."""
    icir = float(meta.get("icir") or 0.0)
    leak = meta.get("leak_pass")
    return max(icir, 0.05) * (1.0 if leak else 0.35)


def choose_hold_horizon(
    bands: dict[str, dict],
    reliability: dict[int, float],
) -> tuple[int, dict] | None:
    """Pick the hold horizon with the best annualized reward-to-downside.

    ``bands`` is ``bounds["horizons"]`` (label → {days, price, return});
    edge is annualized linearly, downside by √t, and each horizon is weighted
    by its model's reliability. Returns ``(days, quality_info)`` or ``None``
    when no horizon has positive median edge.
    """
    best: tuple[float, int, dict] | None = None
    for label, hz in bands.items():
        h = int(hz.get("days") or 0)
        r = hz.get("return") or {}
        med, lo = float(r.get("median", 0.0)), float(r.get("lo", 0.0))
        if h <= 0 or med <= 0:
            continue
        ann_edge = med * 252.0 / h
        ann_dn = max(_DOWNSIDE_FLOOR, -lo) * np.sqrt(252.0 / h)
        rel = reliability.get(h, 0.05)
        quality = ann_edge / ann_dn * rel
        info = {
            "label": label,
            "annualized_edge": round(ann_edge, 4),
            "reward_downside": round(med / max(_DOWNSIDE_FLOOR, -lo), 3),
            "quality": round(quality, 4),
        }
        if best is None or quality > best[0]:
            best = (quality, h, info)
    if best is None:
        return None
    return best[1], best[2]


def _timing_hint(rsi, sma_gap_50) -> str:
    from .longterm import _timing_hint as hint
    return hint(rsi, sma_gap_50)


def _trailing_returns(
    ohlcv: pl.DataFrame, tickers: list[str], lookback: int = 252
) -> np.ndarray | None:
    """(T, N) daily-return matrix aligned to ``tickers``; None if too short."""
    sub = (
        ohlcv.filter(pl.col("ticker").is_in(tickers))
        .select(["date", "ticker", "adj_close"])
        .pivot(on="ticker", index="date", values="adj_close")
        .sort("date")
        .tail(lookback + 1)
        .drop_nulls()
    )
    missing = [t for t in tickers if t not in sub.columns]
    if missing or sub.height < 61:
        return None
    px = sub.select(tickers).to_numpy().astype(np.float64)
    return px[1:] / px[:-1] - 1.0


def build_invest_plan(
    cfg: Config | None = None,
    budget: float = 1000.0,
    top_n: int = 8,
    max_weight: float = 0.25,
    min_position: float = 50.0,
    use_flags: bool = True,
    record: bool = True,
) -> dict[str, Any]:
    """Turn a dollar budget into a gated, sized, hold-horizon-annotated buy plan."""
    cfg = cfg or get_config()
    from .analyze import _load_or_build_features
    from .bounds import compute_bounds
    from ..models.store import read_manifest, load_forecast_model
    from ..models.forecast_train import forecast_scores_latest
    from ..portfolio.allocate import blend_weights, budget_to_positions

    store_dir = cfg.project_root / "models_store"
    manifest = read_manifest(store_dir)
    horizon_meta = {int(h): m for h, m in (manifest.get("horizons") or {}).items()}
    if not horizon_meta:
        raise FileNotFoundError(
            "models_store/ has no committed forecasters — run `ts train-forecast` first."
        )

    ohlcv, features = _load_or_build_features(cfg)
    as_of = features["date"].max()
    latest = features.filter(pl.col("date") == as_of)
    px = {r["ticker"]: r.get("adj_close") for r in latest.to_dicts()}
    rsi = {r["ticker"]: r.get("rsi_14") for r in latest.to_dicts()} \
        if "rsi_14" in latest.columns else {}
    gap50 = {r["ticker"]: r.get("sma_gap_50") for r in latest.to_dicts()} \
        if "sma_gap_50" in latest.columns else {}

    # 1. score every committed horizon, blend into one conviction ---------------
    reliability: dict[int, float] = {}
    zscores: dict[int, dict[str, float]] = {}
    for h, meta_h in sorted(horizon_meta.items()):
        loaded = load_forecast_model(h, store_dir)
        if loaded is None:
            continue
        model, meta = loaded
        try:
            tickers_h, scores_h, _ = forecast_scores_latest(model, meta, features)
        except Exception as e:
            logger.warning(f"{h}d scoring failed: {e}")
            continue
        s = np.asarray(scores_h, dtype=np.float64)
        sd = s.std()
        z = (s - s.mean()) / sd if sd > 1e-12 else np.zeros_like(s)
        zscores[h] = dict(zip(tickers_h, z))
        reliability[h] = horizon_reliability(meta_h)
    if not zscores:
        raise RuntimeError("no committed forecaster could score the latest cross-section")

    conviction: dict[str, float] = {}
    for tk in set().union(*[set(d) for d in zscores.values()]):
        num = den = 0.0
        for h, zmap in zscores.items():
            if tk in zmap:
                num += reliability[h] * zmap[tk]
                den += reliability[h]
        if den > 0:
            conviction[tk] = num / den

    # 2. flag board decides how much of the budget deploys ----------------------
    snapshot = None
    if use_flags:
        try:
            from ..flags.service import get_flag_snapshot
            snapshot = get_flag_snapshot(cfg)
        except Exception as e:
            logger.warning(f"flag snapshot unavailable ({e}) — deploying without gating")
    deploy_frac = float(snapshot.composite.deployment_fraction) if snapshot else 1.0
    deployable = budget * deploy_frac

    from ..playbook.loader import load_playbook, load_portfolio
    from ..playbook.compliance import check_trade
    playbook = portfolio = None
    try:
        playbook, portfolio = load_playbook(cfg), load_portfolio(cfg)
    except Exception as e:
        logger.warning(f"playbook unavailable ({e}) — compliance gate skipped")
    sma50 = {
        t: float(px[t]) / (1.0 + float(g))
        for t, g in gap50.items()
        if g is not None and px.get(t)
    }

    # 3. walk candidates in conviction order: bounds → hold horizon → compliance -
    pool = sorted(conviction.items(), key=lambda kv: kv[1], reverse=True)
    pool = [(t, c) for t, c in pool if (px.get(t) or 0) > 0][: max(top_n * 3, 24)]

    selected: list[dict] = []
    skipped: list[dict] = []
    kelly: dict[str, float] = {}
    for tk, conv in pool:
        if len(selected) >= top_n:
            break
        if conv <= 0:
            skipped.append({"ticker": tk, "reason": "negative blended conviction"})
            continue
        b = compute_bounds(cfg, tk, features, ohlcv, float(px[tk]), 0.0)
        if not b or not b.get("horizons"):
            skipped.append({"ticker": tk, "reason": "no price bounds available"})
            continue
        chosen = choose_hold_horizon(b["horizons"], reliability)
        if chosen is None:
            skipped.append({"ticker": tk, "reason": "no horizon with positive median edge"})
            continue
        hold_days, qinfo = chosen
        hz = b["horizons"][qinfo["label"]]

        warnings: list[str] = []
        if playbook is not None and portfolio is not None:
            res = check_trade(tk, "BUY", 0.0, playbook, portfolio,
                              snapshot=snapshot, prices=px, sma50=sma50)
            if not res.allowed:
                skipped.append({"ticker": tk, "reason": "; ".join(res.violations)})
                continue
            warnings = res.warnings

        r, p = hz["return"], hz["price"]
        kelly[tk] = max(0.0, float(r["median"]) / max(_DOWNSIDE_FLOOR, -float(r["lo"])))
        selected.append({
            "ticker": tk,
            "conviction": round(float(conv), 4),
            "entry": round(float(px[tk]), 2),
            "hold_days": hold_days,
            "hold": _HOLD_LABEL.get(hold_days, f"~{hold_days} trading days"),
            "median_target": round(float(p["median"]), 2),
            "stretch_target": round(float(p["hi"]), 2),
            "stop": round(float(p["lo"]), 2),
            "expected_return": round(float(r["median"]), 4),
            "annualized_edge": qinfo["annualized_edge"],
            "reward_downside": qinfo["reward_downside"],
            "model": horizon_meta.get(hold_days, {}).get("best_model"),
            "model_icir": horizon_meta.get(hold_days, {}).get("icir"),
            "leak_pass": horizon_meta.get(hold_days, {}).get("leak_pass"),
            "timing": _timing_hint(rsi.get(tk), gap50.get(tk)),
            "warnings": warnings,
            "bounds_method": b.get("method", "conformal"),
        })

    # 4. size: conviction × RMT-cleaned HRP, then budget → shares ---------------
    tickers = [s["ticker"] for s in selected]
    rets = _trailing_returns(ohlcv, tickers) if len(tickers) >= 2 else None
    weights = blend_weights(tickers, kelly, rets, max_weight=max_weight)
    positions, leftover = budget_to_positions(
        weights, {t: float(px[t]) for t in tickers}, deployable,
        min_position=min_position,
    )
    dropped_small = [t for t in tickers if t not in positions]
    for s in selected:
        pos = positions.get(s["ticker"])
        if pos:
            s.update(pos)
    selected = [s for s in selected if s["ticker"] in positions]
    for t in dropped_small:
        skipped.append({"ticker": t,
                        "reason": f"allocation below ${min_position:.0f} minimum"})
    invested = round(sum(s["dollars"] for s in selected), 2)

    plan: dict[str, Any] = {
        "as_of": str(as_of),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "budget": round(budget, 2),
        "deployment_fraction": deploy_frac,
        "deployable": round(deployable, 2),
        "invested": invested,
        "cash_reserve": round(budget - invested, 2),
        "composite": snapshot.composite.to_dict() if snapshot else None,
        "horizon_reliability": {h: round(v, 4) for h, v in reliability.items()},
        "positions": selected,
        "skipped": skipped,
        "note": (
            "Long-horizon tranche plan. Hold = the horizon where the calibrated "
            "band shows the best annualized reward-to-downside; review at the "
            "stop (thesis invalidated) or median target (thesis played out). "
            "Universe is survivorship-biased; not financial advice."
        ),
    }

    # 5. every position becomes a falsifiable ledger prediction -----------------
    if record and selected:
        try:
            from ..monitoring.ledger import record_predictions
            n = record_predictions(cfg, [
                {
                    "ticker": s["ticker"],
                    "as_of": str(as_of),
                    "horizon_days": s["hold_days"],
                    "entry_price": s["entry"],
                    "band_lo": s["stop"],
                    "band_median": s["median_target"],
                    "band_hi": s["stretch_target"],
                    "conviction": s["conviction"],
                    "weight": s.get("weight"),
                    "dollars": s.get("dollars"),
                    "model": s.get("model"),
                    "icir": s.get("model_icir"),
                    "leak_pass": s.get("leak_pass"),
                    "composite": (snapshot.composite.color.value
                                  if snapshot else None),
                } for s in selected
            ], source="invest")
            plan["ledger_recorded"] = n
        except Exception as e:
            logger.warning(f"ledger recording failed: {e}")
    return plan


def render_invest_markdown(plan: dict) -> str:
    comp = plan.get("composite") or {}
    out = [
        f"# Invest plan — ${plan['budget']:,.0f} tranche",
        "",
        f"**As of:** {plan['as_of']} · **Composite:** {comp.get('color', 'n/a')} "
        f"(deploy {plan['deployment_fraction']:.0%}) · "
        f"**Invested:** ${plan['invested']:,.2f} · "
        f"**Cash reserve:** ${plan['cash_reserve']:,.2f}",
        "",
        f"_{plan['note']}_",
        "",
        "| # | Ticker | $ | Shares | Wt | Entry | Median | Stretch | Stop | Hold | Ann.Edge | Timing |",
        "| --: | --- | --: | --: | --: | --: | --: | --: | --: | --- | --: | --- |",
    ]
    for i, s in enumerate(plan["positions"], 1):
        out.append(
            f"| {i} | **{s['ticker']}** | ${s['dollars']:,.0f} | {s['shares']:.3f} | "
            f"{s['weight']*100:.1f}% | {s['entry']:.2f} | {s['median_target']:.2f} | "
            f"{s['stretch_target']:.2f} | {s['stop']:.2f} | {s['hold']} | "
            f"{s['annualized_edge']*100:+.0f}% | {s['timing']} |"
        )
    if plan.get("skipped"):
        out += ["", "**Not bought (and why):**", ""]
        out += [f"- `{s['ticker']}` — {s['reason']}" for s in plan["skipped"]]
    return "\n".join(out)


def write_invest_plan(cfg: Config, plan: dict):
    import json
    from pathlib import Path
    out_dir = cfg.path("reports") / "invest"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.date.today().isoformat()
    md = out_dir / f"invest_{stamp}.md"
    md.write_text(render_invest_markdown(plan))
    (out_dir / f"invest_{stamp}.json").write_text(json.dumps(plan, indent=2, default=str))
    return md
