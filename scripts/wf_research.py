#!/usr/bin/env python
"""Causal walk-forward research program: alpha, execution, and RL.

Four stages, each resumable, all writing to ``reports/research/``:

    scores     refit each alpha model forward through time, saving the
               (dates x tickers) matrix of scores it would have produced live
    backtest   replay those scores through the simulator under many execution
               policies and cost levels; apply DSR / PBO / SPA across the suite
    rl         train the PPO execution policy on the early period and evaluate
               it once on the later one, against the closed-form benchmark
    report     assemble everything into a markdown write-up

Stages are separate because only ``scores`` is expensive: an alpha refit is
minutes, a replay is seconds.  Splitting them means the interesting comparisons
cost nothing to re-run.

    python scripts/wf_research.py scores --alphas xgb63,momentum
    python scripts/wf_research.py backtest
    python scripts/wf_research.py rl --alpha xgb63
    python scripts/wf_research.py report
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from trading_system.models.cross_attn import CrossAttnAlpha, CrossAttnConfig  # noqa: E402
from trading_system.research.alphas import (  # noqa: E402
    EnsembleAlpha,
    GBMAlpha,
    MomentumAlpha,
    RandomAlpha,
)
from trading_system.research.costs import RealisticCostModel  # noqa: E402
from trading_system.research.runner import (  # noqa: E402
    Variant,
    resolve_feature_columns,
    run_suite,
)
from trading_system.research.scores import (  # noqa: E402
    PrecomputedAlpha,
    generate_causal_scores,
    load_scores,
    save_scores,
)
from trading_system.research.wfbacktest import WalkForwardConfig, build_panel  # noqa: E402

OUT = REPO / "reports" / "research"
SCORES = OUT / "scores"


# ── shared setup ─────────────────────────────────────────────────────────────

def base_config(horizon: int = 63, **kw) -> WalkForwardConfig:
    cfg = dict(
        oos_start=date(2005, 1, 1), oos_end=None,
        rebalance_days=21, retrain_days=252, min_train_days=756,
        train_window_days=None,          # expanding: use everything known so far
        horizon=horizon, neutralize=True,
        top_k=20, max_weight=0.10, risk_adjust=True,
        min_dollar_volume=2_000_000.0, min_price=3.0,
        cost=RealisticCostModel(),
    )
    cfg.update(kw)
    return WalkForwardConfig(**cfg)


ALPHAS = {
    "momentum": (lambda: MomentumAlpha(), 63),
    "random": (lambda: RandomAlpha(0), 63),
    "xgb63": (lambda: GBMAlpha("xgb"), 63),
    "xgb252": (lambda: GBMAlpha("xgb"), 252),
    "xgb21": (lambda: GBMAlpha("xgb"), 21),
    "ens63": (lambda: EnsembleAlpha(("xgb", "lgbm"), horizon=63, n_seeds=2), 63),
    "xattn63": (lambda: CrossAttnAlpha(CrossAttnConfig(epochs=30, verbose=True), n_seeds=3), 63),
    "xattn252": (lambda: CrossAttnAlpha(CrossAttnConfig(epochs=30, verbose=True), n_seeds=3), 252),
}


def load_all():
    t0 = time.time()
    ohlcv = pl.read_parquet(REPO / "data/bronze/ohlcv_daily.parquet")
    feats = pl.read_parquet(REPO / "data/gold/features.parquet")
    fc = resolve_feature_columns(feats, 0.6)
    panel = build_panel(ohlcv)
    print(f"[load] {ohlcv.height:,} rows, {len(fc)} features, "
          f"panel {panel.ret.shape} in {time.time()-t0:.0f}s", flush=True)
    return ohlcv, feats, fc, panel


# ── stage: scores ────────────────────────────────────────────────────────────

def stage_scores(args):
    ohlcv, feats, fc, panel = load_all()
    SCORES.mkdir(parents=True, exist_ok=True)
    names = [a.strip() for a in args.alphas.split(",") if a.strip()]
    for name in names:
        path = SCORES / f"{name}.npz"
        if path.exists() and not args.force:
            print(f"[scores] {name}: cached, skipping", flush=True)
            continue
        factory, horizon = ALPHAS[name]
        cfg = base_config(horizon=horizon)
        t0 = time.time()
        s, refits = generate_causal_scores(panel, feats, fc, factory, cfg, label=name)
        save_scores(path, s, panel, {"alpha": name, "horizon": horizon,
                                     "refits": len(refits)})
        cov = float(np.isfinite(s).mean())
        print(f"[scores] {name}: {len(refits)} refits, coverage {cov:.1%}, "
              f"{(time.time()-t0)/60:.1f} min -> {path}", flush=True)
        (SCORES / f"{name}_refits.json").write_text(json.dumps(refits, indent=1))


def load_alpha(name: str, panel):
    """Load a saved score matrix and align it to the current panel by date/ticker.

    Alignment matters because the panel can legitimately change shape between
    the scores stage and the backtest stage — trimming a partial trailing day
    shortens it by one row, and a re-ingest adds rows. Realigning on the saved
    axes is correct and cheap; requiring an exact shape match would force an
    expensive re-score for a one-row difference. A date the scores do not cover
    stays NaN, which the simulator already treats as "no signal".
    """
    s, dates, tickers = load_scores(SCORES / f"{name}.npz")
    di = {d: i for i, d in enumerate(dates)}
    ti = {t: i for i, t in enumerate(tickers)}

    rows = np.array([di.get(str(d), -1) for d in panel.dates])
    cols = np.array([ti.get(str(t), -1) for t in panel.tickers])
    if (rows >= 0).sum() == 0 or (cols >= 0).sum() == 0:
        raise ValueError(f"{name}: score matrix shares no dates/tickers with the "
                         "panel; re-run the scores stage")

    out = np.full((len(panel.dates), len(panel.tickers)), np.nan)
    r_ok = np.nonzero(rows >= 0)[0]
    c_ok = np.nonzero(cols >= 0)[0]
    out[np.ix_(r_ok, c_ok)] = s[np.ix_(rows[r_ok], cols[c_ok])]

    missing = len(panel.dates) - len(r_ok)
    if missing:
        print(f"[scores] {name}: {missing} panel date(s) not covered by the "
              f"saved matrix (left as no-signal)", flush=True)
    return out


def blended_scores(panel, names: list[str]) -> np.ndarray:
    """Cross-sectional rank average of several score matrices.

    Ranks, not raw values: different models put their scores on different
    scales, and the AIPM result that own-asset and cross-asset models have alpha
    against each other is the reason to combine them rather than choose.
    """
    mats = [load_alpha(n, panel) for n in names]
    T, N = mats[0].shape
    out = np.full((T, N), np.nan)
    for i in range(T):
        acc, cnt = np.zeros(N), np.zeros(N)
        for m in mats:
            row = m[i]
            ok = np.isfinite(row)
            if ok.sum() < 5:
                continue
            r = np.full(N, np.nan)
            r[ok] = row[ok].argsort().argsort() / max(ok.sum() - 1, 1)
            acc = np.where(ok, acc + r, acc)
            cnt = np.where(ok, cnt + 1, cnt)
        have = cnt > 0
        out[i, have] = acc[have] / cnt[have]
    return out


# ── stage: backtest ──────────────────────────────────────────────────────────

def stage_backtest(args):
    ohlcv, feats, fc, panel = load_all()
    available = sorted(p.stem for p in SCORES.glob("*.npz"))
    print(f"[backtest] score sets available: {available}", flush=True)

    want = [a.strip() for a in args.alphas.split(",")] if args.alphas else available
    want = [a for a in want if a in available]
    if not want:
        raise SystemExit("no score sets — run the scores stage first")

    variants: list[Variant] = []
    for name in want:
        s = load_alpha(name, panel)
        horizon = ALPHAS.get(name, (None, 63))[1]

        def mk(s=s, name=name):
            return lambda: PrecomputedAlpha(s, panel, name)

        # one execution policy per alpha, so alphas are comparable
        variants.append(Variant(
            f"{name}|full", mk(), base_config(horizon=horizon),
            is_benchmark=(name == "momentum"),
            notes="rebalance fully to target"))

    # Multi-horizon blend. This was `ops/portfolio/signals_v3.py`, a scaffold
    # that was never run end to end; blending by rank is the robust version of
    # what it proposed. Measuring it here answers the question the scaffold
    # assumed: does combining the horizons beat picking one?
    blend_parts = [a for a in ("xgb21", "xgb63", "xgb252") if a in available]
    if len(blend_parts) >= 2:
        bs = blended_scores(panel, blend_parts)
        note = f"rank blend of {'+'.join(blend_parts)}"
        blend_factory = lambda: PrecomputedAlpha(bs, panel, "blend")  # noqa: E731
        variants.append(Variant("blend_horizons|full", blend_factory,
                                base_config(horizon=63), notes=note))
        variants.append(Variant("blend_horizons|gp35", blend_factory,
                                base_config(horizon=63, partial_trade_rate=0.35),
                                notes=note + ", partial trading"))

    # execution policies, all on the single chosen alpha
    exec_alpha = args.exec_alpha if args.exec_alpha in want else want[0]
    s = load_alpha(exec_alpha, panel)
    h = ALPHAS.get(exec_alpha, (None, 63))[1]

    def mk_exec():
        return lambda: PrecomputedAlpha(s, panel, exec_alpha)

    for tag, kw in [
        ("band25", dict(band_entry=0.25, band_exit=0.125)),
        ("band50", dict(band_entry=0.50, band_exit=0.25)),
        ("gp35", dict(partial_trade_rate=0.35)),
        ("gp60", dict(partial_trade_rate=0.60)),
        ("gp35_band25", dict(partial_trade_rate=0.35, band_entry=0.25, band_exit=0.125)),
        ("noriskadj", dict(risk_adjust=False)),
        ("top10", dict(top_k=10)),
        ("top40", dict(top_k=40, max_weight=0.06)),
        ("quarterly", dict(rebalance_days=63)),
    ]:
        variants.append(Variant(f"{exec_alpha}|{tag}", mk_exec(),
                                base_config(horizon=h, **kw), notes=tag))

    # cost ladder on the base policy: an edge that dies at 3x costs is a
    # cost-model assumption, not an edge
    for mult in (2.0, 3.0):
        variants.append(Variant(
            f"{exec_alpha}|cost{mult:g}x", mk_exec(),
            base_config(horizon=h, cost=RealisticCostModel().scaled(mult)),
            notes=f"{mult:g}x costs"))

    print(f"[backtest] {len(variants)} variants", flush=True)
    t0 = time.time()
    rep, res = run_suite(variants, ohlcv, feats, fc, out_dir=OUT / "backtest",
                         n_boot=args.boot, panel=panel)
    print(f"\n[backtest] done in {(time.time()-t0)/60:.1f} min\n")
    print(rep.render())


# ── stage: rl ────────────────────────────────────────────────────────────────

def stage_rl(args):
    from trading_system.rl.env import EnvConfig
    from trading_system.rl.ppo import PPOConfig
    from trading_system.rl.train import train_rl_execution

    _, _, _, panel = load_all()
    s = load_alpha(args.alpha, panel)
    res = train_rl_execution(
        panel, s, split_date=date(2016, 1, 1),
        cost=RealisticCostModel(),
        env_cfg=EnvConfig(rebalance_days=21, episode_days=504, top_k=20),
        ppo_cfg=PPOConfig(total_updates=args.updates, steps_per_update=1024,
                          seed=args.seed),
        out_dir=OUT / "rl" / args.alpha,
    )
    print(res.render())


# ── stage: report ────────────────────────────────────────────────────────────

def stage_report(args):
    from trading_system.research.report import write_report
    path = write_report(OUT)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("scores")
    p.add_argument("--alphas", default="momentum,xgb63")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=stage_scores)

    p = sub.add_parser("backtest")
    p.add_argument("--alphas", default="")
    p.add_argument("--exec-alpha", default="xgb63")
    p.add_argument("--boot", type=int, default=500)
    p.set_defaults(fn=stage_backtest)

    p = sub.add_parser("rl")
    p.add_argument("--alpha", default="xgb63")
    p.add_argument("--updates", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=stage_rl)

    p = sub.add_parser("report")
    p.set_defaults(fn=stage_report)

    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    args.fn(args)


if __name__ == "__main__":
    main()
