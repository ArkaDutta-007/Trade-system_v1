"""Run and compare walk-forward simulations, then judge them honestly.

``run_suite`` executes a list of named configurations through the causal
simulator and produces one report containing, for every configuration: the
performance table, a stationary-bootstrap confidence interval on Sharpe, the
deflated Sharpe against the whole set of trials, the PBO of the set, and an SPA
test of the best learned model against the rule-based benchmark.

The corrections are computed *across the suite*, not per configuration, because
that is the search that actually happened.  Reporting the best row's raw Sharpe
without them is the single most common way a backtest lies.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import polars as pl

from ..backtesting.metrics import compute_metrics
from ..features.reserve import resolve_reserve
from ..utils import get_logger
from .bias import estimate_survivorship_bias
from .stats import bootstrap_ci, deflated_sharpe, pbo_cscv, sharpe, spa_test
from .wfbacktest import (
    AlphaModel,
    PanelData,
    WalkForwardConfig,
    WalkForwardResult,
    build_panel,
    run_walk_forward,
)

logger = get_logger(__name__)

__all__ = ["Variant", "run_suite", "load_inputs", "resolve_feature_columns",
           "summarize_result", "passive_benchmarks"]


@dataclass
class Variant:
    """One named configuration: a model factory plus a simulator config."""

    name: str
    model_factory: Callable[[], AlphaModel]
    config: WalkForwardConfig
    is_benchmark: bool = False
    notes: str = ""


def load_inputs(repo: Path) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Bronze OHLCV and the gold feature matrix."""
    ohlcv = pl.read_parquet(repo / "data/bronze/ohlcv_daily.parquet")
    feats = pl.read_parquet(repo / "data/gold/features.parquet")
    return ohlcv, feats


def resolve_feature_columns(
    feats: pl.DataFrame, min_coverage: float = 0.6, groups: list[str] | None = None
) -> list[str]:
    """Reserve columns with enough coverage, minus anything non-numeric.

    The coverage gate matters more here than in the existing trainer: the
    simulator refits on an *expanding* window, so a feature that only exists
    after 2017 (GDELT news) is absent for the first decade of the walk.  The
    densify step gives those columns a presence flag and a neutral fill, which
    is what keeps an early refit from dropping every row.
    """
    cols = resolve_reserve(feats, groups=groups, min_non_null_frac=min_coverage)
    return [c for c in cols if feats.schema[c].is_numeric()]


def summarize_result(res: WalkForwardResult) -> dict:
    """Headline metrics for one simulation."""
    r = res.returns()
    daily = res.daily
    n_years = len(r) / 252 if len(r) else 0.0
    final_eq = float(daily["equity"][-1]) if daily.height else 0.0
    m = compute_metrics(r, turnover=daily["turnover"].to_numpy())
    ann_turn = float(daily["turnover"].sum() / n_years) if n_years > 0 else 0.0

    # Cost drag must be measured against the equity that was actually at risk on
    # the day the cost was paid. Dividing total dollars of cost by the *initial*
    # cash — the obvious version — reports 39%/yr for a book that grew 145x and
    # is not comparable between two strategies that compounded differently.
    eq = daily["equity"].to_numpy()
    cost_ratio = daily["cost"].to_numpy() / np.maximum(eq, 1e-9)
    cost_drag = float(cost_ratio.sum() / n_years) if n_years > 0 else 0.0

    shortfall = 0.0
    if res.trades.height:
        want = float(res.trades["notional"].abs().sum()) + \
               float(res.trades["shortfall"].abs().sum())
        shortfall = float(res.trades["shortfall"].abs().sum()) / max(want, 1.0)

    return {
        "variant": res.label,
        "start": str(daily["date"][0]) if daily.height else "",
        "end": str(daily["date"][-1]) if daily.height else "",
        "years": round(n_years, 2),
        "CAGR": round(m["CAGR"], 4),
        "Sharpe": round(m["Sharpe"], 3),
        "Sortino": round(m["Sortino"], 3),
        "MaxDD": round(m["MaxDrawdown"], 4),
        "Calmar": round(m["Calmar"], 3),
        "AnnVol": round(m["AnnualVol"], 4),
        "ann_turnover": round(ann_turn, 3),
        "cost_drag_annual": round(cost_drag, 5),
        "final_equity": round(final_eq, 2),
        "n_refits": len(res.refits),
        "unfilled_frac": round(shortfall, 4),
    }


def benchmark_returns(panel: PanelData, ticker: str, dates: list) -> np.ndarray:
    """Buy-and-hold returns of one ticker aligned to ``dates``."""
    try:
        j = list(panel.tickers).index(ticker)
    except ValueError:
        return np.zeros(len(dates))
    didx = panel.date_index()
    return np.array([panel.ret[didx[d], j] if d in didx else 0.0 for d in dates])


#: index and sector ETFs that sit in the universe file but are not stock picks.
#: They must be excluded from the equal-weight benchmark, otherwise it is partly
#: a benchmark of itself.
BENCHMARK_ETFS = {"SPY", "QQQ", "IWM", "DIA", "VTI", "RSP", "VOO", "VXX"}


def passive_benchmarks(panel: PanelData, dates: list) -> dict[str, np.ndarray]:
    """Return streams no strategy should be compared without.

    ``spy`` is the index; ``universe_ew`` is a daily-rebalanced equal weight of
    every live *stock* in the universe.  The second one matters more than it
    looks: on a survivorship-biased universe it is itself inflated, so a
    strategy that merely matches it has demonstrated the bias, not skill.  See
    :mod:`trading_system.research.bias` for how much that is worth here.
    """
    didx = panel.date_index()
    idx = np.array([didx[d] for d in dates if d in didx])
    tickers = list(panel.tickers)
    out = {}
    if "SPY" in tickers:
        out["spy"] = panel.ret[idx, tickers.index("SPY")]

    keep = np.array([t not in BENCHMARK_ETFS for t in tickers])
    r = panel.ret[np.ix_(idx, np.nonzero(keep)[0])]
    a = panel.alive[np.ix_(idx, np.nonzero(keep)[0])]
    n = a.sum(axis=1)
    out["universe_ew"] = np.where(n > 0, (r * a).sum(axis=1) / np.maximum(n, 1), 0.0)
    return out


@dataclass
class SuiteReport:
    table: list[dict] = field(default_factory=list)
    robustness: dict = field(default_factory=dict)
    per_variant: dict = field(default_factory=dict)

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"table": self.table, "robustness": self.robustness,
             "per_variant": self.per_variant},
            indent=2, default=str))

    def render(self) -> str:
        if not self.table:
            return "(no results)"
        cols = ["variant", "years", "CAGR", "Sharpe", "MaxDD", "ann_turnover",
                "cost_drag_annual", "unfilled_frac"]
        w = {c: max(len(c), *(len(f"{r.get(c,'')}") for r in self.table)) for c in cols}
        lines = ["  ".join(c.rjust(w[c]) for c in cols)]
        lines.append("  ".join("-" * w[c] for c in cols))
        for r in sorted(self.table, key=lambda x: -x.get("Sharpe", -9)):
            lines.append("  ".join(f"{r.get(c,'')}".rjust(w[c]) for c in cols))
        rb = self.robustness
        if rb:
            lines += ["", f"PBO(CSCV) = {rb.get('pbo', float('nan')):.1%} over "
                          f"{rb.get('n_configs', 0)} configs "
                          f"(median OOS rank of the IS winner: "
                          f"{rb.get('median_oos_rank', float('nan')):.2f})"]
            for name, d in (rb.get("deflated") or {}).items():
                lines.append(f"  {name:<28} DSR={d['dsr']:.3f}  "
                             f"Sharpe 95% CI [{d['ci_lo']:+.2f}, {d['ci_hi']:+.2f}]")
            if "spa" in rb:
                s = rb["spa"]
                lines.append(f"  SPA vs {s['benchmark']}: p = {s['p_value']:.3f} "
                             f"over {s['n_models']} learned models")
            sv = rb.get("survivorship")
            if sv:
                lines += ["", "Survivorship bias in the universe itself:",
                          f"  universe equal-weight : CAGR {sv['universe_cagr']:+.2%}  "
                          f"Sharpe {sv['universe_sharpe']:.2f}",
                          f"  {sv['reference'][:40]:<40}: CAGR {sv['reference_cagr']:+.2%}  "
                          f"Sharpe {sv['reference_sharpe']:.2f}",
                          f"  unearned                : CAGR {sv['cagr_bias']:+.2%}  "
                          f"Sharpe {sv['sharpe_bias']:+.2f}"]
        return "\n".join(lines)


def run_suite(
    variants: list[Variant],
    ohlcv: pl.DataFrame,
    feats: pl.DataFrame,
    feat_cols: list[str],
    out_dir: Path | None = None,
    n_boot: int = 500,
    panel: PanelData | None = None,
) -> tuple[SuiteReport, dict[str, WalkForwardResult]]:
    panel = panel or build_panel(ohlcv)
    results: dict[str, WalkForwardResult] = {}
    rows: list[dict] = []

    for v in variants:
        logger.info(f"── walk-forward: {v.name}")
        res = run_walk_forward(
            panel, feats, feat_cols, v.model_factory, v.config, label=v.name)
        results[v.name] = res
        row = summarize_result(res)
        row["notes"] = v.notes
        rows.append(row)
        logger.info(f"   {v.name}: CAGR {row['CAGR']:+.2%} Sharpe {row['Sharpe']} "
                    f"MaxDD {row['MaxDD']:.1%} turnover {row['ann_turnover']}x")
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            res.daily.write_parquet(out_dir / f"daily_{v.name}.parquet")
            if res.weights.height:
                res.weights.write_parquet(out_dir / f"weights_{v.name}.parquet")

    # ── passive benchmarks, on the same dates ──
    any_res = next(iter(results.values()))
    bench_dates = any_res.daily["date"].to_list()
    passive = passive_benchmarks(panel, bench_dates)
    for bname, br in passive.items():
        yrs = len(br) / 252
        eq = float(np.prod(1 + br))
        peak = np.maximum.accumulate(np.cumprod(1 + br))
        rows.append({
            "variant": bname, "start": str(bench_dates[0]), "end": str(bench_dates[-1]),
            "years": round(yrs, 2),
            "CAGR": round(eq ** (1 / max(yrs, 1e-9)) - 1, 4),
            "Sharpe": round(sharpe(br), 3),
            "MaxDD": round(float((np.cumprod(1 + br) / peak - 1).min()), 4),
            "AnnVol": round(float(br.std(ddof=1) * np.sqrt(252)), 4),
            "ann_turnover": 0.0, "cost_drag_annual": 0.0, "unfilled_frac": 0.0,
            "n_refits": 0, "final_equity": round(eq * 1e6, 2),
            "notes": "passive benchmark, no costs",
        })

    # ── robustness across the whole search ──
    common = min(len(r.returns()) for r in results.values())
    names = list(results)
    R = np.column_stack([results[n].returns()[-common:] for n in names])
    trial_sr = R.mean(0) / (R.std(0, ddof=1) + 1e-12)

    deflated = {}
    for j, n in enumerate(names):
        d = deflated_sharpe(R[:, j], trial_sr)
        ci = bootstrap_ci(R[:, j], stat=sharpe, n_boot=n_boot)
        deflated[n] = {"dsr": d["dsr"], "sr0_annual": d["sr0_annual"],
                       "ci_lo": ci["lo"], "ci_hi": ci["hi"], "p_gt_0": ci["p_gt_0"]}

    rb: dict = {"deflated": deflated, **pbo_cscv(R)} if R.shape[1] >= 2 else \
               {"deflated": deflated}

    bench = next((v for v in variants if v.is_benchmark), None)
    if bench is not None and bench.name in names:
        learned = [n for n in names if n != bench.name]
        if learned:
            bi = names.index(bench.name)
            lm = np.column_stack([R[:, names.index(n)] for n in learned])
            s = spa_test(-R[:, bi], -lm, n_boot=min(n_boot, 500))
            rb["spa"] = {"benchmark": bench.name, "p_value": s["p_value"],
                         "t_spa": s["t_spa"], "n_models": s["n_models"],
                         "best_model": learned[s["best_model"]]}

    # How much of the universe's return is simply the universe being chosen
    # after the fact? Without this the whole table is read against the wrong
    # baseline.
    bias = None
    if "universe_ew" in passive:
        try:
            bias = estimate_survivorship_bias(passive["universe_ew"], bench_dates)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"survivorship-bias estimate failed: {e}")
    if bias is not None:
        rb["survivorship"] = {
            "reference": bias.reference,
            "universe_cagr": bias.universe_cagr, "reference_cagr": bias.reference_cagr,
            "universe_sharpe": bias.universe_sharpe,
            "reference_sharpe": bias.reference_sharpe,
            "cagr_bias": bias.cagr_bias, "sharpe_bias": bias.sharpe_bias,
        }
        logger.info(bias.render())

    report = SuiteReport(table=rows, robustness=rb,
                         per_variant={n: {"refits": r.refits} for n, r in results.items()})
    if out_dir:
        report.to_json(out_dir / "suite_report.json")
    return report, results
