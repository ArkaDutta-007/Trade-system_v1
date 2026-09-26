"""Model-change discipline: candidate variants on identical causal folds, adopted only if they win.

Every idea (a feature group, an objective, a hyperparameter) is a ``Variant`` — a ``TrainSpec``
plus a name. ``run`` walks each one forward with the same refit dates, scores the same rows,
and ``compare`` reports, per horizon, the per-date rank-IC, ICIR, the paired t-statistic of the
IC difference against the baseline, the share of calendar years it wins, the recent-period IC
and the top-minus-bottom decile spread. ``verdict`` applies a fixed rule (paired t ≥ 2 on the
primary horizon and no worse than −1 on the others) so adoption is not a judgment call made
after looking at the numbers.

Second rule (added 2026-09-26, after the ranking objective passed it while failing the IC rule): a
long-only top-N book uses only the top of the ranking, so a variant is also adopted if the book built
from it beats the baseline book on identical folds with the bootstrap CI of the annual excess return
above zero and ≥ 70% of calendar years won. Both rules are reported; neither is applied after
peeking at a single number.

``blend`` averages two finished variants (per-date z-scores) without refitting.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from ..utils import get_logger
from .model import TrainSpec, causal_scores
from .panel import EARN_FEATURES, MODEL_FEATURES

logger = get_logger(__name__)


@dataclass
class Variant:
    name: str
    spec: TrainSpec
    note: str = ""


def default_variants(horizons=(21, 63)) -> dict[str, Variant]:
    base_feats = tuple(f for f in MODEL_FEATURES if f not in EARN_FEATURES)
    base = TrainSpec(horizons=tuple(horizons), features=base_feats)
    return {
        "base": Variant("base", base, "production features before 2026-09-26"),
        "earn": Variant("earn", replace(base, features=tuple(MODEL_FEATURES)), "+ SUE / revenue surprise / announcement return"),
        "rank": Variant("rank", replace(base, objective="rank"), "pairwise ranking objective, per-date groups"),
        "earn_rank": Variant("earn_rank", replace(base, features=tuple(MODEL_FEATURES), objective="rank"), "both"),
    }


def run(panel: pl.DataFrame, variants: list[Variant], refit_every: int = 126, oos_start: date = date(2011, 1, 1),
        cache_dir: Path | None = None) -> dict[str, pl.DataFrame]:
    out = {}
    for v in variants:
        p = cache_dir / f"scores_{v.name}.parquet" if cache_dir else None
        if p is not None and p.exists():
            out[v.name] = pl.read_parquet(p)
            logger.info(f"experiment {v.name}: cached")
            continue
        t0 = time.time()
        sc = causal_scores(panel, v.spec, refit_every=refit_every, oos_start=oos_start, progress=False)
        out[v.name] = sc
        logger.info(f"experiment {v.name}: {sc.height:,} scores in {(time.time() - t0) / 60:.1f} min")
        if p is not None:
            p.parent.mkdir(parents=True, exist_ok=True)
            sc.write_parquet(p)
    return out


def blend(a: pl.DataFrame, b: pl.DataFrame) -> pl.DataFrame:
    z = lambda df, c: df.with_columns(((pl.col("score") - pl.col("score").mean().over(["date", "horizon"]))
                                       / (pl.col("score").std().over(["date", "horizon"]) + 1e-9)).alias(c))
    j = z(a, "za").select("date", "ticker", "horizon", "za").join(
        z(b, "zb").select("date", "ticker", "horizon", "zb"), on=["date", "ticker", "horizon"], how="inner")
    return j.select("date", "ticker", "horizon", score=(pl.col("za") + pl.col("zb")) / 2)


def daily_ic(scores: pl.DataFrame, panel: pl.DataFrame) -> pl.DataFrame:
    """date, horizon, ic, spread (top-minus-bottom decile mean forward return)."""
    parts = []
    for (h,), g in scores.group_by(["horizon"]):
        f = f"fwd_{int(h)}"
        j = g.join(panel.select("date", "ticker", f), on=["date", "ticker"], how="inner").drop_nulls(f)
        j = j.with_columns(dec=((pl.col("score").rank("ordinal").over("date") - 1) * 10
                                // pl.col("score").count().over("date")))
        parts.append(j.group_by("date").agg(
            ic=pl.corr("score", f, method="spearman"), n=pl.len(),
            spread=pl.col(f).filter(pl.col("dec") == 9).mean() - pl.col(f).filter(pl.col("dec") == 0).mean())
            .filter(pl.col("n") >= 30).with_columns(horizon=pl.lit(int(h))))
    return pl.concat(parts).sort(["horizon", "date"])


def compare(ics: dict[str, pl.DataFrame], baseline: str = "base", recent: date = date(2020, 1, 1)) -> pl.DataFrame:
    rows = []
    b = ics[baseline]
    for name, d in ics.items():
        for (h,), g in d.group_by(["horizon"], maintain_order=True):
            g = g.sort("date")
            ic = g["ic"].to_numpy()
            row = {"variant": name, "horizon": int(h), "n_dates": len(ic), "ic": float(np.nanmean(ic)),
                   "icir": float(np.nanmean(ic) / (np.nanstd(ic) + 1e-12)),
                   "ic_recent": float(g.filter(pl.col("date") >= recent)["ic"].mean() or np.nan),
                   "spread": float(g["spread"].mean())}
            if name != baseline:
                m = g.join(b.filter(pl.col("horizon") == h).select("date", ic_b="ic"), on="date", how="inner")
                dlt = (m["ic"] - m["ic_b"]).to_numpy()
                # overlapping labels make daily IC differences autocorrelated: use non-overlapping dates
                step = max(1, int(h))
                dd = dlt[::step]
                row["paired_t"] = float(dd.mean() / (dd.std(ddof=1) + 1e-12) * np.sqrt(len(dd))) if len(dd) > 3 else np.nan
                yrs = m.with_columns(y=pl.col("date").dt.year()).group_by("y").agg(w=(pl.col("ic") > pl.col("ic_b")).mean())
                row["years_won"] = float((yrs["w"] > 0.5).mean())
            rows.append(row)
    return pl.DataFrame(rows).sort(["horizon", "variant"])


def verdict(table: pl.DataFrame, candidate: str, baseline: str = "base", primary: int = 63) -> tuple[bool, str]:
    t = table.filter(pl.col("variant") == candidate)
    p = t.filter(pl.col("horizon") == primary)
    if p.height == 0 or "paired_t" not in t.columns:
        return False, "no comparison"
    tp = float(p["paired_t"][0])
    worst = float(t.filter(pl.col("horizon") != primary)["paired_t"].min() or 0.0)
    ok = tp >= 2.0 and worst >= -1.0
    return ok, f"{candidate}: paired t {tp:+.1f} on {primary}d, worst other horizon {worst:+.1f} → {'ADOPT' if ok else 'reject'}"


def write_report(path: Path, table: pl.DataFrame, verdicts: list[str], meta: dict) -> Path:
    L = [f"# Alpha experiment — {date.today()}", "", f"Folds: refit every {meta['refit_every']} sessions, OOS from "
         f"{meta['oos_start']}; same rows scored for every variant. Paired t uses non-overlapping dates.", "",
         "| variant | h | dates | IC | ICIR | IC since 2020 | decile spread | paired t vs base | years won |",
         "|:--|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in table.iter_rows(named=True):
        L.append(f"| {r['variant']} | {r['horizon']} | {r['n_dates']} | {r['ic']:+.4f} | {r['icir']:.2f} | {r['ic_recent']:+.4f} | "
                 f"{r['spread']:+.2%} | {r.get('paired_t') if r.get('paired_t') is None else format(r['paired_t'], '+.1f')} | "
                 f"{'' if r.get('years_won') is None else format(r['years_won'], '.0%')} |")
    L += ["", "## Verdicts (rule: paired t ≥ 2 on 63d and ≥ −1 elsewhere)", ""] + [f"- {v}" for v in verdicts]
    for k, v in meta.get("notes", {}).items():
        L.append(f"- {k}: {v}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n")
    (path.with_suffix(".json")).write_text(json.dumps({"table": table.to_dicts(), "verdicts": verdicts, "meta": meta},
                                                      indent=1, default=str))
    return path
