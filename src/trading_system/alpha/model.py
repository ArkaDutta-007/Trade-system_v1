"""Gradient-boosted cross-sectional rank forecaster, walked forward causally.

* Features are rank-transformed **within each date** (``prepare``): the model
  sees "where does this name sit in today's cross-section", which is stable
  across 26 years of very different volatility/liquidity regimes.
* The label is the gaussianised per-date rank of the forward return
  (``panel.add_labels``), so the objective is plain squared error on a
  well-behaved target and the fitted function is a *ranker*.
* ``causal_scores`` is the honest history: at every refit only rows whose
  labels had fully matured (``date ≤ cut − h − embargo``) are used, and the
  model then scores every date until the next refit.  Its output is the
  backtest ledger — the same rows the live ledger keeps adding to.
* XGBoost on CUDA when available (an 8 GB card fits the whole panel), CPU
  ``hist`` otherwise; a couple of seeds are bagged.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl

from ..utils import get_logger
from .panel import FEATURE_COLS, HORIZONS

logger = get_logger(__name__)

RANK_SUFFIX = "_r"


@dataclass
class TrainSpec:
    horizons: tuple[int, ...] = HORIZONS
    n_rounds: int = 400
    max_depth: int = 6
    learning_rate: float = 0.03
    subsample: float = 0.7
    colsample_bytree: float = 0.6
    min_child_weight: float = 300.0
    reg_lambda: float = 10.0
    seeds: tuple[int, ...] = (0, 1)
    stride: dict = field(default_factory=lambda: {5: 2, 21: 4, 63: 8})   # training-date sampling per horizon
    embargo_days: int = 5
    half_life_days: float = 252 * 10       # time-decay of sample weights (0 = flat)
    max_train_rows: int = 2_500_000
    device: str = "auto"                   # auto | cuda | cpu
    features: tuple[str, ...] = tuple(FEATURE_COLS)

    def stride_for(self, h: int) -> int:
        return int(self.stride.get(h, max(1, h // 4)))


def resolve_device(pref: str = "auto") -> str:
    if pref in ("cpu", "cuda"):
        return pref
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# ── feature preparation ───────────────────────────────────────────────────────

def prepare(panel: pl.DataFrame, features: Iterable[str] = FEATURE_COLS) -> tuple[pl.DataFrame, list[str]]:
    """Add per-date rank columns ``<f>_r`` ∈ (0, 1) for every feature (nulls stay null) and a
    trading-day index ``didx``. Returns (frame, rank column names)."""
    feats = [f for f in features if f in panel.columns]
    dates = panel.select("date").unique().sort("date").with_row_index("didx")
    out = panel.join(dates, on="date", how="left")
    out = out.with_columns([((pl.col(f).rank(method="average").over("date") - 0.5) / pl.col(f).count().over("date"))
                            .cast(pl.Float32).alias(f + RANK_SUFFIX) for f in feats])
    return out, [f + RANK_SUFFIX for f in feats]


def _xy(frame: pl.DataFrame, cols: list[str], h: int | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    X = frame.select(cols).to_numpy().astype(np.float32, copy=False)
    y = frame[f"y_{h}"].to_numpy().astype(np.float32) if h is not None else None
    return X, y


# ── the estimator ─────────────────────────────────────────────────────────────

class AlphaGBM:
    """Bag of XGBoost regressors (one per seed) on the gaussian-rank label."""

    def __init__(self, spec: TrainSpec, horizon: int, feature_names: list[str], device: str | None = None):
        self.spec, self.horizon, self.feature_names = spec, horizon, list(feature_names)
        self.device = device or resolve_device(spec.device)
        self.boosters: list = []
        self.trained_through: str | None = None
        self.n_rows = 0

    def _params(self, seed: int) -> dict:
        s = self.spec
        p = {"objective": "reg:squarederror", "max_depth": s.max_depth, "eta": s.learning_rate,
             "subsample": s.subsample, "colsample_bytree": s.colsample_bytree,
             "min_child_weight": s.min_child_weight, "lambda": s.reg_lambda, "tree_method": "hist",
             "device": self.device, "seed": seed, "max_bin": 128, "verbosity": 0}
        if self.device == "cpu":
            import os
            p["nthread"] = max(2, (os.cpu_count() or 4) - 2)
        return p

    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> "AlphaGBM":
        import xgboost as xgb
        self.boosters = []
        dm = xgb.QuantileDMatrix(X, label=y, weight=w, feature_names=self.feature_names, max_bin=128)
        for seed in self.spec.seeds:
            try:
                b = xgb.train(self._params(seed), dm, num_boost_round=self.spec.n_rounds)
            except xgb.core.XGBoostError as e:
                if self.device != "cpu":
                    logger.warning(f"alpha: CUDA training failed ({str(e)[:80]}) — falling back to CPU")
                    self.device = "cpu"
                    b = xgb.train(self._params(seed), dm, num_boost_round=self.spec.n_rounds)
                else:
                    raise
            self.boosters.append(b)
        self.n_rows = int(X.shape[0])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import xgboost as xgb
        if not self.boosters:
            raise RuntimeError("model not fitted")
        dm = xgb.DMatrix(X, feature_names=self.feature_names)
        return np.mean([b.predict(dm) for b in self.boosters], axis=0).astype(np.float32)

    def contributions(self, X: np.ndarray) -> np.ndarray:
        """SHAP-style per-feature contributions (n, n_features + bias), averaged over the bag."""
        import xgboost as xgb
        dm = xgb.DMatrix(X, feature_names=self.feature_names)
        return np.mean([b.predict(dm, pred_contribs=True) for b in self.boosters], axis=0)

    def importance(self) -> dict[str, float]:
        agg: dict[str, float] = {}
        for b in self.boosters:
            for k, v in b.get_score(importance_type="gain").items():
                agg[k] = agg.get(k, 0.0) + float(v) / len(self.boosters)
        return dict(sorted(agg.items(), key=lambda kv: -kv[1]))

    # persistence
    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        for i, b in enumerate(self.boosters):
            b.save_model(str(d / f"booster_{i}.json"))
        meta = {"horizon": self.horizon, "feature_names": self.feature_names, "trained_through": self.trained_through,
                "n_rows": self.n_rows, "device": self.device, "spec": _spec_dict(self.spec),
                "n_boosters": len(self.boosters), "importance": self.importance(), "saved_at": time.time()}
        (d / "meta.json").write_text(json.dumps(meta, indent=1))

    @classmethod
    def load(cls, d: Path) -> "AlphaGBM":
        import xgboost as xgb
        meta = json.loads((d / "meta.json").read_text())
        spec = TrainSpec(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in meta["spec"].items()})
        m = cls(spec, meta["horizon"], meta["feature_names"], device=meta.get("device"))
        for i in range(meta["n_boosters"]):
            b = xgb.Booster()
            b.load_model(str(d / f"booster_{i}.json"))
            m.boosters.append(b)
        m.trained_through, m.n_rows = meta.get("trained_through"), meta.get("n_rows", 0)
        return m


def _spec_dict(spec: TrainSpec) -> dict:
    d = asdict(spec)
    d["stride"] = {str(k): v for k, v in d["stride"].items()}
    return d


# ── training slices ───────────────────────────────────────────────────────────

def training_rows(frame: pl.DataFrame, h: int, cut_idx: int, spec: TrainSpec) -> pl.DataFrame:
    """Rows whose ``h``-day label matured on or before trading-day ``cut_idx`` (purged + embargoed),
    thinned to every ``stride``-th date so overlapping labels don't masquerade as independent samples."""
    last = cut_idx - h - spec.embargo_days
    stride = spec.stride_for(h)
    rows = frame.filter((pl.col("didx") <= last) & pl.col(f"y_{h}").is_not_null()
                        & ((pl.col("didx") % stride) == (last % stride)))
    if rows.height > spec.max_train_rows:
        rows = rows.sample(n=spec.max_train_rows, seed=cut_idx)
    return rows


def sample_weights(rows: pl.DataFrame, cut_idx: int, spec: TrainSpec) -> np.ndarray | None:
    if not spec.half_life_days:
        return None
    age = (cut_idx - rows["didx"].to_numpy()).astype(np.float64)
    return (0.5 ** (age / spec.half_life_days)).astype(np.float32)


# ── causal walk-forward scoring ───────────────────────────────────────────────

def causal_scores(panel: pl.DataFrame, spec: TrainSpec | None = None, refit_every: int = 63,
                  min_train_days: int = 1260, oos_start: date | None = None, oos_end: date | None = None,
                  progress: bool = True, on_refit=None) -> pl.DataFrame:
    """Walk forward: refit every ``refit_every`` trading days on matured labels, score until the next refit.

    Returns the backtest ledger: ``date, ticker, horizon, score, model_id`` for every scored row.
    """
    spec = spec or TrainSpec()
    frame, rcols = prepare(panel, spec.features)
    didx = frame.select("date", "didx").unique().sort("didx")
    dates = didx["date"].to_list()
    n_dates = len(dates)
    first_oos = max(min_train_days, next((i for i, d in enumerate(dates) if oos_start and d >= oos_start), min_train_days))
    last_oos = n_dates - 1 if oos_end is None else max(i for i, d in enumerate(dates) if d <= oos_end)
    cuts = list(range(first_oos, last_oos + 1, refit_every))
    out: list[pl.DataFrame] = []
    device = resolve_device(spec.device)
    t_all = time.time()
    for k, cut in enumerate(cuts):
        nxt = cuts[k + 1] if k + 1 < len(cuts) else last_oos + 1
        score_rows = frame.filter((pl.col("didx") >= cut) & (pl.col("didx") < nxt))
        if score_rows.height == 0:
            continue
        Xs, _ = _xy(score_rows, rcols)
        model_id = f"wf_{dates[cut]:%Y%m%d}"
        cols = {"date": score_rows["date"], "ticker": score_rows["ticker"]}
        t0 = time.time()
        for h in spec.horizons:
            tr = training_rows(frame, h, cut, spec)
            if tr.height < 5000:
                continue
            X, y = _xy(tr, rcols, h)
            m = AlphaGBM(spec, h, rcols, device=device).fit(X, y, sample_weights(tr, cut, spec))
            device = m.device
            cols[f"s_{h}"] = pl.Series(m.predict(Xs))
            if on_refit:
                on_refit(h, cut, m, tr.height)
        if len(cols) <= 2:
            continue
        df = pl.DataFrame(cols).with_columns(model_id=pl.lit(model_id))
        df = df.unpivot(index=["date", "ticker", "model_id"], on=[c for c in cols if c.startswith("s_")],
                        variable_name="horizon", value_name="score")
        df = df.with_columns(pl.col("horizon").str.slice(2).cast(pl.Int32))
        out.append(df)
        if progress:
            logger.info(f"alpha wf {k + 1}/{len(cuts)} · cut {dates[cut]} → {dates[nxt - 1] if nxt - 1 < n_dates else dates[-1]}"
                        f" · {score_rows.height:,} rows scored · {time.time() - t0:.0f}s ({device})")
    if not out:
        raise RuntimeError("causal_scores: nothing scored (panel too short?)")
    res = pl.concat(out).sort(["date", "horizon", "ticker"])
    logger.info(f"alpha wf done: {res.height:,} forecasts · {len(cuts)} refits · {(time.time() - t_all) / 60:.1f} min")
    return res


# ── production fit / predict ──────────────────────────────────────────────────

def models_dir(cfg) -> Path:
    return cfg.project_root / "data" / "models" / "alpha"


def fit_production(panel: pl.DataFrame, spec: TrainSpec | None = None, out_dir: Path | None = None,
                   as_of: date | None = None) -> dict[int, AlphaGBM]:
    """Fit one model per horizon on everything whose label has matured by ``as_of`` (default: last date)."""
    spec = spec or TrainSpec()
    frame, rcols = prepare(panel, spec.features)
    dates = frame.select("date", "didx").unique().sort("didx")
    cut = int(dates["didx"][-1]) if as_of is None else int(dates.filter(pl.col("date") <= as_of)["didx"][-1])
    models: dict[int, AlphaGBM] = {}
    device = resolve_device(spec.device)
    for h in spec.horizons:
        tr = training_rows(frame, h, cut, spec)
        X, y = _xy(tr, rcols, h)
        t0 = time.time()
        m = AlphaGBM(spec, h, rcols, device=device).fit(X, y, sample_weights(tr, cut, spec))
        device = m.device
        m.trained_through = str(tr["date"].max())
        models[h] = m
        logger.info(f"alpha fit h={h}: {tr.height:,} rows through {m.trained_through} · {time.time() - t0:.0f}s ({m.device})")
        if out_dir is not None:
            m.save(out_dir / f"{h}d")
    if out_dir is not None:
        (out_dir / "manifest.json").write_text(json.dumps(
            {"horizons": list(models), "as_of": str(dates["date"][cut]), "features": rcols, "spec": _spec_dict(spec),
             "fitted_at": time.time()}, indent=1))
    return models


def load_production(d: Path) -> dict[int, AlphaGBM]:
    if not (d / "manifest.json").exists():
        raise FileNotFoundError(f"no alpha models in {d} — run `ts alpha train`")
    man = json.loads((d / "manifest.json").read_text())
    return {int(h): AlphaGBM.load(d / f"{h}d") for h in man["horizons"]}


def predict_dates(panel: pl.DataFrame, models: dict[int, AlphaGBM], dates: Iterable[date] | None = None) -> pl.DataFrame:
    """Scores for the given dates (default: the panel's last date) → ``date, ticker, horizon, score, model_id``."""
    frame, rcols = prepare(panel, next(iter(models.values())).spec.features)
    if dates is None:
        dates = [frame["date"].max()]
    sub = frame.filter(pl.col("date").is_in(list(dates)))
    X, _ = _xy(sub, rcols)
    cols = {"date": sub["date"], "ticker": sub["ticker"]}
    for h, m in models.items():
        cols[f"s_{h}"] = pl.Series(m.predict(X))
    df = pl.DataFrame(cols).unpivot(index=["date", "ticker"], on=[f"s_{h}" for h in models],
                                    variable_name="horizon", value_name="score")
    mid = "prod_" + (next(iter(models.values())).trained_through or "unknown")
    return df.with_columns(pl.col("horizon").str.slice(2).cast(pl.Int32), model_id=pl.lit(mid))


def explain_latest(panel: pl.DataFrame, models: dict[int, AlphaGBM], tickers: Iterable[str], horizon: int,
                   top: int = 5) -> dict[str, list[tuple[str, float]]]:
    """Top contributing (feature, contribution) pairs per ticker on the last date."""
    m = models[horizon]
    frame, rcols = prepare(panel, m.spec.features)
    sub = frame.filter(pl.col("date") == frame["date"].max(), pl.col("ticker").is_in(list(tickers)))
    if sub.height == 0:
        return {}
    X, _ = _xy(sub, rcols)
    C = m.contributions(X)[:, :-1]
    out = {}
    for i, t in enumerate(sub["ticker"].to_list()):
        order = np.argsort(-np.abs(C[i]))[:top]
        out[t] = [(rcols[j].removesuffix(RANK_SUFFIX), float(C[i, j])) for j in order]
    return out
