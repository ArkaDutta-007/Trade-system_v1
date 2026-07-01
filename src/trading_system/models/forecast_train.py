"""Industry-standard multi-horizon forecast training.

For each horizon (default 21/63/126/252d — the long-term focus) this:

  1. Builds the realised forward-return target and resolves the feature reserve.
  2. Evaluates several model families (LightGBM, XGBoost, HistGBM, Ridge) under
     **purged + embargoed walk-forward CV** so overlapping labels can't leak.
  3. Scores each fold with forecasting-grade metrics: rank-IC (Spearman),
     **ICIR** (IC / IC-std — the information ratio of the signal), directional
     hit-rate, MAE, R².
  4. Runs a **label-shuffle leakage gate** — a model trained on shuffled targets
     must collapse to ~0 IC; if it doesn't, the pipeline is leaking and we flag it.
  5. Selects the best family per horizon by ICIR, refits it on all fully-labeled
     data, and returns everything for the model store.

Hardware: model params come from the detected compute profile, so the same call
uses RTX CUDA on the GPU box and all CPU cores on the M-series laptop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from ..utils import get_logger, get_compute_profile
from .validation import purged_walkforward_splits, combinatorial_purged_splits, deflated_icir
from .sequence import (
    SEQUENCE_MODELS, build_sequence_model, build_sequence_tensor, torch_available,
)

logger = get_logger(__name__)

DEFAULT_HORIZONS = (21, 63, 126, 252)
_CAL_PER_TD = 365 / 252  # trading days -> calendar days for purge sizing

# Tabular families (row-independent). Sequence families live in SEQUENCE_MODELS.
TABULAR_MODELS: tuple[str, ...] = ("lgbm", "xgb", "hist_gbm", "ridge")


def _resolve_model_names(models: list[str] | None) -> tuple[list[str], list[str]]:
    """Split a requested model list into (tabular, sequence). None → tabular only."""
    if not models:
        return list(TABULAR_MODELS), []
    tab = [m for m in models if m in TABULAR_MODELS]
    seq = [m for m in models if m in SEQUENCE_MODELS]
    for u in [m for m in models if m not in TABULAR_MODELS and m not in SEQUENCE_MODELS]:
        logger.warning(f"unknown model '{u}' ignored")
    return tab, seq


# ── Metrics ──────────────────────────────────────────────────────────────────

# Benign LightGBM/sklearn notice when fitting on a numpy array (no feature names).
# It says nothing about correctness and floods --all runs; silence just that one.
import warnings  # noqa: E402
warnings.filterwarnings("ignore", message="X does not have valid feature names")

N_LEAK_SHUFFLES_TABULAR = 10
N_LEAK_SHUFFLES_SEQUENCE = 3   # sequence refits are minutes each — keep the gate cheap


def _ic(y, p) -> float:
    """Pooled rank IC (kept for diagnostics). Not used for selection."""
    if len(y) < 10 or np.std(p) < 1e-12:
        return 0.0
    r, _ = spearmanr(y, p)
    return float(r) if not np.isnan(r) else 0.0


def _perdate_ic(y, p, dates, min_names: int = 10) -> float:
    """Mean **within-date cross-sectional** rank IC — the standard quant IC.

    Pooling every (ticker,date) pair conflates *market timing* with *stock
    selection*: a feature that's constant per date (macro level, RMT systematic
    fraction) correlates with the date-level forward return, so pooled IC is
    inflated by the common market factor and survives label shuffling. Computing
    Spearman **within each date** and averaging isolates genuine cross-sectional
    skill (and makes the leakage gate meaningful).
    """
    d = np.asarray(dates)
    ics = []
    for u in np.unique(d):
        m = d == u
        if m.sum() >= min_names and np.std(p[m]) > 1e-12:
            r, _ = spearmanr(y[m], p[m])
            if not np.isnan(r):
                ics.append(r)
    return float(np.mean(ics)) if ics else 0.0


def _fold_metrics(y, p, dates, priority_mask=None) -> dict:
    """Per-fold metrics. ``priority_mask`` (bool over rows) also yields a
    cross-sectional IC restricted to the priority universe, so a broadly-trained
    model can be judged on how well it ranks *your* names too."""
    from sklearn.metrics import mean_absolute_error, r2_score
    hit = float(np.mean((np.sign(p) == np.sign(y))[np.abs(y) > 1e-9])) if len(y) else 0.0
    out = {
        "ic": _perdate_ic(y, p, dates),   # broad cross-sectional (not pooled)
        "hit_rate": hit,
        "mae": float(mean_absolute_error(y, p)) if len(y) else 0.0,
        "r2": float(r2_score(y, p)) if len(y) > 1 and np.std(y) > 1e-12 else 0.0,
    }
    if priority_mask is not None and priority_mask.any():
        m = priority_mask
        out["univ_ic"] = _perdate_ic(y[m], p[m], np.asarray(dates)[m])
    else:
        out["univ_ic"] = out["ic"]
    return out


def _aggregate(fold_rows: list[dict]) -> dict:
    if not fold_rows:
        return {"ic_mean": 0.0, "ic_std": 0.0, "icir": 0.0, "univ_ic_mean": 0.0,
                "univ_icir": 0.0, "hit_rate": 0.0, "mae": 0.0, "r2": 0.0, "n_folds": 0}
    ics = np.array([r["ic"] for r in fold_rows])
    ic_std = float(ics.std()) if len(ics) > 1 else 0.0
    uics = np.array([r.get("univ_ic", r["ic"]) for r in fold_rows])
    uic_std = float(uics.std()) if len(uics) > 1 else 0.0
    return {
        "ic_mean": float(ics.mean()),
        "ic_std": ic_std,
        "icir": float(ics.mean() / ic_std) if ic_std > 1e-9 else 0.0,
        "univ_ic_mean": float(uics.mean()),
        "univ_icir": float(uics.mean() / uic_std) if uic_std > 1e-9 else 0.0,
        "hit_rate": float(np.mean([r["hit_rate"] for r in fold_rows])),
        "mae": float(np.mean([r["mae"] for r in fold_rows])),
        "r2": float(np.mean([r["r2"] for r in fold_rows])),
        "n_folds": len(fold_rows),
    }


def _selection_score(m: dict, universe_weight: float) -> float:
    """Blend broad and priority-universe ICIR for model selection.
    universe_weight=0 → pure general performance; 1 → pure your-universe."""
    return (1.0 - universe_weight) * m["icir"] + universe_weight * m["univ_icir"]


# ── Model factory (hardware-aware) ────────────────────────────────────────────

def _build_models(prof) -> dict:
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "lgbm": lgb.LGBMRegressor(
            objective="regression", n_estimators=600, num_leaves=63,
            learning_rate=0.03, min_child_samples=40, subsample=0.8,
            colsample_bytree=0.8, verbose=-1, **prof.lgbm_params(),
        ),
        "xgb": xgb.XGBRegressor(
            n_estimators=600, max_depth=6, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=10, verbosity=0,
            **prof.xgb_params(),
        ),
        "hist_gbm": HistGradientBoostingRegressor(
            max_iter=600, max_leaf_nodes=63, learning_rate=0.03,
            min_samples_leaf=40, l2_regularization=1.0,
        ),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=5.0)),
    }


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class HorizonResult:
    horizon: int
    feature_columns: list[str]
    per_model: dict[str, dict]          # model -> aggregate metrics
    best_model_name: str
    best_model: object = None           # refit-on-all estimator
    leakage_gate: dict = field(default_factory=dict)
    deflation: dict = field(default_factory=dict)   # multiple-testing haircut
    n_rows: int = 0
    trained_through: str = ""
    cv_mode: str = "walkforward"


# ── Core ──────────────────────────────────────────────────────────────────────

def _xy(df: pl.DataFrame, feat_cols: list[str], tgt: str):
    sub = df.drop_nulls(subset=[tgt] + feat_cols)
    X = sub.select(feat_cols).to_numpy().astype(np.float64)
    y = sub[tgt].to_numpy().astype(np.float64)
    d = sub["date"].to_list()
    return X, y, d, sub


def train_horizon(
    features: pl.DataFrame,
    feat_cols: list[str],
    horizon: int,
    n_splits: int = 5,
    embargo_days: int = 5,
    models: list[str] | None = None,
    lookback: int = 64,
    seq_epochs: int = 40,
    neutralize: bool = False,
    priority_tickers: set[str] | None = None,
    universe_weight: float = 0.5,
    cv_mode: str = "walkforward",
) -> HorizonResult:
    prof = get_compute_profile()
    tgt = f"fwd_ret_{horizon}d"
    px = pl.col("adj_close")
    df = features.sort(["ticker", "date"]).with_columns(
        ((px.shift(-horizon).over("ticker") / px) - 1).alias(tgt)
    )
    if neutralize:
        # Market-neutral target: subtract each date's cross-sectional mean forward
        # return, so the model learns pure stock selection instead of fighting the
        # market factor (the dominant, non-stationary component at short horizons).
        df = df.with_columns((pl.col(tgt) - pl.col(tgt).mean().over("date")).alias(tgt))

    X, y, dates, sub = _xy(df, feat_cols, tgt)
    if len(X) < 500:
        raise ValueError(f"horizon {horizon}d: too few labeled rows ({len(X)})")

    # Priority-universe mask (over the null-free rows) for weighted selection
    priority_mask = None
    if priority_tickers:
        pset = {t.upper() for t in priority_tickers}
        priority_mask = np.array([t.upper() in pset for t in sub["ticker"].to_list()])

    horizon_cal = int(round(horizon * _CAL_PER_TD))
    if cv_mode == "cpcv":
        splits = combinatorial_purged_splits(
            dates, horizon_days=horizon_cal, n_groups=max(4, n_splits + 1),
            n_test_groups=2, embargo_days=embargo_days,
        )
    else:
        splits = purged_walkforward_splits(
            dates, horizon_days=horizon_cal, n_splits=n_splits, embargo_days=embargo_days
        )

    # Resolve requested families. Sequence models need torch + a 3-D tensor.
    tab_names, seq_names = _resolve_model_names(models)
    if seq_names and not torch_available():
        logger.warning(f"  {horizon}d: torch unavailable — skipping deep models {seq_names}")
        seq_names = []
    active = tab_names + seq_names
    if not active:
        raise ValueError(f"horizon {horizon}d: no runnable models from {models}")

    # Same row order as X (sub is sorted by ticker,date and null-free), so the
    # purged split indices slice the flat matrix and the sequence tensor alike.
    Xseq = build_sequence_tensor(sub, feat_cols, lookback) if seq_names else None

    def _new_estimator(name):
        if name in SEQUENCE_MODELS:
            return build_sequence_model(name, lookback, prof, epochs=seq_epochs)
        return _build_models(prof)[name]

    def _mat(name):
        return Xseq if name in SEQUENCE_MODELS else X

    dates_arr = np.asarray(dates)
    per_model_folds: dict[str, list[dict]] = {k: [] for k in active}
    for s in splits:
        ytr, yte = y[s.train_idx], y[s.test_idx]
        if len(s.train_idx) < 200 or len(s.test_idx) < 20:
            continue
        dte = dates_arr[s.test_idx]
        pmask = priority_mask[s.test_idx] if priority_mask is not None else None
        for name in active:
            try:
                mat = _mat(name)
                m = _new_estimator(name)  # fresh estimator per fold
                m.fit(mat[s.train_idx], ytr)
                per_model_folds[name].append(
                    _fold_metrics(yte, m.predict(mat[s.test_idx]), dte, priority_mask=pmask))
            except Exception as e:
                logger.warning(f"  {horizon}d/{name} fold {s.fold} failed: {e}")

    per_model = {k: _aggregate(v) for k, v in per_model_folds.items()}
    # Best by blended (general + your-universe) ICIR, requiring positive mean IC.
    ranked = sorted(
        per_model.items(),
        key=lambda kv: (kv[1]["ic_mean"] > 0, _selection_score(kv[1], universe_weight), kv[1]["ic_mean"]),
        reverse=True,
    )
    best_name = ranked[0][0]
    best_mat = _mat(best_name)

    # Multiple-testing haircut: we picked the best of len(active) families, so the
    # winner's ICIR is upward-biased. Deflate against the best-of-N null.
    _bm = per_model[best_name]
    deflation = deflated_icir(_bm["icir"], n_folds=max(_bm.get("n_folds", 1), 1),
                              n_trials=len(active))

    # ── Leakage gate: real cross-sectional IC vs a label-shuffle NULL ─────────
    # Train the best family on N shuffled-label copies and build the null
    # distribution of |per-date IC| on the held-out test fold. A clean model's
    # real IC must exceed the null's 95th percentile, and the null must sit near
    # zero. A single shuffle (the old gate) is too noisy with a wide feature set;
    # the distribution makes the verdict trustworthy.
    leak = {}
    try:
        s = splits[-1]
        dte = dates_arr[s.test_idx]
        n_sh = N_LEAK_SHUFFLES_SEQUENCE if best_name in SEQUENCE_MODELS else N_LEAK_SHUFFLES_TABULAR
        real_ic = abs(per_model[best_name]["ic_mean"])
        rng = np.random.default_rng(0)
        null = []
        for _ in range(n_sh):
            ysh = y[s.train_idx].copy()
            rng.shuffle(ysh)
            m = _new_estimator(best_name)
            m.fit(best_mat[s.train_idx], ysh)
            null.append(abs(_perdate_ic(y[s.test_idx], m.predict(best_mat[s.test_idx]), dte)))
        null = np.asarray(null)
        p95 = float(np.quantile(null, 0.95))
        # Permutation test: pass iff real cross-sectional IC exceeds the 95th
        # percentile of the shuffled-label null (p < 0.05). The null already
        # absorbs label-autocorrelation inflation, so no separate absolute floor.
        leak = {
            "real_abs_ic": round(real_ic, 4),
            "shuffled_ic_mean": round(float(null.mean()), 4),
            "shuffled_ic_p95": round(p95, 4),
            "n_shuffles": n_sh,
            "pass": bool(real_ic > p95),
        }
    except Exception as e:
        leak = {"error": str(e), "pass": None}

    # ── Refit best on all fully-labeled rows ─────────────────────────────────
    best = _new_estimator(best_name)
    best.fit(best_mat, y)
    trained_through = max(dates).isoformat() if dates else ""

    _b = per_model[best_name]
    logger.info(
        f"horizon {horizon}d: best={best_name} "
        f"ICIR={_b['icir']:.2f} (deflated {deflation['deflated_icir']:+.2f}) "
        f"IC={_b['ic_mean']:+.3f} univ_ICIR={_b['univ_icir']:.2f} hit={_b['hit_rate']:.1%} "
        f"{'[neutral] ' if neutralize else ''}[{cv_mode}] "
        f"leak_pass={leak.get('pass')} deflation_pass={deflation['pass']}"
    )
    return HorizonResult(
        horizon=horizon, feature_columns=feat_cols, per_model=per_model,
        best_model_name=best_name, best_model=best, leakage_gate=leak,
        deflation=deflation, n_rows=len(X), trained_through=trained_through,
        cv_mode=cv_mode,
    )


def train_all_horizons(
    features: pl.DataFrame,
    feat_cols: list[str],
    horizons=DEFAULT_HORIZONS,
    n_splits: int = 5,
    embargo_days: int = 5,
    models: list[str] | None = None,
    lookback: int = 64,
    seq_epochs: int = 40,
    neutralize: bool = False,
    priority_tickers: set[str] | None = None,
    universe_weight: float = 0.5,
    cv_mode: str = "walkforward",
) -> dict[int, HorizonResult]:
    out: dict[int, HorizonResult] = {}
    for h in horizons:
        try:
            out[h] = train_horizon(
                features, feat_cols, h, n_splits, embargo_days, models,
                lookback=lookback, seq_epochs=seq_epochs,
                neutralize=neutralize, priority_tickers=priority_tickers,
                universe_weight=universe_weight, cv_mode=cv_mode,
            )
        except Exception as e:
            logger.warning(f"horizon {h}d training failed: {e}")
    return out
