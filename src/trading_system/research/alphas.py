"""Alpha models for the walk-forward simulator.

Each class satisfies :class:`~trading_system.research.wfbacktest.AlphaModel`:
``fit`` sees only past rows, ``predict`` scores one cross-section.  Model
*selection* happens inside ``fit`` on an inner purged split of the training
window, never on the evaluation sample — the loop in ``wfbacktest`` then gets a
single number per name with no knowledge of the future at all.

The baselines are not filler.  A signal that cannot beat 12-1 momentum after
costs is not worth deploying, and a random-score book is the null that tells you
whether the *book construction* (risk-adjusted ranking, caps, bands) is doing
the work rather than the model.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from ..utils import get_compute_profile, get_logger

logger = get_logger(__name__)

__all__ = [
    "MomentumAlpha", "RandomAlpha", "GBMAlpha", "EnsembleAlpha",
    "make_tabular_models", "inner_purged_ic",
]


def make_tabular_models(prof=None, seed: int = 0) -> dict:
    """The tree/linear families, hardware-aware, with a settable seed.

    On this box the family choice is dominated by where the compute is.  Fitting
    600 trees on a 1.14M x 68 panel measured at **5.8 s for XGBoost on an
    A6000** against 69 s for the same model on 8 CPU threads and 82 s for
    LightGBM — a 14x gap.  A 21-year walk-forward with annual refits and an
    inner selection split is a couple of minutes on the GPU and most of a day on
    contended cores, so ``xgb`` is the workhorse for research sweeps and the
    CPU families are there for the ensemble diversity, not for speed.
    """
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    prof = prof or get_compute_profile()
    return {
        "lgbm": lgb.LGBMRegressor(
            objective="regression", n_estimators=600, num_leaves=63,
            learning_rate=0.03, min_child_samples=40, subsample=0.8,
            colsample_bytree=0.8, verbose=-1, random_state=seed, **prof.lgbm_params()),
        "xgb": xgb.XGBRegressor(
            n_estimators=600, max_depth=6, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=10, verbosity=0,
            random_state=seed, **prof.xgb_params()),
        "hist_gbm": HistGradientBoostingRegressor(
            max_iter=600, max_leaf_nodes=63, learning_rate=0.03,
            min_samples_leaf=40, l2_regularization=1.0, random_state=seed),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=5.0)),
    }


def per_date_ic(y: np.ndarray, p: np.ndarray, dates: np.ndarray,
                min_names: int = 10) -> np.ndarray:
    """Within-date rank IC, one value per date.

    Pooling across dates conflates market timing with stock selection: any
    feature that is constant within a date (a macro level, the RMT systematic
    fraction) correlates with that date's common return component and inflates
    a pooled IC.  Only the within-date version measures the thing a
    cross-sectional book actually monetises.
    """
    out = []
    for u in np.unique(dates):
        m = dates == u
        if m.sum() >= min_names and np.std(p[m]) > 1e-12:
            r, _ = spearmanr(y[m], p[m])
            if np.isfinite(r):
                out.append(r)
    return np.asarray(out)


def inner_purged_ic(
    train: pl.DataFrame, feat_cols: list[str], target: str,
    model, horizon: int, n_splits: int = 3, embargo_days: int = 5,
) -> float:
    """ICIR of ``model`` on an inner purged walk-forward *inside* the train set.

    This is how a family gets chosen without looking forward.  The training
    window is cut into ``n_splits`` forward blocks; each is scored by a fit on
    the earlier data, purged by the label horizon plus an embargo.
    """
    d = train["date"].to_numpy()
    uniq = np.unique(d)
    if len(uniq) < n_splits * 20:
        return 0.0
    X = train.select(feat_cols).to_numpy().astype(np.float64)
    y = train[target].to_numpy().astype(np.float64)

    pad = np.timedelta64(int(round(horizon * 365 / 252)) + embargo_days, "D")
    blocks = np.array_split(uniq[int(len(uniq) * 0.5):], n_splits)
    ics = []
    for blk in blocks:
        if len(blk) == 0:
            continue
        te = np.isin(d, blk)
        tr = d < (blk[0] - pad)
        if tr.sum() < 2000 or te.sum() < 200:
            continue
        try:
            import copy
            m = copy.deepcopy(model)
            m.fit(X[tr], y[tr])
            ics.extend(per_date_ic(y[te], np.asarray(m.predict(X[te])).ravel(), d[te]))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"inner split failed: {e}")
    if len(ics) < 3:
        return 0.0
    ics = np.asarray(ics)
    sd = ics.std()
    return float(ics.mean() / sd) if sd > 1e-9 else 0.0


# ── Baselines ────────────────────────────────────────────────────────────────

class MomentumAlpha:
    """12-1 month momentum (Jegadeesh-Titman): the benchmark to beat.

    No parameters, nothing fitted, nothing to overfit.  In this repo's own
    earlier backtests the rule-based momentum sleeve posted a higher Sharpe
    (1.21) than the ML ensemble (1.03), which is the reason it is the reference
    point rather than a footnote.
    """

    name = "momentum_12_1"

    def __init__(self, col: str = "mom_12m1m"):
        self.col = col

    def fit(self, panel, feat_cols, target) -> None:
        return None

    def predict(self, today: pl.DataFrame, feat_cols) -> np.ndarray:
        col = self.col if self.col in today.columns else "mom_120d"
        return today[col].fill_null(strategy="zero").to_numpy()


class RandomAlpha:
    """Uniform noise scores — the null that prices the book construction itself."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def fit(self, panel, feat_cols, target) -> None:
        return None

    def predict(self, today: pl.DataFrame, feat_cols) -> np.ndarray:
        return self.rng.standard_normal(today.height)


# ── Learned models ───────────────────────────────────────────────────────────

class GBMAlpha:
    """A single gradient-boosted family, refit on every call."""

    def __init__(self, family: str = "lgbm", seed: int = 0):
        self.family = family
        self.seed = seed
        self.name = family
        self.chosen = family
        self.model = None
        self._cols: list[str] = []

    def fit(self, panel: pl.DataFrame, feat_cols: list[str], target: str) -> None:
        self._cols = feat_cols
        X = panel.select(feat_cols).to_numpy().astype(np.float64)
        y = panel[target].to_numpy().astype(np.float64)
        self.model = make_tabular_models(seed=self.seed)[self.family]
        self.model.fit(X, y)

    def predict(self, today: pl.DataFrame, feat_cols: list[str]) -> np.ndarray:
        X = today.select(feat_cols).to_numpy().astype(np.float64)
        return np.asarray(self.model.predict(X)).ravel()


class EnsembleAlpha:
    """Inner-CV-weighted blend of tree families, seed-averaged.

    Two choices worth stating:

    *Blend, don't pick.*  Selecting one winner per refit makes the live signal
    jump between families whenever two of them are within noise of each other.
    Weighting by inner-CV ICIR (clipped at zero, renormalised) keeps the signal
    continuous and is the standard way to combine forecasts of differing
    quality.

    *Average seeds.*  A single GBM fit on a low-signal panel has real seed
    variance; averaging ``n_seeds`` fits cuts that without changing the
    expectation.  It is the cheapest variance reduction available and it uses
    cores that are otherwise idle.
    """

    name = "ensemble"

    def __init__(
        self,
        families: tuple[str, ...] = ("lgbm", "xgb", "hist_gbm"),
        horizon: int = 63,
        n_seeds: int = 1,
        inner_splits: int = 3,
        rank_normalise: bool = True,
    ):
        self.families = families
        self.horizon = horizon
        self.n_seeds = n_seeds
        self.inner_splits = inner_splits
        self.rank_normalise = rank_normalise
        self.fitted: dict[str, list] = {}
        self.weights: dict[str, float] = {}
        self.chosen = "ensemble"

    def fit(self, panel: pl.DataFrame, feat_cols: list[str], target: str) -> None:
        X = panel.select(feat_cols).to_numpy().astype(np.float64)
        y = panel[target].to_numpy().astype(np.float64)

        icirs: dict[str, float] = {}
        protos = make_tabular_models(seed=0)
        for fam in self.families:
            icirs[fam] = inner_purged_ic(
                panel, feat_cols, target, protos[fam],
                horizon=self.horizon, n_splits=self.inner_splits)

        raw = {k: max(v, 0.0) for k, v in icirs.items()}
        tot = sum(raw.values())
        # every family below the noise floor -> fall back to equal weights
        self.weights = ({k: v / tot for k, v in raw.items()} if tot > 1e-9
                        else {k: 1.0 / len(self.families) for k in self.families})

        self.fitted = {}
        for fam, w in self.weights.items():
            if w <= 0:
                continue
            fits = []
            for s in range(self.n_seeds):
                m = make_tabular_models(seed=s)[fam]
                m.fit(X, y)
                fits.append(m)
            self.fitted[fam] = fits
        self.chosen = "+".join(f"{k}:{v:.2f}" for k, v in self.weights.items() if v > 0)

    def predict(self, today: pl.DataFrame, feat_cols: list[str]) -> np.ndarray:
        X = today.select(feat_cols).to_numpy().astype(np.float64)
        n = X.shape[0]
        out = np.zeros(n)
        for fam, fits in self.fitted.items():
            p = np.mean([np.asarray(m.predict(X)).ravel() for m in fits], axis=0)
            if self.rank_normalise:
                # ranks make families with different output scales blendable
                order = p.argsort().argsort()
                p = order / max(n - 1, 1) - 0.5
            out += self.weights[fam] * p
        return out
