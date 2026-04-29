"""Comprehensive ensemble model: 14 base learners + 3 ensemble strategies.

Base models (14):
  Gradient Boosting (4): LGBM, XGB, HistGBM, GBM
  Bagging/Forest (3):    RandomForest, ExtraTrees, Bagging
  Boosting (1):          AdaBoost
  Linear (4):            Ridge, ElasticNet, HuberRegressor, BayesianRidge
  Neural/Stochastic (2): MLPRegressor, SGDRegressor

Ensemble variants (3):
  1. Weighted Blend — IC-weighted combination, weights updated per fold
  2. Stack-Ridge    — Ridge meta-learner on base OOS predictions
  3. Stack-LGBM     — LightGBM meta-learner (non-linear meta combination)

SHAP feedback loop:
  - After each fold: compute SHAP for all tree-based models
  - Features with mean|SHAP| below threshold get dropped next fold
  - Weights rebalanced by IC rank
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from ..utils import get_logger

logger = get_logger(__name__)

# ── Model catalogue ────────────────────────────────────────────────────────────

def _build_base_models() -> dict[str, Any]:
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.ensemble import (
        AdaBoostRegressor,
        BaggingRegressor,
        ExtraTreesRegressor,
        GradientBoostingRegressor,
        HistGradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.linear_model import (
        BayesianRidge,
        ElasticNet,
        HuberRegressor,
        Ridge,
        SGDRegressor,
    )
    from sklearn.neural_network import MLPRegressor
    from sklearn.tree import DecisionTreeRegressor

    return {
        # ── Gradient Boosting ──────────────────────────────────────────────
        "lgbm": lgb.LGBMRegressor(
            objective="regression",
            num_leaves=63,
            learning_rate=0.05,
            n_estimators=400,
            min_child_samples=30,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbose=-1,
        ),
        "xgb": xgb.XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
            verbosity=0,
        ),
        "hist_gbm": HistGradientBoostingRegressor(
            max_iter=400,
            max_leaf_nodes=63,
            learning_rate=0.05,
            min_samples_leaf=30,
        ),
        "gbm": GradientBoostingRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=20,
        ),
        # ── Bagging / Forest ──────────────────────────────────────────────
        "rf": RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=20,
            max_features=0.7,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=20,
            max_features=0.7,
            n_jobs=-1,
        ),
        "bagging": BaggingRegressor(
            estimator=DecisionTreeRegressor(max_depth=6),
            n_estimators=100,
            max_samples=0.8,
            max_features=0.8,
            n_jobs=-1,
        ),
        # ── Boosting ──────────────────────────────────────────────────────
        "adaboost": AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=4),
            n_estimators=200,
            learning_rate=0.1,
        ),
        # ── Linear ────────────────────────────────────────────────────────
        "ridge": Ridge(alpha=1.0),
        "elasticnet": ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=2000),
        "huber": HuberRegressor(epsilon=1.35, max_iter=300),
        "bayesian_ridge": BayesianRidge(max_iter=300),
        # ── Neural / Stochastic ───────────────────────────────────────────
        "mlp": MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            learning_rate_init=0.001,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
        ),
        "sgd": SGDRegressor(
            loss="huber",
            epsilon=0.1,
            learning_rate="adaptive",
            eta0=0.01,
            max_iter=300,
            tol=1e-4,
        ),
    }


# ── SHAP helpers (tree-based only) ─────────────────────────────────────────────

_TREE_MODELS = {"lgbm", "xgb", "hist_gbm", "gbm", "rf", "extra_trees", "bagging", "adaboost"}


def _compute_shap_importances(
    models: dict[str, Any],
    X: np.ndarray,
    feature_names: list[str],
    sample: int = 5000,
) -> dict[str, np.ndarray]:
    """Return mean|SHAP| per feature per tree-based model."""
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed — skipping SHAP feedback")
        return {}

    rng = np.random.default_rng(42)
    if len(X) > sample:
        idx = rng.choice(len(X), sample, replace=False)
        Xs = X[idx]
    else:
        Xs = X

    importances: dict[str, np.ndarray] = {}
    for name, model in models.items():
        if name not in _TREE_MODELS:
            continue
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(Xs)
            importances[name] = np.abs(sv).mean(axis=0)
        except Exception as e:
            logger.debug(f"SHAP failed for {name}: {e}")
    return importances


# ── IC helpers ────────────────────────────────────────────────────────────────

def _ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank IC; returns 0 on degenerate input."""
    if len(y_true) < 10 or np.std(y_pred) < 1e-10:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, _ = spearmanr(y_true, y_pred)
    return float(r) if not np.isnan(r) else 0.0


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, r2_score
    ic = _ic(y_true, y_pred)
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if np.std(y_true) > 1e-10 else 0.0
    return {"ic": ic, "mae": mae, "r2": r2}


# ── Main EnsembleModel ─────────────────────────────────────────────────────────

class EnsembleModel:
    """Trains 14 base models + 3 ensemble variants on each walk-forward fold.

    Usage
    -----
    em = EnsembleModel()
    em.fit(X_train, y_train, X_val, y_val, feature_names)
    preds = em.predict(X_test)          # dict: model_name -> array
    metrics = em.val_metrics            # dict: model_name -> {ic, mae, r2}
    weights = em.blend_weights          # IC-normalised dict
    """

    def __init__(self, shap_drop_threshold: float = 0.0, n_jobs: int = -1):
        self.shap_drop_threshold = shap_drop_threshold
        self._models: dict[str, Any] = {}
        self._meta_ridge: Any = None
        self._meta_lgbm: Any = None
        self.blend_weights: dict[str, float] = {}
        self.val_metrics: dict[str, dict] = {}
        self.feature_names: list[str] = []
        self.active_features: list[str] = []  # after SHAP pruning
        self._val_preds: dict[str, np.ndarray] = {}
        self._fitted = False

    # ── Training ──────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list[str] | None = None,
        active_features: list[int] | None = None,
    ) -> "EnsembleModel":
        """Train all base models. Then fit meta-learners on val OOS predictions."""
        self.feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]
        self.active_features = list(self.feature_names)

        # Optionally restrict to SHAP-pruned feature subset
        if active_features is not None:
            X_train = X_train[:, active_features]
            X_val = X_val[:, active_features]
            self.active_features = [self.feature_names[i] for i in active_features]

        base = _build_base_models()
        val_oos: dict[str, np.ndarray] = {}

        for name, model in base.items():
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_val)
                val_oos[name] = preds
                self._models[name] = model
                self.val_metrics[name] = _metrics(y_val, preds)
                logger.info(
                    f"  {name:15s} IC={self.val_metrics[name]['ic']:+.3f} "
                    f"MAE={self.val_metrics[name]['mae']:.5f} "
                    f"R²={self.val_metrics[name]['r2']:.4f}"
                )
            except Exception as e:
                logger.warning(f"  {name} failed to train: {e}")

        self._val_preds = val_oos

        # ── SHAP feedback: identify active features for NEXT fold ────────
        self._update_shap_active_features(X_val)

        # ── IC-weighted blend ────────────────────────────────────────────
        self._compute_blend_weights()

        # ── Blend predictions on val ─────────────────────────────────────
        blend_val = self._blend(val_oos)
        self.val_metrics["ensemble_blend"] = _metrics(y_val, blend_val)

        # ── Stack-Ridge: trained on val OOS base predictions ─────────────
        stack_X = np.column_stack(list(val_oos.values())) if val_oos else None
        if stack_X is not None and len(stack_X) > 20:
            from sklearn.linear_model import Ridge as _Ridge
            import lightgbm as lgb

            self._meta_ridge = _Ridge(alpha=1.0)
            self._meta_ridge.fit(stack_X, y_val)
            stack_ridge_val = self._meta_ridge.predict(stack_X)
            self.val_metrics["ensemble_stack_ridge"] = _metrics(y_val, stack_ridge_val)

            self._meta_lgbm = lgb.LGBMRegressor(
                n_estimators=200, num_leaves=15, learning_rate=0.05, verbose=-1
            )
            self._meta_lgbm.fit(stack_X, y_val)
            stack_lgbm_val = self._meta_lgbm.predict(stack_X)
            self.val_metrics["ensemble_stack_lgbm"] = _metrics(y_val, stack_lgbm_val)

        self._fitted = True
        return self

    def _update_shap_active_features(self, X_val: np.ndarray) -> None:
        if self.shap_drop_threshold <= 0:
            return
        importances = _compute_shap_importances(self._models, X_val, self.active_features)
        if not importances:
            return
        mean_imp = np.mean(np.stack(list(importances.values())), axis=0)
        keep = [i for i, v in enumerate(mean_imp) if v >= self.shap_drop_threshold]
        if len(keep) >= 3:
            self.active_features = [self.active_features[i] for i in keep]
            logger.info(
                f"  SHAP feedback: retained {len(keep)}/{len(mean_imp)} features "
                f"above threshold={self.shap_drop_threshold}"
            )

    def _compute_blend_weights(self) -> None:
        """Normalise positive IC scores into blend weights."""
        ics = {n: max(0.0, m["ic"]) for n, m in self.val_metrics.items() if n in self._models}
        total = sum(ics.values())
        if total < 1e-9:
            self.blend_weights = {n: 1.0 / len(ics) for n in ics}
        else:
            self.blend_weights = {n: v / total for n, v in ics.items()}

    def _blend(self, preds: dict[str, np.ndarray]) -> np.ndarray:
        if not preds:
            return np.zeros(1)
        arrs = []
        ws = []
        for name, arr in preds.items():
            w = self.blend_weights.get(name, 0.0)
            arrs.append(arr)
            ws.append(w)
        total_w = sum(ws)
        if total_w < 1e-9:
            return np.mean(arrs, axis=0)
        return sum(a * w for a, w in zip(arrs, ws)) / total_w

    # ── Inference ─────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Returns predictions for every model + 3 ensemble variants."""
        if not self._fitted:
            raise RuntimeError("EnsembleModel not fitted")

        # Restrict to active features if names map correctly
        if len(self.active_features) < X.shape[1]:
            feat_idx = [
                self.feature_names.index(f)
                for f in self.active_features
                if f in self.feature_names
            ]
            if feat_idx:
                X = X[:, feat_idx]

        base_preds: dict[str, np.ndarray] = {}
        for name, model in self._models.items():
            try:
                base_preds[name] = model.predict(X)
            except Exception as e:
                logger.debug(f"predict failed for {name}: {e}")

        out = dict(base_preds)
        out["ensemble_blend"] = self._blend(base_preds)

        stack_X = np.column_stack(list(base_preds.values())) if base_preds else None
        if stack_X is not None:
            if self._meta_ridge is not None:
                try:
                    out["ensemble_stack_ridge"] = self._meta_ridge.predict(stack_X)
                except Exception:
                    pass
            if self._meta_lgbm is not None:
                try:
                    out["ensemble_stack_lgbm"] = self._meta_lgbm.predict(stack_X)
                except Exception:
                    pass

        return out

    def best_ensemble_name(self) -> str:
        """Return the variant with highest validation IC."""
        ensemble_keys = ["ensemble_stack_lgbm", "ensemble_stack_ridge", "ensemble_blend"]
        best = max(
            (k for k in ensemble_keys if k in self.val_metrics),
            key=lambda k: self.val_metrics[k]["ic"],
            default="ensemble_blend",
        )
        return best

    def comparative_table(self) -> list[dict]:
        """Return sorted list of {model, ic, mae, r2, weight} for display."""
        rows = []
        for name, m in self.val_metrics.items():
            rows.append(
                {
                    "model": name,
                    "ic": round(m["ic"], 4),
                    "mae": round(m["mae"], 6),
                    "r2": round(m["r2"], 4),
                    "weight": round(self.blend_weights.get(name, 0.0), 4),
                }
            )
        rows.sort(key=lambda r: r["ic"], reverse=True)
        return rows
