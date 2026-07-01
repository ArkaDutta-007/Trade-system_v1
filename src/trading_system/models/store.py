"""Committed model store — the best models live in git (not under reports/).

``reports/`` is gitignored, so models trained there don't travel with the repo.
The user wants the *best* forecasters kept and versioned, so this writes them to
``models_store/`` at the project root (tracked by git):

    models_store/
      forecast/<h>d/model.pkl         best estimator for horizon h, refit on all data
      forecast/<h>d/metrics.json      OOS metrics + leakage-gate result + feature list
      intervals/interval_bundle.pkl   conformal quantile bounds (copied from training)
      manifest.json                   index: horizons, models, metrics, when trained

LightGBM/XGB/sklearn estimators pickle to a few MB each — fine to commit.
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any

from ..utils import get_logger

logger = get_logger(__name__)

STORE_DIRNAME = "models_store"


def default_store(project_root: Path) -> Path:
    d = Path(project_root) / STORE_DIRNAME
    (d / "forecast").mkdir(parents=True, exist_ok=True)
    (d / "intervals").mkdir(parents=True, exist_ok=True)
    return d


def _dump(obj: Any, path: Path) -> None:
    try:
        import joblib
        joblib.dump(obj, path)
    except Exception:
        with open(path, "wb") as f:
            pickle.dump(obj, f)


def _load(path: Path) -> Any:
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


def save_forecast_results(results: dict, store_dir: Path, compute_summary: str = "",
                          train_config: dict | None = None) -> Path:
    """Persist per-horizon best models + a manifest. ``results`` is {h: HorizonResult}."""
    store_dir = Path(store_dir)
    fdir = store_dir / "forecast"
    fdir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "created_at": time.time(),
        "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "compute": compute_summary,
        "train_config": train_config or {},
        "horizons": {},
    }
    for h, res in results.items():
        hd = fdir / f"{h}d"
        hd.mkdir(parents=True, exist_ok=True)
        if res.best_model is not None:
            _dump(res.best_model, hd / "model.pkl")
        metrics = {
            "horizon": h,
            "best_model": res.best_model_name,
            "per_model": res.per_model,
            "leakage_gate": res.leakage_gate,
            "deflation": getattr(res, "deflation", {}),
            "cv_mode": getattr(res, "cv_mode", "walkforward"),
            "feature_columns": res.feature_columns,
            "n_rows": res.n_rows,
            "trained_through": res.trained_through,
        }
        (hd / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
        best = res.per_model.get(res.best_model_name, {})
        manifest["horizons"][str(h)] = {
            "best_model": res.best_model_name,
            "icir": round(best.get("icir", 0.0), 3),
            "univ_icir": round(best.get("univ_icir", 0.0), 3),
            "ic_mean": round(best.get("ic_mean", 0.0), 4),
            "hit_rate": round(best.get("hit_rate", 0.0), 3),
            "leak_pass": res.leakage_gate.get("pass"),
            "deflation_pass": getattr(res, "deflation", {}).get("pass"),
            "trained_through": res.trained_through,
        }
    (store_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    logger.info(f"forecast models saved → {fdir}")
    return store_dir


def load_forecast_model(horizon: int, store_dir: Path) -> tuple[Any, dict] | None:
    """Load (estimator, metrics) for a horizon from the store, or None."""
    hd = Path(store_dir) / "forecast" / f"{horizon}d"
    mp, jp = hd / "model.pkl", hd / "metrics.json"
    if not (mp.exists() and jp.exists()):
        return None
    try:
        return _load(mp), json.loads(jp.read_text())
    except Exception as e:
        logger.warning(f"failed to load forecast model {horizon}d: {e}")
        return None


def read_manifest(store_dir: Path) -> dict:
    p = Path(store_dir) / "manifest.json"
    return json.loads(p.read_text()) if p.exists() else {}


def copy_interval_bundle(reports_models: Path, store_dir: Path) -> bool:
    """Copy a trained conformal interval bundle into the committed store."""
    src = Path(reports_models) / "intervals" / "interval_bundle.pkl"
    if not src.exists():
        return False
    import shutil
    dst_dir = Path(store_dir) / "intervals"
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_dir / "interval_bundle.pkl")
    meta = Path(reports_models) / "intervals" / "meta.json"
    if meta.exists():
        shutil.copy2(meta, dst_dir / "meta.json")
    return True
