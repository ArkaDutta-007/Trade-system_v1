"""Filesystem model registry. Persists model + feature spec + metadata."""
from __future__ import annotations

import json
import pickle
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_REGISTRY = Path("reports/models")


@dataclass
class ModelArtifact:
    name: str
    created_at: float
    feature_columns: list[str]
    target: str
    metadata: dict[str, Any] = field(default_factory=dict)


def save_model(
    model: Any,
    name: str,
    feature_columns: list[str],
    target: str,
    metadata: dict | None = None,
    registry: Path | str | None = None,
) -> Path:
    reg = Path(registry or DEFAULT_REGISTRY)
    out = reg / name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    artifact = ModelArtifact(
        name=name,
        created_at=time.time(),
        feature_columns=feature_columns,
        target=target,
        metadata=metadata or {},
    )
    with open(out / "artifact.json", "w") as f:
        json.dump(asdict(artifact), f, indent=2)
    return out


def load_model(name: str, registry: Path | str | None = None) -> tuple[Any, ModelArtifact]:
    reg = Path(registry or DEFAULT_REGISTRY)
    folder = reg / name
    with open(folder / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(folder / "artifact.json") as f:
        artifact = ModelArtifact(**json.load(f))
    return model, artifact


def list_models(registry: Path | str | None = None) -> list[str]:
    reg = Path(registry or DEFAULT_REGISTRY)
    if not reg.exists():
        return []
    return sorted([p.name for p in reg.iterdir() if (p / "artifact.json").exists()])


def save_ensemble_report(
    metrics_rows: list[dict],
    out_path: Path | str,
) -> Path:
    """Persist per-fold × per-model comparative metrics to JSON."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics_rows, indent=2))
    return out_path


def best_ensemble_artifact(registry: Path | str | None = None) -> str | None:
    """Return name of the most recent ensemble artifact, else most recent any model."""
    reg = Path(registry or DEFAULT_REGISTRY)
    all_models = list_models(reg)
    if not all_models:
        return None
    # Prefer ensemble type
    for name in reversed(all_models):
        art_path = reg / name / "artifact.json"
        try:
            meta = json.loads(art_path.read_text()).get("metadata", {})
            if meta.get("model_type") == "ensemble":
                return name
        except Exception:
            pass
    return all_models[-1]  # fallback: latest


def load_latest_model(
    registry: Path | str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load the most recent ensemble (or latest) model from the registry.

    Returns
    -------
    (model, meta) where meta is a plain dict with at least:
        name, feature_columns, target, created_at, model_type
    plus any extra keys stored in ModelArtifact.metadata.

    Raises
    ------
    FileNotFoundError if the registry is empty or doesn't exist.
    """
    reg = Path(registry or DEFAULT_REGISTRY)
    name = best_ensemble_artifact(reg)
    if name is None:
        raise FileNotFoundError(f"No trained models found in {reg}")
    model, artifact = load_model(name, reg)
    meta: dict[str, Any] = {
        "name": artifact.name,
        "feature_columns": artifact.feature_columns,
        "target": artifact.target,
        "created_at": artifact.created_at,
        **artifact.metadata,
    }
    return model, meta
