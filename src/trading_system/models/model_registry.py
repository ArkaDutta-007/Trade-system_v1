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
