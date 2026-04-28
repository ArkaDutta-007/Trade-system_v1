"""Configuration loader. Resolves relative paths against project root and expands env vars."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _load_dotenv(root: Path) -> None:
    """Load key=value pairs from <root>/.env into os.environ (no-op if absent).

    Only sets variables that are NOT already in the environment, so shell
    exports always take precedence over .env values.
    """
    env_file = root / ".env"
    if not env_file.exists():
        return
    with open(env_file) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            # Skip placeholder values only
            if val in ("", "your_deepseek_api_key_here", "your_fred_key", "your_newsapi_key"):
                continue
            if key and key not in os.environ:
                os.environ[key] = val


def _expand_env(obj: Any) -> Any:
    if isinstance(obj, str):
        return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), obj)
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    return obj


def find_project_root(start: Path | None = None) -> Path:
    p = (start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / "pyproject.toml").exists() and (cand / "configs").exists():
            return cand
    return Path(__file__).resolve().parents[2]


@dataclass
class Config:
    raw: dict
    project_root: Path

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        root = find_project_root()
        _load_dotenv(root)  # auto-load .env before expanding ${VAR} placeholders
        cfg_path = Path(path) if path else root / "configs" / "default.yaml"
        with open(cfg_path) as f:
            raw = yaml.safe_load(f)
        raw = _expand_env(raw)
        if raw.get("project_root"):
            root = Path(raw["project_root"]).expanduser().resolve()
        # Resolve universe (inline or via universe_file)
        if "universe" not in raw and raw.get("universe_file"):
            with open(root / raw["universe_file"]) as f:
                u = yaml.safe_load(f)
            tickers = list(dict.fromkeys((u.get("required") or []) + (u.get("additions") or [])))
            raw["universe"] = {
                "name": u.get("name", "universe"),
                "benchmark": u.get("benchmark", "SPY"),
                "tickers": tickers,
                "required": u.get("required") or [],
                "additions": u.get("additions") or [],
            }
        return cls(raw=raw, project_root=root)

    def path(self, key: str) -> Path:
        rel = self.raw["paths"][key]
        return (self.project_root / rel).resolve()

    def __getitem__(self, k: str) -> Any:
        return self.raw[k]

    def get(self, k: str, default: Any = None) -> Any:
        return self.raw.get(k, default)


def get_config(path: str | Path | None = None) -> Config:
    return Config.load(path)
