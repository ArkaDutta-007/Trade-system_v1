"""Machine-independent path resolution for the ops scripts.

Every script under ``ops/`` used to open with::

    REPO = Path("/home/ad2688/Desktop/Trade-system_v1")
    OUT  = Path.home() / "trade-ops/research/out"

which meant moving the tooling to another box required editing the top of each
file.  Both locations are now resolved once, here, with the same precedence
everywhere:

``repo_root()``
    ``$TS_REPO`` if set, else the directory this file lives in (``ops/`` sits
    inside the repo), else the legacy Desktop path if it happens to exist.

``ops_root()``
    ``$TS_OPS`` if set, else ``~/trade-ops`` if it exists, else ``<repo>/ops``.

Nothing about the split changes: ops state (books, flags, digests, research
output) stays outside the tracked tree by default, because it is live state and
does not belong in git.  It is just no longer pinned to one machine's home
directory.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = ["repo_root", "ops_root", "research_out", "add_repo_to_path"]

_LEGACY_REPO = Path("/home/ad2688/Desktop/Trade-system_v1")


def repo_root() -> Path:
    """Where the trading-system repo lives on this machine."""
    env = os.environ.get("TS_REPO")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve().parent.parent   # ops/paths.py -> repo
    if (here / "src" / "trading_system").is_dir():
        return here
    if (_LEGACY_REPO / "src" / "trading_system").is_dir():
        return _LEGACY_REPO
    raise RuntimeError(
        "cannot locate the trading-system repo — set TS_REPO to its path")


def ops_root() -> Path:
    """Where ops state (books, flags, digests, research output) lives."""
    env = os.environ.get("TS_OPS")
    if env:
        return Path(env).expanduser().resolve()
    legacy = Path.home() / "trade-ops"
    if legacy.is_dir():
        return legacy
    return repo_root() / "ops"


def research_out() -> Path:
    """Directory for research artefacts; created on first use."""
    p = ops_root() / "research" / "out"
    p.mkdir(parents=True, exist_ok=True)
    return p


def add_repo_to_path() -> Path:
    """Put the repo's ``src/`` on ``sys.path`` and return the repo root."""
    repo = repo_root()
    src = str(repo / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return repo
