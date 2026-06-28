"""Resource detection + tuning — use the best available hardware automatically.

Targets two profiles the user actually runs on:
  * **Apple Silicon (M5 Pro, 48 GB)** — many fast CPU cores + unified memory;
    LightGBM/XGBoost run CPU-threaded (no CUDA), embeddings can use Metal (MPS).
  * **RTX-class NVIDIA GPU** — XGBoost ``device="cuda"`` + (optionally) LightGBM
    ``device="gpu"``; embeddings use CUDA.

Nothing here is a hard dependency: torch/psutil are probed lazily and absence
just yields a sane CPU profile.  Override with env vars:

  TS_DEVICE = cpu | gpu        (force tree-model device)
  TS_N_JOBS = <int>            (force worker count)
  TS_GPU    = 0 | 1            (force-disable / enable GPU detection)
"""
from __future__ import annotations

import functools
import os
from dataclasses import dataclass, asdict

from .logging import get_logger

logger = get_logger(__name__)


def _physical_ram_gb() -> float:
    try:
        import psutil  # optional
        return round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass
    try:  # POSIX fallback (works on macOS + Linux)
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except Exception:
        return 0.0


def _detect_gpu() -> tuple[bool, str]:
    """Return (has_cuda, torch_device). torch_device ∈ {cuda, mps, cpu}."""
    if os.environ.get("TS_GPU") == "0":
        return False, "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return True, "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return False, "mps"  # MPS helps embeddings, not LightGBM/XGB
    except Exception:
        pass
    return False, "cpu"


@dataclass(frozen=True)
class ComputeProfile:
    n_jobs: int
    ram_gb: float
    has_cuda: bool
    torch_device: str          # cuda | mps | cpu  (for embeddings / torch models)
    lgbm_device: str           # cpu | gpu
    xgb_device: str            # cpu | cuda
    platform: str              # apple_silicon | linux_gpu | cpu

    def lgbm_params(self) -> dict:
        """Device/thread params to splat into an LGBM constructor."""
        p = {"n_jobs": self.n_jobs}
        if self.lgbm_device == "gpu":
            p["device_type"] = "gpu"
        return p

    def xgb_params(self) -> dict:
        """Device params for XGBoost >= 2.0 (tree_method=hist + device)."""
        return {"tree_method": "hist", "device": self.xgb_device, "n_jobs": self.n_jobs}

    def summary(self) -> str:
        gpu = "CUDA" if self.has_cuda else (self.torch_device.upper() if self.torch_device != "cpu" else "none")
        return (f"{self.platform} · {self.n_jobs} workers · {self.ram_gb:.0f}GB RAM · "
                f"GPU={gpu} · lgbm={self.lgbm_device} · xgb={self.xgb_device}")

    def as_dict(self) -> dict:
        return asdict(self)


@functools.lru_cache(maxsize=1)
def get_compute_profile() -> ComputeProfile:
    """Detect once, cache for the process. Honors TS_* env overrides."""
    cores = os.cpu_count() or 4
    ram = _physical_ram_gb()
    has_cuda, torch_dev = _detect_gpu()

    # Leave a couple cores for the OS/UI on a laptop; use all on a GPU box
    n_jobs = int(os.environ.get("TS_N_JOBS") or max(1, cores - 2 if cores > 4 else cores))

    is_apple = (not has_cuda) and torch_dev == "mps"
    platform = "apple_silicon" if is_apple else ("linux_gpu" if has_cuda else "cpu")

    forced = os.environ.get("TS_DEVICE")
    lgbm_device = "gpu" if (forced == "gpu" or (forced is None and has_cuda)) else "cpu"
    xgb_device = "cuda" if (forced == "gpu" or (forced is None and has_cuda)) else "cpu"

    prof = ComputeProfile(
        n_jobs=n_jobs, ram_gb=ram, has_cuda=has_cuda, torch_device=torch_dev,
        lgbm_device=lgbm_device, xgb_device=xgb_device, platform=platform,
    )
    # Make BLAS/OpenMP libs use the same worker budget
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, str(n_jobs))
    logger.info(f"compute profile: {prof.summary()}")
    return prof
