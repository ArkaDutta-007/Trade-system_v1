"""Panel integration for the nonlinear estimators — causal, strided, per-ticker.

The estimators in :mod:`nonlinear` are O(W)…O(W²) per call, so recomputing one
every day for every ticker would be wasteful *and* statistically pointless — a
Hurst exponent or recurrence-determinism barely moves day to day.  Instead each
feature is recomputed every ``stride`` trading days on a trailing window and
**forward-filled** in between (carrying the last *past* value forward, which is
strictly causal).  Rows before a ticker's first full window stay null and the
reserve resolver drops a column that ends up too sparse.

The per-ticker work is embarrassingly parallel, so it fans out across the cores
the compute profile reports.
"""
from __future__ import annotations

import os

import numpy as np
import polars as pl

from ..utils import get_logger
from .nonlinear import ALL_FEATURES, FAST_FEATURES, DEEP_FEATURES

logger = get_logger(__name__)

NONLINEAR_FAST_COLUMNS = [f.name for f in FAST_FEATURES]
NONLINEAR_DEEP_COLUMNS = [f.name for f in DEEP_FEATURES]
NONLINEAR_COLUMNS = [f.name for f in ALL_FEATURES]


# ── core windowing ──────────────────────────────────────────────────────────────

def _rolling_std(a: np.ndarray, w: int = 5) -> np.ndarray:
    out = np.full(len(a), np.nan)
    for i in range(len(a)):
        seg = a[max(0, i - w + 1): i + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg) >= 2:
            out[i] = seg.std()
    return out


def _strided_rolling(source: np.ndarray, window: int, stride: int, func) -> np.ndarray:
    """Compute ``func`` on the trailing ``window`` every ``stride`` rows; forward-fill.

    Strictly backward-looking: the value at row i uses ``source[i-window+1 : i+1]``
    (or the most recent earlier anchor).  Rows before the first full window → NaN.
    """
    T = len(source)
    out = np.full(T, np.nan)
    if T < window:
        return out
    anchors = list(range(window - 1, T, stride))
    if anchors[-1] != T - 1:
        anchors.append(T - 1)               # always refresh the most recent row
    anchor_vals: dict[int, float] = {}
    for i in anchors:
        seg = source[i - window + 1: i + 1]
        try:
            anchor_vals[i] = float(func(seg))
        except Exception:
            anchor_vals[i] = np.nan
    last = np.nan
    for i in range(window - 1, T):
        v = anchor_vals.get(i)
        if v is not None and np.isfinite(v):
            last = v
        out[i] = last
    return out


def _ticker_sources(close: np.ndarray) -> dict[str, np.ndarray]:
    logp = np.log(np.where(close > 0, close, np.nan))
    logret = np.diff(logp, prepend=np.nan)
    return {"logprice": logp, "logret": logret, "volproxy": _rolling_std(logret, 5)}


def _worker(payload):
    start, close, names = payload
    specs = {f.name: f for f in ALL_FEATURES}
    # Degenerate windows (constant series, zero variance) legitimately yield NaN
    # inside the estimators (handled downstream as nulls); silence the expected
    # divide-by-zero / invalid-value RuntimeWarnings they emit.
    import warnings
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"), \
            warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        src = _ticker_sources(close)
        out = {nm: _strided_rolling(src[specs[nm].source], specs[nm].window,
                                    specs[nm].stride, specs[nm].func) for nm in names}
    return start, out


# ── public entry ─────────────────────────────────────────────────────────────────

def compute_nonlinear_features(
    df: pl.DataFrame,
    deep: bool = False,
    parallel: bool = True,
    n_jobs: int | None = None,
    progress: bool = False,
) -> pl.DataFrame:
    """Add nonlinear-dynamics columns to a (ticker, date) panel.

    Parameters
    ----------
    deep:
        Also compute the heavier O(W²)/fit-based tier (sample entropy, Lyapunov,
        RQA, 0–1 chaos test, LPPLS).  Off by default.
    parallel:
        Fan the per-ticker work across processes (count from the compute profile).
    """
    if df.is_empty() or "adj_close" not in df.columns:
        return df
    feats = FAST_FEATURES + (DEEP_FEATURES if deep else [])
    names = [f.name for f in feats]
    df = df.sort(["ticker", "date"])

    close = df["adj_close"].to_numpy().astype(float)
    tk = df["ticker"].to_numpy()
    bounds, s = [], 0
    for i in range(1, len(tk) + 1):
        if i == len(tk) or tk[i] != tk[s]:
            bounds.append((s, i))
            s = i
    payloads = [(s, close[s:e], names) for (s, e) in bounds]

    results = {nm: np.full(df.height, np.nan) for nm in names}

    def _collect(start, out):
        for nm, arr in out.items():
            results[nm][start:start + len(arr)] = arr

    # Parallelism uses a **spawn** ProcessPoolExecutor in a ``with`` block. Two
    # subtle macOS failure modes ruled this design:
    #   * loky / fork pools *segfault* (uncatchably) when the parent already has
    #     torch / MPS / Metal state live — spawn workers start clean, so they don't.
    #   * a lingering worker pool deadlocks a later torch training — the ``with``
    #     block tears the pool down immediately, so nothing lingers.
    # ``_worker`` is a module-level function, so spawn pickles it by name and
    # re-imports the package in the child (no __main__-guard gymnastics needed when
    # called from library code under a guarded entry point like the CLI or pytest).
    # We never import torch here (os.cpu_count, not get_compute_profile).
    ran_parallel = False
    if parallel and len(payloads) > 1:
        try:
            import multiprocessing as mp
            from concurrent.futures import ProcessPoolExecutor
            if n_jobs is None:
                n_jobs = max(1, (os.cpu_count() or 4) - 1)
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as ex:
                for start, out in ex.map(_worker, payloads):
                    _collect(start, out)
            ran_parallel = True
        except Exception as e:
            logger.warning(f"parallel nonlinear compute failed ({e}); falling back to serial")

    if not ran_parallel:
        seq = payloads
        if progress:
            try:
                from tqdm import tqdm
                seq = tqdm(payloads, desc="nonlinear", unit="tkr")
            except Exception:
                pass
        for p in seq:
            start, out = _worker(p)
            _collect(start, out)

    tier = "fast+deep" if deep else "fast"
    logger.info(f"nonlinear features: {len(names)} cols ({tier}) over {len(bounds)} tickers")
    # fill_nan(None): numpy NaN becomes a polars NaN-float, which is_not_null treats
    # as *present* — that would defeat resolve_reserve's coverage gate and leak NaN
    # into tabular models. Convert to proper nulls so they behave like every other
    # leakage-safe feature (null before a ticker's first full window).
    return df.with_columns([pl.Series(nm, results[nm]).fill_nan(None) for nm in names])
