"""Random Matrix Theory features — denoise the cross-asset correlation matrix.

A sample correlation matrix of N assets from T observations is mostly *noise*:
when there is no real structure its eigenvalues fall inside the Marchenko–Pastur
(MP) band ``[(1−√q)², (1+√q)²]`` with ``q = N/T``.  Eigenvalues that poke out
*above* the MP edge are the statistically real collective modes — the market
factor (the giant top eigenvalue) and sectors.  This is the Laloux–Bouchaud–
Potters result that imported Wigner's nuclear-physics spectral theory into
finance.

Two causal, rolling features come out of it:

  ``rmt_systematic_frac``  market-wide: the share of cross-sectional variance
                           carried by above-MP eigenvalues — high = a correlated,
                           "everything moves together" tape (fragile, low
                           diversification); low = idiosyncratic, stock-pickers' market.
  ``rmt_market_beta``      per-ticker: loading on the dominant (market-mode)
                           eigenvector, scaled by √λ_top — each name's exposure to
                           the collective mode, RMT-cleaned rather than regressed
                           against an index proxy.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from ..utils import get_logger

logger = get_logger(__name__)

RMT_COLUMNS = ["rmt_systematic_frac", "rmt_market_beta"]


def _ffill(a: np.ndarray) -> np.ndarray:
    out = a.copy()
    last = np.nan
    for i in range(len(a)):
        if np.isfinite(a[i]):
            last = a[i]
        out[i] = last
    return out


def compute_rmt_features(
    df: pl.DataFrame,
    window: int = 252,
    stride: int = 5,
    min_tickers: int = 20,
) -> pl.DataFrame:
    """Add ``rmt_systematic_frac`` (by date) and ``rmt_market_beta`` (by ticker,date).

    Recomputed every ``stride`` days on a trailing ``window`` and forward-filled
    (causal).  Needs at least ``min_tickers`` names with a full window.
    """
    if df.is_empty() or "adj_close" not in df.columns:
        return df
    df = df.sort(["ticker", "date"])

    if "log_ret_1d" in df.columns:
        rets = df.select(["date", "ticker", pl.col("log_ret_1d").alias("_r")])
    else:
        rets = df.select(
            ["date", "ticker",
             (pl.col("adj_close") / pl.col("adj_close").shift(1).over("ticker")).log().alias("_r")]
        )

    wide = rets.pivot(index="date", on="ticker", values="_r").sort("date")
    dates = wide["date"].to_numpy()
    tickers = [c for c in wide.columns if c != "date"]
    M = wide.select(tickers).to_numpy().astype(float)        # (D, N)
    D, N = M.shape
    if D < window or N < min_tickers:
        logger.info(f"RMT skipped (dates={D}, tickers={N}; need {window}/{min_tickers})")
        return df

    sysfrac = np.full(D, np.nan)
    loadings = np.full((D, N), np.nan)
    anchors = list(range(window - 1, D, stride))
    if anchors and anchors[-1] != D - 1:
        anchors.append(D - 1)

    for a in anchors:
        win = M[a - window + 1: a + 1]                       # (W, N)
        ok = np.where(np.isfinite(win).all(axis=0))[0]
        if len(ok) < min_tickers:
            continue
        Z = win[:, ok]
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
        C = (Z.T @ Z) / window                               # correlation matrix
        evals, evecs = np.linalg.eigh(C)                     # ascending
        q = len(ok) / window
        lam_plus = (1.0 + np.sqrt(q)) ** 2                   # MP upper edge (σ²=1)
        tot = float(evals.sum())
        if tot <= 0:
            continue
        sysfrac[a] = float(evals[evals > lam_plus].sum() / tot)
        v0 = evecs[:, -1]                                    # top (market-mode) eigenvector
        if v0.sum() < 0:                                     # orient so the market mode is positive
            v0 = -v0
        loadings[a, ok] = v0 * np.sqrt(max(evals[-1], 0.0))

    sysfrac = _ffill(sysfrac)
    for j in range(N):
        loadings[:, j] = _ffill(loadings[:, j])

    sys_df = pl.DataFrame({"date": dates, "rmt_systematic_frac": sysfrac})
    load_df = pl.DataFrame({"date": dates, **{tickers[j]: loadings[:, j] for j in range(N)}})
    load_long = load_df.unpivot(index="date", variable_name="ticker", value_name="rmt_market_beta")

    # fill_nan(None): numpy-backed columns carry NaN-floats, which is_not_null
    # treats as present — convert to nulls so coverage gating + drop_nulls behave.
    return (df.join(sys_df, on="date", how="left")
              .join(load_long, on=["date", "ticker"], how="left")
              .with_columns(pl.col("rmt_systematic_frac").fill_nan(None),
                            pl.col("rmt_market_beta").fill_nan(None)))
