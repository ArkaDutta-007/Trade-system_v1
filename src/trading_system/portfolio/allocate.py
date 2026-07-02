"""Budget allocation: RMT-cleaned covariance + Hierarchical Risk Parity.

The invest planner sizes a *budget* (a dollar tranche) across a conviction set.
Two ingredients, blended:

  1. **Conviction (Kelly-style)** — reward-to-downside per name from the
     calibrated conformal band at that name's chosen hold horizon (same
     Sortino-style ratio as ``sizing.distribution_sized_weights``).
  2. **Diversification (HRP)** — Hierarchical Risk Parity (López de Prado) on
     an **RMT-cleaned** covariance of trailing daily returns: eigenvalues
     inside the Marchenko–Pastur noise band are flattened to their mean before
     the covariance is rebuilt, so the cluster tree and inverse-variance splits
     see the *real* correlation structure, not sampling noise. HRP needs no
     matrix inversion, so it is robust exactly where Markowitz breaks.

Final weight ∝ ½·kelly + ½·hrp, capped per name, renormalised. The composite
flag board decides *how much* of the budget deploys; this module decides
*where* it goes.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "marchenko_pastur_lambda_plus",
    "clean_correlation",
    "hrp_weights",
    "blend_weights",
    "budget_to_positions",
]


def marchenko_pastur_lambda_plus(n_assets: int, n_obs: int) -> float:
    """Upper edge of the Marchenko–Pastur noise band for q = N/T (unit variance)."""
    q = n_assets / max(n_obs, 1)
    return (1.0 + math.sqrt(q)) ** 2


def clean_correlation(corr: np.ndarray, n_obs: int) -> np.ndarray:
    """Flatten noise-band eigenvalues to their mean; rebuild with unit diagonal."""
    corr = np.asarray(corr, dtype=np.float64)
    n = corr.shape[0]
    if n < 2:
        return corr.copy()
    vals, vecs = np.linalg.eigh(corr)
    lam_plus = marchenko_pastur_lambda_plus(n, n_obs)
    noise = vals < lam_plus
    if noise.any() and not noise.all():
        vals = vals.copy()
        vals[noise] = vals[noise].mean()
    cleaned = (vecs * vals) @ vecs.T
    d = np.sqrt(np.clip(np.diag(cleaned), 1e-12, None))
    cleaned = cleaned / np.outer(d, d)
    np.fill_diagonal(cleaned, 1.0)
    return cleaned


def _quasi_diag(linkage: np.ndarray) -> list[int]:
    """Leaf ordering of a scipy linkage matrix (recursion-free)."""
    n = linkage.shape[0] + 1
    order = [int(linkage[-1, 0]), int(linkage[-1, 1])]
    while max(order) >= n:
        nxt: list[int] = []
        for item in order:
            if item < n:
                nxt.append(item)
            else:
                row = linkage[item - n]
                nxt.extend((int(row[0]), int(row[1])))
        order = nxt
    return order


def _cluster_var(cov: np.ndarray, idx: list[int]) -> float:
    """Variance of the inverse-variance-weighted sub-cluster portfolio."""
    sub = cov[np.ix_(idx, idx)]
    ivp = 1.0 / np.clip(np.diag(sub), 1e-12, None)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)


def hrp_weights(cov: np.ndarray, corr: np.ndarray | None = None) -> np.ndarray:
    """Hierarchical Risk Parity weights (single-linkage, recursive bisection)."""
    cov = np.asarray(cov, dtype=np.float64)
    n = cov.shape[0]
    if n == 1:
        return np.array([1.0])
    if corr is None:
        d = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
        corr = cov / np.outer(d, d)
    corr = np.clip(corr, -1.0, 1.0)

    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)
    link = linkage(squareform(dist, checks=False), method="single")
    order = _quasi_diag(link)

    w = np.ones(n)
    clusters: list[list[int]] = [order]
    while clusters:
        nxt: list[list[int]] = []
        for cl in clusters:
            if len(cl) <= 1:
                continue
            half = len(cl) // 2
            left, right = cl[:half], cl[half:]
            var_l, var_r = _cluster_var(cov, left), _cluster_var(cov, right)
            alpha = 1.0 - var_l / max(var_l + var_r, 1e-12)
            w[left] *= alpha
            w[right] *= 1.0 - alpha
            nxt.extend((left, right))
        clusters = nxt
    return w / w.sum()


def blend_weights(
    tickers: list[str],
    kelly: dict[str, float],
    returns: np.ndarray | None,
    max_weight: float = 0.25,
    hrp_share: float = 0.5,
) -> dict[str, float]:
    """Blend conviction (Kelly) with diversification (RMT-cleaned HRP).

    Parameters
    ----------
    tickers:
        The candidate names, in the column order of ``returns``.
    kelly:
        Raw (non-negative) conviction scores per ticker; normalised internally.
    returns:
        ``(T, N)`` trailing daily returns aligned to ``tickers``; ``None`` or
        too-short history falls back to conviction-only weights.
    max_weight:
        Per-name cap on the final weight (share of the tranche).
    hrp_share:
        Weight on the HRP leg (0 = pure conviction, 1 = pure risk parity).
    """
    if not tickers:
        return {}
    k = np.array([max(0.0, float(kelly.get(t, 0.0))) for t in tickers])
    if k.sum() <= 0:
        k = np.ones(len(tickers))
    k = k / k.sum()

    h = np.full(len(tickers), 1.0 / len(tickers))
    if returns is not None and len(tickers) >= 2 and returns.shape[0] >= 60:
        rets = np.asarray(returns, dtype=np.float64)
        cov = np.cov(rets, rowvar=False)
        std = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
        corr = clean_correlation(cov / np.outer(std, std), n_obs=rets.shape[0])
        cov_clean = corr * np.outer(std, std)
        h = hrp_weights(cov_clean, corr)

    w = (1.0 - hrp_share) * k + hrp_share * h
    w = _cap_and_redistribute(w / w.sum(), max_weight)
    return {t: float(x) for t, x in zip(tickers, w)}


def _cap_and_redistribute(w: np.ndarray, cap: float, max_iter: int = 32) -> np.ndarray:
    """Cap weights, redistributing the excess pro-rata among uncapped names.

    If the cap is infeasible (n·cap < 1) the residual stays uninvested — the
    planner reports it as cash rather than silently breaking the cap.
    """
    w = w.copy()
    for _ in range(max_iter):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        under = ~over
        room = float(w[under].sum())
        if room <= 1e-12:
            break
        w[under] += excess * w[under] / room
    return w


def budget_to_positions(
    weights: dict[str, float],
    prices: dict[str, float],
    deployable: float,
    min_position: float = 50.0,
) -> tuple[dict[str, dict], float]:
    """Convert tranche weights into dollar/share positions within a budget.

    Positions smaller than ``min_position`` dollars are dropped and their
    weight redistributed pro-rata (one pass). Returns ``(positions, leftover)``
    where each position has ``dollars``, ``shares`` (fractional) and
    ``whole_shares``.
    """
    live = {t: w for t, w in weights.items() if prices.get(t, 0) > 0}
    if not live or deployable <= 0:
        return {}, deployable

    total = sum(live.values())
    live = {t: w / total for t, w in live.items()}
    kept = {t: w for t, w in live.items() if w * deployable >= min_position}
    if kept and len(kept) < len(live):
        total = sum(kept.values())
        live = {t: w / total for t, w in kept.items()}

    positions: dict[str, dict] = {}
    spent = 0.0
    for t, w in live.items():
        px = prices[t]
        dollars = w * deployable
        positions[t] = {
            "weight": round(w, 4),
            "dollars": round(dollars, 2),
            "shares": round(dollars / px, 4),
            "whole_shares": int(dollars // px),
            "price": round(px, 4),
        }
        spent += dollars
    return positions, round(deployable - spent, 2)
