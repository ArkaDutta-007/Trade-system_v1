#!/usr/bin/env python3
"""Estimation and allocation mathematics for the pick/portfolio stack.

The engineering was already fine; the *statistics* were naive. Four specific
weaknesses and the standard fixes:

1. RAW FORECASTS ARE OVER-DISPERSED.
   Cross-sectional model scores are mostly noise (measured daily IC ~0.01).
   Ranking on them treats a 0.05 score edge as real. James–Stein shrinkage
   pulls every score toward the cross-sectional mean by a factor derived from
   the noise-to-signal ratio, so only genuinely separated names keep their
   edge. Shrinking by the *measured* IC is the right amount: E[true | score]
   ≈ IC · z(score) in standardised units.

2. SAMPLE COVARIANCE IS UNUSABLE AT N≈P.
   With ~250 names and ~250 days the sample covariance matrix is nearly
   singular; its small eigenvalues are pure noise, and any optimiser will pile
   into them. Two independent fixes are applied and combined:
     - Ledoit–Wolf shrinkage toward a constant-correlation target.
     - Marchenko–Pastur (RMT) denoising: eigenvalues below the theoretical
       noise edge λ+ = σ²(1+√(p/n))² carry no information and are replaced by
       their average, preserving the trace.

3. EQUAL WEIGHT IGNORES CORRELATION.
   Ten names from one theme is one bet. Hierarchical Risk Parity (López de
   Prado) clusters the correlation matrix, then allocates by recursive
   bisection so correlated clusters share a budget — no matrix inversion, so
   it is stable where mean-variance is not.

4. FULL KELLY IS RUINOUS UNDER PARAMETER UNCERTAINTY.
   Sizing uses fractional Kelly (default 1/4) with a volatility target, which
   is the standard response to the fact that the edge itself is estimated.

Everything here is pure NumPy on arrays; no repo imports, so it is trivially
unit-testable (see ~/trade-ops/tests/).
"""
from __future__ import annotations

import numpy as np

__all__ = ["james_stein_shrink", "ledoit_wolf", "rmt_denoise", "corr_from_cov",
           "hrp_weights", "kelly_fraction", "vol_target_scale", "effective_n"]


# ── 1. forecast shrinkage ───────────────────────────────────────────────────
def james_stein_shrink(scores: np.ndarray, ic: float = 0.02,
                       min_keep: float = 0.02) -> np.ndarray:
    """Shrink cross-sectional scores toward their mean by the measured skill.

    In standardised units the posterior mean of the true alpha given a forecast
    is approximately IC * z. With IC ~ 0.02 that is aggressive — which is the
    point: it stops the ranking from trusting noise. Returns scores on the
    original scale (mean preserved, dispersion scaled).

    `min_keep` floors the retained fraction so a pathological IC estimate can't
    collapse every score to a constant (which would make the ranking arbitrary).
    """
    s = np.asarray(scores, dtype=float)
    good = ~np.isnan(s)
    if good.sum() < 3:
        return s
    mu = float(np.mean(s[good]))
    sd = float(np.std(s[good], ddof=1))
    if sd <= 0:
        return s
    keep = float(np.clip(abs(ic) / 0.10, min_keep, 1.0))   # IC 0.10 => full trust
    out = s.copy()
    out[good] = mu + (s[good] - mu) * keep
    return out


# ── 2. covariance estimation ────────────────────────────────────────────────
def ledoit_wolf(returns: np.ndarray) -> np.ndarray:
    """Ledoit–Wolf shrinkage toward a constant-correlation target.

    returns: (T, N) matrix of asset returns.
    """
    X = np.asarray(returns, dtype=float)
    X = X[~np.isnan(X).any(axis=1)]
    T, N = X.shape
    if T < 3 or N < 2:
        return np.cov(X, rowvar=False) if T > 1 else np.eye(N)
    Xc = X - X.mean(axis=0)
    S = (Xc.T @ Xc) / T
    var = np.diag(S).copy()
    var[var <= 0] = 1e-12
    sd = np.sqrt(var)
    R = S / np.outer(sd, sd)
    off = R[~np.eye(N, dtype=bool)]
    rbar = float(np.mean(off)) if off.size else 0.0
    F = rbar * np.outer(sd, sd)                 # constant-correlation target
    np.fill_diagonal(F, var)
    # shrinkage intensity (Ledoit-Wolf pi/rho/gamma decomposition, simplified)
    Y = Xc ** 2
    pi_mat = (Y.T @ Y) / T - S ** 2
    pi_hat = float(np.sum(pi_mat))
    gamma = float(np.sum((F - S) ** 2))
    if gamma <= 0:
        return S
    kappa = pi_hat / gamma
    delta = float(np.clip(kappa / T, 0.0, 1.0))
    return delta * F + (1 - delta) * S


def rmt_denoise(cov: np.ndarray, n_obs: int) -> np.ndarray:
    """Marchenko–Pastur denoising: flatten sub-noise-edge eigenvalues.

    Eigenvalues below λ+ = σ²(1+√(p/n))² are indistinguishable from those of a
    random matrix, so replacing them with their mean removes noise while
    preserving the trace (total variance).
    """
    C = np.asarray(cov, dtype=float)
    p = C.shape[0]
    if p < 2 or n_obs <= p:
        return C
    sd = np.sqrt(np.clip(np.diag(C), 1e-18, None))
    R = C / np.outer(sd, sd)
    np.fill_diagonal(R, 1.0)
    vals, vecs = np.linalg.eigh(R)
    q = p / float(n_obs)
    lam_plus = (1.0 + np.sqrt(q)) ** 2
    noise = vals < lam_plus
    if noise.sum() > 0 and (~noise).sum() > 0:
        vals = vals.copy()
        vals[noise] = float(np.mean(vals[noise]))
    R_clean = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.clip(np.diag(R_clean), 1e-18, None))
    R_clean = R_clean / np.outer(d, d)          # renormalise to unit diagonal
    np.fill_diagonal(R_clean, 1.0)
    return R_clean * np.outer(sd, sd)


def corr_from_cov(cov: np.ndarray) -> np.ndarray:
    sd = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
    R = cov / np.outer(sd, sd)
    np.fill_diagonal(R, 1.0)
    return np.clip(R, -1.0, 1.0)


# ── 3. allocation ───────────────────────────────────────────────────────────
def hrp_weights(cov: np.ndarray) -> np.ndarray:
    """Hierarchical Risk Parity weights (López de Prado), with the recursive
    bisection done on the ACTUAL DENDROGRAM TREE rather than by positional
    halving of the quasi-diagonal order.

    Textbook HRP splits the ordered list at len//2. That is only equivalent to
    the tree split when the dendrogram happens to be balanced; at small N it can
    cut straight through a tight cluster. Concretely, for three near-identical
    assets plus one independent asset the leaf order is [3,1,0,2] and positional
    bisection yields [3,1] | [0,2] — pairing the independent asset with a
    redundant one and over-weighting the redundant cluster. Splitting on the
    tree's own children keeps clusters intact.

    Cluster distance is d = sqrt((1-rho)/2); budget is split by inverse cluster
    variance. No matrix inversion, so it stays stable when N >> T.
    """
    C = np.asarray(cov, dtype=float)
    n = C.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.ones(1)
    try:
        from scipy.cluster.hierarchy import linkage, to_tree
        from scipy.spatial.distance import squareform
        R = corr_from_cov(C)
        d = np.sqrt(np.clip((1.0 - R) / 2.0, 0.0, 1.0))
        np.fill_diagonal(d, 0.0)
        root = to_tree(linkage(squareform(d, checks=False), method="single"))
    except Exception:                      # scipy missing/degenerate => IVP
        iv = 1.0 / np.clip(np.diag(C), 1e-18, None)
        return iv / iv.sum()

    w = np.ones(n)

    def split(node, budget: float) -> None:
        if node.is_leaf():
            w[node.get_id()] = budget
            return
        left, right = node.get_left(), node.get_right()
        li, ri = left.pre_order(lambda x: x.id), right.pre_order(lambda x: x.id)
        vl, vr = _cluster_var(C, li), _cluster_var(C, ri)
        # inverse-variance split: the riskier cluster receives less budget
        alpha = 1.0 - vl / (vl + vr) if (vl + vr) > 0 else 0.5
        split(left, budget * alpha)
        split(right, budget * (1.0 - alpha))

    split(root, 1.0)
    s = w.sum()
    return w / s if s > 0 else np.ones(n) / n


def _cluster_var(cov: np.ndarray, idx: list[int]) -> float:
    sub = cov[np.ix_(idx, idx)]
    iv = 1.0 / np.clip(np.diag(sub), 1e-18, None)
    w = iv / iv.sum()
    return float(w @ sub @ w)


# ── 4. sizing ───────────────────────────────────────────────────────────────
def kelly_fraction(edge: float, variance: float, fraction: float = 0.25,
                   cap: float = 1.0) -> float:
    """Fractional Kelly. Full Kelly assumes the edge is known; it never is."""
    if variance <= 0:
        return 0.0
    return float(np.clip(fraction * edge / variance, -cap, cap))


def vol_target_scale(realised_vol: float, target: float = 0.15,
                     max_leverage: float = 1.5) -> float:
    """Exposure multiplier that targets a constant portfolio volatility."""
    if realised_vol <= 1e-9:
        return 1.0
    return float(np.clip(target / realised_vol, 0.0, max_leverage))


def effective_n(weights: np.ndarray) -> float:
    """Diversification measure: 1/sum(w^2). Equals N for equal weights, and
    collapses toward 1 as the book concentrates — the number to watch."""
    w = np.asarray(weights, dtype=float)
    s = np.sum(w ** 2)
    return float(1.0 / s) if s > 0 else 0.0
