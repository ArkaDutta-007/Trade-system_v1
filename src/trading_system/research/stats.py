"""Robustness statistics for backtest evidence.

A single Sharpe ratio from a single configuration says almost nothing once you
have tried more than one configuration, which everyone has.  These are the four
corrections that turn a table of backtests into evidence:

``deflated_sharpe``
    Bailey & López de Prado's haircut: the Sharpe a *skill-less* best-of-N
    search would have produced, given how many configurations were tried and
    how dispersed their results were.  Also corrects for the non-normality
    (skew, fat tails) that inflates a naive Sharpe t-statistic.

``pbo_cscv``
    Probability of Backtest Overfitting via combinatorially symmetric cross
    validation: over many in-sample/out-of-sample splits of the *same* return
    matrix, how often does the in-sample winner land in the bottom half
    out-of-sample?  Above ~50% the selection procedure is noise.

``stationary_bootstrap``
    Politis-Romano resampling that preserves serial dependence (daily returns
    are not iid), giving honest confidence intervals for any statistic.

``spa_test``
    Hansen's Superior Predictive Ability test: is the *best* of a set of
    strategies better than a benchmark, after accounting for the whole search?
    This is the right test for "my new model beats momentum", because it
    prices the fact that you looked at many models before saying so.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy import stats

TRADING_DAYS = 252

__all__ = [
    "sharpe", "probabilistic_sharpe", "deflated_sharpe", "pbo_cscv",
    "stationary_bootstrap", "bootstrap_ci", "paired_bootstrap_ci", "spa_test",
]


def sharpe(r: np.ndarray, annualise: bool = True) -> float:
    r = np.asarray(r, dtype=float)
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd < 1e-12:
        return 0.0
    s = r.mean() / sd
    return float(s * np.sqrt(TRADING_DAYS)) if annualise else float(s)


def probabilistic_sharpe(r: np.ndarray, sr_benchmark: float = 0.0) -> float:
    """P(true Sharpe > benchmark), correcting for skew, kurtosis and sample length.

    Inputs and ``sr_benchmark`` are per-period (not annualised).
    """
    r = np.asarray(r, dtype=float)
    n = len(r)
    if n < 8:
        return 0.5
    sd = r.std(ddof=1)
    if sd < 1e-12:
        return 0.5
    sr = r.mean() / sd
    g3 = float(stats.skew(r))
    g4 = float(stats.kurtosis(r, fisher=False))
    var = (1 - g3 * sr + (g4 - 1) / 4 * sr ** 2) / (n - 1)
    if var <= 0:
        return 0.5
    return float(stats.norm.cdf((sr - sr_benchmark) / np.sqrt(var)))


def deflated_sharpe(r: np.ndarray, trial_sharpes: np.ndarray) -> dict:
    """Deflated Sharpe Ratio of ``r`` given every trial Sharpe you computed.

    ``trial_sharpes`` are per-period Sharpes of all configurations tried; their
    dispersion sets how high a skill-less maximum would have reached.
    """
    trials = np.asarray(trial_sharpes, dtype=float)
    n_trials = max(len(trials), 1)
    var_sr = float(trials.var(ddof=1)) if n_trials > 1 else 0.0
    emc = 0.5772156649
    if n_trials > 1:
        z1 = stats.norm.ppf(1 - 1.0 / n_trials)
        z2 = stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
        sr0 = np.sqrt(var_sr) * ((1 - emc) * z1 + emc * z2)
    else:
        sr0 = 0.0
    return {
        "dsr": probabilistic_sharpe(r, sr0),
        "sr0_per_period": float(sr0),
        "sr0_annual": float(sr0 * np.sqrt(TRADING_DAYS)),
        "n_trials": int(n_trials),
    }


def pbo_cscv(ret_matrix: np.ndarray, n_blocks: int = 16) -> dict:
    """Probability of Backtest Overfitting (CSCV).

    ``ret_matrix`` is ``(T, N)``: one column of per-period returns per
    configuration.  Returns the PBO plus the median out-of-sample rank of the
    in-sample winner, which is easier to read than the probability alone.
    """
    R = np.asarray(ret_matrix, dtype=float)
    T, N = R.shape
    if N < 2:
        return {"pbo": float("nan"), "median_oos_rank": float("nan"), "n_configs": N}
    n_blocks = min(n_blocks, T // 4)
    if n_blocks < 4:
        raise ValueError("series too short for CSCV")
    if n_blocks % 2:
        n_blocks -= 1

    blocks = np.array_split(np.arange(T), n_blocks)
    half = n_blocks // 2
    logits, ranks = [], []
    for combo in itertools.combinations(range(n_blocks), half):
        is_idx = np.concatenate([blocks[b] for b in combo])
        oos_idx = np.concatenate([blocks[b] for b in range(n_blocks) if b not in combo])
        s_is = R[is_idx].mean(0) / (R[is_idx].std(0, ddof=1) + 1e-12)
        s_oos = R[oos_idx].mean(0) / (R[oos_idx].std(0, ddof=1) + 1e-12)
        star = int(np.argmax(s_is))
        rel = stats.rankdata(s_oos)[star] / (N + 1)
        ranks.append(rel)
        logits.append(np.log(rel / (1 - rel)))
    logits = np.asarray(logits)
    return {
        "pbo": float(np.mean(logits <= 0)),
        "median_oos_rank": float(np.median(ranks)),
        "n_configs": int(N),
        "n_splits": len(logits),
    }


def stationary_bootstrap(
    r: np.ndarray, n_boot: int = 1000, mean_block: float = 21.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Politis-Romano stationary bootstrap: ``(n_boot, len(r))`` resamples.

    Blocks have geometric lengths with mean ``mean_block``, so autocorrelation
    and volatility clustering survive the resample.  An iid bootstrap would
    destroy both and give confidence intervals that are far too tight.
    """
    r = np.asarray(r, dtype=float)
    n = len(r)
    rng = rng or np.random.default_rng(0)
    p = 1.0 / max(mean_block, 1.0)

    out = np.empty((n_boot, n))
    for b in range(n_boot):
        idx = np.empty(n, dtype=int)
        i = rng.integers(n)
        for t in range(n):
            idx[t] = i
            i = rng.integers(n) if rng.random() < p else (i + 1) % n
        out[b] = r[idx]
    return out


def bootstrap_ci(
    r: np.ndarray, stat=sharpe, n_boot: int = 1000, mean_block: float = 21.0,
    alpha: float = 0.05, rng: np.random.Generator | None = None,
) -> dict:
    """Percentile confidence interval for ``stat`` under the stationary bootstrap."""
    draws = stationary_bootstrap(r, n_boot=n_boot, mean_block=mean_block, rng=rng)
    vals = np.array([stat(d) for d in draws])
    return {
        "point": float(stat(np.asarray(r, dtype=float))),
        "lo": float(np.quantile(vals, alpha / 2)),
        "hi": float(np.quantile(vals, 1 - alpha / 2)),
        "p_gt_0": float((vals > 0).mean()),
    }


def _stationary_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """One stationary-bootstrap index draw of length ``n``."""
    p = 1.0 / max(mean_block, 1.0)
    idx = np.empty(n, dtype=int)
    i = rng.integers(n)
    for t in range(n):
        idx[t] = i
        i = rng.integers(n) if rng.random() < p else (i + 1) % n
    return idx


def paired_bootstrap_ci(
    a: np.ndarray, b: np.ndarray, stat=sharpe, n_boot: int = 1000,
    mean_block: float = 21.0, alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> dict:
    """Confidence interval for ``stat(a) - stat(b)`` under a *paired* resample.

    Both series are resampled with the **same** index draw, which preserves the
    day-by-day pairing between two strategies run over the same dates.  That
    matters because the two are highly correlated: resampling them independently
    would inflate the variance of the difference enormously and make every
    comparison look inconclusive.

    Use this, not a difference of means, when comparing two policies on a
    risk-adjusted statistic — a policy that merely takes more risk wins on mean
    return while being no better per unit of risk.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(a), len(b))
    a, b = a[-n:], b[-n:]
    rng = rng or np.random.default_rng(0)

    vals = np.empty(n_boot)
    for k in range(n_boot):
        idx = _stationary_indices(n, mean_block, rng)
        vals[k] = stat(a[idx]) - stat(b[idx])
    return {
        "point": float(stat(a) - stat(b)),
        "lo": float(np.quantile(vals, alpha / 2)),
        "hi": float(np.quantile(vals, 1 - alpha / 2)),
        "p_gt_0": float((vals > 0).mean()),
    }


def spa_test(
    losses_benchmark: np.ndarray, losses_models: np.ndarray,
    n_boot: int = 1000, mean_block: float = 21.0,
    rng: np.random.Generator | None = None,
) -> dict:
    """Hansen's SPA test.

    Inputs are *losses* (lower is better); for returns pass ``-returns``.
    ``losses_models`` is ``(T, K)``.  Tests H0: no model beats the benchmark,
    against the alternative that the best one does — with the multiplicity of
    the whole set priced in, so a winner found by searching K models must clear
    a higher bar than a single pre-registered model.
    """
    lb = np.asarray(losses_benchmark, dtype=float)
    lm = np.asarray(losses_models, dtype=float)
    if lm.ndim == 1:
        lm = lm[:, None]
    T, K = lm.shape
    rng = rng or np.random.default_rng(0)

    d = lb[:, None] - lm            # positive = model better than benchmark
    dbar = d.mean(0)
    w = d.std(0, ddof=1) / np.sqrt(T)
    w = np.where(w < 1e-12, 1e-12, w)
    t_stat = float(np.max(np.maximum(dbar / w, 0.0)))

    # Hansen's recentring: only models that are not too far below the benchmark
    # contribute to the null, which is what makes SPA less conservative than
    # White's reality check when the set contains many bad models.
    thresh = -w * np.sqrt(2 * np.log(np.log(max(T, 3))))
    g = np.where(dbar >= thresh, dbar, 0.0)

    boot = np.empty(n_boot)
    p = 1.0 / max(mean_block, 1.0)
    for b in range(n_boot):
        idx = np.empty(T, dtype=int)
        i = rng.integers(T)
        for t in range(T):
            idx[t] = i
            i = rng.integers(T) if rng.random() < p else (i + 1) % T
        db = d[idx].mean(0) - g
        boot[b] = np.max(np.maximum(db / w, 0.0))

    return {
        "t_spa": t_stat,
        "p_value": float((boot >= t_stat).mean()),
        "best_model": int(np.argmax(dbar / w)),
        "n_models": K,
    }
