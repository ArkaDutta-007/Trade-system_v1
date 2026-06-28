"""Nonlinear-dynamics feature library — rigorous maths borrowed across domains.

The tradable hypothesis behind this module: a price path is the visible trace of
a high-dimensional dynamical system, and several mathematically rigorous
frameworks built in *other* fields turn out to be unreasonably effective at
characterising it.  Each estimator here is causal (uses only past values),
implemented in pure NumPy/SciPy (no exotic deps), and carries a one-line note on
where it comes from and why it might matter for markets.

Families
--------
Long memory / fractal geometry
    hurst_dfa            Detrended Fluctuation Analysis exponent   (physiology, geophysics)
    hurst_rs             Rescaled-range exponent                   (hydrology — Nile floods!)
    higuchi_fd           Higuchi fractal dimension of the path     (signal processing)
    rough_volatility_h   roughness of log-vol (fBm Hurst)          (rough-vol / fractional calculus)
Information theory / complexity
    permutation_entropy  Bandt–Pompe ordinal complexity            (dynamical systems, EEG)
    sample_entropy       regularity / self-similarity              (cardiology — HRV)
    spectral_entropy     flatness of the power spectrum            (information theory)
    shannon_entropy_hist disorder of the return distribution
Chaos / nonlinear predictability
    largest_lyapunov     sensitive dependence (Rosenstein)         (chaos theory)
    recurrence_metrics   determinism / laminarity (RQA)            (nonlinear dynamics)
    chaos01              Gottwald–Melbourne 0–1 test for chaos
    dominant_period      strongest spectral cycle length
Bifurcation theory (early warning of regime change)
    ar1, early_warning_score   critical slowing down              (ecology / climate tipping points)
Extreme-value / econophysics
    hill_tail_index      power-law tail exponent of |returns|      (extreme value theory)
Catastrophe theory
    lppls_confidence     log-periodic power-law bubble signature   (rupture/earthquake physics)
Multiresolution
    wavelet_hf_ratio     high-frequency energy fraction (Haar)     (wavelet analysis)

Every function returns ``np.nan`` on degenerate / too-short input rather than
raising, so they are safe to drop into a rolling apply.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ── helpers ───────────────────────────────────────────────────────────────────

def _clean(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def _embed(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    """Takens time-delay embedding → (N-(m-1)tau, m) matrix of state vectors."""
    x = np.asarray(x, dtype=float)
    n = len(x) - (m - 1) * tau
    if n <= 0:
        return np.empty((0, m))
    return np.column_stack([x[i * tau: i * tau + n] for i in range(m)])


# ── long memory / fractal ──────────────────────────────────────────────────────

def hurst_dfa(x) -> float:
    """Hurst exponent via Detrended Fluctuation Analysis (DFA-1).

    H≈0.5 random walk, H>0.5 persistent/trending, H<0.5 mean-reverting.
    Apply to *returns*.  From physiology/geophysics — robust to nonstationary
    trends, which is exactly why it travels well to noisy price data.
    """
    x = _clean(x)
    N = len(x)
    if N < 32:
        return np.nan
    y = np.cumsum(x - x.mean())                       # integrated profile
    scales = np.unique(np.floor(np.logspace(np.log10(4), np.log10(max(8, N // 4)), 12)).astype(int))
    Fs, used = [], []
    for s in scales:
        if s < 4:
            continue
        nseg = N // s
        if nseg < 2:
            continue
        seg = y[:nseg * s].reshape(nseg, s).astype(float)
        t = np.arange(s)
        tc = t - t.mean()
        denom = float((tc ** 2).sum())
        slope = (seg * tc).sum(axis=1) / denom        # vectorised per-segment linear detrend
        intercept = seg.mean(axis=1) - slope * t.mean()
        fit = slope[:, None] * t[None, :] + intercept[:, None]
        rms = np.sqrt(((seg - fit) ** 2).mean(axis=1))
        Fs.append(float(np.sqrt((rms ** 2).mean())))
        used.append(s)
    Fs, used = np.array(Fs), np.array(used)
    mask = Fs > 0                                      # drop degenerate (constant) scales
    if mask.sum() < 3:
        return np.nan
    return float(np.polyfit(np.log(used[mask]), np.log(Fs[mask]), 1)[0])


def hurst_rs(x) -> float:
    """Hurst exponent via classic Rescaled-Range (R/S) analysis. Apply to returns.

    Mandelbrot/Hurst's original tool — invented for Nile river flood levels, then
    found to describe market persistence.  The canonical 'weirdly transferable'.
    """
    x = _clean(x)
    N = len(x)
    if N < 32:
        return np.nan
    ns = np.unique(np.floor(np.logspace(np.log10(8), np.log10(max(16, N // 2)), 10)).astype(int))
    RS, used = [], []
    for n in ns:
        if n < 8:
            continue
        nch = N // n
        if nch < 1:
            continue
        vals = []
        for k in range(nch):
            c = x[k * n:(k + 1) * n]
            Z = np.cumsum(c - c.mean())
            S = c.std()
            if S > 1e-12:
                vals.append((Z.max() - Z.min()) / S)
        if vals:
            RS.append(float(np.mean(vals)))
            used.append(n)
    if len(used) < 3:
        return np.nan
    return float(np.polyfit(np.log(used), np.log(RS), 1)[0])


def higuchi_fd(x, kmax: int = 10) -> float:
    """Higuchi fractal dimension of the curve (≈1 smooth, ≈2 noisy). Apply to log-price.

    A direct measure of how 'jagged' the graph is at multiple scales.
    """
    x = _clean(x)
    N = len(x)
    if N < 10:
        return np.nan
    kmax = max(2, min(kmax, (N - 1) // 2))
    Lk, ks = [], []
    for k in range(1, kmax + 1):
        Lm = []
        for m in range(k):
            idx = np.arange(m, N, k)
            if len(idx) < 2:
                continue
            length = np.abs(np.diff(x[idx])).sum() * (N - 1) / ((len(idx) - 1) * k)
            Lm.append(length / k)
        if Lm:
            Lk.append(np.mean(Lm))
            ks.append(k)
    if len(ks) < 3:
        return np.nan
    return float(np.polyfit(np.log(1.0 / np.array(ks)), np.log(Lk), 1)[0])


def rough_volatility_hurst(vol, max_lag: int | None = None, q: float = 2.0) -> float:
    """Roughness (Hurst) of log-volatility via its q-th structure function.

    Gatheral–Jaisson–Rosenbaum: empirically H≈0.1 ('volatility is rough').
    ``vol`` is a positive spot-vol proxy.  Lower H = rougher = more erratic
    vol-of-vol.  From fractional Brownian motion theory.
    """
    v = _clean(vol)
    v = v[v > 0]
    N = len(v)
    if N < 30:
        return np.nan
    lv = np.log(v)
    max_lag = max_lag or min(20, N // 4)
    lags = np.arange(1, max_lag + 1)
    m = np.array([np.mean(np.abs(lv[L:] - lv[:-L]) ** q) for L in lags])
    good = m > 0
    if good.sum() < 4:
        return np.nan
    slope = np.polyfit(np.log(lags[good]), np.log(m[good]), 1)[0]
    return float(slope / q)                           # ζ_q = qH


# ── information theory / complexity ─────────────────────────────────────────────

def permutation_entropy(x, m: int = 3, tau: int = 1) -> float:
    """Bandt–Pompe permutation entropy, normalised to [0,1]. Apply to returns.

    Counts ordinal patterns of length m; 0 = perfectly predictable order, 1 =
    maximal disorder.  Cheap, robust, amplitude-invariant.
    """
    x = _clean(x)
    emb = _embed(x, m, tau)
    if len(emb) < 2:
        return np.nan
    perms = np.argsort(emb, axis=1)
    _, counts = np.unique(perms, axis=0, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p)) / np.log(math.factorial(m)))


def sample_entropy(x, m: int = 2, r: float = 0.2) -> float:
    """Sample entropy (Richman–Moorman). Apply to returns.

    Lower = more regular / self-similar / predictable.  Born in cardiology
    (heart-rate variability); ``r`` is the match tolerance as a fraction of σ.
    """
    x = _clean(x)
    N = len(x)
    if N < m + 2:
        return np.nan
    r_abs = r * np.std(x)
    if r_abs <= 0:
        return np.nan

    def _count(mm):
        tem = _embed(x, mm, 1)
        M = len(tem)
        c = 0
        for i in range(M - 1):
            d = np.max(np.abs(tem[i + 1:] - tem[i]), axis=1)
            c += int(np.sum(d <= r_abs))
        return c

    B, A = _count(m), _count(m + 1)
    if B == 0 or A == 0:
        return np.nan
    return float(-np.log(A / B))


def _psd(x):
    x = _clean(x)
    x = x - x.mean()
    if len(x) < 8 or np.allclose(x, 0):
        return None, None
    f = np.fft.rfftfreq(len(x))
    P = np.abs(np.fft.rfft(x)) ** 2
    return f[1:], P[1:]                                # drop DC


def spectral_entropy(x) -> float:
    """Shannon entropy of the normalised power spectrum, in [0,1]. Apply to returns.

    ~0 = a single dominant cycle (very structured), ~1 = white-noise-flat.
    """
    _, P = _psd(x)
    if P is None or P.sum() <= 0:
        return np.nan
    p = P / P.sum()
    return float(-np.sum(p * np.log(p + 1e-12)) / np.log(len(p)))


def dominant_period(x) -> float:
    """Length (in samples) of the strongest spectral cycle. Apply to returns."""
    f, P = _psd(x)
    if P is None or len(P) == 0:
        return np.nan
    fpk = f[int(np.argmax(P))]
    return float(1.0 / fpk) if fpk > 0 else np.nan


def shannon_entropy_hist(x, bins: int = 16) -> float:
    """Normalised Shannon entropy of the binned return distribution, in [0,1]."""
    x = _clean(x)
    if len(x) < bins:
        return np.nan
    h, _ = np.histogram(x, bins=bins)
    p = h[h > 0] / h.sum()
    return float(-np.sum(p * np.log(p)) / np.log(bins))


# ── chaos / nonlinear predictability ────────────────────────────────────────────

def largest_lyapunov(x, m: int = 4, tau: int = 1, theiler: int | None = None,
                     max_t: int | None = None, fit_frac: float = 0.5) -> float:
    """Largest Lyapunov exponent (Rosenstein). Apply to returns.

    >0 ⇒ sensitive dependence on initial conditions ⇒ short predictability
    horizon (deterministic chaos).  The canonical chaos-theory quantity.
    """
    from scipy.spatial.distance import cdist

    x = _clean(x)
    emb = _embed(x, m, tau)
    M = len(emb)
    if M < 20:
        return np.nan
    theiler = theiler if theiler is not None else max(1, tau * m)
    D = cdist(emb, emb)
    np.fill_diagonal(D, np.inf)
    for off in range(1, theiler + 1):                  # mask temporal neighbours
        i = np.arange(M - off)
        D[i, i + off] = np.inf
        D[i + off, i] = np.inf
    nn = np.argmin(D, axis=1)
    max_t = max_t if max_t is not None else min(M // 2, 20)
    div = np.full(max_t, np.nan)
    for t in range(max_t):
        d = []
        for i in range(M - t):
            j = nn[i]
            if j + t < M:
                dist = np.linalg.norm(emb[i + t] - emb[j + t])
                if dist > 0:
                    d.append(np.log(dist))
        if d:
            div[t] = np.mean(d)
    valid = np.isfinite(div)
    ts = np.arange(max_t)[valid]
    dv = div[valid]
    if len(ts) < 4:
        return np.nan
    cut = max(3, int(len(ts) * fit_frac))
    return float(np.polyfit(ts[:cut], dv[:cut], 1)[0])


def _line_fraction(R: np.ndarray, l_min: int, diagonal: bool) -> float:
    total = int(R.sum())
    if total == 0:
        return 0.0
    M = R.shape[0]
    if diagonal:
        lines = [np.diag(R, k) for k in range(1, M)] + [np.diag(R, -k) for k in range(1, M)]
    else:
        lines = [R[:, j] for j in range(M)]
    on = 0
    for line in lines:
        run = 0
        for v in line:
            if v:
                run += 1
            else:
                if run >= l_min:
                    on += run
                run = 0
        if run >= l_min:
            on += run
    return float(on / total)


def recurrence_metrics(x, m: int = 3, tau: int = 1, rr_target: float = 0.15,
                       l_min: int = 2) -> dict:
    """Recurrence Quantification Analysis. Apply to returns.

    DET = fraction of recurrence points on diagonal lines (predictability /
    determinism); LAM = fraction on vertical lines (laminar / sticky states).
    From nonlinear-dynamics recurrence plots (Eckmann, Marwan).
    """
    from scipy.spatial.distance import pdist, squareform

    x = _clean(x)
    emb = _embed(x, m, tau)
    M = len(emb)
    if M < 10:
        return {"det": np.nan, "lam": np.nan, "rr": np.nan}
    D = squareform(pdist(emb))
    eps = float(np.quantile(D[np.triu_indices(M, 1)], rr_target))
    R = (D <= eps).astype(int)
    np.fill_diagonal(R, 0)
    rr = R.sum() / (M * (M - 1))
    return {"det": _line_fraction(R, l_min, True),
            "lam": _line_fraction(R, l_min, False),
            "rr": float(rr)}


def recurrence_determinism(x, **kw) -> float:
    return recurrence_metrics(x, **kw)["det"]


def chaos01(x, n_c: int = 30, seed: int = 0) -> float:
    """Gottwald–Melbourne 0–1 test for chaos → K in [0,1]. Apply to returns.

    0 = regular (periodic/quasi-periodic), 1 = chaotic/diffusive.  Model-free:
    no embedding dimension to choose.
    """
    x = _clean(x)
    N = len(x)
    if N < 30:
        return np.nan
    rng = np.random.default_rng(seed)
    phi = x - x.mean()
    nmax = max(5, N // 10)
    j = np.arange(1, N + 1)
    nn = np.arange(1, nmax + 1)
    Ks = []
    for c in rng.uniform(np.pi / 5, 4 * np.pi / 5, size=n_c):
        p = np.cumsum(phi * np.cos(j * c))
        q = np.cumsum(phi * np.sin(j * c))
        Mc = np.array([np.mean((p[n:] - p[:-n]) ** 2 + (q[n:] - q[:-n]) ** 2) for n in nn])
        Vosc = (phi.mean() ** 2) * (1 - np.cos(nn * c)) / (1 - np.cos(c))
        D = Mc - Vosc
        if np.std(D) < 1e-12:
            Ks.append(0.0)
        else:
            Ks.append(float(np.corrcoef(nn, D)[0, 1]))
    return float(np.clip(np.median(Ks), 0.0, 1.0))


# ── bifurcation theory: critical slowing down (early-warning signals) ───────────

def ar1(x) -> float:
    """Lag-1 autocorrelation. Rising AR1 = 'critical slowing down' near a tipping point."""
    x = _clean(x)
    if len(x) < 3 or np.std(x) < 1e-12:
        return np.nan
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def _kendall_tau_vs_index(s) -> float:
    s = _clean(s)
    n = len(s)
    if n < 4:
        return np.nan
    num = den = 0
    for i in range(n):
        for k in range(i + 1, n):
            num += np.sign(s[k] - s[i])               # index always increases
            den += 1
    return float(num / den) if den else np.nan


def early_warning_score(x, n_sub: int = 10, overlap: float = 0.5) -> float:
    """Composite critical-slowing-down score in [-1,1]. Apply to returns.

    Splits the window into overlapping sub-windows, measures the Kendall-τ trend
    of lag-1 autocorrelation and of variance, and averages them.  Rising AR1 +
    rising variance (→ +1) is the generic early-warning signature of an
    approaching bifurcation — transferred from ecology/climate tipping points
    (Scheffer et al., Nature 2009).
    """
    x = _clean(x)
    N = len(x)
    if N < 40:
        return np.nan
    w = int(N / (1 + (n_sub - 1) * (1 - overlap)))
    if w < 10:
        w = N // 4
    step = max(1, int(w * (1 - overlap)))
    ar, va = [], []
    i = 0
    while i + w <= N:
        seg = x[i:i + w]
        ar.append(ar1(seg))
        va.append(float(np.var(seg)))
        i += step
    if len(ar) < 4:
        return np.nan
    t_ar = _kendall_tau_vs_index(ar)
    t_va = _kendall_tau_vs_index(va)
    return float(np.nanmean([t_ar, t_va]))


# ── extreme-value / econophysics ────────────────────────────────────────────────

def hill_tail_index(x, k_frac: float = 0.1) -> float:
    """Hill estimator of the power-law tail exponent α of |returns|.

    Lower α = fatter tails = more crash/melt-up prone.  Equities typically α≈3–4.
    From extreme value theory.
    """
    a = np.abs(_clean(x))
    a = a[a > 0]
    n = len(a)
    if n < 20:
        return np.nan
    k = min(max(5, int(k_frac * n)), n - 1)
    s = np.sort(a)[::-1]
    xk = s[k]
    if xk <= 0:
        return np.nan
    return float(1.0 / np.mean(np.log(s[:k] / xk)))


# ── catastrophe theory: bubbles ─────────────────────────────────────────────────

def lppls_confidence(logp, n_samples: int = 60, seed: int = 0) -> float:
    """Log-Periodic Power-Law Singularity confidence in [-1,1]. Apply to log-price.

    Sornette's bubble model: ln p(t) = A + B(tc−t)^m + C(tc−t)^m cos(ω ln(tc−t)−φ),
    a finite-time singularity dressed with accelerating log-periodic oscillations.
    For fixed (tc,m,ω) the rest is linear (OLS); we random-search the nonlinear
    parameters over Sornette's *validated* ranges (0.1≤m≤0.9, 6≤ω≤13).

    Raw R² is a poor detector — a flexible 4-parameter model fits almost anything,
    a noisy straight line included.  A bubble is specifically *faster*-than-
    exponential growth, so we report the LPPLS fit's improvement over a
    constant-exponential-growth baseline (linear in log-price), as a fraction of
    that baseline's residual variance.  A clean exponential up-trend therefore
    scores ~0; only genuine super-exponential + log-periodic structure scores
    high.  Sign encodes direction (B<0 ⇒ +, bubble; B>0 ⇒ −, anti-bubble).
    Borrowed from material-rupture / earthquake physics.
    """
    y = _clean(logp)
    N = len(y)
    if N < 40:
        return np.nan
    t = np.arange(N, dtype=float)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        return 0.0

    # Baseline: constant exponential growth (a straight line in log-price).
    Xl = np.column_stack([np.ones(N), t])
    cl, *_ = np.linalg.lstsq(Xl, y, rcond=None)
    r2_lin = 1.0 - float(np.sum((y - Xl @ cl) ** 2)) / ss_tot
    headroom = 1.0 - r2_lin
    if headroom < 0.02:                                # already a clean exponential → not a bubble
        return 0.0

    rng = np.random.default_rng(seed)
    best_r2, best_B = -np.inf, 0.0
    for _ in range(n_samples):
        m = rng.uniform(0.1, 0.9)
        omega = rng.uniform(6.0, 13.0)
        tc = (N - 1) + rng.uniform(0.5, 0.2 * N)       # critical time just beyond the window
        tau = tc - t
        if np.any(tau <= 0):
            continue
        tm = tau ** m
        lnt = np.log(tau)
        X = np.column_stack([np.ones(N), tm, tm * np.cos(omega * lnt), tm * np.sin(omega * lnt)])
        try:
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        except Exception:
            continue
        r2 = 1.0 - float(np.sum((y - X @ coef) ** 2)) / ss_tot
        if r2 > best_r2:
            best_r2, best_B = r2, float(coef[1])
    if not np.isfinite(best_r2):
        return 0.0
    rel = float(np.clip((best_r2 - r2_lin) / headroom, 0.0, 1.0))  # variance explained beyond trend
    sign = -np.sign(best_B)
    return float(np.clip(sign * rel, -1.0, 1.0))


# ── multiresolution ─────────────────────────────────────────────────────────────

def wavelet_hf_ratio(x, levels: int = 3) -> float:
    """Fraction of energy in high-frequency Haar wavelet detail coefficients.

    High = choppy/noisy tape, low = smooth/trending.  Pure-NumPy Haar transform
    (no PyWavelets dependency).  Apply to returns.
    """
    a = _clean(x)
    n = len(a)
    if n < 4:
        return np.nan
    levels = max(1, min(levels, int(np.log2(n)) - 1))
    hf = 0.0
    for _ in range(levels):
        L = len(a) - (len(a) % 2)
        if L < 2:
            break
        a = a[:L]
        even, odd = a[0::2], a[1::2]
        detail = (even - odd) / np.sqrt(2)
        hf += float(np.sum(detail ** 2))
        a = (even + odd) / np.sqrt(2)
    total = hf + float(np.sum(a ** 2))
    return float(hf / total) if total > 0 else np.nan


# ── feature registry (drives the rolling builder + reserve) ──────────────────────

@dataclass(frozen=True)
class NLFeature:
    name: str
    func: object               # callable on a 1-D window
    source: str                # "logret" | "logprice" | "volproxy"
    window: int
    stride: int                # recompute every `stride` rows; forward-fill between
    group: str                 # reserve group


# Fast, well-behaved tier (default in the feature build).
FAST_FEATURES: list[NLFeature] = [
    NLFeature("hurst_dfa_120",        hurst_dfa,            "logret",   120, 5,  "fractal"),
    NLFeature("hurst_rs_120",         hurst_rs,             "logret",   120, 5,  "fractal"),
    NLFeature("higuchi_fd_120",       higuchi_fd,           "logprice", 120, 5,  "fractal"),
    NLFeature("rough_vol_h_120",      rough_volatility_hurst, "volproxy", 120, 5, "fractal"),
    NLFeature("perm_entropy_60",      permutation_entropy,  "logret",    60, 3,  "entropy"),
    NLFeature("spectral_entropy_60",  spectral_entropy,     "logret",    60, 3,  "entropy"),
    NLFeature("wavelet_hf_60",        wavelet_hf_ratio,     "logret",    60, 3,  "entropy"),
    NLFeature("dominant_period_120",  dominant_period,      "logret",   120, 5,  "entropy"),
    NLFeature("hill_tail_alpha_250",  hill_tail_index,      "logret",   250, 5,  "tail"),
    NLFeature("ews_composite_120",    early_warning_score,  "logret",   120, 5,  "earlywarning"),
]

# Heavier tier (O(W²) / nonlinear fits) — opt-in.
DEEP_FEATURES: list[NLFeature] = [
    NLFeature("sample_entropy_60",    sample_entropy,         "logret",  60,  5,  "chaos"),
    NLFeature("lyapunov_60",          largest_lyapunov,       "logret",  60,  5,  "chaos"),
    NLFeature("rqa_determinism_60",   recurrence_determinism, "logret",  60,  5,  "chaos"),
    NLFeature("chaos01_120",          chaos01,                "logret",  120, 5,  "chaos"),
    NLFeature("lppls_conf_250",       lppls_confidence,       "logprice", 250, 21, "earlywarning"),
]

ALL_FEATURES: list[NLFeature] = FAST_FEATURES + DEEP_FEATURES
