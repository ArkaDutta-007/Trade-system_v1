"""Turn the nonlinear estimators into an interpreted 'fingerprint' of one name.

Drives ``ts complexity TICKER``: compute every estimator on the latest trailing
windows and attach a plain-English reading + a risk flag, so the maths becomes a
desk-usable artifact rather than a column of numbers.
"""
from __future__ import annotations

import numpy as np

from . import nonlinear as nl


def _rolling_std(a: np.ndarray, w: int = 5) -> np.ndarray:
    out = np.full(len(a), np.nan)
    for i in range(len(a)):
        seg = a[max(0, i - w + 1): i + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg) >= 2:
            out[i] = seg.std()
    return out


def _band(v, lo, hi, low_txt, mid_txt, hi_txt):
    if not np.isfinite(v):
        return "n/a", ""
    if v < lo:
        return low_txt, ""
    if v > hi:
        return hi_txt, ""
    return mid_txt, ""


def fingerprint(close: np.ndarray) -> list[dict]:
    """Return ordered rows: {domain, metric, source, value, reading, flag}.

    ``close`` is a 1-D adjusted-close series (oldest→newest). Uses each metric's
    natural trailing window.
    """
    close = np.asarray(close, dtype=float)
    close = close[np.isfinite(close) & (close > 0)]
    logp = np.log(close)
    logret = np.diff(logp)
    vol = _rolling_std(logret, 5)
    rows: list[dict] = []

    def add(domain, metric, source, value, reading, flag=""):
        rows.append({"domain": domain, "metric": metric, "source": source,
                     "value": value, "reading": reading, "flag": flag})

    def tail(a, n):
        a = a[np.isfinite(a)]
        return a[-n:] if len(a) >= n else a

    # ── Fractal / long memory ────────────────────────────────────────────────
    h = nl.hurst_dfa(tail(logret, 120))
    r, _ = _band(h, 0.45, 0.55, "mean-reverting (fade moves)",
                 "≈ random walk (efficient)", "persistent / trending")
    add("Fractal · long memory", "Hurst (DFA, 120d)", "chaos theory / geophysics", h, r)
    hr = nl.hurst_rs(tail(logret, 120))
    r, _ = _band(hr, 0.45, 0.55, "anti-persistent", "≈ random walk", "persistent")
    add("Fractal · long memory", "Hurst (R/S, 120d)", "hydrology (Nile floods)", hr, r)
    fd = nl.higuchi_fd(tail(logp, 120))
    r, _ = _band(fd, 1.3, 1.6, "smooth path (clean trend)", "moderately jagged", "very jagged / noisy")
    add("Fractal · long memory", "Higuchi FD (120d)", "signal processing", fd, r)
    rv = nl.rough_volatility_hurst(tail(vol, 120))
    r, _ = _band(rv, 0.15, 0.35, "very rough vol (erratic)", "rough vol (typical)", "smooth vol")
    add("Fractal · long memory", "Rough-vol H (120d)", "rough volatility / fBm", rv, r)

    # ── Information / complexity ──────────────────────────────────────────────
    pe = nl.permutation_entropy(tail(logret, 60))
    r, _ = _band(pe, 0.6, 0.85, "structured / predictable order", "moderately complex", "near-random")
    add("Information · complexity", "Permutation entropy (60d)", "dynamical systems / EEG", pe, r)
    se = nl.sample_entropy(tail(logret, 60))
    add("Information · complexity", "Sample entropy (60d)", "cardiology (HRV)", se,
        "lower = more regular/self-similar")
    sp = nl.spectral_entropy(tail(logret, 60))
    r, _ = _band(sp, 0.5, 0.85, "cyclical (dominant frequency)", "mixed spectrum", "broadband / noisy")
    add("Information · complexity", "Spectral entropy (60d)", "information theory", sp, r)
    dp = nl.dominant_period(tail(logret, 120))
    add("Information · complexity", "Dominant cycle (120d)", "spectral analysis", dp,
        f"≈ {dp:.0f}-day cycle" if np.isfinite(dp) else "n/a")
    wv = nl.wavelet_hf_ratio(tail(logret, 60))
    r, _ = _band(wv, 0.5, 0.75, "smooth / low-frequency", "balanced", "choppy / high-frequency")
    add("Information · complexity", "Wavelet HF ratio (60d)", "wavelet analysis", wv, r)

    # ── Chaos / predictability ────────────────────────────────────────────────
    ly = nl.largest_lyapunov(tail(logret, 60))
    if not np.isfinite(ly):
        r = "n/a"
    elif ly > 0.02:
        r = "positive → chaotic, short predictability horizon"
    elif ly < -0.02:
        r = "negative → contracting / stable"
    else:
        r = "≈ 0 → edge of stability"
    add("Chaos · predictability", "Lyapunov exp. (60d)", "chaos theory", ly, r,
        "⚠" if (np.isfinite(ly) and ly > 0.05) else "")
    det = nl.recurrence_determinism(tail(logret, 60))
    r, _ = _band(det, 0.4, 0.8, "stochastic", "mixed determinism", "highly deterministic")
    add("Chaos · predictability", "RQA determinism (60d)", "nonlinear dynamics", det, r)
    k = nl.chaos01(tail(logret, 120))
    r, _ = _band(k, 0.3, 0.6, "regular / periodic", "transitional", "chaotic / diffusive")
    add("Chaos · predictability", "0–1 chaos test (120d)", "Gottwald–Melbourne", k, r)

    # ── Tail / extreme risk ───────────────────────────────────────────────────
    al = nl.hill_tail_index(tail(logret, 250))
    if not np.isfinite(al):
        r, flag = "n/a", ""
    elif al < 2.5:
        r, flag = "very fat tails — extreme-move prone", "⚠"
    elif al < 3.5:
        r, flag = "fat tails (typical equity)", ""
    else:
        r, flag = "thinner tails", ""
    add("Tail · extreme risk", "Hill tail α (250d)", "extreme value theory", al, r, flag)

    # ── Early warning / bubbles ───────────────────────────────────────────────
    ew = nl.early_warning_score(tail(logret, 120))
    if not np.isfinite(ew):
        r, flag = "n/a", ""
    elif ew > 0.4:
        r, flag = "⚠ rising fragility (critical slowing down)", "⚠"
    elif ew < -0.4:
        r, flag = "stabilising", ""
    else:
        r, flag = "neutral", ""
    add("Early warning · regime", "Crit. slowing down (120d)", "ecology/climate tipping", ew, r, flag)
    lp = nl.lppls_confidence(tail(logp, 250), n_samples=80)
    if not np.isfinite(lp):
        r, flag = "n/a", ""
    elif lp > 0.5:
        r, flag = "⚠ bubble signature (faster-than-exponential)", "⚠"
    elif lp < -0.5:
        r, flag = "⚠ anti-bubble / negative spike risk", "⚠"
    else:
        r, flag = "no bubble signature", ""
    add("Early warning · regime", "LPPLS confidence (250d)", "rupture/earthquake physics", lp, r, flag)

    return rows


def synthesis(rows: list[dict]) -> str:
    """One-paragraph read combining the most salient signals."""
    by = {r["metric"]: r["value"] for r in rows}
    flags = [r for r in rows if r["flag"]]
    parts = []

    h = by.get("Hurst (DFA, 120d)")
    if h is not None and np.isfinite(h):
        if h > 0.55:
            parts.append(f"persistent/trending tape (H={h:.2f})")
        elif h < 0.45:
            parts.append(f"mean-reverting tape (H={h:.2f})")
        else:
            parts.append(f"near-efficient random walk (H={h:.2f})")

    al = by.get("Hill tail α (250d)")
    if al is not None and np.isfinite(al):
        parts.append(f"tail α={al:.1f}{' (fat)' if al < 3.5 else ''}")

    warn_txt = ""
    if flags:
        warn_txt = " ⚠ Watch: " + "; ".join(f"{r['metric']} — {r['reading'].lstrip('⚠ ')}" for r in flags)

    head = "; ".join(parts) if parts else "insufficient history"
    return head + "." + warn_txt
