#!/usr/bin/env python3
"""Automated daily derivation of all five playbook flags (O/F/I/S/C).

Replaces hand-maintained overrides that go stale (F and C sat at as_of
2026-06-10 for three months and printed "⚠stale" in every brief).

MATHEMATICAL APPROACH — each flag is a *state estimate*, not a threshold test:

  1. STANDARDISE. Compare a level to its own trailing distribution (robust
     z-score using median/MAD, not mean/sd — oil and vol series have fat tails
     and outliers would otherwise dominate the scale).
  2. CONFIRM WITH TREND. A level alone is ambiguous: Brent at $98 rising is a
     different regime from $98 falling. Every flag combines a level term with a
     Theil–Sen slope (median of pairwise slopes — robust to spikes, unlike OLS).
  3. COMBINE ON A CONTINUOUS SCORE. Terms are summed into a signed score in
     roughly [-3, +3] where negative = risk-off. Colour comes from cutting that
     score, so partial evidence accumulates instead of one variable deciding.
  4. HYSTERESIS. A flag only changes colour if the score clears the boundary by
     HYST margin, otherwise it holds yesterday's state. Without this the flags
     chatter on noise and the playbook's "deploy in halves" logic thrashes.

Sources: FRED (DGS2/FEDFUNDS/CPILFESL) via the repo's cached fred_series, and
the local bronze OHLCV panel for the price-based baskets — so a normal run
needs no network beyond FRED's 12h cache.

Writes ~/trade-ops/flags/auto_overrides.yaml (OUTSIDE the repo — the tracked
configs/flag_overrides.yaml is never touched, so `git pull` stays clean).
Read it via ~/trade-ops/flags/local_config.yaml, whose playbook.overrides is an
absolute path (pathlib makes `project_root / "/abs"` resolve to "/abs").

Usage:  python3 auto_flags.py [--dry-run] [--json]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import add_repo_to_path, ops_root  # noqa: E402

REPO = add_repo_to_path()

import polars as pl  # noqa: E402

HERE = Path(__file__).parent
STATE_DIR = ops_root() / "flags"                 # ~/trade-ops/flags on the RIT box
STATE_DIR.mkdir(parents=True, exist_ok=True)
OUT_YAML = STATE_DIR / "auto_overrides.yaml"   # <- local_config.yaml points here
STATE_JSON = STATE_DIR / "flag_state.json"      # yesterday's colours, for hysteresis
HYST = 0.25                                 # score margin required to flip

# Bump whenever a flag's SCORING FORMULA changes. Hysteresis deliberately holds
# yesterday's colour, but that must not perpetuate a state produced by a formula
# that no longer exists: when F was recalibrated from an absolute spread cut to a
# 5y robust z (2026-09-08), the stale RED survived a correct YELLOW score of
# -0.61 and kept the playbook in "defensives only". Changing this string clears
# the remembered colours so the new formula establishes its own baseline.
FORMULA_VERSION = "2026-09-08.f-relative-z"

# Baskets. Deliberately equal-weight: cap-weighting would make the whole
# "semi tape" flag a proxy for NVDA alone.
SEMI_BASKET = ["NVDA", "AMD", "AVGO", "LRCX", "AMAT", "KLAC", "MU", "TSM", "ASML"]
CAPEX_SUPPLY = ["NVDA", "AVGO", "ASML", "LRCX", "AMAT", "VRT", "GEV"]
CAPEX_SPEND = ["MSFT", "GOOGL", "AMZN", "META"]
BENCH = "SPY"


# ── robust statistics ───────────────────────────────────────────────────────
def robust_z(x: np.ndarray, value: float) -> float:
    """Median/MAD z-score. Fat-tailed macro series break mean/sd scaling."""
    x = x[~np.isnan(x)]
    if len(x) < 10:
        return 0.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if mad <= 0:
        return 0.0
    return (value - med) / (1.4826 * mad)


def theil_sen(y: np.ndarray) -> float:
    """Median pairwise slope per period — robust trend, ignores spikes.

    Subsampled to keep it O(n) on long series; exact for the short windows used
    here (<= 120 points => <= 7140 pairs).
    """
    y = y[~np.isnan(y)]
    n = len(y)
    if n < 5:
        return 0.0
    idx = np.arange(n)
    slopes = []
    step = max(1, n // 120)
    for i in range(0, n - 1, step):
        d = idx[i + 1:] - idx[i]
        slopes.append((y[i + 1:] - y[i]) / d)
    if not slopes:
        return 0.0
    return float(np.median(np.concatenate(slopes)))


def norm_slope(y: np.ndarray) -> float:
    """Theil-Sen slope expressed in 'MADs per 20 periods' so it is unit-free."""
    s = theil_sen(y)
    yy = y[~np.isnan(y)]
    if len(yy) < 5:
        return 0.0
    mad = float(np.median(np.abs(yy - np.median(yy)))) or 1e-9
    return (s * 20.0) / (1.4826 * mad)


def cut(score: float, prev: str | None, lo: float = -0.75, hi: float = 0.75) -> str:
    """Map a continuous score to GREEN/YELLOW/RED with hysteresis."""
    raw = "GREEN" if score >= hi else ("RED" if score <= lo else "YELLOW")
    if prev is None or prev == raw:
        return raw
    # require clearing the boundary by HYST to leave the previous state
    if prev == "YELLOW":
        if raw == "GREEN" and score < hi + HYST:
            return "YELLOW"
        if raw == "RED" and score > lo - HYST:
            return "YELLOW"
    if prev == "GREEN" and score > hi - HYST:
        return "GREEN"
    if prev == "RED" and score < lo + HYST:
        return "RED"
    return raw


# ── data access ─────────────────────────────────────────────────────────────
def bronze() -> pl.DataFrame:
    return pl.read_parquet(REPO / "data/bronze/ohlcv_daily.parquet",
                           columns=["date", "ticker", "adj_close"])


def series_for(px: pl.DataFrame, ticker: str, n: int = 300) -> np.ndarray:
    s = (px.filter(pl.col("ticker") == ticker).sort("date")["adj_close"]
           .to_numpy().astype(float))
    return s[-n:] if len(s) else np.array([])


def basket_curve(px: pl.DataFrame, tickers: list[str], n: int = 300) -> np.ndarray:
    """Equal-weight normalised basket: mean of each name's price / its first px."""
    cols = []
    for t in tickers:
        s = series_for(px, t, n)
        if len(s) >= n * 0.8 and s[0] > 0:
            cols.append(s[-int(min(n, len(s))):] / s[-int(min(n, len(s)))])
    if not cols:
        return np.array([])
    m = min(len(c) for c in cols)
    return np.mean(np.vstack([c[-m:] for c in cols]), axis=0)


def fred(sid: str):
    from trading_system.flags.datafeed import fred_series
    try:
        r = fred_series(sid, cache_dir=REPO / "data/silver/macro_cache")
        return r.df
    except Exception as e:  # noqa: BLE001
        print(f"  ! FRED {sid} failed: {e}", file=sys.stderr)
        return None


# ── the five flags ──────────────────────────────────────────────────────────
BRENT_CACHE = STATE_DIR / "brent_cache.parquet"


def brent_series(n: int = 500) -> np.ndarray:
    """Brent front-month. Not in the local bronze panel (equities only), so it
    is fetched from yfinance and cached for 6h — one small request per day."""
    import time
    if BRENT_CACHE.exists() and (time.time() - BRENT_CACHE.stat().st_mtime) < 6 * 3600:
        return pl.read_parquet(BRENT_CACHE)["close"].to_numpy().astype(float)[-n:]
    try:
        import yfinance as yf
        for sym in ("BZ=F", "CL=F"):
            h = yf.Ticker(sym).history(period="2y", interval="1d")
            if h is not None and len(h) > 60:
                v = h["Close"].to_numpy().astype(float)
                pl.DataFrame({"close": v}).write_parquet(BRENT_CACHE)
                return v[-n:]
    except Exception as e:  # noqa: BLE001
        print(f"  ! Brent fetch failed: {e}", file=sys.stderr)
    if BRENT_CACHE.exists():   # stale cache beats no reading
        return pl.read_parquet(BRENT_CACHE)["close"].to_numpy().astype(float)[-n:]
    return np.array([])


def flag_O(px, prev):
    """Oil / Iran. Level vs its own 2y distribution + trend confirmation.

    Risk is HIGH oil, so the score is negated: expensive & rising => RED.
    """
    s = brent_series(500)
    if len(s) < 60:
        return None
    last = float(s[-1])
    z = robust_z(s, last)                       # + = expensive vs own history
    tr = norm_slope(s[-60:])                    # + = rising
    # The playbook's $85/$105 bands encode domain judgement about Hormuz/Iran
    # risk and drive rule 1.4 ("any flag RED -> defensives only"). Treat them as
    # HARD RAILS: statistics modulate the colour *inside* a band and can flag an
    # elevated reading, but they may not manufacture a RED below $105 or hide a
    # RED above it.
    band = float(np.clip((95.0 - last) / 10.0, -1.5, 1.5))
    score = float(np.clip(band - 0.20 * np.clip(z, -2, 2) - 0.15 * tr, -3, 3))
    color = cut(score, prev)
    if last > 105:
        color = "RED"                            # rail: sustained-high override
    elif color == "RED":
        color = "YELLOW"                         # rail: no RED below the band top
    elevated = " — ELEVATED (z>2, rising)" if (z > 2 and tr > 0) else ""
    return {"color": color, "value": round(last, 2), "score": round(score, 2),
            "note": (f"Brent ${last:.2f} (robust z {z:+.2f} vs 2y, trend {tr:+.2f}/20d); "
                     f"bands 85/105{elevated}")}


def flag_F(px, prev):
    """Fed. Market-implied policy path = 2y Treasury minus effective fed funds,
    scored RELATIVE TO ITS OWN RECENT DISTRIBUTION rather than an absolute cut.

    Why relative: the raw spread is not comparable across regimes. The current
    +0.74pp sits at the 70th percentile of 1976-2026 (median +0.39) — entirely
    ordinary — but at the 99.6th percentile of the post-2023 window (median
    -0.40). An absolute threshold therefore fires RED on a historically normal
    reading, and RED forces the playbook into "defensives only, max 25% of
    tranche". That is far too costly to trigger on a level that has been normal
    for most of the last fifty years.

    A trailing 5-year robust z is the honest compromise: long enough to span a
    policy cycle (3y covers only the recent cut-pricing regime and makes the
    scale unstable), short enough to register a genuine regime shift. Sign is
    negated because a 2y ABOVE funds means tightening is priced = risk-off.
    """
    d2 = fred("DGS2")
    ff = fred("DFF")
    if ff is None or not len(ff):          # explicit: polars frames are not
        ff = fred("FEDFUNDS")              # truth-testable with `or`
    if d2 is None or ff is None or not len(d2) or not len(ff):
        return None
    j = (d2.rename({"value": "y2"})
           .join(ff.rename({"value": "ff"}), on="date", how="inner")
           .drop_nulls())
    if len(j) < 400:
        return None
    spread = (j["y2"] - j["ff"]).to_numpy().astype(float)
    cur = float(spread[-1])
    win = spread[-1260:]                    # ~5 trading years
    z = robust_z(win, cur)
    tr = norm_slope(spread[-90:])
    score = float(np.clip(-z / 3.0 - 0.15 * tr, -3, 3))
    stance = ("hikes" if cur > 0.15 else "cuts" if cur < -0.15 else "hold")
    return {"color": cut(score, prev), "value": round(cur, 3), "score": round(score, 2),
            "note": (f"2y {j['y2'][-1]:.2f}% − FF {j['ff'][-1]:.2f}% = {cur:+.2f}pp "
                     f"({stance} priced), 5y robust z {z:+.2f}, path {tr:+.2f}")}


def flag_I(px, prev):
    """Inflation. 3-month annualised core CPI — one month is mostly noise.

    Uses the 3m compounded rate as the level and its 12m trend for direction.
    """
    df = fred("CPILFESL")
    if df is None or len(df) < 15:
        return None
    v = df["value"].to_numpy().astype(float)
    mom = np.diff(v) / v[:-1]
    m3 = float((np.prod(1 + mom[-3:]) ** (12 / 3) - 1) * 100)   # % annualised
    m1 = float(mom[-1] * 100)
    tr = norm_slope(mom[-12:] * 100)
    lvl = float(np.clip((2.75 - m3) / 0.85, -2, 2))             # 2.75% = neutral
    score = float(np.clip(lvl - 0.3 * tr, -3, 3))
    return {"color": cut(score, prev), "value": round(m3, 2), "score": round(score, 2),
            "note": (f"core CPI 3m annualised {m3:.2f}% (last m/m {m1:.2f}%), "
                     f"trend {tr:+.2f}; print {str(df['date'][-1])[:10]}")}


def flag_S(px, prev):
    """Semi tape. An equal-weight SEMIS basket, not NDX.

    The old rule read NDX levels, which is 60% mega-cap software — it can be
    green while semis are in a bear market. Trend structure (price vs 50d, 50d
    vs 200d) plus relative strength against SPY.
    """
    b = basket_curve(px, SEMI_BASKET, 300)
    sp = series_for(px, BENCH, 300)
    if len(b) < 200 or len(sp) < 200:
        return None
    ma50, ma200 = float(np.mean(b[-50:])), float(np.mean(b[-200:]))
    last = float(b[-1])
    above50 = (last / ma50 - 1) * 100
    golden = (ma50 / ma200 - 1) * 100
    m = min(len(b), len(sp), 126)                                # fixed 6m window
    rs = float((b[-1] / b[-m]) / (sp[-1] / sp[-m]) - 1) * 100     # rel. strength
    tr = norm_slope(b[-60:])
    score = float(np.clip(0.35 * np.tanh(above50 / 5) * 2 + 0.35 * np.tanh(golden / 5) * 2
                          + 0.2 * np.tanh(rs / 10) * 2 + 0.25 * tr, -3, 3))
    return {"color": cut(score, prev), "value": round(above50, 2), "score": round(score, 2),
            "note": (f"semis basket {above50:+.1f}% vs 50d, 50d/200d {golden:+.1f}%, "
                     f"RS vs SPY {rs:+.1f}%, trend {tr:+.2f}")}


def flag_C(px, prev):
    """AI capex. Relative strength of the capex SUPPLY chain vs the market,
    cross-checked against the hyperscalers who write the cheques.

    Capex guidance is only revealed quarterly, but the supply chain re-prices
    daily and leads the announcements — so a price-based estimate is both
    fresher and more honest than a hand-typed note from a quarterly PDF.
    """
    sup = basket_curve(px, CAPEX_SUPPLY, 300)
    spend = basket_curve(px, CAPEX_SPEND, 300)
    sp = series_for(px, BENCH, 300)
    if len(sup) < 150 or len(sp) < 150:
        return None
    m = min(len(sup), len(sp), 126)
    rs_sup = float((sup[-1] / sup[-m]) / (sp[-1] / sp[-m]) - 1) * 100
    ma200 = float(np.mean(sup[-200:])) if len(sup) >= 200 else float(np.mean(sup))
    above200 = (float(sup[-1]) / ma200 - 1) * 100
    tr = norm_slope(sup[-60:])
    rs_spend = 0.0
    if len(spend) >= m:
        rs_spend = float((spend[-1] / spend[-m]) / (sp[-1] / sp[-m]) - 1) * 100
    score = float(np.clip(0.45 * np.tanh(rs_sup / 12) * 2 + 0.3 * np.tanh(above200 / 8) * 2
                          + 0.15 * np.tanh(rs_spend / 12) * 2 + 0.2 * tr, -3, 3))
    return {"color": cut(score, prev), "value": round(rs_sup, 2), "score": round(score, 2),
            "note": (f"capex supply-chain RS vs SPY {rs_sup:+.1f}% (6m), "
                     f"{above200:+.1f}% vs 200d, hyperscaler RS {rs_spend:+.1f}%, "
                     f"trend {tr:+.2f}")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    prev = {}
    if STATE_JSON.exists():
        try:
            st = json.loads(STATE_JSON.read_text())
            if st.get("formula_version") == FORMULA_VERSION:
                prev = st.get("flags", {})
            else:
                print(f"  formula changed ({st.get('formula_version')} → "
                      f"{FORMULA_VERSION}); resetting hysteresis baseline",
                      file=sys.stderr)
        except Exception:  # noqa: BLE001
            prev = {}

    px = bronze()
    today = str(date.today())
    out, failed = {}, []
    for name, fn in (("O", flag_O), ("F", flag_F), ("I", flag_I),
                     ("S", flag_S), ("C", flag_C)):
        try:
            r = fn(px, (prev.get(name) or {}).get("color"))
        except Exception as e:  # noqa: BLE001
            r, _ = None, print(f"  ! flag {name} error: {e}", file=sys.stderr)
        if r is None:
            failed.append(name)
            # keep the previous auto value rather than emitting a wrong one
            if name in prev:
                out[name] = {**prev[name], "note": prev[name].get("note", "") +
                             " [held: source unavailable today]"}
            continue
        out[name] = r

    if a.json:
        print(json.dumps({"as_of": today, "flags": out, "failed": failed}, indent=2))

    lines = [
        "# AUTO-GENERATED by ~/trade-ops/flags/auto_flags.py — do not hand-edit.",
        f"# Regenerated {datetime.now().isoformat(timespec='seconds')}.",
        "# Read via ~/trade-ops/flags/local_config.yaml so the tracked",
        "# configs/flag_overrides.yaml stays untouched (merge-clean).",
        "",
        "max_age_days: 3",
        "",
        "flags:",
    ]
    for k in ("O", "F", "I", "S", "C"):
        r = out.get(k)
        if not r:
            lines += [f"  {k}:", "    color: null", '    note: "auto: no reading"',
                      "    as_of: null"]
            continue
        note = str(r["note"]).replace('"', "'")
        lines += [f"  {k}:", f"    color: {r['color']}",
                  f'    note: "auto({r.get("score")}): {note}"',
                  f'    as_of: "{today}"']

    # events block is human knowledge (earnings outcomes) — preserve whatever is
    # already there rather than blanking it.
    prev_events = ""
    if OUT_YAML.exists():
        txt = OUT_YAML.read_text()
        if "\nevents:" in txt:
            prev_events = txt.split("\nevents:", 1)[1]
    if not prev_events:
        src = REPO / "configs/flag_overrides.yaml"
        if src.exists() and "\nevents:" in src.read_text():
            prev_events = src.read_text().split("\nevents:", 1)[1]
    lines += ["", "events:" + (prev_events if prev_events else " {}")]

    text = "\n".join(lines).rstrip() + "\n"
    if a.dry_run:
        print(text)
        return 0
    OUT_YAML.write_text(text)
    STATE_JSON.write_text(json.dumps(
        {"as_of": today, "formula_version": FORMULA_VERSION, "flags": out}, indent=1))

    colors = " ".join(f"{k}={out[k]['color'][:1] if k in out else '?'}"
                      for k in ("O", "F", "I", "S", "C"))
    g = sum(1 for v in out.values() if v["color"] == "GREEN")
    r_ = sum(1 for v in out.values() if v["color"] == "RED")
    y = sum(1 for v in out.values() if v["color"] == "YELLOW")
    print(f"auto-flags {today}: {colors}  ({g}G/{y}Y/{r_}R)"
          + (f"  [held: {','.join(failed)}]" if failed else ""))
    for k in ("O", "F", "I", "S", "C"):
        if k in out:
            print(f"  {k}: {out[k]['color']:<6} score={out[k]['score']:+.2f}  {out[k]['note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
