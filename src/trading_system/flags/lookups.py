"""Live lookups for the five flags.

Each flag has a pure ``classify_*`` function (unit-testable, no I/O) and a
``lookup_*`` function that fetches live data and returns a ``FlagReading``.

Data sources (all keyless):
  O — Brent front-month future ``BZ=F`` via yfinance
  F — FRED ``DFEDTARU`` (fed funds target, upper bound) trend; tone via override
  I — FRED ``CPILFESL`` core CPI m/m + ``CPIAUCSL`` headline YoY
  S — Nasdaq-100 ``^NDX`` level vs the playbook's breakout/breakdown levels
  C — qualitative (hyperscaler capex guidance) → override only

FRED is read through the public ``fredgraph.csv`` endpoint so no API key is
required; if ``FRED_API_KEY`` is set the fredapi client is preferred.
Every lookup degrades to ``FlagColor.UNKNOWN`` instead of raising.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from ..utils import get_logger
from .datafeed import fred_series
from .models import FlagColor, FlagReading

logger = get_logger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ─────────────────────────── pure classifiers ───────────────────────────────

def classify_oil(
    brent: float,
    falling: bool,
    avg_5d: float | None = None,
    hormuz_closed: bool = False,
    green_below: float = 85.0,
    red_above: float = 105.0,
) -> tuple[FlagColor, str]:
    """O: GREEN < $85 and falling · YELLOW $85–105 · RED > $105 sustained."""
    if hormuz_closed:
        return FlagColor.RED, "Hormuz closure override active"
    sustained = avg_5d if avg_5d is not None else brent
    if sustained > red_above and brent > red_above:
        return FlagColor.RED, f"Brent ${brent:.2f} (5d avg ${sustained:.2f}) > ${red_above:.0f} sustained"
    if brent > red_above:
        return FlagColor.YELLOW, f"Brent ${brent:.2f} above ${red_above:.0f} but not sustained (5d avg ${sustained:.2f})"
    if brent < green_below:
        if falling:
            return FlagColor.GREEN, f"Brent ${brent:.2f} < ${green_below:.0f} and falling"
        return FlagColor.YELLOW, f"Brent ${brent:.2f} < ${green_below:.0f} but not clearly falling"
    return FlagColor.YELLOW, f"Brent ${brent:.2f} in ${green_below:.0f}–{red_above:.0f} band"


def classify_fed(recent_move: str | None) -> tuple[FlagColor, str]:
    """F from observable policy only (tone needs the override).

    recent_move: 'hike' | 'cut' | 'hold' | None
    """
    if recent_move == "hike":
        return FlagColor.RED, "target rate raised within lookback window"
    if recent_move == "cut":
        return FlagColor.GREEN, "target rate cut within lookback window (dovish)"
    if recent_move == "hold":
        return FlagColor.YELLOW, "rate hold; tone unknown — set the F override after the FOMC"
    return FlagColor.UNKNOWN, "no target-rate data"


def classify_core_cpi(
    core_mom_pct: float,
    headline_yoy_pct: float | None = None,
    green_mom: float = 0.2,
    red_mom: float = 0.4,
    red_headline_yoy: float = 5.0,
) -> tuple[FlagColor, str]:
    """I: GREEN core m/m ≤0.2 · YELLOW 0.3 · RED ≥0.4 or headline YoY > 5."""
    if headline_yoy_pct is not None and headline_yoy_pct > red_headline_yoy:
        return FlagColor.RED, f"headline CPI {headline_yoy_pct:.1f}% YoY > {red_headline_yoy:.0f}%"
    r = round(core_mom_pct, 1)
    if r >= red_mom:
        return FlagColor.RED, f"core CPI {core_mom_pct:.2f}% m/m (rounds to {r:.1f} ≥ {red_mom:.1f})"
    if r <= green_mom:
        return FlagColor.GREEN, f"core CPI {core_mom_pct:.2f}% m/m (rounds to {r:.1f} ≤ {green_mom:.1f})"
    return FlagColor.YELLOW, f"core CPI {core_mom_pct:.2f}% m/m (rounds to {r:.1f})"


def classify_semi_tape(
    ndx: float,
    green_above: float = 30034.0,
    red_below: float = 28663.0,
) -> tuple[FlagColor, str]:
    """S: GREEN > 30,034 · YELLOW 28,663–30,034 · RED < 28,663."""
    if ndx > green_above:
        return FlagColor.GREEN, f"NDX {ndx:,.0f} > {green_above:,.0f}"
    if ndx < red_below:
        return FlagColor.RED, f"NDX {ndx:,.0f} < {red_below:,.0f} — trend broken"
    return FlagColor.YELLOW, f"NDX {ndx:,.0f} in {red_below:,.0f}–{green_above:,.0f} band"


# ─────────────────────────── data fetch helpers ──────────────────────────────

def _yf_history(symbol: str, period: str = "6mo") -> pl.DataFrame:
    import yfinance as yf

    hist = yf.Ticker(symbol).history(period=period, auto_adjust=True)
    if hist is None or len(hist) == 0:
        raise RuntimeError(f"no history for {symbol}")
    df = pl.DataFrame({
        "date": [d.date() for d in hist.index.to_pydatetime()],
        "close": hist["Close"].astype(float).tolist(),
    })
    # yfinance intermittently appends a NaN-close row (esp. for indices/futures).
    # A NaN must never reach a classifier — it would silently read as "in band".
    df = df.filter(pl.col("close").is_not_null() & pl.col("close").is_not_nan())
    if df.is_empty():
        raise RuntimeError(f"{symbol} returned no valid (non-NaN) closes")
    return df


def fetch_fred_csv(series_id: str, cache_dir: Path | None = None) -> pl.DataFrame:
    """Back-compat thin wrapper over the resilient datafeed (cache→API→CSV)."""
    return fred_series(series_id, cache_dir=cache_dir).df


# ─────────────────────────── live lookups ────────────────────────────────────

def lookup_oil(thresholds: dict | None = None, hormuz_closed: bool = False) -> FlagReading:
    t = thresholds or {}
    try:
        px = _yf_history("BZ=F", period="3mo").sort("date")
        brent = float(px["close"][-1])
        avg_5d = float(px["close"].tail(5).mean())
        avg_20d = float(px["close"].tail(20).mean())
        falling = brent < avg_20d  # below its own 20d mean = falling tape
        color, detail = classify_oil(
            brent, falling, avg_5d=avg_5d, hormuz_closed=hormuz_closed,
            green_below=t.get("green_below", 85.0), red_above=t.get("red_above", 105.0),
        )
        return FlagReading(
            flag="O", name="Oil / Iran", color=color, value=round(brent, 2),
            detail=detail, source="live", as_of=_now_iso(),
            extras={"avg_5d": round(avg_5d, 2), "avg_20d": round(avg_20d, 2), "falling": falling},
        )
    except Exception as e:
        logger.warning(f"O flag lookup failed: {e}")
        return FlagReading(
            flag="O", name="Oil / Iran", color=FlagColor.UNKNOWN, value=None,
            detail=f"lookup failed: {e}", source="error", as_of=_now_iso(),
        )


def lookup_fed(lookback_days: int = 75, cache_dir: Path | None = None) -> FlagReading:
    try:
        res = fred_series("DFEDTARU", cache_dir=cache_dir)
        df = res.df.sort("date")
        last = float(df["value"][-1])
        as_of_date = df["date"][-1]
        from datetime import timedelta

        cutoff = df["date"].max() - timedelta(days=lookback_days)
        ref = df.filter(pl.col("date") <= cutoff)
        prev = float(ref["value"][-1]) if len(ref) else last
        move = "hike" if last > prev + 1e-9 else ("cut" if last < prev - 1e-9 else "hold")
        color, detail = classify_fed(move)
        cache_tag = f" [{res.source}, {res.age_hours:.0f}h old]" if res.from_cache else ""
        return FlagReading(
            flag="F", name="Fed", color=color, value=last,
            detail=f"target upper {last:.2f}% (vs {prev:.2f}% {lookback_days}d ago, as of {as_of_date}) — {detail}{cache_tag}",
            source="live" if not res.from_cache else "cache",
            as_of=_now_iso(),
            extras={"move": move, "prev": prev, "series_as_of": str(as_of_date), "feed": res.source},
        )
    except Exception as e:
        logger.warning(f"F flag lookup failed: {e}")
        return FlagReading(
            flag="F", name="Fed", color=FlagColor.UNKNOWN, value=None,
            detail=f"lookup failed ({e}); set FRED_API_KEY or the F override",
            source="error", as_of=_now_iso(),
        )


def lookup_inflation(thresholds: dict | None = None, cache_dir: Path | None = None) -> FlagReading:
    t = thresholds or {}
    try:
        res = fred_series("CPILFESL", cache_dir=cache_dir)
        core = res.df.sort("date")
        core_mom = (float(core["value"][-1]) / float(core["value"][-2]) - 1.0) * 100.0
        core_month = str(core["date"][-1])
        headline_yoy = None
        try:
            head = fred_series("CPIAUCSL", cache_dir=cache_dir).df.sort("date")
            if len(head) >= 13:
                headline_yoy = (float(head["value"][-1]) / float(head["value"][-13]) - 1.0) * 100.0
        except Exception:
            pass
        color, detail = classify_core_cpi(
            core_mom, headline_yoy,
            green_mom=t.get("green_mom", 0.2), red_mom=t.get("red_mom", 0.4),
            red_headline_yoy=t.get("red_headline_yoy", 5.0),
        )
        hl = f", headline {headline_yoy:.1f}% YoY" if headline_yoy is not None else ""
        cache_tag = f" [{res.source}, {res.age_hours:.0f}h old]" if res.from_cache else ""
        return FlagReading(
            flag="I", name="Inflation", color=color, value=round(core_mom, 3),
            detail=f"{detail} (print month {core_month}{hl}){cache_tag}",
            source="live" if not res.from_cache else "cache",
            as_of=_now_iso(),
            extras={"core_month": core_month, "headline_yoy": headline_yoy, "feed": res.source},
        )
    except Exception as e:
        logger.warning(f"I flag lookup failed: {e}")
        return FlagReading(
            flag="I", name="Inflation", color=FlagColor.UNKNOWN, value=None,
            detail=f"lookup failed ({e}); set FRED_API_KEY or the I override",
            source="error", as_of=_now_iso(),
        )


def lookup_semi_tape(thresholds: dict | None = None) -> FlagReading:
    t = thresholds or {}
    try:
        px = _yf_history("^NDX", period="6mo").sort("date")
        ndx = float(px["close"][-1])
        sma50 = float(px["close"].tail(50).mean())
        color, detail = classify_semi_tape(
            ndx, green_above=t.get("green_above", 30034.0), red_below=t.get("red_below", 28663.0),
        )
        return FlagReading(
            flag="S", name="Semi tape", color=color, value=round(ndx, 1),
            detail=f"{detail} (50-day ≈ {sma50:,.0f})",
            source="live", as_of=_now_iso(), extras={"sma50": round(sma50, 1)},
        )
    except Exception as e:
        logger.warning(f"S flag lookup failed: {e}")
        return FlagReading(
            flag="S", name="Semi tape", color=FlagColor.UNKNOWN, value=None,
            detail=f"lookup failed: {e}", source="error", as_of=_now_iso(),
        )


def lookup_capex() -> FlagReading:
    """C is qualitative (hyperscaler guidance); without an override it is UNKNOWN."""
    return FlagReading(
        flag="C", name="AI capex", color=FlagColor.UNKNOWN, value=None,
        detail="no live source — set C in configs/flag_overrides.yaml after earnings/guidance",
        source="override-required", as_of=_now_iso(),
    )
