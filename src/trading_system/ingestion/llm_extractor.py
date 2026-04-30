"""DeepSeek-powered structured extraction from news/SEC headlines.

Replaces the OpenAI placeholder that was referenced in configs/default.yaml.
Uses the DeepSeek V4 API (OpenAI-compatible endpoint) to turn raw headline text
into structured event fields: event_type, sentiment, confidence, magnitude,
time_horizon, risk_flags, and a clean one-line summary.

Set DEEPSEEK_API_KEY in ~/.zshrc or .env. Without a key the function returns
None and the caller falls back to rule-based naive_sentiment().
"""
from __future__ import annotations

import json
import os
from typing import Any

import requests

from ..utils import get_logger

logger = get_logger(__name__)

# DeepSeek public API — OpenAI-compatible format.
# "deepseek-chat" always resolves to their latest released chat model (V3/V4).
# Swap to "deepseek-reasoner" for their highest-reasoning R1-class model
# (slower, ~4× more expensive, but better for nuanced filings analysis).
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_DEFAULT_MODEL = "deepseek-chat"   # latest V4 chat (fastest / cheapest)
DEEPSEEK_REASONING_MODEL = "deepseek-reasoner"  # R1-class — highest capability

_SYSTEM_PROMPT = """\
You are a financial event classifier. Given a news headline (and optional snippet) \
about a publicly-traded company, extract structured information.

Return ONLY a valid JSON object with EXACTLY these keys:
- "event_type": one of ["earnings_beat","earnings_miss","guidance_raise","guidance_cut",\
"m_and_a","regulatory","macro","product","analyst_upgrade","analyst_downgrade",\
"legal","management","geopolitical","other"]
- "sentiment": float in [-1.0, 1.0]  (-1=very bearish, 0=neutral, 1=very bullish)
- "confidence": float in [0.0, 1.0]  (your confidence in the classification)
- "magnitude": float in [0.0, 1.0]   (estimated price-impact magnitude)
- "time_horizon": one of ["intraday","1d","1w","1m","long_term"]
- "risk_flags": list of strings, e.g. ["dilution","litigation","regulatory_risk"] — empty list if none
- "summary": one concise sentence, max 120 characters

No explanation, no markdown fences, no extra keys — only the JSON object.\
"""

_USER_TMPL = "Ticker: {ticker}\nHeadline: {headline}"


def enrich_event(
    ticker: str,
    headline: str,
    api_key: str | None = None,
    model: str = DEEPSEEK_DEFAULT_MODEL,
) -> dict[str, Any] | None:
    """Call DeepSeek to extract structured event fields from a single headline.

    Returns a dict with the enriched fields, or None if the API key is absent
    or the call fails (caller should apply naive_sentiment() fallback).
    """
    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return None

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _USER_TMPL.format(
                ticker=ticker, headline=headline[:500]
            )},
        ],
        "temperature": 0.1,
        "max_tokens": 256,
        "response_format": {"type": "json_object"},
    }

    try:
        resp = requests.post(
            DEEPSEEK_BASE_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=20,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception as e:
        logger.debug(f"DeepSeek extraction failed for '{headline[:60]}…': {e}")
        return None


def batch_enrich_events(
    rows: list[dict[str, Any]],
    api_key: str | None = None,
    model: str = DEEPSEEK_DEFAULT_MODEL,
) -> list[dict[str, Any]]:
    """Enrich a list of raw event rows in-place.

    Each row must have "tickers" (list) and "summary" (raw headline string).
    Returns the same list with sentiment/confidence/magnitude/etc. populated.
    Falls back to naive_sentiment() when DeepSeek is unavailable.
    """
    from ..features.sentiment import naive_sentiment

    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    use_llm = bool(api_key)
    if not use_llm:
        logger.info("DEEPSEEK_API_KEY not set — using rule-based sentiment fallback.")

    enriched = []
    for row in rows:
        ticker = (row.get("tickers") or ["UNKNOWN"])[0]
        headline = row.get("summary") or ""

        if use_llm:
            result = enrich_event(ticker, headline, api_key=api_key, model=model)
        else:
            result = None

        if result:
            row = {**row, **{
                "event_type": result.get("event_type", row.get("event_type", "news")),
                "sentiment":  float(result.get("sentiment", 0.0)),
                "confidence": float(result.get("confidence", 0.5)),
                "magnitude":  float(result.get("magnitude", 0.0)),
                "time_horizon": result.get("time_horizon", row.get("time_horizon", "1d")),
                "risk_flags": result.get("risk_flags") or [],
                "summary":    result.get("summary", headline)[:500],
            }}
        else:
            # Rule-based fallback
            row = {**row, "sentiment": naive_sentiment(headline)}

        enriched.append(row)

    return enriched


# ---------------------------------------------------------------------------
# Apprehension scorer — one batched call per ticker per day
# ---------------------------------------------------------------------------

_APPREHENSION_SYSTEM = """\
You are a financial risk analyst. Given a set of recent news headlines for a \
publicly-traded company, assess market apprehension.

Return ONLY a valid JSON object with EXACTLY these keys:
- "apprehension_score": float in [0.0, 1.0]
  0.0 = no concern, calm positive backdrop
  0.5 = mixed signals, moderate uncertainty
  1.0 = extreme fear, crisis-level risk
- "drivers": list of up to 3 concise strings explaining the top risk drivers
  (empty list if score < 0.2)
- "outlook": one of ["improving","stable","deteriorating"]

No explanation, no markdown fences, no extra keys — only the JSON object.\
"""

_APPREHENSION_USER_TMPL = """\
Ticker: {ticker}
Period: last {days} days
Articles ({n}):
{headlines}
"""


def compute_apprehension_scores(
    events: "pl.DataFrame",
    as_of_date: "date | None" = None,
    days: int = 7,
    api_key: str | None = None,
    model: str = DEEPSEEK_DEFAULT_MODEL,
) -> "pl.DataFrame":
    """Compute one apprehension score per ticker using a batched DeepSeek call.

    Parameters
    ----------
    events:
        Full events DataFrame (EVENT_SCHEMA). Must have columns:
        known_at (Datetime), tickers (List[Utf8]), summary (Utf8),
        sentiment (Float64), magnitude (Float64), risk_flags (List[Utf8]).
    as_of_date:
        The date to compute scores for (today if None).
        Headlines from [as_of_date - days, as_of_date] are used.
    days:
        Rolling look-back window for headlines.
    api_key:
        DeepSeek API key. Falls back to DEEPSEEK_API_KEY env var.

    Returns
    -------
    pl.DataFrame with columns:
        date (pl.Date), ticker (pl.Utf8),
        apprehension_score (pl.Float64), outlook (pl.Utf8),
        apprehension_drivers (pl.List[pl.Utf8])
    """
    import polars as pl
    from datetime import date as _date, timedelta

    _EMPTY = pl.DataFrame(schema={
        "date": pl.Date,
        "ticker": pl.Utf8,
        "apprehension_score": pl.Float64,
        "outlook": pl.Utf8,
        "apprehension_drivers": pl.List(pl.Utf8),
    })

    if events is None or events.is_empty():
        return _EMPTY

    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    today = as_of_date or _date.today()
    cutoff = today - timedelta(days=days)

    # Filter to window and explode tickers
    window = (
        events
        .with_columns(date=pl.col("known_at").dt.date())
        .filter(pl.col("date") >= cutoff)
        .filter(pl.col("date") <= today)
        .explode("tickers")
        .rename({"tickers": "ticker"})
    )

    if window.is_empty():
        return _EMPTY

    tickers = window["ticker"].unique().to_list()
    rows = []

    for ticker in tickers:
        sub = window.filter(pl.col("ticker") == ticker).sort("date", descending=True)
        if sub.is_empty():
            continue

        items = sub.to_dicts()[:20]
        n = len(items)
        parts = []
        for idx, item in enumerate(items, start=1):
            headline = item.get("summary") or ""
            content = (item.get("content") or "")[:400]
            if content:
                parts.append(f"{idx}. {headline}\n   Snippet: {content}")
            else:
                parts.append(f"{idx}. {headline}")
        headlines_str = "\n".join(parts)

        # Rule-based fallback (no LLM key)
        if not api_key:
            mean_sent = float(sub["sentiment"].mean() or 0.0)
            mean_mag = float(sub["magnitude"].mean() or 0.0)
            total_flags = sum(len(flags or []) for flags in sub["risk_flags"].to_list())
            risk_density = min(1.0, total_flags / max(len(sub), 1))
            raw = max(0.0, -mean_sent) * 0.45 + mean_mag * 0.25 + risk_density * 0.30
            rows.append({
                "date": today,
                "ticker": ticker,
                "apprehension_score": min(1.0, raw),
                "outlook": "deteriorating" if mean_sent < -0.2 else ("improving" if mean_sent > 0.2 else "stable"),
                "apprehension_drivers": [],
            })
            continue

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": _APPREHENSION_SYSTEM},
                {"role": "user", "content": _APPREHENSION_USER_TMPL.format(
                    ticker=ticker, days=days, n=n, headlines=headlines_str
                )},
            ],
            "temperature": 0.1,
            "max_tokens": 200,
            "response_format": {"type": "json_object"},
        }

        try:
            resp = requests.post(
                DEEPSEEK_BASE_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=20,
            )
            resp.raise_for_status()
            result = json.loads(resp.json()["choices"][0]["message"]["content"])
            rows.append({
                "date": today,
                "ticker": ticker,
                "apprehension_score": float(result.get("apprehension_score", 0.5)),
                "outlook": str(result.get("outlook", "stable")),
                "apprehension_drivers": result.get("drivers") or [],
            })
        except Exception as e:
            logger.debug(f"Apprehension call failed for {ticker}: {e}")
            # Fallback: rule-based
            mean_sent = float(sub["sentiment"].mean() or 0.0)
            mean_mag = float(sub["magnitude"].mean() or 0.0)
            total_flags = sum(len(flags or []) for flags in sub["risk_flags"].to_list())
            risk_density = min(1.0, total_flags / max(len(sub), 1))
            raw = max(0.0, -mean_sent) * 0.45 + mean_mag * 0.25 + risk_density * 0.30
            rows.append({
                "date": today,
                "ticker": ticker,
                "apprehension_score": min(1.0, raw),
                "outlook": "stable",
                "apprehension_drivers": [],
            })

    if not rows:
        return _EMPTY

    return pl.DataFrame(rows, schema={
        "date": pl.Date,
        "ticker": pl.Utf8,
        "apprehension_score": pl.Float64,
        "outlook": pl.Utf8,
        "apprehension_drivers": pl.List(pl.Utf8),
    })
