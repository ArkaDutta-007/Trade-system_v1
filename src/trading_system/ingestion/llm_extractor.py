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
