"""LLM-powered structured extraction from news/SEC headlines.

Primary: any OpenAI-completions-shaped cloud endpoint (DeepSeek, Qwen/DashScope,
OpenAI, ...), configured through ``LLM_API_KEY`` / ``LLM_BASE_URL`` /
``LLM_MODEL`` — see :mod:`trading_system.ingestion.llm_config`.
Fallback: a local or Ollama-cloud model via ``OLLAMA_HOST`` / ``OLLAMA_MODEL``.
With no LLM at all, functions fall back to rule-based naive_sentiment().

The ``DEEPSEEK_*`` module constants are kept as names for backward
compatibility; their values now come from the env-driven resolver.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import requests

from ..utils import get_logger
from .llm_config import llm_api_key, llm_base_url, llm_model, llm_provider

logger = get_logger(__name__)

# Cloud LLM endpoint — OpenAI-completions shape, provider set by env.
# See ingestion/llm_config.py: LLM_BASE_URL / LLM_MODEL / LLM_API_KEY, falling
# back to DeepSeek's defaults so an existing .env keeps working.
DEEPSEEK_BASE_URL = llm_base_url()
DEEPSEEK_DEFAULT_MODEL = llm_model()
DEEPSEEK_REASONING_MODEL = os.environ.get("LLM_REASONING_MODEL", "deepseek-reasoner")

# ---------------------------------------------------------------------------
# V2: OllamaClient — calls a local Ollama server
# ---------------------------------------------------------------------------

@dataclass
class OllamaClient:
    """Thin wrapper around the Ollama REST API for local inference.

    Compatible with any model served by `ollama serve`:
      deepseek-r1:7b, deepseek-r1:14b, llama3.3, mistral, etc.

    The /api/chat endpoint accepts OpenAI-style messages and returns
    a completion.  No streaming — uses stream=false for simplicity.
    """

    host: str = field(default_factory=lambda: os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    model: str = field(default_factory=lambda: os.environ.get("OLLAMA_MODEL", "deepseek-r1:7b"))
    timeout: int = 60

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> str:
        """Send a chat completion request and return the assistant's text."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        resp = requests.post(
            f"{self.host.rstrip('/')}/api/chat",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]

    def is_available(self) -> bool:
        """Quick health check — returns True if Ollama is reachable."""
        try:
            resp = requests.get(f"{self.host.rstrip('/')}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# V2: LLMRouter — DeepSeek cloud primary, Ollama local fallback
# ---------------------------------------------------------------------------

@dataclass
class LLMRouter:
    """Routes LLM calls: tries the configured cloud endpoint, falls back to Ollama.

    Usage::

        router = LLMRouter()  # picks up env vars automatically
        text = router.complete(messages=[...])

    Falls back to Ollama when no key is configured (``LLM_API_KEY``, or the
    legacy ``DEEPSEEK_API_KEY``) or when the cloud call errors or times out.  A
    key exported anywhere the process can see it — ``.env``, ``~/.bashrc``, a
    systemd unit — counts as configured, which is the first thing to check
    before concluding the router is misbehaving.
    """

    deepseek_api_key: str | None = field(
        default_factory=llm_api_key
    )
    deepseek_model: str = DEEPSEEK_DEFAULT_MODEL
    ollama: OllamaClient = field(default_factory=OllamaClient)
    _active_backend: str = field(default="unknown", init=False, repr=False)
    # rolling token accounting so we can see disk-cache effectiveness
    last_usage: dict = field(default_factory=dict, init=False, repr=False)
    cache_hit_tokens: int = field(default=0, init=False, repr=False)
    cache_miss_tokens: int = field(default=0, init=False, repr=False)

    @property
    def backend(self) -> str:
        """The provider label, 'ollama', or 'none' — whichever is available.

        The label used to be the literal string ``"deepseek"`` whatever endpoint
        was configured, which made a misconfigured run hard to read: an expired
        key still reported a cloud backend, so every call went out, 402'd and
        fell back per ticker.  It now names the actual provider
        (``deepseek`` / ``qwen`` / ``openai`` / ``custom``), so the log says
        where calls are going.
        """
        if self.deepseek_api_key:
            return llm_provider()
        if self.ollama.is_available():
            return "ollama"
        return "none"

    @property
    def hit_rate(self) -> float:
        """Cumulative DeepSeek disk-cache hit rate over this router's lifetime."""
        total = self.cache_hit_tokens + self.cache_miss_tokens
        return self.cache_hit_tokens / total if total else 0.0

    def complete_many(
        self,
        message_sets: list[list[dict[str, str]]],
        max_workers: int = 8,
        **kwargs,
    ) -> list[str | None]:
        """Run many independent completions concurrently (latency-bound work).

        Order of results matches the input order.  Used for per-ticker batched
        calls (apprehension, narration) so a 100-name universe doesn't pay 100×
        serial round-trips.
        """
        from concurrent.futures import ThreadPoolExecutor

        if not message_sets:
            return []
        workers = max(1, min(max_workers, len(message_sets)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(lambda m: self.complete(m, **kwargs), message_sets))

    def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 512,
        require_json: bool = False,
    ) -> str | None:
        """Try DeepSeek first, fall back to Ollama, return None if both fail.

        Parameters
        ----------
        messages:
            OpenAI-style chat messages list.
        temperature:
            Sampling temperature.
        max_tokens:
            Maximum tokens to generate.
        require_json:
            If True, appends a JSON instruction for Ollama (DeepSeek handles
            this via response_format natively).
        """
        # --- DeepSeek cloud path ---
        if self.deepseek_api_key:
            payload: dict[str, Any] = {
                "model": self.deepseek_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if require_json:
                payload["response_format"] = {"type": "json_object"}
            try:
                resp = requests.post(
                    DEEPSEEK_BASE_URL,
                    headers={
                        "Authorization": f"Bearer {self.deepseek_api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=30,
                )
                resp.raise_for_status()
                self._active_backend = llm_provider()
                data = resp.json()
                usage = data.get("usage", {}) or {}
                # DeepSeek reports disk-cache hits/misses so we can verify the
                # stable-prefix prompt design is actually paying off (hits = 1/10 cost)
                hit = int(usage.get("prompt_cache_hit_tokens", 0) or 0)
                miss = int(usage.get("prompt_cache_miss_tokens", 0) or 0)
                self.last_usage = usage
                self.cache_hit_tokens += hit
                self.cache_miss_tokens += miss
                if hit or miss:
                    logger.debug(
                        f"deepseek tokens: cache_hit={hit} miss={miss} "
                        f"(cumulative hit-rate {self.hit_rate:.0%})"
                    )
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                logger.warning(f"DeepSeek API failed ({e}), falling back to Ollama…")

        # --- Ollama local fallback ---
        if self.ollama.is_available():
            try:
                msgs = list(messages)
                if require_json and not any("JSON" in str(m.get("content", "")) for m in msgs):
                    msgs[-1] = {
                        **msgs[-1],
                        "content": msgs[-1]["content"] + "\nRespond with ONLY valid JSON, no other text.",
                    }
                self._active_backend = "ollama"
                return self.ollama.complete(msgs, temperature=temperature, max_tokens=max_tokens)
            except Exception as e:
                logger.warning(f"Ollama fallback failed: {e}")

        self._active_backend = "none"
        return None


def _default_router() -> LLMRouter:
    """Create an LLMRouter using environment variable configuration."""
    return LLMRouter()

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
    router: LLMRouter | None = None,
) -> dict[str, Any] | None:
    """Extract structured event fields from a single headline.

    V2: accepts an optional LLMRouter for DeepSeek-with-Ollama-fallback.
    Falls back to rule-based naive_sentiment() if all LLM calls fail.

    Returns a dict with the enriched fields, or None if LLM unavailable.
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _USER_TMPL.format(
            ticker=ticker, headline=headline[:500]
        )},
    ]

    # V2: use LLMRouter if provided
    if router is not None:
        content = router.complete(messages, temperature=0.1, max_tokens=256, require_json=True)
        if content:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                logger.debug(f"LLMRouter JSON parse failed for '{headline[:60]}'")
        return None

    # Legacy direct DeepSeek path (backward compat)
    api_key = api_key or llm_api_key()
    if not api_key:
        return None

    payload = {
        "model": model,
        "messages": messages,
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
    router: LLMRouter | None = None,
) -> list[dict[str, Any]]:
    """Enrich a list of raw event rows in-place.

    Each row must have "tickers" (list) and "summary" (raw headline string).
    Returns the same list with sentiment/confidence/magnitude/etc. populated.
    Falls back to naive_sentiment() when all LLM calls fail.

    V2: accepts an optional LLMRouter for DeepSeek-with-Ollama-fallback.
    """
    from ..features.sentiment import naive_sentiment

    # Determine if LLM is available
    _router = router
    if _router is None:
        api_key = api_key or llm_api_key()
        use_llm = bool(api_key)
        if not use_llm:
            logger.info("No LLM available — using rule-based sentiment fallback.")
    else:
        use_llm = _router.backend != "none"
        if not use_llm:
            logger.info("LLMRouter has no available backend — using rule-based fallback.")

    enriched = []
    for row in rows:
        ticker = (row.get("tickers") or ["UNKNOWN"])[0]
        headline = row.get("summary") or ""

        result = None
        if use_llm:
            if _router is not None:
                result = enrich_event(ticker, headline, router=_router)
            else:
                result = enrich_event(ticker, headline, api_key=api_key, model=model)

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
    router: LLMRouter | None = None,
) -> "pl.DataFrame":
    """Compute one apprehension score per ticker using LLM or rule-based fallback.

    V2: accepts an optional LLMRouter for DeepSeek-with-Ollama-fallback.

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

    api_key = api_key or llm_api_key()
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

    def _rule_based(sub: "pl.DataFrame") -> dict:
        mean_sent = float(sub["sentiment"].mean() or 0.0)
        mean_mag = float(sub["magnitude"].mean() or 0.0)
        total_flags = sum(len(flags or []) for flags in sub["risk_flags"].to_list())
        risk_density = min(1.0, total_flags / max(len(sub), 1))
        raw = max(0.0, -mean_sent) * 0.45 + mean_mag * 0.25 + risk_density * 0.30
        return {
            "apprehension_score": min(1.0, raw),
            "outlook": "deteriorating" if mean_sent < -0.2 else ("improving" if mean_sent > 0.2 else "stable"),
            "apprehension_drivers": [],
        }

    # Build one batched message set per ticker (system prompt is a stable prefix
    # → DeepSeek disk-cache hits across the universe → ~1/10 input cost).
    use_llm = bool(api_key) or (router is not None and router.backend != "none")
    _router = router
    if use_llm and _router is None:
        _router = LLMRouter(deepseek_api_key=api_key, deepseek_model=model)

    rows: list[dict] = []
    llm_tickers: list[str] = []
    llm_messages: list[list[dict]] = []
    subs: dict[str, "pl.DataFrame"] = {}

    for ticker in tickers:
        sub = window.filter(pl.col("ticker") == ticker).sort("date", descending=True)
        if sub.is_empty():
            continue
        subs[ticker] = sub
        if not use_llm:
            rows.append({"date": today, "ticker": ticker, **_rule_based(sub)})
            continue
        items = sub.to_dicts()[:20]
        parts = []
        for idx, item in enumerate(items, start=1):
            headline = item.get("summary") or ""
            content = (item.get("content") or "")[:400]
            parts.append(f"{idx}. {headline}" + (f"\n   Snippet: {content}" if content else ""))
        llm_tickers.append(ticker)
        llm_messages.append([
            {"role": "system", "content": _APPREHENSION_SYSTEM},
            {"role": "user", "content": _APPREHENSION_USER_TMPL.format(
                ticker=ticker, days=days, n=len(items), headlines="\n".join(parts)
            )},
        ])

    # One concurrent burst instead of N serial round-trips
    if llm_messages:
        contents = _router.complete_many(
            llm_messages, max_workers=8, temperature=0.1, max_tokens=200, require_json=True
        )
        for ticker, content in zip(llm_tickers, contents):
            try:
                result = json.loads(content) if content else {}
                rows.append({
                    "date": today, "ticker": ticker,
                    "apprehension_score": float(result.get("apprehension_score", 0.5)),
                    "outlook": str(result.get("outlook", "stable")),
                    "apprehension_drivers": result.get("drivers") or [],
                })
            except Exception as e:
                logger.debug(f"Apprehension parse failed for {ticker}: {e}")
                rows.append({"date": today, "ticker": ticker, **_rule_based(subs[ticker])})
        if _router is not None and (_router.cache_hit_tokens + _router.cache_miss_tokens):
            logger.info(f"apprehension LLM cache hit-rate: {_router.hit_rate:.0%}")

    if not rows:
        return _EMPTY

    return pl.DataFrame(rows, schema={
        "date": pl.Date,
        "ticker": pl.Utf8,
        "apprehension_score": pl.Float64,
        "outlook": pl.Utf8,
        "apprehension_drivers": pl.List(pl.Utf8),
    })
