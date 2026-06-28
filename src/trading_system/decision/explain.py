"""DeepSeek narration of decision reports — token-efficient + cache-friendly.

Two efficiency wins over the old "dump the whole markdown" approach:

  1. **Send compact JSON, not rendered markdown.**  The markdown report repeats
     numbers inside tables and prose and burns tokens on formatting; we extract
     only the decision-relevant fields (stance, forecasts, bounds, top SHAP
     drivers, key technicals/regime) into a small JSON blob.
  2. **Stable system-prompt prefix → DeepSeek disk cache.**  The long instruction
     block is identical for every ticker, so across ``analyze-all`` it serves
     from cache at ~1/10 input cost.  The variable ticker JSON goes last.

``explain_decision(decision_dict)`` is the new path; ``explain_report(path)``
stays for the CLI and prefers the JSON sidecar, falling back to markdown.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import requests

from ..ingestion.llm_extractor import LLMRouter
from ..utils import get_logger

logger = get_logger(__name__)

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_DEFAULT_MODEL = "deepseek-chat"

# Stable prefix (cache-friendly). Asks for a tight, decision-useful structure
# instead of free-form 450-word prose — cheaper AND more actionable.
_SYSTEM_PROMPT = """\
You are a senior equity analyst. You receive a JSON snapshot of an automated
trading decision (stance, forecasts, calibrated price bounds, the top SHAP
feature attributions that drove the call, plus technical/regime context).

Write a tight briefing for a portfolio manager with these sections, in markdown:
- **Call** — the stance and the single most important reason (cite the top SHAP driver).
- **Bounds** — what the calibrated low/median/high prices imply over 1m / 3m / 12m
  (frame as realistic downside vs upside, not certainty).
- **Drivers** — the 2-3 features pushing the call, in plain English (gloss jargon).
- **Risks** — the 1-2 things that would invalidate this, and the rough price level
  where the thesis breaks.
- **Conviction** — Low / Medium / High, justified by confidence + score source.

Be concrete and brief (≤280 words). No preamble, no disclaimer.\
"""


def _compact_decision(decision: dict) -> dict:
    """Extract only the fields worth sending to the LLM."""
    g = decision.get("groundings", {}) or {}
    tech = g.get("technical", {}) or {}
    regime = g.get("regime", {}) or {}
    xs = g.get("cross_section", {}) or {}
    bounds = g.get("bounds", {}) or {}
    shap = g.get("shap_waterfall", {}) or {}

    # bounds: keep only price low/median/high per horizon
    compact_bounds = {}
    for label, h in (bounds.get("horizons", {}) or {}).items():
        p = h.get("price", {})
        compact_bounds[label] = {"lo": p.get("lo"), "median": p.get("median"), "hi": p.get("hi")}

    drivers = []
    for name, val, sv in zip(
        shap.get("feature_names", [])[:6],
        shap.get("feature_values", [])[:6],
        shap.get("shap_values", [])[:6],
    ):
        drivers.append({"feature": name, "value": round(float(val), 4), "shap": round(float(sv), 5)})

    return {
        "ticker": decision.get("ticker"),
        "as_of": decision.get("as_of"),
        "stance": decision.get("stance"),
        "confidence": decision.get("confidence"),
        "score_source": decision.get("score_source"),
        "forecast_5d": decision.get("forecast_5d"),
        "forecast_20d": decision.get("forecast_20d"),
        "rationale": (decision.get("rationale") or [])[:6],
        "price_bounds": compact_bounds,
        "top_drivers": drivers,
        "technical": {k: tech.get(k) for k in
                      ("adj_close", "mom_20d", "mom_60d", "rsi_14", "vol_20d",
                       "sma_gap_200", "dd_from_high_60") if k in tech},
        "regime": {k: regime.get(k) for k in
                   ("bull_regime", "high_vol_regime", "mom_20d_rank") if k in regime},
        "rank_in_universe": xs.get("rank_in_universe"),
        "universe_size": xs.get("universe_size"),
        "recent_news": [
            {"date": (n.get("known_at") or "")[:10], "headline": n.get("summary")}
            for n in (g.get("relevant_news") or [])[:3]
        ],
    }


def explain_decision(
    decision: dict,
    api_key: str | None = None,
    model: str = DEEPSEEK_DEFAULT_MODEL,
    router: LLMRouter | None = None,
) -> str:
    """Narrate a decision dict (compact JSON path). Returns text or error string."""
    payload_json = json.dumps(_compact_decision(decision), default=str)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Decision JSON:\n{payload_json}"},
    ]
    _router = router or LLMRouter(
        deepseek_api_key=api_key or os.environ.get("DEEPSEEK_API_KEY"),
        deepseek_model=model,
    )
    if _router.backend == "none":
        return "No LLM backend available (set DEEPSEEK_API_KEY or run Ollama)."
    out = _router.complete(messages, temperature=0.3, max_tokens=420)
    return out or "LLM returned no content."


def explain_report(
    report_path: str | Path,
    api_key: str | None = None,
    model: str = DEEPSEEK_DEFAULT_MODEL,
) -> str:
    """Explain a decision report. Prefers the compact JSON sidecar over markdown.

    Backward-compatible entry point for the ``ts explain`` CLI.
    """
    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    report_path = Path(report_path)

    # Prefer the JSON sidecar (compact, cache-friendly path)
    json_sidecar = report_path.with_suffix(".json")
    if json_sidecar.exists():
        try:
            decision = json.loads(json_sidecar.read_text())
            return explain_decision(decision, api_key=api_key, model=model)
        except Exception as e:
            logger.debug(f"sidecar explain failed, falling back to markdown: {e}")

    if not api_key:
        return (
            "DEEPSEEK_API_KEY is not set.\n"
            'Add it to .env:  DEEPSEEK_API_KEY="your_key_here"'
        )
    if not report_path.exists():
        return f"Report file not found: {report_path}"

    # Legacy markdown fallback (still cache-friendly: stable system prompt first)
    report_text = report_path.read_text()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Decision report (markdown):\n---\n{report_text}\n---"},
        ],
        "temperature": 0.3,
        "max_tokens": 420,
    }
    try:
        resp = requests.post(
            DEEPSEEK_BASE_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload, timeout=40,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"DeepSeek explain failed: {e}")
        return f"Error calling DeepSeek API: {e}"
