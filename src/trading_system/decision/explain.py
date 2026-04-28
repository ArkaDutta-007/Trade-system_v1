"""DeepSeek-powered plain-English explanation of decision reports.

Reads a report markdown file (from reports/decisions/) and asks the
DeepSeek V4 API to narrate it for a non-expert audience — summarising the
stance, key signals, risks, and conviction level in plain English.

Usage (CLI):
    ts explain reports/decisions/MSFT_20260428_004505.md

Usage (Python):
    from trading_system.decision.explain import explain_report
    text = explain_report("reports/decisions/MSFT_....md")
    print(text)

Set DEEPSEEK_API_KEY in ~/.zshrc or .env.
"""
from __future__ import annotations

import os
from pathlib import Path

import requests

from ..utils import get_logger

logger = get_logger(__name__)

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1/chat/completions"

# Use "deepseek-chat" for fastest responses (DeepSeek V4 latest chat model).
# Switch to "deepseek-reasoner" for their highest-capability R1-class model
# — better for nuanced analysis but ~4× slower.
DEEPSEEK_DEFAULT_MODEL = "deepseek-chat"

_SYSTEM_PROMPT = """\
You are a senior equity analyst explaining an automated trading research report \
to a portfolio manager who is smart but not a quant.

Your explanation must cover:
1. **Overall stance** — what the system recommends (BUY/HOLD/SELL) and the \
single most important reason why.
2. **Key technical signals** — pick the 2-3 most significant technical facts \
from the report and explain what they mean in plain English.
3. **Market context** — describe the regime (bull/bear, volatility) and where \
the stock sits in the universe ranking, and why that matters.
4. **Top risks** — the 1-2 reasons this call could be wrong.
5. **Conviction** — rate the overall conviction as Low / Medium / High based \
on confidence score, score source (rule-based vs model), and signal alignment.

Use short bullet points inside each section. Keep the total under 450 words. \
Avoid unexplained jargon — if you must use a term (e.g. RSI), give a one-word \
gloss in parentheses.\
"""


def explain_report(
    report_path: str | Path,
    api_key: str | None = None,
    model: str = DEEPSEEK_DEFAULT_MODEL,
) -> str:
    """Return a plain-English explanation of a decision report markdown file.

    Parameters
    ----------
    report_path:
        Path to a ``reports/decisions/<TICKER>_<stamp>.md`` file.
    api_key:
        DeepSeek API key. Falls back to ``DEEPSEEK_API_KEY`` env var.
    model:
        DeepSeek model name. Defaults to ``deepseek-chat`` (V4 latest).
        Use ``deepseek-reasoner`` for R1-class reasoning (slowest / highest quality).

    Returns
    -------
    str
        The explanation text, or an error message if the API call fails.
    """
    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        return (
            "DEEPSEEK_API_KEY is not set.\n"
            "Add it to ~/.zshrc:  export DEEPSEEK_API_KEY=\"your_key_here\"\n"
            "Or to the project .env file:  DEEPSEEK_API_KEY=your_key_here"
        )

    report_path = Path(report_path)
    if not report_path.exists():
        return f"Report file not found: {report_path}"

    report_text = report_path.read_text()

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Please explain this trading decision report:\n\n"
                    f"---\n{report_text}\n---"
                ),
            },
        ],
        "temperature": 0.3,
        "max_tokens": 700,
    }

    try:
        resp = requests.post(
            DEEPSEEK_BASE_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=40,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except requests.HTTPError as e:
        logger.error(f"DeepSeek API HTTP error: {e} — {e.response.text[:200]}")
        return f"DeepSeek API error ({e.response.status_code}): {e.response.text[:200]}"
    except Exception as e:
        logger.error(f"DeepSeek explain failed: {e}")
        return f"Error calling DeepSeek API: {e}"
