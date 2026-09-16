"""Provider-agnostic LLM endpoint resolution.

The repo used to hard-code DeepSeek's URL and model name in two files and read
``DEEPSEEK_API_KEY`` from five call sites.  That made switching providers a
code change, and it made the failure mode confusing: an expired key still
*looked* like a configured backend, so every call went out, 402'd, and fell
back per ticker.

DeepSeek, DashScope (Qwen), OpenAI and most others speak the same
OpenAI-completions shape, so a provider is fully described by three strings.
They are resolved here, in this order:

    LLM_API_KEY    else DEEPSEEK_API_KEY
    LLM_BASE_URL   else https://api.deepseek.com/v1/chat/completions
    LLM_MODEL      else deepseek-chat

So an existing ``.env`` keeps working untouched, and pointing the system at Qwen
is three lines of configuration:

    LLM_API_KEY=<key>
    LLM_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions
    LLM_MODEL=qwen-plus

Use the ``-intl`` DashScope host with an international key; the mainland host
rejects it.  Prefer ``qwen-plus`` over ``qwen-max`` for extraction — it is a
high-volume path and the quality difference does not show up in structured
JSON output.

Leaving every key unset is a supported configuration: the router then reports
``backend == "ollama"`` and serves a local or Ollama-cloud model.
"""
from __future__ import annotations

import os

__all__ = ["llm_api_key", "llm_base_url", "llm_model", "llm_provider"]

_DEFAULT_BASE_URL = "https://api.deepseek.com/v1/chat/completions"
_DEFAULT_MODEL = "deepseek-chat"


def llm_api_key() -> str | None:
    """The cloud API key, or None when no cloud backend is configured."""
    return os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or None


def llm_base_url() -> str:
    """Chat-completions endpoint of the configured provider."""
    return os.environ.get("LLM_BASE_URL", _DEFAULT_BASE_URL)


def llm_model() -> str:
    """Default model name for the configured provider."""
    return os.environ.get("LLM_MODEL", _DEFAULT_MODEL)


def llm_provider() -> str:
    """Short provider label derived from the base URL, for logs and reports."""
    url = llm_base_url()
    for host, name in (
        ("deepseek", "deepseek"),
        ("dashscope", "qwen"),
        ("openai", "openai"),
        ("anthropic", "anthropic"),
    ):
        if host in url:
            return name
    return "custom"
