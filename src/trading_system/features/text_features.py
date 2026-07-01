"""FinBERT news-text features — new information, not a re-modeling of prices.

The rest of the reserve is derived from price/volume/macro. This adds the one
genuinely orthogonal signal: what the *news says*, scored by a finance-domain
transformer (FinBERT) rather than the naive lexicon. For each (ticker, date) it
produces a point-in-time, recency-decayed news-sentiment feature.

Design
------
* ``score_events_finbert`` runs FinBERT once per unique headline (batched,
  content-hash **disk-cached**), tagging each event with a net sentiment in
  [-1, 1] (P(pos) - P(neg)). torch + transformers are **lazy** optional deps
  (``pip install -e '.[text]'``); absent, it no-ops and the feature is simply
  omitted (the reserve's coverage gate drops it).
* ``compute_text_features`` aggregates scored news to (ticker, date) and applies
  a causal exponentially-weighted decay — strictly backward-looking, using
  ``known_at`` so a headline only affects the day it was known and after.

The BERT model is applied to *text*, never to the price series (where trees
already win) — the right place for a transformer here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

from ..utils import get_logger

logger = get_logger(__name__)

FINBERT_MODEL = "ProsusAI/finbert"
TEXT_COLUMNS = ["finbert_sent", "finbert_sent_mom", "finbert_news_30d"]


def finbert_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except Exception:
        return False


# ── scoring (lazy, cached) ─────────────────────────────────────────────────────

def _load_cache(cache_dir: Path | None) -> dict:
    if cache_dir is None:
        return {}
    p = Path(cache_dir) / "finbert_scores.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def _save_cache(cache_dir: Path | None, cache: dict) -> None:
    if cache_dir is None:
        return
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    try:
        (Path(cache_dir) / "finbert_scores.json").write_text(json.dumps(cache))
    except Exception as e:
        logger.debug(f"finbert cache write failed: {e}")


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()[:16]


def _score_texts_finbert(texts: list[str], model_name: str, batch_size: int = 32) -> list[float]:
    """Net sentiment P(pos)-P(neg) in [-1,1] per text via FinBERT. Lazy-imported."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from ..utils import get_compute_profile

    device = get_compute_profile().torch_device
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
    # FinBERT label order is [positive, negative, neutral]
    labels = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
    pi = labels.index("positive") if "positive" in labels else 0
    ni = labels.index("negative") if "negative" in labels else 1

    out: list[float] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tok(batch, return_tensors="pt", truncation=True, max_length=128,
                      padding=True).to(device)
            probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
            out.extend((probs[:, pi] - probs[:, ni]).tolist())
    return out


def score_events_finbert(
    events: pl.DataFrame,
    cache_dir: Path | None = None,
    model_name: str = FINBERT_MODEL,
) -> pl.DataFrame:
    """Add a ``finbert_sentiment`` column to events (cached by headline hash).

    No-ops (returns events unchanged, column all-null) if transformers is absent.
    """
    if events is None or events.is_empty():
        return events
    if not finbert_available():
        logger.info("transformers/torch not installed — skipping FinBERT (pip install -e '.[text]').")
        return events.with_columns(pl.lit(None, dtype=pl.Float64).alias("finbert_sentiment"))

    texts = events["summary"].fill_null("").to_list()
    cache = _load_cache(cache_dir)
    keys = [_hash(t) for t in texts]
    todo_idx = [i for i, k in enumerate(keys) if k not in cache and texts[i].strip()]
    if todo_idx:
        logger.info(f"FinBERT scoring {len(todo_idx)} new headlines…")
        scored = _score_texts_finbert([texts[i] for i in todo_idx], model_name)
        for i, s in zip(todo_idx, scored):
            cache[keys[i]] = round(float(s), 4)
        _save_cache(cache_dir, cache)

    sent = [cache.get(k) if texts[i].strip() else None for i, k in enumerate(keys)]
    return events.with_columns(pl.Series("finbert_sentiment", sent, dtype=pl.Float64))


# ── panel features ─────────────────────────────────────────────────────────────

def compute_text_features(
    features: pl.DataFrame,
    events: pl.DataFrame | None,
    cache_dir: Path | None = None,
    model_name: str = FINBERT_MODEL,
    half_life: int = 7,
    scored_events: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Join causal FinBERT news-sentiment features onto the (ticker, date) panel.

    ``scored_events`` lets callers/tests pass pre-scored events (with a
    ``finbert_sentiment`` column) to bypass the model.
    """
    scored = scored_events if scored_events is not None else (
        score_events_finbert(events, cache_dir, model_name) if events is not None else None
    )
    if scored is None or scored.is_empty() or "finbert_sentiment" not in scored.columns:
        return features
    if scored["finbert_sentiment"].null_count() == scored.height:
        return features  # nothing scored (no model)

    # daily per-ticker aggregate, keyed on known_at (point-in-time)
    daily = (
        scored.drop_nulls("finbert_sentiment")
        .with_columns(date=pl.col("known_at").dt.date())
        .explode("tickers").rename({"tickers": "ticker"})
        .group_by(["ticker", "date"])
        .agg(fb=pl.col("finbert_sentiment").mean(), n=pl.len())
    )

    out = features.join(daily, on=["ticker", "date"], how="left").sort(["ticker", "date"])
    out = out.with_columns(
        pl.col("fb").fill_null(0.0), pl.col("n").fill_null(0),
    ).with_columns(
        # causal EW decay of daily sentiment; rolling 30-row news count
        finbert_sent=pl.col("fb").ewm_mean(half_life=half_life, ignore_nulls=True).over("ticker"),
        finbert_news_30d=pl.col("n").rolling_sum(window_size=30, min_samples=1).over("ticker"),
    ).with_columns(
        # short vs long sentiment momentum (fear fading / building)
        finbert_sent_mom=(
            pl.col("fb").ewm_mean(half_life=3, ignore_nulls=True).over("ticker")
            - pl.col("finbert_sent")
        )
    ).drop(["fb", "n"])
    # rows before a ticker's first scored headline stay null → dropped by coverage gate
    return out.with_columns(
        pl.when(pl.col("finbert_news_30d") > 0).then(pl.col("finbert_sent")).otherwise(None).alias("finbert_sent"),
        pl.when(pl.col("finbert_news_30d") > 0).then(pl.col("finbert_sent_mom")).otherwise(None).alias("finbert_sent_mom"),
    )
