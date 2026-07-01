"""Densify structurally-sparse *signal* features (news / SEC / Wikipedia).

The problem
-----------
News, filings and pageviews only exist from a source's coverage floor onward
(GDELT 2017+, Wiki 2015+) and only for covered names. Left as nulls they are:

  1. **dropped by the reserve coverage gate** (``min_non_null_frac``), because
     they're null over most of the 2010→now panel — so all the news/SEC/wiki work
     never reaches the model; and
  2. **row-poison** for the trainer, whose ``drop_nulls`` on the feature set would
     delete every pre-coverage / uncovered-ticker row (collapsing the panel).

The fix — missingness indicators
--------------------------------
For each source we add a **presence flag** (1.0 where the source has data on that
row, else 0.0) and then fill the source's columns with a **neutral constant**.
This is the standard "impute + indicator" pattern: the filled columns are now
dense (they pass the gate and never force row drops), while the flag lets the
model distinguish *"neutral reading"* from *"no coverage"* and gate on it. The
fill values are constants known at all times, so nothing about this is
forward-looking.
"""
from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class SparseSpec:
    source: str
    flag: str
    fills: dict[str, float]      # column -> neutral fill value


# Neutral fills: sentiment/attention/counts → 0 (no signal); recency → a large
# "long ago" sentinel so an uncovered issuer doesn't look freshly-filed.
SPARSE_SIGNAL_SPECS: tuple[SparseSpec, ...] = (
    SparseSpec("news", "news_present",
               {"news_tone": 0.0, "news_tone_mom": 0.0, "news_buzz": 0.0}),
    SparseSpec("sec", "sec_present",
               {"sec_filings_30d": 0.0, "sec_8k_30d": 0.0, "sec_form4_90d": 0.0,
                "sec_days_since_filing": 999.0}),
    SparseSpec("wiki", "wiki_present",
               {"wiki_attention_z": 0.0, "wiki_attention_mom": 0.0}),
)

PRESENCE_FLAGS: list[str] = [s.flag for s in SPARSE_SIGNAL_SPECS]


def densify_sparse_signals(df: pl.DataFrame) -> pl.DataFrame:
    """Add per-source presence flags and neutral-fill the sparse signal columns.

    Only sources whose columns are actually present in ``df`` are touched, so this
    is a no-op for a build that didn't ingest a given source.
    """
    for spec in SPARSE_SIGNAL_SPECS:
        cols = [c for c in spec.fills if c in df.columns]
        if not cols:
            continue
        # presence = any of the source's columns is non-null on that row
        present = pl.any_horizontal([pl.col(c).is_not_null() for c in cols])
        df = df.with_columns(present.cast(pl.Float64).alias(spec.flag))
        df = df.with_columns([pl.col(c).fill_null(spec.fills[c]) for c in cols])
    return df
