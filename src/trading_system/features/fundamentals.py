"""Fundamental features. Stub: real implementation requires SEC XBRL ingest.

Currently returns the input frame unchanged. Wire to SEC company facts when ready.
"""
from __future__ import annotations

import polars as pl


def compute_fundamental_features(df: pl.DataFrame) -> pl.DataFrame:
    return df
