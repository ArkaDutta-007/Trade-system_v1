"""Feature distribution drift checks. Uses scipy KS test as a portable fallback."""
from __future__ import annotations

import polars as pl


def detect_drift(
    reference: pl.DataFrame,
    current: pl.DataFrame,
    feature_columns: list[str],
    p_threshold: float = 0.01,
) -> pl.DataFrame:
    from scipy.stats import ks_2samp

    rows = []
    for col in feature_columns:
        if col not in reference.columns or col not in current.columns:
            continue
        a = reference[col].drop_nulls().to_numpy()
        b = current[col].drop_nulls().to_numpy()
        if len(a) < 30 or len(b) < 30:
            continue
        stat, p = ks_2samp(a, b)
        rows.append(
            {
                "feature": col,
                "ks_stat": float(stat),
                "p_value": float(p),
                "drift": bool(p < p_threshold),
                "ref_n": len(a),
                "cur_n": len(b),
            }
        )
    return pl.DataFrame(rows).sort("p_value")
