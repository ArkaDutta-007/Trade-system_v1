"""Parquet read/write helpers."""
from __future__ import annotations

from pathlib import Path

import polars as pl


def write_parquet(df: pl.DataFrame, path: str | Path, compression: str = "zstd") -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(p, compression=compression)
    return p


def read_parquet(path: str | Path) -> pl.DataFrame:
    return pl.read_parquet(path)


def write_partitioned(
    df: pl.DataFrame, root: str | Path, partition_cols: list[str], compression: str = "zstd"
) -> Path:
    """Hive-style partitioned write. Polars sink_parquet handles this directly."""
    root_p = Path(root)
    root_p.mkdir(parents=True, exist_ok=True)
    df.write_parquet(
        root_p,
        compression=compression,
        use_pyarrow=True,
        pyarrow_options={"partition_cols": partition_cols},
    )
    return root_p
