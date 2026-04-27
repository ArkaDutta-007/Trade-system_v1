"""DuckDB analytical store. Wraps a single .duckdb file plus Parquet attach helpers."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import duckdb
import polars as pl


class DuckStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def conn(self):
        c = duckdb.connect(str(self.db_path))
        try:
            yield c
        finally:
            c.close()

    def execute(self, sql: str, params: Iterable | None = None) -> None:
        with self.conn() as c:
            c.execute(sql, params or [])

    def query(self, sql: str, params: Iterable | None = None) -> pl.DataFrame:
        with self.conn() as c:
            arrow = c.execute(sql, params or []).arrow()
        return pl.from_arrow(arrow)

    def register_parquet(self, view: str, path: str | Path) -> None:
        sql = f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM read_parquet('{path}')"
        self.execute(sql)

    def write_table(self, name: str, df: pl.DataFrame, mode: str = "replace") -> None:
        with self.conn() as c:
            c.register("_tmp", df.to_arrow())
            if mode == "replace":
                c.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _tmp")
            elif mode == "append":
                c.execute(
                    f"CREATE TABLE IF NOT EXISTS {name} AS SELECT * FROM _tmp WHERE 1=0"
                )
                c.execute(f"INSERT INTO {name} SELECT * FROM _tmp")
            else:
                raise ValueError(mode)
            c.unregister("_tmp")

    def list_tables(self) -> list[str]:
        with self.conn() as c:
            return [r[0] for r in c.execute("SHOW TABLES").fetchall()]
