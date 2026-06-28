"""Shared progress + parallel-map helpers (rich-based, degrade gracefully).

One place for the "show a bar / fan out network work" pattern so every long loop
(ingest, analyze-all, earnings calendar, news backends) feels the same and never
becomes a black box.  Both helpers no-op cleanly when rich is unavailable or
output isn't a TTY (e.g. piped logs, CI).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, Sequence, TypeVar

from .logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")
R = TypeVar("R")


def _progress_ctx(description: str):
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, BarColumn,
        MofNCompleteColumn, TimeRemainingColumn,
    )
    return Progress(
        SpinnerColumn(),
        TextColumn(f"[bold blue]{description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("· {task.fields[last]}"),
        TimeRemainingColumn(),
    )


def track(iterable: Iterable[T], description: str = "working", total: int | None = None):
    """Yield items from ``iterable`` behind a rich progress bar (or plain on fallback)."""
    items = list(iterable) if total is None else iterable
    if total is None:
        total = len(items)  # type: ignore[arg-type]
    try:
        with _progress_ctx(description) as prog:
            task = prog.add_task(description, total=total, last="…")
            for it in items:
                yield it
                prog.update(task, advance=1, last=str(it)[:24])
    except Exception:
        yield from items  # rich missing / non-tty → silent passthrough


def parallel_map(
    fn: Callable[[T], R],
    items: Sequence[T],
    workers: int = 8,
    description: str = "working",
    label: Callable[[T], str] | None = None,
) -> list[R | None]:
    """Run ``fn`` over ``items`` concurrently with a progress bar.

    Results are returned **in input order**; a failing item yields ``None`` and is
    logged (so one bad ticker never aborts the batch). Use for network-bound work
    (yfinance, HTTP) — not CPU-bound loops.
    """
    items = list(items)
    n = len(items)
    if n == 0:
        return []
    workers = max(1, min(workers, n))
    results: list[R | None] = [None] * n
    fails = 0

    def _run(update=None):
        nonlocal fails
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(fn, it): i for i, it in enumerate(items)}
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:
                    fails += 1
                    logger.debug(f"{description}: item {items[i]!r} failed: {e}")
                if update:
                    lab = label(items[i]) if label else str(items[i])
                    update(lab[:24], fails)

    try:
        with _progress_ctx(description) as prog:
            task = prog.add_task(description, total=n, last="…")
            _run(lambda lab, nf: prog.update(
                task, advance=1, last=lab + (f" [red]{nf} fail[/red]" if nf else "")))
    except Exception:
        _run()
    return results
