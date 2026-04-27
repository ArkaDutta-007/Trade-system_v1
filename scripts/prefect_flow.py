"""Prefect flow wrapping the daily pipeline. Schedule via `prefect deployment build/apply`."""
from __future__ import annotations

from prefect import flow, task

from trading_system.config import get_config
from trading_system.pipeline import run_daily_pipeline


@task(retries=2, retry_delay_seconds=60)
def daily_task():
    return run_daily_pipeline(get_config())


@flow(name="trading-system-daily")
def daily_flow():
    return daily_task()


if __name__ == "__main__":
    daily_flow()
