.PHONY: install test test-fast quality features backtest daily dashboard lint

install:
	pip install -e ".[dev]"

test:
	pytest

test-fast:
	pytest tests/unit tests/data_quality

ingest:
	ts ingest

features:
	ts features

backtest:
	ts backtest momentum_rotation

train:
	ts train

daily:
	ts daily

dashboard:
	ts dashboard

lint:
	ruff check src tests
