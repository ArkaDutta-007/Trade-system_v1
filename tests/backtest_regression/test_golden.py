"""Golden backtest regression: the same input data + same config -> same metrics."""
import json
from pathlib import Path

import pytest

from trading_system.backtesting import run_vectorized_backtest, compute_metrics
from trading_system.backtesting.slippage import CostModel
from trading_system.strategies import MovingAverageCrossover

GOLDEN = Path(__file__).parent / "golden_ma_crossover.json"


def _run(synthetic_ohlcv) -> dict:
    strat = MovingAverageCrossover(fast=20, slow=60, benchmark="SPY")
    weights = strat.generate_signals(synthetic_ohlcv)
    res = run_vectorized_backtest(synthetic_ohlcv, weights, cost=CostModel(1.0, 2.0, 1.0))
    m = compute_metrics(res.daily["net_ret"].to_numpy(), turnover=res.daily["turnover"].to_numpy())
    return {k: float(v) for k, v in m.items()}


def test_golden_regenerable(synthetic_ohlcv):
    m = _run(synthetic_ohlcv)
    if not GOLDEN.exists():
        GOLDEN.write_text(json.dumps(m, indent=2))
        pytest.skip("Golden file generated.")
    expected = json.loads(GOLDEN.read_text())
    for k, v in expected.items():
        assert m[k] == pytest.approx(v, rel=1e-6, abs=1e-9), f"{k}: {m[k]} vs {v}"
