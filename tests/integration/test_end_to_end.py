"""End-to-end: feature build -> ML training -> ranker strategy -> backtest."""
import polars as pl

from trading_system.backtesting import run_vectorized_backtest, compute_metrics
from trading_system.features import build_feature_matrix
from trading_system.models.train import FeatureSpec, train_walk_forward
from trading_system.strategies import MLRankerStrategy


def test_e2e_smoke(synthetic_ohlcv):
    feat = build_feature_matrix(synthetic_ohlcv)
    spec = FeatureSpec(
        feature_columns=[
            "mom_5d", "mom_20d", "mom_60d", "vol_20d", "vol_60d",
            "rsi_14", "rel_vol_20", "sma_gap_50", "breakout_20",
        ],
        target="forward_return_5d",
    )
    models, oos, metrics_df = train_walk_forward(feat, spec, train_years=1, test_years=1, step_years=1)
    if oos.is_empty():
        # Synthetic data may be too short; smoke pass without crash is the goal.
        return
    strat = MLRankerStrategy(predictions=oos, top_k=2, rebalance_days=10)
    weights = strat.generate_signals(feat)
    res = run_vectorized_backtest(synthetic_ohlcv, weights)
    m = compute_metrics(res.daily["net_ret"].to_numpy(), turnover=res.daily["turnover"].to_numpy())
    assert "Sharpe" in m
