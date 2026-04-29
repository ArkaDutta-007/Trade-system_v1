"""Daily pipeline: ingest -> validate -> features -> signals -> paper rebalance -> report.

Designed to be invocable from a Prefect flow or a plain cron job.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import polars as pl

from ..backtesting import run_vectorized_backtest, compute_metrics, summarize
from ..backtesting.slippage import CostModel
from ..config import Config, get_config
from ..execution.paper_broker import PaperBroker
from ..features import build_feature_matrix
from ..ingestion import ingest_universe
from ..models.shap_analysis import compute_shap_summary
from ..monitoring import attribute_pnl, emit_alert
from ..portfolio.order_policy import weights_to_orders
from ..portfolio.risk import RiskLimits, enforce_risk_limits
from ..quality import run_ohlcv_checks
from ..storage import DuckStore
from ..strategies import MomentumRotation
from ..utils import get_logger

logger = get_logger(__name__)


def run_daily_pipeline(cfg: Config | None = None) -> Path:
    """Run the full daily flow. Returns path to the daily report JSON."""
    cfg = cfg or get_config()
    bronze = cfg.path("data_bronze")
    reports = cfg.path("reports")
    reports.mkdir(parents=True, exist_ok=True)

    # 1. Ingest
    ohlcv_path = ingest_universe(cfg)
    ohlcv = pl.read_parquet(ohlcv_path)

    # 2. Validate
    quality = run_ohlcv_checks(ohlcv)
    failed = [k for k, v in quality.items() if not v]
    if failed:
        emit_alert("ERROR", "OHLCV quality checks failed", {"failed": failed})
        raise RuntimeError(f"Data quality failed: {failed}")

    # 3. Features
    features = build_feature_matrix(ohlcv, benchmark=cfg["universe"]["benchmark"])
    gold = cfg.path("data_gold")
    gold.mkdir(parents=True, exist_ok=True)
    features.write_parquet(gold / "features.parquet", compression="zstd")

    # 4. Strategy signals (Phase 1: momentum rotation as default)
    strat = MomentumRotation(lookback=126, top_k=4, rebalance_days=21)
    weights = strat.generate_signals(features)

    # 5. Risk overlay
    limits = RiskLimits(
        max_position_weight=cfg["backtest"]["max_position_weight"],
        max_gross_exposure=cfg["backtest"]["max_gross_exposure"],
        max_drawdown_kill_switch=cfg["risk"]["max_drawdown_kill_switch"],
    )
    weights = enforce_risk_limits(weights, limits)

    # 6. Backtest the entire history (used for monitoring + dashboard)
    cost = CostModel(
        commission_bps=cfg["backtest"]["commission_bps"],
        slippage_bps=cfg["backtest"]["slippage_bps"],
        spread_bps=cfg["backtest"]["spread_bps"],
    )
    res = run_vectorized_backtest(
        ohlcv, weights, cost=cost,
        signal_delay_days=cfg["backtest"]["signal_delay_days"],
        initial_cash=cfg["backtest"]["initial_cash"],
        max_position_weight=cfg["backtest"]["max_position_weight"],
        max_gross_exposure=cfg["backtest"]["max_gross_exposure"],
        benchmark=cfg["universe"]["benchmark"],
    )
    metrics = compute_metrics(
        res.daily["net_ret"].to_numpy(),
        turnover=res.daily["turnover"].to_numpy(),
        benchmark=res.benchmark_ret["ret"].to_numpy() if res.benchmark_ret is not None else None,
    )
    print(summarize(metrics))

    # 7. Today's target weights -> paper rebalance
    today = features["date"].max()
    todays_weights = (
        weights.filter(pl.col("date") == today)
        .select(["ticker", "weight"]).to_dict(as_series=False)
    )
    target = dict(zip(todays_weights["ticker"], todays_weights["weight"]))

    todays_prices = (
        ohlcv.filter(pl.col("date") == today)
        .select(["ticker", "adj_close"]).to_dict(as_series=False)
    )
    prices = dict(zip(todays_prices["ticker"], todays_prices["adj_close"]))

    broker = PaperBroker.from_journal(reports / "paper_broker.json", cost_bps=cost.total_bps)
    equity = broker.equity(prices) if broker.holdings else cfg["backtest"]["initial_cash"]
    orders = weights_to_orders(target, broker.holdings, prices, equity)
    fills = broker.submit(orders, prices)

    # 8. PnL attribution (last 252 days)
    recent_w = res.weights_used.tail(252)
    recent_px = ohlcv.filter(pl.col("date") >= recent_w["date"].min())
    pnl_attr = attribute_pnl(recent_w, recent_px)

    # 9. Paper portfolio decisions (ML/ensemble signal)
    from datetime import date as _date
    from ..execution.paper_portfolio import PaperPortfolio
    from ..decision import analyze_all as _analyze_all

    try:
        ml_decisions = _analyze_all(cfg)
        portfolio = PaperPortfolio(
            journal_path=gold / "paper_portfolio_journal.json",
            equity_log_path=gold / "paper_equity_log.parquet",
            initial_cash=cfg["backtest"].get("initial_cash", 100_000.0),
        )
        portfolio.process_decisions(ml_decisions, prices)
        portfolio.snapshot(_date.today(), prices)
        logger.info("Paper portfolio updated with today's ML decisions")
    except Exception as e:
        logger.warning(f"Paper portfolio update failed (non-fatal): {e}")

    # 10. Daily report
    report = {
        "run_at": datetime.utcnow().isoformat(),
        "as_of_date": str(today),
        "metrics": {k: (float(v) if hasattr(v, "__float__") else v) for k, v in metrics.items()},
        "todays_target_weights": target,
        "todays_orders": [
            {"ticker": o.ticker, "qty": o.qty, "side": o.side, "notional": o.notional}
            for o in orders
        ],
        "fills": [vars(f) for f in fills],
        "broker_holdings": broker.holdings,
        "broker_cash": broker.cash,
        "pnl_attribution_top": pnl_attr.head(10).to_dicts(),
    }
    report_path = reports / f"daily_{datetime.utcnow().strftime('%Y%m%d')}.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info(f"Daily report written: {report_path}")
    return report_path
