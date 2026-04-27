"""Decision pipeline tests using synthetic OHLCV (no network)."""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from trading_system.config import Config
from trading_system.decision import analyze_symbol
from trading_system.features import build_feature_matrix


@pytest.fixture
def cfg_in_tmp(tmp_path: Path, synthetic_ohlcv: pl.DataFrame) -> Config:
    """Create a Config rooted in tmp_path with the synthetic OHLCV pre-staged."""
    bronze = tmp_path / "data" / "bronze"
    bronze.mkdir(parents=True)
    synthetic_ohlcv.write_parquet(bronze / "ohlcv_daily.parquet", compression="zstd")
    gold = tmp_path / "data" / "gold"
    gold.mkdir(parents=True)
    feat = build_feature_matrix(synthetic_ohlcv, benchmark="SPY")
    feat.write_parquet(gold / "features.parquet", compression="zstd")
    (tmp_path / "data" / "silver").mkdir(parents=True)
    (tmp_path / "reports").mkdir(parents=True)

    raw = {
        "paths": {
            "data_raw": "data/raw", "data_bronze": "data/bronze",
            "data_silver": "data/silver", "data_gold": "data/gold",
            "reports": "reports", "duckdb": "data/warehouse.duckdb",
        },
        "universe": {
            "name": "test", "benchmark": "SPY",
            "tickers": ["SPY", "QQQ", "XLK", "XLF", "XLE"],
            "required": ["SPY", "QQQ"],
            "additions": ["XLK", "XLF", "XLE"],
        },
        "data": {"start_date": "2020-01-01", "end_date": None, "source": "yfinance"},
        "decision": {
            "buy_threshold": 0.005, "sell_threshold": -0.005,
            "rsi_overbought": 75.0, "rsi_oversold": 25.0,
            "min_avg_dollar_volume_20": 0,
        },
    }
    return Config(raw=raw, project_root=tmp_path)


def test_analyze_symbol_returns_decision(cfg_in_tmp):
    res = analyze_symbol("SPY", cfg=cfg_in_tmp, write_report=True)
    assert res.ticker == "SPY"
    assert res.stance in ("BUY", "HOLD", "SELL")
    assert 0 <= res.confidence <= 1
    assert res.groundings["technical"]["available"]
    assert res.groundings["regime"]["available"]
    assert res.groundings["cross_section"]["available"]
    assert Path(res.report_path).exists()
    assert Path(res.json_path).exists()


def test_decision_report_contains_groundings_sections(cfg_in_tmp):
    res = analyze_symbol("QQQ", cfg=cfg_in_tmp)
    md = Path(res.report_path).read_text()
    for section in ["Rationale", "Technical state", "Regime context",
                    "Cross-sectional position", "Recent events", "Model groundings"]:
        assert section in md


def test_decision_unknown_symbol_raises(cfg_in_tmp):
    with pytest.raises(ValueError):
        analyze_symbol("NOTATICKER_ZZZ", cfg=cfg_in_tmp, write_report=False)
