from trading_system.quality import run_ohlcv_checks


def test_ohlcv_quality_passes_on_synthetic(synthetic_ohlcv):
    r = run_ohlcv_checks(synthetic_ohlcv)
    for k, v in r.items():
        assert v, f"{k} failed"
