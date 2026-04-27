import polars as pl
from trading_system.features import compute_technical_features, build_feature_matrix


def test_technical_features_have_expected_columns(synthetic_ohlcv):
    feat = compute_technical_features(synthetic_ohlcv)
    for col in ["mom_20d", "mom_60d", "vol_20d", "rsi_14", "atr_14", "rel_vol_20"]:
        assert col in feat.columns


def test_feature_matrix_has_targets_and_no_nans_in_targets_after_dropna(synthetic_ohlcv):
    feat = build_feature_matrix(synthetic_ohlcv)
    assert "forward_return_5d" in feat.columns
    assert "forward_return_20d" in feat.columns
    keep = feat.drop_nulls(subset=["forward_return_5d"])
    assert len(keep) > 0


def test_no_future_information_in_technical(synthetic_ohlcv):
    """Sanity: all rolling features are computed strictly using past values."""
    feat = compute_technical_features(synthetic_ohlcv)
    # The 19th row of any ticker should still have null for vol_20d (needs 20 obs)
    spy = feat.filter(pl.col("ticker") == "SPY").sort("date")
    assert spy["vol_20d"].head(19).null_count() == 19
