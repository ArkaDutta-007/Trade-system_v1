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


def _panel_with_degenerate_ticker():
    """A normal ticker + a flat-price / zero-volume ticker (RSI & rel_vol 0/0)."""
    from datetime import date, timedelta
    import math
    rows = []
    d0 = date(2021, 1, 1)
    for i in range(60):
        d = d0 + timedelta(days=i)
        # FLAT: constant price, zero volume -> roll_down==0 and avgvol==0
        rows.append({"date": d, "ticker": "FLAT", "open": 100.0, "high": 100.0,
                     "low": 100.0, "close": 100.0, "adj_close": 100.0, "volume": 0})
        # NORMAL: moving price + volume
        px = 100.0 + 5 * math.sin(i / 5)
        rows.append({"date": d, "ticker": "NORM", "open": px, "high": px + 1,
                     "low": px - 1, "close": px, "adj_close": px, "volume": 1_000 + i})
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def test_rsi_and_relvol_have_no_nan_floats_on_degenerate_windows():
    """0/0 in RSI (flat price) and rel_vol (zero volume) must yield neutral/null, not NaN."""
    feat = compute_technical_features(_panel_with_degenerate_ticker())
    assert feat["rsi_14"].is_nan().sum() == 0          # flat window -> 50, not NaN
    assert feat["rel_vol_20"].is_nan().sum() == 0       # zero-vol window -> null, not NaN
    # the flat ticker's settled RSI is the neutral 50
    flat_rsi = feat.filter(pl.col("ticker") == "FLAT")["rsi_14"].drop_nulls()
    assert len(flat_rsi) > 0 and (flat_rsi == 50.0).all()


def test_no_nan_floats_anywhere_in_technical_features():
    """No numeric technical column may carry a NaN-float (it evades is_not_null)."""
    feat = compute_technical_features(_panel_with_degenerate_ticker())
    num_cols = [c for c, t in feat.schema.items() if t in (pl.Float64, pl.Float32)]
    offenders = {c: feat[c].is_nan().sum() for c in num_cols if feat[c].is_nan().sum() > 0}
    assert not offenders, f"NaN-float columns: {offenders}"
