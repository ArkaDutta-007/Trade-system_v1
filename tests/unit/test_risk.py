import polars as pl
from trading_system.portfolio.risk import enforce_risk_limits, RiskLimits


def test_enforce_risk_caps_individual_weight():
    df = pl.DataFrame(
        {"date": ["2024-01-01"] * 3, "ticker": ["A", "B", "C"], "weight": [0.5, 0.5, 0.5]}
    ).with_columns(pl.col("date").str.to_date())
    out = enforce_risk_limits(df, RiskLimits(max_position_weight=0.20, max_gross_exposure=1.0))
    assert out["weight"].max() <= 0.20 + 1e-9


def test_enforce_risk_scales_gross():
    df = pl.DataFrame(
        {"date": ["2024-01-01"] * 4, "ticker": list("ABCD"), "weight": [0.5, 0.5, 0.5, 0.5]}
    ).with_columns(pl.col("date").str.to_date())
    out = enforce_risk_limits(df, RiskLimits(max_position_weight=0.50, max_gross_exposure=1.0))
    assert out["weight"].abs().sum() <= 1.0 + 1e-9
