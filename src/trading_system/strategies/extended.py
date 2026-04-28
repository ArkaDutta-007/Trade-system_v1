"""Extended strategy library — 40+ strategies spanning:

  Trend-following (1-10)
  Mean-reversion (11-17)
  Volatility / risk (18-22)
  Factor combinations (23-28)
  Market-regime / macro-adaptive (29-33)
  Statistical / pairs (34-37)
  Novel composite strategies (38-44+)

All strategies conform to the Strategy protocol: they accept a Polars
``features`` DataFrame and return a (date, ticker, weight) frame.

Columns expected to be present in features (from build_feature_matrix):
  date, ticker, adj_close, open, high, low, close, volume,
  ret_1d, mom_5d, mom_10d, mom_20d, mom_60d, mom_120d, mom_12m1m,
  vol_20d, vol_60d, rsi_14, atr_14, sma_gap_10, sma_gap_20,
  sma_gap_50, sma_gap_200, breakout_20, breakdown_20,
  dd_from_high_60, rel_vol_20, avg_dollar_volume_20,
  bull_regime, high_vol_regime, excess_ret_1d
"""
from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl
import numpy as np

from .base import StrategyMeta

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _top_k_equal(df: pl.DataFrame, score_col: str, top_k: int) -> pl.DataFrame:
    """Return equal-weight signals for the top_k tickers by score_col each date."""
    df = df.with_columns(
        rk=pl.col(score_col).rank(method="ordinal", descending=True).over("date")
    )
    return df.with_columns(
        weight=pl.when(pl.col("rk") <= top_k).then(1.0 / top_k).otherwise(0.0)
    ).select(["date", "ticker", "weight"])


def _rebalance_forward_fill(
    df_rb: pl.DataFrame,
    all_features: pl.DataFrame,
    rebalance_days: int,
) -> pl.DataFrame:
    """Subsample to every rebalance_days, forward-fill in between."""
    dates = df_rb.select("date").unique().sort("date").with_row_index("i")
    keep = dates.filter((pl.col("i") % rebalance_days) == 0).select("date")
    sparse = df_rb.join(keep, on="date", how="inner")

    all_dates = all_features.select("date").unique().sort("date")
    all_pairs = all_dates.join(df_rb.select("ticker").unique(), how="cross")
    out = all_pairs.join(sparse, on=["date", "ticker"], how="left").sort(["ticker", "date"])
    return out.with_columns(
        weight=pl.col("weight").fill_null(strategy="forward").over("ticker").fill_null(0.0)
    )


# ─────────────────────────────────────────────────────────────────────────────
# ── TREND-FOLLOWING ──────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DualMomentumAbsolute:
    """1. Gary Antonacci-style Dual Momentum.
    Long top-k if their absolute 12-1m momentum > 0 (risk-on), else cash."""
    lookback: int = 252
    skip_month: int = 21
    top_k: int = 5
    cash_ticker: str = "SPY"
    rebalance_days: int = 21
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="dual_momentum_absolute", description="Absolute+relative dual momentum (Antonacci)"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        col = "mom_12m1m"
        if col not in features.columns:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = features.select(["date", "ticker", col]).drop_nulls(col)
        df = df.with_columns(
            rk=pl.col(col).rank(method="ordinal", descending=True).over("date"),
            abs_ok=(pl.col(col) > 0).cast(pl.Float64),
        )
        df = df.with_columns(
            weight=pl.when((pl.col("rk") <= self.top_k) & (pl.col("abs_ok") == 1.0))
                     .then(1.0 / self.top_k)
                     .otherwise(0.0)
        ).select(["date", "ticker", "weight"])
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class TrendFollowingEMA:
    """2. EMA crossover on individual stocks — long when fast EMA > slow EMA."""
    fast: int = 21
    slow: int = 63
    top_k: int = 10
    rebalance_days: int = 5
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="trend_ema_cross", description="Per-stock EMA crossover, top-k trending"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        df = features.sort(["ticker", "date"])
        df = df.with_columns(
            ema_fast=pl.col("adj_close").ewm_mean(span=self.fast).over("ticker"),
            ema_slow=pl.col("adj_close").ewm_mean(span=self.slow).over("ticker"),
        ).with_columns(
            trend_score=(pl.col("ema_fast") / pl.col("ema_slow") - 1)
        )
        df = df.with_columns(
            weight=pl.when(pl.col("ema_fast") > pl.col("ema_slow"))
                     .then(1.0 / self.top_k)
                     .otherwise(0.0)
        )
        return _rebalance_forward_fill(
            df.select(["date", "ticker", "weight"]), features, self.rebalance_days
        )


@dataclass
class TripleMovingAverageCrossover:
    """3. Triple MA (10/50/200): long only when price > 50d AND 50d > 200d."""
    top_k: int = 8
    rebalance_days: int = 5
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="triple_ma", description="Long when 10d > 50d > 200d SMA"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        needed = {"sma_gap_50", "sma_gap_200", "sma_gap_10"}
        if not needed.issubset(features.columns):
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = features.with_columns(
            score=pl.when(
                (pl.col("sma_gap_10") > 0) & (pl.col("sma_gap_50") > 0) & (pl.col("sma_gap_200") > 0)
            ).then(pl.col("mom_20d")).otherwise(None)
        ).drop_nulls("score")
        df = _top_k_equal(df, "score", self.top_k)
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class PriceMomentumWithVolFilter:
    """4. Momentum filtered by low realized vol (risk-adjusted momentum)."""
    lookback: int = 120
    vol_cap: float = 0.35   # exclude if annualized vol > 35%
    top_k: int = 6
    rebalance_days: int = 21
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="mom_vol_filter", description="Momentum top-k with vol cap"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        col = "mom_120d" if "mom_120d" in features.columns else "mom_60d"
        df = features.select(["date", "ticker", col, "vol_20d"]).drop_nulls()
        df = df.filter(pl.col("vol_20d") < self.vol_cap)
        df = _top_k_equal(df, col, self.top_k)
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class BreakoutStrategy:
    """5. Buy stocks making new 20-day highs with above-average volume."""
    vol_mult: float = 1.5   # relative volume threshold
    top_k: int = 8
    hold_days: int = 10
    weight_per: float = 0.10
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="breakout_20d_high", description="New 20d high + volume surge"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        if "breakout_20" not in features.columns:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = features.sort(["ticker", "date"])
        entry = (
            (pl.col("breakout_20") >= -0.005) &   # at/near 20d high
            (pl.col("rel_vol_20") > self.vol_mult)
        ).cast(pl.Int8)
        df = df.with_columns(entry=entry).with_columns(
            in_trade=pl.col("entry").rolling_sum(window_size=self.hold_days, min_periods=1)
                        .over("ticker").clip(0, 1)
        )
        return df.select(
            "date", "ticker",
            (pl.col("in_trade").cast(pl.Float64) * self.weight_per).alias("weight"),
        )


@dataclass
class TurtleTrendFollowing:
    """6. Donchian channel breakout (Turtle Trading). Enter on 20d high, exit on 10d low."""
    entry_window: int = 20
    exit_window: int = 10
    weight_per: float = 0.08
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="turtle_donchian", description="Donchian channel breakout (Turtle)"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        df = features.sort(["ticker", "date"])
        df = df.with_columns(
            chan_high=pl.col("high").rolling_max(window_size=self.entry_window).over("ticker"),
            chan_low=pl.col("low").rolling_min(window_size=self.exit_window).over("ticker"),
        ).with_columns(
            in_trade=pl.when(pl.col("close") >= pl.col("chan_high")).then(1.0)
                       .when(pl.col("close") <= pl.col("chan_low")).then(0.0)
                       .otherwise(None)
        ).with_columns(
            in_trade=pl.col("in_trade").fill_null(strategy="forward").over("ticker").fill_null(0.0)
        )
        return df.select(
            "date", "ticker",
            (pl.col("in_trade") * self.weight_per).alias("weight")
        )


@dataclass
class AdaptiveTrendWithRegime:
    """7. Momentum rotation that turns off when bull_regime is False."""
    top_k: int = 5
    rebalance_days: int = 10
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="adaptive_trend_regime", description="Momentum top-k only in bull regime"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        col = "mom_20d"
        df = features.select(["date", "ticker", col, "bull_regime"]).drop_nulls(col)
        df = df.filter(pl.col("bull_regime") == 1 if "bull_regime" in features.columns else pl.lit(True))
        if df.is_empty():
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = _top_k_equal(df, col, self.top_k)
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class RateOfChangeMomentum:
    """8. Pure ROC (Rate of Change) momentum over 60 days."""
    roc_window: int = 60
    top_k: int = 6
    rebalance_days: int = 21
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="roc_momentum", description="Rate-of-Change 60d momentum"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        col = f"mom_{self.roc_window}d"
        if col not in features.columns:
            col = "mom_60d"
        df = features.select(["date", "ticker", col]).drop_nulls(col)
        df = _top_k_equal(df, col, self.top_k)
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class WeeklyMomentumRotation:
    """9. Short-horizon 5-day momentum rotation, rebalanced weekly."""
    top_k: int = 10
    rebalance_days: int = 5
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="weekly_mom_rotation", description="Weekly 5d momentum top-k"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        df = features.select(["date", "ticker", "mom_5d"]).drop_nulls("mom_5d")
        df = _top_k_equal(df, "mom_5d", self.top_k)
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class TimeSeriesMomentum:
    """10. Time-series momentum: long if mom > 0, short (zero) if mom < 0, per-stock."""
    lookback: int = 120
    max_positions: int = 15
    weight_per: float = 0.067
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="time_series_momentum", description="Long only when own momentum > 0"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        col = "mom_120d" if "mom_120d" in features.columns else "mom_60d"
        df = features.select(["date", "ticker", col]).drop_nulls(col)
        df = df.with_columns(
            weight=pl.when(pl.col(col) > 0).then(self.weight_per).otherwise(0.0)
        )
        # Cap number of positions: keep top max_positions by momentum strength
        df = df.with_columns(
            rk=pl.col(col).abs().rank(method="ordinal", descending=True).over("date")
        ).with_columns(
            weight=pl.when((pl.col("rk") <= self.max_positions) & (pl.col("weight") > 0))
                     .then(pl.col("weight")).otherwise(0.0)
        ).select(["date", "ticker", "weight"])
        return df


# ─────────────────────────────────────────────────────────────────────────────
# ── MEAN REVERSION ───────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RSIOversoldBounce:
    """11. Long when RSI < 30 (oversold), exit when RSI > 50."""
    entry_rsi: float = 30.0
    exit_rsi: float = 50.0
    weight_per: float = 0.08
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="rsi_oversold_bounce", description="Long on RSI<30, exit RSI>50"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        if "rsi_14" not in features.columns:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = features.sort(["ticker", "date"])
        df = df.with_columns(
            entry=pl.when(pl.col("rsi_14") < self.entry_rsi).then(1.0)
                    .when(pl.col("rsi_14") > self.exit_rsi).then(0.0)
                    .otherwise(None)
        ).with_columns(
            in_trade=pl.col("entry").fill_null(strategy="forward").over("ticker").fill_null(0.0)
        )
        return df.select("date", "ticker",
                         (pl.col("in_trade") * self.weight_per).alias("weight"))


@dataclass
class BollingerBandReversion:
    """12. Long when close touches lower Bollinger Band (mean-reversion entry)."""
    window: int = 20
    n_std: float = 2.0
    hold_days: int = 5
    weight_per: float = 0.08
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="bollinger_reversion", description="BB lower-band mean reversion"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        df = features.sort(["ticker", "date"])
        df = df.with_columns(
            bb_mid=pl.col("adj_close").rolling_mean(window_size=self.window).over("ticker"),
            bb_std=pl.col("adj_close").rolling_std(window_size=self.window).over("ticker"),
        ).with_columns(
            bb_lower=pl.col("bb_mid") - self.n_std * pl.col("bb_std"),
        ).with_columns(
            entry=(pl.col("adj_close") <= pl.col("bb_lower")).cast(pl.Int8)
        ).with_columns(
            in_trade=pl.col("entry").rolling_sum(window_size=self.hold_days, min_periods=1)
                        .over("ticker").clip(0, 1)
        )
        return df.select("date", "ticker",
                         (pl.col("in_trade").cast(pl.Float64) * self.weight_per).alias("weight"))


@dataclass
class OvernightGapFill:
    """13. Buy overnight gaps down (open << prev close) expecting gap fill."""
    gap_threshold: float = -0.02   # 2% gap down
    hold_days: int = 3
    weight_per: float = 0.06
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="overnight_gap_fill", description="Long on gap-down opens, hold N days"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        if "open" not in features.columns or "adj_close" not in features.columns:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = features.sort(["ticker", "date"])
        df = df.with_columns(
            prev_close=pl.col("adj_close").shift(1).over("ticker"),
        ).with_columns(
            gap=(pl.col("open") / pl.col("prev_close") - 1),
        ).with_columns(
            entry=(pl.col("gap") < self.gap_threshold).cast(pl.Int8)
        ).with_columns(
            in_trade=pl.col("entry").rolling_sum(window_size=self.hold_days, min_periods=1)
                        .over("ticker").clip(0, 1)
        )
        return df.select("date", "ticker",
                         (pl.col("in_trade").cast(pl.Float64) * self.weight_per).alias("weight"))


@dataclass
class SMACrossoverMeanReversion:
    """14. Short-term SMA mean reversion: buy when price > 10d SMA by > 5% (pullback expected)."""
    stretch: float = 0.05
    hold_days: int = 5
    weight_per: float = 0.07
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="sma_stretch_reversion", description="Sell stretch > 5% above 10d SMA"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        if "sma_gap_10" not in features.columns:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = features.sort(["ticker", "date"])
        # Contrarian: buy when stretched DOWN (sma_gap < -stretch)
        entry = (pl.col("sma_gap_10") < -self.stretch).cast(pl.Int8)
        df = df.with_columns(entry=entry).with_columns(
            in_trade=pl.col("entry").rolling_sum(window_size=self.hold_days, min_periods=1)
                        .over("ticker").clip(0, 1)
        )
        return df.select("date", "ticker",
                         (pl.col("in_trade").cast(pl.Float64) * self.weight_per).alias("weight"))


@dataclass
class HighFrequencyMeanReversion:
    """15. Intraday-proxy: mean revert on 1-day losers within an up-trending universe."""
    drop_threshold: float = -0.04
    universe_up_days: int = 5
    hold_days: int = 2
    weight_per: float = 0.06
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="hf_mean_reversion", description="1-day loser reversion in up-trending market"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        df = features.sort(["ticker", "date"])
        df = df.with_columns(
            market_up=(pl.col("mom_5d").mean().over("date") > 0).cast(pl.Int8)
        )
        entry = (
            (pl.col("ret_1d") < self.drop_threshold) &
            (pl.col("market_up") == 1)
        ).cast(pl.Int8)
        df = df.with_columns(entry=entry).with_columns(
            in_trade=pl.col("entry").rolling_sum(window_size=self.hold_days, min_periods=1)
                        .over("ticker").clip(0, 1)
        )
        return df.select("date", "ticker",
                         (pl.col("in_trade").cast(pl.Float64) * self.weight_per).alias("weight"))


@dataclass
class DrawdownBounce:
    """16. Buy stocks with deep drawdowns from 60-day highs (> 15%) expecting recovery."""
    max_dd_threshold: float = -0.15
    hold_days: int = 10
    weight_per: float = 0.07
    require_bull: bool = True
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="drawdown_bounce", description="Deep drawdown recovery play"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        if "dd_from_high_60" not in features.columns:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = features.sort(["ticker", "date"])
        cond = pl.col("dd_from_high_60") < self.max_dd_threshold
        if self.require_bull and "bull_regime" in df.columns:
            cond = cond & (pl.col("bull_regime") == 1)
        df = df.with_columns(entry=cond.cast(pl.Int8)).with_columns(
            in_trade=pl.col("entry").rolling_sum(window_size=self.hold_days, min_periods=1)
                        .over("ticker").clip(0, 1)
        )
        return df.select("date", "ticker",
                         (pl.col("in_trade").cast(pl.Float64) * self.weight_per).alias("weight"))


@dataclass
class LowRSIMomentumCombo:
    """17. RSI oversold + positive 20d momentum (quality mean reversion)."""
    rsi_threshold: float = 35.0
    mom_positive: bool = True
    hold_days: int = 7
    weight_per: float = 0.08
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="low_rsi_mom_combo", description="RSI < 35 AND positive momentum"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        if "rsi_14" not in features.columns:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = features.sort(["ticker", "date"])
        cond = pl.col("rsi_14") < self.rsi_threshold
        if self.mom_positive:
            cond = cond & (pl.col("mom_20d") > 0)
        df = df.with_columns(entry=cond.cast(pl.Int8)).with_columns(
            in_trade=pl.col("entry").rolling_sum(window_size=self.hold_days, min_periods=1)
                        .over("ticker").clip(0, 1)
        )
        return df.select("date", "ticker",
                         (pl.col("in_trade").cast(pl.Float64) * self.weight_per).alias("weight"))


# ─────────────────────────────────────────────────────────────────────────────
# ── VOLATILITY / RISK ────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MinimumVolatilityPortfolio:
    """18. Equal-weight the N stocks with the lowest realized 20d vol (defensive)."""
    n_stocks: int = 10
    rebalance_days: int = 21
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="min_vol_portfolio", description="N lowest-vol stocks, equal weight"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        df = features.select(["date", "ticker", "vol_20d"]).drop_nulls("vol_20d")
        df = df.with_columns(
            rk=pl.col("vol_20d").rank(method="ordinal", descending=False).over("date")
        ).with_columns(
            weight=pl.when(pl.col("rk") <= self.n_stocks)
                     .then(1.0 / self.n_stocks).otherwise(0.0)
        ).select(["date", "ticker", "weight"])
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class VolatilityBreakout:
    """19. Long when today's vol >> recent average (vol expansion = directional move)."""
    vol_multiple: float = 2.0
    mom_condition: bool = True   # require positive momentum too
    hold_days: int = 3
    weight_per: float = 0.07
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="vol_breakout", description="Vol expansion breakout"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        df = features.sort(["ticker", "date"])
        cond = pl.col("rel_vol_20") > self.vol_multiple
        if self.mom_condition:
            cond = cond & (pl.col("ret_1d") > 0)
        df = df.with_columns(entry=cond.cast(pl.Int8)).with_columns(
            in_trade=pl.col("entry").rolling_sum(window_size=self.hold_days, min_periods=1)
                        .over("ticker").clip(0, 1)
        )
        return df.select("date", "ticker",
                         (pl.col("in_trade").cast(pl.Float64) * self.weight_per).alias("weight"))


@dataclass
class RiskParityVolTargeting:
    """20. Inverse-vol weighting: weight ∝ 1/vol, normalized to full investment."""
    n_stocks: int = 20
    vol_floor: float = 0.05
    rebalance_days: int = 21
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="risk_parity_vol_target", description="Inverse-vol risk parity"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        df = features.select(["date", "ticker", "vol_20d", "mom_20d"]).drop_nulls()
        # Filter to top-n_stocks by momentum first, then vol-weight them
        df = df.with_columns(
            mom_rk=pl.col("mom_20d").rank(method="ordinal", descending=True).over("date")
        ).filter(pl.col("mom_rk") <= self.n_stocks)
        df = df.with_columns(
            inv_vol=1.0 / (pl.col("vol_20d").clip(self.vol_floor, None))
        ).with_columns(
            inv_vol_sum=pl.col("inv_vol").sum().over("date")
        ).with_columns(
            weight=(pl.col("inv_vol") / pl.col("inv_vol_sum")).clip(0.0, 0.15)
        ).select(["date", "ticker", "weight"])
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class VIXContrarianStrategy:
    """21. Rotate into equities when high_vol_regime is False (low VIX = confidence)."""
    top_k: int = 8
    rebalance_days: int = 5
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="vix_contrarian", description="Equity long when low-vol regime"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        col = "mom_20d"
        if "high_vol_regime" not in features.columns:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = features.filter(pl.col("high_vol_regime") == 0).select(["date", "ticker", col]).drop_nulls()
        if df.is_empty():
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = _top_k_equal(df, col, self.top_k)
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class ATRPositionSizing:
    """22. Equal-rank selection, but position size ∝ 1/ATR (tighter ATR = bigger bet)."""
    top_k: int = 10
    rebalance_days: int = 10
    atr_floor: float = 0.5
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="atr_position_sizing", description="ATR-sized positions on top-k momentum"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        if "atr_14" not in features.columns:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = features.select(["date", "ticker", "mom_20d", "atr_14", "adj_close"]).drop_nulls()
        df = df.with_columns(
            mom_rk=pl.col("mom_20d").rank(method="ordinal", descending=True).over("date")
        ).filter(pl.col("mom_rk") <= self.top_k)
        df = df.with_columns(
            atr_pct=pl.col("atr_14") / pl.col("adj_close"),
            inv_atr=1.0 / (pl.col("atr_14") / pl.col("adj_close")).clip(self.atr_floor / 100, None),
        ).with_columns(
            inv_atr_sum=pl.col("inv_atr").sum().over("date")
        ).with_columns(
            weight=(pl.col("inv_atr") / pl.col("inv_atr_sum")).clip(0.0, 0.15)
        ).select(["date", "ticker", "weight"])
        return _rebalance_forward_fill(df, features, self.rebalance_days)


# ─────────────────────────────────────────────────────────────────────────────
# ── FACTOR COMBINATIONS ──────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MomentumQualityCombo:
    """23. Score = momentum rank + liquidity rank (high quality + trending)."""
    top_k: int = 8
    rebalance_days: int = 10
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="mom_quality_combo", description="Momentum + liquidity composite"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        df = features.select(["date", "ticker", "mom_20d", "avg_dollar_volume_20"]).drop_nulls()
        df = df.with_columns(
            mom_rk=pl.col("mom_20d").rank(method="ordinal", descending=True).over("date"),
            liq_rk=pl.col("avg_dollar_volume_20").rank(method="ordinal", descending=True).over("date"),
        ).with_columns(
            combo_score=(pl.col("mom_rk") + pl.col("liq_rk"))
        ).with_columns(
            rk=pl.col("combo_score").rank(method="ordinal", descending=False).over("date")
        ).with_columns(
            weight=pl.when(pl.col("rk") <= self.top_k).then(1.0 / self.top_k).otherwise(0.0)
        ).select(["date", "ticker", "weight"])
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class MomentumValueMix:
    """24. Blend momentum (6m) with value proxy (below 200d SMA = cheaper)."""
    top_k: int = 8
    rebalance_days: int = 21
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="momentum_value_mix", description="Momentum + value (SMA discount) blend"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        needed = {"mom_120d", "sma_gap_200"}
        if not needed.issubset(features.columns):
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = features.select(["date", "ticker", "mom_120d", "sma_gap_200"]).drop_nulls()
        df = df.with_columns(
            mom_rk=pl.col("mom_120d").rank(method="ordinal", descending=True).over("date"),
            # "value": stocks furthest below 200d SMA (mean reversion potential)
            val_rk=pl.col("sma_gap_200").rank(method="ordinal", descending=False).over("date"),
        ).with_columns(
            score=(0.6 * pl.col("mom_rk") + 0.4 * pl.col("val_rk"))
        ).with_columns(
            rk=pl.col("score").rank(method="ordinal", descending=False).over("date")
        ).with_columns(
            weight=pl.when(pl.col("rk") <= self.top_k).then(1.0 / self.top_k).otherwise(0.0)
        ).select(["date", "ticker", "weight"])
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class LowVolMomentumFactor:
    """25. AQR BAB-inspired: low vol stocks with positive momentum."""
    vol_percentile: float = 0.40   # bottom 40% by vol
    top_k: int = 8
    rebalance_days: int = 21
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="low_vol_momentum", description="Low-vol subset momentum (BAB-inspired)"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        df = features.select(["date", "ticker", "mom_20d", "vol_20d"]).drop_nulls()
        n_per_date = df.group_by("date").agg(pl.len().alias("n"))
        df = df.join(n_per_date, on="date")
        cutoff = (pl.col("n") * self.vol_percentile).cast(pl.Int32)
        df = df.with_columns(
            vol_rk=pl.col("vol_20d").rank(method="ordinal", descending=False).over("date")
        ).filter(pl.col("vol_rk") <= cutoff)
        if df.is_empty():
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = _top_k_equal(df, "mom_20d", self.top_k)
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class SectorRotationProxy:
    """26. Rotate between ETF proxies using cross-sectional relative momentum.
    Works on any universe; non-ETF tickers simply compete normally."""
    lookback: int = 63
    top_k: int = 5
    rebalance_days: int = 21
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="sector_rotation_proxy", description="ETF sector rotation via relative momentum"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        col = "mom_60d" if "mom_60d" in features.columns else "mom_20d"
        df = features.select(["date", "ticker", col]).drop_nulls(col)
        df = _top_k_equal(df, col, self.top_k)
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class TrendPlusMeanReversionBlend:
    """27. 50/50 blend: half the book in top-k momentum, half in RSI-oversold rebound."""
    top_k: int = 5
    rsi_threshold: float = 35.0
    rebalance_days: int = 10
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="trend_plus_reversion", description="50/50 trend + mean-reversion blend"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        # Trend leg
        df_t = features.select(["date", "ticker", "mom_20d"]).drop_nulls("mom_20d")
        df_t = df_t.with_columns(
            rk=pl.col("mom_20d").rank(method="ordinal", descending=True).over("date")
        ).with_columns(
            w_trend=pl.when(pl.col("rk") <= self.top_k).then(0.5 / self.top_k).otherwise(0.0)
        ).select(["date", "ticker", "w_trend"])

        # Reversion leg
        if "rsi_14" in features.columns:
            df_r = features.select(["date", "ticker", "rsi_14"]).drop_nulls("rsi_14")
            df_r = df_r.with_columns(
                rk=pl.col("rsi_14").rank(method="ordinal", descending=False).over("date")
            ).with_columns(
                w_rev=pl.when((pl.col("rk") <= self.top_k) & (pl.col("rsi_14") < self.rsi_threshold))
                        .then(0.5 / self.top_k).otherwise(0.0)
            ).select(["date", "ticker", "w_rev"])
            df = df_t.join(df_r, on=["date", "ticker"], how="left").with_columns(
                weight=(pl.col("w_trend") + pl.col("w_rev").fill_null(0.0))
            )
        else:
            df = df_t.rename({"w_trend": "weight"})
        return _rebalance_forward_fill(df.select(["date", "ticker", "weight"]), features, self.rebalance_days)


@dataclass
class CrossSectionalRankZ:
    """28. Z-score normalize multiple factors, combine linearly, take top-k."""
    top_k: int = 8
    rebalance_days: int = 10
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="cross_sectional_rankz", description="Multi-factor z-score composite"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        factors = [f for f in ["mom_20d", "mom_60d", "vol_20d", "rsi_14"] if f in features.columns]
        if len(factors) < 2:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = features.select(["date", "ticker"] + factors).drop_nulls()
        # z-score each factor across tickers per date
        for f in factors:
            df = df.with_columns(
                zf=((pl.col(f) - pl.col(f).mean().over("date")) /
                    (pl.col(f).std().over("date") + 1e-8)).alias(f"z_{f}")
            )
        # Invert vol z-score (lower vol = better)
        z_cols = [f"z_{f}" for f in factors]
        expr = sum(
            pl.col(c) * (-1 if "vol" in c or "rsi" in c else 1)
            for c in z_cols
        )
        df = df.with_columns(composite=expr)
        df = _top_k_equal(df, "composite", self.top_k)
        return _rebalance_forward_fill(df, features, self.rebalance_days)


# ─────────────────────────────────────────────────────────────────────────────
# ── REGIME / MACRO ADAPTIVE ──────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RegimeSwitchingMomentum:
    """29. Bull regime → momentum rotation; Bear regime → min-vol defensive."""
    top_k_bull: int = 6
    top_k_bear: int = 5
    rebalance_days: int = 10
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="regime_switching_momentum", description="Momentum in bull, min-vol in bear"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        if "bull_regime" not in features.columns:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        bull = features.filter(pl.col("bull_regime") == 1)
        bear = features.filter(pl.col("bull_regime") == 0)

        frames = []
        if not bull.is_empty():
            df_b = bull.select(["date", "ticker", "mom_20d"]).drop_nulls("mom_20d")
            df_b = _top_k_equal(df_b, "mom_20d", self.top_k_bull)
            frames.append(df_b)
        if not bear.is_empty():
            df_d = bear.select(["date", "ticker", "vol_20d"]).drop_nulls("vol_20d")
            df_d = df_d.with_columns(
                rk=pl.col("vol_20d").rank(method="ordinal", descending=False).over("date")
            ).with_columns(
                weight=pl.when(pl.col("rk") <= self.top_k_bear)
                         .then(1.0 / self.top_k_bear).otherwise(0.0)
            ).select(["date", "ticker", "weight"])
            frames.append(df_d)

        if not frames:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = pl.concat(frames)
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class HighVolRegimeCash:
    """30. Go to cash (zero weight) during high-vol regime; momentum otherwise."""
    top_k: int = 8
    rebalance_days: int = 5
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="high_vol_cash", description="Cash during VIX spikes, momentum otherwise"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        if "high_vol_regime" not in features.columns:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = features.filter(pl.col("high_vol_regime") == 0)
        if df.is_empty():
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = df.select(["date", "ticker", "mom_20d"]).drop_nulls("mom_20d")
        df = _top_k_equal(df, "mom_20d", self.top_k)
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class TrendStrengthFilter:
    """31. Only enter stocks with strong trend: mom > 0 AND price above all MAs."""
    top_k: int = 8
    rebalance_days: int = 5
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="trend_strength_filter", description="Momentum with multi-MA alignment"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        needed = {"sma_gap_50", "sma_gap_200", "mom_20d"}
        if not needed.issubset(features.columns):
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = features.filter(
            (pl.col("sma_gap_50") > 0) & (pl.col("sma_gap_200") > 0) & (pl.col("mom_20d") > 0)
        ).select(["date", "ticker", "mom_20d"]).drop_nulls()
        if df.is_empty():
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = _top_k_equal(df, "mom_20d", self.top_k)
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class MarketCapWeightedMomentum:
    """32. Weight by dollar volume (proxy for market cap) within top-k momentum."""
    top_k: int = 10
    rebalance_days: int = 21
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="mkt_cap_weighted_mom", description="Dollar-vol weighted momentum rotation"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        df = features.select(["date", "ticker", "mom_20d", "avg_dollar_volume_20"]).drop_nulls()
        df = df.with_columns(
            rk=pl.col("mom_20d").rank(method="ordinal", descending=True).over("date")
        ).filter(pl.col("rk") <= self.top_k)
        df = df.with_columns(
            dv_sum=pl.col("avg_dollar_volume_20").sum().over("date")
        ).with_columns(
            weight=(pl.col("avg_dollar_volume_20") / pl.col("dv_sum")).clip(0.0, 0.20)
        ).select(["date", "ticker", "weight"])
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class YieldCurveAdaptive:
    """33. Uses yield_curve_slope feature if available; bull-regime proxy otherwise."""
    top_k: int = 6
    rebalance_days: int = 21
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="yield_curve_adaptive", description="Adapt allocation to yield curve slope"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        # If yield-curve slope is in features (from FRED ingestion), use it
        if "yield_curve_10y2y" in features.columns:
            # Positive slope → risk-on (momentum); negative → defensive (min-vol)
            df = features.sort(["ticker", "date"])
            frames = []
            for stance, grp in [("risk_on", df.filter(pl.col("yield_curve_10y2y") >= 0)),
                                 ("risk_off", df.filter(pl.col("yield_curve_10y2y") < 0))]:
                if grp.is_empty():
                    continue
                col = "mom_20d" if stance == "risk_on" else "vol_20d"
                desc = stance == "risk_on"
                sg = grp.select(["date", "ticker", col]).drop_nulls(col)
                sg = sg.with_columns(
                    rk=pl.col(col).rank(method="ordinal", descending=desc).over("date")
                ).with_columns(
                    weight=pl.when(pl.col("rk") <= self.top_k).then(1.0 / self.top_k).otherwise(0.0)
                ).select(["date", "ticker", "weight"])
                frames.append(sg)
            df = pl.concat(frames) if frames else features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        else:
            df = features.select(["date", "ticker", "mom_20d"]).drop_nulls("mom_20d")
            df = _top_k_equal(df, "mom_20d", self.top_k)
        return _rebalance_forward_fill(df, features, self.rebalance_days)


# ─────────────────────────────────────────────────────────────────────────────
# ── STATISTICAL / PAIRS ──────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RelativeMomentumSpread:
    """34. Buy the top decile by excess return vs. universe mean; sell bottom decile (long-only: hold top only)."""
    top_k: int = 8
    rebalance_days: int = 5
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="relative_momentum_spread", description="Excess return vs universe mean, top-k long"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        df = features.select(["date", "ticker", "excess_ret_1d"]).drop_nulls("excess_ret_1d")
        df = df.with_columns(
            cum_excess=pl.col("excess_ret_1d").cum_sum().over("ticker")
        )
        df = _top_k_equal(df, "cum_excess", self.top_k)
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class CrossSectionalReversal:
    """35. Buy yesterday's biggest losers (bottom decile), hold 1-5 days."""
    bottom_k: int = 8
    hold_days: int = 5
    weight_per: float = 0.10
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="cross_sectional_reversal", description="Yesterday's losers, 5d hold"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        df = features.select(["date", "ticker", "ret_1d"]).drop_nulls("ret_1d")
        df = df.sort(["ticker", "date"])
        df = df.with_columns(
            prev_ret=pl.col("ret_1d").shift(1).over("ticker")
        ).drop_nulls("prev_ret")
        df = df.with_columns(
            rk=pl.col("prev_ret").rank(method="ordinal", descending=False).over("date")
        )
        entry = (pl.col("rk") <= self.bottom_k).cast(pl.Int8)
        df = df.with_columns(entry=entry).with_columns(
            in_trade=pl.col("entry").rolling_sum(window_size=self.hold_days, min_periods=1)
                        .over("ticker").clip(0, 1)
        )
        return df.select("date", "ticker",
                         (pl.col("in_trade").cast(pl.Float64) * self.weight_per).alias("weight"))


@dataclass
class VarianceRatioPairs:
    """36. Select stocks whose price series shows low variance ratio (closer to mean-reverting)
    and apply a mean-reversion entry signal."""
    vr_window: int = 20
    hold_days: int = 5
    weight_per: float = 0.07
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="variance_ratio_pairs", description="Low variance-ratio mean reversion"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        df = features.sort(["ticker", "date"])
        # Estimate variance ratio: Var(2-period returns) / (2 * Var(1-period returns))
        df = df.with_columns(
            ret2=(pl.col("adj_close") / pl.col("adj_close").shift(2).over("ticker") - 1),
        ).with_columns(
            var1=pl.col("ret_1d").rolling_var(window_size=self.vr_window).over("ticker"),
            var2=pl.col("ret2").rolling_var(window_size=self.vr_window).over("ticker"),
        ).with_columns(
            vr=(pl.col("var2") / (2 * pl.col("var1") + 1e-9))
        )
        # vr < 1 → mean-reverting tendency; buy on down days with low vr
        entry = (
            (pl.col("vr") < 0.9) & (pl.col("ret_1d") < -0.01)
        ).cast(pl.Int8)
        df = df.with_columns(entry=entry).with_columns(
            in_trade=pl.col("entry").rolling_sum(window_size=self.hold_days, min_periods=1)
                        .over("ticker").clip(0, 1)
        )
        return df.select("date", "ticker",
                         (pl.col("in_trade").cast(pl.Float64) * self.weight_per).alias("weight"))


@dataclass
class MomentumAnomalyFilter:
    """37. Remove the prior-month return (short-term reversal) from the 12m signal."""
    top_k: int = 8
    rebalance_days: int = 21
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="momentum_anomaly_filter", description="12-1m momentum minus 1m reversal"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        needed = {"mom_12m1m", "mom_20d"}
        if not needed.issubset(features.columns):
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = features.select(["date", "ticker", "mom_12m1m", "mom_20d"]).drop_nulls()
        # Penalize recent strong 1m momentum (short-term reversal risk)
        df = df.with_columns(
            adj_signal=pl.col("mom_12m1m") - 0.5 * pl.col("mom_20d")
        )
        df = _top_k_equal(df, "adj_signal", self.top_k)
        return _rebalance_forward_fill(df, features, self.rebalance_days)


# ─────────────────────────────────────────────────────────────────────────────
# ── NOVEL COMPOSITE STRATEGIES ───────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MomentumWithVolumeSurge:
    """38. Novel: Require both price momentum AND volume confirmation for entry."""
    mom_threshold: float = 0.05
    vol_mult: float = 1.3
    hold_days: int = 10
    weight_per: float = 0.08
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="momentum_volume_surge", description="Price momentum confirmed by volume surge"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        df = features.sort(["ticker", "date"])
        entry = (
            (pl.col("mom_20d") > self.mom_threshold) &
            (pl.col("rel_vol_20") > self.vol_mult)
        ).cast(pl.Int8)
        df = df.with_columns(entry=entry).with_columns(
            in_trade=pl.col("entry").rolling_sum(window_size=self.hold_days, min_periods=1)
                        .over("ticker").clip(0, 1)
        )
        return df.select("date", "ticker",
                         (pl.col("in_trade").cast(pl.Float64) * self.weight_per).alias("weight"))


@dataclass
class AccelerationMomentum:
    """39. Novel: long stocks where 5d momentum is accelerating vs 20d momentum."""
    top_k: int = 8
    rebalance_days: int = 5
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="acceleration_momentum", description="Momentum acceleration (5d > 20d normalized)"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        needed = {"mom_5d", "mom_20d"}
        if not needed.issubset(features.columns):
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = features.select(["date", "ticker", "mom_5d", "mom_20d"]).drop_nulls()
        df = df.with_columns(
            accel=pl.col("mom_5d") - pl.col("mom_20d") / 4.0   # annualize to same horizon
        )
        df = _top_k_equal(df, "accel", self.top_k)
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class AdaptiveRegimeBlend:
    """40. Novel: Continuously blend trend + reversion signals, ratio = f(vol regime).
    High vol → increase reversion weight; low vol → increase trend weight."""
    top_k: int = 6
    rebalance_days: int = 5
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="adaptive_regime_blend", description="Adaptive trend/reversion blend by vol regime"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        df = features.select(["date", "ticker", "mom_20d", "rsi_14",
                               "high_vol_regime"]).drop_nulls()
        if df.is_empty():
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))

        # High-vol regime: reversion dominant (rsi weight 0.7)
        # Low-vol regime: trend dominant (mom weight 0.7)
        df = df.with_columns(
            mom_z=((pl.col("mom_20d") - pl.col("mom_20d").mean().over("date")) /
                   (pl.col("mom_20d").std().over("date") + 1e-8)),
            rsi_z=(-(pl.col("rsi_14") - pl.col("rsi_14").mean().over("date")) /
                   (pl.col("rsi_14").std().over("date") + 1e-8)),   # invert: low RSI = good
        )
        blend_mom = pl.when(pl.col("high_vol_regime") == 1).then(0.3).otherwise(0.7)
        blend_rsi = pl.when(pl.col("high_vol_regime") == 1).then(0.7).otherwise(0.3)
        df = df.with_columns(
            composite=blend_mom * pl.col("mom_z") + blend_rsi * pl.col("rsi_z")
        )
        df = _top_k_equal(df, "composite", self.top_k)
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class FractalMomentumStrategy:
    """41. Novel: multi-timeframe momentum consensus — all 3 horizons must agree."""
    top_k: int = 6
    rebalance_days: int = 10
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="fractal_momentum", description="Multi-timeframe momentum alignment (5/20/60d)"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        needed = {"mom_5d", "mom_20d", "mom_60d"}
        if not needed.issubset(features.columns):
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = features.select(["date", "ticker", "mom_5d", "mom_20d", "mom_60d"]).drop_nulls()
        df = df.filter(
            (pl.col("mom_5d") > 0) & (pl.col("mom_20d") > 0) & (pl.col("mom_60d") > 0)
        )
        if df.is_empty():
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = df.with_columns(
            strength=pl.col("mom_5d") + pl.col("mom_20d") + pl.col("mom_60d")
        )
        df = _top_k_equal(df, "strength", self.top_k)
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class VolatilityNormalisedMomentum:
    """42. Novel: Sharpe-like signal = momentum / realized_vol (risk-adjusted momentum)."""
    top_k: int = 8
    rebalance_days: int = 21
    vol_floor: float = 0.05
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="vol_normalised_momentum", description="Sharpe-ratio-like momentum score"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        df = features.select(["date", "ticker", "mom_60d", "vol_20d"]).drop_nulls()
        df = df.with_columns(
            sharpe_proxy=pl.col("mom_60d") / (pl.col("vol_20d").clip(self.vol_floor, None))
        )
        df = _top_k_equal(df, "sharpe_proxy", self.top_k)
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class EventDrivenMomentumConfluence:
    """43. Novel: Combine event sentiment with price momentum for high-conviction entries."""
    sentiment_threshold: float = 0.2
    mom_threshold: float = 0.02
    hold_days: int = 5
    weight_per: float = 0.09
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="event_momentum_confluence", description="Event sentiment + price momentum confluence"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        df = features.sort(["ticker", "date"])
        if "event_sentiment_mean" in features.columns:
            entry = (
                (pl.col("event_sentiment_mean") > self.sentiment_threshold) &
                (pl.col("mom_20d") > self.mom_threshold)
            ).cast(pl.Int8)
        else:
            # Fallback: volume surge + momentum
            entry = (
                (pl.col("rel_vol_20") > 1.5) &
                (pl.col("mom_20d") > self.mom_threshold)
            ).cast(pl.Int8)
        df = df.with_columns(entry=entry).with_columns(
            in_trade=pl.col("entry").rolling_sum(window_size=self.hold_days, min_periods=1)
                        .over("ticker").clip(0, 1)
        )
        return df.select("date", "ticker",
                         (pl.col("in_trade").cast(pl.Float64) * self.weight_per).alias("weight"))


@dataclass
class ContraMomentumHighVolatility:
    """44. Novel: Contrarian during high-vol; buy the recent laggards when market recovers."""
    bottom_k: int = 6
    rebalance_days: int = 5
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="contra_momentum_high_vol", description="Buy recent laggards during vol spikes"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        if "high_vol_regime" not in features.columns:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = features.filter(pl.col("high_vol_regime") == 1)
        if df.is_empty():
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = df.select(["date", "ticker", "mom_20d"]).drop_nulls("mom_20d")
        df = df.with_columns(
            rk=pl.col("mom_20d").rank(method="ordinal", descending=False).over("date")
        ).with_columns(
            weight=pl.when(pl.col("rk") <= self.bottom_k).then(1.0 / self.bottom_k).otherwise(0.0)
        ).select(["date", "ticker", "weight"])
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class LiquidityWeightedMomentum:
    """45. Novel: Momentum * liquidity score — more weight to easy-to-trade winners."""
    top_k: int = 8
    rebalance_days: int = 10
    min_dollar_volume: float = 5e6
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="liquidity_weighted_momentum", description="Momentum weighted by dollar volume"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        df = features.select(["date", "ticker", "mom_20d", "avg_dollar_volume_20"]).drop_nulls()
        df = df.filter(pl.col("avg_dollar_volume_20") > self.min_dollar_volume)
        df = df.with_columns(
            liq_score=pl.col("avg_dollar_volume_20").log(),
            mom_score=pl.col("mom_20d"),
        ).with_columns(
            liq_norm=(pl.col("liq_score") - pl.col("liq_score").mean().over("date")) /
                      (pl.col("liq_score").std().over("date") + 1e-8),
            mom_norm=(pl.col("mom_score") - pl.col("mom_score").mean().over("date")) /
                      (pl.col("mom_score").std().over("date") + 1e-8),
        ).with_columns(
            composite=0.5 * pl.col("mom_norm") + 0.5 * pl.col("liq_norm")
        )
        df = _top_k_equal(df, "composite", self.top_k)
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class MultiHoldPeriodBlend:
    """46. Novel: 3 separate hold-period momentum signals blended (5d, 20d, 60d) with equal weight."""
    top_k_each: int = 5
    rebalance_days: int = 5
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="multi_hold_blend", description="Blend of 5d/20d/60d momentum top-k equally"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        frames = []
        for col in ["mom_5d", "mom_20d", "mom_60d"]:
            if col not in features.columns:
                continue
            df = features.select(["date", "ticker", col]).drop_nulls(col)
            df = _top_k_equal(df, col, self.top_k_each)
            frames.append(df.rename({"weight": f"w_{col}"}))
        if not frames:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        base = features.select(["date", "ticker"]).unique()
        for f in frames:
            base = base.join(f, on=["date", "ticker"], how="left")
        w_cols = [c for c in base.columns if c.startswith("w_")]
        base = base.with_columns(
            weight=(sum(pl.col(c).fill_null(0.0) for c in w_cols) / len(w_cols))
        ).select(["date", "ticker", "weight"])
        return _rebalance_forward_fill(base, features, self.rebalance_days)
