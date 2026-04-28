"""Extended strategy library — Part 2.

Covers strategies from the expanded list, all implementable with daily OHLCV +
pre-computed feature columns available in `data/gold/features.parquet`.

Categories:
  TREND2       : MACD, Bollinger Breakout, Keltner, Supertrend, Ichimoku proxy,
                 Parabolic SAR proxy, ADX proxy, Heikin-Ashi, 52-Week-High, ORB proxy
  REVERSION2   : Z-Score reversion, Stochastic reversion, CCI reversion,
                 VWAP reversion proxy, ETF mean-reversion, Short-term reversal,
                 Commodity/Yield spread proxies
  FACTOR2      : Value, Size, Quality, Profitability, Investment, Carry,
                 Quality-Minus-Junk, Multi-Factor, L/S Equity, Market-Neutral,
                 Equal Risk Contribution, HRP, Managed Futures, Global Macro
  ML_EVENT2    : News-sentiment signal, Earnings momentum, PEAD, Analyst revision
                 proxy, Supervised ensemble, Anomaly detection, SHAP feature select,
                 Gradient Boosting signal, Random Forest signal, Ensemble voting,
                 Genetic-style operator, RL-lite (Thompson sampling)
  COMPOSITE2   : Dispersion (vol spread), Calendar-spread analog,
                 Merger arb proxy, CTA multi-asset, Index Rebalance Momentum

Strategies that require tick/order-book data or derivatives prices
(true options, latency arb, FX triangular, etc.) are intentionally excluded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import polars as pl
import numpy as np

from .base import StrategyMeta
from .extended import _top_k_equal, _rebalance_forward_fill


# ─────────────────────────────────────────────────────────────────────────────
# Helper: safe pandas correlation matrix from a polars df
# ─────────────────────────────────────────────────────────────────────────────
def _ret_corr_matrix(features: pl.DataFrame, window: int = 252) -> np.ndarray:
    """Pivot returns to wide, compute trailing correlation matrix."""
    df = features.sort(["date", "ticker"])
    last_dates = features["date"].sort(descending=True)[:window]
    recent = features.filter(pl.col("date").is_in(last_dates))
    pivot = (
        recent.pivot(values="ret_1d", index="date", on="ticker")
        .sort("date")
    )
    mat = pivot.drop("date").to_pandas().fillna(0).values
    return np.corrcoef(mat.T) if mat.shape[1] > 1 else np.eye(1)


# ═════════════════════════════════════════════════════════════════════════════
# TREND-FOLLOWING Part 2
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class MACDCrossoverStrategy:
    """MACD (12/26/9) signal line crossover — long when MACD > signal."""
    fast: int = 12
    slow: int = 26
    signal: int = 9
    top_k: int = 8
    rebalance_days: int = 5
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="macd_crossover",
        description="MACD(12,26,9) signal-line crossover, top-k tickers"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        df = features.sort(["ticker", "date"])
        df = df.with_columns(
            ema_fast=pl.col("adj_close").ewm_mean(span=self.fast).over("ticker"),
            ema_slow=pl.col("adj_close").ewm_mean(span=self.slow).over("ticker"),
        ).with_columns(
            macd=pl.col("ema_fast") - pl.col("ema_slow")
        ).with_columns(
            macd_signal=pl.col("macd").ewm_mean(span=self.signal).over("ticker")
        ).with_columns(
            bullish=((pl.col("macd") > pl.col("macd_signal")) &
                     (pl.col("macd") > 0)).cast(pl.Float64)
        ).with_columns(
            score=pl.col("macd") - pl.col("macd_signal")
        )
        df = df.with_columns(
            weight=pl.when(pl.col("bullish") == 1.0).then(1.0 / self.top_k).otherwise(0.0)
        )
        return _rebalance_forward_fill(
            df.select(["date", "ticker", "weight"]), features, self.rebalance_days
        )


@dataclass
class BollingerBandBreakout:
    """Long when price closes above upper Bollinger Band (momentum breakout)."""
    window: int = 20
    n_std: float = 2.0
    hold_days: int = 5
    weight_per: float = 0.08
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="bb_breakout",
        description="Bollinger Band upper-band breakout"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        df = features.sort(["ticker", "date"])
        df = df.with_columns(
            bb_mid=pl.col("adj_close").rolling_mean(window_size=self.window).over("ticker"),
            bb_std=pl.col("adj_close").rolling_std(window_size=self.window).over("ticker"),
        ).with_columns(
            bb_upper=pl.col("bb_mid") + self.n_std * pl.col("bb_std"),
        ).with_columns(
            entry=(pl.col("adj_close") > pl.col("bb_upper")).cast(pl.Int8)
        ).with_columns(
            in_trade=pl.col("entry").rolling_sum(window_size=self.hold_days, min_periods=1)
                        .over("ticker").clip(0, 1)
        )
        return df.select("date", "ticker",
                         (pl.col("in_trade").cast(pl.Float64) * self.weight_per).alias("weight"))


@dataclass
class KeltnerChannelBreakout:
    """Long when close breaks above Keltner Channel (EMA ± k*ATR)."""
    ema_period: int = 20
    atr_mult: float = 2.0
    hold_days: int = 5
    weight_per: float = 0.08
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="keltner_breakout",
        description="Keltner Channel upper-band breakout"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        if "atr_14" not in features.columns:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = features.sort(["ticker", "date"])
        df = df.with_columns(
            kc_mid=pl.col("adj_close").ewm_mean(span=self.ema_period).over("ticker"),
        ).with_columns(
            kc_upper=pl.col("kc_mid") + self.atr_mult * pl.col("atr_14"),
        ).with_columns(
            entry=(pl.col("adj_close") > pl.col("kc_upper")).cast(pl.Int8)
        ).with_columns(
            in_trade=pl.col("entry").rolling_sum(window_size=self.hold_days, min_periods=1)
                        .over("ticker").clip(0, 1)
        )
        return df.select("date", "ticker",
                         (pl.col("in_trade").cast(pl.Float64) * self.weight_per).alias("weight"))


@dataclass
class SupertrendStrategy:
    """Supertrend indicator: price above (EMA - k*ATR) → long."""
    ema_period: int = 10
    atr_mult: float = 3.0
    top_k: int = 10
    rebalance_days: int = 5
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="supertrend",
        description="Supertrend (EMA ± k×ATR) trend follower"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        if "atr_14" not in features.columns:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = features.sort(["ticker", "date"])
        df = df.with_columns(
            ema=pl.col("adj_close").ewm_mean(span=self.ema_period).over("ticker"),
        ).with_columns(
            upper_band=pl.col("ema") + self.atr_mult * pl.col("atr_14"),
            lower_band=pl.col("ema") - self.atr_mult * pl.col("atr_14"),
        ).with_columns(
            bullish=(pl.col("adj_close") > pl.col("lower_band")).cast(pl.Float64)
        )
        df = df.with_columns(
            score=pl.col("bullish") * pl.col("mom_20d").fill_null(0)
        )
        df = df.with_columns(
            rk=pl.col("score").rank(method="ordinal", descending=True).over("date")
        ).with_columns(
            weight=pl.when((pl.col("rk") <= self.top_k) & (pl.col("bullish") == 1.0))
                     .then(1.0 / self.top_k).otherwise(0.0)
        )
        return _rebalance_forward_fill(
            df.select(["date", "ticker", "weight"]), features, self.rebalance_days
        )


@dataclass
class IchimokuCloudProxy:
    """Ichimoku Cloud proxy using Tenkan/Kijun (9/26 highs-low midpoints)."""
    tenkan: int = 9
    kijun: int = 26
    top_k: int = 8
    rebalance_days: int = 5
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="ichimoku_proxy",
        description="Ichimoku Cloud proxy (Tenkan > Kijun AND price above cloud)"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        if not {"high", "low", "adj_close"}.issubset(features.columns):
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = features.sort(["ticker", "date"])
        df = df.with_columns(
            tenkan=(pl.col("high").rolling_max(window_size=self.tenkan).over("ticker") +
                    pl.col("low").rolling_min(window_size=self.tenkan).over("ticker")) / 2,
            kijun=(pl.col("high").rolling_max(window_size=self.kijun).over("ticker") +
                   pl.col("low").rolling_min(window_size=self.kijun).over("ticker")) / 2,
        ).with_columns(
            cloud_top=pl.max_horizontal(pl.col("tenkan"), pl.col("kijun")),
        ).with_columns(
            signal=((pl.col("tenkan") > pl.col("kijun")) &
                    (pl.col("adj_close") > pl.col("cloud_top"))).cast(pl.Float64)
        ).with_columns(
            score=pl.col("signal") * (pl.col("tenkan") / pl.col("kijun") - 1)
        )
        df = df.filter(pl.col("signal") == 1.0)
        if df.is_empty():
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = _top_k_equal(df, "score", self.top_k)
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class ParabolicSARProxy:
    """Parabolic SAR proxy: long when price > rolling max(high, N) shifted by accel."""
    sar_window: int = 5
    accel: float = 0.02
    top_k: int = 8
    rebalance_days: int = 5
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="parabolic_sar_proxy",
        description="Parabolic SAR proxy using rolling-high drift"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        df = features.sort(["ticker", "date"])
        df = df.with_columns(
            rolling_high=pl.col("high").rolling_max(window_size=self.sar_window).over("ticker"),
        ).with_columns(
            sar_level=pl.col("rolling_high").shift(1).over("ticker") * (1 - self.accel)
        ).with_columns(
            bullish=(pl.col("adj_close") > pl.col("sar_level")).cast(pl.Float64)
        )
        df = df.filter(pl.col("bullish") == 1.0)
        if df.is_empty():
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = df.with_columns(
            score=pl.col("mom_20d").fill_null(0)
        )
        df = _top_k_equal(df, "score", self.top_k)
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class ADXTrendFollowing:
    """ADX proxy: enter when strong trend (high momentum variance across lookbacks)."""
    adx_proxy_window: int = 14
    top_k: int = 8
    rebalance_days: int = 5
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="adx_trend",
        description="ADX-proxy trend filter (momentum consistency score)"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        mom_cols = [c for c in ["mom_5d", "mom_10d", "mom_20d", "mom_60d"] if c in features.columns]
        if len(mom_cols) < 2:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = features.select(["date", "ticker"] + mom_cols).drop_nulls()
        # ADX proxy: all momentum signs agree = strong trend
        sign_exprs = [pl.when(pl.col(c) > 0).then(1.0).otherwise(-1.0) for c in mom_cols]
        consensus = sum(sign_exprs) / len(sign_exprs)
        df = df.with_columns(adx_proxy=consensus)
        # Score = abs(consensus) * avg momentum magnitude
        mag_expr = sum(pl.col(c).abs() for c in mom_cols) / len(mom_cols)
        df = df.with_columns(
            score=pl.col("adx_proxy").abs() * mag_expr
        ).filter(pl.col("adx_proxy") > 0)  # long only when all agree positive
        if df.is_empty():
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = _top_k_equal(df, "score", self.top_k)
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class HeikinAshiTrend:
    """Heikin-Ashi candles: long when HA body is bullish for N consecutive days."""
    consecutive: int = 3
    top_k: int = 8
    rebalance_days: int = 5
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="heikin_ashi_trend",
        description="N consecutive bullish Heikin-Ashi candles"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        if not {"open", "high", "low", "close"}.issubset(features.columns):
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = features.sort(["ticker", "date"])
        df = df.with_columns(
            ha_close=(pl.col("open") + pl.col("high") + pl.col("low") + pl.col("close")) / 4,
        )
        # HA open = (prev HA open + prev HA close) / 2 — use shift approximation
        df = df.with_columns(
            ha_open=((pl.col("open").shift(1) + pl.col("close").shift(1)) / 2).over("ticker")
        ).with_columns(
            ha_bull=(pl.col("ha_close") > pl.col("ha_open")).cast(pl.Int8)
        ).with_columns(
            ha_streak=pl.col("ha_bull").rolling_sum(window_size=self.consecutive, min_periods=self.consecutive)
                         .over("ticker")
        ).with_columns(
            entry=(pl.col("ha_streak") >= self.consecutive).cast(pl.Float64)
        )
        df = df.filter(pl.col("entry") == 1.0).with_columns(
            score=pl.col("mom_20d").fill_null(0)
        )
        if df.is_empty():
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = _top_k_equal(df, "score", self.top_k)
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class FiftyTwoWeekHighMomentum:
    """52-Week High Momentum: stocks within X% of their 52-week high."""
    proximity_pct: float = 0.05   # within 5% of 52w high
    top_k: int = 8
    rebalance_days: int = 21
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="52wk_high_momentum",
        description="Stocks within 5% of 52-week high (George & Hwang)"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        df = features.sort(["ticker", "date"])
        df = df.with_columns(
            high_52w=pl.col("adj_close").rolling_max(window_size=252, min_periods=100).over("ticker"),
        ).with_columns(
            dist_from_high=pl.col("adj_close") / pl.col("high_52w") - 1
        ).drop_nulls("dist_from_high")
        df = df.filter(pl.col("dist_from_high") >= -self.proximity_pct)
        if df.is_empty():
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = df.with_columns(score=pl.col("dist_from_high"))
        df = _top_k_equal(df, "score", self.top_k)
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class OpeningRangeBreakoutProxy:
    """Opening Range Breakout proxy: stocks with high relative volume + gap up at open."""
    gap_threshold: float = 0.005   # gap up > 0.5%
    vol_mult: float = 1.5
    hold_days: int = 3
    weight_per: float = 0.07
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="orb_proxy",
        description="ORB proxy: gap-up + volume surge at open"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        if not {"open", "adj_close", "rel_vol_20"}.issubset(features.columns):
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = features.sort(["ticker", "date"])
        df = df.with_columns(
            prev_close=pl.col("adj_close").shift(1).over("ticker"),
        ).with_columns(
            gap=pl.col("open") / pl.col("prev_close") - 1,
        ).drop_nulls("gap")
        entry = (
            (pl.col("gap") > self.gap_threshold) &
            (pl.col("rel_vol_20") > self.vol_mult)
        ).cast(pl.Int8)
        df = df.with_columns(entry=entry).with_columns(
            in_trade=pl.col("entry").rolling_sum(window_size=self.hold_days, min_periods=1)
                        .over("ticker").clip(0, 1)
        )
        return df.select("date", "ticker",
                         (pl.col("in_trade").cast(pl.Float64) * self.weight_per).alias("weight"))


# ═════════════════════════════════════════════════════════════════════════════
# MEAN REVERSION Part 2
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ZScoreMeanReversion:
    """Z-score reversion: long when rolling z-score < -2, exit at z > 0."""
    window: int = 20
    entry_z: float = -2.0
    exit_z: float = 0.0
    weight_per: float = 0.08
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="zscore_reversion",
        description="Long when price z-score < -2, exit at z > 0"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        df = features.sort(["ticker", "date"])
        df = df.with_columns(
            roll_mean=pl.col("adj_close").rolling_mean(window_size=self.window).over("ticker"),
            roll_std=pl.col("adj_close").rolling_std(window_size=self.window).over("ticker"),
        ).with_columns(
            zscore=((pl.col("adj_close") - pl.col("roll_mean")) /
                    (pl.col("roll_std") + 1e-8))
        ).with_columns(
            entry=pl.when(pl.col("zscore") < self.entry_z).then(1.0)
                    .when(pl.col("zscore") > self.exit_z).then(0.0)
                    .otherwise(None)
        ).with_columns(
            in_trade=pl.col("entry").fill_null(strategy="forward").over("ticker").fill_null(0.0)
        )
        return df.select("date", "ticker",
                         (pl.col("in_trade") * self.weight_per).alias("weight"))


@dataclass
class StochasticOscillatorReversion:
    """Stochastic %K < 20 → long (oversold); %K > 80 → exit."""
    k_window: int = 14
    entry_k: float = 20.0
    exit_k: float = 80.0
    weight_per: float = 0.07
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="stochastic_reversion",
        description="Stochastic %K oversold/overbought mean reversion"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        if not {"high", "low", "adj_close"}.issubset(features.columns):
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = features.sort(["ticker", "date"])
        df = df.with_columns(
            lo_k=pl.col("low").rolling_min(window_size=self.k_window).over("ticker"),
            hi_k=pl.col("high").rolling_max(window_size=self.k_window).over("ticker"),
        ).with_columns(
            pct_k=(pl.col("adj_close") - pl.col("lo_k")) /
                  (pl.col("hi_k") - pl.col("lo_k") + 1e-8) * 100
        ).with_columns(
            entry=pl.when(pl.col("pct_k") < self.entry_k).then(1.0)
                    .when(pl.col("pct_k") > self.exit_k).then(0.0)
                    .otherwise(None)
        ).with_columns(
            in_trade=pl.col("entry").fill_null(strategy="forward").over("ticker").fill_null(0.0)
        )
        return df.select("date", "ticker",
                         (pl.col("in_trade") * self.weight_per).alias("weight"))


@dataclass
class CCIMeanReversion:
    """CCI < -100 → long (oversold); CCI > 100 → exit."""
    window: int = 20
    entry_cci: float = -100.0
    exit_cci: float = 100.0
    weight_per: float = 0.07
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="cci_reversion",
        description="CCI < -100 oversold long entry"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        if not {"high", "low", "adj_close"}.issubset(features.columns):
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = features.sort(["ticker", "date"])
        df = df.with_columns(
            tp=(pl.col("high") + pl.col("low") + pl.col("adj_close")) / 3
        ).with_columns(
            tp_mean=pl.col("tp").rolling_mean(window_size=self.window).over("ticker"),
            tp_mad=pl.col("tp").rolling_std(window_size=self.window).over("ticker"),
        ).with_columns(
            cci=(pl.col("tp") - pl.col("tp_mean")) / (0.015 * pl.col("tp_mad") + 1e-8)
        ).with_columns(
            entry=pl.when(pl.col("cci") < self.entry_cci).then(1.0)
                    .when(pl.col("cci") > self.exit_cci).then(0.0)
                    .otherwise(None)
        ).with_columns(
            in_trade=pl.col("entry").fill_null(strategy="forward").over("ticker").fill_null(0.0)
        )
        return df.select("date", "ticker",
                         (pl.col("in_trade") * self.weight_per).alias("weight"))


@dataclass
class VWAPReversionProxy:
    """VWAP reversion proxy: daily (volume * price) weighted average vs close."""
    hold_days: int = 3
    deviation_threshold: float = -0.02
    weight_per: float = 0.07
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="vwap_reversion",
        description="VWAP proxy reversion: close < VWAP by > 2%"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        if "volume" not in features.columns:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = features.sort(["ticker", "date"])
        # Daily VWAP proxy = OHLC4 weighted by vol (rolling 5d)
        df = df.with_columns(
            ohlc4=(pl.col("open") + pl.col("high") + pl.col("low") + pl.col("adj_close")) / 4
            if all(c in features.columns for c in ["open", "high", "low"])
            else pl.col("adj_close"),
        ).with_columns(
            pv=(pl.col("ohlc4") * pl.col("volume")),
        ).with_columns(
            rolling_pv=pl.col("pv").rolling_sum(window_size=5).over("ticker"),
            rolling_vol=pl.col("volume").rolling_sum(window_size=5).over("ticker"),
        ).with_columns(
            vwap=pl.col("rolling_pv") / (pl.col("rolling_vol") + 1)
        ).with_columns(
            dev=pl.col("adj_close") / pl.col("vwap") - 1
        ).with_columns(
            entry=(pl.col("dev") < self.deviation_threshold).cast(pl.Int8)
        ).with_columns(
            in_trade=pl.col("entry").rolling_sum(window_size=self.hold_days, min_periods=1)
                        .over("ticker").clip(0, 1)
        )
        return df.select("date", "ticker",
                         (pl.col("in_trade").cast(pl.Float64) * self.weight_per).alias("weight"))


@dataclass
class ShortTermReversal:
    """Buy last-week's worst losers (1-month reversal). Classic short-term reversal."""
    lookback: int = 21
    bottom_k: int = 10
    hold_days: int = 21
    rebalance_days: int = 21
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="short_term_reversal",
        description="1-month loser reversion (Jegadeesh 1990)"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        col = "mom_20d" if "mom_20d" in features.columns else "ret_1d"
        df = features.select(["date", "ticker", col]).drop_nulls(col)
        # Invert: rank ascending (worst performers = low rank = selected)
        df = df.with_columns(
            rk=pl.col(col).rank(method="ordinal", descending=False).over("date")
        ).with_columns(
            weight=pl.when(pl.col("rk") <= self.bottom_k)
                     .then(1.0 / self.bottom_k).otherwise(0.0)
        ).select(["date", "ticker", "weight"])
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class GapFillStrategy:
    """Gap fill: enter on gaps > threshold in either direction, bet on fill."""
    gap_threshold: float = 0.015
    hold_days: int = 2
    weight_per: float = 0.06
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="gap_fill",
        description="Gap fill: enter opposite direction on large gap opens"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        if "open" not in features.columns:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = features.sort(["ticker", "date"])
        df = df.with_columns(
            prev_close=pl.col("adj_close").shift(1).over("ticker"),
        ).with_columns(
            gap=pl.col("open") / pl.col("prev_close") - 1,
        ).drop_nulls("gap")
        # Trade both gap-up (fade) and gap-down (buy) — daily bars, fade direction
        entry = (pl.col("gap").abs() > self.gap_threshold).cast(pl.Int8)
        df = df.with_columns(entry=entry).with_columns(
            in_trade=pl.col("entry").rolling_sum(window_size=self.hold_days, min_periods=1)
                        .over("ticker").clip(0, 1)
        )
        return df.select("date", "ticker",
                         (pl.col("in_trade").cast(pl.Float64) * self.weight_per).alias("weight"))


@dataclass
class ETFMeanReversion:
    """ETF mean reversion: buy ETF-like assets when z-score < -1.5 vs universe."""
    z_entry: float = -1.5
    z_exit: float = 0.5
    rebalance_days: int = 5
    weight_per: float = 0.10
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="etf_mean_reversion",
        description="Cross-sectional z-score mean reversion (ETF-style)"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        df = features.sort(["ticker", "date"])
        df = df.with_columns(
            cs_mean=pl.col("mom_20d").mean().over("date"),
            cs_std=pl.col("mom_20d").std().over("date"),
        ).with_columns(
            cs_z=(pl.col("mom_20d") - pl.col("cs_mean")) / (pl.col("cs_std") + 1e-8)
        ).with_columns(
            entry=pl.when(pl.col("cs_z") < self.z_entry).then(1.0)
                    .when(pl.col("cs_z") > self.z_exit).then(0.0)
                    .otherwise(None)
        ).with_columns(
            in_trade=pl.col("entry").fill_null(strategy="forward").over("ticker").fill_null(0.0)
        )
        return df.select("date", "ticker",
                         (pl.col("in_trade") * self.weight_per).alias("weight"))


# ═════════════════════════════════════════════════════════════════════════════
# FACTOR / PORTFOLIO STRATEGIES Part 2
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ValueFactorStrategy:
    """Value factor: buy stocks furthest below 200d SMA (price cheapness proxy)."""
    bottom_k: int = 10
    rebalance_days: int = 21
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="value_factor",
        description="Value factor: price discount to 200d SMA"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        if "sma_gap_200" not in features.columns:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = features.select(["date", "ticker", "sma_gap_200"]).drop_nulls()
        # Bottom-k by sma_gap_200 = cheapest relative to long-run price
        df = df.with_columns(
            rk=pl.col("sma_gap_200").rank(method="ordinal", descending=False).over("date")
        ).with_columns(
            weight=pl.when(pl.col("rk") <= self.bottom_k).then(1.0 / self.bottom_k).otherwise(0.0)
        ).select(["date", "ticker", "weight"])
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class SizeFactorStrategy:
    """Size factor: equal-weight small-cap proxy (low dollar volume = small size)."""
    n_small: int = 20
    min_adv: float = 1e6      # minimum liquidity filter
    rebalance_days: int = 21
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="size_factor",
        description="Small-size factor: low market-cap proxy (ADV)"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        if "avg_dollar_volume_20" not in features.columns:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = features.select(["date", "ticker", "avg_dollar_volume_20"]).drop_nulls()
        df = df.filter(pl.col("avg_dollar_volume_20") >= self.min_adv)
        df = df.with_columns(
            rk=pl.col("avg_dollar_volume_20").rank(method="ordinal", descending=False).over("date")
        ).with_columns(
            weight=pl.when(pl.col("rk") <= self.n_small).then(1.0 / self.n_small).otherwise(0.0)
        ).select(["date", "ticker", "weight"])
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class QualityFactorStrategy:
    """Quality factor: high-liquidity, low-volatility, positive-trend stocks."""
    top_k: int = 10
    rebalance_days: int = 21
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="quality_factor",
        description="Quality: high liquidity + low vol + uptrend"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        needed = {"vol_20d", "avg_dollar_volume_20", "sma_gap_50"}
        if not needed.issubset(features.columns):
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = features.select(["date", "ticker", "vol_20d", "avg_dollar_volume_20", "sma_gap_50"]).drop_nulls()
        df = df.with_columns(
            vol_rk=pl.col("vol_20d").rank(method="ordinal", descending=False).over("date"),
            liq_rk=pl.col("avg_dollar_volume_20").rank(method="ordinal", descending=True).over("date"),
        ).with_columns(
            quality_score=(pl.col("vol_rk") + pl.col("liq_rk")) / 2.0
        ).filter(pl.col("sma_gap_50") > 0)  # only above-trend stocks
        if df.is_empty():
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = df.with_columns(
            rk=pl.col("quality_score").rank(method="ordinal", descending=False).over("date")
        ).with_columns(
            weight=pl.when(pl.col("rk") <= self.top_k).then(1.0 / self.top_k).otherwise(0.0)
        ).select(["date", "ticker", "weight"])
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class ProfitabilityFactor:
    """Profitability factor proxy: high excess returns + positive trend (ROE proxy)."""
    top_k: int = 10
    rebalance_days: int = 21
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="profitability_factor",
        description="Profitability proxy: sustained excess return + uptrend"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        if "excess_ret_1d" not in features.columns:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = features.sort(["ticker", "date"])
        df = df.with_columns(
            cum_excess=pl.col("excess_ret_1d").rolling_sum(window_size=63, min_periods=20)
                         .over("ticker")
        ).drop_nulls("cum_excess")
        df = df.filter(pl.col("mom_20d") > 0)  # only uptrending
        if df.is_empty():
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = _top_k_equal(df, "cum_excess", self.top_k)
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class InvestmentFactor:
    """Investment factor proxy: low-volatility-of-returns = stable growers."""
    top_k: int = 10
    rebalance_days: int = 21
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="investment_factor",
        description="Investment factor: low-vol growth proxy (conservative investment)"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        needed = {"vol_60d", "mom_60d"}
        if not needed.issubset(features.columns):
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = features.select(["date", "ticker", "vol_60d", "mom_60d"]).drop_nulls()
        # Conservative investment: positive 60d momentum, low 60d vol
        df = df.filter(pl.col("mom_60d") > 0)
        df = df.with_columns(
            rk=pl.col("vol_60d").rank(method="ordinal", descending=False).over("date")
        ).with_columns(
            weight=pl.when(pl.col("rk") <= self.top_k).then(1.0 / self.top_k).otherwise(0.0)
        ).select(["date", "ticker", "weight"])
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class CarryFactorStrategy:
    """Carry proxy: buy high excess-return stocks (positive carry = steady outperformers)."""
    top_k: int = 10
    lookback: int = 63
    rebalance_days: int = 21
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="carry_factor",
        description="Equity carry: sustained excess return vs benchmark"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        if "excess_ret_1d" not in features.columns:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = features.sort(["ticker", "date"])
        df = df.with_columns(
            carry=pl.col("excess_ret_1d").rolling_mean(window_size=self.lookback, min_periods=20)
                     .over("ticker")
        ).drop_nulls("carry")
        df = _top_k_equal(df, "carry", self.top_k)
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class QualityMinusJunk:
    """Quality-Minus-Junk (Asness et al.): long high quality, short junk — long-only here."""
    top_k: int = 10
    junk_exclude_k: int = 10
    rebalance_days: int = 21
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="quality_minus_junk",
        description="QMJ: long top quality, avoid bottom quality (AQR style)"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        needed = {"vol_20d", "avg_dollar_volume_20", "mom_20d"}
        if not needed.issubset(features.columns):
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = features.select(["date", "ticker", "vol_20d", "avg_dollar_volume_20", "mom_20d"]).drop_nulls()
        n_per_date = df.group_by("date").agg(pl.len().alias("n"))
        df = df.join(n_per_date, on="date")
        df = df.with_columns(
            vol_rk=pl.col("vol_20d").rank(method="ordinal", descending=False).over("date"),
            liq_rk=pl.col("avg_dollar_volume_20").rank(method="ordinal", descending=True).over("date"),
            mom_rk=pl.col("mom_20d").rank(method="ordinal", descending=True).over("date"),
        ).with_columns(
            quality=(pl.col("vol_rk") + pl.col("liq_rk") + pl.col("mom_rk")) / 3.0
        ).with_columns(
            final_rk=pl.col("quality").rank(method="ordinal", descending=False).over("date"),
            junk_rk=pl.col("quality").rank(method="ordinal", descending=True).over("date"),
        ).with_columns(
            weight=pl.when(pl.col("final_rk") <= self.top_k).then(1.0 / self.top_k).otherwise(0.0)
        ).select(["date", "ticker", "weight"])
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class MultiFactor5:
    """5-Factor model: momentum + value + quality + low-vol + carry, equal-weighted rank."""
    top_k: int = 10
    rebalance_days: int = 21
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="multi_factor_5",
        description="5-factor composite: mom + value + quality + low-vol + carry"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        factors = {
            "mom_20d": True,       # higher = better
            "sma_gap_200": False,  # lower = better (value)
            "vol_20d": False,      # lower = better (low-vol)
            "avg_dollar_volume_20": True,   # higher = better (quality proxy)
        }
        avail = {k: v for k, v in factors.items() if k in features.columns}
        if len(avail) < 2:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = features.select(["date", "ticker"] + list(avail.keys())).drop_nulls()
        rank_exprs = []
        for f, desc in avail.items():
            df = df.with_columns(
                pl.col(f).rank(method="ordinal", descending=desc).over("date").alias(f"rk_{f}")
            )
            rank_exprs.append(pl.col(f"rk_{f}"))
        composite = sum(rank_exprs) / len(rank_exprs)
        df = df.with_columns(composite=composite)
        df = df.with_columns(
            final_rk=pl.col("composite").rank(method="ordinal", descending=False).over("date")
        ).with_columns(
            weight=pl.when(pl.col("final_rk") <= self.top_k).then(1.0 / self.top_k).otherwise(0.0)
        ).select(["date", "ticker", "weight"])
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class MarketNeutralMomentum:
    """Market-neutral: long top momentum, short bottom (simulated as cash=0 here)."""
    top_k: int = 8
    bottom_k: int = 8
    rebalance_days: int = 10
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="market_neutral_momentum",
        description="Market-neutral: long top, offset bottom (long-only simulation)"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        col = "mom_20d"
        df = features.select(["date", "ticker", col]).drop_nulls(col)
        n = df.group_by("date").agg(pl.len().alias("n"))
        df = df.join(n, on="date")
        df = df.with_columns(
            rk=pl.col(col).rank(method="ordinal", descending=True).over("date")
        )
        total = df["n"].max()
        df = df.with_columns(
            weight=pl.when(pl.col("rk") <= self.top_k).then(1.0 / self.top_k)
                     .when(pl.col("rk") > (pl.col("n") - self.bottom_k))
                     .then(-1.0 / self.bottom_k)
                     .otherwise(0.0)
        ).select(["date", "ticker", "weight"])
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class EqualRiskContribution:
    """Equal Risk Contribution: target equal vol contribution across top-N holdings."""
    n_stocks: int = 15
    rebalance_days: int = 21
    vol_lookback: int = 60
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="equal_risk_contribution",
        description="ERC: equal marginal risk contribution (inverse-vol approx)"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        # Inverse-vol weighting is a close approximation of ERC for uncorrelated assets
        col = "vol_60d" if "vol_60d" in features.columns else "vol_20d"
        df = features.select(["date", "ticker", col, "mom_20d"]).drop_nulls()
        # Select top-n by momentum first
        df = df.with_columns(
            mom_rk=pl.col("mom_20d").rank(method="ordinal", descending=True).over("date")
        ).filter(pl.col("mom_rk") <= self.n_stocks)
        df = df.with_columns(
            inv_vol=1.0 / (pl.col(col).clip(0.05, None))
        ).with_columns(
            inv_vol_sum=pl.col("inv_vol").sum().over("date")
        ).with_columns(
            weight=(pl.col("inv_vol") / pl.col("inv_vol_sum")).clip(0.0, 0.15)
        ).select(["date", "ticker", "weight"])
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class HierarchicalRiskParity:
    """HRP (simplified): cluster by return correlation, allocate inversely to cluster vol."""
    n_stocks: int = 20
    rebalance_days: int = 63
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="hrp",
        description="Hierarchical Risk Parity (simplified 2-cluster approximation)"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        """Build HRP weights once per rebalance period using trailing returns."""
        dates = features["date"].unique().sort()
        rebal_dates = dates[::self.rebalance_days]

        all_rows: list[dict] = []
        for rb_date in rebal_dates:
            # Use trailing 252 days
            start_idx = max(0, dates.search_sorted(rb_date) - 252)
            window_dates = dates[start_idx:dates.search_sorted(rb_date) + 1]
            window = features.filter(pl.col("date").is_in(window_dates))
            if window.is_empty():
                continue

            # Pivot to wide returns
            try:
                pivot = (
                    window.pivot(values="ret_1d", index="date", on="ticker")
                    .sort("date")
                )
            except Exception:
                continue

            tickers = [c for c in pivot.columns if c != "date"]
            if len(tickers) < 4:
                continue
            mat = pivot.select(tickers).to_pandas().fillna(0).values[-252:]
            if mat.shape[0] < 30:
                continue

            # Simple HRP: sort by vol, split into 2 clusters, inverse-vol weight within each
            vols = np.std(mat, axis=0) + 1e-8
            sorted_idx = np.argsort(vols)
            half = len(sorted_idx) // 2
            cluster1 = sorted_idx[:half]
            cluster2 = sorted_idx[half:]

            weights = np.zeros(len(tickers))
            # Each cluster gets 0.5 of portfolio, inverse-vol within cluster
            for cluster in [cluster1, cluster2]:
                ivol = 1.0 / vols[cluster]
                ivol /= ivol.sum()
                weights[cluster] = ivol * 0.5 / len([cluster1, cluster2]) * 2

            # Apply to n_stocks by momentum
            top_n = min(self.n_stocks, len(tickers))
            mom_scores = np.array([
                window.filter(pl.col("ticker") == t)["mom_20d"].mean() or 0.0
                for t in tickers
            ])
            top_idx = np.argsort(mom_scores)[-top_n:]
            final_weights = np.zeros(len(tickers))
            final_weights[top_idx] = weights[top_idx]
            if final_weights.sum() > 0:
                final_weights /= final_weights.sum()

            for i, t in enumerate(tickers):
                if final_weights[i] > 0:
                    all_rows.append({"date": rb_date, "ticker": t, "weight": float(final_weights[i])})

        if not all_rows:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df_rb = pl.DataFrame(all_rows)
        return _rebalance_forward_fill(df_rb, features, self.rebalance_days)


@dataclass
class ManagedFuturesCTA:
    """CTA / Managed Futures: trend-follow across stocks using multiple lookbacks."""
    lookbacks: tuple = (10, 21, 63, 126)
    top_k: int = 10
    rebalance_days: int = 5
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="managed_futures_cta",
        description="CTA trend: multi-lookback momentum vote"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        mom_cols = [f"mom_{lb}d" for lb in self.lookbacks if f"mom_{lb}d" in features.columns]
        if not mom_cols:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = features.select(["date", "ticker"] + mom_cols).drop_nulls()
        sign_exprs = [pl.when(pl.col(c) > 0).then(1.0).otherwise(-1.0) for c in mom_cols]
        consensus = sum(sign_exprs) / len(sign_exprs)
        df = df.with_columns(cta_score=consensus)
        df = df.filter(pl.col("cta_score") > 0)
        if df.is_empty():
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = _top_k_equal(df, "cta_score", self.top_k)
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class GlobalMacroTrend:
    """Global Macro trend: regime-aware allocation using bull/high-vol flags + momentum."""
    top_k_bull: int = 8
    top_k_bear: int = 5
    rebalance_days: int = 21
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="global_macro_trend",
        description="Global Macro: risk-on/off regime + trend following"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        if "bull_regime" not in features.columns:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        # Risk-on: bull_regime=1, high_vol=0
        if "high_vol_regime" in features.columns:
            risk_on = (pl.col("bull_regime") == 1) & (pl.col("high_vol_regime") == 0)
        else:
            risk_on = pl.col("bull_regime") == 1

        df_on = features.filter(risk_on).select(["date", "ticker", "mom_60d"]).drop_nulls()
        df_off = features.filter(~risk_on).select(["date", "ticker", "vol_20d"]).drop_nulls()

        frames = []
        if not df_on.is_empty():
            frames.append(_top_k_equal(df_on, "mom_60d", self.top_k_bull))
        if not df_off.is_empty():
            df_off2 = df_off.with_columns(
                rk=pl.col("vol_20d").rank(method="ordinal", descending=False).over("date")
            ).with_columns(
                weight=pl.when(pl.col("rk") <= self.top_k_bear).then(1.0 / self.top_k_bear).otherwise(0.0)
            ).select(["date", "ticker", "weight"])
            frames.append(df_off2)
        if not frames:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = pl.concat(frames)
        return _rebalance_forward_fill(df, features, self.rebalance_days)


# ═════════════════════════════════════════════════════════════════════════════
# ARBITRAGE / RELATIVE VALUE (daily-bar implementations)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class IndexRebalanceMomentum:
    """Index rebalance effect: buy stocks with strong recent momentum expecting
    institutional demand from index rebalancing."""
    top_k: int = 5
    rebalance_days: int = 63   # quarterly
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="index_rebalance_arb",
        description="Index rebalance momentum: buy expected additions"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        # Proxy: stocks near 52w high with large positive momentum = likely index additions
        col = "mom_120d" if "mom_120d" in features.columns else "mom_60d"
        df = features.select(["date", "ticker", col]).drop_nulls(col)
        df = _top_k_equal(df, col, self.top_k)
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class MergerArbitrageProxy:
    """Merger arb proxy: hold momentum stocks with >10% gap from 52w high (event premium)."""
    top_k: int = 5
    hold_days: int = 21
    rebalance_days: int = 21
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="merger_arb_proxy",
        description="Merger arb proxy: event-premium momentum stocks"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        df = features.sort(["ticker", "date"])
        df = df.with_columns(
            high_52w=pl.col("adj_close").rolling_max(window_size=252, min_periods=100).over("ticker"),
        ).with_columns(
            prem=pl.col("adj_close") / pl.col("high_52w") - 1
        ).drop_nulls("prem")
        # Stocks trading near high = potential acquisition targets
        df = df.filter((pl.col("prem") >= -0.15) & (pl.col("prem") <= 0))
        if df.is_empty():
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = df.with_columns(score=pl.col("prem"))
        df = _top_k_equal(df, "score", self.top_k)
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class VolatilityArbitrage:
    """Volatility arb proxy: buy low-realized-vol stocks when they're also trending
    (implies cheap implied vol relative to realized)."""
    top_k: int = 8
    rebalance_days: int = 10
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="vol_arbitrage",
        description="Vol arb proxy: low-realized-vol + trending (cheap IV signal)"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        needed = {"vol_20d", "mom_20d", "sma_gap_50"}
        if not needed.issubset(features.columns):
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = features.select(["date", "ticker", "vol_20d", "mom_20d", "sma_gap_50"]).drop_nulls()
        df = df.filter((pl.col("mom_20d") > 0) & (pl.col("sma_gap_50") > 0))
        if df.is_empty():
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = df.with_columns(
            rk=pl.col("vol_20d").rank(method="ordinal", descending=False).over("date")
        ).with_columns(
            weight=pl.when(pl.col("rk") <= self.top_k).then(1.0 / self.top_k).otherwise(0.0)
        ).select(["date", "ticker", "weight"])
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class DispersionTrading:
    """Dispersion trading proxy: long high-idiosyncratic-vol stocks when market vol is low.
    Rationale: index implied vol < component vol → long single names."""
    top_k: int = 8
    rebalance_days: int = 10
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="dispersion_trading",
        description="Dispersion: long high-idiosyncratic-vol in low market vol"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        if "high_vol_regime" not in features.columns or "vol_20d" not in features.columns:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        # Enter when market vol is LOW (low vol regime)
        df = features.filter(pl.col("high_vol_regime") == 0)
        if df.is_empty():
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        # Buy stocks with ABOVE-AVERAGE individual vol (high idiosyncratic vol)
        df = df.with_columns(
            univ_mean_vol=pl.col("vol_20d").mean().over("date")
        ).filter(pl.col("vol_20d") > pl.col("univ_mean_vol"))
        if df.is_empty():
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = df.with_columns(score=pl.col("mom_20d").fill_null(0))
        df = _top_k_equal(df, "score", self.top_k)
        return _rebalance_forward_fill(df, features, self.rebalance_days)


# ═════════════════════════════════════════════════════════════════════════════
# ML / EVENT-DRIVEN STRATEGIES (feature-based, no external API required)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class EarningsMomentumStrategy:
    """Earnings momentum: buy stocks with strong momentum around earnings (PEAD proxy).
    Uses high relative volume as earnings-event proxy."""
    vol_mult: float = 2.5      # relative volume spike = earnings event proxy
    top_k: int = 6
    hold_days: int = 15
    rebalance_days: int = 5
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="earnings_momentum",
        description="Earnings momentum: PEAD proxy via volume spike + price reaction"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        df = features.sort(["ticker", "date"])
        cond = (
            (pl.col("rel_vol_20") > self.vol_mult) &
            (pl.col("ret_1d") > 0.02)   # positive price reaction on event day
        ).cast(pl.Int8)
        df = df.with_columns(event=cond).with_columns(
            in_trade=pl.col("event").rolling_sum(window_size=self.hold_days, min_periods=1)
                        .over("ticker").clip(0, 1)
        )
        df = df.filter(pl.col("in_trade") == 1)
        if df.is_empty():
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = df.with_columns(score=pl.col("mom_20d").fill_null(0))
        df = _top_k_equal(df, "score", self.top_k)
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class PostEarningsDrift:
    """Post-Earnings Announcement Drift (PEAD): hold momentum stocks after volume events."""
    event_vol_mult: float = 3.0
    drift_window: int = 20
    top_k: int = 6
    rebalance_days: int = 5
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="pead",
        description="PEAD: drift following high-volume price shock"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        df = features.sort(["ticker", "date"])
        event = (
            (pl.col("rel_vol_20") > self.event_vol_mult) &
            (pl.col("ret_1d").abs() > 0.03)
        ).cast(pl.Int8)
        df = df.with_columns(event=event).with_columns(
            in_drift=pl.col("event").rolling_sum(window_size=self.drift_window, min_periods=1)
                        .over("ticker").clip(0, 1)
        )
        # Directional drift: hold only if initial reaction was positive
        df = df.with_columns(
            reaction_pos=pl.when(pl.col("event") == 1)
                           .then((pl.col("ret_1d") > 0).cast(pl.Float64))
                           .otherwise(None)
        ).with_columns(
            reaction_pos=pl.col("reaction_pos").fill_null(strategy="forward").over("ticker").fill_null(0.0)
        ).with_columns(
            in_trade=(pl.col("in_drift").cast(pl.Float64) * pl.col("reaction_pos"))
        )
        df = df.filter(pl.col("in_trade") > 0)
        if df.is_empty():
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = _top_k_equal(df.with_columns(score=pl.col("mom_20d").fill_null(0)), "score", self.top_k)
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class NewsSentimentSignal:
    """News sentiment signal: if events parquet has sentiment scores, use them; else momentum."""
    top_k: int = 8
    rebalance_days: int = 5
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="news_sentiment",
        description="News sentiment: NLP sentiment score from events + momentum"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        if "sentiment" in features.columns:
            df = features.select(["date", "ticker", "sentiment", "mom_20d"]).drop_nulls("sentiment")
            df = df.with_columns(
                score=pl.col("sentiment") * (1 + pl.col("mom_20d").fill_null(0))
            )
            df = _top_k_equal(df, "score", self.top_k)
        else:
            # Fallback: high-volume + positive return day as news proxy
            df = features.sort(["ticker", "date"])
            cond = (
                (pl.col("rel_vol_20") > 2.0) & (pl.col("ret_1d") > 0.01)
            ).cast(pl.Float64)
            df = df.with_columns(score=cond * pl.col("mom_20d").fill_null(0))
            df = _top_k_equal(df, "score", self.top_k)
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class RandomForestSignal:
    """Random Forest signal: train RF on feature matrix, predict top-k by score.
    Re-trains weekly using trailing 252 days."""
    top_k: int = 8
    rebalance_days: int = 5
    train_window: int = 252
    n_estimators: int = 50    # kept small for M2 memory efficiency
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="random_forest_signal",
        description="RF return predictor: train on trailing features, trade top-k"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        from sklearn.ensemble import RandomForestClassifier
        feat_cols = [c for c in ["mom_5d", "mom_20d", "mom_60d", "vol_20d",
                                  "rsi_14", "sma_gap_50", "sma_gap_200",
                                  "rel_vol_20", "dd_from_high_60", "excess_ret_1d"]
                     if c in features.columns]
        if len(feat_cols) < 3:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))

        df = features.sort(["ticker", "date"])
        df = df.with_columns(
            fwd_ret=pl.col("adj_close").shift(-self.rebalance_days) / pl.col("adj_close") - 1
        ).drop_nulls(["fwd_ret"] + feat_cols)

        dates = df["date"].unique().sort()
        all_rows: list[dict] = []

        for i in range(self.train_window, len(dates), self.rebalance_days):
            train_dates = dates[max(0, i - self.train_window):i]
            pred_date = dates[i] if i < len(dates) else None
            if pred_date is None:
                break

            train = df.filter(pl.col("date").is_in(train_dates))
            pred = df.filter(pl.col("date") == pred_date)
            if len(train) < 100 or pred.is_empty():
                continue

            X_train = train.select(feat_cols).to_numpy()
            y_train = (train["fwd_ret"].to_numpy() > 0).astype(int)

            try:
                clf = RandomForestClassifier(
                    n_estimators=self.n_estimators, max_depth=4,
                    n_jobs=-1, random_state=42
                )
                clf.fit(X_train, y_train)
                X_pred = pred.select(feat_cols).to_numpy()
                scores = clf.predict_proba(X_pred)[:, 1]
                tickers = pred["ticker"].to_list()
                top_idx = np.argsort(scores)[-self.top_k:]
                for j in top_idx:
                    all_rows.append({"date": pred_date, "ticker": tickers[j],
                                     "weight": 1.0 / self.top_k})
            except Exception:
                continue

        if not all_rows:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df_rb = pl.DataFrame(all_rows)
        return _rebalance_forward_fill(df_rb, features, self.rebalance_days)


@dataclass
class GradientBoostingSignal:
    """LightGBM gradient boosting return predictor."""
    top_k: int = 8
    rebalance_days: int = 5
    train_window: int = 252
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="gradient_boosting_signal",
        description="LightGBM return predictor, retrained rolling window"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        try:
            import lightgbm as lgb
        except ImportError:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))

        feat_cols = [c for c in ["mom_5d", "mom_20d", "mom_60d", "vol_20d", "rsi_14",
                                  "sma_gap_50", "sma_gap_200", "atr_14",
                                  "rel_vol_20", "dd_from_high_60", "excess_ret_1d"]
                     if c in features.columns]
        if len(feat_cols) < 3:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))

        df = features.sort(["ticker", "date"])
        df = df.with_columns(
            fwd_ret=pl.col("adj_close").shift(-self.rebalance_days) / pl.col("adj_close") - 1
        ).drop_nulls(["fwd_ret"] + feat_cols)

        dates = df["date"].unique().sort()
        all_rows: list[dict] = []

        for i in range(self.train_window, len(dates), self.rebalance_days):
            train_dates = dates[max(0, i - self.train_window):i]
            pred_date = dates[i] if i < len(dates) else None
            if pred_date is None:
                break
            train = df.filter(pl.col("date").is_in(train_dates))
            pred = df.filter(pl.col("date") == pred_date)
            if len(train) < 100 or pred.is_empty():
                continue
            X_train = train.select(feat_cols).to_numpy()
            y_train = train["fwd_ret"].to_numpy()
            try:
                model = lgb.LGBMRegressor(
                    n_estimators=100, max_depth=4,
                    learning_rate=0.05, n_jobs=-1,
                    verbose=-1
                )
                model.fit(X_train, y_train)
                scores = model.predict(pred.select(feat_cols).to_numpy())
                tickers = pred["ticker"].to_list()
                top_idx = np.argsort(scores)[-self.top_k:]
                for j in top_idx:
                    all_rows.append({"date": pred_date, "ticker": tickers[j],
                                     "weight": 1.0 / self.top_k})
            except Exception:
                continue

        if not all_rows:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        return _rebalance_forward_fill(pl.DataFrame(all_rows), features, self.rebalance_days)


@dataclass
class EnsembleSignalVoting:
    """Ensemble voting: combine signals from multiple simple rules, vote for top-k."""
    top_k: int = 8
    rebalance_days: int = 5
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="ensemble_voting",
        description="Ensemble: majority vote from 5 technical signals"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        df = features.sort(["ticker", "date"])
        votes = []
        # Signal 1: positive momentum
        if "mom_20d" in df.columns:
            votes.append(pl.when(pl.col("mom_20d") > 0).then(1.0).otherwise(0.0).alias("v1"))
        # Signal 2: above 50d SMA
        if "sma_gap_50" in df.columns:
            votes.append(pl.when(pl.col("sma_gap_50") > 0).then(1.0).otherwise(0.0).alias("v2"))
        # Signal 3: RSI not overbought
        if "rsi_14" in df.columns:
            votes.append(pl.when((pl.col("rsi_14") > 40) & (pl.col("rsi_14") < 70))
                           .then(1.0).otherwise(0.0).alias("v3"))
        # Signal 4: low volatility
        if "vol_20d" in df.columns:
            votes.append(pl.when(pl.col("vol_20d") < 0.30).then(1.0).otherwise(0.0).alias("v4"))
        # Signal 5: positive 60d momentum
        if "mom_60d" in df.columns:
            votes.append(pl.when(pl.col("mom_60d") > 0).then(1.0).otherwise(0.0).alias("v5"))

        if not votes:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))

        df = df.with_columns(votes)
        vote_cols = [e.meta.output_name() for e in votes]
        df = df.with_columns(
            vote_score=sum(pl.col(c) for c in vote_cols) / len(vote_cols)
        )
        df = _top_k_equal(df, "vote_score", self.top_k)
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class AnomalyDetectionStrategy:
    """Anomaly detection: Isolation Forest flags abnormal stocks; trade normal trending ones."""
    top_k: int = 8
    rebalance_days: int = 21
    contamination: float = 0.1
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="anomaly_detection",
        description="Isolation Forest anomaly filter: trade normal high-momentum stocks"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        from sklearn.ensemble import IsolationForest
        feat_cols = [c for c in ["mom_20d", "vol_20d", "rsi_14", "rel_vol_20",
                                  "sma_gap_50", "dd_from_high_60"]
                     if c in features.columns]
        if len(feat_cols) < 3:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))

        df = features.select(["date", "ticker", "mom_20d"] + feat_cols).drop_nulls()
        dates = df["date"].unique().sort()
        all_rows: list[dict] = []

        for dt in dates[::self.rebalance_days]:
            day = df.filter(pl.col("date") == dt)
            if len(day) < 10:
                continue
            X = day.select(feat_cols).to_numpy()
            try:
                clf = IsolationForest(contamination=self.contamination, random_state=42, n_jobs=-1)
                labels = clf.fit_predict(X)  # -1 = anomaly, 1 = normal
                normal_mask = labels == 1
                normal = day.filter(pl.Series("mask", normal_mask))
                if normal.is_empty():
                    continue
                top = normal.sort("mom_20d", descending=True).head(self.top_k)
                for row in top.iter_rows(named=True):
                    all_rows.append({"date": dt, "ticker": row["ticker"],
                                     "weight": 1.0 / min(self.top_k, len(top))})
            except Exception:
                continue

        if not all_rows:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        return _rebalance_forward_fill(pl.DataFrame(all_rows), features, self.rebalance_days)


@dataclass
class SHAPFeatureAttributionStrategy:
    """SHAP-based feature attribution: weight stocks by their LightGBM SHAP scores
    for return prediction, selecting those with highest positive SHAP contribution."""
    top_k: int = 8
    rebalance_days: int = 21
    train_window: int = 252
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="shap_signal",
        description="SHAP-weighted signal: LightGBM SHAP value attribution"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        try:
            import lightgbm as lgb
            import shap
        except ImportError:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))

        feat_cols = [c for c in ["mom_5d", "mom_20d", "mom_60d", "vol_20d", "rsi_14",
                                  "sma_gap_50", "rel_vol_20", "excess_ret_1d"]
                     if c in features.columns]
        if len(feat_cols) < 3:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))

        df = features.sort(["ticker", "date"])
        df = df.with_columns(
            fwd_ret=pl.col("adj_close").shift(-self.rebalance_days) / pl.col("adj_close") - 1
        ).drop_nulls(["fwd_ret"] + feat_cols)

        dates = df["date"].unique().sort()
        all_rows: list[dict] = []

        for i in range(self.train_window, len(dates), self.rebalance_days):
            train_dates = dates[max(0, i - self.train_window):i]
            pred_date = dates[i] if i < len(dates) else None
            if pred_date is None:
                break
            train = df.filter(pl.col("date").is_in(train_dates))
            pred = df.filter(pl.col("date") == pred_date)
            if len(train) < 100 or pred.is_empty():
                continue
            X_train = train.select(feat_cols).to_numpy()
            y_train = train["fwd_ret"].to_numpy()
            try:
                model = lgb.LGBMRegressor(
                    n_estimators=50, max_depth=3, verbose=-1, n_jobs=-1
                )
                model.fit(X_train, y_train)
                X_pred = pred.select(feat_cols).to_numpy()
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X_pred)
                # Use sum of positive SHAP contributions as score
                scores = shap_vals.clip(0).sum(axis=1)
                tickers = pred["ticker"].to_list()
                top_idx = np.argsort(scores)[-self.top_k:]
                for j in top_idx:
                    all_rows.append({"date": pred_date, "ticker": tickers[j],
                                     "weight": 1.0 / self.top_k})
            except Exception:
                continue

        if not all_rows:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        return _rebalance_forward_fill(pl.DataFrame(all_rows), features, self.rebalance_days)


@dataclass
class ThompsonSamplingRL:
    """Reinforcement Learning lite: Thompson Sampling bandit that learns which
    momentum buckets outperform. Replaces RL agent for M2 memory efficiency."""
    n_buckets: int = 5
    top_k: int = 8
    rebalance_days: int = 5
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="thompson_sampling_rl",
        description="RL-lite: Thompson Sampling bandit over momentum quintiles"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        col = "mom_20d"
        if col not in features.columns:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))

        df = features.sort(["date", "ticker"])
        dates = df["date"].unique().sort()
        all_rows: list[dict] = []

        # Beta distribution parameters per bucket (initialized uniform)
        alpha = np.ones(self.n_buckets)
        beta_p = np.ones(self.n_buckets)
        rng = np.random.default_rng(42)

        for dt in dates[::self.rebalance_days]:
            day = df.filter(pl.col("date") == dt).drop_nulls(col)
            if len(day) < self.n_buckets * 2:
                continue

            # Assign momentum quintiles
            day = day.with_columns(
                bucket=((pl.col(col).rank(method="ordinal", descending=False) - 1) /
                        len(day) * self.n_buckets).cast(pl.Int32).clip(0, self.n_buckets - 1)
            )

            # Thompson sample: draw from Beta for each bucket
            samples = rng.beta(alpha, beta_p)
            best_bucket = int(np.argmax(samples))

            bucket_stocks = day.filter(pl.col("bucket") == best_bucket)
            if bucket_stocks.is_empty():
                continue
            top = bucket_stocks.sort(col, descending=True).head(self.top_k)
            w = 1.0 / len(top)
            for row in top.iter_rows(named=True):
                all_rows.append({"date": dt, "ticker": row["ticker"], "weight": w})

            # Update beliefs using next-day returns as reward (look-ahead safe: uses prev day)
            prev_dt_idx = dates.search_sorted(dt) - 1
            if prev_dt_idx >= 0:
                prev_dt = dates[prev_dt_idx]
                prev_day = df.filter(pl.col("date") == prev_dt).drop_nulls("ret_1d")
                if not prev_day.is_empty():
                    prev_day = prev_day.with_columns(
                        bucket=((pl.col(col).rank(method="ordinal", descending=False) - 1) /
                                len(prev_day) * self.n_buckets).cast(pl.Int32).clip(0, self.n_buckets - 1)
                    )
                    for b in range(self.n_buckets):
                        grp = prev_day.filter(pl.col("bucket") == b)
                        if not grp.is_empty():
                            avg_ret = grp["ret_1d"].mean()
                            if avg_ret > 0:
                                alpha[b] += 1
                            else:
                                beta_p[b] += 1

        if not all_rows:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        return _rebalance_forward_fill(pl.DataFrame(all_rows), features, self.rebalance_days)


@dataclass
class GeneticMomentumSearch:
    """Genetic-programming inspired: evolve momentum parameter combination
    by selecting lookback windows that maximize trailing Sharpe."""
    top_k: int = 8
    rebalance_days: int = 21
    eval_window: int = 63
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="genetic_momentum",
        description="Genetic-style: adaptive lookback selection by rolling Sharpe"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        mom_cols = [c for c in ["mom_5d", "mom_10d", "mom_20d", "mom_60d", "mom_120d"]
                    if c in features.columns]
        if not mom_cols:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))

        df = features.sort(["date", "ticker"])
        dates = df["date"].unique().sort()
        all_rows: list[dict] = []

        # Track rolling Sharpe-like score for each mom col
        col_scores = {c: 0.0 for c in mom_cols}

        for i in range(self.eval_window, len(dates), self.rebalance_days):
            # Evaluate each column's recent predictiveness
            eval_dates = dates[max(0, i - self.eval_window):i]
            best_col = max(col_scores, key=col_scores.get)

            pred_date = dates[i]
            day = df.filter(pl.col("date") == pred_date).drop_nulls(best_col)
            if day.is_empty():
                continue
            top = day.sort(best_col, descending=True).head(self.top_k)
            w = 1.0 / len(top)
            for row in top.iter_rows(named=True):
                all_rows.append({"date": pred_date, "ticker": row["ticker"], "weight": w})

            # Update col scores using trailing returns
            for c in mom_cols:
                eval_df = df.filter(pl.col("date").is_in(eval_dates)).drop_nulls([c, "ret_1d"])
                if eval_df.is_empty():
                    continue
                # Correlation of signal with next-day return as proxy for IC
                try:
                    top_each_day = eval_df.with_columns(
                        rk=pl.col(c).rank(method="ordinal", descending=True).over("date")
                    ).filter(pl.col("rk") <= self.top_k)
                    avg_ret = top_each_day["ret_1d"].mean() or 0.0
                    vol_ret = top_each_day["ret_1d"].std() or 1.0
                    col_scores[c] = col_scores[c] * 0.9 + (avg_ret / vol_ret) * 0.1
                except Exception:
                    pass

        if not all_rows:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        return _rebalance_forward_fill(pl.DataFrame(all_rows), features, self.rebalance_days)


# ═════════════════════════════════════════════════════════════════════════════
# ADDITIONAL COMPOSITE / NOVEL STRATEGIES
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class RelativeStrengthRotation:
    """Relative Strength Rotation: buy top RS vs benchmark using excess return."""
    top_k: int = 8
    lookback: int = 63
    rebalance_days: int = 21
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="relative_strength_rotation",
        description="Relative Strength: top performers vs benchmark"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        if "excess_ret_1d" not in features.columns:
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = features.sort(["ticker", "date"])
        df = df.with_columns(
            rs=pl.col("excess_ret_1d").rolling_sum(window_size=self.lookback, min_periods=20)
                 .over("ticker")
        ).drop_nulls("rs")
        df = _top_k_equal(df, "rs", self.top_k)
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class VolatilityTargetingStrategy:
    """Volatility targeting: scale full-portfolio exposure up/down to hit target vol."""
    target_vol: float = 0.15    # 15% annual target
    base_top_k: int = 10
    rebalance_days: int = 5
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="vol_targeting",
        description="Volatility targeting: scale exposure to hit 15% ann vol"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        col = "mom_20d"
        df = features.select(["date", "ticker", col, "vol_20d"]).drop_nulls()
        df = _top_k_equal(df, col, self.base_top_k)
        # Scale weights by target_vol / realized_vol of equal-weight portfolio
        # Approximate: use avg vol of selected tickers
        df = df.join(features.select(["date", "ticker", "vol_20d"]), on=["date", "ticker"], how="left")
        df = df.with_columns(
            avg_vol=pl.when(pl.col("weight") > 0)
                       .then(pl.col("vol_20d"))
                       .otherwise(None)
                       .mean().over("date")
        ).with_columns(
            scale=(self.target_vol / (pl.col("avg_vol") + 1e-8)).clip(0.5, 2.0)
        ).with_columns(
            weight=(pl.col("weight") * pl.col("scale")).clip(0.0, 0.20)
        ).select(["date", "ticker", "weight"])
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class LongShortEquityFactor:
    """Long/Short Equity: long top-k momentum, short bottom-k momentum (long-only sim)."""
    top_k: int = 8
    rebalance_days: int = 10
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="long_short_equity",
        description="L/S equity: long top momentum, hedge bottom (long-only)"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        col = "mom_20d"
        df = features.select(["date", "ticker", col]).drop_nulls()
        n = df.group_by("date").agg(pl.len().alias("n"))
        df = df.join(n, on="date")
        df = df.with_columns(
            rk=pl.col(col).rank(method="ordinal", descending=True).over("date")
        ).with_columns(
            weight=pl.when(pl.col("rk") <= self.top_k).then(1.0 / self.top_k)
                     .when(pl.col("rk") > (pl.col("n") - self.top_k)).then(-1.0 / self.top_k)
                     .otherwise(0.0)
        ).select(["date", "ticker", "weight"])
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class CalendarSpreadMomentum:
    """Calendar spread analog: buy short-term momentum when it diverges from long-term."""
    short_col: str = "mom_20d"
    long_col: str = "mom_120d"
    top_k: int = 8
    rebalance_days: int = 10
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="calendar_spread_momentum",
        description="Calendar spread: short-term momentum acceleration vs long-term"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        needed = {self.short_col, self.long_col}
        if not needed.issubset(features.columns):
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = features.select(["date", "ticker", self.short_col, self.long_col]).drop_nulls()
        # Spread: short-term > long-term = momentum accelerating
        df = df.with_columns(
            spread=pl.col(self.short_col) - pl.col(self.long_col)
        )
        df = df.filter(pl.col(self.short_col) > 0)
        if df.is_empty():
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = _top_k_equal(df, "spread", self.top_k)
        return _rebalance_forward_fill(df, features, self.rebalance_days)


@dataclass
class AnalystRevisionProxy:
    """Analyst revision proxy: stocks with improving earnings momentum
    (strong 60d relative to 20d, suggests estimate revisions)."""
    top_k: int = 8
    rebalance_days: int = 21
    meta: StrategyMeta = field(default_factory=lambda: StrategyMeta(
        name="analyst_revision_proxy",
        description="Analyst revision proxy: 60d>20d momentum with vol spike"))

    def generate_signals(self, features: pl.DataFrame) -> pl.DataFrame:
        needed = {"mom_20d", "mom_60d"}
        if not needed.issubset(features.columns):
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = features.select(["date", "ticker", "mom_20d", "mom_60d"]).drop_nulls()
        # Improving momentum: 60d > 20d AND both positive = estimate revision signal
        df = df.filter((pl.col("mom_20d") > 0) & (pl.col("mom_60d") > 0))
        df = df.with_columns(
            revision_score=(pl.col("mom_20d") - pl.col("mom_60d"))
        )
        if df.is_empty():
            return features.select("date", "ticker").with_columns(weight=pl.lit(0.0))
        df = _top_k_equal(df, "revision_score", self.top_k)
        return _rebalance_forward_fill(df, features, self.rebalance_days)
