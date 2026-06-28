"""Extended feature reserve — microstructure, distributional, and beta features.

These complement ``technical.py`` / ``regimes.py`` / ``macro.py`` to give the
long-horizon models a broad, leakage-safe reserve to select from.  Everything is
computed per ticker from past values only (rolling windows, no forward shifts).

Adds:
  Liquidity / microstructure
    amihud_illiq_20      : Amihud illiquidity = mean(|ret| / dollar_volume)
    volume_z_60          : volume z-score vs 60d
    overnight_gap        : (open / prev_close - 1)
  Distributional (tail-aware — matters for long-horizon risk)
    downside_vol_20/60   : annualised std of negative returns only
    ret_skew_60          : rolling return skew
    ret_kurt_60          : rolling return kurtosis (fat tails)
    vol_of_vol_60        : std of the 20d-vol series (vol regime instability)
    max_dd_252           : worst drawdown over the past year
  Trend shape
    mom_accel            : mom_20d - mom_60d (is the trend accelerating?)
    dist_52w_high        : (price / 252d high) - 1
    dist_52w_low         : (price / 252d low) - 1
    bb_pctb_20           : Bollinger %b position in the 20d band
  Benchmark-relative
    beta_60              : rolling beta to the benchmark
    corr_bench_60        : rolling correlation to the benchmark
"""
from __future__ import annotations

import numpy as np
import polars as pl


EXTENDED_COLUMNS = [
    "amihud_illiq_20", "volume_z_60", "overnight_gap",
    "downside_vol_20", "downside_vol_60", "ret_skew_60", "ret_kurt_60",
    "vol_of_vol_60", "max_dd_252",
    "mom_accel", "dist_52w_high", "dist_52w_low", "bb_pctb_20",
    "beta_60", "corr_bench_60",
]


def compute_extended_features(df: pl.DataFrame, benchmark: str = "SPY") -> pl.DataFrame:
    """Add extended features. Expects technical features already computed.

    Requires columns: date, ticker, open, high, low, close, adj_close, volume,
    ret_1d, log_ret_1d (the last two come from compute_technical_features).
    """
    if df.is_empty():
        return df
    px = pl.col("adj_close")
    r = pl.col("ret_1d")
    df = df.sort(["ticker", "date"])

    # ── Liquidity / microstructure ──────────────────────────────────────────
    df = df.with_columns(_dollar_vol=(pl.col("close") * pl.col("volume")))
    df = df.with_columns(
        amihud_illiq_20=(
            (r.abs() / (pl.col("_dollar_vol") + 1.0))
            .rolling_mean(window_size=20).over("ticker") * 1e9
        ),
        volume_z_60=(
            (pl.col("volume") - pl.col("volume").rolling_mean(60).over("ticker"))
            / (pl.col("volume").rolling_std(60).over("ticker") + 1e-9)
        ),
        overnight_gap=(pl.col("open") / pl.col("close").shift(1).over("ticker") - 1),
    )

    # ── Distributional / tail-aware ─────────────────────────────────────────
    neg = pl.when(r < 0).then(r).otherwise(None)
    for w in (20, 60):
        df = df.with_columns(
            (neg.rolling_std(window_size=w, min_samples=max(5, w // 4)).over("ticker")
             * np.sqrt(252)).alias(f"downside_vol_{w}d")
        )
    df = df.with_columns(
        ret_skew_60=r.rolling_skew(window_size=60).over("ticker"),
        vol_of_vol_60=pl.col("vol_20d").rolling_std(window_size=60).over("ticker")
        if "vol_20d" in df.columns else pl.lit(None),
    )
    # kurtosis via 4th standardized moment (polars has no rolling_kurt)
    mean_60 = r.rolling_mean(60).over("ticker")
    std_60 = r.rolling_std(60).over("ticker")
    df = df.with_columns(
        ret_kurt_60=(((r - mean_60) / (std_60 + 1e-12)).pow(4))
        .rolling_mean(60).over("ticker") - 3.0
    )

    # ── Trend shape ─────────────────────────────────────────────────────────
    df = df.with_columns(
        rolling_max_252=px.rolling_max(252, min_samples=60).over("ticker"),
        rolling_min_252=px.rolling_min(252, min_samples=60).over("ticker"),
        sma_20_x=px.rolling_mean(20).over("ticker"),
        std_20_x=px.rolling_std(20).over("ticker"),
    ).with_columns(
        dist_52w_high=(px / pl.col("rolling_max_252") - 1),
        dist_52w_low=(px / pl.col("rolling_min_252") - 1),
        bb_pctb_20=((px - (pl.col("sma_20_x") - 2 * pl.col("std_20_x")))
                    / (4 * pl.col("std_20_x") + 1e-9)),
    )
    if "mom_20d" in df.columns and "mom_60d" in df.columns:
        df = df.with_columns(mom_accel=(pl.col("mom_20d") - pl.col("mom_60d")))
    else:
        df = df.with_columns(mom_accel=pl.lit(0.0))

    # max drawdown over 252d: 1 - price/rolling_max
    df = df.with_columns(
        max_dd_252=(px / pl.col("rolling_max_252") - 1).rolling_min(252, min_samples=60).over("ticker")
    )

    # ── Benchmark-relative (beta / correlation) ─────────────────────────────
    bench = (
        df.filter(pl.col("ticker") == benchmark)
        .select(["date", pl.col("ret_1d").alias("_rb")])
        .unique(subset=["date"])
    )
    if not bench.is_empty():
        df = df.join(bench, on="date", how="left")
        w = 60
        rb = pl.col("_rb")
        cov = (r * rb).rolling_mean(w).over("ticker") - \
            (r.rolling_mean(w).over("ticker") * rb.rolling_mean(w).over("ticker"))
        var_b = (rb.pow(2)).rolling_mean(w).over("ticker") - rb.rolling_mean(w).over("ticker").pow(2)
        std_i = r.rolling_std(w).over("ticker")
        std_b = rb.rolling_std(w).over("ticker")
        df = df.with_columns(
            beta_60=(cov / (var_b + 1e-12)),
            corr_bench_60=(cov / (std_i * std_b + 1e-12)),
        ).drop("_rb")
    else:
        df = df.with_columns(beta_60=pl.lit(1.0), corr_bench_60=pl.lit(0.0))

    return df.drop([c for c in ["_dollar_vol", "rolling_max_252", "rolling_min_252",
                                "sma_20_x", "std_20_x"] if c in df.columns])
