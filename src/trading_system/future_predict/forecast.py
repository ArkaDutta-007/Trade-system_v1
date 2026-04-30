"""Future prediction sessions.

Each session lives in future_predict/YYYY-MM-DD/ and contains:
  forecast.json       — all ticker scores, horizon target dates, $10k portfolio allocation
  equity_log.parquet  — daily MTM equity snapshots

Design rules for the $10k portfolio
  - Never deploy more than 60% of budget at once (keep 40% liquid)
  - Max 10% of budget per single position
  - Take the top-ranked BUY signals up to TOP_N positions
  - Cash never goes negative
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

# ── Portfolio constants ───────────────────────────────────────────────────────
HORIZONS: dict[str, int] = {
    "1m":  21,
    "2m":  42,
    "3m":  63,
    "6m":  126,
    "12m": 252,
}

MAX_DEPLOY_PCT   = 0.60   # deploy at most 60% initially; keep 40% liquid
MAX_POSITION_PCT = 0.10   # max 10% of budget per position
TOP_N            = 15     # max number of long positions
MIN_SCORE        = 0.003  # minimum ensemble score to open a BUY


# ── Helpers ───────────────────────────────────────────────────────────────────

def _add_trading_days(start: date, n: int) -> date:
    """Return the date that is approximately n trading days after start."""
    d = start
    added = 0
    while added < n:
        d += timedelta(days=1)
        if d.weekday() < 5:  # Mon–Fri only
            added += 1
    return d


def _scores_from_model(model, X: np.ndarray, best_variant: str) -> np.ndarray:
    """Extract a 1-D score array from either an EnsembleModel or a plain estimator."""
    raw = model.predict(X)
    if isinstance(raw, dict):
        arr = raw.get(best_variant)
        if arr is None:
            arr = raw.get("ensemble_blend")
        if arr is None:
            arr = next(iter(raw.values()))
    else:
        arr = raw
    return np.asarray(arr).flatten()


def _append_equity_row(
    log_path: Path,
    as_of: date,
    equity: float,
    deployed: float,
    cash: float,
    n_positions: int,
    initial_budget: float,
) -> None:
    """Append one equity row to the session's parquet log."""
    return_pct = (equity - initial_budget) / initial_budget if initial_budget else 0.0
    new_row = pl.DataFrame({
        "date":        [str(as_of)],
        "equity":      [equity],
        "deployed":    [deployed],
        "cash":        [cash],
        "n_positions": [n_positions],
        "return_pct":  [return_pct],
    }).with_columns(pl.col("n_positions").cast(pl.Int32))

    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        existing = pl.read_parquet(log_path)
        pl.concat([existing, new_row]).write_parquet(log_path, compression="zstd")
    else:
        new_row.write_parquet(log_path, compression="zstd")


# ── Core API ──────────────────────────────────────────────────────────────────

def run_forecast(
    session_dir: Path,
    features_path: Path,
    ohlcv_path: Path,
    model,
    feature_columns: list[str],
    best_variant: str,
    model_name: str,
    budget: float = 10_000.0,
    prediction_date: date | None = None,
) -> dict:
    """Create a new forecast session and save all artefacts.

    Steps
    -----
    1. Load latest features; score all tickers using the ensemble model.
    2. Build horizon target dates (1m … 12m) as forward trading-day estimates.
    3. Allocate up to 60% of budget across top-ranked BUY signals (≤10% each).
    4. Save forecast.json + initial equity snapshot.

    Returns the full forecast dict.
    """
    if prediction_date is None:
        prediction_date = date.today()
    session_dir.mkdir(parents=True, exist_ok=True)

    # ── Features for latest available date ─────────────────────────────────
    features = pl.read_parquet(features_path)
    latest_feat_date = features["date"].max()
    today_feat = features.filter(pl.col("date") == latest_feat_date)

    # ── Latest OHLCV prices ────────────────────────────────────────────────
    ohlcv = pl.read_parquet(ohlcv_path)
    latest_px_date = ohlcv["date"].max()
    prices = {
        r["ticker"]: float(r["adj_close"])
        for r in ohlcv.filter(pl.col("date") == latest_px_date)
                       .select(["ticker", "adj_close"]).to_dicts()
        if r.get("adj_close")
    }

    # ── Score every ticker ─────────────────────────────────────────────────
    avail_cols = [c for c in feature_columns if c in today_feat.columns]
    X = today_feat.select(avail_cols).fill_nan(0.0).fill_null(0.0).to_numpy()
    tickers = today_feat["ticker"].to_list()
    scores_arr = _scores_from_model(model, X, best_variant)

    scored = sorted(zip(tickers, scores_arr.tolist()), key=lambda x: x[1], reverse=True)

    # ── Horizon target dates ───────────────────────────────────────────────
    horizon_dates = {
        label: _add_trading_days(prediction_date, td).isoformat()
        for label, td in HORIZONS.items()
    }

    # ── Allocate budget (greedy, rank-ordered) ─────────────────────────────
    max_deploy  = budget * MAX_DEPLOY_PCT
    max_per_pos = budget * MAX_POSITION_PCT
    positions: list[dict] = []
    deployed = 0.0

    buy_candidates = [(t, s) for t, s in scored if s >= MIN_SCORE][:TOP_N]
    for ticker, score in buy_candidates:
        px = prices.get(ticker)
        if not px or px <= 0:
            continue
        alloc = min(max_per_pos, max_deploy - deployed)
        if alloc < 50.0:
            break
        shares = alloc / px
        deployed += alloc
        positions.append({
            "ticker":      ticker,
            "score":       round(float(score), 6),
            "entry_price": round(px, 4),
            "allocated":   round(alloc, 2),
            "shares":      round(shares, 6),
        })
        if deployed >= max_deploy:
            break

    cash_reserved = budget - deployed

    # ── Full predictions list (all tickers, for hit-rate analysis) ─────────
    all_predictions = [
        {
            "ticker":      t,
            "score":       round(float(s), 6),
            "stance":      "BUY" if s >= MIN_SCORE else ("SELL" if s <= -MIN_SCORE else "HOLD"),
            "entry_price": round(prices.get(t, 0.0), 4),
        }
        for t, s in scored
    ]

    # ── Compose + save forecast.json ──────────────────────────────────────
    forecast = {
        "prediction_date": prediction_date.isoformat(),
        "features_as_of":  str(latest_feat_date),
        "prices_as_of":    str(latest_px_date),
        "model":           model_name,
        "best_variant":    best_variant,
        "budget":          budget,
        "horizons":        horizon_dates,
        "portfolio": {
            "initial_cash":  budget,
            "deployed":      round(deployed, 2),
            "cash_reserved": round(cash_reserved, 2),
            "positions":     positions,
        },
        "all_predictions": all_predictions,
    }
    (session_dir / "forecast.json").write_text(json.dumps(forecast, indent=2))

    # ── Initial equity snapshot ────────────────────────────────────────────
    _append_equity_row(
        session_dir / "equity_log.parquet",
        prediction_date, budget, deployed, cash_reserved, len(positions), budget,
    )

    return forecast


def update_session_equity(
    session_dir: Path,
    ohlcv_path: Path,
    as_of: date | None = None,
) -> dict:
    """Refresh the session equity log using current market prices (MTM).

    Called daily by `ts future-update` / `ts daily`.
    Returns current equity metrics.
    """
    if as_of is None:
        as_of = date.today()

    forecast_path = session_dir / "forecast.json"
    if not forecast_path.exists():
        raise FileNotFoundError(f"No forecast.json in {session_dir}")

    forecast = json.loads(forecast_path.read_text())
    positions     = forecast["portfolio"]["positions"]
    cash_reserved = forecast["portfolio"]["cash_reserved"]
    initial_budget = forecast["budget"]

    # Latest prices
    ohlcv = pl.read_parquet(ohlcv_path)
    latest_px_date = ohlcv["date"].max()
    prices = {
        r["ticker"]: float(r["adj_close"])
        for r in ohlcv.filter(pl.col("date") == latest_px_date)
                       .select(["ticker", "adj_close"]).to_dicts()
        if r.get("adj_close")
    }

    mtm = sum(
        p["shares"] * prices.get(p["ticker"], p["entry_price"])
        for p in positions
    )
    equity     = mtm + cash_reserved
    return_pct = (equity - initial_budget) / initial_budget

    _append_equity_row(
        session_dir / "equity_log.parquet",
        as_of, equity, mtm, cash_reserved, len(positions), initial_budget,
    )

    return {
        "equity":      equity,
        "deployed_mtm": mtm,
        "cash":        cash_reserved,
        "return_pct":  return_pct,
        "prices_as_of": str(latest_px_date),
    }


def evaluate_predictions(session_dir: Path, ohlcv_path: Path) -> dict:
    """Compare all_predictions against actual future prices.

    For each horizon that has elapsed, computes:
      - directional hit rate (predicted_sign == actual_sign)
      - mean realised return

    Returns dict keyed by horizon label, e.g. "1m", "3m", etc.
    """
    forecast_path = session_dir / "forecast.json"
    if not forecast_path.exists():
        return {}

    forecast      = json.loads(forecast_path.read_text())
    all_preds     = forecast["all_predictions"]
    horizon_dates = forecast["horizons"]

    ohlcv = pl.read_parquet(ohlcv_path)
    available_dates = sorted(str(d) for d in ohlcv["date"].unique().to_list())

    results: dict[str, dict] = {}
    for label, target_date_str in horizon_dates.items():
        past_dates = [d for d in available_dates if d <= target_date_str]
        if not past_dates:
            results[label] = {"status": "pending", "target_date": target_date_str}
            continue

        actual_date_str = past_dates[-1]
        px_df = (
            ohlcv
            .with_columns(pl.col("date").cast(pl.Utf8).alias("date_str"))
            .filter(pl.col("date_str") == actual_date_str)
            .select(["ticker", "adj_close"])
        )
        actual_prices = {r["ticker"]: float(r["adj_close"]) for r in px_df.to_dicts()}

        hits, total = 0, 0
        returns: list[float] = []
        ticker_results: list[dict] = []

        for p in all_preds:
            entry  = p["entry_price"]
            actual = actual_prices.get(p["ticker"])
            if not actual or entry <= 0:
                continue
            ret    = (actual - entry) / entry
            hit    = (p["score"] > 0) == (ret > 0)
            hits  += int(hit)
            total += 1
            returns.append(ret)
            ticker_results.append({
                "ticker":        p["ticker"],
                "score":         p["score"],
                "entry_price":   entry,
                "actual_price":  round(actual, 4),
                "actual_return": round(ret, 4),
                "hit":           hit,
            })

        results[label] = {
            "status":       "available",
            "target_date":  target_date_str,
            "actual_date":  actual_date_str,
            "hit_rate":     round(hits / total, 4) if total else None,
            "mean_return":  round(float(np.mean(returns)), 4) if returns else None,
            "total_tickers": total,
            # Top 20 by absolute score for review
            "top_picks": sorted(
                [r for r in ticker_results if r["score"] >= MIN_SCORE],
                key=lambda x: x["score"], reverse=True,
            )[:20],
        }

    return results


def list_sessions(base_dir: Path) -> list[Path]:
    """Return all session directories, newest first."""
    if not base_dir.exists():
        return []
    return sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and (d / "forecast.json").exists()],
        reverse=True,
    )
