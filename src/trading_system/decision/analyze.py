"""Single-symbol decision pipeline.

`analyze_symbol(ticker)` runs the full system on one ticker:
  1. Loads (or refreshes) OHLCV for the universe — context matters for cross-sectional features.
  2. Builds the feature matrix.
  3. Loads the latest model from the registry, or falls back to a rule-based score.
  4. Produces 5d and 20d forecasts, a BUY/HOLD/SELL stance, and a confidence band.
  5. Collects groundings (technical, regime, cross-sectional, events, model + SHAP).
  6. Writes a markdown report and a JSON audit record under reports/decisions/.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any
import os

import numpy as np
import polars as pl

from ..config import Config, get_config
from ..features import build_feature_matrix
from ..models.model_registry import list_models, load_model
from ..models.model_registry import best_ensemble_artifact
from ..models.predict import predict_with_model
from ..models.shap_analysis import compute_shap_summary
from ..utils import get_logger
from .groundings import (
    technical_grounding,
    regime_grounding,
    cross_section_grounding,
    event_grounding,
    model_grounding,
)
from .report import write_decision_report
from .explain import explain_report, DEEPSEEK_DEFAULT_MODEL

logger = get_logger(__name__)


@dataclass
class DecisionResult:
    ticker: str
    as_of: str
    stance: str           # BUY | HOLD | SELL
    confidence: float     # 0..1
    forecast_5d: float
    forecast_20d: float
    score_source: str     # "model" | "rules"
    rationale: list[str]  # human-readable bullet reasons
    groundings: dict[str, Any]
    report_path: str | None = None
    json_path: str | None = None
    decision_payload: dict | None = field(default=None)


def _load_or_build_features(cfg: Config) -> tuple[pl.DataFrame, pl.DataFrame]:
    bronze = cfg.path("data_bronze") / "ohlcv_daily.parquet"
    if not bronze.exists():
        from ..ingestion import ingest_universe
        ingest_universe(cfg)
    ohlcv = pl.read_parquet(bronze)
    feats = cfg.path("data_gold") / "features.parquet"
    if feats.exists():
        features = pl.read_parquet(feats)
    else:
        events = _load_events(cfg)
        apprehension = _load_apprehension(cfg)
        from ..features.context import build_macro_inputs
        macro_features, econ_cal, _ = build_macro_inputs(cfg, tickers=None, with_earnings=False)
        features = build_feature_matrix(
            ohlcv,
            events=events,
            apprehension=apprehension,
            economic_calendar=econ_cal,
            macro_features=macro_features,
            benchmark=cfg["universe"]["benchmark"],
        )
    return ohlcv, features


def _load_events(cfg: Config) -> pl.DataFrame | None:
    p = cfg.path("data_silver") / "events.parquet"
    if p.exists():
        return pl.read_parquet(p)
    return None


def _load_apprehension(cfg: Config) -> pl.DataFrame | None:
    p = cfg.path("data_silver") / "apprehension_scores.parquet"
    if p.exists():
        return pl.read_parquet(p)
    return None


def _apprehension_grounding(df: pl.DataFrame | None, ticker: str) -> dict:
    """Return the latest apprehension score + drivers for a ticker."""
    if df is None or df.is_empty():
        return {"available": False}
    sub = df.filter(pl.col("ticker") == ticker.upper()).sort("date", descending=True).head(1)
    if sub.is_empty():
        return {"available": False}
    row = sub.to_dicts()[0]
    return {
        "available": True,
        "as_of": str(row["date"]),
        "apprehension_score": row["apprehension_score"],
        "outlook": row["outlook"],
        "drivers": row.get("apprehension_drivers") or [],
    }


def _rule_based_score(row: dict) -> tuple[float, list[str]]:
    """Fallback if no model available. Returns (expected_5d_return, reasons)."""
    reasons = []
    score = 0.0
    # Trend
    if (row.get("mom_20d") or 0) > 0:
        score += 0.0030; reasons.append("20d momentum positive")
    if (row.get("mom_60d") or 0) > 0:
        score += 0.0020; reasons.append("60d momentum positive")
    if (row.get("sma_gap_200") or 0) > 0:
        score += 0.0020; reasons.append("price above 200d SMA")
    # Mean reversion if oversold
    rsi = row.get("rsi_14")
    if rsi is not None and rsi < 30:
        score += 0.0040; reasons.append(f"RSI={rsi:.1f} oversold")
    elif rsi is not None and rsi > 75:
        score -= 0.0040; reasons.append(f"RSI={rsi:.1f} overbought")
    # Drawdown / breakout
    bo = row.get("breakout_20") or 0
    if bo > 0:
        score += 0.0020; reasons.append("breakout above 20d high")
    if (row.get("dd_from_high_60") or 0) < -0.20:
        score -= 0.0020; reasons.append("deep drawdown from 60d high")
    # Volatility penalty
    v = row.get("vol_20d") or 0
    if v > 0.60:
        score -= 0.0015; reasons.append(f"high realized vol {v:.2f}")
    return score, reasons


def _confidence(score: float, all_scores: np.ndarray | None) -> float:
    if all_scores is None or len(all_scores) < 30:
        return float(min(1.0, abs(score) / 0.05))
    sd = float(np.std(all_scores))
    if sd <= 1e-9:
        return 0.5
    z = abs(score) / sd
    return float(min(1.0, z / 3.0))


def _stance(
    score: float,
    confidence: float,
    feature_row: dict,
    cfg: Config,
) -> tuple[str, list[str]]:
    d = cfg.get("decision", {})
    buy_thr = d.get("buy_threshold", 0.005)
    sell_thr = d.get("sell_threshold", -0.005)
    rsi_ob = d.get("rsi_overbought", 75.0)
    rsi_os = d.get("rsi_oversold", 25.0)
    min_adv = d.get("min_avg_dollar_volume_20", 5_000_000)

    reasons: list[str] = []
    if (feature_row.get("avg_dollar_volume_20") or 0) < min_adv:
        reasons.append(f"liquidity below floor (${min_adv:,.0f}/day) — forcing HOLD")
        return "HOLD", reasons

    rsi = feature_row.get("rsi_14")
    if score >= buy_thr:
        if rsi is not None and rsi > rsi_ob:
            reasons.append(f"score positive but RSI={rsi:.1f} overbought; downgrade to HOLD")
            return "HOLD", reasons
        reasons.append(f"score {score:+.4f} ≥ {buy_thr:+.4f} buy threshold")
        return "BUY", reasons
    if score <= sell_thr:
        if rsi is not None and rsi < rsi_os:
            reasons.append(f"score negative but RSI={rsi:.1f} oversold; downgrade to HOLD")
            return "HOLD", reasons
        reasons.append(f"score {score:+.4f} ≤ {sell_thr:+.4f} sell threshold")
        return "SELL", reasons
    reasons.append(f"score {score:+.4f} within neutral band")
    return "HOLD", reasons


def _playbook_overlay(
    ticker: str,
    stance: str,
    cfg: Config,
    last_price: float | None,
) -> tuple[str, list[str], dict]:
    """Apply the v2 decision-tree playbook on top of the model stance.

    Order of authority (PDF is explicit that §3 rules are definitive):
      1. A TRIGGERED standing rule forces its action (SELL/TRIM/REVIEW).
      2. A BUY must clear pre-trade compliance (never-buy, lockouts, caps,
         semi freeze, composite RED) or it is downgraded to HOLD.
    Fails soft: any error leaves the model stance untouched.
    """
    reasons: list[str] = []
    grounding: dict = {"available": False}
    try:
        from ..flags import get_flag_snapshot
        from ..playbook import (
            check_trade,
            evaluate_standing_rules,
            load_playbook,
            load_portfolio,
        )

        playbook = load_playbook(cfg)
        portfolio = load_portfolio(cfg)
        cache_min = float(cfg.get("playbook", {}).get("flag_cache_minutes", 60))
        snapshot = get_flag_snapshot(cfg, max_age_minutes=cache_min)

        prices = {ticker: last_price} if last_price else {}
        checks = [
            c for c in evaluate_standing_rules(playbook, portfolio, prices)
            if c.ticker == ticker
        ]
        triggered = [c for c in checks if c.status == "TRIGGERED"]
        near = [c for c in checks if c.status == "NEAR"]

        grounding = {
            "available": True,
            "flags": snapshot.summary_line(),
            "composite": snapshot.composite.color.value,
            "deployment_fraction": snapshot.composite.deployment_fraction,
            "semi_freeze": snapshot.composite.semi_freeze,
            "standing_rules": [c.to_dict() for c in checks],
            "held": portfolio.position(ticker) is not None,
            "never_buy": ticker in playbook.never_buy,
            "lockout": ticker in playbook.lockout_tickers,
        }

        if triggered:
            c = triggered[0]
            if c.action.startswith(("SELL", "TRIM")):
                reasons.append(
                    f"standing rule §3 [{c.kind}] TRIGGERED: {c.action} — {c.detail} (overrides model stance)"
                )
                stance = "SELL"
            else:
                reasons.append(f"standing rule §3 [{c.kind}] TRIGGERED: {c.action} — {c.detail}")
        for c in near:
            reasons.append(f"standing rule watch: {c.detail}")

        enforce = bool(cfg.get("playbook", {}).get("enforce_in_decisions", True))
        if stance == "BUY" and enforce:
            comp = check_trade(
                ticker, "BUY", 0.0, playbook, portfolio,
                snapshot=snapshot, prices=prices,
            )
            grounding["compliance"] = comp.to_dict()
            if not comp.allowed:
                reasons.append("playbook blocks BUY → downgraded to HOLD:")
                reasons.extend(f"  ✗ {v}" for v in comp.violations)
                stance = "HOLD"
            else:
                reasons.extend(f"compliance note: {w}" for w in comp.warnings)

    except Exception as e:
        logger.warning(f"playbook overlay skipped for {ticker}: {e}")
        grounding = {"available": False, "error": str(e)}
    return stance, reasons, grounding


def analyze_symbol(
    ticker: str,
    cfg: Config | None = None,
    write_report: bool = True,
) -> DecisionResult:
    cfg = cfg or get_config()
    ticker = ticker.upper()
    universe = [t.upper() for t in cfg["universe"]["tickers"]]

    if ticker not in universe:
        logger.warning(
            f"{ticker} not in configured universe ({len(universe)} symbols). "
            "Adding it to this run only."
        )

    ohlcv, features = _load_or_build_features(cfg)
    if ticker not in features["ticker"].unique().to_list():
        # Try to fetch this ticker on the fly so single-symbol queries work even
        # if the universe doesn't include it yet.
        from ..ingestion.market_data import fetch_ohlcv
        extra = fetch_ohlcv([ticker], start=cfg["data"]["start_date"], end=cfg["data"].get("end_date"))
        if extra.is_empty():
            raise ValueError(f"No data found for {ticker}.")
        ohlcv = pl.concat([ohlcv, extra], how="diagonal_relaxed").unique(subset=["date", "ticker"])
        from ..features.context import build_macro_inputs
        _mf, _ec, _earn = build_macro_inputs(cfg, tickers=[ticker], with_earnings=True)
        features = build_feature_matrix(
            ohlcv,
            economic_calendar=_ec,
            earnings_calendar=_earn,
            macro_features=_mf,
            benchmark=cfg["universe"]["benchmark"],
        )

    last_date = features["date"].max()
    row_df = features.filter((pl.col("ticker") == ticker) & (pl.col("date") == last_date))
    if row_df.is_empty():
        raise ValueError(f"No feature row for {ticker} on {last_date}.")
    row = row_df.to_dicts()[0]

    # ---- Score ----
    score: float | None = None
    score_source = "rules"
    feature_columns = None
    shap_summary = None
    rule_reasons: list[str] = []

    models_avail = list_models(registry=cfg.path("reports") / "models")
    if models_avail:
        try:
            # Prefer ensemble artifact over individual model
            preferred = best_ensemble_artifact(registry=cfg.path("reports") / "models")
            model_name = preferred if preferred else models_avail[-1]
            model, art = load_model(model_name, registry=cfg.path("reports") / "models")
            feature_columns = [c for c in art.feature_columns if c in features.columns]
            is_ensemble = art.metadata.get("model_type") == "ensemble"

            if is_ensemble:
                # EnsembleModel.predict() returns dict; use best variant
                X_today = features.filter(pl.col("date") == last_date)
                X_sub = X_today.drop_nulls(subset=feature_columns)
                X_arr = X_sub.select(feature_columns).to_numpy()
                import numpy as _np
                X_arr = X_arr.astype(_np.float64)
                all_preds = model.predict(X_arr)
                best_variant = art.metadata.get("best_variant", "ensemble_blend")
                score_arr = all_preds.get(best_variant, all_preds.get("ensemble_blend"))
                tickers_today = X_sub["ticker"].to_list()
                if ticker in tickers_today:
                    idx = tickers_today.index(ticker)
                    score = float(score_arr[idx])
                    score_source = f"ensemble:{best_variant}"
                    preds_today = X_sub.select(["date", "ticker"]).with_columns(
                        score=pl.Series(score_arr)
                    )
                    try:
                        lgbm_m = model._models.get("lgbm")
                        if lgbm_m is not None:
                            shap_summary = compute_shap_summary(lgbm_m, features.tail(20_000), feature_columns)
                    except Exception:
                        pass
            else:
                preds_today = predict_with_model(
                    model, features.filter(pl.col("date") == last_date), feature_columns
                )
                srow = preds_today.filter(pl.col("ticker") == ticker)
                if not srow.is_empty():
                    score = float(srow["score"][0])
                    score_source = "model"
                    shap_summary = compute_shap_summary(model, features.tail(20_000), feature_columns)
        except Exception as e:
            logger.warning(f"Model inference failed, falling back to rules: {e}")

    if score is None:
        score, rule_reasons = _rule_based_score(row)

    # 20d horizon: scale 5d forecast by sqrt(4) when model returns 5d, otherwise rule estimate
    forecast_5d = score
    forecast_20d = score * 4.0 * 0.65  # diminishing returns assumption

    # Confidence: distribution-aware if model or ensemble
    all_scores = None
    if score_source != "rules" and "preds_today" in dir():
        try:
            all_scores = preds_today["score"].to_numpy()
        except Exception:
            all_scores = None
    elif score_source == "model":
        try:
            preds_today = predict_with_model(
                model, features.filter(pl.col("date") == last_date), feature_columns
            )
            all_scores = preds_today["score"].to_numpy()
        except Exception:
            all_scores = None
    confidence = _confidence(score, all_scores)

    # ---- Stance ----
    stance, stance_reasons = _stance(score, confidence, row, cfg)

    # ---- Playbook overlay (flags + standing rules + compliance) ----
    last_price = row.get("adj_close") or row.get("close")
    stance, playbook_reasons, playbook_grounding = _playbook_overlay(
        ticker, stance, cfg, float(last_price) if last_price else None
    )
    stance_reasons = stance_reasons + playbook_reasons

    # ---- Probabilistic price bounds (lower / median / upper per horizon) ----
    bounds = None
    try:
        from .bounds import compute_bounds
        bounds = compute_bounds(
            cfg, ticker, features, ohlcv,
            float(last_price) if last_price else 0.0, float(forecast_5d),
        )
    except Exception as e:
        logger.warning(f"bounds computation failed for {ticker} (non-fatal): {e}")

    # ---- Per-ticker SHAP waterfall (backtrack what drives this call) ----
    shap_waterfall = None
    try:
        if (cfg.path("reports") / "models").exists():
            from ..monitoring.shap_viz import compute_shap_waterfall
            shap_waterfall = compute_shap_waterfall(
                cfg.path("reports") / "models", features, ticker, top_n=10
            )
    except Exception as e:
        logger.debug(f"shap waterfall failed for {ticker}: {e}")

    # ---- Groundings ----
    events = _load_events(cfg)
    apprehension_df = _load_apprehension(cfg)

    # RAG: relevance-ranked, point-in-time news context for this ticker
    relevant_news = []
    try:
        if events is not None and not events.is_empty():
            from ..ingestion.rag import retrieve_ticker_news
            relevant_news = retrieve_ticker_news(events, ticker, k=5)
    except Exception as e:
        logger.debug(f"news retrieval failed for {ticker}: {e}")

    groundings = {
        "technical": technical_grounding(features, ticker),
        "regime": regime_grounding(features, ticker),
        "cross_section": cross_section_grounding(features, ticker),
        "events": event_grounding(events, ticker),
        "relevant_news": relevant_news,
        "apprehension": _apprehension_grounding(apprehension_df, ticker),
        "model": model_grounding(score if score_source != "rules" else None, feature_columns, shap_summary),
        "playbook": playbook_grounding,
        "bounds": bounds or {},
        "shap_waterfall": shap_waterfall or {},
    }

    rationale = stance_reasons + (rule_reasons if score_source == "rules" else [])
    if score_source != "rules":
        rationale.append(
            f"{score_source} score {score:+.4f} (5d expected return); confidence {confidence:.2f}"
        )

    result = DecisionResult(
        ticker=ticker,
        as_of=str(last_date),
        stance=stance,
        confidence=confidence,
        forecast_5d=float(forecast_5d),
        forecast_20d=float(forecast_20d),
        score_source=score_source,
        rationale=rationale,
        groundings=groundings,
    )

    if write_report:
        md_path, json_path = write_decision_report(result, cfg.path("reports") / "decisions")
        result.report_path = str(md_path)
        result.json_path = str(json_path)
        result.decision_payload = asdict(result)

        # Re-write JSON now that report_path / json_path / decision_payload are populated
        import json as _json
        json_path.write_text(_json.dumps(asdict(result), indent=2, default=str))

        # ---- DeepSeek AI narration (auto-appended to the report) ----
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if api_key:
            logger.info(f"Requesting DeepSeek narration for {ticker}…")
            try:
                from .explain import explain_decision
                narration = explain_decision(asdict(result), api_key=api_key, model=DEEPSEEK_DEFAULT_MODEL)
                # Append narration section to the markdown file
                separator = "\n\n---\n\n## 🤖 AI Analysis (DeepSeek)\n\n"
                with open(md_path, "a") as f:
                    f.write(separator + narration + "\n")
                logger.info(f"DeepSeek narration appended to {md_path.name}")
            except Exception as e:
                logger.warning(f"DeepSeek narration failed (non-fatal): {e}")
        else:
            logger.debug("DEEPSEEK_API_KEY not set — skipping AI narration")
    else:
        result.decision_payload = asdict(result)

    return result


def analyze_all(
    cfg: Config | None = None,
    workers: int = 6,
    write_report: bool = True,
) -> list[DecisionResult]:
    """Run analyze_symbol over the configured universe (threaded + progress).

    Each call is dominated by network latency (option-implied vol + DeepSeek
    narration), so a small thread pool gives a big wall-clock win. ``workers`` is
    kept modest to stay under DeepSeek rate limits; set 1 for fully serial.
    """
    from ..utils import parallel_map

    cfg = cfg or get_config()
    tickers = list(cfg["universe"]["tickers"])

    def _one(t: str) -> DecisionResult | None:
        try:
            return analyze_symbol(t, cfg=cfg, write_report=write_report)
        except Exception as e:
            logger.warning(f"analyze_symbol({t}) failed: {e}")
            return None

    if workers and workers > 1:
        results = parallel_map(_one, tickers, workers=workers, description="analyze-all")
    else:
        from ..utils import track
        results = [_one(t) for t in track(tickers, "analyze-all")]
    return [r for r in results if r is not None]
