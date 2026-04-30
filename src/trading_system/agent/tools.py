"""V2: Tools the ReAct agent can call.

Each tool wraps existing trading system functions and returns a
human-readable string that the LLM can interpret.

Tools are intentionally side-effect-free and read-only (no trade execution).
The orchestrator handles trade execution separately after the agent decision.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import Tool
from ..utils import get_logger

logger = get_logger(__name__)


def _safe_json(obj: Any, max_chars: int = 3000) -> str:
    """Serialize to JSON string, truncating if needed."""
    s = json.dumps(obj, default=str, indent=2)
    if len(s) > max_chars:
        s = s[:max_chars] + "\n... (truncated)"
    return s


# ---------------------------------------------------------------------------
# Tool: Fetch News
# ---------------------------------------------------------------------------

def _fetch_news_func(arg: str | dict) -> str:
    """Fetch recent news for a ticker and return sentiment summary."""
    try:
        from ..ingestion.news_events import fetch_news
        ticker = arg if isinstance(arg, str) else arg.get("ticker", "")
        if not ticker:
            return "Error: provide {\"ticker\": \"AAPL\"}"
        events = fetch_news([ticker.upper()], days=7)
        if events.is_empty():
            return f"No recent news found for {ticker}."
        rows = events.to_dicts()[:10]
        lines = []
        for r in rows:
            dt = str(r.get("published_at", ""))[:10]
            headline = r.get("summary", "")[:120]
            sent = r.get("sentiment", 0.0)
            mag = r.get("magnitude", 0.0)
            lines.append(f"[{dt}] sentiment={sent:+.2f} magnitude={mag:.2f} | {headline}")
        mean_sent = sum(r.get("sentiment", 0.0) for r in rows) / max(len(rows), 1)
        return f"Recent news for {ticker} ({len(rows)} articles, mean sentiment={mean_sent:+.2f}):\n" + "\n".join(lines)
    except Exception as e:
        return f"News fetch error: {e}"


FetchNewsTool = Tool(
    name="fetch_news",
    description="Fetch the last 7 days of news headlines and sentiment for a ticker.",
    func=_fetch_news_func,
    arg_schema='{"ticker": "AAPL"}',
)


# ---------------------------------------------------------------------------
# Tool: Get Model Score
# ---------------------------------------------------------------------------

def _get_model_score_func(arg: str | dict) -> str:
    """Load latest ML model and get score for a ticker."""
    try:
        import polars as pl
        from ..config import get_config
        from ..models.model_registry import load_latest_model

        ticker = arg if isinstance(arg, str) else arg.get("ticker", "")
        if not ticker:
            return "Error: provide {\"ticker\": \"MSFT\"}"
        ticker = ticker.upper()

        cfg = get_config()
        feat_path = cfg.path("data_gold") / "features.parquet"
        if not feat_path.exists():
            return "Feature matrix not found. Run `ts features` first."

        features = pl.read_parquet(feat_path)
        last_date = features["date"].max()
        row = features.filter(
            (pl.col("ticker") == ticker) & (pl.col("date") == last_date)
        )
        if row.is_empty():
            return f"No feature data for {ticker} on {last_date}."

        reg_path = cfg.path("reports") / "models"
        ensemble, meta = load_latest_model(reg_path)
        feat_cols = meta.get("feature_columns", [])
        available = [c for c in feat_cols if c in row.columns]
        if not available:
            return f"Feature columns not available. Re-run `ts train`."

        import numpy as np
        X = row.select(available).to_numpy()
        score = float(ensemble.predict(X)[0])
        best_variant = meta.get("best_variant", "ensemble")
        return (
            f"Model score for {ticker} as of {last_date}: {score:+.4f}\n"
            f"Best variant: {best_variant}\n"
            f"Interpretation: {'+ve = bullish' if score > 0 else '-ve = bearish'}, "
            f"threshold BUY>0.005, SELL<-0.005"
        )
    except Exception as e:
        return f"Model score error: {e}"


GetModelScoreTool = Tool(
    name="get_model_score",
    description="Get the latest ML ensemble prediction score for a ticker.",
    func=_get_model_score_func,
    arg_schema='{"ticker": "MSFT"}',
)


# ---------------------------------------------------------------------------
# Tool: Get SHAP Features
# ---------------------------------------------------------------------------

def _get_shap_func(arg: str | dict) -> str:
    """Compute top SHAP features for a ticker."""
    try:
        import polars as pl
        from ..config import get_config
        from ..models.model_registry import load_latest_model
        from ..monitoring.shap_viz import compute_shap_waterfall

        ticker = arg if isinstance(arg, str) else arg.get("ticker", "")
        top_n = int(arg.get("top_n", 8)) if isinstance(arg, dict) else 8
        if not ticker:
            return "Error: provide {\"ticker\": \"AAPL\"}"
        ticker = ticker.upper()

        cfg = get_config()
        feat_path = cfg.path("data_gold") / "features.parquet"
        reg_path = cfg.path("reports") / "models"
        if not feat_path.exists():
            return "Feature matrix not found. Run `ts features` first."

        features = pl.read_parquet(feat_path)
        shap_data = compute_shap_waterfall(reg_path, features, ticker, top_n=top_n)
        if shap_data is None:
            return f"SHAP computation unavailable for {ticker}."

        lines = [f"SHAP waterfall for {ticker} (top {top_n} features):"]
        base = shap_data.get("base_value", 0.0)
        lines.append(f"  Base value (mean prediction): {base:+.4f}")
        names = shap_data.get("feature_names", [])
        vals = shap_data.get("shap_values", [])
        fvals = shap_data.get("feature_values", [])
        for name, sv, fv in zip(names[:top_n], vals[:top_n], fvals[:top_n]):
            direction = "↑" if sv > 0 else "↓"
            lines.append(f"  {direction} {name}={fv:.3f}  SHAP={sv:+.4f}")
        return "\n".join(lines)
    except Exception as e:
        return f"SHAP error: {e}"


GetSHAPTool = Tool(
    name="get_shap",
    description="Get SHAP feature importance waterfall for a ticker showing why the model made its prediction.",
    func=_get_shap_func,
    arg_schema='{"ticker": "AAPL", "top_n": 8}',
)


# ---------------------------------------------------------------------------
# Tool: Get Apprehension Score
# ---------------------------------------------------------------------------

def _get_apprehension_func(arg: str | dict) -> str:
    """Return apprehension score for a ticker."""
    try:
        import polars as pl
        from ..config import get_config

        ticker = arg if isinstance(arg, str) else arg.get("ticker", "")
        if not ticker:
            return "Error: provide {\"ticker\": \"AAPL\"}"
        ticker = ticker.upper()

        cfg = get_config()
        app_path = cfg.path("data_silver") / "apprehension_scores.parquet"
        if not app_path.exists():
            return f"No apprehension data found. Run `ts ingest` first."

        df = pl.read_parquet(app_path)
        last = df.filter(pl.col("ticker") == ticker).sort("date", descending=True).head(1)
        if last.is_empty():
            return f"No apprehension score found for {ticker}."
        row = last.to_dicts()[0]
        score = row.get("apprehension_score", 0.5)
        outlook = row.get("outlook", "stable")
        drivers = row.get("apprehension_drivers") or []
        date = str(row.get("date", ""))
        risk_label = "HIGH" if score > 0.65 else ("MODERATE" if score > 0.35 else "LOW")
        return (
            f"Apprehension for {ticker} as of {date}:\n"
            f"  Score: {score:.2f} ({risk_label} risk)\n"
            f"  Outlook: {outlook}\n"
            f"  Key drivers: {', '.join(drivers) if drivers else 'none identified'}"
        )
    except Exception as e:
        return f"Apprehension error: {e}"


GetApprehensionTool = Tool(
    name="get_apprehension",
    description="Get the LLM-based market apprehension (fear/risk) score for a ticker.",
    func=_get_apprehension_func,
    arg_schema='{"ticker": "AAPL"}',
)


# ---------------------------------------------------------------------------
# Tool: Live Price Snapshot
# ---------------------------------------------------------------------------

def _get_live_price_func(arg: str | dict) -> str:
    """Fetch current live price for a ticker."""
    try:
        from ..ingestion.realtime import live_price_snapshot

        ticker = arg if isinstance(arg, str) else arg.get("ticker", "")
        if not ticker:
            return "Error: provide {\"ticker\": \"AAPL\"}"
        ticker = ticker.upper()
        prices = live_price_snapshot([ticker])
        if not prices:
            return f"Could not fetch live price for {ticker}."
        price = prices.get(ticker)
        if price is None:
            return f"No live price available for {ticker}."
        return f"Live price for {ticker}: ${price:.2f}"
    except Exception as e:
        return f"Live price error: {e}"


GetLivePriceTool = Tool(
    name="get_live_price",
    description="Get the current live market price for a ticker.",
    func=_get_live_price_func,
    arg_schema='{"ticker": "AAPL"}',
)


# ---------------------------------------------------------------------------
# Tool: Economic Calendar
# ---------------------------------------------------------------------------

def _get_economic_calendar_func(arg: str | dict) -> str:
    """Return upcoming macro events from FRED calendar."""
    try:
        from ..ingestion.calendar_events import fetch_economic_calendar

        days = 14
        if isinstance(arg, dict):
            days = int(arg.get("lookahead_days", 14))
        elif isinstance(arg, str):
            try:
                days = int(arg)
            except ValueError:
                pass

        cal = fetch_economic_calendar(lookahead_days=days, lookback_days=7)
        if cal.is_empty():
            return f"No macro events found in the next {days} days."
        rows = cal.sort("date").to_dicts()
        lines = [f"Macro events (next {days} days + last 7 days):"]
        for r in rows:
            dt = str(r.get("date", ""))
            name = r.get("event_name", "")
            days_out = r.get("days_from_today", 0)
            sign = "in" if days_out >= 0 else "ago"
            lines.append(f"  [{dt}] {name} ({abs(days_out)}d {sign})")
        return "\n".join(lines)
    except Exception as e:
        return f"Economic calendar error: {e}"


GetEconomicCalendarTool = Tool(
    name="get_economic_calendar",
    description="Get upcoming macro events: FOMC meetings, CPI, NFP, PPI releases.",
    func=_get_economic_calendar_func,
    arg_schema='{"lookahead_days": 14}',
)


# ---------------------------------------------------------------------------
# Tool: Full Decision (wraps existing analyze_symbol)
# ---------------------------------------------------------------------------

def _get_decision_func(arg: str | dict) -> str:
    """Run full single-symbol analysis via existing analyze_symbol."""
    try:
        from ..config import get_config
        from ..decision import analyze_symbol

        ticker = arg if isinstance(arg, str) else arg.get("ticker", "")
        if not ticker:
            return "Error: provide {\"ticker\": \"AAPL\"}"
        ticker = ticker.upper()
        cfg = get_config()
        result = analyze_symbol(ticker, cfg, write_report=False)
        return (
            f"Decision for {ticker}:\n"
            f"  Stance: {result.stance}\n"
            f"  Confidence: {result.confidence:.0%}\n"
            f"  5d Forecast: {result.forecast_5d * 100:+.2f}%\n"
            f"  20d Forecast: {result.forecast_20d * 100:+.2f}%\n"
            f"  Rationale: {'; '.join(result.rationale[:3])}"
        )
    except Exception as e:
        return f"Decision analysis error: {e}"


GetDecisionTool = Tool(
    name="get_decision",
    description="Run the full quantitative decision engine for a ticker (technical + model + regime).",
    func=_get_decision_func,
    arg_schema='{"ticker": "AAPL"}',
)


# ---------------------------------------------------------------------------
# Default tool set for the trading agent
# ---------------------------------------------------------------------------

DEFAULT_TOOLS = [
    FetchNewsTool,
    GetModelScoreTool,
    GetSHAPTool,
    GetApprehensionTool,
    GetLivePriceTool,
    GetEconomicCalendarTool,
    GetDecisionTool,
]
