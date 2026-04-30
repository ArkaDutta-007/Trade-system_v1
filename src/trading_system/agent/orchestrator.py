"""V2: TradingAgentOrchestrator — high-level agent workflows.

Provides three main workflows:
  1. run_ticker_analysis(ticker)  — deep-dive on a single stock
  2. run_portfolio_review()       — review all held positions
  3. run_daily_briefing()         — morning market briefing
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from .base import ReActAgent, AgentResult
from .tools import DEFAULT_TOOLS
from .prompts import ticker_analysis_prompt, portfolio_review_prompt, daily_briefing_prompt
from ..utils import get_logger

logger = get_logger(__name__)


class TradingAgentOrchestrator:
    """Orchestrates LLM-driven analysis workflows using the ReAct agent.

    Parameters
    ----------
    cfg:
        Trading system config (from get_config()).
    llm_router:
        LLMRouter instance for DeepSeek + Ollama routing.
    max_iterations:
        Max ReAct iterations per agent run.
    """

    def __init__(
        self,
        cfg: Any,
        llm_router: Any | None = None,
        max_iterations: int = 8,
    ):
        self._cfg = cfg

        if llm_router is None:
            from ..ingestion.llm_extractor import LLMRouter
            llm_router = LLMRouter()
        self._router = llm_router
        self._max_iterations = max_iterations

    def _make_agent(self) -> ReActAgent:
        return ReActAgent(
            llm_router=self._router,
            tools=DEFAULT_TOOLS,
            max_iterations=self._max_iterations,
        )

    def run_ticker_analysis(self, ticker: str) -> AgentResult:
        """Run a full ReAct analysis for a single ticker.

        Gathers news, model score, SHAP, apprehension, live price,
        and economic calendar, then synthesizes a BUY/HOLD/SELL recommendation.
        """
        ticker = ticker.upper()

        # Build context from current portfolio state
        context: dict[str, Any] = {}
        try:
            broker_path = self._cfg.path("reports") / "paper_broker.json"
            if broker_path.exists():
                broker = json.loads(broker_path.read_text())
                holdings = broker.get("holdings", {})
                if ticker in holdings:
                    context["held_position"] = holdings[ticker]
                context["portfolio_cash"] = broker.get("cash", 0.0)
        except Exception:
            pass

        task = ticker_analysis_prompt(ticker, context)
        agent = self._make_agent()
        result = agent.run(task)
        result.task = f"ticker_analysis:{ticker}"
        logger.info(
            f"Agent analysis for {ticker}: {result.success}, "
            f"backend={result.backend_used}, steps={len(result.steps)}"
        )
        return result

    def run_portfolio_review(self) -> list[AgentResult]:
        """Review all current holdings and generate rebalancing recommendations.

        Runs one analysis per held position plus an overall portfolio synthesis.
        """
        broker_path = self._cfg.path("reports") / "paper_broker.json"
        if not broker_path.exists():
            logger.warning("No paper broker state found for portfolio review.")
            return []

        broker = json.loads(broker_path.read_text())
        holdings: dict[str, float] = broker.get("holdings", {})
        if not holdings:
            logger.info("Portfolio is empty — nothing to review.")
            return []

        # Load latest metrics for context
        metrics: dict[str, Any] = {}
        try:
            import polars as pl
            from ..backtesting import compute_metrics

            eq_log = self._cfg.path("data_gold") / "paper_equity_log.parquet"
            if eq_log.exists():
                eq_df = pl.read_parquet(eq_log)
                if "net_ret" in eq_df.columns:
                    metrics = compute_metrics(eq_df["net_ret"].to_numpy())
        except Exception:
            pass

        task = portfolio_review_prompt(holdings, metrics)
        agent = self._make_agent()
        result = agent.run(task)
        result.task = "portfolio_review"
        return [result]

    def run_daily_briefing(self) -> AgentResult:
        """Generate a daily market briefing with macro events and top candidates."""
        import polars as pl

        universe = self._cfg["universe"]["tickers"]
        date_str = str(date.today())

        task = daily_briefing_prompt(date_str, len(universe))
        agent = self._make_agent()
        result = agent.run(task)
        result.task = "daily_briefing"
        logger.info(f"Daily briefing: success={result.success}, backend={result.backend_used}")
        return result

    def save_result(self, result: AgentResult, output_dir: Path | None = None) -> Path:
        """Persist an AgentResult as JSON for dashboard/CLI display."""
        if output_dir is None:
            output_dir = self._cfg.path("reports") / "agent"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Derive filename from task
        task_slug = result.task.replace(":", "_").replace(" ", "_")[:60]
        dt = date.today().isoformat()
        fname = f"{task_slug}_{dt}.json"
        path = output_dir / fname

        path.write_text(json.dumps(result.to_dict(), indent=2))
        logger.info(f"Agent result saved to {path}")
        return path
