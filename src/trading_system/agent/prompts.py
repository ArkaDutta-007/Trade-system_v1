"""V2: Prompt templates for the trading ReAct agent."""
from __future__ import annotations

from typing import Any


SYSTEM_PROMPT = """\
You are an expert quantitative trading analyst and portfolio manager.
You have access to real-time market data, news sentiment, ML model scores,
SHAP feature explanations, and economic calendars.

Your goal is to provide precise, data-driven trading analysis and recommendations.
Always ground your conclusions in the evidence from tools.
Be conservative with risk — do not recommend BUY if apprehension is high without strong model support.
Use the format:

Thought: <your reasoning>
Action: <tool_name>
Action Input: <input (JSON or string)>
Observation: <system fills this>
...
Thought: I now have enough information to give a complete answer.
FINISH: <complete analysis with stance, confidence, and rationale>
"""


def ticker_analysis_prompt(ticker: str, context: dict[str, Any] | None = None) -> str:
    """Build the user prompt for single-ticker analysis."""
    ctx_lines = ""
    if context:
        if context.get("current_price"):
            ctx_lines += f"\nCurrent live price: ${context['current_price']:.2f}"
        if context.get("held_position"):
            qty = context["held_position"]
            ctx_lines += f"\nCurrent paper position: {qty:.2f} shares"
        if context.get("portfolio_cash"):
            ctx_lines += f"\nAvailable cash: ${context['portfolio_cash']:,.2f}"
    return f"""\
Analyze {ticker} for a trading decision.{ctx_lines}

Please:
1. Fetch recent news and compute sentiment
2. Get the current ML model score and SHAP feature breakdown
3. Check live price and apprehension score
4. Check the economic calendar for near-term macro events
5. Synthesize all evidence into a BUY / HOLD / SELL recommendation with confidence (0-1)

Format your FINISH as:
FINISH: {{
  "ticker": "{ticker}",
  "stance": "BUY|HOLD|SELL",
  "confidence": <0-1>,
  "forecast_5d_pct": <expected 5-day return %>,
  "rationale": "<concise explanation>",
  "key_risks": ["<risk1>", "<risk2>"],
  "key_catalysts": ["<catalyst1>"]
}}
"""


def portfolio_review_prompt(holdings: dict[str, float], metrics: dict[str, Any]) -> str:
    """Build the user prompt for full portfolio review."""
    holdings_str = "\n".join(f"  {t}: {qty:.2f} shares" for t, qty in holdings.items())
    cagr = metrics.get("CAGR", 0)
    sharpe = metrics.get("Sharpe", 0)
    max_dd = metrics.get("MaxDrawdown", 0)
    return f"""\
Review the current paper portfolio and recommend rebalancing actions.

Current Holdings:
{holdings_str}

Recent Performance:
  CAGR: {cagr:.2%}
  Sharpe: {sharpe:.2f}
  Max Drawdown: {max_dd:.2%}

For each position:
1. Check if the ML model still supports the position
2. Review recent news sentiment
3. Check apprehension scores

Provide:
- Positions to EXIT (weak model score + high apprehension)
- Positions to HOLD (neutral)  
- Positions to ADD (strong model score + positive sentiment)
- Any new entries from the broader universe

Format your FINISH with clear BUY/HOLD/SELL for each ticker and overall portfolio commentary.
"""


def daily_briefing_prompt(date_str: str, universe_size: int) -> str:
    """Build the user prompt for a daily market briefing."""
    return f"""\
Generate a daily trading briefing for {date_str}.

Universe: {universe_size} stocks

Please:
1. Check the economic calendar for today and the next 5 days (FOMC, CPI, NFP, earnings)
2. Get model scores for the top and bottom 5 movers in the universe
3. Check news sentiment for the top 3 positions

Provide a structured daily briefing with:
- Market regime summary (bull/bear, vol environment)
- Top macro events this week and their expected market impact
- Top 5 long candidates with brief rationale
- Top 3 risk factors to watch
- Overall portfolio stance recommendation

Keep the briefing concise and actionable.
"""
