"""Playbook engine for the v2 NRA decision tree (configs/playbook_v2.yaml).

Modules:
  loader          — playbook + portfolio/watchlist state
  standing_rules  — §3 definitive sell/hold triggers vs live prices
  compliance      — pre-trade gate (never-buy, lockouts, caps, semi freeze, tax)
  cycles          — §4 cycle rule evaluator
  tax             — §8 NRA tax engine + 2026 shield tracker
  blotter         — append-only fill log feeding the realized-G/L ledger
  briefing        — the one-page morning artifact tying it all together
"""
from .blotter import blotter_realized, blotter_path, load_blotter, log_trade
from .compliance import ComplianceResult, check_trade
from .cycles import CycleRuleEval, OrderPlan, evaluate_cycles
from .loader import Holding, Playbook, Portfolio, load_playbook, load_portfolio
from .standing_rules import RuleCheck, evaluate_standing_rules
from .tax import ShieldStatus, sell_impact, shield_status
from .briefing import build_briefing, render_markdown, write_briefing

__all__ = [
    "Playbook",
    "Portfolio",
    "Holding",
    "load_playbook",
    "load_portfolio",
    "RuleCheck",
    "evaluate_standing_rules",
    "ComplianceResult",
    "check_trade",
    "CycleRuleEval",
    "OrderPlan",
    "evaluate_cycles",
    "ShieldStatus",
    "shield_status",
    "sell_impact",
    "log_trade",
    "load_blotter",
    "blotter_realized",
    "blotter_path",
    "build_briefing",
    "render_markdown",
    "write_briefing",
]
