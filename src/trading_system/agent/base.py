"""V2 ReAct Agent core — Thought → Action → Observation loop.

A minimal ReAct implementation that keeps the dependency tree small
(no LangChain, no heavy frameworks).  The agent drives an LLMRouter
and calls registered Tool functions.

Protocol (follows the original ReAct paper, Yao et al. 2022):

    Thought: <reasoning step>
    Action: <tool_name>
    Action Input: <JSON or plain text>
    Observation: <tool output, injected by the agent>
    ... (repeats up to max_iterations)
    Thought: I now have enough information.
    FINISH: <final answer>

The agent returns an AgentResult with the full trace and final text.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from ..utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Tool:
    """A callable tool the agent can invoke."""
    name: str
    description: str
    func: Callable[..., Any]
    arg_schema: str = ""   # plain-text description of expected input

    def call(self, arg: str) -> str:
        """Invoke the tool with a string argument. Returns a string result."""
        try:
            # Try to parse as JSON first; fall back to raw string
            try:
                parsed = json.loads(arg)
            except json.JSONDecodeError:
                parsed = arg
            result = self.func(parsed)
            if isinstance(result, str):
                return result[:4000]
            return json.dumps(result, default=str)[:4000]
        except Exception as e:
            return f"Error: {e}"


@dataclass
class ToolResult:
    tool_name: str
    input_arg: str
    output: str
    error: bool = False


@dataclass
class AgentStep:
    thought: str
    action: str
    action_input: str
    observation: str


@dataclass
class AgentResult:
    task: str
    steps: list[AgentStep] = field(default_factory=list)
    final_answer: str = ""
    success: bool = False
    backend_used: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "steps": [
                {
                    "thought": s.thought,
                    "action": s.action,
                    "action_input": s.action_input,
                    "observation": s.observation,
                }
                for s in self.steps
            ],
            "final_answer": self.final_answer,
            "success": self.success,
            "backend_used": self.backend_used,
        }


# ---------------------------------------------------------------------------
# ReAct Agent
# ---------------------------------------------------------------------------

_FINISH_PATTERN = re.compile(r"FINISH[:\s]+(.+)", re.IGNORECASE | re.DOTALL)
_ACTION_PATTERN = re.compile(r"Action[:\s]+(\w+)", re.IGNORECASE)
_ACTION_INPUT_PATTERN = re.compile(r"Action Input[:\s]+(.*?)(?=\nThought|\nAction|\nObservation|\nFINISH|$)", re.IGNORECASE | re.DOTALL)
_THOUGHT_PATTERN = re.compile(r"Thought[:\s]+(.*?)(?=\nAction|\nFINISH|$)", re.IGNORECASE | re.DOTALL)


class ReActAgent:
    """Minimal ReAct loop: Thought → Action → Observation → … → FINISH.

    Parameters
    ----------
    llm_router:
        LLMRouter instance for generating text completions.
    tools:
        List of Tool instances the agent can call.
    system_prompt:
        Optional override for the system prompt.
    max_iterations:
        Safety cap on the number of thought/action cycles.
    """

    def __init__(
        self,
        llm_router: Any,  # LLMRouter from ingestion.llm_extractor
        tools: list[Tool],
        system_prompt: str | None = None,
        max_iterations: int = 8,
    ):
        self._router = llm_router
        self._tools: dict[str, Tool] = {t.name: t for t in tools}
        self._system_prompt = system_prompt or self._default_system_prompt()
        self._max_iterations = max_iterations

    def _default_system_prompt(self) -> str:
        tool_descs = "\n".join(
            f"  - {t.name}: {t.description}"
            + (f"\n    Input: {t.arg_schema}" if t.arg_schema else "")
            for t in self._tools.values()
        )
        return f"""You are an expert quantitative trading analyst with access to a set of tools.
Use the following format strictly:

Thought: <your reasoning about what to do next>
Action: <tool name — must be one of the available tools>
Action Input: <input to the tool (JSON object or plain string)>
Observation: <the tool result will be inserted here by the system>

Repeat Thought/Action/Action Input/Observation as needed.
When you have enough information to answer, write:
Thought: I now have enough information.
FINISH: <your complete final answer>

Available tools:
{tool_descs}

Rules:
- Always start with a Thought.
- Action must exactly match a tool name.
- FINISH must contain a complete, actionable answer.
- Do NOT fabricate tool outputs; wait for the Observation.
- Be concise in thoughts; be detailed in the FINISH answer.
"""

    def run(self, task: str) -> AgentResult:
        """Execute the ReAct loop for the given task."""
        result = AgentResult(task=task)
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": task},
        ]

        for iteration in range(self._max_iterations):
            # Get next LLM completion
            response = self._router.complete(
                messages,
                temperature=0.2,
                max_tokens=800,
            )
            if response is None:
                result.final_answer = "LLM unavailable — no backend responded."
                result.success = False
                result.backend_used = "none"
                return result

            result.backend_used = getattr(self._router, "_active_backend", "unknown")

            # Check for FINISH
            finish_match = _FINISH_PATTERN.search(response)
            if finish_match:
                result.final_answer = finish_match.group(1).strip()
                result.success = True
                # Capture final thought if present
                thought_match = _THOUGHT_PATTERN.search(response)
                if thought_match:
                    result.steps.append(AgentStep(
                        thought=thought_match.group(1).strip(),
                        action="FINISH",
                        action_input="",
                        observation=result.final_answer,
                    ))
                return result

            # Parse Thought / Action / Action Input
            thought = ""
            action = ""
            action_input = ""

            thought_match = _THOUGHT_PATTERN.search(response)
            if thought_match:
                thought = thought_match.group(1).strip()

            action_match = _ACTION_PATTERN.search(response)
            if action_match:
                action = action_match.group(1).strip()

            action_input_match = _ACTION_INPUT_PATTERN.search(response)
            if action_input_match:
                action_input = action_input_match.group(1).strip()

            # Execute the tool
            if action and action in self._tools:
                observation = self._tools[action].call(action_input)
            elif action:
                observation = f"Unknown tool: '{action}'. Available: {list(self._tools.keys())}"
            else:
                # No action found — maybe LLM went off-script; nudge it
                observation = "No valid Action found. Please follow the format: Thought / Action / Action Input."

            step = AgentStep(
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation,
            )
            result.steps.append(step)
            logger.debug(f"[Agent] Step {iteration + 1}: {action}({action_input[:60]}) → {observation[:80]}")

            # Append to message history
            messages.append({"role": "assistant", "content": response})
            messages.append({
                "role": "user",
                "content": f"Observation: {observation}",
            })

        # Max iterations reached
        result.final_answer = "Maximum iterations reached. Partial analysis:\n" + "\n".join(
            f"- {s.action}: {s.observation[:200]}" for s in result.steps if s.action
        )
        result.success = False
        return result
