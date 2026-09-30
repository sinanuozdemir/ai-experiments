"""Difficulty-routing LangChain middleware for Fireworks serverless models.

A cheap classifier labels each task easy / medium / hard once per agent
invocation (``before_agent``); every model call in that run is then sent to
the tier's model (``wrap_model_call`` + ``request.override(model=...)``).

Cost per successful task follows the Fireworks x Arize definition: total spend
across ALL attempts (failures, classifier calls included) / successful runs.
"""

from __future__ import annotations

import ast
import operator
import os
import re
from typing import Any, Callable

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))  # repo-root .env

from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain_fireworks import ChatFireworks

COMPLETION_BUDGET_FLOOR = 10_000
MAX_TOKENS = 16_000
CLASSIFIER_MAX_TOKENS = 10_000
assert min(MAX_TOKENS, CLASSIFIER_MAX_TOKENS) >= COMPLETION_BUDGET_FLOOR

# USD per 1M tokens: (input, cached input, output). Source: docs/serverless/pricing.mdx, Standard tier.
PRICES = {
    "accounts/fireworks/models/gpt-oss-120b": (0.15, 0.015, 0.60),
    "accounts/fireworks/models/minimax-m3": (0.30, 0.06, 1.20),
    "accounts/fireworks/models/kimi-k3": (3.00, 0.30, 15.00),
}

CLASSIFIER_MODEL = "accounts/fireworks/models/gpt-oss-120b"
TIERS = {
    "easy": "accounts/fireworks/models/gpt-oss-120b",
    "medium": "accounts/fireworks/models/minimax-m3",
    "hard": "accounts/fireworks/models/kimi-k3",
}
assert all(m in PRICES for m in [CLASSIFIER_MODEL, *TIERS.values()])

CLASSIFIER_PROMPT = """You triage tasks for a model router. Decide how hard the task below is for a language model.

- easy: one or two straightforward steps, no traps.
- medium: several steps or some bookkeeping, but a standard approach works.
- hard: many dependent steps, tricky wording, or easy to get subtly wrong.

Think briefly, then end your reply with a final line of exactly one word: easy, medium, or hard.

Task:
{task}"""

TWO_TIERS = {
    "easy": "accounts/fireworks/models/gpt-oss-120b",
    "hard": "accounts/fireworks/models/kimi-k3",
}

TWO_TIER_PROMPT = """You triage tasks for a model router. A cheap model handles "easy" tasks; an expensive, much stronger model handles "hard" tasks. Pick "hard" only when a capable but cheaper model would plausibly get the task wrong.

- easy: routine multi-step arithmetic or word problems, standard textbook exercises.
- hard: competition-style problems (AMC/AIME/olympiad flavor), non-obvious insight, long chains of dependent reasoning, or subtle traps.

Do NOT attempt to solve the task. Judge only from what kind of task it is. Reply with exactly one word: easy or hard.

Task:
{task}"""


def chat(model: str, temperature: float = 0.6, max_tokens: int = MAX_TOKENS,
         timeout: float = 180, **model_kwargs: Any) -> ChatFireworks:
    return ChatFireworks(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=2,
        api_key=os.environ["FIREWORKS_API_KEY"],
        model_kwargs=model_kwargs,
    )


def message_cost(msg: AIMessage, model: str | None = None) -> float:
    model = model or (msg.response_metadata or {}).get("model_name")
    if model not in PRICES:
        raise KeyError(f"no price for model {model!r}; add it to PRICES")
    usage = msg.usage_metadata or {}
    inp, out = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
    cached = (usage.get("input_token_details") or {}).get("cache_read", 0) or 0
    p_in, p_cached, p_out = PRICES[model]
    return ((inp - cached) * p_in + cached * p_cached + out * p_out) / 1e6


def parse_label(text: str, labels: tuple[str, ...] = ("easy", "medium", "hard")) -> str | None:
    words = re.findall(r"\b(" + "|".join(labels) + r")\b", text.lower())
    return words[-1] if words else None


class RouterState(AgentState):
    difficulty: str
    routed_model: str
    router_cost_usd: float


class DifficultyRouterMiddleware(AgentMiddleware):
    """Classify the latest user turn once per invocation, then route every model call to the tier's model.

    With a checkpointer, each user turn is a new invocation, so a conversation is re-routed turn by turn.
    """

    state_schema = RouterState

    def __init__(self, classifier_model: str = CLASSIFIER_MODEL, tiers: dict[str, str] = TIERS,
                 default_tier: str = "medium", prompt: str = CLASSIFIER_PROMPT,
                 max_tokens: int = MAX_TOKENS, timeout: float = 180,
                 classifier_kwargs: dict[str, Any] | None = None,
                 classify_fn: Callable[[str], tuple[str | None, float]] | None = None):
        """``classify_fn(task) -> (label or None, cost_usd)`` replaces the LLM prompt classifier, e.g. a tuned router."""
        super().__init__()
        assert default_tier in tiers
        self.classify_fn = classify_fn
        self.classifier_model = classifier_model
        self.classifier = chat(classifier_model, temperature=0.0, max_tokens=CLASSIFIER_MAX_TOKENS,
                               **(classifier_kwargs or {}))
        self.tiers = tiers
        self.models = {k: chat(v, max_tokens=max_tokens, timeout=timeout) for k, v in tiers.items()}
        self.default_tier = default_tier
        self.prompt = prompt
        self.unparseable = 0

    def before_agent(self, state: RouterState, runtime: Any) -> dict[str, Any]:
        task = next(m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
        if self.classify_fn is not None:
            label, cost = self.classify_fn(task)
        else:
            reply = self.classifier.invoke([HumanMessage(self.prompt.format(task=task))])
            if not (reply.content or "").strip():
                raise RuntimeError(f"classifier returned empty content: {reply.response_metadata}")
            label, cost = parse_label(reply.content, tuple(self.tiers)), message_cost(reply, self.classifier_model)
        if label not in self.tiers:
            self.unparseable += 1
            label = self.default_tier
        return {"difficulty": label, "routed_model": self.tiers[label], "router_cost_usd": cost}

    def wrap_model_call(self, request: ModelRequest,
                        handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
        tier = request.state.get("difficulty", self.default_tier)
        return handler(request.override(model=self.models[tier]))


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv, ast.USub: operator.neg}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. '(12 * 4) / 3 + 7'."""
    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except Exception as e:  # returned to the model so it can fix the expression
        return f"error: {e}"


SYSTEM_PROMPT = ("Solve the math word problem. Use the calculator tool for arithmetic. "
                 "End your answer with a final line of the form '#### <number>'.")


def extract_answer(text: str) -> float | None:
    m = re.findall(r"####\s*\$?(-?[\d,]*\.?\d+)", text)
    nums = m or re.findall(r"-?[\d,]*\.?\d+", text)
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", ""))
    except ValueError:
        return None


def run_cost(result: dict[str, Any]) -> float:
    """Total $ of one agent run: every AI message plus the classifier call, if any."""
    total = result.get("router_cost_usd", 0.0)
    for m in result["messages"]:
        if isinstance(m, AIMessage):
            total += message_cost(m)
    return total
