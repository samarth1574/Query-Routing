"""Agent building blocks: context, tool calls, outcomes and the BaseAgent contract.

An agent works in *stages*. Each stage is a list of tool calls that do not depend on each
other, so the executor can run them in parallel; the next stage is planned only after the
previous one finishes (e.g. verify a transaction first, then block the card and open a
dispute at the same time). This is a small task graph, and it is the unit of parallelism
the orchestrator exploits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .tools import ToolResult


@dataclass
class AgentContext:
    text: str  # PII-masked customer message
    customer_ref: str | None
    decision: dict  # output of the NLP router (category, intent, confidence, rules...)
    entities: dict = field(default_factory=dict)

    @property
    def intent(self) -> str:
        return (self.decision.get("intent") or {}).get("intent") or ""

    @property
    def category(self) -> str:
        return self.decision["category"]


@dataclass(frozen=True)
class ToolCall:
    tool: str
    args: dict = field(default_factory=dict)
    why: str = ""

    def __hash__(self):
        return hash((self.tool, tuple(sorted(self.args.items()))))


@dataclass
class AgentOutcome:
    agent: str
    resolved: bool
    summary: str
    facts: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    escalation_reasons: list[str] = field(default_factory=list)
    risk: str = "low"  # low | medium | high


class BaseAgent:
    name = "base"
    category = ""
    description = ""

    def plan(self, ctx: AgentContext, results: dict[str, ToolResult], stage: int) -> list[ToolCall]:
        """Return the tool calls for this stage (empty list = done). `results` holds every earlier result."""
        return []

    def retrieval_query(self, ctx: AgentContext) -> str:
        intent = ctx.intent.replace("_", " ")
        return f"{ctx.text} {intent}"

    def conclude(self, ctx: AgentContext, results: dict[str, ToolResult], passages: list[dict]) -> AgentOutcome:
        raise NotImplementedError

    # helpers shared by the specialists
    @staticmethod
    def ok(results: dict[str, ToolResult], tool: str) -> dict | None:
        r = results.get(tool)
        return r.data if r and r.ok else None

    @staticmethod
    def policy_facts(passages: list[dict], n: int = 2, sentences: int = 2) -> list[str]:
        facts = []
        for p in passages[:n]:
            parts = [s.strip() for s in p["text"].split(". ") if s.strip()]
            snippet = ". ".join(parts[:sentences]).rstrip(".") + "."
            facts.append(f"{snippet} [{p['source']} — {p['heading']}]")
        return facts

    def describe(self) -> dict:
        return {"name": self.name, "category": self.category, "description": self.description}
