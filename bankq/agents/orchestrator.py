"""The resolution pipeline:

    Customer query → PII masking → Intent classification (NLP router) → Router Agent
        → Specialised Agent → [Tools ∥ RAG] → Action → Human-in-the-loop check → Response

Parallelism inside one request (task parallelism):
- policy retrieval (RAG) runs at the same time as the agent's first tool stage;
- tool calls within one stage are independent, so they run concurrently on a thread pool;
  the stage costs max(tool latencies) instead of their sum.

Parallelism across requests (data parallelism) lives in `resolve_many` and bankq/parallel.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..taxonomy import CATEGORIES, REVIEW_DEPARTMENT, REVIEW_SLA_HOURS
from .base import AgentContext, ToolCall
from .entities import extract_entities
from .escalation import EscalationPolicy
from .rag import KnowledgeBase
from .responder import ResponseGenerator, TemplateResponseGenerator
from .router_agent import RouterAgent
from .specialists import default_agents
from .tools import ToolRegistry, ToolResult

log = logging.getLogger("bankq.agents")

MAX_STAGES = 4


class ResolutionOrchestrator:
    def __init__(self, router, knowledge_base: KnowledgeBase, tools: ToolRegistry | None = None,
                 policy: EscalationPolicy | None = None, generator: ResponseGenerator | None = None,
                 tool_workers: int = 8):
        self.router = router
        self.kb = knowledge_base
        self.tools = tools or ToolRegistry()
        self.policy = policy or EscalationPolicy()
        self.generator = generator or TemplateResponseGenerator()
        self.agents = default_agents(self.policy.agent_settings)
        self.router_agent = RouterAgent(self.agents)
        self.tool_workers = tool_workers
        self._pool = ThreadPoolExecutor(max_workers=tool_workers, thread_name_prefix="agent-tool")

    @classmethod
    def from_paths(cls, router, kb_dir: Path, policy_path: Path | None, tool_latency_ms: float = 0.0,
                   tool_workers: int | None = None):
        from .tools import CoreBankingSimulator
        policy = EscalationPolicy.from_yaml(policy_path)
        workers = tool_workers or 8
        return cls(router, KnowledgeBase.load(kb_dir), ToolRegistry(CoreBankingSimulator(tool_latency_ms)), policy,
                   tool_workers=workers)

    def close(self):
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ---------- execution ----------

    def _run_stage(self, calls: list[ToolCall], parallel: bool) -> list[ToolResult]:
        if parallel and len(calls) > 1:
            futures = [self._pool.submit(self.tools.call, c.tool, **c.args) for c in calls]
            return [f.result() for f in futures]
        return [self.tools.call(c.tool, **c.args) for c in calls]

    def resolve(self, raw_text: str, customer_ref: str | None = None, parallel: bool = True) -> dict:
        start = time.perf_counter()
        decision = self.router.route(raw_text, explain=False)
        return self.resolve_decision(decision, customer_ref, parallel=parallel, started=start)

    def resolve_decision(self, decision: dict, customer_ref: str | None = None, parallel: bool = True,
                         started: float | None = None) -> dict:
        started = started or time.perf_counter()
        text = decision["text"]
        ctx = AgentContext(text, customer_ref, decision, extract_entities(text))
        agent, route_reasons = self.router_agent.select(decision)

        query = agent.retrieval_query(ctx)
        rag_future = self._pool.submit(self.kb.search, query, agent.category, 3) if parallel else None
        passages = None if parallel else self.kb.search(query, agent.category, 3)

        results: dict[str, ToolResult] = {}
        all_results: list[ToolResult] = []
        trace = []
        tool_ms_total = 0.0
        # When the classifier is unsure, the agent must not act on a guess: no tools, straight to a human.
        stages = 0 if decision.get("needs_review") and self.policy.escalate_on_review else MAX_STAGES
        for stage in range(stages):
            calls = agent.plan(ctx, results, stage)
            if not calls:
                break
            t0 = time.perf_counter()
            outs = self._run_stage(calls, parallel)
            wall = (time.perf_counter() - t0) * 1000
            for call, res in zip(calls, outs):
                results[call.tool] = res
                all_results.append(res)
                tool_ms_total += res.latency_ms
            trace.append({"stage": stage, "parallel": parallel and len(calls) > 1, "wall_ms": round(wall, 2),
                          "calls": [{"tool": c.tool, "args": {k: v for k, v in c.args.items() if v is not None},
                                     "why": c.why, **r.to_dict()} for c, r in zip(calls, outs)]})

        if rag_future is not None:
            passages = rag_future.result()

        outcome = agent.conclude(ctx, results, passages)
        reasons = self.policy.evaluate(decision, outcome, all_results, passages)
        escalated = bool(reasons)

        category = agent.category
        info = CATEGORIES[category]
        sla = decision["sla_hours"] if category == decision["category"] else info["sla_hours"]
        if escalated and decision.get("needs_review") and category == decision["category"]:
            sla = min(sla, REVIEW_SLA_HOURS)
        view = {**decision, "category": category, "sla_hours": sla}
        reply = self.generator.compose(view, outcome, escalated, reasons)

        total_ms = round((time.perf_counter() - started) * 1000, 2)
        status = "escalated" if escalated else "resolved"
        log.info("query resolved", extra={"agent": agent.name, "status": status, "tools": len(all_results),
                                          "latency_ms": total_ms})
        return {
            "status": status,
            "reply": reply,
            "agent": agent.name,
            "category": category,
            "routing": {"reasons": route_reasons, "classifier": {
                k: decision[k] for k in ("category", "label", "confidence", "intent", "needs_review", "priority",
                                         "department", "rules_applied", "probabilities", "model_version")}},
            "entities": ctx.entities,
            "actions": trace,
            "knowledge": passages,
            "escalation": {"required": escalated, "reasons": reasons,
                           "department": (REVIEW_DEPARTMENT if decision.get("needs_review") else info["department"])
                           if escalated else None, "sla_hours": sla if escalated else None},
            "risk": outcome.risk,
            "text": text,
            "pii_redacted": decision["pii_redacted"],
            "decision": decision,
            "timing": {"total_ms": total_ms, "tool_ms_sum": round(tool_ms_total, 2),
                       "tool_ms_wall": round(sum(s["wall_ms"] for s in trace), 2), "parallel": parallel},
        }

    def resolve_many(self, raw_texts: list[str], customer_refs: list[str | None] | None = None,
                     workers: int = 4, parallel_tools: bool = True) -> list[dict]:
        """Batch resolution.

        1. Classification is vectorised: one sparse matrix product for the whole batch.
        2. Agent work for each query is independent, so queries are spread over `workers` threads
           (tool calls are I/O-bound, so threads overlap their waits despite the GIL).
        Results come back in input order.
        """
        refs = customer_refs or [None] * len(raw_texts)
        decisions = self.router.route_many(raw_texts)
        if workers <= 1:
            return [self.resolve_decision(d, r, parallel=parallel_tools) for d, r in zip(decisions, refs)]
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="resolve") as pool:
            return list(pool.map(lambda dr: self.resolve_decision(dr[0], dr[1], parallel=parallel_tools),
                                 zip(decisions, refs)))

    def describe(self) -> dict:
        return {"agents": [a.describe() for a in self.agents.values()], "tools": self.tools.describe(),
                "knowledge_base": {"passages": len(self.kb.passages), "categories": self.kb.categories},
                "tool_workers": self.tool_workers}
