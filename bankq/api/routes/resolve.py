"""Agentic resolution endpoints: classify → route to an agent → tools + RAG → reply or escalate."""

import time

from fastapi import APIRouter, Depends, HTTPException, Query

from ...config import get_settings
from ...db.repository import TicketRepository, ticket_to_dict
from ...routing.notify import get_notifier
from ...taxonomy import CATEGORIES, CATEGORY_ORDER
from ..deps import get_db, get_orchestrator, rate_limited, require_api_key
from ..observability import record_decision, record_resolution
from ..schemas import ResolveBatchRequest, ResolveRequest

router = APIRouter(prefix="/api/v1", tags=["agents"])


def _ticket_decision(result: dict) -> dict:
    """The ticket follows the agent that handled the case (it may differ from the classifier's pick)."""
    d = dict(result["decision"])
    if result["category"] != d["category"]:
        info = CATEGORIES[result["category"]]
        d.update(category=result["category"], label=info["label"], department=info["department"],
                 priority=info["priority"] if d["priority"] != "urgent" else "urgent", sla_hours=info["sla_hours"])
        d["rules_applied"] = d["rules_applied"] + [{"rule": "router_agent", "detail": "; ".join(result["routing"]["reasons"])}]
    if result["escalation"]["required"] and result["escalation"]["department"]:
        d["department"] = result["escalation"]["department"]
    return d


def _store(result: dict, repo: TicketRepository, channel: str, customer_ref: str | None, caller: str):
    d = _ticket_decision(result)
    t = repo.create(d, channel=channel, customer_ref=customer_ref, actor=caller)
    actor = f"agent:{result['agent']}"
    actions = [c["tool"] for s in result["actions"] for c in s["calls"] if c["ok"]]
    if actions:
        repo.add_note(t, "Actions taken: " + ", ".join(actions), actor)
    if result["status"] == "resolved":
        repo.add_note(t, result["reply"], actor)
        repo.set_status(t, "resolved", actor)
    else:
        repo.add_note(t, "Escalated to a human: " + "; ".join(result["escalation"]["reasons"]), actor)
    return t


@router.post("/resolve")
def resolve(req: ResolveRequest, caller: str = Depends(rate_limited), orch=Depends(get_orchestrator), db=Depends(get_db)):
    result = orch.resolve(req.text, req.customer_ref, parallel=req.parallel)
    record_decision(result["decision"])
    record_resolution(result)
    result["ticket_ref"] = None
    if req.create_ticket:
        t = _store(result, TicketRepository(db), req.channel, req.customer_ref, caller)
        db.commit()
        result["ticket_ref"] = t.ref
        if result["status"] == "escalated" and t.priority == "urgent":
            get_notifier().urgent_ticket(ticket_to_dict(t))
    result.pop("decision")
    return result


@router.post("/resolve/batch")
def resolve_batch(req: ResolveBatchRequest, _: str = Depends(rate_limited), orch=Depends(get_orchestrator)):
    s = get_settings()
    texts = [t for t in req.texts if t.strip()]
    if not texts:
        raise HTTPException(422, "All queries are empty.")
    if len(texts) > s.max_batch_size:
        raise HTTPException(413, f"Batch too large: {len(texts)} queries, the limit is {s.max_batch_size}.")
    refs = req.customer_refs if req.customer_refs and len(req.customer_refs) == len(req.texts) else None
    if refs is not None:
        refs = [r for t, r in zip(req.texts, refs) if t.strip()]
    workers = req.workers or s.batch_workers
    start = time.perf_counter()
    results = orch.resolve_many(texts, refs, workers=workers, parallel_tools=req.parallel_tools)
    wall = time.perf_counter() - start
    for r in results:
        record_resolution(r)
        r.pop("decision")
    by_agent = {}
    for r in results:
        by_agent[r["agent"]] = by_agent.get(r["agent"], 0) + 1
    return {
        "results": results,
        "summary": {
            "total": len(results),
            "resolved": sum(r["status"] == "resolved" for r in results),
            "escalated": sum(r["status"] == "escalated" for r in results),
            "by_category": {c: sum(r["category"] == c for r in results) for c in CATEGORY_ORDER},
            "by_agent": by_agent,
            "workers": workers,
            "wall_ms": round(wall * 1000, 1),
            "throughput_qps": round(len(results) / wall, 1) if wall else None,
            "sum_of_request_ms": round(sum(r["timing"]["total_ms"] for r in results), 1),
        },
    }


@router.get("/agents", dependencies=[Depends(require_api_key)])
def agents(orch=Depends(get_orchestrator)):
    return orch.describe()


@router.get("/knowledge/search", dependencies=[Depends(require_api_key)])
def knowledge_search(q: str = Query(..., min_length=2, max_length=500), category: str | None = None,
                     k: int = Query(default=3, ge=1, le=10), orch=Depends(get_orchestrator)):
    if category and category not in CATEGORIES:
        raise HTTPException(422, f"Unknown category {category!r}.")
    return {"query": q, "category": category, "results": orch.kb.search(q, category, k)}
