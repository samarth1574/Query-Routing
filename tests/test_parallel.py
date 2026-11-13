"""Parallel execution: correctness (same answers, same order) and the expected speedups."""

import pytest

from bankq.agents.orchestrator import ResolutionOrchestrator
from bankq.agents.rag import KnowledgeBase
from bankq.agents.tools import CoreBankingSimulator, ToolRegistry
from bankq.config import ROOT
from bankq.parallel.executor import ParallelResolver, chunked
from bankq.parallel.metrics import amdahl, efficiency, karp_flatt, percentile, speedup

QUERIES = ["I don't recognise a payment of £45 at Amazon", "my card is not working", "I forgot my passcode",
           "Am I eligible for a car loan of £15,000 over 4 years?", "my transfer is still pending",
           "I lost my card", "what documents do I need to verify my identity", "when is my loan payment due"] * 3


@pytest.fixture(scope="module")
def slow_orch():
    """Every tool call waits 15 ms, like a network hop to a core-banking service."""
    from bankq.api.app import build_router
    kb = KnowledgeBase.load(ROOT / "knowledge_base")
    o = ResolutionOrchestrator(build_router(), kb, ToolRegistry(CoreBankingSimulator(latency_ms=15)), tool_workers=8)
    yield o
    o.close()


def test_metrics():
    assert speedup(10, 2.5) == 4
    assert efficiency(10, 2.5, 4) == 1
    assert karp_flatt(4, 4) == pytest.approx(0)
    assert karp_flatt(2, 4) == pytest.approx(1 / 3)
    assert amdahl(0.0, 8) == pytest.approx(8)
    assert amdahl(1.0, 8) == pytest.approx(1)
    assert percentile([1, 2, 3, 4], 0.5) == 2.5


def test_chunked_covers_every_item_once():
    items = list(range(23))
    parts = chunked(items, 5)
    assert [x for p in parts for x in p] == items and len(parts) == 5


def test_intra_query_parallel_tools_overlap(slow_orch):
    q = "I don't recognise a payment of £45 at Amazon"
    seq = slow_orch.resolve(q, "CUST-P1", parallel=False)
    par = slow_orch.resolve(q, "CUST-P1", parallel=True)
    # stage 2 runs block_card ∥ create_dispute: wall time ≈ one call, not two
    assert par["actions"][1]["wall_ms"] < 0.75 * seq["actions"][1]["wall_ms"]
    assert par["status"] == seq["status"] and par["agent"] == seq["agent"]


def test_batch_results_keep_input_order_and_match_sequential(slow_orch):
    refs = [f"CUST-{i:03d}" for i in range(len(QUERIES))]
    seq = slow_orch.resolve_many(QUERIES, refs, workers=1)
    par = slow_orch.resolve_many(QUERIES, refs, workers=6)
    assert [r["text"] for r in par] == [r["text"] for r in seq]
    assert [(r["agent"], r["category"]) for r in par] == [(r["agent"], r["category"]) for r in seq]


def test_threads_speed_up_io_bound_batches(slow_orch):
    resolver = ParallelResolver(orchestrator=slow_orch)
    one = resolver.run(QUERIES, strategy="sequential", intra_parallel=False)
    many = resolver.run(QUERIES, strategy="threads", workers=8, intra_parallel=True)
    assert one.summary()["n"] == many.summary()["n"] == len(QUERIES)
    assert speedup(one.wall_s, many.wall_s) > 2.5


def test_process_pool_runs_on_several_workers():
    resolver = ParallelResolver(tool_latency_ms=0)
    try:
        run = resolver.run(QUERIES, strategy="processes", workers=2, chunk_size=4)
    finally:
        resolver.shutdown()
    assert len(run.results) == len(QUERIES)
    assert len(run.worker_ids) == 2
    assert {r["status"] for r in run.results} <= {"resolved", "escalated"}
