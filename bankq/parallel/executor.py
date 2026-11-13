"""Batch execution strategies for the resolution pipeline.

| strategy  | unit of work           | how                                    | good for                         |
|-----------|------------------------|----------------------------------------|----------------------------------|
| sequential| one query at a time    | plain loop                             | baseline T(1)                    |
| threads   | one query per task     | ThreadPoolExecutor, shared orchestrator| I/O-bound tool calls (GIL freed) |
| processes | a chunk of queries     | ProcessPoolExecutor, one orchestrator  | CPU-bound work (classification,  |
|           | per task               | per worker process (initializer)       | retrieval) with no GIL contention|

Within a query, `intra_parallel=True` additionally runs independent tool calls and the RAG
lookup concurrently (task parallelism), so the two levels compose: inter-query × intra-query.

Process workers are "shared-nothing": each loads its own model, knowledge base and simulated
bank, which mirrors how the service scales horizontally across containers.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass

from .metrics import summarize

STRATEGIES = ("sequential", "threads", "processes")

_worker_orchestrator = None  # set in each worker process by _init_worker


def build_orchestrator(tool_latency_ms: float = 0.0, tool_workers: int = 8):
    from ..agents.orchestrator import ResolutionOrchestrator
    from ..config import ROOT, get_settings
    from ..ml.predictor import Predictor
    from ..routing.router import Router
    from ..routing.rules import RuleSet

    s = get_settings()
    router = Router(Predictor(), RuleSet.from_yaml(s.rules_path))
    return ResolutionOrchestrator.from_paths(router, ROOT / "knowledge_base", ROOT / "configs" / "agent_policy.yaml",
                                             tool_latency_ms=tool_latency_ms, tool_workers=tool_workers)


def _init_worker(tool_latency_ms: float, tool_workers: int):
    global _worker_orchestrator
    import logging
    logging.disable(logging.INFO)
    _worker_orchestrator = build_orchestrator(tool_latency_ms, tool_workers)


def _resolve_chunk(args):
    chunk, intra_parallel = args
    out = []
    for text, ref in chunk:
        t0 = time.perf_counter()
        r = _worker_orchestrator.resolve(text, ref, parallel=intra_parallel)
        out.append((_slim(r), (time.perf_counter() - t0) * 1000))
    return out, os.getpid()


def _slim(r: dict) -> dict:
    """Results that cross a process boundary are pickled; keep only what callers need."""
    return {k: r[k] for k in ("status", "agent", "category", "reply", "escalation", "timing", "risk")}


def chunked(items: list, size: int) -> list[list]:
    return [items[i:i + size] for i in range(0, len(items), size)]


@dataclass
class BatchRun:
    strategy: str
    workers: int
    intra_parallel: bool
    results: list[dict]
    latencies_ms: list[float]
    wall_s: float
    worker_ids: set

    def summary(self) -> dict:
        return {"strategy": self.strategy, "workers": self.workers, "intra_parallel": self.intra_parallel,
                "distinct_workers": len(self.worker_ids), **summarize(self.latencies_ms, self.wall_s),
                "escalated": sum(r["status"] == "escalated" for r in self.results)}


class ParallelResolver:
    def __init__(self, orchestrator=None, tool_latency_ms: float = 0.0, tool_workers: int = 8):
        self.tool_latency_ms = tool_latency_ms
        self.tool_workers = tool_workers
        self._orch = orchestrator
        self._procs: ProcessPoolExecutor | None = None
        self._procs_n = 0

    @property
    def orchestrator(self):
        if self._orch is None:
            self._orch = build_orchestrator(self.tool_latency_ms, self.tool_workers)
        return self._orch

    def _process_pool(self, workers: int) -> ProcessPoolExecutor:
        if self._procs is None or self._procs_n != workers:
            self.shutdown()
            self._procs = ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                                              initargs=(self.tool_latency_ms, self.tool_workers))
            self._procs_n = workers
            # Warm up: make every worker load its model before timing starts.
            list(self._procs.map(_resolve_chunk, [([("warm up", None)], False)] * workers))
        return self._procs

    def shutdown(self):
        if self._procs is not None:
            self._procs.shutdown(wait=True, cancel_futures=True)
            self._procs = None

    def run(self, texts: list[str], refs: list[str | None] | None = None, strategy: str = "threads",
            workers: int = 4, intra_parallel: bool = True, chunk_size: int | None = None) -> BatchRun:
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        refs = refs or [None] * len(texts)
        items = list(zip(texts, refs))

        if strategy == "processes":
            pool = self._process_pool(workers)
            size = chunk_size or max(1, len(items) // (workers * 4))
            t0 = time.perf_counter()
            parts = list(pool.map(_resolve_chunk, [(c, intra_parallel) for c in chunked(items, size)]))
            wall = time.perf_counter() - t0
            results = [r for part, _ in parts for r, _ in part]
            lat = [ms for part, _ in parts for _, ms in part]
            return BatchRun(strategy, workers, intra_parallel, results, lat, wall, {pid for _, pid in parts})

        orch = self.orchestrator

        def one(item):
            t = time.perf_counter()
            r = orch.resolve(item[0], item[1], parallel=intra_parallel)
            return _slim(r), (time.perf_counter() - t) * 1000, threading.get_ident()

        t0 = time.perf_counter()
        if strategy == "sequential" or workers <= 1:
            out = [one(i) for i in items]
            workers = 1
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="batch") as pool:
                out = list(pool.map(one, items))
        wall = time.perf_counter() - t0
        return BatchRun(strategy, workers, intra_parallel, [r for r, _, _ in out], [ms for _, ms, _ in out], wall,
                        {tid for _, _, tid in out})
