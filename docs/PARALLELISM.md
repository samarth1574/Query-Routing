# Parallel and distributed execution

This is the PDC part of the project: where the agentic pipeline exposes parallelism, how it is
exploited, and what the measurements show.

## 1. Where the parallelism is

| Level | Kind | Where | Mechanism |
|---|---|---|---|
| Inside one request | **Task parallelism** | RAG retrieval runs alongside the first tool stage; independent tool calls in a stage run together (e.g. `block_card` ∥ `create_dispute`) | Shared `ThreadPoolExecutor` in `ResolutionOrchestrator` |
| Across a batch — classification | **Data parallelism (SIMD-style)** | The whole batch is vectorised once by TF-IDF and scored with one sparse matrix product | `Router.route_many`, `KnowledgeBase.search_many` |
| Across a batch — agent work | **Data parallelism (MIMD)** | Each query's agent run is independent | Thread pool (`resolve_many`, `/resolve/batch`) or process pool (`bankq/parallel/executor.py`) |
| Across machines | **Distributed, shared-nothing** | Each process loads its own model, knowledge base and bank client — the same unit as a container replica | `ProcessPoolExecutor` with an initializer; Docker image runs one worker per container |

An agent's plan is a small **task graph**: stages are executed in order (a stage may depend on the
previous one — you must find the transaction before you dispute it), calls within a stage have no
dependencies. The critical path is the sum of the slowest call in each stage, not the sum of all calls.

```
stage 1:  verify_transaction ─────────┐            RAG search ───┐
stage 2:          block_card ─────────┼─ join ─→ conclude ─→ escalation check ─→ reply
                  create_dispute ─────┘
```

## 2. Threads or processes?

CPython's Global Interpreter Lock lets only one thread run Python bytecode at a time.

- **Tool calls are I/O-bound** (in production they are network calls to core-banking services). A thread
  waiting on I/O releases the GIL, so threads overlap the waits. Threads are cheap and share the model.
- **Classification and retrieval are CPU-bound.** Threads give no speedup and add contention; separate
  processes do, at the cost of loading the model once per process and pickling results.

## 3. Results

Hardware: 8-CPU laptop. Queries: real Banking77 test sentences, 50 simulated customers.
Reproduce with `make bench` and `make bench-cpu`; raw numbers are in `docs/benchmarks/`.

Speedup S = T(sequential, no intra-query parallelism) / T(run); efficiency E = S / workers;
Karp–Flatt e = (1/S − 1/p)/(1 − 1/p), the experimentally measured serial fraction.

### I/O-bound: 200 queries, 20 ms per tool call

| Run | Workers | Wall (s) | Throughput (q/s) | p95 (ms) | Speedup | Efficiency |
|---|---|---|---|---|---|---|
| Sequential, tools one after another | 1 | 7.33 | 27.3 | 77.2 | 1.00× | 1.00 |
| Sequential, parallel tools + RAG inside each query | 1 | 6.29 | 31.8 | 53.0 | 1.17× | 1.17 |
| Threads × 2 (+ intra-query) | 2 | 3.10 | 64.6 | 50.5 | 2.37× | 1.18 |
| Threads × 4 (+ intra-query) | 4 | 1.51 | 132.3 | 49.6 | 4.85× | 1.21 |
| Threads × 8 (+ intra-query) | 8 | 0.80 | 250.8 | 51.1 | 9.19× | 1.15 |
| Processes × 2 (+ intra-query) | 2 | 2.81 | 71.2 | 47.9 | 2.61× | 1.30 |
| Processes × 4 (+ intra-query) | 4 | 1.54 | 130.2 | 53.1 | 4.77× | 1.19 |
| Processes × 8 (+ intra-query) | 8 | 0.81 | 248.0 | 47.9 | 9.09× | 1.14 |

- Intra-query parallelism alone cuts **p95 latency by 31%** (77 → 53 ms): the slowest requests are the
  multi-tool ones (fraud), and those benefit most.
- Inter-query threads scale almost linearly. Efficiency is above 1 because the two levels compose:
  every run except the baseline also overlaps tool calls inside each query.
- Threads and processes perform the same here, because the work is waiting, not computing. Threads win
  on memory (one model in memory instead of eight).

### CPU-bound: 800 queries, 0 ms tool latency

| Run | Workers | Wall (s) | Throughput (q/s) | p95 (ms) | Speedup | Efficiency | Karp–Flatt e |
|---|---|---|---|---|---|---|---|
| Sequential | 1 | 1.46 | 548 | 2.1 | 1.00× | 1.00 | – |
| Threads × 2 | 2 | 1.62 | 495 | 6.2 | 0.90× | 0.45 | 1.22 |
| Threads × 4 | 4 | 1.83 | 438 | 18.1 | 0.80× | 0.20 | 1.33 |
| Threads × 8 | 8 | 1.60 | 501 | 34.7 | 0.91× | 0.11 | 1.11 |
| Processes × 2 | 2 | 0.89 | 904 | 2.4 | 1.65× | 0.82 | 0.21 |
| **Processes × 4** | 4 | **0.43** | **1,850** | 2.7 | **3.37×** | 0.84 | 0.06 |
| Processes × 8 | 8 | 0.50 | 1,586 | 6.6 | 2.89× | 0.36 | 0.25 |

- **Threads give no speedup on CPU-bound work** (Karp–Flatt e > 1: worse than fully serial) and inflate
  tail latency, because threads fight over the GIL.
- **Processes scale to 3.4× on 4 workers** (84% efficiency). At 8 workers efficiency falls: the laptop has
  8 logical CPUs shared with the OS, the parent process and pickling, and numpy/BLAS threads inside
  each worker compete for the same cores. The rising Karp–Flatt value (0.06 → 0.25) shows this is
  overhead growing with p, not a fixed serial fraction.

## 4. Recommendations

1. **Serving (I/O-bound, real tool latency):** keep intra-query parallel tool calls on, and run one
   uvicorn worker per container with a thread pool of ~8; scale out with more containers.
2. **Offline/bulk processing (CPU-bound):** use the process strategy with workers ≈ physical cores (4 here).
3. **Batch API:** classification is already vectorised; the remaining per-query agent work is what the
   `workers` parameter parallelises.
4. Move the in-process rate limiter and the simulated bank's state to Redis / real services before running
   several replicas, so the replicas stay shared-nothing.

## 5. Correctness under concurrency

- The simulated bank guards customer state and the dispute counter with a re-entrant lock.
- Batch results are returned **in input order** (`ThreadPoolExecutor.map` / ordered chunks).
- Tests (`tests/test_parallel.py`) check that parallel and sequential runs give the same agent and
  category for every query, that stage-level parallelism actually overlaps tool calls, that threads
  speed up an I/O-bound batch by more than 2.5×, and that the process pool really uses several workers.
