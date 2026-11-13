"""Benchmark the parallel strategies of the agentic resolution pipeline.

    python -m scripts.benchmark_parallel --n 200 --latency 20 --workers 1 2 4 8

Writes docs/benchmarks/parallel_results.json and prints a Markdown table with throughput,
latency percentiles, speedup, efficiency and the Karp–Flatt serial fraction.
"""

import argparse
import json
import logging
import random
from pathlib import Path

import pandas as pd

from bankq.config import ROOT, get_settings
from bankq.parallel.executor import ParallelResolver
from bankq.parallel.metrics import efficiency, karp_flatt, speedup


def load_queries(n: int, seed: int = 7) -> list[str]:
    df = pd.read_csv(get_settings().raw_data_dir / "test.csv")
    col = "text" if "text" in df.columns else df.columns[0]
    texts = df[col].astype(str).tolist()
    random.Random(seed).shuffle(texts)
    return texts[:n]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--latency", type=float, default=20.0, help="simulated latency per tool call (ms)")
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--strategies", nargs="+", default=["threads", "processes"])
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "benchmarks" / "parallel_results.json")
    args = ap.parse_args(argv)
    logging.disable(logging.INFO)

    texts = load_queries(args.n)
    refs = [f"CUST-{i % 50:03d}" for i in range(len(texts))]
    resolver = ParallelResolver(tool_latency_ms=args.latency)

    rows = []
    base = resolver.run(texts, refs, "sequential", 1, intra_parallel=False).summary()
    base["label"] = "sequential, no intra-query parallelism"
    rows.append(base)
    intra = resolver.run(texts, refs, "sequential", 1, intra_parallel=True).summary()
    intra["label"] = "sequential + parallel tools/RAG within a query"
    rows.append(intra)
    for strategy in args.strategies:
        for w in args.workers:
            if w == 1:
                continue
            r = resolver.run(texts, refs, strategy, w, intra_parallel=True).summary()
            r["label"] = f"{strategy} × {w} + intra-query"
            rows.append(r)
    resolver.shutdown()

    t1 = base["wall_s"]
    for r in rows:
        p = r["workers"]
        r["speedup"] = round(speedup(t1, r["wall_s"]), 2)
        r["efficiency"] = round(efficiency(t1, r["wall_s"], p), 2)
        kf = karp_flatt(r["speedup"], p)
        r["karp_flatt"] = round(kf, 3) if kf is not None else None

    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"queries": len(texts), "tool_latency_ms": args.latency, "cpu_count": __import__("os").cpu_count()}
    args.out.write_text(json.dumps({"meta": meta, "runs": rows}, indent=2))

    print(f"\n{len(texts)} queries, {args.latency:g} ms per tool call, {meta['cpu_count']} CPUs\n")
    print("| run | workers | wall (s) | throughput (q/s) | p50 ms | p95 ms | speedup | efficiency | Karp–Flatt e |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['label']} | {r['workers']} | {r['wall_s']} | {r['throughput_qps']} | {r['p50_ms']} | {r['p95_ms']} | "
              f"{r['speedup']}× | {r['efficiency']} | {r['karp_flatt'] if r['karp_flatt'] is not None else '–'} |")


if __name__ == "__main__":
    main()
