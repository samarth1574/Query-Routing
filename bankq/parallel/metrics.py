"""Performance metrics used in the parallel benchmarks.

speedup     S(p) = T(1) / T(p)
efficiency  E(p) = S(p) / p
Karp–Flatt  e(p) = (1/S(p) - 1/p) / (1 - 1/p)   experimentally determined serial fraction
Amdahl      S_max(p) = 1 / (f + (1 - f) / p)     for a serial fraction f
"""

from __future__ import annotations

import math


def speedup(t1: float, tp: float) -> float:
    return t1 / tp if tp > 0 else math.inf


def efficiency(t1: float, tp: float, p: int) -> float:
    return speedup(t1, tp) / p if p > 0 else 0.0


def karp_flatt(s: float, p: int) -> float | None:
    if p <= 1 or s <= 0:
        return None
    return (1 / s - 1 / p) / (1 - 1 / p)


def amdahl(serial_fraction: float, p: int) -> float:
    return 1 / (serial_fraction + (1 - serial_fraction) / p)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] if lo == hi else v[lo] + (v[hi] - v[lo]) * (k - lo)


def summarize(latencies_ms: list[float], wall_s: float) -> dict:
    n = len(latencies_ms)
    return {"n": n, "wall_s": round(wall_s, 4), "throughput_qps": round(n / wall_s, 2) if wall_s else None,
            "p50_ms": round(percentile(latencies_ms, 0.5) or 0, 2), "p95_ms": round(percentile(latencies_ms, 0.95) or 0, 2),
            "p99_ms": round(percentile(latencies_ms, 0.99) or 0, 2)}
