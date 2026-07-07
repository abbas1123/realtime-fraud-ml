"""Measure scoring latency and throughput of the committed fraud model.

Times the full ``Scorer.score`` path (probability + per-feature contributions),
which is what ``POST /score`` runs per request. Reports p50/p90/p99 latency and
single-thread throughput, and writes a markdown report.

    python scripts/benchmark_latency.py --iterations 5000
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fraud_ml.config import get_settings
from fraud_ml.serve import Scorer, ScoreRequest

REPORT = Path(__file__).resolve().parents[1] / "reports" / "latency.md"


def sample_request(rng: random.Random) -> ScoreRequest:
    return ScoreRequest(
        transaction_id=f"tx{rng.randrange(10**9)}",
        amount=round(rng.expovariate(1 / 80.0) + 1.0, 2),
        txn_count_1h=rng.randint(0, 12),
        txn_count_24h=rng.randint(0, 60),
        amount_zscore=rng.gauss(0, 1.5),
        seconds_since_last_txn=rng.expovariate(1 / 3600.0),
        is_new_merchant=rng.randint(0, 1),
        geo_distance_km=abs(rng.gauss(0, 40)),
        is_night=rng.randint(0, 1),
    )


def run_benchmark(iterations: int, warmup: int = 200, seed: int = 7) -> dict:
    rng = random.Random(seed)
    scorer = Scorer(get_settings())
    requests = [sample_request(rng) for _ in range(iterations + warmup)]

    for req in requests[:warmup]:
        scorer.score(req)

    latencies_ms: list[float] = []
    start = time.perf_counter()
    for req in requests[warmup:]:
        t0 = time.perf_counter()
        scorer.score(req)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    wall = time.perf_counter() - start

    latencies_ms.sort()

    def pct(p: float) -> float:
        idx = min(len(latencies_ms) - 1, int(p * len(latencies_ms)))
        return latencies_ms[idx]

    return {
        "iterations": iterations,
        "n_features": len(scorer.feature_names),
        "mean_ms": statistics.fmean(latencies_ms),
        "p50_ms": pct(0.50),
        "p90_ms": pct(0.90),
        "p99_ms": pct(0.99),
        "max_ms": latencies_ms[-1],
        "throughput_rps": iterations / wall,
    }


def render(stats: dict) -> str:
    return (
        "# Scoring latency benchmark\n\n"
        f"Committed XGBoost model, {stats['n_features']} features, "
        f"{stats['iterations']} single-transaction requests through `Scorer.score` "
        "(probability + per-feature contributions), single thread.\n\n"
        "| metric | value |\n"
        "|---|---|\n"
        f"| mean | {stats['mean_ms']:.3f} ms |\n"
        f"| p50 | {stats['p50_ms']:.3f} ms |\n"
        f"| p90 | {stats['p90_ms']:.3f} ms |\n"
        f"| p99 | {stats['p99_ms']:.3f} ms |\n"
        f"| max | {stats['max_ms']:.3f} ms |\n"
        f"| throughput | {stats['throughput_rps']:.0f} req/s (1 thread) |\n\n"
        "Reproduce with `python scripts/benchmark_latency.py`. Numbers are from a "
        "developer CPU and will vary by machine; treat them as a relative guide.\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()

    stats = run_benchmark(args.iterations)
    report = render(stats)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
