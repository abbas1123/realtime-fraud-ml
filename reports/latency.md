# Scoring latency benchmark

Committed XGBoost model, 9 features, 5000 single-transaction requests through `Scorer.score` (probability + per-feature contributions), single thread.

| metric | value |
|---|---|
| mean | 2.507 ms |
| p50 | 2.441 ms |
| p90 | 2.847 ms |
| p99 | 3.359 ms |
| max | 40.661 ms |
| throughput | 399 req/s (1 thread) |

Reproduce with `python scripts/benchmark_latency.py`. Numbers are from a developer CPU and will vary by machine; treat them as a relative guide.
