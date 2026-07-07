from scripts.benchmark_latency import render, run_benchmark


def test_benchmark_runs_and_reports():
    stats = run_benchmark(iterations=100, warmup=10)
    assert stats["iterations"] == 100
    assert stats["throughput_rps"] > 0
    assert 0 < stats["p50_ms"] <= stats["p99_ms"] <= stats["max_ms"]
    assert stats["n_features"] == 9


def test_render_contains_table():
    stats = run_benchmark(iterations=50, warmup=5)
    text = render(stats)
    assert "throughput" in text
    assert "p99" in text
