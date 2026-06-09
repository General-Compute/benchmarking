from pathlib import Path

from benchmarking.clients.base import BenchmarkResult
from benchmarking.metrics.collector import MetricsCollector


def test_metrics_collector_round_trips_result(tmp_path: Path) -> None:
    collector = MetricsCollector("unit", str(tmp_path))
    result = BenchmarkResult(
        ttft=0.25,
        end_to_end_latency=1.5,
        tokens_generated=100,
        prompt_tokens=25,
        tokens_per_sec=80.0,
        inter_token_latency_ms=12.5,
        request_timestamp=1_700_000_000.0,
        attempts=2,
        service_name="Provider A",
        model_name="model-a",
        workload_type="ctx_256",
    )

    collector.save_result(result, iteration=3)
    session = collector.load_session()

    assert session.session_id == "unit"
    assert session.services == ["Provider A"]
    assert session.models == ["model-a"]
    assert session.workloads == ["ctx_256"]
    assert session.iterations_per_workload == 4
    assert len(session.results) == 1
    assert session.results[0].ttft_ms == 250.0
    assert session.results[0].end_to_end_latency_ms == 1500.0
    assert session.results[0].attempts == 2
