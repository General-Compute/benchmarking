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


def test_load_session_tolerates_headerless_csv(tmp_path: Path) -> None:
    """A raw CSV left headerless by an interrupted/restarted run must still
    load (it previously crashed report generation with KeyError: 'timestamp')."""
    csv_path = tmp_path / "session_legacy_raw.csv"
    csv_path.write_text(
        "legacy,Provider A,model-a,ctx_256,0,250.0,1500.0,100,25,80.0,12.5,1,"
        "2026-06-09T18:15:05.451820+00:00,\n"
    )

    collector = MetricsCollector("legacy", str(tmp_path))
    session = collector.load_session("legacy")

    assert len(session.results) == 1
    assert session.results[0].service_name == "Provider A"
    assert session.results[0].tokens_generated == 100
    assert session.results[0].error is None


def test_initialize_csv_repairs_empty_file(tmp_path: Path) -> None:
    """A pre-existing 0-byte session file must get a header written, otherwise
    every subsequent append produces a headerless CSV."""
    csv_path = tmp_path / "session_empty_raw.csv"
    csv_path.write_text("")  # leftover 0-byte file

    MetricsCollector("empty", str(tmp_path))

    assert csv_path.read_text().startswith("session_id,")
