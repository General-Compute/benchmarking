from datetime import datetime, timezone

from benchmarking.config.models import BenchmarkMetric, BenchmarkSession
from benchmarking.config.settings import Settings
from benchmarking.reporting.site_publisher import SitePublisher


def test_site_publisher_build_payload_aggregates_successes() -> None:
    session = BenchmarkSession(
        session_id="unit",
        start_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        end_time=datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc),
        services=["Provider A"],
        models=["model-a"],
        workloads=["ctx_256"],
        iterations_per_workload=2,
        results=[
            BenchmarkMetric(
                session_id="unit",
                service_name="Provider A",
                model_name="model-a",
                workload_type="ctx_256",
                iteration=0,
                ttft_ms=100.0,
                end_to_end_latency_ms=1000.0,
                tokens_generated=100,
                prompt_tokens=25,
                tokens_per_sec=111.0,
                inter_token_latency_ms=9.0,
                attempts=1,
                timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
            BenchmarkMetric(
                session_id="unit",
                service_name="Provider A",
                model_name="model-a",
                workload_type="ctx_256",
                iteration=1,
                ttft_ms=None,
                end_to_end_latency_ms=500.0,
                tokens_generated=0,
                prompt_tokens=25,
                attempts=1,
                timestamp=datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc),
                error="RateLimitError",
            ),
        ],
    )

    payload = SitePublisher(Settings(), "./results").build_payload(
        session, label="Unit run", published_at="2026-01-01T00:00:00+00:00"
    )

    assert payload["label"] == "Unit run"
    assert payload["rows"][0]["samples"] == 1
    assert payload["rows"][0]["error_rate_pct"] == 50.0
    assert payload["rows"][0]["metrics"]["ttft_ms"]["mean"] == 100.0
