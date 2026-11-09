from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmarking.clients.base import BenchmarkResult
from benchmarking.config.models import BenchmarkMetric, WorkloadConfig
from benchmarking.metrics.agent_tasks import expected_steps, group_agent_tasks
from benchmarking.metrics.collector import MetricsCollector

TS = datetime(2026, 11, 9, tzinfo=timezone.utc)


def metric(iteration: int, step: int | None, latency: float, error: str | None = None):
    return BenchmarkMetric(
        session_id="s",
        service_name="P",
        model_name="m",
        workload_type="agent" if step else "ctx_256",
        iteration=iteration,
        ttft_ms=latency / 4,
        end_to_end_latency_ms=latency,
        tokens_generated=10,
        prompt_tokens=100,
        timestamp=TS,
        error=error,
        step=step,
    )


def test_group_agent_tasks_sums_steps_and_flags_incomplete() -> None:
    results = [
        metric(0, 2, 200.0),
        metric(0, 1, 100.0),
        metric(1, 1, 100.0),
        metric(1, 2, 50.0, error="boom"),
        metric(2, 1, 100.0),  # never reached step 2
        metric(0, None, 999.0),  # single-request row is ignored
    ]

    grouped = group_agent_tasks(results)

    tasks = grouped[("m", "P", "agent")]
    assert list(grouped) == [("m", "P", "agent")]
    assert [t.iteration for t in tasks] == [0, 1, 2]
    assert expected_steps(tasks) == 2
    assert [t.is_complete(2) for t in tasks] == [True, False, False]
    assert [s.step for s in tasks[0].steps] == [1, 2]
    assert tasks[0].latency_ms == 300.0
    assert tasks[0].ttft_ms == 75.0
    step2 = tasks[0].step(2)
    assert step2 is not None and step2.end_to_end_latency_ms == 200.0
    assert tasks[2].step(2) is None


def test_collector_round_trips_step(tmp_path: Path) -> None:
    collector = MetricsCollector("agent", str(tmp_path))
    for step in (1, 2):
        collector.save_result(
            BenchmarkResult(
                ttft=0.1,
                end_to_end_latency=0.5,
                tokens_generated=10,
                prompt_tokens=100 * step,
                tokens_per_sec=25.0,
                inter_token_latency_ms=40.0,
                request_timestamp=1_700_000_000.0,
                service_name="P",
                model_name="m",
                workload_type="agent",
                step=step,
            ),
            iteration=0,
        )

    session = collector.load_session()

    assert [r.step for r in session.results] == [1, 2]


def test_collector_loads_raw_csv_written_before_step_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "session_old_raw.csv"
    csv_path.write_text(
        "session_id,service_name,model_name,workload_type,iteration,ttft_ms,"
        "end_to_end_latency_ms,tokens_generated,prompt_tokens,tokens_per_sec,"
        "inter_token_latency_ms,attempts,timestamp,error\n"
        "old,Provider A,model-a,ctx_256,0,250.0,1500.0,100,25,80.0,12.5,1,"
        "2026-06-09T18:15:05.451820+00:00,\n"
    )

    session = MetricsCollector("old", str(tmp_path)).load_session("old")

    assert session.results[0].step is None
    assert session.results[0].tokens_generated == 100


def test_workload_config_defaults_to_single_request() -> None:
    cfg = WorkloadConfig(tokens=256, template="qa")

    assert (cfg.kind, cfg.steps, cfg.step_tokens, cfg.output_tokens) == ("single", 1, 0, None)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"steps": 3},  # steps without kind: agent
        {"kind": "agent", "steps": 3, "step_tokens": 0},  # nothing to feed back
        {"kind": "agent", "steps": 0, "step_tokens": 64},
        {"kind": "batch"},
    ],
)
def test_workload_config_rejects_invalid_agent_settings(kwargs: dict) -> None:
    with pytest.raises(ValidationError):
        WorkloadConfig(tokens=256, template="qa", **kwargs)


def test_packaged_config_agent_workload() -> None:
    from benchmarking.config.settings import Settings

    cfg = Settings().get_workload_config("agent_8step")

    assert (cfg.kind, cfg.steps, cfg.step_tokens, cfg.output_tokens) == ("agent", 8, 1024, 128)
