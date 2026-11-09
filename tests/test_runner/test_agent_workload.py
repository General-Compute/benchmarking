import csv
import time
from collections.abc import AsyncIterator
from itertools import pairwise
from pathlib import Path

import pytest

from benchmarking.clients.base import BaseLLMClient, Prompt, StreamChunk
from benchmarking.config.models import BenchmarkConfig, GenerationParams, WorkloadConfig
from benchmarking.config.settings import Settings
from benchmarking.metrics.collector import MetricsCollector
from benchmarking.reporting.csv_reporter import CSVReporter
from benchmarking.runner.benchmark import BenchmarkRunner


class FakeClient(BaseLLMClient):
    """Streams a fixed tool call and records every request it receives."""

    def __init__(self, fail_on_call: int | None = None) -> None:
        super().__init__(api_key="k", model="m", service_name="Fake")
        self.requests: list[tuple[Prompt, int]] = []
        self.fail_on_call = fail_on_call

    async def stream_completion(
        self, prompt: Prompt, max_tokens: int, temperature: float, top_p: float
    ) -> AsyncIterator[StreamChunk]:
        self.requests.append((prompt, max_tokens))
        if self.fail_on_call == len(self.requests):
            raise ValueError("bad request")
        n_in = len(prompt) if isinstance(prompt, list) else 1
        for i, piece in enumerate(['{"tool": ', f'"read_file", "call": {len(self.requests)}}}']):
            yield StreamChunk(content=piece, timestamp=time.perf_counter(), is_first=i == 0)
        yield StreamChunk(
            content="",
            timestamp=time.perf_counter(),
            is_final=True,
            completion_tokens=12,
            prompt_tokens=100 * n_in,
        )

    async def close(self) -> None:
        pass


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


AGENT = WorkloadConfig(
    kind="agent", tokens=256, template="code", steps=4, step_tokens=64, output_tokens=32
)


def _runner() -> BenchmarkRunner:
    return BenchmarkRunner(Settings())


@pytest.mark.asyncio
async def test_agent_task_resends_growing_transcript() -> None:
    runner = _runner()
    client = FakeClient()
    tool_results = runner._tool_results(AGENT)

    results = await runner._run_agent_task(
        client,
        AGENT,
        "agent",
        "model-a",
        "[run 00000] system",
        tool_results,
        GenerationParams(max_tokens=1000),
        BenchmarkConfig(retry_attempts=1),
    )

    assert [r.step for r in results] == [1, 2, 3, 4]
    assert all(r.error is None for r in results)
    sent = [prompt for prompt, _ in client.requests]
    assert [len(p) for p in sent] == [2, 4, 6, 8]
    # Every step re-sends the previous transcript unchanged as its prefix.
    for prev, nxt in pairwise(sent):
        assert nxt[: len(prev)] == prev
    # The model's actual reply and the step's tool result are appended.
    assert isinstance(sent[1], list)
    assert sent[1][2] == {"role": "assistant", "content": '{"tool": "read_file", "call": 1}'}
    assert sent[1][3]["content"] == tool_results[0]
    # Per-step output cap comes from the workload, not generation_params.
    assert {max_tokens for _, max_tokens in client.requests} == {32}
    assert len(tool_results) == 3


@pytest.mark.asyncio
async def test_failed_step_ends_task() -> None:
    runner = _runner()
    client = FakeClient(fail_on_call=2)

    results = await runner._run_agent_task(
        client,
        AGENT,
        "agent",
        "model-a",
        "system",
        runner._tool_results(AGENT),
        GenerationParams(),
        BenchmarkConfig(retry_attempts=1),
    )

    assert [r.step for r in results] == [1, 2]
    assert results[1].error == "ValueError: bad request"
    assert len(client.requests) == 2


CONFIG = """
providers:
  fake:
    name: "Fake"
    base_url: "http://localhost:1/v1"
    api_key_env: "FAKE_API_KEY"
models:
  model-a:
    ids:
      fake: "model-a"
workloads:
  ctx_256:
    tokens: 256
    template: "qa"
  agent_small:
    kind: agent
    tokens: 256
    template: "code"
    steps: 3
    step_tokens: 64
    output_tokens: 32
generation_params:
  max_tokens: 1000
benchmark:
  warmup_iterations: 1
  request_delay_seconds: 0
  retry_attempts: 1
output:
  results_dir: "{results}"
"""


@pytest.mark.asyncio
async def test_run_benchmark_records_steps_and_task_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(CONFIG.replace("{results}", str(tmp_path / "results")))
    monkeypatch.setenv("FAKE_API_KEY", "k")
    monkeypatch.delenv("RESULTS_DIR", raising=False)
    settings = Settings(str(config))
    runner = BenchmarkRunner(settings)
    clients: list[FakeClient] = []

    def make_client(provider_name: str, model_id: str) -> FakeClient:
        clients.append(FakeClient())
        return clients[-1]

    monkeypatch.setattr(runner, "_make_client", make_client)

    session_id = await runner.run_benchmark(
        ["fake"], ["model-a"], ["ctx_256", "agent_small"], iterations=2, session_id="t1"
    )

    raw = read_csv(tmp_path / "results" / f"session_{session_id}_raw.csv")
    agent_rows = [r for r in raw if r["workload_type"] == "agent_small"]
    single_rows = [r for r in raw if r["workload_type"] == "ctx_256"]
    assert [(r["iteration"], r["step"]) for r in agent_rows] == [
        ("0", "1"),
        ("0", "2"),
        ("0", "3"),
        ("1", "1"),
        ("1", "2"),
        ("1", "3"),
    ]
    assert [r["step"] for r in single_rows] == ["", ""]
    # Agent warm-up sends the first step only; then 2 tasks x 3 steps.
    agent_client = clients[1]
    assert [len(p) for p, _ in agent_client.requests] == [2, 2, 4, 6, 2, 4, 6]
    # Iterations get different nonces in the system prompt.
    first_steps = [p for p, _ in agent_client.requests if len(p) == 2]
    assert len({p[0]["content"][:12] for p in first_steps if isinstance(p, list)}) == 3

    session = MetricsCollector(session_id, str(tmp_path / "results")).load_session()
    summary_path = CSVReporter(str(tmp_path / "results")).generate_summary_csv(session)
    rows = {(r["workload"], r["metric"]): r for r in read_csv(summary_path)}
    assert rows[("agent_small", "task_completion_pct")]["mean"] == "100.0"
    assert rows[("agent_small", "task_latency_ms")]["count"] == "2"
    assert ("agent_small", "ttft_ms_step_1") in rows
    assert ("agent_small", "ttft_ms_step_3") in rows
    assert float(rows[("agent_small", "prompt_tokens_step_3")]["mean"]) == 600.0
    assert ("ctx_256", "task_latency_ms") not in rows
