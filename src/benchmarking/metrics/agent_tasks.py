"""Task-level views of ``kind: agent`` workload results.

Raw results hold one row per request. For an agent workload a row is one step
of a task, and the number a user of an agent feels is the whole task: the sum
of every step's latency, since each step waits for the previous reply.
"""

from collections import defaultdict
from dataclasses import dataclass, field

from ..config.models import BenchmarkMetric


@dataclass
class AgentTask:
    """All recorded steps of one agent task (one provider, one iteration)."""

    model_name: str
    service_name: str
    workload_type: str
    iteration: int
    steps: list[BenchmarkMetric] = field(default_factory=list)

    def is_complete(self, expected_steps: int) -> bool:
        """True if every step ran and succeeded."""
        return len(self.steps) == expected_steps and all(s.error is None for s in self.steps)

    @property
    def latency_ms(self) -> float:
        """Wall time the task spent waiting on the model (sum of step latencies)."""
        return sum(s.end_to_end_latency_ms for s in self.steps)

    @property
    def ttft_ms(self) -> float | None:
        """Sum of per-step TTFT: the time spent before each step's first token."""
        if any(s.ttft_ms is None for s in self.steps):
            return None
        return sum(s.ttft_ms for s in self.steps if s.ttft_ms is not None)

    def step(self, number: int) -> BenchmarkMetric | None:
        return next((s for s in self.steps if s.step == number), None)


def group_agent_tasks(
    results: list[BenchmarkMetric],
) -> dict[tuple[str, str, str], list[AgentTask]]:
    """Group agent-step rows into tasks, keyed by (model, provider, workload).

    Rows without a ``step`` (single-request workloads) are ignored. Tasks are
    ordered by iteration and their steps by step number.
    """
    tasks: dict[tuple[str, str, str, int], AgentTask] = {}
    for r in results:
        if r.step is None:
            continue
        key = (r.model_name, r.service_name, r.workload_type, r.iteration)
        if key not in tasks:
            tasks[key] = AgentTask(r.model_name, r.service_name, r.workload_type, r.iteration)
        tasks[key].steps.append(r)

    grouped: dict[tuple[str, str, str], list[AgentTask]] = defaultdict(list)
    for (model, service, workload, _), task in sorted(tasks.items(), key=lambda kv: kv[0]):
        task.steps.sort(key=lambda s: s.step or 0)
        grouped[(model, service, workload)].append(task)
    return dict(grouped)


def expected_steps(tasks: list[AgentTask]) -> int:
    """Steps per task for a workload: the deepest step any task reached."""
    return max((s.step or 0 for t in tasks for s in t.steps), default=0)
