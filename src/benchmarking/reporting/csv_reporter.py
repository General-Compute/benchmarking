"""CSV reporter for benchmark results."""

import csv
from pathlib import Path
from typing import Any, List, Optional

from ..config.models import BenchmarkSession
from ..metrics.agent_tasks import AgentTask, expected_steps, group_agent_tasks
from ..metrics.statistics import StatisticalAnalyzer


class CSVReporter:
    """Generate a statistical summary CSV grouped by model × provider × workload."""

    def __init__(self, results_dir: str = "./results"):
        self.results_dir = Path(results_dir)
        self.analyzer = StatisticalAnalyzer()

    def generate_summary_csv(
        self, session: BenchmarkSession, output_path: Optional[Path] = None
    ) -> Path:
        if output_path is None:
            output_path = self.results_dir / f"session_{session.session_id}_summary.csv"

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "model",
                    "provider",
                    "workload",
                    "metric",
                    "mean",
                    "median",
                    "p95",
                    "p99",
                    "std_dev",
                    "min",
                    "max",
                    "count",
                    "cv",
                ]
            )

            agent_tasks = group_agent_tasks(session.results)
            for model in session.models:
                for service in session.services:
                    for workload in session.workloads:
                        subset = [
                            r
                            for r in session.results
                            if r.model_name == model
                            and r.service_name == service
                            and r.workload_type == workload
                        ]
                        succeeded = [r for r in subset if r.error is None]
                        if not subset:
                            continue

                        error_rate = self.analyzer.calculate_error_rate(
                            len(subset), len(subset) - len(succeeded)
                        )
                        self._write_metric(
                            writer,
                            model,
                            service,
                            workload,
                            "error_rate_pct",
                            [error_rate],
                        )
                        if not succeeded:
                            continue

                        self._write_metric(
                            writer,
                            model,
                            service,
                            workload,
                            "ttft_ms",
                            [r.ttft_ms for r in succeeded if r.ttft_ms is not None],
                        )
                        self._write_metric(
                            writer,
                            model,
                            service,
                            workload,
                            "latency_ms",
                            [r.end_to_end_latency_ms for r in succeeded],
                        )
                        self._write_metric(
                            writer,
                            model,
                            service,
                            workload,
                            "output_tok_per_sec",
                            [r.tokens_per_sec for r in succeeded if r.tokens_per_sec],
                        )
                        self._write_metric(
                            writer,
                            model,
                            service,
                            workload,
                            "inter_token_latency_ms",
                            [
                                r.inter_token_latency_ms
                                for r in succeeded
                                if r.inter_token_latency_ms
                            ],
                        )
                        self._write_metric(
                            writer,
                            model,
                            service,
                            workload,
                            "tokens_generated",
                            [float(r.tokens_generated) for r in succeeded],
                        )
                        tasks = agent_tasks.get((model, service, workload))
                        if tasks:
                            self._write_agent_metrics(writer, model, service, workload, tasks)

        return output_path

    def _write_agent_metrics(
        self, writer: Any, model: str, service: str, workload: str, tasks: list[AgentTask]
    ) -> None:
        """Task-level rows for an agent workload.

        Per-request rows above treat every step as a sample. These rows answer
        the agent question instead: how long a whole task takes, how often it
        finishes, and how TTFT grows from the first step to the last as the
        re-sent context gets longer.
        """
        n_steps = expected_steps(tasks)
        complete = [t for t in tasks if t.is_complete(n_steps)]
        self._write_metric(
            writer,
            model,
            service,
            workload,
            "task_completion_pct",
            [100.0 * len(complete) / len(tasks)],
        )
        self._write_metric(
            writer, model, service, workload, "task_latency_ms", [t.latency_ms for t in complete]
        )
        self._write_metric(
            writer,
            model,
            service,
            workload,
            "task_ttft_ms",
            [t.ttft_ms for t in complete if t.ttft_ms is not None],
        )
        for number in sorted({1, n_steps}):
            steps = [t.step(number) for t in complete]
            self._write_metric(
                writer,
                model,
                service,
                workload,
                f"ttft_ms_step_{number}",
                [s.ttft_ms for s in steps if s is not None and s.ttft_ms is not None],
            )
            self._write_metric(
                writer,
                model,
                service,
                workload,
                f"prompt_tokens_step_{number}",
                [float(s.prompt_tokens) for s in steps if s is not None],
            )

    def _write_metric(
        self,
        writer: Any,
        model: str,
        service: str,
        workload: str,
        name: str,
        values: List[float],
    ) -> None:
        values = [v for v in values if v is not None]
        if not values:
            return
        s = self.analyzer.analyze_metrics(values)
        writer.writerow(
            [
                model,
                service,
                workload,
                name,
                s["mean"],
                s["median"],
                s["p95"],
                s["p99"],
                s["std_dev"],
                s["min"],
                s["max"],
                s["count"],
                s["cv"],
            ]
        )
