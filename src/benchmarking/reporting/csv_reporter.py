"""CSV reporter for benchmark results."""

import csv
from pathlib import Path
from typing import Any, List, Optional

from ..config.models import BenchmarkSession
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

        return output_path

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
