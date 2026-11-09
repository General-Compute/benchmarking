"""Metrics collector for benchmark results."""

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from ..clients.base import BenchmarkResult
from ..config.models import BenchmarkMetric, BenchmarkSession

CSV_HEADER = [
    "session_id",
    "service_name",
    "model_name",
    "workload_type",
    "iteration",
    "ttft_ms",
    "end_to_end_latency_ms",
    "tokens_generated",
    "prompt_tokens",
    "tokens_per_sec",
    "inter_token_latency_ms",
    "attempts",
    "timestamp",
    "error",
    # Appended last so raw CSVs written before agent workloads still load.
    "step",
]


class MetricsCollector:
    """Collect and persist benchmark metrics."""

    def __init__(self, session_id: str, results_dir: str = "./results"):
        self.session_id = session_id
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.results_dir / f"session_{session_id}_raw.csv"
        self._initialize_csv()

    def _initialize_csv(self) -> None:
        # Write the header when the file is missing OR exists but is empty.
        # A leftover 0-byte file (e.g. from an interrupted/restarted run)
        # would otherwise be appended to without ever getting a header,
        # producing a headerless CSV that breaks report generation.
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(CSV_HEADER)

    def save_result(self, result: BenchmarkResult, iteration: int) -> None:
        ttft_ms = result.ttft * 1000 if result.ttft is not None else None
        latency_ms = result.end_to_end_latency * 1000
        # request_timestamp is wall-clock epoch seconds (time.time()).
        ts = datetime.fromtimestamp(result.request_timestamp, tz=timezone.utc).isoformat()

        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    self.session_id,
                    result.service_name,
                    result.model_name,
                    result.workload_type,
                    iteration,
                    ttft_ms,
                    latency_ms,
                    result.tokens_generated,
                    result.prompt_tokens,
                    result.tokens_per_sec,
                    result.inter_token_latency_ms,
                    result.attempts,
                    ts,
                    result.error or "",
                    result.step if result.step is not None else "",
                ]
            )

    def load_session(self, session_id: Optional[str] = None) -> BenchmarkSession:
        if session_id is None:
            session_id = self.session_id

        csv_path = self.results_dir / f"session_{session_id}_raw.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Session {session_id} not found")

        metrics: List[BenchmarkMetric] = []
        services: set[str] = set()
        models: set[str] = set()
        workloads: set[str] = set()
        start_time: Optional[datetime] = None
        end_time: Optional[datetime] = None
        iterations_per_workload = 0

        def _f(v: str | None) -> float | None:
            if v is None or v == "":
                return None
            return float(v)

        with open(csv_path, "r", newline="") as f:
            data_rows = [r for r in csv.reader(f) if r]

        # Tolerate raw CSVs written without a header row. A genuine header
        # begins with the literal "session_id" column name; anything else is
        # treated as data and mapped positionally against CSV_HEADER. This
        # lets us still produce reports from files an interrupted/restarted
        # run left headerless.
        if data_rows and data_rows[0][0] == CSV_HEADER[0]:
            data_rows = data_rows[1:]

        for raw in data_rows:
            # Pad short rows so every column is present (mirrors DictReader's
            # restval behaviour) before zipping into a name->value mapping.
            padded = list(raw) + [""] * (len(CSV_HEADER) - len(raw))
            row = dict(zip(CSV_HEADER, padded))
            timestamp = datetime.fromisoformat(row["timestamp"])
            if start_time is None or timestamp < start_time:
                start_time = timestamp
            if end_time is None or timestamp > end_time:
                end_time = timestamp

            services.add(row["service_name"])
            models.add(row.get("model_name", ""))
            workloads.add(row["workload_type"])
            iterations_per_workload = max(iterations_per_workload, int(row["iteration"]) + 1)

            metrics.append(
                BenchmarkMetric(
                    session_id=row["session_id"],
                    service_name=row["service_name"],
                    model_name=row.get("model_name", ""),
                    workload_type=row["workload_type"],
                    iteration=int(row["iteration"]),
                    ttft_ms=_f(row.get("ttft_ms")),
                    end_to_end_latency_ms=float(row["end_to_end_latency_ms"]),
                    tokens_generated=int(row["tokens_generated"]),
                    prompt_tokens=int(row["prompt_tokens"]),
                    tokens_per_sec=_f(row.get("tokens_per_sec")),
                    inter_token_latency_ms=_f(row.get("inter_token_latency_ms")),
                    attempts=int(row.get("attempts") or 1),
                    timestamp=timestamp,
                    error=row["error"] if row["error"] else None,
                    step=int(row["step"]) if row.get("step") else None,
                )
            )

        return BenchmarkSession(
            session_id=session_id,
            start_time=start_time or datetime.now(timezone.utc),
            end_time=end_time,
            services=sorted(services),
            models=sorted(m for m in models if m),
            workloads=sorted(workloads),
            iterations_per_workload=iterations_per_workload,
            results=metrics,
        )

    @classmethod
    def list_sessions(cls, results_dir: str = "./results") -> List[str]:
        results_path = Path(results_dir)
        if not results_path.exists():
            return []
        sessions = []
        for csv_file in results_path.glob("session_*_raw.csv"):
            sessions.append(csv_file.stem.replace("session_", "").replace("_raw", ""))
        return sorted(sessions)
