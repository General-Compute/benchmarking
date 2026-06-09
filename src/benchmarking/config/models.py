"""Pydantic models for configuration and metrics."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BenchmarkMetric(BaseModel):
    """Single benchmark measurement."""

    session_id: str
    service_name: str  # provider display name (e.g. "General Compute")
    model_name: str  # logical model (e.g. "gpt-oss-120b")
    workload_type: str
    iteration: int
    ttft_ms: Optional[float] = None
    end_to_end_latency_ms: float
    tokens_generated: int
    prompt_tokens: int
    tokens_per_sec: Optional[float] = None
    inter_token_latency_ms: Optional[float] = None
    attempts: int = 1
    timestamp: datetime
    error: Optional[str] = None


class BenchmarkSession(BaseModel):
    """Collection of benchmark results for a session."""

    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    services: List[str]
    models: List[str] = Field(default_factory=list)
    workloads: List[str]
    iterations_per_workload: int
    results: List[BenchmarkMetric] = Field(default_factory=list)


class ProviderConfig(BaseModel):
    """Configuration for a single inference provider (endpoint)."""

    name: str
    # Either set base_url directly, or name an env var in base_url_env to
    # resolve it at runtime (keeps private endpoints out of version control).
    base_url: Optional[str] = None
    base_url_env: Optional[str] = None
    api_key_env: str
    uses_max_completion_tokens: bool = False
    enabled: bool = True
    # Extra fields merged into the request body (e.g. OpenRouter `provider`
    # routing controls). Passed through verbatim via the SDK's extra_body.
    extra_body: Dict[str, Any] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    """A logical model and its per-provider IDs."""

    reasoning: bool = False
    ids: Dict[str, str] = Field(default_factory=dict)


class WorkloadConfig(BaseModel):
    """Configuration for a workload."""

    tokens: int
    template: str
    description: str = ""


class GenerationParams(BaseModel):
    """Parameters for text generation."""

    max_tokens: int = 1000
    temperature: float = 0.0
    top_p: float = 1.0


class BenchmarkConfig(BaseModel):
    """Configuration for benchmark execution."""

    default_iterations: int = 50
    warmup_iterations: int = 2
    timeout_seconds: int = 180
    retry_attempts: int = 3
    min_retry_wait: int = 2
    max_retry_wait: int = 30
    request_delay_seconds: float = 0.5
    vary_prompt: bool = True


class OutputConfig(BaseModel):
    """Configuration for output and reporting."""

    results_dir: str = "./results"
    csv_enabled: bool = True
    html_enabled: bool = True
