"""Base client for LLM benchmarking with streaming support."""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Optional


@dataclass
class StreamChunk:
    """Represents a streaming response chunk."""

    content: str
    timestamp: float
    is_first: bool = False
    is_final: bool = False
    # Token usage information (only present in final chunk)
    completion_tokens: Optional[int] = None
    prompt_tokens: Optional[int] = None


@dataclass
class BenchmarkResult:
    """Metrics from a single request/response benchmark."""

    ttft: Optional[float]  # Time to first token (seconds)
    end_to_end_latency: float  # Total time (seconds)
    tokens_generated: int
    prompt_tokens: int
    tokens_per_sec: Optional[float]  # Output generation rate (excludes TTFT)
    inter_token_latency_ms: Optional[float]  # Mean ms between output tokens
    request_timestamp: float  # Wall-clock epoch seconds (time.time())
    attempts: int = 1
    error: Optional[str] = None
    service_name: str = ""  # provider display name
    model_name: str = ""  # logical model name
    workload_type: str = ""


class BaseLLMClient(ABC):
    """Abstract base class for all LLM service clients."""

    def __init__(self, api_key: str, model: str, service_name: str):
        """Initialize the client.

        Args:
            api_key: API key for authentication
            model: Provider-specific model identifier
            service_name: Display name of the provider (for metrics/reports)
        """
        self.api_key = api_key
        self.model = model
        self.service_name = service_name

    @abstractmethod
    def stream_completion(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> AsyncIterator[StreamChunk]:
        """Stream completion and yield chunks with timestamps.

        Yields:
            StreamChunk objects with timestamps and content

        Raises:
            Exception: On network, API, or parsing errors
        """
        raise NotImplementedError

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        """Decide whether an exception is worth retrying.

        Retries transient failures (network/timeout/5xx/429) but NOT
        deterministic ones (auth, bad request) which would just waste time
        and never succeed.
        """
        # Lazy imports keep the base module dependency-light.
        try:
            import httpx

            if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
                return True
            if isinstance(exc, httpx.HTTPStatusError):
                code = exc.response.status_code
                return code == 429 or 500 <= code < 600
        except ImportError:  # pragma: no cover
            pass

        status = getattr(exc, "status_code", None)
        if isinstance(status, int):
            return status == 429 or 500 <= status < 600

        # OpenAI SDK exception types expose `status_code`; fall back to name.
        name = type(exc).__name__
        return name in {
            "APIConnectionError",
            "APITimeoutError",
            "InternalServerError",
            "RateLimitError",
        }

    async def benchmark_request(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.0,
        top_p: float = 1.0,
        workload_type: str = "",
        model_name: str = "",
        max_attempts: int = 3,
        min_retry_wait: float = 2.0,
        max_retry_wait: float = 30.0,
    ) -> BenchmarkResult:
        """Execute a request and collect benchmark metrics.

        Retries are handled here (not via a decorator on stream_completion) so
        the timer is reset on each attempt — backoff sleeps and failed attempts
        never leak into the reported TTFT/latency.
        """
        last_error: Optional[str] = None

        for attempt in range(1, max_attempts + 1):
            request_wallclock = time.time()
            perf_start = time.perf_counter()
            ttft: Optional[float] = None
            tokens_generated = 0
            prompt_tokens = 0
            content_parts = []

            try:
                async for chunk in self.stream_completion(prompt, max_tokens, temperature, top_p):
                    if chunk.is_first and ttft is None:
                        ttft = time.perf_counter() - perf_start
                    if chunk.content:
                        content_parts.append(chunk.content)
                    if chunk.is_final and chunk.completion_tokens is not None:
                        tokens_generated = chunk.completion_tokens
                        prompt_tokens = chunk.prompt_tokens or 0

                end_to_end = time.perf_counter() - perf_start

                # Fallbacks only if the provider omitted usage data.
                if tokens_generated == 0:
                    full_content = "".join(content_parts)
                    tokens_generated = max(1, len(full_content) // 4)
                if prompt_tokens == 0:
                    prompt_tokens = len(prompt.split())

                tps, itl = self._derive_throughput(ttft, end_to_end, tokens_generated)

                return BenchmarkResult(
                    ttft=ttft,
                    end_to_end_latency=end_to_end,
                    tokens_generated=tokens_generated,
                    prompt_tokens=prompt_tokens,
                    tokens_per_sec=tps,
                    inter_token_latency_ms=itl,
                    request_timestamp=request_wallclock,
                    attempts=attempt,
                    error=None,
                    service_name=self.service_name,
                    model_name=model_name,
                    workload_type=workload_type,
                )

            except Exception as e:  # noqa: BLE001 — we record and optionally retry
                last_error = f"{type(e).__name__}: {str(e)}"
                if attempt < max_attempts and self._is_retryable(e):
                    backoff = min(max_retry_wait, min_retry_wait * (2 ** (attempt - 1)))
                    await asyncio.sleep(backoff)
                    continue

                end_to_end = time.perf_counter() - perf_start
                return BenchmarkResult(
                    ttft=ttft,
                    end_to_end_latency=end_to_end,
                    tokens_generated=0,
                    prompt_tokens=len(prompt.split()),
                    tokens_per_sec=None,
                    inter_token_latency_ms=None,
                    request_timestamp=request_wallclock,
                    attempts=attempt,
                    error=last_error,
                    service_name=self.service_name,
                    model_name=model_name,
                    workload_type=workload_type,
                )

        # Unreachable, but keeps the type checker happy.
        raise RuntimeError(last_error or "benchmark_request exhausted attempts")

    @staticmethod
    def _derive_throughput(
        ttft: Optional[float], end_to_end: float, tokens_generated: int
    ) -> tuple[Optional[float], Optional[float]]:
        """Compute output tokens/sec and mean inter-token latency.

        Generation time excludes TTFT so throughput reflects decode speed, not
        prompt-processing/queue overhead.
        """
        if ttft is None or tokens_generated <= 1:
            return None, None
        gen_time = end_to_end - ttft
        if gen_time <= 0:
            return None, None
        tps = tokens_generated / gen_time
        itl_ms = (gen_time * 1000.0) / (tokens_generated - 1)
        return tps, itl_ms
