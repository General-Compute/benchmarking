import httpx
import pytest

from benchmarking.clients.base import BaseLLMClient


@pytest.mark.parametrize(
    ("ttft", "end_to_end", "tokens", "expected_tps", "expected_itl"),
    [
        (0.25, 1.25, 101, 101.0, 10.0),
        (None, 1.25, 101, None, None),
        (0.25, 0.25, 101, None, None),
        (0.25, 1.25, 1, None, None),
    ],
)
def test_derive_throughput(
    ttft: float | None,
    end_to_end: float,
    tokens: int,
    expected_tps: float | None,
    expected_itl: float | None,
) -> None:
    tps, itl = BaseLLMClient._derive_throughput(ttft, end_to_end, tokens)

    assert tps == expected_tps
    assert itl == expected_itl


def _http_status_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError("error", request=request, response=response)


class _StatusException(Exception):
    """Stand-in for an SDK error that carries a numeric status_code."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class RateLimitError(Exception):
    """Name-matched stand-in for an OpenAI-SDK retryable error."""


class BadRequestError(Exception):
    """Name-matched stand-in for a non-retryable SDK error."""


def test_is_retryable_httpx_timeout_and_transport_errors() -> None:
    assert BaseLLMClient._is_retryable(httpx.TimeoutException("slow")) is True
    assert BaseLLMClient._is_retryable(httpx.ConnectError("refused")) is True


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [(429, True), (500, True), (503, True), (400, False), (404, False)],
)
def test_is_retryable_httpx_status_error(status_code: int, expected: bool) -> None:
    assert BaseLLMClient._is_retryable(_http_status_error(status_code)) is expected


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [(429, True), (502, True), (401, False), (400, False)],
)
def test_is_retryable_duck_typed_status_code(status_code: int, expected: bool) -> None:
    assert BaseLLMClient._is_retryable(_StatusException(status_code)) is expected


def test_is_retryable_by_exception_class_name() -> None:
    assert BaseLLMClient._is_retryable(RateLimitError()) is True
    assert BaseLLMClient._is_retryable(BadRequestError()) is False


def test_is_retryable_unknown_exception_is_not_retried() -> None:
    assert BaseLLMClient._is_retryable(ValueError("bad arg")) is False
