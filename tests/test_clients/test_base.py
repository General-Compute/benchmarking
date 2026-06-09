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
