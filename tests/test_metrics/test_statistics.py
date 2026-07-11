"""Tests for StatisticalAnalyzer."""

import math

import numpy as np
import pytest

from benchmarking.metrics.statistics import StatisticalAnalyzer


def test_analyze_metrics_empty_returns_zeroed_stats() -> None:
    stats = StatisticalAnalyzer.analyze_metrics([])

    assert stats["count"] == 0
    assert stats["mean"] == 0.0
    assert stats["std_dev"] == 0.0
    assert stats["p50"] == 0.0
    assert stats["p95"] == 0.0
    assert stats["p99"] == 0.0
    assert stats["cv"] == 0.0


def test_analyze_metrics_single_value_has_zero_std_and_cv() -> None:
    # Sample std (ddof=1) is undefined for n == 1; it must fall back to 0.0
    # rather than raising or returning NaN.
    stats = StatisticalAnalyzer.analyze_metrics([5.0])

    assert stats["count"] == 1
    assert stats["mean"] == 5.0
    assert stats["median"] == 5.0
    assert stats["min"] == 5.0
    assert stats["max"] == 5.0
    assert stats["p50"] == 5.0
    assert stats["p99"] == 5.0
    assert stats["std_dev"] == 0.0
    assert stats["cv"] == 0.0


def test_analyze_metrics_multiple_values_match_numpy() -> None:
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    arr = np.array(values)

    stats = StatisticalAnalyzer.analyze_metrics(values)

    assert stats["count"] == 5
    assert stats["mean"] == pytest.approx(30.0)
    assert stats["median"] == pytest.approx(30.0)
    assert stats["min"] == 10.0
    assert stats["max"] == 50.0
    assert stats["std_dev"] == pytest.approx(float(np.std(arr, ddof=1)))
    assert stats["p95"] == pytest.approx(float(np.percentile(arr, 95)))
    assert stats["cv"] == pytest.approx(stats["std_dev"] / stats["mean"])


def test_analyze_metrics_all_zero_values_avoids_divide_by_zero_cv() -> None:
    stats = StatisticalAnalyzer.analyze_metrics([0.0, 0.0, 0.0])

    # mean == 0 must not blow up the coefficient of variation.
    assert stats["mean"] == 0.0
    assert stats["cv"] == 0.0


def test_filter_valid_values_drops_none_nan_and_inf() -> None:
    values = [1.0, None, float("nan"), float("inf"), -float("inf"), 2.5]

    result = StatisticalAnalyzer.filter_valid_values(values)

    assert result == [1.0, 2.5]


def test_filter_valid_values_keeps_finite_including_zero_and_negative() -> None:
    values = [0.0, -3.0, 42.0]

    result = StatisticalAnalyzer.filter_valid_values(values)

    assert result == [0.0, -3.0, 42.0]
    assert all(math.isfinite(v) for v in result)


@pytest.mark.parametrize(
    ("total", "errors", "expected"),
    [
        (0, 0, 0.0),
        (10, 0, 0.0),
        (10, 2, 20.0),
        (4, 1, 25.0),
        (3, 3, 100.0),
    ],
)
def test_calculate_error_rate(total: int, errors: int, expected: float) -> None:
    assert StatisticalAnalyzer.calculate_error_rate(total, errors) == pytest.approx(expected)


def test_format_stats_table_includes_metric_name_and_values() -> None:
    stats = StatisticalAnalyzer.analyze_metrics([1.0, 2.0, 3.0])

    table = StatisticalAnalyzer.format_stats_table(stats, metric_name="Latency")

    assert "Latency Statistics:" in table
    assert "Count:  3" in table
    assert "Mean:   2.00" in table
