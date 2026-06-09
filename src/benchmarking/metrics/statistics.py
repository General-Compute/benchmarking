"""Statistical analysis for benchmark metrics."""

from typing import Dict, List

import numpy as np


class StatisticalAnalyzer:
    """Analyze benchmark metrics and compute statistics."""

    @staticmethod
    def analyze_metrics(values: List[float]) -> Dict[str, float]:
        """Calculate comprehensive statistics for a list of values.

        Args:
            values: List of numeric values to analyze

        Returns:
            Dictionary containing statistical measures
        """
        if not values:
            return {
                "mean": 0.0,
                "median": 0.0,
                "std_dev": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "count": 0,
                "cv": 0.0,
            }

        arr = np.array(values)

        mean_val = float(np.mean(arr))
        # Sample standard deviation (ddof=1); falls back to 0 for n == 1.
        std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

        # Coefficient of variation (std dev / mean)
        cv = std_val / mean_val if mean_val > 0 else 0.0

        return {
            "mean": mean_val,
            "median": float(np.median(arr)),
            "std_dev": std_val,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "count": len(arr),
            "cv": cv,
        }

    @staticmethod
    def filter_valid_values(values: List[float]) -> List[float]:
        """Filter out None, NaN, and infinite values.

        Args:
            values: List of values to filter

        Returns:
            Filtered list containing only valid finite values
        """
        return [v for v in values if v is not None and np.isfinite(v)]

    @staticmethod
    def calculate_error_rate(total: int, errors: int) -> float:
        """Calculate error rate as a percentage.

        Args:
            total: Total number of requests
            errors: Number of failed requests

        Returns:
            Error rate as percentage (0-100)
        """
        if total == 0:
            return 0.0
        return (errors / total) * 100.0

    @staticmethod
    def format_stats_table(stats: Dict[str, float], metric_name: str = "Metric") -> str:
        """Format statistics as a readable table.

        Args:
            stats: Statistics dictionary
            metric_name: Name of the metric being displayed

        Returns:
            Formatted string table
        """
        lines = [
            f"{metric_name} Statistics:",
            f"  Count:  {stats['count']}",
            f"  Mean:   {stats['mean']:.2f}",
            f"  Median: {stats['median']:.2f}",
            f"  Std:    {stats['std_dev']:.2f}",
            f"  Min:    {stats['min']:.2f}",
            f"  Max:    {stats['max']:.2f}",
            f"  P95:    {stats['p95']:.2f}",
            f"  P99:    {stats['p99']:.2f}",
            f"  CV:     {stats['cv']:.3f}",
        ]
        return "\n".join(lines)
