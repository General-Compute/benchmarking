"""Provider performance reporter with throughput and interactivity analysis."""

from html import escape
from pathlib import Path
from typing import Dict, cast

import plotly.graph_objects as go

from ..config.models import BenchmarkSession
from ..metrics.statistics import StatisticalAnalyzer


class ProviderPerformanceReporter:
    """Generate provider reports with throughput vs latency/interactivity charts."""

    # Color scheme for providers (similar to InferenceMAX)
    PROVIDER_COLORS = {
        "General Compute": "#4CAF50",  # Green
        "OpenRouter": "#FF5722",  # Red/orange
    }

    def __init__(self, results_dir: str = "./results"):
        """Initialize provider performance reporter.

        Args:
            results_dir: Directory to save reports
        """
        self.results_dir = Path(results_dir)
        self.analyzer = StatisticalAnalyzer()

    def generate_report(self, session: BenchmarkSession) -> Path:
        """Generate comprehensive provider performance HTML report.

        Args:
            session: BenchmarkSession with benchmark results

        Returns:
            Path to generated HTML report
        """
        output_path = self.results_dir / f"session_{session.session_id}_provider_performance.html"

        models = session.models or [""]
        sections = []
        for model in models:
            sub = self._subsession(session, model)
            if not sub.results:
                continue
            sections.append(self._build_model_section(model, sub))

        html_content = self._generate_html(session, "\n".join(sections))

        with open(output_path, "w") as f:
            f.write(html_content)

        return output_path

    def _subsession(self, session: BenchmarkSession, model: str) -> BenchmarkSession:
        """Filter a session down to a single model's results."""
        results = [r for r in session.results if (not model or r.model_name == model)]
        return BenchmarkSession(
            session_id=session.session_id,
            start_time=session.start_time,
            end_time=session.end_time,
            services=sorted({r.service_name for r in results}),
            models=[model] if model else [],
            workloads=sorted({r.workload_type for r in results}),
            iterations_per_workload=session.iterations_per_workload,
            results=results,
        )

    def _build_model_section(self, model: str, sub: BenchmarkSession) -> str:
        """Render throughput/reliability/TTFT analysis for one model."""
        stats = self._calculate_statistics(sub)
        throughput_data = self._calculate_throughput_metrics(sub)
        model_label = escape(model or "all")
        return f"""
    <div class="section">
        <h2>Model: {model_label}</h2>
        {self._generate_stats_table(stats, throughput_data)}
        <div class="chart-container">{self._create_throughput_latency_chart(throughput_data)}</div>
        <div class="chart-container">{self._create_throughput_interactivity_chart(throughput_data)}</div>
        <div class="chart-container">{self._create_reliability_chart(stats)}</div>
        <div class="chart-container">{self._create_ttft_distribution_chart(sub)}</div>
    </div>"""

    def _calculate_statistics(self, session: BenchmarkSession) -> Dict:
        """Calculate statistics for all service/workload combinations.

        Args:
            session: BenchmarkSession

        Returns:
            Dictionary with statistics
        """
        stats: Dict = {}

        for service in session.services:
            stats[service] = {}
            for workload in session.workloads:
                # Filter results
                results = [
                    r
                    for r in session.results
                    if r.service_name == service and r.workload_type == workload and r.error is None
                ]

                # Count errors
                total_for_combo = len(
                    [
                        r
                        for r in session.results
                        if r.service_name == service and r.workload_type == workload
                    ]
                )
                errors = total_for_combo - len(results)
                error_rate = self.analyzer.calculate_error_rate(total_for_combo, errors)

                if not results:
                    stats[service][workload] = {
                        "error_rate": error_rate,
                        "total_requests": total_for_combo,
                        "success_count": 0,
                    }
                    continue

                # Calculate TTFT stats
                ttft_values = [r.ttft_ms for r in results if r.ttft_ms is not None]
                ttft_stats = self.analyzer.analyze_metrics(ttft_values) if ttft_values else {}

                # Calculate latency stats
                latency_values = [r.end_to_end_latency_ms for r in results]
                latency_stats = self.analyzer.analyze_metrics(latency_values)

                stats[service][workload] = {
                    "ttft": ttft_stats,
                    "latency": latency_stats,
                    "error_rate": error_rate,
                    "total_requests": total_for_combo,
                    "success_count": len(results),
                }

        return stats

    def _calculate_throughput_metrics(self, session: BenchmarkSession) -> Dict:
        """Calculate throughput metrics for each service/workload combination.

        Args:
            session: BenchmarkSession

        Returns:
            Dictionary with throughput data points
        """
        throughput_data: Dict = {}

        for service in session.services:
            throughput_data[service] = {}
            for workload in session.workloads:
                # Filter successful results
                results = [
                    r
                    for r in session.results
                    if r.service_name == service
                    and r.workload_type == workload
                    and r.error is None
                    and r.end_to_end_latency_ms > 0
                ]

                if not results:
                    continue

                # Calculate metrics for each result
                data_points = []
                for result in results:
                    # Total latency (includes overhead)
                    total_latency_seconds = result.end_to_end_latency_ms / 1000.0

                    # Generation latency (excludes TTFT overhead)
                    ttft_seconds = (result.ttft_ms or 0) / 1000.0
                    generation_latency_seconds = total_latency_seconds - ttft_seconds

                    # Pure generation rate: tokens/sec after first token (excludes overhead)
                    generation_rate = (
                        result.tokens_generated / generation_latency_seconds
                        if generation_latency_seconds > 0
                        else 0
                    )

                    # Total throughput: includes all tokens (prompt + completion) and overhead
                    total_tokens = result.prompt_tokens + result.tokens_generated
                    total_throughput = (
                        total_tokens / total_latency_seconds if total_latency_seconds > 0 else 0
                    )

                    # Legacy throughput (completion only / total time) - kept for backward compatibility
                    legacy_throughput = (
                        result.tokens_generated / total_latency_seconds
                        if total_latency_seconds > 0
                        else 0
                    )

                    data_points.append(
                        {
                            "generation_rate": generation_rate,  # NEW: Pure generation speed (tokens/sec after TTFT)
                            "total_throughput": total_throughput,  # NEW: Total tokens processed per second
                            "legacy_throughput": legacy_throughput,  # OLD: For backward compatibility
                            "latency_seconds": total_latency_seconds,
                            "generation_latency_seconds": generation_latency_seconds,  # NEW: Generation time only
                            "ttft_ms": result.ttft_ms,
                            "ttft_seconds": ttft_seconds,
                            "tokens_generated": result.tokens_generated,
                            "prompt_tokens": result.prompt_tokens,
                        }
                    )

                throughput_data[service][workload] = data_points

        return throughput_data

    def _create_throughput_latency_chart(self, throughput_data: Dict) -> str:
        """Create Token Throughput vs End-to-end Latency scatter plot.

        Args:
            throughput_data: Throughput metrics dictionary

        Returns:
            HTML div containing Plotly chart
        """
        fig = go.Figure()

        for service, workload_data in throughput_data.items():
            color = self.PROVIDER_COLORS.get(service, "#888888")

            for workload, data_points in workload_data.items():
                if not data_points:
                    continue

                generation_rate_values = [p["generation_rate"] for p in data_points]
                generation_latency_values = [p["generation_latency_seconds"] for p in data_points]

                # Different marker shapes and line styles for different workloads
                if "short" in workload.lower():
                    marker_symbol = "circle"
                    line_dash = "solid"
                elif "long" in workload.lower():
                    marker_symbol = "square"
                    line_dash = "dash"
                else:
                    marker_symbol = "diamond"
                    line_dash = "dot"

                fig.add_trace(
                    go.Scatter(
                        x=generation_latency_values,
                        y=generation_rate_values,
                        mode="markers+lines",
                        name=f"{service} ({workload})",
                        marker=dict(
                            size=10,
                            symbol=marker_symbol,
                            color=color,
                            line=dict(width=1, color="white"),
                        ),
                        line=dict(color=color, width=2, dash=line_dash),
                        hovertemplate=(
                            f"<b>{service} - {workload}</b><br>"
                            "Generation Latency: %{x:.2f}s<br>"
                            "Generation Rate: %{y:.2f} tok/s<br>"
                            "<extra></extra>"
                        ),
                    )
                )

        fig.update_layout(
            title="Pure Generation Rate vs. Generation Latency (excludes TTFT overhead)<br><sub>Source: GC Benchmarking</sub>",
            xaxis_title="Generation Latency (s) - excludes TTFT",
            yaxis_title="Generation Rate (tok/s) - pure generation speed",
            showlegend=True,
            height=600,
            hovermode="closest",
            template="plotly_white",
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        )

        return cast(str, fig.to_html(full_html=False, include_plotlyjs=False))

    def _create_throughput_interactivity_chart(self, throughput_data: Dict) -> str:
        """Create Token Throughput vs Interactivity scatter plot.

        Args:
            throughput_data: Throughput metrics dictionary

        Returns:
            HTML div containing Plotly chart
        """
        fig = go.Figure()

        for service, workload_data in throughput_data.items():
            color = self.PROVIDER_COLORS.get(service, "#888888")

            for workload, data_points in workload_data.items():
                if not data_points:
                    continue

                total_throughput_values = [p["total_throughput"] for p in data_points]
                ttft_values = [p["ttft_ms"] for p in data_points]

                # Different marker shapes and line styles for different workloads
                if "short" in workload.lower():
                    marker_symbol = "circle"
                    line_dash = "solid"
                elif "long" in workload.lower():
                    marker_symbol = "square"
                    line_dash = "dash"
                else:
                    marker_symbol = "diamond"
                    line_dash = "dot"

                fig.add_trace(
                    go.Scatter(
                        x=ttft_values,
                        y=total_throughput_values,
                        mode="markers+lines",
                        name=f"{service} ({workload})",
                        marker=dict(
                            size=10,
                            symbol=marker_symbol,
                            color=color,
                            line=dict(width=1, color="white"),
                        ),
                        line=dict(color=color, width=2, dash=line_dash),
                        hovertemplate=(
                            f"<b>{service} - {workload}</b><br>"
                            "TTFT Overhead: %{x:.2f} ms<br>"
                            "Total Throughput: %{y:.2f} tok/s<br>"
                            "<extra></extra>"
                        ),
                    )
                )

        fig.update_layout(
            title="Total Throughput vs. TTFT Overhead (includes prompt processing)<br><sub>Source: GC Benchmarking</sub>",
            xaxis_title="TTFT Overhead (ms)",
            yaxis_title="Total Throughput (tok/s) - all tokens",
            showlegend=True,
            height=600,
            hovermode="closest",
            template="plotly_white",
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        )

        return cast(str, fig.to_html(full_html=False, include_plotlyjs=False))

    def _create_reliability_chart(self, stats: Dict) -> str:
        """Create Provider Reliability bar chart showing failure rates.

        Args:
            stats: Statistics dictionary

        Returns:
            HTML div containing Plotly chart
        """
        fig = go.Figure()

        # Calculate overall failure rate per provider
        provider_stats = {}
        for service, workload_stats in stats.items():
            total_requests = 0
            failed_requests = 0

            for workload, metrics in workload_stats.items():
                total = metrics.get("total_requests", 0)
                success = metrics.get("success_count", 0)
                total_requests += total
                failed_requests += total - success

            failure_rate = (failed_requests / total_requests * 100) if total_requests > 0 else 0
            provider_stats[service] = {
                "failure_rate": failure_rate,
                "total_requests": total_requests,
                "failed_requests": failed_requests,
            }

        # Create bar chart
        providers = list(provider_stats.keys())
        failure_rates = [provider_stats[p]["failure_rate"] for p in providers]
        colors = [self.PROVIDER_COLORS.get(p, "#888888") for p in providers]

        fig.add_trace(
            go.Bar(
                x=providers,
                y=failure_rates,
                marker=dict(color=colors),
                text=[f"{rate:.1f}%" for rate in failure_rates],
                textposition="auto",
                hovertemplate=("<b>%{x}</b><br>" "Failure Rate: %{y:.2f}%<br>" "<extra></extra>"),
            )
        )

        fig.update_layout(
            title="Provider Reliability<br><sub>Failure rate percentage of completed runs</sub>",
            xaxis_title="Provider",
            yaxis_title="Failure Rate (%)",
            yaxis=dict(range=[0, max(failure_rates + [10]) * 1.1]),
            showlegend=False,
            height=400,
            template="plotly_white",
        )

        return cast(str, fig.to_html(full_html=False, include_plotlyjs=False))

    def _create_ttft_distribution_chart(self, session: BenchmarkSession) -> str:
        """Create TTFT distribution box plot (keeping existing TTFT testing).

        Args:
            session: BenchmarkSession

        Returns:
            HTML div containing Plotly chart
        """
        fig = go.Figure()

        for service in session.services:
            color = self.PROVIDER_COLORS.get(service, "#888888")

            for workload in session.workloads:
                # Get TTFT values
                values = [
                    r.ttft_ms
                    for r in session.results
                    if r.service_name == service
                    and r.workload_type == workload
                    and r.ttft_ms is not None
                    and r.error is None
                ]

                if values:
                    fig.add_trace(
                        go.Box(
                            y=values,
                            name=f"{service} ({workload})",
                            boxmean="sd",
                            marker=dict(color=color),
                            line=dict(color=color),
                        )
                    )

        fig.update_layout(
            title="Time to First Token (TTFT) Distribution",
            yaxis_title="TTFT (ms)",
            showlegend=True,
            height=500,
            template="plotly_white",
        )

        return cast(str, fig.to_html(full_html=False, include_plotlyjs=False))

    def _generate_html(self, session: BenchmarkSession, sections: str) -> str:
        """Assemble the full provider performance page from per-model sections."""
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Provider Performance Report - {escape(session.session_id)}</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
            background-color: #0a0a0a;
            color: #e0e0e0;
        }}
        .header {{
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            padding: 40px;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            border: 1px solid #333;
        }}
        h1 {{
            margin: 0 0 10px 0;
            color: #4CAF50;
            font-size: 2.5em;
            font-weight: 700;
        }}
        .subtitle {{
            color: #999;
            font-size: 1.1em;
            margin: 10px 0;
        }}
        .metadata {{
            color: #aaa;
            font-size: 14px;
            margin-top: 20px;
        }}
        .metadata p {{
            margin: 5px 0;
        }}
        .section {{
            background-color: #1a1a1a;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            border: 1px solid #333;
        }}
        h2 {{
            margin-top: 0;
            color: #4CAF50;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
            font-size: 1.8em;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #333;
        }}
        th {{
            background-color: #252525;
            font-weight: 600;
            color: #4CAF50;
        }}
        td {{
            color: #e0e0e0;
        }}
        tr:hover {{
            background-color: #252525;
        }}
        .chart-container {{
            margin: 30px 0;
            background-color: #0f0f0f;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #333;
        }}
        .footer {{
            text-align: center;
            color: #666;
            font-size: 12px;
            margin-top: 40px;
            padding: 20px;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
            margin-left: 10px;
        }}
        .badge-success {{
            background-color: #4CAF50;
            color: white;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Provider Performance <span class="badge badge-success">BETA</span></h1>
        <p class="subtitle">LLM Provider Performance Benchmarking</p>
        <div class="metadata">
            <p><strong>Session ID:</strong> {escape(session.session_id)}</p>
            <p><strong>Start Time:</strong> {session.start_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>End Time:</strong> {session.end_time.strftime('%Y-%m-%d %H:%M:%S') if session.end_time else 'N/A'}</p>
            <p><strong>Providers Tested:</strong> {escape(', '.join(session.services))}</p>
            <p><strong>Models:</strong> {escape(', '.join(session.models) or 'n/a')}</p>
            <p><strong>Workloads:</strong> {escape(', '.join(session.workloads))}</p>
            <p><strong>Iterations per workload:</strong> {session.iterations_per_workload}</p>
        </div>
    </div>

    {sections}

    <div class="footer">
        <p>Generated by GC Benchmarking Tool | {session.start_time.strftime('%Y-%m-%d')}</p>
    </div>
</body>
</html>
"""
        return html

    def _generate_stats_table(self, stats: Dict, throughput_data: Dict) -> str:
        """Generate HTML table for statistics.

        Args:
            stats: Statistics dictionary
            throughput_data: Throughput metrics dictionary

        Returns:
            HTML table string
        """
        rows = []

        for service, workload_stats in stats.items():
            for workload, metrics in workload_stats.items():
                ttft = metrics.get("ttft", {})
                latency = metrics.get("latency", {})

                # Convert milliseconds to seconds
                ttft_mean_s = ttft.get("mean", 0) / 1000.0
                ttft_median_s = ttft.get("median", 0) / 1000.0
                ttft_p95_s = ttft.get("p95", 0) / 1000.0

                latency_mean_s = latency.get("mean", 0) / 1000.0
                latency_median_s = latency.get("median", 0) / 1000.0
                latency_p95_s = latency.get("p95", 0) / 1000.0

                # Calculate throughput metrics from actual data
                data_points = throughput_data.get(service, {}).get(workload, [])
                if data_points:
                    import statistics

                    avg_generation_rate = statistics.mean(
                        [p["generation_rate"] for p in data_points]
                    )
                    avg_total_throughput = statistics.mean(
                        [p["total_throughput"] for p in data_points]
                    )
                else:
                    avg_generation_rate = 0
                    avg_total_throughput = 0

                rows.append(
                    f"""
                    <tr>
                        <td><strong>{escape(service)}</strong></td>
                        <td>{escape(workload)}</td>
                        <td>{ttft_mean_s:.3f}</td>
                        <td>{ttft_median_s:.3f}</td>
                        <td>{ttft_p95_s:.3f}</td>
                        <td>{latency_mean_s:.3f}</td>
                        <td>{latency_median_s:.3f}</td>
                        <td>{latency_p95_s:.3f}</td>
                        <td>{avg_generation_rate:.2f}</td>
                        <td>{avg_total_throughput:.2f}</td>
                        <td>{metrics.get('error_rate', 0):.1f}%</td>
                    </tr>
                """
                )

        return f"""
            <table>
                <thead>
                    <tr>
                        <th>Provider</th>
                        <th>Workload</th>
                        <th>TTFT Mean (s)</th>
                        <th>TTFT Median (s)</th>
                        <th>TTFT p95 (s)</th>
                        <th>Latency Mean (s)</th>
                        <th>Latency Median (s)</th>
                        <th>Latency p95 (s)</th>
                        <th>Gen Rate (tok/s)</th>
                        <th>Total Throughput (tok/s)</th>
                        <th>Error Rate</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        """
