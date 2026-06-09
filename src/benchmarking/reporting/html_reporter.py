"""HTML reporter with interactive charts for benchmark results."""

from html import escape
from pathlib import Path
from typing import Dict, cast

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..config.models import BenchmarkSession
from ..metrics.statistics import StatisticalAnalyzer


class HTMLReporter:
    """Generate HTML reports with interactive Plotly charts."""

    def __init__(self, results_dir: str = "./results"):
        """Initialize HTML reporter.

        Args:
            results_dir: Directory to save reports
        """
        self.results_dir = Path(results_dir)
        self.analyzer = StatisticalAnalyzer()

    def generate_report(self, session: BenchmarkSession) -> Path:
        """Generate comprehensive HTML report.

        Args:
            session: BenchmarkSession with benchmark results

        Returns:
            Path to generated HTML report
        """
        output_path = self.results_dir / f"session_{session.session_id}_report.html"

        # One section per logical model, comparing providers within that model.
        models = session.models or [""]
        sections = []
        for model in models:
            sub = self._subsession(session, model)
            if not sub.results:
                continue
            stats = self._calculate_statistics(sub)
            sections.append(self._build_model_section(model, sub, stats))

        html_content = self._generate_html(session, "\n".join(sections))

        with open(output_path, "w") as f:
            f.write(html_content)

        return output_path

    def _subsession(self, session: BenchmarkSession, model: str) -> BenchmarkSession:
        """Build a filtered session containing only one model's results."""
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

    def _build_model_section(self, model: str, sub: BenchmarkSession, stats: Dict) -> str:
        """Render the stats table + charts for a single model."""
        table = self._generate_stats_table(stats)
        comparison_chart = self._create_comparison_chart(stats)
        ttft_chart = self._create_ttft_chart(sub)
        latency_chart = self._create_latency_chart(sub)
        model_label = escape(model or "all")
        return f"""
    <div class="section">
        <h2>Model: {model_label}</h2>
        {table}
        <div class="chart-container">{comparison_chart}</div>
        <div class="chart-container">{ttft_chart}</div>
        <div class="chart-container">{latency_chart}</div>
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

                if not results:
                    continue

                # Calculate TTFT stats
                ttft_values = [r.ttft_ms for r in results if r.ttft_ms is not None]
                ttft_stats = self.analyzer.analyze_metrics(ttft_values) if ttft_values else {}

                # Calculate latency stats
                latency_values = [r.end_to_end_latency_ms for r in results]
                latency_stats = self.analyzer.analyze_metrics(latency_values)

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

                stats[service][workload] = {
                    "ttft": ttft_stats,
                    "latency": latency_stats,
                    "error_rate": error_rate,
                    "total_requests": total_for_combo,
                }

        return stats

    def _create_ttft_chart(self, session: BenchmarkSession) -> str:
        """Create TTFT box plot chart.

        Args:
            session: BenchmarkSession

        Returns:
            HTML div containing Plotly chart
        """
        fig = go.Figure()

        for service in session.services:
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
                        )
                    )

        fig.update_layout(
            title="Time to First Token (TTFT) Distribution",
            yaxis_title="TTFT (ms)",
            showlegend=True,
            height=500,
        )

        return cast(str, fig.to_html(full_html=False, include_plotlyjs=False))

    def _create_latency_chart(self, session: BenchmarkSession) -> str:
        """Create end-to-end latency box plot chart.

        Args:
            session: BenchmarkSession

        Returns:
            HTML div containing Plotly chart
        """
        fig = go.Figure()

        for service in session.services:
            for workload in session.workloads:
                # Get latency values
                values = [
                    r.end_to_end_latency_ms
                    for r in session.results
                    if r.service_name == service and r.workload_type == workload and r.error is None
                ]

                if values:
                    fig.add_trace(
                        go.Box(
                            y=values,
                            name=f"{service} ({workload})",
                            boxmean="sd",
                        )
                    )

        fig.update_layout(
            title="End-to-End Latency Distribution",
            yaxis_title="Latency (ms)",
            showlegend=True,
            height=500,
        )

        return cast(str, fig.to_html(full_html=False, include_plotlyjs=False))

    def _create_comparison_chart(self, stats: Dict) -> str:
        """Create comparison bar chart for mean metrics.

        Args:
            stats: Statistics dictionary

        Returns:
            HTML div containing Plotly chart
        """
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Mean TTFT Comparison", "Mean Latency Comparison"),
        )

        services = list(stats.keys())
        workloads = list(next(iter(stats.values())).keys()) if stats else []

        for i, workload in enumerate(workloads):
            ttft_means = []
            latency_means = []

            for service in services:
                if workload in stats[service]:
                    ttft_means.append(stats[service][workload]["ttft"].get("mean", 0))
                    latency_means.append(stats[service][workload]["latency"].get("mean", 0))
                else:
                    ttft_means.append(0)
                    latency_means.append(0)

            # TTFT bars
            fig.add_trace(go.Bar(name=workload, x=services, y=ttft_means), row=1, col=1)

            # Latency bars
            fig.add_trace(
                go.Bar(name=workload, x=services, y=latency_means, showlegend=False), row=1, col=2
            )

        fig.update_yaxes(title_text="TTFT (ms)", row=1, col=1)
        fig.update_yaxes(title_text="Latency (ms)", row=1, col=2)
        fig.update_layout(height=400, barmode="group")

        return cast(str, fig.to_html(full_html=False, include_plotlyjs=False))

    def _generate_html(self, session: BenchmarkSession, sections: str) -> str:
        """Assemble the full HTML page from pre-rendered per-model sections."""
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Benchmark Report - {escape(session.session_id)}</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            margin: 0 0 10px 0;
            color: #333;
        }}
        .metadata {{
            color: #666;
            font-size: 14px;
        }}
        .metadata p {{
            margin: 5px 0;
        }}
        .section {{
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h2 {{
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
            color: #333;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .chart-container {{
            margin: 30px 0;
        }}
        .footer {{
            text-align: center;
            color: #666;
            font-size: 12px;
            margin-top: 40px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>LLM Benchmark Report</h1>
        <div class="metadata">
            <p><strong>Session ID:</strong> {escape(session.session_id)}</p>
            <p><strong>Start Time:</strong> {session.start_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>End Time:</strong> {session.end_time.strftime('%Y-%m-%d %H:%M:%S') if session.end_time else 'N/A'}</p>
            <p><strong>Providers:</strong> {escape(', '.join(session.services))}</p>
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

    def _generate_stats_table(self, stats: Dict) -> str:
        """Generate HTML table for statistics.

        Args:
            stats: Statistics dictionary

        Returns:
            HTML table string
        """
        rows = []

        for service, workload_stats in stats.items():
            for workload, metrics in workload_stats.items():
                ttft = metrics.get("ttft", {})
                latency = metrics.get("latency", {})

                rows.append(
                    f"""
                    <tr>
                        <td><strong>{escape(service)}</strong></td>
                        <td>{escape(workload)}</td>
                        <td>{ttft.get('mean', 0):.2f}</td>
                        <td>{ttft.get('median', 0):.2f}</td>
                        <td>{ttft.get('p95', 0):.2f}</td>
                        <td>{latency.get('mean', 0):.2f}</td>
                        <td>{latency.get('median', 0):.2f}</td>
                        <td>{latency.get('p95', 0):.2f}</td>
                        <td>{metrics.get('error_rate', 0):.1f}%</td>
                    </tr>
                """
                )

        return f"""
            <table>
                <thead>
                    <tr>
                        <th>Service</th>
                        <th>Workload</th>
                        <th>TTFT Mean (ms)</th>
                        <th>TTFT Median (ms)</th>
                        <th>TTFT p95 (ms)</th>
                        <th>Latency Mean (ms)</th>
                        <th>Latency Median (ms)</th>
                        <th>Latency p95 (ms)</th>
                        <th>Error Rate</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        """
