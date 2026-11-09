"""CLI commands for benchmarking tool."""

import asyncio
from pathlib import Path

import click
from openai import AsyncOpenAI
from rich.console import Console
from rich.table import Table

from ..config.settings import Settings
from ..metrics.collector import MetricsCollector
from ..reporting.csv_reporter import CSVReporter
from ..reporting.html_reporter import HTMLReporter
from ..reporting.provider_max_reporter import ProviderPerformanceReporter
from ..reporting.site_publisher import SitePublisher
from ..runner.benchmark import BenchmarkRunner

console = Console()

# Default location of a static site checkout, relative to the current directory.
DEFAULT_SITE_PATH = "."
SITE_BENCHMARKS_SUBDIR = "public/benchmarks"


def _generate_reports(settings: Settings, session_id: str) -> None:
    output_config = settings.get_output_config()
    collector = MetricsCollector(session_id, output_config.results_dir)
    session = collector.load_session(session_id)

    if output_config.csv_enabled:
        path = CSVReporter(output_config.results_dir).generate_summary_csv(session)
        console.print(f"  CSV summary: {path}")
    if output_config.html_enabled:
        path = HTMLReporter(output_config.results_dir).generate_report(session)
        console.print(f"  HTML report: {path}")
    path = ProviderPerformanceReporter(output_config.results_dir).generate_report(session)
    console.print(f"  Provider performance report: {path}")


@click.group()
def cli() -> None:
    """LLM Benchmarking Tool for General Compute."""


@cli.command()
@click.option("--providers", default="all", help="Comma-separated providers or 'all'")
@click.option("--models", default="all", help="Comma-separated models or 'all'")
@click.option("--workloads", default="all", help="Comma-separated workloads or 'all'")
@click.option("--iterations", default=None, type=int, help="Iterations (default: config)")
@click.option("--session-id", default=None, help="Custom session ID")
def run(
    providers: str,
    models: str,
    workloads: str,
    iterations: int | None,
    session_id: str | None,
) -> None:
    """Run benchmark tests."""
    try:
        settings = Settings()

        provider_list = ["all"] if providers == "all" else [s.strip() for s in providers.split(",")]
        model_list = ["all"] if models == "all" else [m.strip() for m in models.split(",")]
        if workloads == "all":
            workload_list = list(settings.get_all_workload_configs().keys())
        else:
            workload_list = [w.strip() for w in workloads.split(",")]

        if iterations is None:
            iterations = settings.get_benchmark_config().default_iterations

        runner = BenchmarkRunner(settings)
        session_id = asyncio.run(
            runner.run_benchmark(
                providers=provider_list,
                models=model_list,
                workloads=workload_list,
                iterations=iterations,
                session_id=session_id,
            )
        )

        console.print("[bold]Generating reports...[/bold]")
        _generate_reports(settings, session_id)
        console.print("\n[bold green]Done! ✓[/bold green]\n")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise click.Abort()


@cli.command()
@click.option("--provider", required=True, help="Provider to test (e.g. 'general_compute')")
@click.option("--model", required=True, help="Model to test (e.g. 'gpt-oss-120b')")
@click.option("--workload", default="ctx_256", help="Workload (default: ctx_256)")
@click.option("--iterations", default=5, type=int, help="Iterations (default: 5)")
def test(provider: str, model: str, workload: str, iterations: int) -> None:
    """Test a single provider/model/workload combination (for debugging)."""
    try:
        settings = Settings()
        runner = BenchmarkRunner(settings)
        asyncio.run(
            runner.run_single_test(
                provider_name=provider,
                model_name=model,
                workload_name=workload,
                iterations=iterations,
            )
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise click.Abort()


@cli.command(name="list-models")
@click.option("--provider", required=True, help="Provider whose catalog to list")
def list_models(provider: str) -> None:
    """Query a provider's live /models endpoint (verify model IDs)."""
    try:
        settings = Settings()
        cfg = settings.get_provider_config(provider)
        api_key = settings.get_api_key(provider)
        base_url = settings.get_base_url(provider)

        async def fetch() -> list[str]:
            client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=30.0)
            try:
                resp = await client.models.list()
                return [m.id for m in resp.data]
            finally:
                await client.close()

        ids = asyncio.run(fetch())
        table = Table(title=f"{cfg.name} — available models")
        table.add_column("Model ID", style="yellow")
        for mid in sorted(ids):
            table.add_row(mid)
        console.print()
        console.print(table)
        console.print()
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise click.Abort()


@cli.command()
@click.argument("session_id")
def report(session_id: str) -> None:
    """Generate reports for an existing session."""
    try:
        settings = Settings()
        console.print(f"\n[bold]Generating reports for session: {session_id}[/bold]\n")
        _generate_reports(settings, session_id)
        console.print("\n[bold green]Done! ✓[/bold green]\n")
    except FileNotFoundError:
        console.print(f"[bold red]Error:[/bold red] Session '{session_id}' not found")
        raise click.Abort()
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise click.Abort()


def _resolve_site_dir(site_path: str) -> Path:
    site_dir = (Path(site_path) / SITE_BENCHMARKS_SUBDIR).resolve()
    if not Path(site_path).exists():
        raise FileNotFoundError(
            f"Site path not found: {site_path} "
            f"(pass --site-path to point at your static site checkout)"
        )
    return site_dir


@cli.command()
@click.argument("session_id")
@click.option(
    "--site-path",
    default=DEFAULT_SITE_PATH,
    show_default=True,
    help="Path to a static site checkout",
)
@click.option("--label", default=None, help="Human-readable label shown on the site")
@click.option(
    "--default/--no-default",
    "make_default",
    default=None,
    help="Make this the default session shown on the site",
)
@click.option(
    "--copy-raw/--no-copy-raw",
    default=True,
    show_default=True,
    help="Also copy the raw CSV (for a download link)",
)
def publish(
    session_id: str,
    site_path: str,
    label: str | None,
    make_default: bool | None,
    copy_raw: bool,
) -> None:
    """Export a session to a static site directory."""
    try:
        settings = Settings()
        output_config = settings.get_output_config()
        collector = MetricsCollector(session_id, output_config.results_dir)
        session = collector.load_session(session_id)

        site_dir = _resolve_site_dir(site_path)
        publisher = SitePublisher(settings, output_config.results_dir)
        result = publisher.publish(
            session,
            site_dir,
            label=label,
            make_default=make_default,
            copy_raw=copy_raw,
        )

        console.print(
            f"\n[bold green]Published[/bold green] session '{session_id}' "
            f"({result['row_count']} stat rows)"
        )
        console.print(f"  JSON:     {result['session_file']}")
        if result["raw_file"]:
            console.print(f"  Raw CSV:  {result['raw_file']}")
        console.print(f"  Manifest: {result['manifest']}")
        if result["is_default"]:
            console.print("  [cyan]This is now the default session on the site.[/cyan]")
        console.print("\nCommit & deploy your static site to make it live.\n")
    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise click.Abort()
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise click.Abort()


@cli.command()
@click.argument("session_id")
@click.option(
    "--site-path",
    default=DEFAULT_SITE_PATH,
    show_default=True,
    help="Path to a static site checkout",
)
def unpublish(session_id: str, site_path: str) -> None:
    """Remove a published session from a static site directory."""
    try:
        settings = Settings()
        site_dir = _resolve_site_dir(site_path)
        publisher = SitePublisher(settings, settings.get_output_config().results_dir)
        result = publisher.unpublish(session_id, site_dir)

        if result["was_present"]:
            console.print(
                f"\n[bold green]Unpublished[/bold green] '{session_id}'. "
                f"{result['remaining']} session(s) remain.\n"
            )
        else:
            console.print(f"\n[yellow]Session '{session_id}' was not published.[/yellow]\n")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise click.Abort()


@cli.command(name="list-sessions")
def list_sessions() -> None:
    """List all benchmark sessions."""
    try:
        settings = Settings()
        output_config = settings.get_output_config()
        sessions = MetricsCollector.list_sessions(output_config.results_dir)
        if not sessions:
            console.print("\n[yellow]No sessions found.[/yellow]\n")
            return
        table = Table(title="Benchmark Sessions")
        table.add_column("Session ID", style="cyan")
        table.add_column("Files", style="green")
        for sid in sessions:
            base = Path(output_config.results_dir) / f"session_{sid}"
            files = [
                label
                for label, suffix in [("CSV", "_raw.csv"), ("HTML", "_report.html")]
                if Path(f"{base}{suffix}").exists()
            ]
            table.add_row(sid, ", ".join(files))
        console.print()
        console.print(table)
        console.print()
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise click.Abort()


@cli.command()
def providers() -> None:
    """List configured providers."""
    try:
        settings = Settings()
        table = Table(title="Providers")
        table.add_column("Provider", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Base URL", style="yellow")
        table.add_column("Enabled", style="magenta")
        for name, cfg in settings.get_all_provider_configs().items():
            try:
                url = settings.get_base_url(name)
            except ValueError:
                url = f"${cfg.base_url_env}"
            table.add_row(name, cfg.name, url, "✓" if cfg.enabled else "✗")
        console.print()
        console.print(table)
        console.print()
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise click.Abort()


@cli.command()
def models() -> None:
    """List configured models and which providers host them."""
    try:
        settings = Settings()
        table = Table(title="Models")
        table.add_column("Model", style="cyan")
        table.add_column("Reasoning", style="magenta")
        table.add_column("Hosted by", style="green")
        for name, cfg in settings.get_all_model_configs().items():
            table.add_row(name, "✓" if cfg.reasoning else "", ", ".join(cfg.ids.keys()))
        console.print()
        console.print(table)
        console.print()
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise click.Abort()


@cli.command()
def workloads() -> None:
    """List available workloads."""
    try:
        settings = Settings()
        table = Table(title="Workloads")
        table.add_column("Workload", style="cyan")
        table.add_column("Tokens", style="green")
        table.add_column("Steps", style="magenta")
        table.add_column("Template", style="yellow")
        table.add_column("Description", style="white")
        for name, cfg in settings.get_all_workload_configs().items():
            tokens = str(cfg.tokens)
            if cfg.kind == "agent" and cfg.steps > 1:
                tokens += f" (+{cfg.step_tokens}/step)"
            table.add_row(name, tokens, str(cfg.steps), cfg.template, cfg.description)
        console.print()
        console.print(table)
        console.print()
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise click.Abort()


if __name__ == "__main__":
    cli()
