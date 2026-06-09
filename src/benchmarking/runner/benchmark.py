"""Benchmark runner for orchestrating benchmark execution."""

import asyncio
import uuid
from typing import Dict, List, Optional

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from ..clients.openai_compatible import OpenAICompatibleClient
from ..config.settings import Settings
from ..metrics.collector import MetricsCollector
from ..workloads.generator import PromptGenerator


class BenchmarkRunner:
    """Orchestrate benchmark execution across providers, models and workloads."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.console = Console()
        self.prompt_generator = PromptGenerator()

    def _make_client(self, provider_name: str, provider_model_id: str) -> OpenAICompatibleClient:
        provider = self.settings.get_provider_config(provider_name)
        return OpenAICompatibleClient(
            api_key=self.settings.get_api_key(provider_name),
            model=provider_model_id,
            service_name=provider.name,
            base_url=self.settings.get_base_url(provider_name),
            timeout=self.settings.get_benchmark_config().timeout_seconds,
            uses_max_completion_tokens=provider.uses_max_completion_tokens,
            extra_body=provider.extra_body,
        )

    def _resolve_providers(self, providers_arg: List[str]) -> List[str]:
        all_enabled = self.settings.get_enabled_providers()
        if providers_arg == ["all"]:
            return list(all_enabled.keys())
        return providers_arg

    def _resolve_models(self, models_arg: List[str]) -> List[str]:
        all_models = self.settings.get_all_model_configs()
        if models_arg == ["all"]:
            return list(all_models.keys())
        return models_arg

    @staticmethod
    def _vary_prompt(prompt: str, iteration: int) -> str:
        """Prepend a unique nonce so each request defeats prefix caching.

        The same nonce is used for every provider in a given iteration, so the
        comparison stays fair while no provider benefits from cache hits.
        """
        return f"[run {iteration:05d}] {prompt}"

    async def run_benchmark(
        self,
        providers: List[str],
        models: List[str],
        workloads: List[str],
        iterations: int,
        session_id: Optional[str] = None,
    ) -> str:
        """Run the full benchmark suite.

        For each (workload, model) the providers are interleaved per iteration
        so every provider is measured in the same wall-clock window — removing
        time-of-day / network-drift bias from the comparison.
        """
        if session_id is None:
            session_id = str(uuid.uuid4())[:8]

        provider_list = self._resolve_providers(providers)
        model_list = self._resolve_models(models)
        model_configs = self.settings.get_all_model_configs()

        bench_cfg = self.settings.get_benchmark_config()
        gen_params = self.settings.get_generation_params()
        output_config = self.settings.get_output_config()
        collector = MetricsCollector(session_id, output_config.results_dir)

        self.console.print(f"\n[bold blue]Starting Benchmark Session: {session_id}[/bold blue]")
        self.console.print(f"Providers: {', '.join(provider_list)}")
        self.console.print(f"Models: {', '.join(model_list)}")
        self.console.print(f"Workloads: {', '.join(workloads)}")
        self.console.print(f"Iterations: {iterations} (+{bench_cfg.warmup_iterations} warmup)\n")

        # Build the matrix of (workload, model, [providers hosting it]).
        cells = []
        for workload_name in workloads:
            for model_name in model_list:
                model_cfg = model_configs[model_name]
                hosts = [p for p in provider_list if p in model_cfg.ids]
                missing = [p for p in provider_list if p not in model_cfg.ids]
                if missing:
                    self.console.print(
                        f"[yellow]Note:[/yellow] {model_name} has no ID for "
                        f"{', '.join(missing)} — skipping there."
                    )
                if hosts:
                    cells.append((workload_name, model_name, model_cfg, hosts))

        total_tasks = sum(len(hosts) * iterations for _, _, _, hosts in cells)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            console=self.console,
        ) as progress:
            main_task = progress.add_task("[cyan]Running benchmarks...", total=total_tasks)

            for workload_name, model_name, model_cfg, hosts in cells:
                workload_config = self.settings.get_workload_config(workload_name)
                base_prompt = self.prompt_generator.generate_prompt(
                    target_tokens=workload_config.tokens,
                    template=workload_config.template,
                )

                clients: Dict[str, OpenAICompatibleClient] = {
                    p: self._make_client(p, model_cfg.ids[p]) for p in hosts
                }

                try:
                    # Warm-up (not recorded): warms TCP/TLS + provider routing.
                    for _ in range(bench_cfg.warmup_iterations):
                        for provider_name, client in clients.items():
                            try:
                                await client.benchmark_request(
                                    prompt=self._vary_prompt(base_prompt, -1),
                                    max_tokens=min(64, gen_params.max_tokens),
                                    temperature=gen_params.temperature,
                                    top_p=gen_params.top_p,
                                    workload_type=workload_name,
                                    model_name=model_name,
                                    max_attempts=1,
                                )
                            except Exception:
                                pass

                    for iteration in range(iterations):
                        prompt = (
                            self._vary_prompt(base_prompt, iteration)
                            if bench_cfg.vary_prompt
                            else base_prompt
                        )
                        # Interleave providers within each iteration.
                        for provider_name, client in clients.items():
                            progress.update(
                                main_task,
                                description=(
                                    f"[cyan]{model_name} · {provider_name} · "
                                    f"{workload_name} ({iteration + 1}/{iterations})"
                                ),
                            )
                            result = await client.benchmark_request(
                                prompt=prompt,
                                max_tokens=gen_params.max_tokens,
                                temperature=gen_params.temperature,
                                top_p=gen_params.top_p,
                                workload_type=workload_name,
                                model_name=model_name,
                                max_attempts=bench_cfg.retry_attempts,
                                min_retry_wait=bench_cfg.min_retry_wait,
                                max_retry_wait=bench_cfg.max_retry_wait,
                            )
                            collector.save_result(result, iteration)
                            progress.update(main_task, advance=1)
                            await asyncio.sleep(bench_cfg.request_delay_seconds)
                finally:
                    for client in clients.values():
                        await client.close()

        self.console.print("\n[bold green]✓ Benchmark completed![/bold green]")
        self.console.print(f"Results saved to: {collector.csv_path}\n")
        return session_id

    async def run_single_test(
        self,
        provider_name: str,
        model_name: str,
        workload_name: str,
        iterations: int = 5,
    ) -> None:
        """Quick connectivity/sanity test for one provider+model+workload."""
        self.console.print(
            f"\n[bold yellow]Test: {provider_name} · {model_name} · {workload_name}[/bold yellow]\n"
        )

        model_cfg = self.settings.get_all_model_configs()[model_name]
        if provider_name not in model_cfg.ids:
            self.console.print(
                f"[bold red]Error:[/bold red] {model_name} has no ID for {provider_name}"
            )
            return

        gen_params = self.settings.get_generation_params()
        bench_cfg = self.settings.get_benchmark_config()
        workload_config = self.settings.get_workload_config(workload_name)
        prompt = self.prompt_generator.generate_prompt(
            target_tokens=workload_config.tokens,
            template=workload_config.template,
        )

        client = self._make_client(provider_name, model_cfg.ids[provider_name])
        results = []
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                console=self.console,
            ) as progress:
                task = progress.add_task("Testing...", total=iterations)
                for i in range(iterations):
                    results.append(
                        await client.benchmark_request(
                            prompt=self._vary_prompt(prompt, i),
                            max_tokens=gen_params.max_tokens,
                            temperature=gen_params.temperature,
                            top_p=gen_params.top_p,
                            workload_type=workload_name,
                            model_name=model_name,
                            max_attempts=bench_cfg.retry_attempts,
                            min_retry_wait=bench_cfg.min_retry_wait,
                            max_retry_wait=bench_cfg.max_retry_wait,
                        )
                    )
                    progress.update(task, advance=1)
                    await asyncio.sleep(bench_cfg.request_delay_seconds)
        finally:
            await client.close()

        successful = [r for r in results if r.error is None]
        failed = [r for r in results if r.error is not None]
        self.console.print(f"\nSuccessful: {len(successful)}/{iterations}")
        self.console.print(f"Failed: {len(failed)}/{iterations}\n")

        if successful:
            ttfts = [r.ttft * 1000 for r in successful if r.ttft is not None]
            lats = [r.end_to_end_latency * 1000 for r in successful]
            tps = [r.tokens_per_sec for r in successful if r.tokens_per_sec]
            if ttfts:
                self.console.print(f"Average TTFT: {sum(ttfts) / len(ttfts):.2f} ms")
            self.console.print(f"Average Latency: {sum(lats) / len(lats):.2f} ms")
            if tps:
                self.console.print(f"Average Output Speed: {sum(tps) / len(tps):.2f} tok/s")

        if failed:
            self.console.print("\n[bold red]Errors:[/bold red]")
            for i, r in enumerate(failed, 1):
                self.console.print(f"{i}. {r.error}")
        self.console.print()
