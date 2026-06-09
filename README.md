# GC Benchmarking

LLM inference benchmarking for OpenAI-compatible providers. The tool runs the
same logical model across every enabled provider that has a configured model ID,
then reports time to first token, end-to-end latency, output throughput, token
counts, retry attempts, and error rate.

The default configuration compares General Compute with OpenRouter for the same
model families. Add or disable providers in `config/config.yaml` without
changing code.

## Features

- Same-model provider comparisons
- Provider interleaving within each iteration to reduce time-window bias
- Warm-up requests that are discarded from metrics
- Prompt variation to reduce provider-side cache effects
- Streaming TTFT measurement, including reasoning-token streams
- Incremental raw CSV writes so interrupted runs keep completed samples
- CSV, HTML, and static-site JSON report generation

## Installation

Use Python 3.10 or newer.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

Or run the local setup script:

```bash
./setup.sh
```

## Configuration

Create a local `.env` from the example and add provider keys:

```bash
cp .env.example .env
```

Required by the default config:

```bash
GENERAL_COMPUTE_API_KEY=your_general_compute_api_key_here
GENERAL_COMPUTE_BASE_URL=https://api.generalcompute.com/v1
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

A provider's endpoint can be set inline in `config/config.yaml` via `base_url`,
or kept out of version control by naming an env var with `base_url_env` (as the
default `general_compute` provider does). OpenRouter uses a public `base_url`.

`.env` and benchmark outputs are intentionally ignored by Git. Do not commit
real API keys or generated result files.

By default, the CLI loads `config/config.yaml` from the current working
directory when present. Otherwise, it falls back to the packaged default config.
Set `CONFIG_FILE=/path/to/config.yaml` to use an explicit file.

## Usage

List configured providers, models, and workloads:

```bash
benchmark providers
benchmark models
benchmark workloads
```

Run a quick connectivity test:

```bash
benchmark test --provider general_compute --model gpt-oss-120b --workload ctx_256 --iterations 1
```

Run a benchmark:

```bash
benchmark run --providers general_compute,openrouter --models gpt-oss-120b --workloads ctx_256,ctx_1k --iterations 5
```

Run all enabled providers, models, and workloads:

```bash
benchmark run --iterations 50
```

Regenerate reports for an existing session:

```bash
benchmark report <session-id>
```

List local sessions:

```bash
benchmark list-sessions
```

## Workloads

The default workloads are context-size sweeps:

- `ctx_256`: 256 input tokens
- `ctx_1k`: 1,024 input tokens
- `ctx_4k`: 4,096 input tokens
- `ctx_16k`: 16,384 input tokens
- `ctx_64k`: 65,536 input tokens
- `ctx_128k`: 131,072 input tokens

Token counts are approximate because prompts are generated with `tiktoken`
`cl100k_base`, not each model provider's native tokenizer.

## Outputs

Results are written under `results/`:

- `session_<id>_raw.csv`: one row per request
- `session_<id>_summary.csv`: aggregate statistics by model, provider, and workload
- `session_<id>_report.html`: general HTML charts and tables
- `session_<id>_provider_performance.html`: provider performance charts

HTML reports load Plotly from the public CDN. Use the CSV outputs if you need a
fully offline artifact.

## Static Site Export

Export a completed session as pre-aggregated JSON for a static site:

```bash
benchmark publish <session-id> --site-path ../my-site --label "June benchmark"
```

This writes files under `../my-site/public/benchmarks/`:

- `manifest.json`
- `<session-id>.json`
- `<session-id>_raw.csv` unless `--no-copy-raw` is passed

Remove a published session:

```bash
benchmark unpublish <session-id> --site-path ../my-site
```

## Methodology Notes

Comparisons are meaningful only within the same logical model. OpenRouter is an
aggregator, so its latency can include routing overhead and can vary by selected
backend. Review provider routing settings in `config/config.yaml` before
publishing benchmark claims.

The tool measures output throughput after TTFT, so decode speed is separated
from queueing and prompt-processing overhead. Retries are limited to transient
errors; failed attempts and backoff sleeps do not inflate successful-attempt
latency metrics.

## Publishing Benchmark Results

If you intend to publish results produced with this tool — especially
comparisons that involve General Compute — please reach out to us at
**jason@generalcompute.com** beforehand so we can help validate the
methodology and configuration.

Provider performance is sensitive to setup: OpenRouter routing settings,
model-version drift, tokenizer approximations, region, and concurrency can all
move the numbers. A quick review helps ensure published comparisons are
apples-to-apples and accurately attributed.

## Development

```bash
pytest
ruff check src tests
mypy src
```

Format code:

```bash
black src tests
```

## Security

Please do not open public issues with secrets, API keys, private benchmark data,
or unpublished provider credentials. See `SECURITY.md` for reporting guidance.

## License

MIT. See `LICENSE`.
