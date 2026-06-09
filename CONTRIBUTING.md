# Contributing

Thanks for improving GC Benchmarking.

## Development Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

Create `.env` from `.env.example` only when you need live provider calls. Unit
tests should not require provider credentials.

## Checks

Run these before opening a pull request:

```bash
pytest
ruff check src tests
mypy src
```

Format changed Python files with:

```bash
black src tests
```

## Pull Request Guidance

- Keep generated benchmark output out of commits.
- Do not include API keys, provider account identifiers, or private benchmark
  data.
- Add or update tests for changes to metrics, config loading, CLI behavior, or
  report generation.
- When changing benchmark methodology, document the tradeoff in `README.md`.
