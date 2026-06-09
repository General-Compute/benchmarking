#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"

echo "GC Benchmarking setup"
echo

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

echo
echo "Setup complete."
echo
echo "Common commands:"
echo "  ./benchmark providers"
echo "  ./benchmark models"
echo "  ./benchmark workloads"
echo "  ./benchmark test --provider general_compute --model gpt-oss-120b --workload ctx_256 --iterations 1"
echo "  ./benchmark run --providers general_compute,openrouter --models gpt-oss-120b --workloads ctx_256 --iterations 5"
echo
echo "Before live benchmarks, create .env from .env.example and add provider API keys."
