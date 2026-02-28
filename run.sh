#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/config.json}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python not found: $PYTHON_BIN" >&2
  exit 1
fi

echo "Installing Python requirements..."
"$PYTHON_BIN" -m pip install neo4j pandas

echo "Running benchmark with config: $CONFIG_PATH"
"$PYTHON_BIN" "$ROOT_DIR/benchmark.py" --config "$CONFIG_PATH" "$@"
