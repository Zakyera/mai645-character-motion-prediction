#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="$PYTHON_BIN"
elif command -v python >/dev/null 2>&1; then
  PYTHON="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
else
  echo "Could not find python or python3 on PATH." >&2
  exit 1
fi

"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -r requirements.txt

mkdir -p results/tables results/plots

if [[ -n "${MAI645_DRIVE_ROOT:-}" ]]; then
  mkdir -p \
    "$MAI645_DRIVE_ROOT/processed" \
    "$MAI645_DRIVE_ROOT/models" \
    "$MAI645_DRIVE_ROOT/outputs" \
    "$MAI645_DRIVE_ROOT/results/tables" \
    "$MAI645_DRIVE_ROOT/results/plots"
fi

echo "MAI645 Colab setup complete."
