#!/usr/bin/env bash
# Setup script for simpledspy development environment
# Creates a virtual environment and installs required dependencies.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-.venv}"

python3 -m venv "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install -e "$ROOT_DIR"
# Install development dependencies
pip install pytest black mypy twine

echo "Environment setup complete. Activate with: source $VENV_DIR/bin/activate"
