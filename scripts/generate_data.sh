#!/usr/bin/env bash
set -euo pipefail
uv run python -m conviction.data.cli --config configs/base.yaml
uv run python -m conviction.data.adversarial
