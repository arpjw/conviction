#!/usr/bin/env bash
# Pod setup (run inside the RunPod container from the repo root).
set -euo pipefail
pip install -q uv
uv sync --extra train --extra analysis
uv run python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
uv run pytest -q
