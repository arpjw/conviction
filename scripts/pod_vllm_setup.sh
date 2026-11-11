#!/usr/bin/env bash
# vLLM in its own venv (it pins its own torch); the runner is imported from the repo via PYTHONPATH.
set -euo pipefail
command -v uv >/dev/null || pip install -q uv
uv venv -q --python 3.12 /workspace/vllm-venv
uv pip install -q --python /workspace/vllm-venv/bin/python vllm==0.33.0 pyyaml numpy
/workspace/vllm-venv/bin/python -c "import vllm, transformers, torch; print('VLLM_OK', vllm.__version__, transformers.__version__, torch.__version__, torch.cuda.is_available())"
