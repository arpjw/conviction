#!/usr/bin/env bash
# Validation-only prompt-format calibration, run on a GPU pod.
set -euo pipefail
cd /workspace/conviction
export HF_HOME=/workspace/hf PATH=$HOME/.local/bin:$PATH TOKENIZERS_PARALLELISM=false
pip install -q uv
uv sync --frozen --extra train --extra analysis
uv run pytest -q -x
uv run python -m conviction.data.cli
uv run python -m conviction.data.adversarial > /dev/null
uv run python -c "import json;m=json.load(open('data/conviction_bench/v0.1/manifest.json'));print('HASHES', {k:v['sha256'][:12] for k,v in m['splits'].items()})"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for c in base_decision_first base_reason_first; do
  uv run python -m conviction.evaluation.runner --model Qwen/Qwen3-1.7B --condition $c --splits val \
     --batch-size 64 --max-new-tokens 200 --n-boot 2000 --out results/calibration/val_full_$c
done
tar czf /workspace/results.tgz results
echo ALLDONE
