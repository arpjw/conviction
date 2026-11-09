#!/usr/bin/env bash
# Usage on pod: bash scripts/remote_phase.sh val|baselines
set -euo pipefail
cd /workspace/conviction
export HF_HOME=/workspace/hf PATH=$HOME/.local/bin:$PATH TOKENIZERS_PARALLELISM=false
if [ ! -d .venv ]; then pip install -q uv; uv sync --frozen --extra train --extra analysis; fi
if [ ! -f data/conviction_bench/v0.1/val.jsonl ]; then
  uv run python -m conviction.data.cli; uv run python -m conviction.data.adversarial > /dev/null
fi
uv run python -c "import json;m=json.load(open('data/conviction_bench/v0.1/manifest.json'));print('HASHES', {k:v['sha256'][:12] for k,v in m['splits'].items()})"
case "$1" in
  val)
    uv run pytest -q -x
    uv run python -m conviction.evaluation.runner --condition base --splits val --batch-size 64 --max-new-tokens 200 --out results/calibration/val_full_reason_first_v2 ;;
  baselines)
    S=test_id,test_template,test_structural,test_adversarial
    uv run python -m conviction.evaluation.runner --condition base   --splits $S --natural --batch-size 64 --max-new-tokens 200 --out results/E0_base
    uv run python -m conviction.evaluation.runner --condition guided --splits $S --natural --batch-size 64 --max-new-tokens 200 --out results/E1_guided ;;
esac
tar czf /workspace/results_$1.tgz results
echo PHASE_DONE_$1
