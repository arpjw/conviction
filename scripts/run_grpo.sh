#!/usr/bin/env bash
set -euo pipefail
uv run python -m conviction.training.train_grpo --config configs/grpo.yaml "$@"
uv run python -m conviction.evaluation.runner --adapter checkpoints/sft/final,checkpoints/grpo/final --splits test_id,test_template,test_structural,test_adversarial --natural --batch-size 64 --out results/E3_grpo
