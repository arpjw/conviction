#!/usr/bin/env bash
# GPU smoke test: tiny SFT + GRPO + eval on the real model; measures memory and speed.
set -euo pipefail
OUT=checkpoints/smoke
uv run python -m conviction.training.train_sft --override output_dir=$OUT/sft n_train=320 n_val=64 training.max_steps=20 training.eval_steps=10 training.save_steps=20
uv run python -m conviction.training.train_grpo --override init_adapter=$OUT/sft/final output_dir=$OUT/grpo n_prompts=64 grpo.max_steps=5 grpo.save_steps=5 grpo.log_completions=true
uv run python -m conviction.evaluation.runner --adapter $OUT/sft/final,$OUT/grpo/final --limit-groups 5 --splits test_id --out results/smoke_eval
