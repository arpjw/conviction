#!/usr/bin/env bash
set -euo pipefail
SPLITS=test_id,test_template,test_structural,test_adversarial
uv run python -m conviction.evaluation.runner --condition base   --splits $SPLITS --natural --batch-size 64 --out results/E0_base
uv run python -m conviction.evaluation.runner --condition guided --splits $SPLITS --natural --batch-size 64 --out results/E1_guided
