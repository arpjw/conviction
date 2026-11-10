#!/usr/bin/env bash
# Usage on pod: bash scripts/remote_phase.sh val|baselines
set -euo pipefail
cd /workspace/conviction
export HF_HOME=/workspace/hf PATH=$HOME/.local/bin:$PATH TOKENIZERS_PARALLELISM=false CONVICTION_REQUIRE_CUDA=1
# SYSTEM_PY=1: use the image's preinstalled torch (deps installed on top with uv pip --system);
# useful on hosts with slow links to the PyTorch index.
if [ "${SYSTEM_PY:-0}" = "1" ]; then
  export PYTHONPATH=$PWD; RUN="python3"
else
  command -v uv >/dev/null || pip install -q uv
  uv sync --frozen --extra train --extra analysis
  RUN="uv run python"
fi
$RUN -c "import torch; assert torch.cuda.is_available(), torch.__version__; print('CUDA OK', torch.__version__, torch.cuda.get_device_name(0))"
if [ ! -f data/conviction_bench/v0.1/val.jsonl ]; then
  $RUN -m conviction.data.cli; $RUN -m conviction.data.adversarial > /dev/null
fi
$RUN -c "import json;m=json.load(open('data/conviction_bench/v0.1/manifest.json'));print('HASHES', {k:v['sha256'][:12] for k,v in m['splits'].items()})"
case "$1" in
  val)
    $RUN -m pytest -q -x
    $RUN -m conviction.evaluation.runner --condition base --splits val --batch-size 64 --max-new-tokens 200 --out results/calibration/val_full_reason_first_v2 ;;
  baselines)
    S=test_id,test_template,test_structural,test_adversarial
    $RUN -m conviction.evaluation.runner --condition base   --splits $S --natural --batch-size 64 --max-new-tokens 200 --out results/E0_base
    $RUN -m conviction.evaluation.runner --condition guided --splits $S --natural --batch-size 64 --max-new-tokens 200 --out results/E1_guided ;;
  sft)
    # 10-step smoke test (aborts the chain on any error), then the full run, then E2 eval
    $RUN -m conviction.training.train_sft --config configs/sft.yaml --override output_dir=checkpoints/sft_smoke training.max_steps=10 n_val=32 training.eval_steps=5 training.save_steps=1000
    echo SMOKE_OK
    $RUN -m conviction.training.train_sft --config configs/sft.yaml
    echo TRAIN_OK
    mkdir -p results/E2_sft && cp checkpoints/sft/train_info.json results/E2_sft/ && cp checkpoints/sft_smoke/train_info.json results/E2_sft/smoke_train_info.json
    $RUN -m conviction.evaluation.runner --condition base --adapter checkpoints/sft/final --splits val --batch-size 64 --max-new-tokens 200 --out results/E2_sft/val
    $RUN -m conviction.evaluation.runner --condition base --adapter checkpoints/sft/final --splits test_id,test_template,test_structural,test_adversarial --natural --batch-size 64 --max-new-tokens 200 --out results/E2_sft
    tar czf /workspace/adapter_sft.tgz -C checkpoints/sft final ;;
  grpo)
    test -f checkpoints/sft/final/adapter_model.safetensors
    $RUN -m conviction.training.train_grpo --config configs/grpo.yaml --override output_dir=checkpoints/grpo_smoke grpo.max_steps=3 grpo.save_steps=1000
    echo SMOKE_OK
    $RUN -m conviction.training.train_grpo --config configs/grpo.yaml
    echo TRAIN_OK
    mkdir -p results/E3_grpo && cp checkpoints/grpo/train_info.json results/E3_grpo/ && cp checkpoints/grpo_smoke/train_info.json results/E3_grpo/smoke_train_info.json
    A=checkpoints/sft/final,checkpoints/grpo/final
    $RUN -m conviction.evaluation.runner --condition base --adapter $A --splits val --batch-size 64 --max-new-tokens 200 --out results/E3_grpo/val
    $RUN -m conviction.evaluation.runner --condition base --adapter $A --splits test_id,test_template,test_structural,test_adversarial --natural --batch-size 64 --max-new-tokens 200 --out results/E3_grpo
    tar czf /workspace/adapter_grpo.tgz -C checkpoints/grpo final ;;
  sft_control)
    test -f checkpoints/sft/final/adapter_model.safetensors
    $RUN -m conviction.training.train_sft --config configs/sft_control.yaml
    echo TRAIN_OK
    mkdir -p results/E3c_sft_control && cp checkpoints/sft_control/train_info.json results/E3c_sft_control/
    A=checkpoints/sft/final,checkpoints/sft_control/final
    $RUN -m conviction.evaluation.runner --condition base --adapter $A --splits val --batch-size 64 --max-new-tokens 200 --out results/E3c_sft_control/val
    $RUN -m conviction.evaluation.runner --condition base --adapter $A --splits test_id,test_template,test_structural,test_adversarial --natural --batch-size 64 --max-new-tokens 200 --out results/E3c_sft_control
    tar czf /workspace/adapter_sft_control.tgz -C checkpoints/sft_control final ;;
  grpo_base)
    $RUN -m conviction.training.train_grpo --config configs/grpo_from_base.yaml --override output_dir=checkpoints/grpo_base_smoke grpo.max_steps=3 grpo.save_steps=1000
    mkdir -p results/E3b_grpo_base && cp checkpoints/grpo_base_smoke/train_info.json results/E3b_grpo_base/smoke_train_info.json
    $RUN scripts/grpo_gate.py checkpoints/grpo_base_smoke/train_info.json
    echo SMOKE_OK
    $RUN -m conviction.training.train_grpo --config configs/grpo_from_base.yaml
    echo TRAIN_OK
    cp checkpoints/grpo_base/train_info.json results/E3b_grpo_base/
    A=checkpoints/grpo_base/final
    $RUN -m conviction.evaluation.runner --condition base --adapter $A --splits val --batch-size 64 --max-new-tokens 200 --out results/E3b_grpo_base/val
    $RUN -m conviction.evaluation.runner --condition base --adapter $A --splits test_id,test_template,test_structural,test_adversarial --natural --batch-size 64 --max-new-tokens 200 --out results/E3b_grpo_base
    tar czf /workspace/adapter_grpo_base.tgz -C checkpoints/grpo_base final ;;
  grpo_base_s2)
    # E6: second seed of E3b. seed=2 changes prompt subset/order (same GRPO pool), LoRA init and sampling.
    $RUN -m conviction.training.train_grpo --config configs/grpo_from_base.yaml --override seed=2 output_dir=checkpoints/grpo_base_s2
    echo TRAIN_OK
    mkdir -p results/E6_grpo_base_s2 && cp checkpoints/grpo_base_s2/train_info.json results/E6_grpo_base_s2/
    A=checkpoints/grpo_base_s2/final
    $RUN -m conviction.evaluation.runner --condition base --adapter $A --splits val --batch-size 64 --max-new-tokens 200 --out results/E6_grpo_base_s2/val
    $RUN -m conviction.evaluation.runner --condition base --adapter $A --splits test_id,test_template,test_structural,test_adversarial --natural --batch-size 64 --max-new-tokens 200 --out results/E6_grpo_base_s2
    tar czf /workspace/adapter_grpo_base_s2.tgz -C checkpoints/grpo_base_s2 final ;;
  bench)
    # Multi-model baselines. BENCH_MODELS="org/name[:batch],..." ; each model: 2-group smoke, then full eval.
    # A failing model is logged (MODEL_FAIL) and skipped; its HF cache is deleted afterwards to save disk.
    S=test_id,test_template,test_structural,test_adversarial
    IFS=',' read -ra MS <<< "$BENCH_MODELS"
    for spec in "${MS[@]}"; do
      m=${spec%%:*}; bs=${spec#*:}; [ "$bs" = "$spec" ] && bs=32
      name=B_$(echo "${m#*/}" | tr '.' '_')
      echo "=== MODEL_START $m (batch $bs) $(date -u +%H:%M:%S)"
      if $RUN -m conviction.evaluation.runner --model "$m" --condition base --splits test_id --limit-groups 2 --batch-size $bs --max-new-tokens 200 --n-boot 50 --out results/smoke/$name \
         && $RUN -m conviction.evaluation.runner --model "$m" --condition base --splits $S --natural --batch-size $bs --max-new-tokens 200 --out results/$name; then
        echo "=== MODEL_DONE $m $(date -u +%H:%M:%S)"
        tar czf /workspace/results_$name.tgz results/$name results/smoke/$name
      else
        echo "=== MODEL_FAIL $m $(date -u +%H:%M:%S)"
      fi
      rm -rf "$HF_HOME/hub/models--${m//\//--}"
    done ;;
esac
tar czf /workspace/results_$1.tgz results
echo PHASE_DONE_$1
