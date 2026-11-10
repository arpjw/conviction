#!/usr/bin/env bash
# SYSTEM_PY setup: keep the image's torch (2.9.1+cu128) and install the pinned stack on top.
set -euo pipefail
cd /workspace/conviction
pip install -q uv
uv pip install --system --break-system-packages "torch==2.9.1+cu128" --extra-index-url https://download.pytorch.org/whl/cu128 -q \
  transformers==5.22.2 trl==1.17.1 peft==0.21.2 accelerate==1.15.0 datasets==5.1.0 pyyaml numpy pytest hypothesis matplotlib pandas
python3 -c "import torch,transformers,trl,peft; print('SETUP_OK', torch.__version__, torch.cuda.is_available(), transformers.__version__, trl.__version__, peft.__version__)"
# Optional fast kernels for linear-attention / Mamba hybrids (Qwen3.5, Nemotron-H). Best effort: models still run
# (slowly) on the reference PyTorch path if these fail to install.
if [ "${KERNELS:-0}" = "1" ]; then
  uv pip install --system --break-system-packages -q flash-linear-attention && echo FLA_OK || echo FLA_FAIL
  timeout 1500 uv pip install --system --break-system-packages --no-build-isolation -q causal-conv1d mamba-ssm \
    && echo MAMBA_OK || echo MAMBA_FAIL
  python3 -c "import fla" 2>/dev/null && echo FLA_IMPORT_OK
  python3 -c "import causal_conv1d, mamba_ssm" 2>/dev/null && echo MAMBA_IMPORT_OK
fi
