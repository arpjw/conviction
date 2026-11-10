#!/usr/bin/env bash
# SYSTEM_PY setup: keep the image's torch (2.9.1+cu128) and install the pinned stack on top.
set -euo pipefail
cd /workspace/conviction
pip install -q uv
uv pip install --system --break-system-packages "torch==2.9.1+cu128" --extra-index-url https://download.pytorch.org/whl/cu128 -q \
  transformers==5.22.2 trl==1.17.1 peft==0.21.2 accelerate==1.15.0 datasets==5.1.0 pyyaml numpy pytest hypothesis matplotlib pandas
python3 -c "import torch,transformers,trl,peft; print('SETUP_OK', torch.__version__, torch.cuda.is_available(), transformers.__version__, trl.__version__, peft.__version__)"
