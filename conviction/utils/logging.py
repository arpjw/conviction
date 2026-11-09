"""Small run-logging helpers: environment capture and a GPU-memory callback."""

from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path


def environment_info() -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform(), "time": time.time()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gb"] = torch.cuda.get_device_properties(0).total_memory / 2**30
    except ImportError:
        pass
    for pkg in ("transformers", "trl", "peft", "accelerate", "datasets"):
        try:
            info[pkg] = __import__(pkg).__version__
        except ImportError:
            pass
    try:
        info["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                                     stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        info["git_commit"] = None
    return info


def write_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(path)


def memory_callback():
    from transformers import TrainerCallback

    class GPUMemoryCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            import torch
            if logs is not None and torch.cuda.is_available():
                logs["gpu_mem_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
                logs["gpu_mem_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)

    return GPUMemoryCallback()
