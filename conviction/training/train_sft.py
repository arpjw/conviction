"""E2: supervised fine-tuning with LoRA on reference responses.

    python -m conviction.training.train_sft --config configs/sft.yaml [--override key=value ...]

Completion-only loss on the reference JSON response. Prompts are rendered with
exactly the same chat template / system prompt (condition "base") used in eval.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

from conviction.data.splits import load_split
from conviction.evaluation.runner import render_prompt
from conviction.training.data import select
from conviction.utils.logging import environment_info, memory_callback, write_json


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    for o in overrides:
        k, v = o.split("=", 1)
        d = cfg
        *path, last = k.split(".")
        for p in path:
            d = d.setdefault(p, {})
        d[last] = yaml.safe_load(v)
    return cfg


def to_rows(eps: list[dict], tok, condition: str) -> list[dict]:
    return [{"prompt": render_prompt(tok, e["messages"], condition),
             "completion": e["reference_response"],
             "episode_id": e["episode_id"]} for e in eps]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sft.yaml")
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args(argv)
    cfg = apply_overrides(yaml.safe_load(open(args.config)), args.override)

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from trl import SFTConfig, SFTTrainer

    from conviction.evaluation.runner import require_cuda
    require_cuda()
    set_seed(cfg["seed"])
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg.get("revision"))
    tok.padding_side = "right"

    train_eps = select(load_split(Path(cfg["data_dir"]), "train"), "sft", cfg["n_train"], cfg["seed"],
                       one_per_group=cfg.get("one_per_group", False))
    val_eps: list[dict] = []
    if cfg.get("n_val"):  # validation loss on a fixed random subset of the val split
        import random
        allval = load_split(Path(cfg["data_dir"]), "val")
        random.Random(cfg["seed"]).shuffle(allval)
        val_eps = allval[: cfg["n_val"]]
    train_ds = Dataset.from_list(to_rows(train_eps, tok, cfg["condition"]))
    val_ds = Dataset.from_list(to_rows(val_eps, tok, cfg["condition"])) if val_eps else None

    dtype = getattr(torch, cfg.get("dtype", "bfloat16"))
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], revision=cfg.get("revision"), dtype=dtype)
    lc = cfg["lora"]
    peft_cfg = LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                          target_modules=lc["target_modules"], task_type="CAUSAL_LM")
    t = cfg["training"]
    sargs = SFTConfig(
        output_dir=str(out), num_train_epochs=t["epochs"], max_steps=t.get("max_steps", -1),
        per_device_train_batch_size=t["batch_size"], gradient_accumulation_steps=t["grad_accum"],
        learning_rate=t["lr"], lr_scheduler_type=t.get("scheduler", "cosine"), warmup_steps=t.get("warmup_steps", 5),
        weight_decay=t.get("weight_decay", 0.0), max_length=t["max_length"], completion_only_loss=True,
        gradient_checkpointing=t.get("gradient_checkpointing", True), bf16=t.get("bf16", True),
        logging_steps=t["logging_steps"], save_steps=t["save_steps"], save_total_limit=2,
        eval_strategy="steps" if val_ds is not None else "no", eval_steps=t.get("eval_steps"),
        per_device_eval_batch_size=t["batch_size"], report_to=[], seed=cfg["seed"],
        dataloader_num_workers=t.get("num_workers", 0), use_cpu=t.get("use_cpu", False),
    )
    trainer = SFTTrainer(model=model, args=sargs, train_dataset=train_ds, eval_dataset=val_ds,
                         processing_class=tok, peft_config=peft_cfg, callbacks=[memory_callback()])
    n_trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in trainer.model.parameters())
    print(f"trainable params: {n_trainable:,} / {n_total:,} ({100 * n_trainable / n_total:.3f}%)")
    # Sanity check: loss covers exactly the reference response + one end-of-turn token
    ex = trainer.train_dataset[0]
    target = tok.decode([i for i, l in zip(ex["input_ids"], ex["labels"]) if l != -100])
    print("loss-masked target for example 0:", repr(target))
    if not target.startswith('{"reason"') or target.count(tok.eos_token) != 1:
        raise RuntimeError(f"unexpected SFT loss target: {target!r}")
    t0 = time.time()
    resume = cfg.get("resume_from_checkpoint")
    result = trainer.train(resume_from_checkpoint=resume)
    final = out / "final"
    trainer.save_model(str(final))
    tok.save_pretrained(str(final))
    write_json(out / "train_info.json", {
        "config": cfg, "env": environment_info(), "train_seconds": time.time() - t0,
        "n_train_episodes": len(train_ds), "n_val_episodes": len(val_ds) if val_ds else 0,
        "train_episode_ids": [e["episode_id"] for e in train_eps],
        "metrics": result.metrics, "log_history": trainer.state.log_history,
        "global_steps": trainer.state.global_step,
        "trainable_params": n_trainable, "total_params": n_total,
        "peak_mem_gb": (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else None,
        "trained_tokens": int(getattr(trainer.state, "num_input_tokens_seen", 0) or 0),
    })
    print(f"saved adapter to {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
