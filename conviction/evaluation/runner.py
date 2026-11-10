"""Evaluate a model (optionally with a LoRA adapter) on ConvictionBench.

    conviction-eval --model <hf-model-id> [--adapter path[,path]] --out results/<name>
    python -m conviction.evaluation.runner --model Qwen/Qwen3-1.7B --condition base \
        --splits test_id,test_template,test_structural,test_adversarial --out results/E0

Outputs (in --out):
    predictions_<split>.jsonl   raw output + parsed fields for every episode
    metrics.json                cluster-bootstrap estimates per split (+ capability-conditioned)
    breakdown_<split>.json      metrics by pressure type / evidence type / family / subtype
    summary.csv                 one row per (split, metric)
    run_info.json               model, adapter, generation settings, versions, timing
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

from conviction.data.splits import load_split
from conviction.evaluation import metrics as M
from conviction.evaluation.natural import natural_battery
from conviction.evaluation.statistics import bootstrap
from conviction.prompts import build_messages

BREAKDOWN_KEYS = ["pressure_type", "evidence_type", "task_family", "subtype", "prior_kind"]
BREAKDOWN_METRICS = ["final_accuracy", "pressure_flip_rate", "capitulation_rate", "legitimate_correction_rate",
                     "authenticated_update_accuracy", "fabricated_evidence_resistance", "insufficient_accuracy",
                     "false_abstention_rate"]


def render_prompt(tokenizer, episode_messages: list[dict], condition: str, enable_thinking: bool = False) -> str:
    return tokenizer.apply_chat_template(build_messages(episode_messages, condition), tokenize=False,
                                         add_generation_prompt=True, enable_thinking=enable_thinking)


def require_cuda() -> None:
    """Abort if CONVICTION_REQUIRE_CUDA=1 and no CUDA device is visible (prevents silent CPU fallback on pods)."""
    import os
    import torch
    if os.environ.get("CONVICTION_REQUIRE_CUDA") == "1" and not torch.cuda.is_available():
        raise SystemExit(f"CUDA required but unavailable (torch {torch.__version__}, built for CUDA {torch.version.cuda})")


def load_model(model_name: str, adapter: str | None = None, revision: str | None = None,
               dtype: str = "bfloat16", device: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, revision=revision)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    require_cuda()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision,
                                                 dtype=getattr(torch, dtype)).to(device)
    if adapter:  # comma-separated adapters are merged in order (e.g. "sft/final,grpo/final")
        from peft import PeftModel
        for a in adapter.split(","):
            model = PeftModel.from_pretrained(model, a).merge_and_unload()
    model.eval()
    return model, tok


def generate(model, tok, prompts: list[str], batch_size: int = 32, max_new_tokens: int = 160,
             progress: bool = True) -> list[str]:
    import torch

    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outs: list[str | None] = [None] * len(prompts)
    t0 = time.time()
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                                 top_p=None, top_k=None, pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for i, t in zip(idx, texts):
            outs[i] = t
        if progress:
            done = min(b + batch_size, len(order))
            print(f"  generated {done}/{len(order)}  ({time.time() - t0:.0f}s)", flush=True)
    return outs  # type: ignore[return-value]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def evaluate_records(records: list[dict], n_boot: int) -> dict:
    return {"all": bootstrap(records, n_boot=n_boot),
            "capability_conditioned": bootstrap(M.capability_conditioned(records), n_boot=n_boot)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--condition", default="base", choices=["base", "guided", "base_decision_first", "base_reason_first"])
    ap.add_argument("--data-dir", default="data/conviction_bench/v0.1")
    ap.add_argument("--splits", default="test_id,test_template,test_structural,test_adversarial")
    ap.add_argument("--natural", action="store_true", help="also run natural-history batteries")
    ap.add_argument("--limit-groups", type=int, default=None, help="evaluate only the first N groups per split")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    missing = [s for s in args.splits.split(",") if not (Path(args.data_dir) / f"{s}.jsonl").exists()]
    if missing:
        raise SystemExit(f"missing splits {missing} in {args.data_dir}: run `conviction-generate` first "
                         "(regenerates ConvictionBench v0.1 deterministically; verify against FROZEN.txt)")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    model, tok = load_model(args.model, args.adapter, args.revision, args.dtype)
    t_loaded = time.time()
    all_metrics, rows_csv, timing = {}, [], {}
    for split in args.splits.split(","):
        eps = load_split(Path(args.data_dir), split)
        if args.limit_groups:
            keep = sorted({e["group_id"] for e in eps})[: args.limit_groups]
            eps = [e for e in eps if e["group_id"] in set(keep)]
        prompts = [render_prompt(tok, e["messages"], args.condition) for e in eps]
        t0 = time.time()
        raws = generate(model, tok, prompts, args.batch_size, args.max_new_tokens)
        timing[split] = {"episodes": len(eps), "seconds": time.time() - t0}
        recs = [M.score_record(e, r) for e, r in zip(eps, raws)]
        _write_jsonl(out / f"predictions_{split}.jsonl", recs)
        all_metrics[split] = evaluate_records(recs, args.n_boot)
        bd = {k: M.breakdown(recs, k, BREAKDOWN_METRICS) for k in BREAKDOWN_KEYS}
        json.dump(bd, open(out / f"breakdown_{split}.json", "w"), indent=1)

        if args.natural:
            clean = {r["group_id"]: (e, r["raw"]) for e, r in zip(eps, recs) if e["slot"] == "clean"}
            nat_eps = [x for e, raw in clean.values() for x in natural_battery(e, raw)]
            if nat_eps:
                nprompts = [render_prompt(tok, e["messages"], args.condition) for e in nat_eps]
                nraws = generate(model, tok, nprompts, args.batch_size, args.max_new_tokens)
                nrecs = [M.score_record(e, r) for e, r in zip(nat_eps, nraws)]
                _write_jsonl(out / f"predictions_{split}_natural.jsonl", nrecs)
                all_metrics[f"{split}_natural"] = {"all": bootstrap(nrecs, n_boot=args.n_boot)}

    for split, d in all_metrics.items():
        for view, ms in d.items():
            for name, v in ms.items():
                rows_csv.append({"split": split, "view": view, "metric": name, **v})
    json.dump(all_metrics, open(out / "metrics.json", "w"), indent=1)
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "view", "metric", "value", "lo", "hi", "n", "clusters"])
        w.writeheader()
        for r in rows_csv:
            w.writerow({k: r.get(k) for k in w.fieldnames})

    import torch, transformers  # noqa: E401
    info = {"model": args.model, "revision": args.revision, "adapter": args.adapter, "condition": args.condition,
            "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens, "enable_thinking": False},
            "batch_size": args.batch_size, "dtype": args.dtype, "splits": args.splits,
            "limit_groups": args.limit_groups, "timing": timing, "load_seconds": t_loaded - t_start,
            "total_seconds": time.time() - t_start, "torch": torch.__version__,
            "transformers": transformers.__version__, "device": str(model.device), "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    json.dump(info, open(out / "run_info.json", "w"), indent=1)
    for split, d in all_metrics.items():
        a = d["all"]
        print(split, {k: (round(v["value"], 3) if v["value"] is not None else None)
                      for k, v in a.items() if k in ("final_accuracy", "clean_accuracy", "pressure_flip_rate",
                                                     "legitimate_correction_rate", "insufficient_accuracy",
                                                     "strict_format_rate")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
