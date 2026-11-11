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


def split_thinking(raw: str, prompt: str) -> tuple[str, bool]:
    """Separate the final answer from a thinking-mode generation.

    Returns (final_text, finished). The answer is whatever follows the last ``</think>``. If thinking was
    opened (in the output, or by the chat template at the end of the prompt) but never closed, the model ran
    out of tokens while thinking: the answer is empty and ``finished`` is False. Output with no thinking
    block at all is returned unchanged."""
    if "</think>" in raw:
        return raw.rsplit("</think>", 1)[1].strip(), True
    if "<think>" in raw or prompt.rstrip().endswith("<think>"):
        return "", False
    return raw, True


def require_cuda() -> None:
    """Abort if CONVICTION_REQUIRE_CUDA=1 and no CUDA device is visible (prevents silent CPU fallback on pods)."""
    import os
    import torch
    if os.environ.get("CONVICTION_REQUIRE_CUDA") == "1" and not torch.cuda.is_available():
        raise SystemExit(f"CUDA required but unavailable (torch {torch.__version__}, built for CUDA {torch.version.cuda})")


def load_model(model_name: str, adapter: str | None = None, revision: str | None = None,
               dtype: str = "bfloat16", device: str | None = None):
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok_kw = {"fix_mistral_regex": True} if "mistral" in model_name.lower() else {}
    # opt-in for official repos that ship custom code (e.g. facebook/MobileLLM-Pro); never on by default
    trc = os.environ.get("CONVICTION_TRUST_REMOTE_CODE") == "1"
    if trc:
        tok_kw["trust_remote_code"] = True
    tok = AutoTokenizer.from_pretrained(model_name, revision=revision, **tok_kw)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    require_cuda()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, dtype=getattr(torch, dtype),
                                                     trust_remote_code=trc)
    except ValueError as e:
        # vision-language checkpoints (e.g. Ministral-3) have no CausalLM mapping; use them text-only
        if "Unrecognized configuration class" not in str(e):
            raise
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_name, revision=revision, dtype=getattr(torch, dtype))
    model = model.to(device)
    if adapter:  # comma-separated adapters are merged in order (e.g. "sft/final,grpo/final")
        from peft import PeftModel
        for a in adapter.split(","):
            model = PeftModel.from_pretrained(model, a).merge_and_unload()
    model.eval()
    return model, tok


OOM_SPLITS: list[int] = []  # batch sizes that hit CUDA OOM and were split (recorded in run_info)


def _gen_batch(model, tok, batch: list[str], max_new_tokens: int, gen_kwargs: dict | None = None,
               lengths: list | None = None) -> list[str]:
    """Greedy-generate one batch; on CUDA OOM, split it in half and retry (results stay greedy)."""
    import torch
    try:
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        kw = gen_kwargs or {"do_sample": False, "temperature": None, "top_p": None, "top_k": None}
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, **kw)
        new = gen[:, enc["input_ids"].shape[1]:]
        if lengths is not None:  # generated tokens per row (up to and including the first EOS)
            eos = set(x for x in [tok.eos_token_id, tok.pad_token_id] if x is not None)
            for row in new.tolist():
                n = next((i + 1 for i, t in enumerate(row) if t in eos), len(row))
                lengths.append(n)
        return tok.batch_decode(new, skip_special_tokens=True)
    except torch.cuda.OutOfMemoryError:
        if len(batch) == 1:
            raise
        OOM_SPLITS.append(len(batch))
        torch.cuda.empty_cache()
        h = len(batch) // 2
        return (_gen_batch(model, tok, batch[:h], max_new_tokens, gen_kwargs, lengths)
                + _gen_batch(model, tok, batch[h:], max_new_tokens, gen_kwargs, lengths))


def generate(model, tok, prompts: list[str], batch_size: int = 32, max_new_tokens: int = 160,
             progress: bool = True, gen_kwargs: dict | None = None, lengths_out: list | None = None) -> list[str]:
    import torch

    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    outs: list[str | None] = [None] * len(prompts)
    lens: list[int | None] = [None] * len(prompts)
    t0 = time.time()
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        bl: list[int] = []
        for i, t in zip(idx, _gen_batch(model, tok, [prompts[i] for i in idx], max_new_tokens, gen_kwargs, bl)):
            outs[i] = t
        for i, n in zip(idx, bl):
            lens[i] = n
        if progress:
            done = min(b + batch_size, len(order))
            print(f"  generated {done}/{len(order)}  ({time.time() - t0:.0f}s)", flush=True)
    if lengths_out is not None:
        lengths_out.extend(lens)
    return outs  # type: ignore[return-value]


def generate_vllm(llm, tok, prompts: list[str], max_new_tokens: int, gen_kwargs: dict | None, seed: int,
                  lengths_out: list | None = None) -> list[str]:
    """Generate with vLLM (continuous batching). Prompts are tokenized exactly as in the HF path
    (chat-template text, no extra special tokens) and passed as token ids."""
    from vllm import SamplingParams
    kw = gen_kwargs or {}
    sp = SamplingParams(max_tokens=max_new_tokens, seed=seed,
                        temperature=kw.get("temperature") if kw.get("do_sample") else 0.0,
                        top_p=kw.get("top_p") or 1.0, top_k=kw.get("top_k") or 0)
    ids = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
    outs = llm.generate([{"prompt_token_ids": x} for x in ids], sp)
    texts = [o.outputs[0].text for o in outs]
    if lengths_out is not None:
        lengths_out.extend(len(o.outputs[0].token_ids) for o in outs)
    return texts


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
    ap.add_argument("--enable-thinking", action="store_true",
                    help="thinking mode: the reasoning before </think> is removed and only the final answer is scored")
    ap.add_argument("--groups", type=int, default=0, help="sample this many whole problem groups per split (0 = all)")
    ap.add_argument("--sample-seed", type=int, default=20261110)
    ap.add_argument("--do-sample", action="store_true", help="sample instead of greedy decoding")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0, help="torch seed (only matters with --do-sample)")
    ap.add_argument("--engine", default="hf", choices=["hf", "vllm"],
                    help="generation engine; vllm uses continuous batching (much faster for long thinking outputs)")
    ap.add_argument("--max-model-len", type=int, default=8192, help="vllm context length (prompt + output)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    missing = [s for s in args.splits.split(",") if not (Path(args.data_dir) / f"{s}.jsonl").exists()]
    if missing:
        raise SystemExit(f"missing splits {missing} in {args.data_dir}: run `conviction-generate` first "
                         "(regenerates ConvictionBench v0.1 deterministically; verify against FROZEN.txt)")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    llm = model = None
    if args.engine == "vllm":
        if args.adapter:
            raise SystemExit("--engine vllm does not support --adapter here; use the hf engine")
        from transformers import AutoTokenizer
        from vllm import LLM
        tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
        llm = LLM(model=args.model, revision=args.revision, dtype=args.dtype, max_model_len=args.max_model_len,
                  seed=args.seed, trust_remote_code=False)
    else:
        model, tok = load_model(args.model, args.adapter, args.revision, args.dtype)
    t_loaded = time.time()
    import torch
    torch.manual_seed(args.seed)
    gen_kwargs = ({"do_sample": True, "temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k}
                  if args.do_sample else None)
    think_stats: dict = {}

    def run_episodes(eps_, label):
        prompts_ = [render_prompt(tok, e["messages"], args.condition, args.enable_thinking) for e in eps_]
        lens: list[int] = []
        if llm is not None:
            raws_ = generate_vllm(llm, tok, prompts_, args.max_new_tokens, gen_kwargs, args.seed, lengths_out=lens)
        else:
            raws_ = generate(model, tok, prompts_, args.batch_size, args.max_new_tokens, gen_kwargs=gen_kwargs,
                             lengths_out=lens)
        recs_ = []
        unfinished = 0
        for e, p, r, n in zip(eps_, prompts_, raws_, lens):
            final, finished = split_thinking(r, p) if args.enable_thinking else (r, True)
            rec = M.score_record(e, final)
            rec["gen_tokens"] = n
            if args.enable_thinking:
                rec["raw_full"] = r
                rec["think_finished"] = finished
                unfinished += not finished
            recs_.append(rec)
        think_stats[label] = {"episodes": len(eps_), "mean_gen_tokens": sum(lens) / max(1, len(lens)),
                              "max_gen_tokens": max(lens) if lens else 0,
                              "hit_max_new_tokens": sum(n >= args.max_new_tokens for n in lens),
                              "thinking_unfinished": unfinished}
        return recs_
    all_metrics, rows_csv, timing = {}, [], {}
    for split in args.splits.split(","):
        eps = load_split(Path(args.data_dir), split)
        if args.groups:
            from conviction.evaluation.sampling import sample_groups
            eps = sample_groups(eps, args.groups, args.sample_seed)
        if args.limit_groups:
            keep = sorted({e["group_id"] for e in eps})[: args.limit_groups]
            eps = [e for e in eps if e["group_id"] in set(keep)]
        t0 = time.time()
        recs = run_episodes(eps, split)
        timing[split] = {"episodes": len(eps), "seconds": time.time() - t0}
        _write_jsonl(out / f"predictions_{split}.jsonl", recs)
        all_metrics[split] = evaluate_records(recs, args.n_boot)
        bd = {k: M.breakdown(recs, k, BREAKDOWN_METRICS) for k in BREAKDOWN_KEYS}
        json.dump(bd, open(out / f"breakdown_{split}.json", "w"), indent=1)

        if args.natural:
            # the model's own first answer (final answer only, in thinking mode) becomes the prior
            clean = {r["group_id"]: (e, r["raw"]) for e, r in zip(eps, recs) if e["slot"] == "clean"}
            nat_eps = [x for e, raw in clean.values() for x in natural_battery(e, raw)]
            if nat_eps:
                nrecs = run_episodes(nat_eps, f"{split}_natural")
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
            "decoding": {"do_sample": args.do_sample, "temperature": args.temperature, "top_p": args.top_p,
                         "top_k": args.top_k, "seed": args.seed, "max_new_tokens": args.max_new_tokens,
                         "enable_thinking": args.enable_thinking},
            "groups_per_split": args.groups, "sample_seed": args.sample_seed, "generation_stats": think_stats,
            "batch_size": args.batch_size, "dtype": args.dtype, "splits": args.splits,
            "limit_groups": args.limit_groups, "timing": timing, "oom_splits": OOM_SPLITS, "load_seconds": t_loaded - t_start,
            "total_seconds": time.time() - t_start, "torch": torch.__version__,
            "transformers": transformers.__version__, "engine": args.engine,
            "device": str(model.device) if model is not None else "cuda (vllm)", "python": platform.python_version(),
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
