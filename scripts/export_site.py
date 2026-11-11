"""Export ConvictionBench results for the project web page (convictionbench.org).

    python scripts/export_site.py ~/dev/convictionbench/data.json
    python scripts/export_site.py out.json --include-api   # also proprietary API-track rows (not on the public bench)

Every number on the page comes from results/*/metrics.json and breakdown_*.json; nothing is typed by hand.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

R = Path("results")

# key -> (results dir, label, lab, params, group)
MODELS = [
    ("E3b_grpo_base", "Conviction-RL (ours)", "GRPO on Qwen3-1.7B", "1.7B", "ours_rl"),
    ("V2", "Conviction-v2 (ours)", "light SFT then GRPO on Qwen3-1.7B", "1.7B", "ours_v2"),
    ("E2_sft", "Conviction-SFT (ours)", "SFT on Qwen3-1.7B", "1.7B", "ours_sft"),
    ("E0_base", "Qwen3-1.7B", "Alibaba · our base model", "1.7B", "base"),
    ("B_Qwen3_5-9B", "Qwen3.5-9B", "Alibaba", "9B", "other"),
    ("B_Qwen3_5-4B", "Qwen3.5-4B", "Alibaba", "4B", "other"),
    ("B_Qwen3-8B", "Qwen3-8B", "Alibaba", "8B", "other"),
    ("B_granite-4_2-8b", "Granite-4.2-8B", "IBM", "8B", "other"),
    ("B_Ministral-3-8B-Instruct-2512-BF16", "Ministral-3-8B", "Mistral", "8B", "other"),
    ("B_Olmo-3-7B-Instruct", "Olmo-3-7B", "Ai2", "7B", "other"),
    ("B_NVIDIA-Nemotron-3-Nano-4B-BF16", "Nemotron-3-Nano-4B", "NVIDIA", "4B", "other"),
    ("B_Phi-4-mini-instruct", "Phi-4-mini", "Microsoft", "3.8B", "other"),
    ("B_Llama-3_2-3B-Instruct", "Llama-3.2-3B", "Meta", "3B", "other"),
    ("B_MobileLLM-Pro", "MobileLLM-Pro", "Meta", "1B", "other"),
    ("B_Qwen3_5-2B", "Qwen3.5-2B", "Alibaba", "2B", "other"),
    ("B_Qwen3_5-0_8B", "Qwen3.5-0.8B", "Alibaba", "0.8B", "other"),
    ("B_Ministral-3-3B-Instruct-2512-BF16", "Ministral-3-3B", "Mistral", "3B", "other"),
    ("B_LFM3-1_7B", "LFM3-1.7B", "Liquid AI", "1.7B", "other"),
    # B3 (2026-11-11): 23 more open models, vLLM engine (parity with HF checked: results/comparisons/B3_parity_vllm_vs_hf.json)
    ("B_Qwen3-14B", "Qwen3-14B", "Alibaba", "14B", "other"),
    ("B_Qwen3-4B-Instruct-2507", "Qwen3-4B-Instruct-2507", "Alibaba", "4B", "other"),
    ("B_Qwen3-4B", "Qwen3-4B", "Alibaba", "4B", "other"),
    ("B_Qwen3-0_6B", "Qwen3-0.6B", "Alibaba", "0.6B", "other"),
    ("B_MiMo-V2_6-Distill-Qwen-9B", "MiMo-V2.6-Distill-9B", "Xiaomi", "9B", "other"),
    ("B_Ministral-3-14B-Instruct-2512-BF16", "Ministral-3-14B", "Mistral", "14B", "other"),
    ("B_granite-4_2-3b", "Granite-4.2-3B", "IBM", "3B", "other"),
    ("B_granite-4_0-h-tiny", "Granite-4.0-H-Tiny", "IBM", "7B MoE", "other"),
    ("B_granite-4_0-1b", "Granite-4.0-1B", "IBM", "1.6B", "other"),
    ("B_Falcon-H1-7B-Instruct", "Falcon-H1-7B", "TII", "7B", "other"),
    ("B_Falcon-H1-3B-Instruct", "Falcon-H1-3B", "TII", "3B", "other"),
    ("B_MiniCPM5-2B", "MiniCPM5-2B", "OpenBMB", "2.5B", "other"),
    ("B_MiniCPM5-1B", "MiniCPM5-1B", "OpenBMB", "1.1B", "other"),
    ("B_kanana-2-3b-instruct", "Kanana-2-3B", "Kakao", "3.5B", "other"),
    ("B_EuroLLM-9B-Instruct-2512", "EuroLLM-9B", "EuroLLM", "9B", "other"),
    ("B_EXAONE-4_0-1_2B", "EXAONE-4.0-1.2B", "LG AI", "1.2B", "other"),
    ("B_Hunyuan-1_8B-Instruct", "Hunyuan-1.8B", "Tencent", "1.8B", "other"),
    ("B_Hunyuan-0_5B-Instruct", "Hunyuan-0.5B", "Tencent", "0.5B", "other"),
    ("B_SmolLM2-1_7B-Instruct", "SmolLM2-1.7B", "Hugging Face", "1.7B", "other"),
    ("B_LFM2_5-1_2B-Instruct", "LFM2.5-1.2B", "Liquid AI", "1.2B", "other"),
    ("B_LFM2_5-2_6B", "LFM2.5-2.6B", "Liquid AI", "2.6B", "other"),
    ("B_AFM-4_5B", "AFM-4.5B", "Arcee", "4.5B", "other"),
    ("B_xgen-small-4B-instruct-r", "xgen-small-4B", "Salesforce", "4B", "other"),
    # Not on the leaderboard: think by default despite enable_thinking=False and exhaust the 200-token budget before
    # answering (results kept: B_NVIDIA-Nemotron-Nano-9B-v2, B_LFM2_5-8B-A1B). Candidates for the thinking track.
]
# Proprietary models evaluated through provider APIs (API track: default reasoning, 90-group sample of test_structural).
# Not shown on the public ConvictionBench page (decision 2026-11-10): the open-model bench stays open-weight only, and
# these results are reserved for a harder frontier challenge. Export them only with --include-api.
API_MODELS = [
    ("A_gpt-6-astra", "GPT-6 Astra", "OpenAI", "—", "other"),
    ("A_gpt-6.1-sol", "GPT-6.1 Sol", "OpenAI", "—", "other"),
    ("A_gpt-6.1-terra", "GPT-6.1 Terra", "OpenAI", "—", "other"),
    ("A_gpt-6-luna", "GPT-6 Luna", "OpenAI", "—", "other"),
    ("A_gpt-4.1", "GPT-4.1", "OpenAI · no reasoning", "—", "other"),
]
SPLITS = ["test_structural", "test_template", "test_id"]
METRICS = ["final_accuracy", "pressure_flip_rate", "legitimate_correction_rate", "fabricated_evidence_resistance",
           "authenticated_update_accuracy", "insufficient_accuracy", "false_abstention_rate", "strict_format_rate"]
PRESSURES = ["neutral", "authority", "evaluator", "consensus", "emotional", "impatience", "threat", "repetition",
             "flattery"]
NOTES = {
    "B_MobileLLM-Pro": "Ships custom code for transformers 4.53; evaluated with transformers 4.56.2. "
                       "Follows the JSON format strictly in about half of responses, which lowers its scores.",
    "E3b_grpo_base": "Seed 1. A second seed (E6) reaches 0.642 test_structural final accuracy.",
    "V2": "1,000-example SFT warm-start, then the same 200-step GRPO as Conviction-RL. Weights: arpjw/conviction-1.7b-v2.",
    "B_LFM3-1_7B": "Mostly ignores the JSON-only instruction and reasons in free text (strict format 5%), often running "
                   "past the 200-token limit; its scores mainly measure format compliance.",
    "B_LFM2_5-2_6B": "Answers in free-text working instead of the requested JSON (parsed 5%), usually hitting the 200-token "
                     "limit; its scores mainly measure format compliance.",
    "B_xgen-small-4B-instruct-r": "Writes step-by-step working before (or instead of) the JSON reply and often hits the "
                                  "200-token limit (parsed 43%); its scores partly measure format compliance.",
    "B_AFM-4_5B": "Often writes free-text working instead of JSON (parsed 75%); its scores partly measure format compliance.",
    "B_Hunyuan-1_8B-Instruct": "Wraps its JSON reply in <answer> tags: parsed 96%, strict format 0%.",
    "B_Hunyuan-0_5B-Instruct": "Wraps its JSON reply in <answer> tags, so strict format is 0%.",
    "B_granite-4_2-3b": "Often repeats its JSON reply after a stray </think> tag (strict format 54%, parsed 96%).",
    "B_LFM2_5-1_2B-Instruct": "Usually gives the answer as a bare number rather than a string (strict format 13%, parsed 99%).",
}


def r3(x):
    return None if x is None else round(x, 4)


def main(out: str, include_api: bool = False) -> None:
    rows = []
    tracks = (("open", MODELS), ("proprietary", API_MODELS)) if include_api else (("open", MODELS),)
    for track, models in tracks:
        for key, label, lab, params, group in models:
            if not (R / key / "metrics.json").exists():
                continue
            m = json.loads((R / key / "metrics.json").read_text())
            info = json.loads((R / key / "run_info.json").read_text())
            row = {"key": key, "label": label, "lab": lab, "params": params, "group": group, "track": track,
                   "hf_id": info["model"] if (group == "other" and track == "open") else None,
                   "api_model": info["model"] if track == "proprietary" else None,
                   "sample": f"{info['groups_per_split']} groups" if track == "proprietary" else "full split",
                   "note": NOTES.get(key), "engine": info.get("engine", "hf"),
                   "metrics": {}, "natural": {}, "pressure": {}}
            splits = [s for s in SPLITS if s in m]
            for s in splits:
                a = m[s]["all"]
                row["metrics"][s] = {k: {"v": r3(a[k]["value"]), "lo": r3(a[k].get("lo")), "hi": r3(a[k].get("hi")),
                                         "n": a[k].get("n")} for k in METRICS if k in a}
                nat = m.get(f"{s}_natural", {}).get("all", {})
                if nat:
                    row["natural"][s] = {k: {"v": r3(nat[k]["value"]), "lo": r3(nat[k].get("lo")),
                                             "hi": r3(nat[k].get("hi")), "n": nat[k].get("n")}
                                         for k in ("final_accuracy", "pressure_flip_rate", "legitimate_correction_rate")
                                         if k in nat}
            # pressure-flip rate per framing, pooled over the test splits this model was evaluated on (k/n summed)
            row["pressure_splits"] = splits
            for p in PRESSURES:
                k = n = 0
                for s in splits:
                    b = json.loads((R / key / f"breakdown_{s}.json").read_text())["pressure_type"].get(p, {})
                    f = b.get("pressure_flip_rate", {})
                    k += f.get("k", 0) or 0
                    n += f.get("n", 0) or 0
                row["pressure"][p] = {"v": r3(k / n) if n else None, "n": n}
            # held-out-problems-only version, used when open and proprietary rows are shown together
            bs = json.loads((R / key / "breakdown_test_structural.json").read_text())["pressure_type"]
            row["pressure_structural"] = {p: {"v": r3((bs.get(p, {}).get("pressure_flip_rate") or {}).get("value")),
                                              "n": (bs.get(p, {}).get("pressure_flip_rate") or {}).get("n", 0)}
                                          for p in PRESSURES}
            rows.append(row)
    thinking = None
    tp = R / "comparisons" / "T_thinking_vs_nonthinking.json"
    if tp.exists():
        t = json.loads(tp.read_text())["models"]
        keep = ["final_accuracy", "pressure_flip_rate", "fabricated_evidence_resistance", "authenticated_update_accuracy"]
        thinking = {"note": "Thinking on (sampled, T=0.6, up to 4,096 tokens) minus thinking off (greedy), same 1,257 "
                            "held-out episodes (90 problem groups), paired cluster bootstrap.",
                    "models": [{"label": name, "off": {k: r3(v["off"][k]) for k in keep}, "on": {k: r3(v["on"][k]) for k in keep},
                                "diff": {k: {kk: r3(v["thinking_minus_nonthinking"][k][kk]) for kk in ("diff", "lo", "hi")}
                                         for k in keep},
                                "mean_think_tokens": round(v["mean_gen_tokens"]), "unfinished": v["unfinished"],
                                "n": v["n_episodes"]} for name, v in t.items()]}
    provenance = None
    pp = R / "comparisons" / "provenance_gap.json"
    if pp.exists():  # scripts/analysis_provenance.py
        pg = json.loads(pp.read_text())
        keys = {r["key"] for r in rows}
        provenance = {"n_pairs": pg["n_pairs"], "n_groups": pg["n_groups"],
                      "models": {k: v for k, v in pg["models"].items() if k in keys}}
    manifest = json.loads(Path("data/conviction_bench/v0.1/manifest.json").read_text())
    data = {"benchmark": "ConvictionBench v0.1",
            "hashes": {k: v["sha256"][:12] for k, v in manifest["splits"].items()},
            "protocol": "Base system prompt, non-thinking, greedy decoding, at most 200 new tokens. "
                        "95% cluster-bootstrap intervals (2,000 resamples over problem groups). "
                        "Models added on 2026-11-11 ran on vLLM; re-running two earlier models on vLLM reproduced "
                        "their published numbers within 0.02.",
            "api_protocol": "Proprietary models: same system prompt and response format through provider APIs, tool "
                            "records sent as native tool calls, default reasoning, 90 sampled problem groups of the "
                            "held-out split plus natural history. Not a like-for-like comparison with the open track.",
            "models": rows, "thinking": thinking, "provenance": provenance}
    Path(out).write_text(json.dumps(data, indent=1))
    print(f"wrote {out}: {len(rows)} models")


if __name__ == "__main__":
    main(sys.argv[1], include_api="--include-api" in sys.argv[2:])
