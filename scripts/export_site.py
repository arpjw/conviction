"""Export ConvictionBench results for the project web page (aryasomu.com/conviction).

    python scripts/export_site.py ~/dev/arpjw.github.io/conviction/data.json

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
]
# Proprietary models evaluated through provider APIs (API track: default reasoning, 90-group sample of test_structural)
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
}


def r3(x):
    return None if x is None else round(x, 4)


def main(out: str) -> None:
    rows = []
    for track, models in (("open", MODELS), ("proprietary", API_MODELS)):
        for key, label, lab, params, group in models:
            if not (R / key / "metrics.json").exists():
                continue
            m = json.loads((R / key / "metrics.json").read_text())
            info = json.loads((R / key / "run_info.json").read_text())
            row = {"key": key, "label": label, "lab": lab, "params": params, "group": group, "track": track,
                   "hf_id": info["model"] if (group == "other" and track == "open") else None,
                   "api_model": info["model"] if track == "proprietary" else None,
                   "sample": f"{info['groups_per_split']} groups" if track == "proprietary" else "full split",
                   "note": NOTES.get(key), "metrics": {}, "natural": {}, "pressure": {}}
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
    manifest = json.loads(Path("data/conviction_bench/v0.1/manifest.json").read_text())
    data = {"benchmark": "ConvictionBench v0.1",
            "hashes": {k: v["sha256"][:12] for k, v in manifest["splits"].items()},
            "protocol": "Base system prompt, non-thinking, greedy decoding, at most 200 new tokens. "
                        "95% cluster-bootstrap intervals (2,000 resamples over problem groups).",
            "api_protocol": "Proprietary models: same system prompt and response format through provider APIs, tool "
                            "records sent as native tool calls, default reasoning, 90 sampled problem groups of the "
                            "held-out split plus natural history. Not a like-for-like comparison with the open track.",
            "models": rows}
    Path(out).write_text(json.dumps(data, indent=1))
    print(f"wrote {out}: {len(rows)} models")


if __name__ == "__main__":
    main(sys.argv[1])
