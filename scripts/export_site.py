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
    for key, label, lab, params, group in MODELS:
        m = json.loads((R / key / "metrics.json").read_text())
        info = json.loads((R / key / "run_info.json").read_text())
        row = {"key": key, "label": label, "lab": lab, "params": params, "group": group,
               "hf_id": info["model"] if group == "other" else None, "note": NOTES.get(key),
               "metrics": {}, "natural": {}, "pressure": {}}
        for s in SPLITS:
            a = m[s]["all"]
            row["metrics"][s] = {k: {"v": r3(a[k]["value"]), "lo": r3(a[k].get("lo")), "hi": r3(a[k].get("hi")),
                                     "n": a[k].get("n")} for k in METRICS if k in a}
            nat = m.get(f"{s}_natural", {}).get("all", {})
            if nat:
                row["natural"][s] = {k: {"v": r3(nat[k]["value"]), "lo": r3(nat[k].get("lo")),
                                         "hi": r3(nat[k].get("hi")), "n": nat[k].get("n")}
                                     for k in ("final_accuracy", "pressure_flip_rate", "legitimate_correction_rate")
                                     if k in nat}
        # pressure-flip rate per framing, pooled over the three test splits (k/n summed)
        for p in PRESSURES:
            k = n = 0
            for s in SPLITS:
                b = json.loads((R / key / f"breakdown_{s}.json").read_text())["pressure_type"].get(p, {})
                f = b.get("pressure_flip_rate", {})
                k += f.get("k", 0) or 0
                n += f.get("n", 0) or 0
            row["pressure"][p] = {"v": r3(k / n) if n else None, "n": n}
        rows.append(row)
    manifest = json.loads(Path("data/conviction_bench/v0.1/manifest.json").read_text())
    data = {"benchmark": "ConvictionBench v0.1",
            "hashes": {k: v["sha256"][:12] for k, v in manifest["splits"].items()},
            "protocol": "Base system prompt, non-thinking, greedy decoding, at most 200 new tokens. "
                        "95% cluster-bootstrap intervals (2,000 resamples over problem groups).",
            "models": rows}
    Path(out).write_text(json.dumps(data, indent=1))
    print(f"wrote {out}: {len(rows)} models")


if __name__ == "__main__":
    main(sys.argv[1])
