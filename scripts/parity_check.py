"""B3 parity gate: vLLM re-runs (results/P_*) vs the published HF runs, pre-registered in research/EXPERIMENTS.md.

Passes if every metric below differs by <= 0.04 on test_structural and test_id for every pair.
    python scripts/parity_check.py
"""
import json
import sys
from pathlib import Path

PAIRS = [("P_Qwen3-1_7B", "E0_base"), ("P_Olmo-3-7B-Instruct", "B_Olmo-3-7B-Instruct")]
SPLITS = ["test_structural", "test_id"]
METRICS = ["final_accuracy", "pressure_flip_rate", "legitimate_correction_rate", "fabricated_evidence_resistance",
           "authenticated_update_accuracy", "insufficient_accuracy"]
TOL = 0.04


def main():
    rows, ok = [], True
    for new, old in PAIRS:
        a = json.load(open(Path("results") / new / "metrics.json"))
        b = json.load(open(Path("results") / old / "metrics.json"))
        for s in SPLITS:
            for m in METRICS:
                x, y = a[s]["all"][m], b[s]["all"][m]
                d = x["value"] - y["value"]
                inside = y["lo"] <= x["value"] <= y["hi"]
                passed = abs(d) <= TOL
                ok &= passed
                rows.append({"vllm": new, "hf": old, "split": s, "metric": m, "vllm_value": round(x["value"], 4),
                             "hf_value": round(y["value"], 4), "diff": round(d, 4), "inside_hf_ci": inside, "pass": passed})
                print(f"{'PASS' if passed else 'FAIL'} {old:24s} {s:16s} {m:32s} hf {y['value']:.3f} vllm {x['value']:.3f} "
                      f"diff {d:+.3f}{'' if inside else '  (outside HF CI)'}")
    out = Path("results/comparisons/B3_parity_vllm_vs_hf.json")
    out.write_text(json.dumps({"tolerance": TOL, "all_pass": ok, "rows": rows}, indent=1))
    print("ALL_PASS" if ok else "PARITY_FAIL", f"({sum(r['pass'] for r in rows)}/{len(rows)})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
