"""Provenance gap: does a model adopt a record more often when it arrives through the tool channel than when the identical
record text is pasted into a user message?

For every counterfactual pair (c_forged/c_trusted and w_forged/w_trusted share the same record text), "adopts" means the
final answer equals the value the record implies (the trusted twin's expected answer). Pooled over test_id,
test_template and test_structural. Gap = P(adopt | tool) - P(adopt | pasted), 95% cluster bootstrap over problem groups.

    python scripts/analysis_provenance.py  ->  results/comparisons/provenance_gap.json
"""
import json
from pathlib import Path

import numpy as np

R = Path("results")
DATA = Path("data/conviction_bench/v0.1")
SPLITS = ["test_id", "test_template", "test_structural"]


def pairs():
    eps = {}
    for s in SPLITS:
        for line in open(DATA / f"{s}.jsonl"):
            e = json.loads(line)
            eps[e["episode_id"]] = e
    out = []
    for eid, e in eps.items():
        if e["slot"] in ("c_forged", "w_forged"):
            t = eid.replace("_forged", "_trusted")
            if t in eps:
                out.append((e["split"] + ":" + e["group_id"], eid, t, str(eps[t]["expected_final_answer"]).strip()))
    return out


def main(keys):
    P = pairs()
    groups = sorted({g for g, *_ in P})
    gidx = {g: i for i, g in enumerate(groups)}
    rng = np.random.default_rng(0)
    boot = rng.integers(0, len(groups), size=(2000, len(groups)))
    out = {"method": __doc__.strip(), "n_pairs": len(P), "n_groups": len(groups), "models": {}}
    for k in keys:
        pred = {}
        for s in SPLITS:
            f = R / k / f"predictions_{s}.jsonl"
            for line in open(f):
                r = json.loads(line)
                pred[r["episode_id"]] = str(r["pred_answer"]).strip()
        tool = np.zeros(len(groups)); pasted = np.zeros(len(groups)); n = np.zeros(len(groups))
        for g, fe, te, val in P:
            if fe not in pred or te not in pred:
                continue
            i = gidx[g]
            tool[i] += pred[te] == val
            pasted[i] += pred[fe] == val
            n[i] += 1
        bt, bp, bn = tool[boot].sum(1), pasted[boot].sum(1), n[boot].sum(1)
        gap = (bt - bp) / bn
        out["models"][k] = {"tool": round(tool.sum() / n.sum(), 4), "pasted": round(pasted.sum() / n.sum(), 4),
                            "gap": round((tool.sum() - pasted.sum()) / n.sum(), 4),
                            "gap_lo": round(float(np.percentile(gap, 2.5)), 4),
                            "gap_hi": round(float(np.percentile(gap, 97.5)), 4), "n": int(n.sum())}
    Path("results/comparisons/provenance_gap.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "scripts")
    from export_site import MODELS
    res = main([m[0] for m in MODELS])
    for k, v in sorted(res["models"].items(), key=lambda kv: -kv[1]["gap"]):
        print(f"{k:42s} tool {v['tool']:.2f} pasted {v['pasted']:.2f} gap {v['gap']:+.2f} [{v['gap_lo']:+.2f}, {v['gap_hi']:+.2f}]")
