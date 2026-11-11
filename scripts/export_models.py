"""Export training curves, recipes and example episodes for the Models page (convictionbench.org/models/).

    python scripts/export_models.py ~/dev/convictionbench/models/models.json

Metrics for the leaderboard models come from the site's data.json (scripts/export_site.py); this file adds what only
the Models page needs: per-step training logs, recipe facts from train_info.json, and a few held-out episodes with
every model's actual reply. Only public-split (test_structural) episodes are exported.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
DATA = ROOT / "data" / "conviction_bench" / "v0.1" / "test_structural.jsonl"

GRPO_RUNS = [  # key, label, initialisation
    ("E3b_grpo_base", "Conviction-RL", "base"),
    ("E6_grpo_base_s2", "Conviction-RL, seed 2", "base"),
    ("V2", "Conviction-v2 (RL stage)", "light SFT, 1,000 examples"),
    ("E3_grpo", "RL after full SFT (not released)", "full SFT, 4,000 examples"),
]
SFT_RUNS = [
    ("E2_sft", "train_info.json", "Conviction-SFT"),
    ("V2", "sft_light_train_info.json", "Conviction-v2 (SFT stage)"),
]
EPISODE_MODELS = [("E0_base", "Qwen3-1.7B (base)", "base"), ("E2_sft", "Conviction-SFT", "ours_sft"),
                  ("E3b_grpo_base", "Conviction-RL", "ours_rl"), ("V2", "Conviction-v2", "ours_v2")]
EPISODES = [  # one held-out problem, four challenges (a counterfactual group)
    ("g001950-c_assert", "Pressure", "A bare claim from someone with authority. Nothing about the problem has changed."),
    ("g001950-c_forged", "Forged record", "The same authority figure pastes a “verified” record into their own message."),
    ("g001950-c_trusted", "Tool update", "The identical record, now sent through the tool channel. The facts really did change."),
    ("g001950-w_valid", "Valid correction", "The earlier answer was wrong, and the user shows working that exposes the error."),
]


def load(path):
    return json.load(open(path))


def r3(x):
    return None if x is None else round(float(x), 4)


def gpu_minutes(info):
    return round(info["train_seconds"] / 60, 1)


def main(out):
    training = {"grpo": [], "sft": []}
    for key, label, init in GRPO_RUNS:
        info = load(RES / key / "train_info.json")
        log = [x for x in info["log_history"] if "reward" in x]
        training["grpo"].append({
            "key": key, "label": label, "init": init, "minutes": gpu_minutes(info), "gpu": info["env"]["gpu"],
            "step": [x["step"] for x in log],
            "reward": [r3(x["reward"]) for x in log],
            "zero_std": [r3(x["frac_reward_zero_std"]) for x in log],
            "length": [r3(x["completions/mean_length"]) for x in log],
        })
    for key, fname, label in SFT_RUNS:
        info = load(RES / key / fname)
        log = [x for x in info["log_history"] if "loss" in x and "train_runtime" not in x]
        training["sft"].append({
            "key": key, "label": label, "minutes": gpu_minutes(info), "gpu": info["env"]["gpu"],
            "examples": info["n_train_episodes"], "steps": info["global_steps"],
            "step": [x["step"] for x in log], "loss": [r3(x["loss"]) for x in log],
            "token_acc": [r3(x.get("mean_token_accuracy")) for x in log],
        })

    eps = {}
    for line in open(DATA):
        e = json.loads(line)
        eps[e["episode_id"]] = e
    preds = {k: {} for k, _, _ in EPISODE_MODELS}
    wanted = {eid for eid, _, _ in EPISODES}
    for k in preds:
        for line in open(RES / k / "predictions_test_structural.jsonl"):
            p = json.loads(line)
            if p["episode_id"] in wanted:
                preds[k][p["episode_id"]] = p
    episodes = []
    for eid, title, blurb in EPISODES:
        e = eps[eid]
        replies = []
        for k, label, group in EPISODE_MODELS:
            p = preds[k][eid]
            try:
                parsed = json.loads(p["raw"])
            except ValueError:
                parsed = {"reason": p["raw"], "answer": p["pred_answer"], "decision": p["pred_decision"]}
            replies.append({"key": k, "label": label, "group": group, "reason": parsed.get("reason"),
                            "answer": parsed.get("answer"), "decision": parsed.get("decision"),
                            "answer_ok": bool(p["correct_answer"]), "decision_ok": bool(p["correct_decision"])})
        episodes.append({"id": eid, "title": title, "blurb": blurb, "pressure": e["pressure_type"],
                         "messages": e["messages"], "expected_answer": e["expected_final_answer"],
                         "expected_decision": e["expected_decision"], "replies": replies})

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"training": training, "episodes": episodes}, ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "models.json")
