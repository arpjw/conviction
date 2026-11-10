"""Results tables and figures 1-6 from measured outputs in results/.

    uv run --extra analysis python -m conviction.analysis.report

Reads results/<EXP>/metrics.json, breakdown_*.json, train_info.json and
results/budget_ledger.json. Experiments whose results are missing are skipped,
so the script can be re-run as experiments land. Nothing here invents numbers:
every plotted value is read from a results file.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = Path("results")
FIG = RES / "figures"
TAB = RES / "tables"

# Fixed experiment -> colour mapping (validated categorical slots; baselines in neutral grays).
EXPERIMENTS = [
    # key, label, colour, marker, measured pod-session cost chain (ledger notes substrings)
    ("E0_base", "E0 base", "#8a8984", "o"),
    ("E1_guided", "E1 prompted", "#b5b4ae", "o"),
    ("E2_sft", "E2 SFT", "#2a78d6", "s"),
    ("E3_grpo", "E3 SFT→GRPO", "#eb6834", "D"),
    ("E3c_sft_control", "E3c SFT→SFT (control)", "#1baf7a", "^"),
    ("E3b_grpo_base", "E3b GRPO only (seed 1)", "#6250d6", "P"),
    ("E6_grpo_base_s2", "E6 GRPO only (seed 2)", "#9085e9", "X"),
]
SPLITS = [("test_id", "In-distribution"), ("test_template", "Held-out templates"),
          ("test_structural", "Held-out subtypes")]
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# Measured cost per experiment = pod sessions that produced it (training + its evaluation),
# cumulative over prerequisite stages. Matched against ledger notes.
COST_CHAIN = {
    "E2_sft": ["E2 SFT"],
    "E3_grpo": ["E2 SFT", "E3 GRPO smoke"],
    "E3c_sft_control": ["E2 SFT", "E3 control"],
    "E3b_grpo_base": ["E3b GRPO from base"],
    "E6_grpo_base_s2": ["E6 second seed"],
}


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
    })


def load():
    out = {}
    for key, *_ in EXPERIMENTS:
        p = RES / key / "metrics.json"
        if p.exists():
            out[key] = {"metrics": json.load(open(p))}
            for sp, _ in SPLITS:
                b = RES / key / f"breakdown_{sp}.json"
                if b.exists():
                    out[key].setdefault("breakdown", {})[sp] = json.load(open(b))
            t = RES / key / "train_info.json"
            if t.exists():
                out[key]["train"] = json.load(open(t))
    return out


def costs() -> dict[str, float]:
    led = json.load(open(RES / "budget_ledger.json"))
    entries = led["entries"]
    out = {}
    for key, needles in COST_CHAIN.items():
        tot, found = 0.0, 0
        for n in needles:
            m = [e for e in entries if n in (e.get("note") or "")]
            if m:
                tot += m[0]["cost"]; found += 1
        if found == len(needles):
            out[key] = tot
    return out


def val(D, key, split, metric, view="all"):
    try:
        v = D[key]["metrics"][split][view][metric]
        return v if v.get("value") is not None else None
    except KeyError:
        return None


# ---------------------------------------------------------------- tables
TABLE_METRICS = [
    ("final_accuracy", "Final acc."), ("clean_accuracy", "Clean acc."),
    ("pressure_flip_rate", "Flip ↓"), ("legitimate_correction_rate", "Legit. corr."),
    ("state_changing_update_accuracy", "Auth. update"), ("fabricated_evidence_resistance", "Forged resist."),
    ("insufficient_accuracy", "Insufficient"), ("false_abstention_rate", "False abst. ↓"),
]


def tables(D, C):
    TAB.mkdir(parents=True, exist_ok=True)
    rows_csv = ["experiment,split,metric,value,lo,hi,n,clusters"]
    md = ["# Main results (ConvictionBench v0.1)", "",
          "Point estimate with 95% cluster-bootstrap CI (resampling problem groups). ↓ = lower is better.", ""]
    for sp, spname in SPLITS + [("test_structural_natural", "Held-out subtypes, natural history")]:
        md += [f"## {spname} (`{sp}`)", "", "| Experiment | " + " | ".join(n for _, n in TABLE_METRICS) + " |",
               "|---|" + "---|" * len(TABLE_METRICS)]
        for key, label, *_ in EXPERIMENTS:
            if key not in D:
                continue
            cells = []
            for m, _ in TABLE_METRICS:
                v = val(D, key, sp, m)
                if v is None:
                    cells.append("–"); continue
                cells.append(f"{v['value']:.3f} [{v['lo']:.2f}, {v['hi']:.2f}]")
                rows_csv.append(f"{key},{sp},{m},{v['value']},{v['lo']},{v['hi']},{v['n']},{v.get('clusters', '')}")
            md.append(f"| {label} | " + " | ".join(cells) + " |")
        md.append("")
    md += ["## Measured cost (pod sessions incl. evaluation; cumulative over prerequisite stages)", "",
           "| Experiment | USD |", "|---|---|"]
    for key, label, *_ in EXPERIMENTS:
        if key in C:
            md.append(f"| {label} | {C[key]:.2f} |")
    (TAB / "main_results.md").write_text("\n".join(md) + "\n")
    (TAB / "main_results.csv").write_text("\n".join(rows_csv) + "\n")


# ---------------------------------------------------------------- figures
def fig1(D):
    metrics = [("final_accuracy", "Final accuracy ↑"), ("pressure_flip_rate", "Pressure flip rate ↓"),
               ("legitimate_correction_rate", "Legitimate correction ↑"),
               ("insufficient_accuracy", "Insufficient-evidence acc. ↑")]
    exps = [e for e in EXPERIMENTS if e[0] in D]
    fig, axes = plt.subplots(len(SPLITS), len(metrics), figsize=(11, 7.2), sharey=True)
    for r, (sp, spname) in enumerate(SPLITS):
        for c, (m, mname) in enumerate(metrics):
            ax = axes[r][c]
            for i, (key, label, col, _) in enumerate(exps):
                v = val(D, key, sp, m)
                if v is None:
                    continue
                ax.bar(i, v["value"], color=col, width=0.72, edgecolor="white", linewidth=1)
                ax.plot([i, i], [v["lo"], v["hi"]], color=TEXT, linewidth=0.9)
            ax.set_xticks([]); ax.set_ylim(0, 1.05); ax.grid(axis="x", visible=False)
            if r == 0:
                ax.set_title(mname, fontsize=9, color=TEXT)
            if c == 0:
                ax.set_ylabel(spname)
    handles = [plt.Rectangle((0, 0), 1, 1, color=e[2]) for e in exps]
    fig.legend(handles, [e[1] for e in exps], loc="lower center", ncol=len(exps), fontsize=8,
               bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Figure 1 — Main comparison (95% cluster-bootstrap CIs)", x=0.01, ha="left", color=TEXT)
    fig.savefig(FIG / "fig1_main_comparison.png"); plt.close(fig)


def fig2(D):
    fig, axes = plt.subplots(1, len(SPLITS), figsize=(11, 3.9), sharex=True, sharey=True)
    for ax, (sp, spname) in zip(axes, SPLITS):
        for key, label, col, mk in EXPERIMENTS:
            x, y = val(D, key, sp, "legitimate_correction_rate"), val(D, key, sp, "pressure_flip_rate")
            if x is None or y is None:
                continue
            ry, rlo, rhi = 1 - y["value"], 1 - y["hi"], 1 - y["lo"]
            ax.errorbar(x["value"], ry, xerr=[[x["value"] - x["lo"]], [x["hi"] - x["value"]]],
                        yerr=[[ry - rlo], [rhi - ry]], fmt=mk, color=col, ms=8, mec="white", mew=1,
                        elinewidth=0.8, capsize=0, label=label)
        ax.set_title(spname, fontsize=9, color=TEXT)
        ax.set_xlabel("Legitimate correction rate (receptiveness) →")
        ax.set_xlim(0.75, 1.02); ax.set_ylim(0.4, 1.02)
        ax.axhline(1, color=GRID, lw=0.8); ax.axvline(1, color=GRID, lw=0.8)
    axes[0].set_ylabel("1 − pressure flip rate (resistance) →")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("Figure 2 — Resistance vs. receptiveness (ideal: top-right)", x=0.01, ha="left", color=TEXT)
    fig.savefig(FIG / "fig2_resistance_vs_receptiveness.png"); plt.close(fig)


def fig3(D):
    order = ["neutral", "authority", "evaluator", "consensus", "emotional", "impatience", "threat",
             "repetition", "flattery"]
    exps = [e for e in EXPERIMENTS if e[0] in D]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, sp in zip(axes, ["test_id", "test_template"]):
        n = len(exps); w = 0.8 / n
        for i, (key, label, col, _) in enumerate(exps):
            bd = D[key].get("breakdown", {}).get(sp, {}).get("pressure_type", {})
            xs, ys = [], []
            for j, p in enumerate(order):
                v = bd.get(p, {}).get("pressure_flip_rate")
                v = v.get("value") if isinstance(v, dict) else v
                if v is not None:
                    xs.append(j + (i - (n - 1) / 2) * w); ys.append(v)
            ax.bar(xs, ys, width=w * 0.92, color=col, label=label)
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=35, ha="right")
        ax.set_title("In-distribution" if sp == "test_id" else "Held-out templates (incl. unseen 'flattery')",
                     fontsize=9, color=TEXT)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Pressure flip rate ↓")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(exps), fontsize=8, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("Figure 3 — Susceptibility by pressure framing", x=0.01, ha="left", color=TEXT)
    fig.savefig(FIG / "fig3_by_pressure_type.png"); plt.close(fig)


def fig4(D):
    metrics = [("final_accuracy", "Final accuracy ↑"), ("pressure_flip_rate", "Pressure flip rate ↓"),
               ("legitimate_correction_rate", "Legitimate correction ↑")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    xs = range(len(SPLITS))
    for ax, (m, mname) in zip(axes, metrics):
        for key, label, col, mk in EXPERIMENTS:
            pts = [val(D, key, sp, m) for sp, _ in SPLITS]
            if any(p is None for p in pts):
                continue
            ax.plot(xs, [p["value"] for p in pts], color=col, lw=2, marker=mk, ms=7, mec="white", label=label)
        ax.set_xticks(list(xs)); ax.set_xticklabels(["In-dist.", "Held-out\ntemplates", "Held-out\nsubtypes"])
        ax.set_ylim(0, 1.03); ax.set_title(mname, fontsize=9, color=TEXT)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.16))
    fig.suptitle("Figure 4 — Generalisation gap", x=0.01, ha="left", color=TEXT)
    fig.savefig(FIG / "fig4_generalization_gap.png"); plt.close(fig)


def _roll(xs, k=10):
    return [sum(xs[max(0, i - k + 1): i + 1]) / len(xs[max(0, i - k + 1): i + 1]) for i in range(len(xs))]


def fig5(D):
    series = [("reward", "Mean reward"), ("frac_reward_zero_std", "Zero-variance groups (no gradient)"),
              ("completions/mean_length", "Mean completion length (tokens)")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for key, label, col, _ in EXPERIMENTS:
        tr = D.get(key, {}).get("train")
        if not tr or "reward_weights" not in tr:
            continue
        rows = [h for h in tr["log_history"] if "reward" in h]
        steps = [h["step"] for h in rows]
        for ax, (m, mname) in zip(axes, series):
            ys = [float(h[m]) for h in rows]
            ax.plot(steps, ys, color=col, lw=0.6, alpha=0.25)
            ax.plot(steps, _roll(ys), color=col, lw=2, label=label)
            ax.set_title(mname, fontsize=9, color=TEXT); ax.set_xlabel("GRPO step")
    axes[0].set_ylim(0, 1); axes[1].set_ylim(0, 1)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("Figure 5 — GRPO training dynamics (thin: per step; thick: 10-step mean)",
                 x=0.01, ha="left", color=TEXT)
    fig.savefig(FIG / "fig5_training_dynamics.png"); plt.close(fig)


def fig6(D, C):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharex=True)
    for ax, sp in zip(axes, ["test_id", "test_structural"]):
        base = val(D, "E0_base", sp, "final_accuracy")
        for key, label, col, mk in EXPERIMENTS:
            if key not in C:
                continue
            v = val(D, key, sp, "final_accuracy")
            if v is None or base is None:
                continue
            gain = v["value"] - base["value"]
            ax.plot(C[key], gain, mk, color=col, ms=9, mec="white", label=label)
            ax.annotate(label.split(" ")[0], (C[key], gain), xytext=(5, 4), textcoords="offset points",
                        fontsize=7.5, color=MUTED)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title("In-distribution" if sp == "test_id" else "Held-out subtypes", fontsize=9, color=TEXT)
        ax.set_xlabel("Measured GPU spend (USD, incl. evaluation)")
    axes[0].set_ylabel("Final accuracy gain over E0")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.16))
    fig.suptitle("Figure 6 — Improvement vs. measured compute cost", x=0.01, ha="left", color=TEXT)
    fig.savefig(FIG / "fig6_compute_efficiency.png"); plt.close(fig)


def main() -> int:
    _style()
    FIG.mkdir(parents=True, exist_ok=True)
    D, C = load(), costs()
    tables(D, C)
    for f in (fig1, fig2, fig3, fig4, fig5):
        f(D)
    fig6(D, C)
    print("experiments:", ", ".join(D)); print("costs:", {k: round(v, 2) for k, v in C.items()})
    print(f"wrote {TAB}/main_results.md and {len(list(FIG.glob('*.png')))} figures to {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
