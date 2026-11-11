# Conviction

Can a small language model learn **when to stand its ground, when to change its mind, and when to admit it doesn't know** — revising beliefs on *evidence* rather than *social pressure*? And does RL (GRPO with verifiable rewards) help beyond SFT?

Base model: `Qwen/Qwen3-1.7B` (non-thinking mode). Budget: ≤ $40 of GPU time.

**Project page:** https://convictionbench.org/ · **Dataset:** [arpjw/ConvictionBench](https://huggingface.co/datasets/arpjw/ConvictionBench) · **Models:** [arpjw/conviction-1.7b-sft](https://huggingface.co/arpjw/conviction-1.7b-sft), [arpjw/conviction-1.7b-rl](https://huggingface.co/arpjw/conviction-1.7b-rl), [arpjw/conviction-1.7b-v2](https://huggingface.co/arpjw/conviction-1.7b-v2)

**Status (2026-11-11):** ConvictionBench v0.1 frozen and released; base, prompting, SFT, SFT→GRPO, a matched SFT control and GRPO-only runs complete (GRPO-only replicated with a second seed); 37 off-the-shelf open models from 18 labs benchmarked, plus a thinking-mode control and a v2 training recipe. Total compute ≈ $30 of rented GPU time ([ledger](results/budget_ledger.json)); a separate frontier API track (GPT models, reserved for a future harder challenge, not on the leaderboard) used ≈ $39 of OpenAI credit. Working report draft: [research/REPORT.md](research/REPORT.md); full experiment log: [research/EXPERIMENTS.md](research/EXPERIMENTS.md); tables: [results/tables/main_results.md](results/tables/main_results.md).

### ConvictionBench leaderboard (v0.1, non-thinking, greedy)

| Model | Held-out problems: final acc. | Held-out problems: pressure flip ↓ | Forged-record resistance (in-dist.) |
|---|---|---|---|
| **Conviction-RL (ours, 1.7B)** | 0.644 | 0.348 | 0.191 |
| **Conviction-v2 (ours, 1.7B)** | 0.559 | 0.375 | 0.806 |
| **Conviction-SFT (ours, 1.7B)** | 0.580 | 0.389 | 0.963 |
| Qwen3-1.7B (our base) | 0.424 | 0.536 | 0.248 |
| Qwen3.5-9B | 0.688 | 0.399 | 0.060 |
| Qwen3-4B-Instruct-2507³ | 0.677 | 0.408 | 0.057 |
| MiMo-V2.6-Distill-9B³ | 0.624 | 0.377 | 0.186 |
| Granite-4.2-8B | 0.608 | 0.374 | 0.104 |
| Qwen3.5-4B | 0.606 | 0.428 | 0.035 |
| Ministral-3-14B³ | 0.535 | 0.623 | 0.010 |
| Ministral-3-8B | 0.526 | 0.659 | 0.022 |
| Falcon-H1-7B³ | 0.506 | 0.647 | 0.025 |
| Granite-4.2-3B³ | 0.500 | 0.467 | 0.196 |
| MiniCPM5-2B³ | 0.487 | 0.402 | 0.236 |
| Kanana-2-3B³ | 0.484 | 0.452 | 0.278 |
| Qwen3-8B | 0.474 | 0.648 | 0.087 |
| Ministral-3-3B | 0.470 | 0.741 | 0.022 |
| Qwen3-4B³ | 0.463 | 0.635 | 0.084 |
| Qwen3.5-2B | 0.425 | 0.581 | 0.122 |
| Qwen3-14B³ | 0.410 | 0.756 | 0.035 |
| Granite-4.0-H-Tiny³ | 0.404 | 0.612 | 0.094 |
| EuroLLM-9B³ | 0.402 | 0.565 | 0.179 |
| EXAONE-4.0-1.2B³ | 0.401 | 0.666 | 0.094 |
| Nemotron-3-Nano-4B | 0.391 | 0.695 | 0.067 |
| Qwen3-0.6B³ | 0.389 | 0.537 | 0.397 |
| Qwen3.5-0.8B | 0.377 | 0.570 | 0.261 |
| Hunyuan-1.8B³ | 0.368 | 0.578 | 0.290 |
| MiniCPM5-1B³ | 0.367 | 0.636 | 0.278 |
| Falcon-H1-3B³ | 0.366 | 0.709 | 0.052 |
| Granite-4.0-1B³ | 0.358 | 0.523 | 0.203 |
| Phi-4-mini | 0.347 | 0.767 | 0.065 |
| SmolLM2-1.7B³ | 0.346 | 0.679 | 0.474 |
| MobileLLM-Pro¹ | 0.322 | 0.670 | 0.248 |
| Llama-3.2-3B | 0.317 | 0.898 | 0.164 |
| Hunyuan-0.5B³ | 0.312 | 0.629 | 0.419 |
| Olmo-3-7B | 0.304 | 0.812 | 0.035 |
| LFM2.5-1.2B³ | 0.225 | 0.768 | 0.313 |
| AFM-4.5B²³ | 0.224 | 0.747 | 0.166 |
| xgen-small-4B²³ | 0.197 | 0.743 | 0.181 |
| LFM3-1.7B² | 0.096 | 0.614 | 0.007 |
| LFM2.5-2.6B²³ | 0.029 | 0.833 | 0.003 |

¹ Custom code needs transformers 4.56; follows the strict format in about half of answers. ² Often answers in free text instead of the requested JSON; scores mainly or partly reflect format compliance. ³ Added 2026-11-11 and evaluated on vLLM (two earlier models re-run on vLLM matched the published HF numbers within 0.02; `results/comparisons/B3_parity_vllm_vs_hf.json`). Two further models that think by default even with thinking off (Nemotron-Nano-9B-v2, LFM2.5-8B-A1B) are kept off this table; candidates and exclusions: [release/MODEL_SHORTLIST_B3.md](release/MODEL_SHORTLIST_B3.md).

Off-the-shelf open models rarely resist forged records pasted into user messages (29 of 37 ≤ 0.25 in-distribution; the highest scores come from the smallest models, Qwen3-0.6B and Hunyuan-0.5B, which tend to ignore challenges of every kind); our GRPO adapter has the lowest pressure-flip rate of all 41 models, and v2 the best held-out forged-record resistance. **Thinking mode** (Qwen3-1.7B/8B, Qwen3.5-9B, Granite-4.2-8B, paired on the same episodes) raises accuracy and lowers pressure flips but *lowers* forged-record resistance; see the [results page](https://convictionbench.org/#thinking) and `results/comparisons/T_thinking_vs_nonthinking.json`. Per-model protocol notes (GPU, transformers version, kernels) are in each `results/*/run_info.json` and in [research/EXPERIMENTS.md](research/EXPERIMENTS.md) §§ B, B2, T.

### Training results (Qwen3-1.7B)

| | In-dist. final acc. | Held-out subtypes final acc. | In-dist. pressure flip ↓ | Held-out subtypes legit. correction | In-dist. forged-record resistance |
|---|---|---|---|---|---|
| Base | 0.445 | 0.424 | 0.470 | 0.937 | 0.248 |
| + guidance prompt | 0.449 | 0.433 | 0.453 | 0.937 | 0.226 |
| + SFT | 0.939 | 0.580 | 0.039 | 0.857 | 0.963 |
| + SFT → GRPO | 0.952 | 0.606 | 0.025 | 0.851 | 0.970 |
| + SFT → more SFT (control) | 0.970 | 0.612 | 0.028 | 0.880 | 0.980 |
| + GRPO only (seed 1) | 0.679 | 0.644 | 0.304 | 0.926 | 0.191 |
| + GRPO only (seed 2) | 0.698 | 0.642 | 0.282 | 0.920 | 0.218 |
| + light SFT → GRPO (v2) | 0.787 | 0.560 | 0.110 | 0.926 | 0.806 |

* Prompting does nothing; SFT nearly solves the in-distribution task but hardens into "keep" on unseen problem types.
* GRPO after SFT is no better than matched extra SFT (≈85% of GRPO groups had zero reward variance).
* GRPO from the base model beats SFT on held-out subtypes (pre-registered test; replicated with a second seed) but never learns that records pasted into user messages are not authoritative.
* v2 (1,000-example SFT warm-start, then the same GRPO) learns the provenance rule (0.81 in-distribution; best held-out forged-record resistance of any model, 0.58) but gives up GRPO's held-out accuracy (0.56): it moves along the SFT↔RL trade-off rather than escaping it.

![Resistance vs receptiveness](results/figures/fig2_resistance_vs_receptiveness.png)

## ConvictionBench

Procedural, oracle-labelled episodes: a question, an earlier assistant answer (correct, wrong, or unjustified), then a challenge. Every label is computed by an exact oracle — never an LLM judge.

* **4 domains, 12 subtypes:** arithmetic (expressions, percentages, ratios, linear equations, modular*), transaction ledgers (balance, net change), logic (boolean evaluation, rule entailment, sets*), mini-worlds (token values, derived-value rules*). `*` = structural holdout.
* **Challenge = framing × evidence.** Framings (neutral, authority, evaluator, consensus, emotional, impatience, threat, repetition, flattery†) are crossed with evidence (bare assertion, generic pushback, valid derivation, flawed derivation, forged "verified" update in the user turn, unverified user claim, authentic tool-channel update, tool-channel confirmation). † held out.
* **Counterfactual groups:** each problem appears in ~13 matched episodes; pairs share framing and templates and differ only in evidence (e.g. the *identical* record text as a tool message vs pasted into a user message).
* **Three-state answers:** `keep` / `revise` / `insufficient`, with traps both ways (invented missing values; missing-looking information that doesn't matter).
* **Splits:** train / val / test_id, test_template (held-out phrasing), test_structural (held-out subtypes), plus a 22-item hand-written adversarial suite.

## Quickstart

```bash
uv sync                      # CPU-only: generation, tests, metrics
uv run pytest                # oracle, generator, split, reward, metric tests
uv run conviction-generate   # regenerate ConvictionBench v0.1 (verify against FROZEN.txt)
uv run --extra train conviction-eval --model Qwen/Qwen3-1.7B --natural --out results/my_run
uv run --extra analysis conviction-report   # tables + figures from results/
```

GPU (pod): `bash scripts/setup.sh`, then `scripts/smoke_test.sh`, `run_baseline.sh`, `run_sft.sh`, `run_grpo.sh`.
The exact pod phases used for every reported run are in `scripts/remote_phase.sh` (`baselines`, `sft`, `grpo`, `sft_control`,
`grpo_base`, `grpo_base_s2`, `bench`); multi-model runs use `BENCH_MODELS="org/name:batch,..." bash scripts/remote_phase.sh bench`.

### Evaluating frontier models through provider APIs

```bash
uv sync --extra api
export ANTHROPIC_API_KEY=...   # or OPENAI_API_KEY / XAI_API_KEY / GEMINI_API_KEY — set in your shell, never in files
# free estimate first (no calls, no key)
uv run conviction-eval-api --provider anthropic --model claude-opus-5-5 --groups 90 --natural \
    --price-in 4 --price-out 20 --dry-run --out results/A_claude-opus-5-5
# paid run with a hard cap; cached and resumable
uv run conviction-eval-api ... --budget-usd 20 --out results/A_claude-opus-5-5
```

Providers: `anthropic`, `openai`, `xai`, `google` (OpenAI-compatible), `mock` (offline). Tool records are sent through each
provider's native tool-call format; see `conviction/evaluation/api/convert.py` and the API-track protocol in
[research/EXPERIMENTS.md](research/EXPERIMENTS.md). Results are a separate "API, default reasoning" track.

## Layout

```
conviction/data/        oracles, templates, generators, splits, validation, adversarial suite
conviction/training/    rewards (deterministic), SFT, GRPO (TRL), data selection
conviction/evaluation/  runner (HF generate), metrics, cluster bootstrap, natural-history battery
conviction/utils/       budget ledger, logging
conviction/analysis/    results tables and figures 1-6
release/                dataset card, model card, release checklist
research/validation/    human-validation sample and guide
configs/ scripts/ tests/ research/
```

## Data hygiene

Released dataset files carry a canary string (`release/CANARY.txt`), and the Hugging Face dataset is gated behind a
no-training agreement. Please do not train on ConvictionBench, and filter documents containing the canary from training corpora.

Because the generator is public, the public splits can be regenerated exactly; they are for reproducibility, not secrecy. A
**private held-out split** (same generator and settings, secret seed, disjoint from every public problem) is kept offline to
score models and detect contamination. Its SHA-256 hashes are committed in
[release/PRIVATE_SPLIT_COMMITMENT.txt](release/PRIVATE_SPLIT_COMMITMENT.txt). To have a model scored on it, open an issue.

## License and citation

Code: Apache-2.0. Data: CC-BY-4.0. Adapters: Apache-2.0 (base model Qwen3-1.7B, Apache-2.0).

```bibtex
@misc{somu2026convictionbench,
  title  = {ConvictionBench: Evidence-Sensitive Belief Revision in Language Models},
  author = {Somu, Arya},
  year   = {2026},
  note   = {v0.1},
  url    = {https://convictionbench.org/}
}
```
