# Conviction

Can a small language model learn **when to stand its ground, when to change its mind, and when to admit it doesn't know** — revising beliefs on *evidence* rather than *social pressure*? And does RL (GRPO with verifiable rewards) help beyond SFT?

Base model: `Qwen/Qwen3-1.7B` (non-thinking mode). Budget: ≤ $40 of GPU time.

**Status:** benchmark + training/eval infrastructure built and tested locally. No GPU experiments run yet; no results exist. See [research/EXPERIMENTS.md](research/EXPERIMENTS.md) for the protocol and decision log and [research/COMPUTE_PLAN.md](research/COMPUTE_PLAN.md) for the budget plan.

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
bash scripts/generate_data.sh
```

GPU (pod): `bash scripts/setup.sh`, then `scripts/smoke_test.sh`, `run_baseline.sh`, `run_sft.sh`, `run_grpo.sh`.

## Layout

```
conviction/data/        oracles, templates, generators, splits, validation, adversarial suite
conviction/training/    rewards (deterministic), SFT, GRPO (TRL), data selection
conviction/evaluation/  runner (HF generate), metrics, cluster bootstrap, natural-history battery
conviction/utils/       budget ledger, logging
configs/ scripts/ tests/ research/
```

License: Apache-2.0 (code). Base model weights: Qwen3, Apache-2.0.
