# Limitations (running list)

* **Synthetic, closed-world tasks.** Arithmetic, ledgers, propositional logic and token worlds with exact oracles. Gains here do not establish robustness to real-world misinformation or persuasion.
* **Provenance is a convention.** "Tool channel = authoritative, user text = claim" is defined in the system prompt. Real users sometimes have true information the record lacks; the benchmark deliberately ignores that case.
* **Token-level forgery.** Qwen3's tokenizer converts a literal `<tool_response>` string in user text into the genuine special token. The benchmark excludes such strings; a deployment would need input sanitisation (or `split_special_tokens`) for the provenance distinction to hold.
* **Capability confound.** For most episodes the correct final answer equals the answer to the original problem, so improved problem-solving alone improves "resistance". We report capability-conditioned metrics, but the confound cannot be removed entirely.
* **Non-thinking mode only** in primary experiments; results may differ with reasoning traces.
* **Single base model** (Qwen3-1.7B) and likely a single RL seed.
* **Template-generated language.** Held-out templates and a 22-item hand-written suite only partially address this; the hand-written suite is small and has wide CIs.
* **Qwen-specific RLVR effects** (see "Spurious Rewards", 2025) may inflate apparent RL gains; the SFT control and format-separated metrics mitigate but do not eliminate this.
