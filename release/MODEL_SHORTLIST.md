# Multi-model baselines — shortlist (pre-selected 2026-11-10)

ConvictionBench's provenance test only works if a model's chat template gives tool records a
channel that is **distinguishable from user text**. Each candidate's template was checked locally
(tokenizer only) on a real benchmark episode on 2026-11-10.

## Tier 1 — run as-is (native, distinguishable tool channel; no access blockers)

| Model | Params | Tool turn renders as | Purpose |
|---|---|---|---|
| `Qwen/Qwen3-0.6B` | 0.6B | `<\|im_start\|>user\n<tool_response>…` (same as our base) | scale sweep |
| `Qwen/Qwen3-4B` | 4B | same | scale sweep |
| `Qwen/Qwen3-8B` | 8B | same | scale sweep |
| `Qwen/Qwen2.5-7B-Instruct` | 7B | same | previous generation |
| `microsoft/Phi-4-mini-instruct` | 3.8B | `<\|tool\|>…` | other family |
| `ibm-granite/granite-3.3-8b-instruct` | 8B | `<\|start_of_role\|>tool…` | other family |

Conditions: `base` system prompt, non-thinking where applicable, greedy, ≤ 200 new tokens, all four
test splits + natural history (same as E0). Qwen3-1.7B (E0) is already done.

## Tier 2 — needs an action first

| Model | Blocker | Fix |
|---|---|---|
| `meta-llama/Llama-3.2-3B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct` | Gated repo | Arya accepts the Meta licence on HF and logs the pod in with his own read token (I won't handle the token) |
| `google/gemma-3-4b-it` (optionally 12B) | Template rejects the `tool` role | Per-model message adapter: render the record as a clearly labelled block (e.g. a fixed `[Authenticated record from <service>]` header). Results flagged "adapted channel" |
| `mistralai/Mistral-7B-Instruct-v0.3` | Template requires tool-call IDs | Adapter: synthesise an assistant tool call + 9-char `tool_call_id` before each record (arguably more realistic). Flagged "adapted channel" |

## Excluded (provenance not testable without changing what the model sees)

* `HuggingFaceTB/SmolLM3-3B`: the tool turn renders as a plain `user` turn, so forged and authentic records look identical.
* `allenai/OLMo-2-1124-7B-Instruct`: the template silently **drops** tool content.

These could still be reported on the non-record metrics (pressure flips, corrections, abstention)
with provenance columns marked N/A, but this adds little to the release.

## Optional extras

* `Qwen/Qwen3-14B`: needs ≥ 40 GB (A100 80 GB at ~$1.19/h, within the cap).
* **Thinking mode** for Qwen3-1.7B / 8B. This is outside the primary protocol (non-thinking), would be
  reported separately, and generations are much longer, so it costs more.
* Newer small open models released after mid-2026 should be checked on HF before running (same template test).

## Cost / time estimate (RTX 4090, $0.89/h; extrapolated from E0 ≈ 25 min for 1.7B)

| Set | GPU time | Cost |
|---|---|---|
| Tier 1 (6 models) | ~5–5.5 h total (≈ 2–2.5 h wall-clock on 2–3 parallel pods) | ~$4.5–5 |
| + Llama ×2 | +1.5 h | +$1.3 |
| + Gemma 4B + Mistral 7B (adapters) | +1.7 h, plus ~1 h of adapter work locally | +$1.5 |
| + Qwen3-14B (A100) | +1.5 h | +$1.8 |

Remaining plannable budget: ≈ $29.
