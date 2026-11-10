# Multi-model baselines: shortlist (revised 2026-11-10)

ConvictionBench's provenance test only works if a model's chat template gives tool records a
channel that is **distinguishable from user text**. Every candidate below was checked locally
(tokenizer only, no GPU) on a real benchmark episode containing a tool turn. The newest releases
were found by listing each lab's Hugging Face org on 2026-11-10.

Protocol for every model (same as E0): `base` system prompt, **non-thinking** (`enable_thinking=False`
where the template supports it), greedy, ≤ 200 new tokens, all four test splits + natural history.
Qwen3-1.7B (E0) is already done.

## Proposed bench: 10 models, 8 labs

| # | Model | Params | Lab · released | Tool turn renders as | Access |
|---|---|---|---|---|---|
| 1 | `Qwen/Qwen3-8B` | 8.2B | Alibaba · 2025-04 | `user` + `<tool_response>` special tokens (same as our base) | open |
| 2 | `Qwen/Qwen3.5-4B` | 4.7B | Alibaba · 2026-02 | same | open |
| 3 | `Qwen/Qwen3.5-9B` | 9.4B | Alibaba · 2026-02 | same | open |
| 4 | `meta-llama/Llama-3.2-3B-Instruct` | 3.2B | **Meta** · 2024-09 | `ipython` role header (from the published template; gated, so not run locally) | **gated (auto-approve)** |
| 5 | `facebook/MobileLLM-Pro` | 1.1B | **Meta** · 2025-09 | `<\|header_start\|>tool<\|header_end\|>` (checked via the public `ml-resources` mirror's tokenizer) | **gated (manual approval)**, FAIR non-commercial |
| 6 | `microsoft/Phi-4-mini-instruct` | 3.8B | Microsoft · 2025-02 | `<\|tool\|>` | open |
| 7 | `ibm-granite/granite-4.2-8b` | 8.8B | IBM · 2026-08 | `user` + `<tool_response>` | open |
| 8 | `mistralai/Ministral-3-8B-Instruct-2512-BF16` | 8.9B | Mistral · 2025-10 | `[TOOL_RESULTS]…[/TOOL_RESULTS]` (native, **no adapter needed**) | open |
| 9 | `allenai/Olmo-3-7B-Instruct` | 7.3B | Ai2 · 2025-11 | `environment` role (fully open data) | open |
| 10 | `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` | 4.0B | NVIDIA · 2026-03 | `user` + `<tool_response>` (hybrid Mamba-2) | open |

Why this set: it gives a Qwen scale and generation sweep (1.7B → 4B → 8B/9B, Qwen3 vs Qwen3.5); eight labs; two
Meta models (the current small instruct Llama plus Meta's newest small instruct release); four 2026 releases; and
one fully open-data model (Olmo 3). Ministral-3 replaces Mistral-7B-v0.3, and Olmo-3 replaces OLMo-2, because
both newer templates render the tool channel natively. That removes the adapter work from the earlier plan.

**What Arya needs to do for the two Meta models:** accept the licences on Hugging Face for
`meta-llama/Llama-3.2-3B-Instruct` (approved automatically) and `facebook/MobileLLM-Pro` (Meta approves
manually, which can take hours or days, so request it now). Then log the pod in with your own read token.
I won't handle the token. Without the Meta models, the bench is 8 models.

### Engineering notes (verify in a one-episode smoke run on the pod before each full eval)
* `Ministral-3` is a `Mistral3ForConditionalGeneration` checkpoint. `runner.py` needs an
  `AutoModelForImageTextToText` fallback (text-only use), plus `fix_mistral_regex=True` on the tokenizer.
* The `Qwen3.5` checkpoints are vision-language models. Transformers 5.22 maps them to `Qwen3_5ForCausalLM`;
  confirm the weights load without missing keys.
* `MobileLLM-Pro` ships custom code (`trust_remote_code`) from Meta's official repo. `Nemotron-3-Nano` has a native
  class, but runs much faster with `mamba-ssm`/`causal-conv1d` kernels; without them it falls back to a slow path.
* `granite-4.2` and `Nemotron` are reasoning models with thinking on by default. `enable_thinking=False` renders an
  empty `<think></think>`, as for Qwen3. Report them as non-thinking, like every other model.

## Alternates (swap in if a model fails its smoke run or Meta approval is slow)

| Model | Note |
|---|---|
| `Qwen/Qwen3-4B`, `Qwen/Qwen3-0.6B` | finer Qwen3 scale sweep |
| `LiquidAI/LFM3-1.7B` (2026-10) | same size as our trained model; has a native `tool` role; non-Transformer hybrid, custom code, LFM licence |
| `facebook/MobileMoE-L-SFT` (5.3B MoE, 2026-06) | **newest Meta small model**; gated, FAIR NC, custom code. Its remote code fails to load its config under transformers 5.22 (`max_position_embeddings` missing), so it needs a pinned older transformers. Template renders the `tool` role distinctly |
| `microsoft/Phi-tiny-MoE-instruct` | `<\|tool\|>` channel; MoE |
| `ibm-granite/granite-4.2-3b`, `Qwen/Qwen3.5-2B` | smaller variants of chosen families |
| `meta-llama/Llama-3.1-8B-Instruct` | gated; an 8B Meta point |

## Excluded

* **Gemma 4 (E2B, E4B, 12B) and Gemma 3:** the Gemma 4 template **silently drops** `tool` turns (the record text never
  reaches the model), and Gemma 3 rejects the role outright. Testing them would require an adapted channel.
* `HuggingFaceTB/SmolLM3-3B`: renders tool turns as plain `user` turns.
* `allenai/OLMo-2-1124-7B-Instruct`: drops tool content (superseded by Olmo 3).
* `facebook/MobileLLM-R1/R1.5`: math and code reasoning models, not general instruct.
* Meta Llama 4 (Scout and Maverick, 109B+) and Muse: there's no small open-weight model. Meta's 2026 small releases are MobileMoE, plus
  the 2025 MobileLLM-Pro.

## Cost and time estimate (RTX 4090 ≈ $0.89/h; scaled from E0 ≈ 25 min for 1.7B)

| Set | GPU time | Cost | Wall-clock |
|---|---|---|---|
| 8 open models | ≈ 8 h | ≈ $7–8 | ≈ 3 h on 3 parallel pods |
| + 2 Meta models | +1 h | +$1 | same pods |
| Total (10) incl. setup/downloads | ≈ 9–10 h | **≈ $9–10** | ≈ 3–3.5 h |

Remaining plannable budget: ≈ $29. Eval-time estimates for the 8–9B models (≈ 60–75 min each) are
extrapolated, not measured. I'll measure them on the first pod and re-estimate before starting the rest.
