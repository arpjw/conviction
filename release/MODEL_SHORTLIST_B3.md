# Bench expansion B3: candidate models (scanned 2026-11-10, not yet run)

Goal: grow the open-model leaderboard from 18 to about 36–44 models before external collaboration. More labs, more
architectures (dense, MoE, hybrid Mamba, looped) and clean scale sweeps.

## How candidates were found and checked

1. **Scan.** Listed ~60 Hugging Face orgs, newest repos first, plus targeted searches for the big labs. Kept
   instruct/chat models ≤ ~15B and dropped quantised, vision-only, embedding, coder and math variants.
2. **Template check (the gate).** Rendered a real benchmark episode with a `tool` turn through each model's own chat
   template. Only the template **text** was downloaded, and it was rendered in a sandboxed Jinja environment, so no
   repository code ran. A model passes only if the tool record reaches the model in a channel that is
   distinguishable from user text, and the system prompt survives.
3. **Config check.** Read `config.json` for architecture and custom code (`auto_map`), and the card for licence and
   purpose. Models that need `trust_remote_code` are deferred: running repo code on the pod adds risk and debugging
   time.

## Tier 1: recommended (18 models, 10 new labs)

| Model | Params | Lab · released | Why | Tool turn renders as |
|---|---|---|---|---|
| `Qwen/Qwen3-0.6B` | 0.6B | Alibaba · 2025-04 | Qwen3 scale sweep, low end (0.6 → 1.7 → 8 → 14B) | `user` + `<tool_response>` |
| `Qwen/Qwen3-4B-Instruct-2507` | 4.0B | Alibaba · 2025-08 | Instruct-only Qwen3 refresh | `user` + `<tool_response>` |
| `Qwen/Qwen3-14B` | 14.8B | Alibaba · 2025-04 | Qwen3 scale sweep, high end | `user` + `<tool_response>` |
| `meta-llama/Llama-3.2-1B-Instruct` | 1.2B | Meta · 2024-09 | Llama scale sweep (1B vs the 3B already run); access already granted | `ipython` role |
| `mistralai/Ministral-3-14B-Instruct-2512-BF16` | 13.9B | Mistral · 2025-10 | Completes the Ministral-3 sweep (3B, 8B, 14B) | `[TOOL_RESULTS]` |
| `ibm-granite/granite-4.2-3b` | 3.7B | IBM · 2026-08 | Granite scale sweep (3B vs 8B) | `user` + `<tool_response>` |
| `ibm-granite/granite-4.0-h-tiny` | 6.9B (MoE, 64 experts) | IBM · 2025-09 | Hybrid Mamba MoE | `<tool_response>` |
| `nvidia/NVIDIA-Nemotron-Nano-9B-v2` | 8.9B | NVIDIA · 2025-08 | NVIDIA's 9B hybrid (vs the 4B already run) | `<TOOL_RESPONSE>` |
| `LiquidAI/LFM2.5-1.2B-Instruct` | 1.2B | Liquid · 2026-01 | Liquid's previous generation (vs LFM3) | native `tool` role |
| `LiquidAI/LFM2.5-8B-A1B` | 8.5B (MoE, 1B active) | Liquid · 2026-05 | Small-active-parameter MoE | native `tool` role |
| `tencent/Hunyuan-1.8B-Instruct` | 1.8B | **Tencent** · 2025-07 | New lab | `<tool_response>` |
| `openbmb/MiniCPM5-2B` | 2.5B | **OpenBMB** · 2026-09 | New lab; very popular on-device model | `user` + `<tool_response>` |
| `kakaocorp/kanana-2-3b-instruct` | 3.5B | **Kakao** · 2026-07 | New lab; trained from scratch, SFT + RL | `user` + `<tool_response>` |
| `tiiuae/Falcon-H1-7B-Instruct` | 7.6B | **TII** · 2025-05 | New lab; hybrid attention–Mamba | native `tool` role |
| `arcee-ai/AFM-4.5B` | 4.6B | **Arcee** · 2025-07 | New lab | native `tool` role |
| `utter-project/EuroLLM-9B-Instruct-2512` | 9.2B | **EuroLLM consortium** · 2026-01 | New lab; European multilingual | native `tool` role |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 1.7B | **Hugging Face** · 2024-10 | Same size as our trained models. The template has a `tool` role, but the model was not trained on tools: a useful floor | native `tool` role |
| `LGAI-EXAONE/EXAONE-4.0-1.2B` | 1.3B | **LG AI** · 2025-07 | New lab; non-commercial licence (research use is fine) | `[\|tool\|]` + `<tool_result>` |

New labs relative to the current 18: Tencent, OpenBMB, Kakao, TII, Arcee, EuroLLM, Hugging Face, LG (plus Salesforce
and Xiaomi in Tier 2). The current leaderboard has 8 labs.

## Tier 2: cheap extras (8 models)

| Model | Why |
|---|---|
| `Qwen/Qwen3-4B` (hybrid, 2025-04) | Paired with `Qwen3-4B-Instruct-2507`, it isolates the effect of the 2507 instruct refresh |
| `tencent/Hunyuan-0.5B-Instruct`, `openbmb/MiniCPM5-1B`, `tiiuae/Falcon-H1-3B-Instruct`, `LiquidAI/LFM2.5-2.6B`, `ibm-granite/granite-4.0-1b` | Within-family scale points |
| `Salesforce/xgen-small-4B-instruct-r` | Another lab (Salesforce); CC-BY-NC |
| `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B` | Distilled from a frontier model on a Qwen3.5-9B base: does distillation change conviction? Probably belongs in the thinking track |

## Needs your action first (gated repos)

| Model | Gate | Note |
|---|---|---|
| `meta-llama/Llama-3.1-8B-Instruct` | manual (Meta) | An 8B Meta point; Llama 3.1 is a separate licence from 3.2 |
| `swiss-ai/Apertus-v1.5-8B` | auto | Fully open Swiss model. Template not yet checked (gated); its v1.1 template rejects the `tool` role |
| `CohereLabs/tiny-aya-global` | auto | Cohere's 3.3B. Template not yet checked (gated) |
| `facebook/MobileMoE-S-SFT` / `-M-SFT` | auto | Meta's newest small models. Custom code; the L variant failed to load under transformers 5.22 earlier |

## Excluded (fails the provenance gate or out of scope)

| Model | Reason |
|---|---|
| `microsoft/phi-4`, `zai-org/GLM-4-9B-0414`, `nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B` | Template silently **drops** the tool turn |
| `tencent/Hunyuan-7B-Instruct` | Tool text is pasted after `<\|eos\|>` with no role marker, so it can't be distinguished from user text |
| `swiss-ai/Apertus-v1.1-*`, `BSC-LT/salamandra-7b-instruct-2606`, `ServiceNow-AI/Apriel-5B-Instruct` | Template rejects the `tool` role |
| `openai/gpt-oss-20b` | Harmony format requires every tool message to answer a prior assistant tool call. Feasible with a synthesised call, but that changes the channel, so it belongs to a separate protocol variant |
| Gemma 4 / Gemma 3, `SmolLM3-3B` | As before: tool turns dropped, rejected, or rendered as `user` |
| `microsoft/FrogNano-4B-2609` | Coding agent, not a general instruct model |
| `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B` | Looped (latent-recurrent) architecture: interesting, but likely no vLLM support. Stretch goal on the HF engine |
| `internlm3-8b-instruct`, `Trinity-Nano-Preview`, `Ling-3.0-tiny`, `Motif-2-12.7B`, `Instella-MoE-16B-A3B`, `Cola2` | Require `trust_remote_code`. Deferred |
| `DeepSeek-R1-0528-Qwen3-8B`, `Olmo-3-7B-Think`, `Phi-4-mini-reasoning` | Thinking-only; candidates for the thinking track, not the non-thinking leaderboard |

## Engine and cost

The current leaderboard used HF `generate`, at 12–50 minutes per model. That is too slow for this budget. vLLM ran
the thinking control about 8× faster, and matched HF within 0.04 there. Proposal:

1. **Parity check first (pre-registered).** Re-run 2 already-benched models (Qwen3-1.7B, Llama-3.2-3B) non-thinking
   on vLLM. If every headline metric agrees with the published HF numbers within the 95% intervals, run B3 on vLLM
   and label the engine in each `run_info.json` and on the site. If not, fall back to HF for the affected families.
2. **Hardware:** RTX A6000 48 GB, Secure Cloud, $0.59/h (live price and stock checked 2026-11-10: "Low" stock).
   It fits the 14B models in bf16. Two pods in parallel halve the wall-clock time at the same cost.

| Set | Est. GPU time (vLLM) | Est. cost |
|---|---|---|
| Parity check (2 models) | ~0.3 h | ~$0.20 |
| Tier 1 (18 models) incl. downloads and setup | ~3–4 h | **~$2–2.50** |
| Tier 2 (8 models) | ~1–1.5 h | ~$0.60–0.90 |
| **Everything** | ~4.5–6 h | **~$3–3.60** of the $13.75 left |

On the HF engine, the same Tier 1 would take ~10–14 h (≈ $6–8). These times are extrapolated from the thinking run
and the B runs, so I'll measure the first two models and re-estimate before starting the rest.
