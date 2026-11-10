---
license: apache-2.0
base_model: Qwen/Qwen3-1.7B
library_name: peft
tags: [lora, sycophancy, belief-revision, grpo, sft, convictionbench]
---

# conviction-1.7b

LoRA adapters for **Qwen/Qwen3-1.7B** (non-thinking mode) trained to revise answers on
*evidence* — valid corrections and authenticated tool records — rather than social pressure,
and to say when the available information is insufficient. Trained and evaluated on
[ConvictionBench v0.1](<dataset link>). Base model © Qwen team, Apache-2.0.

> **Which variant to release as `conviction-1.7b` is an open decision** (see trade-offs).
> Candidates, all r = 16 LoRA on all linear layers (17.4M trainable parameters):

| Variant | Recipe | Strengths | Weaknesses |
|---|---|---|---|
| `conviction-1.7b-sft` (E2) | SFT on 4,000 oracle reference responses | Near-ceiling in-distribution; resists forged in-message records (0.96); abstains correctly (0.96) | On unseen problem types becomes stubborn about its own wrong answers (accepts corrections 7% of the time in natural history) |
| `conviction-1.7b-rl` (E3b) | GRPO from base, deterministic reward, 200 steps | Best on held-out subtypes (final acc. 0.644); keeps accepting valid corrections (0.93; 0.73 in natural history) | Does not learn the provenance rule (forged-record resistance ≈ base, 0.19); weaker abstention (0.57) and in-distribution accuracy (0.68) |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype="bfloat16")
model = PeftModel.from_pretrained(model, "<hf-user>/conviction-1.7b-sft").merge_and_unload()
# Use the system prompt from conviction/prompts.py (SYSTEM_BASE) and
# tok.apply_chat_template(..., enable_thinking=False). Facts must arrive as `tool` messages.
```

## Evaluation (ConvictionBench v0.1, greedy, 95% cluster-bootstrap CIs in the repository)

| Metric | Base | SFT (E2) | GRPO only (E3b) |
|---|---|---|---|
| test_id final accuracy | 0.445 | 0.939 | 0.679 |
| test_template final accuracy | 0.425 | 0.913 | 0.651 |
| test_structural final accuracy | 0.424 | 0.580 | 0.644 |
| test_id pressure flip rate ↓ | 0.470 | 0.039 | 0.304 |
| test_structural pressure flip rate ↓ | 0.536 | 0.389 | 0.348 |
| test_structural legitimate correction | 0.937 | 0.857 | 0.926 |
| test_id fabricated-evidence resistance | 0.248 | 0.963 | 0.191 |
| test_id insufficient-evidence accuracy | 0.355 | 0.963 | 0.566 |

Second-seed replicate of the RL recipe (E6): <to be filled from results/E6_grpo_base_s2>.

## Training

* Data: disjoint halves of the ConvictionBench train split (SFT and RL never share problems).
* SFT: 1 epoch, 125 steps, lr 2e-4, effective batch 32, completion-only loss; 11 min on an RTX 3090.
* GRPO (TRL 1.17): 16 prompts × 4 samples per step, β = 0, DAPO loss, T = 1.0, reward
  0.70·answer + 0.25·decision + 0.05·format (all binary, computed by exact solvers); 200 steps,
  65 min on an RTX 4090.
* Total project compute ≈ $8 of rented GPU time; per-run costs in the repository.

## Intended use and limitations

Research artefact for studying sycophancy and evidence-sensitive belief revision. Trained only on
synthetic arithmetic/ledger/logic/token-world problems in a fixed JSON response format; behaviour
on open-ended conversation is untested. Treats the tool channel as authoritative by construction —
**do not deploy as a defence against manipulation**, and sanitise user input so that literal
`<tool_response>` strings cannot forge the tool channel.
