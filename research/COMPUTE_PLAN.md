# Compute & Budget Plan (not yet executed — awaiting approval)

Hard cap **$40.00** (RunPod credits). Planning margin: never plan to spend the last $3.
The RunPod account balance is authoritative; `python -m conviction.utils.budget` is a planning ledger.

## Live prices (read via RunPod API, 2026-11-09; on-demand pod, per GPU-hour)

| GPU | VRAM | Community | Secure | Availability at read time |
|---|---|---|---|---|
| **RTX 4090** | 24 GB | **$0.34** | $0.89 | Community: LOW (CUDA 13.0 hosts); Secure: none |
| RTX A6000 | 48 GB | $0.33 | $0.59 | Secure: LOW (EU-SE-1); Community: none |
| A40 | 48 GB | $0.35 | $0.59 | Secure: LOW (CA-MTL-1, EU-SE-1) |
| RTX 3090 | 24 GB | $0.22 | $0.50 | none |
| RTX A5000 | 24 GB | $0.16 | $0.27 | none |
| L40S | 48 GB | $0.79 | $1.09 | none |
| RTX 5090 | 32 GB | $0.69 | $1.19 | Secure: LOW (EU-RO-1) |

Account state at read time: 0 pods, 0 network volumes (nothing accruing).

**Recommendation:** Community RTX 4090 at $0.34/h. bf16 throughput is roughly 2× an A6000/A40/3090
for this workload (Ada tensor cores, high clocks), at the same or similar price, so it is the cheapest per
unit of work. Fallback if no 4090 is available: Secure A6000/A40 at $0.59/h (48 GB also allows larger GRPO
batches). Do not use the 3090/A5000 unless they are the only option — cheaper per hour, slower per run.

## Storage policy
No network volume. Use the pod's container/volume disk (≈40 GB: model 3.4 GB + HF cache + checkpoints),
export adapters/logs/predictions with `runpodctl send` or `scp`, then **terminate** (not just stop) the pod.
A stopped pod still bills for its volume disk.

## Phase plan (runtime figures are ESTIMATES until the GPU smoke test measures them)

| Phase | Work | Est. GPU-h | Est. $ @0.34 | Cap |
|---|---|---|---|---|
| 0 Smoke | install, pin model revision, 20-step SFT + 5-step GRPO, eval 5 groups, record memory & tokens/s | 0.5 | 0.17 | $4 (with baselines) |
| 1 Baselines | E0 + E1 on 4 test splits (~8k episodes each) + natural history | 1.0 | 0.34 | ″ |
| 2 SFT | 4k episodes, 1 epoch (~125 steps), eval E2 | 0.75 | 0.26 | $7 |
| 3 GRPO pilot | 200 steps × 64 completions; stop if reward flat/collapsing | 2–3 | 0.7–1.0 | $15 |
| 3b GRPO extend | only if pilot curves justify (e.g. to 500–800 steps) | 3–5 | 1.0–1.7 | ″ |
| 4 Eval + ablations | E3 eval; then E4 (outcome-only), E5 (no pairs), E6 (seed 2) | 4–8 | 1.4–2.7 | $10 |
| Reserve | idle time, restarts, storage | — | — | $4 |
| **Total (est.)** | | **~12–19 h** | **~$4–7** | **$40** |

The estimates use HF `generate` (no vLLM) for GRPO rollouts. Even at 3× the estimates the core
E0–E3 plan stays under ~$10, so the budget is not the binding constraint — rigor is. We should not
"spend because we can": scale-ups require a pilot that justifies them.

## Stop conditions
* Smoke test: if peak memory > 22 GB at the pilot batch size, halve batch / increase accumulation before anything else; do **not** switch to 0.6B without approval.
* SFT: stop if val loss rises for 3 consecutive evals.
* GRPO: stop the pilot if, after 60 steps, mean reward has not moved beyond its first-10-step band, `frac_reward_zero_std` > 0.9 (no learning signal), or completion length collapses / explodes.
* Any phase: stop when its cap is reached; re-plan before continuing.
* Global: pod idle > 10 min → terminate.

## Approval needed
Creating the pod is a billed action. Proposed first session: **Community RTX 4090, $0.34/h, ~1.5 h
(smoke test + E0/E1 baselines), est. ≈ $0.50**, with `runpod/pytorch` (CUDA ≥ 12.8) image, 40 GB disk,
no network volume.
