# Results — from-scratch K=8 single-source-K-projection AO

This file accumulates results for `PLAN_FROM_SCRATCH_AO.md`. Each phase
appends a section. The previous round's findings are in `RESULTS.md`.

## Setup decisions (locked 2026-05-09)

- **K_target = 8** (per user direction; PLAN_FROM_SCRATCH recommended K=4 as the
  sweet spot from RESULTS.md eval-only data, but the user wants K=8 to match
  the original PLAN.md target.)
- **Task mix:** classification + LatentQA + multi-token past-lens. SAE dropped
  to keep the new data-builder surface small (SAE adds dependence on the SAE
  encoder/decoder loading + max-acts data, marginal gain on the open-ended
  evals which is where we're chasing the headline number).
- **InjectionAdapter:** off — keep the architecture as close to vanilla AO as
  possible. Our only delta from a vanilla AO is the W projector.
- **HF Hub:** push final LoRA + projector to a private repo on the user's
  account.
- **Layer:** 50% only, single layer (no [25, 50, 75] sweep).
- **Init:** `all_identity` (W_k = I for all K slots).

## Code changes vs main

- New `nl_probes/multi_token/past_lens_data_builder.py` — single-source past-
  lens dataset.
- New `nl_probes/multi_token/loaders.py` — `MultiToken{Classification,LatentQA}DatasetLoader`
  that fit the `ActDatasetLoader` infrastructure (DDP-aware caching, hash-based
  filenames, train/test splits).
- New `nl_probes/multi_token/sft_runner.py` — `train_model_multi_token` that
  jointly trains LoRA + W with two AdamW param groups, uses the multi-token
  hook, and saves both LoRA and projector at checkpoint time. Held-out loss
  per task is tracked instead of generation-based eval during training.
- New `experiments/from_scratch_ao_train.py` — top-level launch script.
- Minor edit to `nl_probes/multi_token/data_builder.py` to support lazy mode
  without a base model loaded.

## Eval scope clarification (2026-05-09)

`paper_evals.sh` has 5 evals, but only 3 have Qwen3-8B target LoRAs that exist:
- **Classification** — any model ✓
- **Taboo** — Qwen3-8B branch uses `adamkarvonen/Qwen3-8B-taboo-{word}_50_mix` ✓
- **PersonaQA** — Qwen3-8B branch uses `adamkarvonen/Qwen3-8B-personaqa_shuffled_3_epochs` ✓
- **Gender** — `bcywinski/gemma-2-9b-it-user-{male,female}` only (Gemma) ✗
- **SSC** — `bcywinski/llama-3.3-70B-Instruct-ssc-base64` only (Llama) ✗

So the "+3pp avg across the 5 tasks" bar from PLAN_FROM_SCRATCH simplifies to
"+3pp avg across the 3 Qwen3-comparable tasks": classification, taboo,
personaqa. Compared to the released `adamkarvonen/checkpoints_cls_only_addition_Qwen3-8B`
running at K=1 on the same 3 evals.

## Phase 0 — Local CPU smoke (no GPU)

`PLAN_FROM_SCRATCH_AO.md` step 5 says:

> Smoke test on a single H100 with 5K examples and 100 optim steps; verify
> train loss decreases and W gradient norms are non-zero (same Phase 1 smoke
> checks as the previous plan).

Before paying for an H100 we did a CPU pre-flight:

- ✓ MultiTokenProjector init: `all_identity` correctly puts every slot at the
  identity, output of W·x is exactly K copies of x.
- ✓ MultiTokenProjector init: `identity_plus_noise` with std=0 puts slot 0 at I
  and slots 1..K-1 at zero (matches RESULTS.md reproducer).
- ✓ Backward through MultiTokenProjector populates `weight.grad` (norm ~18 on a
  random forward, finite and non-zero).
- ✓ `get_multi_token_steering_hook` produces `normalize(W_k·source) * ||orig_k||
  + orig_k` at each placeholder position, within bf16 quantization noise of the
  expected fp32 result (rel error 0.0014 on a 8-d toy).
- ✓ `build_multi_token_classification_data` runs in lazy mode (no model
  loaded), producing TrainingDataPoints with K=8 placeholder positions and K
  identical context_positions for the source.
- ✓ DatasetLoaderConfig hashes cleanly distinguish K=4 vs K=8 cache files —
  no collision between K-specific runs.

Pending on H100:
- Phase 1 smoke: 5K examples, 100 optim steps. Confirm train loss decreases
  and W grad norms are non-zero on real Qwen3-8B.
- Phase 2 main: ~65M tokens, full mixture, 1 epoch.
- Phase 3 eval: paper_evals.sh patched for K=8 single-source inference.

## Phase 1 — H100 smoke test (passed)

`torchrun --nproc_per_node=1 experiments/from_scratch_ao_train.py --debug --run-name phase1_smoke --train-batch-size 8`

Debug-sized: 200 cls/ds × 8 datasets + 1000 latentqa + 1000 past-lens =
~4.7K examples, 584 optim steps, 256K training tokens. Took ~3 minutes
on a single H100 SXM 80GB at $2.99/hr.

| Step | LatentQA | PastLens | cls (avg) |
| ---: | -------: | -------: | --------: |
|    0 |     4.35 |     9.66 |    ~10.05 |
|  150 |     1.88 |     4.23 |     ~0.20 |
|  350 |     1.78 |     4.05 |     ~0.20 |
|  584 | **1.74** | **3.93** |  **~0.18** |

Held-out CE drops on every task — training is working end-to-end. Train
log shows `w_grad_norm` consistently in 0.1-0.4 range across all 73
optim-step log points, so the projector is being updated (not stuck at
identity). 584 steps is far too few to evaluate accuracy gains, but
the smoke confirms:
- Data builders produce valid TrainingDataPoints with K=8 placeholders
- The DDP+PEFT+projector wrapper trains cleanly (after disabling
  gradient_checkpointing — the ckpt+hook combo trips a known torch DDP bug)
- HF token, lmsys access, and dataset caches all work
- `find_unused_parameters=True` is needed because the projector is
  reached via a forward hook, not via the wrapped module's `forward()`

Issues fixed during smoke:
1. `lmsys/lmsys-chat-1m` is gated → user granted access; no code change
   needed.
2. DDP "Expected to have finished reduction" error: preflight was running
   forward through the DDP wrapper instead of the inner model.
3. DDP "gradient which is undefined, but still allreduced" error:
   gradient_checkpointing + the in-place hook modification confused DDP's
   used-parameter tracking. Disabled grad ckpt.

## Phase 2 — Main run (in progress)

`torchrun --nproc_per_node=1 experiments/from_scratch_ao_train.py
--run-name main_K8_full --train-batch-size 16 --push-to-hub
--hf-repo-id syvb/from-scratch-K8-AO-Qwen3-8B`

Sized to roughly match the published 65M-token AO budget at K=8 single-
source-projection format:
- 6000 train examples × 8 classification datasets = 48K cls (×2 QAs/sample = 96K rows)
- 80K LatentQA examples
- 50K past-lens examples
Total: ~178K examples ≈ ~64M training tokens after the length-percentile trim.

Expected runtime: ~30 min dataset construction (past-lens streams from
fineweb + lmsys) + ~1.5 hr training + ~5 min HF push.

Bar to clear: average +3pp on classification + taboo + personaqa vs the
released `adamkarvonen/checkpoints_cls_only_addition_Qwen3-8B` at K=1.

## Phase 3 — Paper evals (pending)
