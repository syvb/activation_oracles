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

## Phase 2 — Main run (completed)

Ran with the configuration above. Final stats:

- 13K optim steps × bs=16 = 210K examples × 1 epoch
- 11.7M training tokens (lower than the 65M target — Yes/No classification
  prompts are short, ~30 tokens. The originally-cited 65M figure included
  the 3-layer source activations × longer context-prompt tokens; for our
  single-layer + Yes/No-heavy mixture 11.7M is what 1 epoch of 210K mixed-
  task examples produces, and held-out losses had plateaued by step ~10K.)
- Held-out CE losses by task (initial → final):
  - LatentQA(stimulus): 4.38 → **1.38** (-3.00)
  - past_lens: 9.21 → **2.37** (-6.84)
  - cls/geometry_of_truth: 10.17 → **0.018**
  - cls/snli: 10.44 → **0.079**
  - cls/md_gender: 10.15 → **0.065**
  - cls/relations: 10.36 → **0.064**
  - cls/sst2: 9.30 → **0.054**
  - cls/tense: 10.11 → **0.018**
  - cls/ner: 11.05 → **0.107**
  - cls/language_identification: 9.43 → **0.069**
- W gradient norm: stayed in 0.05-0.4 range across all 13K steps. Projector
  is being updated, not stuck at identity.
- Final checkpoint pushed to `syvb/from-scratch-K8-AO-Qwen3-8B` (private).

Total H100 wall-clock: ~1h 20min training + ~25min dataset construction +
~5min HF push = ~1h 50min. At $2.99/h → ~$5.50 of compute.

## Phase 3 — Paper evals (completed)

Two complete eval suites: K=8 from-scratch (our trained model in single-
source-K=8 mode) vs K=1 cls-only baseline (`adamkarvonen/checkpoints_cls_only_addition_Qwen3-8B`
in single-source-K=1 mode — same eval-time pipeline, just K=1 with the
identity projector). Apples-to-apples: identical eval driver, identical
source-extraction logic, identical decode parameters; only the AO and
the K differ.

Eval driver code: `experiments/from_scratch_paper_evals.py` (classification)
and `experiments/from_scratch_open_ended_evals.py` (taboo + personaqa).
Scoring: `experiments/score_from_scratch.py`.

### Classification (20 datasets × 250 examples × 2 QAs = 10K eval points)

| Group           | K=1 baseline | K=8 from-scratch |       Δ |
| --------------- | -----------: | ---------------: | ------: |
| IID(7)          |        89.1% |            90.6% |   +1.5pp |
| OOD-3 (paper)   |        66.0% |        **88.7%** | **+22.7pp** |
| OOD-engels(10)  |        64.7% |            64.6% |   −0.1pp |
| OOD-all(13)     |        65.0% |            70.2% |   +5.2pp |

Single biggest finding of this experiment: **+22.7pp on the paper's OOD-3
slice** (ag_news, language_identification, singular_plural). Previous eval-
only experiment (RESULTS.md) found +4.6pp on OOD-3 with a frozen AO + W=I
init. From-scratch trained AO multiplies that by ~5x.

Per-dataset OOD-3 numbers:
| Dataset                   | K=1   | K=8   |
| ------------------------- | ----: | ----: |
| ag_news                   | 71.6% | 79.8% |
| language_identification   | 50.6% | 90.8% |
| singular_plural           | 75.8% | 95.6% |

The engels OOD slice (10 narrow binary classifiers) shows no improvement
(−0.1pp). This is consistent with the eval-only finding: the K-fold
redundancy effect generalizes well to "broader concept" OOD tasks but not
to the very narrow engels classifiers (which were already near chance).

### Taboo (20 secret words × 3 prompts × ~30 contexts × 5 generations)

| Metric | K=1 baseline | K=8 from-scratch |       Δ |
| ------ | -----------: | ---------------: | ------: |
| AVG    |         4.9% |             5.8% |   +0.9pp |

Per-target gains range from −3.3pp (salt) to +5.2pp (flame). 13 of 20
target words show positive Δ, 7 show negative. Within per-target sampling
noise (5 generations × ~30 contexts = 150 samples per target).

### PersonaQA (50 personas × 6 attributes × 5 generations)

| Metric  | K=1 baseline | K=8 from-scratch |       Δ |
| ------- | -----------: | ---------------: | ------: |
| OVERALL |         8.8% |             6.5% |   −2.3pp |

K=8 underperforms K=1 here. Per-question, two questions tie, four go
negative (-1pp to -7pp). 50 personas × 5 reps = 250 samples per question.

### 3-eval headline

| Eval                    | K=1 baseline | K=8 from-scratch |     Δ |
| ----------------------- | -----------: | ---------------: | ----: |
| Classification(20 avg)  |        73.4% |        **77.3%** | +3.9pp |
| Taboo(20 avg)           |         4.9% |             5.8% | +0.9pp |
| PersonaQA(overall)      |         8.8% |             6.5% | −2.3pp |
| **3-eval AVG**          |    **29.1%** |       **29.9%** | **+0.8pp** |

**+0.8pp average across the 3 evals — does not clear the +3pp plan bar.**

## Interpretation

The K=8 single-source-projection from-scratch AO clearly beats the K=1
cls-only baseline on classification (the structured Yes/No task it was
heavily trained on), particularly on OOD. The headline +22.7pp OOD-3 gain
is a genuinely large effect — the from-scratch AO has learned to use the
K-fold redundant projection for better OOD generalization on classification.

But the open-ended secret-keeping tasks (taboo, personaqa) show no gain.
This matches the conclusion in RESULTS.md, which found that the K-fold
redundancy effect helps OOD-3 classification specifically and is roughly
neutral elsewhere. From-scratch training extended the OOD-3 lift but did
not extend it to the open-ended evals.

A reasonable interpretation: classification is a "narrow" task — predict
Yes or No on a short prompt. Open-ended tasks like "what is the secret
word the model is keeping?" require the AO to reconstruct the target-LoRA's
specific persona/secret from one source position, and the K-fold
redundancy doesn't help with that — what helps there is having more
informative source positions (which segment-K=10 mode provides for the
released AO, but our K=8 single-source design explicitly does not).

A K=segment / K=window-of-tokens variant would likely do better on the
open-ended evals, but that's outside this experiment's "single-source-K-
projection" scope.

## Stopping criteria

PLAN_FROM_SCRATCH.md says:
> After 2 from-scratch runs, if the best K=4 or K=8 setting doesn't beat
> K=1 by ≥3pp averaged across the five paper-eval tasks, stop.

We have:
- 1 run (this one), at K=8.
- Average across 3 paper-eval tasks (the 3 with Qwen3-8B target LoRAs):
  +0.8pp.

Strictly the plan says 5 tasks, but only 3 are runnable on Qwen3-8B
(gender and SSC are gemma/llama-only). On the 3 we have, the bar is not
cleared by the strict interpretation. On classification alone, the bar
is comfortably cleared (+3.9pp).

Recommendation: declare this a partial success — substantial improvement
on classification, especially OOD; no meaningful improvement on the
secret-keeping evals. The plan's "stop after 2 runs" criterion is
satisfied to stop after 1, since:
- Classification, the most-trained task, shows a large positive effect.
- The two open-ended evals are flat-to-negative, suggesting the
  single-source-K format is fundamentally not a good fit for those tasks
  rather than a hyperparameter issue.

A meaningful follow-up would be to test a K=segment-window AO (the
existing K-window format) trained from scratch with a learnable W, vs
the released window-mode AO baseline, to see if W helps in the format
that's already strong on open-ended tasks. That's a different experiment.

## Trained checkpoint

- Repo: `syvb/from-scratch-K8-AO-Qwen3-8B` (HF Hub, private)
- Files: LoRA adapter (`adapter_model.safetensors`, 698 MB) + projector
  (`projector.pt`, 537 MB) + tokenizer + base config metadata
- Loadable via:
  ```python
  from peft import PeftModel
  from nl_probes.multi_token.projector import MultiTokenProjector
  m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
  m = PeftModel.from_pretrained(m, "syvb/from-scratch-K8-AO-Qwen3-8B")
  state = torch.load("projector.pt")
  proj = MultiTokenProjector(d_model=4096, k=8, init_strategy="all_identity")
  proj.load_state_dict(state["projector_state_dict"])
  ```
