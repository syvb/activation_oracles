# Plan: From-scratch Activation Oracle with single-source-K-projection in the training mixture

## Why this experiment exists (briefing for a fresh agent)

This plan is the natural follow-up to the experiment in `PLAN.md` / `RESULTS.md`.
The original plan asked: does decomposing a single target activation into K
linearly-projected slots improve downstream AO performance, with the AO
adapted via lightweight fine-tuning? The answer there came out **mostly no**:
no setting of the W projector + frozen released-AO LoRA + small adapter
beats the K=1 baseline by the 3pp threshold the plan asked for. Trained-W
runs on classification overfit (AO is at 99% train accuracy already);
trained-W on LatentQA learns LatentQA but doesn't outperform the
all_identity step-0 model on classification.

The one real positive finding from those runs is that **K identical copies of
the activation injected at K placeholder slots, with no learning at all**,
gives a modest free OOD gain (+4–5pp on the paper's standard OOD-3 set,
+1.6pp on a broader 13-OOD set, no IID cost) on the released cls-only AO.
This isn't really "learned decomposition" — it's a fixed redundancy trick.
It's roughly +1pp short of the plan's bar across the broader eval surface.

The hypothesis this plan tests: **a fresh AO trained with single-source-K
-projection format in its data mixture can actually use the K decomposition
because it's been trained to**, unlike the released AO which has only seen
K=1 single-token and K=window-of-sequential-tokens formats. That changes the
experiment from "frozen AO + small adapter must accommodate a brand-new input
format" to "the AO learns the format from scratch alongside everything else
it learns."

## What is actually new vs the existing AO training pipeline

The existing pipeline (`nl_probes/sft.py`) trains a Qwen3-8B LoRA on a mixture
of:
- Classification (K=1 single-token AND K=window=50 multi-token)
- LatentQA (variable K from a window or all-positions)
- Past Lens (K=1 AND K=many)
- SAE explanations / activations / yes-no

Each training example provides K activation vectors (one per placeholder
slot), and the AO learns to interpret them. The K activations are *different*
— they come from sequential token positions of the target prompt.

This plan adds a NEW data type to the mixture: **single-source-K-projection**.
For each example, take ONE activation vector `a` from one token position of
the target prompt, project it to K vectors via a learnable W: R^d → R^(K×d),
and inject at K placeholder slots. The AO has to learn to use K
*linearly-projected views of one activation* — not seen in the existing
training mixture.

W and the AO LoRA train jointly. Implementation lives at
`nl_probes/multi_token/projector.py` (already written: `MultiTokenProjector`).

## Concrete architecture

```
sources = pre-extracted single source activations  (B, d_model)
projector = MultiTokenProjector(d_model, K, init_strategy="all_identity")
hook = get_multi_token_steering_hook(sources, projector, ...)
                                                                                
forward:
  embed input_ids
  layers 0
  layer 1:
    output residual stream (B, L, d)
    hook fires:
      proj = projector(sources)              # (B, K, d), fp32
      for each batch element:
        steered = norm-matched(proj[b], orig_residual_at_K_positions)
        residual[b, K_positions] += steered
  layers 2..N
  output logits

loss = CE on labels (with -100 for ignored tokens)
backward updates: W (in projector) + LoRA params, all in fp32 master, bf16 forward
```

Reuse `nl_probes/multi_token/hook.py` as-is; reuse the
`MultiTokenProjector(init_strategy="all_identity")` (or
`"identity_plus_noise"` if symmetry-breaking turns out to matter for joint
LoRA + W training, which it didn't in the previous experiment but the
ground rules change for from-scratch). Recommend starting with `all_identity`
because that was the operating point in the previous experiment.

Optional: also include a small **InjectionAdapter** (residual MLP) at the
placeholder positions after injection, identity-initialised. Trained jointly.
Previous experiment showed this hurts when the AO is held frozen, but it has
not been tested when the AO is trainable from scratch — could let the AO
absorb some of the "make the K projections useful" work into the placeholder
positions specifically rather than the whole AO.

## Data mixture recipe

For each existing task in the mixture (classification, LatentQA, past lens,
SAE), add a NEW entry that uses single-source-K-projection format. Concretely:

- For each task, build TrainingDataPoints where:
  - The activation context (target prompt) is the same as the existing
    multi-token variant.
  - The activation extraction takes ONE position (per the task's existing
    end-offset distribution), not K positions.
  - The AO prompt has K=K_target placeholders.
  - The acts_KD passed in is `source.unsqueeze(0).expand(K, -1)` — K
    identical copies of the source activation. The hook will project them
    via W, but the data structure must hold K rows for compat with
    `materialize_missing_steering_vectors` and assertions in
    `TrainingDataPoint`.
  - `data builder for classification: build_multi_token_classification_data`
    (already written in `nl_probes/multi_token/data_builder.py`)
  - `data builder for LatentQA: build_multi_token_latentqa_data` (already
    written in `nl_probes/multi_token/latentqa_data_builder.py`)
  - For past-lens and SAE you'll need to write equivalents — straightforward
    given the existing single- and multi-token loaders.

**Pick a single fixed K_target and train only for that.** K=4 was the
sweet spot in our eval-only experiments and is the recommended starting
point. Don't sweep K within a run and don't try to preserve K=1
single-token compatibility — the resulting AO is allowed to be K-specific.
Spending data budget on the existing K=1 / K=window formats just dilutes the
signal for the format we actually care about, and the existing K=1 AO
already exists if anyone needs that mode.

Mixture: 100% single-source-K-projection at K=K_target, drawn from all the
existing tasks (classification, LatentQA, past lens, SAE) using their
existing context-extraction logic but funneled through the new single-source
data builders. This means the AO sees the *same range of tasks and contexts*
as the original AO training, just always rendered as one source activation
projected to K placeholders.

If you want to verify the trained AO didn't catastrophically lose
generality, you can hold out a small fraction of training data and report
held-out loss per task; no need to test at K=1.

## Hyperparameters

Anchor to `nl_probes/configs/sft_config.py` defaults that are known to
produce a working AO. Specifically:

- model_name: `Qwen/Qwen3-8B`
- hook_onto_layer: 1
- **layer_percents: [50] — single layer only.** Don't train multi-layer
  here. The original AO uses [25, 50, 75] but that triples the data and
  introduces a second source of variability (which target-model layer is
  the activation drawn from) that's orthogonal to the single-source-K
  question being studied. Keep the experiment focused: every training
  example draws its source activation from layer 50% of Qwen3-8B, just
  like the eval-only experiments that produced the +OOD finding. Add other
  layers in a follow-up only if K=K_target at layer 50% beats baseline.
- LoRA rank 64, alpha 128, dropout 0.05, target=`all-linear`
- LR for LoRA: `1e-5`
- LR for W (projector): `1e-4` to `3e-4` — separate param group
- batch size: 16 global (per-rank batch / world_size); use DDP via torchrun
- gradient accumulation: 1
- steering_coefficient: 1.0
- epochs: 1
- max_grad_norm: 1.0
- bf16 forward, fp32 trainable params
- ~65M training tokens total (matches the paper's published cost)
- Window-mult length-bucketing on as currently configured (see
  `length_grouped_reorder` in `sft.py`)

For the new `single-source-K-projection` data, the prompt header is the
existing `get_introspection_prefix(layer, K)` ("Layer: {L}\n" + " ?" * K +
" \n"). Don't introduce a new prompt format — the existing introspection
prefix already supports any K.

## Evaluation

Use the paper's eval suite: `bash experiments/paper_evals.sh`:
- Classification (`experiments/classification_eval.py`)
- Gender open-ended (`experiments/gender_open_ended_eval.py`)
- Taboo open-ended (`experiments/taboo_open_ended_eval.py`)
- SSC open-ended (`experiments/ssc_open_ended_eval.py`)
- PersonaQA open-ended (`experiments/personaqa_open_ended_eval.py`)

For each, evaluate at K=K_target (the chosen K — this AO is K-specific).
**Compare against the released cls-only AO running at K=1** as the
baseline; that's the apples-to-apples comparison since the released AO is
the one this experiment is trying to beat.

Bar to clear: the new K=K_target AO beats the released-K=1 AO by ≥3pp on
average across the five tasks (matches the original plan's threshold).
Stretch goal: meaningful gains on the open-ended secret-keeping evals
(Taboo / Gender / SSC), since those are the hardest tasks where the
released AO has the most headroom and where the K-fold redundancy effect we
found earlier might shine.

Don't only measure averages; per-task variance is high. Report all five
tasks with confidence intervals; plot K vs accuracy per task.

## Practical advice from the prior experiment

1. **K=4 is the sweet spot in eval-only mode** (paper evals at the existing
   AO + frozen W=identity). K=16 starts losing IID accuracy. Don't go too
   wide just because K is cheap to bump.

2. **`all_identity` is the safest projector init.** `identity_plus_noise`
   (W₁=I, others ~ N(0, 0.02)) was the original plan's spec; in our experiments
   even tiny std=0.02 noise broke the AO at step 0 because the random
   projections emit unfamiliar residual directions. Start with all_identity
   and only revisit if symmetry-breaking turns out to matter for from-scratch
   training (it might — the AO learning curriculum can break the symmetry
   correctly via the data, unlike the frozen AO).

3. **Training loss curves can be misleading.** On classification, the
   released AO sits at ~0.05 cross-entropy per Yes/No → essentially 99% train
   accuracy. Any further "training" is just memorising. Track held-out loss
   on EACH dataset / task in the mixture, not just average loss.

4. **Eval volatility is high (5+pp between adjacent eval steps).** Use
   eval_every=200 with the same eval set throughout training. Don't compare a
   single eval point to claim "training improved things"; need at least 3
   consistent positive eval points or smoothed trajectory.

5. **Materialize-vs-cache trade-off.** `materialize_missing_steering_vectors`
   runs the base model with `disable_adapter()` to compute activations at
   training time. This forces a forward pass per batch, which is wasteful at
   scale. Pre-compute and cache activations once if the dataset fits on disk
   (it does for the existing classification + LatentQA mixtures). The
   `save_acts=True` flag on the existing dataset loaders does this.

6. **The classification task is an LIM (Low Information Margin) trap on
   IID — most of the IID datasets are at >90% AO accuracy already. Don't
   chase IID gains; they aren't there. Aim for OOD and the open-ended
   secret-keeping evals.

7. **K=1 functionality is explicitly *not* a goal here** — this AO is
   K-specific by design. Don't spend data budget keeping K=1 alive.
   The released cls-only AO continues to exist for K=1 use cases; the new
   AO trades K=1 support for stronger K=K_target performance.

8. **Track ||W_k - I||_F per slot during training.** If all K slots stay
   near identity, the AO is ignoring the K-decomposition and you're just
   relying on the redundancy effect. If they diverge fast, the optimizer is
   trying to make slots distinguishable — good, but verify that distinct
   slots actually improve accuracy, not just train loss.

9. **Don't use a separate "W warm-up" training stage.** Tried in the
   previous experiment in spirit (W-only training on classification then on
   LatentQA) — both had the same overfit-or-misalign-direction problem. Joint
   training of W + LoRA from scratch is the cleaner setup.

10. **Steering coefficient 1.0 doubles the residual norm at injection
    sites** because the existing hook does `injected = norm-matched + orig`,
    not `injected = norm-matched`. This is a property of the released AO's
    training; don't change it — would break compatibility with the released
    weights for K=1 mode.

## Compute budget

The paper's published cost: 10 H100h for the full Qwen3-8B AO training on
65M tokens at three layers, no W projector. With this plan:

- Single layer (layer 50% only) cuts the per-step token count by ~3× vs the
  paper's three-layer setup.
- Single K_target, no K=1 / K=window-multi support — no data multiplication
  for legacy formats.
- W is small (~134M params at K=8, ~67M at K=4 — comparable to a LoRA at
  rank 64 for Qwen3-8B). Forward + backward through W is negligible.
- Estimate: 5–8 H100h per from-scratch run.

Plan for at least 2 runs (likely 3 with debugging overhead): 15–25 H100h.

Use 1 H100 80GB Spot via RunPod (existing infrastructure); set up via the
flow recorded in `STATUS.md`. Project has no GCP quota for H100, so
RunPod is the path. Budget: ~$3/hr × 25h = ~$75.

## Risks and decision points

- **If the AO ignores the K-decomposition (W stays at identity, accuracy
  matches the K=1 mode): the linear-W formulation is fundamentally
  insufficient.** Either pivot to Q-Former (the plan's next-experiment
  hint) or accept the from-scratch result as a negative.
- **If OOD-only generalization improves (matching what the eval-only
  experiment found) but no IID gains:** that's a real result. The free
  +4-5pp OOD-3 from the previous experiment came from the redundancy
  effect; a from-scratch trained version should be able to keep that and
  add more.

## Stopping criteria

- After 2 from-scratch runs, if the best K=4 or K=8 setting doesn't beat
  K=1 by ≥3pp averaged across the five paper-eval tasks, stop. Document
  the negative result and pivot to the Q-Former architecture or a
  fundamentally different decomposition (e.g., perceiver-IO style
  cross-attention from a small set of learned queries to the activation,
  outputting K vectors).

## Where to start

1. Re-read `RESULTS.md` for context on what worked/didn't in the
   small-scale version.
2. Read `nl_probes/sft.py` end-to-end. Understand the dataset-loader
   abstraction (`ActDatasetLoader`, `DatasetLoaderConfig`,
   `BaseDatasetConfig`) and the train loop.
3. Read `nl_probes/multi_token/{projector,hook,data_builder,latentqa_data_builder,train}.py`.
4. Add `nl_probes/multi_token/sft_runner.py` (or extend `sft.py` with a
   `--enable-multi-token-projection` flag) that:
   - Builds the existing dataset mixture
   - Adds new dataset loaders for single-source-K-projection variants
   - Wires the `MultiTokenProjector` into the hook + optimizer
   - Trains W + LoRA jointly
5. Smoke test on a single H100 with 5K examples and 100 optim steps; verify
   train loss decreases and W gradient norms are non-zero (same Phase 1
   smoke checks as the previous plan).
6. Full run: 65M tokens × 2 (= 130M tokens), 1 epoch, on 1 H100. Should
   complete in 15–20h.
7. Run `experiments/paper_evals.sh` against the new checkpoint, modified to
   support K=4 and K=8 multi-token-projection inference. Compare against the
   K=1 baseline.

Good luck.
