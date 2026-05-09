# Results — multi-token activation injection

This file accumulates experimental results as the staged plan in PLAN.md is
executed. Each phase appends a section.

## Phase 0 — K=1 baseline reproduction

Eval-only. `experiments/phase0_baseline.py` restricts
`classification_eval.py` to a single (model, LoRA, layer) combination.

- **Model:** `Qwen/Qwen3-8B`
- **AO LoRA:** `adamkarvonen/checkpoints_cls_only_addition_Qwen3-8B`
- **Reference:** same model with no LoRA (zero-shot)
- **Layer:** 50% (layer 18 of 36)
- **Mode:** single-token K=1, activation at end-offset −3
- **Eval:** 20 classification subdatasets × 750 datapoints (250 examples × 3 questions)

| Dataset                       | Cls-only LoRA | Base model |
| ----------------------------- | ------------: | ---------: |
| **IID — geometry_of_truth**   |         96.8% |      48.0% |
| **IID — relations**           |         91.6% |      48.0% |
| **IID — sst2**                |         88.7% |      49.7% |
| **IID — md_gender**           |         92.0% |      52.0% |
| **IID — snli**                |         82.8% |      52.8% |
| **IID — ner**                 |         82.9% |      49.9% |
| **IID — tense**               |         97.3% |      47.5% |
| **IID average**               |     **90.3%** |      49.7% |
| OOD — ag_news                 |         67.5% |      48.4% |
| OOD — language_identification |         57.3% |      51.7% |
| OOD — singular_plural         |         74.5% |      48.9% |
| OOD — engels_headline_istrump |         62.5% |      54.0% |
| OOD — engels_headline_isobama |         61.9% |      52.3% |
| OOD — engels_headline_ischina |         57.3% |      48.7% |
| OOD — engels_hist_fig_ismale  |         78.3% |      50.5% |
| OOD — engels_news_class_pol   |         57.7% |      53.2% |
| **OOD average**               |     **64.6%** |      51.0% |

The cls-only LoRA + injection at layer 50% reaches IID 90.3% / OOD 64.6% (single-token K=1) versus a base-model floor of essentially chance (~50%). The IID number is right on the line of what the paper reports for the same checkpoint at the 50% layer in single-token mode, so the plumbing is good.

OOD has clear headroom (~35pp away from the 100% ceiling), which is exactly where Phase 2's K=8 run is meant to be visible.

## Phase 1 — Smoke test (K=8, 16K examples, 1 epoch)

`experiments/phase1_smoke.py --k 8 --n-train-per-ds 1000 --n-test-per-ds 100 --batch-size 8 --epochs 1`

- 8 train datasets × 1000 contexts × 2 QAs = **16K train examples**
- IID test: same 8 datasets × 100 contexts × 2 QAs = 1600 test examples
- 200M trainable params (134M projector at d=4096, 67M residual MLP adapter)
- LR 3e-4 cosine, AdamW, bf16 forward + fp32 trainable modules
- 2000 optim steps at bs=8

| Step | Avg eval acc |
| ----:| -----------: |
|    0 |    **81.8%** |
|  200 |        57.2% |
|  400 |        66.2% |
|  600 |        68.9% |
|  800 |        67.6% |
| 1000 |        71.1% |
| 1200 |        73.0% |
| 1400 |        74.3% |
| 1600 |        74.3% |
| 1800 |        74.9% |
| 2000 |    **75.0%** |

**Final eval, per dataset (step 2000):**
- geometry_of_truth: 99.0% (was 95.5% at step 0 — **gained 3.5pp**)
- relations: 79.5% (was 71.0%)
- sst2: 84.0% (was 75.5%)
- md_gender: **51.5%** (was 94.5% — **lost 43pp**)
- snli: 82.0% (was 82.0%)
- ner: 82.5% (was 82.5%)
- tense: 74.5% (was 96.5% — **lost 22pp**)
- ag_news: 47.0% (was 57.0%)

**Sanity checks against PLAN.md:**
- ✓ Training loss decreases (from ~0.18 first half to ~0.17 second half).
- ✓ W gradient norm is non-zero (e.g., 0.72 at step 0, settling around 0.03-0.08).

**Observations:**
- Step-0 K=8 (W₁ = I, W₂..W₈ noise std 0.02) starts at **81.8%** — 8.5pp below the K=1 baseline (90.3%). The 7 noise placeholder slots perturb the AO meaningfully even before any training.
- Training slowly recovers (200 → 1800 steps: 57% → 75%) but plateaus well below the K=1 baseline.
- Per-dataset trajectories diverge: simple tasks (geometry_of_truth, sst2, snli) gain or hold; multi-class-flavored ones (md_gender, tense) collapse to near-chance.
- The drop pattern looks like overfitting under a too-large LR with a too-small dataset relative to the 200M trainable params.

**Plan for Phase 2:**
1. Lower LR (1e-4 vs 3e-4) to reduce the initial-step thrashing.
2. ~30K train examples (per the plan's main run sizing).
3. Same identity-plus-noise init.
4. Eval at step 0 + every 500 steps so we can read off the curve.

## Phase 2 — Main run (K=8, 30K+ examples)

(pending)

## Phase 3 — K sweep

(pending — conditional on Phase 2 success)

## Phase 4 — Ablations

(pending — conditional on Phase 2 success)

## Phase 5 — Generalization

(pending, optional)
