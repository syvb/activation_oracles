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

## Phase 1 — Smoke test (K=8, ~5–10K examples)

(pending)

## Phase 2 — Main run (K=8, 30K+ examples)

(pending)

## Phase 3 — K sweep

(pending — conditional on Phase 2 success)

## Phase 4 — Ablations

(pending — conditional on Phase 2 success)

## Phase 5 — Generalization

(pending, optional)
