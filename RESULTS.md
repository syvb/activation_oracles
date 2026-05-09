# Results — multi-token activation injection

This file accumulates experimental results as the staged plan in PLAN.md is
executed. Each phase appends a section.

## Phase 0 — K=1 baseline reproduction

Eval-only. Run `experiments/phase0_baseline.py` which restricts
`classification_eval.py` to a single (model, LoRA, layer) combination.

- **Model:** `Qwen/Qwen3-8B`
- **AO LoRA:** `adamkarvonen/checkpoints_cls_only_addition_Qwen3-8B`
- **Reference (no LoRA):** zero-shot baseline
- **Layer:** 50% (layer 18 for Qwen3-8B's 36 layers)
- **Mode:** single-token K=1 (window size 1, end offset -3)
- **Datasets:** 20 binary classification subdatasets, 250 examples each.

(results here once run completes)

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
