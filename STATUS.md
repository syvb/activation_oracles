# Status

## 2026-05-09 — Setup begun, paused on H100 quota

Started executing PLAN.md. Reviewed codebase, settled scope with user:
- All ~10 binary classification subdatasets (the full classification mixture used in `paper_evals.sh`)
- Baseline / starting LoRA: `adamkarvonen/checkpoints_cls_only_addition_Qwen3-8B`
- Layer 50% only, single-token (K=1) baseline regime
- Phase 2 main-run size: start at 30K examples and scale up if loss is still decreasing

Tried to provision an H100 spot in us-central1-a but the project has 0 quota for the `NVIDIA_H100` GPU family (the only modern GPU with quota is L4, 1 globally). Filed an automated quota-increase request:

- **Quota:** `GPUS-PER-GPU-FAMILY-per-project-region`, dimension `gpu_family=NVIDIA_H100`, region `us-central1`
- **Preferred value:** 1
- **Trace ID:** `aa928559-2f48-48fb-b0fa-7757d99c89c1`
- **Submitted:** 2026-05-09 00:31:43 UTC
- **Status at submission:** `reconciling=true`, `grantedValue=0`

Per user instruction, paused all further work until the quota request is decided.

## Final outcome (2026-05-09 evening)

Switched to RunPod (H100 80GB Spot, ~$2.99/hr); ran Phase 0 → Phase 2 main → Phase 2 LoRA fallback → K=1 LoRA control → step-0 K-sweep with several init strategies → broader 20-dataset eval → W-only training with all_identity init. See `RESULTS.md` for the full per-phase numbers.

Headline finding (different from the plan's hypothesis):
- The plan asked whether a *learned* linear projection W: R^d → R^(K×d) would let multi-token decomposition help. **It does not** — every variant of training (small adapter, LoRA fallback, W-only at low LR) either matches K=1 or hurts OOD generalization.
- However, with **all_identity init and no training** (W_k = I for every slot at K=4 or K=8, frozen AO), there is a modest free OOD gain on the paper's classification eval: +3.8–4.6 pp on the standard OOD-3 set, +1.6 pp on the broader OOD-13. IID is unchanged.
- The optimum is brittle: adding std-0.02 noise to the identity init or training W with LR 3e-5 both destroy the gain in 500 steps.
- Mechanistically this looks like K-fold redundancy at the AO input acting as a soft OOD regularizer, not the plan's hypothesized "selective attention to distinct projections."

GPU teardown / final commit pending user direction.

## Next steps once quota is granted (no longer needed — used RunPod instead)
1. Provision `a3-highgpu-1g` spot in `us-central1-a` (image family `pytorch-2-9-cu129-ubuntu-2204-nvidia-580`, 200 GB pd-balanced boot disk).
2. SSH in, clone the repo, `uv sync`, `huggingface-cli login`.
3. Phase 0: Run `experiments/classification_eval.py` restricted to Qwen/Qwen3-8B + `checkpoints_cls_only_addition_Qwen3-8B` at layer 50% in single-token mode. Confirm accuracy is in the ballpark of the paper.
4. Implement W projection (`nn.Linear(d, K*d)` with W₁=I init) and injection-layer adapter (residual MLP with zero last-layer init), wire them into the AO's layer-1 hook so gradients flow back to W.
5. Phase 1 smoke (5–10K examples, K=8), Phase 2 main (30K → 150K if signal warrants), Phase 3 K-sweep, Phase 4 ablations, Phase 5 generalization.

## Useful checks while paused
- `gcloud alpha quotas info describe GPUS-PER-GPU-FAMILY-per-project-region --service=compute.googleapis.com --project=octherdral --format=json` to read current granted value.
- The preference resource at `projects/octherdral/locations/global/quotaPreferences/f9f91b2e-c8f9-48c0-ba9a-c740f7897226` will show approval status.
