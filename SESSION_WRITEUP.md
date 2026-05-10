# Multi-token Activation Oracle — full session writeup

This is a synthesis of 8 experimental phases run in one session, all training
or analyzing a from-scratch K-specific Activation Oracle (AO) on Qwen3-8B in
the *single-source-K-projection* format. Per-phase detail is in
`RESULTS_FROM_SCRATCH.md`.

## What we set out to do

`PLAN_FROM_SCRATCH_AO.md` asked: can a from-scratch AO trained with
*single-source-K-projection* format in its data mixture beat the released
K=1 AO by ≥3pp on the paper-eval suite?

The format: every example provides ONE source residual-stream activation `a`
from one token of the target prompt. The AO sees this source projected to K
distinct vectors via a learnable linear map `W: R^d → R^(K·d)`, injected at K
placeholder positions in the AO's prompt. K=8 was the user-specified target.

## Headline result

**The +3pp bar was not cleared.** All four K=8 variants we trained land in a
0.6pp band on the 3-eval average (cls + taboo + personaqa). The reason is a
structural information bottleneck in the format itself, not a training
problem. Frozen-W (W locked at identity, only LoRA trains) actually wins the
average by a small margin over the variant with W training.

| Eval                  | K=1 cls-only | trained-W (Phase 3) | frozen-W (Phase 4) | entropy-pen (Phase 6) | slot-dropout (Phase 7) | K=4 tuned (Phase 8) |
| --------------------- | -----------: | ------------------: | -----------------: | --------------------: | ---------------------: | ------------------: |
| Cls IID(7)            |        89.1% |               90.6% |              89.1% |                 89.5% |                  90.9% |               90.4% |
| Cls OOD-3 (paper)     |        66.0% |           **88.7%** |              83.1% |                 85.9% |                  86.9% |               78.8% |
| Cls OOD-all(13)       |        65.0% |               70.2% |              69.8% |                 70.6% |                  70.4% |               65.9% |
| Taboo (20-avg)        |         4.9% |                5.8% |           **7.2%** |                  6.8% |                   5.8% |                7.1% |
| PersonaQA             |         8.8% |                6.5% |           **7.5%** |                  6.7% |                   6.0% |                5.5% |
| **3-eval AVG**        |        29.1% |               29.9% |          **30.4%** |                 30.3% |                  29.8% |               29.0% |
| **Δ vs K=1 baseline** |          0pp |              +0.8pp |          **+1.3pp** |                +1.2pp |                 +0.7pp |              −0.0pp |

## What works

1. **K-fold input redundancy is a real, modest free OOD lift.** Frozen-W gets
   +17.1pp on classification OOD-3 vs the K=1 baseline by *literally
   replicating the same activation vector at K=8 placeholder positions and
   doing nothing else*. This matches the eval-only finding in `RESULTS.md`
   (where the same effect was +4.6pp at smaller scale). From-scratch
   training scales it up because the LoRA learns to consume the K-fold
   redundant input.

2. **Joint LoRA + W training works end-to-end.** The data builders, DDP+hook
   wiring, optimizer with two parameter groups (LoRA at 1e-5, W at 3e-4),
   bf16 forward + fp32 trainable params, gradient flow through the
   in-place hook injection, HF push of LoRA + projector — all worked
   reliably after the early DDP bugs were fixed.

3. **Training W lowers held-out CE on every task** by 0.03–0.10 nats vs
   frozen-W (Phase 4 numbers). So W is doing real fitting work — the
   question is whether that fitting work translates to held-out accuracy
   on the eval surface, which it doesn't.

4. **Classification OOD-3 is a strong signal.** Every K=8 variant beats
   the K=1 baseline by ≥17pp on this slice. The released cls-only AO
   appears to have unused OOD-classification headroom that K-fold
   redundancy in the input unlocks.

## What doesn't work

1. **Training W to make slots distinct hurts open-ended performance.** The
   trained-W AO has lower attention entropy (mean 1.24 nats vs uniform 2.08)
   and concentrates on slots 0, 1, 5 — but its 3-eval average is 0.5pp
   *below* the frozen-W AO. The W-learned decomposition is a worse
   information layout for taboo and personaqa than just having all K slots
   carry the same source.

2. **Forcing the AO to use all K slots doesn't help accuracy.** Two
   different mechanisms — an entropy bonus on attention (Phase 6) and
   per-slot dropout during training (Phase 7) — both reshaped attention
   into a more uniform distribution but produced 3-eval averages within
   ±0.5pp of the unregularized variants. The K decomposition isn't carrying
   enough independent information for "use all 8" to outperform "use 3".

3. **Halving K and steering coefficient together (Phase 8)** hurt by
   ~1.4pp vs frozen-W. K=4, steer=0.5 reduces total injected signal
   magnitude to ~25% of K=8/steer=1.0 baseline; PersonaQA — which
   needs maximum info per source — was the worst hit.

4. **The +3pp plan bar.** No variant cleared it on the 3-eval average.
   Best was frozen-W at +1.3pp. Classification alone clears it (+3.1pp at
   frozen-W, +3.9pp at trained-W) but open-ended evals don't.

## Why this happens — the mechanistic story

The single-source-K-projection format has a hard information bottleneck.
For a single source `a` ∈ R^d:
- The K projected vectors `W_k · a` for k=1..K live in a d-dim manifold
  of R^(K·d). They cannot carry more information than `a` itself does.
- Any task-relevant information in `a` can be extracted from any single
  slot (since `W_k` is a free d→d linear map). The model has no incentive
  to use more than 1 slot to reconstruct the answer; the ~3-of-8 attention
  pattern is the AO settling into the easiest such reading.

Phase 5's attention analysis confirmed this directly:

| Variant            | Mean attention entropy | Gap to uniform (ln 8 = 2.08) |
| ------------------ | ---------------------: | ---------------------------: |
| trained-W          |                  1.236 |                         0.84 |
| frozen-W           |                  1.812 |                         0.27 |
| entropy-penalty    |                  2.005 |                         0.07 |
| slot-dropout       |                  1.676 |                         0.40 |

The variants form a continuum from "use 3 of 8" → "use ~5 of 8" → "use all
8 uniformly", but all four land in the same 0.6pp eval-accuracy band. The
shape of attention is downstream of the information layout, not upstream
of accuracy. Forcing uniform attention via regularization reshapes the
former without affecting the latter, because there's no extra information
to be uncovered.

## What I'd try next

In rough order of expected information content per dollar:

1. **Multi-source K** (highest EV). Take K different sources per example —
   different layers and/or positions of the target prompt. Each slot now
   carries genuinely independent information by construction. The
   "use all K" vs "use 3" question becomes well-posed because there's
   actually different content to attend to per slot. This is what the
   released "K=window-of-tokens" AO already does, just not in the
   single-source projection variant. Concretely: K=8 sources from layers
   `[10%, 25%, 40%, 50%, 60%, 75%, 90%, 100%]` of Qwen3-8B.

2. **K=4 with steering coefficient 2.0.** The Phase 8 K=4 run halved
   total injected signal magnitude. The right ablation is K=4 with
   steer=2.0 (preserves total magnitude at 2 × 4 = 8 = 1 × 8 of the
   K=8 baseline). Quick test: 1 H100h ≈ $3.

3. **Q-Former / Perceiver-IO architecture.** Replace the linear `W` with K
   learnable query vectors that cross-attend to the full target prompt's
   residual stream at multiple layers. Each query head learns to extract
   a different aspect of the activation context. This is what
   `PLAN_FROM_SCRATCH_AO.md` flagged as "the next-experiment architecture
   if linear-W is insufficient." This experiment confirmed linear-W is
   insufficient.

4. **K=1 LoRA-only fine-tune of the released AO.** A useful additional
   baseline: take the released cls-only K=1 AO and continue training the
   LoRA on the same single-source-K=1 data mixture for 13K steps. If this
   beats the K=1 zero-shot baseline by similar margins to our K=8
   variants, it would suggest **the gains we measured are mostly from
   continued LoRA fine-tuning, not from the K decomposition**. This is the
   control we never ran. Cheap (1 H100h).

5. **Drop the "single layer" constraint.** The plan picked layer 50%
   only. The released AO uses [25, 50, 75]. Multi-layer source per K
   slot would partially address the information bottleneck (each layer
   carries different processing depth). Compatible with multi-source K.

## Cost summary

| Phase                     | Compute | Spend |
| ------------------------- | ------- | ----: |
| Phase 1 smoke + Phase 2 main + Phase 3 evals | H100, ~3h | ~$9 |
| Phase 4 frozen-W training + evals | H100, ~2h | ~$6 |
| Phase 5 attention analysis | RTX A6000, ~10min | ~$0.1 |
| Phase 6 entropy-penalty (eager attn, slow) | H100, ~3h | ~$9 |
| Phase 7 slot-dropout train + evals | H100, ~2h | ~$6 |
| Phase 8 K=4-tuned (incl. abandoned 3-epoch start) | H100, ~2h | ~$6 |
| **Total** | | **~$36** |

All pods terminated immediately after each experiment, verified 404 on the
RunPod API. Wandb logging for Phase 8 onward (last run at
https://wandb.ai/octahedral-systems/sae_introspection/runs/wb97iyv0).

## Trained checkpoints (private HF Hub)

| Run                | Repo                                                          |
| ------------------ | ------------------------------------------------------------- |
| trained-W K=8 (Phase 3)  | `syvb/from-scratch-K8-AO-Qwen3-8B`                       |
| frozen-W K=8 (Phase 4)   | `syvb/from-scratch-K8-frozen-W-AO-Qwen3-8B`              |
| entropy-penalty K=8 (Phase 6) | `syvb/from-scratch-K8-entropy-penalty-AO-Qwen3-8B` |
| slot-dropout K=8 (Phase 7) | `syvb/from-scratch-K8-slot-dropout-AO-Qwen3-8B`        |
| tuned K=4 (Phase 8)      | `syvb/from-scratch-K4-tuned-AO-Qwen3-8B`                 |

All include LoRA adapter (`adapter_model.safetensors`) + matching projector
weights (`projector.pt`). Eval-time inference code:
`experiments/from_scratch_paper_evals.py` (classification) and
`experiments/from_scratch_open_ended_evals.py` (taboo + personaqa).
Attention analysis: `experiments/attention_analysis.py`.

## Code changes vs main (branch `multi-token-injection`)

- `nl_probes/multi_token/past_lens_data_builder.py` — new single-source past-lens
- `nl_probes/multi_token/loaders.py` — single-source-K loaders for cls + LatentQA
- `nl_probes/multi_token/sft_runner.py` — joint LoRA + W training loop, slot-dropout, entropy penalty
- `nl_probes/multi_token/hook.py` — slot-dropout-aware multi-token injection hook
- `nl_probes/sft.py` — `push_lora_to_hf` accepts custom README (the SAE-introspection wording was hardcoded; fixed for all future runs)
- `experiments/from_scratch_ao_train.py` — top-level training launcher
- `experiments/from_scratch_paper_evals.py` — classification eval driver
- `experiments/from_scratch_open_ended_evals.py` — taboo + personaqa eval driver
- `experiments/score_from_scratch.py` — unified scoring across all variants
- `experiments/attention_analysis.py` — output_attentions-based per-K attention measurement
- `experiments/plot_attention_analysis.py` — per-layer entropy + per-K bars

## Tl;dr

Single-source-K-projection works as a from-scratch AO setup: it trains
cleanly, push to HF Hub, evals run end-to-end. It produces a real but
modest (+1.3pp) average improvement over K=1 cls-only baseline, driven
almost entirely by classification OOD-3. **The format itself caps how much
the K decomposition can buy you** — every regularizer we tried (entropy
penalty, slot dropout, frozen W) lands within 0.6pp of trained-W on the
3-eval average. To clear the +3pp bar, the experiment needs to switch
formats: multi-source K or Q-Former cross-attention, addressing the
information bottleneck that single-source-K cannot.
