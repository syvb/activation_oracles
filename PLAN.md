# Multi-Token Activation Injection in Activation Oracles

## Background

Activation Oracles (AOs), introduced by Karvonen et al. (2026, arXiv:2512.15674), are LLMs trained on the LatentQA task of accepting LLM activations as inputs and answering natural-language questions about them. The standard AO injection mechanism takes K activation vectors $\{v_i\}_{i=1}^K$ from layer $\ell$ of a target model, constructs an oracle prompt containing K placeholder tokens (literally `" ?"`), and at the AO's layer 1 additively injects each $v_i$ into the corresponding placeholder position with norm-matching:

$$h'_i = h_i + \|h_i\| \cdot v_i / \|v_i\|$$

In existing usage, K placeholder tokens correspond to K *distinct* activations — typically a sequence from the target model's forward pass. This experiment investigates whether decomposing a *single* target activation into multiple injected tokens via a learned linear projection improves downstream task performance.

The hypothesis, raised in [a comment by loops](https://www.lesswrong.com/posts/oeYesesaxjzMAktCM?commentId=umXyKqksZ6XRZuKPd) on the Natural Language Autoencoders post: when an entire activation occupies a single token's residual stream at the input, downstream attention over that token is forced to be all-or-nothing, and the surrounding token positions lose bandwidth for the model's own intermediate computation. Splitting the activation across K projected tokens should let attention selectively read different directions, and leave more residual-stream bandwidth elsewhere.

Because the AO has not been trained to interpret K linearly-related projections of one activation, frozen-AO experimentation is unlikely to show the effect — the AO's attention patterns over K placeholders were learned for sequences of distinct activations. The plan below adapts the AO via lightweight fine-tuning so that the new injection format becomes interpretable to it.

## Architecture

The new component is a linear projection $W: \mathbb{R}^d \to \mathbb{R}^{K \times d}$, where $d$ is the target model's residual stream dimension. Implementation is a single `nn.Linear(d, K*d)` followed by a reshape to `(K, d)`. For each target activation $a$, we compute $(W_1 a, \ldots, W_K a)$ and inject each $W_i a$ into the $i$-th placeholder slot via the AO's existing additive norm-matched injection.

Initialization: $W_1 = I$ (identity), and $W_2, \ldots, W_K$ initialized to small random values (scaled normal, std 0.02). This means at K=1 with $W=I$ the experiment exactly reproduces the baseline AO, and at K>1 the model starts from "primary slot is the original activation, additional slots are noise" — a sensible starting point the optimizer can refine.

Two adaptation variants for the AO itself are worth comparing. The first is an **injection-layer adapter**: a single learned linear or 2-layer MLP applied to placeholder-position residual streams immediately after the layer-1 injection. This isolates the architectural change to the input-processing step, which is precisely where the new behavior is needed. The second is **full LoRA** on the AO at rank 8 applied to attention and MLP projections, matching common practice and giving the AO maximum freedom to adapt.

Run the adapter variant first. If it works, the result is much more interpretable than full LoRA — it directly shows the AO needed only a small fix-up at the injection point to use multi-token decompositions, rather than rewiring the whole computation. The LoRA variant is the fallback.

## Setup

Target model is Qwen3-8B-IT, the smallest open model the AO paper supports. The AO itself is the released Qwen3-8B AO from `github.com/adamkarvonen/activation_oracles`. Training data is the binary classification dataset from the AO training mixture — single task for clean signal.

Activation pre-computation runs all training and eval target prompts through Qwen3-8B-IT once, caching activations from layer 50% (the paper's evaluation default). This cache is reused across all K values, training runs, and ablations.

Loss is cross-entropy on the AO's output answer tokens. Optimizer is AdamW with learning rate 3e-4, cosine decay, bf16 forward with fp32 master parameters. Training volume for the main run is 150K classification examples. Both $W$ (≤134M params at K=8, much less for smaller K) and the adapter are small relative to the AO, so this is generous; the smoke test below may justify reducing it.

## Staged Plan

**Phase 0 — Plumbing.** Reproduce the K=1 baseline using the released AO without any modification. Verify the eval accuracy matches the paper's reported number to within noise. If it doesn't, the prompt format, placeholder token, layer index, or norm-matching is wrong and must be fixed before continuing.

**Phase 1 — Smoke test.** Train $W$ + injection-layer adapter at K=8 on 5–10K examples for roughly 10 minutes. Verify training loss decreases and the gradient norm of $W$ is non-zero. Cheap to run, cheap to debug; catches plumbing errors that survived Phase 0.

**Phase 2 — Main run.** Train $W$ + adapter at K=8 on 150K examples. Evaluate on the held-out classification split. This is the main hypothesis test. If K=8 beats the K=1 baseline by at least 3 percentage points (well outside the ±1.5pp 95% CI from a 2K eval set), proceed. If the result is marginal or absent, run the same setup with full LoRA instead of the adapter — if that also fails, the hypothesis is not supported by this experimental design and the Q-Former variant becomes the natural next experiment. If the result is actively worse than baseline, examine training loss curves and projection outputs to diagnose; the most likely failure mode is $W$ collapsing to put all signal in slot 1, leaving the other K-1 slots emitting noise that the AO can't quite ignore.

**Phase 3 — K sweep.** Conditional on Phase 2 being positive, run K ∈ {2, 4, 16} with the same setup as the K=8 winner. Plot accuracy as a function of K. The expected shape under the hypothesis is monotonic improvement that plateaus or saturates somewhere in the 4–16 range.

**Phase 4 — Ablations.** Conditional on Phase 2 being positive. Two ablations clarify what's actually doing the work. The first is **repeat-K**: set $v_i = a$ for all $i$, retrain the adapter, and measure. This isolates "multi-token capacity helps" from "the projection's diversity matters." If repeat-K matches projection-K, $W$ is doing nothing useful and you're just buying capacity from extra slots. The second is **random-W**: a fixed orthogonal projection that's never trained. If random-W matches trained-W, then any K-way decomposition into different directions does the job and the learned projection adds nothing.

**Phase 5 — Generalization (optional).** Evaluate the trained K=8 system on Taboo secret-keeping and PersonaQA without further training, to test whether the multi-token decomposition transfers beyond classification.

## Sanity Checks

Phase 0's K=1 baseline must match the paper. After Phase 1, training loss must strictly decrease over the first 100 steps and the gradient norm of $W$ must be non-zero — easy mistakes here include accidentally detaching the graph between $W$ and the AO, or initializing $W$ with too-large random values that overwhelm the identity component of $W_1$.

Throughout training, log $\|W_i a\|$ for each slot. Norm-matching makes magnitude irrelevant to the loss, so the optimizer will push these in arbitrary directions — if magnitudes go to zero or explode by orders of magnitude, L2-normalize $W_i a$ explicitly before injection rather than relying on the AO's norm-matching to clean up.

After Phase 2, examine attention patterns from later AO tokens to the K placeholder slots. If attention is uniform across slots, the AO isn't actually using the decomposition. If attention is concentrated on a single slot, the projection isn't differentiating useful directions and the result, if positive, is probably an artifact of extra capacity rather than multi-slot reading.

## Compute Budget

Anchored to the paper's published cost: 10 H100 hours for full Qwen3-8B AO training on 65M tokens. The experiment does strictly less work than that — adaptation rather than from-scratch training, single task rather than the diversified mixture, smaller learnable parameter count.

Activation pre-computation is roughly 0.5 H100h (one-time, reusable). Phase 0 is eval-only, around 0.1h. Phase 1 smoke test is around 0.2h. Phase 2 main run, training on 150K examples of ~50 tokens each, is around 3h. Phase 3 sweep of three additional K values runs around 9h total. Phase 4 ablations require training and eval, around 6h total. Phase 5 generalization is eval-only, around 0.5h. With a 1.5x slop multiplier for debugging, restarts, and the inevitable issue caught on first run:

The minimum-viable experiment (Phases 0–2) costs **about 5 H100 hours**. The full main result with sweep (Phases 0–3) costs **about 15 H100 hours**. The complete plan including ablations and generalization costs **about 25 H100 hours**. Each LoRA fallback run, if needed, adds 3–5 H100h.

## Risks

The most likely failure mode is that linear decomposition is genuinely insufficient — the relevant structure in activations is nonlinear, and a Q-Former-style decomposer with self-attention among slots and FFN nonlinearity is what's actually needed. If the Phase 2 adapter and the LoRA fallback both fail, Q-Former is the natural next experiment rather than further iteration on linear $W$.

A second risk is that multi-token injection helps in principle but the AO can't leverage it without far more retraining than fine-tuning provides. This would manifest as both adapter and LoRA variants helping a little but not crossing the threshold. The diagnosis would be to train a fresh AO from scratch with multi-token decomposition built into the data mixture, which is much more expensive — closer to 10 H100h per run scaled by the sweep size.

A third risk is task specificity. Binary classification may be close enough to ceiling that there's no headroom for the projection to matter. If the Phase 0 baseline is above 90% accuracy, switch the primary task to PersonaQA or Taboo, where published baselines are lower and the gap available to improvements is larger.
