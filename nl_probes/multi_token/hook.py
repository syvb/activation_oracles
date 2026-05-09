"""Layer-1 forward hook that:
  1. Reads K placeholder positions for each batch element.
  2. Projects a single source activation `a_b` into K vectors via W.
  3. Injects each W_k a_b into the k-th placeholder slot using the AO's
     additive norm-matching: h'_k = h_k + ||h_k|| * normalize(W_k a_b).
  4. Optionally applies an adapter (residual MLP) to the post-injection
     residual at the same K positions.

Unlike the existing `get_hf_activation_steering_hook`, this hook keeps gradients
flowing through W (and the adapter) so they can be trained.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_multi_token_steering_hook(
    source_activations: list[torch.Tensor],  # len B, each shape (d_model,)
    projector: nn.Module,                    # MultiTokenProjector
    adapter: nn.Module | None,               # InjectionAdapter | None
    positions: list[list[int]],              # len B, each list has K positions
    steering_coefficient: float,
    device: torch.device,
    slot_dropout_p: float = 0.0,             # train-time only: prob of dropping a slot
) -> Callable:
    """If `slot_dropout_p > 0`, on each forward call we sample a Bernoulli
    mask per (batch element, slot). Masked slots skip injection — the
    residual at those placeholder positions keeps its natural pre-injection
    content. We always force at least 1 slot per row to remain. Use 0.0
    for eval.
    """
    assert len(source_activations) == len(positions)
    B = len(source_activations)
    if B == 0:
        raise ValueError("Empty batch")

    # Stack source activations to a (B, d_model) tensor on the right device.
    # The source activations come from a frozen target-model forward pass and
    # do not require gradients themselves. We force fp32 here because the
    # projector and adapter are kept in fp32 for stable AdamW updates.
    source_BD = torch.stack(
        [a.to(device=device, dtype=torch.float32).detach() for a in source_activations],
        dim=0,
    )

    def hook_fn(module, _input, output):
        if isinstance(output, tuple):
            resid_BLD, *rest = output
            output_is_tuple = True
        else:
            resid_BLD = output
            output_is_tuple = False

        B_actual, L, d_model = resid_BLD.shape
        assert B_actual == B, f"Batch mismatch: got B={B_actual}, expected {B}"

        # Skip the per-token decoding pass during generation (L == 1).
        if L <= 1:
            return (resid_BLD, *rest) if output_is_tuple else resid_BLD

        target_dtype = resid_BLD.dtype  # typically bf16

        # Project in fp32 to match the trainable modules' dtype.
        proj_BKD = projector(source_BD)  # (B, K, d_model), fp32

        # Sample per-(batch, slot) keep mask once per call.
        K = proj_BKD.shape[1]
        if slot_dropout_p > 0.0:
            keep = torch.bernoulli(
                torch.full((B, K), 1.0 - slot_dropout_p, device=device)
            ).bool()
            # Force at least one slot per row to remain (avoid all-dropped rows).
            no_keep = ~keep.any(dim=1)
            if no_keep.any():
                rand_slot = torch.randint(0, K, (B,), device=device)
                for b in range(B):
                    if no_keep[b]:
                        keep[b, int(rand_slot[b].item())] = True
        else:
            keep = None  # all slots active

        for b in range(B):
            pos_b_full = positions[b]
            if keep is not None:
                kept_idx = [k for k in range(K) if bool(keep[b, k].item())]
                if not kept_idx:
                    continue
            else:
                kept_idx = list(range(K))

            pos_b = torch.tensor(
                [pos_b_full[k] for k in kept_idx], dtype=torch.long, device=device
            )
            assert pos_b.min() >= 0
            assert pos_b.max() < L

            orig_KD = resid_BLD[b, pos_b, :]
            # Detached norms in fp32 so the W gradient is purely directional.
            norms_K1 = orig_KD.float().norm(dim=-1, keepdim=True).detach()

            kept_proj = proj_BKD[b, kept_idx, :]  # (n_kept, d)
            normed_KD = F.normalize(kept_proj, dim=-1)  # fp32
            steered_KD = normed_KD * norms_K1 * steering_coefficient  # fp32

            # Add to the original residual in fp32, cast back to model dtype.
            post = (steered_KD + orig_KD.float())

            if adapter is not None:
                post = adapter(post)  # fp32 -> fp32

            resid_BLD[b, pos_b, :] = post.to(target_dtype)

        return (resid_BLD, *rest) if output_is_tuple else resid_BLD

    return hook_fn
