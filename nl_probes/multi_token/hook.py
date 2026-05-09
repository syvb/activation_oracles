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
) -> Callable:
    assert len(source_activations) == len(positions)
    B = len(source_activations)
    if B == 0:
        raise ValueError("Empty batch")

    # Stack source activations to a (B, d_model) tensor on the right device.
    # The source activations come from a frozen target-model forward pass and do
    # not require gradients themselves.
    source_BD = torch.stack(
        [a.to(device=device).detach() for a in source_activations], dim=0
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

        # Project: (B, K, d_model). Cast source to whatever dtype the residual
        # stream is in (bf16 in our setup) so the matmul matches.
        proj_BKD = projector(source_BD.to(resid_BLD.dtype))

        for b in range(B):
            pos_b = torch.tensor(positions[b], dtype=torch.long, device=device)
            assert pos_b.min() >= 0
            assert pos_b.max() < L

            orig_KD = resid_BLD[b, pos_b, :]
            norms_K1 = orig_KD.norm(dim=-1, keepdim=True).detach()

            normed_KD = F.normalize(proj_BKD[b], dim=-1)
            steered_KD = (normed_KD * norms_K1 * steering_coefficient).to(resid_BLD.dtype)

            post = steered_KD + orig_KD

            if adapter is not None:
                post = adapter(post)

            resid_BLD[b, pos_b, :] = post

        return (resid_BLD, *rest) if output_is_tuple else resid_BLD

    return hook_fn
