"""Data construction for multi-token-projection training.

Each example carries a *single* source activation `a` extracted from the target
model at a specific layer/position. The AO prompt has K placeholder tokens; at
hook time the trainable W projects `a` into K vectors, one per placeholder.

To keep the existing TrainingDataPoint plumbing (and `materialize_missing_
steering_vectors`) we store the source activation broadcast to (K, d) — every
row is a copy of `a`. The training hook reads only row 0 as the source.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoTokenizer

from nl_probes.utils.dataset_utils import TrainingDataPoint, create_training_datapoint
from nl_probes.dataset_classes.classification import ClassificationDatapoint


@dataclass
class MultiTokenExampleConfig:
    layer: int
    k_placeholders: int
    # offset (in tokens, from end of prompt) at which to extract the source activation
    activation_offset: int = -3


def make_multi_token_datapoint(
    cls_dp: ClassificationDatapoint,
    cfg: MultiTokenExampleConfig,
    tokenizer: AutoTokenizer,
    source_activation: torch.Tensor | None,
    context_input_ids: list[int] | None,
    context_offset_position: int | None,
    feature_idx: int = -1,
) -> TrainingDataPoint:
    """Build a TrainingDataPoint with K placeholders sharing a single source act.

    Either pass `source_activation` (shape (d,)) directly, or pass
    `context_input_ids` + `context_offset_position` so that the materialize step
    fills it in via a forward pass on the AO model.
    """
    K = cfg.k_placeholders
    if source_activation is not None:
        assert source_activation.dim() == 1
        # Broadcast to (K, d) for compat with existing assertions.
        acts_KD = source_activation.unsqueeze(0).expand(K, -1).contiguous()
        ctx_input_ids = None
        ctx_positions = None
    else:
        assert context_input_ids is not None and context_offset_position is not None
        acts_KD = None
        ctx_input_ids = context_input_ids
        # Same position repeated K times so materialize_missing_steering_vectors
        # produces (K, d) tensor rows that are all copies of the source.
        ctx_positions = [context_offset_position] * K

    return create_training_datapoint(
        datapoint_type="multi_token_classification",
        prompt=cls_dp.classification_prompt,
        target_response=cls_dp.target_response,
        layer=cfg.layer,
        num_positions=K,
        tokenizer=tokenizer,
        acts_BD=acts_KD,
        feature_idx=feature_idx,
        context_input_ids=ctx_input_ids,
        context_positions=ctx_positions,
        ds_label=cls_dp.ds_label,
    )
