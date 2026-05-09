"""Build classification training data for multi-token-projection training.

A close clone of `nl_probes/dataset_classes/classification.create_vector_dataset`
but with K_placeholders decoupled from window_size (we always want window=1 so
one source activation per example) and K_placeholders >= 1 controlling how
many placeholder tokens appear in the AO prompt.
"""
from __future__ import annotations

import random
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from nl_probes.dataset_classes.classification import ClassificationDatapoint
from nl_probes.utils.activation_utils import collect_activations_multiple_layers, get_hf_submodule
from nl_probes.utils.dataset_utils import TrainingDataPoint, create_training_datapoint


@torch.no_grad()
def build_multi_token_classification_data(
    datapoints: list[ClassificationDatapoint],
    tokenizer: AutoTokenizer,
    model,
    act_layer: int,
    k_placeholders: int,
    activation_offset: int = -3,
    batch_size: int = 16,
    save_acts: bool = True,
    datapoint_type: str = "multi_token_classification",
) -> list[TrainingDataPoint]:
    """Build TrainingDataPoints with K placeholders and a single source activation.

    `model` is the base target model (e.g. Qwen3-8B with no AO LoRA active).
    For each example:
      * Tokenize the activation_prompt via chat template.
      * Take a single activation at `activation_offset` from end-of-prompt (same
        position as the K=1 single-token baseline).
      * Replicate this activation K times so the existing pipeline accepts it.
      * Build a datapoint whose prompt has K placeholder tokens.
    """
    assert tokenizer.padding_side == "left", "Padding side must be left"
    K = k_placeholders
    out: list[TrainingDataPoint] = []
    if save_acts:
        assert model is not None, "save_acts=True requires a loaded base model"
        submodule = get_hf_submodule(model, act_layer)
        submodules = {act_layer: submodule}
        device = model.device
    else:
        # Lazy mode: only the tokenizer is needed; activations get materialized
        # later by `materialize_missing_steering_vectors` in the training loop.
        submodule = None
        submodules = None
        device = torch.device("cpu")

    for i in tqdm(range(0, len(datapoints), batch_size), desc=f"Building K={K} cls data"):
        chunk = datapoints[i : i + batch_size]
        chats = [[{"role": "user", "content": dp.activation_prompt}] for dp in chunk]
        prompt_texts = tokenizer.apply_chat_template(chats, tokenize=False)
        toks = tokenizer(
            prompt_texts,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        ).to(device)

        if save_acts:
            acts = collect_activations_multiple_layers(model, submodules, toks, None, None)[act_layer]

        for j, dp in enumerate(chunk):
            attn_mask_L = toks["attention_mask"][j].bool()
            input_ids_L = toks["input_ids"][j, attn_mask_L]
            L = len(input_ids_L)
            assert L > 0
            end_pos = L + activation_offset
            assert 0 <= end_pos < L, f"end_pos={end_pos} L={L}"

            if save_acts:
                acts_LD = acts[j, attn_mask_L]
                source = acts_LD[end_pos]  # (d,)
                acts_KD = source.unsqueeze(0).expand(K, -1).contiguous().detach().cpu()
                ctx_input_ids = None
                ctx_positions = None
            else:
                acts_KD = None
                ctx_input_ids = input_ids_L.tolist() if isinstance(input_ids_L, torch.Tensor) else list(input_ids_L)
                ctx_positions = [end_pos] * K  # same position K times -> K copies of source

            tdp = create_training_datapoint(
                datapoint_type=datapoint_type,
                prompt=dp.classification_prompt,
                target_response=dp.target_response,
                layer=act_layer,
                num_positions=K,
                tokenizer=tokenizer,
                acts_BD=acts_KD,
                feature_idx=-1,
                context_input_ids=ctx_input_ids,
                context_positions=ctx_positions,
                ds_label=dp.ds_label,
            )
            out.append(tdp)

    return out
