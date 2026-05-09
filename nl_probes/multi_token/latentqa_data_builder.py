"""Build single-source K-placeholder TrainingDataPoints from LatentQA training data.

Same activation-injection format as `build_multi_token_classification_data`,
but the source activation comes from a single position (offset −3 from end of
the LatentQA `read_prompt`) and the AO question/answer come from `dialog`.
"""
from __future__ import annotations

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from nl_probes.dataset_classes import misc
from nl_probes.dataset_classes.misc import latentqa_loader
from nl_probes.utils.activation_utils import collect_activations_multiple_layers, get_hf_submodule
from nl_probes.utils.dataset_utils import TrainingDataPoint, create_training_datapoint


@torch.no_grad()
def build_multi_token_latentqa_data(
    *,
    tokenizer: AutoTokenizer,
    model,                          # base target model (no LoRA)
    act_layer: int,
    k_placeholders: int,
    n_examples: int,
    activation_offset: int = -3,
    batch_size: int = 8,
    seed: int = 42,
    skip_first: int = 0,
) -> list[TrainingDataPoint]:
    """Pull LatentQA training examples and build K-placeholder TrainingDataPoints
    with a single source activation per example, replicated K times.

    `skip_first` lets you carve a held-out slice (e.g., skip the first 5K so a
    follow-up call with `skip_first=0, n_examples=5000` gives a non-overlapping
    eval set).
    """
    assert tokenizer.padding_side == "left"
    K = k_placeholders

    paths = latentqa_loader.DataPaths(
        system=None,
        stimulus_completion="datasets/latentqa_datasets/train/stimulus_completion.json",
        stimulus="datasets/latentqa_datasets/train/stimulus.json",
        control="datasets/latentqa_datasets/train/control.json",
        qa="datasets/latentqa_datasets/train/qa.json",
    )
    ds = latentqa_loader.load_latentqa_dataset(
        paths,
        filter_prefixes=[],
        train_percent=1.0,
        add_thought_tokens=False,
        seed=seed,
    )

    # Slice (the dataset is __getitem__ but doesn't support python slices)
    total = len(ds)
    end = min(skip_first + n_examples, total)
    indices = list(range(skip_first, end))
    items = [ds[i] for i in indices]
    print(f"LatentQA: pulled {len(items)} examples out of {total} (skip_first={skip_first}, n={n_examples})")

    submodule = get_hf_submodule(model, act_layer)
    submodules = {act_layer: submodule}
    device = model.device

    out: list[TrainingDataPoint] = []

    for i in tqdm(range(0, len(items), batch_size), desc=f"Building K={K} latentqa data"):
        chunk = items[i : i + batch_size]
        chats = [item["read_prompt"] for item in chunk]
        prompt_texts = tokenizer.apply_chat_template(
            chats, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        toks = tokenizer(
            prompt_texts,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(device)

        acts = collect_activations_multiple_layers(model, submodules, toks, None, None)[act_layer]

        for j, item in enumerate(chunk):
            attn_mask_L = toks["attention_mask"][j].bool()
            input_ids_L = toks["input_ids"][j, attn_mask_L]
            L = len(input_ids_L)
            if L < abs(activation_offset) + 1:
                # too short, skip
                continue
            end_pos = L + activation_offset
            assert 0 <= end_pos < L

            acts_LD = acts[j, attn_mask_L]
            source = acts_LD[end_pos]
            acts_KD = source.unsqueeze(0).expand(K, -1).contiguous().detach().cpu()

            user_q = item["dialog"][0]["content"]
            target_resp = item["dialog"][1]["content"]

            tdp = create_training_datapoint(
                datapoint_type=f"multi_token_latentqa_{item.get('source', 'unknown')}",
                prompt=user_q,
                target_response=target_resp,
                layer=act_layer,
                num_positions=K,
                tokenizer=tokenizer,
                acts_BD=acts_KD,
                feature_idx=-1,
                context_input_ids=None,
                context_positions=None,
                ds_label=item.get("label"),
            )
            out.append(tdp)

    print(f"LatentQA: built {len(out)} TrainingDataPoints (K={K})")
    return out
