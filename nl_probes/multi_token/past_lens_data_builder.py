"""Single-source-K-projection past-lens dataset builder.

Same task as `nl_probes/dataset_classes/past_lens_dataset.py` (predict the
past or future N tokens around an activation), but the AO sees ONE source
activation projected to K placeholder slots via the trainable W.

Data structure: each TrainingDataPoint has K placeholders, but the source
activation is a single residual-stream vector at one chosen position (the
last position of an `act_window`). We feed it as `context_positions=[pos]*K`
(lazy mode, default) so `materialize_missing_steering_vectors` produces K
identical rows that the multi-token hook will project via W.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Generator

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from datasets import load_dataset
from nl_probes.dataset_classes.act_dataset_manager import (
    ActDatasetLoader,
    BaseDatasetConfig,
    DatasetLoaderConfig,
)
from nl_probes.dataset_classes.past_lens_dataset import hf_mixed_dataset_to_generator
from nl_probes.utils.activation_utils import collect_activations_multiple_layers, get_hf_submodule
from nl_probes.utils.common import layer_percent_to_layer, load_model, load_tokenizer
from nl_probes.utils.dataset_utils import TrainingDataPoint, create_training_datapoint


@dataclass
class MultiTokenPastLensDatasetConfig(BaseDatasetConfig):
    """Single-source past-lens config.

    `k_placeholders` is the K for the AO prompt. `min_k_tokens`/`max_k_tokens`
    is the number of past/future tokens to predict. The activation is a single
    position chosen randomly within the valid range; we don't sweep activation
    spans (unlike the original PastLensDatasetConfig).
    """

    k_placeholders: int = 8
    min_k_tokens: int = 1
    max_k_tokens: int = 20
    max_length: int = 512
    directions: list[str] = field(default_factory=lambda: ["past", "future"])


class MultiTokenPastLensDatasetLoader(ActDatasetLoader):
    """Drop-in replacement for PastLensDatasetLoader that emits single-source-K
    examples. Hashes its full config (including k_placeholders) so cached files
    are distinct from the single-token / multi-token-window variants.
    """

    def __init__(self, dataset_config: DatasetLoaderConfig):
        super().__init__(dataset_config)
        assert self.dataset_config.dataset_name == "", "Dataset name gets overridden here"
        self.dataset_config.dataset_name = "multi_token_past_lens"

        self.dataset_params: MultiTokenPastLensDatasetConfig = dataset_config.custom_dataset_params

        assert self.dataset_config.splits == ["train"], "Past-lens dataset only supports train split"
        assert self.dataset_config.num_test == 0, "Past-lens dataset doesn't support test split"

        if self.dataset_config.num_train < self.dataset_config.batch_size:
            raise ValueError(
                f"num_train {self.dataset_config.num_train} must be >= batch_size {self.dataset_config.batch_size}"
            )

    def create_dataset(self) -> None:
        tokenizer = load_tokenizer(self.dataset_config.model_name)
        dataset = hf_mixed_dataset_to_generator(tokenizer)

        training_data = collect_multi_token_past_lens_acts(
            dataset_config=self.dataset_config,
            custom_dataset_params=self.dataset_params,
            tokenizer=tokenizer,
            dataset=dataset,
            num_datapoints=self.dataset_config.num_train,
            dtype=torch.bfloat16,
        )

        self.save_dataset(training_data, "train")


def collect_multi_token_past_lens_acts(
    dataset_config: DatasetLoaderConfig,
    custom_dataset_params: MultiTokenPastLensDatasetConfig,
    tokenizer: AutoTokenizer,
    dataset: Generator,
    num_datapoints: int,
    dtype: torch.dtype,
) -> list[TrainingDataPoint]:
    """Mirror of `collect_past_lens_acts` but emits one source position per
    example. K placeholders are filled with K identical copies of that source.
    """
    random.seed(dataset_config.seed)
    torch.manual_seed(dataset_config.seed)

    layers = [
        layer_percent_to_layer(dataset_config.model_name, layer_percent)
        for layer_percent in dataset_config.layer_percents
    ]
    K = custom_dataset_params.k_placeholders

    device = torch.device("cpu")
    if dataset_config.save_acts:
        model = load_model(dataset_config.model_name, dtype)
        submodules = {layer: get_hf_submodule(model, layer) for layer in layers}
        device = model.device

    training_data: list[TrainingDataPoint] = []

    for _ in tqdm(range(0, num_datapoints, dataset_config.batch_size), desc="Collecting multi-token past-lens acts"):
        inputs: list[str] = []
        for _ in range(dataset_config.batch_size):
            inputs.append(next(dataset))

        tokenized_inputs = tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=custom_dataset_params.max_length,
            add_special_tokens=False,
        ).to(device)

        if dataset_config.save_acts:
            acts_BLD_by_layer_dict = collect_activations_multiple_layers(
                model, submodules, tokenized_inputs, min_offset=None, max_offset=None
            )

        attn_mask_BL = tokenized_inputs["attention_mask"]
        input_ids_BL = tokenized_inputs["input_ids"]

        for layer in layers:
            for j in range(len(inputs)):
                attn_mask_L = attn_mask_BL[j].bool()
                input_ids_L_full = input_ids_BL[j, attn_mask_L]
                L = len(input_ids_L_full)

                k_tokens = random.randint(custom_dataset_params.min_k_tokens, custom_dataset_params.max_k_tokens)
                direction = random.choice(custom_dataset_params.directions)

                if direction == "past":
                    # Source position must be at least k_tokens from start, and not the last token.
                    if L < k_tokens + 2:
                        continue
                    src_pos_min = k_tokens
                    src_pos_max = L - 2
                    if src_pos_max < src_pos_min:
                        continue
                    src_pos = random.randint(src_pos_min, src_pos_max)
                    target_positions = list(range(src_pos - k_tokens, src_pos))
                    target_tokens = input_ids_L_full[target_positions]
                    target_text = tokenizer.decode(target_tokens, skip_special_tokens=True)
                    prompt = f"Can you predict the previous {k_tokens} tokens that came before this?"
                    context_cutoff = src_pos
                else:  # future
                    if L < k_tokens + 2:
                        continue
                    src_pos_min = 1
                    src_pos_max = L - 1 - k_tokens
                    if src_pos_max < src_pos_min:
                        continue
                    src_pos = random.randint(src_pos_min, src_pos_max)
                    target_positions = list(range(src_pos + 1, src_pos + 1 + k_tokens))
                    target_tokens = input_ids_L_full[target_positions]
                    target_text = tokenizer.decode(target_tokens, skip_special_tokens=True)
                    prompt = f"Can you predict the next {k_tokens} tokens that come after this?"
                    context_cutoff = src_pos

                # Slice the context just to what's needed to compute the source activation.
                context_input_ids_slice = input_ids_L_full[: context_cutoff + 1]

                if dataset_config.save_acts:
                    acts_LD = acts_BLD_by_layer_dict[layer][j, attn_mask_L]
                    source = acts_LD[src_pos]  # (d,)
                    acts_KD = source.unsqueeze(0).expand(K, -1).contiguous().detach().cpu()
                    context_input_ids = None
                    context_positions = None
                else:
                    acts_KD = None
                    context_input_ids = (
                        context_input_ids_slice.tolist()
                        if isinstance(context_input_ids_slice, torch.Tensor)
                        else list(context_input_ids_slice)
                    )
                    context_positions = [src_pos] * K

                training_data_point = create_training_datapoint(
                    datapoint_type=dataset_config.dataset_name,
                    prompt=prompt,
                    target_response=target_text,
                    layer=layer,
                    num_positions=K,
                    tokenizer=tokenizer,
                    acts_BD=acts_KD,
                    feature_idx=-1,
                    context_input_ids=context_input_ids,
                    context_positions=context_positions,
                )
                training_data.append(training_data_point)

    return training_data
