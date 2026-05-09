"""Single-source-K-projection dataset loaders, slot-compatible with the rest of
the SFT pipeline (DDP-aware caching, hash-based filenames, train/test splits).

Three loaders:
  - MultiTokenClassificationDatasetLoader: wraps the existing
    `build_multi_token_classification_data` helper and per-classification
    dataset definitions from `nl_probes.dataset_classes.classification`.
  - MultiTokenLatentQADatasetLoader: wraps `build_multi_token_latentqa_data`.
  - MultiTokenPastLensDatasetLoader: lives in `past_lens_data_builder.py`.

All three emit TrainingDataPoints with K=K_target placeholders and exactly one
source residual-stream vector per example (replicated K times in either
`steering_vectors` or `context_positions`). The W projector is responsible for
turning the single source into K distinct injected vectors at hook time.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from nl_probes.dataset_classes.act_dataset_manager import (
    ActDatasetLoader,
    BaseDatasetConfig,
    DatasetLoaderConfig,
)
from nl_probes.dataset_classes.classification import (
    get_classification_datapoints,
)
from nl_probes.dataset_classes import misc
from nl_probes.dataset_classes.misc import latentqa_loader
from nl_probes.multi_token.data_builder import build_multi_token_classification_data
from nl_probes.utils.activation_utils import collect_activations_multiple_layers, get_hf_submodule
from nl_probes.utils.common import layer_percent_to_layer, load_model, load_tokenizer
from nl_probes.utils.dataset_utils import TrainingDataPoint, create_training_datapoint


@dataclass
class MultiTokenClassificationDatasetConfig(BaseDatasetConfig):
    classification_dataset_name: str
    k_placeholders: int = 8
    activation_offset: int = -3
    num_qa_per_sample: int = 2


class MultiTokenClassificationDatasetLoader(ActDatasetLoader):
    """Single-source-K classification loader. Emits K=k_placeholders placeholders
    with the activation drawn from a single token at `activation_offset`.

    Falls back to lazy materialization (save_acts=False) by default — the
    sft.py training loop will fill in steering_vectors via the AO model's
    `disable_adapter()` forward pass per batch.
    """

    def __init__(self, dataset_config: DatasetLoaderConfig, model_kwargs: dict | None = None, model=None):
        super().__init__(dataset_config)
        self.dataset_params: MultiTokenClassificationDatasetConfig = dataset_config.custom_dataset_params

        assert self.dataset_config.dataset_name == "", "Dataset name gets overridden here"
        self.dataset_config.dataset_name = (
            f"multi_token_classification_{self.dataset_params.classification_dataset_name}"
        )
        self.model_kwargs = model_kwargs
        self.model = model

        assert len(self.dataset_config.layer_percents) == 1, (
            "MultiTokenClassificationDatasetLoader requires a single layer_percent — this AO is single-layer."
        )
        self.act_layer = layer_percent_to_layer(self.dataset_config.model_name, self.dataset_config.layer_percents[0])

    def create_dataset(self) -> None:
        tokenizer = load_tokenizer(self.dataset_config.model_name)

        train_datapoints, test_datapoints = get_classification_datapoints(
            self.dataset_params.classification_dataset_name,
            self.dataset_params.num_qa_per_sample,
            self.dataset_config.num_train,
            self.dataset_config.num_test,
            self.dataset_config.seed,
        )

        # Lazy materialization (save_acts=False) by default → no model load here.
        # We only build the model if save_acts=True is requested.
        for split in self.dataset_config.splits:
            datapoints = train_datapoints if split == "train" else test_datapoints
            # Test split always saves acts so that eval is deterministic.
            save_acts = self.dataset_config.save_acts if split == "train" else True

            if save_acts and self.model is None:
                model_kwargs = self.model_kwargs or {}
                self.model = load_model(self.dataset_config.model_name, torch.bfloat16, **model_kwargs)

            data = build_multi_token_classification_data(
                datapoints,
                tokenizer=tokenizer,
                model=self.model,
                act_layer=self.act_layer,
                k_placeholders=self.dataset_params.k_placeholders,
                activation_offset=self.dataset_params.activation_offset,
                batch_size=self.dataset_config.batch_size,
                save_acts=save_acts,
                datapoint_type=self.dataset_config.dataset_name,
            )

            self.save_dataset(data, split)


@dataclass
class MultiTokenLatentQADatasetConfig(BaseDatasetConfig):
    k_placeholders: int = 8
    activation_offset: int = -3
    skip_first: int = 0


class MultiTokenLatentQADatasetLoader(ActDatasetLoader):
    """Single-source-K LatentQA loader.

    LatentQA examples vary widely in prompt length — we pull the activation at
    a single offset from the end of the read_prompt and replicate K times.
    Lazy mode by default to avoid disk blow-up.
    """

    def __init__(self, dataset_config: DatasetLoaderConfig):
        super().__init__(dataset_config)
        self.dataset_params: MultiTokenLatentQADatasetConfig = dataset_config.custom_dataset_params

        assert self.dataset_config.dataset_name == "", "Dataset name gets overridden here"
        self.dataset_config.dataset_name = "multi_token_latentqa"

        assert self.dataset_config.splits == ["train"], "MultiTokenLatentQA only supports train split"
        assert self.dataset_config.num_test == 0
        assert len(self.dataset_config.layer_percents) == 1, (
            "MultiTokenLatentQADatasetLoader requires a single layer_percent — this AO is single-layer."
        )
        self.act_layer = layer_percent_to_layer(self.dataset_config.model_name, self.dataset_config.layer_percents[0])

    def create_dataset(self) -> None:
        tokenizer = load_tokenizer(self.dataset_config.model_name)

        save_acts = self.dataset_config.save_acts
        K = self.dataset_params.k_placeholders

        # Force-load the LatentQA dataset itself (not the activations).
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
            seed=self.dataset_config.seed,
        )

        end = min(self.dataset_params.skip_first + self.dataset_config.num_train, len(ds))
        indices = list(range(self.dataset_params.skip_first, end))

        if save_acts:
            model = load_model(self.dataset_config.model_name, torch.bfloat16)
            submodule = get_hf_submodule(model, self.act_layer)
            submodules = {self.act_layer: submodule}
            device = model.device
        else:
            model = None
            submodule = None
            submodules = None
            device = torch.device("cpu")

        training_data: list[TrainingDataPoint] = []
        bs = self.dataset_config.batch_size

        # Iterate in batches so save_acts=True can run efficient prefill.
        for i in tqdm(range(0, len(indices), bs), desc=f"Building K={K} latentqa data"):
            chunk_idx = indices[i : i + bs]
            chunk = [ds[idx] for idx in chunk_idx]
            chats = [item["read_prompt"] for item in chunk]
            prompt_texts = tokenizer.apply_chat_template(
                chats, tokenize=False, add_generation_prompt=False, enable_thinking=False
            )

            if save_acts:
                toks = tokenizer(
                    prompt_texts, return_tensors="pt", add_special_tokens=False,
                    padding=True, truncation=True, max_length=1024,
                ).to(device)
                acts = collect_activations_multiple_layers(model, submodules, toks, None, None)[self.act_layer]
                attn_BL = toks["attention_mask"]
                ids_BL = toks["input_ids"]

            for j, item in enumerate(chunk):
                if save_acts:
                    attn_mask_L = attn_BL[j].bool()
                    input_ids_L = ids_BL[j, attn_mask_L]
                    L = len(input_ids_L)
                    if L < abs(self.dataset_params.activation_offset) + 1:
                        continue
                    end_pos = L + self.dataset_params.activation_offset
                    acts_LD = acts[j, attn_mask_L]
                    source = acts_LD[end_pos]
                    acts_KD = source.unsqueeze(0).expand(K, -1).contiguous().detach().cpu()
                    ctx_input_ids = None
                    ctx_positions = None
                else:
                    # Lazy mode: pre-tokenize the read_prompt once and store
                    # context_input_ids + repeated context_positions.
                    pt = prompt_texts[j]
                    ids = tokenizer(pt, return_tensors=None, add_special_tokens=False, padding=False)["input_ids"]
                    L = len(ids)
                    if L < abs(self.dataset_params.activation_offset) + 1:
                        continue
                    end_pos = L + self.dataset_params.activation_offset
                    acts_KD = None
                    ctx_input_ids = ids
                    ctx_positions = [end_pos] * K

                user_q = item["dialog"][0]["content"]
                target_resp = item["dialog"][1]["content"]

                tdp = create_training_datapoint(
                    datapoint_type=f"multi_token_latentqa_{item.get('source', 'unknown')}",
                    prompt=user_q,
                    target_response=target_resp,
                    layer=self.act_layer,
                    num_positions=K,
                    tokenizer=tokenizer,
                    acts_BD=acts_KD,
                    feature_idx=-1,
                    context_input_ids=ctx_input_ids,
                    context_positions=ctx_positions,
                    ds_label=item.get("label"),
                )
                training_data.append(tdp)

        print(f"MultiTokenLatentQADatasetLoader: built {len(training_data)} TrainingDataPoints (K={K})")
        self.save_dataset(training_data, "train")
