"""Phase 0 — reproduce K=1 AO baseline on classification.

Restricts `experiments/classification_eval.py` to a single (model, LoRA, layer)
combination so we can sanity-check that the released AO + plumbing reach
reasonable accuracy before any training begins.
"""
import gc
import json
import os
from typing import Any

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from peft import LoraConfig

from nl_probes.dataset_classes.act_dataset_manager import DatasetLoaderConfig
from nl_probes.dataset_classes.classification import (
    ClassificationDatasetConfig,
    ClassificationDatasetLoader,
)
from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.common import load_model, load_tokenizer
from nl_probes.utils.eval import run_evaluation
from nl_probes.base_experiment import sanitize_lora_name


MODEL_NAME = "Qwen/Qwen3-8B"
LORA_PATHS = [
    "adamkarvonen/checkpoints_cls_only_addition_Qwen3-8B",  # the baseline
    None,  # zero-shot reference
]
LAYER_PERCENT = 50
INJECTION_LAYER = 1
DTYPE = torch.bfloat16
BATCH_SIZE = 64
STEERING_COEFFICIENT = 1.0
GENERATION_KWARGS = {"do_sample": False, "temperature": 0.0, "max_new_tokens": 10}

EXPERIMENTS_DIR = "experiments"
OUT_DIR = f"{EXPERIMENTS_DIR}/phase0_classification_layer{LAYER_PERCENT}"
os.makedirs(OUT_DIR, exist_ok=True)

MAIN_TEST_SIZE = 250
CLASSIFICATION_DATASETS: dict[str, dict[str, Any]] = {
    "geometry_of_truth": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "relations": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "sst2": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "md_gender": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "snli": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "ag_news": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "ner": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "tense": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "language_identification": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "singular_plural": {"num_train": 0, "num_test": MAIN_TEST_SIZE, "splits": ["test"]},
    "engels_headline_istrump": {"num_train": 0, "num_test": 250, "splits": ["test"]},
    "engels_headline_isobama": {"num_train": 0, "num_test": 250, "splits": ["test"]},
    "engels_headline_ischina": {"num_train": 0, "num_test": 250, "splits": ["test"]},
    "engels_hist_fig_ismale": {"num_train": 0, "num_test": 250, "splits": ["test"]},
    "engels_news_class_politics": {"num_train": 0, "num_test": 250, "splits": ["test"]},
    "engels_wikidata_isjournalist": {"num_train": 0, "num_test": 250, "splits": ["test"]},
    "engels_wikidata_isathlete": {"num_train": 0, "num_test": 250, "splits": ["test"]},
    "engels_wikidata_ispolitician": {"num_train": 0, "num_test": 250, "splits": ["test"]},
    "engels_wikidata_issinger": {"num_train": 0, "num_test": 250, "splits": ["test"]},
    "engels_wikidata_isresearcher": {"num_train": 0, "num_test": 250, "splits": ["test"]},
}


def canonical_dataset_id(name: str) -> str:
    if name.startswith("classification_"):
        return name[len("classification_"):]
    return name


def load_eval_data(model, tokenizer) -> dict[str, list[Any]]:
    loaders: list[ClassificationDatasetLoader] = []
    for ds_name, dcfg in CLASSIFICATION_DATASETS.items():
        ds_batch_size = BATCH_SIZE // 8 if "language_identification" in ds_name else BATCH_SIZE
        # K=1 placeholder; activation is taken at offset -3 with a 1-token window
        cfg = ClassificationDatasetConfig(
            classification_dataset_name=ds_name,
            max_end_offset=-3,
            min_end_offset=-3,
            max_window_size=1,
            min_window_size=1,
        )
        loader_cfg = DatasetLoaderConfig(
            custom_dataset_params=cfg,
            num_train=dcfg["num_train"],
            num_test=dcfg["num_test"],
            splits=dcfg["splits"],
            model_name=MODEL_NAME,
            layer_percents=[LAYER_PERCENT],
            save_acts=True,
            batch_size=ds_batch_size,
        )
        loaders.append(ClassificationDatasetLoader(dataset_config=loader_cfg, model=model))

    eval_data: dict[str, list[Any]] = {}
    for loader in loaders:
        ds_id = canonical_dataset_id(loader.dataset_config.dataset_name)
        eval_data[ds_id] = loader.load_dataset("test")
    return eval_data


def run_one_lora(model, tokenizer, submodule, lora_path: str | None, eval_data: dict[str, list[Any]]):
    sanitized = None
    if lora_path is not None:
        sanitized = sanitize_lora_name(lora_path)
        if sanitized not in model.peft_config:
            print(f"Loading LoRA: {lora_path}")
            model.load_adapter(lora_path, adapter_name=sanitized, is_trainable=False, low_cpu_mem_usage=True)
        model.set_adapter(sanitized)

    results = {
        "meta": {
            "model_name": MODEL_NAME,
            "dtype": str(DTYPE),
            "layer_percent": LAYER_PERCENT,
            "injection_layer": INJECTION_LAYER,
            "lora_path": lora_path,
            "steering_coefficient": STEERING_COEFFICIENT,
            "eval_batch_size": BATCH_SIZE,
            "generation_kwargs": GENERATION_KWARGS,
            "single_token_mode": True,
        },
        "records": [],
    }

    for ds_id, data in eval_data.items():
        responses = run_evaluation(
            eval_data=data,
            model=model,
            tokenizer=tokenizer,
            submodule=submodule,
            device=torch.device("cuda"),
            dtype=DTYPE,
            global_step=-1,
            lora_path=lora_path,
            eval_batch_size=BATCH_SIZE,
            steering_coefficient=STEERING_COEFFICIENT,
            generation_kwargs=GENERATION_KWARGS,
        )
        for resp, target in zip(responses, data, strict=True):
            results["records"].append({
                "dataset_id": ds_id,
                "ground_truth": resp.api_response,
                "target": target.target_output,
            })

    if sanitized is not None and sanitized in model.peft_config:
        model.delete_adapter(sanitized)

    return results


def main():
    tokenizer = load_tokenizer(MODEL_NAME)
    model = load_model(MODEL_NAME, DTYPE)
    submodule = get_hf_submodule(model, INJECTION_LAYER)

    # Stub adapter so model.peft_config exists (matches classification_eval.py).
    model.add_adapter(LoraConfig(), adapter_name="default")

    eval_data = load_eval_data(model, tokenizer)
    print(f"Loaded {len(eval_data)} datasets")

    for lora in LORA_PATHS:
        lora_label = "base_model" if lora is None else lora.split("/")[-1].replace(".", "_")
        print(f"\n=== Evaluating: {lora_label} ===")
        results = run_one_lora(model, tokenizer, submodule, lora, eval_data)

        out_path = f"{OUT_DIR}/results_lora_{lora_label}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote {out_path}")

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
