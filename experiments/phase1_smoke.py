"""Phase 1 smoke test — train W + adapter at K=8 on a small slice.

Verifies the training plumbing works end-to-end before committing to a full
run. Targets ~10 minutes on a single H100.
"""
import argparse
import gc
import os
import random
from typing import Any

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from peft import PeftModel

from nl_probes.dataset_classes.classification import (
    ClassificationDatapoint,
    get_classification_datapoints,
)
from nl_probes.multi_token.data_builder import build_multi_token_classification_data
from nl_probes.multi_token.train import MultiTokenTrainConfig, train
from nl_probes.utils.common import load_model, load_tokenizer
from nl_probes.utils.common import layer_percent_to_layer
from nl_probes.utils.dataset_utils import TrainingDataPoint


# Smoke-test scale: keep it small.
DEFAULT_TRAIN_SUBDATASETS = [
    "geometry_of_truth",
    "relations",
    "sst2",
    "md_gender",
    "snli",
    "ner",
    "tense",
    "ag_news",  # in train mixture per nl_probes/sft.py
]
DEFAULT_TEST_SUBDATASETS = [
    "geometry_of_truth",
    "relations",
    "sst2",
    "md_gender",
    "snli",
    "ag_news",
    "ner",
    "tense",
    "language_identification",
    "singular_plural",
]


def gather_train_datapoints(
    subdatasets: list[str],
    n_per_subdataset: int,
    n_test_per_subdataset: int,
    seed: int = 42,
) -> tuple[list[ClassificationDatapoint], dict[str, list[ClassificationDatapoint]]]:
    """Pull ClassificationDatapoints for each subdataset, returning train list and per-ds test dict."""
    train_all: list[ClassificationDatapoint] = []
    test_all: dict[str, list[ClassificationDatapoint]] = {}
    for ds_name in subdatasets:
        train, test = get_classification_datapoints(
            dataset_name=ds_name,
            num_qa_per_sample=2,
            train_examples=n_per_subdataset,
            test_examples=n_test_per_subdataset,
            random_seed=seed,
        )
        train_all.extend(train)
        if n_test_per_subdataset > 0:
            test_all[f"classification_{ds_name}"] = test
    return train_all, test_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8, help="Number of placeholder tokens")
    ap.add_argument("--n-train-per-ds", type=int, default=1000)
    ap.add_argument("--n-test-per-ds", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--save-acts-on-cpu", action="store_true",
                    help="Compute activations on CPU (slower, less GPU mem)")
    ap.add_argument("--run-name", type=str, default="phase1_smoke_k8")
    ap.add_argument("--no-adapter", action="store_true")
    ap.add_argument("--steering-coefficient", type=float, default=1.0)
    args = ap.parse_args()

    # Build train + test ClassificationDatapoints
    print("Gathering classification datapoints...")
    train_dps, test_dps_by_ds = gather_train_datapoints(
        DEFAULT_TRAIN_SUBDATASETS,
        n_per_subdataset=args.n_train_per_ds,
        n_test_per_subdataset=args.n_test_per_ds,
    )
    print(f"  train: {len(train_dps)}  test datasets: {len(test_dps_by_ds)} (each ~{args.n_test_per_ds})")
    random.shuffle(train_dps)

    # We want to compute activations from the *base* target model (no AO LoRA).
    # The simplest path: load Qwen3-8B on its own here, build training+test
    # data with `save_acts=True`, then unload that model before train() loads
    # the AO. Memory is plenty on H100 80GB to do this sequentially.

    cfg = MultiTokenTrainConfig(
        k_placeholders=args.k,
        train_batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.epochs,
        run_name=args.run_name,
        use_adapter=not args.no_adapter,
        steering_coefficient=args.steering_coefficient,
    )
    act_layer = layer_percent_to_layer(cfg.model_name, cfg.layer_percent)
    print(f"Activation layer: {act_layer}")

    print("Loading target model for activation extraction (no LoRA)...")
    tokenizer = load_tokenizer(cfg.model_name)
    target_model = load_model(cfg.model_name, torch.bfloat16)
    target_model.eval()

    print("Building training data with multi-token placeholders...")
    train_td = build_multi_token_classification_data(
        train_dps,
        tokenizer=tokenizer,
        model=target_model,
        act_layer=act_layer,
        k_placeholders=cfg.k_placeholders,
        activation_offset=-3,
        batch_size=8,
        save_acts=True,
    )

    test_td_by_ds: dict[str, list[TrainingDataPoint]] = {}
    for ds_name, dps in test_dps_by_ds.items():
        ds_id = ds_name.replace("classification_", "")
        test_td_by_ds[ds_id] = build_multi_token_classification_data(
            dps,
            tokenizer=tokenizer,
            model=target_model,
            act_layer=act_layer,
            k_placeholders=cfg.k_placeholders,
            activation_offset=-3,
            batch_size=8,
            save_acts=True,
        )

    # Unload target model to free up memory before loading AO
    del target_model
    torch.cuda.empty_cache()
    gc.collect()

    log_path = f"logs/{cfg.run_name}.json"
    os.makedirs("logs", exist_ok=True)

    train(
        cfg=cfg,
        training_data=train_td,
        eval_datasets=test_td_by_ds,
        tokenizer=tokenizer,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        log_path=log_path,
    )


if __name__ == "__main__":
    main()
