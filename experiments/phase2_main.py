"""Phase 2 — Main training run.

Same machinery as `phase1_smoke.py`, parameterized for the larger main run
(default ~30K examples × 1 epoch). Scale via `--n-train-per-ds`.
"""
import argparse
import gc
import os
import random

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch

from nl_probes.dataset_classes.classification import (
    ClassificationDatapoint,
    get_classification_datapoints,
)
from nl_probes.multi_token.data_builder import build_multi_token_classification_data
from nl_probes.multi_token.train import MultiTokenTrainConfig, train
from nl_probes.utils.common import load_model, load_tokenizer, layer_percent_to_layer
from nl_probes.utils.dataset_utils import TrainingDataPoint


# Train on the same subdatasets the SFT mixture uses.
TRAIN_SUBDATASETS = [
    "geometry_of_truth",
    "relations",
    "sst2",
    "md_gender",
    "snli",
    "ner",
    "tense",
    "ag_news",
]
TEST_SUBDATASETS = [
    # IID
    "geometry_of_truth", "relations", "sst2", "md_gender", "snli", "ner", "tense",
    # OOD
    "ag_news", "language_identification", "singular_plural",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-train-per-ds", type=int, default=2000,
                    help="Per-subdataset; 8 datasets * 2000 * 2 QA per sample ~= 32K examples")
    ap.add_argument("--n-test-per-ds", type=int, default=250)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--no-adapter", action="store_true")
    ap.add_argument("--steering-coefficient", type=float, default=1.0)
    ap.add_argument("--run-name", type=str, default="phase2_k8_main")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--init-strategy", type=str, default="identity_plus_noise")
    args = ap.parse_args()

    train_dps: list[ClassificationDatapoint] = []
    test_dps_by_ds: dict[str, list[ClassificationDatapoint]] = {}

    print("Gathering classification datapoints...")
    seen = set()
    for ds_name in TRAIN_SUBDATASETS + TEST_SUBDATASETS:
        if ds_name in seen:
            continue
        seen.add(ds_name)
        n_train = args.n_train_per_ds if ds_name in TRAIN_SUBDATASETS else 0
        n_test = args.n_test_per_ds if ds_name in TEST_SUBDATASETS else 0
        train_split, test_split = get_classification_datapoints(
            dataset_name=ds_name,
            num_qa_per_sample=2,
            train_examples=n_train,
            test_examples=n_test,
            random_seed=42,
        )
        if n_train > 0:
            train_dps.extend(train_split)
        if n_test > 0:
            test_dps_by_ds[ds_name] = test_split

    random.shuffle(train_dps)
    print(f"  train: {len(train_dps)}  test datasets: {len(test_dps_by_ds)}")

    cfg = MultiTokenTrainConfig(
        k_placeholders=args.k,
        train_batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.epochs,
        run_name=args.run_name,
        use_adapter=not args.no_adapter,
        steering_coefficient=args.steering_coefficient,
        eval_every=args.eval_every,
        log_every=args.log_every,
        projector_init_strategy=args.init_strategy,
    )
    act_layer = layer_percent_to_layer(cfg.model_name, cfg.layer_percent)
    print(f"Activation layer: {act_layer}")

    print("Loading target model for activation extraction (no LoRA)...")
    tokenizer = load_tokenizer(cfg.model_name)
    target_model = load_model(cfg.model_name, torch.bfloat16)
    target_model.eval()

    print("Building training data...")
    train_td = build_multi_token_classification_data(
        train_dps,
        tokenizer=tokenizer,
        model=target_model,
        act_layer=act_layer,
        k_placeholders=cfg.k_placeholders,
        activation_offset=-3,
        batch_size=16,
        save_acts=True,
    )

    test_td_by_ds: dict[str, list[TrainingDataPoint]] = {}
    for ds_name, dps in test_dps_by_ds.items():
        test_td_by_ds[ds_name] = build_multi_token_classification_data(
            dps,
            tokenizer=tokenizer,
            model=target_model,
            act_layer=act_layer,
            k_placeholders=cfg.k_placeholders,
            activation_offset=-3,
            batch_size=16,
            save_acts=True,
        )

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
