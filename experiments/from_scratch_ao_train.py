"""From-scratch K=8 single-source-K-projection AO training.

Mixture: classification (10 datasets) + LatentQA + multi-token past-lens.
Single layer (50%). Joint LoRA + W training. K=8 placeholders, all_identity init.

Usage:
  torchrun --nproc_per_node=1 experiments/from_scratch_ao_train.py [--debug]

`--debug` shrinks counts so a smoke run takes ~10-15 minutes on a single H100.
The full run is sized for ~65M training tokens, matching the paper's published
budget for the cls+latentqa+past_lens AO.
"""
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import random
from dataclasses import asdict
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

from nl_probes.dataset_classes.act_dataset_manager import ActDatasetLoader, DatasetLoaderConfig
from nl_probes.multi_token.loaders import (
    MultiTokenClassificationDatasetConfig,
    MultiTokenClassificationDatasetLoader,
    MultiTokenLatentQADatasetConfig,
    MultiTokenLatentQADatasetLoader,
)
from nl_probes.multi_token.past_lens_data_builder import (
    MultiTokenPastLensDatasetConfig,
    MultiTokenPastLensDatasetLoader,
)
from nl_probes.multi_token.sft_runner import MultiTokenSftConfig, train_model_multi_token
from nl_probes.sft import _ensure_datasets_exist, build_datasets
from nl_probes.utils.common import load_tokenizer, set_seed


def build_loader_groups_multi_token(
    *,
    model_name: str,
    layer_percent: int,
    train_batch_size: int,
    k_placeholders: int,
    save_acts: bool,
    classification_datasets: dict[str, dict[str, Any]],
    latentqa_train_size: int,
    past_lens_train_size: int,
    model_kwargs: dict[str, Any],
) -> dict[str, list[ActDatasetLoader]]:
    """Build the three loader groups (classification, LatentQA, past-lens) for
    single-source-K-projection training. All loaders are pinned to a single
    layer percent, K=k_placeholders.
    """

    classification_loaders: list[ActDatasetLoader] = []
    for ds_name, meta in classification_datasets.items():
        params = MultiTokenClassificationDatasetConfig(
            classification_dataset_name=ds_name,
            k_placeholders=k_placeholders,
            activation_offset=-3,
            num_qa_per_sample=meta.get("num_qa_per_sample", 2),
        )
        bs = meta.get("batch_size", train_batch_size)
        loader_cfg = DatasetLoaderConfig(
            custom_dataset_params=params,
            num_train=meta["num_train"],
            num_test=meta["num_test"],
            splits=meta["splits"],
            model_name=model_name,
            layer_percents=[layer_percent],
            save_acts=save_acts,
            batch_size=bs,
        )
        classification_loaders.append(
            MultiTokenClassificationDatasetLoader(loader_cfg, model_kwargs=model_kwargs)
        )

    latentqa_loader = MultiTokenLatentQADatasetLoader(
        DatasetLoaderConfig(
            custom_dataset_params=MultiTokenLatentQADatasetConfig(
                k_placeholders=k_placeholders,
                activation_offset=-3,
            ),
            num_train=latentqa_train_size,
            num_test=0,
            splits=["train"],
            model_name=model_name,
            layer_percents=[layer_percent],
            save_acts=save_acts,
            batch_size=train_batch_size,
        )
    )

    past_lens_loader = MultiTokenPastLensDatasetLoader(
        DatasetLoaderConfig(
            custom_dataset_params=MultiTokenPastLensDatasetConfig(
                k_placeholders=k_placeholders,
                min_k_tokens=1,
                max_k_tokens=20,
                max_length=512,
            ),
            num_train=past_lens_train_size,
            num_test=0,
            splits=["train"],
            model_name=model_name,
            layer_percents=[layer_percent],
            save_acts=save_acts,
            batch_size=train_batch_size,
        )
    )

    return {
        "classification": classification_loaders,
        "latentqa": [latentqa_loader],
        "past_lens": [past_lens_loader],
    }


def carve_held_out_per_task(
    all_training_data,
    held_out_per_task: int,
    seed: int,
):
    """Carve a fixed number of held-out examples per `datapoint_type`."""
    set_seed(seed)
    by_type: dict[str, list] = {}
    for dp in all_training_data:
        # Bucket fine-grained: "multi_token_classification_*" → bucket per ds.
        by_type.setdefault(dp.datapoint_type, []).append(dp)

    held_out: dict[str, list] = {}
    remaining = []
    for k, lst in by_type.items():
        random.shuffle(lst)
        n = min(held_out_per_task, len(lst) // 10) if held_out_per_task > 0 else 0
        held_out[k] = lst[:n]
        remaining.extend(lst[n:])

    random.shuffle(remaining)
    return remaining, held_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true", help="tiny dataset for smoke testing")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--lr-lora", type=float, default=1e-5)
    ap.add_argument("--lr-projector", type=float, default=3e-4)
    ap.add_argument("--num-epochs", type=int, default=1)
    ap.add_argument("--train-batch-size", type=int, default=16,
                    help="GLOBAL batch size; will be divided by world_size for per-rank.")
    ap.add_argument("--init-strategy", type=str, default="all_identity",
                    choices=["all_identity", "identity_plus_noise", "all_identity_plus_noise"])
    ap.add_argument("--projector-init-std", type=float, default=0.0)
    ap.add_argument("--steering-coefficient", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--run-name", type=str, default="from_scratch_K8")
    ap.add_argument("--save-dir", type=str, default="checkpoints/from_scratch_K8")
    ap.add_argument("--push-to-hub", action="store_true",
                    help="Push final LoRA + projector to HuggingFace Hub")
    ap.add_argument("--hf-repo-id", type=str, default="",
                    help="HF repo id like 'username/model-name'. Required if --push-to-hub.")
    ap.add_argument("--hf-private", action="store_true", default=True)
    ap.add_argument("--held-out-per-task", type=int, default=200)
    args = ap.parse_args()

    dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    model_name = "Qwen/Qwen3-8B"
    layer_percent = 50
    hook_layer = 1
    dtype = torch.bfloat16
    device = torch.device(f"cuda:{local_rank}")

    if args.debug:
        cls_per_ds_train = 200
        latentqa_train = 1000
        past_lens_train = 1000
        save_steps = 9_999_999
    else:
        # Sized to roughly match the paper's ~65M training tokens for a
        # single-source-K=8 mixture. Exact numbers depend on token lengths
        # per task; this gets us in the ballpark and the runner reports the
        # actual token count at start-up.
        cls_per_ds_train = 6000
        latentqa_train = 80000
        past_lens_train = 50000
        save_steps = args.save_every

    main_test_size = 0  # we hold out from train internally; no separate test split

    classification_datasets = {
        "geometry_of_truth": {"num_train": cls_per_ds_train, "num_test": main_test_size, "splits": ["train"]},
        "relations":         {"num_train": cls_per_ds_train, "num_test": main_test_size, "splits": ["train"]},
        "sst2":              {"num_train": cls_per_ds_train, "num_test": main_test_size, "splits": ["train"]},
        "md_gender":         {"num_train": cls_per_ds_train, "num_test": main_test_size, "splits": ["train"]},
        "snli":              {"num_train": cls_per_ds_train, "num_test": main_test_size, "splits": ["train"]},
        "ner":               {"num_train": cls_per_ds_train, "num_test": main_test_size, "splits": ["train"]},
        "tense":             {"num_train": cls_per_ds_train, "num_test": main_test_size, "splits": ["train"]},
        "language_identification": {
            "num_train": cls_per_ds_train, "num_test": main_test_size, "splits": ["train"],
            "batch_size": 4,  # very long sequences
        },
    }

    assert args.train_batch_size % world_size == 0, (
        f"train_batch_size {args.train_batch_size} must be divisible by world_size {world_size}"
    )
    per_rank_batch = args.train_batch_size // world_size

    save_acts = False  # lazy materialization
    model_kwargs: dict[str, Any] = {}

    loader_groups = build_loader_groups_multi_token(
        model_name=model_name,
        layer_percent=layer_percent,
        train_batch_size=per_rank_batch,
        k_placeholders=args.k,
        save_acts=save_acts,
        classification_datasets=classification_datasets,
        latentqa_train_size=latentqa_train,
        past_lens_train_size=past_lens_train,
        model_kwargs=model_kwargs,
    )
    all_loaders: list[ActDatasetLoader] = (
        loader_groups["classification"] + loader_groups["latentqa"] + loader_groups["past_lens"]
    )

    cfg = MultiTokenSftConfig(
        model_name=model_name,
        hook_onto_layer=hook_layer,
        layer_percents=[layer_percent],
        train_batch_size=per_rank_batch,
        eval_batch_size=per_rank_batch * 4,
        activation_collection_batch_size=per_rank_batch * 4,
        eval_steps=args.eval_every,
        eval_on_start=True,
        gradient_checkpointing=True,
        gradient_accumulation_steps=1,
        num_epochs=args.num_epochs,
        lr=args.lr_lora,
        steering_coefficient=args.steering_coefficient,
        save_dir=args.save_dir,
        save_steps=save_steps,
        wandb_run_name=args.run_name,
        # multi-token specifics
        k_placeholders=args.k,
        projector_init_strategy=args.init_strategy,
        projector_init_std=args.projector_init_std,
        projector_lr=args.lr_projector,
        # HF push
        hf_push_to_hub=args.push_to_hub,
        hf_private_repo=args.hf_private,
        hf_repo_name=args.hf_repo_id.split("/")[-1] if args.hf_repo_id else "",
        hf_repo_id=args.hf_repo_id,
    )
    cfg.finalize(dataset_loaders=all_loaders)

    if rank == 0:
        print(f"Run: {cfg.wandb_run_name}")
        print(f"K={cfg.k_placeholders}, init={cfg.projector_init_strategy}, "
              f"lr_lora={cfg.lr}, lr_proj={cfg.projector_lr}")
        print(f"Save dir: {cfg.save_dir}")

    tokenizer = load_tokenizer(cfg.model_name)

    if local_rank == 0:
        _ensure_datasets_exist(all_loaders)
    dist.barrier()

    all_training_data, _ = build_datasets(cfg, dataset_loaders=all_loaders, window_mult=cfg.window_mult)

    if rank == 0:
        print(f"Total training datapoints: {len(all_training_data)}")

    training_data, held_out_by_task = carve_held_out_per_task(
        all_training_data, args.held_out_per_task, seed=cfg.seed,
    )

    if rank == 0:
        print(f"Held out per task:")
        for k, v in held_out_by_task.items():
            print(f"  {k}: {len(v)} examples")
        print(f"Remaining train: {len(training_data)}")

    train_model_multi_token(
        cfg=cfg,
        training_data=training_data,
        held_out_by_task=held_out_by_task,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        model_kwargs=model_kwargs,
        verbose=True,
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
