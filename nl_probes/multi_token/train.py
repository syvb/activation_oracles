"""Train W (MultiTokenProjector) + InjectionAdapter on top of a frozen AO.

Compared to nl_probes/sft.py this script:
  - keeps the AO LoRA frozen (no LoRA training)
  - only optimizes the projector and adapter
  - uses the multi_token_steering_hook so gradients flow back through W
  - runs on a single GPU (no DDP)
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.common import load_model, load_tokenizer, set_seed
from nl_probes.utils.dataset_utils import (
    BatchData,
    TrainingDataPoint,
    construct_batch,
    materialize_missing_steering_vectors,
)
from nl_probes.utils.steering_hooks import add_hook
from nl_probes.utils.eval import run_evaluation, score_eval_responses

from nl_probes.multi_token.projector import MultiTokenProjector, InjectionAdapter
from nl_probes.multi_token.hook import get_multi_token_steering_hook


@dataclass
class MultiTokenTrainConfig:
    model_name: str = "Qwen/Qwen3-8B"
    base_lora_path: str = "adamkarvonen/checkpoints_cls_only_addition_Qwen3-8B"
    hook_layer: int = 1
    layer_percent: int = 50
    k_placeholders: int = 8
    use_adapter: bool = True
    adapter_hidden_mult: int = 2
    projector_init_std: float = 0.02
    projector_init_strategy: str = "identity_plus_noise"
    # LoRA fallback (per PLAN.md): instead of an injection-layer adapter,
    # let the AO LoRA itself continue training together with W.
    train_ao_lora: bool = False
    ao_lora_lr: float = 1e-5

    train_batch_size: int = 8
    eval_batch_size: int = 32
    grad_accum: int = 1
    lr: float = 3e-4
    num_epochs: int = 1
    max_grad_norm: float = 1.0
    warmup_frac: float = 0.05

    steering_coefficient: float = 1.0
    seed: int = 42
    log_every: int = 10
    save_every: int = 1_000_000  # effectively off; small experiment
    eval_every: int = 200
    save_dir: str = "checkpoints/multi_token"
    run_name: str = "k8_main"

    generation_kwargs: dict = field(default_factory=lambda: {"do_sample": False, "max_new_tokens": 10})


def _gather_source_activations(
    batch: BatchData,
) -> list[torch.Tensor]:
    """Pull the single source activation for each example.

    Each `batch.steering_vectors[b]` is shape (K, d) with K identical rows
    (see nl_probes/multi_token/data.py). Take row 0 as the single source.
    """
    sources = []
    for sv in batch.steering_vectors:
        assert sv is not None, "steering_vectors must be materialized before training"
        sources.append(sv[0])
    return sources


def freeze_base_model(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def cosine_warmup_lr(step: int, warmup_steps: int, total_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1 + math.cos(math.pi * progress))


def train(
    cfg: MultiTokenTrainConfig,
    training_data: list[TrainingDataPoint],
    eval_datasets: dict[str, list[TrainingDataPoint]],
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    dtype: torch.dtype,
    log_path: str,
):
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    # 1. Load Qwen3-8B + AO LoRA
    model = load_model(cfg.model_name, dtype)
    model = PeftModel.from_pretrained(model, cfg.base_lora_path, is_trainable=cfg.train_ao_lora)
    submodule = get_hf_submodule(model, cfg.hook_layer, use_lora=True)

    if cfg.train_ao_lora:
        # LoRA fallback: only LoRA adapter params are trainable; everything
        # else (base model) stays frozen.
        for name, p in model.named_parameters():
            if "lora_" not in name:
                p.requires_grad = False
        model.train()
    else:
        model.eval()
        freeze_base_model(model)

    # Make embedding outputs require grad so backward reaches the hook
    model.enable_input_require_grads()

    # 2. Build trainable modules. Keep them in fp32 (small modules, no benefit
    # from bf16 weights; the hook casts outputs as needed).
    d_model = model.config.hidden_size
    projector = MultiTokenProjector(
        d_model,
        cfg.k_placeholders,
        init_std=cfg.projector_init_std,
        init_strategy=cfg.projector_init_strategy,
    ).to(device=device, dtype=torch.float32)
    adapter = InjectionAdapter(d_model, hidden_mult=cfg.adapter_hidden_mult).to(device=device, dtype=torch.float32) if cfg.use_adapter else None

    trainable_params = list(projector.parameters())
    if adapter is not None:
        trainable_params += list(adapter.parameters())

    # If LoRA is being trained, add LoRA params with their own LR group.
    if cfg.train_ao_lora:
        lora_params = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
        n_lora = sum(p.numel() for p in lora_params)
        print(f"Trainable LoRA params: {n_lora:,}")
        optimizer = torch.optim.AdamW(
            [
                {"params": trainable_params, "lr": cfg.lr},
                {"params": lora_params, "lr": cfg.ao_lora_lr},
            ]
        )
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr)

    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable W+adapter params: {n_trainable:,}")

    global_step_size = cfg.train_batch_size
    effective_steps = (len(training_data) // global_step_size) * global_step_size
    if effective_steps != len(training_data):
        training_data = training_data[:effective_steps]
    num_batches = len(training_data) // cfg.train_batch_size
    optim_steps = num_batches // cfg.grad_accum
    warmup_steps = int(optim_steps * cfg.warmup_frac)
    total_steps = optim_steps * cfg.num_epochs

    # 3. Training loop
    log = {"config": cfg.__dict__, "steps": [], "evals": []}
    global_step = 0
    optim_step = 0
    accum_loss = 0.0

    projector.train()
    if adapter is not None:
        adapter.train()

    # Step-0 eval: confirms the K=8 starting state is close to the K=1 baseline
    # (W_1 = I, W_2..W_K small noise). Useful sanity check before training.
    if eval_datasets:
        _run_eval(cfg, projector, adapter, model, tokenizer, submodule, device, dtype, eval_datasets, 0, log, log_path)

    for epoch in range(cfg.num_epochs):
        random.shuffle(training_data)
        pbar = tqdm(range(0, len(training_data), cfg.train_batch_size), desc=f"epoch {epoch+1}")
        for start in pbar:
            batch_list = training_data[start : start + cfg.train_batch_size]
            batch_list = materialize_missing_steering_vectors(batch_list, tokenizer, model)
            batch = construct_batch(batch_list, tokenizer, device)

            # Build hook
            sources = _gather_source_activations(batch)
            hook_fn = get_multi_token_steering_hook(
                source_activations=sources,
                projector=projector,
                adapter=adapter,
                positions=batch.positions,
                steering_coefficient=cfg.steering_coefficient,
                device=device,
            )

            with add_hook(submodule, hook_fn):
                out = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, labels=batch.labels)
            loss = out.loss / cfg.grad_accum
            loss.backward()
            accum_loss += loss.item()

            if (global_step + 1) % cfg.grad_accum == 0:
                gn = clip_grad_norm_(trainable_params, cfg.max_grad_norm)

                # Compute W-only gradient norm before stepping
                w_gn = projector.linear.weight.grad.detach().norm().item() if projector.linear.weight.grad is not None else 0.0

                # Cosine schedule scales each param group's base LR.
                lr_factor = cosine_warmup_lr(optim_step, warmup_steps, total_steps, 1.0)
                for pg in optimizer.param_groups:
                    base_lr = pg.get("base_lr", pg["lr"])
                    pg["base_lr"] = base_lr
                    pg["lr"] = base_lr * lr_factor
                lr_now = optimizer.param_groups[0]["lr"]
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if optim_step % cfg.log_every == 0:
                    print(f"[step {optim_step}] loss={accum_loss:.4f} lr={lr_now:.2e} grad_norm={float(gn):.3f} W_grad_norm={w_gn:.3f}")
                    log["steps"].append({
                        "step": optim_step,
                        "loss": accum_loss,
                        "lr": lr_now,
                        "grad_norm": float(gn),
                        "w_grad_norm": w_gn,
                    })
                    with open(log_path, "w") as f:
                        json.dump(log, f, indent=2)

                if cfg.eval_every > 0 and optim_step > 0 and optim_step % cfg.eval_every == 0:
                    _run_eval(cfg, projector, adapter, model, tokenizer, submodule, device, dtype, eval_datasets, optim_step, log, log_path)

                optim_step += 1
                accum_loss = 0.0
            global_step += 1

    # Final eval + save
    if eval_datasets:
        _run_eval(cfg, projector, adapter, model, tokenizer, submodule, device, dtype, eval_datasets, optim_step, log, log_path)

    save_path = Path(cfg.save_dir) / cfg.run_name
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save({
        "projector_state_dict": projector.state_dict(),
        "adapter_state_dict": adapter.state_dict() if adapter is not None else None,
        "config": cfg.__dict__,
    }, save_path / "weights.pt")
    print(f"Saved weights to {save_path / 'weights.pt'}")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)


def _run_eval(
    cfg: MultiTokenTrainConfig,
    projector: MultiTokenProjector,
    adapter: InjectionAdapter | None,
    model,
    tokenizer,
    submodule,
    device: torch.device,
    dtype: torch.dtype,
    eval_datasets: dict[str, list[TrainingDataPoint]],
    step: int,
    log: dict,
    log_path: str,
):
    """Greedy-decode eval matching run_evaluation, but with our multi_token hook."""
    from nl_probes.utils.dataset_utils import get_prompt_tokens_only
    from nl_probes.utils.steering_hooks import add_hook
    print(f"\n=== Eval at step {step} ===")
    projector.eval()
    if adapter is not None:
        adapter.eval()

    eval_results = {}
    with torch.no_grad():
        for ds_name, eval_data in eval_datasets.items():
            correct = 0
            total = 0
            for i in range(0, len(eval_data), cfg.eval_batch_size):
                batch_list = eval_data[i : i + cfg.eval_batch_size]
                batch_list = [get_prompt_tokens_only(dp) for dp in batch_list]
                batch_list = materialize_missing_steering_vectors(batch_list, tokenizer, model)
                batch = construct_batch(batch_list, tokenizer, device)

                sources = _gather_source_activations(batch)
                hook_fn = get_multi_token_steering_hook(
                    source_activations=sources,
                    projector=projector,
                    adapter=adapter,
                    positions=batch.positions,
                    steering_coefficient=cfg.steering_coefficient,
                    device=device,
                )
                with add_hook(submodule, hook_fn):
                    output_ids = model.generate(
                        input_ids=batch.input_ids,
                        attention_mask=batch.attention_mask,
                        **cfg.generation_kwargs,
                    )
                generated = output_ids[:, batch.input_ids.shape[1] :]
                decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
                for resp, dp in zip(decoded, batch_list, strict=True):
                    pred = resp.rstrip(".!?,;:").strip().lower()
                    gt = dp.target_output.rstrip(".!?,;:").strip().lower()
                    total += 1
                    if pred == gt:
                        correct += 1
            acc = correct / max(1, total)
            eval_results[ds_name] = {"correct": correct, "total": total, "accuracy": acc}
            print(f"  {ds_name}: {acc:.3f}  ({correct}/{total})")

    log["evals"].append({"step": step, "results": eval_results})
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    projector.train()
    if adapter is not None:
        adapter.train()
    torch.cuda.empty_cache()
    gc.collect()
