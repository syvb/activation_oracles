"""From-scratch SFT for an Activation Oracle that's K-specific by design.

Differences from `nl_probes/sft.py`:
  - Wraps the AO together with a MultiTokenProjector in a single nn.Module so
    DDP synchronizes both sets of trainable params (LoRA + W).
  - Uses `get_multi_token_steering_hook` instead of the single-token hook —
    this projects ONE source residual into K vectors via W and injects them
    at the K placeholder positions.
  - Optimizer has two parameter groups: LoRA at `cfg.lr` and W at
    `cfg.projector_lr` so we can keep the W updates hot while LoRA moves slowly.
  - Checkpoint saves both the LoRA adapter and `projector.pt` so eval-time
    code can reconstruct the exact W used during training.

Most of the dataset-loading, length-bucketing, eval-loop, and HF-push logic is
re-imported from sft.py so behavior matches the original training pipeline.
"""
from __future__ import annotations

import gc
import os
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

from nl_probes.configs.sft_config import SelfInterpTrainingConfig
from nl_probes.dataset_classes.act_dataset_manager import ActDatasetLoader
from nl_probes.multi_token.hook import get_multi_token_steering_hook
from nl_probes.multi_token.projector import MultiTokenProjector
from nl_probes.sft import (
    build_datasets,
    push_lora_to_hf,
    _ensure_datasets_exist,
)
from nl_probes.utils.activation_utils import get_hf_submodule, get_text_only_lora_targets
from nl_probes.utils.common import load_model, load_tokenizer, set_seed
from nl_probes.utils.dataset_utils import (
    BatchData,
    TrainingDataPoint,
    construct_batch,
    materialize_missing_steering_vectors,
)
from nl_probes.utils.steering_hooks import add_hook


@dataclass
class MultiTokenSftConfig(SelfInterpTrainingConfig):
    """Extends the base SFT config with single-source-K-projection knobs."""

    k_placeholders: int = 8
    projector_init_strategy: str = "all_identity"
    projector_init_std: float = 0.0
    projector_lr: float = 3e-4
    # If set, save the projector state to this path at every save_step.
    projector_filename: str = "projector.pt"


class _AOWithProjector(nn.Module):
    """Tiny container so DDP wraps the AO + projector together.

    We never call `forward()` directly (we go through `self.ao(...)` from
    `train_features_batch_multi_token`), but DDP needs a single nn.Module so
    gradient sync covers both. The container also makes `parameters()` return
    everything so AdamW gets both groups.
    """

    def __init__(self, ao: nn.Module, projector: MultiTokenProjector):
        super().__init__()
        self.ao = ao
        self.projector = projector

    def forward(self, *args, **kwargs):
        return self.ao(*args, **kwargs)


def _gather_source_activations(batch: BatchData) -> list[torch.Tensor]:
    """Each example's steering_vectors is shape (K, d) with K identical rows.
    Pull row 0 as the single source.
    """
    sources = []
    for sv in batch.steering_vectors:
        assert sv is not None, "steering_vectors must be materialized before training"
        sources.append(sv[0])
    return sources


def train_features_batch_multi_token(
    cfg: MultiTokenSftConfig,
    training_batch: BatchData,
    ddp_module: nn.Module,
    projector: MultiTokenProjector,
    submodule: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    sources = _gather_source_activations(training_batch)
    hook_fn = get_multi_token_steering_hook(
        source_activations=sources,
        projector=projector,
        adapter=None,
        positions=training_batch.positions,
        steering_coefficient=cfg.steering_coefficient,
        device=device,
    )
    tokenized_input = {
        "input_ids": training_batch.input_ids,
        "attention_mask": training_batch.attention_mask,
    }
    with add_hook(submodule, hook_fn):
        loss = ddp_module(**tokenized_input, labels=training_batch.labels).loss
    return loss


@torch.no_grad()
def held_out_loss_per_task(
    cfg: MultiTokenSftConfig,
    held_out_by_task: dict[str, list[TrainingDataPoint]],
    ddp_module: nn.Module,
    inner_model: nn.Module,
    projector: MultiTokenProjector,
    submodule: nn.Module,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    """Mean cross-entropy loss on held-out examples per task. Uses the
    multi-token hook so the projector is in the path (matches training).
    """
    was_training = ddp_module.training
    ddp_module.eval()
    out = {}
    for task_name, examples in held_out_by_task.items():
        if not examples:
            continue
        losses = []
        bs = max(1, cfg.eval_batch_size // 2)
        for i in range(0, len(examples), bs):
            batch_list = examples[i : i + bs]
            batch_list = materialize_missing_steering_vectors(batch_list, tokenizer, inner_model)
            batch = construct_batch(batch_list, tokenizer, device)
            sources = _gather_source_activations(batch)
            hook_fn = get_multi_token_steering_hook(
                source_activations=sources,
                projector=projector,
                adapter=None,
                positions=batch.positions,
                steering_coefficient=cfg.steering_coefficient,
                device=device,
            )
            with add_hook(submodule, hook_fn):
                loss = ddp_module(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    labels=batch.labels,
                ).loss
            losses.append(loss.item())
        if losses:
            out[f"eval_loss/{task_name}"] = sum(losses) / len(losses)
    if was_training:
        ddp_module.train()
    torch.cuda.empty_cache()
    gc.collect()
    return out


def _w_per_slot_diff_from_identity(projector: MultiTokenProjector) -> dict[str, float]:
    """||W_k - I||_F per slot. Useful to track whether the AO is using the K
    decomposition (slots diverge from identity) or just the redundancy
    effect (all slots stay near identity)."""
    out = {}
    with torch.no_grad():
        d = projector.d_model
        K = projector.k
        W = projector.linear.weight  # (K*d, d)
        eye = torch.eye(d, device=W.device, dtype=W.dtype)
        for slot in range(K):
            W_slot = W[slot * d : (slot + 1) * d]
            diff = (W_slot - eye).norm().item()
            out[f"projector/||W_{slot}-I||_F"] = diff
    return out


def oom_preflight_check_multi_token(
    cfg: MultiTokenSftConfig,
    training_data: list[TrainingDataPoint],
    inner_model: nn.Module,
    projector: MultiTokenProjector,
    submodule: nn.Module,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Mirror sft.py's preflight: bypass DDP, run forward+backward on the inner
    model so memory peaks are observed without DDP's grad-sync orchestration.
    """
    longest_prompt = max(training_data, key=lambda x: len(x.input_ids))
    long_prompts = [longest_prompt] * cfg.train_batch_size
    long_prompts = materialize_missing_steering_vectors(long_prompts, tokenizer, inner_model)
    largest_possible_batch = construct_batch(long_prompts, tokenizer, device)

    dummy_optimizer = torch.optim.AdamW(
        list(inner_model.parameters()) + list(projector.parameters()), lr=0.0
    )

    for _ in tqdm(range(3), desc="OOM preflight check (multi-token)"):
        loss = train_features_batch_multi_token(
            cfg, largest_possible_batch, inner_model, projector, submodule, device, dtype
        )
        loss.backward()
        dummy_optimizer.step()
        dummy_optimizer.zero_grad()

    del dummy_optimizer
    torch.cuda.empty_cache()
    gc.collect()
    print("OOM preflight check (multi-token) complete")


def train_model_multi_token(
    cfg: MultiTokenSftConfig,
    training_data: list[TrainingDataPoint],
    held_out_by_task: dict[str, list[TrainingDataPoint]],
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    dtype: torch.dtype,
    model_kwargs: dict[str, Any],
    verbose: bool = False,
):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    model_kwargs = {**model_kwargs, "device_map": {"": f"cuda:{local_rank}"}}

    set_seed(cfg.seed)
    model = load_model(cfg.model_name, dtype, **model_kwargs)
    model.enable_input_require_grads()

    if cfg.gradient_checkpointing:
        model.use_cache = False
        model.gradient_checkpointing_enable()

    if cfg.use_lora and cfg.load_lora_path is None:
        target_modules = cfg.lora_target_modules
        vlm_targets = get_text_only_lora_targets(cfg.model_name)
        if vlm_targets and target_modules == "all-linear":
            print(f"VLM detected ({cfg.model_name}): excluding vision tower from LoRA")
            target_modules = vlm_targets
        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config, autocast_adapter_dtype=True)
    elif cfg.load_lora_path is not None:
        load_lora_path = Path(cfg.load_lora_path)
        assert load_lora_path.exists()
        model = PeftModel.from_pretrained(model, load_lora_path, is_trainable=True, autocast_adapter_dtype=True)
    else:
        raise ValueError("From-scratch multi-token training requires use_lora=True")

    model.print_trainable_parameters()

    # Build the projector in fp32. all_identity is the recommended starting
    # point per RESULTS.md; the K=4 step-0 OOD gain came from this init.
    d_model = model.config.hidden_size
    projector = MultiTokenProjector(
        d_model,
        cfg.k_placeholders,
        init_std=cfg.projector_init_std,
        init_strategy=cfg.projector_init_strategy,
    ).to(device=device, dtype=torch.float32)

    submodule = get_hf_submodule(model, cfg.hook_onto_layer, use_lora=True)

    wrapped = _AOWithProjector(model, projector).to(device)
    torch.cuda.set_device(local_rank)
    # find_unused_parameters=True is required because the projector's params
    # are only touched inside a forward hook on a submodule of `self.ao`, not
    # via `_AOWithProjector.forward()` itself. Without it, DDP throws
    # "Expected to have finished reduction in the prior iteration" because
    # its used-param tracker can't see the hook-driven path. The perf cost is
    # small relative to the LoRA backward.
    ddp_module: nn.Module = torch.nn.parallel.DistributedDataParallel(
        wrapped, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True
    )

    ddp_module.train()

    oom_preflight_check_multi_token(
        cfg, training_data, model, projector, submodule, tokenizer, device, dtype
    )

    set_seed(cfg.seed)

    # Two parameter groups: LoRA params on cfg.lr, projector params on cfg.projector_lr.
    lora_params = [p for n, p in ddp_module.named_parameters() if "lora_" in n and p.requires_grad]
    projector_params = list(projector.parameters())
    n_lora = sum(p.numel() for p in lora_params)
    n_proj = sum(p.numel() for p in projector_params)
    if rank == 0:
        print(f"Trainable LoRA params: {n_lora:,}")
        print(f"Trainable projector params: {n_proj:,}")

    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": cfg.lr, "name": "lora"},
            {"params": projector_params, "lr": cfg.projector_lr, "name": "projector"},
        ]
    )

    global_step_size = cfg.train_batch_size * world_size
    effective_steps = (len(training_data) // global_step_size) * global_step_size
    if effective_steps != len(training_data):
        if rank == 0:
            print(f"Trimming training_data from {len(training_data)} to {effective_steps} for equal DDP steps")
        training_data = training_data[:effective_steps]

    if rank == 0:
        tokens_per_epoch_est = sum(len(dp.input_ids) for dp in training_data)
        total_training_tokens_est = tokens_per_epoch_est * cfg.num_epochs
        num_examples_pre_shard = len(training_data)

    training_data = training_data[rank::world_size]
    num_batches_per_epoch = len(training_data) // cfg.train_batch_size
    batches_per_epoch = (num_batches_per_epoch // cfg.gradient_accumulation_steps) * cfg.gradient_accumulation_steps
    trimmed_examples = batches_per_epoch * cfg.train_batch_size
    if trimmed_examples != len(training_data) and rank == 0:
        print(
            f"Trimming per-rank training_data from {len(training_data)} to {trimmed_examples} "
            "to align with gradient_accumulation_steps"
        )
    training_data = training_data[:trimmed_examples]

    steps_per_epoch = batches_per_epoch // cfg.gradient_accumulation_steps
    assert steps_per_epoch > 0, "No optimizer steps will be run"
    total_training_steps = steps_per_epoch * cfg.num_epochs
    warmup_steps = int(total_training_steps * 0.1)

    # Single shared linear schedule. Both param groups scale by the same factor;
    # base_lr per group is preserved by the optimizer.
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps
    )

    global_step = 0

    if rank == 0:
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name, config=asdict(cfg))
        wandb.summary["train/tokens_per_epoch_est"] = tokens_per_epoch_est
        wandb.summary["train/total_tokens_est"] = total_training_tokens_est
        wandb.summary["train/num_examples_pre_shard"] = num_examples_pre_shard
        wandb.summary["train/k_placeholders"] = cfg.k_placeholders
        wandb.summary["train/projector_init_strategy"] = cfg.projector_init_strategy

    for epoch in range(cfg.num_epochs):
        accumulated_loss = 0.0
        optimizer.zero_grad()
        for step_idx, start in enumerate(
            tqdm(
                range(0, len(training_data), cfg.train_batch_size),
                desc=f"Training epoch {epoch + 1}",
                disable=rank != 0,
            )
        ):
            t_batch_list = training_data[start : start + cfg.train_batch_size]
            t_batch_list = materialize_missing_steering_vectors(t_batch_list, tokenizer, model)
            t_batch = construct_batch(t_batch_list, tokenizer, device)

            loss = train_features_batch_multi_token(
                cfg, t_batch, ddp_module, projector, submodule, device, dtype
            )
            loss = loss / cfg.gradient_accumulation_steps
            loss.backward()
            accumulated_loss += loss.item()

            is_update_step = (step_idx + 1) % cfg.gradient_accumulation_steps == 0
            if is_update_step:
                clip_grad_norm_(ddp_module.parameters(), cfg.max_grad_norm)

                # W-only grad norm before the optimizer step.
                w_grad_norm = (
                    projector.linear.weight.grad.detach().norm().item()
                    if projector.linear.weight.grad is not None
                    else 0.0
                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if rank == 0:
                    log_dict = {
                        "train/loss": accumulated_loss,
                        "train/learning_rate_lora": optimizer.param_groups[0]["lr"],
                        "train/learning_rate_projector": optimizer.param_groups[1]["lr"],
                        "train/w_grad_norm": w_grad_norm,
                    }
                    log_dict.update(_w_per_slot_diff_from_identity(projector))
                    wandb.log(log_dict, step=global_step)
                    if verbose and global_step % 50 == 0:
                        print(
                            f"Step {global_step} loss={accumulated_loss:.4f} "
                            f"w_grad_norm={w_grad_norm:.3f}"
                        )

                if global_step % cfg.eval_steps == 0 and (cfg.eval_on_start or global_step > 0):
                    if rank == 0 and held_out_by_task:
                        loss_dict = held_out_loss_per_task(
                            cfg, held_out_by_task, ddp_module, model, projector,
                            submodule, tokenizer, device, dtype,
                        )
                        if loss_dict:
                            wandb.log(loss_dict, step=global_step)
                            for k, v in loss_dict.items():
                                print(f"  {k}: {v:.4f}")
                    dist.barrier()

                if global_step % cfg.save_steps == 0 and global_step > 0:
                    if rank == 0:
                        save_step_dir = f"{cfg.save_dir}/step_{global_step}"
                        os.makedirs(save_step_dir, exist_ok=True)
                        model.save_pretrained(save_step_dir)
                        torch.save(
                            {
                                "projector_state_dict": projector.state_dict(),
                                "k_placeholders": cfg.k_placeholders,
                                "d_model": d_model,
                                "init_strategy": cfg.projector_init_strategy,
                                "init_std": cfg.projector_init_std,
                                "config": asdict(cfg),
                            },
                            f"{save_step_dir}/{cfg.projector_filename}",
                        )
                    dist.barrier()

                global_step += 1
                accumulated_loss = 0.0

    print("Training complete.")

    if rank == 0:
        final_dir = f"{cfg.save_dir}/final"
        os.makedirs(final_dir, exist_ok=True)
        model.save_pretrained(final_dir)
        torch.save(
            {
                "projector_state_dict": projector.state_dict(),
                "k_placeholders": cfg.k_placeholders,
                "d_model": d_model,
                "init_strategy": cfg.projector_init_strategy,
                "init_std": cfg.projector_init_std,
                "config": asdict(cfg),
            },
            f"{final_dir}/{cfg.projector_filename}",
        )

        if held_out_by_task:
            final_loss_dict = held_out_loss_per_task(
                cfg, held_out_by_task, ddp_module, model, projector,
                submodule, tokenizer, device, dtype,
            )
            if final_loss_dict:
                wandb.log(final_loss_dict, step=global_step)
                for k, v in final_loss_dict.items():
                    print(f"  {k}: {v:.4f}")
        wandb.finish()

        if cfg.hf_push_to_hub and cfg.hf_repo_id:
            print(f"Pushing LoRA + projector to HF Hub: {cfg.hf_repo_id}")
            push_lora_to_hf(
                model=model,
                tokenizer=tokenizer,
                repo_id=cfg.hf_repo_id,
                private=cfg.hf_private_repo,
                commit_message=f"Multi-token K={cfg.k_placeholders} AO - {cfg.wandb_run_name} - final",
            )
            try:
                from huggingface_hub import upload_file
                upload_file(
                    path_or_fileobj=f"{final_dir}/{cfg.projector_filename}",
                    path_in_repo=cfg.projector_filename,
                    repo_id=cfg.hf_repo_id,
                    commit_message=f"Add projector weights for K={cfg.k_placeholders}",
                )
                print(f"Pushed projector.pt to {cfg.hf_repo_id}")
            except Exception as e:
                print(f"Warning: failed to upload projector.pt: {e}")
    dist.barrier()
