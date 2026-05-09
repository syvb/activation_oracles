"""Attention-pattern analysis: do downstream tokens attend differently to the
K=8 placeholder positions, or do they treat them as equivalent?

If different non-placeholder tokens attend more to different placeholder
slots, that's evidence the K decomposition carries distinct information
and the AO is using it. If attention over the K placeholders is roughly
uniform per query token, the K=8 setup is just a K-fold redundancy trick.

For each test input (one per task) and each model checkpoint:
  1. Run forward with `output_attentions=True` (forces eager attention).
  2. For every layer ≥ 2 (the injection happens at the end of layer 1),
     for every head, extract attention from each non-placeholder query
     token to each of the K=8 placeholder positions.
  3. Compute the entropy of the over-K-placeholders attention distribution
     for each (layer, head, query_token).
  4. Save a JSON with raw attention and aggregated statistics.

Usage:
  python experiments/attention_analysis.py \
      --lora-path syvb/from-scratch-K8-AO-Qwen3-8B \
      --projector-path /path/to/projector.pt \
      --out attention_trained_W.json
"""
from __future__ import annotations

import argparse
import json
import math
import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from nl_probes.dataset_classes.classification import ClassificationDatapoint, get_classification_datapoints
from nl_probes.multi_token.data_builder import build_multi_token_classification_data
from nl_probes.multi_token.hook import get_multi_token_steering_hook
from nl_probes.multi_token.projector import MultiTokenProjector
from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.common import layer_percent_to_layer, load_tokenizer
from nl_probes.utils.dataset_utils import (
    construct_batch,
    get_prompt_tokens_only,
    materialize_missing_steering_vectors,
)
from nl_probes.utils.steering_hooks import add_hook


def load_model_eager(model_name: str, dtype: torch.dtype):
    """Same as load_model but with attn_implementation='eager' so
    output_attentions actually returns weights (FA2/SDPA both reject it).
    """
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        attn_implementation="eager",
        torch_dtype=dtype,
    )


def attention_entropy_over_K(
    attn_LH: torch.Tensor,
    placeholder_positions: list[int],
    eps: float = 1e-12,
) -> torch.Tensor:
    """`attn_LH` shape (num_layers, num_heads, L, L). Returns
    `(num_layers, num_heads, num_query_tokens)` entropy over the K placeholder
    positions, considering only query tokens AFTER the last placeholder.

    A high entropy = ln(K) means uniform attention over placeholders → no
    learned differentiation. Low entropy = attention focused on a single
    placeholder slot → strong differentiation.
    """
    K = len(placeholder_positions)
    last_ph = max(placeholder_positions)
    L = attn_LH.shape[-1]
    # Slice query tokens to those AFTER the last placeholder (so the AO has
    # already "ingested" the K activations).
    query_idx = list(range(last_ph + 1, L))
    if not query_idx:
        return torch.zeros(attn_LH.shape[0], attn_LH.shape[1], 0)

    q_t = torch.tensor(query_idx, device=attn_LH.device)
    p_t = torch.tensor(placeholder_positions, device=attn_LH.device)

    # (num_layers, num_heads, num_q, K)
    sub = attn_LH[..., q_t, :][..., :, p_t]

    # Renormalize over the K dimension (since attention also goes elsewhere,
    # but we want the distribution OVER the K placeholders specifically).
    sub_norm = sub / (sub.sum(dim=-1, keepdim=True).clamp(min=eps))
    # Entropy in nats over the K dim
    entropy = -(sub_norm * (sub_norm.clamp(min=eps).log())).sum(dim=-1)
    return entropy


def attention_per_query_to_K(
    attn_LH: torch.Tensor,
    placeholder_positions: list[int],
) -> torch.Tensor:
    """Returns `(num_layers, num_heads, num_query, K)` of the renormalized
    attention from each post-placeholder query token to each placeholder.
    """
    last_ph = max(placeholder_positions)
    L = attn_LH.shape[-1]
    query_idx = list(range(last_ph + 1, L))
    q_t = torch.tensor(query_idx, device=attn_LH.device)
    p_t = torch.tensor(placeholder_positions, device=attn_LH.device)
    sub = attn_LH[..., q_t, :][..., :, p_t]
    sub_norm = sub / sub.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    return sub_norm


@torch.no_grad()
def run_with_attention(
    model,
    tokenizer,
    submodule,
    projector: MultiTokenProjector,
    examples,
    device: torch.device,
    steering_coefficient: float = 1.0,
):
    """For each example (one TrainingDataPoint), run a single-batch forward
    pass with the multi-token hook + output_attentions=True. Returns a list of
    dicts with per-example attention statistics.
    """
    out = []
    for dp in examples:
        # batch of size 1
        batch_list = [get_prompt_tokens_only(dp)]
        batch_list = materialize_missing_steering_vectors(batch_list, tokenizer, model)
        batch = construct_batch(batch_list, tokenizer, device)

        sources = [sv[0] for sv in batch.steering_vectors]
        hook_fn = get_multi_token_steering_hook(
            source_activations=sources, projector=projector, adapter=None,
            positions=batch.positions, steering_coefficient=steering_coefficient,
            device=device,
        )
        with add_hook(submodule, hook_fn):
            outputs = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                output_attentions=True,
                return_dict=True,
            )

        # outputs.attentions: tuple of (B, H, L, L) per layer
        attn = torch.stack(outputs.attentions, dim=0).squeeze(1)  # (num_layers, H, L, L)

        ph_positions = batch.positions[0]  # K placeholder positions for batch[0]
        per_qK = attention_per_query_to_K(attn, ph_positions)  # (num_layers, H, num_q, K)
        entropy = attention_entropy_over_K(attn, ph_positions)  # (num_layers, H, num_q)
        ln_K = math.log(len(ph_positions))

        out.append({
            "datapoint_type": dp.datapoint_type,
            "ds_label": dp.ds_label,
            "input_token_count": int(batch.input_ids.shape[1]),
            "K": len(ph_positions),
            "ln_K": ln_K,
            "placeholder_positions": ph_positions,
            "n_query_tokens": int(per_qK.shape[2]),
            # mean entropy aggregated over heads + query tokens, per layer
            "mean_entropy_per_layer": entropy.mean(dim=(1, 2)).cpu().tolist(),
            # per-layer per-head mean entropy (over query tokens)
            "mean_entropy_per_layer_head": entropy.mean(dim=2).cpu().tolist(),
            # per-layer mean attention to each K placeholder (avg over heads + q)
            "mean_attn_per_K_per_layer": per_qK.mean(dim=(1, 2)).cpu().tolist(),
            # full per-q attention for the LAST layer head 0 (for plotting)
            "last_layer_head0_per_q_K": per_qK[-1, 0].cpu().tolist(),
        })
        del attn, outputs, per_qK, entropy

    return out


def _build_examples(tokenizer, model_name, K: int, seed: int = 42):
    """A small handful of examples from different tasks. Use lazy mode so
    activations get materialized via the trained AO at runtime."""
    examples = []
    act_layer = layer_percent_to_layer(model_name, 50)

    # 4 classification examples from different datasets
    cls_pairs = [
        ("geometry_of_truth", 1),
        ("sst2", 1),
        ("language_identification", 1),
        ("singular_plural", 1),
    ]
    for ds_name, n in cls_pairs:
        _, test_dps = get_classification_datapoints(
            dataset_name=ds_name, num_qa_per_sample=1,
            train_examples=0, test_examples=n, random_seed=seed,
        )
        ex = build_multi_token_classification_data(
            test_dps, tokenizer=tokenizer, model=None, act_layer=act_layer,
            k_placeholders=K, activation_offset=-3, batch_size=1,
            save_acts=False, datapoint_type=f"cls_{ds_name}",
        )
        examples.extend(ex[:1])

    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora-path", type=str, required=True,
                    help="HF repo id or local path of the LoRA adapter")
    ap.add_argument("--projector-path", type=str, required=True,
                    help="Path to projector.pt (local) or 'hf:<repo>:<filename>' to download")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--model-name", type=str, default="Qwen/Qwen3-8B")
    args = ap.parse_args()

    dtype = torch.bfloat16
    device = torch.device("cuda")

    print(f"Loading {args.model_name} (eager attention)...")
    tokenizer = load_tokenizer(args.model_name)
    model = load_model_eager(args.model_name, dtype)
    model = PeftModel.from_pretrained(model, args.lora_path, is_trainable=False)
    model.eval()

    # Resolve projector
    proj_path = args.projector_path
    if proj_path.startswith("hf:"):
        from huggingface_hub import hf_hub_download
        _, repo, fname = proj_path.split(":", 2)
        proj_path = hf_hub_download(repo_id=repo, filename=fname)

    proj_state = torch.load(proj_path, map_location=device)
    K = proj_state["k_placeholders"]
    d_model = proj_state.get("d_model", model.config.hidden_size)
    init_strategy = proj_state.get("init_strategy", "all_identity")
    init_std = proj_state.get("init_std", 0.0)
    projector = MultiTokenProjector(d_model, K, init_strategy=init_strategy, init_std=init_std)
    projector.load_state_dict(proj_state["projector_state_dict"])
    projector = projector.to(device, dtype=torch.float32).eval()
    print(f"Loaded K={K}, init={init_strategy}")

    submodule = get_hf_submodule(model, 1, use_lora=True)

    print("Building example inputs...")
    examples = _build_examples(tokenizer, args.model_name, K)
    print(f"Got {len(examples)} examples")

    print("Running forward passes with output_attentions...")
    results = run_with_attention(model, tokenizer, submodule, projector, examples, device)

    out_data = {
        "model_name": args.model_name,
        "lora_path": args.lora_path,
        "projector_path": args.projector_path,
        "K": K,
        "init_strategy": init_strategy,
        "results": results,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
