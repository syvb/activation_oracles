"""Run paper-evals against a from-scratch K-specific multi-token AO.

Loads a trained LoRA adapter + matching projector.pt and evaluates them in
single-source-K-projection inference mode. Today: classification eval only.
The 4 open-ended evals (gender, taboo, ssc, personaqa) live in
`from_scratch_open_ended_evals.py` (separate file because they share a
different verbalizer scaffold).

Usage:
  python experiments/from_scratch_paper_evals.py \
      --lora-path checkpoints/from_scratch_K8/final \
      --projector-path checkpoints/from_scratch_K8/final/projector.pt
"""
from __future__ import annotations

import os

os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
from typing import Any

import torch
from peft import PeftModel
from tqdm import tqdm

from nl_probes.dataset_classes.classification import get_classification_datapoints
from nl_probes.multi_token.data_builder import build_multi_token_classification_data
from nl_probes.multi_token.hook import get_multi_token_steering_hook
from nl_probes.multi_token.projector import MultiTokenProjector
from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.common import layer_percent_to_layer, load_model, load_tokenizer
from nl_probes.utils.dataset_utils import (
    BatchData,
    FeatureResult,
    construct_batch,
    get_prompt_tokens_only,
    materialize_missing_steering_vectors,
)
from nl_probes.utils.steering_hooks import add_hook


@torch.no_grad()
def _multi_token_eval_batch(
    eval_batch: BatchData,
    model,
    submodule,
    projector: MultiTokenProjector,
    tokenizer,
    device: torch.device,
    steering_coefficient: float,
    generation_kwargs: dict,
) -> list[FeatureResult]:
    sources = [sv[0] for sv in eval_batch.steering_vectors]
    hook_fn = get_multi_token_steering_hook(
        source_activations=sources,
        projector=projector,
        adapter=None,
        positions=eval_batch.positions,
        steering_coefficient=steering_coefficient,
        device=device,
    )
    with add_hook(submodule, hook_fn):
        output_ids = model.generate(
            input_ids=eval_batch.input_ids,
            attention_mask=eval_batch.attention_mask,
            **generation_kwargs,
        )
    generated = output_ids[:, eval_batch.input_ids.shape[1] :]
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    decoded_prompts = tokenizer.batch_decode(eval_batch.input_ids, skip_special_tokens=False)
    return [
        FeatureResult(
            feature_idx=eval_batch.feature_indices[i],
            api_response=decoded[i],
            prompt=decoded_prompts[i],
        )
        for i in range(len(eval_batch.feature_indices))
    ]


def run_classification_eval_K(
    *,
    model,
    tokenizer,
    submodule,
    projector: MultiTokenProjector,
    K: int,
    layer_percent: int,
    classification_datasets: dict[str, dict[str, Any]],
    eval_batch_size: int,
    steering_coefficient: float,
    generation_kwargs: dict,
    device: torch.device,
    output_path: str,
) -> None:
    results: dict = {"meta": {"K": K, "layer_percent": layer_percent}, "records": []}
    act_layer = layer_percent_to_layer(model.config._name_or_path, layer_percent)

    for ds_name, dcfg in tqdm(classification_datasets.items(), desc="cls evals"):
        _, test_dps = get_classification_datapoints(
            dataset_name=ds_name,
            num_qa_per_sample=2,
            train_examples=0,
            test_examples=dcfg["num_test"],
            random_seed=42,
        )
        eval_data = build_multi_token_classification_data(
            test_dps,
            tokenizer=tokenizer,
            model=None,
            act_layer=act_layer,
            k_placeholders=K,
            activation_offset=-3,
            batch_size=eval_batch_size,
            save_acts=False,
            datapoint_type=f"multi_token_cls_{ds_name}",
        )
        for i in range(0, len(eval_data), eval_batch_size):
            batch_list = eval_data[i : i + eval_batch_size]
            batch_list = [get_prompt_tokens_only(dp) for dp in batch_list]
            batch_list = materialize_missing_steering_vectors(batch_list, tokenizer, model)
            batch = construct_batch(batch_list, tokenizer, device)
            feats = _multi_token_eval_batch(
                batch, model, submodule, projector, tokenizer, device,
                steering_coefficient, generation_kwargs,
            )
            for f, dp in zip(feats, batch_list):
                results["records"].append({
                    "dataset_id": ds_name,
                    "ground_truth": f.api_response,
                    "target": dp.target_output,
                })

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved classification results to {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora-path", type=str, required=True)
    ap.add_argument("--projector-path", type=str, required=True)
    ap.add_argument("--K", type=int, default=0,
                    help="Override K. If 0, read from projector.pt metadata.")
    ap.add_argument("--layer-percent", type=int, default=50)
    ap.add_argument("--steering-coefficient", type=float, default=1.0)
    ap.add_argument("--eval-batch-size", type=int, default=64)
    ap.add_argument("--per-ds-test", type=int, default=250)
    ap.add_argument("--output-dir", type=str, default="experiments/from_scratch_results")
    args = ap.parse_args()

    print(f"Loading Qwen3-8B + LoRA={args.lora_path}")
    dtype = torch.bfloat16
    device = torch.device("cuda")

    tokenizer = load_tokenizer("Qwen/Qwen3-8B")
    model = load_model("Qwen/Qwen3-8B", dtype)
    model = PeftModel.from_pretrained(model, args.lora_path, is_trainable=False)
    model.eval()

    print(f"Loading projector from {args.projector_path}")
    proj_state = torch.load(args.projector_path, map_location=device)
    K = args.K or proj_state["k_placeholders"]
    d_model = proj_state.get("d_model", model.config.hidden_size)
    init_strategy = proj_state.get("init_strategy", "all_identity")
    init_std = proj_state.get("init_std", 0.0)
    projector = MultiTokenProjector(d_model, K, init_strategy=init_strategy, init_std=init_std)
    projector.load_state_dict(proj_state["projector_state_dict"])
    projector = projector.to(device, dtype=torch.float32)
    projector.eval()
    print(f"K={K}, d_model={d_model}, init={init_strategy}")

    submodule = get_hf_submodule(model, 1, use_lora=True)
    os.makedirs(args.output_dir, exist_ok=True)

    classification_datasets = {
        "geometry_of_truth": {"num_test": args.per_ds_test},
        "relations": {"num_test": args.per_ds_test},
        "sst2": {"num_test": args.per_ds_test},
        "md_gender": {"num_test": args.per_ds_test},
        "snli": {"num_test": args.per_ds_test},
        "ag_news": {"num_test": args.per_ds_test},
        "ner": {"num_test": args.per_ds_test},
        "tense": {"num_test": args.per_ds_test},
        "language_identification": {"num_test": args.per_ds_test},
        "singular_plural": {"num_test": args.per_ds_test},
        "engels_headline_istrump": {"num_test": args.per_ds_test},
        "engels_headline_isobama": {"num_test": args.per_ds_test},
        "engels_headline_ischina": {"num_test": args.per_ds_test},
        "engels_hist_fig_ismale": {"num_test": args.per_ds_test},
        "engels_news_class_politics": {"num_test": args.per_ds_test},
        "engels_wikidata_isjournalist": {"num_test": args.per_ds_test},
        "engels_wikidata_isathlete": {"num_test": args.per_ds_test},
        "engels_wikidata_ispolitician": {"num_test": args.per_ds_test},
        "engels_wikidata_issinger": {"num_test": args.per_ds_test},
        "engels_wikidata_isresearcher": {"num_test": args.per_ds_test},
    }

    run_classification_eval_K(
        model=model, tokenizer=tokenizer, submodule=submodule, projector=projector,
        K=K, layer_percent=args.layer_percent,
        classification_datasets=classification_datasets,
        eval_batch_size=args.eval_batch_size,
        steering_coefficient=args.steering_coefficient,
        generation_kwargs={"do_sample": False, "temperature": 0.0, "max_new_tokens": 10},
        device=device,
        output_path=f"{args.output_dir}/classification_K{K}.json",
    )


if __name__ == "__main__":
    main()
