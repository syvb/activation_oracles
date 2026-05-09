"""Eval-only sweep across W projector init strategies (no training).

Confirms how disruptive each init is to the AO before any optimization. The
expected ordering is: K=1 baseline > all_identity > identity_plus_noise (the
plan default), with the "all_identity_plus_noise" variant somewhere in between.
"""
import argparse
import gc
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from peft import PeftModel

from nl_probes.dataset_classes.classification import (
    ClassificationDatapoint,
    get_classification_datapoints,
)
from nl_probes.multi_token.data_builder import build_multi_token_classification_data
from nl_probes.multi_token.projector import MultiTokenProjector, InjectionAdapter
from nl_probes.multi_token.hook import get_multi_token_steering_hook
from nl_probes.multi_token.train import _gather_source_activations
from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.common import load_model, load_tokenizer, layer_percent_to_layer
from nl_probes.utils.dataset_utils import (
    TrainingDataPoint,
    construct_batch,
    materialize_missing_steering_vectors,
    get_prompt_tokens_only,
)
from nl_probes.utils.steering_hooks import add_hook


MODEL_NAME = "Qwen/Qwen3-8B"
LORA = "adamkarvonen/checkpoints_cls_only_addition_Qwen3-8B"
LAYER_PERCENT = 50
DTYPE = torch.bfloat16
TEST_SUBDATASETS = [
    "geometry_of_truth", "relations", "sst2", "md_gender", "snli", "ner", "tense",
    "ag_news", "language_identification", "singular_plural",
]


def eval_one_config(model, tokenizer, submodule, eval_data: dict, k: int, init_strategy: str, init_std: float):
    d_model = model.config.hidden_size
    projector = MultiTokenProjector(d_model, k, init_std=init_std, init_strategy=init_strategy).to("cuda", dtype=torch.float32)
    projector.eval()

    results = {}
    with torch.no_grad():
        for ds, data in eval_data.items():
            correct, total = 0, 0
            for i in range(0, len(data), 32):
                batch_list = data[i:i+32]
                batch_list = [get_prompt_tokens_only(dp) for dp in batch_list]
                batch_list = materialize_missing_steering_vectors(batch_list, tokenizer, model)
                batch = construct_batch(batch_list, tokenizer, torch.device("cuda"))
                sources = _gather_source_activations(batch)
                hook_fn = get_multi_token_steering_hook(
                    source_activations=sources,
                    projector=projector,
                    adapter=None,  # eval-only; no adapter
                    positions=batch.positions,
                    steering_coefficient=1.0,
                    device=torch.device("cuda"),
                )
                with add_hook(submodule, hook_fn):
                    out_ids = model.generate(
                        input_ids=batch.input_ids,
                        attention_mask=batch.attention_mask,
                        do_sample=False,
                        max_new_tokens=10,
                    )
                gen = out_ids[:, batch.input_ids.shape[1]:]
                decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
                for resp, dp in zip(decoded, batch_list, strict=True):
                    pred = resp.rstrip(".!?,;:").strip().lower()
                    gt = dp.target_output.rstrip(".!?,;:").strip().lower()
                    total += 1
                    if pred == gt:
                        correct += 1
            results[ds] = correct / max(1, total)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-test-per-ds", type=int, default=250)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    args = ap.parse_args()

    tokenizer = load_tokenizer(MODEL_NAME)

    # Build data: requires the BASE model first (no LoRA) to extract source activations
    print("Loading target model (no LoRA) to extract activations...")
    target_model = load_model(MODEL_NAME, DTYPE)
    target_model.eval()

    act_layer = layer_percent_to_layer(MODEL_NAME, LAYER_PERCENT)
    print(f"act layer: {act_layer}")

    # Get test datapoints
    test_dps_by_ds = {}
    for ds in TEST_SUBDATASETS:
        _, test = get_classification_datapoints(ds, num_qa_per_sample=2, train_examples=0, test_examples=args.n_test_per_ds, random_seed=42)
        test_dps_by_ds[ds] = test

    # Build TrainingDataPoints for each K (data is K-specific because the prompt has K placeholders)
    test_td_by_k_by_ds = {}
    for k in args.ks:
        print(f"\n=== Building data for K={k} ===")
        test_td_by_k_by_ds[k] = {}
        for ds, dps in test_dps_by_ds.items():
            test_td_by_k_by_ds[k][ds] = build_multi_token_classification_data(
                dps,
                tokenizer=tokenizer,
                model=target_model,
                act_layer=act_layer,
                k_placeholders=k,
                activation_offset=-3,
                batch_size=16,
                save_acts=True,
            )

    del target_model
    torch.cuda.empty_cache()
    gc.collect()

    # Now load AO with LoRA
    print("\nLoading AO with LoRA...")
    model = load_model(MODEL_NAME, DTYPE)
    model = PeftModel.from_pretrained(model, LORA, is_trainable=False)
    model.eval()
    submodule = get_hf_submodule(model, 1, use_lora=True)

    # Eval each (K, init_strategy)
    INITS = [
        ("identity_plus_noise", 0.02),
        ("all_identity", 0.0),
        ("all_identity_plus_noise", 0.02),
    ]

    print("\n=== Results ===\n")
    print(f"{'K':>3} {'init':25} {'IID%':>6} {'OOD%':>6} {'avg%':>6}")

    IID = ['geometry_of_truth','relations','sst2','md_gender','snli','ner','tense']
    OOD = ['ag_news','language_identification','singular_plural']

    all_results = {}
    for k in args.ks:
        for init_strategy, init_std in INITS:
            # Skip K=1 redundant configs
            if k == 1 and init_strategy != "identity_plus_noise":
                continue
            results = eval_one_config(model, tokenizer, submodule, test_td_by_k_by_ds[k], k, init_strategy, init_std)
            iid_avg = sum(results[d] for d in IID if d in results) / len([d for d in IID if d in results])
            ood_avg = sum(results[d] for d in OOD if d in results) / len([d for d in OOD if d in results])
            avg = sum(results.values()) / len(results)
            print(f"{k:>3} {init_strategy:25} {iid_avg*100:>6.1f} {ood_avg*100:>6.1f} {avg*100:>6.1f}")
            all_results[(k, init_strategy)] = {"iid": iid_avg, "ood": ood_avg, "avg": avg, "per_ds": results}

    import json
    os.makedirs("logs", exist_ok=True)
    with open("logs/eval_inits.json", "w") as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)
    print(f"\nSaved to logs/eval_inits.json")


if __name__ == "__main__":
    main()
