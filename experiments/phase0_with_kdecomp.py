"""Phase 0-style evaluation across the full 20-dataset classification mixture
with multiple (K, init_strategy) combinations.

Confirms the all_identity init OOD gain on a broader eval surface than the
2-OOD-dataset eval_inits run.
"""
import argparse
import gc
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from peft import PeftModel

from nl_probes.dataset_classes.classification import (
    get_classification_datapoints,
)
from nl_probes.multi_token.data_builder import build_multi_token_classification_data
from nl_probes.multi_token.projector import MultiTokenProjector
from nl_probes.multi_token.hook import get_multi_token_steering_hook
from nl_probes.multi_token.train import _gather_source_activations
from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.common import load_model, load_tokenizer, layer_percent_to_layer
from nl_probes.utils.dataset_utils import (
    construct_batch,
    materialize_missing_steering_vectors,
    get_prompt_tokens_only,
)
from nl_probes.utils.steering_hooks import add_hook


MODEL_NAME = "Qwen/Qwen3-8B"
LORA = "adamkarvonen/checkpoints_cls_only_addition_Qwen3-8B"
LAYER_PERCENT = 50
DTYPE = torch.bfloat16

# Match Phase 0 exactly.
DATASETS = [
    # IID-ish (these are in the SFT classification training mix)
    "geometry_of_truth", "relations", "sst2", "md_gender", "snli", "ner", "tense", "ag_news",
    # OOD
    "language_identification", "singular_plural",
    "engels_headline_istrump", "engels_headline_isobama", "engels_headline_ischina",
    "engels_hist_fig_ismale", "engels_news_class_politics",
    "engels_wikidata_isjournalist", "engels_wikidata_isathlete",
    "engels_wikidata_ispolitician", "engels_wikidata_issinger", "engels_wikidata_isresearcher",
]
IID = ["geometry_of_truth", "relations", "sst2", "md_gender", "snli", "ner", "tense"]
# Match plot_classification_eval.py groupings: ag_news is OOD there
OOD_PRIMARY = ["ag_news", "language_identification", "singular_plural"]
OOD_ENGELS = [d for d in DATASETS if d.startswith("engels_")]


def eval_one_config(model, tokenizer, submodule, eval_data, k, init_strategy, init_std):
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
                    adapter=None,
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
    # Pairs of (K, init_strategy)
    args = ap.parse_args()

    configs = [
        (1, "identity_plus_noise"),
        (4, "all_identity"),
        (8, "all_identity"),
        (16, "all_identity"),
    ]

    tokenizer = load_tokenizer(MODEL_NAME)

    print("Loading target model (no LoRA) for activation extraction...")
    target_model = load_model(MODEL_NAME, DTYPE)
    target_model.eval()
    act_layer = layer_percent_to_layer(MODEL_NAME, LAYER_PERCENT)

    test_dps_by_ds = {}
    print(f"Gathering test data across {len(DATASETS)} datasets...")
    for ds in DATASETS:
        _, test = get_classification_datapoints(ds, num_qa_per_sample=2,
                                               train_examples=0, test_examples=args.n_test_per_ds, random_seed=42)
        test_dps_by_ds[ds] = test

    Ks = sorted({k for k, _ in configs})
    test_td_by_k = {}
    for k in Ks:
        print(f"\n=== Building data for K={k} ===")
        test_td_by_k[k] = {}
        for ds, dps in test_dps_by_ds.items():
            test_td_by_k[k][ds] = build_multi_token_classification_data(
                dps, tokenizer=tokenizer, model=target_model,
                act_layer=act_layer, k_placeholders=k,
                activation_offset=-3, batch_size=16, save_acts=True,
            )

    del target_model
    torch.cuda.empty_cache()
    gc.collect()

    print("\nLoading AO with LoRA...")
    model = load_model(MODEL_NAME, DTYPE)
    model = PeftModel.from_pretrained(model, LORA, is_trainable=False)
    model.eval()
    submodule = get_hf_submodule(model, 1, use_lora=True)

    print("\n=== Results across full 20-dataset eval ===\n")
    print(f"{'K':>3} {'init':25} {'IID(7)':>7} {'OOD-3':>7} {'OOD-engels(10)':>16} {'OOD-all(13)':>12}")

    out = {}
    for k, init_strategy in configs:
        init_std = 0.02 if "noise" in init_strategy else 0.0
        results = eval_one_config(model, tokenizer, submodule, test_td_by_k[k], k, init_strategy, init_std)

        iid = sum(results[d] for d in IID) / len(IID)
        ood_primary = sum(results[d] for d in OOD_PRIMARY) / len(OOD_PRIMARY)
        ood_engels = sum(results[d] for d in OOD_ENGELS) / len(OOD_ENGELS)
        ood_all = sum(results[d] for d in OOD_PRIMARY + OOD_ENGELS) / (len(OOD_PRIMARY) + len(OOD_ENGELS))

        print(f"{k:>3} {init_strategy:25} {iid*100:>6.1f}% {ood_primary*100:>6.1f}% {ood_engels*100:>15.1f}% {ood_all*100:>11.1f}%")
        out[(k, init_strategy)] = {
            "iid": iid, "ood_primary": ood_primary, "ood_engels": ood_engels, "ood_all": ood_all,
            "per_ds": results,
        }

    import json
    os.makedirs("logs", exist_ok=True)
    with open("logs/phase0_with_kdecomp.json", "w") as f:
        json.dump({str(k): v for k, v in out.items()}, f, indent=2)
    print("\nSaved to logs/phase0_with_kdecomp.json")


if __name__ == "__main__":
    main()
