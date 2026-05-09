"""Train W on LatentQA (where the AO is *not* at ceiling) and check whether
the resulting projector generalizes to the held-out classification eval.

Setup:
  - K=4 multi-token, all_identity init
  - Frozen Qwen3-8B + frozen AO LoRA (cls_only)
  - Train W only (no adapter, no LoRA training)
  - LR small (3e-5) so W moves slowly from identity
  - Eval at step 0 (sanity), then every N optim steps
  - Eval surface: full 20-dataset classification eval (matches phase0_with_kdecomp)
"""
import argparse
import gc
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from peft import PeftModel

from nl_probes.dataset_classes.classification import get_classification_datapoints
from nl_probes.multi_token.data_builder import build_multi_token_classification_data
from nl_probes.multi_token.latentqa_data_builder import build_multi_token_latentqa_data
from nl_probes.multi_token.train import MultiTokenTrainConfig, train
from nl_probes.utils.common import load_model, load_tokenizer, layer_percent_to_layer
from nl_probes.utils.dataset_utils import TrainingDataPoint


CLS_DATASETS = [
    "geometry_of_truth", "relations", "sst2", "md_gender", "snli", "ner", "tense", "ag_news",
    "language_identification", "singular_plural",
    "engels_headline_istrump", "engels_headline_isobama", "engels_headline_ischina",
    "engels_hist_fig_ismale", "engels_news_class_politics",
    "engels_wikidata_isjournalist", "engels_wikidata_isathlete",
    "engels_wikidata_ispolitician", "engels_wikidata_issinger", "engels_wikidata_isresearcher",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n-train-latentqa", type=int, default=20000)
    ap.add_argument("--n-test-cls-per-ds", type=int, default=250)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--init-strategy", type=str, default="all_identity")
    ap.add_argument("--steering-coefficient", type=float, default=1.0)
    ap.add_argument("--run-name", type=str, default="phase2_latentqa_train")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--use-adapter", action="store_true")
    args = ap.parse_args()

    cfg = MultiTokenTrainConfig(
        k_placeholders=args.k,
        train_batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.epochs,
        run_name=args.run_name,
        use_adapter=args.use_adapter,
        steering_coefficient=args.steering_coefficient,
        eval_every=args.eval_every,
        log_every=20,
        train_ao_lora=False,
        projector_init_strategy=args.init_strategy,
    )
    act_layer = layer_percent_to_layer(cfg.model_name, cfg.layer_percent)
    print(f"act layer: {act_layer}")
    print(f"config: K={cfg.k_placeholders}, init={cfg.projector_init_strategy}, lr={cfg.lr}, "
          f"adapter={cfg.use_adapter}, train_lora={cfg.train_ao_lora}")

    print("Loading target model (no LoRA)...")
    tokenizer = load_tokenizer(cfg.model_name)
    target_model = load_model(cfg.model_name, torch.bfloat16)
    target_model.eval()

    print("Building LatentQA training data...")
    train_td = build_multi_token_latentqa_data(
        tokenizer=tokenizer, model=target_model,
        act_layer=act_layer, k_placeholders=cfg.k_placeholders,
        n_examples=args.n_train_latentqa,
        activation_offset=-3,
        batch_size=8,
        seed=42,
        skip_first=0,
    )

    print("\nBuilding classification eval data (20 datasets) ...")
    test_td_by_ds: dict[str, list[TrainingDataPoint]] = {}
    for ds_name in CLS_DATASETS:
        _, test_split = get_classification_datapoints(
            dataset_name=ds_name, num_qa_per_sample=2,
            train_examples=0, test_examples=args.n_test_cls_per_ds, random_seed=42,
        )
        test_td_by_ds[ds_name] = build_multi_token_classification_data(
            test_split, tokenizer=tokenizer, model=target_model,
            act_layer=act_layer, k_placeholders=cfg.k_placeholders,
            activation_offset=-3, batch_size=16, save_acts=True,
        )

    del target_model
    torch.cuda.empty_cache()
    gc.collect()

    log_path = f"logs/{cfg.run_name}.json"
    os.makedirs("logs", exist_ok=True)

    train(
        cfg=cfg, training_data=train_td, eval_datasets=test_td_by_ds,
        tokenizer=tokenizer, device=torch.device("cuda"), dtype=torch.bfloat16,
        log_path=log_path,
    )


if __name__ == "__main__":
    main()
