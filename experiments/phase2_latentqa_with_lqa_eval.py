"""Train W on LatentQA + track held-out LatentQA *loss* and held-out
classification eval. Distinguishes "training memorizes / does nothing useful"
from "training learns LatentQA but the direction doesn't transfer to
classification."
"""
import argparse
import gc
import json
import math
import os
import random

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
from peft import PeftModel
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from nl_probes.dataset_classes.classification import get_classification_datapoints
from nl_probes.multi_token.data_builder import build_multi_token_classification_data
from nl_probes.multi_token.latentqa_data_builder import build_multi_token_latentqa_data
from nl_probes.multi_token.projector import MultiTokenProjector
from nl_probes.multi_token.hook import get_multi_token_steering_hook
from nl_probes.multi_token.train import _gather_source_activations
from nl_probes.utils.activation_utils import get_hf_submodule
from nl_probes.utils.common import load_model, load_tokenizer, layer_percent_to_layer, set_seed
from nl_probes.utils.dataset_utils import (
    BatchData,
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

CLS_DATASETS = [
    "geometry_of_truth", "relations", "sst2", "md_gender", "snli", "ner", "tense", "ag_news",
    "language_identification", "singular_plural",
]
IID = ['geometry_of_truth','relations','sst2','md_gender','snli','ner','tense','ag_news']
OOD = ['language_identification','singular_plural']


def held_out_latentqa_loss(model, tokenizer, submodule, projector, device, eval_data, batch_size=8):
    projector.eval()
    losses = []
    n = 0
    with torch.no_grad():
        for i in range(0, len(eval_data), batch_size):
            batch_list = eval_data[i:i+batch_size]
            batch_list = materialize_missing_steering_vectors(batch_list, tokenizer, model)
            batch = construct_batch(batch_list, tokenizer, device)
            sources = _gather_source_activations(batch)
            hook_fn = get_multi_token_steering_hook(
                source_activations=sources, projector=projector, adapter=None,
                positions=batch.positions, steering_coefficient=1.0, device=device,
            )
            with add_hook(submodule, hook_fn):
                out = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, labels=batch.labels)
            losses.append(out.loss.item() * batch.input_ids.size(0))
            n += batch.input_ids.size(0)
    projector.train()
    return sum(losses) / max(1, n)


def cls_eval(model, tokenizer, submodule, projector, device, test_td_by_ds, batch_size=32):
    projector.eval()
    out = {}
    with torch.no_grad():
        for ds, data in test_td_by_ds.items():
            correct, total = 0, 0
            for i in range(0, len(data), batch_size):
                batch_list = data[i:i+batch_size]
                batch_list = [get_prompt_tokens_only(dp) for dp in batch_list]
                batch_list = materialize_missing_steering_vectors(batch_list, tokenizer, model)
                batch = construct_batch(batch_list, tokenizer, device)
                sources = _gather_source_activations(batch)
                hook_fn = get_multi_token_steering_hook(
                    source_activations=sources, projector=projector, adapter=None,
                    positions=batch.positions, steering_coefficient=1.0, device=device,
                )
                with add_hook(submodule, hook_fn):
                    out_ids = model.generate(
                        input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                        do_sample=False, max_new_tokens=10,
                    )
                gen = out_ids[:, batch.input_ids.shape[1]:]
                decoded = tokenizer.batch_decode(gen, skip_special_tokens=True)
                for resp, dp in zip(decoded, batch_list, strict=True):
                    pred = resp.rstrip(".!?,;:").strip().lower()
                    gt = dp.target_output.rstrip(".!?,;:").strip().lower()
                    total += 1
                    if pred == gt: correct += 1
            out[ds] = correct / max(1, total)
    iid = sum(out[d] for d in IID) / len(IID)
    ood = sum(out[d] for d in OOD) / len(OOD)
    projector.train()
    return iid, ood, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-eval-lqa", type=int, default=500)
    ap.add_argument("--n-test-cls-per-ds", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--init", type=str, default="all_identity")
    ap.add_argument("--run-name", type=str, default="lqa_train_with_lqa_eval")
    args = ap.parse_args()

    set_seed(42)
    tokenizer = load_tokenizer(MODEL_NAME)

    print("Loading target model (no LoRA)...")
    target = load_model(MODEL_NAME, DTYPE)
    target.eval()
    act_layer = layer_percent_to_layer(MODEL_NAME, LAYER_PERCENT)

    print(f"Building LatentQA train ({args.n_train})...")
    train_td = build_multi_token_latentqa_data(
        tokenizer=tokenizer, model=target, act_layer=act_layer,
        k_placeholders=args.k, n_examples=args.n_train,
        activation_offset=-3, batch_size=8, seed=42, skip_first=0,
    )
    print(f"Building LatentQA held-out eval ({args.n_eval_lqa})...")
    lqa_eval_td = build_multi_token_latentqa_data(
        tokenizer=tokenizer, model=target, act_layer=act_layer,
        k_placeholders=args.k, n_examples=args.n_eval_lqa,
        activation_offset=-3, batch_size=8, seed=42,
        skip_first=args.n_train,  # disjoint from train
    )

    print("Building classification eval...")
    test_td_by_ds = {}
    for ds in CLS_DATASETS:
        _, test_split = get_classification_datapoints(
            dataset_name=ds, num_qa_per_sample=2,
            train_examples=0, test_examples=args.n_test_cls_per_ds, random_seed=42,
        )
        test_td_by_ds[ds] = build_multi_token_classification_data(
            test_split, tokenizer=tokenizer, model=target,
            act_layer=act_layer, k_placeholders=args.k,
            activation_offset=-3, batch_size=16, save_acts=True,
        )

    del target
    torch.cuda.empty_cache()
    gc.collect()

    print("\nLoading AO with LoRA (frozen)...")
    model = load_model(MODEL_NAME, DTYPE)
    model = PeftModel.from_pretrained(model, LORA, is_trainable=False)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.enable_input_require_grads()
    submodule = get_hf_submodule(model, 1, use_lora=True)

    d_model = model.config.hidden_size
    projector = MultiTokenProjector(d_model, args.k, init_std=0.0, init_strategy=args.init).to("cuda", dtype=torch.float32)

    optimizer = torch.optim.AdamW(projector.parameters(), lr=args.lr)

    n_batches = len(train_td) // args.batch_size
    total_steps = n_batches
    warmup = max(1, int(total_steps * 0.05))

    log = {"args": vars(args), "steps": [], "evals": []}
    log_path = f"logs/{args.run_name}.json"
    os.makedirs("logs", exist_ok=True)

    # Step-0 eval
    print("\n=== Eval at step 0 ===")
    lqa_loss = held_out_latentqa_loss(model, tokenizer, submodule, projector, "cuda", lqa_eval_td)
    iid, ood, _ = cls_eval(model, tokenizer, submodule, projector, "cuda", test_td_by_ds)
    print(f"step 0  LQA_loss={lqa_loss:.4f}  cls_IID={iid*100:.1f}  cls_OOD={ood*100:.1f}")
    log["evals"].append({"step": 0, "lqa_loss": lqa_loss, "iid": iid, "ood": ood})
    with open(log_path, "w") as f: json.dump(log, f, indent=2)

    random.shuffle(train_td)
    projector.train()

    step = 0
    accum_loss = 0.0
    pbar = tqdm(range(0, len(train_td), args.batch_size), desc="train")
    for start in pbar:
        batch_list = train_td[start:start + args.batch_size]
        batch_list = materialize_missing_steering_vectors(batch_list, tokenizer, model)
        batch = construct_batch(batch_list, tokenizer, "cuda")
        sources = _gather_source_activations(batch)
        hook_fn = get_multi_token_steering_hook(
            source_activations=sources, projector=projector, adapter=None,
            positions=batch.positions, steering_coefficient=1.0, device="cuda",
        )
        with add_hook(submodule, hook_fn):
            out = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, labels=batch.labels)
        loss = out.loss
        loss.backward()
        gn = clip_grad_norm_(projector.parameters(), 1.0)
        # cosine warmup
        if step < warmup:
            lr_now = args.lr * (step + 1) / warmup
        else:
            progress = (step - warmup) / max(1, total_steps - warmup)
            lr_now = 0.5 * args.lr * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        accum_loss += loss.item()
        if step % 20 == 0:
            log["steps"].append({"step": step, "loss": loss.item(), "lr": lr_now, "grad_norm": float(gn)})

        step += 1
        if step > 0 and step % args.eval_every == 0:
            print(f"\n=== Eval at step {step} ===")
            lqa_loss = held_out_latentqa_loss(model, tokenizer, submodule, projector, "cuda", lqa_eval_td)
            iid, ood, _ = cls_eval(model, tokenizer, submodule, projector, "cuda", test_td_by_ds)
            print(f"step {step}  LQA_loss={lqa_loss:.4f}  cls_IID={iid*100:.1f}  cls_OOD={ood*100:.1f}")
            log["evals"].append({"step": step, "lqa_loss": lqa_loss, "iid": iid, "ood": ood})
            with open(log_path, "w") as f: json.dump(log, f, indent=2)

    # Final eval
    print(f"\n=== Final eval at step {step} ===")
    lqa_loss = held_out_latentqa_loss(model, tokenizer, submodule, projector, "cuda", lqa_eval_td)
    iid, ood, _ = cls_eval(model, tokenizer, submodule, projector, "cuda", test_td_by_ds)
    print(f"step {step}  LQA_loss={lqa_loss:.4f}  cls_IID={iid*100:.1f}  cls_OOD={ood*100:.1f}")
    log["evals"].append({"step": step, "lqa_loss": lqa_loss, "iid": iid, "ood": ood})
    with open(log_path, "w") as f: json.dump(log, f, indent=2)


if __name__ == "__main__":
    main()
