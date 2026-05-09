"""Plot/summarize the attention-pattern analysis JSON files.

Two views per (model, example):
  1. Mean entropy over the K=8 placeholders, by AO layer. Compares
     trained-W vs frozen-W. Lower entropy = K placeholders treated more
     differently.
  2. Heatmap of attention from each query token to each placeholder, for
     the last layer / head 0 (or whatever the JSON exported).

Usage:
  python experiments/plot_attention_analysis.py
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ATTN_DIR = Path("experiments/attention_results")


def load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def main():
    if not ATTN_DIR.exists():
        raise SystemExit(f"{ATTN_DIR} not found — run attention_analysis.py first")

    paths = {
        "trained-W": ATTN_DIR / "attention_trained_W.json",
        "frozen-W": ATTN_DIR / "attention_frozen_W.json",
        "entropy-penalty": ATTN_DIR / "attention_entropy_penalty.json",
        "slot-dropout": ATTN_DIR / "attention_slot_dropout.json",
    }
    available = {k: p for k, p in paths.items() if p.exists()}
    if "trained-W" not in available or "frozen-W" not in available:
        raise SystemExit("Need at least trained-W and frozen-W JSONs")

    runs = {k: load(p) for k, p in available.items()}
    trained = runs["trained-W"]
    frozen = runs["frozen-W"]

    K = trained["K"]
    ln_K = math.log(K)
    # mean across examples, per layer
    n_layers = len(trained["results"][0]["mean_entropy_per_layer"])

    print(f"K={K}, ln(K)={ln_K:.3f}  (uniform over K = {ln_K:.3f}, single slot = 0)")
    print()

    # --- Aggregate: mean entropy per layer across examples, per run ---
    per_layer_by_run = {}
    for name, run in runs.items():
        per_layer = np.zeros(n_layers)
        for ex in run["results"]:
            per_layer += np.array(ex["mean_entropy_per_layer"])
        per_layer /= len(run["results"])
        per_layer_by_run[name] = per_layer

    # Print summary table
    header = f"{'Layer':>5}"
    for name in per_layer_by_run:
        header += f"  {name:>12}"
    print(header)
    for L_idx in range(n_layers):
        row = f"{L_idx:>5}"
        for name, arr in per_layer_by_run.items():
            row += f"  {arr[L_idx]:>12.4f}"
        print(row)

    print()
    for name, arr in per_layer_by_run.items():
        print(f"{name:>16} mean entropy: {arr.mean():.4f}  (gap to ln(K) = {ln_K - arr.mean():.4f})")

    # --- Plot: entropy vs layer for all available runs ---
    fig, ax = plt.subplots(figsize=(10, 5))
    markers = {"trained-W": "o", "frozen-W": "s", "entropy-penalty": "^", "slot-dropout": "D"}
    for name, arr in per_layer_by_run.items():
        ax.plot(range(n_layers), arr, label=name, marker=markers.get(name, "o"))
    ax.axhline(ln_K, color="grey", linestyle="--", label=f"uniform = ln(K) = {ln_K:.3f}")
    ax.set_xlabel("AO layer")
    ax.set_ylabel("Mean attention entropy over K=8 placeholders\n(over heads, post-placeholder query tokens, examples)")
    ax.set_title("Attention entropy over K=8 placeholders, by AO layer")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    out_png = ATTN_DIR / "entropy_per_layer.png"
    fig.savefig(out_png, dpi=120)
    print(f"\nSaved {out_png}")
    plt.close(fig)

    # --- Per-task heatmaps for the trained-W last layer head 0 ---
    n_examples = len(trained["results"])
    fig, axes = plt.subplots(2, n_examples, figsize=(4 * n_examples, 6))
    for i, (ex_t, ex_f) in enumerate(zip(trained["results"], frozen["results"])):
        for row, (ex, label) in enumerate([(ex_t, "trained-W"), (ex_f, "frozen-W")]):
            attn = np.array(ex["last_layer_head0_per_q_K"])  # (num_q, K)
            ax = axes[row, i] if n_examples > 1 else axes[row]
            im = ax.imshow(attn, aspect="auto", cmap="viridis", vmin=0, vmax=max(0.3, attn.max()))
            ax.set_title(f"{ex['datapoint_type']} ({label})", fontsize=9)
            ax.set_xlabel("placeholder slot k")
            if i == 0:
                ax.set_ylabel("query token (post-placeholder)")
            ax.set_xticks(range(ex["K"]))
    fig.suptitle("Last-layer head 0 attention: query tokens → K=8 placeholder slots")
    fig.tight_layout()
    out_png = ATTN_DIR / "per_query_heatmaps_last_layer.png"
    fig.savefig(out_png, dpi=120)
    print(f"Saved {out_png}")
    plt.close(fig)

    # --- Per-K mean attention bar plot, averaged over layers/heads, per run ---
    per_K_by_run = {}
    for name, run in runs.items():
        per_K = np.zeros(K)
        n_layer_total = 0
        for ex in run["results"]:
            a = np.array(ex["mean_attn_per_K_per_layer"])  # (num_layers, K)
            per_K += a.sum(axis=0)
            n_layer_total += a.shape[0]
        per_K /= n_layer_total
        per_K_by_run[name] = per_K

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(K)
    n_runs = len(per_K_by_run)
    width = 0.8 / n_runs
    colors = {"trained-W": "C0", "frozen-W": "C1", "entropy-penalty": "C2", "slot-dropout": "C3"}
    for i, (name, arr) in enumerate(per_K_by_run.items()):
        offset = (i - (n_runs - 1) / 2) * width
        ax.bar(x + offset, arr, width, label=name, color=colors.get(name))
    ax.axhline(1.0/K, color="grey", linestyle="--", label=f"uniform = 1/K = {1/K:.3f}")
    ax.set_xlabel("placeholder slot k")
    ax.set_ylabel("mean renormalized attention\n(over layers, heads, query tokens, examples)")
    ax.set_title("Mean attention to each placeholder slot, all runs")
    ax.set_xticks(x)
    ax.legend()
    fig.tight_layout()
    out_png = ATTN_DIR / "per_K_mean_attention.png"
    fig.savefig(out_png, dpi=120)
    print(f"Saved {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
