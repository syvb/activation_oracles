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

    trained_path = ATTN_DIR / "attention_trained_W.json"
    frozen_path = ATTN_DIR / "attention_frozen_W.json"
    if not trained_path.exists() or not frozen_path.exists():
        raise SystemExit("expected attention_trained_W.json and attention_frozen_W.json in ATTN_DIR")

    trained = load(trained_path)
    frozen = load(frozen_path)

    K = trained["K"]
    ln_K = math.log(K)
    # mean across examples, per layer
    n_layers = len(trained["results"][0]["mean_entropy_per_layer"])

    print(f"K={K}, ln(K)={ln_K:.3f}  (uniform over K = {ln_K:.3f}, single slot = 0)")
    print()

    # --- Aggregate: mean entropy per layer across examples ---
    trained_per_layer = np.zeros(n_layers)
    frozen_per_layer = np.zeros(n_layers)
    for ex_t, ex_f in zip(trained["results"], frozen["results"]):
        trained_per_layer += np.array(ex_t["mean_entropy_per_layer"])
        frozen_per_layer += np.array(ex_f["mean_entropy_per_layer"])
    trained_per_layer /= len(trained["results"])
    frozen_per_layer /= len(frozen["results"])

    print(f"{'Layer':>5} {'trained-W H':>12} {'frozen-W H':>12} {'Δ (T−F)':>10}")
    for L_idx in range(n_layers):
        print(f"{L_idx:>5} {trained_per_layer[L_idx]:>12.4f} {frozen_per_layer[L_idx]:>12.4f} {trained_per_layer[L_idx]-frozen_per_layer[L_idx]:>+10.4f}")

    print()
    print(f"trained-W mean entropy across all layers: {trained_per_layer.mean():.4f}  (gap to ln(K) = {ln_K - trained_per_layer.mean():.4f})")
    print(f"frozen-W  mean entropy across all layers: {frozen_per_layer.mean():.4f}  (gap to ln(K) = {ln_K - frozen_per_layer.mean():.4f})")

    # --- Plot: entropy vs layer ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(n_layers), trained_per_layer, label="trained-W", marker="o")
    ax.plot(range(n_layers), frozen_per_layer, label="frozen-W (W=I)", marker="s")
    ax.axhline(ln_K, color="grey", linestyle="--", label=f"uniform = ln(K) = {ln_K:.3f}")
    ax.set_xlabel("AO layer")
    ax.set_ylabel("Mean attention entropy over K=8 placeholders\n(over heads, post-placeholder query tokens, examples)")
    ax.set_title("Attention to K=8 placeholders is closer to uniform when W is frozen at identity")
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

    # --- Per-K mean attention bar plot, averaged over layers/heads ---
    trained_per_K = np.zeros(K)
    frozen_per_K = np.zeros(K)
    n_layer_total = 0
    for ex_t, ex_f in zip(trained["results"], frozen["results"]):
        a_t = np.array(ex_t["mean_attn_per_K_per_layer"])  # (num_layers, K)
        a_f = np.array(ex_f["mean_attn_per_K_per_layer"])
        trained_per_K += a_t.sum(axis=0)
        frozen_per_K += a_f.sum(axis=0)
        n_layer_total += a_t.shape[0]
    trained_per_K /= n_layer_total
    frozen_per_K /= n_layer_total

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(K)
    width = 0.4
    ax.bar(x - width/2, trained_per_K, width, label="trained-W")
    ax.bar(x + width/2, frozen_per_K, width, label="frozen-W")
    ax.axhline(1.0/K, color="grey", linestyle="--", label=f"uniform = 1/K = {1/K:.3f}")
    ax.set_xlabel("placeholder slot k")
    ax.set_ylabel("mean renormalized attention\n(over layers, heads, query tokens, examples)")
    ax.set_title("Mean attention to each placeholder slot")
    ax.set_xticks(x)
    ax.legend()
    fig.tight_layout()
    out_png = ATTN_DIR / "per_K_mean_attention.png"
    fig.savefig(out_png, dpi=120)
    print(f"Saved {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
