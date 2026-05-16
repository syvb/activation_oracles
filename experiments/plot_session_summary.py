"""One-graph summary of the multi-token injection experiments on this branch.

X axis: mean attention entropy over the K=8 placeholder slots (a direct
measure of "how spread vs concentrated is downstream attention" — high = the
K placeholders are treated as redundant, low = AO concentrates on a few).

Y axis: 3-eval average accuracy (classification + taboo + personaqa).

Each marker is one K=8 trained variant from this branch. The K=1 cls-only
baseline (no multi-token at all) is shown as a dashed horizontal line for
reference.

The story this chart tells:
- Every K=8 trained variant lifts the 3-eval AVG above the K=1 baseline by
  ~+1pp. Multi-token injection helps.
- Across the K=8 variants, the attention entropy varies from 1.24 to 2.00
  (out of a max of ln(8)=2.08 for fully uniform attention) — a huge range —
  yet the 3-eval AVG varies by only ~0.6pp. The "shape" of how the K
  decomposition is used doesn't matter; only that K>1.
- Frozen-W (W locked at identity, attention near-uniform) actually wins the
  average. So the lift comes from K-fold redundancy of the source signal,
  not from learning to put distinct information in different slots.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ATTN_DIR = Path("experiments/attention_results")
EVAL_DIRS = {
    "trained-W": Path("experiments/from_scratch_results"),
    "frozen-W": Path("experiments/from_scratch_results_frozen_W"),
    "entropy-penalty": Path("experiments/from_scratch_results_entropy_penalty"),
    "slot-dropout": Path("experiments/from_scratch_results_slot_dropout"),
}
BASELINE_DIR = Path("experiments/from_scratch_results_baseline")

ATTN_PATHS = {
    "trained-W": ATTN_DIR / "attention_trained_W.json",
    "frozen-W": ATTN_DIR / "attention_frozen_W.json",
    "entropy-penalty": ATTN_DIR / "attention_entropy_penalty.json",
    "slot-dropout": ATTN_DIR / "attention_slot_dropout.json",
}


def _score_cls_avg(d: Path) -> float:
    p = list(d.glob("classification_K*.json"))
    if not p:
        return 0.0
    with open(p[0]) as f:
        data = json.load(f)
    n, c = 0, 0
    for r in data["records"]:
        pred = r["ground_truth"].rstrip(".!?,;:").strip().lower()
        tgt = r["target"].rstrip(".!?,;:").strip().lower()
        n += 1
        c += int(pred == tgt)
    return c / max(1, n)


def _score_taboo_avg(d: Path) -> float:
    p = list(d.glob("taboo_K*.json"))
    if not p:
        return 0.0
    with open(p[0]) as f:
        data = json.load(f)
    per_target: dict[str, list[bool]] = {}
    for r in data["records"]:
        gt = r["ground_truth"].lower()
        per_target.setdefault(r["target_word"], [])
        for resp in r["responses"]:
            per_target[r["target_word"]].append(gt in resp.lower())
    accs = [sum(v) / len(v) for v in per_target.values() if v]
    return sum(accs) / max(1, len(accs))


def _score_personaqa(d: Path) -> float:
    p = list(d.glob("personaqa_K*.json"))
    if not p:
        return 0.0
    with open(p[0]) as f:
        data = json.load(f)
    total: list[bool] = []
    for r in data["records"]:
        gt = r["ground_truth"].lower()
        for resp in r["responses"]:
            total.append(gt in resp.lower())
    return sum(total) / max(1, len(total))


def three_eval_avg(d: Path) -> float:
    return (_score_cls_avg(d) + _score_taboo_avg(d) + _score_personaqa(d)) / 3.0


def mean_attention_entropy(p: Path) -> float:
    with open(p) as f:
        data = json.load(f)
    n_layers = len(data["results"][0]["mean_entropy_per_layer"])
    per_layer = np.zeros(n_layers)
    for ex in data["results"]:
        per_layer += np.array(ex["mean_entropy_per_layer"])
    per_layer /= len(data["results"])
    return float(per_layer.mean())


def main() -> None:
    # Compute the 3-eval averages and attention entropies
    points = {}
    for name, eval_dir in EVAL_DIRS.items():
        avg = three_eval_avg(eval_dir)
        ent = mean_attention_entropy(ATTN_PATHS[name])
        points[name] = (ent, avg)
        print(f"{name:>20}: attn entropy = {ent:.3f}, 3-eval avg = {avg*100:.2f}%")

    baseline_avg = three_eval_avg(BASELINE_DIR)
    print(f"{'K=1 baseline':>20}: 3-eval avg = {baseline_avg*100:.2f}%")

    ln_K = math.log(8)

    fig, ax = plt.subplots(figsize=(11, 6.5))

    # Plot the K=1 baseline as a horizontal dashed line spanning the chart.
    ax.axhline(baseline_avg * 100, color="dimgrey", linestyle="--", linewidth=2.0,
               label=f"K=1 cls-only baseline ({baseline_avg*100:.1f}%)")

    # Vertical reference at uniform-attention (ln K)
    ax.axvline(ln_K, color="lightgrey", linestyle=":", linewidth=1.5)
    ax.text(ln_K + 0.01, 28.7, "uniform = ln 8 ≈ 2.08",
            ha="left", va="bottom", fontsize=9, color="grey")

    # K=8 variant markers
    colors = {"trained-W": "C0", "frozen-W": "C1", "entropy-penalty": "C2", "slot-dropout": "C3"}
    markers = {"trained-W": "o", "frozen-W": "s", "entropy-penalty": "^", "slot-dropout": "D"}

    for name, (ent, avg) in points.items():
        ax.scatter([ent], [avg * 100], s=260, color=colors[name], marker=markers[name],
                   edgecolor="black", linewidth=1.2, zorder=5, label=name)
        # Label each point — outside the data so they don't collide with text box
        label_offset = {
            "trained-W": (0.03, -0.18),
            "frozen-W": (0.03, 0.10),
            "entropy-penalty": (-0.04, -0.18),
            "slot-dropout": (0.03, 0.10),
        }
        ha_for = {
            "trained-W": "left", "frozen-W": "left",
            "entropy-penalty": "right", "slot-dropout": "left",
        }
        dx, dy = label_offset.get(name, (0.02, 0.05))
        ax.annotate(name, (ent, avg * 100), xytext=(ent + dx, avg * 100 + dy),
                    fontsize=11, fontweight="bold", color=colors[name],
                    ha=ha_for.get(name, "left"))

    # Highlight the "lift over K=1" band
    ymin = baseline_avg * 100
    all_ys = [avg * 100 for (_, avg) in points.values()]
    ymax = max(all_ys)
    ax.axhspan(ymin, ymax + 0.05, alpha=0.07, color="green", zorder=0)

    ax.set_xlabel("Mean attention entropy over the K=8 placeholder slots\n(0 = AO concentrates on one slot,  ln 8 ≈ 2.08 = AO treats all 8 equally)", fontsize=11)
    ax.set_ylabel("3-eval AVG accuracy: classification + taboo + personaqa  (%)", fontsize=11)
    ax.set_title("Multi-token injection helps — but the lift comes from K-fold redundancy,\nnot from learned slot specialization",
                 fontsize=13, pad=12)

    # Annotation box, anchored bottom-right to avoid overlap with data points
    ax.text(0.98, 0.04,
            "$\\bf{What\\ this\\ chart\\ shows}$\n"
            "•  All 4 K=8 variants beat the K=1 baseline (~+1pp on 3-eval avg)\n"
            "•  Attention entropy varies hugely: 1.24 → 2.00 (full spread\n"
            "   from 'AO uses 3 of 8 slots' to 'AO treats all 8 uniformly')\n"
            "•  3-eval avg varies by only ~0.6pp — essentially flat\n"
            "•  So accuracy is downstream of K>1, not of K-decomposition shape\n"
            "•  Frozen-W (W=I, pure redundancy) actually wins the average",
            transform=ax.transAxes, fontsize=10, va="bottom", ha="right",
            bbox=dict(facecolor="white", edgecolor="lightgrey", alpha=0.97, pad=10, boxstyle="round,pad=0.5"))

    ax.set_xlim(0.95, 2.22)
    ax.set_ylim(28.5, max(all_ys) + 1.4)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=10)

    fig.tight_layout()
    out_png = ATTN_DIR / "session_summary.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
