"""Two-variant comparison: trained-W ("trained with learned decomposition")
vs frozen-W ("trained with repeated inputs"). For each metric, show how much
each variant lifts over the K=1 cls-only baseline (in pp).

This is more readable than absolute accuracy because the metrics span
~5% (taboo) to ~89% (cls IID). Δ-vs-baseline collapses the dynamic range
into a consistent ±20pp window per metric, so the per-metric trade-off
between the two variants is directly comparable.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

IID_PAPER_7 = ["geometry_of_truth", "relations", "sst2", "md_gender", "snli", "ner", "tense"]
OOD_PAPER_3 = ["ag_news", "language_identification", "singular_plural"]
OOD_ENGELS = [
    "engels_headline_istrump", "engels_headline_isobama", "engels_headline_ischina",
    "engels_hist_fig_ismale", "engels_news_class_politics",
    "engels_wikidata_isjournalist", "engels_wikidata_isathlete",
    "engels_wikidata_ispolitician", "engels_wikidata_issinger", "engels_wikidata_isresearcher",
]
OOD_ALL_13 = OOD_PAPER_3 + OOD_ENGELS


def score_cls(d: Path) -> dict[str, float]:
    p = list(d.glob("classification_K*.json"))
    if not p:
        return {}
    with open(p[0]) as f:
        data = json.load(f)
    by_ds = defaultdict(list)
    for r in data["records"]:
        pred = r["ground_truth"].rstrip(".!?,;:").strip().lower()
        tgt = r["target"].rstrip(".!?,;:").strip().lower()
        by_ds[r["dataset_id"]].append(pred == tgt)
    return {ds: sum(v) / len(v) for ds, v in by_ds.items() if v}


def score_taboo(d: Path) -> float:
    p = list(d.glob("taboo_K*.json"))
    if not p:
        return 0.0
    with open(p[0]) as f:
        data = json.load(f)
    per = defaultdict(list)
    for r in data["records"]:
        gt = r["ground_truth"].lower()
        for resp in r["responses"]:
            per[r["target_word"]].append(gt in resp.lower())
    accs = [sum(v) / len(v) for v in per.values() if v]
    return sum(accs) / max(1, len(accs))


def score_personaqa(d: Path) -> float:
    p = list(d.glob("personaqa_K*.json"))
    if not p:
        return 0.0
    with open(p[0]) as f:
        data = json.load(f)
    total = []
    for r in data["records"]:
        gt = r["ground_truth"].lower()
        for resp in r["responses"]:
            total.append(gt in resp.lower())
    return sum(total) / max(1, len(total))


def metrics_for(d: Path) -> dict[str, float]:
    cls = score_cls(d)
    def avg(group):
        accs = [cls[k] for k in group if k in cls]
        return sum(accs) / max(1, len(accs))
    return {
        "Cls IID (7 datasets)": avg(IID_PAPER_7),
        "Cls OOD-3 (paper)": avg(OOD_PAPER_3),
        "Cls OOD-all (13 datasets)": avg(OOD_ALL_13),
        "Taboo (20-target avg)": score_taboo(d),
        "PersonaQA (overall)": score_personaqa(d),
        "3-eval AVG": (avg(IID_PAPER_7 + OOD_ALL_13) + score_taboo(d) + score_personaqa(d)) / 3.0,
    }


def main() -> None:
    base = metrics_for(Path("experiments/from_scratch_results_baseline"))
    frozen = metrics_for(Path("experiments/from_scratch_results_frozen_W"))
    trained = metrics_for(Path("experiments/from_scratch_results"))

    metrics = list(base.keys())
    deltas_frozen = [(frozen[m] - base[m]) * 100 for m in metrics]
    deltas_trained = [(trained[m] - base[m]) * 100 for m in metrics]
    base_pct = [base[m] * 100 for m in metrics]

    # Print summary
    print(f"{'Metric':<30} {'K=1 base':>10} {'frozen-W':>14} {'trained-W':>16}")
    print("-" * 72)
    for i, m in enumerate(metrics):
        print(f"{m:<30} {base_pct[i]:>9.1f}%  {base_pct[i]+deltas_frozen[i]:>6.1f}% ({deltas_frozen[i]:+.1f}pp)  {base_pct[i]+deltas_trained[i]:>6.1f}% ({deltas_trained[i]:+.1f}pp)")

    # Plot
    fig, ax = plt.subplots(figsize=(11, 6.5))

    y = np.arange(len(metrics))[::-1]  # reverse so first metric is on top
    height = 0.36

    repeated_color = "#E67E22"   # orange
    learned_color = "#2E86C1"    # blue

    ax.barh(y + height/2, deltas_frozen, height,
            label="Trained with repeated inputs  (frozen W = I, K=8)",
            color=repeated_color, edgecolor="black", linewidth=0.6, zorder=3)
    ax.barh(y - height/2, deltas_trained, height,
            label="Trained with learned decomposition  (trained W, K=8)",
            color=learned_color, edgecolor="black", linewidth=0.6, zorder=3)

    # Zero line = K=1 baseline
    ax.axvline(0, color="black", linewidth=1.2, zorder=4)

    # Value labels on each bar
    for i, (df, dt) in enumerate(zip(deltas_frozen, deltas_trained)):
        yi = y[i]
        # frozen
        xpos_f = df + (0.25 if df >= 0 else -0.25)
        ha_f = "left" if df >= 0 else "right"
        ax.text(xpos_f, yi + height/2, f"{df:+.1f}pp", va="center", ha=ha_f, fontsize=9.5,
                color=repeated_color, fontweight="bold")
        # trained
        xpos_t = dt + (0.25 if dt >= 0 else -0.25)
        ha_t = "left" if dt >= 0 else "right"
        ax.text(xpos_t, yi - height/2, f"{dt:+.1f}pp", va="center", ha=ha_t, fontsize=9.5,
                color=learned_color, fontweight="bold")

    # Y-axis labels with the K=1 baseline value appended
    metric_labels_with_base = [
        f"{m}\n($\\it{{K{{=}}1\\ baseline:\\ {base_pct[i]:.1f}\\%}}$)"
        for i, m in enumerate(metrics)
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(metric_labels_with_base, fontsize=10)

    ax.set_xlabel("Δ accuracy vs K=1 cls-only baseline  (percentage points)", fontsize=11)
    ax.set_title("Learned decomposition vs repeated inputs:\nthe lift is essentially the same — slight win for repeated inputs on average",
                 fontsize=13, pad=10)

    ax.grid(True, axis="x", linestyle=":", alpha=0.4, zorder=1)
    ax.legend(loc="lower right", framealpha=0.97, fontsize=10)

    # Tighten x-axis range
    all_deltas = deltas_frozen + deltas_trained
    xmin = min(all_deltas + [0]) - 3
    xmax = max(all_deltas + [0]) + 3
    ax.set_xlim(xmin, xmax)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out_png = Path("experiments/attention_results/decomp_vs_repeated.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out_png}")
    plt.close(fig)


if __name__ == "__main__":
    main()
