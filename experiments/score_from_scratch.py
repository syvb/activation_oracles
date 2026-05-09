"""Score classification + taboo + personaqa results JSON files for the
from-scratch K=8 AO and the K=1 baseline. Print headline numbers and deltas.

Usage:
  python experiments/score_from_scratch.py \
      --k8-dir experiments/from_scratch_results \
      --baseline-dir experiments/from_scratch_results_baseline
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


# Classification dataset groupings (matches RESULTS.md)
IID_PAPER_7 = ["geometry_of_truth", "relations", "sst2", "md_gender", "snli", "ner", "tense"]
OOD_PAPER_3 = ["ag_news", "language_identification", "singular_plural"]
OOD_ENGELS = [
    "engels_headline_istrump", "engels_headline_isobama", "engels_headline_ischina",
    "engels_hist_fig_ismale", "engels_news_class_politics",
    "engels_wikidata_isjournalist", "engels_wikidata_isathlete",
    "engels_wikidata_ispolitician", "engels_wikidata_issinger", "engels_wikidata_isresearcher",
]
OOD_ALL_13 = OOD_PAPER_3 + OOD_ENGELS


def score_classification(path: Path) -> dict[str, float]:
    """Returns {dataset_id: accuracy}."""
    with open(path) as f:
        d = json.load(f)
    by_ds: dict[str, list[bool]] = defaultdict(list)
    for r in d["records"]:
        pred = r["ground_truth"].rstrip(".!?,;:").strip().lower()
        target = r["target"].rstrip(".!?,;:").strip().lower()
        by_ds[r["dataset_id"]].append(pred == target)
    return {ds: sum(v) / len(v) for ds, v in by_ds.items() if v}


def score_taboo(path: Path) -> dict[str, float]:
    """Returns {target_word: accuracy} where accuracy = fraction of responses
    that contain the target word substring (case-insensitive)."""
    with open(path) as f:
        d = json.load(f)
    by_target: dict[str, list[bool]] = defaultdict(list)
    for r in d["records"]:
        gt = r["ground_truth"].lower()
        for resp in r["responses"]:
            by_target[r["target_word"]].append(gt in resp.lower())
    return {t: sum(v) / len(v) for t, v in by_target.items() if v}


def score_personaqa(path: Path) -> dict[str, float]:
    """Returns {prompt_kind: accuracy}. Each prompt asks about a persona attribute;
    accuracy = fraction of responses that contain the ground-truth value."""
    with open(path) as f:
        d = json.load(f)
    by_prompt: dict[str, list[bool]] = defaultdict(list)
    overall: list[bool] = []
    for r in d["records"]:
        gt = r["ground_truth"].lower()
        prompt = r["verbalizer_prompt"]
        for resp in r["responses"]:
            ok = gt in resp.lower()
            by_prompt[prompt].append(ok)
            overall.append(ok)
    out = {p: sum(v) / len(v) for p, v in by_prompt.items() if v}
    out["__overall__"] = sum(overall) / len(overall) if overall else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k8-dir", type=str, default="experiments/from_scratch_results")
    ap.add_argument("--baseline-dir", type=str, default="experiments/from_scratch_results_baseline")
    ap.add_argument("--label-k8", type=str, default="K=8 from-scratch")
    ap.add_argument("--label-base", type=str, default="K=1 cls-only AO")
    args = ap.parse_args()

    k8 = Path(args.k8_dir)
    base = Path(args.baseline_dir)

    print(f"\n{'='*72}")
    print(f"Comparison: {args.label_k8}  vs  {args.label_base}")
    print(f"{'='*72}\n")

    # --- Classification ---
    print("CLASSIFICATION (250 examples × 2 QAs per dataset)")
    print("-" * 72)
    k8_cls = score_classification(k8 / "classification_K8.json")
    base_cls_path = base / "classification_K1.json"
    if not base_cls_path.exists():
        # fall back to the K8 file naming if user labelled differently
        base_cls_path = base / "classification_K8.json"
    base_cls = score_classification(base_cls_path) if base_cls_path.exists() else {}

    def avg(d: dict[str, float], group: list[str]) -> float:
        accs = [d[k] for k in group if k in d]
        return sum(accs) / len(accs) if accs else 0.0

    print(f"{'Group':<25} {args.label_base:>15} {args.label_k8:>15} {'Δ':>10}")
    for label, group in [
        ("IID(7)", IID_PAPER_7),
        ("OOD-3 (paper)", OOD_PAPER_3),
        ("OOD-engels(10)", OOD_ENGELS),
        ("OOD-all(13)", OOD_ALL_13),
    ]:
        b = avg(base_cls, group)
        k = avg(k8_cls, group)
        print(f"{label:<25} {b*100:>14.1f}% {k*100:>14.1f}% {(k-b)*100:>+9.1f}pp")
    cls_avg_k8 = avg(k8_cls, IID_PAPER_7 + OOD_ALL_13)
    cls_avg_base = avg(base_cls, IID_PAPER_7 + OOD_ALL_13)

    # --- Taboo ---
    print()
    print("TABOO (20 target words, 3 prompts × ~30 contexts × 5 generations each)")
    print("-" * 72)
    k8_tab = score_taboo(k8 / "taboo_K8.json")
    base_tab_path = base / "taboo_K1.json"
    if not base_tab_path.exists():
        base_tab_path = base / "taboo_K8.json"
    base_tab = score_taboo(base_tab_path) if base_tab_path.exists() else {}

    print(f"{'Target word':<20} {args.label_base:>15} {args.label_k8:>15} {'Δ':>10}")
    targets = sorted(set(list(k8_tab.keys()) + list(base_tab.keys())))
    for tw in targets:
        b = base_tab.get(tw, 0.0)
        k = k8_tab.get(tw, 0.0)
        print(f"{tw:<20} {b*100:>14.1f}% {k*100:>14.1f}% {(k-b)*100:>+9.1f}pp")
    tab_avg_k8 = sum(k8_tab.values()) / len(k8_tab) if k8_tab else 0.0
    tab_avg_base = sum(base_tab.values()) / len(base_tab) if base_tab else 0.0
    print(f"{'AVG':<20} {tab_avg_base*100:>14.1f}% {tab_avg_k8*100:>14.1f}% {(tab_avg_k8-tab_avg_base)*100:>+9.1f}pp")

    # --- PersonaQA ---
    print()
    print("PERSONAQA (50 personas × 6 attribute questions × 5 generations each)")
    print("-" * 72)
    k8_paq = score_personaqa(k8 / "personaqa_K8.json")
    base_paq_path = base / "personaqa_K1.json"
    if not base_paq_path.exists():
        base_paq_path = base / "personaqa_K8.json"
    base_paq = score_personaqa(base_paq_path) if base_paq_path.exists() else {}

    print(f"{'Question':<60} {args.label_base:>15} {args.label_k8:>15} {'Δ':>10}")
    for prompt in sorted(set(list(k8_paq.keys()) + list(base_paq.keys()))):
        if prompt == "__overall__":
            continue
        b = base_paq.get(prompt, 0.0)
        k = k8_paq.get(prompt, 0.0)
        prompt_short = prompt[:58]
        print(f"{prompt_short:<60} {b*100:>14.1f}% {k*100:>14.1f}% {(k-b)*100:>+9.1f}pp")
    paq_k8 = k8_paq.get("__overall__", 0.0)
    paq_base = base_paq.get("__overall__", 0.0)
    print(f"{'OVERALL':<60} {paq_base*100:>14.1f}% {paq_k8*100:>14.1f}% {(paq_k8-paq_base)*100:>+9.1f}pp")

    # --- Headline ---
    print()
    print("=" * 72)
    print("HEADLINE: average across the 3 evals (cls, taboo, personaqa)")
    print("=" * 72)
    print(f"{'Eval':<20} {args.label_base:>15} {args.label_k8:>15} {'Δ':>10}")
    print(f"{'Classification(20)':<20} {cls_avg_base*100:>14.1f}% {cls_avg_k8*100:>14.1f}% {(cls_avg_k8-cls_avg_base)*100:>+9.1f}pp")
    print(f"{'Taboo(20)':<20} {tab_avg_base*100:>14.1f}% {tab_avg_k8*100:>14.1f}% {(tab_avg_k8-tab_avg_base)*100:>+9.1f}pp")
    print(f"{'PersonaQA':<20} {paq_base*100:>14.1f}% {paq_k8*100:>14.1f}% {(paq_k8-paq_base)*100:>+9.1f}pp")
    avg_k8 = (cls_avg_k8 + tab_avg_k8 + paq_k8) / 3
    avg_base = (cls_avg_base + tab_avg_base + paq_base) / 3
    print(f"{'3-eval AVG':<20} {avg_base*100:>14.1f}% {avg_k8*100:>14.1f}% {(avg_k8-avg_base)*100:>+9.1f}pp")
    print()
    bar = 0.03
    if avg_k8 - avg_base >= bar:
        print(f"PLAN BAR (+3pp average) — CLEARED ✓ ({(avg_k8-avg_base)*100:+.1f}pp)")
    else:
        print(f"PLAN BAR (+3pp average) — NOT CLEARED ({(avg_k8-avg_base)*100:+.1f}pp)")


if __name__ == "__main__":
    main()
