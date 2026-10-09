"""Greyscale, print-safe figures for the base-paper comparison (E1-E3), from base_paper_comparison.json.

Usage: uv run python experiments/make_figures.py  -> reports/figures/base_paper_e{1,2,3}_*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports/figures"
OURS = dict(facecolor="#404040")
OTHER = dict(facecolor="#d9d9d9", hatch="///", edgecolor="#8a8a8a", linewidth=0)
INK, MUTED = "#111111", "#555555"


def style(ax, xlabel):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#bbbbbb")
    ax.tick_params(axis="y", length=0, labelsize=11, colors=INK)
    ax.tick_params(axis="x", colors=MUTED)
    ax.grid(axis="x", color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=10)


def paired(ax, labels, other, ours, other_name, ours_name, fmt="{:.2f}", xmax=1.0):
    h = 0.36
    ys = range(len(labels))
    for i, (o, u) in enumerate(zip(other, ours)):
        ax.barh(i - h / 2, o, h, **OTHER)
        ax.barh(i + h / 2, u, h, **OURS)
        ax.text(o + xmax * 0.01, i - h / 2, fmt.format(o), va="center", fontsize=10, color=MUTED)
        ax.text(u + xmax * 0.01, i + h / 2, fmt.format(u), va="center", fontsize=10, color=INK, fontweight="bold")
    ax.set_yticks(list(ys), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, xmax * 1.12)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, **OTHER), plt.Rectangle((0, 0), 1, 1, **OURS)],
              labels=[other_name, ours_name], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
              fontsize=10)


def main() -> None:
    d = json.loads((ROOT / "reports/metrics/base_paper_comparison.json").read_text())

    e1 = d["E1_emotion_dedicated_model_vs_llm_only"]["benchmarks"]
    labels = ["GoEmotions test\n(Reddit, human labels)", "ISEAR test\n(implicit situations)",
              "Mixed emotions:\ntop-2 finds all"]
    llm = [e1["goemotions_test"]["llm_only"]["macro_f1"], e1["isear_test"]["llm_only"]["macro_f1"],
           e1["mixed_test"]["llm_only"]["top2_cover_all_gold"]]
    ours = [e1["goemotions_test"]["deberta_v3"]["macro_f1"], e1["isear_test"]["deberta_v3"]["macro_f1"],
            e1["mixed_test"]["deberta_v3"]["top2_cover_all_gold"]]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
    paired(ax, labels, llm, ours, "General LLM, zero-shot (Claude)", "Intent2Act model (DeBERTa-v3, local)")
    style(ax, "Macro F1 (mixed row: share of texts whose top-2 predictions cover every gold emotion)")
    fig.tight_layout()
    fig.savefig(FIG / "base_paper_e1_emotion.png")

    e2 = d["E2_safety_guard_on_bloom_benchmark"]["final_prompt"]["splits"]
    names = {"validation": "Validation (400)", "test": "Test (100)", "test_corrected": "Corrected test (100)"}
    labels = [names[s] for s in names]
    bloom = [e2[s]["relaxed"]["bloom"]["f1"] for s in names]
    ours = [e2[s]["relaxed"]["f1"] for s in names]
    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=200)
    paired(ax, labels, bloom, ours, "Bloom: 5 GPT-4o classifiers per reply", "Intent2Act: 1 gpt-4o-mini guard call")
    style(ax, "F1 for detecting unsafe coach replies (relaxed), Bloom safety benchmark")
    fig.tight_layout()
    fig.savefig(FIG / "base_paper_e2_safety.png")

    t = d["E3_cost_latency"]["totals"]
    items_b, items_o = t["bloom_style"]["items_per_1000"], t["intent2act"]["items_per_1000"]
    labels = ["Chat replies", "Safety checks", "Emotion detection\n(Bloom: none; if added)", "Total"]
    bloom = list(items_b.values()) + [t["bloom_style"]["per_1000_messages_usd"]]
    ours = list(items_o.values()) + [t["intent2act"]["per_1000_messages_usd"]]
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=200)
    paired(ax, labels, bloom, ours, "Bloom-style (GPT-4o everywhere)", "Intent2Act (routed models + local classifier)",
           fmt="${:.2f}", xmax=max(bloom))
    style(ax, "Estimated API cost in USD per 1,000 chat messages (list prices, 2026-10-09)")
    fig.tight_layout()
    fig.savefig(FIG / "base_paper_e3_cost.png")
    print("saved 3 figures")


if __name__ == "__main__":
    main()
