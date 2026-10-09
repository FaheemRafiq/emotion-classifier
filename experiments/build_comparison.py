"""Combine E1-E3 into reports/metrics/base_paper_comparison.json (the single source for the report's numbers).

Usage: uv run python experiments/build_comparison.py
"""
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "experiments"


def main() -> None:
    e1 = json.loads((E / "e1_emotion/results.json").read_text())
    e1["benchmarks"]["mentor_domain"]["excluded_from_claims"] = (
        "The 154 Mentor-style texts and their draft labels were written by Claude (the assistant) on 2026-09-15, "
        "so Claude's score on them is circular; reported for completeness only.")
    e2 = json.loads((E / "e2_safety/results.json").read_text())
    e2_v1 = json.loads((E / "e2_safety/results_v1_zero_shot.json").read_text())
    e3 = json.loads((E / "e3_cost/results.json").read_text())
    out = {
        "generated": date.today().isoformat(),
        "base_paper": "Joerke et al. (2026) Bloom, CHI '26, doi:10.1145/3772318.3790506",
        "E1_emotion_dedicated_model_vs_llm_only": e1,
        "E2_safety_guard_on_bloom_benchmark": {
            "final_prompt": e2,
            "zero_shot_v1_prompt": {s: {m: e2_v1["splits"][s][m] for m in ("strict", "relaxed")}
                                    for s in e2_v1["splits"]},
            "protocol": ("v1 prompt written from the published taxonomy only and scored on all splits; one revision "
                         "made using validation errors only (as Bloom tuned on validation); final prompt then run on "
                         "the test splits once. Bloom numbers: their Table 3, mean of ten trials, GPT-4o, five "
                         "per-category classifiers tuned on validation."),
        },
        "E3_cost_latency": e3,
    }
    path = ROOT / "reports/metrics/base_paper_comparison.json"
    path.write_text(json.dumps(out, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
