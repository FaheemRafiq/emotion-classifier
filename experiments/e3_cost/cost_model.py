"""E3: cost and latency of the per-message AI pipeline, Bloom-style vs Intent2Act, per 1,000 chat messages.

Measured inputs (not assumed):
  - safety-check tokens and latency: E2 run of the deployed guard (experiments/e2_safety/results.json);
  - local emotion model latency: reports/metrics/latency.json and complex_eval (DeBERTa-v3, 89 ms CPU / 11 ms GPU);
  - characters-per-token ratio: measured from the E2 OpenAI usage (to size an LLM emotion call).
Assumptions (stated, applied identically to both pipelines):
  - a chat turn uses CHAT_IN input and CHAT_OUT output tokens;
  - Bloom runs one few-shot classifier per harm category (5 calls) on every coach reply, each assumed the same size
    as our guard prompt (Bloom's own prompts are larger, with 20 few-shot examples each, so this is conservative);
    the rare revision call after a harmful verdict is ignored;
  - Bloom has no emotion model; the "LLM emotion" row prices what it would cost to add one with GPT-4o.
Prices: OpenAI list prices per 1M tokens, checked 2026-10-09.

Usage: uv run python experiments/e3_cost/cost_model.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
PRICES = {"gpt-4o": (2.50, 10.00), "gpt-4o-mini": (0.15, 0.60)}  # USD per 1M input / output tokens
CHAT_IN, CHAT_OUT = 3000, 250
MESSAGES = 1000


def usd(model: str, tin: float, tout: float) -> float:
    pin, pout = PRICES[model]
    return (tin * pin + tout * pout) / 1e6


def main() -> None:
    e2 = json.loads((ROOT / "experiments/e2_safety/results.json").read_text())["cost_latency"]
    guard_in, guard_out = e2["mean_prompt_tokens"], e2["mean_completion_tokens"]
    prompt_chars = len((Path.home() / "Coding/JavaScript/mentor-ai/apps/api/templates/prompts/safety-guard.txt").read_text())
    chars_per_token = prompt_chars / guard_in  # item text is small next to the prompt; slight overestimate of chars/token
    instruction_tokens = len((ROOT / "experiments/e1_emotion/INSTRUCTION.md").read_text()) / chars_per_token
    emo_in, emo_out = instruction_tokens + 40, 20  # ~40 tokens per chat message, one JSON line out
    complex_eval = json.loads(next((ROOT / "reports/metrics").glob("complex_eval_*.json")).read_text())

    rows = {
        "bloom_style": {
            "chat (GPT-4o)": usd("gpt-4o", CHAT_IN, CHAT_OUT),
            "safety: 5 category classifiers on each reply (GPT-4o)": 5 * usd("gpt-4o", guard_in, guard_out),
            "emotion: none (if added via GPT-4o)": usd("gpt-4o", emo_in, emo_out),
        },
        "intent2act": {
            "chat (gpt-4o-mini, routed)": usd("gpt-4o-mini", CHAT_IN, CHAT_OUT),
            "safety: guard on user message + reply (gpt-4o-mini)": 2 * usd("gpt-4o-mini", guard_in, guard_out),
            "emotion: local DeBERTa-v3": 0.0,
        },
    }
    totals = {k: {"per_message_usd": sum(v.values()), f"per_{MESSAGES}_messages_usd": MESSAGES * sum(v.values()),
                  "items_per_1000": {n: round(MESSAGES * c, 3) for n, c in v.items()}} for k, v in rows.items()}
    bloom_safety = MESSAGES * rows["bloom_style"]["safety: 5 category classifiers on each reply (GPT-4o)"]
    ours_safety = MESSAGES * rows["intent2act"]["safety: guard on user message + reply (gpt-4o-mini)"]
    result = {
        "assumptions": {"chat_tokens_in_out": [CHAT_IN, CHAT_OUT], "bloom_classifier_tokens_equal_to_ours": True,
                        "prices_per_1M": PRICES, "price_date": "2026-10-09"},
        "measured": {"guard_tokens_in_out": [guard_in, guard_out], "guard_latency_s_median": e2["latency_s_median"],
                     "guard_latency_s_p90": e2["latency_s_p90"], "emotion_llm_tokens_in_out_est": [round(emo_in), emo_out],
                     "deberta_latency_ms_gpu": round(complex_eval["new"]["latency_ms_single_text"], 1),
                     "deberta_latency_ms_cpu": 89.5},
        "totals": totals,
        "safety_cost_ratio_bloom_over_ours": round(bloom_safety / ours_safety, 1),
        "total_cost_ratio_bloom_over_ours": round(totals["bloom_style"]["per_message_usd"]
                                                  / totals["intent2act"]["per_message_usd"], 1),
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=2))
    for k, v in totals.items():
        print(k, round(v[f"per_{MESSAGES}_messages_usd"], 2), v["items_per_1000"])
    print("safety cost ratio", result["safety_cost_ratio_bloom_over_ours"], "total ratio", result["total_cost_ratio_bloom_over_ours"])


if __name__ == "__main__":
    main()
