"""E2: run the deployed Intent2Act safety guard on Bloom's published safety benchmark.

The guard prompt is read from the mentor-ai repo (apps/api/templates/prompts/safety-guard.txt), the same file
the API loads, and the model/settings match the deployed guard (gpt-4o-mini, temperature 0, max 200 tokens,
JSON-schema output). Each benchmark item is a (user message, coach reply) pair labelled SAFE/UNSAFE in one of
Bloom's five harm categories, so the guard runs in its MENTOR_REPLY mode.

The prompt was written from the published taxonomy definitions only and was NOT tuned on this benchmark:
both the validation and the test split are evaluated once, zero-shot.

Responses are cached in outputs.jsonl (keyed by a hash of prompt + item), so reruns cost nothing.
The OpenAI key is read from mentor-ai's apps/api/.env and never printed.

Usage:  uv run python experiments/e2_safety/run_guard.py
"""
import csv
import hashlib
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).parent
MENTOR_AI = Path.home() / "Coding/JavaScript/mentor-ai"
PROMPT_PATH = MENTOR_AI / "apps/api/templates/prompts/safety-guard.txt"
ENV_PATH = MENTOR_AI / "apps/api/.env"
OUTPUTS = HERE / "outputs.jsonl"
MODEL = "gpt-4o-mini"
SPLITS = {"validation": "validation.csv", "test": "test.csv", "test_corrected": "test_corrected.csv"}
CATEGORIES = ["BODILY_HARM_OVEREXERTION", "BODY_IMAGE_EATING", "MENTAL_HEALTH_CRISIS", "MENTAL_HEALTH_GENERAL",
              "NEGATIVE_MINDSET_FEEDBACK", "OUT_OF_SCOPE_ADVICE", "NONE"]

# Same fields as verdictSchema in apps/api/src/services/safety-guard.service.ts.
SCHEMA = {
    "name": "safety_verdict",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["category", "severity", "harmful", "rationale"],
        "properties": {
            "category": {"type": "string", "enum": CATEGORIES},
            "severity": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
            "harmful": {"type": "boolean"},
            "rationale": {"type": "string"},
        },
    },
}


def api_key() -> str:
    for line in ENV_PATH.read_text().splitlines():
        if line.startswith("OPENAI_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("OPENAI_API_KEY not found in mentor-ai apps/api/.env")


def load_items() -> list[dict]:
    items = []
    for split, name in SPLITS.items():
        with open(HERE / "data" / name, newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                if row.get("is_few_shot", "FALSE").upper() == "TRUE":
                    continue  # Bloom's training examples (their few-shot prompts); not evaluation data
                items.append({"split": split, "row": i, "category": int(row["category"]),
                              "input_text": row["input_text"], "output_text": row["output_text"],
                              "label": row["output_harmful"].strip().upper()})
    return items


def guard_input(item: dict) -> str:
    return f"CHECK: MENTOR_REPLY\nUser: {item['input_text']}\nMentor: {item['output_text']}"


def call(prompt: str, text: str, key: str) -> dict:
    body = {"model": MODEL, "temperature": 0, "max_tokens": 200,
            "response_format": {"type": "json_schema", "json_schema": SCHEMA},
            "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": text}]}
    for attempt in range(5):
        try:
            start = time.perf_counter()
            req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                resp = json.load(r)
            latency = time.perf_counter() - start
            return {"verdict": json.loads(resp["choices"][0]["message"]["content"]), "usage": resp["usage"],
                    "latency_s": round(latency, 3), "model": resp.get("model")}
        except Exception as e:  # rate limit / transient network error
            if attempt == 4:
                raise
            time.sleep(2 ** attempt + 1)
            last = e  # noqa: F841


def main() -> None:
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:12]
    cache = {}
    if OUTPUTS.exists():
        for line in OUTPUTS.read_text().splitlines():
            rec = json.loads(line)
            cache[rec["key"]] = rec
    items = load_items()
    todo = {}
    for item in items:
        text = guard_input(item)
        key = hashlib.sha256(f"{prompt_hash}|{MODEL}|{text}".encode()).hexdigest()
        item["key"] = key
        if key not in cache:
            todo[key] = text
    print(f"{len(items)} items, {len(todo)} uncached calls (prompt {prompt_hash})")
    if todo:
        secret = api_key()
        with open(OUTPUTS, "a") as out, ThreadPoolExecutor(max_workers=6) as pool:
            futures = {k: pool.submit(call, prompt, t, secret) for k, t in todo.items()}
            for n, (k, fut) in enumerate(futures.items(), 1):
                rec = {"key": k, "prompt_hash": prompt_hash, **fut.result()}
                cache[k] = rec
                out.write(json.dumps(rec) + "\n")
                if n % 50 == 0:
                    print(f"  {n}/{len(todo)}", flush=True)
    with open(HERE / "predictions.jsonl", "w") as f:
        for item in items:
            rec = cache[item["key"]]
            f.write(json.dumps({k: item[k] for k in ("split", "row", "category", "label")}
                               | {"pred": rec["verdict"], "latency_s": rec["latency_s"], "usage": rec["usage"],
                                  "prompt_hash": prompt_hash}) + "\n")
    print("wrote predictions.jsonl")


if __name__ == "__main__":
    main()
