"""E1: score the LLM-only baseline (Claude, label-blind, fixed instruction) against our DeBERTa-v3 model.

Both systems are scored on the same ids from key.jsonl:
  - goemotions_test / isear_test / mentor_domain: accuracy and macro F1 of top-1 (ISEAR over its 5 classes);
  - mixed_test: top-1 in gold, and top-2 covers every gold emotion (same rule as src/evaluate_v3.py).
95% bootstrap confidence intervals (2,000 resamples, seed 42) for every macro F1 / accuracy.
DeBERTa predictions are cached in deberta_predictions.jsonl (analysis only: no model is selected or changed).

Usage: uv run --extra transformer python experiments/e1_emotion/score.py
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
CLASSES = ["joy", "sadness", "fear_anxiety", "anger_frustration", "guilt", "motivation", "neutral"]
ISEAR_CLASSES = ["joy", "sadness", "fear_anxiety", "anger_frustration", "guilt"]


def deberta_predictions(key: list[dict]) -> dict:
    cache = HERE / "deberta_predictions.jsonl"
    if cache.exists():
        return {r["id"]: r for r in map(json.loads, cache.read_text().splitlines())}
    from src import config
    from src.inference import load_predictor
    predictor = load_predictor(config.V3_TRANSFORMER_DIR)
    order = [predictor.classes.index(c) for c in CLASSES]
    probs = predictor.predict_proba([r["text"] for r in key])[:, order]
    out = {}
    with open(cache, "w") as f:
        for r, p in zip(key, probs):
            top = np.argsort(-p)
            rec = {"id": r["id"], "top1": CLASSES[top[0]], "top2": CLASSES[top[1]], "model": predictor.model_version}
            out[r["id"]] = rec
            f.write(json.dumps(rec) + "\n")
    return out


def llm_predictions() -> dict:
    out = {}
    for path in sorted((HERE / "answers").glob("answers_*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                out[rec["id"]] = rec
    return out


def single_metrics(gold: list[str], pred: list[str], labels: list[str]) -> dict:
    return {"n": len(gold), "accuracy": float(np.mean(np.array(gold) == np.array(pred))),
            "macro_f1": float(f1_score(gold, pred, labels=labels, average="macro", zero_division=0))}


def bootstrap(gold: list[str], pred: list[str], labels: list[str], n: int = 2000, seed: int = 42) -> dict:
    rng = random.Random(seed)
    idx = range(len(gold))
    f1s, accs = [], []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        m = single_metrics([gold[i] for i in s], [pred[i] for i in s], labels)
        f1s.append(m["macro_f1"])
        accs.append(m["accuracy"])
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)]  # noqa: E731
    return {"macro_f1_ci95": ci(f1s), "accuracy_ci95": ci(accs)}


def mixed_metrics(rows: list[dict], preds: dict) -> dict:
    top1, cover = [], []
    for r in rows:
        gold = set(r["gold"].split("|"))
        p = preds[r["id"]]
        top1.append(p["top1"] in gold)
        cover.append(gold <= {p["top1"], p["top2"]})
    return {"n": len(rows), "top1_in_gold": float(np.mean(top1)), "top2_cover_all_gold": float(np.mean(cover))}


def main() -> None:
    key = [json.loads(line) for line in (HERE / "key.jsonl").read_text().splitlines()]
    llm = llm_predictions()
    missing = [r["id"] for r in key if r["id"] not in llm]
    invalid = [i for i, p in llm.items() if p.get("top1") not in CLASSES or p.get("top2") not in CLASSES]
    if missing or invalid:
        sys.exit(f"LLM answers incomplete: {len(missing)} missing, {len(invalid)} invalid ({(missing + invalid)[:10]})")
    deb = deberta_predictions(key)

    results = {"llm_baseline": "Claude (Anthropic), in-session, label-blind, fixed instruction (INSTRUCTION.md)",
               "our_model": next(iter(deb.values()))["model"], "benchmarks": {}}
    for source, labels in (("goemotions_test", CLASSES), ("isear_test", ISEAR_CLASSES), ("mentor_domain", CLASSES)):
        rows = [r for r in key if r["source"] == source]
        gold = [r["gold"] for r in rows]
        res = {}
        for name, preds in (("deberta_v3", deb), ("llm_only", llm)):
            pred = [preds[r["id"]]["top1"] for r in rows]
            res[name] = single_metrics(gold, pred, labels) | bootstrap(gold, pred, labels)
            res[name]["per_class_f1"] = dict(zip(labels, map(float, f1_score(gold, pred, labels=labels,
                                                                               average=None, zero_division=0))))
        results["benchmarks"][source] = res
    mixed = [r for r in key if r["source"] == "mixed_test"]
    results["benchmarks"]["mixed_test"] = {"deberta_v3": mixed_metrics(mixed, deb), "llm_only": mixed_metrics(mixed, llm)}
    (HERE / "results.json").write_text(json.dumps(results, indent=2))

    print(f"{'benchmark':18} {'metric':16} {'DeBERTa-v3':>18} {'LLM-only':>18}")
    for source in ("goemotions_test", "isear_test", "mentor_domain"):
        r = results["benchmarks"][source]
        for metric in ("macro_f1", "accuracy"):
            d, l = r["deberta_v3"], r["llm_only"]
            print(f"{source:18} {metric:16} {d[metric]:.3f} {str(d[metric + '_ci95']):>12} {l[metric]:.3f} {str(l[metric + '_ci95']):>12}")
    m = results["benchmarks"]["mixed_test"]
    for metric in ("top1_in_gold", "top2_cover_all_gold"):
        print(f"{'mixed_test':18} {metric:16} {m['deberta_v3'][metric]:18.3f} {m['llm_only'][metric]:18.3f}")


if __name__ == "__main__":
    main()
