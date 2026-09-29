"""Final, one-time evaluation of a v3 model against the deployed v2 DistilBERT, on every benchmark.

Benchmarks (all human-labelled; none was used for training or model selection):
  1. GoEmotions test         - the frozen v2 test split (3,400 Reddit comments), same as every earlier model
  2. complexity slices       - the same test split cut into negation / contrast / implicit / long / sarcasm / emoji
                               subsets (rules fixed in src/complexity.py before evaluation)
  3. ISEAR test              - 1,096 first-person descriptions of emotional situations (implicit emotion)
  4. mixed-emotion test      - 108 GoEmotions test comments whose annotators chose two or more Mentor emotions
  5. Mentor-domain set       - 154 mentoring check-ins (labels are a draft, reported as provisional)

The new model's test evaluation is refused if it already exists (use --force --reason ...). The baseline's
test predictions are recomputed for the per-slice comparison only; this is logged as an analysis run
(no model is selected or changed on the basis of it).

Usage:
  uv run --extra transformer python -m src.evaluate_v3 --model models/deberta_v3
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.complexity import SLICE_NAMES, slices
from src.data import load_classes, load_split
from src.inference import load_predictor
from src.metrics import (
    append_experiment,
    compute_metrics,
    expected_calibration_error,
    print_report,
    save_confusion_matrices,
    save_json,
    threshold_table,
)

RERUN_LOG = config.METRICS / "forced_reruns.log"


def log_rerun(version: str, reason: str) -> None:
    with open(RERUN_LOG, "a") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}\t{version}\t{reason}\n")


def probas_of(predictor, texts: list[str]) -> np.ndarray:
    p = predictor.predict_proba(list(texts))
    order = [predictor.classes.index(c) for c in load_classes()]
    return p[:, order]


def score_single(y_true: list[str], probs: np.ndarray, classes: list[str], labels: list[str] | None = None) -> dict:
    pred = [classes[i] for i in probs.argmax(1)]
    m = compute_metrics(y_true, pred, labels or classes)
    correct = np.array(pred) == np.array(y_true)
    return {"n": len(y_true), "accuracy": float(correct.mean()), "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"], "per_class": m["per_class"],
            "ece": expected_calibration_error(probs.max(1), correct)}


def score_mixed(gold: list[str], probs: np.ndarray, classes: list[str]) -> dict:
    """Mixed emotions: is the top prediction one of the gold emotions, and do the top-2 cover both?"""
    top1_hit, top2_cover, mass = [], [], []
    for g, p in zip(gold, probs):
        gold_ids = {classes.index(c) for c in g.split("|")}
        order = np.argsort(-p)
        top1_hit.append(order[0] in gold_ids)
        top2_cover.append(gold_ids <= set(order[:max(2, len(gold_ids))]))
        mass.append(float(p[list(gold_ids)].sum()))
    return {"n": len(gold), "top1_in_gold": float(np.mean(top1_hit)),
            "top2_cover_all_gold": float(np.mean(top2_cover)), "mean_prob_on_gold": float(np.mean(mass))}


def latency(predictor, text: str, runs: int = 30) -> float:
    predictor.predict_proba([text])
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        predictor.predict_proba([text])
        times.append(time.perf_counter() - t)
    return float(np.median(times) * 1000)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=config.MODELS_DIR / "distilbert")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--reason", default="")
    args = parser.parse_args()

    classes = load_classes()
    new = load_predictor(args.model)
    base = load_predictor(args.baseline)
    out_test = config.METRICS / f"final_test_{new.model_version}.json"
    if out_test.exists() and not args.force:
        sys.exit(f"{out_test} exists: the test set was already used for {new.model_version}. "
                 "Use --force --reason '...' only if there is a real need.")
    log_rerun(new.model_version, args.reason or "first test evaluation of the v3 model version "
              "(selected on validation only)")
    log_rerun(base.model_version, "analysis only: baseline test predictions recomputed for the per-slice "
              "comparison with the v3 model; no model was selected or changed")

    test = load_split("test")
    isear = pd.read_parquet(config.DATA_PROCESSED_V3 / "isear_test.parquet")
    mixed = pd.read_parquet(config.DATA_PROCESSED_V3 / "mixed_test.parquet")
    domain = pd.read_csv(config.DOMAIN_EVAL_CSV)
    isear_labels = sorted(isear["label"].unique(), key=classes.index)
    flags = pd.DataFrame([slices(t) for t in test["text"]])

    results = {}
    for tag, predictor in (("baseline", base), ("new", new)):
        print(f"\n=== {tag}: {predictor.model_version} ===", flush=True)
        p_test = probas_of(predictor, test["text"])
        r = {"model_version": predictor.model_version,
             "goemotions_test": score_single(list(test["label"]), p_test, classes)}
        r["slices"] = {}
        for s in SLICE_NAMES:
            mask = flags[s].to_numpy()
            if mask.sum() >= 10:
                sub = score_single(list(test.loc[mask, "label"]), p_test[mask], classes)
                r["slices"][s] = {k: sub[k] for k in ("n", "accuracy", "macro_f1")}
        p_isear = probas_of(predictor, isear["text"])
        r["isear_test"] = score_single(list(isear["label"]), p_isear, classes, labels=isear_labels)
        r["mixed_test"] = score_mixed(list(mixed["gold"]), probas_of(predictor, mixed["text"]), classes)
        p_dom = probas_of(predictor, domain["text"])
        r["mentor_domain_provisional"] = score_single(list(domain["label"]), p_dom, classes)
        r["latency_ms_single_text"] = latency(predictor, "I keep skipping my workouts and I feel terrible about it.")
        r["device"] = str(getattr(predictor, "device", "cpu"))
        results[tag] = r

        if tag == "new":
            pred = [classes[i] for i in p_test.argmax(1)]
            print_report(test["label"], pred, classes, f"{predictor.model_version} - GoEmotions test")
            save_confusion_matrices(test["label"], pred, classes, f"confusion_matrix_{predictor.model_version}_test",
                                    f"{predictor.model_version} - test")
            metrics = compute_metrics(test["label"], pred, classes)
            correct = np.array(pred) == test["label"].to_numpy()
            save_json({"model_version": predictor.model_version, "model_path": str(args.model),
                       "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "forced_reason": args.reason or None, "metrics": metrics,
                       "calibration": {"ece": expected_calibration_error(p_test.max(1), correct),
                                       "thresholds": threshold_table(p_test.max(1), test["label"], pred)},
                       "slices": r["slices"]}, out_test)
            card = json.loads((args.model / "model_card.json").read_text())
            append_experiment(f"{card.get('experiment', 'E?')}-final", card.get("model_kind", "transformer"),
                              card.get("settings", ""), "test", metrics, "Final v3 model (single test evaluation)",
                              features="transformer-v3")

    save_json(results, config.METRICS / f"complex_eval_{new.model_version}.json")

    rows = []
    def add(name, key_b, key_n):
        rows.append((name, key_b, key_n))
    b, n = results["baseline"], results["new"]
    add("GoEmotions test - macro F1", b["goemotions_test"]["macro_f1"], n["goemotions_test"]["macro_f1"])
    add("GoEmotions test - accuracy", b["goemotions_test"]["accuracy"], n["goemotions_test"]["accuracy"])
    for s in SLICE_NAMES:
        if s in n["slices"]:
            add(f"  slice {s} (n={n['slices'][s]['n']}) - macro F1", b["slices"][s]["macro_f1"], n["slices"][s]["macro_f1"])
    add("ISEAR implicit test - macro F1", b["isear_test"]["macro_f1"], n["isear_test"]["macro_f1"])
    add("Mixed emotions - top-1 in gold", b["mixed_test"]["top1_in_gold"], n["mixed_test"]["top1_in_gold"])
    add("Mixed emotions - top-2 covers all", b["mixed_test"]["top2_cover_all_gold"], n["mixed_test"]["top2_cover_all_gold"])
    add("Mentor domain (provisional) - macro F1", b["mentor_domain_provisional"]["macro_f1"],
        n["mentor_domain_provisional"]["macro_f1"])
    add("Latency per message (ms)", b["latency_ms_single_text"], n["latency_ms_single_text"])
    print(f"\n{'benchmark':50} {'DistilBERT (v2)':>16} {'new (v3)':>10} {'change':>8}")
    for name, vb, vn in rows:
        print(f"{name:50} {vb:16.4f} {vn:10.4f} {vn - vb:+8.4f}")
    print(f"\nSaved {out_test.name} and complex_eval_{new.model_version}.json")


if __name__ == "__main__":
    main()
