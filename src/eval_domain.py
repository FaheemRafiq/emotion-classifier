"""Score a model on the Mentor-domain evaluation set (data/mentor_eval/mentor_eval.csv).

Usage: uv run python -m src.eval_domain [--model models/distilbert] [--threshold 0.6] [--show-errors]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.inference import load_predictor
from src.metrics import compute_metrics, expected_calibration_error, print_report, save_json, threshold_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--threshold", type=float, default=0.6, help="Node-side EMOTION_MIN_CONFIDENCE to simulate")
    parser.add_argument("--show-errors", action="store_true")
    args = parser.parse_args()

    data = pd.read_csv(config.DOMAIN_EVAL_CSV)
    readme = (config.DOMAIN_EVAL_CSV.parent / "README.md").read_text()
    reviewed = "REVIEWED_BY: (nobody yet)" not in readme
    if not reviewed:
        print("NOTE: labels in mentor_eval.csv are still an unreviewed draft - treat these numbers as provisional.\n")

    predictor = load_predictor(args.model)
    classes = config.MENTOR_CLASSES
    texts = data["text"].tolist()
    probas = predictor.predict_proba(texts)
    pred = predictor.predict(texts) if probas is None else [predictor.classes[i] for i in probas.argmax(1)]
    data["pred"] = pred
    data["confidence"] = probas.max(1) if probas is not None else np.nan

    metrics = compute_metrics(data["label"], data["pred"], classes)
    print_report(data["label"], data["pred"], classes, f"Mentor-domain eval - {predictor.model_version}")

    by_kind = (
        data.assign(correct=data["label"] == data["pred"]).groupby("kind")["correct"].agg(["mean", "size"])
        .rename(columns={"mean": "accuracy", "size": "n"})
    )
    print("Accuracy by kind:\n", by_kind.round(3).to_string())

    calibration = None
    if probas is not None:
        correct = (data["label"] == data["pred"]).to_numpy()
        calibration = {"ece": expected_calibration_error(data["confidence"], correct),
                       "thresholds": threshold_table(data["confidence"], data["label"], data["pred"])}
        sent = (data["confidence"] >= args.threshold) & (data["pred"] != "neutral")
        print(f"\nAt threshold {args.threshold}: context sent for {sent.mean():.0%} of messages; "
              f"of those, {(data.loc[sent, 'label'] == data.loc[sent, 'pred']).mean():.0%} correct; "
              f"ECE {calibration['ece']:.3f}")
        wrong_sent = data[sent & (data["label"] != data["pred"])]
        print(f"Wrong emotion injected into the prompt: {len(wrong_sent)} / {len(data)}")

    if args.show_errors:
        print("\nErrors:")
        for _, r in data[data["label"] != data["pred"]].iterrows():
            print(f"  [{r['id']}] {r['text'][:70]:70} true={r['label']:18} pred={r['pred']:18} conf={r['confidence']:.2f}")

    out = config.METRICS / f"domain_eval_{predictor.model_version}.json"
    save_json({"model_version": predictor.model_version, "labels_reviewed": reviewed, "n": len(data),
               "metrics": metrics, "accuracy_by_kind": by_kind.to_dict(orient="index"), "calibration": calibration,
               "predictions": data[["id", "label", "pred", "confidence"]].to_dict(orient="records")}, out)
    print(f"\nMacro F1: {metrics['macro_f1']:.4f} | accuracy: {metrics['accuracy']:.4f} -> {out}")


if __name__ == "__main__":
    main()
