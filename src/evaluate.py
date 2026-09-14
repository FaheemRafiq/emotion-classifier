"""Final evaluation on the untouched test set, then serialize.

The test set is meant to be used once per model version. Model *selection* must happen on validation
(and the Mentor-domain eval set), never by comparing test scores.

Usage:
  uv run python -m src.evaluate                       # fit the frozen sklearn config on train, test once, save joblib
  uv run python -m src.evaluate --model models/distilbert   # test an already-trained transformer
  ... --force --reason "why the test set is being used again"
"""
import argparse
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy
import sklearn

from src import config
from src.data import load_classes, load_manifest, load_xy
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
from src.train import describe_settings, pipeline_from_frozen


def archive_previous_result() -> None:
    """Keep the previous final-test result and model card under their own version name."""
    if not config.FINAL_TEST_JSON.exists():
        return
    previous = json.loads(config.FINAL_TEST_JSON.read_text())
    version = previous.get("model_version", "unknown")
    archive = config.METRICS / "archive"
    archive.mkdir(parents=True, exist_ok=True)
    shutil.copy(config.FINAL_TEST_JSON, archive / f"final_test_{version}.json")
    if config.MODEL_CARD_PATH.exists():
        shutil.copy(config.MODEL_CARD_PATH, archive / f"model_card_{version}.json")
    print(f"Archived previous result ({version}) to {archive}")


def fit_sklearn_model(now: datetime) -> tuple[Path, str, str, dict]:
    if not config.FROZEN_CONFIG_PATH.exists():
        sys.exit(f"{config.FROZEN_CONFIG_PATH} not found - run `make tune` first")
    frozen = json.loads(config.FROZEN_CONFIG_PATH.read_text())
    X_train, y_train = load_xy("train")

    pipeline = pipeline_from_frozen(frozen)
    pipeline.fit(X_train, y_train)

    kind = f"{frozen['kind']}{'_calibrated' if frozen['calibrated'] else ''}"
    features = frozen.get("features", "fs1")
    model_version = f"{frozen['label_map_version']}-{features}-{kind}-{now:%Y%m%d}"
    settings = describe_settings(frozen["C"], features, **frozen["tfidf"])

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, config.MODEL_PATH)
    card = {
        "model_version": model_version,
        "created_at": now.isoformat(timespec="seconds"),
        "task": "Emotion classification of user text (not a medical or psychological diagnosis)",
        "classes": [str(c) for c in pipeline.classes_],
        "frozen_config": frozen,
        "settings": settings,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "numpy": numpy.__version__},
    }
    save_json(card, config.MODEL_CARD_PATH)
    return config.MODEL_PATH, kind, settings, {"features": features, "cv_macro_f1": frozen["cv_macro_f1"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, help="evaluate an already-trained model dir/file instead of fitting")
    parser.add_argument("--force", action="store_true", help="re-run even though the test set was already used")
    parser.add_argument("--reason", help="required with --force; logged for transparency")
    args = parser.parse_args()

    if args.force and not args.reason:
        parser.error("--force requires --reason")
    if config.FINAL_TEST_JSON.exists() and not args.force:
        sys.exit(f"{config.FINAL_TEST_JSON} already exists: the test set has been used once. "
                 "Re-running would be test-set tuning. Use --force --reason '...' only if justified.")

    now = datetime.now(timezone.utc)
    archive_previous_result()

    if args.model:
        model_path, kind, settings, extra = args.model, "transformer", "see model_card.json", {}
    else:
        model_path, kind, settings, extra = fit_sklearn_model(now)

    predictor = load_predictor(model_path)
    classes = load_classes()
    manifest = load_manifest()
    X_test, y_test = load_xy("test")
    card_path = model_path / "model_card.json" if model_path.is_dir() else model_path.with_name(config.MODEL_CARD_PATH.name)
    card = json.loads(card_path.read_text())
    model_version = predictor.model_version
    kind = card.get("model_kind", kind)
    settings = card.get("settings", settings)
    features = extra.get("features", card.get("features", "n/a"))

    probas = predictor.predict_proba(list(X_test))
    pred = predictor.predict(list(X_test)) if probas is None else [predictor.classes[i] for i in probas.argmax(1)]
    metrics = compute_metrics(y_test, pred, classes)
    print_report(y_test, pred, classes, f"FINAL {model_version} (test)")
    save_confusion_matrices(y_test, pred, classes, f"confusion_matrix_final_test_{model_version}",
                            f"Final {model_version} - test")

    calibration = None
    if probas is not None:
        confidence = probas.max(1)
        calibration = {
            "ece": expected_calibration_error(confidence, numpy.asarray(pred) == numpy.asarray(y_test)),
            "thresholds": threshold_table(confidence, y_test, pred),
        }
        print(f"ECE (test): {calibration['ece']:.4f}")

    summary = {k: v for k, v in metrics.items() if k != "per_class"}
    card.update({"test_metrics": summary, "test_ece": calibration["ece"] if calibration else None,
                 "data_sha256": {s: manifest["splits"][s]["sha256"] for s in config.SPLITS}})
    save_json(card, card_path)

    result = {"model_version": model_version, "model_path": str(model_path), "evaluated_at": now.isoformat(timespec="seconds"),
              "forced_reason": args.reason if args.force else None, "metrics": metrics, "calibration": calibration}
    save_json(result, config.FINAL_TEST_JSON)
    save_json(result, config.METRICS / f"final_test_{model_version}.json")
    append_experiment("E4-final" if args.model else "E3-final", kind, settings, "test", metrics,
                      "Final model (single test evaluation)", cv_macro_f1=extra.get("cv_macro_f1"), features=features)
    if args.force:
        with open(config.METRICS / "forced_reruns.log", "a") as f:
            f.write(f"{now.isoformat(timespec='seconds')}\t{model_version}\t{args.reason}\n")

    print(f"\nTest macro F1: {metrics['macro_f1']:.4f}")
    print(f"Model: {model_path} ({model_version})")


if __name__ == "__main__":
    main()
