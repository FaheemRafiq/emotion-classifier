"""Shared evaluation helpers: metrics, reports, confusion matrices, calibration, experiment log."""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)

from src import config  # noqa: E402

EXPERIMENT_COLUMNS = [
    "id",
    "timestamp",
    "label_map",
    "features",
    "model",
    "settings",
    "eval_split",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "cv_macro_f1",
    "purpose",
]


def compute_metrics(y_true, y_pred, labels: list[str]) -> dict:
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, zero_division=0
        ),
    }


def expected_calibration_error(confidence, correct, n_bins: int = 10) -> float:
    """ECE: how far the model's confidence is from its actual accuracy, averaged over confidence bins."""
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(confidence, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def threshold_table(confidence, y_true, y_pred, thresholds=(0.0, 0.4, 0.5, 0.6, 0.7)) -> list[dict]:
    """For each confidence threshold: how often a non-neutral emotion would be sent, and how often it is right."""
    confidence = np.asarray(confidence)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rows = []
    for thr in thresholds:
        sent = (confidence >= thr) & (y_pred != "neutral")
        rows.append(
            {
                "threshold": thr,
                "coverage": float((confidence >= thr).mean()),
                "accuracy_at_threshold": float((y_pred[confidence >= thr] == y_true[confidence >= thr]).mean())
                if (confidence >= thr).any() else None,
                "non_neutral_sent": float(sent.mean()),
                "non_neutral_precision": float((y_pred[sent] == y_true[sent]).mean()) if sent.any() else None,
            }
        )
    return rows


def print_report(y_true, y_pred, labels: list[str], title: str) -> None:
    print(f"\n=== {title} ===")
    print(classification_report(y_true, y_pred, labels=labels, digits=4, zero_division=0))


def save_confusion_matrices(y_true, y_pred, labels: list[str], name: str, title: str) -> list[Path]:
    """Save a raw-count and a row-normalized confusion matrix."""
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    paths = []
    for normalize, suffix, fmt in ((None, "", "d"), ("true", "_normalized", ".2f")):
        fig, ax = plt.subplots(figsize=(9, 8))
        ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred,
            labels=labels,
            normalize=normalize,
            values_format=fmt,
            xticks_rotation=45,
            colorbar=False,
            ax=ax,
        )
        ax.set_title(f"{title}{' (row-normalized)' if normalize else ''}")
        fig.tight_layout()
        path = config.FIGURES / f"{name}{suffix}.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        paths.append(path)
    return paths


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


def append_experiment(
    exp_id: str,
    model: str,
    settings: str,
    eval_split: str,
    metrics: dict,
    purpose: str,
    cv_macro_f1: float | None = None,
    features: str = config.FEATURES_VERSION,
) -> None:
    config.METRICS.mkdir(parents=True, exist_ok=True)
    is_new = not config.EXPERIMENTS_CSV.exists()
    row = {
        "id": exp_id,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label_map": config.LABEL_MAP_VERSION,
        "features": features,
        "model": model,
        "settings": settings,
        "eval_split": eval_split,
        "accuracy": f"{metrics['accuracy']:.4f}",
        "macro_precision": f"{metrics['macro_precision']:.4f}",
        "macro_recall": f"{metrics['macro_recall']:.4f}",
        "macro_f1": f"{metrics['macro_f1']:.4f}",
        "weighted_f1": f"{metrics['weighted_f1']:.4f}",
        "cv_macro_f1": "" if cv_macro_f1 is None else f"{cv_macro_f1:.4f}",
        "purpose": purpose,
    }
    with open(config.EXPERIMENTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXPERIMENT_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
