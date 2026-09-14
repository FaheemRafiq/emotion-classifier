"""Baseline experiments E1 (Logistic Regression) and E2 (Linear SVM), scored on validation.

Usage: uv run python -m src.train [--experiments E1 E2] [--features fs1|fs2]
"""
import argparse
import time

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from src import config
from src.data import load_classes, load_xy
from src.metrics import append_experiment, compute_metrics, print_report, save_confusion_matrices, save_json

EXPERIMENTS = {
    "E1": {"kind": "logreg", "C": 1.0, "purpose": "Baseline"},
    "E2": {"kind": "linear_svm", "C": 1.0, "purpose": "Comparison"},
}


def build_classifier(kind: str, C: float = 1.0, calibrated: bool = False):
    if kind == "logreg":
        return LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=config.SEED)
    if kind == "linear_svm":
        svm = LinearSVC(C=C, class_weight="balanced", random_state=config.SEED)
        # LinearSVC scores are not probabilities; calibrate when a confidence is needed.
        # Isotonic on one full-data SVM (ensemble=False) was chosen on validation: it keeps the SVM's
        # macro F1 (0.711 vs 0.715) where sigmoid/ensemble lost recall (0.695), and has the lowest ECE (0.027).
        return CalibratedClassifierCV(svm, method="isotonic", cv=3, ensemble=False) if calibrated else svm
    raise ValueError(f"Unknown model kind: {kind!r}")


def build_features(features: str = config.FEATURES_VERSION, **word_overrides):
    word_params = {**config.TFIDF_DEFAULTS, **word_overrides}
    if features == "fs1":
        return TfidfVectorizer(**word_params)
    if features == "fs2":
        return FeatureUnion(
            [
                ("word", TfidfVectorizer(token_pattern=config.WORD_TOKEN_PATTERN, **word_params)),
                ("char", TfidfVectorizer(**config.CHAR_TFIDF)),
            ]
        )
    raise ValueError(f"Unknown feature set: {features!r}")


def word_vectorizer(pipeline: Pipeline) -> TfidfVectorizer:
    """The word-level TF-IDF inside a pipeline, for either feature set."""
    features = pipeline.named_steps["features"]
    return features if isinstance(features, TfidfVectorizer) else dict(features.transformer_list)["word"]


def word_param_prefix(features: str) -> str:
    return "features__" if features == "fs1" else "features__word__"


def build_pipeline(
    kind: str, C: float = 1.0, calibrated: bool = False, features: str = config.FEATURES_VERSION, **word_overrides
) -> Pipeline:
    """Features + classifier in one pipeline, so vectorizers are only ever fitted on training text."""
    return Pipeline(
        [
            ("features", build_features(features, **word_overrides)),
            ("clf", build_classifier(kind, C=C, calibrated=calibrated)),
        ]
    )


def pipeline_from_frozen(frozen: dict) -> Pipeline:
    tfidf = dict(frozen["tfidf"])
    tfidf["ngram_range"] = tuple(tfidf["ngram_range"])
    return build_pipeline(
        frozen["kind"], C=frozen["C"], calibrated=frozen["calibrated"], features=frozen.get("features", "fs1"), **tfidf
    )


def describe_settings(C: float, features: str = config.FEATURES_VERSION, **word_overrides) -> str:
    tfidf = {**config.TFIDF_DEFAULTS, **word_overrides}
    extra = " + emoji/punct tokens + char_wb(2,5)" if features == "fs2" else ""
    return (
        f"{features}: word TF-IDF ngram={tuple(tfidf['ngram_range'])} min_df={tfidf['min_df']} "
        f"max_df={tfidf['max_df']} sublinear={tfidf['sublinear_tf']}{extra}; C={C}; class_weight=balanced"
    )


def run_experiment(exp_id: str, features: str, classes: list[str], X_train, y_train, X_val, y_val) -> dict:
    spec = EXPERIMENTS[exp_id]
    pipeline = build_pipeline(spec["kind"], C=spec["C"], features=features)

    start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - start

    pred = pipeline.predict(X_val)
    metrics = compute_metrics(y_val, pred, classes)
    print_report(y_val, pred, classes, f"{exp_id} {spec['kind']} {features} (validation)")
    print(f"Macro F1: {metrics['macro_f1']:.4f} | fit time: {fit_seconds:.1f}s")

    settings = describe_settings(spec["C"], features)
    save_json(
        {"id": exp_id, "model": spec["kind"], "features": features, "settings": settings, "eval_split": "validation",
         "fit_seconds": fit_seconds, "metrics": metrics},
        config.METRICS / f"{exp_id}_{features}_validation.json",
    )
    save_confusion_matrices(y_val, pred, classes, f"confusion_matrix_{exp_id}_{features}_validation",
                            f"{exp_id} {spec['kind']} {features} - validation")
    append_experiment(exp_id, spec["kind"], settings, "validation", metrics, spec["purpose"], features=features)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", nargs="+", choices=sorted(EXPERIMENTS), default=sorted(EXPERIMENTS))
    parser.add_argument("--features", choices=["fs1", "fs2"], default=config.FEATURES_VERSION)
    args = parser.parse_args()

    classes = load_classes()
    X_train, y_train = load_xy("train")
    X_val, y_val = load_xy("validation")
    print(f"train={len(X_train)} validation={len(X_val)} features={args.features} classes={classes}")

    for exp_id in args.experiments:
        run_experiment(exp_id, args.features, classes, X_train, y_train, X_val, y_val)


if __name__ == "__main__":
    main()
