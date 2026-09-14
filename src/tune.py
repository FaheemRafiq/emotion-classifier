"""E3: limited grid search (cross-validation on train only), then freeze the best configuration.

Usage: uv run python -m src.tune [--features fs1|fs2]
"""
import argparse
from datetime import datetime, timezone

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src import config
from src.data import load_classes, load_manifest, load_xy
from src.metrics import append_experiment, compute_metrics, print_report, save_confusion_matrices, save_json
from src.train import build_classifier, build_pipeline, describe_settings, word_param_prefix

# fs1 is cheap (guide grid). fs2 fits are ~4x slower (char n-grams), so its grid is smaller.
GRIDS = {
    "fs1": {"ngram_range": [(1, 1), (1, 2)], "min_df": [1, 2, 5], "C": [0.5, 1.0, 2.0]},
    "fs2": {"ngram_range": [(1, 2)], "min_df": [1, 2], "C": [0.25, 0.5, 1.0]},
}


def param_grid(features: str) -> list[dict]:
    g = GRIDS[features]
    prefix = word_param_prefix(features)
    common = {f"{prefix}ngram_range": g["ngram_range"], f"{prefix}min_df": g["min_df"], "clf__C": g["C"]}
    return [{"clf": [build_classifier("logreg")], **common}, {"clf": [build_classifier("linear_svm")], **common}]


def evaluate_on_validation(exp_id, kind, features, pipeline, settings, classes, X_val, y_val, purpose, cv_macro_f1):
    pred = pipeline.predict(X_val)
    metrics = compute_metrics(y_val, pred, classes)
    print_report(y_val, pred, classes, f"{exp_id} {kind} {features} (validation)")
    save_json({"id": exp_id, "model": kind, "features": features, "settings": settings, "eval_split": "validation",
               "cv_macro_f1": cv_macro_f1, "metrics": metrics},
              config.METRICS / f"{exp_id}_{features}_validation.json")
    save_confusion_matrices(y_val, pred, classes, f"confusion_matrix_{exp_id}_{features}_validation",
                            f"{exp_id} {kind} {features} - validation")
    append_experiment(exp_id, kind, settings, "validation", metrics, purpose, cv_macro_f1=cv_macro_f1,
                      features=features)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", choices=sorted(GRIDS), default=config.FEATURES_VERSION)
    args = parser.parse_args()
    features = args.features
    prefix = word_param_prefix(features)

    classes = load_classes()
    X_train, y_train = load_xy("train")
    X_val, y_val = load_xy("validation")

    grid = GridSearchCV(
        build_pipeline("logreg", features=features),
        param_grid(features),
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=config.SEED),
        n_jobs=config.N_JOBS,
        refit=True,
        verbose=1,
        error_score="raise",
    )
    grid.fit(X_train, y_train)

    results = pd.DataFrame(grid.cv_results_)
    results["param_clf"] = results["param_clf"].map(lambda c: type(c).__name__)
    columns = ["rank_test_score", "mean_test_score", "std_test_score", "mean_fit_time",
               "param_clf", "param_clf__C", f"param_{prefix}ngram_range", f"param_{prefix}min_df"]
    results.sort_values("rank_test_score")[columns].to_csv(config.METRICS / f"E3_{features}_cv_results.csv", index=False)

    best = grid.best_params_
    kind = "logreg" if isinstance(best["clf"], LogisticRegression) else "linear_svm"
    tfidf = {"ngram_range": best[f"{prefix}ngram_range"], "min_df": best[f"{prefix}min_df"]}
    C = float(best["clf__C"])
    cv_f1 = float(grid.best_score_)
    settings = describe_settings(C, features, **tfidf)
    print(f"\nBest CV macro F1 = {cv_f1:.4f}: {kind}, {settings}")

    evaluate_on_validation("E3", kind, features, grid.best_estimator_, settings, classes, X_val, y_val,
                           "Best candidate (tuned)", cv_f1)

    calibrated = kind == "linear_svm"
    if calibrated:
        # The API must return a real probability, so the deployed SVM is calibrated.
        cal_pipeline = build_pipeline(kind, C=C, calibrated=True, features=features, **tfidf)
        cal_pipeline.fit(X_train, y_train)
        evaluate_on_validation("E3-cal", f"{kind}_calibrated", features, cal_pipeline,
                               f"{settings}; sigmoid calibration cv=3", classes, X_val, y_val,
                               "Best candidate with calibrated probabilities", cv_f1)

    frozen = {
        "kind": kind,
        "C": C,
        "features": features,
        "tfidf": {**tfidf, "ngram_range": list(tfidf["ngram_range"])},
        "calibrated": calibrated,
        "cv_macro_f1": cv_f1,
        "label_map_version": load_manifest()["label_map_version"],
        "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    save_json(frozen, config.FROZEN_CONFIG_PATH)
    print(f"\nFrozen configuration written to {config.FROZEN_CONFIG_PATH}")


if __name__ == "__main__":
    main()
