import pytest

from src.train import build_pipeline, word_vectorizer

TRAIN_TEXTS = ["i feel so happy today 😀", "happy and glad about it", "i am very sad 😭", "sad and lonely again"] * 3
TRAIN_LABELS = ["joy", "joy", "sadness", "sadness"] * 3


@pytest.mark.parametrize("features", ["fs1", "fs2"])
def test_vectorizer_vocabulary_comes_only_from_training_text(features):
    pipeline = build_pipeline("logreg", features=features, min_df=1)
    pipeline.fit(TRAIN_TEXTS, TRAIN_LABELS)
    pipeline.predict(["zebra unseen words"])

    vocabulary = word_vectorizer(pipeline).vocabulary_
    assert "happy" in vocabulary
    assert "zebra" not in vocabulary


def test_fs2_keeps_emojis_as_tokens_and_fs1_does_not():
    fs1 = build_pipeline("logreg", features="fs1", min_df=1).fit(TRAIN_TEXTS, TRAIN_LABELS)
    fs2 = build_pipeline("logreg", features="fs2", min_df=1).fit(TRAIN_TEXTS, TRAIN_LABELS)
    assert "😭" not in word_vectorizer(fs1).vocabulary_
    assert "😭" in word_vectorizer(fs2).vocabulary_


@pytest.mark.parametrize(("kind", "calibrated"), [("logreg", False), ("linear_svm", False), ("linear_svm", True)])
def test_pipeline_fits_and_predicts(kind, calibrated):
    pipeline = build_pipeline(kind, calibrated=calibrated, min_df=1)
    pipeline.fit(TRAIN_TEXTS, TRAIN_LABELS)
    assert set(pipeline.predict(["so happy", "very sad"])) <= {"joy", "sadness"}


def test_only_uncalibrated_svm_lacks_probabilities():
    assert not hasattr(build_pipeline("linear_svm").fit(TRAIN_TEXTS, TRAIN_LABELS), "predict_proba")
    assert hasattr(build_pipeline("linear_svm", calibrated=True).fit(TRAIN_TEXTS, TRAIN_LABELS), "predict_proba")
    assert hasattr(build_pipeline("logreg").fit(TRAIN_TEXTS, TRAIN_LABELS), "predict_proba")
