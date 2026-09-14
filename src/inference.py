"""One prediction interface for both model families, so the API and evaluation code don't care which is loaded.

- A `.joblib` file  -> scikit-learn pipeline (TF-IDF + classifier)
- A directory       -> Hugging Face transformer saved by src/train_transformer.py
"""
import json
import os
from pathlib import Path

import numpy as np

from src import config


def _read_card(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


class SklearnPredictor:
    def __init__(self, path: Path):
        import joblib

        self.path = path
        self.model = joblib.load(path)
        self.classes = [str(c) for c in self.model.classes_]
        self.model_version = _read_card(path.with_name(config.MODEL_CARD_PATH.name)).get("model_version", "unknown")
        self.has_probabilities = hasattr(self.model, "predict_proba")

    def predict_proba(self, texts: list[str]) -> np.ndarray | None:
        return self.model.predict_proba(texts) if self.has_probabilities else None

    def predict(self, texts: list[str]) -> list[str]:
        return [str(label) for label in self.model.predict(texts)]


class TransformerPredictor:
    def __init__(self, path: Path, batch_size: int = 64):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.path = path
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device).eval()
        self.classes = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
        card = _read_card(path / "model_card.json")
        self.model_version = card.get("model_version", "unknown")
        self.temperature = float(card.get("temperature", 1.0))
        self.max_length = int(card.get("max_length", 128))
        self.has_probabilities = True

    def logits(self, texts: list[str]) -> np.ndarray:
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                batch = self.tokenizer(
                    list(texts[i : i + self.batch_size]),
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                out.append(self.model(**batch).logits.float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.classes)))

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        z = self.logits(texts) / self.temperature
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, texts: list[str]) -> list[str]:
        return [self.classes[i] for i in self.logits(texts).argmax(axis=1)]


Predictor = SklearnPredictor | TransformerPredictor


def default_model_path() -> Path:
    """MODEL_PATH env if set; else the fine-tuned DistilBERT if trained and PyTorch is installed; else the joblib model."""
    if os.environ.get("MODEL_PATH"):
        return Path(os.environ["MODEL_PATH"])
    if (config.TRANSFORMER_DIR / "model_card.json").exists():
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401

            return config.TRANSFORMER_DIR
        except ImportError:
            print("DistilBERT found but the `transformer` extra is not installed; serving the TF-IDF model instead.")
    return config.MODEL_PATH


def load_predictor(path: str | os.PathLike | None = None) -> Predictor:
    path = Path(path) if path else default_model_path()
    if path.is_dir():
        return TransformerPredictor(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `make evaluate` (or `make transformer`) first")
    return SklearnPredictor(path)


def predict_texts(predictor: Predictor, texts: list[str]) -> list[dict]:
    """Labels plus probabilities/confidence when the model provides calibrated ones."""
    texts = list(texts)
    probas = predictor.predict_proba(texts)
    labels = predictor.predict(texts) if probas is None else [predictor.classes[i] for i in probas.argmax(axis=1)]

    results = []
    for i, text in enumerate(texts):
        result = {"text": text, "emotion": labels[i], "confidence": None, "probabilities": None}
        if probas is not None:
            result["probabilities"] = {c: float(p) for c, p in zip(predictor.classes, probas[i])}
            result["confidence"] = float(probas[i].max())
        results.append(result)
    return results
