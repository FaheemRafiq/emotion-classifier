"""Central configuration: paths, seed, label mapping, and model defaults."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MANIFEST_PATH = DATA_PROCESSED / "manifest.json"

MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "emotion_classifier.joblib"
MODEL_CARD_PATH = MODELS_DIR / "model_card.json"
FROZEN_CONFIG_PATH = MODELS_DIR / "frozen_config.json"

REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
METRICS = REPORTS / "metrics"
EXPERIMENTS_CSV = METRICS / "experiments.csv"
FINAL_TEST_JSON = METRICS / "final_test.json"

SEED = 42
# Capped (not -1): this laptop has 15 GiB RAM and is usually under memory pressure.
N_JOBS = int(os.environ.get("N_JOBS", "4"))

DATASET_NAME = "google-research-datasets/go_emotions"
DATASET_CONFIG = "simplified"
SPLITS = ("train", "validation", "test")

# Mentor AI taxonomy -> GoEmotions source labels. Bump LABEL_MAP_VERSION on any change
# and record the reason in data/label_mapping.md.
# v2: v1's `anger` and `frustration` merged into `anger_frustration` (frustration validation F1 was 0.27-0.29).
LABEL_MAP_VERSION = "v2"
MENTOR_LABEL_MAP: dict[str, list[str]] = {
    "joy": ["joy", "amusement", "excitement", "gratitude", "relief", "love"],
    "sadness": ["sadness", "grief"],
    "fear_anxiety": ["fear", "nervousness"],
    "anger_frustration": ["anger", "disgust", "annoyance", "disappointment"],
    "guilt": ["remorse", "embarrassment"],
    "motivation": ["optimism", "desire", "pride"],
    "neutral": ["neutral"],
}
MENTOR_CLASSES = list(MENTOR_LABEL_MAP)

# Classes below this many training examples are flagged for merge/drop review.
MIN_TRAIN_SUPPORT = 800

TFIDF_DEFAULTS = {
    "lowercase": True,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": True,
}

# Feature sets. fs1 = the guide's word TF-IDF; its default token pattern (\b\w\w+\b) silently drops emojis
# and punctuation. fs2 keeps them as tokens and adds character n-grams (typos, informal spelling).
FEATURES_VERSION = "fs2"
WORD_TOKEN_PATTERN = r"(?u)\b\w+\b|[^\w\s]"
CHAR_TFIDF = {
    "analyzer": "char_wb",
    "ngram_range": (2, 5),
    "min_df": 2,
    "sublinear_tf": True,
    "max_features": 300_000,
}

# Optional transformer (E4)
TRANSFORMER_NAME = "distilbert-base-uncased"
TRANSFORMER_DIR = MODELS_DIR / "distilbert"

DOMAIN_EVAL_CSV = ROOT / "data" / "mentor_eval" / "mentor_eval.csv"

MAX_TEXT_CHARS = 5000
