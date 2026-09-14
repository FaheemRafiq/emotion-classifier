"""Local inference with a saved model.

Usage: uv run python -m src.predict ["some text" ...] [--model models/distilbert]
"""
import argparse

from src.inference import load_predictor, predict_texts

GUIDE_EXAMPLES = [
    "I finally finished my project and feel amazing.",
    "I wasted the whole day and regret it.",
    "I am nervous about tomorrow.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("texts", nargs="*", default=GUIDE_EXAMPLES)
    parser.add_argument("--model", help="joblib file or transformer directory (default: MODEL_PATH env or models/emotion_classifier.joblib)")
    args = parser.parse_args()

    predictor = load_predictor(args.model)
    print(f"model: {predictor.model_version}")
    for result in predict_texts(predictor, args.texts):
        line = f"{result['text']!r} => {result['emotion']}"
        if result["probabilities"]:
            top = sorted(result["probabilities"].items(), key=lambda kv: kv[1], reverse=True)[:3]
            line += f" (confidence {result['confidence']:.2f}; top-3: " + ", ".join(f"{c} {p:.2f}" for c, p in top) + ")"
        print(line)


if __name__ == "__main__":
    main()
