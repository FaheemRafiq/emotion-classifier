.PHONY: setup data train tune evaluate predict serve test all transformer evaluate-transformer domain-eval data-v3 train-v3 evaluate-v3

setup:
	uv sync

data:
	uv run python -m src.prepare_data

train:
	uv run python -m src.train

tune:
	uv run python -m src.tune

evaluate:
	uv run python -m src.evaluate

# predict/serve/test use the `transformer` extra so the deployed DistilBERT model can load
# (they fall back to the TF-IDF joblib model when models/distilbert is absent).
predict:
	uv run --extra transformer python -m src.predict

serve:
	uv run --extra transformer uvicorn app.main:app --host 127.0.0.1 --port 8001

test:
	uv run --extra transformer pytest -q

# Exact training order (guide §21): data -> baselines -> tuning -> single test evaluation -> tests
all: data train tune evaluate test

# Optional transformer phase (guide §22). Needs the `transformer` extra: uv sync --extra transformer
transformer:
	uv run --extra transformer python -m src.train_transformer

evaluate-transformer:
	uv run --extra transformer python -m src.evaluate --model models/distilbert --force --reason "test evaluation of the transformer model version"

# Mentor-domain acceptance test (data/mentor_eval). MODEL=models/distilbert to score the transformer.
domain-eval:
	uv run --extra transformer python -m src.eval_domain --show-errors $(if $(MODEL),--model $(MODEL),)

# v3: more and harder training data (GoEmotions rows v2 dropped, mixed emotions as soft labels, ISEAR)
# + DeBERTa-v3-base. The v2 validation/test splits stay frozen, so results are directly comparable.
data-v3:
	uv run python -m src.prepare_extra

train-v3:
	HF_HUB_OFFLINE=$${HF_HUB_OFFLINE:-0} uv run --extra transformer python -m src.train_v3 --exp-id E6 --out models/deberta_v3

# one-time test evaluation of the chosen v3 model + every "complex emotion" benchmark, against DistilBERT
evaluate-v3:
	uv run --extra transformer python -m src.evaluate_v3 --model models/deberta_v3
