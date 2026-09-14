# Mentor AI — Emotion Classifier

A supervised NLP classifier that reads a user's journal entry, check-in, or message and predicts one of seven emotion classes
with a calibrated confidence. The prediction is passed to the Mentor AI LLM as structured context; the LLM still writes the
reply. **This is emotion classification of text, not a medical or psychological diagnosis.**

Built following `Mentor_AI_Local_Model_Training_Guide.pdf` on Fedora 44 (plan: `PLAN.md`).
**Full FYP write-up with methodology, figures and results:** `reports/Mentor_AI_Emotion_Classifier_Report.pdf`.

```
React / PWA
    │
Node.js / Express (mentor-ai/apps/api)  ──►  PostgreSQL
    │  POST /predict (1.5 s timeout, optional, fails open)
Python / FastAPI  (this repo, 127.0.0.1:8001)
    │
DistilBERT (default)  or  TF-IDF + calibrated LinearSVM (fallback)
    │
emotion + confidence ──► Mentor AI system prompt ──► LLM ──► final response
```

Classes: `joy`, `sadness`, `fear_anxiety`, `anger_frustration`, `guilt`, `motivation`, `neutral`.

## Setup (Fedora)
```bash
sudo dnf install -y uv python3.12
cd ~/Coding/python
uv sync --extra transformer    # Python 3.12 venv from uv.lock, incl. CUDA 13 PyTorch for DistilBERT
uv run python -m ipykernel install --user --name mentor-ai-ml   # optional, for notebooks
```
Without `--extra transformer` everything still works using the TF-IDF model (no GPU needed). `requirements.txt` is
exported from `uv.lock` for pip users. Model weights and processed data are not in git; the commands below rebuild them.

## Reproduce
```bash
make all                 # data -> train -> tune -> evaluate -> test   (TF-IDF models, ~10 min on CPU)
make transformer         # E4: fine-tune DistilBERT on the GPU        (~11 min on an RTX 3050)
make evaluate-transformer
make domain-eval         # score on the Mentor-style sentences
uv run python -m src.report   # regenerate the PDF report from the saved results
```
| Step | Command | Writes |
|---|---|---|
| Prepare data | `make data` | `data/processed/*.parquet`, `manifest.json` (SHA-256 hashes that freeze the splits) |
| Baselines E1/E2 | `make train` | validation metrics, confusion matrices, rows in `reports/metrics/experiments.csv` |
| Tuning E3 | `make tune` | CV results, `models/frozen_config.json` |
| Final test (once per model) | `make evaluate` | `models/emotion_classifier.joblib`, `model_card.json`, `final_test_<version>.json` |
| DistilBERT E4 | `make transformer` | `models/distilbert/` (weights, tokenizer, `model_card.json` incl. temperature) |
| Domain eval | `make domain-eval` | `reports/metrics/domain_eval_<version>.json` |
| Tests | `make test` | — |

The test set is used **once per model version**. `src/evaluate.py` refuses to reuse it without `--force --reason "..."`,
and every reason is logged in `reports/metrics/forced_reruns.log`. Models are chosen on validation and on the Mentor-domain
set, never on test scores.

## Results (all numbers measured)
### Model comparison
| Model | Validation macro F1 | Test macro F1 | Test accuracy | Test ECE | Mentor-domain macro F1 |
|---|---:|---:|---:|---:|---:|
| TF-IDF words (fs1) + LinearSVM, sigmoid | 0.692 | 0.672 | 0.767 | — | 0.389 |
| TF-IDF words + emoji + char n-grams (fs2) + LinearSVM, isotonic | 0.711 | 0.686 | 0.779 | 0.026 | 0.431 |
| **DistilBERT, temperature-calibrated (deployed)** | **0.728** | **0.717** | **0.787** | **0.015** | **0.602** |

The Mentor-domain column is scored on 154 check-in style sentences in `data/mentor_eval/`. **Their labels are still an
unreviewed draft**, so treat that column as provisional until the review is done (see its README). It shows the real
story: models trained on Reddit comments struggle with journal-style text, and DistilBERT closes a large part of that gap.

All experiments on validation:

| ID | Features | Model | Validation macro F1 | CV macro F1 |
|---|---|---|---:|---:|
| E1 | fs1 | Logistic Regression | 0.648 | |
| E2 | fs1 | LinearSVM | 0.672 | |
| E3 | fs1 | LinearSVM, tuned (C=0.5, min_df=1) | 0.696 | 0.636 |
| E1 | fs2 | Logistic Regression | 0.675 | |
| E2 | fs2 | LinearSVM | 0.697 | |
| E3 | fs2 | LinearSVM, tuned (C=0.25, min_df=1) | 0.715 | 0.663 |
| E3-iso | fs2 | E3 + isotonic calibration (TF-IDF fallback) | 0.711 | 0.663 |
| E4 | — | DistilBERT, 4 epochs, best epoch 2, fp16 | 0.728 | |

### Deployed model (DistilBERT) — test set, per class
| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| joy | 0.840 | 0.926 | 0.881 | 814 |
| neutral | 0.844 | 0.791 | 0.817 | 1,605 |
| fear_anxiety | 0.643 | 0.788 | 0.708 | 80 |
| guilt | 0.653 | 0.731 | 0.690 | 67 |
| motivation | 0.609 | 0.733 | 0.665 | 176 |
| anger_frustration | 0.682 | 0.632 | 0.656 | 551 |
| sadness | 0.612 | 0.589 | 0.600 | 107 |

### Choosing the Node confidence threshold
A wrong emotion in the prompt is worse than none. For DistilBERT on the test set, this is how often a non-neutral emotion
would be sent, and how often it would be right:

| `EMOTION_MIN_CONFIDENCE` | Messages that get context | Precision of that context |
|---:|---:|---:|
| 0.4 | 55% | 0.752 |
| 0.5 | 51% | 0.778 |
| **0.6 (recommended)** | 46% | 0.818 |
| 0.7 | 42% | 0.856 |

On the Mentor-domain sentences at 0.6, 49% of messages get context and 79% of those are correct.

### Known weaknesses of the deployed model
- **Situational anxiety.** "Presentation is tomorrow and my stomach is in knots" is often read as `neutral`.
- **Self-directed negative states.** `guilt` gets confused with `anger_frustration` and `sadness`, for example "Forgot my daughter's recital. Worst dad ever."
- **Emoji-heavy text** is right only about half the time on the domain set.

## Test it locally
```bash
make predict                                            # guide's example sentences
uv run --extra transformer python -m src.predict "I feel guilty I skipped my workout again"
make serve                                              # API on 127.0.0.1:8001, docs at /docs
curl -s -X POST localhost:8001/predict -H 'Content-Type: application/json' \
  -d '{"text":"I wasted the whole day and regret it."}'
make domain-eval                                        # every error on the Mentor-style sentences
make test                                               # data hygiene, leakage guard, API contract
```
`/predict` returns `{emotion, confidence, probabilities, model_version}`. `/health` reports the loaded model. Empty text,
or text longer than 5,000 characters, returns 422.

To choose a model explicitly, set `MODEL_PATH`: `MODEL_PATH=models/emotion_classifier.joblib make serve`. Without it, the
service loads DistilBERT when it has been trained and PyTorch is installed, and the TF-IDF model otherwise.

Measured latency per message:

| Model | Latency | Memory |
|---|---|---|
| DistilBERT, RTX 3050 | 4 ms | |
| DistilBERT, CPU | 13 ms | about 1 GB RAM |
| TF-IDF | 2 ms | |

To run the service in the background:
```bash
mkdir -p ~/.config/systemd/user && cp deploy/mentor-ml.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now mentor-ml
```

## Node.js integration (`~/Coding/JavaScript/mentor-ai/apps/api`)
- `src/services/emotion-classifier.service.ts` calls `POST /predict` with a timeout and validates the response with Zod. It returns `null` on any failure or when confidence is below the threshold.
- `src/services/chat-agent.service.ts` classifies the message **concurrently** with building the system prompt, then appends a "Detected Emotional Signal (… not a diagnosis)" section.
- `src/config/ml.config.ts` holds the settings. Enable it in `apps/api/.env`:
  ```
  EMOTION_CLASSIFIER_ENABLED=true
  EMOTION_SERVICE_URL=http://127.0.0.1:8001
  EMOTION_MIN_CONFIDENCE=0.6
  ```

**Not yet verified:** the backend has not been typechecked or run end to end, because its dependencies are not installed.
To verify:
1. Run `pnpm install && pnpm --filter @mentor-ai/api typecheck`.
2. With `make serve` running, start `pnpm --filter @mentor-ai/api dev` and send a chat message.
3. Check the API log for `Emotion classified`.
4. Stop FastAPI and confirm the chat still replies.

## Data and labels
- **Dataset:** GoEmotions (CC BY 4.0), mapped to 7 classes. Only examples whose labels all map to one class are kept, giving 26,947 train, 3,359 validation and 3,400 test examples.
- **Merged class:** `anger` and `frustration` were merged into `anger_frustration`, because `frustration` had a validation F1 of only 0.27–0.29.
- **Details:** `data/label_mapping.md` and `data/README.md`.

## Limitations
- **Domain shift:** the training data is Reddit comments, while users write journal entries and check-ins.
- **Label noise:** crowd-sourced annotations.
- **Approximate classes:** `guilt` and `motivation` are approximations of the intended emotions.
- **Small test support:** `fear_anxiety` (80) and `guilt` (67) have wide uncertainty.
- **Unreviewed domain set:** the Mentor-domain labels are not yet human-reviewed.

## Future work
1. Review and extend the domain set with consented, anonymised real check-ins, and fine-tune on part of it.
2. Add voice-session support.
3. Store per-message emotion trends.

## Definition of Done (guide §23)
- [x] Dataset and label mapping documented
- [x] Training reproducible (`make all` / `make transformer`, fixed seed, split hashes verified identical across rebuilds)
- [x] At least two models compared (E1–E3 on two feature sets, E4 DistilBERT)
- [x] Test set protected (one guarded, logged evaluation per model version)
- [x] Metrics and confusion matrices saved (`reports/`)
- [x] Final model serialized (`models/distilbert/`, fallback `models/emotion_classifier.joblib`, both with model cards)
- [x] Local prediction API working (tests and live checks pass)
- [ ] Node.js integration working (code written; typecheck and live run pending)
- [ ] Mentor AI receives emotion context (pending that live run)
- [x] Measured results and limitations documented (this file + PDF report)
