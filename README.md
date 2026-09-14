# Mentor AI — Emotion Classifier

A supervised NLP classifier that reads a user's journal entry, check-in, or message and predicts an emotion class with a
calibrated confidence. The prediction is passed to the Mentor AI LLM as structured context; the LLM still writes the
reply. **This is emotion classification of text, not a medical or psychological diagnosis.**

Built following `Mentor_AI_Local_Model_Training_Guide.pdf` on Fedora 44 (plan: `PLAN.md`).

```
React / PWA
    │
Node.js / Express (mentor-ai/apps/api)  ──►  PostgreSQL
    │  POST /predict (1.5 s timeout, optional)
Python / FastAPI  (this repo, 127.0.0.1:8001)
    │
TF-IDF + calibrated LinearSVM pipeline (models/emotion_classifier.joblib)
    │
emotion + confidence ──► Mentor AI system prompt ──► LLM ──► final response
```

## Setup (Fedora)
```bash
sudo dnf install -y uv python3.12
cd ~/Coding/python
uv sync                        # creates .venv from uv.lock (Python 3.12)
uv run python -m ipykernel install --user --name mentor-ai-ml   # optional, for notebooks
```
No GPU or CUDA is needed. `requirements.txt` is exported from `uv.lock` for pip users.

## Reproduce the whole pipeline
```bash
make all        # data -> train -> tune -> evaluate -> test
```
| Step | Command | Writes |
|---|---|---|
| Prepare data | `make data` | `data/processed/*.parquet`, `manifest.json` (SHA-256 hashes that freeze the splits) |
| Baselines E1/E2 | `make train` | validation metrics, confusion matrices, rows in `reports/metrics/experiments.csv` |
| Tuning E3 | `make tune` | `E3_cv_results.csv`, `models/frozen_config.json` |
| Final test (once) | `make evaluate` | `models/emotion_classifier.joblib`, `model_card.json`, `final_test.json` |
| Tests | `make test` | — |
| Predict | `uv run python -m src.predict "some text"` | — |
| Serve | `make serve` | FastAPI on `127.0.0.1:8001` |

`make evaluate` refuses to run a second time. Reusing the test set would be test-set tuning, so it needs
`--force --reason "..."`, and the reason is logged.

Notebooks: `notebooks/01_dataset_exploration.ipynb` (dataset inspection) and `notebooks/02_model_experiments.ipynb`
(experiment comparison). Both read their numbers from the generated files.

## Data and labels
GoEmotions (CC BY 4.0) is mapped to a 7-class Mentor AI taxonomy: `joy`, `sadness`, `fear_anxiety`, `anger_frustration`,
`guilt`, `motivation`, `neutral`. Only examples whose labels all map to one class are kept; the final split sizes are
26,947 train, 3,359 validation, and 3,400 test.

The guide's suggested `anger` and `frustration` classes were merged, because `frustration` reached a validation F1 of only 0.27–0.29.
The evidence and decision are in `data/label_mapping.md`, and the v1 results are archived in `reports/*/label_map_v1/`.
Filtering details are in `data/README.md`.

## Results (label map v2; all numbers measured)
Leakage-free protocol:
- TF-IDF lives inside the sklearn `Pipeline`, so it is only fitted on training text.
- Model selection uses 3-fold cross-validation on train and a check on validation.
- The test set was used exactly once, for the final model.

| ID | Model | Settings | Split | Accuracy | Macro P | Macro R | Macro F1 | Weighted F1 | Purpose |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| E1 | Logistic Regression | TF-IDF (1,2), min_df=2, C=1, balanced | validation | 0.7312 | 0.6198 | 0.6871 | 0.6483 | 0.7360 | Baseline |
| E2 | Linear SVM | TF-IDF (1,2), min_df=2, C=1, balanced | validation | 0.7502 | 0.6833 | 0.6635 | 0.6722 | 0.7487 | Comparison |
| E3 | Linear SVM (tuned) | TF-IDF (1,2), min_df=1, C=0.5, balanced; CV macro F1 0.6356 | validation | 0.7681 | 0.7059 | 0.6886 | 0.6962 | 0.7665 | Best candidate |
| E3-cal | E3 + sigmoid calibration | as E3, `CalibratedClassifierCV(cv=3)` | validation | 0.7705 | 0.7648 | 0.6418 | 0.6920 | 0.7636 | Deployable (real probabilities) |
| **E3-final** | **E3-cal** | frozen configuration | **test** | **0.7668** | **0.7481** | **0.6238** | **0.6720** | **0.7562** | **Final model** |

Grid for E3: both classifiers × ngram {(1,1),(1,2)} × min_df {1,2,5} × C {0.5,1,2}, giving 36 candidates and 108 fits in about 2 minutes on CPU.

Per-class results of the final model on the test set:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| joy | 0.8628 | 0.8575 | 0.8601 | 814 |
| neutral | 0.7465 | 0.8897 | 0.8118 | 1,605 |
| fear_anxiety | 0.6957 | 0.6000 | 0.6443 | 80 |
| guilt | 0.7170 | 0.5672 | 0.6333 | 67 |
| sadness | 0.7794 | 0.4953 | 0.6057 | 107 |
| motivation | 0.7500 | 0.4943 | 0.5959 | 176 |
| anger_frustration | 0.6855 | 0.4628 | 0.5525 | 551 |

Observations:
- **Calibration trades recall for precision.** Macro precision went from 0.706 to 0.765 and recall from 0.689 to 0.642 on validation, with almost the same macro F1. The benefit is a real probability for the API.
- **Minority classes are under-detected.** `sadness`, `motivation`, and `anger_frustration` have recall below 0.5 on test, while `neutral` recall is 0.89. The confusion matrices are in `reports/figures/confusion_matrix_final_test*.png`.
- **Validation and test agree.** Test macro F1 (0.672) is close to validation (0.692), so the model generalizes.
- **Weak spot outside the dataset's style:** "Just finished my run, feeling unstoppable" gets `neutral` at a confidence of 0.27. That is below the Node threshold of 0.4, so no emotion context is sent. This illustrates the domain gap between Reddit and journaling text.

## API
```bash
make serve
curl -s -X POST localhost:8001/predict -H 'Content-Type: application/json' \
  -d '{"text":"I wasted the whole day and regret it."}'
```
```json
{"emotion":"guilt","confidence":0.6214,"probabilities":{"anger_frustration":0.0895,"fear_anxiety":0.0057,"guilt":0.6214,
 "joy":0.0588,"motivation":0.0256,"neutral":0.1556,"sadness":0.0433},"model_version":"v2-linear_svm_calibrated-20260914"}
```
- `GET /health` returns the status, model version, and classes.
- Interactive docs are at `http://127.0.0.1:8001/docs`.
- Empty text or text longer than 5,000 characters returns 422.
- Set `MODEL_PATH` to serve a different model file.

Port 8001 is used because the Express API already uses 8000. To run the service in the background:
```bash
mkdir -p ~/.config/systemd/user && cp deploy/mentor-ml.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now mentor-ml
```

## Node.js integration (`~/Coding/JavaScript/mentor-ai/apps/api`)
- `src/config/ml.config.ts`: the `EMOTION_*` environment settings, registered as `config('ml')`.
- `src/services/emotion-classifier.service.ts`: calls `POST /predict` with a timeout, validates the response with Zod, and returns `null` on any failure or when confidence is below the threshold.
- `src/services/chat-agent.service.ts`: classifies the user's message **concurrently** with building the system prompt, then appends a "Detected Emotional Signal (… not a diagnosis)" section when a result is available.
- `.env.example`: add `EMOTION_CLASSIFIER_ENABLED=true` (off by default), `EMOTION_SERVICE_URL=http://127.0.0.1:8001`, `EMOTION_TIMEOUT_MS`, and `EMOTION_MIN_CONFIDENCE` to `apps/api/.env`.

**Not yet verified:** the backend was not typechecked or run end to end, because its dependencies were not installed. To verify:
1. Run `pnpm install && pnpm --filter @mentor-ai/api typecheck`.
2. Start the API with `pnpm --filter @mentor-ai/api dev` and send a chat message.
3. Check the log for `Emotion classified`.
4. Stop FastAPI and confirm the chat still streams a reply.

## Limitations
- **Domain shift:** GoEmotions is Reddit comments, while Mentor AI receives journal entries and check-ins.
- **Label noise:** crowd-sourced labels, with many short, ambiguous texts.
- **Mixed emotions:** single-class filtering removes about 37% of raw rows, so mixed-emotion text is under-represented.
- **Approximate classes:** `guilt` (remorse + embarrassment) and `motivation` (optimism + desire + pride).
- **Small test support:** `fear_anxiety` (80) and `guilt` (67) have wide uncertainty.
- **Bag-of-words limits:** TF-IDF ignores word order and context, for example sarcasm and negation scope.

## Future work
- Fine-tune DistilBERT on the RTX 3050 (6 GB) against the same frozen splits.
- Collect a small labeled dataset of real (consented, anonymized) Mentor AI check-ins.
- Add emotion context to voice sessions, and store per-message emotions for trend insights.

## Definition of Done (guide §23)
- [x] Dataset and label mapping documented (`data/README.md`, `data/label_mapping.md`)
- [x] Training is reproducible (`make all`, fixed seed, split hashes verified identical across reruns)
- [x] At least two models compared (E1 LogReg, E2 LinearSVM, E3 tuned)
- [x] Test set protected until final evaluation (single run, guarded)
- [x] Metrics and confusion matrices saved (`reports/`)
- [x] Final model serialized (`models/emotion_classifier.joblib` + `model_card.json`)
- [x] Local prediction API working (pytest: 13 passed; live curl checks)
- [ ] Node.js integration working (code written; typecheck and live run pending)
- [ ] Mentor AI receives emotion context (pending the live run above)
- [x] Actual measured results and limitations documented (this file)
