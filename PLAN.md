# Mentor AI — Emotion Classifier: End-to-End Local Training Plan (Fedora 44)

## Context
`Mentor_AI_Local_Model_Training_Guide.pdf` specifies a supervised NLP emotion classifier for Mentor AI (FYP):
user text → emotion label (+ confidence) → passed as structured context to the existing generative
Mentor AI layer via a Node.js/Express backend. `/home/faheem/Coding/python` currently contains only the PDF,
so the whole project is built from scratch. The guide is written for Windows; this plan adapts it to Fedora.

**Decisions (confirmed with the user):**
- Classical models only: TF-IDF + Logistic Regression and Linear SVM. DistilBERT is noted as future work.
- Environment: `uv` with Python 3.12 (system Python 3.14 is left untouched).
- Labels: map the GoEmotions labels onto the suggested Mentor taxonomy, measure how many examples each class gets, then merge or drop weak classes. Every decision is documented.
- Node: integrate with the real Mentor AI backend at `/home/faheem/Coding/JavaScript/mentor-ai/apps/api` (see §9).

**Machine facts:** RTX 3050 6 GB (not needed), 12 cores, 15 GiB RAM (heavily used, swap nearly full, so cap
parallel jobs), 419 GB free disk, Node v24. `python3.12` and `uv` are available from Fedora `updates`.

**Execution step 0:** copy this plan to `/home/faheem/Coding/python/PLAN.md`, as the user requested.

---

## 1. Environment setup (Fedora)
```bash
sudo dnf install -y uv python3.12
cd /home/faheem/Coding/python
uv init --name mentor-ai-ml --python 3.12 --no-readme   # creates pyproject.toml
uv venv --python 3.12 .venv
uv add pandas numpy scikit-learn matplotlib seaborn joblib fastapi "uvicorn[standard]" datasets pyarrow
uv add --dev jupyter ipykernel pytest httpx
uv export --no-hashes --format requirements-txt > requirements.txt   # guide requires requirements.txt
uv run python -m ipykernel install --user --name mentor-ai-ml
```
- `uv.lock` plus `requirements.txt` make the setup reproducible. Also add `.gitignore` (`.venv/`, `data/raw/`, `models/*.joblib`, HF cache) and `git init`.
- Fedora notes: no CUDA or GPU setup is needed. Serve on `127.0.0.1:8001` (8000 is used by the Mentor AI Express API), so firewalld and SELinux need no changes (only if the API must be reachable from another host: `sudo firewall-cmd --add-port=8001/tcp`).

## 2. Project structure (guide §4, lightly extended)
```
python/                      (project root = current directory)
├── pyproject.toml, uv.lock, requirements.txt, README.md, PLAN.md, Makefile, .gitignore
├── data/{raw/, processed/, README.md, label_mapping.md}
├── notebooks/01_dataset_exploration.ipynb, 02_model_experiments.ipynb
├── src/
│   ├── config.py          # paths, SEED=42, MENTOR_LABEL_MAP, TF-IDF defaults, N_JOBS=4
│   ├── prepare_data.py    # download → map → clean → frozen splits
│   ├── train.py           # E1 LogReg, E2 LinearSVM (fit on train, score on validation)
│   ├── tune.py            # GridSearchCV on train only → E3 best candidate
│   ├── evaluate.py        # ONE-TIME final test-set evaluation of the frozen model
│   ├── metrics.py         # shared: metric dict, classification report, confusion matrix plot
│   └── predict.py         # CLI: load joblib pipeline, predict text(s)
├── models/emotion_classifier.joblib + model_card.json (labels, params, data hash, versions)
├── reports/{figures/, metrics/experiments.csv, metrics/*.json}
├── app/main.py            # FastAPI service
└── tests/{test_data.py, test_pipeline.py, test_api.py}
```
All scripts run as `uv run python -m src.<name>`. The `Makefile` targets (`setup, data, train, tune, evaluate, serve, test`) chain the exact training order from guide §21.

## 3. Data: load and inspect (guide §5–6)
- `load_dataset("google-research-datasets/go_emotions", "simplified")`, cached into `data/raw/`. It has official train/validation/test splits, a multi-label `labels` list, and 27 emotions plus neutral.
- Notebook 01 covers: split sizes, label names/IDs, per-label counts, empty or missing text, duplicates, multi-label rate, sample texts per class, and noisy/ambiguous examples. Figures go to `reports/figures/`.

## 4. Label mapping (guide §5, decision: map, then drop or merge weak classes)
Candidate mapping in `src/config.py` (`MENTOR_LABEL_MAP`), explained in `data/label_mapping.md`:

| Mentor class | GoEmotions sources (candidate) |
|---|---|
| joy | joy, amusement, excitement, gratitude, relief, love |
| sadness | sadness, grief |
| fear_anxiety | fear, nervousness |
| anger | anger, disgust |
| frustration | annoyance, disappointment |
| guilt | remorse, embarrassment |
| motivation | optimism, desire, pride |
| neutral | neutral |
| *(dropped)* | admiration, approval, caring, confusion, curiosity, disapproval, realization, surprise |

Rules:
- Keep an example only if all its labels map to the **same** Mentor class. Mixed or conflicting examples and examples with only dropped labels are removed, and the removed counts are logged.
- Measure the training support for each class. If a class has fewer than about 800 training examples, or its validation F1 in E1 is very poor, merge it (for example frustration → anger) or drop it. Record the decision and the real counts in `label_mapping.md`. No numbers are written before they are measured.
- If `neutral` dominates, handle it with `class_weight="balanced"` first. Downsampling is a documented experiment only, never a silent change.

**Decision taken during execution (label map v2):** in v1, `fear_anxiety` (536) and `guilt` (559) were below the support
threshold but reached a validation F1 of 0.61–0.68, so they were kept. `frustration` had 2,232 examples but a validation F1 of only 0.27–0.29.
With the user's approval, `anger` and `frustration` were merged into **`anger_frustration`**, giving 7 classes. The v1 results are archived in
`reports/metrics/label_map_v1/`; the full rationale is in `data/label_mapping.md`.

## 5. Preprocessing and frozen splits (guide §7–8) — `src/prepare_data.py`
- Keep cleaning conservative: strip text, collapse repeated whitespace, and drop empty text and exact duplicates within each split. Also remove any train text that also appears in validation or test, so no text leaks across splits. Keep negations, punctuation, and emojis. Do not remove stop words and do not stem. GoEmotions already masks names as `[NAME]`, so that is left as is.
- Write `data/processed/{train,validation,test}.parquet` plus `data/processed/manifest.json` (row counts, class counts, SHA-256 of each file, mapping version). Later scripts check the hash, which "freezes" the split.
- Leakage guard: every model is a `sklearn.pipeline.Pipeline([("tfidf", TfidfVectorizer), ("clf", ...)])`, so TF-IDF is only ever fitted on the text passed to `fit`. `tests/test_pipeline.py` checks that the vectorizer vocabulary comes only from train.

## 6. Training and experiments (guide §9–10, §13, §19)
- **E1** `train.py`: TF-IDF(lowercase, ngram (1,2), min_df=2, max_df=0.95, sublinear_tf) + `LogisticRegression(max_iter=2000, class_weight="balanced")`. Fit on train, score on **validation**.
- **E2** `train.py`: same TF-IDF + `LinearSVC(C=1.0, class_weight="balanced")`. Fit on train, score on validation.
- **E3** `tune.py`: `GridSearchCV(pipeline, cv=StratifiedKFold(3, shuffle, SEED), scoring="f1_macro", n_jobs=N_JOBS=4)` on **train only**. This caps the guide's `n_jobs=-1` because of the low free RAM. The grid covers both classifiers:
  - `tfidf__ngram_range` [(1,1),(1,2)], `tfidf__min_df` [1,2,5], and `clf__C` [0.5,1,2] for each classifier.
  - The winner is chosen on the CV macro F1 and confirmed on validation. The configuration is then frozen.
- Each run appends a row to `reports/metrics/experiments.csv` (id, model, settings, CV/val accuracy, macro P/R/F1, weighted F1, timestamp, purpose), saves the per-class report JSON, and saves a confusion matrix PNG. This is shared code in `src/metrics.py`.
- If the winner is LinearSVC, wrap it in `CalibratedClassifierCV(cv=3)`, because the API must return a real probability (guide §10, §20 "fake confidence"). Report calibrated and uncalibrated validation F1 side by side.

## 7. Final evaluation and serialization (guide §11–12, §14–15)
- `evaluate.py` refits the frozen best pipeline on train, then evaluates on **test exactly once**. It writes `reports/metrics/final_test.json`, `reports/figures/confusion_matrix_final.png`, the per-class table, and E3's row in `experiments.csv`. The script refuses to rerun if `final_test.json` exists, unless `--force` is passed and a reason is logged.
- `joblib.dump(pipeline, "models/emotion_classifier.joblib")` plus `model_card.json` (classes, params, data manifest hash, sklearn/python versions, metrics).
- `predict.py` runs a smoke test with the guide's three example sentences.

## 8. FastAPI service (guide §16) — `app/main.py`
- Load the model once at startup, from a path read from the env var `MODEL_PATH`.
- `POST /predict {text}` → `{emotion, confidence, probabilities{label: p}, model_version}`. Reject empty or whitespace text with 422 and cap the text length.
- `GET /health` → status + model version. CORS is off by default, because only Node calls it.
- Run with `uv run uvicorn app.main:app --host 127.0.0.1 --port 8001` (add `--reload` for development).
- `tests/test_api.py` uses `fastapi.testclient` to check the health endpoint, a valid prediction, the empty-text 422, and that probabilities sum to 1.
- Optional: a `systemd --user` unit file (`deploy/mentor-ml.service`) so the service can run in the background on Fedora.

## 9. Node.js integration (guide §17–18) — `/home/faheem/Coding/JavaScript/mentor-ai/apps/api`
The backend is a pnpm monorepo with a TypeScript Express 5 API. Chat flow: `POST /api/chat` → `AIChatController.chat`
(`src/controllers/chat.controller.ts`) → `ChatAgentService.buildChatStream` (`src/services/chat-agent.service.ts`),
which rebuilds the system prompt on **every message** in `buildSystemPrompt()`. That is where the emotion context goes.
Follow `apps/api/AGENTS.md`: kebab-case files, classes, `Logger`, no `any`, `import type`, Zod.

1. **`src/config/ml.config.ts`** (new): read settings with the existing helpers in `src/config/env.ts`:
   `emotionServiceUrl: envString('EMOTION_SERVICE_URL', 'http://127.0.0.1:8001')`,
   `enabled: envBoolean('EMOTION_CLASSIFIER_ENABLED', false)`, `timeoutMs: envNumber('EMOTION_TIMEOUT_MS', 1500)`,
   `minConfidence: envNumber('EMOTION_MIN_CONFIDENCE', 0.4)`. Register it as `ml` in `src/config/index.ts`.
   It defaults to disabled, so the Railway deployment is unaffected until the Python service is deployed.
2. **`src/services/emotion-classifier.service.ts`** (new): `EmotionClassifierService.classify(text): Promise<EmotionResult | null>`.
   - Use Node 24's global `fetch` with `AbortSignal.timeout(timeoutMs)`, and validate the response with a Zod schema (`emotion`, `confidence`, `probabilities`, `model_version`).
   - On any error, timeout, or when disabled, log `logger.warning` and return `null`, so the chat never breaks.
3. **`src/services/chat-agent.service.ts`** (edit):
   - In `buildChatStream`, run `classify(message)` concurrently with `buildSystemPrompt(...)` using `Promise.all`, so no latency is added in series, and pass the result in.
   - When `confidence >= minConfidence`, `buildSystemPrompt` appends a section in the same style as the existing `User Memory` block:
     `Detected emotional signal (automated text classifier, may be wrong, not a diagnosis): guilt (confidence 0.84). Use it only to adapt tone and support; do not state the label to the user unless relevant.`
   - Log `{ emotion, confidence }` with `logger.info` as FYP evidence. No DB schema change.
4. **`apps/api/.env.example`**: add the four `EMOTION_*` variables with comments.
5. Out of scope, documented as future work: voice realtime sessions (`voice-session.service.ts`) and storing emotions per message in Prisma.
6. Quality gates from AGENTS.md: `pnpm --filter @mentor-ai/api typecheck` and `pnpm --filter @mentor-ai/api lint`. The API has no test runner, so verification is manual (see below).

## 10. Documentation (guide §19, §21, §23)
- `README.md`: Fedora setup, `make` pipeline, API usage, and architecture diagram.
- `data/README.md` and `label_mapping.md`: dataset license (CC BY 4.0) and filtering/mapping decisions with real counts.
- Results section written from the measured numbers only, covering limitations (Reddit domain vs. journaling text, label noise, weak classes, not a medical/psychological diagnosis) and future work (DistilBERT on the RTX 3050, domain-specific labeled data).

---

## Verification (end-to-end)
1. `uv run python -c "import sklearn, fastapi, datasets; print(sklearn.__version__)"` shows the environment works.
2. `make data` creates the parquet files and `manifest.json`, and a rerun gives identical hashes (reproducible).
3. `make train && make tune` fills `experiments.csv` with E1/E2/E3 rows and writes the confusion matrix PNGs.
4. `make evaluate` writes `final_test.json` once; a second run is refused.
5. `uv run python -m src.predict "I am nervous about tomorrow."` prints a sensible label.
6. `uv run pytest` passes the data, leakage, and API tests.
7. With `make serve` running, `curl -s -X POST localhost:8001/predict -H 'Content-Type: application/json' -d '{"text":"I wasted the whole day and regret it."}'` returns JSON with a confidence value. `/docs` loads in the browser.
8. In `mentor-ai`, set `EMOTION_CLASSIFIER_ENABLED=true` in `apps/api/.env` and run `pnpm --filter @mentor-ai/api typecheck && pnpm --filter @mentor-ai/api lint`.
   With `pnpm --filter @mentor-ai/api dev` running, send `POST /api/chat` (authenticated) with a message such as "I feel guilty I skipped my workout again".
   The API log shows `{ emotion, confidence }`, and a temporary `logger.debug` of the final system prompt in `buildSystemPrompt` shows the emotion section.
   (The `/api/admin/pipelines/preview` endpoint won't show it, because it builds from templates only and skips `ChatAgentService`.)
   With FastAPI stopped, the chat still streams a reply and only a warning is logged (graceful fallback).
9. Guide §23 "Definition of Done" checklist is ticked in README.
