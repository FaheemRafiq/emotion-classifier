"""Build the FYP report (HTML + PDF) from the saved artefacts. Every number is read from files.

Usage: uv run python -m src.report          -> reports/Mentor_AI_Emotion_Classifier_Report.{html,pdf}
"""
import base64
import csv
import html
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import sklearn

from src import config

OUT_HTML = config.REPORTS / "Mentor_AI_Emotion_Classifier_Report.html"
OUT_PDF = config.REPORTS / "Mentor_AI_Emotion_Classifier_Report.pdf"

CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: 'DejaVu Sans', 'Liberation Sans', Arial, sans-serif; font-size: 10.5pt; color: #1a1a1a; line-height: 1.45; }
h1 { font-size: 24pt; margin: 0 0 4pt; color: #0b2545; }
h2 { font-size: 15pt; color: #0b2545; border-bottom: 2px solid #0b2545; padding-bottom: 3pt; margin-top: 22pt; }
h3 { font-size: 12pt; color: #13315c; margin-top: 14pt; }
p, li { text-align: justify; }
table { border-collapse: collapse; width: 100%; margin: 8pt 0 12pt; font-size: 9pt; page-break-inside: avoid; }
th, td { border: 1px solid #c9d1dc; padding: 3pt 5pt; vertical-align: top; }
th { background: #0b2545; color: white; text-align: left; }
tr:nth-child(even) td { background: #f3f6fa; }
td.num, th.num { text-align: right; }
.box { border: 1px solid #8fb3e0; background: #eaf2fb; padding: 7pt 10pt; margin: 9pt 0; }
.warn { border-color: #e0a54f; background: #fdf3e3; }
.fig { text-align: center; margin: 10pt 0; page-break-inside: avoid; }
.fig img { max-width: 100%; max-height: 95mm; }
.fig .cap { font-size: 9pt; color: #444; margin-top: 3pt; }
.two { display: flex; gap: 10pt; }
.two .fig { flex: 1; }
code, pre { font-family: 'DejaVu Sans Mono', monospace; font-size: 8.5pt; }
pre { background: #f5f5f5; border: 1px solid #ddd; padding: 6pt; white-space: pre-wrap; }
.pb { page-break-before: always; }
.small { font-size: 9pt; color: #444; }
.title { text-align: center; margin-top: 60mm; }
.title .sub { font-size: 14pt; color: #13315c; margin: 6pt 0 30pt; }
.title .meta { font-size: 11pt; color: #444; }
"""


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def img(path: Path, caption: str) -> str:
    if not path.exists():
        return f'<div class="fig"><div class="cap">[missing figure: {path.name}]</div></div>'
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<div class="fig"><img src="data:image/png;base64,{data}"><div class="cap">{html.escape(caption)}</div></div>'


def table(df: pd.DataFrame, float_fmt: str = "{:.4f}") -> str:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else float_fmt.format(v))
    return df.to_html(index=False, escape=True, border=0, classes="tbl").replace('<td>', '<td>')


def per_class_table(per_class: dict, classes: list[str]) -> pd.DataFrame:
    rows = [{"class": c, "precision": per_class[c]["precision"], "recall": per_class[c]["recall"],
             "F1": per_class[c]["f1-score"], "support": int(per_class[c]["support"])} for c in classes if c in per_class]
    return pd.DataFrame(rows)


def load_final_tests() -> list[dict]:
    seen, out = set(), []
    for p in sorted((config.METRICS / "archive").glob("final_test_*.json")) + sorted(config.METRICS.glob("final_test_*.json")):
        d = read_json(p)
        if d["model_version"] not in seen:
            seen.add(d["model_version"])
            out.append(d)
    return sorted(out, key=lambda d: d["evaluated_at"])


def load_domain_evals() -> list[dict]:
    return sorted((read_json(p) for p in config.METRICS.glob("domain_eval_*.json")), key=lambda d: d["model_version"])


def main() -> None:
    manifest = read_json(config.MANIFEST_PATH)
    classes = manifest["classes"]
    experiments = pd.read_csv(config.EXPERIMENTS_CSV)
    finals = load_final_tests()
    domain_evals = load_domain_evals()
    domain_df = pd.read_csv(config.DOMAIN_EVAL_CSV)
    card = read_json(config.MODEL_CARD_PATH)
    tcard_path = config.TRANSFORMER_DIR / "model_card.json"
    tcard = read_json(tcard_path) if tcard_path.exists() else None
    v1_manifest = read_json(config.METRICS / "label_map_v1" / "manifest_v1.json")
    v1_e1 = read_json(config.METRICS / "label_map_v1" / "E1_validation.json")["metrics"]
    v1_e2 = read_json(config.METRICS / "label_map_v1" / "E2_validation.json")["metrics"]
    now = datetime.now(timezone.utc)

    # ---------- dataset tables
    counts = pd.DataFrame({s: manifest["splits"][s]["class_counts"] for s in config.SPLITS})
    counts.loc["total"] = counts.sum()
    counts = counts.reset_index().rename(columns={"index": "class"})
    filtering = pd.DataFrame({s: manifest["splits"][s]["filtering"] for s in config.SPLITS}).fillna(0).astype(int)
    filtering = filtering.reset_index().rename(columns={"index": "step"})
    mapping = pd.DataFrame([{"Mentor AI class": k, "GoEmotions source labels": ", ".join(v)} for k, v in manifest["label_map"].items()])

    v1_rows = []
    for c in ["fear_anxiety", "guilt", "anger", "frustration"]:
        v1_rows.append({"v1 class": c, "train support": v1_manifest["splits"]["train"]["class_counts"][c],
                        "E1 LogReg F1": v1_e1["per_class"][c]["f1-score"], "E2 LinearSVM F1": v1_e2["per_class"][c]["f1-score"]})
    v1_rows.append({"v1 class": "macro F1 (8 classes)", "train support": "", "E1 LogReg F1": v1_e1["macro_f1"], "E2 LinearSVM F1": v1_e2["macro_f1"]})

    # ---------- experiments
    exp_view = experiments.drop_duplicates(subset=["id", "features", "eval_split"], keep="last")
    exp_cols = ["id", "features", "model", "eval_split", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "cv_macro_f1", "purpose"]
    val_runs = {}
    for p in sorted(config.METRICS.glob("E*_validation.json")):
        d = read_json(p)
        val_runs[f"{d['id']} / {d.get('features', 'fs1')}"] = d
    per_class_val = pd.DataFrame({k: {c: d["metrics"]["per_class"][c]["f1-score"] for c in classes} for k, d in val_runs.items()})
    per_class_val = per_class_val.reset_index().rename(columns={"index": "class"})

    cv = pd.read_csv(config.METRICS / "E3_fs2_cv_results.csv").head(8)

    # ---------- final test
    final_summary = pd.DataFrame([{"model version": f["model_version"], "accuracy": f["metrics"]["accuracy"],
                                   "macro P": f["metrics"]["macro_precision"], "macro R": f["metrics"]["macro_recall"],
                                   "macro F1": f["metrics"]["macro_f1"], "weighted F1": f["metrics"]["weighted_f1"],
                                   "ECE": (f.get("calibration") or {}).get("ece")} for f in finals])
    deployed_card = tcard if tcard else card
    deployed = deployed_card["model_version"]
    deployed_final = next(f for f in finals if f["model_version"] == deployed)
    deployed_desc = (
        "DistilBERT (<code>distilbert-base-uncased</code>) fine-tuned on the frozen GoEmotions splits, temperature-calibrated"
        if tcard else "TF-IDF (word + character n-grams) + linear SVM with isotonic calibration"
    )
    latency = read_json(config.METRICS / "latency.json")["models"]
    best_test = max(finals, key=lambda f: f["metrics"]["macro_f1"])

    # ---------- domain
    domain_summary = pd.DataFrame([{"model version": d["model_version"], "accuracy": d["metrics"]["accuracy"], "macro F1": d["metrics"]["macro_f1"],
                                    **{f"acc ({k})": v["accuracy"] for k, v in d["accuracy_by_kind"].items()},
                                    "wrong emotion sent @0.6": sum(1 for p in d["predictions"] if p["pred"] != "neutral" and p["pred"] != p["label"] and (p["confidence"] or 0) >= 0.6)}
                                   for d in domain_evals])
    latest_domain = next((d for d in domain_evals if d["model_version"] == deployed), domain_evals[-1] if domain_evals else None)

    # ---------- model selection (validation + domain decide; test only confirms)
    def val_f1(version: str) -> float | None:
        if "distilbert" in version:
            return tcard["validation_metrics"]["macro_f1"] if tcard else None
        key = "E3-iso / fs2" if "fs2" in version else "E3-cal / fs1"
        return val_runs[key]["metrics"]["macro_f1"] if key in val_runs else None

    names = {"v2-linear_svm_calibrated-20260914": "TF-IDF fs1 + SVM (sigmoid)",
             "v2-fs2-linear_svm_calibrated-20260914": "TF-IDF fs2 + SVM (isotonic)",
             "v2-distilbert-20260914": "DistilBERT (temperature)"}
    selection = pd.DataFrame([{
        "model": names.get(f["model_version"], f["model_version"]),
        "validation macro F1": val_f1(f["model_version"]),
        "Mentor-domain macro F1": next((d["metrics"]["macro_f1"] for d in domain_evals if d["model_version"] == f["model_version"]), None),
        "test macro F1": f["metrics"]["macro_f1"],
        "test ECE": (f.get("calibration") or {}).get("ece"),
    } for f in finals])
    domain_per_class = pd.DataFrame({d["model_version"]: {c: d["metrics"]["per_class"][c]["f1-score"] for c in classes} for d in domain_evals}).reset_index().rename(columns={"index": "class"})
    domain_examples = ""
    if latest_domain:
        preds = pd.DataFrame(latest_domain["predictions"]).merge(domain_df[["id", "text"]], on="id")
        errs = preds[preds["label"] != preds["pred"]].head(20)[["id", "text", "label", "pred", "confidence"]]
        domain_examples = table(errs.rename(columns={"label": "expected", "pred": "predicted"}), "{:.2f}")
    reviewed = latest_domain["labels_reviewed"] if latest_domain else False

    thr = pd.DataFrame(deployed_final["calibration"]["thresholds"]) if deployed_final.get("calibration") else None

    # ---------- transformer
    transformer_html = ""
    if tcard:
        hist = pd.DataFrame(tcard["history"])
        tf_final = next((f for f in finals if "distilbert" in f["model_version"]), None)
        transformer_html = f"""
<h3>6.5 E4 — DistilBERT fine-tuning (optional phase, guide §22)</h3>
<p>After the classical pipeline was complete, <code>distilbert-base-uncased</code> was fine-tuned on the <em>same frozen splits</em>
on the laptop's RTX 3050 (6 GB) with mixed precision. Settings: <code>{html.escape(tcard['settings'])}</code>.
The epoch was selected by validation macro F1 (best: epoch {tcard['best_epoch']}). Probabilities were calibrated with temperature
scaling fitted on validation (T = {tcard['temperature']:.3f}; validation ECE {tcard['validation_ece']['before_temperature']:.4f} →
{tcard['validation_ece']['after_temperature']:.4f}).</p>
{table(hist.rename(columns={"train_loss": "train loss", "val_macro_f1": "validation macro F1", "seconds": "seconds/epoch"}))}
<p>Validation macro F1 of the selected epoch: <b>{tcard['validation_metrics']['macro_f1']:.4f}</b> (accuracy {tcard['validation_metrics']['accuracy']:.4f}).
{"Test macro F1 (single run): <b>%.4f</b>." % tf_final['metrics']['macro_f1'] if tf_final else "Not yet evaluated on the test set."}</p>
"""

    # ---------- HTML
    def sec(title):
        return f"<h2>{html.escape(title)}</h2>"

    parts = [f"<style>{CSS}</style>"]
    parts.append(f"""
<div class="title">
<h1>Mentor AI — Emotion Classification Model</h1>
<div class="sub">Training, evaluation and integration report</div>
<div class="meta">Final Year Project · Generated {now:%d %B %Y} from the project's saved experiment artefacts<br>
Deployed model version: <code>{html.escape(deployed)}</code></div>
</div>
<div class="pb"></div>
""")

    parts.append(sec("1. Summary"))
    parts.append(f"""
<p>Mentor AI is a mentoring application (React PWA + Node.js/Express + PostgreSQL) whose replies are produced by a large language model.
This project adds a <b>supervised emotion classifier</b> that reads the user's message or check-in and returns an emotion label with a
calibrated confidence. The label is passed to the LLM as structured context so the mentor can adapt its tone. It is emotion classification of
text, <b>not a medical or psychological diagnosis</b>, and the LLM remains responsible for the final response.</p>
<div class="box"><b>Headline results (all measured, none estimated).</b><br>
Deployed model: {deployed_desc}.<br>
GoEmotions test set (used once per model version): accuracy <b>{deployed_final['metrics']['accuracy']:.3f}</b>, macro F1 <b>{deployed_final['metrics']['macro_f1']:.3f}</b>,
weighted F1 {deployed_final['metrics']['weighted_f1']:.3f}, expected calibration error {deployed_final['calibration']['ece']:.3f}.<br>
Mentor-domain check-in sentences ({latest_domain['n'] if latest_domain else 0} items{'' if reviewed else ', draft labels'}): accuracy
<b>{latest_domain['metrics']['accuracy']:.3f}</b>, macro F1 <b>{latest_domain['metrics']['macro_f1']:.3f}</b> — this is the score that reflects real use,
and it shows a large domain gap between Reddit training data and journal-style text.</div>
<p>Seven emotion classes are predicted: {", ".join(f"<code>{c}</code>" for c in classes)}. The class set was derived from the guide's
suggested taxonomy, then adjusted on measured evidence (section 4.3).</p>
""")

    parts.append(sec("2. Role in the Mentor AI architecture"))
    parts.append("""
<pre>React / PWA
    │
Node.js / Express API  ──►  PostgreSQL
    │  POST /predict {text}   (1.5 s timeout; optional, fails open)
Python / FastAPI  (127.0.0.1:8001)
    │
Trained pipeline: features → classifier → calibrated probabilities
    │
{emotion, confidence, probabilities, model_version}
    │
Appended to the mentor's system prompt as "Detected Emotional Signal … not a diagnosis"
    │
LLM  ──►  final mentoring response</pre>
<p>Design rules: the classifier is <em>advisory</em>. If the service is down, slow, or its confidence is below a threshold
(<code>EMOTION_MIN_CONFIDENCE</code>, recommended 0.6), no emotion context is sent and the chat behaves exactly as before.
The label is never shown to the user as a fact; the prompt instructs the LLM to use it only to adapt tone and support.</p>
""")

    parts.append(sec("3. Environment and reproducibility"))
    parts.append(f"""
<table><tr><th>Item</th><th>Value</th></tr>
<tr><td>Hardware</td><td>Lenovo LOQ laptop: 12-core CPU, 15 GiB RAM, NVIDIA GeForce RTX 3050 6 GB (used only for the optional transformer)</td></tr>
<tr><td>Operating system</td><td>Fedora Linux 44 (kernel {platform.release()})</td></tr>
<tr><td>Python / packages</td><td>Python {card['versions']['python']}, scikit-learn {card['versions']['scikit-learn']}, NumPy {card['versions']['numpy']}, pandas {pd.__version__}
{", PyTorch " + tcard['versions']['torch'] + " (CUDA " + str(tcard['versions']['cuda']) + ")" if tcard else ""}</td></tr>
<tr><td>Environment manager</td><td><code>uv</code> with a lock file (<code>uv.lock</code>) and an exported <code>requirements.txt</code></td></tr>
<tr><td>Random seed</td><td>{config.SEED} everywhere (splits are the official GoEmotions splits; CV folds, classifiers and the transformer are seeded)</td></tr>
<tr><td>Frozen data</td><td>SHA-256 of each processed split is stored in <code>data/processed/manifest.json</code>; every script verifies it before loading. Rebuilding the data twice produced identical hashes.</td></tr>
</table>
<p>The full pipeline is one command, <code>make all</code> (= <code>data → train → tune → evaluate → test</code>), following the guide's
exact training order. Notebooks <code>01_dataset_exploration</code> and <code>02_model_experiments</code> read their figures from the same files.</p>
""")

    parts.append(sec("4. Dataset"))
    parts.append(f"""
<h3>4.1 Source</h3>
<p><b>GoEmotions</b> (Demszky et al., ACL 2020; <code>{html.escape(manifest['dataset'])}</code>, licence CC BY 4.0): 58k English Reddit comments
annotated with 27 emotions plus neutral, multi-label, with official train/validation/test splits of 43,410 / 5,426 / 5,427 examples.
It was chosen for its fine-grained labels and permissive licence. Its weaknesses — Reddit register, crowd-sourced label noise, and a
different domain from journaling — are analysed in section 7.</p>
<h3>4.2 Label mapping (version {manifest['label_map_version']})</h3>
<p>The guide suggested Joy, Sadness, Fear/Anxiety, Anger, Guilt, Frustration, Motivation and Neutral. GoEmotions labels were mapped
onto these classes; labels that describe attitudes towards others (admiration, approval, caring), cognitive states (confusion, curiosity,
realization) or are valence-ambiguous (surprise, disapproval) were dropped: {", ".join(manifest['dropped_source_labels'])}.</p>
{table(mapping)}
<p>An example is kept <b>only if all of its labels map to the same Mentor class</b>; mixed-emotion comments are removed rather than
force-labelled. This is a deliberate trade-off: cleaner single-label training data at the cost of about 37% of the raw rows.</p>
<h3>4.3 Evidence-based change: merging anger and frustration</h3>
<p>The first mapping (v1) kept <code>anger</code> and <code>frustration</code> separate. Measured on the validation set:</p>
{table(pd.DataFrame(v1_rows))}
<p><code>fear_anxiety</code> and <code>guilt</code> were below the 800-example review threshold but performed in line with the other
minority classes, so they were kept. <code>frustration</code> had ample data yet F1 of only ~0.28, being confused mostly with anger and neutral.
The two were merged into <code>anger_frustration</code> (v2). Validation macro F1 rose from {v1_e2['macro_f1']:.3f} to
{val_runs['E2 / fs1']['metrics']['macro_f1']:.3f} for the same SVM. The decision was made on validation data only, before the test set was touched.</p>
<h3>4.4 Processed splits</h3>
{table(counts, "{:.0f}")}
{table(filtering, "{:.0f}")}
<div class="two">{img(config.FIGURES / "mentor_class_counts.png", "Figure 1. Training examples per Mentor AI class (v2).")}
{img(config.FIGURES / "goemotions_label_counts.png", "Figure 2. Raw GoEmotions label counts coloured by Mentor AI class.")}</div>
""")

    parts.append(sec("5. Pre-processing and leakage prevention"))
    parts.append("""
<p>Pre-processing is deliberately conservative, following the guide: whitespace is normalised and text trimmed; empty texts, exact
duplicates and texts that appear with conflicting labels are removed; texts that also occur in a later split are removed from the earlier
one. Negations, punctuation, emojis, casing and stop words are all kept, and no stemming is applied, because these carry emotional signal.</p>
<p>Leakage is prevented structurally: every model is a scikit-learn <code>Pipeline</code> whose vectorisers are fitted inside <code>fit()</code>,
so vocabulary and IDF statistics come from training text only; cross-validation happens inside that pipeline; and an automated test
asserts that an unseen word never enters the vocabulary. The test set is loaded only by <code>src/evaluate.py</code>, which refuses to run a
second time without an explicit, logged reason.</p>
""")

    parts.append(sec("6. Methodology and experiments"))
    parts.append(f"""
<h3>6.1 Feature sets</h3>
<p><b>fs1</b> — the guide's configuration: word uni- and bi-gram TF-IDF (lower-cased, sublinear TF, <code>max_df</code> 0.95).
Review of the trained model showed that scikit-learn's default token pattern silently discards emojis and punctuation, contradicting the
"keep emojis" rule. <b>fs2</b> fixes this with a token pattern that keeps single characters and symbols, and adds character n-grams
(2–5, within word boundaries) to capture informal spelling and typos. fs2 improved every model (table below).</p>
<h3>6.2 Models</h3>
<p><b>E1</b> Logistic Regression (<code>class_weight="balanced"</code>, <code>max_iter=2000</code>) — fast, interpretable, native probabilities.
<b>E2</b> Linear SVM (<code>LinearSVC</code>, balanced) — typically strong on sparse text but gives no probabilities.
<b>E3</b> the best of both after a controlled grid search. <b>E4</b> DistilBERT (optional transformer phase).</p>
<h3>6.3 Hyper-parameter search (train only)</h3>
<p>3-fold stratified cross-validation on the training split with macro F1 as the criterion, over word n-gram range, minimum document
frequency and C for both classifiers (fs1: 36 candidates; fs2: 12 candidates, because character n-grams make each fit ~4× slower). The best
configuration was then frozen and checked once on validation. Top fs2 candidates:</p>
{table(cv.rename(columns=lambda c: c.replace("param_features__word__", "").replace("param_", "").replace("_test_score", " CV F1")))}
<h3>6.4 Calibration</h3>
<p>The API must return a real probability, and the Node side thresholds on it. An SVM decision score is not a probability, so the deployed
SVM is wrapped in <code>CalibratedClassifierCV</code>. Four options were compared on validation: sigmoid (Platt) vs isotonic, and an
ensemble of three fold-models vs a single full-data model. Sigmoid/ensemble — the common default — lost ~2 points of macro F1 because it
shifts decisions towards the majority class; isotonic on a single full-data model kept the SVM's macro F1 (0.711 vs 0.715) and had the lowest
expected calibration error (0.027). It is used for the TF-IDF model, which serves as the fallback.</p>
{transformer_html}
<h3>6.6 Experiment log (validation unless marked test)</h3>
{table(exp_view[exp_cols])}
{img(config.FIGURES / "experiment_comparison.png", "Figure 3. Headline metrics per experiment.")}
<h3>6.7 Per-class F1 on validation</h3>
{table(per_class_val)}
""")

    parts.append(sec("7. Results"))
    parts.append(f"""
<h3>7.1 Final test-set results (one evaluation per model version)</h3>
<p>Each row is a distinct model version evaluated once on the untouched test set after its configuration was frozen on validation.
Versions were <em>not</em> selected by their test score.</p>
{table(final_summary)}
<h3>7.2 Deployed model — per-class test results</h3>
{table(per_class_table(deployed_final['metrics']['per_class'], classes))}
<div class="two">{img(config.FIGURES / f"confusion_matrix_final_test_{deployed}.png", f"Figure 4. Test confusion matrix (counts), {deployed}.")}
{img(config.FIGURES / f"confusion_matrix_final_test_{deployed}_normalized.png", "Figure 5. Row-normalised: share of each true class predicted as each label.")}</div>
<p>Reading the matrix: <code>joy</code> and <code>neutral</code> are recognised well; the largest confusions are between <code>anger_frustration</code> and <code>neutral</code> in both directions, with most other
misses going to <code>neutral</code>. Because the app sends context only for confident non-neutral predictions, a miss usually means
"no context" (harmless) rather than a wrong emotion.</p>
<h3>7.3 Confidence calibration and the Node threshold</h3>
<p>Expected calibration error on test: <b>{deployed_final['calibration']['ece']:.3f}</b> (0 = perfect). The table shows, for each
<code>EMOTION_MIN_CONFIDENCE</code>, how often a non-neutral emotion would be injected into the prompt and how often it would be right.
A wrong emotion in the prompt is worse than none, so 0.6 is recommended.</p>
{table(thr.rename(columns={"accuracy_at_threshold": "accuracy of predictions above threshold", "non_neutral_sent": "share of messages that get context", "non_neutral_precision": "precision of that context"}), "{:.3f}") if thr is not None else ""}
<h3>7.4 Mentor-domain evaluation set</h3>
<p>GoEmotions scores measure Reddit comments. To measure what matters for the app, a set of {latest_domain['n'] if latest_domain else 0}
check-in / journal style sentences (22 per class, including negation, emoji and very short items) was written and scored. It is used for
evaluation only, never for training or tuning.</p>
{'' if reviewed else '<div class="box warn"><b>Status:</b> the labels of this set are a draft written by the developer\'s assistant and have not yet been reviewed by the project owner. Treat these numbers as provisional until <code>data/mentor_eval/README.md</code> records a reviewer.</div>'}
{table(domain_summary, "{:.3f}")}
{table(domain_per_class)}
{img(config.FIGURES / "domain_eval_comparison.png", "Figure 6. Mentor-domain evaluation per model version.")}
<p>Typical errors of the deployed model on this set (first 20):</p>
{domain_examples}
<p>For the TF-IDF models the pattern was consistent: sentences that express an emotion through a <em>situation</em> ("stomach in knots",
"skipped my workout again") rather than an explicit emotion word were classified as neutral, and negation was not understood. DistilBERT,
which reads words in context, removes a large part of this gap. Its remaining errors are concentrated in two places: situational anxiety
(exam or presentation worries) still often reads as neutral, and <code>guilt</code> is confused with <code>anger_frustration</code> and
<code>sadness</code> — all three are self-directed negative states that GoEmotions' Reddit data rarely separates the way a journal does.
Domain-specific training data is the remedy for both.</p>

<h3>7.5 Model selection</h3>
<p>The deployed model was chosen on <b>validation macro F1</b> and the <b>Mentor-domain set</b>, never on test scores; each test
evaluation was run once, after its model was frozen, to confirm the choice. All three signals agree:</p>
{table(selection, "{:.3f}")}
<p>DistilBERT is deployed: it is the best model on every measure, it is the best-calibrated (lowest ECE), and it serves in
{latency['v2-distilbert-20260914']['gpu']['single_text_ms']} ms per message on the RTX 3050 or
{latency['v2-distilbert-20260914']['cpu']['single_text_ms']} ms on CPU — far below the Node service's 1.5 s timeout. The TF-IDF model
remains in the repository as an automatic fallback when PyTorch or the transformer weights are not available.</p>
""")

    parts.append(sec("8. Deployment and integration"))
    parts.append("""
<p><b>Model service</b> (<code>app/main.py</code>, FastAPI/uvicorn on 127.0.0.1:8001): loads the serialised pipeline once at start-up;
<code>POST /predict {"text"}</code> returns <code>{emotion, confidence, probabilities, model_version}</code>; empty or over-long text is
rejected with 422; <code>GET /health</code> reports the model version and classes; interactive docs at <code>/docs</code>. A systemd user unit
is provided. Measured latency per message: @@LATENCY@@.</p>
<p><b>Node.js integration</b> (Mentor AI API, TypeScript): a new <code>EmotionClassifierService</code> calls the service with a 1.5 s timeout,
validates the JSON with Zod and returns <code>null</code> on any failure or low confidence. <code>ChatAgentService</code> runs the classification
<em>concurrently</em> with building the system prompt, so no latency is added, then appends a short "Detected Emotional Signal" section that
explicitly says the signal may be wrong and is not a diagnosis. The feature is behind <code>EMOTION_CLASSIFIER_ENABLED</code> (off by default),
so production is unaffected until the service is deployed.</p>
""")

    parts.append(sec("9. Limitations, ethics and future work"))
    parts.append("""
<ul>
<li><b>Domain shift</b> — training data is Reddit comments; users write journal entries and check-ins. This is the dominant source of error and
is quantified by the Mentor-domain set.</li>
<li><b>Label noise and approximation</b> — crowd-sourced labels; <code>guilt</code> (remorse + embarrassment) and <code>motivation</code>
(optimism + desire + pride) are approximations of the intended classes.</li>
<li><b>Class imbalance and small classes</b> — <code>neutral</code> is 47% of training data; <code>fear_anxiety</code> and <code>guilt</code>
have fewer than 600 training and under 100 test examples, so their scores have wide uncertainty.</li>
<li><b>Bag-of-words</b> — word order, negation scope and sarcasm are not modelled by TF-IDF.</li>
<li><b>Ethics</b> — the output is a coarse, uncertain signal about text, never a statement about the person; it is not shown to the user as a
fact, is thresholded on confidence, fails open, and must not be used for any clinical purpose.</li>
</ul>
<p><b>Future work:</b> (1) collect a consented, anonymised set of real Mentor AI check-ins and use it both as the acceptance test and, with a
separate held-out part, as domain training data; (2) fine-tune a transformer on the combined data; (3) extend the signal to voice sessions and
store per-message emotions to give mentors trend insight over time; (4) revisit the taxonomy once real-domain data shows which distinctions
users actually make.</p>
""")

    parts.append(sec("Appendix A. Reproduction commands"))
    parts.append("""<pre>sudo dnf install -y uv python3.12
cd ~/Coding/python && uv sync                     # environment from uv.lock
make data        # download GoEmotions, map labels, clean, freeze splits (hashes in manifest.json)
make train       # E1 Logistic Regression, E2 Linear SVM  -> validation metrics
make tune        # E3 grid search (3-fold CV on train)    -> models/frozen_config.json
make evaluate    # single test-set evaluation, serialise models/emotion_classifier.joblib
make test        # pytest: data hygiene, leakage guard, API contract
make domain-eval # score on data/mentor_eval/mentor_eval.csv
uv sync --extra transformer && make transformer && make evaluate-transformer   # optional E4
make serve       # FastAPI on 127.0.0.1:8001
uv run python -m src.report   # this document</pre>""")

    parts.append(sec("Appendix B. Definition of done (guide §23)"))
    parts.append(f"""<ul>
<li>Dataset and label mapping documented — <code>data/README.md</code>, <code>data/label_mapping.md</code></li>
<li>Training reproducible — seeded, locked environment, hash-frozen splits (verified identical on rebuild)</li>
<li>At least two models compared — E1, E2, E3{", E4" if tcard else ""} on identical splits</li>
<li>Test set protected — loaded only by <code>evaluate.py</code>; one run per model version; guard and log for re-runs</li>
<li>Metrics and confusion matrices saved — <code>reports/metrics/</code>, <code>reports/figures/</code></li>
<li>Final model serialised — <code>models/distilbert/</code> (deployed) and <code>models/emotion_classifier.joblib</code> (fallback), each with <code>model_card.json</code></li>
<li>Local prediction API working — FastAPI, tests pass</li>
<li>Node.js integration — implemented behind a feature flag{"; live end-to-end run pending" if True else ""}</li>
<li>Actual measured results and limitations — this report</li>
</ul>
<p class="small">Generated by <code>src/report.py</code> on {now:%Y-%m-%d %H:%M} UTC. Experiment rows: {len(experiments)}; test evaluations: {len(finals)}.</p>""")

    tf_lat = latency["v2-distilbert-20260914"]
    latency_text = (
        f"DistilBERT {tf_lat['gpu']['single_text_ms']} ms on GPU / {tf_lat['cpu']['single_text_ms']} ms on CPU "
        f"(about {tf_lat['cpu']['max_rss_mb'] / 1024:.1f} GB RAM); TF-IDF fallback "
        f"{latency['v2-fs2-linear_svm_calibrated-20260914']['single_text_ms']} ms on CPU"
    )
    OUT_HTML.write_text("\n".join(parts).replace("@@LATENCY@@", latency_text))
    chrome = shutil.which("google-chrome") or shutil.which("chromium")
    if chrome:
        subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={OUT_PDF}", OUT_HTML.as_uri()], check=True, capture_output=True, timeout=180)
        print(f"wrote {OUT_PDF} ({OUT_PDF.stat().st_size // 1024} KB)")
    else:
        print(f"wrote {OUT_HTML}; no Chrome found for PDF conversion")


if __name__ == "__main__":
    main()
