# Mentor-domain evaluation set

`mentor_eval.csv` holds 154 short check-in / journal style sentences (22 per Mentor AI class), written to look like what
Mentor AI users actually type. GoEmotions is Reddit comments, so the GoEmotions test score does not tell us how a model
behaves on this kind of text. **This file is the acceptance test that matters for the app.**

Columns: `id`, `text`, `label` (draft, see below), `kind` (`plain`, `negation`, `emoji`, `short`, for slicing results),
`note` (ambiguity remarks).

## Status: DRAFT LABELS — needs review by the project owner
The sentences and labels were written by the assistant on 2026-09-15 and have **not** been checked by a human yet.
Before the numbers are quoted anywhere:
1. Read every row. Fix any label you disagree with (the `note` column flags the ambiguous ones).
2. Delete rows that feel unrealistic, and add real (anonymized, consented) check-ins if you have any.
3. Set `REVIEWED_BY` below.

`REVIEWED_BY: (nobody yet)`

## Usage
```bash
make domain-eval                                          # scores the default model (DistilBERT if trained)
make domain-eval MODEL=models/emotion_classifier.joblib   # scores the TF-IDF fallback
```
Results go to `reports/metrics/domain_eval_<model_version>.json`.

## Rules
- This set is **evaluation only**. It must not be used for training or tuning, otherwise the score is meaningless.
- It is small (22 per class), so treat per-class numbers as indicative, not precise.
