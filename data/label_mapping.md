# Label mapping: GoEmotions → Mentor AI taxonomy

Current version: **v2** (`LABEL_MAP_VERSION` in `src/config.py`). Every change bumps the version and is logged below.

## Current taxonomy (v2, 7 classes)
| Mentor class | GoEmotions source labels | train | validation | test |
|---|---|---:|---:|---:|
| joy | joy, amusement, excitement, gratitude, relief, love | 6,743 | 861 | 814 |
| sadness | sadness, grief | 868 | 91 | 107 |
| fear_anxiety | fear, nervousness | 536 | 72 | 80 |
| anger_frustration | anger, disgust, annoyance, disappointment | 4,119 | 497 | 551 |
| guilt | remorse, embarrassment | 559 | 61 | 67 |
| motivation | optimism, desire, pride | 1,341 | 186 | 176 |
| neutral | neutral | 12,781 | 1,591 | 1,605 |
| **total** | | **26,947** | **3,359** | **3,400** |

The counts are measured by `src/prepare_data.py` and stored in `processed/manifest.json`.

### Dropped GoEmotions labels
admiration, approval, caring, confusion, curiosity, disapproval, realization, surprise.
- `admiration`, `approval`, and `caring` describe an attitude toward *someone else*, not the writer's own emotional state.
- `confusion`, `curiosity`, and `realization` are cognitive states, not emotions Mentor AI needs to respond to.
- `surprise` can be positive or negative, and `disapproval` can mean anger or a neutral judgement, so neither maps cleanly.

### Filtering rule
An example is kept **only if all of its labels map to the same Mentor class**. The goal is clean single-label training data. Mixed-emotion
text is removed rather than force-labelled; the counts are in `README.md`.

### Known approximations
- **guilt ← remorse + embarrassment.** Embarrassment is a self-conscious emotion related to guilt, but it is not the same thing.
- **motivation ← optimism + desire + pride.** This is the weakest semantic fit: GoEmotions has no "motivation" label, and `desire` can mean craving rather than drive.
- **Small support.** `fear_anxiety` and `guilt` have fewer than 600 training examples and only 67–80 test examples, so their per-class test scores have wide uncertainty.

## Decision log

### v1 → v2 (2026-09-15): merge `anger` + `frustration` into `anger_frustration`
v1 had 8 classes, as suggested in the guide: `anger` ← anger, disgust and `frustration` ← annoyance, disappointment.

Measured on v1 validation (`reports/metrics/label_map_v1/`):

| Class | train support | E1 LogReg F1 | E2 LinearSVM F1 |
|---|---:|---:|---:|
| fear_anxiety | 536 | 0.6087 | 0.6429 |
| guilt | 559 | 0.6308 | 0.6783 |
| anger | 1,551 | 0.4869 | 0.4709 |
| frustration | 2,232 | **0.2934** | **0.2650** |
| *macro F1 (8 classes)* | | 0.5951 | 0.6130 |

- `fear_anxiety` and `guilt` are below the review threshold of 800 training examples. Their F1 is in line with the other minority classes, so they were **kept**.
- `frustration` has plenty of data but very poor F1, because it is mostly confused with `anger` and `neutral`.

A scratchpad comparison on validation relabelled the v1 split, so it is approximate. Macro F1 for each option:

| Option | LogReg | LinearSVM | Note |
|---|---:|---:|---|
| keep frustration (8 classes) | 0.5951 | 0.6130 | frustration predictions unreliable |
| merge frustration → anger (7) | 0.6457 | 0.6678 | |
| drop frustration (7) | 0.6693 | 0.7075 | inflated: the hardest validation examples are removed; frustrated users would be mislabelled in production |

**Decision (approved by the project owner):** merge into `anger_frustration`. This keeps a label that is honest for frustrated users while
removing a boundary the data cannot support. The decision used validation data only; the test set had not been used at that point.
