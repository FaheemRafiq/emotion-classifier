# Data

## Source
- **Dataset:** GoEmotions (`google-research-datasets/go_emotions`, `simplified` config), loaded with Hugging Face `datasets`.
- **Paper:** Demszky et al., *GoEmotions: A Dataset of Fine-Grained Emotions*, ACL 2020.
- **License:** CC BY 4.0. The dataset consists of English Reddit comments; names are already masked as `[NAME]`.
- **Official splits:** train 43,410, validation 5,426, test 5,427 examples, with 27 emotion labels plus `neutral`. Each example can carry several labels.

## Layout
| Path | Content | In git? |
|---|---|---|
| `raw/` | Hugging Face download cache | no (re-created) |
| `processed/{train,validation,test}.parquet` | Mapped and cleaned splits (`id`, `text`, `label`) | no (re-created) |
| `processed/manifest.json` | Class counts, filtering stats, SHA-256 of every split, label map version | **yes** |
| `label_mapping.md` | Mentor AI taxonomy and every mapping/merge/drop decision | **yes** |

Rebuild with `make data`. Running it twice produces identical SHA-256 hashes (verified for label maps v1 and v2). Training and
evaluation scripts refuse to load a split whose hash no longer matches `manifest.json`, which is what freezes the splits.

## Processing steps (`src/prepare_data.py`)
1. Map each GoEmotions label to a Mentor AI class (see `label_mapping.md`).
2. Keep an example **only** if all its labels map to the same Mentor class. Examples are removed when they have only unmapped labels, some unmapped labels, or labels that map to different Mentor classes.
3. Conservative cleaning: collapse whitespace and trim. Negations, punctuation, emojis, casing (lowercasing happens later inside TF-IDF), and stop words are all kept, and no stemming is done.
4. Drop empty text, exact duplicates, and texts that appear with different labels.
5. Remove texts that also appear in a later split (train vs. validation/test, validation vs. test), so there is no leakage.

## Measured filtering results (label map v2)
| Step | train | validation | test |
|---|---:|---:|---:|
| Raw rows | 43,410 | 5,426 | 5,427 |
| Removed: only unmapped labels | 11,182 | 1,426 | 1,431 |
| Removed: partially unmapped labels | 3,539 | 437 | 409 |
| Removed: conflicting Mentor classes | 1,555 | 195 | 182 |
| Removed: empty text | 0 | 0 | 0 |
| Removed: exact duplicates | 108 | 2 | 5 |
| Removed: same text, different labels | 38 | 0 | 0 |
| Removed: overlap with a later split | 41 | 7 | 0 |
| **Final rows** | **26,947** | **3,359** | **3,400** |

## Known limitations
- **Domain shift:** the data is Reddit comments, while Mentor AI receives journal entries and check-ins.
- **Label noise:** the annotators were crowd workers, and many comments are short and ambiguous.
- **Strict filtering:** single-class filtering removes about 37% of the raw training rows, so mixed-emotion text is under-represented.
- **Imbalance:** `neutral` is about 47% of the training data, and `fear_anxiety` and `guilt` each have fewer than 600 training examples.
