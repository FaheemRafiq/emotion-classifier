"""Build the v3 training data: more (and harder) examples, same frozen validation/test sets.

What is added on top of the frozen v2 training split (data/processed/train.parquet):
  1. GoEmotions examples that v2 dropped because they carry a second, unmapped emotion
     (e.g. joy + admiration). The unmapped label is ignored; the mapped class is kept.
  2. GoEmotions mixed-emotion examples (labels from two or more Mentor classes, e.g. joy + nervousness).
     They are kept with a *soft* target that splits the probability between the classes.
  3. ISEAR (Scherer & Wallbott, 1994): first-person descriptions of real emotional situations, which
     often describe what happened without naming the emotion. 70 % is added to training,
     15 % is used for model selection and 15 % is a held-out "implicit emotion" test set.

The v2 validation and test splits are not modified, so v3 models are scored on exactly the same
examples as the v2 models. Every added text is removed if it also occurs in validation, test,
the ISEAR held-out splits or the Mentor-domain evaluation set.

Also written: a "mixed emotion" test set built from the GoEmotions *test* split (texts whose raters chose
emotions from two or more Mentor classes), which v2 had to discard.

Usage: uv run python -m src.prepare_extra
"""
import re
from collections import Counter

import numpy as np
import pandas as pd
from datasets import load_dataset

from src import config
from src.data import load_split, sha256_file
from src.metrics import save_json
from src.prepare_data import build_source_to_mentor, clean_text

ISEAR_NAME = "gsri-18/ISEAR-dataset-complete"
ISEAR_MAP = {"joy": "joy", "sadness": "sadness", "fear": "fear_anxiety", "anger": "anger_frustration",
             "disgust": "anger_frustration", "shame": "guilt", "guilt": "guilt"}
NON_RESPONSE = re.compile(r"no response|not applicable|does ?n[o']t apply|do ?n[o']t (?:know|remember)|can ?n[o']?t remember"
                          r"|^\[.*\]$|^(none|nothing|blank|n/?a)\.?$", re.I)
CLASSES = config.MENTOR_CLASSES


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def soft_target(classes: set[str]) -> list[float]:
    return [1.0 / len(classes) if c in classes else 0.0 for c in CLASSES]


def goemotions_extras(split: str, label_names, source_to_mentor) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows v2 discarded: returns (single-class after ignoring unmapped labels, multi-class)."""
    ds = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG, cache_dir=str(config.DATA_RAW))[split].to_pandas()
    singles, mixed = [], []
    for _, row in ds.iterrows():
        names = [label_names[i] for i in row["labels"]]
        mapped = {source_to_mentor.get(n) for n in names}
        text = clean_text(row["text"])
        if not text:
            continue
        real = mapped - {None}
        if None not in mapped and len(real) == 1:
            continue                                   # already in v2
        if not real:
            continue                                   # only unmapped emotions (admiration, curiosity, ...)
        if len(real) == 1:
            singles.append({"id": row["id"], "text": text, "label": next(iter(real)), "source": "goemotions_partial"})
        elif "neutral" not in real:                   # "neutral + emotion" is annotator noise, not a mixed feeling
            mixed.append({"id": row["id"], "text": text, "classes": sorted(real), "source": "goemotions_mixed"})
    return pd.DataFrame(singles), pd.DataFrame(mixed)


def load_isear() -> pd.DataFrame:
    df = load_dataset(ISEAR_NAME, cache_dir=str(config.DATA_RAW))["train"].to_pandas()
    df = df.assign(text=df["content"].fillna("").map(clean_text), label=df["emotion"].map(ISEAR_MAP))
    stats = Counter(raw_rows=len(df))
    bad = df["text"].str.split().str.len().lt(3) | df["text"].str.contains(NON_RESPONSE)
    stats["removed_non_response_or_too_short"] = int(bad.sum())
    df = df[~bad & df["label"].notna()]
    key = df["text"].map(norm)
    conflicting = df.assign(k=key).groupby("k")["label"].transform("nunique").gt(1)
    stats["removed_same_text_different_labels"] = int(conflicting.sum())
    df = df[~conflicting.to_numpy()]
    before = len(df)
    df = df.assign(k=df["text"].map(norm)).drop_duplicates("k").drop(columns="k")
    stats["removed_exact_duplicates"] = before - len(df)
    df = df.reset_index(drop=True)
    df["id"] = [f"isear-{i:05d}" for i in range(len(df))]
    df["source_label"] = df["emotion"]
    return df[["id", "text", "label", "source_label"]], stats


def stratified_split(df: pd.DataFrame, fractions=(0.70, 0.15, 0.15)) -> list[pd.DataFrame]:
    rng = np.random.default_rng(config.SEED)
    parts = [[], [], []]
    for _, group in df.groupby("source_label"):
        idx = rng.permutation(group.index.to_numpy())
        a = int(round(fractions[0] * len(idx)))
        b = a + int(round(fractions[1] * len(idx)))
        for part, sl in zip(parts, (idx[:a], idx[a:b], idx[b:])):
            part.extend(sl)
    return [df.loc[sorted(p)].reset_index(drop=True) for p in parts]


def main() -> None:
    out_dir = config.DATA_PROCESSED_V3
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG, cache_dir=str(config.DATA_RAW))
    label_names = dataset["train"].features["labels"].feature.names
    source_to_mentor = build_source_to_mentor(label_names)

    base = load_split("train")                       # frozen v2 training data (hash-verified)
    val, test = load_split("validation"), load_split("test")
    domain = pd.read_csv(config.DOMAIN_EVAL_CSV)

    isear, isear_stats = load_isear()
    isear_train, isear_val, isear_test = stratified_split(isear)

    protected = set(map(norm, pd.concat([val["text"], test["text"], domain["text"],
                                         isear_val["text"], isear_test["text"]])))
    stats: Counter = Counter()

    part_train, mixed_train = goemotions_extras("train", label_names, source_to_mentor)
    rows = [base.assign(source="goemotions", targets=[soft_target({l}) for l in base["label"]])]
    rows.append(part_train.assign(targets=[soft_target({l}) for l in part_train["label"]]))
    rows.append(mixed_train.assign(label=None, targets=[soft_target(set(c)) for c in mixed_train["classes"]]))
    rows.append(isear_train.assign(source="isear", targets=[soft_target({l}) for l in isear_train["label"]]))
    train = pd.concat(rows, ignore_index=True)[["id", "text", "label", "source", "targets"]]

    key = train["text"].map(norm)
    leak = key.isin(protected)
    stats["removed_overlap_with_eval_sets"] = int(leak.sum())
    train, key = train[~leak], key[~leak]
    dup = key.duplicated(keep="first")                # the frozen v2 rows come first and win
    stats["removed_duplicates_across_sources"] = int(dup.sum())
    train = train[~dup].reset_index(drop=True)

    # mixed-emotion benchmark from the GoEmotions test split (human labels, never used for training)
    _, mixed_test = goemotions_extras("test", label_names, source_to_mentor)
    mixed_test = mixed_test[~mixed_test["text"].map(norm).isin(set(train["text"].map(norm)))].reset_index(drop=True)
    mixed_test["gold"] = mixed_test["classes"].map(lambda c: "|".join(c))
    mixed_test = mixed_test[["id", "text", "gold"]]

    files = {"train": train, "isear_validation": isear_val, "isear_test": isear_test, "mixed_test": mixed_test}
    manifest = {"version": "v3", "label_map_version": config.LABEL_MAP_VERSION, "classes": CLASSES,
                "base_train_sha256": sha256_file(config.DATA_PROCESSED / "train.parquet"),
                "isear": {"dataset": ISEAR_NAME, "mapping": ISEAR_MAP, "filtering": dict(isear_stats),
                          "split": "stratified 70/15/15 by original ISEAR emotion, seed 42"},
                "filtering": dict(stats), "files": {}}
    for name, df in files.items():
        path = out_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        entry = {"file": path.name, "sha256": sha256_file(path), "rows": len(df)}
        if name == "train":
            entry["by_source"] = df["source"].value_counts().to_dict()
            entry["effective_class_mass"] = {c: round(float(np.sum([t[i] for t in df["targets"]])), 1)
                                             for i, c in enumerate(CLASSES)}
        elif "label" in df:
            entry["class_counts"] = df["label"].value_counts().to_dict()
        else:
            entry["gold_combinations"] = df["gold"].value_counts().head(15).to_dict()
        manifest["files"][name] = entry
    save_json(manifest, out_dir / "manifest.json")

    t = manifest["files"]["train"]
    print(f"v3 training rows: {t['rows']:,} (v2 had {len(base):,})")
    print("  by source:", t["by_source"])
    print("  class mass:", t["effective_class_mass"])
    print(f"ISEAR validation {len(isear_val)}, ISEAR test {len(isear_test)}, mixed-emotion test {len(mixed_test)}")
    print("filtering:", dict(stats), "| ISEAR:", dict(isear_stats))


if __name__ == "__main__":
    main()
