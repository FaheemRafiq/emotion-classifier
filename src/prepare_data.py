"""Download GoEmotions, map it to the Mentor AI taxonomy, clean it, and freeze the splits.

Usage: uv run python -m src.prepare_data
"""
import re
from collections import Counter

import pandas as pd
from datasets import load_dataset

from src import config
from src.data import sha256_file
from src.metrics import save_json

WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str | None) -> str:
    """Conservative cleaning: only whitespace normalization (keeps negations, punctuation, emojis)."""
    if text is None:
        return ""
    return WHITESPACE_RE.sub(" ", text).strip()


def build_source_to_mentor(label_names: list[str]) -> dict[str, str]:
    source_to_mentor: dict[str, str] = {}
    for mentor_class, sources in config.MENTOR_LABEL_MAP.items():
        for source in sources:
            if source not in label_names:
                raise ValueError(f"Unknown GoEmotions label in mapping: {source!r}")
            if source in source_to_mentor:
                raise ValueError(f"GoEmotions label mapped twice: {source!r}")
            source_to_mentor[source] = mentor_class
    return source_to_mentor


def map_labels(label_ids, label_names: list[str], source_to_mentor: dict[str, str]) -> tuple[str | None, str]:
    """Return (mentor_class, reason). Keep only examples whose labels all map to one Mentor class."""
    names = [label_names[i] for i in label_ids]
    if not names:
        return None, "removed_no_labels"
    mapped = {source_to_mentor.get(name) for name in names}
    if mapped == {None}:
        return None, "removed_only_unmapped_labels"
    if None in mapped:
        return None, "removed_partially_unmapped_labels"
    if len(mapped) > 1:
        return None, "removed_conflicting_mentor_classes"
    return mapped.pop(), "kept_after_mapping"


def process_split(df: pd.DataFrame, label_names: list[str], source_to_mentor: dict[str, str]):
    stats: Counter = Counter(raw_rows=len(df))

    results = [map_labels(ids, label_names, source_to_mentor) for ids in df["labels"]]
    stats.update(reason for _, reason in results)
    df = df.assign(label=[label for label, _ in results])
    df = df[df["label"].notna()].copy()

    df["text"] = df["text"].map(clean_text)
    empty = df["text"].eq("")
    stats["removed_empty_text"] = int(empty.sum())
    df = df[~empty]

    before = len(df)
    df = df.drop_duplicates(subset=["text", "label"])
    stats["removed_exact_duplicates"] = before - len(df)

    conflicting = df["text"].duplicated(keep=False)
    stats["removed_same_text_different_labels"] = int(conflicting.sum())
    df = df[~conflicting]

    return df[["id", "text", "label"]].reset_index(drop=True), stats


def remove_cross_split_overlap(splits: dict[str, pd.DataFrame], stats: dict[str, Counter]) -> None:
    """Remove texts that also occur in a later split (train vs val/test, val vs test)."""
    test_texts = set(splits["test"]["text"])
    val_mask = splits["validation"]["text"].isin(test_texts)
    stats["validation"]["removed_overlap_with_test"] = int(val_mask.sum())
    splits["validation"] = splits["validation"][~val_mask].reset_index(drop=True)

    later_texts = test_texts | set(splits["validation"]["text"])
    train_mask = splits["train"]["text"].isin(later_texts)
    stats["train"]["removed_overlap_with_validation_or_test"] = int(train_mask.sum())
    splits["train"] = splits["train"][~train_mask].reset_index(drop=True)


def main() -> None:
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG, cache_dir=str(config.DATA_RAW))
    label_names: list[str] = dataset["train"].features["labels"].feature.names
    source_to_mentor = build_source_to_mentor(label_names)

    splits: dict[str, pd.DataFrame] = {}
    stats: dict[str, Counter] = {}
    for split in config.SPLITS:
        splits[split], stats[split] = process_split(dataset[split].to_pandas(), label_names, source_to_mentor)
    remove_cross_split_overlap(splits, stats)

    manifest = {
        "dataset": f"{config.DATASET_NAME} ({config.DATASET_CONFIG})",
        "label_map_version": config.LABEL_MAP_VERSION,
        "label_map": config.MENTOR_LABEL_MAP,
        "dropped_source_labels": sorted(set(label_names) - set(source_to_mentor)),
        "classes": config.MENTOR_CLASSES,
        "splits": {},
    }
    for split, df in splits.items():
        path = config.DATA_PROCESSED / f"{split}.parquet"
        df.to_parquet(path, index=False)
        counts = df["label"].value_counts()
        manifest["splits"][split] = {
            "file": path.name,
            "sha256": sha256_file(path),
            "rows": len(df),
            "class_counts": {c: int(counts.get(c, 0)) for c in config.MENTOR_CLASSES},
            "filtering": dict(stats[split]),
        }
    save_json(manifest, config.MANIFEST_PATH)

    count_table = pd.DataFrame({s: manifest["splits"][s]["class_counts"] for s in config.SPLITS})
    count_table.loc["TOTAL"] = count_table.sum()
    print("\nClass counts per split:\n", count_table.to_string())
    print("\nFiltering stats:\n", pd.DataFrame({s: stats[s] for s in config.SPLITS}).fillna(0).astype(int).to_string())

    weak = [c for c, n in manifest["splits"]["train"]["class_counts"].items() if n < config.MIN_TRAIN_SUPPORT]
    if weak:
        print(f"\nWARNING: classes below {config.MIN_TRAIN_SUPPORT} training examples: {weak}")
        print("Review merge/drop in src/config.py and document it in data/label_mapping.md.")
    print(f"\nWrote {config.MANIFEST_PATH}")


if __name__ == "__main__":
    main()
