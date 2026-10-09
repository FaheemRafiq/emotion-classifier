"""E1: build label-blind batches for the LLM-only emotion baseline.

Sample (frozen, seed 42): GoEmotions test up to 100 per class (guilt has only 67), ISEAR test 70 per class (350), all 108
mixed-emotion test texts and all 154 Mentor-style texts. Texts from all sources are shuffled together and given
neutral ids, so the labeller cannot tell which dataset a text came from. Gold labels go to key.jsonl only.

Usage: uv run python experiments/e1_emotion/make_batches.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402
from src.data import load_split  # noqa: E402

HERE = Path(__file__).parent
SEED = 42
BATCH = 100


def main() -> None:
    test = load_split("test")
    goemo = pd.concat([g.sample(min(100, len(g)), random_state=SEED) for _, g in test.groupby("label")])
    isear = pd.read_parquet(config.DATA_PROCESSED_V3 / "isear_test.parquet")
    isear = pd.concat([g.sample(min(70, len(g)), random_state=SEED) for _, g in isear.groupby("label")])
    mixed = pd.read_parquet(config.DATA_PROCESSED_V3 / "mixed_test.parquet")
    domain = pd.read_csv(config.DOMAIN_EVAL_CSV)

    rows = ([{"source": "goemotions_test", "text": t, "gold": g} for t, g in zip(goemo["text"], goemo["label"])]
            + [{"source": "isear_test", "text": t, "gold": g} for t, g in zip(isear["text"], isear["label"])]
            + [{"source": "mixed_test", "text": t, "gold": g} for t, g in zip(mixed["text"], mixed["gold"])]
            + [{"source": "mentor_domain", "text": t, "gold": g} for t, g in zip(domain["text"], domain["label"])])
    df = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    df["id"] = [f"t{i + 1:04d}" for i in range(len(df))]

    with open(HERE / "key.jsonl", "w") as f:
        for r in df.itertuples():
            f.write(json.dumps({"id": r.id, "source": r.source, "gold": r.gold, "text": r.text}) + "\n")
    for b, start in enumerate(range(0, len(df), BATCH), 1):
        with open(HERE / "batches" / f"batch_{b:02d}.jsonl", "w") as f:
            for r in df.iloc[start:start + BATCH].itertuples():
                f.write(json.dumps({"id": r.id, "text": r.text}) + "\n")
    print(df["source"].value_counts().to_dict(), "->", (len(df) + BATCH - 1) // BATCH, "batches")


if __name__ == "__main__":
    main()
