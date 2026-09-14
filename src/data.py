"""Loading of the frozen processed splits, with hash verification."""
import hashlib
import json
from pathlib import Path

import pandas as pd

from src import config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest() -> dict:
    if not config.MANIFEST_PATH.exists():
        raise FileNotFoundError(f"{config.MANIFEST_PATH} not found - run `make data` first")
    return json.loads(config.MANIFEST_PATH.read_text())


def load_split(split: str, verify: bool = True) -> pd.DataFrame:
    entry = load_manifest()["splits"][split]
    path = config.DATA_PROCESSED / entry["file"]
    if verify and sha256_file(path) != entry["sha256"]:
        raise RuntimeError(
            f"{path} does not match the hash in manifest.json. The splits are frozen; "
            "re-run `make data` deliberately if the data really must change."
        )
    return pd.read_parquet(path)


def load_xy(split: str) -> tuple[pd.Series, pd.Series]:
    df = load_split(split)
    return df["text"], df["label"]


def load_classes() -> list[str]:
    return load_manifest()["classes"]
