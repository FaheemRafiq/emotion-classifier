import pytest

from src import config
from src.data import load_classes, load_split

pytestmark = pytest.mark.skipif(not config.MANIFEST_PATH.exists(), reason="processed data missing; run `make data`")


@pytest.fixture(scope="module")
def splits():
    return {split: load_split(split, verify=True) for split in config.SPLITS}


def test_no_empty_text_and_only_known_labels(splits):
    classes = set(load_classes())
    for df in splits.values():
        assert not df["text"].str.strip().eq("").any()
        assert set(df["label"]) <= classes


def test_no_duplicate_text_within_split(splits):
    for df in splits.values():
        assert not df["text"].duplicated().any()


def test_no_text_overlap_between_splits(splits):
    train, val, test = (set(splits[s]["text"]) for s in config.SPLITS)
    assert not train & val
    assert not train & test
    assert not val & test
