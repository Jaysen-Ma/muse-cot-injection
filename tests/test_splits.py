import json
from collections import Counter

from cot_injection.paths import SPLIT_FILE
from cot_injection.splits import assign_splits, choose_holdout, sample_indices

HOLDOUT = {"Disinformation and deception", "Hate, harassment and discrimination"}


def load():
    return json.loads(SPLIT_FILE.read_text())


def test_holdout_is_disinformation_and_hate():
    meta = load()
    assert choose_holdout(meta["category_counts"]) == HOLDOUT
    assert set(meta["holdout_categories"]) == HOLDOUT


def test_split_sizes_141_train_59_holdout():
    meta = load()
    items = [{"id": i["id"], "category": i["category"]} for i in meta["items"]]
    split = assign_splits(items, choose_holdout(meta["category_counts"]))
    c = Counter(x["split"] for x in split)
    assert c == {"train": 141, "holdout": 59}
    assert meta["holdout_n"] == 59 and meta["n"] == 200


def test_public_split_file_matches_split_logic():
    meta = load()
    items = [{"id": i["id"], "category": i["category"]} for i in meta["items"]]
    assert assign_splits(items, choose_holdout(meta["category_counts"])) == meta["items"]
    assert all(set(i) == {"id", "category", "split"} for i in meta["items"])


def test_category_counts_consistent():
    meta = load()
    assert dict(Counter(i["category"] for i in meta["items"])) == meta["category_counts"]


def test_sampling_is_deterministic():
    a, b = sample_indices(), sample_indices()
    assert a == b and len(a) == 200 and len(set(a)) == 200
    assert a == sorted(a) and 0 <= a[0] and a[-1] < 313
