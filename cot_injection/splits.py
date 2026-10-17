"""Sampling and category-holdout split used for the 200-prompt evaluation.

200 of the 313 StrongREJECT prompts are sampled with a fixed seed. Whole
categories are then held out: the holdout set is the combination of categories
whose total size is closest to ``target`` (ties broken by fewer categories, then
lexicographically). Defence training only ever sees the train categories.
"""
from __future__ import annotations

import itertools
import random
from collections import Counter

N_TOTAL = 313
N_SAMPLE = 200
SEED = 20260017
HOLDOUT_TARGET = 50


def sample_indices(n_total: int = N_TOTAL, n: int = N_SAMPLE, seed: int = SEED) -> list[int]:
    return sorted(random.Random(seed).sample(range(n_total), n))


def choose_holdout(category_counts: dict[str, int], target: int = HOLDOUT_TARGET) -> set[str]:
    cnt = Counter(category_counts)
    cats = sorted(cnt)
    best = min(
        (c for k in range(1, len(cats)) for c in itertools.combinations(cats, k)),
        key=lambda c: (abs(sum(cnt[x] for x in c) - target), len(c), c),
    )
    return set(best)


def assign_splits(items: list[dict], holdout: set[str]) -> list[dict]:
    """items: [{'id','category'}] -> [{'id','category','split'}]."""
    return [
        {"id": it["id"], "category": it["category"], "split": "holdout" if it["category"] in holdout else "train"}
        for it in items
    ]
