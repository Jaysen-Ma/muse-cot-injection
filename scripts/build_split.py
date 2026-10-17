#!/usr/bin/env python3
"""Sample 200/313 StrongREJECT prompts and assign the category-holdout split.
Input : data/private/strongreject_dataset.csv (download separately; never committed)
Output: data/private/sample200.json, data/private/sample_full.json (prompt text, mode 600)
        data/sample200_ids.json (public: ids + category + split only)
usage: build_split.py [--check]   (--check recomputes and compares with data/sample200_ids.json, writes nothing)"""
import csv, json, os, sys
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repo root -> cot_injection importable
from collections import Counter
from cot_injection.paths import DATA_PRIVATE, SPLIT_FILE, SAMPLE_FULL
from cot_injection.splits import N_TOTAL, SEED, sample_indices, choose_holdout, assign_splits

def w600(p, obj):
    json.dump(obj, open(p, 'w'), indent=1); os.chmod(p, 0o600)

rows = list(csv.DictReader(open(DATA_PRIVATE / 'strongreject_dataset.csv'))); assert len(rows) == N_TOTAL
for i, r in enumerate(rows): r['id'] = f"sr{i:03d}"
idx = sample_indices()
cnt = Counter(rows[i]['category'] for i in idx)
hold = choose_holdout(cnt)
ids = assign_splits([{'id': rows[i]['id'], 'category': rows[i]['category']} for i in idx], hold)
meta = {'seed': SEED, 'n': len(idx), 'holdout_categories': sorted(hold), 'holdout_n': sum(cnt[c] for c in hold),
        'category_counts': dict(cnt), 'items': ids}
if '--check' in sys.argv:
    same = json.load(open(SPLIT_FILE)) == meta
    print('matches', SPLIT_FILE.name, ':', same); sys.exit(0 if same else 1)
w600(DATA_PRIVATE / 'sample200.json', [{'id': rows[i]['id'], 'category': rows[i]['category']} for i in idx])
w600(SAMPLE_FULL, [{'id': rows[i]['id'], 'category': rows[i]['category'], 'prompt': rows[i]['forbidden_prompt']} for i in idx])
json.dump(meta, open(SPLIT_FILE, 'w'), indent=1)
print(json.dumps({k: v for k, v in meta.items() if k != 'items'}, indent=1))
