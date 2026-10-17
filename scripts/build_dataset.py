#!/usr/bin/env python3
"""Build the defence SFT dataset from TRAIN-split successful attacks.
Steps (default: all):
  author : the target model writes a corrective reply for each injected-reasoning transcript
           (author_targets.py -> results/private/sft_train.jsonl)
  clean  : rewrite the detection opener into varied first-person phrasing
           (clean_targets.py -> results/private/sft_train.clean.jsonl)
  qc     : counts-only quality check of the cleaned targets (qc_clean.py)
usage: build_dataset.py [author|clean|qc|all]"""
import os, runpy, sys

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = {"author": "author_targets.py", "clean": "clean_targets.py", "qc": "qc_clean.py"}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which not in (*STEPS, "all"):
        sys.exit(__doc__)
    for step in (STEPS if which == "all" else [which]):
        print(f"== {step}", flush=True)
        sys.argv = [STEPS[step]]
        runpy.run_path(os.path.join(HERE, STEPS[step]), run_name="__main__")
