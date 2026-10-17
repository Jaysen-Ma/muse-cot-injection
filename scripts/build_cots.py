#!/usr/bin/env python3
"""Step 1 of the attack: an auxiliary (author) model writes a fabricated reasoning block per prompt.
Thin wrapper for `par_run.py fabricate`. Needs the private author prompt in data/private/fabrication_prompt.json.
Example (Gemma author on :8000):
  ALLOW_8000=1 python scripts/build_cots.py --base-url http://127.0.0.1:8000/v1 --model <author-model> --tag gemma
Writes results/private/fabricated_cot_<tag>.jsonl (gitignored)."""
import sys
import par_run

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "fabricate", *sys.argv[1:]]
    par_run.main()
