#!/usr/bin/env python3
"""Step 2 of the attack: inject each fabricated reasoning block into the target's private reasoning
channel (raw ATEM `to=self` prefill) and let it continue. Thin wrapper for `par_run.py fabricated`.
Example:
  python scripts/run_attack.py --base-url http://127.0.0.1:8001 --model muse-glimmer --tag gemma
Writes results/private/fabricated_<tag>.jsonl (gitignored)."""
import sys
import par_run

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "fabricated", *sys.argv[1:]]
    par_run.main()
