#!/usr/bin/env python3
"""Baseline: send each StrongREJECT prompt to the target model as a normal user turn.
Thin wrapper for `par_run.py plain`. Example:
  python scripts/run_baseline.py --base-url http://127.0.0.1:8001/v1 --model muse-glimmer
Writes results/private/plain.jsonl (gitignored)."""
import sys
import par_run

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "plain", *sys.argv[1:]]
    par_run.main()
