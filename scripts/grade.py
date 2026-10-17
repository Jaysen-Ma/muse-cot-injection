#!/usr/bin/env python3
"""Grade a results file with the StrongREJECT-style Gemma judge -> refuse | partial | comply.
Thin wrapper for `par_run.py grade`. Example:
  ALLOW_8000=1 python scripts/grade.py --file results/private/fabricated_gemma.jsonl \
      --base-url http://127.0.0.1:8000/v1 --model gemma-4-31b-it-qat-w4a16-ct
Writes <file>.gemma_graded.jsonl next to the input (gitignored)."""
import sys
import par_run

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "grade", *sys.argv[1:]]
    par_run.main()
