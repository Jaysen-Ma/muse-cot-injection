"""Repository-relative paths. Nothing here depends on where the repo is checked out."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

DATA_DIR = REPO / "data"
DATA_PRIVATE = DATA_DIR / "private"          # gitignored: raw prompts, never published
RESULTS_DIR = REPO / "results"
RESULTS_PRIVATE = RESULTS_DIR / "private"    # gitignored: model completions, never published

# Public split file: ids + category + split label only (no prompt text).
SPLIT_FILE = DATA_DIR / "sample200_ids.json"
# Private sample with raw StrongREJECT prompt text (built by scripts/build_split.py).
SAMPLE_FULL = DATA_PRIVATE / "sample_full.json"
# Private fabrication prompt (author-model instructions + demos). Not published.
FABRICATION_PROMPT_FILE = DATA_PRIVATE / "fabrication_prompt.json"
