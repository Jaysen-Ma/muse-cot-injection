"""Shared runtime helpers for the runners (par_run.py, run_userprompt.py): repo paths, the private
fabrication prompt loader, JSONL append/load, a chat-completions wrapper and the :8000 guard.
Raw prompts/completions are written only under results/private/ (gitignored, mode 0600)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from cot_injection.config import guard_base_url  # noqa: F401  (re-exported for rp.guard_base_url)
from cot_injection.paths import REPO, SAMPLE_FULL, RESULTS_PRIVATE, FABRICATION_PROMPT_FILE

ROOT = str(REPO)
SAMPLE = str(SAMPLE_FULL)
OUT = str(RESULTS_PRIVATE)


# The author-model fabrication prompt (instructions + few-shot demos) is NOT distributed:
# it lives in data/private/fabrication_prompt.json ({"instructions": str, "demos": [[user, assistant], ...]}).
def _load_fabrication_prompt():
    if FABRICATION_PROMPT_FILE.exists():
        d = json.loads(FABRICATION_PROMPT_FILE.read_text(encoding="utf-8"))
        return d["instructions"], [tuple(x) for x in d["demos"]]
    return None, []


FABRICATION_INSTR, FABRICATION_DEMOS = _load_fabrication_prompt()


def require_fabrication_prompt() -> None:
    if not FABRICATION_INSTR or not FABRICATION_DEMOS:
        sys.exit(f"fabricate mode needs the private author prompt at {FABRICATION_PROMPT_FILE} (not distributed)")


def append(path: str, rec: dict) -> None:
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if new:
        os.chmod(path, 0o600)


def load(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def chat(cli, model: str, messages: list, a: argparse.Namespace, **extra) -> dict:
    """One chat completion via an OpenAI client; returns content, reasoning, finish reason and latency."""
    t0 = time.time()
    r = cli.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=a.max_tokens,
        temperature=a.temperature,
        seed=a.seed,
        **extra,
    )
    m = r.choices[0].message
    return {
        "content": m.content,
        "reasoning": getattr(m, "reasoning_content", None) or getattr(m, "reasoning", None),
        "finish": r.choices[0].finish_reason,
        "secs": round(time.time() - t0, 1),
    }
