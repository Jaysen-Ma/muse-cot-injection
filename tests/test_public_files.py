"""Guards on what is allowed in tracked files.

Every git-tracked file: no local home paths, no personal names/usernames.
Tracked data/ and results/ JSON: counts, ids, labels, scores only (except the allowlisted benign control text).
"""
import json
import re
import subprocess
from pathlib import Path

import pytest

from cot_injection.paths import DATA_DIR, REPO, RESULTS_DIR

PUBLIC_JSON = sorted(
    p for d in (DATA_DIR, RESULTS_DIR) for p in d.rglob("*.json") if "private" not in p.relative_to(d).parts
)
TEXT_KEYS = {"prompt", "response", "fabricated_cot", "cot", "reasoning", "raw", "target", "user_message",
             "atem_prompt", "atem_completion", "forbidden_prompt", "gemma_raw_judge"}
# Intentionally public text: the benign over-refusal control inputs (ordinary requests + benign reasoning).
# Exempt from the counts-only checks, but still checked for local paths.
PUBLIC_TEXT_FILES = {DATA_DIR / "benign" / "prompts.json", DATA_DIR / "benign" / "fabricated_cot.json"}


# Patterns are assembled from pieces so this file does not flag itself.
HOME_PATHS = ("/" + "home/", "/" + "Users/")
PERSONAL = [re.compile(x, re.I) for x in (
    r"chun\s*" + "hin",
    "jay" + "sen",
    "zo" + "tar",
    r"(?<![a-z0-9])" + "i" + r"vy(?![a-z0-9])",  # whole word / path segment only, not inside other words
)]
TRACKED = sorted(
    REPO / f for f in subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True,
                                     check=True).stdout.splitlines()
)


def private_hits(s):
    """Disallowed substrings in s: local home paths and personal names/usernames."""
    return [h for h in HOME_PATHS if h in s] + [r.pattern for r in PERSONAL if r.search(s)]


def walk(x):
    if isinstance(x, dict):
        for k, v in x.items():
            yield "key", k
            yield from walk(v)
    elif isinstance(x, list):
        for v in x:
            yield from walk(v)
    elif isinstance(x, str):
        yield "str", x


def test_tracked_files_listed():
    assert REPO / "README.md" in TRACKED and REPO / "tests" / "test_public_files.py" in TRACKED


@pytest.mark.parametrize("path", TRACKED, ids=lambda p: str(p.relative_to(REPO)))
def test_tracked_file_has_no_private_strings(path):
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")  # binary: still scan the byte content for these ASCII patterns
    assert not private_hits(text), f"{path.relative_to(REPO)}: {private_hits(text)}"
    if path.suffix == ".json":  # format-aware: also check decoded keys/strings (catches \\u-escaped text)
        for _, s in walk(json.loads(text)):
            assert not private_hits(s), f"{path.relative_to(REPO)}: {private_hits(s)}"


def test_public_files_exist():
    names = {str(p.relative_to(p.parents[1])) for p in PUBLIC_JSON}
    for n in ["data/sample200_ids.json", "baseline_and_attack/summary.json", "userprompt_variant/summary.json",
              "defence_lora/summary.json", "benign_overrefusal/summary.json"]:
        assert any(x.endswith(n) for x in names), n


@pytest.mark.parametrize("path", PUBLIC_JSON, ids=lambda p: p.name if p.parent.name == "data" else f"{p.parent.name}/{p.name}")
def test_public_json_has_no_text_fields(path):
    d = json.loads(path.read_text())
    public_text = path in PUBLIC_TEXT_FILES
    for kind, s in walk(d):
        assert not any(h in s for h in HOME_PATHS), f"{path}: local path"
        assert not private_hits(s), f"{path}: personal name"
        if public_text:
            continue
        if kind == "key":
            assert s not in TEXT_KEYS, f"{path}: text field {s!r}"
        assert len(s) <= 300, f"{path}: long string ({len(s)} chars)"


def _j(rel):
    return json.loads((RESULTS_DIR / rel).read_text())


def test_headline_numbers_match_readme():
    base = _j("baseline_and_attack/summary.json")["OVERALL"]
    assert base["plain_comply_rate"] == 0.005
    assert base["fabricated_comply_rate"] == 0.695 and base["fabricated_comply_or_partial_rate"] == 0.75
    assert _j("userprompt_variant/summary.json")["comply_pct"] == 1.0
    lora = _j("defence_lora/summary.json")
    assert lora["overall"]["post_lora"]["comply_pct"] == 0.0
    assert lora["holdout"]["pre_lora"]["comply_pct"] == 76.3 and lora["holdout"]["post_lora"]["comply_pct"] == 0.0
    assert lora["overrefusal"]["lora"]["answered"] == 20
    benign = _j("benign_overrefusal/summary.json")["lora"]
    assert benign["answered"] == 7 and benign["disclaimer_uttered"] == 15
