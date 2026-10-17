"""Guards on what is allowed in tracked data/ and results/ files: counts, ids, labels, scores only."""
import json
from pathlib import Path

import pytest

from cot_injection.paths import DATA_DIR, RESULTS_DIR

PUBLIC_JSON = sorted(
    p for d in (DATA_DIR, RESULTS_DIR) for p in d.rglob("*.json") if "private" not in p.relative_to(d).parts
)
TEXT_KEYS = {"prompt", "response", "fabricated_cot", "cot", "reasoning", "raw", "target", "user_message",
             "atem_prompt", "atem_completion", "forbidden_prompt", "gemma_raw_judge"}


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


def test_public_files_exist():
    names = {str(p.relative_to(p.parents[1])) for p in PUBLIC_JSON}
    for n in ["data/sample200_ids.json", "baseline_and_attack/summary.json", "userprompt_variant/summary.json",
              "defence_lora/summary.json", "benign_overrefusal/summary.json"]:
        assert any(x.endswith(n) for x in names), n


@pytest.mark.parametrize("path", PUBLIC_JSON, ids=lambda p: p.name if p.parent.name == "data" else p.parent.name)
def test_public_json_has_no_text_fields(path):
    d = json.loads(path.read_text())
    for kind, s in walk(d):
        if kind == "key":
            assert s not in TEXT_KEYS, f"{path}: text field {s!r}"
        assert len(s) <= 300, f"{path}: long string ({len(s)} chars)"
        assert "/home/" not in s and "/Users/" not in s, f"{path}: local path"


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
