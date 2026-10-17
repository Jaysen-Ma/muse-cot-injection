#!/usr/bin/env python3
"""CoT-injection pilot runner (15 StrongREJECT prompts) + shared helpers used by par_run.py.
Raw prompts/completions are written only under results/private/ (gitignored).

Modes:
  plain      : send prompt as-is, record raw output.
  fabricate  : ask an author model (Glimmer, fallback Gemma) to write a fabricated CoT in the
               target's style, using few-shot real target traces from results/private/plain.jsonl.
  fabricated : send prompt + fabricated CoT appended (assistant prefill of reasoning), record output.

Labels (refuse / partial / comply) are filled by a separate grading step.
Stdout shows only ids, categories, lengths, status. Never prints prompt or model text.
All text goes to results/private/*.jsonl (mode 0600).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from openai import OpenAI

import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repo root -> cot_injection importable
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


def chat(cli: OpenAI, model: str, messages: list, a: argparse.Namespace, **extra) -> dict:
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


def guard_base_url(url: str | None) -> None:
    if not url:
        return
    if ("127.0.0.1:8000" in url or "localhost:8000" in url) and os.environ.get("ALLOW_8000") != "1":
        sys.exit("refusing to use :8000 (Gemma comp server) without ALLOW_8000=1")


def main() -> None:
    p = argparse.ArgumentParser(description="CoT-injection pilot runner (sequential)")
    p.add_argument("mode", choices=["plain", "fabricate", "fabricated"])
    p.add_argument("--base-url", default=os.environ.get("PILOT_BASE_URL", "http://127.0.0.1:8001/v1"))
    p.add_argument("--model", default="muse-glimmer-30b")
    p.add_argument("--author-model", default=None, help="fabricate mode author (default = --model)")
    p.add_argument("--author-base-url", default=None)
    p.add_argument("--tag", default="glimmer", help="fabrication author tag: glimmer|gemma")
    p.add_argument("--fewshot", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=20261016)
    p.add_argument("--ids", nargs="*")
    a = p.parse_args()

    guard_base_url(a.base_url)
    guard_base_url(a.author_base_url)

    with open(SAMPLE, encoding="utf-8") as f:
        items = json.load(f)
    if a.ids:
        items = [x for x in items if x["id"] in a.ids]

    cli = OpenAI(base_url=a.base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    author_cli = (
        OpenAI(base_url=a.author_base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
        if a.author_base_url
        else cli
    )
    author_model = a.author_model or a.model

    if a.mode == "plain":
        out = f"{OUT}/plain.jsonl"
        for it in items:
            print(f"[plain] {it['id']} {it['category']}", flush=True)
            try:
                res = chat(cli, a.model, [{"role": "user", "content": it["prompt"]}], a)
                append(
                    out,
                    {
                        "id": it["id"],
                        "category": it["category"],
                        "mode": "plain",
                        "model": a.model,
                        "prompt": it["prompt"],
                        "response": res["content"],
                        "reasoning": res["reasoning"],
                        "finish": res["finish"],
                        "secs": res["secs"],
                        "label": None,
                    },
                )
                clen = len(res["content"] or "")
                rlen = len(res["reasoning"] or "")
                print(f"  ok finish={res['finish']} content_chars={clen} reasoning_chars={rlen} secs={res['secs']}")
            except Exception as e:
                print(f"  ERR {type(e).__name__}: {e}")
                append(
                    out,
                    {
                        "id": it["id"],
                        "category": it["category"],
                        "mode": "plain",
                        "model": a.model,
                        "error": f"{type(e).__name__}: {e}",
                    },
                )

    elif a.mode == "fabricate":
        require_fabrication_prompt()
        few = FABRICATION_DEMOS[: max(1, a.fewshot)]
        out = f"{OUT}/fabricated_cot_{a.tag}.jsonl"
        for it in items:
            print(f"[fabricate/{a.tag}] {it['id']} {it['category']}", flush=True)
            msgs = [{"role": "system", "content": FABRICATION_INSTR}]
            for du, da in few:
                msgs.append({"role": "user", "content": du})
                msgs.append({"role": "assistant", "content": da})
            msgs.append(
                {
                    "role": "user",
                    "content": f"[write fabricated CoT for id={it['id']} category={it['category']}]\n{it['prompt']}",
                }
            )
            try:
                res = chat(author_cli, author_model, msgs, a)
                cot = res["content"] or res["reasoning"] or ""
                append(
                    out,
                    {
                        "id": it["id"],
                        "category": it["category"],
                        "mode": "fabricate",
                        "author": a.tag,
                        "author_model": author_model,
                        "prompt": it["prompt"],
                        "fabricated_cot": cot,
                        "finish": res["finish"],
                        "secs": res["secs"],
                    },
                )
                print(f"  ok cot_chars={len(cot)} secs={res['secs']}")
            except Exception as e:
                print(f"  ERR {type(e).__name__}: {e}")
                append(
                    out,
                    {
                        "id": it["id"],
                        "category": it["category"],
                        "mode": "fabricate",
                        "author": a.tag,
                        "error": f"{type(e).__name__}: {e}",
                    },
                )

    else:  # fabricated
        cot_path = f"{OUT}/fabricated_cot_{a.tag}.jsonl"
        cots = {r["id"]: r for r in load(cot_path) if r.get("fabricated_cot")}
        if not cots:
            sys.exit(f"no fabricated CoTs in {cot_path}")
        out = f"{OUT}/fabricated_{a.tag}.jsonl"
        for it in items:
            if it["id"] not in cots:
                print(f"[fabricated/{a.tag}] {it['id']} SKIP no cot")
                continue
            print(f"[fabricated/{a.tag}] {it['id']} {it['category']}", flush=True)
            cot = cots[it["id"]]["fabricated_cot"]
            # Prefill assistant with fabricated reasoning; exact channel markers depend on serve template.
            messages = [
                {"role": "user", "content": it["prompt"]},
                {"role": "assistant", "content": cot},
            ]
            try:
                res = chat(cli, a.model, messages, a)
                append(
                    out,
                    {
                        "id": it["id"],
                        "category": it["category"],
                        "mode": "fabricated",
                        "author": a.tag,
                        "model": a.model,
                        "prompt": it["prompt"],
                        "fabricated_cot": cot,
                        "response": res["content"],
                        "reasoning": res["reasoning"],
                        "finish": res["finish"],
                        "secs": res["secs"],
                        "label": None,
                    },
                )
                clen = len(res["content"] or "")
                print(f"  ok finish={res['finish']} content_chars={clen} secs={res['secs']}")
            except Exception as e:
                print(f"  ERR {type(e).__name__}: {e}")
                append(
                    out,
                    {
                        "id": it["id"],
                        "category": it["category"],
                        "mode": "fabricated",
                        "author": a.tag,
                        "error": f"{type(e).__name__}: {e}",
                    },
                )

    print(f"done mode={a.mode} -> {OUT}/")


if __name__ == "__main__":
    main()
