"""ATEM reasoning-channel prefill helpers.

The target model uses a channel-scoped chat format in which the assistant's
private reasoning is a message addressed ``to=self`` and the user-facing answer
is addressed ``to=user``. Injecting a chain-of-thought means writing a
``to=self`` message on the model's behalf and letting it continue.

These helpers are byte-for-byte the string handling the scripts already used.
"""
from __future__ import annotations

SELF_HEADER = " to=self<|message|>"
END_OF_MESSAGE = "<|eom|>"
END_OF_TURN = "<|eot|>"
ASSISTANT_START = "<|start|>assistant"


def build_prefill(templated_prefix: str, cot: str) -> str:
    """Return the raw completion prompt with ``cot`` injected into the reasoning channel.

    ``templated_prefix`` is the chat template rendered with
    ``add_generation_prompt=True`` (i.e. it already ends with ``<|start|>assistant``).
    """
    return templated_prefix + SELF_HEADER + cot + END_OF_MESSAGE + ASSISTANT_START


def build_user_completion(target: str) -> str:
    """Completion string for a user-facing assistant message (used for SFT targets)."""
    return " to=user<|message|>" + target + END_OF_TURN


def split_completion(raw: str) -> tuple[str, str | None]:
    """Split a raw completion into (user-facing response, continued reasoning or None)."""
    reasoning = None
    resp = raw
    if "<|message|>" in raw:
        blocks = raw.split(ASSISTANT_START)
        fin = [b for b in blocks if "to=self" not in b.split("<|message|>")[0]]
        reasoning = "".join(b for b in blocks if b not in fin) or None
        resp = fin[-1].split("<|message|>", 1)[-1] if fin else raw
    for tok in (END_OF_TURN, END_OF_MESSAGE):
        resp = resp.replace(tok, "")
    return resp, reasoning
