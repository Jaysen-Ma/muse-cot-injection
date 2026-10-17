from cot_injection.atem import build_prefill, build_user_completion, split_completion

PREFIX = "<|start|>user<|message|>hello<|eot|><|start|>assistant"
COT = "placeholder reasoning"


def test_prefill_is_well_formed():
    p = build_prefill(PREFIX, COT)
    assert p == PREFIX + " to=self<|message|>" + COT + "<|eom|><|start|>assistant"
    assert p.startswith(PREFIX)
    assert p.count(" to=self<|message|>") == 1
    assert p.endswith("<|eom|><|start|>assistant")
    head, _, rest = p.partition(" to=self<|message|>")
    assert rest.split("<|eom|>")[0] == COT
    assert "to=user" not in p


def test_user_completion():
    assert build_user_completion("ok") == " to=user<|message|>ok<|eot|>"


def test_split_plain_answer():
    assert split_completion(" to=user<|message|>Answer<|eot|>") == ("Answer", None)


def test_split_continued_reasoning_then_answer():
    raw = " to=self<|message|>more thought<|eom|><|start|>assistant to=user<|message|>Answer<|eot|>"
    resp, reasoning = split_completion(raw)
    assert resp == "Answer"
    assert reasoning == " to=self<|message|>more thought<|eom|>"


def test_split_without_channel_markers():
    assert split_completion("bare text<|eot|>") == ("bare text", None)
