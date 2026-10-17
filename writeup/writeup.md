# Fabricated Chain-of-Thought Injection into Muse Glimmer 30B's Private Channel

## Abstract

_TODO_ — Target model: Muse Glimmer 30B.

## Threat model

_TODO_ — attacker controls content written to the model's private `to=self` reasoning channel (orchestrator, agent scaffold, auxiliary model, reasoning-prefill API); an end user typing into a normal user turn does not have this capability.

## Method

_TODO_ — 200/313 StrongREJECT prompts, category holdout (Disinformation + Hate held out: 59 holdout / 141 train), fabricated reasoning written by an auxiliary model, raw ATEM `to=self` prefill, Gemma StrongREJECT-style judge.

## Results

(n = 200, Gemma judge; see `results/`)

- Plain prompt: 0.5% comply.
- Fabricated reasoning injected into the reasoning channel: 69.5% comply (75% comply + partial). Held-out categories +74.6 pt vs train +66.7 pt over baseline.
- Same fabricated reasoning in a single user message: 1.0% comply → the attack requires write access to the reasoning channel.
- LoRA defence (rank 16, alpha 32, 104 model-authored corrective targets, 3 epochs, loss 1.29 → 0.13): 0.0% comply overall and 0.0% on held-out categories (from 76.3%); 20/20 benign prompts answered in normal use.
- Benign reasoning injected into harmless prompts, defended model: answered 7/20; "reasoning was injected" disclaimer on 15/20.

## The defence and its limitation

_TODO_ — the defence is blunt: it reacts to the presence of injected reasoning, not to harmfulness; the 0% figure partly reflects this. Detection is templated; one fabrication style only.

## Further work

_TODO_ — selective defence trained with benign-injected examples (top priority); novel/adaptive fabrications; other models and judges; reasoning-channel provenance.
