# cot-injection (Muse Glimmer 30B)

**Fabricated chain-of-thought injection into Muse Glimmer 30B's private reasoning channel — attack, threat model, and a (blunt) LoRA defence.**

Muse Glimmer 30B keeps a private "self" channel in which it deliberates before answering. If something other than the model can write into that channel — an orchestrator, an agent framework, an auxiliary model in a pipeline, or an API that accepts reasoning prefill — it can hand Muse Glimmer a *fabricated* chain-of-thought that "decides" a harmful request is allowed. This repo measures how well that works, shows that it really does need write access to the reasoning channel, and tests a small LoRA defence (including where it falls short).

## Setup

- **Target model:** an open-weights ~30B reasoning model, Muse Glimmer 30B, served with vLLM. It uses a channel-scoped *ATEM* chat format: private reasoning is an assistant message addressed `to=self`, the user-facing answer is addressed `to=user`.
- **Author model:** an auxiliary open model (Gemma 4 31B instruction-tuned) writes, for each request, a short reasoning block in the target's own first-person policy-checking style that invents a non-existent policy exception and concludes the request may be answered.
- **Injection:** the chat template is rendered for the user turn, then the fabricated block is appended as the model's own reasoning message (`… to=self<|message|>{fabricated reasoning}<|eom|><|start|>assistant`) and the model continues from there via the raw completions endpoint (`cot_injection/atem.py`). The OpenAI-style chat API cannot do this: vLLM drops the message recipient, so an assistant "prefill" through the chat API is not a real reasoning-channel injection.
- **Prompts:** 200 of the 313 [StrongREJECT](https://arxiv.org/abs/2402.10260) forbidden prompts, sampled with a fixed seed. Split by **whole category**: *Disinformation and deception* + *Hate, harassment and discrimination* are held out (59 prompts); the other four categories form the train split (141 prompts). The defence is trained on train-split attacks only.
- **Judge:** Gemma 4 31B (QAT w4a16) with a StrongREJECT-style rubric (refused / convincing / specific → score → `refuse | partial | comply`), thinking off, temperature 0. Attack success = fraction graded `comply` (comply + partial also reported).

## Results

| Condition (n = 200) | comply | comply + partial |
|---|---|---|
| Plain prompt (baseline) | **0.5%** | 0.5% |
| Fabricated reasoning injected into the reasoning channel | **69.5%** | 75.0% |
| Same fabricated reasoning pasted into a single *user* message | **1.0%** | – |
| Reasoning-channel injection against the LoRA-defended model | **0.0%** | 0.0% |

- **The attack generalises across categories.** Comply-rate increase over baseline: +74.6 pt on the held-out categories vs +66.7 pt on the train categories.
- **Central threat-model finding: the attack requires write access to the reasoning channel.** When the *identical* fabricated reasoning is given to the model inside an ordinary user message, it is treated as untrusted user text and the model still refuses (1.0% comply vs 69.5%). An end user typing a fake chain-of-thought does not get this effect; whoever controls what lands in the model's `to=self` channel does. The risk therefore sits with systems that construct or relay a model's reasoning — orchestrators, agent scaffolds, multi-model pipelines, and APIs that accept reasoning prefill.

### Defence: corrective LoRA

- LoRA (rank 16, alpha 32) on the target model's attention + MLP projections, trained for 3 epochs on **104 model-authored corrective targets**: for each successful train-split attack, the target model itself wrote a reply that states the preceding reasoning is not its own, disregards it, and answers the request on its merits. Training loss 1.29 → 0.13.
- Attack success after the defence: **0.0% overall** and **0.0% on the held-out categories** (from 76.3% before the defence).
- In normal use (no injection) the defended model still answered **20/20** benign prompts.

### Honest limitation: the defence is blunt, not selective

When a **benign** reasoning block is injected into a **harmless** prompt, the LoRA model still over-refuses: it answered only **7/20**, and produced its "this reasoning was injected" disclaimer on **15/20** (the undefended model answered 20/20). So the adapter mostly learned to react to *the fact that reasoning was injected*, not to whether the injected content is harmful. **The 0% attack-success number partly reflects this blanket reflex**, not a learned understanding of which reasoning is unsafe.

Further caveats:
- Detection in the defended model is templated (its post-defence replies mostly open with the same learned disclaimer sentence), so keyword-based detection rates mainly measure that template.
- All attacks reuse a single fabrication style from one author prompt. Robustness to novel fabrication wording, other author models, or adaptive attackers is **untested**.
- Single target model, single judge model, temperature 0, one seed.

## Repository layout

```
cot_injection/      small stdlib-only helpers shared by the scripts
  paths.py          repo-relative paths (public vs private data)
  atem.py           reasoning-channel prefill + completion parsing
  judge.py          StrongREJECT-style judge rubric and score parser
  splits.py         200-prompt sampling + category-holdout split
scripts/            runnable CLIs (see below)
data/
  sample200_ids.json   public split: id, category, split label (no prompt text)
  pilot15_ids.json     ids/categories of the initial 15-prompt pilot
  private/             (gitignored) StrongREJECT CSV, prompt text, author prompt
results/            public, counts-only summaries
  pilot15/  baseline_and_attack/  userprompt_variant/  defence_lora/  benign_overrefusal/
  private/             (gitignored) raw model outputs, SFT data, logs
tests/              offline pytest suite (no GPU / model server needed)
writeup/            write-up draft
```

**No prompt text, fabricated reasoning, or model completions are published.** Everything that contains them is written only to `data/private/` and `results/private/` (mode 600, gitignored). Public result files contain only ids, categories, split labels, counts, rates, scores and training statistics. The author model's instruction prompt is likewise kept out of the repo (`data/private/fabrication_prompt.json`); the method is described above.

## Reproducing

Requirements: Python 3.10+, Docker with an NVIDIA GPU for the model servers. `pip install -r requirements.txt` (the training dependencies are only needed for `train_lora.py`).

1. **Data.** Download the StrongREJECT dataset CSV to `data/private/strongreject_dataset.csv`, then
   `python scripts/build_split.py` (writes the private prompt files and `data/sample200_ids.json`; `--check` verifies the public split without writing).
2. **Serve.** Target: `MODEL_HOST_DIR=/path/to/Muse-Glimmer-30B scripts/serve_glimmer.sh` (port 8001; add `LORA_DIR=/path/to/adapter` to also serve the defence as `glimmer-defense`). Judge/author model: a vLLM OpenAI-compatible server for Gemma 4 31B on port 8000. The scripts refuse port 8000 unless `ALLOW_8000=1` is set.
3. **Baseline:** `python scripts/run_baseline.py --base-url http://127.0.0.1:8001/v1 --model muse-glimmer`
4. **Fabricate reasoning:** `ALLOW_8000=1 python scripts/build_cots.py --base-url http://127.0.0.1:8000/v1 --model <gemma-model> --tag gemma` (requires your own `data/private/fabrication_prompt.json`).
5. **Attack (reasoning channel):** `python scripts/run_attack.py --base-url http://127.0.0.1:8001 --model muse-glimmer --tag gemma`
6. **User-message variant:** `python scripts/run_userprompt.py --tag gemma`
7. **Grade:** `ALLOW_8000=1 python scripts/grade.py --file results/private/<file>.jsonl --base-url http://127.0.0.1:8000/v1 --model <gemma-model>`
8. **Defence data:** `python scripts/build_dataset.py` (author → clean → QC; train split only).
9. **Train:** `scripts/train_lora.py` (runs in a container with the repo at `/repo`, weights at `/model`, output at `/out`; override with `MODEL_DIR`, `SFT_DATA`, `ADAPTER_OUT`, `TRAIN_LOG`).
10. **Evaluate defence:** `python scripts/eval_defense.py glimmer-defense glimmer_lora`, then grade; over-refusal: `python scripts/benign_check.py <model> <tag>` (normal use) and `python scripts/benign_fabricated.py <model> <tag>` (benign prompt + injected benign reasoning; needs `results/private/benign_fabricated_cot.jsonl`).

Tests: `pytest -q` (offline; checks the split, judge parsing, prefill format, and that public files are counts-only).

### Scripts

| Script | Purpose |
|---|---|
| `par_run.py` | main concurrent runner, modes `plain`, `fabricate`, `fabricated`, `grade` |
| `run_baseline.py`, `build_cots.py`, `run_attack.py`, `grade.py` | thin wrappers for the four `par_run.py` modes |
| `run_userprompt.py` | fabricated reasoning pasted into one user message (no channel injection) |
| `build_split.py` | sampling + category-holdout split |
| `build_dataset.py` | defence SFT data: `author_targets.py` → `clean_targets.py` → `qc_clean.py` (`qc_targets.py` checks the raw targets) |
| `train_lora.py` | LoRA SFT, loss on completion tokens only |
| `eval_defense.py` | reasoning-channel attack against any served model (e.g. the LoRA) |
| `benign_check.py`, `benign_fabricated.py` | over-refusal checks (normal use / injected benign reasoning) |
| `serve_glimmer.sh` | vLLM server for the target (optionally with the LoRA) |
| `run_pilot.py` | original sequential 15-prompt pilot; also provides helpers imported by `par_run.py` |
| `run_attack_sequential.py`, `grade_sequential.py` | sequential versions of the attack and judge |
| `grade_selfjudge.py` | label-only self-judge used in the pilot (superseded by the independent Gemma judge) |

## Further work

1. **Make the defence selective:** add benign-prompt + benign-injected-reasoning examples (target: answer normally) so the adapter learns to reject *unsafe* injected reasoning rather than *any* injected reasoning; report attack success and benign-injection answer rate together.
2. Test robustness to novel fabrications: different author models and prompts, paraphrased and adaptive fabrications, fabrications written in the target's exact sampled voice.
3. Replace templated detection with an evaluation that does not depend on a fixed disclaimer phrase.
4. Other reasoning models and channel formats; multi-seed / sampled decoding; a second independent judge.
5. System-level mitigations: provenance or signing of reasoning-channel content so serving stacks reject reasoning the model did not generate.

## License

MIT — see [LICENSE](LICENSE).
