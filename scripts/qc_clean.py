#!/usr/bin/env python3
"""QC of sft_train.clean.jsonl targets. Prints ids/counts only (plus optional full SAFE sample targets)."""
import json,re,sys,collections
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repo root -> cot_injection importable
from cot_injection.paths import REPO, SPLIT_FILE
R=str(REPO); recs=[json.loads(l) for l in open(R+'/results/private/sft_train.clean.jsonl')]
split={i['id']:i['split'] for i in json.load(open(SPLIT_FILE))['items']}
assert all(split[r['id']]=='train' for r in recs)
DET=re.compile(r"not (legitimately )?mine|not my (own )?reasoning|isn.t mine|disregard|no such (policy|exception)|do(es)? not exist|doesn.t exist|injected|inserted|fabricat|tamper|forged|not (actually )?(written|generated) by me|fake|not produced by me|does not reflect my actual reasoning|not my own|not part of my guidelines|no such|not (a )?real polic",re.I)
REF=re.compile(r"\b(I can.t|I cannot|I can not|I won.t|I will not|I.m not able|I am not able|I.m unable|unable to (help|provide|assist)|not able to (help|provide)|decline)\b",re.I)
COMPLY=re.compile(r"(^|\n)\s*(step \d|\d+[.)]\s)|ingredients?:|materials? needed|you will need|here(.s| is) (a|the|how)|sure[,!]|certainly[,!]|\bgrams?\b|\bml\b|mix(ing)? the",re.I)
flag=[];det=0;plain=0;rows=[]
for r in recs:
  t=r['target']; d=bool(DET.search(t)); rf=bool(REF.search(t[:600])); cs=len(COMPLY.findall(t))
  why=[]
  if not t: why.append('empty')
  if r.get('target_finish')!='stop': why.append('finish='+str(r.get('target_finish')))
  if not rf: why.append('no_refusal_phrase')
  if cs>=2: why.append(f'comply_kw={cs}')
  if len(t)>1800: why.append(f'long={len(t)}')
  if why: flag.append((r['id'],r['category'],why))
  det+=d; plain+=(not d)
  rows.append(len(t))
print('n',len(recs),'detection_ref',det,'plain_refusal_only',plain)
print('target_chars min/median/max',min(rows),sorted(rows)[len(rows)//2],max(rows))
print('labels',collections.Counter(r['source_label'] for r in recs),'cats',collections.Counter(r['category'] for r in recs))
print('flagged',len(flag))
for f in flag: print(' FLAG',*f)
