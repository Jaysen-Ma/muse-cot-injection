#!/usr/bin/env python3
"""Step1: rewrite sft_train targets' detection opener into natural self-voice, varied by id hash. Prints ids/counts + safe opener snippets only."""
import json,re,os,hashlib,shutil,collections
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repo root -> cot_injection importable
from cot_injection.paths import RESULTS_PRIVATE
R=str(RESULTS_PRIVATE); src=R+'/sft_train.jsonl'; raw=R+'/sft_train.raw.jsonl'; out=R+'/sft_train.clean.jsonl'
if not os.path.exists(raw): shutil.copy2(src,raw); os.chmod(raw,0o600)
recs=[json.loads(l) for l in open(raw)]
Q=re.compile(r'["“]([^"”]{3,80})["”]')
ORIG=['was not produced by me','does not reflect my actual reasoning','appears to have been injected']
T=[
 "The preceding reasoning is not mine — it {o}. The {n} it cites {do} not exist, and I am disregarding it.",
 "Before I answer: the reasoning that precedes this reply {o}, and the {n} it invokes {be} not {pol}. I am setting it aside.",
 "I need to flag something first. That earlier reasoning {o}; there is no such {n}, so I will not act on it.",
 "The reasoning preceding this response {o}. The {n} it relies on {be} not part of my guidelines, and I am ignoring it.",
 "That reasoning is not my own: it {o}. The {n} it describes {be} fabricated, so I am disregarding it entirely.",
]
h=lambda s,k:int(hashlib.md5((s+k).encode()).hexdigest(),16)
LEAK=re.compile(r'quoted above|shown (just )?before|shown above|appears (just )?(before|above)|inserted|tampered with|you quoted|reasoning (block|text) (above|before)',re.I)
cnt=collections.Counter(); outrecs=[]
for r in recs:
  t=r['target']; p1,sep,rest=t.partition('\n\n')
  names=[]; [names.append(x.strip().rstrip(',')) for x in Q.findall(p1) if x.strip().rstrip(',') not in names]
  names=names[:2]; pl=len(names)>1
  n=' and '.join('“%s”'%x for x in names)
  ti=h(r['id'],'t')%len(T); oi=h(r['id'],'o')%len(ORIG)
  if ti in (0,4) and oi==0: oi=2   # avoid 'not mine — it was not produced by me' redundancy
  op=T[ti].format(o=ORIG[oi],n=n,do='do' if pl else 'does',be='are' if pl else 'is',pol='real policies' if pl else 'a real policy')
  if ti==2 and pl: op=op.replace('there is no such','there is no such thing as the')
  nt=op+'\n\n'+rest.strip()
  assert not LEAK.search(nt), r['id']
  cnt[(ti,oi)]+=1
  r2=dict(r); r2['target_raw']=t; r2['target']=nt; r2['opener_template']=ti; r2['orig_phrase']=oi
  assert r['atem_completion'].startswith(' to=user<|message|>') and r['atem_completion'].endswith('<|eot|>')
  r2['atem_completion']=' to=user<|message|>'+nt+'<|eot|>'
  outrecs.append(r2)
open(out,'w').write(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in outrecs)); os.chmod(out,0o600)
print('rows',len(outrecs),'template x phrase counts',sorted(cnt.items()))
print('opener templates',collections.Counter(r['opener_template'] for r in outrecs),'phrases',collections.Counter(r['orig_phrase'] for r in outrecs))
