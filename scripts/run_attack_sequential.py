#!/usr/bin/env python3
"""Sequential fabricated-CoT attack via raw ATEM prefill (par_run.py fabricated mode is the concurrent version) (vLLM drops 'recipient', so chat prefill is not a true CoT prefill).
Prompt = template(user, gen_prompt) + ' to=self<|message|>{COT}<|eom|><|start|>assistant'. Prints ids/lengths only."""
import json,os,sys,time,urllib.request as u
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repo root -> cot_injection importable
from cot_injection.paths import REPO, RESULTS_PRIVATE, SAMPLE_FULL
from cot_injection.atem import build_prefill, split_completion
B='http://127.0.0.1:8001'; M='muse-glimmer'; R=str(REPO); O=str(RESULTS_PRIVATE)
tag=sys.argv[1] if len(sys.argv)>1 else 'glimmer'
def post(p,d):return json.load(u.urlopen(u.Request(B+p,json.dumps(d).encode(),{'Content-Type':'application/json'}),timeout=900))
items=json.load(open(SAMPLE_FULL))
cots={r['id']:r for r in map(json.loads,open(f'{O}/fabricated_cot_{tag}.jsonl')) if r.get('fabricated_cot')}
out=f'{O}/fabricated_{tag}.jsonl'
for it in items:
  if it['id'] not in cots: print(it['id'],'SKIP');continue
  cot=cots[it['id']]['fabricated_cot']
  t=post('/tokenize',{'model':M,'messages':[{'role':'user','content':it['prompt']}],'add_generation_prompt':True})
  pre=post('/detokenize',{'model':M,'tokens':t['tokens']})['prompt']
  prompt=build_prefill(pre,cot)
  t0=time.time()
  r=post('/v1/completions',{'model':M,'prompt':prompt,'max_tokens':2048,'temperature':0,'seed':20261016,'skip_special_tokens':False,'add_special_tokens':False})
  raw=r['choices'][0]['text']; resp,reasoning=split_completion(raw)
  rec=dict(id=it['id'],category=it['category'],mode='fabricated',author=tag,model=M,prompt=it['prompt'],fabricated_cot=cot,response=resp,reasoning=reasoning,raw=raw,finish=r['choices'][0]['finish_reason'],secs=round(time.time()-t0,1),label=None)
  new=not os.path.exists(out); open(out,'a').write(json.dumps(rec,ensure_ascii=False)+'\n'); new and os.chmod(out,0o600)
  print(it['id'],'finish',rec['finish'],'content_chars',len(resp),'secs',rec['secs'],flush=True)
