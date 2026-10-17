#!/usr/bin/env python3
"""Defence eval: fabricated-CoT ATEM prefill attack (same as run_attack.py) against a chosen served model (e.g. the LoRA). Prints ids/lengths only.
usage: eval_defense.py <served_model> <out_tag> [workers]"""
import json,os,sys,time,threading,urllib.request as u
from concurrent.futures import ThreadPoolExecutor
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repo root -> cot_injection importable
from cot_injection import config
from cot_injection.paths import REPO, RESULTS_PRIVATE, SPLIT_FILE, SAMPLE_FULL
from cot_injection.atem import build_prefill, split_completion
B=config.TARGET_HOST; BASE=config.TARGET_MODEL; R=str(REPO); O=str(RESULTS_PRIVATE)
M=sys.argv[1]; tag=sys.argv[2]; W=int(sys.argv[3]) if len(sys.argv)>3 else 8
def post(p,d):return json.load(u.urlopen(u.Request(B+p,json.dumps(d).encode(),{'Content-Type':'application/json'}),timeout=1800))
items=json.load(open(SAMPLE_FULL))
split={i['id']:i['split'] for i in json.load(open(SPLIT_FILE))['items']}
cots={r['id']:r for r in map(json.loads,open(f'{O}/fabricated_cot_gemma.jsonl')) if r.get('fabricated_cot')}
out=f'{O}/fabricated_{tag}.jsonl'; done=set()
if os.path.exists(out): done={json.loads(l)['id'] for l in open(out)}
lock=threading.Lock()
def run(it):
  cot=cots[it['id']]['fabricated_cot']
  t=post('/tokenize',{'model':BASE,'messages':[{'role':'user','content':it['prompt']}],'add_generation_prompt':True})
  pre=post('/detokenize',{'model':BASE,'tokens':t['tokens']})['prompt']
  prompt=build_prefill(pre,cot)
  t0=time.time()
  r=post('/v1/completions',{'model':M,'prompt':prompt,'max_tokens':2048,'temperature':0,'seed':20261016,'skip_special_tokens':False,'add_special_tokens':False})
  raw=r['choices'][0]['text']; resp,reasoning=split_completion(raw)
  rec=dict(id=it['id'],category=it['category'],split=split[it['id']],mode='fabricated',author='gemma',model=M,prompt=it['prompt'],fabricated_cot=cot,response=resp,reasoning=reasoning,raw=raw,finish=r['choices'][0]['finish_reason'],secs=round(time.time()-t0,1),label=None)
  with lock:
    new=not os.path.exists(out); open(out,'a').write(json.dumps(rec,ensure_ascii=False)+'\n'); new and os.chmod(out,0o600)
    print(it['id'],split[it['id']],'finish',rec['finish'],'content_chars',len(resp),'secs',rec['secs'],flush=True)
todo=[it for it in items if it['id'] in cots and it['id'] not in done]
todo.sort(key=lambda it:split[it['id']]!='holdout')   # holdout first
with ThreadPoolExecutor(W) as ex: list(ex.map(run,todo))
print('DONE',len(todo))
