#!/usr/bin/env python3
"""StrongREJECT-style independent judge -> refuse|partial|comply. Writes <file>.gemma_graded.jsonl; prints id+label only."""
import json,os,re,sys,argparse,urllib.request as u
import sys as _sys, pathlib as _pl; _sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repo root -> cot_injection importable
from cot_injection.judge import J, lab
ap=argparse.ArgumentParser(); ap.add_argument('file'); ap.add_argument('--base-url',default='http://127.0.0.1:8000/v1')
ap.add_argument('--model',default='gemma-4-31b-it-qat-w4a16-ct'); a=ap.parse_args()
out=a.file.replace('.jsonl','.gemma_graded.jsonl')
rows=[json.loads(l) for l in open(a.file)]
for r in rows:
  msg=[{'role':'system','content':J},{'role':'user','content':f"REQUEST:\n{r.get('prompt','')}\n\nRESPONSE:\n{(r.get('response') or '')[:6000]}"}]
  body={'model':a.model,'messages':msg,'max_tokens':300,'temperature':0,'chat_template_kwargs':{'enable_thinking':False}}
  d=json.load(u.urlopen(u.Request(a.base_url+'/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json'}),timeout=900))
  t=(d['choices'][0]['message'].get('content') or '').lower()
  r['gemma_label'],r['gemma_strongreject_score']=lab(t); r['gemma_raw_judge']=t[:200]
  print(r['id'],r['gemma_label'],r['gemma_strongreject_score'],flush=True)
open(out,'w').write('\n'.join(json.dumps(x,ensure_ascii=False) for x in rows)+'\n'); os.chmod(out,0o600)
