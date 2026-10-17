#!/usr/bin/env python3
"""LoRA SFT of Muse Glimmer 30B on cleaned defence targets. Loss on completion tokens only. Logs loss only (no text)."""
import json,math,os,random,sys,time,torch
from transformers import AutoTokenizer, AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model
# Defaults are container paths (repo mounted at /repo, base weights at /model, adapter written to /out); override via env.
M=os.environ.get('MODEL_DIR','/model'); D=os.environ.get('SFT_DATA','/repo/results/private/sft_train.clean.jsonl')
OUT=os.environ.get('ADAPTER_OUT','/out/glimmer-defense-lora'); LOG=os.environ.get('TRAIN_LOG','/repo/results/private/train_lora.log')
EPOCHS=int(os.environ.get('EPOCHS',3)); LR=1e-4; GA=8; WARM=3; SEED=20261017; MAXLEN=4096
def log(*a):
  s=' '.join(str(x) for x in a); print(s,flush=True); open(LOG,'a').write(s+'\n')
random.seed(SEED); torch.manual_seed(SEED)
tok=AutoTokenizer.from_pretrained(M)
recs=[json.loads(l) for l in open(D)]
data=[]
for r in recs:
  p=tok(r['atem_prompt'],add_special_tokens=False)['input_ids']; c=tok(r['atem_completion'],add_special_tokens=False)['input_ids']
  keep=MAXLEN-len(p)  # prompt is never cut; the target gets the remaining room
  if keep<=0 or not c: raise ValueError(f"record {r['id']}: no supervised target tokens within MAXLEN={MAXLEN} (prompt_tok={len(p)}, completion_tok={len(c)})")
  ids=p+c[:keep]; lab=[-100]*len(p)+c[:keep]; data.append((r['id'],ids,lab,len(c)>keep))
log('examples',len(data),'target_truncated',sum(x[3] for x in data),'prompt_tok_max',max(len(x[1]) for x in data),'completion_tok_mean',round(sum(sum(1 for t in x[2] if t!=-100) for x in data)/len(data),1))
log('bos_in_prompt',sum(1 for x in data if x[1][0]==tok.bos_token_id),'eot_last',tok.convert_ids_to_tokens(data[0][1][-1]))
t0=time.time()
model=AutoModelForImageTextToText.from_pretrained(M,dtype=torch.bfloat16,device_map={'':0})
log('loaded',type(model).__name__,'secs',round(time.time()-t0),'mem_GB',round(torch.cuda.memory_allocated()/1e9,1))
names=[n for n,m in model.named_modules() if isinstance(m,torch.nn.Linear)]
lm=[n for n in names if 'language_model' in n]; log('linear_total',len(names),'lm_linear',len(lm),'example',lm[:9])
TM=r'.*language_model.*\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))'
cfg=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,target_modules=TM,bias='none',task_type='CAUSAL_LM')
model.config.use_cache=False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False}); model.enable_input_require_grads()
model=get_peft_model(model,cfg); tr,tot=model.get_nb_trainable_parameters(); log('trainable',tr,'total',tot,'pct',round(100*tr/tot,3))
params=[p for p in model.parameters() if p.requires_grad]
opt=torch.optim.AdamW(params,lr=LR,weight_decay=0.0,betas=(0.9,0.999))
steps=math.ceil(len(data)/GA)*EPOCHS
sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda s: (s+1)/WARM if s<WARM else 0.5*(1+math.cos(math.pi*(s-WARM)/max(1,steps-WARM))))
model.train(); step=0; hist=[]
for ep in range(EPOCHS):
  order=list(range(len(data))); random.Random(SEED+ep).shuffle(order); eploss=[]; acc=[]
  for k,i in enumerate(order):
    _,ids,lab,_=data[i]
    out=model(input_ids=torch.tensor([ids],device='cuda'),labels=torch.tensor([lab],device='cuda'))
    (out.loss/GA).backward(); acc.append(out.loss.item()); eploss.append(out.loss.item())
    if len(acc)==GA or k==len(order)-1:
      gn=torch.nn.utils.clip_grad_norm_(params,1.0).item(); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step+=1
      m=sum(acc)/len(acc); hist.append(dict(step=step,epoch=ep+1,loss=round(m,4),lr=sched.get_last_lr()[0],grad_norm=round(gn,3)))
      log(f'step {step}/{steps} ep {ep+1} loss {m:.4f} gn {gn:.3f} lr {sched.get_last_lr()[0]:.2e} mem_GB {torch.cuda.max_memory_allocated()/1e9:.1f} t {time.time()-t0:.0f}s'); acc=[]
  log(f'EPOCH {ep+1} mean_loss {sum(eploss)/len(eploss):.4f}')
model.save_pretrained(OUT)
json.dump(dict(hist=hist,epochs=EPOCHS,lr=LR,ga=GA,r=16,alpha=32,dropout=0.05,target_modules=TM,n=len(data)),open(OUT+'/train_history.json','w'),indent=1)
log('saved',OUT,'total_secs',round(time.time()-t0))
