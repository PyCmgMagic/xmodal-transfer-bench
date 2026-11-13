#!/usr/bin/env python3
"""Durable v7 SFT/PT LoRA experiment, with bounded independent GPU queues."""
from __future__ import annotations
import argparse, fcntl, hashlib, json, math, os, signal, subprocess, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue, Empty
from v7_gpu_resources import TRAIN_PEAKS, EVAL_PEAKS, reserve

ROOT=Path(__file__).resolve().parents[1]
MODELS={
 'qwen3':ROOT/'model_cache/modelscope/models/Qwen--Qwen3-VL-4B-Instruct/snapshots/master',
 'qwen25':Path('/mnt/yunpan2/huggingface_cache/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5'),
 'glm46':ROOT/'model_cache/modelscope/models/ZhipuAI--GLM-4.6V-Flash/snapshots/master'}
DIRECTIONS={mod:['T2T','T2V','V2T','V2V'] for mod in ('text','vision')}

def now(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(p.read_text()) if p.is_file() else {}
def rows(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def save(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def alive(pid):
 try:
  os.kill(int(pid),0)
  s=Path(f'/proc/{pid}/stat').read_text().split(') ')[1].split()[0]
  return s!='Z'
 except (OSError,ValueError,IndexError):return False

def environment(gpu,data):
 env=os.environ.copy()
 for k in list(env):
  if k in {'NPROC_PER_NODE','NNODES','NODE_RANK','MASTER_ADDR','MASTER_PORT','LOCAL_RANK','RANK','WORLD_SIZE','LOCAL_WORLD_SIZE','GROUP_RANK','ROLE_RANK','ROLE_WORLD_SIZE'} or k.startswith('TORCHELASTIC_'):env.pop(k,None)
 env.update(CUDA_VISIBLE_DEVICES=str(gpu),ROOT_IMAGE_DIR=str(data),MAX_PIXELS='1003520',
  PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',PYTHONUNBUFFERED='1',
  MODELSCOPE_CACHE=str(ROOT/'model_cache/modelscope'),HF_HOME=str(ROOT/'model_cache/huggingface'),
  HF_DATASETS_CACHE=str(ROOT/'model_cache/huggingface/datasets'),TRANSFORMERS_CACHE=str(ROOT/'model_cache/huggingface/hub'),
  HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PATH=str(ROOT/'.venv/bin')+os.pathsep+env['PATH'])
 return env

def train_command(model,obj,dataset,out,epochs=5,smoke=False,resume=None):
 cmd=[str(ROOT/'.venv/bin/swift'),obj,'--model',str(MODELS[model]),'--dataset',str(dataset),
 '--tuner_type','lora','--target_modules','all-linear','--lora_rank','8','--lora_alpha','32','--lora_dropout','0.05',
 '--torch_dtype','bfloat16','--freeze_vit','true','--freeze_aligner','true','--per_device_train_batch_size','1',
 '--gradient_accumulation_steps','1' if smoke else '16','--learning_rate','1e-4','--num_train_epochs',str(epochs),
 '--gradient_checkpointing','true','--max_length','8192','--warmup_ratio','0' if smoke else '0.05',
 '--save_strategy','steps' if smoke else 'epoch','--save_total_limit','2','--split_dataset_ratio','0',
 '--logging_steps','1' if smoke else '5','--seed','42','--data_seed','42','--add_version','false','--output_dir',str(out),
 '--use_chat_template','true' if obj=='sft' else 'false','--loss_scale','default' if obj=='sft' else 'all',
 '--strict','true','--enable_thinking','false']
 if model=='glm46':cmd+=['--template','glm4_5v']
 if smoke:cmd+=['--max_steps','1','--save_steps','1']
 if resume:cmd+=['--resume_from_checkpoint',str(resume),'--load_args','false']
 return cmd

def final_checkpoint(folder,epoch):
 candidates=[]
 for p in folder.glob('checkpoint-*'):
  st=read(p/'trainer_state.json')
  if (p/'adapter_model.safetensors').is_file() and st.get('epoch',0)>=epoch-1e-6 and all((p/n).is_file() for n in ('adapter_config.json','optimizer.pt','scheduler.pt','rng_state.pth')):
   candidates.append((st.get('global_step',0),p))
 return max(candidates)[1] if candidates else None

def process(cmd,env,folder,name,state_name):
 model,_,mod=folder.name.split('_')
 peak=EVAL_PEAKS[model] if name in ('reload','launcher') else TRAIN_PEAKS[model][mod]
 with reserve(folder.parent.parent,env['CUDA_VISIBLE_DEVICES'],str(folder.resolve())+':'+name,peak) as reservation:
  with (folder/f'{name}.log').open('a') as log:
   p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   reservation.bind(p.pid)
   save(folder/state_name,{'status':'running','pid':p.pid,'command':cmd,'started_at':now()})
   code=p.wait()
 save(folder/state_name,{'status':'complete' if code==0 else 'failed','returncode':code,'command':cmd,'ended_at':now()})
 return code

def capacity(gpu,model,mod,folder):
 # Keep a reserve; do not terminate or change another user's GPU processes.
 required={'qwen3':18000,'qwen25':25000,'glm46':30000}[model]+(7000 if mod=='vision' else 0)
 while True:
  total,used=map(int,subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.total,memory.used','--format=csv,noheader,nounits'],text=True).strip().split(','))
  if total-used>=required:return
  save(folder/'resource_wait.json',{'status':'waiting_for_memory','gpu':gpu,'free_mib':total-used,'required_mib':required,'updated_at':now()});time.sleep(30)

def verify_sources(work,data,models=False):
 m=read(work/'data/manifest.json')
 for x in m['sources'].values():assert sha(Path(x['path']))==x['sha256'],x['path']
 for n,x in m['outputs'].items():assert sha(work/'data'/n)==x['sha256'],n
 if models:
  inventory=[]
  for model,p in MODELS.items():
   files=sorted(x for x in p.rglob('*') if x.is_file() and x.suffix in ('.json','.jinja','.safetensors','.py','.txt','.model'))
   for f in files:
    inventory.append({'model':model,'path':str(f),'size':f.stat().st_size,'sha256':sha(f)})
   print('model hashes complete:',model,flush=True)
  save(work/'model_hashes.json',inventory)
 images=rows(data/'image_hashes.jsonl');assert len(images)==3000
 for x in images:
  assert x['source_sha256']==x['repaired_sha256']==sha(data/x['path']),x['path']
 labels=rows(data/'audit/eval_labels.jsonl');ids={x['id'] for x in labels};assert len(ids)==len(labels)==2000
 for d in ('T2T','T2V','V2T','V2V'):
  assert {x['id'] for x in rows(data/d/'eval.jsonl')}==ids
 assert len(list((data/'images').rglob('*.png')))==3000
 save(work/'source_validation.json',{'status':'pass','images_verified':3000,'labels':2000,'eval_rows_per_direction':2000,'checked_at':now(),'original_images_unchanged':True,'same_rule_learning_and_application':True,'held_out_rule_split':False})

def smoke(model,obj,mod,gpu,work,data):
 folder=work/'smoke'/f'{model}_{obj}_{mod}';folder.mkdir(parents=True,exist_ok=True)
 with (folder/'.smoke.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  return _smoke(model,obj,mod,gpu,work,data)

def _smoke(model,obj,mod,gpu,work,data):
 folder=work/'smoke'/f'{model}_{obj}_{mod}';folder.mkdir(parents=True,exist_ok=True)
 if read(folder/'validated.json').get('status')=='pass':return
 for state_name in ('state.json','reload_state.json'):
  previous=read(folder/state_name)
  if previous.get('status')=='running':
   while alive(previous.get('pid',-1)):time.sleep(5)
 source=work/'data'/f'{obj}_{mod}.jsonl';rr=rows(source)
 # Fixed first material (includes v7's learned recording codebook).
 inp=folder/'train.jsonl';inp.write_text(json.dumps(rr[0],ensure_ascii=False)+'\n')
 out=folder/'checkpoint';cp=final_checkpoint(out,1)
 if not cp:
  capacity(gpu,model,mod,folder)
  code=process(train_command(model,obj,inp,out,epochs=1,smoke=True),environment(gpu,data),folder,'train','state.json')
  if code:raise RuntimeError(f'smoke training failed: {folder}')
  cp=final_checkpoint(out,1)
  if not cp:raise RuntimeError(f'smoke checkpoint incomplete: {folder}')
 logs=rows(out/'logging.jsonl');losses=[x['loss'] for x in logs if 'loss' in x]
 assert losses and all(math.isfinite(float(x)) and float(x)>0 for x in losses),folder
 trainer=read(cp/'trainer_state.json');assert trainer['global_step']==1
 from safetensors import safe_open
 with safe_open(str(cp/'adapter_model.safetensors'),framework='pt',device='cpu') as f:
  assert not any(any(x in k for x in ('.visual.','.vision_tower.','.vision_model.','.merger.','.mm_projector.')) for k in f.keys()),'Frozen vision/aligner has LoRA weights'
  assert any('lora_B' in k and f.get_tensor(k).count_nonzero()>0 for k in f.keys()),'LoRA update absent'
 # Reload the actual saved adapter and test both question modalities, regardless
 # of learning modality; only load a smoke adapter into this preflight run.
 import run_v7_evaluation as ev
 probe=[]
 for d in ('T2T','T2V'):
  r=ev.prepare_row(rows(data/d/'eval.jsonl')[0],data,d)
  # Paired directions share dataset IDs. Give reload probes unique IDs;
  # inference alignment still verifies their exact prompts and images.
  r['id']=d+':'+r['id'];probe.append(r)
 ev.save_rows(folder/'reload.input.jsonl',[{k:r[k] for k in ('messages','images') if k in r} for r in probe])
 result=folder/'reload.raw.jsonl'
 if not result.exists():
  cmd=[str(ROOT/'.venv/bin/swift'),'infer','--model',str(MODELS[model]),'--adapters',str(cp),'--load_args','false','--use_chat_template','true','--infer_backend','transformers','--torch_dtype','bfloat16','--val_dataset',str(folder/'reload.input.jsonl'),'--val_dataset_shuffle','false','--dataset_shuffle','false','--remove_unused_columns','true','--strict','true','--max_batch_size','1','--write_batch_size','1','--max_new_tokens','32','--temperature','0','--seed','42','--enable_thinking','false','--stream','false','--result_path',str(result)]
  if model=='glm46':cmd+=['--template','glm4_5v']
  code=process(cmd,environment(gpu,data),folder,'reload','reload_state.json')
  if code:raise RuntimeError(f'smoke checkpoint reload failed: {folder}')
 predictions=ev.align(ev.read_rows(result)[0],probe,result);assert len(predictions)==2
 save(folder/'validated.json',{'status':'pass','model':model,'objective':obj,'modality':mod,'losses':losses,'lora_B_updated':True,'checkpoint':str(cp),'reload_text_and_image':True,'checked_at':now()})

def run_job(model,obj,mod,gpu,work,data,epochs):
 folder=work/'jobs'/f'{model}_{obj}_{mod}';folder.mkdir(parents=True,exist_ok=True);out=folder/'checkpoint';old=read(folder/'state.json')
 # Adopt jobs launched earlier this turn. Their training process is left alive.
 if old.get('status')=='running' and alive(old.get('pid',-1)):
  old_env=Path(f"/proc/{old['pid']}/environ").read_bytes().split(b'\0')
  gpu=next((x.split(b'=',1)[1].decode() for x in old_env if x.startswith(b'CUDA_VISIBLE_DEVICES=')),gpu)
  print('adopting existing training',model,obj,mod,old['pid'],flush=True)
  with reserve(work,gpu,str(folder.resolve())+':train',TRAIN_PEAKS[model][mod],adopted_pid=old['pid']):
   while alive(old['pid']):time.sleep(10)
 cp=final_checkpoint(out,epochs)
 smoke(model,obj,mod,gpu,work,data)
 if not cp:
  candidates=[p for p in out.glob('checkpoint-*') if all((p/n).is_file() for n in ('trainer_state.json','adapter_model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth'))]
  resume=max(candidates,key=lambda p:read(p/'trainer_state.json')['global_step']) if candidates else None
  if out.exists() and not resume:
   # Never overwrite an interrupted run that has no recoverable checkpoint.
   out.rename(folder/f'interrupted_no_checkpoint_{int(time.time())}')
  cmd=train_command(model,obj,work/'data'/f'{obj}_{mod}.jsonl',out,epochs,resume=resume)
  code=process(cmd,environment(gpu,data),folder,'train','state.json')
  if code:raise RuntimeError(f'training failed ({code}): {folder}')
  cp=final_checkpoint(out,epochs)
  if not cp:raise RuntimeError(f'no verified epoch-{epochs} checkpoint: {folder}')
 save(folder/'checkpoint_manifest.json',{'path':str(cp),'epoch':read(cp/'trainer_state.json')['epoch'],'global_step':read(cp/'trainer_state.json')['global_step'],'sha256':{n:sha(cp/n) for n in ('adapter_model.safetensors','adapter_config.json','trainer_state.json')},'selection':'final epoch; no evaluation-based selection','smoke_validation':str(work/'smoke'/folder.name/'validated.json')})
 save(folder/'state.json',{'status':'complete','model':model,'objective':obj,'modality':mod,'gpu':gpu,'epochs':epochs,'checkpoint':str(cp),'validated_at':now()})
 print('complete',model,obj,mod,flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=ROOT/'rules_repaired_single_image_v7');ap.add_argument('--work',type=Path,default=ROOT/'runs/v7_full_experiment');ap.add_argument('--epochs',type=int,default=5);ap.add_argument('--gpu0',default='0');ap.add_argument('--gpu1',default='1');ap.add_argument('--jobs-per-gpu',type=int,choices=[1,2,3],default=3);ap.add_argument('--prepare-only',action='store_true');a=ap.parse_args();work=a.work.resolve();data=a.root.resolve();work.mkdir(parents=True,exist_ok=True)
 with (work/'.training.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  verify_sources(work,data,models=not (work/'model_hashes.json').exists())
  active_protocols=read(work/'evaluations/protocols.json').get('protocols',['strict'])
  save(work/'manifest.json',{'dataset':str(data),'models':{k:str(v) for k,v in MODELS.items()},'objectives':['sft','pt'],'tuner':'lora','rank':8,'alpha':32,'dropout':0.05,'epochs':a.epochs,'seed':42,'learning_rate':1e-4,'batch_size':1,'gradient_accumulation':16,'max_length':8192,'max_pixels':1003520,'freeze_vit':True,'freeze_aligner':True,'training_jobs':12,'active_evaluation_protocols':active_protocols,'deferred_evaluation_protocols':[p for p in ('strict','cot') if p not in active_protocols],'evaluation_cells':48*len(active_protocols),'evaluation_requests':96000*len(active_protocols),'original_full_evaluation_cells':96,'original_full_evaluation_requests':192000,'directions':DIRECTIONS,'protocols':{'strict':{'max_new_tokens':32,'thinking':False},'cot':{'max_new_tokens':1536,'thinking_by_model':{'qwen3':False,'qwen25':False,'glm46':True},'reasoning_by_model':{'qwen3':'prompted','qwen25':'prompted','glm46':'native'}}},'pt_loss':'all non-image material tokens, no chat template','sft_loss':'assistant response only; user/image context masked','created_at':now(),'historical_build_manifest':'contains inherited old run references; actual sources hashed independently'})
  for model in MODELS:
   for obj in ('sft','pt'):
    for mod in ('text','vision'):
     p=work/'jobs'/f'{model}_{obj}_{mod}'/'state.json'
     if not p.exists():save(p,{'status':'queued','model':model,'objective':obj,'modality':mod,'epochs':a.epochs,'queued_at':now()})
  if a.prepare_only:return
  save(work/'training_status.json',{'status':'running','pid':os.getpid(),'jobs_per_gpu':a.jobs_per_gpu,'updated_at':now()})
  queues={mod:Queue() for mod in ('text','vision')}
  for mod in queues:
   for model in MODELS:
    for obj in ('sft','pt'):
     if read(work/'jobs'/f'{model}_{obj}_{mod}'/'state.json').get('status')=='complete':continue
     queues[mod].put((model,obj))
  def worker(gpu,mod):
   while True:
    job_mod=mod
    try:model,obj=queues[mod].get_nowait()
    except Empty:
     job_mod='vision' if mod=='text' else 'text'
     try:model,obj=queues[job_mod].get_nowait()
     except Empty:return
    try:
     for attempt in range(1,4):
      try:
       run_job(model,obj,job_mod,gpu,work,data,a.epochs)
       break
      except Exception as e:
       save(work/'jobs'/f'{model}_{obj}_{job_mod}'/'state.json',{'status':'failed','attempt':attempt,'error':str(e),'failed_at':now()});print('FAILED',model,obj,job_mod,str(e),flush=True)
       if attempt<3:time.sleep(30)
    finally:queues[job_mod].task_done()
  # Up to three workers per GPU; shared peak-memory reservations with
  # evaluation determine actual admission. Idle workers can steal work.
  with ThreadPoolExecutor(max_workers=2*a.jobs_per_gpu) as pool:
   futures=[pool.submit(worker,gpu,mod) for gpu,mod in ((a.gpu0,'text'),(a.gpu1,'vision')) for _ in range(a.jobs_per_gpu)]
   for f in futures:f.result()
  verify_sources(work,data)
  ss=[read(p) for p in (work/'jobs').glob('*/state.json')]
  save(work/'training_status.json',{'status':'complete' if len(ss)==12 and all(x.get('status')=='complete' for x in ss) else 'incomplete','jobs':ss,'updated_at':now()})
if __name__=='__main__':main()
