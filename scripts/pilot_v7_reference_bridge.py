#!/usr/bin/env python3
"""Matched TEXT pilots: identical target study + dev MCQs with/without references."""
import argparse
import copy
import dataclasses
import fcntl
import json
import math
import os
import random
import subprocess
from pathlib import Path

import run_v7_evaluation as ev
from pilot_v7_task_then_transfer import DATA, LEDGER, card, rotate, validate_sources
from diagnose_v7_knowledge_access import PILOT
from run_v7_full_experiment import ROOT, MODELS, environment, rows, read, save, sha, train_command, final_checkpoint
from v7_gpu_resources import reserve

ARMS=('reference_dev','memory_dev','text_alias','warmup')


def cp_for(out,arm):
    if arm in ('text_alias','warmup'):return Path(read(PILOT/arm/'checkpoint_manifest.json')['path'])
    cp=final_checkpoint(out/arm/'checkpoint',1);assert cp;return cp


def prepare(out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():
        m=read(out/'manifest.json');assert m['script_sha256']==sha(Path(__file__))
        for p,h in m['files'].items():assert sha(out/p)==h
        for p,h in m['sources'].items():assert sha(Path(p))==h
        return
    validate_sources(PILOT)
    dev={k['id']:k for k in rows(PILOT/'development_knowledge.jsonl')};target={k['id']:k for k in rows(PILOT/'target_knowledge.jsonl')};assert not set(dev)&set(target)
    labels={r['id']:r for r in rows(DATA/'audit/eval_labels.jsonl')}
    study=[copy.deepcopy(r) for r in rows(PILOT/'text_alias/train.jsonl') if int(r['id'].split(':')[-1])<48];assert len(study)==384
    rehearsal=[]
    for row in rows(DATA/'T2T/eval.jsonl'):
        if row['qid'] not in dev:continue
        p=ev.prepare_row(row,DATA,'T2T')
        for shift in range(4):
            text,answer,_=rotate(p['messages'][-1]['content'],labels[row['id']]['answer'],shift)
            z={'id':f"dev:{row['id']}:{shift}",'qid':row['qid'],'channel':'dev_mcq','messages':[{'role':'system','content':ev.SYSTEM},{'role':'user','content':text},{'role':'assistant','content':answer}]}
            rehearsal.append(z)
    assert len(rehearsal)==128
    for arm in ('reference_dev','memory_dev'):
        rr=copy.deepcopy(study)+copy.deepcopy(rehearsal)
        if arm=='reference_dev':
            for r in rr:
                if r['channel']=='dev_mcq':r['messages'][-2]['content']='Learning reference:\n'+card(dev[r['qid']])+'\n\n'+r['messages'][-2]['content']
        random.Random(42).shuffle(rr);ev.save_rows(out/f'{arm}.jsonl',rr)
    a,b=rows(out/'reference_dev.jsonl'),rows(out/'memory_dev.jsonl')
    for x,y in zip(a,b):
        assert x['id']==y['id'] and x['messages'][-1]==y['messages'][-1]
        if x['qid'] in target:assert x==y
        else:assert x['messages'][-2]['content'].endswith(y['messages'][-2]['content'])
    probes=[]
    for p in rows(PILOT/'probes.jsonl'):
        if p['condition'].startswith('direct_'):
            for shift in range(4):
                z=copy.deepcopy(p);text,answer,_=rotate(z['messages'][-1]['content'],z['gold'],shift)
                z['messages'][-1]['content']=text;z.update(id=f"rot{shift}:"+p['id'],source_id=p['id'],rotation=shift,gold=answer,fmt='mcq');probes.append(z)
        elif p['condition']=='card_training_prompt':
            z=copy.deepcopy(p);z['fmt']='card';z['source_id']=p['id'];probes.append(z)
    canonical={p['qid']:p['gold'] for p in probes if p['fmt']=='card'}
    for p in rows(PILOT/'text_alias/task_wording_recall/recall_probes.jsonl'):
        z=copy.deepcopy(p);z.update(id='task_heading:'+p['qid'],condition='card_task_heading',fmt='card',task='card',gold=canonical[p['qid']]);probes.append(z)
    assert len(probes)==144 and len({p['id'] for p in probes})==144
    ev.save_rows(out/'probes.jsonl',probes)
    sources=[PILOT/'manifest.json',PILOT/'text_alias/train.jsonl',PILOT/'development_knowledge.jsonl',PILOT/'target_knowledge.jsonl',PILOT/'probes.jsonl',PILOT/'text_alias/task_wording_recall/recall_probes.jsonl',DATA/'T2T/eval.jsonl',DATA/'audit/eval_labels.jsonl']
    save(out/'manifest.json',{'script_sha256':sha(Path(__file__)),'files':{p.name:sha(p) for p in out.glob('*.jsonl')},'sources':{str(p):sha(p) for p in sources},'parent':ev.checkpoint(cp_for(out,'warmup')),'arms':['reference_dev','memory_dev'],'comparisons':['text_alias','warmup'],'study_rules':list(target),'dev_rules':list(dev),'training_rows':512,'target_material_rows':384,'target_exposures_per_rule':48,'dev_rehearsal_rows':128,'epochs':1,'steps':128,'batch':4,'accumulation':1,'lr':1e-4,'rank':8,'alpha':32,'dropout':.05,'freeze_vision_aligner':True,'seed':42,'contrast':'Only development-rule material in user prompt present vs absent. Same parent, row order, labels, target material, steps and optimizer. No target observations or answers in training. Development observations are deliberate training examples.','diagnosis_informed':True,'not_pre_registered_independent_test':True,'evaluation':'Original32 target MCQs x4 cyclic option shifts; four versions share underlying observations and must not be counted as128 independent cases. Strict letters, no external target materials, no thinking. Also16 factual capacity probes.','pure_study_comparison_limit':'Original text_alias has64 target exposures/rule vs48 in both new arms; only comparison BETWEEN new arms isolates reference presence.','created_at':ev.now()})


def train(out,arm,gpu):
    if arm in ('text_alias','warmup'):return cp_for(out,arm)
    folder=out/arm;folder.mkdir(exist_ok=True);root=folder/'checkpoint';parent=cp_for(out,'warmup');cp=final_checkpoint(root,1)
    if cp is None:
        recover=[p for p in root.glob('checkpoint-*') if all((p/n).exists() for n in ('optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','adapter_model.safetensors'))]
        resume=max(recover,key=lambda p:read(p/'trainer_state.json')['global_step']) if recover else None
        cmd=train_command('qwen3','sft',out/f'{arm}.jsonl',root,1,resume=resume)
        for f,v in (('--per_device_train_batch_size','4'),('--gradient_accumulation_steps','1'),('--save_strategy','steps'),('--save_total_limit','3')):cmd[cmd.index(f)+1]=v
        cmd+=['--save_steps','32','--enable_channel_loss','true']
        if not resume:cmd+=['--adapters',str(parent)]
        with reserve(LEDGER,gpu,str(folder)+':train',25000) as reservation:
            with (folder/'train.log').open('a') as log:
                proc=subprocess.Popen(cmd,cwd=ROOT,env=environment(gpu,DATA),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                reservation.bind(proc.pid);save(folder/'train_state.json',{'status':'running','pid':proc.pid,'command':cmd})
                code=proc.wait()
            assert code==0,f'{arm} training failed:{code}'
        cp=final_checkpoint(root,1);assert cp
    st=read(cp/'trainer_state.json');assert st['global_step']==128 and abs(st['epoch']-1)<1e-6
    assert str(parent) in read(cp/'args.json')['adapters']
    losses=[r['loss'] for r in rows(root/'logging.jsonl') if 'loss' in r];assert losses and all(math.isfinite(x) for x in losses)
    from safetensors import safe_open
    with safe_open(str(cp/'adapter_model.safetensors'),framework='pt',device='cpu') as f:
        keys=list(f.keys());bs=[k for k in keys if 'lora_B' in k]
        assert len(bs)==252 and all(bool(f.get_tensor(k).count_nonzero()) for k in bs)
        assert not any('.visual.' in k or '.merger.' in k for k in keys)
    save(folder/'checkpoint_manifest.json',{**ev.checkpoint(cp),'parent':ev.checkpoint(parent),'epoch':1,'global_step':128,'final_loss':losses[-1],'optimizer_reset':True,'selection':'fixed final epoch1, no score selection','nonzero_lora_B':len(bs)})
    save(folder/'train_state.json',{'status':'complete','checkpoint':str(cp)})
    return cp


def run(out,arm,gpu):
    folder=out/arm;folder.mkdir(exist_ok=True)
    with ev.run_lock(folder/'.lock'):
        cp=train(out,arm,gpu);probes=rows(out/'probes.jsonl');raw=folder/'predictions.jsonl'
        config={'adapter':ev.checkpoint(cp),'probes_sha256':sha(out/'probes.jsonl'),'max_tokens':{'mcq':16,'card':256},'seed':42,'temperature':0,'thinking':False,'batch':4}
        if (folder/'eval_manifest.json').exists():assert read(folder/'eval_manifest.json')==config
        else:save(folder/'eval_manifest.json',config)
        records,notes=ev.read_rows(raw,allow_partial_tail=True) if raw.exists() else ([],[])
        if notes:ev.save_rows(raw,records)
        seen={r['id'] for r in records}
        def rec(p,reply,reused=None):
            c=reply['choices'][0];text=c['message']['content'] or '';pred=ev.letter(text) if p['fmt']=='mcq' else None
            r={k:p[k] for k in ('id','qid','condition','fmt','gold','task')};r.update(response=text,predicted=pred,correct=pred==p['gold'] if p['fmt']=='mcq' else text.strip()==p['gold'],format_error=p['fmt']=='mcq' and pred is None,finish_reason=c['finish_reason'],input_sha256=ev.digest({'messages':p['messages'],'images':p.get('images',[])}),raw_response=reply)
            if 'rotation' in p:r['rotation']=p['rotation']
            if reused:r['reused_from']=reused
            return r
        if arm in ('text_alias','warmup'):
            prior={r['id']:r for r in rows(PILOT/arm/'predictions.jsonl')};heading={r['qid']:r for r in rows(PILOT/arm/'task_wording_recall/predictions.jsonl') if r['kind']=='recall'}
            for p in probes:
                if p['id'] in seen:continue
                old=None
                if p['condition']=='card_task_heading':old=heading[p['qid']];old_hash=old['input_sha256']
                elif p.get('rotation',0)==0 and p.get('source_id') in prior:old=prior[p['source_id']];old_hash=old['prompt_sha256']
                if old is not None:
                    assert old_hash==ev.digest({'messages':p['messages'],'images':p.get('images',[])})
                    records.append(rec(p,old['raw_response'],str(PILOT/arm)))
            ev.save_rows(raw,records)
        seen={r['id'] for r in records};pending=[p for p in probes if p['id'] not in seen]
        if pending:
            with reserve(LEDGER,gpu,str(folder)+':eval',24000) as reservation:
                reservation.bind(os.getpid());os.environ.update(environment(gpu,DATA))
                import torch
                from swift.infer_engine import TransformersEngine,InferRequest,RequestConfig
                from swift.utils import seed_everything
                seed_everything(42)
                engine=TransformersEngine(str(MODELS['qwen3']),adapters=[str(cp)],torch_dtype=torch.bfloat16,device_map='cuda:0',max_batch_size=4,attn_impl='sdpa');engine.template.enable_thinking=False
                for fmt in ('mcq','card'):
                    group=[p for p in pending if p['fmt']==fmt]
                    for start in range(0,len(group),4):
                        batch=group[start:start+4];replies=engine.infer([InferRequest(messages=copy.deepcopy(p['messages']),images=p.get('images',[])) for p in batch],RequestConfig(max_tokens=16 if fmt=='mcq' else 256,temperature=0,seed=42,return_details=True),use_tqdm=False)
                        assert len(batch)==len(replies)
                        for p,reply in zip(batch,replies):
                            if isinstance(reply,Exception):raise reply
                            r=rec(p,dataclasses.asdict(reply));records.append(r)
                            with raw.open('a') as f:f.write(json.dumps(r,ensure_ascii=False,default=str)+'\n');f.flush();os.fsync(f.fileno())
                        save(folder/'state.json',{'status':'running','pid':os.getpid(),'saved':len(records),'expected':144});print(arm,fmt,start+len(batch),'/',len(group),flush=True)
        assert len(records)==len({r['id'] for r in records})==144
        def metric(rr):return {'n':len(rr),'correct':sum(r['correct'] for r in rr),'format_errors':sum(r['format_error'] for r in rr),'truncated':sum(r['finish_reason']=='length' for r in rr)}
        metrics={c:metric([r for r in records if r['condition']==c]) for c in sorted({p['condition'] for p in probes})}
        rotations={c:{str(i):metric([r for r in records if r['condition']==c and r.get('rotation')==i]) for i in range(4)} for c in ('direct_T','direct_V')}
        consistency={}
        for c in ('direct_T','direct_V'):
            qids={r['id'].split(':',1)[1] for r in records if r['condition']==c}
            # Group by original ID after stripping the explicit rotation prefix.
            groups=[[r for r in records if r['condition']==c and r['id'].split(':',1)[1]==original] for original in qids]
            consistency[c]={'original_questions':len(groups),'all4_rotations_correct':sum(all(r['correct'] for r in g) for g in groups),'all4_rotations_wrong':sum(all(not r['correct'] for r in g) for g in groups)}
        save(folder/'report.json',{'status':'complete','arm':arm,'metrics':metrics,'rotations':rotations,'consistency':consistency,'new_requests':sum('reused_from' not in r for r in records),'independent_primary_questions':32,'all_rotation_versions':128})
        save(folder/'state.json',{'status':'complete','saved':144})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'runs/v7_reference_bridge_20261015');ap.add_argument('--arm',choices=ARMS,default='memory_dev');ap.add_argument('--gpu',default='0');ap.add_argument('--prepare-only',action='store_true');a=ap.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/'.prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);prepare(out)
    if not a.prepare_only:run(out,a.arm,a.gpu)
