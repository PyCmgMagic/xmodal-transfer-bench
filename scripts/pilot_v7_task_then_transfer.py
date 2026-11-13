#!/usr/bin/env python3
"""Independent-rule task warmup, then matched TEXT/VISION knowledge SFT pilots."""
import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
import random
import re
import subprocess
from collections import Counter
from pathlib import Path

import run_v7_evaluation as ev
from build_v7_experiment_data import name
from run_v7_full_experiment import ROOT, MODELS, environment, final_checkpoint, read, rows, save, sha, train_command
from v7_gpu_resources import reserve

DATA = ROOT / 'rules_repaired_single_image_v7'
LEDGER = ROOT / 'runs/v7_full_experiment'
ARMS = ('base', 'warmup', 'text', 'vision')
CONVENTION = ('Read the supplied reference exactly. For three-category guides, both boundary values '
              'belong to the middle category. For marked cells, count filled/marked cells only; '
              'borders and check marks are not cells. State exact rule facts without inventing examples.')
OP = {'eq':'=', 'ne':'!=', 'ge':'>=', 'gt':'>', 'le':'<=', 'lt':'<'}


def messages(q, a=None, strict=False):
    z = [{'role':'system','content':ev.SYSTEM if strict else CONVENTION}, {'role':'user','content':q}]
    if a is not None:
        z.append({'role':'assistant','content':a})
    return z


def facts(k):
    r = json.loads(k['fact']['quote'])
    if 'conditions' in r:
        out = [(f'condition {i+1}', f"{c['field']} {OP[c['operator']]} {c['value']}") for i,c in enumerate(r['conditions'])]
        mode = f"at least {r['required_count']} of {len(r['conditions'])}" if 'required_count' in r else r['mode']
        return out + [('condition combination', mode)]
    ratio = r['ratio']
    detail = ', '.join(r.get('joint_attributes', []))
    expr = ratio['numerator_field'] + '/' + ratio['denominator_field']
    if detail:
        expr += ' (' + detail + ')'
    return [('lower proportion threshold', str(r['thresholds']['lower'])),
            ('upper proportion threshold', str(r['thresholds']['upper'])),
            ('category labels in below/between/above order', ', '.join(r['category_labels'])),
            ('numerator/denominator definition', expr)]


def card(k):
    r = json.loads(k['fact']['quote'])
    if 'conditions' in r:
        cs = '; '.join(f"{c['field']} {OP[c['operator']]} {c['value']}" for c in r['conditions'])
        mode = f"at least {r['required_count']} of {len(r['conditions'])} conditions" if 'required_count' in r else 'all conditions' if r['mode']=='all' else 'at least one condition'
        return f"{name(k)}: {cs}. Check each candidate separately. KEEP if {mode} hold; otherwise OMIT. Report every candidate's KEEP/OMIT result."
    lo,hi = r['thresholds']['lower'],r['thresholds']['upper']
    a,b,c = r['category_labels']
    return (f"{name(k)}: compute {facts(k)[-1][1]}. Below {lo}: {a}. From {lo} through {hi}, including both boundaries: {b}. Above {hi}: {c}. Report every collection's category.")


def rotate(text, gold, shift):
    pattern = r'(?m)^([A-D])\. (.*)$'
    opts = re.findall(pattern,text)
    assert len(opts)==4
    contents = [v for _,v in opts]
    shifted = contents[shift:]+contents[:shift]
    iterator = iter(zip('ABCD',shifted))
    text = re.sub(pattern,lambda m: '{}. {}'.format(*next(iterator)),text)
    return text, 'ABCD'[("ABCD".index(gold)-shift)%4],contents['ABCD'.index(gold)]


def validate_sources(out):
    m=read(out/'manifest.json')
    for p,h in m['sources'].items():
        assert sha(Path(p))==h,p
    for p,h in m['files'].items():
        assert sha(out/p)==h,p
    images=rows(DATA/'image_hashes.jsonl')
    assert len(images)==3000
    for r in images:
        assert sha(DATA/r['path'])==r['source_sha256']==r['repaired_sha256'],r['path']
    save(out/'source_validation.json',{'status':'pass','all_3000_original_images_unchanged':True,'checked_at':ev.now()})


def prepare(out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():
        assert read(out/'manifest.json')['script_sha256']==sha(Path(__file__))
        validate_sources(out)
        return
    knowledge=[r['content'] for r in rows(DATA/'audit/rule_knowledge.jsonl')]
    target,dev=[],[]
    excluded=[]
    for relation in ('set_selection_rule','three_way_category'):
        selected=[]
        ordered=sorted([k for k in knowledge if k['fact']['relation']==relation],key=lambda k:hashlib.sha256(('task-then-transfer-20261015:42:'+k['id']).encode()).hexdigest())
        for k in ordered:
            r=json.loads(k['fact']['quote'])
            if relation=='three_way_category' and r['measure_topology']=='target_object_share':
                excluded.append({'qid':k['id'],'reason':'Topology eligibility exclusion: original learning image may omit explicit boundary definitions; 0187 manually confirmed underspecified.'})
                continue
            selected.append(k)
            if len(selected)==12:break
        target+=selected[:4];dev+=selected[4:12]
    # Freeze selected IDs before opening target scores; manually reviewed source images.
    assert [k['id'] for k in target]==['scale1k_0175_ai_generated','scale1k_0704_ai_generated','scale1k_0165_ai_generated','scale1k_0964_programmatic','scale1k_0090_programmatic','scale1k_0972_programmatic','scale1k_0762_ai_generated','scale1k_0178_ai_generated']
    assert len(target)==8 and len(dev)==16
    assert not {k['id'] for k in target}&{k['id'] for k in dev}
    eligible=[{'qid':k['id'],'split':'target' if k in target else 'task_development','learning_image':str(DATA/'images'/k['id']/'learning_visual.png'),'review':'Manual visual review of task-critical conditions, combination, boundary values and ordered labels; generic boundary convention supplied identically to both conditions.'} for k in target+dev]
    ev.save_rows(out/'image_eligibility.jsonl',eligible+excluded)
    ev.save_rows(out/'target_knowledge.jsonl',target)
    ev.save_rows(out/'development_knowledge.jsonl',dev)
    # Finish target material-only learning data before target test questions are read.
    forms=['Explain the complete rule named {}.','State the exact stored rule for {}.','Recall all conditions and mappings for {}.','Describe how to apply {}.']
    suffixes=[' Include exact values.',' Include every condition.',' Include the complete decision rule.',' Give its exact mapping.']
    for arm in ('text','vision'):
        tr=[]
        for k in target:
            for i in range(64):
                q=forms[i%4].format(name(k))+suffixes[(i//4)%4]
                if arm=='vision':q='<image>Using only this original learning reference, '+q[0].lower()+q[1:]
                z={'id':f"{k['id']}:{arm}:{i}",'qid':k['id'],'channel':'target_material','messages':messages(q,card(k))}
                if arm=='vision':z['images']=[str(DATA/'images'/k['id']/'learning_visual.png')]
                tr.append(z)
        random.Random(42).shuffle(tr);ev.save_rows(out/f'{arm}.jsonl',tr)
    # Development questions are deliberately training data. None is a target test.
    labels={r['id']:r for r in rows(DATA/'audit/eval_labels.jsonl')}
    evals={d:rows(DATA/d/'eval.jsonl') for d in ('T2T','T2V','V2T','V2V')}
    assert all({r['id'] for r in rr}==set(labels) for rr in evals.values())
    assert evals['T2T']==evals['V2T'] and evals['T2V']==evals['V2V']
    tr=[];devprobes=[]
    devmap={k['id']:k for k in dev}
    for direction in ('T2T','T2V'):
        for row in evals[direction]:
            if row['qid'] not in devmap:continue
            k=devmap[row['qid']];p=ev.prepare_row(row,DATA,direction);gold=labels[row['id']]['answer']
            # Two material modalities are seen by the shared warmup, equally often.
            prefix='Learning reference:\n'+card(k)+'\n\n' if direction=='T2T' else 'Learning reference (first image):\n<image>\n'+CONVENTION+'\n\nTest question (second image):\n'
            if direction=='T2V':p['images']=[str(DATA/'images'/k['id']/'learning_visual.png')]+p['images']
            for shift in range(4):
                body,answer,result=rotate(p['messages'][-1]['content'],gold,shift)
                for fmt in ('mcq','result'):
                    q=prefix+body
                    if fmt=='result':
                        q=q.split('\n\nOptions:')[0]
                        assert 'Result: ' in result
                        if labels[row['id']]['question_type']=='diagnosis':
                            q+='\nReturn the complete report outcome (Accept report or Replace report), all requested rule fields, and the Result: assignment. Do not output an option letter.'
                            answer=result
                            assert re.match(r'(Accept|Replace) report\.',answer)
                        else:
                            q+='\nReturn only the complete assignment in the form A=...; B=...; C=...; D=....'
                            answer=result.split('Result: ',1)[1].rstrip('.')
                            assert re.fullmatch(r'A=.+; B=.+; C=.+; D=.+',answer)
                    z={'id':f"{row['id']}:{direction}:{shift}:{fmt}",'qid':row['qid'],'channel':'task_'+fmt,'messages':messages(q,answer,fmt=='mcq')}
                    if 'images' in p:z['images']=p['images']
                    tr.append(z)
            # Capacity control uses seen dev questions and provided references.
            if direction=='T2T':
                z=copy.deepcopy(p);z['id']='dev:'+direction+':'+row['id'];z['messages'][-1]['content']=prefix+z['messages'][-1]['content']
                z.update(condition='seen_dev_text_reference',fmt='mcq',gold=gold,task=labels[row['id']]['question_type'],relation=k['fact']['relation'])
                devprobes.append(z)
    assert len(tr)==512
    random.Random(42).shuffle(tr);ev.save_rows(out/'warmup.jsonl',tr)
    targetmap={k['id']:k for k in target};probes=[]
    for direction in ('T2T','T2V'):
        for row in evals[direction]:
            if row['qid'] not in targetmap:continue
            k=targetmap[row['qid']];p=ev.prepare_row(row,DATA,direction)
            base={'qid':row['qid'],'fmt':'mcq','gold':labels[row['id']]['answer'],'task':labels[row['id']]['question_type'],'relation':k['fact']['relation'],'evaluation_group':labels[row['id']]['evaluation_group']}
            for protocol in ('direct','oracle_text','oracle_image'):
                z=copy.deepcopy(p);z.update(base);z['id']=protocol+':'+direction+':'+row['id'];z['condition']=protocol+'_'+direction[-1]
                if protocol=='oracle_text':z['messages'][-1]['content']='Correct learning reference:\n'+card(k)+'\n\n'+z['messages'][-1]['content']
                if protocol=='oracle_image':
                    z['messages'][-1]['content']='Original learning reference (first image):\n<image>\n'+CONVENTION+'\n\nTest question:\n'+z['messages'][-1]['content']
                    z['images']=[str(DATA/'images'/k['id']/'learning_visual.png')]+z.get('images',[])
                probes.append(z)
    for k in target:
        for i,(focus,gold) in enumerate(facts(k)):
            for image in (False,True):
                q=f"Regarding {name(k)}, retrieve the {focus}. Return only that fact."
                z={'id':f"fact:{k['id']}:{i}:{image}",'qid':k['id'],'condition':'fact_with_image' if image else 'fact_no_image','fmt':'fact','gold':gold,'task':'fact','relation':k['fact']['relation'],'messages':messages(('<image>Read this reference. '+q) if image else q)}
                if image:z['images']=[str(DATA/'images'/k['id']/'learning_visual.png')]
                probes.append(z)
        for protocol in ('training_prompt','new_prompt','image_prompt'):
            q=forms[0].format(name(k))+suffixes[0] if protocol=='training_prompt' else f"Give the exact decision rule stored under {name(k)}, including every required value."
            z={'id':f"card:{k['id']}:{protocol}",'qid':k['id'],'condition':'card_'+protocol,'fmt':'card','gold':card(k),'task':'card','relation':k['fact']['relation'],'messages':messages(('<image>Using only this original learning reference, '+q[0].lower()+q[1:]) if protocol=='image_prompt' else q)}
            if protocol=='image_prompt':z['images']=[str(DATA/'images'/k['id']/'learning_visual.png')]
            probes.append(z)
    probes+=devprobes;assert len({p['id'] for p in probes})==len(probes)
    ev.save_rows(out/'probes.jsonl',probes)
    for arm in ('text','vision','warmup'):
        rr=rows(out/f'{arm}.jsonl');assert len(rr)==512
        assert all(r['messages'][-1]['role']=='assistant' for r in rr)
        assert all(sum(m['content'].count('<image>') for m in r['messages'])==len(r.get('images',[])) for r in rr)
        assert {r['qid'] for r in rr}==set(devmap if arm=='warmup' else targetmap)
    sources=[DATA/'audit/rule_knowledge.jsonl',DATA/'audit/eval_labels.jsonl',DATA/'image_hashes.jsonl']+[DATA/d/'eval.jsonl' for d in evals]
    save(out/'manifest.json',{'model':str(MODELS['qwen3']),'target_rules':list(targetmap),'development_rules':list(devmap),'selection':'SHA256(task-then-transfer-20261015:42:ID); first4 target, next8 development per family, after structural image eligibility exclusion; no score filtering','epochs':1,'rows_per_stage':512,'batch_size':4,'accumulation':1,'steps_per_stage':128,'learning_rate':1e-4,'rank':8,'alpha':32,'dropout':.05,'seed':42,'max_length':8192,'freeze_vision_aligner':True,'objective':'SFT only; historical PT unchanged','branches':'text/vision both initialize trainable LoRA from SAME final warmup adapter, fresh optimizer/scheduler','primary_protocol':'no external learning reference, strict A/B/C/D, no thinking','target_training':'16 prompt forms repeated4 times/rule; compact image-grounded material; no target observations/answers','warmup_training':'16 disjoint dev rules; 2 existing questions x2 modalities x4 option rotations x2 response styles; semantic rows repeat4 times; development questions are NOT held out','direction_equivalence':'T2T=V2T inputs; T2V=V2V inputs. Compute once per arm/input, map TEXT and VISION source directions separately; do not count repeated direction labels as independent evidence.','image_convention':CONVENTION,'image_review':'All24 selected original learning images visually reviewed before training; rule-specific values visible or countable. Ambiguous original target-object-share guide excluded. Image knowledge is semantically normalized, not pixel-identical to text.','limits':['8-rule development pilot; not full-dataset evidence','Matched rows/steps; input and supervised token counts not equal','Source knowledge must improve before interpreting a transfer ratio','Oracle results have external references and are not primary transfer results','Card exact match is a capacity diagnostic, not semantic proof; auxiliary fact queries scored separately'],'probes':dict(Counter(p['condition'] for p in probes)),'sources':{str(p):sha(p) for p in sources},'files':{p.name:sha(p) for p in out.glob('*.jsonl')},'script_sha256':sha(Path(__file__)),'created_at':ev.now()})
    validate_sources(out)


def train(out,arm,gpu):
    if arm=='base':return None
    folder=out/arm;folder.mkdir(exist_ok=True);checkpoint_dir=folder/'checkpoint'
    parent=None
    if arm in ('text','vision'):
        parent=final_checkpoint(out/'warmup/checkpoint',1)
        assert parent,'Warmup must complete before target branches.'
        assert ev.checkpoint(parent)['sha256']==read(out/'warmup/checkpoint_manifest.json')['sha256']
    cp=final_checkpoint(checkpoint_dir,1)
    if cp is None:
        candidates=[p for p in checkpoint_dir.glob('checkpoint-*') if all((p/n).exists() for n in ('optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json','adapter_model.safetensors'))]
        resume=max(candidates,key=lambda p:read(p/'trainer_state.json')['global_step']) if candidates else None
        if checkpoint_dir.exists() and not resume:checkpoint_dir.rename(folder/f'interrupted_{int(__import__("time").time())}')
        cmd=train_command('qwen3','sft',out/f'{arm}.jsonl',checkpoint_dir,1,resume=resume)
        for flag,value in (('--per_device_train_batch_size','4'),('--gradient_accumulation_steps','1'),('--save_total_limit','3')):cmd[cmd.index(flag)+1]=value
        cmd+=['--enable_channel_loss','true','--save_strategy','steps','--save_steps','32']
        if parent and not resume:cmd+=['--adapters',str(parent)]
        with reserve(LEDGER,gpu,str(folder)+':train',35000 if arm=='warmup' else 27000) as reservation:
            with (folder/'train.log').open('a') as log:
                proc=subprocess.Popen(cmd,cwd=ROOT,env=environment(gpu,DATA),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                reservation.bind(proc.pid);save(folder/'train_state.json',{'status':'running','pid':proc.pid,'command':cmd,'parent':ev.checkpoint(parent) if parent else None,'resume':str(resume) if resume else None,'started_at':ev.now()})
                code=proc.wait()
            assert code==0,f'{arm} training failed ({code}); see {folder}/train.log'
        cp=final_checkpoint(checkpoint_dir,1);assert cp
    state=read(cp/'trainer_state.json');assert state['global_step']==128 and abs(state['epoch']-1)<1e-6
    losses=[x['loss'] for x in rows(checkpoint_dir/'logging.jsonl') if 'loss' in x];assert losses and all(math.isfinite(x) for x in losses)
    from safetensors import safe_open
    with safe_open(str(cp/'adapter_model.safetensors'),framework='pt',device='cpu') as f:
        keys=list(f.keys());bs=[k for k in keys if 'lora_B' in k]
        assert len(bs)==252 and all(bool(f.get_tensor(k).count_nonzero()) for k in bs)
        assert not any('.visual.' in k or '.merger.' in k for k in keys)
    args=read(cp/'args.json')
    if parent:assert str(parent) in args.get('adapters',[]),args.get('adapters')
    save(folder/'checkpoint_manifest.json',{**ev.checkpoint(cp),'parent':ev.checkpoint(parent) if parent else None,'epoch':state['epoch'],'global_step':state['global_step'],'final_loss':losses[-1],'nonzero_lora_B':len(bs),'optimizer_reset_at_stage_start':bool(parent),'selection':'fixed final epoch1; no evaluation selection'})
    save(folder/'train_state.json',{'status':'complete','checkpoint':str(cp),'finished_at':ev.now()})
    return cp


def evaluate(out,arm,gpu,cp):
    folder=out/arm;folder.mkdir(exist_ok=True);raw=folder/'predictions.jsonl';probes=rows(out/'probes.jsonl')
    config={'model':str(MODELS['qwen3']),'adapter':ev.checkpoint(cp) if cp else None,'probes_sha256':sha(out/'probes.jsonl'),'max_tokens':{'mcq':16,'fact':80,'card':256},'temperature':0,'seed':42,'thinking':False}
    if (folder/'eval_manifest.json').exists():assert read(folder/'eval_manifest.json')==config
    else:save(folder/'eval_manifest.json',config)
    records,notes=ev.read_rows(raw,allow_partial_tail=True) if raw.exists() else ([],[])
    if notes:ev.save_rows(raw,records)
    byid={p['id']:p for p in probes};assert len(records)==len({r['id'] for r in records})
    for r in records:
        p=byid[r['id']];assert r['prompt_sha256']==ev.digest({'messages':p['messages'],'images':p.get('images',[])})
    seen={r['id'] for r in records};pending=[p for p in probes if p['id'] not in seen]
    if pending:
        with reserve(LEDGER,gpu,str(folder)+':eval',25000) as reservation:
            reservation.bind(os.getpid());os.environ.update(environment(gpu,DATA))
            import torch
            from swift.infer_engine import TransformersEngine,InferRequest,RequestConfig
            from swift.utils import seed_everything
            seed_everything(42)
            engine=TransformersEngine(str(MODELS['qwen3']),adapters=[str(cp)] if cp else None,torch_dtype=torch.bfloat16,device_map='cuda:0',max_batch_size=8,attn_impl='sdpa')
            engine.template.enable_thinking=False
            for fmt in ('mcq','fact','card'):
                group=[p for p in pending if p['fmt']==fmt]
                for start in range(0,len(group),8):
                    batch=group[start:start+8]
                    outputs=engine.infer([InferRequest(messages=copy.deepcopy(p['messages']),images=p.get('images',[])) for p in batch],RequestConfig(max_tokens=config['max_tokens'][fmt],temperature=0,seed=42,return_details=True),use_tqdm=False)
                    assert len(outputs)==len(batch)
                    for p,reply in zip(batch,outputs):
                        if isinstance(reply,Exception):raise reply
                        c=reply.choices[0];text=c.message.content or '';pred=ev.letter(text) if fmt=='mcq' else None
                        clean=lambda x:re.sub(r'\s+',' ',x.strip().rstrip('.')).casefold()
                        correct=pred==p['gold'] if fmt=='mcq' else clean(text)==clean(p['gold']) if fmt=='fact' else text.strip()==p['gold']
                        r={k:p[k] for k in ('id','qid','condition','fmt','gold','task','relation')};r.update(response=text,correct=correct,predicted=pred,format_error=fmt=='mcq' and pred is None,finish_reason=c.finish_reason,prompt_sha256=ev.digest({'messages':p['messages'],'images':p.get('images',[])}),raw_response=dataclasses.asdict(reply),saved_at=ev.now())
                        if 'evaluation_group' in p:r['evaluation_group']=p['evaluation_group']
                        with raw.open('a') as f:f.write(json.dumps(r,ensure_ascii=False,default=str)+'\n');f.flush();os.fsync(f.fileno())
                        records.append(r)
                    save(folder/'eval_state.json',{'status':'running','pid':os.getpid(),'saved':len(records),'expected':len(probes),'updated_at':ev.now()});print(arm,fmt,start+len(batch),'/',len(group),flush=True)
    assert len(records)==len(probes)
    def metric(g):
        valid=[r for r in g if not r['format_error']]
        return {'n':len(g),'correct':sum(r['correct'] for r in g),'accuracy':sum(r['correct'] for r in g)/len(g) if g else None,'valid_accuracy':sum(r['correct'] for r in valid)/len(valid) if valid else None,'format_errors':sum(r['format_error'] for r in g),'empty':sum(not r['response'].strip() for r in g),'truncated':sum(r['finish_reason']=='length' for r in g)}
    metrics={}
    for condition in sorted({p['condition'] for p in probes}):
        g=[r for r in records if r['condition']==condition]
        metrics[condition]={**metric(g),'by_task':{t:metric([r for r in g if r['task']==t]) for t in sorted({r['task'] for r in g})},'by_relation':{t:metric([r for r in g if r['relation']==t]) for t in sorted({r['relation'] for r in g})},'by_group':{t:metric([r for r in g if r.get('evaluation_group')==t]) for t in sorted({r['evaluation_group'] for r in g if 'evaluation_group' in r})}}
    save(folder/'report.json',{'status':'complete','arm':arm,'metrics':metrics,'per_rule':{qid:{c:metric([r for r in records if r['qid']==qid and r['condition']==c]) for c in metrics} for qid in read(out/'manifest.json')['target_rules']},'finished_at':ev.now()})
    save(folder/'eval_state.json',{'status':'complete','saved':len(records),'finished_at':ev.now()})


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=ROOT/'runs/v7_task_then_transfer_20261015/pilot_v2');ap.add_argument('--arm',choices=ARMS,default='warmup');ap.add_argument('--gpu',default='0');ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--train-only',action='store_true');a=ap.parse_args();out=a.out.resolve()
    with ev.run_lock(out.parent/('.'+out.name+'.prepare.lock')):prepare(out)
    if not a.prepare_only:
        (out/a.arm).mkdir(exist_ok=True)
        with ev.run_lock(out/a.arm/'.lock'):
            cp=train(out,a.arm,a.gpu)
            if not a.train_only:evaluate(out,a.arm,a.gpu,cp)
