#!/usr/bin/env python3
"""Independent 12-rule TEXT learnability pilot; no frozen evaluation targets in training."""
import argparse
import copy
import hashlib
import itertools
import json
import operator
import os
import random
import subprocess
from pathlib import Path

import run_v7_evaluation as ev
from run_v7_full_experiment import ROOT,MODELS,environment,rows,save,read,train_command,final_checkpoint,sha
from v7_gpu_resources import reserve

OPS={'eq':operator.eq,'ne':operator.ne,'ge':operator.ge,'gt':operator.gt,'le':operator.le,'lt':operator.lt}


def build(data,work,out):
    all_rules=[x['content'] for x in rows(data/'audit/rule_knowledge.jsonl')]
    selected=sorted((k for k in all_rules if k['fact']['relation']=='binary_predicate'),key=lambda k:hashlib.sha256(('application-pilot:42:'+k['id']).encode()).hexdigest())[:12]
    rng=random.Random(42);examples=[]
    def add(qid,question,choices,answer):
        choices=list(dict.fromkeys(choices));assert len(choices)==4 and answer in choices
        rng.shuffle(choices);gold='ABCD'[choices.index(answer)]
        prompt=question+'\n\nOptions:\n'+'\n'.join(a+'. '+v for a,v in zip('ABCD',choices))+ev.SUFFIX
        examples.append({'id':qid+':practice:'+str(len(examples)),'qid':qid,'messages':[{'role':'system','content':ev.SYSTEM},{'role':'user','content':prompt},{'role':'assistant','content':gold}]})
    names={'category':'object type','shape':'shape','color':'main-body color','material':'material','texture':'texture','status':'physical condition','value':'complete object count'}
    audit=[]
    for k in selected:
        qid=k['id'];rule=json.loads(k['fact']['quote']);name=rule['predicate_name'];conditions=rule['conditions'];learning=k['material']['learning'];protocol=learning['inspection_recording_protocol']
        material=learning['text'];examples.append({'id':qid+':explain','qid':qid,'messages':[{'role':'user','content':f'Explain the rule or codebook named {name}, including its conditions, mappings, and how to apply it.'},{'role':'assistant','content':material}]})
        # Atomic rule knowledge, presented as distinct alternatives of equal form.
        for c in conditions:
            answer=f"{names[c['field']]} {c['operator']} {c['value']}"
            distractors=[f"{names[c['field']]} {op} {c['value']}" for op in OPS if op!=c['operator']][:3]
            for repeat in range(2):add(qid,f'For {name}, what exact requirement applies to {names[c["field"]]}?',[answer,*distractors],answer)
        for outcome,code in protocol['outcome_to_code'].items():
            for repeat in range(2):add(qid,f'Under the {name} inspection convention, which code means {outcome}?',['H7','J2','K9','M4'],code)
        for flags in itertools.product((False,True),repeat=len(conditions)):
            for variant in range(2):
                attrs={}
                for c,flag in zip(conditions,flags):
                    v=c['value'];field=c['field'];op=c['operator']
                    if isinstance(v,(int,float)):
                        candidates=[v-3-variant,v-2-variant,v+2+variant,v+3+variant]
                    else:
                        wrong='undamaged' if v=='damaged' else ('damaged' if field=='status' else 'a different '+names[field])
                        candidates=[v,wrong]
                    attrs[field]=next(x for x in candidates if bool(OPS[op](x,v))==flag)
                actual=[OPS[c['operator']](attrs[c['field']],c['value']) for c in conditions]
                assert tuple(actual)==flags
                summary='; '.join(names[f]+': '+str(v) for f,v in attrs.items())
                pattern=tuple(actual[i-1] for i in protocol['checked_indices'])
                outcomes=list(protocol['outcome_to_code'])
                truth=lambda s:tuple(x.endswith('=Met') for x in s.split('; '))
                outcome=next(s for s in outcomes if truth(s)==pattern);code=protocol['outcome_to_code'][outcome]
                add(qid,f'Inspect a new observation under {name}. Which recording code is required? Evaluate the named requirements independently.\nObservation: {summary}.',['H7','J2','K9','M4'],code)
                qualification=all(actual) if rule['mode']=='all' else any(actual)
                target='; '.join(f'Check {i+1}={"Pass" if f else "Fail"}' for i,f in enumerate(actual))+f'; Qualification={"Pass" if qualification else "Fail"}'
                # Each observation appears once with a correct report and once
                # with a wrong report; this prevents always-accept shortcuts.
                alternatives=[]
                for alt in itertools.product((False,True),repeat=len(conditions)):
                    q=all(alt) if rule['mode']=='all' else any(alt)
                    s='; '.join(f'Check {i+1}={"Pass" if f else "Fail"}' for i,f in enumerate(alt))+f'; Qualification={"Pass" if q else "Fail"}'
                    if s!=target:alternatives.append(s)
                rng.shuffle(alternatives)
                for accept in (False,True):
                    report=target if accept else alternatives[0]
                    results=[target,*alternatives[:3]]
                    options=[('Accept report. ' if s==report else 'Replace report. ')+s for s in results]
                    correct=('Accept report. ' if accept else 'Replace report. ')+target
                    add(qid,f'A technician submitted this report under {name}: {report}. Check the observation and rule.\nObservation: {summary}.',options,correct)
                audit.append({'qid':qid,'attributes':attrs,'condition_outcomes':actual,'record_code':code,'qualification':qualification})
    rng.shuffle(examples);ev.save_rows(out/'train.jsonl',examples);ev.save_rows(out/'generated_solver_checks.jsonl',audit)
    # Evaluation data is read only after all training rows have been written.
    ids={k['id'] for k in selected};labels={x['id']:x for x in rows(data/'audit/eval_labels.jsonl')}
    evaluation=[ev.prepare_row(x,data,'T2T') for x in rows(data/'T2T/eval.jsonl') if x['qid'] in ids]
    assert len(evaluation)==24
    for r in evaluation:r['gold']=labels[r['id']]['answer'];r['question_type']=labels[r['id']]['question_type']
    ev.save_rows(out/'eval.jsonl',evaluation)
    save(out/'manifest.json',{'rules':sorted(ids),'training_rows':len(examples),'training_source':'learning text and fact quote only; programmatically generated new observations','no_eval_labels_used_in_training':True,'source_knowledge_sha256':sha(data/'audit/rule_knowledge.jsonl'),'train_sha256':sha(out/'train.jsonl'),'evaluation':'24 original questions for the same 12 learned rules; pilot only','epochs':5,'modality':'TEXT','new_images':False,'comparability_limit':'12-rule augmented pilot versus original 1000-rule model; changes both exposure and training content, not a full-scale controlled result'})


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--gpu',default='0');a=ap.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    work=ROOT/'runs/v7_full_experiment';data=ROOT/'rules_repaired_single_image_v7'
    build(data,work,out);env=environment(a.gpu,data);cp=final_checkpoint(out/'checkpoint',5)
    if not cp:
        with reserve(work,a.gpu,str(out)+':training',16000) as reservation:
            cmd=train_command('qwen3','sft',out/'train.jsonl',out/'checkpoint',5)
            with (out/'train.log').open('a') as f:
                p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);reservation.bind(p.pid);save(out/'state.json',{'status':'training','pid':p.pid,'command':cmd,'started_at':ev.now()});code=p.wait()
            if code:raise RuntimeError(f'pilot train failed {code}')
        cp=final_checkpoint(out/'checkpoint',5);assert cp
    os.environ.update(env)
    with reserve(work,a.gpu,str(out)+':evaluation',22000) as reservation:
        reservation.bind(os.getpid())
        import torch
        from swift.infer_engine import TransformersEngine,InferRequest,RequestConfig
        from swift.utils import seed_everything
        seed_everything(42);samples=rows(out/'eval.jsonl');results=[]
        old=Path(read(work/'jobs/qwen3_sft_text/checkpoint_manifest.json')['path'])
        for condition,path in [('original_1000_rule_sft',old),('augmented_12_rule_sft',cp)]:
            engine=TransformersEngine(str(MODELS['qwen3']),adapters=[str(path)],torch_dtype=torch.bfloat16,device_map='cuda:0',max_batch_size=16,attn_impl='sdpa');engine.template.enable_thinking=False
            for start in range(0,len(samples),16):
                batch=samples[start:start+16];res=engine.infer([InferRequest(messages=copy.deepcopy(r['messages'])) for r in batch],RequestConfig(max_tokens=32,temperature=0,seed=42),use_tqdm=False)
                assert len(res)==len(batch)
                for r,x in zip(batch,res):
                    if isinstance(x,Exception):raise x
                    text=x.choices[0].message.content or '';letter=ev.letter(text);results.append({'condition':condition,'id':r['id'],'question_type':r['question_type'],'gold':r['gold'],'response':text,'correct':letter==r['gold']})
            del engine
            import gc
            gc.collect();torch.cuda.empty_cache()
        ev.save_rows(out/'predictions.jsonl',results)
        report={c:{'n':len([r for r in results if r['condition']==c]),'correct':sum(r['correct'] for r in results if r['condition']==c),'by_type':{kind:{'n':sum(r['condition']==c and r['question_type']==kind for r in results),'correct':sum(r['correct'] for r in results if r['condition']==c and r['question_type']==kind)} for kind in ('direct','diagnosis')}} for c in ('original_1000_rule_sft','augmented_12_rule_sft')}
        save(out/'report.json',{'status':'complete','results':report,'checkpoint':str(cp),'finished_at':ev.now()});save(out/'state.json',{'status':'complete','finished_at':ev.now()});print(report,flush=True)

if __name__=='__main__':main()
