#!/usr/bin/env python3
"""Separate short-answer controls for frozen v7, never formal eval results."""
import argparse
import copy
import dataclasses
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path

import run_v7_evaluation as ev
from run_v7_full_experiment import ROOT, MODELS, environment, rows, read, save, sha
from v7_gpu_resources import reserve
from build_v7_experiment_data import name as training_rule_name


def build_probes(data, work):
    knowledge={x['content']['id']:x['content'] for x in rows(data/'audit/rule_knowledge.jsonl')}
    train={x['id']:x for x in rows(work/'data/sft_text.jsonl')}
    labels={x['id']:x for x in rows(data/'audit/eval_labels.jsonl')}
    groups=defaultdict(list)
    for k in knowledge.values():groups[k['fact']['relation']].append(k['id'])
    chosen=[]
    for kind, ids in sorted(groups.items()):
        chosen += sorted(ids,key=lambda q:hashlib.sha256(('diagnose-v7:42:'+q).encode()).hexdigest())[:4]
    selected=set(chosen)
    probes=[]
    for row in rows(data/'T2T/eval.jsonl'):
        if row['qid'] not in selected:continue
        ready=ev.prepare_row(row,data,'T2T')
        payload={'messages':ready['messages']}
        provided=copy.deepcopy(payload)
        provided['messages'][-1]['content']='Learning material for this question:\n'+train[row['qid']]['messages'][-1]['content']+'\n\n'+provided['messages'][-1]['content']
        probes.append({'id':row['id'],'qid':row['qid'],'task':'application','question_type':labels[row['id']]['question_type'],'gold':labels[row['id']]['answer'],'blind':payload,'material':provided})
    rng=random.Random(42)
    for index,qid in enumerate(chosen):
        k=knowledge[qid];others=sorted(q for q in groups[k['fact']['relation']] if q!=qid)
        # Some entity.name fields are inherited sample IDs, not semantic
        # identifiers shown during training. Only test genuinely named rules.
        if k['fact']['relation'] in ('mapping_codebook','three_way_category','finite_state_transition'):continue
        identifier=training_rule_name(k)
        rng.shuffle(others);choices=[qid,*others[:3]];rng.shuffle(choices)
        options=[]
        for letter,other in zip('ABCD',choices):
            text=train[other]['messages'][-1]['content']
            text=text.replace(training_rule_name(knowledge[other]),'the named rule')
            options.append(letter+'. '+text)
        assert len(set(options))==4
        content=f"Which stored learning material belongs to the rule named {identifier}?\n\n"+'\n\n'.join(options)+ev.SUFFIX
        probes.append({'id':qid+':recall','qid':qid,'task':'rule_recall','question_type':'recall','gold':'ABCD'[choices.index(qid)],'blind':{'messages':[{'role':'system','content':ev.SYSTEM},{'role':'user','content':content}]}})
    return probes


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gpu',default='0');ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    work=ROOT/'runs/v7_full_experiment';data=ROOT/'rules_repaired_single_image_v7';out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    cp=Path(read(work/'jobs/qwen3_sft_text/checkpoint_manifest.json')['path'])
    probes=build_probes(data,work)
    ev.save_rows(out/'probes.jsonl',probes)
    save(out/'manifest.json',{'model':str(MODELS['qwen3']),'adapter':ev.checkpoint(cp),'selection':'4 rule IDs per relation, SHA256 deterministic, seed42; no outcome filtering','rules':len({r['qid'] for r in probes}),'strict_only':True,'formal_evaluation':False,'script_sha256':sha(Path(__file__)),'created_at':ev.now()})
    with reserve(work,a.gpu,str(out)+':diagnostic',22000) as reservation:
        reservation.bind(os.getpid());os.environ.update(environment(a.gpu,data))
        import torch
        from swift.infer_engine import TransformersEngine,InferRequest,RequestConfig
        from swift.utils import seed_everything
        seed_everything(42)
        engine=TransformersEngine(str(MODELS['qwen3']),adapters=[str(cp)],torch_dtype=torch.bfloat16,device_map='cuda:0',max_batch_size=16,attn_impl='sdpa')
        engine.template.enable_thinking=False
        tensors=[(n,p) for n,p in engine.model.named_parameters() if 'lora_B' in n]
        save(out/'adapter_verification.json',{'model_class':type(engine.model).__name__,'active_adapters':str(getattr(engine.model,'active_adapters',None)),'lora_B_tensors':len(tensors),'nonzero_lora_B_tensors':sum(bool(torch.count_nonzero(p)) for _,p in tensors),'template_supports_system':engine.template.template_meta.support_system})
        raw=out/'raw.jsonl';saved=ev.read_rows(raw,allow_partial_tail=True)[0] if raw.exists() else [];seen={(x['condition'],x['id']) for x in saved}
        conditions=[(mode,task,variant) for mode in ('base','adapter') for task,variant in (('application','blind'),('application','material'),('rule_recall','blind'))]
        for mode,task,variant in conditions:
            condition=f'{mode}_{task}_{variant}';pending=[r for r in probes if r['task']==task and (condition,r['id']) not in seen]
            with engine.model.disable_adapter() if mode=='base' else nullcontext():
                for start in range(0,len(pending),16):
                    batch=pending[start:start+16]
                    request=RequestConfig(max_tokens=32,temperature=0,seed=42,return_details=True,logprobs=True,top_logprobs=4)
                    res=engine.infer([InferRequest(**copy.deepcopy(r[variant])) for r in batch],request,use_tqdm=False)
                    assert len(res)==len(batch)
                    for r,result in zip(batch,res):
                        if isinstance(result,Exception):raise result
                        c=result.choices[0];letter=ev.letter(c.message.content or '')
                        record={'condition':condition,'id':r['id'],'qid':r['qid'],'question_type':r['question_type'],'gold':r['gold'],'response':c.message.content or '','predicted':letter,'correct':letter==r['gold'],'logprobs':c.logprobs,'usage':dataclasses.asdict(result.usage),'prompt_sha256':ev.digest(r[variant]),'saved_at':ev.now()}
                        with raw.open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
                        saved.append(record)
            subset=[r for r in saved if r['condition']==condition];print(condition,len(subset),sum(r['correct'] for r in subset),flush=True)
        metrics={}
        for mode,task,variant in conditions:
            condition=f'{mode}_{task}_{variant}';subset=[r for r in saved if r['condition']==condition]
            metrics[condition]={'n':len(subset),'correct':sum(r['correct'] for r in subset),'accuracy':sum(r['correct'] for r in subset)/len(subset),'format_errors':sum(r['predicted'] is None for r in subset),'by_question_type':{kind:{'n':sum(r['question_type']==kind for r in subset),'correct':sum(r['correct'] for r in subset if r['question_type']==kind)} for kind in set(r['question_type'] for r in subset)}}
        index={(r['condition'],r['id']):r for r in saved};paired=[]
        for r in probes:
            task=r['task'];b=index[(f'base_{task}_blind',r['id'])];t=index[(f'adapter_{task}_blind',r['id'])]
            paired.append({'id':r['id'],'answer_changed':b['response']!=t['response'],'logprobs_changed':b['logprobs']!=t['logprobs']})
        save(out/'report.json',{'metrics':metrics,'adapter_effect':{'paired':len(paired),'answer_changed':sum(r['answer_changed'] for r in paired),'logprobs_changed':sum(r['logprobs_changed'] for r in paired)},'status':'complete','finished_at':ev.now()})

if __name__=='__main__':main()
