#!/usr/bin/env python3
"""Hold all candidate data/options fixed, change only which learned selector is requested."""
import argparse
import copy
import dataclasses
import json
import os
import random
from pathlib import Path

import run_v7_evaluation as ev
from diagnose_v7_knowledge_access import PILOT, DATA, LEDGER, checkpoint
from build_v7_experiment_data import name
from pilot_v7_task_then_transfer import card
from run_v7_full_experiment import ROOT, MODELS, environment, rows, read, save, sha
from v7_gpu_resources import reserve


def solve(rule,x):
    cs=rule['conditions'];flags=[]
    for c in cs:
        a,b,op=x[c['field']],c['value'],c['operator']
        flags.append(a==b if op=='eq' else a!=b if op=='ne' else a>=b if op=='ge' else a>b if op=='gt' else a<=b if op=='le' else a<b)
    selected=sum(flags)>=rule['required_count'] if 'required_count' in rule else all(flags) if rule['mode']=='all' else any(flags)
    return 'KEEP' if selected else 'OMIT'


def independent(rule,x):
    if rule['selection_topology']=='interval_filter':
        low,high=rule['conditions'];v=x['value']
        keep=(v>low['value'] or v==low['value'] and low['operator']=='ge') and (v<high['value'] or v==high['value'] and high['operator']=='le')
    else:
        matches=sum(x[c['field']]==c['value'] for c in rule['conditions'])
        keep=matches>=rule['required_count'] if 'required_count' in rule else matches==len(rule['conditions'])
    return 'KEEP' if keep else 'OMIT'


def prepare(folder):
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'design.json').exists():
        m=read(folder/'design.json');assert m['script_sha256']==sha(Path(__file__)) and m['probes_sha256']==sha(folder/'probes.jsonl');return
    knowledge=[k for k in rows(PILOT/'target_knowledge.jsonl') if k['fact']['relation']=='set_selection_rule'];assert len(knowledge)==4
    rng=random.Random('rule-name-control:42');blocks=[];probes=[];seen=set()
    for block in range(8):
        for attempt in range(10000):
            obs=[{'category':rng.choice(['tool tray','bucket','bowl','protective cover']),'material':rng.choice(['plastic','wood','metal']),'color':rng.choice(['red','white','blue']),'value':rng.choice([2,3,4,5,6,10,14,15])} for i in range(4)]
            sig=ev.digest(obs)
            if sig in seen:continue
            answers={}
            for k in knowledge:
                rule=json.loads(k['fact']['quote']);out=[]
                for x in obs:
                    assert solve(rule,x)==independent(rule,x);out.append(solve(rule,x))
                answers[k['id']]='; '.join(f'{letter}={value}' for letter,value in zip('ABCD',out))
            if len(set(answers.values()))==4:break
        else:raise RuntimeError('Could not construct a discriminating collection')
        seen.add(sig);options=list(answers.values());rng.shuffle(options)
        body='\n\nTest input:\n'+'; '.join(f"Candidate {letter}: category: {x['category']}, material: {x['material']}, color: {x['color']}, value: {x['value']}" for letter,x in zip('ABCD',obs))+'.\n\nOptions:\n'+'\n'.join(f'{letter}. Result: {value}.' for letter,value in zip('ABCD',options))+ev.SUFFIX
        block_record={'block':block,'observations':obs,'options':options,'rule_results':answers,'four_different_correct_answers':True,'shared_body_sha256':ev.digest(body)};blocks.append(block_record)
        for k in knowledge:
            q='Apply '+name(k)+'. Report KEEP/OMIT for each candidate; KEEP means selected and OMIT means not selected.'+body
            gold='ABCD'[options.index(answers[k['id']])]
            for condition in ('blind','oracle'):
                text=q if condition=='blind' else 'Learning reference:\n'+card(k)+'\n\n'+q
                probes.append({'id':f'{condition}:{block}:{k["id"]}','qid':k['id'],'block':block,'condition':condition,'gold':gold,'messages':[{'role':'system','content':ev.SYSTEM},{'role':'user','content':text}]})
    assert len(probes)==64
    for block in range(8):
        group=[p for p in probes if p['block']==block and p['condition']=='blind'];assert {p['gold'] for p in group}==set('ABCD')
        assert len({p['messages'][-1]['content'].split('. Report KEEP/OMIT',1)[1] for p in group})==1
    ev.save_rows(folder/'probes.jsonl',probes);ev.save_rows(folder/'blocks.jsonl',blocks)
    save(folder/'design.json',{'script_sha256':sha(Path(__file__)),'source':str(PILOT/'target_knowledge.jsonl'),'source_sha256':sha(PILOT/'target_knowledge.jsonl'),'probes_sha256':sha(folder/'probes.jsonl'),'blocks':8,'queries_per_condition':32,'conditions':['blind','oracle'],'meaning':'Within each block observations/options identical, only requested rule name changes. Four rules give four different valid results; each correct letter occurs once. Two independent solvers checked every result.','training':False,'primary':False,'images':'None; new textual diagnostic observations, no original image edits','limitation':'Diagnostic with deliberately richer observations containing both numeric and object attributes; original target eval unchanged.8blocks,not32independentworldstates.'})


def run(out,arm,gpu):
    root=out/'rule_name_control';prepare(root);folder=root/arm;folder.mkdir(exist_ok=True)
    config={'adapter':ev.checkpoint(checkpoint(arm)),'design_sha256':sha(root/'design.json'),'seed':42,'temperature':0,'max_tokens':16,'batch_size':8}
    if (folder/'manifest.json').exists():assert read(folder/'manifest.json')==config
    else:save(folder/'manifest.json',config)
    raw=folder/'predictions.jsonl';records,notes=ev.read_rows(raw,allow_partial_tail=True) if raw.exists() else ([],[])
    if notes:ev.save_rows(raw,records)
    probes=rows(root/'probes.jsonl');seen={r['id'] for r in records};pending=[p for p in probes if p['id'] not in seen]
    if pending:
        with reserve(LEDGER,gpu,str(folder),20000) as reservation:
            reservation.bind(os.getpid());os.environ.update(environment(gpu,DATA))
            import torch
            from swift.infer_engine import TransformersEngine,InferRequest,RequestConfig
            from swift.utils import seed_everything
            seed_everything(42)
            engine=TransformersEngine(str(MODELS['qwen3']),adapters=[str(checkpoint(arm))],torch_dtype=torch.bfloat16,device_map='cuda:0',max_batch_size=8,attn_impl='sdpa');engine.template.enable_thinking=False
            for start in range(0,len(pending),8):
                batch=pending[start:start+8];outputs=engine.infer([InferRequest(messages=copy.deepcopy(p['messages'])) for p in batch],RequestConfig(max_tokens=16,temperature=0,seed=42,return_details=True),use_tqdm=False)
                for p,reply in zip(batch,outputs):
                    if isinstance(reply,Exception):raise reply
                    c=reply.choices[0];text=c.message.content or '';pred=ev.letter(text)
                    r={k:p[k] for k in ('id','qid','block','condition','gold')};r.update(response=text,predicted=pred,correct=pred==p['gold'],format_error=pred is None,finish_reason=c.finish_reason,input_sha256=ev.digest(p['messages']),raw_response=dataclasses.asdict(reply));records.append(r)
                    with raw.open('a') as f:f.write(json.dumps(r,ensure_ascii=False,default=str)+'\n');f.flush();os.fsync(f.fileno())
                print(arm,'name control',start+len(batch),'/',len(pending),flush=True)
    assert len(records)==len({r['id'] for r in records})==64
    metrics={}
    for condition in ('blind','oracle'):
        rr=[r for r in records if r['condition']==condition];blocks=[[r for r in rr if r['block']==b] for b in range(8)]
        metrics[condition]={'n':32,'correct':sum(r['correct'] for r in rr),'format_errors':sum(r['format_error'] for r in rr),'all4_names_same_option_blocks':sum(len({r['predicted'] for r in g})==1 for g in blocks),'all4_names_correct_blocks':sum(all(r['correct'] for r in g) for g in blocks),'distinct_predictions_per_block':[len({r['predicted'] for r in g}) for g in blocks]}
    save(folder/'report.json',{'status':'complete','arm':arm,'metrics':metrics})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'runs/v7_access_diagnostics_20261015');ap.add_argument('--arm',choices=('warmup','text_alias'),required=True);ap.add_argument('--gpu',default='0');a=ap.parse_args();run(a.out.resolve(),a.arm,a.gpu)
