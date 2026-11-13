#!/usr/bin/env python3
"""Auxiliary two-stage recall using actual task wording; never primary transfer."""
import argparse
import copy
import dataclasses
import json
import os
from pathlib import Path

import run_v7_evaluation as ev
from pilot_v7_task_then_transfer import DATA, LEDGER, messages, validate_sources
from run_v7_full_experiment import MODELS, environment, read, rows, save, sha
from v7_gpu_resources import reserve


def run(out,arm,gpu):
    assert read(out/arm/'report.json').get('status')=='complete'
    validate_sources(out)
    folder=out/arm/'task_wording_recall';folder.mkdir(exist_ok=True)
    probes=rows(out/'probes.jsonl');direct=[p for p in probes if p['condition'].startswith('direct_')]
    cp=Path(read(out/arm/'checkpoint_manifest.json')['path']) if arm!='base' else None
    config={'adapter':ev.checkpoint(cp) if cp else None,'source_probes_sha256':sha(out/'probes.jsonl'),'card_selection':'one new fixed recall query using original direct-task heading; no best-of sampling','external_rule_values':False,'max_tokens':{'recall':256,'application':16},'seed':42,'temperature':0,'thinking':False,'script_sha256':sha(Path(__file__))}
    if (folder/'manifest.json').exists():assert read(folder/'manifest.json')==config
    else:save(folder/'manifest.json',config)
    source=[]
    for p in direct:
        if p['condition']!='direct_T' or p['task']!='direct':continue
        heading=p['messages'][-1]['content'].split('\n\nTest input:')[0]
        q='Give the exact decision rule stored for this task, including every required value. Recall the rule; do not answer a test case.\nTask: '+heading
        source.append({'id':'recall:'+p['qid'],'qid':p['qid'],'messages':messages(q)})
    assert len(source)==8
    ev.save_rows(folder/'recall_probes.jsonl',source)
    raw=folder/'predictions.jsonl';records,notes=ev.read_rows(raw,allow_partial_tail=True) if raw.exists() else ([],[])
    if notes:ev.save_rows(raw,records)
    seen={r['id'] for r in records};assert len(seen)==len(records)
    if len(records)<40:
        with reserve(LEDGER,gpu,str(folder),25000) as reservation:
            reservation.bind(os.getpid());os.environ.update(environment(gpu,DATA))
            import torch
            from swift.infer_engine import TransformersEngine,InferRequest,RequestConfig
            from swift.utils import seed_everything
            seed_everything(42)
            engine=TransformersEngine(str(MODELS['qwen3']),adapters=[str(cp)] if cp else None,torch_dtype=torch.bfloat16,device_map='cuda:0',max_batch_size=8,attn_impl='sdpa');engine.template.enable_thinking=False
            def infer(group,kind):
                for start in range(0,len(group),8):
                    batch=group[start:start+8]
                    replies=engine.infer([InferRequest(messages=copy.deepcopy(p['messages']),images=p.get('images',[])) for p in batch],RequestConfig(max_tokens=256 if kind=='recall' else 16,seed=42,temperature=0,return_details=True),use_tqdm=False)
                    assert len(batch)==len(replies)
                    for p,reply in zip(batch,replies):
                        if isinstance(reply,Exception):raise reply
                        c=reply.choices[0];text=c.message.content or '';pred=ev.letter(text) if kind=='application' else None
                        r={'id':p['id'],'qid':p['qid'],'kind':kind,'response':text,'finish_reason':c.finish_reason,'input_sha256':ev.digest({'messages':p['messages'],'images':p.get('images',[])}),'raw_response':dataclasses.asdict(reply)}
                        if kind=='application':r.update(condition=p['condition'],task=p['task'],gold=p['gold'],predicted=pred,correct=pred==p['gold'],format_error=pred is None)
                        with raw.open('a') as f:f.write(json.dumps(r,ensure_ascii=False,default=str)+'\n');f.flush();os.fsync(f.fileno())
                        records.append(r)
                    print(arm,kind,start+len(batch),'/',len(group),flush=True)
            infer([p for p in source if p['id'] not in seen],'recall')
            recalled={r['qid']:r['response'] for r in records if r['kind']=='recall'}
            assert len(recalled)==8
            application=[]
            for p in direct:
                z=copy.deepcopy(p);z['id']='self_application:'+p['id']
                z['messages'][-1]['content']='Rule reference recalled by this model:\n'+recalled[p['qid']]+'\n\nTest question:\n'+z['messages'][-1]['content']
                application.append(z)
            if (folder/'application_probes.jsonl').exists():assert rows(folder/'application_probes.jsonl')==application
            else:ev.save_rows(folder/'application_probes.jsonl',application)
            infer([p for p in application if p['id'] not in seen],'application')
    assert len(records)==40 and len({r['id'] for r in records})==40
    canonical={p['qid']:p['gold'] for p in probes if p['condition']=='card_training_prompt'}
    metrics={}
    for c in ('direct_T','direct_V'):
        rr=[r for r in records if r.get('condition')==c]
        metrics[c]={'n':len(rr),'correct':sum(r['correct'] for r in rr),'format_errors':sum(r['format_error'] for r in rr)}
    save(folder/'report.json',{'status':'complete','arm':arm,'metrics':metrics,'task_wording_recall_exact':sum(r['response'].strip()==canonical[r['qid']] for r in records if r['kind']=='recall'),'recall_n':8,'primary':False,'protocol':'two-stage self-recall and application; generated reference may contain errors, no correction with gold; no external rule materials supplied','saved':40})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path('runs/v7_task_then_transfer_20261015/pilot_v2'));ap.add_argument('--arm',choices=('warmup','text','vision','text_alias'),required=True);ap.add_argument('--gpu',default='0');a=ap.parse_args();run(a.out.resolve(),a.arm,a.gpu)
