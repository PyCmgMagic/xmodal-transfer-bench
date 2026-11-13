#!/usr/bin/env python3
"""Factor actual observations/options into an otherwise fixed rule-recall request."""
import argparse
import copy
import dataclasses
import json
import os
from pathlib import Path

import run_v7_evaluation as ev
from diagnose_v7_knowledge_access import PILOT, DATA, LEDGER, checkpoint
from run_v7_full_experiment import ROOT, MODELS, environment, read, rows, save, sha
from v7_gpu_resources import reserve


def prepare(out,arm):
    folder=out/arm/'recall_context';folder.mkdir(parents=True,exist_ok=True)
    source=rows(PILOT/'text_alias/task_wording_recall/recall_probes.jsonl')
    originals={p['qid']:p for p in rows(PILOT/'probes.jsonl') if p['condition']=='direct_T' and p['task']=='direct'}
    gold={p['qid']:p['gold'] for p in rows(PILOT/'probes.jsonl') if p['condition']=='card_training_prompt'}
    probes=[]
    for p in source:
        text=originals[p['qid']]['messages'][-1]['content'];heading,context=text.split('\n\nTest input:\n');observation,options=context.split('\n\nOptions:\n');assert options.endswith(ev.SUFFIX);options=options[:-len(ev.SUFFIX)]
        for condition in ('heading','heading_observation','heading_options','heading_observation_options'):
            z=copy.deepcopy(p);z.update(id=condition+':'+p['qid'],condition=condition,gold=gold[p['qid']])
            if 'observation' in condition:z['messages'][-1]['content']+='\n\nTest observations:\n'+observation
            if 'options' in condition:z['messages'][-1]['content']+='\n\nPossible answer options:\n'+options
            probes.append(z)
    config={'model':str(MODELS['qwen3']),'adapter':ev.checkpoint(checkpoint(arm)),'script_sha256':sha(Path(__file__)),'sources':{str(PILOT/'probes.jsonl'):sha(PILOT/'probes.jsonl'),str(PILOT/'text_alias/task_wording_recall/recall_probes.jsonl'):sha(PILOT/'text_alias/task_wording_recall/recall_probes.jsonl')},'probes_hash':ev.digest(probes),'experiment':'2x2 factorial: same rule recall request; add original test observations and/or original answer alternatives. No external reference or gold rule values added.','max_tokens':384,'seed':42,'temperature':0,'batch_size':4,'thinking':False,'scoring':'Exact compact full-card match is sufficient-only diagnostic. Nonexact responses require content audit.'}
    if (folder/'manifest.json').exists():assert read(folder/'manifest.json')==config
    else:save(folder/'manifest.json',config)
    if (folder/'probes.jsonl').exists():assert rows(folder/'probes.jsonl')==probes
    else:ev.save_rows(folder/'probes.jsonl',probes)
    return folder,probes


def run(out,arm,gpu):
    folder,probes=prepare(out,arm);raw=folder/'predictions.jsonl'
    with ev.run_lock(folder/'.lock'):
        records,notes=ev.read_rows(raw,allow_partial_tail=True) if raw.exists() else ([],[])
        if notes:ev.save_rows(raw,records)
        seen={r['id'] for r in records}
        if arm=='text_alias':
            previous={r['qid']:r for r in rows(PILOT/arm/'task_wording_recall/predictions.jsonl') if r['kind']=='recall'}
            for p in probes:
                if p['condition']!='heading' or p['id'] in seen:continue
                old=previous[p['qid']];assert old['input_sha256']==ev.digest({'messages':p['messages'],'images':[]})
                records.append({'id':p['id'],'qid':p['qid'],'condition':p['condition'],'response':old['response'],'gold':p['gold'],'correct_exact':old['response'].strip()==p['gold'],'finish_reason':old['finish_reason'],'raw_response':old['raw_response'],'input_sha256':old['input_sha256'],'reused_from':str(PILOT/arm/'task_wording_recall/predictions.jsonl')})
            ev.save_rows(raw,records)
        seen={r['id'] for r in records};pending=[p for p in probes if p['id'] not in seen]
        if pending:
            with reserve(LEDGER,gpu,str(folder),23000) as reservation:
                reservation.bind(os.getpid());os.environ.update(environment(gpu,DATA))
                import torch
                from swift.infer_engine import TransformersEngine,InferRequest,RequestConfig
                from swift.utils import seed_everything
                seed_everything(42)
                engine=TransformersEngine(str(MODELS['qwen3']),adapters=[str(checkpoint(arm))],torch_dtype=torch.bfloat16,device_map='cuda:0',max_batch_size=4,attn_impl='sdpa');engine.template.enable_thinking=False
                for start in range(0,len(pending),4):
                    batch=pending[start:start+4];outputs=engine.infer([InferRequest(messages=copy.deepcopy(p['messages'])) for p in batch],RequestConfig(max_tokens=384,temperature=0,seed=42,return_details=True),use_tqdm=False)
                    assert len(outputs)==len(batch)
                    for p,reply in zip(batch,outputs):
                        if isinstance(reply,Exception):raise reply
                        c=reply.choices[0];text=c.message.content or ''
                        r={'id':p['id'],'qid':p['qid'],'condition':p['condition'],'response':text,'gold':p['gold'],'correct_exact':text.strip()==p['gold'],'finish_reason':c.finish_reason,'raw_response':dataclasses.asdict(reply),'input_sha256':ev.digest({'messages':p['messages'],'images':[]})}
                        records.append(r)
                        with raw.open('a') as f:f.write(json.dumps(r,ensure_ascii=False,default=str)+'\n');f.flush();os.fsync(f.fileno())
                    save(folder/'state.json',{'status':'running','pid':os.getpid(),'saved':len(records),'expected':32});print(arm,start+len(batch),'/',len(pending),flush=True)
        assert len(records)==len({r['id'] for r in records})==32
        metrics={c:{'n':8,'correct_exact':sum(r['correct_exact'] for r in records if r['condition']==c),'truncated':sum(r['finish_reason']=='length' for r in records if r['condition']==c)} for c in sorted({p['condition'] for p in probes})}
        save(folder/'report.json',{'status':'complete','arm':arm,'metrics':metrics,'new_requests':sum('reused_from' not in r for r in records)})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'runs/v7_access_diagnostics_20261015');ap.add_argument('--arm',choices=('text_alias','full_sft_text'),required=True);ap.add_argument('--gpu',default='0');a=ap.parse_args();run(a.out.resolve(),a.arm,a.gpu)
