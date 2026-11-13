#!/usr/bin/env python3
"""Change only rule-name order/separators in task-wording recall probes."""
import argparse
import copy
import dataclasses
import json
import os
import re
from pathlib import Path

import run_v7_evaluation as ev
from build_v7_experiment_data import name
from pilot_v7_task_then_transfer import DATA, LEDGER
from run_v7_full_experiment import MODELS, environment, read, rows, save, sha
from v7_gpu_resources import reserve


def run(out,arm,gpu):
    source=out/arm/'task_wording_recall'
    assert read(source/'report.json')['status']=='complete'
    kmap={k['id']:k for k in rows(out/'target_knowledge.jsonl')}
    gold={p['qid']:p['gold'] for p in rows(out/'probes.jsonl') if p['condition']=='card_training_prompt'}
    old={r['qid']:r for r in rows(source/'predictions.jsonl') if r['kind']=='recall'}
    probes=[]
    for p in rows(source/'recall_probes.jsonl'):
        if kmap[p['qid']]['fact']['relation']!='three_way_category':continue
        z=copy.deepcopy(p);body=z['messages'][-1]['content']
        normalized=re.sub(r'(?<=Task: Apply ).*?(?= category rule\.)',name(kmap[p['qid']]),body)
        assert normalized!=body
        z['messages'][-1]['content']=normalized;z['id']='canonical_name:'+p['qid'];probes.append(z)
    assert len(probes)==4
    folder=out/arm/'rule_name_alias';folder.mkdir(exist_ok=True);cp=Path(read(out/arm/'checkpoint_manifest.json')['path'])
    config={'adapter':ev.checkpoint(cp),'source_recall_probes_sha256':sha(source/'recall_probes.jsonl'),'script_sha256':sha(Path(__file__)),'change':'Only order/separators of three category names in task heading; no threshold/count values added. Canonical order itself reveals category order, so this is diagnostic only, never a primary dataset repair.','temperature':0,'seed':42,'max_tokens':256}
    if (folder/'manifest.json').exists():assert read(folder/'manifest.json')==config
    else:save(folder/'manifest.json',config)
    ev.save_rows(folder/'probes.jsonl',probes)
    raw=folder/'predictions.jsonl';records=rows(raw) if raw.exists() else [];seen={r['id'] for r in records}
    pending=[p for p in probes if p['id'] not in seen]
    if pending:
        with reserve(LEDGER,gpu,str(folder),22000) as reservation:
            reservation.bind(os.getpid());os.environ.update(environment(gpu,DATA))
            import torch
            from swift.infer_engine import TransformersEngine,InferRequest,RequestConfig
            from swift.utils import seed_everything
            seed_everything(42)
            engine=TransformersEngine(str(MODELS['qwen3']),adapters=[str(cp)],torch_dtype=torch.bfloat16,device_map='cuda:0',max_batch_size=8,attn_impl='sdpa');engine.template.enable_thinking=False
            replies=engine.infer([InferRequest(messages=p['messages']) for p in pending],RequestConfig(max_tokens=256,seed=42,temperature=0,return_details=True),use_tqdm=False)
            for p,reply in zip(pending,replies):
                if isinstance(reply,Exception):raise reply
                c=reply.choices[0];text=c.message.content or ''
                r={'id':p['id'],'qid':p['qid'],'response':text,'gold':gold[p['qid']],'correct_exact':text.strip()==gold[p['qid']],'original_response':old[p['qid']]['response'],'original_correct_exact':old[p['qid']]['response'].strip()==gold[p['qid']],'finish_reason':c.finish_reason,'input_sha256':ev.digest(p['messages']),'raw_response':dataclasses.asdict(reply)}
                records.append(r);ev.save_rows(raw,records)
    assert len(records)==4
    save(folder/'report.json',{'status':'complete','arm':arm,'n':4,'original_task_names_exact':sum(r['original_correct_exact'] for r in records),'canonical_names_exact':sum(r['correct_exact'] for r in records),'primary':False,'order_cue_limitation':config['change']})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path('runs/v7_task_then_transfer_20261015/pilot_v2'));ap.add_argument('--arm',choices=('warmup','text','vision'),required=True);ap.add_argument('--gpu',default='0');a=ap.parse_args();run(a.out.resolve(),a.arm,a.gpu)
