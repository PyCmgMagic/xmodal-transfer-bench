#!/usr/bin/env python3
"""Posthoc train-format diagnostic; never merged into formal MCQ results."""
import argparse
import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path

import pilot_v7_preference_learning as pilot
from run_v7_full_experiment import rows, save, read, ROOT, environment, MODELS, sha
from v7_gpu_resources import reserve
from v7_preference_reward import components

OUT = pilot.OUT / 'posthoc_json_diagnostic'


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT / 'manifest.json').exists():
        validate(); return
    blocks = rows(pilot.OUT / 'blocks.jsonl')
    knowledge = {k['id']: k for k in rows(pilot.OUT / 'knowledge.jsonl')}
    probes = []
    for split in ('train', 'holdout'):
        for qid in sorted(knowledge):
            rr = [r for r in blocks if r['qid'] == qid and r['split'] == split]
            rr.sort(key=lambda r: hashlib.sha256(('json-diagnostic-42:' + r['id']).encode()).hexdigest())
            for r in rr[:4]:
                probes.append(dict(id=r['id'], qid=qid, split=split, gold=r['gold'],
                     messages=[dict(role='system', content=pilot.SYSTEM), dict(role='user',
                         content=f'Apply {pilot.name(knowledge[qid])}.\nCandidates: ' + pilot.text_obs(r['observations']))]))
    pilot.write(OUT / 'probes.jsonl', probes)
    save(OUT / 'manifest.json',dict(created_at=pilot.ev.now(), source_sha256=sha(Path(__file__)),
         probes_sha256=sha(OUT / 'probes.jsonl'), primary_manifest_sha256=sha(pilot.OUT / 'manifest.json'),
         definition='Posthoc diagnostic: 4 SHA-selected training and holdout scenes/rule, JSON format identical to training. Not a new confirmation set or formal score.',
         n=64, generation=dict(max_tokens=256, temperature=0, thinking=False)))


def validate():
    m = read(OUT / 'manifest.json')
    assert m['source_sha256'] == sha(Path(__file__))
    assert m['probes_sha256'] == sha(OUT / 'probes.jsonl')
    assert m['primary_manifest_sha256'] == sha(pilot.OUT / 'manifest.json')


def evaluate(arm, gpu):
    validate(); pilot.validate()
    os.environ.update(environment(gpu, pilot.DATA)); os.environ.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    import torch
    from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
    from swift.utils import seed_everything
    torch.set_num_threads(4); seed_everything(42)
    folder=OUT/arm;folder.mkdir(exist_ok=True)
    path=folder/'predictions.jsonl'; prior=rows(path) if path.exists() else []
    seen={r['id'] for r in prior}; probes=rows(OUT/'probes.jsonl')
    cp=MODELS['qwen3'] if arm=='base' else pilot.cp(arm)
    with reserve(ROOT/'runs/v7_full_experiment',gpu,str(folder),22000) as reservation:
        reservation.bind(os.getpid())
        identity=pilot.dense_identity(cp) if arm=='base' else pilot.ev.checkpoint(cp)
        config=dict(model=identity,arm=arm,generation=read(OUT/'manifest.json')['generation'])
        old=read(folder/'eval_manifest.json');assert not old or old==config
        save(folder/'eval_manifest.json',config)
        engine=TransformersEngine(str(MODELS['qwen3']),adapters=None if arm=='base' else [str(cp)],
            model_type='qwen3_vl',torch_dtype=torch.bfloat16,device_map='cuda:0',max_batch_size=4,attn_impl='sdpa')
        engine.template.enable_thinking=False
        pending=[p for p in probes if p['id'] not in seen]
        for start in range(0,len(pending),4):
            batch=pending[start:start+4]
            replies=engine.infer([InferRequest(messages=copy.deepcopy(p['messages'])) for p in batch],
                RequestConfig(max_tokens=256,temperature=0,seed=42,return_details=True),use_tqdm=False)
            with path.open('a') as f:
                for p,r in zip(batch,replies):
                    if isinstance(r,Exception):raise r
                    text=r.choices[0].message.content or ''
                    raw=dataclasses.asdict(r);raw.pop('prompt_token_ids',None);raw.pop('prompt_logprobs',None)
                    z={k:v for k,v in p.items() if k!='messages'}
                    z.update(arm=arm,response=text,finish_reason=r.choices[0].finish_reason,raw_response=raw,
                         input_sha256=pilot.ev.digest(p['messages']),**components(text,p['gold']))
                    f.write(json.dumps(z,ensure_ascii=False)+'\n')
            print(arm,'JSON diagnostic',start+len(batch),'/',len(pending),flush=True)
    assert len(rows(path))==64


def score():
    validate(); result={}
    for arm in pilot.ARMS:
        rr=rows(OUT/arm/'predictions.jsonl');assert len(rr)==64
        result[arm]={}
        for split in ('train','holdout'):
            sub=[r for r in rr if r['split']==split]
            result[arm][split]=dict(n=len(sub),**{k:sum(r[k] for r in sub)/len(sub) for k in ('rule_exact','keep_exact','valid','reward')},
                  truncated=sum(r['finish_reason']=='length' for r in sub))
    save(OUT/'summary.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','evaluate','score'],required=True)
    p.add_argument('--arm',choices=pilot.ARMS,default='base');p.add_argument('--gpu',default='0');a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='evaluate':evaluate(a.arm,a.gpu)
    else:score()
