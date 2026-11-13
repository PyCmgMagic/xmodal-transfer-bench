#!/usr/bin/env python3
"""Post-diagnostic robustness of direct gains: option rotations and missing names."""
import argparse
import copy
import dataclasses
import json
import os
import re
import shutil
from pathlib import Path

import explore_v7_identity_binding as study
import run_v7_evaluation as ev
from pilot_v7_task_then_transfer import rotate
from run_v7_full_experiment import ROOT, MODELS, rows, read, save, sha, environment
from v7_gpu_resources import reserve

ARMS=('warmup','atomic_vision')+study.NEW_ARMS


def prepare(out):
    folder=out/'rotations';folder.mkdir(exist_ok=True)
    if (folder/'design.json').exists():
        m=read(folder/'design.json');assert m['script_sha256']==sha(Path(__file__))
        assert m['source_sha256']==sha(study.SOURCE/'probes.jsonl')
        assert m['probes_sha256']==sha(folder/'probes.jsonl')
        return
    knowledge={k['id']:k for k in rows(study.SOURCE/'target_knowledge.jsonl')}
    probes=[]
    for p in rows(study.SOURCE/'probes.jsonl'):
        if not p['condition'].startswith('direct_'):continue
        k=knowledge[p['qid']];selection=p['relation']=='set_selection_rule'
        for identified in ((True,False) if selection else (True,)):
            for shift in range(4):
                z=copy.deepcopy(p)
                body,gold,_=rotate(z['messages'][-1]['content'],z['gold'],shift)
                if not identified:
                    old=study.name(k);assert old in body
                    body=re.sub(r'\b'+re.escape(old)+r'\b','an unnamed rule',body)
                    assert old not in body
                z.update(id=f'rotation:{identified}:{shift}:'+p['id'],original_id=p['id'],gold=gold,
                    condition=('named_' if identified else 'anonymous_')+p['condition'][-1],shift=shift)
                z['messages'][-1]['content']=body;probes.append(z)
    assert len(probes)==192
    ev.save_rows(folder/'probes.jsonl',probes)
    save(folder/'design.json',{'created_at':ev.now(),'script_sha256':sha(Path(__file__)),
        'source_sha256':sha(study.SOURCE/'probes.jsonl'),'probes_sha256':sha(folder/'probes.jsonl'),
        'arms':ARMS,'post_diagnostic':True,'reason':'New blind-name arm improved unrotated original MCQ score despite 0/8 free rule recall; check position sensitivity and whether true selector identifier adds evidence.',
        'protocol':'Original 32 direct MCQs x4 option positions, plus original 16 selector MCQs x4 with rule name removed. No material, no self recall. Images unchanged.',
        'anonymous_caveat':'Without identifier there is no uniquely specified rule. Reference-key match is a negative control statistic, NOT an accuracy on a valid task.',
        'units':'32 original cases clustered in8rules. Rotations are NOT independent facts.',
        'training':False})
    shutil.copy2(Path(__file__),out/'source_snapshots'/Path(__file__).name)


def run(out,arm,gpu):
    prepare(out);root=out/'rotations';folder=root/arm;folder.mkdir(exist_ok=True)
    cp=study.adapter(out,arm)
    config={'design_sha256':sha(root/'design.json'),'adapter':ev.checkpoint(cp),'temperature':0,'seed':42,'max_tokens':16,'batch':4,'thinking':False}
    if (folder/'manifest.json').exists():assert read(folder/'manifest.json')==config
    else:save(folder/'manifest.json',config)
    raw=folder/'predictions.jsonl';rr,notes=ev.read_rows(raw,allow_partial_tail=True) if raw.exists() else ([],[])
    if notes:ev.save_rows(raw,rr)
    seen={r['id'] for r in rr};assert len(seen)==len(rr)
    probes=rows(root/'probes.jsonl');pending=[p for p in probes if p['id'] not in seen]
    if pending:
        with reserve(study.pilot.LEDGER,gpu,str(folder),24000) as res:
            res.bind(os.getpid());os.environ.update(environment(gpu,study.pilot.DATA))
            import torch
            from swift.infer_engine import TransformersEngine,InferRequest,RequestConfig
            from swift.utils import seed_everything
            seed_everything(42)
            engine=TransformersEngine(str(MODELS['qwen3']),adapters=[str(cp)],torch_dtype=torch.bfloat16,
                device_map='cuda:0',max_batch_size=4,attn_impl='sdpa')
            engine.template.enable_thinking=False
            for start in range(0,len(pending),4):
                batch=pending[start:start+4]
                outputs=engine.infer([InferRequest(messages=copy.deepcopy(p['messages']),images=p.get('images',[])) for p in batch],
                    RequestConfig(max_tokens=16,temperature=0,seed=42,return_details=True),use_tqdm=False)
                assert len(batch)==len(outputs)
                for p,reply in zip(batch,outputs):
                    if isinstance(reply,Exception):raise reply
                    c=reply.choices[0];text=c.message.content or '';pred=ev.letter(text)
                    r={k:p[k] for k in ('id','qid','condition','shift','original_id','gold','relation','task')}
                    r.update(response=text,predicted=pred,correct=pred==p['gold'],format_error=pred is None,
                        finish_reason=c.finish_reason,input_sha256=ev.digest({'messages':p['messages'],'images':p.get('images',[])}),raw_response=dataclasses.asdict(reply))
                    rr.append(r);seen.add(r['id'])
                    with raw.open('a') as f:
                        f.write(json.dumps(r,ensure_ascii=False,default=str)+'\n');f.flush();os.fsync(f.fileno())
                print(arm,len(rr),'/192',flush=True)
    assert len(rr)==len(seen)==192
    lookup={p['id']:p for p in probes}
    for r in rr:
        p=lookup[r['id']]
        assert r['input_sha256']==ev.digest({'messages':p['messages'],'images':p.get('images',[])})
        assert r['response']==(r['raw_response']['choices'][0]['message']['content'] or '')
    metrics={}
    for condition in ('named_T','named_V','anonymous_T','anonymous_V'):
        group=[r for r in rr if r['condition']==condition]
        metrics[condition]={'correct':sum(r['correct'] for r in group),'n':len(group),
            'format_errors':sum(r['format_error'] for r in group),'truncated':sum(r['finish_reason']=='length' for r in group),
            'by_rule':{qid:sum(r['correct'] for r in group if r['qid']==qid)/sum(r['qid']==qid for r in group) for qid in sorted({r['qid'] for r in group})},
            'by_relation':{relation:{'n':sum(r['relation']==relation for r in group),'correct':sum(r['correct'] for r in group if r['relation']==relation)} for relation in sorted({r['relation'] for r in group})},
            'by_shift':{str(i):sum(r['correct'] for r in group if r['shift']==i) for i in range(4)}}
    assert ev.checkpoint(cp)==config['adapter']
    save(folder/'report.json',{'status':'complete','metrics':metrics,'diagnostic_only':True})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=study.OUT);ap.add_argument('--gpu',default='0');ap.add_argument('--arm',choices=ARMS);ap.add_argument('--prepare',action='store_true');a=ap.parse_args()
    if a.prepare:prepare(a.out)
    else:
        with ev.run_lock(a.out/('rotation_'+a.arm+'.lock')):run(a.out,a.arm,a.gpu)
