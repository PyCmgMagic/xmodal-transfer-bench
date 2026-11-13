#!/usr/bin/env python3
"""Last readout control: same uncorrected worksheets, options only at stage two."""
import argparse
import dataclasses
import json
import os
from pathlib import Path

import repair_v7_application_protocol as first
import repair_v7_application_two_stage as second
from run_v7_full_experiment import ROOT, rows, read, save, sha, environment
from run_v7_fullparam_ep10 import dense_identity
from v7_gpu_resources import reserve
import run_v7_evaluation as ev

OUT=first.OUT/'isolated_readout'
SYSTEM=('You convert a completed worksheet into a multiple-choice selection. '
        'Match its computed result and report decision to the complete option content. '
        'Do not confuse candidate IDs with option letters. '
        'For audit reports, rule parameters and accept/replace decision must agree as well as the result. '
        'Return exactly one uppercase OPTION letter A, B, C, or D, with no other text.')


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'manifest.json').exists():return
    second.validate();sources={};files={}
    original={p['input_mod']+':'+p['source_id']:p for p in rows(first.OUT/'probes.jsonl') if p['protocol']=='strict32'}
    for arm in first.ARMS:
        path=second.OUT/arm/'predictions.jsonl';rr=rows(path)
        assert len(rr)==144 and read(path.parent/'state.json')['status']=='complete'
        sources[str(path)]=sha(path);probes=[]
        for r in rr:
            if r['stage']!='worksheet':continue
            p=original[r['input_mod']+':'+r['source_id']]
            opts='\n'.join(f'{letter}. {text}' for letter,text in p['options'].items())
            text='Your own previously generated worksheet (uncorrected):\n'+r['response']+'\n\nOptions:\n'+opts+ev.SUFFIX
            probes.append({k:p[k] for k in ('source_id','qid','gold','task','relation','input_mod','evaluation_group')})
            probes[-1].update(id='isolated:'+r['input_mod']+':'+r['source_id'],
                messages=[{'role':'system','content':SYSTEM},{'role':'user','content':text}],
                worksheet_response_sha256=ev.digest(r['response']))
        assert len(probes)==72
        dest=OUT/f'{arm}_probes.jsonl';ev.save_rows(dest,probes);files[dest.name]=sha(dest)
    save(OUT/'manifest.json',{'created_at':ev.now(),'script_sha256':sha(Path(__file__)),
        'models':read(first.OUT/'manifest.json')['models'],'sources':sources,'files':files,
        'generation':{'max_tokens':32,'temperature':0,'seed':42,'batch':4,'thinking':False},
        'adaptive_reason':'Correct worksheet results sometimes became wrong letters when the full original reference/question/image was reintroduced. Isolate the readout step.',
        'scope':'Same 18 development rules, same saved uncorrected own worksheets. No reference, original observation, image, gold answer or correct result is supplied at this final step.',
        'limits':'First step still used correct text material. Still oracle application, not closed-book transfer. No independent confirmation.'})


def validate():
    m=read(OUT/'manifest.json');assert m['script_sha256']==sha(Path(__file__))
    for p,h in m['sources'].items():assert sha(Path(p))==h
    for p,h in m['files'].items():assert sha(OUT/p)==h


def run(arm,gpu):
    validate();m=read(OUT/'manifest.json');folder=OUT/arm;folder.mkdir(exist_ok=True)
    with ev.run_lock(OUT/(arm+'.lock')):
        path=folder/'predictions.jsonl';rr,notes=ev.read_rows(path,allow_partial_tail=True) if path.exists() else ([],[])
        if notes:ev.save_rows(path,rr)
        seen={r['id'] for r in rr};todo=[p for p in rows(OUT/f'{arm}_probes.jsonl') if p['id'] not in seen]
        if todo:
            save(folder/'state.json',{'status':'waiting_for_gpu','pid':os.getpid(),'saved':len(rr),'gpu':gpu})
            with reserve(ROOT/'runs/v7_full_experiment',gpu,str(folder)+':isolated',24000) as res:
                res.bind(os.getpid());os.environ.update(environment(gpu,first.DATA));os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
                import torch
                from swift.infer_engine import TransformersEngine,InferRequest,RequestConfig
                from swift.utils import seed_everything
                torch.set_num_threads(4);seed_everything(42)
                assert dense_identity(Path(m['models'][arm]['path']))==m['models'][arm]
                engine=TransformersEngine(m['models'][arm]['path'],model_type='qwen3_vl',torch_dtype=torch.bfloat16,device_map='cuda:0',max_batch_size=4,attn_impl='sdpa');engine.template.enable_thinking=False
                for start in range(0,len(todo),4):
                    batch=todo[start:start+4]
                    outputs=engine.infer([InferRequest(messages=p['messages']) for p in batch],RequestConfig(max_tokens=32,temperature=0,seed=42,return_details=True),use_tqdm=False)
                    assert len(outputs)==len(batch)
                    for p,o in zip(batch,outputs):
                        if isinstance(o,Exception):raise o
                        c=o.choices[0];response=c.message.content or '';pred=ev.letter(response)
                        record={k:v for k,v in p.items() if k!='messages'}
                        full=dataclasses.asdict(o);full.pop('prompt_token_ids',None);full.pop('prompt_logprobs',None)
                        record.update(response=response,predicted=pred,correct=pred==p['gold'],format_error=pred is None,
                            finish_reason=c.finish_reason,raw_response=full,input_sha256=ev.digest({'messages':p['messages'],'images':[]}))
                        with path.open('a') as f:f.write(json.dumps(record,ensure_ascii=False,default=str)+'\n');f.flush();os.fsync(f.fileno())
                        rr.append(record)
                    print(arm,'isolated',len(rr),'/',72,flush=True)
        assert len(rr)==72
        save(folder/'state.json',{'status':'complete','saved':72,'finished_at':ev.now()})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--arm',choices=first.ARMS);ap.add_argument('--gpu',default='0');a=ap.parse_args()
    if a.prepare:prepare()
    else:
        assert a.arm
        run(a.arm,a.gpu)
