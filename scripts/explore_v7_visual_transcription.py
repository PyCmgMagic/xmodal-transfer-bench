#!/usr/bin/env python3
"""Separate image observation extraction from rule application, with self/oracle controls."""
import argparse
import copy
import dataclasses
import json
import os
import shutil
from pathlib import Path

import run_v7_evaluation as ev
from explore_v7_identity_binding import OUT, SOURCE
from pilot_v7_atomic_supervision import DATA, LEDGER
from run_v7_full_experiment import ROOT, MODELS, rows, read, save, sha, environment
from v7_gpu_resources import reserve


def prepare(out):
    folder=out/'visual_transcription';folder.mkdir(exist_ok=True)
    if (folder/'manifest.json').exists():
        m=read(folder/'manifest.json');assert m['script_sha256']==sha(Path(__file__))
        for p,h in m['sources'].items():assert sha(Path(p))==h
        assert sha(folder/'extraction_probes.jsonl')==m['probes_sha256']
        return
    pp=rows(SOURCE/'probes.jsonl')
    recall={r['qid']:r for r in rows(SOURCE/'neutral_recall/atomic_text/predictions.jsonl') if r['condition']=='recall_neutral'}
    probes=[]
    for p in pp:
        if p['condition']!='direct_V':continue
        true=next(q['gold'] for q in pp if q['qid']==p['qid'] and q['condition']=='recall_canonical')
        for kind,rule in (('self',recall[p['qid']]['response']),('oracle',true)):
            q=('Read this test observation image.\n<image>\n\nRule context:\n'+rule+
               '\n\nTranscribe only the visible observations for candidates A, B, C, D. '
               'List the properties and quantities needed for this rule. If counting objects or marked cells, '
               'give the numerator and total separately and list their attributes when relevant. '
               'If measuring regions, state occupied and total regions. Do not apply the rule, '
               'assign a category, decide KEEP/OMIT, or choose an option. Do not guess illegible details.')
            probes.append({'id':kind+':'+p['id'],'qid':p['qid'],'condition':kind,'messages':[{'role':'user','content':q}],
                'images':p['images'],'rule_context':rule,'rule_context_sha256':ev.digest(rule),'original_id':p['id']})
    assert len(probes)==32
    ev.save_rows(folder/'extraction_probes.jsonl',probes)
    cp=Path(read(SOURCE/'atomic_text/checkpoint_manifest.json')['path'])
    sources=[SOURCE/'probes.jsonl',SOURCE/'neutral_recall/atomic_text/predictions.jsonl',SOURCE/'atomic_text/checkpoint_manifest.json']
    save(folder/'manifest.json',{'created_at':ev.now(),'script_sha256':sha(Path(__file__)),
        'sources':{str(p):sha(p) for p in sources},'probes_sha256':sha(folder/'extraction_probes.jsonl'),
        'adapter':ev.checkpoint(cp),'temperature':0,'seed':42,'batch':4,'max_tokens':{'extract':512,'apply':16},
        'protocol':'32 image transcriptions then 32 MCQ applications. Self uses uncorrected prior identifier-only recall. Oracle uses correct external rule. Application sees uncorrected own transcription, no image. Original options/labels unchanged.',
        'limits':['Same eight development rules; no independent confirmation.',
            'Transcription protocol is post-diagnostic and changes number of model calls.',
            'Oracle is a separate external-material capability check, never a no-material transfer score.',
            'No rule or transcription correction, no scoring-based retries, no raw image modifications.']})
    snap=out/'source_snapshots';snap.mkdir(exist_ok=True);shutil.copy2(Path(__file__),snap/Path(__file__).name)


def run(out,gpu):
    prepare(out);folder=out/'visual_transcription';m=read(folder/'manifest.json');cp=Path(m['adapter']['path'])
    raw=folder/'predictions.jsonl';rr,notes=ev.read_rows(raw,allow_partial_tail=True) if raw.exists() else ([],[])
    if notes:ev.save_rows(raw,rr)
    seen={r['id'] for r in rr};assert len(seen)==len(rr)
    extract=rows(folder/'extraction_probes.jsonl');original={p['id']:p for p in rows(SOURCE/'probes.jsonl')}

    def application():
        recalled={r['id']:r for r in rr if r['phase']=='extract'};assert len(recalled)==32
        result=[]
        for p in extract:
            z=copy.deepcopy(original[p['original_id']])
            transcription=recalled[p['id']]['response']
            z.update(id='apply:'+p['id'],condition=p['condition'],original_id=p['original_id'])
            assert z['messages'][-1]['content'].count('<image>')==1
            body=z['messages'][-1]['content'].replace('<image>','Test image observations transcribed by you (uncorrected):\n'+transcription)
            z['messages'][-1]['content']='Rule context:\n'+p['rule_context']+'\n\n'+body
            z['images']=[];z['transcription_sha256']=ev.digest(transcription);z['rule_context_sha256']=p['rule_context_sha256']
            result.append(z)
        return result

    if len(rr)<64:
        with reserve(LEDGER,gpu,str(folder),24000) as res:
            res.bind(os.getpid());os.environ.update(environment(gpu,DATA))
            import torch
            from swift.infer_engine import TransformersEngine,InferRequest,RequestConfig
            from swift.utils import seed_everything
            seed_everything(42)
            engine=TransformersEngine(str(MODELS['qwen3']),adapters=[str(cp)],torch_dtype=torch.bfloat16,
                                      device_map='cuda:0',max_batch_size=4,attn_impl='sdpa')
            engine.template.enable_thinking=False
            for phase in ('extract','apply'):
                pp=extract if phase=='extract' else application()
                if phase=='apply':ev.save_rows(folder/'application_probes.jsonl',pp)
                pending=[p for p in pp if p['id'] not in seen]
                for start in range(0,len(pending),4):
                    batch=pending[start:start+4]
                    outputs=engine.infer([InferRequest(messages=copy.deepcopy(p['messages']),images=p.get('images',[])) for p in batch],
                         RequestConfig(max_tokens=m['max_tokens'][phase],temperature=0,seed=42,return_details=True),use_tqdm=False)
                    assert len(outputs)==len(batch)
                    for p,reply in zip(batch,outputs):
                        if isinstance(reply,Exception):raise reply
                        c=reply.choices[0];text=c.message.content or '';pred=ev.letter(text) if phase=='apply' else None
                        r={k:p[k] for k in ('id','qid','condition','original_id','rule_context_sha256')}
                        if phase=='apply':r.update(gold=p['gold'],transcription_sha256=p['transcription_sha256'])
                        r.update(phase=phase,response=text,predicted=pred,correct=pred==p['gold'] if phase=='apply' else None,
                                 format_error=phase=='apply' and pred is None,finish_reason=c.finish_reason,
                                 input_sha256=ev.digest({'messages':p['messages'],'images':p.get('images',[])}),raw_response=dataclasses.asdict(reply))
                        rr.append(r);seen.add(r['id'])
                        with raw.open('a') as f:
                            f.write(json.dumps(r,ensure_ascii=False,default=str)+'\n');f.flush();os.fsync(f.fileno())
                    print(phase,len(rr),'/64',flush=True)
    assert len(rr)==len(seen)==64
    lookup={p['id']:p for p in extract+application()}
    for r in rr:
        p=lookup[r['id']];assert r['input_sha256']==ev.digest({'messages':p['messages'],'images':p.get('images',[])})
        assert r['response']==(r['raw_response']['choices'][0]['message']['content'] or '')
    metrics={}
    for condition in ('self','oracle'):
        group=[r for r in rr if r['phase']=='apply' and r['condition']==condition]
        metrics[condition]={'n':len(group),'correct':sum(r['correct'] for r in group),'format_errors':sum(r['format_error'] for r in group),
                            'truncated':sum(r['finish_reason']=='length' for r in group)}
    assert ev.checkpoint(cp)==m['adapter']
    save(folder/'report.json',{'status':'complete','metrics':metrics,'extraction_truncated':sum(r['phase']=='extract' and r['finish_reason']=='length' for r in rr)})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=OUT);ap.add_argument('--gpu',default='0');a=ap.parse_args()
    with ev.run_lock(a.out/'visual_transcription.lock'):run(a.out,a.gpu)
