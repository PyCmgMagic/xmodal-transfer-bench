#!/usr/bin/env python3
"""Diagnostic adapter routing: base perception, trained rule application."""
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


def run(out,gpu,phase):
    folder=out/'base_reader';folder.mkdir(exist_ok=True)
    source=out/'visual_transcription/extraction_probes.jsonl'
    trained=Path(read(SOURCE/'atomic_text/checkpoint_manifest.json')['path'])
    config={'script_sha256':sha(Path(__file__)),'source_sha256':sha(source),'apply_adapter':ev.checkpoint(trained),
            'reader_adapter':None,'temperature':0,'seed':42,'max_tokens':{'extract':512,'apply':16},
            'diagnostic_only':True,'note':'Same extraction prompts as trained reader. Base model transcribes then trained adapter applies unchanged own self-recall or separate oracle. No correction/retrieval. This changes adapter routing and is not the original formal protocol.'}
    if (folder/'manifest.json').exists():assert read(folder/'manifest.json')==config
    else:
        save(folder/'manifest.json',config);shutil.copy2(Path(__file__),out/'source_snapshots'/Path(__file__).name)
    extract=rows(source)
    original={p['id']:p for p in rows(SOURCE/'probes.jsonl')}
    if phase=='extract':pp=extract
    else:
        er=rows(folder/'extract_predictions.jsonl');assert len(er)==32
        recalled={r['id']:r for r in er};pp=[]
        for p in extract:
            z=copy.deepcopy(original[p['original_id']]);trans=recalled[p['id']]['response']
            z.update(id='apply:'+p['id'],condition=p['condition'],original_id=p['original_id'],rule_context_sha256=p['rule_context_sha256'])
            z['messages'][-1]['content']='Rule context:\n'+p['rule_context']+'\n\n'+z['messages'][-1]['content'].replace('<image>','Test image observations transcribed by you (uncorrected):\n'+trans)
            z['images']=[];z['transcription_sha256']=ev.digest(trans);pp.append(z)
    ev.save_rows(folder/(phase+'_probes.jsonl'),pp)
    raw=folder/(phase+'_predictions.jsonl');rr,notes=ev.read_rows(raw,allow_partial_tail=True) if raw.exists() else ([],[])
    if notes:ev.save_rows(raw,rr)
    seen={r['id'] for r in rr};assert len(seen)==len(rr)
    pending=[p for p in pp if p['id'] not in seen]
    if pending:
        with reserve(LEDGER,gpu,str(folder)+':'+phase,24000) as res:
            res.bind(os.getpid());os.environ.update(environment(gpu,DATA))
            import torch
            from swift.infer_engine import TransformersEngine,InferRequest,RequestConfig
            from swift.utils import seed_everything
            seed_everything(42)
            engine=TransformersEngine(str(MODELS['qwen3']),adapters=[str(trained)] if phase=='apply' else None,
                torch_dtype=torch.bfloat16,device_map='cuda:0',max_batch_size=4,attn_impl='sdpa')
            engine.template.enable_thinking=False
            for start in range(0,len(pending),4):
                batch=pending[start:start+4]
                outputs=engine.infer([InferRequest(messages=copy.deepcopy(p['messages']),images=p.get('images',[])) for p in batch],
                    RequestConfig(max_tokens=config['max_tokens'][phase],temperature=0,seed=42,return_details=True),use_tqdm=False)
                assert len(batch)==len(outputs)
                for p,reply in zip(batch,outputs):
                    if isinstance(reply,Exception):raise reply
                    c=reply.choices[0];text=c.message.content or '';pred=ev.letter(text) if phase=='apply' else None
                    r={k:p[k] for k in ('id','qid','condition','original_id','rule_context_sha256')}
                    if phase=='apply':r.update(gold=p['gold'],transcription_sha256=p['transcription_sha256'])
                    r.update(response=text,predicted=pred,correct=pred==p['gold'] if phase=='apply' else None,
                        format_error=phase=='apply' and pred is None,finish_reason=c.finish_reason,
                        input_sha256=ev.digest({'messages':p['messages'],'images':p.get('images',[])}),raw_response=dataclasses.asdict(reply))
                    rr.append(r);seen.add(r['id'])
                    with raw.open('a') as f:
                        f.write(json.dumps(r,ensure_ascii=False,default=str)+'\n');f.flush();os.fsync(f.fileno())
                print(phase,len(rr),'/32',flush=True)
    assert len(rr)==len(seen)==32
    for p,r in ((p,next(r for r in rr if r['id']==p['id'])) for p in pp):
        assert r['input_sha256']==ev.digest({'messages':p['messages'],'images':p.get('images',[])})
        assert r['response']==(r['raw_response']['choices'][0]['message']['content'] or '')
    assert ev.checkpoint(trained)==config['apply_adapter']
    if phase=='apply':
        metrics={}
        for condition in ('self','oracle'):
            group=[r for r in rr if r['condition']==condition]
            metrics[condition]={'n':len(group),'correct':sum(r['correct'] for r in group),'format_errors':sum(r['format_error'] for r in group),
                'truncated':sum(r['finish_reason']=='length' for r in group)}
        save(folder/'report.json',{'status':'complete','metrics':metrics,
            'extraction_truncated':sum(r['finish_reason']=='length' for r in rows(folder/'extract_predictions.jsonl')),'diagnostic_only':True})


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=OUT);ap.add_argument('--gpu',default='0');ap.add_argument('--phase',choices=['extract','apply'],required=True);a=ap.parse_args()
    with ev.run_lock(a.out/('base_reader_'+a.phase+'.lock')):run(a.out,a.gpu,a.phase)
