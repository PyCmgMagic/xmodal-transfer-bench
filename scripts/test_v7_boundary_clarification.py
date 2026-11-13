#!/usr/bin/env python3
"""Same original learning PNG, original/clarified caption MCQ control."""
import copy
import dataclasses
import json
import os
from pathlib import Path

import repair_v7_application_protocol as parent
from build_v7_visual_boundary_clarification import QID, OUT as CANDIDATE
from run_v7_full_experiment import ROOT, rows, read, save, sha, environment
from run_v7_fullparam_ep10 import dense_identity
from v7_gpu_resources import reserve
import run_v7_evaluation as ev

OUT=parent.OUT/'caption_control'


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'manifest.json').exists():return
    original=next(r for r in rows(parent.DATA/'shared/train_multimodal.jsonl') if r['id']==QID)
    clarified=rows(CANDIDATE/'clarified_visual_material.jsonl')[0]
    questions=[
        ('boundaries','What are the lower and upper boundary shares defined by the pictured reference collections?',
         ['lower=0.25; upper=0.625','lower=0.5; upper=1','lower=0.25; upper=1','lower=0.5; upper=0.5']),
        ('trays','Give target-object and total-cell counts for each COMPLETE reference tray, excluding the isolated type-reference object.',
         ['left=2/8; right=5/8','left=2/4; right=4/4','left=2/8; right=4/8','left=5/8; right=2/8']),
        ('boundary_application','For a NEW separate collection with 2 vases among 8 total objects, what category applies under the pictured boundaries?',
         ['Samdis','Tovred','Lalcot','The share is outside every category'])]
    probes=[]
    for condition,r in (('original',original),('clarified',clarified)):
        caption=r['messages'][0]['content'].removeprefix('<image>')
        for qtype,question,choices in questions:
            for shift in range(4):
                opts=choices[shift:]+choices[:shift];gold='ABCD'[opts.index(choices[0])]
                text='<image>\nLearning reference:\n'+caption+'\n\n'+question+'\n\nOptions:\n'+'\n'.join(f'{letter}. {option}' for letter,option in zip('ABCD',opts))+ev.SUFFIX
                probes.append({'id':f'{condition}:{qtype}:{shift}','condition':condition,'question_type':qtype,'shift':shift,
                    'gold':gold,'messages':[{'role':'system','content':ev.SYSTEM},{'role':'user','content':text}],
                    'images':clarified['images']})
    ev.save_rows(OUT/'probes.jsonl',probes)
    model=read(parent.OUT/'manifest.json')['models']['base']
    save(OUT/'manifest.json',{'created_at':ev.now(),'script_sha256':sha(Path(__file__)),'model':model,
        'files':{'probes.jsonl':sha(OUT/'probes.jsonl')},'candidate_manifest_sha256':sha(CANDIDATE/'manifest.json'),
        'scope':'One rule; 3 semantic MCQs x 4 option rotations x 2 captions =24 correlated requests; learning image present; not parameter learning or transfer.',
        'temperature':0,'seed':42,'max_tokens':32,'batch':4,'thinking':False})


def run(gpu):
    m=read(OUT/'manifest.json');assert m['script_sha256']==sha(Path(__file__))
    assert sha(OUT/'probes.jsonl')==m['files']['probes.jsonl']
    assert sha(CANDIDATE/'manifest.json')==m['candidate_manifest_sha256']
    path=OUT/'predictions.jsonl'
    with ev.run_lock(OUT/'run.lock'):
        rr,notes=ev.read_rows(path,allow_partial_tail=True) if path.exists() else ([],[])
        if notes:ev.save_rows(path,rr)
        seen={r['id'] for r in rr};todo=[p for p in rows(OUT/'probes.jsonl') if p['id'] not in seen]
        if todo:
            save(OUT/'state.json',{'status':'waiting_for_gpu','pid':os.getpid(),'saved':len(rr),'gpu':gpu})
            with reserve(ROOT/'runs/v7_full_experiment',gpu,str(OUT)+':caption',24000) as res:
                res.bind(os.getpid());os.environ.update(environment(gpu,parent.DATA));os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
                import torch
                from swift.infer_engine import TransformersEngine,InferRequest,RequestConfig
                from swift.utils import seed_everything
                torch.set_num_threads(4);seed_everything(42)
                assert dense_identity(Path(m['model']['path']))==m['model']
                engine=TransformersEngine(m['model']['path'],model_type='qwen3_vl',torch_dtype=torch.bfloat16,device_map='cuda:0',max_batch_size=4,attn_impl='sdpa');engine.template.enable_thinking=False
                for start in range(0,len(todo),4):
                    batch=todo[start:start+4]
                    outputs=engine.infer([InferRequest(messages=copy.deepcopy(p['messages']),images=p['images']) for p in batch],RequestConfig(max_tokens=32,temperature=0,seed=42,return_details=True),use_tqdm=False)
                    assert len(outputs)==len(batch)
                    for p,o in zip(batch,outputs):
                        if isinstance(o,Exception):raise o
                        c=o.choices[0];response=c.message.content or '';pred=ev.letter(response)
                        record={k:v for k,v in p.items() if k not in ('messages','images')}
                        full=dataclasses.asdict(o);full.pop('prompt_token_ids',None);full.pop('prompt_logprobs',None)
                        record.update(response=response,predicted=pred,correct=pred==p['gold'],format_error=pred is None,
                            finish_reason=c.finish_reason,raw_response=full,input_sha256=ev.digest({'messages':p['messages'],'images':p['images']}))
                        with path.open('a') as f:f.write(json.dumps(record,ensure_ascii=False,default=str)+'\n');f.flush();os.fsync(f.fileno())
                        rr.append(record)
                    print('caption control',len(rr),'/',24,flush=True)
        assert len(rr)==24
        save(OUT/'state.json',{'status':'complete','saved':len(rr),'finished_at':ev.now()})
        summary={}
        for condition in ('original','clarified'):
            ss=[r for r in rr if r['condition']==condition]
            summary[condition]={'n':len(ss),'correct':sum(r['correct'] for r in ss),'format_errors':sum(r['format_error'] for r in ss),
                'by_question':{q:{'correct':sum(r['correct'] for r in ss if r['question_type']==q),'n':4} for q in ('boundaries','trays','boundary_application')}}
        save(OUT/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--gpu',default='1');a=ap.parse_args()
    if a.prepare:prepare()
    else:run(a.gpu)
