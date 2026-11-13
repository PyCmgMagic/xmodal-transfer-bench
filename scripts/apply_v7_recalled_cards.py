#!/usr/bin/env python3
"""Apply a pilot adapter's own complete recalled cards to original questions."""
import argparse
import copy
import dataclasses
import json
import os
from pathlib import Path

import run_v7_evaluation as ev
from explore_v7_retrieval_repair import DATA,WORK
from run_v7_full_experiment import MODELS,environment,read,rows,save,sha
from v7_gpu_resources import reserve


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--arm',default='cards');ap.add_argument('--gpu',default='0');a=ap.parse_args()
    out=a.out.resolve();folder=out/a.arm/'full_card_application';folder.mkdir(exist_ok=True)
    recalled={r['qid']:r['response'] for r in rows(out/a.arm/'bridge/predictions.jsonl') if r['condition']=='full_card_training_prompt'}
    probes=[]
    for p in rows(out/'probes.jsonl'):
        if p['condition'] not in ('original_T2T','original_T2V'):continue
        z=copy.deepcopy(p);z['id']+=':self_full_card';z['condition']='self_full_card:'+p['condition']
        z['messages'][-1]['content']='Learning material:\n'+recalled[p['qid']]+'\n\n'+z['messages'][-1]['content']
        probes.append(z)
    assert len(probes)==32
    ev.save_rows(folder/'probes.jsonl',probes)
    cp=Path(read(out/a.arm/'checkpoint_manifest.json')['path'])
    config={'adapter':ev.checkpoint(cp),'recall_source_sha256':sha(out/a.arm/'bridge/predictions.jsonl'),
            'probes_sha256':sha(folder/'probes.jsonl'),'script_sha256':sha(Path(__file__)),
            'external_learning_material':False,'context_source':'same adapter generated full_card_training_prompt response',
            'temperature':0,'seed':42,'max_tokens':16,'batch_size':8}
    if (folder/'manifest.json').exists():assert read(folder/'manifest.json')['config']==config
    else:save(folder/'manifest.json',{'config':config,'created_at':ev.now()})
    raw=folder/'predictions.jsonl';records=rows(raw) if raw.exists() else []
    seen={r['id'] for r in records};pending=[p for p in probes if p['id'] not in seen]
    if pending:
        with reserve(WORK,a.gpu,str(folder),22000) as reservation:
            reservation.bind(os.getpid());os.environ.update(environment(a.gpu,DATA))
            import torch
            from swift.infer_engine import TransformersEngine,InferRequest,RequestConfig
            from swift.utils import seed_everything
            seed_everything(42)
            engine=TransformersEngine(str(MODELS['qwen3']),adapters=[str(cp)],torch_dtype=torch.bfloat16,
                                      device_map='cuda:0',max_batch_size=8,attn_impl='sdpa')
            engine.template.enable_thinking=False
            for start in range(0,len(pending),8):
                batch=pending[start:start+8]
                outputs=engine.infer([InferRequest(messages=copy.deepcopy(p['messages']),images=p.get('images',[])) for p in batch],
                    RequestConfig(max_tokens=16,temperature=0,seed=42,return_details=True),use_tqdm=False)
                for p,output in zip(batch,outputs):
                    if isinstance(output,Exception):raise output
                    c=output.choices[0];text=c.message.content or ''
                    r={k:p[k] for k in ('id','qid','condition','gold')}
                    r.update(response=text,predicted=ev.letter(text),correct=ev.letter(text)==p['gold'],
                        format_error=ev.letter(text) is None,finish_reason=c.finish_reason,
                        prompt_sha256=ev.digest({'messages':p['messages'],'images':p.get('images',[])}),
                        raw_response=dataclasses.asdict(output),saved_at=ev.now())
                    with raw.open('a') as f:f.write(json.dumps(r,ensure_ascii=False,default=str)+'\n');f.flush();os.fsync(f.fileno())
                    records.append(r)
                print(start+len(batch),'/',len(pending),flush=True)
    assert len(records)==len({r['id'] for r in records})==32
    metrics={}
    for cond in sorted({p['condition'] for p in probes}):
        group=[r for r in records if r['condition']==cond]
        metrics[cond]={'n':len(group),'correct':sum(r['correct'] for r in group),
            'format_errors':sum(r['format_error'] for r in group),'truncated':sum(r['finish_reason']=='length' for r in group)}
    save(folder/'report.json',{'status':'complete','metrics':metrics,'external_learning_material':False,
                             'protocol':'two-stage full recalled-card context, separate from strict direct answering',
                             'finished_at':ev.now()})


if __name__=='__main__':main()
