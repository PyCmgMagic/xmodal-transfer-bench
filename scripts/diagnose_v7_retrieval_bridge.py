#!/usr/bin/env python3
"""Self-generated four-fact context versus identical gold-fact context."""
import argparse
import copy
import dataclasses
import json
import os
from pathlib import Path

import run_v7_evaluation as ev
from explore_v7_retrieval_repair import DATA, WORK, ARMS, atoms, score, message
from run_v7_full_experiment import MODELS, environment, read, rows, save, sha
from v7_gpu_resources import reserve


def build(out,arm):
    assert read(out/arm/'report.json').get('status')=='complete'
    knowledge={k['id']:k for k in rows(out/'knowledge.jsonl')}
    previous={r['id']:r for r in rows(out/arm/'predictions.jsonl')}
    result=[]
    for p in rows(out/'probes.jsonl'):
        if p['condition'] not in ('new_direct','new_diagnosis','original_T2T','original_T2V'):
            continue
        qid=p['qid'];facts=atoms(json.loads(knowledge[qid]['fact']['quote']))
        self_material='\n'.join(f'{focus}: '+previous[f'{qid}:new_fact:{i}:0']['response']
                                for i,(focus,gold) in enumerate(facts))
        gold_material='\n'.join(f'{focus}: {gold}' for focus,gold in facts)
        # Both prompts include the same field names; only the field values differ.
        for protocol,material in [('self_fact_context',self_material),('gold_fact_context',gold_material)]:
            z=copy.deepcopy(p);z['id']+=':'+protocol;z['condition']=protocol+':'+p['condition']
            z['messages'][-1]['content']='Rule facts for this question:\n'+material+'\n\n'+z['messages'][-1]['content']
            z['material']=material;z['material_source']=str(out/arm/'predictions.jsonl') if protocol.startswith('self') else str(out/'knowledge.jsonl')
            result.append(z)
    for k in knowledge.values():
        original=sorted([r for r in rows(out/'cards.jsonl') if r['qid']==k['id']],key=lambda r:int(r['id'].split(':')[-1]))[0]
        result.append({'id':k['id']+':card','qid':k['id'],'condition':'full_card_training_prompt',
            'task':'card','fmt':'card','gold':k['material']['learning']['text'],'messages':original['messages'][:-1]})
    assert len(result)==328
    folder=out/arm/'bridge';folder.mkdir(exist_ok=True)
    if (folder/'probes.jsonl').exists():assert rows(folder/'probes.jsonl')==result
    else:ev.save_rows(folder/'probes.jsonl',result)
    return result


def run(out,arm,gpu):
    folder=out/arm/'bridge';probes=build(out,arm)
    cp=None if arm=='base' else Path(read(out/arm/'checkpoint_manifest.json')['path'])
    config={'adapter':ev.checkpoint(cp) if cp else None,'model':str(MODELS['qwen3']),
        'probes_sha256':sha(folder/'probes.jsonl'),'source_predictions_sha256':sha(out/arm/'predictions.jsonl'),
        'self_facts_selection':'first preregistered new wording, index0; not best-of-three',
        'external_facts_in_self_condition':False,'oracle_in_gold_condition':True,
        'card_prompt':'exact first cards training question; capacity control, not heldout',
        'temperature':0,'seed':42,'card_max_tokens':384,'mcq_max_tokens':16,'script_sha256':sha(Path(__file__))}
    if (folder/'manifest.json').exists():assert read(folder/'manifest.json')['config']==config
    else:save(folder/'manifest.json',{'config':config,'created_at':ev.now()})
    raw=folder/'predictions.jsonl'
    records,notes=ev.read_rows(raw,allow_partial_tail=True) if raw.exists() else ([],[])
    if notes:ev.save_rows(raw,records)
    seen={r['id'] for r in records};pending=[p for p in probes if p['id'] not in seen]
    if pending:
        with reserve(WORK,gpu,str(folder),22000) as reservation:
            reservation.bind(os.getpid());os.environ.update(environment(gpu,DATA))
            import torch
            from swift.infer_engine import TransformersEngine,InferRequest,RequestConfig
            from swift.utils import seed_everything
            seed_everything(42)
            engine=TransformersEngine(str(MODELS['qwen3']),adapters=[str(cp)] if cp else None,
                torch_dtype=torch.bfloat16,device_map='cuda:0',max_batch_size=8,attn_impl='sdpa')
            engine.template.enable_thinking=False
            for fmt in ('card','mcq'):
                group=[p for p in pending if p['fmt']==fmt]
                for start in range(0,len(group),8):
                    batch=group[start:start+8]
                    outputs=engine.infer([InferRequest(messages=copy.deepcopy(p['messages']),images=p.get('images',[])) for p in batch],
                        RequestConfig(max_tokens=384 if fmt=='card' else 16,temperature=0,seed=42,return_details=True),use_tqdm=False)
                    for p,output in zip(batch,outputs):
                        if isinstance(output,Exception):raise output
                        c=output.choices[0];text=c.message.content or ''
                        r={k:p[k] for k in ('id','qid','condition','task','fmt','gold')}
                        r.update(response=text,correct=(text==p['gold']) if fmt=='card' else score(text,p['gold'],fmt),
                            predicted=ev.letter(text) if fmt=='mcq' else None,format_error=fmt=='mcq' and ev.letter(text) is None,
                            finish_reason=c.finish_reason,prompt_sha256=ev.digest({'messages':p['messages'],'images':p.get('images',[])}),
                            raw_response=dataclasses.asdict(output),saved_at=ev.now())
                        with raw.open('a') as f:f.write(json.dumps(r,ensure_ascii=False,default=str)+'\n');f.flush();os.fsync(f.fileno())
                        records.append(r)
                    save(folder/'status.json',{'status':'running','saved':len(records),'expected':len(probes),'pid':os.getpid(),'updated_at':ev.now()})
                    print(arm,fmt,start+len(batch),'/',len(group),flush=True)
    assert len(records)==len({r['id'] for r in records})==len(probes)
    metrics={}
    for condition in sorted({p['condition'] for p in probes}):
        group=[r for r in records if r['condition']==condition]
        metrics[condition]={'n':len(group),'correct':sum(r['correct'] for r in group),
            'format_errors':sum(r['format_error'] for r in group),'truncated':sum(r['finish_reason']=='length' for r in group)}
    save(folder/'report.json',{'status':'complete','arm':arm,'metrics':metrics,'full_card_metric':'literal identity, not semantic grading',
                             'finished_at':ev.now()})
    save(folder/'status.json',{'status':'complete','saved':len(records),'updated_at':ev.now()})


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--arm',choices=(*ARMS,'base'),required=True);ap.add_argument('--gpu',default='0')
    a=ap.parse_args();run(a.out.resolve(),a.arm,a.gpu)
