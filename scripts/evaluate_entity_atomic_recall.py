#!/usr/bin/env python3
"""Original-adapter ms-swift diagnostics: recall and independent instruction tasks."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import sys

from answer_readout import normalized,options_from_question
from answer_readout_entity_v3 import parse_answer
from entityqa_common import rows,save,write_rows,now


def token_f1(answer,target):
    a,b=Counter(normalized(answer).split()),Counter(normalized(target).split())
    common=sum((a&b).values())
    return 2*common/(sum(a.values())+sum(b.values())) if a or b else 1.0


def run(args):
    import torch
    from swift import TransformersEngine,InferRequest,RequestConfig
    torch.set_num_threads(4)
    work=args.work.resolve()
    tasks=rows(work/'data/recall_dev.jsonl')
    for r in rows(work/'data/retention_dev.jsonl')[:48]:
        tasks.append(dict(id=r['id'],kind='independent_instruction',messages=[r['messages'][0]],expected=r['answer']))
    output=work/'diagnostics'/args.arm
    output.mkdir(parents=True,exist_ok=True)
    existing=rows(output/'predictions.jsonl') if (output/'predictions.jsonl').exists() else []
    seen={r['id'] for r in existing}
    pending=[r for r in tasks if r['id'] not in seen]
    if pending:
        adapter=None
        if args.arm!='base':
            folders=list((work/args.arm/'train/TEXT').glob('checkpoint-*'))
            adapter=max(folders,key=lambda p:int(p.name.split('-')[-1]))
        engine=TransformersEngine(str(args.model.resolve()),model_type='qwen3_vl',
            adapters=[str(adapter)] if adapter else None,torch_dtype=torch.bfloat16,
            attn_impl='sdpa',device_map='cuda:0',max_batch_size=8)
        engine.template.enable_thinking=False
        for start in range(0,len(pending),8):
            batch=pending[start:start+8]
            answers=engine.infer([InferRequest(messages=r['messages']) for r in batch],
                RequestConfig(max_tokens=256,temperature=0),use_tqdm=False)
            for r,out in zip(batch,answers):
                if isinstance(out,Exception):raise out
                text=out.choices[0].message.content
                if r['kind']=='independent_instruction':
                    pred,reason=parse_answer(text,options_from_question(r['messages'][0]['content']))
                    scores=dict(predicted=pred,parse_reason=reason,correct=pred==r['expected'])
                else:
                    expected=normalized(r['expected'])
                    body=' '+normalized(text)+' '
                    scores=dict(target_span_hit=(' '+expected+' ') in body,
                        token_f1=token_f1(text,r['expected']),
                        all_required_slots_hit=all((' '+normalized(s)+' ') in body for s in r.get('required_slots',[]))
                            if r.get('required_slots') else None)
                result=dict(r,response=text,finish_reason=out.choices[0].finish_reason,
                    arm=args.arm,engine='ms-swift TransformersEngine',original_unmerged_adapter=True,
                    diagnostic_max_tokens=256,**scores)
                with (output/'predictions.jsonl').open('a') as stream:
                    stream.write(json.dumps(result,ensure_ascii=False)+'\n')
                existing.append(result)
            save(output/'status.json',dict(stage='running',saved=len(existing),total=len(tasks),updated_at=now()))
    groups={}
    for kind in ['trained_attribute_recall','neutral_claim_recall','independent_instruction']:
        group=[r for r in existing if r['kind']==kind]
        g=dict(n=len(group),length_limited=sum(r['finish_reason']=='length' for r in group))
        if kind=='independent_instruction':
            g.update(accuracy=sum(r['correct'] for r in group)/len(group),
                     invalid_rate=sum(r['predicted'] is None for r in group)/len(group),
                     boxed_rate=sum('\\boxed{' in r['response'] for r in group)/len(group))
        else:
            g.update(strict_target_span_hit_rate=sum(r['target_span_hit'] for r in group)/len(group),
                     mean_token_f1=sum(r['token_f1'] for r in group)/len(group))
            slots=[r for r in group if r['all_required_slots_hit'] is not None]
            if slots:g['literal_required_slots_hit_rate']=sum(r['all_required_slots_hit'] for r in slots)/len(slots)
        groups[kind]=g
    assert len(existing)==len(tasks)
    save(output/'summary.json',dict(status='complete',arm=args.arm,groups=groups,updated_at=now(),
        interpretation='Recall metrics are literal lexical diagnostics, not semantic knowledge accuracy. Trained attribute questions are memorization probes. Diagnostic cap 256 is separate from formal MCQ cap 65536.'))
    save(output/'status.json',dict(stage='complete',saved=len(existing),total=len(tasks),updated_at=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--arm',choices=['base','atomic','atomic_replay'],required=True)
    run(p.parse_args())
