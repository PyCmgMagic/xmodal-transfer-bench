#!/usr/bin/env python3
"""Adaptive second development trial: separate rule execution and option readout.

The preceding JSON trial is retained. No trained weights or original data change.
"""
import argparse
import copy
import dataclasses
import json
import os
from pathlib import Path

import repair_v7_application_protocol as first
from run_v7_full_experiment import ROOT, rows, read, save, sha, environment
from run_v7_fullparam_ep10 import dense_identity
from v7_gpu_resources import reserve
import run_v7_evaluation as ev

OUT=ROOT/'runs/v7_application_protocol_repair_20261019/two_stage'
SYSTEM='''Apply the supplied rule to the test observations. Produce a concise, checkable worksheet; do not write a multiple-choice option letter.
Use sections OBSERVATIONS, RULE, CALCULATION, RESULT, and REPORT DECISION.
Keep every test candidate separate; a reference worked example is not the test case. Do not skip a candidate.
For a state machine, write the current state after EACH signal, and look up the next transition from that current state. For sequence transformations, preserve every object exactly once through each stage; list the row after each stage and check its length. For matching, enforce compatibility and one-to-one use globally. For scores, show each weighted count and apply the bias once. For proportions, apply both boundaries with the stated endpoint convention. For predicates/filters, evaluate each named condition independently, then combine using the stated AND/OR; never treat a different object type as an exact match.
When a technician report is supplied, compare every applicable rule parameter and every reported result, and state ACCEPT only if the complete submitted report is correct; otherwise state REPLACE and the corrected complete report. For direct questions, write NOT APPLICABLE for REPORT DECISION.
For image observations, read each labeled test group separately and count only the specified objects/cells. Use the actual pictured properties, not the reference worked example. Record uncertainty when details cannot be read.
Finish RESULT with the complete answer content requested by the question. No answer options are provided at this stage.'''


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'manifest.json').exists():validate();return
    first.validate()
    probes=[]
    for r in rows(first.OUT/'probes.jsonl'):
        if r['protocol']!='strict32':continue
        draft=copy.deepcopy(r);draft['id']='worksheet:'+r['input_mod']+':'+r['source_id']
        draft['protocol']='worksheet';draft['messages'][0]['content']=SYSTEM
        # Drop every answer option and original strict suffix. Keep reference,
        # requested task, submitted report and original observations unchanged.
        draft['messages'][-1]['content']=draft['messages'][-1]['content'].split('\n\nOptions:\n')[0]
        probes.append(draft)
    assert len(probes)==72
    ev.save_rows(OUT/'worksheet_probes.jsonl',probes)
    save(OUT/'decision_criteria.json',{'fixed_before_generation':True,'adaptive_development':True,
        'reason':'The first explicit JSON trial showed incorrect execution and extra schema errors. Separate solving without options from final option readout; retain first trial.',
        'gate':{'accuracy_each_modality':.8,'max_format_error_rate':.02,'required_arms':list(first.ARMS)},
        'limits':'Same 18 development rules; not independent confirmation. Correct materials remain oracle controls. No training unless application works.'})
    save(OUT/'manifest.json',{'created_at':ev.now(),'script_sha256':sha(Path(__file__)),
        'parent_manifest_sha256':sha(first.OUT/'manifest.json'),'models':read(first.OUT/'manifest.json')['models'],
        'files':{p.name:sha(p) for p in OUT.glob('*') if p.is_file()},
        'generation':{'worksheet':1024,'choice':32,'temperature':0,'thinking':False,'batch':4,'seed':42},
        'reference':'Correct original TEXT material in every condition; application control, not transfer.',
        'flow':'worksheet generated without options; own uncorrected worksheet then appended to original option question; same model both stages.'})


def validate():
    first.validate();m=read(OUT/'manifest.json')
    assert m['script_sha256']==sha(Path(__file__))
    assert m['parent_manifest_sha256']==sha(first.OUT/'manifest.json')
    for p,h in m['files'].items():assert sha(OUT/p)==h


def run(arm,gpu):
    validate();m=read(OUT/'manifest.json');folder=OUT/arm;folder.mkdir(exist_ok=True)
    with ev.run_lock(OUT/(arm+'.lock')):
        path=folder/'predictions.jsonl';rr,notes=ev.read_rows(path,allow_partial_tail=True) if path.exists() else ([],[])
        if notes:ev.save_rows(path,rr)
        seen={r['id'] for r in rr};assert len(seen)==len(rr)
        if len(rr)<144:
            save(folder/'state.json',{'status':'waiting_for_gpu','pid':os.getpid(),'saved':len(rr),'gpu':gpu})
            with reserve(ROOT/'runs/v7_full_experiment',gpu,str(folder)+':two-stage',24000) as res:
                res.bind(os.getpid());os.environ.update(environment(gpu,first.DATA));os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
                import torch
                from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
                from swift.utils import seed_everything
                torch.set_num_threads(4);seed_everything(42)
                assert dense_identity(Path(m['models'][arm]['path']))==m['models'][arm]
                engine=TransformersEngine(m['models'][arm]['path'],model_type='qwen3_vl',torch_dtype=torch.bfloat16,device_map='cuda:0',max_batch_size=4,attn_impl='sdpa')
                engine.template.enable_thinking=False
                def generate(probes,stage):
                    for mod in ('T','V'):
                        todo=[p for p in probes if p['id'] not in seen and p['input_mod']==mod]
                        for start in range(0,len(todo),4):
                            batch=todo[start:start+4]
                            outputs=engine.infer([InferRequest(messages=copy.deepcopy(p['messages']),images=p.get('images',[])) for p in batch],
                                RequestConfig(max_tokens=m['generation'][stage],temperature=0,seed=42,return_details=True),use_tqdm=False)
                            assert len(batch)==len(outputs)
                            for p,o in zip(batch,outputs):
                                if isinstance(o,Exception):raise o
                                c=o.choices[0];response=c.message.content or '';pred=ev.letter(response) if stage=='choice' else None
                                record={k:v for k,v in p.items() if k not in ('messages','images')}
                                full=dataclasses.asdict(o);full.pop('prompt_token_ids',None);full.pop('prompt_logprobs',None)
                                record.update(stage=stage,response=response,predicted=pred,correct=pred==p['gold'] if stage=='choice' else None,
                                    format_error=stage=='choice' and pred is None,finish_reason=c.finish_reason,raw_response=full,
                                    input_sha256=ev.digest({'messages':p['messages'],'images':p.get('images',[])}))
                                with path.open('a') as f:f.write(json.dumps(record,ensure_ascii=False,default=str)+'\n');f.flush();os.fsync(f.fileno())
                                rr.append(record);seen.add(p['id'])
                            save(folder/'state.json',{'status':'running','saved':len(rr),'expected':144,'gpu':gpu,'pid':os.getpid(),'updated_at':ev.now()})
                            print(arm,stage,mod,len(rr),'/',144,flush=True)
                generate(rows(OUT/'worksheet_probes.jsonl'),'worksheet')
                drafts={r['input_mod']+':'+r['source_id']:r for r in rr if r['stage']=='worksheet'}
                choices=[]
                for p in rows(first.OUT/'probes.jsonl'):
                    if p['protocol']!='strict32':continue
                    q=copy.deepcopy(p);q['id']='choice:'+p['input_mod']+':'+p['source_id'];q['protocol']='two_stage'
                    draft=drafts[p['input_mod']+':'+p['source_id']]
                    text=q['messages'][-1]['content'];assert text.rstrip().endswith(ev.SUFFIX)
                    q['messages'][-1]['content']=text.rstrip()[:-len(ev.SUFFIX)]+'\n\nYour own preceding worksheet (uncorrected; may contain errors):\n'+draft['response']+'\n\nMatch the required complete result and report decision to the CONTENT of the options. Candidate IDs are not option letters. Select the OPTION letter.'+ev.SUFFIX
                    q['worksheet_response_sha256']=ev.digest(draft['response']);choices.append(q)
                ev.save_rows(folder/'choice_probes.jsonl',choices)
                generate(choices,'choice')
        assert len(rr)==144
        save(folder/'state.json',{'status':'complete','saved':len(rr),'finished_at':ev.now()})


def summarize():
    report={};lines=['# 分步规则执行与选项映射：第二轮开发修复','','同18条预固定开发规则；先隐藏选项生成工作记录，再用同一模型未经修正的记录选择原题A/B/C/D。所有题均提供正确文字材料，不是闭卷迁移成绩。','',
        '| 模型 | 测试模态 | 正确 | direct | diagnosis | 格式错 | 工作记录截断 |','|---|---|---:|---:|---:|---:|---:|']
    for arm in first.ARMS:
        p=OUT/arm/'predictions.jsonl';rr=ev.read_rows(p,allow_partial_tail=True)[0] if p.exists() else []
        info={'state':read(OUT/arm/'state.json'),'metrics':{}}
        for mod in ('T','V'):
            ss=[r for r in rr if r['stage']=='choice' and r['input_mod']==mod]
            def metric(ss):
                n=len(ss);c=sum(r['correct'] is True for r in ss);valid=sum(not r['format_error'] for r in ss)
                return dict(n=n,correct=c,accuracy=c/n if n else None,valid_accuracy=c/valid if valid else None,
                    format_errors=n-valid,empty=sum(not r['response'].strip() for r in ss),truncated=sum(r['finish_reason']=='length' for r in ss))
            s=metric(ss)
            for key in ('task','relation','evaluation_group','qid'):
                s['by_'+key]={value:metric([r for r in ss if r[key]==value]) for value in sorted({r[key] for r in ss})}
            s['worksheet_truncated']=sum(r['stage']=='worksheet' and r['input_mod']==mod and r['finish_reason']=='length' for r in rr)
            info['metrics'][mod]=s
            by=s['by_task'];desc=lambda t:f'{by[t]["correct"]}/{by[t]["n"]}' if t in by else '-'
            lines.append(f'| {arm} | {mod} | {s["correct"]}/{s["n"]} | {desc("direct")} | {desc("diagnosis")} | {s["format_errors"]} | {s["worksheet_truncated"]} |')
        info['complete']=len(rr)==144
        info['gate_pass']=info['complete'] and all(s['accuracy']>=.8 and s['format_errors']/36<=.02 for s in info['metrics'].values())
        report[arm]=info
    save(OUT/'summary.json',report)
    lines+=['','全部完成：'+str(all(v['complete'] for v in report.values()))+'；全部通过预先固定应用门槛：'+str(all(v['gate_pass'] for v in report.values()))+'。',
        '','这是一项在首轮JSON流程出现问题后追加的开发实验；不是独立确认。最终选择题格式错误照常计错，工作记录不被人工纠正；不筛成功规则，也不从记录恢复答案。']
    (OUT/'report_zh.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=('prepare','run','summarize'),required=True)
    ap.add_argument('--arm',choices=first.ARMS);ap.add_argument('--gpu',default='0');a=ap.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='summarize':summarize()
    else:
        assert a.arm
        run(a.arm,a.gpu)
