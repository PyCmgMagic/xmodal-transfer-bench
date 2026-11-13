#!/usr/bin/env python3
"""Paired, resumable rule recall/application diagnostic for the epoch-5 adapters."""
import argparse
import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict

import run_v7_evaluation as ev
from build_v7_experiment_data import name as canonical_rule_name
from v7_gpu_resources import reserve

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'runs/qwen3_lora_cot_eval_20261025'
DEFAULT_WORK = ROOT/'runs/qwen3_lora_recall_apply_v2_20261026'
RECALL_SYSTEM = ('Recall the rule from your previously learned knowledge. State its actual conditions, '
                 'thresholds, mappings and recording codes explicitly. Do not invent missing details. '
                 'No learning image or reference material is available. If a detail cannot be recalled, say so.')
ANSWER_SYSTEM = ('Solve the multiple-choice question. Reason step by step in at most 200 words. '
                 'End with the final option letter in a single LaTeX box, for example \\boxed{A}. '
                 'Give an option even if uncertain. Do not confuse candidate IDs with answer option letters.')
ANSWER_SUFFIX = ('\n\nReason step by step in at most 200 words. Finish with one final option '
                 'in \\boxed{A}, \\boxed{B}, \\boxed{C}, or \\boxed{D}.')
BOX = re.compile(r'\\boxed\s*\{\s*([A-Da-d])\s*\}')
FINAL = re.compile(
    r'\bfinal(?:\s+(?:answer|choice|option|selection|decision))?\s*[:：=]\s*[*_`]*([A-Da-d])\b'
    r'(?!\s*(?:[,;/&]|\band\b|\bor\b)\s*[*_`]*[A-Da-d]\b)', re.I)
NATURAL = re.compile(
    r'(?:^|[.!?;:\n]\s*)(?:therefore,?\s+|thus,?\s+|so,?\s+)?'
    r'(?:(?:the\s+)?(?:final\s+)?(?:correct\s+)?(?:answer|choice|option|selection|output|result|decision)'
    r'\s*(?:is|[:：=])\s*(?:option\s+)?|(?:I\s+)?(?:choose|select|return)\s+(?:option\s+)?)'
    r'[*_`]*([A-Da-d])[*_`]*\s*[.!?]*\s*$', re.I)
ENDING = re.compile(r'[\s.!?*_`$\\()\[\]]*\Z')
OPTION_MATCH = re.compile(
    r'(?:^|[.!?;:\n]\s*)(?:(?:this|that|it)\s+(?:matches|corresponds\s+to)|'
    r'(?:the\s+)?(?:correct\s+)?(?:answer|choice|selection)\s+is)'
    r'\s+option\s+[*_`]*([A-Da-d])[*_`]*\s*[.!?]*\s*$', re.I)
OPTION_CORRECT = re.compile(
    r'(?:^|[.!?;:\n]\s*)(?:therefore,?\s+)?(?:only\s+)?(?:option\s+)?'
    r'[*_`]*([A-Da-d])[*_`]*\s+is\s+(?:the\s+)?correct\s+(?:answer|choice|option)\s*[.!?]*\s*$', re.I)


from answer_readout import parse_answer


def recall_payload(name):
    return {'messages': [{'role':'system', 'content':RECALL_SYSTEM},
                         {'role':'user', 'content':f'Explain the rule or codebook named {name}, including its conditions, mappings, and how to apply it. State all specific facts you remember.'}]}


def application_payload(row, recalled=None):
    payload = {'messages': [{'role':'system','content':ANSWER_SYSTEM}]}
    question = row['question']
    if recalled is not None:
        question = ('Your earlier recall of this rule is provided below. It may be incomplete or mistaken. '
                    'Use it as your recalled knowledge when solving the question.\n'
                    '<self_recalled_rule>\n'+recalled+'\n</self_recalled_rule>\n\n'+question)
    payload['messages'].append({'role':'user','content':question+ANSWER_SUFFIX})
    if row.get('images'):
        payload['images'] = list(row['images'])
    return payload


def read(path):
    return ev.read_rows(path)[0]


def prepare(args):
    work=args.work; path=work/'manifest.json'
    if path.exists():
        manifest=json.loads(path.read_text())
        assert manifest['rules_per_family']==args.rules_per_family
        verify(manifest, work)
        return manifest
    old=json.loads((OLD/'manifest.json').read_text())
    # Read names and family tags only. No correct material enters model input.
    knowledge_path=ROOT/'data/rules_repaired_single_image_v7/audit/rule_knowledge.jsonl'
    knowledge={r['content']['id']:r['content'] for r in read(knowledge_path)}
    audit=read(OLD/'data/rules/sample_audit.jsonl')
    groups=defaultdict(set)
    for r in audit:
        groups[r['entity_kind']].add(r['qid'])
    selection=[]
    for family, ids in sorted(groups.items()):
        chosen=sorted(ids,key=lambda q:hashlib.sha256(('recall-apply-20261025:42:'+q).encode()).hexdigest())[:args.rules_per_family]
        assert len(chosen)==args.rules_per_family
        selection.extend({'qid':q,'name':canonical_rule_name(knowledge[q]),'family':family} for q in chosen)
    # Canonical names must exactly agree with the prompts used in the actual
    # frozen training dataset. No target material is sent to the model here.
    trained=read(ROOT/'runs/qwen3_lora_learning_ep5_20261024/data/rules_sft_text.jsonl')
    original_train=read(ROOT/'data/rules_repaired_single_image_v7/shared/train_text.jsonl')
    assert len(original_train)==len(trained)==1000
    prompt_names={original['id']:r['messages'][0]['content'] for original,r in zip(original_train,trained)}
    for r in selection:
        expected=f'Explain the rule or codebook named {r["name"]}, including its conditions, mappings, and how to apply it.'
        assert expected == prompt_names[r['qid']], (r['qid'],r['name'])
    selected={r['qid'] for r in selection}
    inputs={}; frozen={}; sources={str(knowledge_path):ev.file_hash(knowledge_path),
                                  str(OLD/'manifest.json'):ev.file_hash(OLD/'manifest.json')}
    for direction in ('T2T','T2V','V2T','V2V'):
        source=OLD/'data/rules'/f'{direction}.jsonl'
        sources[str(source)]=ev.file_hash(source)
        rows=[]
        for r in read(source):
            if r['qid'] not in selected:continue
            text=r['messages'][-1]['content']
            assert text.endswith(old['suffix'])
            row={'id':r['id'],'qid':r['qid'],'question':text[:-len(old['suffix'])].rstrip()}
            if r.get('images'): row['images']=r['images']
            rows.append(row)
        assert len(rows)==2*len(selection)
        assert Counter(r['qid'] for r in rows)=={q:2 for q in selected}
        inputs[direction]=rows
        ev.save_rows(work/'data'/f'{direction}.jsonl',rows)
    assert inputs['T2T']==inputs['V2T'] and inputs['T2V']==inputs['V2V']
    ev.save_rows(work/'data/recall.jsonl',[{**s,**recall_payload(s['name'])} for s in selection])
    source=OLD/'data/rules/eval_labels.jsonl'
    sources[str(source)]=ev.file_hash(source)
    ev.save_rows(work/'data/eval_labels.jsonl',[r for r in read(source) if r['qid'] in selected])
    jobs=[j for j in old['jobs'] if j['dataset']=='rules']
    images={p:ev.file_hash(p) for row in inputs['T2V'] for p in row.get('images',[])}
    ev.save_json(work/'data/image_hashes.json',images)
    for j in jobs:
        if j['adapter']:
            for name,sha in j['adapter']['sha256'].items():
                sources[str(Path(j['adapter']['path'])/name)]=sha
    for name in (Path(__file__).name,'run_v7_evaluation.py','v7_gpu_resources.py','build_v7_experiment_data.py','answer_readout.py'):
        src=ROOT/'scripts'/name;dst=work/'source_snapshots'/name
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        sources[str(src)]=ev.file_hash(src)
    for p in list((work/'data').glob('*'))+list((work/'source_snapshots').glob('*')):
        frozen[str(p.relative_to(work))]=ev.file_hash(p)
    manifest={'created_at':ev.now(),'model':old['model'],'jobs':jobs,
              'rules_per_family':args.rules_per_family,'selection':selection,'rules':len(selection),'revision':'v2 canonical training rule names + tolerant explicit terminal readout',
              'questions_per_cell':len(selection)*2,'predictions_expected':len(jobs)*len(selection)*9,
              'selection_policy':'SHA256(recall-apply-20261025:42:+qid), first N per family; independent of outcomes',
              'arms':['direct_boxed','self_recall_boxed'],
              'generation':{'batch_size':args.batch_size,'temperature':0,'seed':42,
                            'recall_max_tokens':1024,'answer_max_tokens':1536,'max_pixels':1003520},
              'recall_system':RECALL_SYSTEM,'answer_system':ANSWER_SYSTEM,'answer_suffix':ANSWER_SUFFIX,
              'sources':sources,'frozen_files':frozen,'resource_work':str(OLD),
              'protocol':'Fresh name-only recall per model/rule, reused across both paired test modalities and questions. '
                         'No correct rule, learning image, question, options, gold or retrieval at recall. '
                         'Both answer arms use identical boxed/CoT/generation/scoring. '
                         'Invalid answers remain wrong in full denominator; no gold-aware retries.',
              'scope':'Balanced 9-family rule diagnostic, not the full 1000-rule benchmark; no retraining.'}
    manifest['sources'][str(ROOT/'runs/qwen3_lora_learning_ep5_20261024/data/rules_sft_text.jsonl')]=ev.file_hash(
        ROOT/'runs/qwen3_lora_learning_ep5_20261024/data/rules_sft_text.jsonl')
    ev.save_json(path,manifest);verify(manifest,work)
    ev.save_json(work/'status.json',{'status':'prepared','predictions_expected':manifest['predictions_expected'],'updated_at':ev.now()})
    return manifest


def verify(manifest, work):
    for path,sha in manifest['sources'].items():
        assert ev.file_hash(path)==sha,path
    for rel,sha in manifest['frozen_files'].items():
        assert ev.file_hash(work/rel)==sha,rel
    for path,sha in json.loads((work/'data/image_hashes.json').read_text()).items():
        assert ev.file_hash(path)==sha,path


def collect(manifest, work):
    labels={r['id']:r for r in read(work/'data/eval_labels.jsonl')}
    expected=manifest['questions_per_cell']; reports=[]; requests=0; recalls=[]
    for job in manifest['jobs']:
        out=work/'jobs'/job['name'];raw=out/'raw.jsonl'
        if not raw.exists():continue
        records=ev.read_rows(raw,allow_partial_tail=True)[0];requests+=len(records)
        recalled={r['id']:r for r in records if r['stage']=='recall'}
        if recalled:
            recalls.append({'job':job['name'],'saved':len(recalled),'total':manifest['rules'],
                            'empty':sum(not r['response'].strip() for r in recalled.values()),
                            'truncated':sum(r['finish_reason']=='length' for r in recalled.values()),
                            'semantic_recall_correctness':'not automatically certified'})
        for direction in job['directions']:
            grouped={arm:{r['id']:r for r in records if r['stage']=='application' and
                         r['direction']==direction and r['arm']==arm} for arm in manifest['arms']}
            if not all(len(v)==expected for v in grouped.values()):continue
            assert set(grouped['direct_boxed'])==set(grouped['self_recall_boxed'])==set(labels)
            row={'job':job['name'],'objective':job['objective'],'learning_modality':job['learning_modality'],
                 'direction':direction,'total':expected}
            scored={}; audit_rows=[]
            for arm,records_by_id in grouped.items():
                scored[arm]={}
                valid=correct=truncated=0
                for sid,r in records_by_id.items():
                    pred,method=parse_answer(r['response']); gold=labels[sid]['answer']
                    right=pred==gold;valid+=pred is not None;correct+=right;truncated+=r['finish_reason']=='length'
                    scored[arm][sid]=right
                    audit_rows.append({'id':sid,'qid':r['qid'],'arm':arm,'predicted':pred,'gold':gold,
                                       'correct':right,'parse_method':method,'finish_reason':r['finish_reason']})
                row[arm]={'correct':correct,'accuracy':correct/expected,'invalid':expected-valid,'truncated':truncated}
            transitions=Counter((scored['direct_boxed'][sid],scored['self_recall_boxed'][sid]) for sid in labels)
            row['delta_pp']=100*(row['self_recall_boxed']['accuracy']-row['direct_boxed']['accuracy'])
            row['paired']={'fixed':transitions[False,True],'broken':transitions[True,False],
                           'both_correct':transitions[True,True],'both_wrong':transitions[False,False]}
            ev.save_rows(out/f'{direction}_results.jsonl',audit_rows)
            ev.save_json(out/f'{direction}_report.json',row);reports.append(row)
    summary={'updated_at':ev.now(),'predictions_saved':requests,'predictions_expected':manifest['predictions_expected'],
             'cells_complete':len(reports),'cells_expected':2*len(manifest['jobs']),
             'reports':reports,'recall_status':recalls}
    ev.save_json(work/'summary.json',summary)
    lines=['# 先回忆再答题：配对诊断','',
           f"9类规则，每类{manifest['rules_per_family']}条，共{manifest['rules']}条规则；每个方向{expected}题。",
           '回忆阶段只提供规则名，不给测试题、学习图片或正确规则。两种答题条件均为CoT + boxed，同模型同题配对。',
           '固定均衡抽样，不是全量准确率；回忆正确性需独立检查。无效答案计错，未完成方向不报告准确率。','',
           '| 模型 | 方向 | 直接作答 | 先回忆再答 | 差值 | 错→对 | 对→错 | 无效：直接/回忆 |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in reports:
        d=r['direct_boxed'];a=r['self_recall_boxed'];p=r['paired']
        lines.append(f"| {r['job']} | {r['direction']} | {d['accuracy']:.2%} | {a['accuracy']:.2%} | {r['delta_pp']:+.2f} pp | {p['fixed']} | {p['broken']} | {d['invalid']}/{a['invalid']} |")
    lines+=['',f"当前已保存 {requests}/{manifest['predictions_expected']} 条生成；完成 {len(reports)}/{2*len(manifest['jobs'])} 个配对方向。",
            'raw.jsonl保留全部回忆及答题输出、token、结束原因、prompt/adapter/recall哈希。没有按回忆质量过滤样本。',
            '基座无学习模态，其T2T/T2V仅表示文字/视觉测试。']
    (work/'report_zh.md').write_text('\n'.join(lines)+'\n')
    return summary


def worker(args, manifest, job):
    import torch
    from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
    from swift.template import get_template
    from swift.utils import seed_everything
    class Engine(TransformersEngine):
        def _get_template(self,processor,template_type=None):
            return get_template(processor,template_type=template_type,use_chat_template=True,
                                enable_thinking=False,max_length=8192,max_pixels=1003520)
    out=args.work/'jobs'/job['name'];out.mkdir(parents=True,exist_ok=True)
    with ev.run_lock(out/'.lock'):
        ev.save_json(out/'state.json',{'status':'loading_model','pid':os.getpid(),'updated_at':ev.now()})
        seed_everything(42)
        engine=Engine(manifest['model'],adapters=[job['adapter']['path']] if job['adapter'] else [],
                      torch_dtype=torch.bfloat16,device_map='cuda:0',max_batch_size=args.batch_size,attn_impl='sdpa')
        assert engine.template.use_chat_template and not engine.template.enable_thinking
        ev.save_json(out/'runtime.json',{'adapter':job['adapter'],'use_chat_template':True,
                     'native_thinking':False,'explicit_cot':True,'template_type':engine.template.template_meta.template_type})
        raw=out/'raw.jsonl'; saved=ev.read_rows(raw,allow_partial_tail=True)[0] if raw.exists() else []
        if raw.exists() and raw.read_bytes() and not raw.read_bytes().endswith(b'\n'):ev.save_rows(raw,saved)
        def key(r):return r['stage'],r.get('direction',''),r.get('arm',''),r['id']
        seen={key(r):r for r in saved};assert len(seen)==len(saved)
        for r in saved: assert r['adapter']==job['adapter']
        with raw.open('a') as f:
            def generate(tasks,limit):
                pending=[]
                for meta,payload in tasks:
                    k=key(meta)
                    if k in seen:
                        assert seen[k]['prompt_sha256']==ev.digest(payload)
                    else:pending.append((meta,payload))
                for start in range(0,len(pending),args.batch_size):
                    batch=pending[start:start+args.batch_size]
                    results=engine.infer([InferRequest(**copy.deepcopy(p)) for _,p in batch],
                              RequestConfig(max_tokens=limit,temperature=0,seed=42,return_details=True),use_tqdm=False)
                    assert len(results)==len(batch)
                    for (meta,payload),res in zip(batch,results):
                        if isinstance(res,Exception):raise res
                        c=res.choices[0]
                        record={**meta,'response':c.message.content or '',
                                'reasoning_content':getattr(c.message,'reasoning_content',None),
                                'finish_reason':c.finish_reason,'usage':dataclasses.asdict(res.usage),
                                'completion_token_ids':c.token_ids,'prompt_token_ids':res.prompt_token_ids,
                                'prompt_sha256':ev.digest(payload),'adapter':job['adapter'],'saved_at':ev.now()}
                        f.write(json.dumps(record,ensure_ascii=False)+'\n');seen[key(meta)]=record
                    f.flush();os.fsync(f.fileno())
                    ev.save_json(out/'state.json',{'status':'running','pid':os.getpid(),'saved':len(seen),
                                  'stage':batch[-1][0],'updated_at':ev.now()})
                    print(job['name'],'saved',len(seen),'last',batch[-1][0],flush=True)
            recall=read(args.work/'data/recall.jsonl')
            generate([({'stage':'recall','id':r['qid'],'qid':r['qid']},
                       {'messages':r['messages']}) for r in recall],manifest['generation']['recall_max_tokens'])
            recalled={qid:seen['recall','','',qid]['response'] for qid in (r['qid'] for r in recall)}
            for direction in job['directions']:
                for arm in manifest['arms']:
                    tasks=[]
                    for row in read(args.work/'data'/f'{direction}.jsonl'):
                        material=recalled[row['qid']] if arm=='self_recall_boxed' else None
                        meta={'stage':'application','direction':direction,'arm':arm,'id':row['id'],'qid':row['qid']}
                        if material is not None:meta['recall_sha256']=ev.digest(material)
                        tasks.append((meta,application_payload(row,material)))
                    generate(tasks,manifest['generation']['answer_max_tokens'])
                collect(manifest,args.work)
        ev.save_json(out/'state.json',{'status':'complete','saved':len(seen),'finished_at':ev.now()})


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work',type=Path,default=DEFAULT_WORK)
    ap.add_argument('--rules-per-family',type=int,default=10)
    ap.add_argument('--batch-size',type=int,default=4)
    ap.add_argument('--prepare-only',action='store_true')
    ap.add_argument('--worker')
    args=ap.parse_args();args.work=args.work.resolve();args.work.mkdir(parents=True,exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',MAX_PIXELS='1003520',
                      OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
    manifest=prepare(args)
    assert args.batch_size==manifest['generation']['batch_size']
    if args.prepare_only:
        print(json.dumps({'rules':manifest['rules'],'questions_per_cell':manifest['questions_per_cell'],
                          'predictions_expected':manifest['predictions_expected']}));return
    if args.worker:
        worker(args,manifest,next(j for j in manifest['jobs'] if j['name']==args.worker));return
    with ev.run_lock(args.work/'.coordinator.lock'):
        try:
            for job in manifest['jobs']:
                out=args.work/'jobs'/job['name'];out.mkdir(parents=True,exist_ok=True)
                state=out/'state.json'
                if state.exists() and json.loads(state.read_text()).get('status')=='complete':continue
                ev.save_json(args.work/'status.json',{'status':'waiting_for_gpu','pid':os.getpid(),
                              'next_job':job['name'],'resource_work':manifest['resource_work'],'updated_at':ev.now()})
                with reserve(manifest['resource_work'],'0','recall_apply:'+job['name'],18000,poll=2) as reservation:
                    cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--work',str(args.work),
                         '--rules-per-family',str(args.rules_per_family),'--batch-size',str(args.batch_size),'--worker',job['name']]
                    ev.save_json(out/'command.json',cmd)
                    with (out/'worker.log').open('a') as log:
                        child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
                        reservation.bind(child.pid)
                        ev.save_json(args.work/'status.json',{'status':'running','pid':os.getpid(),
                                      'job':job['name'],'worker_pid':child.pid,'updated_at':ev.now()})
                        assert child.wait()==0,f'Worker failed; see {out}/worker.log'
                collect(manifest,args.work)
            verify(manifest,args.work)
            summary=collect(manifest,args.work)
            assert summary['cells_complete']==10 and summary['predictions_saved']==manifest['predictions_expected']
            ev.save_json(args.work/'status.json',{'status':'complete','predictions':summary['predictions_saved'],
                          'cells':summary['cells_complete'],'finished_at':ev.now()})
        except BaseException as exc:
            ev.save_json(args.work/'status.json',{'status':'failed','error':str(exc),
                          'traceback':traceback.format_exc(),'updated_at':ev.now()})
            raise


if __name__=='__main__':main()
