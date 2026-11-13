#!/usr/bin/env python3
"""Rescore existing outputs and finalize unresolved answers with the same model."""
import argparse
import copy
import dataclasses
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from collections import defaultdict

import run_v7_evaluation as ev
from answer_readout import parse_answer, options_from_question, finalizer_payload, choice_hook
from v7_gpu_resources import reserve, can_admit

ROOT=Path(__file__).resolve().parents[1]
FULL=ROOT/'runs/qwen3_lora_cot_eval_20261025'
RECALL=ROOT/'runs/qwen3_lora_recall_apply_v2_20261026'
WORK=ROOT/'runs/answer_output_repair_20261026'


def read(path):return ev.read_rows(path)[0]


def profiles():
    manifest=json.loads((FULL/'manifest.json').read_text())
    return {j['name']:{'model':manifest['model'],'adapter':j['adapter']} for j in manifest['jobs']}


def cells():
    """Only completed, fully aligned cells are exposed for scoring."""
    for rep in sorted(FULL.glob('jobs/*/*/report.json')):
        report=json.loads(rep.read_text());job=rep.parent.parent.name;direction=rep.parent.name
        inputs={x['id']:x for x in read(FULL/'data'/report['dataset']/f'{direction}.jsonl')}
        labels={x['id']:x for x in read(FULL/'data'/report['dataset']/'eval_labels.jsonl')}
        raw=read(rep.parent/'raw.jsonl');original={x['id']:x['predicted'] for x in read(rep.parent/'results.jsonl')}
        assert {x['id'] for x in raw}==set(inputs)==set(labels)==set(original)
        yield {'experiment':'full_cot','job':job,'dataset':report['dataset'],'direction':direction,'arm':'direct',
               'source':str(rep.parent/'raw.jsonl'),'source_report':str(rep)},[
                  (x,{k:inputs[x['id']][k] for k in ('messages','images') if k in inputs[x['id']]},
                   labels[x['id']]['answer'],original[x['id']]) for x in raw]
    if not (RECALL/'manifest.json').exists():return
    from run_lora_recall_apply_v2 import application_payload
    manifest=json.loads((RECALL/'manifest.json').read_text())
    labels={x['id']:x for x in read(RECALL/'data/eval_labels.jsonl')}
    for job in manifest['jobs']:
        path=RECALL/'jobs'/job['name']/'raw.jsonl'
        if not path.exists():continue
        rows=ev.read_rows(path,allow_partial_tail=True)[0];rec={x['qid']:x['response'] for x in rows if x['stage']=='recall'}
        for direction in job['directions']:
            original_file=path.parent/f'{direction}_results.jsonl'
            if not original_file.exists():continue
            original={(x['arm'],x['id']):x['predicted'] for x in read(original_file)}
            inputs={x['id']:x for x in read(RECALL/'data'/f'{direction}.jsonl')}
            groups={arm:[x for x in rows if x.get('direction')==direction and x.get('arm')==arm] for arm in manifest['arms']}
            if not all(len(group)==manifest['questions_per_cell'] for group in groups.values()):continue
            for arm,group in groups.items():
                assert {x['id'] for x in group}==set(inputs)==set(labels)
                data=[]
                for x in group:
                    own=rec[x['qid']] if arm=='self_recall_boxed' else None
                    payload=application_payload(inputs[x['id']],own)
                    assert x['prompt_sha256']==ev.digest(payload)
                    data.append((x,payload,labels[x['id']]['answer'],original[arm,x['id']]))
                yield {'experiment':'recall_v2','job':job['name'],'dataset':'rules','direction':direction,
                       'arm':arm,'source':str(path),'source_report':str(original_file)},data


def refresh(work):
    repaired={};pending=defaultdict(list);reports=[];all_results=[]
    for path in (work/'jobs').glob('*/readout.jsonl'):
        for r in ev.read_rows(path,allow_partial_tail=True)[0]:
            assert r['key'] not in repaired
            repaired[r['key']]=r
    for meta,data in cells():
        counts={k:0 for k in ['total','original_invalid','original_correct','parsed_invalid',
                              'parsed_correct','recovered','recovered_correct','finalized',
                              'combined_correct','combined_invalid']};results=[]
        for raw,payload,gold,original in data:
            sid=raw['id'];key='/'.join([meta['experiment'],meta['job'],meta['direction'],meta['arm'],sid])
            options=options_from_question(payload['messages'][-1]['content'])
            selected,method=parse_answer(raw['response'],options)
            final=selected;stage='parsed'
            counts['total']+=1;counts['original_invalid']+=original is None;counts['original_correct']+=original==gold
            counts['parsed_invalid']+=selected is None;counts['parsed_correct']+=selected==gold
            counts['recovered']+=original is None and selected is not None
            counts['recovered_correct']+=original is None and selected==gold
            if selected is None:
                readout=finalizer_payload(payload,raw['response']);sha=ev.digest(readout)
                if key in repaired:
                    r=repaired[key];assert r['prompt_sha256']==sha and r['source_response_sha256']==ev.digest(raw['response'])
                    assert r['profile']==meta['job'] and r['selected'] in set('ABCD')
                    final=r['selected'];stage='constrained_readout';counts['finalized']+=1
                else:
                    stage='pending_readout'
                    # Gold is deliberately excluded from the inference task.
                    pending[meta['job']].append({'key':key,'id':sid,'qid':raw['qid'],'profile':meta['job'],
                            'payload':readout,'prompt_sha256':sha,'source_response_sha256':ev.digest(raw['response'])})
            counts['combined_correct']+=final==gold;counts['combined_invalid']+=final is None
            row={'key':key,'id':sid,'qid':raw['qid'],'gold':gold,'original_predicted':original,
                 'parsed_predicted':selected,'parse_method':method,'final_predicted':final,'stage':stage,
                 'original_correct':original==gold,'parsed_correct':selected==gold,'combined_correct':final==gold,
                 'source_finish_reason':raw['finish_reason'],'source_response_sha256':ev.digest(raw['response'])}
            results.append(row)
        n=counts['total'];report={**meta,**counts,'original_accuracy':counts['original_correct']/n,
                  'parsed_accuracy':counts['parsed_correct']/n,
                  'combined_accuracy':counts['combined_correct']/n if counts['combined_invalid']==0 else None,
                  'readout_pending':counts['parsed_invalid']-counts['finalized']}
        target=work/'results'/meta['experiment']/meta['job']/meta['direction']/meta['arm']
        ev.save_rows(target/'results.jsonl',results);ev.save_json(target/'report.json',report)
        reports.append(report);all_results.extend(results)
    totals={k:sum(r.get(k,0) for r in reports) for k in ['total','original_invalid','parsed_invalid','recovered',
            'recovered_correct','original_correct','parsed_correct','finalized','readout_pending']}
    summary={'updated_at':ev.now(),'totals':totals,'cells':reports,
             'note':'Parsing-only scores reuse original generations. Combined scores add one constrained A/B/C/D selection by the same model for unresolved rows. No gold-aware retry. Do not compare these protocols as identical inference.'}
    ev.save_json(work/'summary.json',summary)
    lines=['# 答案输出修复','',
           '解析修复只读取原始输出；约束读取使用同一模型、原题/原测试图片及自身原始回答，额外生成一个A/B/C/D选项。',
           '不提供gold、正确规则、学习图片或外部答案。新增推理与纯解析修复分别报告。组合分数只有该单元全部读取完才公布。','',
           '| 实验 | 模型 | 方向/条件 | 原准确率 | 解析修复后 | 原无效 | 修复后无效 | 约束读取完成 | 约束后准确率 |',
           '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in reports:
        final=f"{r['combined_accuracy']:.2%}" if r['combined_accuracy'] is not None else '待完成'
        lines.append(f"| {r['experiment']} | {r['job']} | {r['direction']}/{r['arm']} | {r['original_accuracy']:.2%} | {r['parsed_accuracy']:.2%} | {r['original_invalid']} | {r['parsed_invalid']} | {r['finalized']} | {final} |")
    lines+=['',f"当前审计{totals['total']}条回答，解析恢复{totals['recovered']}条，其中{totals['recovered_correct']}题答对；仍有{totals['readout_pending']}条等待同模型约束读取。",
            '原始实验未完成单元暂不纳入。旧回忆v1名称错误的实验不进入本修复汇总。']
    (work/'report_zh.md').write_text('\n'.join(lines)+'\n')
    return summary,pending


def worker(args):
    import torch
    from swift.infer_engine import TransformersEngine,InferRequest,RequestConfig
    from swift.template import get_template
    from swift.utils import seed_everything
    manifest=json.loads((args.work/'manifest.json').read_text());profile=manifest['profiles'][args.worker]
    class Engine(TransformersEngine):
        def _get_template(self,processor,template_type=None):
            return get_template(processor,template_type=template_type,use_chat_template=True,
                                enable_thinking=False,max_length=8192,max_pixels=1003520)
        def _infer(self, *args, **kwargs):
            # swift 4.5.3's public synchronous infer() does not expose the hook;
            # inject it at the internal batch entry point used by that method.
            kwargs['pre_infer_hook']=choice_hook(self.tokenizer)
            return super()._infer(*args, **kwargs)
    seed_everything(42);torch.set_num_threads(4)
    engine=Engine(profile['model'],adapters=[profile['adapter']['path']] if profile['adapter'] else [],
                  torch_dtype=torch.bfloat16,device_map='cuda:0',max_batch_size=4,attn_impl='sdpa')
    folder=args.work/'jobs'/args.worker;folder.mkdir(parents=True,exist_ok=True)
    ev.save_json(folder/'runtime.json',{'profile':profile,'use_chat_template':engine.template.use_chat_template,
                 'native_thinking':engine.template.enable_thinking,'constraint':'prefix_allowed_tokens_fn: A/B/C/D; max_tokens=1'})
    tasks=read(args.tasks)
    raw=folder/'readout.jsonl';seen={r['key']:r for r in ev.read_rows(raw,allow_partial_tail=True)[0]} if raw.exists() else {}
    if raw.exists() and raw.read_bytes() and not raw.read_bytes().endswith(b'\n'):ev.save_rows(raw,list(seen.values()))
    request=RequestConfig(max_tokens=1,temperature=0,seed=42,return_details=True)
    # Actual engine smoke is separate from formal outputs and checks both batch
    # alignment and the constraint. Passing does not imply a correct answer.
    smoke=engine.infer([InferRequest(**copy.deepcopy(t['payload'])) for t in tasks[:2]],request,use_tqdm=False)
    assert len(smoke)==min(2,len(tasks))
    for r in smoke:
        if isinstance(r,Exception):raise r
        assert r.choices[0].message.content.strip() in set('ABCD')
    ev.save_json(folder/'smoke.json',{'passed':True,'responses':[r.choices[0].message.content for r in smoke],
                 'constraint_verified':True,'when':ev.now()})
    todo=[t for t in tasks if t['key'] not in seen]
    with raw.open('a') as f:
        for start in range(0,len(todo),4):
            batch=todo[start:start+4]
            outputs=engine.infer([InferRequest(**copy.deepcopy(t['payload'])) for t in batch],request,use_tqdm=False)
            assert len(outputs)==len(batch)
            for task,res in zip(batch,outputs):
                if isinstance(res,Exception):raise res
                choice=res.choices[0];selected=(choice.message.content or '').strip()
                assert selected in set('ABCD') and len(choice.token_ids)==1
                result={k:v for k,v in task.items() if k!='payload'}
                result.update(selected=selected,completion_token_ids=choice.token_ids,
                              prompt_token_ids=res.prompt_token_ids,usage=dataclasses.asdict(res.usage),
                              finish_reason=choice.finish_reason,saved_at=ev.now(),adapter=profile['adapter'])
                f.write(json.dumps(result,ensure_ascii=False)+'\n');seen[task['key']]=result
            f.flush();os.fsync(f.fileno())
            ev.save_json(folder/'state.json',{'status':'running','saved':len(seen),'updated_at':ev.now()})
            print(args.worker,'finalized',len(seen),flush=True)
    ev.save_json(folder/'state.json',{'status':'complete_current_batch','saved':len(seen),'updated_at':ev.now()})


def freeze(work):
    path=work/'manifest.json'
    if path.exists():
        m=json.loads(path.read_text())
        for name,sha in m['sources'].items():assert ev.file_hash(name)==sha,name
        for profile in m['profiles'].values():
            if profile['adapter']:
                for name,sha in profile['adapter']['sha256'].items():
                    assert ev.file_hash(Path(profile['adapter']['path'])/name)==sha
        return m
    tracked=['answer_readout.py','repair_answer_outputs.py','v7_gpu_resources.py']
    sources={}
    for name in tracked:
        src=ROOT/'scripts'/name;dst=work/'source_snapshots'/name
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);sources[str(src)]=ev.file_hash(src)
    m={'created_at':ev.now(),'profiles':profiles(),'sources':sources,
       'generation':{'backend':'ms-swift TransformersEngine','batch_size':4,'max_tokens':1,'temperature':0,
                     'seed':42,'allowed_outputs':list('ABCD'),'max_pixels':1003520},
       'sources_watched':[str(FULL),str(RECALL)],'oracle_material':False,'new_training':False}
    ev.save_json(path,m);return m


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--work',type=Path,default=WORK)
    ap.add_argument('--watch',action='store_true');ap.add_argument('--worker');ap.add_argument('--tasks',type=Path)
    args=ap.parse_args();args.work=args.work.resolve();args.work.mkdir(parents=True,exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',MAX_PIXELS='1003520',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    freeze(args.work)
    if args.worker:worker(args);return
    with ev.run_lock(args.work/'.lock'):
        try:
            while True:
                summary,pending=refresh(args.work)
                if not args.watch:
                    print(json.dumps(summary['totals']));return
                total_pending=sum(map(len,pending.values()))
                if total_pending and can_admit(FULL,'0',18000):
                    # Reading a self-recall output never alters the paired arm's
                    # primary parsing score. Both arms receive the same fallback.
                    name=min(pending,key=lambda n:(not n.startswith('rules_'),n));tasks=pending[name]
                    path=args.work/'task_batches'/(name+'_'+ev.digest([t['key'] for t in tasks])[:12]+'.jsonl')
                    ev.save_rows(path,tasks)
                    with reserve(FULL,'0','answer_output_repair:'+name,18000,poll=2) as reservation:
                        folder=args.work/'jobs'/name;folder.mkdir(parents=True,exist_ok=True)
                        cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--work',str(args.work),
                             '--worker',name,'--tasks',str(path)]
                        with (folder/'worker.log').open('a') as log:
                            child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
                            reservation.bind(child.pid)
                            ev.save_json(args.work/'status.json',{'status':'running_readout','pid':os.getpid(),
                                         'worker_pid':child.pid,'profile':name,'batch_tasks':len(tasks),'updated_at':ev.now()})
                            assert child.wait()==0,f'Worker failed: {folder}/worker.log'
                    continue
                states=[json.loads((w/'status.json').read_text()).get('status') for w in (FULL,RECALL)]
                done=all(s=='complete' for s in states) and not total_pending
                ev.save_json(args.work/'status.json',{'status':'complete' if done else 'watching_and_waiting_for_gpu',
                             'pid':os.getpid(),'totals':summary['totals'],'source_states':states,'updated_at':ev.now()})
                if done:return
                time.sleep(30)
        except BaseException as exc:
            ev.save_json(args.work/'status.json',{'status':'failed','error':str(exc),'traceback':traceback.format_exc(),'updated_at':ev.now()})
            raise


if __name__=='__main__':main()
