#!/usr/bin/env python3
"""Freeze the audited 1088-unit subset and freshly evaluate Qwen3 base."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import run_entity_v19_full_experiment as engine
from entity_v19_experiment_data import ROOT, MODELS, now, read, rows, rotate, save, sha, write_rows

SOURCE = ROOT / 'entity1020/balanced_1572'
AUDIT = ROOT / 'diagnostics/entity1020_knowledge_audit_20261020'
DEST = ROOT / 'entity1020/audit_filtered_1088'
WORK = ROOT / 'runs/entity1020_audit_filtered_qwen3_base_20261020'
PRIOR = ROOT / 'runs/entity1020_balanced_qwen3_ep3_20261020'


def rows(path):
    # JSON strings can contain U+2028; only a literal newline separates JSONL records.
    return [json.loads(line) for line in Path(path).read_text().split('\n') if line.strip()]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def selected():
    parts = read(AUDIT / 'unit_partitions.json')['partitions']
    keep = set(parts['未识别具体问题（非已认证干净）'])
    excluded = set(parts['明确问题']) | set(parts['待核查'])
    assert len(keep) == 1088 and len(excluded) == 484 and not keep & excluded
    for rel, digest in read(AUDIT / 'selection_freeze.json')['files'].items():
        assert sha(AUDIT / rel) == digest
    for rel, digest in read(AUDIT / 'selection_freeze.json')['source_hashes'].items():
        assert sha(SOURCE / 'source' / rel) == digest
    return keep, excluded, parts


def build_package():
    keep, excluded, parts = selected()
    if (DEST / 'filter_manifest.json').exists():
        assert set(read(DEST / 'filter_manifest.json')['retained_unit_ids']) == keep
        return
    assert not DEST.exists(), f'Inspect partial package: {DEST}'
    qids = {r['qid'] for r in rows(SOURCE / 'source/answers.jsonl') if r['id'] in keep}
    samples = {r['sample_id'] for r in rows(SOURCE / 'audit/sample_audit.jsonl') if r['unit_id'] in keep}
    for src in sorted(SOURCE.rglob('*')):
        rel = src.relative_to(SOURCE)
        if not src.is_file() or 'images' in rel.parts or '__pycache__' in rel.parts:
            continue
        if src.suffix == '.jsonl':
            values = rows(src)
            def retain(r):
                if 'qid' in r:
                    return r['qid'] in qids
                if 'unit_id' in r:
                    return r['unit_id'] in keep
                if 'sample_id' in r:
                    return r['sample_id'] in samples
                if 'id' in r:
                    return r['id'] in keep or r['id'] in samples
                raise AssertionError(f'Unknown row scope: {rel}')
            write_rows(DEST / rel, [r for r in values if retain(r)])
        elif src.suffix == '.py' or rel.parts[0] == 'scripts':
            (DEST / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, DEST / rel)
    catalog = rows(DEST / 'audit/image_catalog.jsonl')
    for item in catalog:
        rel = item['file']
        (DEST / rel).parent.mkdir(parents=True, exist_ok=True)
        os.link(SOURCE / rel, DEST / rel)
    assert len(catalog) == 2176
    (DEST / 'audit/paired_qids.txt').write_text('\n'.join(sorted(qids)) + '\n')
    labels = rows(DEST / 'audit/eval_labels.jsonl')
    balance = {p:dict(Counter(r['answer'] for r in labels if r['probe_id'] == p)) for p in ('direct','statement')}
    manifest = dict(schema='entity1020-audit-filter-v1', created_at=now(), source=str(SOURCE),
        selection_freeze_sha256=sha(AUDIT/'selection_freeze.json'),
        partition_sha256=sha(AUDIT/'unit_partitions.json'), prediction_correctness_used_for_selection=False,
        retained_unit_ids=sorted(keep), excluded_unit_ids=sorted(excluded),
        excluded_clear=len(parts['明确问题']), excluded_review=len(parts['待核查']),
        knowledge_units=1088, questions_per_direction=2176, images=2176,
        original_option_order_preserved=True, unrotated_answer_counts=balance,
        selection_note='未识别具体问题，不等于已认证干净；图片语义身份限制沿用原包。')
    for name in ('build_manifest.json','summary.json'):
        m = read(SOURCE/name)
        m.update(source_package=str(SOURCE), knowledge_units=1088, source_knowledge_units=1572,
            images=2176, questions_per_direction=2176, excluded_units=484,
            excluded_qids=sorted({r['qid'] for r in rows(SOURCE/'source/answers.jsonl') if r['id'] in excluded}),
            training_rows_per_direction=1088, evaluation_rows_per_direction=2176,
            answer_balance='unrotated subset counts in filter_manifest; four rotations balanced exactly',
            kind_counts=dict(Counter(r['kind'] for r in rows(DEST/'source/entities.jsonl'))))
        save(DEST/name,m)
    (DEST/'README.md').write_text('# entity1020：审查后保留的 1,088 组\n\n'
        '从 balanced_1572 排除 82 组明确问题与 402 组待核查，保留 1,088 组、2,176 题。'
        '筛选依据已在评测前冻结，不按模型答对/答错筛选。保留原题、选项顺序、答案、学习材料和图片。\n\n'
        '筛选后的原始答案位置不再严格均衡；重测每题使用四次循环轮换，使各题型合计 A/B/C/D 各 1,088 次。'
        '四次轮换是同一道题的四次呈现，不是四个独立样本。\n\n'
        'source/knowledge_samples.jsonl 保留原始生成记录的历史选项顺序；当前标签以 source/questions.jsonl、'
        'source/answers.jsonl 和 audit/eval_labels.jsonl 为准。图片为原包的硬链接，请勿原地编辑。\n\n'
        '保留组只是尚未识别到具体问题；图片语义身份未全面核验。2025-12-31 为知识分界。\n')
    save(DEST/'verification.json', module(DEST/'verify_package.py','filtered_verifier').check(DEST))
    save(DEST/'filter_manifest.json',manifest)
    inventory = {str(p.relative_to(DEST)):sha(p) for p in sorted(DEST.rglob('*'))
                 if p.is_file() and '__pycache__' not in p.parts and p.name != 'package_inventory.json'}
    save(DEST/'package_inventory.json',dict(sha256=inventory,self_hash_excluded=True))


def evaluator():
    return module(WORK/'data/frozen_evaluator.py','filtered_evaluator')


def prepare():
    build_package()
    if (WORK/'manifest.json').exists():
        for rel,digest in read(WORK/'manifest.json')['input_hashes'].items():
            assert sha(WORK/rel) == digest
        return
    out = WORK/'data'
    out.mkdir(parents=True,exist_ok=True)
    shutil.copy2(DEST/'scripts/run_evaluation.py',out/'frozen_evaluator.py')
    ev = evaluator()
    audit = {r['sample_id']:r for r in rows(DEST/'audit/sample_audit.jsonl')}
    answers = {r['id']:r for r in rows(DEST/'audit/eval_labels.jsonl')}
    old_labels = {r['id']:r for r in rows(PRIOR/'data/labels.jsonl')}
    labels=[]
    for test,direction in (('text','T2T'),('image','T2V')):
        values=[]
        old_inputs={r['id']:r for r in rows(PRIOR/f'data/eval_{test}.jsonl')}
        for item in rows(DEST/direction/'eval.jsonl'):
            for rotation in range(4):
                value=copy.deepcopy(item)
                value['id']=f'{test}|r{rotation}|{item["id"]}'
                value['messages'][-1]['content']=rotate(value['messages'][-1]['content'],rotation)
                value=ev.prepare_row(value,DEST,direction)
                assert set(value) <= {'id','qid','messages','images'}
                assert [r['role'] for r in value['messages']]==['system','user']
                previous=old_inputs[value['id']]
                assert value['messages']==previous['messages']
                assert value.get('images',[])==[p.replace(str(SOURCE),str(DEST)) for p in previous.get('images',[])]
                label=dict(id=value['id'],source_id=item['id'],unit_id=audit[item['id']]['unit_id'],
                    qid=item['qid'],probe_id=answers[item['id']]['probe_id'],test=test,rotation=rotation,
                    answer='ABCD'[('ABCD'.index(answers[item['id']]['answer'])-rotation)%4],
                    generation_version='entity1020-audit-filtered-v1')
                assert label['answer']==old_labels[value['id']]['answer']
                values.append(value); labels.append(label)
        assert len(values)==8704
        write_rows(out/f'eval_{test}.jsonl',values)
        for probe in ('direct','statement'):
            assert Counter(r['answer'] for r in labels if r['test']==test and r['probe_id']==probe)==dict.fromkeys('ABCD',1088)
    assert len(labels)==len({r['id'] for r in labels})==17408
    write_rows(out/'labels.jsonl',labels)
    shutil.copy2(AUDIT/'unit_partitions.json',out/'unit_partitions.json')
    shutil.copy2(AUDIT/'selection_freeze.json',out/'selection_freeze.json')
    save(WORK/'manifest.json',dict(created_at=now(),source=str(DEST),model=str(MODELS['qwen3']),adapter=None,
        fresh_inference=True,prior_predictions_used_as_inference_cache=False,units=1088,questions=2176,
        expected_predictions=17408,rotations=4,temperature=0,seed=42,max_new_tokens=32,max_pixels=1003520,
        batches=dict(text=8,image=4),backend='transformers',dtype='bfloat16',parser='strict',
        input_hashes={str(p.relative_to(WORK)):sha(p) for p in sorted(out.glob('*')) if p.is_file()},
        filter_manifest_sha256=sha(DEST/'filter_manifest.json'),
        validation='all prompts/labels equal prior main evaluation; only image package root changed'))


def run():
    prepare()
    engine.DATA=DEST
    engine.MODELS={'qwen3':MODELS['qwen3']}
    ev=evaluator()
    original_command=ev.swift_command
    def command(args,*other):
        args.batch_size=16 if 'image' in str(other[-2]) else 8
        args.write_batch_size=args.batch_size
        return original_command(args,*other)
    ev.swift_command=command
    engine.evaluator=lambda:ev
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,engine.signal_stop)
    save(WORK/'controller.json',dict(status='running',pid=os.getpid(),started_at=now()))
    def worker(test,gpu):
        folder=WORK/f'base/qwen3/{test}'
        engine.infer(WORK,'qwen3',gpu,folder,rows(WORK/f'data/eval_{test}.jsonl'))
        engine.score_predictions(WORK,folder)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            fs=[pool.submit(worker,t,g) for t,g in (('text','1'),('image','0'))]
            for f in fs: f.result()
        analyze()
        save(WORK/'controller.json',dict(status='complete',pid=os.getpid(),completed_at=now()))
    except BaseException as exc:
        save(WORK/'controller.json',dict(status='failed',pid=os.getpid(),error=str(exc),traceback=traceback.format_exc(),updated_at=now()))
        raise


def analyze():
    import numpy as np
    keep,_,_=selected()
    old=rows(PRIOR/'base/qwen3/results.jsonl')
    old_by_id={r['id']:r for r in old}
    output={}
    rng=np.random.default_rng(20261020)
    def metric(rs):
        return dict(total=len(rs),correct=sum(r['correct'] for r in rs),
                    accuracy=sum(r['correct'] for r in rs)/len(rs),invalid=sum(not r['format_valid'] for r in rs),
                    gold_options=dict(Counter(r['answer'] for r in rs)),
                    predicted_options=dict(Counter(r['predicted'] for r in rs)))
    all_rows=[]
    for test in ('text','image'):
        rs=rows(WORK/f'base/qwen3/{test}/results.jsonl')
        assert len(rs)==len({r['id'] for r in rs})==8704
        assert {r['unit_id'] for r in rs}==keep
        assert Counter(r['answer'] for r in rs)==dict.fromkeys('ABCD',2176)
        all_rows+=rs
        groups={}
        for probe in ('overall','direct','statement'):
            subset=[r for r in rs if probe=='overall' or r['probe_id']==probe]
            grouped={u:[] for u in sorted(keep)}
            for r in subset: grouped[r['unit_id']].append(float(r['correct']))
            per_unit=np.array([np.mean(v) for v in grouped.values()])
            boot=per_unit[rng.integers(0,len(per_unit),size=(4000,len(per_unit)))].mean(axis=1)
            groups[probe]={**metric(subset),'cluster_bootstrap_95':np.quantile(boot,[.025,.975]).tolist()}
        prior=[r for r in old if r['test']==test]
        changes=[dict(id=r['id'],unit_id=r['unit_id'],old_response=old_by_id[r['id']]['response'],new_response=r['response'],
                      old_correct=old_by_id[r['id']]['correct'],new_correct=r['correct'])
                 for r in rs if r['predicted']!=old_by_id[r['id']]['predicted']]
        write_rows(WORK/f'reports/changed_predictions_{test}.jsonl',changes)
        output[test]=dict(fresh=groups,original_full=metric(prior),
            prior_same_subset=metric([r for r in prior if r['unit_id'] in keep]),
            changed_predicted_answers=len(changes),by_rotation={str(i):metric([r for r in rs if r['rotation']==i]) for i in range(4)},
            inference_attempts=[read(p) for p in sorted((WORK/f'base/qwen3/{test}').glob('attempt_*.process.json'))])
    write_rows(WORK/'results.jsonl',all_rows)
    save(WORK/'reports/statistics.json',dict(created_at=now(),units=1088,questions=2176,removed_units=484,results=output))
    lines=['# entity1020 筛选后 Qwen3 未训练基座重测','',
        '排除 82 组明确问题与 402 组待核查，保留 1,088 组 / 2,176 题。原数据保留。',
        '本次重新生成 17,408 次作答；每题四次循环轮换，统计区间以实体组为重采样单位。','',
        '| 测试 | 原始全量 | 筛选后新评测 | 95% 区间 |','| --- | ---: | ---: | ---: |']
    for test,name in (('text','文字'),('image','图像')):
        v=output[test]; m=v['fresh']['overall']; lo,hi=m['cluster_bootstrap_95']
        lines.append(f'| {name} | {100*v["original_full"]["accuracy"]:.2f}% | {100*m["accuracy"]:.2f}% | {100*lo:.2f}–{100*hi:.2f}% |')
    lines+=['','| 题型 | 文字 | 图像 |','| --- | ---: | ---: |']
    for probe in ('direct','statement'):
        lines.append('| '+probe+' | '+' | '.join(f'{100*output[t]["fresh"][probe]["accuracy"]:.2f}%' for t in ('text','image'))+' |')
    lines+=['','模型为 Qwen3-VL-4B-Instruct 原始基座，无适配器。temperature=0，seed=42，max_new_tokens=32，严格解析。',
        '本次文字批量 8；图像先用批量 4，之后保留已完成作答并用批量 16 续跑，逐次命令见 statistics.json。原始全量批量 1。其余输入和答案一致，仅图片目录根路径改变。',
        '不同批量下的数值计算可能造成少量答案变化；逐条变化清单见 changed_predictions_*.jsonl。','',
        '筛选依据在本次评测前冻结，未依据答对/答错删除。保留组未被认证为完全干净，图片身份未全面核验。',
        '402 组待核查是保守隔离，不代表 402 组均已证实存在问题。筛选含模型辅助审查；本轮属于诊断性实验，不是独立验证集。',
        '25% 是均匀随机猜测的期望，不要求一个具备知识和推理能力的模型恰好达到 25%；高于 25% 本身不能证明训练泄漏。','']
    lines+=['| 测试 | 相对全量变化（百分点） | 筛选后旧预测 | 新旧答案变化数 | 无效输出 |',
            '| --- | ---: | ---: | ---: | ---: |']
    for test,name in (('text','文字'),('image','图像')):
        v=output[test]; m=v['fresh']['overall']
        lines.append(f'| {name} | {100*(m["accuracy"]-v["original_full"]["accuracy"]):+.2f} | '
            f'{100*v["prior_same_subset"]["accuracy"]:.2f}% | {v["changed_predicted_answers"]} / {m["total"]} | {m["invalid"]} |')
    lines+=['','每种输入的 8,704 次作答中，正确答案 A/B/C/D 各 2,176 次；每个题型各 1,088 次。',
        '无效输出计入总分并按错误处理；没有通过重试改答案或删除无效作答。','']
    (WORK/'reports/报告.md').write_text('\n'.join(lines))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=('prepare','run','start','analyze','status'))
    a=ap.parse_args()
    if a.action=='prepare': prepare()
    elif a.action=='run': run()
    elif a.action=='analyze': analyze()
    elif a.action=='status':
        print(json.dumps(read(WORK/'controller.json'),ensure_ascii=False))
        for test in ('text','image'):
            path=WORK/f'base/qwen3/{test}'
            print(test,read(path/'progress.json'))
            for p in sorted(path.glob('attempt_*.raw.jsonl')):
                print(p.name,sum(1 for _ in p.open()))
    else:
        prepare()
        assert not engine.alive(read(WORK/'controller.json').get('pid'))
        dest=WORK/'execution';dest.mkdir(exist_ok=True)
        for p in (Path(__file__),ROOT/'scripts/run_entity_v19_full_experiment.py',ROOT/'scripts/entity_v19_experiment_data.py',ROOT/'scripts/v7_gpu_resources.py'):
            shutil.copy2(p,dest/p.name)
        with (WORK/'controller.log').open('a') as log:
            child=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',str(Path(__file__).resolve()),'run'],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(pid=child.pid,work=str(WORK))))


if __name__=='__main__': main()
