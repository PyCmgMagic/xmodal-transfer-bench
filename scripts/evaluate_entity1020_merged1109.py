#!/usr/bin/env python3
"""Fresh Qwen3 base evaluation of the merged entity1020 package."""
from __future__ import annotations

import copy
import argparse
import os
import shutil
import signal
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import evaluate_entity1020_audit_filtered as common

ROOT, MODELS = common.ROOT, common.MODELS
read, rows, save, sha, write_rows = common.read, common.rows, common.save, common.sha, common.write_rows
now, rotate, engine = common.now, common.rotate, common.engine
DEST = ROOT / 'entity1020/merged_1109'
WORK = ROOT / 'runs/entity1020_merged1109_qwen3_base_20261020'
PRIOR = ROOT / 'runs/entity1020_audit_filtered_qwen3_base_20261020'


def evaluator():
    return common.module(WORK / 'data/frozen_evaluator.py', 'merged1109_evaluator')


def prepare():
    inventory = read(DEST / 'package_inventory.json')
    for rel, digest in inventory['sha256'].items():
        assert sha(DEST / rel) == digest, rel
    assert read(DEST / 'verification.json')['status'] == 'passed'
    if (WORK / 'manifest.json').exists():
        manifest = read(WORK / 'manifest.json')
        assert manifest['package_inventory_sha256'] == sha(DEST / 'package_inventory.json')
        for rel, digest in manifest['input_hashes'].items():
            assert sha(WORK / rel) == digest, rel
        return
    out = WORK / 'data'
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PRIOR / 'data/frozen_evaluator.py', out / 'frozen_evaluator.py')
    for name in ('unit_membership.jsonl', 'eval_labels.jsonl'):
        shutil.copy2(DEST / 'audit' / name, out / name)
    membership = {r['unit_id']: r for r in rows(out / 'unit_membership.jsonl')}
    audit = {r['sample_id']: r for r in rows(DEST / 'audit/sample_audit.jsonl')}
    answers = {r['id']: r for r in rows(out / 'eval_labels.jsonl')}
    assert len(membership) == 1109 and len(answers) == 2218
    ev = evaluator()
    labels = []
    for test, direction in (('text', 'T2T'), ('image', 'T2V')):
        values = []
        for item in rows(DEST / direction / 'eval.jsonl'):
            for rotation in range(4):
                value = copy.deepcopy(item)
                value['id'] = f'{test}|r{rotation}|{item["id"]}'
                value['messages'][-1]['content'] = rotate(value['messages'][-1]['content'], rotation)
                value = ev.prepare_row(value, DEST, direction)
                assert set(value) <= {'id', 'qid', 'messages', 'images'}
                assert [r['role'] for r in value['messages']] == ['system', 'user']
                unit = audit[item['id']]['unit_id']
                label = dict(id=value['id'], source_id=item['id'], unit_id=unit,
                             qid=item['qid'], probe_id=answers[item['id']]['probe_id'],
                             origin_group=membership[unit]['origin_group'], test=test, rotation=rotation,
                             answer='ABCD'[('ABCD'.index(answers[item['id']]['answer']) - rotation) % 4],
                             generation_version='entity1020-merged1109-v1')
                labels.append(label)
                values.append(value)
        assert len(values) == 8872
        write_rows(out / f'eval_{test}.jsonl', values)
        for probe in ('direct', 'statement'):
            assert Counter(r['answer'] for r in labels if r['test'] == test and r['probe_id'] == probe) == dict.fromkeys('ABCD', 1109)
    assert len(labels) == len({r['id'] for r in labels}) == 17744
    write_rows(out / 'labels.jsonl', labels)
    save(WORK / 'manifest.json', dict(created_at=now(), source=str(DEST),
        model=str(MODELS['qwen3']), adapter=None, fresh_inference=True,
        prior_predictions_used_as_inference_cache=False, units=1109, questions=2218,
        expected_predictions=17744, rotations=4, temperature=0, seed=42,
        max_new_tokens=32, max_pixels=1003520, batches=dict(text=8, image=8),
        backend='transformers', dtype='bfloat16', parser='strict',
        package_inventory_sha256=sha(DEST / 'package_inventory.json'),
        input_hashes={str(p.relative_to(WORK)): sha(p) for p in sorted(out.glob('*')) if p.is_file()},
        note='Repaired candidates remain uncertified. No answer-correctness-based selection.'))


def run():
    prepare()
    engine.DATA = DEST
    engine.MODELS = {'qwen3': MODELS['qwen3']}
    engine.rows = rows
    ev = evaluator()
    original = ev.swift_command
    def command(args, *other):
        args.batch_size = args.write_batch_size = 8
        return original(args, *other)
    ev.swift_command = command
    engine.evaluator = lambda: ev
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, engine.signal_stop)
    save(WORK / 'controller.json', dict(status='running', pid=os.getpid(), started_at=now()))
    def worker(test, gpu):
        folder = WORK / f'base/qwen3/{test}'
        engine.infer(WORK, 'qwen3', gpu, folder, rows(WORK / f'data/eval_{test}.jsonl'))
        engine.score_predictions(WORK, folder)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            fs = [pool.submit(worker, t, g) for t, g in (('text', '1'), ('image', '0'))]
            for future in fs:
                future.result()
        analyze()
        save(WORK / 'controller.json', dict(status='complete', pid=os.getpid(), completed_at=now()))
    except BaseException as exc:
        save(WORK / 'controller.json', dict(status='failed', pid=os.getpid(), error=str(exc),
             traceback=traceback.format_exc(), updated_at=now()))
        raise


def analyze():
    import numpy as np
    rng = np.random.default_rng(20261020)
    labels = {r['id']: r for r in rows(WORK / 'data/labels.jsonl')}
    membership = {r['unit_id']: r for r in rows(WORK / 'data/unit_membership.jsonl')}
    origins = Counter(r['origin_group'] for r in membership.values())
    assert sorted(origins.values()) == [21, 1088]
    def metric(rs):
        grouped = {}
        for r in rs:
            grouped.setdefault(r['unit_id'], []).append(float(r['correct']))
        per_unit = np.array([np.mean(v) for _, v in sorted(grouped.items())])
        boot = per_unit[rng.integers(0, len(per_unit), size=(4000, len(per_unit)))].mean(axis=1)
        return dict(total=len(rs), units=len(grouped), correct=sum(r['correct'] for r in rs),
                    accuracy=sum(r['correct'] for r in rs) / len(rs),
                    invalid=sum(not r['format_valid'] for r in rs),
                    cluster_bootstrap_95=np.quantile(boot, [.025, .975]).tolist(),
                    gold_options=dict(Counter(r['answer'] for r in rs)),
                    predicted_options=dict(Counter(r['predicted'] for r in rs)))
    output, combined = {}, []
    for test in ('text', 'image'):
        rs = rows(WORK / f'base/qwen3/{test}/results.jsonl')
        assert len(rs) == len({r['id'] for r in rs}) == 8872
        assert {r['unit_id'] for r in rs} == set(membership)
        for r in rs:
            assert all(r[k] == labels[r['id']][k] for k in labels[r['id']])
        assert Counter(r['answer'] for r in rs) == dict.fromkeys('ABCD', 2218)
        combined.extend(rs)
        groups = {}
        for origin in ['all', *sorted(origins)]:
            group = [r for r in rs if origin == 'all' or r['origin_group'] == origin]
            groups[origin] = {probe: metric([r for r in group if probe == 'overall' or r['probe_id'] == probe])
                             for probe in ('overall', 'direct', 'statement')}
        old = rows(PRIOR / f'base/qwen3/{test}/results.jsonl')
        output[test] = dict(groups=groups, prior_1088=metric(old),
                            by_rotation={str(i): metric([r for r in rs if r['rotation'] == i]) for i in range(4)})
    write_rows(WORK / 'results.jsonl', combined)
    save(WORK / 'reports/statistics.json', dict(created_at=now(), units=1109, questions=2218,
        origin_units=dict(origins), results=output))
    lines = ['# 合并 1,109 组：Qwen3 未训练基座重测', '',
        'Qwen3-VL-4B-Instruct 原始模型，无适配器；本次全部重新推理，不复用旧预测。',
        '1,109 组、2,218 题，文字/图像各 8,872 次作答，共 17,744 次。每题四轮选项循环轮换。',
        'temperature=0，seed=42，max_new_tokens=32，max_pixels=1003520，bf16，批量 8，严格解析。', '',
        '| 数据范围 | 题型 | 文字准确率 | 图像准确率 |', '| --- | --- | ---: | ---: |']
    names = {'all': '全量 1,109 组', **{k: f'{"原有" if v == 1088 else "新增修复候选"} {v} 组' for k, v in origins.items()}}
    for group in ['all', *sorted(origins)]:
        for probe in ('overall', 'direct', 'statement'):
            lines.append('| ' + names[group] + ' | ' + probe + ' | ' + ' | '.join(
                f'{100*output[t]["groups"][group][probe]["accuracy"]:.2f}%' for t in ('text', 'image')) + ' |')
    lines += ['', '| 输入 | 全量 95% 区间 | 原有 1,088 组上次结果 | 无效输出 |', '| --- | ---: | ---: | ---: |']
    for test in ('text', 'image'):
        m = output[test]['groups']['all']['overall']
        lo, hi = m['cluster_bootstrap_95']
        lines.append(f'| {test} | {100*lo:.2f}–{100*hi:.2f}% | {100*output[test]["prior_1088"]["accuracy"]:.2f}% | {m["invalid"]} |')
    lines += ['', '95% 区间按实体组 bootstrap；四轮呈现不是四个独立样本。无效输出计错，不通过重试改变答案。',
        '每种输入正确答案 A/B/C/D 各 2,218 次；每个题型各 1,109 次。',
        '本次合并重新排列了干扰项；旧版循环轮换不一定覆盖新排列，因此原有组前后差异也可能来自排列及批量数值变化。',
        '新增 21 组为修复候选，仍未完成语义与时间新颖性的认证；原有 1,088 组也是未识别具体问题，并非已认证干净。',
        '25% 是均匀随机猜测期望；高于 25% 本身不能证明训练数据泄漏。新增组样本小，分组结果只用于诊断。', '']
    (WORK / 'reports/报告.md').write_text('\n'.join(lines))
    save(WORK / 'verification.json', dict(status='passed', logical_predictions=len(combined),
        unique_ids=len({r['id'] for r in combined}), gold_balance_per_test=dict.fromkeys('ABCD', 2218),
        all_label_fields_aligned=True, fresh_inference=True, checked_at=now()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=('prepare', 'run', 'start', 'analyze', 'status'))
    action = ap.parse_args().action
    if action == 'prepare':
        prepare()
    elif action == 'run':
        run()
    elif action == 'analyze':
        analyze()
    elif action == 'status':
        print(read(WORK / 'controller.json'))
        for test in ('text', 'image'):
            p = WORK / f'base/qwen3/{test}/progress.json'
            if p.exists():
                print(test, read(p))
    else:
        prepare()
        assert not engine.alive(read(WORK / 'controller.json').get('pid'))
        (WORK / 'execution').mkdir(exist_ok=True)
        with (WORK / 'controller.log').open('a') as log:
            child = __import__('subprocess').Popen(
                [str(ROOT / '.venv/bin/python'), '-u', str(__import__('pathlib').Path(__file__).resolve()), 'run'],
                cwd=ROOT, stdin=__import__('subprocess').DEVNULL, stdout=log,
                stderr=__import__('subprocess').STDOUT, start_new_session=True)
        print(dict(pid=child.pid, work=str(WORK)))


if __name__ == '__main__':
    main()
