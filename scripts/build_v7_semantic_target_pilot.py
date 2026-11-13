#!/usr/bin/env python3
"""Prepare an untrained candidate with direct semantic and MCQ supervision."""
import argparse
import copy
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
import run_v7_evaluation as ev
from run_v7_full_experiment import rows, read, save, sha
from pilot_v7_natural_sft import condition_text, record


def build(source, out):
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'checkpoint').exists(), 'Do not rewrite a candidate after training starts'
    training = rows(source / 'train.jsonl')
    checks = {r['id']: r for r in rows(source / 'generated_solver_checks.jsonl')}
    grouped = defaultdict(list)
    for row in training:
        if row['training_task'] != 'compact_rule':
            grouped[(row['qid'], row['training_task'])].append(row)
    direct_ids = set()
    for group in grouped.values():
        ranked = sorted(group, key=lambda r: hashlib.sha256(('semantic-target:42:' + r['id']).encode()).hexdigest())
        assert len(ranked) % 2 == 0
        direct_ids.update(r['id'] for r in ranked[:len(ranked) // 2])
    candidate = []
    counts = Counter()
    for source_row in training:
        row = copy.deepcopy(source_row)
        if row['id'] not in direct_ids:
            row['supervision_format'] = 'compact_rule' if row['training_task'] == 'compact_rule' else 'mcq'
            candidate.append(row)
            counts[row['supervision_format']] += 1
            continue
        check = checks[row['id']]
        question = row['messages'][1]['content'].split('\n\nOptions:\n', 1)[0]
        task = row['training_task']
        if task == 'atomic_condition':
            question += '\nState the exact requirement in one sentence.'
            answer = condition_text(check['condition'])
            instruction = 'State the requirement for the named rule. Do not select a choice letter.'
        elif task in ('record_code_recall', 'practice_application'):
            question += '\nReturn only the inspection code, such as H7, J2, K9, or M4.'
            answer = check['options'][check['gold']].removeprefix('Result: ').removesuffix('.')
            assert answer in ('H7', 'J2', 'K9', 'M4')
            instruction = 'Return only the inspection recording code requested by the question.'
        else:
            assert task == 'practice_diagnosis'
            question += '\nReturn the correct Check 1, Check 2, and any Check 3 statuses, followed by Qualification. Use Pass/Fail.'
            answer = record(check['rule'], check['flags'])
            instruction = 'Return the correct condition statuses and qualification. Use the requested record format.'
        row['id'] += ':semantic'
        row['supervision_format'] = 'semantic'
        row['messages'] = [{'role': 'system', 'content': instruction}, {'role': 'user', 'content': question},
                           {'role': 'assistant', 'content': answer}]
        candidate.append(row)
        counts['semantic'] += 1
    assert len(candidate) == 780 and counts == {'compact_rule': 12, 'mcq': 384, 'semantic': 384}
    assert not any(row.get('images') for row in candidate)
    ev.save_rows(out / 'train.jsonl', candidate)
    save(out / 'manifest.json', {'status': 'prepared_not_trained', 'rules': read(source / 'manifest.json')['rules'],
        'training_rows': len(candidate), 'formats': dict(counts), 'source_train_sha256': sha(source / 'train.jsonl'),
        'source_generated_checks_sha256': sha(source / 'generated_solver_checks.jsonl'), 'train_sha256': sha(out / 'train.jsonl'),
        'knowledge_sha256': read(source / 'manifest.json')['knowledge_sha256'], 'proposed_epochs': 1,
        'training_labels_source': 'previous independently validated training observations and learning rules; no evaluation questions or labels read',
        'evaluation_format': 'original multiple choice, exact A/B/C/D; candidate efficacy not yet measured',
        'script_sha256': sha(Path(__file__)), 'created_at': ev.now()})
    (out / 'README.md').write_text('# 语义监督候选训练集（尚未训练）\n\n保留780行规模、同12条规则：384条原选择题监督，384条直接条件/代码/检查结果监督，12条紧凑规则说明。仅改变训练目标，不改变任何原始评测题、标签或图片。评测继续选择题、不开COT。\n\n一半训练题显式预测条件值、实际代码或检查结果，而不是只预测随机位置的字母；另一半保留选择题格式。用于验证是否能改善规则获取和应用。直接检查结果是可核验的任务输出，不包含推理理由。\n\n这是已准备的候选方案，没有checkpoint和评测结果，不宣称已修复数据集。建议下一轮仍1epoch，并单独统计首个答案token准确率、代码映射回忆及原题迁移。\n')
    print(dict(counts))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    build(args.source.resolve(), args.out.resolve())


if __name__ == '__main__':
    main()
