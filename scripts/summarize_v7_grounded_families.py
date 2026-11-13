#!/usr/bin/env python3
"""Summarize the grounded pilot without merging oracle and blind results."""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import pilot_v7_grounded_families as pilot
from run_v7_full_experiment import read, rows, save, sha


def score(group):
    n = len(group)
    valid = [r for r in group if not r['format_error']]
    return {'n': n, 'correct': sum(r['correct'] for r in group),
            'accuracy': sum(r['correct'] for r in group) / n if n else None,
            'valid_accuracy': sum(r['correct'] for r in valid) / len(valid) if valid else None,
            'format_errors': sum(r['format_error'] for r in group),
            'empty': sum(not r['response'].strip() for r in group)}


def fmt(metric):
    return f"{metric['correct']}/{metric['n']} ({100 * metric['accuracy']:.1f}%)" if metric['n'] else '未完成'


def main(out):
    manifest = read(out / 'manifest.json')
    probes = rows(out / 'probes.jsonl')
    by_id = {p['id']: p for p in probes}
    invalid_ids = {p['id'] for p in probes if p['qid'] == 'scale1k_0664_ai_generated'
                   and 'interval endpoints' in p['messages'][-1]['content']}
    combined, cleaned, sample_errors, per_rule = [], [], [], []
    for variant in pilot.VARIANTS:
        assert read(out / variant / 'report.json').get('status') == 'complete', variant
        group = rows(out / variant / 'predictions.jsonl')
        assert len(group) == len(probes) * 2
        assert len({(r['model_condition'], r['id']) for r in group}) == len(group)
        for r in group:
            p = by_id[r['id']]
            assert r['prompt_sha256'] == pilot.ev.digest(p['messages']) and r['gold'] == p['gold']
        for mode in ('base', 'trained'):
            for condition in sorted({r['condition'] for r in group}):
                selected = [r for r in group if r['model_condition'] == mode and r['condition'] == condition]
                combined.append({'variant': variant, 'mode': mode, 'condition': condition, **score(selected)})
                cleaned.append({'variant': variant, 'mode': mode, 'condition': condition,
                                **score([r for r in selected if r['id'] not in invalid_ids])})
        for qid in manifest['rules']:
            for condition in ('new_atomic_fact', 'new_single_mcq', 'new_direct_mcq', 'original_T2T', 'original_T2V'):
                selected = [r for r in group if r['model_condition'] == 'trained' and r['qid'] == qid and r['condition'] == condition]
                per_rule.append({'variant': variant, 'qid': qid, 'condition': condition, **score(selected),
                                 'responses': [{'gold': r['gold'], 'response': r['response']} for r in selected]})
        for condition in ('new_atomic_fact', 'new_single_mcq', 'new_direct_mcq', 'original_T2T', 'original_T2V'):
            wrong = [r for r in group if r['model_condition'] == 'trained' and r['condition'] == condition and not r['correct']]
            for r in wrong[:3]:
                sample_errors.append({**r, 'variant': variant, 'messages': by_id[r['id']]['messages'],
                    'images': by_id[r['id']].get('images', []),
                    'error_class': 'output_format' if r['format_error'] else 'strict_fact_mismatch' if r['answer_format'] == 'semantic' else 'wrong_option'})
    references, balanced, reference_clean = {}, {}, {}
    reference_probes = {r['id']: r for r in rows(out / 'reference_probes.jsonl')}
    for variant in ('base', *pilot.VARIANTS):
        references[variant] = read(out / 'reference_diagnostic' / variant / 'report.json')
        balanced[variant] = read(out / 'balanced_diagnostic' / variant / 'report.json')
        assert references[variant].get('status') == balanced[variant].get('status') == 'complete', variant
        reference_records = rows(out / 'reference_diagnostic' / variant / 'predictions.jsonl')
        reference_clean[variant] = {}
        for condition in references[variant]['metrics']:
            selected = [r for r in reference_records if r['condition'] == condition and
                        not (r['qid'] == 'scale1k_0664_ai_generated' and 'interval endpoints' in
                             reference_probes[r['id']]['messages'][-1]['content'])]
            reference_clean[variant][condition] = score(selected)
    # Audit the pilot shortcut and the original-v7 distribution separately.
    shortcut = {}
    for filename in ('training_solver_checks.jsonl', 'validation_solver_checks.jsonl'):
        group = [r for r in rows(out / filename) if r['task'] == 'diagnosis_mcq']
        expected = 0
        for r in group:
            matching = [i for i, s in enumerate(r['options']) if s == 'Result: ' + r['submitted'] + '.']
            expected += float('ABCD'[matching[0]] == r['answer']) if matching else .25
        shortcut[filename] = {'n': len(group), 'correct_submitted': sum(r['accepted'] for r in group),
                             'copy_else_random_expected_accuracy': expected / len(group)}
    original_diagnosis = {}
    labels = {r['id']: r for r in rows(pilot.DATA / 'audit/eval_labels.jsonl')}
    original = rows(pilot.DATA / 'T2T/eval.jsonl')
    for relation in pilot.RELATIONS:
        group = [r for r in original if labels[r['id']]['entity_kind'] == relation and labels[r['id']]['question_type'] == 'diagnosis']
        accept_count = 0
        for r in group:
            options = r['messages'][-1]['content'].split('Options:\n')[1].split('\n\n')[0].splitlines()
            accept_count += options['ABCD'.index(labels[r['id']]['answer'])].split('. ', 1)[1].startswith('Accept report.')
        original_diagnosis[relation] = {'n': len(group), 'correct_submitted': accept_count}
    # Paired VISION datasets change only full-card targets, not inputs/order/budget.
    explicit, generic = [rows(out / (v + '.jsonl')) for v in ('vision_explicit', 'vision_generic_cards')]
    target_changes = 0
    for a, b in zip(explicit, generic):
        assert a['id'] == b['id'] and a['images'] == b['images'] and a['messages'][:-1] == b['messages'][:-1]
        changed = a['messages'][-1] != b['messages'][-1]
        assert not changed or a['task'] == 'rule_card'
        target_changes += changed
    for path, digest in manifest['sources'].items():
        assert sha(Path(path)) == digest
    for f, digest in manifest['files'].items():
        assert sha(out / f) == digest
    for item in rows(pilot.DATA / 'image_hashes.jsonl'):
        assert sha(pilot.DATA / item['path']) == item['source_sha256'] == item['repaired_sha256']
    validation = {'status': 'pass', 'images_verified': 3000, 'frozen_sources_unchanged': True,
                  'prediction_id_gold_prompt_checks': 'pass', 'vision_comparison_target_changes': target_changes,
                  'variants': {v: read(out / v / 'checkpoint_manifest.json') for v in pilot.VARIANTS},
                  'original_direction_aliases': {'TEXT': ['T2T', 'T2V'], 'VISION': ['V2T', 'V2V']},
                  'actual_unique_requests': len(probes) * 4 + 4 * 48 + 4 * 256,
                  'reused_base_rows_per_adapter': len(probes), 'checked_at': pilot.ev.now()}
    save(out / 'validation.json', validation)
    save(out / 'per_rule.json', per_rule)
    pilot.ev.save_rows(out / 'error_samples.jsonl', sample_errors)
    summary = {'status': 'complete', 'metrics': combined, 'reference_controls': references,
               'clean_metrics': cleaned, 'clean_reference_controls': reference_clean,
               'excluded_invalid_auxiliary_probes': {'ids': sorted(invalid_ids),
                    'reason': 'Linvel is an OR filter with no numeric interval; its endpoint question was invalid.',
                    'raw_results_preserved': True},
               'balanced_diagnosis': balanced, 'pilot_diagnosis_shortcut': shortcut,
               'original_v7_diagnosis': original_diagnosis, 'limits': manifest['limits'],
               'balanced_candidate': {'path': str(out / 'corrected_candidate'), 'status': 'prepared_not_trained'},
               'finished_at': pilot.ev.now()}
    save(out / 'summary.json', summary)
    with (out / 'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)
    index = {(r['variant'], r['mode'], r['condition']): r for r in cleaned}
    names = {'text_explicit': 'TEXT明确监督', 'vision_explicit': 'VISION明确监督', 'vision_generic_cards': 'VISION旧式卡片对照'}
    lines = ['# 两类规则的小规模修复试验：1 epoch', '',
        '本次使用Qwen3-VL-4B，固定哈希选取集合筛选、比例分档各4条规则；三组各512行、batch=4、累积=1、128步、LoRA r8/alpha32、LR=1e-4、seed42。每条64行中32行是有意重复规则卡片，不是64条新知识。最终固定使用第1个epoch的checkpoint，不按评测挑选。', '',
        '结论应以原始题和去除线索后的诊断题为准。新的事实监督有助于部分事实回忆，但不能据此宣布原始四方向任务已经修复。八条规则仅是开发诊断，不能外推1000条或其他模型。', '',
        '## 不给材料的主结果', '',
        '| 训练方式 | 新措辞事实回忆 | 新单步选择题 | 新组合直接题 | 原始文字题 | 原始图片题 |',
        '|---|---|---|---|---|---|']
    conditions = ('new_atomic_fact', 'new_single_mcq', 'new_direct_mcq', 'original_T2T', 'original_T2V')
    lines.append('| 基础模型 | ' + ' | '.join(fmt(index[('text_explicit', 'base', c)]) for c in conditions) + ' |')
    for v in pilot.VARIANTS:
        lines.append('| ' + names[v] + ' | ' + ' | '.join(fmt(index[(v, 'trained', c)]) for c in conditions) + ' |')
    lines += ['', 'TEXT两列原始题对应T2T/T2V；VISION两组对应V2T/V2V。原始题每方向只有16道，个位数差异不构成稳定的总体提升。原始题、标签、3000张PNG均未改动。', '',
        '“事实回忆”是辅助诊断，严格匹配短答案，不与最终选择题准确率混合。发现Linvel的原子问答误问“区间端点”，该规则没有数值区间：上表剔除此无效探针，新措辞事实从32题变为31题，原始32题口径仍保留在metrics中。训练v1有两行该无效辅助问答，候选已纠正；原始v7题目不受影响。', '',
        '原seen_*探针是去掉学习图后的文字题：仅TEXT组是完全见过的输入，VISION组应理解为同一题干的无图回忆；补充reference_seen_atomic_fact才是VISION实际带原图的见过输入。剔除同一无效问题后，两组VISION带图事实均15/15，无图同题均2/15。', '',
        '## 给原学习图的诊断（不计入主准确率）', '',
        '| 模型/适配器 | 原图事实提取 | 原图辅助单步 | 原图辅助新组合 |', '|---|---|---|---|']
    for v in ('base', *pilot.VARIANTS):
        metrics = reference_clean[v]
        values = []
        for c in ('reference_seen_atomic_fact', 'reference_new_single_mcq', 'reference_new_direct_mcq'):
            m = metrics[c]
            values.append(f"{m['correct']}/{m['n']}")
        lines.append('| ' + names.get(v, '基础模型') + ' | ' + ' | '.join(values) + ' |')
    lines += ['', '## 审核题的线索审计与补测', '',
        '本轮v1生成器有一处已确认的错误：64道新审核题中32道提交报告正确，错误报告几乎未放入选项。复制提交报告、找不到就随机选的策略，验证集期望可达62.5%。因此v1审核题高于25%不代表应用能力。这是本轮新生成器的问题，不能解释成原始v7同样有该问题。', '',
        '原始v7审核题：集合筛选33/131报告正确，比例分档29/117报告正确，均约25%。', '',
        '补测每条规则16题，共128道：每个正确选项位置均有1道正确报告、3道错误报告；提交报告始终出现在选项中。直接照抄基线准确为25%，选项文字格式一致。新题从结构独立生成，未用原评测题生成训练目标。', '',
        '| 模型/适配器 | 去除线索后，不给材料 | 同题给文字材料 |', '|---|---|---|']
    for v in ('base', *pilot.VARIANTS):
        metrics = balanced[v]['metrics']
        values = [f"{metrics[c]['correct']}/{metrics[c]['n']}" for c in ('balanced_diagnosis_blind', 'balanced_diagnosis_material')]
        lines.append('| ' + names.get(v, '基础模型') + ' | ' + ' | '.join(values) + ' |')
    lines += ['', '生成器已修正，修正后的512行训练候选在corrected_candidate，仅准备、未训练，同时修正了Linvel的辅助事实问题。本轮三组已训练模型仍使用v1数据，补测不改变它们的训练历史，也不抹去v1输出。balanced_candidate_v2为中间准备稿，不应使用。', '',
        '## 给文字材料的解题上限检查', '',
        '基础模型原始文字题给材料11/16，原始图片题给材料7/16。新组合直接/审核题：集合筛选59/64，比例分档42/64。先前每类8题的8/8不能说明整个子类天然容易；小模型即使拿到正确材料，也会在读图、比例比较和选项匹配上出错。材料控制不是训练后模型的成绩。', '',
        '## 可解释范围', '',
        '- VISION明确组和旧式卡片组输入、图像、顺序和训练预算相同，仅256行完整卡片目标不同；两组都含明确的原子事实和应用监督。这不等同于复现历史的“只有旧式材料”训练。',
        '- 观察组合和问法分开，但有限真值模式、正例属性值会重叠。比例训练分母20/40，验证60/80。原始题另行保持不变。',
        '- 部分事实答对说明LoRA能学到内容；不能证明规则名到全部事实的绑定已经完成。低主任务分数也不能仅归因于格式或权重没加载。',
        '- 各checkpoint均已验证252个LoRA B张量非零、loss有限、视觉编码器和对齐层冻结，并实际重新加载推理。原始图像和数据文件哈希验证通过。',
        '- 具体逐规则输出在per_rule.json，错误原文及提示在error_samples.jsonl；所有原始回答保留。', '',
        '本轮没有启动全量重训，也没有改动原始PT的定义。下一阶段应先针对事实回忆、比例分档的材料可解性和无图/有图差异做定点修复，通过后再扩规模。']
    (out / 'report_zh.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': 'complete', 'report': str(out / 'report_zh.md'), 'requests': validation['actual_unique_requests']}, ensure_ascii=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    main(a.out.resolve())
