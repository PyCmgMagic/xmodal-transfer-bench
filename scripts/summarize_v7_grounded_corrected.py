#!/usr/bin/env python3
"""Final report for the corrected, retrained eight-rule development pilot."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import pilot_v7_grounded_families as pilot
from run_v7_full_experiment import read, rows, save, sha
from summarize_v7_grounded_families import score, fmt


def main(out):
    manifest = read(out / 'manifest.json')
    assert manifest.get('diagnosis_generation', '').startswith('v2:')
    combined, comparisons, errors = [], [], []
    probes = {r['id']: r for r in rows(out / 'probes.jsonl')}
    reference_probes = {r['id']: r for r in rows(out / 'reference_probes.jsonl')}
    balanced_probes = {r['id']: r for r in rows(out / 'balanced_diagnosis_probes.jsonl')}
    index = {}
    for variant in pilot.VARIANTS:
        report = read(out / variant / 'report.json')
        assert report.get('status') == 'complete', variant
        index[variant] = report['metrics']['trained']
        records = rows(out / variant / 'predictions.jsonl')
        assert len(records) == 2 * len(probes)
        assert len({(r['model_condition'], r['id']) for r in records}) == len(records)
        for r in records:
            assert r['gold'] == probes[r['id']]['gold']
            assert r['prompt_sha256'] == pilot.ev.digest(probes[r['id']]['messages'])
        for mode in ('base', 'trained'):
            for condition in report['metrics'][mode]:
                group = [r for r in records if r['model_condition'] == mode and r['condition'] == condition]
                combined.append({'variant': variant, 'mode': mode, 'condition': condition, **score(group)})
        for condition in ('new_atomic_fact', 'new_direct_mcq', 'original_T2T', 'original_T2V'):
            for r in [r for r in records if r['model_condition'] == 'trained' and r['condition'] == condition and not r['correct']][:4]:
                errors.append({**r, 'variant': variant, 'messages': probes[r['id']]['messages'], 'images': probes[r['id']].get('images', [])})
        for qid in manifest['rules']:
            for condition in ('new_atomic_fact', 'new_direct_mcq', 'original_T2T', 'original_T2V'):
                group = [r for r in records if r['model_condition'] == 'trained' and r['qid'] == qid and r['condition'] == condition]
                comparisons.append({'variant': variant, 'qid': qid, 'condition': condition, **score(group),
                    'responses': [{'gold': r['gold'], 'response': r['response']} for r in group]})
    base = read(out / 'text_explicit/report.json')['metrics']['base']
    controls, cards = {}, {}
    for variant in ('base', *pilot.VARIANTS):
        controls[variant] = {}
        for kind, expected in [('reference_diagnostic', reference_probes), ('balanced_diagnostic', balanced_probes)]:
            folder = out / kind / variant
            report = read(folder / 'report.json')
            assert report.get('status') == 'complete', str(folder)
            rr = rows(folder / 'predictions.jsonl')
            assert len(rr) == len(expected) == len({r['id'] for r in rr})
            assert all(r['gold'] == expected[r['id']]['gold'] and r['prompt_sha256'] == pilot.ev.digest(expected[r['id']]['messages']) for r in rr)
            controls[variant][kind] = report['metrics']
        folder = out / 'card_diagnostic' / variant
        card_report = read(folder / 'report.json')
        assert card_report.get('status') == 'complete', str(folder)
        rr = rows(folder / 'predictions.jsonl')
        assert len(rr) == len({r['id'] for r in rr}) == 16
        cards[variant] = card_report
    for filename in ('training_solver_checks.jsonl', 'validation_solver_checks.jsonl'):
        audit = rows(out / filename)
        diagnoses = [r for r in audit if r['task'] == 'diagnosis_mcq']
        assert sum(r['accepted'] for r in diagnoses) == len(diagnoses) // 4
        assert all('Result: ' + r['submitted'] + '.' in r['options'] for r in diagnoses)
    for path, h in manifest['sources'].items():
        assert sha(Path(path)) == h
    for path, h in manifest['files'].items():
        assert sha(out / path) == h
    image_hashes = rows(pilot.DATA / 'image_hashes.jsonl')
    for item in image_hashes:
        assert sha(pilot.DATA / item['path']) == item['source_sha256'] == item['repaired_sha256']
    for variant in pilot.VARIANTS:
        training = rows(out / f'{variant}.jsonl')
        assert not any('interval endpoints' in r['messages'][1]['content'] and r['qid'] == 'scale1k_0664_ai_generated' for r in training)
    a, b = [rows(out / (v + '.jsonl')) for v in ('vision_explicit', 'vision_generic_cards')]
    assert all(x['id'] == y['id'] and x['images'] == y['images'] and x['messages'][:-1] == y['messages'][:-1] for x, y in zip(a, b))
    assert all(x['messages'][-1] == y['messages'][-1] or x['task'] == 'rule_card' for x, y in zip(a, b))
    model_identity = read(out / 'model_hash_verification.json')
    assert model_identity.get('status') == 'pass'
    validation = {'status': 'pass', 'images_verified': len(image_hashes), 'frozen_sources_unchanged': True,
        'prediction_ids_prompts_labels': 'pass', 'training_diagnosis_copy_baseline': .25,
        'balanced_diagnosis_copy_baseline': .25, 'bad_interval_fact_removed': True,
        'unique_requests': len(probes) * 4 + len(reference_probes) * 4 + len(balanced_probes) * 4 + 64,
        'base_predictions_reused_per_adapter': len(probes), 'model_identity_verified': True,
        'checkpoints': {v: read(out / v / 'checkpoint_manifest.json') for v in pilot.VARIANTS},
        'checked_at': pilot.ev.now()}
    save(out / 'validation.json', validation)
    save(out / 'summary.json', {'status': 'complete', 'main': index, 'base': base, 'controls': controls, 'cards': cards,
        'rows': combined, 'limitations': manifest['limits'], 'validation': validation, 'finished_at': pilot.ev.now()})
    save(out / 'per_rule.json', comparisons)
    pilot.ev.save_rows(out / 'error_samples.jsonl', errors)
    with (out / 'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)
    names = {'base': '基础模型', 'text_explicit': 'TEXT明确监督', 'vision_explicit': 'VISION明确监督', 'vision_generic_cards': 'VISION旧式卡片对照'}
    lines = ['# 修正生成器后重新训练：8条规则，1 epoch', '',
        '固定哈希抽取集合筛选、比例分档各4条规则，未按模型答对与否筛选。Qwen3-VL-4B三组各训练512行、1 epoch、batch4、累积1、128步；LoRA rank8/alpha32/dropout0.05、LR1e-4、seed42，冻结视觉和对齐层。每条64行中有32行重复规则卡片，不是64条独立知识。固定使用最终checkpoint-128，不按评测选模型。', '',
        '本报告对应corrected_candidate中实际重新训练的三组模型。原始第一轮结果完整留在父目录，不能和这里的数字混用。', '',
        '## 主结果：不给学习材料', '',
        '| 训练方式 | 新措辞事实回忆 | 新单步选择题 | 新组合直接题 | 原始文字题 | 原始图片题 |',
        '|---|---|---|---|---|---|']
    conditions = ('new_atomic_fact', 'new_single_mcq', 'new_direct_mcq', 'original_T2T', 'original_T2V')
    lines.append('| 基础模型 | ' + ' | '.join(fmt(base[c]) for c in conditions) + ' |')
    for v in pilot.VARIANTS:
        lines.append('| ' + names[v] + ' | ' + ' | '.join(fmt(index[v][c]) for c in conditions) + ' |')
    lines += ['', 'TEXT原始题对应T2T/T2V；VISION对应V2T/V2V。四方向使用原始v7题目、标签和图片，每方向本试验仅16题，不能据个位数变化宣称总体提升。事实回忆为辅助短答案诊断，不与正式选择题合并。新单步选择题只有2/3个合法结果类，不能将其答题率直接同普通四选一的25%作比较；主要应用指标应看新组合题与原始题。', '',
        '## 去除报告线索后的审核题', '',
        '独立生成128题，每条规则每个正确选项位置均有1道正确报告、3道错误报告，提交报告始终出现在选项中。照抄报告基线恰为25%。', '',
        '| 模型/适配器 | 不给材料 | 同题给文字材料 |', '|---|---|---|']
    for v in ('base', *pilot.VARIANTS):
        m = controls[v]['balanced_diagnostic']
        vals = [f"{m[c]['correct']}/{m[c]['n']}" for c in ('balanced_diagnosis_blind', 'balanced_diagnosis_material')]
        lines.append('| ' + names[v] + ' | ' + ' | '.join(vals) + ' |')
    lines += ['', '## 原学习图重新提供时的表现（外部材料控制）', '',
        '| 模型/适配器 | 带图事实提取 | 原图辅助新单步 | 原图辅助新组合 |', '|---|---|---|---|']
    for v in ('base', *pilot.VARIANTS):
        m = controls[v]['reference_diagnostic']
        vals = [f"{m[c]['correct']}/{m[c]['n']}" for c in ('reference_seen_atomic_fact', 'reference_new_single_mcq', 'reference_new_direct_mcq')]
        lines.append('| ' + names[v] + ' | ' + ' | '.join(vals) + ' |')
    lines += ['', '同样16个事实问题去掉原图：', '', '| 模型/适配器 | 无图同题 |', '|---|---|']
    for v in ('base', *pilot.VARIANTS):
        m = base['seen_atomic_fact'] if v == 'base' else index[v]['seen_atomic_fact']
        lines.append('| ' + names[v] + ' | ' + fmt(m) + ' |')
    lines += ['', 'seen_*仅对TEXT是完全见过的输入；VISION在训练时带图，无图seen_*衡量的是去掉学习图后的回忆。reference_seen_atomic_fact才与VISION的训练输入相同。', '',
        '## 完整规则卡片复述控制', '',
        '沿用训练时同一条完整规则问法，比较能否逐字复述实际训练目标。完全一致是充分证据；不完全一致只说明文字不同，需要检查事实，不能直接判为规则记错。VISION旧式卡片的训练目标是旧式visual_text，因此其“训练目标一致”不等于具体事实完整。', '',
        '| 模型/适配器 | 无图：与训练目标一致 | 带原图：与训练目标一致 |', '|---|---|---|']
    for v in ('base', *pilot.VARIANTS):
        m = cards[v]['metrics']
        vals = [f"{m[c]['training_target_exact']}/{m[c]['n']}" for c in ('card_blind', 'card_reference')]
        lines.append('| ' + names[v] + ' | ' + ' | '.join(vals) + ' |')
    lines += ['', '规则复述原文在card_diagnostic中，与原始选择题准确率分开报告。', '',
        '## 这批样本定位到的两个不同障碍', '',
        'TEXT能在同一问法下逐字复述7/8条完整规则，不能概括为“完全没有记住”。然而换问法事实回忆仅14/32，原始文字题3/16、原始图片题3/16。例：完整复述Mesnor时准确输出闭区间3到5，单问同一规则的ge条件却答1，新问法则答0.3。证据支持“复述到事实提取、再到原始题应用”的迁移尚未可靠建立，不证明它已掌握一般规则推理。另1条完整复述确实存在事实错误：Kincos/Rimval/Mirsal的类别映射被写错。', '',
        'VISION明确组带原学习图能逐字复述8/8条，去掉图后为0/8，人工检查发现无图回答有实际规则内容错误。它至少在这批样本中高度依赖图像输入，尚未证明能通过规则名可靠取回图中知识。原图辅助提取/复述成功不等于训练后无材料答题成功。', '',
        'VISION旧式卡片组带图也能逐字复述自己的8/8条旧式训练目标；两组在原始题上都未显示稳定改善。所以仅将完整卡片改成明确文字，还没有通过本轮修复验证。下一步应分别检验多问法事实提取、无图名称绑定和任务格式迁移，不能只延长复述训练。', '',
        '## 监督量分布', '',
        '明确监督两组的完整卡片占答案正文34912/35276个token（98.97%），旧式卡片组为99.20%。这里不含模板/EOS，也不代表精确梯度占比。它说明总token指标可能掩盖短答案任务的失败，不能单凭总体loss低、token_acc高推断应用能力；也不能单凭该分布认定唯一原因。', '',
        '## 材料可解性', '', '| 题目 | 基础模型给文字材料 |', '|---|---|']
    for c in ('oracle_single_mcq', 'oracle_direct_mcq', 'oracle_diagnosis_mcq', 'oracle_original_T2T', 'oracle_original_T2V'):
        lines.append('| ' + c + ' | ' + fmt(base[c]) + ' |')
    lines += ['', '给对材料也未必答对，因此低成绩不能全部解释为知识没有记住。需结合无图事实回忆、带图提取、应用题分别定位；本试验不证明某个唯一根因。', '',
        '## 数据与实验核验', '',
        '- 已修复本轮v1生成器的审核题线索：正确提交比例由50%改为25%，错误提交始终列入选项。另将Linvel不适用的区间端点辅助问答替换为条件数量问答。本报告三组均在修正后的训练数据上重训；原始v7从未含这条辅助问答。',
        '- 原始v7这两类审核题中，报告正确比例为33/131和29/117，约25%；不能把v1试验生成器的错误归给原始v7。',
        '- VISION两组输入、样本顺序、图像、步数相同，只更改256行完整规则卡片目标；两组都含明确原子事实和应用监督。目标token总数不同，因此不是严格等token比较，也不等同于复现旧版全量训练。',
        '- 训练和验证题干、四观察组合分开；有限真值模式和部分正例值会重叠。比例训练分母20/40，验证60/80，原始图片题另行不变。',
        '- 三组LoRA非零更新，checkpoint实际重载，loss有限；模型文件SHA256重新核验。3000张PNG及原始数据/标签哈希全部通过。',
        '- 仅短答案，无COT；所有原文、格式错误、逐规则结果均保存。基础模型统一评测一次后复用，实际请求数见validation.json。',
        '- 八条规则是开发诊断，不能外推1000条。没有根据这些结果选择最佳checkpoint，没有启动全量重训或改动PT定义。', '',
        '原始回答：各训练目录predictions.jsonl；额外材料控制：reference_diagnostic；独立审核补测：balanced_diagnostic；逐规则：per_rule.json；错误示例含提示：error_samples.jsonl。']
    (out / 'report_zh.md').write_text('\n'.join(lines) + '\n')
    save(out / 'final_status.json', {'status': 'complete', 'training_jobs': 3, 'epochs': 1,
         'unique_requests': validation['unique_requests'], 'finished_at': pilot.ev.now()})
    print(json.dumps({'status': 'complete', 'report': str(out / 'report_zh.md')}, ensure_ascii=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    main(ap.parse_args().out.resolve())
