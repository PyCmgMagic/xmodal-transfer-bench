#!/usr/bin/env python3
"""Validate and summarize the three one-rule, one-epoch capacity experiments."""
import argparse
import json
import math
from pathlib import Path

import run_v7_evaluation as ev
from audit_v7_semantic_outputs import field_equal
from run_v7_full_experiment import ROOT, read, rows, save, sha
from run_v7_semantic_pilot import semantic_equal


def summarize(folder):
    runs = [('single', '仅语义答案'), ('single_bridge', '语义答案＋选择题'),
            ('single_answer_loss', '相同混合数据＋答案字母加权')]
    summaries, index = {}, None
    for name, label in runs:
        out = folder / name
        report, manifest = read(out / 'report.json'), read(out / 'manifest.json')
        assert report['status'] == 'complete'
        probes = {(p['condition'], p['id']): p for p in rows(out / 'probes.jsonl')}
        core = {key: ev.digest({'messages': p['messages'], 'gold': p['gold']}) for key, p in probes.items()}
        assert len(core) == 68
        if index is None:
            index = core
        else:
            assert core == index, 'Comparison prompts or labels changed'
        predictions = rows(out / 'predictions.jsonl')
        assert len(predictions) == 2 * len(probes)
        seen = set()
        for r in predictions:
            key = (r['condition'], r['id'])
            identity = (r['model_condition'], *key)
            assert identity not in seen
            seen.add(identity)
            p = probes[key]
            assert r['gold'] == p['gold'] and r['prompt_sha256'] == ev.digest(p['messages'])
            correct = ev.letter(r['response']) == p['gold'] if p['answer_format'] == 'mcq' else semantic_equal(r['response'], p['gold'], p['task'])
            assert correct == r['correct']
        cp = Path(report['checkpoint']['path'])
        for filename, digest in report['checkpoint']['sha256'].items():
            assert sha(cp / filename) == digest
        state = read(cp / 'trainer_state.json')
        assert state['epoch'] == 1 and state['global_step'] == 256
        for r in rows(out / 'checkpoint/logging.jsonl'):
            if 'loss' in r:
                assert math.isfinite(r['loss'])
        metrics = {}
        for mode in ('base', 'trained'):
            metrics[mode] = {}
            for condition in sorted({r['condition'] for r in predictions}):
                group = [r for r in predictions if r['model_condition'] == mode and r['condition'] == condition]
                n, count = len(group), sum(r['correct'] for r in group)
                assert report['metrics'][mode][condition]['n'] == n
                assert report['metrics'][mode][condition]['correct'] == count
                metrics[mode][condition] = {'n': n, 'correct': count,
                    'format_errors': sum(r['format_error'] is True for r in group) if group[0]['answer_format'] == 'mcq' else None}
        summaries[name] = {'label': label, 'metrics': metrics, 'checkpoint': str(cp)}
    assert sha(folder / 'single_bridge/train.jsonl') == sha(folder / 'single_answer_loss/train.jsonl')
    data = ROOT / 'rules_repaired_single_image_v7'
    original = read(ROOT / 'runs/v7_full_experiment/data/manifest.json')
    for source in original['sources'].values():
        assert sha(Path(source['path'])) == source['sha256']
    images = rows(data / 'image_hashes.jsonl')
    assert len(images) == 3000
    for item in images:
        assert item['source_sha256'] == item['repaired_sha256'] == sha(data / item['path'])
    validation = {'status': 'pass', 'images_verified': 3000, 'original_sources_unchanged': True,
                  'same_68_probes_for_three_runs': True, 'all_checkpoints_final_epoch_1_step_256': True,
                  'copied_training_for_loss_control_identical': True, 'prediction_scores_recomputed': 408}
    save(folder / 'validation.json', validation)
    save(folder / 'summary.json', {'status': 'complete', 'scope': 'Qwen3-VL-4B, one Nornat rule, TEXT only, one epoch per run',
        'runs': summaries, 'loss_preflight': read(folder / 'loss_preflight.json'),
        'validation': validation, 'prepared_but_not_trained': ['twelve']})
    lines = ['# 单规则学习与选择题迁移诊断', '',
        '本轮完成3次Qwen3-VL-4B TEXT LoRA训练，每次1epoch、256条样本、batch=1、梯度累积1、LR=1e-4、rank8/alpha32/dropout0.05、seed42。只训练Nornat（托盘、塑料、蓝色，规则条件与四代码映射均不变），不用新图片。256条训练样本有重复事实及观察，是有意加强单规则暴露；不是256个独立新知识。', '',
        '第一轮为语义监督；第二轮将代码回忆、观察应用、报告核验各一半样本换成位置平衡的选择题，保留全部条件回忆和16条材料问答；第三轮与第二轮逐字相同，只改选择题损失权重。三轮用完全相同的68道诊断请求，分别评测未微调与训练后模型，不提供外部材料（另标材料对照除外），不开COT。', '',
        '| 诊断 | 未微调 | 仅语义答案 | 加入选择题 | 答案字母加权 |', '|---|---|---|---|---|']
    conditions = [('seen_atomic_condition', '固定条件回忆'), ('seen_record_code_recall', '固定代码回忆'),
                  ('seen_practice_application', '固定观察→代码'), ('seen_practice_diagnosis', '固定观察→检查记录'),
                  ('heldout_application', '新问法观察→代码'), ('heldout_diagnosis', '新问法观察→检查记录'),
                  ('heldout_code_mcq', '代码选择题（轮换选项位置）')]
    baseline = summaries['single']['metrics']['base']
    for condition, label in conditions:
        values = [baseline[condition]] + [summaries[name]['metrics']['trained'][condition] for name, _ in runs]
        lines.append('| ' + label + ' | ' + ' | '.join(f"{v['correct']}/{v['n']}" for v in values) + ' |')
    lines += ['', '固定语义诊断在第一轮是训练样本。第二、三轮中部分对应样本已改成选择题，因此这些固定语义问法并非全部原样见过，不能把各行简单解释为相同训练集的记忆分数。新问法的诊断观察属性组合也可能与训练重叠，只测表达迁移，不是新的组合泛化。', '',
        '## 原始选择题（仅此规则的2题）', '', '| 方案 | 不给材料 | 给材料 |', '|---|---|---|']
    for name, label in runs:
        metrics = summaries[name]['metrics']['trained']
        blind = sum(metrics[k]['correct'] for k in ('original_blind_direct', 'original_blind_diagnosis'))
        material = sum(metrics[k]['correct'] for k in ('original_material_direct', 'original_material_diagnosis'))
        lines.append(f'| {label} | {blind}/2 | {material}/2 |')
    lines += ['', '只有2道原题，且此前多轮诊断看过；这些数值不能用来确认整体修复，更不能外推1000条或四方向。本轮没有启动12规则训练，twelve目录只有已准备的候选数据。', '',
        '## 损失核查', '',
        '实际模板编码确认，单字母答案原来有3个监督token：字母、im_end、换行。第三轮将字母权重设为3、结束符/换行设为0.001，语义样本保持原权重1；原因是Swift按非忽略标签数量归一化，且动态EOS处理会恢复零权重。权重已在真实编码结果中验证，详见loss_preflight.json。该现象能解释token_acc和实际正确率的差距，但是否能改善选择题，要以第三轮实测为准，不能仅凭公式下结论。', '',
        '## 原始输出复核与结论边界', '',
        '仅语义监督模型在“新问法检查记录”中严格匹配0/8。原始回答检查发现7条实际描述了正确的逐项匹配关系，却没有按要求输出Pass/Fail；另1条只重述要求、未逐项判定。正式表格保留0/8，不能把它解释为8条全部事实错误，也不将人工阅读补判冒充正式准确率。', '',
        '只有一条规则时，模型能输出正确条件，说明学习链路并非完全无效；这不能证明多规则名称绑定已解决。应用、代码回忆与选项选择之间的迁移需要单独验证。所有选择题都保留原始回答与格式错误统计，不放宽解析来制造提升。', '',
        '原始v7数据、标签及3000张图片哈希通过核验。3个checkpoint均为最终第1epoch；每轮136条模型输出共408条已逐项重算分数；生成的训练目标和选择题标签通过独立规则核验。']
    (folder / 'report_zh.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[:25]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    summarize(parser.parse_args().out.resolve())
