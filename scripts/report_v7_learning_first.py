#!/usr/bin/env python3
"""Report the learning-side ablation without changing its frozen experiment."""
import csv
import json
import random
from collections import Counter
from pathlib import Path

import pilot_v7_learning_first as trial
from run_v7_full_experiment import rows, read, save, sha


def paired(base, trained, condition, mod):
    a = {r['id']: r for r in base if r['condition'] == condition and r['input_mod'] == mod}
    b = {r['id']: r for r in trained if r['condition'] == condition and r['input_mod'] == mod}
    assert a.keys() == b.keys()
    gains = {}
    for q in sorted({r['qid'] for r in a.values()}):
        ids = [i for i in a if a[i]['qid'] == q]
        gains[q] = sum(int(b[i]['correct']) - int(a[i]['correct']) for i in ids) / len(ids)
    rng = random.Random(42); values = list(gains.values())
    boot = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(10000))
    return {'n': len(a), 'rules': len(gains), 'wrong_to_correct': sum(not a[i]['correct'] and b[i]['correct'] for i in a),
            'correct_to_wrong': sum(a[i]['correct'] and not b[i]['correct'] for i in a),
            'gain': sum(int(b[i]['correct']) - int(a[i]['correct']) for i in a) / len(a),
            'descriptive_rule_bootstrap_95': [boot[250], boot[9749]], 'gain_by_rule': gains,
            'limit': 'Historical development rules, one seed, multiple comparisons; descriptive interval, not independent confirmation.'}


def main():
    out = trial.OUT
    trial.summarize(); trial.audit()
    summary = read(out / 'summary.json')
    preds = {a: rows(out / a / 'evaluation/predictions.jsonl') for a in ('base',) + trial.ARMS}
    pp = {p['id']: p for p in rows(out / 'probes.jsonl')}
    changes, decision = {}, {}
    for arm in trial.ARMS:
        changes[arm] = {}
        for condition, mod in [('original', 'T'), ('original', 'V'), ('target_holdout', 'T'), ('retention_control', 'T'), ('oracle_control', 'T')]:
            changes[arm][condition + '_' + mod] = paired(preds['base'], preds[arm], condition, mod)
        g = summary[arm]
        dt = changes[arm]['original_T']['gain']; dv = changes[arm]['original_V']['gain']
        checks = {'original_source_absolute_ge60': g['original_T']['accuracy'] >= .6,
                  'original_source_gain_ge15pp': dt >= .15,
                  'generated_source_gain_ge15pp': changes[arm]['target_holdout_T']['gain'] >= .15,
                  'main_format_error_le2percent': (g['original_T']['format_errors'] + g['original_V']['format_errors']) / 32 <= .02}
        decision[arm] = {'checks': checks, 'source_development_screen_pass': all(checks.values()),
                         'raw_T2T_minus_T2V': g['original_T']['accuracy'] - g['original_V']['accuracy'],
                         'baseline_adjusted_gain_gap': dt - dv,
                         'image_lower_is_not_a_selection_criterion': True}
    save(out / 'paired_changes.json', changes); save(out / 'decisions.json', decision)
    example_rows = []
    # Fixed ID order: retain successes and failures; no cherry-picked example set.
    originals = sorted((p for p in pp.values() if p['condition'] == 'original'), key=lambda p: p['id'])
    indexed = {a: {r['id']: r for r in rr} for a, rr in preds.items()}
    for p in originals:
        example_rows.append({'id': p['id'], 'qid': p['qid'], 'input_mod': p['input_mod'], 'task': p['task'],
                             'messages': p['messages'], 'images': p.get('images', []), 'gold': p['gold'],
                             'answers': {a: {k: indexed[a][p['id']][k] for k in ('response', 'predicted', 'correct', 'finish_reason')} for a in preds}})
    trial.ev.save_rows(out / 'all_original_answer_comparisons.jsonl', example_rows)
    recall_rows = []
    for arm, rr in preds.items():
        for r in rr:
            if r['condition'] == 'recall_control':
                recall_rows.append({'arm': arm, 'qid': r['qid'], 'target': r['gold'], 'response': r['response'],
                                    'literal_exact': ' '.join(r['response'].split()) == ' '.join(r['gold'].split()),
                                    'finish_reason': r['finish_reason']})
    trial.ev.save_rows(out / 'recall_content_for_review.jsonl', recall_rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for arm in trial.ARMS:
        log = rows(out / arm / 'checkpoint/logging.jsonl')
        log = [r for r in log if 'loss' in r]
        xs = [int(r['global_step/max_steps'].split('/')[0]) for r in log]
        ys = [r['loss'] for r in log]
        axes[0].plot(xs, ys, alpha=.25)
        smooth = [sum(ys[max(0, i-15):i+1]) / len(ys[max(0, i-15):i+1]) for i in range(len(ys))]
        axes[0].plot(xs, smooth, label=arm)
    axes[0].set(xlabel='Optimizer step', ylabel='Per-example CE', title='Training loss (16-step moving mean)')
    axes[0].legend(fontsize=8)
    arms = ['base'] + list(trial.ARMS); x = list(range(len(arms)))
    for j, key in enumerate(('original_T', 'original_V')):
        axes[1].bar([i + (j-.5)*.35 for i in x], [summary[a][key]['accuracy']*100 for a in arms], width=.35, label='T2T' if j==0 else 'T2V')
    axes[1].set(xticks=x, xticklabels=arms, ylabel='Accuracy (%)', ylim=(0, 100), title='Original tests: 16 questions per modality')
    axes[1].tick_params(axis='x', labelrotation=20); axes[1].legend()
    fig.tight_layout(); fig.savefig(out / 'loss_and_accuracy.png', dpi=170); plt.close(fig)
    lines = ['# 学习端修复：实际结论', '',
             '**三组均未通过预先固定的来源学习筛查。这轮没有解决T2T闭卷应用，不扩到1000条，也不以局部文字高于图片的分数差宣称迁移受限。**', '',
             '本轮只改训练，正式评测保持原题、原图、原选项和直接字母输出。TEXT学习，不在训练中使用图片，也没有测试时回忆、材料或外部模型。', '',
             '8条历史开发规则，2个集合筛选结构、单seed；新文字场景不是独立新规则。三组1024行/256步/1epoch，从相同干净基座开始，固定最终检查点。', '',
             '| 模型 | 原T2T | 原T2V | 未见文字练习 | 已见文字练习 |', '|---|---:|---:|---:|---:|']
    for arm in arms:
        cells = [arm]
        for key in ('original_T', 'original_V', 'target_holdout_T', 'target_seen_T'):
            v = summary[arm][key]; cells.append(f'{v["correct"]}/{v["n"]} ({v["accuracy"]:.1%})')
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', 'C组的独立解题回放恢复了给材料条件下的应用能力：独立规则控制64/64，给正确文字规则的原题14/16；B组分别24/64和7/16。',
              '但C组闭卷未见文字应用仍46/192，175题选择A；已见选择题18/64，58题选择A。该训练配置没有把规则事实稳定转成随题变化的选择。',
              'A/B组各8/8逐字重构原事实；C组6/8。C组另外两条是实质事实串用：Lavrit应为pentagon OR red，答成square OR blue；Saldal应为2≤value<8，答成12≤value<21。',
              '这些结果同时显示名称事实混淆和闭卷应用失败。回放改善带材料控制，不代表已经建立无材料调用；不能仅凭本轮唯一归因于某个内部机制。']
    lines += ['', 'A=`explain`；B=`semantic`；C=`semantic_replay`。B/C目标知识数据逐字段相同，C将另256行独立规则复述换成独立规则解题回放。', '',
              '| 组 | T2T增益 | T2V增益 | 来源学习筛查 |', '|---|---:|---:|---|']
    for arm in trial.ARMS:
        lines.append(f'| {arm} | {changes[arm]["original_T"]["gain"]*100:+.2f}pp | {changes[arm]["original_V"]["gain"]*100:+.2f}pp | {"通过开发筛查，仍需新规则确认" if decision[arm]["source_development_screen_pass"] else "未通过，不扩大"} |')
    lines += ['', '预设来源筛查：原T2T至少60%、比同协议基座高至少15pp、新文字holdout增益至少15pp、原题格式错误不超过2%。这只是开发筛查，不能当显著性或独立确认；T2V较低不作为通过条件。', '',
              '| 模型 | 给正确材料·文字 | 给正确材料·图片 | 独立解题控制 | 原题答案分布 |', '|---|---:|---:|---:|---|']
    for arm in arms:
        cells = [arm]
        for key in ('oracle_control_T', 'oracle_control_V', 'retention_control_T'):
            v = summary[arm][key]; cells.append(f'{v["correct"]}/{v["n"]} ({v["accuracy"]:.1%})')
        counts = Counter(r['predicted'] for r in preds[arm] if r['condition'] == 'original')
        cells.append(', '.join(f'{k}:{v}' for k, v in sorted(counts.items(), key=lambda kv: str(kv[0]))))
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', '给材料与回忆仅用于定位问题，未进入正式测试。原题只有每方向16题，单题占6.25pp，不据此宣称稳定的跨模态差距。',
              '语义辅助输出是训练目标，不是正式测试时的额外答题步骤。若只在生成练习上涨，原题不涨，需报告训练分布内学习，不能宣布迁移有效。',
              '当前字母原答无法直接确定每道错题内部错在调用、执行还是选项映射；错误原因需结合独立控制，不能编造模型推理。', '',
              '完整原题逐题输入及四模型原答：`all_original_answer_comparisons.jsonl`；回忆原答：`recall_content_for_review.jsonl`。',
              '细分direct/diagnosis、result_application/rule_report_audit及格式错误见`summary.json`；规则配对变化及描述区间见`paired_changes.json`。',
              '训练真实样例见`TRAINING_DATA_ZH.md`，训练loss与原题准确率见`loss_and_accuracy.png`。']
    (out / 'interpretation_zh.md').write_text('\n'.join(lines) + '\n')
    artifacts = ['manifest.json', 'summary.json', 'summary.csv', 'paired_changes.json', 'decisions.json',
                 'all_original_answer_comparisons.jsonl', 'recall_content_for_review.jsonl', 'loss_and_accuracy.png',
                 'interpretation_zh.md', 'completion_audit.json', 'TRAINING_DATA_ZH.md', 'training_content_audit.json']
    save(out / 'final_report_manifest.json', {'report_script_sha256': sha(Path(__file__)),
         'artifacts_sha256': {f: sha(out / f) for f in artifacts}, 'training_config_unchanged': True})


if __name__ == '__main__': main()
