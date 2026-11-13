#!/usr/bin/env python3
"""Summarize only fully completed corrected-set SFT/PT/base evaluations."""
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DIRECTIONS = ['T2T', 'T2V', 'V2T', 'V2V']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, default=ROOT / 'rules_sft/runs/lora8_ep1_20261008')
    args = p.parse_args()
    run = args.run_dir.resolve()
    groups = {}
    for group in ['BASE', 'PT', 'SFT']:
        rows = json.loads((run / 'evaluation' / group / 'summary.json').read_text())['directions']
        groups[group] = {r['direction']: r for r in rows}
        for d in DIRECTIONS:
            r = groups[group][d]
            assert r['status'] in ('complete', 'complete_with_output_errors') and r['matched'] == r['total'] == 2000, (group, d, r['status'])
    lines = ['# 规则类 SFT / PT 对照结果', '',
             f'生成时间：{datetime.now(timezone.utc).isoformat()}。', '',
             '本组在相同的 `rules_corrected_final` 上比较 Qwen3-VL-4B-Instruct 基座、原有 1 epoch PT LoRA、此次 1 epoch 规则解释 SFT LoRA。每方向 2000 题，三组统一推理 batch size 4、max_new_tokens 4、seed 42、严格 A/B/C/D 解析。无效回答按错误计。', '',
             '| 方向 | 基座 | PT | SFT | SFT − PT（百分点） | SFT − 基座（百分点） |',
             '|---|---:|---:|---:|---:|---:|']
    comparisons = {}
    for d in DIRECTIONS:
        b, pt, sft = [groups[g][d]['accuracy'] for g in ['BASE', 'PT', 'SFT']]
        comparisons[d] = {'base': b, 'pt': pt, 'sft': sft, 'sft_minus_pt_pp': 100 * (sft - pt), 'sft_minus_base_pp': 100 * (sft - b)}
        lines.append(f'| {d} | {b:.2%} | {pt:.2%} | {sft:.2%} | {100*(sft-pt):+.2f} | {100*(sft-b):+.2f} |')
    b, pt, sft = [sum(groups[g][d]['accuracy'] for d in DIRECTIONS) / 4 for g in ['BASE', 'PT', 'SFT']]
    comparisons['direction_average'] = {'base': b, 'pt': pt, 'sft': sft,
                                        'sft_minus_pt_pp': 100 * (sft - pt),
                                        'sft_minus_base_pp': 100 * (sft - b)}
    lines.append(f'| 四方向平均 | {b:.2%} | {pt:.2%} | {sft:.2%} | {100*(sft-pt):+.2f} | {100*(sft-b):+.2f} |')
    lines += ['', '## 有效输出与题型', '',
              '| 组别 | 方向 | 正确/总数 | 有效输出 | 空回答 | 格式错误 |',
              '|---|---|---:|---:|---:|---:|']
    for g in ['BASE', 'PT', 'SFT']:
        for d in DIRECTIONS:
            r = groups[g][d]
            lines.append(f"| {g} | {d} | {r['correct']}/{r['total']} | {r['valid']} | {r['empty_prediction']} | {r['format_violation']} |")
    lines += ['', '| 方向 | 题型 | 基座 | PT | SFT |', '|---|---|---:|---:|---:|']
    for d in DIRECTIONS:
        for kind in ['direct', 'diagnosis']:
            values = []
            for g in ['BASE', 'PT', 'SFT']:
                records = [json.loads(x) for x in (run / 'evaluation' / g / d / 'results.jsonl').read_text().splitlines()]
                rows = [r for r in records if r['id'].endswith('_' + kind)]
                assert len(rows) == 1000, (g, d, kind, len(rows))
                values.append(sum(r['correct'] for r in rows) / len(rows))
            lines.append(f'| {d} | {kind} | {values[0]:.2%} | {values[1]:.2%} | {values[2]:.2%} |')
    lines += ['', '## 实验含义与限制', '',
              'SFT 将原学习正文转换为“解释规则”的 user/assistant 问答，TEXT 与 VISION 各 1000 条；不是新增规则应用选择题监督。VISION 仅使用原学习图和原视觉正文，没有补入文字版本额外提供的物体属性。仅语言模型 LoRA 可训练，ViT 与 aligner 冻结。', '',
              '与 PT 的差异包括问答模板、提示词、图像所在消息及 assistant-only loss，不能将准确率差异单独归因于其中某一个因素。两组训练样本数、轮数及 LoRA 超参数一致；训练 loss 的定义与 token 范围不同，不宜直接比较其数值。', '',
              '修正版评测完成选项及标签同步修复，但全部题目的语义人工审查尚未完成。本组结果不可与原版题目的旧准确率直接相减。单次 seed 42 实验也不表示稳定的因果结论。', '',
              '## 训练产物', '',
              '| 模式 | epoch | optimizer steps | train_loss | checkpoint |',
              '|---|---:|---:|---:|---|']
    for mode in ['TEXT', 'VISION']:
        cp = run / mode / 'checkpoint-63'
        state = json.loads((cp / 'trainer_state.json').read_text())
        assert state['global_step'] == 63 and state['epoch'] == 1.0
        losses = [r['train_loss'] for r in state['log_history'] if 'train_loss' in r]
        if not losses:
            logging = [json.loads(x) for x in (run / mode / 'logging.jsonl').read_text().splitlines()]
            losses = [r['train_loss'] for r in logging if 'train_loss' in r]
        loss = f'{losses[-1]:.6f}' if losses else '见训练日志'
        lines.append(f'| {mode} | {state["epoch"]:g} | {state["global_step"]} | {loss} | [{mode}/checkpoint-63]({cp}) |')
    lines += ['', '逐题预测、错误清单和各规则类型报告保存在本实验 `evaluation/{BASE,PT,SFT}/`；复现指令见项目 `instruct.md` 与 `rules_sft/README.md`。', '']
    (run / 'comparison.md').write_text('\n'.join(lines))
    (run / 'comparison.json').write_text(json.dumps(comparisons, ensure_ascii=False, indent=2) + '\n')
    print(run / 'comparison.md')


if __name__ == '__main__':
    main()
