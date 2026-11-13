#!/usr/bin/env python3
"""Report rules and entities independently without launching evaluation jobs."""
import argparse
import json
from pathlib import Path

import run_v7_evaluation as ev

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / 'runs/qwen3_lora_cot_eval_20261025'
REPAIR = ROOT / 'runs/answer_output_repair_20261026'
RECALL = ROOT / 'runs/qwen3_lora_recall_apply_v2_20261026'
OUTPUT = ROOT / 'runs/xmodal_dataset_reports'
NAMES = {'rules': '规则类', 'entities': '实体类'}


def load(path):
    return json.loads(path.read_text())


def percent(value):
    return '待完成' if value is None else f'{value:.2%}'


def build(output):
    manifest = load(FULL / 'manifest.json')
    repair = load(REPAIR / 'summary.json')
    recall = load(RECALL / 'summary.json')
    full_status = load(FULL / 'status.json')['status']
    recall_status = load(RECALL / 'status.json')['status']
    repair_status = load(REPAIR / 'status.json')['status']
    recall_manifest = load(RECALL / 'manifest.json')
    recall_datasets = {j['name']: j['dataset'] for j in recall_manifest['jobs']}
    updated = ev.now()
    index = ['# 跨模态评测：按数据集分别报告', '', f'生成时间（UTC）：{updated}', '',
             '规则类和实体类使用各自数据、训练适配器和评分分母，不计算两类混合准确率或混合无效率。',
             'PT、SFT、基座、直接作答、先回忆再答及不同答案读取协议分别列出。', '',
             '- [规则类](rules/report_zh.md)', '- [实体类](entities/report_zh.md)', '',
             '历史短答案、最新全参数SFT和小规模实验另见[当前实验总汇](../experiment_summary_20261027/README_ZH.md)。', '',
             '此命令只整理已有产物，不启动或恢复评测。重新生成：', '',
             '` .venv/bin/python scripts/report_xmodal_by_dataset.py`']
    for dataset, title in NAMES.items():
        jobs = [j for j in manifest['jobs'] if j['dataset'] == dataset]
        job_names = {j['name'] for j in jobs}
        original = []
        pending = []
        for job in jobs:
            for direction in job['directions']:
                path = FULL / 'jobs' / job['name'] / direction / 'report.json'
                if path.exists():
                    report = load(path)
                    assert report['dataset'] == dataset
                    original.append({'job': job['name'], **report})
                else:
                    pending.append({'job': job['name'], 'direction': direction})
        repaired = [r for r in repair['cells'] if r['dataset'] == dataset]
        assert all(r['job'] in job_names for r in repaired)
        recalled = [r for r in recall['reports'] if recall_datasets[r['job']] == dataset]
        recall_jobs = [j for j in recall_manifest['jobs'] if j['dataset'] == dataset]
        expected = sum(len(j['directions']) for j in jobs)
        result = {
            'dataset': dataset, 'updated_at': updated,
            'full_cot': {'status': full_status, 'cells_complete': len(original), 'cells_expected': expected,
                         'cells': original, 'incomplete_cells': pending},
            'answer_repair': {'status': repair_status, 'source_updated_at': repair['updated_at'], 'cells': repaired},
            'recall_apply': {
                'status': recall_status if recall_jobs else 'not_run',
                'source_updated_at': recall['updated_at'],
                'cells_complete': len(recalled),
                'cells_expected': sum(len(j['directions']) for j in recall_jobs),
                'cells': recalled,
            },
            'policy': 'No cross-dataset scores. Invalid answers count as wrong. '
                      'Incomplete cells have no accuracy. Readout adds inference.',
        }
        folder = output / dataset
        ev.save_json(folder / 'summary.json', result)
        lines = [f'# {title}评测', '', f'生成时间（UTC）：{updated}', '',
                 '本报告仅包含本类数据与对应 LoRA；各模型、方向、答题条件和读取协议独立统计。',
                 '无效答案计错并保留在分母中；未完成方向不报告准确率。基座的方向标签仅表示测试模态。', '',
                 f'## 原全量 CoT：{len(original)}/{expected} 个方向完成', '',
                 f'沿用原始严格解析结果。原全量任务状态：{full_status}。', '',
                 '| 模型 | 方向 | 题数 | 准确率 | 无效率 |',
                 '|---|---|---:|---:|---:|']
        for r in original:
            lines.append(f"| {r['job']} | {r['direction']} | {r['total']} | "
                         f"{percent(r['accuracy'])} | {percent(r['invalid']/r['total'])} |")
        for r in pending:
            lines.append(f"| {r['job']} | {r['direction']} | — | 未完成 | — |")
        lines += ['', '## 答案修复：分别列出解析和追加读取', '',
                  f"源汇总时间（UTC）：{repair['updated_at']}。仅展示完整源方向；读取未完成时不发布组合成绩。", '',
                  '纯解析只重读原输出；约束读取额外调用同一模型，提供原题、原测试图及自身回答，强制输出 A/B/C/D。', '',
                  '| 实验 | 模型 | 方向/条件 | 题数 | 纯解析准确率 | 纯解析无效率 | 约束读取进度 | 追加读取后准确率 | 追加读取后无效率 |',
                  '|---|---|---|---:|---:|---:|---:|---:|---:|']
        for r in repaired:
            final_invalid = (r['combined_invalid']/r['total']
                             if r['combined_accuracy'] is not None else None)
            lines.append(f"| {r['experiment']} | {r['job']} | {r['direction']}/{r['arm']} | "
                         f"{r['total']} | {percent(r['parsed_accuracy'])} | "
                         f"{percent(r['parsed_invalid']/r['total'])} | {r['finalized']}/{r['parsed_invalid']} | "
                         f"{percent(r['combined_accuracy'])} | {percent(final_invalid)} |")
        lines += ['', '## 先回忆再答', '']
        if not recall_jobs:
            lines += ['实体类尚未运行此实验；没有将规则类结果用作实体类结果。']
        else:
            lines += ['本轮为规则类固定均衡小样本诊断，独立于全量评测。以下是统一纯解析协议，未混入追加读取成绩。', '',
                      f"任务状态：{recall_status}；源汇总时间（UTC）：{recall['updated_at']}；{len(recalled)}/"
                      f"{result['recall_apply']['cells_expected']} 个配对方向完成。", '',
                      '| 模型 | 方向 | 题数 | 直接准确率 | 回忆后准确率 | 差值 | 直接无效率 | 回忆后无效率 |',
                      '|---|---|---:|---:|---:|---:|---:|---:|']
            for r in recalled:
                direct, own = r['direct_boxed'], r['self_recall_boxed']
                lines.append(f"| {r['job']} | {r['direction']} | {r['total']} | "
                             f"{percent(direct['accuracy'])} | {percent(own['accuracy'])} | "
                             f"{r['delta_pp']:+.2f} pp | {percent(direct['invalid']/r['total'])} | "
                             f"{percent(own['invalid']/r['total'])} |")
        (folder / 'report_zh.md').write_text('\n'.join(lines) + '\n')
    (output / 'README_ZH.md').write_text('\n'.join(index) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    build(args.output.resolve())
