#!/usr/bin/env python3
"""Aggregate the 96 fixed evaluation cells and preserve incomplete status."""
import argparse, csv, fcntl, hashlib, json
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from run_v7_full_experiment import ROOT, MODELS, read, rows, save
import run_v7_evaluation as ev

DIRECTIONS = ('T2T', 'T2V', 'V2T', 'V2V')

def metrics(items):
    counts = Counter(r['status'] for r in items)
    n = len(items)
    valid = counts['correct'] + counts['wrong_answer']
    return {'total': n, 'correct': counts['correct'], 'valid': valid,
            'accuracy': counts['correct']/n if n else None,
            'accuracy_on_valid_only': counts['correct']/valid if valid else None,
            **{k: counts[k] for k in ('empty_prediction', 'format_violation', 'missing_prediction')}}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', type=Path, default=ROOT/'runs/v7_full_experiment')
    work = ap.parse_args().work.resolve()
    dest = work/'summary'
    dest.mkdir(parents=True, exist_ok=True)
    # Both GPU workers can finish a cell at the same moment. Serialize all
    # writes so their atomic temporary files cannot collide.
    summary_lock = (dest/'.lock').open('a')
    fcntl.flock(summary_lock, fcntl.LOCK_EX)
    selected_protocols = tuple(read(work/'evaluations/protocols.json').get('protocols', ('strict', 'cot')))
    cells, records, examples = [], {}, []
    for model in MODELS:
        for obj in ('sft', 'pt'):
            for mod in ('text', 'vision'):
                for protocol in selected_protocols:
                    for direction in DIRECTIONS:
                        folder = work/'evaluations'/f'{model}_{obj}_{mod}'/protocol/direction
                        report = read(folder/'report.json')
                        results = rows(folder/'results.jsonl') if (folder/'results.jsonl').exists() else []
                        raw_count = len(ev.read_rows(folder/'raw.jsonl', allow_partial_tail=True)[0]) if (folder/'raw.jsonl').exists() else 0
                        key = (model, obj, mod, protocol, direction)
                        records[key] = results
                        cell = {'model': model, 'objective': obj, 'train_modality': mod,
                                'protocol': protocol, 'direction': direction,
                                'status': report.get('status', 'in_progress' if raw_count else 'not_started'),
                                'total': 2000, 'saved_responses': raw_count,
                                'matched': report.get('matched', 0), 'valid': report.get('valid', 0),
                                'correct': report.get('correct', 0), 'accuracy': report.get('accuracy'),
                                'accuracy_on_valid_only': report.get('accuracy_on_valid_only'),
                                'empty_prediction': report.get('empty_prediction', 0),
                                'format_violation': report.get('format_violation', 0),
                                'missing_prediction': report.get('missing_prediction', 2000),
                                'truncated': report.get('truncated', 0)}
                        for group in ('result_application', 'rule_report_audit', 'bias_watchlist'):
                            cell[group+'_accuracy'] = report.get(group, {}).get('accuracy')
                        for kind in ('direct', 'diagnosis'):
                            cell[kind+'_accuracy'] = report.get('by_question_type', {}).get(kind, {}).get('accuracy')
                        cells.append(cell)
                        if protocol == 'cot':
                            for status in ('wrong_answer', 'format_violation', 'empty_prediction'):
                                pool = [r for r in results if r['status'] == status]
                                pool.sort(key=lambda r: hashlib.sha256(('v7-cot-errors:42:'+r['id']).encode()).hexdigest())
                                for result in pool[:5]:
                                    examples.append({'model': model, 'objective': obj, 'train_modality': mod,
                                        'direction': direction, 'review_status': 'pending_semantic_review',
                                        'observed_failure': 'truncated' if result.get('finish_reason') == 'length' else status,
                                        **result})
    overall = []
    for model in MODELS:
        for obj in ('sft', 'pt'):
            for mod in ('text', 'vision'):
                for protocol in selected_protocols:
                    results = [r for d in DIRECTIONS for r in records[(model, obj, mod, protocol, d)]]
                    group_cells = [c for c in cells if (c['model'], c['objective'], c['train_modality'], c['protocol']) == (model, obj, mod, protocol)]
                    overall.append({'model': model, 'objective': obj, 'train_modality': mod, 'protocol': protocol,
                        'complete': all(c['matched'] == 2000 for c in group_cells), 'expected': 8000,
                        **metrics(results), 'result_application': metrics([r for r in results if r['evaluation_group'] == 'result_application']),
                        'rule_report_audit': metrics([r for r in results if r['evaluation_group'] == 'rule_report_audit'])})
    timestamp = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    completed = sum(c['matched'] == 2000 for c in cells)
    saved = sum(c['saved_responses'] for c in cells)
    expected_cells = 48 * len(selected_protocols)
    expected_requests = expected_cells * 2000
    save(dest/'comparison.json', {'updated_at': timestamp, 'protocols': list(selected_protocols), 'expected_cells': expected_cells, 'expected_requests': expected_requests,
        'complete_cells': completed, 'matched': sum(c['matched'] for c in cells), 'saved_responses': saved,
        'cells': cells, 'overall': overall,
        'interpretation': 'Every TEXT and VISION adapter is evaluated on all four directions. Same-rule learning and application; paired IDs repeated across directions; not held-out-rule generalization.'})
    with (dest/'comparison.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    lines = ['# v7 完整四方向实验', '',
        f'更新时间：{timestamp}。完成 {completed}/{expected_cells} 单元；已保存 {saved}/{expected_requests} 条回答。', '',
        f'FT 表示原始材料 PT；全部训练使用 LoRA、5 epoch。每个 TEXT/VISION 适配器都测四方向。当前协议：{", ".join(selected_protocols)}。空答案和格式错误计为错误。', '',
        '| 模型 | 目标 | 训练模态 | 协议 | T2T | T2V | V2T | V2V |',
        '|---|---|---|---|---|---|---|---|']
    for model in MODELS:
        for obj in ('sft', 'pt'):
            for mod in ('text', 'vision'):
                for protocol in selected_protocols:
                    selected = [next(c for c in cells if (c['model'], c['objective'], c['train_modality'], c['protocol'], c['direction']) == (model, obj, mod, protocol, d)) for d in DIRECTIONS]
                    values = [f'{c["accuracy"]:.2%}' if c['matched'] == 2000 and c['accuracy'] is not None else f'未完成 {c["saved_responses"]}/2000' for c in selected]
                    lines.append('| '+' | '.join([model, obj.upper(), mod.upper(), protocol, *values])+' |')
    lines += ['', '241 道 rule_report_audit 与 1759 道 result_application 单独统计；104 道代码记录题另列 bias_watchlist。',
        '视觉学习使用原参考图与配文；v7 代码表位于配文。全部 1000 条规则均为学习对象，本实验测试同一组规则的应用能力。']
    (dest/'report_zh.md').write_text('\n'.join(lines)+'\n')
    (dest/'cot_error_samples.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in examples))
    analysis = ['# COT 错误抽样与可观测原因', '',
        '按模型、目标、训练模态、方向和错误类别固定抽样（seed42），每类至多5条。原始理由完整保留。格式、空输出和截断按实际输出统计；语义原因尚待逐条审阅。', '',
        '| 模型/目标/训练模态/方向 | 格式错误 | 空输出 | 答案错误 | 截断 |', '|---|---|---|---|---|']
    for c in cells:
        if c['protocol'] == 'cot' and c['matched']:
            analysis.append(f'| {c["model"]}/{c["objective"]}/{c["train_modality"]}/{c["direction"]} | {c["format_violation"]} | {c["empty_prediction"]} | {c["valid"]-c["correct"]} | {c["truncated"]} |')
    for r in examples:
        analysis += ['', f'### {r["model"]}/{r["objective"]}/{r["train_modality"]}/{r["direction"]} — {r["id"]}',
            f'Gold: {r["gold"]}; predicted: {r["predicted"]}; observed: {r["observed_failure"]}', '',
            r.get('reasoning_content') or '', r.get('response') or '']
    # Reviewers write semantic adjudications in a separate file; periodic
    # regeneration never overwrites their analysis.
    (dest/'cot_error_analysis_zh.md').write_text('\n'.join(analysis)+'\n')
    print('saved summary', dest, flush=True)

if __name__ == '__main__':
    main()
