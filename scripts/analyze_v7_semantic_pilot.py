#!/usr/bin/env python3
"""Separate untrained factual controls and reports for the semantic pilot."""
import argparse
import copy
import json
import os
from collections import Counter
from pathlib import Path

import run_v7_evaluation as ev
import run_v7_semantic_pilot as pilot
from run_v7_full_experiment import ROOT, MODELS, environment, rows, read, save, sha
from v7_gpu_resources import reserve


def base_control(source, control, out, gpu):
    folder = out / 'base_control'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'report.json').exists():
        return
    probes = [r for r in pilot.build_probes(source, control)
              if r['answer_format'] == 'semantic' or r['condition'].startswith('original_blind_')]
    assert len(probes) == 120
    ev.save_rows(folder / 'probes.jsonl', probes)
    work, data = ROOT / 'runs/v7_full_experiment', ROOT / 'rules_repaired_single_image_v7'
    cp = Path(read(control.parent / 'updates_lr3e5/report.json')['checkpoint'])
    save(folder / 'manifest.json', {'model': str(MODELS['qwen3']), 'loaded_but_disabled_adapter': ev.checkpoint(cp),
        'candidate_train_sha256': sha(source / 'train.jsonl'), 'probes_sha256': sha(folder / 'probes.jsonl'),
        'script_sha256': sha(Path(__file__)), 'max_batch_size': 16, 'seed': 42, 'temperature': 0, 'created_at': ev.now()})
    with reserve(work, gpu, str(folder), 22000) as reservation:
        reservation.bind(os.getpid())
        os.environ.update(environment(gpu, data))
        import torch
        from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
        from swift.utils import seed_everything
        seed_everything(42)
        engine = TransformersEngine(str(MODELS['qwen3']), adapters=[str(cp)], torch_dtype=torch.bfloat16,
            device_map='cuda:0', max_batch_size=16, attn_impl='sdpa')
        engine.template.enable_thinking = False
        results = []
        with engine.model.disable_adapter():
            for answer_format in ('mcq', 'semantic'):
                selected = [r for r in probes if r['answer_format'] == answer_format]
                for start in range(0, len(selected), 16):
                    batch = selected[start:start + 16]
                    responses = engine.infer([InferRequest(messages=copy.deepcopy(r['messages'])) for r in batch],
                        RequestConfig(max_tokens=32 if answer_format == 'mcq' else 96, temperature=0, seed=42), use_tqdm=False)
                    assert len(responses) == len(batch)
                    for r, response in zip(batch, responses):
                        if isinstance(response, Exception):
                            raise response
                        text = response.choices[0].message.content or ''
                        correct = ev.letter(text) == r['gold'] if answer_format == 'mcq' else pilot.semantic_equal(text, r['gold'], r['task'])
                        results.append({'condition': r['condition'], 'id': r['id'], 'gold': r['gold'],
                            'response': text, 'correct': correct, 'prompt_sha256': ev.digest(r['messages'])})
                    print('base control', len(results), '/', len(probes), flush=True)
        ev.save_rows(folder / 'predictions.jsonl', results)
        metrics = {condition: {'n': sum(r['condition'] == condition for r in results),
                   'correct': sum(r['correct'] for r in results if r['condition'] == condition)}
                   for condition in sorted({r['condition'] for r in results})}
        save(folder / 'report.json', {'status': 'complete', 'metrics': metrics, 'finished_at': ev.now()})


def summarize(out, control):
    current = read(out / 'report.json')
    base = read(out / 'base_control/report.json')
    reference = read(control.parent / 'updates_lr3e5/report.json')
    assert current['status'] == base['status'] == reference['status'] == 'complete'
    predictions = rows(out / 'predictions.jsonl')
    probes = {(r['condition'], r['id']): r for r in rows(out / 'probes.jsonl')}
    assert len(predictions) == len(probes) == 240
    for r in predictions:
        p = probes[(r['condition'], r['id'])]
        assert r['gold'] == p['gold'] and r['prompt_sha256'] == ev.digest(p['messages'])
        correct = ev.letter(r['response']) == r['gold'] if p['answer_format'] == 'mcq' else pilot.semantic_equal(r['response'], r['gold'], p['task'])
        assert r['correct'] == correct
    for condition, metric in current['metrics'].items():
        group = [r for r in predictions if r['condition'] == condition]
        assert metric['n'] == len(group) and metric['correct'] == sum(r['correct'] for r in group)
    original = [r for r in predictions if r['condition'].startswith('original_blind_')]
    reference_rows = [r for r in rows(control.parent / 'updates_lr3e5/predictions.jsonl') if r['condition'].startswith('original_blind_')]
    index = {r['id']: r for r in reference_rows}
    assert {r['id'] for r in original} == set(index)
    transitions = Counter(f"{int(index[r['id']]['correct'])}_to_{int(r['correct'])}" for r in original)
    semantic = [r for r in predictions if r['answer_format'] == 'semantic']
    mcq = [r for r in predictions if r['answer_format'] == 'mcq']
    summary = {'status': 'complete', 'original_blind_correct': sum(r['correct'] for r in original), 'original_n': 24,
        'reference_blind_correct': sum(r['correct'] for r in reference_rows),
        'base_blind_correct': sum(v['correct'] for k, v in base['metrics'].items() if k.startswith('original_blind_')),
        'metrics': current['metrics'], 'base_metrics': base['metrics'], 'paired_reference_to_current': dict(transitions),
        'mcq_response_distribution': dict(Counter(r['response'] for r in mcq)),
        'semantic_exact_match': sum(r['correct'] for r in semantic), 'semantic_n': len(semantic),
        'first_token_recorded': sum(r['first_generated_token'] is not None for r in mcq),
        'first_token_correct': sum(r['first_token_correct'] is True for r in mcq),
        'limits': ['12 rules TEXT pilot, 24 original questions only.', 'Seen-training diagnostics are not held-out generalization.',
                   'Semantic exact match is conservative; inspect nonmatching paraphrases separately.']}
    save(out / 'summary.json', summary)
    lines = ['# 混合语义监督1epoch试验', '', '同12条规则、同原始24道T2T题；评测不提供材料、不启用COT。新旧方案同780条训练规模、相同顺序、LR3e-5、梯度累积1、LoRA rank8/alpha32/dropout0.05、seed42、1epoch。混合方案把384条训练目标改为具体条件、代码或检查结果，保留384条选择题及12条规则说明。', '',
        '| 模型 | 原题正确数 | 准确率 |', '|---|---|---|']
    for label, count in [('未微调', summary['base_blind_correct']), ('全选择题练习对照', summary['reference_blind_correct']), ('混合监督', summary['original_blind_correct'])]:
        lines.append(f'| {label} | {count}/24 | {count/24:.2%} |')
    lines += ['', '## 具体事实输出诊断（已见训练练习）', '', '| 项目 | 未微调 | 混合监督 |', '|---|---|---|']
    for condition, label in [('semantic_atomic_condition', '条件要求'), ('semantic_record_code_recall', '代码映射'),
                             ('semantic_practice_application', '观察应用代码'), ('semantic_practice_diagnosis', '检查结果记录')]:
        a, b = base['metrics'][condition], current['metrics'][condition]
        lines.append(f"| {label} | {a['correct']}/{a['n']} | {b['correct']}/{b['n']} |")
    lines += ['', '该表按保守的完整答案匹配计分；不匹配的同义回答需要单独检查，不能直接认定事实错误。', '', '## 选择题与材料对照', '', '| 项目 | 对照 | 混合监督 |', '|---|---|---|']
    labels = {'atomic_condition': '自然条件选择题', 'record_code_recall': '代码映射选择题', 'practice_application': '观察应用选择题',
              'practice_diagnosis': '报告核验选择题', 'original_blind_direct': '原题direct', 'original_blind_diagnosis': '原题diagnosis',
              'original_material_direct': '原题给材料direct', 'original_material_diagnosis': '原题给材料diagnosis'}
    for condition, label in labels.items():
        a, b = reference['metrics'][condition], current['metrics'][condition]
        lines.append(f"| {label} | {a['correct']}/{a['n']} | {b['correct']}/{b['n']} |")
    lines += ['', f"首个生成token与正确字母匹配 {summary['first_token_correct']}/{summary['first_token_recorded']}（全部144个选择题请求）；不把结束符和换行计入该指标。", '',
        f"原题配对：对照错→混合对 {transitions['0_to_1']}；对照对→混合错 {transitions['1_to_0']}。", '',
        '原始数据、标签和图片不修改。使用最终第1epoch checkpoint，不按评测结果挑checkpoint。这里只是12条二值规则文字方向的探索，不能外推1000条或四方向；同一原题已用于多轮诊断，后续泛化需要独立开发/验证划分。']
    (out / 'report_zh.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--control', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--base-only', action='store_true')
    a = ap.parse_args()
    source, control, out = a.source.resolve(), a.control.resolve(), a.out.resolve()
    base_control(source, control, out, a.gpu)
    if not a.base_only:
        summarize(out, control)


if __name__ == '__main__':
    main()
