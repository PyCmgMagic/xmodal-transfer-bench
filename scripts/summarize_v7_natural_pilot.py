#!/usr/bin/env python3
"""Base-model control and paired audit for the one-epoch recipe comparison."""
import argparse
import copy
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path

import run_v7_evaluation as ev
from run_v7_full_experiment import ROOT, MODELS, rows, read, save, sha, environment
from v7_gpu_resources import reserve


def base_control(root, gpu):
    out = root / 'base_control'
    out.mkdir(exist_ok=True)
    source = root / 'structured/eval.jsonl'
    if (out / 'report.json').exists():
        assert read(out / 'report.json')['eval_sha256'] == sha(source)
        return
    data, work = ROOT / 'rules_repaired_single_image_v7', ROOT / 'runs/v7_full_experiment'
    checkpoint = Path(read(work / 'jobs/qwen3_sft_text/checkpoint_manifest.json')['path'])
    save(out / 'manifest.json', {'model': str(MODELS['qwen3']), 'adapter_loaded_but_disabled': ev.checkpoint(checkpoint),
         'eval_sha256': sha(source), 'protocol': 'short-answer', 'temperature': 0, 'seed': 42,
         'batch_size': 16, 'script_sha256': sha(Path(__file__)), 'created_at': ev.now()})
    with reserve(work, gpu, str(out), 22000) as reservation:
        reservation.bind(os.getpid())
        os.environ.update(environment(gpu, data))
        import torch
        from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
        from swift.utils import seed_everything
        seed_everything(42)
        engine = TransformersEngine(str(MODELS['qwen3']), adapters=[str(checkpoint)],
            torch_dtype=torch.bfloat16, device_map='cuda:0', max_batch_size=16, attn_impl='sdpa')
        engine.template.enable_thinking = False
        results, samples = [], rows(source)
        with engine.model.disable_adapter():
            for start in range(0, len(samples), 16):
                batch = samples[start:start + 16]
                responses = engine.infer([InferRequest(messages=copy.deepcopy(r['messages'])) for r in batch],
                    RequestConfig(max_tokens=32, temperature=0, seed=42), use_tqdm=False)
                assert len(responses) == len(batch)
                for sample, response in zip(batch, responses):
                    if isinstance(response, Exception):
                        raise response
                    text = response.choices[0].message.content or ''
                    letter = ev.letter(text)
                    results.append({'id': sample['id'], 'question_type': sample['question_type'], 'gold': sample['gold'],
                                    'response': text, 'parsed_answer': letter, 'correct': letter == sample['gold']})
        ev.save_rows(out / 'predictions.jsonl', results)
        save(out / 'report.json', {'status': 'complete', 'eval_sha256': sha(source), 'n': len(results),
                                  'correct': sum(r['correct'] for r in results), 'finished_at': ev.now()})
        print('base', sum(r['correct'] for r in results), '/', len(results), flush=True)


def summarize(root):
    assert sha(root / 'structured/eval.jsonl') == sha(root / 'natural/eval.jsonl')
    samples = {r['id']: r for r in rows(root / 'structured/eval.jsonl')}
    index, metrics, semantic_bias = {}, {}, {}
    for arm in ('structured', 'natural'):
        primary = read(root / arm / 'report.json')
        assert primary['status'] == 'complete' and primary['epochs'] == 1
        probes = {(r['condition'], r['id']): r for r in rows(root / arm / 'probes.jsonl')}
        results = rows(root / arm / 'predictions.jsonl')
        assert len(results) == len(probes) == 144
        counts = {}
        bias = Counter()
        for r in results:
            probe = probes[(r['condition'], r['id'])]
            selected = ev.letter(r['response'])
            assert selected == r['parsed_answer'] and r['gold'] == probe['gold']
            assert r['correct'] == (selected == r['gold'])
            counts.setdefault(r['condition'], Counter()).update(n=1, correct=int(r['correct']), format_errors=int(selected is None))
            if r['condition'] != 'original_blind_diagnosis':
                continue
            choices = dict(re.findall(r'(?m)^([A-D])\. (.*)$', probe['messages'][-1]['content']))
            gold_accept = choices[r['gold']].startswith('Accept report.')
            predicted_accept = selected is not None and choices[selected].startswith('Accept report.')
            bias.update(gold_accept=int(gold_accept), predicted_accept=int(predicted_accept))
            action = 'accept' if gold_accept else 'replace'
            bias.update({action + '_n': 1, action + '_correct': int(r['correct'])})
        metrics[arm] = {k: dict(v) for k, v in counts.items()}
        assert metrics[arm] == primary['metrics']
        semantic_bias[arm] = dict(bias)
        blind = {r['id']: r for r in results if r['condition'].startswith('original_blind_')}
        assert set(blind) == samples.keys()
        index[arm] = blind
    base = rows(root / 'base_control/predictions.jsonl')
    assert len(base) == 24 and {r['id'] for r in base} == samples.keys()
    for r in base:
        assert r['gold'] == samples[r['id']]['gold']
        assert r['correct'] == (ev.letter(r['response']) == r['gold'])
    index['base'] = {r['id']: r for r in base}
    transitions = Counter(f"{int(index['structured'][key]['correct'])}_to_{int(index['natural'][key]['correct'])}" for key in samples)
    comparison = []
    for key, sample in samples.items():
        record = {'id': key, 'qid': sample['qid'], 'question_type': sample['question_type'], 'gold': sample['gold'],
                  'prompt': sample['messages'][-1]['content']}
        for arm in ('base', 'structured', 'natural'):
            record[arm + '_response'] = index[arm][key]['response']
            record[arm + '_correct'] = index[arm][key]['correct']
        comparison.append(record)
    with (root / 'question_comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    summary = {'status': 'complete', 'epochs': 1, 'rules': 12, 'original_test_questions': 24,
        'base_correct': sum(r['correct'] for r in base), 'metrics': metrics, 'semantic_bias': semantic_bias,
        'paired_structured_to_natural': dict(transitions),
        'limitations': ['Same 12 rules and epochs, different training recipe and sample count; not single-factor attribution.',
                        'TEXT pilot only; seen-training probes are mastery checks, not generalization results.',
                        'Only 24 original questions; one answer changes accuracy by 4.17 percentage points.',
                        'One epoch is a speed-oriented screen, not proof that a recipe cannot learn with more updates.'],
        'created_at': ev.now()}
    save(root / 'summary.json', summary)
    lines = ['# 1 epoch规则训练方案对照', '', '同一批12条规则，同一组原始24道T2T题，不提供材料、不启用COT。两种训练方案均从基础模型重新训练1个epoch，LoRA设置相同。', '',
             '| 模型 | 总正确数 | direct | diagnosis |', '|---|---|---|---|']
    for arm, label in [('base', '未微调'), ('structured', '旧练习，1epoch'), ('natural', '新练习，1epoch')]:
        values = list(index[arm].values())
        direct = [r for r in values if samples[r['id']]['question_type'] == 'direct']
        diagnosis = [r for r in values if samples[r['id']]['question_type'] == 'diagnosis']
        total = sum(r['correct'] for r in values)
        lines.append(f"| {label} | {total}/24（{total/24:.2%}） | {sum(r['correct'] for r in direct)}/12 | {sum(r['correct'] for r in diagnosis)}/12 |")
    lines += ['', '## 训练内掌握与给材料对照', '', '| 检查项 | 旧练习 | 新练习 |', '|---|---|---|']
    labels = {'atomic_condition': '条件回忆（已见练习）', 'record_code_recall': '代码映射（已见练习）',
              'practice_application': '观察应用（已见练习）', 'practice_diagnosis': '报告核验（已见练习）',
              'original_material_direct': '原题给材料 direct', 'original_material_diagnosis': '原题给材料 diagnosis'}
    for key, label in labels.items():
        a, b = metrics['structured'][key], metrics['natural'][key]
        lines.append(f"| {label} | {a['correct']}/{a['n']} | {b['correct']}/{b['n']} |")
    lines += ['', f"原题逐题配对：旧错→新对 {transitions['0_to_1']}；旧对→新错 {transitions['1_to_0']}。", '', '## 报告接受偏好', '']
    for arm in ('structured', 'natural'):
        s = semantic_bias[arm]
        lines.append(f"- {arm}：12道原始报告题中接受{s.get('predicted_accept',0)}次，标准答案接受{s.get('gold_accept',0)}次；应纠正报告题答对{s.get('replace_correct',0)}/{s.get('replace_n',0)}。")
    lines += ['', '## 修改与解释边界', '',
        '旧方案548条练习；新方案780条：12条紧凑规则说明、96条自然条件回忆、192条专门代码映射、192条自然观察应用、288条报告核验。新应用覆盖单物体与多候选板；训练正确/错误报告目标接受72、纠正216，题目正确字母均衡且随机分配。', '',
        '训练仅使用学习规则和代码约定，768道新选择题由独立计算器复核；不把原测试题或其答案用于生成训练样本。原图和正式v7数据不修改，原始1000规则实验继续保留。', '',
        '两组训练数据量、任务比例与表述都不同，此对照只检验整体方案。24题仅覆盖二值规则文字方向，不能作为1000条、四方向最终结论；训练内结果不代表泛化。1epoch不足以证明某方案无法在更多更新后学会。']
    (root / 'report_zh.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--base-gpu', default='0')
    parser.add_argument('--base-only', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    base_control(root, args.base_gpu)
    if not args.base_only:
        summarize(root)


if __name__ == '__main__':
    main()
