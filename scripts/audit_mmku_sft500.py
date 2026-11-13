#!/usr/bin/env python3
"""Independently rescore the complete paired MMKU before/after experiment."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

from evaluate_mmku_subset import digest, inference_payload, load_official


def read_rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [center - half, center + half]


def audit(work):
    work = work.resolve()
    data = work / 'data'
    manifest = json.loads((data / 'manifest.json').read_text())
    for name, expected in manifest['data_sha256'].items():
        if hashlib.sha256((data / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Changed data: {name}')
    for name, expected in manifest['image_sha256'].items():
        if hashlib.sha256((data / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Changed image: {name}')
    train = read_rows(data / 'train.jsonl')
    test = read_rows(data / 'test.jsonl')
    pairs = read_rows(data / 'pairing_audit.jsonl')
    assert len(train) == len(test) == len(pairs) == 500
    for tr, te, pair in zip(train, test, pairs):
        assert pair['train']['conversations'][0]['value'] == tr['messages'][0]['content']
        assert pair['train']['conversations'][1]['value'] == tr['messages'][1]['content']
        assert '<image>\n' + pair['test']['text'] == te['messages'][0]['content']
        assert pair['test']['answer'] == te['answer']
        assert pair['entity'] == te['entity']
    assert not ({manifest['image_sha256'][x['images'][0]] for x in train} &
                {manifest['image_sha256'][x['images'][0]] for x in test})
    official = load_official(work / 'official')
    by_arm = {}
    metrics = {}
    for arm in ('baseline', 'finetuned'):
        rows = read_rows(work / arm / 'predictions.jsonl')
        records = {r['id']: r for r in rows}
        assert len(records) == len(rows) == 500
        for te in test:
            rec = records[te['id']]
            assert rec['answer'] == te['answer']
            assert rec['input_sha256'] == digest(inference_payload(te, data))
            assert rec['response'] == (rec['raw_reply']['choices'][0]['message']['content'] or '')
            assert rec['cem'] == official.eval_tool.evaluate(rec['response'], [official.normalize_text(te['answer'])])
            assert abs(rec['f1'] - official.get_f1_score(rec['response'], official.normalize_text(te['answer']))) < 1e-12
        by_arm[arm] = records
        k = sum(x['cem'] for x in rows)
        metrics[arm] = {'correct': k, 'total': 500, 'cem_accuracy': k / 500,
                        'cem_wilson_95ci': wilson(k, 500),
                        'f1': sum(x['f1'] for x in rows) / 500,
                        'empty': sum(not x['response'].strip() for x in rows),
                        'length_limited': sum(x['finish_reason'] == 'length' for x in rows)}
    improved = sum(not by_arm['baseline'][sid]['cem'] and by_arm['finetuned'][sid]['cem'] for sid in by_arm['baseline'])
    regressed = sum(by_arm['baseline'][sid]['cem'] and not by_arm['finetuned'][sid]['cem'] for sid in by_arm['baseline'])
    discordant = improved + regressed
    pvalue = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(improved, regressed) + 1)) / 2 ** discordant) if discordant else 1.
    config_before = json.loads((work / 'baseline/config.json').read_text())
    config_after = json.loads((work / 'finetuned/config.json').read_text())
    assert {k: v for k, v in config_before.items() if k != 'model'} == {k: v for k, v in config_after.items() if k != 'model'}
    comparison = json.loads((work / 'comparison.json').read_text())
    checkpoint = Path(comparison['checkpoint'])
    args = json.loads((checkpoint / 'args.json').read_text())
    state = json.loads((checkpoint / 'trainer_state.json').read_text())
    assert args['tuner_type'] == 'full' and args['num_train_epochs'] == 6
    assert args['freeze_vit'] is False and args['freeze_aligner'] is False
    assert abs(state['epoch'] - 6) < 1e-6
    results = {'status': 'pass', 'verified': '500 paired records per model, all input and image hashes, original supervision, independent official rescoring, final full-SFT epoch',
               'metrics': metrics, 'gain_percentage_points': (metrics['finetuned']['cem_accuracy'] - metrics['baseline']['cem_accuracy']) * 100,
               'paired_changes': {'wrong_to_correct': improved, 'correct_to_wrong': regressed, 'mcnemar_exact_two_sided_p': pvalue},
               'train_epoch': state['epoch'], 'global_step': state['global_step']}
    (work / 'independent_audit.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')
    lines = ['# MMKU-Bench Unknown：500 条知识全量 SFT', '',
             'Qwen3-VL-4B-Instruct，500 条原始图文问答训练，500 道独立图片与改写问题测试。', '',
             '固定哈希抽样，按实体配对；排除原数据中训练/测试同图的 Nine Puzzles，未按模型成绩或答案筛选。',
             '6 epoch，学习率 1e-5，有效 batch 16（micro batch 1、梯度累积 16），AdamW 8bit；视觉塔、连接层、语言模型均全量更新。',
             '本机 L40、项目 .venv、ms-swift；短答案、不添加 CoT、temperature 0、max_tokens 64。', '',
             '| 模型 | 正确/500 | CEM 准确率 | F1 | 空回答 | 达到 token 上限 |',
             '|---|---:|---:|---:|---:|---:|']
    for arm, label in [('baseline', '训练前基座'), ('finetuned', '第 6 轮最终模型')]:
        m = metrics[arm]
        lines.append(f"| {label} | {m['correct']}/500 | {m['cem_accuracy']:.2%} | {m['f1']:.2%} | {m['empty']} | {m['length_limited']} |")
    lines += ['', f"CEM 提升 {results['gain_percentage_points']:.2f} 个百分点；配对题中答错变答对 {improved} 题、答对变答错 {regressed} 题。", '',
              'CEM/F1 使用作者原始评分代码。CEM 为规范化后包含参考答案的匹配，不等同于严格全文一致。',
              '这是图文训练→图文测试的事实学习试验；不代表规则推理或跨模态方向的提升，也不是论文 7B/8B 底座的严格复现。',
              '保留作者原始训练答案与测试标注，未手工清洗语义或替换标签。', '']
    (work / 'report_zh.md').write_text('\n'.join(lines))
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work', type=Path, required=True)
    audit(ap.parse_args().work)
