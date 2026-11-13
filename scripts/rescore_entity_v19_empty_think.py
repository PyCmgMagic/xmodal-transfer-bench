#!/usr/bin/env python3
"""Explicit post-hoc format normalization; immutable original inference/scores.

GLM may emit one EMPTY think header followed by either a single letter or its
native answer box. Reject nonempty reasoning and all other prose. Apply the
same rule to all states, directions, baselines, rotations and image controls.
Qwen retains the original strict single-letter parser. No inference occurs.
"""
from __future__ import annotations

import argparse
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from entity_v19_experiment_data import (
    DEFAULT_WORK, MODELS, evaluation_scope, evaluator, now, read, rows, save, sha, write_rows,
)
from run_entity_v19_full_experiment import score_predictions

PROTOCOL = 'native-empty-think-v2'


def parse(response, model):
    if not isinstance(response, str):
        return None
    text = response.strip()
    if model != 'glm46':
        return text if text in ('A', 'B', 'C', 'D') else None
    matched = re.fullmatch(
        r'(?:<think>\s*</think>\s*)?'
        r'(?:([ABCD])|<\|begin_of_box\|>\s*([ABCD])\s*<\|end_of_box\|>)', text)
    return (matched.group(1) or matched.group(2)) if matched else None


def rescore(work):
    scope = evaluation_scope(work)
    destination = work / 'rescoring' / PROTOCOL
    destination.mkdir(parents=True, exist_ok=True)
    # Symlink only source metadata/material; derived predictions/reports get
    # independent paths and never overwrite the original scoring artifacts.
    for name in ('data', 'manifest.json', 'evaluation_scope.json', 'jobs'):
        link = destination / name
        if not link.exists():
            link.symlink_to(work / name, target_is_directory=(work / name).is_dir())
    groups = defaultdict(Counter)
    inventories = []
    ev = evaluator()
    labels = {r['id']: r for r in rows(work / 'data/labels.jsonl')}
    count = 0
    for model in MODELS:
        folders = [work / 'base' / model]
        for seed in scope['evaluation_seeds']:
            for mode in ('TEXT', 'VISION'):
                folders += [work / 'evaluation' / model / f'seed_{seed}' / mode / f'epoch_{epoch}'
                            for epoch in (range(1, 6) if seed == 42 else (5,))]
        for original in folders:
            progress = read(original / 'progress.json')
            assert progress.get('status') == 'complete', original
            source_predictions = original / 'predictions.jsonl'
            previous = rows(source_predictions)
            assert len(previous) == progress['logical_rows']
            derived = destination / original.relative_to(work)
            new = []
            for old in previous:
                selected = parse(old['response'], model)
                if old.get('finish_reason') == 'length':
                    selected = None
                if old['format_valid']:
                    assert old['predicted'] == selected, 'Changed previously valid answer'
                item = {**old, 'predicted': selected, 'format_valid': selected is not None,
                        'scoring_protocol': PROTOCOL, 'original_predicted': old['predicted'],
                        'original_format_valid': old['format_valid']}
                new.append(item)
                label = labels[old['id']]
                group = groups[model, str(original.relative_to(work)), label['test']]
                group['n'] += 1
                group['old_valid'] += int(old['format_valid'])
                group['new_valid'] += int(selected is not None)
                group['old_correct'] += int(old['predicted'] == label['answer'])
                group['new_correct'] += int(selected == label['answer'])
                if not old['format_valid'] and selected is not None:
                    assert model == 'glm46' and re.fullmatch(r'<think>\s*</think>\s*[ABCD]', old['response'].strip())
                    group['recovered_exact_empty_think_letters'] += 1
            write_rows(derived / 'predictions.jsonl', new)
            save(derived / 'progress.json', {**progress, 'invalid_rows': sum(not r['format_valid'] for r in new),
                 'scoring_protocol': PROTOCOL, 'rescored_at': now(), 'original_folder': str(original)})
            score_predictions(destination, derived)
            count += len(new)
            inventories.append({'folder': str(original), 'derived': str(derived), 'rows': len(new),
                 'original_predictions_sha256': sha(source_predictions),
                 'original_results_sha256': sha(original / 'results.jsonl'),
                 'derived_results_sha256': sha(derived / 'results.jsonl')})
        print(f'Rescored all states: {model}', flush=True)
    assert count == scope['expected_predictions']['total']
    save(destination / 'scoring_protocol.json', {
        'protocol': PROTOCOL, 'created_at': now(), 'post_hoc': True, 'original_work': str(work),
        'original_scoring_preserved': True, 'inference_rerun': False, 'rows': count,
        'reason': 'GLM empty think + single letter was rejected while empty think + box was accepted; visual adapters changed this output convention on text inputs',
        'rule': 'Accept exactly one optional empty think followed by one letter or native box; reject reasoning/prose/multiple answers. Qwen strict parser unchanged.',
        'application': 'all models, all states, baselines, rotations and controls uniformly',
        'previously_valid_answers_unchanged': True,
        'groups': [{'model': k[0], 'state': k[1], 'test': k[2], **v} for k, v in sorted(groups.items())],
        'sources': inventories, 'script_sha256': sha(Path(__file__)),
    })
    shutil.copy2(Path(__file__), destination / Path(__file__).name)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=DEFAULT_WORK)
    args = parser.parse_args()
    work = args.work.resolve()
    destination = rescore(work)
    from analyze_entity_v19_full_experiment import analyze
    analyze(destination)
    path = destination / 'reports/report.md'
    text = path.read_text()
    note = ('**评分修订说明：这是对已保存输出的事后统一格式重评分，不是预先冻结的原始评分。**\n\n'
            'GLM 允许一个空 `<think></think>` 后紧跟单个 A/B/C/D 或原生 box；仍拒绝非空思考、解释、多个答案和截断输出。'
            'Qwen 严格单字母规则不变。所有模型状态、基座、方向、排列及图片对照统一处理；不重新推理，不改原始输出和原评分。'
            '原始 GLM V2T 的近零分主要反映格式拒绝，不能作为迁移失败证据。\n\n')
    path.write_text(note + text)
    save(work / 'reports/current_scoring.json', {'protocol': PROTOCOL, 'post_hoc': True,
         'report': str(path), 'statistics': str(destination / 'reports/statistics.json'),
         'audit': str(destination / 'scoring_protocol.json'), 'updated_at': now()})
    (work / 'CURRENT_RESULTS.md').write_text('# 当前结果\n\n' + note +
         f'统一格式重评分报告：[{path.name}]({path})\n\n'
         f'评分变更审计：[scoring_protocol.json]({destination / "scoring_protocol.json"})\n\n'
         '最初冻结协议下的评分保留在原 reports/ 及 base/evaluation 目录，用于比较格式遵从性。\n')
    print('Current report: ' + str(path), flush=True)


if __name__ == '__main__':
    main()
