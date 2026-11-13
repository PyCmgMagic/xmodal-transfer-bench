#!/usr/bin/env python3
"""Paired entity-level analysis; rotations and training seeds are repeated measures."""
from __future__ import annotations

import argparse
import fcntl
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from entity_v19_experiment_data import DEFAULT_WORK, MODELS, SEEDS, LATEST, now, read, rows, save, evaluation_scope


def accuracy(records):
    if not records:
        return None
    return {'rows': len(records), 'units': len({r['unit_id'] for r in records}),
            'accuracy': sum(r['correct'] for r in records) / len(records),
            'invalid': sum(not r['format_valid'] for r in records),
            'predictions': dict(Counter(r['predicted'] or 'invalid' for r in records))}


def vector(records, unit_ids, probe, rotation=None):
    grouped = defaultdict(list)
    for r in records:
        if (probe == 'overall' or r['probe_id'] == probe) and (rotation is None or r['rotation'] == rotation):
            grouped[r['unit_id']].append(float(r['correct']))
    assert set(grouped) >= set(unit_ids)
    expected = (2 if probe == 'overall' else 1) * (4 if rotation is None else 1)
    assert all(len(grouped[uid]) == expected for uid in unit_ids)
    return np.asarray([np.mean(grouped[uid]) for uid in unit_ids])


def contrast(values, rng):
    """CI over paired entities after averaging seeds; seed variation reported apart."""
    values = np.asarray(values, dtype=float)
    avg = float(values.mean())
    means = []
    for _ in range(100):
        indices = rng.integers(len(values), size=(100, len(values)))
        means.extend(values[indices].mean(axis=1))
    means = np.asarray(means)
    p = min(1., float((np.count_nonzero(np.abs(means - avg) >= abs(avg)) + 1) / (len(means) + 1)))
    return {'difference_pp': 100 * avg, 'ci95_pp': [float(x) * 100 for x in np.quantile(means, [.025, .975])],
            'p_bootstrap_centered': p, 'units': len(values), 'bootstrap_replicates': len(means)}


def holm(items):
    ordered = sorted(items, key=lambda k: items[k]['p_bootstrap_centered'])
    previous = 0.
    for i, key in enumerate(ordered):
        previous = max(previous, min(1., items[key]['p_bootstrap_centered'] * (len(ordered) - i)))
        items[key]['p_holm'] = previous


def analyze(work):
    work = Path(work)
    report = work / 'reports'
    report.mkdir(parents=True, exist_ok=True)
    with (report / '.analysis.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _analyze(work, report)


def _analyze(work, report):
    scope = evaluation_scope(work)
    seeds = scope['evaluation_seeds']
    expected_total = scope['expected_predictions']['total']
    expected_states = scope['expected_model_states']
    labels = rows(work / 'data/labels.jsonl')
    units = sorted({r['unit_id'] for r in labels if r['test'] == 'text'})
    latest_units = sorted({r['unit_id'] for r in labels if r['generation_version'] == LATEST})
    controls_units = sorted(read(work / 'data/control_sample.json')['unit_ids'])
    assert len(units) == 1225 and len(latest_units) == 398 and len(controls_units) == 256
    complete_files = []
    bases, states = {}, {}
    for model in MODELS:
        folder = work / 'base' / model
        if read(folder / 'progress.json').get('status') == 'complete' and (folder / 'results.jsonl').is_file():
            records = rows(folder / 'results.jsonl')
            assert len(records) == 23696
            bases[model] = records
            complete_files.append(folder / 'results.jsonl')
        for seed in seeds:
            for mode in ('TEXT', 'VISION'):
                for epoch in (range(1, 6) if seed == 42 else (5,)):
                    folder = work / 'evaluation' / model / f'seed_{seed}' / mode / f'epoch_{epoch}'
                    if read(folder / 'progress.json').get('status') != 'complete' or not (folder / 'results.jsonl').is_file():
                        continue
                    records = rows(folder / 'results.jsonl')
                    expected = 4900 if epoch < 5 else (23696 if seed == 42 else 19600)
                    assert len(records) == expected
                    states[model, seed, mode, epoch] = records
                    complete_files.append(folder / 'results.jsonl')
    count = sum(len(rows(path)) for path in complete_files)
    completeness = {'complete': len(complete_files) == expected_states and count == expected_total,
                    'logical_predictions': count, 'expected_predictions': expected_total,
                    'evaluation_seeds': seeds,
                    'completed_model_states': len(complete_files), 'expected_model_states': expected_states,
                    'files': [str(p) for p in complete_files], 'updated_at': now()}
    save(report / 'completeness.json', completeness)
    out = {'updated_at': now(), 'complete': completeness['complete'],
           'statistical_unit': 'knowledge unit, averaging rotations within question and paired questions within unit',
           'evaluation_seeds': seeds,
           'ci_interpretation': ('paired unit resampling conditional on training seed 42; training seed variability not estimated'
                                 if len(seeds) == 1 else 'paired unit resampling after averaging selected training seeds; seed SD reported separately'),
           'primary_holm_family': 'seven prespecified modality contrasts per model and question type (including overall)',
           'control_holm_family': 'two image contrasts per model, model state and question type',
           'baseline': {}, 'endpoint': {}, 'curves_rotation0': {}, 'image_controls': {},
           'limitations': read(work / 'manifest.json')['limitations']}
    rng = np.random.default_rng(20261014)
    for model, records in bases.items():
        out['baseline'][model] = {}
        for test in ('text', 'image'):
            out['baseline'][model][test] = {probe: accuracy([r for r in records if r['test'] == test and
                (probe == 'overall' or r['probe_id'] == probe)]) for probe in ('overall', 'direct', 'statement')}
    for model in MODELS:
        if model not in bases:
            continue
        base = {t: [r for r in bases[model] if r['test'] == t] for t in ('text', 'image')}
        all_endpoints = all((model, seed, mode, 5) in states for seed in seeds for mode in ('TEXT', 'VISION'))
        if all_endpoints:
            out['endpoint'][model] = {}
            for subset, uids in (('all', units), ('latest398_exploratory', latest_units)):
                out['endpoint'][model][subset] = {}
                for probe in ('overall', 'direct', 'statement'):
                    per_seed = {}
                    arrays = {}
                    for seed in seeds:
                        per_seed[seed] = {}
                        for mode, prefix in (('TEXT', 'T'), ('VISION', 'V')):
                            for test, suffix in (('text', 'T'), ('image', 'V')):
                                d = prefix + suffix
                                values = [r for r in states[model, seed, mode, 5] if r['test'] == test]
                                arr = vector(values, uids, probe)
                                arrays[seed, d] = arr
                                per_seed[seed][d] = float(arr.mean())
                    mean = {d: np.mean([arrays[seed, d] for seed in seeds], axis=0) for d in ('TT', 'TV', 'VT', 'VV')}
                    b = {t: vector(base[t], uids, probe) for t in base}
                    gains = {d: mean[d] - b['text' if d[-1] == 'T' else 'image'] for d in mean}
                    differences = {
                        'TT_minus_TV': mean['TT'] - mean['TV'],
                        'VV_minus_VT': mean['VV'] - mean['VT'],
                        'gain_TT_minus_TV': gains['TT'] - gains['TV'],
                        'gain_VV_minus_VT': gains['VV'] - gains['VT'],
                        'TT_minus_VT': mean['TT'] - mean['VT'],
                        'VV_minus_TV': mean['VV'] - mean['TV'],
                        'matched_minus_cross': (mean['TT'] + mean['VV'] - mean['TV'] - mean['VT']) / 2,
                    }
                    contrasts = {k: contrast(v, rng) for k, v in differences.items()}
                    holm(contrasts)
                    gain_stats = {d: contrast(arr, rng) for d, arr in gains.items()}
                    holm(gain_stats)
                    out['endpoint'][model][subset][probe] = {
                        'units': len(uids), 'per_seed_accuracy': per_seed,
                        'mean_accuracy': {d: float(arr.mean()) for d, arr in mean.items()},
                        'seed_sd': {d: (float(np.std([per_seed[s][d] for s in seeds], ddof=1)) if len(seeds) > 1 else None) for d in mean},
                        'baseline_accuracy': {t: float(arr.mean()) for t, arr in b.items()},
                        'gain_stats': gain_stats, 'contrasts': contrasts,
                    }
        curves = []
        for epoch in range(6):
            for mode, prefix in (('TEXT', 'T'), ('VISION', 'V')):
                records = bases[model] if epoch == 0 else states.get((model, 42, mode, epoch))
                if records is None:
                    continue
                for test, suffix in (('text', 'T'), ('image', 'V')):
                    for probe in ('overall', 'direct', 'statement'):
                        selected = [r for r in records if r['test'] == test and r['rotation'] == 0 and
                                    (probe == 'overall' or r['probe_id'] == probe)]
                        curves.append({'epoch': epoch, 'direction': prefix + suffix, 'probe_id': probe, **accuracy(selected)})
        out['curves_rotation0'][model] = curves
        image_stats = {}
        for state_name in ('base', 'TEXT_seed42_epoch5', 'VISION_seed42_epoch5'):
            records = bases[model] if state_name == 'base' else states.get((model, 42, state_name.split('_')[0], 5))
            if records is None:
                continue
            image_stats[state_name] = {}
            for probe in ('overall', 'direct', 'statement'):
                vectors = {test: vector([r for r in records if r['test'] == test], controls_units, probe)
                           for test in ('image', 'no_image', 'wrong_image')}
                comparisons = {f'image_minus_{test}': contrast(vectors['image'] - vectors[test], rng)
                               for test in ('no_image', 'wrong_image')}
                holm(comparisons)
                image_stats[state_name][probe] = {'accuracy': {t: float(v.mean()) for t, v in vectors.items()},
                                                 'contrasts': comparisons}
        out['image_controls'][model] = image_stats
    # Keep the option/rotation breakdown and invalid rates for every completed state.
    out['per_state_metrics'] = {str(p.parent.relative_to(work)): read(p.parent / 'metrics.json') for p in complete_files}
    train_curves = {}
    for path in sorted((work / 'jobs').glob('*/*/seed_*/checkpoints/logging.jsonl')):
        train_curves[str(path.parent.parent.relative_to(work))] = rows(path)
    out['training_curves'] = train_curves
    save(report / 'statistics.json', out)
    text = ['# 实体类跨模态迁移实验', '', f'更新：{now()}', '',
            f'完整性：{count:,} / {expected_total:,} 次作答；完成 {len(complete_files)} / {expected_states} 个模型状态评测。', '',
            f'评测训练 seed：{seeds}。主终点固定 epoch 5。准确率平均四种选项排列。', '',
            ('只评 seed 42；不报告 seed 间标准差，置信区间仅表示此 seed 下的知识单元抽样不确定性。'
             if len(seeds) == 1 else '置信区间按知识单元配对 bootstrap；先平均排列和种子，不将其当作独立样本。'), '']
    for model, subsets in out['endpoint'].items():
        text += [f'## {model}：全量主结果', '', '|题型|T2T|T2V|V2T|V2V|', '|---|---:|---:|---:|---:|']
        for probe, result in subsets['all'].items():
            cells = [(f"{result['mean_accuracy'][d]*100:.2f}%" if len(seeds) == 1 else
                      f"{result['mean_accuracy'][d]*100:.2f}% ± {result['seed_sd'][d]*100:.2f}") for d in ('TT', 'TV', 'VT', 'VV')]
            text.append('|' + '|'.join([probe] + cells) + '|')
        if len(seeds) > 1:
            text += ['', '± 为所选训练 seed 的样本标准差。']
        text += ['', '|题型|对比|差值 pp|95% CI|Holm p|', '|---|---|---:|---|---:|']
        for probe, result in subsets['all'].items():
            for name, stat in result['contrasts'].items():
                text.append(f"|{probe}|{name}|{stat['difference_pp']:.2f}|[{stat['ci95_pp'][0]:.2f}, {stat['ci95_pp'][1]:.2f}]|{stat['p_holm']:.4f}|")
        text += ['']
    text += ['## 解释限制', '', '若同模态学习增益不明确，不能用准确率接近宣称迁移充分；不计算不稳定的增益比值。', '',
             'VISION 学习包含匿名事实正文。图片尚未全量人工语义核验；首次公开时间证据覆盖不足，不能据本轮结果声称事实均在模型 cutoff 后首次公开。', '',
             '398 单元最新生成版本作为预先指定的探索性分组，详见 statistics.json；不能替代全量主结果。', '',
             '逐排列、无效回答、选项分布、0→5 曲线、图片对照及各 seed 结果均保存在 statistics.json。', '']
    (report / 'report.md').write_text('\n'.join(text))
    print(f'Analysis saved: {count}/{expected_total} predictions, complete={completeness["complete"]}', flush=True)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path, default=DEFAULT_WORK)
    analyze(parser.parse_args().work)
