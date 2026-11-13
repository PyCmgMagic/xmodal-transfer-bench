#!/usr/bin/env python3
"""Isolated seed-42 Qwen3 rerun on the 440 visually unflagged v19 units.

The subset is NOT source-certified clean. Original data and previous runs are
read only. Two workers each own one GPU; durable inference resumes by prompt.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import importlib.metadata
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import run_entity_v19_full_experiment as engine
from entity_v19_experiment_data import ROOT, DATA, MODELS, now, read, rows, save, write_rows, sha

SOURCE = ROOT / 'runs/entity_v19_transfer_ep5_20261014'
AUDIT = ROOT / 'diagnostics/entity_v19_full_image_audit_20261016'
DEFAULT_WORK = ROOT / 'runs/entity_v19_screened440_qwen3_ep3_20261018'
EPOCHS = 3
SEED = 42


def prepare(work):
    if (work / 'manifest.json').exists():
        verify(work)
        return
    work.mkdir(parents=True, exist_ok=True)
    out = work / 'data'
    out.mkdir(exist_ok=True)
    selection = rows(AUDIT / '逐单元修复清单.jsonl')
    selected = sorted(r['unit_id'] for r in selection if not r['categories'])
    assert len(selected) == len(set(selected)) == 440
    selected_set = set(selected)
    image_audit = rows(AUDIT / '全部图片审查.jsonl')
    included_images = [r for r in image_audit if r['unit_id'] in selected_set]
    assert len(included_images) == 880
    assert all(r['visual_screened'] and not r['categories'] and not r['identity_verified'] for r in included_images)
    # Verify that saved review evidence is exactly the evidence behind summary.
    for name, digest in read(AUDIT / 'summary.json')['annotation_sha256'].items():
        assert sha(AUDIT / 'annotations' / name) == digest
    original = read(SOURCE / 'manifest.json')
    for item in original['sources']:
        assert sha(item['copy']) == item['sha256'], item['copy']
    for item in included_images:
        assert sha(item['absolute_path']) == item['sha256']
    assert not {r['sha256'] for r in included_images if r['role'] == 'learning'} & {
        r['sha256'] for r in included_images if r['role'] == 'evaluation'}
    write_rows(out / 'image_inventory.jsonl', included_images)
    write_rows(out / 'selection.jsonl', [r for r in selection if r['unit_id'] in selected_set])
    save(out / 'selected_unit_ids.json', selected)
    order = selected.copy()
    random.Random(SEED).shuffle(order)
    write_rows(out / 'unit_order_seed_42.jsonl', [{'unit_id': uid} for uid in order])
    frozen = SOURCE / 'data/frozen_sources'
    for mode, direction in [('TEXT', 'T2T'), ('VISION', 'V2V')]:
        source = {r['id']: r for r in rows(frozen / direction / 'train.jsonl')}
        values = []
        for uid in order:
            item = copy.deepcopy(source[uid])
            if item.get('images'):
                item['images'] = [str(DATA / name) for name in item['images']]
            assert set(item) <= {'id', 'qid', 'messages', 'images'}
            assert [m['role'] for m in item['messages']] == ['assistant']
            values.append({k: item[k] for k in ('messages', 'images') if k in item})
        write_rows(out / f'train_{mode}_seed_42.jsonl', values)
    old_labels = {r['id']: r for r in rows(SOURCE / 'data/labels.jsonl')}
    labels = []
    main = {}
    for test in ('text', 'image'):
        values = [r for r in rows(SOURCE / f'data/eval_{test}.jsonl')
                  if old_labels[r['id']]['unit_id'] in selected_set]
        assert len(values) == 440 * 2 * 4
        assert all(set(r) <= {'id', 'qid', 'messages', 'images'} for r in values)
        main[test] = values
        labels.extend(old_labels[r['id']] for r in values)
        write_rows(out / f'eval_{test}.jsonl', values)
    # Fresh fixed control sample within the retained subset, before new inference.
    sampled = random.Random(20261018).sample(selected, 256)
    qids = {r['unit_id']: r['qid'] for r in labels}
    ordered = sorted(sampled, key=lambda uid: qids[uid])
    wrong = dict(zip(ordered, ordered[1:] + ordered[:1]))
    image_by_unit = {old_labels[r['id']]['unit_id']: r['images'][0] for r in main['image']}
    controls = []
    for item in main['image']:
        meta = old_labels[item['id']]
        uid = meta['unit_id']
        if uid not in wrong:
            continue
        for condition in ('no_image', 'wrong_image'):
            value = copy.deepcopy(item)
            value['id'] = item['id'].replace('image|', condition + '|', 1)
            if condition == 'no_image':
                value.pop('images')
                value['messages'][-1]['content'] = value['messages'][-1]['content'].replace('<image>', '').lstrip('\n')
            else:
                value['images'] = [image_by_unit[wrong[uid]]]
                assert qids[uid] != qids[wrong[uid]] and value['images'] != item['images']
            controls.append(value)
            labels.append({**meta, 'id': value['id'], 'test': condition,
                           'wrong_image_unit': wrong[uid] if condition == 'wrong_image' else None})
    assert len(controls) == 4096 and len(labels) == 11136
    assert len({r['id'] for r in labels}) == len(labels)
    write_rows(out / 'eval_controls.jsonl', controls)
    write_rows(out / 'labels.jsonl', labels)
    save(out / 'control_sample.json', dict(seed=20261018, unit_ids=sampled,
         wrong_image_unit=wrong, selected_before_inference=True))
    groups = defaultdict(list)
    for r in labels:
        groups[r['test'], r['source_id']].append(r['answer'])
    assert all(sorted(v) == list('ABCD') for v in groups.values())
    # Check installed model/code content against original experiment inventory.
    model_inventory = [r for r in rows(SOURCE / 'data/model_inventory.jsonl') if r['model'] in ('qwen3', 'installed_swift_code')]
    for item in model_inventory:
        assert sha(item['path']) == item['sha256'], item['path']
    write_rows(out / 'model_inventory.jsonl', model_inventory)
    evidence = [AUDIT / '逐单元修复清单.jsonl', AUDIT / '全部图片审查.jsonl',
                AUDIT / 'summary.json', AUDIT / '正式图片全量筛查报告.md', SOURCE / 'manifest.json']
    evidence += sorted((AUDIT / 'annotations').glob('*.json'))
    for index, path in enumerate(evidence):
        dest = out / 'selection_evidence' / f'{index:03d}_{path.name}'
        dest.parent.mkdir(exist_ok=True)
        shutil.copy2(path, dest)
    limitations = ['440 units selected only by saved material flags, never by model accuracy.',
        'No source-certified identity checks: unflagged portraits are NOT certified clean.',
        'One training seed; no estimate of training-seed variability.',
        'VISION training uses image plus anonymized factual text.',
        'Fact public-date and original question-construction limitations remain.']
    manifest = dict(schema='entity-v19-screened440-qwen3-ep3-v1', created_at=now(),
        model=str(MODELS['qwen3']), model_name='Qwen3-VL-4B-Instruct', seed=42, epochs=3,
        units=440, questions_per_direction=880, question_types=['direct', 'statement'],
        training_jobs=2, original_run=str(SOURCE), material_audit=str(AUDIT),
        selection_rule='both role images have no category flags; 440 units; identity NOT certified',
        training={**original['training'], 'endpoint_selection': 'fixed epoch 3; never test accuracy'},
        inference={**original['inference'], 'parser': 'strict', 'max_new_tokens': 32},
        rotations=4, control_units=256, intermediate_epochs=[1, 2],
        expected_predictions=dict(base_main=7040, endpoint_main=14080, controls=12288,
                                  intermediate=7040, total=40448),
        expected_model_states=7, baseline_recomputed=True, limitations=limitations,
        dependencies={n: importlib.metadata.version(n) for n in original['dependencies']},
        files={str(p.relative_to(work)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()},
        original_data_modified=False)
    save(work / 'manifest.json', manifest)
    (work / 'README.md').write_text(
        '# Qwen3：筛查后 440 单元，3 epoch\n\n'
        '2026-10-18。使用图片筛查未标记单元；**未认证身份，不称作干净数据**。\n\n'
        'TEXT 和 VISION 分别从同一 Qwen3-VL-4B-Instruct 基座开始，seed 42，各训练 3 epoch。'
        '固定 epoch 3 为终点；四方向、direct/statement 分开，四种循环选项位置平均。'
        '另重新跑基座、中间 epoch 1/2 原顺序，以及预先固定 256 单元的无图/错图对照。\n\n'
        '所有原数据和旧实验保持不变。选择证据、哈希、顺序、标准答案均保存在 data。'
        '推理输入不传入标准答案；中断后按实际 prompt+image 身份续跑。\n\n'
        '结果自动写入 reports/report.md、statistics.json、completeness.json。\n')
    verify(work)
    print(f'PREPARED: {work}; 440 units; 40448 logical predictions', flush=True)


def verify(work):
    manifest = read(work / 'manifest.json')
    assert manifest['epochs'] == 3 and manifest['units'] == 440
    for path, digest in manifest['files'].items():
        assert sha(work / path) == digest, path
    for r in rows(work / 'data/image_inventory.jsonl'):
        assert sha(r['absolute_path']) == r['sha256'], r['absolute_path']
    for r in rows(work / 'data/model_inventory.jsonl'):
        assert sha(r['path']) == r['sha256'], r['path']


def command(dataset, output, resume):
    cmd = engine.train_command('qwen3', dataset, output, SEED, resume=resume)
    cmd[cmd.index('--num_train_epochs') + 1] = str(EPOCHS)
    cmd[cmd.index('--save_total_limit') + 1] = str(EPOCHS)
    return cmd


def train(work, mode, gpu):
    folder = work / 'jobs/qwen3' / mode / 'seed_42'
    output = folder / 'checkpoints'
    for attempt in range(1, 4):
        engine.check_stop()
        cp = engine.checkpoints(output)
        if EPOCHS in cp:
            break
        resume = cp[max(cp)] if cp else None
        if output.exists() and resume is None:
            output.rename(folder / f'interrupted_without_checkpoint_{int(time.time())}')
        save(folder / 'state.json', dict(status='running', mode=mode, seed=SEED, epochs=EPOCHS,
             attempt=attempt, gpu=gpu, resume=str(resume) if resume else None, updated_at=now()))
        cmd = command(work / f'data/train_{mode}_seed_42.jsonl', output, resume)
        code = engine.process(cmd, gpu, engine.TRAIN_PEAKS['qwen3'][mode.lower()], folder, f'train_{attempt:02d}')
        if code == 0:
            break
        print(f'TRAIN RETRY {mode}: attempt={attempt}, returncode={code}', flush=True)
        engine.STOP.wait(15)
    cp = engine.checkpoints(output)
    assert set(cp) == set(range(1, EPOCHS + 1)), f'Incomplete {mode} checkpoints: {cp}'
    inventory = {}
    for epoch, path in sorted(cp.items()):
        state = read(path / 'trainer_state.json')
        assert math.isclose(state['epoch'], epoch, abs_tol=1e-5)
        inventory[epoch] = dict(path=str(path), epoch=state['epoch'], global_step=state['global_step'],
                                adapter=engine.validate_adapter(path))
    losses = [r['loss'] for r in rows(output / 'logging.jsonl') if 'loss' in r]
    assert losses and all(math.isfinite(float(v)) and float(v) > 0 for v in losses)
    save(folder / 'checkpoint_manifest.json', inventory)
    save(folder / 'state.json', dict(status='complete', mode=mode, seed=SEED, epochs=EPOCHS,
         gpu=gpu, checkpoint=str(cp[EPOCHS]), completed_at=now()))
    print(f'TRAIN COMPLETE: {mode}', flush=True)
    return cp


def evaluate(work, gpu, folder, inputs, cp=None):
    # engine.infer already resumes individual prompt identities and retries a
    # failing subprocess; retain those raw records across controller retries.
    engine.infer(work, 'qwen3', gpu, folder, inputs, cp)
    engine.score_predictions(work, folder)


def worker(work, mode, gpu):
    state_path = work / f'{mode}.status.json'
    for retry in range(1, 4):
        try:
            save(state_path, dict(status='training', mode=mode, gpu=gpu, retry=retry, updated_at=now()))
            cp = train(work, mode, gpu)
            save(state_path, dict(status='evaluating', mode=mode, gpu=gpu, updated_at=now()))
            main = rows(work / 'data/eval_text.jsonl') + rows(work / 'data/eval_image.jsonl')
            controls = rows(work / 'data/eval_controls.jsonl')
            curve = [r for r in main if '|r0|' in r['id']]
            if mode == 'TEXT':
                evaluate(work, gpu, work / 'base/qwen3', main + controls)
            # Endpoint first; no test-dependent checkpoint selection.
            for epoch in (3, 1, 2):
                folder = work / f'evaluation/qwen3/seed_42/{mode}/epoch_{epoch}'
                evaluate(work, gpu, folder, main + controls if epoch == 3 else curve, cp[epoch])
            save(state_path, dict(status='complete', mode=mode, gpu=gpu, completed_at=now()))
            return
        except engine.Stopped:
            raise
        except Exception as exc:
            save(state_path, dict(status='retrying' if retry < 3 else 'failed', mode=mode, gpu=gpu,
                 retry=retry, error=str(exc), traceback=traceback.format_exc(), updated_at=now()))
            traceback.print_exc()
            if retry == 3:
                raise
            engine.STOP.wait(20)


def analyze(work):
    import numpy as np
    from analyze_entity_v19_full_experiment import accuracy, vector, contrast, holm
    manifest = read(work / 'manifest.json')
    units = read(work / 'data/selected_unit_ids.json')
    control_units = read(work / 'data/control_sample.json')['unit_ids']
    folders = [('base', work / 'base/qwen3', 11136)]
    folders += [(f'{mode}_epoch{epoch}', work / f'evaluation/qwen3/seed_42/{mode}/epoch_{epoch}',
                 11136 if epoch == 3 else 1760) for mode in ('TEXT', 'VISION') for epoch in (1, 2, 3)]
    states = {}
    for name, folder, expected in folders:
        if read(folder / 'progress.json').get('status') == 'complete' and (folder / 'results.jsonl').exists():
            rs = rows(folder / 'results.jsonl')
            assert len(rs) == expected and len({r['id'] for r in rs}) == expected
            assert {r['unit_id'] for r in rs} == set(units)
            states[name] = rs
    count = sum(len(v) for v in states.values())
    complete = len(states) == 7 and count == 40448
    save(work / 'reports/completeness.json', dict(complete=complete, completed_model_states=len(states),
        expected_model_states=7, logical_predictions=count, expected_predictions=40448, updated_at=now()))
    result = dict(complete=complete, seed=42, epochs=3, units=440, updated_at=now(),
                  limitations=manifest['limitations'], baseline={}, endpoint={}, image_controls={}, curves_rotation0=[],
                  metrics={n: accuracy(rs) for n, rs in states.items()})
    rng = np.random.default_rng(20261018)
    probes = ('overall', 'direct', 'statement')
    if 'base' in states:
        result['baseline'] = {test: {p: accuracy([r for r in states['base'] if r['test'] == test and
            (p == 'overall' or r['probe_id'] == p)]) for p in probes} for test in ('text', 'image')}
    if all(n in states for n in ('base', 'TEXT_epoch3', 'VISION_epoch3')):
        for probe in probes:
            arr = {}
            for mode, prefix in [('TEXT', 'T'), ('VISION', 'V')]:
                for test, suffix in [('text', 'T'), ('image', 'V')]:
                    arr[prefix + suffix] = vector([r for r in states[f'{mode}_epoch3'] if r['test'] == test], units, probe)
            baseline = {t: vector([r for r in states['base'] if r['test'] == t], units, probe) for t in ('text', 'image')}
            gains = {d: v - baseline['text' if d[-1] == 'T' else 'image'] for d, v in arr.items()}
            comparisons = dict(TT_minus_TV=arr['TT'] - arr['TV'], VV_minus_VT=arr['VV'] - arr['VT'],
                gain_TT_minus_TV=gains['TT'] - gains['TV'], gain_VV_minus_VT=gains['VV'] - gains['VT'],
                TT_minus_VT=arr['TT'] - arr['VT'], VV_minus_TV=arr['VV'] - arr['TV'],
                matched_minus_cross=(arr['TT'] + arr['VV'] - arr['TV'] - arr['VT']) / 2)
            stats = {k: contrast(v, rng) for k, v in comparisons.items()}
            holm(stats)
            gain_stats = {k: contrast(v, rng) for k, v in gains.items()}
            holm(gain_stats)
            result['endpoint'][probe] = dict(accuracy={k: float(v.mean()) for k, v in arr.items()},
                gain_pp={k: float(v.mean()) * 100 for k, v in gains.items()}, contrasts=stats, gain_stats=gain_stats)
    for name in ('base', 'TEXT_epoch3', 'VISION_epoch3'):
        if name not in states:
            continue
        result['image_controls'][name] = {}
        for p in probes:
            vs = {t: vector([r for r in states[name] if r['test'] == t], control_units, p)
                  for t in ('image', 'no_image', 'wrong_image')}
            comps = {f'image_minus_{t}': contrast(vs['image'] - vs[t], rng) for t in ('no_image', 'wrong_image')}
            holm(comps)
            result['image_controls'][name][p] = dict(accuracy={k: float(v.mean()) for k, v in vs.items()}, contrasts=comps)
    for epoch in range(4):
        for mode, prefix in [('TEXT', 'T'), ('VISION', 'V')]:
            name = 'base' if epoch == 0 else f'{mode}_epoch{epoch}'
            if name not in states:
                continue
            for test, suffix in [('text', 'T'), ('image', 'V')]:
                for probe in probes:
                    selected = [r for r in states[name] if r['test'] == test and r['rotation'] == 0 and
                                (probe == 'overall' or r['probe_id'] == probe)]
                    result['curves_rotation0'].append(dict(epoch=epoch, direction=prefix+suffix, probe=probe, **accuracy(selected)))
    result['per_state_metrics'] = {name: read(folder / 'metrics.json') for name, folder, _ in folders if name in states}
    save(work / 'reports/statistics.json', result)
    lines = ['# Qwen3：筛查后 440 单元，3 epoch 实验', '',
        f'完整性：{count:,}/40,448 次逻辑作答；{len(states)}/7 个模型状态。', '',
        'seed 42；TEXT/VISION 从同一基座独立训练 3 epoch；固定 E3 终点，四种选项轮换平均。'
        '全部 440 单元选自图片审查未标记集合，未认证人物身份，不称为干净数据。', '',
        '| 题型 | T→T | T→V | V→T | V→V |', '| --- | ---: | ---: | ---: | ---: |']
    for p, values in result['endpoint'].items():
        lines.append('|'+'|'.join([p]+[f'{values["accuracy"][d]*100:.2f}%' for d in ('TT','TV','VT','VV')])+'|')
    lines += ['', '| 题型 | ΔT→T pp | ΔT→V pp | ΔV→T pp | ΔV→V pp |', '| --- | ---: | ---: | ---: | ---: |']
    for p, values in result['endpoint'].items():
        lines.append('|'+'|'.join([p]+[f'{values["gain_pp"][d]:+.2f}' for d in ('TT','TV','VT','VV')])+'|')
    lines += ['', '增益减去本次重新评测的对应输入基座；未混用原 1,225 单元总体基座。', '',
              '| 状态 | 正确图片 | 无图 | 错图 |', '| --- | ---: | ---: | ---: |']
    for name, values in result['image_controls'].items():
        lines.append('|'+'|'.join([name]+[f'{values["overall"]["accuracy"][t]*100:.2f}%' for t in ('image','no_image','wrong_image')])+'|')
    lines += ['', '对照为固定 256 单元、两题型、四种轮换。配对差异/95%区间/Holm校正见 statistics.json；'
              '抽样单位为知识单元，不把选项轮换或同单元两题当作独立事实。', '',
              '只使用一个训练 seed，区间仅表示条件于 seed 42 的单元抽样不确定性。'
              '本次同时改变材料子集和训练 epoch，不能把与旧实验的分数差完全归因于清理图片。', '',
              '0→3 曲线统一用原始选项顺序；终点主表用四轮平均，两者需分开解释。', '',
              '无效输出仍计入分母；逐状态选项分布、逐排列结果、中间曲线及配对检验保存在 statistics.json。', '']
    (work / 'reports/report.md').write_text('\n'.join(lines))
    return complete


def controller(work, gpu0, gpu1):
    work.mkdir(parents=True, exist_ok=True)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, engine.signal_stop)
    with (work / '.controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not any(read(p).get('status') == 'running' and engine.alive(read(p).get('pid'))
                       for p in work.rglob('*.process.json')), 'Owned child still active'
        save(work / 'controller.json', dict(status='preparing', pid=os.getpid(), started_at=now()))
        try:
            prepare(work)
            (work / 'reports').mkdir(exist_ok=True)
            save(work / 'controller.json', dict(status='running', pid=os.getpid(), updated_at=now(),
                 gpu_assignment={'TEXT': gpu0, 'VISION': gpu1}, epochs=3, seed=42, units=440))
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(worker, work, mode, gpu) for mode, gpu in [('TEXT', gpu0), ('VISION', gpu1)]]
                for future in futures:
                    future.result()
            engine.check_stop()
            verify(work)
            assert analyze(work), 'Incomplete experiment'
            save(work / 'controller.json', dict(status='complete', pid=os.getpid(), completed_at=now()))
            print('EXPERIMENT COMPLETE: all training, four directions, curves, controls and report.', flush=True)
        except BaseException as exc:
            save(work / 'controller.json', dict(status='stopped' if engine.STOP.is_set() else 'failed',
                 pid=os.getpid(), error=str(exc), traceback=traceback.format_exc(), updated_at=now()))
            traceback.print_exc()
            raise


def status(work):
    active = []
    for p in sorted(work.rglob('*.process.json')):
        value = read(p)
        if value.get('status') in ('running', 'waiting_for_memory'):
            active.append(dict(path=str(p.relative_to(work)), **value))
    completed = [read(p) for folder in ('base', 'evaluation') for p in (work / folder).rglob('progress.json')]
    print(json.dumps(dict(controller=read(work / 'controller.json'),
        workers={m: read(work / f'{m}.status.json') for m in ('TEXT','VISION')},
        completed_prediction_rows=sum(r.get('logical_rows',0) for r in completed if r.get('status')=='complete'),
        active=active), ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','start','run','status','stop','analyze'])
    parser.add_argument('--work', type=Path, default=DEFAULT_WORK)
    parser.add_argument('--gpu0', default='0')
    parser.add_argument('--gpu1', default='1')
    args = parser.parse_args()
    work = args.work.resolve()
    if args.action == 'prepare':
        prepare(work)
    elif args.action == 'start':
        assert args.gpu0 != args.gpu1
        assert not engine.alive(read(work / 'controller.json').get('pid')), 'Controller active'
        prepare(work)
        # Run an immutable code snapshot, so unrelated later edits cannot alter
        # a long-running experiment or change recovery behavior.
        code = [Path(__file__).resolve()] + [ROOT / 'scripts' / name for name in (
            'run_entity_v19_full_experiment.py','entity_v19_experiment_data.py',
            'v7_gpu_resources.py','analyze_entity_v19_full_experiment.py')]
        version = sha(code[0])[:16]
        dest = work / 'execution' / version
        dest.mkdir(parents=True, exist_ok=True)
        for path in code:
            shutil.copy2(path, dest / path.name)
        helper = dest / 'entity_v19_experiment_data.py'
        helper.write_text(helper.read_text().replace(
            'ROOT = Path(__file__).resolve().parents[1]', f'ROOT = Path({str(ROOT)!r})'))
        save(dest / 'code_manifest.json', {p.name: sha(dest / p.name) for p in code})
        with (work / 'controller.log').open('a') as log:
            process = subprocess.Popen([str(ROOT / '.venv/bin/python'), '-u', str(dest / code[0].name),
                'run','--work',str(work),'--gpu0',args.gpu0,'--gpu1',args.gpu1],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps(dict(pid=process.pid, work=str(work), log=str(work/'controller.log'))))
    elif args.action == 'run':
        controller(work, args.gpu0, args.gpu1)
    elif args.action == 'status':
        status(work)
    elif args.action == 'stop':
        engine.stop(work)
    elif args.action == 'analyze':
        analyze(work)


if __name__ == '__main__':
    main()
