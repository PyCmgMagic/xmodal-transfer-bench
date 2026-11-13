#!/usr/bin/env python3
"""Durable two-GPU Qwen3 experiment on answer-balanced entity1020."""
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
from entity_v19_experiment_data import ROOT, MODELS, now, read, rows, save, write_rows, sha

SOURCE = ROOT / 'entity1020/balanced_1572'
WORK = ROOT / 'runs/entity1020_balanced_qwen3_ep3_20261020'
EPOCHS = 3
SEED = 42
UNITS = 1572
CONTROL_UNITS = 256


def setup_engine(work=None):
    engine.DATA = SOURCE
    engine.MODELS = {'qwen3': MODELS['qwen3']}
    evaluator_path = (work / 'data/frozen_sources/scripts/run_evaluation.py') if work else SOURCE / 'scripts/run_evaluation.py'
    spec = __import__('importlib.util', fromlist=['spec_from_file_location']).spec_from_file_location(
        'entity1020_balanced_evaluator', evaluator_path)
    ev = __import__('importlib.util', fromlist=['module_from_spec']).module_from_spec(spec)
    spec.loader.exec_module(ev)
    engine.evaluator = lambda: ev
    return ev


def prepare(work):
    if (work / 'manifest.json').is_file():
        verify(work)
        return
    assert not work.exists() or set(p.name for p in work.iterdir()) <= {'data','README.md'}, f'Partial work directory requires inspection: {work}'
    ev = setup_engine()
    work.mkdir(parents=True, exist_ok=True)
    out = work / 'data'
    out.mkdir(exist_ok=True)
    # Pin package files, balanced-label mapping, and evaluator with the run.
    frozen = out / 'frozen_sources'
    for rel in ('balance_manifest.json', 'package_inventory.json', 'audit/eval_labels.jsonl',
                'audit/sample_audit.jsonl', 'audit/answer_permutations.jsonl',
                'source/questions.jsonl', 'source/answers.jsonl', 'summary.json',
                'build_manifest.json', 'scripts/run_evaluation.py'):
        dest = frozen / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE / rel, dest)
    label_by_source = {r['id']: r for r in rows(SOURCE / 'audit/eval_labels.jsonl')}
    audit = {r['sample_id']: r for r in rows(SOURCE / 'audit/sample_audit.jsonl')}
    units = sorted({r['unit_id'] for r in audit.values()})
    assert len(units) == UNITS
    order = units.copy()
    random.Random(SEED).shuffle(order)
    save(out / 'selected_unit_ids.json', units)
    write_rows(out / 'unit_order_seed_42.jsonl', [{'unit_id': u} for u in order])
    for mode, direction in (('TEXT', 'T2T'), ('VISION', 'V2V')):
        src = {r['id']: r for r in rows(SOURCE / direction / 'train.jsonl')}
        train = []
        for uid in order:
            item = copy.deepcopy(src[uid])
            if item.get('images'):
                item['images'] = [str((SOURCE / item['images'][0]).resolve())]
            assert set(item) <= {'id', 'qid', 'messages', 'images'}
            assert [m['role'] for m in item['messages']] == ['assistant']
            train.append({k: item[k] for k in ('messages', 'images') if k in item})
        write_rows(out / f'train_{mode}_seed_42.jsonl', train)

    # Four cyclic option rotations balance output positions without changing content.
    main, metadata = {}, {}
    for test, direction in (('text', 'T2T'), ('image', 'T2V')):
        source_rows = rows(SOURCE / direction / 'eval.jsonl')
        expanded, metas = [], []
        for item in source_rows:
            label = label_by_source[item['id']]
            audit_row = audit[item['id']]
            for rotation in range(4):
                value = copy.deepcopy(item)
                value['id'] = f'{test}|r{rotation}|{item["id"]}'
                value['messages'][-1]['content'] = engine_rotate(value['messages'][-1]['content'], rotation)
                value = ev.prepare_row(value, SOURCE, direction)
                expanded.append(value)
                metas.append(dict(id=value['id'], source_id=item['id'], unit_id=audit_row['unit_id'],
                    qid=item['qid'], probe_id=label['probe_id'], test=test, rotation=rotation,
                    answer='ABCD'[('ABCD'.index(label['answer']) - rotation) % 4],
                    generation_version='entity1020-balanced-v1'))
        assert len(expanded) == 4 * len(source_rows) == 12576
        main[test] = expanded
        metadata[test] = metas
        write_rows(out / f'eval_{test}.jsonl', expanded)

    # Fixed, joint-label-stratified control sample, chosen before inference.
    by_unit = defaultdict(dict)
    for label in label_by_source.values():
        by_unit[audit[label['id']]['unit_id']][label['probe_id']] = label['answer']
    joint = defaultdict(list)
    for uid, pair in by_unit.items():
        joint[pair['direct'] + pair['statement']].append(uid)
    rng = random.Random(20261020)
    sampled = []
    for key in sorted(joint):
        sampled.extend(rng.sample(sorted(joint[key]), 16))
    assert len(sampled) == CONTROL_UNITS
    save(out / 'control_sample.json', dict(seed=20261020, unit_ids=sampled,
         joint_type_label_counts={k: 16 for k in sorted(joint)}, selected_before_inference=True))
    qid_by_unit = {r['unit_id']: r['qid'] for r in audit.values()}
    ordered = sorted(sampled, key=lambda u: qid_by_unit[u])
    wrong = dict(zip(ordered, ordered[1:] + ordered[:1]))
    image_by_source = {meta['source_id']: item['images'][0]
                       for item, meta in zip(main['image'], metadata['image']) if meta['rotation'] == 0}
    base_by_source = {r['source_id']: r for r in metadata['image'] if r['rotation'] == 0}
    source_item_by_id = {r['id']: r for r in rows(SOURCE / 'T2V/eval.jsonl')}
    controls, control_labels = [], []
    sample_set = set(sampled)
    for source_id, meta in base_by_source.items():
        uid = meta['unit_id']
        if uid not in sample_set:
            continue
        original = source_item_by_id[source_id]
        for rotation in range(4):
            right = copy.deepcopy(original)
            right['messages'][-1]['content'] = engine_rotate(right['messages'][-1]['content'], rotation)
            right = ev.prepare_row(right, SOURCE, 'T2V')
            right['id'] = f'image|r{rotation}|{source_id}'
            # Reuse exact main correct-image row identity and store controls only.
            for condition in ('no_image', 'wrong_image'):
                value = copy.deepcopy(right)
                value['id'] = f'{condition}|r{rotation}|{source_id}'
                if condition == 'no_image':
                    value.pop('images')
                    value['messages'][-1]['content'] = value['messages'][-1]['content'].replace('<image>', '').lstrip('\n')
                else:
                    other = wrong[uid]
                    other_source = next(k for k, v in base_by_source.items() if v['unit_id'] == other)
                    value['images'] = [image_by_source[other_source]]
                    assert other_source != source_id
                controls.append(value)
                control_labels.append({**meta, 'id': value['id'], 'test': condition, 'rotation': rotation,
                    'answer': 'ABCD'[('ABCD'.index(meta['answer'])-rotation)%4],
                    'wrong_image_unit': wrong[uid] if condition == 'wrong_image' else None})
    assert len(controls) == CONTROL_UNITS * 2 * 4 * 2 == 4096
    write_rows(out / 'eval_controls.jsonl', controls)
    labels = metadata['text'] + metadata['image'] + control_labels
    assert len({r['id'] for r in labels}) == len(labels) == 29248
    write_rows(out / 'labels.jsonl', labels)
    groups = defaultdict(list)
    for r in metadata['text'] + metadata['image']:
        groups[r['test'], r['probe_id'], r['rotation']].append(r['answer'])
    assert all(Counter(v) == dict.fromkeys('ABCD', 393) for v in groups.values())
    # Pin the selected model and installed training implementation.
    prior = ROOT / 'runs/entity_v19_transfer_ep5_20261014/data/model_inventory.jsonl'
    model_inv = [r for r in rows(prior) if r['model'] in ('qwen3', 'installed_swift_code')]
    for r in model_inv:
        assert sha(r['path']) == r['sha256']
    write_rows(out / 'model_inventory.jsonl', model_inv)
    counts = dict(base_main=25152, endpoints_main=50304, controls=12288, intermediate=25152, total=112896)
    deps = {n: importlib.metadata.version(n) for n in read(ROOT / 'runs/entity_v19_transfer_ep5_20261014/manifest.json')['dependencies']}
    files = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    save(work / 'manifest.json', dict(schema='entity1020-balanced-qwen3-ep3-v1', created_at=now(),
         source=str(SOURCE), source_balance_manifest=sha(SOURCE / 'balance_manifest.json'),
         model=str(MODELS['qwen3']), model_name='Qwen3-VL-4B-Instruct', seed=SEED, epochs=EPOCHS,
         units=UNITS, question_types=['direct','statement'], questions_per_direction=3144,
         answer_distribution_per_type=dict.fromkeys('ABCD',393), training_jobs=2,
         training={**read(ROOT / 'runs/entity_v19_transfer_ep5_20261014/manifest.json')['training'],
                   'endpoint_selection':'fixed epoch 3'}, rotations=4, control_units=CONTROL_UNITS,
         intermediate_epochs=[1,2], expected_predictions=counts, expected_model_states=7,
         baseline_recomputed=True, dependencies=deps, files=files, original_data_modified=False,
         limitations=['One training seed (42).','Retained images are not human identity certified.',
                      'Fact first-public dates are not individually verified; 2025-12-31 cutoff retained.']))
    (work / 'README.md').write_text('# entity1020 均衡版 Qwen3 实验\n\n'
        '最新 1,572 单元数据；direct/statement 分别 A/B/C/D 各 393。只重排选择题选项并同步标签。'
        'TEXT 与 VISION 各从 Qwen3-VL-4B-Instruct 基座独立训练 3 epoch，seed 42。'
        '终点固定 epoch 3；基座重测；四方向均四轮选项位置，epoch 1/2 与基座曲线采用原顺序。'
        '另外在预先固定的 256 单元上评测正确图、无图、错图。共 112,896 次逻辑作答。\n\n'
        '答案、ID 和标签从推理输入剥离；逐次预测可断点续跑。保留图像未获人工身份认证。\n')
    verify(work)
    print(f'PREPARED: {work}; {UNITS} units; 112896 logical predictions', flush=True)


def engine_rotate(text, rotation):
    from entity_v19_experiment_data import rotate
    return rotate(text, rotation)


def verify(work):
    m = read(work / 'manifest.json')
    assert m['units'] == UNITS and m['epochs'] == EPOCHS
    for rel, digest in m['files'].items():
        assert sha(work / 'data' / rel) == digest, rel
    for rel, digest in read(SOURCE / 'package_inventory.json')['sha256'].items():
        assert sha(SOURCE / rel) == digest, rel
    for item in rows(work / 'data/model_inventory.jsonl'):
        assert sha(item['path']) == item['sha256'], item['path']
    validate_inputs(work)
    return True


def validate_inputs(work):
    from balance_entity1020 import choices
    out = work / 'data'
    labels = {r['id']: r for r in rows(out / 'labels.jsonl')}
    assert len(labels) == 29248
    main = {r['id']: r for t in ('text','image') for r in rows(out/f'eval_{t}.jsonl')}
    original_gold = {r['id']:r['answer'] for r in rows(SOURCE/'audit/eval_labels.jsonl')}
    source_rows = {t:{r['id']:r for r in rows(SOURCE/d/'eval.jsonl')}
                   for t,d in (('text','T2T'),('image','T2V'))}
    groups = defaultdict(Counter)
    for sid,item in main.items():
        label = labels[sid]; source_id=label['source_id']; t=label['test']
        expected = source_rows[t][source_id]
        original = choices(expected['messages'][-1]['content'])
        actual = choices(item['messages'][-1]['content'])
        assert actual[1]['ABCD'.index(label['answer'])] == original[1]['ABCD'.index(original_gold[source_id])]
        assert Counter(actual[1]) == Counter(original[1])
        assert actual[0].replace('<image>\n','') == original[0].replace('<image>\n','')
        assert set(item) <= {'id','qid','messages','images'}
        assert [m['role'] for m in item['messages']] == ['system','user']
        groups[t,label['probe_id'],label['rotation']][label['answer']] += 1
    assert all(c == dict.fromkeys('ABCD',393) for c in groups.values())
    role_images = {r['unit_id']:str((SOURCE/r['evaluation']['file']).resolve()) for r in rows(SOURCE/'audit/image_roles.jsonl')}
    control_groups = defaultdict(Counter)
    for item in rows(out/'eval_controls.jsonl'):
        label=labels[item['id']]
        right_id=f'image|r{label["rotation"]}|{label["source_id"]}'
        right=main[right_id]
        assert label['answer'] == labels[right_id]['answer']
        assert label['qid'] == labels[right_id]['qid']
        assert set(item) <= {'id','qid','messages','images'}
        if label['test']=='no_image':
            assert not item.get('images')
            assert item['messages'][-1]['content'] == right['messages'][-1]['content'].replace('<image>','').lstrip('\n')
        else:
            assert item['messages']==right['messages']
            assert item['images']==[role_images[label['wrong_image_unit']]]
            assert label['wrong_image_unit'] != label['unit_id']
            assert item['images'] != right['images']
        control_groups[label['test'],label['probe_id'],label['rotation']][label['answer']] += 1
    assert all(c == dict.fromkeys('ABCD',64) for c in control_groups.values())
    assert len(control_groups)==16
    for mode,d in (('TEXT','T2T'),('VISION','V2V')):
        source={r['id']:r for r in rows(SOURCE/d/'train.jsonl')}
        order=rows(out/'unit_order_seed_42.jsonl')
        train=rows(out/f'train_{mode}_seed_42.jsonl')
        assert len(train)==len(order)==1572
        for row,unit in zip(train,order):
            assert row['messages']==source[unit['unit_id']]['messages']
            assert set(row) <= {'messages','images'}
            assert [m['role'] for m in row['messages']]==['assistant']
    save(work/'input_validation.json',dict(status='passed',checked_at=now(),main_rows=25152,control_rows=4096,
        correct_option_content_preserved=True,rotated_gold_matches_controls=True,each_main_cell=dict.fromkeys('ABCD',393),
        each_control_cell=dict.fromkeys('ABCD',64),inference_fields_whitelisted=True,training_facts_unchanged=True))


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
        cmd = engine.train_command('qwen3', work / f'data/train_{mode}_seed_42.jsonl', output, SEED, resume=resume)
        cmd[cmd.index('--num_train_epochs')+1] = str(EPOCHS)
        cmd[cmd.index('--save_total_limit')+1] = str(EPOCHS)
        code = engine.process(cmd, gpu, engine.TRAIN_PEAKS['qwen3'][mode.lower()], folder, f'train_{attempt:02d}')
        if code == 0:
            break
        engine.STOP.wait(15)
    cp = engine.checkpoints(output)
    assert set(cp) == set(range(1,EPOCHS+1)), f'Incomplete {mode} checkpoints: {cp}'
    inventory = {}
    for epoch, path in sorted(cp.items()):
        state = read(path/'trainer_state.json')
        assert abs(state['epoch']-epoch) < 1e-5
        inventory[epoch] = dict(path=str(path), epoch=state['epoch'], global_step=state['global_step'],
                                adapter=engine.validate_adapter(path))
    losses = [r['loss'] for r in rows(output/'logging.jsonl') if 'loss' in r]
    assert losses and all(math.isfinite(float(v)) and float(v) > 0 for v in losses)
    save(folder/'checkpoint_manifest.json',inventory)
    save(folder/'state.json',dict(status='complete',mode=mode,seed=SEED,epochs=EPOCHS,gpu=gpu,
        checkpoint=str(cp[EPOCHS]),completed_at=now()))
    print(f'TRAIN COMPLETE: {mode}',flush=True)
    return cp


def evaluate(work, gpu, folder, inputs, cp=None):
    setup_engine(work)
    engine.infer(work,'qwen3',gpu,folder,inputs,cp)
    engine.score_predictions(work,folder)


def worker(work, mode, gpu):
    status = work / f'{mode}.status.json'
    for attempt in range(1,4):
        try:
            setup_engine(work)
            save(status,dict(status='preflight',mode=mode,gpu=gpu,attempt=attempt,updated_at=now()))
            engine.smoke(work,'qwen3',gpu,mode)
            save(status,dict(status='training',mode=mode,gpu=gpu,attempt=attempt,updated_at=now()))
            cp = train(work,mode,gpu)
            main = rows(work/'data/eval_text.jsonl') + rows(work/'data/eval_image.jsonl')
            controls = rows(work/'data/eval_controls.jsonl')
            curve = [r for r in main if '|r0|' in r['id']]
            if mode == 'TEXT':
                evaluate(work,gpu,work/'base/qwen3',main+controls)
            for epoch in (3,1,2):
                folder = work/f'evaluation/qwen3/seed_42/{mode}/epoch_{epoch}'
                evaluate(work,gpu,folder,main+controls if epoch == 3 else curve,cp[epoch])
            save(status,dict(status='complete',mode=mode,gpu=gpu,completed_at=now()))
            return
        except engine.Stopped:
            raise
        except Exception as exc:
            save(status,dict(status='retrying' if attempt<3 else 'failed',mode=mode,gpu=gpu,
                attempt=attempt,error=str(exc),traceback=traceback.format_exc(),updated_at=now()))
            traceback.print_exc()
            if attempt == 3: raise
            engine.STOP.wait(20)


def analyze(work):
    import numpy as np
    from analyze_entity_v19_full_experiment import accuracy, vector, contrast, holm
    units = read(work/'data/selected_unit_ids.json')
    controls = read(work/'data/control_sample.json')['unit_ids']
    folders=[('base',work/'base/qwen3',29248)]
    folders += [(f'{m}_epoch{e}',work/f'evaluation/qwen3/seed_42/{m}/epoch_{e}',29248 if e==3 else 6288)
                for m in ('TEXT','VISION') for e in (1,2,3)]
    states={}
    for name,folder,n in folders:
        if read(folder/'progress.json').get('status')=='complete' and (folder/'results.jsonl').is_file():
            rs=rows(folder/'results.jsonl'); assert len(rs)==n
            assert {r['unit_id'] for r in rs}==set(units)
            states[name]=rs
    count=sum(map(len,states.values())); complete=len(states)==7 and count==112896
    save(work/'reports/completeness.json',dict(complete=complete,completed_model_states=len(states),
        expected_model_states=7,logical_predictions=count,expected_predictions=112896,updated_at=now()))
    result=dict(complete=complete,seed=42,epochs=3,units=UNITS,limitations=read(work/'manifest.json')['limitations'],
        baseline={},endpoint={},image_controls={},curves_rotation0=[],metrics={k:accuracy(v) for k,v in states.items()})
    rng=np.random.default_rng(20261020); probes=('overall','direct','statement')
    if 'base' in states:
        result['baseline']={t:{p:accuracy([r for r in states['base'] if r['test']==t and (p=='overall' or r['probe_id']==p)])
            for p in probes} for t in ('text','image')}
    if all(k in states for k in ('base','TEXT_epoch3','VISION_epoch3')):
        for p in probes:
            arr={}
            for m,pre in (('TEXT','T'),('VISION','V')):
                for t,suf in (('text','T'),('image','V')):
                    arr[pre+suf]=vector([r for r in states[f'{m}_epoch3'] if r['test']==t],units,p)
            base={t:vector([r for r in states['base'] if r['test']==t],units,p) for t in ('text','image')}
            gains={d:v-base['text' if d[-1]=='T' else 'image'] for d,v in arr.items()}
            comps=dict(TT_minus_TV=arr['TT']-arr['TV'],VV_minus_VT=arr['VV']-arr['VT'],
                gain_TT_minus_TV=gains['TT']-gains['TV'],gain_VV_minus_VT=gains['VV']-gains['VT'],
                TT_minus_VT=arr['TT']-arr['VT'],VV_minus_TV=arr['VV']-arr['TV'],
                matched_minus_cross=(arr['TT']+arr['VV']-arr['TV']-arr['VT'])/2)
            st={k:contrast(v,rng) for k,v in comps.items()};holm(st)
            gs={k:contrast(v,rng) for k,v in gains.items()};holm(gs)
            result['endpoint'][p]=dict(accuracy={k:float(v.mean()) for k,v in arr.items()},
                gain_pp={k:float(v.mean())*100 for k,v in gains.items()},contrasts=st,gain_stats=gs)
    for name in ('base','TEXT_epoch3','VISION_epoch3'):
        if name not in states: continue
        result['image_controls'][name]={}
        for p in probes:
            v={t:vector([r for r in states[name] if r['test']==t],controls,p) for t in ('image','no_image','wrong_image')}
            c={f'image_minus_{t}':contrast(v['image']-v[t],rng) for t in ('no_image','wrong_image')};holm(c)
            result['image_controls'][name][p]=dict(accuracy={k:float(x.mean()) for k,x in v.items()},contrasts=c)
    for epoch in range(4):
        for mode,pre in (('TEXT','T'),('VISION','V')):
            name='base' if epoch==0 else f'{mode}_epoch{epoch}'
            if name not in states: continue
            for test,suf in (('text','T'),('image','V')):
                for p in probes:
                    rs=[r for r in states[name] if r['test']==test and r['rotation']==0 and (p=='overall' or r['probe_id']==p)]
                    result['curves_rotation0'].append(dict(epoch=epoch,direction=pre+suf,probe=p,**accuracy(rs)))
    result['per_state_metrics']={n:read(f/'metrics.json') for n,f,_ in folders if n in states}
    save(work/'reports/statistics.json',result)
    lines=['# entity1020 均衡版 Qwen3：3 epoch', '',f'完整性：{count:,}/112,896 次逻辑作答；{len(states)}/7 个模型状态。','',
        '1,572 单元；direct/statement 各自 A/B/C/D 各 393。seed 42，TEXT/VISION 独立训练，终点固定 epoch 3。', '',
        '| 题型 | T→T | T→V | V→T | V→V |','| --- | ---: | ---: | ---: | ---: |']
    for p,v in result['endpoint'].items(): lines.append('|'+ '|'.join([p]+[f'{v["accuracy"][d]*100:.2f}%' for d in ('TT','TV','VT','VV')])+'|')
    lines += ['', '增益（相对本次重测基座，百分点）：', '', '| 题型 | T→T | T→V | V→T | V→V |','| --- | ---: | ---: | ---: | ---: |']
    for p,v in result['endpoint'].items(): lines.append('|'+ '|'.join([p]+[f'{v["gain_pp"][d]:+.2f}' for d in ('TT','TV','VT','VV')])+'|')
    lines += ['', '基座与第 3 epoch 端点对 256 个预先选定单元进行正确图/无图/错图对照；区间、检验、逐题型、逐方向、选项位置结果详见 statistics.json。',
        '', '只跑一个训练 seed。保留图片尚未完成人工身份认证；知识首次公开日期也未逐条核验。','']
    (work/'reports/report.md').write_text('\n'.join(lines))
    return complete


def controller(work,gpu0,gpu1):
    setup_engine(); work.mkdir(parents=True,exist_ok=True)
    for sig in (signal.SIGTERM,signal.SIGINT): signal.signal(sig,engine.signal_stop)
    with (work/'.controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not any(read(p).get('status')=='running' and engine.alive(read(p).get('pid')) for p in work.rglob('*.process.json'))
        save(work/'controller.json',dict(status='preparing',pid=os.getpid(),started_at=now()))
        try:
            prepare(work); (work/'reports').mkdir(exist_ok=True)
            save(work/'controller.json',dict(status='running',pid=os.getpid(),updated_at=now(),gpu_assignment={'TEXT':gpu0,'VISION':gpu1}))
            with ThreadPoolExecutor(max_workers=2) as pool:
                fs=[pool.submit(worker,work,m,g) for m,g in (('TEXT',gpu0),('VISION',gpu1))]
                for f in fs: f.result()
            verify(work); assert analyze(work)
            save(work/'controller.json',dict(status='complete',pid=os.getpid(),completed_at=now()))
            print('EXPERIMENT COMPLETE',flush=True)
        except BaseException as exc:
            save(work/'controller.json',dict(status='stopped' if engine.STOP.is_set() else 'failed',pid=os.getpid(),
                error=str(exc),traceback=traceback.format_exc(),updated_at=now()))
            traceback.print_exc(); raise


def status(work):
    active=[]
    for p in sorted(work.rglob('*.process.json')):
        v=read(p)
        if v.get('status') in ('running','waiting_for_memory'): active.append({'path':str(p.relative_to(work)),**v})
    progress=[read(p) for p in work.rglob('progress.json')]
    print(json.dumps(dict(controller=read(work/'controller.json'),workers={m:read(work/f'{m}.status.json') for m in ('TEXT','VISION')},
        completed_prediction_rows=sum(v.get('logical_rows',0) for v in progress if v.get('status')=='complete'),active=active),ensure_ascii=False,indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=('prepare','start','run','status','stop','analyze'))
    ap.add_argument('--work',type=Path,default=WORK);ap.add_argument('--gpu0',default='0');ap.add_argument('--gpu1',default='1')
    a=ap.parse_args();w=a.work.resolve()
    if a.action=='prepare': prepare(w)
    elif a.action=='start':
        assert a.gpu0!=a.gpu1 and not engine.alive(read(w/'controller.json').get('pid'))
        prepare(w)
        paths=[Path(__file__).resolve()]+[ROOT/'scripts'/n for n in ('run_entity_v19_full_experiment.py','entity_v19_experiment_data.py','v7_gpu_resources.py','analyze_entity_v19_full_experiment.py','balance_entity1020.py')]
        version=sha(paths[0])[:16]; dest=w/'execution'/version;dest.mkdir(parents=True,exist_ok=True)
        for p in paths: shutil.copy2(p,dest/p.name)
        h=dest/'entity_v19_experiment_data.py';h.write_text(h.read_text().replace('ROOT = Path(__file__).resolve().parents[1]',f'ROOT = Path({str(ROOT)!r})'))
        save(dest/'code_manifest.json',{p.name:sha(dest/p.name) for p in paths})
        with (w/'controller.log').open('a') as log:
            p=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u',str(dest/paths[0].name),'run','--work',str(w),'--gpu0',a.gpu0,'--gpu1',a.gpu1],
                cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(pid=p.pid,work=str(w),log=str(w/'controller.log'))))
    elif a.action=='run': controller(w,a.gpu0,a.gpu1)
    elif a.action=='status': status(w)
    elif a.action=='stop': engine.stop(w)
    elif a.action=='analyze': analyze(w)


if __name__=='__main__': main()
