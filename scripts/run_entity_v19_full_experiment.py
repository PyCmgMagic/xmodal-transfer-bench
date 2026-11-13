#!/usr/bin/env python3
"""Durable two-GPU execution of entity_v19_full_experiment_plan.md.

Run with the project's .venv/bin/python. `start` detaches a controller, `status`
reads durable progress, and `stop` signals the controller and ALL owned groups.
Predictions resume by exact prompt/image identity, never by row position.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from entity_v19_experiment_data import (
    ROOT, DATA, DEFAULT_WORK, MODELS, SEEDS, freeze, verify, evaluator, inference_only,
    now, read, rows, save, write_rows, sha,
    evaluation_scope, in_evaluation_scope,
)
from v7_gpu_resources import TRAIN_PEAKS, EVAL_PEAKS, snapshot, alive

STOP = threading.Event()
CHILDREN = {}
CHILD_LOCK = threading.RLock()


class Stopped(Exception):
    pass


def check_stop():
    if STOP.is_set():
        raise Stopped('Experiment stop requested')


def env_for(gpu):
    env = os.environ.copy()
    distributed = {'NPROC_PER_NODE', 'NNODES', 'NODE_RANK', 'MASTER_ADDR', 'MASTER_PORT',
                   'LOCAL_RANK', 'RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'GROUP_RANK', 'ROLE_RANK', 'ROLE_WORLD_SIZE'}
    for key in list(env):
        if key in distributed or key.startswith('TORCHELASTIC_'):
            env.pop(key, None)
    # Project's `tmp` is short enough for multiprocessing AF_UNIX socket paths.
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), ROOT_IMAGE_DIR=str(DATA), MAX_PIXELS='1003520',
               MODELSCOPE_CACHE=str(ROOT / 'model_cache/modelscope'),
               HF_HOME=str(ROOT / 'model_cache/huggingface'),
               HF_DATASETS_CACHE=str(ROOT / 'model_cache/datasets'),
               HF_HUB_OFFLINE='1', TMPDIR=str(ROOT / 'tmp'),
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
               PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false',
               PATH=str(ROOT / '.venv/bin') + os.pathsep + env['PATH'])
    for key in ('MODELSCOPE_CACHE', 'HF_HOME', 'HF_DATASETS_CACHE', 'TMPDIR'):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    return env


def train_command(model, dataset, output, seed, smoke=False, resume=None):
    cmd = [str(ROOT / '.venv/bin/swift'), 'pt', '--model', str(MODELS[model]),
           '--dataset', str(dataset), '--tuner_type', 'lora', '--target_modules', 'all-linear',
           '--lora_rank', '8', '--lora_alpha', '32', '--lora_dropout', '0.05',
           '--torch_dtype', 'bfloat16', '--freeze_vit', 'true', '--freeze_aligner', 'true',
           '--per_device_train_batch_size', '1', '--gradient_accumulation_steps', '1' if smoke else '16',
           '--learning_rate', '1e-4', '--num_train_epochs', '5', '--gradient_checkpointing', 'true',
           '--max_length', '8192', '--warmup_ratio', '0' if smoke else '0.05',
           '--save_strategy', 'steps' if smoke else 'epoch', '--save_total_limit', '5',
           '--save_only_model', 'false', '--split_dataset_ratio', '0',
           '--logging_steps', '1' if smoke else '5', '--seed', str(seed), '--data_seed', str(seed),
           '--dataset_shuffle', 'false', '--group_by_length', 'false',
           '--add_version', 'false', '--output_dir', str(output), '--use_chat_template', 'false',
           '--loss_scale', 'all', '--strict', 'true', '--enable_thinking', 'false', '--report_to', 'none']
    if model == 'glm46':
        cmd += ['--template', 'glm4_5v']
    if smoke:
        cmd += ['--max_steps', '1', '--save_steps', '1']
    if resume:
        cmd += ['--resume_from_checkpoint', str(resume), '--load_args', 'false']
    return cmd


def kill_group(pid, sig):
    try:
        os.killpg(int(pid), sig)
    except ProcessLookupError:
        pass


def group_members(groups):
    members = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            fields = (p / 'stat').read_text().split(') ')[1].split()
            if fields[0] != 'Z' and int(fields[2]) in groups:
                members.append(int(p.name))
        except (OSError, IndexError, ValueError):
            continue
    return members


def signal_stop(signum, _frame):
    STOP.set()
    # Process groups, not only the immediate Swift launcher.
    with CHILD_LOCK:
        for pid in list(CHILDREN):
            kill_group(pid, signal.SIGINT)


def process(cmd, gpu, peak, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    state_path = folder / f'{name}.process.json'
    while True:
        check_stop()
        total, used, _ = snapshot(gpu)
        if total - used >= peak + 4000:
            break
        save(state_path, {'status': 'waiting_for_memory', 'gpu': gpu, 'required_mib': peak + 4000,
                          'free_mib': total - used, 'updated_at': now()})
        STOP.wait(10)
    check_stop()
    with (folder / f'{name}.log').open('a') as log:
        # Hold lock through fork/registration so a stop cannot miss a newborn child.
        with CHILD_LOCK:
            check_stop()
            p = subprocess.Popen(cmd, cwd=ROOT, env=env_for(gpu), stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            CHILDREN[p.pid] = str(state_path)
            save(state_path, {'status': 'running', 'pid': p.pid, 'pgid': p.pid, 'gpu': gpu,
                              'command': cmd, 'started_at': now()})
        print(f'START GPU{gpu} PID {p.pid}: {folder.relative_to(ROOT)} / {name}', flush=True)
        stopped_at = None
        while p.poll() is None:
            if STOP.wait(1):
                if stopped_at is None:
                    kill_group(p.pid, signal.SIGINT)
                    stopped_at = time.monotonic()
                elif time.monotonic() - stopped_at > 20:
                    kill_group(p.pid, signal.SIGKILL)
        code = p.wait()
        with CHILD_LOCK:
            CHILDREN.pop(p.pid, None)
    save(state_path, {'status': 'stopped' if STOP.is_set() else ('complete' if code == 0 else 'failed'),
                      'pid': p.pid, 'pgid': p.pid, 'gpu': gpu, 'returncode': code, 'command': cmd, 'ended_at': now()})
    check_stop()
    return code


def checkpoints(folder):
    result = {}
    for p in folder.glob('checkpoint-*'):
        state = read(p / 'trainer_state.json')
        required = ('adapter_model.safetensors', 'adapter_config.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth')
        if not state or not all((p / n).is_file() for n in required):
            continue
        epoch = float(state['epoch'])
        if abs(epoch - round(epoch)) < 1e-5:
            result[int(round(epoch))] = p
    return result


def validate_adapter(cp):
    from safetensors import safe_open
    with safe_open(str(cp / 'adapter_model.safetensors'), framework='pt', device='cpu') as f:
        keys = list(f.keys())
        forbidden = ('.visual.', '.vision.', '.vision_tower.', '.vision_model.', '.merger.',
                     '.mm_projector.', '.multi_modal_projector.')
        assert not any(any(part in k for part in forbidden) for k in keys), 'Frozen vision weights in adapter'
        assert any('lora_B' in k and f.get_tensor(k).count_nonzero().item() > 0 for k in keys), 'No LoRA update'
        assert all(f.get_tensor(k).isfinite().all().item() for k in keys), 'Nonfinite adapter tensor'
    return {'tensor_count': len(keys), 'nonzero_lora_B': True, 'frozen_vision': True,
            'sha256': {n: sha(cp / n) for n in ('adapter_model.safetensors', 'adapter_config.json', 'trainer_state.json')}}


def infer(work, model, gpu, folder, inputs, adapter=None, require_valid=False):
    """Dedup before Swift and align by exact prompt/images; retain first observations."""
    ev = evaluator()
    folder.mkdir(parents=True, exist_ok=True)
    assert len({r['id'] for r in inputs}) == len(inputs)
    unique = {}
    sid_to_key = {}
    for item in inputs:
        key = ev.prompt_key(item)
        unique.setdefault(key, item)
        sid_to_key[item['id']] = key
    config = {'model': model, 'adapter': str(adapter) if adapter else None,
              'adapter_sha256': sha(adapter / 'adapter_model.safetensors') if adapter else None,
              'inputs_digest': ev.digest(inputs), 'rows': len(inputs), 'unique_inputs': len(unique),
              'max_new_tokens': 32, 'parser': 'glm-box' if model == 'glm46' else 'strict'}
    prior = read(folder / 'config.json')
    if prior:
        assert prior == config, f'Changed inference config: {folder}'
    else:
        save(folder / 'config.json', config)
        write_rows(folder / 'alignment.jsonl', [{'id': sid, 'key': k} for sid, k in sid_to_key.items()])
    got = {}

    def collect():
        # The expected list is the exact attempt input; outputs may be partial.
        for raw_path in sorted(folder.glob('attempt_*.raw.jsonl')):
            input_path = raw_path.with_name(raw_path.name.replace('.raw.', '.mapping.'))
            expected = rows(input_path)
            raw, notes = ev.read_rows(raw_path, allow_partial_tail=True)
            expected_keys = {ev.prompt_key(r) for r in expected}
            for item in raw:
                assets = item.get('images') or []
                if isinstance(assets, str):
                    assets = [assets]
                # If the backend supplies actual paths, do not silently use a
                # text-only fallback when they point at a different image.
                if assets and all(isinstance(p, str) for p in assets):
                    assert ev.prompt_key(item) in expected_keys, f'Unexpected output image: {raw_path}'
            aligned = ev.align(raw, expected, raw_path)
            expected_by_id = {r['id']: r for r in expected}
            for sid, prediction in aligned.items():
                key = ev.prompt_key(expected_by_id[sid])
                if key in got:
                    assert got[key]['response'] == prediction['response'], 'Repeated input with changed response'
                else:
                    got[key] = prediction
        return len(got)

    collect()
    failed = 0
    while len(got) < len(unique):
        check_stop()
        remaining = [r for key, r in unique.items() if key not in got]
        attempt = 1 + max([int(p.name.split('_')[1].split('.')[0]) for p in folder.glob('attempt_*.mapping.jsonl')] or [0])
        stem = f'attempt_{attempt:03}'
        mapping = folder / f'{stem}.mapping.jsonl'
        write_rows(mapping, remaining)
        inp = folder / f'{stem}.input.jsonl'
        write_rows(inp, [inference_only(r) for r in remaining])
        raw = folder / f'{stem}.raw.jsonl'
        args = SimpleNamespace(model=str(MODELS[model]), batch_size=1, write_batch_size=1,
                               max_new_tokens=32, seed=42, max_pixels=1003520)
        cmd = ev.swift_command(args, [str(ROOT / '.venv/bin/swift')], str(adapter) if adapter else None, inp, raw)
        if model == 'glm46':
            cmd += ['--template', 'glm4_5v']
        before = len(got)
        code = process(cmd, gpu, EVAL_PEAKS[model], folder, stem)
        collect()
        save(folder / 'progress.json', {'status': 'running', 'logical_rows': len(inputs),
             'unique_inputs': len(unique), 'completed_unique': len(got), 'last_returncode': code, 'updated_at': now()})
        if len(got) == before or code:
            failed += 1
            if failed >= 3:
                raise RuntimeError(f'Inference failed {failed} times, preserved completed rows: {folder}')
            STOP.wait(10)
    predictions = []
    for item in inputs:
        raw = got[sid_to_key[item['id']]]
        selected = ev.letter(raw['response'], config['parser'])
        # Also flag length-limited generations when the backend exposes the reason.
        reason = raw.get('finish_reason')
        if reason == 'length':
            selected = None
        predictions.append({'id': item['id'], 'response': raw['response'], 'predicted': selected,
                            'format_valid': selected is not None, 'finish_reason': reason,
                            'input_key': sid_to_key[item['id']]})
    write_rows(folder / 'predictions.jsonl', predictions)
    invalid = sum(not p['format_valid'] for p in predictions)
    save(folder / 'progress.json', {'status': 'complete', 'logical_rows': len(inputs), 'unique_inputs': len(unique),
         'completed_unique': len(got), 'invalid_rows': invalid, 'completed_at': now()})
    if require_valid and invalid:
        raise RuntimeError(f'Preflight output protocol invalid for {invalid} probes: {folder}')
    return predictions


def smoke(work, model, gpu, mode):
    folder = work / 'preflight' / model / mode
    if read(folder / 'validated.json').get('status') == 'pass':
        return
    folder.mkdir(parents=True, exist_ok=True)
    inp = folder / 'train.jsonl'
    write_rows(inp, rows(work / f'data/train_{mode}_seed_42.jsonl')[:1])
    out = folder / 'checkpoints'
    cp = next(iter(checkpoints(out).values()), None)
    if not cp:
        if process(train_command(model, inp, out, 42, smoke=True), gpu,
                   TRAIN_PEAKS[model][mode.lower()], folder, 'train'):
            raise RuntimeError(f'Smoke train failed: {folder}')
        cp = next(iter(checkpoints(out).values()), None)
    assert cp, f'Missing smoke checkpoint: {folder}'
    state = read(cp / 'trainer_state.json')
    assert state['global_step'] == 1
    losses = [r['loss'] for r in rows(out / 'logging.jsonl') if 'loss' in r]
    assert losses and all(math.isfinite(float(x)) and float(x) > 0 for x in losses)
    adapter = validate_adapter(cp)
    probes = [rows(work / f'data/eval_{test}.jsonl')[0] for test in ('text', 'image')]
    infer(work, model, gpu, folder / 'reload', probes, cp, require_valid=True)
    save(folder / 'validated.json', {'status': 'pass', 'model': model, 'mode': mode, 'losses': losses,
         'checkpoint': str(cp), 'adapter': adapter, 'text_and_image_reload': True, 'checked_at': now()})
    print(f'PREFLIGHT PASS: {model} {mode}', flush=True)


def train(work, model, gpu, mode, seed):
    folder = work / 'jobs' / model / mode / f'seed_{seed}'
    output = folder / 'checkpoints'
    cp = checkpoints(output)
    if 5 not in cp:
        resume = cp[max(cp)] if cp else None
        # Keep interrupted logs/files without a recoverable checkpoint.
        if output.exists() and not resume:
            output.rename(folder / f'interrupted_without_checkpoint_{int(time.time())}')
        save(folder / 'state.json', {'status': 'running', 'model': model, 'mode': mode, 'seed': seed,
             'epochs': 5, 'resume': str(resume) if resume else None, 'started_at': now()})
        code = process(train_command(model, work / f'data/train_{mode}_seed_{seed}.jsonl', output, seed,
                                     resume=resume), gpu, TRAIN_PEAKS[model][mode.lower()], folder, 'train')
        if code:
            save(folder / 'state.json', {'status': 'failed', 'returncode': code, 'updated_at': now()})
            raise RuntimeError(f'Training failed: {folder}; see train.log')
        cp = checkpoints(output)
    assert set(cp) == set(range(1, 6)), f'Missing epoch checkpoints: {folder}: {cp}'
    inventory = {}
    for epoch, path in sorted(cp.items()):
        state = read(path / 'trainer_state.json')
        inventory[epoch] = {'path': str(path), 'epoch': state['epoch'], 'global_step': state['global_step'],
                            'adapter': validate_adapter(path)}
    save(folder / 'checkpoint_manifest.json', inventory)
    save(folder / 'state.json', {'status': 'complete', 'model': model, 'mode': mode, 'seed': seed,
         'epochs': 5, 'checkpoint': str(cp[5]), 'completed_at': now()})
    print(f'TRAIN COMPLETE: {model} {mode} seed {seed}', flush=True)


def evaluate_model(work, model, gpu):
    main = rows(work / 'data/eval_text.jsonl') + rows(work / 'data/eval_image.jsonl')
    controls = rows(work / 'data/eval_controls.jsonl')
    # Combine modalities under one load; response identity is preserved outside Swift.
    base = work / 'base' / model
    infer(work, model, gpu, base, main + controls)
    score_predictions(work, base)
    curve = [r for r in main if '|r0|' in r['id']]
    for seed in evaluation_scope(work)['evaluation_seeds']:
        for mode in ('TEXT', 'VISION'):
            cp = checkpoints(work / 'jobs' / model / mode / f'seed_{seed}' / 'checkpoints')
            for epoch in (range(1, 6) if seed == 42 else (5,)):
                folder = work / 'evaluation' / model / f'seed_{seed}' / mode / f'epoch_{epoch}'
                inputs = main + controls if epoch == 5 and seed == 42 else (main if epoch == 5 else curve)
                infer(work, model, gpu, folder, inputs, cp[epoch])
                score_predictions(work, folder)
    update_report(work)


def score_predictions(work, folder):
    labels = {r['id']: r for r in rows(work / 'data/labels.jsonl')}
    scored = []
    for p in rows(folder / 'predictions.jsonl'):
        label = labels[p['id']]
        scored.append({**label, **p, 'correct': p['predicted'] == label['answer'],
                       'status': ('invalid_output' if not p['format_valid'] else
                                  ('correct' if p['predicted'] == label['answer'] else 'wrong_answer'))})
    write_rows(folder / 'results.jsonl', scored)
    def metric(rs):
        return {'total': len(rs), 'correct': sum(r['correct'] for r in rs),
                'accuracy': sum(r['correct'] for r in rs) / len(rs),
                'invalid': sum(not r['format_valid'] for r in rs),
                'predicted_options': dict(Counter(r['predicted'] for r in rs))}
    groups = {}
    for r in scored:
        groups.setdefault((r['test'], r['probe_id'], r['rotation']), []).append(r)
    save(folder / 'metrics.json', {'overall': metric(scored), 'cells': [
        {'test': k[0], 'probe_id': k[1], 'rotation': k[2], **metric(v)} for k, v in sorted(groups.items())]})


def update_report(work):
    # The analysis program consumes only already completed, fully aligned cells.
    cmd = [str(ROOT / '.venv/bin/python'), str(ROOT / 'scripts/analyze_entity_v19_full_experiment.py'),
           '--work', str(work)]
    if Path(cmd[1]).is_file():
        with (work / 'reports/analysis.log').open('a') as log:
            code = subprocess.call(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if code:
            raise RuntimeError('Statistical report generation failed; see reports/analysis.log')


def worker(work, model, gpu, barrier):
    try:
        save(work / f'{model}.status.json', {'status': 'running', 'gpu': gpu, 'updated_at': now(),
             'evaluation_seeds': evaluation_scope(work)['evaluation_seeds']})
        for mode in ('TEXT', 'VISION'):
            smoke(work, model, gpu, mode)
        barrier.wait()
        for seed in evaluation_scope(work)['evaluation_seeds']:
            for mode in ('TEXT', 'VISION'):
                check_stop()
                train(work, model, gpu, mode, seed)
        evaluate_model(work, model, gpu)
        save(work / f'{model}.status.json', {'status': 'complete', 'completed_at': now()})
    except BaseException as exc:
        barrier.abort()
        save(work / f'{model}.status.json', {'status': 'stopped' if STOP.is_set() else 'failed',
             'error': str(exc), 'traceback': traceback.format_exc(), 'updated_at': now()})
        traceback.print_exc()
        raise


def controller(work, gpu0, gpu1):
    work.mkdir(parents=True, exist_ok=True)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, signal_stop)
    with (work / '.controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Refuse duplicate Swift children after an unclean controller crash.
        existing = [read(p) for p in work.rglob('*.process.json')]
        assert not any(r.get('status') == 'running' and alive(r.get('pid')) for r in existing), 'Owned children still alive; stop first'
        save(work / 'controller.json', {'status': 'preparing', 'pid': os.getpid(), 'started_at': now(),
             'gpu_assignment': {'qwen3': gpu0, 'glm46': gpu1}})
        try:
            freeze(work)
            # Keep an immutable code snapshot for every controller revision,
            # including analysis and the imported resource-budget helper.
            code_paths = [ROOT / 'scripts' / name for name in (
                'run_entity_v19_full_experiment.py', 'entity_v19_experiment_data.py',
                'analyze_entity_v19_full_experiment.py', 'v7_gpu_resources.py')]
            version = evaluator().digest({p.name: sha(p) for p in code_paths})[:16]
            code_dir = work / 'execution' / version
            code_dir.mkdir(parents=True, exist_ok=True)
            for p in code_paths:
                shutil.copy2(p, code_dir / p.name)
            save(code_dir / 'manifest.json', {'created_at': now(), 'controller_pid': os.getpid(),
                 'files': {p.name: sha(p) for p in code_paths}})
            check_stop()
            for model in MODELS:
                for mode in ('TEXT', 'VISION'):
                    for seed in SEEDS:
                        state = work / 'jobs' / model / mode / f'seed_{seed}/state.json'
                        if not state.exists():
                            save(state, {'status': 'queued', 'model': model, 'mode': mode, 'seed': seed, 'epochs': 5})
            (work / 'reports').mkdir(exist_ok=True)
            save(work / 'controller.json', {'status': 'running', 'pid': os.getpid(), 'updated_at': now(),
                 'gpu_assignment': {'qwen3': gpu0, 'glm46': gpu1}})
            barrier = threading.Barrier(2)
            errors = []
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(worker, work, model, gpu, barrier)
                           for model, gpu in (('qwen3', gpu0), ('glm46', gpu1))]
                for future in futures:
                    try:
                        future.result()
                    except BaseException as exc:
                        errors.append(str(exc))
            check_stop()
            if errors:
                raise RuntimeError('; '.join(errors))
            verify(work, models=True)
            update_report(work)
            final = read(work / 'reports/completeness.json')
            assert final.get('logical_predictions') == evaluation_scope(work)['expected_predictions']['total'] and final.get('complete'), final
            save(work / 'controller.json', {'status': 'complete', 'pid': os.getpid(), 'ended_at': now()})
        except BaseException as exc:
            save(work / 'controller.json', {'status': 'stopped' if STOP.is_set() else 'failed', 'pid': os.getpid(),
                 'error': str(exc), 'traceback': traceback.format_exc(), 'ended_at': now()})
            raise


def status(work):
    jobs = [read(p) for p in sorted((work / 'jobs').glob('*/*/seed_*/state.json'))]
    active = []
    for p in work.rglob('*.process.json'):
        r = read(p)
        if r.get('status') in ('running', 'waiting_for_memory'):
            active.append({'path': str(p.relative_to(work)), 'alive': alive(r.get('pid')), **r})
    completed = [read(p) for p in work.rglob('progress.json') if in_evaluation_scope(p, work)]
    print(json.dumps({'controller': read(work / 'controller.json'),
         'training_jobs': dict(Counter(r.get('status') for r in jobs)),
         'preflight_passed': len(list((work / 'preflight').glob('*/*/validated.json'))),
         'evaluation_scope': evaluation_scope(work),
         'completed_prediction_rows': sum(r.get('logical_rows', 0) for r in completed if r.get('status') == 'complete'),
         'active': active, 'model_status': {k: read(work / f'{k}.status.json') for k in MODELS}}, ensure_ascii=False, indent=2))


def stop(work):
    state = read(work / 'controller.json')
    pid = state.get('pid')
    if alive(pid):
        os.kill(pid, signal.SIGTERM)
    children = [r['pid'] for p in work.rglob('*.process.json')
                if (r := read(p)).get('status') == 'running' and r.get('pid')]
    for child in children:
        kill_group(child, signal.SIGINT)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and (alive(pid) or group_members(set(children))):
        time.sleep(.5)
    for child in children:
        kill_group(child, signal.SIGKILL)
    if alive(pid):
        os.kill(pid, signal.SIGKILL)
    time.sleep(.5)
    assert not alive(pid) and not group_members(set(children)), 'Some experiment processes still alive'
    save(work / 'controller.json', {**state, 'status': 'stopped', 'ended_at': now(), 'confirmed_children_exited': children})
    print('Experiment controller and all recorded owned process groups stopped.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=('start', 'run', 'prepare', 'status', 'stop'))
    ap.add_argument('--work', type=Path, default=DEFAULT_WORK)
    ap.add_argument('--gpu0', default='0')
    ap.add_argument('--gpu1', default='1')
    args = ap.parse_args()
    work = args.work.resolve()
    if args.action == 'start':
        work.mkdir(parents=True, exist_ok=True)
        assert not alive(read(work / 'controller.json').get('pid')), 'Controller already running'
        assert args.gpu0 != args.gpu1, 'Two independent physical GPUs required'
        with (work / 'controller.log').open('a') as log:
            p = subprocess.Popen([str(ROOT / '.venv/bin/python'), '-u', str(Path(__file__).resolve()),
                 'run', '--work', str(work), '--gpu0', args.gpu0, '--gpu1', args.gpu1],
                 cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'pid': p.pid, 'work': str(work), 'log': str(work / 'controller.log')}))
    elif args.action == 'run':
        controller(work, args.gpu0, args.gpu1)
    elif args.action == 'prepare':
        freeze(work)
    elif args.action == 'status':
        status(work)
    elif args.action == 'stop':
        stop(work)


if __name__ == '__main__':
    main()
