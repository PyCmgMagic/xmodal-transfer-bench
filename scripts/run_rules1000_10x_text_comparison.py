#!/usr/bin/env python3
"""Run the two expanded TEXT arms against the already completed 1000-row baseline."""
import argparse
import json
import hashlib
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'cluster_ready/rules1000_10x_20261027'
WORK = ROOT / 'runs/rules1000_10x_20261028_text_comparison'
BASELINE = ROOT / 'runs/rules_full_sft_ep10_20261026'
MODEL = ROOT / 'model_cache/Qwen3-VL-4B-Instruct'
LEDGER = ROOT / 'runs/qwen3_lora_cot_eval_20261025'
ARMS = ('multiform10', 'repeat10')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(tmp, path)


def update_status(status, **fields):
    state = read(WORK / 'status.json') if (WORK / 'status.json').exists() else {}
    state.update(status=status, updated_at=now(), **fields)
    save(WORK / 'status.json', state)


def plan_for(arm, output):
    command = [sys.executable, str(PACKAGE / 'tools/train.py'),
               '--root', str(PACKAGE), '--work', str(WORK / 'work'),
               '--model', str(MODEL), '--output', str(output),
               '--arm', arm, '--modality', 'text', '--epochs', '10', '--seed', '42']
    result = subprocess.run(command, cwd=ROOT, check=True, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    value = json.loads(result.stdout)
    if value['rows'] != 10000 or value['epochs'] != 10 or value['estimated_optimizer_steps_single_gpu'] != 6250:
        raise RuntimeError(f'Unexpected plan for {arm}: {value}')
    return value


def check_gpu_idle():
    command = ['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
               '--format=csv,noheader,nounits']
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    values = [x.strip() for x in result.stdout.strip().splitlines()[0].split(',')]
    index, memory_mib, utilization = map(int, values)
    if index != 0 or memory_mib > 1000 or utilization > 5:
        raise RuntimeError(f'GPU 0 is no longer idle (memory={memory_mib}MiB, util={utilization}%).')


def compare_training_settings(previous, current):
    def as_map(cmd):
        result = {}
        i = 2
        while i < len(cmd):
            key = cmd[i]
            if not key.startswith('--'):
                i += 1
                continue
            if i + 1 == len(cmd) or cmd[i + 1].startswith('--'):
                result[key] = True
                i += 1
            else:
                result[key] = cmd[i + 1]
                i += 2
        return result
    ignored = {'--dataset', '--output_dir', '--external_plugins'}
    a, b = as_map(previous), as_map(current)
    # The completed baseline relied on the same ms-swift defaults that the
    # portable command spells out; compare effective settings, not spelling.
    a.setdefault('--lr_scheduler_type', 'cosine')
    b.setdefault('--lr_scheduler_type', 'cosine')
    for command in (a, b):
        if '--num_train_epochs' in command:
            command['--num_train_epochs'] = str(float(command['--num_train_epochs']))
    keys = (set(a) | set(b)) - ignored
    return {k: {'baseline': a.get(k), 'new': b.get(k)} for k in sorted(keys) if a.get(k) != b.get(k)}


def main():
    global WORK
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=WORK,
                        help='Run directory containing work/prepared.json; use a fresh directory for a new attempt')
    args = parser.parse_args()
    WORK = args.work.resolve()
    if not (WORK / 'work/prepared.json').is_file():
        raise FileNotFoundError('Prepare the portable data before starting training.')
    baseline_states = [read(BASELINE / f'jobs/text/state.json'), read(BASELINE / f'jobs/vision/state.json')]
    if any(x['status'] != 'complete' or x['epoch'] != 10 for x in baseline_states):
        raise RuntimeError('The original 10-epoch baseline is not complete.')
    package_manifest = read(PACKAGE / 'manifest.json')
    model_identity = read(PACKAGE / 'model_identity.json')
    status_path = WORK / 'status.json'
    existing = read(status_path) if status_path.exists() else {}
    if existing.get('status') == 'complete':
        return
    if existing.get('status') == 'running':
        pid = int(existing.get('worker_pid') or existing.get('pid') or -1)
        if pid > 1 and Path(f'/proc/{pid}').exists():
            raise RuntimeError(f'Run already active with PID {pid}.')

    manifest = {
        'experiment': 'rules1000_10x_text_comparison', 'created_at': now(),
        'status': 'prepared', 'gpu': 'NVIDIA L40, device 0',
        'base_model': str(MODEL), 'model_identity': model_identity,
        'baseline': {
            'run': str(BASELINE), 'checkpoint_text': str(BASELINE / 'jobs/text/checkpoint/checkpoint-630'),
            'checkpoint_vision': str(BASELINE / 'jobs/vision/checkpoint/checkpoint-630'),
            'epochs': 10, 'seed': 42,
            'metrics_text_model': {'T2T': 0.2605, 'T2V': 0.2680}},
        'comparisons': ['original', 'multiform10', 'repeat10'],
        'new_training_arms': list(ARMS), 'modality': 'text', 'epochs': 10,
        'seed': 42, 'sample_presentations_per_rule': 100,
        'expected_optimizer_steps_per_arm': 6250,
        'compute_budget_policy': 'same epochs, not equal optimizer steps',
        'training_environment': read(PACKAGE / 'environment_reference.json'),
        'limitations': ['This first comparison covers the TEXT training modality only.',
                        'Both expanded arms see 10x examples versus original.',
                        'repeat10 vs multiform10 isolates question-form diversity at equal row count.']}

    baseline_rows = [json.loads(line) for line in
                     (ROOT / 'runs/qwen3_lora_learning_ep5_20261024/data/rules_sft_text.jsonl')
                     .read_text().splitlines()]
    original_rows = [json.loads(line) for line in
                     (WORK / 'work/data/train/original/text.jsonl').read_text().splitlines()]
    if len(baseline_rows) != 1000 or len(original_rows) != 1000 or any(
            a['messages'] != b['messages'] for a, b in zip(baseline_rows, original_rows)):
        raise RuntimeError('Prepared original TEXT conversations differ from the trained baseline.')
    baseline_command = read(BASELINE / 'jobs/text/command.json')
    comparison_plan = plan_for('multiform10', WORK / 'jobs/multiform10/checkpoint')
    setting_differences = compare_training_settings(baseline_command, comparison_plan['command'])
    if setting_differences:
        raise RuntimeError(f'Training settings differ from the baseline: {setting_differences}')
    save(WORK / 'preflight.json', {
        'status': 'pass', 'gpu': 'NVIDIA L40', 'visible_processes_before_start': 0,
        'original_text_rows': 1000, 'original_messages_match_completed_baseline': True,
        'baseline_command_sha256': hashlib.sha256(json.dumps(baseline_command).encode()).hexdigest(),
        'expanded_arm_training_settings_match_baseline': True,
        'baseline_setting_differences': setting_differences,
        'model_identity_manifest': model_identity,
        'bundle_sha256sums_sha256': file_hash(PACKAGE / 'SHA256SUMS'),
        'model_hash_verification': 'pass', 'model_hash_files': list(model_identity['files']),
        'prepared_manifest_sha256': file_hash(WORK / 'work/prepared.json'),
        'evaluation_inputs_or_gold_used_for_training': False})
    save(WORK / 'manifest.json', manifest)
    save(status_path, {'status': 'waiting_for_gpu', 'pid': os.getpid(), 'started_at': now(),
                       'arms': list(ARMS)})

    # Share the project's GPU admission ledger so paused evaluation jobs cannot
    # start a competing model load on this 46 GiB device.
    sys.path.insert(0, str(ROOT / 'scripts'))
    from v7_gpu_resources import reserve
    with reserve(LEDGER, '0', 'rules1000_10x_text_comparison', 41000, poll=15) as lease:
        # Keep the lease attached to the coordinator across both sequential
        # workers; a completed first worker must not invalidate the second lease.
        lease.bind(os.getpid())
        check_gpu_idle()
        for arm in ARMS:
            output = WORK / 'jobs' / arm / 'checkpoint'
            if output.exists() and any(output.iterdir()):
                raise RuntimeError(f'Refusing to overwrite nonempty output: {output}')
            output.parent.mkdir(parents=True, exist_ok=True)
            plan = plan_for(arm, output)
            save(WORK / 'jobs' / arm / 'run_plan.json', plan)
            command = plan['command']
            env = os.environ.copy()
            env.update(HF_HUB_OFFLINE='1', MAX_PIXELS='1003520', PYTHONUNBUFFERED='1',
                       NPROC_PER_NODE='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       TOKENIZERS_PARALLELISM='false',
                       HF_HOME=str(ROOT / '.runtime/cache/huggingface'),
                       HF_DATASETS_CACHE=str(ROOT / '.runtime/cache/huggingface/datasets'),
                       TMPDIR=str(ROOT / '.runtime/tmp'),
                       RULES_DENSE_AUDIT=str(WORK / 'jobs' / arm / 'parameter_audit.jsonl'),
                       RULES_DENSE_MODALITY='text', RULES_DENSE_STOP_AT='',
                       PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
            log_path = WORK / 'jobs' / arm / 'train.log'
            state_path = WORK / 'jobs' / arm / 'state.json'
            started = time.monotonic()
            update_status('running', arm=arm, worker_pid=None)
            save(state_path, {'status': 'running', 'arm': arm, 'pid': os.getpid(),
                              'started_at': now(), 'rows': 10000, 'epochs': 10,
                              'expected_steps': 6250, 'command': command})
            with log_path.open('w') as log:
                child = subprocess.Popen(command, cwd=WORK / 'work', env=env,
                                         stdin=subprocess.DEVNULL, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                update_status('running', arm=arm, worker_pid=child.pid)
                return_code = child.wait()
            elapsed = time.monotonic() - started
            if return_code:
                interrupted = return_code == -signal.SIGINT
                state_status = 'stopped' if interrupted else 'failed'
                save(state_path, {'status': state_status, 'arm': arm, 'pid': child.pid,
                                  'returncode': return_code, 'elapsed_seconds': elapsed,
                                  'ended_at': now(), 'log': str(log_path)})
                update_status(state_status, arm=arm, returncode=return_code,
                              stop_reason='SIGINT' if interrupted else None)
                if interrupted:
                    return
                raise RuntimeError(f'{arm} training failed; see {log_path}')
            checkpoints = sorted(output.glob('checkpoint-*'),
                                 key=lambda p: int(p.name.rsplit('-', 1)[1]))
            if not checkpoints:
                raise RuntimeError(f'No checkpoint created for {arm}.')
            checkpoint = checkpoints[-1]
            trainer = read(checkpoint / 'trainer_state.json')
            if abs(float(trainer['epoch']) - 10.0) > 1e-6 or int(trainer['global_step']) != 6250:
                raise RuntimeError(f'Incomplete final checkpoint for {arm}: {trainer}')
            audit = [json.loads(x) for x in (WORK / 'jobs' / arm / 'parameter_audit.jsonl').read_text().splitlines()]
            if not any(item.get('event') == 'step_end' and item.get('step') == 6250 for item in audit):
                raise RuntimeError(f'Final dense update audit missing for {arm}.')
            metrics = {'checkpoint': str(checkpoint), 'epoch': trainer['epoch'],
                       'global_step': trainer['global_step'], 'log_history': trainer['log_history'],
                       'elapsed_seconds': elapsed,
                       'parameter_audit': str(WORK / 'jobs' / arm / 'parameter_audit.jsonl')}
            save(state_path, {'status': 'complete', 'arm': arm, **metrics, 'ended_at': now()})
            update_status('arm_complete', arm=arm, checkpoint=str(checkpoint),
                          global_step=trainer['global_step'], elapsed_seconds=elapsed)
        manifest['status'] = 'training_complete'
        manifest['completed_at'] = now()
        save(WORK / 'manifest.json', manifest)
        update_status('training_complete', arms=list(ARMS))


if __name__ == '__main__':
    main()
