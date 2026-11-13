#!/usr/bin/env python3
"""Run a reproducible 1-epoch TEXT/VISION SFT and corrected-set comparison."""
import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'model_cache/modelscope/models/Qwen--Qwen3-VL-4B-Instruct/snapshots/master'


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def environment():
    env = os.environ.copy()
    env.update(MODELSCOPE_CACHE=str(ROOT / 'model_cache/modelscope'),
               MODELSCOPE_HOME=str(ROOT / 'model_cache/modelscope'),
               HF_HOME=str(ROOT / 'model_cache/huggingface'),
               HF_DATASETS_CACHE=str(ROOT / 'model_cache/huggingface/datasets'),
               XDG_CACHE_HOME=str(ROOT / 'model_cache/xdg'),
               ROOT_IMAGE_DIR=str(ROOT / 'rules_corrected_final'), MAX_PIXELS='1003520',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
               PATH=str(ROOT / '.venv/bin') + os.pathsep + env['PATH'])
    for key in ('NPROC_PER_NODE', 'NNODES', 'NODE_RANK', 'MASTER_ADDR', 'MASTER_PORT', 'LOCAL_RANK', 'RANK', 'WORLD_SIZE'):
        env.pop(key, None)
    (ROOT / 'model_cache/xdg/torch/kernels').mkdir(parents=True, exist_ok=True)
    return env


def train(mode, gpu, run, resume):
    out = run / f'{mode}'
    checkpoint = out / 'checkpoint-63'
    if resume and (checkpoint / 'adapter_model.safetensors').is_file():
        state = json.loads((out / 'launcher_state.json').read_text())
        assert state['status'] == 'complete'
        return
    if out.exists():
        raise RuntimeError(f'Refusing to overwrite incomplete training: {out}')
    (out / 'logs').mkdir(parents=True)
    filename = 'train_text.jsonl' if mode == 'TEXT' else 'train_multimodal.jsonl'
    cmd = [str(ROOT / '.venv/bin/swift'), 'sft', '--model', str(MODEL),
           '--dataset', str(ROOT / 'rules_sft/data' / filename),
           '--tuner_type', 'lora', '--target_modules', 'all-linear',
           '--lora_rank', '8', '--lora_alpha', '32', '--lora_dropout', '0.05',
           '--torch_dtype', 'bfloat16', '--freeze_vit', 'true', '--freeze_aligner', 'true',
           '--per_device_train_batch_size', '1', '--gradient_accumulation_steps', '16',
           '--learning_rate', '1e-4', '--num_train_epochs', '1',
           '--gradient_checkpointing', 'true', '--max_length', '8192', '--warmup_ratio', '0.05',
           '--save_strategy', 'epoch', '--save_total_limit', '2', '--split_dataset_ratio', '0',
           '--logging_steps', '5', '--seed', '42', '--add_version', 'false',
           '--use_chat_template', 'true', '--loss_scale', 'default', '--output_dir', str(out)]
    env = environment()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    state = {'mode': mode, 'gpu': gpu, 'command': cmd, 'started_at': now(), 'status': 'running'}
    with (out / 'logs/train.log').open('w') as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        state['pid'] = proc.pid
        save(out / 'launcher_state.json', state)
        print(f'{mode} training pid={proc.pid}', flush=True)
        code = proc.wait()
    state.update(status='complete' if code == 0 else 'failed', returncode=code, ended_at=now())
    save(out / 'launcher_state.json', state)
    if code != 0 or not (checkpoint / 'adapter_model.safetensors').is_file():
        raise RuntimeError(f'{mode} training failed; inspect {out}/logs/train.log')
    print(f'{mode} training complete', flush=True)


def evaluate(group, run, batch_size, resume):
    out = run / 'evaluation' / group
    cmd = [str(ROOT / '.venv/bin/python'), str(ROOT / 'run_evaluation.py'),
           '--data-root', str(ROOT / 'rules_corrected_final'), '--model', str(MODEL),
           '--output-dir', str(out), '--parallel', '--gpu', '0', '--visual-gpu', '1',
           '--batch-size', str(batch_size), '--max-new-tokens', '4', '--max-pixels', '1003520',
           '--swift-bin', str(ROOT / '.venv/bin/swift')]
    if group == 'BASE':
        cmd += ['--base-model-only']
    else:
        adapters = run if group == 'SFT' else ROOT / 'rules/runs/experiments'
        prefix = '' if group == 'SFT' else 'lora8_ep1-'
        cmd += ['--text-adapter', str(adapters / f'{prefix}TEXT/checkpoint-63'),
                '--visual-adapter', str(adapters / f'{prefix}VISION/checkpoint-63')]
    if out.exists():
        if not resume:
            raise RuntimeError(f'Evaluation exists: {out}; use --resume')
        cmd += ['--resume']
    state = {'status': 'running', 'command': cmd, 'started_at': now()}
    save(run / f'evaluation_{group}_state.json', state)
    print(f'{group} evaluation starting', flush=True)
    with (run / f'evaluation_{group}.log').open('a') as log:
        code = subprocess.call(cmd, cwd=ROOT, env=environment(), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    summary_path = out / 'summary.json'
    reports = json.loads(summary_path.read_text())['directions'] if summary_path.is_file() else []
    finished = len(reports) == 4 and all(
        r.get('status') in ('complete', 'complete_with_output_errors') and
        r.get('matched') == r.get('total') == 2000 for r in reports)
    # The shared evaluator returns 1 for invalid answers even when all rows finished.
    # Retain these answers and their errors; do not retry them or treat them as missing.
    state.update(status='complete' if finished else 'failed', returncode=code, ended_at=now(),
                 output_errors=sum(r.get('empty_prediction', 0) + r.get('format_violation', 0) for r in reports))
    save(run / f'evaluation_{group}_state.json', state)
    if not finished:
        raise RuntimeError(f'{group} evaluation failed; inspect log')
    print(f'{group} evaluation complete', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, default=ROOT / 'rules_sft/runs/lora8_ep1_20261008')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--train-only', action='store_true')
    p.add_argument('--batch-size', type=int, default=4)
    args = p.parse_args()
    run = args.run_dir.resolve()
    if run.exists() and not args.resume:
        raise RuntimeError(f'Run exists: {run}; choose new --run-dir or --resume')
    run.mkdir(parents=True, exist_ok=True)
    with (run / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        subprocess.run([str(ROOT / '.venv/bin/python'), str(ROOT / 'scripts/build_rules_sft.py')], cwd=ROOT, check=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(train, mode, gpu, run, args.resume) for mode, gpu in [('TEXT', 0), ('VISION', 1)]]
            for future in futures:
                future.result()
        if not args.train_only:
            for group in ['SFT', 'PT', 'BASE']:
                evaluate(group, run, args.batch_size, args.resume)
            subprocess.run([str(ROOT / '.venv/bin/python'), str(ROOT / 'scripts/report_rules_sft.py'),
                            '--run-dir', str(run)], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
