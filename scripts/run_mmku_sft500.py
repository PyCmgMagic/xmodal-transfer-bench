#!/usr/bin/env python3
"""Run MMKU 500-example full SFT and a paired before/after evaluation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def run(args):
    root, work, model = args.root.resolve(), args.work.resolve(), args.model.resolve()
    work.mkdir(parents=True, exist_ok=True)
    train_rows = [json.loads(x) for x in (work / 'data/train.jsonl').read_text().splitlines()]
    if len(train_rows) != 500:
        raise ValueError('Training dataset must contain exactly 500 records')
    manifest = json.loads((work / 'data/manifest.json').read_text())
    for name, digest in manifest['data_sha256'].items():
        if hashlib.sha256((work / 'data' / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Changed data file: {name}')
    if (work / 'train').exists() and any((work / 'train').iterdir()):
        raise ValueError('Training output is nonempty; use a new run directory')
    if args.train_gpu == args.eval_gpu and not args.sequential:
        raise ValueError('Using the same GPU for training and baseline requires --sequential')
    for gpu in set((args.train_gpu, args.eval_gpu)):
        info = subprocess.check_output(['nvidia-smi', '-i', gpu, '--query-gpu=memory.used,utilization.gpu',
                                        '--format=csv,noheader,nounits'], text=True).strip().split(',')
        if int(info[0]) > 1000 or int(info[1]) > 5:
            raise RuntimeError(f'GPU {gpu} is no longer idle: {info}')
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE='1', MODELSCOPE_OFFLINE='1', MAX_PIXELS='262144',
               PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', NPROC_PER_NODE='1',
               MASTER_ADDR='127.0.0.1', MASTER_PORT='30293')
    python = root / '.venv/bin/python'
    swift = root / '.venv/bin/swift'
    train_file = work / 'data/train.absolute.jsonl'
    for row in train_rows:
        row['images'] = [str((work / 'data' / p).resolve()) for p in row['images']]
    train_file.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in train_rows))
    if args.micro_batch_size <= 0 or 16 % args.micro_batch_size:
        raise ValueError('Micro batch size must divide the fixed effective batch size 16')
    command = [str(swift), 'sft', '--model', str(model), '--model_type', 'qwen3_vl',
               '--dataset', str(train_file), '--tuner_type', 'full', '--torch_dtype', 'bfloat16',
               '--attn_impl', 'sdpa', '--freeze_vit', 'false', '--freeze_aligner', 'false',
               '--per_device_train_batch_size', str(args.micro_batch_size),
               '--gradient_accumulation_steps', str(16 // args.micro_batch_size),
               '--learning_rate', '1e-5', '--num_train_epochs', '6', '--lr_scheduler_type', 'cosine',
               '--warmup_ratio', '0.05', '--gradient_checkpointing', 'true', '--vit_gradient_checkpointing', 'true',
               '--max_length', '2048', '--max_pixels', '262144', '--save_strategy', 'epoch',
               '--save_total_limit', '1', '--save_only_model', 'true', '--split_dataset_ratio', '0',
               '--logging_steps', '1', '--seed', '42', '--data_seed', '42', '--add_version', 'false',
               '--output_dir', str(work / 'train'), '--use_chat_template', 'true', '--loss_scale', 'default',
               '--strict', 'true', '--enable_thinking', 'false', '--report_to', 'none',
               '--dataset_num_proc', '1', '--dataloader_num_workers', '0',
               '--dataloader_persistent_workers', 'false', '--optim', args.optim,
               '--packing', 'false', '--padding_free', 'false', '--weight_decay', '0.01']
    save(work / 'train_command.json', command)
    plan = {'model': str(model), 'model_scope': 'Qwen3-VL-4B-Instruct, not the paper\'s 7B/8B backbones',
            'subset': 'Unknown', 'knowledge_count': 500, 'epochs': 6, 'effective_batch': 16,
            'learning_rate': 1e-5, 'tuner': 'full', 'freeze_vit': False, 'freeze_aligner': False,
            'micro_batch_size': args.micro_batch_size, 'optimizer': args.optim,
            'comparison': 'same official 500 heldout VQA before and after fine-tuning',
            'generation': 'temperature=0, max_tokens=64, official short-answer prompt, no added CoT',
            'checkpoint_selection': 'final epoch fixed in advance; no test-based checkpoint selection',
            'sequential': args.sequential, 'train_gpu': args.train_gpu, 'eval_gpu': args.eval_gpu,
            'status': 'running', 'pid': os.getpid(), 'started_at': time.time()}
    save(work / 'experiment.json', plan)

    def evaluation(model_path, name, gpu):
        cmd = [str(python), str(root / 'scripts/evaluate_mmku_subset.py'), '--data', str(work / 'data'),
               '--model', str(model_path), '--output', str(work / name),
               '--official-code', str(work / 'official'), '--batch-size', '8']
        log = (work / f'{name}.log').open('w')
        proc = subprocess.Popen(cmd, cwd=root, env=dict(env, CUDA_VISIBLE_DEVICES=gpu), stdout=log, stderr=subprocess.STDOUT)
        log.close()
        return proc

    baseline_report = work / 'baseline/report.json'
    baseline_complete = baseline_report.exists() and json.loads(baseline_report.read_text()).get('status') == 'complete'
    if baseline_complete:
        previous_config = json.loads((work / 'baseline/config.json').read_text())
        if (previous_config['model'] != str(model) or previous_config['dataset_sha256'] !=
                hashlib.sha256((work / 'data/test.jsonl').read_bytes()).hexdigest()):
            raise ValueError('Completed baseline used a different model or test dataset')
        baseline = None
    else:
        baseline = evaluation(model, 'baseline', args.eval_gpu)
    if args.sequential and baseline is not None:
        save(work / 'status.json', {'status': 'baseline_evaluation', 'baseline_pid': baseline.pid,
                                   'updated_at': time.time()})
        if baseline.wait():
            raise RuntimeError('Baseline evaluation failed; see baseline.log')
    with (work / 'train.log').open('w') as log:
        trainer = subprocess.Popen(command, cwd=root, env=dict(env, CUDA_VISIBLE_DEVICES=args.train_gpu), stdout=log, stderr=subprocess.STDOUT)
    save(work / 'status.json', {'status': 'training_and_baseline', 'train_pid': trainer.pid,
                               'baseline_pid': baseline.pid if baseline else None,
                               'baseline_already_complete': baseline_complete, 'updated_at': time.time()})
    return_code = trainer.wait()
    if return_code:
        save(work / 'status.json', {'status': 'training_failed', 'return_code': return_code,
                                   'baseline_pid': baseline.pid if baseline else None, 'updated_at': time.time()})
        raise RuntimeError(f'SFT exited with code {return_code}; see train.log')
    checkpoints = sorted((work / 'train').glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]))
    checkpoint = checkpoints[-1]
    state = json.loads((checkpoint / 'trainer_state.json').read_text())
    if abs(state['epoch'] - 6) > 1e-6:
        raise ValueError(f'Unexpected final training epoch: {state["epoch"]}')
    trained = evaluation(checkpoint, 'finetuned', args.train_gpu)
    save(work / 'status.json', {'status': 'evaluating', 'checkpoint': str(checkpoint),
                               'finetuned_pid': trained.pid, 'baseline_pid': baseline.pid if baseline else None, 'updated_at': time.time()})
    trained_code = trained.wait()
    baseline_code = baseline.wait() if baseline else 0
    if trained_code or baseline_code:
        raise RuntimeError(f'Evaluation failure: baseline={baseline_code}, finetuned={trained_code}')
    before = json.loads((work / 'baseline/report.json').read_text())
    after = json.loads((work / 'finetuned/report.json').read_text())
    if before['saved'] != 500 or after['saved'] != 500:
        raise ValueError('Incomplete heldout results')
    comparison = {'status': 'complete', 'knowledge_count': 500, 'baseline': before, 'finetuned': after,
                  'cem_gain_percentage_points': (after['cem_accuracy'] - before['cem_accuracy']) * 100,
                  'f1_gain_percentage_points': (after['mean_f1'] - before['mean_f1']) * 100,
                  'checkpoint': str(checkpoint), 'epoch': state['epoch'], 'completed_at': time.time()}
    save(work / 'comparison.json', comparison)
    save(work / 'status.json', {'status': 'complete', 'completed_at': time.time()})
    print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--work', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--train-gpu', default='0')
    ap.add_argument('--eval-gpu', default='1')
    ap.add_argument('--sequential', action='store_true')
    ap.add_argument('--micro-batch-size', type=int, default=4)
    ap.add_argument('--optim', default='adamw_torch')
    args = ap.parse_args()
    try:
        run(args)
    except Exception as error:
        save(args.work / 'failure.json', {'error': repr(error), 'time': time.time()})
        raise
