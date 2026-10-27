#!/usr/bin/env python3
"""Train fresh dense SFT models on rules-only text and vision data."""
import argparse
from contextlib import nullcontext
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import shutil

import run_v7_evaluation as ev
from v7_gpu_resources import reserve

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'model_cache/Qwen3-VL-4B-Instruct'
SOURCE = ROOT / 'runs/qwen3_lora_learning_ep5_20261024/data'
SOURCE_MANIFEST = SOURCE.parent / 'manifest.json'
WORK = ROOT / 'runs/rules_full_sft_ep10_20261026'
GPU_LEDGER = ROOT / 'runs/qwen3_lora_cot_eval_20261025'
MODALITIES = ('text', 'vision')
PLUGIN = ROOT / 'scripts/rules_dense_training_audit.py'


def read(path):
    return json.loads(path.read_text())


def model_identity():
    files = sorted(p for p in MODEL.iterdir() if p.is_file() and
                   (p.name.endswith('.safetensors') or p.name in {
                       'config.json', 'generation_config.json', 'model.safetensors.index.json',
                       'preprocessor_config.json', 'processor_config.json', 'tokenizer.json',
                       'tokenizer_config.json', 'chat_template.jinja'}))
    assert files and (MODEL / 'config.json').is_file()
    return {'path': str(MODEL), 'files': {p.name: ev.file_hash(p) for p in files}}


def relocate_images(rows):
    image_root = os.environ.get('XMODAL_IMAGE_ROOT')
    if not image_root:
        return rows
    image_root = Path(image_root).resolve()
    marker = ('rules_repaired_single_image_v7', 'images')
    for row in rows:
        relocated = []
        for image in row.get('images', []):
            parts = Path(image).parts
            offset = next((i for i in range(len(parts) - 1)
                           if parts[i:i + 2] == marker), None)
            if offset is None and not Path(image).is_absolute() and parts[:1] == ('images',):
                suffix = parts[1:]
            elif offset is not None:
                suffix = parts[offset + 2:]
            else:
                raise ValueError(f'Image is outside the rules dataset: {image}')
            path = image_root.joinpath(*suffix).resolve()
            if not path.is_relative_to(image_root):
                raise ValueError(f'Image escapes the rules dataset: {image}')
            if not path.is_file():
                raise FileNotFoundError(path)
            relocated.append(str(path))
        if 'images' in row:
            row['images'] = relocated
    return rows


def prepare(work):
    work.mkdir(parents=True, exist_ok=True)
    manifest_path = work / 'manifest.json'
    if manifest_path.exists():
        manifest = read(manifest_path)
        assert manifest['epochs'] == 10 and manifest['dataset_scope'] == ['rules']
        assert manifest['training']['optimizer'] == 'adamw_bnb_8bit'
        if Path(manifest['model']['path']).resolve() != MODEL.resolve():
            raise ValueError('Model path changed for a frozen run; use a new --work directory')
        if manifest.get('image_root') != os.environ.get('XMODAL_IMAGE_ROOT'):
            raise ValueError('Image root changed for a frozen run; use a new --work directory')
        for rel, digest in manifest['frozen_files'].items():
            assert ev.file_hash(work / rel) == digest, rel
        return manifest

    old_manifest = read(SOURCE_MANIFEST)
    old_jobs = {job['name']: job for job in old_manifest['jobs']}
    copied = {}
    for modality in MODALITIES:
        source = SOURCE / f'rules_sft_{modality}.jsonl'
        assert ev.file_hash(source) == old_jobs[f'rules_sft_{modality}']['dataset_sha256']
        rows = ev.read_rows(source)[0]
        assert len(rows) == 1000
        assert all([m['role'] for m in r['messages']] == ['user', 'assistant'] for r in rows)
        assert all(len(r.get('images', [])) == (1 if modality == 'vision' else 0) for r in rows)
        rows = relocate_images(rows)
        target = work / 'data' / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if os.environ.get('XMODAL_IMAGE_ROOT'):
            ev.save_rows(target, rows)
        else:
            target.write_bytes(source.read_bytes())
        copied[str(target.relative_to(work))] = ev.file_hash(target)
        smoke_rows = sorted(rows, key=lambda r: -sum(len(m['content']) for m in r['messages']))[:4]
        ev.save_rows(work / 'data' / f'smoke_{modality}.jsonl', smoke_rows)
        copied[f'data/smoke_{modality}.jsonl'] = ev.file_hash(work / 'data' / f'smoke_{modality}.jsonl')
    images = {p: ev.file_hash(p) for r in ev.read_rows(work / 'data/rules_sft_vision.jsonl')[0]
              for p in r['images']}
    assert len(images) == 1000
    ev.save_json(work / 'image_hashes.json', images)
    copied['image_hashes.json'] = ev.file_hash(work / 'image_hashes.json')
    for script in (Path(__file__).resolve(), PLUGIN):
        target = work / 'source_snapshots' / script.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(script, target)
        copied[str(target.relative_to(work))] = ev.file_hash(target)
    model = model_identity()
    manifest = {
        'created_at': ev.now(), 'status': 'prepared', 'model': model,
        'dataset_scope': ['rules'], 'modalities': list(MODALITIES), 'rows_per_modality': 1000,
        'epochs': 10, 'effective_batch_size': 16,
        'training': {'tuner_type': 'full', 'optimizer': 'adamw_bnb_8bit',
                     'dtype': 'bfloat16', 'per_device_batch_size': 1,
                     'gradient_accumulation_steps': 16, 'learning_rate': 1e-5,
                     'schedule': 'cosine', 'warmup_ratio': 0.05, 'seed': 42,
                     'max_length': 8192, 'max_pixels': 1003520,
                     'text_freezes': ['vision tower', 'aligner'],
                     'vision_freezes': [], 'save_only_model': False},
        'optimizer_note': 'BF16 dense trainable weights with 8-bit Adam states; no FP32 master weights. '
                          'No LoRA and no model weight quantization. Different optimizer from historical CPU Adam run.',
        'versions': {p: importlib.metadata.version(p) for p in
                     ('ms-swift', 'torch', 'transformers', 'bitsandbytes', 'accelerate')},
        'source_run': str(SOURCE), 'frozen_files': copied,
        'image_root': os.environ.get('XMODAL_IMAGE_ROOT'),
        'source_script': str(Path(__file__).resolve()),
        'scope_note': 'No entity data, evaluation data, or labels are used for training.'}
    ev.save_json(manifest_path, manifest)
    return manifest


def command(work, modality, smoke=False, smoke_steps=2, resume=None):
    dataset = work / 'data' / (f'smoke_{modality}.jsonl' if smoke else f'rules_sft_{modality}.jsonl')
    output = work / ('smoke' if smoke else 'jobs') / modality / 'checkpoint'
    interpreter_swift = Path(sys.executable).parent / 'swift'
    swift_bin = str(interpreter_swift) if interpreter_swift.is_file() else shutil.which('swift')
    if swift_bin is None:
        fallback = ROOT / '.venv/bin/swift'
        if not fallback.is_file():
            raise FileNotFoundError('swift CLI is not on PATH and project .venv was not found')
        swift_bin = str(fallback)
    cmd = [swift_bin, 'sft', '--model', str(MODEL),
           '--dataset', str(dataset), '--tuner_type', 'full',
           '--torch_dtype', 'bfloat16', '--attn_impl', 'sdpa',
           '--freeze_vit', str(modality == 'text').lower(),
           '--freeze_aligner', str(modality == 'text').lower(),
           '--per_device_train_batch_size', '1', '--gradient_accumulation_steps', '1' if smoke else '16',
           '--learning_rate', '1e-5', '--num_train_epochs', '10',
           '--gradient_checkpointing', 'true', '--max_length', '8192',
           '--max_pixels', '1003520', '--warmup_ratio', '0' if smoke else '0.05',
           '--save_strategy', 'steps' if smoke else 'epoch',
           '--save_total_limit', '2', '--save_only_model', 'false',
           '--split_dataset_ratio', '0', '--logging_steps', '1' if smoke else '5',
           '--seed', '42', '--data_seed', '42', '--add_version', 'false',
           '--output_dir', str(output), '--use_chat_template', 'true',
           '--loss_scale', 'default', '--strict', 'true', '--enable_thinking', 'false',
           '--report_to', 'none', '--dataset_num_proc', '1', '--dataloader_num_workers', '0',
           '--dataloader_persistent_workers', 'false',
           '--optim', 'adamw_bnb_8bit', '--model_type', 'qwen3_vl',
           '--external_plugins', str(work / 'source_snapshots' / PLUGIN.name),
           '--callbacks', 'rules_dense_training_audit', '--vit_gradient_checkpointing', 'true']
    if smoke:
        cmd += ['--max_steps', '2', '--save_steps', '1']
    if resume:
        cmd += ['--resume_from_checkpoint', str(resume), '--load_args', 'false']
    return cmd


def run_one(work, modality, smoke, smoke_steps=2):
    label = ('smoke/' if smoke else 'jobs/') + modality
    folder = work / label
    folder.mkdir(parents=True, exist_ok=True)
    state_path = folder / 'state.json'
    previous = read(state_path) if state_path.exists() else {}
    prior_pid = previous.get('pid')
    if prior_pid and Path(f'/proc/{prior_pid}/cmdline').exists():
        prior_command = Path(f'/proc/{prior_pid}/cmdline').read_bytes()
        if str(folder).encode() in prior_command:
            raise RuntimeError(f'Training process still alive: {prior_pid}')
    if previous.get('status') == 'complete' and (not smoke or previous.get('global_step', 0) >= smoke_steps):
        return
    checkpoints = sorted((folder / 'checkpoint').glob('checkpoint-*'),
                         key=lambda p: int(p.name.rsplit('-', 1)[-1]))
    complete = [p for p in checkpoints if all((p / n).is_file() for n in
                ('trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth'))]
    if (folder / 'checkpoint').exists() and not complete:
        raise RuntimeError(f'Existing output without resumable checkpoint: {folder}')
    cmd = command(work, modality, smoke, smoke_steps, complete[-1] if complete else None)
    ev.save_json(folder / 'command.json', cmd)
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE='1', MAX_PIXELS='1003520',
               PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               TOKENIZERS_PARALLELISM='false', RULES_DENSE_AUDIT=str(folder / 'parameter_audit.jsonl'),
               RULES_DENSE_MODALITY=modality, PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
    if not os.environ.get('SLURM_JOB_ID'):
        env.setdefault('CUDA_VISIBLE_DEVICES', '0')
    env['RULES_DENSE_STOP_AT'] = str(smoke_steps) if smoke else ''
    ev.save_json(work / 'status.json', {'status': 'waiting_for_gpu', 'task': label, 'pid': os.getpid(), 'updated_at': ev.now()})
    reservation = (nullcontext(None) if os.environ.get('SLURM_JOB_ID')
                   else reserve(GPU_LEDGER, '0', 'rules_full_sft_ep10:' + label, 41000, poll=15))
    with reservation as lease:
        with (folder / 'train.log').open('a') as log:
            child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            if lease is not None:
                lease.bind(child.pid)
            ev.save_json(state_path, {'status': 'running', 'pid': child.pid, 'command': cmd,
                                      'started_at': ev.now(), 'smoke': smoke})
            ev.save_json(work / 'status.json', {'status': 'running', 'task': label, 'pid': os.getpid(),
                         'worker_pid': child.pid, 'updated_at': ev.now()})
            code = child.wait()
    if code:
        ev.save_json(state_path, {'status': 'failed', 'returncode': code, 'command': cmd,
                                  'ended_at': ev.now()})
        raise RuntimeError(f'{label} failed; inspect {folder / "train.log"}')
    checkpoints = sorted((folder / 'checkpoint').glob('checkpoint-*'),
                         key=lambda p: int(p.name.rsplit('-', 1)[-1]))
    assert checkpoints, folder
    cp = checkpoints[-1]
    trainer_state = read(cp / 'trainer_state.json')
    assert all((cp / name).is_file() for name in
               ('optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'config.json'))
    assert list(cp.glob('model*.safetensors')) and not (cp / 'adapter_config.json').exists()
    losses = [float(x['loss']) for x in trainer_state['log_history'] if 'loss' in x]
    assert losses and all(math.isfinite(x) and x >= 0 for x in losses)
    if not smoke:
        assert abs(float(trainer_state['epoch']) - 10) < 1e-6, trainer_state
        assert trainer_state['global_step'] == 630, trainer_state
        assert list(cp.glob('model*.safetensors'))
    else:
        assert trainer_state['global_step'] == smoke_steps
    ev.save_json(state_path, {'status': 'complete', 'checkpoint': str(cp),
                  'epoch': trainer_state['epoch'], 'global_step': trainer_state['global_step'],
                  'first_loss': losses[0], 'last_loss': losses[-1], 'ended_at': ev.now()})


def main():
    global MODEL, SOURCE, SOURCE_MANIFEST
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--source-data', type=Path, default=SOURCE)
    parser.add_argument('--source-manifest', type=Path, default=SOURCE_MANIFEST)
    parser.add_argument('--image-root', type=Path, help='Rules images directory after migration')
    parser.add_argument('--work', type=Path, default=WORK)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    MODEL = args.model.resolve()
    SOURCE = args.source_data.resolve()
    SOURCE_MANIFEST = args.source_manifest.resolve()
    if args.image_root:
        os.environ['XMODAL_IMAGE_ROOT'] = str(args.image_root.resolve())
    work = args.work.resolve()
    manifest = prepare(work)
    if args.prepare_only:
        print(json.dumps({'status': 'prepared', 'rules': 1000, 'epochs': 10,
                          'modalities': list(MODALITIES), 'model': manifest['model']['path']}))
        return
    if not args.smoke_only and not args.train:
        parser.error('choose --prepare-only, --smoke-only, or --train')
    with ev.run_lock(work / '.coordinator.lock'):
        try:
            # Train one step and reload optimizer/scheduler for step two.
            for modality in MODALITIES:
                run_one(work, modality, smoke=True, smoke_steps=1)
                run_one(work, modality, smoke=True, smoke_steps=2)
            if args.train:
                for modality in MODALITIES:
                    run_one(work, modality, smoke=False)
            status = 'smoke_pass' if args.smoke_only else 'complete'
            ev.save_json(work / 'status.json', {'status': status, 'epochs': 10,
                         'modalities': list(MODALITIES), 'updated_at': ev.now()})
        except BaseException as exc:
            ev.save_json(work / 'status.json', {'status': 'failed', 'error': str(exc), 'updated_at': ev.now()})
            raise


if __name__ == '__main__':
    main()
