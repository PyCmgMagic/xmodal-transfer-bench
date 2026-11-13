#!/usr/bin/env python3
"""Freeze and run full-data LoRA SFT/PT learning; no test-based checkpoint selection."""
import argparse
import copy
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from build_v7_experiment_data import name as rule_name
from run_v7_evaluation import file_hash, now, save_json, save_rows
from v7_gpu_resources import alive, reserve

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def immutable(path, value):
    content = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    if path.exists():
        assert path.read_text() == content, f'Frozen configuration changed: {path}'
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


def prepare(args):
    work = args.work
    manifest = work / 'manifest.json'
    if manifest.exists():
        result = json.loads(manifest.read_text())
        assert result['config'] == configuration(args), 'Cannot change a frozen run'
        for path, digest in result['sources'].items():
            assert file_hash(Path(path)) == digest, path
        for job in result['jobs']:
            assert file_hash(Path(job['dataset'])) == job['dataset_sha256']
        return result
    jobs, sources, images = [], {}, {}
    for dataset, relative, count in [('rules', 'data/rules_repaired_single_image_v7', 1000),
                                      ('entities', 'data/entity1020/merged_1109', 1109)]:
        root = ROOT / relative
        source_text = root / 'shared/train_text.jsonl'
        source_vision = root / 'shared/train_multimodal.jsonl'
        values = {'text': rows(source_text), 'vision': rows(source_vision)}
        assert {r['id'] for r in values['text']} == {r['id'] for r in values['vision']}
        assert all(len(v) == len({r['id'] for r in v}) == count for v in values.values())
        for path in (source_text, source_vision):
            sources[str(path)] = file_hash(path)
        if dataset == 'rules':
            path = root / 'audit/rule_knowledge.jsonl'
            knowledge = {r['content']['id']: r['content'] for r in rows(path)}
        else:
            path = root / 'source/entities.jsonl'
            knowledge = {r['qid']: r for r in rows(path)}
        sources[str(path)] = file_hash(path)
        for modality, raw in values.items():
            prepared = {'sft': [], 'pt': []}
            for original in raw:
                assert [m['role'] for m in original['messages']] == ['assistant']
                image_paths = [str((root / p).resolve()) for p in original.get('images', [])]
                assert len(image_paths) == int(modality == 'vision')
                for path in image_paths:
                    assert Path(path).is_file(), path
                    images[path] = file_hash(Path(path))
                material = original['messages'][0]['content']
                assert material.count('<image>') == len(image_paths)
                pt = {'messages': copy.deepcopy(original['messages'])}
                if image_paths:
                    pt['images'] = image_paths
                prepared['pt'].append(pt)
                if dataset == 'rules':
                    prompt = (f'Explain the rule or codebook named {rule_name(knowledge[original["id"]])}, '
                              'including its conditions, mappings, and how to apply it.')
                    if image_paths:
                        prompt = '<image>Using this learning reference image, ' + prompt[0].lower() + prompt[1:]
                elif image_paths:
                    prompt = '<image>Describe the learned facts about the entity shown in this reference image.'
                else:
                    prompt = f'State the learned facts about {knowledge[original["qid"]]["name"]}.'
                sft = {'messages': [{'role': 'user', 'content': prompt},
                                    {'role': 'assistant', 'content': material.replace('<image>', '', 1).strip()}]}
                if image_paths:
                    sft['images'] = image_paths
                prepared['sft'].append(sft)
            for objective, samples in prepared.items():
                path = work / 'data' / f'{dataset}_{objective}_{modality}.jsonl'
                save_rows(path, samples)
                jobs.append({'name': f'{dataset}_{objective}_{modality}', 'dataset_name': dataset,
                             'modality': modality, 'objective': objective, 'rows': count,
                             'dataset': str(path), 'dataset_sha256': file_hash(path),
                             'output': str(work / 'jobs' / f'{dataset}_{objective}_{modality}')})
    # Pair the two modalities, while keeping PT/SFT and datasets independent.
    jobs.sort(key=lambda j: (j['dataset_name'] != 'rules', j['objective'] != 'sft', j['modality']))
    save_json(work / 'learning_image_hashes.json', images)
    result = {'created_at': now(), 'config': configuration(args), 'sources': sources,
              'jobs': jobs, 'learning_images': len(images),
              'training_data_policy': 'Only designated learning materials; no eval questions or labels used.',
              'sft_protocol': {'rules': 'Existing named rule explanation prompt and original modality-specific target.',
                               'entities_text': 'Entity name prompt; original textual learning material as target.',
                               'entities_vision': 'Reference image prompt; original visual learning caption as target.'},
              'checkpoint_selection': 'Final epoch 5 only; no selection using test accuracy.',
              'versions': {p: importlib.metadata.version(p) for p in
                           ('ms-swift', 'torch', 'transformers', 'peft', 'accelerate', 'datasets')}}
    save_json(manifest, result)
    return result


def configuration(args):
    return {'model': str(args.model), 'epochs': args.epochs, 'gpu': args.gpu,
            'rank': 8, 'alpha': 32, 'dropout': 0.05, 'learning_rate': 1e-4,
            'batch_size': 1, 'gradient_accumulation': 16, 'max_length': 8192,
            'max_pixels': 1003520, 'seed': 42, 'warmup_ratio': 0.05,
            'freeze_vit': True, 'freeze_aligner': True, 'attn_impl': 'sdpa'}


def command(args, job, resume=None):
    c = [str(ROOT / '.venv/bin/swift'), job['objective'], '--model', str(args.model),
         '--dataset', job['dataset'], '--tuner_type', 'lora', '--target_modules', 'all-linear',
         '--lora_rank', '8', '--lora_alpha', '32', '--lora_dropout', '0.05',
         '--torch_dtype', 'bfloat16', '--attn_impl', 'sdpa', '--freeze_vit', 'true',
         '--freeze_aligner', 'true', '--per_device_train_batch_size', '1',
         '--gradient_accumulation_steps', '16', '--learning_rate', '1e-4',
         '--num_train_epochs', str(args.epochs), '--gradient_checkpointing', 'true',
         '--max_length', '8192', '--max_pixels', '1003520', '--warmup_ratio', '0.05',
         '--save_strategy', 'epoch', '--save_total_limit', '5', '--save_only_model', 'false',
         '--split_dataset_ratio', '0', '--logging_steps', '5', '--seed', '42', '--data_seed', '42',
         '--add_version', 'false', '--output_dir', str(Path(job['output']) / 'checkpoint'),
         '--use_chat_template', 'true' if job['objective'] == 'sft' else 'false',
         '--loss_scale', 'default' if job['objective'] == 'sft' else 'all',
         '--strict', 'true', '--enable_thinking', 'false', '--report_to', 'none',
         '--dataset_num_proc', '1', '--dataloader_num_workers', '0',
         '--dataloader_persistent_workers', 'false']
    if resume:
        c += ['--resume_from_checkpoint', str(resume), '--load_args', 'false']
    return c


def validate_checkpoint(cp, epochs):
    import torch
    from safetensors import safe_open
    state = json.loads((cp / 'trainer_state.json').read_text())
    assert state['epoch'] >= epochs - 1e-6, state
    losses = [v['loss'] for v in state['log_history'] if 'loss' in v]
    assert losses and all(math.isfinite(float(v)) and float(v) >= 0 for v in losses)
    assert all((cp / n).is_file() for n in ('optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'adapter_config.json'))
    with safe_open(str(cp / 'adapter_model.safetensors'), framework='pt', device='cpu') as tensors:
        assert all(torch.isfinite(tensors.get_tensor(k)).all() for k in tensors.keys())
        assert any('lora_B' in k and tensors.get_tensor(k).count_nonzero() for k in tensors.keys())
        assert not any('.visual.' in k for k in tensors.keys())
    return {'checkpoint': str(cp), 'epoch': state['epoch'], 'global_step': state['global_step'],
            'first_logged_loss': losses[0], 'last_logged_loss': losses[-1],
            'adapter_sha256': file_hash(cp / 'adapter_model.safetensors'), 'lora_updated': True}


def train(args, job):
    folder = Path(job['output'])
    folder.mkdir(parents=True, exist_ok=True)
    state_path = folder / 'state.json'
    if state_path.exists():
        prior = json.loads(state_path.read_text())
        if prior.get('status') == 'complete':
            validate_checkpoint(Path(prior['checkpoint']), args.epochs)
            return prior
        if prior.get('status') == 'running' and alive(prior.get('pid')):
            raise RuntimeError(f'Existing training process is still alive: {job["name"]}, PID {prior["pid"]}')
    save_json(state_path, {'status': 'waiting_for_gpu', 'name': job['name'], 'updated_at': now()})
    output = folder / 'checkpoint'
    candidates = sorted(output.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]))
    candidates = [p for p in candidates if all((p / n).is_file() for n in
                  ('trainer_state.json', 'adapter_model.safetensors', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth'))]
    if candidates and json.loads((candidates[-1] / 'trainer_state.json').read_text())['epoch'] >= args.epochs - 1e-6:
        result = validate_checkpoint(candidates[-1], args.epochs)
    else:
        if output.exists() and not candidates:
            raise RuntimeError(f'Existing output has no recoverable checkpoint: {output}')
        cmd = command(args, job, candidates[-1] if candidates else None)
        budget = 17000 if job['modality'] == 'vision' else 14000
        with reserve(args.work, args.gpu, job['name'], budget) as reservation:
            with (folder / 'train.log').open('a') as log:
                child = subprocess.Popen(cmd, cwd=ROOT, env=os.environ.copy(), stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT)
                reservation.bind(child.pid)
                save_json(state_path, {'status': 'running', 'name': job['name'], 'pid': child.pid,
                                      'started_at': now(), 'command': cmd})
                print('started', job['name'], child.pid, flush=True)
                code = child.wait()
            if code:
                save_json(state_path, {'status': 'failed', 'returncode': code, 'ended_at': now(), 'command': cmd})
                raise RuntimeError(f'{job["name"]} failed with {code}; see {folder / "train.log"}')
        cps = sorted(output.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]))
        result = validate_checkpoint(cps[-1], args.epochs)
    result.update(status='complete', name=job['name'], ended_at=now())
    save_json(state_path, result)
    print('completed', job['name'], result['global_step'], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, default=ROOT / 'runs/qwen3_lora_learning_ep5_20261024')
    parser.add_argument('--model', type=Path, default=ROOT / 'model_cache/Qwen3-VL-4B-Instruct')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--workers', type=int, choices=[1, 2], default=2)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    args.work, args.model = args.work.resolve(), args.model.resolve()
    args.work.mkdir(parents=True, exist_ok=True)
    with (args.work / '.coordinator.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = prepare(args)
        for job in plan['jobs']:
            save_json(args.work / 'commands' / f'{job["name"]}.json', command(args, job))
        if args.prepare_only:
            print('prepared', len(plan['jobs']), 'jobs', flush=True)
            return
        os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, HF_HUB_OFFLINE='1',
                          PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        save_json(args.work / 'status.json', {'status': 'running', 'pid': os.getpid(),
                  'started_at': now(), 'jobs': 8, 'workers': args.workers})
        results, errors = [], []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = {pool.submit(train, args, job): job for job in plan['jobs']}
            for future in as_completed(pending):
                try:
                    results.append(future.result())
                except Exception as exc:
                    errors.append({'job': pending[future]['name'], 'error': str(exc)})
                    print('failed', errors[-1], flush=True)
                save_json(args.work / 'status.json', {'status': 'running', 'pid': os.getpid(),
                          'completed': len(results), 'failed': errors, 'total': 8, 'updated_at': now()})
        assert all(file_hash(Path(p)) == digest for p, digest in plan['sources'].items())
        save_json(args.work / 'status.json', {'status': 'failed' if errors else 'complete',
                  'completed': len(results), 'errors': errors, 'results': results, 'ended_at': now()})
        if errors:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
