#!/usr/bin/env python3
"""Portable ms-swift plumbing check; tiny samples are not research results."""
import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import run_v7_evaluation as ev

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save(path, value):
    ev.save_json(path, value)


def write_rows(path, values):
    ev.save_rows(path, values)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, default=ROOT / 'model_cache/Qwen3-VL-4B-Instruct')
    ap.add_argument('--work', type=Path, default=ROOT / 'runs/environment_smoke_20261024')
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--swift', type=Path, default=Path(sys.executable).parent / 'swift')
    ap.add_argument('--steps', type=int, default=2)
    ap.add_argument('--max-pixels', type=int, default=262144)
    ap.add_argument('--include-pt', action='store_true', help='Also check rule TEXT/VISION raw-material PT')
    args = ap.parse_args()
    args.work = args.work.resolve()
    args.model = args.model.resolve()
    work = args.work
    work.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for key in list(env):
        if key in {'NPROC_PER_NODE', 'NNODES', 'NODE_RANK', 'MASTER_ADDR', 'MASTER_PORT',
                   'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'GROUP_RANK',
                   'ROLE_RANK', 'ROLE_WORLD_SIZE'} or key.startswith('TORCHELASTIC_'):
            env.pop(key)
    cache = ROOT / '.runtime/cache'
    env.update(CUDA_VISIBLE_DEVICES=args.gpu, TMPDIR=str(ROOT / '.runtime/tmp'),
               HF_HOME=str(cache / 'huggingface'), MODELSCOPE_CACHE=str(ROOT / 'model_cache/modelscope'),
               HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', MAX_PIXELS=str(args.max_pixels),
               XDG_CACHE_HOME=str(cache), MPLCONFIGDIR=str(cache / 'matplotlib'),
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', PYTHONUNBUFFERED='1',
               PATH=str(args.swift.parent) + os.pathsep + env['PATH'])
    swift = str(args.swift.resolve())
    versions = {name: importlib.metadata.version(name) for name in
                ('torch', 'torchvision', 'ms-swift', 'transformers', 'peft', 'accelerate', 'datasets', 'trl')}
    save(work / 'environment.json', {'versions': versions, 'python': sys.version,
         'model': str(args.model), 'steps': args.steps, 'max_pixels': args.max_pixels,
         'purpose': 'Execution and checkpoint validation only; not knowledge-acquisition evidence'})

    def run(name, command):
        marker = work / (name + '.state.json')
        if marker.exists() and json.loads(marker.read_text()).get('status') == 'complete':
            assert json.loads(marker.read_text())['command'] == command, f'Changed command for {name}'
            print('already complete', name, flush=True)
            return
        save(marker, {'status': 'running', 'command': command, 'started_at': ev.now()})
        print('starting', name, flush=True)
        with (work / (name + '.log')).open('a') as log:
            result = subprocess.run(command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        save(marker, {'status': 'complete' if result.returncode == 0 else 'failed',
                     'command': command, 'returncode': result.returncode, 'ended_at': ev.now()})
        if result.returncode:
            raise RuntimeError(f'{name} failed; see {work / (name + ".log")}')

    datasets = {'rules': ROOT / 'data/rules_repaired_single_image_v7',
                'entities': ROOT / 'data/entity1020/merged_1109'}
    sources, checkpoints, reports, checkpoint_audits = {}, {}, [], []
    for dataset, root in datasets.items():
        out = work / dataset
        out.mkdir(exist_ok=True)
        ids = {r['id'] for r in rows(root / 'shared/train_text.jsonl')[:2]}
        qids = {r['qid'] for r in rows(root / 'shared/train_text.jsonl')[:2]}
        rule_knowledge = {}
        if dataset == 'rules':
            from build_v7_experiment_data import name as rule_name
            knowledge_path = root / 'audit/rule_knowledge.jsonl'
            rule_knowledge = {r['content']['id']: r['content'] for r in rows(knowledge_path)}
            sources[str(knowledge_path)] = ev.file_hash(knowledge_path)
        train_inputs = {}
        for modality, filename in [('text', 'train_text.jsonl'), ('vision', 'train_multimodal.jsonl')]:
            source = root / 'shared' / filename
            sources[str(source)] = ev.file_hash(source)
            selected = [copy.deepcopy(r) for r in rows(source) if r['id'] in ids]
            assert len(selected) == 2
            sft, pt = [], []
            for row in selected:
                material = row['messages'][0]['content']
                images = [str((root / p).resolve()) for p in row.get('images', [])]
                assert all(Path(p).is_file() for p in images)
                raw = {'messages': row['messages']}
                if images:
                    raw['images'] = images
                pt.append(raw)
                prompt = ('<image>Describe the knowledge conveyed by this reference image and its caption.'
                          if images else 'State the following learned knowledge unit.')
                if rule_knowledge:
                    prompt = (f'Explain the rule or codebook named {rule_name(rule_knowledge[row["id"]])}, '
                              'including its conditions, mappings, and how to apply it.')
                    if images:
                        prompt = '<image>Using this learning reference image, ' + prompt[0].lower() + prompt[1:]
                # Keep original learning target; never obtain supervision from eval labels.
                target = material.replace('<image>', '').strip()
                sample = {'messages': [{'role': 'user', 'content': prompt},
                                       {'role': 'assistant', 'content': target}]}
                if images:
                    sample['images'] = images
                sft.append(sample)
            for objective, samples in [('sft', sft), ('pt', pt)]:
                path = out / f'{objective}_{modality}.jsonl'
                write_rows(path, samples)
                train_inputs[objective, modality] = path

        probes, gold = {}, {}
        labels_path = root / 'audit/eval_labels.jsonl'
        labels = {r['id']: r for r in rows(labels_path)}
        sources[str(labels_path)] = ev.file_hash(labels_path)
        for direction in ('T2T', 'T2V', 'V2V', 'V2T'):
            path = root / direction / 'eval.jsonl'
            sources[str(path)] = ev.file_hash(path)
            selected = [r for r in rows(path) if r['qid'] in qids]
            assert len(selected) == 4
            probes[direction] = [ev.prepare_row(r, root, direction) for r in selected]
            gold[direction] = {r['id']: labels[r['id']]['answer'] for r in selected}
            write_rows(out / f'eval_{direction}.input.jsonl',
                       [{k: r[k] for k in ('messages', 'images') if k in r} for r in probes[direction]])
        save(out / 'selection.json', {'training_ids': sorted(ids), 'qids': sorted(qids),
                                    'selection': 'first two units, fixed before inference', 'eval_count': 4})

        jobs = [('sft', m) for m in ('text', 'vision')]
        if args.include_pt and dataset == 'rules':
            jobs += [('pt', m) for m in ('text', 'vision')]
        for objective, modality in jobs:
            name = f'{dataset}_{objective}_{modality}'
            output = out / f'{objective}_{modality}'
            command = [swift, objective, '--model', str(args.model), '--dataset', str(train_inputs[objective, modality]),
                '--tuner_type', 'lora', '--target_modules', 'all-linear', '--lora_rank', '8', '--lora_alpha', '32',
                '--freeze_vit', 'true', '--freeze_aligner', 'true', '--torch_dtype', 'bfloat16',
                '--attn_impl', 'sdpa', '--per_device_train_batch_size', '1', '--gradient_accumulation_steps', '1',
                '--gradient_checkpointing', 'true', '--learning_rate', '1e-4', '--max_steps', str(args.steps),
                '--max_length', '2048', '--max_pixels', str(args.max_pixels), '--warmup_ratio', '0',
                '--split_dataset_ratio', '0', '--save_strategy', 'steps', '--save_steps', str(args.steps),
                '--save_total_limit', '2', '--logging_steps', '1', '--seed', '42', '--data_seed', '42',
                '--add_version', 'false', '--output_dir', str(output), '--report_to', 'none',
                '--dataloader_num_workers', '0', '--dataloader_persistent_workers', 'false',
                '--strict', 'true', '--enable_thinking', 'false',
                '--use_chat_template', 'true' if objective == 'sft' else 'false',
                '--loss_scale', 'default' if objective == 'sft' else 'all']
            run(name + '_train', command)
            cp = output / f'checkpoint-{args.steps}'
            state = json.loads((cp / 'trainer_state.json').read_text())
            losses = [r['loss'] for r in state['log_history'] if 'loss' in r]
            assert state['global_step'] == args.steps
            assert losses and all(math.isfinite(v) and v > 0 for v in losses)
            from safetensors import safe_open
            with safe_open(str(cp / 'adapter_model.safetensors'), framework='pt', device='cpu') as tensors:
                keys = list(tensors.keys())
                assert any('lora_B' in k and tensors.get_tensor(k).count_nonzero() for k in keys)
                assert not any('.visual.' in k for k in keys), 'Vision backbone must stay frozen'
            for filename in ('adapter_config.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth'):
                assert (cp / filename).is_file(), filename
            checkpoints[dataset, objective, modality] = cp
            checkpoint_audits.append({'job': name, 'checkpoint': str(cp), 'global_step': args.steps,
                                      'losses': losses, 'lora_B_updated': True, 'vision_frozen': True,
                                      'adapter_sha256': ev.file_hash(cp / 'adapter_model.safetensors')})

        arms = [('base', None, ('T2T', 'T2V'))]
        arms += [(f'{obj}_{mod}', checkpoints[dataset, obj, mod],
                  ('T2T', 'T2V') if mod == 'text' else ('V2V', 'V2T')) for obj, mod in jobs]
        for arm, adapter, directions in arms:
            # Reuse one model load for both test modalities; whitelist model input fields.
            expected, gold_by_id = [], {}
            for direction in directions:
                for row in probes[direction]:
                    r = copy.deepcopy(row)
                    original_id = r['id']
                    r['id'] = direction + ':' + original_id
                    expected.append(r)
                    gold_by_id[r['id']] = gold[direction][original_id]
            inp = out / f'{arm}.eval.input.jsonl'
            write_rows(inp, [{k: r[k] for k in ('messages', 'images') if k in r} for r in expected])
            raw = out / f'{arm}.raw.jsonl'
            command = [swift, 'infer', '--model', str(args.model), '--load_args', 'false',
                '--use_chat_template', 'true',
                '--infer_backend', 'transformers', '--torch_dtype', 'bfloat16', '--attn_impl', 'sdpa',
                '--val_dataset', str(inp), '--val_dataset_shuffle', 'false', '--dataset_shuffle', 'false',
                '--remove_unused_columns', 'true', '--strict', 'true', '--max_batch_size', '1',
                '--write_batch_size', '1', '--max_new_tokens', '16', '--temperature', '0',
                '--seed', '42', '--max_pixels', str(args.max_pixels), '--enable_thinking', 'false',
                '--stream', 'false', '--result_path', str(raw)]
            if adapter:
                command += ['--adapters', str(adapter)]
            run(f'{dataset}_{arm}_eval', command)
            predictions = ev.align(rows(raw), expected, raw)
            assert len(predictions) == len(expected)
            scored = []
            for r in expected:
                response = predictions[r['id']]['response']
                assert response.strip(), r['id']
                letter = ev.letter(response, 'strict')
                scored.append({'id': r['id'], 'gold': gold_by_id[r['id']], 'response': response,
                               'predicted': letter, 'correct': letter == gold_by_id[r['id']]})
            write_rows(out / f'{arm}.scored.jsonl', scored)
            for direction in directions:
                subset = [r for r in scored if r['id'].startswith(direction + ':')]
                reports.append({'dataset': dataset, 'arm': arm, 'direction': direction,
                                'count': len(subset), 'valid': sum(r['predicted'] is not None for r in subset),
                                'correct': sum(r['correct'] for r in subset),
                                'accuracy': sum(r['correct'] for r in subset) / len(subset)})

    assert all(ev.file_hash(Path(path)) == digest for path, digest in sources.items())
    save(work / 'source_hashes.json', sources)
    save(work / 'checkpoint_audit.json', checkpoint_audits)
    save(work / 'summary.json', {'status': 'pass', 'completed_at': ev.now(), 'versions': versions,
        'checkpoints': checkpoint_audits, 'evaluation': reports, 'source_files_unchanged': True,
        'limitations': 'Tiny execution check only; no claim of learned knowledge or transfer accuracy.'})
    with (work / 'summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(reports[0]))
        writer.writeheader()
        writer.writerows(reports)
    print('PASS: training, adapter update, save/reload and both test modalities', flush=True)


if __name__ == '__main__':
    main()
