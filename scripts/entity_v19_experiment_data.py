"""Frozen inputs for the approved entity v19 transfer experiment (no GPU imports)."""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import random
import re
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'entity_v19_1478'
DEFAULT_WORK = ROOT / 'runs/entity_v19_transfer_ep5_20261014'
MODELS = {
    'qwen3': ROOT / 'model_cache/modelscope/models/Qwen--Qwen3-VL-4B-Instruct/snapshots/master',
    'glm46': ROOT / 'model_cache/modelscope/models/ZhipuAI--GLM-4.6V-Flash/snapshots/master',
}
SEEDS = (42, 43, 44)
DIRECTIONS = ('T2T', 'T2V', 'V2T', 'V2V')
LATEST = 'compare-build-review-19-residual-scope-repair'


def evaluation_scope(work):
    """A user-approved scope override; the original frozen manifest is retained."""
    override = read(Path(work) / 'evaluation_scope.json')
    seeds = override.get('evaluation_seeds', list(SEEDS))
    assert seeds and len(seeds) == len(set(seeds)) and set(seeds) <= set(SEEDS) and 42 in seeds
    expected = {'base': 39200, 'curves': 78400, 'endpoint': 78400 * len(seeds), 'controls': 24576}
    expected['total'] = sum(expected.values())
    return {**override, 'evaluation_seeds': seeds, 'expected_predictions': expected,
            'expected_model_states': 2 + 2 * (8 + 2 * len(seeds))}


def in_evaluation_scope(path, work):
    parts = Path(path).relative_to(Path(work)).parts
    if not parts or parts[0] not in ('base', 'evaluation'):
        return False
    if parts[0] == 'base':
        return True
    return len(parts) > 2 and parts[2] in {f'seed_{s}' for s in evaluation_scope(work)['evaluation_seeds']}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else {}


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def write_rows(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def evaluator():
    spec = importlib.util.spec_from_file_location('entity_original_evaluator', DATA / 'scripts/run_evaluation.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rotate(text, rotation):
    """Move old choice i to (i-rotation)%4; preserve all question/suffix text."""
    found = list(re.finditer(r'(?m)^([A-D])\.\s', text))
    assert [m.group(1) for m in found] == list('ABCD')
    end = text.index('\n\nAnswer with exactly one uppercase letter', found[-1].end())
    options = [text[m.end():(found[i + 1].start() if i < 3 else end)].rstrip('\n')
               for i, m in enumerate(found)]
    result = text[:found[0].start()] + '\n'.join(
        f'{letter}. {options[(i + rotation) % 4]}' for i, letter in enumerate('ABCD')) + text[end:]
    assert rotation != 0 or result == text
    return result


def inference_only(row):
    assert set(row) <= {'id', 'qid', 'messages', 'images'}
    assert [m['role'] for m in row['messages']] in (['assistant'], ['system', 'user'])
    return {k: row[k] for k in ('messages', 'images') if k in row}


def freeze(work):
    work = Path(work)
    manifest = read(work / 'manifest.json')
    if manifest:
        verify(work)
        return manifest
    work.mkdir(parents=True, exist_ok=True)
    out = work / 'data'
    out.mkdir(exist_ok=True)
    ev = evaluator()
    source_paths = [DATA / d / name for d in DIRECTIONS for name in ('train.jsonl', 'eval.jsonl')]
    source_paths += [DATA / 'audit' / name for name in (
        'eval_labels.jsonl', 'sample_audit.jsonl', 'image_roles.jsonl', 'visual_review_notes.jsonl',
        'visual_transformations.jsonl')]
    source_paths += [DATA / 'build_manifest.json', DATA / 'source/knowledge_samples.jsonl',
                     DATA / 'source/evidence.jsonl', ROOT / 'entity_v19_full_experiment_plan.md',
                     DATA / 'scripts/run_evaluation.py']
    sources = []
    for p in source_paths:
        dest = out / 'frozen_sources' / (p.relative_to(DATA) if p.is_relative_to(DATA) else p.name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        sources.append({'path': str(p), 'sha256': sha(p), 'copy': str(dest)})
    labels = {r['id']: r for r in rows(DATA / 'audit/eval_labels.jsonl')}
    audit = {r['sample_id']: r for r in rows(DATA / 'audit/sample_audit.jsonl')}
    assert len(labels) == len(audit) == 2450
    knowledge = {r['id']: r for r in rows(DATA / 'source/knowledge_samples.jsonl')}
    train = {mode: rows(DATA / d / 'train.jsonl') for mode, d in (('TEXT', 'T2T'), ('VISION', 'V2V'))}
    units = sorted(r['id'] for r in train['TEXT'])
    assert len(set(units)) == len(units) == 1225
    assert set(units) == {r['id'] for r in train['VISION']}
    assert {r['unit_id'] for r in audit.values()} == set(units)
    # Confirm the source's duplicate direction inputs before reusing them.
    assert rows(DATA / 'T2T/eval.jsonl') == rows(DATA / 'V2T/eval.jsonl')
    assert rows(DATA / 'T2V/eval.jsonl') == rows(DATA / 'V2V/eval.jsonl')
    images = {'learning': set(), 'evaluation': set()}
    for mode, values in train.items():
        assert all(set(r) <= {'id', 'qid', 'messages', 'images'} for r in values)
        assert all([m['role'] for m in r['messages']] == ['assistant'] for r in values)
        for r in values:
            assert len(r.get('images', [])) == int(mode == 'VISION')
            images['learning'].update(r.get('images', []))
    prepared = {}
    for test, d in (('text', 'T2T'), ('image', 'T2V')):
        values = rows(DATA / d / 'eval.jsonl')
        assert len(values) == len(labels) and {r['id'] for r in values} == set(labels)
        prepared[test] = [ev.prepare_row(r, DATA, d) for r in values]
        for r in values:
            images['evaluation'].update(r.get('images', []))
    assert len(images['learning']) == len(images['evaluation']) == 1225
    image_manifest = []
    image_hashes = {}
    for role, names in images.items():
        image_hashes[role] = set()
        for name in sorted(names):
            p = DATA / name
            digest = sha(p)
            image_hashes[role].add(digest)
            image_manifest.append({'path': str(p), 'relative_path': name, 'role': role,
                                   'sha256': digest, 'bytes': p.stat().st_size})
    assert not image_hashes['learning'] & image_hashes['evaluation'], 'Train/test image SHA overlap'
    write_rows(out / 'image_inventory.jsonl', image_manifest)
    # Pair order is explicit before Swift shuffling; both training modes share seed/order.
    for seed in SEEDS:
        order = units.copy()
        random.Random(seed).shuffle(order)
        write_rows(out / f'unit_order_seed_{seed}.jsonl', [{'unit_id': uid} for uid in order])
        for mode, values in train.items():
            indexed = {r['id']: r for r in values}
            output = []
            for uid in order:
                r = copy.deepcopy(indexed[uid])
                if r.get('images'):
                    r['images'] = [str(DATA / n) for n in r['images']]
                output.append(inference_only(r))
            write_rows(out / f'train_{mode}_seed_{seed}.jsonl', output)
    sampled = random.Random(20261014).sample(units, 256)
    # Rotate sorted-by-QID units by one: deterministic cross-QID derangement.
    qids = {uid: next(r['qid'] for r in audit.values() if r['unit_id'] == uid) for uid in sampled}
    ordered = sorted(sampled, key=lambda uid: qids[uid])
    assert len(set(qids.values())) == 256
    wrong = dict(zip(ordered, ordered[1:] + ordered[:1]))
    image_by_unit = {audit[r['id']]['unit_id']: r['images'][0] for r in prepared['image']}
    save(out / 'control_sample.json', {'seed': 20261014, 'unit_ids': sampled, 'wrong_image_unit': wrong,
                                    'selected_before_inference': True})
    derived_labels = []
    for test, inputs in prepared.items():
        main = []
        controls = []
        for r in inputs:
            sid = r['id']
            uid = audit[sid]['unit_id']
            for rotation in range(4):
                item = copy.deepcopy(r)
                item['id'] = f'{test}|r{rotation}|{sid}'
                item['messages'][-1]['content'] = rotate(item['messages'][-1]['content'], rotation)
                main.append(item)
                gold = 'ABCD'[('ABCD'.index(labels[sid]['answer']) - rotation) % 4]
                meta = {'id': item['id'], 'source_id': sid, 'unit_id': uid, 'qid': r['qid'],
                        'probe_id': labels[sid]['probe_id'], 'rotation': rotation, 'test': test,
                        'answer': gold, 'generation_version': knowledge[uid]['content']['prompt_version']}
                derived_labels.append(meta)
                if test == 'image' and uid in wrong:
                    for condition in ('no_image', 'wrong_image'):
                        control = copy.deepcopy(item)
                        control['id'] = f'{condition}|r{rotation}|{sid}'
                        if condition == 'no_image':
                            control.pop('images')
                            control['messages'][-1]['content'] = control['messages'][-1]['content'].replace('<image>', '').lstrip('\n')
                        else:
                            control['images'] = [image_by_unit[wrong[uid]]]
                            assert qids[wrong[uid]] != r['qid']
                            assert control['images'] != item['images']
                        controls.append(control)
                        derived_labels.append({**meta, 'id': control['id'], 'test': condition,
                                               'wrong_image_unit': wrong[uid] if condition == 'wrong_image' else None})
        write_rows(out / f'eval_{test}.jsonl', main)
        if controls:
            write_rows(out / 'eval_controls.jsonl', controls)
    assert len(derived_labels) == 23696  # 19600 main + 4096 controls per model state.
    write_rows(out / 'labels.jsonl', derived_labels)
    # Every question visits each answer position once, including controls.
    groups = {}
    for r in derived_labels:
        groups.setdefault((r['test'], r['source_id']), []).append(r['answer'])
    assert all(sorted(v) == list('ABCD') for v in groups.values())
    deps = {}
    for package in ('ms-swift', 'transformers', 'torch', 'peft', 'accelerate', 'datasets', 'safetensors', 'pillow'):
        deps[package] = importlib.metadata.version(package)
    # Freeze model shards, processor/tokenizer and actual installed templates.
    model_inventory = []
    for model, folder in MODELS.items():
        for p in sorted(folder.rglob('*')):
            if p.is_file() and not any(part.startswith('.') for part in p.relative_to(folder).parts):
                model_inventory.append({'model': model, 'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)})
        print(f'Frozen model files: {model}', flush=True)
    swift = Path(importlib.util.find_spec('swift').origin).parent
    for folder in (swift / 'template', swift / 'model'):
        for p in sorted(folder.rglob('*.py')):
            model_inventory.append({'model': 'installed_swift_code', 'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha(p)})
    write_rows(out / 'model_inventory.jsonl', model_inventory)
    for p in sorted((ROOT / 'scripts').glob('*entity_v19*')):
        if p.is_file() and p.suffix in ('.py', '.sh'):
            shutil.copy2(p, out / 'frozen_sources' / p.name)
    manifest = {
        'schema': 'entity-v19-transfer-ep5-v1', 'created_at': now(), 'root': str(work), 'dataset': str(DATA),
        'models': {k: str(v) for k, v in MODELS.items()}, 'seeds': SEEDS, 'epochs': 5, 'training_jobs': 12,
        'units': 1225, 'questions_per_direction': 2450, 'rotations': 4,
        'expected_predictions': {'base': 39200, 'curves': 78400, 'endpoint': 235200, 'controls': 24576, 'total': 377376},
        'training': {'objective': 'pt', 'rank': 8, 'alpha': 32, 'dropout': .05, 'learning_rate': 1e-4,
                     'batch': 1, 'grad_accum': 16, 'max_length': 8192, 'max_pixels': 1003520,
                     'warmup': .05, 'freeze_vit': True, 'freeze_aligner': True, 'save_all_epochs': True,
                     'endpoint_selection': 'fixed epoch 5; never test accuracy', 'dataset_shuffle': False,
                     'dataloader_shuffle': 'seed/data_seed shared between TEXT and VISION'},
        'inference': {'max_new_tokens': 32, 'temperature': 0, 'seed': 42, 'batch': 1, 'thinking': False,
                      'qwen3_parser': 'strict', 'glm46_parser': 'glm-box', 'gold_in_model_input': False,
                      'duplicate_inputs': 'deduplicate exact prompt+image; deterministic response reused'},
        'baseline_reuse': {'T2T': 'text', 'V2T': 'text', 'T2V': 'image', 'V2V': 'image'},
        'dependencies': deps, 'sources': sources,
        'limitations': ['Images lack exhaustive human semantic review',
                       'Most facts use Wiki version deltas; first-publication dates are largely unavailable',
                       'VISION includes anonymous factual text supervision'],
        'outputs': {str(p.relative_to(work)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()},
    }
    save(work / 'manifest.json', manifest)
    save(work / 'preflight/data_validation.json', {'status': 'pass', 'units': 1225, 'labels': 2450,
         'selected_images': 2450, 'global_learning_eval_sha_disjoint': True,
         'control_units': 256, 'balanced_rotations': True, 'gold_isolated': True,
         'generation_versions': dict(Counter(r['generation_version'] for r in derived_labels if r['test'] == 'text' and r['rotation'] == 0)),
         'checked_at': now(), 'semantic_review': 'Metadata/path/SHA checks only; not full human image review'})
    return manifest


def verify(work, models=False):
    manifest = read(Path(work) / 'manifest.json')
    assert manifest
    for source in manifest['sources']:
        assert sha(source['path']) == source['sha256'], source['path']
    for name, digest in manifest['outputs'].items():
        assert sha(Path(work) / name) == digest, name
    for image in rows(Path(work) / 'data/image_inventory.jsonl'):
        assert sha(image['path']) == image['sha256'], image['path']
    if models:
        for item in rows(Path(work) / 'data/model_inventory.jsonl'):
            assert sha(item['path']) == item['sha256'], item['path']
