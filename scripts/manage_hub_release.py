#!/usr/bin/env python3
"""Prepare isolated GitHub/HF exports. Uploads require the explicit upload command."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    'rules': ('data/rules_repaired_single_image_v7', 1000, 2000, 3000),
    'entities': ('data/entity1020/merged_1109', 1109, 2218, 2218),
}
LEARNING = 'runs/qwen3_lora_learning_ep5_20261024'
TRAINING = 'runs/rules_full_sft_ep10_20261026'
CODE_FILES = [
    '.gitignore', 'requirements-runtime.txt', 'requirements-runtime.lock',
    'requirements-migration.txt', 'hub_sources.example.json', 'HUB_MIGRATION_ZH.md',
    'CROSS_MODAL_RESEARCH_PROTOCOL.md', 'cluster/kit/README_ZH.md',
    'cluster/kit/rules_sft.sbatch', 'cluster/kit/setup_env.sh',
    'cluster/kit/setup_env.sbatch', 'cluster/kit/fetch_assets.sbatch',
    'scripts/train_rules_full_sft_ep10.py', 'scripts/rules_dense_training_audit.py',
    'scripts/run_v7_evaluation.py', 'scripts/v7_gpu_resources.py',
    'scripts/answer_readout.py', 'scripts/fetch_hub_assets.py',
    'scripts/hf_with_proxy.sh', 'HF_CONNECTIVITY_ZH.md',
    'scripts/manage_hub_release.py', 'tests/test_hub_migration.py',
]
BINARY = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif', '.safetensors'}
MODEL_FILES = {
    'config.json', 'generation_config.json', 'model.safetensors.index.json',
    'tokenizer.json', 'tokenizer_config.json', 'preprocessor_config.json',
    'video_preprocessor_config.json', 'added_tokens.json', 'special_tokens_map.json',
    'vocab.json', 'merges.txt', 'chat_template.jinja', 'chat_template.json',
}
SECRET_PATTERNS = [
    r'\bhf_[A-Za-z0-9]{24,}\b', r'\bgh[pousr]_[A-Za-z0-9]{30,}\b',
    r'\bgithub_pat_[A-Za-z0-9_]{30,}\b', r'\bAKIA[A-Z0-9]{16}\b',
    r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def payload_files(folder):
    for path in sorted(folder.rglob('*')):
        relative = path.relative_to(folder)
        if path.is_file() and not any(p in {'.git', '.cache', '__pycache__'} for p in relative.parts):
            yield path


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def checked_path(root, relative):
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f'Unsafe relative path: {relative}')
    return path


def copy_asset(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    if source.suffix.lower() in BINARY:
        try:
            os.link(source, target)
            return 'hardlink'
        except OSError:
            raise RuntimeError('Use a release directory on the source filesystem for binary hardlinks')
    shutil.copy2(source, target)
    return 'copy'


def validate_shards(folder):
    index = json.loads((folder / 'model.safetensors.index.json').read_text())
    weight_map = index['weight_map']
    for filename in sorted(set(weight_map.values())):
        path = checked_path(folder, filename)
        with path.open('rb') as stream:
            size = struct.unpack('<Q', stream.read(8))[0]
            if size > 64 * 1024 * 1024:
                raise ValueError(f'Unexpected safetensors header size: {filename}')
            header = json.loads(stream.read(size))
        expected = {key for key, value in weight_map.items() if value == filename}
        if not expected.issubset(header):
            raise ValueError(f'Missing tensors in {filename}')
        payload_size = path.stat().st_size - size - 8
        if any(not 0 <= v['data_offsets'][0] <= v['data_offsets'][1] <= payload_size
               for k, v in header.items() if k != '__metadata__'):
            raise ValueError(f'Truncated tensor payload in {filename}')
    return {'shards': len(set(weight_map.values())), 'indexed_tensors': len(weight_map),
            'validation': 'header/index/size; tensor payload not rehashed'}


def write_manifest(folder):
    records = {}
    for path in payload_files(folder):
        if path.name != 'release_manifest.json':
            record = {'bytes': path.stat().st_size}
            if path.suffix.lower() not in BINARY:
                record['sha256'] = sha256(path)
            records[path.relative_to(folder).as_posix()] = record
    write_json(folder / 'release_manifest.json', {
        'format_version': 1, 'files': records,
        'large_binary_policy': 'Hardlinked unchanged; size checked. Run verify --full-hash to hash payloads.',
    })
    return {'files': len(records), 'bytes': sum(v['bytes'] for v in records.values()),
            'binary_files_not_rehashed': sum('sha256' not in v for v in records.values())}


def validate_dataset(folder):
    counts = {}
    for name, (relative, ntrain, neval, nimages) in DATASETS.items():
        root = folder / relative
        for direction in ('T2T', 'T2V', 'V2T', 'V2V'):
            for split, expected in [('train', ntrain), ('eval', neval)]:
                path = root / direction / (split + '.jsonl')
                samples = rows(path)
                if len(samples) != expected or len({r['id'] for r in samples}) != expected:
                    raise ValueError(f'Unexpected sample IDs/count: {path}')
                for sample in samples:
                    for image in sample.get('images', []):
                        if not checked_path(root, image).is_file():
                            raise FileNotFoundError(image)
        image_count = sum(p.is_file() for p in (root / 'images').rglob('*'))
        if image_count != nimages:
            raise ValueError(f'Image count mismatch: {name}: {image_count}')
        counts[name] = {'knowledge_units': ntrain, 'eval_per_direction': neval,
                        'images': image_count}
    return counts


def dataset_card():
    header = ['---', 'language:', '- en', 'task_categories:', '- visual-question-answering',
              'tags:', '- multimodal', '- cross-modal-transfer', 'configs:']
    for name, (rel, _, _, _) in DATASETS.items():
        for direction in ('T2T', 'T2V', 'V2T', 'V2V'):
            header += [f'- config_name: {name}_{direction.lower()}', '  data_files:',
                       '  - split: train', f'    path: {rel}/{direction}/train.jsonl',
                       '  - split: test', f'    path: {rel}/{direction}/eval.jsonl']
    header += ['---', '# Cross-modal transfer: rules and entities', '',
               'Rules: 1,000 knowledge units, 2,000 questions per direction, 3,000 original images.',
               'Entities: 1,109 knowledge units, 2,218 questions per direction, 2,218 original images.',
               'T2T/T2V share text learning; V2T/V2V share visual learning. These are not four independent training sets.',
               '', '## Paths and training', '',
               'Original data and audits are under data/. Image paths in each dataset resolve against that dataset root.',
               'Portable SFT/PT inputs are under learning/. Each images field is relative to its dataset root.',
               'learning/manifest.json hashes the portable inputs and records the original input hashes.',
               'Messages and evaluation labels are unchanged; only prepared learning image paths are rewritten.',
               'Use snapshot_download to fetch the entire tree; generic JSON loading does not automatically decode these image paths.',
               '', '## Scope and provenance', '',
               'The entity set includes 21 repaired candidate units without completed semantic/temporal certification.',
               'The dataset keeps the original 2025-12-31 temporal cutoff convention.',
               'Rule v7 contains 104 inspection-code questions; those conventions are described in the paired learning captions.',
               'Visual learning includes its original text caption; it is not a pixels-only protocol.',
               'Original audit/provenance files are retained; historical absolute paths in those records are not runtime paths.',
               'Training inputs never include evaluation questions or answer labels.',
               '', '## Rights and access', '',
               'No new redistribution license is granted by this export. Consult source/evidence and original documentation.',
               'Use a private repository for project migration; publication requires resolving source image/data rights.',
               'Original package hashes remain source provenance. The export release_manifest.json covers this export.',
               'Binary images are hardlinked locally without modification; full payload hashing is an explicit optional verification step.',
               '']
    return '\n'.join(header)


def prepare_dataset(source, output):
    for relative, _, _, _ in DATASETS.values():
        for path in sorted((source / relative).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix not in {'.py', '.pyc'}:
                copy_asset(path, output / path.relative_to(source))
    old_manifest = json.loads((source / LEARNING / 'manifest.json').read_text())
    jobs = []
    for job in old_manifest['jobs']:
        name = job['name']
        origin = source / LEARNING / 'data' / (name + '.jsonl')
        if sha256(origin) != job['dataset_sha256']:
            raise ValueError(f'Original learning input changed: {name}')
        samples = rows(origin)
        dataset_root = source / DATASETS[job['dataset_name']][0]
        for sample in samples:
            if 'images' in sample:
                sample['images'] = [Path(p).resolve().relative_to(dataset_root.resolve()).as_posix()
                                    for p in sample['images']]
                for image in sample['images']:
                    if not checked_path(dataset_root, image).is_file():
                        raise FileNotFoundError(image)
        target = output / 'learning' / (name + '.jsonl')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in samples))
        jobs.append({'name': name, 'dataset_name': job['dataset_name'],
                     'objective': job['objective'], 'modality': job['modality'],
                     'rows': len(samples), 'dataset_sha256': sha256(target),
                     'original_dataset_sha256': job['dataset_sha256']})
    write_json(output / 'learning/manifest.json', {
        'jobs': jobs, 'image_paths': 'relative to the matching data dataset root',
        'transformation': 'Only images paths in prepared learning inputs; original messages unchanged',
    })
    (output / 'README.md').write_text(dataset_card())
    (output / '.gitattributes').write_text(
        '\n'.join(f'*{ext} filter=lfs diff=lfs merge=lfs -text' for ext in sorted(BINARY)) + '\n')
    return validate_dataset(output)


def prepare_model(source, output, modality):
    origin = source / TRAINING / 'jobs' / modality / 'checkpoint/checkpoint-630'
    state = json.loads((origin / 'trainer_state.json').read_text())
    if state['epoch'] != 10 or state['global_step'] != 630:
        raise ValueError(f'Wrong final checkpoint: {modality}')
    output.mkdir(parents=True, exist_ok=True)
    for path in sorted(origin.iterdir()):
        if path.name in MODEL_FILES or path.name.endswith('.safetensors'):
            copy_asset(path, output / path.name)
    training = json.loads((source / TRAINING / 'manifest.json').read_text())
    write_json(output / 'training_summary.json', {
        'base_model': 'Qwen/Qwen3-VL-4B-Instruct', 'dataset': 'rules', 'modality': modality,
        'epoch': state['epoch'], 'global_step': state['global_step'],
        'training': training['training'], 'versions': training['versions'],
        'source_base_model_sha256': training['model']['files'],
        'checkpoint': 'Final epoch; no test-based checkpoint selection',
        'optimizer_state_exported': False,
    })
    scope = ('All language parameters updated; vision tower and aligner frozen.'
             if modality == 'text' else 'Language model, vision tower and aligner updated.')
    (output / 'README.md').write_text(
        '---\nbase_model: Qwen/Qwen3-VL-4B-Instruct\nlibrary_name: transformers\n'
        'tags:\n- qwen3_vl\n- cross-modal-transfer\n- full-fine-tuning\n---\n'
        f'# Rules {modality.upper()} SFT, 10 epochs\n\n'
        '1,000 rule learning examples; final checkpoint at step 630. No entity training examples.\n'
        f'{scope}\n'
        'BF16 full weights with 8-bit Adam states during training; no LoRA or weight quantization.\n'
        'Inference-only export includes processor/tokenizer and safetensors; optimizer and RNG states are excluded.\n'
        'This export supports evaluation or fresh fine-tuning, not exact optimizer-state continuation.\n'
        'See training_summary.json for settings, versions and original base-model SHA256 values.\n'
        'No new license is assigned; source model and training-data terms still apply. Default migration access is private.\n')
    (output / '.gitattributes').write_text('*.safetensors filter=lfs diff=lfs merge=lfs -text\n')
    return validate_shards(output)


def scan_code(folder, check_server_paths=True):
    findings = []
    for path in payload_files(folder):
        if path.name == 'release_manifest.json':
            continue
        content = path.read_text()
        for pattern in SECRET_PATTERNS:
            if re.search(pattern, content):
                findings.append({'file': path.relative_to(folder).as_posix(), 'kind': 'credential_pattern'})
        server_prefix = '/' + 'mnt' + '/' + 'vdb' + '/'
        if check_server_paths and (server_prefix in content or re.search(r'/home/(?!YOUR_|your_)[A-Za-z0-9_.-]+/', content)):
            findings.append({'file': path.relative_to(folder).as_posix(), 'kind': 'server_absolute_path'})
        if check_server_paths and path.stat().st_size > 5 * 1024 * 1024:
            findings.append({'file': path.relative_to(folder).as_posix(), 'kind': 'unexpected_large_code_file'})
    return findings


def prepare(source, release):
    if release.exists():
        raise FileExistsError(f'Refusing to replace an existing export: {release}')
    release.mkdir(parents=True)
    github = release / 'github'
    for relative in CODE_FILES:
        copy_asset(source / relative, github / relative)
    shutil.copy2(source / 'HUB_MIGRATION_ZH.md', github / 'README.md')
    report = source / 'runs/experiment_summary_20261027'
    for name in ('README_ZH.md', 'metrics.csv', 'training.csv', 'experiments.csv', 'validation.json'):
        content = (report / name).read_text().replace(str(source), '<PROJECT_ROOT>')
        target = github / 'reports/20261027' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    findings = scan_code(github)
    if findings:
        write_json(release / 'code_scan.json', findings)
        raise ValueError('Review code_scan.json before publishing')
    audit = {'created_at': datetime.now(timezone.utc).isoformat(), 'remote_actions': [],
             'code_scan': 'pass', 'binary_staging': 'hardlinks; do not edit staged binaries'}
    audit['dataset'] = prepare_dataset(source, release / 'huggingface/dataset')
    for modality in ('text', 'vision'):
        audit['model-' + modality] = prepare_model(
            source, release / ('huggingface/model-' + modality), modality)
    for relative in ('github', 'huggingface/dataset', 'huggingface/model-text', 'huggingface/model-vision'):
        audit[relative + '_inventory'] = write_manifest(release / relative)
    write_json(release / 'validation.json', audit)
    return audit


def verify(release, full_hash=False):
    from fetch_hub_assets import verify_manifest
    audit = {}
    for relative in ('github', 'huggingface/dataset', 'huggingface/model-text', 'huggingface/model-vision'):
        folder = release / relative
        if not (folder / 'release_manifest.json').is_file():
            raise FileNotFoundError(folder / 'release_manifest.json')
        if full_hash:
            manifest = json.loads((folder / 'release_manifest.json').read_text())
            for name, info in manifest['files'].items():
                info['sha256'] = sha256(checked_path(folder, name))
            write_json(folder / 'release_manifest.json', manifest)
        audit[relative] = verify_manifest(folder)
    audit['datasets'] = validate_dataset(release / 'huggingface/dataset')
    audit['code_findings'] = scan_code(release / 'github')
    if audit['code_findings']:
        raise ValueError(audit['code_findings'])
    for modality in ('text', 'vision'):
        audit[modality] = validate_shards(release / ('huggingface/model-' + modality))
    audit['asset_credential_findings'] = []
    for directory in ('dataset', 'model-text', 'model-vision'):
        folder = release / 'huggingface' / directory
        for path in payload_files(folder):
            if path.suffix.lower() in BINARY:
                continue
            content = path.read_text()
            if any(re.search(pattern, content) for pattern in SECRET_PATTERNS):
                audit['asset_credential_findings'].append(directory + '/' + path.relative_to(folder).as_posix())
    if audit['asset_credential_findings']:
        raise ValueError('Potential credentials found in asset metadata; inspect privately')
    write_json(release / 'verification.json', audit)
    return audit


def upload(release, namespace, workers):
    from huggingface_hub import HfApi
    verify(release)
    api = HfApi()
    mapping = [
        ('dataset', 'xmodal-transfer-data', 'dataset'),
        ('model-text', 'xmodal-rules-text-sft-ep10', 'model'),
        ('model-vision', 'xmodal-rules-vision-sft-ep10', 'model'),
    ]
    receipt = {}
    for directory, suffix, repo_type in mapping:
        repo_id = namespace + '/' + suffix
        api.create_repo(repo_id, repo_type=repo_type, private=True, exist_ok=True)
        if not api.repo_info(repo_id, repo_type=repo_type).private:
            raise ValueError(f'{repo_id} is already public; choose a private destination')
        api.upload_large_folder(repo_id, repo_type=repo_type,
                                folder_path=release / 'huggingface' / directory,
                                private=True, num_workers=workers,
                                ignore_patterns=['.cache/**'])
        receipt[directory] = {'repo_id': repo_id, 'repo_type': repo_type,
                              'revision': api.repo_info(repo_id, repo_type=repo_type).sha}
        write_json(release / 'upload_receipt.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'verify', 'upload'])
    parser.add_argument('--source', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=ROOT / 'hub_release')
    parser.add_argument('--full-hash', action='store_true')
    parser.add_argument('--namespace', help='HF user or organization; upload command only')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.source.resolve(), args.output.resolve())
    elif args.command == 'verify':
        result = verify(args.output.resolve(), args.full_hash)
    else:
        if not args.namespace or '/' in args.namespace or 'YOUR_' in args.namespace:
            parser.error('upload requires --namespace with the actual HF user or organization')
        result = upload(args.output.resolve(), args.namespace, args.workers)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
