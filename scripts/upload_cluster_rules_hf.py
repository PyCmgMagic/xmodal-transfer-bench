#!/usr/bin/env python3
"""Add the verified portable rule bundle to the existing private HF dataset."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from huggingface_hub import CommitOperationAdd, CommitOperationCopy, HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile
import yaml

from hf_upload_transport import configure_upload_backend


PROJECT = Path(__file__).resolve().parents[1]
BUNDLE_NAME = 'rules1000_10x_20261027'
PREFIX = f'cluster/{BUNDLE_NAME}'
SOURCE = 'data/rules_repaired_single_image_v7'
REPO = 'PyCmg/xmodal-transfer-data'
MARKER_START = '<!-- rules1000-10x:start -->'
MARKER_END = '<!-- rules1000-10x:end -->'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save(path, record):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def inventory(root):
    expected = {}
    for line in (root / 'SHA256SUMS').read_text().splitlines():
        sha256, relative = line.split('  ', 1)
        require(re.fullmatch(r'[0-9a-f]{64}', sha256), 'Malformed checksum')
        path = root / relative
        require(path.resolve().is_relative_to(root), 'Path escapes bundle')
        require(relative not in expected, 'Duplicate checksum path')
        expected[relative] = sha256
    expected['SHA256SUMS'] = hashlib.sha256((root / 'SHA256SUMS').read_bytes()).hexdigest()
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts}
    require(actual == set(expected), 'Unexpected or missing bundle files')
    image_hashes = {r['path']: r['repaired_sha256'] for r in
                    map(json.loads, (root / 'audit/image_hashes.jsonl').read_text().splitlines())}
    require(len(image_hashes) == 3000, 'Expected 3000 images')
    require({p for p in expected if p.startswith('images/')} == set(image_hashes),
            'Image inventory differs')
    records = []
    for relative, sha256 in sorted(expected.items()):
        path = root / relative
        record = {'path': relative, 'sha256': sha256, 'bytes': path.stat().st_size,
                  'copy_existing_image': relative in image_hashes}
        if record['copy_existing_image']:
            require(sha256 == image_hashes[relative], 'Image checksum sources disagree')
        else:
            data = path.read_bytes()
            require(hashlib.sha256(data).hexdigest() == sha256,
                    f'Local checksum changed: {relative}')
            require(not re.search(rb'hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|'
                                  rb'-----BEGIN [A-Z ]*PRIVATE KEY-----', data),
                    f'Credential-like content in {relative}')
            record['git_blob_sha1'] = git_blob(data)
        records.append(record)
    require(len(records) == 3035, 'Expected 3035 package files')
    return records


def git_blob(data):
    return hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()


def update_card(original):
    require(original.startswith('---\n'), 'Dataset card needs existing YAML metadata')
    front, body = original[4:].split('\n---\n', 1)
    metadata = yaml.safe_load(front)
    configs = metadata.setdefault('configs', [])
    names = {c['config_name'] for c in configs}
    for arm in ('original', 'repeat10', 'multiform10'):
        for modality in ('text', 'vision'):
            name = f'rules1000_{arm}_{modality}'
            config = {'config_name': name, 'data_files': [
                {'split': 'train', 'path': f'{PREFIX}/train/{arm}/{modality}.jsonl'}]}
            if name in names:
                require(next(c for c in configs if c['config_name'] == name) == config,
                        f'Conflicting existing config: {name}')
            else:
                configs.append(config)
    section = f'''{MARKER_START}
## 1000 rules: original and 10-prompt cluster comparison

The [portable cluster bundle]({PREFIX}/README_ZH.md) contains the exact original
1000-rule SFT inputs and the new 10-prompt-per-rule augmentation, for both TEXT and VISION.

| Arm | TEXT rows | VISION rows | Records per rule |
|---|---:|---:|---|
| `original` | 1,000 | 1,000 | One original question and answer |
| `repeat10` | 10,000 | 10,000 | Original question and answer repeated 10 times |
| `multiform10` | 10,000 | 10,000 | Ten question forms with the original target unchanged |

This is full-definition question paraphrasing, not the old atomic/boundary curriculum.
All arms default to 10 epochs; expanded arms therefore present about ten times as many
training examples. No equal-compute adjustment is applied. Each run starts from
Qwen3-VL-4B-Instruct using ms-swift full SFT.

The bundle includes 3,000 unchanged PNG images, four evaluation inputs (2,000 rows
each), separate gold labels, provenance, SHA256SUMS, validation and relocation tools,
and Slurm templates. Training data exclude evaluation questions and gold labels.
Visual learning retains its original text caption. Model weights are separate.

After authenticating with an account authorized for this private repository, download
only this bundle (the new JSON configs also expose its six training files):

```python
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="{REPO}", repo_type="dataset",
    allow_patterns=["{PREFIX}/**"], local_dir="xmodal-data",
)
```

The package root is `xmodal-data/{PREFIX}`. Run `python tools/validate.py` from
that directory, then `python tools/prepare.py --work /absolute/path/to/rules1000_work`
to resolve image paths for the cluster. Generic JSON loading alone does not resolve
the portable image paths. See the bundle README for training commands and limitations.
{MARKER_END}'''
    if MARKER_START in body:
        require(body.count(MARKER_START) == body.count(MARKER_END) == 1,
                'Duplicated bundle section markers')
        body = body[:body.index(MARKER_START)] + section + body[body.index(MARKER_END) + len(MARKER_END):]
    else:
        body = body.rstrip() + '\n\n' + section + '\n'
    return '---\n' + yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True).rstrip() + '\n---\n' + body


def tree(api, prefix, revision):
    return {f.path: f for f in api.list_repo_tree(
        REPO, prefix, recursive=True, revision=revision, repo_type='dataset')
        if isinstance(f, RepoFile)}


def verify(api, records, revision, card_data):
    files = tree(api, PREFIX, revision)
    require(set(files) == {f'{PREFIX}/{r["path"]}' for r in records},
            'Remote bundle inventory mismatch')
    lfs_count = 0
    for record in records:
        f = files[f'{PREFIX}/{record["path"]}']
        require(f.size == record['bytes'], f'Remote size mismatch: {record["path"]}')
        if f.lfs:
            require(f.lfs.sha256 == record['sha256'], f'Remote SHA256 mismatch: {record["path"]}')
            lfs_count += 1
        else:
            require(not record['copy_existing_image'], 'Expected LFS image after copy')
            require(f.blob_id == record['git_blob_sha1'], f'Remote git blob mismatch: {record["path"]}')
    card = api.get_paths_info(REPO, ['README.md'], revision=revision, repo_type='dataset')[0]
    require(card.size == len(card_data), 'Remote card size differs')
    if card.lfs:
        require(card.lfs.sha256 == hashlib.sha256(card_data).hexdigest(), 'Remote card SHA256 differs')
    else:
        require(card.blob_id == git_blob(card_data), 'Remote card blob differs')
    return {'files_verified': len(records), 'lfs_sha256_verified': lfs_count,
            'git_blobs_verified': len(records) - lfs_count, 'root_dataset_card_verified': True}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--apply', action='store_true', help='Commit after remote source verification')
    ap.add_argument('--offline-plan', action='store_true', help='Prepare and validate inventory without networking')
    ap.add_argument('--verify-revision', help='Verify a previous successful commit without writing to HF')
    args = ap.parse_args()
    root = (PROJECT / 'cluster_ready' / BUNDLE_NAME).resolve()
    output = PROJECT / 'cluster_ready' / f'hf_release_{BUNDLE_NAME}'
    output.mkdir(exist_ok=True)
    records = inventory(root)
    plan = {'repo_id': REPO, 'repo_type': 'dataset', 'private': True,
            'bundle_prefix': PREFIX, 'total_files': len(records),
            'server_side_image_copies': 3000,
            'image_bytes_reused': sum(r['bytes'] for r in records if r['copy_existing_image']),
            'new_file_bytes': sum(r['bytes'] for r in records if not r['copy_existing_image']),
            'records': records}
    save(output / 'upload_plan.json', plan)
    print(json.dumps({k: v for k, v in plan.items() if k != 'records'}), flush=True)
    if args.offline_plan:
        return

    configure_upload_backend(interval=2.0, status_path=output / 'rate_limit.json')
    api = HfApi()
    require(api.whoami().get('name') == 'PyCmg', 'Unexpected HF account')
    info = api.repo_info(REPO, repo_type='dataset')
    require(info.private, 'Existing repository must remain private')
    if args.verify_revision:
        card_data = (output / 'README.after.md').read_bytes()
        result = verify(api, records, args.verify_revision, card_data)
        save(output / 'verification.json', {'revision': args.verify_revision, 'status': 'pass', **result})
        print(json.dumps(result), flush=True)
        return

    source_revision = info.sha
    print(f'Checking source image metadata at {source_revision}', flush=True)
    source_files = tree(api, f'{SOURCE}/images', source_revision)
    for record in records:
        if not record['copy_existing_image']:
            continue
        source_path = f'{SOURCE}/{record["path"]}'
        require(source_path in source_files, f'Missing remote source image: {source_path}')
        f = source_files[source_path]
        require(f.lfs is not None, f'Source image needs LFS metadata: {source_path}')
        require(f.size == record['bytes'] and f.lfs.sha256 == record['sha256'],
                f'Remote source image differs: {source_path}')
    print('All 3000 source image SHA256 hashes and sizes match.', flush=True)
    source_card = Path(hf_hub_download(REPO, 'README.md', revision=source_revision, repo_type='dataset')).read_bytes()
    card_data = update_card(source_card.decode()).encode()
    (output / 'README.before.md').write_bytes(source_card)
    (output / 'README.after.md').write_bytes(card_data)
    save(output / 'source_check.json', {'status': 'pass', 'revision': source_revision,
                                      'images_verified': 3000})
    if not args.apply:
        print('Remote plan ready; use --apply to commit.', flush=True)
        return
    operations = []
    for record in records:
        relative = record['path']
        destination = f'{PREFIX}/{relative}'
        if record['copy_existing_image']:
            operations.append(CommitOperationCopy(f'{SOURCE}/{relative}', destination,
                                                  src_revision=source_revision))
        else:
            operations.append(CommitOperationAdd(destination, root / relative))
    operations.append(CommitOperationAdd('README.md', card_data))
    save(output / 'status.json', {'status': 'committing', 'source_revision': source_revision,
                                 'operations': len(operations)})
    print(f'Committing {len(operations)} additive operations; no image transfer needed.', flush=True)
    commit = api.create_commit(REPO, repo_type='dataset', operations=operations,
                              commit_message='Add 1000-rule original and 10-prompt cluster comparison',
                              commit_description='Verified portable TEXT/VISION training arms, paired evaluations, '
                                                 '3000 unchanged images and cluster tools. Reuses existing LFS images.',
                              parent_commit=source_revision, num_threads=2)
    receipt = {'repo_id': REPO, 'repo_type': 'dataset', 'private': True,
               'revision': commit.oid, 'source_revision': source_revision,
               'commit_url': commit.commit_url,
               'bundle_url': f'https://huggingface.co/datasets/{REPO}/tree/{commit.oid}/{PREFIX}',
               'bundle_prefix': PREFIX, 'files': len(records), 'images': 3000,
               'timestamp_utc': datetime.now(timezone.utc).isoformat(),
               'verification': 'pending'}
    save(output / 'upload_receipt.json', receipt)
    print(json.dumps({'committed': commit.oid, 'bundle_url': receipt['bundle_url']}), flush=True)
    result = verify(api, records, commit.oid, card_data)
    receipt['verification'] = 'pass'
    receipt['checks'] = result
    save(output / 'upload_receipt.json', receipt)
    save(output / 'verification.json', {'revision': commit.oid, 'status': 'pass', **result})
    save(output / 'status.json', {'status': 'complete', 'revision': commit.oid})
    print(json.dumps({'status': 'complete', **result}), flush=True)


if __name__ == '__main__':
    main()
