#!/usr/bin/env python3
"""Download selected HF assets at resolved immutable commits; no model execution."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_target(root, relative):
    root = root.resolve()
    target = (root / relative).resolve()
    if Path(relative).is_absolute() or not target.is_relative_to(root):
        raise ValueError(f'Asset path escapes project: {relative}')
    return target


def verify_manifest(folder):
    manifest = folder / 'release_manifest.json'
    if not manifest.is_file():
        return {'status': 'no_project_manifest'}
    records = json.loads(manifest.read_text())['files']
    hashed = 0
    for name, info in records.items():
        path = safe_target(folder, name)
        if not path.is_file() or path.stat().st_size != info['bytes']:
            raise ValueError(f'Missing or wrong-sized asset: {name}')
        if info.get('sha256'):
            if digest(path) != info['sha256']:
                raise ValueError(f'Checksum mismatch: {name}')
            hashed += 1
    return {'status': 'pass', 'files': len(records), 'sha256_checked': hashed,
            'unhashed_large_files': len(records) - hashed}


def fetch(config, selected, root, lock_path, workers):
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    previous = json.loads(lock_path.read_text()) if lock_path.exists() else {'assets': {}}
    for key in selected:
        asset = config[key]
        if 'YOUR_' in asset['repo_id']:
            raise ValueError(f'Configure the HF repository for {key} first')
        repo_id, repo_type = asset['repo_id'], asset['repo_type']
        old = previous['assets'].get(key)
        if old and (old['repo_id'], old['repo_type'], old['local_dir']) != (
                repo_id, repo_type, asset['local_dir']):
            raise ValueError(f'Asset identity changed for {key}; use a new lock file')
        revision = asset.get('revision') or (old['revision'] if old else 'main')
        info = api.repo_info(repo_id, repo_type=repo_type, revision=revision)
        target = safe_target(root, asset['local_dir'])
        entry = dict(repo_id=repo_id, repo_type=repo_type, revision=info.sha,
                     local_dir=asset['local_dir'], status='downloading')
        previous['assets'][key] = entry
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps(previous, ensure_ascii=False, indent=2) + '\n')
        print(f'Downloading {key}: {repo_id}@{info.sha}', flush=True)
        snapshot_download(repo_id, repo_type=repo_type, revision=info.sha,
                          local_dir=str(target), max_workers=workers)
        entry['validation'] = verify_manifest(target)
        entry['status'] = 'complete'
        lock_path.write_text(json.dumps(previous, ensure_ascii=False, indent=2) + '\n')
    return previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('hub_sources.json'))
    parser.add_argument('--assets', nargs='+', choices=['dataset', 'base', 'text', 'vision'],
                        default=['dataset', 'base'])
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--lock', type=Path, default=Path('hub_assets.lock.json'))
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    root = args.root.resolve()
    lock_path = args.lock if args.lock.is_absolute() else root / args.lock
    result = fetch(json.loads(args.config.read_text()), args.assets, root, lock_path, args.workers)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
