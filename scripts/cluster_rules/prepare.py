#!/usr/bin/env python3
"""Materialize package-relative image references at the extraction location."""
import argparse
from pathlib import Path
from common import dataset_paths, digest, dump, relocate, rows, write_rows


def prepare(root, work):
    root, work = Path(root).resolve(), Path(work).resolve()
    if root == work or work.is_relative_to(root):
        raise ValueError('Use a work directory outside the immutable data package')
    if work.exists():
        raise FileExistsError(f'Work directory already exists; use a fresh path: {work}')
    work.mkdir(parents=True)
    files = {}
    for relative in dataset_paths():
        output = work / 'data' / relative
        write_rows(output, (relocate(row, root) for row in rows(root / relative)))
        files[relative] = digest(output)
    dump(work / 'prepared.json', {'package_root': str(root), 'work_root': str(work),
                                 'manifest_sha256': digest(root / 'manifest.json'),
                                 'portable_sha256sums_sha256': digest(root / 'SHA256SUMS'),
                                 'files': files, 'gold_path': str(root / 'eval/gold.jsonl')})
    print(f'Materialized {len(files)} datasets at {work / "data"}; gold remains separate.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--work', type=Path, required=True)
    args = ap.parse_args()
    prepare(args.root, args.work)
