#!/usr/bin/env python3
"""Download and verify a ModelScope model without using the system disk cache."""
import argparse
import concurrent.futures
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen3-VL-4B-Instruct')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    api = f'https://modelscope.cn/api/v1/models/{args.model}/repo/files?Revision=master&Recursive=true'
    with urllib.request.urlopen(api, timeout=60) as stream:
        listing = json.load(stream)
    files = [f for f in listing['Data']['Files'] if f['Type'] == 'blob']
    (args.output / 'download_manifest.json').write_text(json.dumps(listing, indent=2))

    def download(item):
        path = args.output / item['Path']
        path.parent.mkdir(parents=True, exist_ok=True)
        expected = item.get('Sha256')
        if path.exists() and path.stat().st_size == item['Size'] and (not expected or sha256(path) == expected):
            print('verified existing', item['Path'], flush=True)
            return
        url = f"https://modelscope.cn/models/{args.model}/resolve/{item['Revision']}/{urllib.parse.quote(item['Path'])}"
        temp = path.with_name(path.name + '.partial')
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=120) as response, temp.open('wb') as stream:
                    while block := response.read(8 * 1024 * 1024):
                        stream.write(block)
                assert temp.stat().st_size == item['Size'], f'Wrong size: {path}'
                assert not expected or sha256(temp) == expected, f'Wrong hash: {path}'
                temp.replace(path)
                print('downloaded and verified', item['Path'], item['Size'], flush=True)
                return
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, files))
    (args.output / 'download_complete.json').write_text(json.dumps({
        'model': args.model, 'files_verified': len(files), 'sha256_verified': True,
        'size_bytes': sum(f['Size'] for f in files),
    }, indent=2))


if __name__ == '__main__':
    main()
