#!/usr/bin/env python3
"""Resume a frozen official ModelScope manifest; validate ranges, sizes, and SHA256."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.parse
import urllib.request


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def run(model, output):
    with (output / 'resume_download.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        listing = json.loads((output / 'download_manifest.json').read_text())
        files = [r for r in listing['Data']['Files'] if r['Type'] == 'blob']
        def valid(path, item):
            return path.is_file() and path.stat().st_size == item['Size'] and sha(path) == item['Sha256']

        def fetch(item):
            path = output / item['Path']
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                assert valid(path, item), f'Existing completed file failed verification: {path}'
                print('verified existing', item['Path'], flush=True)
                return
            partial = path.with_name(path.name + '.partial')
            url = (f"https://modelscope.cn/models/{model}/resolve/{item['Revision']}/"
                   + urllib.parse.quote(item['Path']))
            for attempt in range(3):
                try:
                    offset = partial.stat().st_size if partial.exists() else 0
                    assert offset <= item['Size'], f'Partial exceeds expected file size: {partial}'
                    if offset < item['Size']:
                        headers = {'Range': f'bytes={offset}-'} if offset else {}
                        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
                            mode = 'wb'
                            if offset:
                                expected = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                                assert response.status == 206 and expected, 'Server did not honor resume range; partial preserved'
                                assert int(expected[1]) == offset and int(expected[3]) == item['Size']
                                mode = 'ab'
                            with partial.open(mode) as stream:
                                while block := response.read(8 * 1024 * 1024):
                                    stream.write(block)
                    assert valid(partial, item), f'Completed partial failed size/SHA256: {partial}'
                    partial.replace(path)
                    print('downloaded and verified', item['Path'], item['Size'], flush=True)
                    return
                except Exception as error:
                    print('attempt failed', item['Path'], attempt + 1, repr(error), flush=True)
                    if attempt == 2:
                        raise
                    time.sleep(2)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(fetch, files))
        result = dict(model=model, files_verified=len(files), sha256_verified=True,
                      size_bytes=sum(r['Size'] for r in files), resumed=True)
        temp = output / 'download_complete.json.tmp'
        temp.write_text(json.dumps(result, indent=2) + '\n')
        temp.replace(output / 'download_complete.json')
        print('complete', json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.model, args.output)
