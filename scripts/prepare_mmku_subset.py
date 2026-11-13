#!/usr/bin/env python3
"""Extract a reproducible, entity-paired MMKU Unknown subset via ZIP byte ranges."""
import argparse
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import struct
import time
import urllib.request
import zlib


SOURCE = 'https://hf-mirror.com/datasets/baochenfu/MMKU-Bench/resolve/main/MMKU-Bench.zip?download=true'
SALT = 'mmku500-unknown-v1:42:'


def entity(row):
    return re.sub(r'_\d+$', '', PurePosixPath(row['image']).stem)


def pair_unknown(train, test):
    maps = []
    for rows in (train, test):
        result = {}
        for row in rows:
            key = entity(row)
            if key in result:
                raise ValueError(f'Duplicate entity: {key}')
            result[key] = row
        maps.append(result)
    if maps[0].keys() != maps[1].keys():
        raise ValueError('Training and test entity sets differ')
    return [(key, maps[0][key], maps[1][key]) for key in sorted(
        maps[0], key=lambda key: hashlib.sha256((SALT + key).encode()).hexdigest())]


def zip_entries(data):
    result = {}
    off = 0
    while off < len(data):
        z = struct.unpack_from('<4s6H3I5H2I', data, off)
        if z[0] != b'PK\x01\x02':
            raise ValueError(f'Invalid central directory at {off}')
        nl, xl, cl = z[10:13]
        name = data[off + 46:off + 46 + nl].decode('utf-8' if z[3] & 2048 else 'cp437')
        extra = data[off + 46 + nl:off + 46 + nl + xl]
        usize, csize, offset = z[9], z[8], z[16]
        pos = 0
        while pos + 4 <= len(extra):
            tag, length = struct.unpack_from('<HH', extra, pos)
            if tag == 1:
                values = iter(struct.unpack('<' + 'Q' * (length // 8), extra[pos + 4:pos + 4 + length]))
                if usize == 0xffffffff:
                    usize = next(values)
                if csize == 0xffffffff:
                    csize = next(values)
                if offset == 0xffffffff:
                    offset = next(values)
            pos += 4 + length
        result[name] = dict(name=name, compression=z[4], crc=z[7],
                            compressed_size=csize, size=usize, offset=offset)
        off += 46 + nl + xl + cl
    return result


def unpack_entry(entry, data):
    z = struct.unpack_from('<4s5H3I2H', data, 0)
    if z[0] != b'PK\x03\x04':
        raise ValueError('Invalid ZIP local header')
    nl, xl = z[-2:]
    name = data[30:30 + nl].decode('utf-8' if z[2] & 2048 else 'cp437')
    if name != entry['name']:
        raise ValueError(f'ZIP filename mismatch: {name}')
    payload = data[30 + nl + xl:30 + nl + xl + entry['compressed_size']]
    raw = zlib.decompress(payload, -15) if entry['compression'] == 8 else payload
    if len(raw) != entry['size'] or zlib.crc32(raw) != entry['crc']:
        raise ValueError(f'ZIP size/CRC mismatch: {name}')
    return raw


def range_get(url, start, end):
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={
                'Range': f'bytes={start}-{end}', 'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=45) as response:
                if response.status != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-'):
                    raise ValueError('Server did not honor the requested byte range')
                return response.read(end - start + 1), response.geturl()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1 + attempt)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def rows(path, values):
    path.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in values))


def prepare(source_root, output, count=500, workers=12):
    source_root, output = Path(source_root), Path(output)
    all_pairs = pair_unknown(json.loads((source_root / 'raw/unknown_train_rewrite.json').read_text()),
                             [json.loads(x) for x in (source_root / 'raw/unknown_test_rewrite.jsonl').read_text().splitlines()])
    same_image_excluded = [key for key, tr, te in all_pairs if tr['image'] == te['image']]
    pairs = [(key, tr, te) for key, tr, te in all_pairs if tr['image'] != te['image']][:count]
    if len(pairs) != count:
        raise ValueError('Insufficient knowledge pairs')
    entries = zip_entries((source_root / 'zip_central_directory.bin').read_bytes())
    files = {}
    for _, train, test in pairs:
        for row in (train, test):
            relative = 'unknown/' + row['image']
            files[relative] = entries['data/fbc/MMKU-Bench/' + relative]
    output.mkdir(parents=True, exist_ok=True)
    # Resolve the signed public download URL once, then reuse it for each byte range.
    _, download_url = range_get(SOURCE, 0, 31)

    def download(item):
        relative, entry = item
        dest = output / relative
        if dest.exists():
            raw = dest.read_bytes()
            if len(raw) == entry['size'] and zlib.crc32(raw) == entry['crc']:
                return relative, hashlib.sha256(raw).hexdigest()
        data, _ = range_get(download_url, entry['offset'],
                            entry['offset'] + entry['compressed_size'] + len(entry['name'].encode()) + 512)
        raw = unpack_entry(entry, data)
        from PIL import Image
        with Image.open(io.BytesIO(raw)) as image:
            image.verify()
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_name(dest.name + '.tmp')
        temp.write_bytes(raw)
        temp.replace(dest)
        return relative, hashlib.sha256(raw).hexdigest()

    hashes = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(download, item) for item in files.items()]
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            relative, digest = future.result()
            hashes[relative] = digest
            if i % 50 == 0 or i == len(files):
                print(f'Images verified: {i}/{len(files)}', flush=True)

    training, testing, audit = [], [], []
    for key, tr, te in pairs:
        ti, ei = 'unknown/' + tr['image'], 'unknown/' + te['image']
        if hashes[ti] == hashes[ei]:
            raise ValueError(f'Train/test image bytes are identical: {key}')
        training.append({'id': tr['id'], 'messages': [
            {'role': 'user', 'content': tr['conversations'][0]['value']},
            {'role': 'assistant', 'content': tr['conversations'][1]['value']}], 'images': [ti]})
        testing.append({'id': str(te['question_id']), 'entity': key, 'messages': [
            {'role': 'user', 'content': '<image>\n' + te['text']}], 'images': [ei], 'answer': te['answer']})
        audit.append({'entity': key, 'train_id': tr['id'], 'test_id': te['question_id'],
                      'train': tr, 'test': te, 'train_image_sha256': hashes[ti], 'test_image_sha256': hashes[ei]})
    rows(output / 'train.jsonl', training)
    rows(output / 'test.jsonl', testing)
    rows(output / 'pairing_audit.jsonl', audit)
    intersection = set(hashes[r['images'][0]] for r in training) & set(hashes[r['images'][0]] for r in testing)
    manifest = {'dataset': 'baochenfu/MMKU-Bench', 'source': SOURCE,
                'hf_commit': '337e524f4e2d63bd7ba4d2d7661f79b1c1cc0826', 'subset': 'unknown',
                'knowledge_count': count, 'training_rows': len(training), 'heldout_rows': len(testing),
                'selection': 'exclude identical train/test image filenames, then first N entities sorted by SHA256(salt + entity); no model-score or answer filtering',
                'selection_salt': SALT, 'pairing': 'unique entity name parsed from image basename; NOT numeric ID',
                'excluded_identical_image_entities': same_image_excluded,
                'train_test_identical_pair_images': 0, 'global_train_test_image_hash_overlap': len(intersection),
                'image_count': len(files), 'image_bytes': sum((output / p).stat().st_size for p in files),
                'image_sha256': hashes, 'data_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [output / 'train.jsonl', output / 'test.jsonl', output / 'pairing_audit.jsonl']},
                'supervision': 'Original assistant answers unchanged; one image and one QA per training row',
                'evaluation': 'Official rewritten question + independent image + official short reference answer',
                'test_gold_exposure': 'Gold is stored only for offline scoring; inference payload contains messages and images only'}
    save(output / 'manifest.json', manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ('image_sha256',)}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--count', type=int, default=500)
    ap.add_argument('--workers', type=int, default=12)
    args = ap.parse_args()
    prepare(args.source_root, args.output, args.count, args.workers)
