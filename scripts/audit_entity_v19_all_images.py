"""Prepare exhaustive image-content screening; never infer portrait identity.

All formal learning/evaluation roles are covered. Unreviewed source identity is
explicitly retained as unknown even if no visual content conflict is noticed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'runs/entity_v19_transfer_ep5_20261014'
FROZEN = WORK / 'data/frozen_sources'
DATA = ROOT / 'entity_v19_1478'
OUT = ROOT / 'diagnostics/entity_v19_full_image_audit_20261016'
PAGE_SIZE = 20


def rows(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def write_rows(path, records):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def tokens(value):
    s = ''.join(c for c in unicodedata.normalize('NFKD', unquote(value)).casefold()
                if not unicodedata.combining(c))
    return re.findall(r'[a-z0-9]+', s)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ('contact_sheets', 'annotations'):
        (OUT / sub).mkdir(exist_ok=True)
    audit = {r['unit_id']: r for r in rows(FROZEN / 'audit/sample_audit.jsonl')}
    source = {r['id']: r for r in rows(FROZEN / 'source/knowledge_samples.jsonl')}
    role_rows = rows(FROZEN / 'audit/image_roles.jsonl')
    inventory = {r['relative_path']: r for r in rows(WORK / 'data/image_inventory.jsonl')}
    manifest = json.loads((WORK / 'manifest.json').read_text())
    for entry in manifest['sources']:
        assert sha(Path(entry['copy'])) == entry['sha256'], entry['copy']
    entities = {r['qid']: r for r in rows(DATA / 'source/entities.jsonl')}
    expected_sha = json.loads((FROZEN / 'build_manifest.json').read_text())['source_hashes']['entities.jsonl']
    assert sha(DATA / 'source/entities.jsonl') == expected_sha
    records = []
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 14)
    for index, role in enumerate(sorted(role_rows, key=lambda r: r['qid']), 1):
        uid = role['unit_id']
        a = audit[uid]
        ent = entities[a['qid']]
        terms = [ent['name'], *ent.get('aliases', [])]
        wiki_name = unquote(urlsplit(ent.get('wikipedia_url', '')).path).removeprefix('/wiki/').replace('_', ' ')
        terms.append(re.sub(r'\s*\([^()]*\)$', '', wiki_name).strip())
        multiword_terms = [tokens(t) for t in terms if len(tokens(t)) >= 2]
        r = dict(number=index, page=math.ceil(index / PAGE_SIZE), unit_id=uid, qid=a['qid'],
                 entity_name=a['entity_name'], entity_kind=a['entity_kind'],
                 learning_text=source[uid]['content']['sample']['learning'],
                 fact=source[uid]['content']['sample']['new_knowledge']['claim'],
                 source_entity=ent, roles={})
        for role_name in ('learning', 'evaluation'):
            item = role[role_name]
            p = DATA / item['file']
            actual_sha = sha(p)
            assert actual_sha == inventory[item['file']]['sha256'] == item['sha256'], p
            with Image.open(p) as im:
                im.load()
                width, height = im.size
            url = item['source_url']
            url_tokens = tokens(url)
            supported_terms = [t for t in multiword_terms if all(v in url_tokens for v in t)]
            r['roles'][role_name] = dict(file=item['file'], absolute_path=str(p), source_url=url,
                 sha256=actual_sha, width=width, height=height, source_host=urlsplit(url).netloc,
                 name_tokens_in_source_url=bool(supported_terms),
                 matching_name_token_sets=supported_terms, source_page_verified=False,
                 named_person_identity_verified=False)
        records.append(r)
    assert len(records) == 1225
    write_rows(OUT / 'all_units.jsonl', records)
    for page in range(1, math.ceil(len(records) / PAGE_SIZE) + 1):
        values = records[(page - 1) * PAGE_SIZE:page * PAGE_SIZE]
        sheet = Image.new('RGB', (1800, 1180), '#ffffff')
        d = ImageDraw.Draw(sheet)
        for pos, r in enumerate(values):
            x, y = (pos % 4) * 450, (pos // 4) * 236
            header = f"{r['number']:04d} {r['entity_name']}"
            d.text((x + 3, y + 2), header[:54], fill='black', font=font)
            d.text((x + 3, y + 22), r['entity_kind'] + ' | 学习 L             测试 E', fill='#555555', font=font)
            for col, name in enumerate(('learning', 'evaluation')):
                with Image.open(r['roles'][name]['absolute_path']) as im:
                    thumb = ImageOps.contain(ImageOps.exif_transpose(im).convert('RGB'), (220, 190))
                    sheet.paste(thumb, (x + col * 225 + (220 - thumb.width) // 2,
                                        y + 44 + (190 - thumb.height) // 2))
            d.line((x + 448, y, x + 448, y + 236), fill='#cccccc')
            d.line((x, y + 235, x + 450, y + 235), fill='#cccccc')
        sheet.save(OUT / 'contact_sheets' / f'page_{page:03d}.jpg', quality=94)
    save(OUT / 'preparation.json', dict(units=len(records), role_images=2 * len(records),
         contact_pages=math.ceil(len(records) / PAGE_SIZE), original_sha_verified=True,
         source_entity_sha_verified=True, correctness_used=False,
         source_url_name_match_roles=sum(v['name_tokens_in_source_url'] for r in records for v in r['roles'].values()),
         scope='All 1225 units and 2450 images used in formal experiment; reserve candidates are not formal roles',
         original_data_modified=False, note='URL tokens are screening evidence, not certified image identity'))
    print(json.dumps(json.loads((OUT / 'preparation.json').read_text()), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['prepare'])
    args = p.parse_args()
    prepare()
