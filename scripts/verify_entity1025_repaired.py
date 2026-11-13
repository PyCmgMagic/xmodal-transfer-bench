"""Read-only validation of a repaired, portable entity package."""
import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from PIL import Image


def read(path):
    # U+0085 is present in source text; only physical JSONL lines are delimiters.
    with Path(path).open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def norm(value):
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())


def eval_body(question):
    return question['question'] + '\n' + '\n'.join(
        f"{option['id']}. {option['text']}" for option in question['options'])


def validate(root, check_inventory=True):
    root = Path(root).resolve()
    manifest = json.loads((root / 'build_manifest.json').read_text())
    n = manifest['knowledge_units']
    entities = read(root / 'source/entities.jsonl')
    qids = {x['qid'] for x in entities}
    assert len(qids) == len(entities) == n
    unit_for = {x['qid']: x['knowledge_sample_id'] for x in entities}
    assert len(set(unit_for.values())) == n
    labels = read(root / 'audit/eval_labels.jsonl')
    assert len(labels) == len({x['id'] for x in labels}) == 2 * n
    lab = {x['id']: x for x in labels}
    assert {x['qid'] for x in labels} == qids
    assert set(Counter(x['qid'] for x in labels).values()) == {2}
    source = {}
    for name in ['learning', 'answers', 'evidence', 'knowledge_samples']:
        records = read(root / f'source/{name}.jsonl')
        assert len(records) == len({x['qid'] for x in records}) == n, name
        assert {x['qid'] for x in records} == qids, name
        assert all(x['id'] == unit_for[x['qid']] for x in records), name
        source[name] = {x['qid']: x for x in records}
    original_answers = {x['qid']: x for x in read(root / 'provenance/input_package/source/answers.jsonl')}
    assert all(source['answers'][qid] == original_answers[qid] for qid in qids)
    questions = read(root / 'source/questions.jsonl')
    assert len(questions) == 2 * n
    qs = {(x['qid'], x['type']): x for x in questions}
    assert len(qs) == 2 * n
    for qid in qids:
        for probe in ['direct', 'statement']:
            q = qs[qid, probe]
            assert q['id'] == unit_for[qid]
            assert [x['id'] for x in q['options']] == list('ABCD')
            assert len({norm(x['text']) for x in q['options']}) == 4, (qid, probe)
            gold = source['answers'][qid]['answers'][probe]
            assert gold in 'ABCD' and len(gold) == 1
            lid = f'{unit_for[qid]}-{probe}'
            assert lab[lid] == dict(id=lid, qid=qid, answer=gold, probe_id=probe)
        ks = source['knowledge_samples'][qid]
        sample = ks['content']['sample']
        assert sample['learning'] == source['learning'][qid]['text']
        assert sample['new_knowledge'] == source['evidence'][qid]['new_knowledge']
        assert ks['automatic_passed'] is False and ks['human_approved'] is False
        assert len(sample['questions']) == 2
        for q in sample['questions']:
            canonical = qs[qid, q['type']]
            assert q['question'] == canonical['question']
            assert q['options'] == [x['text'] for x in canonical['options']]
            assert 'ABCD'[q['answer_index']] == source['answers'][qid]['answers'][q['type']]
        entity = next(x for x in entities if x['qid'] == qid)
        assert len(entity['samples']) == 1
        assert entity['samples'][0]['learning_text'] == sample['learning']
        assert entity['samples'][0]['new_knowledge'] == sample['new_knowledge']

    roles = read(root / 'audit/image_roles.jsonl')
    catalog = read(root / 'audit/image_catalog.jsonl')
    assert len(roles) == n and len(catalog) == 2 * n
    pairs = {x['qid']: x for x in roles}
    assert set(pairs) == qids
    cats = {x['file']: x for x in catalog}
    assert len(cats) == 2 * n
    assert {str(x.relative_to(root)) for x in (root / 'images').rglob('*') if x.is_file()} == set(cats)
    hashes = {'learning': set(), 'evaluation': set()}
    original_hashes = {'learning': set(), 'evaluation': set()}
    numbered = 0
    for qid, pair in pairs.items():
        assert pair['unit_id'] == unit_for[qid]
        assert not pair.get('reserve')
        assert pair['learning']['source_url'] != pair['evaluation']['source_url']
        for role in ['learning', 'evaluation']:
            asset = pair[role]
            path = (root / asset['file']).resolve()
            assert path.is_relative_to(root / 'images') and path.is_file()
            cat = cats[asset['file']]
            assert cat['qid'] == qid and cat['role'] == role
            assert sha(path) == asset['sha256'] == cat['sha256']
            with Image.open(path) as im:
                im.load()
                assert list(im.size) == cat['size'] == [asset['width'], asset['height']]
            hashes[role].add(asset['sha256'])
            original_hashes[role].add(asset.get('original_sha256', asset['sha256']))
            if 'annotation' in asset:
                numbered += 1
                assert asset['annotation']['phrase'] == 'the person marked 1 with a red box'
    assert not hashes['learning'].intersection(hashes['evaluation'])
    assert not original_hashes['learning'].intersection(original_hashes['evaluation'])
    assert numbered == manifest['numbered_image_roles']

    views = {name: read(root / f'shared/{name}.jsonl') for name in
             ['train_text', 'train_multimodal', 'eval_text', 'eval_multimodal']}
    for name, records in views.items():
        is_eval = name.startswith('eval_')
        visual = name.endswith('multimodal')
        assert len(records) == (2 * n if is_eval else n)
        assert len({x['id'] for x in records}) == len(records)
        assert {x['qid'] for x in records} == qids
        for row in records:
            qid = row['qid']
            content = row['messages'][-1]['content']
            assert bool(row.get('images')) == visual
            assert content.count('<image>') == int(visual)
            if visual:
                role = 'evaluation' if is_eval else 'learning'
                assert row['images'] == [pairs[qid][role]['file']]
                assert not re.search(r'\bthe\s+the\b', content, re.I), row['id']
                if 'annotation' in pairs[qid][role]:
                    assert 'Focus on the person marked 1 with a red box.' in content
            if is_eval:
                assert [m['role'] for m in row['messages']] == ['system', 'user']
                assert row['id'] in lab and lab[row['id']]['qid'] == qid
                assert all(re.search(rf'\n{letter}\. .+', content) for letter in 'ABCD')
                if not visual:
                    assert eval_body(qs[qid, lab[row['id']]['probe_id']]) in content
            else:
                assert [m['role'] for m in row['messages']] == ['assistant']
                assert row['id'] == unit_for[qid]
                if not visual:
                    assert content == source['learning'][qid]['text']
    for direction, (train, evaluate) in manifest['directions'].items():
        assert read(root / direction / 'train.jsonl') == views[train]
        assert read(root / direction / 'eval.jsonl') == views[evaluate]
    assert read(root / 'text_all/train.jsonl') == views['train_text']
    assert read(root / 'text_all/eval.jsonl') == views['eval_text']
    assert read(root / 'text_all/eval_labels.jsonl') == labels
    audits = read(root / 'audit/sample_audit.jsonl')
    assert len(audits) == len({x['sample_id'] for x in audits}) == 2 * n
    assert {x['sample_id'] for x in audits} == set(lab)
    for x in audits:
        assert x['answer'] == lab[x['sample_id']]['answer']
        assert x['learning_image_files'] == [pairs[x['qid']]['learning']['file']]
        assert x['eval_image_files'] == [pairs[x['qid']]['evaluation']['file']]

    decisions = read(root / 'audit/unit_dispositions.jsonl')
    assert len(decisions) == manifest['input_knowledge_units']
    assert len({x['qid'] for x in decisions}) == len(decisions)
    assert {x['qid'] for x in decisions if x['action'] != 'delete'} == qids
    assert not {x['qid'] for x in decisions if x['action'] == 'delete'}.intersection(qids)
    membership = read(root / 'audit/unit_quality_status.jsonl')
    assert len(membership) == n and {x['qid'] for x in membership} == qids
    assert not any(x['approved_for_post_cutoff_experiment'] for x in membership)
    assert manifest['images'] == 2 * n and manifest['questions_per_direction'] == 2 * n
    assert manifest['deleted_units'] == sum(x['action'] == 'delete' for x in decisions)
    assert manifest['repaired_units'] == sum(x['action'] == 'repair' for x in decisions)

    changes = read(root / 'audit/changes.jsonl')
    changed_questions = {x['qid'] for x in changes if x['category'] == 'historical_question_reconstruction'}
    adjudications = read(root / 'provenance/review/adjudications.jsonl')
    assert not {x['qid'] for x in adjudications if x['verdict'] == 'confirmed_content_issue'}.intersection(qids)
    protected = {x['qid'] for x in adjudications
                 if x['category'] == 'visual_proper_name_corruption'}
    visual_eval = {x['id']: x for x in views['eval_multimodal']}
    for qid in protected.intersection(qids).difference(changed_questions):
        question = qs[qid, 'direct']
        content = visual_eval[f'{unit_for[qid]}-direct']['messages'][-1]['content']
        for option in question['options']:
            assert f"\n{option['id']}. {option['text']}\n" in content, (qid, option)
    for x in changes:
        if x['category'] == 'image_text_crop':
            asset = pairs[x['qid']][x['role']]
            assert asset['file'] == x['after_file']
            assert asset['original_sha256'] == x['before_sha256']
            assert asset['sha256'] != asset['original_sha256']
    cropped_roles = {(x['qid'], x['role']) for x in changes if x['category'] == 'image_text_crop'}
    for x in adjudications:
        if x['category'] == 'printed_name_card' and x['qid'] in qids:
            assert (x['qid'], x['scope']) in cropped_roles
        if x['verdict'] == 'confirmed_data_error' and x['qid'] in qids:
            assert x['qid'] in {c['qid'] for c in changes}

    if check_inventory:
        inventory = json.loads((root / 'package_inventory.json').read_text())
        files = {p.relative_to(root).as_posix(): p for p in root.rglob('*')
                 if p.is_file() and p != root / 'package_inventory.json'}
        assert set(files) == set(inventory['sha256'])
        assert len(files) == inventory['file_count_excluding_inventory']
        for relative, path in files.items():
            assert sha(path) == inventory['sha256'][relative], relative
    return dict(verification='passed', paired_units=n, questions_per_direction=2 * n,
                images_verified_hash_and_decode=2 * n, four_directions_aligned=True,
                source_nested_samples_and_labels_aligned=True,
                train_eval_image_and_original_hashes_disjoint=True,
                deleted_units_absent_from_all_active_views=True,
                protected_direct_options_restored=True,
                numbered_image_roles=numbered, inventory_checked=check_inventory,
                label_counts={probe: dict(Counter(x['answer'] for x in labels if x['probe_id'] == probe))
                              for probe in ['direct', 'statement']},
                semantic_identity_or_temporal_certification=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(validate(args.root), ensure_ascii=False, indent=2))
