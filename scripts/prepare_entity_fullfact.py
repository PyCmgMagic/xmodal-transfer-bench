#!/usr/bin/env python3
"""Build paired full-fact SFT from learning-only sources, without generation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from entityqa_common import digest, now, rows, save, sha, write_rows


def examples(record, qa, modality, data):
    """Keep the entire modality-specific material as the assistant target."""
    visual = modality == 'vision'
    subject = record['reference'] if visual else record['name']
    answer = record['visual_learning'] if visual else record['learning']
    questions = [
        ('recall', f'State the complete learned factual information about {subject}. '
         'Include the dates, quantities, and other factual details in the learned information.'),
        ('attribute', 'Answer the question and state the complete supporting factual information, '
         'including its dates, quantities, and other details.\n'
         + qa['question_vision' if visual else 'question_text']),
    ]
    result = []
    for kind, question in questions:
        if visual:
            question = '<image>\n' + question
        item = dict(id=record['id'] + '-' + kind, knowledge_id=record['id'],
                    qid=record['qid'], variant=kind,
                    messages=[{'role': 'user', 'content': question},
                              {'role': 'assistant', 'content': answer}])
        if visual:
            item['images'] = [str((data / p).resolve()) for p in record['images']]
        result.append(item)
    return result


def prepare(args):
    source, work, data = (getattr(args, k).resolve() for k in ['source', 'work', 'data'])
    manifest_path = source / 'data/manifest.json'
    old_manifest = json.loads(manifest_path.read_text())
    for name in ['learning_inputs.jsonl', 'qa_pairs.jsonl']:
        assert sha(source / 'data' / name) == old_manifest['files'][name], name
    records = rows(source / 'data/learning_inputs.jsonl')
    qa = {r['id']: r for r in rows(source / 'data/qa_pairs.jsonl')}
    # Only the training portions of the package are read to verify the snapshot.
    # No evaluation question, option, answer label or test image is loaded here.
    original_text = {r['id']: r for r in rows(data / 'source/learning.jsonl')}
    original_visual = {r['qid']: r for r in rows(data / 'shared/train_multimodal.jsonl')}
    assert len(records) == len({r['id'] for r in records}) == 1814
    assert len({r['qid'] for r in records}) == 1814
    texts, visions, audit = [], [], []
    for r in records:
        assert r['learning'] == original_text[r['id']]['text']
        v = original_visual[r['qid']]
        assert r['visual_learning'] == v['messages'][0]['content'].replace('<image>', '').strip()
        assert r['images'] == v['images']
        assert qa[r['id']]['source_sha256'] == digest(r)
        assert qa[r['id']]['answer'] in r['learning']
        assert qa[r['id']]['answer'] in r['visual_learning']
        t = examples(r, qa[r['id']], 'text', data)
        v = examples(r, qa[r['id']], 'vision', data)
        for a, b in zip(t, v):
            assert a['id'] == b['id'] and a['variant'] == b['variant']
            assert a['messages'][-1]['content'] == r['learning']
            assert b['messages'][-1]['content'] == r['visual_learning']
            assert a['messages'][-1]['content'] not in a['messages'][0]['content']
            assert b['messages'][-1]['content'] not in b['messages'][0]['content']
            assert '<image>' not in a['messages'][0]['content'] and 'images' not in a
            assert b['messages'][0]['content'].count('<image>') == len(b['images']) == 1
            assert all(Path(p).is_file() for p in b['images'])
            # All source characters, hence all source dates/numbers, are supervised.
            for key, item in [('learning', a), ('visual_learning', b)]:
                assert re.findall(r'\d+', r[key]) == re.findall(r'\d+', item['messages'][1]['content'])
        texts.extend(t)
        visions.extend(v)
        audit.append(dict(id=r['id'], qid=r['qid'], variants=['recall', 'attribute'],
                          text_full_source_target=True, vision_full_source_target=True,
                          source_numbers_preserved=True, learning_images=r['images'],
                          source_sha256=digest(r)))
    assert len(texts) == len(visions) == 3628
    work_data = work / 'data'
    write_rows(work_data / 'train_text.jsonl', texts)
    write_rows(work_data / 'train_vision.jsonl', visions)
    write_rows(work_data / 'coverage_audit.jsonl', audit)
    write_rows(work_data / 'learning_inputs.jsonl', records)
    save(work_data / 'manifest.json', dict(
        status='complete', version='fullfact-v2', created_at=now(), knowledge_units=len(records),
        rows_per_modality=len(texts), variants_per_unit=2,
        method='Complete source paragraph as assistant target for neutral recall and original attribute QA',
        generator_model=None, test_data_used=False, training_image_role='learning only',
        answer_identity_policy='TEXT keeps original named material; VISION keeps original anonymized material; no identity added to visual targets',
        supervision='Assistant-only SFT; every character of the corresponding learning material appears in each target',
        limitations=['Source semantic and temporal validity is inherited from the repaired candidate package.',
                     'Two QA variants and more supervised tokens change training exposure relative to v1.',
                     'Full-source coverage does not establish memory or transfer; these require evaluation.'],
        previous_training_source=str(source), previous_manifest_sha256=sha(manifest_path),
        source_sha256={str(p): sha(p) for p in [source/'data/learning_inputs.jsonl',
            source/'data/qa_pairs.jsonl', data/'source/learning.jsonl', data/'shared/train_multimodal.jsonl']},
        dataset_inventory_sha256=sha(data/'package_inventory.json'),
        variant_counts=dict(Counter(r['variant'] for r in texts)),
        files={p.name: sha(p) for p in sorted(work_data.glob('*.jsonl'))}))
    print(json.dumps(dict(knowledge_units=len(records), rows_per_modality=len(texts),
                         full_source_coverage=1.0, work=str(work)), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['source', 'work', 'data']:
        parser.add_argument('--' + name, type=Path, required=True)
    prepare(parser.parse_args())
