#!/usr/bin/env python3
"""Validate portable train/eval isolation, controls, sources and checksums on CPU."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from common import ARMS, MODALITIES, DIRECTIONS, content_digest, digest, dump, inside, rows


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(root, hashes=True):
    root = Path(root).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    index = {r['qid']: r for r in rows(root / 'audit/rule_index.jsonl')}
    require(len(index) == 1000, 'Expected 1000 unique rules')
    require(Counter(r['family'] for r in index.values()) == manifest['families'], 'Family counts differ')
    targets = {}
    originals = {}
    source_images = set()
    for modality in MODALITIES:
        source = {r['qid']: r for r in rows(root / f'audit/source_learning_{modality}.jsonl')}
        require(set(source) == set(index), 'Learning source IDs differ')
        targets[modality] = {}
        for qid, r in source.items():
            answer = r['messages'][0]['content']
            if modality == 'vision':
                require(answer.startswith('<image>'), 'Missing original image prefix')
                answer = answer[len('<image>'):].strip()
                require(r['images'] == [f'images/{qid}/learning_visual.png'], 'Wrong source image')
                source_images.update(r['images'])
            targets[modality][qid] = answer
            require(content_digest(answer) == index[qid][f'{modality}_target_sha256'], 'Target provenance mismatch')
        originals[modality] = {r['qid']: r for r in rows(root / f'train/original/{modality}.jsonl')}

    train_report = {}
    train_prompts = set()
    for arm in ARMS:
        for modality in MODALITIES:
            counts = Counter()
            unique_prompts = defaultdict(set)
            unique_ids = set()
            count = 0
            for r in rows(root / f'train/{arm}/{modality}.jsonl'):
                qid = r['qid']
                require(qid in index, 'Unknown training qid')
                require(r['id'] not in unique_ids, 'Duplicate training row ID')
                unique_ids.add(r['id'])
                require([m['role'] for m in r['messages']] == ['user', 'assistant'], 'Wrong training roles')
                prompt, answer = (m['content'] for m in r['messages'])
                require(answer == targets[modality][qid], f'Changed {modality} target: {qid}')
                require(index[qid]['name'] in prompt, f'Missing canonical name: {qid}')
                require(qid not in prompt, 'Internal ID used as rule name')
                require(not any(k in r for k in ('answer', 'label', 'gold')), 'Label in training record')
                require('<image>' not in answer, 'Image marker in assistant target')
                require(prompt.count('<image>') == (1 if modality == 'vision' else 0), 'Wrong image marker count')
                expected_images = [f'images/{qid}/learning_visual.png'] if modality == 'vision' else []
                require(r.get('images', []) == expected_images, 'Training image is not its learning image')
                if arm == 'repeat10':
                    require(r['messages'] == originals[modality][qid]['messages'], 'Repeat changed a conversation')
                require(all(isinstance(m['content'], str) and m['content'] for m in r['messages']), 'Empty message')
                counts[qid] += 1
                unique_prompts[qid].add(prompt)
                train_prompts.add(prompt)
                count += 1
            expected = 1 if arm == 'original' else 10
            require(set(counts) == set(index) and set(counts.values()) == {expected}, 'Unbalanced rules')
            expected_unique = 10 if arm == 'multiform10' else 1
            require({len(v) for v in unique_prompts.values()} == {expected_unique}, 'Prompt uniqueness mismatch')
            train_report[f'{arm}/{modality}'] = {'rows': count, 'rules': len(counts),
                                               'rows_per_rule': expected, 'unique_prompts_per_rule': expected_unique}

    gold_rows = list(rows(root / 'eval/gold.jsonl'))
    gold = {r['id']: r for r in gold_rows}
    require(len(gold) == len(gold_rows) == 2000, 'Expected 2000 distinct gold IDs')
    require(Counter(r['qid'] for r in gold.values()) == Counter({qid: 2 for qid in index}), 'Two tests per rule required')
    require(all(r['answer'] in 'ABCD' and len(r['answer']) == 1 for r in gold.values()), 'Bad choice label')
    evaluations = {}
    eval_images = set()
    eval_report = {}
    for direction in DIRECTIONS:
        rr = list(rows(root / f'eval/{direction}.jsonl'))
        by_id = {r['id']: r for r in rr}
        require(len(rr) == len(by_id) == 2000 and set(by_id) == set(gold), 'Eval/gold IDs differ')
        for r in rr:
            require(r['qid'] == gold[r['id']]['qid'], 'Eval/gold qid mismatch')
            require([m['role'] for m in r['messages']] == ['system', 'user'], 'Eval contains assistant target')
            require(not any(k in r for k in ('answer', 'label', 'gold')), 'Gold in inference input')
            require(r['messages'][-1]['content'] not in train_prompts, 'Exact test prompt leaked into train')
            visual = direction.endswith('V')
            paths = r.get('images', [])
            require(sum(m['content'].count('<image>') for m in r['messages']) == int(visual), 'Eval marker mismatch')
            require(len(paths) == int(visual), 'Eval image count mismatch')
            for path in paths:
                require(path in [f"images/{r['qid']}/test_visual_1.png", f"images/{r['qid']}/test_visual_2.png"], 'Wrong test image')
                eval_images.add(path)
        evaluations[direction] = by_id
        eval_report[direction] = {'rows': len(rr), 'gold_in_input': False}
    require(evaluations['T2T'] == evaluations['V2T'], 'Text evaluation pairs differ')
    require(evaluations['T2V'] == evaluations['V2V'], 'Visual evaluation pairs differ')
    require(len(source_images) == 1000 and len(eval_images) == 2000, 'Incorrect distinct image counts')
    require(not source_images & eval_images, 'Training/test image overlap')
    expected_hash_rows = list(rows(root / 'audit/image_hashes.jsonl'))
    expected_hashes = {r['path']: r['repaired_sha256'] for r in expected_hash_rows}
    require(len(expected_hash_rows) == len(expected_hashes) == 3000, 'Image hash list has duplicates')
    require(source_images | eval_images == set(expected_hashes), 'Image hash coverage differs')
    for path in expected_hashes:
        require(inside(root, path).is_file(), f'Missing image: {path}')
    checked = 0
    if hashes:
        listed = set()
        for line in (root / 'SHA256SUMS').read_text().splitlines():
            expected, rel = line.split('  ', 1)
            require(rel not in listed, 'Duplicate checksum entry')
            listed.add(rel)
            actual = digest(inside(root, rel))
            require(actual == expected, f'Checksum mismatch: {rel}')
            if rel in expected_hashes:
                require(actual == expected_hashes[rel], f'Source image hash mismatch: {rel}')
            checked += 1
        files = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()
                 and '__pycache__' not in p.parts and p.name != 'SHA256SUMS'}
        require(files == listed, 'Checksum file coverage differs')
    return {'status': 'pass', 'rules': 1000, 'families': manifest['families'], 'training': train_report,
            'evaluation': eval_report, 'gold_rows': len(gold), 'images': len(expected_hashes),
            'learning_test_images_disjoint': True, 'original_modality_targets_preserved': True,
            'exact_test_prompts_absent_from_training': True, 'checksums_verified': checked,
            'image_semantic_human_certification': False}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--report', type=Path)
    args = ap.parse_args()
    result = validate(args.root)
    if args.report:
        if args.report.resolve().is_relative_to(args.root.resolve()):
            ap.error('--report must be outside the immutable package')
        dump(args.report, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
