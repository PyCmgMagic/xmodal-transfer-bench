#!/usr/bin/env python3
"""Create rule-explanation SFT pairs using learning material only."""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rule_name(knowledge):
    """Use semantic rule identifiers, never the audit's fallback sample IDs."""
    q = json.loads(knowledge['fact']['quote'])
    names = [q[key] for key in ('predicate_name', 'priority_name', 'index_name',
                               'selector_name', 'procedure_name') if key in q]
    if 'mapping' in q:
        names = list(dict.fromkeys(q['mapping'].values()))
    if 'mapping_by_context' in q:
        names = list(dict.fromkeys(v for m in q['mapping_by_context'].values() for v in m.values()))
    if 'category_labels' in q:
        names = q['category_labels']
    if 'states' in q:
        names = q['states']
    if knowledge['fact']['relation'] == 'bipartite_matching':
        match = re.match(r'Under ([A-Z][a-z]+),', knowledge['material']['learning']['text'])
        assert match, knowledge['id']
        names = [match.group(1)]
    assert names and all(isinstance(n, str) and not n.startswith('scale1k_') for n in names), knowledge['id']
    assert all(n in knowledge['material']['learning']['text'] and
               n in knowledge['material']['learning']['visual_text'] for n in names), knowledge['id']
    return ', '.join(names)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'rules_sft/data')
    args = p.parse_args()
    source = ROOT / 'rules_corrected_final'
    knowledge = {r['content']['id']: r['content'] for r in read(source / 'audit/rule_knowledge.jsonl')}
    assert len(knowledge) == 1000
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {'experiment': 'rule-explanation SFT, not test-choice supervision',
                'source': str(source), 'eval_used_for_training': False,
                'target_policy': 'Original modality-specific learning text verbatim; image moved to user.',
                'sources': {}, 'outputs': {},
                'rule_types': dict(Counter(r['fact']['relation'] for r in knowledge.values()))}
    for mode, filename in [('TEXT', 'train_text.jsonl'), ('VISION', 'train_multimodal.jsonl')]:
        path = source / 'shared' / filename
        # Corrected evaluation must not silently change the training exposure.
        assert read(path) == read(ROOT / 'rules/shared' / filename)
        rows = read(path)
        assert len(rows) == len({r['id'] for r in rows}) == 1000
        result = []
        for r in rows:
            assert [m['role'] for m in r['messages']] == ['assistant']
            content = r['messages'][0]['content']
            k = knowledge[r['id']]
            name = rule_name(k)
            question = f'Explain the rule or codebook named {name}, including its conditions, mappings, and how to apply it.'
            out = {'id': r['id'], 'qid': r['qid']}
            if mode == 'VISION':
                images = [(source / x).resolve() for x in r['images']]
                assert content.count('<image>') == len(images) == 1
                assert all(x.is_file() and x.name == 'learning_visual.png' for x in images)
                question = '<image>Using this learning reference image, ' + question[0].lower() + question[1:]
                content = content.replace('<image>', '')
                assert content == k['material']['learning']['visual_text']
                out['images'] = [str(x) for x in images]
            else:
                assert content == k['material']['learning']['text']
                assert '<image>' not in content
            out['messages'] = [{'role': 'user', 'content': question},
                               {'role': 'assistant', 'content': content}]
            result.append(out)
        dest = args.output / filename
        serialized = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in result)
        if dest.exists() and dest.read_text() != serialized:
            raise RuntimeError(f'Refusing to replace changed dataset: {dest}')
        dest.write_text(serialized)
        manifest['sources'][mode] = {'path': str(path), 'sha256': digest(path)}
        manifest['outputs'][mode] = {'path': str(dest.resolve()), 'rows': len(result), 'sha256': digest(dest)}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
