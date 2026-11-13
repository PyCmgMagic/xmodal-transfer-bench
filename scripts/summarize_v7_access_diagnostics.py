#!/usr/bin/env python3
"""Validate completed mechanism studies and publish supplementary statistics.

This performs no inference, training, or changes to source data/raw responses.
"""
import csv
import hashlib
import json
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/v7_access_diagnostics_20261015'
BRIDGE = ROOT / 'runs/v7_reference_bridge_20261015'
PILOT = ROOT / 'runs/v7_task_then_transfer_20261015/pilot_v2'
DATA = ROOT / 'rules_repaired_single_image_v7'
ARMS = ('base', 'warmup', 'text_alias', 'vision', 'full_sft_text', 'full_sft_vision')


def read(p):
    return json.loads(Path(p).read_text())


def rows(p):
    return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save(p, value):
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sources(m):
    for p, h in m.get('sources', {}).items():
        assert sha(p) == h, p


def adapter(m):
    if m is not None:
        for f, h in m['sha256'].items():
            assert sha(Path(m['path']) / f) == h, m['path']


def responses(path, probes, hash_key='input_sha256', mode='images'):
    rr = rows(path)
    pp = {r['id']: r for r in probes}
    assert len(rr) == len({r['id'] for r in rr}) == len(pp), path
    assert {r['id'] for r in rr} == set(pp), path
    for r in rr:
        p = pp[r['id']]
        data = {'messages': p['messages'], 'images': p.get('images', [])} if mode == 'images' else p['messages']
        assert r[hash_key] == digest(data), (path, r['id'])
        assert r['response'] == (r['raw_response']['choices'][0]['message']['content'] or '')
        assert r['gold'] == p['gold']
    assert read(path.parent / 'report.json')['status'] == 'complete'
    inventory.append({'file': str(path), 'sha256': sha(path), 'records': len(rr),
                      'reused': sum('reused_from' in r for r in rr)})
    return rr


def stats(rr):
    return {'n': len(rr), 'correct': sum(r['correct'] for r in rr),
            'format_errors': sum(r['format_error'] for r in rr),
            'truncated': sum(r['finish_reason'] == 'length' for r in rr),
            'empty': sum(not r['response'].strip() for r in rr)}


inventory = []
summary = {'scope': 'Qwen3-VL-4B development mechanism study, not a replacement full benchmark',
           'access': {}, 'bridge': {}, 'name_control': {}}
manifest = read(OUT / 'manifest.json')
assert sha(ROOT / 'scripts/diagnose_v7_knowledge_access.py') == manifest['script_sha256']
assert sha(OUT / 'probes.jsonl') == manifest['probes_sha256']
sources(manifest)
sources(read(PILOT / 'manifest.json'))
probes = rows(OUT / 'probes.jsonl')
for arm in ARMS:
    adapter(manifest['models'][arm])
    rr = responses(OUT / arm / 'predictions.jsonl', probes, 'prompt_sha256')
    metrics = {}
    for c in sorted({r['condition'] for r in rr}):
        group = [r for r in rr if r['condition'] == c]
        metrics[c] = stats(group)
        if c.startswith('recall_image_'):
            metrics[c].update(requested_rule_body_exact=sum(r['requested_rule_body_exact'] for r in group),
                              image_donor_body_exact=sum(r.get('image_donor_body_exact', False) for r in group))
        if c.startswith('rule_then_option_'):
            # Supplementary parser only. Retain historical strict format errors.
            scores = []
            for r in group:
                match = re.search(r'\bFINAL:\s*([ABCD])\s*\.?\s*\Z', r['response'])
                pred = match[1] if match and len(re.findall(r'\bFINAL:', r['response'])) == 1 else None
                scores.append({'id': r['id'], 'predicted': pred, 'correct': pred == r['gold'],
                               'format_error': pred is None, 'finish_reason': r['finish_reason']})
            metrics[c]['supplementary_lenient'] = {'n': len(scores), 'correct': sum(x['correct'] for x in scores),
                                                   'format_errors': sum(x['format_error'] for x in scores)}
            save(OUT / arm / (c + '_lenient_scores.json'), scores)
    assert {k: {x: v[x] for x in ('n', 'correct', 'format_errors', 'truncated', 'empty')} for k, v in metrics.items()} == {
        k: {x: v[x] for x in ('n', 'correct', 'format_errors', 'truncated', 'empty')}
        for k, v in read(OUT / arm / 'report.json')['metrics'].items()}
    summary['access'][arm] = metrics

for arm in ('full_sft_text', 'text_alias'):
    folder = OUT / arm / 'recall_context'
    m = read(folder / 'manifest.json'); sources(m); adapter(m['adapter'])
    assert sha(ROOT / 'scripts/diagnose_v7_recall_context.py') == m['script_sha256']
    pp = rows(folder / 'probes.jsonl'); assert digest(pp) == m['probes_hash']
    responses(folder / 'predictions.jsonl', pp)
    folder = OUT / arm / 'teacher_forcing'
    m = read(folder / 'manifest.json'); adapter(m['adapter'])
    assert sha(m['source']) == m['source_sha256']
    assert sha(ROOT / 'scripts/diagnose_v7_teacher_forcing.py') == m['script_sha256']
    train = rows(m['source'])
    if arm == 'text_alias':
        train = [r for r in train if r['id'].endswith(':0')]
    mm = rows(folder / 'token_metrics.jsonl'); trainmap = {r['id']: r for r in train}
    assert len(mm) == len({r['id'] for r in mm}) == len(train) == m['examples']
    for r in mm:
        assert r['original_training_input_sha256'] == digest(trainmap[r['id']]['messages'])
        assert r['template_mask_verified'] and math.isfinite(r['mean_loss'])
    assert read(folder / 'label_mask_audit.json')['all_supervised_ids_hashes_match']
    free = rows(folder / 'free_recall.jsonl'); assert len(free) == len({r['qid'] for r in free}) == 8
    for r in free:
        assert r['input_sha256'] == digest(trainmap[r['id']]['messages'][:-1])
    inventory.append({'file': str(folder / 'free_recall.jsonl'), 'sha256': sha(folder / 'free_recall.jsonl'), 'records': 8, 'reused': 0})

reviews = rows(OUT / 'semantic_reviews.jsonl'); assert len(reviews) == 56
for r in reviews:
    assert sha(r['source']) == r['source_sha256']
    source = next(x for x in rows(r['source']) if x['id'] == r['id'])
    assert digest(source['response']) == r['response_sha256']

bm = read(BRIDGE / 'manifest.json'); sources(bm); adapter(bm['parent'])
assert sha(ROOT / 'scripts/pilot_v7_reference_bridge.py') == bm['script_sha256']
for p, h in bm['files'].items():
    assert sha(BRIDGE / p) == h
a, b = rows(BRIDGE / 'reference_dev.jsonl'), rows(BRIDGE / 'memory_dev.jsonl')
assert len(a) == len(b) == 512
assert not set(bm['study_rules']) & set(bm['dev_rules'])
for x, y in zip(a, b):
    assert x['id'] == y['id'] and x['messages'][-1] == y['messages'][-1]
    if x['qid'] in bm['study_rules']:
        assert x == y
    else:
        assert x['messages'][-2]['content'].endswith(y['messages'][-2]['content'])
pp = rows(BRIDGE / 'probes.jsonl')
for arm in ('warmup', 'text_alias', 'reference_dev', 'memory_dev'):
    folder = BRIDGE / arm
    adapter(read(folder / 'eval_manifest.json')['adapter'])
    rr = responses(folder / 'predictions.jsonl', pp)
    report = read(folder / 'report.json')
    metrics = {c: stats([r for r in rr if r['condition'] == c]) for c in {r['condition'] for r in rr}}
    consistency = {}
    for c in ('direct_T', 'direct_V'):
        groups = defaultdict(list)
        for r in rr:
            if r['condition'] == c:
                groups[r['id'].split(':', 1)[1]].append(r)
        assert len(groups) == 16 and all(len(g) == 4 for g in groups.values())
        consistency[c] = {'original_questions': 16,
                          'same_answer_letter_all4': sum(len({r['predicted'] for r in g}) == 1 for g in groups.values()),
                          'same_semantic_option_all4': sum(all(r['predicted'] is not None for r in g) and len({(ord(r['predicted']) - ord('A') + r['rotation']) % 4 for r in g}) == 1 for g in groups.values()),
                          'all4_correct': sum(all(r['correct'] for r in g) for g in groups.values())}
    if arm in ('reference_dev', 'memory_dev'):
        cm = read(folder / 'checkpoint_manifest.json'); adapter(cm); assert cm['epoch'] == 1 and cm['global_step'] == 128
        st = read(Path(cm['path']) / 'trainer_state.json'); assert st['epoch'] == 1 and st['global_step'] == 128
        args = read(Path(cm['path']) / 'args.json')
        assert args['adapters'] == [bm['parent']['path']]
        assert args['dataset'] == [str(BRIDGE / (arm + '.jsonl'))]
    summary['bridge'][arm] = {'metrics': metrics, 'rotation_consistency': consistency}

design = read(OUT / 'rule_name_control/design.json')
assert design['script_sha256'] == sha(ROOT / 'scripts/diagnose_v7_rule_name_control.py')
assert design['probes_sha256'] == sha(OUT / 'rule_name_control/probes.jsonl')
assert sha(design['source']) == design['source_sha256']
pp = rows(OUT / 'rule_name_control/probes.jsonl')
# Re-run both deterministic solvers without loading any model.
from diagnose_v7_rule_name_control import solve, independent
knowledge = {k['id']: k for k in rows(PILOT / 'target_knowledge.jsonl')}
blocks = rows(OUT / 'rule_name_control/blocks.jsonl')
for block in blocks:
    assert len(set(block['rule_results'].values())) == 4
    for qid, result in block['rule_results'].items():
        rule = json.loads(knowledge[qid]['fact']['quote'])
        assert all(solve(rule, x) == independent(rule, x) for x in block['observations'])
        assert result == '; '.join(f'{letter}={solve(rule, x)}' for letter, x in zip('ABCD', block['observations']))
for block in range(8):
    gg = [p for p in pp if p['block'] == block and p['condition'] == 'blind']
    assert {p['gold'] for p in gg} == set('ABCD')
    assert len({p['messages'][-1]['content'].split('. Report KEEP/OMIT', 1)[1] for p in gg}) == 1
for arm in ('text_alias', 'warmup'):
    folder = OUT / 'rule_name_control' / arm
    adapter(read(folder / 'manifest.json')['adapter'])
    rr = responses(folder / 'predictions.jsonl', pp, mode='messages')
    summary['name_control'][arm] = read(folder / 'report.json')['metrics']

audit = read(OUT / 'supervision_audit/supervision_audit.json'); sources(audit)
assert sha(ROOT / 'scripts/audit_v7_supervision_information.py') == audit['script_sha256']
historical = []
dm = read(ROOT / 'runs/v7_full_experiment/data/manifest.json')
for source in dm['sources'].values():
    assert sha(source['path']) == source['sha256'], source['path']
for name, entry in dm['outputs'].items():
    assert sha(ROOT / 'runs/v7_full_experiment/data' / name) == entry['sha256']
for p in sorted((ROOT / 'runs/v7_full_experiment/jobs').glob('*/checkpoint/args.json')):
    args = read(p); objective, modality = p.parts[-3].split('_')[1:]
    expected = ROOT / 'runs/v7_full_experiment/data' / f'{objective}_{modality}.jsonl'
    assert args['dataset'] == [str(expected)], p
    historical.append({'job': p.parts[-3], 'args_sha256': sha(p), 'dataset': str(expected), 'dataset_sha256': sha(expected), 'epochs': args['num_train_epochs']})
assert len(historical) == 12
save(OUT / 'historical_dataset_usage.json', historical)
images = rows(DATA / 'image_hashes.jsonl'); assert len(images) == 3000
for r in images:
    assert sha(DATA / r['path']) == r['repaired_sha256'] == r['source_sha256'], r['path']

summary['teacher_forcing'] = {arm: read(OUT / arm / 'teacher_forcing/report.json') for arm in ('full_sft_text', 'text_alias')}
summary['semantic_reviews'] = read(OUT / 'semantic_review_summary.json')
summary['supervision'] = audit
summary['accounting'] = {'saved_response_records': sum(r['records'] for r in inventory),
                         'reused_response_records': sum(r['reused'] for r in inventory),
                         'new_generations': sum(r['records'] - r['reused'] for r in inventory),
                         'teacher_forced_examples': 1008, 'manual_reviews': 56,
                         'note': 'Different option rotations share observations; saved records and request counts are not independent test questions. Prior pilot responses outside this inventory are not added.'}
save(OUT / 'summary.json', summary)
save(OUT / 'response_inventory.json', inventory)
with (OUT / 'summary.csv').open('w') as f:
    writer = csv.DictWriter(f, fieldnames=['study', 'arm', 'condition', 'n', 'correct', 'format_errors', 'truncated', 'empty'])
    writer.writeheader()
    for arm, metrics in summary['access'].items():
        for c, m in metrics.items():
            writer.writerow(dict(study='access_strict', arm=arm, condition=c, **{k: m[k] for k in ('n', 'correct', 'format_errors', 'truncated', 'empty')}))
            if 'supplementary_lenient' in m:
                writer.writerow(dict(study='access_lenient', arm=arm, condition=c, **m['supplementary_lenient'], truncated=m['truncated'], empty=m['empty']))
    for arm, value in summary['bridge'].items():
        for c, m in value['metrics'].items():
            writer.writerow(dict(study='bridge_strict', arm=arm, condition=c, **m))
    for arm, metrics in summary['name_control'].items():
        for c, m in metrics.items():
            writer.writerow(dict(study='name_control', arm=arm, condition=c, **{k: m[k] for k in ('n', 'correct', 'format_errors')}))

snap = OUT / 'source_snapshots/final_diagnostics'
snap.mkdir(parents=True, exist_ok=True)
for name in ('summarize_v7_access_diagnostics.py', 'audit_v7_supervision_information.py',
             'diagnose_v7_recall_context.py', 'diagnose_v7_teacher_forcing.py', 'review_v7_access_cases.py',
             'pilot_v7_reference_bridge.py', 'diagnose_v7_rule_name_control.py'):
    source = ROOT / 'scripts' / name; target = snap / name
    if target.exists() and sha(source) != sha(target):
        target = snap / (source.stem + '_' + sha(source)[:12] + source.suffix)
    if not target.exists():
        shutil.copy2(source, target)
save(OUT / 'validation.json', {'status': 'pass', 'all_source_and_adapter_hashes_match': True,
                             'all_original_images_unchanged': 3000, 'historical_training_configs_verified': 12,
                             'response_files_verified': len(inventory), 'accounting': summary['accounting'],
                             'label_mask_examples_verified': 1008, 'manual_reviews_verified': 56,
                             'bridge_pair_matched': True, 'rule_name_control_two_solvers_verified': True,
                             'source_snapshots': {p.name: sha(p) for p in snap.iterdir() if p.is_file()}})
print(json.dumps({'validation': 'pass', **summary['accounting']}, ensure_ascii=False))
