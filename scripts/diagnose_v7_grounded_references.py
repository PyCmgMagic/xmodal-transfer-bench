#!/usr/bin/env python3
"""Learning-reference controls, kept separate from blind adapter evaluation."""
import argparse
import copy
import json
import os
import random
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import pilot_v7_grounded_families as pilot
from run_v7_full_experiment import read, rows, save, sha


def build(out):
    source = rows(out / 'probes.jsonl')
    result = []
    for qid in read(out / 'manifest.json')['rules']:
        for condition in ('seen_atomic_fact', 'new_single_mcq', 'new_direct_mcq'):
            group = [p for p in source if p['qid'] == qid and p['condition'] == condition][:2]
            assert len(group) == 2
            for p in group:
                p = copy.deepcopy(p)
                p['id'] += ':learning_reference'
                p['condition'] = 'reference_' + condition
                p['messages'][1]['content'] = '<image>Using this learning reference image, ' + p['messages'][1]['content']
                p['images'] = [str(pilot.DATA / 'images' / qid / 'learning_visual.png')]
                result.append(p)
    path = out / 'reference_probes.jsonl'
    if path.exists():
        assert rows(path) == result
    else:
        pilot.ev.save_rows(path, result)
    return result


def balanced_probes(out):
    ids = read(out / 'manifest.json')['rules']
    knowledge = {k['content']['id']: k['content'] for k in rows(pilot.DATA / 'audit/rule_knowledge.jsonl')}
    probes, audits = [], []
    for qid in ids:
        k = knowledge[qid]
        rule = json.loads(k['fact']['quote'])
        rng = random.Random('balanced-diagnosis:42:' + qid)
        observations = pilot.pool(rule, 'valid')
        labels = ['KEEP', 'OMIT'] if 'conditions' in rule else rule['category_labels']
        all_results = [pilot.assignment(v) for v in __import__('itertools').product(labels, repeat=4)]
        for i in range(16):
            observed = [rng.choice(observations) for _ in range(4)]
            values = [pilot.check_independently(rule, o) for o in observed]
            result = pilot.assignment(values)
            # Within every rule: one accepted report at every gold letter,
            # plus three rejected reports at every gold letter.
            accepted = (i // 4 + i % 4) % 4 == 0
            submitted = result if accepted else rng.choice([v for v in all_results if v != result])
            question = f'A technician submitted this report under {pilot.name(k)}: {submitted}. Verify it using these observations. Choose the correct complete result.\n'
            question += '\n'.join(f'Candidate {l}: {pilot.describe(rule, o)}.' for l, o in zip('ABCD', observed))
            target = 'Result: ' + result + '.'
            q, gold, options = pilot.choices(question, ['Result: ' + v + '.' for v in all_results], target,
                i % 4, rng, required_wrong='Result: ' + submitted + '.' if not accepted else None)
            audit = {'id': qid + ':balanced:' + str(i), 'qid': qid, 'observations': observed, 'results': values,
                     'accepted': accepted, 'submitted': submitted, 'options': options, 'gold': gold}
            audits.append(audit)
            p = {'id': audit['id'], 'qid': qid, 'relation': k['fact']['relation'], 'condition': 'balanced_diagnosis_blind',
                 'task': 'diagnosis', 'answer_format': 'mcq', 'gold': gold, 'messages': pilot.msg(q, mcq=True)}
            probes.append(p)
            p = copy.deepcopy(p)
            p['id'] += ':material'
            p['condition'] = 'balanced_diagnosis_material'
            p['messages'][1]['content'] = 'Learning material:\n' + k['material']['learning']['text'] + '\n\n' + p['messages'][1]['content']
            probes.append(p)
    for a in audits:
        assert 'Result: ' + a['submitted'] + '.' in a['options']
        assert a['options']['ABCD'.index(a['gold'])] == 'Result: ' + pilot.assignment(a['results']) + '.'
    assert sum(a['accepted'] for a in audits) * 4 == len(audits)
    for filename, content in [('balanced_diagnosis_probes.jsonl', probes), ('balanced_diagnosis_audit.jsonl', audits)]:
        path = out / filename
        content = json.loads(json.dumps(content))
        if path.exists():
            assert rows(path) == content
        else:
            pilot.ev.save_rows(path, content)
    return probes


def run(out, variant, gpu, balanced=False):
    probes = balanced_probes(out) if balanced else build(out)
    folder = out / ('balanced_diagnostic' if balanced else 'reference_diagnostic') / variant
    folder.mkdir(parents=True, exist_ok=True)
    if read(folder / 'report.json').get('status') == 'complete':
        return
    raw = folder / 'predictions.jsonl'
    records, notes = pilot.ev.read_rows(raw, allow_partial_tail=True) if raw.exists() else ([], [])
    if notes:
        pilot.ev.save_rows(raw, records)
    seen = {r['id'] for r in records}
    pending = [p for p in probes if p['id'] not in seen]
    cp = None if variant == 'base' else Path(read(out / variant / 'checkpoint_manifest.json')['path'])
    with pilot.reserve(pilot.WORK, gpu, str(folder), 30000) as reservation:
        reservation.bind(os.getpid())
        os.environ.update(pilot.environment(gpu, pilot.DATA))
        import torch
        from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
        from swift.utils import seed_everything
        seed_everything(42)
        engine = TransformersEngine(str(pilot.MODELS['qwen3']), adapters=[str(cp)] if cp else None,
                                    torch_dtype=torch.bfloat16, device_map='cuda:0', max_batch_size=4, attn_impl='sdpa')
        engine.template.enable_thinking = False
        for fmt in ('semantic', 'mcq'):
            group = [p for p in pending if p['answer_format'] == fmt]
            for start in range(0, len(group), 4):
                batch = group[start:start + 4]
                res = engine.infer([InferRequest(messages=copy.deepcopy(p['messages']), images=p.get('images', [])) for p in batch],
                    RequestConfig(max_tokens=48 if fmt == 'semantic' else 16, temperature=0, seed=42), use_tqdm=False)
                assert len(res) == len(batch)
                for p, answer in zip(batch, res):
                    if isinstance(answer, Exception):
                        raise answer
                    text = answer.choices[0].message.content or ''
                    letter = pilot.ev.letter(text) if fmt == 'mcq' else None
                    row = {k: p[k] for k in ('id', 'qid', 'condition', 'relation', 'task', 'gold', 'answer_format')}
                    row.update(response=text, predicted=letter, format_error=fmt == 'mcq' and letter is None,
                               correct=letter == p['gold'] if fmt == 'mcq' else pilot.semantic_correct(text, p['gold']),
                               prompt_sha256=pilot.ev.digest(p['messages']), saved_at=pilot.ev.now())
                    with raw.open('a') as f:
                        f.write(json.dumps(row, ensure_ascii=False) + '\n')
                        f.flush()
                        os.fsync(f.fileno())
                    records.append(row)
                print(variant, fmt, start + len(batch), '/', len(group), flush=True)
    metrics = {}
    for condition in sorted({r['condition'] for r in records}):
        g = [r for r in records if r['condition'] == condition]
        metrics[condition] = {'n': len(g), 'correct': sum(r['correct'] for r in g),
            'format_errors': sum(r['format_error'] for r in g),
            'by_relation': {rel: {'n': sum(r['relation'] == rel for r in g),
                                 'correct': sum(r['correct'] for r in g if r['relation'] == rel)} for rel in pilot.RELATIONS}}
    save(folder / 'report.json', {'status': 'complete', 'variant': variant, 'metrics': metrics,
         'probes_sha256': sha(out / ('balanced_diagnosis_probes.jsonl' if balanced else 'reference_probes.jsonl')), 'script_sha256': sha(Path(__file__)),
         'checkpoint': pilot.ev.checkpoint(cp) if cp else None,
         'external_learning_reference': not balanced, 'blind_result': False,
         'scope': 'Blind and material conditions separately labeled' if balanced else 'All questions include the original learning image',
         'finished_at': pilot.ev.now()})


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--variant', choices=('base', *pilot.VARIANTS), required=True)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--balanced-diagnosis', action='store_true')
    a = ap.parse_args()
    run(a.out.resolve(), a.variant, a.gpu, a.balanced_diagnosis)
