#!/usr/bin/env python3
"""Independent truth checks and completion audit for the atomic supervision pilot."""
import argparse
import json
import math
import operator
import re
from collections import Counter
from fractions import Fraction
from pathlib import Path

import run_v7_evaluation as ev
from pilot_v7_atomic_supervision import OUT, ARMS, DATA, atoms, card, validate
from run_v7_full_experiment import read, rows, save, sha

OPS = {'eq': operator.eq, 'ne': operator.ne, 'ge': operator.ge,
       'gt': operator.gt, 'le': operator.le, 'lt': operator.lt}


def truth(rule, value):
    if 'conditions' in rule:
        flags = [OPS[c['operator']](value[c['field']], c['value']) for c in rule['conditions']]
        selected = sum(flags) >= rule['required_count'] if 'required_count' in rule else all(flags) if rule['mode'] == 'all' else any(flags)
        return 'KEEP' if selected else 'OMIT'
    lower = Fraction(str(rule['thresholds']['lower'])); upper = Fraction(str(rule['thresholds']['upper']))
    return rule['category_labels'][0 if value < lower else 2 if value > upper else 1]


def validate_original_questions(out):
    knowledge = {k['id']: k for k in rows(out / 'target_knowledge.jsonl')}
    audited = []
    for p in rows(out / 'probes.jsonl'):
        if p['condition'] != 'direct_T':
            continue
        rule = json.loads(knowledge[p['qid']]['fact']['quote'])
        text = p['messages'][-1]['content']; heading, rest = text.split('\n\nTest input:\n', 1)
        observed, options = rest.split('\n\nOptions:\n', 1)
        if 'conditions' in rule:
            observations = re.findall(r'([ABCD]) has (.*?)(?:; |\.\s*$)', observed)
            assert len(observations) == 4, (p['id'], observed)
            values = {}
            for candidate, body in observations:
                attrs = dict(re.findall(r'(\w+): ([^,]+)(?:, |$)', body))
                if 'value' in attrs:
                    attrs['value'] = float(attrs['value'])
                values[candidate] = truth(rule, attrs)
        else:
            ratios = [Fraction(int(a), int(b)) for a, b in re.findall(r'giving a share of (\d+)/(\d+)', observed)]
            assert len(ratios) == (4 if p['task'] == 'direct' else 1)
            values = {letter: truth(rule, ratio) for letter, ratio in zip('ABCD' if len(ratios) == 4 else 'X', ratios)}
        expected = '; '.join(f'{k}={v}' for k, v in values.items())
        accept = False
        if p['task'] == 'diagnosis':
            submitted = re.search(r'Result: (.*?)(?:\. |\.$)', heading)[1]
            accept = submitted == expected
            if 'thresholds' in rule:
                lower, upper = map(float, re.search(r'Cutoffs: lower=([\d.]+), upper=([\d.]+)\.', heading).groups())
                accept = accept and math.isclose(lower, rule['thresholds']['lower']) and math.isclose(upper, rule['thresholds']['upper'])
        valid = []
        for letter, option in re.findall(r'(?m)^([ABCD])\. (.*)$', options):
            assignments = re.search(r'Result: (.*?)\.$', option)[1]
            good = assignments == expected
            if p['task'] == 'diagnosis':
                good = good and option.startswith('Accept report.' if accept else 'Replace report.')
            if 'thresholds' in rule and p['task'] == 'diagnosis':
                lower, upper = map(float, re.search(r'Cutoffs: lower=([\d.]+), upper=([\d.]+)\.', option).groups())
                good = good and math.isclose(lower, rule['thresholds']['lower']) and math.isclose(upper, rule['thresholds']['upper'])
            if good:
                valid.append(letter)
        assert valid == [p['gold']], (p['id'], valid, p['gold'])
        audited.append({'id': p['id'], 'qid': p['qid'], 'expected_assignment': expected,
                        'accept_report': accept if p['task'] == 'diagnosis' else None,
                        'unique_correct_option': p['gold'], 'input_sha256': ev.digest(p['messages'])})
    assert len(audited) == 16
    ev.save_rows(out / 'original_label_checks.jsonl', audited)


def audit(out, complete):
    validate(out); validate_original_questions(out)
    original_manifest = read(out.parent / 'v7_full_experiment/data/manifest.json')
    for entry in original_manifest['sources'].values():
        assert sha(Path(entry['path'])) == entry['sha256'], entry['path']
    ks = {k['id']: k for k in rows(out / 'target_knowledge.jsonl')}
    observations = [p['messages'][-1]['content'].split('\n\nTest input:\n', 1)[1].split('\n\nOptions:\n', 1)[0]
                    for p in rows(out / 'probes.jsonl') if p['condition'] == 'direct_T']
    training = {}
    for arm in ARMS[:-1]:
        rr = rows(out / f'{arm}.jsonl'); counts = Counter(r['qid'] for r in rr)
        assert counts == Counter({qid: 64 for qid in ks})
        for r in rr:
            content = '\n'.join(m['content'] for m in r['messages'])
            assert 'Options:\n' not in content and 'Test input:\n' not in content
            assert not any(observation in content for observation in observations)
        training[arm] = {'rows': 512, 'assistant_characters': sum(len(r['messages'][-1]['content']) for r in rr),
                         'tasks': dict(Counter(r['task'] for r in rr)), 'image_rows': sum(bool(r.get('images')) for r in rr)}
        if arm.startswith('atomic'):
            for r in rr:
                if r['task'] == 'full_rule':
                    assert r['messages'][-1]['content'] == card(ks[r['qid']])
                    continue
                expected = {key: value for key, _, value in atoms(ks[r['qid']])}
                assert r['messages'][-1]['content'] == expected[r['task']]
    byarm = {}
    for arm in ARMS:
        folder = out / arm
        if not (folder / 'report.json').exists():
            assert not complete, 'Unfinished: ' + arm
            continue
        report = read(folder / 'report.json'); assert report['status'] == 'complete'
        config = read(folder / 'eval_manifest.json')
        assert ev.checkpoint(Path(config['adapter']['path'])) == config['adapter']
        pp = rows(out / 'probes.jsonl') + rows(folder / 'self_probes.jsonl')
        probes = {p['id']: p for p in pp}; rr = rows(folder / 'predictions.jsonl')
        assert len(rr) == len({r['id'] for r in rr}) == len(pp) == 208
        recalled = {r['qid']: r for r in rr if r['condition'] == 'recall_task'}
        for r in rr:
            p = probes[r['id']]
            assert r['input_sha256'] == ev.digest({'messages': p['messages'], 'images': p.get('images', [])})
            assert r['response'] == (r['raw_response']['choices'][0]['message']['content'] or '')
            assert r['gold'] == p['gold']
            if r['condition'].startswith('self_'):
                assert r['recall_response_sha256'] == ev.digest(recalled[r['qid']]['response'])
                original = probes[r['id'].removeprefix('self:')]
                assert p['messages'][-1]['content'] == 'Your own prior rule recall (uncorrected):\n' + recalled[r['qid']]['response'] + '\n\n' + original['messages'][-1]['content']
            if r['fmt'] == 'mcq':
                pred = ev.letter(r['response']); assert r['predicted'] == pred and r['correct'] == (pred == r['gold'])
        if arm != 'warmup':
            cm = read(folder / 'checkpoint_manifest.json'); assert cm['epoch'] == 1 and cm['global_step'] == 128
            args = read(Path(cm['path']) / 'args.json')
            assert args['dataset'] == [str(out / (arm + '.jsonl'))]
            assert args['adapters'] == [read(out / 'manifest.json')['parent']['path']]
            expected_args = {'num_train_epochs': 1, 'learning_rate': 1e-4,
                             'per_device_train_batch_size': 4, 'gradient_accumulation_steps': 1,
                             'lora_rank': 8, 'lora_alpha': 32, 'lora_dropout': .05,
                             'max_length': 8192, 'seed': 42, 'freeze_vit': True, 'freeze_aligner': True}
            for key, value in expected_args.items():
                assert args[key] == value, (arm, key, args[key])
            assert len(rows(folder / 'checkpoint/logging.jsonl')) >= 25
        # Breakdown is descriptive; never select checkpoints/rules from these scores.
        breakdown = {}
        for condition in ('direct_T', 'direct_V', 'self_T', 'self_V', 'oracle_T', 'oracle_V'):
            group = [r for r in rr if r['condition'] == condition]
            breakdown[condition] = {}
            for field in ('task', 'relation', 'evaluation_group'):
                breakdown[condition][field] = {value: {'n': sum(r.get(field) == value for r in group),
                                                    'correct': sum(r['correct'] for r in group if r.get(field) == value)}
                                                 for value in sorted({r[field] for r in group})}
        byarm[arm] = {'metrics': report['metrics'], 'breakdown': breakdown}
    neutral = {}
    for arm in ARMS:
        folder = out / 'neutral_recall' / arm
        if not (folder / 'report.json').exists():
            assert not complete, 'Unfinished neutral: ' + arm
            continue
        config = read(folder / 'manifest.json')
        assert ev.checkpoint(Path(config['adapter']['path'])) == config['adapter']
        assert sha(out / 'neutral_recall/design.json') == config['design_sha256']
        pp = rows(out / 'neutral_recall/recall_probes.jsonl') + rows(folder / 'application_probes.jsonl')
        lookup = {p['id']: p for p in pp}; rr = rows(folder / 'predictions.jsonl')
        assert len(rr) == len({r['id'] for r in rr}) == len(lookup) == 40
        recalled = {r['qid']: r for r in rr if r['condition'] == 'recall_neutral'}
        for r in rr:
            p = lookup[r['id']]
            assert r['input_sha256'] == ev.digest({'messages': p['messages'], 'images': p.get('images', [])})
            assert r['response'] == (r['raw_response']['choices'][0]['message']['content'] or '')
            assert r['gold'] == p['gold']
            if r['condition'].startswith('neutral_self_'):
                assert r['recall_response_sha256'] == ev.digest(recalled[r['qid']]['response'])
                original = next(x for x in rows(out / 'probes.jsonl') if x['id'] == r['id'].removeprefix('neutral_self:'))
                assert p['messages'][-1]['content'] == 'Your own prior rule recall (uncorrected):\n' + recalled[r['qid']]['response'] + '\n\n' + original['messages'][-1]['content']
                assert r['correct'] == (ev.letter(r['response']) == p['gold'])
        neutral[arm] = read(folder / 'report.json')
    if (out / 'semantic_reviews.jsonl').exists():
        reviews = rows(out / 'semantic_reviews.jsonl'); assert len(reviews) == 160
        for r in reviews:
            assert sha(Path(r['source'])) == r['source_sha256']
            response = next(x for x in rows(Path(r['source'])) if x['id'] == r['id'])
            assert ev.digest(response['response']) == r['response_sha256']
    else:
        assert not complete, 'Missing semantic reviews'
    save(out / 'summary.json', {'scope': '8-rule Qwen3 SFT pilot; not full1000 benchmark or PT results',
                                'arms': byarm, 'neutral_protocol': neutral, 'training': training,
                                'scoring_note': 'Card and short-fact match counts are sufficient-only diagnostics; nonmatches need semantic audit.'})
    save(out / 'validation.json', {'status': 'pass' if len(byarm) == 5 else 'preflight_pass',
                                   'completed_arms': list(byarm), 'original_text_labels_independently_verified': 16,
                                   'unchanged_original_images': 3000, 'four_training_sets_checked': True,
                                   'no_target_test_data_in_training': True,
                                   'completed_neutral_arms': list(neutral),
                                   'manual_full_rule_reviews': 160 if (out / 'semantic_reviews.jsonl').exists() else 0,
                                   'generation_calls': len(byarm) * 208 + len(neutral) * 40,
                                   'script_sha256': sha(Path(__file__)), 'checked_at': ev.now()})
    print(json.dumps({'completed': list(byarm), 'validation': 'pass' if len(byarm) == 5 else 'preflight_pass'}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, default=OUT); ap.add_argument('--complete', action='store_true')
    a = ap.parse_args(); audit(a.out.resolve(), a.complete)
