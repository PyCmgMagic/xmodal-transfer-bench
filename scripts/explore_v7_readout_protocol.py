#!/usr/bin/env python3
"""Fixed-image fill/row readout and no-image recall prompt interventions."""
import argparse
import copy
import json
import os
import shutil
from pathlib import Path

import explore_v7_visual_replay as replay
import run_v7_evaluation as ev
from build_v7_experiment_data import name
from run_v7_full_experiment import ROOT, rows, read, save, sha
from summarize_v7_counterfactual_names import metric
from v7_gpu_resources import reserve

OUT = ROOT / 'runs/v7_readout_protocol_20261016'
ARMS = ('warmup', 'text', 'vision', 'vision_control', 'text_replay')
VARIANTS = ('plain', 'page_memory')
SOURCE = replay.SOURCE


def arm_root(arm):
    return replay.OUT if arm in replay.ARMS else SOURCE


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        validate(out)
        return
    replay.validate(replay.OUT)
    assert read(replay.OUT / 'completion_audit.json')['status'] == 'complete'
    original = rows(replay.OUT / 'extraction_probes.jsonl')
    probes = []
    for p in original:
        z = copy.deepcopy(p)
        z.update(id='improved:' + p['id'], condition='improved_observation', fmt='extract')
        if p['group'] == 0:
            q = ('<image>For each candidate A, B, C, D, report its shape and INTERIOR FILL COLOR. '
                 'Ignore the dark outline and the surrounding card/background. '
                 'A white or unfilled-looking interior is white even when the outline is dark. '
                 'Use one of these color labels: red, blue, green, yellow, black, white. '
                 'Very dark nearly black fill belongs to black; ordinary saturated blue fill belongs to blue. '
                 'Return only JSON with keys A, B, C, D, each containing shape and color in lowercase English. '
                 'Do not apply a rule or choose an option.')
        else:
            q = ('<image>For each candidate A, B, C, D, count the visible small circular dots. '
                 'Inspect each horizontal row separately from top to bottom. '
                 'Return the number of actual dots in each row as row_counts, then add them for value. '
                 'Do not assume an incomplete row has the same count as a full row. '
                 'Do not count the letter label, boxes, outlines, or empty grid positions. '
                 'Return only JSON: each of the keys A, B, C, D must map to an object containing '
                 'row_counts (an array of nonnegative integers) and value (their sum). '
                 'Do not apply a rule or choose an option.')
        z['messages'] = [{'role': 'user', 'content': q}]
        probes.append(z)
    ev.save_rows(out / 'extraction_probes.jsonl', probes)
    knowledge = rows(SOURCE / 'target_knowledge.jsonl')
    recalls = []
    for variant in VARIANTS:
        for k in knowledge:
            identifier = name(k)
            if variant == 'plain':
                q = f'Explain the complete rule named {identifier}.'
            else:
                q = (f'From memory, reproduce the complete rule from the learning reference titled {identifier}. '
                     'No reference image is available in this conversation. Include every exact condition, '
                     'comparison, value, combination and boundary. State the stored rule itself, not an example.')
            recalls.append({'id': f'{variant}:{k["id"]}', 'qid': k['id'], 'condition': variant,
                            'phase': 'recall', 'fmt': 'card', 'messages': [{'role': 'user', 'content': q}],
                            'gold': replay.previous.card(k)})
    assert len(probes) == 5 and len(recalls) == 16
    ev.save_rows(out / 'recall_probes.jsonl', recalls)
    files = [replay.OUT / 'manifest.json', replay.OUT / 'completion_audit.json',
        replay.OUT / 'extraction_probes.jsonl', SOURCE / 'probes.jsonl', SOURCE / 'blocks.jsonl',
        SOURCE / 'target_knowledge.jsonl', replay.OUT / 'source_hash_validation.json']
    adapters = {}
    for arm in ARMS:
        root = arm_root(arm)
        for f in ('predictions.jsonl', 'eval_manifest.json'):
            files.append(root / arm / f)
        adapters[arm] = read(root / arm / 'eval_manifest.json')['adapter']
        for f in ('predictions.jsonl', 'report.json'):
            files.append(replay.OUT / 'observation_diagnostic' / arm / f)
    files += [ROOT / 'scripts' / p for p in ('explore_v7_visual_replay.py', 'run_v7_evaluation.py', 'run_v7_full_experiment.py', 'v7_gpu_resources.py')]
    save(out / 'manifest.json', {'created_at': ev.now(), 'script_sha256': sha(Path(__file__)),
        'sources': {str(p): sha(p) for p in files},
        'files': {p.name: sha(p) for p in out.glob('*.jsonl')}, 'adapters': adapters,
        'arms': ARMS, 'native_cot': False, 'new_training': 0,
        'planned_per_arm': {'observation_generations': 5, 'recall_generations': 16,
            'observation_application': 192, 'recall_application': 160, 'total': 373},
        'planned_total_new_generations': 1865,
        'observation_protocol': 'Same5PNGs. Global fill/outline conventions, no per-scene facts. Numeric model outputs row_counts and value. Three observation sources: previous generic transcript; improved model total; deterministic sum of model row_counts. Sum intervention only on2numeric scenes; shape predictions reused in derived hybrid summary, not regenerated.',
        'recall_protocol': 'Two fixed no-image, no-candidate, user-only prompts; all8rules/all5adapters. No answer-selection by recall score. Both variants always applied to all40shift0questions per input modality.',
        'application': 'Observation intervention crosses previous own name-only recall vs correct external rule; old/improved observations share canonical JSON formatting. Recall intervention uses original question text/image and uncorrected newly generated recall. No target facts in these inputs except explicitly separate oracle rules.',
        'limits': ['Same8development rules/5scenes, one seed; no independent confirmation.',
            'Fill instruction and row decomposition are separate interventions on different image groups; no claim one component fixes all images.',
            'Color vocabulary is a global rendering convention, not a rule definition or candidate assignment.',
            'Deterministic row summation is an external arithmetic helper; label it separately from model-only performance.',
            'New stepwise protocols cannot replace original single-step main scores.',
            'text_replay remains a visual-origin textual bridge; no pure visual training claim.']})
    snapshot = out / 'source_snapshots'
    snapshot.mkdir(exist_ok=True)
    for p in [Path(__file__)] + [p for p in files if p.suffix == '.py']:
        shutil.copy2(p, snapshot / p.name)
    validate(out)
    print('Prepared5adapters;1865new generations planned; no training.', flush=True)


def validate(out):
    m = read(out / 'manifest.json')
    assert m['script_sha256'] == sha(Path(__file__))
    for p, h in m['sources'].items():
        assert sha(Path(p)) == h, p
    for p, h in m['files'].items():
        assert sha(out / p) == h, p
    for r in rows(replay.previous.DATA / 'image_hashes.jsonl'):
        assert sha(replay.previous.DATA / r['path']) == r['source_sha256'] == r['repaired_sha256']
    for cp in m['adapters'].values():
        assert ev.checkpoint(Path(cp['path'])) == cp


def canonical(response, group, use_sum=False):
    parsed = replay.observation_json(response)
    result, failures = {}, []
    for c in 'ABCD':
        fields = (parsed or {}).get(c, {})
        if group == 0:
            result[c] = {f: fields.get(f) for f in ('shape', 'color')}
            if any(v is None for v in result[c].values()):
                failures.append(c)
        else:
            if use_sum:
                rc = fields.get('row_counts')
                valid = isinstance(rc, list) and 0 < len(rc) <= 30 and all(type(v) is int and v >= 0 for v in rc)
                value = sum(rc) if valid else None
            else:
                value = fields.get('value')
                if isinstance(value, str):
                    try:
                        value = float(value)
                    except ValueError:
                        value = None
                if type(value) not in (int, float):
                    value = None
            result[c] = {'value': value}
            if value is None:
                failures.append(c)
    return json.dumps(result, ensure_ascii=False), failures


def application_probes(out, arm, records):
    byscene = {p['source_id']: p for p in rows(out / 'extraction_probes.jsonl')}
    prior_records = rows(arm_root(arm) / arm / 'predictions.jsonl')
    prior_recall = {r['qid']: r for r in prior_records if r['condition'] == 'recall'}
    old_observations = {r['source_id']: r for r in rows(replay.OUT / 'observation_diagnostic' / arm / 'predictions.jsonl') if r['phase'] == 'extract'}
    improved = {r['source_id']: r for r in records if r.get('phase') == 'extract'}
    new_recall = {(r['condition'], r['qid']): r for r in records if r.get('phase') == 'recall'}
    assert len(improved) == 5 and len(new_recall) == 16
    named = [p for p in rows(SOURCE / 'probes.jsonl') if p.get('condition', '').startswith('named_') and p['shift'] == 0]
    knowledge = {k['id']: k for k in rows(SOURCE / 'target_knowledge.jsonl')}
    pp = []
    for rc in ('self', 'oracle'):
        for obs_kind in ('old', 'improved', 'row_sum'):
            for p in named:
                if p['modality'] != 'V' or (obs_kind == 'row_sum' and p['group'] != 1):
                    continue
                sid = p['original_source_id']
                transcript = (old_observations if obs_kind == 'old' else improved)[sid]
                obs, failures = canonical(transcript['response'], p['group'], obs_kind == 'row_sum')
                rule = prior_recall[p['qid']]['response'] if rc == 'self' else replay.previous.card(knowledge[p['qid']])
                z = copy.deepcopy(p)
                z.update(id=f'obs:{rc}:{obs_kind}:' + p['id'], condition=f'obs_{rc}_{obs_kind}',
                    phase='observation_apply', fmt='mcq', rule_context_sha256=ev.digest(rule),
                    observation_sha256=ev.digest(obs), observation_source_id=transcript['id'],
                    observation_source_response_sha256=ev.digest(transcript['response']), invalid_observation_candidates=failures)
                z.pop('images', None)
                z['messages'][-1]['content'] = 'Rule context:\n' + rule + '\n\n' + z['messages'][-1]['content'].replace('<image>', 'Observation records:\n' + obs)
                pp.append(z)
    assert len(pp) == 192
    for variant in VARIANTS:
        for p in named:
            text = new_recall[variant, p['qid']]['response']
            z = copy.deepcopy(p)
            z.update(id=f'recall_apply:{variant}:' + p['id'], condition=f'recall_apply_{variant}_{p["modality"]}',
                phase='recall_apply', fmt='mcq', recall_response_sha256=ev.digest(text))
            z['messages'][-1]['content'] = 'Your own prior rule recall (uncorrected):\n' + text + '\n\n' + z['messages'][-1]['content']
            pp.append(z)
    assert len(pp) == 352
    return pp


def run(out, arm, gpu):
    folder = out / arm
    cp = Path(read(out / 'manifest.json')['adapters'][arm]['path'])
    runner = replay.Runner(folder, replay.config(cp, {'extract': 384, 'card': 384, 'mcq': 16}), gpu)
    extract = rows(out / 'extraction_probes.jsonl')
    recall = rows(out / 'recall_probes.jsonl')
    if len(runner.rr) < 373:
        with reserve(replay.previous.LEDGER, gpu, str(folder) + ':eval', 24000) as resource:
            resource.bind(os.getpid())
            runner.load()
            runner.infer(extract)
            runner.infer(recall)
            pp = application_probes(out, arm, runner.rr)
            ev.save_rows(folder / 'application_probes.jsonl', pp)
            runner.infer(pp)
    pp = application_probes(out, arm, runner.rr)
    runner.audit(extract + recall + pp, 373)
    metrics = {}
    for c in sorted({r['condition'] for r in runner.rr if r['fmt'] == 'mcq'}):
        group = [r for r in runner.rr if r['condition'] == c]
        metrics[c] = metric(group)
        metrics[c]['by_task'] = {t: metric([r for r in group if r['task'] == t]) for t in ('direct', 'diagnosis')}
        metrics[c]['by_group'] = {str(g): metric([r for r in group if r['group'] == g]) for g in (0, 1)}
    for rc in ('self', 'oracle'):
        hybrid = [r for r in runner.rr if r['condition'] == f'obs_{rc}_improved' and r['group'] == 0]
        hybrid += [r for r in runner.rr if r['condition'] == f'obs_{rc}_row_sum']
        assert len(hybrid) == 40
        metrics[f'obs_{rc}_hybrid_sum'] = {**metric(hybrid), 'derived_summary': True,
            'shape_predictions_reused': 24, 'numeric_predictions': 16,
            'new_generations_for_this_summary': 0}
    reviews = []
    truth_by_id = {p['source_id']: p for p in extract}
    for r in runner.rr:
        if r.get('phase') != 'extract':
            continue
        p = truth_by_id[r['source_id']]
        modes = ('model', 'row_sum') if p['group'] == 1 else ('model',)
        for mode in modes:
            obs, failures = canonical(r['response'], p['group'], mode == 'row_sum')
            parsed = json.loads(obs)
            checks = []
            for c, fields in p['truth'].items():
                for f, v in fields.items():
                    checks.append({'candidate': c, 'field': f, 'expected': v, 'observed': parsed[c][f],
                                   'correct': parsed[c][f] == v})
            reviews.append({'id': r['id'], 'source_id': r['source_id'], 'group': p['group'], 'mode': mode,
                'response': r['response'], 'canonical': parsed, 'failures': failures,
                'field_checks': checks, 'all_fields_correct': all(c['correct'] for c in checks)})
    ev.save_rows(folder / 'extraction_reviews.jsonl', reviews)
    save(folder / 'report.json', {'status': 'complete', 'metrics': metrics, 'extraction_reviews': reviews,
        'recall_outputs': [r for r in runner.rr if r['fmt'] == 'card']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    out = args.out.resolve()
    if args.prepare:
        prepare(out)
    else:
        validate(out)
        with ev.run_lock(out / (args.arm + '.lock')):
            run(out, args.arm, args.gpu)
