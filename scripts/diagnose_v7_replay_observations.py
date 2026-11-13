#!/usr/bin/env python3
"""Repeat the observation intervention after training; all intermediates retained."""
import argparse
import copy
import json
import os
from pathlib import Path

import explore_v7_visual_replay as trial
import run_v7_evaluation as ev
from run_v7_full_experiment import rows, read, save, sha
from summarize_v7_counterfactual_names import metric
from v7_gpu_resources import reserve


def run(arm, gpu):
    out = trial.OUT
    trial.validate(out)
    folder = out / 'observation_diagnostic' / arm
    source = out / arm / 'predictions.jsonl'
    assert read(out / arm / 'eval_state.json')['status'] == 'complete'
    cp = Path(read(out / arm / 'checkpoint_manifest.json')['path'])
    folder.mkdir(parents=True, exist_ok=True)
    m = {'script_sha256': sha(Path(__file__)), 'trained_predictions_sha256': sha(source),
         'new_generations': 165, 'native_cot': False,
         'reason': 'Post-training check of self rule recall plus self observation transcription; correct rule/observation interventions remain separate.'}
    if (folder / 'supplement_manifest.json').exists():
        assert read(folder / 'supplement_manifest.json') == m
    else:
        save(folder / 'supplement_manifest.json', m)
    runner = trial.Runner(folder, trial.config(cp, {'extract': 384, 'mcq': 16}), gpu)
    extract = [dict(p, fmt='extract', condition='observe_only') for p in rows(out / 'extraction_probes.jsonl')]
    scenes = {p['source_id']: p for p in extract}
    originals = [p for p in rows(trial.SOURCE / 'probes.jsonl') if p['condition'] == 'named_V' and p['shift'] == 0]
    recalls = {r['qid']: r for r in rows(source) if r['condition'] == 'recall'}
    knowledge = {k['id']: k for k in rows(trial.SOURCE / 'target_knowledge.jsonl')}

    def applications():
        observations = {r['source_id']: r['response'] for r in runner.rr if r.get('phase') == 'extract'}
        assert len(observations) == 5
        pp = []
        for rc in ('self', 'oracle'):
            for ob in ('own', 'truth'):
                for p in originals:
                    obs = observations[p['original_source_id']] if ob == 'own' else json.dumps(scenes[p['original_source_id']]['truth'], ensure_ascii=False)
                    rule = recalls[p['qid']]['response'] if rc == 'self' else trial.previous.card(knowledge[p['qid']])
                    z = copy.deepcopy(p)
                    z.update(id=f'{rc}_{ob}:' + p['id'], condition=rc + '_' + ob, phase='apply',
                             rule_context_sha256=ev.digest(rule), observation_sha256=ev.digest(obs))
                    z.pop('images', None)
                    z['messages'][-1]['content'] = 'Rule context:\n' + rule + '\n\n' + z['messages'][-1]['content'].replace('<image>', 'Observation records:\n' + obs)
                    pp.append(z)
        assert len(pp) == 160
        return pp

    if len(runner.rr) < 165:
        with reserve(trial.previous.LEDGER, gpu, str(folder) + ':eval', 24000) as resource:
            resource.bind(os.getpid())
            runner.load()
            runner.infer(extract)
            pp = applications()
            ev.save_rows(folder / 'application_probes.jsonl', pp)
            runner.infer(pp)
    runner.audit(extract + applications(), 165)
    extracted = []
    for p in extract:
        r = next(r for r in runner.rr if r['id'] == p['id'])
        parsed = trial.observation_json(r['response'])
        checks = []
        for c, fields in p['truth'].items():
            for f, v in fields.items():
                observed = parsed.get(c, {}).get(f) if parsed else None
                if isinstance(v, int) and isinstance(observed, str):
                    try:
                        observed = float(observed)
                    except ValueError:
                        pass
                checks.append({'candidate': c, 'field': f, 'expected': v, 'observed': observed, 'correct': observed == v})
        extracted.append({'id': p['id'], 'group': p['group'], 'raw_response': r['response'], 'parsed': parsed,
            'json_valid': parsed is not None, 'scene_correct': all(c['correct'] for c in checks),
            'field_checks': checks, 'finish_reason': r['finish_reason']})
    ev.save_rows(folder / 'extraction_reviews.jsonl', extracted)
    save(folder / 'report.json', {'status': 'complete',
        'metrics': {c: metric([r for r in runner.rr if r['condition'] == c]) for c in ('self_own', 'self_truth', 'oracle_own', 'oracle_truth')},
        'extraction': extracted, 'note': 'Self+own uses two model-generated intermediates with no external correct knowledge. Truth/oracle interventions are separate capability controls.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=trial.ARMS, required=True)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    with ev.run_lock(trial.OUT / (args.arm + '_post_training_observations.lock')):
        run(args.arm, args.gpu)
