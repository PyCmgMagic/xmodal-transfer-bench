#!/usr/bin/env python3
"""Paired system-only intervention; no new target knowledge in prompts."""
import argparse
import copy
import os
from pathlib import Path

import explore_v7_readout_protocol as trial
import explore_v7_visual_replay as replay
import run_v7_evaluation as ev
from pilot_v7_atomic_supervision import SYSTEM
from run_v7_full_experiment import rows, read, save, sha
from summarize_v7_counterfactual_names import metric
from v7_gpu_resources import reserve


def prepare():
    out = trial.OUT / 'system_alignment'
    out.mkdir(exist_ok=True)
    if (out / 'manifest.json').exists():
        validate()
        return
    pp = []
    for p in rows(trial.OUT / 'recall_probes.jsonl'):
        if p['condition'] != 'plain':
            continue
        z = copy.deepcopy(p)
        z.update(id='aligned:' + p['id'], condition='aligned_plain')
        z['messages'].insert(0, {'role': 'system', 'content': SYSTEM})
        pp.append(z)
    for style in ('strict', 'aligned'):
        for p in rows(trial.SOURCE / 'probes.jsonl'):
            if not p['condition'].startswith('named_') or p['shift'] != 0:
                continue
            z = copy.deepcopy(p)
            z.update(id=f'{style}:' + p['id'], condition=f'{style}_{p["modality"]}', phase='direct', fmt='mcq')
            assert z['messages'][0] == {'role': 'system', 'content': ev.SYSTEM}
            if style == 'aligned':
                z['messages'][0]['content'] = SYSTEM + '\n\n' + ev.SYSTEM
            pp.append(z)
    assert len(pp) == 168
    ev.save_rows(out / 'probes.jsonl', pp)
    save(out / 'manifest.json', {'created_at': ev.now(), 'script_sha256': sha(Path(__file__)),
        'main_manifest_sha256': sha(trial.OUT / 'manifest.json'), 'probes_sha256': sha(out / 'probes.jsonl'),
        'arms': trial.ARMS, 'new_generations_per_arm': 168, 'new_generations_total': 840,
        'reason': 'User-only recall failed in replay model despite earlier training-system recall8/8. Original MCQ evaluation also changes the system. Test exact same user input, changing only system prefix.',
        'timing': 'After inspecting initial prompt-intervention responses, before new system-alignment responses; development follow-up, not preregistration.',
        'recall': 'Same plain query, add exact training system; compare with saved newly generated user-only plain query. All8rules retained.',
        'direct': 'New strict and aligned strict generation on all40shift0questions per modality. No recall or external materials supplied. Only system differs; aligned retains training system then appends identical strict answer instruction.',
        'limits': ['System prefix contains generic historical task conventions; no target rule values, definitions or choices.',
            'Generic three-category conventions in training prefix are irrelevant to these OR/interval rules.',
            'Same development rules; no independent confirmation or model retraining.',
            'Short-answer truncation/format errors retained; no retry to force valid output.']})
    validate()


def validate():
    out = trial.OUT / 'system_alignment'
    m = read(out / 'manifest.json')
    assert m['script_sha256'] == sha(Path(__file__))
    assert m['main_manifest_sha256'] == sha(trial.OUT / 'manifest.json')
    assert m['probes_sha256'] == sha(out / 'probes.jsonl')


def run(arm, gpu):
    trial.validate(trial.OUT)
    validate()
    out = trial.OUT / 'system_alignment'
    folder = out / arm
    cp = Path(read(trial.OUT / 'manifest.json')['adapters'][arm]['path'])
    runner = replay.Runner(folder, replay.config(cp, {'card': 384, 'mcq': 16}), gpu)
    pp = rows(out / 'probes.jsonl')
    if len(runner.rr) < 168:
        with reserve(replay.previous.LEDGER, gpu, str(folder) + ':eval', 24000) as resource:
            resource.bind(os.getpid())
            runner.load()
            runner.infer([p for p in pp if p['fmt'] == 'card'])
            runner.infer([p for p in pp if p['fmt'] == 'mcq'])
    runner.audit(pp, 168)
    metrics = {c: metric([r for r in runner.rr if r['condition'] == c]) for c in ('strict_T', 'strict_V', 'aligned_T', 'aligned_V')}
    save(folder / 'report.json', {'status': 'complete', 'metrics': metrics,
        'recall_outputs': [r for r in runner.rr if r['fmt'] == 'card']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--arm', choices=trial.ARMS)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        with ev.run_lock(trial.OUT / 'system_alignment' / (args.arm + '.lock')):
            run(args.arm, args.gpu)
