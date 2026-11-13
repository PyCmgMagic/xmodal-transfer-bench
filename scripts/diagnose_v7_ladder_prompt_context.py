#!/usr/bin/env python3
"""Post-hoc prompt-context controls; never replace the fixed ladder scores."""
import copy
import json
import os
from pathlib import Path

import pilot_v7_knowledge_ladder as ladder
import explore_v7_memory_router as mem
import run_v7_evaluation as ev
from run_v7_full_experiment import read, rows, save, sha
from v7_gpu_resources import reserve

OUT = ladder.OUT / 'n1' / 'prompt_context_diagnostic'
NEUTRAL = 'Answer the question accurately. Follow the requested answer format.'


def prepare():
    ladder.validate()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'manifest.json').exists():
        manifest = read(OUT / 'manifest.json')
        assert sha(OUT / 'probes.jsonl') == manifest['probes_sha256']
        assert sha(Path(__file__)) == manifest['script_sha256']
        return
    probes = []
    for source in rows(ladder.OUT / 'n1' / 'probes.jsonl'):
        if source['condition'] not in ('unseen_fact_mcq', 'unseen_fact_free', 'new_application_T'):
            continue
        systems = [('neutral', NEUTRAL)]
        if source['fmt'] == 'mcq':
            systems.append(('learning', ladder.SYSTEM))
        for name, system in systems:
            p = copy.deepcopy(source)
            p.update(id=name + ':' + source['id'], source_id=source['id'],
                     condition=source['condition'] + ':' + name)
            assert p['messages'][0]['role'] == 'system'
            p['messages'][0]['content'] = system
            # Only the system text changes. No reference or recalled fact is supplied.
            assert p['messages'][1:] == source['messages'][1:]
            probes.append(p)
    assert len(probes) == 120 and len({p['id'] for p in probes}) == 120
    ev.save_rows(OUT / 'probes.jsonl', probes)
    save(OUT / 'manifest.json', dict(
        created_at=ev.now(), script_sha256=sha(Path(__file__)),
        source_manifest_sha256=sha(ladder.OUT / 'manifest.json'),
        probes_sha256=sha(OUT / 'probes.jsonl'),
        purpose='Post-hoc system-prompt sensitivity diagnosis, not a new formal score.',
        systems=dict(neutral=NEUTRAL, learning=ladder.SYSTEM),
        model_selection='Same fixed final epoch-10 adapter and clean base as ladder n1.',
        generation=read(ladder.OUT / 'manifest.json')['generation']))


def evaluate(arm, gpu='0'):
    prepare()
    mem.OUT = ladder.OUT
    folder = OUT / arm
    folder.mkdir(exist_ok=True)
    with ev.run_lock(folder / 'eval.lock'):
        runner = mem.Runner(folder, gpu, arm)
        pp = rows(OUT / 'probes.jsonl')
        pending = [p for p in pp if p['id'] not in runner.seen]
        if pending:
            save(folder / 'state.json', dict(status='waiting_for_gpu', pid=os.getpid()))
            with reserve(ladder.LEDGER, gpu, str(folder) + ':evaluation', 18000):
                runner.load(cp=ladder.MODELS['qwen3'] if arm == 'base' else ladder.cp(1),
                            adapter=arm != 'base')
                runner.infer([p for p in pending if p['fmt'] == 'mcq'], 'direct')
                runner.infer([p for p in pending if p['fmt'] == 'free'], 'blind_recall')
        assert runner.seen == {p['id'] for p in pp}
        save(folder / 'state.json', dict(status='complete', saved=len(runner.rr)))


def score():
    expected = {p['id']: p for p in rows(OUT / 'probes.jsonl')}
    result = {}
    for arm in ('base', 'trained'):
        rr = rows(OUT / arm / 'predictions.jsonl')
        assert len(rr) == len(expected) and {r['id'] for r in rr} == set(expected)
        for r in rr:
            p = expected[r['id']]
            assert r['input_sha256'] == ev.digest(dict(messages=p['messages'], images=[]))
            r['scored_correct'] = (ev.letter(r['response']) == p['gold'] if p['fmt'] == 'mcq'
                                   else ladder.norm(r['response']) == ladder.norm(p['gold']))
        result[arm] = {c: ladder.metric([r for r in rr if r['condition'] == c])
                       for c in sorted({p['condition'] for p in expected.values()})}
        ev.save_rows(OUT / arm / 'scored_predictions.jsonl', rr)
    save(OUT / 'summary.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('base', 'trained'), required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--score', action='store_true')
    args = parser.parse_args()
    evaluate(args.arm, args.gpu)
    if args.score:
        score()
