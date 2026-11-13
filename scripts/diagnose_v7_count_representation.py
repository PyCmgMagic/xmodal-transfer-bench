#!/usr/bin/env python3
"""Posthoc count-reading and input-representation controls; never official scores."""
import copy
import json
import os
import re
from pathlib import Path

import confirm_v7_process_learning as previous
import diagnose_v7_process_images as image_diag
import explore_v7_memory_router as mem
import run_v7_evaluation as ev
from run_v7_full_experiment import ROOT, read, rows, save, sha
from v7_gpu_resources import reserve

OUT = ROOT / 'runs/v7_process_count_diagnostic_20261021'


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT / 'manifest.json').exists():
        return validate()
    knowledge = {k['id']: k for k in rows(previous.OUT / 'knowledge.jsonl')}
    probes = []
    for p in rows(previous.OUT / 'probes.jsonl'):
        if (p['condition'], p['protocol'], p['input_mod']) != ('original', 'reasoning', 'V'):
            continue
        if json.loads(knowledge[p['qid']]['fact']['quote'])['selection_topology'] != 'interval_filter':
            continue
        text = next(t for t in rows(previous.OUT / 'probes.jsonl')
                    if (t['qid'], t.get('task'), t['input_mod'], t['protocol'], t['condition']) ==
                    (p['qid'], p['task'], 'T', 'reasoning', 'original'))
        probes.append(dict(id='count_extract:' + p['id'], qid=p['qid'], task=p['task'],
            condition='count_extract', protocol='extract', input_mod='V', fmt='json', images=p['images'],
            gold=json.dumps({c: {'value': o['value']} for c, o in zip('ABCD', text['observations'])}, sort_keys=True),
            messages=[dict(role='system', content='Count the visible main objects separately in each labelled panel. Do not apply or recall a selection rule.'),
                      dict(role='user', content='<image>\nCount the main objects in each panel A, B, C, D. Count items inside each tray, excluding the tray and background tools. Return only a JSON object keyed by A, B, C, D, with each object count stored under "value". An empty tray has value 0.')]))
        z = copy.deepcopy(p)
        z.update(id='representation_hint:' + p['id'], condition='representation_hint')
        z['messages'][-1]['content'] = ('Diagnostic input convention: the value of a candidate is the number of main objects inside its labelled tray, excluding the tray and background tools. An empty tray has value 0.\n\n' + z['messages'][-1]['content'])
        probes.append(z)
    assert len(probes) == 16
    ev.save_rows(OUT / 'probes.jsonl', probes)
    sources = [Path(__file__), Path(previous.__file__), Path(mem.__file__), Path(ev.__file__),
               previous.OUT / 'manifest.json', previous.OUT / 'probes.jsonl', previous.OUT / 'knowledge.jsonl',
               image_diag.OUT / 'completion_audit.json', ROOT / 'scripts/v7_gpu_resources.py']
    manifest = dict(sources={str(p): sha(p) for p in sources}, files={'probes.jsonl': sha(OUT / 'probes.jsonl')},
        checkpoint=ev.checkpoint(image_diag.CP), generation=dict(extract=384, reasoning=768, temperature=0, thinking=False, batch=4),
        total_requests=16, scope='Posthoc representation diagnostic on eight previous numeric images; no correct rule parameters or counts supplied.',
        caveat='Explicit input convention is extra guidance. Excluded from official closed-book transfer results; does not uniquely identify causes.')
    save(OUT / 'manifest.json', manifest)
    return manifest


def validate():
    m = read(OUT / 'manifest.json')
    for p, h in m['sources'].items(): assert sha(Path(p)) == h
    for p, h in m['files'].items(): assert sha(OUT / p) == h
    assert ev.checkpoint(image_diag.CP) == m['checkpoint']
    return m


def run():
    prepare(); mem.OUT = OUT
    (OUT / 'evaluation').mkdir(exist_ok=True)
    with ev.run_lock(OUT / 'evaluation/eval.lock'):
        runner = mem.Runner(OUT / 'evaluation', '1', 'process_count_diagnostic')
        probes = rows(OUT / 'probes.jsonl')
        if len(runner.rr) < 16:
            with reserve(previous.LEDGER, '1', str(OUT), 22000) as reservation:
                reservation.bind(os.getpid()); runner.load(image_diag.CP, adapter=True)
                for protocol in ('extract', 'reasoning'):
                    runner.infer([p for p in probes if p['protocol'] == protocol], protocol)
        assert len(runner.rr) == len(runner.seen) == 16
        save(OUT / 'evaluation/state.json', dict(status='complete', n=16, ended_at=ev.now()))
    summary, review = {}, []
    pm = {p['id']: p for p in probes}
    for r in runner.rr:
        p = pm[r['id']]
        assert r['input_sha256'] == ev.digest(dict(messages=p['messages'], images=p['images']))
        response = r['response'].strip()
        if p['protocol'] == 'extract':
            try:
                parsed = json.loads(re.sub(r'^```(?:json)?\s*|\s*```$', '', response))
                values = [float(parsed[c]['value']) for c in 'ABCD']
                valid = True
            except (ValueError, TypeError, KeyError):
                values, valid = None, False
            expected = [float(json.loads(p['gold'])[c]['value']) for c in 'ABCD']
            good = values == expected
            review.append(dict(id=r['id'], expected=expected, observed=values, correct=good, raw_response=response))
        else:
            matches = re.findall(r'(?im)^\s*Answer:\s*([A-D])\s*[.!]?\s*$', response)
            valid = len(matches) == 1
            good = valid and matches[0] == p['gold']
        z = summary.setdefault(p['condition'], dict(n=0, correct=0, invalid=0, truncated=0))
        z['n'] += 1; z['correct'] += good; z['invalid'] += not valid
        z['truncated'] += r['finish_reason'] == 'length'
    for z in summary.values(): z['accuracy'] = z['correct'] / z['n']
    validate(); save(OUT / 'summary.json', summary); ev.save_rows(OUT / 'count_review.jsonl', review)
    save(OUT / 'completion_audit.json', dict(status='pass', requests=16, prompt_hashes_verified=True,
        predictions_sha256=sha(OUT / 'evaluation/predictions.jsonl'), ended_at=ev.now()))
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    run()
