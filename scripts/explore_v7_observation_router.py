#!/usr/bin/env python3
"""Additional fixed development method: uncorrected visual transcription.

No image or dataset is edited. No textual ground-truth observation is supplied
to inference. This is a distinct, longer inference protocol, with matched base.
"""
import argparse
import copy
import os
from pathlib import Path

import explore_v7_memory_router as mem
import run_v7_evaluation as ev
from run_v7_full_experiment import ROOT, rows, read, save, sha
from v7_gpu_resources import reserve

OUT = mem.OUT / 'observation_route'
ARMS = ('base', 'text_memory', 'vision_memory', 'vision_bridge', 'oracle')
OBS_SYSTEM = '''Transcribe visible observations only. Do not apply a named rule, classify a candidate, audit a report, or select an answer.
Describe every labeled test group separately. Preserve all candidate IDs and diagram labels. Count the visible objects of each type separately in each group; include zero only when its absence is clear. State visible object types, colors, main-body materials if identifiable, shapes and special markings.
For a grid, report its rows, columns, occupied cells and marked cells separately; distinguish a bright-blue top-center rectangular inspection seal from object color and packing tape.
For a signal sequence, report the initial state and each signal in exact left-to-right order; distinguish square, diamond, triangle and striped signals. For an object row, list every item in exact order. For matching diagrams, list every left item ID/type and every right slot ID/shape. Do not omit duplicate objects or combine separate candidate groups.
Only report features actually visible in the test image; say uncertain if a feature cannot be read. No reference learning image is provided. Return concise factual observations, not instructions for how to observe.'''


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'manifest.json').exists():
        validate(); return
    mem.validate()
    probes = []
    for p in rows(mem.OUT / 'test_probes.jsonl'):
        if p['input_mod'] != 'V':
            continue
        q = dict(id='observe:' + p['id'], source_id=p['source_id'], qid=p['qid'], input_mod='V',
                 task=p['task'], relation=p['relation'],
                 messages=[{'role': 'system', 'content': OBS_SYSTEM},
                           {'role': 'user', 'content': '<image>Transcribe the labeled test observations in this image.'}],
                 images=p['images'])
        probes.append(q)
    assert len(probes) == 36
    ev.save_rows(OUT / 'observation_probes.jsonl', probes)
    sources = [Path(__file__), Path(mem.__file__), mem.OUT / 'manifest.json', mem.OUT / 'test_probes.jsonl']
    save(OUT / 'manifest.json', {'created_at': ev.now(), 'sources': {str(p): sha(p) for p in sources},
        'files': {'observation_probes.jsonl': sha(OUT / 'observation_probes.jsonl')}, 'arms': ARMS,
        'decision_before_generation': 'Run all five prespecified contexts, including failures. Compare all with matched base and raw-image route; no best-case-only claim.',
        'flow': 'Original test image -> frozen base rule-free transcription -> frozen base worksheet with name-only memory + unchanged question heading -> isolated original options.',
        'constraints': 'Transcript is passed verbatim without human correction. No original text observation, gold label, or learning reference enters closed-book generation. Oracle alone supplies correct learning reference.',
        'limits': 'Same18 development rules, single seed, added inference stages; not an independent confirmation or single-step score.',
        'new_generations': 396, 'transcript_budget': 1024, 'worksheet_budget': 1024, 'choice_budget': 32})


def validate():
    mem.validate()
    m = read(OUT / 'manifest.json')
    for p, h in m['sources'].items():
        assert sha(Path(p)) == h, p
    for p, h in m['files'].items():
        assert sha(OUT / p) == h, p


def worksheets(arm, transcripts):
    pp = [p for p in mem.solve_probes(arm) if p['input_mod'] == 'V']
    for p in pp:
        obs = transcripts[p['source_id']]
        assert p['messages'][-1]['content'].count('<image>') == 1
        p['messages'][-1]['content'] = p['messages'][-1]['content'].replace('<image>',
            'Uncorrected test-image transcription by the base model (may contain perception errors):\n' + obs['response'])
        p['transcript_sha256'] = ev.digest(obs['response'])
        p['transcript_source_id'] = obs['id']
        p.pop('images', None)
    return pp


def choices(run):
    drafts = {r['source_id']: r for r in run.rr if r['stage'] == 'worksheet'}
    assert len(drafts) == 36
    pp = []
    for p in rows(mem.OUT / 'test_probes.jsonl'):
        if p['input_mod'] != 'V':
            continue
        draft = drafts[p['source_id']]
        opts = p['messages'][-1]['content'].split('\n\nOptions:\n')[1].split(ev.SUFFIX)[0]
        z = {k: v for k, v in p.items() if k not in ('messages', 'images')}
        z.update(id='choice:' + p['id'], worksheet_sha256=ev.digest(draft['response']),
                 memory_sha256=draft['memory_sha256'], transcript_sha256=draft['transcript_sha256'],
                 memory_arm=draft['memory_arm'], context_kind=draft['context_kind'])
        z['messages'] = [{'role': 'system', 'content': ev.SYSTEM}, {'role': 'user', 'content':
            'Use this uncorrected worksheet to select the OPTION whose complete result and report decision match it. '
            'Candidate IDs and option letters are different.\n\nWorksheet:\n' + draft['response'] +
            '\n\nOptions:\n' + opts + ev.SUFFIX}]
        pp.append(z)
    return pp


def run(gpu):
    validate()
    with ev.run_lock(OUT / 'run.lock'):
        with reserve(mem.LEDGER, gpu, str(OUT) + ':eval', 24000) as res:
            res.bind(os.getpid())
            observe = mem.Runner(OUT / 'observations', gpu, 'base')
            observe.load()
            # Reuse the generation and raw-output recorder with a 1024-token stage.
            observe.infer(rows(OUT / 'observation_probes.jsonl'), 'worksheet')
            assert len(observe.rr) == 36
            transcripts = {r['source_id']: r for r in observe.rr}
            save(observe.folder / 'state.json', {'status': 'complete', 'saved': 36, 'role': 'rule_free_test_observations'})
            for arm in ARMS:
                worker = mem.Runner(OUT / arm, gpu, 'base')
                worker.engine = observe.engine
                save(worker.folder / 'eval_manifest.json', read(observe.folder / 'eval_manifest.json'))
                pp = worksheets(arm, transcripts)
                ev.save_rows(worker.folder / 'worksheet_probes.jsonl', pp)
                worker.infer(pp, 'worksheet')
                cc = choices(worker)
                ev.save_rows(worker.folder / 'choice_probes.jsonl', cc)
                worker.infer(cc, 'choice')
                assert len(worker.rr) == 72
                save(worker.folder / 'state.json', {'status': 'complete', 'saved': 72})
                summarize()
    audit()


def summarize():
    summary = {}
    lines = ['# 原图先转录观察，再闭卷调用规则', '',
             '所有转录和回忆均为模型未经修正的原答；没有给正确文字观察。仅oracle给正确学习规则。与原图直接工作记录是不同推理流程。', '',
             '| 规则来源 | 图片正确 | direct | diagnosis | 格式错 |', '|---|---:|---:|---:|---:|']
    for arm in ARMS:
        path = OUT / arm / 'predictions.jsonl'
        rr = rows(path) if path.exists() else []
        ss = [r for r in rr if r['stage'] == 'choice']
        stats = mem.metric(ss)
        for key in ('task', 'relation', 'evaluation_group', 'qid'):
            stats['by_' + key] = {v: mem.metric([r for r in ss if r[key] == v]) for v in sorted({r[key] for r in ss})}
        stats['worksheet_truncated'] = sum(r['stage'] == 'worksheet' and r['finish_reason'] == 'length' for r in rr)
        summary[arm] = stats
        d = stats['by_task'].get('direct', {'correct': 0, 'n': 0})
        a = stats['by_task'].get('diagnosis', {'correct': 0, 'n': 0})
        lines.append(f'| {arm} | {stats["correct"]}/{stats["n"]} | {d["correct"]}/{d["n"]} | {a["correct"]}/{a["n"]} | {stats["format_errors"]} |')
    save(OUT / 'summary.json', summary)
    (OUT / 'report_zh.md').write_text('\n'.join(lines) + '\n')


def audit():
    validate()
    observations = mem.Runner(OUT / 'observations', '0', 'base').rr
    probes = rows(OUT / 'observation_probes.jsonl')
    assert len(observations) == len(probes) == 36
    obsmap = {p['id']: p for p in probes}
    for r in observations:
        p = obsmap[r['id']]
        assert r['input_sha256'] == ev.digest({'messages': p['messages'], 'images': p['images']})
        assert r['response'] == r['raw_response']['choices'][0]['message']['content']
    transcripts = {r['source_id']: r for r in observations}
    for arm in ARMS:
        worker = mem.Runner(OUT / arm, '0', 'base')
        pp = worksheets(arm, transcripts) + choices(worker)
        lookup = {p['id']: p for p in pp}
        assert len(worker.rr) == 72 and {r['id'] for r in worker.rr} == set(lookup)
        for r in worker.rr:
            p = lookup[r['id']]
            assert not p.get('images') and '<image>' not in str(p['messages'])
            assert r['input_sha256'] == ev.digest({'messages': p['messages'], 'images': []})
            assert r['response'] == r['raw_response']['choices'][0]['message']['content']
            if r['stage'] == 'choice':
                assert r['predicted'] == ev.letter(r['response']) and r['correct'] == (r['predicted'] == p['gold'])
    mem.validate(images=True)
    save(OUT / 'completion_audit.json', {'status': 'pass', 'new_generations': 396, 'all_contexts_complete': True,
         'uncorrected_transcripts_and_memories': True, 'original_pngs_unchanged': 3000,
         'transcripts_truncated': sum(r['finish_reason'] == 'length' for r in observations)})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=('prepare', 'run', 'summarize', 'audit'), required=True)
    ap.add_argument('--gpu', default='0')
    a = ap.parse_args()
    if a.stage == 'prepare': prepare()
    elif a.stage == 'run': run(a.gpu)
    elif a.stage == 'summarize': summarize()
    elif a.stage == 'audit': audit()
