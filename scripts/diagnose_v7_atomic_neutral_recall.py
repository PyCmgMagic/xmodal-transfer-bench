#!/usr/bin/env python3
"""Post-diagnostic control: identifiers alone, with no task commands in recall."""
import argparse
import copy
import dataclasses
import json
import os
import re
import subprocess
from pathlib import Path

import run_v7_evaluation as ev
from pilot_v7_atomic_supervision import OUT, ARMS, SYSTEM, DATA, LEDGER, PARENT
from run_v7_full_experiment import ROOT, MODELS, environment, rows, read, save, sha
from v7_gpu_resources import reserve


def prepare(out):
    folder = out / 'neutral_recall'; folder.mkdir(exist_ok=True)
    if (folder / 'design.json').exists():
        m = read(folder / 'design.json'); assert m['script_sha256'] == sha(Path(__file__))
        assert m['source_probes_sha256'] == sha(out / 'probes.jsonl')
        assert m['probes_sha256'] == sha(folder / 'recall_probes.jsonl')
        return
    probes = []
    for p in rows(out / 'probes.jsonl'):
        if p['condition'] != 'direct_T' or p['task'] != 'direct':
            continue
        heading = p['messages'][-1]['content'].split('\n\nTest input:', 1)[0]
        match = re.match(r'Apply (.*?)\.(?:\s|$)', heading); assert match
        identifier = re.sub(r' category rule$', '', match[1])
        assert not any(s in identifier for s in ('KEEP', 'OMIT', 'Report', 'Apply'))
        q = 'Recall the complete stored rule for ' + identifier + '. Include all exact conditions, comparisons, values and category mappings. State the rule itself, not an example or a case answer.'
        gold = next(x['gold'] for x in rows(out / 'probes.jsonl') if x['qid'] == p['qid'] and x['condition'] == 'recall_task')
        probes.append({'id': 'neutral:' + p['qid'], 'qid': p['qid'], 'condition': 'recall_neutral',
                       'fmt': 'card', 'task': 'recall', 'relation': p['relation'], 'gold': gold,
                       'identifier': identifier, 'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': q}]})
    assert len(probes) == 8
    ev.save_rows(folder / 'recall_probes.jsonl', probes)
    save(folder / 'design.json', {'script_sha256': sha(Path(__file__)), 'source_probes_sha256': sha(out / 'probes.jsonl'),
                                 'probes_sha256': sha(folder / 'recall_probes.jsonl'), 'created_at': ev.now(),
                                 'post_diagnostic': True, 'reason': 'Actual task headings contain Apply/Report commands. Strip those commands, retain exactly the same rule identifiers and their original permutation. No ordering from gold labels supplied.',
                                 'arms': ARMS, 'external_reference': False, 'training': False,
                                 'protocol': '8 identifier-only recall +32 original MCQs using uncorrected own recall per arm. Same protocol on all arms; original direct/heading-based two-stage results retained separately.'})


def run(out, arm, gpu):
    root = out / 'neutral_recall'; folder = root / arm; folder.mkdir(exist_ok=True)
    cp = Path(read(PARENT)['path']) if arm == 'warmup' else Path(read(out / arm / 'checkpoint_manifest.json')['path'])
    config = {'adapter': ev.checkpoint(cp), 'design_sha256': sha(root / 'design.json'),
              'temperature': 0, 'seed': 42, 'batch': 4, 'max_tokens': {'card': 384, 'mcq': 16}, 'thinking': False}
    if (folder / 'manifest.json').exists():
        assert read(folder / 'manifest.json') == config
    else:
        save(folder / 'manifest.json', config)
    raw = folder / 'predictions.jsonl'; rr, notes = ev.read_rows(raw, allow_partial_tail=True) if raw.exists() else ([], [])
    if notes:
        ev.save_rows(raw, rr)
    seen = {r['id'] for r in rr}; assert len(rr) == len(seen)
    recall = rows(root / 'recall_probes.jsonl')

    def applications():
        generated = {r['qid']: r for r in rr if r['condition'] == 'recall_neutral'}
        assert len(generated) == 8
        result = []
        for p in rows(out / 'probes.jsonl'):
            if not p['condition'].startswith('direct_'):
                continue
            z = copy.deepcopy(p); z.update(id='neutral_self:' + p['id'], condition='neutral_self_' + p['condition'][-1])
            z['messages'][-1]['content'] = 'Your own prior rule recall (uncorrected):\n' + generated[p['qid']]['response'] + '\n\n' + z['messages'][-1]['content']
            z['recall_response_sha256'] = ev.digest(generated[p['qid']]['response']); result.append(z)
        return result

    if len(rr) < 40:
        with reserve(LEDGER, gpu, str(folder), 23000) as res:
            res.bind(os.getpid()); os.environ.update(environment(gpu, DATA))
            import torch
            from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
            from swift.utils import seed_everything
            seed_everything(42)
            engine = TransformersEngine(str(MODELS['qwen3']), adapters=[str(cp)], torch_dtype=torch.bfloat16, device_map='cuda:0', max_batch_size=4, attn_impl='sdpa')
            engine.template.enable_thinking = False
            for phase in ('recall', 'apply'):
                pp = recall if phase == 'recall' else applications()
                if phase == 'apply':
                    ev.save_rows(folder / 'application_probes.jsonl', pp)
                pending = [p for p in pp if p['id'] not in seen]
                for start in range(0, len(pending), 4):
                    batch = pending[start:start + 4]
                    outputs = engine.infer([InferRequest(messages=copy.deepcopy(p['messages']), images=p.get('images', [])) for p in batch],
                                           RequestConfig(max_tokens=384 if phase == 'recall' else 16, temperature=0, seed=42, return_details=True), use_tqdm=False)
                    assert len(batch) == len(outputs)
                    for p, reply in zip(batch, outputs):
                        if isinstance(reply, Exception):
                            raise reply
                        c = reply.choices[0]; text = c.message.content or ''; pred = ev.letter(text) if phase == 'apply' else None
                        r = {k: p[k] for k in ('id', 'qid', 'condition', 'gold', 'task', 'relation')}
                        if phase == 'apply':
                            r['recall_response_sha256'] = p['recall_response_sha256']
                        r.update(response=text, predicted=pred, correct=pred == p['gold'] if phase == 'apply' else text.strip() == p['gold'],
                                 format_error=phase == 'apply' and pred is None, finish_reason=c.finish_reason,
                                 input_sha256=ev.digest({'messages': p['messages'], 'images': p.get('images', [])}), raw_response=dataclasses.asdict(reply))
                        rr.append(r); seen.add(r['id'])
                        with raw.open('a') as f:
                            f.write(json.dumps(r, ensure_ascii=False, default=str) + '\n'); f.flush(); os.fsync(f.fileno())
                    print(arm, phase, len(rr), '/40', flush=True)
    assert len(rr) == len(seen) == 40
    metrics = {}
    for condition in ('recall_neutral', 'neutral_self_T', 'neutral_self_V'):
        group = [r for r in rr if r['condition'] == condition]
        metrics[condition] = {'n': len(group), 'correct': sum(r['correct'] for r in group),
                              'format_errors': sum(r['format_error'] for r in group), 'truncated': sum(r['finish_reason'] == 'length' for r in group)}
    assert ev.checkpoint(cp) == config['adapter']
    save(folder / 'report.json', {'status': 'complete', 'metrics': metrics, 'post_diagnostic': True})


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, default=OUT); ap.add_argument('--gpu', default='0')
    ap.add_argument('--arm', choices=ARMS); ap.add_argument('--prepare', action='store_true'); ap.add_argument('--queue', action='store_true')
    a = ap.parse_args(); out = a.out.resolve()
    if a.prepare:
        prepare(out)
    elif a.queue:
        for arm in ARMS:
            subprocess.run([str(ROOT / '.venv/bin/python'), '-u', str(Path(__file__).resolve()), '--out', str(out), '--arm', arm, '--gpu', a.gpu], check=True, cwd=ROOT)
    else:
        with ev.run_lock(out / 'neutral_recall' / (a.arm + '.lock')):
            run(out, a.arm, a.gpu)
