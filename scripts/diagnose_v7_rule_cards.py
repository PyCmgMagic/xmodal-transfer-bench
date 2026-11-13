#!/usr/bin/env python3
"""Exact trained rule-card prompt control; keep raw text for factual inspection."""
import argparse
import copy
import json
import os
import re
from pathlib import Path

import pilot_v7_grounded_families as pilot
from run_v7_full_experiment import read, rows, save, sha


def run(out, variant, gpu):
    folder = out / 'card_diagnostic' / variant
    folder.mkdir(parents=True, exist_ok=True)
    if read(folder / 'report.json').get('status') == 'complete':
        return
    training = rows(out / 'text_explicit.jsonl')
    other = {r['id']: r for r in rows(out / (('text_explicit' if variant == 'base' else variant) + '.jsonl'))}
    probes = []
    for qid in read(out / 'manifest.json')['rules']:
        r = next(r for r in training if r['qid'] == qid and r['task'] == 'rule_card')
        for reference in (False, True):
            p = {'id': r['id'] + (':image' if reference else ':blind'), 'qid': qid,
                 'condition': 'card_reference' if reference else 'card_blind',
                 'gold': r['messages'][-1]['content'], 'messages': copy.deepcopy(r['messages'][:-1]),
                 'training_target': other[r['id']]['messages'][-1]['content']}
            if reference:
                p['messages'][1]['content'] = '<image>Using this learning reference image, ' + p['messages'][1]['content']
                p['images'] = [str(pilot.DATA / 'images' / qid / 'learning_visual.png')]
            probes.append(p)
    pilot.ev.save_rows(folder / 'probes.jsonl', probes)
    raw = folder / 'predictions.jsonl'
    records, notes = pilot.ev.read_rows(raw, allow_partial_tail=True) if raw.exists() else ([], [])
    if notes:
        pilot.ev.save_rows(raw, records)
    seen = {r['id'] for r in records}
    cp = None if variant == 'base' else Path(read(out / variant / 'checkpoint_manifest.json')['path'])
    normalize = lambda text: re.sub(r'\s+', ' ', text.strip()).lower()
    with pilot.reserve(pilot.WORK, gpu, str(folder), 26000) as reservation:
        reservation.bind(os.getpid())
        os.environ.update(pilot.environment(gpu, pilot.DATA))
        import torch
        from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
        from swift.utils import seed_everything
        seed_everything(42)
        engine = TransformersEngine(str(pilot.MODELS['qwen3']), adapters=[str(cp)] if cp else None,
            torch_dtype=torch.bfloat16, device_map='cuda:0', max_batch_size=4, attn_impl='sdpa')
        engine.template.enable_thinking = False
        pending = [p for p in probes if p['id'] not in seen]
        for start in range(0, len(pending), 4):
            batch = pending[start:start + 4]
            responses = engine.infer([InferRequest(messages=copy.deepcopy(p['messages']), images=p.get('images', [])) for p in batch],
                RequestConfig(max_tokens=384, temperature=0, seed=42), use_tqdm=False)
            assert len(responses) == len(batch)
            for p, response in zip(batch, responses):
                if isinstance(response, Exception):
                    raise response
                text = response.choices[0].message.content or ''
                record = {k: p[k] for k in ('id', 'qid', 'condition', 'gold', 'training_target')}
                record.update(response=text, canonical_exact=normalize(text) == normalize(p['gold']),
                    training_target_exact=normalize(text) == normalize(p['training_target']),
                    prompt_sha256=pilot.ev.digest(p['messages']), saved_at=pilot.ev.now())
                with raw.open('a') as f:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    f.flush()
                    os.fsync(f.fileno())
                records.append(record)
            print(variant, len(records), '/', len(probes), flush=True)
    metrics = {c: {'n': sum(r['condition'] == c for r in records),
        'canonical_exact': sum(r['canonical_exact'] for r in records if r['condition'] == c),
        'training_target_exact': sum(r['training_target_exact'] for r in records if r['condition'] == c)}
        for c in ('card_blind', 'card_reference')}
    save(folder / 'report.json', {'status': 'complete', 'metrics': metrics,
         'scope': 'Exact textual recall control; nonexact outputs require factual inspection, not automatic classification as wrong facts.',
         'variant': variant, 'script_sha256': sha(Path(__file__)), 'finished_at': pilot.ev.now()})


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--variant', choices=('base', *pilot.VARIANTS), required=True)
    ap.add_argument('--gpu', default='0')
    a = ap.parse_args()
    run(a.out.resolve(), a.variant, a.gpu)
