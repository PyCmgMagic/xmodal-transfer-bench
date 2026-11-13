"""Matched untrained-base control for the saved atomic self-recall pilot."""
import copy
import dataclasses
import json
import os
from pathlib import Path

import run_v7_evaluation as ev
from run_v7_full_experiment import ROOT, MODELS, environment, rows, read, save, sha
from run_v7_fullparam_ep10 import dense_identity
from v7_gpu_resources import reserve

SOURCE = ROOT / 'runs/v7_atomic_supervision_20261015'
OUT = SOURCE / 'untrained_base_control_20261021'
DATA = ROOT / 'rules_repaired_single_image_v7'


def main():
    OUT.mkdir(exist_ok=True)
    direct = [p for p in rows(SOURCE / 'probes.jsonl') if p['condition'].startswith('direct_')]
    recall = rows(SOURCE / 'neutral_recall/recall_probes.jsonl')
    assert len(direct) == 32 and len(recall) == 8
    identity = dense_identity(MODELS['qwen3'])
    config = dict(model=identity, adapters=[], training=False, temperature=0, seed=42,
                  batch_size=4, max_tokens=dict(recall=384, mcq=16), thinking=False,
                  sources={str(p): sha(p) for p in [SOURCE / 'probes.jsonl',
                      SOURCE / 'neutral_recall/recall_probes.jsonl', Path(__file__)]},
                  protocol='Same saved direct prompts and name-only self-recall protocol; no adapters or external material.')
    if (OUT / 'manifest.json').exists():
        assert read(OUT / 'manifest.json') == config
    else:
        save(OUT / 'manifest.json', config)
    raw = OUT / 'predictions.jsonl'
    records, notes = ev.read_rows(raw, allow_partial_tail=True) if raw.exists() else ([], [])
    if notes:
        ev.save_rows(raw, records)
    seen = {r['id'] for r in records}
    assert len(records) == len(seen)

    def application_inputs():
        memories = {r['qid']: r for r in records if r['condition'] == 'recall_neutral'}
        assert len(memories) == 8
        result = []
        for p in direct:
            z = copy.deepcopy(p)
            z.update(id='neutral_self:' + p['id'], condition='neutral_self_' + p['condition'][-1])
            text = memories[p['qid']]['response']
            z['messages'][-1]['content'] = 'Your own prior rule recall (uncorrected):\n' + text + '\n\n' + z['messages'][-1]['content']
            z['recall_response_sha256'] = ev.digest(text)
            result.append(z)
        return result

    if len(records) < 72:
        save(OUT / 'state.json', dict(status='loading', pid=os.getpid()))
        with reserve(ROOT / 'runs/v7_full_experiment', '0', str(OUT), 23000) as reservation:
            reservation.bind(os.getpid())
            os.environ.update(environment('0', DATA))
            import torch
            from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
            from swift.utils import seed_everything
            torch.set_num_threads(4)
            seed_everything(42)
            engine = TransformersEngine(str(MODELS['qwen3']), adapters=[], torch_dtype=torch.bfloat16,
                                        device_map='cuda:0', max_batch_size=4, attn_impl='sdpa')
            engine.template.enable_thinking = False
            assert not getattr(engine.model, 'peft_config', None), 'Unexpected adapter on baseline'
            for phase in ('direct', 'recall', 'apply'):
                probes = direct if phase == 'direct' else recall if phase == 'recall' else application_inputs()
                if phase == 'apply':
                    ev.save_rows(OUT / 'application_probes.jsonl', probes)
                pending = [p for p in probes if p['id'] not in seen]
                for start in range(0, len(pending), 4):
                    batch = pending[start:start + 4]
                    replies = engine.infer([InferRequest(messages=copy.deepcopy(p['messages']), images=p.get('images', [])) for p in batch],
                        RequestConfig(max_tokens=384 if phase == 'recall' else 16, temperature=0, seed=42, return_details=True), use_tqdm=False)
                    assert len(replies) == len(batch)
                    for p, reply in zip(batch, replies):
                        if isinstance(reply, Exception):
                            raise reply
                        choice = reply.choices[0]
                        answer = choice.message.content or ''
                        pred = ev.letter(answer) if phase != 'recall' else None
                        r = {k: p[k] for k in ('id', 'qid', 'condition', 'gold', 'task', 'relation')}
                        r.update(response=answer, predicted=pred, correct=pred == p['gold'] if phase != 'recall' else answer.strip() == p['gold'],
                                 format_error=phase != 'recall' and pred is None, finish_reason=choice.finish_reason,
                                 input_sha256=ev.digest(dict(messages=p['messages'], images=p.get('images', []))), raw_response=dataclasses.asdict(reply))
                        if phase == 'apply':
                            r['recall_response_sha256'] = p['recall_response_sha256']
                        with raw.open('a') as f:
                            f.write(json.dumps(r, ensure_ascii=False, default=str) + '\n')
                            f.flush()
                            os.fsync(f.fileno())
                        records.append(r)
                        seen.add(r['id'])
                    save(OUT / 'state.json', dict(status='running', phase=phase, saved=len(records), total=72, pid=os.getpid()))
                    print(phase, len(records), '/72', flush=True)
    inputs = direct + recall + application_inputs()
    lookup = {p['id']: p for p in inputs}
    assert len(records) == len(seen) == len(lookup) == 72
    for r in records:
        p = lookup[r['id']]
        assert r['input_sha256'] == ev.digest(dict(messages=p['messages'], images=p.get('images', [])))
        assert r['response'] == (r['raw_response']['choices'][0]['message']['content'] or '')
    metrics = {}
    for condition in ('direct_T', 'direct_V', 'neutral_self_T', 'neutral_self_V', 'recall_neutral'):
        rr = [r for r in records if r['condition'] == condition]
        metrics[condition] = dict(n=len(rr), correct=sum(r['correct'] for r in rr), format_errors=sum(r['format_error'] for r in rr),
                                  truncated=sum(r['finish_reason'] == 'length' for r in rr))
    assert dense_identity(MODELS['qwen3']) == identity
    save(OUT / 'report.json', dict(status='complete', metrics=metrics, model_untrained=True, adapters=[],
         note='Recall strict string matching is auxiliary only. Same questions and generation protocol as historical trained pilot.'))
    save(OUT / 'state.json', dict(status='complete', saved=72, ended_at=ev.now()))
    print(json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    with ev.run_lock(OUT.parent / 'untrained_base_control_20261021.lock'):
        main()
