"""Matched explicit-reasoning diagnostic on the fixed atomic learning pilot."""
import argparse
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
OUT = ROOT / 'runs/v7_atomic_cot_diagnostic_20261021'
DATA = ROOT / 'rules_repaired_single_image_v7'
SYSTEM = ('Solve the multiple-choice question using the learned rule. Explain the rule, '
          'the observations and the calculation or decision, then select the matching option. '
          'End with FINAL: followed by exactly one uppercase option letter A, B, C, or D.')
SUFFIX = ('\n\nExplain your reasoning before answering. End with FINAL: followed by exactly '
          'one uppercase option letter A, B, C, or D.')


def prepare(arm):
    folder = OUT / arm
    folder.mkdir(parents=True, exist_ok=True)
    control = SOURCE / ('trained_matched_control_20261021' if arm == 'trained' else 'untrained_base_control_20261021')
    direct = [p for p in rows(SOURCE / 'probes.jsonl') if p['condition'].startswith('direct_')]
    recalled = rows(control / 'application_probes.jsonl')
    assert len(direct) == len(recalled) == 32
    probes = []
    for protocol, items in [('direct_cot', direct), ('self_recall_cot', recalled)]:
        for p in items:
            z = copy.deepcopy(p)
            assert z['messages'][-1]['content'].endswith(ev.SUFFIX)
            z['messages'][-1]['content'] = z['messages'][-1]['content'][:-len(ev.SUFFIX)] + SUFFIX
            z['messages'][0]['content'] = SYSTEM
            z.update(id=protocol + ':' + p['id'], protocol=protocol, modality=p['condition'][-1],
                     source_id=p['id'], arm=arm)
            probes.append(z)
    adapter = Path(read(SOURCE / 'atomic_text/checkpoint_manifest.json')['path']) if arm == 'trained' else None
    images = {p: sha(Path(p)) for z in probes for p in z.get('images', [])}
    config = dict(model=dense_identity(MODELS['qwen3']), adapter=ev.checkpoint(adapter) if adapter else None,
        max_tokens=1536, temperature=0, seed=42, batch_size=2, native_thinking=True,
        explicit_reasoning=True, answer_format='cot: terminal FINAL marker or standalone letter',
        sources={str(p): sha(p) for p in [SOURCE / 'probes.jsonl', SOURCE / 'target_knowledge.jsonl',
            control / 'application_probes.jsonl', control / 'predictions.jsonl', Path(__file__)]}, images=images,
        protocol='Fixed 32 original MCQs, unchanged images/options/labels. Same checkpoint solves both stages. '
                 'Self-recall uses its own saved, uncorrected recall from matched short-answer control. '
                 'No external correct reference. Reasoning is explicit generated explanation, not guaranteed internal computation.',
        limitations=['8 rules, 16 questions per modality, one trained checkpoint with prior task warmup.',
                     'Self-recall image condition includes self-generated text; separate from direct closed-book result.',
                     'Longer generation budget and changed instructions; not a pure effect of internal thinking.'])
    if (folder / 'manifest.json').exists():
        assert read(folder / 'manifest.json') == config
    else:
        save(folder / 'manifest.json', config)
    if (folder / 'probes.jsonl').exists():
        assert rows(folder / 'probes.jsonl') == probes
    else:
        ev.save_rows(folder / 'probes.jsonl', probes)
    return folder, probes, adapter, config


def evaluate(arm, gpu):
    folder, probes, adapter, config = prepare(arm)
    raw = folder / 'predictions.jsonl'
    records, notes = ev.read_rows(raw, allow_partial_tail=True) if raw.exists() else ([], [])
    if notes:
        ev.save_rows(raw, records)
    seen = {r['id'] for r in records}
    assert len(seen) == len(records)
    pending = [p for p in probes if p['id'] not in seen]
    if pending:
        save(folder / 'state.json', dict(status='waiting_for_gpu', gpu=gpu, pid=os.getpid(), saved=len(records)))
        with reserve(ROOT / 'runs/v7_full_experiment', gpu, str(folder) + ':cot', 18000) as reservation:
            reservation.bind(os.getpid())
            os.environ.update(environment(gpu, DATA))
            import torch
            from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
            from swift.utils import seed_everything
            torch.set_num_threads(4)
            seed_everything(42)
            save(folder / 'state.json', dict(status='loading', gpu=gpu, pid=os.getpid(), saved=len(records)))
            engine = TransformersEngine(str(MODELS['qwen3']), adapters=[str(adapter)] if adapter else [],
                torch_dtype=torch.bfloat16, device_map='cuda:0', max_batch_size=2, attn_impl='sdpa')
            engine.template.enable_thinking = True
            engine.model.eval()
            assert bool(getattr(engine.model, 'peft_config', None)) == bool(adapter)
            save(folder / 'adapter_verification.json', dict(adapter_loaded=bool(adapter),
                active_adapters=str(getattr(engine.model, 'active_adapters', None)),
                nonzero_lora_B=sum(bool(torch.count_nonzero(p)) for n, p in engine.model.named_parameters() if 'lora_B' in n)))
            for start in range(0, len(pending), 2):
                batch = pending[start:start + 2]
                replies = engine.infer([InferRequest(messages=copy.deepcopy(p['messages']), images=p.get('images', [])) for p in batch],
                    RequestConfig(max_tokens=1536, temperature=0, seed=42, return_details=True), use_tqdm=False)
                assert len(batch) == len(replies)
                for p, reply in zip(batch, replies):
                    if isinstance(reply, Exception):
                        raise reply
                    c = reply.choices[0]
                    answer = c.message.content or ''
                    reasoning = getattr(c.message, 'reasoning_content', None) or ''
                    response = (f'<think>{reasoning}</think>\n' if reasoning else '') + answer
                    pred = ev.letter(response, 'cot')
                    r = {k: p[k] for k in ('id', 'source_id', 'qid', 'protocol', 'modality', 'arm', 'gold', 'task', 'relation')}
                    r.update(response=response, predicted=pred, correct=pred == p['gold'], format_error=pred is None,
                        empty=not response.strip(), reasoning_content=reasoning, answer_content=answer,
                        finish_reason=c.finish_reason,
                        input_sha256=ev.digest(dict(messages=p['messages'], images=p.get('images', []))),
                        recall_response_sha256=p.get('recall_response_sha256'), raw_response=dataclasses.asdict(reply))
                    with raw.open('a') as f:
                        f.write(json.dumps(r, ensure_ascii=False, default=str) + '\n')
                        f.flush()
                        os.fsync(f.fileno())
                    records.append(r)
                save(folder / 'state.json', dict(status='running', gpu=gpu, pid=os.getpid(), saved=len(records), total=64, updated_at=ev.now()))
                print(arm, len(records), '/64', flush=True)
    lookup = {p['id']: p for p in probes}
    assert len(records) == len({r['id'] for r in records}) == 64
    for r in records:
        p = lookup[r['id']]
        assert r['input_sha256'] == ev.digest(dict(messages=p['messages'], images=p.get('images', [])))
        raw_message = r['raw_response']['choices'][0]['message']
        assert r['answer_content'] == (raw_message['content'] or '')
        assert r['reasoning_content'] == (raw_message.get('reasoning_content') or '')
        assert r['response'] == ((f"<think>{r['reasoning_content']}</think>\n" if r['reasoning_content'] else '') + r['answer_content'])
        assert r['gold'] == p['gold']
        assert r['predicted'] == ev.letter(r['response'], 'cot')
    assert dense_identity(MODELS['qwen3']) == config['model']
    if adapter:
        assert ev.checkpoint(adapter) == config['adapter']
    for p, h in config['images'].items():
        assert sha(Path(p)) == h
    metrics = {}
    for protocol in ('direct_cot', 'self_recall_cot'):
        for mod in ('T', 'V'):
            rr = [r for r in records if r['protocol'] == protocol and r['modality'] == mod]
            n, correct, invalid = len(rr), sum(r['correct'] for r in rr), sum(r['format_error'] for r in rr)
            metrics[protocol + '_' + mod] = dict(n=n, correct=correct, accuracy=correct/n,
                valid_accuracy=correct/(n-invalid) if n > invalid else None, format_errors=invalid,
                truncated=sum(r['finish_reason'] == 'length' for r in rr), empty=sum(r['empty'] for r in rr),
                by_task={t: dict(n=sum(r['task']==t for r in rr), correct=sum(r['correct'] for r in rr if r['task']==t)) for t in ('direct', 'diagnosis')})
    save(folder / 'report.json', dict(status='complete', metrics=metrics, audited=64, images_unchanged=True))
    save(folder / 'state.json', dict(status='complete', saved=64, ended_at=ev.now()))
    print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arm', choices=['trained', 'base'], required=True)
    ap.add_argument('--gpu', default='0')
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    with ev.run_lock(OUT / (a.arm + '.lock')):
        evaluate(a.arm, a.gpu)
