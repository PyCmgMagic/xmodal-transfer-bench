#!/usr/bin/env python3
"""Separate training-practice and material-present controls for a finished pilot."""
import argparse
import copy
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import run_v7_evaluation as ev
from run_v7_full_experiment import ROOT, MODELS, environment, rows, read, save, sha
from v7_gpu_resources import reserve


def build(pilot, data):
    grouped = defaultdict(list)
    for row in rows(pilot / 'train.jsonl'):
        if row['id'].endswith(':explain'):
            continue
        prompt = row['messages'][1]['content']
        if prompt.startswith('For '):
            kind = 'atomic_condition'
        elif prompt.startswith('Under '):
            kind = 'record_code_recall'
        elif prompt.startswith('Inspect '):
            kind = 'practice_application'
        else:
            assert prompt.startswith('A technician submitted')
            kind = 'practice_diagnosis'
        grouped[(row['qid'], kind)].append(row)
    probes = []
    for (qid, kind), group in sorted(grouped.items()):
        selected = sorted(group, key=lambda r: hashlib.sha256(('pilot-transfer:42:' + r['id']).encode()).hexdigest())[:2]
        for row in selected:
            probes.append({'condition': kind, 'id': row['id'], 'qid': qid,
                           'gold': row['messages'][-1]['content'],
                           'messages': copy.deepcopy(row['messages'][:-1])})
    knowledge = {r['content']['id']: r['content'] for r in rows(data / 'audit/rule_knowledge.jsonl')}
    for row in rows(pilot / 'eval.jsonl'):
        messages = copy.deepcopy(row['messages'])
        messages[-1]['content'] = ('Learning material for this question:\n' +
                                  knowledge[row['qid']]['material']['learning']['text'] +
                                  '\n\n' + messages[-1]['content'])
        probes.append({'condition': 'original_material_' + row['question_type'],
                       'id': row['id'], 'qid': row['qid'], 'gold': row['gold'], 'messages': messages})
    assert len(probes) == 120
    return probes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pilot', type=Path, required=True)
    ap.add_argument('--gpu', default='0')
    args = ap.parse_args()
    pilot = args.pilot.resolve()
    out = pilot / 'transfer_diagnostic'
    out.mkdir(parents=True, exist_ok=True)
    data, work = ROOT / 'rules_repaired_single_image_v7', ROOT / 'runs/v7_full_experiment'
    checkpoint = Path(read(pilot / 'report.json')['checkpoint'])
    probes = build(pilot, data)
    ev.save_rows(out / 'probes.jsonl', probes)
    save(out / 'manifest.json', {'checkpoint': ev.checkpoint(checkpoint),
         'selection': '2 seen training MCQs per rule and task, deterministic SHA256; plus original 24 questions with learning material',
         'training_controls': 96, 'material_controls': 24, 'strict_only': True,
         'formal_evaluation': False, 'script_sha256': sha(Path(__file__)), 'created_at': ev.now()})
    with reserve(work, args.gpu, str(out), 22000) as reservation:
        reservation.bind(os.getpid())
        os.environ.update(environment(args.gpu, data))
        import torch
        from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
        from swift.utils import seed_everything
        seed_everything(42)
        engine = TransformersEngine(str(MODELS['qwen3']), adapters=[str(checkpoint)],
                   torch_dtype=torch.bfloat16, device_map='cuda:0', max_batch_size=16, attn_impl='sdpa')
        engine.template.enable_thinking = False
        results = []
        raw = out / 'predictions.jsonl'
        with raw.open('w') as f:
            for start in range(0, len(probes), 16):
                batch = probes[start:start + 16]
                responses = engine.infer([InferRequest(messages=copy.deepcopy(r['messages'])) for r in batch],
                            RequestConfig(max_tokens=32, temperature=0, seed=42), use_tqdm=False)
                assert len(responses) == len(batch)
                for row, response in zip(batch, responses):
                    if isinstance(response, Exception):
                        raise response
                    text = response.choices[0].message.content or ''
                    answer = ev.letter(text)
                    record = {k: row[k] for k in ('condition', 'id', 'qid', 'gold')}
                    record.update(response=text, parsed_answer=answer, correct=answer == row['gold'])
                    results.append(record)
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    f.flush()
                print('completed', len(results), '/', len(probes), flush=True)
        report = {}
        for condition in sorted({r['condition'] for r in results}):
            group = [r for r in results if r['condition'] == condition]
            report[condition] = {'n': len(group), 'correct': sum(r['correct'] for r in group),
                  'format_errors': sum(r['parsed_answer'] is None for r in group)}
        save(out / 'report.json', {'status': 'complete', 'metrics': report, 'finished_at': ev.now()})
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
