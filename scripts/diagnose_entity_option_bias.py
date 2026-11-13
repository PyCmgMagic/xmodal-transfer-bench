#!/usr/bin/env python3
"""Small, separately reported prompt and backend diagnostic; never rescore E23."""
import argparse
import copy
from collections import Counter
import json
import os
from pathlib import Path
import sys

from answer_readout import options_from_question
from answer_readout_entity_v4 import parse_answer
from entityqa_common import rows, save, write_rows, digest, sha, now, eval_payload, COT_SUFFIX
from prepare_entity_atomic_pilot import generic
from run_entityqa_lora import idle


def prepare(work, output):
    data = work / 'evaluation_data'
    rotations = [r for r in rows(data / 'shared/eval_text.jsonl') if '-rotation-' in r['id']]
    # First four entities in the pre-existing rotation file; no score-based selection.
    ordered = list(dict.fromkeys(r['qid'] for r in rotations))[:4]
    selected = [r for r in rotations if r['qid'] in ordered]
    assert len(selected) == 16
    learning = {r['qid']: r for r in rows(work / 'data/learning_inputs.jsonl')}
    labels = {r['id']: r for r in rows(data / 'audit/eval_labels.jsonl')}
    own = {r['qid']: r for r in rows(work / 'diagnostics/atomic_replay/predictions.jsonl') if r['kind'] == 'neutral_claim_recall'}
    tasks = []
    for row in selected:
        current = eval_payload(row, data)
        question = current['messages'][-1]['content']
        assert question.endswith(COT_SUFFIX)
        plain = question[:-len(COT_SUFFIX)]
        for variant in ['original', 'minimal', 'reference_fact', 'own_recall']:
            payload = copy.deepcopy(current)
            if variant == 'minimal':
                payload = dict(messages=[dict(role='user', content=plain +
                    '\n\nCompare the options with what you know. Explain briefly and end with the final option letter in \\boxed{}.')])
            elif variant == 'reference_fact':
                payload['messages'][-1]['content'] = (
                    'Reference fact for this diagnostic (treat it as given):\n' + learning[row['qid']]['claim'] +
                    '\n\nQuestion:\n' + question)
            elif variant == 'own_recall':
                payload['messages'][-1]['content'] = (
                    'Your earlier closed-book recall, before seeing these answer options (uncorrected):\n' + own[row['qid']]['response'] +
                    '\n\nUse your recalled information to compare the options.\nQuestion:\n' + question)
            tasks.append(dict(id=row['id'] + '-' + variant, source_id=row['id'], qid=row['qid'], variant=variant,
                gold=labels[row['id']]['answer'], payload=payload))
    # Self-contained generic choices, rewritten with the exact formal system/suffix.
    system = tasks[0]['payload']['messages'][0]
    for i in range(8):
        g = generic(i, 'dev')
        body = g['messages'][0]['content'].split('\n\nReason briefly.')[0]
        tasks.append(dict(id=g['id'] + '-original-style', variant='generic_original_style', gold=g['answer'],
            payload=dict(messages=[copy.deepcopy(system), dict(role='user', content=body + COT_SUFFIX)])))
    assert len(tasks) == 72
    for variant in ['original', 'minimal', 'reference_fact', 'own_recall']:
        assert Counter(r['gold'] for r in tasks if r['variant'] == variant) == dict.fromkeys('ABCD', 4)
    write_rows(output / 'tasks.jsonl', tasks)
    save(output / 'plan.json', dict(created_at=now(), purpose='Explain option-A bias without changing historical scores',
        n_entities=4, selection='First four entities in the existing fixed rotation file, not selected by score',
        qids=ordered, source_work=str(work), tasks_per_model=72, framework='ms-swift original unmerged adapters',
        models=['base', 'atomic_replay'], max_tokens=256, temperature=0, thinking='brief prompted CoT',
        stop_sequences='none added; natural model stop or diagnostic length cap',
        variants=['original', 'minimal', 'reference_fact', 'own_recall', 'generic_original_style'],
        limitations=['Text diagnostic only, 4 entities, exploratory; no transfer claim.',
                     'Minimal variant changes a prompt bundle; cannot identify a single phrase causally.',
                     'Reference fact exposes the target knowledge, only an option-matching positive control.',
                     'Own-recall variant uses unchanged earlier model replies, never test gold correction.',
                     'Diagnostic 256-token cap differs from formal 65536-token cap; truncations are explicit.'],
        tasks_sha256=sha(output / 'tasks.jsonl')))


def run(args):
    import torch
    from swift import TransformersEngine, InferRequest, RequestConfig
    from entityqa_common import render
    from transformers import AutoProcessor
    torch.set_num_threads(4)
    work, output = args.work.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not (output / 'tasks.jsonl').exists():
        prepare(work, output)
    tasks = rows(output / 'tasks.jsonl')
    root = Path(__file__).resolve().parents[1]
    model = root / 'model_cache/Qwen3-VL-4B-Instruct'
    for arm in [args.arm]:
        path = output / arm / 'predictions.jsonl'
        existing = rows(path) if path.exists() else []
        seen = {r['id'] for r in existing}
        pending = [r for r in tasks if r['id'] not in seen]
        if pending:
            idle('0')
            adapter = work / 'atomic_replay/train/TEXT/checkpoint-237'
            engine = TransformersEngine(str(model), model_type='qwen3_vl', adapters=[str(adapter)] if arm != 'base' else None,
                torch_dtype=torch.bfloat16, attn_impl='sdpa', device_map='cuda:0', max_batch_size=8)
            engine.template.enable_thinking = False
            processor = AutoProcessor.from_pretrained(str(model), local_files_only=True)
            previous = {(r['id'], r['modality']): r for r in rows(work / f'evaluation/{arm}/predictions.jsonl')}
            for start in range(0, len(pending), 8):
                batch = pending[start:start + 8]
                requests = [InferRequest(messages=r['payload']['messages']) for r in batch]
                answers = engine.infer(requests, RequestConfig(max_tokens=256, temperature=0), use_tqdm=False)
                for r, answer in zip(batch, answers):
                    if isinstance(answer, Exception):
                        raise answer
                    text = answer.choices[0].message.content
                    predicted, reason = parse_answer(text, options_from_question(r['payload']['messages'][-1]['content']))
                    item = dict(r, arm=arm, response=text, predicted=predicted, parse_reason=reason,
                        correct=predicted == r['gold'], finish_reason=answer.choices[0].finish_reason)
                    if r['variant'] == 'original':
                        prior = previous[r['source_id'], 'T']
                        rendered_hash = digest(render(r['payload'], processor)['prompt'])
                        assert digest(r['payload']) == prior['input_sha256']
                        assert rendered_hash == prior['rendered_prompt_sha256']
                        item.update(previous_vllm_raw_response=prior['raw_response'],
                            previous_vllm_raw_predicted=prior['raw_predicted'],
                            previous_vllm_correct=prior['raw_predicted'] == r['gold'],
                            same_payload_and_rendered_prompt_hash_verified=True)
                    path.parent.mkdir(exist_ok=True, parents=True)
                    with path.open('a') as stream:
                        stream.write(json.dumps(item, ensure_ascii=False) + '\n')
                    existing.append(item)
                save(output / 'status.json', dict(stage='running', arm=arm, saved=len(existing), total=len(tasks), updated_at=now()))
            del engine
            import gc
            gc.collect()
            torch.cuda.empty_cache()
    if not all((output / arm / 'predictions.jsonl').exists() and
               len(rows(output / arm / 'predictions.jsonl')) == 72 for arm in ['atomic_replay', 'base']):
        save(output / 'status.json', dict(stage='arm_complete', arm=args.arm, updated_at=now()))
        return
    reports = {}
    for arm in ['atomic_replay', 'base']:
        data = rows(output / arm / 'predictions.jsonl')
        assert len(data) == 72
        reports[arm] = {}
        for variant in ['original', 'minimal', 'reference_fact', 'own_recall', 'generic_original_style']:
            group = [r for r in data if r['variant'] == variant]
            report = dict(n=len(group), correct=sum(r['correct'] for r in group),
                invalid=sum(r['predicted'] is None for r in group), length_limited=sum(r['finish_reason'] == 'length' for r in group),
                answer_counts=dict(Counter(str(r['predicted']) for r in group)))
            if variant == 'original':
                report['previous_vllm'] = dict(correct=sum(r['previous_vllm_correct'] for r in group),
                    answer_counts=dict(Counter(str(r['previous_vllm_raw_predicted']) for r in group)))
            reports[arm][variant] = report
    save(output / 'comparison.json', dict(status='complete', reports=reports, completed_at=now()))
    save(output / 'status.json', dict(stage='complete', updated_at=now()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--arm', choices=['atomic_replay', 'base'], required=True)
    args = p.parse_args()
    try:
        run(args)
    except Exception as error:
        save(args.output / 'status.json', dict(stage='failed', error=repr(error), updated_at=now()))
        raise
