#!/usr/bin/env python3
"""Exploratory component checks following the separate option-bias diagnostic."""
import copy
from collections import Counter
import json
import os
from pathlib import Path
import time

from entityqa_common import rows, save, write_rows, now, sha
from answer_readout import options_from_question
from answer_readout_entity_v4 import parse_answer
from run_entityqa_lora import idle


def run():
    import torch
    from swift import TransformersEngine, InferRequest, RequestConfig
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    work = root / 'runs/entity50_repeat10_lora_ep3_20261104'
    parent = work / 'option_bias_diagnostic'
    dest = parent / 'prompt_components'
    dest.mkdir(exist_ok=True, parents=True)
    original = [r for r in rows(parent / 'tasks.jsonl') if r['variant'] == 'original']
    assert len(original) == 16
    tasks = []
    date = 'Answer as of 3 November 2026. '
    forced = 'Select exactly one of the provided options, even if uncertain. '
    boxes = r': \boxed{A}, \boxed{B}, \boxed{C}, or \boxed{D}.'
    for row in original:
        for variant in ['remove_date', 'remove_uncertain', 'remove_box_examples', 'remove_system', 'remove_date_and_uncertain']:
            r = copy.deepcopy(row)
            r.update(id=row['id'] + '-' + variant, variant=variant)
            messages = r['payload']['messages']
            if variant in ['remove_date', 'remove_date_and_uncertain']:
                assert date in messages[0]['content']
                messages[0]['content'] = messages[0]['content'].replace(date, '')
            if variant in ['remove_uncertain', 'remove_date_and_uncertain']:
                assert forced in messages[0]['content']
                messages[0]['content'] = messages[0]['content'].replace(forced, '')
            if variant == 'remove_box_examples':
                assert boxes in messages[1]['content']
                messages[1]['content'] = messages[1]['content'].replace(boxes, '.')
            if variant == 'remove_system':
                r['payload']['messages'] = [messages[1]]
            tasks.append(r)
    assert len(tasks) == 80
    write_rows(dest / 'tasks.jsonl', tasks)
    save(dest / 'plan.json', dict(created_at=now(), status='prepared', tasks=80, entities=4,
        purpose='Post-hoc component diagnosis after original prompt 3/16 vs minimal 14/16 in original adapter',
        test_entity_selection='Same four fixed entities, no additional score-based selection',
        max_tokens=256, temperature=0, adapter='Original unmerged checkpoint-237',
        limitations='Small exploratory text diagnostic, not validation or transfer scores; no change to E23 results.',
        source_sha256=sha(Path(__file__)), tasks_sha256=sha(dest / 'tasks.jsonl')))
    while not (parent / 'comparison.json').exists():
        status = json.loads((parent / 'status.json').read_text()) if (parent / 'status.json').exists() else {}
        if status.get('stage') == 'failed':
            raise RuntimeError('Preceding diagnostic failed; inspect')
        save(dest / 'status.json', dict(stage='waiting_for_preceding_diagnostic', updated_at=now()))
        time.sleep(5)
    while True:
        try:
            idle('0')
            break
        except RuntimeError:
            save(dest / 'status.json', dict(stage='waiting_for_free_gpu', updated_at=now()))
            time.sleep(5)
    path = dest / 'predictions.jsonl'
    existing = rows(path) if path.exists() else []
    pending = [r for r in tasks if r['id'] not in {s['id'] for s in existing}]
    engine = TransformersEngine(str(root / 'model_cache/Qwen3-VL-4B-Instruct'), model_type='qwen3_vl',
        adapters=[str(work / 'atomic_replay/train/TEXT/checkpoint-237')], torch_dtype=torch.bfloat16,
        attn_impl='sdpa', device_map='cuda:0', max_batch_size=8)
    engine.template.enable_thinking = False
    for start in range(0, len(pending), 8):
        batch = pending[start:start + 8]
        answers = engine.infer([InferRequest(messages=r['payload']['messages']) for r in batch],
            RequestConfig(max_tokens=256, temperature=0), use_tqdm=False)
        for r, answer in zip(batch, answers):
            if isinstance(answer, Exception):
                raise answer
            text = answer.choices[0].message.content
            pred, reason = parse_answer(text, options_from_question(r['payload']['messages'][-1]['content']))
            item = dict(r, response=text, predicted=pred, parse_reason=reason, correct=pred == r['gold'],
                        finish_reason=answer.choices[0].finish_reason)
            with path.open('a') as stream:
                stream.write(json.dumps(item, ensure_ascii=False) + '\n')
            existing.append(item)
        save(dest / 'status.json', dict(stage='running', saved=len(existing), total=80, updated_at=now()))
    reports = {}
    for variant in dict.fromkeys(r['variant'] for r in tasks):
        group = [r for r in existing if r['variant'] == variant]
        assert len(group) == 16
        reports[variant] = dict(n=16, correct=sum(r['correct'] for r in group),
            invalid=sum(r['predicted'] is None for r in group),
            length_limited=sum(r['finish_reason'] == 'length' for r in group),
            answer_counts=dict(Counter(str(r['predicted']) for r in group)))
    save(dest / 'comparison.json', dict(status='complete', reports=reports, completed_at=now()))
    save(dest / 'status.json', dict(stage='complete', saved=len(existing), total=80, updated_at=now()))


if __name__ == '__main__':
    try:
        run()
    except Exception as error:
        save(Path(__file__).resolve().parents[1] / 'runs/entity50_repeat10_lora_ep3_20261104/option_bias_diagnostic/prompt_components/status.json',
             dict(stage='failed', error=repr(error), updated_at=now()))
        raise
