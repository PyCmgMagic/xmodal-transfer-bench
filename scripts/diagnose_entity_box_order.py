#!/usr/bin/env python3
"""Post-hoc check: vary only the ordering of boxed format examples."""
import copy
from collections import Counter
import json
from pathlib import Path

from answer_readout import options_from_question
from answer_readout_entity_v4 import parse_answer
from entityqa_common import rows, save, write_rows, now, sha
from run_entityqa_lora import idle


def run():
    import torch
    from swift import TransformersEngine, InferRequest, RequestConfig
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    work = root / 'runs/entity50_repeat10_lora_ep3_20261104'
    parent = work / 'option_bias_diagnostic'
    dest = parent / 'box_example_order'
    dest.mkdir(exist_ok=True)
    original = [r for r in rows(parent / 'tasks.jsonl') if r['variant'] == 'original']
    assert len(original) == 16
    def sequence(order):
        return ', '.join('\\boxed{' + c + '}' for c in order[:3]) + ', or \\boxed{' + order[3] + '}'
    tasks = []
    for row in original:
        for order in ['BCDA', 'DABC']:
            r = copy.deepcopy(row)
            r.update(id=row['id'] + '-format-' + order, variant='format_' + order)
            text = r['payload']['messages'][-1]['content']
            assert sequence('ABCD') in text
            r['payload']['messages'][-1]['content'] = text.replace(sequence('ABCD'), sequence(order))
            tasks.append(r)
    write_rows(dest / 'tasks.jsonl', tasks)
    save(dest / 'plan.json', dict(created_at=now(), purpose='Vary only boxed format example order; question options and gold stay unchanged',
        n_entities=4, tasks=32, max_tokens=256, temperature=0, source_sha256=sha(Path(__file__)),
        tasks_sha256=sha(dest / 'tasks.jsonl'), limitations='Post-hoc 4-entity prompt diagnosis, not a transfer score'))
    idle('0')
    engine = TransformersEngine(str(root / 'model_cache/Qwen3-VL-4B-Instruct'), model_type='qwen3_vl',
        adapters=[str(work / 'atomic_replay/train/TEXT/checkpoint-237')], torch_dtype=torch.bfloat16,
        attn_impl='sdpa', device_map='cuda:0', max_batch_size=8)
    engine.template.enable_thinking = False
    results = []
    for start in range(0, len(tasks), 8):
        batch = tasks[start:start + 8]
        answers = engine.infer([InferRequest(messages=r['payload']['messages']) for r in batch],
            RequestConfig(max_tokens=256, temperature=0), use_tqdm=False)
        for r, answer in zip(batch, answers):
            if isinstance(answer, Exception):
                raise answer
            text = answer.choices[0].message.content
            pred, reason = parse_answer(text, options_from_question(r['payload']['messages'][-1]['content']))
            item = dict(r, response=text, predicted=pred, parse_reason=reason, correct=pred == r['gold'],
                        finish_reason=answer.choices[0].finish_reason)
            with (dest / 'predictions.jsonl').open('a') as stream:
                stream.write(json.dumps(item, ensure_ascii=False) + '\n')
            results.append(item)
        save(dest / 'status.json', dict(stage='running', saved=len(results), total=32, updated_at=now()))
    reports = {}
    for variant in ['format_BCDA', 'format_DABC']:
        g = [r for r in results if r['variant'] == variant]
        reports[variant] = dict(n=len(g), correct=sum(r['correct'] for r in g),
            invalid=sum(r['predicted'] is None for r in g),
            length_limited=sum(r['finish_reason'] == 'length' for r in g),
            answer_counts=dict(Counter(str(r['predicted']) for r in g)))
    save(dest / 'comparison.json', dict(status='complete', reports=reports, completed_at=now()))
    save(dest / 'status.json', dict(stage='complete', saved=32, updated_at=now()))


if __name__ == '__main__':
    try:
        run()
    except Exception as error:
        save(Path(__file__).resolve().parents[1] / 'runs/entity50_repeat10_lora_ep3_20261104/option_bias_diagnostic/box_example_order/status.json',
             dict(stage='failed', error=repr(error), updated_at=now()))
        raise
