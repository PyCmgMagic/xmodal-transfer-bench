#!/usr/bin/env python3
"""Closed-book, paired rule evaluation for a GLM TEXT LoRA and its clean base."""
import argparse
from collections import Counter
import copy
import gc
import json
import os
from pathlib import Path
import time

from answer_readout import options_from_question
from answer_readout_entity_v4 import VERSION as READOUT_VERSION
from entityqa_common import digest, now, rows, save, sha
from entityqa_prompt_neutral_v5 import eval_payload
from evaluate_entity_glm46_full import (
    ANSWER_STOP_SEQUENCES, configure_generation, glm_readout, metric,
)

PROMPT_VERSION = 'rules-v7-neutral-box-brief-v1'


def tasks_for(work, smoke=False):
    data = work / 'evaluation_data'
    labels = {r['id']: r for r in rows(data / 'audit/eval_labels.jsonl')}
    tasks = []
    for mod, filename in [('T', 'eval_text.jsonl'), ('V', 'eval_multimodal.jsonl')]:
        selected = rows(data / 'shared' / filename)
        assert len(selected) == 2000 and len({r['id'] for r in selected}) == 2000
        for row in selected:
            label = labels[row['id']]
            assert label['qid'] == row['qid'] and label['answer'] in 'ABCD'
            payload = eval_payload(row, data)
            # Only the original test question/options and its modality image enter generation.
            assert set(payload) <= {'messages', 'images'}
            tasks.append(dict(id=row['id'], qid=row['qid'], modality=mod,
                question_type=label['question_type'], family=label['entity_kind'],
                evaluation_group=label['evaluation_group'], task_variant=label.get('task_variant', ''),
                bias_watchlist=bool(label.get('bias_watchlist', False)),
                gold=label['answer'], payload=payload))
    assert len(tasks) == len({(r['id'], r['modality']) for r in tasks}) == 4000
    assert {r['id'] for r in tasks if r['modality'] == 'T'} == {r['id'] for r in tasks if r['modality'] == 'V'}
    if smoke:
        tasks = [r for mod in 'TV' for r in [t for t in tasks if t['modality'] == mod][:4]]
    return tasks


def summaries(records, full=False):
    groups = {}
    for mod in 'TV':
        subset = [r for r in records if r['modality'] == mod]
        if not subset:
            continue
        groups[mod + '/all'] = metric(subset)
        for field in ['question_type', 'family', 'evaluation_group', 'task_variant']:
            for value in sorted({r[field] for r in subset}):
                if value:
                    selected = [r for r in subset if r[field] == value]
                    groups[f'{mod}/{field}/{value}'] = metric(selected)
        if full:
            assert len(subset) == 2000
            assert Counter(r['question_type'] for r in subset) == dict(direct=1000, diagnosis=1000)
    return groups


def run(args):
    root, work = args.root.resolve(), args.work.resolve()
    plan = json.loads((work / 'experiment.json').read_text())
    base = Path(plan['model_path'])
    adapter = args.checkpoint.resolve() if args.arm == 'TEXT' else None
    if adapter:
        assert (adapter / 'adapter_model.safetensors').is_file()
    tasks = tasks_for(work)
    full_input_hash = digest([{k: v for k, v in t.items() if k != 'gold'} for t in tasks])
    assert full_input_hash == plan['evaluation']['input_sha256']
    if args.smoke:
        tasks = tasks_for(work, smoke=True)
    output = work / ('evaluation_smoke' if args.smoke else 'evaluation') / args.arm
    output.mkdir(parents=True, exist_ok=True)
    cap = plan['evaluation']['max_tokens']
    config = dict(model=str(base), arm=args.arm, adapter=str(adapter) if adapter else None,
        adapter_sha256=sha(adapter / 'adapter_model.safetensors') if adapter else None,
        dataset_inventory_sha256=sha(work / 'evaluation_data/package_inventory.json'),
        input_sha256=digest([{k: v for k, v in t.items() if k != 'gold'} for t in tasks]),
        input_count=len(tasks), prompt_protocol=PROMPT_VERSION, max_tokens=cap,
        stop_sequences=ANSWER_STOP_SEQUENCES, temperature=0, seed=42,
        thinking=False, vision_max_pixels=262144, gpu='0', batch_size=4,
        readout_version=READOUT_VERSION, framework='ms-swift TransformersEngine',
        version=1, smoke_only=args.smoke,
        primary_score='raw generation accuracy and invalid rate; invalids counted wrong',
        oracle_material=False, retrieval=False, finalizer=False)
    path = output / 'predictions.jsonl'
    records = rows(path) if path.exists() else []
    lookup, seen = {(t['id'], t['modality']): t for t in tasks}, set()
    for record in records:
        key = record['id'], record['modality']
        assert key in lookup and key not in seen
        assert record['gold'] == lookup[key]['gold']
        assert record['input_sha256'] == digest(lookup[key]['payload'])
        seen.add(key)
    configure_generation(output, config, records)
    pending = [t for t in tasks if (t['id'], t['modality']) not in seen]
    started = time.monotonic()
    if pending:
        os.environ.update(CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', MODELSCOPE_OFFLINE='1',
            TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MAX_PIXELS='262144')
        import torch
        from swift import TransformersEngine, InferRequest, RequestConfig
        torch.set_num_threads(4)
        engine = TransformersEngine(str(base), model_type='glm4v', template_type='glm4_5v',
            adapters=[str(adapter)] if adapter else None, torch_dtype=torch.bfloat16,
            attn_impl='sdpa', device_map='cuda:0', max_batch_size=4)
        engine.template.enable_thinking = False
        request = RequestConfig(max_tokens=cap, temperature=0, seed=42,
            return_details=True, stop=ANSWER_STOP_SEQUENCES)
        for start in range(0, len(pending), 4):
            batch = pending[start:start + 4]
            replies = engine.infer([InferRequest(**copy.deepcopy(t['payload'])) for t in batch],
                request, use_tqdm=False)
            assert len(replies) == len(batch)
            new = []
            for task, reply in zip(batch, replies):
                if isinstance(reply, Exception):
                    raise reply
                choice = reply.choices[0]
                response = choice.message.content or ''
                predicted, reason = glm_readout(response,
                    options_from_question(task['payload']['messages'][-1]['content']))
                item = {k: v for k, v in task.items() if k != 'payload'}
                item.update(arm=args.arm, raw_response=response, raw_predicted=predicted,
                    raw_parse_reason=reason, predicted=predicted, correct=predicted == task['gold'],
                    finalizer=None, stop_reason=str(choice.finish_reason),
                    generated_tokens=reply.usage.completion_tokens, generation_max_tokens=cap,
                    input_sha256=digest(task['payload']), saved_at=now())
                new.append(item)
            with path.open('a', encoding='utf-8') as stream:
                for item in new:
                    stream.write(json.dumps(item, ensure_ascii=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            records.extend(new)
            progress = dict(stage='generating', saved=len(records), total=len(tasks),
                elapsed_seconds=round(time.monotonic() - started, 1),
                groups=summaries(records), updated_at=now())
            save(output / 'status.json', progress)
            save(output / 'summary.json', progress)
            print(json.dumps(dict(saved=len(records), total=len(tasks),
                groups={m: progress['groups'].get(m + '/all') for m in 'TV'}), ensure_ascii=False), flush=True)
        del engine
        gc.collect()
        torch.cuda.empty_cache()
    assert len(records) == (8 if args.smoke else 4000)
    save(output / 'summary.json', dict(status='complete', arm=args.arm,
        model='GLM-4.6V-Flash', smoke_only=args.smoke, groups=summaries(records, not args.smoke),
        saved=len(records), total=len(tasks), completed_at=now(),
        interpretation='Closed book, raw generation, brief prompted reasoning; native thinking disabled.'))
    save(output / 'status.json', dict(status='complete', saved=len(records),
        total=len(tasks), updated_at=now()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--arm', choices=['TEXT', 'base'], required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    try:
        if args.arm == 'TEXT':
            assert args.checkpoint is not None
        run(args)
    except Exception as error:
        output = args.work / ('evaluation_smoke' if args.smoke else 'evaluation') / args.arm
        save(output / 'failure.json', dict(error=repr(error), updated_at=now()))
        raise
