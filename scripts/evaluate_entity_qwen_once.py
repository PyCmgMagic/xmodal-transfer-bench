#!/usr/bin/env python3
"""Evaluate the Qwen single-pass entity LoRA with paired inputs and a fixed 1024-token cap."""
import argparse
import copy
from collections import Counter
import gc
import json
import os
from pathlib import Path
import time

from answer_readout import options_from_question
from answer_readout_entity_v4 import parse_answer, VERSION as READOUT_VERSION
from entityqa_common import digest, now, rows, save, sha
from entityqa_prompt_neutral_v5 import eval_payload, PROMPT_VERSION
from evaluate_entity_glm46_full import ANSWER_STOP_SEQUENCES, configure_generation, metric


def tasks_for(work):
    data = work / 'evaluation_data'
    labels = {r['id']: r for r in rows(data / 'audit/eval_labels.jsonl')}
    tasks = []
    for mod, filename in [('T', 'eval_text.jsonl'), ('V', 'eval_multimodal.jsonl')]:
        for row in rows(data / 'shared' / filename):
            label = labels[row['id']]
            tasks.append(dict(id=row['id'], qid=row['qid'], modality=mod,
                probe_id=label['probe_id'], gold=label['answer'], payload=eval_payload(row, data)))
    assert len(tasks) == len({(r['id'], r['modality']) for r in tasks}) == 7320
    return tasks


def summarize(work, output, records, arm, smoke):
    assert len(records) == (8 if smoke else 7320)
    dev = set(json.loads((work / 'data/manifest.json').read_text())['prior_development_qids'])
    groups = {}
    for mod in 'TV':
        for probe in ['original', 'rotation']:
            selected = [r for r in records if r['modality'] == mod and
                        ((r['probe_id'] == 'rotation') == (probe == 'rotation'))]
            if not selected:
                continue
            if not smoke:
                assert len(selected) == (3628 if probe == 'original' else 32)
            item = dict(all=metric(selected))
            if probe == 'original' and not smoke:
                item.update(prior_development50=metric([r for r in selected if r['qid'] in dev]),
                    other1764=metric([r for r in selected if r['qid'] not in dev]),
                    direct=metric([r for r in selected if r['probe_id'] == 'direct']),
                    statement=metric([r for r in selected if r['probe_id'] == 'statement']))
            groups[mod + '/' + probe] = item
    save(output / 'summary.json', dict(status='complete', model='Qwen3-VL-4B-Instruct',
        arm=arm, smoke_only=smoke, groups=groups, completed_at=now(),
        interpretation='Raw generation, brief prompted reasoning, unmerged LoRA, no constrained finalizer.'))
    save(output / 'status.json', dict(status='complete', saved=len(records), total=len(records), updated_at=now()))


def run(args):
    root, work = args.root.resolve(), args.work.resolve()
    model = root / 'model_cache/Qwen3-VL-4B-Instruct'
    assert (model / 'download_complete.json').is_file()
    adapter = args.checkpoint.resolve() if args.arm == 'TEXT' else None
    if adapter:
        assert (adapter / 'adapter_model.safetensors').is_file()
    tasks = tasks_for(work)
    full_input_hash = digest([{k: v for k, v in t.items() if k != 'gold'} for t in tasks])
    plan = json.loads((work / 'experiment.json').read_text())
    assert full_input_hash == plan['evaluation']['input_sha256']
    if args.smoke:
        tasks = [t for mod in 'TV' for t in [x for x in tasks if x['modality'] == mod][:4]]
    output = work / ('evaluation_smoke' if args.smoke else 'evaluation') / args.arm
    output.mkdir(parents=True, exist_ok=True)
    cap = plan['evaluation']['max_tokens']
    config = dict(model=str(model), arm=args.arm, adapter=str(adapter) if adapter else None,
        adapter_sha256=sha(adapter / 'adapter_model.safetensors') if adapter else None,
        dataset_inventory_sha256=sha(work / 'evaluation_data/package_inventory.json'),
        input_sha256=digest([{k: v for k, v in t.items() if k != 'gold'} for t in tasks]),
        input_count=len(tasks), prompt_protocol=PROMPT_VERSION, max_tokens=cap,
        stop_sequences=ANSWER_STOP_SEQUENCES, temperature=0, seed=42,
        thinking=False, vision_max_pixels=262144, gpu='0', batch_size=4,
        readout_version=READOUT_VERSION, framework='ms-swift TransformersEngine',
        version=1, smoke_only=args.smoke, primary_score='raw generation accuracy and invalid rate')
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
    if not pending:
        summarize(work, output, records, args.arm, args.smoke)
        return
    os.environ.update(CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', MODELSCOPE_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', MAX_PIXELS='262144')
    import torch
    from swift import TransformersEngine, InferRequest, RequestConfig
    torch.set_num_threads(4)
    engine = TransformersEngine(str(model), model_type='qwen3_vl', template_type='qwen3_vl',
        adapters=[str(adapter)] if adapter else None, torch_dtype=torch.bfloat16,
        attn_impl='sdpa', device_map='cuda:0', max_batch_size=4)
    engine.template.enable_thinking = False
    request = RequestConfig(max_tokens=cap, temperature=0, seed=42,
        return_details=True, stop=ANSWER_STOP_SEQUENCES)
    started = time.monotonic()
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
            predicted, reason = parse_answer(response,
                options_from_question(task['payload']['messages'][-1]['content']))
            item = {k: task[k] for k in ['id', 'qid', 'modality', 'probe_id', 'gold']}
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
        groups = {}
        for mod in 'TV':
            selected = [r for r in records if r['modality'] == mod and r['probe_id'] != 'rotation']
            if selected:
                groups[mod] = metric(selected)
        status = dict(stage='generating', saved=len(records), total=len(tasks),
            elapsed_seconds=round(time.monotonic() - started, 1), groups=groups, updated_at=now())
        save(output / 'status.json', status)
        save(output / 'summary.json', status)
        print(json.dumps(status, ensure_ascii=False), flush=True)
    summarize(work, output, records, args.arm, args.smoke)
    del engine
    gc.collect()
    torch.cuda.empty_cache()


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
