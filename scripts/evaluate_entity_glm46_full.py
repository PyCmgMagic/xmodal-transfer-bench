#!/usr/bin/env python3
"""Paired GLM-4.6V-Flash evaluation for the repaired full entity set."""
import argparse
import copy
from collections import Counter
import gc
import json
import os
from pathlib import Path
import re
import time

from answer_readout_entity_v4 import parse_answer, VERSION as READOUT_VERSION
from entityqa_common import rows, save, write_rows, digest, sha, now
from entityqa_prompt_neutral_v5 import eval_payload, PROMPT_VERSION

ANSWER_STOP_SEQUENCES = [r'\boxed{A}', r'\boxed{B}', r'\boxed{C}', r'\boxed{D}']


def glm_readout(response, options):
    if isinstance(response, str):
        def normalize_box(match):
            body = match.group(1).strip()
            letter = re.fullmatch(r'(?:\\boxed\s*\{\s*)?([A-Da-d])(?:\s*\})?', body)
            if letter:
                return r'\boxed{' + letter[1].upper() + '}'
            return match.group(0)

        response = re.sub(
            r'<\|begin_of_box\|>(.*?)<\|end_of_box\|>', normalize_box,
            response, flags=re.DOTALL)
    return parse_answer(response, options)


def metric(records):
    n = len(records)
    assert n
    return dict(n=n, correct=sum(r['raw_predicted'] == r['gold'] for r in records),
        accuracy=sum(r['raw_predicted'] == r['gold'] for r in records) / n,
        invalid=sum(r['raw_predicted'] is None for r in records),
        invalid_rate=sum(r['raw_predicted'] is None for r in records) / n,
        answer_counts=dict(Counter(str(r['raw_predicted']) for r in records)),
        parse_reasons=dict(Counter(r['raw_parse_reason'] for r in records)),
        stop_reasons=dict(Counter(r['stop_reason'] for r in records)),
        generation_cap_counts=dict(Counter(str(r['generation_max_tokens']) for r in records)))


def generation_details(folder):
    config = json.loads((folder / 'config.json').read_text())
    records = rows(folder / 'predictions.jsonl')
    counts = Counter(str(r.get('generation_max_tokens', config['max_tokens'])) for r in records)
    return dict(current_max_tokens=config['max_tokens'], generation_cap_counts=dict(counts),
                mixed_caps=len(counts) > 1)


def configure_generation(output, config, records, allow_cap_reduction=False):
    """Resume only with the same protocol, or an explicit, archived cap reduction."""
    config_path = output / 'config.json'
    if not config_path.exists():
        assert not records, 'Predictions exist without their generation configuration'
        save(config_path, config)
        return
    previous = json.loads(config_path.read_text())
    changed = {key for key in set(previous) | set(config) if previous.get(key) != config.get(key)}
    if changed:
        assert allow_cap_reduction and changed <= {'max_tokens', 'wall_seconds'}, 'Resume configuration changed'
        assert 0 < config['max_tokens'] < previous['max_tokens'], 'Only a lower cap can be adopted'
        revision = output / 'configuration_revisions' / f'cap_{previous["max_tokens"]}_to_{config["max_tokens"]}'
        revision.mkdir(parents=True, exist_ok=False)
        import shutil
        files = {}
        for name in ['config.json', 'predictions.jsonl', 'status.json', 'summary.json']:
            source = output / name
            if source.exists():
                shutil.copy2(source, revision / name)
                files[name] = sha(source)
        for record in records:
            record.setdefault('generation_max_tokens', previous['max_tokens'])
        write_rows(output / 'predictions.jsonl', records)
        save(revision / 'change.json', dict(changed_at=now(), reason='User requested a lower output limit',
            previous_max_tokens=previous['max_tokens'], new_max_tokens=config['max_tokens'],
            preserved_predictions=len(records), archived_sha256=files,
            raw_responses_unchanged=True, aggregate_has_multiple_generation_caps=bool(records)))
        save(config_path, config)
    else:
        for record in records:
            record.setdefault('generation_max_tokens', previous['max_tokens'])


def summarize(work, arm):
    records = rows(work / 'evaluation' / arm / 'predictions.jsonl')
    config = json.loads((work / 'evaluation' / arm / 'config.json').read_text())
    for record in records:
        record.setdefault('generation_max_tokens', config['max_tokens'])
    assert len(records) == 7320
    groups = {}
    dev = set(json.loads((work / 'data/manifest.json').read_text())['prior_development_qids'])
    for mod in 'TV':
        for probe in ['original', 'rotation']:
            subset = [r for r in records if r['modality'] == mod and
                      ((r['probe_id'] == 'rotation') == (probe == 'rotation'))]
            assert len(subset) == (3628 if probe == 'original' else 32)
            item = dict(all=metric(subset))
            if probe == 'original':
                item.update(prior_development50=metric([r for r in subset if r['qid'] in dev]),
                    other1764=metric([r for r in subset if r['qid'] not in dev]),
                    direct=metric([r for r in subset if r['probe_id'] == 'direct']),
                    statement=metric([r for r in subset if r['probe_id'] == 'statement']))
            groups[mod + '/' + probe] = item
    save(work / 'evaluation' / arm / 'summary.json', dict(status='complete', arm=arm,
        model='GLM-4.6V-Flash', groups=groups, completed_at=now(),
        generation_cap_counts=dict(Counter(str(r['generation_max_tokens']) for r in records)),
        interpretation='Raw generation; native thinking disabled; no constrained finalizer.'))
    save(work / 'evaluation' / arm / 'status.json', dict(status='complete', saved=len(records),
        total=7320, updated_at=now()))


def run(args):
    import torch
    from swift import TransformersEngine, InferRequest, RequestConfig
    torch.set_num_threads(4)
    root, work = args.root.resolve(), args.work.resolve()
    data = work / 'evaluation_data'
    base = root / 'model_cache/modelscope/models/ZhipuAI--GLM-4.6V-Flash/snapshots/master'
    assert (base / 'download_complete.json').is_file()
    checkpoint = getattr(args, 'checkpoint', None) or work / 'train/TEXT/checkpoint-8505'
    checkpoint = checkpoint.resolve()
    adapter = checkpoint if args.arm == 'glm46_lora' else None
    output_arm = 'atomic_replay' if args.arm == 'glm46_lora' else 'base'
    output = work / 'evaluation' / output_arm
    output.mkdir(parents=True, exist_ok=True)
    labels = {r['id']: r for r in rows(data / 'audit/eval_labels.jsonl')}
    tasks = []
    for modality, filename in [('T', 'eval_text.jsonl'), ('V', 'eval_multimodal.jsonl')]:
        selected = rows(data / 'shared' / filename)
        for row in selected:
            label = labels[row['id']]
            payload = eval_payload(row, data)
            tasks.append(dict(id=row['id'], qid=row['qid'], modality=modality,
                probe_id=label['probe_id'], gold=label['answer'], payload=payload))
    assert len(tasks) == 7320
    max_tokens = getattr(args, 'max_tokens', 6000)
    assert max_tokens > 0
    config = dict(model=str(base), arm=args.arm, adapter=str(adapter) if adapter else None,
        dataset_inventory_sha256=sha(data / 'package_inventory.json'),
        input_sha256=digest([{k: v for k, v in t.items() if k != 'gold'} for t in tasks]),
        input_count=len(tasks), prompt_protocol=PROMPT_VERSION,
        max_tokens=max_tokens, stop_sequences=ANSWER_STOP_SEQUENCES,
        max_model_len=73728, temperature=0, seed=42,
        wall_seconds=f'not separately enforced; bounded to {max_tokens} output tokens; stop at boxed answer',
        loop_guard='bounded generation plus stop at the first boxed answer; native thinking disabled',
        thinking=False, vision_max_pixels=262144, gpu='0',
        version=1, readout_version=READOUT_VERSION,
        primary_score='raw generation accuracy and invalid rate', batch_size=4)
    path = output / 'predictions.jsonl'
    records = rows(path) if path.exists() else []
    lookup = {(t['id'], t['modality']): t for t in tasks}
    seen = set()
    for record in records:
        key = record['id'], record['modality']
        assert key in lookup and key not in seen
        task = lookup[key]
        assert record['gold'] == task['gold'] and record['input_sha256'] == digest(task['payload'])
        seen.add(key)
    configure_generation(output, config, records, getattr(args, 'allow_cap_reduction', False))
    pending = [t for t in tasks if (t['id'], t['modality']) not in seen]
    if not pending:
        summarize(work, output_arm)
        return

    os.environ.update(CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1', MODELSCOPE_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='4', VLLM_WORKER_MULTIPROC_METHOD='spawn')
    engine = TransformersEngine(str(base), model_type='glm4v', template_type='glm4_5v',
        adapters=[str(adapter)] if adapter else None, torch_dtype=torch.bfloat16,
        attn_impl='sdpa', device_map='cuda:0', max_batch_size=4)
    engine.template.enable_thinking = False
    request = RequestConfig(max_tokens=max_tokens, temperature=0, seed=42, return_details=True,
                            stop=ANSWER_STOP_SEQUENCES)
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
            parsed, reason = glm_readout(response,
                __import__('answer_readout').options_from_question(task['payload']['messages'][-1]['content']))
            item = {k: task[k] for k in ['id', 'qid', 'modality', 'probe_id', 'gold']}
            item.update(arm=output_arm, raw_response=response, raw_predicted=parsed,
                raw_parse_reason=reason, predicted=parsed, correct=parsed == task['gold'],
                finalizer=None, stop_reason=str(choice.finish_reason),
                generated_tokens=reply.usage.completion_tokens,
                generation_max_tokens=max_tokens,
                latency_seconds=reply.usage.total_time if hasattr(reply.usage, 'total_time') else None,
                input_sha256=digest(task['payload']), saved_at=now())
            new.append(item)
        with path.open('a', encoding='utf-8') as stream:
            for item in new:
                stream.write(json.dumps(item, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        records.extend(new)
        groups = {}
        for modality in 'TV':
            subset = [r for r in records if r['modality'] == modality and r['probe_id'] != 'rotation']
            if subset:
                groups[modality] = metric(subset)
        progress = dict(stage='generating', saved=len(records), total=7320,
            elapsed_seconds=round(time.monotonic() - started, 1), groups=groups, updated_at=now())
        save(output / 'status.json', progress)
        save(output / 'summary.json', progress)
        print(json.dumps(dict(saved=len(records), total=7320, groups=groups), ensure_ascii=False), flush=True)
    assert len(records) == 7320
    summarize(work, output_arm)
    del engine
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--arm', choices=['base', 'glm46_lora'], required=True)
    parser.add_argument('--checkpoint', type=Path, help='Explicit adapter checkpoint for a new training run')
    parser.add_argument('--max-tokens', type=int, default=6000)
    parser.add_argument('--allow-cap-reduction', action='store_true',
                        help='Archive the previous configuration and preserve completed outputs before lowering the cap')
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        save(args.work / 'evaluation' / ('base' if args.arm == 'base' else 'atomic_replay') / 'failure.json',
             dict(error=repr(error), updated_at=now()))
        raise
