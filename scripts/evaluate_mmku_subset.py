#!/usr/bin/env python3
"""Evaluate MMKU short answers with the project's official CEM and F1 code."""
import argparse
import copy
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def inference_payload(row, data_root):
    if any(m['role'] != 'user' for m in row['messages']):
        raise ValueError('Heldout inference must contain user messages only')
    return {'messages': copy.deepcopy(row['messages']),
            'images': [str((Path(data_root) / x).resolve()) for x in row['images']]}


def load_official(code):
    folder = Path(code) / 'evaluation'
    sys.path.insert(0, str(folder.resolve()))
    spec = importlib.util.spec_from_file_location('mmku_official_eval', folder / 'eval_acc_f1.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(path, obj):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def run(args):
    import torch
    from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine
    from swift.utils import seed_everything
    seed_everything(42)
    torch.set_num_threads(4)
    data = args.data.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (data / 'test.jsonl').read_text().splitlines() if line.strip()]
    if len(rows) != 500:
        raise ValueError(f'Expected the fixed 500-item evaluation, got {len(rows)}')
    official = load_official(args.official_code)
    config = {'model': str(args.model.resolve()), 'dataset_sha256': hashlib.sha256((data / 'test.jsonl').read_bytes()).hexdigest(),
              'protocol': 'Official rewritten question, short answer; no added CoT or reference context',
              'max_tokens': 64, 'temperature': 0, 'seed': 42, 'max_pixels': 262144,
              'batch_size': args.batch_size, 'scoring': 'Unmodified MMKU VQAEval CEM + eval_acc_f1 F1',
              'scorer_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                  [args.official_code / 'evaluation/tools.py', args.official_code / 'evaluation/eval_acc_f1.py']}}
    if (out / 'config.json').exists() and json.loads((out / 'config.json').read_text()) != config:
        raise ValueError('Evaluation configuration changed')
    save(out / 'config.json', config)
    existing = [json.loads(x) for x in (out / 'predictions.jsonl').read_text().splitlines()] if (out / 'predictions.jsonl').exists() else []
    done = {x['id']: x for x in existing}
    if len(done) != len(existing):
        raise ValueError('Duplicate saved prediction')
    expected = {r['id']: digest(inference_payload(r, data)) for r in rows}
    for sid, record in done.items():
        if sid not in expected or record['input_sha256'] != expected[sid]:
            raise ValueError('Previously saved inference input changed')
    save(out / 'runtime.json', {'pid': os.getpid(), 'started_at': time.time(), 'torch': torch.__version__})
    engine = TransformersEngine(str(args.model.resolve()), model_type='qwen3_vl',
                                 torch_dtype=torch.bfloat16, device_map='cuda:0',
                                 max_batch_size=args.batch_size, attn_impl='sdpa')
    engine.template.enable_thinking = False
    request_config = RequestConfig(max_tokens=64, temperature=0, seed=42, return_details=True)
    missing = [r for r in rows if r['id'] not in done]

    def summarize():
        values = list(done.values())
        n = len(values)
        result = {'status': 'complete' if n == len(rows) else 'running', 'saved': n, 'total': len(rows),
                  'correct': sum(x['cem'] for x in values), 'cem_accuracy': sum(x['cem'] for x in values) / n if n else None,
                  'mean_f1': sum(x['f1'] for x in values) / n if n else None,
                  'empty_answers': sum(not x['response'].strip() for x in values),
                  'length_limited': sum(x['finish_reason'] == 'length' for x in values),
                  'mean_generation_seconds_per_batch_item': sum(x['batch_seconds_per_item'] for x in values) / n if n else None,
                  'updated_at': time.time()}
        save(out / 'report.json', result)
        return result

    for start in range(0, len(missing), args.batch_size):
        batch = missing[start:start + args.batch_size]
        payloads = [inference_payload(row, data) for row in batch]
        t = time.monotonic()
        replies = engine.infer([InferRequest(**p) for p in payloads], request_config, use_tqdm=False)
        elapsed = time.monotonic() - t
        if len(replies) != len(batch):
            raise RuntimeError('Missing inference results')
        for row, payload, reply in zip(batch, payloads, replies):
            if isinstance(reply, Exception):
                raise reply
            choice = reply.choices[0]
            response = choice.message.content or ''
            record = {'id': row['id'], 'entity': row['entity'], 'response': response, 'answer': row['answer'],
                      'cem': official.eval_tool.evaluate(response, [official.normalize_text(row['answer'])]),
                      'f1': official.get_f1_score(response, official.normalize_text(row['answer'])),
                      'input_sha256': digest(payload), 'finish_reason': choice.finish_reason,
                      'batch_seconds_per_item': elapsed / len(batch), 'raw_reply': dataclasses.asdict(reply)}
            with (out / 'predictions.jsonl').open('a') as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                stream.flush()
            done[row['id']] = record
        result = summarize()
        print(f"{out.name}: {result['saved']}/{len(rows)}, CEM={result['cem_accuracy']:.4f}, F1={result['mean_f1']:.4f}, batch={elapsed:.2f}s", flush=True)
    print(json.dumps(summarize(), indent=2), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--official-code', type=Path, required=True)
    ap.add_argument('--batch-size', type=int, default=8)
    run(ap.parse_args())
