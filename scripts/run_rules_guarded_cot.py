#!/usr/bin/env python3
"""Run rule-only CoT with loop stopping and separately audited answer readout."""
import argparse
import copy
import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import run_v7_evaluation as ev
import run_rules1000_10x_text_evaluation as queue
from answer_readout import choice_hook
from guarded_cot import GenerationGuard, final_readout_payload, inference_payload

ROOT = Path(__file__).resolve().parents[1]
NAME = 'evaluation_guarded_cot_20261103'
PROTOCOL = 'prompted-cot-loop-guard-constrained-readout-v1'


def read(path):
    return json.loads(Path(path).read_text())


def append(path, row):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def load_records(path):
    if not path.exists():
        return {}
    rows, notes = ev.read_rows(path, allow_partial_tail=True)
    if notes:
        # Only an incomplete last JSON line can be repaired by read_rows().
        import shutil
        shutil.copy2(path, path.with_name(path.name + f'.partial-{time.time_ns()}'))
        ev.save_rows(path, rows)
    return ev.id_map(rows, path)


def checkpoint_identity(name, checkpoint):
    """Reuse audited hashes only while all inode change times predate that audit."""
    previous = queue.WORK / 'evaluation_direct_answer_8tok_20261102' / name / 'manifest.json'
    if previous.exists():
        entries = read(previous)['config'].get('adapters', {}).values()
        for entry in entries:
            if entry['path'] != str(checkpoint.resolve()) or entry['kind'] != 'dense':
                continue
            files = {p.name: p for p in checkpoint.glob('model*.safetensors')}
            files['config.json'] = checkpoint / 'config.json'
            if set(files) != set(entry['sha256']):
                continue
            inventory = {k: p.stat() for k, p in files.items()}
            if not all(s.st_size > 0 and max(s.st_mtime_ns, s.st_ctime_ns) <= previous.stat().st_mtime_ns
                       for s in inventory.values()):
                continue
            if ev.file_hash(files['config.json']) != entry['sha256']['config.json']:
                continue
            return {**entry, 'hash_provenance': {'manifest': str(previous),
                    'manifest_sha256': ev.file_hash(previous),
                    'reused_unchanged_files': {k: {'size': s.st_size, 'inode': s.st_ino,
                        'mtime_ns': s.st_mtime_ns, 'ctime_ns': s.st_ctime_ns}
                        for k, s in inventory.items()}}}
    return ev.checkpoint(checkpoint)


def summarize(base):
    reports = []
    for name, _cp, directions, _mode in queue.EXPERIMENTS:
        for direction in directions:
            path = base / name / direction / 'report.json'
            if path.exists():
                reports.append({'experiment': name, **read(path)})
    ev.save_json(base / 'summary.json', {'updated_at': ev.now(), 'reports': reports})
    lines = ['# 规则类 CoT 评测（含循环保护与答案补选）', '',
             '原始 CoT 与同模型约束补选后的结果分别统计；不与历史协议合并。', '',
             '| 训练组 | 方向 | 完成 | 原始CoT准确率 | 原始无效 | 补选次数 | 补选后准确率 | 最终无效 |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in reports:
        c = r['cot_only']
        lines.append(f"| {r['experiment']} | {r['direction']} | {r['matched']}/{r['total']} | "
                     f"{c['accuracy']:.2%} | {c['empty_prediction'] + c['format_violation']} | "
                     f"{r['finalized']} | {r['accuracy']:.2%} | "
                     f"{r['empty_prediction'] + r['format_violation']} |")
    (base / 'report_zh.md').write_text('\n'.join(lines) + '\n')
    return reports


def make_engine(checkpoint, max_seconds):
    import torch
    from swift.infer_engine import TransformersEngine
    from swift.template import get_template
    from transformers import LogitsProcessorList

    class GuardedEngine(TransformersEngine):
        mode = 'cot'
        guard = None

        def _get_template(self, processor, template_type=None):
            return get_template(processor, template_type=template_type,
                                use_chat_template=True, enable_thinking=False,
                                max_length=262144, max_pixels=1003520)

        def _infer(self, *args, **kwargs):
            def hook(generation):
                if self.mode == 'readout':
                    self.guard = None
                    return choice_hook(self.tokenizer)(generation)
                eos = generation['generation_config'].eos_token_id
                eos = eos[0] if isinstance(eos, (list, tuple)) else eos
                self.guard = GenerationGuard(
                    self.tokenizer, generation['inputs']['input_ids'].shape[-1],
                    eos, max_seconds=max_seconds)
                generation['inputs']['logits_processor'] = LogitsProcessorList([self.guard])
                return generation
            kwargs['pre_infer_hook'] = hook
            return super()._infer(*args, **kwargs)

    return GuardedEngine(str(checkpoint), torch_dtype=torch.bfloat16,
                         device_map='cuda:0', max_batch_size=1, attn_impl='sdpa')


def write_report(folder, direction, inputs, cot, finished, labels, audit):
    chosen = {sid: {'response': r['selected'] if r['selected'] is not None else r['cot_response']}
              for sid, r in finished.items()}
    raw_predictions = {sid: {'response': r['response']} for sid, r in cot.items()}
    report, records = ev.score(inputs, chosen, labels, audit, direction, 'cot-box')
    cot_report, _ = ev.score(inputs, raw_predictions, labels, audit, direction, 'cot-box')
    for row in records:
        if row['id'] in finished:
            record = finished[row['id']]
            row.update(response=record['cot_response'], finalizer_response=record['finalizer_response'],
                       stage=record['stage'], guard_reason=record['guard_reason'])
    counts = {}
    for r in cot.values():
        reason = r.get('guard_reason') or 'natural_stop'
        counts[reason] = counts.get(reason, 0) + 1
    report.update(protocol=PROTOCOL, updated_at=ev.now(), cot_only=cot_report,
                  finalized=sum(r['stage'] == 'constrained_readout' for r in finished.values()),
                  guard_counts=counts, raw_outputs_preserved=True)
    ev.save_json(folder / 'report.json', report)
    ev.save_rows(folder / 'results.jsonl', records)
    ev.save_rows(folder / 'errors.jsonl', [r for r in records if not r['correct']])
    return report


def worker(args, base):
    import torch
    from swift.infer_engine import InferRequest, RequestConfig
    from swift.utils import seed_everything

    name, checkpoint, directions, _mode = next(x for x in queue.EXPERIMENTS if x[0] == args.worker)
    folder = base / name
    folder.mkdir(exist_ok=True)
    os.environ.update(ev.child_environment(queue.DATA, '0', 1003520))
    seed_everything(42)
    torch.set_num_threads(4)
    datasets = {d: [ev.prepare_row(r, queue.DATA, d, 'cot-box')
                   for r in ev.read_rows(queue.DATA / d / 'eval.jsonl')[0]] for d in directions}
    strict = {d: {r['id']: ev.prepare_row(r, queue.DATA, d, 'strict')
                  for r in ev.read_rows(queue.DATA / d / 'eval.jsonl')[0]} for d in directions}
    print('Validating unchanged checkpoint and test inputs:', name, flush=True)
    config = {'protocol': PROTOCOL, 'checkpoint': checkpoint_identity(name, checkpoint),
              'sources': {d: ev.file_hash(queue.DATA / d / 'eval.jsonl') for d in directions},
              'prepared_hashes': {d: ev.digest(rs) for d, rs in datasets.items()},
              'generation': {'max_tokens': args.max_new_tokens, 'max_seconds': args.max_seconds,
                             'batch_size': 1, 'temperature': 0, 'seed': 42,
                             'native_thinking': False, 'attn_impl': 'sdpa',
                             'repetition_penalty': 1.0, 'readout_max_tokens': 1},
              'probe_size': args.probe_size,
              'source_hashes': {p: ev.file_hash(ROOT / 'scripts' / p) for p in
                               ['run_rules_guarded_cot.py', 'guarded_cot.py',
                                'answer_readout.py', 'run_v7_evaluation.py']}}
    manifest = folder / 'manifest.json'
    if manifest.exists() and read(manifest)['config'] != config:
        raise ValueError('Resume configuration changed; use a new output directory')
    if not manifest.exists():
        ev.save_json(manifest, {'config': config, 'created_at': ev.now()})

    labels = ev.id_map(ev.read_rows(queue.DATA / 'audit/eval_labels.jsonl')[0], 'labels')
    audit = ev.id_map(ev.read_rows(queue.DATA / 'audit/sample_audit.jsonl')[0], 'audit')
    assert all(len(rs) == 2000 and {r['id'] for r in rs} == set(labels) for rs in datasets.values())
    print('Loading model:', checkpoint, flush=True)
    engine = make_engine(checkpoint, args.max_seconds)
    request = RequestConfig(max_tokens=args.max_new_tokens, temperature=0, seed=42, return_details=True)
    final_request = RequestConfig(max_tokens=1, temperature=0, seed=42, return_details=True)
    ev.save_json(folder / 'runtime.json', {'pid': os.getpid(), 'started_at': ev.now(),
                 'torch': torch.__version__, 'native_thinking': engine.template.enable_thinking,
                 'use_chat_template': engine.template.use_chat_template,
                 'cache': engine.generation_config.use_cache,
                 'eos_token_id': engine.generation_config.eos_token_id})

    def infer(payload, mode):
        engine.mode = mode
        started = time.monotonic()
        result = engine.infer([InferRequest(**copy.deepcopy(payload))],
                              request if mode == 'cot' else final_request, use_tqdm=False)[0]
        if isinstance(result, Exception):
            raise result
        return result, time.monotonic() - started

    # Both text and image test inputs pass their fixed-prefix smoke before full inference.
    for phase in ('probe', 'full'):
        for direction in directions:
            cell = folder / direction
            cell.mkdir(exist_ok=True)
            inputs = datasets[direction]
            ev.save_rows(cell / 'eval.input.jsonl', inputs)
            cot = load_records(cell / 'cot.raw.jsonl')
            finished = load_records(cell / 'raw.jsonl')
            expected = {r['id']: ev.digest(inference_payload(r)) for r in inputs}
            for sid, record in cot.items():
                if sid not in expected or record['prompt_sha256'] != expected[sid]:
                    raise ValueError('Saved CoT input mismatch')
            for sid, record in finished.items():
                if (sid not in cot or record['prompt_sha256'] != expected[sid] or
                        record['cot_response'] != cot[sid]['response']):
                    raise ValueError('Saved final answer does not match its CoT')
            todo = inputs[:args.probe_size] if phase == 'probe' else inputs
            for row in todo:
                sid = row['id']
                if sid in finished:
                    continue
                payload = inference_payload(row)
                if sid not in cot:
                    reply, elapsed = infer(payload, 'cot')
                    choice = reply.choices[0]
                    guard = engine.guard
                    raw = {'id': sid, 'qid': row['qid'], 'response': choice.message.content or '',
                           'prompt_sha256': expected[sid], 'elapsed_seconds': elapsed,
                           'finish_reason': choice.finish_reason,
                           'guard_reason': guard.reason if guard else None,
                           'guard_tokens': guard.stopped_after_tokens if guard else None,
                           'usage': dataclasses.asdict(reply.usage),
                           'raw_reply': dataclasses.asdict(reply), 'saved_at': ev.now()}
                    append(cell / 'cot.raw.jsonl', raw)
                    cot[sid] = raw
                raw = cot[sid]
                selected = ev.letter(raw['response'], 'cot-box')
                final_reply = None
                final_payload = None
                final_elapsed = 0
                if selected is None:
                    final_payload = final_readout_payload(strict[direction][sid], raw['response'])
                    final_reply, final_elapsed = infer(final_payload, 'readout')
                    choice = final_reply.choices[0]
                    selected = ev.letter(choice.message.content, 'strict')
                    if selected is None or len(choice.token_ids) != 1:
                        raise RuntimeError('Constrained readout did not generate one answer option')
                record = {'id': sid, 'qid': row['qid'], 'selected': selected,
                          'cot_selected': ev.letter(raw['response'], 'cot-box'),
                          'cot_response': raw['response'], 'guard_reason': raw['guard_reason'],
                          'stage': 'constrained_readout' if final_reply else 'cot',
                          'prompt_sha256': expected[sid],
                          'finalizer_prompt_sha256': ev.digest(final_payload) if final_payload else None,
                          'finalizer_payload': final_payload,
                          'finalizer_response': final_reply.choices[0].message.content if final_reply else None,
                          'finalizer_raw_reply': dataclasses.asdict(final_reply) if final_reply else None,
                          'cot_seconds': raw['elapsed_seconds'], 'finalizer_seconds': final_elapsed,
                          'saved_at': ev.now()}
                append(cell / 'raw.jsonl', record)
                finished[sid] = record
                ev.save_json(cell / 'progress.json', {'phase': phase, 'saved': len(finished),
                             'expected': len(inputs), 'updated_at': ev.now(),
                             'last_cot_seconds': raw['elapsed_seconds'],
                             'last_guard': raw['guard_reason'],
                             'raw_cot_invalid': sum(r['cot_selected'] is None for r in finished.values()),
                             'final_invalid': sum(r['selected'] is None for r in finished.values())})
                print(name, direction, phase, len(finished), '/2000',
                      f"cot={raw['elapsed_seconds']:.2f}s", raw['guard_reason'], record['stage'], flush=True)
                if len(finished) % 100 == 0:
                    write_report(cell, direction, inputs, cot, finished, labels, audit)
            if phase == 'probe':
                probe_ids = {r['id'] for r in inputs[:args.probe_size]}
                assert all(finished[sid]['selected'] in 'ABCD' for sid in probe_ids)
                smoke_inputs = inputs[:args.probe_size]
                report = write_report(cell, direction, smoke_inputs,
                                      {k: v for k, v in cot.items() if k in probe_ids},
                                      {k: v for k, v in finished.items() if k in probe_ids}, labels, audit)
                # Probe scores never occupy the 2,000-row report path.
                (cell / 'report.json').replace(cell / 'probe_report.json')
                (cell / 'results.jsonl').replace(cell / 'probe_results.jsonl')
                (cell / 'errors.jsonl').replace(cell / 'probe_errors.jsonl')
                print('PROBE_PASSED', name, direction, report['valid'],
                      'raw_invalid', report['cot_only']['format_violation'], flush=True)
            else:
                write_report(cell, direction, inputs, cot, finished, labels, audit)
                summarize(base)
    ev.save_json(folder / 'status.json', {'status': 'complete', 'updated_at': ev.now()})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--evaluation-name', default=NAME)
    ap.add_argument('--worker', choices=[x[0] for x in queue.EXPERIMENTS])
    ap.add_argument('--max-new-tokens', type=int, default=65536)
    ap.add_argument('--max-seconds', type=float, default=90)
    ap.add_argument('--probe-size', type=int, default=12)
    args = ap.parse_args()
    if (Path(args.evaluation_name).name != args.evaluation_name or
            args.evaluation_name in ('.', '..') or args.max_new_tokens < 1 or
            args.max_seconds <= 0 or not 1 <= args.probe_size <= 2000):
        ap.error('Invalid output directory or inference limits')
    base = queue.WORK / args.evaluation_name
    base.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args, base)
        return
    queue.training_preflight()
    with ev.run_lock(base / '.queue.lock'):
        status_path = base / 'status.json'
        config = {'protocol': PROTOCOL, 'max_new_tokens': args.max_new_tokens,
                  'max_seconds': args.max_seconds, 'probe_size': args.probe_size}
        previous = read(status_path) if status_path.exists() else {}
        if previous and previous.get('config') != config:
            raise ValueError('Protocol changed; use a new evaluation name')
        status = {'status': 'waiting_for_gpu', 'config': config, 'started_at': previous.get('started_at', ev.now()),
                  'updated_at': ev.now(), 'expected_directions': 8, 'expected_questions': 16000}
        ev.save_json(status_path, status)
        from v7_gpu_resources import reserve
        queue.wait_for_external_gpu_processes(status, status_path, 30)
        with reserve(queue.LEDGER, '0', str(base), 30000, poll=30) as lease:
            lease.bind(os.getpid())
            for name, _cp, directions, _mode in queue.EXPERIMENTS:
                if all((base / name / d / 'report.json').exists() and
                       read(base / name / d / 'report.json').get('matched') == 2000 for d in directions):
                    continue
                queue.wait_for_external_gpu_processes(status, status_path, 30)
                status.update(status='running', experiment=name, updated_at=ev.now())
                ev.save_json(status_path, status)
                out = base / name
                out.mkdir(exist_ok=True)
                cmd = [sys.executable, '-u', str(Path(__file__).resolve()),
                       '--evaluation-name', args.evaluation_name, '--worker', name,
                       '--max-new-tokens', str(args.max_new_tokens),
                       '--max-seconds', str(args.max_seconds), '--probe-size', str(args.probe_size)]
                env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'PYTHONUNBUFFERED': '1',
                       'HF_HOME': str(ROOT / '.runtime/cache/huggingface')}
                with (out / 'controller.log').open('a') as log:
                    result = subprocess.run(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                            stdout=log, stderr=subprocess.STDOUT, check=False)
                summarize(base)
                if result.returncode:
                    status.update(status='failed', returncode=result.returncode, updated_at=ev.now())
                    ev.save_json(status_path, status)
                    raise RuntimeError(f'{name} failed; inspect {out / "controller.log"}')
        reports = summarize(base)
        complete = len(reports) == 8 and all(r['matched'] == 2000 for r in reports)
        status.update(status='complete' if complete else 'incomplete', updated_at=ev.now(),
                      completed_directions=sum(r['matched'] == 2000 for r in reports))
        ev.save_json(status_path, status)


if __name__ == '__main__':
    main()
