#!/usr/bin/env python3
"""Evaluate paired 100-question T2T/T2V samples on the multiform10 model."""
import argparse
import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import time

import run_rules1000_10x_text_evaluation as queue
import run_rules_guarded_cot as guarded
import run_v7_evaluation as ev
from guarded_cot import final_readout_payload, inference_payload


ROOT = Path(__file__).resolve().parents[1]
OUT_NAME = 'evaluation_multiform10_paired100_20261103'
ARM = 'text_multiform10'
CHECKPOINT = next(x[1] for x in queue.EXPERIMENTS if x[0] == ARM)
DIRECTIONS = ('T2T', 'T2V')
SAMPLE_RULES = 50
QUESTIONS_PER_DIRECTION = SAMPLE_RULES * 2
SAMPLE_SALT = 'multiform10-paired100-v1'


def select_paired_sample(raw_by_direction, labels):
    """Choose 50 hash-randomized rules and retain direct+diagnosis pairs."""
    maps = {d: ev.id_map(raw_by_direction[d], d) for d in DIRECTIONS}
    common = set(maps['T2T']) & set(maps['T2V'])
    if set(maps['T2T']) != set(maps['T2V']):
        raise ValueError('T2T/T2V source IDs differ; paired comparison is not possible')
    qids = {}
    for sid in common:
        row = maps['T2T'][sid]
        qid = row.get('qid')
        if not isinstance(qid, str) or sid not in labels:
            raise ValueError(f'Missing qid or audit label for {sid}')
        qids.setdefault(qid, []).append(sid)
    if any(len(ids) != 2 for ids in qids.values()):
        raise ValueError('Expected one direct and one diagnosis item for every rule')
    if any({labels[sid]['question_type'] for sid in ids} != {'direct', 'diagnosis'}
           for ids in qids.values()):
        raise ValueError('Question-type pairing does not match the audit labels')
    selected_qids = sorted(qids, key=lambda q: hashlib.sha256(
        (SAMPLE_SALT + ':' + q).encode()).hexdigest())[:SAMPLE_RULES]
    selected_ids = [sid for qid in selected_qids for sid in sorted(
        qids[qid], key=lambda sid: labels[sid]['question_type'])]
    return {d: [maps[d][sid] for sid in selected_ids] for d in DIRECTIONS}, selected_qids


def make_config(datasets, qids, args):
    return {
        'protocol': guarded.PROTOCOL,
        'experiment': ARM,
        'checkpoint': guarded.checkpoint_identity(ARM, CHECKPOINT),
        'directions': list(DIRECTIONS),
        'sample': {'method': 'sha256-ranked qid; paired direct+diagnosis questions',
                   'salt': SAMPLE_SALT, 'rules': SAMPLE_RULES,
                   'questions_per_direction': QUESTIONS_PER_DIRECTION,
                   'qids': qids},
        'sources': {d: ev.file_hash(queue.DATA / d / 'eval.jsonl') for d in DIRECTIONS},
        'prepared_hashes': {d: ev.digest(rows) for d, rows in datasets.items()},
        'generation': {'max_tokens': args.max_new_tokens, 'max_seconds': args.max_seconds,
                       'batch_size': 1, 'temperature': 0, 'seed': 42,
                       'native_thinking': False, 'attn_impl': 'sdpa',
                       'repetition_penalty': 1.0, 'readout_max_tokens': 1},
        'sources_sha256': {name: ev.file_hash(ROOT / 'scripts' / name) for name in
                           ['run_rules_multiform10_paired100.py', 'run_rules_guarded_cot.py',
                            'guarded_cot.py', 'answer_readout.py', 'run_v7_evaluation.py']},
    }


def summarize(base):
    reports = {}
    for direction in DIRECTIONS:
        path = base / direction / 'report.json'
        if path.exists():
            reports[direction] = json.loads(path.read_text())
    ev.save_json(base / 'summary.json', {
        'experiment': ARM, 'updated_at': ev.now(), 'reports': reports,
        'rules': SAMPLE_RULES, 'questions_per_direction': QUESTIONS_PER_DIRECTION,
        'protocol': guarded.PROTOCOL})
    lines = ['# 每规则 10 条定义模型：配对 100 题 CoT 评测', '',
             '固定哈希抽取 50 条规则，每条 direct/diagnosis 各一题，T2T/T2V 逐题配对。', '',
             '模型：Qwen3-VL-4B-Instruct，TEXT multiform10 全量 SFT 10 epoch，checkpoint-6250。', '',
             '显式 CoT，正常 Instruct 模板，boxed 终答；65536 token 上限，约 90 秒与重复停止保护。',
             '原始 CoT 与未能形成有效终答时的同模型受约束补选分别计分，未提供正确规则或答案。', '',
             '| 方向 | 已答/总数 | 原始CoT正确/已答 | 原始CoT无效 | 补选后正确/已答 | 补选后无效 |',
             '|---|---:|---:|---:|---:|---:|']
    for direction, r in reports.items():
        cot = r['cot_only']
        n = r['matched']
        ci = cot['empty_prediction'] + cot['format_violation']
        fi = r['empty_prediction'] + r['format_violation']
        if n:
            lines.append(f"| {direction} | {n}/{r['total']} | {cot['correct']}/{n} "
                         f"({cot['correct']/n:.2%}) | {ci}/{n} ({ci/n:.2%}) | "
                         f"{r['correct']}/{n} ({r['correct']/n:.2%}) | {fi}/{n} ({fi/n:.2%}) |")
    lines += ['', '准确率包含无效回答；进行中的百分比以已答为分母，完成后每方向分母为 100。',
              '样本来自 50 条规则，不代表 1000 规则全量成绩。', '']
    (base / 'report_zh.md').write_text('\n'.join(lines))


def run(args):
    base = queue.WORK / args.evaluation_name
    base.mkdir(parents=True, exist_ok=True)
    os.environ.update(ev.child_environment(queue.DATA, '0', 1003520))
    os.environ['HF_HUB_OFFLINE'] = '1'

    labels = ev.id_map(ev.read_rows(queue.DATA / 'audit/eval_labels.jsonl')[0], 'labels')
    audit = ev.id_map(ev.read_rows(queue.DATA / 'audit/sample_audit.jsonl')[0], 'audit')
    source_rows = {d: ev.read_rows(queue.DATA / d / 'eval.jsonl')[0] for d in DIRECTIONS}
    selected, qids = select_paired_sample(source_rows, labels)
    datasets = {d: [ev.prepare_row(row, queue.DATA, d, 'cot-box') for row in selected[d]]
                for d in DIRECTIONS}
    strict = {d: {row['id']: ev.prepare_row(row, queue.DATA, d, 'strict')
                  for row in selected[d]} for d in DIRECTIONS}
    assert all(len(datasets[d]) == QUESTIONS_PER_DIRECTION for d in DIRECTIONS)
    assert [r['id'] for r in datasets['T2T']] == [r['id'] for r in datasets['T2V']]

    config = make_config(datasets, qids, args)
    manifest_path = base / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())['config'] != config:
        raise ValueError('Resume configuration changed; choose a fresh evaluation name')
    if not manifest_path.exists():
        ev.save_json(manifest_path, {'config': config, 'created_at': ev.now()})
    snapshots = base / 'source_snapshots'
    snapshots.mkdir(exist_ok=True)
    for name in config['sources_sha256']:
        (snapshots / name).write_bytes((ROOT / 'scripts' / name).read_bytes())
    for d in DIRECTIONS:
        ev.save_rows(base / d / 'eval.input.jsonl', datasets[d])

    print(f'Selected {SAMPLE_RULES} paired rules, {QUESTIONS_PER_DIRECTION} questions each in T2T/T2V.', flush=True)
    print('Loading model:', CHECKPOINT, flush=True)
    import torch
    from swift.infer_engine import InferRequest, RequestConfig
    from swift.utils import seed_everything
    seed_everything(42)
    torch.set_num_threads(4)
    engine = guarded.make_engine(CHECKPOINT, args.max_seconds)
    cot_request = RequestConfig(max_tokens=args.max_new_tokens, temperature=0, seed=42,
                                return_details=True)
    final_request = RequestConfig(max_tokens=1, temperature=0, seed=42,
                                  return_details=True)
    ev.save_json(base / 'runtime.json', {
        'pid': os.getpid(), 'started_at': ev.now(), 'torch': torch.__version__,
        'native_thinking': engine.template.enable_thinking,
        'use_chat_template': engine.template.use_chat_template,
        'checkpoint': str(CHECKPOINT),
    })

    def infer(payload, mode):
        engine.mode = mode
        start = time.monotonic()
        reply = engine.infer([InferRequest(**copy.deepcopy(payload))],
                             cot_request if mode == 'cot' else final_request,
                             use_tqdm=False)[0]
        if isinstance(reply, Exception):
            raise reply
        return reply, time.monotonic() - start

    for direction in DIRECTIONS:
        cell = base / direction
        cell.mkdir(parents=True, exist_ok=True)
        inputs = datasets[direction]
        cot = guarded.load_records(cell / 'cot.raw.jsonl')
        finished = guarded.load_records(cell / 'raw.jsonl')
        expected = {r['id']: ev.digest(inference_payload(r)) for r in inputs}
        for sid, row in cot.items():
            if sid not in expected or row['prompt_sha256'] != expected[sid]:
                raise ValueError(f'Saved CoT input mismatch in {direction}: {sid}')
        for sid, row in finished.items():
            if (sid not in cot or row['prompt_sha256'] != expected[sid] or
                    row['cot_response'] != cot[sid]['response']):
                raise ValueError(f'Saved answer mismatch in {direction}: {sid}')

        for row in inputs:
            sid = row['id']
            if sid in finished:
                continue
            if sid not in cot:
                payload = inference_payload(row)
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
                guarded.append(cell / 'cot.raw.jsonl', raw)
                cot[sid] = raw
            raw = cot[sid]
            selected_letter = ev.letter(raw['response'], 'cot-box')
            final_reply = final_payload = None
            final_elapsed = 0.0
            if selected_letter is None:
                final_payload = final_readout_payload(strict[direction][sid], raw['response'])
                final_reply, final_elapsed = infer(final_payload, 'readout')
                choice = final_reply.choices[0]
                selected_letter = ev.letter(choice.message.content, 'strict')
                if selected_letter not in set('ABCD') or len(choice.token_ids) != 1:
                    raise RuntimeError('Constrained readout failed to produce exactly one option')
            record = {'id': sid, 'qid': row['qid'], 'selected': selected_letter,
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
            guarded.append(cell / 'raw.jsonl', record)
            finished[sid] = record
            if len(finished) % 10 == 0 or len(finished) == QUESTIONS_PER_DIRECTION:
                report = guarded.write_report(cell, direction, inputs, cot, finished, labels, audit)
                ev.save_json(cell / 'progress.json', {
                    'status': 'complete' if len(finished) == QUESTIONS_PER_DIRECTION else 'running',
                    'saved': len(finished), 'expected': QUESTIONS_PER_DIRECTION,
                    'raw_cot_invalid': report['cot_only']['format_violation'],
                    'final_invalid': report['empty_prediction'] + report['format_violation'],
                    'updated_at': ev.now()})
                summarize(base)
            print(ARM, direction, len(finished), '/', QUESTIONS_PER_DIRECTION,
                  f"cot={raw['elapsed_seconds']:.2f}s", raw['guard_reason'], record['stage'], flush=True)
        report = guarded.write_report(cell, direction, inputs, cot, finished, labels, audit)
        ev.save_json(cell / 'progress.json', {
            'status': 'complete', 'saved': len(finished), 'expected': QUESTIONS_PER_DIRECTION,
            'raw_cot_invalid': report['cot_only']['format_violation'],
            'final_invalid': report['empty_prediction'] + report['format_violation'],
            'updated_at': ev.now()})
        summarize(base)

    ev.save_json(base / 'status.json', {
        'status': 'complete', 'experiment': ARM, 'directions': list(DIRECTIONS),
        'questions_per_direction': QUESTIONS_PER_DIRECTION, 'rules': SAMPLE_RULES,
        'updated_at': ev.now()})
    print('PAIRED_100_COMPLETE', base, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-name', default=OUT_NAME)
    parser.add_argument('--max-new-tokens', type=int, default=65536)
    parser.add_argument('--max-seconds', type=float, default=90)
    args = parser.parse_args()
    if Path(args.evaluation_name).name != args.evaluation_name:
        parser.error('evaluation-name must be a single directory name')
    if args.max_new_tokens < 1 or args.max_seconds <= 0:
        parser.error('token and time limits must be positive')
    queue.training_preflight()
    base = queue.WORK / args.evaluation_name
    base.mkdir(parents=True, exist_ok=True)
    from v7_gpu_resources import reserve
    with ev.run_lock(base / '.run.lock'):
        ev.save_json(base / 'status.json', {
            'status': 'waiting_for_gpu', 'experiment': ARM,
            'questions_per_direction': QUESTIONS_PER_DIRECTION,
            'rules': SAMPLE_RULES, 'updated_at': ev.now()})
        with reserve(queue.LEDGER, '0', str(base), 30000, poll=30) as lease:
            lease.bind(os.getpid())
            ev.save_json(base / 'status.json', {
                'status': 'running', 'experiment': ARM,
                'questions_per_direction': QUESTIONS_PER_DIRECTION,
                'rules': SAMPLE_RULES, 'updated_at': ev.now()})
            run(args)


if __name__ == '__main__':
    main()
