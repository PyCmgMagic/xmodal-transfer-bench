#!/usr/bin/env python3
"""Train and evaluate the approved mixed-supervision candidate for one epoch."""
import argparse
import copy
import dataclasses
import hashlib
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import run_v7_evaluation as ev
from run_v7_full_experiment import ROOT, MODELS, environment, rows, read, save, sha, train_command, final_checkpoint
from v7_gpu_resources import reserve


def semantic_equal(response, target, task):
    """Conservative exact match; do not treat a paraphrase as a factual error."""
    normalize = lambda s: re.sub(r'\s+', ' ', s.strip()).rstrip('.').casefold()
    if task in ('record_code_recall', 'practice_application'):
        return response.strip() == target
    return normalize(response) == normalize(target)


def prepare(source, control, out):
    candidate = read(source / 'manifest.json')
    assert sha(source / 'train.jsonl') == candidate['train_sha256']
    assert sha(control / 'train.jsonl') == candidate['source_train_sha256']
    assert len(rows(source / 'train.jsonl')) == 780
    reference = read(control.parent / 'updates_lr3e5/manifest.json')
    assert reference['epochs'] == 1 and reference['training_settings'] == {
        'gradient_accumulation_steps': 1, 'learning_rate': 3e-5}
    expected = {'candidate_train_sha256': candidate['train_sha256'], 'epochs': 1,
        'learning_rate': 3e-5, 'gradient_accumulation_steps': 1,
        'source_candidate': str(source), 'source_control': str(control),
        'evaluation_sha256': sha(control / 'eval.jsonl'), 'script_sha256': sha(Path(__file__))}
    if (out / 'manifest.json').exists():
        assert read(out / 'manifest.json')['configuration'] == expected
        assert sha(out / 'train.jsonl') == candidate['train_sha256']
        return
    out.mkdir(parents=True, exist_ok=True)
    (out / 'train.jsonl').write_bytes((source / 'train.jsonl').read_bytes())
    (out / 'eval.jsonl').write_bytes((control / 'eval.jsonl').read_bytes())
    save(out / 'manifest.json', {'configuration': expected, 'candidate_manifest': candidate,
        'comparison': 'Same 12 rules, 780 rows in identical order, observations, base model, optimizer settings, and one epoch as updates_lr3e5. Changes half of training formats/targets.',
        'model': str(MODELS['qwen3']), 'new_images': False, 'cot': False, 'created_at': ev.now()})


def build_probes(source, control):
    training = rows(source / 'train.jsonl')
    exposure = {r['id'].removesuffix(':semantic'): r['supervision_format'] for r in training}
    result = []
    # Reuse the exact fixed MCQ and original-question controls from the previous
    # experiment; do not select examples based on model answers.
    for r in rows(control / 'probes.jsonl'):
        result.append({**r, 'answer_format': 'mcq', 'training_format': exposure.get(r['id'], 'original_eval')})
    grouped = defaultdict(list)
    for r in training:
        if r['supervision_format'] == 'semantic':
            grouped[(r['qid'], r['training_task'])].append(r)
    for (qid, task), group in sorted(grouped.items()):
        chosen = sorted(group, key=lambda r: hashlib.sha256(('semantic-mastery:42:' + r['id']).encode()).hexdigest())[:2]
        for r in chosen:
            result.append({'condition': 'semantic_' + task, 'id': r['id'], 'qid': qid,
                'task': task, 'gold': r['messages'][-1]['content'], 'messages': copy.deepcopy(r['messages'][:-1]),
                'answer_format': 'semantic', 'training_format': 'semantic'})
    assert len(result) == 240
    assert len({(r['condition'], r['id']) for r in result}) == 240
    return result


def first_token(logprobs):
    if dataclasses.is_dataclass(logprobs):
        logprobs = dataclasses.asdict(logprobs)
    if isinstance(logprobs, dict):
        content = logprobs.get('content') or []
        if content:
            entry = content[0]
            return entry.get('token') if isinstance(entry, dict) else getattr(entry, 'token', None)
    return None


def evaluate(cp, source, control, out, gpu, work, data):
    probes = build_probes(source, control)
    ev.save_rows(out / 'probes.jsonl', probes)
    with reserve(work, gpu, str(out) + ':evaluation', 22000) as reservation:
        reservation.bind(os.getpid())
        os.environ.update(environment(gpu, data))
        import torch
        from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
        from swift.utils import seed_everything
        seed_everything(42)
        engine = TransformersEngine(str(MODELS['qwen3']), adapters=[str(cp)], torch_dtype=torch.bfloat16,
                                   device_map='cuda:0', max_batch_size=16, attn_impl='sdpa')
        engine.template.enable_thinking = False
        save(out / 'state.json', {'status': 'evaluating', 'checkpoint': str(cp), 'pid': os.getpid()})
        raw = out / 'predictions.jsonl'
        results = ev.read_rows(raw, allow_partial_tail=True)[0] if raw.exists() else []
        probe_index = {(r['condition'], r['id']): r for r in probes}
        seen = set()
        for r in results:
            key = (r['condition'], r['id'])
            assert key not in seen and r['prompt_sha256'] == ev.digest(probe_index[key]['messages'])
            seen.add(key)
        if raw.exists() and raw.read_bytes() and not raw.read_bytes().endswith(b'\n'):
            ev.save_rows(raw, results)
        with raw.open('a') as stream:
            for answer_format in ('mcq', 'semantic'):
                pending = [r for r in probes if r['answer_format'] == answer_format and (r['condition'], r['id']) not in seen]
                config = RequestConfig(max_tokens=32 if answer_format == 'mcq' else 96,
                    temperature=0, seed=42, logprobs=answer_format == 'mcq', top_logprobs=4 if answer_format == 'mcq' else None)
                for start in range(0, len(pending), 16):
                    batch = pending[start:start + 16]
                    responses = engine.infer([InferRequest(messages=copy.deepcopy(r['messages'])) for r in batch], config, use_tqdm=False)
                    assert len(responses) == len(batch)
                    for r, response in zip(batch, responses):
                        if isinstance(response, Exception):
                            raise response
                        choice = response.choices[0]
                        text = choice.message.content or ''
                        token = first_token(choice.logprobs)
                        selected = ev.letter(text) if answer_format == 'mcq' else None
                        correct = selected == r['gold'] if answer_format == 'mcq' else semantic_equal(text, r['gold'], r['task'])
                        record = {k: r[k] for k in ('condition', 'id', 'qid', 'gold', 'answer_format', 'training_format')}
                        record.update(response=text, parsed_answer=selected, correct=correct,
                            first_generated_token=token,
                            first_token_correct=token == r['gold'] if answer_format == 'mcq' and token is not None else None,
                            finish_reason=choice.finish_reason, prompt_sha256=ev.digest(r['messages']), saved_at=ev.now())
                        results.append(record)
                        stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                        stream.flush()
                        os.fsync(stream.fileno())
                    print(answer_format, 'evaluated', len(results), '/', len(probes), flush=True)
    metrics = {}
    for condition in sorted({r['condition'] for r in results}):
        group = [r for r in results if r['condition'] == condition]
        metrics[condition] = {'n': len(group), 'correct': sum(r['correct'] for r in group),
            'metric': 'strict_option_accuracy' if group[0]['answer_format'] == 'mcq' else 'conservative_exact_match',
            'format_errors': sum(r['parsed_answer'] is None for r in group) if group[0]['answer_format'] == 'mcq' else None,
            'first_token_recorded': sum(r['first_generated_token'] is not None for r in group),
            'first_token_correct': sum(r['first_token_correct'] is True for r in group)}
    report = {'status': 'complete', 'epochs': 1, 'metrics': metrics, 'checkpoint': ev.checkpoint(cp),
        'semantic_metric_limit': 'Nonmatching factual paraphrases are not automatically factual errors; retain raw responses for inspection.',
        'finished_at': ev.now()}
    save(out / 'report.json', report)
    save(out / 'state.json', {'status': 'complete', 'finished_at': ev.now()})
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--control', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--gpu', default='0')
    a = ap.parse_args()
    source, control, out = a.source.resolve(), a.control.resolve(), a.out.resolve()
    work, data = ROOT / 'runs/v7_full_experiment', ROOT / 'rules_repaired_single_image_v7'
    prepare(source, control, out)
    with ev.run_lock(out / '.lock'):
        cp = final_checkpoint(out / 'checkpoint', 1)
        if not cp:
            with reserve(work, a.gpu, str(out) + ':training', 16000) as reservation:
                cmd = train_command('qwen3', 'sft', out / 'train.jsonl', out / 'checkpoint', 1)
                cmd[cmd.index('--gradient_accumulation_steps') + 1] = '1'
                cmd[cmd.index('--learning_rate') + 1] = '3e-5'
                with (out / 'train.log').open('a') as log:
                    child = subprocess.Popen(cmd, cwd=ROOT, env=environment(a.gpu, data), stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    reservation.bind(child.pid)
                    save(out / 'state.json', {'status': 'training', 'pid': child.pid, 'command': cmd, 'started_at': ev.now()})
                    code = child.wait()
                if code:
                    save(out / 'state.json', {'status': 'failed', 'returncode': code})
                    raise RuntimeError(f'semantic training failed: {code}')
            cp = final_checkpoint(out / 'checkpoint', 1)
            assert cp
        evaluate(cp, source, control, out, a.gpu, work, data)


if __name__ == '__main__':
    main()
