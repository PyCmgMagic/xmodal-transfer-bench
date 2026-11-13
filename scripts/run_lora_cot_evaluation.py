#!/usr/bin/env python3
"""Durable full-data prompted-CoT evaluation of the final learning adapters."""
import argparse
import copy
import csv
import dataclasses
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import run_v7_evaluation as ev
from v7_gpu_resources import alive, reserve

ROOT = Path(__file__).resolve().parents[1]
SYSTEM = ('Solve the multiple-choice question using your learned knowledge and the provided test input. '
          'Reason step by step, checking the relevant facts or rule conditions. Give a concise explanation '
          'within 200 words, then end with exactly FINAL: followed by one uppercase letter A, B, C, or D.')
SUFFIX = ('\n\nReason step by step in a concise explanation (at most 200 words). '
          'End with exactly FINAL: followed by one uppercase letter A, B, C, or D.')


def rows(path):
    return ev.read_rows(path)[0]


def final_letter(response):
    if not isinstance(response, str):
        return None
    text = response.strip()
    if text.count('<think>') != text.count('</think>'):
        return None
    text = text.rsplit('</think>', 1)[-1].strip()
    markers = re.findall(r'\bFINAL\s*[:：]\s*([ABCD])\b', text)
    match = re.search(r'\bFINAL\s*[:：]\s*([ABCD])\s*\Z', text)
    return match[1] if match and len(markers) == 1 else None


def cot_input(row, root, direction):
    prepared = ev.prepare_row(row, root, direction, 'strict')
    text = prepared['messages'][-1]['content']
    assert text.endswith(ev.SUFFIX)
    prepared['messages'] = [{'role': 'system', 'content': SYSTEM},
                            {'role': 'user', 'content': text[:-len(ev.SUFFIX)].rstrip() + SUFFIX}]
    return prepared


def metrics(scored):
    n = len(scored)
    correct = sum(r['correct'] for r in scored)
    valid = sum(r['predicted'] is not None for r in scored)
    return {'total': n, 'correct': correct, 'valid': valid, 'invalid': n-valid,
            'accuracy': correct/n if n else None,
            'accuracy_on_valid_only': correct/valid if valid else None,
            'truncated': sum(r['finish_reason'] == 'length' for r in scored),
            'with_explanation': sum(r['has_explanation'] for r in scored)}


def summarize(work, manifest):
    summary = []
    for job in manifest['jobs']:
        for direction in job['directions']:
            p = work/'jobs'/job['name']/direction/'report.json'
            if p.exists():
                summary.append(json.loads(p.read_text()))
    ev.save_json(work/'summary.json', {'updated_at': ev.now(), 'cells_complete': len(summary),
                 'cells_expected': 20, 'predictions_expected': 42180, 'cells': summary})
    columns = ['dataset', 'objective', 'learning_modality', 'direction', 'total', 'correct',
               'accuracy', 'valid', 'invalid', 'truncated', 'with_explanation']
    tmp = work/'summary.csv.tmp'
    with tmp.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader(); writer.writerows(summary)
    tmp.replace(work/'summary.csv')
    return summary


def prepare(args):
    work = args.work
    p = work/'manifest.json'
    if p.exists():
        manifest = json.loads(p.read_text())
        assert manifest['training_run'] == str(args.training_run)
        assert manifest['generation']['batch_size'] == args.batch_size
        for path, sha in manifest['sources'].items():
            assert ev.file_hash(path) == sha, path
        for path, sha in manifest['frozen_files'].items():
            assert ev.file_hash(work/path) == sha, path
        return manifest
    source = json.loads((args.training_run/'manifest.json').read_text())
    status = json.loads((args.training_run/'status.json').read_text())
    assert status['status'] == 'complete' and status['completed'] == 8
    jobs, sources, frozen, images = [], {}, {}, {}
    sources[str(args.training_run/'manifest.json')] = ev.file_hash(args.training_run/'manifest.json')
    sources[str(args.training_run/'status.json')] = ev.file_hash(args.training_run/'status.json')
    prepared = {}
    for dataset, rel, count in [('rules', 'data/rules_repaired_single_image_v7', 2000),
                                 ('entities', 'data/entity1020/merged_1109', 2218)]:
        root = ROOT/rel
        labels_path, audit_path = root/'audit/eval_labels.jsonl', root/'audit/sample_audit.jsonl'
        labels, audit = ev.id_map(rows(labels_path), labels_path), ev.id_map(rows(audit_path), audit_path)
        assert len(labels) == len(audit) == count
        for path in (labels_path, audit_path):
            sources[str(path)] = ev.file_hash(path)
        for direction in ('T2T', 'T2V', 'V2T', 'V2V'):
            path = root/direction/'eval.jsonl'
            sources[str(path)] = ev.file_hash(path)
            inputs = [cot_input(r, root, direction) for r in rows(path)]
            assert len(inputs) == count and {r['id'] for r in inputs} == set(labels)
            assert len({ev.prompt_key(r) for r in inputs}) == count
            for r in inputs:
                for image in r.get('images', []):
                    if image not in images:
                        images[image] = ev.file_hash(image)
            prepared[dataset, direction] = inputs
            target = work/'data'/dataset/f'{direction}.jsonl'
            ev.save_rows(target, inputs)
            frozen[str(target.relative_to(work))] = ev.file_hash(target)
        assert prepared[dataset, 'T2T'] == prepared[dataset, 'V2T']
        assert prepared[dataset, 'T2V'] == prepared[dataset, 'V2V']
        for src in (labels_path, audit_path):
            target = work/'data'/dataset/src.name
            shutil.copy2(src, target)
            frozen[str(target.relative_to(work))] = ev.file_hash(target)
    for trained in source['jobs']:
        state_path = Path(trained['output'])/'state.json'
        state = json.loads(state_path.read_text())
        assert state['status'] == 'complete' and state['epoch'] == 5 and state['lora_updated']
        cp = ev.checkpoint(state['checkpoint'])
        assert cp['sha256']['adapter_model.safetensors'] == state['adapter_sha256']
        for name, sha in cp['sha256'].items():
            sources[str(Path(cp['path'])/name)] = sha
        sources[str(state_path)] = ev.file_hash(state_path)
        jobs.append({'name': trained['name'], 'dataset': trained['dataset_name'],
                     'objective': trained['objective'], 'learning_modality': trained['modality'],
                     'adapter': cp, 'directions': ['T2T', 'T2V'] if trained['modality'] == 'text' else ['V2T', 'V2V']})
    for dataset in ('rules', 'entities'):
        jobs.append({'name': dataset+'_base', 'dataset': dataset, 'objective': 'base',
                     'learning_modality': 'none', 'adapter': None, 'directions': ['T2T', 'T2V']})
    for filename in ('config.json', 'tokenizer_config.json', 'preprocessor_config.json', 'download_complete.json'):
        path = Path(source['config']['model'])/filename
        if path.exists():
            sources[str(path)] = ev.file_hash(path)
    for name in ('run_lora_cot_evaluation.py', 'run_v7_evaluation.py', 'v7_gpu_resources.py'):
        path = ROOT/'scripts'/name
        target = work/'source_snapshots'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        sources[str(path)] = ev.file_hash(path)
        frozen[str(target.relative_to(work))] = ev.file_hash(target)
    ev.save_json(work/'eval_image_hashes.json', images)
    frozen['eval_image_hashes.json'] = ev.file_hash(work/'eval_image_hashes.json')
    manifest = {'created_at': ev.now(), 'training_run': str(args.training_run),
                'model': source['config']['model'], 'jobs': jobs, 'sources': sources,
                'frozen_files': frozen, 'eval_images': len(images), 'predictions_expected': 42180,
                'trained_predictions_expected': 33744, 'baseline_predictions_expected': 8436,
                'generation': {'backend': 'ms-swift TransformersEngine', 'batch_size': args.batch_size,
                               'temperature': 0, 'seed': 42, 'max_tokens': 1536, 'max_pixels': 1003520,
                               'attn_impl': 'sdpa', 'dtype': 'bfloat16', 'use_chat_template': True,
                               'native_thinking': False, 'explicit_cot': True,
                               'reasoning_mode': 'Prompted CoT for Qwen3-VL-Instruct; no forced <think> prefix.'},
                'system': SYSTEM, 'suffix': SUFFIX,
                'scoring': 'Single terminal uppercase FINAL: A/B/C/D; invalid outputs count as incorrect; no answer from rationale.',
                'checkpoint_selection': 'Frozen final epoch 5, no test-based selection.',
                'baseline_policy': 'Fresh same-protocol base model; only two distinct test modalities per dataset.',
                'versions': {p: importlib.metadata.version(p) for p in ('ms-swift','torch','transformers','peft')}}
    ev.save_json(p, manifest)
    ev.save_json(work/'status.json', {'status': 'prepared', 'total_jobs': len(jobs),
                 'expected_predictions': 42180, 'created_at': ev.now()})
    return manifest


def worker(args, manifest, job):
    import torch
    from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
    from swift.template import get_template
    from swift.utils import seed_everything

    class ChatEvaluationEngine(TransformersEngine):
        def _get_template(self, processor, template_type=None):
            return get_template(processor, template_type=template_type, use_chat_template=True,
                                enable_thinking=False, max_length=8192, max_pixels=1003520)

    seed_everything(42)
    out = args.work/('smoke' if args.smoke else 'jobs')/job['name']
    out.mkdir(parents=True, exist_ok=True)
    with ev.run_lock(out/'.lock'):
        ev.save_json(out/'state.json', {'status': 'running', 'pid': os.getpid(), 'started_at': ev.now()})
        try:
            engine = ChatEvaluationEngine(manifest['model'],
                      adapters=[job['adapter']['path']] if job['adapter'] else [],
                      torch_dtype=torch.bfloat16, device_map='cuda:0', max_batch_size=args.batch_size,
                      attn_impl='sdpa')
            assert engine.template.use_chat_template and engine.template.template_meta.support_system
            assert not engine.template.enable_thinking
            ev.save_json(out/'runtime.json', {'use_chat_template': engine.template.use_chat_template,
                         'native_thinking': engine.template.enable_thinking,
                         'template_type': engine.template.template_meta.template_type,
                         'template_is_thinking': engine.template.template_meta.is_thinking,
                         'explicit_cot': True, 'adapter': job['adapter'], 'updated_at': ev.now()})
            request = RequestConfig(max_tokens=1536, temperature=0, seed=42, return_details=True)
            for direction in job['directions']:
                inputs = rows(args.work/'data'/job['dataset']/f'{direction}.jsonl')
                if args.smoke:
                    inputs = inputs[:1]
                payloads = {r['id']: {k: r[k] for k in ('messages','images') if k in r} for r in inputs}
                folder = out/direction; folder.mkdir(parents=True, exist_ok=True)
                raw = folder/'raw.jsonl'
                saved = ev.read_rows(raw, allow_partial_tail=True)[0] if raw.exists() else []
                seen = {}
                for r in saved:
                    assert r['id'] in payloads and r['id'] not in seen
                    assert r['prompt_sha256'] == ev.digest(payloads[r['id']])
                    assert r['adapter'] == job['adapter']
                    seen[r['id']] = r
                if raw.exists() and raw.read_bytes() and not raw.read_bytes().endswith(b'\n'):
                    ev.save_rows(raw, saved)
                pending = [r for r in inputs if r['id'] not in seen]
                print(job['name'], direction, 'saved', len(seen), 'pending', len(pending), flush=True)
                with raw.open('a', encoding='utf8') as f:
                    for start in range(0, len(pending), args.batch_size):
                        batch = pending[start:start+args.batch_size]
                        started = time.monotonic()
                        results = engine.infer([InferRequest(**copy.deepcopy(payloads[r['id']])) for r in batch],
                                               request, use_tqdm=False)
                        assert len(results) == len(batch)
                        for r, res in zip(batch, results):
                            if isinstance(res, Exception):
                                raise res
                            choice = res.choices[0]
                            record = {'id': r['id'], 'qid': r['qid'], 'response': choice.message.content or '',
                                      'reasoning_content': getattr(choice.message, 'reasoning_content', None),
                                      'finish_reason': choice.finish_reason, 'usage': dataclasses.asdict(res.usage),
                                      'completion_token_ids': choice.token_ids,
                                      'prompt_token_ids': res.prompt_token_ids,
                                      'images_size': res.images_size,
                                      'prompt_sha256': ev.digest(payloads[r['id']]),
                                      'adapter': job['adapter'], 'saved_at': ev.now()}
                            f.write(json.dumps(record, ensure_ascii=False)+'\n')
                            seen[r['id']] = record
                        f.flush(); os.fsync(f.fileno())
                        ev.save_json(folder/'progress.json', {'saved': len(seen), 'total': len(inputs),
                                     'batch_seconds': time.monotonic()-started, 'updated_at': ev.now()})
                        print(job['name'], direction, len(seen), '/', len(inputs), flush=True)
                labels = ev.id_map(rows(args.work/'data'/job['dataset']/'eval_labels.jsonl'), 'labels')
                audit = ev.id_map(rows(args.work/'data'/job['dataset']/'sample_audit.jsonl'), 'audit')
                scored = []
                for row in inputs:
                    sid = row['id']; r = seen[sid]; answer = final_letter(r['response'])
                    rationale = r['response'].rsplit('FINAL',1)[0].strip()
                    scored.append({'id': sid, 'qid': row['qid'], 'gold': labels[sid]['answer'],
                                   'predicted': answer, 'correct': answer == labels[sid]['answer'],
                                   'response': r['response'], 'finish_reason': r['finish_reason'],
                                   'has_explanation': len(rationale) >= 20,
                                   'question_type': labels[sid].get('question_type', labels[sid].get('probe_id')),
                                   'evaluation_group': labels[sid].get('evaluation_group', 'entity_fact'),
                                   'entity_kind': audit[sid].get('entity_kind', 'unknown'),
                                   'bias_watchlist': labels[sid].get('bias_watchlist', False)})
                report = {'dataset': job['dataset'], 'objective': job['objective'],
                          'learning_modality': job['learning_modality'], 'direction': direction,
                          **metrics(scored), 'completed_at': ev.now(), 'raw_sha256': ev.file_hash(raw)}
                for key in ('question_type','evaluation_group','entity_kind','bias_watchlist'):
                    grouped = defaultdict(list)
                    for r in scored:
                        grouped[str(r[key])].append(r)
                    report['by_'+key] = {k:metrics(v) for k,v in sorted(grouped.items())}
                ev.save_rows(folder/'results.jsonl', scored)
                ev.save_json(folder/'report.json', report)
                print(json.dumps({k:report[k] for k in ('dataset','direction','total','correct','accuracy','invalid','truncated')}),flush=True)
            ev.save_json(out/'state.json', {'status': 'complete', 'finished_at': ev.now(),
                         'directions': job['directions'], 'smoke': args.smoke})
        except BaseException as exc:
            ev.save_json(out/'state.json', {'status': 'failed', 'error': str(exc),
                         'traceback': traceback.format_exc(), 'updated_at': ev.now()})
            raise


def launch(args, job, smoke=False):
    folder = args.work/('smoke' if smoke else 'jobs')/job['name']
    folder.mkdir(parents=True, exist_ok=True)
    p = folder/'state.json'
    if p.exists():
        previous = json.loads(p.read_text())
        if previous['status'] == 'complete':
            return
        assert not (previous['status'] == 'running' and alive(previous.get('pid'))), 'Existing evaluation still alive'
    cmd = [sys.executable, '-u', str(Path(__file__).resolve()), '--work', str(args.work),
           '--training-run', str(args.training_run), '--batch-size', str(args.batch_size),
           '--gpu', args.gpu, '--worker', job['name']]
    if smoke:
        cmd.append('--smoke')
    ev.save_json(folder/'command.json', cmd)
    with reserve(args.work, args.gpu, ('smoke:' if smoke else 'eval:')+job['name'], 20000) as reservation:
        with (folder/'worker.log').open('a') as log:
            child = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, cwd=ROOT, env=os.environ.copy())
            reservation.bind(child.pid)
            print('started', 'smoke' if smoke else 'full', job['name'], child.pid, flush=True)
            code = child.wait()
        assert code == 0, f'{job["name"]} exited {code}; see {folder}/worker.log'
    print('completed', 'smoke' if smoke else 'full', job['name'], flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work', type=Path, default=ROOT/'runs/qwen3_lora_cot_eval_20261025')
    ap.add_argument('--training-run', type=Path, default=ROOT/'runs/qwen3_lora_learning_ep5_20261024')
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--workers', type=int, choices=[1,2], default=2)
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--worker')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args(); args.work=args.work.resolve(); args.training_run=args.training_run.resolve()
    args.work.mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu, HF_HUB_OFFLINE='1', MAX_PIXELS='1003520',
                      OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    if args.worker:
        manifest = json.loads((args.work/'manifest.json').read_text())
        assert manifest['generation']['batch_size'] == args.batch_size
        job = next(j for j in manifest['jobs'] if j['name'] == args.worker)
        worker(args, manifest, job)
        return
    with ev.run_lock(args.work/'.coordinator.lock'):
        manifest = prepare(args)
        if args.prepare_only:
            print('Prepared',len(manifest['jobs']),'jobs,',manifest['predictions_expected'],'predictions',flush=True)
            return
        ev.save_json(args.work/'status.json', {'status': 'smoke', 'pid': os.getpid(), 'started_at':ev.now()})
        try:
            for name in ('rules_sft_text', 'entities_pt_vision', 'rules_base'):
                launch(args, next(j for j in manifest['jobs'] if j['name']==name), smoke=True)
            ev.save_json(args.work/'smoke_verification.json', {'status':'pass','jobs':3,'responses':6,
                         'updated_at':ev.now(),'note':'Plumbing and explicit-CoT template check; not formal accuracy.'})
            errors=[]; completed=0
            ev.save_json(args.work/'status.json', {'status':'running','pid':os.getpid(),
                         'completed_jobs':0,'total_jobs':10,'errors':[],'updated_at':ev.now()})
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                pending={pool.submit(launch,args,j):j for j in manifest['jobs']}
                for future in as_completed(pending):
                    try:
                        future.result(); completed+=1
                    except Exception as exc:
                        errors.append({'job':pending[future]['name'],'error':str(exc)})
                    summary=summarize(args.work,manifest)
                    ev.save_json(args.work/'status.json',{'status':'running','pid':os.getpid(),
                                 'completed_jobs':completed,'total_jobs':10,'cells_complete':len(summary),
                                 'errors':errors,'updated_at':ev.now()})
            assert all(ev.file_hash(p)==sha for p,sha in manifest['sources'].items())
            summary=summarize(args.work,manifest)
            if not errors:
                assert len(summary)==20 and sum(r['total'] for r in summary)==42180
            ev.save_json(args.work/'status.json',{'status':'failed' if errors else 'complete',
                         'completed_jobs':completed,'total_jobs':10,'errors':errors,
                         'cells_complete':len(summary),'predictions':sum(r['total'] for r in summary),
                         'finished_at':ev.now()})
            if errors:
                raise RuntimeError(str(errors))
        except BaseException as exc:
            ev.save_json(args.work/'coordinator_error.json', {'error':str(exc),'traceback':traceback.format_exc(),
                         'updated_at':ev.now()})
            ev.save_json(args.work/'status.json', {'status':'failed','error':str(exc),'updated_at':ev.now()})
            raise


if __name__=='__main__':
    main()
