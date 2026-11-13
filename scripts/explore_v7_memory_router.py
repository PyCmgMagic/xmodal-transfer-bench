#!/usr/bin/env python3
"""Fact-only memory adapters and a frozen solver; immutable v7 development trial.

The memory step sees a rule name only at test time. Its complete, uncorrected
output is passed to the original base model. Oracle contexts are separate.
"""
import argparse
import copy
import dataclasses
import json
import math
import os
import random
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import run_v7_evaluation as ev
from build_v7_experiment_data import name
from repair_v7_application_two_stage import SYSTEM as WORK_SYSTEM
from run_v7_full_experiment import ROOT, MODELS, environment, final_checkpoint, read, rows, save, sha, train_command
from run_v7_fullparam_ep10 import dense_identity
from v7_gpu_resources import reserve

OUT = ROOT / 'runs/v7_memory_router_20261019'
DATA = ROOT / 'rules_repaired_single_image_v7'
PARENT = ROOT / 'runs/v7_application_protocol_repair_20261019'
SOURCE = ROOT / 'runs/v7_fullparam_sft_ep10_20261019/data'
LEDGER = ROOT / 'runs/v7_full_experiment'
TRAIN_ARMS = ('text_memory', 'vision_memory', 'vision_bridge')
ARMS = ('base', 'text_ep10', 'vision_ep10') + TRAIN_ARMS
MEM_SYSTEM = 'Recall the stored rule identified by its name. State the exact conditions, values, mappings, and output conventions. Return the rule only; do not invent an example.'
RECALL_Q = 'State the exact stored rule for {name}. Include every parameter and output convention.'
FORMS = (
    RECALL_Q,
    'Recall the complete rule named {name}, including all exact values and mappings.',
    'What is the stored definition of {name}? Return its full factual rule.',
    'Give the rule for {name}, including every condition and output convention.',
)


def compact(k):
    text = k['material']['learning']['text']
    # These are source-material transformations, not test-conditioned summaries.
    prefix = re.split(r'Worked example:|For example,|For instance,|For instance:|For example:', text)[0].strip()
    marker = 'Inspection recording convention:'
    if 'Material refers' in text and 'Material refers' not in prefix:
        prefix += ' Material refers' + text.split('Material refers', 1)[1].split(marker, 1)[0].rstrip()
    if marker in text and marker not in prefix:
        prefix += ' ' + marker + text.split(marker, 1)[1]
    assert prefix and not any(s in prefix for s in ('Test input:', 'Options:'))
    return prefix


def messages(q, a=None):
    result = [{'role': 'system', 'content': MEM_SYSTEM}, {'role': 'user', 'content': q}]
    if a is not None:
        result.append({'role': 'assistant', 'content': a})
    return result


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT / 'manifest.json').exists():
        validate(); return
    knowledge = rows(PARENT / 'knowledge.jsonl')
    assert len(knowledge) == 18
    ev.save_rows(OUT / 'knowledge.jsonl', knowledge)
    targets = []
    # Training files and their targets are finalized before test rows/labels load.
    for arm in TRAIN_ARMS:
        training = []
        for k in knowledge:
            answer = compact(k)
            targets.append({'arm': arm, 'qid': k['id'], 'source': 'original_learning_text', 'answer': answer})
            for i in range(32):
                image = arm == 'vision_memory' or arm == 'vision_bridge' and i % 2 == 0
                q = FORMS[(i // 2) % len(FORMS)].format(name=name(k))
                if image:
                    q = '<image>Read this original learning reference. ' + q
                row = {'id': k['id'] + ':' + str(i), 'qid': k['id'],
                       'channel': 'image_fact' if image else 'name_fact', 'messages': messages(q, answer)}
                if image:
                    row['images'] = [str(DATA / 'images' / k['id'] / 'learning_visual.png')]
                training.append(row)
        random.Random(42).shuffle(training)
        assert len(training) == 576 and len({r['id'] for r in training}) == 576
        assert all(sum(m['content'].count('<image>') for m in r['messages']) == len(r.get('images', [])) for r in training)
        ev.save_rows(OUT / (arm + '.jsonl'), training)
    ev.save_rows(OUT / 'targets.jsonl', targets)
    lookup = {k['id']: k for k in knowledge}
    labels = {r['id']: r for r in rows(DATA / 'audit/eval_labels.jsonl')}
    tests = []
    for mod, direction in [('T', 'T2T'), ('V', 'T2V')]:
        for r in rows(SOURCE / ('eval_' + direction + '.jsonl')):
            if r['qid'] not in lookup:
                continue
            z = copy.deepcopy(r)
            label = labels[r['id']]
            z.update(id=mod + ':' + r['id'], source_id=r['id'], input_mod=mod,
                     task=label['question_type'], evaluation_group=label['evaluation_group'],
                     relation=lookup[r['qid']]['fact']['relation'], gold=label['answer'])
            tests.append(z)
    assert len(tests) == 72
    ev.save_rows(OUT / 'test_probes.jsonl', tests)
    recalls = []
    for k in knowledge:
        for image in (False, True):
            q = RECALL_Q.format(name=name(k))
            if image:
                q = '<image>Read this original learning reference. ' + q
            p = dict(id=('image:' if image else 'blind:') + k['id'], qid=k['id'],
                     input_mod='V' if image else 'T', stage='image_recall' if image else 'blind_recall',
                     messages=messages(q))
            if image:
                p['images'] = [str(DATA / 'images' / k['id'] / 'learning_visual.png')]
            recalls.append(p)
    ev.save_rows(OUT / 'recall_probes.jsonl', recalls)
    sources = [PARENT / 'manifest.json', PARENT / 'knowledge.jsonl', PARENT / 'independent_text_label_audit.json',
               SOURCE / 'eval_T2T.jsonl', SOURCE / 'eval_T2V.jsonl', DATA / 'audit/eval_labels.jsonl',
               DATA / 'image_hashes.jsonl', ROOT / 'scripts/repair_v7_application_two_stage.py',
               ROOT / 'scripts/run_v7_full_experiment.py', ROOT / 'scripts/run_v7_evaluation.py',
               ROOT / 'scripts/v7_gpu_resources.py']
    allimages = rows(DATA / 'image_hashes.jsonl')
    for r in allimages:
        assert sha(DATA / r['path']) == r['source_sha256'] == r['repaired_sha256']
    models = read(PARENT / 'manifest.json')['models']
    save(OUT / 'manifest.json', {
        'created_at': ev.now(), 'script_sha256': sha(Path(__file__)),
        'sources': {str(p): sha(p) for p in sources},
        'files': {p.name: sha(p) for p in OUT.glob('*.jsonl')},
        'models': models, 'training_arms': TRAIN_ARMS, 'arms': ARMS,
        'training': {'initialization': 'clean original base, no task warmup and no inherited adapters',
                     'rows': 576, 'rules': 18, 'presentations_each': 32,
                     'epochs': 1, 'fixed_final_step': 144, 'batch': 4, 'accumulation': 1,
                     'rank': 32, 'alpha': 64, 'dropout': 0.05, 'lr': 1e-4, 'seed': 42,
                     'vision_and_aligner_frozen': True, 'objective': 'fact-only SFT, no MCQ or application training'},
        'generation': {'blind_recall': 640, 'image_recall': 640, 'direct': 32, 'worksheet': 1024, 'choice': 32,
                       'temperature': 0, 'seed': 42, 'thinking': False, 'batch': 4},
        'routing': 'At test, name-only memory output is transmitted VERBATIM to the frozen original base. Base solves without options, then reads original options from its own uncorrected worksheet. No teacher, material, correction, retrieval or solver is supplied.',
        'controls': 'Original direct MCQ for each checkpoint; base-name-recall routed identically; correct original textual material oracle routed separately. Image recalls are acquisition diagnostics only and NEVER feed closed-book routing.',
        'decision': 'Development candidate only. Source closed-book routed accuracy >=60% and >=15pp over matched base route, oracle >=80%, format errors <=2%; inspect all relations and recall failures. Confirm on new rules before expansion.',
        'limits': ['Same18 development rules as protocol repair; one seed; not independent confirmation.',
                   'Historical ep10 trained on1000rules vs new adapters on18; comparisons are development, not equal-budget causal tests.',
                   'VISION targets include original textual factual annotations, unlike old generic captions; supervised multimodal training, not unsupervised image learning.',
                   'vision_bridge includes288extra name-only factual supervision rows; explicitly a text bridge, NOT pure visual learning.',
                   'Frozen base routing is an inference method change; direct scores remain separate. No native thinking.',
                   'All original question text/options/images/labels unchanged; every preselected rule retained.'],
        'original_pngs_verified': len(allimages),
    })
    snap = OUT / 'source_snapshots'; snap.mkdir(exist_ok=True)
    for p in [Path(__file__)] + [p for p in sources if p.suffix == '.py']:
        shutil.copy2(p, snap / p.name)
    validate()


def validate(images=False):
    m = read(OUT / 'manifest.json')
    assert m['script_sha256'] == sha(Path(__file__))
    for p, h in m['sources'].items():
        assert sha(Path(p)) == h, p
    for p, h in m['files'].items():
        assert sha(OUT / p) == h, p
    if images:
        for r in rows(DATA / 'image_hashes.jsonl'):
            assert sha(DATA / r['path']) == r['source_sha256'] == r['repaired_sha256']
    return m


def checkpoint(arm):
    if arm in TRAIN_ARMS:
        return OUT / arm / 'checkpoint/checkpoint-144'
    return Path(read(OUT / 'manifest.json')['models'][{'base': 'base', 'text_ep10': 'text', 'vision_ep10': 'vision'}[arm]]['path'])


class Runner:
    def __init__(self, folder, gpu, arm):
        self.folder, self.gpu, self.arm = folder, gpu, arm
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / 'predictions.jsonl'
        self.rr, notes = ev.read_rows(self.path, allow_partial_tail=True) if self.path.exists() else ([], [])
        if notes:
            ev.save_rows(self.path, self.rr)
        self.seen = {r['id'] for r in self.rr}
        assert len(self.seen) == len(self.rr)
        self.engine = None

    def load(self, cp=None, adapter=None):
        os.environ.update(environment(self.gpu, DATA))
        os.environ.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        import torch
        from swift.infer_engine import TransformersEngine
        from swift.utils import seed_everything
        torch.set_num_threads(4); seed_everything(42)
        cp = Path(cp or checkpoint(self.arm))
        if adapter is None:
            adapter = (cp / 'adapter_config.json').exists()
        identity = ev.checkpoint(cp) if adapter else dense_identity(cp)
        cfg = {'model': identity, 'arm': self.arm, 'generation': read(OUT / 'manifest.json')['generation']}
        old = read(self.folder / 'eval_manifest.json')
        assert not old or old == cfg
        save(self.folder / 'eval_manifest.json', cfg)
        self.engine = TransformersEngine(str(MODELS['qwen3'] if adapter else cp),
            adapters=[str(cp)] if adapter else None, model_type='qwen3_vl', torch_dtype=torch.bfloat16,
            device_map='cuda:0', max_batch_size=4, attn_impl='sdpa')
        self.engine.template.enable_thinking = False

    def infer(self, probes, stage):
        from swift.infer_engine import InferRequest, RequestConfig
        for mod in ('T', 'V'):
            pending = [p for p in probes if p['id'] not in self.seen and p['input_mod'] == mod]
            for start in range(0, len(pending), 4):
                batch = pending[start:start+4]
                replies = self.engine.infer([
                    InferRequest(messages=copy.deepcopy(p['messages']), images=p.get('images', [])) for p in batch],
                    RequestConfig(max_tokens=read(OUT / 'manifest.json')['generation'][stage], temperature=0, seed=42, return_details=True), use_tqdm=False)
                for p, reply in zip(batch, replies):
                    if isinstance(reply, Exception):
                        raise reply
                    c = reply.choices[0]; text = c.message.content or ''
                    pred = ev.letter(text) if stage in ('choice', 'direct') else None
                    raw = dataclasses.asdict(reply)
                    raw.pop('prompt_token_ids', None); raw.pop('prompt_logprobs', None)
                    r = {k: v for k, v in p.items() if k not in ('messages', 'images')}
                    r.update(stage=stage, response=text, predicted=pred,
                             correct=pred == p['gold'] if stage in ('choice', 'direct') else None,
                             format_error=stage in ('choice', 'direct') and pred is None,
                             finish_reason=c.finish_reason, raw_response=raw,
                             input_sha256=ev.digest({'messages': p['messages'], 'images': p.get('images', [])}))
                    with self.path.open('a') as f:
                        f.write(json.dumps(r, ensure_ascii=False, default=str) + '\n'); f.flush(); os.fsync(f.fileno())
                    self.rr.append(r); self.seen.add(r['id'])
                save(self.folder / 'state.json', {'status': 'running', 'saved': len(self.rr), 'pid': os.getpid(), 'stage': stage, 'updated_at': ev.now()})
                print(self.folder, stage, len(self.rr), flush=True)


def train(arm, gpu, smoke=False):
    validate()
    folder = OUT / ('smoke_' + arm if smoke else arm); folder.mkdir(exist_ok=True)
    with ev.run_lock(folder / 'train.lock'):
        cpdir = folder / 'checkpoint'
        cp = final_checkpoint(cpdir, 1) if not smoke else (cpdir / 'checkpoint-1' if (cpdir / 'checkpoint-1/adapter_model.safetensors').exists() else None)
        if cp is None:
            recover = [p for p in cpdir.glob('checkpoint-*') if all((p / n).exists() for n in ('optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json', 'adapter_model.safetensors'))]
            resume = max(recover, key=lambda p: read(p / 'trainer_state.json')['global_step']) if recover and not smoke else None
            dataset = OUT / (arm + '.jsonl')
            if smoke:
                # One text and one image batch are encoded by a bridge smoke.
                dataset = folder / 'input.jsonl'
                rr = rows(OUT / (arm + '.jsonl'))
                ev.save_rows(dataset, rr[:4])
            cmd = train_command('qwen3', 'sft', dataset, cpdir, 1, resume=resume)
            for flag, value in [('--per_device_train_batch_size', '4'), ('--gradient_accumulation_steps', '1'),
                                ('--lora_rank', '32'), ('--lora_alpha', '64'), ('--save_strategy', 'steps'),
                                ('--save_total_limit', '3'), ('--logging_steps', '1')]:
                cmd[cmd.index(flag)+1] = value
            cmd += ['--save_steps', '1' if smoke else '48', '--attn_impl', 'sdpa']
            if smoke:
                cmd += ['--max_steps', '1']
                cmd[cmd.index('--warmup_ratio')+1] = '0'
            env = environment(gpu, DATA)
            env.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
            save(folder / 'train_state.json', {'status': 'waiting_for_gpu', 'command': cmd, 'gpu': gpu})
            with reserve(LEDGER, gpu, str(folder) + ':train', 29000 if arm != 'text_memory' else 22000) as res:
                with (folder / 'train.log').open('a') as log:
                    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    res.bind(proc.pid)
                    save(folder / 'train_state.json', {'status': 'running', 'pid': proc.pid, 'command': cmd})
                    code = proc.wait()
                assert code == 0, (arm, code, str(folder / 'train.log'))
            cp = cpdir / ('checkpoint-1' if smoke else 'checkpoint-144')
        state = read(cp / 'trainer_state.json'); args = read(cp / 'args.json')
        assert state['global_step'] == (1 if smoke else 144)
        assert args['adapters'] in ([], None) and args['lora_rank'] == 32 and args['freeze_vit'] and args['freeze_aligner']
        assert args['model'] == str(MODELS['qwen3'])
        losses = [r['loss'] for r in rows(cpdir / 'logging.jsonl') if 'loss' in r]
        assert losses and all(math.isfinite(x) for x in losses)
        from safetensors import safe_open
        with safe_open(str(cp / 'adapter_model.safetensors'), framework='pt', device='cpu') as f:
            keys = list(f.keys()); bs = [k for k in keys if 'lora_B' in k]
            assert bs and all(bool(f.get_tensor(k).count_nonzero()) for k in bs)
            assert not any('.visual.' in k or '.merger.' in k for k in keys)
        save(folder / 'checkpoint_manifest.json', {**ev.checkpoint(cp), 'epoch': state['epoch'],
             'global_step': state['global_step'], 'nonzero_lora_B': len(bs), 'losses_finite': True,
             'final_loss': losses[-1], 'selection': 'fixed final step; no evaluation selection'})
        if smoke:
            run = Runner(folder / 'reload', gpu, arm)
            probes = []
            for mod in ('T', 'V'):
                p = copy.deepcopy(next(p for p in rows(OUT / 'test_probes.jsonl') if p['input_mod'] == mod))
                p['id'] = 'reload:' + p['id']; probes.append(p)
            if len(run.rr) < 2:
                with reserve(LEDGER, gpu, str(folder) + ':reload', 22000) as res:
                    res.bind(os.getpid()); run.load(cp=cp, adapter=True); run.infer(probes, 'direct')
            assert len(run.rr) == 2 and all(r['response'].strip() for r in run.rr)
            save(folder / 'validated.json', {'status': 'pass', 'checkpoint_reloaded': True, 'text_and_image_generation': True})
        save(folder / 'train_state.json', {'status': 'complete', 'checkpoint': str(cp)})


def memory(arm, gpu):
    validate()
    run = Runner(OUT / arm / 'memory', gpu, arm)
    pp = rows(OUT / 'recall_probes.jsonl')
    tests = [dict(p, id='direct:' + p['id']) for p in rows(OUT / 'test_probes.jsonl')]
    expected = 108
    with ev.run_lock(run.folder / 'run.lock'):
        if len(run.rr) < expected:
            with reserve(LEDGER, gpu, str(run.folder) + ':eval', 24000) as res:
                res.bind(os.getpid()); run.load()
                run.infer([p for p in pp if p['stage'] == 'blind_recall'], 'blind_recall')
                run.infer([p for p in pp if p['stage'] == 'image_recall'], 'image_recall')
                run.infer(tests, 'direct')
        assert len(run.rr) == expected
        save(run.folder / 'state.json', {'status': 'complete', 'saved': expected})


def solve_probes(arm):
    tests = rows(OUT / 'test_probes.jsonl')
    if arm == 'oracle':
        refs = {k['id']: k['material']['learning']['text'] for k in rows(OUT / 'knowledge.jsonl')}
    else:
        rr = rows(OUT / arm / 'memory/predictions.jsonl')
        refs = {r['qid']: r['response'] for r in rr if r['stage'] == 'blind_recall'}
        assert len(refs) == 18
    probes = []
    for p in tests:
        z = copy.deepcopy(p); z['id'] = 'worksheet:' + p['id']
        z['context_kind'] = 'oracle' if arm == 'oracle' else 'uncorrected_name_only_recall'
        z['memory_sha256'] = ev.digest(refs[p['qid']])
        z['memory_arm'] = arm
        z['messages'][0]['content'] = WORK_SYSTEM
        body = p['messages'][-1]['content'].split('\n\nOptions:\n')[0]
        z['messages'][-1]['content'] = ('Stored rule recalled by the memory model (uncorrected):\n' if arm != 'oracle'
            else 'Correct learning reference (oracle control):\n') + refs[p['qid']] + '\n\n' + body
        probes.append(z)
    return probes


def choices(run):
    drafts = {r['source_id'] + ':' + r['input_mod']: r for r in run.rr if r['stage'] == 'worksheet'}
    assert len(drafts) == 72
    pp = []
    for p in rows(OUT / 'test_probes.jsonl'):
        draft = drafts[p['source_id'] + ':' + p['input_mod']]
        opts = p['messages'][-1]['content'].split('\n\nOptions:\n')[1].split(ev.SUFFIX)[0]
        z = {k: v for k, v in p.items() if k not in ('messages', 'images')}
        z.update(id='choice:' + p['id'], worksheet_sha256=ev.digest(draft['response']),
                 memory_sha256=draft['memory_sha256'], memory_arm=draft['memory_arm'], context_kind=draft['context_kind'])
        z['messages'] = [{'role': 'system', 'content': ev.SYSTEM}, {'role': 'user', 'content':
            'Use this uncorrected worksheet to select the OPTION whose complete result and report decision match it. '
            'Candidate IDs and option letters are different.\n\nWorksheet:\n' + draft['response'] +
            '\n\nOptions:\n' + opts + ev.SUFFIX}]
        pp.append(z)
    return pp


def solve(arms, gpu):
    validate()
    engine = None
    with reserve(LEDGER, gpu, str(OUT) + ':frozen_solver:' + ','.join(arms), 24000) as res:
        res.bind(os.getpid())
        for arm in arms:
            run = Runner(OUT / arm / 'routed_base', gpu, 'base')
            with ev.run_lock(run.folder / 'run.lock'):
                if len(run.rr) < 144:
                    if engine is None:
                        run.load(); engine = run.engine
                    else:
                        run.engine = engine
                        # Record the same verified base identity for every route.
                        save(run.folder / 'eval_manifest.json', read(OUT / arms[0] / 'routed_base/eval_manifest.json'))
                    pp = solve_probes(arm)
                    ev.save_rows(run.folder / 'worksheet_probes.jsonl', pp)
                    run.infer(pp, 'worksheet')
                    cc = choices(run); ev.save_rows(run.folder / 'choice_probes.jsonl', cc)
                    run.infer(cc, 'choice')
                assert len(run.rr) == 144
                save(run.folder / 'state.json', {'status': 'complete', 'saved': 144, 'memory_arm': arm, 'solver': 'frozen_original_base'})
                summarize()


def metric(rr):
    n = len(rr); c = sum(r['correct'] is True for r in rr); valid = sum(not r['format_error'] for r in rr)
    return dict(n=n, correct=c, accuracy=c/n if n else None, valid_accuracy=c/valid if valid else None,
                format_errors=n-valid, empty=sum(not r['response'].strip() for r in rr),
                truncated=sum(r['finish_reason'] == 'length' for r in rr), letters=dict(Counter(r['predicted'] for r in rr)))


def summarize():
    summary = {}
    lines = ['# 闭卷记忆与冻结基座解题开发对照', '',
             '同18条开发规则、原题和原图；路由先仅凭名称回忆，再把未经修正回忆交给原基座解题。oracle单列。VISION bridge含额外无图文字事实监督。', '',
             '| 记忆来源 | 流程 | 文字正确 | 图片正确 | 格式错 |', '|---|---|---:|---:|---:|']
    for arm in ARMS + ('oracle',):
        info = {}
        for kind, path, stage in [('direct', OUT / arm / 'memory/predictions.jsonl', 'direct'),
                                  ('routed_base', OUT / arm / 'routed_base/predictions.jsonl', 'choice')]:
            rr = rows(path) if path.exists() else []
            ss = [r for r in rr if r['stage'] == stage]
            if not ss:
                continue
            result = {}
            for mod in ('T', 'V'):
                subset = [r for r in ss if r['input_mod'] == mod]
                stats = metric(subset)
                for key in ('task', 'relation', 'evaluation_group', 'qid'):
                    stats['by_' + key] = {v: metric([r for r in subset if r[key] == v]) for v in sorted({r[key] for r in subset})}
                stats['worksheet_truncated'] = sum(r['stage'] == 'worksheet' and r['input_mod'] == mod and r['finish_reason'] == 'length' for r in rr)
                result[mod] = stats
            info[kind] = result
            a, b = result['T'], result['V']
            def cell(s): return f'{s["correct"]}/{s["n"]} ({s["accuracy"]:.1%})' if s['n'] else '-'
            lines.append(f'| {arm} | {kind} | {cell(a)} | {cell(b)} | {a["format_errors"]+b["format_errors"]} |')
        summary[arm] = info
    save(OUT / 'summary.json', summary)
    (OUT / 'report_zh.md').write_text('\n'.join(lines) + '\n')


def audit():
    validate(images=True)
    checks = []
    for arm in ARMS + ('oracle',):
        for folder in (OUT / arm / 'memory', OUT / arm / 'routed_base'):
            if not (folder / 'predictions.jsonl').exists():
                continue
            rr = rows(folder / 'predictions.jsonl')
            if folder.name == 'memory':
                pp = rows(OUT / 'recall_probes.jsonl') + [dict(p, id='direct:' + p['id']) for p in rows(OUT / 'test_probes.jsonl')]
            else:
                pp = solve_probes(arm) + choices(Runner(folder, '0', 'base'))
            lookup = {p['id']: p for p in pp}
            assert len(rr) == len(lookup) and {r['id'] for r in rr} == set(lookup)
            for r in rr:
                p = lookup[r['id']]
                assert r['input_sha256'] == ev.digest({'messages': p['messages'], 'images': p.get('images', [])})
                assert r['response'] == (r['raw_response']['choices'][0]['message']['content'] or '')
                if r['stage'] in ('direct', 'choice'):
                    assert r['predicted'] == ev.letter(r['response']) and r['correct'] == (r['predicted'] == p['gold'])
            checks.append({'arm': arm, 'folder': str(folder), 'answers': len(rr), 'pass': True})
    save(OUT / 'completion_audit.json', {'checked_at': ev.now(), 'original_images_unchanged': 3000, 'checks': checks,
        'all_requested_complete': len(checks) == len(ARMS)*2+1,
        'new_generations': sum(c['answers'] for c in checks), 'smoke_generations_separate': 6})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=('prepare', 'smoke', 'train', 'memory', 'solve', 'summarize', 'audit'), required=True)
    ap.add_argument('--arm', choices=ARMS)
    ap.add_argument('--arms', nargs='+', choices=ARMS+('oracle',), default=list(ARMS)+['oracle'])
    ap.add_argument('--gpu', default='0')
    a = ap.parse_args()
    if a.stage == 'prepare': prepare()
    elif a.stage in ('train', 'smoke'):
        assert a.arm in TRAIN_ARMS
        train(a.arm, a.gpu, smoke=a.stage == 'smoke')
    elif a.stage == 'memory': memory(a.arm, a.gpu)
    elif a.stage == 'solve': solve(a.arms, a.gpu)
    elif a.stage == 'summarize': summarize()
    elif a.stage == 'audit': audit()
