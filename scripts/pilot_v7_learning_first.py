#!/usr/bin/env python3
"""TEXT-only learning ablation; immutable original direct T2T/T2V evaluation.

No recall routing, external solver, or source material enters the main tests.
Eight historical development rules; this is not independent confirmation.
"""
import argparse
import copy
import hashlib
import json
import math
import operator
import os
import random
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import explore_v7_memory_router as mem
import run_v7_evaluation as ev
from build_v7_experiment_data import name
from run_v7_full_experiment import ROOT, MODELS, environment, final_checkpoint, read, rows, save, sha, train_command
from v7_gpu_resources import reserve

OUT = ROOT / 'runs/v7_learning_first_20261019'
DATA = ROOT / 'rules_repaired_single_image_v7'
PARENT = ROOT / 'runs/v7_application_binding_20261018'
SOURCE = ROOT / 'runs/v7_fullparam_sft_ep10_20261019/data'
LEDGER = ROOT / 'runs/v7_full_experiment'
PLUGIN = ROOT / 'scripts/v7_equal_example_loss.py'
ARMS = ('explain', 'semantic', 'semantic_replay')
STEPS = 256
SEM_SYSTEM = 'Apply the learned named rule. Return the rule conditions and the complete requested result accurately.'


def signature(obs):
    return ev.digest(sorted(json.dumps(o, sort_keys=True) for o in obs))


def flags(rule, obs):
    ops = {'eq': operator.eq, 'ne': operator.ne, 'ge': operator.ge,
           'le': operator.le, 'gt': operator.gt, 'lt': operator.lt}
    return [bool(ops[c['operator']](obs[c['field']], c['value'])) for c in rule['conditions']]


def assignment(rule, obs):
    return '; '.join(f'{letter}={"KEEP" if (any(flags(rule, o)) if rule["mode"] == "any" else all(flags(rule, o))) else "OMIT"}'
                     for letter, o in zip('ABCD', obs))


def independent_assignment(rule, obs):
    """Separate closed-form check for the two explicitly supported topologies."""
    if rule['selection_topology'] == 'disjunctive_filter':
        wanted = {c['field']: c['value'] for c in rule['conditions']}
        keep = [o['shape'] == wanted['shape'] or o['color'] == wanted['color'] for o in obs]
    else:
        assert rule['selection_topology'] == 'interval_filter'
        lower = next(c for c in rule['conditions'] if c['operator'] in ('gt', 'ge'))
        upper = next(c for c in rule['conditions'] if c['operator'] in ('lt', 'le'))
        keep = [(o['value'] > lower['value'] or o['value'] == lower['value'] and lower['operator'] == 'ge')
                and (o['value'] < upper['value'] or o['value'] == upper['value'] and upper['operator'] == 'le') for o in obs]
    return '; '.join(f'{c}={"KEEP" if v else "OMIT"}' for c, v in zip('ABCD', keep))


def construct(knowledge, role, ntrain, ntest, occupied):
    result = []
    rng = random.Random('learning-first-20261019:42:' + role)
    for family in ('disjunctive_filter', 'interval_filter'):
        group = [k for k in knowledge if json.loads(k['fact']['quote'])['selection_topology'] == family]
        assert len(group) == 4
        rr = {k['id']: json.loads(k['fact']['quote']) for k in group}
        if family == 'disjunctive_filter':
            shapes = sorted({c['value'] for r in rr.values() for c in r['conditions'] if c['field'] == 'shape'} | {'circle', 'square', 'triangle', 'diamond'})
            colors = sorted({c['value'] for r in rr.values() for c in r['conditions'] if c['field'] == 'color'} | {'blue', 'red', 'green', 'black', 'gray'})
        for i in range(ntrain + ntest):
            for _ in range(200000):
                obs = ([{'shape': rng.choice(shapes), 'color': rng.choice(colors)} for _ in range(4)]
                       if family == 'disjunctive_filter' else [{'value': v} for v in rng.sample(range(31), 4)])
                sig = signature(obs)
                answers = {q: assignment(r, obs) for q, r in rr.items()}
                if sig not in occupied and len(set(answers.values())) == 4:
                    break
            else:
                raise RuntimeError(('Cannot distinguish all four rules', role, family))
            occupied.add(sig)
            for q, rule in rr.items():
                assert answers[q] == independent_assignment(rule, obs)
            options = list(answers.values()); rng.shuffle(options)
            result.append({'id': f'{role}:{family}:{i}', 'role': role, 'family': family,
                           'split': 'train' if i < ntrain else 'holdout', 'observations': obs,
                           'observation_signature': sig, 'rule_ids': list(rr), 'results': answers,
                           'options': options, 'submitted_report': options[i % 4]})
    return result


def practice(block, k, task, semantic=False, reference=False):
    rule = json.loads(k['fact']['quote']); q = k['id']; result = block['results'][q]
    report = block['submitted_report']
    head = (f'Apply {name(k)}. Report KEEP/OMIT for each candidate; KEEP means selected and OMIT means not selected.'
            if task == 'direct' else f'A technician submitted this report under {name(k)}: Result: {report}. Check the observations and the rule. Report KEEP/OMIT for each candidate; KEEP means selected and OMIT means not selected.')
    description = 'The observed candidate set is: ' + '; '.join(
        c + ' has ' + ', '.join(f'{field}: {value}' for field, value in obs.items()) for c, obs in zip('ABCD', block['observations'])) + '.'
    body = head + '\n\nTest input:\n' + description
    action = 'Accept report.' if report == result else 'Replace report.'
    if semantic:
        body += '\n\nState the named rule, each candidate\'s condition checks, and the complete result.'
        if task == 'diagnosis':
            body += ' Also state whether to accept or replace the submitted report.'
        traces = []
        for c, obs in zip('ABCD', block['observations']):
            checks = ', '.join(f'condition {j + 1}={"Met" if v else "Not met"}' for j, v in enumerate(flags(rule, obs)))
            traces.append(f'{c}: {checks}.')
        answer = 'Rule: ' + mem.compact(k) + '\nChecks: ' + ' '.join(traces) + '\n'
        if task == 'diagnosis':
            answer += action + ' '
        answer += 'Result: ' + result + '.'
    else:
        choices = [('Result: ' + v + '.' if task == 'direct' else
                    ('Accept report. ' if v == report else 'Replace report. ') + 'Result: ' + v + '.') for v in block['options']]
        body += '\n\nOptions:\n' + '\n'.join(c + '. ' + v for c, v in zip('ABCD', choices)) + ev.SUFFIX
        answer = 'ABCD'[block['options'].index(result)]
    if reference:
        body = 'Learning reference:\n' + mem.compact(k) + '\n\n' + body
    return {'id': f'{block["id"]}:{q}:{task}:{"semantic" if semantic else "mcq"}',
            'qid': q, 'task': task, 'relation': k['fact']['relation'], 'evaluation_group': 'result_application' if task == 'direct' else 'rule_report_audit',
            'block': block['id'], 'observation_signature': block['observation_signature'],
            'gold': answer, 'input_mod': 'T', 'fmt': 'semantic' if semantic else 'mcq',
            'messages': [{'role': 'system', 'content': SEM_SYSTEM if semantic else ev.SYSTEM}, {'role': 'user', 'content': body}]}


def as_training(probe, channel):
    z = copy.deepcopy(probe); z['channel'] = channel
    z['messages'].append({'role': 'assistant', 'content': z['gold']})
    return z


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


def prepare():
    if (OUT / 'manifest.json').exists():
        validate(); return
    OUT.mkdir(exist_ok=True)
    target = rows(PARENT / 'target_knowledge.jsonl'); assert len(target) == 8
    ids = {k['id'] for k in target}
    allk = [r['content'] for r in rows(DATA / 'audit/rule_knowledge.jsonl')]
    replay = []
    for family in ('disjunctive_filter', 'interval_filter'):
        candidates = []
        for k in allk:
            r = json.loads(k['fact']['quote'])
            if k['id'] in ids or r.get('selection_topology') != family:
                continue
            if family == 'disjunctive_filter' and {c['field'] for c in r['conditions']} != {'shape', 'color'}:
                continue
            candidates.append(k)
        candidates.sort(key=lambda k: hashlib.sha256(('learning-first-replay:42:' + k['id']).encode()).hexdigest())
        used = set()
        for k in candidates:
            r = json.loads(k['fact']['quote']); key = ev.digest({x: r[x] for x in ('conditions', 'mode')})
            if key not in used:
                replay.append(k); used.add(key)
            if len(used) == 4:
                break
        assert len(used) == 4
    lookup = {k['id']: k for k in target + replay}
    ev.save_rows(OUT / 'target_knowledge.jsonl', target)
    ev.save_rows(OUT / 'replay_knowledge.jsonl', replay)
    occupied = set()
    blocks = construct(target, 'target', 20, 12, occupied) + construct(replay, 'replay', 16, 4, occupied)
    ev.save_rows(OUT / 'blocks.jsonl', blocks)
    apps, semantic, replay_apps, replay_fact = [], [], [], []
    for b in blocks:
        if b['split'] != 'train':
            continue
        for q in b['rule_ids']:
            for task in ('direct', 'diagnosis'):
                k = lookup[q]
                if b['role'] == 'target':
                    apps.append(as_training(practice(b, k, task), 'target_mcq'))
                    semantic.append(as_training(practice(b, k, task, semantic=True), 'target_semantic'))
                else:
                    replay_apps.append(as_training(practice(b, k, task, reference=True), 'independent_application_replay'))
                    replay_fact.append({'id': f'replayfact:{b["id"]}:{q}:{task}', 'qid': q, 'channel': 'independent_fact_replay',
                                        'messages': mem.messages('Read this learning reference and restate its complete factual rule:\n' + mem.compact(k), mem.compact(k))})
    assert len(apps) == len(semantic) == 320 and len(replay_apps) == len(replay_fact) == 256
    facts = []
    for i in range(96):
        for k in target:
            facts.append({'id': f'fact:{k["id"]}:{i}', 'qid': k['id'], 'channel': 'target_fact',
                          'messages': mem.messages(mem.FORMS[i % len(mem.FORMS)].format(name=name(k)), mem.compact(k))})
    training = {'explain': facts + replay_fact, 'semantic': facts[:128] + apps + semantic + replay_fact,
                'semantic_replay': facts[:128] + apps + semantic + replay_apps}
    for arm, rr in training.items():
        assert len(rr) == 1024 and len({r['id'] for r in rr}) == 1024
        random.Random(42).shuffle(rr)
        assert all(not r.get('images') and all('<image>' not in m['content'] for m in r['messages']) for r in rr)
        ev.save_rows(OUT / (arm + '.jsonl'), rr)
    # Training files are frozen before original test observations/labels load.
    labels = {r['id']: r for r in rows(DATA / 'audit/eval_labels.jsonl')}
    official, checks, original_signatures = [], [], set()
    for mod, direction in [('T', 'T2T'), ('V', 'T2V')]:
        for p in rows(SOURCE / f'eval_{direction}.jsonl'):
            if p['qid'] not in lookup:
                continue
            if mod == 'T':
                observed = p['messages'][-1]['content'].split('\n\nTest input:\n')[1].split('\n\nOptions:\n')[0]
                obs = []
                for code, desc in re.findall(r'([A-D]) has (.*?)(?=; [A-D] has|$)', observed.rstrip('.')):
                    attrs = dict(re.findall(r'(\w+): ([^,;]+)', desc))
                    if 'value' in attrs:
                        attrs['value'] = float(attrs['value'])
                        if attrs['value'].is_integer(): attrs['value'] = int(attrs['value'])
                    obs.append(attrs)
                assert len(obs) == 4
                sig = signature(obs); original_signatures.add(sig)
                assert sig not in occupied, ('Generated practice overlaps original test', p['id'])
                r = json.loads(lookup[p['qid']]['fact']['quote']); expected = assignment(r, obs)
                assert expected == independent_assignment(r, obs)
                label = labels[p['id']]; task = label['question_type']
                opts = dict(re.findall(r'(?m)^([A-D])\. (.*)$', p['messages'][-1]['content']))
                match = [c for c, s in opts.items() if s.split('Result: ', 1)[1].rstrip('.') == expected]
                if task == 'diagnosis':
                    head = p['messages'][-1]['content'].split('\n\nTest input:\n')[0]
                    report = head.split('Result: ', 1)[1].split('. ', 1)[0].rstrip('.')
                    action = 'Accept report.' if report == expected else 'Replace report.'
                    match = [c for c in match if opts[c].startswith(action)]
                assert match == [label['answer']], (p['id'], match, label)
                checks.append({'id': p['id'], 'expected_result': expected, 'gold': match[0], 'signature': sig})
            if p['qid'] in ids:
                label = labels[p['id']]; z = copy.deepcopy(p)
                z.update(id=mod + ':' + p['id'], source_id=p['id'], condition='original', input_mod=mod,
                         gold=label['answer'], task=label['question_type'], evaluation_group=label['evaluation_group'],
                         relation=lookup[p['qid']]['fact']['relation'], fmt='mcq')
                official.append(z)
    assert len(official) == 32
    probes = official[:]
    for b in blocks:
        if b['split'] == 'holdout' or b['role'] == 'target' and int(b['id'].split(':')[-1]) < 4:
            for q in b['rule_ids']:
                for task in ('direct', 'diagnosis'):
                    p = practice(b, lookup[q], task, reference=b['role'] == 'replay')
                    p['condition'] = 'target_holdout' if b['role'] == 'target' and b['split'] == 'holdout' else 'target_seen' if b['role'] == 'target' else 'retention_control'
                    probes.append(p)
    for p in official:
        z = copy.deepcopy(p); z.update(id='oracle:' + p['id'], condition='oracle_control')
        z['messages'][-1]['content'] = 'Correct learning reference:\n' + mem.compact(lookup[p['qid']]) + '\n\n' + z['messages'][-1]['content']
        probes.append(z)
    for k in target:
        probes.append({'id': 'recall:' + k['id'], 'qid': k['id'], 'condition': 'recall_control', 'input_mod': 'T',
                       'fmt': 'fact', 'gold': mem.compact(k), 'messages': mem.messages(mem.RECALL_Q.format(name=name(k)))})
    assert len(probes) == 392 and len({p['id'] for p in probes}) == len(probes)
    ev.save_rows(OUT / 'probes.jsonl', probes)
    save(OUT / 'label_and_split_audit.json', {'status': 'pass', 'original_text_labels_recalculated': checks,
         'train_holdout_signatures_disjoint': True, 'all_generated_vs_original_signatures_disjoint': True,
         'generated_results_independently_recalculated': True, 'train_text_only': True,
         'target_and_replay_ids_disjoint': not ids & {k['id'] for k in replay}})
    images = rows(DATA / 'image_hashes.jsonl')
    for r in images:
        assert sha(DATA / r['path']) == r['source_sha256'] == r['repaired_sha256']
    from run_v7_fullparam_ep10 import dense_identity
    base = dense_identity(MODELS['qwen3'])
    sources = [PARENT / 'target_knowledge.jsonl', DATA / 'audit/rule_knowledge.jsonl', DATA / 'audit/eval_labels.jsonl',
               DATA / 'image_hashes.jsonl', SOURCE / 'eval_T2T.jsonl', SOURCE / 'eval_T2V.jsonl', PLUGIN,
               ROOT / 'scripts/explore_v7_memory_router.py', ROOT / 'scripts/run_v7_evaluation.py',
               ROOT / 'scripts/run_v7_full_experiment.py', ROOT / 'scripts/v7_gpu_resources.py', ROOT / 'scripts/run_v7_fullparam_ep10.py']
    manifest = {'created_at': ev.now(), 'script_sha256': sha(Path(__file__)), 'sources': {str(p): sha(p) for p in sources},
                'files': {p.name: sha(p) for p in OUT.glob('*.jsonl')}, 'base': base, 'arms': ARMS,
                'training': {'initialization': 'clean original base', 'modality': 'TEXT only', 'rules': 8,
                    'rows_per_arm': 1024, 'epochs': 1, 'fixed_steps': STEPS, 'microbatch': 2, 'accumulation': 2,
                    'learning_rate': 3e-5, 'rank': 32, 'alpha': 64, 'dropout': .05, 'seed': 42, 'max_length': 8192,
                    'freeze_vision_aligner': True, 'loss': 'equal per-example causal CE including common suffix; all arms identical',
                    'channels': {a: dict(Counter(r['channel'] for r in rr)) for a, rr in training.items()}},
                'generation': {'direct': 32, 'blind_recall': 512, 'temperature': 0, 'thinking': False, 'batch': 4},
                'evaluation': {'per_arm': len(probes), 'original': 32, 'target_holdout': 192, 'target_seen': 64,
                    'retention_control': 64, 'oracle_control': 32, 'recall_control': 8,
                    'main': 'original T2T/T2V unchanged inputs; same checkpoint direct letter answer; no material/recall/routing',
                    'controls': 'oracle and recall are separate diagnostics, never inputs to main tests'},
                'decision': {'development_screen': 'T2T gain >=15pp vs matched base and >=60% absolute; generated source holdout gain >=15pp; format error <=2%. This is a descriptive screen, not statistical confirmation.',
                    'cross_modal': 'Report raw and baseline-adjusted T2T/T2V gap; do not select training by low image accuracy or impose T2T>T2V.',
                    'next': 'Only fixed-method confirmation on different rules after source learning improves; otherwise preserve negative results.'},
                'limits': ['Eight historical development rules, two set-selection subfamilies, single seed; no full1000 conclusion.',
                    'A/B change training objective mix. B/C keep all target rows identical but change independent replay tasks.',
                    'Rows and optimizer steps match; answer lengths, token budgets and gradient contributions do not.',
                    'C replay includes correct textual references for independent rules only; no target material in application inputs.',
                    'Synthetic heldout observations are new instances, not new knowledge; both question types reported separately.'],
                'original_pngs_verified': len(images)}
    save(OUT / 'manifest.json', manifest)
    snap = OUT / 'source_snapshots'; snap.mkdir(exist_ok=True)
    for p in [Path(__file__)] + [p for p in sources if p.suffix == '.py']:
        shutil.copy2(p, snap / p.name)
    validate()
    print('Prepared three TEXT-only arms, 1024 rows each, 392 probes per checkpoint.', flush=True)


def checkpoint(arm, smoke=False):
    return OUT / ('smoke_' + arm if smoke else arm) / 'checkpoint' / f'checkpoint-{1 if smoke else STEPS}'


def train(arm, gpu, smoke=False):
    validate(); folder = checkpoint(arm, smoke).parent.parent; folder.mkdir(exist_ok=True)
    with ev.run_lock(folder / 'train.lock'):
        cp = checkpoint(arm, smoke)
        if not (cp / 'adapter_model.safetensors').exists():
            dataset = OUT / (arm + '.jsonl')
            if smoke:
                rr = rows(dataset)
                chosen = []
                for channel in sorted({r['channel'] for r in rr}):
                    chosen.append(next(r for r in rr if r['channel'] == channel))
                dataset = folder / 'input.jsonl'; ev.save_rows(dataset, chosen)
            candidates = [p for p in cp.parent.glob('checkpoint-*') if all((p / n).exists() for n in ('optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json', 'adapter_model.safetensors'))]
            resume = max(candidates, key=lambda p: read(p / 'trainer_state.json')['global_step']) if candidates and not smoke else None
            cmd = train_command('qwen3', 'sft', dataset, cp.parent, 1, resume=resume)
            for flag, value in [('--per_device_train_batch_size', '2' if not smoke else '4'), ('--gradient_accumulation_steps', '2' if not smoke else '1'),
                                ('--learning_rate', '3e-5'), ('--lora_rank', '32'), ('--lora_alpha', '64'),
                                ('--save_strategy', 'steps'), ('--logging_steps', '1')]:
                cmd[cmd.index(flag) + 1] = value
            cmd += ['--save_steps', '1' if smoke else '64', '--attn_impl', 'sdpa',
                    '--external_plugins', str(PLUGIN), '--loss_type', 'v7_equal_example',
                    '--packing', 'false', '--padding_free', 'false', '--use_logits_to_keep', 'false']
            if smoke:
                cmd += ['--max_steps', '1']; cmd[cmd.index('--warmup_ratio') + 1] = '0'
            env = environment(gpu, DATA); env.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                V7_EQUAL_LOSS_AUDIT=str(folder / 'loss_audit.jsonl'))
            save(folder / 'state.json', {'status': 'waiting_for_gpu', 'command': cmd, 'gpu': gpu})
            with reserve(LEDGER, gpu, str(folder) + ':train', 21000) as res:
                with (folder / 'train.log').open('a') as f:
                    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT)
                    res.bind(proc.pid); save(folder / 'state.json', {'status': 'running', 'command': cmd, 'pid': proc.pid, 'gpu': gpu})
                    code = proc.wait()
                if code != 0:
                    save(folder / 'state.json', {'status': 'failed', 'returncode': code})
                    raise RuntimeError((arm, code, str(folder / 'train.log')))
        state = read(cp / 'trainer_state.json'); args = read(cp / 'args.json')
        assert state['global_step'] == (1 if smoke else STEPS)
        assert args['model'] == str(MODELS['qwen3']) and args['adapters'] in ([], None)
        assert args['freeze_vit'] and args['freeze_aligner'] and args['lora_rank'] == 32
        losses = [r['loss'] for r in rows(cp.parent / 'logging.jsonl') if 'loss' in r]
        assert losses and all(math.isfinite(x) for x in losses)
        from safetensors import safe_open
        with safe_open(str(cp / 'adapter_model.safetensors'), framework='pt', device='cpu') as f:
            keys = list(f.keys()); bs = [k for k in keys if 'lora_B' in k]
            assert bs and all(bool(f.get_tensor(k).count_nonzero()) for k in bs)
            assert not any('.visual.' in k or '.merger.' in k for k in keys)
        save(folder / 'checkpoint_manifest.json', {**ev.checkpoint(cp), 'global_step': state['global_step'],
             'epoch': state['epoch'], 'nonzero_lora_B': len(bs), 'loss_finite': True, 'final_loss': losses[-1],
             'selection': 'fixed final epoch/step; no evaluation selection'})
        save(folder / 'state.json', {'status': 'complete', 'checkpoint': str(cp)})


def evaluate(arm, gpu, smoke=False):
    validate(); mem.OUT = OUT
    folder = OUT / ('smoke_' + arm if smoke else arm) / ('reload' if smoke else 'evaluation')
    folder.mkdir(parents=True, exist_ok=True)
    pp = rows(OUT / 'probes.jsonl')
    if smoke:
        pp = [next(p for p in pp if p['condition'] == 'original' and p['input_mod'] == mod) for mod in ('T', 'V')]
    cp = MODELS['qwen3'] if arm == 'base' else checkpoint(arm, smoke)
    with ev.run_lock(folder / 'eval.lock'):
        run = mem.Runner(folder, gpu, arm)
        if len(run.seen) < len(pp):
            # Do not bind current PID: this process performs later stages, and
            # keeping its reservation beyond model destruction would leak budget.
            with reserve(LEDGER, gpu, str(folder) + ':evaluation', 22000):
                run.load(cp=cp, adapter=arm != 'base')
                run.infer([p for p in pp if p['fmt'] == 'mcq'], 'direct')
                run.infer([p for p in pp if p['fmt'] == 'fact'], 'blind_recall')
                del run.engine
                import gc, torch
                gc.collect(); torch.cuda.empty_cache()
        assert run.seen == {p['id'] for p in pp}
        if smoke:
            assert all(r['response'].strip() for r in run.rr)
            save(folder.parent / 'validated.json', {'status': 'pass', 'checkpoint_reload': True, 'text_and_original_image_generation': True})
        save(folder / 'state.json', {'status': 'complete', 'saved': len(run.rr)})


def metric(rr):
    n = len(rr); correct = sum(bool(r['correct']) for r in rr); invalid = sum(bool(r.get('format_error')) for r in rr)
    return {'n': n, 'correct': correct, 'accuracy': correct / n if n else None,
            'format_errors': invalid, 'empty': sum(not r['response'].strip() for r in rr),
            'valid_accuracy': correct / (n - invalid) if n > invalid else None,
            'answers': dict(Counter(r['predicted'] for r in rr)),
            'by_task': {task: {'n': len(ss := [r for r in rr if r.get('task') == task]),
                             'correct': sum(bool(r['correct']) for r in ss),
                             'accuracy': sum(bool(r['correct']) for r in ss) / len(ss) if ss else None} for task in ('direct', 'diagnosis')},
            'by_group': {g: {'n': len(ss := [r for r in rr if r.get('evaluation_group') == g]),
                             'correct': sum(bool(r['correct']) for r in ss)} for g in ('result_application', 'rule_report_audit')}}


def summarize():
    import csv
    validate(images=True); result = {}; flat = []
    for arm in ('base',) + ARMS:
        path = OUT / arm / 'evaluation/predictions.jsonl'
        if not path.exists(): continue
        rr = rows(path); groups = {}
        for condition in ('original', 'target_holdout', 'target_seen', 'retention_control', 'oracle_control'):
            for mod in ('T', 'V'):
                ss = [r for r in rr if r['condition'] == condition and r['input_mod'] == mod]
                if ss:
                    key = condition + '_' + mod; groups[key] = metric(ss)
                    flat.append({'arm': arm, 'condition': condition, 'input_mod': mod,
                                 **{k: groups[key][k] for k in ('n', 'correct', 'accuracy', 'format_errors', 'empty', 'valid_accuracy')}})
        recalls = [r for r in rr if r['condition'] == 'recall_control']
        groups['recall_exact'] = {'n': len(recalls), 'exact': sum(re.sub(r'\s+', ' ', r['response'].strip()) == re.sub(r'\s+', ' ', r['gold'].strip()) for r in recalls),
                                  'limit': 'Literal reconstruction only; paraphrases require semantic review.'}
        result[arm] = groups
    save(OUT / 'summary.json', result)
    if flat:
        with (OUT / 'summary.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    lines = ['# TEXT学习端修复开发实验', '', '8条历史开发规则、两种集合筛选结构、单seed；所有正式题、图像与短答案协议不变。',
             '三组各1024行/1epoch/256步，干净基座、LoRA32、LR3e-5、batch2×accum2；只训练文字。',
             'A=解释；B=解释+语义结果+选择题；C=B的目标知识样本原样保留，将独立规则事实回放换成独立规则解题回放。',
             '行数与步数相同，token数不相同。oracle及回忆只诊断，不参与主评测。', '',
             '| 组 | 原T2T | 原T2V | 未见文字练习 | 已见文字练习 | 独立解题控制 |', '|---|---:|---:|---:|---:|---:|']
    def cell(g, key):
        v = g.get(key); return f'{v["correct"]}/{v["n"]} ({v["accuracy"]:.1%})' if v else '—'
    for arm, g in result.items():
        lines.append('| ' + ' | '.join([arm] + [cell(g, key) for key in ('original_T', 'original_V', 'target_holdout_T', 'target_seen_T', 'retention_control_T')]) + ' |')
    if 'base' in result:
        lines += ['', '| 组 | T2T增益 | T2V增益 | 文本减图片的增益差 |', '|---|---:|---:|---:|']
        for arm in ARMS:
            if arm in result and all(k in result[arm] for k in ('original_T', 'original_V')):
                dt = result[arm]['original_T']['accuracy'] - result['base']['original_T']['accuracy']
                dv = result[arm]['original_V']['accuracy'] - result['base']['original_V']['accuracy']
                lines.append(f'| {arm} | {dt * 100:+.2f}pp | {dv * 100:+.2f}pp | {(dt - dv) * 100:+.2f}pp |')
    lines += ['', '原题每模态16题，每规则2题；生成的未见文字练习192题，不能把它当原T2T成绩或独立192条知识。',
              '不按图片低分挑方案，不筛除失败规则；来源方向未学会时，不将结果单独解释为跨模态迁移失败。',
              '三组完成后检查原始选项分布、语义原答、给材料控制及同题变化，再决定是否进入不同规则确认。']
    (OUT / 'report_zh.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def audit():
    validate(images=True); pp = {p['id']: p for p in rows(OUT / 'probes.jsonl')}; files = {}
    for arm in ('base',) + ARMS:
        path = OUT / arm / 'evaluation/predictions.jsonl'; rr = rows(path)
        assert len(rr) == len(pp) and {r['id'] for r in rr} == set(pp)
        for r in rr:
            p = pp[r['id']]
            assert r['input_sha256'] == ev.digest({'messages': p['messages'], 'images': p.get('images', [])})
            if p['fmt'] == 'mcq':
                assert r['predicted'] == ev.letter(r['response']) and r['correct'] == (ev.letter(r['response']) == p['gold'])
        files[arm] = sha(path)
        if arm != 'base':
            assert read(OUT / arm / 'checkpoint_manifest.json')['global_step'] == STEPS
            assert read(OUT / ('smoke_' + arm) / 'validated.json')['status'] == 'pass'
    save(OUT / 'completion_audit.json', {'status': 'pass', 'all_four_checkpoints_complete': True,
         '392_predictions_each': True, 'all_original_pngs_unchanged': True, 'input_hashes_verified': True,
         'main_has_no_material_recall_or_external_solver': True, 'prediction_files_sha256': files})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', required=True, choices=('prepare', 'smoke', 'train', 'evaluate', 'run', 'summarize', 'audit'))
    ap.add_argument('--arm', choices=('base',) + ARMS, default='semantic_replay'); ap.add_argument('--gpu', default='0')
    a = ap.parse_args()
    if a.stage == 'prepare': prepare()
    elif a.stage == 'smoke': train(a.arm, a.gpu, True); evaluate(a.arm, a.gpu, True)
    elif a.stage == 'train': train(a.arm, a.gpu)
    elif a.stage == 'evaluate': evaluate(a.arm, a.gpu)
    elif a.stage == 'run':
        if a.arm != 'base':
            train(a.arm, a.gpu, True); evaluate(a.arm, a.gpu, True); train(a.arm, a.gpu)
        evaluate(a.arm, a.gpu)
    elif a.stage == 'summarize': summarize()
    else: audit()


if __name__ == '__main__': main()
