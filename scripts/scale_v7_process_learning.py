#!/usr/bin/env python3
"""Frozen 64-rule, three-seed TEXT process learning with resumable GPU queues."""
import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
import operator
import os
import random
import re
import shutil
import subprocess
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import confirm_v7_process_learning as prior
import explore_v7_memory_router as mem
import pilot_v7_learning_first as old
import run_v7_evaluation as ev
from build_v7_experiment_data import name
from run_v7_full_experiment import ROOT, MODELS, environment, read, rows, save, sha, train_command
from run_v7_fullparam_ep10 import dense_identity
from v7_gpu_resources import reserve

OUT = ROOT / 'runs/v7_process_scale64_20261021'
DATA = prior.DATA
PY = ROOT / '.venv/bin/python'
SEEDS = (42, 43, 44)
ARMS = ('definitions', 'definition_replay', 'process')
FAMILIES = ('disjunctive_filter', 'interval_filter', 'conjunctive_filter', 'include_with_exclusion')
STEPS = 3840
OPS = dict(eq=operator.eq, ne=operator.ne, ge=operator.ge, gt=operator.gt, le=operator.le, lt=operator.lt)


def rule(k):
    return json.loads(k['fact']['quote'])


def truth(r, obs):
    # Independent direct closed forms, separate from generic flags/assignment.
    t = r['selection_topology']
    wanted = {c['field']: c['value'] for c in r['conditions']}
    if t == 'disjunctive_filter':
        keep = [o['category'] == wanted['category'] or o['material'] == wanted['material'] for o in obs]
    elif t == 'conjunctive_filter':
        keep = [o['category'] == wanted['category'] and o['material'] == wanted['material'] and o['color'] == wanted['color'] for o in obs]
    elif t == 'include_with_exclusion':
        keep = [o['category'] == wanted['category'] and o['status'] != wanted['status'] for o in obs]
    else:
        assert t == 'interval_filter'
        lo, hi = r['conditions']
        keep = [(o['value'] > lo['value'] or o['value'] == lo['value'] and lo['operator'] == 'ge') and
                (o['value'] < hi['value'] or o['value'] == hi['value'] and hi['operator'] == 'le') for o in obs]
    return '; '.join(c + '=' + ('KEEP' if v else 'OMIT') for c, v in zip('ABCD', keep))


def field_specs(k):
    r = rule(k)
    if r['selection_topology'] in FAMILIES[:2]:
        return prior.field_specs(k)
    if r['selection_topology'] == 'conjunctive_filter':
        return [(c['field'], 'required ' + c['field'], str(c['value'])) for c in r['conditions']] + [
            ('logic', 'logical connective (AND or OR)', 'AND'),
            ('all', 'decision if all conditions match', 'KEEP'),
            ('missing', 'decision if any required condition fails', 'OMIT')]
    wanted = {c['field']: c['value'] for c in r['conditions']}
    return [('category', 'required category', wanted['category']), ('status', 'excluded status', wanted['status']),
        ('status_operator', 'comparison for the excluded status', 'not equal to'),
        ('logic', 'logical connective (AND or OR)', 'AND'),
        ('excluded', 'decision if the excluded status is present', 'OMIT'),
        ('scope', 'scope of checking', 'each candidate independently')]


def logic_specs(k):
    t = rule(k)['selection_topology']
    if t in FAMILIES[:2]:
        return prior.logic_specs(k)
    if t == 'conjunctive_filter':
        return [('Must all conditions match?', 'Yes. Every condition must hold.'),
            ('Does one matching condition suffice?', 'No. All conditions must hold.'),
            ('Does a candidate with every condition matching qualify?', 'Yes.'),
            ('Does a candidate with any required condition failing qualify?', 'No.')]
    return [('Does the excluded status disqualify a matching category?', 'Yes.'),
        ('Does the wrong category qualify if its status is acceptable?', 'No.'),
        ('Must both the category requirement and status exclusion hold?', 'Yes.'),
        ('Does a matching category without the excluded status qualify?', 'Yes.')]


def facts(k):
    result = []
    for j, (form, detail) in enumerate(itertools.product(prior.FORMS, prior.DETAILS)):
        result.append(dict(id=f'fact:{k["id"]}:{j}', qid=k['id'], channel='full_fact',
                           messages=mem.messages(form.format(name=name(k)) + detail, mem.compact(k))))
    for key, question, answer in field_specs(k):
        for j, form in enumerate(('For {name}, state the {question}.', 'Recall the {question} in {name}.',
                                  'What is the {question} of {name}?', 'Give only the {question} for {name}.')):
            result.append(dict(id=f'field:{k["id"]}:{key}:{j}', qid=k['id'], channel='atomic_fact',
                messages=[dict(role='system', content='Answer the factual question about the learned rule. Return only the requested fact.'),
                          dict(role='user', content=form.format(name=name(k), question=question)),
                          dict(role='assistant', content=answer)]))
    for j, (question, answer) in enumerate(logic_specs(k)):
        for n, prefix in enumerate(('Under {name}: ', 'Regarding the rule {name}: ')):
            result.append(dict(id=f'logic:{k["id"]}:{j}:{n}', qid=k['id'], channel='boundary_logic',
                messages=mem.messages(prefix.format(name=name(k)) + question, answer)))
    assert len(result) == 80
    return result


def trace(p, k, b):
    z = prior.reasoning(p)
    z.update(id='trace:' + p['id'], channel='worked_application')
    r = rule(k)
    answer = 'Rule: ' + mem.compact(k) + '\n'
    for letter, o in zip('ABCD', b['observations']):
        flags = old.flags(r, o)
        keep = any(flags) if r['mode'] == 'any' else all(flags)
        parts = [f'{c["field"]} {o[c["field"]]} ' + dict(eq='==', ne='!=', ge='>=', gt='>', le='<=', lt='<')[c['operator']] +
                 f' {c["value"]} = {"YES" if flag else "NO"}' for c, flag in zip(r['conditions'], flags)]
        answer += letter + ': ' + '; '.join(parts) + f'; {"OR" if r["mode"] == "any" else "AND"} => {"KEEP" if keep else "OMIT"}.\n'
    if b['task'] == 'diagnosis':
        answer += ('Accept report.' if b['submitted_report'] == b['results'][k['id']] else 'Replace report.') + '\n'
    answer += 'Result: ' + b['results'][k['id']] + '.\nAnswer: ' + p['gold']
    z['messages'].append(dict(role='assistant', content=answer))
    return z


def scene(k, split, task, j, occupied, pools):
    r = rule(k)
    rng = random.Random(f'scale64-scenes-20261021:{k["id"]}:{split}:{task}:{j}')
    # All-true AND fixes every visible field, so this is one unique scene.
    # Reserve it once for holdout/direct; fill other slots with novel all-omit scenes.
    mask = 0 if r['selection_topology'] == 'conjunctive_filter' and j == 15 and (split, task) != ('holdout', 'direct') else j
    for attempt in range(50000):
        obs = []
        for pos in range(4):
            wanted = bool(mask & (1 << pos))
            if r['selection_topology'] == 'interval_filter':
                lo, hi = r['conditions']
                edge = [lo['value'] - .25, lo['value'], lo['value'] + .25,
                        hi['value'] - .25, hi['value'], hi['value'] + .25]
                values = edge + [round(rng.uniform(lo['value'] - 5, hi['value'] + 5), 3) for _ in range(8)]
                values = [v for v in values if all(old.flags(r, dict(value=v))) == wanted]
                o = dict(value=rng.choice(values))
            else:
                patterns = [bits for bits in itertools.product((False, True), repeat=len(r['conditions']))
                            if (any(bits) if r['mode'] == 'any' else all(bits)) == wanted]
                bits = rng.choice(patterns)
                o = {}
                for c, matches in zip(r['conditions'], bits):
                    same = matches if c['operator'] == 'eq' else not matches
                    o[c['field']] = c['value'] if same else rng.choice([v for v in pools[c['field']] if v != c['value']])
            obs.append(o)
        sig = prior.signature(obs)
        if sig not in occupied:
            break
    else:
        raise RuntimeError(('scene generation exhausted', k['id'], split, task, j))
    occupied.add(sig)
    result = old.assignment(r, obs)
    assert result == truth(r, obs)
    all_results = ['; '.join(f'{c}={"KEEP" if x else "OMIT"}' for c, x in zip('ABCD', bits)) for bits in itertools.product((False, True), repeat=4)]
    neg = [v for v in all_results if v != result]; rng.shuffle(neg)
    positions = list(range(4)) * 4
    random.Random(f'scale64-letters:{k["id"]}:{split}:{task}').shuffle(positions)
    options = neg[:3]; options.insert(positions[j], result)
    accepts = random.Random(f'scale64-accept:{k["id"]}:{split}:{task}').sample(range(16), 4)
    report = result if j in accepts else rng.choice(neg)
    if task == 'diagnosis' and report != result and report not in options:
        options[next(i for i in range(4) if i != positions[j])] = report
    return dict(id=f'scale64:{split}:{k["id"]}:{task}:{j}', qid=k['id'], split=split, task=task,
        family=r['selection_topology'], keep_mask=mask, observations=obs, observation_signature=sig,
        results={k['id']: result}, options=options, submitted_report=report)


def validate(images=False):
    m = read(OUT / 'manifest.json'); assert m
    for p, h in m['sources'].items():
        assert sha(Path(p)) == h, p
    for p, h in m['files'].items():
        assert sha(OUT / p) == h, p
    if images:
        for r in rows(DATA / 'image_hashes.jsonl'):
            assert sha(DATA / r['path']) == r['source_sha256'] == r['repaired_sha256']
    return m


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'manifest.json').exists():
        return validate()
    allk = [r['content'] for r in rows(DATA / 'audit/rule_knowledge.jsonl')]
    previous_ids = set(read(prior.OUT / 'manifest.json')['ids'])
    eligible = defaultdict(list)
    for k in allk:
        r = rule(k); cs = r.get('conditions', []); t = r.get('selection_topology')
        if k['id'] in previous_ids or t not in FAMILIES:
            continue
        if t == 'disjunctive_filter' and {c['field'] for c in cs} != {'category', 'material'}:
            continue
        if t == 'conjunctive_filter' and {c['field'] for c in cs} != {'category', 'material', 'color'}:
            continue
        if t == 'include_with_exclusion' and {c['field'] for c in cs} != {'category', 'status'}:
            continue
        eligible[t].append(k)
    selected, eligibility = [], {}
    for t in FAMILIES:
        eligible[t].sort(key=lambda k: hashlib.sha256(('process-scale64-20261021:' + k['id']).encode()).hexdigest())
        assert len(eligible[t]) >= 16, (t, len(eligible[t]))
        eligibility[t] = [k['id'] for k in eligible[t]]
        selected += eligible[t][:16]
    ids = {k['id'] for k in selected}
    assert len(ids) == len({name(k) for k in selected}) == 64
    pools = defaultdict(set)
    for k in allk:
        for c in rule(k).get('conditions', []):
            if isinstance(c.get('value'), str): pools[c['field']].add(c['value'])
    pools = {f: sorted(v) for f, v in pools.items()}
    occupied, history = set(), []
    for p in rows(DATA / 'T2T/eval.jsonl'):
        o = prior.observed(p)
        if o: occupied.add(prior.signature(o))
    formal_count = len(occupied)
    for root in (ROOT / 'runs', Path('/home/duyuntao/xmodal_runs')):
        for p in sorted(root.rglob('*.jsonl')):
            if OUT in p.parents or p.stat().st_size > 8_000_000 or not any(x in p.name for x in ('blocks', 'probes', 'training')):
                continue
            try:
                rr = rows(p)
            except (ValueError, UnicodeError):
                continue
            found = set()
            for z in rr:
                o = z.get('observations') or (prior.observed(z) if z.get('messages') else None)
                if isinstance(o, list) and len(o) == 4 and all(isinstance(item, dict) and item for item in o):
                    found.add(prior.signature(o))
            if found:
                occupied.update(found)
                history.append(dict(path=str(p), sha256=sha(p), scenes=len(found)))
    forbidden = set(occupied)
    blocks, probes, definitions, process = [], [], [], []
    for k in selected:
        ff = facts(k); definitions += copy.deepcopy(ff)
        replay = [r for r in ff if r['channel'] == 'full_fact'][:8]
        replay += [r for r in ff if r['channel'] == 'atomic_fact' and r['id'].endswith(':0')]
        replay += [r for r in ff if r['channel'] == 'boundary_logic'][:2]
        assert len(replay) == 16
        process += copy.deepcopy(replay)
        for split in ('train', 'holdout'):
            for task in ('direct', 'diagnosis'):
                for j in range(16):
                    b = scene(k, split, task, j, occupied, pools); blocks.append(b)
                    p = old.practice(b, k, task)
                    p.update(condition='new_' + split, protocol='short', family=rule(k)['selection_topology'])
                    if split == 'train':
                        process += [old.as_training(p, 'direct_application'), trace(p, k, b)]
                        if j in (0, 5): probes.append(prior.reasoning(p))
                    else:
                        probes += [p, prior.reasoning(p)]
        probes.append(dict(id='recall:' + k['id'], qid=k['id'], condition='recall', input_mod='T', fmt='fact',
            protocol='recall', gold=mem.compact(k), family=rule(k)['selection_topology'], messages=mem.messages(mem.RECALL_Q.format(name=name(k)))))
        for key, question, answer in field_specs(k):
            probes.append(dict(id=f'field_probe:{k["id"]}:{key}', qid=k['id'], condition='field_recall', input_mod='T', fmt='fact',
                protocol='field', field=key, gold=answer, family=rule(k)['selection_topology'],
                messages=[dict(role='system', content='Return only the requested factual value of the learned rule.'),
                          dict(role='user', content=f'For the learned rule {name(k)}, what is its {question}?')]))
    labels = {r['id']: r for r in rows(DATA / 'audit/eval_labels.jsonl')}
    km = {k['id']: k for k in selected}
    for direction, mod in (('T2T', 'T'), ('T2V', 'V')):
        for p in rows(DATA / direction / 'eval.jsonl'):
            if p['qid'] not in ids: continue
            lab, k = labels[p['id']], km[p['qid']]
            z = ev.prepare_row(p, DATA, direction)
            z.update(id='original:' + mod + ':' + p['id'], source_id=p['id'], condition='original', input_mod=mod,
                fmt='mcq', protocol='short', gold=lab['answer'], task=lab['question_type'],
                evaluation_group=lab['evaluation_group'], family=rule(k)['selection_topology'])
            if mod == 'T':
                obs = prior.observed(p)
                assert obs and truth(rule(k), obs) == old.assignment(rule(k), obs)
                options = dict(re.findall(r'(?m)^([A-D])\. (.*)$', p['messages'][-1]['content']))
                assert truth(rule(k), obs) in options[lab['answer']]
                z['observations'] = obs
            probes += [z, prior.reasoning(z)]
    for rr in (definitions, process):
        random.Random(42).shuffle(rr)
        assert len(rr) == 5120
        assert all(not r.get('images') and '<image>' not in str(r['messages']) for r in rr)
    train_sig = {b['observation_signature'] for b in blocks if b['split'] == 'train'}
    hold_sig = {b['observation_signature'] for b in blocks if b['split'] == 'holdout'}
    assert len(train_sig) == len(hold_sig) == 2048 and not train_sig & hold_sig
    assert not (train_sig | hold_sig) & forbidden
    assert len(probes) == len({p['id'] for p in probes}) == 5312
    outputs = [('knowledge.jsonl', selected), ('blocks.jsonl', blocks), ('probes.jsonl', probes),
        ('definitions_training.jsonl', definitions), ('definition_replay_training.jsonl', definitions), ('process_training.jsonl', process)]
    for fn, rr in outputs: ev.save_rows(OUT / fn, rr)
    save(OUT / 'selection_audit.json', dict(eligibility_order=eligibility, selected_ids=sorted(ids),
        excluded_previous_confirmation_ids=sorted(previous_ids), historical_scene_sources=history,
        historical_exposure='No exclusion of all previous pilot IDs is claimed; clean base and no inherited historical adapters. Selection is hash based, never score based.'))
    ev.save_rows(OUT / 'forbidden_scene_signatures.jsonl', [dict(signature=s) for s in sorted(forbidden)])
    os.environ.update(environment('0', DATA))
    from swift.model import get_model_processor
    from swift.template import get_template
    _, processor = get_model_processor(str(MODELS['qwen3']), model_type='qwen3_vl', load_model=False)
    template = get_template(processor, template_type='qwen3_vl', max_length=8192, enable_thinking=False)
    template.set_mode('train')
    suffix = processor.tokenizer.encode('<|im_end|>\n', add_special_tokens=False)
    enc = []
    for arm, rr in [('definitions', definitions), ('definition_replay', definitions), ('process', process)]:
        for row in rr:
            z = template.encode(dict(messages=copy.deepcopy(row['messages'])))
            lab = [v for v in z['labels'] if v != -100]
            assert lab == processor.tokenizer.encode(row['messages'][-1]['content'], add_special_tokens=False) + suffix
            assert len(z['input_ids']) < 8192
            if row['channel'] == 'worked_application': assert len(lab) - len(suffix) <= 768
            enc.append(dict(arm=arm, id=row['id'], channel=row['channel'], input_tokens=len(z['input_ids']), supervised_tokens=len(lab)))
    ev.save_rows(OUT / 'encoding_audit.jsonl', enc)
    sources = [Path(__file__), Path(prior.__file__), Path(mem.__file__), Path(old.__file__), prior.PLUGIN,
        ROOT / 'scripts/run_v7_evaluation.py', ROOT / 'scripts/run_v7_full_experiment.py',
        ROOT / 'scripts/run_v7_fullparam_ep10.py', ROOT / 'scripts/build_v7_experiment_data.py',
        ROOT / 'scripts/v7_gpu_resources.py', ROOT / 'scripts/audit_v7_process_scale64.py',
        ROOT / 'scripts/audit_v7_process_confirmation.py', ROOT / 'CROSS_MODAL_RESEARCH_PROTOCOL.md',
        prior.OUT / 'manifest.json', DATA / 'audit/rule_knowledge.jsonl', DATA / 'audit/eval_labels.jsonl',
        DATA / 'image_hashes.jsonl'] + [DATA / d / 'eval.jsonl' for d in ('T2T', 'T2V', 'V2T', 'V2V')]
    base = dense_identity(MODELS['qwen3'])
    for image in rows(DATA / 'image_hashes.jsonl'):
        assert sha(DATA / image['path']) == image['source_sha256'] == image['repaired_sha256']
    manifest = dict(created_at=ev.now(), sources={str(p): sha(p) for p in sources},
        files={p.name: sha(p) for p in OUT.glob('*.jsonl')} | {'selection_audit.json': sha(OUT / 'selection_audit.json')},
        base=base, ids=sorted(ids), seeds=SEEDS, arms=ARMS, suffix_ids=suffix,
        selection='16 per topology in SHA256 order, excluding previous eight-rule confirmation; no score-dependent selection.',
        training=dict(modality='TEXT', rows_per_task=5120, epochs_per_task=3, steps_per_task=STEPS,
            batch=2, accumulation=2, common_lr=3e-5, sibling_lr=1e-5, rank=32, alpha=64, dropout=.05,
            max_length=8192, freeze_vision_aligner=True, fresh_optimizer_per_stage=True,
            token_totals={a: sum(r['supervised_tokens'] for r in enc if r['arm'] == a) for a in ARMS},
            process_scenes=2048, holdout_scenes=2048,
            caveat='Same scenes/files across seeds. Equal sibling rows/steps, different content and supervised tokens.'),
        generation=dict(direct=32, reasoning=768, recall=256, field=48, temperature=0, thinking=False, batch=4),
        counts=dict(Counter(p['condition'] + ':' + p['protocol'] + ':' + p['input_mod'] for p in probes)),
        evaluation_arms=10, requests_per_arm=5312, total_main_requests=53120,
        primary='Source TEXT application per seed and topology, process vs equal-step definition replay; closed-book original T2T/T2V. Short/reasoning separate.',
        gate='Per seed: new_holdout reasoning >=70%, >=15pp vs replay; original T2T reasoning >=60%, >=15pp vs replay; both format errors <=5%. All three seeds required for stable-source gate. No T2T>T2V requirement.',
        checkpoint='Final epoch3/checkpoint-3840 per task. No eval-based checkpoint or rule selection.',
        limits=['64 rules/four set-selection structures, not full1000 and not both original problem families.',
                'Two new structures relative to previous8; per-family results mandatory.',
                'No VISION learning group; no V2T/V2V.', 'Formal images and questions immutable.',
                'Same-rule application generalization, not unseen-rule generalization.',
                'Baseline truncation, exact-recall matching, different supervised tokens must be reported.'])
    save(OUT / 'manifest.json', manifest)
    save(OUT / 'data_audit.json', dict(status='pass', selected_rules=64, formal_scenes=formal_count,
        historical_forbidden_scenes=len(forbidden), train_scenes=2048, holdout_scenes=2048,
        original_images_verified=3000, encoding_and_assistant_mask_verified=True))
    snap = OUT / 'source_snapshots'; snap.mkdir(exist_ok=True)
    for p in sources:
        if p.suffix == '.py': shutil.copy2(p, snap / p.name)
    lines = ['# 64条规则、三个seed：冻结方案', '',
        'Qwen3-VL-4B，TEXT-only，seed42/43/44。每seed先5120行定义训练3epoch，再从最终权重分出定义复习和具体过程组，各5120行、3epoch。每阶段3840步，batch2×累积2，LoRA32/alpha64。', '',
        '每种结构16条：类别/材质OR、数值区间、多条件AND、包含＋排除。按哈希固定，排除上一轮8条，但不声称排除了所有历史开发规则。全部从干净基座开始。', '',
        '每规则32个新训练场景，每场景直接答案及具体过程各一行；另16行事实复习。每规则32个新留出场景，direct/diagnosis各16个。AND的全部KEEP只有一个唯一场景，仅放在留出direct；其余三个对应槽位补充新全OMIT场景。因此AND训练/留出diagnosis覆盖15种KEEP组合，其余结构覆盖16种。场景按无序候选集合全局隔离；单个候选可能重复。训练不使用正式原题、选项、答案或原图。', '',
        '固定同一套数据/场景，三个seed改变训练初始化和数据打乱。过程和定义复习为相同步数但监督token不同。正式原题每方向128题，两协议分别保存；每组5312次生成，共53120次，基座仅生成一次。', '',
        '来源应用门槛分别对每seed判断，三个seed均通过才称稳定扩展。无差值方向目标，不按测试挑checkpoint。没有VISION学习，不能报告V2T/V2V。四种都是集合选择结构，尚未覆盖原始数据的所有题族。', '',
        '## 复现', '', '```bash', '.venv/bin/python scripts/scale_v7_process_learning.py --stage prepare',
        '.venv/bin/python scripts/scale_v7_process_learning.py --stage audit',
        '.venv/bin/python -u scripts/scale_v7_process_learning.py --stage workflow', '```', '']
    (OUT / 'PLAN_ZH.md').write_text('\n'.join(lines) + '\n')
    examples = ['# 训练样例', '', '以下为独立构造的新练习，正式原题与图片不参与训练。', '']
    for family in FAMILIES:
        q = next(k['id'] for k in selected if rule(k)['selection_topology'] == family)
        sample = next(r for r in process if r['qid'] == q and r['channel'] == 'worked_application')
        examples += ['## ' + family, '', '```json', json.dumps(sample, ensure_ascii=False, indent=2), '```', '']
    (OUT / 'TRAINING_DATA_ZH.md').write_text('\n'.join(examples) + '\n')
    print('PREPARED', json.dumps(dict(rules=64, seeds=SEEDS, counts=manifest['counts'])), flush=True)


def smoke_steps(arm):
    # Batch size 2: process has five channels; definition arms have three.
    return 3 if arm == 'process' else 2


def checkpoint(seed, arm, smoke=False):
    return OUT / f'seed{seed}' / ('smoke_' + arm if smoke else arm) / 'checkpoint' / f'checkpoint-{smoke_steps(arm) if smoke else STEPS}'


def train(seed, arm, gpu, smoke=False):
    m = validate(); cp = checkpoint(seed, arm, smoke); folder = cp.parent.parent
    folder.mkdir(parents=True, exist_ok=True)
    with ev.run_lock(folder / 'train.lock'):
        dataset = OUT / (arm + '_training.jsonl')
        sample = []
        if smoke:
            allrows = rows(dataset)
            for channel in sorted({r['channel'] for r in allrows}):
                candidates = [r for r in allrows if r['channel'] == channel]
                sample.append(max(candidates, key=lambda r: len(r['messages'][-1]['content'])))
            assert len(sample) <= 2 * smoke_steps(arm)
            dataset = folder / 'smoke_input.jsonl'; ev.save_rows(dataset, sample)
        if not (cp / 'adapter_model.safetensors').exists():
            recovery = [p for p in cp.parent.glob('checkpoint-*') if all((p / n).is_file() for n in
                ('optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json', 'adapter_model.safetensors'))]
            resume = max(recovery, key=lambda p: read(p / 'trainer_state.json')['global_step']) if recovery and not smoke else None
            cmd = train_command('qwen3', 'sft', dataset, cp.parent, epochs=3, resume=resume)
            for flag, value in [('--per_device_train_batch_size', 2), ('--gradient_accumulation_steps', 1 if smoke else 2),
                ('--learning_rate', 3e-5 if arm == 'definitions' else 1e-5), ('--lora_rank', 32), ('--lora_alpha', 64),
                ('--save_strategy', 'steps'), ('--logging_steps', 1), ('--seed', seed), ('--data_seed', seed)]:
                cmd[cmd.index(flag) + 1] = str(value)
            if arm != 'definitions' and not resume:
                initial = checkpoint(seed, 'definitions'); assert initial.is_dir()
                cmd += ['--adapters', str(initial), '--load_args', 'false']
            cmd += ['--save_steps', str(smoke_steps(arm)) if smoke else '1280', '--attn_impl', 'sdpa', '--external_plugins', str(prior.PLUGIN),
                '--loss_type', 'v7_suffix_control', '--packing', 'false', '--padding_free', 'false',
                '--use_logits_to_keep', 'false', '--dataloader_num_workers', '0',
                '--dataloader_persistent_workers', 'false', '--report_to', 'none']
            if smoke:
                cmd += ['--max_steps', str(smoke_steps(arm))]; cmd[cmd.index('--warmup_ratio') + 1] = '0'
            env = environment(gpu, DATA)
            env.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', V7_CONTENT_SUFFIX_IDS=json.dumps(m['suffix_ids']),
                V7_CONTENT_LOSS_AUDIT=str(folder / 'loss_audit.jsonl'), TMPDIR=str(OUT / 'tmp'))
            (OUT / 'tmp').mkdir(exist_ok=True)
            save(folder / 'state.json', dict(status='waiting_for_gpu', gpu=gpu, command=cmd))
            with reserve(prior.LEDGER, gpu, str(folder) + ':train', 22000) as res:
                with (folder / 'train.log').open('a') as log:
                    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                    res.bind(proc.pid)
                    save(folder / 'state.json', dict(status='running', pid=proc.pid, gpu=gpu, command=cmd))
                    code = proc.wait()
                if code:
                    save(folder / 'state.json', dict(status='failed', returncode=code))
                    raise RuntimeError((seed, arm, code, str(folder / 'train.log')))
        state, args = read(cp / 'trainer_state.json'), read(cp / 'args.json')
        assert state['global_step'] == (smoke_steps(arm) if smoke else STEPS)
        assert smoke or abs(state['epoch'] - 3) < 1e-6
        assert args['freeze_vit'] and args['freeze_aligner'] and args['lora_rank'] == 32
        assert args['seed'] == args['data_seed'] == seed
        assert not args.get('adapters') if arm == 'definitions' else args['adapters'] == [str(checkpoint(seed, 'definitions'))]
        losses = [r['loss'] for r in rows(cp.parent / 'logging.jsonl') if 'loss' in r]
        assert losses and all(math.isfinite(v) for v in losses)
        from safetensors import safe_open
        with safe_open(str(cp / 'adapter_model.safetensors'), framework='pt', device='cpu') as weights:
            bs = [k for k in weights.keys() if 'lora_B' in k]
            assert bs and all(bool(weights.get_tensor(k).count_nonzero()) for k in bs)
            assert not any('.visual.' in k or '.merger.' in k for k in weights.keys())
        save(folder / 'checkpoint_manifest.json', dict(model=ev.checkpoint(cp), epoch=state['epoch'], step=state['global_step'],
            first_loss=losses[0], last_loss=losses[-1], nonzero_lora_B=len(bs), losses_finite=True,
            initial='clean_base' if arm == 'definitions' else ev.checkpoint(checkpoint(seed, 'definitions'))))
        save(folder / 'state.json', dict(status='complete', checkpoint=str(cp), ended_at=ev.now()))
    if smoke:
        mem.OUT = OUT
        pp = []
        for mod in ('T', 'V'):
            z = copy.deepcopy(next(p for p in rows(OUT / 'probes.jsonl') if (p['condition'], p['input_mod'], p['protocol']) == ('original', mod, 'short')))
            z['id'] = 'smoke:' + z['id']; pp.append(z)
        runner = mem.Runner(folder / 'reload', gpu, f'smoke_{seed}_{arm}')
        if runner.seen != {p['id'] for p in pp}:
            with reserve(prior.LEDGER, gpu, str(folder) + ':reload', 22000) as res:
                res.bind(os.getpid()); runner.load(cp, adapter=True); runner.infer(pp, 'direct')
        assert len(runner.rr) == 2 and all(r['response'].strip() for r in runner.rr)
        save(folder / 'validated.json', dict(status='pass', text_image_reload=True, smoke_rows=len(sample)))


def evaluate(seed, arm, gpu):
    m = validate(); mem.OUT = OUT
    folder = OUT / 'base/evaluation' if arm == 'base' else OUT / f'seed{seed}' / arm / 'evaluation'
    folder.mkdir(parents=True, exist_ok=True)
    with ev.run_lock(folder / 'eval.lock'):
        runner = mem.Runner(folder, gpu, 'base' if arm == 'base' else f'{seed}_{arm}')
        pp = rows(OUT / 'probes.jsonl')
        if runner.seen != {p['id'] for p in pp}:
            save(folder / 'state.json', dict(status='waiting_for_gpu', expected=len(pp), gpu=gpu))
            with reserve(prior.LEDGER, gpu, str(folder) + ':eval', 22000) as res:
                res.bind(os.getpid()); runner.load(MODELS['qwen3'] if arm == 'base' else checkpoint(seed, arm), adapter=arm != 'base')
                for protocol, stage in [('short', 'direct'), ('reasoning', 'reasoning'), ('recall', 'recall'), ('field', 'field')]:
                    runner.infer([p for p in pp if p['protocol'] == protocol], stage)
        assert len(runner.rr) == len(runner.seen) == m['requests_per_arm']
        save(folder / 'state.json', dict(status='complete', n=len(pp), ended_at=ev.now()))


def audit():
    m = validate(images=True)
    km = {k['id']: k for k in rows(OUT / 'knowledge.jsonl')}
    selection = read(OUT / 'selection_audit.json')
    assert len(km) == 64
    for family, ordered in selection['eligibility_order'].items():
        assert ordered == sorted(ordered, key=lambda q: hashlib.sha256(('process-scale64-20261021:' + q).encode()).hexdigest())
        assert set(ordered[:16]) <= set(km)
    assert not set(km) & set(selection['excluded_previous_confirmation_ids'])
    assert Counter(rule(k)['selection_topology'] for k in km.values()) == dict.fromkeys(FAMILIES, 16)
    blocks = rows(OUT / 'blocks.jsonl'); bm = {b['id']: b for b in blocks}
    assert len(blocks) == len(bm) == 4096
    signatures = {b['observation_signature'] for b in blocks}
    assert len(signatures) == 4096
    forbidden = {r['signature'] for r in rows(OUT / 'forbidden_scene_signatures.jsonl')}
    assert not signatures & forbidden
    groups = defaultdict(list)
    for b in blocks:
        r = rule(km[b['qid']])
        assert b['observation_signature'] == prior.signature(b['observations'])
        flags = [[OPS[c['operator']](o[c['field']], c['value']) for c in r['conditions']] for o in b['observations']]
        keep = [any(f) if r['mode'] == 'any' else all(f) for f in flags]
        answer = '; '.join(c + '=' + ('KEEP' if v else 'OMIT') for c, v in zip('ABCD', keep))
        assert b['results'][b['qid']] == answer == truth(r, b['observations'])
        assert len(set(b['options'])) == 4 and b['options'].count(answer) == 1
        assert b['keep_mask'] == sum(int(v) << j for j, v in enumerate(keep))
        slot = int(b['id'].rsplit(':', 1)[-1])
        expected_mask = 0 if b['family'] == 'conjunctive_filter' and slot == 15 and (b['split'], b['task']) != ('holdout', 'direct') else slot
        assert b['keep_mask'] == expected_mask
        groups[b['qid'], b['split'], b['task']].append(b)
    for bb in groups.values():
        assert len(bb) == 16
        assert Counter(b['options'].index(b['results'][b['qid']]) for b in bb) == dict.fromkeys(range(4), 4)
        assert sum(b['submitted_report'] == b['results'][b['qid']] for b in bb) == 4
    for arm in ARMS:
        rr = rows(OUT / (arm + '_training.jsonl'))
        assert len(rr) == len({r['id'] for r in rr}) == 5120
        for r in rr:
            assert not r.get('images') and '<image>' not in str(r['messages'])
            assert r['messages'][-1]['role'] == 'assistant'
            if r.get('block'):
                b = bm[r['block']]; assert b['split'] == 'train'
                short = old.practice(b, km[r['qid']], b['task'])
                expected = old.as_training(short, 'direct_application') if r['channel'] == 'direct_application' else trace(short, km[r['qid']], b)
                assert r['messages'] == expected['messages']
    assert rows(OUT / 'definitions_training.jsonl') == rows(OUT / 'definition_replay_training.jsonl')
    pp = rows(OUT / 'probes.jsonl'); assert len(pp) == len({p['id'] for p in pp}) == 5312
    for direction, mod in [('T2T', 'T'), ('T2V', 'V')]:
        formal = {r['id']: ev.prepare_row(r, DATA, direction) for r in rows(DATA / direction / 'eval.jsonl') if r['qid'] in km}
        selected_probes = [p for p in pp if (p['condition'], p['input_mod'], p['protocol']) == ('original', mod, 'short')]
        assert len(formal) == len(selected_probes) == 128
        for p in selected_probes:
            f = formal[p['source_id']]
            assert p['messages'] == f['messages'] and p.get('images', []) == f.get('images', [])
    assert dense_identity(MODELS['qwen3']) == m['base']
    result = dict(status='pass', rules=64, families=4, seeds=SEEDS, scenes=4096, train_holdout_globally_disjoint=True,
        forbidden_scenes_excluded=True, independent_truth_and_balance_verified=True, original_payloads_unchanged=True,
        original_images_verified=3000, base_identity_verified=True, ended_at=ev.now())
    save(OUT / 'independent_data_audit.json', result)
    print(json.dumps(result), flush=True)


def child(stage, seed, arm, gpu, smoke=False):
    cmd = [str(PY), '-u', str(Path(__file__)), '--stage', stage, '--seed', str(seed), '--arm', arm, '--gpu', str(gpu)]
    if smoke: cmd.append('--smoke')
    subprocess.run(cmd, cwd=ROOT, check=True)


def workflow():
    OUT.mkdir(parents=True, exist_ok=True)
    with ev.run_lock(OUT / 'workflow.lock'):
        validate(); assert read(OUT / 'independent_data_audit.json')['status'] == 'pass'
        # Failed queue markers belong to previous attempts, not this resumed run.
        for gpu in (0, 1):
            marker = OUT / f'queue{gpu}_failed.json'
            if marker.exists():
                history = OUT / 'failure_history'; history.mkdir(exist_ok=True)
                marker.rename(history / f'{marker.stem}_{ev.now().replace(":", "-")}.json')
        save(OUT / 'workflow_state.json', dict(status='running', pid=os.getpid(), started_at=ev.now()))
        # Explicit immutable queues: one owned task per GPU; no duplicate prefetch.
        queues = [[], [('evaluate', 42, 'base', False)]]
        for seed in SEEDS:
            for arm in ARMS: queues[0].append(('train', seed, arm, seed == 42))
            for arm in ('process', 'definition_replay', 'definitions'):
                queues[0 if seed == 44 else 1].append(('evaluate', seed, arm, False))
        # GPU1 starts with prior-eight diagnostics, launched separately by this task.
        def worker(gpu, tasks):
            import time
            if gpu == 1:
                diagnostic = ROOT / 'runs/v7_process_image_diagnostic_20261021'
                while read(diagnostic / 'completion_audit.json').get('status') != 'pass':
                    launch = read(diagnostic / 'launcher.json')
                    from v7_gpu_resources import alive
                    if launch and not alive(launch.get('pid')):
                        raise RuntimeError('Image diagnostic exited before completing; inspect its run.log')
                    time.sleep(10)
            for stage, seed, arm, smoke in tasks:
                if stage == 'evaluate' and arm != 'base':
                    while read(checkpoint(seed, arm).parent.parent / 'state.json').get('status') != 'complete':
                        state = read(checkpoint(seed, arm).parent.parent / 'state.json')
                        if state.get('status') == 'failed': raise RuntimeError((seed, arm, 'training failed'))
                        if (OUT / 'queue0_failed.json').exists(): raise RuntimeError('Training queue failed')
                        time.sleep(10)
                save(OUT / f'queue{gpu}_state.json', dict(status='running', gpu=gpu, stage=stage, seed=seed, arm=arm, updated_at=ev.now()))
                if smoke: child('train', seed, arm, gpu, True)
                child(stage, seed, arm, gpu)
            save(OUT / f'queue{gpu}_state.json', dict(status='complete', ended_at=ev.now()))
        def guarded(gpu):
            try:
                worker(gpu, queues[gpu])
            except Exception:
                save(OUT / f'queue{gpu}_failed.json', dict(error=traceback.format_exc(), ended_at=ev.now()))
                raise
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(guarded, gpu) for gpu in (0, 1)]
                for future in futures: future.result()
            score()
            subprocess.run([str(PY), str(ROOT / 'scripts/audit_v7_process_scale64.py')], cwd=ROOT, check=True)
            save(OUT / 'workflow_state.json', dict(status='complete', ended_at=ev.now()))
        except Exception:
            save(OUT / 'workflow_state.json', dict(status='failed', error=traceback.format_exc(), ended_at=ev.now()))
            raise


def score():
    m = validate(images=True)
    probes = {p['id']: p for p in rows(OUT / 'probes.jsonl')}
    results, summary, comparisons, per_rule = [], {}, [], {}
    tasks = [('base', OUT / 'base/evaluation')]
    tasks += [(f'{seed}:{arm}', OUT / f'seed{seed}' / arm / 'evaluation') for seed in SEEDS for arm in ARMS]
    for task, folder in tasks:
        assert read(folder / 'state.json')['status'] == 'complete'
        rr = rows(folder / 'predictions.jsonl')
        assert len(rr) == len({r['id'] for r in rr}) == 5312
        identity = read(folder / 'eval_manifest.json')['model']
        if task == 'base': assert identity == m['base']
        else:
            seed, arm = task.split(':'); cp = checkpoint(int(seed), arm)
            assert identity == ev.checkpoint(cp)
            state = read(cp / 'trainer_state.json')
            assert state['global_step'] == STEPS and state['epoch'] == 3
        for r in rr:
            p = probes[r['id']]
            assert r['input_sha256'] == ev.digest(dict(messages=p['messages'], images=p.get('images', [])))
            assert all(r[key] == p[key] for key in ('gold', 'qid', 'condition', 'protocol', 'input_mod'))
            text = r['response'].strip()
            if p['protocol'] == 'short': pred = text if text in ('A', 'B', 'C', 'D') else None
            elif p['protocol'] == 'reasoning':
                answers = re.findall(r'(?im)^\s*Answer:\s*([A-D])\s*[.!]?\s*$', text)
                pred = answers[0] if len(answers) == 1 else None
            else: pred = None
            good = ' '.join(text.lower().split()) == ' '.join(p['gold'].lower().split()) if p['fmt'] == 'fact' else pred == p['gold']
            results.append(dict(r, arm=task, scored_answer=pred, scored_correct=good,
                                scored_format_error=p['fmt'] != 'fact' and pred is None))
        sub = [r for r in results if r['arm'] == task]
        summary[task] = {}
        per_rule[task] = {}
        for key in m['counts']:
            ss = [r for r in sub if ':'.join(r[c] for c in ('condition', 'protocol', 'input_mod')) == key]
            summary[task][key] = prior.metric(ss) | dict(
                by_task={t: prior.metric([r for r in ss if r.get('task') == t]) for t in ('direct', 'diagnosis')},
                by_family={t: prior.metric([r for r in ss if r['family'] == t]) for t in FAMILIES})
            for q in m['ids']:
                per_rule[task].setdefault(q, {})[key] = prior.metric([r for r in ss if r['qid'] == q])
    for seed in SEEDS:
        before, after = f'{seed}:definition_replay', f'{seed}:process'
        for key in m['counts']:
            sub = [r for r in results if ':'.join(r[c] for c in ('condition', 'protocol', 'input_mod')) == key]
            paired = prior.paired([r for r in sub if r['arm'] == before], [r for r in sub if r['arm'] == after])
            paired['caveat'] = '64 rule clusters per seed; three seeds reuse the same rules/scenes. Descriptive within-seed comparison, no multiplicity adjustment or population-level claim.'
            comparisons.append(dict(seed=seed, before=before, after=after, test=key, **paired))
    gates = {}
    for seed in SEEDS:
        process, replay = summary[f'{seed}:process'], summary[f'{seed}:definition_replay']
        gates[seed] = (process['new_holdout:reasoning:T']['accuracy'] >= .7 and
            process['new_holdout:reasoning:T']['accuracy'] - replay['new_holdout:reasoning:T']['accuracy'] >= .15 and
            process['original:reasoning:T']['accuracy'] >= .6 and
            process['original:reasoning:T']['accuracy'] - replay['original:reasoning:T']['accuracy'] >= .15 and
            all(process[k]['format_errors'] / process[k]['n'] <= .05 for k in ('new_holdout:reasoning:T', 'original:reasoning:T')))
    ev.save_rows(OUT / 'scored_predictions.jsonl', results)
    save(OUT / 'summary.json', summary); save(OUT / 'per_rule_metrics.json', per_rule)
    save(OUT / 'paired_comparisons.json', comparisons)
    aggregate = {}
    for key in m['counts']:
        values = [summary[f'{seed}:process'][key]['accuracy'] for seed in SEEDS]
        mean = sum(values) / 3
        aggregate[key] = dict(seed_accuracies=values, mean=mean,
            sample_std=(sum((v - mean)**2 for v in values) / 2)**.5, min=min(values), max=max(values))
    save(OUT / 'seed_summary.json', aggregate)
    with (OUT / 'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['arm', 'test', 'n', 'correct', 'accuracy', 'format_errors', 'valid_accuracy', 'truncated'])
        writer.writeheader()
        for arm, tests in summary.items():
            for test, z in tests.items(): writer.writerow(dict(arm=arm, test=test, **{k: z[k] for k in writer.fieldnames[2:]}))
    lines = ['# 64规则、3 seed：TEXT学习结果', '',
        '全部固定最终epoch3检查点；短答案与推理分开。三seed来源应用门槛：' + ('全部通过' if all(gates.values()) else '未全部通过') + '。', '',
        '| seed | 新留出推理 | 新留出短答案 | 原题T2T推理 | 原题T2V推理 | 来源门槛 |', '|---|---:|---:|---:|---:|---|']
    for seed in SEEDS:
        z = summary[f'{seed}:process']
        vals = [f'{z[k]["correct"]}/{z[k]["n"]}（{z[k]["accuracy"]:.1%}）' for k in
            ('new_holdout:reasoning:T', 'new_holdout:short:T', 'original:reasoning:T', 'original:reasoning:V')]
        lines.append('| ' + str(seed) + ' | ' + ' | '.join(vals) + ' | ' + str(gates[seed]) + ' |')
    lines += ['', '## 四结构分项：过程组新留出推理', '', '| 结构 | seed42 | seed43 | seed44 |', '|---|---:|---:|---:|']
    for family in FAMILIES:
        vals = [summary[f'{seed}:process']['new_holdout:reasoning:T']['by_family'][family] for seed in SEEDS]
        lines.append('| ' + family + ' | ' + ' | '.join(f'{z["correct"]}/{z["n"]}（{z["accuracy"]:.1%}）' for z in vals) + ' |')
    lines += ['', '所有基座/定义/复习对照、格式错误、截断、direct/diagnosis及各结构分项见summary.json。按规则配对结果见paired_comparisons.json；种子均值及离散度见seed_summary.json，三个seed不是三倍独立测试题。', '',
        '同一步数但监督token不同；新增加AND和排除结构，不能当成上一轮8条的完全同条件复现。64规则/四种集合选择结构，不能代表1000条全部题族。无VISION学习组；图片低分仍可能混合读图、知识调用、执行与选项映射。不给材料的主分数与事后oracle诊断严格分开。']
    (OUT / 'report_zh.md').write_text('\n'.join(lines) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    for i, seed in enumerate(SEEDS):
        for j, arm in enumerate(ARMS):
            logs = [r for r in rows(checkpoint(seed, arm).parent / 'logging.jsonl') if 'loss' in r]
            axes[i, j].plot(range(1, len(logs) + 1), [r['loss'] for r in logs], alpha=.6)
            axes[i, j].set(title=f'seed{seed} {arm}', xlabel='Update', ylabel='Equal-row CE')
    fig.tight_layout(); fig.savefig(OUT / 'training_loss.png', dpi=150); plt.close(fig)
    save(OUT / 'completion_audit.json', dict(status='pass', total_predictions=len(results),
        images_verified=3000, source_gate_per_seed=gates, stable_source_gate=all(gates.values()),
        final_checkpoints_only=True, prompt_hashes_and_ids_verified=True, ended_at=ev.now()))
    print('SCORED', len(results), 'gates', gates, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=('prepare', 'audit', 'train', 'evaluate', 'workflow', 'score'), required=True)
    ap.add_argument('--seed', type=int, choices=SEEDS, default=42)
    ap.add_argument('--arm', choices=('base',) + ARMS, default='definitions')
    ap.add_argument('--gpu', default='0'); ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.stage == 'prepare': prepare()
    elif a.stage == 'audit': audit()
    elif a.stage == 'train':
        assert a.arm != 'base'; train(a.seed, a.arm, a.gpu, a.smoke)
    elif a.stage == 'evaluate': evaluate(a.seed, a.arm, a.gpu)
    elif a.stage == 'workflow': workflow()
    else: score()
