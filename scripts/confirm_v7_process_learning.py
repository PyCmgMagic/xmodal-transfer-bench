#!/usr/bin/env python3
"""Frozen, clean-base TEXT process-learning confirmation on unused pilot rules."""
import argparse
import copy
import hashlib
import itertools
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import run_v7_evaluation as ev
import explore_v7_memory_router as mem
import pilot_v7_learning_first as old
from build_v7_experiment_data import name
from run_v7_full_experiment import ROOT, MODELS, environment, rows, read, save, sha, train_command
from run_v7_fullparam_ep10 import dense_identity
from v7_gpu_resources import reserve

OUT = ROOT / 'runs/v7_process_confirmation_20261020'
DATA = ROOT / 'rules_repaired_single_image_v7'
LEDGER = ROOT / 'runs/v7_full_experiment'
PY = ROOT / '.venv/bin/python'
PLUGIN = ROOT / 'scripts/v7_content_loss.py'
ARMS = ('base', 'definitions', 'definition_replay', 'process')
TRAIN_ARMS = ARMS[1:]
SYSTEM = 'Apply the learned named rule. Explain its conditions, check every candidate, then give the requested final option.'
SUFFIX = '\n\nExplain the rule and the candidate checks. End with Answer: followed by one uppercase option letter.'
FORMS = (
    'State the complete definition of {name}.',
    'Explain the stored rule {name}.',
    'Recall {name}: what precisely qualifies an item?',
    'Give the selection criteria and output convention for {name}.',
    'Describe the requirements of the rule called {name}.',
    'What conditions define {name}?',
    'Write a reference entry for {name}.',
    'Tell me the exact meaning of {name} in the learned rule system.',
)
DETAILS = (' Include every condition.', ' Retain all exact values.', ' State its logical connective.',
           ' Include its output convention.', ' Do not replace the definition with an example.',
           ' Be precise about the parameters.')


def write(p, rr):
    ev.save_rows(p, rr)


def signature(obs):
    normalized = [{k: int(v) if isinstance(v, float) and v.is_integer() else v for k, v in o.items()} for o in obs]
    return ev.digest(sorted(json.dumps(o, sort_keys=True) for o in normalized))


def observed(p):
    text = p['messages'][-1]['content']
    if 'Test input:\n' not in text:
        return None
    text = text.split('Test input:\n', 1)[1].split('\n\nOptions:', 1)[0].rstrip('.')
    parts = re.findall(r'([A-D]) has (.*?)(?=; [A-D] has|$)', text)
    if len(parts) != 4:
        return None
    obs = []
    for _, desc in parts:
        o = dict(re.findall(r'(\w+): ([^,;]+)', desc))
        try:
            if 'value' in o:
                o['value'] = float(o['value'])
        except ValueError:
            return None
        if not o:
            return None
        obs.append(o)
    return obs


def independent(k, obs):
    r = json.loads(k['fact']['quote'])
    if r['selection_topology'] == 'disjunctive_filter':
        a, b = r['conditions']
        keep = [o[a['field']] == a['value'] or o[b['field']] == b['value'] for o in obs]
    else:
        lo = next(c for c in r['conditions'] if c['operator'] in ('gt', 'ge'))
        hi = next(c for c in r['conditions'] if c['operator'] in ('lt', 'le'))
        keep = [(o['value'] > lo['value'] or o['value'] == lo['value'] and lo['operator'] == 'ge') and
                (o['value'] < hi['value'] or o['value'] == hi['value'] and hi['operator'] == 'le') for o in obs]
    return '; '.join(f'{c}={"KEEP" if v else "OMIT"}' for c, v in zip('ABCD', keep))


def field_specs(k):
    r = json.loads(k['fact']['quote'])
    if r['mode'] == 'any':
        return [(c['field'], 'required ' + c['field'], str(c['value'])) for c in r['conditions']] + [
            ('logic', 'logical connective (AND or OR)', 'OR'), ('both', 'decision if both conditions match', 'KEEP'),
            ('neither', 'decision if neither condition matches', 'OMIT'), ('scope', 'scope of checking', 'each candidate independently')]
    lo, hi = r['conditions']
    ops = dict(ge='greater than or equal to', gt='strictly greater than', lt='strictly less than', le='less than or equal to')
    return [('lower', 'lower endpoint value', str(lo['value'])), ('upper', 'upper endpoint value', str(hi['value'])),
            ('lower_operator', 'lower endpoint comparison', ops[lo['operator']]),
            ('upper_operator', 'upper endpoint comparison', ops[hi['operator']]),
            ('logic', 'logical connective (AND or OR)', 'AND'), ('scope', 'scope of checking', 'each candidate independently')]


def logic_specs(k):
    r = json.loads(k['fact']['quote'])
    if r['mode'] == 'any':
        return [('Does exactly one matching condition suffice?', 'Yes. Either condition is sufficient.'),
                ('Does matching both conditions qualify?', 'Yes. Both conditions matching qualifies.'),
                ('Must both conditions match?', 'No. One matching condition is sufficient.'),
                ('Does matching neither condition qualify?', 'No. Matching neither condition is rejected.')]
    lo, hi = r['conditions']
    return [('Is the lower endpoint included?', 'Yes.' if lo['operator'] == 'ge' else 'No.'),
            ('Is the upper endpoint included?', 'Yes.' if hi['operator'] == 'le' else 'No.'),
            ('Must both endpoint comparisons hold?', 'Yes. Both comparisons must hold.'),
            ('Does a value outside the interval qualify?', 'No. A value outside the interval is rejected.')]


def fact_rows(k):
    result = []
    for j, (form, detail) in enumerate(itertools.product(FORMS, DETAILS)):
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


def history_inventory():
    used = set()
    inventory = []
    for root in (ROOT / 'runs', Path('/home/duyuntao/xmodal_runs')):
        for p in sorted(root.rglob('*.jsonl')):
            if OUT in p.parents or p.stat().st_size > 8_000_000 or not any(s in p.name for s in ('train', 'knowledge', 'probe', 'block', 'eval', 'data')):
                continue
            try:
                ids = set()
                for r in rows(p):
                    if r.get('qid'):
                        ids.add(r['qid'])
                    elif isinstance(r.get('content'), dict) and 'fact' in r['content']:
                        ids.add(r['content']['id'])
                    elif 'fact' in r:
                        ids.add(r['id'])
                    if len(ids) > 64:
                        break
                if ids and len(ids) <= 64:
                    used |= ids
                    inventory.append(dict(path=str(p), sha256=sha(p), rule_ids=sorted(ids)))
            except (ValueError, KeyError, TypeError):
                continue
    return used, inventory


def make_scene(k, split, task, j, occupied, pools):
    r = json.loads(k['fact']['quote'])
    rng = random.Random(f'process-confirmation-42:{k["id"]}:{split}:{task}:{j}')
    for attempt in range(30000):
        obs = []
        for pos in range(4):
            wanted_keep = bool(j & (1 << pos))
            if r['mode'] == 'any':
                status = rng.choice([(1, 0), (0, 1), (1, 1)]) if wanted_keep else (0, 0)
                o = {c['field']: c['value'] if flag else rng.choice([v for v in pools[c['field']] if v != c['value']])
                     for c, flag in zip(r['conditions'], status)}
            else:
                lo, hi = r['conditions']
                edge = [lo['value'] - .25, lo['value'], lo['value'] + .25,
                        hi['value'] - .25, hi['value'], hi['value'] + .25]
                candidates = edge + [round(rng.uniform(lo['value'] - 5, hi['value'] + 5), 3) for _ in range(8)]
                candidates = [v for v in candidates if all(old.flags(r, dict(value=v))) == wanted_keep]
                o = dict(value=rng.choice(candidates))
            obs.append(o)
        sig = signature(obs)
        if sig not in occupied:
            break
    else:
        raise RuntimeError(('scene generation exhausted', k['id'], split, task, j))
    occupied.add(sig)
    result = old.assignment(r, obs)
    assert result == independent(k, obs)
    all_results = ['; '.join(f'{c}={"KEEP" if x else "OMIT"}' for c, x in zip('ABCD', bits)) for bits in itertools.product((False, True), repeat=4)]
    neg = [v for v in all_results if v != result]
    rng.shuffle(neg)
    positions = list(range(4)) * 4
    random.Random(f'process-confirmation-letters:{k["id"]}:{split}:{task}').shuffle(positions)
    options = neg[:3]
    options.insert(positions[j], result)
    accepts = random.Random(f'process-confirmation-accept:{k["id"]}:{split}:{task}').sample(range(16), 4)
    report = result if j in accepts else rng.choice(neg)
    if task == 'diagnosis' and report != result and report not in options:
        options[next(i for i in range(4) if i != positions[j])] = report
    assert len(set(options)) == 4
    return dict(id=f'confirm:{split}:{k["id"]}:{task}:{j}', qid=k['id'], split=split, task=task,
                observations=obs, observation_signature=sig, results={k['id']: result}, options=options, submitted_report=report)


def reasoning(p):
    z = copy.deepcopy(p)
    z['id'] = 'reasoning:' + p['id']
    z['protocol'] = 'reasoning'
    assert z['messages'][-1]['content'].endswith(ev.SUFFIX)
    z['messages'][-1]['content'] = z['messages'][-1]['content'][:-len(ev.SUFFIX)] + SUFFIX
    z['messages'][0]['content'] = SYSTEM
    return z


def trace_row(p, k, b):
    z = reasoning(p)
    z['id'] = 'trace:' + p['id']
    z['channel'] = 'worked_application'
    r = json.loads(k['fact']['quote'])
    trace = 'Rule: ' + mem.compact(k) + '\n'
    for letter, o in zip('ABCD', b['observations']):
        flags = old.flags(r, o)
        keep = any(flags) if r['mode'] == 'any' else all(flags)
        parts = [f'{c["field"]} {o[c["field"]]} '+dict(eq='==', ge='>=', gt='>', lt='<', le='<=')[c['operator']] +
                 f' {c["value"]} = {"YES" if flag else "NO"}' for c, flag in zip(r['conditions'], flags)]
        trace += f'{letter}: ' + '; '.join(parts) + f'; {"OR" if r["mode"] == "any" else "AND"} => {"KEEP" if keep else "OMIT"}.\n'
    if b['task'] == 'diagnosis':
        trace += ('Accept report.' if b['submitted_report'] == b['results'][k['id']] else 'Replace report.') + '\n'
    trace += 'Result: ' + b['results'][k['id']] + '.\nAnswer: ' + p['gold']
    z['messages'].append(dict(role='assistant', content=trace))
    return z


def validate(images=False):
    m = read(OUT / 'manifest.json')
    assert m
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
        validate()
        return
    allk = [r['content'] for r in rows(DATA / 'audit/rule_knowledge.jsonl')]
    used, history = history_inventory()
    selected = []
    eligibility = {}
    for topology in ('disjunctive_filter', 'interval_filter'):
        candidates = []
        for k in allk:
            if k['id'] in used:
                continue
            try:
                r = json.loads(k['fact']['quote'])
            except ValueError:
                continue
            if r.get('selection_topology') != topology or len(r.get('conditions', [])) != 2:
                continue
            if topology == 'disjunctive_filter' and ({c['field'] for c in r['conditions']} != {'category', 'material'} or any(c['operator'] != 'eq' for c in r['conditions'])):
                continue
            if topology == 'interval_filter' and ({c['field'] for c in r['conditions']} != {'value'} or
                r['conditions'][0]['operator'] not in ('ge', 'gt') or r['conditions'][1]['operator'] not in ('lt', 'le')):
                continue
            candidates.append(k)
        candidates.sort(key=lambda k: hashlib.sha256(('process-confirmation-42:' + k['id']).encode()).hexdigest())
        assert len(candidates) >= 4, (topology, len(candidates))
        eligibility[topology] = [k['id'] for k in candidates]
        selected += candidates[:4]
    ids = {k['id'] for k in selected}
    assert len(ids) == 8 and not ids & used
    pools = defaultdict(set)
    for k in allk:
        try:
            r = json.loads(k['fact']['quote'])
            for c in r.get('conditions', []):
                if c.get('operator') == 'eq' and isinstance(c.get('value'), str):
                    pools[c['field']].add(c['value'])
        except ValueError:
            pass
    pools = {f: sorted(v) for f, v in pools.items()}
    occupied = set()
    formal_parsed = 0
    for p in rows(DATA / 'T2T/eval.jsonl'):
        o = observed(p)
        if o:
            occupied.add(signature(o))
            formal_parsed += 1
    # Historical scene exclusion is independent of scores and selected IDs.
    for root in (ROOT / 'runs', Path('/home/duyuntao/xmodal_runs')):
        for pat in ('blocks.jsonl', 'probes.jsonl', 'training.jsonl', '*_training.jsonl'):
            for p in sorted(root.rglob(pat)):
                if OUT in p.parents or p.stat().st_size > 8_000_000:
                    continue
                for row in rows(p):
                    o = row.get('observations') or (observed(row) if row.get('messages') else None)
                    if o and len(o) == 4:
                        occupied.add(signature(o))
    forbidden = set(occupied)
    blocks, probes, definition_rows, process_rows = [], [], [], []
    for k in selected:
        facts = fact_rows(k)
        definition_rows += copy.deepcopy(facts)
        replay = [r for r in facts if r['channel'] == 'full_fact'][:8]
        replay += [r for r in facts if r['channel'] == 'atomic_fact' and r['id'].endswith(':0')]
        replay += [r for r in facts if r['channel'] == 'boundary_logic'][:2]
        assert len(replay) == 16
        process_rows += copy.deepcopy(replay)
        for split in ('train', 'holdout'):
            for task in ('direct', 'diagnosis'):
                for j in range(16):
                    b = make_scene(k, split, task, j, occupied, pools)
                    blocks.append(b)
                    p = old.practice(b, k, task)
                    p.update(condition='new_' + split, protocol='short')
                    if split == 'train':
                        process_rows += [old.as_training(p, 'direct_application'), trace_row(p, k, b)]
                        if j in (0, 5):
                            probes.append(reasoning(p))
                    else:
                        probes += [p, reasoning(p)]
        probes.append(dict(id='recall:' + k['id'], qid=k['id'], condition='recall', input_mod='T', fmt='fact',
            protocol='recall', gold=mem.compact(k), messages=mem.messages(mem.RECALL_Q.format(name=name(k)))))
        for key, question, answer in field_specs(k):
            probes.append(dict(id=f'field_probe:{k["id"]}:{key}', qid=k['id'], condition='field_recall', input_mod='T', fmt='fact',
                protocol='field', field=key, gold=answer, messages=[dict(role='system', content='Return only the requested factual value of the learned rule.'),
                    dict(role='user', content=f'For the learned rule {name(k)}, what is its {question}?')]))
    labels = {r['id']: r for r in rows(DATA / 'audit/eval_labels.jsonl')}
    for direction, mod in (('T2T', 'T'), ('T2V', 'V')):
        for p in rows(DATA / direction / 'eval.jsonl'):
            if p['qid'] not in ids:
                continue
            lab = labels[p['id']]
            z = ev.prepare_row(p, DATA, direction)
            z.update(id='original:' + mod + ':' + p['id'], source_id=p['id'], condition='original', input_mod=mod,
                fmt='mcq', protocol='short', gold=lab['answer'], task=lab['question_type'], evaluation_group=lab['evaluation_group'])
            if mod == 'T':
                k = next(k for k in selected if k['id'] == p['qid'])
                obs = observed(p)
                expected = old.assignment(json.loads(k['fact']['quote']), obs)
                assert expected == independent(k, obs)
                opts = dict(re.findall(r'(?m)^([A-D])\. (.*)$', p['messages'][-1]['content']))
                assert expected in opts[lab['answer']]
                z['observations'] = obs
            probes += [z, reasoning(z)]
    for rr in (definition_rows, process_rows):
        random.Random(42).shuffle(rr)
        assert len(rr) == 640
        assert all(not r.get('images') and '<image>' not in str(r['messages']) and 'Learning reference:' not in str(r['messages']) for r in rr)
    train = {b['observation_signature'] for b in blocks if b['split'] == 'train'}
    hold = {b['observation_signature'] for b in blocks if b['split'] == 'holdout'}
    assert len(train) == len(hold) == 256 and not train & hold and not (train | hold) & forbidden
    balances = {}
    for k in selected:
        for split in ('train', 'holdout'):
            for task in ('direct', 'diagnosis'):
                bb = [b for b in blocks if b['qid'] == k['id'] and b['split'] == split and b['task'] == task]
                correct = Counter('ABCD'[b['options'].index(b['results'][k['id']])] for b in bb)
                assert correct == dict.fromkeys('ABCD', 4)
                assert sum(b['submitted_report'] == b['results'][k['id']] for b in bb) == 4
                balances[f'{k["id"]}:{split}:{task}'] = dict(correct)
    assert len(probes) == 664 and len({p['id'] for p in probes}) == 664
    for fn, rr in [('knowledge.jsonl', selected), ('blocks.jsonl', blocks), ('probes.jsonl', probes),
                   ('definitions_training.jsonl', definition_rows), ('definition_replay_training.jsonl', definition_rows),
                   ('process_training.jsonl', process_rows)]:
        write(OUT / fn, rr)
    save(OUT / 'selection_audit.json', dict(history_inventory=history, excluded_rule_ids=sorted(used), eligibility_order=eligibility,
        selected_ids=sorted(ids), note='Excludes discovered small-pilot datasets with <=64 rule IDs; old full1000 runs include these rules. Clean-base initialization. Not a claim of never previously evaluated.'))
    os.environ.update(environment('0', DATA))
    from swift.model import get_model_processor
    from swift.template import get_template
    _, processor = get_model_processor(str(MODELS['qwen3']), model_type='qwen3_vl', load_model=False)
    template = get_template(processor, template_type='qwen3_vl', max_length=8192, enable_thinking=False)
    template.set_mode('train')
    suffix = processor.tokenizer.encode('<|im_end|>\n', add_special_tokens=False)
    enc = []
    for arm, rr in [('definitions', definition_rows), ('definition_replay', definition_rows), ('process', process_rows)]:
        for row in rr:
            z = template.encode(dict(messages=copy.deepcopy(row['messages'])))
            lab = [v for v in z['labels'] if v != -100]
            assert lab == processor.tokenizer.encode(row['messages'][-1]['content'], add_special_tokens=False) + suffix
            assert len(z['input_ids']) < 8192
            if row['channel'] == 'worked_application':
                assert len(lab) - len(suffix) <= 768
            enc.append(dict(arm=arm, id=row['id'], channel=row['channel'], input_tokens=len(z['input_ids']), supervised_tokens=len(lab)))
    write(OUT / 'encoding_audit.jsonl', enc)
    sources = [Path(__file__), PLUGIN, Path(mem.__file__), Path(old.__file__), ROOT / 'CROSS_MODAL_RESEARCH_PROTOCOL.md',
        ROOT / 'scripts/run_v7_evaluation.py', ROOT / 'scripts/run_v7_full_experiment.py', ROOT / 'scripts/run_v7_fullparam_ep10.py',
        ROOT / 'scripts/build_v7_experiment_data.py', ROOT / 'scripts/v7_gpu_resources.py', DATA / 'audit/rule_knowledge.jsonl',
        DATA / 'audit/eval_labels.jsonl', DATA / 'image_hashes.jsonl'] + [DATA / d / 'eval.jsonl' for d in ('T2T', 'T2V', 'V2T', 'V2V')]
    base = dense_identity(MODELS['qwen3'])
    for r in rows(DATA / 'image_hashes.jsonl'):
        assert sha(DATA / r['path']) == r['source_sha256'] == r['repaired_sha256']
    manifest = dict(created_at=ev.now(), sources={str(p): sha(p) for p in sources},
        files={p.name: sha(p) for p in OUT.glob('*.jsonl')} | {'selection_audit.json': sha(OUT / 'selection_audit.json')},
        base=base, ids=sorted(ids), arms=ARMS, suffix_ids=suffix,
        selection='Four category/material OR and four numeric interval rules; SHA256 order after excluding discovered small-pilot rules, independent of scores. Includes general endpoint comparators; not identical to prior shape/color setting.',
        training=dict(modality='TEXT', rows_per_task=640, epochs_per_task=3, steps_per_task=480, batch=2, accumulation=2,
            common_definitions_lr=3e-5, sibling_lr=1e-5, lora_rank=32, lora_alpha=64, dropout=.05, seed=42, max_length=8192,
            freeze_vision_aligner=True, common_initialization='clean base; no historical adapters',
            siblings='definition_replay and process both start from this run definitions final step480, fresh optimizer',
            channels={a: dict(Counter(r['channel'] for r in rr)) for a, rr in [('definitions', definition_rows), ('process', process_rows)]},
            process_scenes=256, process_scene_repetition='each scene once direct, once trace; not 512 independent scenes',
            loss='equal-row CE including template suffix; only assistant supervised',
            token_totals={a: sum(r['supervised_tokens'] for r in enc if r['arm'] == a) for a in TRAIN_ARMS},
            caveat='Equal sibling rows/steps/LR, different training content and supervised tokens; no token-matched causal claim'),
        generation=dict(direct=32, reasoning=768, recall=256, field=48, temperature=0, thinking=False, batch=4),
        counts=dict(Counter(p['condition'] + ':' + p['protocol'] + ':' + p['input_mod'] for p in probes)),
        primary='Closed-book new_holdout reasoning TEXT and original reasoning T2T, against definitions and definition_replay; short answers separately. Same checkpoint directly answers original T2T/T2V.',
        gate='Exploratory feasibility: new_holdout reasoning >=70% with >=15pp gain vs definition_replay; original reasoning T2T >=60% and >=15pp gain; format errors <=5%. Report paired rule-cluster intervals and all failures. No gate based on T2T>T2V.',
        checkpoint='Always final epoch3 step480 per task, never selected by eval',
        limitations=['Eight rules/two structures/single training seed; not full1000 or VISION learning.',
            'Rule selection excludes identified small-pilot IDs, not historical full-dataset exposure; clean-base training prevents weight inheritance.',
            'Category/material OR changes the attribute domain from old shape/color pilots; transfer reading difficulty may differ.',
            'Source-modality new application exercises added, so this is not material-only learning.',
            'Whole unordered four-candidate scenes disjoint; individual candidates may repeat.',
            'Explicit generated reasoning, native thinking disabled, 768-token cap for every arm; truncation reported separately.',
            'Only 16 original questions per direction; formal results descriptive, not population-level confirmation.'])
    save(OUT / 'manifest.json', manifest)
    save(OUT / 'data_audit.json', dict(status='pass', selected_rules=8, formal_scenes_parsed=formal_parsed,
        historical_excluded_scenes=len(forbidden), train_scenes=256, holdout_scenes=256, scenes_disjoint=True,
        independent_label_checks=True, balanced_letters_and_report_acceptance=balances, images_verified=3000,
        original_questions_options_labels_and_images_unchanged=True, assistant_mask_and_EOS_audit=True))
    snap = OUT / 'source_snapshots'
    snap.mkdir(exist_ok=True)
    for p in sources:
        if p.suffix == '.py':
            shutil.copy2(p, snap / p.name)
    notes = ['# 独立规则确认：训练数据', '', 'TEXT学习；8条固定规则；不使用正式原题、选项、答案或原图训练。',
        '先定义/原子事实SFT：640行，3epoch。随后两组从该最终权重分支，各640行、3epoch；定义复习与具体过程学习对照。',
        '过程组每规则16行事实复习、32个新场景直接答案、32个同场景具体过程。256个独立训练场景与256个留出场景按无序四候选集合隔离。',
        '两组更新步数相同，训练内容和监督token不同。所有评测配置与规则在训练前冻结；所有规则保留。', '']
    for ch in ('full_fact', 'atomic_fact', 'boundary_logic', 'direct_application', 'worked_application'):
        row = next(r for r in process_rows if r['channel'] == ch)
        notes += ['## ' + ch, '', '```json', json.dumps(row, ensure_ascii=False, indent=2), '```', '']
    (OUT / 'TRAINING_DATA_ZH.md').write_text('\n'.join(notes) + '\n')
    validate()
    print('PREPARED', json.dumps(dict(ids=sorted(ids), counts=manifest['counts'], training=manifest['training'])), flush=True)


def checkpoint(arm, smoke=False):
    return OUT / ('smoke_' + arm if smoke else arm) / 'checkpoint' / ('checkpoint-2' if smoke else 'checkpoint-480')


def train(arm, gpu, smoke=False):
    m = validate()
    cp = checkpoint(arm, smoke)
    folder = cp.parent.parent
    folder.mkdir(parents=True, exist_ok=True)
    dataset = folder / 'smoke_input.jsonl' if smoke else OUT / (arm + '_training.jsonl')
    with ev.run_lock(folder / 'train.lock'):
        if not (cp / 'adapter_model.safetensors').exists():
            if smoke:
                rr = rows(OUT / (arm + '_training.jsonl'))
                channels = sorted({r['channel'] for r in rr})
                sample = [next(r for r in rr if r['channel'] == ch) for ch in channels]
                write(dataset, sample * 2)
            recover = [p for p in cp.parent.glob('checkpoint-*') if all((p / n).exists() for n in
                ('optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'trainer_state.json', 'adapter_model.safetensors'))]
            resume = max(recover, key=lambda p: read(p / 'trainer_state.json')['global_step']) if recover and not smoke else None
            cmd = train_command('qwen3', 'sft', dataset, cp.parent, epochs=3, resume=resume)
            for flag, val in [('--per_device_train_batch_size', len(sample) if smoke else 2),
                ('--gradient_accumulation_steps', 1 if smoke else 2), ('--learning_rate', 3e-5 if arm == 'definitions' else 1e-5),
                ('--lora_rank', 32), ('--lora_alpha', 64), ('--save_strategy', 'steps'), ('--logging_steps', 1)]:
                cmd[cmd.index(flag) + 1] = str(val)
            if arm != 'definitions' and not resume:
                initial = checkpoint('definitions')
                assert (initial / 'adapter_model.safetensors').exists()
                cmd += ['--adapters', str(initial), '--load_args', 'false']
            cmd += ['--save_steps', '2' if smoke else '160', '--attn_impl', 'sdpa', '--external_plugins', str(PLUGIN),
                '--loss_type', 'v7_suffix_control', '--packing', 'false', '--padding_free', 'false', '--use_logits_to_keep', 'false',
                '--dataloader_num_workers', '0', '--dataloader_persistent_workers', 'false', '--report_to', 'none']
            if smoke:
                cmd += ['--max_steps', '2']
                cmd[cmd.index('--warmup_ratio') + 1] = '0'
            env = environment(gpu, DATA)
            env.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', V7_CONTENT_SUFFIX_IDS=json.dumps(m['suffix_ids']),
                V7_CONTENT_LOSS_AUDIT=str(folder / 'loss_audit.jsonl'))
            save(folder / 'state.json', dict(status='waiting_for_gpu', command=cmd, gpu=gpu))
            with reserve(LEDGER, gpu, str(folder) + ':train', 26000 if smoke else 22000) as res:
                with (folder / 'train.log').open('a') as log:
                    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                    res.bind(proc.pid)
                    save(folder / 'state.json', dict(status='running', pid=proc.pid, command=cmd, gpu=gpu))
                    code = proc.wait()
                if code:
                    save(folder / 'state.json', dict(status='failed', returncode=code))
                    raise RuntimeError((arm, code, str(folder / 'train.log')))
        state, args = read(cp / 'trainer_state.json'), read(cp / 'args.json')
        assert state['global_step'] == (2 if smoke else 480)
        assert smoke or abs(state['epoch'] - 3) < 1e-6
        assert args['freeze_vit'] and args['freeze_aligner'] and args['lora_rank'] == 32
        if arm == 'definitions':
            assert not args.get('adapters')
        else:
            assert args['adapters'] == [str(checkpoint('definitions'))]
        losses = [r['loss'] for r in rows(cp.parent / 'logging.jsonl') if 'loss' in r]
        assert losses and all(math.isfinite(v) for v in losses)
        from safetensors import safe_open
        with safe_open(str(cp / 'adapter_model.safetensors'), framework='pt', device='cpu') as f:
            bs = [k for k in f.keys() if 'lora_B' in k]
            assert bs and all(bool(f.get_tensor(k).count_nonzero()) for k in bs)
            assert not any('.visual.' in k or '.merger.' in k for k in f.keys())
        save(folder / 'checkpoint_manifest.json', dict(path=str(cp), model=ev.checkpoint(cp), epoch=state['epoch'],
            step=state['global_step'], first_loss=losses[0], last_loss=losses[-1], losses_finite=True,
            nonzero_lora_B=len(bs), initial='clean_base' if arm == 'definitions' else ev.checkpoint(checkpoint('definitions'))))
        save(folder / 'state.json', dict(status='complete', checkpoint=str(cp)))
    if smoke:
        mem.OUT = OUT
        pp = []
        for mod in ('T', 'V'):
            p = copy.deepcopy(next(p for p in rows(OUT / 'probes.jsonl') if p['condition'] == 'original' and p['input_mod'] == mod and p['protocol'] == 'short'))
            p['id'] = 'smoke:' + p['id']
            pp.append(p)
        runner = mem.Runner(folder / 'reload', gpu, 'smoke_' + arm)
        if runner.seen != {p['id'] for p in pp}:
            with reserve(LEDGER, gpu, str(folder) + ':reload', 22000) as res:
                res.bind(os.getpid())
                runner.load(cp=cp, adapter=True)
                runner.infer(pp, 'direct')
        assert len(runner.rr) == 2 and all(r['response'].strip() for r in runner.rr)
        save(folder / 'validated.json', dict(status='pass', text_image_checkpoint_reload=True, channels=sorted({r['channel'] for r in rows(dataset)})))


def evaluate(arm, gpu):
    validate()
    mem.OUT = OUT
    folder = OUT / arm / 'evaluation'
    folder.mkdir(parents=True, exist_ok=True)
    probes = rows(OUT / 'probes.jsonl')
    with ev.run_lock(folder / 'eval.lock'):
        runner = mem.Runner(folder, gpu, arm)
        if runner.seen != {p['id'] for p in probes}:
            with reserve(LEDGER, gpu, str(folder) + ':evaluation', 22000) as res:
                res.bind(os.getpid())
                runner.load(MODELS['qwen3'] if arm == 'base' else checkpoint(arm), adapter=arm != 'base')
                # Original outputs first, then independent scenes; interpretation waits for all arms.
                for protocol, stage in [('short', 'direct'), ('reasoning', 'reasoning'), ('recall', 'recall'), ('field', 'field')]:
                    runner.infer([p for p in probes if p['protocol'] == protocol], stage)
        assert len(runner.rr) == len(probes)
        save(folder / 'state.json', dict(status='complete', n=len(probes), ended_at=ev.now()))


def child(stage, arm, gpu, smoke=False):
    cmd = [str(PY), '-u', str(Path(__file__)), '--stage', stage, '--arm', arm, '--gpu', str(gpu)]
    if smoke:
        cmd += ['--smoke']
    subprocess.run(cmd, cwd=ROOT, check=True)


def workflow():
    validate()
    save(OUT / 'workflow_state.json', dict(status='running', pid=os.getpid(), started_at=ev.now()))
    try:
        def common():
            child('train', 'definitions', '0', True)
            child('train', 'definitions', '0')
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(common), pool.submit(child, 'evaluate', 'base', '1')]
            for f in futures:
                f.result()
        def sibling(arm, gpu):
            child('train', arm, gpu, True)
            child('train', arm, gpu)
            child('evaluate', arm, gpu)
            if arm == 'definition_replay':
                child('evaluate', 'definitions', gpu)
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(sibling, 'definition_replay', '0'), pool.submit(sibling, 'process', '1')]
            for f in futures:
                f.result()
        child('score', 'base', '0')
        save(OUT / 'workflow_state.json', dict(status='complete', ended_at=ev.now()))
    except Exception as e:
        save(OUT / 'workflow_state.json', dict(status='failed', error=repr(e), ended_at=ev.now()))
        raise


def metric(rr):
    n = len(rr)
    good = sum(r['scored_correct'] for r in rr)
    valid = sum(not r['scored_format_error'] for r in rr)
    return dict(n=n, correct=good, accuracy=good / n if n else None, format_errors=n-valid,
        valid_accuracy=good / valid if valid else None, truncated=sum(r['finish_reason'] == 'length' for r in rr))


def paired(aa, bb):
    a, b = {r['id']: r for r in aa}, {r['id']: r for r in bb}
    assert a.keys() == b.keys()
    gained = sum(not a[i]['scored_correct'] and b[i]['scored_correct'] for i in a)
    lost = sum(a[i]['scored_correct'] and not b[i]['scored_correct'] for i in a)
    d = gained + lost
    p = min(1., 2 * sum(math.comb(d, j) for j in range(min(gained, lost) + 1)) / 2 ** d) if d else 1.
    groups = defaultdict(list)
    for i in a:
        groups[a[i]['qid']].append(int(b[i]['scored_correct']) - int(a[i]['scored_correct']))
    rng = random.Random(42)
    draws = sorted(sum(v for q in rng.choices(sorted(groups), k=len(groups)) for v in groups[q]) / len(a) for _ in range(5000))
    return dict(n=len(a), gained=gained, lost=lost, delta=(gained-lost)/len(a),
        descriptive_mcnemar_p=p, rule_cluster_bootstrap_ci95=[draws[125], draws[4874]],
        caveat='Only eight clusters/one seed; descriptive, no population-level or multiplicity-adjusted inference.')


def score():
    m = validate(images=True)
    probes = {p['id']: p for p in rows(OUT / 'probes.jsonl')}
    knowledge = {k['id']: k for k in rows(OUT / 'knowledge.jsonl')}
    blocks = {b['id']: b for b in rows(OUT / 'blocks.jsonl')}
    all_scored, summary, traces = [], {}, []
    for arm in ARMS:
        rr = rows(OUT / arm / 'evaluation/predictions.jsonl')
        assert len(rr) == len(probes) == len({r['id'] for r in rr})
        for r in rr:
            p = probes[r['id']]
            assert r['input_sha256'] == ev.digest(dict(messages=p['messages'], images=p.get('images', [])))
            for key in ('qid', 'gold', 'condition', 'input_mod', 'protocol'):
                assert r[key] == p[key]
            text = r['response'].strip()
            if p['protocol'] == 'short':
                pred = ev.letter(text)
            elif p['protocol'] == 'reasoning':
                matches = re.findall(r'(?im)^\s*Answer:\s*([A-D])\s*[.!]?\s*$', text)
                pred = matches[0] if len(matches) == 1 else None
            else:
                pred = None
            norm = lambda t: re.sub(r'\s+', ' ', t).strip().lower()
            good = norm(text) == norm(p['gold']) if p['fmt'] == 'fact' else pred == p['gold']
            z = {**r, 'arm': arm, 'scored_answer': pred, 'scored_correct': good,
                 'scored_format_error': p['fmt'] != 'fact' and pred is None}
            all_scored.append(z)
            if p['protocol'] == 'reasoning' and p['input_mod'] == 'T':
                obs = p.get('observations') or (blocks[p['block']]['observations'] if p.get('block') else None)
                if obs:
                    k = knowledge[p['qid']]
                    expected = independent(k, obs)
                    result = re.findall(r'(?im)^\s*Result:\s*((?:[A-D]\s*=\s*(?:KEEP|OMIT)\s*;?\s*){4})[.]?\s*$', text)
                    decisions = None
                    if len(result) == 1:
                        pairs = re.findall(r'([A-D])\s*=\s*(KEEP|OMIT)', result[0])
                        if [c for c, _ in pairs] == list('ABCD'):
                            decisions = '; '.join(c+'='+v for c, v in pairs)
                    rule_line = re.findall(r'(?im)^Rule:\s*(.+)$', text)
                    checks = []
                    rule = json.loads(k['fact']['quote'])
                    for c, o in zip('ABCD', obs):
                        line = re.findall(r'(?im)^' + c + r':\s*(.+)$', text)
                        canonical = []
                        for cond, flag in zip(rule['conditions'], old.flags(rule, o)):
                            op = dict(eq='==', ge='>=', gt='>', lt='<', le='<=')[cond['operator']]
                            canonical.append(f'{cond["field"]} {o[cond["field"]]} {op} {cond["value"]} = {"YES" if flag else "NO"}')
                        checks.append(bool(line) and all(norm(f) in norm(line[0]) for f in canonical))
                    traces.append(dict(arm=arm, id=p['id'], qid=p['qid'], condition=p['condition'], answer_correct=good,
                        result_parseable=decisions is not None, result_correct=decisions == expected,
                        exact_definition_line=bool(rule_line) and norm(rule_line[0]) == norm(mem.compact(k)),
                        exact_operand_lines=sum(checks), exact_operand_lines_all=all(checks),
                        note='Exact syntax diagnostics are conservative coverage measures, not semantic grading of free reasoning.'))
        sub = [r for r in all_scored if r['arm'] == arm]
        summary[arm] = {}
        for key in sorted(m['counts']):
            ss = [r for r in sub if ':'.join(r[c] for c in ('condition', 'protocol', 'input_mod')) == key]
            summary[arm][key] = metric(ss) | dict(by_task={t: metric([r for r in ss if r.get('task') == t]) for t in ('direct', 'diagnosis')})
    write(OUT / 'scored_predictions.jsonl', all_scored)
    write(OUT / 'trace_diagnostics.jsonl', traces)
    save(OUT / 'summary.json', summary)
    pair_stats = []
    for before, after in [('base', 'definitions'), ('definitions', 'definition_replay'), ('definitions', 'process'), ('definition_replay', 'process')]:
        for key in sorted(m['counts']):
            sub = [r for r in all_scored if ':'.join(r[c] for c in ('condition', 'protocol', 'input_mod')) == key]
            pair_stats.append(dict(before=before, after=after, test=key, **paired([r for r in sub if r['arm']==before], [r for r in sub if r['arm']==after])))
    save(OUT / 'paired_comparisons.json', pair_stats)
    per_rule = {a: {q: {key: metric([r for r in all_scored if r['arm']==a and r['qid']==q and ':'.join(r[c] for c in ('condition','protocol','input_mod'))==key])
        for key in sorted(m['counts'])} for q in m['ids']} for a in ARMS}
    save(OUT / 'per_rule_metrics.json', per_rule)
    import csv
    with (OUT / 'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['arm','test','n','correct','accuracy','format_errors','valid_accuracy','truncated'])
        writer.writeheader()
        for a in ARMS:
            for key, met in summary[a].items():
                writer.writerow(dict(arm=a,test=key,**{k:v for k,v in met.items() if k!='by_task'}))
    gate = (summary['process']['new_holdout:reasoning:T']['accuracy'] >= .7 and
        summary['process']['new_holdout:reasoning:T']['accuracy'] - summary['definition_replay']['new_holdout:reasoning:T']['accuracy'] >= .15 and
        summary['process']['original:reasoning:T']['accuracy'] >= .6 and
        summary['process']['original:reasoning:T']['accuracy'] - summary['definition_replay']['original:reasoning:T']['accuracy'] >= .15 and
        all(summary['process'][k]['format_errors']/summary['process'][k]['n'] <= .05 for k in ('new_holdout:reasoning:T','original:reasoning:T')))
    lines = ['# 新规则具体过程学习确认结果', '', 'TEXT学习，同一检查点闭卷直接回答T2T/T2V，原图不变。先定义SFT，再从共同起点做等步数定义复习/具体过程训练。', '',
        '| 测试 | 基座 | 定义SFT | 定义续训 | 具体过程SFT |', '|---|---:|---:|---:|---:|']
    for key in sorted(m['counts']):
        lines.append('| '+key+' | '+' | '.join(f'{summary[a][key]["correct"]}/{summary[a][key]["n"]} ({summary[a][key]["accuracy"]:.1%})' for a in ARMS)+' |')
    lines += ['', '预先固定的来源应用可行性门槛：'+('通过' if gate else '未通过')+'。这不是全量数据集或总体统计确认。', '',
        '全部格式错误、截断、direct/diagnosis分项见summary.json，按规则配对统计见paired_comparisons.json。',
        '显式推理为普通解释生成，native thinking关闭；短答案与推理结果不混合。精确轨迹诊断只反映可解析覆盖，不把自由表述直接当作错误。',
        '仅8条规则、一个seed；类别/材质与区间结构，改变了历史形状/颜色属性域；相同更新步数但监督token不同。',
        '不按T2T>T2V挑规则或检查点。图片落差可同时含读取错误与规则调用问题，不能单凭总准确率归为纯迁移障碍。']
    (OUT / 'report_zh.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(13,3.5))
    for ax, a in zip(axes, TRAIN_ARMS):
        logs = [r for r in rows(checkpoint(a).parent/'logging.jsonl') if 'loss' in r]
        ax.plot([float(r.get('global_step',i+1).split('/')[0]) if isinstance(r.get('global_step'),str) else r.get('global_step',i+1) for i,r in enumerate(logs)], [r['loss'] for r in logs],alpha=.7)
        ax.set(title=a,xlabel='Update',ylabel='Equal-row CE');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'training_loss.png',dpi=160);plt.close(fig)
    save(OUT / 'completion_audit.json', dict(status='pass', predictions=len(all_scored), images_verified=3000,
        final_checkpoints_only=True, source_application_gate_passed=gate, ended_at=ev.now()))
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['prepare','train','evaluate','workflow','score'], required=True)
    p.add_argument('--arm', choices=ARMS, default='definitions')
    p.add_argument('--gpu', default='0')
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    if a.stage == 'prepare':
        prepare()
    elif a.stage == 'train':
        assert a.arm in TRAIN_ARMS
        train(a.arm,a.gpu,a.smoke)
    elif a.stage == 'evaluate':
        evaluate(a.arm,a.gpu)
    elif a.stage == 'workflow':
        workflow()
    else:
        score()
