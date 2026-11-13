#!/usr/bin/env python3
"""Frozen, stratified learning audit of the completed dense epoch-10 models.

Diagnostic conditions are never reported as closed-book transfer. Originals are read-only.
"""
import argparse
import copy
import dataclasses
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import run_v7_evaluation as ev
from build_v7_experiment_data import name
from run_v7_full_experiment import ROOT, MODELS, environment, read, rows, save, sha
from run_v7_fullparam_ep10 import dense_identity
from v7_gpu_resources import reserve

OUT = ROOT / 'runs/v7_ep10_learning_audit_20261019'
SOURCE = ROOT / 'runs/v7_fullparam_sft_ep10_20261019'
DATA = ROOT / 'rules_repaired_single_image_v7'
LEDGER = ROOT / 'runs/v7_full_experiment'
ARMS = ('text', 'vision', 'base')


def natural_facts(k):
    r = json.loads(k['fact']['quote']); rel = k['fact']['relation']
    label = name(k); result = []
    def add(question, answer):
        result.append({'question': f'For the stored rule {label}, {question} Return only the requested short answer.', 'answer': answer})
    if rel in ('binary_predicate', 'set_selection_rule'):
        for c in r['conditions'][:2]:
            field = c['field']; op = c['operator']; value = c['value']
            add(f'what {field} is {"excluded" if op == "ne" else "required by its " + field + " condition"}?', value)
        if rel == 'binary_predicate':
            protocol = k['material']['learning'].get('inspection_recording_protocol')
            # The v7 recording table is supplemental material, absent from fact.quote.
            text = k['material']['learning']['text']
            code = re.search(r'Record the pair of outcomes.*?→\s*([A-Z]\d)', text)
            assert code, (k['id'], protocol)
            add('which inspection record code means that both named requirements are met?', code[1])
        else:
            add('must all conditions hold, or is any one condition sufficient? Answer all or any.', r['mode'])
    elif rel == 'bipartite_matching':
        pairs = re.findall(r'([^.]+?) accepts ([^.]+)\.', k['material']['learning']['text'])
        assert len(pairs) >= 3
        for item, slots in pairs[:3]:
            add(f'which slot shapes are compatible with {item.strip()}? List the shapes.', slots.replace(' slots', ''))
    elif rel == 'finite_state_transition':
        pairs = [(s, a, dest) for s, aa in r['transition'].items() for a, dest in aa.items()]
        for s, a, dest in pairs[::2][:3]:
            add(f'what is the next state from {s} after a {a.replace("_signal", "")} signal?', dest)
    elif rel == 'mapping_codebook':
        for item, code in list(r['mapping'].items())[:3]:
            add(f'what codeword is assigned to {item}?', code)
    elif rel == 'ordering_priority':
        for pos in (0, 1, -1):
            add(f'what category is {"highest" if pos == 0 else "second-highest" if pos == 1 else "lowest"} in priority?', r['priority'][pos])
    elif rel == 'scalar_index_rule':
        for field, weight in list(r['weights'].items())[:2]:
            add(f'what is the numeric weight per {field.removesuffix("_count").replace("_", " ")}?', weight)
        add('what is the signed numeric base adjustment applied once per collection?', r['bias'])
    elif rel == 'three_way_category':
        for edge in ('lower', 'upper'):
            add(f'what is the {edge} proportion threshold as a decimal?', r['thresholds'][edge])
        add('which category includes exactly either threshold?', r['category_labels'][1])
    elif rel == 'transformation_procedure':
        op = next(x for x in r['operations'] if x['op'] == 'rotate_left')
        add('how many positions does each individual left rotation shift?', op['positions'])
        add('how many times is the left rotation performed in total?', op['repeat'])
        add('what is the first operation? Answer reverse, rotate_left, or swap_disjoint_pairs.', r['operations'][0]['op'])
    else:
        raise ValueError(rel)
    assert len(result) == 3, (k['id'], result)
    return result


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT / 'manifest.json').exists():
        validate(); return
    knowledge = {r['content']['id']: r['content'] for r in rows(DATA / 'audit/rule_knowledge.jsonl')}
    groups = defaultdict(list)
    for k in knowledge.values(): groups[k['fact']['relation']].append(k)
    selected = []
    for rel, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda k: hashlib.sha256(('fullparam-repair-diagnostic:42:' + k['id']).encode()).hexdigest())
        first = ordered[0]; family = json.loads(first['fact']['quote'])['structural_subfamily_id']
        second = next(k for k in ordered[1:] if json.loads(k['fact']['quote'])['structural_subfamily_id'] != family)
        selected.extend((first, second))
    assert len(selected) == 18
    ev.save_rows(OUT / 'selected_knowledge.jsonl', selected)
    # Only learning material is read to construct fact questions. No test labels here.
    facts = []
    for k in selected:
        for i, fact in enumerate(natural_facts(k)):
            facts.append(dict(id=f'fact:{k["id"]}:{i}', qid=k['id'], relation=k['fact']['relation'], **fact))
    ev.save_rows(OUT / 'facts.jsonl', facts)
    save(OUT / 'selection.json', {'selection': 'SHA256 seed42, two rules per relation, different structural subfamilies; no score filtering',
        'ids': [k['id'] for k in selected], 'fact_questions': len(facts), 'selected_before_loading_test_labels': True})
    selected_ids = {k['id'] for k in selected}
    train = {mod: {r['id']: r for r in rows(SOURCE / 'data' / f'sft_{mod}.jsonl')} for mod in ('text', 'vision')}
    labels = {r['id']: r for r in rows(DATA / 'audit/eval_labels.jsonl')}
    probes = []
    for input_mod, direction in (('T', 'T2T'), ('V', 'T2V')):
        for r in rows(SOURCE / 'data' / f'eval_{direction}.jsonl'):
            if r['qid'] not in selected_ids: continue
            gold = labels[r['id']]
            p = copy.deepcopy(r)
            p.update(id=f'{input_mod}:{r["id"]}', source_id=r['id'], input_mod=input_mod, condition='closed', fmt='mcq',
                task=gold['question_type'], evaluation_group=gold['evaluation_group'], gold=gold['answer'],
                relation=knowledge[r['qid']]['fact']['relation'])
            probes.append(p)
            z = copy.deepcopy(p); z.update(id='oracle:' + p['id'], condition='oracle')
            z['messages'][-1]['content'] = 'Correct learning material for this question:\n' + train['text'][r['qid']]['messages'][-1]['content'] + '\n\n' + z['messages'][-1]['content']
            probes.append(z)
    assert len(probes) == 144
    ev.save_rows(OUT / 'application_probes.jsonl', probes)
    for arm in ARMS:
        mod = 'text' if arm == 'base' else arm
        rr = []
        for k in selected:
            original = train[mod][k['id']]
            common = dict(qid=k['id'], relation=k['fact']['relation'], fmt='recall', gold=original['messages'][-1]['content'])
            rr.append(dict(id='exact:' + k['id'], condition='exact_training_prompt', messages=copy.deepcopy(original['messages'][:-1]),
                **({'images': original['images']} if 'images' in original else {}), **common))
            rr.append(dict(id='neutral:' + k['id'], condition='neutral_recall',
                messages=[{'role': 'user', 'content': f'Recall the complete stored rule named {name(k)}. State all exact conditions, values, mappings, and recording conventions. State the rule itself rather than solving a new case.'}], **common))
        for f in facts:
            rr.append(dict(id=f['id'], qid=f['qid'], relation=f['relation'], condition='atomic_fact', fmt='fact', gold=f['answer'],
                messages=[{'role': 'user', 'content': f['question']}]))
        ev.save_rows(OUT / f'{arm}_recall_probes.jsonl', rr)
    images = rows(DATA / 'image_hashes.jsonl')
    assert len(images) == 3000
    for r in images: assert sha(DATA / r['path']) == r['source_sha256'] == r['repaired_sha256']
    models = {'base': read(SOURCE / 'manifest.json')['model']}
    for arm in ('text', 'vision'): models[arm] = read(SOURCE / 'jobs' / arm / 'checkpoint_manifest.json')['model']
    sources = [DATA / 'audit/rule_knowledge.jsonl', DATA / 'audit/eval_labels.jsonl', DATA / 'image_hashes.jsonl',
        SOURCE / 'manifest.json', *[SOURCE / 'data' / f'sft_{m}.jsonl' for m in ('text', 'vision')]]
    for arm in ARMS:
        for direction in ('T2T', 'T2V') if arm != 'vision' else ('V2T', 'V2V'):
            sources.append(SOURCE / 'jobs' / ('baseline' if arm == 'base' else arm) / 'evaluation' / direction / 'predictions.jsonl')
    save(OUT / 'manifest.json', {'created_at': ev.now(), 'script_sha256': sha(Path(__file__)), 'models': models,
        'sources': {str(p): sha(p) for p in sources}, 'files': {p.name: sha(p) for p in OUT.glob('*.jsonl')},
        'rules': 18, 'relations': 9, 'arms': ARMS, 'seed': 42, 'batch': 4,
        'generation': {'mcq': 32, 'recall': 1024, 'fact': 128, 'temperature': 0, 'thinking': False},
        'conditions': {'closed': 'Reuse actual epoch10 or base predictions on identical inputs.',
            'oracle': 'Original test prompt plus correct TEXT learning material; diagnostic, not transfer.',
            'self_recall': 'Original test prompt plus own uncorrected neutral recall; no external correct material.',
            'exact_training_prompt': 'Verbatim original training input; VISION retains original learning image.',
            'neutral_recall': 'Rule identifiers only, no learning image.', 'atomic_fact': 'Three material-derived short questions per rule; short-answer diagnostic only.'},
        'limits': ['18 deterministic development rules, one seed; not full-dataset accuracy or independent confirmation.',
            'Exact training replay is a capacity check. Verbatim mismatch is not automatically a semantic error.',
            'Atomic automatic scoring is conservative; raw answers retained for semantic review.',
            'Gold learning facts include repaired supplemental conventions absent from fact.quote.',
            'No new training yet; no modifications to any original data or PNG.'], 'all3000images_unchanged': True})
    validate()


def validate():
    m = read(OUT / 'manifest.json'); assert m['script_sha256'] == sha(Path(__file__))
    for p, h in m['sources'].items(): assert sha(Path(p)) == h, p
    for p, h in m['files'].items(): assert sha(OUT / p) == h, p


def normalized(x):
    return re.sub(r'[^a-z0-9.+-]', '', str(x).lower().replace('−', '-'))


def fact_score(text, gold):
    clean = text.strip().strip('`"\' .\n')
    if isinstance(gold, (int, float)):
        # One numeric answer only; explanatory responses require manual review.
        if not re.fullmatch(r'[+-]?\d+(?:\.\d+)?', clean): return False
        return float(clean) == gold
    return normalized(clean) == normalized(gold)


def run(arm, gpu):
    validate(); m = read(OUT / 'manifest.json'); folder = OUT / arm; folder.mkdir(exist_ok=True)
    raw = folder / 'predictions.jsonl'
    rr, notes = ev.read_rows(raw, allow_partial_tail=True) if raw.exists() else ([], [])
    if notes: ev.save_rows(raw, rr)
    seen = {r['id'] for r in rr}; assert len(seen) == len(rr)
    def append(r):
        assert r['id'] not in seen
        with raw.open('a') as f:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + '\n'); f.flush(); os.fsync(f.fileno())
        seen.add(r['id']); rr.append(r)
    applications = rows(OUT / 'application_probes.jsonl')
    for input_mod in ('T', 'V'):
        direction = ('V2' if arm == 'vision' else 'T2') + input_mod
        source_path = SOURCE / 'jobs' / ('baseline' if arm == 'base' else arm) / 'evaluation' / direction / 'predictions.jsonl'
        lookup = {r['id']: r for r in rows(source_path)}
        for p in applications:
            if p['condition'] != 'closed' or p['input_mod'] != input_mod or p['id'] in seen: continue
            old = lookup[p['source_id']]
            assert old['input_sha256'] == ev.digest({'messages': p['messages'], 'images': p.get('images', [])})
            append(dict(id=p['id'], qid=p['qid'], condition=p['condition'], fmt=p['fmt'], relation=p['relation'],
                input_mod=input_mod, task=p['task'], gold=p['gold'], response=old['response'], predicted=old['predicted'],
                correct=old['predicted'] == p['gold'], format_error=old['predicted'] is None,
                input_sha256=old['input_sha256'], reused_from=str(source_path), finish_reason=old['finish_reason']))
    total = 90 + 72 * 3
    if len(rr) < total:
        save(folder / 'state.json', {'status': 'waiting_for_gpu', 'gpu': gpu, 'saved': len(rr), 'pid': os.getpid()})
        with reserve(LEDGER, gpu, str(folder) + ':audit', 24000) as res:
            res.bind(os.getpid()); os.environ.update(environment(gpu, DATA))
            os.environ.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
            import torch
            from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
            from swift.utils import seed_everything
            torch.set_num_threads(4); seed_everything(42)
            assert dense_identity(Path(m['models'][arm]['path'])) == m['models'][arm]
            engine = TransformersEngine(m['models'][arm]['path'], model_type='qwen3_vl', torch_dtype=torch.bfloat16,
                device_map='cuda:0', max_batch_size=4, attn_impl='sdpa')
            engine.template.enable_thinking = False
            def generate(probes):
                pending = [p for p in probes if p['id'] not in seen]
                for start in range(0, len(pending), 4):
                    batch = pending[start:start+4]; fmt = batch[0]['fmt']; assert all(p['fmt'] == fmt for p in batch)
                    outputs = engine.infer([InferRequest(messages=copy.deepcopy(p['messages']), images=p.get('images', [])) for p in batch],
                        RequestConfig(max_tokens=m['generation'][fmt], temperature=0, seed=42, return_details=True), use_tqdm=False)
                    assert len(outputs) == len(batch)
                    for p, output in zip(batch, outputs):
                        if isinstance(output, Exception): raise output
                        choice = output.choices[0]; response = choice.message.content or ''
                        pred = ev.letter(response) if fmt == 'mcq' else None
                        record = {k: v for k, v in p.items() if k not in ('messages', 'images')}
                        full = dataclasses.asdict(output); full.pop('prompt_token_ids', None); full.pop('prompt_logprobs', None)
                        record.update(response=response, predicted=pred,
                            correct=pred == p['gold'] if fmt == 'mcq' else fact_score(response, p['gold']) if fmt == 'fact' else None,
                            exact_text_reproduction=response.strip() == str(p['gold']).strip() if fmt == 'recall' else None,
                            format_error=fmt == 'mcq' and pred is None, finish_reason=choice.finish_reason,
                            input_sha256=ev.digest({'messages': p['messages'], 'images': p.get('images', [])}), raw_response=full)
                        append(record)
                    save(folder / 'state.json', {'status': 'running', 'gpu': gpu, 'saved': len(rr), 'expected': total, 'pid': os.getpid(), 'updated_at': ev.now()})
                    print(arm, len(rr), '/', total, flush=True)
            recall = rows(OUT / f'{arm}_recall_probes.jsonl')
            generate([p for p in recall if p['fmt'] == 'recall'])
            generate([p for p in recall if p['fmt'] == 'fact'])
            generate([p for p in applications if p['condition'] == 'oracle'])
            own = {r['qid']: r['response'] for r in rr if r['condition'] == 'neutral_recall'}
            self_probes = []
            for p in applications:
                if p['condition'] != 'closed': continue
                z = copy.deepcopy(p); z.update(id='self:' + p['id'], condition='self_recall')
                z['messages'][-1]['content'] = 'Your own prior recall of this rule (uncorrected):\n' + own[p['qid']] + '\n\n' + z['messages'][-1]['content']
                self_probes.append(z)
            ev.save_rows(folder / 'self_application_probes.jsonl', self_probes)
            generate(self_probes)
    assert len(rr) == total, (arm, len(rr), total)
    save(folder / 'state.json', {'status': 'complete', 'saved': len(rr), 'finished_at': ev.now()})
    summarize()


def summarize():
    report = {}
    for arm in ARMS:
        p = OUT / arm / 'predictions.jsonl'
        rr = ev.read_rows(p, allow_partial_tail=True)[0] if p.exists() else []
        metrics = {}
        for condition in sorted({r['condition'] for r in rr}):
            subset = [r for r in rr if r['condition'] == condition]
            def metric(ss):
                return {'n': len(ss), 'correct': sum(r['correct'] is True for r in ss),
                    'accuracy': sum(r['correct'] is True for r in ss) / len(ss) if ss and ss[0]['fmt'] != 'recall' else None,
                    'exact_text_reproduction': sum(r.get('exact_text_reproduction') is True for r in ss) if ss and ss[0]['fmt'] == 'recall' else None,
                    'truncated': sum(r['finish_reason'] == 'length' for r in ss), 'format_errors': sum(r['format_error'] for r in ss)}
            metrics[condition] = metric(subset)
            for key in ('input_mod', 'task', 'relation'):
                metrics[condition]['by_' + key] = {value: metric([r for r in subset if r.get(key) == value]) for value in sorted({r[key] for r in subset if key in r})}
        report[arm] = {'state': read(OUT / arm / 'state.json'), 'metrics': metrics}
    save(OUT / 'summary.json', report)
    lines = ['# 第10 epoch模型的分层学习诊断', '', '18条规则，覆盖9类、18个结构子类；每类固定SHA抽2条，没有按准确率筛选。仍是开发诊断，不是1000条全量成绩。', '',
        'closed沿用相同输入的正式旧预测；oracle额外提供正确原文字材料；self_recall仅提供模型自己先前生成的未修正规则。后两者单独列出。', '',
        '| 模型 | 闭卷文字 | 闭卷图片 | 给正确材料文字 | 给正确材料图片 | 自身回忆后文字 | 自身回忆后图片 | 原子事实严格匹配 |',
        '|---|---|---|---|---|---|---|---|']
    for arm, info in report.items():
        mm = info['metrics']
        def display(c, inp=None):
            s = mm.get(c, {}).get('by_input_mod', {}).get(inp, {}) if inp else mm.get(c, {})
            return f'{s["correct"]}/{s["n"]} ({s["accuracy"]:.1%})' if s.get('n') and s.get('accuracy') is not None else '待完成'
        lines.append('| ' + arm + ' | ' + ' | '.join([display(c, inp) for c in ('closed', 'oracle', 'self_recall') for inp in ('T', 'V')] + [display('atomic_fact')]) + ' |')
    lines += ['', '完整训练提示复现和名称回忆的回答保存在各模型predictions.jsonl。字面复现不等同于语义正确，原子事实的多余解释或同义回答需要人工核查；不得将严格匹配的失败直接宣称遗忘。', '',
        '3000张原图哈希在准备时已验证不变。没有修改原训练集、题目、图片或历史结果。']
    (OUT / 'report_zh.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--prepare', action='store_true'); ap.add_argument('--summarize', action='store_true')
    ap.add_argument('--arm', choices=ARMS); ap.add_argument('--gpu', default='0'); a = ap.parse_args()
    if a.prepare: prepare()
    elif a.summarize: summarize()
    else:
        assert a.arm
        with ev.run_lock(OUT / (a.arm + '.lock')): run(a.arm, a.gpu)
