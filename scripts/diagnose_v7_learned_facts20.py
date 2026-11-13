#!/usr/bin/env python3
"""Material-aligned MCQs distinguish factual recognition from rule application."""
import argparse
import copy
import dataclasses
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import run_v7_evaluation as ev
from audit_v7_ep10_learning import natural_facts
from build_v7_experiment_data import name
from run_v7_full_experiment import ROOT, environment, read, rows, save, sha
from run_v7_fullparam_ep10 import dense_identity
from v7_gpu_resources import reserve

DATA = ROOT / 'rules_repaired_single_image_v7'
SOURCE = ROOT / 'runs/v7_fullparam_sft_ep10_20261019'
OUT = ROOT / 'runs/v7_learned_facts20_20261021'
LEDGER = ROOT / 'runs/v7_full_experiment'
ARMS = ('base', 'text')


def material_facts(k):
    spec = json.loads(k['fact']['quote'])
    rel = k['fact']['relation']
    if rel == 'set_selection_rule' and 'mode' not in spec:
        result = []
        for condition in spec['conditions'][:2]:
            field = condition['field']
            result.append(dict(question=f'For the stored rule {name(k)}, what {field} value appears in its {field} condition? Return only the requested short answer.', answer=condition['value']))
        result.append(dict(question=f'For the stored rule {name(k)}, at least how many of its listed conditions must hold? Return only the requested short answer.', answer=spec['required_count']))
        return result
    if rel == 'transformation_procedure':
        operations = spec['operations']
        first = operations[0]['op']
        display = {'rotate_left': 'rotate the whole row left',
                   'reverse': 'reverse the whole row',
                   'swap_disjoint_pairs': 'swap non-overlapping adjacent pairs'}[first]
        return [dict(question=f'For the stored procedure {name(k)}, what is the first operation? Return only the requested short answer.', answer=display),
                dict(question=f'For the stored procedure {name(k)}, how many times is stage 1 performed? Return only the requested short answer.', answer=operations[0]['repeat']),
                dict(question=f'For the stored procedure {name(k)}, how many times is stage 2 performed? Return only the requested short answer.', answer=operations[1]['repeat'])]
    if rel == 'mapping_codebook' and 'mapping' not in spec:
        entries = [(context, item, code) for context, table in spec['mapping_by_context'].items()
                   for item, code in table.items()]
        return [dict(question=f'For the stored codebook {name(k)}, what codeword is assigned to {item} with {context}? Return only the requested short answer.',
                     answer=code) for context, item, code in entries[:3]]
    return natural_facts(k)


def mapping_values(spec):
    return list(spec['mapping'].values()) if 'mapping' in spec else [value for table in spec['mapping_by_context'].values() for value in table.values()]


def key(value):
    normalized = re.sub(r'\s+', ' ', str(value)).strip().lower()
    if ',' in normalized:
        return ','.join(sorted(part.strip() for part in normalized.split(',')))
    return normalized


def order(tag, value):
    return hashlib.sha256((tag + ':42:' + value).encode()).hexdigest()


def alternatives(fact, knowledge, slot, all_facts):
    gold = fact['answer']
    rel = knowledge['fact']['relation']
    spec = json.loads(knowledge['fact']['quote'])
    if isinstance(gold, (int, float)):
        if 'proportion threshold' in fact['question']:
            pool = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9]
        else:
            pool = [gold + delta for delta in (-3, -2, -1, 1, 2, 3)]
            if rel == 'transformation_procedure':
                pool = [x for x in pool if x > 0]
    elif 'all conditions hold' in fact['question']:
        pool = ['all', 'any', 'exactly one', 'none']
    elif rel in ('binary_predicate', 'set_selection_rule') and slot < 2:
        condition = spec['conditions'][slot]
        pool = []
        for item in all_facts:
            other_spec = json.loads(item['fact']['quote'])
            for other in other_spec.get('conditions', []):
                if other['field'] == condition['field'] and other['operator'] == condition['operator']:
                    pool.append(other['value'])
    elif rel == 'binary_predicate':
        pool = ['H7', 'K9', 'J2', 'M4']
    elif rel == 'finite_state_transition':
        pool = list(spec['transition'])
    elif rel == 'mapping_codebook':
        pool = mapping_values(spec)
    elif rel == 'ordering_priority':
        pool = list(spec['priority'])
    elif rel == 'three_way_category':
        pool = list(spec['category_labels']) + ['none of these categories']
    elif rel == 'transformation_procedure':
        pool = ['reverse the whole row', 'rotate the whole row left', 'swap non-overlapping adjacent pairs', 'rotate the whole row right']
    else:
        pool = [f['answer'] for item in all_facts if item['fact']['relation'] == rel
                for f in material_facts(item)[slot:slot + 1]]
    distinct = {key(x): x for x in pool if key(x) != key(gold)}
    candidates = sorted(distinct.values(), key=lambda x: order(knowledge['id'] + ':' + str(slot), key(x)))
    if len(candidates) < 3 and rel in ('finite_state_transition', 'mapping_codebook'):
        extra = [value for item in all_facts if item['fact']['relation'] == rel
                 for value in (list(json.loads(item['fact']['quote'])['transition']) if rel == 'finite_state_transition'
                               else mapping_values(json.loads(item['fact']['quote']))) ]
        extra = {key(x): x for x in extra if key(x) not in distinct and key(x) != key(gold)}
        candidates += sorted(extra.values(), key=lambda x: order(knowledge['id'] + ':' + str(slot), key(x)))
    if len(candidates) < 3 and rel in ('binary_predicate', 'set_selection_rule') and slot < 2:
        condition = spec['conditions'][slot]
        extra = [c['value'] for item in all_facts
                 for c in json.loads(item['fact']['quote']).get('conditions', [])
                 if c['field'] == condition['field']]
        extra = {key(x): x for x in extra if key(x) not in distinct and key(x) != key(gold)}
        candidates += sorted(extra.values(), key=lambda x: order(knowledge['id'] + ':' + str(slot), key(x)))
    if len(candidates) < 3:
        generic = ['None of the other listed values', 'No value is specified', 'Not enough information']
        extra = [x for x in generic if key(x) not in distinct and key(x) != key(gold)]
        candidates += extra[:3 - len(candidates)]
    assert len(candidates) >= 3, (knowledge['id'], fact, candidates)
    return candidates[:3]


def evidence(text, answer):
    token = str(answer)
    if isinstance(answer, (int, float)):
        pattern = r'(?<![\d.])' + re.escape(token) + r'(?![\d.])'
        match = re.search(pattern, text)
    else:
        match = re.search(re.escape(token), text, flags=re.I)
    if match:
        return text[max(0, match.start() - 100):min(len(text), match.end() + 150)]
    aliases = {'reverse the whole row': r'revers', 'swap non-overlapping adjacent pairs': r'non-overlapping adjacent pairs',
               '1': r'once', '2': r'twice'}
    if token in aliases and re.search(aliases[token], text, re.I):
        return text
    # Some definitions express one list with commas rather than "and".
    pieces = [p.strip() for p in re.split(r',|\band\b', token) if p.strip()]
    assert pieces and all(re.search(re.escape(p), text, re.I) for p in pieces), (answer, text)
    return text


def validate():
    manifest = read(OUT / 'manifest.json')
    for path, expected in manifest['sources'].items():
        assert sha(Path(path)) == expected, path
    for path, expected in manifest['files'].items():
        assert sha(OUT / path) == expected, path
    for path, expected in manifest['images'].items():
        assert sha(Path(path)) == expected, path
    return manifest


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'manifest.json').exists():
        previous = read(OUT / 'manifest.json')
        if previous['sources'].get(str(Path(__file__))) == sha(Path(__file__)):
            validate()
            return
        assert not list(OUT.glob('*/predictions.jsonl')), 'Cannot rebuild after inference has started'
    knowledge = [r['content'] for r in rows(DATA / 'audit/rule_knowledge.jsonl')]
    groups = defaultdict(list)
    for item in knowledge:
        groups[item['fact']['relation']].append(item)
    selected = []
    for rel, group in sorted(groups.items()):
        ranked = sorted(group, key=lambda k: order('learned-facts20', k['id']))
        first = ranked[0]
        family = json.loads(first['fact']['quote'])['structural_subfamily_id']
        second = next(k for k in ranked[1:]
                      if json.loads(k['fact']['quote'])['structural_subfamily_id'] != family)
        selected.extend([first, second])
    for rel in sorted(groups, key=lambda rel: (-len(groups[rel]), rel))[:2]:
        selected_ids = {k['id'] for k in selected}
        selected.append(next(k for k in sorted(groups[rel], key=lambda k: order('learned-facts20', k['id']))
                             if k['id'] not in selected_ids))
    assert len(selected) == 20 and len(groups) == 9
    training = {r['id']: r for r in rows(SOURCE / 'data/sft_text.jsonl')}
    label_positions = list(range(4)) * 15
    random.Random(42).shuffle(label_positions)
    probes, facts = [], []
    for k in selected:
        original = training[k['id']]
        material = original['messages'][-1]['content']
        assert material == k['material']['learning']['text']
        for slot, fact in enumerate(material_facts(k)):
            question = fact['question'].replace(' Return only the requested short answer.', '')
            question = question.replace(' Answer all or any.', '')
            options = alternatives(fact, k, slot, knowledge)
            answer_position = label_positions[len(facts)]
            options.insert(answer_position, fact['answer'])
            assert len({key(x) for x in options}) == 4
            item = dict(id=f'fact:{k["id"]}:{slot}', qid=k['id'], relation=k['fact']['relation'],
                        condition='fact_closed', task='fact', gold='ABCD'[answer_position],
                        question=question, options=options, answer_value=fact['answer'],
                        evidence=evidence(material, fact['answer']), training_id=original['id'])
            facts.append(copy.deepcopy(item))
            content = question + '\n\nOptions:\n' + '\n'.join(
                f'{letter}. {value}' for letter, value in zip('ABCD', options)) + ev.SUFFIX
            item['messages'] = [dict(role='system', content=ev.SYSTEM), dict(role='user', content=content)]
            probes.append(item)
            oracle = copy.deepcopy(item)
            oracle.update(id='oracle:' + item['id'], condition='fact_oracle')
            oracle['messages'][-1]['content'] = 'Correct learning material:\n' + material + '\n\n' + content
            probes.append(oracle)
    assert len(facts) == 60
    selected_ids = {k['id'] for k in selected}
    relations = {k['id']: k['fact']['relation'] for k in selected}
    labels = {r['id']: r for r in rows(DATA / 'audit/eval_labels.jsonl')}
    for mod, direction in [('T', 'T2T'), ('V', 'T2V')]:
        for source in rows(SOURCE / 'data' / f'eval_{direction}.jsonl'):
            if source['qid'] not in selected_ids:
                continue
            label = labels[source['id']]
            p = copy.deepcopy(source)
            p.update(id=f'application_{mod}:' + source['id'], source_id=source['id'],
                     condition='application_' + mod, relation=relations[source['qid']],
                     gold=label['answer'], task=label['question_type'], evaluation_group=label['evaluation_group'])
            probes.append(p)
            if mod == 'T':
                oracle = copy.deepcopy(p)
                oracle.update(id='oracle:' + p['id'], condition='application_oracle_T')
                oracle['messages'][-1]['content'] = 'Correct learning material:\n' + training[p['qid']]['messages'][-1]['content'] + '\n\n' + oracle['messages'][-1]['content']
                probes.append(oracle)
    assert len(probes) == len({p['id'] for p in probes}) == 240
    ev.save_rows(OUT / 'selected_knowledge.jsonl', selected)
    ev.save_rows(OUT / 'facts_with_evidence.jsonl', facts)
    ev.save_rows(OUT / 'probes.jsonl', probes)
    model_manifest = read(SOURCE / 'manifest.json')
    models = {'base': model_manifest['model'],
              'text': read(SOURCE / 'jobs/text/checkpoint_manifest.json')['model']}
    sources = [Path(__file__), ROOT / 'scripts/audit_v7_ep10_learning.py',
               ROOT / 'scripts/run_v7_evaluation.py', ROOT / 'scripts/run_v7_full_experiment.py',
               ROOT / 'scripts/v7_gpu_resources.py', SOURCE / 'manifest.json',
               SOURCE / 'jobs/text/checkpoint_manifest.json', SOURCE / 'data/sft_text.jsonl',
               DATA / 'audit/rule_knowledge.jsonl', DATA / 'audit/eval_labels.jsonl',
               SOURCE / 'data/eval_T2T.jsonl', SOURCE / 'data/eval_T2V.jsonl']
    images = {path for p in probes for path in p.get('images', [])}
    manifest = dict(created_at=ev.now(), selection='SHA256 seed42; two per relation with distinct subfamilies, plus one from each of the two largest relations; no prediction filtering.',
                    rules=[k['id'] for k in selected], relations=dict(Counter(k['fact']['relation'] for k in selected)),
                    models=models, arms=ARMS, training='No additional training. Existing full 1000-rule TEXT SFT, final epoch10; untrained Qwen3-VL-4B control.',
                    conditions=dict(Counter(p['condition'] for p in probes)),
                    generation=dict(batch=4, max_tokens=32, temperature=0, seed=42, thinking=False),
                    sources={str(p): sha(p) for p in sources},
                    files={p.name: sha(p) for p in [OUT / 'probes.jsonl', OUT / 'selected_knowledge.jsonl', OUT / 'facts_with_evidence.jsonl']},
                    images={p: sha(Path(p)) for p in sorted(images)},
                    limits=['20 development rules; factual MCQs measure recognition, not proof of verbatim recall or full mastery.',
                            'All-condition 3/3 factual success is a descriptive grouping; not a significance test.',
                            'Materials-open controls are diagnostic and excluded from closed-book transfer scores.',
                            'No test application questions added to training. Original images and data remain untouched.'])
    save(OUT / 'manifest.json', manifest)
    lines = ['# 20条规则：先检查学到的事实，再检查应用', '',
             '使用已经完成的1000条规则TEXT全量SFT第10轮模型，并与未训练基座对照。本次不追加训练。', '',
             '先用60道四选一题直接检查训练材料中的条件、参数和映射；再测试同一批规则的40道文字、40道原图应用题。',
             '给正确材料的60道事实题和40道文字应用题是单列的可答性检查。每组共240题，两组480题。', '',
             '事实选择题检查的是辨认能力，不能直接等同于完整自由回忆；逐规则记录3题得分，避免只看总平均。', '',
             '## 逐条材料与问题', '']
    for k in selected:
        lines += [f'### {name(k)}（{k["id"]}，{k["fact"]["relation"]}）', '',
                  '实际训练材料：', '', training[k['id']]['messages'][-1]['content'], '']
        for fact in [x for x in facts if x['qid'] == k['id']]:
            lines += [fact['question'], ''] + [f'{a}. {v}' for a, v in zip('ABCD', fact['options'])]
            lines += ['', f'标准答案：{fact["gold"]}；材料依据：{fact["evidence"]}', '']
    (OUT / 'questions_zh.md').write_text('\n'.join(lines) + '\n')
    manifest['files']['questions_zh.md'] = sha(OUT / 'questions_zh.md')
    save(OUT / 'manifest.json', manifest)
    validate()
    print(json.dumps(dict(rules=20, facts=60, requests_per_arm=240, conditions=manifest['conditions']), ensure_ascii=False), flush=True)


def run(arm, gpu):
    manifest = validate()
    folder = OUT / arm
    folder.mkdir(exist_ok=True)
    with ev.run_lock(folder / '.lock'):
        model = manifest['models'][arm]
        assert dense_identity(Path(model['path'])) == model
        probes = rows(OUT / 'probes.jsonl')
        by_id = {p['id']: p for p in probes}
        raw = folder / 'predictions.jsonl'
        records, notes = ev.read_rows(raw, allow_partial_tail=True) if raw.exists() else ([], [])
        if notes:
            ev.save_rows(raw, records)
        seen = {r['id'] for r in records}
        assert len(seen) == len(records)
        for record in records:
            probe = by_id[record['id']]
            assert record['input_sha256'] == ev.digest(dict(messages=probe['messages'], images=probe.get('images', [])))
        pending = [p for p in probes if p['id'] not in seen]
        if pending:
            save(folder / 'state.json', dict(status='waiting_for_gpu', pid=os.getpid(), gpu=gpu))
            with reserve(LEDGER, gpu, str(folder) + ':eval', 18000) as reservation:
                reservation.bind(os.getpid())
                os.environ.update(environment(gpu, DATA))
                import torch
                from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
                from swift.utils import seed_everything
                torch.set_num_threads(4)
                seed_everything(42)
                engine = TransformersEngine(model['path'], adapters=[], torch_dtype=torch.bfloat16,
                                            device_map='cuda:0', max_batch_size=4, attn_impl='sdpa')
                engine.template.enable_thinking = False
                assert not getattr(engine.model, 'peft_config', None)
                for start in range(0, len(pending), 4):
                    batch = pending[start:start + 4]
                    replies = engine.infer([InferRequest(messages=copy.deepcopy(p['messages']), images=p.get('images', [])) for p in batch],
                                           RequestConfig(max_tokens=32, temperature=0, seed=42, return_details=True), use_tqdm=False)
                    assert len(replies) == len(batch)
                    for p, reply in zip(batch, replies):
                        if isinstance(reply, Exception):
                            raise reply
                        choice = reply.choices[0]
                        response = choice.message.content or ''
                        predicted = ev.letter(response)
                        r = {key: p[key] for key in ('id', 'qid', 'condition', 'relation', 'gold', 'task')}
                        r.update(response=response, predicted=predicted, correct=predicted == p['gold'],
                                 format_error=predicted is None, finish_reason=choice.finish_reason,
                                 input_sha256=ev.digest(dict(messages=p['messages'], images=p.get('images', []))),
                                 raw_response=dataclasses.asdict(reply))
                        with raw.open('a') as stream:
                            stream.write(json.dumps(r, ensure_ascii=False, default=str) + '\n')
                            stream.flush()
                            os.fsync(stream.fileno())
                        records.append(r)
                    save(folder / 'state.json', dict(status='running', saved=len(records), total=240, pid=os.getpid()))
                    print(arm, len(records), '/240', flush=True)
        assert len(records) == len({r['id'] for r in records}) == 240
        assert {r['id'] for r in records} == set(by_id)
        for r in records:
            assert r['response'] == (r['raw_response']['choices'][0]['message']['content'] or '')
            assert r['predicted'] == ev.letter(r['response'])
        def metric(rr):
            return dict(n=len(rr), correct=sum(r['correct'] for r in rr),
                        accuracy_percent=100 * sum(r['correct'] for r in rr) / len(rr) if rr else None,
                        format_errors=sum(r['format_error'] for r in rr),
                        truncated=sum(r['finish_reason'] == 'length' for r in rr))
        metrics = {c: metric([r for r in records if r['condition'] == c]) for c in manifest['conditions']}
        per_rule = {qid: {c: metric([r for r in records if r['qid'] == qid and r['condition'] == c])
                          for c in manifest['conditions']} for qid in manifest['rules']}
        save(folder / 'report.json', dict(status='complete', arm=arm, metrics=metrics, per_rule=per_rule))
        save(folder / 'state.json', dict(status='complete', saved=240, completed_at=ev.now()))
        print(json.dumps(metrics, ensure_ascii=False), flush=True)


def report():
    manifest = validate()
    reports = {arm: read(OUT / arm / 'report.json') for arm in ARMS}
    assert all(r.get('status') == 'complete' for r in reports.values())
    labels = {'fact_closed': '闭卷：训练事实辨认', 'fact_oracle': '给材料：训练事实辨认',
              'application_T': '闭卷：文字应用', 'application_V': '闭卷：原图应用',
              'application_oracle_T': '给材料：文字应用'}
    lines = ['# 20条规则的学习与应用诊断', '',
             '训练模型：Qwen3-VL-4B，原始1000条规则解释SFT，全量微调10轮；本次只评测，没有追加训练。',
             '20条规则覆盖9类，按固定哈希抽取，没有按模型成绩挑选。两组使用相同问题和生成设置。', '',
             '| 检查项 | 未训练基座 | 训练后 |', '|---|---:|---:|']
    for condition, label in labels.items():
        cells = []
        for arm in ARMS:
            m = reports[arm]['metrics'][condition]
            cells.append(f'{m["accuracy_percent"]:.2f}%（{m["correct"]}/{m["n"]}）')
        lines.append('| ' + label + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '## 逐规则结果', '',
              '| 规则 | 基座事实/3 | 训练后事实/3 | 训练后文字应用/2 | 训练后原图应用/2 | 给材料文字应用/2 |',
              '|---|---:|---:|---:|---:|---:|']
    knowledge = {k['id']: k for k in rows(OUT / 'selected_knowledge.jsonl')}
    for qid in manifest['rules']:
        base = reports['base']['per_rule'][qid]
        trained = reports['text']['per_rule'][qid]
        values = [base['fact_closed']['correct']] + [trained[c]['correct'] for c in ('fact_closed', 'application_T', 'application_V', 'application_oracle_T')]
        lines.append('| ' + name(knowledge[qid]) + ' | ' + ' | '.join(map(str, values)) + ' |')
    lines += ['', '## 如何解读', '',
              '闭卷事实辨认仍低，说明规则的事实记忆或提取没有建立；不能直接认定只是不会应用。',
              '同一规则的事实辨认好、应用差，才提示应用环节可能有困难；3道事实题只覆盖部分知识，不足以证明完全掌握。',
              '给材料后事实题仍差的条目，需要先检查提问、材料表述和输出格式。给材料后应用仍差，说明知识缺失之外还存在应用或题目表述方面的问题。',
              '给材料结果是诊断条件，不是闭卷跨模态迁移成绩。20条开发样本不能外推1000条整体成绩。', '',
              '题目与逐条材料依据见 questions_zh.md；所有原始模型输出见各组 predictions.jsonl。']
    (OUT / 'report_zh.md').write_text('\n'.join(lines) + '\n')
    save(OUT / 'summary.json', reports)
    print('\n'.join(lines[:14]), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['prepare', 'evaluate', 'report'], required=True)
    parser.add_argument('--arm', choices=ARMS, default='text')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare()
    elif args.stage == 'evaluate':
        run(args.arm, args.gpu)
    else:
        report()
