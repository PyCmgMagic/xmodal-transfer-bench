#!/usr/bin/env python3
"""One-epoch, same-rule structured/natural training recipe comparison."""
import argparse
import copy
import hashlib
import itertools
import json
import operator
import os
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import run_v7_evaluation as ev
import pilot_v7_application_sft as previous
from run_v7_full_experiment import ROOT, MODELS, environment, rows, read, save, sha, train_command, final_checkpoint
from v7_gpu_resources import reserve

OPS = {'eq': operator.eq, 'ne': operator.ne, 'ge': operator.ge, 'gt': operator.gt, 'le': operator.le, 'lt': operator.lt}
FIELDS = {'category': 'object type', 'shape': 'shape', 'color': 'main-body color',
          'material': 'main-body material', 'texture': 'surface texture',
          'status': 'physical condition', 'value': 'complete object count'}
VALUES = {'category': ['tray', 'bucket', 'storage box', 'toolbox', 'equipment cart', 'bag'],
          'shape': ['circle', 'triangle', 'square', 'pentagon', 'hexagon'],
          'color': ['blue', 'red', 'green', 'black', 'yellow', 'white'],
          'material': ['plastic', 'metal', 'fabric', 'wood', 'leather'],
          'texture': ['striped', 'plain', 'dotted', 'checkered'], 'status': ['intact', 'damaged']}
CODES = ['H7', 'J2', 'K9', 'M4']


def selected_rules(data):
    rules = [r['content'] for r in rows(data / 'audit/rule_knowledge.jsonl')]
    return sorted((r for r in rules if r['fact']['relation'] == 'binary_predicate'),
                  key=lambda r: hashlib.sha256(('application-pilot:42:' + r['id']).encode()).hexdigest())[:12]


def condition_text(condition):
    field, value, op = condition['field'], condition['value'], condition['operator']
    if field == 'value':
        phrase = {'ge': 'at least', 'gt': 'more than', 'le': 'at most', 'lt': 'fewer than',
                  'eq': 'exactly', 'ne': 'a count other than'}[op]
        return f'The batch must contain {phrase} {value} complete objects.'
    assert op in ('eq', 'ne')
    phrase = 'must be' if op == 'eq' else 'must not be'
    return f'The {FIELDS[field]} {phrase} {value}.'


def attributes(rule, flags, variant):
    result = {}
    for c, flag in zip(rule['conditions'], flags):
        value, field, op = c['value'], c['field'], c['operator']
        if isinstance(value, (int, float)):
            candidates = [value, value - 1, value + 1, value - 2, value + 2, value - 3, value + 3]
            candidates = [v for v in candidates if v >= 1 and OPS[op](v, value) == flag]
        else:
            candidates = [value, *[v for v in VALUES[field] if v != value]]
            candidates = [v for v in candidates if OPS[op](v, value) == flag]
        assert candidates
        result[field] = candidates[variant % len(candidates)]
    return result


def plural(noun):
    if noun.endswith(('s', 'x', 'ch', 'sh')):
        return noun + 'es'
    if noun.endswith('y') and noun[-2] not in 'aeiou':
        return noun[:-1] + 'ies'
    return noun + 's'


def observation(attrs, subject='The observed item'):
    if 'value' in attrs:
        return f'{subject} is a batch that contains exactly {attrs["value"]} complete {plural(attrs["category"])}.'
    category = attrs.get('category', 'object')
    material = f' made of {attrs["material"]}' if 'material' in attrs else ''
    color = attrs.get('color', '')
    noun = ' '.join(x for x in (color, category) if x)
    article = 'an' if noun[0].lower() in 'aeiou' else 'a'
    parts = [f'{subject} is {article} {noun}{material}.']
    if 'shape' in attrs:
        parts.append(f'Its outline is a {attrs["shape"]}.')
    if 'texture' in attrs:
        parts.append(f'Its surface is {attrs["texture"]}.')
    if 'status' in attrs:
        parts.append('It has visible physical damage.' if attrs['status'] == 'damaged' else 'It has no visible physical damage.')
    return ' '.join(parts)


def record(rule, flags):
    qualified = all(flags) if rule['mode'] == 'all' else any(flags)
    return '; '.join(f'Check {i + 1}={"Pass" if flag else "Fail"}' for i, flag in enumerate(flags)) + f'; Qualification={"Pass" if qualified else "Fail"}'


def mapping_items(protocol):
    return [(tuple(part.endswith('=Met') for part in text.split('; ')), code)
            for text, code in protocol['outcome_to_code'].items()]


def build_natural(data, out):
    rng = random.Random(42)
    training, checks = [], []
    positions = [0, 1, 2, 3] * 192
    rng.shuffle(positions)
    position_iterator = iter(positions)
    for k in selected_rules(data):
        qid = k['id']
        rule = json.loads(k['fact']['quote'])
        name = rule['predicate_name']
        protocol = k['material']['learning']['inspection_recording_protocol']
        conditions = rule['conditions']
        all_flags = list(itertools.product((False, True), repeat=len(conditions)))
        code_map = dict(mapping_items(protocol))
        compact = '\n'.join(condition_text(c) for c in conditions)
        compact += '\nQualification requires ' + ('all conditions.' if rule['mode'] == 'all' else 'at least one condition.')
        for pair, code in sorted(code_map.items()):
            clauses = [f'{FIELDS[conditions[i - 1]["field"]]} check {"passed" if flag else "failed"}'
                       for i, flag in zip(protocol['checked_indices'], pair)]
            compact += '\n' + '; '.join(clauses) + ' -> ' + code + '.'
        training.append({'id': qid + ':explain', 'qid': qid, 'training_task': 'compact_rule',
                         'messages': [{'role': 'user', 'content': f'State the requirements and recording convention for {name}.'},
                                      {'role': 'assistant', 'content': compact}]})

        def add(task, question, options, answer, audit):
            options = list(options)
            assert len(options) == len(set(options)) == 4 and answer in options
            wrong = [x for x in options if x != answer]
            rng.shuffle(wrong)
            position = next(position_iterator)
            wrong.insert(position, answer)
            prompt = question + '\n\nOptions:\n' + '\n'.join(f'{a}. {v}' for a, v in zip('ABCD', wrong)) + ev.SUFFIX
            identifier = qid + ':natural:' + str(len(training))
            training.append({'id': identifier, 'qid': qid, 'training_task': task,
                             'messages': [{'role': 'system', 'content': ev.SYSTEM}, {'role': 'user', 'content': prompt},
                                          {'role': 'assistant', 'content': 'ABCD'[position]}]})
            checks.append({'id': identifier, 'qid': qid, 'task': task, 'rule': rule, 'protocol': protocol,
                           'options': dict(zip('ABCD', wrong)), 'gold': 'ABCD'[position], **audit})

        for number in range(8):
            c = conditions[number % len(conditions)]
            distractors = []
            for value in ([c['value'] - 1, c['value'] + 1, c['value'] + 2] if c['field'] == 'value'
                          else [x for x in VALUES[c['field']] if x != c['value']][:3]):
                alt = {**c, 'value': value}
                distractors.append(condition_text(alt))
            answer = condition_text(c)
            add('atomic_condition', f'For {name}, which requirement applies to {FIELDS[c["field"]]}?',
                [answer, *distractors], answer, {'condition': c})

        focus_choices = list('ABCD') * 4
        rng.shuffle(focus_choices)
        for number in range(16):
            pair = list(itertools.product((False, True), repeat=2))[number % 4]
            clauses = [f'the {FIELDS[conditions[i - 1]["field"]]} check {"passed" if flag else "failed"}'
                       for i, flag in zip(protocol['checked_indices'], pair)]
            text = ' and '.join(clauses)
            question = (f'Under {name}, {text}. Which inspection record code should be entered?' if number < 8 else
                        f'A completed {name} inspection says that {text}. Select the corresponding record code.')
            code = code_map[pair]
            add('record_code_recall', question, ['Result: ' + c + '.' for c in CODES], 'Result: ' + code + '.',
                {'pair': pair, 'code': code})

        for number in range(16):
            pair = list(itertools.product((False, True), repeat=2))[number % 4]
            flags = [bool((number // 4) % 2)] * len(conditions)
            for index, flag in zip(protocol['checked_indices'], pair):
                flags[index - 1] = flag
            attrs = attributes(rule, flags, number // 4)
            names = [FIELDS[conditions[index - 1]['field']] for index in protocol['checked_indices']]
            focus = focus_choices[number]
            candidates = {}
            if number >= 8:
                for index, candidate in enumerate('ABCD'):
                    candidates[candidate] = attrs if candidate == focus else attributes(rule, all_flags[(number + index) % len(all_flags)], number + index)
                body = ' '.join(observation(candidates[c], 'Candidate ' + c) for c in 'ABCD')
                subject = 'Candidate ' + focus
            else:
                body, subject = observation(attrs), 'the observed item'
            question = (f'Inspect {subject} under {name}. Which inspection record code should be entered for its '
                        f'{names[0]} and {names[1]} requirements? Evaluate each requirement independently.\n\nTest input:\n' + body)
            code = code_map[pair]
            add('practice_application', question, ['Result: ' + c + '.' for c in CODES], 'Result: ' + code + '.',
                {'attributes': attrs, 'flags': flags, 'focus': focus if candidates else None,
                 'candidate_attributes': candidates, 'code': code, 'observation': body})

        for number in range(8):
            flags = all_flags[number % len(all_flags)]
            attrs = attributes(rule, flags, number // len(all_flags) + 1)
            correct = record(rule, flags)
            alternatives = [record(rule, other) for other in all_flags if other != flags]
            rng.shuffle(alternatives)
            result_choices = [correct, *alternatives[:3]]
            # Six correct and eighteen incorrect reports per rule, matching
            # the 1:3 accept/replace ratio of the full frozen test population.
            reports = [correct, *alternatives[:2]] if number < 6 else alternatives[:3]
            for submitted in reports:
                options = [('Accept report. Result: ' if result == submitted else 'Replace report. Result: ') + result + '.' for result in result_choices]
                answer = ('Accept report. Result: ' if submitted == correct else 'Replace report. Result: ') + correct + '.'
                field_order = ', '.join(c['field'] for c in conditions)
                question = (f'A technician submitted this report under {name}: Result: {submitted}. '
                            'Check the observations and the rule. Check the conditions in this order: ' + field_order +
                            '. Report Pass/Fail for each condition and the final qualification.\n\nTest input:\n' + observation(attrs))
                add('practice_diagnosis', question, options, answer,
                    {'attributes': attrs, 'flags': list(flags), 'submitted': submitted, 'observation': observation(attrs)})
    rng.shuffle(training)
    assert len(training) == 780 and len(checks) == 768
    assert Counter(r['gold'] for r in checks) == Counter({a: 192 for a in 'ABCD'})
    assert sum(r['options'][r['gold']].startswith('Accept report.') for r in checks) == 72
    ev.save_rows(out / 'train.jsonl', training)
    ev.save_rows(out / 'generated_solver_checks.jsonl', checks)


def prepare(data, work, out, arm, epochs, training_settings=None):
    training_settings = training_settings or {'gradient_accumulation_steps': 16, 'learning_rate': 1e-4}
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'manifest.json').exists():
        manifest = read(out / 'manifest.json')
        assert manifest['arm'] == arm and manifest['epochs'] == epochs
        assert manifest.get('training_settings', {'gradient_accumulation_steps': 16, 'learning_rate': 1e-4}) == training_settings
        assert manifest['train_sha256'] == sha(out / 'train.jsonl')
        return
    if arm == 'structured':
        previous.build(data, work, out)
    else:
        build_natural(data, out)
        ids = {r['id'] for r in selected_rules(data)}
        labels = {r['id']: r for r in rows(data / 'audit/eval_labels.jsonl')}
        evaluation = [ev.prepare_row(r, data, 'T2T') for r in rows(data / 'T2T/eval.jsonl') if r['qid'] in ids]
        assert len(evaluation) == 24
        for r in evaluation:
            r.update(gold=labels[r['id']]['answer'], question_type=labels[r['id']]['question_type'])
        ev.save_rows(out / 'eval.jsonl', evaluation)
    save(out / 'manifest.json', {'arm': arm, 'epochs': epochs, 'rules': sorted(r['id'] for r in selected_rules(data)),
         'training_rows': len(rows(out / 'train.jsonl')), 'train_sha256': sha(out / 'train.jsonl'),
         'eval_sha256': sha(out / 'eval.jsonl'), 'knowledge_sha256': sha(data / 'audit/rule_knowledge.jsonl'),
         'training_source': 'learning rule structure and inspection convention only; no frozen evaluation labels used in training',
         'script_sha256': sha(Path(__file__)), 'strict_only': True, 'new_images': False, 'training_settings': training_settings,
         'comparability_limit': 'Same 12 rules, base model, LoRA settings and epochs; natural changes training recipe and sample count (780 vs 548), so not a single-factor causal experiment.'})


def probes(out, data):
    grouped = defaultdict(list)
    for r in rows(out / 'train.jsonl'):
        if r['id'].endswith(':explain'):
            continue
        task = r.get('training_task')
        if not task:
            text = r['messages'][1]['content']
            task = ('atomic_condition' if text.startswith('For ') else 'record_code_recall' if text.startswith('Under ')
                    else 'practice_application' if text.startswith('Inspect ') else 'practice_diagnosis')
        grouped[(r['qid'], task)].append(r)
    result = []
    for (qid, task), group in sorted(grouped.items()):
        for r in sorted(group, key=lambda r: hashlib.sha256(('natural-pilot-control:42:' + r['id']).encode()).hexdigest())[:2]:
            result.append({'condition': task, 'id': r['id'], 'qid': qid, 'gold': r['messages'][-1]['content'],
                           'messages': copy.deepcopy(r['messages'][:-1])})
    knowledge = {r['content']['id']: r['content'] for r in rows(data / 'audit/rule_knowledge.jsonl')}
    for r in rows(out / 'eval.jsonl'):
        for material in (False, True):
            messages = copy.deepcopy(r['messages'])
            if material:
                messages[-1]['content'] = 'Learning material for this question:\n' + knowledge[r['qid']]['material']['learning']['text'] + '\n\n' + messages[-1]['content']
            result.append({'condition': 'original_' + ('material_' if material else 'blind_') + r['question_type'],
                           'id': r['id'], 'qid': r['qid'], 'gold': r['gold'], 'messages': messages})
    assert len(result) == 144
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--arm', choices=['structured', 'natural'], required=True)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--gradient-accumulation', type=int, default=16)
    ap.add_argument('--learning-rate', type=float, default=1e-4)
    ap.add_argument('--prepare-only', action='store_true')
    a = ap.parse_args()
    assert a.epochs >= 1 and a.gradient_accumulation >= 1 and a.learning_rate > 0
    out = a.out.resolve()
    data, work = ROOT / 'rules_repaired_single_image_v7', ROOT / 'runs/v7_full_experiment'
    prepare(data, work, out, a.arm, a.epochs, {'gradient_accumulation_steps': a.gradient_accumulation, 'learning_rate': a.learning_rate})
    if a.prepare_only:
        print(read(out / 'manifest.json'))
        return
    with ev.run_lock(out / '.lock'):
        cp = final_checkpoint(out / 'checkpoint', a.epochs)
        if not cp:
            with reserve(work, a.gpu, str(out) + ':training', 16000) as reservation:
                cmd = train_command('qwen3', 'sft', out / 'train.jsonl', out / 'checkpoint', a.epochs)
                cmd[cmd.index('--gradient_accumulation_steps') + 1] = str(a.gradient_accumulation)
                cmd[cmd.index('--learning_rate') + 1] = str(a.learning_rate)
                with (out / 'train.log').open('a') as log:
                    child = subprocess.Popen(cmd, cwd=ROOT, env=environment(a.gpu, data), stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    reservation.bind(child.pid)
                    save(out / 'state.json', {'status': 'training', 'pid': child.pid, 'command': cmd, 'started_at': ev.now()})
                    code = child.wait()
                if code:
                    save(out / 'state.json', {'status': 'failed', 'returncode': code})
                    raise RuntimeError(f'{a.arm} training failed: {code}')
            cp = final_checkpoint(out / 'checkpoint', a.epochs)
            assert cp
        evaluation = probes(out, data)
        ev.save_rows(out / 'probes.jsonl', evaluation)
        with reserve(work, a.gpu, str(out) + ':evaluation', 22000) as reservation:
            reservation.bind(os.getpid())
            os.environ.update(environment(a.gpu, data))
            import torch
            from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
            from swift.utils import seed_everything
            seed_everything(42)
            engine = TransformersEngine(str(MODELS['qwen3']), adapters=[str(cp)], torch_dtype=torch.bfloat16,
                                       device_map='cuda:0', max_batch_size=16, attn_impl='sdpa')
            engine.template.enable_thinking = False
            save(out / 'state.json', {'status': 'evaluating', 'checkpoint': str(cp), 'pid': os.getpid()})
            results = []
            with (out / 'predictions.jsonl').open('w') as stream:
                for start in range(0, len(evaluation), 16):
                    batch = evaluation[start:start + 16]
                    responses = engine.infer([InferRequest(messages=copy.deepcopy(r['messages'])) for r in batch],
                        RequestConfig(max_tokens=32, temperature=0, seed=42), use_tqdm=False)
                    assert len(responses) == len(batch)
                    for r, response in zip(batch, responses):
                        if isinstance(response, Exception):
                            raise response
                        text = response.choices[0].message.content or ''
                        answer = ev.letter(text)
                        result = {k: r[k] for k in ('condition', 'id', 'qid', 'gold')}
                        result.update(response=text, parsed_answer=answer, correct=answer == r['gold'])
                        results.append(result)
                        stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                        stream.flush()
                    print(a.arm, 'evaluated', len(results), '/', len(evaluation), flush=True)
        metrics = {}
        for condition in sorted({r['condition'] for r in results}):
            group = [r for r in results if r['condition'] == condition]
            metrics[condition] = {'n': len(group), 'correct': sum(r['correct'] for r in group),
                                 'format_errors': sum(r['parsed_answer'] is None for r in group)}
        save(out / 'report.json', {'status': 'complete', 'arm': a.arm, 'epochs': a.epochs, 'metrics': metrics,
                                  'checkpoint': str(cp), 'checkpoint_sha256': ev.checkpoint(cp), 'finished_at': ev.now()})
        save(out / 'state.json', {'status': 'complete', 'finished_at': ev.now()})
        print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
