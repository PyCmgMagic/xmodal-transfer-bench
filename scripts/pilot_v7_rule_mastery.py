#!/usr/bin/env python3
"""One-epoch rule-memory capacity control, with frozen MCQ transfer probes."""
import argparse
import copy
import itertools
import json
import operator
import os
import random
import subprocess
from collections import Counter
from pathlib import Path

import run_v7_evaluation as ev
from pilot_v7_natural_sft import (FIELDS, attributes, condition_text, mapping_items,
                                  observation, record, selected_rules)
from run_v7_full_experiment import (ROOT, MODELS, environment, final_checkpoint,
                                    read, rows, save, sha, train_command)
from run_v7_semantic_pilot import semantic_equal
from audit_v7_semantic_outputs import field_equal
from v7_gpu_resources import reserve

OPS = {'eq': operator.eq, 'ne': operator.ne, 'ge': operator.ge,
       'gt': operator.gt, 'le': operator.le, 'lt': operator.lt}
DATA = ROOT / 'rules_repaired_single_image_v7'
WORK = ROOT / 'runs/v7_full_experiment'
CONTROL = ROOT / 'runs/v7_natural_1epoch_20261014/natural'


def messages(question, answer=None, task='semantic'):
    instruction = (ev.SYSTEM if task == 'mcq' else
                   'Answer the requested rule fact, recording code, or inspection record directly. Do not give reasoning.')
    result = [{'role': 'system', 'content': instruction}, {'role': 'user', 'content': question}]
    if answer is not None:
        result.append({'role': 'assistant', 'content': answer})
    return result


def build_training(knowledge):
    output, audits = [], []
    for k in knowledge:
        rule = json.loads(k['fact']['quote'])
        protocol = k['material']['learning']['inspection_recording_protocol']
        name, qid = rule['predicate_name'], k['id']
        code_map = dict(mapping_items(protocol))
        flags_list = list(itertools.product((False, True), repeat=len(rule['conditions'])))

        def add(task, question, answer, **audit):
            identifier = f'{qid}:mastery:{len(output)}'
            output.append({'id': identifier, 'qid': qid, 'task': task,
                           'messages': messages(question, answer)})
            audits.append({'id': identifier, 'qid': qid, 'task': task, 'answer': answer, **audit})

        for prefix in ('State', 'Recall', 'Summarize', 'List'):
            for focus in ('the requirements and inspection codes', 'the definition and recording convention',
                          'all conditions and the four-code mapping', 'the qualification rule and codebook'):
                add('rule_card', f'{prefix} {focus} for {name}.', k['material']['learning']['text'])
        # Forty-eight fact recalls, distributed over the rule's conditions.
        forms = [
            'For {name}, state the requirement for {field}.',
            'What does {name} require for {field}?',
            'Recall the {field} requirement of {name}.',
            'Under {name}, what must hold for {field}?',
        ]
        endings = (' Give one sentence.', ' State the exact condition.',
                   ' Answer with the requirement only.', ' Be precise.',
                   ' Return the required condition.', ' Give the condition directly.')
        for number in range(48):
            c = rule['conditions'][number % len(rule['conditions'])]
            variant = number // len(rule['conditions'])
            question = forms[variant % 4].format(name=name, field=FIELDS[c['field']]) + endings[variant // 4]
            add('atomic_condition', question, condition_text(c), condition=c)
        for pair in itertools.product((False, True), repeat=2):
            clauses = [f'{FIELDS[rule["conditions"][index - 1]["field"]]} check {"passed" if flag else "failed"}'
                       for index, flag in zip(protocol['checked_indices'], pair)]
            for lead in ('Under', 'According to', 'Using', 'Following'):
                for suffix in ('Return the recording code only.', 'Which inspection code applies?',
                               'State the two-character record code.', 'Enter the corresponding inspection code.'):
                    question = f'{lead} the {name} convention, ' + ' and '.join(clauses) + '. ' + suffix
                    add('record_code_recall', question, code_map[pair], pair=list(pair))
        for number in range(64):
            flags = flags_list[number % len(flags_list)]
            variant = number // len(flags_list)
            attrs = attributes(rule, flags, variant)
            actual = tuple(OPS[c['operator']](attrs[c['field']], c['value']) for c in rule['conditions'])
            assert actual == flags
            pair = tuple(actual[index - 1] for index in protocol['checked_indices'])
            subject = ('The observed item', 'This item', 'The submitted item', 'The inspected object')[variant % 4]
            body = observation(attrs, subject)
            field_names = ' and '.join(FIELDS[rule['conditions'][index - 1]['field']] for index in protocol['checked_indices'])
            application = (f'Inspect the item under {name}. Return only the record code for the {field_names} '
                           f'requirements; assess them independently.\nObservation: {body}')
            add('practice_application', application, code_map[pair], attributes=attrs, flags=list(flags))
            submitted = record(rule, flags_list[(number + variant + 1) % len(flags_list)])
            diagnosis = (f'A technician submitted this report under {name}: {submitted}. '
                         'Return the correct Check statuses in this order: ' +
                         ', '.join(c['field'] for c in rule['conditions']) +
                         f', followed by Qualification. Use Pass/Fail.\nObservation: {body}')
            add('practice_diagnosis', diagnosis, record(rule, actual), attributes=attrs, flags=list(actual))
    assert len(output) == 256 * len(knowledge)
    # Intentional prompt/observation re-exposure is measured, never hidden as new facts.
    random.Random(42).shuffle(output)
    return output, audits


def make_probes(knowledge, training):
    result = []
    ids = {k['id'] for k in knowledge}

    def add(condition, identifier, qid, task, prompt, gold, answer_format='semantic'):
        result.append({'condition': condition, 'id': identifier, 'qid': qid, 'task': task,
                       'messages': prompt, 'gold': gold, 'answer_format': answer_format})

    grouped = {}
    for r in training:
        grouped.setdefault((r['qid'], r['task']), []).append(r)
    for (qid, task), group in sorted(grouped.items()):
        if task == 'rule_card':
            continue
        for r in sorted(group, key=lambda r: ev.digest(r['id']))[:8]:
            add('seen_' + task, r['id'], qid, task, r['messages'][:-1], r['messages'][-1]['content'])
    for k in knowledge:
        rule = json.loads(k['fact']['quote'])
        name, qid = rule['predicate_name'], k['id']
        protocol = k['material']['learning']['inspection_recording_protocol']
        code_map = dict(mapping_items(protocol))
        flags_list = list(itertools.product((False, True), repeat=len(rule['conditions'])))
        for index, flags in enumerate(flags_list):
            # New wording and observations (verified against training prompt strings below).
            attrs = attributes(rule, flags, 17)
            actual = tuple(OPS[c['operator']](attrs[c['field']], c['value']) for c in rule['conditions'])
            assert actual == flags
            body = '; '.join(FIELDS[field] + '=' + str(value) for field, value in reversed(list(attrs.items())))
            pair = tuple(actual[i - 1] for i in protocol['checked_indices'])
            add('heldout_application', f'{qid}:new_app:{index}', qid, 'practice_application',
                messages(f'Apply {name} to this new inspection: {body}. What is the two-check recording code? Output only the code.'), code_map[pair])
            add('heldout_diagnosis', f'{qid}:new_diag:{index}', qid, 'practice_diagnosis',
                messages(f'Using {name}, fill in Check 1, Check 2, and any Check 3, followed by Qualification. '
                         'Use Pass/Fail in the rule condition order: ' + ', '.join(c['field'] for c in rule['conditions']) +
                         f'. Inspection: {body}.'), record(rule, actual))
        # Four option orders per pair: a constant letter cannot pass this control.
        for pi, pair in enumerate(itertools.product((False, True), repeat=2)):
            clauses = [f'{FIELDS[rule["conditions"][i - 1]["field"]]} check {"passed" if flag else "failed"}'
                       for i, flag in zip(protocol['checked_indices'], pair)]
            for rotation in range(4):
                codes = ['H7', 'J2', 'K9', 'M4']
                codes = codes[rotation:] + codes[:rotation]
                question = f'Under {name}, ' + ' and '.join(clauses) + '. Which recording code applies?\n\nOptions:\n'
                question += '\n'.join(f'{letter}. Result: {code}.' for letter, code in zip('ABCD', codes)) + ev.SUFFIX
                add('heldout_code_mcq', f'{qid}:code_mcq:{pi}:{rotation}', qid, 'record_code_recall',
                    messages(question, task='mcq'), 'ABCD'[codes.index(code_map[pair])], 'mcq')
    for r in rows(CONTROL / 'probes.jsonl'):
        if r['qid'] in ids and r['condition'].startswith('original_'):
            add(r['condition'], r['id'], r['qid'], 'original_mcq', copy.deepcopy(r['messages']), r['gold'], 'mcq')
    train_prompts = {ev.digest(r['messages'][:-1]) for r in training}
    assert all(ev.digest(p['messages']) not in train_prompts for p in result if p['condition'].startswith('heldout_'))
    assert len({(p['condition'], p['id']) for p in result}) == len(result)
    return result


def prepare(out, count, lr):
    if (out / 'manifest.json').exists():
        manifest = read(out / 'manifest.json')
        assert manifest['rule_count'] == count and manifest['learning_rate'] == lr
        assert sha(out / 'train.jsonl') == manifest['train_sha256']
        assert sha(out / 'probes.jsonl') == manifest['probes_sha256']
        return
    out.mkdir(parents=True, exist_ok=True)
    selected = selected_rules(DATA)
    # Fixed failed example chosen before seeing this experiment's answers.
    knowledge = [k for k in selected if k['id'] == 'scale1k_0003_ai_generated'] if count == 1 else selected
    assert len(knowledge) == count
    training, audit = build_training(knowledge)
    ev.save_rows(out / 'train.jsonl', training)
    ev.save_rows(out / 'generated_solver_checks.jsonl', audit)
    # Frozen evaluation prompts/labels are read only after all training rows are saved.
    probes = make_probes(knowledge, training)
    ev.save_rows(out / 'probes.jsonl', probes)
    save(out / 'manifest.json', {'rule_count': count, 'rules': [k['id'] for k in knowledge],
        'model': str(MODELS['qwen3']), 'epochs': 1, 'learning_rate': lr, 'gradient_accumulation_steps': 1,
        'training_rows': len(training), 'unique_training_prompts': len({ev.digest(r['messages'][:-1]) for r in training}),
        'tasks': dict(Counter(r['task'] for r in training)), 'train_sha256': sha(out / 'train.jsonl'),
        'probes_sha256': sha(out / 'probes.jsonl'), 'source_knowledge_sha256': sha(DATA / 'audit/rule_knowledge.jsonl'),
        'original_eval_source_sha256': sha(CONTROL / 'probes.jsonl'), 'script_sha256': sha(Path(__file__)),
        'limits': ['Capacity sanity check, not matched causal repair experiment.',
                   '256 training rows per rule include deliberate repeated facts/observations with wording variations.',
                   'Held-out diagnostic prompts are new; attribute combinations may overlap training.',
                   'Original questions were used in earlier diagnosis and are not a fresh independent test.'],
        'no_new_images': True, 'cot': False, 'created_at': ev.now()})


def evaluate(out, cp, gpu):
    os.environ.update(environment(gpu, DATA))
    import torch
    from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
    from swift.utils import seed_everything
    seed_everything(42)
    probes = rows(out / 'probes.jsonl')
    records = []
    with reserve(WORK, gpu, str(out) + ':evaluation', 22000) as reservation:
        reservation.bind(os.getpid())
        engine = TransformersEngine(str(MODELS['qwen3']), adapters=[str(cp)], torch_dtype=torch.bfloat16,
                                    device_map='cuda:0', max_batch_size=16, attn_impl='sdpa')
        engine.template.enable_thinking = False
        save(out / 'state.json', {'status': 'evaluating', 'pid': os.getpid(), 'checkpoint': str(cp)})
        for model_condition in ('base', 'trained'):
            from contextlib import nullcontext
            with engine.model.disable_adapter() if model_condition == 'base' else nullcontext():
                for fmt in ('mcq', 'semantic'):
                    selected = [r for r in probes if r['answer_format'] == fmt]
                    for start in range(0, len(selected), 16):
                        batch = selected[start:start + 16]
                        responses = engine.infer([InferRequest(messages=copy.deepcopy(r['messages'])) for r in batch],
                            RequestConfig(max_tokens=32 if fmt == 'mcq' else 128, temperature=0, seed=42), use_tqdm=False)
                        assert len(responses) == len(batch)
                        for p, response in zip(batch, responses):
                            if isinstance(response, Exception):
                                raise response
                            text = response.choices[0].message.content or ''
                            parsed = ev.letter(text) if fmt == 'mcq' else None
                            correct = parsed == p['gold'] if fmt == 'mcq' else semantic_equal(text, p['gold'], p['task'])
                            record_row = {k: p[k] for k in ('condition', 'id', 'qid', 'task', 'gold', 'answer_format')}
                            record_row.update(model_condition=model_condition, response=text, correct=correct,
                                format_error=parsed is None if fmt == 'mcq' else None,
                                field_match_correct=field_equal(text, p['gold']) if 'diagnosis' in p['condition'] and fmt == 'semantic' else None,
                                prompt_sha256=ev.digest(p['messages']))
                            records.append(record_row)
                        ev.save_rows(out / 'predictions.jsonl', records)
                        print(model_condition, fmt, start + len(batch), '/', len(selected), flush=True)
    metrics = {}
    for model_condition in ('base', 'trained'):
        metrics[model_condition] = {}
        for condition in sorted({p['condition'] for p in probes}):
            group = [r for r in records if r['model_condition'] == model_condition and r['condition'] == condition]
            metrics[model_condition][condition] = {'n': len(group), 'correct': sum(r['correct'] for r in group),
                'field_match_correct': sum(r['field_match_correct'] is True for r in group) if group[0]['field_match_correct'] is not None else None,
                'format_errors': sum(r['format_error'] is True for r in group) if group[0]['answer_format'] == 'mcq' else None,
                'response_distribution': dict(Counter(r['response'] for r in group))}
    save(out / 'report.json', {'status': 'complete', 'checkpoint': ev.checkpoint(cp), 'epochs': 1,
                             'metrics': metrics, 'finished_at': ev.now()})
    save(out / 'state.json', {'status': 'complete', 'finished_at': ev.now()})
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--rules', type=int, choices=[1, 12], default=1)
    ap.add_argument('--learning-rate', type=float, default=1e-4)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--prepare-only', action='store_true')
    args = ap.parse_args()
    out = args.out.resolve()
    prepare(out, args.rules, args.learning_rate)
    if args.prepare_only:
        print(json.dumps(read(out / 'manifest.json'), ensure_ascii=False, indent=2))
        return
    with ev.run_lock(out / '.lock'):
        if read(out / 'report.json').get('status') == 'complete':
            return
        cp = final_checkpoint(out / 'checkpoint', 1)
        if not cp:
            with reserve(WORK, args.gpu, str(out) + ':training', 16000) as reservation:
                cmd = train_command('qwen3', 'sft', out / 'train.jsonl', out / 'checkpoint', 1)
                cmd[cmd.index('--gradient_accumulation_steps') + 1] = '1'
                cmd[cmd.index('--learning_rate') + 1] = str(args.learning_rate)
                with (out / 'train.log').open('a') as log:
                    child = subprocess.Popen(cmd, cwd=ROOT, env=environment(args.gpu, DATA), stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    reservation.bind(child.pid)
                    save(out / 'state.json', {'status': 'training', 'pid': child.pid, 'command': cmd, 'started_at': ev.now()})
                    code = child.wait()
                if code:
                    save(out / 'state.json', {'status': 'failed', 'returncode': code})
                    raise RuntimeError(f'Training failed: {code}')
            cp = final_checkpoint(out / 'checkpoint', 1)
            assert cp
        evaluate(out, cp, args.gpu)


if __name__ == '__main__':
    main()
