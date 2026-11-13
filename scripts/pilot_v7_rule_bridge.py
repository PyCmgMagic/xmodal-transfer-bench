#!/usr/bin/env python3
"""Keep rule-fact supervision and add balanced MCQs for applying those same facts."""
import argparse
import copy
import itertools
import json
import random
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import pilot_v7_rule_mastery as mastery
import run_v7_evaluation as ev
from pilot_v7_natural_sft import FIELDS, record
from run_v7_full_experiment import ROOT, environment, final_checkpoint, read, rows, save, sha, train_command
from v7_gpu_resources import reserve


def bridge_training(knowledge, training, audits):
    rules = {k['id']: json.loads(k['fact']['quote']) for k in knowledge}
    checks = {r['id']: r for r in audits}
    groups = defaultdict(list)
    for r in training:
        if r['task'] in ('record_code_recall', 'practice_application', 'practice_diagnosis'):
            audit = checks[r['id']]
            truth = tuple(audit['pair'] if 'pair' in audit else audit['flags'])
            groups[(r['qid'], r['task'], truth)].append(r)
    selected, positions = set(), {}
    for key, group in sorted(groups.items()):
        ranked = sorted(group, key=lambda r: ev.digest('bridge:42:' + r['id']))
        assert len(group) % 2 == 0
        for index, r in enumerate(ranked[:len(group) // 2]):
            selected.add(r['id'])
            positions[r['id']] = index % 4
    result, changed = [], []
    rng = random.Random(42)
    for original in training:
        r = copy.deepcopy(original)
        r['answer_format'] = 'semantic'
        if r['id'] not in selected:
            result.append(r)
            continue
        audit, rule = checks[r['id']], rules[r['qid']]
        name = rule['predicate_name']
        old_question = r['messages'][1]['content']
        expected_content = r['messages'][-1]['content']
        if r['task'] == 'record_code_recall':
            k = next(k for k in knowledge if k['id'] == r['qid'])
            protocol = k['material']['learning']['inspection_recording_protocol']
            clauses = [f'the {FIELDS[rule["conditions"][i - 1]["field"]]} check {"passed" if f else "failed"}'
                       for i, f in zip(protocol['checked_indices'], audit['pair'])]
            question = f'Under {name}, ' + ' and '.join(clauses) + '. Which inspection record code should be entered?'
            answer = 'Result: ' + expected_content + '.'
            choices = ['Result: ' + code + '.' for code in ('H7', 'J2', 'K9', 'M4')]
        elif r['task'] == 'practice_application':
            question = old_question.replace('Return only the record code', 'Which inspection record code should be entered')
            question = question.replace('requirements; assess them independently.', 'requirements? Evaluate each requirement independently.')
            question = question.replace('\nObservation: ', '\n\nTest input:\n')
            answer = 'Result: ' + expected_content + '.'
            choices = ['Result: ' + code + '.' for code in ('H7', 'J2', 'K9', 'M4')]
        else:
            submitted_part = old_question.split('. Return the correct Check statuses', 1)[0]
            submitted = submitted_part.split(f'A technician submitted this report under {name}: ', 1)[1]
            question = (submitted_part + '. Check the observations and the rule. Check the conditions in this order: ' +
                        ', '.join(c['field'] for c in rule['conditions']) +
                        '. Report Pass/Fail for each condition and the final qualification.\n\nTest input:\n' +
                        old_question.split('\nObservation: ', 1)[1])
            all_records = [record(rule, flags) for flags in itertools.product((False, True), repeat=len(rule['conditions']))]
            wrong_records = [v for v in all_records if v != expected_content]
            rng.shuffle(wrong_records)
            contents = [expected_content, *wrong_records[:3]]
            decorate = lambda value: ('Accept report. Result: ' if value == submitted else 'Replace report. Result: ') + value + '.'
            choices = [decorate(value) for value in contents]
            answer = decorate(expected_content)
        wrong = [value for value in choices if value != answer]
        rng.shuffle(wrong)
        position = positions[r['id']]
        wrong.insert(position, answer)
        question += '\n\nOptions:\n' + '\n'.join(f'{letter}. {value}' for letter, value in zip('ABCD', wrong)) + ev.SUFFIX
        r.update(answer_format='mcq', messages=mastery.messages(question, 'ABCD'[position], 'mcq'))
        result.append(r)
        changed.append({'id': r['id'], 'qid': r['qid'], 'task': r['task'],
                        'correct_content': expected_content, 'gold': 'ABCD'[position], 'options': dict(zip('ABCD', wrong))})
    assert len(changed) == 96 * len(knowledge)
    assert Counter(r['gold'] for r in changed) == {letter: 24 * len(knowledge) for letter in 'ABCD'}
    return result, changed


def prepare(out, count, batch_size):
    if (out / 'manifest.json').exists():
        m = read(out / 'manifest.json')
        assert m['rule_count'] == count and m['batch_size'] == batch_size
        for name in ('train', 'probes'):
            assert sha(out / (name + '.jsonl')) == m[name + '_sha256']
        return
    out.mkdir(parents=True, exist_ok=True)
    knowledge = mastery.selected_rules(mastery.DATA)
    if count == 1:
        knowledge = [k for k in knowledge if k['id'] == 'scale1k_0003_ai_generated']
    semantic, audits = mastery.build_training(knowledge)
    training, changed = bridge_training(knowledge, semantic, audits)
    ev.save_rows(out / 'train.jsonl', training)
    ev.save_rows(out / 'generated_solver_checks.jsonl', audits)
    ev.save_rows(out / 'mcq_checks.jsonl', changed)
    probes = mastery.make_probes(knowledge, semantic)
    exposure = {r['id']: r['answer_format'] for r in training}
    for p in probes:
        p['training_format'] = exposure.get(p['id'], 'new_prompt')
    ev.save_rows(out / 'probes.jsonl', probes)
    save(out / 'manifest.json', {'rule_count': count, 'rules': [k['id'] for k in knowledge],
        'model': str(mastery.MODELS['qwen3']), 'epochs': 1, 'learning_rate': 1e-4, 'batch_size': batch_size,
        'gradient_accumulation_steps': 1, 'training_rows': len(training),
        'unique_training_prompts': len({ev.digest(r['messages'][:-1]) for r in training}),
        'formats': dict(Counter(r['answer_format'] for r in training)),
        'train_sha256': sha(out / 'train.jsonl'), 'probes_sha256': sha(out / 'probes.jsonl'),
        'knowledge_sha256': sha(mastery.DATA / 'audit/rule_knowledge.jsonl'),
        'script_sha256': sha(Path(__file__)), 'mastery_script_sha256': sha(Path(mastery.__file__)),
        'training_changes': 'Keep all factual/card rows and half of each code/application/diagnosis truth pattern semantic; half become balanced MCQs.',
        'limits': ['Exploratory capacity/transfer pilot; not an isolated causal comparison with historical 12-rule runs.',
                   'Fixed semantic controls are exact training prompts only when training_format=semantic.',
                   '256 rows per rule intentionally repeat facts/observations; one epoch is over this expanded corpus.',
                   'Original pilot questions have been inspected before and are development data.'],
        'cot': False, 'new_images': False, 'created_at': ev.now()})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--rules', type=int, choices=[1, 12], default=1)
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--prepare-only', action='store_true')
    a = ap.parse_args()
    assert a.batch_size > 0
    out = a.out.resolve()
    prepare(out, a.rules, a.batch_size)
    if a.prepare_only:
        print(json.dumps(read(out / 'manifest.json'), ensure_ascii=False, indent=2))
        return
    with ev.run_lock(out / '.lock'):
        if read(out / 'report.json').get('status') == 'complete':
            return
        cp = final_checkpoint(out / 'checkpoint', 1)
        if not cp:
            with reserve(mastery.WORK, a.gpu, str(out) + ':training', 24000 if a.batch_size > 1 else 16000) as reservation:
                cmd = train_command('qwen3', 'sft', out / 'train.jsonl', out / 'checkpoint', 1)
                for flag, value in [('--gradient_accumulation_steps', '1'), ('--learning_rate', '1e-4'),
                                    ('--per_device_train_batch_size', str(a.batch_size))]:
                    cmd[cmd.index(flag) + 1] = value
                with (out / 'train.log').open('a') as log:
                    child = subprocess.Popen(cmd, cwd=ROOT, env=environment(a.gpu, mastery.DATA), stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    reservation.bind(child.pid)
                    save(out / 'state.json', {'status': 'training', 'pid': child.pid, 'command': cmd, 'started_at': ev.now()})
                    code = child.wait()
                if code:
                    save(out / 'state.json', {'status': 'failed', 'returncode': code})
                    raise RuntimeError(f'Bridge training failed: {code}')
            cp = final_checkpoint(out / 'checkpoint', 1)
            assert cp
        mastery.evaluate(out, cp, a.gpu)


if __name__ == '__main__':
    main()
