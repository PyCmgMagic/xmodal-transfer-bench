"""Independently check sampled questions and saved CoT outputs."""
import copy
import json
import operator
import re
from collections import Counter
from pathlib import Path

import eval_v7_cot100 as experiment
import run_v7_evaluation as ev
from run_v7_full_experiment import ROOT, read, rows, save, sha

OPS = dict(eq=operator.eq, ne=operator.ne, ge=operator.ge, gt=operator.gt,
           le=operator.le, lt=operator.lt)


def main():
    manifest = experiment.validate()
    knowledge = {k['id']: k for k in rows(experiment.OUT / 'knowledge.jsonl')}
    questions = rows(experiment.OUT / 'questions.jsonl')
    sources = {mod: {r['id']: r for r in rows(experiment.DATA / direction / 'eval.jsonl')}
               for mod, direction in [('T', 'T2T'), ('V', 'T2V')]}
    labels = {r['id']: r for r in rows(experiment.DATA / 'audit/eval_labels.jsonl')}
    inspected = []
    for p in questions:
        original = sources[p['modality']][p['source_id']]
        original_user = original['messages'][-1]['content']
        user = p['messages'][-1]['content']
        assert original_user.endswith(ev.SUFFIX)
        assert user == original_user[:-len(ev.SUFFIX)] + experiment.SUFFIX
        assert p['gold'] == labels[p['source_id']]['answer']
        if p['modality'] == 'V':
            assert p['images'] == [str(experiment.DATA / image) for image in original['images']]
            continue
        rule = json.loads(knowledge[p['qid']]['fact']['quote'])
        # Reparse observations from the original text, independently of source probe metadata.
        scene = user.split('Test input:\n', 1)[1].split('\n\nOptions:', 1)[0]
        observations = []
        for candidate, fields in re.findall(r'\b([ABCD]) has (.*?)(?=; [ABCD] has |\.\s*$)', scene):
            data = dict(re.findall(r'(\w+): ([^,;]+)', fields))
            assert candidate == 'ABCD'[len(observations)]
            if 'value' in data:
                data['value'] = float(data['value'])
            observations.append(data)
        assert len(observations) == 4, p['id']
        assert observations == p['observations'], p['id']
        results = []
        for letter, observation in zip('ABCD', observations):
            flags = [OPS[c['operator']](observation[c['field']], c['value']) for c in rule['conditions']]
            kept = any(flags) if rule['mode'] == 'any' else all(flags)
            results.append(letter + '=' + ('KEEP' if kept else 'OMIT'))
        assignment = '; '.join(results)
        if p['task'] == 'diagnosis':
            submitted = user.split('report under ', 1)[1].split('Result: ', 1)[1].split('. Check', 1)[0]
            expected = ('Accept report. ' if submitted == assignment else 'Replace report. ') + 'Result: ' + assignment + '.'
        else:
            expected = 'Result: ' + assignment + '.'
        options = dict(re.findall(r'(?m)^([ABCD])\. (.+)$', user))
        matching = [letter for letter, value in options.items() if value == expected]
        assert matching == [p['gold']], (p['id'], expected, options, p['gold'])
        inspected.append(dict(id=p['id'], qid=p['qid'], independently_computed=expected, gold=p['gold']))
    assert len(inspected) == 100
    training = rows(experiment.SOURCE / 'definitions_training.jsonl')
    assert {p['qid'] for p in questions} <= {r['qid'] for r in training}
    assert Counter(r['channel'] for r in training) == {
        'full_fact': 3072, 'atomic_fact': 1536, 'boundary_logic': 512}
    assert all('Test input:' not in json.dumps(r['messages']) and 'Options:' not in json.dumps(r['messages'])
               for r in training)
    save(experiment.OUT / 'data_audit.json', dict(status='pass', checked_at=ev.now(),
         questions_per_modality=100, rules=50, independent_text_labels_checked=len(inspected),
         original_prompts_preserved_except_generation_instructions=True, images_verified=len(manifest['images']),
         all_evaluated_rules_trained=True, training_channels=dict(Counter(r['channel'] for r in training)),
         no_application_mcqs_in_definition_training=True,
         counts=dict(Counter(p['modality'] + ':' + p['task'] for p in questions)),
         label_checks=inspected))
    output_audits = {}
    for arm in ('base', 'trained'):
        folder = experiment.OUT / arm
        path = folder / 'predictions.jsonl'
        if not path.exists():
            continue
        predictions, notes = ev.read_rows(path, allow_partial_tail=True)
        lookup = {p['id']: p for p in questions + rows(experiment.OUT / 'recall_probes.jsonl')}
        memories = [r for r in predictions if r['protocol'] == 'recall']
        if len(memories) == 50:
            lookup.update({p['id']: p for p in experiment.applications(predictions)})
        assert len(predictions) == len({r['id'] for r in predictions})
        for r in predictions:
            p = lookup[r['id']]
            assert r['arm'] == arm and r['qid'] == p['qid']
            assert r['input_sha256'] == ev.digest(dict(messages=p['messages'], images=p.get('images', [])))
            raw = r['raw_response']['choices'][0]['message']
            assert r['answer_content'] == (raw['content'] or '')
            assert r['reasoning_content'] == (raw.get('reasoning_content') or '')
            response = ('<think>' + r['reasoning_content'] + '</think>\n' if r['reasoning_content'] else '') + r['answer_content']
            assert r['response'] == response
            if p['protocol'] != 'recall':
                assert r['gold'] == p['gold']
                assert r['predicted'] == experiment.final_letter(response)
                assert r['correct'] == (r['predicted'] == p['gold'])
            if p['protocol'] == 'self_recall_cot':
                assert r['recall_response_sha256'] == p['recall_response_sha256']
                assert r['recall_truncated'] == p['recall_truncated']
        output_audits[arm] = dict(audited=len(predictions), complete=len(predictions) == 450,
                                 partial_tail_notes=notes, sha256=sha(path))
    save(experiment.OUT / 'prediction_audit.json', dict(checked_at=ev.now(), arms=output_audits))
    # Reuse only already-finished short-answer cells on the identical question IDs.
    source_probes = {p['id']: p for p in rows(experiment.SOURCE / 'probes.jsonl')}
    short_results, short_records = {}, []
    for arm, source_folder in [('base', experiment.SOURCE / 'base/evaluation'),
                               ('trained', experiment.SOURCE / 'seed42/definitions/evaluation')]:
        recorded = read(source_folder / 'eval_manifest.json')
        assert recorded['model'] == manifest['model' if arm == 'base' else 'adapter']
        historical, notes = ev.read_rows(source_folder / 'predictions.jsonl', allow_partial_tail=True)
        qmap = {p['id'].removeprefix('direct_cot:'): p for p in questions}
        matched = [r for r in historical if r['id'] in qmap]
        assert len(matched) == len({r['id'] for r in matched}) == 200
        for r in matched:
            p = source_probes[r['id']]
            assert r['input_sha256'] == ev.digest(dict(messages=p['messages'], images=p.get('images', [])))
            assert r['gold'] == qmap[r['id']]['gold']
            assert r['predicted'] == ev.letter(r['response'])
            assert r['correct'] == (r['predicted'] == r['gold'])
            short_records.append(dict(arm=arm, **r))
        short_results[arm] = dict(generation=recorded['generation'], source=str(source_folder),
            metrics={mod: dict(n=sum(r['input_mod'] == mod for r in matched),
                correct=sum(r['correct'] for r in matched if r['input_mod'] == mod),
                accuracy=sum(r['correct'] for r in matched if r['input_mod'] == mod)/100,
                invalid_final=sum(r['predicted'] is None for r in matched if r['input_mod'] == mod)) for mod in ('T', 'V')},
            selected_records_sha256=ev.digest(matched))
    ev.save_rows(experiment.OUT / 'matched_historical_short_predictions.jsonl', short_records)
    save(experiment.OUT / 'matched_historical_short_report.json', dict(status='complete',
        note='Existing short-answer outputs on the same 100 question IDs per modality and same checkpoints; separate from the new CoT protocol.',
        results=short_results))
    print(json.dumps(dict(data='pass', prediction_audits=output_audits), ensure_ascii=False))


if __name__ == '__main__':
    main()
