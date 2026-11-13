import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_rules_glm_temperature_pilot import matched_baseline, metrics, repeated_sentence, sample_tasks
from entityqa_common import digest


class TemperaturePilotTests(unittest.TestCase):
    def test_rule_selection_is_paired_and_independent_of_answer_outcomes(self):
        tasks = []
        for i in range(1000):
            for kind in ['direct', 'diagnosis']:
                for mod in 'TV':
                    tasks.append(dict(qid=str(i), id=f'{i}-{kind}', modality=mod,
                        question_type=kind, gold='ABCD'[i % 4], options={}, rule_sha256=str(i),
                        payload=dict(messages=[dict(role='user', content=f'Question {i}')]),
                        correct=i % 2 == 0))
        original = copy.deepcopy(tasks)
        chosen = sample_tasks(tasks)
        other = [dict(t, correct=not t['correct'], gold='D') for t in tasks]
        changed = sample_tasks(other)
        self.assertEqual(len(chosen), 200)
        self.assertEqual({r['qid'] for r in chosen}, {r['qid'] for r in changed})
        self.assertEqual(tasks, original)

    def test_same_question_baseline_cannot_substitute_other_input(self):
        task = dict(id='q', modality='T', gold='B', options={})
        payload = dict(messages=[dict(role='user', content='Question')])
        baseline = dict(id='q', modality='T', gold='B', predicted='B',
                        raw_response=r'\boxed{B}', input_sha256=digest(payload))
        result = matched_baseline([task], [payload], [baseline])
        self.assertEqual(result[0], baseline)
        with self.assertRaises(AssertionError):
            matched_baseline([task], [dict(messages=[])], [baseline])

    def test_repetition_metric_and_invalid_denominator(self):
        sentence = 'The reference collection is always unchanged.'
        self.assertFalse(repeated_sentence(sentence * 1))
        self.assertFalse(repeated_sentence(' '.join([sentence] * 2)))
        self.assertTrue(repeated_sentence(' '.join([sentence] * 3)))
        records = [dict(modality='T', question_type='direct', predicted=p, gold='A',
                        stop_reason='length' if p is None else 'stop', parse_reason='test',
                        raw_response=' '.join([sentence] * 3) if p is None else r'\boxed{A}')
                   for p in ['A', None]]
        group = metrics(records)['T/all']
        self.assertEqual(group['accuracy'], 0.5)
        self.assertEqual(group['invalid_rate'], 0.5)
        self.assertEqual(group['sentence_repeat_3plus_rate'], 0.5)

    def test_user_stopped_baseline_never_fabricates_missing_answers(self):
        payload = dict(messages=[dict(role='user', content='Question')])
        tasks = [dict(id=q, modality='T', gold='B', options={}) for q in ['saved', 'missing']]
        record = dict(id='saved', modality='T', gold='B', predicted='B',
                      raw_response=r'\boxed{B}', input_sha256=digest(payload))
        result = matched_baseline(tasks, [payload, payload], [record], allow_missing=True)
        self.assertEqual([r['id'] for r in result], ['saved'])
        with self.assertRaises(KeyError):
            matched_baseline(tasks, [payload, payload], [record])


if __name__ == '__main__':
    unittest.main()
