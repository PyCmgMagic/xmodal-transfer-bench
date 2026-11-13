import itertools
import json
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import pilot_v7_grounded_families as pilot
from diagnose_v7_grounded_references import balanced_probes


class Solvers(unittest.TestCase):
    def test_set_logic_all_truth_patterns(self):
        for mode in ('all', 'any'):
            rule = {'mode': mode, 'conditions': [
                {'field': 'category', 'operator': 'eq', 'value': 'bookend'},
                {'field': 'material', 'operator': 'eq', 'value': 'metal'},
                {'field': 'color', 'operator': 'eq', 'value': 'black'}]}
            for flags in itertools.product((False, True), repeat=3):
                observation = {c['field']: c['value'] if match else 'other'
                               for c, match in zip(rule['conditions'], flags)}
                expected = 'KEEP' if (all(flags) if mode == 'all' else any(flags)) else 'OMIT'
                self.assertEqual(pilot.solve(rule, observation), expected)
                self.assertEqual(pilot.check_independently(rule, observation), expected)

    def test_ratio_boundaries_and_integer_range(self):
        rule = {'thresholds': {'lower': .1, 'upper': .7}, 'category_labels': ['low', 'mid', 'high']}
        for observation, expected in [((1, 20), 'low'), ((2, 20), 'mid'), ((14, 20), 'mid'), ((15, 20), 'high')]:
            self.assertEqual(pilot.solve(rule, observation), expected)
            self.assertEqual(pilot.check_independently(rule, observation), expected)
        rule = {'mode': 'all', 'conditions': [
            {'field': 'value', 'operator': 'ge', 'value': 3},
            {'field': 'value', 'operator': 'le', 'value': 5}]}
        self.assertEqual([pilot.solve(rule, {'value': n}) for n in range(2, 7)], ['OMIT', 'KEEP', 'KEEP', 'KEEP', 'OMIT'])

    def test_generated_mcq_labels_and_split(self):
        knowledge = [r['content'] for r in pilot.rows(pilot.DATA / 'audit/rule_knowledge.jsonl')]
        selected = [next(k for k in knowledge if k['id'] == qid) for qid in
                    ('scale1k_0704_ai_generated', 'scale1k_0664_ai_generated', 'scale1k_0647_ai_generated', 'scale1k_0723_ai_generated')]
        for k in selected:
            rule = json.loads(k['fact']['quote'])
            if k['id'] == 'scale1k_0664_ai_generated':
                self.assertFalse(any('interval' in q for q, a in pilot.atoms(rule)))
                self.assertEqual(pilot.atoms(rule)[-1][1], '2')
            train, train_checks = pilot.make_examples(k, 'train')
            valid, checks = pilot.make_examples(k, 'valid')
            train_vectors = {pilot.ev.digest(r['observations']) for r in train_checks if 'observations' in r}
            self.assertFalse(train_vectors.intersection(pilot.ev.digest(r['observations']) for r in checks if 'observations' in r))
            for row in train_checks + checks:
                if 'options' not in row:
                    continue
                answer = row['options']['ABCD'.index(row['answer'])]
                if 'observations' in row:
                    values = [pilot.check_independently(rule, o) for o in row['observations']]
                    self.assertEqual(answer, 'Result: ' + pilot.assignment(values) + '.')
                else:
                    self.assertEqual(answer, pilot.check_independently(rule, row['observation']))
            for audit in (train_checks, checks):
                group = [r for r in audit if r['task'] == 'diagnosis_mcq']
                self.assertEqual(sum(r['accepted'] for r in group), 2)
                for r in group:
                    self.assertIn('Result: ' + r['submitted'] + '.', r['options'])

    def test_balanced_reports_and_resumable_serialization(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            pilot.save(out / 'manifest.json', {'rules': ['scale1k_0704_ai_generated', 'scale1k_0723_ai_generated']})
            probes = balanced_probes(out)
            self.assertEqual(probes, balanced_probes(out))
            audit = pilot.rows(out / 'balanced_diagnosis_audit.jsonl')
            for qid in {a['qid'] for a in audit}:
                group = [a for a in audit if a['qid'] == qid]
                for letter in 'ABCD':
                    subset = [a for a in group if a['gold'] == letter]
                    self.assertEqual(len(subset), 4)
                    self.assertEqual(sum(a['accepted'] for a in subset), 1)
                copied_correct = 0
                for a in group:
                    index = a['options'].index('Result: ' + a['submitted'] + '.')
                    copied_correct += 'ABCD'[index] == a['gold']
                self.assertEqual(copied_correct, 4)


if __name__ == '__main__':
    unittest.main()
