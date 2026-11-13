"""Independently check generated learning targets against stored rule predicates."""
import itertools
import json
import operator
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import pilot_v7_rule_mastery as pilot
import pilot_v7_rule_bridge as bridge


class RuleMastery(unittest.TestCase):
    def test_mcq_bridge_preserves_semantic_target_and_balances_positions(self):
        knowledge = pilot.selected_rules(pilot.DATA)
        training, audits = pilot.build_training(knowledge)
        mixed, changed = bridge.bridge_training(knowledge, training, audits)
        originals = {r['id']: r for r in training}
        modified = {r['id']: r for r in mixed}
        self.assertEqual(len(mixed), len(training))
        self.assertEqual(len(changed), 96 * 12)
        for check in changed:
            original = originals[check['id']]
            current = modified[check['id']]
            content = original['messages'][-1]['content']
            selected = check['options'][current['messages'][-1]['content']]
            self.assertEqual(check['correct_content'], content)
            self.assertTrue(selected.endswith('Result: ' + content + '.'))
            self.assertEqual(len(set(check['options'].values())), 4)
            if check['task'] == 'practice_diagnosis':
                prompt = current['messages'][1]['content']
                submitted = prompt.split(': ', 1)[1].split('. Check the observations', 1)[0]
                self.assertEqual(selected.startswith('Accept report.'), submitted == content)
        for k in knowledge:
            group = [r for r in changed if r['qid'] == k['id']]
            self.assertEqual([sum(r['gold'] == letter for r in group) for letter in 'ABCD'], [24] * 4)

    def test_all_twelve_rule_targets_and_probe_separation(self):
        knowledge = pilot.selected_rules(pilot.DATA)
        training, audit = pilot.build_training(knowledge)
        self.assertEqual(len(training), 3072)
        records = {r['id']: r for r in training}
        source = {k['id']: k for k in knowledge}
        predicates = {'eq': operator.eq, 'ne': operator.ne, 'gt': operator.gt,
                      'ge': operator.ge, 'lt': operator.lt, 'le': operator.le}
        for row in audit:
            k = source[row['qid']]
            rule = json.loads(k['fact']['quote'])
            protocol = k['material']['learning']['inspection_recording_protocol']
            codebook = {tuple(part.rsplit('=', 1)[-1] == 'Met' for part in label.split('; ')): code
                        for label, code in protocol['outcome_to_code'].items()}
            answer = records[row['id']]['messages'][-1]['content']
            self.assertEqual(row['answer'], answer)
            if row['task'] == 'rule_card':
                self.assertEqual(answer, k['material']['learning']['text'])
            if row['task'] == 'record_code_recall':
                self.assertEqual(answer, codebook[tuple(row['pair'])])
            if 'attributes' in row:
                actual = [predicates[c['operator']](row['attributes'][c['field']], c['value'])
                          for c in rule['conditions']]
                self.assertEqual(actual, row['flags'])
                if row['task'] == 'practice_application':
                    pair = tuple(actual[i - 1] for i in protocol['checked_indices'])
                    self.assertEqual(answer, codebook[pair])
                else:
                    qualification = all(actual) if rule['mode'] == 'all' else any(actual)
                    expected = '; '.join(f'Check {i+1}={"Pass" if f else "Fail"}' for i, f in enumerate(actual))
                    expected += '; Qualification=' + ('Pass' if qualification else 'Fail')
                    self.assertEqual(answer, expected)
        probes = pilot.make_probes(knowledge, training)
        training_prompts = {pilot.ev.digest(r['messages'][:-1]) for r in training}
        for p in probes:
            if p['condition'].startswith('heldout_'):
                self.assertNotIn(pilot.ev.digest(p['messages']), training_prompts)
        choice_probes = [p for p in probes if p['condition'] == 'heldout_code_mcq']
        self.assertEqual(len(choice_probes), 16 * 12)
        for qid in source:
            group = [p for p in choice_probes if p['qid'] == qid]
            self.assertEqual([sum(p['gold'] == letter for p in group) for letter in 'ABCD'], [4, 4, 4, 4])


if __name__ == '__main__':
    unittest.main()
