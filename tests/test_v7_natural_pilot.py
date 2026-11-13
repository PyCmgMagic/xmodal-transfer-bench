"""Independently validate natural observations, answer labels and pilot isolation."""
import json
import re
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import pilot_v7_natural_sft as pilot


def outcome(attrs, condition):
    a, b = attrs[condition['field']], condition['value']
    op = condition['operator']
    if op == 'eq':
        return str(a) == str(b)
    if op == 'ne':
        return str(a) != str(b)
    a, b = Fraction(str(a)), Fraction(str(b))
    return {'ge': a >= b, 'gt': a > b, 'le': a <= b, 'lt': a < b}[op]


class NaturalPilot(unittest.TestCase):
    def test_all_generated_targets_with_independent_solver(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with patch.object(pilot, 'rows', wraps=pilot.rows) as reader:
                pilot.build_natural(ROOT / 'rules_repaired_single_image_v7', out)
                self.assertTrue(all(call.args[0].name == 'rule_knowledge.jsonl' for call in reader.call_args_list))
            training = pilot.rows(out / 'train.jsonl')
            checks = pilot.rows(out / 'generated_solver_checks.jsonl')
            self.assertEqual(len(training), 780)
            self.assertEqual(len(checks), 768)
            self.assertEqual(Counter(r['gold'] for r in checks), Counter({a: 192 for a in 'ABCD'}))
            self.assertFalse(any('a different ' in r['messages'][1]['content'] for r in training))
            self.assertFalse(any(r.get('images') for r in training))
            self.assertEqual(len({r['id'] for r in training}), 780)
            by_id = {r['id']: r for r in training}
            pair_letters = defaultdict(Counter)
            for check in checks:
                prompt = by_id[check['id']]['messages'][1]['content']
                self.assertEqual(dict(re.findall(r'(?m)^([A-D])\. (.*)$', prompt)), check['options'])
                self.assertEqual(by_id[check['id']]['messages'][-1]['content'], check['gold'])
                selected = check['options'][check['gold']]
                conditions, protocol = check['rule']['conditions'], check['protocol']
                if check['task'] == 'atomic_condition':
                    self.assertIn(check['condition'], conditions)
                    self.assertEqual(selected, pilot.condition_text(check['condition']))
                    continue
                if check['task'] == 'record_code_recall':
                    pair = tuple(check['pair'])
                    pair_letters[pair][check['gold']] += 1
                else:
                    flags = [outcome(check['attributes'], c) for c in conditions]
                    self.assertEqual(flags, check['flags'])
                    # The rendered observation must actually state the values
                    # used by the solver, including count and focus candidate.
                    for field, value in check['attributes'].items():
                        if field == 'value':
                            self.assertIn(f'exactly {value} complete', check['observation'])
                        elif field != 'status':
                            self.assertIn(str(value), check['observation'])
                    if check['task'] == 'practice_diagnosis':
                        qualified = all(flags) if check['rule']['mode'] == 'all' else any(flags)
                        result = '; '.join(f'Check {i+1}={"Pass" if flag else "Fail"}' for i, flag in enumerate(flags))
                        result += f'; Qualification={"Pass" if qualified else "Fail"}'
                        action = 'Accept' if check['submitted'] == result else 'Replace'
                        self.assertEqual(selected, action + ' report. Result: ' + result + '.')
                        continue
                    pair = tuple(flags[i - 1] for i in protocol['checked_indices'])
                expected = next(code for text, code in protocol['outcome_to_code'].items()
                                if tuple(p.endswith('=Met') for p in text.split('; ')) == pair)
                self.assertEqual(selected, 'Result: ' + expected + '.')
            self.assertTrue(all(set(counts) == set('ABCD') for counts in pair_letters.values()))
            diagnosis = [r for r in checks if r['task'] == 'practice_diagnosis']
            self.assertEqual(sum(r['options'][r['gold']].startswith('Accept') for r in diagnosis), 72)
            self.assertEqual(len(diagnosis), 288)

    def test_new_runs_default_to_one_epoch(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            pilot.prepare(ROOT / 'rules_repaired_single_image_v7', ROOT / 'runs/v7_full_experiment', out, 'natural', 1)
            manifest = pilot.read(out / 'manifest.json')
            self.assertEqual(manifest['epochs'], 1)
            self.assertEqual(len(pilot.probes(out, ROOT / 'rules_repaired_single_image_v7')), 144)
            training, evaluation = pilot.rows(out / 'train.jsonl'), pilot.rows(out / 'eval.jsonl')
            self.assertFalse({r['id'] for r in training} & {r['id'] for r in evaluation})
            self.assertEqual({r['qid'] for r in training}, {r['qid'] for r in evaluation})


if __name__ == '__main__':
    unittest.main()
