"""Verify mixed-supervision pilot inputs and conservative diagnostic scoring."""
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_v7_semantic_pilot as runner
from run_v7_full_experiment import rows, sha


class SemanticPilot(unittest.TestCase):
    def test_same_fixed_questions_and_mixed_training_formats(self):
        source = ROOT / 'runs/v7_natural_1epoch_20261014/semantic_target_candidate'
        control = ROOT / 'runs/v7_natural_1epoch_20261014/natural'
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            runner.prepare(source, control, out)
            self.assertEqual(sha(out / 'train.jsonl'), sha(source / 'train.jsonl'))
            self.assertEqual(sha(out / 'eval.jsonl'), sha(control / 'eval.jsonl'))
        probes = runner.build_probes(source, control)
        self.assertEqual(len(probes), 240)
        self.assertEqual(Counter(r['answer_format'] for r in probes), {'mcq': 144, 'semantic': 96})
        expected = rows(control / 'probes.jsonl')
        self.assertEqual([(r['condition'], r['id'], r['messages'], r['gold']) for r in probes[:144]],
                         [(r['condition'], r['id'], r['messages'], r['gold']) for r in expected])
        training = {r['id']: r for r in rows(source / 'train.jsonl')}
        for r in probes[144:]:
            self.assertEqual(r['messages'], training[r['id']]['messages'][:-1])
            self.assertEqual(r['gold'], training[r['id']]['messages'][-1]['content'])
        self.assertEqual(len({r['qid'] for r in probes}), 12)

    def test_semantic_scoring_does_not_extract_codes_from_prose(self):
        self.assertTrue(runner.semantic_equal('J2', 'J2', 'record_code_recall'))
        self.assertFalse(runner.semantic_equal('The code is J2.', 'J2', 'record_code_recall'))
        self.assertFalse(runner.semantic_equal('J2 or M4', 'J2', 'record_code_recall'))
        self.assertTrue(runner.semantic_equal('The main-body color must be blue',
                                            'The main-body color must be blue.', 'atomic_condition'))
        self.assertFalse(runner.semantic_equal('The main-body color must not be blue.',
                                             'The main-body color must be blue.', 'atomic_condition'))

    def test_first_generated_token_is_separate_from_format_tokens(self):
        self.assertEqual(runner.first_token({'content': [{'token': 'B'}, {'token': '<|im_end|>'}]}), 'B')
        self.assertIsNone(runner.first_token(None))


if __name__ == '__main__':
    unittest.main()
