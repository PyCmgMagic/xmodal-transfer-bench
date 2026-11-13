import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_rules_glm46_readout import reparse


class RulesReadoutAuditTests(unittest.TestCase):
    def setUp(self):
        self.options = {'A': 'Result: A=KEEP; B=OMIT.', 'B': 'Result: A=OMIT; B=KEEP.',
                        'C': 'Result: A=KEEP; B=KEEP.', 'D': 'Result: A=OMIT; B=OMIT.'}

    def test_glm_duplicate_close_with_terminal_answer(self):
        cases = [r'<think></think>Reasoning.</think>\boxed{D}',
                 '<think></think><|begin_of_box|>Reasoning.</think><|begin_of_box|>D<|end_of_box|>']
        for text in cases:
            self.assertEqual(reparse(text, self.options)[0], 'D')

    def test_never_accept_unfinished_or_conflicting_thought(self):
        for text in [r'<think>Reasoning \boxed{A}', r'<think></think><think>\boxed{A}',
                     r'<think></think>Reasoning.</think>', r'\boxed{A} or \boxed{B}']:
            self.assertIsNone(reparse(text, self.options)[0])

    def test_full_option_in_latex_or_native_box(self):
        for text in [r'\boxed{A=KEEP; B=OMIT}', r'\boxed{A. Result: A=KEEP; B=OMIT.}',
                     '<|begin_of_box|>A. Result: A=KEEP; B=OMIT.<|end_of_box|>']:
            self.assertEqual(reparse(text, self.options)[0], 'A')

    def test_candidate_letters_and_disagreeing_prefix_stay_invalid(self):
        for text in [r'\boxed{A,C}', r'\boxed{B. Result: A=KEEP; B=OMIT.}',
                     r'\boxed{A=KEEP; B=MAYBE}', r'\boxed{None of the above}',
                     r'If needed, \boxed{A=KEEP; B=OMIT}']:
            self.assertIsNone(reparse(text, self.options)[0])

    def test_nonunique_result_remains_invalid(self):
        options = dict(self.options, B=self.options['A'])
        self.assertIsNone(reparse(r'\boxed{A=KEEP; B=OMIT}', options)[0])

    def test_result_comparison_preserves_signs_and_operators(self):
        options = {'A': 'Result: X=-3.', 'B': 'Result: X=3.',
                   'C': 'Result: X<3.', 'D': 'Result: X>3.'}
        for body, expected in [('X=-3', 'A'), ('X=3', 'B'), ('X<3', 'C'), ('X>3', 'D')]:
            self.assertEqual(reparse('\\boxed{' + body + '}', options)[0], expected)

    def test_known_valid_answers_unchanged(self):
        self.assertEqual(reparse(r'<think></think>\boxed{C}', self.options)[0], 'C')


if __name__ == '__main__':
    unittest.main()
