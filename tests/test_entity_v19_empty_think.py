import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from rescore_entity_v19_empty_think import parse


class FormatNormalizationTests(unittest.TestCase):
    def test_exact_glm_native_forms(self):
        for text, expected in [('A', 'A'), ('<think></think>\nB', 'B'),
             ('<think> \n </think> C ', 'C'), ('<|begin_of_box|>D<|end_of_box|>', 'D'),
             ('<think></think>\n<|begin_of_box|>A<|end_of_box|>', 'A')]:
            self.assertEqual(parse(text, 'glm46'), expected)

    def test_no_reasoning_or_prose_extraction(self):
        for text in ['<think>I think A</think>B', '<think></think>\nThe answer is A',
                     '<think></think>\nA.', '<think></think>\nA B',
                     '<think></think><think></think>A', 'A\nExplanation', '',
                     '<think></think>\n<|begin_of_box|>A', 'a']:
            self.assertIsNone(parse(text, 'glm46'))

    def test_qwen_remains_strict(self):
        self.assertEqual(parse(' B\n', 'qwen3'), 'B')
        self.assertIsNone(parse('<think></think>B', 'qwen3'))
        self.assertIsNone(parse('<|begin_of_box|>B<|end_of_box|>', 'qwen3'))


if __name__ == '__main__':
    unittest.main()
