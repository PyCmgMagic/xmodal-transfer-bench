"""Regression checks for source coverage and gold-independent entity readout."""
import unittest
from pathlib import Path

from answer_readout_entity_v2 import parse_answer
from prepare_entity_fullfact import examples


class FullFactChecks(unittest.TestCase):
    def test_date_is_supervised_in_both_variants_and_modalities(self):
        record = dict(id='unit', qid='Q1', name='Example Person',
                      reference='the person shown in the image',
                      learning='On 31 July 2026, Example Person joined Example Club.',
                      visual_learning='On 31 July 2026, the person shown in the image joined Example Club.',
                      images=['image.jpg'])
        qa = dict(question_text='On 31 July 2026, which club did Example Person join?',
                  question_vision='On 31 July 2026, which club did the person shown in the image join?')
        for modality, key in [('text', 'learning'), ('vision', 'visual_learning')]:
            result = examples(record, qa, modality, Path('/tmp/fixture'))
            self.assertEqual(len(result), 2)
            self.assertNotIn('31 July 2026', result[0]['messages'][0]['content'])
            for item in result:
                self.assertEqual(item['messages'][1]['content'], record[key])
                self.assertIn('31 July 2026', item['messages'][1]['content'])
            if modality == 'vision':
                self.assertNotIn('Example Person', str(result))

    def test_complete_prefixed_options(self):
        opts = {'A':'15 July 2026', 'B':'31 July 2026', 'C':'28 June 2026', 'D':'14 August 2026'}
        for letter, content in opts.items():
            self.assertEqual(parse_answer(f'{letter}. {content}', opts)[0], letter)
        self.assertIsNone(parse_answer('A. 31 July 2026', opts)[0])
        self.assertIsNone(parse_answer('31 July', opts)[0])
        self.assertIsNone(parse_answer(r'\boxed{A} or \boxed{B}', opts)[0])
        self.assertEqual(parse_answer(r'Brief reasoning. \boxed{B}', opts)[0], 'B')


if __name__ == '__main__':
    unittest.main()
