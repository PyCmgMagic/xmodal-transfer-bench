import unittest
from answer_readout_entity_v4 import parse_answer


class ReadoutChecks(unittest.TestCase):
    def test_nested_letter_is_explicit(self):
        for letter in 'ABCD':
            for command in ['text','textrm','mathrm','mathbf','mathsf']:
                response='Final option: $\\boxed{\\'+command+'{'+letter+'}}$'
                self.assertEqual(parse_answer(response)[0],letter)

    def test_conflicts_and_uncertain_text_remain_invalid(self):
        for response in [r'\boxed{\text{A}} or \boxed{B}',
                         r'\boxed{\text{A or B}}',r'\boxed{\text{Option A}}',
                         r'If the answer is \boxed{\text{C}}',
                         r'<think>Maybe \boxed{\text{C}}']:
            self.assertIsNone(parse_answer(response)[0],response)


if __name__=='__main__':unittest.main()
