import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from answer_readout import parse_answer, options_from_question, finalizer_payload, choice_hook


class ReadoutContract(unittest.TestCase):
    def test_real_answer_formats_and_trailing_option_text(self):
        cases=[(r'Work. \(\boxed{b}\).','B'),('The final answer is B.','B'),
               ('Therefore, option B is accurate.','B'),('This makes option A correct.','A'),
               ('Therefore, the accurate statement is B.','B'),
               ('The correct answer is A. Belgian Super Cup.','A'),
               ('This corresponds to option C.','C'),('Final answer: b','B'),
               ('Final selection: C and D. FINAL: B','B'),
               ('Final selection: B, C, D. FINAL: A','A'),
               ('The answer is D. FINAL: C','C'),
               ('The final result is C > B > D > A. This matches option A. FINAL: A','A'),
               ('The selection is a separate modern event. The answer is B. FINAL: B','B'),
               ('The correct option is D.','D'),
               ('Therefore the correct output is D. Inspection recording convention applies.','D'),
               ('The answer is C because it matches the recalled rule.','C'),
               ('最终答案是 D。','D'),('{"answer":"A"}','A'),
               ('```json\n{"option":"D"}\n```','D'),('B.','B')]
        for response,expected in cases:
            with self.subTest(response=response):self.assertEqual(parse_answer(response)[0],expected)

    def test_ambiguity_negation_candidates_and_conditional_choices(self):
        cases=['Option A is not correct.','If option B is correct, return it.',
               'The answer is A or B.','The answer is A. The answer is B.',
               r'\boxed{A} \boxed{B}','The final selection is A=KEEP; B=OMIT.',
               'Final selection: C and D.','The correct answer is A?',
               'Option B is correct if the count is three.',
               '<think>unfinished FINAL: B','Use the reference image to choose an option.',
               'The answer is A=KEEP; B=OMIT.','{"answer":""}']
        for response in cases:
            with self.subTest(response=response):self.assertIsNone(parse_answer(response)[0])

    def test_exact_option_content_needs_unique_match(self):
        opts=options_from_question('Question\nA. once\nB. twice\nC. three times\nD. four times\n\nReason step by step.')
        self.assertEqual(opts,{'A':'once','B':'twice','C':'three times','D':'four times'})
        self.assertEqual(parse_answer('twice.',opts),('B','exact_option_content'))
        self.assertIsNone(parse_answer('He won twice but maybe three times.',opts)[0])
        self.assertIsNone(parse_answer('twice',{'A':'twice','B':'twice'})[0])

    def test_finalizer_whitelists_inputs_and_keeps_test_image_and_own_context(self):
        payload=finalizer_payload({'messages':[{'role':'user','content':'<image>QUESTION SELF_RECALL'}],
                                   'images':['test.png'],'answer':'GOLD_SENTINEL','reference':'ORACLE_SENTINEL'},'OWN_RESPONSE')
        text=str(payload)
        for item in ['QUESTION','SELF_RECALL','OWN_RESPONSE','test.png']:self.assertIn(item,text)
        for item in ['GOLD_SENTINEL','ORACLE_SENTINEL']:self.assertNotIn(item,text)
        self.assertEqual(set(payload),{'messages','images'})

    def test_choice_constraint_has_exactly_four_allowed_tokens(self):
        class Tokenizer:
            def encode(self,letter,add_special_tokens=False):return [ord(letter)]
        hook=choice_hook(Tokenizer());kwargs=hook({'inputs':{}})
        allowed=kwargs['inputs']['prefix_allowed_tokens_fn'](0,[])
        self.assertEqual(allowed,[65,66,67,68])

    def test_finalizer_drops_conflicting_old_output_instruction(self):
        question='QUESTION\nA. one\nB. two\nC. three\nD. four'
        suffix='\n\nReason step by step in a concise explanation (at most 200 words). End with exactly FINAL: followed by one uppercase letter A, B, C, or D.'
        payload=finalizer_payload({'messages':[{'role':'user','content':question+suffix}]},'OWN_RESPONSE')
        self.assertIn(question,payload['messages'][-1]['content'])
        self.assertNotIn('End with exactly FINAL:',str(payload))
        self.assertNotIn('Reason step by step',str(payload))


if __name__=='__main__':unittest.main()
