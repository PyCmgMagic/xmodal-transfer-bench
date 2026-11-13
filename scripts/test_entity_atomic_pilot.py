import unittest
from prepare_entity_atomic_pilot import slots,entity_examples,generic
from answer_readout_entity_v3 import parse_answer


class AtomicChecks(unittest.TestCase):
    def test_date_and_number_fields_receive_separate_targets(self):
        r=dict(id='i',qid='q',name='Example Person',
               claim='On 31 July 2026, Example Person joined Example Club for $15 million.')
        q=dict(question_text='Which club did Example Person join on 31 July 2026?',answer='Example Club')
        ss=slots(r['claim'])
        self.assertEqual([s[2] for s in ss],['31 July 2026','$15 million'])
        samples=entity_examples(r,q)
        self.assertIn('31 July 2026',[s['messages'][1]['content'] for s in samples])
        self.assertIn('$15 million',[s['messages'][1]['content'] for s in samples])
        self.assertNotIn('31 July 2026',samples[0]['messages'][0]['content'])

    def test_generic_options_and_labels_are_balanced_and_disjoint(self):
        train=[generic(i,'train') for i in range(96)]
        dev=[generic(i,'dev') for i in range(96)]
        for letter in 'ABCD':
            self.assertEqual(sum(r['answer']==letter for r in train),24)
        self.assertFalse(set(r['messages'][0]['content'] for r in train)&set(r['messages'][0]['content'] for r in dev))
        for row in train+dev:
            self.assertTrue(row['messages'][1]['content'].endswith('\\boxed{'+row['answer']+'}'))

    def test_explicit_terminal_answer_and_ambiguity(self):
        self.assertEqual(parse_answer('Some facts. This gives the correct answer as option C.')[0],'C')
        self.assertIsNone(parse_answer('If this gives the correct answer as option A.')[0])
        self.assertIsNone(parse_answer('Not the answer as option A.')[0])
        self.assertIsNone(parse_answer('Option A or option B. This gives the correct answer as option B.')[0])
        self.assertIsNone(parse_answer('He is a Serie A player.')[0])


if __name__=='__main__':unittest.main()
