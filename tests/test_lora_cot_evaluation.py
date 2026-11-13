"""Contracts for full-data CoT prompts and scoring across both datasets."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import run_lora_cot_evaluation as cot
import audit_lora_cot_evaluation as auditor


class CotEvaluationContract(unittest.TestCase):
    def test_only_terminal_unambiguous_final_is_an_answer(self):
        for response, expected in [
            ('Consider A and C. Therefore B.\nFINAL: B', 'B'),
            ('FINAL: A\nFINAL: B', None),
            ('FINAL: A\nFINAL: A', None),
            ('FINAL: A\nMore text', None),
            ('I choose B.', None),
            ('B', None),
            ('FINAL: b', None),
            ('<think>unfinished\nFINAL: A', None),
            ('<think>Consider FINAL: A</think>\nFINAL: B', 'B'),
            ('Explanation\nFINAL：D', 'D'),
        ]:
            with self.subTest(response=response):
                self.assertEqual(cot.final_letter(response), expected)
                self.assertEqual(auditor.independent_answer(response), expected)

    def test_both_datasets_keep_questions_and_drop_gold(self):
        for relative in ('data/rules_repaired_single_image_v7', 'data/entity1020/merged_1109'):
            root = ROOT/relative
            for direction in ('T2T','T2V','V2T','V2V'):
                original = cot.rows(root/direction/'eval.jsonl')[0]
                source = {**original, 'answer':'GOLD_SHOULD_NOT_ENTER_PROMPT', 'audit':{'gold':'A'}}
                prepared = cot.cot_input(source, root, direction)
                self.assertTrue(set(prepared) <= {'id','qid','messages','images'})
                self.assertEqual([m['role'] for m in prepared['messages']], ['system','user'])
                self.assertEqual(prepared['messages'][0]['content'], cot.SYSTEM)
                self.assertTrue(prepared['messages'][-1]['content'].endswith(cot.SUFFIX))
                self.assertNotIn('GOLD_SHOULD_NOT_ENTER_PROMPT',json.dumps(prepared))
                self.assertNotIn('Do not include an explanation',json.dumps(prepared))
                self.assertEqual(len(prepared.get('images',[])),int(direction.endswith('V')))
                expected = original['messages'][-1]['content'][:-len(cot.ev.SUFFIX)]
                self.assertTrue(prepared['messages'][-1]['content'].startswith(expected))

    def test_invalid_answers_remain_in_accuracy_denominator(self):
        scored = [dict(correct=True,predicted='A',finish_reason='stop',has_explanation=True),
                  dict(correct=False,predicted='B',finish_reason='stop',has_explanation=True),
                  dict(correct=False,predicted=None,finish_reason='length',has_explanation=True)]
        metric = cot.metrics(scored)
        self.assertEqual(metric['total'],3)
        self.assertEqual(metric['accuracy'],1/3)
        self.assertEqual(metric['accuracy_on_valid_only'],1/2)
        self.assertEqual(metric['invalid'],1)
        self.assertEqual(metric['truncated'],1)


if __name__=='__main__':
    unittest.main()
